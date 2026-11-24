"""A box on a VM cloud: the image, run as a container on a rented VM.

The VM clouds (Shadeform and the clouds behind it; later Hyperstack, Lyceum, Radiant,
Massed Compute directly) rent virtual machines with an OS of theirs, not containers. A
box there is still the caller's image: the VM's startup script, which the cloud runs as
root once the VM is up (`startup_script`), puts the run's key on the VM and starts the
image as a container named `gpubox`, with the VM's GPUs, network and IPC, under a
command of ours (`CONTAINER_INIT`) that saves its environment and then just waits.

Everything gpubox runs on such a box runs in that container: SSH to the VM as the user
the cloud names, with the run's key, then `sudo docker exec` (`VmProber`). No port is
opened and the image needs no sshd. Jobs work as on Vast: started with `setsid` inside
the container, they outlive the `docker exec` that started them.

While the container is not up (the pull is under way), the probe fails, and says why
if the pull failed (`PULL_ERROR`): the launcher reads an image that cannot be pulled
from that, as from a Vast host's status message.

The lease is the cloud's: a VM cannot delete itself without an account-wide key, which
never goes on a box. So the provider gives each VM an expiry of its own where it can
(Shadeform's auto_delete), and otherwise `down` and a scheduled `reap` end it.
"""

from __future__ import annotations

import shlex

from . import vast as v
from .gate import READY_TIMEOUT, Prober, read_ready

CONTAINER = "gpubox"
PULL_ERROR = "/var/lib/gpubox/pull-error"
# The container's command: its environment saved for what `docker exec` runs (which gets
# the image's ENV anyway, but not what the entrypoint would have set), the variables of a
# job cut short by a restart removed, then nothing. Run by bash (SAVE_ENV needs it, and so
# do jobs, as on Vast); no `sleep infinity`, which busybox lacks.
CONTAINER_INIT = "; ".join(
    [
        "mkdir -p /etc/gpubox /etc/profile.d",
        v.SAVE_ENV,
        'rm -f "${GPUBOX_JOBS:-/var/lib/gpubox/jobs}"/*/env',
        f"echo {shlex.quote(f'[ -r {v.ENV_FILE} ] && . {v.ENV_FILE}')} > /etc/profile.d/gpubox-env.sh",
        "while :; do sleep 3600; done",
    ]
)


def startup_script(pubkey: str, image: str) -> str:
    """The VM's startup script (bash, run as root once the VM is up): the run's key for
    every login user, then the image as the `gpubox` container. A pull that fails five
    times leaves its last words in PULL_ERROR for the probe to report."""
    if "\n" in pubkey or not pubkey.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-")):
        raise ValueError("not a public key")
    if not image or any(c.isspace() for c in image) or image.startswith("-"):
        raise ValueError(f"not an image reference: {image!r}")
    key, img = shlex.quote(pubkey), shlex.quote(image)
    run = (
        f"docker run -d --name {CONTAINER} --restart unless-stopped --gpus all --network host "
        f"--ipc host --entrypoint bash {img} -c {shlex.quote(CONTAINER_INIT)}"
    )
    return f"""#!/bin/bash
# gpubox: this VM is a box. The run's key, then the image as the container `{CONTAINER}`.
for h in /root /home/*; do
    [ -d "$h" ] || continue
    u=$(stat -c %U "$h")
    mkdir -p "$h/.ssh" && printf '%s\\n' {key} >> "$h/.ssh/authorized_keys"
    chown -R "$u": "$h/.ssh"; chmod 700 "$h/.ssh"; chmod 600 "$h/.ssh/authorized_keys"
done
mkdir -p "$(dirname {PULL_ERROR})"; rm -f {PULL_ERROR}
for i in $(seq 1 150); do docker info >/dev/null 2>&1 && break; sleep 2; done
for i in 1 2 3 4 5; do
    if out=$(docker pull {img} 2>&1); then rm -f {PULL_ERROR}; break; fi
    printf '%s\\n' "$out" | tail -n 3 > {PULL_ERROR}
    sleep 10
done
docker rm -f {CONTAINER} >/dev/null 2>&1
{run} || docker logs {CONTAINER} 2>&1 | tail -n 3 >> {PULL_ERROR}
"""


class VmProber(Prober):
    """SSH to a VM box with the run's key, as the user its cloud names, and into the box's
    container with `docker exec`. An endpoint is (host, port, user)."""

    def ssh(self, ep, remote, *, connect_timeout: int = 10, tty=None) -> list[str]:
        host, port, user = ep
        if remote is None:
            inner = f"sudo -n docker exec -it {CONTAINER} bash -l || sudo -n docker exec -it {CONTAINER} sh -l"
        else:
            flags = "-it" if tty else "-i"
            inner = f"sudo -n docker exec {flags} {CONTAINER} sh -c {shlex.quote(remote)}"
        argv = super().ssh((host, port), inner, connect_timeout=connect_timeout, tty=tty)
        return [f"{user}@{host}" if a == f"root@{host}" else a for a in argv]

    def start_ready(self, ep):
        """Probe the container, not just the VM; if it is not up, say why when the pull
        said."""
        host, port, user = ep
        check = (
            f"sudo -n docker exec {CONTAINER} true 2>/dev/null || "
            f"{{ sudo -n cat {PULL_ERROR} >&2 2>/dev/null; echo 'the box is not up yet' >&2; exit 1; }}"
        )
        argv = Prober.ssh(self, (host, port), check, connect_timeout=8)
        argv = [f"{user}@{host}" if a == f"root@{host}" else a for a in argv]
        return self._start(argv, read_ready, timeout=READY_TIMEOUT)
