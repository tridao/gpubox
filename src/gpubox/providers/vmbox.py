"""A box on a VM cloud: the image, run as a container on a rented VM.

The VM clouds (Shadeform and the clouds behind it; later Hyperstack, Lyceum, Radiant,
Massed Compute directly) rent virtual machines with an OS of theirs, not containers. A
box there is still the caller's image: the VM's startup script, which the cloud runs as
root once the VM is up (`startup_script`), puts the run's key on the VM and starts the
image as a container named `gpubox`, with the VM's GPUs, network and IPC, under a
command of ours (`onbox.CONTAINER_INIT`) that saves its environment and then just waits.

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

from .. import onbox
from ..gate import READY_TIMEOUT, Prober, read_ready

CONTAINER = "gpubox"
PULL_ERROR = "/var/lib/gpubox/pull-error"
BOOT_LOG = "/var/lib/gpubox/boot.log"  # what the startup script said; the probe shows its end
PLACEHOLDER = "gpubox-placeholder"


def placeholder_key() -> str:
    """A fresh public key whose private half is deleted at once: for a cloud that wants a
    key at the create (Hyperstack, Verda), while the run's key goes in with the script.
    An earlier run's key would let that run's holder into later VMs."""
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        key = Path(d) / "k"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", PLACEHOLDER,
                        "-f", str(key)], check=True)  # fmt: skip
        return key.with_suffix(".pub").read_text().strip()


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
        f"--ipc host --user 0:0 --entrypoint bash {img} -c {shlex.quote(onbox.CONTAINER_INIT)}"
    )
    return f"""#!/bin/bash
# gpubox: this VM is a box. The run's key, then the image as the container `{CONTAINER}`.
mkdir -p "$(dirname {BOOT_LOG})" && exec >>{BOOT_LOG} 2>&1
echo "gpubox: setting up the box"
for h in /root /home/*; do
    [ -d "$h" ] || continue
    u=$(stat -c %U "$h")
    mkdir -p "$h/.ssh" && printf '%s\\n' {key} >> "$h/.ssh/authorized_keys"
    chown -R "$u": "$h/.ssh"; chmod 700 "$h/.ssh"; chmod 600 "$h/.ssh/authorized_keys"
done
mkdir -p "$(dirname {PULL_ERROR})"; rm -f {PULL_ERROR}
if ! command -v docker >/dev/null && command -v apt-get >/dev/null; then
    DEBIAN_FRONTEND=noninteractive apt-get update -qq &&
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io && systemctl start docker
fi
for i in $(seq 1 150); do docker info >/dev/null 2>&1 && break; sleep 2; done
# An image that has docker but not NVIDIA's container toolkit gets it, from NVIDIA's apt
# repository (Ubuntu), so that `--gpus all` works.
if ! docker info 2>/dev/null | grep -qi nvidia && command -v apt-get >/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
        gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg &&
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list &&
    apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit &&
    nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
    for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
fi
pulled=
for i in 1 2 3 4 5; do
    if out=$(docker pull {img} 2>&1); then pulled=1; break; fi
    sleep 10
done
if [ -z "$pulled" ]; then  # the verdict, once it is final: the probe reports it
    printf '%s\\n' "$out" | tail -n 3 > {PULL_ERROR}.tmp && mv {PULL_ERROR}.tmp {PULL_ERROR}
    exit 1
fi
docker rm -f {CONTAINER} >/dev/null 2>&1
{run} && echo "gpubox: the box is up"
"""


# Root's commands on the VM: as the login user through sudo, or as root without it.
AS_ROOT = 'S=; [ "$(id -u)" = 0 ] || S="sudo -n"; '
STARTED = "/var/lib/gpubox/boot-started"


class VmProber(Prober):
    """SSH to a VM box with the run's key, as the user its cloud names, and into the box's
    container with `docker exec`. An endpoint is (host, port, user).

    `boot`: the startup script, for a cloud that cannot run one at boot (Lyceum). The
    first probe that gets in sends it and starts it in a session of its own, once."""

    boot: str | None = None

    def _argv(self, ep, inner: str, **kw) -> list[str]:
        host, port, user = ep
        argv = super().ssh((host, port), inner, **kw)
        return [f"{user}@{host}" if a == f"root@{host}" else a for a in argv]

    def ssh(self, ep, remote, *, connect_timeout: int = 10, tty=None) -> list[str]:
        if remote is None:
            inner = (
                f"{AS_ROOT}$S docker exec -u 0 -it {CONTAINER} bash -l || "
                f"$S docker exec -u 0 -it {CONTAINER} sh -l"
            )
        else:
            flags = "-it" if tty else "-i"
            inner = f"{AS_ROOT}$S docker exec -u 0 {flags} {CONTAINER} sh -c {shlex.quote(remote)}"
        return self._argv(ep, inner, connect_timeout=connect_timeout, tty=tty)

    def start_ready(self, ep):
        """Probe the container, not just the VM; if it is not up, say why when the pull
        said. With `boot`, start the box's script if no probe has yet."""
        check = (
            f"{AS_ROOT}$S docker exec -u 0 {CONTAINER} true 2>/dev/null && exit 0; "
            f"$S test -e {PULL_ERROR} && {{ $S cat {PULL_ERROR} >&2; exit 1; }}; "
        )
        if self.boot is not None:
            check += (
                # The script is read in the foreground: a job sent to the background
                # reads its stdin from /dev/null. Only its run goes there.
                f"if ! $S test -e {STARTED}; then $S mkdir -p $(dirname {STARTED}) && "
                f'f=$(mktemp) && cat > "$f" && $S touch {STARTED} && '
                f'{{ $S setsid nohup bash "$f" </dev/null >/dev/null 2>&1 & }}; '
                "echo 'the box is being set up' >&2; exit 1; fi; "
            )
        check += (
            f"$S tail -n 2 {BOOT_LOG} >&2 2>/dev/null; echo 'the box is not up yet' >&2; exit 1"
        )
        argv = self._argv(ep, check, connect_timeout=8)
        return self._start(argv, read_ready, stdin=self.boot, timeout=READY_TIMEOUT)
