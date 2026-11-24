"""A box that is a container the provider runs for us (Vast, RunPod): our start-up command
is its entrypoint. It starts the lease timer, saves the container's environment, puts
the self-destroy helper in place and runs an sshd that takes only the run's key.

What differs between such providers is how a box deletes itself: each gives the
container a key confined to it, and `bootstrap_script` takes the command that uses it.
The reading of docker's error messages (`classify_status`) is here too: Vast shows
them as a box's status, and a VM box's probe reports its pull's (vmbox.py).
"""

from __future__ import annotations

import re
import shlex

from ..onbox import (
    DESTROY_SELF,
    DESTROY_SELF_SCRIPT,
    ENV_FILE,
    EXPIRY_FILE,
    SAVE_ENV,
    check_expiry,
)

# Docker's error messages, as a box's status shows them. An image problem is ours and identical on every host;
# a runtime problem belongs to the machine.
IMAGE_ERRORS = (
    "manifest unknown",
    "pull access denied",
    "repository does not exist",
    "authentication required",
)
MACHINE_ERRORS = (
    "oci runtime create failed",
    "failed to create shim",
    "nvidia-container-cli",
    "no space left on device",
)
_CDI = re.compile(r"\bCDI\b")


# The run's key goes into a file of our own, and sshd is told to read only that.
# /root/.ssh/authorized_keys is not ours to rely on: on a real box Vast was seen to
# replace it with the account's registered key as the container started, which locked
# the launcher out of an image whose sshd was up within the first second.
KEYS_FILE = "/etc/ssh/gpubox_authorized_keys"


def bootstrap_script(pubkey: str, expiry: int, destroy: str) -> str:
    """The container's command, as one line of bash: a self-destruct timer, then the
    container's environment saved, the self-destroy helper installed, and sshd for our key.

    `destroy` is the shell command with which the box deletes itself, with a key the
    provider gives that one container (vast.DESTROY, runpod.DESTROY).

    The timer is a dead-man switch that needs neither the launcher nor our API
    key. It has to destroy the instance: merely exiting does not stop the
    billing, because Vast restarts a container whose main process ends. It is
    part of the create request, so even a box whose create response we never
    saw carries it. One line, because Vast passes this as `docker run` arguments.

    It is started first, so a box stuck installing sshd is covered. It waits
    for an absolute time, so a container that restarts does not start counting
    again. That time is read from a file on the box, which `renew` rewrites;
    the file is created with the expiry of the create request unless it is
    already there (a restart must not undo a renewal), and anything in it
    that is not a number of at most ten digits falls back to that first expiry. Once the time has
    passed, the delete is sent every 30 s for as long as the timer runs,
    whatever the answers say: if it works, the loop ends with the box.

    It is the container's entrypoint (`create_payload`), so an image's own
    ENTRYPOINT does not run first; a job that wants it runs it itself.
    """
    check_expiry(expiry)
    watchdog = (
        f"[ -f {EXPIRY_FILE} ] || echo {expiry} > {EXPIRY_FILE}; "
        f"(while :; do e=$(cat {EXPIRY_FILE} 2>/dev/null); "
        # Empty, not a number, or longer than a time: bash could not compare it.
        f'case "$e" in ""|*[!0-9]*|???????????*) e={expiry};; esac; '
        f'if [ "$(date +%s)" -ge "$e" ]; then'
        f" {destroy}; fi; sleep 30; done) >/dev/null 2>&1 &"
    )
    script = (
        watchdog
        + " "
        + "; ".join(
            [
                "set -eu",
                "if ! command -v sshd >/dev/null 2>&1; then apt-get update -qq"
                " && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq"
                " --no-install-recommends openssh-server; fi",
                "mkdir -p /run/sshd /etc/ssh /etc/gpubox /etc/profile.d /usr/local/bin",
                SAVE_ENV,
                # The variables of a job whose start the container stopped under, the only
                # ones still on disk: a started job deletes its own at once.
                'rm -f "${GPUBOX_JOBS:-/var/lib/gpubox/jobs}"/*/env',
                f"echo {shlex.quote(f'[ -r {ENV_FILE} ] && . {ENV_FILE}')}"
                " > /etc/profile.d/gpubox-env.sh",
                "printf '%s\\n' "
                + " ".join(shlex.quote(line) for line in DESTROY_SELF_SCRIPT)
                + f" > {DESTROY_SELF}",
                f"chmod 755 {DESTROY_SELF}",
                f"printf '%s\\n' {shlex.quote(pubkey)} > {KEYS_FILE}",
                f"chmod 600 {KEYS_FILE}",
                "ssh-keygen -A >/dev/null",
                "exec /usr/sbin/sshd -D -e"
                " -o PermitRootLogin=prohibit-password -o PasswordAuthentication=no"
                f" -o AuthorizedKeysFile={KEYS_FILE}",
            ]
        )
    )
    return script


def bootstrap_command(pubkey: str, expiry: int, destroy: str) -> str:
    """The bootstrap as one shell command line, for a person to read or run."""
    return "bash -c " + shlex.quote(bootstrap_script(pubkey, expiry, destroy))


def classify_status(msg: str) -> str | None:
    """'image', 'machine', or None for a status message that decides nothing."""
    low = msg.lower()
    if any(s in low for s in IMAGE_ERRORS):
        return "image"
    if any(s in low for s in MACHINE_ERRORS) or _CDI.search(msg):
        return "machine"
    return None
