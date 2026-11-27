"""What gpubox keeps on a box, whatever the provider: the end of its lease, the container's
saved environment, the command of a container that only waits, and the helper with which
a job ends its box. Shell snippets and paths shared by the boxes (providers/container.py,
providers/vmbox.py, providers/lium.py), the jobs (job.py) and the gate (gate.py)."""

from __future__ import annotations

import shlex

EXPIRY_FILE = "/etc/gpubox-expiry"


def check_expiry(expiry) -> None:
    if type(expiry) is not int or not 0 < expiry < 10**10:
        raise ValueError(f"bad expiry {expiry!r}")


def set_expiry_command(expiry: int) -> str:
    """Shell for the box: move the time at which its timer destroys it (see `renew`)."""
    check_expiry(expiry)
    return f"echo {expiry} > {EXPIRY_FILE}.tmp && mv {EXPIRY_FILE}.tmp {EXPIRY_FILE}"


# The container's environment (Docker's ENV, Vast's variables), saved for what runs
# over SSH, which does not inherit it: login shells load it, and so does every
# command gpubox runs on the box (job.LOAD_ENV). POSIX syntax, so sh can read it.
ENV_FILE = "/etc/gpubox/env.sh"
# Written afresh and renamed into place, so it is 0600 whatever the image had there.
# Each value in single quotes (a quote in it as '\''), which every POSIX shell reads:
# not bash's `export -p`, whose $'...' for a value with a newline (bash 5.2) dash cannot
# read. Run by bash, for `compgen -e` (the exported names) and `${!n}`.
SAVE_ENV = (
    "(unset HOME PWD OLDPWD SHLVL _ HOSTNAME TERM SHELL USER LOGNAME MAIL; umask 077; "
    f"rm -f {ENV_FILE}.tmp; q=\"'\\''\"; "
    f'for n in $(compgen -e); do v=${{!n}}; printf "export %s=\'%s\'\\n" "$n" "${{v//\\\'/$q}}"; '
    f"done > {ENV_FILE}.tmp && mv {ENV_FILE}.tmp {ENV_FILE})"
)
# Loads it into a shell on the box. POSIX sh: root's login shell, which runs what ssh
# is asked to run, need not be bash.
LOAD_ENV = f"[ -r {ENV_FILE} ] && . {ENV_FILE}; "
# A job ends its own box with this: the lease timer, which reads the expiry file
# every 30 s, then destroys it. One way to destroy a box from inside, not two.
DESTROY_SELF = "/usr/local/bin/gpubox-destroy-self"
DESTROY_SELF_SCRIPT = (
    "#!/bin/sh",
    "# Ends this box: its lease timer destroys it within 30 s.",
    f"echo 1 > {EXPIRY_FILE}.tmp && mv {EXPIRY_FILE}.tmp {EXPIRY_FILE} &&",
    'echo "gpubox: this box will be destroyed within 30 s" >&2',
)
# The command of a box's container where gpubox starts no sshd (a VM box, vmbox.py; Lium's
# pods, whose sshd is Lium's): its environment saved for what runs over `docker exec` or
# SSH (which gets the image's ENV, but not what the entrypoint would have set), the
# variables of a job cut short by a restart removed, then nothing. Run by bash (SAVE_ENV
# needs it, and so do jobs); no `sleep infinity`, which busybox lacks.
CONTAINER_INIT = "; ".join(
    [
        "mkdir -p /etc/gpubox /etc/profile.d",
        SAVE_ENV,
        'rm -f "${GPUBOX_JOBS:-/var/lib/gpubox/jobs}"/*/env',
        f"echo {shlex.quote(f'[ -r {ENV_FILE} ] && . {ENV_FILE}')} > /etc/profile.d/gpubox-env.sh",
        "while :; do sleep 3600; done",
    ]
)
