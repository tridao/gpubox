"""What gpubox keeps on a box, whatever the provider: the end of its lease, the container's
saved environment, and the helper with which a job ends its box. Shell snippets and
paths shared by the boxes (providers/container.py, providers/vmbox.py), the jobs
(job.py) and the gate (gate.py)."""

from __future__ import annotations

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
SAVE_ENV = (
    "(unset HOME PWD OLDPWD SHLVL _ HOSTNAME TERM SHELL USER LOGNAME MAIL; set -o posix; "
    f"umask 077; rm -f {ENV_FILE}.tmp; export -p > {ENV_FILE}.tmp && mv {ENV_FILE}.tmp {ENV_FILE})"
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
