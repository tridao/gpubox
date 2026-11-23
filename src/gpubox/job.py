"""Jobs on a box: a script started detached, its log followed from wherever we are.

A job lives in a directory on the box, `/var/lib/gpubox/jobs/<id>/` (or under
$GPUBOX_JOBS, if the box sets it):

    script   what to run
    env      the caller's variables (secrets among them), 0600; deleted as the job starts
    log      the script's stdout and stderr
    wpid     "<pid> <start time>" of the wrapper that runs the script and waits for it
    pid      the script's process group
    exit     its exit code, written (atomically) once it has ended
    expired  present if the box killed it at its timeout

The wrapper runs in a session of its own, so nothing that happens to an SSH
connection reaches it: a dropped connection, a killed `gpubox run`, a laptop
that went to sleep. The timeout is kept on the box for the same reason. What
reads the log is a small program on the box (FOLLOW) that prints the log from
a byte offset and ends with one status line on stderr; the controller counts
the bytes it got, so after a dropped connection it carries on where it was.

The job sees the container's environment (Docker's ENV, Vast's variables,
saved by the bootstrap), then the caller's variables, then GPUBOX_RUN_ID,
GPUBOX_JOB, GPUBOX_BOX_INDEX and GPUBOX_BOX_COUNT.
"""

from __future__ import annotations

import os
import re
import secrets
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass

from . import vast as v

JOBS_DIR = "/var/lib/gpubox/jobs"
# The jobs directory, as the shell on the box finds it.
JOBS = '"${GPUBOX_JOBS:-%s}"' % JOBS_DIR
# Loads the container's environment into a shell on the box. POSIX sh: root's
# login shell, which runs what ssh is asked to run, need not be bash.
LOAD_ENV = f"[ -r {v.ENV_FILE} ] && . {v.ENV_FILE}; "
JOB_ID = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{4}")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
MARK = "GPUBOX-JOB"  # starts the follower's status line
KILL_GRACE = 10  # seconds between SIGTERM and SIGKILL when a job's time is up or it is killed


class JobError(RuntimeError):
    """A job cannot be started or found as asked."""


def new_job_id(now: float | None = None) -> str:
    """Sortable by start time, so the newest job is the last name in the directory."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(time.time() if now is None else now))
    return f"{stamp}-{secrets.token_hex(2)}"


def check_job_id(job_id) -> str:
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
        raise JobError(f"not a job id: {job_id!r}")
    return job_id


def env_lines(env: dict[str, str]) -> str:
    """Variables as lines a POSIX shell can source."""
    out = []
    for name, value in env.items():
        if not ENV_NAME.fullmatch(name):
            raise JobError(f"not a variable name: {name!r}")
        if "\0" in value:
            raise JobError(f"the value of {name} contains a NUL byte")
        out.append(f"export {name}={shlex.quote(value)}\n")
    return "".join(out)


def _heredoc(path: str, text: str) -> str:
    """Shell that writes `text` to `path` (a shell word) exactly: a here-document whose
    end marker cannot occur in it. The text goes through `printf %s` so a missing
    final newline is kept missing."""
    while True:
        tag = f"GPUBOX_{secrets.token_hex(8).upper()}"
        if tag not in text:
            break
    # A here-document always ends in a newline; strip the one we add.
    return (
        f"{{ IFS= read -r -d '' x <<'{tag}' || true\n{text}\n{tag}\n"
        f'printf %s "${{x%?}}" > {path}; }}\n'
    )


# The wrapper: $1 is the job directory, $2 the timeout in seconds (0: none), $3 is 1 to
# destroy the box when the script ends. Run by bash, in a session of its own.
_WRAPPER = r"""
d=$1; limit=$2; destroy=$3
echo "$$ $(cut -d' ' -f22 /proc/$$/stat 2>/dev/null)" > "$d/wpid"
[ -r %(env_file)s ] && . %(env_file)s
set -a; . "$d/env"; set +a
rm -f "$d/env"
cd "${GPUBOX_WORKDIR:-$HOME}" 2>/dev/null || cd /
setsid bash "$d/script" </dev/null >>"$d/log" 2>&1 &
child=$!
echo "$child" > "$d/pid"
watcher=
if [ "$limit" -gt 0 ]; then
    (
        sleep "$limit" & s=$!
        trap 'kill $s 2>/dev/null; exit 0' TERM
        wait $s
        if kill -0 -- -"$child" 2>/dev/null; then
            touch "$d/expired"
            kill -TERM -- -"$child" 2>/dev/null
            sleep %(grace)d
            kill -KILL -- -"$child" 2>/dev/null
        fi
    ) &
    watcher=$!
fi
wait "$child"; rc=$?
[ -n "$watcher" ] && kill "$watcher" 2>/dev/null
[ -e "$d/expired" ] && rc=124
echo "$rc" > "$d/exit.tmp" && mv "$d/exit.tmp" "$d/exit"
if [ "$destroy" = 1 ]; then %(destroy_self)s; fi
"""


def wrapper() -> str:
    return _WRAPPER % {
        "grace": KILL_GRACE, "env_file": v.ENV_FILE, "destroy_self": v.DESTROY_SELF,
    }  # fmt: skip


def start_command(
    job_id: str,
    script: str,
    env: dict[str, str],
    *,
    timeout_s: int = 0,
    destroy_after: bool = False,
) -> tuple[str, str]:
    """(remote command, its stdin) that starts a job and returns at once. The script and
    the variables travel on stdin, never on a command line. The command prints the job id
    once the job is started."""
    check_job_id(job_id)
    if type(timeout_s) is not int or timeout_s < 0:
        raise JobError(f"bad timeout {timeout_s!r}")
    program = (
        "set -eu\n"
        "umask 077\n"
        f'J={JOBS}; d="$J/{job_id}"\n'
        'mkdir -p "$J" && mkdir "$d"\n'
        + _heredoc('"$d/script"', script)
        + _heredoc('"$d/env"', env_lines(env))
        + _heredoc('"$d/wrapper"', wrapper())
        + ': > "$d/log"\n'
        + f'setsid bash "$d/wrapper" "$d" {timeout_s} {int(destroy_after)}'
        " </dev/null >/dev/null 2>&1 &\n"
        f"echo {job_id}\n"
    )
    return "bash -s", program


# The follower: argv is the job directory, the byte offset to start from, and 1 to wait
# for the job to end (0: print what there is and stop). Ends with one line on stderr:
# "GPUBOX-JOB exit N", "GPUBOX-JOB running", "GPUBOX-JOB lost" or "GPUBOX-JOB none".
FOLLOW = r"""
import os, sys, time
d, off, wait = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
out = sys.stdout.buffer
def end(what):
    out.flush()
    sys.stderr.write("\nGPUBOX-JOB %s\n" % what)
    sys.exit(0)
def chunk():
    global off
    try:
        with open(os.path.join(d, "log"), "rb") as f:
            f.seek(off)
            data = f.read(1 << 16)
    except FileNotFoundError:
        return False
    if data:
        out.write(data)
        out.flush()
        off += len(data)
    return bool(data)
def exit_code():
    try:
        with open(os.path.join(d, "exit")) as f:
            return f.read().strip()
    except FileNotFoundError:
        return None
def wrapper_alive():
    # The wrapper, recognised by its pid and its start time, so that a process that got
    # its pid after a container restart is not taken for it. (Without /proc, by its pid.)
    try:
        with open(os.path.join(d, "wpid")) as f:
            pid, start = (f.read().split() + [""])[:2]
        pid = int(pid)
        if not start:
            os.kill(pid, 0)
            return True
        with open("/proc/%d/stat" % pid) as f:
            return f.read().rsplit(")", 1)[1].split()[19] == start
    except PermissionError:
        return True
    except (OSError, ValueError, IndexError):
        return False
if not os.path.isdir(d):
    end("none")
started = time.monotonic()
gone = None
while True:
    if chunk():
        continue
    rc = exit_code()
    if rc is not None:
        while chunk():
            pass
        end("exit " + rc)
    if not wait:
        end("running")
    if wrapper_alive() or time.monotonic() - started < 5:
        gone = None
    elif gone is None:
        gone = time.monotonic()
    elif time.monotonic() - gone > 3:  # not just between the script's end and the exit file
        end("lost")
    time.sleep(0.5)
"""


def follow_command(job_id: str, offset: int, wait: bool) -> str:
    check_job_id(job_id)
    return (
        LOAD_ENV
        + f"exec python3 -c {shlex.quote(FOLLOW)} {JOBS}/{job_id} {int(offset)} {int(wait)}"
    )


def latest_command() -> str:
    """Prints the newest job's id, or nothing."""
    return f"ls -1 {JOBS} 2>/dev/null | grep -E '^{JOB_ID.pattern}$' | tail -n 1; true"


def list_command() -> str:
    """One line per job, oldest first: its id, then `exit N`, `running` or `lost`."""
    return (
        f'for d in {JOBS}/*/; do [ -d "$d" ] || continue; j=$(basename "$d"); '
        'if [ -e "$d/exit" ]; then echo "$j exit $(cat "$d/exit")"; '
        'elif [ -e "$d/wpid" ] && kill -0 "$(cut -d" " -f1 "$d/wpid")" 2>/dev/null; '
        'then echo "$j running"; else echo "$j lost"; fi; done'
    )


def kill_command(job_id: str) -> str:
    """SIGTERM to the job's process group, SIGKILL if it is still there after the grace."""
    check_job_id(job_id)
    return (
        f"p=$(cat {JOBS}/{job_id}/pid 2>/dev/null) || {{ echo 'no such job' >&2; exit 3; }}; "
        'kill -TERM -- -"$p" 2>/dev/null; sleep 1; '
        f'if kill -0 -- -"$p" 2>/dev/null; then sleep {KILL_GRACE}; '
        'kill -KILL -- -"$p" 2>/dev/null; fi; true'
    )


@dataclass(frozen=True)
class End:
    """How following a job ended.

    exit      the job ended with `rc`
    running   it is still running (we were not asked to wait)
    lost      it ended without an exit code: its box restarted, or it was killed
    none      there is no such job on the box
    gone      the box no longer exists
    unreachable  no connection for longer than we were prepared to wait
    """

    kind: str
    rc: int | None = None
    detail: str = ""


def parse_end(stderr: str) -> End | None:
    """The follower's status line, or None if it never got to say (a dropped connection)."""
    for line in reversed(stderr.splitlines()):
        if not line.startswith(MARK + " "):
            continue
        parts = line.split()
        if parts[1:2] == ["exit"] and len(parts) == 3 and re.fullmatch(r"-?[0-9]{1,3}", parts[2]):
            return End("exit", int(parts[2]))
        if len(parts) == 2 and parts[1] in ("running", "lost", "none"):
            return End(parts[1])
        return None
    return None


def follow(
    ssh_argv,
    job_id: str,
    out,
    *,
    wait: bool = True,
    offset: int = 0,
    still_there=None,
    patience: float = 600,
    retry: float = 5,
) -> tuple[End, int]:
    """Copy a job's log to `out` (a binary stream) from `offset` until the job ends (or,
    without `wait`, until what there is has been copied). Returns how it ended and the
    offset reached.

    A dropped connection is not the end of anything: the follower is started again
    from the offset reached. Between tries `still_there()`, if given, says whether the
    box still exists (False ends it as "gone"; None, unknown, keeps trying). After
    `patience` seconds without a connection that got anywhere, it gives up as
    "unreachable"; the job, if any, carries on regardless.
    """
    failing_since = None
    while True:
        before = offset
        with tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(
                ssh_argv(follow_command(job_id, offset, wait)),
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err,
            )  # fmt: skip
            try:
                while True:
                    data = proc.stdout.read1(1 << 16)
                    if not data:
                        break
                    out.write(data)
                    out.flush()
                    offset += len(data)
                proc.wait()
            except BaseException:
                proc.kill()
                proc.wait()
                raise
            err.seek(0)
            stderr = err.read().decode("utf-8", errors="replace")
        end = parse_end(stderr)
        if end is not None:
            return end, offset
        now = time.monotonic()
        if offset > before or failing_since is None:
            failing_since = now
        if still_there is not None and still_there() is False:
            return End("gone", detail="the box no longer exists"), offset
        if now - failing_since > patience:
            tail = stderr.strip().splitlines()[-1:] or ["no answer"]
            return End("unreachable", detail=tail[0][:200]), offset
        time.sleep(retry)


def run_remote(ssh_argv, remote: str, stdin: str | None = None, timeout: float = 60):
    """Run a short command on the box; (exit code, stdout, stderr). 255 is ssh's own failure."""
    try:
        r = subprocess.run(
            ssh_argv(remote),
            input=(stdin or "").encode(),
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 255, "", "timed out"
    return (
        r.returncode,
        r.stdout.decode("utf-8", errors="replace"),
        r.stderr.decode("utf-8", errors="replace"),
    )


def env_from(items: list[str], files: list[str], environ=os.environ) -> dict[str, str]:
    """The variables a job is given: `NAME=VALUE`, or `NAME` alone for the value in our own
    environment (so a secret need not be on a command line), then the `NAME=VALUE` lines
    of each file. Later ones win."""
    env: dict[str, str] = {}
    for path in files:
        try:
            text = open(path).read()
        except OSError as exc:
            raise JobError(f"cannot read --env-file {path}: {exc}") from None
        for n, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, sep, value = line.removeprefix("export ").partition("=")
            name = name.strip()
            if not sep or not ENV_NAME.fullmatch(name):
                raise JobError(f"{path}:{n}: expected NAME=VALUE")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            env[name] = value
    for item in items:
        name, sep, value = item.partition("=")
        if not ENV_NAME.fullmatch(name):
            raise JobError(f"--env expects NAME=VALUE or NAME, got {item!r}")
        if not sep:
            if name not in environ:
                raise JobError(f"--env {name}: {name} is not set here")
            value = environ[name]
        env[name] = value
    return env
