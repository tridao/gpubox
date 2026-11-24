"""Jobs on a box: a script started detached, its log followed from wherever we are.

A job lives in a directory on the box, `/var/lib/gpubox/jobs/<id>/` (or under
$GPUBOX_JOBS, if the box's environment sets it):

    script     what to run
    env        the caller's variables (secrets among them), 0600; deleted as the job starts
    log        the script's stdout and stderr
    wpid       "<pid> <start time>" of the wrapper that runs the script and waits for it
    pid        the script's process group
    exit       its exit code, written (atomically) once it has ended
    expired    present if the box ended it at its timeout
    cancelled  present if it was killed (`gpubox kill`)

The wrapper runs in a session of its own, so nothing that happens to an SSH
connection reaches it: a dropped connection, a killed `gpubox run`, a laptop
that went to sleep. It owns the job's end: the timeout, kept on the box for
the same reason, and a kill both go through it, and it signals only its own
child's process group. The job's variables are loaded in that child alone, so
nothing the caller passes can change what the wrapper does.

What reads the log is a small program on the box (TOOL) that prints the log
from a byte offset and ends with one status line on stderr; the controller
counts the bytes it got, so after a dropped connection it carries on where it
was. The same program lists jobs and kills them, with one test of whether a
wrapper is alive: its pid, its start time, and not a zombie.

The job sees the container's environment (Docker's ENV, Vast's variables,
saved by the bootstrap), then the caller's variables, then GPUBOX_RUN_ID,
GPUBOX_JOB, GPUBOX_BOX_INDEX, GPUBOX_BOX_COUNT and GPUBOX_BOX_NAME.
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
# The jobs directory, as a shell on the box finds it (after loading the environment).
JOBS = '"${GPUBOX_JOBS:-%s}"' % JOBS_DIR
LOAD_ENV = v.LOAD_ENV
JOB_ID = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9]{3}-[0-9a-f]{4}")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
MARK = "GPUBOX-JOB"  # starts the follower's status line
KILL_GRACE = 10  # seconds between SIGTERM and SIGKILL when a job's time is up or it is killed
START_WAIT = 10  # seconds the start waits for the wrapper to have started the script


class JobError(RuntimeError):
    """A job cannot be started or found as asked."""


def new_job_id(now: float | None = None) -> str:
    """Sortable by start time, to the millisecond, so the newest job is the last name."""
    now = time.time() if now is None else now
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now))
    return f"{stamp}-{int(now * 1000) % 1000:03d}-{secrets.token_hex(2)}"


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
# destroy the box when the script ends. Run by bash, in a session of its own. Its own
# variables are never touched by the job's: those are loaded in the child only.
_WRAPPER = r"""
d=$1; limit=$2; destroy=$3
echo "$$ $(cut -d' ' -f22 /proc/$$/stat 2>/dev/null)" > "$d/wpid"
setsid bash -c '
    [ -r %(env_file)s ] && . %(env_file)s
    set -a; . "$1/env"; set +a
    rm -f "$1/env"
    cd "${GPUBOX_WORKDIR:-$HOME}" 2>/dev/null || cd /
    exec bash "$1/script"
' gpubox-job "$d" </dev/null >>"$d/log" 2>&1 &
child=$!
echo "$child" > "$d/pid.tmp" && mv "$d/pid.tmp" "$d/pid"
# The end of the job: SIGTERM to its process group, SIGKILL to whatever is left of it
# after the grace. Waits for that, so no exit code is written while it still runs.
escalate() {
    kill -TERM -- -"$child" 2>/dev/null
    n=0
    while kill -0 -- -"$child" 2>/dev/null && [ "$n" -lt %(grace2)d ]; do
        sleep 0.5; n=$((n + 1))
    done
    kill -KILL -- -"$child" 2>/dev/null
}
trap 'touch "$d/cancelled"; escalate' TERM
deadline=0
[ "$limit" -gt 0 ] && deadline=$(( $(date +%%s) + limit ))
while kill -0 "$child" 2>/dev/null; do
    if [ "$deadline" -gt 0 ] && [ "$(date +%%s)" -ge "$deadline" ]; then
        touch "$d/expired"; escalate; deadline=0
    fi
    sleep 1 & wait $!
done
wait "$child"; rc=$?
[ -e "$d/expired" ] && rc=124
echo "$rc" > "$d/exit.tmp" && mv "$d/exit.tmp" "$d/exit"
if [ "$destroy" = 1 ]; then "${GPUBOX_DESTROY_SELF:-%(destroy_self)s}"; fi
"""


def wrapper() -> str:
    return _WRAPPER % {
        "grace2": 2 * KILL_GRACE, "env_file": v.ENV_FILE, "destroy_self": v.DESTROY_SELF,
    }  # fmt: skip


def start_command(
    job_id: str,
    script: str,
    env: dict[str, str],
    *,
    timeout_s: int = 0,
    destroy_after: bool = False,
    spawn: str = "setsid",
) -> tuple[str, str]:
    """(remote command, its stdin) that starts a job. The script and the variables travel
    on stdin, never on a command line. It prints the job id once the wrapper has started
    the script; if the wrapper does not, it removes the variables and fails.

    `spawn`: "setsid" starts the wrapper from this command, in a session of its own.
    "runner" asks a process on the box that outlives this command to start it (a Slurm
    box's batch script, slurm.batch_script: Slurm kills whatever a step started when the
    step ends): the command leaves a `start` file and waits until it has been taken. If
    it is not taken within 30 s, the command takes it back and fails."""
    check_job_id(job_id)
    if type(timeout_s) is not int or timeout_s < 0:
        raise JobError(f"bad timeout {timeout_s!r}")
    hint = "is setsid there?" if spawn == "setsid" else "is the batch script of the box alive?"
    if spawn == "setsid":
        launch = (
            f'setsid bash "$d/wrapper" "$d" {timeout_s} {int(destroy_after)}'
            " </dev/null >/dev/null 2>&1 &\n"
        )
    elif spawn == "runner":
        launch = (
            f'echo {timeout_s} {int(destroy_after)} > "$d/start.tmp" && mv "$d/start.tmp" "$d/start"\n'
            'i=0; while [ -e "$d/start" ] && [ "$i" -lt 300 ]; do sleep 0.1; i=$((i + 1)); done\n'
            # Not taken: take it back, unless the runner took it in the meantime.
            'if [ -e "$d/start" ] && rm "$d/start" 2>/dev/null; then rm -f "$d/env"; '
            "echo 'the box did not take the job in 30 s' >&2; exit 1; fi\n"
        )
    else:
        raise JobError(f"unknown way to start a job: {spawn!r}")
    program = (
        LOAD_ENV + "\n"
        "set -eu\n"
        "umask 077\n"
        f'J={JOBS}; d="$J/{job_id}"\n'
        'mkdir -p "$J" && mkdir "$d"\n'
        + _heredoc('"$d/script"', script)
        + _heredoc('"$d/env"', env_lines(env))
        + _heredoc('"$d/wrapper"', wrapper())
        + ': > "$d/log"\n'
        + launch
        + f'i=0; while [ ! -e "$d/pid" ] && [ "$i" -lt {START_WAIT * 10} ]; do '
        "sleep 0.1; i=$((i + 1)); done\n"
        f'if [ -e "$d/pid" ]; then echo {job_id}; else rm -f "$d/env"; '
        f"echo 'the job did not start ({hint})' >&2; exit 1; fi\n"
    )
    return "bash -s", program


# On the box: `follow DIR OFFSET MODE`, `list JOBS`, `latest JOBS` or `kill DIR`.
# follow prints the log from OFFSET (MODE 0: what there is; 1: until the job ends;
# 2: wait without printing) and ends with one line on stderr: "GPUBOX-JOB exit N",
# "GPUBOX-JOB running", "GPUBOX-JOB lost" or "GPUBOX-JOB none".
TOOL = r"""
import os, re, signal, sys, time
def alive(d):
    # The job's wrapper, recognised by its pid and its start time (a process that got
    # its pid after a container restart is not it), and not a zombie.
    try:
        with open(os.path.join(d, "wpid")) as f:
            pid, start = (f.read().split() + [""])[:2]
        pid = int(pid)
        if not start:  # no /proc where it started
            os.kill(pid, 0)
            return pid
        with open("/proc/%%d/stat" %% pid) as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return pid if fields[0] != "Z" and fields[19] == start else None
    except PermissionError:
        return pid
    except (OSError, ValueError, IndexError):
        return None
def exit_code(d):
    try:
        with open(os.path.join(d, "exit")) as f:
            return f.read().strip()
    except FileNotFoundError:
        return None
def status(d):
    rc = exit_code(d)
    if rc is not None:
        return "exit " + rc
    if not os.path.exists(os.path.join(d, "wpid")):
        # Being started, unless that was long ago: a start the box stopped under.
        young = time.time() - os.path.getmtime(d) < %(start_wait)d + 60
        return "running" if young else "lost"
    return "running" if alive(d) else "lost"
def jobs(root):
    try:
        names = os.listdir(root)
    except FileNotFoundError:
        return []
    return sorted(n for n in names if re.fullmatch(r"%(job_id)s", n))
def follow(d, off, mode):
    out = sys.stdout.buffer
    def end(what):
        out.flush()
        sys.stderr.write("\nGPUBOX-JOB %%s\n" %% what)
        sys.stderr.flush()
        sys.exit(0)
    def chunk():
        nonlocal off
        if mode == 2:
            return False
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
    if not os.path.isdir(d):
        end("none")
    started, gone = time.monotonic(), None
    while True:
        if chunk():
            continue
        rc = exit_code(d)
        if rc is not None:
            while chunk():
                pass
            end("exit " + rc)
        if mode == 0:
            end("running")
        if alive(d) or time.monotonic() - started < 5:
            gone = None
        elif gone is None:
            gone = time.monotonic()
        elif time.monotonic() - gone > 3:  # not just between the script's end and the exit file
            end("lost")
        time.sleep(0.5)
def kill(d, grace):
    if not os.path.isdir(d):
        sys.exit("no such job")
    if exit_code(d) is not None:
        print("already ended: exit " + exit_code(d))
        return
    pid = alive(d)
    if pid is None:
        sys.exit("the job's wrapper is gone: the job is lost, not killed")
    os.kill(pid, signal.SIGTERM)  # the wrapper ends its own job's process group
    deadline = time.monotonic() + grace + 15
    while exit_code(d) is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if exit_code(d) is None:
        sys.exit("the job did not end within %%ds" %% (grace + 15))
    print("killed: exit " + exit_code(d))
cmd = sys.argv[1]
if cmd == "follow":
    follow(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
elif cmd == "list":
    for n in jobs(sys.argv[2]):
        print(n, status(os.path.join(sys.argv[2], n)))
elif cmd == "latest":
    print((jobs(sys.argv[2]) or [""])[-1])
elif cmd == "kill":
    kill(sys.argv[2], int(sys.argv[3]))
""" % {"job_id": JOB_ID.pattern, "start_wait": START_WAIT}


def _tool(*args: str) -> str:
    return LOAD_ENV + f"exec python3 -c {shlex.quote(TOOL)} " + " ".join(args)


def follow_command(job_id: str, offset: int, mode: int) -> str:
    """mode 0: print what there is; 1: until the job ends; 2: wait for it, printing nothing."""
    check_job_id(job_id)
    if mode not in (0, 1, 2):
        raise JobError(f"bad follow mode {mode!r}")
    return _tool("follow", f"{JOBS}/{job_id}", str(int(offset)), str(mode))


def latest_command() -> str:
    """Prints the newest job's id, or an empty line."""
    return _tool("latest", JOBS)


def list_command() -> str:
    """One line per job, oldest first: its id, then `exit N`, `running` or `lost`."""
    return _tool("list", JOBS)


def kill_command(job_id: str) -> str:
    """Asks the job's wrapper to end it, and waits until it has."""
    check_job_id(job_id)
    return _tool("kill", f"{JOBS}/{job_id}", str(KILL_GRACE))


@dataclass(frozen=True)
class End:
    """How following a job ended.

    exit      the job ended with `rc`
    running   it is still running (we were not asked to wait)
    lost      it ended without an exit code: its box restarted, or its wrapper was killed
    none      there is no such job on the box
    gone      the box no longer exists
    unreachable  no connection for longer than we were prepared to wait
    cancelled    the caller asked to stop following (the job goes on)
    """

    kind: str
    rc: int | None = None
    detail: str = ""


def parse_end(stderr: str) -> End | None:
    """The follower's status line, or None if it never finished saying it (a dropped
    connection). Only a whole line counts: `exit 1` may be the start of `exit 124`."""
    for line in reversed(stderr.split("\n")[:-1]):  # the last piece has no newline yet
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
    quiet: bool = False,
    offset: int = 0,
    still_there=None,
    patience: float = 600,
    retry: float = 5,
    started=None,
    cancel=None,
) -> tuple[End, int]:
    """Copy a job's log to `out` (a binary stream) from `offset` until the job ends (or,
    without `wait`, until what there is has been copied; with `quiet`, copy nothing and
    wait). Returns how it ended and the offset reached.

    A dropped connection is not the end of anything: the follower is started again
    from the offset reached. Its status line counts only if it is whole and the
    follower exited cleanly. Between tries `still_there()`, if given, says whether
    the box still exists (False ends it as "gone"; None, unknown, keeps trying). After
    `patience` seconds without a connection, it gives up as "unreachable"; the job, if
    any, carries on regardless. `started(proc)`, if given, is told of each ssh process,
    so that an interrupted caller can stop it; once `cancel` (a threading.Event) is set,
    no new connection is made and it ends as "cancelled".
    """
    mode = 2 if quiet else (1 if wait else 0)
    failing_since = None
    while True:
        if cancel is not None and cancel.is_set():
            return End("cancelled"), offset
        before, t0 = offset, time.monotonic()
        with tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(
                ssh_argv(follow_command(job_id, offset, mode)),
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err,
            )  # fmt: skip
            if started is not None:
                started(proc)
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
        end = parse_end(stderr) if proc.returncode == 0 else None
        if end is not None:
            return end, offset
        now = time.monotonic()
        # A connection that brought something, or held for a while, was a connection:
        # the next outage is counted from now.
        if failing_since is None or offset > before or now - t0 > 30:
            failing_since = now
        if still_there is not None and still_there() is False:
            return End("gone", detail="the box no longer exists"), offset
        if now - failing_since > patience:
            tail = stderr.strip().splitlines()[-1:] or ["no answer"]
            return End("unreachable", detail=tail[0][:200]), offset
        if cancel is not None:
            cancel.wait(retry)
        else:
            time.sleep(retry)


def run_remote(ssh_argv, remote: str, stdin: str | None = None, timeout: float = 60):
    """Run a short command on the box; (exit code, stdout, stderr). 255 is ssh's own failure.

    stderr goes to a file, not a pipe: an ssh that starts a ControlMaster (a user's
    configuration may say so, and a Slurm box is reached with it) leaves a process
    behind that holds its stderr, and a pipe would not end until that process does."""
    with tempfile.TemporaryFile() as err:
        try:
            r = subprocess.run(
                ssh_argv(remote),
                input=(stdin or "").encode(),
                stdout=subprocess.PIPE,
                stderr=err,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return 255, "", "timed out"
        err.seek(0)
        stderr = err.read().decode("utf-8", errors="replace")
    return r.returncode, r.stdout.decode("utf-8", errors="replace"), stderr


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
