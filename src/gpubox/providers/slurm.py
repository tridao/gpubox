"""A Slurm cluster reached over SSH: a box is one batch job holding an allocation.

gpubox runs Slurm's own commands on the cluster's login host, through the
user's own SSH configuration (`ssh LOGIN squeue ...`), so it needs nothing on
the cluster but non-interactive SSH and a POSIX login shell. The mapping:

    rent       sbatch, with the run's tag as the job name; the job id is the box's id
    look up    squeue --jobs ID --states=all
    list       squeue --me
    destroy    scancel ID
    relabel    scontrol update JobName=TAG TimeLimit=...
    reach      ssh LOGIN srun --jobid=ID --overlap ...: a step inside the allocation,
               with the job's GPUs, CPUs and environment (CUDA_VISIBLE_DEVICES)

A job counts as gone once Slurm says it is over: in a final state, or
COMPLETING (told to end, and it cannot run again), or no longer known to the
controller ("Invalid job id specified", which it says of a job that ended
more than MinJobAge ago). A job waiting in the queue is not the machine's
fault: the launcher's start-up and stall rules do not count that time.

The batch script is the box (`batch_script`). It holds the allocation until
the lease ends and then exits, which ends the job and everything in it. Like
the timer in a Vast box, it reads the end of the lease from a file on the node,
which `renew` rewrites; Slurm's time limit is set a little past the lease, as a
backstop, and `renew` moves it too. A cluster may refuse to extend a user's
time limit: then `renew` fails, and the box ends at its old limit.

Jobs (`job.py`) cannot be started from a step: when a step ends Slurm kills
everything in it, `setsid` or not. So the batch script also starts them: a
step writes the job's directory and asks, and the batch script, which lives
as long as the box, starts the wrapper (`job.start_command(spawn="runner")`).

Not used here: an image. Boxes run in the nodes' own environment.
Observations of machines and the reputation store are Vast's: node names
are not machine ids, and a cluster's nodes are its operators' business. A
node that fails the gate is avoided for the rest of the launch (`--exclude`).
"""

from __future__ import annotations

import math
import os
import re
import shlex
import signal
import subprocess
import tempfile
import time

from . import base as prov
from ..gate import Prober
from .base import ConfigError
from ..state import parse_tag

# Each box keeps what it needs in a directory of its own on its node. Every command
# gpubox runs in the job starts with this, and so does the batch script.
BOX_ENV = (
    'export GPUBOX_BOX="/tmp/gpubox-${SLURM_JOB_ID:?not in a Slurm job}"; '
    'export GPUBOX_JOBS="$GPUBOX_BOX/jobs" '
    'GPUBOX_DESTROY_SELF="$GPUBOX_BOX/bin/gpubox-destroy-self" '
    'PATH="$GPUBOX_BOX/bin:$PATH"; '
)
# Slurm's time limit is the lease plus this, so the batch script, not Slurm, ends the
# box; Slurm only if the script could not.
SLACK_MIN = 2
FORMAT = "%i|%T|%N|%r|%j"  # the name last: it is the only field that may contain "|"
# Final states, and COMPLETING: the job holds nothing any more, or is releasing it and
# cannot run again. Anything else may still be, or become, a running job.
OVER = frozenset({
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "COMPLETING", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
})  # fmt: skip
# Waiting its turn, or held: nobody's fault.
QUEUED = frozenset({
    "PENDING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "RESV_DEL_HOLD", "SPECIAL_EXIT",
    "STOPPED", "SUSPENDED",
})  # fmt: skip
# Reasons a pending job gives when it can never run as asked: our request, not the queue.
NEVER = frozenset({
    "AccountNotAllowed", "BadConstraints", "InvalidAccount", "InvalidQOS",
    "PartitionConfig", "PartitionNodeLimit", "PartitionTimeLimit", "QOSNotAllowed",
})  # fmt: skip
# sbatch's words for trouble talking to the controller: the job may or may not exist.
UNSURE = ("timed out", "unable to contact", "zero bytes", "temporarily unavailable", "socket")
NODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
LOGIN = re.compile(r"[A-Za-z0-9_.@%:\[\]-]+")


class SlurmError(prov.ProviderError):
    pass


def ssh_argv(login: str, command: str | None, *, tty: bool = False, connect_timeout: int = 10):
    """ssh's argv for a command on the login host, with the user's own SSH configuration
    (keys, ProxyJump, a ControlMaster). A terminal only if asked: a configuration that
    forces one would mangle what gpubox reads."""
    opts = {
        "BatchMode": "yes",
        "ForwardAgent": "no",
        "ConnectTimeout": str(connect_timeout),
        "ServerAliveInterval": "15",
        "ServerAliveCountMax": "8",
        "LogLevel": "ERROR",
    }
    argv = ["ssh", "-t" if tty else "-T"]
    for k, v in opts.items():
        argv += ["-o", f"{k}={v}"]
    return [*argv, login, *([] if command is None else [command])]


def step_command(job_id: int, remote: str | None, *, tty: bool = False) -> str:
    """The login host's command for `remote` in a step of the job: srun into the job's
    allocation, overlapping whatever runs there, with the box's variables. None: a login
    shell."""
    inner = BOX_ENV + ("exec bash -l" if remote is None else remote)
    argv = ["srun", f"--jobid={int(job_id)}", "--overlap", "--nodes=1", "--ntasks=1", "--quiet"]
    argv += ["--pty"] if tty else ["--unbuffered"]
    return shlex.join([*argv, "bash", "-c", inner])


class SlurmProber(Prober):
    """The launcher's prober (gate.Prober) for a Slurm box: every command is a step of the
    job, run over the user's own SSH access to the login host. No key of the run's own:
    gpubox starts no sshd here. The gate looks only at the allocation's GPUs."""

    scoped = True

    def __init__(self):
        self.checks = []
        self.key = self.known_hosts = None
        self.pubkey = None

    def ssh(self, ep, remote, *, connect_timeout: int = 10, tty=None) -> list[str]:
        login, job_id = ep
        command = step_command(job_id, remote, tty=bool(tty))
        return ssh_argv(login, command, tty=bool(tty), connect_timeout=connect_timeout)


def batch_script(expiry: int) -> str:
    """The job's script: the box. It holds the allocation until the lease ends, then
    exits, which ends the job and everything in it.

    The end of the lease is read every second from `$GPUBOX_BOX/expiry`, which `renew`
    rewrites; it is created from `expiry` unless it is there (a requeued job must not
    undo a renewal), and what in it is not a time falls back to `expiry`. A job asks to
    be started by leaving a `start` file in its directory; the script renames it, so a
    job is started once, and starts the job's wrapper in a session of its own. The
    directory goes when the box ends, or is cancelled (SIGTERM).
    """
    if type(expiry) is not int or not 0 < expiry < 10**10:
        raise ValueError(f"bad expiry {expiry!r}")
    return f"""#!/bin/bash
# gpubox: this job is a box. It holds the allocation until its lease ends.
{BOX_ENV}
umask 077
mkdir -p "$GPUBOX_JOBS" "$GPUBOX_BOX/bin" || exit 1
if [ ! -O "$GPUBOX_BOX" ] || [ -L "$GPUBOX_BOX" ]; then
    echo "gpubox: $GPUBOX_BOX is not this user's" >&2; exit 1
fi
[ -f "$GPUBOX_BOX/expiry" ] || echo {expiry} > "$GPUBOX_BOX/expiry"
printf '%s\\n' '#!/bin/sh' '# Ends this box: it is released within seconds.' \\
    "echo 1 > $GPUBOX_BOX/expiry.tmp && mv $GPUBOX_BOX/expiry.tmp $GPUBOX_BOX/expiry &&" \\
    'echo "gpubox: this box ends within a few seconds" >&2' > "$GPUBOX_DESTROY_SELF"
chmod 700 "$GPUBOX_DESTROY_SELF"
cd "$HOME" 2>/dev/null || cd /
# A cancelled job gets SIGTERM, and KillWait before SIGKILL: time to leave the node clean.
trap 'rm -rf "$GPUBOX_BOX"; exit 0' TERM
while :; do
    e=$(cat "$GPUBOX_BOX/expiry" 2>/dev/null)
    case "$e" in ""|*[!0-9]*|???????????*) e={expiry};; esac
    [ "$(date +%s)" -ge "$e" ] && break
    for s in "$GPUBOX_JOBS"/*/start; do
        [ -e "$s" ] || continue
        j=${{s%/start}}
        mv "$s" "$j/started" 2>/dev/null || continue
        read -r limit destroy < "$j/started"
        setsid bash "$j/wrapper" "$j" "$limit" "$destroy" </dev/null >/dev/null 2>&1 &
    done
    sleep 1
done
rm -rf "$GPUBOX_BOX"
"""


def _elapsed(text: str) -> int:
    """Seconds, from squeue's elapsed time: [DD-[HH:]]MM:SS."""
    m = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", text.strip())
    if m is None:
        raise ValueError(f"not an elapsed time: {text!r}")
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + s


def run_ssh(login: str, argv: list[str], stdin: str | None, timeout: float):
    """Run a command on the login host: (exit code, stdout, stderr), exit code None if
    it did not finish in time. Output goes to files, not pipes: an ssh that starts a
    ControlMaster (the user's configuration may say so) leaves a process behind that
    holds its stderr, and a pipe would not end until it does."""
    ssh = ssh_argv(login, shlex.join(argv), connect_timeout=max(1, min(10, math.ceil(timeout))))
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        stdin_file = None
        if stdin is not None:
            stdin_file = tempfile.TemporaryFile()
            stdin_file.write(stdin.encode())
            stdin_file.seek(0)
        try:
            proc = subprocess.Popen(
                ssh, stdin=stdin_file or subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True,
            )  # fmt: skip
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                rc = None
            except BaseException:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise
            if rc is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait()
        finally:
            if stdin_file is not None:
                stdin_file.close()
        out.seek(0)
        err.seek(0)
        text = lambda f: f.read().decode("utf-8", errors="replace")
        return rc, text(out), text(err)


class Slurm:
    """A Slurm cluster as a provider of boxes (provider.py).

    `runner(argv, stdin, timeout) -> (exit code or None, stdout, stderr)` runs one
    command on the login host; tests give a fake cluster.
    """

    name = "slurm"
    noun = "job"
    credential = None  # the user's own SSH access to the login host
    access = "srun"
    spawn = "runner"  # Slurm ends what a step started: the batch script starts jobs
    own_sshd = False  # the user's own SSH and srun: nothing of ours answers SSH
    self_destroy = True  # the batch script exits
    lease = "box"  # the batch script, and Slurm's time limit behind it
    observes_machines = False  # node names are not Vast's machine ids (module docstring)
    default_spare = 0  # a spare holds a share of the cluster that others could use
    renewable = True  # its lease can be moved (relabel)
    requires = frozenset(())
    features = frozenset({"any_gpu"})
    searchable = False  # Slurm chooses the node: there is nothing to show

    def __init__(
        self,
        login: str,
        *,
        partition: str | None = None,
        account: str | None = None,
        qos: str | None = None,
        constraint: str | None = None,
        sbatch_args: tuple[str, ...] = (),
        runner=None,
        timeout: float = 30,
    ):
        if not isinstance(login, str) or not LOGIN.fullmatch(login) or login.startswith("-"):
            raise ConfigError(f"not an SSH destination: {login!r}")
        for name, value in (("partition", partition), ("account", account), ("qos", qos)):
            if value is not None and not re.fullmatch(r"[A-Za-z0-9_.,-]+", value):
                raise ConfigError(f"not a Slurm {name}: {value!r}")
        if constraint is not None and (not constraint or "\n" in constraint):
            raise ConfigError(f"not a Slurm constraint: {constraint!r}")
        for arg in sbatch_args:
            if not isinstance(arg, str) or not arg.startswith("-") or "\n" in arg:
                raise ConfigError(f"--sbatch-arg takes one sbatch option, got {arg!r}")
        # The Slurm account jobs are charged to (sbatch --account), not who we are (account()).
        self.login, self.partition, self.slurm_account = login, partition, account
        self.qos, self.constraint, self.sbatch_args = qos, constraint, tuple(sbatch_args)
        self.timeout = timeout
        self.runner = runner or (lambda argv, stdin, t: run_ssh(login, argv, stdin, t))

    # -------------------------------------------------------- settings

    def settings(self) -> dict:
        """What the state file keeps, so that later commands reach the same cluster."""
        return {
            "login": self.login, "partition": self.partition, "account": self.slurm_account,
            "qos": self.qos, "constraint": self.constraint, "sbatch_args": list(self.sbatch_args),
        }  # fmt: skip

    def options(self) -> dict:
        return self.settings()

    @classmethod
    def from_options(cls, options: dict, key=None) -> Slurm:
        return cls.from_settings(options)

    @classmethod
    def from_settings(cls, settings: dict, **kw) -> Slurm:
        if not isinstance(settings, dict) or not isinstance(settings.get("login"), str):
            raise ConfigError("the run's Slurm settings are missing from its state file")
        return cls(
            settings["login"],
            partition=settings.get("partition"),
            account=settings.get("account"),
            qos=settings.get("qos"),
            constraint=settings.get("constraint"),
            sbatch_args=tuple(settings.get("sbatch_args") or ()),
            **kw,
        )

    # -------------------------------------------------------- calls

    def _run(self, argv, *, stdin=None, timeout=None, retries=1):
        """(exit code, stdout, stderr) of a command on the login host. A connection that
        failed (ssh's 255, or no answer in time) is tried again `retries` times; only
        ever for commands that can safely run twice."""
        timeout = self.timeout if timeout is None else max(1.0, min(self.timeout, timeout))
        deadline = time.monotonic() + timeout
        for attempt in range(retries + 1):
            left = max(1.0, deadline - time.monotonic())
            rc, out, err = self.runner(list(argv), stdin, left)
            if rc not in (None, 255) or attempt == retries or time.monotonic() >= deadline:
                return rc, out, err
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        raise AssertionError("unreachable")

    def _check(self, what: str, rc, out: str, err: str) -> str:
        if rc is None:
            raise SlurmError(0, f"slurm: {what}: no answer from {self.login} in time")
        if rc == 255:
            raise SlurmError(0, f"slurm: {what}: ssh to {self.login} failed: {_tail(err)}")
        if rc != 0:
            raise SlurmError(rc, f"slurm: {what} failed (exit {rc}): {_tail(err or out)}")
        return out

    def account(self) -> str:
        """`user@cluster`: who the login belongs to, on which cluster."""
        user = _last_line(self._check("id", *self._run(["id", "-un"])))
        config = self._check("scontrol show config", *self._run(["scontrol", "show", "config"]))
        m = re.search(r"^ClusterName\s*=\s*(\S+)\s*$", config, re.M)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", user) or m is None:
            raise SlurmError(0, "slurm: cannot tell the user and the cluster's name")
        return f"{user}@{m.group(1)}"

    def candidates(self, spec, blocked: dict, policy) -> tuple[list[dict], str]:
        """No offers to choose from: Slurm chooses the node. One request per attempt."""
        offers = [
            {"id": i, "machine_id": None, "host_id": None, "dph_total": 0.0}
            for i in range(1, policy.attempts + 1)
        ]
        where = f"partition {self.partition}" if self.partition else "the default partition"
        return offers, f"Slurm via {self.login}, {where}, {spec.gpu_text}"

    def sbatch_argv(self, spec, label: str, expiry: int, avoid=(), now=None) -> list[str]:
        now = time.time() if now is None else now
        minutes = max(1, math.ceil((expiry - now) / 60)) + SLACK_MIN
        argv = [
            "sbatch", "--parsable", f"--job-name={label}", "--nodes=1", "--ntasks=1",
            f"--gres=gpu:{spec.num_gpus}", f"--time={minutes}", "--output=/dev/null",
        ]  # fmt: skip
        if spec.min_cpus is not None:
            argv.append(f"--cpus-per-task={spec.min_cpus}")
        if spec.min_ram_gb is not None:
            argv.append(f"--mem={spec.min_ram_gb}G")
        for flag, value in (
            ("--partition", self.partition), ("--account", self.slurm_account),
            ("--qos", self.qos), ("--constraint", self.constraint),
        ):  # fmt: skip
            if value is not None:
                argv.append(f"{flag}={value}")
        nodes = sorted(n for n in avoid if isinstance(n, str) and NODE.fullmatch(n))
        if nodes:
            argv.append(f"--exclude={','.join(nodes)}")
        return [*argv, *self.sbatch_args]

    def create(self, offer, spec, label: str, pubkey, expiry: int, avoid=()) -> int:
        """Submit the box's job. Sent exactly once. An explicit refusal is our request's
        fault on a cluster (no such partition, a limit), so it is a ConfigError; a job id
        is taken only from a clean exit that prints one; anything else may have
        submitted a job, and is CreateUnknown."""
        argv = self.sbatch_argv(spec, label, expiry, avoid)
        rc, out, err = self._run(argv, stdin=batch_script(expiry), retries=0)
        if rc == 0:
            # The last line: a login shell may print something of its own first.
            first = _last_line(out).split(";")[0]
            if re.fullmatch(r"[1-9][0-9]{0,15}", first):
                return int(first)
            raise prov.CreateUnknown(0, f"sbatch answered {_tail(out)!r}, not a job id")
        if rc is None or rc == 255:
            raise prov.CreateUnknown(0, f"sbatch: no usable answer: {_tail(err) or 'timed out'}")
        said = _tail(err or out)
        if rc == 127:
            raise ConfigError(f"there is no sbatch on {self.login}: {said}")
        if "sbatch: error" in err and not any(u in err.lower() for u in UNSURE):
            raise ConfigError(f"sbatch refused the job: {said}")
        raise prov.CreateUnknown(rc, f"sbatch exited {rc}: {said}")

    def _rows(self, out: str, *, strict_ids: bool) -> list[dict]:
        rows, seen = [], set()
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split("|", 4)
            if len(parts) != 5:
                raise SlurmError(0, f"slurm: unreadable squeue line: {line[:200]!r}")
            jid, state, node, reason = (p.strip() for p in parts[:4])
            name = parts[4]
            if not re.fullmatch(r"[1-9][0-9]{0,15}", jid):
                # An array or heterogeneous job (123_4, 123+0): never one of ours, unless
                # it carries a tag, which only we give.
                if strict_ids or parse_tag(name) is not None:
                    raise SlurmError(0, f"slurm: unreadable job id in squeue: {line[:200]!r}")
                continue
            if not re.fullmatch(r"[A-Z_]+", state):
                raise SlurmError(0, f"slurm: unreadable job state in squeue: {line[:200]!r}")
            if int(jid) in seen:
                raise SlurmError(0, f"slurm: job {jid} is listed twice")
            seen.add(int(jid))
            rows.append({
                "id": int(jid), "label": name, "actual_status": state,
                "node": node if NODE.fullmatch(node) else None, "reason": reason,
            })  # fmt: skip
        return rows

    def instance(self, job_id: int, *, timeout=None, retries=1) -> dict | None:
        """The job, or None when Slurm says it is over or does not know it."""
        argv = ["squeue", "--noheader", "--states=all", f"--jobs={int(job_id)}"]
        argv.append(f"--format={FORMAT}")
        rc, out, err = self._run(argv, timeout=timeout, retries=retries)
        if rc not in (None, 0, 255) and "Invalid job id specified" in err:
            return None
        rows = self._rows(self._check(f"squeue for job {job_id}", rc, out, err), strict_ids=True)
        if len(rows) != 1 or rows[0]["id"] != job_id:
            # Nothing, or something else: not evidence that the job is gone.
            raise SlurmError(0, f"slurm: unexpected answer for job {job_id}: {out[:200]!r}")
        return None if rows[0]["actual_status"] in OVER else rows[0]

    def instances(self, *, timeout=None, retries=1) -> list[dict]:
        """Every job of this user that may still hold or get an allocation, or an error."""
        argv = ["squeue", "--noheader", "--me", f"--format={FORMAT}"]
        out = self._check("squeue", *self._run(argv, timeout=timeout, retries=retries))
        return [r for r in self._rows(out, strict_ids=False) if r["actual_status"] not in OVER]

    def destroy(self, job_id: int, *, timeout=None, retries=1) -> None:
        rc, out, err = self._run(["scancel", str(int(job_id))], timeout=timeout, retries=retries)
        if rc not in (None, 0, 255) and "Invalid job id specified" in err:
            return
        self._check(f"scancel {job_id}", rc, out, err)
        if "error" in err.lower():  # scancel says some refusals on stderr with exit 0
            raise SlurmError(0, f"slurm: scancel {job_id}: {_tail(err)}")

    def relabel(self, job_id: int, label: str) -> None:
        """Set the job's name to the tag, and its time limit to the tag's expiry plus the
        slack, in one update: Slurm applies both or refuses."""
        tag = parse_tag(label)
        if tag is None:
            raise ValueError(f"not a tag: {label!r}")
        argv = ["squeue", "--noheader", "--states=all", f"--jobs={int(job_id)}", "--format=%M"]
        out = self._check(f"squeue for job {job_id}", *self._run(argv))
        try:
            used = _elapsed(_last_line(out))
        except ValueError:
            raise SlurmError(0, f"slurm: cannot read the run time of job {job_id}: {out[:100]!r}")
        minutes = math.ceil((used + max(0, tag.expiry - time.time())) / 60) + SLACK_MIN
        argv = ["scontrol", "update", f"JobId={int(job_id)}", f"JobName={label}"]
        argv.append(f"TimeLimit={minutes}")
        rc, out, err = self._run(argv, retries=0)
        self._check(f"scontrol update of job {job_id}", rc, out, err)
        if err.strip():
            raise SlurmError(0, f"slurm: scontrol update of job {job_id}: {_tail(err)}")

    def view(self, row: dict) -> prov.View:
        status = str(row.get("actual_status") or "")
        reason = str(row.get("reason") or "")
        msg = "" if reason in ("None", "(null)") else reason
        if status == "RUNNING":
            node = row.get("node")
            return prov.View("running", status, msg, None, (self.login, row["id"]), node)
        if status in OVER:
            return prov.View("exited", status, msg)
        if status in QUEUED:
            problem = "config" if status == "PENDING" and reason in NEVER else None
            return prov.View("queued", status, msg, problem)
        return prov.View("loading", status, msg)  # CONFIGURING, RESIZING, ...

    def expiry_command(self, expiry: int) -> str:
        if type(expiry) is not int or not 0 < expiry < 10**10:
            raise ValueError(f"bad expiry {expiry!r}")
        return (
            f'[ -d "$GPUBOX_BOX" ] && echo {expiry} > "$GPUBOX_BOX/expiry.tmp" && '
            'mv "$GPUBOX_BOX/expiry.tmp" "$GPUBOX_BOX/expiry"'
        )

    def manual_hint(self, run_id: str, ids) -> str:
        if ids:
            return f"Or cancel them by hand:  ssh {self.login} scancel {' '.join(map(str, ids))}"
        return f"Or look for them:  ssh {self.login} squeue --me --format=%i,%j | grep {run_id}"

    def describe(self, a) -> str:
        return f"node {a.machine_id}" if a.machine_id else ""


def _last_line(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _tail(text: str) -> str:
    lines = [line for line in (text or "").strip().splitlines() if line.strip()]
    return " / ".join(lines[-3:])[:300]
