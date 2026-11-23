"""SSH access to a rented box, and the health gate that admits it.

The gate runs on the box with the image's own python3 and prints one JSON
line. It checks the machine, not the project: a bug in the code under test
must fail its tests, not get machines blocked. Its CUDA checks use the image's
PyTorch, and run only where that is a CUDA build.

A caller's own checks (`Spec.gate_script`) run after it passes, and speak the
same verdict protocol (`read_hook`).
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .policy import REASONS

GATE_VERSION = 3  # 2: CUDA checks skipped without a CUDA build of PyTorch; 3: driver_too_old

# Verdicts the gate gives that are about the caller's choice, not the machine: the box
# fails, nobody is blamed. driver_too_old: the host's driver supports an older CUDA major
# version than the image's PyTorch was built for (CUDA 13 on a 12.x driver fails; any
# 13.x driver runs a 13.x build).
NO_BLAME = ("driver_too_old",)

GATE_SCRIPT = r"""
import json, os, re, shutil, subprocess, sys, time, urllib.error, urllib.request, warnings

want = json.loads(sys.argv[1])
out = {"ok": False}

def done(reason=None, detail=""):
    if reason:
        out["reason"], out["detail"] = reason, detail
    else:
        out["ok"] = True
    print(json.dumps(out))
    sys.exit(0)

def smi():
    r = subprocess.run(
        ["nvidia-smi", "--format=csv,noheader,nounits", "--query-gpu="
         "index,name,memory.total,memory.used,utilization.gpu,power.draw,driver_version"],
        capture_output=True, text=True, timeout=30)
    if r.returncode:
        done("runtime_error", "nvidia-smi failed: " + (r.stderr or r.stdout).strip()[:300])
    return [[c.strip() for c in line.split(",")] for line in r.stdout.strip().splitlines()]

def driver_cuda():
    # (major, minor) of the newest CUDA the driver supports, from nvidia-smi's header.
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
        m = re.search(r"CUDA Version: *(\d+)\.(\d+)", r.stdout)
        return (int(m.group(1)), int(m.group(2))) if m else None
    except Exception:
        return None

rows = smi()
out["gpus"] = [r[1] for r in rows]
out["driver"] = rows[0][6] if rows else None
out["cpus"] = os.cpu_count()
out["disk_free_gb"] = shutil.disk_usage("/").free >> 30
if len(rows) != want["num_gpus"]:
    done("wrong_hardware", "expected %d GPUs, nvidia-smi lists %d" % (want["num_gpus"], len(rows)))
for r in rows:
    if not any(model in r[1] for model in want["gpus"]):
        done("wrong_hardware", "GPU %s is %r, not %s" % (r[0], r[1], " or ".join(want["gpus"])))
    if int(r[2]) < want["min_vram_mib"]:
        done("wrong_hardware", "GPU %s has %s MiB, need %d" % (r[0], r[2], want["min_vram_mib"]))
if out["disk_free_gb"] < 10:
    done("wrong_hardware", "only %d GB of disk free" % out["disk_free_gb"])

# A box that cannot reach these would fail later, looking like a test failure.
for url in want["must_reach"]:
    for attempt in range(2):
        try:
            urllib.request.urlopen(url, timeout=15).close()
            break
        except urllib.error.HTTPError:
            break  # any HTTP answer means the network path works
        except Exception as exc:
            if attempt:
                done("no_network", "cannot reach %s: %s" % (url, str(exc)[:200]))

# Hidden host-side load (another tenant, a miner): sample before this process
# creates a CUDA context, so anything seen here is not ours.
util = used = 0
watts = "?"
for i in range(3):
    if i:
        time.sleep(2)
    for r in smi():
        if int(r[4]) >= util:
            watts = r[5]
        util, used = max(util, int(r[4])), max(used, int(r[3]))
out["idle_util"], out["idle_used_mib"], out["idle_watts"] = util, used, watts
if util > 20 or used > 512:
    done("busy_gpu", "GPU busy before our first CUDA context: util %d%%, %d MiB used, %s W"
         % (util, used, watts))

# From here on a failure we did not foresee (an import that dies on a broken
# driver, a CUDA error) is still a statement about this box, so it becomes a
# verdict instead of a traceback. done() leaves through SystemExit, untouched.
#
# The CUDA checks need a CUDA build of PyTorch. An image without PyTorch, or
# with a CPU-only build, has no use for them, and that says nothing about the
# machine: they are skipped. A CUDA build that cannot use the GPU is the machine,
# unless the driver is for an older CUDA major version than the build: that is the
# caller's choice of host (`--min-cuda`), and it is said so.
try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            import torch
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise  # PyTorch is there, and something it needs is not
            torch = None
        cuda_build = torch is not None and bool(getattr(torch.version, "cuda", None))
        available = cuda_build and torch.cuda.is_available()
    if torch is None:
        out["checks"] = "no PyTorch in the image: CUDA checks skipped"
        done()
    out["torch"] = torch.__version__
    if not cuda_build:
        out["checks"] = "CPU-only PyTorch: CUDA checks skipped"
        done()
    if not available:
        built = re.match(r"(\d+)\.(\d+)", str(torch.version.cuda))
        drv = driver_cuda()
        if built and drv and drv[0] < int(built.group(1)):
            out["needs_cuda"] = int(built.group(1))
            done("driver_too_old", "the host's driver supports CUDA %d.%d, the image's PyTorch "
                 "is built for CUDA %s" % (drv[0], drv[1], torch.version.cuda))
        done("cuda_init", "CUDA unavailable: " + "; ".join(str(w.message) for w in caught)[:300])
    if torch.cuda.device_count() != want["num_gpus"]:
        done("cuda_init", "torch sees %d GPUs, expected %d"
             % (torch.cuda.device_count(), want["num_gpus"]))

    gen = torch.Generator().manual_seed(0)
    a, b = torch.randn(1024, 1024, generator=gen), torch.randn(1024, 1024, generator=gen)
    ref = a.double() @ b.double()
    for i in range(want["num_gpus"]):
        dev = "cuda:%d" % i
        cap = list(torch.cuda.get_device_capability(i))
        if want["capability"] and cap != want["capability"]:
            done("compute_mismatch", "%s has compute capability %s, expected %s"
                 % (dev, cap, want["capability"]))
        err = ((a.to(dev) @ b.to(dev)).cpu().double() - ref).abs().max().item()
        if not err < 1e-2:
            done("runtime_error", "%s: matmul differs from the CPU reference by %g" % (dev, err))
        # Most of the free VRAM must be allocatable and writable.
        free, _ = torch.cuda.mem_get_info(i)
        x = torch.empty(int(free * 0.8), dtype=torch.uint8, device=dev)
        x.fill_(7)
        if int(x[-1]) != 7 or int(x[0]) != 7:
            done("runtime_error", "%s: memory read-back failed" % dev)
        del x
        torch.cuda.empty_cache()
    out["checks"] = "CUDA checks passed"
except Exception as exc:
    done("runtime_error", "%s: %s" % (type(exc).__name__, str(exc)[:300]))
done()
"""


def _clean_report(raw: dict) -> dict:
    """The fields of the gate's report that callers print, in known types."""
    gpus = raw.get("gpus")
    text = lambda x: x if isinstance(x, str) else None
    return {
        "gpus": [g for g in gpus if isinstance(g, str)] if isinstance(gpus, list) else [],
        "cpus": raw.get("cpus") if type(raw.get("cpus")) is int else None,
        "driver": text(raw.get("driver")),
        "torch": text(raw.get("torch")),
        "checks": text(raw.get("checks")),
        # The CUDA major version the image needs, said only with driver_too_old.
        "needs_cuda": raw.get("needs_cuda") if type(raw.get("needs_cuda")) is int else None,
    }


class Prober:
    """SSH to candidate boxes with a throwaway key that belongs to one run."""

    def __init__(self, key_dir: Path, create: bool = True):
        """`create=False` only loads: a run that has been torn down has no key,
        and looking for it must not bring one back."""
        self.checks: list[Check] = []  # every child process started, for kill_all
        self.key = Path(key_dir) / "id_ed25519"
        self.known_hosts = Path(key_dir) / "known_hosts"
        if not self.key.exists():
            if not create:
                raise FileNotFoundError(f"no SSH key in {key_dir}")
            Path(key_dir).mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "gpubox", "-f", self.key],
                check=True,
            )
        self.pubkey = self.key.with_name(self.key.name + ".pub").read_text().strip()

    def ssh(
        self, ep: tuple[str, int], remote: str | None, *, connect_timeout: int = 10, tty=None
    ) -> list[str]:
        """ssh's argv for `remote` on the box; None for a login shell. `tty`: True to ask
        for a terminal, False to refuse one, None to let ssh decide."""
        opts = {
            "BatchMode": "yes",
            "IdentitiesOnly": "yes",
            "IdentityAgent": "none",  # never forward or offer the user's own keys
            "ForwardAgent": "no",
            # Trust on first use, into a file that lives with this run. The box
            # holds no secret of ours; the key is what authenticates us to it.
            "StrictHostKeyChecking": "accept-new",
            "UserKnownHostsFile": str(self.known_hosts),
            "ConnectTimeout": str(connect_timeout),
            "ServerAliveInterval": "15",
            "ServerAliveCountMax": "8",
            "LogLevel": "ERROR",
        }
        # -F /dev/null: nothing from the user's or the system's ssh_config
        # (SendEnv, extra identities, proxies) applies to these connections.
        argv = ["ssh", "-F", "/dev/null", "-i", str(self.key), "-p", str(ep[1])]
        if tty is not None:
            argv.append("-t" if tty else "-T")
        for k, v in opts.items():
            argv += ["-o", f"{k}={v}"]
        return [*argv, f"root@{ep[0]}", *([] if remote is None else [remote])]

    def run(self, ep: tuple[str, int], remote: str, timeout: float = 60) -> int:
        """Run a short command on the box and wait for it. Returns ssh's exit code, 255 if
        it could not connect or did not finish in time."""
        try:
            return subprocess.run(
                self.ssh(ep, remote), stdin=subprocess.DEVNULL, timeout=timeout
            ).returncode
        except subprocess.TimeoutExpired:
            return 255

    def _start(self, argv: list[str], read, **kw) -> Check:
        """Start a child and remember it, as one step that a signal cannot split: a child
        that is running but that nobody knows about could not be stopped."""
        old = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        try:
            check = Check(Child(argv, **kw), read)
            self.checks.append(check)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)  # a pending signal arrives here
        return check

    def kill_all(self) -> None:
        """Stop every child this prober started. Safe to call at any time, and twice."""
        for check in self.checks:
            check.kill()

    def start_ready(self, ep: tuple[str, int]) -> Check:
        """Start an SSH probe. Its result is (ok, last words of ssh)."""
        argv = self.ssh(ep, "true", connect_timeout=8)
        return self._start(argv, read_ready, timeout=READY_TIMEOUT)

    def start_gate(self, ep: tuple[str, int], spec) -> Check:
        """Start the health gate. Its result is (ok, reason_code, detail, report)."""
        want = {
            "num_gpus": spec.num_gpus,
            "gpus": list(spec.gpus),
            "min_vram_mib": spec.min_vram_mib,
            "capability": list(spec.capability) if spec.capability else None,
            "must_reach": list(spec.must_reach),
        }
        argv = self.ssh(ep, "python3 - " + shlex.quote(json.dumps(want)))
        return self._start(argv, read_gate, stdin=GATE_SCRIPT, timeout=GATE_TIMEOUT)

    def start_hook(self, ep: tuple[str, int], spec) -> Check:
        """Start the caller's own gate script. Its result is (ok, reason_code, detail, {})."""
        argv = self.ssh(ep, HOOK_REMOTE)
        timeout = spec.gate_timeout
        read = lambda done: read_hook(done, timeout)
        return self._start(argv, read, stdin=spec.gate_script, timeout=timeout)


READY_TIMEOUT = 25
GATE_TIMEOUT = 240
# The caller's gate script is stored first, so nothing it runs can read it from
# stdin, then run: by its own #! line if it has one, else by bash.
HOOK_REMOTE = (
    'f=$(mktemp /tmp/gpubox-gate.XXXXXX) && cat > "$f" && chmod 700 "$f" && '
    'if [ "$(head -c 2 "$f")" = "#!" ]; then exec "$f" </dev/null; '
    'else exec bash "$f" </dev/null; fi'
)
KEEP = 1 << 20  # bytes of a child's output that are read back


@dataclass
class Done:
    """How a child process ended."""

    rc: int | None  # None: it was killed at its deadline
    stdout: str
    stderr: str

    @property
    def timed_out(self) -> bool:
        return self.rc is None


class Child:
    """A process that is polled, never waited for.

    What it reads comes from a file and what it writes goes to files, so
    neither side can block on a pipe however much the box prints. It runs in
    a session of its own: a Ctrl-C at the terminal reaches the launcher, which
    decides what happens to its children.
    """

    def __init__(self, argv: list[str], *, stdin: str | None = None, timeout: float):
        self._out, self._err = tempfile.TemporaryFile(), tempfile.TemporaryFile()
        self._in = None
        if stdin is not None:
            self._in = tempfile.TemporaryFile()
            self._in.write(stdin.encode())
            self._in.seek(0)
        self.deadline = time.monotonic() + timeout
        self.done: Done | None = None
        try:
            self.proc = subprocess.Popen(
                argv,
                stdin=self._in if self._in is not None else subprocess.DEVNULL,
                stdout=self._out,
                stderr=self._err,
                start_new_session=True,
            )
        except BaseException:
            self._close()
            raise

    def _close(self) -> None:
        for f in (self._in, self._out, self._err):
            if f is not None:
                f.close()

    def _read(self, f) -> str:
        """The end of what was written, bounded: the gate's verdict is its last line."""
        size = f.seek(0, os.SEEK_END)
        f.seek(max(0, size - KEEP))
        return f.read(KEEP).decode("utf-8", errors="replace")  # whatever the box printed

    def poll(self) -> Done | None:
        """None while it runs. At its deadline it is killed, and that is its result."""
        if self.done is not None:
            return self.done
        rc = self.proc.poll()
        if rc is None:
            if time.monotonic() < self.deadline:
                return None
            self.kill()
            return self.done
        self.done = Done(rc, self._read(self._out), self._read(self._err))
        self._close()
        return self.done

    def kill(self) -> None:
        """Stop it now and collect it. Safe to call twice, and after it ended."""
        if self.done is not None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)  # its own session: ssh and anything it started
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=1)  # SIGKILL: this returns at once, or something is very wrong
        except subprocess.TimeoutExpired:
            pass
        self.done = Done(None, self._read(self._out), self._read(self._err))
        self._close()


class Check:
    """A child process together with how to read its result."""

    def __init__(self, child: Child, read):
        self.child, self._read = child, read

    def poll(self):
        done = self.child.poll()
        return None if done is None else self._read(done)

    def kill(self) -> None:
        self.child.kill()


def read_ready(done: Done) -> tuple[bool, str]:
    if done.timed_out:
        return False, "ssh timed out"
    return done.rc == 0, done.stderr.strip()[-200:]


def read_gate(done: Done) -> tuple[bool, str | None, str, dict]:
    """(ok, reason_code, detail, report) from how the gate's process ended.

    A box is admitted only on a clean exit with an explicit `ok: true`.
    A reason code is returned only for a verdict the script itself gave;
    output we cannot read (a crash, garbage, an unknown code) fails the
    box without a reason, so nobody is blamed for our own bugs.
    """
    if done.timed_out:
        return False, "unreachable", f"health gate hung for {GATE_TIMEOUT}s", _clean_report({})
    if done.rc == 255:
        detail = f"ssh dropped during gate: {done.stderr.strip()[-200:]}"
        return False, "unreachable", detail, _clean_report({})
    try:
        raw = json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raw = None
    if done.rc != 0 or not isinstance(raw, dict) or type(raw.get("ok")) is not bool:
        tail = (done.stderr or done.stdout).strip()[-300:]
        detail = f"health gate gave no verdict (rc={done.rc}): {tail}"
        return False, None, detail, _clean_report({})
    report = _clean_report(raw)
    if raw["ok"]:
        return True, None, "", report
    reason, detail = raw.get("reason"), str(raw.get("detail") or "")[:400]
    if reason in NO_BLAME:
        return False, None, detail, report
    if not isinstance(reason, str) or reason not in REASONS:
        return False, None, f"health gate gave an unknown reason {reason!r}: {detail}", report
    return False, reason, detail, report


def read_hook(done: Done, timeout: float) -> tuple[bool, str | None, str, dict]:
    """(ok, reason_code, detail, {}) from how the caller's gate script ended.

    It passes on exit 0. It blames the machine only with an explicit verdict,
    a last line `{"ok": false, "reason": CODE, "detail": TEXT}` naming a known
    reason code; any other failure (an exit code, a crash, a hang) fails the box
    and blames nobody, since the script is the caller's code, not ours. As for
    the built-in gate, exit 255 reads as a dropped connection.
    """
    if done.timed_out:
        return False, None, f"gate script did not finish within {timeout:g}s", {}
    if done.rc == 255:
        detail = f"ssh dropped during gate script: {done.stderr.strip()[-200:]}"
        return False, "unreachable", detail, {}
    try:
        raw = json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raw = None
    tail = (done.stderr or done.stdout).strip()[-300:]
    if isinstance(raw, dict) and raw.get("ok") is False:
        reason, detail = raw.get("reason"), str(raw.get("detail") or "")[:400]
        if isinstance(reason, str) and reason in REASONS:
            return False, reason, f"gate script: {detail}", {}
        return False, None, f"gate script failed (reason {reason!r}): {detail or tail}", {}
    if done.rc == 0:
        return True, None, "", {}
    return False, None, f"gate script exited {done.rc}: {tail}", {}
