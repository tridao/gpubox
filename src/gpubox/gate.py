"""SSH access to a rented box, and the health gate that admits it.

The gate runs on the box with the image's own python3 and PyTorch and prints
one JSON line. It checks the machine, not the project: a bug in the code under
test must fail its tests, not get machines blocked.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from .policy import REASONS

GATE_VERSION = 1

GATE_SCRIPT = r"""
import json, os, shutil, subprocess, sys, time, urllib.error, urllib.request, warnings

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

rows = smi()
out["gpus"] = [r[1] for r in rows]
out["driver"] = rows[0][6] if rows else None
out["cpus"] = os.cpu_count()
out["disk_free_gb"] = shutil.disk_usage("/").free >> 30
if len(rows) != want["num_gpus"]:
    done("wrong_hardware", "expected %d GPUs, nvidia-smi lists %d" % (want["num_gpus"], len(rows)))
for r in rows:
    if want["gpu"] not in r[1]:
        done("wrong_hardware", "GPU %s is %r, not %s" % (r[0], r[1], want["gpu"]))
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

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import torch
    available = torch.cuda.is_available()
out["torch"] = torch.__version__
if not available:
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
done()
"""


class Prober:
    """SSH to candidate boxes with a throwaway key generated for this state directory."""

    def __init__(self, state_dir: Path, fresh: bool = False):
        self.key = Path(state_dir) / "id_ed25519"
        self.known_hosts = Path(state_dir) / "known_hosts"
        if fresh:
            # A new run may be given an address an earlier run's box had, with
            # a different host key; the old entry would make it look unreachable.
            self.known_hosts.unlink(missing_ok=True)
        if not self.key.exists():
            Path(state_dir).mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "gpubox", "-f", self.key],
                check=True,
            )
        self.pubkey = self.key.with_name(self.key.name + ".pub").read_text().strip()

    def ssh(self, ep: tuple[str, int], remote: str, *, connect_timeout: int = 10) -> list[str]:
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
        for k, v in opts.items():
            argv += ["-o", f"{k}={v}"]
        return [*argv, f"root@{ep[0]}", remote]

    def ready(self, ep: tuple[str, int]) -> tuple[bool, str]:
        try:
            r = subprocess.run(
                self.ssh(ep, "true", connect_timeout=8), capture_output=True, text=True, timeout=25
            )
        except subprocess.TimeoutExpired:
            return False, "ssh timed out"
        return r.returncode == 0, r.stderr.strip()[-200:]

    def gate(self, ep: tuple[str, int], spec) -> tuple[bool, str | None, str, dict]:
        """Returns (ok, reason_code, detail, report)."""
        want = {
            "num_gpus": spec.num_gpus,
            "gpu": spec.gpu,
            "min_vram_mib": spec.min_vram_mib,
            "capability": list(spec.capability) if spec.capability else None,
            "must_reach": list(spec.must_reach),
        }
        try:
            r = subprocess.run(
                self.ssh(ep, "python3 - " + shlex.quote(json.dumps(want))),
                input=GATE_SCRIPT,
                capture_output=True,
                text=True,
                timeout=240,
            )
        except subprocess.TimeoutExpired:
            return False, "unreachable", "health gate hung for 240s", {}
        if r.returncode == 255:
            return False, "unreachable", f"ssh dropped during gate: {r.stderr.strip()[-200:]}", {}
        try:
            report = json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            report = None
        if not isinstance(report, dict) or not isinstance(report.get("ok"), bool):
            tail = (r.stderr or r.stdout).strip()[-300:]
            return False, "runtime_error", f"health gate crashed (rc={r.returncode}): {tail}", {}
        if report["ok"]:
            return True, None, "", report
        reason, detail = report.get("reason"), str(report.get("detail") or "")
        if reason not in REASONS:
            # A verdict we do not understand is our bug, not the machine's.
            return False, None, f"health gate gave an unknown reason {reason!r}: {detail}", report
        return False, reason, detail, report
