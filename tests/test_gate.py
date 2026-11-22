"""The SSH wrapper and the reading of the health gate's verdict."""

import json
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from gpubox import gate, policy
from gpubox.gate import Prober
from gpubox.launch import Spec

SPEC = Spec(image="img", gpu="RTX 5090", max_dph_per_gpu=1.25)
REFERENCE = json.loads((Path(__file__).parent / "fixtures/reference_requests.json").read_text())


def prober(tmp_path):
    (tmp_path / "id_ed25519").write_text("")
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA gpubox\n")
    return Prober(tmp_path)


def verdict(tmp_path, monkeypatch, rc, stdout, spec=SPEC):
    calls = []

    def fake(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, rc, stdout=stdout, stderr="")

    monkeypatch.setattr(gate.subprocess, "run", fake)
    return prober(tmp_path).gate(("10.0.0.1", 22), spec), calls


@pytest.mark.parametrize(
    "rc,stdout,expect",
    [
        (0, 'noise\n{"ok": true, "cpus": 32}\n', (True, None)),
        (0, '{"ok": false, "reason": "wrong_hardware", "detail": "GPU 0 is RTX 4090"}', (False, "wrong_hardware")),
        (0, '{"ok": false, "reason": "busy_gpu", "detail": "util 100%"}', (False, "busy_gpu")),
        (0, '{"ok": false, "reason": "runtime_error", "detail": "ImportError: libcuda"}', (False, "runtime_error")),
        (255, "", (False, "unreachable")),  # ssh itself failed
        # Output we cannot read fails the box and blames nobody.
        (0, '{"ok": false, "reason": "gremlins"}', (False, None)),
        (0, '{"ok": false, "reason": ["busy_gpu"]}', (False, None)),
        (0, '{"ok": false, "reason": {"a": 1}}', (False, None)),
        (0, '{"ok": false}', (False, None)),
        (0, "[1, 2]", (False, None)),
        (0, '{"ok": "yes"}', (False, None)),
        (0, '{"ok": 1}', (False, None)),
        (0, "", (False, None)),
        (1, "Traceback (most recent call last): ...", (False, None)),
        # A success is believed only from a process that exited cleanly.
        (1, '{"ok": true}', (False, None)),
        (137, '{"ok": true}', (False, None)),
    ],
)  # fmt: skip
def test_gate_verdicts(tmp_path, monkeypatch, rc, stdout, expect):
    (ok, reason, _, _), _ = verdict(tmp_path, monkeypatch, rc, stdout)
    assert (ok, reason) == expect


@pytest.mark.parametrize(
    "stdout",
    [
        '{"ok": true, "gpus": null, "cpus": "many", "driver": 5, "torch": []}',
        '{"ok": true, "gpus": "RTX 5090"}',
        '{"ok": true, "gpus": [1, null, "RTX 5090"]}',
        '{"ok": true}',
    ],
)
def test_the_report_handed_on_always_has_the_same_shape(tmp_path, monkeypatch, stdout):
    (ok, _, _, report), _ = verdict(tmp_path, monkeypatch, 0, stdout)
    assert ok and set(report) == {"gpus", "cpus", "driver", "torch"}
    assert isinstance(report["gpus"], list) and all(isinstance(g, str) for g in report["gpus"])
    assert report["cpus"] is None and report["driver"] is None and report["torch"] is None


def test_a_hung_gate_is_a_soft_failure(tmp_path, monkeypatch):
    def hang(*a, **k):
        raise subprocess.TimeoutExpired(a[0], 240)

    monkeypatch.setattr(gate.subprocess, "run", hang)
    assert prober(tmp_path).gate(("10.0.0.1", 22), SPEC)[:2] == (False, "unreachable")


def test_the_gate_is_asked_exactly_what_the_reference_launcher_asked(tmp_path, monkeypatch):
    spec = Spec(
        image="img", gpu="RTX 5090", num_gpus=2, max_dph_per_gpu=1.25, capability=(12, 0),
        min_vram_mib=30000, must_reach=("https://github.com/", "https://pypi.org/simple/pip/"),
    )  # fmt: skip
    _, calls = verdict(tmp_path, monkeypatch, 0, '{"ok": true}', spec)
    argv, kw = calls[0]
    prefix = REFERENCE["gate"]["remote_prefix"]
    assert argv[-1].startswith(prefix)
    (quoted,) = shlex.split(argv[-1][len(prefix) :])  # what the remote shell will pass on
    sent = json.loads(quoted)
    assert sent == REFERENCE["gate"]["want"]
    assert kw["input"] == gate.GATE_SCRIPT


def test_gate_script_is_valid_python_and_names_only_known_reasons():
    compile(gate.GATE_SCRIPT, "gate", "exec")
    named = set(re.findall(r'done\("([a-z_]+)"', gate.GATE_SCRIPT))
    assert named and named <= set(policy.REASONS)


def run_script(tmp_path, body):
    """Run the gate's own verdict machinery around a stand-in for its checks."""
    head = gate.GATE_SCRIPT[: gate.GATE_SCRIPT.index("def smi():")]
    tail = gate.GATE_SCRIPT[gate.GATE_SCRIPT.index("# From here on") :]
    tail = tail[: tail.index("    with warnings")] + body + tail[tail.index("except Exception") :]
    path = tmp_path / "gate.py"
    path.write_text(head + tail)
    r = subprocess.run(["python3", str(path), "{}"], capture_output=True, text=True)
    return r.returncode, json.loads(r.stdout.strip().splitlines()[-1])


def test_an_unforeseen_failure_on_the_box_becomes_a_verdict_not_a_traceback(tmp_path):
    rc, out = run_script(tmp_path, "    import module_that_does_not_exist\n")
    assert rc == 0 and out["ok"] is False and out["reason"] == "runtime_error"
    assert "ModuleNotFoundError" in out["detail"]
    rc, out = run_script(tmp_path, '    done("cuda_init", "no CUDA")\n')
    assert (rc, out["reason"]) == (0, "cuda_init")  # done() is not swallowed
    rc, out = run_script(tmp_path, "    pass\n")
    assert rc == 0 and out["ok"] is True


def test_ssh_ignores_ambient_configuration(tmp_path):
    argv = prober(tmp_path).ssh(("10.0.0.1", 2222), "true")
    assert argv[:3] == ["ssh", "-F", "/dev/null"]
    joined = " ".join(argv)
    for opt in ("IdentitiesOnly=yes", "IdentityAgent=none", "ForwardAgent=no", "BatchMode=yes"):
        assert opt in joined
    assert f"UserKnownHostsFile={tmp_path}/known_hosts" in joined
    assert argv[-2:] == ["root@10.0.0.1", "true"]


def test_each_key_directory_gets_its_own_key(tmp_path):
    a, b = Prober(tmp_path / "run-a"), Prober(tmp_path / "run-b")
    assert a.pubkey != b.pubkey and a.known_hosts != b.known_hosts
    assert Prober(tmp_path / "run-a").pubkey == a.pubkey  # stable within a run
