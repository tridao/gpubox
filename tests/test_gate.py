"""The SSH wrapper and the reading of the health gate's verdict."""

import json
import re
import shlex
import subprocess
import sys
import time
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
    """What the gate's verdict is when its process ends this way."""
    return gate.read_gate(gate.Done(rc, stdout, "")), None


def finish(check, within=10):
    """Poll a check until it is done, as the launch loop does."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        result = check.poll()
        if result is not None:
            return result
        time.sleep(0.01)
    raise AssertionError("the child process did not finish")


def python(code, **kw):
    return gate.Child([sys.executable, "-c", code], **{"timeout": 10, **kw})


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


def test_a_hung_gate_is_a_soft_failure():
    killed_at_its_deadline = gate.Done(None, "", "")
    assert gate.read_gate(killed_at_its_deadline)[:2] == (False, "unreachable")
    assert gate.read_ready(killed_at_its_deadline) == (False, "ssh timed out")


def test_the_gate_is_asked_exactly_what_the_reference_launcher_asked(tmp_path, monkeypatch):
    spec = Spec(
        image="img", gpu="RTX 5090", num_gpus=2, max_dph_per_gpu=1.25, capability=(12, 0),
        min_vram_mib=30000, must_reach=("https://github.com/", "https://pypi.org/simple/pip/"),
    )  # fmt: skip
    calls = []

    class Recorded:
        def __init__(self, argv, **kw):
            calls.append((argv, kw))

    monkeypatch.setattr(gate, "Child", Recorded)
    prober(tmp_path).start_gate(("10.0.0.1", 22), spec)
    argv, kw = calls[0]
    prefix = REFERENCE["gate"]["remote_prefix"]
    assert argv[-1].startswith(prefix)
    (quoted,) = shlex.split(argv[-1][len(prefix) :])  # what the remote shell will pass on
    sent = json.loads(quoted)
    assert sent == REFERENCE["gate"]["want"]
    assert kw == {"stdin": gate.GATE_SCRIPT, "timeout": gate.GATE_TIMEOUT}


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


# ------------------------------------------------------ whatever the box prints


def test_whatever_bytes_the_box_prints_are_read_without_raising(tmp_path, monkeypatch):
    # For real: a process that emits bytes that are not UTF-8.
    p = prober(tmp_path)
    monkeypatch.setattr(
        p,
        "ssh",
        lambda ep, remote, **k: [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'\\xff\\xfe junk'); sys.stderr.buffer.write(b'\\xff')",
        ],
    )
    ok, reason, detail, report = finish(p.start_gate(("10.0.0.1", 22), SPEC))
    assert (ok, reason) == (False, None) and set(report) == {"gpus", "cpus", "driver", "torch"}
    assert finish(p.start_ready(("10.0.0.1", 22))) == (True, "\ufffd")


# --------------------------------------------------------- child processes


def test_a_child_is_polled_and_gets_its_input_from_a_file():
    child = python("import sys; print(sys.stdin.read().upper()); sys.exit(3)", stdin="hello")
    done = finish(child)
    assert (done.rc, done.stdout.strip(), done.timed_out) == (3, "HELLO", False)
    assert child.poll() is done  # asking again changes nothing


def test_a_child_that_prints_a_lot_never_blocks_on_a_pipe():
    # Far more than a pipe holds. With pipes that nobody drains, this would hang.
    done = finish(
        python("import sys; sys.stdout.write('x' * 5_000_000); sys.stderr.write('e' * 5_000_000)")
    )
    assert done.rc == 0 and len(done.stdout) == 1 << 20  # and what we keep of it is bounded


def test_what_is_kept_of_a_noisy_child_is_its_end_where_the_verdict_is():
    code = (
        "import sys; sys.stdout.write('noise\\n' * 500_000); print('{\"ok\": true, \"cpus\": 8}')"
    )
    ok, reason, _, report = finish(gate.Check(python(code), gate.read_gate))
    assert (ok, reason, report["cpus"]) == (True, None, 8)


def test_a_signal_cannot_come_between_starting_a_child_and_remembering_it(tmp_path, monkeypatch):
    import os
    import signal

    class Started:
        def __init__(self, argv, **kw):
            os.kill(os.getpid(), signal.SIGINT)  # as if Ctrl-C arrived right after the fork
            time.sleep(0.05)
            self.killed = False

        def kill(self):
            self.killed = True

    monkeypatch.setattr(gate, "Child", Started)
    p = prober(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        p.start_ready(("10.0.0.1", 22))
    assert len(p.checks) == 1  # the signal arrived after the child was on the list
    p.kill_all()
    assert p.checks[0].child.killed


def test_a_child_is_killed_at_its_deadline_and_that_is_its_result():
    child = python("import time; print('started', flush=True); time.sleep(60)", timeout=0.3)
    t0 = time.monotonic()
    done = finish(child)
    assert done.timed_out and done.rc is None and time.monotonic() - t0 < 5
    assert child.proc.poll() is not None  # collected, not left as a zombie


def test_killing_a_child_stops_what_it_started_and_is_safe_to_repeat(tmp_path):
    started, finished = tmp_path / "grandchild-started", tmp_path / "grandchild-finished"
    grandchild = tmp_path / "grandchild.py"
    grandchild.write_text(
        f"import time\nopen({str(started)!r}, 'w').close()\ntime.sleep(1)\n"
        f"open({str(finished)!r}, 'w').close()\n"
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, {str(grandchild)!r}])\ntime.sleep(60)\n"
    )
    child = gate.Child([sys.executable, str(parent)], timeout=10)
    deadline = time.monotonic() + 10
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started.exists()  # there is something to stop
    child.kill()
    child.kill()
    assert child.poll().timed_out and child.proc.poll() is not None
    time.sleep(1.3)
    assert not finished.exists()  # its whole session went with it
    ended = python("pass")
    finish(ended)
    ended.kill()
    assert ended.poll().rc == 0  # killing something that already ended does not rewrite its result


def test_a_check_reads_the_result_of_its_child():
    check = gate.Check(python('print(\'{"ok": true, "cpus": 4}\')'), gate.read_gate)
    ok, reason, _, report = finish(check)
    assert (ok, reason, report["cpus"]) == (True, None, 4)


@pytest.mark.parametrize(
    "rc,stdout", [(255, ""), (1, "Traceback"), (0, "[1]"), (0, '{"ok": false, "reason": 5}')]
)
def test_every_outcome_carries_a_report_of_the_same_shape(tmp_path, monkeypatch, rc, stdout):
    (_, _, _, report), _ = verdict(tmp_path, monkeypatch, rc, stdout)
    assert report == {"gpus": [], "cpus": None, "driver": None, "torch": None}


def test_loading_a_key_never_creates_one(tmp_path):
    with pytest.raises(FileNotFoundError):
        Prober(tmp_path / "gone", create=False)
    assert not (tmp_path / "gone").exists()
    made = Prober(tmp_path / "run-a")
    assert Prober(tmp_path / "run-a", create=False).pubkey == made.pubkey
