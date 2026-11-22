"""The commands end to end, against the fake provider and a fake ssh."""

import subprocess

import pytest
from fakes import FAST, FakeProber, FakeVast, foreign_untouched

from gpubox import cli
from gpubox import policy as pol
from gpubox import vast as v
from gpubox.journal import Journal, state_lock

UP = [
    "up", "--gpu", "RTX 5090", "--max-dph-per-gpu", "1.25", "--image", "img",
    "--namespace", "quack-ci", "--name", "77-1",
]  # fmt: skip


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Wire the CLI to one fake provider; returns (vast, run) where run(argv) -> exit code."""
    vast = FakeVast({1: "ok", 2: "ok"})
    monkeypatch.setenv("VASTAI_API_KEY", "k")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(cli.v, "Vast", lambda key: vast)
    monkeypatch.setattr(cli, "Prober", lambda state_dir, fresh=False: FakeProber(vast))
    monkeypatch.setattr(cli, "Policy", lambda **kw: type(FAST)(**{**FAST.__dict__, **kw}))

    def run(argv):
        state = ["--state-dir", str(tmp_path)] if argv[0] != "reap" else []
        return cli.main([*argv, *state])

    return vast, run


def fake_ssh(monkeypatch, rc=0, timeout=False):
    calls = []

    class FakeSsh:
        def __init__(self, state_dir, fresh=False):
            pass

        def ssh(self, ep, remote):
            return ["ssh", f"{ep[0]}:{ep[1]}", remote]

    def fake_run(argv, input=None, timeout=None):
        calls.append((argv, input))
        if fake_run.timeout:
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, fake_run.rc)

    fake_run.rc, fake_run.timeout = rc, timeout
    monkeypatch.setattr(cli, "Prober", FakeSsh)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    return calls, fake_run


def test_up_run_down(world, tmp_path, monkeypatch, capsys):
    vast, run = world
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    assert run(UP) == 0
    assert len(vast.ours()) == 1 and "READY" in capsys.readouterr().out
    assert "observations_changed=true" in out.read_text()
    assert "| healthy |" in (tmp_path / "summary.md").read_text()
    box = vast.live[vast.ours()[0]]
    assert box["label"].startswith("quack-ci.77-1.") and box["label"][-3:] in (".a1", ".a2")

    script = tmp_path / "s.sh"
    script.write_text("echo hi\n")
    calls, ssh = fake_ssh(monkeypatch, rc=7)
    assert run(["run", "--script", str(script), "--env", "GREETING=it's $HOME; x"]) == 7
    argv, sent = calls[0]
    assert sent.decode() == "export GREETING='it'\"'\"'s $HOME; x'\necho hi\n"
    assert "mktemp /tmp/gpubox-script." in argv[-1] and "</dev/null" in argv[-1]

    assert run(["down"]) == 0
    assert vast.ours() == [] and foreign_untouched(vast)
    assert run(["down"]) == 0  # idempotent


def test_up_without_a_healthy_box_is_an_infrastructure_failure(world):
    vast, run = world
    vast.behaviour.update({1: "bad_gpu", 2: "bad_gpu"})
    assert run(UP) == cli.EXIT_INFRA
    assert vast.ours() == []
    assert run(["down"]) == 0


def test_up_refuses_a_state_directory_whose_run_still_has_boxes(world, capsys):
    vast, run = world
    assert run(UP) == 0
    creates = vast.creates
    assert run(UP) == cli.EXIT_CONFIG
    assert "gpubox down" in capsys.readouterr().out and vast.creates == creates
    assert run(["down"]) == 0
    assert run(UP) == 0  # a closed run does not block the next one
    assert len(vast.ours()) == 1


def test_bad_image_is_a_configuration_error(world):
    vast, run = world
    vast.behaviour.update({1: "image", 2: "image"})
    assert run(UP) == cli.EXIT_CONFIG
    assert vast.ours() == []


def test_missing_key_is_a_configuration_error(world, monkeypatch, tmp_path):
    vast, _ = world
    monkeypatch.delenv("VASTAI_API_KEY")
    argv = ["--keys-file", str(tmp_path / "none"), *UP, "--state-dir", str(tmp_path)]
    assert cli.main(argv) == cli.EXIT_CONFIG
    assert vast.creates == 0


def test_run_without_a_box_is_a_configuration_error(world, tmp_path):
    _, run = world
    (tmp_path / "s.sh").write_text("true\n")
    assert run(["run", "--script", str(tmp_path / "s.sh")]) == cli.EXIT_CONFIG


def test_run_rejects_malformed_env(world, tmp_path):
    _, run = world
    assert run(UP) == 0
    (tmp_path / "s.sh").write_text("true\n")
    for bad in ("NOVALUE", "1X=2", "A B=1"):
        assert run(["run", "--script", str(tmp_path / "s.sh"), "--env", bad]) == cli.EXIT_CONFIG


def test_run_timeout(world, tmp_path, monkeypatch):
    _, run = world
    assert run(UP) == 0
    (tmp_path / "s.sh").write_text("true\n")
    fake_ssh(monkeypatch, timeout=True)
    assert run(["run", "--script", str(tmp_path / "s.sh")]) == cli.EXIT_TIMEOUT


def observations(tmp_path):
    rows = pol.Observations(tmp_path / "observations.jsonl").rows
    return [(o["kind"], o["reason_code"]) for o in rows]


def test_box_that_dies_mid_run_is_recorded_once_the_provider_confirms(world, tmp_path, monkeypatch):
    vast, run = world
    assert run(UP) == 0
    (tmp_path / "s.sh").write_text("true\n")
    fake_ssh(monkeypatch, rc=255)
    vast.live.pop(vast.ours()[0])  # the host pulled it
    assert run(["run", "--script", str(tmp_path / "s.sh")]) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None), ("died_after_admission", "died")]
    died = pol.Observations(tmp_path / "observations.jsonl").rows[-1]
    assert died["gpu"] == "RTX 5090" and died["gate_version"] == 1


def test_lost_ssh_to_a_running_box_blames_nobody(world, tmp_path, monkeypatch):
    _, run = world
    assert run(UP) == 0
    (tmp_path / "s.sh").write_text("true\n")
    fake_ssh(monkeypatch, rc=255)
    assert run(["run", "--script", str(tmp_path / "s.sh")]) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]


def test_lost_ssh_with_the_provider_unreachable_blames_nobody(world, tmp_path, monkeypatch):
    vast, run = world
    assert run(UP) == 0
    (tmp_path / "s.sh").write_text("true\n")
    fake_ssh(monkeypatch, rc=255)

    def outage(iid):
        raise v.VastError(503, "unavailable")

    monkeypatch.setattr(vast, "instance", outage)
    assert run(["run", "--script", str(tmp_path / "s.sh")]) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]


def test_our_own_down_during_run_blames_nobody(world, tmp_path, monkeypatch):
    vast, run = world
    assert run(UP) == 0
    (tmp_path / "s.sh").write_text("true\n")
    iid = vast.ours()[0]
    _, ssh = fake_ssh(monkeypatch, rc=255)
    real = cli.subprocess.run

    def torn_down_meanwhile(argv, **kw):
        Journal(tmp_path).append("destroying", instance_id=iid)  # what `down` writes first
        vast.live.pop(iid)
        return real(argv, **kw)

    monkeypatch.setattr(cli.subprocess, "run", torn_down_meanwhile)
    assert run(["run", "--script", str(tmp_path / "s.sh")]) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]


def test_down_that_starts_after_the_connection_dropped_is_still_not_blamed(
    world, tmp_path, monkeypatch
):
    # SSH fails for its own reasons; before we ask the provider, a teardown
    # removes the box. "Gone" is then our doing, not the machine's.
    vast, run = world
    assert run(UP) == 0
    (tmp_path / "s.sh").write_text("true\n")
    iid = vast.ours()[0]
    fake_ssh(monkeypatch, rc=255)

    def gone_because_of_us(instance_id):
        Journal(tmp_path).append("destroying", instance_id=iid)
        vast.live.pop(iid, None)
        return None

    monkeypatch.setattr(vast, "instance", gone_because_of_us)
    assert run(["run", "--script", str(tmp_path / "s.sh")]) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]


@pytest.mark.parametrize(
    "status,blamed", [("exited", True), ("offline", True), ("loading", False), (None, False)]
)
def test_only_a_terminal_provider_status_counts_as_died(
    world, tmp_path, monkeypatch, status, blamed
):
    vast, run = world
    assert run(UP) == 0
    (tmp_path / "s.sh").write_text("true\n")
    fake_ssh(monkeypatch, rc=255)
    monkeypatch.setattr(vast, "instance", lambda iid: {"id": iid, "actual_status": status})
    assert run(["run", "--script", str(tmp_path / "s.sh")]) == cli.EXIT_INFRA
    assert (("died_after_admission", "died") in observations(tmp_path)) is blamed


def test_run_after_down_has_no_box(world, tmp_path):
    _, run = world
    assert run(UP) == 0 and run(["down"]) == 0
    (tmp_path / "s.sh").write_text("true\n")
    assert run(["run", "--script", str(tmp_path / "s.sh")]) == cli.EXIT_CONFIG


def test_a_late_teardown_of_an_earlier_run_does_not_touch_the_next_one(world, tmp_path):
    vast, run = world
    assert run(UP) == 0
    first = Journal(tmp_path).state().run_id
    assert run(["down", "--run-id", first]) == 0
    assert run(UP) == 0
    second_box = vast.ours()
    assert run(["down", "--run-id", first]) == 0  # the retried teardown of the first run
    assert vast.ours() == second_box
    assert run(["down", "--run-id", "quack-ci.nope.00000000"]) == cli.EXIT_CONFIG
    assert run(["down", "--run-id", "../../etc/passwd"]) == cli.EXIT_CONFIG
    assert vast.ours() == second_box
    assert run(["down"]) == 0 and vast.ours() == []


def test_a_second_command_on_a_busy_state_directory_is_refused(world, tmp_path):
    vast, run = world
    with state_lock(tmp_path):
        assert run(UP) == cli.EXIT_CONFIG
        assert run(["down"]) == cli.EXIT_CONFIG
    assert vast.creates == 0


def test_down_reports_an_unconfirmed_teardown_as_infrastructure(world):
    vast, run = world
    assert run(UP) == 0
    vast.sticky.add(vast.live[vast.ours()[0]]["machine"])
    assert run(["down"]) == cli.EXIT_INFRA
    vast.sticky.clear()
    assert run(["down"]) == 0 and vast.ours() == []


def test_failures_after_the_box_is_ours_do_not_fail_the_launch(world, monkeypatch, tmp_path):
    vast, run = world
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "no" / "such" / "dir" / "s.md"))
    monkeypatch.setattr(cli, "observe", lambda *a: (_ for _ in ()).throw(ValueError("boom")))
    assert run(UP) == 0 and len(vast.ours()) == 1
    assert run(["down"]) == 0


@pytest.mark.parametrize(
    "bad",
    [
        ["--spare", "-1"], ["--max-attempts", "0"], ["--num-gpus", "0"], ["--ttl-min", "0"],
        ["--max-dph-per-gpu", "0"], ["--max-dph-per-gpu", "nan"], ["--overall-timeout", "inf"],
    ],
)  # fmt: skip
def test_nonsense_arguments_are_rejected_before_anything_is_rented(world, bad):
    vast, run = world
    with pytest.raises(SystemExit) as exc:
        run([*UP, *bad])
    assert exc.value.code == 2 and vast.creates == 0


def test_bad_capability_is_a_configuration_error(world):
    vast, run = world
    assert run([*UP, "--capability", "twelve"]) == cli.EXIT_CONFIG and vast.creates == 0


def test_reap_rejects_a_negative_age(world):
    _, run = world
    with pytest.raises(SystemExit):
        run(["reap", "--namespace", "quack-ci", "--max-age-min", "-1", "--destroy"])


def test_a_machine_that_failed_is_skipped_by_the_next_up(world, tmp_path):
    vast, run = world
    vast.behaviour.update({1: "bad_gpu", 2: "ok", 3: "ok"})
    vast._offers = [
        vast._offers[0],
        vast._offers[1],
        {**vast._offers[1], "id": 1003, "machine_id": 3, "host_id": 3},
    ]
    assert run([*UP, "--spare", "0"]) == 0
    assert run(["down"]) == 0
    first = vast.creates
    assert run([*UP, "--spare", "0"]) == 0
    # Machine 1 is blocked now, so the second launch goes straight to machine 2.
    assert vast.creates == first + 1


def test_reap_needs_explicit_permission_to_destroy(world, capsys):
    import time

    vast, run = world
    vast.live[50] = {
        "id": 50,
        "label": "quack-ci.old.0123abcd.a1",
        "machine": 1,
        "start_date": time.time() - 9e4,
    }
    assert run(["reap", "--namespace", "quack-ci", "--max-age-min", "90"]) == 0
    assert 50 in vast.live and "report only" in capsys.readouterr().out
    both = ["reap", "--namespace", "quack-ci", "--max-age-min", "90", "--destroy", "--dry-run"]
    assert run(both) == 0 and 50 in vast.live
    assert run(["reap", "--namespace", "quack-ci", "--max-age-min", "90", "--destroy"]) == 0
    assert 50 not in vast.live and foreign_untouched(vast)
