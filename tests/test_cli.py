"""The commands end to end, against the fake provider and a fake ssh."""

import json
import subprocess
import time
from pathlib import Path

import pytest
from fakes import FAST, FakeProber, FakeVast, foreign_untouched

from gpubox import cli
from gpubox import policy as pol
from gpubox import vast as v
from gpubox.journal import Journal, state_lock

REFERENCE = json.loads((Path(__file__).parent / "fixtures/reference_requests.json").read_text())
UP = [
    "up", "--gpu", "RTX 5090", "--max-dph-per-gpu", "1.25", "--image", "img",
    "--namespace", "quack-ci", "--name", "77-1",
]  # fmt: skip
QUACK_CI = [
    "up", "--gpu", "RTX 5090", "--num-gpus", "2", "--max-dph-per-gpu", "1.25",
    "--capability", "12.0", "--min-vram-mib", "30000", "--min-cuda", "13.2", "--ttl-min", "60",
    "--must-reach", "https://github.com/", "--must-reach", "https://pypi.org/simple/pip/",
    "--image", "tridao/quack-kernels:cu13.2-26.11.22",
    "--namespace", "quack-ci", "--name", "77-1",
]  # fmt: skip


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Wire the CLI to one fake provider; returns (vast, run) where run(argv) -> exit code."""
    vast = FakeVast({1: "ok", 2: "ok"})
    vast.key_dirs, vast.probers = [], []

    def prober(key_dir):
        key_dir.mkdir(parents=True, exist_ok=True)
        (key_dir / "id_ed25519").touch()
        vast.key_dirs.append(key_dir)
        vast.probers.append(FakeProber(vast))
        return vast.probers[-1]

    monkeypatch.setenv("VASTAI_API_KEY", "k")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(cli.v, "Vast", lambda key: vast)
    monkeypatch.setattr(cli, "Prober", prober)
    monkeypatch.setattr(cli, "Policy", lambda **kw: type(FAST)(**{**FAST.__dict__, **kw}))

    def run(argv):
        state = ["--state-dir", str(tmp_path)] if argv[0] != "reap" else []
        return cli.main([*argv, *state])

    return vast, run


def fake_ssh(monkeypatch, rc=0, timeout=False, before=None):
    """Replace ssh. `before()` runs when the connection would be made."""
    calls = []

    class FakeSsh:
        def __init__(self, key_dir):
            self.key_dir = key_dir

        def ssh(self, ep, remote):
            return ["ssh", str(self.key_dir), f"{ep[0]}:{ep[1]}", remote]

    def fake_run(argv, input=None, timeout=None):
        calls.append((argv, input))
        if before:
            before()
        if fake_run.timeout:
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, fake_run.rc)

    fake_run.rc, fake_run.timeout = rc, timeout
    monkeypatch.setattr(cli, "Prober", FakeSsh)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    return calls


def script(tmp_path):
    (tmp_path / "s.sh").write_text("true\n")
    return ["run", "--script", str(tmp_path / "s.sh")]


def observations(tmp_path):
    rows = pol.Observations(tmp_path / "observations.jsonl").rows
    return [(o["kind"], o["reason_code"]) for o in rows]


# ------------------------------------------------------------------- up/down


def test_up_run_down(world, tmp_path, monkeypatch, capsys):
    vast, run = world
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    assert run(UP) == 0
    assert len(vast.ours()) == 1 and "READY" in capsys.readouterr().out
    run_id = Journal(tmp_path).state().run_id
    assert f"run_id={run_id}" in out.read_text()
    assert "observations_changed=true" in out.read_text()
    assert "| healthy |" in (tmp_path / "summary.md").read_text()
    box = vast.live[vast.ours()[0]]
    assert box["label"].startswith(run_id + ".a") and run_id.startswith("quack-ci.77-1.")
    assert vast.key_dirs == [tmp_path / "keys" / run_id]

    (tmp_path / "s.sh").write_text("echo hi\n")
    calls = fake_ssh(monkeypatch, rc=7)
    argv = ["run", "--script", str(tmp_path / "s.sh"), "--env", "GREETING=it's $HOME; x"]
    assert run(argv) == 7
    ssh_argv, sent = calls[0]
    assert sent.decode() == "export GREETING='it'\"'\"'s $HOME; x'\necho hi\n"
    assert ssh_argv[1] == str(tmp_path / "keys" / run_id)  # this run's key, no other
    assert "mktemp /tmp/gpubox-script." in ssh_argv[-1] and "</dev/null" in ssh_argv[-1]

    assert run(["down"]) == 0
    assert vast.ours() == [] and foreign_untouched(vast)
    assert not (tmp_path / "keys" / run_id).exists()  # nothing of this run is left to log in to
    assert run(["down"]) == 0  # idempotent


def test_the_quack_ci_flags_send_what_the_reference_launcher_sent(world):
    """Search, create and gate input for the real CI flag set, against the pinned recording."""
    vast, run = world
    vast.behaviour.update({1: "ok", 2: "ok"})
    FakeProber.pubkey, saved = "KEY", FakeProber.pubkey
    try:
        assert run(QUACK_CI) == 0
    finally:
        FakeProber.pubkey = saved
    assert vast.queries == [REFERENCE["search"]["body"]]
    for payload in vast.payloads:
        assert {**payload, "label": "LABEL"} == REFERENCE["create"]["body"]
    want = REFERENCE["gate"]["want"]
    for spec in vast.probers[0].wants:
        asked = (
            spec.num_gpus,
            spec.gpu,
            spec.min_vram_mib,
            list(spec.capability),
            list(spec.must_reach),
        )
        assert asked == tuple(
            want[k] for k in ("num_gpus", "gpu", "min_vram_mib", "capability", "must_reach")
        )
    assert vast.probers[0].wants and run(["down"]) == 0


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
    assert len(set(vast.key_dirs)) == 2  # and the next run has a key of its own


def test_up_refuses_a_damaged_journal(world, tmp_path, capsys):
    vast, run = world
    (tmp_path / "journal.jsonl").write_text(
        '{"ev": "run", "run_id": "a.b.0123abcd", "namespace": "a"}\ngarbage\n{"ev": "closed"}\n'
    )
    assert run(UP) == cli.EXIT_CONFIG
    assert "check the provider for leftover boxes" in capsys.readouterr().out
    assert vast.creates == 0


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


def test_nothing_after_the_decision_can_fail_the_launch(world, monkeypatch, tmp_path):
    vast, run = world
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "no" / "such" / "dir" / "out"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "no" / "such" / "dir" / "s.md"))

    def boom(*a, **k):
        raise TypeError("len() of NoneType")

    monkeypatch.setattr(cli, "observe", boom)
    monkeypatch.setattr(cli, "report_result", boom)
    assert run(UP) == 0 and len(vast.ours()) == 1
    assert run(["down"]) == 0


def test_a_report_of_any_shape_can_be_printed(world, monkeypatch, capsys):
    vast, run = world
    monkeypatch.setattr(FakeProber, "gate", lambda self, ep, spec: (True, None, "", {}))
    assert run(UP) == 0 and "READY" in capsys.readouterr().out
    assert run(["down"]) == 0


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


def test_an_unseen_create_keeps_down_failing_until_the_operator_decides(
    world, tmp_path, monkeypatch
):
    vast, run = world
    vast.behaviour.update({1: "ghost", 2: "ok"})
    assert run(UP) == 0  # a healthy winner; the ghost is a warning
    assert run(["down"]) == cli.EXIT_INFRA and vast.ours() == []
    assert run(["down", "--abandon-unseen"]) == cli.EXIT_INFRA  # not old enough yet
    assert (tmp_path / "keys").exists()  # the run is not over
    monkeypatch.setattr(
        cli, "Policy", lambda **kw: type(FAST)(**{**FAST.__dict__, "settle": 0, **kw})
    )
    time.sleep(0.01)
    assert run(["down", "--abandon-unseen"]) == 0
    assert Journal(tmp_path).state().closed


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


def test_a_machine_that_failed_is_skipped_by_the_next_up(world, tmp_path):
    vast, run = world
    vast.behaviour.update({1: "bad_gpu", 2: "ok", 3: "ok"})
    vast._offers = [*vast._offers, {**vast._offers[1], "id": 1003, "machine_id": 3, "host_id": 3}]
    assert run([*UP, "--spare", "0"]) == 0
    assert run(["down"]) == 0
    first = vast.creates
    assert run([*UP, "--spare", "0"]) == 0
    # Machine 1 is blocked now, so the second launch goes straight to machine 2.
    assert vast.creates == first + 1


# ----------------------------------------------------------------------- run


def test_run_without_a_box_is_a_configuration_error(world, tmp_path):
    _, run = world
    assert run(script(tmp_path)) == cli.EXIT_CONFIG


def test_run_after_down_has_no_box_and_no_key(world, tmp_path):
    _, run = world
    assert run(UP) == 0 and run(["down"]) == 0
    assert run(script(tmp_path)) == cli.EXIT_CONFIG


def test_run_is_refused_once_the_box_is_destroyed_but_the_run_is_not_over(
    world, tmp_path, monkeypatch
):
    # `down` destroyed the winner but could not finish (an unseen create is
    # unresolved), so the run's key is still there. The box is not.
    vast, run = world
    vast.behaviour.update({1: "ghost", 2: "ok"})
    assert run(UP) == 0 and run(["down"]) == cli.EXIT_INFRA
    assert (tmp_path / "keys").exists()
    calls = fake_ssh(monkeypatch)
    assert run(script(tmp_path)) == cli.EXIT_CONFIG and not calls


def test_run_rejects_malformed_env(world, tmp_path):
    _, run = world
    assert run(UP) == 0
    for bad in ("NOVALUE", "1X=2", "A B=1"):
        assert run([*script(tmp_path), "--env", bad]) == cli.EXIT_CONFIG


def test_run_timeout(world, tmp_path, monkeypatch):
    _, run = world
    assert run(UP) == 0
    fake_ssh(monkeypatch, timeout=True)
    assert run(script(tmp_path)) == cli.EXIT_TIMEOUT


def test_box_that_dies_mid_run_is_recorded_once_the_provider_confirms(world, tmp_path, monkeypatch):
    vast, run = world
    assert run(UP) == 0
    fake_ssh(monkeypatch, rc=255)
    vast.live.pop(vast.ours()[0])  # the host pulled it
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None), ("died_after_admission", "died")]
    died = pol.Observations(tmp_path / "observations.jsonl").rows[-1]
    assert died["gpu"] == "RTX 5090" and died["gate_version"] == 1


def test_lost_ssh_to_a_running_box_blames_nobody(world, tmp_path, monkeypatch):
    _, run = world
    assert run(UP) == 0
    fake_ssh(monkeypatch, rc=255)
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]


def test_lost_ssh_with_the_provider_unreachable_blames_nobody(world, tmp_path, monkeypatch):
    vast, run = world
    assert run(UP) == 0
    fake_ssh(monkeypatch, rc=255)

    def outage(iid):
        raise v.VastError(503, "unavailable")

    monkeypatch.setattr(vast, "instance", outage)
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]


@pytest.mark.parametrize(
    "status,blamed",
    [("exited", True), ("offline", True), ("loading", False), (None, False), (["exited"], False)],
)
def test_only_a_terminal_provider_status_counts_as_died(
    world, tmp_path, monkeypatch, status, blamed
):
    vast, run = world
    assert run(UP) == 0
    fake_ssh(monkeypatch, rc=255)
    monkeypatch.setattr(vast, "instance", lambda iid: {"id": iid, "actual_status": status})
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert (("died_after_admission", "died") in observations(tmp_path)) is blamed


def test_our_own_down_during_run_blames_nobody(world, tmp_path, monkeypatch):
    vast, run = world
    assert run(UP) == 0
    iid = vast.ours()[0]

    def torn_down_meanwhile():
        Journal(tmp_path).append("destroying", instance_id=iid)  # what `down` writes first
        vast.live.pop(iid)

    fake_ssh(monkeypatch, rc=255, before=torn_down_meanwhile)
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]


def test_down_that_starts_after_the_connection_dropped_is_still_not_blamed(
    world, tmp_path, monkeypatch
):
    # SSH fails for its own reasons; before we ask the provider, a teardown
    # removes the box. "Gone" is then our doing, not the machine's.
    vast, run = world
    assert run(UP) == 0
    iid = vast.ours()[0]
    fake_ssh(monkeypatch, rc=255)

    def gone_because_of_us(instance_id):
        Journal(tmp_path).append("destroying", instance_id=iid)
        vast.live.pop(iid, None)
        return None

    monkeypatch.setattr(vast, "instance", gone_because_of_us)
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]


# ---------------------------------------------------------------------- reap


def old_box(vast, label="quack-ci.old.0123abcd.a1"):
    vast.live[50] = {"id": 50, "label": label, "machine": 1, "start_date": time.time() - 9e4}


def test_reap_needs_explicit_permission_to_destroy(world, capsys):
    vast, run = world
    old_box(vast)
    reap = ["reap", "--namespace", "quack-ci", "--max-age-min", "90"]
    assert run(reap) == 0
    assert 50 in vast.live and "report only" in capsys.readouterr().out
    assert run([*reap, "--destroy", "--dry-run"]) == 0 and 50 in vast.live
    assert run([*reap, "--destroy"]) == 0
    assert 50 not in vast.live and foreign_untouched(vast)


def test_reap_reports_an_unconfirmed_delete_as_infrastructure(world):
    vast, run = world
    old_box(vast)
    vast.sticky.add(1)
    reap = ["reap", "--namespace", "quack-ci", "--max-age-min", "90", "--destroy"]
    assert run(reap) == cli.EXIT_INFRA and 50 in vast.live


def test_a_bad_legacy_pattern_is_rejected_before_anything_is_touched(world, monkeypatch):
    vast, run = world
    old_box(vast)
    listed = []
    monkeypatch.setattr(vast, "instances", lambda: listed.append(1) or [])
    reap = ["reap", "--namespace", "quack-ci", "--max-age-min", "90", "--destroy"]
    assert run([*reap, "--legacy-pattern", "quack-ci-(\\d+"]) == cli.EXIT_CONFIG
    assert not listed and 50 in vast.live


# ----------------------------------------------------------------- arguments


@pytest.mark.parametrize(
    "bad",
    [
        ["--spare", "-1"], ["--max-attempts", "0"], ["--num-gpus", "0"], ["--ttl-min", "0"],
        ["--max-dph-per-gpu", "0"], ["--max-dph-per-gpu", "nan"], ["--overall-timeout", "inf"],
        ["--min-vram-mib", "-5"],
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
