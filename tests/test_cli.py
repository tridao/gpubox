"""The commands end to end, against the fake provider and a fake ssh."""

import json
import subprocess
import time
from pathlib import Path

import pytest
from fakes import EXPIRY, FAST, OTHER_RUN, FakeCheck, FakeProber, FakeVast, foreign_untouched

from gpubox import cli
from gpubox import policy as pol
from gpubox import vast as v
from gpubox.state import Snapshot, make_tag, parse_tag, state_lock

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

    def prober(key_dir, create=True):
        key_dir.mkdir(parents=True, exist_ok=True)
        (key_dir / "id_ed25519").touch()
        vast.key_dirs.append(key_dir)
        vast.probers.append(FakeProber(vast))
        return vast.probers[-1]

    monkeypatch.setenv("VASTAI_API_KEY", "k")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("GPUBOX_ACCOUNT_ID", raising=False)
    monkeypatch.setattr(cli.v, "Vast", lambda key: vast)
    monkeypatch.setattr(cli, "Prober", prober)
    monkeypatch.setattr(cli, "Policy", lambda **kw: type(FAST)(**{**FAST.__dict__, **kw}))

    def run(argv):
        state = ["--state-dir", str(tmp_path)] if argv[0] != "reap" else []
        return cli.main([*argv, *state])

    run.state_dir = tmp_path
    return vast, run


def fake_ssh(monkeypatch, rc=0, timeout=False, before=None):
    """Replace ssh. `before()` runs when the connection would be made."""
    calls = []

    class FakeSsh:
        def __init__(self, key_dir, create=True):
            if not create and not (key_dir / "id_ed25519").exists():
                raise FileNotFoundError(key_dir)
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
    run_id = Snapshot.load(tmp_path).run
    assert f"run_id={run_id}" in out.read_text()
    assert "observations_changed=true" in out.read_text()
    assert "| healthy |" in (tmp_path / "summary.md").read_text()
    box = vast.live[vast.ours()[0]]
    tag = parse_tag(box["label"])
    assert tag.run == run_id and run_id.startswith("quack-ci.77-1.")
    assert 89 * 60 < tag.expiry - time.time() <= 90 * 60  # the default --ttl-min
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
    from test_vast import same_but_for_the_timer

    assert vast.payloads
    for payload in vast.payloads:
        tag = parse_tag(payload["label"])
        assert 59 * 60 < tag.expiry - time.time() <= 60 * 60  # --ttl-min 60
        assert f"-ge {tag.expiry} ]" in payload["args_str"]  # the timer waits for the tag's expiry
        assert same_but_for_the_timer({**payload, "label": "LABEL"}, REFERENCE["create"]["body"])
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


def test_up_refuses_a_damaged_snapshot(world, tmp_path, capsys):
    vast, run = world
    (tmp_path / "run.json").write_text('{"version": 1, "run": "a.b.0123456789ab", "closed": tr')
    assert run(UP) == cli.EXIT_CONFIG
    assert "damaged" in capsys.readouterr().out
    assert vast.creates == 0


def test_the_run_id_is_published_before_anything_is_rented(world, tmp_path, monkeypatch):
    # The teardown step is given this output: it must exist whenever a box may.
    vast, run = world
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    seen = []
    vast.hooks["create"] = lambda machine: seen.append(out.read_text())
    assert run(UP) == 0
    assert seen and all("run_id=quack-ci.77-1." in text for text in seen)


def test_up_proves_it_can_list_before_it_rents(world):
    vast, run = world
    vast.list_fails = 1
    assert run(UP) == cli.EXIT_INFRA
    assert vast.creates == 0 and vast.names("offers", "create") == []


def test_a_ttl_that_leaves_no_lease_is_refused(world):
    vast, run = world
    assert run([*UP, "--ttl-min", "10", "--overall-timeout", "900"]) == cli.EXIT_CONFIG
    assert vast.creates == 0


# ------------------------------------------------------------ the account pin


def test_up_rents_nothing_with_a_key_of_another_account(world, monkeypatch, capsys):
    vast, run = world
    assert run([*UP, "--account-id", "999"]) == cli.EXIT_CONFIG
    assert "belongs to account 111" in capsys.readouterr().out
    assert vast.creates == 0 and vast.names("instances", "offers") == []
    monkeypatch.setenv("GPUBOX_ACCOUNT_ID", "999")
    assert run(UP) == cli.EXIT_CONFIG and vast.creates == 0
    monkeypatch.setenv("GPUBOX_ACCOUNT_ID", "111")
    assert run(UP) == 0
    assert Snapshot.load(run.state_dir).account == "111"


def test_down_with_a_key_of_another_account_is_an_error_not_a_clean_result(world, capsys):
    # T3: the other account's listing is empty of our boxes, which says nothing.
    vast, run = world
    assert run([*UP, "--account-id", "111"]) == 0
    vast.account_id = "222"
    deletes = len(vast.names("destroy"))
    assert run(["down"]) == cli.EXIT_CONFIG
    assert "another account" in capsys.readouterr().out
    assert len(vast.ours()) == 1 and len(vast.names("destroy")) == deletes
    assert not Snapshot.load(run.state_dir).closed
    vast.account_id = "111"
    assert run(["down"]) == 0 and vast.ours() == []


def test_a_run_remembers_its_account_even_when_none_was_pinned(world, capsys):
    # Otherwise a `down` with another account's key would find nothing there
    # and call the run clean.
    vast, run = world
    assert run(UP) == 0
    assert Snapshot.load(run.state_dir).account == "111"
    vast.account_id = "222"
    deletes = len(vast.names("destroy"))
    assert run(["down"]) == cli.EXIT_CONFIG and "another account" in capsys.readouterr().out
    assert len(vast.ours()) == 1 and len(vast.names("destroy")) == deletes
    vast.account_id = "111"
    assert run(["down"]) == 0


def test_the_reaper_does_nothing_on_another_account(world, capsys):
    vast, run = world
    vast.live[50] = {"id": 50, "label": None, "machine": 1, "start_date": time.time() - 9e4}
    reap = ["reap", "--exclusive", "--max-age-min", "90", "--destroy"]
    assert run([*reap, "--account-id", "999"]) == cli.EXIT_CONFIG  # R1
    assert vast.names("instances", "destroy") == [] and 50 in vast.live
    assert run(reap) == cli.EXIT_CONFIG  # an exclusive reap without a pinned account
    assert "--account-id" in capsys.readouterr().out and 50 in vast.live


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


def test_the_launch_does_not_start_if_the_run_id_cannot_be_published(world, monkeypatch, tmp_path):
    # CI's teardown step is `down --run-id <this output>`. Without the output
    # that step can do nothing, so nothing may be rented.
    vast, run = world
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "no" / "such" / "dir" / "out"))
    assert run(UP) == cli.EXIT_CONFIG
    assert vast.creates == 0 and vast.names("create") == []
    assert not (tmp_path / "run.json").exists()  # and the directory is free for the next run


def test_nothing_after_the_decision_can_fail_the_launch(world, monkeypatch, tmp_path):
    vast, run = world
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "no" / "such" / "dir" / "s.md"))
    # The run id is written; after that the output file stops being writable.
    broken = str(tmp_path / "no" / "such" / "dir" / "out")
    vast.hooks["create"] = lambda machine: monkeypatch.setenv("GITHUB_OUTPUT", broken)

    def boom(*a, **k):
        raise TypeError("len() of NoneType")

    monkeypatch.setattr(cli, "observe", boom)
    monkeypatch.setattr(cli, "report_result", boom)
    assert run(UP) == 0 and len(vast.ours()) == 1
    assert "run_id=" in out.read_text() and "observations_changed" not in out.read_text()
    assert run(["down"]) == 0


def test_a_report_of_any_shape_can_be_printed(world, monkeypatch, capsys):
    vast, run = world
    empty = lambda self, ep, spec: FakeCheck(lambda: (True, None, "", {}))
    monkeypatch.setattr(FakeProber, "start_gate", empty)
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


def test_an_unseen_create_keeps_down_failing_until_it_is_abandoned(world, tmp_path):
    vast, run = world
    vast.behaviour.update({1: "ghost", 2: "ok"})
    assert run(UP) == 0  # a healthy winner; the ghost is a warning
    key = tmp_path / "keys" / Snapshot.load(tmp_path).run
    assert run(["down"]) == cli.EXIT_INFRA and vast.ours() == []
    assert (key / "id_ed25519").exists()  # the run is not over: its key stays
    assert run(["down", "--abandon-unseen"]) == 0
    assert Snapshot.load(tmp_path).closed and not key.exists()


def test_a_late_teardown_of_an_earlier_run_does_not_touch_the_next_one(world, tmp_path):
    vast, run = world
    assert run(UP) == 0
    first = Snapshot.load(tmp_path).run
    assert run(["down", "--run-id", first]) == 0
    assert run(UP) == 0
    second_box, second = vast.ours(), Snapshot.load(tmp_path).run
    before = (tmp_path / "run.json").read_text()
    assert run(["down", "--run-id", first]) == 0  # the retried teardown of the first run
    assert vast.ours() == second_box
    # T4: by tag alone, and nothing of the directory's current run is touched.
    assert (tmp_path / "run.json").read_text() == before
    assert (tmp_path / "keys" / second).exists()
    assert run(["down"]) == 0 and vast.ours() == []


@pytest.mark.parametrize("run_id", ["", "quack-ci.nope.00000000", "../../etc/passwd", "a.b.c"])
def test_down_refuses_what_is_not_a_run_id(world, tmp_path, run_id):
    # An empty one is what a CI step passes when `up` never printed a run id:
    # that job rented nothing and must not tear down whatever else is here.
    vast, run = world
    assert run(UP) == 0
    deletes = len(vast.names("destroy"))
    assert run(["down", "--run-id", run_id]) == cli.EXIT_CONFIG
    assert len(vast.ours()) == 1 and len(vast.names("destroy")) == deletes
    assert run(["down"]) == 0


def test_down_by_run_id_without_a_state_directory_works_by_tag(world, tmp_path, capsys):
    vast, run = world
    assert 9003 in vast.live  # tagged for OTHER_RUN, whose state is nowhere
    assert run(["down", "--run-id", OTHER_RUN]) == 0
    assert 9003 not in vast.live and "by tag only" in capsys.readouterr().out
    assert not (tmp_path / "run.json").exists()


def test_a_damaged_snapshot_is_never_a_clean_teardown(world, tmp_path, capsys):
    vast, run = world
    assert run(UP) == 0
    run_id, iid = Snapshot.load(tmp_path).run, vast.ours()[0]
    damaged = (tmp_path / "run.json").read_text()[:40]
    (tmp_path / "run.json").write_text(damaged)
    assert run(["down"]) == cli.EXIT_CONFIG  # T3: we cannot even tell which run this is
    assert vast.ours() == [iid]
    # With the run id its tagged boxes are destroyed, but the ids the snapshot
    # held could not be checked: never exit 0, never remove the key.
    assert run(["down", "--run-id", run_id]) == cli.EXIT_INFRA
    assert vast.ours() == [] and (tmp_path / "keys" / run_id).exists()
    assert (tmp_path / "run.json").read_text() == damaged


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


def test_run_refuses_a_damaged_snapshot(world, tmp_path, monkeypatch):
    _, run = world
    assert run(UP) == 0
    (tmp_path / "run.json").write_text((tmp_path / "run.json").read_text()[:-20])
    calls = fake_ssh(monkeypatch)
    assert run(script(tmp_path)) == cli.EXIT_CONFIG and not calls


@pytest.mark.parametrize("run_id", ["../../outside", "/etc", "", "a.b.c", None, 7])
def test_a_run_id_is_checked_before_it_becomes_part_of_a_path(tmp_path, run_id):
    with pytest.raises(cli.StateError):
        cli.key_dir(tmp_path, run_id)
    good = "quack-ci.77-1.0123456789ab"
    assert cli.key_dir(tmp_path, good) == tmp_path / "keys" / good


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


def test_a_box_the_provider_says_is_over_is_recorded_as_died(world, tmp_path, monkeypatch):
    vast, run = world
    assert run(UP) == 0
    fake_ssh(monkeypatch, rc=255)
    vast.live[vast.ours()[0]]["status"] = "exited"
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None), ("died_after_admission", "died")]
    died = pol.Observations(tmp_path / "observations.jsonl").rows[-1]
    assert died["gpu"] == "RTX 5090" and died["gate_version"] == 1
    assert died["machine_id"] == vast.live[vast.ours()[0]]["machine"]


def test_an_instance_that_no_longer_exists_blames_nobody(world, tmp_path, monkeypatch):
    # Someone destroyed it: a teardown elsewhere, its own timer, the reaper.
    vast, run = world
    assert run(UP) == 0
    fake_ssh(monkeypatch, rc=255)
    vast.live.pop(vast.ours()[0])
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]


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

    def outage(iid, **kw):
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
    monkeypatch.setattr(vast, "instance", lambda iid, **kw: {"id": iid, "actual_status": status})
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert (("died_after_admission", "died") in observations(tmp_path)) is blamed


def test_our_own_down_during_run_blames_nobody(world, tmp_path, monkeypatch):
    vast, run = world
    assert run(UP) == 0
    iid = vast.ours()[0]

    def torn_down_meanwhile():
        Snapshot.load(tmp_path).set_closing()  # what `down` writes first
        vast.live[iid]["status"] = "exited"

    fake_ssh(monkeypatch, rc=255, before=torn_down_meanwhile)
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]


def test_down_that_starts_after_the_connection_dropped_is_still_not_blamed(
    world, tmp_path, monkeypatch
):
    # SSH fails for its own reasons; before we ask the provider, a teardown
    # starts. Whatever the provider then says about the box is our doing.
    vast, run = world
    assert run(UP) == 0
    iid = vast.ours()[0]
    fake_ssh(monkeypatch, rc=255)

    def over_because_of_us(instance_id, **kw):
        Snapshot.load(tmp_path).set_closing()
        return {"id": iid, "actual_status": "exited"}

    monkeypatch.setattr(vast, "instance", over_because_of_us)
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]


# ---------------------------------------------------------------------- reap


def old_box(vast, label=None):
    label = label or make_tag("quack-ci.old.0123456789ab", 1, int(time.time()) - 3600)
    vast.live[50] = {"id": 50, "label": label, "machine": 1, "start_date": time.time() - 9e4}


def test_reap_needs_explicit_permission_to_destroy(world, capsys):
    vast, run = world
    old_box(vast)
    reap = ["reap", "--namespace", "quack-ci"]
    assert run(reap) == 0
    assert 50 in vast.live and "report only" in capsys.readouterr().out
    assert vast.names("destroy") == []  # R4
    assert run([*reap, "--destroy"]) == 0
    assert 50 not in vast.live and foreign_untouched(vast)


def test_reap_reports_an_unconfirmed_delete_as_infrastructure(world):
    vast, run = world
    old_box(vast)
    vast.sticky.add(1)
    assert run(["reap", "--namespace", "quack-ci", "--destroy"]) == cli.EXIT_INFRA
    assert 50 in vast.live


def test_reap_reports_an_unreadable_listing_as_infrastructure(world):
    vast, run = world
    old_box(vast)
    vast.list_fails = 1
    assert run(["reap", "--namespace", "quack-ci", "--destroy"]) == cli.EXIT_INFRA  # R5
    assert 50 in vast.live


def test_an_exclusive_reap_destroys_what_carries_no_tag(world):
    vast, run = world
    for iid in [i for i in vast.live if i >= 9000]:
        vast.live.pop(iid)
    old_box(vast, label="renamed-by-the-box")
    vast.live[51] = {
        **vast.live[50],
        "id": 51,
        "label": make_tag(OTHER_RUN, 1, EXPIRY),
        "machine": 2,
    }
    vast.live[51]["start_date"] = time.time() - 60
    reap = ["reap", "--exclusive", "--max-age-min", "90", "--account-id", "111"]
    assert run(reap) == 0 and vast.names("destroy") == []
    assert run([*reap, "--destroy"]) == 0
    assert vast.ours() == [51]  # in its lease, and young


@pytest.mark.parametrize(
    "argv",
    [
        ["reap"],
        ["reap", "--namespace", "quack-ci", "--exclusive", "--account-id", "111"],
        ["reap", "--exclusive", "--account-id", "111"],  # no maximum age
        ["reap", "--namespace", "Not A Namespace"],
        ["reap", "--namespace", "quack-ci", "--max-age-min", "90"],  # revision 3's invocation
    ],
)
def test_a_reap_that_does_not_say_what_it_is_for_is_refused(world, argv):
    vast, run = world
    old_box(vast)
    assert run(argv) == cli.EXIT_CONFIG
    assert vast.names("instances", "destroy") == []


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
        run(["reap", "--exclusive", "--account-id", "111", "--max-age-min", "-1", "--destroy"])
