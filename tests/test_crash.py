"""A launcher process that is killed, and what the next process finds.

These run a real second process against a provider whose state is a file, so
"the process died here" is literal: nothing in memory survives.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from fakes import FAST, RUN_ID, FakeVast, FileVast
from test_cli import UP

from gpubox import cli
from gpubox import launch as L
from gpubox.state import Snapshot

CHILD = str(Path(__file__).with_name("crash_child.py"))
CUES = ["after:open", "after:create", "after:bind", "after:winner"]


def crash(tmp_path, cue):
    r = subprocess.run(
        [sys.executable, CHILD, str(tmp_path), cue], capture_output=True, text=True, timeout=60
    )
    assert r.returncode == 9, (r.returncode, r.stdout, r.stderr)
    return FileVast(tmp_path / "provider.json", {1: "ok"}), Snapshot.load(tmp_path)


def test_killed_after_the_attempt_is_on_disk_but_before_the_request(tmp_path):
    vast, snap = crash(tmp_path, "after:open")
    assert vast.on_disk() == [] and snap.ids() == {} and list(snap.open_attempts()) == [1]
    # Nothing was rented, but nothing proves it: the attempt stays open.
    assert L.down(vast, snap, FAST) == 1 and not Snapshot.load(tmp_path).closed
    assert L.down(vast, Snapshot.load(tmp_path), FAST, abandon_unseen=True) == 0
    assert Snapshot.load(tmp_path).closed


def test_killed_after_the_provider_rented_but_before_we_knew_the_id(tmp_path):
    vast, snap = crash(tmp_path, "after:create")
    (iid,) = vast.on_disk()
    assert snap.ids() == {} and list(snap.open_attempts()) == [1]
    assert L.down(vast, snap, FAST) == 0  # found by its tag, which names the open attempt
    disk = Snapshot.load(tmp_path)
    assert vast.on_disk() == [] and disk.ids() == {iid: "gone"} and disk.closed


def test_killed_after_the_id_was_on_disk(tmp_path):
    vast, snap = crash(tmp_path, "after:bind")
    (iid,) = vast.on_disk()
    assert snap.live_ids() == [iid]
    vast._with_file(lambda: vast.live[iid].update(label="renamed-by-the-box"))
    assert L.down(vast, snap, FAST) == 0  # by id, whatever it is called now
    assert vast.on_disk() == [] and Snapshot.load(tmp_path).closed


def test_killed_right_after_the_hand_off(tmp_path):
    vast, snap = crash(tmp_path, "after:winner")
    (iid,) = vast.on_disk()
    assert snap.winner["id"] == iid
    assert L.down(vast, snap, FAST) == 0 and vast.on_disk() == []


@pytest.mark.parametrize("cue", ["after:create", "after:bind", "after:winner"])
def test_a_runner_whose_disk_is_gone_is_torn_down_with_the_run_id_alone(tmp_path, monkeypatch, cue):
    # The CI case: the job's teardown step runs somewhere that has nothing but
    # the run id. The tag is enough.
    vast, _ = crash(tmp_path, cue)
    assert len(vast.on_disk()) == 1
    monkeypatch.setenv("VASTAI_API_KEY", "k")
    monkeypatch.delenv("GPUBOX_ACCOUNT_ID", raising=False)
    monkeypatch.setattr(cli.v, "Vast", lambda key: vast)
    monkeypatch.setattr(cli, "Policy", lambda **kw: type(FAST)(**{**FAST.__dict__, **kw}))
    empty = tmp_path / "another-runner"
    assert cli.main(["down", "--run-id", RUN_ID, "--state-dir", str(empty)]) == 0
    assert vast.on_disk() == []


def test_a_box_that_changed_its_label_on_a_lost_runner_is_left_to_the_reaper(tmp_path):
    # What neither the tag nor a lost snapshot can reach, stated as a test:
    # only a reaper that owns the whole account destroys this box.
    vast, _ = crash(tmp_path, "after:winner")
    (iid,) = vast.on_disk()
    vast._with_file(lambda: vast.live[iid].update(label="renamed-by-the-box"))
    assert L.down(vast, Snapshot.ephemeral(RUN_ID), FAST) == 0  # sees nothing with the tag
    assert vast.on_disk() == [iid]
    vast._with_file(lambda: [vast.live.pop(i) for i in list(vast.live) if i >= 9000])
    assert L.reap(vast, FAST, max_age_min=90, destroy=True) == 0
    assert vast.on_disk() == []


def test_a_command_is_refused_while_another_process_holds_the_directory(tmp_path, monkeypatch):
    child = subprocess.Popen(
        [sys.executable, CHILD, str(tmp_path), "hold-lock"], stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout.readline().strip() == "locked"
        vast = FakeVast({1: "ok"})
        monkeypatch.setenv("VASTAI_API_KEY", "k")
        monkeypatch.setattr(cli.v, "Vast", lambda key: vast)
        assert cli.main([*UP, "--state-dir", str(tmp_path)]) == cli.EXIT_CONFIG
        assert cli.main(["down", "--state-dir", str(tmp_path)]) == cli.EXIT_CONFIG
        assert vast.creates == 0
    finally:
        child.kill()
        child.wait()
    # The kernel released the lock with the process.
    assert cli.main(["down", "--state-dir", str(tmp_path)]) == 0


@pytest.mark.parametrize("cue", CUES)
def test_the_next_up_refuses_to_start_over_what_a_crash_left(tmp_path, monkeypatch, cue):
    crash(tmp_path, cue)
    vast = FakeVast({1: "ok"})
    monkeypatch.setenv("VASTAI_API_KEY", "k")
    monkeypatch.setattr(cli.v, "Vast", lambda key: vast)
    assert cli.main([*UP, "--state-dir", str(tmp_path)]) == cli.EXIT_CONFIG
    assert vast.creates == 0
