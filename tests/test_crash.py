"""A launcher process that is killed, and what the next process finds.

These run a real second process against a provider whose state is a file, so
"the process died here" is literal: nothing in memory survives.
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest
from fakes import FAST, RUN_ID, FakeVast, FileVast
from test_cli import UP

from gpubox import cli
from gpubox import launch as L
from gpubox.state import State

CHILD = str(Path(__file__).with_name("crash_child.py"))
CUES = ["after:create", "after:id", "after:winner"]


def crash(tmp_path, cue):
    r = subprocess.run(
        [sys.executable, CHILD, str(tmp_path), cue], capture_output=True, text=True, timeout=60
    )
    assert r.returncode == 9, (r.returncode, r.stdout, r.stderr)
    return FileVast(tmp_path / "provider.json", {1: "ok"}), State.load(tmp_path)


def test_killed_after_the_provider_rented_but_before_we_knew_the_id(tmp_path):
    vast, state = crash(tmp_path, "after:create")
    assert len(vast.on_disk()) == 1 and state.ids == []
    from gpubox import receipt

    (attempt,) = receipt(tmp_path, state.run).values()
    assert attempt["outcome"] == "unknown" and attempt["instance_id"] is None
    assert attempt["dph"] == 1 and attempt["offer_id"] == 1001
    vast.hidden.update(vast.on_disk())
    assert L.down(vast, state, FAST) is None and not State.load(tmp_path).closed
    vast.hidden.clear()
    assert L.down(vast, state, FAST) == []  # found by its tag
    (attempt,) = receipt(tmp_path, state.run).values()
    assert attempt["outcome"] == "gone" and attempt["instance_id"] == 1
    assert vast.on_disk() == [] and State.load(tmp_path).closed


def test_killed_after_the_id_was_on_disk(tmp_path):
    vast, state = crash(tmp_path, "after:id")
    (iid,) = vast.on_disk()
    assert state.ids == [iid]
    vast._with_file(lambda: vast.live[iid].update(label="renamed"))
    assert L.down(vast, state, FAST) == []  # by id, whatever it is called now
    assert vast.on_disk() == [] and State.load(tmp_path).closed


def test_killed_right_after_the_hand_off(tmp_path):
    vast, state = crash(tmp_path, "after:winner")
    (iid,) = vast.on_disk()
    assert state.winner["id"] == iid
    assert L.down(vast, state, FAST) == [] and vast.on_disk() == []


@pytest.mark.parametrize("cue", CUES)
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


@pytest.mark.parametrize("cue", CUES)
def test_a_box_nobody_tears_down_is_reaped_once_its_lease_has_ended(tmp_path, cue):
    # No `down` at all: the runner died and nothing else ran. The lease is
    # what bounds it.
    vast, state = crash(tmp_path, cue)
    assert len(vast.on_disk()) == 1
    assert L.reap(vast, FAST, "quack-ci", destroy=True) == 0 and len(vast.on_disk()) == 1
    after = state.expiry + FAST.expiry_margin + 1
    assert L.reap(vast, FAST, "quack-ci", destroy=True, now=after) == 0
    assert vast.on_disk() == [] and time.time() < state.expiry


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
