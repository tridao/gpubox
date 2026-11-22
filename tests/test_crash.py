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
from gpubox.journal import Journal, attempt_label

CHILD = str(Path(__file__).with_name("crash_child.py"))


def crash(tmp_path, cue):
    r = subprocess.run(
        [sys.executable, CHILD, str(tmp_path), cue], capture_output=True, text=True, timeout=60
    )
    assert r.returncode == 9, (r.returncode, r.stdout, r.stderr)
    return FileVast(tmp_path / "provider.json", {1: "ok"}), Journal(tmp_path)


def test_killed_after_the_intent_is_durable_but_before_the_request(tmp_path):
    vast, journal = crash(tmp_path, "after:intent")
    assert vast.on_disk() == [] and journal.state().boxes == {}
    # Nothing was rented, but nothing proves it: the attempt stays open.
    assert L.down(vast, journal, FAST) == 1 and not journal.state().closed
    later = time.time() + FAST.settle + 1
    assert L.down(vast, journal, FAST, abandon_unseen=True, now=later) == 0


def test_killed_after_the_provider_rented_but_before_we_knew_the_id(tmp_path):
    vast, journal = crash(tmp_path, "after:create")
    (iid,) = vast.on_disk()
    state = journal.state()
    assert state.boxes == {} and [a.label for a in state.open_attempts()] == [
        attempt_label(RUN_ID, 1)
    ]
    assert L.down(vast, journal, FAST) == 0  # found: our label, machine, image, and time
    assert vast.on_disk() == [] and journal.state().boxes == {iid: "destroyed"}


def test_killed_after_the_id_was_journalled(tmp_path):
    vast, journal = crash(tmp_path, "after:created")
    (iid,) = vast.on_disk()
    assert journal.state().live_ids() == [iid]
    vast._with_file(lambda: vast.live[iid].update(label="renamed-by-the-box"))
    assert L.down(vast, journal, FAST) == 0  # by id, whatever it is called now
    assert vast.on_disk() == [] and journal.state().closed


def test_killed_right_after_the_winner_was_recorded(tmp_path):
    vast, journal = crash(tmp_path, "after:winner")
    (iid,) = vast.on_disk()
    assert journal.state().winner["instance_id"] == iid
    assert L.down(vast, journal, FAST) == 0 and vast.on_disk() == []


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


@pytest.mark.parametrize("cue", ["after:intent", "after:create", "after:created", "after:winner"])
def test_the_next_up_refuses_to_start_over_what_a_crash_left(tmp_path, monkeypatch, cue):
    crash(tmp_path, cue)
    vast = FakeVast({1: "ok"})
    monkeypatch.setenv("VASTAI_API_KEY", "k")
    monkeypatch.setattr(cli.v, "Vast", lambda key: vast)
    assert cli.main([*UP, "--state-dir", str(tmp_path)]) == cli.EXIT_CONFIG
    assert vast.creates == 0
