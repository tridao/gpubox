"""The tag, the state file and the lock."""

import json
import os

import pytest
from fakes import EXPIRY, RUN_ID

from gpubox.state import RUN_ID as RUN_ID_RE
from gpubox.state import (
    State,
    StateError,
    Tag,
    make_tag,
    name_of,
    new_run_id,
    parse_tag,
    state_lock,
)

META = {"image": "img", "gpu": "RTX 5090", "num_gpus": 1}
WINNER = {"id": 41, "attempt": 1, "host": "10.0.0.1", "port": 2222, "machine": 7, "host_id": 9}


def begin(tmp_path):
    return State.begin(tmp_path, RUN_ID, EXPIRY, "111", META)


# ------------------------------------------------------------------- the tag


def test_a_run_id_has_a_random_part_and_names_are_checked():
    a, b = new_run_id("quack-ci", "77-1"), new_run_id("quack-ci", "77-1")
    assert a != b and RUN_ID_RE.fullmatch(a) and a.startswith("quack-ci.77-1.")
    assert name_of(a) == "77-1"
    for namespace, name in [("Quack", "x"), ("a.b", "x"), ("a", "x.y"), ("", "x"), ("a", "")]:
        with pytest.raises(StateError):
            new_run_id(namespace, name)


def test_a_tag_round_trips():
    tag = make_tag(RUN_ID, 3, 1795400000)
    assert tag == "quack-ci.test.0123456789ab.a3.1795400000"
    assert parse_tag(tag) == Tag(RUN_ID, "quack-ci", 3, 1795400000)


@pytest.mark.parametrize(
    "label",
    [
        None, 7, "", "quack-ci", "quack-ci-old.x.12345678.a1",  # revision 3's scheme
        "quack-ci.test.0123456789ab.a3",  # no expiry
        "quack-ci.test.0123456789ab.a3.179540000",  # nine digits
        "quack-ci.test.0123456789ab.a0.1795400000",  # attempts count from one
        "quack-ci.test.0123456789AB.a3.1795400000",  # not lower-case hex
        "quack-ci.test.0123456789ab.a3.1795400000 ",  # trailing space
        "x quack-ci.test.0123456789ab.a3.1795400000",
        "quack-ci.te.st.0123456789ab.a3.1795400000",  # a dot in the name
        "Quack-ci.test.0123456789ab.a3.1795400000",
    ],
)  # fmt: skip
def test_a_label_that_only_resembles_a_tag_is_not_one(label):
    assert parse_tag(label) is None


def test_a_tag_cannot_be_made_from_nonsense():
    for run, n, expiry in [("nope", 1, EXPIRY), (RUN_ID, 0, EXPIRY), (RUN_ID, 1, 10**10)]:
        with pytest.raises(StateError):
            make_tag(run, n, expiry)


# ------------------------------------------------------------ the state file


def test_the_state_reads_back_what_was_written(tmp_path):
    state = begin(tmp_path)
    state.add_id(41)
    state.add_id(42)
    state.add_id(41)  # once is enough
    state.publish(WINNER)
    state.set_expiry(EXPIRY + 60)
    disk = State.load(tmp_path)
    assert disk.data == state.data
    assert (disk.run, disk.account, disk.expiry) == (RUN_ID, "111", EXPIRY + 60)
    assert disk.ids == [41, 42] and disk.winner == WINNER
    assert not disk.closing and not disk.closed
    state.close()
    assert State.load(tmp_path).closed and State.load(tmp_path).closing


def test_no_state_is_none_and_an_unusable_one_is_an_error(tmp_path):
    assert State.load(tmp_path) is None
    good = begin(tmp_path).path.read_text()
    bad = [
        "",
        "garbage",
        "[]",
        good[: len(good) // 2],
        good.replace('"version": 2', '"version": 1'),
    ]
    bad.append(good.replace(RUN_ID, "../../etc"))
    bad.append(json.dumps({**json.loads(good), "ids": ["41"]}))
    bad.append(json.dumps({**json.loads(good), "winner": {"id": 41}}))
    for text in bad:
        (tmp_path / "run.json").write_text(text)
        with pytest.raises(StateError):
            State.load(tmp_path)


def test_a_change_that_cannot_be_written_changes_nothing(tmp_path, monkeypatch):
    state = begin(tmp_path)
    before_disk, before_memory = state.path.read_text(), json.dumps(state.data, sort_keys=True)

    def fail(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    for change in (lambda: state.add_id(41), lambda: state.publish(WINNER), state.set_closing):
        with pytest.raises(OSError):
            change()
    assert state.path.read_text() == before_disk
    assert json.dumps(state.data, sort_keys=True) == before_memory
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]  # no temporary file left behind


def test_an_ephemeral_state_is_never_written(tmp_path):
    state = State.ephemeral(RUN_ID)
    state.add_id(7)
    state.close()
    assert state.closed and state.ids == [7] and list(tmp_path.iterdir()) == []
    with pytest.raises(StateError):
        State.ephemeral("not-a-run-id")


def test_beginning_a_run_replaces_the_closed_one_before_it(tmp_path):
    begin(tmp_path).close()
    second = State.begin(tmp_path, "quack-ci.next.ba9876543210", EXPIRY, None, META)
    assert State.load(tmp_path).run == second.run and not State.load(tmp_path).closed


# ------------------------------------------------------------------ the lock


def test_taking_the_lock_twice_in_one_process_fails_fast(tmp_path):
    with state_lock(tmp_path):
        with pytest.raises(StateError, match="in use"):
            with state_lock(tmp_path):
                pass
    with state_lock(tmp_path):  # released on the way out
        pass
