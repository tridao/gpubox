"""The tag, the snapshot and the lock."""

import json
import os

import pytest
from fakes import EXPIRY, RUN_ID

from gpubox.state import (
    RUN_ID as RUN_ID_RE,
)
from gpubox.state import (
    Snapshot,
    StateError,
    Tag,
    make_tag,
    new_run_id,
    parse_tag,
    state_lock,
)

META = {"image": "img", "gpu": "RTX 5090", "num_gpus": 1}


def begin(tmp_path):
    return Snapshot.begin(tmp_path, RUN_ID, EXPIRY, "111", META)


def opened(tmp_path, n=1):
    snap = begin(tmp_path)
    for k in range(1, n + 1):
        snap.open_attempt(k, sent=1000.0 + k, tag=make_tag(RUN_ID, k, EXPIRY), machine=k)
    return snap


# ------------------------------------------------------------------- the tag


def test_a_run_id_has_a_random_part_and_names_are_checked():
    a, b = new_run_id("quack-ci", "77-1"), new_run_id("quack-ci", "77-1")
    assert a != b and RUN_ID_RE.fullmatch(a) and a.startswith("quack-ci.77-1.")
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


# -------------------------------------------------------------- the snapshot


def test_a_snapshot_reads_back_what_was_written(tmp_path):
    snap = opened(tmp_path, 2)
    snap.bind(1, 41)
    snap.refuse(2)
    snap.know(42)
    snap.gone(42)
    snap.publish(41, 1, "10.0.0.1", 2222)
    disk = Snapshot.load(tmp_path)
    assert disk.data == snap.data
    assert disk.run == RUN_ID and disk.account == "111" and disk.expiry == EXPIRY
    assert disk.ids() == {41: "live", 42: "gone"} and disk.live_ids() == [41]
    assert {n: a["state"] for n, a in disk.attempts().items()} == {1: "bound", 2: "refused"}
    assert disk.winner == {"id": 41, "attempt": 1, "host": "10.0.0.1", "port": 2222}
    assert disk.attempt_of(41)[0] == 1 and disk.attempt_of(42) is None


def test_no_snapshot_is_none_and_a_damaged_one_is_an_error(tmp_path):
    assert Snapshot.load(tmp_path) is None
    good = begin(tmp_path).path.read_text()
    for text in [
        "",
        "garbage",
        "[]",
        good[: len(good) // 2],
        good.replace('"version": 1', '"version": 2'),
    ]:
        (tmp_path / "run.json").write_text(text)
        with pytest.raises(StateError):
            Snapshot.load(tmp_path)
    (tmp_path / "run.json").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(StateError):
        Snapshot.load(tmp_path)


@pytest.mark.parametrize(
    "damage",
    [
        lambda d: d["ids"].__setitem__("7", "maybe"),
        lambda d: d["ids"].__setitem__("-1", "live"),
        lambda d: d["ids"].__setitem__("007", "live"),
        lambda d: d["attempts"]["1"].__setitem__("state", "bound"),  # bound to no id
        lambda d: d["attempts"]["1"].update(state="bound", id=99),  # an id the snapshot lacks
        lambda d: d["attempts"]["1"].pop("sent"),
        lambda d: d.__setitem__("winner", {"id": 99, "attempt": 1, "host": "h", "port": 1}),
        lambda d: d.__setitem__("run", "../../etc"),
        lambda d: d.__setitem__("closed", "yes"),
        lambda d: d.pop("account"),
        lambda d: d.update(closed=True, closing=True),  # closed, with an attempt still open
        lambda d: d.update(closed=True, closing=False, attempts={}),
        lambda d: d.__setitem__("expiry", "soon"),
    ],
)
def test_a_snapshot_that_does_not_hold_together_is_not_used(tmp_path, damage):
    snap = opened(tmp_path)
    data = json.loads(snap.path.read_text())
    damage(data)
    snap.path.write_text(json.dumps(data))
    with pytest.raises(StateError):
        Snapshot.load(tmp_path)


def test_a_change_that_cannot_be_written_changes_nothing(tmp_path, monkeypatch):
    snap = opened(tmp_path)
    before_disk, before_memory = snap.path.read_text(), json.dumps(snap.data, sort_keys=True)

    def fail(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    for change in (
        lambda: snap.bind(1, 41),
        lambda: snap.know(5),
        snap.set_closing,
        lambda: snap.refuse(1),
    ):
        with pytest.raises(OSError):
            change()
    assert snap.path.read_text() == before_disk
    assert json.dumps(snap.data, sort_keys=True) == before_memory
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]  # no temporary file left behind


def test_a_winner_must_be_the_box_of_its_attempt(tmp_path):
    snap = opened(tmp_path, 2)
    snap.bind(1, 41)
    snap.bind(2, 42)
    for args in ((42, 1), (41, 2), (41, 3)):
        with pytest.raises(ValueError):
            snap.publish(*args, "10.0.0.1", 2222)
    assert Snapshot.load(tmp_path).winner is None
    snap.publish(41, 1, "10.0.0.1", 2222)


def test_a_run_that_may_have_a_box_is_not_closed(tmp_path):
    snap = opened(tmp_path)
    snap.bind(1, 41)
    snap.gone(41)
    snap.close()
    assert Snapshot.load(tmp_path).closed
    snap.know(42)  # a box that turned up after the run was closed
    assert not Snapshot.load(tmp_path).closed and Snapshot.load(tmp_path).closing
    snap.gone(42)
    snap.revive(41)  # a lookup found it after all
    assert Snapshot.load(tmp_path).live_ids() == [41]
    with pytest.raises(StateError):
        snap.close()  # closing says "nothing may exist", and something may
    assert not Snapshot.load(tmp_path).closed


def test_an_error_after_the_rename_is_raised_and_memory_follows_the_file(tmp_path, monkeypatch):
    import errno

    snap = opened(tmp_path)
    real = os.fsync

    def fsync(fd):
        if os.path.isdir(f"/dev/fd/{fd}") or _is_dir(fd):
            raise OSError(fsync.errno, "the directory could not be synced")
        real(fd)

    def _is_dir(fd):
        import stat

        return stat.S_ISDIR(os.fstat(fd).st_mode)

    monkeypatch.setattr(os, "fsync", fsync)
    fsync.errno = errno.EIO
    with pytest.raises(OSError):
        snap.bind(1, 41)  # reported as a failure: it may not survive a power cut
    assert snap.status(41) == "live" == Snapshot.load(tmp_path).status(41)  # but it is in the file
    fsync.errno = errno.EINVAL  # a filesystem that cannot sync a directory at all
    snap.know(42)
    assert Snapshot.load(tmp_path).status(42) == "live"


def test_an_attempt_changes_state_once(tmp_path):
    snap = opened(tmp_path, 3)
    snap.bind(1, 41)
    snap.refuse(2)
    for change in (lambda: snap.bind(1, 42), lambda: snap.refuse(1), lambda: snap.bind(2, 43)):
        with pytest.raises(StateError):
            change()
    with pytest.raises(StateError):
        snap.open_attempt(3, sent=1.0)  # an attempt number is used once
    assert snap.ids() == {41: "live"}


def test_an_id_never_comes_back_from_gone(tmp_path):
    snap = opened(tmp_path)
    snap.bind(1, 41)
    snap.gone(41)
    snap.know(41)
    assert Snapshot.load(tmp_path).status(41) == "gone"


def test_slots_count_attempts_that_may_have_a_box(tmp_path):
    snap = opened(tmp_path, 4)
    assert snap.slots() == 4  # open: a box may exist for each
    snap.refuse(1)
    snap.bind(2, 41)
    snap.bind(3, 42)
    snap.gone(42)
    assert snap.slots() == 2  # the bound one whose box is live, and the one still open
    snap.unseen(4)
    assert snap.slots() == 1 and snap.is_live()


def test_an_ephemeral_snapshot_is_never_written(tmp_path):
    snap = Snapshot.ephemeral(RUN_ID)
    snap.know(7)
    snap.gone(7)
    snap.close()
    assert snap.closed and snap.status(7) == "gone" and list(tmp_path.iterdir()) == []
    with pytest.raises(StateError):
        Snapshot.ephemeral("not-a-run-id")


def test_beginning_a_run_replaces_the_closed_one_before_it(tmp_path):
    first = begin(tmp_path)
    first.close()
    second = Snapshot.begin(tmp_path, "quack-ci.next.ba9876543210", EXPIRY, None, META)
    assert Snapshot.load(tmp_path).run == second.run and not Snapshot.load(tmp_path).closed


# ------------------------------------------------------------------ the lock


def test_taking_the_lock_twice_in_one_process_fails_fast(tmp_path):
    with state_lock(tmp_path):
        with pytest.raises(StateError, match="in use"):
            with state_lock(tmp_path):
                pass
    with state_lock(tmp_path):  # released on the way out
        pass
