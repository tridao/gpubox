"""The race, the snapshot and teardown, against a fake provider.

Tests are grouped by the rules of DESIGN.md (S: snapshot, D: destroying,
L: launch, T: teardown, R: reaper). Every test that rents states which boxes
are left afterwards, and `run` checks the peak number of billable boxes and
that nothing foreign was touched.
"""

import os
import signal
import time

import pytest
from fakes import (
    EXPIRY,
    FAST,
    OTHER_RUN,
    RUN_ID,
    SPEC,
    FakeVast,
    foreign_untouched,
    offer,
    start,
    states,
)

from gpubox import launch as L
from gpubox import policy as pol
from gpubox import vast as v
from gpubox.state import Snapshot, make_tag


def run(tmp_path, behaviour, **kw):
    launch, vast, snap = start(tmp_path, behaviour, **kw)
    try:
        result = launch.run({})
    finally:
        assert foreign_untouched(vast)
        assert vast.peak <= launch.policy.max_live
    return result, vast, snap


def outcome(launch):
    """What `launch.run` raised, or None. Unlike `pytest.raises`, this also
    holds on to a KeyboardInterrupt, which would otherwise end the whole
    test session instead of failing one test."""
    try:
        launch.run({})
    except BaseException as exc:
        return exc
    return None


def on_disk(snap):
    return Snapshot.load(snap.path.parent)


def reasons(result):
    return {a.machine_id: a.reason for a in result.attempts}


def observed(result):
    obs = pol.Observations(None)
    L.observe(result, SPEC, "r", obs)
    return {(o["machine_id"], o["kind"], o["reason_code"]) for o in obs.rows}


def only_winner_left(result, vast, snap):
    """The provider and the snapshot on disk agree: exactly the winner exists."""
    assert vast.ours() == [result.winner.instance_id]
    disk = on_disk(snap)
    assert disk.live_ids() == [result.winner.instance_id]
    assert disk.winner["id"] == result.winner.instance_id
    assert not disk.open_attempts() and not result.left


def nothing_left(vast, snap):
    assert vast.ours() == []
    assert not on_disk(snap).is_live()


def at_call(vast, name, nth, action):
    """Run `action` at the start of the nth call of `name` (counting from 1)."""
    seen = [0]

    def hook(arg):
        seen[0] += 1
        if seen[0] == nth:
            action()

    vast.hooks[name] = hook


def interrupt():
    raise KeyboardInterrupt


# ------------------------------------------------------------------ the race


def test_failed_machine_is_replaced_and_blamed(tmp_path):
    result, vast, snap = run(tmp_path, {1: "bad_gpu", 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    only_winner_left(result, vast, snap)
    assert observed(result) == {(1, "failed_admission", "cuda_init"), (2, "healthy", None)}


def test_race_keeps_one_and_does_not_blame_the_loser(tmp_path):
    result, vast, snap = run(tmp_path, {1: "ok", 2: "ok", 3: "ok"})
    assert result.winner.machine_id in (1, 2)
    assert 3 not in reasons(result)  # never needed
    only_winner_left(result, vast, snap)
    assert {kind for _, kind, _ in observed(result)} == {"healthy"}


def test_a_loser_still_in_its_gate_is_stopped_and_not_blamed(tmp_path):
    launch, vast, snap = start(tmp_path, {1: "late", 2: "ok"})
    result = launch.run({})
    assert result.winner.machine_id == 2 and 1 in vast.gate_started
    assert reasons(result)[1] is None
    assert all(c.killed for c in launch.prober.checks if c.poll() is None)
    only_winner_left(result, vast, snap)


def test_nothing_came_up_rolls_back_and_blames_nobody(tmp_path):
    # Could be Docker Hub or the image: no control, so no verdict on the machines.
    result, vast, snap = run(tmp_path, {1: "stuck", 2: "bad_gpu", 3: "cdi"})
    assert result.winner is None
    nothing_left(vast, snap)
    assert observed(result) == set()


@pytest.mark.parametrize(
    "kind,reason", [("stuck", "unreachable"), ("exited", "unreachable"), ("cdi", "runtime_error")]
)
def test_startup_failures_carry_a_reason_code(tmp_path, kind, reason):
    result, vast, snap = run(tmp_path, {1: kind, 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    assert reasons(result)[1] == reason
    only_winner_left(result, vast, snap)


def test_a_box_whose_loading_status_stops_changing_is_replaced_before_the_startup_timeout(
    tmp_path,
):
    t0 = time.monotonic()
    result, vast, snap = run(
        tmp_path, {1: "stuck", 2: "ok"}, spare=0, startup_timeout=30, stall_timeout=0.2
    )
    assert result.winner.machine_id == 2 and time.monotonic() - t0 < 5
    assert reasons(result)[1] == "unreachable"
    assert "stalled" in next(a.detail for a in result.attempts if a.machine_id == 1)
    only_winner_left(result, vast, snap)


def test_a_box_that_keeps_reporting_progress_is_not_taken_for_stalled(tmp_path):
    # Slow is not stuck: it gets the whole startup timeout.
    result, vast, snap = run(
        tmp_path, {1: "pulling", 2: "ok"}, spare=0, startup_timeout=0.6, stall_timeout=0.2
    )
    assert result.winner.machine_id == 2
    slow = next(a for a in result.attempts if a.machine_id == 1)
    assert "not reachable after 0.6s" in slow.detail and "stalled" not in slow.detail
    only_winner_left(result, vast, snap)


def test_stale_offer_is_skipped_without_blame(tmp_path):
    result, vast, snap = run(tmp_path, {1: "stale", 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    assert reasons(result)[1] is None
    assert states(snap)[1] == "refused" and vast.creates == 1
    only_winner_left(result, vast, snap)


def test_a_silent_provider_api_is_not_held_against_the_machine(tmp_path):
    result, vast, snap = run(tmp_path, {1: "api_down", 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    assert reasons(result)[1] is None  # we never heard about the box: no evidence
    only_winner_left(result, vast, snap)


def test_max_attempts_bounds_the_spend(tmp_path):
    result, vast, snap = run(tmp_path, dict.fromkeys(range(1, 10), "bad_gpu"), max_attempts=3)
    assert result.winner is None and vast.creates == 3
    nothing_left(vast, snap)


def test_timeout_rolls_back_a_candidate_that_would_have_been_healthy_later(tmp_path):
    result, vast, snap = run(tmp_path, {1: "hold"}, overall_timeout=0.2, startup_timeout=5)
    assert result.winner is None
    nothing_left(vast, snap)


# ---------------------------------------------------- L1: when nothing is rented


def test_no_box_is_rented_once_the_renting_window_has_closed(tmp_path):
    result, vast, snap = run(tmp_path, {1: "ok"}, overall_timeout=0)
    assert result.winner is None and vast.creates == 0


def test_the_window_is_checked_again_just_before_each_request(tmp_path):
    # The first create of the pass takes until after the cut-off. The second
    # is not sent, although the pass began in time.
    launch, vast, snap = start(tmp_path, {1: "hold", 2: "hold"}, overall_timeout=0.2)
    at_call(vast, "create", 1, lambda: time.sleep(0.3))
    result = launch.run({})
    assert result.winner is None and vast.creates == 1
    assert list(states(snap)) == [1]  # the second attempt was not even begun
    nothing_left(vast, snap)


@pytest.mark.parametrize("why", ["the window closed", "the recorded time went stale"])
def test_nothing_is_sent_if_recording_the_attempt_took_too_long(tmp_path, monkeypatch, why):
    # The cut-off was checked, then the write of the attempt stalled.
    if why == "the window closed":  # and the recorded time is nowhere near stale
        slow, kw = 0.3, {"overall_timeout": 0.2, "settle": 100}
    else:  # stale, with the window wide open
        slow, kw = FAST.settle / 4 + 0.05, {"overall_timeout": 5}
    launch, vast, snap = start(tmp_path, {1: "ok"}, spare=0, **kw)
    real = snap.open_attempt
    stalls = [slow]

    def open_attempt(*a, **kw):
        real(*a, **kw)
        time.sleep(stalls.pop() if stalls else 0)

    monkeypatch.setattr(snap, "open_attempt", open_attempt)
    result = launch.run({})
    assert states(snap)[1] == "refused"  # nothing was sent for it, and the snapshot says so
    sent = [m for n, m in vast.names("create")]
    assert sent == ([] if why == "the window closed" else [1])  # a later pass may try again
    assert vast.ours() == ([result.winner.instance_id] if result.winner else [])


def test_a_pass_without_a_listing_rents_nothing(tmp_path):
    launch, vast, snap = start(tmp_path, {1: "ok"}, spare=0)
    vast.list_fails = 3
    result = launch.run({})
    assert result.winner is not None
    names = [n for n, _ in vast.names("instances", "create")]
    assert names[:4] == ["instances"] * 4 and names[4] == "create"  # three failed, one answered


def test_bad_image_aborts_instead_of_trying_every_host(tmp_path):
    launch, vast, snap = start(tmp_path, dict.fromkeys(range(1, 6), "image"))
    with pytest.raises(L.ConfigError, match="cannot be pulled"):
        launch.run({})
    assert vast.creates == 2  # no replacement is rented once the image is known bad
    nothing_left(vast, snap)


@pytest.mark.parametrize("behaviour", [{1: "ok", 2: "image"}, {1: "image", 2: "ok"}])
def test_a_healthy_box_outweighs_another_hosts_image_error(tmp_path, behaviour):
    # A healthy box proves the image can be pulled, whichever is looked at first.
    result, vast, snap = run(tmp_path, behaviour)
    assert result.winner is not None
    only_winner_left(result, vast, snap)


def test_an_image_error_does_not_cancel_a_box_that_is_still_coming_up(tmp_path):
    launch, vast, snap = start(tmp_path, {1: "image", 2: "slow", 3: "ok"})
    result = launch.run({})
    assert result.winner.machine_id == 2
    assert vast.creates == 2  # machine 3 was never rented: the window closed with the error
    only_winner_left(result, vast, snap)


def test_after_an_image_error_the_others_get_a_short_while_not_the_whole_timeout(tmp_path):
    launch, vast, snap = start(tmp_path, {1: "image", 2: "hold"}, startup_timeout=60)
    t0 = time.monotonic()
    with pytest.raises(L.ConfigError, match="cannot be pulled"):
        launch.run({})
    assert time.monotonic() - t0 < FAST.overall_timeout / 2  # image_grace, not overall_timeout
    nothing_left(vast, snap)


# ------------------------------------------------------- L2: the cap and adoption


def test_a_create_with_no_answer_holds_its_slot_instead_of_being_replaced(tmp_path):
    result, vast, snap = run(tmp_path, {1: "ghost", 2: "ok", 3: "ok"})
    assert result.winner.machine_id == 2
    assert vast.creates == 1  # machine 3 was never rented in the ghost's place
    assert [a.seq for a in result.unseen] == [1] and states(snap)[1] == "open"
    assert vast.ours() == [result.winner.instance_id]


def test_two_creates_with_no_answer_stop_the_renting(tmp_path):
    # Both boxes may exist unseen. Renting more would break the cap for boxes
    # nobody can see, so the launch ends without a winner once they are stale.
    result, vast, snap = run(tmp_path, {1: "ghost", 2: "ghost", 3: "ok", 4: "ok"})
    assert result.winner is None and vast.creates == 0
    assert [a.seq for a in result.unseen] == [1, 2]


def test_a_box_whose_create_gave_no_answer_is_adopted_from_the_listing_and_can_win(tmp_path):
    result, vast, snap = run(tmp_path, {1: "lost"}, spare=0)
    assert result.winner.machine_id == 1 and result.winner.seq == 1
    assert states(snap)[1] == "bound" and not result.unseen
    only_winner_left(result, vast, snap)


def test_an_adopted_box_fills_its_own_slot_and_no_other(tmp_path):
    result, vast, snap = run(tmp_path, {1: "lost", 2: "hold", 3: "ok"})
    assert result.winner.machine_id == 1
    assert vast.creates == 2  # the lost box and the held one; never a third
    only_winner_left(result, vast, snap)


def test_a_box_with_the_runs_tag_but_no_open_attempt_is_destroyed(tmp_path):
    # And at once, not when the race is over: here the race can only end
    # after the stray has been deleted.
    launch, vast, snap = start(tmp_path, {1: "hold"}, spare=0, overall_timeout=1)
    vast.live[500] = {"id": 500, "label": make_tag(RUN_ID, 9, EXPIRY), "machine": 99}
    vast.hooks["destroy"] = lambda iid: setattr(vast, "release", iid == 500)
    result = launch.run({})
    assert result.winner is not None and 500 not in vast.live
    assert on_disk(snap).status(500) == "gone"
    only_winner_left(result, vast, snap)


def test_a_lookup_that_wrongly_said_gone_is_corrected_by_the_listing(tmp_path):
    # The lookup lags: it says "does not exist" for a box that does. A delete
    # is asked for before that is believed, and when the listing still shows
    # the box later, a second lookup decides and it is destroyed after all.
    launch, vast, snap = start(tmp_path, {1: "hold", 2: "ok"}, spare=0, startup_timeout=5)
    vast.sticky.add(1)
    vast.hooks["created"] = lambda iid: iid == 1 and vast.lookup_hidden.add(1)

    def correct(iid):
        if iid == 2:  # the replacement exists: now the first box's lookup works again
            vast.lookup_hidden.clear()
            vast.sticky.clear()

    vast.hooks["instance"] = correct
    result = launch.run({})
    assert result.winner.machine_id == 2
    assert ("destroy", 1) in vast.names()[: vast.names().index(("create", 2))]  # asked first
    assert vast.ours() == [result.winner.instance_id]  # the first box did not survive
    assert on_disk(snap).status(1) == "gone"


def test_down_destroys_a_box_that_was_wrongly_recorded_as_gone(tmp_path):
    result, vast, snap = up(tmp_path, {1: "ok"}, spare=0)
    snap.gone(result.winner.instance_id)  # as a lagging lookup would have had it
    assert L.down(vast, snap, FAST) == 0
    assert vast.ours() == [] and on_disk(snap).closed


def test_a_failed_box_whose_delete_is_not_confirmed_keeps_its_slot(tmp_path):
    launch, vast, snap = start(tmp_path, {1: "bad_gpu", 2: "ok"}, spare=0)
    vast.sticky.add(1)
    result = launch.run({})
    assert result.winner is None and vast.creates == 1  # never two billable boxes at once
    (stuck,) = vast.ours()
    assert result.left == [stuck] and on_disk(snap).live_ids() == [stuck]


# ----------------------------------------------------------- L3: the winner


def test_the_cheaper_of_two_boxes_healthy_in_the_same_pass_wins(tmp_path):
    vast = FakeVast({1: "ok", 2: "ok"}, offers=[offer(1, dph=2.0), offer(2, dph=1.0)])
    result, vast, snap = run(tmp_path, None, vast=vast)
    assert result.winner.machine_id == 2
    only_winner_left(result, vast, snap)


def test_a_box_is_not_handed_over_while_no_listing_shows_it(tmp_path):
    launch, vast, snap = start(tmp_path, {1: "ok"}, spare=0)
    vast.hooks["created"] = vast.hidden.add
    at_call(vast, "instances", 8, vast.hidden.clear)
    result = launch.run({})
    assert result.winner.machine_id == 1
    assert len(vast.names("instances")) >= 8  # not before the listing showed it
    only_winner_left(result, vast, snap)


def test_a_box_that_vanished_after_its_gate_is_not_handed_over(tmp_path):
    result, vast, snap = run(tmp_path, {1: "vanish", 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    assert reasons(result)[1] is None  # gone is not a verdict on the machine
    assert on_disk(snap).status(1) == "gone"
    only_winner_left(result, vast, snap)


def test_a_box_is_not_handed_over_with_its_lease_about_to_end(tmp_path):
    launch, vast, snap = start(tmp_path, {1: "ok"}, expiry=int(time.time()) + 30, min_lease=600)
    with pytest.raises(L.ConfigError, match="lease"):
        launch.run({})
    nothing_left(vast, snap)


# --------------------------------------------- L4, L5: the hand-off and rollback


def test_a_winner_that_cannot_be_published_is_rolled_back(tmp_path, monkeypatch):
    launch, vast, snap = start(tmp_path, {1: "ok", 2: "ok"})

    def publish(*a):
        raise OSError("disk full")

    monkeypatch.setattr(snap, "publish", publish)
    assert isinstance(outcome(launch), OSError)
    nothing_left(vast, snap)
    assert on_disk(snap).winner is None


@pytest.mark.parametrize("when", ["before the rename", "after the rename"])
def test_an_interrupt_during_the_hand_off_is_decided_by_what_is_on_disk(
    tmp_path, monkeypatch, when
):
    launch, vast, snap = start(tmp_path, {1: "ok", 2: "hold"})
    real_write, real_replace = snap._write, os.replace

    def replace(src, dst):
        if when == "before the rename":
            raise KeyboardInterrupt
        real_replace(src, dst)
        raise KeyboardInterrupt

    def write(data):
        if data["winner"] is None:
            return real_write(data)
        monkeypatch.setattr(os, "replace", replace)
        try:
            real_write(data)
        finally:
            monkeypatch.setattr(os, "replace", real_replace)

    monkeypatch.setattr(snap, "_write", write)
    assert isinstance(outcome(launch), KeyboardInterrupt)
    disk = on_disk(snap)
    if when == "before the rename":
        assert disk.winner is None
        nothing_left(vast, snap)
    else:
        # The winner reached the disk, so it is kept; the other box is not.
        assert vast.ours() == [disk.winner["id"]] == disk.live_ids()
    assert L.down(vast, disk, FAST) == 0 and vast.ours() == []


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_while_both_candidates_are_in_the_gate_rolls_back(tmp_path, signum):
    launch, vast, snap = start(tmp_path, {1: "hold", 2: "hold"})
    old = signal.signal(signal.SIGTERM, lambda *_: interrupt())  # what cli.main installs
    try:
        at_call(vast, "instances", 6, lambda: os.kill(os.getpid(), signum))
        assert isinstance(outcome(launch), KeyboardInterrupt)
    finally:
        signal.signal(signal.SIGTERM, old)
    assert vast.gate_started == {1, 2} and vast.creates == 2
    assert all(c.killed for c in launch.prober.checks if c.poll() is None)
    nothing_left(vast, snap)
    assert foreign_untouched(vast)


def test_a_hand_off_that_cannot_be_read_back_keeps_the_box_and_writes_nothing(
    tmp_path, monkeypatch
):
    # The interrupt lands after the rename, and then the disk cannot be read.
    # "Cannot tell" is not "no winner": `run` may be using the box already.
    launch, vast, snap = start(tmp_path, {1: "ok", 2: "hold"})
    real_write, real_replace = snap._write, os.replace

    def replace(src, dst):
        real_replace(src, dst)
        monkeypatch.setattr(L.Snapshot, "load", classmethod(lambda cls, d: 1 / 0))
        raise KeyboardInterrupt

    def write(data):
        if data["winner"] is None:
            return real_write(data)
        monkeypatch.setattr(os, "replace", replace)
        try:
            real_write(data)
        finally:
            monkeypatch.setattr(os, "replace", real_replace)

    monkeypatch.setattr(snap, "_write", write)
    assert isinstance(outcome(launch), KeyboardInterrupt)
    monkeypatch.undo()
    disk = on_disk(snap)
    assert vast.ours() == [disk.winner["id"]]  # the other box was destroyed, by id
    assert len(disk.live_ids()) == 2  # and nothing was written over the hand-off
    assert L.down(vast, disk, FAST) == 0 and vast.ours() == []


def test_a_failure_before_any_hand_off_never_needs_the_disk(tmp_path, monkeypatch):
    launch, vast, snap = start(tmp_path, {1: "hold", 2: "hold"})
    at_call(vast, "instances", 6, interrupt)
    monkeypatch.setattr(L.Snapshot, "load", classmethod(lambda cls, d: 1 / 0))
    assert isinstance(outcome(launch), KeyboardInterrupt)
    monkeypatch.undo()
    nothing_left(vast, snap)


def test_stopping_the_children_comes_out_of_the_same_budget_as_the_deletes(tmp_path):
    launch, vast, snap = start(tmp_path, {1: "hold", 2: "hold"}, rollback=0.4, call_timeout=30)
    vast.sticky.update({1, 2})
    at_call(vast, "instances", 6, interrupt)
    vast.hooks["kill_all"] = lambda _: time.sleep(0.3)
    assert isinstance(outcome(launch), KeyboardInterrupt)
    calls = vast.calls[[n for n, _, _ in vast.calls].index("kill_all") + 1 :]
    timeouts = [kw["timeout"] for n, _, kw in calls if n in ("destroy", "instance")]
    assert timeouts and max(timeouts) <= 0.1 + 0.02  # what was left, not a fresh 0.4


def test_a_signal_during_loser_cleanup_leaves_the_winner_for_down(tmp_path):
    # The winner is in the snapshot by then: it stays (L4), and `down` follows `up`.
    launch, vast, snap = start(tmp_path, {1: "ok", 2: "hold"})
    vast.hooks["destroy"] = lambda iid: interrupt()
    assert isinstance(outcome(launch), KeyboardInterrupt)
    disk = on_disk(snap)
    assert disk.winner is not None and len(vast.ours()) == 2
    vast.hooks.clear()
    assert L.down(vast, disk, FAST) == 0 and vast.ours() == []


def test_a_second_signal_ends_the_rollback_and_down_finishes_it(tmp_path):
    launch, vast, snap = start(tmp_path, {1: "hold", 2: "hold"})
    at_call(vast, "instances", 6, interrupt)
    vast.hooks["destroy"] = lambda iid: interrupt()  # the second signal, in the rollback
    assert isinstance(outcome(launch), KeyboardInterrupt)
    assert len(vast.ours()) == 2 and len(on_disk(snap).live_ids()) == 2
    vast.hooks.clear()
    assert L.down(vast, on_disk(snap), FAST) == 0 and vast.ours() == []


def test_an_interrupt_in_the_middle_of_a_create_leaves_a_note_for_down(tmp_path):
    # The provider rented the box; we never saw the answer. S1 wrote the
    # attempt before the request, so `down` knows to look for its tag.
    launch, vast, snap = start(tmp_path, {1: "ok"}, spare=0)
    vast.hooks["created"] = lambda iid: interrupt()
    assert isinstance(outcome(launch), KeyboardInterrupt)
    (iid,) = vast.ours()
    disk = on_disk(snap)
    assert states(snap) == {1: "open"} and disk.ids() == {}
    vast.hooks.clear()
    assert L.down(vast, disk, FAST) == 0 and vast.ours() == []
    assert states(snap) == {1: "bound"} and on_disk(snap).status(iid) == "gone"


def test_an_exception_in_the_loop_rolls_back(tmp_path):
    launch, vast, snap = start(tmp_path, {1: "hold", 2: "hold"})

    def boom():
        raise RuntimeError("a bug")

    at_call(vast, "instances", 6, boom)
    assert isinstance(outcome(launch), RuntimeError)
    nothing_left(vast, snap)


def test_the_rollback_sends_every_delete_before_it_waits_for_any(tmp_path):
    # D1: with a budget of seconds, the second box must not wait for the first.
    launch, vast, snap = start(tmp_path, {1: "hold", 2: "hold"})
    vast.sticky.add(1)
    at_call(vast, "instances", 6, interrupt)
    assert isinstance(outcome(launch), KeyboardInterrupt)
    rollback = _since_last_listing(vast)
    assert [n for n, _ in rollback[:2]] == ["destroy", "destroy"]
    assert {a for _, a in rollback[:2]} == {1, 2}
    (stuck,) = vast.ours()
    assert on_disk(snap).live_ids() == [stuck]  # not confirmed: left in the snapshot for `down`


def _since_last_listing(vast):
    names = vast.names()
    last = max(i for i, (n, _) in enumerate(names) if n == "instances")
    return [c for c in names[last + 1 :] if c[0] in ("destroy", "instance")]


def test_no_call_in_a_rollback_is_given_more_time_than_the_budget(tmp_path):
    launch, vast, snap = start(tmp_path, {1: "hold", 2: "hold"}, rollback=0.2, call_timeout=30)
    vast.sticky.update({1, 2})
    at_call(vast, "instances", 6, interrupt)
    t0 = time.monotonic()
    assert isinstance(outcome(launch), KeyboardInterrupt)
    assert time.monotonic() - t0 < 2
    names = vast.calls
    last = max(i for i, c in enumerate(names) if c[0] == "instances")
    timeouts = [kw["timeout"] for n, _, kw in names[last + 1 :] if n in ("destroy", "instance")]
    assert timeouts and max(timeouts) <= 0.2
    assert all(kw.get("retries") == 0 for n, _, kw in names[last + 1 :] if n == "destroy")


def test_the_winner_is_kept_when_a_losers_delete_is_not_confirmed(tmp_path):
    # Decision 2 of the design: destroying the healthy box would remove no leak.
    launch, vast, snap = start(tmp_path, {1: "ok", 2: "hold"})
    vast.sticky.add(2)
    result = launch.run({})
    assert result.winner.machine_id == 1
    loser = next(i for i in vast.ours() if i != result.winner.instance_id)
    assert result.left == [loser] and on_disk(snap).live_ids() == sorted(vast.ours())
    vast.sticky.clear()
    assert L.down(vast, on_disk(snap), FAST) == 0 and vast.ours() == []


def test_the_launch_gives_up_when_it_cannot_list_for_too_long(tmp_path):
    launch, vast, snap = start(tmp_path, {1: "hold"}, spare=0)
    vast.hooks["created"] = lambda iid: setattr(vast, "list_fails", 10**9)
    exc = outcome(launch)
    assert isinstance(exc, v.VastError) and "no instance listing" in str(exc)
    nothing_left(vast, snap)  # by id: the rollback needs no listing


# ------------------------------------------------------- S1, S2: the snapshot


def test_no_create_is_sent_when_the_attempt_cannot_be_recorded(tmp_path, monkeypatch):
    launch, vast, snap = start(tmp_path, {1: "ok"})

    def open_attempt(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(snap, "open_attempt", open_attempt)
    assert isinstance(outcome(launch), OSError)
    assert vast.creates == 0 and vast.names("create") == []


def test_an_id_that_cannot_be_recorded_is_destroyed_from_memory(tmp_path, monkeypatch):
    launch, vast, snap = start(tmp_path, {1: "ok"}, spare=0)

    def bind(*a):
        raise OSError("disk full")

    monkeypatch.setattr(snap, "bind", bind)
    assert isinstance(outcome(launch), OSError)
    assert vast.creates == 1 and vast.ours() == []


def test_an_unrecorded_id_whose_delete_is_not_confirmed_is_left_where_down_finds_it(
    tmp_path, monkeypatch
):
    launch, vast, snap = start(tmp_path, {1: "ok"}, spare=0)
    vast.sticky.add(1)

    def bind(*a):
        raise OSError("disk full")

    monkeypatch.setattr(snap, "bind", bind)
    assert isinstance(outcome(launch), OSError)
    (iid,) = vast.ours()
    disk = on_disk(snap)
    assert disk.live_ids() == [iid]  # recorded on the way out, though its attempt is still open
    vast.sticky.clear()
    vast.live[iid]["label"] = "renamed-by-the-box"
    assert L.down(vast, disk, FAST, abandon_unseen=True) == 0 and vast.ours() == []


# ------------------------------------------------------------ D: destroying


def test_destroy_confirmed_reports_what_it_could_not_confirm():
    vast = FakeVast({1: "ok", 2: "ok"})
    a = vast.create(1001, {"label": "x"})
    b = vast.create(1002, {"label": "x"})
    vast.sticky.add(2)
    gone = []
    assert L.destroy_confirmed(vast, [a, b], 0.1, FAST, gone.append) == [b]
    assert gone == [a] and vast.ours() == [b]


def test_nothing_is_started_once_the_budget_is_spent_except_the_first_deletes():
    vast = FakeVast({1: "ok", 2: "ok"})
    a = vast.create(1001, {"label": "x"})
    b = vast.create(1002, {"label": "x"})
    vast.calls.clear()
    assert L.destroy_confirmed(vast, [a, b], 0, FAST) == [a, b]  # sent, and not confirmed
    assert vast.names() == [("destroy", a), ("destroy", b)]


def test_transient_delete_failure_is_retried_until_confirmed():
    vast = FakeVast({1: "ok"})
    a = vast.create(1001, {"label": "x"})
    vast.fail_destroy[1] = 2
    assert L.destroy_confirmed(vast, [a], 1, FAST) == []
    assert len(vast.names("destroy")) == 3 and vast.ours() == []


def test_a_box_that_is_already_gone_is_confirmed_even_if_delete_keeps_failing():
    vast = FakeVast({1: "ok"})
    a = vast.create(1001, {"label": "x"})
    vast.live.pop(a)
    vast.hooks["destroy"] = lambda iid: (_ for _ in ()).throw(v.VastError(500, "boom"))
    assert L.destroy_confirmed(vast, [a], 0.1, FAST) == []


def test_an_error_on_one_id_does_not_stop_the_others():
    vast = FakeVast({1: "ok", 2: "ok"})
    a = vast.create(1001, {"label": "x"})
    b = vast.create(1002, {"label": "x"})

    def lookup(iid):
        if iid == a:
            raise v.VastError(503, "unavailable")

    vast.hooks["instance"] = lookup
    assert L.destroy_confirmed(vast, [a, b], 0.1, FAST) == [a]  # a lookup error is not "gone"
    assert vast.ours() == []


def test_a_delete_that_cannot_be_recorded_is_not_confirmed():
    vast = FakeVast({1: "ok", 2: "ok"})
    a = vast.create(1001, {"label": "x"})
    b = vast.create(1002, {"label": "x"})
    seen = []

    def on_gone(iid):
        if iid == a:
            raise OSError("disk full")
        seen.append(iid)

    assert L.destroy_confirmed(vast, [a, b], 0.1, FAST, on_gone) == [a]
    assert seen == [b]


# ------------------------------------------------------------ T: teardown


def up(tmp_path, behaviour, **kw):
    result, vast, snap = run(tmp_path, behaviour, **kw)
    return result, vast, on_disk(snap)


def test_down_destroys_the_winner_and_closes_the_run(tmp_path):
    result, vast, snap = up(tmp_path, {1: "ok"}, spare=0)
    assert L.down(vast, snap, FAST) == 0
    assert vast.ours() == [] and on_disk(snap).closed and foreign_untouched(vast)
    vast.calls.clear()
    assert L.down(vast, on_disk(snap), FAST) == 0  # again: it looks, and finds nothing
    assert vast.names() == [("instances", None)] and on_disk(snap).closed


def test_a_box_that_shows_up_after_its_run_was_closed_is_still_destroyed(tmp_path):
    # The create got no answer; 60 s later nothing was visible and the run
    # was closed with the attempt abandoned. Then the box appears. Running
    # `down` again must find it, in this directory as from an empty one.
    result, vast, snap = up(tmp_path, {1: "lost_late", 2: "ok"})
    assert L.down(vast, snap, FAST, abandon_unseen=True) == 0
    assert on_disk(snap).closed and states(snap)[1] == "unseen"
    (late,) = vast.ours()
    vast.revealed = True
    vast.sticky.add(1)
    assert L.down(vast, on_disk(snap), FAST) >= 1
    disk = on_disk(snap)
    assert disk.live_ids() == [late] and not disk.closed  # a run with a live box is not closed
    vast.sticky.clear()
    assert L.down(vast, disk, FAST) == 0 and vast.ours() == [] and on_disk(snap).closed


def test_down_destroys_by_id_even_if_the_box_relabelled_itself(tmp_path):
    result, vast, snap = up(tmp_path, {1: "ok"}, spare=0)
    vast.live[result.winner.instance_id]["label"] = "renamed-by-the-box"
    assert L.down(vast, snap, FAST) == 0 and vast.ours() == []


def test_down_never_touches_another_run_in_the_same_namespace(tmp_path):
    result, vast, snap = up(tmp_path, {1: "ok"}, spare=0)
    assert L.down(vast, snap, FAST) == 0
    assert foreign_untouched(vast)  # one of them is tagged for another run of the namespace


def test_an_empty_listing_does_not_confirm_that_a_known_box_is_gone(tmp_path):
    # S3: the delete had no effect and the listing is stale. Only a lookup of
    # the id may say the box is gone.
    result, vast, snap = up(tmp_path, {1: "ok"}, spare=0)
    vast.sticky.add(1)
    vast.empty_listings = 10**9
    assert L.down(vast, snap, FAST) >= 1
    disk = on_disk(snap)
    assert disk.live_ids() == [result.winner.instance_id] and not disk.closed
    vast.sticky.clear()
    vast.empty_listings = 0
    assert L.down(vast, disk, FAST) == 0 and on_disk(snap).closed


def test_down_is_not_clean_without_a_listing_taken_after_its_last_delete(tmp_path):
    # T1: the id is confirmed gone, but nothing says no other box carries the tag.
    result, vast, snap = up(tmp_path, {1: "ok"}, spare=0)
    vast.list_fails = 10**9
    assert L.down(vast, snap, FAST) == 1
    assert vast.ours() == [] and not on_disk(snap).closed
    vast.list_fails = 0
    assert L.down(vast, on_disk(snap), FAST) == 0 and on_disk(snap).closed


def test_a_box_that_shows_up_while_another_is_being_deleted_is_found(tmp_path):
    # T1 again: the first listing shows only A. B becomes visible during A's
    # delete. A listing after that delete is what finds it.
    result, vast, snap = up(tmp_path, {1: "ok", 2: "lost_late"}, spare=1)
    assert len(vast.ours()) == 2 and snap.live_ids() == [result.winner.instance_id]
    vast.hooks["destroy"] = lambda iid: setattr(vast, "revealed", True)
    assert L.down(vast, snap, FAST) == 0
    assert vast.ours() == [] and on_disk(snap).closed and not on_disk(snap).open_attempts()


def test_an_id_found_by_tag_is_recorded_before_it_is_deleted(tmp_path):
    # S2: the delete fails, then the box changes its label. The next `down`,
    # a new process, must still know the id.
    result, vast, snap = up(tmp_path, {1: "ok", 2: "lost_late"}, spare=1)
    lost = next(i for i in vast.ours() if i != result.winner.instance_id)
    assert snap.status(lost) is None
    vast.revealed = True
    vast.sticky.add(2)
    assert L.down(vast, snap, FAST) >= 1
    assert on_disk(snap).live_ids() == [lost]
    vast.live[lost]["label"] = "renamed-by-the-box"
    vast.sticky.clear()
    assert L.down(vast, on_disk(snap), FAST) == 0 and vast.ours() == []


def test_an_unseen_create_keeps_down_failing_until_it_is_abandoned(tmp_path):
    result, vast, snap = up(tmp_path, {1: "ghost", 2: "ok"})
    assert L.down(vast, snap, FAST) == 1  # T2: without the flag an open attempt counts
    disk = on_disk(snap)
    assert vast.ours() == [] and not disk.closed and list(disk.open_attempts()) == [1]
    assert L.down(vast, disk, FAST, abandon_unseen=True) == 0
    assert states(snap)[1] == "unseen" and on_disk(snap).closed


def test_abandoning_waits_until_the_attempt_is_old_enough(tmp_path):
    launch, vast, snap = start(tmp_path, {1: "ok"}, spare=0)
    vast.hooks["created"] = lambda iid: interrupt()
    assert isinstance(outcome(launch), KeyboardInterrupt)
    vast.hooks.clear()
    vast.live.clear()  # no box after all
    sent = on_disk(snap).attempts()[1]["sent"]
    slow = L.Policy(**{**FAST.__dict__, "settle": 0.4})
    assert L.down(vast, on_disk(snap), slow, abandon_unseen=True) == 0
    assert time.time() >= sent + 0.4  # T2: the listing that counts was taken after that


def test_a_box_that_turns_up_during_the_wait_is_destroyed_not_abandoned(tmp_path):
    result, vast, snap = up(tmp_path, {1: "lost_late", 2: "ok"})
    assert len(vast.ours()) == 2
    slow = L.Policy(**{**FAST.__dict__, "settle": 0.3})
    at_call(vast, "instances", 3, lambda: setattr(vast, "revealed", True))
    assert L.down(vast, snap, slow, abandon_unseen=True) == 0
    assert vast.ours() == [] and states(snap)[1] == "bound"


def test_down_by_tag_alone_destroys_the_runs_boxes_and_nothing_else(tmp_path):
    # T4: a run whose state directory is gone.
    result, vast, snap = up(tmp_path, {1: "ok"}, spare=0)
    before = snap.path.read_text()
    assert L.down(vast, Snapshot.ephemeral(RUN_ID), FAST) == 0
    assert vast.ours() == [] and foreign_untouched(vast)
    assert snap.path.read_text() == before  # the directory's snapshot was not touched
    assert L.down(vast, Snapshot.ephemeral(OTHER_RUN), FAST) == 0
    assert 9003 not in vast.live  # asked for by its run id, so that one goes too


def test_down_by_tag_alone_cannot_find_a_box_that_changed_its_label(tmp_path):
    # Stated, not hidden: this is what the snapshot's ids, and the reaper, are for.
    result, vast, snap = up(tmp_path, {1: "ok"}, spare=0)
    vast.live[result.winner.instance_id]["label"] = "renamed-by-the-box"
    assert L.down(vast, Snapshot.ephemeral(RUN_ID), FAST) == 0
    assert vast.ours() == [result.winner.instance_id]
    assert L.down(vast, snap, FAST) == 0 and vast.ours() == []


def test_down_does_not_report_done_when_it_could_not_record(tmp_path, monkeypatch):
    result, vast, snap = up(tmp_path, {1: "ok"}, spare=0)

    def write(data):
        raise OSError("disk full")

    monkeypatch.setattr(snap, "_write", write)
    assert L.down(vast, snap, FAST) >= 1
    assert vast.ours() == []  # the box is destroyed all the same
    assert not on_disk(snap).closed


def test_a_dead_output_stream_does_not_stop_a_teardown(tmp_path, monkeypatch):
    result, vast, snap = up(tmp_path, {1: "ok"}, spare=0)

    def dead(*a, **kw):
        raise BrokenPipeError

    monkeypatch.setattr("builtins.print", dead)
    assert L.down(vast, snap, FAST) == 0 and vast.ours() == []


# -------------------------------------------------------------- R: the reaper


def box(vast, iid, label, age_min=1.0, **extra):
    vast.live[iid] = {
        "id": iid,
        "label": label,
        "machine": iid,
        "start_date": time.time() - age_min * 60,
        **extra,
    }


def reaper_world():
    vast = FakeVast({})
    now = int(time.time())
    box(vast, 1, make_tag(RUN_ID, 1, now + 600))  # in its lease
    box(vast, 2, make_tag(RUN_ID, 2, now - 600))  # lease ended
    box(vast, 3, make_tag(OTHER_RUN, 1, now - 30))  # ended, but inside the clock margin
    box(vast, 4, make_tag("dev.x.0123456789ab", 1, now - 600))  # another namespace, ended
    box(vast, 5, "someones-box")  # no tag
    box(vast, 6, make_tag(RUN_ID, 3, now + 600), age_min=500)  # tag fine, far too old
    return vast


def test_a_reaper_without_destroy_changes_nothing():
    for kw in ({"namespace": "quack-ci"}, {"max_age_min": 90}):
        vast = reaper_world()
        assert L.reap(vast, FAST, **kw) == 0
        assert vast.names("destroy") == [] and len(vast.ours()) == 6  # R4


def test_the_namespace_reaper_destroys_only_expired_boxes_of_its_namespace():
    vast = reaper_world()
    assert L.reap(vast, FAST, namespace="quack-ci", destroy=True) == 0
    assert vast.ours() == [1, 3, 4, 5, 6]
    assert foreign_untouched(vast)


def test_the_exclusive_reaper_destroys_what_is_untagged_expired_or_too_old():
    vast = reaper_world()
    for iid in list(vast.live):
        if iid >= 9000:  # on the exclusive account there is nothing foreign
            vast.live.pop(iid)
    assert L.reap(vast, FAST, max_age_min=90, destroy=True) == 0
    assert vast.ours() == [1, 3]  # in its lease; and expired only within the clock margin
    assert len(vast.names("instances")) == 2  # R3: two looks before an untagged box goes


def test_an_instance_that_shows_its_tag_at_the_second_look_is_left_alone():
    # R3: a box seconds old, listed before its label is.
    vast = FakeVast({})
    vast.live.clear()
    box(vast, 1, make_tag(RUN_ID, 1, EXPIRY))
    vast.unlabelled.add(1)
    assert (
        L.reap(vast, FAST, max_age_min=90, destroy=True, sleep=lambda s: vast.unlabelled.clear())
        == 0
    )
    assert vast.ours() == [1] and vast.names("destroy") == []


def test_an_untagged_instance_is_destroyed_only_after_the_second_look():
    vast = FakeVast({})
    vast.live.clear()
    box(vast, 1, None)
    waits = []
    assert (
        L.reap(vast, L.Policy(reap_recheck=30), max_age_min=90, destroy=True, sleep=waits.append)
        == 0
    )
    assert waits == [30] and vast.ours() == []


def test_an_untagged_instance_that_is_gone_at_the_second_look_needs_nothing():
    vast = FakeVast({})
    vast.live.clear()
    box(vast, 1, None)
    assert L.reap(vast, FAST, max_age_min=90, destroy=True, sleep=lambda s: vast.live.clear()) == 0
    assert vast.names("destroy") == []


def test_a_tagged_instance_without_a_start_time_is_kept_and_reported(capsys):
    vast = FakeVast({})
    vast.live.clear()
    box(vast, 1, make_tag(RUN_ID, 1, EXPIRY), start_date=None)
    assert L.reap(vast, FAST, max_age_min=90, destroy=True) == 0
    assert vast.ours() == [1] and "age rule cannot apply" in capsys.readouterr().out


def test_the_reaper_counts_a_delete_that_is_not_confirmed():
    vast = reaper_world()
    vast.sticky.add(2)
    assert L.reap(vast, FAST, namespace="quack-ci", destroy=True) == 1  # R5


def test_a_failed_second_look_does_not_spare_what_the_first_look_chose():
    vast = reaper_world()
    for iid in [i for i in vast.live if i >= 9000]:
        vast.live.pop(iid)

    def second_look_fails(seconds):
        vast.list_fails = 1

    assert L.reap(vast, FAST, max_age_min=90, destroy=True, sleep=second_look_fails) == 1  # R5
    assert vast.ours() == [1, 3, 5]  # the expired and the too old went; the untagged was not judged


def test_an_instance_tagged_at_the_second_look_is_still_judged_by_its_lease():
    vast = FakeVast({})
    vast.live.clear()
    box(vast, 1, make_tag(RUN_ID, 1, int(time.time()) - 600))
    vast.unlabelled.add(1)
    assert (
        L.reap(vast, FAST, max_age_min=90, destroy=True, sleep=lambda s: vast.unlabelled.clear())
        == 0
    )
    assert vast.ours() == []


def test_the_reaper_fails_on_a_listing_it_cannot_read():
    vast = reaper_world()
    vast.list_fails = 1
    with pytest.raises(v.VastError):
        L.reap(vast, FAST, namespace="quack-ci", destroy=True)
    assert vast.names("destroy") == []


def test_an_exclusive_reap_needs_a_maximum_age():
    with pytest.raises(L.ConfigError):
        L.reap(reaper_world(), FAST, destroy=True)
