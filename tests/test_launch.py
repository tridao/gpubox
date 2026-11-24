"""The race, teardown, the reaper and renewal, against a fake provider.

Every test that rents states which boxes are left afterwards, and `run`
checks the peak number of billable boxes and that nothing foreign was touched.
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
    FakeProber,
    FakeVast,
    foreign_untouched,
    offer,
    start,
)

from gpubox import launch as L
from gpubox import policy as pol
from gpubox import vast as v
from gpubox.provider import ProviderError
from gpubox.state import State, make_tag, parse_tag


def run(tmp_path, behaviour, **kw):
    launch, vast, state = start(tmp_path, behaviour, **kw)
    try:
        result = launch.run({})
    finally:
        assert foreign_untouched(vast)
        assert vast.peak <= launch.policy.max_live
    return result, vast, state


def outcome(launch):
    """What `launch.run` raised, or None. Unlike `pytest.raises`, this also
    holds on to a KeyboardInterrupt, which would otherwise end the whole
    test session instead of failing one test."""
    try:
        launch.run({})
    except BaseException as exc:
        return exc
    return None


def on_disk(state):
    return State.load(state.path.parent)


def reasons(result):
    return {a.machine_id: a.reason for a in result.attempts}


def observed(result):
    obs = pol.Observations(None)
    L.observe(result, SPEC, "r", obs)
    return {(o["machine_id"], o["kind"], o["reason_code"]) for o in obs.rows}


def only_winner_left(result, vast, state):
    """The provider has exactly the winner, and the state file names it."""
    assert vast.ours() == [result.winner.instance_id]
    assert on_disk(state).winner["id"] == result.winner.instance_id
    assert not result.left


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


def after(vast, name):
    """The (name, argument) of the calls made after the last call of `name`."""
    names = vast.names()
    last = max(i for i, (n, _) in enumerate(names) if n == name)
    return names[last + 1 :]


# ------------------------------------------------------------------ the race


def test_failed_machine_is_replaced_and_blamed(tmp_path):
    result, vast, state = run(tmp_path, {1: "bad_gpu", 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    only_winner_left(result, vast, state)
    assert observed(result) == {(1, "failed_admission", "cuda_init"), (2, "healthy", None)}


def test_race_keeps_one_and_does_not_blame_the_loser(tmp_path):
    result, vast, state = run(tmp_path, {1: "ok", 2: "ok", 3: "ok"})
    assert result.winner.machine_id in (1, 2)
    assert 3 not in reasons(result)  # never needed
    only_winner_left(result, vast, state)
    assert {kind for _, kind, _ in observed(result)} == {"healthy"}
    assert vast.names("instances") == [("instances", None)]  # one listing: the sweep at the end


def test_a_loser_still_in_its_gate_is_stopped_and_not_blamed(tmp_path):
    launch, vast, state = start(tmp_path, {1: "late", 2: "ok"})
    result = launch.run({})
    assert result.winner.machine_id == 2 and 1 in vast.gate_started
    assert reasons(result)[1] is None
    assert all(c.killed for c in launch.prober.checks if c.poll() is None)
    only_winner_left(result, vast, state)


def test_nothing_came_up_rolls_back_and_blames_nobody(tmp_path):
    # Could be Docker Hub or the image: no control, so no verdict on the machines.
    result, vast, state = run(tmp_path, {1: "stuck", 2: "bad_gpu", 3: "cdi"})
    assert result.winner is None and vast.ours() == []
    assert observed(result) == set()


@pytest.mark.parametrize(
    "kind,reason", [("stuck", "unreachable"), ("exited", "unreachable"), ("cdi", "runtime_error")]
)
def test_startup_failures_carry_a_reason_code(tmp_path, kind, reason):
    result, vast, state = run(tmp_path, {1: kind, 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    assert reasons(result)[1] == reason
    only_winner_left(result, vast, state)


def test_a_box_whose_loading_status_stops_changing_is_replaced_before_the_startup_timeout(
    tmp_path,
):
    t0 = time.monotonic()
    result, vast, state = run(
        tmp_path, {1: "stuck", 2: "ok"}, spare=0, startup_timeout=30, stall_timeout=0.2
    )
    assert result.winner.machine_id == 2 and time.monotonic() - t0 < 5
    assert reasons(result)[1] == "unreachable"
    assert "stalled" in next(a.detail for a in result.attempts if a.machine_id == 1)
    only_winner_left(result, vast, state)


def test_a_box_that_keeps_reporting_progress_is_not_taken_for_stalled(tmp_path):
    # Slow is not stuck: it gets the whole startup timeout.
    result, vast, state = run(
        tmp_path, {1: "pulling", 2: "ok"}, spare=0, startup_timeout=0.6, stall_timeout=0.2
    )
    assert result.winner.machine_id == 2
    slow = next(a for a in result.attempts if a.machine_id == 1)
    assert "not reachable after 0.6s" in slow.detail and "stalled" not in slow.detail
    only_winner_left(result, vast, state)


def test_a_driver_too_old_for_the_image_is_not_blamed_and_its_like_are_skipped(tmp_path):
    offers = [offer(1, dph=0.5, cuda=12.8), offer(2, dph=0.6, cuda=12.9),
              offer(3, dph=0.7), offer(4, dph=0.8, cuda=13.0)]  # fmt: skip
    vast = FakeVast({1: "old_driver", 2: "ok", 3: "ok", 4: "ok"}, offers)
    result, vast, state = run(tmp_path, None, vast=vast, spare=0)
    assert result.winner.machine_id == 3  # says nothing about its driver: still tried
    assert [m for n, m in vast.names("create")] == [1, 3]  # 2 is as old as 1
    assert reasons(result)[1] is None and "--min-cuda 13.0" in result.attempts[0].detail
    assert observed(result) == {(3, "healthy", None)}
    only_winner_left(result, vast, state)


def test_stale_offer_is_skipped_without_blame(tmp_path):
    result, vast, state = run(tmp_path, {1: "stale", 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    assert reasons(result)[1] is None and result.attempts[0].state == "refused"
    assert vast.creates == 1
    only_winner_left(result, vast, state)


def test_offers_that_were_already_gone_do_not_use_up_the_attempts(tmp_path):
    # Seen live: two stale offers in a row. They rented nothing, so they do
    # not count against --max-attempts; boxes that were rented do.
    behaviour = {1: "stale", 2: "stale", 3: "stale", 4: "bad_gpu", 5: "bad_gpu", 6: "ok", 7: "ok"}
    result, vast, state = run(tmp_path, behaviour, spare=0, max_attempts=3)
    assert result.winner.machine_id == 6 and vast.creates == 3
    assert [a.state for a in result.attempts][:3] == ["refused"] * 3
    only_winner_left(result, vast, state)


def test_a_silent_provider_api_is_not_held_against_the_machine(tmp_path):
    result, vast, state = run(tmp_path, {1: "api_down", 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    assert reasons(result)[1] is None  # we never heard about the box: no evidence
    only_winner_left(result, vast, state)


def test_max_attempts_bounds_the_spend(tmp_path):
    result, vast, state = run(tmp_path, dict.fromkeys(range(1, 10), "bad_gpu"), max_attempts=3)
    assert result.winner is None and vast.creates == 3 and vast.ours() == []


def test_timeout_rolls_back_a_candidate_that_would_have_been_healthy_later(tmp_path):
    result, vast, state = run(tmp_path, {1: "hold"}, overall_timeout=0.2, startup_timeout=5)
    assert result.winner is None and vast.ours() == []


def test_no_box_is_rented_once_the_renting_window_has_closed(tmp_path):
    result, vast, state = run(tmp_path, {1: "ok"}, overall_timeout=0)
    assert result.winner is None and vast.creates == 0


def test_the_window_is_checked_again_before_each_request(tmp_path):
    # The first create of the pass takes until after the cut-off. The second
    # is not sent, although the pass began in time.
    launch, vast, state = start(tmp_path, {1: "hold", 2: "hold"}, overall_timeout=0.2)
    at_call(vast, "create", 1, lambda: time.sleep(0.3))
    result = launch.run({})
    assert result.winner is None and vast.creates == 1 and vast.ours() == []


# -------------------------------------------------------------- image errors


def test_an_image_that_no_host_can_pull_is_a_configuration_error_after_two_hosts(tmp_path):
    launch, vast, state = start(tmp_path, dict.fromkeys(range(1, 6), "image"))
    with pytest.raises(L.ConfigError, match="cannot be pulled"):
        launch.run({})
    assert vast.creates == 2  # two hosts said so: no third is rented
    assert vast.ours() == []


def test_with_one_slot_the_second_host_to_say_so_settles_it(tmp_path):
    launch, vast, state = start(tmp_path, dict.fromkeys(range(1, 6), "image"), spare=0)
    with pytest.raises(L.ConfigError, match="cannot be pulled"):
        launch.run({})
    assert vast.creates == 2 and vast.ours() == []


@pytest.mark.parametrize("behaviour", [{1: "ok", 2: "image"}, {1: "image", 2: "ok"}])
def test_one_host_that_cannot_pull_the_image_is_just_a_failed_box(tmp_path, behaviour):
    result, vast, state = run(tmp_path, behaviour)
    assert result.winner is not None
    assert all(r is None for r in reasons(result).values())  # and nobody is blamed for it
    only_winner_left(result, vast, state)


def test_a_host_that_cannot_pull_the_image_is_replaced(tmp_path):
    result, vast, state = run(tmp_path, {1: "image", 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2 and vast.creates == 2
    only_winner_left(result, vast, state)


def test_an_image_whose_sshd_refuses_our_key_fails_fast_not_at_the_startup_timeout(tmp_path):
    # Seen live: both boxes "running" and refusing the key for ten minutes each.
    launch, vast, state = start(tmp_path, dict.fromkeys(range(1, 6), "nokey"), startup_timeout=60)
    t0 = time.monotonic()
    with pytest.raises(L.ConfigError, match="refuses the run's key"):
        launch.run({})
    assert time.monotonic() - t0 < 5 and vast.creates == 2 and vast.ours() == []


def test_one_box_that_refuses_our_key_is_replaced_and_nobody_is_blamed(tmp_path):
    result, vast, state = run(tmp_path, {1: "nokey", 2: "ok"}, spare=0, startup_timeout=60)
    assert result.winner.machine_id == 2 and reasons(result)[1] is None
    only_winner_left(result, vast, state)


def test_two_boxes_whose_host_is_unknown_are_not_two_witnesses(tmp_path):
    # They may be the same host. A third, working, host is still tried.
    offers = [{**offer(m), "host_id": None} for m in (1, 2)] + [offer(3)]
    vast = FakeVast({1: "image", 2: "image", 3: "ok"}, offers=offers)
    result, vast, state = run(tmp_path, None, vast=vast)
    assert result.winner.machine_id == 3
    only_winner_left(result, vast, state)


def test_if_nothing_came_up_and_a_host_blamed_the_image_that_is_what_is_reported(tmp_path):
    launch, vast, state = start(tmp_path, {1: "image"}, spare=0)
    with pytest.raises(L.ConfigError, match="cannot be pulled"):
        launch.run({})
    assert vast.ours() == []


# ------------------------------------------------------------------- the cap


def test_a_create_with_no_answer_holds_its_slot_instead_of_being_replaced(tmp_path):
    result, vast, state = run(tmp_path, {1: "ghost", 2: "ok", 3: "ok"})
    assert result.winner.machine_id == 2
    assert vast.creates == 1  # machine 3 was never rented in the ghost's place
    assert [a.seq for a in result.unanswered] == [1]
    assert vast.ours() == [result.winner.instance_id]


def test_two_creates_with_no_answer_stop_the_renting(tmp_path):
    # Both boxes may exist unseen. Renting more would break the cap for boxes
    # nobody can see, so the launch ends without a winner.
    t0 = time.monotonic()
    result, vast, state = run(tmp_path, {1: "ghost", 2: "ghost", 3: "ok", 4: "ok"})
    assert result.winner is None and vast.creates == 0
    assert [a.seq for a in result.unanswered] == [1, 2] and time.monotonic() - t0 < 2


def test_a_box_whose_create_gave_no_answer_is_destroyed_by_its_tag_when_up_ends(tmp_path):
    launch, vast, state = start(tmp_path, {1: "lost_late", 2: "ok"})
    vast.revealed = True  # the listing shows it; only the create's answer was lost
    result = launch.run({})
    assert result.winner.machine_id == 2 and vast.creates == 2
    only_winner_left(result, vast, state)


def test_a_box_the_listing_does_not_show_yet_is_left_for_down(tmp_path):
    result, vast, state = run(tmp_path, {1: "lost_late", 2: "ok"})
    assert len(vast.ours()) == 2 and [a.seq for a in result.unanswered] == [1]
    vast.revealed = True
    assert L.down(vast, on_disk(state), FAST) == [] and vast.ours() == []


def test_a_failed_box_whose_delete_is_not_confirmed_keeps_its_slot(tmp_path):
    launch, vast, state = start(tmp_path, {1: "bad_gpu", 2: "ok"}, spare=0)
    vast.sticky.add(1)
    result = launch.run({})
    assert result.winner is None and vast.creates == 1  # never two billable boxes at once
    (stuck,) = vast.ours()
    assert result.left == [stuck] and on_disk(state).ids == [stuck]


# ---------------------------------------------------------------- the winner


def test_the_cheaper_of_two_boxes_healthy_in_the_same_pass_wins(tmp_path):
    # Rented second, because it was only let back in from the blocklist, but cheaper.
    vast = FakeVast({1: "ok", 2: "ok"}, offers=[offer(1, dph=1.2), offer(2, dph=1.0)])
    launch, vast, state = start(tmp_path, None, vast=vast)
    soft = {"until": time.time() + 60, "class": "soft", "reason": "busy_gpu"}
    result = launch.run({2: soft})
    assert [a.machine_id for a in result.attempts] == [1, 2]
    assert result.winner.machine_id == 2
    only_winner_left(result, vast, state)


def test_an_offer_above_the_price_cap_is_never_rented_whatever_the_search_returns(tmp_path):
    # The cap is part of the search request, but what comes back is what gets
    # rented: it is checked here too. SPEC allows $1.25 for its one GPU.
    offers = [offer(1, dph=2.0), offer(2, dph=1.26), offer(3, dph=1.25)]
    vast = FakeVast({1: "ok", 2: "ok", 3: "ok"}, offers=offers)
    result, vast, state = run(tmp_path, None, vast=vast)
    assert result.winner.machine_id == 3 and vast.names("create") == [("create", 3)]


def test_offers_are_tried_cheapest_first_whatever_order_the_search_returns(tmp_path):
    offers = [offer(1, dph=1.2), offer(2, dph=0.5), offer(3, dph=0.9)]
    vast = FakeVast({1: "ok", 2: "ok", 3: "ok"}, offers=offers)
    result, vast, state = run(tmp_path, None, vast=vast, spare=0)
    assert vast.names("create") == [("create", 2)]


def test_a_running_box_with_an_unreadable_endpoint_does_not_cost_the_other_one(tmp_path):
    launch, vast, state = start(tmp_path, {1: "ok", 2: "ok"})
    real = vast.instance

    def lookup(iid, **kw):
        row = real(iid, **kw)
        if row and vast.live.get(iid, {}).get("machine") == 1:
            row["ports"] = {"22/tcp": [{"HostPort": "not-a-port"}]}
        return row

    vast.instance = lookup
    result = launch.run({})
    assert result.winner.machine_id == 2
    only_winner_left(result, vast, state)


def test_a_box_that_vanished_after_its_gate_is_not_handed_over(tmp_path):
    result, vast, state = run(tmp_path, {1: "vanish", 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    assert reasons(result)[1] is None  # gone is not a verdict on the machine
    only_winner_left(result, vast, state)


def test_a_box_is_not_handed_over_with_its_lease_about_to_end(tmp_path):
    launch, vast, state = start(tmp_path, {1: "ok"}, expiry=int(time.time()) + 30, min_lease=600)
    with pytest.raises(L.ConfigError, match="lease"):
        launch.run({})
    assert vast.ours() == []


def test_the_winner_is_recorded_with_what_run_and_renew_need(tmp_path):
    result, vast, state = run(tmp_path, {1: "ok"}, spare=0)
    w = on_disk(state).winner
    assert w == {
        "id": result.winner.instance_id, "attempt": 1, "host": "10.0.0.1", "port": 2222,
        "machine": 1, "host_id": 1, "gpu": "RTX 5090",
    }  # fmt: skip
    assert parse_tag(vast.live[w["id"]]["label"]).expiry == EXPIRY


def test_the_winner_is_kept_when_a_losers_delete_is_not_confirmed(tmp_path):
    # Destroying the healthy box would remove no leak.
    launch, vast, state = start(tmp_path, {1: "ok", 2: "hold"})
    vast.sticky.add(2)
    result = launch.run({})
    assert result.winner.machine_id == 1
    loser = next(i for i in vast.ours() if i != result.winner.instance_id)
    assert result.left == [loser]
    vast.sticky.clear()
    assert L.down(vast, on_disk(state), FAST) == [] and vast.ours() == []


# ------------------------------------------------ the hand-off and the rollback


def test_a_winner_that_cannot_be_published_is_rolled_back(tmp_path, monkeypatch):
    launch, vast, state = start(tmp_path, {1: "ok", 2: "ok"})

    def publish(*a):
        raise OSError("disk full")

    monkeypatch.setattr(state, "publish", publish)
    assert isinstance(outcome(launch), OSError)
    assert vast.ours() == [] and on_disk(state).winner is None


@pytest.mark.parametrize("when", ["before the rename", "after the rename"])
def test_an_interrupt_during_the_hand_off_is_decided_by_what_is_on_disk(
    tmp_path, monkeypatch, when
):
    launch, vast, state = start(tmp_path, {1: "ok", 2: "hold"})
    real_replace = os.replace

    def replace(src, dst):
        if '"winners": []' in open(src).read():
            return real_replace(src, dst)
        if when == "after the rename":
            real_replace(src, dst)
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", replace)
    assert isinstance(outcome(launch), KeyboardInterrupt)
    monkeypatch.undo()
    disk = on_disk(state)
    if when == "before the rename":
        assert disk.winner is None and vast.ours() == []
    else:
        assert vast.ours() == [disk.winner["id"]]  # it reached the disk, so it is kept
    assert L.down(vast, disk, FAST) == [] and vast.ours() == []


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_while_both_candidates_are_in_the_gate_rolls_back(tmp_path, signum):
    launch, vast, state = start(tmp_path, {1: "hold", 2: "hold"})
    old = signal.signal(signal.SIGTERM, lambda *_: interrupt())  # what cli.main installs
    try:
        at_call(vast, "gate_poll", 6, lambda: os.kill(os.getpid(), signum))
        assert isinstance(outcome(launch), KeyboardInterrupt)
    finally:
        signal.signal(signal.SIGTERM, old)
    assert vast.gate_started == {1, 2} and vast.creates == 2
    assert all(c.killed for c in launch.prober.checks if c.poll() is None)
    assert vast.ours() == [] and foreign_untouched(vast)


def test_a_signal_during_loser_cleanup_leaves_the_winner_for_down(tmp_path):
    # The winner is in the state file by then: it stays, and `down` follows `up`.
    launch, vast, state = start(tmp_path, {1: "ok", 2: "hold"})
    vast.hooks["destroy"] = lambda iid: interrupt()
    assert isinstance(outcome(launch), KeyboardInterrupt)
    disk = on_disk(state)
    assert disk.winner is not None and len(vast.ours()) == 2
    vast.hooks.clear()
    assert L.down(vast, disk, FAST) == [] and vast.ours() == []


def test_a_second_signal_ends_the_rollback_and_down_finishes_it(tmp_path):
    launch, vast, state = start(tmp_path, {1: "hold", 2: "hold"})
    at_call(vast, "gate_poll", 6, interrupt)
    vast.hooks["destroy"] = lambda iid: interrupt()  # the second signal, in the rollback
    assert isinstance(outcome(launch), KeyboardInterrupt)
    assert len(vast.ours()) == 2 and len(on_disk(state).ids) == 2
    vast.hooks.clear()
    assert L.down(vast, on_disk(state), FAST) == [] and vast.ours() == []


def test_an_interrupt_in_the_middle_of_a_create_leaves_a_box_that_down_finds_by_its_tag(tmp_path):
    # The provider rented the box; we never saw the answer, so there is no id
    # anywhere. The tag went with the request.
    launch, vast, state = start(tmp_path, {1: "ok"}, spare=0)
    vast.hooks["created"] = lambda iid: interrupt()
    assert isinstance(outcome(launch), KeyboardInterrupt)
    vast.hooks.clear()
    assert vast.ours() == []  # the rollback's own listing already found it
    assert on_disk(state).ids == []


def test_an_exception_in_the_loop_rolls_back(tmp_path):
    launch, vast, state = start(tmp_path, {1: "hold", 2: "hold"})

    def boom():
        raise RuntimeError("a bug")

    at_call(vast, "gate_poll", 6, boom)
    assert isinstance(outcome(launch), RuntimeError)
    assert vast.ours() == []


def test_an_id_that_cannot_be_written_to_the_state_file_is_destroyed_from_memory(
    tmp_path, monkeypatch
):
    # And without help from the listing, which does not show the box yet.
    launch, vast, state = start(tmp_path, {1: "ok"}, spare=0)
    vast.hooks["created"] = vast.hidden.add

    def add_id(iid):
        raise OSError("disk full")

    monkeypatch.setattr(state, "add_id", add_id)
    assert isinstance(outcome(launch), OSError)
    assert vast.creates == 1 and vast.ours() == []


def test_the_rollback_sends_every_delete_before_it_waits_for_any(tmp_path):
    # With a budget of seconds, the second box must not wait for the first.
    launch, vast, state = start(tmp_path, {1: "hold", 2: "hold"})
    vast.sticky.add(1)
    at_call(vast, "gate_poll", 6, interrupt)
    assert isinstance(outcome(launch), KeyboardInterrupt)
    rollback = [c for c in after(vast, "kill_all") if c[0] in ("destroy", "instance")]
    assert [n for n, _ in rollback[:2]] == ["destroy", "destroy"]
    assert {a for _, a in rollback[:2]} == {1, 2}
    assert len(vast.ours()) == 1  # not confirmed: `down` retries it from the state file


def test_stopping_the_children_comes_out_of_the_same_budget_as_the_deletes(tmp_path):
    launch, vast, state = start(tmp_path, {1: "hold", 2: "hold"}, rollback=0.4, call_timeout=30)
    vast.sticky.update({1, 2})
    at_call(vast, "gate_poll", 6, interrupt)
    vast.hooks["kill_all"] = lambda _: time.sleep(0.3)
    t0 = time.monotonic()
    assert isinstance(outcome(launch), KeyboardInterrupt)
    assert time.monotonic() - t0 < 2
    calls = vast.calls[[n for n, _, _ in vast.calls].index("kill_all") + 1 :]
    timeouts = [kw["timeout"] for n, _, kw in calls if n in ("destroy", "instance")]
    assert timeouts and max(timeouts) <= 0.1 + 0.02  # what was left, not a fresh 0.4
    assert all(kw.get("retries") == 0 for n, _, kw in calls if n == "destroy")


# ---------------------------------------------------------------- destroying


def test_destroy_confirmed_reports_what_it_could_not_confirm():
    vast = FakeVast({1: "ok", 2: "ok"})
    a = vast.create(1001, {"label": "x"})
    b = vast.create(1002, {"label": "x"})
    vast.sticky.add(2)
    assert L.destroy_confirmed(vast, [a, b], 0.1, FAST) == [b]
    assert vast.ours() == [b]


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


# ------------------------------------------------------------------ teardown


def up(tmp_path, behaviour, **kw):
    result, vast, state = run(tmp_path, behaviour, **kw)
    return result, vast, on_disk(state)


def test_down_destroys_the_winner_and_closes_the_run(tmp_path):
    result, vast, state = up(tmp_path, {1: "ok"}, spare=0)
    assert L.down(vast, state, FAST) == []
    assert vast.ours() == [] and on_disk(state).closed and foreign_untouched(vast)
    assert L.down(vast, on_disk(state), FAST) == []  # again: it looks, and finds nothing


def test_down_is_done_only_after_two_listings_that_show_nothing(tmp_path):
    result, vast, state = up(tmp_path, {1: "ok"}, spare=0)
    vast.calls.clear()
    assert L.down(vast, state, FAST) == []
    calls = [n for n, _ in vast.names("destroy", "instances")]
    assert calls[0] == "destroy" and calls.count("instances") == 2
    assert calls.index("destroy") < calls.index("instances")  # both taken after the delete


def test_down_destroys_by_id_even_if_the_box_no_longer_carries_its_tag(tmp_path):
    result, vast, state = up(tmp_path, {1: "ok"}, spare=0)
    vast.live[result.winner.instance_id]["label"] = "renamed"
    assert L.down(vast, state, FAST) == [] and vast.ours() == []


def test_down_never_touches_another_run_in_the_same_namespace(tmp_path):
    result, vast, state = up(tmp_path, {1: "ok"}, spare=0)
    assert L.down(vast, state, FAST) == []
    assert foreign_untouched(vast)  # one of them is tagged for another run of the namespace


def test_a_stale_empty_listing_does_not_make_down_report_a_known_box_gone(tmp_path):
    # The delete had no effect and the listing is stale. Only a lookup of the
    # id may say the box is gone.
    result, vast, state = up(tmp_path, {1: "ok"}, spare=0)
    vast.sticky.add(1)
    vast.empty_listings = 10**9
    assert L.down(vast, state, FAST) == [result.winner.instance_id]
    assert not on_disk(state).closed
    vast.sticky.clear()
    vast.empty_listings = 0
    assert L.down(vast, on_disk(state), FAST) == [] and on_disk(state).closed


def test_down_is_not_done_if_it_cannot_list(tmp_path):
    # The id is confirmed gone, but nothing says no other box carries the tag.
    result, vast, state = up(tmp_path, {1: "ok"}, spare=0)
    vast.list_fails = 10**9
    assert L.down(vast, state, FAST) is None
    assert vast.ours() == [] and not on_disk(state).closed
    vast.list_fails = 0
    assert L.down(vast, on_disk(state), FAST) == [] and on_disk(state).closed


def test_a_listing_that_still_shows_a_destroyed_box_is_not_a_clean_one(tmp_path):
    # The lookup says gone; the listing lags and shows it three more times.
    # "Two listings that show nothing" means nothing.
    result, vast, state = up(tmp_path, {1: "ok"}, spare=0)
    vast.listing_lag = 3
    vast.calls.clear()
    assert L.down(vast, state, FAST) == []
    assert len(vast.names("instances")) == 5  # three that showed it, then two that did not


def test_a_box_that_will_not_go_away_does_not_keep_another_from_being_deleted(tmp_path):
    # One id from the state file survives every delete. A second box, of a
    # create that got no answer, is only found by the listing. It must get
    # its delete in this run of `down`, and not wait behind the first.
    result, vast, state = up(tmp_path, {1: "ok", 2: "lost_late"})
    stuck = result.winner.instance_id
    (other,) = [i for i in vast.ours() if i != stuck]
    vast.revealed = True
    vast.sticky.add(1)
    # Even when the first round alone uses up all the time there is.
    hurried = L.Policy(**{**FAST.__dict__, "confirm_timeout": 0.05, "round": 0.1})
    assert L.down(vast, state, hurried) == [stuck]
    assert vast.ours() == [stuck] and ("destroy", other) in vast.names()
    vast.sticky.clear()
    assert L.down(vast, on_disk(state), FAST) == []


def test_up_looks_for_the_tag_even_when_a_loser_will_not_go_away(tmp_path):
    launch, vast, state = start(tmp_path, {1: "ok", 2: "hold", 3: "lost_late"}, spare=2)
    vast.revealed = True
    vast.sticky.add(2)
    result = launch.run({})
    assert result.winner.machine_id == 1 and len(result.left) == 1
    assert [vast.live[i]["machine"] for i in vast.ours()] == [1, 2]  # the unanswered one is gone


def test_a_box_that_shows_up_between_the_two_listings_is_destroyed_too(tmp_path):
    # The first listing after the delete shows nothing; the box of a create
    # that got no answer becomes visible just then. The second listing is why
    # it is found, and destroying it is what `down` does about it.
    result, vast, state = up(tmp_path, {1: "lost_late", 2: "ok"})
    assert len(vast.ours()) == 2 and state.ids == [result.winner.instance_id]
    at_call(vast, "instances", 1, lambda: None)
    seen = [0]

    def listing(_):
        seen[0] += 1
        if seen[0] == 2:
            vast.revealed = True

    vast.hooks["instances"] = listing
    assert L.down(vast, state, FAST) == []
    assert vast.ours() == [] and on_disk(state).closed and seen[0] >= 4


def test_down_keeps_trying_and_then_says_what_is_left(tmp_path, capsys):
    result, vast, state = up(tmp_path, {1: "ok"}, spare=0)
    vast.sticky.add(1)
    left = L.down(vast, state, FAST)
    assert left == [result.winner.instance_id] and len(vast.names("destroy")) > 3
    capsys.readouterr()
    L.explain_leftovers(v.VastProvider(vast), state, left)
    out = capsys.readouterr().out
    assert str(left[0]) in out and f"gpubox down --run-id {RUN_ID}" in out
    assert "lease ends" in out and time.strftime("%Y-%m-%d", time.gmtime(EXPIRY)) in out


def test_down_by_tag_alone_destroys_the_runs_boxes_and_nothing_else(tmp_path):
    # A run whose state directory is gone.
    result, vast, state = up(tmp_path, {1: "ok", 2: "hold"})
    before = state.path.read_text()
    assert L.down(vast, State.ephemeral(RUN_ID), FAST) == []
    assert vast.ours() == [] and foreign_untouched(vast)
    assert state.path.read_text() == before  # the directory's state was not touched
    assert L.down(vast, State.ephemeral(OTHER_RUN), FAST) == []
    assert 9003 not in vast.live  # asked for by its run id, so that one goes too


def test_a_dead_output_stream_does_not_stop_a_teardown(tmp_path, monkeypatch):
    result, vast, state = up(tmp_path, {1: "ok"}, spare=0)

    def dead(*a, **kw):
        raise BrokenPipeError

    monkeypatch.setattr("builtins.print", dead)
    assert L.down(vast, state, FAST) == [] and vast.ours() == []


def test_down_still_destroys_when_the_state_file_cannot_be_written(tmp_path, monkeypatch):
    result, vast, state = up(tmp_path, {1: "ok"}, spare=0)

    def fail(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    assert L.down(vast, state, FAST) == [] and vast.ours() == []


# ---------------------------------------------------------------- the reaper


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
    box(vast, 6, make_tag(RUN_ID, 3, now + 600), age_min=5000)  # old, but its lease was renewed
    return vast


def test_a_reaper_without_destroy_changes_nothing(capsys):
    vast = reaper_world()
    assert L.reap(vast, FAST, "quack-ci") == 0
    assert vast.names("destroy") == [] and len(vast.ours()) == 6
    assert "would destroy" in capsys.readouterr().out


def test_the_reaper_destroys_only_boxes_of_its_namespace_whose_lease_has_ended():
    vast = reaper_world()
    assert L.reap(vast, FAST, "quack-ci", destroy=True) == 0
    assert vast.ours() == [1, 3, 4, 5, 6]
    assert foreign_untouched(vast)
    assert L.reap(vast, FAST, "dev", destroy=True) == 0 and vast.ours() == [1, 3, 5, 6]


def test_the_reaper_counts_a_delete_that_is_not_confirmed():
    vast = reaper_world()
    vast.sticky.add(2)
    assert L.reap(vast, FAST, "quack-ci", destroy=True) == 1


def test_the_reaper_fails_on_a_listing_it_cannot_read():
    vast = reaper_world()
    vast.list_fails = 1
    with pytest.raises(v.VastError):
        L.reap(vast, FAST, "quack-ci", destroy=True)
    assert vast.names("destroy") == []


# ------------------------------------------------------------------- renewal


def test_renewing_moves_the_lease_on_the_box_in_the_tag_and_in_the_state(tmp_path):
    result, vast, state = up(tmp_path, {1: "ok"}, spare=0)
    prober, later = FakeProber(vast), EXPIRY + 7200
    L.renew(v.VastProvider(vast), state, prober, later)
    iid = result.winner.instance_id
    assert prober.ran == [(("10.0.0.1", 2222), v.set_expiry_command(later))]
    assert parse_tag(vast.live[iid]["label"]) == parse_tag(make_tag(RUN_ID, 1, later))
    assert on_disk(state).expiry == later
    # The reaper now goes by the new expiry, and `down` still finds the box by its tag.
    assert L.reap(vast, FAST, "quack-ci", destroy=True, now=EXPIRY + 3600) == 0
    assert vast.ours() == [iid]
    assert L.down(vast, State.ephemeral(RUN_ID), FAST) == [] and vast.ours() == []


def test_a_renewal_that_cannot_reach_the_box_changes_nothing(tmp_path):
    result, vast, state = up(tmp_path, {1: "ok"}, spare=0)
    prober = FakeProber(vast)
    prober.run_rc = 255
    with pytest.raises(ProviderError, match="expiry"):
        L.renew(v.VastProvider(vast), state, prober, EXPIRY + 7200)
    assert vast.names("relabel") == [] and on_disk(state).expiry == EXPIRY


def test_a_renewal_whose_relabel_fails_leaves_the_reaper_on_the_old_expiry(tmp_path):
    # Half done, the box ends early (the reaper's view), never late.
    result, vast, state = up(tmp_path, {1: "ok"}, spare=0)
    vast.hooks["relabel"] = lambda iid: (_ for _ in ()).throw(v.VastError(500, "boom"))
    with pytest.raises(v.VastError):
        L.renew(v.VastProvider(vast), state, FakeProber(vast), EXPIRY + 7200)
    label = vast.live[result.winner.instance_id]["label"]
    assert parse_tag(label).expiry == EXPIRY and on_disk(state).expiry == EXPIRY


def test_a_relabel_that_took_effect_but_whose_answer_was_lost_is_reported_as_half_done(
    tmp_path, capsys
):
    result, vast, state = up(tmp_path, {1: "ok"}, spare=0)
    real = vast.relabel

    def relabel(iid, label):
        real(iid, label)
        raise v.VastError(0, "timed out")

    vast.relabel = relabel
    with pytest.raises(v.VastError):
        L.renew(v.VastProvider(vast), state, FakeProber(vast), EXPIRY + 7200)
    assert "half done" in capsys.readouterr().out and on_disk(state).expiry == EXPIRY
    vast.relabel = real
    # Running it again settles it.
    L.renew(v.VastProvider(vast), state, FakeProber(vast), EXPIRY + 7200)
    assert on_disk(state).expiry == EXPIRY + 7200


def test_a_run_without_a_box_cannot_be_renewed(tmp_path):
    launch, vast, state = start(tmp_path, {1: "ok"})
    with pytest.raises(L.ConfigError):
        L.renew(v.VastProvider(vast), state, FakeProber(vast), EXPIRY + 60)


# ---------------------------------------------------------- the gate script

HOOKED = L.Spec(image="img", gpus=("RTX 5090",), max_dph_per_gpu=1.25, gate_script="exit 0\n")


def test_a_gate_script_runs_after_the_gate_and_admits_the_box(tmp_path):
    result, vast, state = run(tmp_path, {1: "ok"}, spec=HOOKED, spare=0)
    assert result.winner.machine_id == 1 and result.winner.phase == "healthy"
    only_winner_left(result, vast, state)


def test_no_gate_script_is_run_unless_one_is_given(tmp_path):
    launch, vast, state = start(tmp_path, {1: "ok"}, spare=0)
    assert launch.run({}).winner and launch.prober.hooked == []


def test_a_gate_script_is_not_run_on_a_box_that_failed_the_gate(tmp_path):
    launch, vast, state = start(tmp_path, {1: "bad_gpu", 2: "ok"}, spec=HOOKED, spare=0)
    result = launch.run({})
    assert result.winner.machine_id == 2 and launch.prober.hooked == [2]


def test_a_failed_gate_script_replaces_the_box_and_blames_nobody(tmp_path):
    result, vast, state = run(tmp_path, {1: "hook_fail", 2: "ok"}, spec=HOOKED, spare=0)
    assert result.winner.machine_id == 2 and reasons(result)[1] is None
    assert "missing library" in result.attempts[0].detail
    only_winner_left(result, vast, state)
    assert observed(result) == {(2, "healthy", None)}


def test_a_gate_script_verdict_blames_the_machine(tmp_path):
    result, vast, state = run(tmp_path, {1: "hook_net", 2: "ok"}, spec=HOOKED, spare=0)
    assert result.winner.machine_id == 2 and reasons(result)[1] == "no_network"
    assert observed(result) == {(1, "failed_admission", "no_network"), (2, "healthy", None)}


def test_a_box_in_its_gate_script_is_waited_for_and_stopped_when_it_loses(tmp_path):
    launch, vast, state = start(tmp_path, {1: "hook_late", 2: "ok"}, spec=HOOKED)
    result = launch.run({})
    assert result.winner.machine_id == 2 and 1 in launch.prober.hooked
    assert reasons(result)[1] is None
    only_winner_left(result, vast, state)


# ------------------------------------------------------------ several models

TWO_MODELS = L.Spec(image="img", gpus=("RTX 4090", "RTX 5090"), max_dph_per_gpu=1.25)


def test_any_of_several_models_is_rented_and_recorded_as_what_it_is(tmp_path):
    offers = [offer(1, dph=0.5, gpu="RTX 4090"), offer(2, dph=0.9, gpu="RTX 5090")]
    vast = FakeVast({1: "bad_gpu", 2: "ok"}, offers=offers)
    result, vast, state = run(tmp_path, {}, vast=vast, spec=TWO_MODELS, spare=0)
    assert vast.queries[0]["gpu_name"] == {"in": ["RTX 4090", "RTX 5090"]}
    assert result.winner.machine_id == 2 and on_disk(state).winner["gpu"] == "RTX 5090"
    obs = pol.Observations(None)
    L.observe(result, TWO_MODELS, "r", obs)
    assert {(o["machine_id"], o["gpu"]) for o in obs.rows} == {(1, "RTX 4090"), (2, "RTX 5090")}


def test_a_model_name_the_store_would_refuse_is_not_recorded():
    assert L.gpu_model("RTX 5090", ["RTX 5090"]) == "RTX 5090"
    assert L.gpu_model(None, ["RTX 5090"]) == "RTX 5090"  # the only one asked for
    assert L.gpu_model(None, ["RTX 4090", "RTX 5090"]) is None  # cannot tell which
    assert L.gpu_model("RTX\n5090;", ["RTX 5090"]) is None


def test_a_spec_takes_a_single_name_and_no_blank_one():
    assert L.Spec(image="i", gpus="H100", max_dph_per_gpu=1).gpus == ("H100",)
    assert L.Spec(image="i", gpus=(), max_dph_per_gpu=1).gpu_text == "1x any GPU"  # Slurm
    for bad in (("",), [None], ("H100", " ")):
        with pytest.raises(ValueError):
            L.Spec(image="i", gpus=bad, max_dph_per_gpu=1)


# ------------------------------------------------------------- several boxes


def test_several_boxes_are_raced_with_spares_and_handed_over_together(tmp_path):
    behaviour = dict.fromkeys(range(1, 7), "ok")
    vast = FakeVast(behaviour, offers=[offer(m, dph=m / 10) for m in behaviour])
    result, vast, state = run(tmp_path, {}, vast=vast, count=3)
    assert len(result.winners) == 3
    assert sorted(vast.ours()) == sorted(w.instance_id for w in result.winners)
    disk = on_disk(state)
    assert [w["id"] for w in disk.winners] == [w.instance_id for w in result.winners]
    assert [w.dph for w in result.winners] == sorted(
        w.dph for w in result.winners
    )  # cheapest first
    assert vast.peak <= 4  # count + spare
    assert not result.left


def test_a_failure_among_several_is_replaced_and_blamed(tmp_path):
    result, vast, state = run(tmp_path, {1: "ok", 2: "bad_gpu", 3: "ok"}, count=2, spare=0)
    assert {w.machine_id for w in result.winners} == {1, 3}
    assert observed(result) == {
        (1, "healthy", None), (3, "healthy", None), (2, "failed_admission", "cuda_init"),
    }  # fmt: skip


def test_too_few_healthy_boxes_rolls_back_all_of_them_and_blames_nobody(tmp_path):
    t0 = time.monotonic()
    result, vast, state = run(tmp_path, {1: "ok", 2: "bad_gpu", 3: "bad_gpu"}, count=2, spare=0)
    assert result.winners == [] and result.winner is None
    assert vast.ours() == [] and on_disk(state).winners == []
    assert observed(result) == set()
    assert time.monotonic() - t0 < FAST.overall_timeout / 2  # nothing left to try: no waiting
    assert reasons(result)[1] is None  # healthy, but not kept


def test_attempts_grow_with_the_number_of_boxes():
    assert L.Policy().attempts == 6 and L.Policy(count=3).attempts == 10
    assert L.Policy(count=3, max_attempts=4).attempts == 4
    assert L.Policy(count=3, spare=2).max_live == 5


def test_a_healthy_box_that_dies_while_the_others_race_is_replaced(tmp_path):
    launch, vast, state = start(tmp_path, {1: "ok", 2: "hold", 3: "ok"}, count=2, spare=0)

    def kill_the_first_then_release():
        for iid, box in list(vast.live.items()):
            if box["machine"] == 1:
                vast.live.pop(iid)
        vast.release = True

    at_call(vast, "gate_poll", 3, kill_the_first_then_release)
    result = launch.run({})
    assert {w.machine_id for w in result.winners} == {2, 3}
    assert sorted(vast.ours()) == sorted(w.instance_id for w in result.winners)


def test_an_interrupt_with_some_boxes_healthy_rolls_back_all_of_them(tmp_path):
    launch, vast, state = start(tmp_path, {1: "ok", 2: "hold", 3: "hold"}, count=3, spare=0)
    at_call(vast, "gate_poll", 4, interrupt)
    assert isinstance(outcome(launch), KeyboardInterrupt)
    assert vast.ours() == [] and on_disk(state).winners == []


def test_renewing_several_boxes_moves_the_lease_of_each(tmp_path):
    result, vast, state = up(tmp_path, {1: "ok", 2: "ok"}, count=2, spare=0)
    prober, later = FakeProber(vast), EXPIRY + 7200
    L.renew(v.VastProvider(vast), state, prober, later)
    assert len(prober.ran) == 2
    for w in result.winners:
        assert parse_tag(vast.live[w.instance_id]["label"]).expiry == later
    assert on_disk(state).expiry == later


def test_a_renewal_that_reaches_only_some_boxes_relabels_none(tmp_path):
    result, vast, state = up(tmp_path, {1: "ok", 2: "ok"}, count=2, spare=0)
    prober = FakeProber(vast)
    rcs = iter([0, 255])
    prober.run = lambda ep, remote, timeout=60: next(rcs)
    with pytest.raises(ProviderError, match="1 box"):
        L.renew(v.VastProvider(vast), state, prober, EXPIRY + 7200)
    assert vast.names("relabel") == [] and on_disk(state).expiry == EXPIRY


def test_a_create_answered_with_an_id_already_held_is_not_a_second_box(tmp_path):
    launch, vast, state = start(tmp_path, {1: "ok", 2: "ok", 3: "ok"}, count=2, spare=1)
    real_create = vast.create

    def create(offer_id, payload):
        iid = real_create(offer_id, payload)
        return 1 if offer_id == 1002 else iid  # the second create repeats the first id

    vast.create = create
    result = launch.run({})
    assert len({w.instance_id for w in result.winners}) == 2
    assert result.attempts[1].state == "unanswered"
    assert sorted(vast.ours()) == sorted(w.instance_id for w in result.winners)  # swept by tag


def test_a_launch_that_can_no_longer_succeed_rents_nothing_more(tmp_path):
    # Two of three offers, both needed: one fails, so three healthy boxes are out of reach.
    result, vast, state = run(tmp_path, {1: "ok", 2: "bad_gpu", 3: "late"}, count=3, spare=0)
    assert result.winners == [] and vast.ours() == []
    t0 = time.monotonic()
    result, vast, state = run(tmp_path / "b", {1: "ok", 2: "ok"}, count=3)
    assert result.winners == [] and vast.creates == 0  # too few offers: nothing rented
    assert time.monotonic() - t0 < 1


def test_one_hosts_word_against_the_image_is_not_the_verdict_while_another_box_runs(tmp_path):
    result, vast, state = run(tmp_path, {1: "ok", 2: "image"}, count=2, spare=0)
    assert result.winners == []  # an infrastructure failure, not a configuration error
    assert vast.ours() == []


def test_a_box_that_answered_with_our_key_proves_the_image(tmp_path):
    launch, vast, state = start(tmp_path, {1: "hold", 2: "image", 3: "image", 4: "ok"},
                                count=2, spare=2)  # fmt: skip
    # Machine 1 is in its gate (it answered SSH) before the other two report image errors.
    vast.behaviour.update({2: "stuck", 3: "stuck"})
    polls = [0]

    def gate_poll(machine):
        polls[0] += 1
        if polls[0] == 2:
            vast.behaviour.update({2: "image", 3: "image"})
        if polls[0] == 8:
            vast.release = True

    vast.hooks["gate_poll"] = gate_poll
    result = launch.run({})
    assert {w.machine_id for w in result.winners} == {1, 4}
    assert {a.machine_id for a in result.attempts if "cannot be pulled" in a.detail} == {2, 3}


def test_held_slots_with_too_few_healthy_boxes_end_the_launch_at_once(tmp_path):
    t0 = time.monotonic()
    result, vast, state = run(tmp_path, {1: "ok", 2: "ghost", 3: "ok"}, count=2, spare=0)
    assert result.winners == [] and vast.ours() == []
    assert time.monotonic() - t0 < FAST.overall_timeout / 2


@pytest.mark.parametrize("when", ["before the rename", "after the rename"])
def test_an_interrupt_during_the_hand_off_of_several_boxes(tmp_path, monkeypatch, when):
    launch, vast, state = start(tmp_path, {1: "ok", 2: "ok", 3: "hold"}, count=2, spare=1)
    real_replace = os.replace

    def replace(src, dst):
        if '"winners": []' in open(src).read():
            return real_replace(src, dst)
        if when == "after the rename":
            real_replace(src, dst)
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", replace)
    assert isinstance(outcome(launch), KeyboardInterrupt)
    monkeypatch.undo()
    disk = on_disk(state)
    if when == "before the rename":
        assert disk.winners == [] and vast.ours() == []
    else:
        assert sorted(vast.ours()) == sorted(w["id"] for w in disk.winners)
        assert {vast.live[i]["machine"] for i in vast.ours()} == {1, 2}  # the loser is gone
    assert L.down(vast, disk, FAST) == [] and vast.ours() == []
