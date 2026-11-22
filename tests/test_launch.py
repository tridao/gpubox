"""The race, the journal and cleanup, against a fake provider.

Every test that rents states exactly which boxes are left afterwards, and
`run` checks the peak number of billable boxes and that nothing foreign was
touched.
"""

import os
import re
import signal
import threading
import time

import pytest
from fakes import FAST, RUN_ID, SPEC, FakeVast, foreign_untouched, start, statuses

from gpubox import launch as L
from gpubox import policy as pol
from gpubox import vast as v
from gpubox.journal import Journal, StateError, attempt_label, state_lock


def run(tmp_path, behaviour, **kw):
    launch, vast, journal = start(tmp_path, behaviour, **kw)
    try:
        result = launch.run({})
    finally:
        assert foreign_untouched(vast)
        assert vast.peak <= launch.policy.max_live
    return result, vast, journal


def reasons(result):
    return {a.machine_id: a.reason for a in result.attempts}


def observed(result):
    obs = pol.Observations(None)
    L.observe(result, SPEC, "r", obs)
    return {(o["machine_id"], o["kind"], o["reason_code"]) for o in obs.rows}


def only_winner_left(result, vast, journal):
    """The provider and the journal agree: exactly the winner exists."""
    assert vast.ours() == [result.winner.instance_id]
    assert journal.state().live_ids() == [result.winner.instance_id]
    assert not journal.state().open_attempts()


def nothing_left(vast, journal):
    assert vast.ours() == []
    assert not journal.state().is_live()


# ------------------------------------------------------------------ the race


def test_failed_machine_is_replaced_and_blamed(tmp_path):
    result, vast, journal = run(tmp_path, {1: "bad_gpu", 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    only_winner_left(result, vast, journal)
    assert observed(result) == {(1, "failed_admission", "cuda_init"), (2, "healthy", None)}


def test_race_keeps_one_and_does_not_blame_the_loser(tmp_path):
    result, vast, journal = run(tmp_path, {1: "ok", 2: "ok", 3: "ok"})
    assert result.winner.machine_id in (1, 2)
    assert 3 not in reasons(result)  # never needed
    only_winner_left(result, vast, journal)
    assert {kind for _, kind, _ in observed(result)} == {"healthy"}
    assert not result.unresolved


def test_loser_destroyed_mid_gate_is_not_blamed(tmp_path):
    launch, vast, journal = start(tmp_path, {1: "late", 2: "hold"})
    threading.Thread(  # machine 2 may only win once machine 1 is inside its gate
        target=lambda: vast.gate_started(1).wait(5) and vast.release.set(), daemon=True
    ).start()
    result = launch.run({})
    assert result.winner.machine_id == 2 and vast.gate_started(1).is_set()
    assert reasons(result)[1] is None
    only_winner_left(result, vast, journal)


def test_nothing_came_up_rolls_back_and_blames_nobody(tmp_path):
    # Could be Docker Hub or the image: no control, so no verdict on the machines.
    result, vast, journal = run(tmp_path, {1: "stuck", 2: "bad_gpu", 3: "cdi"})
    assert result.winner is None
    nothing_left(vast, journal)
    assert observed(result) == set()


@pytest.mark.parametrize(
    "kind,reason", [("stuck", "unreachable"), ("exited", "unreachable"), ("cdi", "runtime_error")]
)
def test_startup_failures_carry_a_reason_code(tmp_path, kind, reason):
    result, vast, journal = run(tmp_path, {1: kind, 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    assert reasons(result)[1] == reason
    only_winner_left(result, vast, journal)


def test_stale_offer_is_skipped_without_blame(tmp_path):
    result, vast, journal = run(tmp_path, {1: "stale", 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    assert reasons(result)[1] is None
    assert statuses(journal)[1] == "refused" and vast.creates == 1
    only_winner_left(result, vast, journal)


def test_bad_image_aborts_instead_of_trying_every_host(tmp_path):
    launch, vast, journal = start(tmp_path, dict.fromkeys(range(1, 6), "image"))
    with pytest.raises(L.ConfigError, match="cannot be pulled"):
        launch.run({})
    assert vast.creates == 2
    nothing_left(vast, journal)


@pytest.mark.parametrize("behaviour", [{1: "ok", 2: "image"}, {1: "image", 2: "ok"}])
def test_a_healthy_box_outweighs_another_hosts_image_error(tmp_path, monkeypatch, behaviour):
    # Both finish in the same batch. A healthy box proves the image can be
    # pulled, so the launch succeeds whichever of the two is looked at first.
    launch, vast, journal = start(tmp_path, behaviour)
    real_wait = L.wait

    def both_done(pending, timeout=None, return_when=None):
        return real_wait(pending, timeout=5, return_when="ALL_COMPLETED")

    monkeypatch.setattr(L, "wait", both_done)
    result = launch.run({})
    assert vast.behaviour[result.winner.machine_id] == "ok" and vast.creates == 2
    only_winner_left(result, vast, journal)


def test_max_attempts_bounds_the_spend(tmp_path):
    result, vast, journal = run(tmp_path, dict.fromkeys(range(1, 9), "bad_gpu"), max_attempts=3)
    assert result.winner is None and vast.creates == 3
    nothing_left(vast, journal)


def test_timeout_rolls_back_a_candidate_that_would_have_been_healthy_later(tmp_path):
    result, vast, journal = run(tmp_path, {1: "slow"}, overall_timeout=0.2)
    assert result.winner is None
    nothing_left(vast, journal)


def test_no_box_is_rented_once_the_renting_window_has_closed(tmp_path):
    # Less time overall than is reserved for cleanup: nothing may be created.
    result, vast, _ = run(tmp_path, {1: "ok"}, overall_timeout=30, cleanup_reserve=60)
    assert result.winner is None and vast.creates == 0
    assert result.attempts == []  # not even an intent: nothing to reconcile later


def test_a_worker_checks_the_window_again_just_before_it_sends(tmp_path, monkeypatch):
    # The window closes while the worker prepares its request.
    launch, vast, journal = start(tmp_path, {1: "ok"}, spare=0)
    real = v.create_payload

    def slow_payload(*a):
        launch.rent_until = time.monotonic() - 1
        return real(*a)

    monkeypatch.setattr(v, "create_payload", slow_payload)
    result = launch.run({})
    assert result.winner is None and vast.creates == 0
    assert statuses(journal) == {1: "refused"}


def test_a_silent_provider_api_is_not_held_against_the_machine(tmp_path):
    result, vast, journal = run(tmp_path, {1: "api_down", 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    assert reasons(result)[1] is None and "provider API" in result.attempts[0].detail
    only_winner_left(result, vast, journal)


# ------------------------------------------------------- signals and failures


def signal_when(vast, machines, signum=signal.SIGTERM):
    """Deliver a real signal to this process once the given gates are running."""

    def fire():
        for m in machines:
            assert vast.gate_started(m).wait(5)
        os.kill(os.getpid(), signum)
        time.sleep(0.05)
        vast.release.set()

    threading.Thread(target=fire, daemon=True).start()


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_a_signal_while_both_candidates_are_in_the_gate_rolls_back(tmp_path, signum):
    before = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    launch, vast, journal = start(tmp_path, {1: "hold", 2: "hold"})
    signal_when(vast, (1, 2), signum)
    with pytest.raises(KeyboardInterrupt):
        launch.run({})
    assert vast.creates == 2 and vast.peak == 2
    nothing_left(vast, journal)
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


def test_a_signal_during_loser_cleanup_takes_the_candidate_down_too(tmp_path):
    launch, vast, journal = start(tmp_path, {1: "ok", 2: "stuck"})
    fired = []

    def on_destroy(iid):  # the first delete is the loser's, in cleanup
        if not fired:
            fired.append(iid)
            os.kill(os.getpid(), signal.SIGTERM)

    vast.on_destroy = on_destroy
    with pytest.raises(KeyboardInterrupt):
        launch.run({})
    assert fired and journal.state().winner is None  # never handed off
    nothing_left(vast, journal)


def test_a_signal_right_after_the_winner_is_recorded_still_leaves_nothing(tmp_path, monkeypatch):
    launch, vast, journal = start(tmp_path, {1: "ok"}, spare=0)
    real = journal.append

    def signal_after_winner(ev, **fields):
        real(ev, **fields)
        if ev == "winner":
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(journal, "append", signal_after_winner)
    with pytest.raises(KeyboardInterrupt):
        launch.run({})
    assert journal.state().winner is not None  # it was handed off, then taken back
    nothing_left(vast, journal)


def test_a_failure_takes_precedence_over_a_signal(tmp_path, monkeypatch):
    launch, vast, journal = start(tmp_path, {1: "ok"}, spare=0)
    real = journal.append

    def fail_and_signal(ev, **fields):
        if ev == "winner":
            os.kill(os.getpid(), signal.SIGTERM)
            raise OSError(28, "No space left on device")
        real(ev, **fields)

    monkeypatch.setattr(journal, "append", fail_and_signal)
    with pytest.raises(OSError):
        launch.run({})
    nothing_left(vast, journal)


def test_handlers_are_restored_after_a_normal_launch(tmp_path):
    marker = lambda *a: None
    old = signal.signal(signal.SIGTERM, marker)
    try:
        result, vast, journal = run(tmp_path, {1: "ok"}, spare=0)
        assert signal.getsignal(signal.SIGTERM) is marker
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGTERM, old)
    only_winner_left(result, vast, journal)


def test_a_launch_off_the_main_thread_runs_without_touching_handlers(tmp_path):
    launch, vast, journal = start(tmp_path, {1: "ok"}, spare=0)
    out = []
    t = threading.Thread(target=lambda: out.append(launch.run({})))
    t.start()
    t.join(10)
    only_winner_left(out[0], vast, journal)


def test_an_exception_in_the_main_loop_rolls_back(tmp_path, monkeypatch):
    launch, vast, journal = start(tmp_path, {1: "hold", 2: "hold"})

    def boom(*a, **k):
        for m in (1, 2):
            assert vast.gate_started(m).wait(5)
        vast.release.set()
        raise RuntimeError("bug in the loop")

    monkeypatch.setattr(L, "wait", boom)
    with pytest.raises(RuntimeError):
        launch.run({})
    assert vast.creates == 2
    nothing_left(vast, journal)


def test_an_unexpected_error_in_the_first_cleanup_pass_does_not_skip_the_rest(tmp_path):
    launch, vast, journal = start(tmp_path, {1: "ok", 2: "stuck", 3: "ok"})
    raised = []

    def on_destroy(iid):
        if not raised:
            raised.append(iid)
            raise RuntimeError("not a VastError")

    vast.on_destroy = on_destroy
    result = launch.run({})
    assert raised and result.winner.machine_id == 1
    only_winner_left(result, vast, journal)


def test_no_create_is_sent_when_the_intent_cannot_be_journalled(tmp_path, monkeypatch):
    launch, vast, journal = start(tmp_path, {1: "ok"})
    real = journal.append

    def failing(ev, **fields):
        if ev == "intent":
            raise OSError(28, "No space left on device")
        real(ev, **fields)

    monkeypatch.setattr(journal, "append", failing)
    with pytest.raises(OSError):
        launch.run({})
    assert vast.creates == 0


def fail_once(journal, monkeypatch, event):
    real, failed = journal.append, []

    def flaky(ev, **fields):
        if ev == event and not failed:
            failed.append(1)
            raise OSError(5, "Input/output error")
        real(ev, **fields)

    monkeypatch.setattr(journal, "append", flaky)


def test_an_id_that_cannot_be_journalled_is_not_kept(tmp_path, monkeypatch):
    # The intent is durable, the create returns an id, and the write of that
    # id fails once. The box must not be admitted on the strength of memory.
    launch, vast, journal = start(tmp_path, {1: "ok"}, spare=0)
    fail_once(journal, monkeypatch, "created")
    with pytest.raises(OSError):
        launch.run({})
    assert vast.creates == 1
    nothing_left(vast, journal)


def test_an_unjournalled_id_whose_delete_is_not_confirmed_can_still_be_destroyed_by_id(
    tmp_path, monkeypatch
):
    # As above, and the rollback's delete does not go through either. The id
    # must survive in the journal, so that a later `down` can destroy the box
    # even though it has meanwhile changed its label.
    vast = FakeVast({1: "ok"})
    vast.sticky.add(1)
    launch, vast, journal = start(tmp_path, {}, vast=vast, spare=0)
    fail_once(journal, monkeypatch, "created")
    with pytest.raises(OSError):
        launch.run({})
    (iid,) = vast.ours()
    assert Journal(tmp_path).state().live_ids() == [iid]  # read back from disk
    vast.live[iid]["label"] = "renamed-by-the-box"
    vast.sticky.clear()
    assert L.down(vast, Journal(tmp_path), FAST) == 0
    nothing_left(vast, Journal(tmp_path))


def test_a_winner_that_cannot_be_journalled_is_rolled_back(tmp_path, monkeypatch):
    launch, vast, journal = start(tmp_path, {1: "ok", 2: "ok"})
    real = journal.append

    def no_winner(ev, **fields):
        if ev == "winner":
            raise OSError(28, "No space left on device")
        real(ev, **fields)

    monkeypatch.setattr(journal, "append", no_winner)
    with pytest.raises(OSError):
        launch.run({})
    nothing_left(vast, journal)  # including the candidate that had passed the gate


# ------------------------------------------------- create with unknown outcome


def test_lost_create_response_is_found_by_label_and_destroyed(tmp_path):
    result, vast, journal = run(tmp_path, {1: "lost", 2: "ok"})
    assert result.winner.machine_id == 2
    only_winner_left(result, vast, journal)  # the lost box did not leak
    assert statuses(journal)[1] == "destroyed" and reasons(result)[1] is None


def test_unknown_create_blocks_its_slot_instead_of_being_replaced(tmp_path):
    # Slot 1 may hold a box we cannot see. With two slots, only one more may run.
    result, vast, journal = run(tmp_path, {1: "ghost", 2: "bad_gpu", 3: "ok", 4: "ok"})
    assert result.winner.machine_id == 3
    assert [a.machine_id for a in result.unresolved] == [1]
    assert statuses(journal)[1] == "unknown" and 4 not in reasons(result)


def test_an_unseen_create_is_never_written_off_by_the_passage_of_time(tmp_path):
    result, vast, journal = run(tmp_path, {1: "ghost", 2: "ok"})
    # Nothing carries its label, however long we wait: that is not proof.
    assert L.down(vast, journal, FAST) == 1
    assert L.down(vast, journal, FAST, now=time.time() + 10 * 86400) == 1
    state = journal.state()
    assert statuses(journal)[1] == "unknown" and not state.closed
    assert vast.ours() == []  # the winner was destroyed regardless


def test_the_operator_can_abandon_an_unseen_create_once_it_is_old_enough(tmp_path):
    result, vast, journal = run(tmp_path, {1: "ghost", 2: "ok"})
    assert L.down(vast, journal, FAST, abandon_unseen=True) == 1  # too young
    assert statuses(journal)[1] == "unknown"
    later = time.time() + FAST.settle + 1
    assert L.down(vast, journal, FAST, abandon_unseen=True, now=later) == 0
    assert statuses(journal)[1] == "abandoned" and journal.state().closed


def test_abandoning_does_not_apply_to_a_box_that_is_visible(tmp_path):
    result, vast, journal = run(tmp_path, {1: "lost_late", 2: "ok"})
    assert len(vast.ours()) == 2  # the winner, and a box `up` could not see
    # A teardown that runs before the box is visible must not write it off.
    assert L.down(vast, journal, FAST) == 1
    assert len(vast.ours()) == 1 and not journal.state().closed
    vast.reveal()
    later = time.time() + FAST.settle + 1
    assert L.down(vast, journal, FAST, abandon_unseen=True, now=later) == 0
    nothing_left(vast, journal)
    assert statuses(journal)[1] == "destroyed" and foreign_untouched(vast)


# ------------------------------------------- recovery by label is not trusted


def crashed_after_intent(tmp_path, ts=None):
    """A run whose process died right after journalling an intent for machine 1."""
    _, vast, journal = start(tmp_path, {1: "ok"})
    label = attempt_label(RUN_ID, 1)
    journal.append("intent", seq=1, label=label, offer_id=1001, machine_id=1, host_id=1, dph=1.0)
    return vast, journal, label


def instance(vast, label, **fields):
    """Put an instance with this label on the provider; returns its id."""
    iid = max(vast.live) + 1
    row = {"id": iid, "label": label, "machine": 1, "start_date": time.time(), "image_uuid": "img"}
    vast.live[iid] = {**row, **fields}
    return iid


def test_down_recovers_a_box_whose_create_answer_we_never_saw(tmp_path):
    vast, journal, label = crashed_after_intent(tmp_path)
    iid = vast.create(1001, {"label": label, "image": "img"})  # the provider did rent it
    assert L.down(vast, journal, FAST) == 0
    nothing_left(vast, journal)
    assert journal.state().boxes == {iid: "destroyed"} and foreign_untouched(vast)


@pytest.mark.parametrize(
    "fields",
    [
        {"machine": 99},  # another machine
        {"image_uuid": "someone/else"},
        {"start_date": time.time() - 3600},  # an hour older than the attempt
        {"start_date": time.time() - 60},  # a minute older: beyond clock skew
        {"machine": None},  # the listing does not say which machine
        {"image_uuid": None},
        {"start_date": None},
        {"start_date": "yesterday"},
        {"start_date": float("nan")},
    ],
)
def test_an_instance_that_merely_carries_our_label_is_left_alone(tmp_path, fields):
    vast, journal, label = crashed_after_intent(tmp_path)
    impostor = instance(vast, label, **fields)
    assert L.down(vast, journal, FAST) == 1
    assert impostor in vast.live and not journal.state().closed
    assert journal.state().boxes == {}  # and its id was not adopted


def test_a_box_started_within_clock_skew_before_the_intent_is_still_ours(tmp_path):
    vast, journal, label = crashed_after_intent(tmp_path)
    instance(vast, label, start_date=time.time() - L.CLOCK_SKEW / 2)
    assert L.down(vast, journal, FAST) == 0
    nothing_left(vast, journal)


def test_two_instances_with_our_label_are_both_left_alone_even_if_one_fits(tmp_path):
    vast, journal, label = crashed_after_intent(tmp_path)
    fits, other = instance(vast, label), instance(vast, label, machine=99)
    assert L.down(vast, journal, FAST) == 1
    assert fits in vast.live and other in vast.live


def test_reconcile_does_not_guess_between_the_real_box_and_a_decoy(tmp_path):
    launch, vast, journal = start(tmp_path, {1: "lost", 2: "ok"})
    decoy = instance(vast, attempt_label(RUN_ID, 1), machine=99)
    result = launch.run({})
    assert result.winner.machine_id == 2
    # Two instances carry the label: neither is touched, the attempt stays open.
    assert decoy in vast.live and len(vast.ours()) == 3
    assert [a.machine_id for a in result.unresolved] == [1]


def test_the_listing_is_the_only_evidence_match_lost_box_accepts():
    row = {"id": 7, "label": "l", "machine_id": 1, "image_uuid": "img", "start_date": 100.0}
    ok = dict(not_before=100.0, image="img", machine_id=1)
    assert L.match_lost_box([row], "l", **ok) == (row, 1)
    assert L.match_lost_box([], "l", **ok) == (None, 0)
    for change in ({"id": True}, {"id": "7"}, {"id": 0}, {"id": 7.0}):
        assert L.match_lost_box([{**row, **change}], "l", **ok) == (None, 1)
    for arg in ({"not_before": None}, {"image": None}, {"image": ""}, {"machine_id": None}):
        assert L.match_lost_box([row], "l", **{**ok, **arg}) == (None, 1)


# --------------------------------------------------------------------- delete


def test_transient_delete_failure_is_retried_until_confirmed(tmp_path):
    vast = FakeVast({1: "bad_gpu", 2: "ok"})
    vast.fail_destroy[1] = 2
    result, vast, journal = run(tmp_path, {}, vast=vast, spare=0)
    assert result.winner.machine_id == 2
    only_winner_left(result, vast, journal)
    assert statuses(journal)[1] == "destroyed"


def test_unconfirmed_delete_keeps_its_slot_and_is_reported(tmp_path):
    vast = FakeVast({1: "bad_gpu", 2: "ok", 3: "ok"})
    vast.sticky.add(1)
    result, vast, journal = run(tmp_path, {}, vast=vast, spare=0)
    # One slot, and its box will not go away: renting another would exceed the cap.
    assert result.winner is None and vast.creates == 1
    assert [a.machine_id for a in result.unresolved] == [1]
    assert statuses(journal)[1] == "destroying"

    assert L.down(vast, journal, FAST) == 1 and not journal.state().closed
    vast.sticky.clear()
    assert L.down(vast, journal, FAST) == 0 and journal.state().closed
    nothing_left(vast, journal)


def test_a_box_that_is_already_gone_is_confirmed_even_if_delete_keeps_failing(tmp_path):
    result, vast, journal = run(tmp_path, {1: "ok"}, spare=0)
    iid = result.winner.instance_id
    vast.live.pop(iid)  # gone on its own (its timer, or the host)
    monkeypatch_destroy = lambda i: (_ for _ in ()).throw(v.VastError(500, "refused"))
    vast.destroy = monkeypatch_destroy
    assert L.down(vast, journal, FAST) == 0 and journal.state().closed


# ----------------------------------------------------------------------- down


def test_down_destroys_by_id_even_if_the_box_relabelled_itself(tmp_path):
    result, vast, journal = run(tmp_path, {1: "ok"}, spare=0)
    vast.live[result.winner.instance_id]["label"] = "renamed-by-the-box"
    assert L.down(vast, journal, FAST) == 0
    nothing_left(vast, journal)
    assert foreign_untouched(vast)


def test_down_never_touches_another_run_in_the_same_namespace(tmp_path):
    result, vast, journal = run(tmp_path, {1: "ok"}, spare=0)
    assert L.down(vast, journal, FAST) == 0
    # 9003 is labelled quack-ci.other.89abcdef.a1: same namespace, not our run.
    assert 9003 in vast.live


def test_down_without_a_run_is_a_no_op(tmp_path):
    vast = FakeVast({})
    assert L.down(vast, Journal(tmp_path), FAST) == 0 and foreign_untouched(vast)


def test_a_listing_outage_does_not_stop_known_boxes_from_being_destroyed(tmp_path):
    result, vast, journal = run(tmp_path, {1: "ghost", 2: "ok"})
    assert result.winner.machine_id == 2
    vast.list_fails = 5
    assert L.down(vast, journal, FAST) == 1  # the ghost is unresolved...
    assert vast.ours() == []  # ...but the winner, known by id, is gone


def test_down_does_not_report_done_when_its_records_could_not_be_written(tmp_path, monkeypatch):
    result, vast, journal = run(tmp_path, {1: "ok"}, spare=0)

    def full_disk(ev, **fields):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(journal, "append", full_disk)
    assert L.down(vast, journal, FAST) >= 1  # the box is gone, but the journal cannot say so
    assert vast.ours() == []
    monkeypatch.undo()
    assert L.down(vast, journal, FAST) == 0 and journal.state().closed


def test_a_damaged_journal_is_torn_down_but_never_reported_clean(tmp_path):
    result, vast, journal = run(tmp_path, {1: "ok"}, spare=0)
    lines = journal.path.read_text().splitlines()
    lines.insert(2, '{"ev": "intent", "seq": 9, "lab')  # damage in the middle
    journal.path.write_text("\n".join(lines) + "\n")
    assert L.down(vast, journal, FAST) == 1
    assert vast.ours() == [] and not journal.state().closed


def test_second_process_on_a_state_directory_fails_fast(tmp_path):
    with state_lock(tmp_path):
        with pytest.raises(StateError, match="in use"):
            with state_lock(tmp_path):
                pass
    with state_lock(tmp_path):  # released again
        pass


# ----------------------------------------------------------------------- reap


def labelled(vast, label, age_min):
    iid = max(vast.live) + 1
    vast.live[iid] = {
        "id": iid,
        "label": label,
        "machine": 1,
        "start_date": time.time() - age_min * 60,
    }
    return iid


def reap(vast, **kw):
    return L.reap(vast, "quack-ci", 90, FAST, **kw)


def test_reap_reports_unless_told_the_namespace_is_exclusive():
    vast = FakeVast({})
    old = labelled(vast, "quack-ci.run1.0123abcd.a1", age_min=200)
    assert reap(vast, destroy=False) == 0 and old in vast.live
    assert reap(vast, destroy=True) == 0 and old not in vast.live


def test_reap_matches_the_namespace_exactly_and_respects_age():
    vast = FakeVast({})
    young = labelled(vast, "quack-ci.run2.0123abcd.a1", age_min=10)
    lookalikes = [
        labelled(vast, label, age_min=500)
        for label in (
            "quack-ci-old.run.0123abcd.a1",  # another namespace with our name as a prefix
            "my-quack-ci.run.0123abcd.a1",
            "quack-ci.run.0123abcd",  # not an attempt label
            "quack-ci",
            "dev.tri.0123abcd.a1",
        )
    ]
    assert reap(vast, destroy=True) == 0
    assert young in vast.live and all(i in vast.live for i in lookalikes)
    assert foreign_untouched(vast)


def test_reap_counts_a_delete_that_is_not_confirmed_as_a_failure():
    vast = FakeVast({})
    old = labelled(vast, "quack-ci.run1.0123abcd.a1", age_min=200)
    vast.sticky.add(1)
    assert reap(vast, destroy=True) == 1 and old in vast.live


def test_reap_can_sweep_the_old_label_scheme_during_migration():
    vast = FakeVast({})
    legacy = labelled(vast, "quack-ci-123456-1", age_min=200)
    bystanders = [
        labelled(vast, label, age_min=500)
        for label in (
            "quack-ci-old.run.0123abcd.a1",  # a current-scheme label of another namespace
            "quack-ci-old-123456-1",
            "quack-ci-local-bench4x-1795372136",
            "quack-ci-123456-1-extra",
        )
    ]
    old_scheme = re.compile(r"quack-ci-\d+-\d+")
    assert reap(vast, destroy=True) == 0 and legacy in vast.live
    assert reap(vast, destroy=False, legacy=old_scheme) == 0 and legacy in vast.live
    assert reap(vast, destroy=True, legacy=old_scheme) == 0
    assert legacy not in vast.live and all(i in vast.live for i in bystanders)


def test_a_legacy_pattern_never_matches_a_current_scheme_label():
    vast = FakeVast({})
    other = labelled(vast, "sim.west.0123abcd.a1", age_min=500)
    assert reap(vast, destroy=True, legacy=re.compile(r".*")) == 0
    assert other in vast.live
