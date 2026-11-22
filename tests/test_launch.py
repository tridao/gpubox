"""The race, the journal and cleanup, against a fake provider.

Every test that rents checks what is left afterwards and the peak number of
billable boxes, not just the return value.
"""

import pytest
from fakes import FAST, FakeVast, foreign_untouched, start

from gpubox import launch as L
from gpubox import policy as pol
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


def observed(result, launch_spec=None):
    obs = pol.Observations(None)
    L.observe(
        result, launch_spec or L.Spec(image="img", gpu="RTX 5090", max_dph_per_gpu=1), "r", obs
    )
    return {(o["machine_id"], o["kind"], o["reason_code"]) for o in obs.rows}


# ------------------------------------------------------------------ the race


def test_failed_machine_is_replaced_and_blamed(tmp_path):
    result, vast, _ = run(tmp_path, {1: "bad_gpu", 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    assert vast.ours() == [result.winner.instance_id]
    assert observed(result) == {(1, "failed_admission", "cuda_init"), (2, "healthy", None)}


def test_race_keeps_one_and_does_not_blame_the_loser(tmp_path):
    result, vast, _ = run(tmp_path, {1: "ok", 2: "ok", 3: "ok"})
    assert result.winner.machine_id in (1, 2)
    assert 3 not in reasons(result)  # never needed
    assert vast.ours() == [result.winner.instance_id]
    assert {kind for _, kind, _ in observed(result)} == {"healthy"}
    assert not result.unresolved


def test_loser_destroyed_mid_gate_is_not_blamed(tmp_path):
    result, vast, _ = run(tmp_path, {1: "late", 2: "ok"})
    assert result.winner.machine_id == 2
    assert reasons(result)[1] is None
    assert vast.ours() == [result.winner.instance_id]


def test_nothing_came_up_rolls_back_and_blames_nobody(tmp_path):
    # Could be Docker Hub or the image: no control, so no verdict on the machines.
    result, vast, journal = run(tmp_path, {1: "stuck", 2: "bad_gpu", 3: "cdi"})
    assert result.winner is None
    assert vast.ours() == []
    assert observed(result) == set()
    assert not journal.state().live()


@pytest.mark.parametrize(
    "kind,reason", [("stuck", "unreachable"), ("exited", "unreachable"), ("cdi", "runtime_error")]
)
def test_startup_failures_carry_a_reason_code(tmp_path, kind, reason):
    result, vast, _ = run(tmp_path, {1: kind, 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    assert reasons(result)[1] == reason
    assert vast.ours() == [result.winner.instance_id]


def test_stale_offer_is_skipped_without_blame(tmp_path):
    result, vast, journal = run(tmp_path, {1: "stale", 2: "ok"}, spare=0)
    assert result.winner.machine_id == 2
    assert reasons(result)[1] is None
    assert journal.state().attempts[1].status == "refused"
    assert vast.creates == 1


def test_bad_image_aborts_instead_of_trying_every_host(tmp_path):
    launch, vast, journal = start(tmp_path, dict.fromkeys(range(1, 6), "image"))
    with pytest.raises(L.ConfigError, match="cannot be pulled"):
        launch.run({})
    assert vast.ours() == [] and vast.creates == 2
    assert not journal.state().live()


def test_max_attempts_bounds_the_spend(tmp_path):
    result, vast, _ = run(tmp_path, dict.fromkeys(range(1, 9), "bad_gpu"), max_attempts=3)
    assert result.winner is None and vast.creates == 3 and vast.ours() == []


def test_offers_without_a_price_are_never_rented(tmp_path):
    offers = [{"id": 1001, "machine_id": 1, "host_id": 1, "dph_total": None}]
    vast = FakeVast({1: "ok"}, offers=offers)
    result, vast, _ = run(tmp_path, {1: "ok"}, vast=vast)
    assert result.winner is None and vast.creates == 0


def test_timeout_rolls_back_a_candidate_that_would_have_been_healthy_later(tmp_path):
    result, vast, journal = run(tmp_path, {1: "slow"}, overall_timeout=0.2)
    assert result.winner is None
    assert vast.ours() == [] and not journal.state().live()


def test_interrupt_rolls_back_everything(tmp_path, monkeypatch):
    launch, vast, journal = start(tmp_path, {1: "slow", 2: "slow"})

    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(L, "wait", interrupted)
    with pytest.raises(KeyboardInterrupt):
        launch.run({})
    assert vast.creates == 2 and vast.ours() == []
    assert not journal.state().live()


# ------------------------------------------------- create with unknown outcome


def test_lost_create_response_is_found_by_label_and_destroyed(tmp_path):
    result, vast, journal = run(tmp_path, {1: "lost", 2: "ok"})
    assert result.winner.machine_id == 2
    assert vast.ours() == [result.winner.instance_id]  # the lost box did not leak
    assert journal.state().attempts[1].status == "destroyed"
    assert reasons(result)[1] is None


def test_unknown_create_blocks_its_slot_instead_of_being_replaced(tmp_path):
    # Slot 1 may hold a box we cannot see. With two slots, only one more may run.
    result, vast, journal = run(tmp_path, {1: "ghost", 2: "bad_gpu", 3: "ok", 4: "ok"})
    assert result.winner.machine_id == 3
    assert [a.machine_id for a in result.unresolved] == [1]
    assert journal.state().attempts[1].status == "unknown"
    assert 4 not in reasons(result)

    assert L.down(vast, journal, FAST) == 0  # nothing carries its label: never created
    state = journal.state()
    assert state.attempts[1].status == "absent" and state.closed
    assert vast.ours() == []


def test_box_that_shows_up_after_up_gave_up_looking_is_destroyed_by_down(tmp_path):
    result, vast, journal = run(tmp_path, {1: "lost_late", 2: "ok"})
    assert result.winner.machine_id == 2
    assert len(vast.ours()) == 2  # the winner, and a box `up` could not see
    vast.reveal()
    assert L.down(vast, journal, FAST) == 0
    assert vast.ours() == [] and foreign_untouched(vast)


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


# --------------------------------------------------------------------- delete


def test_transient_delete_failure_is_retried_until_confirmed(tmp_path):
    vast = FakeVast({1: "bad_gpu", 2: "ok"})
    vast.fail_destroy[1] = 2
    result, vast, journal = run(tmp_path, {1: "bad_gpu", 2: "ok"}, vast=vast, spare=0)
    assert result.winner.machine_id == 2
    assert vast.ours() == [result.winner.instance_id]
    assert journal.state().attempts[1].status == "destroyed"


def test_unconfirmed_delete_keeps_its_slot_and_is_reported(tmp_path):
    vast = FakeVast({1: "bad_gpu", 2: "ok", 3: "ok"})
    vast.sticky.add(1)
    result, vast, journal = run(tmp_path, {}, vast=vast, spare=0)
    # One slot, and its box will not go away: renting another would exceed the cap.
    assert result.winner is None and vast.creates == 1
    assert [a.machine_id for a in result.unresolved] == [1]
    assert journal.state().attempts[1].status == "destroying"

    assert L.down(vast, journal, FAST) == 1 and not journal.state().closed
    vast.sticky.clear()
    assert L.down(vast, journal, FAST) == 0 and journal.state().closed
    assert vast.ours() == []


# ----------------------------------------------------------------------- down


def test_down_destroys_by_id_even_if_the_box_relabelled_itself(tmp_path):
    result, vast, journal = run(tmp_path, {1: "ok"}, spare=0)
    vast.live[result.winner.instance_id]["label"] = "renamed-by-the-box"
    assert L.down(vast, journal, FAST) == 0
    assert vast.ours() == [] and foreign_untouched(vast)


def test_down_after_a_crash_between_intent_and_answer(tmp_path):
    # The process died after journalling the intent; the provider did rent the box.
    _, vast, journal = start(tmp_path, {1: "ok"})
    label = attempt_label("quack-ci.test.0123abcd", 1)
    journal.append("intent", seq=1, label=label, offer_id=1001, machine_id=1, host_id=1, dph=1.0)
    vast.create(1001, {"label": label})
    assert L.down(vast, journal, FAST) == 0
    assert vast.ours() == [] and foreign_untouched(vast)


def test_down_never_touches_another_run_in_the_same_namespace(tmp_path):
    result, vast, journal = run(tmp_path, {1: "ok"}, spare=0)
    assert L.down(vast, journal, FAST) == 0
    # 9003 is labelled quack-ci.other.89abcdef.a1: same namespace, not our run.
    assert 9003 in vast.live


def test_down_without_a_run_is_a_no_op(tmp_path):
    vast = FakeVast({})
    assert L.down(vast, Journal(tmp_path), FAST) == 0 and foreign_untouched(vast)


def test_second_process_on_a_state_directory_fails_fast(tmp_path):
    with state_lock(tmp_path):
        with pytest.raises(StateError, match="in use"):
            with state_lock(tmp_path):
                pass
    with state_lock(tmp_path):  # released again
        pass


# ----------------------------------------------------------------------- reap


def labelled(vast, label, age_min):
    import time

    iid = max(vast.live) + 1
    vast.live[iid] = {
        "id": iid,
        "label": label,
        "machine": 1,
        "start_date": time.time() - age_min * 60,
    }
    return iid


def test_reap_reports_unless_told_the_namespace_is_exclusive(tmp_path):
    vast = FakeVast({})
    old = labelled(vast, "quack-ci.run1.0123abcd.a1", age_min=200)
    assert L.reap(vast, "quack-ci", 90, destroy=False) == 0
    assert old in vast.live
    assert L.reap(vast, "quack-ci", 90, destroy=True) == 0
    assert old not in vast.live


def test_reap_matches_the_namespace_exactly_and_respects_age(tmp_path):
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
    assert L.reap(vast, "quack-ci", 90, destroy=True) == 0
    assert young in vast.live and all(i in vast.live for i in lookalikes)
    assert foreign_untouched(vast)


def test_reap_can_sweep_a_legacy_prefix_during_migration(tmp_path):
    vast = FakeVast({})
    legacy = labelled(vast, "quack-ci-123456-1", age_min=200)
    assert L.reap(vast, "quack-ci", 90, destroy=True) == 0 and legacy in vast.live
    assert L.reap(vast, "quack-ci", 90, destroy=True, also_prefix="quack-ci-") == 0
    assert legacy not in vast.live and 9002 in vast.live  # no start time: skipped
