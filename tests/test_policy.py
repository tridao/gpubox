"""Observations, the blocklist derived from them, and offer selection."""

import pytest
from fakes import offer

from gpubox import policy as pol

H, D = pol.HOUR, pol.DAY


def obs(kind, machine, ts, reason=None):
    return pol.observation(kind, machine, machine, "run", reason=reason, now=ts)


def failed(machine, ts, reason):
    return obs("failed_admission", machine, ts, reason)


@pytest.mark.parametrize(
    "reason,cls,first",
    [
        ("busy_gpu", "soft", 2 * H),
        ("unreachable", "soft", 1 * D),
        ("no_network", "soft", 1 * D),
        ("died", "soft", 1 * D),
        ("runtime_error", "hard", 14 * D),
        ("wrong_hardware", "hard", 14 * D),
        ("cuda_init", "hard", 14 * D),
        ("compute_mismatch", "hard", 14 * D),
    ],
)
def test_first_offence_blocks_for_the_listed_time(reason, cls, first):
    rows = [failed(5, 1000, reason)]
    assert pol.blocklist(rows, 1000 + first - 1)[5] == {
        "until": 1000 + first,
        "class": cls,
        "reason": reason,
    }
    assert pol.blocklist(rows, 1000 + first + 1) == {}


def test_every_reason_code_has_a_policy():
    assert set(pol.REASONS) == {
        "busy_gpu", "unreachable", "no_network", "died",
        "runtime_error", "wrong_hardware", "cuda_init", "compute_mismatch",
    }  # fmt: skip


def test_repeats_within_the_window_double_up_to_the_cap():
    rows = [failed(5, 0, "busy_gpu"), failed(5, 3 * H, "busy_gpu")]
    assert pol.blocklist(rows, 3 * H + 1)[5]["until"] == 3 * H + 4 * H
    rows = [failed(5, i * H, "busy_gpu") for i in range(12)]
    assert pol.blocklist(rows, 11 * H + 1)[5]["until"] == 11 * H + 2 * D  # capped


def test_an_old_offence_does_not_count_as_a_repeat():
    rows = [failed(5, 0, "unreachable"), failed(5, pol.STRIKE_WINDOW + 1, "unreachable")]
    assert pol.blocklist(rows, pol.STRIKE_WINDOW + 2)[5]["until"] == pol.STRIKE_WINDOW + 1 + D


def test_healthy_clears_a_soft_block_but_not_a_hard_one():
    assert pol.blocklist([failed(5, 0, "busy_gpu"), obs("healthy", 5, 60)], 120) == {}
    rows = [failed(6, 0, "cuda_init"), obs("healthy", 6, 60)]
    assert pol.blocklist(rows, 120)[6]["class"] == "hard"


def test_healthy_does_not_erase_the_strike_history():
    rows = [failed(5, 0, "busy_gpu"), obs("healthy", 5, 60), failed(5, 120, "busy_gpu")]
    assert pol.blocklist(rows, 121)[5]["until"] == 120 + 4 * H


def test_hard_block_wins_over_a_concurrent_soft_one():
    rows = [failed(5, 0, "cuda_init"), failed(5, 10, "busy_gpu")]
    assert pol.blocklist(rows, 20)[5] == {"until": 14 * D, "class": "hard", "reason": "cuda_init"}


def test_order_of_rows_in_the_file_does_not_matter():
    rows = [failed(5, 0, "busy_gpu"), obs("healthy", 5, 60), failed(5, 120, "unreachable")]
    assert pol.blocklist(rows, 130) == pol.blocklist(rows[::-1], 130)


def test_rows_from_a_newer_policy_or_with_unknown_codes_are_ignored():
    newer = {**failed(5, 0, "cuda_init"), "policy_version": pol.POLICY_VERSION + 1}
    odd = {**failed(6, 0, "cuda_init"), "reason_code": "gremlins"}
    assert pol.blocklist([newer, odd], 10) == {}


def test_observations_reject_free_text_and_mismatched_reasons():
    with pytest.raises(ValueError):
        pol.observation("failed_admission", 1, 1, "r", reason="Error: token=abc at https://x")
    with pytest.raises(ValueError):
        pol.observation("failed_admission", 1, 1, "r")
    with pytest.raises(ValueError):
        pol.observation("healthy", 1, 1, "r", reason="died")
    with pytest.raises(ValueError):
        pol.observation("exploded", 1, 1, "r")


def test_observations_file_round_trips_and_survives_garbage(tmp_path):
    path = tmp_path / "obs.jsonl"
    store = pol.Observations(path)
    store.add(failed(5, 100, "cuda_init"))
    with open(path, "a") as f:
        f.write('{"torn": \n[1, 2]\n"text"\n')
    again = pol.Observations(path)
    assert [r["machine_id"] for r in again.rows] == [5] and again.added == 0
    again.add(obs("healthy", 6, 200))
    assert len(pol.Observations(path).rows) == 2


def test_missing_observations_file_is_empty(tmp_path):
    assert pol.Observations(tmp_path / "nope.jsonl").rows == []
    assert pol.Observations(None).rows == []


# ------------------------------------------------------------ offer selection


def ids(offers):
    return [o["machine_id"] for o in offers]


def test_offers_skip_blocked_machines_and_repeat_hosts():
    offers = [offer(1, host=7), offer(2, host=7), offer(3, host=8), offer(4, host=9)]
    hard = {"until": 9, "class": "hard", "reason": "cuda_init"}
    assert ids(pol.select_offers(offers, {3: hard}, want=2)) == [1, 4]
    # A blocked machine does not use up its host's slot.
    assert ids(pol.select_offers(offers, {1: hard}, want=2)) == [2, 3, 4]


def test_offers_without_a_host_id_are_not_collapsed():
    offers = [{"id": 1, "machine_id": 1, "host_id": None}, {"id": 2, "machine_id": 2}]
    assert ids(pol.select_offers(offers, {}, want=2)) == [1, 2]


def test_soft_blocked_machines_are_readmitted_last_when_the_pool_is_too_small():
    offers = [offer(1), offer(2), offer(3), offer(4)]
    blocked = {
        1: {"until": 500, "class": "soft", "reason": "busy_gpu"},
        2: {"until": 100, "class": "soft", "reason": "unreachable"},
        3: {"until": 900, "class": "hard", "reason": "cuda_init"},
    }
    # Enough unblocked candidates: the blocklist is honoured in full.
    assert ids(pol.select_offers([*offers, offer(5)], blocked, want=2)) == [4, 5]
    # Too few: soft blocks come back, soonest to expire first; the hard one never.
    assert ids(pol.select_offers(offers, blocked, want=2)) == [4, 2, 1]


def test_everything_hard_blocked_leaves_nothing():
    hard = {"until": 9, "class": "hard", "reason": "cuda_init"}
    assert pol.select_offers([offer(1), offer(2)], {1: hard, 2: hard}, want=2) == []


def test_a_row_seen_twice_counts_once():
    row = failed(5, 0, "busy_gpu")
    assert pol.blocklist([row, dict(row)], 1)[5]["until"] == 2 * H


def test_strikes_are_counted_per_class_not_per_reason():
    rows = [failed(5, 0, "busy_gpu"), failed(5, D, "unreachable")]
    assert pol.blocklist(rows, D + 1)[5]["until"] == D + 2 * D  # second soft strike
    rows = [failed(6, 0, "busy_gpu"), failed(6, D, "cuda_init")]
    assert pol.blocklist(rows, D + 1)[6]["until"] == D + 14 * D  # first hard strike


@pytest.mark.parametrize(
    "row",
    [
        {"machine_id": None, "ts": 1, "kind": "failed_admission", "reason_code": "cuda_init"},
        {"machine_id": "5", "ts": 1, "kind": "failed_admission", "reason_code": "cuda_init"},
        {"machine_id": 5, "ts": "soon", "kind": "failed_admission", "reason_code": "cuda_init"},
        {"machine_id": 5, "ts": 1, "kind": "exploded", "reason_code": "cuda_init"},
        {"machine_id": 5, "ts": 1, "kind": "failed_admission", "reason_code": "cuda_init", "policy_version": "x"},
        "text",
        None,
    ],
)  # fmt: skip
def test_malformed_cached_rows_are_skipped_everywhere(tmp_path, row):
    import json

    assert pol.blocklist([row], 2) == {}
    path = tmp_path / "observations.jsonl"
    path.write_text(json.dumps(row) + "\n")
    assert pol.Observations(path).rows == []


@pytest.mark.parametrize(
    "extra",
    [{"id": []}, {"id": {"a": 1}}, {"id": 7}, {"reason_code": ["cuda_init"]}, {"ts": float("nan")}],
)
def test_rows_with_fields_of_the_wrong_type_are_skipped_not_fatal(tmp_path, extra):
    import json

    row = {**failed(5, 1, "cuda_init"), **extra}
    assert pol.blocklist([row], 2) == {}
    path = tmp_path / "observations.jsonl"
    path.write_text(json.dumps(row).replace("NaN", "1e999") + "\n")
    assert pol.blocklist(pol.Observations(path).rows, 2) == {}


def test_a_cache_that_is_not_text_is_read_as_far_as_it_goes(tmp_path):
    import json

    path = tmp_path / "observations.jsonl"
    path.write_bytes(
        b"\xff\xfe\x00garbage\n" + json.dumps(failed(5, 1, "cuda_init")).encode() + b"\n"
    )
    assert [r["machine_id"] for r in pol.Observations(path).rows] == [5]


def test_an_observation_that_could_not_be_written_is_not_counted(tmp_path):
    store = pol.Observations(tmp_path / "missing-dir" / "file-in-the-way" / "obs.jsonl")
    (tmp_path / "missing-dir").mkdir()
    (tmp_path / "missing-dir" / "file-in-the-way").write_text("")  # mkdir will fail
    with pytest.raises(OSError):
        store.add(failed(5, 1, "cuda_init"))
    assert store.added == 0 and store.rows == []


def test_a_timestamp_too_large_to_be_a_time_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "observations.jsonl"
    path.write_text(
        '{"machine_id": 5, "ts": 1%s, "kind": "failed_admission", "reason_code": "cuda_init"}\n'
        % ("0" * 400)
    )
    assert pol.Observations(path).rows == []
    assert pol.blocklist([{**failed(5, 1, "cuda_init"), "ts": 10**400}], 2) == {}


def test_a_healthy_row_that_carries_a_reason_clears_nothing():
    odd = {**obs("healthy", 5, 60), "reason_code": "busy_gpu"}
    assert 5 in pol.blocklist([failed(5, 0, "busy_gpu"), odd], 120)
