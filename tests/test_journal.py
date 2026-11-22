"""Labels, and the journal: what it remembers and what it refuses to forget."""

import pytest

from gpubox.journal import Journal, StateError, attempt_label, label_namespace, new_run_id

# -------------------------------------------------------------------- labels


def test_run_ids_are_unique_and_validated():
    a, b = new_run_id("quack-ci", "123-1"), new_run_id("quack-ci", "123-1")
    assert a != b and a.startswith("quack-ci.123-1.")
    for ns, name in [("Quack", "x"), ("quack.ci", "x"), ("", "x"), ("q", "a b"), ("q", "a.b")]:
        with pytest.raises(StateError):
            new_run_id(ns, name)


@pytest.mark.parametrize(
    "label,ns",
    [
        ("quack-ci.123-1.0123abcd.a1", "quack-ci"),
        ("quack-ci.123-1.0123abcd.a12", "quack-ci"),
        ("quack-ci-old.x.0123abcd.a1", "quack-ci-old"),
        ("quack-ci.123-1.0123abcd", None),
        ("quack-ci.123-1.0123abcd.a1.extra", None),
        ("quack-ci.123-1.XYZ.a1", None),
        ("quack-ci-123456-1", None),
        ("vast-5090-4x-ca", None),
        ("", None),
        (None, None),
        (17, None),
    ],
)
def test_label_namespace_is_parsed_exactly(label, ns):
    assert label_namespace(label) == ns


def test_attempt_labels_parse_back():
    assert label_namespace(attempt_label(new_run_id("sim", "west"), 3)) == "sim"


# -------------------------------------------------------------------- replay


def new_run(tmp_path, intents=1):
    j = Journal(tmp_path)
    j.append("run", run_id="ns.n.0123abcd", namespace="ns", image="img")
    for seq in range(1, intents + 1):
        j.append(
            "intent", seq=seq, label=f"l{seq}", offer_id=seq, machine_id=seq, host_id=seq, dph=1
        )
    return j


def view(j):
    s = j.state()
    return {a.seq: (a.status, a.instance_id) for a in s.attempts.values()}, s.boxes


def test_journal_replays_to_the_state_of_each_attempt(tmp_path):
    j = new_run(tmp_path, intents=5)
    j.append("refused", seq=1)
    j.append("unknown", seq=2)
    j.append("created", seq=3, instance_id=30)
    j.append("created", seq=4, instance_id=40)
    j.append("destroying", instance_id=30, seq=3)
    j.append("destroyed", instance_id=30, seq=3)
    j.append("winner", seq=4, instance_id=40, host="h", port=22)
    j.append("abandoned", seq=5)
    s = Journal(tmp_path).state()
    assert view(j)[0] == {
        1: ("refused", None), 2: ("unknown", None), 3: ("destroyed", 30),
        4: ("created", 40), 5: ("abandoned", None),
    }  # fmt: skip
    assert s.live_ids() == [40] and [a.seq for a in s.open_attempts()] == [2]
    assert s.winner == {"seq": 4, "instance_id": 40, "host": "h", "port": 22}
    assert s.attempt_of(30).machine_id == 3 and not s.closed and not s.problems
    assert s.attempts[1].ts > 0


def test_an_intent_alone_is_live(tmp_path):
    s = new_run(tmp_path).state()
    assert s.is_live() and [a.status for a in s.open_attempts()] == ["intent"]


def test_every_id_in_the_journal_is_ours_even_without_its_intent(tmp_path):
    # `created` before (or without) its intent: the id must not be dropped.
    j = Journal(tmp_path)
    j.append("run", run_id="ns.n.0123abcd", namespace="ns", image="img")
    j.append("created", seq=1, instance_id=10)
    j.append("destroying", instance_id=11)
    s = j.state()
    assert s.live_ids() == [10, 11] and s.attempts == {} and s.is_live()


def test_a_cleanup_record_binds_the_id_when_created_was_lost(tmp_path):
    j = new_run(tmp_path)
    j.append("destroying", instance_id=42, seq=1)  # all we have about instance 42
    assert view(j) == ({1: ("destroying", 42)}, {42: "destroying"})
    assert not j.state().open_attempts()  # the intent is resolved: it is that box


def test_the_winner_record_binds_the_id_when_created_was_lost(tmp_path):
    j = new_run(tmp_path)
    j.append("winner", seq=1, instance_id=10, host="h", port=22)
    assert view(j) == ({1: ("created", 10)}, {10: "created"})


def test_replay_never_loses_or_swaps_an_id(tmp_path):
    j = new_run(tmp_path)
    j.append("created", seq=1, instance_id=10)
    j.append("intent", seq=1, label="l1", offer_id=1, machine_id=1, host_id=1, dph=1)  # duplicate
    j.append("unknown", seq=1)  # late, out of order
    j.append("refused", seq=1)
    j.append("created", seq=1, instance_id=99)  # a second id for the same attempt
    s = j.state()
    assert view(j)[0] == {1: ("created", 10)}
    assert s.live_ids() == [10, 99]  # both were given to us: both are ours
    assert s.problems  # and the contradiction is reported, not hidden


def test_a_destroyed_box_does_not_come_back(tmp_path):
    j = new_run(tmp_path)
    j.append("created", seq=1, instance_id=10)
    j.append("destroyed", instance_id=10)
    j.append("destroying", instance_id=10)
    j.append("created", seq=1, instance_id=10)
    assert view(j) == ({1: ("destroyed", 10)}, {10: "destroyed"})


def test_closed_is_not_believed_while_anything_is_live(tmp_path):
    j = new_run(tmp_path)
    j.append("closed")
    assert not j.state().closed  # an open intent
    j.append("created", seq=1, instance_id=10)
    assert not j.state().closed  # a live box
    j.append("destroyed", instance_id=10)
    assert j.state().closed


@pytest.mark.parametrize("bad_id", [True, 0, -3, 1.5, "10", None, [10]])
def test_a_malformed_instance_id_is_a_problem_not_an_id(tmp_path, bad_id):
    j = new_run(tmp_path)
    j.append("created", seq=1, instance_id=10)
    with open(j.path, "a") as f:
        import json

        f.write(json.dumps({"ev": "destroyed", "instance_id": bad_id}) + "\n")
    s = j.state()
    assert s.boxes == {10: "created"} and s.problems


# -------------------------------------------------------------------- damage


def test_a_torn_last_line_is_tolerated(tmp_path):
    j = new_run(tmp_path, intents=0)
    with open(j.path, "a") as f:
        f.write('{"ev": "intent", "seq": 1, "lab')
    s = j.state()
    assert s.run_id == "ns.n.0123abcd" and not s.attempts and not s.problems


def test_appending_after_a_torn_tail_loses_neither_record(tmp_path):
    j = new_run(tmp_path)
    j.append("created", seq=1, instance_id=10)
    with open(j.path, "a") as f:
        f.write('{"ev": "destroying", "instance_')  # the process died mid-write
    # The very next record must survive: glued to the fragment it would be
    # unreadable, and the box would be taken for still alive.
    j.append("destroyed", instance_id=10)
    s = j.state()
    assert s.boxes == {10: "destroyed"} and not s.problems
    j.append("closed")
    assert j.state().closed


def test_damage_in_the_middle_is_reported_and_blocks_closing(tmp_path):
    j = new_run(tmp_path)
    j.append("created", seq=1, instance_id=10)
    lines = j.path.read_text().splitlines()
    lines.insert(1, '{"ev": "intent", "seq": 9, "lab')
    j.path.write_text("\n".join(lines) + "\n")
    j.append("destroyed", instance_id=10)
    j.append("closed")
    s = j.state()
    assert s.problems and not s.closed and s.boxes == {10: "destroyed"}
    with pytest.raises(StateError, match="not finished"):
        j.rotate()


@pytest.mark.parametrize(
    "line",
    ['[1, 2]', '"text"', '{"no_ev": 1}', '{"ev": 7}', '{"ev": "teleported"}',
     '{"ev": "created", "seq": 1}', '{"ev": "refused", "seq": 99}', '{"ev": "intent"}'],
)  # fmt: skip
def test_records_that_cannot_be_applied_are_problems_not_crashes(tmp_path, line):
    j = new_run(tmp_path)
    with open(j.path, "a") as f:
        f.write(line + "\n")
    j.append("created", seq=1, instance_id=10)  # replay carries on past the bad record
    s = j.state()
    assert s.problems and s.boxes == {10: "created"}


def test_an_intent_without_a_usable_timestamp_has_none(tmp_path):
    j = Journal(tmp_path)
    j.append("run", run_id="ns.n.0123abcd", namespace="ns", image="img")
    with open(j.path, "a") as f:
        f.write('{"ev": "intent", "seq": 1, "label": "l1"}\n')
        f.write('{"ev": "intent", "seq": 2, "label": "l2", "ts": "noon"}\n')
    assert [a.ts for a in j.state().attempts.values()] == [None, None]


# ------------------------------------------------------------------ rotation


def test_rotate_keeps_the_old_journal_and_refuses_a_live_one(tmp_path):
    j = new_run(tmp_path)
    with pytest.raises(StateError, match="not finished"):
        j.rotate()
    j.append("refused", seq=1)
    j.rotate()
    assert not j.exists() and (tmp_path / "journal.ns.n.0123abcd.jsonl").exists()


def test_a_named_run_is_found_after_rotation(tmp_path):
    j = new_run(tmp_path)
    j.append("created", seq=1, instance_id=10)
    j.append("destroyed", instance_id=10)
    j.rotate()
    Journal(tmp_path).append("run", run_id="ns.n.89abcdef", namespace="ns", image="img")
    assert Journal(tmp_path, run_id="ns.n.0123abcd").state().boxes == {10: "destroyed"}
    assert Journal(tmp_path, run_id="ns.n.89abcdef").state().run_id == "ns.n.89abcdef"
    with pytest.raises(StateError):
        Journal(tmp_path, run_id="ns.n.00000000")


def test_an_archive_whose_header_names_another_run_is_not_that_run(tmp_path):
    new_run(tmp_path)
    (tmp_path / "journal.ns.n.11111111.jsonl").write_text(
        '{"ev": "run", "run_id": "ns.n.22222222", "namespace": "ns"}\n'
    )
    with pytest.raises(StateError, match="no journal for run"):
        Journal(tmp_path, run_id="ns.n.11111111")


def test_a_run_id_cannot_name_a_file_outside_the_state_directory(tmp_path):
    state = tmp_path / "state"
    (state / "journal.x").mkdir(parents=True)
    (tmp_path / "elsewhere.jsonl").write_text(
        '{"ev": "run", "run_id": "a.b.0123abcd", "namespace": "a"}\n'
    )
    with pytest.raises(StateError, match="not a run id"):
        Journal(state, run_id="x/../../elsewhere")
