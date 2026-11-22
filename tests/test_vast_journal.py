"""The Vast client's create contract, the bootstrap, labels and the journal."""

import io
import json
import shlex
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

from gpubox import vast as v
from gpubox.journal import Journal, StateError, attempt_label, label_namespace, new_run_id
from gpubox.launch import Spec

SPEC = Spec(
    image="tridao/quack-kernels:cu13.2-26.11.22",
    gpu="RTX 5090",
    num_gpus=2,
    max_dph_per_gpu=1.25,
    min_cuda=13.2,
    ttl_min=60,
)


class Wire:
    """Stands in for urllib: answers each request from a script and records it.

    An int is an HTTP error status, an exception is raised as is, a dict is a
    200 with that JSON body.
    """

    def __init__(self, answers):
        self.answers, self.requests = list(answers), []

    def __call__(self, req, timeout=None):
        self.requests.append((req.get_method(), req.full_url, req.data))
        answer = self.answers.pop(0)
        if isinstance(answer, int):
            raise urllib.error.HTTPError(
                req.full_url, answer, "err", {}, io.BytesIO(b'{"msg":"x"}')
            )
        if isinstance(answer, Exception):
            raise answer
        return io.BytesIO(json.dumps(answer).encode())  # BytesIO is a context manager


@pytest.fixture
def wire(monkeypatch):
    def make(*answers):
        w = Wire(answers)
        monkeypatch.setattr(v.urllib.request, "urlopen", w)
        monkeypatch.setattr(v.time, "sleep", lambda s: None)
        return w

    return make


def test_create_returns_the_contract_id(wire):
    w = wire({"success": True, "new_contract": 42})
    assert v.Vast("k").create(7, {"label": "x"}) == 42
    method, url, body = w.requests[0]
    assert (method, url) == ("PUT", "https://console.vast.ai/api/v0/asks/7/")
    assert json.loads(body) == {"label": "x"}


@pytest.mark.parametrize("status", [400, 404, 410])
def test_a_4xx_on_create_means_no_box(wire, status):
    w = wire(status)
    with pytest.raises(v.CreateRefused):
        v.Vast("k").create(7, {})
    assert len(w.requests) == 1


@pytest.mark.parametrize(
    "answer",
    [500, 502, 503, TimeoutError("timed out"), urllib.error.URLError("reset"), {"success": True}],
)
def test_no_usable_answer_on_create_is_unknown_and_never_retried(wire, answer):
    # A 5xx, a timeout or a success without an id: the box may exist. Sending
    # the request again could rent a second one.
    w = wire(answer, {"success": True, "new_contract": 1})
    with pytest.raises(v.CreateUnknown):
        v.Vast("k").create(7, {})
    assert len(w.requests) == 1


def test_an_explicit_refusal_in_a_200_is_refused(wire):
    wire({"success": False, "msg": "no_such_ask"})
    with pytest.raises(v.CreateRefused):
        v.Vast("k").create(7, {})


def test_reads_are_retried_and_listing_follows_pages(wire):
    w = wire(503, {"instances": [{"id": 1}], "next_token": "t"}, {"instances": [{"id": 2}]})
    assert [i["id"] for i in v.Vast("k").instances()] == [1, 2]
    assert "after_token" in w.requests[2][1] and "/v1/instances/" in w.requests[2][1]


def test_instance_distinguishes_gone_from_an_error(wire):
    wire({"instances": None})
    assert v.Vast("k").instance(5) is None
    wire(404)
    assert v.Vast("k").instance(5) is None
    wire(500, 500, 500)
    with pytest.raises(v.VastError):  # an outage is not evidence that the box is gone
        v.Vast("k").instance(5)


def test_destroying_something_already_gone_is_fine(wire):
    wire(404)
    v.Vast("k").destroy(5)


@pytest.mark.parametrize(
    "msg,kind",
    [
        ("Error response from daemon: manifest unknown", "image"),
        ("pull access denied for tridao/nope", "image"),
        ("failed to create shim task: OCI runtime create failed", "machine"),
        ("unresolvable CDI devices nvidia.com/gpu=GPU-1", "machine"),
        ("success, running tridao/quack-kernels:cu13.2", None),
        ("Pulling fs layer cdi5f3a", None),
    ],
)
def test_status_messages_are_attributed(msg, kind):
    assert v.classify_status(msg) == kind


def test_bootstrap_command_is_one_valid_shell_line():
    cmd = v.bootstrap_command("ssh-ed25519 AAAA'tricky gpubox", ttl_min=90)
    assert "\n" not in cmd
    bash, flag, script = shlex.split(cmd)
    assert (bash, flag) == ("bash", "-c")
    assert subprocess.run(["bash", "-n", "-c", script]).returncode == 0
    # The self-destruct timer starts before sshd replaces the shell, so it runs
    # even if nobody ever connects.
    timer, sshd = script.rsplit(" & ", 1)
    assert "sleep 90m; until python3 -c" in timer and "CONTAINER_API_KEY" in timer
    assert sshd.startswith("exec /usr/sbin/sshd -D")
    compile(v.SELF_DESTROY, "self_destroy", "exec")


def test_requests_match_the_launcher_proven_on_real_boxes():
    """Same search and create payloads as QuACK's tools/ci/vast/vast_ci.py."""
    old_dir = Path(__file__).resolve().parents[2] / "cute-kernels/tools/ci/vast"
    if not (old_dir / "vast_ci.py").exists():
        pytest.skip("reference launcher not checked out next to this repository")
    sys.path.insert(0, str(old_dir))
    import vast_ci as old

    ref = old.Spec(image=SPEC.image, label="quack-ci-x", num_gpus=2, ttl_min=60)
    assert v.offer_query(SPEC) == old.offer_query(ref)
    assert v.create_payload(SPEC, "quack-ci-x", "KEY") == old.create_payload(ref, "KEY")
    assert v.create_payload(SPEC, "a.b", "KEY")["label"] == "a.b"


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
    ],
)
def test_label_namespace_is_parsed_exactly(label, ns):
    assert label_namespace(label) == ns


def test_attempt_labels_parse_back():
    run_id = new_run_id("sim", "west")
    assert label_namespace(attempt_label(run_id, 3)) == "sim"


# ------------------------------------------------------------------- journal


def test_journal_replays_to_the_state_of_each_attempt(tmp_path):
    j = Journal(tmp_path)
    j.append("run", run_id="ns.n.0123abcd", namespace="ns", image="img")
    for seq in (1, 2, 3, 4):
        j.append(
            "intent", seq=seq, label=f"l{seq}", offer_id=seq, machine_id=seq, host_id=seq, dph=1
        )
    j.append("refused", seq=1)
    j.append("unknown", seq=2)
    j.append("created", seq=3, instance_id=30)
    j.append("created", seq=4, instance_id=40)
    j.append("destroying", instance_id=30)
    j.append("destroyed", instance_id=30)
    j.append("winner", seq=4, instance_id=40, host="h", port=22)
    s = Journal(tmp_path).state()
    assert {a.seq: a.status for a in s.attempts.values()} == {
        1: "refused", 2: "unknown", 3: "destroyed", 4: "created",
    }  # fmt: skip
    assert [a.seq for a in s.live()] == [2, 4]
    assert s.winner == {"seq": 4, "instance_id": 40, "host": "h", "port": 22}
    assert s.by_instance(30).machine_id == 3 and not s.closed


def test_an_intent_alone_counts_as_live(tmp_path):
    j = Journal(tmp_path)
    j.append("run", run_id="ns.n.0123abcd", namespace="ns", image="img")
    j.append("intent", seq=1, label="l1", offer_id=1, machine_id=1, host_id=1, dph=1)
    assert [a.status for a in j.state().live()] == ["intent"]


def test_a_torn_last_line_is_ignored(tmp_path):
    j = Journal(tmp_path)
    j.append("run", run_id="ns.n.0123abcd", namespace="ns", image="img")
    with open(j.path, "a") as f:
        f.write('{"ev": "intent", "seq": 1, "lab')
    assert j.state().run_id == "ns.n.0123abcd" and not j.state().attempts


def test_rotate_keeps_the_old_journal(tmp_path):
    j = Journal(tmp_path)
    j.append("run", run_id="ns.n.0123abcd", namespace="ns", image="img")
    j.rotate()
    assert not j.exists() and (tmp_path / "journal.ns.n.0123abcd.jsonl").exists()
