"""The Vast client: what it sends, and how little it trusts what comes back."""

import http.client
import io
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

import pytest

from gpubox import vast as v
from gpubox.launch import Spec

REFERENCE = json.loads((Path(__file__).parent / "fixtures/reference_requests.json").read_text())
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
    last = {"instances": [{"id": 2, "label": None}], "next_token": None}
    w = wire(503, {"instances": [{"id": 1, "label": None}], "next_token": "t"}, last)
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
    cmd = v.bootstrap_command("ssh-ed25519 AAAA'tricky gpubox", 1795400000)
    assert "\n" not in cmd
    bash, flag, script = shlex.split(cmd)
    assert (bash, flag) == ("bash", "-c")
    assert subprocess.run(["bash", "-n", "-c", script]).returncode == 0
    # The self-destruct timer is the first thing started: before a missing
    # sshd is installed, and before sshd replaces the shell.
    timer, rest = script.split(" & ", 1)
    assert timer.startswith("[ -f /etc/gpubox-expiry ] || echo 1795400000 > /etc/gpubox-expiry; (")
    assert "CONTAINER_API_KEY" in timer
    assert rest.startswith("set -eu; ") and "apt-get" in rest
    assert rest.endswith("-o PasswordAuthentication=no") and "exec /usr/sbin/sshd -D" in rest
    compile(v.SELF_DESTROY, "self_destroy", "exec")


class Timer:
    """The bootstrap's timer, for real, with a stand-in for the delete and a short sleep.

    `in_file` is what the expiry file holds when the timer starts (None: no file yet).
    """

    def __init__(self, tmp_path, name, baked, in_file):
        _, _, script = shlex.split(v.bootstrap_command("KEY", baked))
        timer = script.split(" >/dev/null 2>&1 & ", 1)[0]
        assert timer.count("sleep 30") == 1 and timer.count("python3 -c") == 1
        quoted = shlex.quote(v.SELF_DESTROY)
        assert quoted in timer
        self.tries, self.file = tmp_path / f"{name}.tries", tmp_path / f"{name}.expiry"
        if in_file is not None:
            self.file.write_text(in_file)
        # The delete is replaced by a command that records the try and fails,
        # as a delete the provider ignores would.
        timer = timer.replace(f"python3 -c {quoted}", f"{{ echo try >> {self.tries}; false; }}")
        timer = timer.replace(v.EXPIRY_FILE, str(self.file)).replace("sleep 30", "sleep 0.05")
        self.proc = subprocess.Popen(["bash", "-c", timer], start_new_session=True)

    def fired(self) -> int:
        return len(self.tries.read_text().split()) if self.tries.exists() else 0

    def stop(self) -> bool:
        """Kill it and everything it started. True if it was still running."""
        running = self.proc.poll() is None
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc.wait()
        return running


def test_the_timer_waits_for_its_time_and_then_keeps_trying(tmp_path):
    now = int(time.time())
    cases = {
        "early": (now + 3600, None, False),  # before its time: nothing is sent
        "late": (now - 1, None, True),  # after it: the delete, again and again
        "renewed": (now - 1, f"{now + 3600}\n", False),  # the file says later: it waits
        "restarted": (now + 3600, f"{now - 1}\n", True),  # a file already there is kept
        "garbage": (now - 1, "soon\n", True),  # not a number: the first expiry applies
        "empty": (now - 1, "", True),
        # Digits, but too long for the shell to compare: without the length
        # check the test would error every time, and the timer never fire.
        "too-long": (now - 1, "9" * 25 + "\n", True),
        "ten-digits": (now - 1, "9999999999\n", False),
    }
    timers = {k: Timer(tmp_path, k, baked, held) for k, (baked, held, _) in cases.items()}
    try:
        time.sleep(0.8)
        fired = {k: t.fired() for k, t in timers.items()}
    finally:
        running = {k: t.stop() for k, t in timers.items()}
    assert all(running.values()), running  # the loop never ends by itself
    for name, (_, _, should_fire) in cases.items():
        assert (fired[name] >= 3) if should_fire else (fired[name] == 0), (name, fired)
    assert timers["early"].file.read_text().strip() == str(now + 3600)  # made with the first expiry


def test_renewing_while_the_timer_runs_moves_its_time(tmp_path):
    now = int(time.time())
    timer = Timer(tmp_path, "live", now + 3600, None)
    try:
        time.sleep(0.3)
        assert timer.fired() == 0
        cmd = v.set_expiry_command(now - 1).replace(v.EXPIRY_FILE, str(timer.file))
        assert subprocess.run(["bash", "-c", cmd]).returncode == 0
        time.sleep(0.5)
        assert timer.fired() >= 1  # the lease was moved into the past: it fires
    finally:
        timer.stop()


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"public_ipaddr": "1.2.3.4"},
        {"public_ipaddr": "1.2.3.4", "ports": None},
        {"public_ipaddr": "1.2.3.4", "ports": {"22/tcp": []}},
        {"public_ipaddr": "1.2.3.4", "ports": {"22/tcp": [{}]}},
        {"public_ipaddr": "1.2.3.4", "ports": {"22/tcp": [{"HostPort": "x"}]}},
        {"public_ipaddr": "1.2.3.4", "ports": {"22/tcp": [{"HostPort": "0"}]}},
        {"public_ipaddr": "1.2.3.4", "ports": {"22/tcp": [{"HostPort": "70000"}]}},
        {"public_ipaddr": "1.2.3.4", "ports": {"22/tcp": "2222"}},
        {"public_ipaddr": "", "ports": {"22/tcp": [{"HostPort": "22"}]}},
        {"public_ipaddr": "-oProxyCommand=x", "ports": {"22/tcp": [{"HostPort": "22"}]}},
        {"public_ipaddr": "1.2.3.4 evil", "ports": {"22/tcp": [{"HostPort": "22"}]}},
        {"public_ipaddr": "1.2.3.4", "ports": ["22/tcp"]},
    ],
)
def test_an_endpoint_we_cannot_read_is_none_not_an_error(row):
    assert v.endpoint_of(row) is None


def test_an_endpoint_is_the_address_and_the_port_mapped_to_22():
    row = {"public_ipaddr": " 1.2.3.4 ", "ports": {"22/tcp": [{"HostPort": "40123"}]}}
    assert v.endpoint_of(row) == ("1.2.3.4", 40123)


def test_relabel_sends_the_label_and_needs_an_explicit_yes(wire):
    w = wire({"success": True})
    v.Vast("k").relabel(5, "new-label")
    method, url, body = w.requests[0]
    assert (method, url) == ("PUT", "https://console.vast.ai/api/v0/instances/5/")
    assert json.loads(body) == {"label": "new-label"}
    for answer in ({}, {"success": False}, [], {"success": "true"}):
        wire(answer)
        with pytest.raises(v.VastError):
            v.Vast("k").relabel(5, "x")


@pytest.mark.parametrize("expiry", [0, -5, 10**10, "1795400000", 1795400000.5, None])
def test_a_timer_is_not_built_from_a_bad_expiry(expiry):
    with pytest.raises(ValueError):
        v.bootstrap_command("KEY", expiry)


def test_the_pinned_recording_still_matches_the_reference_launcher_if_present():
    """Optional: only meaningful in a checkout that sits next to QuACK."""
    old_dir = Path(__file__).resolve().parents[2] / "cute-kernels/tools/ci/vast"
    if not (old_dir / "vast_ci.py").exists():
        pytest.skip("reference launcher not checked out next to this repository")
    sys.path.insert(0, str(old_dir))
    import vast_ci as old

    ref = old.Spec(image=SPEC.image, label="LABEL", num_gpus=2, ttl_min=60)
    assert REFERENCE["search"]["body"] == old.offer_query(ref)
    assert REFERENCE["create"]["body"] == old.create_payload(ref, "KEY")


# ----------------------------------------------------- answers we cannot trust


def test_a_listing_row_without_a_label_field_fails_the_listing(wire):
    # Null or empty means untagged. Absent means we were not told, and the
    # reaper must not read that as "untagged".
    wire({"instances": [{"id": 1, "label": None}, {"id": 2, "label": ""}], "next_token": None})
    assert [r["id"] for r in v.Vast("k").instances()] == [1, 2]
    for row in ({"id": 3}, {"id": 3, "label": 7}, {"id": 3, "label": ["x"]}):
        wire({"instances": [{"id": 1, "label": "a"}, row], "next_token": None})
        with pytest.raises(v.VastError, match="label"):
            v.Vast("k").instances()


def test_an_id_listed_twice_fails_the_listing(wire):
    page = {"instances": [{"id": 1, "label": None}], "next_token": "t"}
    wire(page, {"instances": [{"id": 1, "label": None}], "next_token": None})
    with pytest.raises(v.VastError, match="twice"):
        v.Vast("k").instances()


def test_a_listing_must_have_as_many_rows_as_it_says(wire):
    rows = [{"id": 1, "label": None}, {"id": 2, "label": None}]
    wire({"instances": rows, "next_token": None, "total_instances": 2})
    assert len(v.Vast("k").instances()) == 2
    for total in (3, 1, "2", 2.0):
        wire({"instances": rows, "next_token": None, "total_instances": total})
        with pytest.raises(v.VastError, match="states"):
            v.Vast("k").instances()
    # Whichever page says it: a first page that promises three rows, and a
    # last page that says nothing, is a listing cut short.
    first = {"instances": rows[:1], "next_token": "t", "total_instances": 3}
    wire(first, {"instances": rows[1:], "next_token": None})
    with pytest.raises(v.VastError, match="states"):
        v.Vast("k").instances()
    wire({**first, "total_instances": 2}, {"instances": rows[1:], "next_token": None})
    assert len(v.Vast("k").instances()) == 2


def test_a_timeout_on_a_listing_is_for_all_its_pages_together(monkeypatch):
    clock, seen = [100.0], []

    def urlopen(req, timeout=None):
        seen.append(round(timeout, 3))
        clock[0] += 4  # each page takes four seconds
        n = len(seen)
        page = {"instances": [{"id": n, "label": None}], "next_token": f"t{n}"}
        return io.BytesIO(json.dumps(page).encode())

    monkeypatch.setattr(v.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(v.time, "monotonic", lambda: clock[0])
    with pytest.raises(v.VastError, match="did not finish in time"):
        v.Vast("k").instances(timeout=10, retries=0)
    assert seen == [10, 6, 2]  # what was left each time, and no fourth page


def test_the_account_is_the_id_the_provider_states(wire):
    w = wire({"id": 4711, "username": "ci"})
    assert v.Vast("k").account() == "4711"
    assert "/v0/users/current/" in w.requests[0][1]
    refused = {"success": False, "id": 4711, "msg": "unauthorized"}
    for answer in ({}, {"id": "4711"}, {"id": 0}, {"id": True}, [], {"user": {"id": 1}}, refused):
        wire(answer)
        with pytest.raises(v.VastError):
            v.Vast("k").account()


def test_a_caller_with_a_deadline_sets_the_timeout_and_does_its_own_retrying(monkeypatch):
    seen = []

    def urlopen(req, timeout=None):
        seen.append(timeout)
        raise urllib.error.HTTPError(req.full_url, 503, "err", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(v.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(v.time, "sleep", lambda s: None)
    vast = v.Vast("k", timeout=20)
    for call in (
        lambda: vast.instance(5, timeout=3, retries=0),
        lambda: vast.destroy(5, timeout=0.01, retries=0),
        lambda: vast.instances(timeout=99, retries=0),
    ):
        with pytest.raises(v.VastError):
            call()
    assert seen == [3, 0.5, 20]  # one request each; never below a floor, never above the client's
    seen.clear()
    with pytest.raises(v.VastError):
        vast.instance(5)
    assert seen == [20, 20, 20]  # without a deadline: the client's timeout, and its retries


@pytest.mark.parametrize(
    "answer",
    [
        {},  # an empty 2xx body
        [],
        {"new_contract": 7},  # an id without an explicit success
        {"success": True, "new_contract": True},
        {"success": True, "new_contract": 1.5},
        {"success": True, "new_contract": "7"},
        {"success": True, "new_contract": 0},
        {"success": "yes", "new_contract": 7},
        408,
        http.client.IncompleteRead(b"{"),
    ],
)
def test_anything_but_an_explicit_yes_or_no_on_create_is_unknown(wire, answer):
    w = wire(answer)
    with pytest.raises(v.CreateUnknown):
        v.Vast("k").create(7, {})
    assert len(w.requests) == 1


@pytest.mark.parametrize(
    "answer",
    [{}, [], {"success": False, "msg": "x"}, {"instances": {"id": 6}}, {"instances": [1]}],
)
def test_a_malformed_answer_is_never_read_as_the_instance_being_gone(wire, answer):
    wire(answer)
    with pytest.raises(v.VastError):
        v.Vast("k").instance(5)


def test_instance_returns_the_row_when_it_is_the_one_asked_for(wire):
    wire({"instances": {"id": 5, "actual_status": "running"}})
    assert v.Vast("k").instance(5)["actual_status"] == "running"


@pytest.mark.parametrize("second_page", [{}, {"instances": None}, {"success": False}, []])
def test_a_listing_with_an_unreadable_page_is_an_error_not_a_short_list(wire, second_page):
    wire({"instances": [{"id": 1, "label": None}], "next_token": "t"}, second_page)
    with pytest.raises(v.VastError):
        v.Vast("k").instances()


def test_an_explicitly_refused_destroy_is_an_error(wire):
    wire({"success": False, "msg": "nope"})
    with pytest.raises(v.VastError):
        v.Vast("k").destroy(5)
    wire({"success": True})
    v.Vast("k").destroy(5)


def test_the_wire_requests_equal_the_reference_launchers(wire):
    """What goes over HTTP for the QuACK CI spec, against a pinned recording."""
    ref = REFERENCE
    w = wire({"offers": []}, {"success": True, "new_contract": 9})
    vast = v.Vast("k")
    vast.offers(v.offer_query(SPEC))
    vast.create(123, v.create_payload(SPEC, "LABEL", "KEY", 1795400000))
    (m1, u1, b1), (m2, u2, b2) = w.requests
    assert (m1, u1) == ("POST", v.API_BASE + ref["search"]["path"])
    assert json.loads(b1) == ref["search"]["body"]
    assert (m2, u2) == ("PUT", v.API_BASE + ref["create"]["path"].format(offer_id=123))
    assert same_but_for_the_timer(json.loads(b2), ref["create"]["body"])


def same_but_for_the_timer(payload, reference):
    """The create payload equals the reference launcher's in everything but the timer.

    The reference starts its timer last (`(sleep 60m; until ...) &` just before
    sshd) and counts a duration; ours starts first and waits for an absolute
    time. Everything else in the bootstrap, and every other field, is equal.
    """
    ours, theirs = dict(payload), dict(reference)
    _, _, script = shlex.split(ours.pop("args_str"))
    _, _, old = shlex.split(theirs.pop("args_str"))
    timer, rest = script.split(" >/dev/null 2>&1 & ", 1)
    old_rest, old_tail = old.split("(sleep 60m; ", 1)
    old_timer, old_sshd = old_tail.split(" >/dev/null 2>&1 & ", 1)
    assert shlex.quote(v.SELF_DESTROY) in timer and shlex.quote(v.SELF_DESTROY) in old_timer
    return ours == theirs and rest == old_rest + old_sshd


def test_offers_in_an_excluded_country_are_dropped_and_unknown_locations_kept():
    offers = [
        {"id": 1, "geolocation": "Anhui, CN"},
        {"id": 2, "geolocation": "China, CN"},
        {"id": 3, "geolocation": "Taiwan, TW"},
        {"id": 4, "geolocation": "California, US"},
        {"id": 5, "geolocation": None},
        {"id": 6},
        {"id": 7, "geolocation": "CN"},
        {"id": 8, "geolocation": "Somewhere"},
    ]
    assert [o["id"] for o in v.in_allowed_country(offers, ("cn",))] == [3, 4, 5, 6, 8]
    assert v.in_allowed_country(offers, ()) == offers


def test_offer_query_scales_with_gpu_count():
    q = v.offer_query(Spec(image="i", gpu="RTX 5090", num_gpus=4, max_dph_per_gpu=1.25))
    assert q["num_gpus"] == {"eq": 4} and q["dph_total"] == {"lte": 5.0}
    assert q["cpu_cores_effective"] == {"gte": 32} and q["cpu_ram"] == {"gte": 4 * 32 * 1024}


class BrokenBody(io.BytesIO):
    def read(self, *a):
        raise http.client.IncompleteRead(b"")


@pytest.mark.parametrize("status,kind", [(503, v.CreateUnknown), (400, v.CreateRefused)])
def test_an_unreadable_error_body_does_not_change_what_the_status_means(monkeypatch, status, kind):
    # The status line arrived; only its description could not be read.
    calls = []

    def urlopen(req, timeout=None):
        calls.append(req)
        raise urllib.error.HTTPError(req.full_url, status, "err", {}, BrokenBody())

    monkeypatch.setattr(v.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(v.time, "sleep", lambda s: None)
    with pytest.raises(kind):
        v.Vast("k").create(7, {})
    assert len(calls) == 1
    with pytest.raises(v.VastError):
        v.Vast("k").destroy(5)


@pytest.mark.parametrize(
    "page",
    [
        {"instances": [None], "next_token": None},
        {"instances": [{"label": "no id"}], "next_token": None},
        {"instances": [{"id": "7"}], "next_token": None},
        {"instances": [{"id": True}], "next_token": None},
        {"instances": [{"id": 0}], "next_token": None},
        {"instances": [{"id": 1, "label": None}, "text"], "next_token": None},
    ],
)
def test_a_listing_with_an_unreadable_row_is_an_error_not_a_shorter_list(wire, page):
    wire(page)
    with pytest.raises(v.VastError):
        v.Vast("k").instances()


def test_a_listing_that_does_not_advance_is_an_error_not_a_loop(wire):
    wire(*[{"instances": [{"id": n, "label": None}], "next_token": "same"} for n in (1, 2, 3)])
    with pytest.raises(v.VastError, match="does not advance"):
        v.Vast("k").instances()


def test_offers_we_cannot_identify_or_price_are_not_candidates(wire):
    good = {"id": 1, "machine_id": 2, "dph_total": 0.5}
    bad = [
        {**good, "dph_total": None}, {**good, "dph_total": 0}, {**good, "dph_total": "0.5"},
        {**good, "dph_total": float("inf")}, {**good, "id": "1"}, {**good, "machine_id": None},
        "text", None,
    ]  # fmt: skip
    wire({"offers": [good, *bad]})
    assert v.Vast("k").offers({}) == [good]
    wire({"offers": None})
    with pytest.raises(v.VastError):
        v.Vast("k").offers({})


# ------------------------------------------------- round three: discriminators


@pytest.mark.parametrize("success", [0, "false", None, [], "no"])
def test_a_null_instance_under_a_bad_success_field_is_not_absence(wire, success):
    wire({"success": success, "instances": None})
    with pytest.raises(v.VastError):
        v.Vast("k").instance(5)


def test_the_two_real_shapes_of_absence_are_accepted(wire):
    wire({"instances": None})  # what the provider sends today
    assert v.Vast("k").instance(5) is None
    wire({"success": True, "instances": None})
    assert v.Vast("k").instance(5) is None


@pytest.mark.parametrize("token", [[], 0, False, "", {}, 7])
def test_only_null_ends_a_listing(wire, token):
    # Anything else in the cursor's place is not "the last page".
    wire({"success": True, "instances": [{"id": 1, "label": None}], "next_token": token})
    with pytest.raises(v.VastError):
        v.Vast("k").instances()


def test_a_listing_page_that_says_it_failed_is_an_error(wire):
    for success in (False, 0, "true", None):
        wire({"success": success, "instances": [], "next_token": None})
        with pytest.raises(v.VastError):
            v.Vast("k").instances()
    wire(
        {"instances": [{"id": 3, "label": None}], "next_token": None}
    )  # no discriminator at all is fine
    assert [r["id"] for r in v.Vast("k").instances()] == [3]


@pytest.mark.parametrize(
    "answer", [[], "ok", {"success": 0}, {"success": "true"}, {"success": None}]
)
def test_a_delete_is_done_only_if_the_answer_does_not_say_otherwise(wire, answer):
    wire(answer)
    with pytest.raises(v.VastError):
        v.Vast("k").destroy(5)


def test_offers_with_nonpositive_ids_are_not_candidates(wire):
    good = {"id": 1, "machine_id": 2, "dph_total": 0.5}
    wire({"offers": [good, {**good, "id": 0}, {**good, "machine_id": -4}, {**good, "id": True}]})
    assert v.Vast("k").offers({}) == [good]


def test_a_page_that_does_not_say_whether_more_follow_is_not_the_last_page(wire):
    # The provider ends a listing with `"next_token": null`. Without the field
    # we do not know that we have seen everything, and "nothing else carries
    # this label" is exactly what callers conclude from a listing.
    wire({"success": True, "instances": [{"id": 1, "label": None}]})
    with pytest.raises(v.VastError, match="next_token"):
        v.Vast("k").instances()
