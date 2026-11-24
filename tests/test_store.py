"""The client of the shared reputation store, against a fake of the Worker's API."""

import json
import os
import queue
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fakes import FAST, FakeProber, FakeVast

from gpubox import api, cli
from gpubox import policy as pol
from gpubox import store as st

TOKEN = "t0ken-" + "x" * 40
H = 3600


class FakeStore:
    """An HTTP server that behaves like the Worker, or misbehaves on request."""

    def __init__(self):
        self.rows, self.requests = [], []
        self.mode = "ok"
        self.fail_posts_after = None  # POST number from which POSTs fail with 500
        self.reject_post = None  # POST number answered with 400
        self.on_post = None
        self.redirect_to = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, body, headers=()):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                for k, v in headers:
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def _handle(self, method):
                body = self.rfile.read(int(self.headers.get("content-length") or 0))
                owner.requests.append((method, self.path, body, self.headers.get("authorization")))
                mode = owner.mode
                if mode == "redirect":
                    return self._send(307, {}, [("location", owner.redirect_to + self.path)])
                if self.headers.get("authorization") != f"Bearer {TOKEN}":
                    return self._send(401, {"error": "unauthorized"})
                if mode == "slow":
                    time.sleep(1.5)
                if mode == "trickle":
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.end_headers()
                    for _ in range(40):  # a byte now and then: no single read ever times out
                        try:
                            self.wfile.write(b" ")
                            self.wfile.flush()
                        except OSError:
                            return
                        time.sleep(0.1)
                    return
                if mode == "error":
                    return self._send(500, {"error": "internal error"})
                if mode == "garbage":
                    return self._send(200, b"<html>not json")
                if mode == "list":
                    return self._send(200, [1, 2])
                if method == "POST":
                    n = sum(1 for r in owner.requests if r[0] == "POST")
                    if owner.on_post:
                        owner.on_post(n)
                    if owner.fail_posts_after is not None and n >= owner.fail_posts_after:
                        return self._send(500, {"error": "internal error"})
                    if owner.reject_post == n:
                        return self._send(400, {"error": "observation 0: bad id"})
                    if mode == "quota":
                        return self._send(429, {"error": "daily limit"})
                    rows = json.loads(body)["observations"]
                    if mode == "bad_counts":
                        return self._send(200, {"accepted": 0, "ignored": 0})
                    known = {r["id"] for r in owner.rows}
                    new = [r for r in rows if r["id"] not in known]
                    for r in new:
                        owner.add(r, src="laptop")
                    return self._send(200, {"accepted": len(new), "ignored": len(rows) - len(new)})
                if mode == "fail_reads":
                    return self._send(500, {"error": "internal error"})
                answer = {"observations": owner.rows, "truncated": False}
                if mode == "no_truncated":
                    del answer["truncated"]
                if mode == "rows_not_a_list":
                    answer["observations"] = {"1": owner.rows}
                return self._send(200, answer)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def add(self, row, src="quack_ci", ts=None):
        """Store a row the way the Worker would serve it."""
        served = {
            "seq": len(self.rows) + 1,
            "id": row["id"],
            "ts": time.time() if ts is None else ts,
            "src": src,
            "provider": "vast",
            "machine_id": row["machine_id"],
            "kind": row["kind"],
            "reason_code": row.get("reason_code"),
            "policy_version": 1,
        }
        self.rows.append(served)
        return served

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def remote():
    server = FakeStore()
    yield server
    server.close()


@pytest.fixture
def elsewhere():
    server = FakeStore()
    yield server
    server.close()


def failed(machine, reason="cuda_init", **extra):
    return pol.observation(
        "failed_admission", machine, machine, "run-x", reason=reason, gpu="RTX 5090", **extra
    )


def healthy(machine, **extra):
    return pol.observation("healthy", machine, machine, "run-x", secs_to_ready=50.0, **extra)


def client(remote, token=TOKEN):
    return st.Store(remote.url, token, allow_loopback_http=True)


def shared(tmp_path, remote, name="observations.jsonl", **kw):
    return st.Shared(tmp_path / name, client(remote), **kw)


def outbox(s):
    return [r["machine_id"] for r in st._read_lines(s.outbox)]


def posted(remote):
    return [len(json.loads(r[2])["observations"]) for r in remote.requests if r[0] == "POST"]


# ------------------------------------------------- where the token may travel


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com", "http://127.0.0.1.evil.example", "http://127.0.0.1@evil.example",
        "http://localhost:8787", "ftp://127.0.0.1", "https://user:pw@store.example",
        "https://store.example/?x=1", "https://store.example/#frag", "https://store.example/path",
        "https://", "store.example", "https://[::1", "",
        # Each of these is refused by one check only:
        "https://10.0.0.1",  # an address, not a name
        "https://store.example:08443",  # not the text we would rebuild
        "https://sto\tre.example",  # urlsplit would quietly drop the tab
        "https://store.example:0",
    ],
)  # fmt: skip
def test_urls_the_token_must_not_be_sent_to(url):
    with pytest.raises(st.StoreError) as exc:
        st.Store(url, TOKEN, allow_loopback_http=True)
    assert "evil" not in str(exc.value) and "pw" not in str(exc.value)  # nor is the URL echoed


def test_cleartext_is_for_tests_only():
    with pytest.raises(st.StoreError):
        st.Store("http://127.0.0.1:8787", TOKEN)  # what from_env does
    with pytest.raises(st.StoreError):
        st.from_env({"GPUBOX_STORE_URL": "http://127.0.0.1:8787", "GPUBOX_STORE_TOKEN": TOKEN})
    assert st.Store("https://store.example", TOKEN)._origin == "https://store.example"
    assert st.Store("https://store.example:8443/", TOKEN)._origin == "https://store.example:8443"
    assert st.from_env({}) is None
    assert st.from_env({"GPUBOX_STORE_URL": "https://store.example"}) is None  # no token
    with pytest.raises(st.StoreError):
        st.Store("https://store.example", "short")


def test_a_redirect_is_refused_and_the_token_goes_nowhere_else(tmp_path, remote, elsewhere):
    remote.mode, remote.redirect_to = "redirect", elsewhere.url
    s = shared(tmp_path, remote)
    s.queue([failed(1)])
    with pytest.raises(st.StoreError):
        s.sync()
    assert remote.requests and elsewhere.requests == []  # the other origin never heard from us


def test_a_proxy_from_the_environment_never_sees_the_token(
    tmp_path, remote, elsewhere, monkeypatch
):
    # With cleartext (tests only) a proxy would be handed the whole request.
    for name in (
        "http_proxy",
        "HTTP_PROXY",
        "https_proxy",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        monkeypatch.setenv(name, elsewhere.url)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    remote.add(failed(1))
    s = shared(tmp_path, remote)
    s.queue([failed(2)])
    assert s.sync() == "sent 1, read 2"  # the row that was there, and our own coming back
    assert elsewhere.requests == [] and len(remote.requests) == 2


@pytest.mark.parametrize("mode", ["error", "garbage", "slow", "redirect"])
def test_errors_never_carry_the_url_or_the_token(tmp_path, remote, elsewhere, mode):
    remote.mode, remote.redirect_to = mode, elsewhere.url
    with pytest.raises(st.StoreError) as exc:
        shared(tmp_path, remote, budget=0.4).sync()
    text = str(exc.value)
    assert (
        TOKEN not in text and "127.0.0.1" not in text and str(remote.server.server_port) not in text
    )


# ------------------------------------------------------------------- sending


def test_only_the_agreed_fields_leave_the_machine(remote):
    row = {**failed(5), "run_id": "quack-ci.77-1.0123abcd", "detail": "token=abc at https://x"}
    client(remote).push([row], time.monotonic() + 5)
    (sent,) = json.loads(remote.requests[0][2])["observations"]
    assert set(sent) == {
        "id", "provider", "machine_id", "host_id", "kind", "reason_code",
        "gpu", "num_gpus", "gate_version", "policy_version", "secs_to_ready",
    }  # fmt: skip
    assert sent["reason_code"] == "cuda_init" and sent["machine_id"] == 5


def test_each_accepted_batch_is_checked_off_even_if_a_later_one_fails(tmp_path, remote):
    s = shared(tmp_path, remote)
    s.queue([failed(i + 1) for i in range(120)])
    remote.fail_posts_after = 2
    account = s.sync()
    assert posted(remote) == [50, 50] and "HTTP 500" in account
    assert outbox(s) == list(range(51, 121))  # the first fifty are not sent again
    remote.fail_posts_after = None
    s = shared(tmp_path, remote)
    assert s.sync().startswith("sent 70")
    assert outbox(s) == [] and len({r["id"] for r in remote.rows}) == 120


def test_a_batch_the_store_will_never_take_does_not_block_the_rest(tmp_path, remote):
    s = shared(tmp_path, remote)
    s.queue([failed(i + 1) for i in range(120)])
    remote.reject_post = 1
    account = s.sync()
    assert "batch dropped" in account and outbox(s) == []
    assert sorted(r["machine_id"] for r in remote.rows) == list(range(51, 121))


def test_a_failed_send_does_not_stop_the_read(tmp_path, remote):
    remote.add(failed(9))
    s = shared(tmp_path, remote)
    s.queue([failed(1)])
    remote.fail_posts_after = 1
    assert "read 1" in s.sync()
    assert [r["machine_id"] for r in s.remote_rows()] == [9] and outbox(s) == [1]


def test_a_failure_whose_block_is_over_is_not_sent_late(tmp_path, remote):
    # Queued while the store was out of reach. The store times rows from when it
    # gets them: sent now, the busy GPU of three hours ago would block for two more.
    now = time.time()
    s = shared(tmp_path, remote)
    s.queue([
        failed(1, reason="busy_gpu", now=now - 3 * H),
        failed(2, reason="cuda_init", now=now - 3 * H),
        healthy(3, now=now - 3 * H),
    ])  # fmt: skip
    assert s.sync(pull=False) == "sent 2"
    assert sorted(r["machine_id"] for r in remote.rows) == [2, 3] and outbox(s) == []


def test_with_nothing_to_write_down_no_lock_is_waited_for(tmp_path, remote):
    import fcntl

    s = shared(tmp_path, remote)
    nothing = {"acked": [], "dropped": [], "read": [], "errors": []}
    with open(s._lockfile, "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # another gpubox process, writing
        t0 = time.monotonic()
        assert s._apply(nothing) == "sent 0"
        with pytest.raises(st.StoreError, match="HTTP 500"):
            s._apply({**nothing, "errors": ["POST /v1/observations: HTTP 500"]})
        assert time.monotonic() - t0 < 0.5


def test_runners_that_share_a_token_are_one_source(tmp_path, remote):
    # CI runner A saw machine 7's GPU busy and reported it; runner B, same token,
    # then saw it work. A's own copy of its report must not outvote the store.
    now = time.time()
    a_obs = pol.Observations(tmp_path / "a.jsonl")
    a_obs.add(failed(7, reason="busy_gpu", now=now - 600))
    a = shared(tmp_path, remote, name="a.jsonl")
    a.queue(a_obs.rows)
    a.sync()
    b = shared(tmp_path, remote, name="b.jsonl")
    b.queue([healthy(7, now=now - 60)])
    b.sync()
    a.sync()
    assert 7 in pol.blocklist(a_obs.rows, now)  # A alone still says busy...
    assert pol.blocklist(api.known_rows(a_obs, a), now) == {}  # ...the source as a whole does not
    # A row the store does not have yet stays this client's own word.
    a_obs.add(failed(8, reason="busy_gpu", now=now - 60))
    assert set(pol.blocklist(api.known_rows(a_obs, a), now)) == {8}


@pytest.mark.parametrize("mode", ["quota", "bad_counts", "error"])
def test_rows_stay_queued_unless_the_store_says_it_has_them(tmp_path, remote, mode):
    s = shared(tmp_path, remote)
    s.queue([failed(1), failed(2)])
    remote.mode = mode
    try:
        s.sync(pull=False)
    except st.StoreError:
        pass
    assert outbox(s) == [1, 2]


def test_rows_queued_while_we_were_talking_are_not_lost(tmp_path, remote):
    s = shared(tmp_path, remote)
    s.queue([failed(1)])
    another_process = shared(tmp_path, remote)
    remote.on_post = lambda n: another_process.queue([failed(2)])
    s.sync(pull=False)
    assert outbox(s) == [2]


def test_the_outbox_is_bounded_and_drops_the_oldest(tmp_path, remote):
    s = shared(tmp_path, remote)
    s.queue([failed(i + 1) for i in range(st.OUTBOX_LIMIT + 30)])
    assert outbox(s) == list(range(31, st.OUTBOX_LIMIT + 31))


def test_nothing_is_queued_or_sent_without_a_store(tmp_path):
    s = st.Shared(tmp_path / "observations.jsonl", None)
    s.queue([failed(1)])
    assert not s.outbox.exists() and s.sync() == "no store configured"


@pytest.mark.parametrize("name", ["remote.jsonl", "outbox.jsonl", "observations.jsonl", "x"])
def test_the_side_files_never_are_the_observations_file(tmp_path, remote, name):
    mine = pol.Observations(tmp_path / name)
    mine.add(failed(1))
    before = (tmp_path / name).read_text()
    remote.add(failed(9))
    s = shared(tmp_path, remote, name=name)
    s.queue([failed(2)])
    s.sync()
    assert (tmp_path / name).read_text() == before
    assert len({tmp_path / name, s.outbox, s.remote, s._lockfile}) == 4
    other = shared(tmp_path, remote, name=name + "2")
    assert other.outbox != s.outbox and other.remote != s.remote


@pytest.mark.parametrize("name", ["observations.jsonl.remote", "x.outbox", "y.lock"])
def test_an_observations_file_cannot_have_a_side_files_name(tmp_path, remote, name):
    # `x` and `x.remote` as two observations files would overwrite each other.
    with pytest.raises(st.StoreError):
        shared(tmp_path, remote, name=name)
    assert st.Shared(tmp_path / name, None).remote_rows() == []  # harmless without a store


def test_an_outbox_that_cannot_be_read_is_not_replaced(tmp_path, remote):
    s = shared(tmp_path, remote)
    s.outbox.mkdir()  # exists, and is not a readable file
    with pytest.raises(st.StoreError):
        s.queue([failed(1)])
    assert s.outbox.is_dir()


def test_a_busy_lock_costs_a_second_not_the_launch(tmp_path, remote):
    import fcntl

    s = shared(tmp_path, remote, budget=0.5)
    s.queue([failed(1)])
    with open(s._lockfile, "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # another gpubox process, suspended
        t0 = time.monotonic()
        with pytest.raises(st.StoreError, match="in use"):
            s.sync()
        with pytest.raises(st.StoreError):
            s.queue([failed(2)])
        assert time.monotonic() - t0 < 2.5
    assert remote.requests == []


# ------------------------------------------------------------------- reading


def test_a_read_replaces_the_copy_whole(tmp_path, remote):
    for m in range(1, 4):
        remote.add(failed(m))
    s = shared(tmp_path, remote)
    assert s.sync() == "sent 0, read 3"
    assert [r["machine_id"] for r in s.remote_rows()] == [1, 2, 3]
    # Rows the store no longer serves (a source cut off, its rows deleted) are gone here too.
    del remote.rows[0]
    remote.add(failed(4))
    assert shared(tmp_path, remote).sync() == "sent 0, read 3"
    assert [r["machine_id"] for r in s.remote_rows()] == [2, 3, 4]


@pytest.mark.parametrize(
    "mode", ["fail_reads", "no_truncated", "rows_not_a_list", "garbage", "list"]
)
def test_a_failed_read_keeps_the_previous_one(tmp_path, remote, mode):
    remote.add(failed(1))
    s = shared(tmp_path, remote)
    s.sync()
    remote.add(failed(2))
    remote.mode = mode
    with pytest.raises(st.StoreError):
        shared(tmp_path, remote).sync()
    assert [r["machine_id"] for r in s.remote_rows()] == [1]


def test_what_was_read_from_another_store_is_not_used(tmp_path, remote, elsewhere):
    remote.add(failed(1))
    shared(tmp_path, remote).sync()
    other = st.Shared(tmp_path / "observations.jsonl", client(elsewhere))
    assert other.remote_rows() == []
    elsewhere.add(failed(2))
    other.sync()
    assert [r["machine_id"] for r in other.remote_rows()] == [2]


def test_rows_older_than_the_window_are_forgotten(tmp_path, remote):
    remote.add(failed(1), ts=time.time() - st.WINDOW + 60)
    remote.add(failed(2))
    s = shared(tmp_path, remote)
    s.sync()
    assert [r["machine_id"] for r in s.remote_rows()] == [1, 2]
    doc = json.loads(s.remote.read_text())
    doc["rows"][0]["ts"] = time.time() - st.WINDOW - 60  # time passes
    s.remote.write_text(json.dumps(doc))
    assert [r["machine_id"] for r in s.remote_rows()] == [2]


@pytest.mark.parametrize(
    "change",
    [
        {"provider": "runpod"}, {"machine_id": "3"}, {"machine_id": 0}, {"kind": "exploded"},
        {"reason_code": "gremlins"}, {"kind": "healthy"}, {"ts": "yesterday"},
        {"ts": time.time() + 86400}, {"ts": time.time() - 30 * 86400 - 60}, {"ts": 10**400},
        {"policy_version": pol.POLICY_VERSION + 1}, {"policy_version": "1"},
        {"reason_code": ["busy_gpu"]}, {"reason_code": {"a": 1}},
        {"id": 7}, {"id": None}, {"src": None}, {"src": ["a"]},
    ],
)  # fmt: skip
def test_remote_rows_are_trusted_less_than_local_ones(tmp_path, remote, change):
    good = remote.add(failed(1))
    remote.rows.append({**remote.add(failed(2)), **change})
    remote.rows.pop(1)
    s = shared(tmp_path, remote)
    s.sync()
    assert [r["machine_id"] for r in s.remote_rows()] == [1]
    assert set(s.remote_rows()[0]) == set(st.KEPT_FIELDS) and good["id"] == s.remote_rows()[0]["id"]


@pytest.mark.parametrize(
    "row",
    [
        "text",
        None,
        [1],
        7,
        {"seq": "7"},
        {"seq": 0},
        {"seq": True},
        {"reason_code": ["x"]},
        "missing",
    ],
)
def test_things_that_are_not_rows_are_not_rows_and_never_raise(row):
    good = {"seq": 3, "id": "ab", "ts": time.time(), "src": "aaaa", "provider": "vast", "machine_id": 1,
            "kind": "failed_admission", "reason_code": "busy_gpu", "policy_version": 1}  # fmt: skip
    assert st.remote_row(good, time.time()) == good
    for field in good if row == "missing" else [None]:
        bad = {k: v for k, v in good.items() if k != field} if row == "missing" else row
        bad = {**good, **bad} if isinstance(bad, dict) and row != "missing" else bad
        assert st.remote_row(bad, time.time()) is None


def test_a_disabled_store_means_no_remote_rows_even_if_some_are_on_disk(tmp_path, remote):
    remote.add(failed(1))
    shared(tmp_path, remote).sync()
    assert st.Shared(tmp_path / "observations.jsonl", None).remote_rows() == []


def test_an_unreadable_copy_is_no_rows_not_an_error(tmp_path, remote):
    s = shared(tmp_path, remote)
    for junk in (
        "not json",
        "[1, 2]",
        '{"origin": 5}',
        json.dumps({"origin": s.store.identity(), "rows": "x"}),
        json.dumps({"origin": s.store.identity(), "rows": [1, None, "x"]}),
    ):
        s.remote.write_text(junk)
        assert s.remote_rows() == []
    s.remote.unlink()
    s.remote.mkdir()  # something is in the way
    assert s.remote_rows() == []


def test_a_wrong_token_is_an_error_not_an_empty_store(tmp_path, remote):
    s = st.Shared(tmp_path / "observations.jsonl", client(remote, "wrong-" + "y" * 40))
    with pytest.raises(st.StoreError, match="401"):
        s.sync()


# ------------------------------------------------------------------ the clock


def test_one_budget_covers_everything_a_command_says_to_the_store(tmp_path, remote):
    remote.mode = "slow"
    s = shared(tmp_path, remote, budget=0.5)
    s.queue([failed(1)])
    t0 = time.monotonic()
    for _ in range(3):  # `up` talks before and after the launch; three is more than enough
        with pytest.raises(st.StoreError):
            s.sync()
    assert time.monotonic() - t0 < 1.2  # not 3 x 0.5, let alone 3 x the server's 1.5
    assert outbox(s) == [1]


def test_a_server_that_trickles_bytes_cannot_hold_us_past_the_budget(tmp_path, remote):
    remote.mode = "trickle"  # every socket read succeeds, so a socket timeout never fires
    s = shared(tmp_path, remote, budget=0.4)
    t0 = time.monotonic()
    with pytest.raises(st.StoreError):
        s.sync()
    assert time.monotonic() - t0 < 1.0


# ---------------------------------------------------------------- the policy


def test_what_others_saw_blocks_a_machine_here(tmp_path, remote):
    remote.add(failed(7))
    s = shared(tmp_path, remote)
    s.sync()
    assert set(pol.blocklist(s.remote_rows(), time.time())) == {7}


def remote_like(row, src, seq=1):
    return {
        **{k: row.get(k) for k in ("id", "machine_id", "kind", "reason_code", "ts")},
        "src": src,
        "seq": seq,
        "provider": "vast",
    }


def test_someone_elses_healthy_does_not_unblock_a_machine_that_failed_here():
    now = time.time()
    for reason in ("busy_gpu", "unreachable", "cuda_init"):  # soft and hard alike
        ours = pol.observation("failed_admission", 7, 7, "r", reason=reason, now=now - 100)
        theirs = remote_like(pol.observation("healthy", 7, 7, "r", now=now - 10), "aaaa")
        assert 7 in pol.blocklist([ours, theirs], now), reason
    # Our own later success does take back our own soft verdict.
    ours = pol.observation("failed_admission", 7, 7, "r", reason="busy_gpu", now=now - 100)
    assert pol.blocklist([ours, pol.observation("healthy", 7, 7, "r", now=now - 10)], now) == {}


def test_a_source_takes_back_only_its_own_soft_failures():
    now = time.time()
    a_failed = remote_like(
        pol.observation("failed_admission", 7, 7, "r", reason="busy_gpu", now=now - 100), "aaaa"
    )
    b_failed = remote_like(
        pol.observation("failed_admission", 7, 7, "r", reason="busy_gpu", now=now - 90), "bbbb"
    )
    a_healthy = remote_like(pol.observation("healthy", 7, 7, "r", now=now - 10), "aaaa")
    assert 7 in pol.blocklist([a_failed, b_failed, a_healthy], now)  # b still says busy
    assert pol.blocklist([a_failed, a_healthy], now) == {}


def test_sources_do_not_add_up_to_a_longer_block():
    now = time.time()
    rows = [
        remote_like(
            pol.observation("failed_admission", 7, 7, "r", reason="busy_gpu", now=now - i),
            f"src{i}",
        )
        for i in range(1, 6)
    ]
    assert pol.blocklist(rows, now)[7]["until"] == pytest.approx(
        now - 1 + 2 * H, abs=1
    )  # one strike each


def test_our_own_row_coming_back_from_the_store_counts_once(tmp_path, remote):
    now = time.time()
    ours = pol.observation("failed_admission", 7, 7, "r", reason="busy_gpu", now=now)
    s = shared(tmp_path, remote)
    s.queue([ours])
    s.sync()
    both = [ours, *s.remote_rows()]
    assert len(both) == 2 and pol.blocklist(both, now + 1)[7]["until"] == pytest.approx(
        now + 2 * H, abs=5
    )


def test_our_own_recovery_takes_back_what_our_source_said_before(tmp_path):
    # A fresh CI runner: the failure it reported last week is known only from
    # the store, under its source's id. Today the machine works, and that row
    # comes back from the store too. The machine is not blocked any more.
    now = time.time()
    last_week = remote_like(
        pol.observation("failed_admission", 7, 7, "r", reason="busy_gpu", now=now - 60),
        "ci01",
        seq=1,
    )
    today = pol.observation("healthy", 7, 7, "r", now=now - 5)
    echo = remote_like(today, "ci01", seq=2)
    assert 7 in pol.blocklist([last_week], now)
    assert pol.blocklist([today, last_week, echo], now) == {}


def test_rows_of_one_moment_are_ordered_by_the_stores_sequence():
    # A failure and the recovery after it, sent in one request: same time.
    now = time.time()
    fail = remote_like(
        pol.observation("failed_admission", 7, 7, "r", reason="busy_gpu", now=now - 5),
        "ci01",
        seq=1,
    )
    heal = {
        **remote_like(pol.observation("healthy", 7, 7, "r", now=now - 5), "ci01", seq=2),
        "ts": fail["ts"],
    }
    for a, b in ((fail, heal), (heal, fail)):
        for ids in (("0" * 32, "f" * 32), ("f" * 32, "0" * 32)):  # whatever the ids are
            rows = [{**a, "id": ids[0]}, {**b, "id": ids[1]}]
            assert pol.blocklist(rows, now) == {}
    assert 7 in pol.blocklist([{**heal, "seq": 1}, {**fail, "seq": 2}], now)


def test_a_row_with_a_source_of_the_wrong_type_is_skipped():
    row = {**failed(7), "src": ["aaaa"]}
    assert pol.blocklist([row], time.time()) == {}


# --------------------------------------------------------------- the command

UP = [
    "up", "--gpu", "RTX 5090", "--max-dph-per-gpu", "1.25", "--image", "img",
    "--namespace", "quack-ci", "--name", "77-1", "--spare", "0",
]  # fmt: skip


@pytest.fixture
def world(tmp_path, monkeypatch, remote):
    vast = FakeVast({1: "ok", 2: "ok"})

    def prober(key_dir, create=True):
        key_dir.mkdir(parents=True, exist_ok=True)
        (key_dir / "id_ed25519").touch()
        return FakeProber(vast)

    monkeypatch.setenv("VASTAI_API_KEY", "k")
    monkeypatch.setenv("GPUBOX_STORE_URL", remote.url)
    monkeypatch.setenv("GPUBOX_STORE_TOKEN", TOKEN)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(cli.v, "Vast", lambda key: vast)
    monkeypatch.setattr(api, "Prober", prober)
    monkeypatch.setattr(cli, "Policy", lambda **kw: type(FAST)(**{**FAST.__dict__, **kw}))
    # The loopback fake speaks http; the command itself only ever accepts https.
    monkeypatch.setattr(
        st,
        "from_env",
        lambda environ=os.environ: st.Store(
            environ["GPUBOX_STORE_URL"], environ["GPUBOX_STORE_TOKEN"], allow_loopback_http=True
        )
        if environ.get("GPUBOX_STORE_URL")
        else None,
    )
    return vast, lambda argv: cli.main([*argv, "--state-dir", str(tmp_path)])


def test_up_skips_a_machine_someone_else_reported_and_reports_its_own(world, remote, tmp_path):
    vast, run = world
    remote.add(failed(1))
    assert run(UP) == 0
    assert vast.live[vast.ours()[0]]["machine"] == 2 and vast.creates == 1
    sent = [r for r in remote.rows if r["src"] == "laptop"]
    assert [(r["machine_id"], r["kind"]) for r in sent] == [(2, "healthy")]
    assert st._read_lines(tmp_path / "observations.jsonl.outbox") == []
    assert run(["down"]) == 0


def test_a_setup_failure_of_the_http_stack_is_a_warning(world, monkeypatch, capsys):
    vast, run = world

    def broken(*a, **k):
        raise OSError("cannot open the TLS key log")

    monkeypatch.setattr(st, "from_env", broken)
    assert run(UP) == 0 and "reputation store not used: OSError" in capsys.readouterr().out
    assert run(["down"]) == 0


@pytest.mark.parametrize("mode", ["error", "garbage", "slow", "trickle"])
def test_a_broken_store_costs_a_launch_a_few_seconds_and_nothing_else(
    world, remote, tmp_path, monkeypatch, capsys, mode
):
    vast, run = world
    monkeypatch.setattr(st, "BUDGET", 0.4)
    remote.mode = mode
    t0 = time.monotonic()
    assert run(UP) == 0 and len(vast.ours()) == 1
    assert time.monotonic() - t0 < 2.5  # both exchanges together, plus the fake launch
    assert "WARNING: reputation store" in capsys.readouterr().out
    assert len(st._read_lines(tmp_path / "observations.jsonl.outbox")) == 1  # kept for next time
    remote.mode = "ok"
    assert run(["down"]) == 0


def test_a_broken_copy_on_disk_does_not_fail_a_launch(world, remote, tmp_path):
    vast, run = world
    remote.mode = "fail_reads"  # so the broken copy is what the launch has to go on
    (tmp_path / "observations.jsonl.remote").write_text("{broken")
    assert run(UP) == 0 and len(vast.ours()) == 1
    assert (tmp_path / "observations.jsonl.remote").read_text() == "{broken"
    assert run(["down"]) == 0


def test_a_death_reaches_the_store_even_if_the_outputs_cannot_be_written(
    world, remote, tmp_path, monkeypatch
):
    from test_cli import fake_ssh, script

    vast, run = world
    assert run(UP) == 0
    fake_ssh(monkeypatch, rc=255)
    vast.live[vast.ours()[0]]["status"] = "exited"
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "no" / "such" / "dir" / "out"))
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert [r["kind"] for r in remote.rows] == ["healthy", "died_after_admission"]
    assert st._read_lines(tmp_path / "observations.jsonl.outbox") == []
    monkeypatch.delenv("GITHUB_OUTPUT")
    assert run(["down"]) == 0


def test_the_real_setting_refuses_cleartext_and_says_so_without_the_url(world, monkeypatch, capsys):
    vast, run = world
    monkeypatch.undo()  # back to the real from_env ...
    monkeypatch.setenv("VASTAI_API_KEY", "k")
    monkeypatch.setenv("GPUBOX_STORE_URL", "http://127.0.0.1.evil.example/secret-path")
    monkeypatch.setenv("GPUBOX_STORE_TOKEN", TOKEN)
    monkeypatch.setattr(cli.v, "Vast", lambda key: vast)
    assert cli.main(["search", "--gpu", "RTX 5090", "--max-dph-per-gpu", "1"]) == 0
    out = capsys.readouterr().out
    assert "reputation store not used" in out and "evil" not in out and TOKEN not in out


def test_without_a_store_nothing_is_contacted_and_old_copies_are_ignored(
    world, remote, monkeypatch, tmp_path
):
    vast, run = world
    remote.add(failed(1))
    assert run(UP) == 0 and run(["down"]) == 0  # machine 1 is skipped: the store says so
    assert vast.live == {k: b for k, b in vast.live.items() if b["machine"] is None}
    monkeypatch.delenv("GPUBOX_STORE_URL")
    remote.requests.clear()
    before = vast.creates
    assert run(UP) == 0 and remote.requests == []
    # Sharing is off: the copy left on disk must not decide anything any more.
    assert vast.creates == before + 1 and vast.live[vast.ours()[0]]["machine"] == 1
    assert run(["down"]) == 0


# ----------------------------------------------- the real Worker, if built

STORE_DIR = Path(__file__).resolve().parents[1] / "store"


@pytest.mark.skipif(
    not (STORE_DIR / "node_modules" / "miniflare").exists(),
    reason="run `npm ci` in store/ to test against the real Worker",
)
def test_the_client_and_the_real_worker_agree(tmp_path):
    """Python client -> the Worker's own code on a local D1 (Miniflare). No network."""
    other = "other-" + "z" * 40
    proc = subprocess.Popen(
        ["node", "test/serve.mjs"], cwd=STORE_DIR, stdout=subprocess.PIPE, text=True,
        env={**os.environ, "TOKEN_LAPTOP": TOKEN, "TOKEN_OTHER": other},
    )  # fmt: skip
    try:
        lines = queue.Queue()
        threading.Thread(target=lambda: lines.put(proc.stdout.readline()), daemon=True).start()
        url = lines.get(timeout=60).strip()  # a deadline: a server that never starts fails the test
        assert url.startswith("http://127.0.0.1:"), url
        mine = st.Shared(
            tmp_path / "mine.jsonl", st.Store(url, TOKEN, allow_loopback_http=True), budget=30
        )
        theirs = st.Shared(
            tmp_path / "theirs.jsonl", st.Store(url, other, allow_loopback_http=True), budget=30
        )
        rows = [failed(11, reason="busy_gpu"), healthy(12), failed(11, reason="cuda_init")]
        mine.queue(rows)
        # Three rows dealt with; one failure for machine 11 counts (the second is
        # spaced out), and the healthy row of an uncomplained-about machine is not served.
        assert mine.sync() == "sent 3, read 1"
        theirs.sync()
        (got,) = theirs.remote_rows()
        assert (got["machine_id"], got["kind"], got["reason_code"]) == (
            11,
            "failed_admission",
            "busy_gpu",
        )
        assert abs(got["ts"] - time.time()) < 60 and got["src"] == "laptop"
        assert set(pol.blocklist(theirs.remote_rows(), time.time())) == {11}
        mine.queue(rows)
        assert mine.sync() == "sent 3, read 1"  # resending changes nothing
        # The laptop sees machine 11 work again: that takes back its own verdict, for everyone.
        mine.queue([healthy(11)])
        mine.sync()
        theirs.sync()
        assert pol.blocklist(theirs.remote_rows(), time.time()) == {}
        with pytest.raises(st.StoreError, match="401"):
            st.Shared(
                tmp_path / "o.jsonl", st.Store(url, "x" * 40, allow_loopback_http=True), budget=30
            ).sync()
    finally:
        proc.kill()
        proc.wait(timeout=10)
