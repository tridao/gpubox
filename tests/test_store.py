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

from gpubox import cli
from gpubox import policy as pol
from gpubox import store as st

TOKEN = "t0ken-" + "x" * 40
H = 3600


class FakeStore:
    """An HTTP server that behaves like the Worker, or misbehaves on request."""

    def __init__(self):
        self.rows, self.requests, self.revoked = [], [], []
        self.page_size = 2
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
                        owner.add(r, src="aaaa000000000001")
                    return self._send(200, {"accepted": len(new), "ignored": len(rows) - len(new)})
                after = int(self.path.split("after=")[1])
                page = [r for r in owner.rows if r["seq"] > after][: owner.page_size]
                more = len([r for r in owner.rows if r["seq"] > after]) > owner.page_size
                answer = {
                    "observations": page,
                    "next": page[-1]["seq"] if more else None,
                    "revoked": owner.revoked,
                }
                if mode == "endless":
                    answer["next"] = after
                if mode == "no_next":
                    del answer["next"]
                if mode == "fail_page_2" and after:
                    return self._send(500, {"error": "internal error"})
                return self._send(200, answer)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def add(self, row, src="bbbb000000000002", ts=None):
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
    return [r["machine_id"] for r in st._read(s.outbox)]


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


@pytest.mark.parametrize("name", ["remote.jsonl", "outbox.jsonl", "observations.jsonl.remote", "x"])
def test_the_side_files_never_are_the_observations_file(tmp_path, remote, name):
    mine = pol.Observations(tmp_path / name)
    mine.add(failed(1))
    before = (tmp_path / name).read_text()
    remote.add(failed(9))
    s = shared(tmp_path, remote, name=name)
    s.queue([failed(2)])
    s.sync()
    assert (tmp_path / name).read_text() == before
    assert len({tmp_path / name, s.outbox, s.remote, s.cursor, s._lockfile}) == 5
    other = shared(tmp_path, remote, name=name + "2")
    assert other.outbox != s.outbox and other.remote != s.remote


# ------------------------------------------------------------------- reading


def test_reading_is_incremental(tmp_path, remote):
    for m in range(1, 6):
        remote.add(failed(m))
    s = shared(tmp_path, remote)
    assert s.sync() == "sent 0, read 5"
    assert [r["machine_id"] for r in s.remote_rows()] == [1, 2, 3, 4, 5]
    remote.requests.clear()
    remote.add(failed(6))
    assert shared(tmp_path, remote).sync() == "sent 0, read 1"
    assert [r[1].split("after=")[1] for r in remote.requests] == ["5"]  # not from the start again
    assert [r["machine_id"] for r in s.remote_rows()] == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("mode", ["fail_page_2", "endless", "no_next"])
def test_an_interrupted_read_keeps_a_consistent_prefix_and_resumes(tmp_path, remote, mode):
    for m in range(1, 6):
        remote.add(failed(m))
    remote.mode = mode
    s = shared(tmp_path, remote)
    try:
        s.sync()
    except st.StoreError:
        pass
    got = [r["machine_id"] for r in s.remote_rows()]
    # A prefix, never a subset with holes; and it gave up at once, not when the budget ran out.
    assert got == list(range(1, len(got) + 1)) and len(got) < 5
    assert len(remote.requests) <= 3
    remote.mode = "ok"
    shared(tmp_path, remote).sync()
    assert [r["machine_id"] for r in s.remote_rows()] == [1, 2, 3, 4, 5]


def test_what_a_revoked_source_wrote_is_dropped(tmp_path, remote):
    remote.add(failed(1), src="aaaa")
    remote.add(failed(2), src="bbbb")
    s = shared(tmp_path, remote)
    s.sync()
    remote.revoked = ["aaaa"]
    remote.add(failed(3), src="bbbb")
    shared(tmp_path, remote).sync()
    assert [r["machine_id"] for r in s.remote_rows()] == [2, 3]


def test_rows_older_than_the_window_are_forgotten(tmp_path, remote):
    remote.add(failed(1), ts=time.time() - st.WINDOW + 60)
    remote.add(failed(2))
    s = shared(tmp_path, remote)
    s.sync()
    assert [r["machine_id"] for r in s.remote_rows()] == [1, 2]
    rows = st._read(s.remote)
    rows[0]["ts"] = time.time() - st.WINDOW - 2 * 86400  # time passes
    s.remote.write_text(st._lines(rows))
    assert [r["machine_id"] for r in s.remote_rows()] == [2]


@pytest.mark.parametrize(
    "change",
    [
        {"provider": "runpod"}, {"machine_id": "3"}, {"machine_id": 0}, {"kind": "exploded"},
        {"reason_code": "gremlins"}, {"kind": "healthy"}, {"ts": "yesterday"},
        {"ts": time.time() + 86400}, {"ts": time.time() - 40 * 86400}, {"ts": 10**400},
        {"id": 7}, {"id": None}, {"src": None}, {"src": ["a"]},
    ],
)  # fmt: skip
def test_remote_rows_are_trusted_less_than_local_ones(tmp_path, remote, change):
    good = remote.add(failed(1))
    remote.rows.append({**remote.add(failed(2)), **change})
    remote.rows.pop(1)
    remote.page_size = 50
    s = shared(tmp_path, remote)
    s.sync()
    assert [r["machine_id"] for r in s.remote_rows()] == [1]
    assert set(s.remote_rows()[0]) == set(st.KEPT_FIELDS) and good["id"] == s.remote_rows()[0]["id"]


@pytest.mark.parametrize("row", ["text", None, [1], 7, {}, {"seq": "7"}, {"seq": 0}, {"seq": True}])
def test_things_that_are_not_rows_are_not_rows(row):
    good = {"seq": 3, "id": "ab", "ts": time.time(), "src": "aaaa", "provider": "vast",
            "machine_id": 1, "kind": "failed_admission", "reason_code": "busy_gpu"}  # fmt: skip
    assert st.remote_row(good, time.time()) == good
    bad = {**good, **row} if isinstance(row, dict) and row else row
    assert st.remote_row(bad, time.time()) is None


def test_a_disabled_store_means_no_remote_rows_even_if_some_are_on_disk(tmp_path, remote):
    remote.add(failed(1))
    shared(tmp_path, remote).sync()
    assert st.Shared(tmp_path / "observations.jsonl", None).remote_rows() == []


def test_an_unreadable_copy_is_no_rows_not_an_error(tmp_path, remote):
    s = shared(tmp_path, remote)
    s.remote.mkdir()  # something is in the way
    assert s.remote_rows() == []
    s.cursor.write_text("not a number")
    assert s._read_cursor() == 0


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
    monkeypatch.setattr(cli, "Prober", prober)
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
    sent = [r for r in remote.rows if r["src"] == "aaaa000000000001"]
    assert [(r["machine_id"], r["kind"]) for r in sent] == [(2, "healthy")]
    assert st._read(tmp_path / "observations.jsonl.outbox") == []
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
    assert len(st._read(tmp_path / "observations.jsonl.outbox")) == 1  # kept for next time
    remote.mode = "ok"
    assert run(["down"]) == 0


def test_a_broken_copy_on_disk_does_not_fail_a_launch(world, remote, tmp_path):
    vast, run = world
    (tmp_path / "observations.jsonl.remote").mkdir(parents=True)
    assert run(UP) == 0 and len(vast.ours()) == 1
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
        assert (
            abs(got["ts"] - time.time()) < 60
            and len(got["src"]) == 16
            and "laptop" not in json.dumps(got)
        )
        assert set(pol.blocklist(theirs.remote_rows(), time.time())) == {11}
        mine.queue(rows)
        assert mine.sync() == "sent 3, read 0"  # resending changes nothing, reading resumes
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
