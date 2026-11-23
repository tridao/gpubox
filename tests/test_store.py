"""The client of the shared reputation store, against a fake of the Worker's API."""

import json
import os
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

TOKEN = "t" * 40


class FakeStore:
    """An HTTP server that behaves like the Worker, or misbehaves on request."""

    def __init__(self):
        self.rows, self.requests = [], []
        self.page_size = 2
        self.mode = "ok"  # ok | slow | error | garbage | endless | fail_page_2 | list
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, body):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle(self, method):
                body = self.rfile.read(int(self.headers.get("content-length") or 0))
                owner.requests.append((method, self.path, body, self.headers.get("authorization")))
                if self.headers.get("authorization") != f"Bearer {TOKEN}":
                    return self._send(401, {"error": "unauthorized"})
                if owner.mode == "slow":
                    time.sleep(1.5)
                if owner.mode == "error":
                    return self._send(500, {"error": "internal error"})
                if owner.mode == "garbage":
                    return self._send(200, b"<html>not json")
                if owner.mode == "list":
                    return self._send(200, [1, 2])
                if method == "POST":
                    rows = json.loads(body)["observations"]
                    known = {r["id"] for r in owner.rows}
                    new = [r for r in rows if r["id"] not in known]
                    owner.rows.extend({**r, "source": "other", "ts": time.time()} for r in new)
                    return self._send(200, {"accepted": len(new), "ignored": len(rows) - len(new)})
                start = int(self.path.split("after=")[1]) if "after=" in self.path else 0
                if owner.mode == "endless":
                    return self._send(200, {"observations": owner.rows[:1], "next": "7"})
                if owner.mode == "fail_page_2" and start:
                    return self._send(500, {"error": "internal error"})
                page = owner.rows[start : start + owner.page_size]
                more = start + owner.page_size < len(owner.rows)
                nxt = str(start + owner.page_size) if more else None
                return self._send(200, {"observations": page, "next": nxt})

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def remote():
    server = FakeStore()
    yield server
    server.close()


def failed(machine, reason="cuda_init", **extra):
    return pol.observation(
        "failed_admission", machine, machine, "run-x", reason=reason, gpu="RTX 5090", **extra
    )


def healthy(machine):
    return pol.observation("healthy", machine, machine, "run-x", secs_to_ready=50.0)


def shared(tmp_path, remote):
    return st.Shared(tmp_path / "observations.jsonl", st.Store(remote.url, TOKEN))


def soon(seconds=5):
    return time.monotonic() + seconds


# ------------------------------------------------------------------- sending


def test_only_the_agreed_fields_leave_the_machine(tmp_path, remote):
    row = {**failed(5), "run_id": "quack-ci.77-1.0123abcd", "detail": "token=abc at https://x"}
    st.Store(remote.url, TOKEN).push([row], soon())
    (sent,) = json.loads(remote.requests[0][2])["observations"]
    assert set(sent) == set(st.SENT_FIELDS)
    assert "run_id" not in sent and "ts" not in sent and "detail" not in sent
    assert sent["reason_code"] == "cuda_init" and sent["machine_id"] == 5


def test_rows_are_sent_in_batches_of_fifty(tmp_path, remote):
    assert st.Store(remote.url, TOKEN).push([failed(i + 1) for i in range(120)], soon()) == 120
    assert [len(json.loads(r[2])["observations"]) for r in remote.requests] == [50, 50, 20]


def test_the_outbox_empties_on_success_and_keeps_what_was_not_sent(tmp_path, remote):
    s = shared(tmp_path, remote)
    s.queue([failed(1), failed(2)])
    assert len(st._read(s.outbox)) == 2
    remote.mode = "error"
    with pytest.raises(st.StoreError):
        s.sync()
    assert len(st._read(s.outbox)) == 2  # nothing lost
    remote.mode = "ok"
    assert s.sync().startswith("sent 2")
    assert st._read(s.outbox) == [] and len(remote.rows) == 2
    assert s.sync().startswith("sent 0")  # and nothing is sent twice


def test_the_outbox_is_bounded(tmp_path, remote):
    s = shared(tmp_path, remote)
    s.queue([failed(i + 1) for i in range(st.OUTBOX_LIMIT + 30)])
    kept = st._read(s.outbox)
    assert len(kept) == st.OUTBOX_LIMIT and kept[-1]["machine_id"] == st.OUTBOX_LIMIT + 30


def test_nothing_is_queued_or_sent_without_a_store(tmp_path):
    s = st.Shared(tmp_path / "observations.jsonl", None)
    s.queue([failed(1)])
    assert not s.outbox.exists() and s.sync() == "no store configured" and s.remote_rows() == []


# ------------------------------------------------------------------- reading


def test_a_read_follows_every_page(tmp_path, remote):
    remote.rows = [{**failed(i + 1), "source": "ci"} for i in range(5)]
    s = shared(tmp_path, remote)
    assert s.sync() == "sent 0, read 5"
    assert [r["machine_id"] for r in s.remote_rows()] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("mode", ["error", "garbage", "list", "endless", "fail_page_2", "slow"])
def test_a_read_that_fails_leaves_the_last_complete_snapshot(tmp_path, remote, mode):
    remote.rows = [{**failed(i + 1), "source": "ci"} for i in range(3)]
    s = shared(tmp_path, remote)
    s.sync()
    remote.rows.append({**failed(9), "source": "ci"})
    remote.mode = mode
    with pytest.raises(st.StoreError):
        s.sync(budget=0.5)
    assert [r["machine_id"] for r in s.remote_rows()] == [1, 2, 3]  # not 1, 2 and not empty


def test_a_slow_store_costs_no_more_than_the_budget(tmp_path, remote):
    remote.mode = "slow"
    s = shared(tmp_path, remote)
    s.queue([failed(1)])
    t0 = time.monotonic()
    with pytest.raises(st.StoreError):
        s.sync(budget=0.3)
    assert time.monotonic() - t0 < 1.0
    assert len(st._read(s.outbox)) == 1


def test_remote_rows_are_trusted_no_further_than_local_ones(tmp_path, remote):
    good = {**failed(1), "source": "ci"}
    remote.rows = [
        good,
        {**failed(2), "reason_code": "gremlins"},
        {**failed(3), "machine_id": "3"},
        {**failed(4), "kind": "exploded"},
        {**healthy(5), "reason_code": "busy_gpu"},
        {**failed(6), "ts": "yesterday"},
        "text",
        None,
    ]
    remote.page_size = 50
    s = shared(tmp_path, remote)
    s.sync()
    assert [r["machine_id"] for r in s.remote_rows()] == [1]


def test_a_wrong_token_is_an_error_not_an_empty_store(tmp_path, remote):
    s = st.Shared(tmp_path / "observations.jsonl", st.Store(remote.url, "wrong" * 10))
    with pytest.raises(st.StoreError, match="401"):
        s.sync()


def test_the_store_must_be_https(monkeypatch):
    with pytest.raises(st.StoreError):
        st.Store("http://example.com", TOKEN)
    assert st.from_env({}) is None
    assert st.from_env({"GPUBOX_STORE_URL": "https://x.example"}) is None  # no token
    assert (
        st.from_env({"GPUBOX_STORE_URL": "https://x.example", "GPUBOX_STORE_TOKEN": TOKEN}).url
        == "https://x.example"
    )


# --------------------------------------------------------------- the policy


def test_what_others_saw_blocks_a_machine_here(tmp_path, remote):
    remote.rows = [{**failed(7), "source": "ci"}]
    s = shared(tmp_path, remote)
    s.sync()
    assert set(pol.blocklist(s.remote_rows(), time.time())) == {7}


def test_someone_elses_healthy_does_not_clear_our_own_hard_block(tmp_path, remote):
    now = time.time()
    ours = pol.observation("failed_admission", 7, 7, "r", reason="cuda_init", now=now - 100)
    remote.rows = [{**pol.observation("healthy", 7, 7, "r", now=now - 10), "source": "ci"}]
    s = shared(tmp_path, remote)
    s.sync()
    assert pol.blocklist([ours, *s.remote_rows()], now)[7]["class"] == "hard"


def test_our_own_row_coming_back_from_the_store_counts_once(tmp_path, remote):
    now = time.time()
    ours = pol.observation("failed_admission", 7, 7, "r", reason="busy_gpu", now=now)
    s = shared(tmp_path, remote)
    s.queue([ours])
    s.sync()
    both = [ours, *s.remote_rows()]
    assert len(both) == 2 and pol.blocklist(both, now + 1)[7]["until"] == pytest.approx(
        now + 2 * 3600, abs=5
    )


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
    return vast, lambda argv: cli.main([*argv, "--state-dir", str(tmp_path)])


def test_up_skips_a_machine_someone_else_reported_and_reports_its_own(world, remote, tmp_path):
    vast, run = world
    remote.rows = [{**failed(1), "source": "ci"}]
    assert run(UP) == 0
    assert vast.live[vast.ours()[0]]["machine"] == 2 and vast.creates == 1
    sent = [r for r in remote.rows if r.get("source") == "other"]
    assert [(r["machine_id"], r["kind"]) for r in sent] == [(2, "healthy")]
    assert st._read(tmp_path / "outbox.jsonl") == []
    assert run(["down"]) == 0


@pytest.mark.parametrize("mode", ["error", "garbage", "slow"])
def test_a_broken_store_never_fails_a_launch(world, remote, tmp_path, monkeypatch, capsys, mode):
    vast, run = world
    monkeypatch.setattr(st, "BUDGET", 0.3)
    remote.mode = mode
    assert run(UP) == 0 and len(vast.ours()) == 1
    assert "WARNING: reputation store" in capsys.readouterr().out
    assert len(st._read(tmp_path / "outbox.jsonl")) == 1  # kept for next time
    remote.mode = "ok"
    assert run(["down"]) == 0


def test_a_nonsense_store_url_is_a_warning(world, monkeypatch, capsys):
    vast, run = world
    monkeypatch.setenv("GPUBOX_STORE_URL", "http://example.com")
    assert run(UP) == 0
    assert "reputation store not used" in capsys.readouterr().out
    assert run(["down"]) == 0


def test_without_a_store_nothing_is_contacted(world, remote, monkeypatch):
    vast, run = world
    monkeypatch.delenv("GPUBOX_STORE_URL")
    assert run(UP) == 0 and remote.requests == []
    assert run(["down"]) == 0


# ----------------------------------------------- the real Worker, if built

STORE_DIR = Path(__file__).resolve().parents[1] / "store"


@pytest.mark.skipif(
    not (STORE_DIR / "node_modules" / "miniflare").exists(),
    reason="run `npm install` in store/ to test against the real Worker",
)
def test_the_client_and_the_real_worker_agree(tmp_path):
    """Python client -> the Worker's own code on a local D1 (Miniflare). No network."""
    proc = subprocess.Popen(
        ["node", "test/serve.mjs"], cwd=STORE_DIR, stdout=subprocess.PIPE, text=True,
        env={**os.environ, "TOKEN_LAPTOP": TOKEN},
    )  # fmt: skip
    try:
        url = proc.stdout.readline().strip()
        assert url.startswith("http://127.0.0.1"), url
        s = st.Shared(tmp_path / "observations.jsonl", st.Store(url, TOKEN))
        rows = [failed(11, reason="busy_gpu"), healthy(12), failed(11, reason="cuda_init")]
        s.queue(rows)
        assert (
            s.sync(budget=20) == "sent 3, read 2"
        )  # the second failure for machine 11 is spaced out
        got = s.remote_rows()
        assert [(r["machine_id"], r["kind"], r["source"]) for r in got] == [
            (11, "failed_admission", "laptop"),
            (12, "healthy", "laptop"),
        ]
        assert abs(got[0]["ts"] - time.time()) < 60  # the Worker's clock, in seconds
        assert set(pol.blocklist(got, time.time())) == {11}
        s.queue(rows)
        assert s.sync(budget=20) == "sent 3, read 2"  # resending changes nothing
        with pytest.raises(st.StoreError, match="401"):
            st.Shared(tmp_path / "o.jsonl", st.Store(url, "x" * 40)).sync(budget=20)
    finally:
        proc.kill()
        proc.wait()
