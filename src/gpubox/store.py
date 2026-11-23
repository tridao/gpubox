"""The shared reputation store: other clients' observations of machines.

Advisory in every respect. A launch never waits more than `BUDGET` seconds for
it and never fails because of it; what it returns can only make an offer be
skipped (see policy.py), and nothing here touches ownership or cleanup.

Three files sit next to the local observations:

    outbox.jsonl   rows observed here and not yet accepted by the store
    remote.jsonl   the last complete read of everyone's rows (30 days)

A read replaces `remote.jsonl` only when every page arrived. A partial or
failed read leaves the previous snapshot in place.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .policy import KINDS, REASONS, _valid

BUDGET = 5.0  # seconds one sync may take, in total
BATCH = 50
OUTBOX_LIMIT = 1000
MAX_ROWS = 50_000  # more than this in 30 days is not a snapshot we want to hold
SENT_FIELDS = (
    "id", "provider", "machine_id", "host_id", "kind", "reason_code",
    "gpu", "num_gpus", "gate_version", "policy_version", "secs_to_ready",
)  # fmt: skip


class StoreError(RuntimeError):
    pass


class Store:
    def __init__(self, url: str, token: str):
        if not url.startswith("https://") and not url.startswith("http://127.0.0.1"):
            raise StoreError("the store URL must be https")
        self.url, self.token = url.rstrip("/"), token

    def _call(self, method: str, path: str, deadline: float, body=None):
        left = deadline - time.monotonic()
        if left <= 0.05:
            raise StoreError("out of time")
        data = None if body is None else json.dumps(body).encode()
        headers = {"Authorization": f"Bearer {self.token}", "User-Agent": "gpubox"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=left) as resp:
                answer = json.loads(resp.read(8 << 20))
        except urllib.error.HTTPError as exc:
            raise StoreError(f"{method} {path.split('?')[0]}: HTTP {exc.code}") from None
        except Exception as exc:  # timeouts, DNS, TLS, a body that is not JSON
            raise StoreError(f"{method} {path.split('?')[0]}: {type(exc).__name__}") from None
        if not isinstance(answer, dict):
            raise StoreError("unexpected answer")
        return answer

    def push(self, rows: list[dict], deadline: float) -> int:
        """Send rows; returns how many the store has now dealt with (accepted or known)."""
        done = 0
        for i in range(0, len(rows), BATCH):
            batch = [{k: r.get(k) for k in SENT_FIELDS} for r in rows[i : i + BATCH]]
            answer = self._call("POST", "/v1/observations", deadline, {"observations": batch})
            if type(answer.get("accepted")) is not int or type(answer.get("ignored")) is not int:
                raise StoreError("unexpected answer")
            done += len(batch)
        return done

    def pull(self, deadline: float, provider: str = "vast") -> list[dict]:
        """Everyone's rows of the last 30 days, or StoreError. Never a partial list."""
        rows, cursor, seen = [], "", set()
        while True:
            answer = self._call("GET", f"/v1/observations?provider={provider}{cursor}", deadline)
            page, nxt = answer.get("observations"), answer.get("next")
            if not isinstance(page, list):
                raise StoreError("unexpected answer")
            rows.extend(r for r in page if _usable(r))
            if nxt is None:
                return rows
            if not isinstance(nxt, str) or not nxt or nxt in seen or len(rows) > MAX_ROWS:
                raise StoreError("the listing does not end")
            seen.add(nxt)
            cursor = "&after=" + urllib.request.quote(nxt, safe="")


def _usable(row) -> bool:
    """A remote row the policy may see. Remote data is trusted no further than local."""
    if not _valid(row) or row["kind"] not in KINDS:
        return False
    return row["kind"] == "healthy" or row.get("reason_code") in REASONS


def from_env(environ=os.environ) -> Store | None:
    """The store named by GPUBOX_STORE_URL and GPUBOX_STORE_TOKEN, if both are set."""
    url, token = environ.get("GPUBOX_STORE_URL"), environ.get("GPUBOX_STORE_TOKEN")
    if not url or not token:
        return None
    return Store(url, token)


def _read(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for raw in path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if _valid(row):
                rows.append(row)
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    tmp.replace(path)


class Shared:
    """The outbox and the remote snapshot kept beside a local observations file."""

    def __init__(self, observations_path: Path, store: Store | None):
        base = Path(observations_path)
        self.outbox = base.with_name("outbox.jsonl")
        self.remote = base.with_name("remote.jsonl")
        self.store = store

    def queue(self, rows: list[dict]) -> None:
        """Remember rows to send. Bounded: the oldest are dropped, never the launch."""
        if not self.store or not rows:
            return
        _write(self.outbox, [*_read(self.outbox), *rows][-OUTBOX_LIMIT:])

    def remote_rows(self) -> list[dict]:
        return [r for r in _read(self.remote) if _usable(r)]

    def sync(self, *, pull: bool = True, budget: float | None = None) -> str:
        """Send the outbox, then (optionally) refresh the snapshot. Returns what happened.

        Raises StoreError if the store could not be used; callers treat that
        as a warning. Whatever was sent before the failure stays sent: rows are
        idempotent by id, so sending them again later is harmless.
        """
        if not self.store:
            return "no store configured"
        deadline = time.monotonic() + (BUDGET if budget is None else budget)
        waiting = _read(self.outbox)
        sent = 0
        if waiting:
            sent = self.store.push(waiting, deadline)
            _write(self.outbox, waiting[sent:])
        if not pull:
            return f"sent {sent}"
        rows = self.store.pull(deadline)
        _write(self.remote, rows)
        return f"sent {sent}, read {len(rows)}"
