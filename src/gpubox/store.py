"""The shared reputation store: what other clients observed about machines.

Advisory in every respect. All exchanges of one command together may take at
most `BUDGET` seconds of wall-clock time and can never fail it; what the store
returns can only make an offer be skipped (see policy.blocklist); nothing here
touches ownership or cleanup.

Beside an observations file `NAME` live three more files:

    NAME.outbox   rows observed here and not yet accepted by the store
    NAME.remote   rows read from the store so far (at most 30 days of them)
    NAME.cursor   how far that reading has got

Reading is incremental: the store numbers rows in commit order and we ask for
what comes after the last number we have. So an interrupted read leaves a
shorter, still consistent, picture, and the next one continues from there.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from .policy import REASONS

BUDGET = 5.0  # seconds all exchanges of one command may take, together
BATCH = 50
OUTBOX_LIMIT = 1000  # beyond this the oldest unsent rows are dropped
WINDOW = 30 * 86400  # rows older than this are forgotten
MAX_REMOTE_ROWS = 50_000
SENT_FIELDS = (
    "id", "provider", "machine_id", "host_id", "kind", "reason_code",
    "gpu", "num_gpus", "gate_version", "policy_version", "secs_to_ready",
)  # fmt: skip
KEPT_FIELDS = ("seq", "id", "ts", "src", "provider", "machine_id", "kind", "reason_code")


class StoreError(RuntimeError):
    """Something about the store went wrong. The message never contains the URL or the token."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """The token goes to the configured origin and nowhere else."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


class Store:
    def __init__(self, url: str, token: str, *, allow_loopback_http: bool = False):
        """`allow_loopback_http` exists for tests against a server on this machine."""
        try:
            parts = urllib.parse.urlsplit(url)
            host, port = parts.hostname, parts.port
        except ValueError:
            raise StoreError("the store URL cannot be parsed") from None
        loopback = allow_loopback_http and parts.scheme == "http" and host == "127.0.0.1"
        if not (parts.scheme == "https" or loopback) or not host:
            raise StoreError("the store URL must be https://HOST[:PORT]")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise StoreError("the store URL must not carry credentials, a query or a fragment")
        if parts.path not in ("", "/"):
            raise StoreError("the store URL must not carry a path")
        if not isinstance(token, str) or len(token) < 32 or not token.isprintable():
            raise StoreError("the store token is not usable")
        self._origin = f"{parts.scheme}://{host}" + (f":{port}" if port else "")
        self._token = token
        self._opener = urllib.request.build_opener(_NoRedirect)

    def _call(self, method: str, path: str, deadline: float, body=None) -> dict:
        what = f"{method} {path.split('?')[0]}"
        try:
            left = deadline - time.monotonic()
            if left <= 0.05:
                raise StoreError("out of time")
            data = None if body is None else json.dumps(body).encode()
            headers = {"Authorization": f"Bearer {self._token}", "User-Agent": "gpubox"}
            if data is not None:
                headers["Content-Type"] = "application/json"
            req = urllib.request.Request(
                self._origin + path, data=data, method=method, headers=headers
            )
            with self._opener.open(req, timeout=left) as resp:
                answer = json.loads(resp.read(4 << 20))
        except StoreError:
            raise
        except urllib.error.HTTPError as exc:
            raise StoreError(f"{what}: HTTP {exc.code}") from None
        except Exception as exc:  # timeouts, DNS, TLS, a body that is not JSON
            raise StoreError(f"{what}: {type(exc).__name__}") from None
        if not isinstance(answer, dict):
            raise StoreError(f"{what}: unexpected answer")
        return answer

    def push(self, batch: list[dict], deadline: float) -> None:
        """Send one batch. Returns only if the store has dealt with every row of it."""
        rows = [{k: r.get(k) for k in SENT_FIELDS} for r in batch]
        answer = self._call("POST", "/v1/observations", deadline, {"observations": rows})
        accepted, ignored = answer.get("accepted"), answer.get("ignored")
        if type(accepted) is not int or type(ignored) is not int:
            raise StoreError("POST /v1/observations: unexpected answer")
        if accepted < 0 or ignored < 0 or accepted + ignored != len(rows):
            raise StoreError("POST /v1/observations: the counts do not add up")

    def page(self, after: int, deadline: float, provider: str = "vast"):
        """One page after sequence number `after`: (rows, next or None, revoked sources)."""
        answer = self._call("GET", f"/v1/observations?provider={provider}&after={after}", deadline)
        rows, revoked = answer.get("observations"), answer.get("revoked")
        if not isinstance(rows, list) or "next" not in answer or not isinstance(revoked, list):
            raise StoreError("GET /v1/observations: unexpected answer")
        nxt = answer["next"]
        if nxt is not None and (type(nxt) is not int or nxt <= after):
            raise StoreError("GET /v1/observations: the listing does not advance")
        if not all(isinstance(s, str) for s in revoked):
            raise StoreError("GET /v1/observations: unexpected answer")
        return rows, nxt, revoked


def remote_row(row, now: float) -> dict | None:
    """A row from the store reduced to what the policy needs, or None if it is not usable.

    Remote data is trusted less than local data, not more: every field is
    checked, and the time must lie within the window we keep.
    """
    if not isinstance(row, dict):
        return None
    kind, reason, ts = row.get("kind"), row.get("reason_code"), row.get("ts")
    ok = (
        type(row.get("seq")) is int
        and row["seq"] > 0
        and isinstance(row.get("id"), str)
        and 0 < len(row["id"]) <= 64
        and isinstance(row.get("src"), str)
        and 0 < len(row["src"]) <= 64
        and row.get("provider") == "vast"
        and type(row.get("machine_id")) is int
        and row["machine_id"] > 0
        and type(ts) in (int, float)
        and now - WINDOW - 86400 < ts < now + 3600
        and (
            (kind == "healthy" and reason is None)
            or (kind in ("failed_admission", "died_after_admission") and reason in REASONS)
        )
    )
    return {k: row[k] for k in KEPT_FIELDS} if ok else None


def from_env(environ=os.environ) -> Store | None:
    """The store named by GPUBOX_STORE_URL and GPUBOX_STORE_TOKEN, if both are set."""
    url, token = environ.get("GPUBOX_STORE_URL"), environ.get("GPUBOX_STORE_TOKEN")
    if not url or not token:
        return None
    return Store(url, token)


def _read(path: Path) -> list[dict]:
    """The JSON objects in a file, one per line. Anything unreadable is skipped."""
    rows = []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return rows
    for raw in text.splitlines():
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _lines(rows: list[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)


class Shared:
    """The outbox and the remote rows kept beside one observations file."""

    def __init__(self, observations_path: Path, store: Store | None, budget: float | None = None):
        base = Path(observations_path)
        self.store = store
        self.outbox = base.with_name(base.name + ".outbox")
        self.remote = base.with_name(base.name + ".remote")
        self.cursor = base.with_name(base.name + ".cursor")
        self._lockfile = base.with_name(base.name + ".lock")
        # One budget for everything this object does: `up` talks to the store
        # before and after the launch, and both come out of the same seconds
        # (only time spent talking counts, not the launch in between).
        self._left = BUDGET if budget is None else budget

    @contextmanager
    def _locked(self):
        """Serialize changes to the three files between processes sharing them."""
        self._lockfile.parent.mkdir(parents=True, exist_ok=True)
        with open(self._lockfile, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    def queue(self, rows: list[dict]) -> None:
        """Remember rows to send. Bounded: beyond OUTBOX_LIMIT the oldest are dropped."""
        if not self.store or not rows:
            return
        with self._locked():
            _write(self.outbox, _lines([*_read(self.outbox), *rows][-OUTBOX_LIMIT:]))

    def remote_rows(self) -> list[dict]:
        """What others observed, as far as we have read it. Empty without a store."""
        if not self.store:
            return []
        now = time.time()
        return [r for r in map(lambda row: remote_row(row, now), _read(self.remote)) if r]

    def sync(self, *, pull: bool = True) -> str:
        """Send the outbox and read what is new, within what is left of the budget.

        Returns a short account of what happened. Raises StoreError only if
        nothing useful happened; partial progress is kept and reported.
        The network runs in a thread we stop waiting for at the deadline, so
        a server that trickles bytes, or a resolver that hangs, cannot hold
        the command; the thread only fetches, and files change here.
        """
        if not self.store:
            return "no store configured"
        left, started = self._left, time.monotonic()
        if left <= 0.1:
            raise StoreError("no time left for the store in this command")
        with self._locked():
            waiting = _read(self.outbox)
            after = self._read_cursor()
        result = {"acked": [], "dropped": [], "pages": [], "errors": []}
        worker = threading.Thread(
            target=self._exchange, args=(waiting, after, pull, time.monotonic() + left, result),
            daemon=True,
        )  # fmt: skip
        worker.start()
        worker.join(left)
        self._left -= time.monotonic() - started
        if worker.is_alive():
            result = {k: list(v) for k, v in result.items()}  # what it managed so far
            result["errors"].append("out of time")
        return self._apply(result)

    def _exchange(self, waiting, after, pull, deadline, result) -> None:
        """Network only. Appends to `result`; never touches the files."""
        try:
            for i in range(0, len(waiting), BATCH):
                batch = waiting[i : i + BATCH]
                try:
                    self.store.push(batch, deadline)
                    result["acked"].extend(r.get("id") for r in batch)
                except StoreError as exc:
                    if "HTTP 400" in str(exc):
                        # The store will never take this batch; keeping it would
                        # block everything queued behind it, for ever.
                        result["dropped"].extend(r.get("id") for r in batch)
                        result["errors"].append(f"{exc} (batch dropped)")
                        continue
                    result["errors"].append(str(exc))
                    break
        except Exception as exc:
            result["errors"].append(type(exc).__name__)
        try:
            while pull:
                rows, nxt, revoked = self.store.page(after, deadline)
                seqs = [r.get("seq") for r in rows if isinstance(r, dict)]
                top = max([s for s in seqs if type(s) is int], default=after)
                result["pages"].append((rows, max(top, after), revoked))
                if nxt is None:
                    break
                after = nxt
        except StoreError as exc:
            result["errors"].append(str(exc))
        except Exception as exc:
            result["errors"].append(type(exc).__name__)

    def _read_cursor(self) -> int:
        try:
            value = json.loads(self.cursor.read_text())
            return value if type(value) is int and value >= 0 else 0
        except (OSError, ValueError):
            return 0

    def _apply(self, result) -> str:
        now = time.time()
        gone = set(result["acked"]) | set(result["dropped"])
        with self._locked():
            if gone:
                # Remove what was dealt with from the queue as it is now: rows
                # queued by another process in the meantime stay.
                keep = [r for r in _read(self.outbox) if r.get("id") not in gone]
                _write(self.outbox, _lines(keep))
            read = 0
            if result["pages"]:
                rows = [r for r in map(lambda row: remote_row(row, now), _read(self.remote)) if r]
                have = {r["seq"] for r in rows}
                cursor, revoked = self._read_cursor(), set()
                for page, top, page_revoked in result["pages"]:
                    for raw in page:
                        row = remote_row(raw, now)
                        if row and row["seq"] not in have:
                            have.add(row["seq"])
                            rows.append(row)
                            read += 1
                    cursor, revoked = max(cursor, top), set(page_revoked)
                rows = [r for r in rows if r["src"] not in revoked and r["ts"] > now - WINDOW]
                _write(self.remote, _lines(rows[-MAX_REMOTE_ROWS:]))
                _write(self.cursor, json.dumps(cursor))
        account = f"sent {len(result['acked'])}, read {read}"
        if result["errors"]:
            if not gone and not result["pages"]:
                raise StoreError("; ".join(result["errors"]))
            account += f" ({'; '.join(result['errors'])})"
        return account
