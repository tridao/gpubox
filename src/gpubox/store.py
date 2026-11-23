"""The shared reputation store: what other clients observed about machines.

Advisory in every respect. Everything one command says to the store takes at
most `BUDGET` seconds of waiting and can never fail the command; what the
store returns can only make an offer be skipped (see policy.blocklist);
nothing here touches ownership or cleanup.

Beside an observations file `NAME` live:

    NAME.outbox   rows observed here and not yet accepted by the store
    NAME.remote   the last complete read from the store, replaced whole
    NAME.lock     serializes changes to the two files above

A read is one request for the store's last 30 days (at most a few thousand
rows; in practice tens). If it fails, the previous read is used.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from .policy import POLICY_VERSION, REASONS

BUDGET = 5.0  # seconds one command may spend waiting for the store, in total
LOCK_WAIT = 1.0  # how long to wait for another gpubox process to release the side files
BATCH = 50
OUTBOX_LIMIT = 1000  # beyond this the oldest unsent rows are dropped
WINDOW = 30 * 86400  # rows older than this are forgotten
SENT_FIELDS = (
    "id", "provider", "machine_id", "host_id", "kind", "reason_code",
    "gpu", "num_gpus", "gate_version", "policy_version", "secs_to_ready",
)  # fmt: skip
KEPT_FIELDS = (
    "seq", "id", "ts", "src", "provider", "machine_id", "kind", "reason_code", "policy_version",
)  # fmt: skip
SIDE_SUFFIXES = (".outbox", ".remote", ".lock")
_HOSTNAME = re.compile(r"(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}")


class StoreError(RuntimeError):
    """Something about the store went wrong. The message never contains the URL or the token."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """The token goes to the configured origin and nowhere else."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


def _origin(url: str, allow_loopback_http: bool) -> str:
    """The origin a store URL names, rebuilt from validated parts, or StoreError.

    Only what we can rebuild exactly is accepted: https, a plain DNS name, an
    optional port. No IP literals (their text forms are parsed differently by
    different layers), no credentials, path, query, fragment or escapes.
    """
    if not isinstance(url, str) or not url.isascii() or any(c in url for c in "%@\\ \t\r\n"):
        raise StoreError("the store URL must be https://HOST[:PORT] with a plain host name")
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError:
        raise StoreError("the store URL cannot be parsed") from None
    host = (parts.hostname or "").lower()
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise StoreError("the store URL must not carry a path, a query or a fragment")
    if port is not None and not 1 <= port <= 65535:
        raise StoreError("the store URL has an unusable port")
    suffix = f":{port}" if port is not None else ""
    if parts.netloc.lower() != host + suffix:
        raise StoreError("the store URL must be https://HOST[:PORT] with a plain host name")
    if parts.scheme == "https" and _HOSTNAME.fullmatch(host):
        return f"https://{host}{suffix}"
    if allow_loopback_http and parts.scheme == "http" and host == "127.0.0.1":
        return f"http://127.0.0.1{suffix}"
    raise StoreError("the store URL must be https://HOST[:PORT] with a plain host name")


class Store:
    def __init__(self, url: str, token: str, *, allow_loopback_http: bool = False):
        """`allow_loopback_http` exists for tests against a server on this machine."""
        self._origin = _origin(url, allow_loopback_http)
        if not isinstance(token, str) or len(token) < 32 or not token.isprintable():
            raise StoreError("the store token is not usable")
        self._token = token
        try:
            # No proxies from the environment, no redirects: the request goes
            # to the origin above or nowhere.
            self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
        except Exception as exc:
            raise StoreError(f"cannot set up HTTP: {type(exc).__name__}") from None

    def identity(self) -> str:
        """A name for this origin that is safe to write to disk."""
        return hashlib.sha256(self._origin.encode()).hexdigest()[:16]

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
        """Send one batch. Returns only if the store has dealt with every row of it
        (stored it, or decided for good not to); raises StoreError otherwise."""
        rows = [{k: r.get(k) for k in SENT_FIELDS} for r in batch]
        answer = self._call("POST", "/v1/observations", deadline, {"observations": rows})
        accepted, ignored = answer.get("accepted"), answer.get("ignored")
        if type(accepted) is not int or type(ignored) is not int:
            raise StoreError("POST /v1/observations: unexpected answer")
        if accepted < 0 or ignored < 0 or accepted + ignored != len(rows):
            raise StoreError("POST /v1/observations: the counts do not add up")

    def read(self, deadline: float, provider: str = "vast") -> list:
        """Everything the store serves for `provider`: its last 30 days, as sent.

        Rows are checked one by one later (remote_row); here only the shape of
        the answer.
        """
        answer = self._call("GET", f"/v1/observations?provider={provider}", deadline)
        rows = answer.get("observations")
        if not isinstance(rows, list) or not isinstance(answer.get("truncated"), bool):
            raise StoreError("GET /v1/observations: unexpected answer")
        return rows


def remote_row(row, now: float) -> dict | None:
    """A row from the store reduced to what the policy needs, or None if it is not usable.

    Remote data is trusted less than local data, not more: every field is
    checked, and the time must lie within the window we keep. Never raises.
    """
    try:
        kind, reason, ts = row["kind"], row["reason_code"], row["ts"]
        ok = (
            type(row["seq"]) is int
            and row["seq"] > 0
            and isinstance(row["id"], str)
            and 0 < len(row["id"]) <= 64
            and isinstance(row["src"], str)
            and 0 < len(row["src"]) <= 64
            and row["provider"] == "vast"
            and type(row["machine_id"]) is int
            and row["machine_id"] > 0
            and type(row["policy_version"]) is int
            and 0 <= row["policy_version"] <= POLICY_VERSION
            and type(ts) in (int, float)
            and now - WINDOW < ts < now + 3600
            and (
                (kind == "healthy" and reason is None)
                or (
                    kind in ("failed_admission", "died_after_admission")
                    and isinstance(reason, str)
                    and reason in REASONS
                )
            )
        )
        return {k: row[k] for k in KEPT_FIELDS} if ok else None
    except (KeyError, TypeError, IndexError, OverflowError):
        return None


def expired(row, now: float) -> bool:
    """A failure whose block is over already. The store times a row from when it
    receives it, so sending this late would block the machine again, for everyone."""
    reason, ts = row.get("reason_code"), row.get("ts")
    return (
        row.get("kind") != "healthy"
        and reason in REASONS
        and type(ts) in (int, float)
        and ts + REASONS[reason][1] <= now
    )


def from_env(environ=os.environ) -> Store | None:
    """The store named by GPUBOX_STORE_URL and GPUBOX_STORE_TOKEN, if both are set."""
    url, token = environ.get("GPUBOX_STORE_URL"), environ.get("GPUBOX_STORE_TOKEN")
    if not url or not token:
        return None
    return Store(url, token)


def _read_lines(path: Path, strict: bool = False) -> list[dict]:
    """The JSON objects in a file, one per line; unreadable lines are skipped.

    A missing file is empty. A file that exists and cannot be read is empty
    too, unless `strict`: someone about to rewrite the file must not mistake
    "I could not read it" for "there was nothing in it".
    """
    try:
        text = path.read_text(errors="replace")
    except FileNotFoundError:
        return []
    except OSError as exc:
        if strict:
            raise StoreError(f"cannot read {path.name}: {type(exc).__name__}") from None
        return []
    rows = []
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
        if store and base.name.endswith(SIDE_SUFFIXES):
            # Its side files would be another observations file's, or its own.
            raise StoreError(f"an observations file cannot be named *{base.suffix}")
        self.store = store
        self.outbox = base.with_name(base.name + ".outbox")
        self.remote = base.with_name(base.name + ".remote")
        self._lockfile = base.with_name(base.name + ".lock")
        # One budget for everything this object does: `up` talks to the store
        # before and after the launch, and both come out of the same seconds
        # (only time spent in sync counts, not the launch in between).
        self._left = BUDGET if budget is None else budget

    @contextmanager
    def _locked(self, wait: float = LOCK_WAIT):
        """Serialize changes to the side files between processes. Gives up after `wait`."""
        self._lockfile.parent.mkdir(parents=True, exist_ok=True)
        with open(self._lockfile, "a") as f:
            deadline = time.monotonic() + max(0.0, wait)
            while True:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise StoreError("the side files are in use by another process") from None
                    time.sleep(0.02)
            yield

    def queue(self, rows: list[dict]) -> None:
        """Remember rows to send. Bounded: beyond OUTBOX_LIMIT the oldest are dropped."""
        if not self.store or not rows:
            return
        with self._locked():
            waiting = _read_lines(self.outbox, strict=True)
            _write(self.outbox, _lines([*waiting, *rows][-OUTBOX_LIMIT:]))

    def remote_rows(self) -> list[dict]:
        """What others observed, as of the last read. Empty without a store. Never raises."""
        if not self.store:
            return []
        try:
            doc = json.loads(self.remote.read_text(errors="replace"))
            if doc["origin"] != self.store.identity() or not isinstance(doc["rows"], list):
                return []  # read from another store: none of it applies here
            now = time.time()
            return [r for r in (remote_row(x, now) for x in doc["rows"]) if r]
        except (OSError, ValueError, KeyError, TypeError):
            return []

    def sync(self, *, pull: bool = True) -> str:
        """Send the outbox and, with `pull`, read the store, within what is left of the budget.

        Returns a short account of what happened. Raises StoreError only if
        nothing useful happened; partial progress is kept and reported.

        The network runs in a thread we stop waiting for at the deadline, so a
        server that trickles bytes, or a resolver that hangs, cannot hold the
        command; that thread only fetches, and files are changed here. Waiting
        for the side-file lock counts against the same deadline.
        """
        if not self.store:
            return "no store configured"
        started = time.monotonic()
        deadline = started + self._left
        try:
            if self._left <= 0.1:
                raise StoreError("no time left for the store in this command")
            with self._locked(min(LOCK_WAIT, self._left)):
                waiting = _read_lines(self.outbox)
            now = time.time()
            late = [r.get("id") for r in waiting if expired(r, now)]
            waiting = [r for r in waiting if not expired(r, now)]
            result = {"acked": [], "dropped": late, "read": [], "errors": []}
            worker = threading.Thread(
                target=self._exchange, args=(waiting, pull, deadline, result), daemon=True
            )
            worker.start()
            worker.join(max(0.0, deadline - time.monotonic()))
            if worker.is_alive():
                result = {k: list(v) for k, v in result.items()}  # what it managed so far
                result["errors"].append("out of time")
            return self._apply(result)
        finally:
            self._left -= time.monotonic() - started

    def _exchange(self, waiting, pull, deadline, result) -> None:
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
        if not pull:
            return
        try:
            result["read"].append(self.store.read(deadline))
        except StoreError as exc:
            result["errors"].append(str(exc))
        except Exception as exc:
            result["errors"].append(type(exc).__name__)

    def _apply(self, result) -> str:
        now = time.time()
        gone = {i for i in [*result["acked"], *result["dropped"]] if isinstance(i, str)}
        read = None
        if not gone and not result["read"]:  # no file to change: no lock to wait for
            if result["errors"]:
                raise StoreError("; ".join(result["errors"]))
            return "sent 0"
        with self._locked():
            if gone:
                # Remove what was dealt with from the queue as it is now: rows
                # queued by another process in the meantime stay.
                waiting = _read_lines(self.outbox, strict=True)
                _write(self.outbox, _lines([r for r in waiting if r.get("id") not in gone]))
            if result["read"]:
                rows = [r for r in (remote_row(x, now) for x in result["read"][0]) if r]
                read = len(rows)
                _write(self.remote, json.dumps({"origin": self.store.identity(), "rows": rows}))
        account = f"sent {len(result['acked'])}" + (f", read {read}" if read is not None else "")
        if result["errors"]:
            if not gone and read is None:
                raise StoreError("; ".join(result["errors"]))
            account += f" ({'; '.join(result['errors'])})"
        return account
