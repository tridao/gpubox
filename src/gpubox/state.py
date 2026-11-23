"""What a run asked for and which boxes it knows: the tag, a snapshot, a lock.

The tag on a box says whose it is (DESIGN.md, section 4). It is set by the
create request, so a box found by its tag can be destroyed by a process that
knows nothing else.

The snapshot (`run.json`) is one JSON document, replaced whole at each change.
It is not a log and is never replayed: it is read whole or not at all. It holds
the run's attempts (so that a create with no answer is not forgotten) and every
instance id the run knows (so that a box that changed its label is still
destroyed by id). Rules S1 to S4 of the design are about this file.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import re
import secrets
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

NAMESPACE = re.compile(r"[a-z0-9][a-z0-9-]*")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]*\.[A-Za-z0-9][A-Za-z0-9_-]*\.[0-9a-f]{12}")
_TAG = re.compile(
    r"(?P<run>(?P<ns>[a-z0-9][a-z0-9-]*)\.[A-Za-z0-9][A-Za-z0-9_-]*\.[0-9a-f]{12})"
    r"\.a(?P<n>[1-9][0-9]{0,5})\.(?P<expiry>[0-9]{10})"
)

SNAPSHOT = "run.json"
ATTEMPT_STATES = ("open", "refused", "bound", "unseen")


class StateError(RuntimeError):
    """The state directory cannot be used as asked (locked, live run, no run, damaged)."""


def positive_int(value) -> bool:
    return type(value) is int and value > 0


def new_run_id(namespace: str, name: str) -> str:
    if not NAMESPACE.fullmatch(namespace):
        raise StateError(f"namespace must match {NAMESPACE.pattern}: {namespace!r}")
    if not NAME.fullmatch(name):
        raise StateError(f"name must match {NAME.pattern}: {name!r}")
    return f"{namespace}.{name}.{secrets.token_hex(6)}"


@dataclass(frozen=True)
class Tag:
    run: str
    namespace: str
    attempt: int
    expiry: int


def make_tag(run_id: str, attempt: int, expiry: int) -> str:
    tag = f"{run_id}.a{attempt}.{expiry:010d}"
    if parse_tag(tag) is None:
        raise StateError(f"cannot make a tag from {run_id!r}, {attempt!r}, {expiry!r}")
    return tag


def parse_tag(label) -> Tag | None:
    """The tag a label is, or None. Exact: a label that merely resembles one is not a tag."""
    if not isinstance(label, str):
        return None
    m = _TAG.fullmatch(label)
    if m is None:
        return None
    return Tag(m["run"], m["ns"], int(m["n"]), int(m["expiry"]))


@contextmanager
def state_lock(state_dir: Path):
    """One `up` or `down` per state directory. Released by the kernel if we die."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(state_dir / "lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise StateError(
                f"{state_dir} is in use by another gpubox process; "
                "interrupt it (it rolls back what it rented) and retry"
            ) from None
        yield
    finally:
        os.close(fd)


def _check(data) -> None:
    """Raise ValueError unless `data` is a snapshot we wrote."""
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("not a version 1 snapshot")
    if not isinstance(data.get("run"), str) or not RUN_ID.fullmatch(data["run"]):
        raise ValueError("bad run id")
    if type(data.get("expiry")) is not int or not 0 < data["expiry"] < 10**10:
        raise ValueError("bad expiry")
    if data.get("account") is not None and not isinstance(data["account"], str):
        raise ValueError("bad account")
    if not isinstance(data.get("meta"), dict):
        raise ValueError("bad meta")
    for flag in ("closing", "closed"):
        if type(data.get(flag)) is not bool:
            raise ValueError(f"bad {flag}")
    ids = data.get("ids")
    if not isinstance(ids, dict):
        raise ValueError("bad ids")
    for key, status in ids.items():
        if not key.isascii() or not key.isdigit() or int(key) <= 0 or str(int(key)) != key:
            raise ValueError(f"bad instance id {key!r}")
        if status not in ("live", "gone"):
            raise ValueError(f"bad status for instance {key}")
    attempts = data.get("attempts")
    if not isinstance(attempts, dict):
        raise ValueError("bad attempts")
    for key, a in attempts.items():
        if not key.isascii() or not key.isdigit() or int(key) <= 0 or str(int(key)) != key:
            raise ValueError(f"bad attempt number {key!r}")
        if not isinstance(a, dict) or a.get("state") not in ATTEMPT_STATES:
            raise ValueError(f"bad attempt {key}")
        if type(a.get("sent")) not in (int, float):
            raise ValueError(f"attempt {key} has no time")
        if a["state"] == "bound":
            if not positive_int(a.get("id")) or str(a["id"]) not in ids:
                raise ValueError(f"attempt {key} is bound to an id the snapshot does not know")
    w = data.get("winner")
    if w is not None:
        ok = (
            isinstance(w, dict)
            and positive_int(w.get("id"))
            and str(w["id"]) in ids
            and positive_int(w.get("attempt"))
            and isinstance(w.get("host"), str)
            and positive_int(w.get("port"))
        )
        if not ok:
            raise ValueError("bad winner")


class Snapshot:
    """The run's snapshot. Every change is on disk before the method returns, or it raises
    OSError and nothing has changed, in memory or on disk."""

    def __init__(self, path: Path | None, data: dict):
        self.path, self.data = path, data

    # ------------------------------------------------------------ reading

    @classmethod
    def load(cls, state_dir) -> Snapshot | None:
        """The directory's snapshot, None if there is none, StateError if it cannot be used."""
        path = Path(state_dir) / SNAPSHOT
        try:
            text = path.read_text()
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError) as exc:
            raise StateError(f"{path} cannot be read: {exc}") from None
        try:
            data = json.loads(text)
            _check(data)
        except (ValueError, TypeError, AttributeError, RecursionError) as exc:
            raise StateError(f"{path} is damaged ({exc})") from None
        return cls(path, data)

    @classmethod
    def begin(cls, state_dir, run_id: str, expiry: int, account: str | None, meta: dict):
        data = {
            "version": 1,
            "account": account,
            "run": run_id,
            "expiry": int(expiry),
            "meta": dict(meta),
            "attempts": {},
            "ids": {},
            "winner": None,
            "closing": False,
            "closed": False,
        }
        _check(data)
        snap = cls(Path(state_dir) / SNAPSHOT, {})
        snap._write(data)
        snap.data = data
        return snap

    @classmethod
    def ephemeral(cls, run_id: str):
        """A snapshot that is never written: for tearing down a run by its tag alone."""
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise StateError(f"not a run id: {run_id!r}")
        return cls(None, {
            "version": 1, "account": None, "run": run_id, "expiry": 1, "meta": {},
            "attempts": {}, "ids": {}, "winner": None, "closing": True, "closed": False,
        })  # fmt: skip

    run = property(lambda self: self.data["run"])
    expiry = property(lambda self: self.data["expiry"])
    account = property(lambda self: self.data["account"])
    meta = property(lambda self: self.data["meta"])
    closing = property(lambda self: self.data["closing"])
    closed = property(lambda self: self.data["closed"])

    @property
    def winner(self) -> dict | None:
        return copy.deepcopy(self.data["winner"])

    def attempts(self) -> dict[int, dict]:
        return {int(n): dict(a) for n, a in self.data["attempts"].items()}

    def open_attempts(self) -> dict[int, dict]:
        return {n: a for n, a in self.attempts().items() if a["state"] == "open"}

    def status(self, instance_id: int) -> str | None:
        return self.data["ids"].get(str(instance_id))

    def ids(self) -> dict[int, str]:
        return {int(i): status for i, status in self.data["ids"].items()}

    def live_ids(self) -> list[int]:
        return sorted(i for i, status in self.ids().items() if status == "live")

    def attempt_of(self, instance_id: int) -> tuple[int, dict] | None:
        for n, a in self.attempts().items():
            if a.get("id") == instance_id:
                return n, a
        return None

    def slots(self) -> int:
        """Attempts that hold a slot: a box may exist for them (rule L2)."""
        ids = self.data["ids"]
        return sum(
            a["state"] == "open" or (a["state"] == "bound" and ids.get(str(a["id"])) != "gone")
            for a in self.data["attempts"].values()
        )

    def is_live(self) -> bool:
        return bool(self.live_ids() or self.open_attempts())

    # ------------------------------------------------------------ changing

    def _write(self, data: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=SNAPSHOT + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(data, sort_keys=True, indent=1) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        try:  # make the rename itself durable; not every filesystem lets us
            dfd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass

    def _change(self, fn) -> None:
        new = copy.deepcopy(self.data)
        fn(new)
        _check(new)
        self._write(new)
        self.data = new

    def open_attempt(self, n: int, *, sent: float, **fields) -> None:
        """Rule S1: called, and on disk, before the create request is sent."""

        def fn(d):
            if str(n) in d["attempts"]:
                raise StateError(f"attempt {n} already exists")
            d["attempts"][str(n)] = {**fields, "sent": sent, "state": "open", "id": None}

        self._change(fn)

    def refuse(self, n: int) -> None:
        self._set_attempt(n, "refused")

    def unseen(self, n: int) -> None:
        """An open attempt whose box nobody saw: left to the timer and the reaper."""
        self._set_attempt(n, "unseen")

    def _set_attempt(self, n: int, state: str) -> None:
        def fn(d):
            a = d["attempts"][str(n)]
            if a["state"] != "open":
                raise StateError(f"attempt {n} is {a['state']}, not open")
            a["state"] = state

        self._change(fn)

    def bind(self, n: int, instance_id: int) -> None:
        """Rule S2: the id is on disk before the box is used."""

        def fn(d):
            a = d["attempts"][str(n)]
            if a["state"] != "open":
                raise StateError(f"attempt {n} is {a['state']}, not open")
            a["state"], a["id"] = "bound", instance_id
            d["ids"].setdefault(str(instance_id), "live")

        self._change(fn)

    def know(self, instance_id: int) -> None:
        """Rule S2: an id is recorded before any delete is sent for it."""
        if self.status(instance_id) is None:
            self._change(lambda d: d["ids"].setdefault(str(instance_id), "live"))

    def gone(self, instance_id: int) -> None:
        """Rule S3: only ever called after a lookup of this id said it does not exist."""
        if self.status(instance_id) != "gone":
            self._change(lambda d: d["ids"].__setitem__(str(instance_id), "gone"))

    def publish(self, instance_id: int, attempt: int, host: str, port: int) -> None:
        """The hand-off (rule L4): once this is on disk the winner is kept."""

        def fn(d):
            d["winner"] = {"id": instance_id, "attempt": attempt, "host": host, "port": port}

        self._change(fn)

    def set_closing(self) -> None:
        if not self.closing:
            self._change(lambda d: d.__setitem__("closing", True))

    def close(self) -> None:
        def fn(d):
            d["closing"] = d["closed"] = True

        self._change(fn)
