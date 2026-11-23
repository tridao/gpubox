"""What a run is called and what it left on disk: the tag, a small state file, a lock.

The tag on a box says whose it is and when its lease ends (DESIGN.md). It is
set by the create request, so any process that knows the run id can find the
run's boxes in the provider's listing and destroy them.

The state file (`run.json`) is a convenience for this directory: the run id,
the instance ids that creates returned, and where the winners are. Losing it
loses the use of the box (its address and key), never the ability to destroy
it: `down --run-id` works from the tag alone.
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

from .errors import GpuboxError

NAMESPACE = re.compile(r"[a-z0-9][a-z0-9-]*")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]*\.[A-Za-z0-9][A-Za-z0-9_-]*\.[0-9a-f]{12}")
_TAG = re.compile(
    r"(?P<run>(?P<ns>[a-z0-9][a-z0-9-]*)\.(?P<name>[A-Za-z0-9][A-Za-z0-9_-]*)\.[0-9a-f]{12})"
    r"\.a(?P<n>[1-9][0-9]{0,5})\.(?P<expiry>[0-9]{10})"
)

STATE = "run.json"


class StateError(GpuboxError):
    """The state directory cannot be used as asked (locked, live run, no run, unreadable)."""


def positive_int(value) -> bool:
    return type(value) is int and value > 0


def new_run_id(namespace: str, name: str) -> str:
    if not NAMESPACE.fullmatch(namespace):
        raise StateError(f"namespace must match {NAMESPACE.pattern}: {namespace!r}")
    if not NAME.fullmatch(name):
        raise StateError(f"name must match {NAME.pattern}: {name!r}")
    return f"{namespace}.{name}.{secrets.token_hex(6)}"


def name_of(run_id: str) -> str:
    """The caller's name inside a run id."""
    return run_id.split(".")[1]


def box_name(run_id: str, index: int, count: int) -> str:
    """What box `index` of a run with `count` boxes is called: the run's name for a run
    of one box, else the name and the index (`mybox-0`, `mybox-1`, ...)."""
    name = name_of(run_id)
    return name if count == 1 else f"{name}-{index}"


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
    """One `up`, `down` or `renew` per state directory. Released by the kernel if we die."""
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


def _box(w) -> bool:
    return (
        isinstance(w, dict)
        and positive_int(w.get("id"))
        and positive_int(w.get("attempt"))
        and isinstance(w.get("host"), str)
        and positive_int(w.get("port"))
    )


def _upgrade(data):
    """A version 2 file (one `winner`, or none) read as version 3 (a list of `winners`).
    It is written back as version 3 at its next change."""
    if isinstance(data, dict) and data.get("version") == 2 and "winners" not in data:
        data = dict(data)
        w = data.pop("winner", None)
        data["winners"] = [] if w is None else [w]
        data["version"] = 3
    return data


def _usable(data) -> bool:
    """Enough of a state file to act on. Not a consistency check: just the shapes we index."""
    winners = data.get("winners") if isinstance(data, dict) else None
    return (
        isinstance(data, dict)
        and data.get("version") == 3
        and isinstance(data.get("run"), str)
        and RUN_ID.fullmatch(data["run"]) is not None
        and type(data.get("expiry")) is int
        and isinstance(data.get("ids"), list)
        and all(positive_int(i) for i in data["ids"])
        and isinstance(data.get("meta"), dict)
        and (data.get("account") is None or isinstance(data["account"], str))
        and isinstance(winners, list)
        and all(_box(w) for w in winners)
    )


class State:
    """The run's state file. Each change is written (temp file, fsync, rename) before the
    method returns; if the write raises OSError, memory is unchanged too."""

    def __init__(self, path: Path | None, data: dict):
        self.path, self.data = path, data

    @classmethod
    def load(cls, state_dir) -> State | None:
        """The directory's state, None if there is none, StateError if it cannot be used."""
        path = Path(state_dir) / STATE
        try:
            data = _upgrade(json.loads(path.read_text()))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise StateError(f"{path} cannot be read ({exc})") from None
        if not _usable(data):
            raise StateError(f"{path} is not a gpubox state file of this version")
        return cls(path, data)

    @classmethod
    def begin(cls, state_dir, run_id: str, expiry: int, account: str | None, meta: dict):
        state = cls(Path(state_dir) / STATE, {})
        state._replace({
            "version": 3, "run": run_id, "expiry": int(expiry), "account": account,
            "meta": dict(meta), "ids": [], "winners": [], "closing": False, "closed": False,
        })  # fmt: skip
        return state

    @classmethod
    def ephemeral(cls, run_id: str):
        """A state that is never written: for tearing down a run by its tag alone."""
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise StateError(f"not a run id: {run_id!r}")
        return cls(None, {
            "version": 3, "run": run_id, "expiry": 0, "account": None, "meta": {}, "ids": [],
            "winners": [], "closing": True, "closed": False,
        })  # fmt: skip

    run = property(lambda self: self.data["run"])
    expiry = property(lambda self: self.data["expiry"])
    account = property(lambda self: self.data.get("account"))
    meta = property(lambda self: self.data["meta"])
    ids = property(lambda self: list(self.data["ids"]))
    closing = property(lambda self: bool(self.data.get("closing")))
    closed = property(lambda self: bool(self.data.get("closed")))

    @property
    def winners(self) -> list[dict]:
        """The boxes handed over, in order: box i is `box_name(run, i, len(winners))`."""
        return copy.deepcopy(self.data["winners"])

    @property
    def winner(self) -> dict | None:
        """The first box, for a run of one."""
        return copy.deepcopy(self.data["winners"][0]) if self.data["winners"] else None

    def _replace(self, data: dict) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=STATE + ".", suffix=".tmp")
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
        self.data = data

    def _set(self, **fields) -> None:
        self._replace({**copy.deepcopy(self.data), **fields})

    def add_id(self, instance_id: int) -> None:
        """An id a create returned, kept so that `down` destroys it without needing its tag."""
        if instance_id not in self.data["ids"]:
            self._set(ids=[*self.data["ids"], instance_id])

    def publish(self, winners: list[dict]) -> None:
        """The hand-off: once this is on disk the winners are kept."""
        self._set(winners=[dict(w) for w in winners])

    def set_expiry(self, expiry: int) -> None:
        self._set(expiry=int(expiry))

    def set_closing(self) -> None:
        if not self.closing:
            self._set(closing=True)

    def close(self) -> None:
        self._set(closing=True, closed=True)
