"""The record of what a run rented. It is what authorizes `down` to destroy.

A label on an instance is mutable (by us, by a teammate, by code on the box),
so it is only ever a search hint. Ownership is this append-only file, which
holds two kinds of fact:

* an **intent**: we were about to ask for a box under this label. Written and
  synced before the request is sent. Until it is resolved (refused, bound to
  an instance id, or abandoned by the operator) a box may exist for it.
* an **instance id** we were given. Every id that appears anywhere in this
  file is ours, whether or not the record that would tie it to an intent
  survived. An id stays live until a `destroyed` record confirms it is gone.

Replay never forgets an id and never moves a box backwards.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

NAMESPACE = re.compile(r"[a-z0-9][a-z0-9-]*")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]*\.[A-Za-z0-9][A-Za-z0-9_-]*\.[0-9a-f]{8}")

# Attempt statuses under which a box may exist and cost money.
LIVE = ("intent", "unknown", "created", "destroying")
OPEN = ("intent", "unknown")  # no instance id known


class StateError(RuntimeError):
    """The state directory cannot be used as asked (locked, live run, no run)."""


def new_run_id(namespace: str, name: str) -> str:
    if not NAMESPACE.fullmatch(namespace):
        raise StateError(f"namespace must match {NAMESPACE.pattern}: {namespace!r}")
    if not NAME.fullmatch(name):
        raise StateError(f"name must match {NAME.pattern}: {name!r}")
    return f"{namespace}.{name}.{secrets.token_hex(4)}"


def attempt_label(run_id: str, seq: int) -> str:
    return f"{run_id}.a{seq}"


def label_namespace(label: str | None) -> str | None:
    """The namespace a label belongs to, or None if it is not one of ours.

    Exact parsing: `quack-ci.x.1234abcd.a1` is in `quack-ci`; `quack-ci-old...`
    and `my-quack-ci...` are not.
    """
    if not isinstance(label, str):
        return None
    parts = label.split(".")
    if len(parts) != 4 or not NAMESPACE.fullmatch(parts[0]) or not NAME.fullmatch(parts[1]):
        return None
    if not re.fullmatch(r"[0-9a-f]{8}", parts[2]) or not re.fullmatch(r"a\d+", parts[3]):
        return None
    return parts[0]


def positive_int(value) -> bool:
    return type(value) is int and value > 0


def finite_number(value) -> bool:
    """A real number we can compute with. `10**400` is an int, but not one of these."""
    return type(value) in (int, float) and -1e15 < value < 1e15


@dataclass
class AttemptState:
    seq: int
    label: str
    ts: float | None = None  # when the intent was written; None if unusable
    # intent|refused|unknown|abandoned, or the status of its box
    status: str = "intent"
    instance_id: int | None = None
    offer_id: int | None = None
    machine_id: int | None = None
    host_id: int | None = None
    dph: float = 0.0


@dataclass
class RunState:
    run_id: str | None = None
    namespace: str | None = None
    meta: dict = field(default_factory=dict)  # the `run` event: image, gpu, ...
    attempts: dict[int, AttemptState] = field(default_factory=dict)
    boxes: dict[int, str] = field(default_factory=dict)  # every id we own -> status
    winner: dict | None = None
    closed: bool = False
    # Damage that is not an ordinary torn last line. A damaged journal can
    # still be torn down, but is never reported closed and never rotated.
    problems: list[str] = field(default_factory=list)

    def live_ids(self) -> list[int]:
        return sorted(i for i, status in self.boxes.items() if status != "destroyed")

    def open_attempts(self) -> list[AttemptState]:
        return [a for a in self.attempts.values() if a.status in OPEN]

    def is_live(self) -> bool:
        return bool(self.live_ids() or self.open_attempts())

    def attempt_of(self, instance_id: int) -> AttemptState | None:
        return next((a for a in self.attempts.values() if a.instance_id == instance_id), None)


@contextmanager
def state_lock(state_dir: Path):
    """One writer per state directory. Released by the kernel if we die."""
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


class Journal:
    def __init__(self, state_dir: Path, run_id: str | None = None):
        """The current run's journal, or (with `run_id`) that run's, wherever it now is."""
        self.path = Path(state_dir) / "journal.jsonl"
        self._lock = threading.Lock()
        if run_id is not None and self.state().run_id != run_id:
            if not RUN_ID.fullmatch(run_id):
                raise StateError(f"not a run id: {run_id!r}")
            self.path = Path(state_dir) / f"journal.{run_id}.jsonl"
            if not self.path.exists() or self.state().run_id != run_id:
                raise StateError(f"no journal for run {run_id} in {state_dir}")

    def exists(self) -> bool:
        return self.path.exists()

    def append(self, ev: str, **fields) -> None:
        """Durably record one event. Raises OSError if it cannot be written.

        Callers write before acting (intent before create) and must treat a
        failure here as a reason not to act.
        """
        fields = {k: v for k, v in fields.items() if v is not None}
        line = json.dumps({"ev": ev, "ts": round(time.time(), 3), **fields}, sort_keys=True)
        data = line.encode() + b"\n"
        with self._lock:
            with open(self.path, "a+b") as f:
                # A crash can leave a partial last line. Terminate it and say
                # so, or this record would be glued to the fragment and lost.
                f.seek(0, os.SEEK_END)
                if f.tell():
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) != b"\n":
                        data = b'\n{"ev": "torn_tail"}\n' + data
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

    def state(self) -> RunState:
        s = RunState()
        if not self.path.exists():
            return s
        lines = self.path.read_text(errors="replace").splitlines(keepends=True)
        events = []
        for raw in lines:
            try:
                e = json.loads(raw)
                if not isinstance(e, dict) or not isinstance(e.get("ev"), str):
                    raise ValueError
                events.append(e)
            except ValueError:
                events.append(None)
        bindings: dict[int, int] = {}  # attempt -> the id some record tied to it
        for n, e in enumerate(events):
            if e is None:
                # Expected only as a write cut short by a crash: a last line
                # without its newline, or one the next append marked as a torn
                # tail. Every record is synced before the step it authorizes,
                # so such a fragment never hides a box. A complete line that
                # cannot be read is something else: damage.
                last_and_unterminated = n == len(events) - 1 and not lines[n].endswith("\n")
                marked = n + 1 < len(events) and (events[n + 1] or {}).get("ev") == "torn_tail"
                if not (last_and_unterminated or marked):
                    s.problems.append(f"line {n + 1} is unreadable")
                continue
            try:
                if self._apply(s, e, bindings) == "stop":
                    # What follows belongs to another run: none of it is ours to act on.
                    s.problems.append(f"line {n + 1} starts another run; the rest is ignored")
                    break
            except (KeyError, TypeError, ValueError) as exc:
                s.problems.append(f"line {n + 1} ({e.get('ev')}) is malformed: {exc!r}")
        for seq, instance_id in bindings.items():
            a = s.attempts.get(seq)
            if a is None or s.attempt_of(instance_id) is not None:
                continue
            if a.instance_id is None:
                a.instance_id = instance_id
            else:
                # Two ids for one attempt: both were given to us, both are ours.
                s.problems.append(f"attempt {seq} has ids {a.instance_id} and {instance_id}")
        for a in s.attempts.values():
            if a.instance_id is not None:
                a.status = s.boxes[a.instance_id]
        # `closed` is believed only if the file is sound and nothing is live.
        s.closed = s.closed and not s.is_live() and not s.problems
        return s

    @staticmethod
    def _know(s: RunState, bindings: dict, e: dict, status: str) -> None:
        """Learn an instance id, or advance its status. Ids are never dropped.

        The id is recorded before anything else about the record is looked at,
        so a record whose other fields are damaged still yields its id.
        """
        instance_id = e["instance_id"]
        if not positive_int(instance_id):
            raise ValueError(f"bad instance id {instance_id!r}")
        order = ("created", "destroying", "destroyed")
        old = s.boxes.get(instance_id)
        if old is None or order.index(status) > order.index(old):
            s.boxes[instance_id] = status
        seq = e.get("seq")
        if seq is None:
            return
        if not positive_int(seq):
            raise ValueError(f"bad attempt number {seq!r}")
        if bindings.setdefault(seq, instance_id) != instance_id:
            s.problems.append(f"attempt {seq} has ids {bindings[seq]} and {instance_id}")

    @classmethod
    def _apply(cls, s: RunState, e: dict, bindings: dict):
        ev = e["ev"]
        if ev in ("created", "destroying", "destroyed"):
            return cls._know(s, bindings, e, ev)
        if ev == "winner":
            cls._know(s, bindings, e, "created")
            s.winner = {k: e[k] for k in ("seq", "instance_id", "host", "port")}
            return None
        if ev == "run":
            if s.run_id is not None:
                # A journal holds one run. Whatever a second header says, or
                # fails to say, what follows it is not this run's.
                return "stop"
            run_id = e["run_id"]
            if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
                raise ValueError(f"bad run id {run_id!r}")
            s.run_id, s.namespace, s.meta = run_id, str(e["namespace"]), e
            return None
        if ev in ("closed", "torn_tail"):
            s.closed = s.closed or ev == "closed"
            return None
        if ev not in ("intent", "refused", "unknown", "abandoned"):
            raise ValueError("unknown event")
        seq = e["seq"]
        if not positive_int(seq):
            raise ValueError(f"bad attempt number {seq!r}")
        a = s.attempts.get(seq)
        if ev == "intent":
            label = e["label"]
            if not isinstance(label, str):
                raise ValueError(f"bad label {label!r}")
            same = (label, e.get("machine_id"), e.get("offer_id"))
            if a is not None:
                if (a.label, a.machine_id, a.offer_id) != same:
                    raise ValueError("a second, different intent for the same attempt")
                return None
            ts = e.get("ts")
            s.attempts[seq] = AttemptState(
                seq=seq,
                label=label,
                ts=float(ts) if finite_number(ts) else None,
                offer_id=e.get("offer_id"),
                machine_id=e.get("machine_id"),
                host_id=e.get("host_id"),
                dph=e.get("dph") or 0.0,
            )
            return None
        if a is None:
            raise ValueError("no such attempt")
        allowed = ("intent",) if ev != "abandoned" else OPEN
        if a.instance_id is None and seq not in bindings and a.status in allowed:
            a.status = ev
        return None

    def rotate(self) -> None:
        """Move a finished run's journal aside so the directory can host a new run."""
        s = self.state()
        if s.is_live() or s.problems:
            raise StateError(f"journal {self.path} is not finished: {s.problems or 'boxes live'}")
        target = self.path.with_name(f"journal.{s.run_id or 'empty'}.jsonl")
        if target.exists():
            target = target.with_name(f"{target.stem}.{secrets.token_hex(3)}.jsonl")
        self.path.rename(target)
