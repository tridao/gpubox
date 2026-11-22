"""The record of what a run rented. It is what authorizes `down` to destroy.

A label on an instance is mutable (by us, by a teammate, by code on the box),
so it is only ever a search hint. Ownership is this append-only file: every
intended create is written before the request is sent, the instance id as
soon as it is known, and every destroy with its confirmation.
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

# An attempt whose box may still exist and cost money.
LIVE = ("intent", "unknown", "created", "destroying")


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
    parts = (label or "").split(".")
    if len(parts) != 4 or not NAMESPACE.fullmatch(parts[0]) or not NAME.fullmatch(parts[1]):
        return None
    if not re.fullmatch(r"[0-9a-f]{8}", parts[2]) or not re.fullmatch(r"a\d+", parts[3]):
        return None
    return parts[0]


@dataclass
class AttemptState:
    seq: int
    label: str
    status: str = "intent"  # intent|refused|unknown|created|destroying|destroyed|absent
    instance_id: int | None = None
    machine_id: int | None = None
    host_id: int | None = None
    dph: float = 0.0


@dataclass
class RunState:
    run_id: str | None = None
    namespace: str | None = None
    attempts: dict[int, AttemptState] = field(default_factory=dict)
    winner: dict | None = None
    closed: bool = False

    def live(self) -> list[AttemptState]:
        return [a for a in self.attempts.values() if a.status in LIVE]

    def by_instance(self, instance_id: int) -> AttemptState | None:
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
    def __init__(self, state_dir: Path):
        self.path = Path(state_dir) / "journal.jsonl"
        self._lock = threading.Lock()

    def exists(self) -> bool:
        return self.path.exists()

    def append(self, ev: str, **fields) -> None:
        """Durably record one event. Raises if it cannot be written.

        Callers write before acting (intent before create) and must treat a
        failure here as a reason not to act.
        """
        line = json.dumps({"ev": ev, "ts": round(time.time(), 3), **fields}, sort_keys=True)
        with self._lock:
            with open(self.path, "a") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

    def state(self) -> RunState:
        s = RunState()
        if not self.path.exists():
            return s
        for raw in self.path.read_text().splitlines():
            try:
                e = json.loads(raw)
            except ValueError:
                continue  # a torn last line after a crash
            ev = e.get("ev")
            a = s.attempts.get(e.get("seq")) if "seq" in e else None
            if ev == "run":
                s.run_id, s.namespace = e["run_id"], e["namespace"]
            elif ev == "intent":
                s.attempts[e["seq"]] = AttemptState(
                    seq=e["seq"],
                    label=e["label"],
                    machine_id=e.get("machine_id"),
                    host_id=e.get("host_id"),
                    dph=e.get("dph") or 0.0,
                )
            elif ev in ("refused", "unknown", "absent") and a:
                a.status = ev
            elif ev == "created" and a:
                a.status, a.instance_id = "created", e["instance_id"]
            elif ev in ("destroying", "destroyed"):
                a = s.by_instance(e["instance_id"])
                if a:
                    a.status = ev
            elif ev == "winner":
                s.winner = {k: e[k] for k in ("seq", "instance_id", "host", "port")}
            elif ev == "closed":
                s.closed = True
        return s

    def rotate(self) -> None:
        """Move a closed run's journal aside so the directory can host a new run."""
        s = self.state()
        self.path.rename(self.path.with_name(f"journal.{s.run_id or 'old'}.jsonl"))
