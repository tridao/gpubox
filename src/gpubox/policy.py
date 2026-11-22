"""Observations of machines, and the blocklist derived from them.

Observations are facts ("this machine's GPU was busy when we rented it"); the
blocklist is a policy applied to them, one pure function with a version. Only
reason codes are stored, never provider messages or script output, so the
file can be shared.

The durations are guesses (see the design notes) except the two-hour cooldown
for a busy GPU, which follows from two machines seen recovering in 25 and 90
minutes.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

POLICY_VERSION = 1
HOUR, DAY = 3600, 86400
STRIKE_WINDOW = 30 * DAY

# reason -> (class, first block, cap). Repeats within STRIKE_WINDOW double it.
REASONS = {
    "busy_gpu": ("soft", 2 * HOUR, 2 * DAY),
    "unreachable": ("soft", 1 * DAY, 14 * DAY),
    "no_network": ("soft", 1 * DAY, 14 * DAY),
    "died": ("soft", 1 * DAY, 14 * DAY),
    "runtime_error": ("hard", 14 * DAY, 60 * DAY),
    "wrong_hardware": ("hard", 14 * DAY, 60 * DAY),
    "cuda_init": ("hard", 14 * DAY, 60 * DAY),
    "compute_mismatch": ("hard", 14 * DAY, 60 * DAY),
}
KINDS = ("healthy", "failed_admission", "died_after_admission")


def observation(kind, machine_id, host_id, run_id, *, reason=None, now=None, **extra) -> dict:
    if kind not in KINDS:
        raise ValueError(f"unknown observation kind {kind!r}")
    if (kind == "healthy") != (reason is None) or (reason and reason not in REASONS):
        raise ValueError(f"bad reason {reason!r} for {kind}")
    return {
        "id": uuid.uuid4().hex,
        "ts": int(now if now is not None else time.time()),
        "provider": "vast",
        "machine_id": int(machine_id),
        "host_id": host_id,
        "run_id": run_id,
        "kind": kind,
        "reason_code": reason,
        "policy_version": POLICY_VERSION,
        **extra,
    }


class Observations:
    """An append-only JSONL file. Unreadable lines are skipped, never fatal."""

    def __init__(self, path: str | os.PathLike | None):
        self.path = Path(path) if path else None
        self.rows: list[dict] = []
        self.added = 0
        if self.path and self.path.exists():
            for raw in self.path.read_text(errors="replace").splitlines():
                try:
                    row = json.loads(raw)
                except ValueError:
                    continue
                if _valid(row):
                    self.rows.append(row)

    def add(self, row: dict) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        self.rows.append(row)
        self.added += 1  # counted only once it is on disk


def _valid(row) -> bool:
    """Is this a row the policy can use? The file is a cache: anything else is skipped."""
    if not isinstance(row, dict):
        return False
    kind, reason, ts = row.get("kind"), row.get("reason_code"), row.get("ts")
    return (
        type(row.get("machine_id")) is int
        and type(ts) in (int, float)
        and -1e15 < ts < 1e15  # also false for NaN, and for integers too big to compare
        and isinstance(kind, str)
        and kind in KINDS
        # The same rule new observations obey: healthy has no reason, a failure has one.
        and (reason is None if kind == "healthy" else isinstance(reason, str))
        and (row.get("id") is None or isinstance(row["id"], str))
        and type(row.get("policy_version", 1)) is int
    )


def blocklist(rows: list[dict], now: float) -> dict[int, dict]:
    """machine_id -> {"until", "class", "reason"} for machines blocked at `now`.

    Repeats are counted per class (soft, hard), not per reason: a machine that
    was busy yesterday and is unreachable today is on its second soft strike.
    A row seen twice (same id) counts once. Rows in the same second are
    ordered by id, which is arbitrary but the same for every reader.
    """
    per_machine: dict[int, list[dict]] = {}
    seen = set()
    for row in rows:
        if not _valid(row) or row.get("policy_version", 1) > POLICY_VERSION:
            continue
        if row.get("id") is not None:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
        per_machine.setdefault(row["machine_id"], []).append(row)
    out = {}
    for machine, obs in per_machine.items():
        block = {"soft": (0, None), "hard": (0, None)}  # class -> (until, reason)
        strikes: dict[str, list[int]] = {"soft": [], "hard": []}
        for o in sorted(obs, key=lambda o: (o["ts"], str(o.get("id", "")))):
            if o["kind"] == "healthy":
                block["soft"] = (0, None)  # it works now; history still counts
                continue
            reason = o.get("reason_code")
            if reason not in REASONS:
                continue
            cls, first, cap = REASONS[reason]
            recent = [t for t in strikes[cls] if o["ts"] - t <= STRIKE_WINDOW]
            strikes[cls] = [*recent, o["ts"]]
            until = o["ts"] + min(first * 2 ** len(recent), cap)
            if until > block[cls][0]:
                block[cls] = (until, reason)
        cls = "hard" if block["hard"][0] > now else "soft"
        until, reason = block[cls]
        if until > now:
            out[machine] = {"until": until, "class": cls, "reason": reason}
    return out


def select_offers(offers: list[dict], blocked: dict[int, dict], want: int) -> list[dict]:
    """Offers to try, in order: one machine per host, blocked machines dropped.

    Racing two machines of one host does not diversify the failure risk. If
    fewer than `want` candidates remain, soft-blocked machines are re-admitted
    at the end, the one whose block expires soonest first. Hard-blocked
    machines never are, and the blocklist is never dropped wholesale.
    """

    def one_per_host(candidates):
        out, hosts = [], set()
        for offer in candidates:
            # An offer without a host id is its own host.
            host = offer.get("host_id") or ("offer", offer.get("id"))
            if host not in hosts:
                hosts.add(host)
                out.append(offer)
        return out

    chosen = one_per_host(o for o in offers if o.get("machine_id") not in blocked)
    if len(chosen) < want:
        soft = [o for o in offers if blocked.get(o.get("machine_id"), {}).get("class") == "soft"]
        soft.sort(key=lambda o: blocked[o["machine_id"]]["until"])
        chosen = one_per_host([*chosen, *soft])
    return chosen
