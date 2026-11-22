"""Rent one verified box: a small race, a health gate, and guaranteed cleanup.

Every attempt moves through

    intent -> created -> reachable -> healthy
       |          \\-> failed -> destroying -> destroyed
       |-> refused   (the provider said no: no box exists)
       \\-> unknown   (no usable answer to the create: a box may exist)

and each transition is journalled. An attempt counts against `max_live` until
its box is confirmed destroyed (or the create was refused), so a failed box
whose delete has not been confirmed, or a create whose outcome is unknown,
blocks further renting instead of being replaced blindly.
"""

from __future__ import annotations

import collections
import re
import signal
import threading
import time
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from . import policy as pol
from . import vast as v
from .gate import GATE_VERSION
from .journal import LIVE, Journal, attempt_label, label_namespace

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


class ConfigError(RuntimeError):
    """Our mistake (image, key, arguments): another host cannot help."""


@dataclass(frozen=True)
class Spec:
    image: str
    gpu: str
    max_dph_per_gpu: float
    num_gpus: int = 1
    capability: tuple[int, int] | None = None  # None: do not check
    min_vram_mib: int = 0
    min_cuda: float = 12.0
    disk_gb: int = 40
    min_reliability: float = 0.98
    min_cpus_per_gpu: int = 8
    min_ram_gb_per_gpu: int = 32
    min_inet_down_mbps: int = 200
    must_reach: tuple[str, ...] = ()
    ttl_min: int = 90


@dataclass(frozen=True)
class Policy:
    spare: int = 1  # candidates raced beyond the one we keep
    max_attempts: int = 6
    startup_timeout: float = 300
    overall_timeout: float = 900
    cleanup_reserve: float = 60  # stop renting this long before overall_timeout
    poll: float = 5
    reconcile_timeout: float = 60  # how long to look for a box after an unknown create
    settle: float = 300  # after this long unseen, an unknown create is taken as never made
    confirm_timeout: float = 30  # how long to wait for a delete to show
    confirm_poll: float = 2

    @property
    def max_live(self) -> int:
        return 1 + self.spare


@dataclass
class Attempt:
    seq: int
    label: str
    machine_id: int
    host_id: int | None
    offer_id: int
    dph: float
    geo: str = ""
    state: str = "intent"  # mirrors the journal
    instance_id: int | None = None
    endpoint: tuple[str, int] | None = None
    ok: bool = False
    reason: str | None = None  # a policy.REASONS code when the machine is at fault
    detail: str = ""  # free text; stays in logs, never in observations
    timings: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)


@dataclass
class Result:
    winner: Attempt | None
    attempts: list[Attempt]
    seconds: float

    @property
    def unresolved(self) -> list[Attempt]:
        """Attempts whose box may still exist."""
        return [a for a in self.attempts if a is not self.winner and a.state in LIVE]


def destroy_confirmed(vast, journal: Journal, instance_id: int, policy: Policy) -> bool:
    """Destroy an instance and wait until the provider no longer lists it."""
    _note(journal, "destroying", instance_id=instance_id)
    deadline = time.monotonic() + policy.confirm_timeout
    sent = False
    while True:
        try:
            if not sent:
                vast.destroy(instance_id)
                sent = True
            if vast.instance(instance_id) is None:
                _note(journal, "destroyed", instance_id=instance_id)
                return True
        except v.VastError as exc:
            log(f"WARNING: destroying instance {instance_id}: {exc}")
        if time.monotonic() >= deadline:
            return False
        time.sleep(policy.confirm_poll)


def _note(journal: Journal, ev: str, **fields) -> None:
    """Journal a step of cleanup; a write failure must not stop the cleanup.

    Not for acquiring ownership: an instance id that cannot be journalled must
    fail the attempt (see Launch._bring_up).
    """
    try:
        journal.append(ev, **fields)
    except OSError as exc:
        log(f"WARNING: journal write failed ({ev} {fields}): {exc}")


@contextmanager
def _signals_deferred():
    """Hold SIGINT/SIGTERM until the block ends, then deliver one.

    Cleanup must not be cut short by the signal that asked for it (or a second
    one). Everything inside is bounded by timeouts. Outside the main thread
    signals cannot be redirected, and nothing is changed.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    got, old = [], {}

    def hold(signum, frame):
        got.append(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        old[sig] = signal.signal(sig, hold)
    try:
        yield
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
    if got:
        raise KeyboardInterrupt


def find_by_label(listing: list[dict], label: str, *, not_before: float, image, machine_id):
    """The one instance that plausibly is the box of an attempt we lost track of.

    A label is mutable, so it is a hint, not proof. To be taken for ours an
    instance must also sit on the machine the attempt asked for, have been
    started after we wrote the intent, and run our image; and it must be the
    only such instance. What remains possible is an impostor matching all
    four, which we accept. Returns (instance or None, number carrying the label).
    """
    labelled = [i for i in listing if i.get("label") == label]
    plausible = []
    for inst in labelled:
        started, img, machine = (
            inst.get("start_date"),
            inst.get("image_uuid"),
            inst.get("machine_id"),
        )
        if started is not None and float(started) < not_before - 300:
            continue
        if image and img and img != image:
            continue
        if machine_id is not None and machine is not None and machine != machine_id:
            continue
        plausible.append(inst)
    return (plausible[0] if len(plausible) == 1 else None), len(labelled)


class Launch:
    def __init__(self, vast, spec: Spec, policy: Policy, journal: Journal, prober, run_id: str):
        self.vast, self.spec, self.policy = vast, spec, policy
        self.journal, self.prober, self.run_id = journal, prober, run_id
        self.stop = threading.Event()
        self.attempts: dict[int, Attempt] = {}
        self.rent_until = float("inf")  # monotonic; no create is sent after this
        self.intent_ts: dict[int, float] = {}

    # ------------------------------------------------------------- one attempt

    def _bring_up(self, a: Attempt) -> Attempt:
        spec, policy = self.spec, self.policy
        t0 = time.monotonic()
        elapsed = lambda: round(time.monotonic() - t0, 1)
        tag = f"[m{a.machine_id} a{a.seq}]"
        if self.stop.is_set() or time.monotonic() >= self.rent_until:
            # Nothing was sent, so there is certainly no box.
            a.state, a.detail = "refused", "cancelled before create"
            _note(self.journal, "refused", seq=a.seq)
            return a
        try:
            payload = v.create_payload(spec, a.label, self.prober.pubkey)
            a.instance_id = self.vast.create(a.offer_id, payload)
        except v.CreateRefused as exc:
            # Usually a stale offer (rented between search and create).
            a.state, a.detail = "refused", f"create refused: {exc}"
            _note(self.journal, "refused", seq=a.seq)
            return a
        except v.CreateUnknown as exc:
            a.state, a.detail = "unknown", f"create outcome unknown: {exc}"
            _note(self.journal, "unknown", seq=a.seq)
            self._reconcile(a)
            return a
        a.state = "created"
        # Ownership is the journal. If the id cannot be recorded the box must
        # not be admitted: this raises, and the launch rolls back by the id we
        # still hold in memory.
        self.journal.append("created", seq=a.seq, instance_id=a.instance_id)
        log(f"  {tag} rented instance {a.instance_id}: ${a.dph:.2f}/hr, host {a.host_id}, {a.geo}")

        status = msg = ssh_err = ""
        answered = False  # did the provider ever tell us anything about this box?
        while True:
            if self.stop.wait(policy.poll):
                a.detail = "cancelled"
                return a
            if elapsed() > policy.startup_timeout:
                extra = ssh_err or msg[:120] or "no status message"
                # Only the provider's own account of the box is evidence
                # against the machine; our failure to reach the API is not.
                a.reason = "unreachable" if answered else None
                a.detail = (
                    f"not reachable after {policy.startup_timeout:g}s (status={status}; {extra})"
                    if answered
                    else f"provider API gave no status for {policy.startup_timeout:g}s"
                )
                return a
            try:
                inst = self.vast.instance(a.instance_id)
            except v.VastError:
                continue
            if not inst:
                continue
            answered = True
            status, msg = inst.get("actual_status") or "none", str(inst.get("status_msg") or "")
            kind = v.classify_status(msg)
            if kind == "image":
                raise ConfigError(f"image {spec.image!r} cannot be pulled: {msg[:200]}")
            if kind == "machine":
                a.reason, a.detail = "runtime_error", msg[:200]
                return a
            if status == "exited":
                # Our container command ends only when sshd could not be started.
                a.reason, a.detail = (
                    "unreachable",
                    f"container exited during bootstrap ({msg[:120]})",
                )
                return a
            ep = v.endpoint_of(inst)
            if status != "running" or ep is None:
                continue
            a.timings.setdefault("running", elapsed())
            ok, ssh_err = self.prober.ready(ep)
            if ok:
                a.endpoint = ep
                break
        a.timings["ssh"] = elapsed()
        log(f"  {tag} reachable after {a.timings['ssh']:.0f}s, running health gate")
        a.ok, a.reason, a.detail, a.report = self.prober.gate(a.endpoint, spec)
        a.timings["healthy"] = elapsed()
        return a

    def _reconcile(self, a: Attempt) -> None:
        """After a create with no usable answer, look for the box by its label.

        The label was part of the create request and nothing of ours has run
        on the box yet, so at this point it is a reliable correlation id.
        """
        deadline = time.monotonic() + self.policy.reconcile_timeout
        count = 0
        while True:
            try:
                inst, count = find_by_label(
                    self.vast.instances(),
                    a.label,
                    not_before=self.intent_ts[a.seq],
                    image=self.spec.image,
                    machine_id=a.machine_id,
                )
                if inst:
                    a.instance_id, a.state = int(inst["id"]), "created"
                    a.detail += f"; found as instance {a.instance_id}"
                    _note(self.journal, "created", seq=a.seq, instance_id=a.instance_id)
                    return
            except v.VastError:
                pass
            if time.monotonic() >= deadline:
                if count:
                    a.detail += f"; {count} instance(s) carry its label, none provably ours"
                return
            time.sleep(self.policy.poll)

    def _destroy(self, a: Attempt) -> None:
        a.state = "destroying"
        if destroy_confirmed(self.vast, self.journal, a.instance_id, self.policy):
            a.state = "destroyed"
        else:
            log(f"WARNING: instance {a.instance_id} is not confirmed destroyed")

    # ---------------------------------------------------------------- the race

    def run(self, blocked: dict[int, dict]) -> Result:
        """Race candidates until one is healthy and durably recorded as the winner.

        Until that record is written nothing is kept: any exception, including
        one raised by another candidate or by the journal, destroys every box
        this launch rented, the would-be winner included.
        """
        spec, policy = self.spec, self.policy
        t0 = time.monotonic()
        self.rent_until = t0 + policy.overall_timeout - policy.cleanup_reserve
        found = [o for o in self.vast.offers(v.offer_query(spec)) if _price(o) is not None]
        offers = collections.deque(pol.select_offers(found, blocked, policy.max_live))
        log(
            f"{len(offers)} candidate host(s) for {spec.num_gpus}x {spec.gpu} "
            f"({len(blocked)} machine(s) blocked); racing {policy.max_live}, "
            f"up to {policy.max_attempts} attempts"
        )
        pool = ThreadPoolExecutor(max_workers=policy.max_live, thread_name_prefix="gpubox")
        pending, judged, winner, handed_off = set(), [], None, False

        def live() -> int:
            return sum(a.state in LIVE for a in self.attempts.values())

        def fill():
            while (
                live() < policy.max_live
                and len(self.attempts) < policy.max_attempts
                and offers
                and time.monotonic() < self.rent_until
            ):
                o = offers.popleft()
                seq = len(self.attempts) + 1
                a = Attempt(
                    seq=seq,
                    label=attempt_label(self.run_id, seq),
                    machine_id=o.get("machine_id"),
                    host_id=o.get("host_id"),
                    offer_id=o["id"],
                    dph=_price(o),
                    geo=str(o.get("geolocation") or ""),
                )
                # Written, and synced, before the request: if this fails we do not rent.
                self.intent_ts[seq] = time.time()
                self.journal.append(
                    "intent",
                    seq=seq,
                    label=a.label,
                    offer_id=a.offer_id,
                    machine_id=a.machine_id,
                    host_id=a.host_id,
                    dph=a.dph,
                )
                self.attempts[seq] = a
                pending.add(pool.submit(self._bring_up, a))

        try:
            fill()
            while pending and winner is None:
                remaining = self.rent_until - time.monotonic()
                if remaining <= 0:
                    log(f"giving up: no healthy box within {policy.overall_timeout:g}s")
                    break
                done, _ = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
                for fut in done:
                    pending.discard(fut)
                    a = fut.result()  # a worker's ConfigError or journal error surfaces here
                    judged.append(a)
                    tag = f"[m{a.machine_id} a{a.seq}]"
                    if a.ok and winner is None:
                        winner = a
                        log(f"  {tag} healthy after {a.timings['healthy']:.0f}s")
                        continue
                    if not a.ok:
                        log(f"  {tag} failed ({a.reason or 'not blamed'}): {a.detail}")
                    if a.state == "created":
                        self._destroy(a)  # stop paying, and free its slot, now
                if winner is None:
                    fill()
            if winner:
                self.journal.append(
                    "winner",
                    seq=winner.seq,
                    instance_id=winner.instance_id,
                    host=winner.endpoint[0],
                    port=winner.endpoint[1],
                )
                handed_off = True
        finally:
            self.stop.set()
            if not handed_off:
                winner = None
            with _signals_deferred():
                self._cleanup(pool, keep=winner)

        for a in self.attempts.values():
            if a not in judged:
                # Finished after the race was decided, with its box destroyed
                # under it: whatever it reports says nothing about the machine.
                a.ok, a.reason = False, None
                a.detail = a.detail or "cancelled"
            elif a.ok and a is not winner:
                a.ok, a.detail = False, "healthy, but another candidate won"
        ordered = [self.attempts[k] for k in sorted(self.attempts)]
        return Result(winner=winner, attempts=ordered, seconds=time.monotonic() - t0)

    def _cleanup(self, pool, keep: Attempt | None) -> None:
        """Destroy every box of this launch except `keep`, and wait for the workers."""
        others = [a for a in self.attempts.values() if a is not keep]
        for a in others:  # unblock threads waiting on SSH to these boxes
            if a.state == "created":
                try:
                    self.vast.destroy(a.instance_id)
                except v.VastError:
                    pass
        pool.shutdown(wait=True)
        for a in others:  # includes boxes created or found after the race was decided
            if a.state in ("created", "destroying"):
                try:
                    self._destroy(a)
                except Exception as exc:  # one box must not stop the cleanup of the rest
                    log(f"WARNING: cleaning up instance {a.instance_id}: {exc!r}")


def _price(offer: dict) -> float | None:
    """The offer's hourly price, or None if it has no usable one."""
    dph = offer.get("dph_total")
    if isinstance(dph, bool) or not isinstance(dph, (int, float)) or not 0 < dph < float("inf"):
        return None
    return float(dph)


def observe(result: Result, spec: Spec, run_id: str, obs: pol.Observations) -> None:
    """Record what this launch taught us about machines.

    A failure is blamed on the machine only if another candidate, with the
    same image, GPU request and gate, became healthy in this launch. If
    nothing came up the cause may be ours, and blaming every machine tried
    would shrink the pool for no reason.
    """
    if not result.winner:
        return
    common = {"gpu": spec.gpu, "num_gpus": spec.num_gpus, "gate_version": GATE_VERSION}
    for a in result.attempts:
        if a is result.winner:
            secs = a.timings.get("healthy")
            obs.add(
                pol.observation(
                    "healthy", a.machine_id, a.host_id, run_id, secs_to_ready=secs, **common
                )
            )
        elif a.reason:
            obs.add(
                pol.observation(
                    "failed_admission", a.machine_id, a.host_id, run_id, reason=a.reason, **common
                )
            )


def down(vast, journal: Journal, policy: Policy, *, now=None) -> int:
    """Destroy everything the journalled run rented. Returns how many attempts are unresolved.

    Boxes whose id is known are destroyed by id, whatever their label says.
    An attempt whose create may have been sent but whose id we never learned
    is looked up by label: a single plausible match is destroyed. If nothing
    carries the label, the attempt stays unresolved until `policy.settle`
    seconds after the intent, and only then is recorded as never created.
    """
    now = time.time() if now is None else now
    state = journal.state()
    if state.run_id is None or state.closed:
        log("nothing to destroy")
        return 0
    if state.skipped:
        log(f"WARNING: {state.skipped} unreadable journal line(s) were skipped")
    left = 0
    attempts = sorted(state.attempts.values(), key=lambda a: a.seq)
    for a in attempts:  # known ids first: nothing below may prevent these
        if a.status in ("created", "destroying"):
            if destroy_confirmed(vast, journal, a.instance_id, policy):
                log(f"destroyed instance {a.instance_id} (attempt {a.seq})")
            else:
                left += 1
                log(f"ERROR: instance {a.instance_id} (attempt {a.seq}) not confirmed destroyed")
    listing = None
    for a in attempts:
        if a.status not in ("intent", "unknown"):
            continue
        try:
            listing = vast.instances() if listing is None else listing
        except v.VastError as exc:
            left += 1
            log(f"ERROR: attempt {a.seq} ({a.label}) unresolved, cannot list instances: {exc}")
            continue
        inst, count = find_by_label(
            listing,
            a.label,
            not_before=a.ts,
            image=state.meta.get("image"),
            machine_id=a.machine_id,
        )
        if inst:
            iid = int(inst["id"])
            _note(journal, "created", seq=a.seq, instance_id=iid)
            if destroy_confirmed(vast, journal, iid, policy):
                log(f"destroyed instance {iid} (attempt {a.seq}, found by label)")
                continue
            log(f"ERROR: instance {iid} (attempt {a.seq}) not confirmed destroyed")
        elif count:
            log(
                f"ERROR: attempt {a.seq}: {count} instance(s) labelled {a.label}, none provably "
                "ours (other machine or image, or older than the attempt); not touching them"
            )
        elif now - a.ts > policy.settle:
            _note(journal, "absent", seq=a.seq)
            log(
                f"attempt {a.seq}: nothing labelled {a.label} after {policy.settle:g}s; never created"
            )
            continue
        else:
            log(
                f"attempt {a.seq}: nothing labelled {a.label} yet; unresolved until "
                f"{policy.settle:g}s after the attempt (run `down` again)"
            )
        left += 1
    if not left:
        _note(journal, "closed")
    return left


def reap(vast, namespace, max_age_min, *, destroy, legacy=None, now=None) -> int:
    """Find instances of a namespace that outlived any run. Returns failures.

    Destroying by label is only safe in a namespace the operator declares
    exclusive (every instance carrying it is disposable), which is what
    passing `destroy=True` asserts. Without it this only reports.

    `legacy` is a regular expression for labels of an older scheme, matched in
    full and never applied to a label of the current scheme.
    """
    now = time.time() if now is None else now
    failed = 0
    for inst in vast.instances():
        label = inst.get("label") or ""
        ns = label_namespace(label)
        ours = ns == namespace
        old = bool(legacy) and ns is None and re.fullmatch(legacy, label) is not None
        if not (ours or old):
            continue
        started = inst.get("start_date")
        if not started:
            log(f"instance {inst['id']} ({label}): no start time, skipped")
            continue
        age = (now - float(started)) / 60
        stale = age > max_age_min
        act = " -> destroy" if stale and destroy else " -> stale (report only)" if stale else ""
        log(f"instance {inst['id']} ({label}): {age:.0f} min old, {inst.get('actual_status')}{act}")
        if stale and destroy:
            try:
                vast.destroy(inst["id"])
            except v.VastError as exc:
                failed += 1
                log(f"ERROR: could not destroy instance {inst['id']}: {exc}")
    return failed
