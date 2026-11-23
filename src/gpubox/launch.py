"""Rent one verified box: a small race, a health gate, and guaranteed cleanup.

The launch is one transaction with a single commit point, the `winner` record
in the journal, written last: after a candidate passed the gate, after every
other box is confirmed gone, and only if nothing failed and nobody asked us to
stop. Before that point any error or signal destroys everything this launch
rented, the candidate included.

Signals do not interrupt the transaction. SIGINT and SIGTERM set a flag; the
loop notices it, rolls back, and only then lets the interruption through.

The box is kept only if all of this holds when the race is over: a candidate
passed the gate; every other box whose id we know is confirmed gone; no
worker ended with an error; no signal arrived. The one thing that may stay
open beside a winner is a create that never gave us an id (see `down`).

An attempt keeps its slot until its box is confirmed destroyed or its create
was explicitly refused. So a box whose delete has not been confirmed, or a
create whose outcome is unknown, blocks further renting instead of being
replaced blindly.
"""

from __future__ import annotations

import collections
import math
import re
import signal
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field

from . import policy as pol
from . import vast as v
from .gate import GATE_VERSION
from .journal import LIVE, Journal, attempt_label, label_namespace, positive_int

_print_lock = threading.Lock()

# How far a provider's clock may run behind ours when we ask whether an
# instance was started after we wrote an intent.
CLOCK_SKEW = 30


def log(msg: str) -> None:
    """Say something. Never raises: a closed pipe must not stop a teardown."""
    try:
        with _print_lock:
            print(msg, flush=True)
    except Exception:
        pass


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
    tick: float = 0.5  # how often the race loop looks at the stop flag
    image_grace: float = 90  # after one host says the image cannot be pulled, how long others get
    reconcile_timeout: float = 60  # how long to look for a box after an unknown create
    settle: float = 300  # youngest age at which the operator may abandon an unseen create
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
    intent_ts: float
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
        """Attempts, other than the winner, whose box may still exist."""
        return [a for a in self.attempts if a is not self.winner and a.state in LIVE]


def _note(journal: Journal, ev: str, **fields) -> bool:
    """Journal a step of cleanup; a write failure must not stop the cleanup.

    Not for acquiring ownership: an instance id that cannot be journalled must
    fail the attempt (see Launch._bring_up). Whoever reports success afterwards
    re-reads the journal, so a record lost here is noticed.
    """
    try:
        journal.append(ev, **fields)
        return True
    except OSError as exc:
        log(f"WARNING: journal write failed ({ev} {fields}): {exc}")
        return False


def destroy_confirmed(vast, journal: Journal, instance_id: int, policy: Policy, seq=None) -> bool:
    """Destroy an instance and wait until the provider says it does not exist.

    The records carry the attempt number as well as the id, so the journal
    learns the id here even if the record that first bound it was lost.
    """
    _note(journal, "destroying", instance_id=instance_id, seq=seq)
    deadline = time.monotonic() + policy.confirm_timeout
    while True:
        try:
            vast.destroy(instance_id)  # idempotent; repeated until the box is gone
        except v.VastError as exc:
            log(f"WARNING: destroying instance {instance_id}: {exc}")
        try:
            # Asked even if the delete failed: the box may be gone already.
            if vast.instance(instance_id) is None:
                _note(journal, "destroyed", instance_id=instance_id, seq=seq)
                return True
        except v.VastError as exc:
            log(f"WARNING: checking instance {instance_id}: {exc}")
        if time.monotonic() >= deadline:
            return False
        time.sleep(policy.confirm_poll)


def match_lost_box(listing: list[dict], label: str, *, not_before, image, machine_id):
    """The instance that is the box of an attempt we lost track of, if we can tell.

    A label is mutable, so it is a hint, not proof. An instance is taken for
    ours only if it is the only one carrying the label and the listing says,
    positively, that it sits on the machine the attempt asked for, runs our
    image, and was started after we wrote the intent. A missing or malformed
    field is missing evidence, not a match. What remains possible is an
    impostor matching all of this while our own create left no visible box;
    that risk is accepted. Returns (instance or None, number carrying the label).
    """
    labelled = [row for row in listing if row.get("label") == label]
    if len(labelled) != 1:
        return None, len(labelled)
    row = labelled[0]
    started = row.get("start_date")
    ours = (
        positive_int(row.get("id"))
        and positive_int(machine_id)
        and type(row.get("machine_id")) is int
        and row["machine_id"] == machine_id
        and isinstance(image, str)
        and bool(image)
        and row.get("image_uuid") == image
        and type(not_before) in (int, float)
        and type(started) in (int, float)
        and math.isfinite(started)
        and started >= not_before - CLOCK_SKEW
    )
    return (row if ours else None), 1


@contextmanager
def _signals_set(flag):
    """While inside, SIGINT and SIGTERM call `flag(signum)` instead of interrupting.

    Only the main thread can redirect signals; elsewhere this changes nothing
    and the caller is responsible for not interrupting the launch. Installed
    before anything is rented, restored on every way out.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    old = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old[sig] = signal.signal(sig, lambda signum, frame: flag(signum))
        yield
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


class Launch:
    def __init__(self, vast, spec: Spec, policy: Policy, journal: Journal, prober, run_id: str):
        self.vast, self.spec, self.policy = vast, spec, policy
        self.journal, self.prober, self.run_id = journal, prober, run_id
        self.stop = threading.Event()  # tells workers to give up; set by ordinary code only
        self.cancelled = False  # a signal asked us to stop; set by the handler only
        self.attempts: dict[int, Attempt] = {}
        self.rent_until = math.inf  # monotonic; no create is sent after this

    def _cancel(self, signum) -> None:
        # A signal handler runs on the main thread between two bytecodes of
        # whatever that thread was doing, possibly `stop.set()` itself, whose
        # lock is not reentrant. So the handler only assigns; the race loop
        # looks at this flag every `tick` and stops the workers itself.
        self.cancelled = True

    # ------------------------------------------------------------- one attempt

    def _bring_up(self, a: Attempt) -> Attempt:
        spec, policy = self.spec, self.policy
        t0 = time.monotonic()
        elapsed = lambda: round(time.monotonic() - t0, 1)
        tag = f"[m{a.machine_id} a{a.seq}]"
        payload = v.create_payload(spec, a.label, self.prober.pubkey)
        if self.stop.is_set() or self.cancelled or time.monotonic() >= self.rent_until:
            # Nothing was sent, so there is certainly no box.
            a.state, a.detail = "refused", "cancelled before create"
            _note(self.journal, "refused", seq=a.seq)
            return a
        try:
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
        # still hold in memory (its `destroying` record names the attempt too).
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
            status, msg = (
                str(inst.get("actual_status") or "none"),
                str(inst.get("status_msg") or ""),
            )
            kind = v.classify_status(msg)
            if kind == "image":
                # Close the renting window here, where the error is seen, and
                # not only when the race loop gets round to this result: it
                # may be busy, and would rent a replacement meanwhile.
                self.rent_until = time.monotonic()
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
        ok, a.reason, a.detail, a.report = self.prober.gate(a.endpoint, spec)
        a.timings["healthy"] = elapsed()
        a.ok = ok
        return a

    def _reconcile(self, a: Attempt) -> None:
        """After a create with no usable answer, look for the box by its label."""
        deadline = time.monotonic() + self.policy.reconcile_timeout
        count = 0
        while True:
            try:
                row, count = match_lost_box(
                    self.vast.instances(),
                    a.label,
                    not_before=a.intent_ts,
                    image=self.spec.image,
                    machine_id=a.machine_id,
                )
                if row:
                    a.instance_id, a.state = row["id"], "created"
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
        """Destroy an attempt's box. Never raises: cleanup of one box must not stop the rest."""
        try:
            a.state = "destroying"
            if destroy_confirmed(self.vast, self.journal, a.instance_id, self.policy, seq=a.seq):
                a.state = "destroyed"
            else:
                log(f"WARNING: instance {a.instance_id} is not confirmed destroyed")
        except Exception as exc:
            log(f"WARNING: cleaning up instance {a.instance_id}: {exc!r}")

    def _cleanup(self, pool, keep: Attempt | None) -> bool:
        """Destroy every box of this launch except `keep`, and wait for the workers.

        Never raises. Returns False if the workers could not be waited for, in
        which case nothing may be concluded about what they rented.
        """
        joined = True
        self.stop.set()
        others = [a for a in self.attempts.values() if a is not keep]
        for a in others:  # unblock threads waiting on SSH to these boxes
            if a.state == "created":
                try:
                    self.vast.destroy(a.instance_id)
                except Exception:
                    pass
        try:
            pool.shutdown(wait=True)
        except Exception as exc:
            joined = False
            log(f"WARNING: waiting for workers: {exc!r}")
        for a in others:  # includes boxes created or found after the race was decided
            if a.state in ("created", "destroying"):
                self._destroy(a)
        return joined

    # ---------------------------------------------------------------- the race

    def run(self, blocked: dict[int, dict]) -> Result:
        """Race candidates; return with at most one box, journalled as the winner.

        Raises whatever failed (after rolling back), or KeyboardInterrupt if a
        signal arrived (likewise). If both happened the failure is raised.
        """
        with _signals_set(self._cancel):
            result = self._run(blocked)
            if self.cancelled and result.winner:
                # Told to stop while still in control: leave nothing behind.
                self._destroy(result.winner)
                result = Result(None, result.attempts, result.seconds)
        if self.cancelled:
            # A signal that arrived after the last check above leaves a winner
            # that is journalled; `down` destroys it.
            raise KeyboardInterrupt
        return result

    def _run(self, blocked: dict[int, dict]) -> Result:
        spec, policy = self.spec, self.policy
        t0 = time.monotonic()
        self.rent_until = t0 + policy.overall_timeout - policy.cleanup_reserve
        offers = collections.deque(
            pol.select_offers(self.vast.offers(v.offer_query(spec)), blocked, policy.max_live)
        )
        log(
            f"{len(offers)} candidate host(s) for {spec.num_gpus}x {spec.gpu} "
            f"({len(blocked)} machine(s) blocked); racing {policy.max_live}, "
            f"up to {policy.max_attempts} attempts"
        )
        pool = ThreadPoolExecutor(max_workers=policy.max_live, thread_name_prefix="gpubox")
        pending, judged, candidate, error, committed = set(), [], None, None, False
        owner: dict = {}  # future -> its attempt

        def live() -> int:
            return sum(a.state in LIVE for a in self.attempts.values())

        def fill():
            while (
                live() < policy.max_live
                and len(self.attempts) < policy.max_attempts
                and offers
                and time.monotonic() < self.rent_until
                and not self.stop.is_set()
                and not self.cancelled
            ):
                o = offers.popleft()
                seq = len(self.attempts) + 1
                a = Attempt(
                    seq=seq,
                    label=attempt_label(self.run_id, seq),
                    machine_id=o["machine_id"],
                    host_id=o.get("host_id"),
                    offer_id=o["id"],
                    dph=float(o["dph_total"]),
                    intent_ts=time.time(),
                    geo=str(o.get("geolocation") or ""),
                )
                # Written, and synced, before the request: if this fails we do not rent.
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
                fut = pool.submit(self._bring_up, a)
                owner[fut] = a
                pending.add(fut)

        image_error = None  # a host said the image cannot be pulled
        give_up_at = self.rent_until
        try:
            fill()
            while pending and candidate is None and not self.cancelled:
                # Look first, then at the clock: a box that came up while we
                # were busy elsewhere (confirming a delete, say) still counts,
                # even if the deadline has passed in the meantime.
                remaining = give_up_at - time.monotonic()
                done, _ = wait(
                    pending,
                    timeout=max(0, min(remaining, policy.tick)),
                    return_when=FIRST_COMPLETED,
                )
                if not done and remaining <= 0:
                    if image_error is None:
                        log(f"giving up: no healthy box within {policy.overall_timeout:g}s")
                    break
                pending.difference_update(done)
                finished = []
                for fut in done:
                    exc = fut.exception()
                    if exc is None:
                        finished.append(fut.result())
                    elif isinstance(exc, ConfigError):
                        if image_error is None:
                            # Usually the image really is missing, and waiting
                            # for the others only costs money: give them a
                            # short while to prove it was this host's problem.
                            image_error = exc
                            give_up_at = min(give_up_at, time.monotonic() + policy.image_grace)
                            # Nothing more is rented, including by a worker that
                            # is queued but has not sent its request yet.
                            self.rent_until = time.monotonic()
                        if owner[fut].state == "created":
                            self._destroy(owner[fut])  # it will never come up
                    else:
                        raise exc  # the journal, or a bug: stop and roll back
                for a in sorted(finished, key=lambda a: a.seq):
                    judged.append(a)
                    tag = f"[m{a.machine_id} a{a.seq}]"
                    if a.ok:
                        log(f"  {tag} healthy after {a.timings['healthy']:.0f}s")
                        candidate = candidate or a
                    else:
                        log(f"  {tag} failed ({a.reason or 'not blamed'}): {a.detail}")
                        if a.state == "created":
                            self._destroy(a)  # stop paying, and free its slot, now
                if candidate is None:
                    # After an image error the renting window is closed, so this
                    # rents nothing; boxes already rented may still come up, and
                    # a healthy one proves the error was that host's own.
                    fill()
            if candidate is None and image_error is not None:
                raise image_error
            if candidate is not None and image_error is not None:
                log(f"  ignored, since a box is healthy: {image_error}")
        except BaseException as exc:
            error = exc

        # One exit path. The candidate is kept only if everything below holds.
        keep = candidate  # set only by an error-free pass of the loop
        joined = self._cleanup(pool, keep)
        late = [f.exception() for f in pending if f.done() and f.exception() is not None]
        late = [e for e in late if not isinstance(e, ConfigError)]
        if error is None and late:
            # A worker failed after the race was decided (or after a signal):
            # e.g. a box whose id could not be journalled. That is the failure
            # to report, whether or not there was a candidate.
            error = late[0]
        if keep is not None and error is None:
            stuck = [
                a
                for a in self.attempts.values()
                if a is not keep and a.instance_id and a.state != "destroyed"
            ]
            if not joined or stuck or self.cancelled:
                why = (
                    "a signal arrived"
                    if self.cancelled
                    else "the workers could not be waited for"
                    if not joined
                    else f"instance {stuck[0].instance_id} is not confirmed destroyed"
                )
                log(f"  not keeping instance {keep.instance_id}: {why}")
            else:
                try:
                    self.journal.append(
                        "winner",
                        seq=keep.seq,
                        instance_id=keep.instance_id,
                        host=keep.endpoint[0],
                        port=keep.endpoint[1],
                    )
                    committed = True
                except OSError as exc:
                    error = exc
        if keep is not None and not committed:
            self._destroy(keep)
            keep = None

        for a in self.attempts.values():
            if a not in judged:
                # Finished after the race was decided, with its box destroyed
                # under it: whatever it reports says nothing about the machine.
                a.ok, a.reason = False, None
                a.detail = a.detail or "cancelled"
            elif a.ok and a is not keep:
                a.ok, a.detail = False, "healthy, but not kept"
        if error is not None:
            raise error
        ordered = [self.attempts[k] for k in sorted(self.attempts)]
        return Result(winner=keep, attempts=ordered, seconds=time.monotonic() - t0)


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


def down(vast, journal: Journal, policy: Policy, *, abandon_unseen=False, now=None) -> int:
    """Destroy everything the journalled run rented. Returns how many things are unresolved.

    Every instance id in the journal is destroyed by id, whatever its label
    says. An attempt whose create may have been sent but whose id we never
    learned is looked up by label (see match_lost_box). If no instance carries
    its label we cannot prove that none exists, so it stays unresolved; with
    `abandon_unseen` the operator declares such attempts, once older than
    `policy.settle`, to have rented nothing.

    The count is taken from the journal as it is on disk afterwards, so a
    record that could not be written is never reported as done.
    """
    now = time.time() if now is None else now
    state = journal.state()
    if state.run_id is None and not state.is_live():
        log("nothing to destroy")
        return 0 if not state.problems else 1
    if state.closed:
        log("nothing to destroy")
        return 0
    for iid in state.live_ids():  # known ids first: nothing below may prevent these
        a = state.attempt_of(iid)
        seq = a.seq if a else None
        if destroy_confirmed(vast, journal, iid, policy, seq=seq):
            log(f"destroyed instance {iid}")
        else:
            log(f"ERROR: instance {iid} is not confirmed destroyed")
    listing = None
    for a in state.open_attempts():
        try:
            listing = vast.instances() if listing is None else listing
        except v.VastError as exc:
            log(f"ERROR: attempt {a.seq} ({a.label}) unresolved, cannot list instances: {exc}")
            continue
        row, count = match_lost_box(
            listing,
            a.label,
            not_before=a.ts,
            image=state.meta.get("image"),
            machine_id=a.machine_id,
        )
        if row:
            if destroy_confirmed(vast, journal, row["id"], policy, seq=a.seq):
                log(f"destroyed instance {row['id']} (attempt {a.seq}, found by label)")
            else:
                log(f"ERROR: instance {row['id']} (attempt {a.seq}) is not confirmed destroyed")
        elif count:
            log(
                f"ERROR: attempt {a.seq}: {count} instance(s) labelled {a.label}, not provably "
                "ours (not alone, other machine or image, or older than the attempt); left alone"
            )
        elif abandon_unseen and a.ts is not None and 0 <= a.ts <= now - policy.settle:
            _note(journal, "abandoned", seq=a.seq)
            log(f"attempt {a.seq}: nothing labelled {a.label}; abandoned on request")
        else:
            log(
                f"attempt {a.seq}: nothing labelled {a.label} is visible, which does not prove "
                "that nothing was rented; unresolved (check the provider, then `down "
                f"--abandon-unseen` once the attempt is {policy.settle:g}s old)"
            )
    final = journal.state()
    for problem in final.problems:
        log(f"ERROR: journal: {problem}")
    left = len(final.live_ids()) + len(final.open_attempts()) + bool(final.problems)
    if not left:
        _note(journal, "closed")  # if this is lost, the next `down` writes it
    return left


def reap(vast, namespace, max_age_min, policy: Policy, *, destroy, legacy=None, now=None) -> int:
    """Find instances of a namespace that outlived any run. Returns failures.

    Destroying by label is only safe in a namespace the operator declares
    exclusive (every instance carrying it is disposable), which is what
    passing `destroy=True` asserts. Without it this only reports.

    `legacy` is a compiled regular expression for labels of an older scheme,
    matched in full and never applied to a label of the current scheme.
    """
    now = time.time() if now is None else now
    failed = 0
    for inst in vast.instances():
        label = inst.get("label")
        ns = label_namespace(label)
        ours = ns is not None and ns == namespace
        old = legacy is not None and ns is None and isinstance(label, str)
        old = old and legacy.fullmatch(label) is not None
        if not (ours or old):
            continue
        started = inst.get("start_date")
        if type(started) not in (int, float) or not math.isfinite(started):
            log(f"instance {inst['id']} ({label}): no start time, skipped")
            continue
        age = (now - started) / 60
        stale = age > max_age_min
        act = " -> destroy" if stale and destroy else " -> stale (report only)" if stale else ""
        log(f"instance {inst['id']} ({label}): {age:.0f} min old, {inst.get('actual_status')}{act}")
        if stale and destroy and not _reap_one(vast, inst["id"], policy):
            failed += 1
            log(f"ERROR: instance {inst['id']} is not confirmed destroyed")
    return failed


def _reap_one(vast, instance_id: int, policy: Policy) -> bool:
    deadline = time.monotonic() + policy.confirm_timeout
    while True:
        try:
            vast.destroy(instance_id)
        except v.VastError as exc:
            log(f"WARNING: destroying instance {instance_id}: {exc}")
        try:
            if vast.instance(instance_id) is None:
                return True
        except v.VastError as exc:
            log(f"WARNING: checking instance {instance_id}: {exc}")
        if time.monotonic() >= deadline:
            return False
        time.sleep(policy.confirm_poll)


def compile_legacy(pattern: str | None):
    if pattern is None:
        return None
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ConfigError(f"--legacy-pattern is not a regular expression: {exc}") from None
