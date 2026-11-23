"""Rent one verified box: a small race, a health gate, and teardown by tag and by id.

DESIGN.md states the rules this module implements and numbers them; the
comments below cite those numbers (S: snapshot, D: destroying, L: launch,
T: teardown, R: reaper).

One thread. The race is a loop: each pass reads the provider's listing, polls
the child processes (SSH probes and health gates), and then decides what to
destroy, whether there is a winner, and whether to rent more.

Two things make a box ours. Its tag, set by the create request, which lets
any process find and destroy it. And its id in the snapshot, which lets this
state directory destroy it even after it changed its label. A box is gone
only when a lookup of its id says it does not exist; a listing never
confirms that.
"""

from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass, field

from . import policy as pol
from . import vast as v
from .gate import GATE_VERSION
from .state import Snapshot, make_tag, parse_tag


def log(msg: str) -> None:
    """Say something. Never raises: a closed pipe must not stop a teardown."""
    try:
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
    cleanup_reserve: float = (
        60  # the race ends, and renting stops, this long before overall_timeout
    )
    poll: float = 5
    image_grace: float = 90  # after one host says the image cannot be pulled, how long others get
    blind_timeout: float = 120  # L7: give up after this long without a complete listing
    settle: float = 60  # how long a create with no answer, or a new id, may stay invisible
    min_lease: float = 600  # L3: a winner is handed over with at least this much lease left
    rollback: float = 10  # L5: what a cancelled CI step gets before it is killed
    confirm_timeout: float = 60  # how long a teardown waits for deletes to show
    confirm_poll: float = 2
    call_timeout: float = 10  # one provider call inside the loop or a teardown
    reap_recheck: float = 30  # R3: between the two looks at an untagged instance
    expiry_margin: float = 60  # R2: clock skew allowed before an expired tag is acted on

    @property
    def max_live(self) -> int:
        return 1 + self.spare


@dataclass
class Attempt:
    seq: int
    tag: str
    machine_id: int
    host_id: int | None
    offer_id: int
    dph: float
    geo: str = ""
    state: str = "open"  # open | refused | bound; mirrors the snapshot
    instance_id: int | None = None
    phase: str = ""  # of a bound attempt: booting | probing | gating | healthy | failed
    endpoint: tuple[str, int] | None = None
    reason: str | None = None  # a policy.REASONS code when the machine is at fault
    detail: str = ""  # free text; stays in logs, never in observations
    timings: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    t0: float = 0.0  # monotonic; when we learned its id
    failed_at: float = 0.0  # monotonic; when it failed
    check: object = None  # the child process it is waiting for, if any
    answered: bool = False  # did the provider ever tell us anything about this box?
    status: str = ""
    msg: str = ""
    ssh_err: str = ""

    @property
    def tag_name(self) -> str:
        return f"[m{self.machine_id} a{self.seq}]"


@dataclass
class Result:
    winner: Attempt | None
    attempts: list[Attempt]
    seconds: float
    left: list[int] = field(
        default_factory=list
    )  # ids, other than the winner's, not confirmed gone
    unseen: list[Attempt] = field(default_factory=list)  # creates with no answer and no box seen


# ------------------------------------------------------------------ destroying


def _try_destroy(vast, instance_id: int, timeout: float) -> None:
    try:
        vast.destroy(instance_id, timeout=timeout, retries=0)
    except v.VastError as exc:
        log(f"WARNING: destroying instance {instance_id}: {exc}")
    except Exception as exc:  # D3: one id's trouble must not stop the others
        log(f"WARNING: destroying instance {instance_id}: {exc!r}")


def _confirm_gone(vast, instance_id: int, timeout: float, on_gone) -> bool:
    """True only if a lookup of this id says it does not exist, and that could be recorded."""
    try:
        if vast.instance(instance_id, timeout=timeout, retries=0) is not None:
            return False
    except v.VastError as exc:
        log(f"WARNING: checking instance {instance_id}: {exc}")
        return False
    except Exception as exc:
        log(f"WARNING: checking instance {instance_id}: {exc!r}")
        return False
    if on_gone is not None:
        try:
            on_gone(instance_id)
        except Exception as exc:
            log(f"WARNING: instance {instance_id} is gone, but that could not be recorded: {exc}")
            return False
    return True


def destroy_confirmed(vast, ids, budget: float, policy: Policy, on_gone=None) -> list[int]:
    """Destroy these instances and confirm each by id. Returns the ids not confirmed gone.

    D1: one delete goes to every id before any is waited for. D2: then each
    is looked up, the delete repeated for those still there, until all are
    gone or the budget ends; no call is asked to take longer than what is
    left, and none is started once nothing is left. D3: an error on one id
    does not stop the others, and that id is not confirmed.

    The first round of deletes is sent whatever the budget: it is the point.
    The timeouts are socket timeouts, so the budget is what we ask for, not a
    bound on the wall clock: a response that trickles in can take longer.
    """
    left = sorted(set(ids))
    if not left:
        return []
    deadline = time.monotonic() + budget

    def share(calls: int) -> float:
        return min(policy.call_timeout, max(0.0, deadline - time.monotonic()) / max(1, calls))

    def spent() -> bool:
        return time.monotonic() >= deadline

    for k, iid in enumerate(left):
        _try_destroy(vast, iid, share(len(left) - k))
    while True:
        for iid in list(left):
            if spent():
                return left
            if _confirm_gone(vast, iid, share(1), on_gone):
                left.remove(iid)
        if not left or spent():
            return left
        time.sleep(min(policy.confirm_poll, max(0.0, deadline - time.monotonic())))
        for k, iid in enumerate(left):
            if spent():
                return left
            _try_destroy(vast, iid, share(len(left) - k))


def _try_list(vast, policy: Policy) -> list[dict] | None:
    """A complete listing, or None. Not retried here: the caller's loop is the retry."""
    try:
        return vast.instances(timeout=policy.call_timeout, retries=0)
    except v.VastError as exc:
        log(f"WARNING: listing instances: {exc}")
        return None


def _usable_time(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


# ---------------------------------------------------------------------- launch


class Launch:
    def __init__(self, vast, spec: Spec, policy: Policy, snap: Snapshot, prober):
        self.vast, self.spec, self.policy = vast, spec, policy
        self.snap, self.prober = snap, prober
        self.attempts: dict[int, Attempt] = {}
        # Every id we hold, whether or not the snapshot could record it (S2).
        self.held: set[int] = set()
        self.rent_open = True  # L1: closed for good by an image error

    # ------------------------------------------------------------ the frame

    def run(self, blocked: dict[int, dict]) -> Result:
        """Race candidates; return with at most one box, published as the winner.

        Raises whatever failed, or KeyboardInterrupt on a signal, after
        trying to roll back. L4: a winner that reached the snapshot is kept
        whatever was raised; everything else the run knows is destroyed.
        """
        t0 = time.monotonic()
        handing_over = None  # the box whose hand-off was begun, if it got that far
        try:
            winner = self._race(blocked, t0)
            if winner is not None:
                handing_over = winner.instance_id
                host, port = winner.endpoint
                self.snap.publish(winner.instance_id, winner.seq, host, port)  # the hand-off
        except BaseException:
            deadline = time.monotonic() + self.policy.rollback  # one budget for all of it
            self._kill_children()
            self._rollback(handing_over, deadline)
            raise
        self._kill_children()
        keep = winner.instance_id if winner else None
        left = self._destroy_all(keep, self.policy.confirm_timeout)
        for a in self.attempts.values():
            if a is winner:
                continue
            if a.phase == "healthy":
                a.reason, a.detail = None, "healthy, but not kept"
            elif a.state == "bound" and a.phase != "failed":
                a.reason, a.detail = None, a.detail or "cancelled"
        ordered = [self.attempts[k] for k in sorted(self.attempts)]
        return Result(
            winner=winner,
            attempts=ordered,
            seconds=time.monotonic() - t0,
            left=left,
            unseen=[a for a in ordered if a.state == "open"],
        )

    def _published(self) -> str:
        """Is a winner in the snapshot on disk? "yes", "no", or "unknown" if the disk
        cannot tell us. Unless unknown, memory is made to agree with the disk: an
        interrupt can land between a rename and our noticing it."""
        try:
            loaded = Snapshot.load(self.snap.path.parent) if self.snap.path else None
        except Exception as exc:
            log(f"WARNING: cannot read the snapshot back: {exc}")
            return "unknown"
        if loaded is None or loaded.run != self.snap.run:
            return "unknown"
        self.snap.data = loaded.data
        return "yes" if loaded.winner else "no"

    def _kill_children(self) -> None:
        try:
            self.prober.kill_all()
        except Exception as exc:
            log(f"WARNING: stopping child processes: {exc!r}")
        for a in self.attempts.values():
            a.check = None

    def _note_gone(self, instance_id: int) -> None:
        self.snap.gone(instance_id)

    def _outstanding(self, keep: int | None) -> set[int]:
        ids = self.held | set(self.snap.live_ids())
        return {i for i in ids if i != keep and self.snap.status(i) != "gone"}

    def _destroy_all(self, keep: int | None, budget: float, record: bool = True) -> list[int]:
        ids = self._outstanding(keep)
        for iid in sorted(ids) if record else ():
            if self.snap.status(iid) is None:  # the snapshot never took it: once more, for `down`
                try:
                    self.snap.know(iid)
                except Exception as exc:
                    log(f"WARNING: instance {iid} could not be recorded: {exc}")
        on_gone = self._note_gone if record else None
        left = destroy_confirmed(self.vast, ids, budget, self.policy, on_gone)
        for iid in left:
            log(f"WARNING: instance {iid} is not confirmed destroyed; `gpubox down` retries")
        return left

    def _rollback(self, handing_over: int | None, deadline: float) -> None:
        """L4, L5. An attempt within the budget, not a promise. Never masks what was
        raised, except by a second signal.

        A box whose hand-off was begun is kept if the snapshot on disk has it,
        and also if the disk cannot tell us: `run` may be using it already.
        In that second case nothing is written either, since what we would
        write could erase the hand-off. `down` settles it.
        """
        published = self._published()  # also makes memory agree with the disk, if it can
        if handing_over is None:
            published = "no"  # nothing was being handed over, whatever the disk can tell us
        keep = handing_over if published != "no" else None
        if published == "yes":
            log(f"keeping instance {keep}: it was handed over before the failure")
        elif published == "unknown":
            log(f"keeping instance {keep}: cannot tell whether it was handed over; run `down`")
        try:
            budget = max(0.0, deadline - time.monotonic())
            self._destroy_all(keep, budget, record=published != "unknown")
        except Exception as exc:
            log(f"WARNING: rolling back: {exc!r}")

    # ------------------------------------------------------------- the race

    def _race(self, blocked: dict[int, dict], t0: float) -> Attempt | None:
        spec, policy = self.spec, self.policy
        deadline = t0 + policy.overall_timeout - policy.cleanup_reserve
        offers = collections.deque(
            pol.select_offers(self.vast.offers(v.offer_query(spec)), blocked, policy.max_live)
        )
        log(
            f"{len(offers)} candidate host(s) for {spec.num_gpus}x {spec.gpu} "
            f"({len(blocked)} machine(s) blocked); racing {policy.max_live}, "
            f"up to {policy.max_attempts} attempts"
        )
        image_error = None
        give_up_at = deadline
        last_listing = time.monotonic()
        strays: set[int] = set()
        while True:
            # One listing, all its pages together, gets `call_timeout`: it either
            # comes back within that or counts as failed.
            rows = _try_list(self.vast, policy)
            if rows is not None:
                last_listing = time.monotonic()
                listed = self._adopt(rows, strays)
            else:
                listed = None
                if time.monotonic() - last_listing > policy.blind_timeout:  # L7
                    raise v.VastError(0, f"no instance listing for {policy.blind_timeout:g}s")
            for a in list(self.attempts.values()):
                if a.state == "bound" and a.phase in ("booting", "probing", "gating"):
                    exc = self._advance(a)
                    if exc is not None and image_error is None:
                        # Usually the image really is missing, and waiting for
                        # the others only costs money: nothing more is rented,
                        # and they get a short while to prove it was this host.
                        image_error = exc
                        self.rent_open = False
                        give_up_at = min(give_up_at, time.monotonic() + policy.image_grace)
            self._destroy_failed(strays)
            winner = self._choose(listed) if listed is not None else None
            if winner is not None:
                if image_error is not None:
                    log(f"  ignored, since a box is healthy: {image_error}")
                return winner
            if time.monotonic() >= give_up_at:
                if image_error is not None:
                    raise image_error
                log(f"giving up: no healthy box within {policy.overall_timeout:g}s")
                return None
            if listed is not None:  # L1: never from a pass without a listing
                self._rent(offers, deadline)
            if not self._worth_waiting(offers, deadline):
                if image_error is not None:
                    raise image_error
                log("giving up: nothing left to try")
                return None
            time.sleep(policy.poll)

    def _adopt(self, rows: list[dict], strays: set[int]) -> set[int]:
        """The ids of this run's boxes in a listing. A box we did not know becomes the box
        of the open attempt its tag names, or a stray to destroy."""
        listed = set()
        for row in rows:
            tag = parse_tag(row.get("label"))
            if tag is None or tag.run != self.snap.run:
                continue
            iid = row["id"]
            listed.add(iid)
            if self.snap.status(iid) == "gone":
                # A lookup said this box did not exist, and here it is. Believe
                # a second lookup: if it exists, it is ours to destroy again.
                answered, inst = self._lookup(iid)
                if answered and inst is not None:
                    self.snap.revive(iid)
                    strays.add(iid)
                    log(f"  instance {iid} was recorded as gone but exists: destroying it")
                continue
            if iid in self.held or self.snap.status(iid) is not None:
                continue
            self.held.add(iid)
            a = self.attempts.get(tag.attempt)
            if a is not None and a.state == "open":
                self.snap.bind(a.seq, iid)  # S2
                a.state, a.instance_id, a.phase, a.t0 = "bound", iid, "booting", time.monotonic()
                a.detail += f"; found as instance {iid}"
                log(f"  {a.tag_name} its create gave no answer, but instance {iid} carries its tag")
            else:
                self.snap.know(iid)  # S2
                strays.add(iid)
                log(f"  instance {iid} carries this run's tag but no open attempt's: destroying it")
        return listed

    def _elapsed(self, a: Attempt) -> float:
        return round(time.monotonic() - a.t0, 1)

    def _fail(self, a: Attempt, reason: str | None, detail: str) -> None:
        a.phase, a.reason, a.detail = "failed", reason, detail
        a.failed_at = time.monotonic()
        log(f"  {a.tag_name} failed ({reason or 'not blamed'}): {detail}")

    def _lookup(self, instance_id: int):
        """(True, row or None) from the provider, or (False, None) if it did not answer."""
        try:
            return True, self.vast.instance(
                instance_id, timeout=self.policy.call_timeout, retries=0
            )
        except v.VastError:
            return False, None

    def _advance(self, a: Attempt) -> ConfigError | None:
        """Move one candidate on as far as it can go without waiting.

        Returns a ConfigError if its host says the image cannot be pulled.
        """
        policy, spec = self.policy, self.spec
        if a.phase == "gating":
            result = a.check.poll()
            if result is None:
                return None
            a.check = None
            ok, reason, detail, a.report = result
            a.timings["healthy"] = self._elapsed(a)
            if ok:
                a.phase = "healthy"
                log(f"  {a.tag_name} healthy after {a.timings['healthy']:.0f}s")
            else:
                self._fail(a, reason, detail)
            return None
        if a.phase == "probing":
            result = a.check.poll()
            if result is None:
                return None
            a.check = None
            ok, a.ssh_err = result
            if ok:
                a.timings["ssh"] = self._elapsed(a)
                log(f"  {a.tag_name} reachable after {a.timings['ssh']:.0f}s, running health gate")
                a.check = self.prober.start_gate(a.endpoint, spec)
                a.phase = "gating"
                return None
            a.phase = "booting"  # look at the box again before the next probe
        if self._elapsed(a) > policy.startup_timeout:
            extra = a.ssh_err or a.msg[:120] or "no status message"
            # Only the provider's own account of the box is evidence against
            # the machine; our failure to reach the API is not.
            if a.answered:
                detail = (
                    f"not reachable after {policy.startup_timeout:g}s (status={a.status}; {extra})"
                )
                self._fail(a, "unreachable", detail)
            else:
                self._fail(a, None, f"provider API gave no status for {policy.startup_timeout:g}s")
            return None
        answered, inst = self._lookup(a.instance_id)
        if not answered:
            return None
        if inst is None:
            # A lookup may lag a create. After that, it is the provider's word
            # that the box does not exist (S3), which blames nobody.
            if time.monotonic() - a.t0 > policy.settle:
                # Not before a delete has been asked for: if the lookup is
                # wrong, the box must not be left running behind a tombstone.
                _try_destroy(self.vast, a.instance_id, policy.call_timeout)
                self._note_gone(a.instance_id)
                self._fail(a, None, "the instance no longer exists")
            return None
        a.answered = True
        a.status = str(inst.get("actual_status") or "none")
        a.msg = str(inst.get("status_msg") or "")
        kind = v.classify_status(a.msg)
        if kind == "image":
            self._fail(a, None, f"image cannot be pulled: {a.msg[:200]}")
            return ConfigError(f"image {spec.image!r} cannot be pulled: {a.msg[:200]}")
        if kind == "machine":
            self._fail(a, "runtime_error", a.msg[:200])
            return None
        if a.status == "exited":
            # Our container command ends only when sshd could not be started.
            self._fail(a, "unreachable", f"container exited during bootstrap ({a.msg[:120]})")
            return None
        ep = v.endpoint_of(inst)
        if a.status != "running" or ep is None:
            return None
        a.timings.setdefault("running", self._elapsed(a))
        a.endpoint = ep
        a.check = self.prober.start_ready(ep)
        a.phase = "probing"
        return None

    def _destroy_failed(self, strays: set[int]) -> None:
        """One delete and one lookup for every box we no longer want. A failed box keeps
        its slot until the lookup says it is gone (L2)."""
        unwanted = set(strays)
        for a in self.attempts.values():
            if a.state == "bound" and a.phase == "failed":
                unwanted.add(a.instance_id)
        for iid in sorted(unwanted):
            if self.snap.status(iid) == "gone":
                strays.discard(iid)
                continue
            _try_destroy(self.vast, iid, self.policy.call_timeout)
            _confirm_gone(self.vast, iid, self.policy.call_timeout, self._note_gone)

    def _choose(self, listed: set[int]) -> Attempt | None:
        """L3: the cheapest candidate whose gate passed, that this pass's listing shows and
        a lookup says is running, with enough of its lease left."""
        healthy = [a for a in self.attempts.values() if a.phase == "healthy"]
        for a in sorted(healthy, key=lambda a: (a.dph, a.seq)):
            answered, inst = self._lookup(a.instance_id)
            if not answered:
                continue
            if inst is None:
                self._note_gone(a.instance_id)
                self._fail(a, None, "passed the gate, then the instance no longer existed")
            elif str(inst.get("actual_status")) != "running":
                self._fail(a, None, f"passed the gate, then was {inst.get('actual_status')}")
            elif a.instance_id in listed:
                if self.snap.expiry - time.time() < self.policy.min_lease:
                    raise ConfigError(
                        f"less than {self.policy.min_lease:g}s of the lease is left; "
                        "raise --ttl-min"
                    )
                return a
        return None

    def _rent(self, offers: collections.deque, deadline: float) -> None:
        spec, snap = self.spec, self.snap
        while (
            self.rent_open
            and snap.slots() < self.policy.max_live  # L2
            and len(self.attempts) < self.policy.max_attempts
            and offers
            and time.monotonic() < deadline  # L1: checked just before each request
        ):
            o = offers.popleft()
            seq = len(self.attempts) + 1
            a = Attempt(
                seq=seq,
                tag=make_tag(snap.run, seq, snap.expiry),
                machine_id=o["machine_id"],
                host_id=o.get("host_id"),
                offer_id=o["id"],
                dph=float(o["dph_total"]),
                geo=str(o.get("geolocation") or ""),
            )
            payload = v.create_payload(spec, a.tag, self.prober.pubkey, snap.expiry)
            # S1: on disk before the request. If this fails nothing is sent.
            sent = time.time()
            snap.open_attempt(
                seq,
                sent=round(sent, 3),
                tag=a.tag,
                offer=a.offer_id,
                machine=a.machine_id,
                host=a.host_id,
                price=a.dph,
            )
            self.attempts[seq] = a
            # L1: the write can be slow. If the window closed meanwhile, or the
            # time we recorded is no longer close to the time of sending (it is
            # what `down --abandon-unseen` counts from), nothing is sent.
            if time.monotonic() >= deadline or time.time() - sent > self.policy.settle / 4:
                snap.refuse(seq)
                a.state, a.detail = "refused", "not sent: recording the attempt took too long"
                log(f"  {a.tag_name} {a.detail}")
                offers.appendleft(o)  # the offer was not used; a later pass may try it again
                return
            try:
                iid = self.vast.create(a.offer_id, payload)
            except v.CreateRefused as exc:
                # Usually a stale offer (rented between search and create).
                snap.refuse(seq)
                a.state, a.detail = "refused", f"create refused: {exc}"
                log(f"  {a.tag_name} {a.detail}")
                continue
            except v.CreateUnknown as exc:
                # Stays open: a box may exist. It holds its slot (L2), and the
                # listing is where it would turn up.
                a.detail = f"create outcome unknown: {exc}"
                log(f"  {a.tag_name} {a.detail}")
                continue
            self.held.add(iid)
            a.instance_id = iid
            snap.bind(seq, iid)  # S2: if this raises, the rollback destroys what `held` names
            a.state, a.phase, a.t0 = "bound", "booting", time.monotonic()
            log(f"  {a.tag_name} rented instance {iid}: ${a.dph:.2f}/hr, host {a.host_id}, {a.geo}")

    def _worth_waiting(self, offers: collections.deque, deadline: float) -> bool:
        """Is anything still going on that could end in a healthy box?

        Slots held for good (a create that never answered, a delete that never
        shows) are not waited for: the launch ends without a winner instead
        of sitting on them until the deadline.
        """
        policy, now, wall = self.policy, time.monotonic(), time.time()
        for a in self.attempts.values():
            if a.state == "bound" and a.phase in ("booting", "probing", "gating", "healthy"):
                return True
            if a.state == "open":
                sent = self.snap.attempts()[a.seq]["sent"]
                if wall - sent < policy.settle:
                    return True  # its box may still turn up
            if a.state == "bound" and a.phase == "failed":
                gone = self.snap.status(a.instance_id) == "gone"
                if not gone and now - a.failed_at < policy.confirm_timeout:
                    return True  # its delete may still show, and free its slot
        return bool(
            self.rent_open
            and offers
            and self.snap.slots() < policy.max_live
            and len(self.attempts) < policy.max_attempts
            and now < deadline
        )


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


# -------------------------------------------------------------------- teardown


def down(vast, snap: Snapshot, policy: Policy, *, abandon_unseen: bool = False) -> int:
    """Destroy everything of a run. Returns how many things are unresolved.

    Chooses every id in the snapshot that is not gone, plus every listed box
    whose tag carries the run id. T1: returns 0, and closes the run, only if
    no id is left live, a complete listing taken after the last delete shows
    no box with the run's tag, and no attempt is open. T2: with
    `abandon_unseen`, an open attempt stops counting once such a listing was
    taken at least `policy.settle` after the attempt was sent.

    With an ephemeral snapshot this is teardown by tag alone (T4).
    """
    # A closed run is looked at again like any other: a box whose create gave
    # no answer can show up after the run was closed.
    dirty = 0  # things that could not be recorded: the run cannot be closed

    def record(what, fn, *args) -> None:
        nonlocal dirty
        try:
            fn(*args)
        except Exception as exc:
            dirty += 1
            log(f"WARNING: could not record {what}: {exc}")

    record("that the run is closing", snap.set_closing)
    held = set(snap.live_ids())
    # Known ids first: nothing below may prevent these, a failing listing least of all.
    destroy_confirmed(vast, held, policy.confirm_timeout, policy, snap.gone)

    not_before = 0.0
    opens = snap.open_attempts()
    if abandon_unseen and opens:
        last = max(a["sent"] for a in opens.values())
        not_before = min(last, time.time()) + policy.settle  # a clock set back must not hold us
    deadline = time.monotonic() + policy.confirm_timeout + max(0.0, not_before - time.time())
    verified = False
    while True:
        asked_at = time.time()
        rows = _try_list(vast, policy)
        found = None
        if rows is not None:
            found = set()
            for row in rows:
                tag = parse_tag(row.get("label"))
                if tag is None or tag.run != snap.run:
                    continue
                iid = row["id"]
                found.add(iid)
                if snap.status(iid) == "gone":
                    # Either this listing is behind, or the lookup that said
                    # "gone" was wrong. A lookup now decides.
                    try:
                        exists = vast.instance(iid, timeout=policy.call_timeout, retries=0)
                    except v.VastError:
                        continue
                    if exists is not None:
                        log(f"instance {iid} was recorded as gone but exists")
                        held.add(iid)
                        record(f"instance {iid}", snap.revive, iid)
                    continue
                if iid in held or snap.status(iid) is not None:
                    continue
                held.add(iid)  # S2: recorded before any delete is sent for it
                if tag.attempt in snap.open_attempts():
                    log(f"found instance {iid}: attempt {tag.attempt}, whose create gave no answer")
                    record(f"instance {iid}", snap.bind, tag.attempt, iid)
                else:
                    log(f"found instance {iid} by its tag")
                    record(f"instance {iid}", snap.know, iid)
        pending = {i for i in held if snap.status(i) != "gone"}
        if pending:
            budget = min(policy.confirm_timeout, max(1.0, deadline - time.monotonic()))
            destroy_confirmed(vast, pending, budget, policy, snap.gone)
        elif found is not None and not found and asked_at >= not_before:
            verified = True  # a complete listing, after the last delete, with none of ours
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(policy.confirm_poll)

    left = sorted(i for i in held if snap.status(i) != "gone")
    for iid in sorted(held - set(left)):
        log(f"destroyed instance {iid}")
    for iid in left:
        log(f"ERROR: instance {iid} is not confirmed destroyed")
    if not verified:
        log(
            "ERROR: no complete listing, taken after the last delete, shows that no box "
            "carries this run's tag"
        )
    for n in sorted(snap.open_attempts()):
        if verified and abandon_unseen:
            log(
                f"attempt {n}: its create got no answer, and no box with its tag is visible "
                f"{policy.settle:g}s later. Not proof that nothing was rented: it is left to "
                "the box's own timer and the reaper"
            )
            record(f"attempt {n} as unseen", snap.unseen, n)
        else:
            log(
                f"attempt {n}: its create got no answer and no box with its tag is visible, "
                "which does not prove that nothing was rented; unresolved (check the provider, "
                "or run `down --abandon-unseen`)"
            )
    unresolved = len(left) + len(snap.open_attempts()) + (not verified) + dirty
    if not unresolved:
        try:
            snap.close()
        except Exception as exc:
            log(f"WARNING: could not record that the run is closed: {exc}")
            return 1
    return unresolved


# ---------------------------------------------------------------------- reaper


def reap(
    vast,
    policy: Policy,
    *,
    namespace: str | None = None,
    max_age_min: float | None = None,
    destroy: bool = False,
    now=None,
    sleep=time.sleep,
) -> int:
    """Find instances that outlived their lease. Returns how many could not be dealt with.

    With `namespace`: only instances whose tag parses to that namespace are
    considered, and the expired ones chosen (for an account shared with other
    work). Without: every instance of the account is considered, which the
    caller must have declared disposable and checked the account of (R1), and
    one is chosen if its tag expired, if it is older than `max_age_min` by
    the provider's clock (R2), or if it carries no valid tag in two listings
    taken `policy.reap_recheck` apart (R3).

    R4: without `destroy` this only reports. R5: a listing that cannot be
    read raises; a delete that is not confirmed is counted.
    """
    exclusive = namespace is None
    if exclusive and not _usable_time(max_age_min):
        raise ConfigError("an exclusive reap needs --max-age-min")
    now = time.time() if now is None else now
    chosen: dict[int, str] = {}
    suspects: dict[int, str] = {}
    failed = 0
    for row in vast.instances():
        iid, label = row["id"], row.get("label")
        tag = parse_tag(label)
        if tag is None and not exclusive:
            continue
        if tag is not None and not exclusive and tag.namespace != namespace:
            continue
        started = row.get("start_date")
        age = (now - started) / 60 if _usable_time(started) else None
        what = f"instance {iid} ({label!r}, {row.get('actual_status')})"
        if tag is not None and tag.expiry + policy.expiry_margin < now:
            chosen[iid] = f"{what}: lease ended {(now - tag.expiry) / 60:.0f} min ago"
        elif exclusive and age is not None and age > max_age_min:
            chosen[iid] = f"{what}: {age:.0f} min old, over the maximum"
        elif tag is None:
            suspects[iid] = what
        elif exclusive and age is None:
            log(f"WARNING: {what}: no start time, so the age rule cannot apply to it")
        else:
            log(f"{what}: lease ends in {(tag.expiry - now) / 60:.0f} min")
    if suspects:
        # R3: a box seconds old may not show its label yet, and deleting it
        # would break a launch in progress. Look twice.
        sleep(policy.reap_recheck)
        try:
            again = {row["id"]: row for row in vast.instances()}
        except v.VastError as exc:
            # R5: a failure. But what the first look chose is still destroyed.
            log(f"ERROR: the second listing failed, so untagged instances were not judged: {exc}")
            again, failed = None, 1
        for iid, what in suspects.items() if again is not None else ():
            if iid not in again:
                log(f"{what}: gone at the second look")
                continue
            tag = parse_tag(again[iid].get("label"))
            if tag is None:
                chosen[iid] = f"{what}: no valid tag in two listings"
            elif tag.expiry + policy.expiry_margin < now:
                chosen[iid] = f"{what}: tagged at the second look, with a lease that has ended"
            else:
                log(f"{what}: carries a tag at the second look; left alone")
    for iid in sorted(chosen):
        log(f"{chosen[iid]} -> {'destroy' if destroy else 'would destroy (report only)'}")
    if not destroy or not chosen:
        return failed
    left = destroy_confirmed(vast, chosen, policy.confirm_timeout, policy)
    for iid in left:
        log(f"ERROR: instance {iid} is not confirmed destroyed")
    return failed + len(left)
