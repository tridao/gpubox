"""Rent one verified box: a small race, a health gate, and teardown by tag.

One thread. The race is a loop: each pass looks each candidate up by id, polls
the child processes (SSH probes and health gates), and then decides what to
destroy, whether there is a winner, and whether to rent more.

What makes a box ours is its tag, set by the create request: it names the run
and the time the lease ends. Teardown is "destroy the ids we were given, and
everything the listing shows with the run's tag". What bounds a leak is the
lease: the box's own timer and the reaper both act on it. `up` and `down` are
best effort on top of that, and say so when they could not finish.

A box counts as gone only when a lookup of its id says it does not exist.

Assumptions (DESIGN.md, section 3, has the reasons and what each costs):

  A1  Labels are trustworthy: nothing renames a box. So a tag finds every box
      of a run, and the reaper needs to look at nothing but tags.
  A2  The lease bounds a leak. A box that `up` or `down` missed runs until
      its lease ends; it is not chased with extra bookkeeping.
  A3  The state directory is sound: written by one process, under a lock.
  A4  The listing shows a box once its create has returned (`down` lists
      twice, a few seconds apart, to cover a short lag).
  A5  The provider does not give an instance id to a second box: `down`
      deletes the ids in the state file without looking at their tags.
  A6  Whoever holds a token for the reputation store is trusted.

Not assumed: that a create is answered (an unanswered one keeps its slot),
that a delete worked (each is confirmed by a lookup), that the provider's
answers are well formed, or that one host is right about the image.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field

from . import policy as pol
from . import vast as v
from .gate import GATE_VERSION
from .state import State, make_tag, parse_tag


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
    # The caller's own checks, run on the box after the built-in gate passes (gate.read_hook).
    gate_script: str | None = None
    gate_timeout: float = 600
    ttl_min: int = 90
    # Hosts in mainland China took a median of 250 s to load the CI image against 65 s
    # elsewhere, and 2 of 6 never did (DESIGN.md, section 10).
    exclude_countries: tuple[str, ...] = ("CN",)


@dataclass(frozen=True)
class Policy:
    spare: int = 1  # candidates raced beyond the one we keep
    max_attempts: int = 6
    startup_timeout: float = 300
    # A box that is not running yet and whose status has not changed for this long is
    # given up on. Pulls that finished went silent for at most 166 s; the ones that never
    # finished, for 600 s and more (DESIGN.md, section 10).
    stall_timeout: float = 300
    overall_timeout: float = 900
    cleanup_reserve: float = (
        60  # the race ends, and renting stops, this long before overall_timeout
    )
    poll: float = 5
    image_errors: int = 2  # hosts that must say the image cannot be pulled before we believe it
    settle: float = 60  # how long a lookup may say "no such instance" right after a create
    min_lease: float = 600  # a winner is handed over with at least this much lease left
    rollback: float = 10  # what a cancelled CI step gets before it is killed
    confirm_timeout: float = 60  # how long a teardown keeps trying
    confirm_poll: float = 2
    round: float = 8  # one delete-and-lookup round of a teardown, before it lists again
    recheck: float = 3  # between the two listings that must both show nothing of the run
    call_timeout: float = 10  # one provider call inside the loop or a teardown
    expiry_margin: float = 60  # clock skew allowed before an expired tag is acted on

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
    # unanswered: the create got no usable answer, so a box may exist that we have no id for
    state: str = "unanswered"  # unanswered | refused | rented
    instance_id: int | None = None
    phase: str = ""  # of a rented box: booting | probing | gating | checking | healthy | failed
    gone: bool = False  # a lookup said its box does not exist
    endpoint: tuple[str, int] | None = None
    reason: str | None = None  # a policy.REASONS code when the machine is at fault
    detail: str = ""  # free text; stays in logs, never in observations
    timings: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    t0: float = 0.0  # monotonic; when we learned its id
    failed_at: float = 0.0  # monotonic
    changed_at: float = 0.0  # monotonic; when its status or status message last changed
    check: object = None  # the child process it is waiting for, if any
    answered: bool = False  # did the provider ever tell us anything about this box?
    status: str = ""
    msg: str = ""
    ssh_err: str = ""

    @property
    def tag_name(self) -> str:
        return f"[m{self.machine_id} a{self.seq}]"

    @property
    def holds_slot(self) -> bool:
        """A box may exist for this attempt, so nothing is rented in its place."""
        return self.state == "unanswered" or (self.state == "rented" and not self.gone)


@dataclass
class Result:
    winner: Attempt | None
    attempts: list[Attempt]
    seconds: float
    left: list[int] = field(
        default_factory=list
    )  # ids, other than the winner's, not confirmed gone
    unanswered: list[Attempt] = field(default_factory=list)  # creates that got no answer


# ------------------------------------------------------------------ destroying


def _try_destroy(vast, instance_id: int, timeout: float) -> None:
    try:
        vast.destroy(instance_id, timeout=timeout, retries=0)
    except v.VastError as exc:
        log(f"WARNING: destroying instance {instance_id}: {exc}")
    except Exception as exc:  # one id's trouble must not stop the others
        log(f"WARNING: destroying instance {instance_id}: {exc!r}")


def _is_gone(vast, instance_id: int, timeout: float) -> bool:
    """True only if a lookup of this id says it does not exist."""
    try:
        return vast.instance(instance_id, timeout=timeout, retries=0) is None
    except v.VastError as exc:
        log(f"WARNING: checking instance {instance_id}: {exc}")
    except Exception as exc:
        log(f"WARNING: checking instance {instance_id}: {exc!r}")
    return False


def destroy_confirmed(vast, ids, budget: float, policy: Policy) -> list[int]:
    """Destroy these instances and confirm each by id. Returns the ids not confirmed gone.

    One delete goes to every id before any is waited for. Then each is looked
    up, and the delete repeated for those still there, until all are gone or
    the budget ends. An error on one id does not stop the others. No call is
    asked to take longer than what is left of the budget, and after the first
    round of deletes none is started once nothing is left. (These are socket
    timeouts: the budget is what we ask for, not a bound on the wall clock.)
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
            if _is_gone(vast, iid, share(1)):
                left.remove(iid)
        if not left or spent():
            return left
        time.sleep(min(policy.confirm_poll, max(0.0, deadline - time.monotonic())))
        for k, iid in enumerate(left):
            if spent():
                return left
            _try_destroy(vast, iid, share(len(left) - k))


def tagged(vast, run_id: str, policy: Policy, timeout: float | None = None) -> set[int] | None:
    """The ids of the boxes the listing shows with this run's tag, or None if it could not
    be listed. Not retried here: callers loop."""
    timeout = policy.call_timeout if timeout is None else min(policy.call_timeout, timeout)
    try:
        rows = vast.instances(timeout=timeout, retries=0)
    except v.VastError as exc:
        log(f"WARNING: listing instances: {exc}")
        return None
    return {r["id"] for r in rows if (t := parse_tag(r.get("label"))) and t.run == run_id}


# ---------------------------------------------------------------------- launch

# The phases of a rented box on its way to a verdict.
PENDING = ("booting", "probing", "gating", "checking")


class Launch:
    def __init__(self, vast, spec: Spec, policy: Policy, state: State, prober):
        self.vast, self.spec, self.policy = vast, spec, policy
        self.state, self.prober = state, prober
        self.attempts: dict[int, Attempt] = {}
        self.held: set[int] = set()  # every id a create gave us, on disk or not

    # ------------------------------------------------------------ the frame

    def run(self, blocked: dict[int, dict]) -> Result:
        """Race candidates; return with at most one box, published as the winner.

        Raises whatever failed, or KeyboardInterrupt on a signal, after
        trying to roll back. A winner that reached the state file is kept
        whatever was raised; everything else the run rented is destroyed.
        """
        t0 = time.monotonic()
        try:
            winner = self._race(blocked, t0)
            if winner is not None:
                self.state.publish({
                    "id": winner.instance_id, "attempt": winner.seq,
                    "host": winner.endpoint[0], "port": winner.endpoint[1],
                    "machine": winner.machine_id, "host_id": winner.host_id,
                })  # fmt: skip
        except BaseException:
            deadline = time.monotonic() + self.policy.rollback  # one budget for all of it
            self._kill_children()
            self._rollback(deadline)
            raise
        self._kill_children()
        keep = winner.instance_id if winner else None
        left = self._sweep(keep, self.policy.confirm_timeout)
        for a in self.attempts.values():
            if a is winner:
                continue
            if a.phase == "healthy":
                a.reason, a.detail = None, "healthy, but not kept"
            elif a.state == "rented" and a.phase != "failed":
                a.reason, a.detail = None, a.detail or "cancelled"
        ordered = [self.attempts[k] for k in sorted(self.attempts)]
        return Result(
            winner=winner,
            attempts=ordered,
            seconds=time.monotonic() - t0,
            left=left,
            unanswered=[a for a in ordered if a.state == "unanswered"],
        )

    def _kill_children(self) -> None:
        try:
            self.prober.kill_all()
        except Exception as exc:
            log(f"WARNING: stopping child processes: {exc!r}")
        for a in self.attempts.values():
            a.check = None

    def _sweep(self, keep: int | None, budget: float) -> list[int]:
        """Destroy every box of the run except `keep`: the ids we hold, then whatever the
        listing shows with the run's tag (a box whose create gave no answer)."""
        deadline = time.monotonic() + budget
        remaining = lambda: max(0.0, deadline - time.monotonic())
        # Half the time for the ids we hold, so that one that will not go away
        # leaves room for the listing and for what it finds.
        left = set(destroy_confirmed(self.vast, self.held - {keep}, budget / 2, self.policy))
        found = tagged(self.vast, self.state.run, self.policy, remaining()) or set()
        extra = found - self.held  # the winner is among those we hold
        for iid in sorted(extra):
            log(f"  instance {iid} carries this run's tag and was not known: destroying it")
        left = sorted(destroy_confirmed(self.vast, left | extra, remaining(), self.policy))
        for iid in left:
            log(f"WARNING: instance {iid} is not confirmed destroyed; `gpubox down` retries")
        return left

    def _rollback(self, deadline: float) -> None:
        """An attempt within the budget, not a promise. Never masks what was raised, except
        by a second signal. A winner that is in the state file on disk is kept."""
        keep = None
        try:
            on_disk = State.load(self.state.path.parent) if self.state.path else None
            if on_disk is not None and on_disk.run == self.state.run and on_disk.winner:
                keep = on_disk.winner["id"]
                self.state.data = on_disk.data
                log(f"keeping instance {keep}: it was handed over before the failure")
        except Exception as exc:
            log(f"WARNING: cannot read the state file back: {exc}")
        try:
            self._sweep(keep, max(0.0, deadline - time.monotonic()))
        except Exception as exc:
            log(f"WARNING: rolling back: {exc!r}")

    # ------------------------------------------------------------- the race

    def _race(self, blocked: dict[int, dict], t0: float) -> Attempt | None:
        spec, policy = self.spec, self.policy
        deadline = t0 + policy.overall_timeout - policy.cleanup_reserve
        found = self.vast.offers(v.offer_query(spec))
        # The search is asked for the price cap and the order; neither is taken
        # on trust, since what comes back is what gets rented.
        cap = spec.max_dph_per_gpu * spec.num_gpus + 1e-9
        found = sorted((o for o in found if o["dph_total"] <= cap), key=lambda o: o["dph_total"])
        allowed = v.in_allowed_country(found, spec.exclude_countries)
        offers = collections.deque(pol.select_offers(allowed, blocked, policy.max_live))
        log(
            f"{len(offers)} candidate host(s) for {spec.num_gpus}x {spec.gpu} "
            f"({len(blocked)} machine(s) blocked, {len(found) - len(allowed)} offer(s) in "
            f"excluded countries); racing {policy.max_live}, "
            f"up to {policy.max_attempts} attempts"
        )
        image_errors: dict = {}  # host -> what it said
        image_said = None  # the same, from any box, also one whose host we cannot name
        while True:
            for a in list(self.attempts.values()):
                if a.state == "rented" and a.phase in PENDING:
                    said = self._advance(a)
                    if said is not None:
                        image_said = said
                        if a.host_id is not None:  # two boxes of one host are one witness
                            image_errors[a.host_id] = said
            self._destroy_failed()
            if len(image_errors) >= policy.image_errors:
                # One host can be wrong about an image (a broken mirror, stale
                # credentials). Several saying the same are not.
                raise ConfigError(image_said)
            winner = self._choose()
            if winner is not None:
                return winner
            out_of_time = time.monotonic() >= deadline
            if not out_of_time:
                self._rent(offers, deadline)
            if out_of_time or not self._worth_waiting(offers, deadline):
                if image_said:
                    raise ConfigError(image_said)
                log(
                    f"giving up: no healthy box within {policy.overall_timeout:g}s"
                    if out_of_time
                    else "giving up: nothing left to try"
                )
                return None
            time.sleep(policy.poll)

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

    def _advance(self, a: Attempt) -> str | None:
        """Move one candidate on as far as it can go without waiting.

        Returns a message if the box shows that the image itself is the
        problem: its host cannot pull it, or it runs but refuses our key.
        """
        policy, spec = self.policy, self.spec
        if a.phase == "checking":
            result = a.check.poll()
            if result is None:
                return None
            a.check = None
            ok, reason, detail, _ = result
            if ok:
                self._healthy(a)
            else:
                self._fail(a, reason, detail)
            return None
        if a.phase == "gating":
            result = a.check.poll()
            if result is None:
                return None
            a.check = None
            ok, reason, detail, a.report = result
            if not ok:
                self._fail(a, reason, detail)
            elif spec.gate_script is not None:
                a.timings["gate"] = self._elapsed(a)
                log(f"  {a.tag_name} passed the health gate, running the gate script")
                a.check = self.prober.start_hook(a.endpoint, spec)
                a.phase = "checking"
            else:
                self._healthy(a)
            return None
        if a.phase == "probing":
            result = a.check.poll()
            if result is None:
                return None
            a.check = None
            ok, a.ssh_err = result
            if not ok and "Permission denied" in a.ssh_err:
                # An sshd answered and refused the run's key. Ours would not:
                # it is started with that key. So the image runs something
                # else in place of our start-up command, and waiting out the
                # start-up timeout (ten minutes per box, when this was seen
                # live) cannot help. Handled like an image that cannot be
                # pulled: this box fails, and a second host settles it.
                said = (
                    f"image {spec.image!r} is running, but its sshd refuses the run's key: it "
                    "does not seem to run gpubox's start-up command (an ENTRYPOINT of its own?)"
                )
                self._fail(a, None, said)
                return said
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
            # A lookup may lag a create. After that, the box does not exist.
            if time.monotonic() - a.t0 > policy.settle:
                a.gone = True
                self._fail(a, None, "the instance no longer exists")
            return None
        status = str(inst.get("actual_status") or "none")
        msg = str(inst.get("status_msg") or "")
        if not a.answered or (status, msg) != (a.status, a.msg):
            a.changed_at = time.monotonic()
        a.answered, a.status, a.msg = True, status, msg
        kind = v.classify_status(a.msg)
        if kind == "image":
            said = f"image {spec.image!r} cannot be pulled: {a.msg[:200]}"
            self._fail(a, None, said)
            return said
        if kind == "machine":
            self._fail(a, "runtime_error", a.msg[:200])
            return None
        if a.status == "exited":
            # Our container command ends only when sshd could not be started.
            self._fail(a, "unreachable", f"container exited during bootstrap ({a.msg[:120]})")
            return None
        if a.status != "running":
            quiet = time.monotonic() - a.changed_at
            if quiet > policy.stall_timeout:
                # Still loading, and the provider has reported nothing new for a
                # long while: a pull that is stuck, or too slow to wait for.
                self._fail(
                    a,
                    "unreachable",
                    f"stalled: status {a.status} unchanged for {quiet:.0f}s ({a.msg[:120]})",
                )
            return None
        ep = v.endpoint_of(inst)
        if ep is None:
            return None
        a.timings.setdefault("running", self._elapsed(a))
        a.endpoint = ep
        a.check = self.prober.start_ready(ep)
        a.phase = "probing"
        return None

    def _healthy(self, a: Attempt) -> None:
        a.phase = "healthy"
        a.timings["healthy"] = self._elapsed(a)
        log(f"  {a.tag_name} healthy after {a.timings['healthy']:.0f}s")

    def _destroy_failed(self) -> None:
        """One delete and one lookup for every failed box. It keeps its slot until the
        lookup says it is gone, so there are never more billable boxes than the cap."""
        for a in self.attempts.values():
            if a.state == "rented" and a.phase == "failed" and not a.gone:
                _try_destroy(self.vast, a.instance_id, self.policy.call_timeout)
                a.gone = _is_gone(self.vast, a.instance_id, self.policy.call_timeout)

    def _choose(self) -> Attempt | None:
        """The cheapest candidate whose gate passed and that a lookup says is still running,
        with enough of its lease left."""
        healthy = [a for a in self.attempts.values() if a.phase == "healthy"]
        for a in sorted(healthy, key=lambda a: (a.dph, a.seq)):
            answered, inst = self._lookup(a.instance_id)
            if not answered:
                continue
            if inst is None:
                a.gone = True
                self._fail(a, None, "passed the gate, then the instance no longer existed")
            elif str(inst.get("actual_status")) != "running":
                self._fail(a, None, f"passed the gate, then was {inst.get('actual_status')}")
            else:
                if self.state.expiry - time.time() < self.policy.min_lease:
                    raise ConfigError(
                        f"less than {self.policy.min_lease:g}s of the lease is left; "
                        "raise --ttl-min"
                    )
                return a
        return None

    def _rent(self, offers: collections.deque, deadline: float) -> None:
        spec, state = self.spec, self.state
        while (
            sum(a.holds_slot for a in self.attempts.values()) < self.policy.max_live
            and self._sent() < self.policy.max_attempts
            and offers
            and time.monotonic() < deadline
        ):
            o = offers.popleft()
            seq = len(self.attempts) + 1
            a = Attempt(
                seq=seq,
                tag=make_tag(state.run, seq, state.expiry),
                machine_id=o["machine_id"],
                host_id=o.get("host_id"),
                offer_id=o["id"],
                dph=float(o["dph_total"]),
                geo=str(o.get("geolocation") or ""),
            )
            payload = v.create_payload(spec, a.tag, self.prober.pubkey, state.expiry)
            self.attempts[seq] = a  # as unanswered: if the request is interrupted, it stays so
            try:
                iid = self.vast.create(a.offer_id, payload)
            except v.CreateRefused as exc:
                # Usually a stale offer (rented between search and create).
                a.state, a.detail = "refused", f"create refused: {exc}"
                log(f"  {a.tag_name} {a.detail}")
                continue
            except v.CreateUnknown as exc:
                # A box may exist. It keeps its slot; the sweep at the end looks
                # for its tag, and its lease bounds it if that does not find it.
                a.detail = f"create outcome unknown: {exc}"
                log(f"  {a.tag_name} {a.detail}")
                continue
            self.held.add(iid)
            a.instance_id, a.state, a.phase = iid, "rented", "booting"
            a.t0 = a.changed_at = time.monotonic()
            state.add_id(iid)  # if this raises, the rollback destroys what `held` names
            log(f"  {a.tag_name} rented instance {iid}: ${a.dph:.2f}/hr, host {a.host_id}, {a.geo}")

    def _sent(self) -> int:
        """Creates that count against `max_attempts`: those that may have rented a box. An
        offer that was already gone ("refused") cost nothing, and a burst of them must not
        end the launch; the list of offers is what bounds those."""
        return sum(a.state != "refused" for a in self.attempts.values())

    def _worth_waiting(self, offers: collections.deque, deadline: float) -> bool:
        """Is anything still going on that could end in a healthy box?

        Slots held for good (a create that never answered, a delete that never
        shows) are not waited for: the launch ends without a winner instead
        of sitting on them until the deadline.
        """
        policy, now = self.policy, time.monotonic()
        for a in self.attempts.values():
            if a.state != "rented":
                continue
            if a.phase in PENDING or a.phase == "healthy":
                return True
            if a.phase == "failed" and not a.gone and now - a.failed_at < policy.confirm_timeout:
                return True  # its delete may still show, and free its slot
        return bool(
            offers
            and sum(a.holds_slot for a in self.attempts.values()) < policy.max_live
            and self._sent() < policy.max_attempts
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


def down(vast, state: State, policy: Policy) -> list[int] | None:
    """Destroy everything of a run.

    Rounds, for up to `policy.confirm_timeout`: one short delete-and-lookup
    round for everything known so far, then a listing, whose boxes with the
    run's tag join the next round. So a box that will not go away cannot keep
    another, found later, from being deleted. The run is gone when two
    listings in a row, `policy.recheck` apart, show nothing with its tag and
    nothing is left to delete.

    Returns [] if the run is gone (and closes it); otherwise the ids still to
    destroy, or None if there are none but the provider could not be listed
    twice to make sure.
    """
    try:
        state.set_closing()
    except OSError as exc:
        log(f"WARNING: could not record that the run is closing: {exc}")
    pending = set(state.ids)
    deadline = time.monotonic() + policy.confirm_timeout
    clean = 0  # listings in a row that showed nothing of the run, with nothing pending
    while True:
        if pending:
            left = set(destroy_confirmed(vast, pending, policy.round, policy))
            for iid in sorted(pending - left):
                log(f"instance {iid}: gone")
            pending = left
        found = tagged(vast, state.run, policy)
        new = (found or set()) - pending  # these have not been sent a delete yet
        if found is None:
            clean = 0
        elif found or pending:
            # Also an id we just confirmed gone, if the listing still shows
            # it: deleting it again costs nothing, and a listing that shows
            # the run's tag is not a clean one.
            for iid in sorted(found - pending):
                log(f"instance {iid} is listed with the run's tag")
            pending |= found
            clean = 0
        else:
            clean += 1
            if clean >= 2:
                break
        # At the deadline, stop; but a box only just found gets its round first
        # (two rounds past the deadline is the end, whatever the listing says).
        late = time.monotonic() - deadline
        if late >= 0 and (not new or late >= 2 * policy.round):
            break
        time.sleep(policy.recheck if not pending else policy.confirm_poll)
    if clean >= 2:
        try:
            state.close()
        except OSError as exc:
            log(f"WARNING: could not record that the run is closed: {exc}")
        return []
    return sorted(pending) if pending else None


def explain_leftovers(state: State, left: list[int] | None) -> None:
    """Tell the user what `down` could not finish, and what to do about it."""
    when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(state.expiry)) if state.expiry else None
    if left:
        log(f"::error::not confirmed destroyed: instance(s) {', '.join(map(str, left))}")
    else:
        log("::error::the provider could not be listed, so boxes of this run may remain")
    log(f"  Run this again:   gpubox down --run-id {state.run}")
    log(
        "  Or destroy them at https://cloud.vast.ai/instances/ (their labels start with the run id)"
    )
    if when:
        log(f"  If nothing is done, each box destroys itself when its lease ends, at {when},")
        log("  and the reaper removes whatever is left after that.")


# ---------------------------------------------------------------------- reaper


def reap(vast, policy: Policy, namespace: str, *, destroy: bool = False, now=None) -> int:
    """Destroy boxes of a namespace whose lease has ended. Returns how many could not be.

    A box is chosen if its label is a tag of this namespace and the expiry in
    it passed more than `policy.expiry_margin` ago. Nothing else on the
    account is looked at. Without `destroy` this only reports. A listing that
    cannot be read raises.
    """
    now = time.time() if now is None else now
    chosen = []
    for row in vast.instances():
        tag = parse_tag(row.get("label"))
        if tag is None or tag.namespace != namespace:
            continue
        what = f"instance {row['id']} ({row.get('label')}, {row.get('actual_status')})"
        if tag.expiry + policy.expiry_margin < now:
            chosen.append(row["id"])
            act = "destroy" if destroy else "would destroy (report only)"
            log(f"{what}: lease ended {(now - tag.expiry) / 60:.0f} min ago -> {act}")
        else:
            log(f"{what}: lease ends in {(tag.expiry - now) / 60:.0f} min")
    if not destroy or not chosen:
        return 0
    left = destroy_confirmed(vast, chosen, policy.confirm_timeout, policy)
    for iid in left:
        log(f"ERROR: instance {iid} is not confirmed destroyed")
    return len(left)


# ----------------------------------------------------------------------- renew


def renew(vast, state: State, prober, expiry: int) -> None:
    """Move the end of the winner's lease to `expiry`.

    The box's own timer reads the expiry from a file on the box; the reaper
    reads it from the tag. The file is written first, so that if the relabel
    is refused the reaper still acts on the old expiry: a refused renewal
    can end the box early, not late. A relabel whose answer is lost, or a
    state file that cannot be written afterwards, leaves the box renewed and
    this directory showing the old expiry; running `renew` again settles it.
    """
    w = state.winner
    if not w:
        raise ConfigError("this run has no box to renew")
    v.check_expiry(expiry)
    r = prober.run((w["host"], w["port"]), v.set_expiry_command(expiry))
    if r != 0:
        raise v.VastError(0, f"could not set the expiry on the box (ssh exit {r}); nothing changed")
    try:
        vast.relabel(w["id"], make_tag(state.run, w["attempt"], expiry))
        state.set_expiry(expiry)
    except Exception:
        log(
            "WARNING: the renewal is half done. The box's own timer has the new expiry; its "
            "tag, which the reaper goes by, may still have the old one, and this directory "
            "still shows the old one. Run `gpubox renew` again."
        )
        raise
