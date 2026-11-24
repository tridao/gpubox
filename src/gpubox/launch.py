"""Rent verified boxes: a small race, a health gate, and teardown by tag.

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
import math
import re
import time
from dataclasses import dataclass, field

from . import policy as pol
from . import provider as prov
from .errors import ConfigError
from . import vast as v
from .gate import GATE_VERSION
from .state import State, make_tag, parse_tag


_sink = None  # where log() sends what it says; None: stdout (api.set_log)


def log(msg: str) -> None:
    """Say something. Never raises: a closed pipe must not stop a teardown."""
    try:
        if _sink is not None:
            _sink(msg)
        else:
            print(msg, flush=True)
    except Exception:
        pass


@dataclass(frozen=True)
class Spec:
    image: str
    # Any of these models (Vast's names), cheapest offer first. Empty: any model (Slurm,
    # where the cluster decides; the gate then does not check the model).
    gpus: tuple[str, ...]
    max_dph_per_gpu: float
    num_gpus: int = 1
    capability: tuple[int, int] | None = None  # None: do not check
    min_vram_mib: int = 0
    # The host driver's newest CUDA (Vast's cuda_max_good). A CUDA 13.x image runs on any
    # 13.x driver and not on a 12.x one (measured on Vast, 2026-11).
    min_cuda: float = 13.0
    disk_gb: int = 40
    min_reliability: float = 0.98
    min_cpus: int | None = None  # for the whole box; None: max(16, 8 per GPU)
    min_ram_gb: int | None = None  # for the whole box; None: 32 per GPU
    min_inet_down_mbps: int = 200
    must_reach: tuple[str, ...] = ()
    # The caller's own checks, run on the box after the built-in gate passes (gate.read_hook).
    gate_script: str | None = None
    gate_timeout: float = 600
    ttl_min: int = 90
    # Hosts in mainland China took a median of 250 s to load the CI image against 65 s
    # elsewhere, and 2 of 6 never did (DESIGN.md, section 10).
    exclude_countries: tuple[str, ...] = ("CN",)
    # A private image: the registry's user and a (read-only) token. Vast and the
    # host's docker daemon see both; the token is kept out of the repr.
    registry_user: str | None = None
    registry_token: str | None = field(default=None, repr=False)
    # Interruptible: bid this times the offer's minimum bid. An outbid box is stopped (and
    # bills its disk until destroyed); its job dies with its container.
    interruptible: bool = False
    bid_markup: float = 1.25

    def __post_init__(self):
        gpus = (self.gpus,) if isinstance(self.gpus, str) else tuple(self.gpus)
        if not all(isinstance(g, str) and g.strip() for g in gpus):
            raise ValueError(f"gpus must be GPU model names: {self.gpus!r}")
        object.__setattr__(self, "gpus", gpus)
        if not 1 <= self.bid_markup <= 100:
            raise ValueError(f"bid_markup must be between 1 and 100: {self.bid_markup!r}")
        if (self.registry_user is None) != (self.registry_token is None):
            raise ValueError("a registry login needs both a user and a token")
        for value in (self.registry_user, self.registry_token):
            if value is not None and (not value or any(c.isspace() for c in value)):
                raise ValueError("a registry user or token cannot be empty or contain spaces")

    @property
    def gpu_text(self) -> str:
        """What was asked for, for people: "2x RTX 5090 or RTX 4090"."""
        return f"{self.num_gpus}x {' or '.join(self.gpus) or 'any GPU'}"


@dataclass(frozen=True)
class Policy:
    count: int = 1  # healthy boxes wanted, all or nothing
    spare: int = 1  # candidates raced beyond the ones we keep
    max_attempts: int | None = None  # rentals in all; None: 2 * count + 4
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
        return self.count + self.spare

    @property
    def attempts(self) -> int:
        return 2 * self.count + 4 if self.max_attempts is None else self.max_attempts


@dataclass(eq=False)  # one attempt is one object: `in` is by identity
class Attempt:
    seq: int
    tag: str
    # Vast: the offer's machine. Slurm: the node, once the job is running (None before).
    machine_id: int | str | None
    host_id: int | None
    offer_id: int
    dph: float
    geo: str = ""
    gpu: str | None = None  # the model the offer said it has
    bid: float | None = None  # interruptible: what we bid for the machine, $/hr
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
    # monotonic; what the start-up timeout counts from: t0, or the last time the box was
    # seen waiting in a queue (a queue's wait is nobody's fault)
    ready_from: float = 0.0
    failed_at: float = 0.0  # monotonic
    changed_at: float = 0.0  # monotonic; when its status or status message last changed
    check: object = None  # the child process it is waiting for, if any
    answered: bool = False  # did the provider ever tell us anything about this box?
    status: str = ""
    msg: str = ""
    ssh_err: str = ""

    @property
    def tag_name(self) -> str:
        where = f"m{self.machine_id}" if isinstance(self.machine_id, int) else self.machine_id
        if not where and self.instance_id is not None:
            where = f"job {self.instance_id}"  # Slurm, before the job has a node
        return f"[{where} a{self.seq}]" if where else f"[a{self.seq}]"

    @property
    def holds_slot(self) -> bool:
        """A box may exist for this attempt, so nothing is rented in its place."""
        return self.state == "unanswered" or (self.state == "rented" and not self.gone)


@dataclass
class Result:
    winners: list[Attempt]  # all that were asked for, or none
    attempts: list[Attempt]
    seconds: float
    left: list[int] = field(
        default_factory=list
    )  # ids, other than the winner's, not confirmed gone
    unanswered: list[Attempt] = field(default_factory=list)  # creates that got no answer

    @property
    def winner(self) -> Attempt | None:
        return self.winners[0] if self.winners else None


# ------------------------------------------------------------------ destroying


def _try_destroy(provider, instance_id: int, timeout: float) -> None:
    try:
        provider.destroy(instance_id, timeout=timeout, retries=0)
    except prov.ProviderError as exc:
        log(f"WARNING: destroying instance {instance_id}: {exc}")
    except Exception as exc:  # one id's trouble must not stop the others
        log(f"WARNING: destroying instance {instance_id}: {exc!r}")


def _is_gone(provider, instance_id: int, timeout: float) -> bool:
    """True only if a lookup of this id says it does not exist."""
    try:
        return provider.instance(instance_id, timeout=timeout, retries=0) is None
    except prov.ProviderError as exc:
        log(f"WARNING: checking instance {instance_id}: {exc}")
    except Exception as exc:
        log(f"WARNING: checking instance {instance_id}: {exc!r}")
    return False


def destroy_confirmed(provider, ids, budget: float, policy: Policy) -> list[int]:
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
        _try_destroy(provider, iid, share(len(left) - k))
    while True:
        for iid in list(left):
            if spent():
                return left
            if _is_gone(provider, iid, share(1)):
                left.remove(iid)
        if not left or spent():
            return left
        time.sleep(min(policy.confirm_poll, max(0.0, deadline - time.monotonic())))
        for k, iid in enumerate(left):
            if spent():
                return left
            _try_destroy(provider, iid, share(len(left) - k))


def tagged(provider, run_id: str, policy: Policy, timeout: float | None = None) -> set[int] | None:
    """The ids of the boxes the listing shows with this run's tag, or None if it could not
    be listed. Not retried here: callers loop."""
    timeout = policy.call_timeout if timeout is None else min(policy.call_timeout, timeout)
    try:
        rows = provider.instances(timeout=timeout, retries=0)
    except prov.ProviderError as exc:
        log(f"WARNING: listing instances: {exc}")
        return None
    return {r["id"] for r in rows if (t := parse_tag(r.get("label"))) and t.run == run_id}


# ---------------------------------------------------------------------- launch


def priced_bids(vast, spec: Spec, found: list[dict]) -> list[dict]:
    """Interruptible offers, each with what we would bid (`_bid`, at least the minimum,
    rounded up) and what that costs an hour with the disk (`dph_total`; the disk's price
    is what the offer's total says beyond its minimum bid). An offer that cannot be priced
    so is left out, and so is any machine this account already has an instance on: two
    bids of ours on one machine outbid each other. A listing row that does not say its
    machine fails the launch rather than let such a bid through."""
    ours = set()
    for r in vast.instances():
        if type(r.get("machine_id")) is not int:
            raise v.VastError(200, f"instance {r.get('id')} is listed without its machine")
        ours.add(r["machine_id"])
    priced = []
    for o in found:
        min_bid, total = o.get("min_bid"), o["dph_total"]
        if type(min_bid) not in (int, float) or not 0 < min_bid <= total < 1e4:
            continue
        if o["machine_id"] in ours:
            continue
        bid = max(min_bid, math.ceil(min_bid * spec.bid_markup * 1e4) / 1e4)
        priced.append({**o, "_bid": bid, "dph_total": bid + (total - min_bid)})
    return priced


def candidates(vast, spec: Spec, blocked: dict[int, dict], want: int):
    """(offers to try, in order; the offers within the price cap; those also in an allowed
    country). What `up` rents from and `search` shows."""
    found = vast.offers(v.offer_query(spec))
    if spec.interruptible:
        found = priced_bids(vast, spec, found)
    # The search is asked for the price cap and the order; neither is taken
    # on trust, since what comes back is what gets rented.
    cap = spec.max_dph_per_gpu * spec.num_gpus + 1e-9
    found = sorted((o for o in found if o["dph_total"] <= cap), key=lambda o: o["dph_total"])
    allowed = v.in_allowed_country(found, spec.exclude_countries)
    return pol.select_offers(allowed, blocked, want), found, allowed


# The phases of a rented box on its way to a verdict.
PENDING = ("booting", "probing", "gating", "checking")


class Launch:
    def __init__(self, provider, spec: Spec, policy: Policy, state: State, prober):
        self.provider, self.spec, self.policy = provider, spec, policy
        self.state, self.prober = state, prober
        self.attempts: dict[int, Attempt] = {}
        self.held: set[int] = set()  # every id a create gave us, on disk or not
        # Machines a box of this launch failed on, through their fault: a provider that
        # chooses the machine itself (Slurm) is asked not to use them again.
        self.avoid: set = set()
        # The CUDA major version a box's gate said the image needs (driver_too_old):
        # offers whose driver is for an older one are not rented after that.
        self.needs_cuda: int | None = None

    # ------------------------------------------------------------ the frame

    def run(self, blocked: dict[int, dict]) -> Result:
        """Race candidates; return with `policy.count` boxes, published as the winners, or
        with none.

        Raises whatever failed, or KeyboardInterrupt on a signal, after
        trying to roll back. Winners that reached the state file are kept
        whatever was raised; everything else the run rented is destroyed.
        """
        t0 = time.monotonic()
        try:
            winners = self._race(blocked, t0) or []
            if winners:
                self.state.publish([{
                    "id": w.instance_id, "attempt": w.seq,
                    "host": w.endpoint[0], "port": w.endpoint[1],
                    "machine": w.machine_id, "host_id": w.host_id, "gpu": w.gpu,
                } for w in winners])  # fmt: skip
        except BaseException:
            deadline = time.monotonic() + self.policy.rollback  # one budget for all of it
            self._kill_children()
            self._rollback(deadline)
            raise
        self._kill_children()
        left = self._sweep({w.instance_id for w in winners}, self.policy.confirm_timeout)
        for a in self.attempts.values():
            if a in winners:
                continue
            if a.phase == "healthy":
                a.reason, a.detail = None, "healthy, but not kept"
            elif a.state == "rented" and a.phase != "failed":
                a.reason, a.detail = None, a.detail or "cancelled"
        ordered = [self.attempts[k] for k in sorted(self.attempts)]
        return Result(
            winners=winners,
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

    def _sweep(self, keep: set[int], budget: float) -> list[int]:
        """Destroy every box of the run except those in `keep`: the ids we hold, then
        whatever the listing shows with the run's tag (a box whose create gave no answer)."""
        deadline = time.monotonic() + budget
        remaining = lambda: max(0.0, deadline - time.monotonic())
        # Half the time for the ids we hold, so that one that will not go away
        # leaves room for the listing and for what it finds.
        left = set(destroy_confirmed(self.provider, self.held - keep, budget / 2, self.policy))
        found = tagged(self.provider, self.state.run, self.policy, remaining()) or set()
        extra = found - self.held  # the winners are among those we hold
        noun = self.provider.noun
        for iid in sorted(extra):
            log(f"  {noun} {iid} carries this run's tag and was not known: destroying it")
        left = sorted(destroy_confirmed(self.provider, left | extra, remaining(), self.policy))
        for iid in left:
            log(f"WARNING: {noun} {iid} is not confirmed destroyed; `gpubox down` retries")
        return left

    def _rollback(self, deadline: float) -> None:
        """An attempt within the budget, not a promise. Never masks what was raised, except
        by a second signal. Winners that are in the state file on disk are kept."""
        keep = set()
        try:
            on_disk = State.load(self.state.path.parent) if self.state.path else None
            if on_disk is not None and on_disk.run == self.state.run and on_disk.winners:
                keep = {w["id"] for w in on_disk.winners}
                self.state.data = on_disk.data
                ids = ", ".join(map(str, sorted(keep)))
                log(f"keeping instance(s) {ids}: handed over before the failure")
        except Exception as exc:
            log(f"WARNING: cannot read the state file back: {exc}")
        try:
            self._sweep(keep, max(0.0, deadline - time.monotonic()))
        except Exception as exc:
            log(f"WARNING: rolling back: {exc!r}")

    # ------------------------------------------------------------- the race

    def _race(self, blocked: dict[int, dict], t0: float) -> list[Attempt] | None:
        spec, policy = self.spec, self.policy
        deadline = t0 + policy.overall_timeout - policy.cleanup_reserve
        found, note = self.provider.candidates(spec, blocked, policy)
        offers = collections.deque(found)
        want = f"{policy.count} boxes" if policy.count > 1 else "a box"
        log(f"{note}; {want}: racing {policy.max_live}, up to {policy.attempts} attempts")
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
            offers = self._drop_old_drivers(offers)
            # A box that ran our start-up command shows the image works, whatever
            # some host says about it.
            if image_said and self._image_works():
                image_errors, image_said = {}, None
            if len(image_errors) >= policy.image_errors:
                # One host can be wrong about an image (a broken mirror, stale
                # credentials). Several saying the same are not.
                raise ConfigError(image_said)
            winners = self._choose()
            if winners is not None:
                return winners
            healthy = sum(a.phase == "healthy" for a in self.attempts.values())
            short = f" ({healthy} of {policy.count} healthy)" if policy.count > 1 else ""
            if not self._possible(offers):
                want = (
                    f"{policy.count} healthy boxes are" if policy.count > 1 else "a healthy box is"
                )
                why = f"{want} no longer possible{short}"
                return self._give_up(image_said, why)
            out_of_time = time.monotonic() >= deadline
            if not out_of_time:
                self._rent(offers, deadline)
            if out_of_time or not self._worth_waiting(offers, deadline):
                why = (
                    f"not enough healthy boxes within {policy.overall_timeout:g}s{short}"
                    if out_of_time
                    else f"nothing left to try{short}"
                )
                return self._give_up(image_said, why)
            time.sleep(policy.poll)

    def _give_up(self, image_said: str | None, why: str) -> None:
        """End the race without winners: as a configuration error if a host's word
        against the image is all there is to go on, else as no box."""
        if image_said and self._only_image_evidence():
            raise ConfigError(image_said)
        log(f"giving up: {why}")

    def _drop_old_drivers(self, offers: collections.deque) -> collections.deque:
        """The offers whose driver supports the CUDA version a gate said the image needs.
        An offer that does not say what its driver supports is kept."""
        if self.needs_cuda is None:
            return offers
        old = lambda o: type(o.get("cuda_max_good")) in (int, float) and (
            o["cuda_max_good"] < self.needs_cuda
        )
        kept = collections.deque(o for o in offers if not old(o))
        if len(kept) < len(offers):
            log(
                f"  skipping {len(offers) - len(kept)} offer(s) whose driver supports a CUDA "
                f"older than {self.needs_cuda}.0, which the image needs"
            )
        return kept

    def _image_works(self) -> bool:
        """Has any box of this launch answered SSH with the run's key? Only our start-up
        command starts an sshd that takes it, so the image was pulled and runs it."""
        return any("ssh" in a.timings for a in self.attempts.values())

    def _only_image_evidence(self) -> bool:
        """When the launch ends with one host's word against the image: is that all there
        is? Not if some box was still on its way or healthy, which may have proved it."""
        return not any(
            a.state == "rented" and (a.phase in PENDING or a.phase == "healthy")
            for a in self.attempts.values()
        )

    def _possible(self, offers: collections.deque) -> bool:
        """Can this launch still end with `policy.count` healthy boxes: those healthy now,
        those on their way, and as many more as the offers and attempts left allow?
        A launch that cannot is ended at once, before it rents or waits for nothing."""
        on_their_way = sum(
            a.state == "rented" and (a.phase in PENDING or a.phase == "healthy")
            for a in self.attempts.values()
        )
        more = min(len(offers), max(0, self.policy.attempts - self._sent()))
        return on_their_way + more >= self.policy.count

    def _elapsed(self, a: Attempt) -> float:
        return round(time.monotonic() - a.t0, 1)

    def _fail(self, a: Attempt, reason: str | None, detail: str) -> None:
        if reason == "unreachable" and self.spec.interruptible and a.state == "rented":
            # Lost while being admitted: if the provider does not say it still runs, it
            # was most likely outbid, which is the market, not the machine.
            answered, inst = self._lookup(a.instance_id)
            if not answered or inst is None or self.provider.view(inst).state != "running":
                reason, detail = None, f"{detail} (stopped: most likely outbid)"
        a.phase, a.reason, a.detail = "failed", reason, detail
        a.failed_at = time.monotonic()
        if reason and a.machine_id is not None:
            self.avoid.add(a.machine_id)
        log(f"  {a.tag_name} failed ({reason or 'not blamed'}): {detail}")

    def _lookup(self, instance_id: int):
        """(True, row or None) from the provider, or (False, None) if it did not answer."""
        try:
            return True, self.provider.instance(
                instance_id, timeout=self.policy.call_timeout, retries=0
            )
        except prov.ProviderError:
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
            need = a.report.get("needs_cuda") if not ok else None
            if need is not None:
                self.needs_cuda = max(self.needs_cuda or 0, need)
                detail += (
                    f"; not blamed: pass --min-cuda {need}.0 to rent only hosts that can run it"
                )
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
            if not ok and self.provider.own_sshd and "Permission denied" in a.ssh_err:
                # An sshd answered and refused the run's key. Ours would not:
                # it is started with that key. So the image runs something
                # else in place of our start-up command, and waiting out the
                # start-up timeout (ten minutes per box, when this was seen
                # live) cannot help. Handled like an image that cannot be
                # pulled: this box fails, and a second host settles it.
                said = (
                    f"image {spec.image!r} is running, but its sshd refuses the run's key: it "
                    "does not seem to run gpubox's start-up command"
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
        if time.monotonic() - a.ready_from > policy.startup_timeout:
            extra = a.ssh_err or a.msg[:120] or "no status message"
            # Only the provider's own account of the box is evidence against
            # the machine; our failure to reach the API is not.
            if a.answered:
                detail = (
                    f"not reachable after {policy.startup_timeout:g}s (status={a.status}; {extra})"
                )
                # An interruptible box that never got to run may simply have been outbid.
                outbid = spec.interruptible and a.status != "running"
                self._fail(a, None if outbid else "unreachable", detail)
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
                self._fail(a, None, f"the {self.provider.noun} no longer exists")
            return None
        view = self.provider.view(inst)
        status, msg = view.status, view.msg
        if not a.answered or (status, msg) != (a.status, a.msg):
            a.changed_at = time.monotonic()
            if view.state == "queued":
                log(f"  {a.tag_name} waiting in the queue ({status}: {msg or 'no reason given'})")
        a.answered, a.status, a.msg = True, status, msg
        if view.machine is not None:
            a.machine_id = view.machine
        if view.problem == "config":
            # The provider says this request can never be met (a cluster's queue: no
            # such partition, a time limit above its maximum). Every box would say so.
            raise ConfigError(
                f"{self.provider.noun} {a.instance_id} can never start: {status} ({msg})"
            )
        if view.problem == "image":
            said = f"image {spec.image!r} cannot be pulled: {a.msg[:200]}"
            self._fail(a, None, said)
            return said
        if view.problem == "machine":
            self._fail(a, "runtime_error", a.msg[:200])
            return None
        if view.state == "queued":
            # Waiting its turn, which is nobody's fault: neither the start-up timeout
            # nor the stall rule counts this time. The launch's own deadline does.
            a.ready_from = a.changed_at = time.monotonic()
            return None
        # An interruptible box that stops or never starts was most likely outbid:
        # that is the market, not the machine.
        blame = None if spec.interruptible else "unreachable"
        if view.state == "exited":
            # Our container command ends only when sshd could not be started.
            why = "outbid, or " if spec.interruptible else ""
            self._fail(a, blame, f"container exited during bootstrap ({why}{a.msg[:120]})")
            return None
        if view.state != "running":
            quiet = time.monotonic() - a.changed_at
            if quiet > policy.stall_timeout:
                # Still loading, and the provider has reported nothing new for a
                # long while: a pull that is stuck, or too slow to wait for.
                self._fail(
                    a,
                    blame,
                    f"stalled: status {a.status} unchanged for {quiet:.0f}s ({a.msg[:120]})",
                )
            return None
        ep = view.endpoint
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
                _try_destroy(self.provider, a.instance_id, self.policy.call_timeout)
                a.gone = _is_gone(self.provider, a.instance_id, self.policy.call_timeout)

    def _choose(self) -> list[Attempt] | None:
        """The `policy.count` cheapest candidates whose gate passed and that a lookup says
        are still running, with enough of the lease left; None until there are that many.
        Nobody is looked up before enough have passed the gate."""
        count = self.policy.count
        healthy = [a for a in self.attempts.values() if a.phase == "healthy"]
        if len(healthy) < count:
            return None
        chosen = []
        for a in sorted(healthy, key=lambda a: (a.dph, a.seq)):
            answered, inst = self._lookup(a.instance_id)
            if not answered:
                continue
            if inst is None:
                a.gone = True
                self._fail(a, None, "passed the gate, then the box no longer existed")
            elif (view := self.provider.view(inst)).state != "running":
                self._fail(a, None, f"passed the gate, then was {view.status}")
            elif (ep := view.endpoint) is not None and ep != a.endpoint:
                # It restarted somewhere else (an interruptible box resumed): what passed
                # the gate is not what would be handed over. Admit it again.
                log(f"  {a.tag_name} moved to {ep[0]}:{ep[1]}; checking it again")
                a.phase, a.endpoint, a.check = "booting", None, None
            else:
                chosen.append(a)
                if len(chosen) == count:
                    break
        if len(chosen) < count:
            return None
        if self.state.expiry - time.time() < self.policy.min_lease:
            raise ConfigError(
                f"less than {self.policy.min_lease:g}s of the lease is left; raise --ttl-min"
            )
        return chosen

    def _rent(self, offers: collections.deque, deadline: float) -> None:
        spec, state = self.spec, self.state
        while (
            sum(a.holds_slot for a in self.attempts.values()) < self.policy.max_live
            and self._sent() < self.policy.attempts
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
                gpu=o.get("gpu_name") if isinstance(o.get("gpu_name"), str) else None,
                bid=o.get("_bid"),
            )
            self.attempts[seq] = a  # as unanswered: if the request is interrupted, it stays so
            try:
                iid = self.provider.create(
                    o, spec, a.tag, self.prober.pubkey, state.expiry, frozenset(self.avoid)
                )
            except prov.CreateRefused as exc:
                # Usually a stale offer (rented between search and create).
                a.state, a.detail = "refused", f"create refused: {exc}"
                log(f"  {a.tag_name} {a.detail}")
                continue
            except prov.CreateUnknown as exc:
                # A box may exist. It keeps its slot; the sweep at the end looks
                # for its tag, and its lease bounds it if that does not find it.
                a.detail = f"create outcome unknown: {exc}"
                log(f"  {a.tag_name} {a.detail}")
                continue
            if iid in self.held:
                # An id we were already given cannot be this create's box. Whatever
                # it rented, if anything, is found by its tag, as for no answer.
                a.detail = f"create answered with {iid}, which an earlier create had"
                log(f"  {a.tag_name} {a.detail}")
                continue
            self.held.add(iid)
            a.instance_id, a.state, a.phase = iid, "rented", "booting"
            a.t0 = a.changed_at = a.ready_from = time.monotonic()
            state.add_id(iid)  # if this raises, the rollback destroys what `held` names
            noun, where = self.provider.noun, self.provider.describe(a)
            log(f"  {a.tag_name} rented {noun} {iid}" + (f": {where}" if where else ""))

    def _sent(self) -> int:
        """Creates that count against the attempts: those that may have rented a box. An
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
        healthy = 0
        for a in self.attempts.values():
            if a.state != "rented":
                continue
            if a.phase in PENDING:
                return True
            if a.phase == "healthy":
                healthy += 1
            if a.phase == "failed" and not a.gone and now - a.failed_at < policy.confirm_timeout:
                return True  # its delete may still show, and free its slot
        if healthy >= policy.count:
            return True  # enough; a lookup did not answer, and the next one may
        return bool(
            offers
            and sum(a.holds_slot for a in self.attempts.values()) < policy.max_live
            and self._sent() < policy.attempts
            and now < deadline
        )


_GPU_MODEL = re.compile(r"[\w .+-]{1,64}")  # what the store accepts


def gpu_model(rented: str | None, asked) -> str | None:
    """The model to record for a box: the one its offer named, else the only one asked for."""
    asked = list(asked or ())
    model = rented or (asked[0] if len(asked) == 1 else None)
    return model if model and _GPU_MODEL.fullmatch(model) else None


def observe(result: Result, spec: Spec, run_id: str, obs: pol.Observations) -> None:
    """Record what this launch taught us about machines.

    A failure is blamed on the machine only if the launch succeeded: other
    candidates, with the same image, GPU request and gate, became healthy. If
    it did not, the cause may be ours, and blaming every machine tried would
    shrink the pool for no reason.
    """
    if not result.winners:
        return
    for a in result.attempts:
        common = {"gpu": gpu_model(a.gpu, spec.gpus), "num_gpus": spec.num_gpus,
                  "gate_version": GATE_VERSION}  # fmt: skip
        if a in result.winners:
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


def down(provider, state: State, policy: Policy) -> list[int] | None:
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
            left = set(destroy_confirmed(provider, pending, policy.round, policy))
            for iid in sorted(pending - left):
                log(f"{provider.noun} {iid}: gone")
            pending = left
        found = tagged(provider, state.run, policy)
        new = (found or set()) - pending  # these have not been sent a delete yet
        if found is None:
            clean = 0
        elif found or pending:
            # Also an id we just confirmed gone, if the listing still shows
            # it: deleting it again costs nothing, and a listing that shows
            # the run's tag is not a clean one.
            for iid in sorted(found - pending):
                log(f"{provider.noun} {iid} is listed with the run's tag")
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


def explain_leftovers(provider, state: State, left: list[int] | None) -> None:
    """Tell the user what `down` could not finish, and what to do about it."""
    when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(state.expiry)) if state.expiry else None
    if left:
        log(f"::error::not confirmed destroyed: {provider.noun}(s) {', '.join(map(str, left))}")
    else:
        log("::error::the provider could not be listed, so boxes of this run may remain")
    log(f"  Run this again:   gpubox down --run-id {state.run}")
    log(f"  {provider.manual_hint(state.run, left or [])}")
    if when:
        log(
            f"  If nothing is done, each running box destroys itself when its lease ends, at {when}."
        )
        ns = state.run.split(".")[0]
        stopped = "an interruptible box that was outbid" if state.meta.get("interruptible") else (
            "one stuck before it ran")  # fmt: skip
        log(f"  A box that is not running ({stopped}) cannot: after the lease, run")
        log(f"  `gpubox reap --namespace {ns} --destroy`, as a scheduled job should anyway.")


# ---------------------------------------------------------------------- reaper


def reap(provider, policy: Policy, namespace: str, *, destroy: bool = False, now=None) -> int:
    """Destroy boxes of a namespace whose lease has ended. Returns how many could not be.

    A box is chosen if its label is a tag of this namespace and the expiry in
    it passed more than `policy.expiry_margin` ago. Nothing else on the
    account is looked at. Without `destroy` this only reports. A listing that
    cannot be read raises.
    """
    now = time.time() if now is None else now
    chosen = []
    for row in provider.instances():
        tag = parse_tag(row.get("label"))
        if tag is None or tag.namespace != namespace:
            continue
        what = f"{provider.noun} {row['id']} ({row.get('label')}, {row.get('actual_status')})"
        if tag.expiry + policy.expiry_margin < now:
            chosen.append(row["id"])
            act = "destroy" if destroy else "would destroy (report only)"
            log(f"{what}: lease ended {(now - tag.expiry) / 60:.0f} min ago -> {act}")
        else:
            log(f"{what}: lease ends in {(tag.expiry - now) / 60:.0f} min")
    if not destroy or not chosen:
        return 0
    left = destroy_confirmed(provider, chosen, policy.confirm_timeout, policy)
    for iid in left:
        log(f"ERROR: {provider.noun} {iid} is not confirmed destroyed")
    return len(left)


# ----------------------------------------------------------------------- renew


def renew(provider, state: State, prober, expiry: int) -> None:
    """Move the end of the lease of every box of the run to `expiry`.

    A box's own timer reads the expiry from a file on the box; the reaper
    reads it from the tag. The files are written first, so that if a relabel
    is refused the reaper still acts on the old expiry: a refused renewal
    can end a box early, not late. A relabel whose answer is lost, or a
    state file that cannot be written afterwards, leaves boxes renewed and
    this directory showing the old expiry; running `renew` again settles it.

    On Slurm the relabel also moves the job's time limit, which a cluster may
    refuse to extend: then the box ends at its old limit (early, not late).
    """
    boxes = state.winners
    if not boxes:
        raise ConfigError("this run has no box to renew")
    v.check_expiry(expiry)
    for i, w in enumerate(boxes):
        r = prober.run((w["host"], w["port"]), provider.expiry_command(expiry))
        if r != 0:
            # A write whose answer was lost may have happened: say what is known.
            raise prov.ProviderError(
                0,
                f"could not confirm the new expiry on {provider.noun} {w['id']} (ssh exit {r}); "
                f"{i} box(es) before it confirmed it; no tag was changed. "
                "Run `gpubox renew` again",
            )
    try:
        for w in boxes:
            provider.relabel(w["id"], make_tag(state.run, w["attempt"], expiry))
        state.set_expiry(expiry)
    except Exception:
        log(
            "WARNING: the renewal is half done. The boxes' own timers have the new expiry; "
            "their tags, which the reaper goes by, may still have the old one, and this "
            "directory still shows the old one. Run `gpubox renew` again."
        )
        raise
