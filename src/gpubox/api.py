"""gpubox as a library: rent verified boxes, run jobs on them, make sure they are destroyed.

    import gpubox

    spec = gpubox.Spec(image="ghcr.io/org/img:1", gpus=("RTX 5090",), max_dph_per_gpu=1.0,
                       ttl_min=120)
    with gpubox.up(spec, namespace="eval", name="shard", count=4) as run:
        jobs = run.start(open("shard.sh").read(), env={"TOKEN": token})
        codes = gpubox.wait_all(jobs)      # streams each box's output, prefixed
    # leaving the block destroys the boxes; `run.keep()` inside it keeps them

    run = gpubox.attach(".gpubox")         # a run that `up` (here or on the command line) left
    job = run.boxes[0].job()               # the newest job on box 0
    job.wait()                             # its exit code
    gpubox.down(run_id="eval.shard.0123456789ab")   # by tag, from anywhere

Everything is in the state directory (`state_dir`, default `.gpubox`): the run, its
SSH key, and what was learnt about machines. Errors are `errors.GpuboxError`s:
`ConfigError` (our mistake), `StateError` (the state directory), and `InfraError`
(the provider or a box) with its kinds `NoBoxes`, `TeardownIncomplete`, `BoxLost` and
`JobLost`. What it has to say goes through `launch.log` (stdout); replace that with
`set_log` to send it elsewhere.

The API key is `api_key`, else VASTAI_API_KEY, else the line `VASTAI_API_KEY=...` in
`keys_file`. `account_id` (else GPUBOX_ACCOUNT_ID) pins the account: a key of another
one is refused before anything is rented or destroyed.

Other providers (providers.py): `provider="runpod"` (its key RUNPOD_API_KEY, as for
Vast), or `provider="slurm"` with `options={"login": HOST, ...}` (`slurm.Slurm`'s
settings: partition, account, qos, constraint, sbatch_args). `options` are a
provider's own settings (RunPod: {"cloud": "COMMUNITY"}). A run remembers where its
boxes came from, so later calls on it need neither.

    with gpubox.up(gpubox.Spec(image="", gpus=(), max_dph_per_gpu=0), namespace="dev",
                   name="x", provider="slurm", options={"login": "mycluster"}) as run:
        ...
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import job as jobs
from . import launch as L
from . import policy as pol
from . import providers
from . import slurm as sl
from . import sshconfig
from . import store as st
from . import vast as v
from .errors import BoxLost, ConfigError, GpuboxError, InfraError, JobLost, NoBoxes, NoSuchJob
from .errors import TeardownIncomplete
from .gate import GATE_VERSION, Prober
from .provider import ProviderError
from .launch import Policy, Spec, log
from .state import NAMESPACE, RUN_ID, State, StateError, box_name, new_run_id, state_lock

LOCAL_KEYS_FILE = "~/.config/gpu-cloud-query/keys.env"
# Provider states that mean the box is over, as opposed to missing or in transition.
DEAD_STATUSES = ("exited", "offline")
EXIT_TIMEOUT = 124  # what a job that ran out of time exits with


def set_log(fn) -> None:
    """Send what gpubox has to say to `fn(message)` instead of stdout (None: stdout)."""
    L._sink = fn


# ---------------------------------------------------------------- the settings


@dataclass
class Settings:
    """Where things are, and how to reach the provider."""

    state_dir: Path = Path(".gpubox")
    api_key: str | None = field(default=None, repr=False)
    account_id: str | None = None
    keys_file: str = LOCAL_KEYS_FILE
    observations: Path | None = None  # None: <state_dir>/observations.jsonl
    # Told, as it happens, that a machine was recorded as died (the CLI writes a step output).
    on_observation: object = field(default=None, repr=False)
    # Where `up`, `reap`, and `down` of a run this directory does not have, get boxes
    # (providers.KINDS), with that provider's own `options` (never a secret). A run's own
    # provider is in its state file.
    provider: str = "vast"
    options: dict | None = None
    _vast: object = field(default=None, repr=False)

    def __post_init__(self):
        self.state_dir = Path(self.state_dir)
        providers.kind(self.provider)  # ConfigError for one we do not know

    @property
    def observations_path(self) -> Path:
        return (
            Path(self.observations) if self.observations else self.state_dir / "observations.jsonl"
        )

    def credential(self, name: str) -> str:
        """An API key: `api_key` (for this settings' provider), else the environment
        variable `name`, else the line `name=...` in `keys_file`."""
        key = self.api_key if name == providers.kind(self.provider).credential else None
        key = key or os.environ.get(name)
        path = Path(os.path.expanduser(self.keys_file))
        if not key and path.exists():  # local use; CI passes the secret through the environment
            for line in path.read_text().splitlines():
                var, _, value = line.removeprefix("export ").partition("=")
                if var.strip() == name:
                    key = value.strip().strip("'\"")
        if not key:
            raise ConfigError(f"{name} is not set")
        return key

    def key(self) -> str:
        return self.credential("VASTAI_API_KEY")

    def vast(self):
        if self._vast is None:
            self._vast = v.Vast(self.key())
        return self._vast

    def _make(self, kind, options: dict):
        if kind is v.VastProvider:
            return v.VastProvider(self.vast())  # one client for these settings
        if kind is sl.Slurm and not options.get("login"):
            raise ConfigError(
                "a Slurm cluster needs its login host (--login HOST): as `ssh HOST` "
                "reaches it without a password prompt"
            )
        return kind.from_options(
            options, self.credential(kind.credential) if kind.credential else None
        )

    def named_provider(self):
        """The provider these settings name (provider.py)."""
        return self._make(providers.kind(self.provider), dict(self.options or {}))

    def provider_of(self, meta: dict):
        """The provider a run's boxes came from, as its state file's `meta` says (Vast if
        it does not say)."""
        return self._make(providers.kind_of(meta), providers.options_of(meta))

    @property
    def pinned_account(self) -> str | None:
        return self.account_id or os.environ.get("GPUBOX_ACCOUNT_ID") or None

    def check_account(self, provider=None) -> str | None:
        """With an account pinned: stop, before anything is rented or destroyed, if the key
        is another account's. Returns the account, or None when none is pinned."""
        expected = self.pinned_account
        if expected is None:
            return None
        actual = (provider or self.named_provider()).account()
        if actual != expected:
            raise ConfigError(
                f"the API key belongs to account {actual}, not to the expected account {expected}"
            )
        return actual

    def same_account(self, state: State, account: str | None, provider=None) -> None:
        """Stop if the run recorded the account it was rented on and the key is another's."""
        provider = provider or self.provider_of(state.meta)
        if state.account and (account or provider.account()) != state.account:
            raise ConfigError(
                f"run {state.run} was rented on account {state.account}; "
                "the API key belongs to another account"
            )

    def shared(self) -> st.Shared:
        """The shared reputation store, if one is configured and its settings make sense."""
        try:
            return st.Shared(self.observations_path, st.from_env())
        except st.StoreError as exc:  # its messages never contain the URL or the token
            log(f"WARNING: reputation store not used: {exc}")
        except Exception as exc:
            log(f"WARNING: reputation store not used: {type(exc).__name__}")
        return st.Shared(self.observations_path, None)


def _settings(state_dir=None, settings: Settings | None = None, **kw) -> Settings:
    """The settings a call gives: `settings=`, or the keywords (and `state_dir`), not both:
    a mix would leave one of them silently unused."""
    if settings is not None:
        if state_dir is not None or kw:
            given = ", ".join(["state_dir"] * (state_dir is not None) + sorted(kw))
            raise ConfigError(f"pass either settings= or {given}, not both")
        return settings
    return Settings(state_dir=Path(state_dir if state_dir is not None else ".gpubox"), **kw)


def share(shared: st.Shared, what: str, **kw) -> None:
    """Talk to the store. Whatever happens there, the caller carries on."""
    if shared.store:
        try:
            log(f"reputation store: {shared.sync(**kw)}")
        except st.StoreError as exc:  # its messages never contain the URL or the token
            log(f"WARNING: reputation store ({what}): {exc}")
        except Exception as exc:
            log(f"WARNING: reputation store ({what}): {type(exc).__name__}")


def known_rows(obs: pol.Observations, shared: st.Shared) -> list[dict]:
    """This client's own rows, then what the store said everyone saw.

    A row of ours that the store has is taken as the store has it, under the
    source that sent it. Several CI runners share one token: that way a
    machine one of them saw fail and another then saw recover is judged as
    one source's word, not kept blocked by the first runner's local copy.
    """
    try:
        remote = shared.remote_rows()
    except Exception as exc:
        log(f"WARNING: reputation store not used: {type(exc).__name__}")
        remote = []
    echoed = {r["id"] for r in remote}
    return [*(r for r in obs.rows if r.get("id") not in echoed), *remote]


def advisory(what: str, fn, *a, **kw) -> None:
    """Run something whose failure must not fail the caller (reports, observations)."""
    try:
        fn(*a, **kw)
    except Exception as exc:  # whatever it is, it is not worth a box
        log(f"WARNING: {what} failed: {exc!r}")


def is_slurm(meta: dict) -> bool:
    """Is this run (its state file's `meta`) on a Slurm cluster?"""
    return providers.kind_of(meta) is sl.Slurm


def prober_for(state_dir: Path, run_id: str, meta: dict):
    """What reaches a run's boxes: SSH with the run's key (Vast, RunPod), or a step of the
    job through the cluster's login host (Slurm, where a box's host is the login host
    and its port the job id). Needs no API key."""
    if providers.kind_of(meta).access == "srun":
        return sl.SlurmProber()
    try:
        return Prober(key_dir(state_dir, run_id), create=False)
    except FileNotFoundError:
        raise StateError(f"run {run_id} has no SSH key in {state_dir} (torn down?)") from None


def key_dir(state_dir: Path, run_id: str) -> Path:
    """Where a run keeps its SSH key and known hosts. Never shared between runs:
    a command still holding an earlier run's endpoint must not be able to log
    in to a later run's box that happens to get the same address."""
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise StateError(f"not a run id: {run_id!r}")  # it becomes part of a path
    return Path(state_dir) / "keys" / run_id


# ------------------------------------------------------------------------ up


def up(
    spec: Spec,
    *,
    namespace: str,
    name: str,
    count: int | None = None,
    policy: Policy | None = None,
    state_dir=None,
    ssh_config: bool = False,
    on_run_id=None,
    on_result=None,
    teardown_on_failure: bool = True,
    **settings,
) -> Run:
    """Rent `count` boxes for `spec` (default: `policy.count`, 1) and wait until they
    pass the health gate.

    Returns the run, with its boxes, once they are in the state file; raises
    `NoBoxes` if the launch ended without them, `ConfigError` for our own
    mistakes. A live run in the state directory must be torn down first.

    `on_run_id(run_id)` is called before anything is rented: whoever must tear the
    run down needs it whenever a box may exist, so if it raises, nothing is
    rented. `on_result(result, run)` is told how the launch went, whichever way;
    it cannot change the outcome.

    If anything fails or interrupts `up` once it has begun the run, it tears the
    run down before raising (`teardown_on_failure`; the command line leaves that
    to its own `down`, as CI runs it anyway). If that teardown cannot be confirmed
    either, the exception carries a note with the run id to retry with.
    """
    s = _settings(state_dir, **settings)
    kind = providers.kind(s.provider)
    policy = policy or Policy(spare=kind.default_spare)
    if count is None:
        count = policy.count
    elif count != policy.count and policy.count != 1:
        raise ConfigError(f"count={count} and policy.count={policy.count} disagree")
    if type(count) is not int or count < 1:
        raise ConfigError(f"count must be a positive integer: {count!r}")
    if count != policy.count:
        policy = Policy(**{**policy.__dict__, "count": count})
    if policy.attempts < policy.count:
        raise ConfigError(
            f"{policy.attempts} attempts (--max-attempts) cannot rent {policy.count} boxes"
        )
    if spec.ttl_min * 60 < policy.overall_timeout + policy.min_lease:
        raise ConfigError(
            f"a lease of {spec.ttl_min} min (--ttl-min) leaves nothing after a launch of up to "
            f"{policy.overall_timeout:g}s; it must cover that plus {policy.min_lease:g}s"
        )
    countries_given = tuple(spec.exclude_countries) != Spec.__dataclass_fields__[
        "exclude_countries"].default  # fmt: skip
    providers.check_spec(kind, spec, ssh_config=ssh_config, countries_given=countries_given)
    provider = s.named_provider()
    begun: list[str] = []  # the run id, once its state is begun: from then on, it is ours
    try:
        return _up(spec, namespace, name, policy, s, provider, ssh_config, on_run_id, on_result,
                   begun)  # fmt: skip
    except BaseException as exc:
        if teardown_on_failure and begun:
            try:
                down(begun[0], settings=s)
            except BaseException as also:
                exc.add_note(
                    f"its teardown failed too ({also!r}): `gpubox down --run-id {begun[0]}` "
                    "retries; each box also ends at its lease"
                )
        raise


def _up(spec, namespace, name, policy, s, provider, ssh_config, on_run_id, on_result, begun):
    with state_lock(s.state_dir):
        old = State.load(s.state_dir)  # StateError if it cannot be read
        if old is not None and not old.closed:
            raise StateError(
                f"run {old.run} in {s.state_dir} is not closed; run `gpubox down` first"
            )
        account = s.check_account(provider)
        # A key that cannot list could never find its boxes again: learn that
        # before renting one.
        provider.instances()
        run_id = new_run_id(namespace, name)
        if on_run_id is not None:
            on_run_id(run_id)
        expiry = int(time.time()) + spec.ttl_min * 60
        meta = {
            "image": spec.image, "gpus": list(spec.gpus), "num_gpus": spec.num_gpus,
            "count": policy.count, "interruptible": spec.interruptible,
        }  # fmt: skip
        if provider.name != "vast":
            meta.update(provider=provider.name, options=provider.options())
        state = State.begin(s.state_dir, run_id, expiry, account, meta)
        begun.append(run_id)
        log(f"run {run_id}")
        if provider.observes_machines:
            obs = pol.Observations(s.observations_path)
            shared = s.shared()
            share(shared, "before the launch")  # at most a few seconds; see store.BUDGET
            blocked = pol.blocklist(known_rows(obs, shared), time.time())
        else:
            # Its machine ids, if any, are not comparable from run to run (providers.py).
            obs, shared, blocked = None, None, {}
        if provider.access == "srun":
            prober = sl.SlurmProber()
        else:
            prober = Prober(key_dir(s.state_dir, run_id))
        launch = L.Launch(provider, spec, policy, state, prober)
        result = launch.run(blocked)
        won = bool(result.winners)  # the outcome, decided: nothing below may change it
        run = Run(s, state)
        run.result = result
        if won and ssh_config:
            advisory("writing the SSH configuration", run.write_ssh_config)
    if obs is not None:
        advisory("recording observations", L.observe, result, spec, run_id, obs)
        if obs.added:
            advisory("queueing observations", shared.queue, obs.rows[-obs.added :])
        share(shared, "after the launch", pull=False)  # also anything left from earlier runs
        run.observations_changed = obs.added > 0
    if on_result is not None:
        advisory("reporting", on_result, result, run)
    if not won:
        raise NoBoxes(f"no healthy {spec.gpu_text} box", result)
    return run


def attach(state_dir=None, **settings) -> Run:
    """The live run in a state directory."""
    s = _settings(state_dir, **settings)
    state = State.load(s.state_dir)
    if state is None or not state.winners or state.closing:
        raise StateError(f"no live box in {s.state_dir}; run `gpubox up` first")
    return Run(s, state)


def down(run_id: str | None = None, state_dir=None, policy: Policy | None = None,
         **settings) -> None:  # fmt: skip
    """Destroy everything of a run: the state directory's, or with `run_id` the one of that
    id, found by its tag if the directory has no state of it. Returns once every box is
    confirmed gone; raises `TeardownIncomplete` otherwise."""
    s = _settings(state_dir, **settings)
    if run_id is not None and not RUN_ID.fullmatch(run_id):
        # Also an empty one: a job whose `up` never got as far as a run id has
        # rented nothing, and must not tear down whatever else is here.
        raise StateError(f"not a run id: {run_id!r}")
    with state_lock(s.state_dir):
        try:
            state = State.load(s.state_dir)
        except StateError as exc:
            if run_id is None:
                raise  # we cannot even tell which run this is
            log(f"WARNING: {exc}; working from the run id alone")
            state = None
        if state is None and run_id is None:
            log(f"no run in {s.state_dir}: nothing to destroy")
            return
        ours = state is not None and (run_id is None or state.run == run_id)
        # The run's own provider, from its state; for a run this directory does not have,
        # the one the settings name.
        provider = s.provider_of(state.meta) if ours else s.named_provider()
        account = s.check_account(provider)
        if ours:
            # An empty listing on another account says nothing about this run.
            s.same_account(state, account, provider)
        else:
            # By tag alone. The directory's own state and keys are not touched.
            log(f"run {run_id} has no state in {s.state_dir}: destroying what carries its tag")
            state = State.ephemeral(run_id)
        left = L.down(provider, state, policy or Policy())
        if left == []:
            log(f"run {state.run}: nothing left")
            if ours:
                # Nothing of this run is left to log in to.
                shutil.rmtree(key_dir(s.state_dir, state.run), ignore_errors=True)
                advisory("removing the SSH configuration entry", sshconfig.remove, state.run)
            return
        L.explain_leftovers(provider, state, left)
        raise TeardownIncomplete(f"run {state.run}: teardown not confirmed", left or [])


def reap(namespace: str, *, destroy: bool = False, policy: Policy | None = None,
         **settings) -> int:  # fmt: skip
    """Destroy (or with `destroy=False`, report) boxes of a namespace whose lease has
    ended. Returns how many could not be destroyed."""
    if not NAMESPACE.fullmatch(namespace):
        raise ConfigError(f"not a namespace: {namespace!r}")
    s = _settings(**settings)
    provider = s.named_provider()
    s.check_account(provider)
    return L.reap(provider, policy or Policy(), namespace, destroy=destroy)


def search(spec: Spec, *, want: int = 2, state_dir=None, **settings):
    """(offers `up` would try, in order; the blocked machines). Rents nothing."""
    s = _settings(state_dir, **settings)
    kind = providers.kind(s.provider)
    if not kind.searchable:
        raise ConfigError(f"there are no offers to search on {kind.name}")
    if kind is not v.VastProvider:
        offers, _ = s.named_provider().candidates(spec, {}, Policy(spare=0))  # each once
        return offers[:want] if want > 2 else offers, {}
    obs = pol.Observations(s.observations_path)
    shared = s.shared()
    share(shared, "search")
    blocked = pol.blocklist(known_rows(obs, shared), time.time())
    offers, _, _ = L.candidates(s.vast(), spec, blocked, want)
    return offers, blocked


# ------------------------------------------------------------------- the run


class Run:
    """A run in a state directory: its boxes, their lease, and their teardown.

    A handle, not a lock: another process (or `gpubox down`) can end the run, and
    then everything here that would touch its boxes raises `StateError`. As a
    context manager it destroys the run when the block ends, however it ends,
    unless `keep()` was called inside that block.
    """

    def __init__(self, settings: Settings, state: State):
        self.settings = settings
        self._id, self._meta, self._expiry = state.run, dict(state.meta), state.expiry
        self._keep = False
        self.observations_changed = False
        self.result = None  # how `up` went, for a run that `up` returned

    id = property(lambda self: self._id)
    meta = property(lambda self: dict(self._meta))  # what was asked for
    interruptible = property(lambda self: bool(self._meta.get("interruptible")))
    kind = property(lambda self: providers.kind_of(self._meta))  # its provider's class
    on_slurm = property(lambda self: self.kind is sl.Slurm)

    @property
    def provider(self):
        """Where the run's boxes came from (provider.py)."""
        return self.settings.provider_of(self._meta)

    def _live(self) -> State:
        """The run's state on disk, or StateError if the directory no longer holds it live
        (torn down, being torn down, replaced by another run, unreadable)."""
        state = State.load(self.settings.state_dir)
        if state is None or state.run != self.id:
            raise StateError(f"run {self.id} is no longer in {self.settings.state_dir}")
        if state.closing or not state.winners:
            raise StateError(f"run {self.id} has been torn down, or is being torn down")
        self._expiry = state.expiry
        return state

    @property
    def expiry(self) -> int:
        """When the lease ends (Unix time), as last seen."""
        try:
            self._live()
        except StateError:
            pass
        return self._expiry

    @property
    def boxes(self) -> list[Box]:
        state = self._live()
        ws = state.winners
        return [Box(self, i, box_name(state.run, i, len(ws)), w) for i, w in enumerate(ws)]

    def box(self, index: int | None = None) -> Box:
        """Box `index`, or the only box if None."""
        boxes = self.boxes
        if index is None:
            if len(boxes) != 1:
                raise StateError(
                    f"the run has {len(boxes)} boxes: say which with --box 0..{len(boxes) - 1}"
                )
            index = 0
        if type(index) is not int or not 0 <= index < len(boxes):
            raise StateError(
                f"no box {index}: the run has {len(boxes)} (--box 0..{len(boxes) - 1})"
            )
        return boxes[index]

    def live(self) -> bool:
        """False once this run's teardown has started (here or in another process)."""
        try:
            state = State.load(self.settings.state_dir)
        except StateError:
            return False
        return state is not None and state.run == self.id and not state.closing

    def start(self, script: str, env: dict[str, str] | None = None, *, boxes=None,
              timeout_min: float = 60, destroy_after: bool = False) -> list[Job]:  # fmt: skip
        """Start `script` on each box (`boxes`: their indices; default all), as one job id.
        If a box cannot start it, the error is raised with `.jobs`, the jobs that did start
        and carry on."""
        chosen = self.boxes if boxes is None else [self.box(i) for i in boxes]
        job_id = jobs.new_job_id()
        started = []
        for box in chosen:
            try:
                started.append(
                    box.start(
                        script,
                        env,
                        timeout_min=timeout_min,
                        destroy_after=destroy_after,
                        job_id=job_id,
                    )  # fmt: skip
                )
            except GpuboxError as exc:
                exc.jobs = started
                raise
        return started

    def renew(self, ttl_min: int) -> int:
        """Move the end of every box's lease to `ttl_min` from now. Returns the new expiry."""
        if type(ttl_min) is not int or ttl_min < 1:
            raise ConfigError(f"ttl_min must be a positive number of minutes: {ttl_min!r}")
        s = self.settings
        with state_lock(s.state_dir):
            state = self._live()  # this run, under the lock: not one that replaced it
            provider = s.provider_of(state.meta)
            account = s.check_account(provider)
            s.same_account(state, account, provider)  # before anything on the box is changed
            prober = prober_for(s.state_dir, state.run, state.meta)
            expiry = int(time.time()) + ttl_min * 60
            L.renew(provider, state, prober, expiry)
        self._expiry = expiry
        return expiry

    def down(self, policy: Policy | None = None) -> None:
        down(self.id, policy=policy, settings=self.settings)

    def keep(self) -> Run:
        """Do not destroy the run when the current `with` block ends."""
        self._keep = True
        return self

    def write_ssh_config(self) -> Path:
        """`Host` entries in ~/.ssh/gpubox_config for each box (see sshconfig)."""
        if "ssh_config" not in self.kind.features:
            raise ConfigError(
                f"a box on {self.kind.name} has no SSH configuration entry: `gpubox ssh` reaches it"
            )
        boxes = self.boxes
        keys = key_dir(self.settings.state_dir, self.id)
        hosts = [(b.name, b.host, b.port) for b in boxes]
        path = sshconfig.add(self.id, hosts, keys / "id_ed25519", keys / "known_hosts")
        names = " / ".join(f"`ssh {b.name}`" for b in boxes)
        log(f"{names} now reach{'es' if len(hosts) == 1 else ''} the box(es) (entries in {path})")
        return path

    def __enter__(self) -> Run:
        self._keep = False
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        keep, self._keep = self._keep, False
        if keep:
            return False
        try:
            self.down()
        except BaseException as failed:
            if exc_type is None:
                raise
            # The block's own exception is what the caller needs to see; the
            # teardown's failure is said, with how to finish it.
            log(f"WARNING: the teardown of run {self.id} failed ({failed!r}); "
                f"`gpubox down --run-id {self.id}` retries; each box also ends at its lease")  # fmt: skip
        return False


# -------------------------------------------------------------------- a box


class Box:
    """One box of a run."""

    def __init__(self, run: Run, index: int, name: str, w: dict):
        self.run, self.index, self.name, self._w = run, index, name, w
        self.instance_id, self.host, self.port = w["id"], w["host"], w["port"]
        self.gpu, self.machine = w.get("gpu"), w.get("machine")

    def __repr__(self) -> str:
        if self.run.kind.access == "srun":  # its host is the login host and its port the job id
            return f"Box({self.name}, job {self.instance_id} on {self.machine}, via {self.host})"
        return f"Box({self.name}, instance {self.instance_id}, root@{self.host}:{self.port})"

    def ssh_argv(self, remote: str | None, *, tty=None) -> list[str]:
        """ssh's argv for `remote` on the box (None: a login shell): with the run's key, or
        on a Slurm cluster as a step of the box's job."""
        prober = prober_for(self.run.settings.state_dir, self.run.id, self.run.meta)
        return prober.ssh((self.host, self.port), remote, tty=tty)

    def sh(self, command: str, stdin: str | None = None, timeout: float = 60):
        """Run a shell command on the box, with the image's environment, and wait for it:
        (exit code, stdout, stderr). 255 is ssh's own failure."""
        return jobs.run_remote(self.ssh_argv, jobs.LOAD_ENV + command, stdin, timeout=timeout)

    def start(self, script: str, env: dict[str, str] | None = None, *, timeout_min: float = 60,
              destroy_after: bool = False, job_id: str | None = None) -> Job:  # fmt: skip
        """Start `script` on the box, detached, and return the job; raises `BoxLost` or
        `InfraError` if the box did not start it."""
        job_id = job_id or jobs.new_job_id()
        run = self.run
        env = {
            **(env or {}), "GPUBOX_RUN_ID": run.id, "GPUBOX_JOB": job_id,
            "GPUBOX_BOX_INDEX": str(self.index), "GPUBOX_BOX_COUNT": str(len(run.boxes)),
            "GPUBOX_BOX_NAME": self.name,
        }  # fmt: skip
        timeout_s = 0 if not timeout_min else max(1, round(timeout_min * 60))
        try:
            remote, stdin = jobs.start_command(
                job_id, script, env, timeout_s=timeout_s, destroy_after=destroy_after,
                spawn=self.run.kind.spawn,
            )  # fmt: skip
        except jobs.JobError as exc:
            raise ConfigError(str(exc)) from None
        rc, out, err = jobs.run_remote(self.ssh_argv, remote, stdin, timeout=120)
        if rc == 0 and out.strip() == job_id:
            log(f"job {job_id} started on {self.name}")
            return Job(self, job_id)
        if rc == 255:
            raise self._lost(
                f"  job {job_id} may or may not have started on {self.name}: "
                f"`gpubox jobs --box {self.index}`"
            )
        raise InfraError(
            f"could not start the job on {self.name} (exit {rc}): {err.strip()[-300:]}"
        )

    def job(self, job_id: str | None = None) -> Job:
        """The job of that id, else the newest job on the box."""
        if job_id is not None:
            try:
                return Job(self, jobs.check_job_id(job_id))
            except jobs.JobError as exc:
                raise ConfigError(str(exc)) from None
        rc, out, err = jobs.run_remote(self.ssh_argv, jobs.latest_command())
        if self._refreshed(rc):
            rc, out, err = jobs.run_remote(self.ssh_argv, jobs.latest_command())
        if rc != 0:
            raise InfraError(
                f"could not ask {self.name} for its jobs (exit {rc}): {err.strip()[-200:]}"
            )
        last = out.strip().splitlines()[-1] if out.strip() else ""
        if not last:
            raise NoSuchJob(f"no job on {self.name}")
        if not jobs.JOB_ID.fullmatch(last):
            raise InfraError(f"{self.name} answered with no job id: {out.strip()[-200:]}")
        return Job(self, last)

    def jobs(self) -> list[tuple[str, str]]:
        """(job id, `exit N` / `running` / `lost`) for each job on the box, oldest first."""
        rc, out, err = jobs.run_remote(self.ssh_argv, jobs.list_command())
        if self._refreshed(rc):
            rc, out, err = jobs.run_remote(self.ssh_argv, jobs.list_command())
        if rc != 0:
            raise InfraError(f"could not list the jobs (exit {rc}): {err.strip()[-200:]}")
        return [tuple(line.split(" ", 1)) for line in out.splitlines() if " " in line]

    # What the provider says about the box, when SSH to it fails.

    def watch(self) -> BoxWatch:
        return BoxWatch(self)

    def _lost(self, hint: str = "", watch: BoxWatch | None = None) -> InfraError:
        """The error for a box we lost the connection to: why if the provider knows, and
        the machine recorded as died if it says the box is over."""
        watch = watch or self.watch()
        if watch.status is None:
            watch()
        if not self.run.live():
            return BoxLost("the box was destroyed by `gpubox down` while the job ran", watch.status)
        msg = f"lost the SSH connection to the box (instance status: {watch.status or 'unknown'})"
        if self.run.interruptible and watch.status in DEAD_STATUSES:
            msg += "; it is interruptible, so it was most likely outbid"
        if hint:
            log(hint)
        self._record_if_died(watch.status)
        return BoxLost(msg, watch.status)

    def _moved(self, host: str, port: int) -> None:
        """The box answers at this address now (a resumed interruptible box may move): use
        it, and write it to the state file unless a teardown holds the directory. A write
        that could not be made (the directory was busy) is tried again at the next call."""
        if (host, port) != (self.host, self.port):
            log(f"{self.name} is now at {host}:{port}")
            self.host, self.port, self._unsaved = host, port, True
        if not getattr(self, "_unsaved", False):
            return
        s = self.run.settings
        try:
            with state_lock(s.state_dir):
                state = State.load(s.state_dir)
                if state is None or state.run != self.run.id or state.closing:
                    self._unsaved = False  # not ours to write any more
                    return
                winners = state.winners
                for w in winners:
                    if w["id"] == self.instance_id:
                        w["host"], w["port"] = host, port
                state.publish(winners)
                self._unsaved = False
        except (StateError, OSError) as exc:
            log(f"WARNING: the new address is not in the state file yet: {exc}")

    def _refreshed(self, rc: int) -> bool:
        """After an ssh failure (255): ask the provider where the box is now. True if it
        answers somewhere else, so that a read-only command is worth one more try."""
        if rc != 255:
            return False
        before = (self.host, self.port)
        self.watch()()
        return (self.host, self.port) != before

    def _record_if_died(self, status) -> None:
        """Blame the machine only on the provider's word that the box is over, and only if
        our own teardown did not start in the meantime. An instance that no longer exists
        was destroyed by someone (a teardown elsewhere, its own timer, the reaper), which
        says nothing about the machine. A missing or transitional status, or an API error,
        is not evidence either."""
        if not self.run.kind.observes_machines:  # its machine ids mean nothing across runs
            return
        if status not in DEAD_STATUSES or not self.run.live():
            return
        if self.run.interruptible:  # stopped by the market, most likely: not the machine
            return
        meta, s = self.run.meta, self.run.settings
        asked = meta.get("gpus") or ([meta["gpu"]] if "gpu" in meta else [])  # v2: one model
        extra = {"gpu": L.gpu_model(self.gpu, asked), "num_gpus": meta.get("num_gpus")}

        def record():
            row = pol.observation(
                "died_after_admission", self.machine, self._w.get("host_id"), self.run.id,
                reason="died", gate_version=GATE_VERSION, **extra,
            )  # fmt: skip
            pol.Observations(s.observations_path).add(row)
            shared = s.shared()
            shared.queue([row])
            share(shared, "after the run", pull=False)

        advisory("recording the observation", record)
        self.run.observations_changed = True
        if s.on_observation is not None:
            advisory("telling of the observation", s.on_observation)


class BoxWatch:
    """Asks the provider about a box whose SSH connection failed, and remembers the
    answer, so that a box that is over ends the wait and can be recorded as died."""

    def __init__(self, box: Box):
        self.box, self.status = box, None

    def __call__(self) -> bool | None:
        """False if the box is over (gone, or in a terminal state); None if not known."""
        try:
            provider = self.box.run.provider
            inst = provider.instance(self.box.instance_id)
        except (ProviderError, ConfigError) as exc:
            log(f"WARNING: could not ask the provider about the box: {exc}")
            return None
        if inst is None:
            self.status = "gone"
            return False
        view = provider.view(inst)
        self.status = inst.get("actual_status")
        if view.state == "running" and view.endpoint is not None:
            self.box._moved(*view.endpoint)
        return False if view.state == "exited" or self.status in DEAD_STATUSES else None


# --------------------------------------------------------------------- a job


class Job:
    """A job on a box. It runs whether or not anything here follows it."""

    def __init__(self, box: Box, job_id: str):
        self.box, self.id = box, job_id

    def __repr__(self) -> str:
        return f"Job({self.id} on {self.box.name})"

    def wait(self, output=None, *, quiet: bool = False, reconnect_min: float = 10,
             started=None, cancel=None) -> int:  # fmt: skip
        """Wait for the job to end, copying its log to `output` (a binary stream; default
        stdout) unless `quiet`, and return its exit code (124: it ran out of time). Raises
        `JobLost`, `BoxLost` or `NoSuchJob`. A dropped connection is picked up again."""
        return self._follow(output, wait=True, quiet=quiet, reconnect_min=reconnect_min,
                            started=started, cancel=cancel)  # fmt: skip

    def logs(self, output=None, *, follow: bool = False, reconnect_min: float = 10,
             started=None, cancel=None) -> int | None:  # fmt: skip
        """Copy the job's log so far to `output`; with `follow`, until it ends. Returns its
        exit code once it has ended, else None."""
        return self._follow(output, wait=follow, quiet=False, reconnect_min=reconnect_min,
                            started=started, cancel=cancel)  # fmt: skip

    def _follow(self, output, *, wait, quiet, reconnect_min, started, cancel):
        watch = self.box.watch()
        out = output if output is not None else sys.stdout.buffer
        end, _ = jobs.follow(
            self.box.ssh_argv, self.id, out, wait=wait, quiet=quiet, still_there=watch,
            patience=reconnect_min * 60, started=started, cancel=cancel,
        )  # fmt: skip
        if end.kind == "cancelled":
            raise KeyboardInterrupt
        if end.kind == "exit":
            return end.rc
        if end.kind == "running":
            return None
        if end.kind == "none":
            raise NoSuchJob(f"no job {self.id} on {self.box.name}")
        if end.kind == "lost":
            why = " (it is interruptible: most likely outbid)" if self.box.run.interruptible else ""
            raise JobLost(
                f"job {self.id} on {self.box.name} ended without an exit code: its box "
                f"restarted{why}, or it was killed"
            )
        raise self.box._lost(watch=watch)

    def kill(self) -> str:
        """Ask the job's wrapper to end it (SIGTERM, then SIGKILL after the grace); returns
        what the box said."""
        timeout = jobs.KILL_GRACE + 45
        rc, out, err = jobs.run_remote(
            self.box.ssh_argv, jobs.kill_command(self.id), timeout=timeout
        )
        if rc != 0:
            raise InfraError(f"could not kill job {self.id} (exit {rc}): {err.strip()[-200:]}")
        return out.strip()


class Prefixed:
    """A binary stream that writes whole lines to `out`, each led by a box's name."""

    LONGEST = 1 << 16  # a longer line is shown in pieces, not held back without end

    def __init__(self, name: str, lock, out=None):
        self.prefix, self.lock, self.partial = f"[{name}] ".encode(), lock, b""
        self.out = out

    def write(self, data: bytes) -> None:
        *lines, self.partial = (self.partial + data).split(b"\n")
        while len(self.partial) > self.LONGEST:
            lines.append(self.partial[: self.LONGEST])
            self.partial = self.partial[self.LONGEST :]
        if lines:
            out = self.out or sys.stdout.buffer
            with self.lock:
                for line in lines:
                    out.write(self.prefix + line + b"\n")
                out.flush()

    def flush(self) -> None:
        pass

    def close(self) -> None:
        if self.partial:
            self.write(b"\n")


@dataclass
class Outcome:
    """How waiting on one job ended: its exit `code` (None: still running, when not
    following), or the `error` it ended with (a `GpuboxError`, or whatever else went
    wrong here, as it was)."""

    job: Job
    code: int | None = None
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.code in (0, None)


def wait_all(job_list: list[Job], *, output=None, quiet: bool = False, follow: bool = True,
             reconnect_min: float = 10) -> list[Outcome]:  # fmt: skip
    """Follow several jobs at once (each box's output led by its name when there is more
    than one) and return how each ended. On KeyboardInterrupt the followers are stopped
    and joined, the jobs are not, and the interrupt is raised again."""
    lock = threading.Lock()
    results: dict[int, Outcome] = {}
    procs: list = []  # the ssh processes following, to stop if we are interrupted
    cancel = threading.Event()

    def one(k, j):
        out = output if (len(job_list) == 1 or quiet) else Prefixed(j.box.name, lock, output)
        kw = {"reconnect_min": reconnect_min, "started": procs.append, "cancel": cancel}
        try:
            code = j.wait(out, quiet=quiet, **kw) if follow else j.logs(out, **kw)
            results[k] = Outcome(j, code)
        except BaseException as exc:  # noqa: BLE001 - handed back, not lost in a thread
            results[k] = Outcome(j, error=exc)
        if isinstance(out, Prefixed):
            out.close()

    threads = [
        threading.Thread(target=one, args=(k, j), daemon=True) for k, j in enumerate(job_list)
    ]
    try:
        for t in threads:
            t.start()
        for t in threads:
            while t.is_alive():
                t.join(0.5)
    except KeyboardInterrupt:
        cancel.set()  # no follower starts another connection
        for proc in procs:  # our followers, not the jobs
            try:
                proc.kill()
            except OSError:
                pass
        for t in threads:
            t.join(5)
        raise
    return [
        results.get(k, Outcome(j, error=InfraError("no result"))) for k, j in enumerate(job_list)
    ]
