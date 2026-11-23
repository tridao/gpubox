"""gpubox: rent a verified GPU box from Vast.ai, and make sure it is destroyed.

    up      rent boxes and wait until they are reachable and pass the health gate
    status  show the run's boxes: names, addresses, lease
    run     run a local script on a box over SSH, streaming its output
    renew   move the end of the boxes' lease
    down    destroy everything this state directory's run rented
    reap    report, or destroy, boxes of a namespace whose lease has ended
    search  show the offers `up` would try (rents nothing)

Exit codes: 0 ok; 2 configuration (image cannot be pulled, no API key, state
directory in use, bad arguments); 75 infrastructure (no healthy box, box lost
mid-run, teardown not confirmed); 124 script timeout. `run` otherwise returns
the script's own exit code, except that 255 is taken for a lost connection.

The API key is read from VASTAI_API_KEY. `--account-id` (or GPUBOX_ACCOUNT_ID)
pins the provider account: the commands then refuse to act with a key that
belongs to another one.

If GPUBOX_STORE_URL and GPUBOX_STORE_TOKEN are set, observations of machines
are shared through that reputation store (`store/`). It is advisory: it costs
a command at most a few seconds and never fails it.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import policy as pol
from . import sshconfig
from . import store as st
from . import vast as v
from .gate import GATE_VERSION, Prober
from .launch import (
    ConfigError,
    Launch,
    Policy,
    Result,
    Spec,
    down,
    explain_leftovers,
    gpu_model,
    log,
    observe,
    reap,
    renew,
)
from .state import NAMESPACE, RUN_ID, State, StateError, box_name, new_run_id, state_lock

EXIT_CONFIG = 2
EXIT_INFRA = 75
EXIT_TIMEOUT = 124
EXIT_INTERRUPTED = 130
DEFAULTS = Policy()
LOCAL_KEYS_FILE = "~/.config/gpu-cloud-query/keys.env"


def api_key(args) -> str:
    key = os.environ.get("VASTAI_API_KEY")
    path = Path(os.path.expanduser(args.keys_file))
    if not key and path.exists():  # local use; CI passes the secret through the environment
        for line in path.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "VASTAI_API_KEY":
                key = value.strip().strip("'\"")
    if not key:
        raise ConfigError("VASTAI_API_KEY is not set")
    return key


def github_output(**values) -> None:
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a") as f:
            f.writelines(f"{k}={v}\n" for k, v in values.items())


def write_summary(spec: Spec, result: Result) -> None:
    if not (path := os.environ.get("GITHUB_STEP_SUMMARY")):
        return
    ws = result.winners
    head = (
        f"{len(ws)} healthy box(es) after {result.seconds:.0f}s: "
        + ", ".join(f"machine {w.machine_id} (${w.dph:.2f}/hr)" for w in ws)
        if ws
        else f"no healthy box after {result.seconds:.0f}s"
    )
    lines = [
        f"### Vast launch: {spec.gpu_text}",
        "",
        head,
        "",
        "| machine | host | location | $/hr | result | seconds | detail |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in result.attempts:
        outcome = "healthy" if a in ws else (a.reason or "not blamed")
        secs = a.timings.get("healthy") or a.timings.get("ssh") or ""
        detail = a.detail.replace("|", "/").replace("\n", " ")[:160]
        lines.append(
            f"| {a.machine_id} | {a.host_id} | {a.geo} | {a.dph:.2f} | {outcome} | {secs} | {detail} |"
        )
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def spec_from(args, image: str = "") -> Spec:
    cap = None
    if args.capability:
        if not re.fullmatch(r"\d+\.\d+", args.capability):
            raise ConfigError(f"--capability expects MAJOR.MINOR, got {args.capability!r}")
        cap = tuple(int(x) for x in args.capability.split("."))
    return Spec(
        image=image,
        gpus=tuple(args.gpu),
        min_cpus=args.min_cpus,
        min_ram_gb=args.min_ram_gb,
        num_gpus=args.num_gpus,
        capability=cap,
        min_vram_mib=args.min_vram_mib,
        min_cuda=args.min_cuda,
        disk_gb=args.disk_gb,
        max_dph_per_gpu=args.max_dph_per_gpu,
        must_reach=tuple(args.must_reach),
        exclude_countries=countries(args),
        gate_script=gate_script(args),
        **registry_login(args),
        gate_timeout=getattr(args, "gate_timeout", 600),
        ttl_min=getattr(args, "ttl_min", 90),
    )


REGISTRY_TOKEN = "GPUBOX_REGISTRY_TOKEN"


def registry_login(args) -> dict:
    """A private image's login: the user from --registry-user, the token from the
    environment, never from the command line."""
    user = getattr(args, "registry_user", None)
    if user is None:
        return {}
    token = os.environ.get(REGISTRY_TOKEN)
    if not token:
        raise ConfigError(f"--registry-user needs the registry token in {REGISTRY_TOKEN}")
    if any(c.isspace() for c in user + token):
        raise ConfigError("the registry user and token cannot contain spaces")
    return {"registry_user": user, "registry_token": token}


def gate_script(args) -> str | None:
    path = getattr(args, "gate_script", None)
    if path is None:
        return None
    try:
        return Path(path).read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read --gate-script {path}: {exc}") from None


def countries(args) -> tuple[str, ...]:
    """The country codes to skip: CN unless --exclude-country says otherwise ("none": skip none)."""
    given = args.exclude_country if args.exclude_country is not None else ["CN"]
    codes = tuple(c.upper() for c in given if c.lower() != "none")
    for code in codes:
        if not re.fullmatch(r"[A-Z]{2}", code):
            raise ConfigError(
                f"--exclude-country expects a two-letter code or 'none', got {code!r}"
            )
    return codes


def observations_path(args) -> Path:
    if args.observations:
        return Path(args.observations)
    return Path(args.state_dir) / "observations.jsonl"


def open_shared(args) -> st.Shared:
    """The shared reputation store, if one is configured and its settings make sense."""
    try:
        return st.Shared(observations_path(args), st.from_env())
    except st.StoreError as exc:  # its messages never contain the URL or the token
        log(f"WARNING: reputation store not used: {exc}")
    except Exception as exc:
        log(f"WARNING: reputation store not used: {type(exc).__name__}")
    return st.Shared(observations_path(args), None)


def share(shared: st.Shared, what: str, **kw) -> None:
    """Talk to the store. Whatever happens there, the command carries on."""
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
    """Run something whose failure must not fail the command (reports, observations)."""
    try:
        fn(*a, **kw)
    except Exception as exc:  # whatever it is, it is not worth a box
        log(f"WARNING: {what} failed: {exc!r}")


def key_dir(state_dir: Path, run_id: str) -> Path:
    """Where a run keeps its SSH key and known hosts. Never shared between runs:
    a command still holding an earlier run's endpoint must not be able to log
    in to a later run's box that happens to get the same address."""
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise StateError(f"not a run id: {run_id!r}")  # it becomes part of a path
    return Path(state_dir) / "keys" / run_id


def pinned_account(args) -> str | None:
    return getattr(args, "account_id", None) or os.environ.get("GPUBOX_ACCOUNT_ID") or None


def check_account(vast, expected: str | None) -> str | None:
    """With an account pinned: stop, before anything is rented or destroyed, if the key is
    another account's. Returns the account, or None when none is pinned."""
    if expected is None:
        return None
    actual = vast.account()
    if actual != expected:
        raise ConfigError(
            f"the API key belongs to account {actual}, not to the expected account {expected}"
        )
    return actual


def same_account(vast, state: State, account: str | None) -> None:
    """Stop if the run recorded the account it was rented on and the key is another one's."""
    if state.account and (account or vast.account()) != state.account:
        raise ConfigError(
            f"run {state.run} was rented on account {state.account}; "
            "the API key belongs to another account"
        )


def cmd_up(args) -> int:
    state_dir = Path(args.state_dir)
    spec = spec_from(args, args.image)
    policy = Policy(
        count=args.count,
        spare=args.spare,
        max_attempts=args.max_attempts,
        startup_timeout=args.startup_timeout,
        stall_timeout=args.stall_timeout,
        overall_timeout=args.overall_timeout,
    )
    if policy.attempts < policy.count:
        raise ConfigError(
            f"--max-attempts {policy.attempts} cannot rent --count {policy.count} boxes"
        )
    if spec.ttl_min * 60 < policy.overall_timeout + policy.min_lease:
        raise ConfigError(
            f"--ttl-min {spec.ttl_min} leaves no lease after a launch of up to "
            f"{policy.overall_timeout:g}s; it must cover that plus {policy.min_lease:g}s"
        )
    vast = v.Vast(api_key(args))
    with state_lock(state_dir):
        old = State.load(state_dir)  # StateError if it cannot be read
        if old is not None and not old.closed:
            raise StateError(f"run {old.run} in {state_dir} is not closed; run `gpubox down` first")
        account = check_account(vast, pinned_account(args))
        # A key that cannot list could never find its boxes again: learn that
        # before renting one.
        vast.instances()
        run_id = new_run_id(args.namespace, args.name)
        # Before anything else: the teardown step is given this, and must have
        # it whenever a box may exist. So unlike every later output, a failure
        # to write it stops the launch.
        try:
            github_output(run_id=run_id)
        except OSError as exc:
            raise ConfigError(
                f"cannot write the run id to GITHUB_OUTPUT ({exc}); nothing was rented"
            ) from None
        expiry = int(time.time()) + spec.ttl_min * 60
        meta = {
            "image": spec.image, "gpus": list(spec.gpus), "num_gpus": spec.num_gpus,
            "count": policy.count,
        }  # fmt: skip
        state = State.begin(state_dir, run_id, expiry, account, meta)
        log(f"run {run_id}")
        obs = pol.Observations(observations_path(args))
        shared = open_shared(args)
        share(shared, "before the launch")  # at most a few seconds; see store.BUDGET
        keys = key_dir(state_dir, run_id)
        prober = Prober(keys)
        launch = Launch(vast, spec, policy, state, prober)
        result = launch.run(pol.blocklist(known_rows(obs, shared), time.time()))
        if result.winners and args.ssh_config:
            advisory("writing the SSH configuration", write_ssh_entry, state, keys)
    # The outcome is decided and in the state file; nothing below may change it.
    advisory("recording observations", observe, result, spec, run_id, obs)
    if obs.added:
        advisory("queueing observations", shared.queue, obs.rows[-obs.added :])
    share(shared, "after the launch", pull=False)  # also anything left from earlier runs
    advisory("writing outputs", github_output, observations_changed=str(obs.added > 0).lower())
    advisory("writing the summary", write_summary, spec, result)
    advisory("reporting", report_result, spec, result, state)
    return 0 if result.winners else EXIT_INFRA


def boxes_of(state: State) -> list[tuple[str, dict]]:
    """(name, box) for each box of the run, in order."""
    ws = state.winners
    return [(box_name(state.run, i, len(ws)), w) for i, w in enumerate(ws)]


def write_ssh_entry(state: State, keys: Path) -> None:
    named = boxes_of(state)
    hosts = [(name, w["host"], w["port"]) for name, w in named]
    path = sshconfig.add(state.run, hosts, keys / "id_ed25519", keys / "known_hosts")
    names = " / ".join(f"`ssh {name}`" for name, _ in named)
    log(f"{names} now reach{'es' if len(named) == 1 else ''} the box(es) (entries in {path})")


def report_result(spec: Spec, result: Result, state: State) -> None:
    for iid in result.left:
        log(f"::warning::instance {iid} is not confirmed destroyed; `gpubox down` retries")
    for a in result.unanswered:
        log(
            f"::warning::attempt {a.seq} got no answer to its create. If a box was rented it "
            f"carries the tag {a.tag}: `gpubox down` looks for it, and its lease ends it"
        )
    if not result.winners:
        log(f"::error::no healthy {spec.gpu_text} box")
        return
    until = time.strftime("%H:%M UTC", time.gmtime(state.expiry))
    n = len(result.winners)
    for i, w in enumerate(result.winners):
        r = w.report
        log(
            f"READY in {result.seconds:.0f}s: {box_name(state.run, i, n)} is instance "
            f"{w.instance_id} on machine {w.machine_id} (${w.dph:.2f}/hr, {w.geo}); "
            f"{len(r.get('gpus') or [])}x {w.gpu or 'GPU'}, {r.get('cpus')} CPUs, "
            f"driver {r.get('driver')}, torch {r.get('torch')}; lease until {until}"
        )


# Provider states that mean the box is over, as opposed to missing or in transition.
DEAD_STATUSES = ("exited", "offline")


def _still_ours(state_dir: Path, run_id: str) -> bool:
    """False once this run's teardown has started."""
    try:
        state = State.load(state_dir)
    except StateError:
        return False
    return state is not None and state.run == run_id and not state.closing


def live_run(state_dir: Path) -> State:
    """The directory's run, or StateError if it has no boxes to use."""
    state = State.load(state_dir)
    if state is None or not state.winners or state.closing:
        raise StateError(f"no live box in {state_dir}; run `gpubox up` first")
    return state


def live_box(state_dir: Path, box: int | None = None) -> tuple[State, dict]:
    """The directory's run and box number `box` of it (the only one if None), or StateError."""
    state = live_run(state_dir)
    ws = state.winners
    if box is None:
        if len(ws) > 1:
            raise StateError(f"the run has {len(ws)} boxes: say which with --box 0..{len(ws) - 1}")
        box = 0
    if not 0 <= box < len(ws):
        raise StateError(f"no box {box}: the run has {len(ws)} (--box 0..{len(ws) - 1})")
    return state, ws[box]


def cmd_run(args) -> int:
    state_dir = Path(args.state_dir)
    # No lock: `down` must be able to end a run in progress.
    state, w = live_box(state_dir, args.box)
    run_id, iid = state.run, w["id"]
    exports = []
    for item in args.env:
        name, sep, value = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ConfigError(f"--env expects NAME=VALUE, got {item!r}")
        exports.append(f"export {name}={shlex.quote(value)}\n")
    script = "".join(exports) + Path(args.script).read_text()
    # Store the script first, under a name of its own, so nothing it runs can
    # swallow it from stdin and a second `run` cannot overwrite it.
    remote = 'f=$(mktemp /tmp/gpubox-script.XXXXXX) && cat > "$f" && exec bash "$f" </dev/null'
    # The key belongs to this run: it is gone once the run is torn down.
    try:
        prober = Prober(key_dir(state_dir, run_id), create=False)
    except FileNotFoundError:
        raise StateError(f"run {run_id} has no SSH key in {state_dir} (torn down?)") from None
    argv = prober.ssh((w["host"], w["port"]), remote)
    try:
        rc = subprocess.run(argv, input=script.encode(), timeout=args.timeout_min * 60).returncode
    except subprocess.TimeoutExpired:
        log(f"::error::remote script exceeded {args.timeout_min} minutes")
        return EXIT_TIMEOUT
    if rc != 255:  # 255 is ssh's own failure code (a script exiting 255 reads the same)
        return rc
    if not _still_ours(state_dir, run_id):
        log("::error::the box was destroyed by `gpubox down` while the script ran")
        return EXIT_INFRA
    status = None
    try:
        inst = v.Vast(api_key(args)).instance(iid)
        status = "gone" if inst is None else inst.get("actual_status")
    except (v.VastError, ConfigError) as exc:
        log(f"WARNING: could not ask the provider what happened: {exc}")
    log(f"::error::lost the SSH connection to the box (instance status: {status or 'unknown'})")
    # Blame the machine only on the provider's word that the box is over, and
    # only if our own teardown did not start in the meantime. An instance
    # that no longer exists was destroyed by someone (a teardown elsewhere,
    # its own timer, the reaper), which says nothing about the machine. A
    # missing or transitional status, or an API error, is not evidence either.
    if status in DEAD_STATUSES and _still_ours(state_dir, run_id):
        meta = state.meta
        asked = meta.get("gpus") or ([meta["gpu"]] if "gpu" in meta else [])  # v2: one model
        extra = {"gpu": gpu_model(w.get("gpu"), asked), "num_gpus": meta.get("num_gpus")}

        def record():
            row = pol.observation(
                "died_after_admission",
                w.get("machine"),
                w.get("host_id"),
                run_id,
                reason="died",
                gate_version=GATE_VERSION,
                **extra,
            )
            pol.Observations(observations_path(args)).add(row)
            shared = open_shared(args)
            shared.queue([row])
            share(shared, "after the run", pull=False)

        advisory("recording the observation", record)
        advisory("writing outputs", github_output, observations_changed="true")
    return EXIT_INFRA


def cmd_renew(args) -> int:
    state_dir = Path(args.state_dir)
    with state_lock(state_dir):
        state = live_run(state_dir)
        vast = v.Vast(api_key(args))
        account = check_account(vast, pinned_account(args))
        same_account(vast, state, account)  # before anything on the box is changed
        try:
            prober = Prober(key_dir(state_dir, state.run), create=False)
        except FileNotFoundError:
            raise StateError(f"run {state.run} has no SSH key in {state_dir}") from None
        expiry = int(time.time()) + args.ttl_min * 60
        renew(vast, state, prober, expiry)
    until = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(expiry))
    ids = ", ".join(str(w["id"]) for w in state.winners)
    log(f"instance(s) {ids}: lease now ends at {until}")
    return 0


def cmd_status(args) -> int:
    state = State.load(Path(args.state_dir))
    if state is None:
        print(f"no run in {args.state_dir}")
        return 0
    if state.closed:
        print(f"run {state.run}: closed")
        return 0
    until = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(state.expiry))
    left = (state.expiry - time.time()) / 60
    what = "closing" if state.closing else f"lease ends {until} ({left:.0f} min)"
    print(f"run {state.run}: {len(state.winners)} box(es), {what}")
    for i, (name, w) in enumerate(boxes_of(state)):
        gpu = f"  {w['gpu']}" if w.get("gpu") else ""
        print(f"  {i}  {name:<20} instance {w['id']:<10} root@{w['host']} -p {w['port']}{gpu}")
    return 0


def cmd_down(args) -> int:
    state_dir = Path(args.state_dir)
    run_id = args.run_id
    if run_id is not None and not RUN_ID.fullmatch(run_id):
        # Also an empty one: a job whose `up` never got as far as a run id has
        # rented nothing, and must not tear down whatever else is here.
        raise StateError(f"not a run id: {run_id!r}")
    with state_lock(state_dir):
        try:
            state = State.load(state_dir)
        except StateError as exc:
            if run_id is None:
                raise  # we cannot even tell which run this is
            log(f"WARNING: {exc}; working from the run id alone")
            state = None
        if state is None and run_id is None:
            log(f"no run in {state_dir}: nothing to destroy")
            return 0
        vast = v.Vast(api_key(args))
        account = check_account(vast, pinned_account(args))
        ours = state is not None and (run_id is None or state.run == run_id)
        if ours:
            # An empty listing on another account says nothing about this run.
            same_account(vast, state, account)
        else:
            # By tag alone. The directory's own state and keys are not touched.
            log(f"run {run_id} has no state in {state_dir}: destroying what carries its tag")
            state = State.ephemeral(run_id)
        left = down(vast, state, Policy())
        if left == []:
            log(f"run {state.run}: nothing left")
            if ours:
                # Nothing of this run is left to log in to.
                shutil.rmtree(key_dir(state_dir, state.run), ignore_errors=True)
                advisory("removing the SSH configuration entry", sshconfig.remove, state.run)
            return 0
        explain_leftovers(state, left)
    return EXIT_INFRA


def cmd_reap(args) -> int:
    if not NAMESPACE.fullmatch(args.namespace):
        raise ConfigError(f"not a namespace: {args.namespace!r}")
    vast = v.Vast(api_key(args))
    check_account(vast, pinned_account(args))
    failed = reap(vast, Policy(), args.namespace, destroy=args.destroy)
    return EXIT_INFRA if failed else 0


def cmd_search(args) -> int:
    spec = spec_from(args)
    obs = pol.Observations(observations_path(args))
    shared = open_shared(args)
    share(shared, "search")
    blocked = pol.blocklist(known_rows(obs, shared), time.time())
    found = v.Vast(api_key(args)).offers(v.offer_query(spec))
    offers = pol.select_offers(v.in_allowed_country(found, spec.exclude_countries), blocked, 2)
    print(f"{len(offers)} host(s); {len(blocked)} machine(s) blocked")
    for o in offers:
        note = blocked.get(o.get("machine_id"), {}).get("reason", "")
        print(
            f"  machine {o.get('machine_id'):>7} host {o.get('host_id'):>7} "
            f"${o['dph_total']:5.2f}/hr rel {o.get('reliability', 0):.3f} "
            f"{o.get('cpu_cores_effective', 0):4.0f} CPUs {o.get('cpu_ram', 0) / 1024:4.0f} GB "
            f"CUDA {o.get('cuda_max_good')} {o.get('geolocation')}"
            + (f"  (re-admitted: {note})" if note else "")
        )
    return 0


def number(kind, low, *, exclusive=False):
    """An argparse type: a finite number of `kind`, above (or at) `low`."""

    def parse(text):
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
        if value != value or value in (float("inf"), float("-inf")):
            raise argparse.ArgumentTypeError(f"not finite: {text!r}")
        if value < low or (exclusive and value == low):
            raise argparse.ArgumentTypeError(
                f"must be {'>' if exclusive else '>='} {low}: {text!r}"
            )
        return value

    return parse


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gpubox", description=__doc__.split("\n\n")[0])
    p.add_argument("--keys-file", default=LOCAL_KEYS_FILE, help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd", required=True)

    def account_arg(s):
        s.add_argument(
            "--account-id",
            help="the provider account this must act on; refuse a key of another one",
        )

    def state_args(s):
        s.add_argument("--state-dir", default=".gpubox")
        s.add_argument("--observations", help="JSONL of machine observations (read and appended)")

    def spec_args(s):
        s.add_argument(
            "--gpu",
            action="append",
            required=True,
            help='Vast GPU name, e.g. "RTX 5090"; repeat to accept any of several models',
        )
        s.add_argument("--num-gpus", type=number(int, 1), default=1)
        s.add_argument(
            "--max-dph-per-gpu",
            type=number(float, 0, exclusive=True),
            required=True,
            help="price cap, $/GPU/hr",
        )
        s.add_argument("--capability", help="required CUDA compute capability, e.g. 12.0")
        s.add_argument("--min-vram-mib", type=number(int, 0), default=0)
        s.add_argument("--min-cuda", type=number(float, 0), default=12.0)
        s.add_argument("--disk-gb", type=number(int, 1), default=40)
        s.add_argument(
            "--min-cpus", type=number(int, 1), help="CPUs for the box (default: max(16, 8 per GPU))"
        )
        s.add_argument(
            "--min-ram-gb", type=number(int, 1), help="GB of RAM for the box (default: 32 per GPU)"
        )
        s.add_argument("--must-reach", action="append", default=[], metavar="URL")
        s.add_argument(
            "--exclude-country",
            action="append",
            metavar="CC",
            help="skip hosts in this country (repeatable); default CN; 'none' to skip none",
        )
        state_args(s)

    s = sub.add_parser("up")
    spec_args(s)
    account_arg(s)
    s.add_argument("--image", required=True)
    s.add_argument("--namespace", required=True, help="who owns the box, e.g. quack-ci")
    s.add_argument("--name", required=True, help="this run, e.g. the CI run id")
    s.add_argument(
        "--count", type=number(int, 1), default=1, help="boxes to rent: all of them, or none"
    )
    s.add_argument("--spare", type=number(int, 0), default=DEFAULTS.spare)
    s.add_argument(
        "--ssh-config",
        action="store_true",
        help="add the box to ~/.ssh/gpubox_config under its --name, so `ssh NAME` works",
    )
    s.add_argument(
        "--max-attempts", type=number(int, 1), help="rentals in all (default: 2 * count + 4)"
    )
    s.add_argument(
        "--startup-timeout", type=number(float, 0, exclusive=True), default=DEFAULTS.startup_timeout
    )
    s.add_argument(
        "--stall-timeout",
        type=number(float, 0, exclusive=True),
        default=DEFAULTS.stall_timeout,
        help="give up on a box that is still loading and has reported nothing new for this long",
    )
    s.add_argument(
        "--overall-timeout", type=number(float, 0, exclusive=True), default=DEFAULTS.overall_timeout
    )
    s.add_argument(
        "--ttl-min", type=number(int, 1), default=90, help="the box destroys itself after this"
    )
    s.add_argument(
        "--registry-user",
        metavar="USER",
        help=f"log in to the image's registry as USER, with the token in ${REGISTRY_TOKEN}",
    )
    s.add_argument(
        "--gate-script",
        metavar="FILE",
        help="run this on each box after the health gate passes; exit 0 admits the box",
    )
    s.add_argument(
        "--gate-timeout",
        type=number(float, 0, exclusive=True),
        default=600,
        help="seconds the gate script may take",
    )
    s.set_defaults(fn=cmd_up)

    s = sub.add_parser("status")
    state_args(s)
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("run")
    state_args(s)
    s.add_argument("--box", type=number(int, 0), help="which box of the run (0, 1, ...)")
    s.add_argument("--script", required=True)
    s.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    s.add_argument("--timeout-min", type=number(float, 0, exclusive=True), default=60)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("renew")
    state_args(s)
    account_arg(s)
    s.add_argument(
        "--ttl-min", type=number(int, 1), required=True, help="the lease ends this long from now"
    )
    s.set_defaults(fn=cmd_renew)

    s = sub.add_parser("down")
    state_args(s)
    account_arg(s)
    s.add_argument(
        "--run-id",
        help="the run to tear down (printed by `up`); without it, the directory's current run. "
        "For a run this directory has no state of, its boxes are found by their tag",
    )
    s.set_defaults(fn=cmd_down)

    s = sub.add_parser("reap")
    account_arg(s)
    s.add_argument("--namespace", required=True, help="whose boxes: those tagged for it")
    s.add_argument("--destroy", action="store_true", help="destroy; without it, only report")
    s.set_defaults(fn=cmd_reap)

    s = sub.add_parser("search")
    spec_args(s)
    s.set_defaults(fn=cmd_search)

    args = p.parse_args(argv)

    # Turn SIGTERM into an exception so `finally` blocks destroy what we rented.
    def on_term(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    try:
        return args.fn(args)
    except (ConfigError, StateError) as exc:
        log(f"::error::{exc}")
        return EXIT_CONFIG
    except v.VastError as exc:
        log(f"::error::Vast API: {exc}")
        return EXIT_INFRA
    except OSError as exc:
        log(f"::error::state directory: {exc}")
        return EXIT_INFRA
    except KeyboardInterrupt:
        # `up` has tried to roll back by now; `down` finishes what it could not.
        log("::error::interrupted")
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
