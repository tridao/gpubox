"""gpubox: rent a verified GPU box from Vast.ai, and make sure it is destroyed.

    up      rent a box and wait until it is reachable and passes the health gate
    run     run a local script on that box over SSH, streaming its output
    down    destroy everything this state directory's run rented
    reap    report (or, in an exclusive namespace, destroy) boxes that outlived any run
    search  show the offers `up` would try (rents nothing)

Exit codes: 0 ok; 2 configuration (image cannot be pulled, no API key, state
directory in use, bad arguments); 75 infrastructure (no healthy box, box lost
mid-run, teardown not confirmed); 124 script timeout. `run` otherwise returns
the script's own exit code, except that 255 is taken for a lost connection.

The API key is read from VASTAI_API_KEY. `--account-id` (or GPUBOX_ACCOUNT_ID)
pins the provider account: `up`, `down` and `reap` then refuse to act with a
key that belongs to another one.
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
from . import vast as v
from .gate import GATE_VERSION, Prober
from .launch import ConfigError, Launch, Policy, Result, Spec, down, log, observe, reap
from .state import NAMESPACE, RUN_ID, Snapshot, StateError, new_run_id, state_lock

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
    w = result.winner
    head = (
        f"healthy box after {result.seconds:.0f}s: machine {w.machine_id}, ${w.dph:.2f}/hr"
        if w
        else f"no healthy box after {result.seconds:.0f}s"
    )
    lines = [
        f"### Vast launch: {spec.num_gpus}x {spec.gpu}",
        "",
        head,
        "",
        "| machine | host | location | $/hr | result | seconds | detail |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in result.attempts:
        outcome = "healthy" if a is w else (a.reason or "not blamed")
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
        gpu=args.gpu,
        num_gpus=args.num_gpus,
        capability=cap,
        min_vram_mib=args.min_vram_mib,
        min_cuda=args.min_cuda,
        disk_gb=args.disk_gb,
        max_dph_per_gpu=args.max_dph_per_gpu,
        must_reach=tuple(args.must_reach),
        exclude_countries=countries(args),
        ttl_min=getattr(args, "ttl_min", 90),
    )


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


def check_account(vast, expected: str | None) -> str:
    """Whose key this is. Stops, before anything is rented or destroyed, if an account is
    pinned and the key is another one's."""
    actual = vast.account()
    if expected is not None and actual != expected:
        raise ConfigError(
            f"the API key belongs to account {actual}, not to the expected account {expected}"
        )
    return actual


def cmd_up(args) -> int:
    state_dir = Path(args.state_dir)
    spec = spec_from(args, args.image)
    policy = Policy(
        spare=args.spare,
        max_attempts=args.max_attempts,
        startup_timeout=args.startup_timeout,
        stall_timeout=args.stall_timeout,
        overall_timeout=args.overall_timeout,
    )
    if spec.ttl_min * 60 < policy.overall_timeout + policy.min_lease:
        raise ConfigError(
            f"--ttl-min {spec.ttl_min} leaves no lease after a launch of up to "
            f"{policy.overall_timeout:g}s; it must cover that plus {policy.min_lease:g}s"
        )
    vast = v.Vast(api_key(args))
    with state_lock(state_dir):
        old = Snapshot.load(state_dir)  # StateError if it is damaged
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
        meta = {"image": spec.image, "gpu": spec.gpu, "num_gpus": spec.num_gpus}
        snap = Snapshot.begin(state_dir, run_id, expiry, account, meta)
        log(f"run {run_id}")
        obs = pol.Observations(observations_path(args))
        prober = Prober(key_dir(state_dir, run_id))
        launch = Launch(vast, spec, policy, snap, prober)
        result = launch.run(pol.blocklist(obs.rows, time.time()))
    # The outcome is decided and in the snapshot; nothing below may change it.
    advisory("recording observations", observe, result, spec, run_id, obs)
    advisory("writing outputs", github_output, observations_changed=str(obs.added > 0).lower())
    advisory("writing the summary", write_summary, spec, result)
    advisory("reporting", report_result, spec, result)
    return 0 if result.winner else EXIT_INFRA


def report_result(spec: Spec, result: Result) -> None:
    for iid in result.left:
        log(f"::warning::instance {iid} is not confirmed destroyed; `gpubox down` retries")
    for a in result.unseen:
        log(f"::warning::attempt {a.seq} got no answer to its create (tag {a.tag}); see `down`")
    w = result.winner
    if not w:
        log(f"::error::no healthy {spec.num_gpus}x {spec.gpu} box")
        return
    r = w.report
    log(
        f"READY in {result.seconds:.0f}s: instance {w.instance_id} on machine {w.machine_id} "
        f"(${w.dph:.2f}/hr, {w.geo}); {len(r.get('gpus') or [])}x GPU, {r.get('cpus')} CPUs, "
        f"driver {r.get('driver')}, torch {r.get('torch')}"
    )


# Provider states that mean the box is over, as opposed to missing or in transition.
DEAD_STATUSES = ("exited", "offline")


def _still_ours(state_dir: Path, run_id: str, instance_id: int) -> bool:
    """False once this run's teardown has started, or the box is known to be gone."""
    try:
        snap = Snapshot.load(state_dir)
    except StateError:
        return False
    return (
        snap is not None
        and snap.run == run_id
        and not snap.closing
        and snap.status(instance_id) == "live"
    )


def cmd_run(args) -> int:
    state_dir = Path(args.state_dir)
    snap = Snapshot.load(state_dir)  # no lock: `down` must be able to end a run in progress
    w = snap.winner if snap is not None else None
    if not w or snap.closing or snap.status(w["id"]) != "live":
        raise StateError(f"no live box in {state_dir}; run `gpubox up` first")
    run_id, iid = snap.run, w["id"]
    box = snap.attempts().get(w["attempt"], {})
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
    if not _still_ours(state_dir, run_id, iid):
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
    if status in DEAD_STATUSES and _still_ours(state_dir, run_id, iid):
        meta = snap.meta
        extra = {k: meta[k] for k in ("gpu", "num_gpus") if k in meta}

        def record():
            row = pol.observation(
                "died_after_admission",
                box.get("machine"),
                box.get("host"),
                run_id,
                reason="died",
                gate_version=GATE_VERSION,
                **extra,
            )
            pol.Observations(observations_path(args)).add(row)
            github_output(observations_changed="true")

        advisory("recording the observation", record)
    return EXIT_INFRA


def cmd_down(args) -> int:
    state_dir = Path(args.state_dir)
    run_id = args.run_id
    if run_id is not None and not RUN_ID.fullmatch(run_id):
        # Also an empty one: a job whose `up` never got as far as a run id has
        # rented nothing, and must not tear down whatever else is here.
        raise StateError(f"not a run id: {run_id!r}")
    with state_lock(state_dir):
        damaged = None
        try:
            snap = Snapshot.load(state_dir)
        except StateError as exc:
            if run_id is None:
                raise  # T3: we cannot even tell which run this is
            snap, damaged = None, exc
        if snap is None and run_id is None:
            log(f"no run in {state_dir}: nothing to destroy")
            return 0
        vast = v.Vast(api_key(args))
        account = check_account(vast, pinned_account(args))
        ours = snap is not None and (run_id is None or snap.run == run_id)
        if ours:
            if snap.account is not None and account != snap.account:
                # T3: an empty listing on another account says nothing about this run.
                raise ConfigError(
                    f"run {snap.run} was rented on account {snap.account}; "
                    "the API key belongs to another account"
                )
            left = down(vast, snap, Policy(), abandon_unseen=args.abandon_unseen)
            if not left:
                # Nothing of this run is left to log in to.
                shutil.rmtree(key_dir(state_dir, snap.run), ignore_errors=True)
        else:
            # T4: by tag alone. The directory's own snapshot and keys are not touched.
            log(f"run {run_id} has no snapshot in {state_dir}: destroying by tag only")
            left = down(vast, Snapshot.ephemeral(run_id), Policy(), abandon_unseen=False)
            if damaged is not None:
                log(f"::error::{damaged}; the ids it held could not be checked")
                left += 1
    if left:
        log(f"::error::{left} thing(s) unresolved; see above, then run `gpubox down` again")
    return EXIT_INFRA if left else 0


def cmd_reap(args) -> int:
    if args.exclusive == (args.namespace is not None):
        raise ConfigError("reap needs exactly one of --namespace NS and --exclusive")
    if args.namespace is not None and not NAMESPACE.fullmatch(args.namespace):
        raise ConfigError(f"not a namespace: {args.namespace!r}")
    expected = pinned_account(args)
    if args.exclusive and expected is None:
        raise ConfigError("reap --exclusive needs --account-id (or GPUBOX_ACCOUNT_ID)")
    if args.exclusive and args.max_age_min is None:
        raise ConfigError("reap --exclusive needs --max-age-min")
    if args.namespace is not None and args.max_age_min is not None:
        raise ConfigError(
            "--max-age-min has no effect with --namespace: a tagged box is reaped when the "
            "lease in its tag has ended"
        )
    vast = v.Vast(api_key(args))
    check_account(vast, expected)  # R1
    failed = reap(
        vast,
        Policy(),
        namespace=args.namespace,
        max_age_min=args.max_age_min,
        destroy=args.destroy,
    )
    return EXIT_INFRA if failed else 0


def cmd_search(args) -> int:
    spec = spec_from(args)
    obs = pol.Observations(observations_path(args))
    blocked = pol.blocklist(obs.rows, time.time())
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
        s.add_argument("--gpu", required=True, help='Vast GPU name, e.g. "RTX 5090"')
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
    s.add_argument("--spare", type=number(int, 0), default=DEFAULTS.spare)
    s.add_argument("--max-attempts", type=number(int, 1), default=DEFAULTS.max_attempts)
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
    s.set_defaults(fn=cmd_up)

    s = sub.add_parser("run")
    state_args(s)
    s.add_argument("--script", required=True)
    s.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    s.add_argument("--timeout-min", type=number(float, 0, exclusive=True), default=60)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("down")
    state_args(s)
    account_arg(s)
    s.add_argument(
        "--run-id",
        help="the run to tear down (printed by `up`); without it, the directory's current run. "
        "For a run this directory has no snapshot of, its boxes are destroyed by tag alone",
    )
    s.add_argument(
        "--abandon-unseen",
        action="store_true",
        help="a create that got no answer, and whose box is still not visible 60 s later, "
        "is left to the box's timer and the reaper instead of failing the teardown",
    )
    s.set_defaults(fn=cmd_down)

    s = sub.add_parser("reap")
    account_arg(s)
    s.add_argument("--namespace", help="only boxes tagged for this namespace, once expired")
    s.add_argument(
        "--exclusive",
        action="store_true",
        help="every instance of the account: asserts that all of them are disposable. "
        "Needs --account-id and --max-age-min",
    )
    s.add_argument("--max-age-min", type=number(float, 0, exclusive=True))
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
        # `up` has tried to roll back by now; what is left is in the snapshot, for `down`.
        log("::error::interrupted")
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
