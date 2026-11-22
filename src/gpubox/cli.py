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

The API key is read from VASTAI_API_KEY.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import policy as pol
from . import vast as v
from .gate import GATE_VERSION, Prober
from .journal import Journal, StateError, new_run_id, state_lock
from .launch import ConfigError, Launch, Policy, Result, Spec, down, log, observe, reap

EXIT_CONFIG = 2
EXIT_INFRA = 75
EXIT_TIMEOUT = 124
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
        min_vram_mib=int(args.min_vram_gb * 1024),
        min_cuda=args.min_cuda,
        disk_gb=args.disk_gb,
        max_dph_per_gpu=args.max_dph_per_gpu,
        must_reach=tuple(args.must_reach),
        ttl_min=getattr(args, "ttl_min", 90),
    )


def observations_path(args) -> Path:
    if args.observations:
        return Path(args.observations)
    return Path(args.state_dir) / "observations.jsonl"


def advisory(what: str, fn, *a, **kw) -> None:
    """Run something whose failure must not fail the command (reports, reputation)."""
    try:
        fn(*a, **kw)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        log(f"WARNING: {what} failed: {exc!r}")


def cmd_up(args) -> int:
    state_dir = Path(args.state_dir)
    spec = spec_from(args, args.image)
    policy = Policy(
        spare=args.spare,
        max_attempts=args.max_attempts,
        startup_timeout=args.startup_timeout,
        overall_timeout=args.overall_timeout,
    )
    key = api_key(args)
    with state_lock(state_dir):
        journal = Journal(state_dir)
        if journal.exists():
            old = journal.state()
            if old.live():
                raise StateError(
                    f"run {old.run_id} in {state_dir} still has boxes; run `gpubox down` first"
                )
            journal.rotate()
        run_id = new_run_id(args.namespace, args.name)
        journal.append(
            "run",
            run_id=run_id,
            namespace=args.namespace,
            image=spec.image,
            gpu=spec.gpu,
            num_gpus=spec.num_gpus,
        )
        github_output(run_id=run_id)
        log(f"run {run_id}")
        obs = pol.Observations(observations_path(args))
        launch = Launch(v.Vast(key), spec, policy, journal, Prober(state_dir, fresh=True), run_id)
        result = launch.run(pol.blocklist(obs.rows, time.time()))
    # From here on the box is ours and recorded; nothing below may fail the launch.
    advisory("recording observations", observe, result, spec, run_id, obs)
    advisory("writing outputs", github_output, observations_changed=str(obs.added > 0).lower())
    advisory("writing the summary", write_summary, spec, result)
    for a in result.unresolved:
        where = f"instance {a.instance_id}" if a.instance_id else f"label {a.label}"
        log(f"::warning::attempt {a.seq} may have left a box ({where}); `gpubox down` retries")
    w = result.winner
    if not w:
        log(f"::error::no healthy {spec.num_gpus}x {spec.gpu} box")
        return EXIT_INFRA
    r = w.report
    log(
        f"READY in {result.seconds:.0f}s: instance {w.instance_id} on machine {w.machine_id} "
        f"(${w.dph:.2f}/hr, {w.geo}); {len(r.get('gpus', []))}x GPU, {r.get('cpus')} CPUs, "
        f"driver {r.get('driver')}, torch {r.get('torch')}"
    )
    return 0


# Provider states that mean the box is over, as opposed to missing or in transition.
DEAD_STATUSES = ("exited", "offline")


def _ours_to_blame(journal: Journal, run_id: str, instance_id: int) -> bool:
    """False if this run (or its teardown) is why the box went away."""
    state = journal.state()
    if state.run_id != run_id or state.closed:
        return False
    attempt = state.by_instance(instance_id)
    return bool(attempt) and attempt.status == "created"


def cmd_run(args) -> int:
    state_dir = Path(args.state_dir)
    journal = Journal(state_dir)
    state = journal.state()
    w = state.winner
    box = state.by_instance(w["instance_id"]) if w else None
    if state.closed or not box or box.status != "created":
        raise StateError(f"no live box in {state_dir}; run `gpubox up` first")
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
    argv = Prober(state_dir).ssh((w["host"], w["port"]), remote)
    try:
        rc = subprocess.run(argv, input=script.encode(), timeout=args.timeout_min * 60).returncode
    except subprocess.TimeoutExpired:
        log(f"::error::remote script exceeded {args.timeout_min} minutes")
        return EXIT_TIMEOUT
    if rc != 255:  # 255 is ssh's own failure code (a script exiting 255 reads the same)
        return rc
    iid = w["instance_id"]
    if not _ours_to_blame(journal, state.run_id, iid):
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
    # only if our own teardown did not start in the meantime. A missing or
    # transitional status, or an API error, is not evidence.
    if (status == "gone" or status in DEAD_STATUSES) and _ours_to_blame(journal, state.run_id, iid):
        meta = state.meta
        extra = {k: meta[k] for k in ("gpu", "num_gpus") if k in meta}

        def record():
            pol.Observations(observations_path(args)).add(
                pol.observation(
                    "died_after_admission",
                    box.machine_id,
                    box.host_id,
                    state.run_id,
                    reason="died",
                    gate_version=GATE_VERSION,
                    **extra,
                )
            )
            github_output(observations_changed="true")

        advisory("recording the observation", record)
    return EXIT_INFRA


def cmd_down(args) -> int:
    state_dir = Path(args.state_dir)
    with state_lock(state_dir):
        journal = Journal(state_dir, run_id=args.run_id)
        if not journal.exists():
            log(f"no run in {state_dir}: nothing to destroy")
            return 0
        left = down(v.Vast(api_key(args)), journal, Policy())
    if left:
        log(f"::error::{left} attempt(s) unresolved; run `gpubox down` again")
    return EXIT_INFRA if left else 0


def cmd_reap(args) -> int:
    failed = reap(
        v.Vast(api_key(args)),
        args.namespace,
        args.max_age_min,
        destroy=args.destroy and not args.dry_run,
        legacy=args.legacy_pattern,
    )
    return EXIT_INFRA if failed else 0


def cmd_search(args) -> int:
    spec = spec_from(args)
    obs = pol.Observations(observations_path(args))
    blocked = pol.blocklist(obs.rows, time.time())
    found = [o for o in v.Vast(api_key(args)).offers(v.offer_query(spec)) if o.get("dph_total")]
    offers = pol.select_offers(found, blocked, 2)
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
        s.add_argument("--min-vram-gb", type=number(float, 0), default=0)
        s.add_argument("--min-cuda", type=number(float, 0), default=12.0)
        s.add_argument("--disk-gb", type=number(int, 1), default=40)
        s.add_argument("--must-reach", action="append", default=[], metavar="URL")
        state_args(s)

    s = sub.add_parser("up")
    spec_args(s)
    s.add_argument("--image", required=True)
    s.add_argument("--namespace", required=True, help="who owns the box, e.g. quack-ci")
    s.add_argument("--name", required=True, help="this run, e.g. the CI run id")
    s.add_argument("--spare", type=number(int, 0), default=DEFAULTS.spare)
    s.add_argument("--max-attempts", type=number(int, 1), default=DEFAULTS.max_attempts)
    s.add_argument(
        "--startup-timeout", type=number(float, 0, exclusive=True), default=DEFAULTS.startup_timeout
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
    s.add_argument(
        "--run-id",
        help="the run to tear down (printed by `up`); without it, the directory's current run",
    )
    s.set_defaults(fn=cmd_down)

    s = sub.add_parser("reap")
    s.add_argument("--namespace", required=True)
    s.add_argument("--max-age-min", type=number(float, 0), required=True)
    s.add_argument(
        "--destroy",
        action="store_true",
        help="destroy stale boxes; asserts every box in this namespace is disposable",
    )
    s.add_argument("--dry-run", action="store_true", help="report only, even with --destroy")
    s.add_argument(
        "--legacy-pattern",
        metavar="REGEX",
        help="also match labels of an older scheme (full match; never a current-scheme label)",
    )
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


if __name__ == "__main__":
    sys.exit(main())
