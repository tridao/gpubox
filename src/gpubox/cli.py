"""gpubox: rent a verified GPU box from Vast.ai or a Slurm cluster, and make sure it is
destroyed.

    up      rent boxes and wait until they are reachable and pass the health gate
    status  show the run's boxes: names, addresses, lease
    run     start a local script on a box, detached, and follow its output
    logs    print a job's output (-f: until it ends)
    wait    wait for a job to end
    jobs    list the jobs on a box
    kill    stop a job
    ssh     a shell on a box, or a command run there
    renew   move the end of the boxes' lease
    down    destroy everything this state directory's run rented
    reap    report, or destroy, boxes of a namespace whose lease has ended
    search  show the offers `up` would try (rents nothing)

Exit codes: 0 ok; 2 configuration (image cannot be pulled, no API key, state
directory in use, bad arguments); 75 infrastructure (no healthy box, box lost
mid-run, teardown not confirmed); 124 script timeout; 130 interrupted (a job
goes on regardless). `run`, `wait` and `logs -f` otherwise return the job's own
exit code.

The API key is read from VASTAI_API_KEY. `--account-id` (or GPUBOX_ACCOUNT_ID)
pins the provider account: the commands then refuse to act with a key that
belongs to another one.

`--provider runpod` rents RunPod pods instead (RUNPOD_API_KEY, providers/runpod.py),
`--provider shadeform` VMs on Shadeform's clouds (SHADEFORM_API_KEY,
providers/shadeform.py and vmbox.py), and `--provider slurm --login HOST` takes boxes from a Slurm cluster: a batch job
holding an allocation, reached with `ssh HOST srun --jobid ...` (providers/slurm.py), whose
account is `user@cluster`. `up`, `search`, `down --run-id` and `reap` take these
options; the other commands read them from the run's state file. Keys are read from
the environment, else from ~/.config/gpu-cloud-query/keys.env.

If GPUBOX_STORE_URL and GPUBOX_STORE_TOKEN are set, observations of machines
are shared through that reputation store (`store/`). It is advisory: it costs
a command at most a few seconds and never fails it.
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

from . import api
from . import job
from . import providers
from .providers import vast as v
from .api import LOCAL_KEYS_FILE, Prefixed, advisory, key_dir  # noqa: F401 (part of the CLI's surface)
from .errors import ConfigError, GpuboxError, InfraError, NoBoxes, TeardownIncomplete
from .launch import Policy, Result, Spec, log
from .state import State, StateError, box_name

EXIT_CONFIG = 2
EXIT_INFRA = 75
EXIT_TIMEOUT = 124
EXIT_INTERRUPTED = 130
DEFAULTS = Policy()


def settings(args) -> api.Settings:
    check_provider_args(args)
    options = None
    if is_slurm(args):
        options = {
            "login": args.login, "partition": args.partition, "account": args.slurm_account,
            "qos": args.qos, "constraint": args.constraint, "sbatch_args": list(args.sbatch_arg),
        }  # fmt: skip
    elif getattr(args, "provider", None) == "runpod" and args.runpod_cloud:
        options = {"cloud": args.runpod_cloud.upper()}
    return api.Settings(
        state_dir=Path(getattr(args, "state_dir", ".gpubox")),
        account_id=getattr(args, "account_id", None),
        keys_file=args.keys_file,
        observations=getattr(args, "observations", None),
        # Said at once, not when the command ends: a CI step may be cancelled first.
        on_observation=lambda: github_output(observations_changed="true"),
        provider=getattr(args, "provider", "vast"),
        options=options,
    )


def is_slurm(args) -> bool:
    return getattr(args, "provider", None) == "slurm"


# The flags of features not every provider has (providers.FEATURES), as argparse names them.
FEATURE_FLAGS = {
    "image": ("--image", "image"),
    "price_cap": ("--max-dph-per-gpu", "max_dph_per_gpu"),
    "interruptible": ("--interruptible", "interruptible"),
    "registry_login": ("--registry-user", "registry_user"),
    "exclude_country": ("--exclude-country", "exclude_country"),
    "ssh_config": ("--ssh-config", "ssh_config"),
}
# Options of one provider only: (flag, argparse name, provider).
OWN_FLAGS = [
    ("--login", "login", "slurm"), ("--partition", "partition", "slurm"),
    ("--slurm-account", "slurm_account", "slurm"), ("--qos", "qos", "slurm"),
    ("--constraint", "constraint", "slurm"), ("--sbatch-arg", "sbatch_arg", "slurm"),
    ("--runpod-cloud", "runpod_cloud", "runpod"),
]  # fmt: skip


def check_provider_args(args) -> None:
    """Options that the chosen provider does not have, and those it needs."""
    kind = providers.kind(getattr(args, "provider", None) or "vast")
    given = lambda name: getattr(args, name, None) not in (None, False, [])
    others = [f for f, name, owner in OWN_FLAGS if owner != kind.name and given(name)]
    if others:
        raise ConfigError(f"{', '.join(others)}: not for {kind.name}")
    if not hasattr(args, "num_gpus"):  # not a command that rents (down, reap)
        return
    refused = [f for feat, (f, name) in FEATURE_FLAGS.items() if feat not in kind.features and given(name)]  # fmt: skip
    if refused:
        raise ConfigError(f"{', '.join(refused)}: not on {kind.name}")
    needed = [
        f for feat, (f, name) in FEATURE_FLAGS.items()
        if feat in ("image", "price_cap") and feat in kind.features and hasattr(args, name)
        and not given(name)
    ]  # fmt: skip
    if not args.gpu and "any_gpu" not in kind.features:
        needed.insert(0, "--gpu")
    if needed:
        raise ConfigError(f"{kind.name} needs {', '.join(needed)}")
    if is_slurm(args) and not args.login:
        raise ConfigError(
            "--provider slurm needs --login HOST: the cluster's login host, as `ssh HOST` "
            "reaches it without a password prompt"
        )


def github_output(**values) -> None:
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a") as f:
            f.writelines(f"{k}={v}\n" for k, v in values.items())


def write_summary(spec: Spec, result: Result, kind=None) -> None:
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
        f"### {(kind.name if kind else 'vast').capitalize()} launch: {spec.gpu_text}",
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
    try:
        return _spec_from(args, image)
    except ValueError as exc:  # what Spec itself refuses
        raise ConfigError(str(exc)) from None


def _spec_from(args, image: str = "") -> Spec:
    cap = None
    if args.capability:
        if not re.fullmatch(r"\d+\.\d+", args.capability):
            raise ConfigError(f"--capability expects MAJOR.MINOR, got {args.capability!r}")
        cap = tuple(int(x) for x in args.capability.split("."))
    return Spec(
        image=image,
        gpus=tuple(args.gpu or ()),
        min_cpus=args.min_cpus,
        min_ram_gb=args.min_ram_gb,
        num_gpus=args.num_gpus,
        capability=cap,
        min_vram_mib=args.min_vram_mib,
        min_cuda=args.min_cuda,
        disk_gb=args.disk_gb,
        max_dph_per_gpu=args.max_dph_per_gpu or 0.0,
        must_reach=tuple(args.must_reach),
        exclude_countries=countries(args),
        gate_script=gate_script(args),
        **registry_login(args),
        interruptible=getattr(args, "interruptible", False),
        bid_markup=getattr(args, "bid_markup", 1.25),
        gate_timeout=getattr(args, "gate_timeout", 600),
        ttl_min=getattr(args, "ttl_min", None) or 90,
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


# ------------------------------------------------------------------ commands


def cmd_up(args) -> int:
    check_provider_args(args)
    spec = spec_from(args, args.image or "")
    kind = providers.kind(args.provider)
    spare = args.spare if args.spare is not None else kind.default_spare
    policy = Policy(
        count=args.count,
        spare=spare,
        max_attempts=args.max_attempts,
        startup_timeout=args.startup_timeout,
        stall_timeout=args.stall_timeout,
        overall_timeout=args.overall_timeout,
    )

    def publish(run_id: str) -> None:
        # The teardown step is given this, and must have it whenever a box may exist.
        # So unlike every later output, a failure to write it stops the launch.
        try:
            github_output(run_id=run_id)
        except OSError as exc:
            raise ConfigError(
                f"cannot write the run id to GITHUB_OUTPUT ({exc}); nothing was rented"
            ) from None

    def reported(result: Result, run: api.Run) -> None:
        changed = str(run.observations_changed).lower()
        advisory("writing outputs", github_output, observations_changed=changed)
        advisory("writing the summary", write_summary, spec, result, run.kind)
        advisory("reporting", report_result, spec, result, run.id, run.expiry, run.kind)

    try:
        api.up(
            spec, namespace=args.namespace, name=args.name, count=args.count, policy=policy,
            ssh_config=args.ssh_config, on_run_id=publish, on_result=reported,
            teardown_on_failure=False,  # CI runs `down` anyway, within its own deadline
            settings=settings(args),
        )  # fmt: skip
    except NoBoxes:
        return EXIT_INFRA  # report_result has said why
    return 0


def report_result(spec: Spec, result: Result, run_id: str, expiry: int, kind=None) -> None:
    noun = kind.noun if kind else "instance"
    for iid in result.left:
        log(f"::warning::{noun} {iid} is not confirmed destroyed; `gpubox down` retries")
    for a in result.unanswered:
        log(
            f"::warning::attempt {a.seq} got no answer to its create. If a box was rented it "
            f"carries the tag {a.tag}: `gpubox down` looks for it, and its lease ends it"
        )
    if not result.winners:
        log(f"::error::no healthy {spec.gpu_text} box")
        return
    until = time.strftime("%H:%M UTC", time.gmtime(expiry))
    n = len(result.winners)
    for i, w in enumerate(result.winners):
        r = w.report
        if kind is not None and kind.access == "srun":
            where = f"job {w.instance_id} on node {w.machine_id}"
        elif kind is not None and not kind.observes_machines:
            where = f"{noun} {w.instance_id} (${w.dph:.2f}/hr, {w.geo})"
        else:
            where = f"instance {w.instance_id} on machine {w.machine_id} (${w.dph:.2f}/hr, {w.geo})"
        gpus = r.get("gpus") or []
        model = w.gpu or (gpus[0] if gpus and len(set(gpus)) == 1 else "GPU")
        log(
            f"READY in {result.seconds:.0f}s: {box_name(run_id, i, n)} is {where}; "
            f"{len(gpus)}x {model}, {r.get('cpus')} CPUs, "
            f"driver {r.get('driver')}, torch {r.get('torch')}; lease until {until}"
        )


def chosen(args) -> tuple[api.Run, list[api.Box]]:
    """The run and the boxes a command is for: `--box I`, the only box, or with `--all`
    every box."""
    run = api.attach(settings=settings(args))
    if getattr(args, "all", False):
        return run, run.boxes
    return run, [run.box(args.box)]


def outcome_code(o: api.Outcome) -> int:
    """The exit code for how waiting on a job ended, said in words where it is not the
    job's own."""
    if o.error is not None:
        if isinstance(o.error, GpuboxError):
            log(f"::error::{o.error}")
            return EXIT_CONFIG if isinstance(o.error, (ConfigError, StateError)) else EXIT_INFRA
        log(f"::error::following job {o.job.id} on {o.job.box.name}: {o.error!r}")
        return EXIT_INFRA
    if o.code is None:  # still running
        return 0
    if o.code == EXIT_TIMEOUT:
        log(f"::error::job {o.job.id} on {o.job.box.name} ran out of time (--timeout-min)")
    return o.code


def wait_jobs(args, run: api.Run, jobs_: list[api.Job], *, quiet=False, follow=True) -> int:
    """Follow the jobs and combine how they ended: a lost box first, then a timeout,
    then the first failure, else 0."""
    try:
        outcomes = api.wait_all(jobs_, quiet=quiet, follow=follow, reconnect_min=args.reconnect_min)
    except KeyboardInterrupt:
        j = jobs_[0]
        where = "" if run.meta.get("count", 1) == 1 else f" --box {j.box.index}"
        log(f"job {j.id} carries on: `gpubox logs -f{where} {j.id}`, "
            f"`gpubox kill{where} {j.id}` to stop it")  # fmt: skip
        return EXIT_INTERRUPTED
    codes = [outcome_code(o) for o in outcomes]
    for wanted in (EXIT_INFRA, EXIT_TIMEOUT):
        if wanted in codes:
            return wanted
    return next((c for c in codes if c != 0), 0)


def cmd_run(args) -> int:
    # No lock: `down` must be able to end a run in progress.
    run, boxes = chosen(args)
    try:
        script = Path(args.script).read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read --script {args.script}: {exc}") from None
    try:
        env = job.env_from(args.env, args.env_file)
    except job.JobError as exc:
        raise ConfigError(str(exc)) from None
    job_id = job.new_job_id()
    started, failed = [], False
    for box in boxes:
        try:
            started.append(
                box.start(
                    script,
                    env,
                    timeout_min=args.timeout_min,
                    destroy_after=args.destroy_after,
                    job_id=job_id,
                )  # fmt: skip
            )
        except InfraError as exc:  # our own mistakes (ConfigError, StateError) end it at once
            log(f"::error::{exc}")
            failed = True
    advisory("writing outputs", github_output, job_id=job_id)
    if failed:
        return EXIT_INFRA
    if args.detach:
        print(job_id)
        return 0
    return wait_jobs(args, run, started)


def cmd_logs(args) -> int:
    run, (box,) = chosen(args)
    return wait_jobs(args, run, [box.job(args.job)], follow=args.follow)


def cmd_wait(args) -> int:
    run, (box,) = chosen(args)
    return wait_jobs(args, run, [box.job(args.job)], quiet=True)


def cmd_jobs(args) -> int:
    run, (box,) = chosen(args)
    for job_id, status in box.jobs():
        print(job_id, status)
    return 0


def cmd_kill(args) -> int:
    run, (box,) = chosen(args)
    j = box.job(args.job)
    log(f"job {j.id}: {j.kill()}")
    return 0


def cmd_ssh(args) -> int:
    """A shell on the box, or a command run there with the container's environment."""
    run, (box,) = chosen(args)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if command:
        remote = job.LOAD_ENV + "exec " + " ".join(shlex.quote(c) for c in command)
        argv = box.ssh_argv(remote, tty=sys.stdin.isatty())
    else:
        argv = box.ssh_argv(None, tty=True)  # a login shell, which loads the environment itself
    rc = subprocess.run(argv).returncode
    if box._refreshed(rc):  # not run again: the command may have run before the drop
        log(f"{box.name} answers at a new address now; run the command again")
    return rc


def cmd_renew(args) -> int:
    run = api.attach(settings=settings(args))
    expiry = run.renew(args.ttl_min)
    until = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(expiry))
    ids = ", ".join(str(b.instance_id) for b in run.boxes)
    log(f"{run.kind.noun}(s) {ids}: lease now ends at {until}")
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
    ws = state.winners
    print(f"run {state.run}: {len(ws)} box(es), {what}")
    for i, w in enumerate(ws):
        name = box_name(state.run, i, len(ws))
        gpu = f"  {w['gpu']}" if w.get("gpu") else ""
        if api.is_slurm(state.meta):
            # Its host is the login host and its port the job id (api.prober_for).
            node = f" on {w['machine']}" if w.get("machine") else ""
            where = f"job {w['id']:<10}{node}  (ssh {w['host']}; `gpubox ssh --box {i}`)"
        else:
            noun = providers.kind_of(state.meta).noun
            user = w.get("user") or "root"
            where = f"{noun} {w['id']:<10} {user}@{w['host']} -p {w['port']}"
        print(f"  {i}  {name:<20} {where}{gpu}")
    return 0


def cmd_down(args) -> int:
    try:
        api.down(run_id=args.run_id, policy=Policy(), settings=settings(args))
    except TeardownIncomplete:
        return EXIT_INFRA  # explain_leftovers has said what is left and what to do
    return 0


def cmd_reap(args) -> int:
    failed = api.reap(
        args.namespace, destroy=args.destroy, policy=Policy(), settings=settings(args)
    )
    return EXIT_INFRA if failed else 0


def cmd_search(args) -> int:
    if args.provider != "vast" and args.ttl_min is not None:
        raise ConfigError("search --ttl-min is only supported on Vast")
    spec = spec_from(args)
    offers, blocked = api.search(spec, settings=settings(args))
    if args.provider != "vast":
        print(f"{len(offers)} offer(s)")
        for o in offers:
            what = str(o.get("id", "")).split("#")[0]  # the offer, once per race slot
            print(f"  ${o['dph_total']:6.2f}/hr  {o.get('gpu_name') or '?':<28} {what}")
        return 0
    print(
        f"{len(offers)} host(s) with at least {spec.ttl_min} min of Vast lifetime; "
        f"{len(blocked)} machine(s) blocked"
    )
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

    def provider_args(s):
        s.add_argument(
            "--provider",
            choices=tuple(providers.KINDS),
            default="vast",
            help="where boxes come from: vast (default), runpod, or slurm (a cluster over SSH)",
        )
        s.add_argument(
            "--runpod-cloud",
            choices=("secure", "community"),
            help="restrict RunPod to one tier (default: consider both)",
        )
        g = s.add_argument_group("Slurm (with --provider slurm)")
        g.add_argument(
            "--login",
            metavar="HOST",
            help="the cluster's login host, as `ssh HOST` reaches it with no password prompt",
        )
        g.add_argument("--partition")
        g.add_argument("--slurm-account", metavar="ACCOUNT", help="sbatch --account")
        g.add_argument("--qos")
        g.add_argument("--constraint", help="sbatch --constraint, e.g. a GPU model's feature")
        g.add_argument(
            "--sbatch-arg",
            action="append",
            default=[],
            metavar="OPTION",
            help="one more sbatch option, e.g. --sbatch-arg=--gres=gpu:h100:1 (repeatable)",
        )

    def spec_args(s):
        s.add_argument(
            "--gpu",
            action="append",
            help='GPU model, e.g. "RTX 5090" (Vast: required; Slurm: checked by the gate); '
            "repeat to accept any of several models",
        )
        s.add_argument("--num-gpus", type=number(int, 1), default=1)
        s.add_argument(
            "--max-dph-per-gpu",
            type=number(float, 0, exclusive=True),
            help="price cap, $/GPU/hr (Vast: required)",
        )
        s.add_argument("--capability", help="required CUDA compute capability, e.g. 12.0")
        s.add_argument("--min-vram-mib", type=number(int, 0), default=0)
        s.add_argument(
            "--min-cuda",
            type=number(float, 0),
            default=13.0,
            help="lowest CUDA version the host's driver must support (default 13.0)",
        )
        s.add_argument("--disk-gb", type=number(int, 1), default=40)
        s.add_argument(
            "--min-cpus", type=number(int, 1), help="CPUs for the box (default: max(16, 8 per GPU))"
        )
        s.add_argument(
            "--min-ram-gb", type=number(int, 1), help="GB of RAM for the box (default: 32 per GPU)"
        )
        s.add_argument("--must-reach", action="append", default=[], metavar="URL")
        s.add_argument(
            "--interruptible",
            action="store_true",
            help="bid for interruptible boxes: cheaper, but stopped when outbid "
            "(run `reap --destroy` on a schedule)",
        )
        s.add_argument(
            "--bid-markup",
            type=number(float, 1),
            default=1.25,
            help="bid this times the offer's minimum bid (default 1.25)",
        )
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
    provider_args(s)
    s.add_argument("--image", help="the box's image (Vast: required)")
    s.add_argument("--namespace", required=True, help="who owns the box, e.g. quack-ci")
    s.add_argument("--name", required=True, help="this run, e.g. the CI run id")
    s.add_argument(
        "--count", type=number(int, 1), default=1, help="boxes to rent: all of them, or none"
    )
    s.add_argument(
        "--spare",
        type=number(int, 0),
        help=f"boxes raced beyond --count (default {DEFAULTS.spare}; on Slurm 0)",
    )
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
        "--ttl-min",
        type=number(int, 1),
        default=90,
        help="cleanup lease in minutes (default 90); on Vast, also the minimum offer lifetime",
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

    def box_arg(s, all_too=False):
        s.add_argument("--box", type=number(int, 0), help="which box of the run (0, 1, ...)")
        if all_too:
            s.add_argument("--all", action="store_true", help="every box of the run")

    def reconnect_arg(s):
        s.add_argument(
            "--reconnect-min",
            type=number(float, 0),
            default=10,
            help="how long to keep trying a lost connection before giving up (the job goes on)",
        )

    s = sub.add_parser("run", help="start a script on the box, detached, and follow it")
    state_args(s)
    box_arg(s, all_too=True)
    s.add_argument("--script", required=True)
    s.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME[=VALUE]",
        help="a variable for the job; NAME alone takes the value from this environment",
    )
    s.add_argument("--env-file", action="append", default=[], metavar="FILE")
    s.add_argument(
        "--timeout-min",
        type=number(float, 0),
        default=60,
        help="the box kills the job after this (0: never); exit 124",
    )
    s.add_argument("--detach", action="store_true", help="print the job id and return at once")
    s.add_argument(
        "--destroy-after",
        action="store_true",
        help="destroy the box when the script ends, whatever its exit code",
    )
    reconnect_arg(s)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("logs", help="print a job's output (default: the newest job)")
    state_args(s)
    box_arg(s)
    s.add_argument("job", nargs="?")
    s.add_argument("-f", "--follow", action="store_true", help="until the job ends; its exit code")
    reconnect_arg(s)
    s.set_defaults(fn=cmd_logs)

    s = sub.add_parser("wait", help="wait for a job to end; its exit code")
    state_args(s)
    box_arg(s)
    s.add_argument("job", nargs="?")
    reconnect_arg(s)
    s.set_defaults(fn=cmd_wait)

    s = sub.add_parser("jobs", help="the jobs on a box and how they are")
    state_args(s)
    box_arg(s)
    s.set_defaults(fn=cmd_jobs)

    s = sub.add_parser("kill", help="stop a job (default: the newest)")
    state_args(s)
    box_arg(s)
    s.add_argument("job", nargs="?")
    s.set_defaults(fn=cmd_kill)

    s = sub.add_parser("ssh", help="a shell on the box, or `-- CMD...` run there")
    state_args(s)
    box_arg(s)
    s.add_argument("command", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_ssh)

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
    provider_args(s)
    s.add_argument(
        "--run-id",
        help="the run to tear down (printed by `up`); without it, the directory's current run. "
        "For a run this directory has no state of, its boxes are found by their tag, from the "
        "provider the options name",
    )
    s.set_defaults(fn=cmd_down)

    s = sub.add_parser("reap")
    account_arg(s)
    provider_args(s)
    s.add_argument("--namespace", required=True, help="whose boxes: those tagged for it")
    s.add_argument("--destroy", action="store_true", help="destroy; without it, only report")
    s.set_defaults(fn=cmd_reap)

    s = sub.add_parser("search")
    spec_args(s)
    provider_args(s)
    s.add_argument(
        "--ttl-min", type=number(int, 1), help="Vast only: minimum offer lifetime (default 90 min)"
    )
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
    except InfraError as exc:
        log(f"::error::{exc}")
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
