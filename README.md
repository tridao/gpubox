# gpubox

Rent a GPU box from [Vast.ai](https://vast.ai), [RunPod](https://runpod.io) or
the VM clouds behind [Shadeform](https://shadeform.ai), or take one from a
Slurm cluster. gpubox checks that it works and manages its teardown.
For CI jobs and dev boxes.

Marketplace hosts are unreliable: some never finish pulling the image, some
come up with a GPU that is already busy. gpubox rents two at once, keeps the
first that passes a health check, and destroys the other. Run `gpubox down`
when done and schedule `gpubox reap --namespace NS --destroy` as a fallback.
Some providers enforce expiry, but others rely on the reaper, as do rentals
whose own lease timer never starts or is stopped by preemption.

The controller runs on Linux or macOS with Python 3.10+ and OpenSSH (`ssh`
and `ssh-keygen`). No third-party Python dependencies.
[DESIGN.md](https://github.com/tridao/gpubox/blob/main/DESIGN.md) explains how
it works, what it assumes, and what was measured.

## Install

```bash
pip install gpubox
export VASTAI_API_KEY=...
```

For a one-off CLI invocation, use `uvx gpubox --help`. In CI, pin a version
(`pip install gpubox==0.1.0`): the job that runs gpubox holds your provider
key. To install unreleased changes, pin a Git commit instead:
`pip install 'git+https://github.com/tridao/gpubox@<commit>'`.

Keys are read from the environment (`VASTAI_API_KEY`, `RUNPOD_API_KEY`,
`SHADEFORM_API_KEY`), else from lines `NAME=value` in
`~/.config/gpu-cloud-query/keys.env`.

## A dev box

```bash
gpubox up --gpu "RTX 4090" --max-dph-per-gpu 0.60 --image my/image:tag \
    --namespace dev --name mybox --ttl-min 480 --ssh-config
ssh mybox                      # also scp, rsync, editors
gpubox ssh -- nvidia-smi       # the same with the run's key, no ~/.ssh/config needed
gpubox renew --ttl-min 480     # the lease now ends 8 hours from now
gpubox down
```

`--ssh-config` adds a `Host mybox` entry to `~/.ssh/gpubox_config` and one
`Include` line at the top of `~/.ssh/config`. `down` removes the entry.

On Vast, `up --ttl-min 480` only tries offers that advertise at least eight
hours of remaining lifetime. Pass the same `--ttl-min` to `gpubox search` to
preview those offers without renting. `renew` moves gpubox's cleanup timer,
not Vast's contract end time.

`--gpu` can be repeated to accept any of several models (the cheapest offer
goes first). `--min-cpus` and `--min-ram-gb` are for the whole box; by
default it gets max(16, 8 per GPU) CPUs and 32 GB of RAM per GPU.
`--min-cuda` (default 13.0) is the lowest CUDA version the host's driver must
support: a CUDA 13.x image runs on any 13.x driver but not on a 12.x one.
A box whose driver turns out too old for the image's PyTorch fails without
the machine being blamed, and offers with drivers as old are skipped.

A private image: `--registry-user USER` with the token in
`GPUBOX_REGISTRY_TOKEN` (the registry comes from the image name). Vast and
the host's docker daemon see the token, so make it a read-only pull token.
Private pull reports omit the provider's raw status message and error body.

`--interruptible` rents interruptible boxes, bidding `--bid-markup` (1.25)
times each offer's minimum bid; the price cap applies to the bid with the
disk. An outbid box is stopped: its job dies, and the box keeps billing its
disk, with its lease timer stopped too, until something destroys it. So run
`gpubox reap --namespace NS --destroy` on a schedule, and expect `wait` to
report a lost job (75). Two launches in different state directories can still
bid on one machine at the same moment: the account's boxes are looked at
once, when a launch starts. Machines
the account already has a box on are skipped (our own bids would outbid each
other), and a box that stops or never starts is not held against its machine.

A large image needs a longer `--startup-timeout` than the default 300 s,
which counts from the rental and so includes the image pull.

## Jobs

```bash
gpubox run --script train.sh --env WANDB_API_KEY --timeout-min 600
```

`run` starts the script on the box in a session of its own and streams its
output. The job does not depend on the connection: a dropped connection is
picked up where it was (for up to `--reconnect-min`, 10), and Ctrl-C stops
the streaming, not the job. `--detach` prints the job id and returns at once.
Then:

```bash
gpubox logs -f [JOB]           # its output, until it ends; its exit code
gpubox wait [JOB]              # just the exit code
gpubox jobs                    # every job on the box: exit code, running or lost
gpubox kill [JOB]
```

(`JOB` defaults to the newest.) The box itself ends a job at `--timeout-min`
(default 60; 0 for never), with exit 124. `--destroy-after` destroys the box
when the script ends; a job can also run `gpubox-destroy-self` on the box
whenever it is done.

The job sees the image's environment (Docker `ENV`, saved when the box
starts; so do `ssh` sessions), then `--env` variables, then `GPUBOX_RUN_ID`,
`GPUBOX_JOB`, `GPUBOX_BOX_INDEX`, `GPUBOX_BOX_COUNT` and `GPUBOX_BOX_NAME`.
gpubox replaces the image's `ENTRYPOINT` to start its own sshd and timer: a
job that needs the entrypoint's setup runs it itself
(`exec /entrypoint.sh python -m my.job`).

**Secrets.** `--env NAME` (no `=`) takes the value from gpubox's own
environment, so it is on no command line: `op run -- gpubox run ...`, a CI
secret, Doppler and the like all work the same way. `--env-file FILE` reads
`NAME=VALUE` lines. Values go over SSH into a file only root can read, which
the job reads and deletes as it starts; they are never in Vast's instance
settings. The host's operator has root on the machine, so send only
credentials you can revoke: scoped, short-lived where possible.

## Several boxes

```bash
gpubox up --count 4 --gpu "RTX 5090" --max-dph-per-gpu 0.80 --image my/image:tag \
    --namespace eval --name shard --ttl-min 240 --ssh-config
gpubox status                  # shard-0 ... shard-3: addresses, lease
ssh shard-2
gpubox run --box 2 --script job.sh
gpubox run --all --script shard.sh   # on every box: $GPUBOX_BOX_INDEX says which
gpubox down                    # all of them
```

All or nothing: `up` races `count + --spare` hosts and returns 0 only with
`--count` healthy boxes; otherwise it destroys the ones it had. A run of one
box is called `NAME`; a run of several, `NAME-0`, `NAME-1`, ...

## RunPod

```bash
gpubox search --provider runpod --gpu "RTX 4090" --max-dph-per-gpu 0.80
gpubox up --provider runpod --gpu "RTX 4090" --max-dph-per-gpu 0.80 --image my/image:tag \
    --namespace dev --name mybox --ttl-min 240 --ssh-config
```

The same as on Vast, with RunPod's pods: the race, the gate, jobs, `ssh`,
`renew` and the lease all work as they do there. RunPod chooses the machine:
an offer is a GPU type in a data center that has it in stock, at RunPod's
list price, and `--gpu` matches its names ("RTX 4090" matches "NVIDIA GeForce
RTX 4090"; without `--gpu`, any model under the price cap). A pod deletes
itself when its lease ends, with the key RunPod gives each pod.
`--runpod-cloud community` rents community hosts instead of RunPod's own
data centers.

As everywhere, `--min-cpus` and `--min-ram-gb` default to max(16, 8 per GPU)
CPUs and 32 GB per GPU: pass smaller ones to admit RunPod's smaller pods.

Not yet on RunPod: `--interruptible`, `--registry-user` (private images) and
`--exclude-country`; and machines are not recorded (RunPod's API does not
name them), so the blocklist does not apply. An image that cannot be pulled
is not recognised as such: each pod is given up on after `--stall-timeout`.

## Shadeform (VM clouds)

```bash
gpubox search --provider shadeform --gpu A6000 --max-dph-per-gpu 0.80
gpubox up --provider shadeform --gpu A6000 --max-dph-per-gpu 0.80 --image my/image:tag \
    --namespace dev --name mybox --ttl-min 120 --overall-timeout 1500
gpubox ssh                     # a shell in the box's container
```

Shadeform rents VMs on Hyperstack, Massed Compute, Lambda, Crusoe, Nebius
and others with one key. A box there is your image run as a container on
the VM: gpubox gives the VM a startup script that adds the run's key and
starts the image, and reaches the box with SSH to the VM and `docker exec`,
so the image needs no sshd (but, as everywhere, bash and python3). An offer
is an instance type in a region, booted with Shadeform's own OS image with
the newest CUDA, which must be at least `--min-cuda` (13.0 by default: pass
`--min-cuda 12.8` for an image built for CUDA 12). VMs take 5 to 8 minutes
to boot; the start-up and stall limits count from when Shadeform expects
the boot to end, but `--overall-timeout` should allow for it.

The lease is Shadeform's own: the VM is created with an auto-delete time,
which `renew` moves. A VM cannot delete itself, so `run --destroy-after` is
refused. Not yet: `--ssh-config`, `--registry-user`, `--interruptible`,
`--exclude-country`, `--account-id`.

## Hyperstack

`--provider hyperstack` (`HYPERSTACK_API_KEY`) rents Hyperstack's VMs
directly, as boxes like Shadeform's: the image as a container on the VM,
reached with SSH and `docker exec`, booted with Hyperstack's image "with
Docker" of the newest CUDA in the region. Hyperstack keeps no expiry and a
VM cannot delete itself, so `up` warns: only `gpubox down`, or a scheduled
`gpubox reap --provider hyperstack --namespace NS --destroy`, ends a VM left
behind. Hyperstack needs a key pair at the create: if the environment has
none, the run's public key is imported once as `gpubox`.

## Massed Compute

`--provider massed` (`MASSED_COMPUTE_API_KEY`) rents Massed Compute's VMs
directly, through its MCP server, as boxes like Shadeform's, on Massed's
"Ubuntu Server 22.04 w/ drivers" (NVIDIA drivers and docker). Its 1-GPU
products have 6 CPUs and 32 GB: pass `--min-cpus 6`. Massed cannot rename an
instance, so the lease set by `up --ttl-min` cannot be moved (`renew` is
refused), and as on Hyperstack only `down` or a scheduled `reap` ends a VM
left behind.

## Lyceum

`--provider lyceum` (`LYCEUM_API_KEY`) rents Lyceum's VMs, as boxes like
Shadeform's. Lyceum runs nothing of ours at boot, so the first probe that
gets in sends the box's setup and starts it. Its region and disk (500 GB)
are its own choice; as on Hyperstack only `down` or a scheduled `reap` ends
a VM left behind.

## A Slurm cluster

```bash
gpubox up --provider slurm --login mycluster --num-gpus 1 --gpu H100 \
    --namespace dev --name mybox --ttl-min 240
gpubox ssh                     # a shell in the allocation
gpubox run --script job.sh     # detached; follows its output
gpubox down
```

A box is one batch job holding an allocation (`--gres=gpu:N`, and
`--min-cpus`, `--min-ram-gb` if given; otherwise the cluster's defaults).
gpubox runs Slurm's commands on the login host over your own SSH
configuration, so `ssh mycluster squeue` must work without a password
prompt (keys, or a ControlMaster opened beforehand), and the login shell
must be POSIX; Slurm must be 20.11 or later (`srun --overlap`). Everything
runs in the nodes' own environment: `--image` is not used.

`--partition`, `--slurm-account`, `--qos`, `--constraint` and
`--sbatch-arg OPTION` (repeatable) go to sbatch. A request the cluster can
never satisfy (no such partition, too many GPUs, a time limit above the
partition's) is a configuration error at once. Time in the queue is nobody's
fault: `--startup-timeout` and `--stall-timeout` count from when the job
starts, and only `--overall-timeout` bounds the wait. `--spare` is 0 by
default here; a node that fails the gate is excluded from the jobs submitted
after it.

The lease is kept twice: by the job's own script, which ends the job when the
lease ends, and by Slurm's time limit, set two minutes past it. `renew` moves
both; a cluster that does not let users extend a time limit refuses it, and
the box then ends at its old one. Commands reach a box with
`srun --jobid=ID --overlap`, so they get its GPUs (`CUDA_VISIBLE_DEVICES`),
CPUs and environment; jobs started by `run` are started by the batch script,
since Slurm ends whatever a step started when the step ends. The gate looks
only at the allocation's GPUs. Not available: `--ssh-config`, `search`, and
the machine blocklist and reputation store, which are about Vast's machines.

## A CI job

```yaml
- id: up
  run: >
    gpubox up --gpu "RTX 5090" --num-gpus 2 --max-dph-per-gpu 1.25
    --image my/ci-image:tag --namespace my-ci
    --name "${{ github.run_id }}-${{ github.run_attempt }}" --ttl-min 60
- run: gpubox run --script remote.sh --env "SHA=${{ github.sha }}" --timeout-min 40
- if: always() && steps.up.outputs.run_id != ''
  run: gpubox down --run-id "${{ steps.up.outputs.run_id }}"
```

and, from a schedule every half hour:

```bash
gpubox reap --namespace my-ci --destroy
```

`up` writes the run id as a step output before it rents anything. Passing it
to `down` lets the teardown work even from a runner that has lost the state
directory. Always run `down` after `up`, whatever `up` returned.

## From Python

```python
import gpubox

spec = gpubox.Spec(image="ghcr.io/org/img:1", gpus=("RTX 5090", "RTX 4090"),
                   max_dph_per_gpu=1.0, min_cpus=12, min_ram_gb=64, ttl_min=240)
with gpubox.up(spec, namespace="eval", name="shard", count=4) as run:
    jobs = run.start(open("shard.sh").read(), env={"TOKEN": token}, timeout_min=180)
    outcomes = gpubox.wait_all(jobs)   # per job: .code, or .error
# the boxes are destroyed when the block ends, however it ends (run.keep() to keep them)
```

If `up` fails or is interrupted after it began the run, it tears the run down
before raising (and says how to retry if that fails too). A `Run` is a
handle, not a lock: once the run is torn down or replaced in its state
directory, whatever would touch its boxes raises `StateError`.

`gpubox.attach(state_dir)` picks up a run left in a state directory (also by
the command line); `run.boxes[i]` has `ssh_argv`, `sh`, `start`, `job`,
`jobs`; a `Job` has `wait`, `logs` and `kill`. `gpubox.down(run_id=...)` and
`gpubox.reap(namespace)` are the teardowns. Failures are exceptions:
`ConfigError` and `StateError` for our own mistakes, `InfraError` (`NoBoxes`,
`TeardownIncomplete`, `BoxLost`, `JobLost`) for the provider's or a box's.
`gpubox.set_log(fn)` sends what it says somewhere other than stdout, for the
whole process. The command line is a thin layer over this (`gpubox/api.py`).
Elsewhere than Vast, pass `provider="runpod"`, or `provider="slurm",
options={"login": "mycluster", ...}`, to `up` (and to `down(run_id=...)`,
`reap` and `search`); a run remembers where its boxes came from.

## Commands

`gpubox.receipt(state_dir, run_id)` reads the durable rental attempts, keyed by
tag. Each has an offer, machine, host, quoted `dph`, `instance_id` (when known)
and `outcome`: `unknown`, `refused`, `rented` or `gone`. Use a persistent state
directory per run. An unanswered create, including an HTTP 429 on Vast, stays
unknown until its box is identified; `down` with that state directory reports
incomplete even if the listing is empty. Retry `down` later. A missing receipt
or teardown by tag alone is not proof that no rental was made. A receipt is a
record of quoted compute cost, not an invoice or account-wide spending cap.

| | |
|---|---|
| `up` | rent boxes (`--count`, default 1) and wait until they pass the health gate; `--provider runpod`, `shadeform` or `slurm` for those |
| `status` | show the run's boxes: names, addresses, lease |
| `run` | start a local script on a box, detached, and stream its output |
| `logs` | a job's output (`-f`: until it ends) |
| `wait` | wait for a job to end |
| `jobs` | the jobs on a box |
| `kill` | stop a job |
| `ssh` | a shell on a box, or `-- CMD` run there |
| `renew` | move the end of the boxes' lease |
| `down` | destroy everything the run rented |
| `reap` | destroy boxes of a namespace whose lease has ended (without `--destroy`: report) |
| `search` | show the offers `up` would try; rents nothing |

Exit codes: 0 ok; 2 configuration (bad arguments, no API key, an image that
cannot be pulled or does not start our sshd); 75 infrastructure (no healthy
box, box lost mid-run, teardown not confirmed); 124 script timeout; 130
interrupted. `run`, `wait` and `logs -f` otherwise return the job's own exit
code.

State lives in `--state-dir` (default `.gpubox`): `run.json`, an SSH key per
run, and `observations.jsonl`.

## How it works

- **Race.** `up` rents `--count + --spare` hosts (2 by default), cheapest
  first, one machine per host. A box that fails is destroyed and replaced, up
  to `--max-attempts` rentals (default `2 * count + 4`).
- **Health gate.** GPU model, count and VRAM; required URLs reachable
  (`--must-reach`); the GPU idle before we touch it. Where the image has a
  CUDA build of PyTorch, also: CUDA works, a matmul on every GPU matches the
  CPU, and most of the VRAM can be allocated. It needs `python3` in the image.
- **Your own checks.** `--gate-script FILE` runs on each box after the gate
  passes (by its `#!` line, else bash; at most `--gate-timeout` seconds).
  Exit 0 admits the box; anything else fails it and another is tried. To
  have the machine blocklisted, print a last line
  `{"ok": false, "reason": "no_network", "detail": "..."}` with one of the
  gate's reason codes (`policy.REASONS`).
- **Tag.** Every box is labelled `<namespace>.<name>.<random>.a<n>.<expiry>`
  when it is rented. Anything that knows the run id can find the run's boxes
  and destroy them.
- **Lease.** The lease ends `--ttl-min` after `up` started. A timer on the
  box or the provider enforces it where supported. A scheduled `reap`
  destroys expired boxes when there is no timer or it could not run.
  `renew` moves the end on providers that support it.
- **Blocklist.** A machine that failed in a launch that succeeded is skipped for 2 hours to 14 days, depending on why
  (`observations.jsonl`; cache it between CI runs). A launch that failed
  blames nobody.
- **Defaults from measurement.** Hosts in mainland China are skipped
  (`--exclude-country`, default `CN`; `none` to skip none). A box whose
  loading status has not changed for `--stall-timeout` (300 s) is replaced.

## What it promises, and what it does not

- `up` exits 0 only with `--count` boxes that passed the gate, were still
  running when looked up afterwards, and are recorded in `run.json`. Before that, a failure
  or Ctrl-C rolls back what was rented. The rollback gets about ten seconds
  and is an attempt; `down` finishes it.
- At most `--count + --spare` boxes are billable at once, as far as gpubox
  can know. A rental request that got no answer keeps its slot.
- No offer above the price cap is rented, whatever the search returned.
- `down` exits 0 only when every box it knows of is confirmed gone by id and
  two listings in a row show nothing with the run's tag. Otherwise it exits
  75 and says what is left and what to do.
- Not promised: a maximum lifetime. With no working lease timer or
  provider-enforced expiry, a box that `down` missed can keep billing until
  the reaper removes it. Nothing here finds a box that was renamed, or that
  the provider does not list. There is no spending cap across runs.

`--account-id` (or `GPUBOX_ACCOUNT_ID`) makes every command refuse an API key
that belongs to another account.

## Sharing what machines did

Optional. With `GPUBOX_STORE_URL` and `GPUBOX_STORE_TOKEN` set, observations
of machines are shared through a small Cloudflare Worker (`store/`), so that
a machine that failed for one client is skipped by the others. It is
advisory: it costs a command a few seconds at most and never fails it.

## Development checks

```bash
python3 -m pip install pytest pre-commit
pre-commit install         # lint and format staged Python files with Ruff
pre-commit run --all-files
(cd store && npm ci && npm test)  # Node 24 LTS; also enables the Python/Worker integration test
python3 -m pytest          # fake provider and fake SSH; rents nothing
python3 tests/mutations.py # breaks each behaviour in turn; a test named for it must fail
```

On PRs and pushes to `main`, GitHub Actions runs the hooks, tests Python 3.10
and current Python 3, and checks the metadata and installation of both the
wheel and the source distribution.
The current-Python job also tests the Worker and Python client against local D1.
All jobs use GitHub-hosted Ubuntu runners; no GPUs or cloud credentials are
needed. Mutation tests run weekly or on demand (Actions → Mutation tests).
