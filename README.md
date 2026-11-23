# gpubox

Rent a GPU box from [Vast.ai](https://vast.ai), check that it works, and make
sure it is destroyed afterwards. For CI jobs and dev boxes.

Marketplace hosts are unreliable: some never finish pulling the image, some
come up with a GPU that is already busy. gpubox rents two at once, keeps the
first that passes a health check, and destroys the other. Every box carries
a lease, and destroys itself when the lease ends even if nothing else is
left to do it.

Standard library only, Python 3.10+. `DESIGN.md` explains how it works, what
it assumes, and what was measured.

## Install

```bash
pip install git+https://github.com/tridao/gpubox@<commit>
export VASTAI_API_KEY=...
```

In CI, pin a commit: the job that runs gpubox holds your Vast key.

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

## Commands

| | |
|---|---|
| `up` | rent boxes (`--count`, default 1) and wait until they pass the health gate |
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
- **Lease.** `--ttl-min` after `up` started, a timer inside the box destroys
  it. `reap` destroys boxes whose lease has ended, in case the timer could
  not. `renew` moves the end.
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
- Not promised: a maximum lifetime. A box that `down` missed runs until its
  lease ends. A box stuck pulling its image has no timer yet and waits for
  the reaper. Nothing here finds a box that was renamed, or that the
  provider does not list. There is no spending cap across runs.

`--account-id` (or `GPUBOX_ACCOUNT_ID`) makes every command refuse an API key
that belongs to another account.

## Sharing what machines did

Optional. With `GPUBOX_STORE_URL` and `GPUBOX_STORE_TOKEN` set, observations
of machines are shared through a small Cloudflare Worker (`store/`), so that
a machine that failed for one client is skipped by the others. It is
advisory: it costs a command a few seconds at most and never fails it.

## Tests

```bash
python3 -m pytest          # fake provider and fake SSH; rents nothing
python3 tests/mutations.py # breaks each behaviour in turn; a test named for it must fail
cd store && npm ci && npm test
```
