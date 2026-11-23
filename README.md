# gpubox

Rent a verified GPU box from Vast.ai, and make sure it is destroyed afterwards.

For a CI job or a dev box. Standard library only (Python 3.10+). `DESIGN.md`
is the design, including the assumptions it rests on.

```bash
export VASTAI_API_KEY=...
export GPUBOX_ACCOUNT_ID=...       # optional: refuse a key of any other account
```

CI:

```bash
gpubox up --gpu "RTX 5090" --num-gpus 2 --max-dph-per-gpu 1.25 --ttl-min 60 \
    --image tridao/quack-kernels:cu13.2-DATE --namespace quack-ci --name "$RUN-$ATTEMPT"
gpubox run --script remote.sh --env SHA=$SHA --timeout-min 40
gpubox down                        # always; exit 75: something is left, and it says what
gpubox reap --namespace quack-ci --destroy    # from a schedule, every 30 minutes
```

In a GitHub workflow, give the teardown the run id as well, so that it works
even if the state directory is lost. `up` writes it as a step output, not as
an environment variable:

```yaml
- id: up
  run: gpubox up ...
- run: gpubox run --script remote.sh
- if: always() && steps.up.outputs.run_id != ''
  run: gpubox down --run-id "${{ steps.up.outputs.run_id }}"
```

A dev box:

```bash
gpubox up --gpu "RTX 4090" --max-dph-per-gpu 0.60 --ttl-min 480 --startup-timeout 600 \
    --image tridao/quack-kernels-dev:DATE --namespace dev --name mybox --ssh-config
ssh mybox                          # also scp, rsync, editors
gpubox renew --ttl-min 480         # the lease now ends 8 hours from now
gpubox down
```

`up` prints the run id and writes it to the step outputs (`run_id`) before it
rents anything, and refuses to start if it cannot. State lives in
`--state-dir` (default `.gpubox`): `run.json`, a throwaway SSH key per run,
and `observations.jsonl` (which machines worked).

## How it works

- `up` races `1 + --spare` hosts (2 by default), cheapest first, one machine
  per host. The first to boot, answer SSH and pass the health gate wins; the
  others are destroyed. A box that fails is replaced, up to `--max-attempts`.
- Every box carries a tag as its label,
  `<namespace>.<name>.<random>.a<attempt>.<expiry>`. Anything that knows the
  run id can find the run's boxes and destroy them. `down --run-id` needs
  nothing else, which is what cleans up after a CI runner whose disk is gone.
- Every box has a lease (`--ttl-min`). A timer inside the box destroys it
  when the lease ends; `reap` destroys boxes of a namespace whose lease has
  ended. `renew` moves the end of the lease.
- Hosts in mainland China are skipped (`--exclude-country`, default `CN`;
  `none` to skip none): they took a median of 250 s to load the CI image
  against 65 s elsewhere.
- A box still loading whose status has not changed for `--stall-timeout`
  (300 s) is replaced. That matters when `--startup-timeout` is longer, as a
  large image needs.
- If two hosts say the image cannot be pulled, `up` stops with exit 2.

## What it promises, and what it does not

- `up` exits 0 only with a box that passed the health gate, was still running
  when looked up afterwards, and is recorded in `run.json`. Before that point
  a failure or a signal rolls back what the launch rented; the rollback has
  ten seconds and is an attempt. Always run `down` after `up`.
- At most `1 + --spare` attempts hold a slot. A failed box keeps its slot
  until a lookup says it is gone; a create that got no answer keeps its slot
  for the rest of the launch.
- `down` exits 0 only when every id it knows is confirmed gone by lookup and
  two listings in a row show nothing with the run's tag. Otherwise it exits
  75 and prints what is left and what to do.
- The price cap is checked on each offer again before it is rented, whatever
  the search returned.
- Not promised: a maximum lifetime. A box that `down` missed runs until its
  lease ends. A box whose container never started has no timer and waits for
  the reaper. Nothing here finds a box that was renamed, or that no listing
  shows. There is no cap across separate runs.

## Not yet verified against the provider

`up`, `run`, `renew` and `down` have not been run end to end on a real box.
The fake provider in the tests follows Vast's documentation and what was
measured. `DESIGN.md`, section 9, lists what is still from documentation
only: the relabel call that `renew` uses, the in-box timer, and whether a
subaccount is separate from its parent.

## Machines that failed

A machine is recorded as failed only if another candidate became healthy in
the same launch. `observations.jsonl` blocks it for 2 hours (GPU busy), a day
(unreachable, no network, died, stalled) or 14 days (wrong hardware, CUDA or
runtime errors); a later healthy run clears the first two. Cache the file
between CI runs.

The shared reputation store (`store/`, `src/gpubox/store.py`) is parked: its
code and tests are kept, and no command talks to it.

## QuACK CI mapping

The flags that reproduce `tools/ci/vast/vast_ci.py` (checked against a pinned
recording of its requests in `tests/fixtures/reference_requests.json`; the
create payload differs in the label and the timer only):

```
--gpu "RTX 5090" --num-gpus 2 --max-dph-per-gpu 1.25 --capability 12.0
--min-vram-mib 30000 --min-cuda 13.2 --ttl-min 60
--must-reach https://github.com/ --must-reach https://pypi.org/simple/pip/
```

## Tests

```bash
python3 -m pytest          # fake provider and fake SSH; rents nothing
python3 tests/mutations.py # breaks each behaviour in turn; a test named for it must fail
```
