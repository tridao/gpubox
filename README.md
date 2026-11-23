# gpubox

Rent a verified GPU box from Vast.ai, and make sure it is destroyed afterwards.

One box for a CI job. Standard library only (Python 3.10+). `DESIGN.md` is the
design (revision 4) with the rules the code and tests are numbered by; the
earlier revision and its review history are in QuACK's `AI/gpubox_design.md`.

```bash
export VASTAI_API_KEY=...          # the key of an account used for nothing else
export GPUBOX_ACCOUNT_ID=...       # that account's id: a key of another account is refused
gpubox up --gpu "RTX 5090" --num-gpus 2 --max-dph-per-gpu 1.25 \
    --image tridao/quack-kernels:cu13.2-DATE --namespace quack-ci --name "$RUN-$ATTEMPT"
gpubox run --script remote.sh --env SHA=$SHA --timeout-min 40
gpubox down --run-id "$RUN_ID" --abandon-unseen   # always; exit 75: something is unresolved
gpubox reap --exclusive --max-age-min 90            # report
gpubox reap --exclusive --max-age-min 90 --destroy  # every 30 minutes, from a schedule
```

`up` prints the run id and writes it to the step outputs (`run_id`) before it
rents anything, and refuses to start if it cannot write it. State lives in `--state-dir` (default `.gpubox`): `run.json`,
a snapshot of what the run asked for and which instance ids it knows; a
throwaway SSH key per run; and `observations.jsonl` (which machines worked).

## How a box is ours

- Every create request carries a tag as the instance's label:
  `<namespace>.<name>.<12 random hex>.a<attempt>.<expiry>`. Any process that
  knows the run id can find the run's boxes in the provider's listing and
  destroy them, with no state at all. That is what cleans up after a CI
  runner whose disk is gone.
- Every instance id the run learns is written to `run.json` before the box is
  used or deleted. With the state directory, `down` therefore also destroys a
  box that changed its own label.
- A box counts as gone only when a lookup of its id says it does not exist. A
  listing, an accepted delete or an error never confirms that.

## What it promises, and what it does not

- `up` exits 0 only with a box that passed the health gate, is still running,
  and is recorded in `run.json`. Before that point a failure or a signal
  triggers a rollback of everything the launch rented; after it the box is
  kept, even if `up` is then interrupted. The rollback has ten seconds and is
  an attempt, not a promise. Always run `down` after `up`, whatever it
  returned.
- A healthy box is kept even if another box's delete is not confirmed yet.
  `up` warns; `down` is where it becomes an error.
- At most `1 + --spare` attempts hold a slot at once. A failed box keeps its
  slot until its delete is confirmed. A create that got no answer keeps its
  slot for the rest of the launch, because its box may exist unseen.
- `down` exits 0 only if every id in the snapshot is confirmed gone, a
  complete listing taken after the last delete shows no box with the run's
  tag, and no create is left without an answer. With `--abandon-unseen` it
  waits until such a create is 60 s old, and if no box with its tag is
  visible then, leaves it to the box's timer and the reaper. That is not
  proof that nothing was rented.
- `down --run-id ID` for a run the state directory has no snapshot of works
  by tag alone, and cannot find a box that changed its label.
- Every box carries a timer that destroys it once the expiry in its tag has
  passed (`--ttl-min` after `up` started). It is the first thing the
  container starts, so it does not cover a box whose container never starts.
- `reap --exclusive` destroys every instance of the account that has no valid
  tag (seen so in two listings 30 s apart), whose lease ended, or that is
  older than `--max-age-min`. It refuses to run unless the key belongs to
  `--account-id`. Without `--destroy` it only reports.
  `reap --namespace NS` is for a shared account: only expired boxes tagged
  for that namespace.
- Not promised: a maximum lifetime. A box that no listing shows cannot be
  found by anything here, and the reaper only helps while its schedule runs
  (have the workflow ping a dead-man check after each pass). Not promised
  either: a cap across separate runs.

## Not yet verified against the provider

The fake provider in the tests follows Vast's documentation and what revision
3 measured. Three things in this revision have not been run against the real
API, and `up` fails before renting anything if the first two are wrong:

- the listing check that `total_instances` equals the rows read, and that
  every row has a `label` field;
- `account()`, which reads the `id` of `GET /api/v0/users/current/`. Every
  `up` and `down` calls it now, so the key needs permission to read its own
  account;
- the new timer in the bootstrap command (tested here by running its loop in
  bash with a stand-in for the delete).

`DESIGN.md`, section 14, lists what else must be checked on the subaccount
before `reap --exclusive --destroy` is switched on.

## Machines that failed

A machine is recorded as failed only if another candidate became healthy in
the same launch. `observations.jsonl` blocks it for 2 hours (GPU busy), a day
(unreachable, no network, died) or 14 days (wrong hardware, CUDA or runtime
errors); a later healthy run clears the first two. Cache the file between CI
runs.

The shared reputation store (`store/`, `src/gpubox/store.py`) is parked: its
code and tests are kept, and no command talks to it.

Hosts in mainland China are skipped by default (`--exclude-country`, default
`CN`; `none` to skip none): in `experiments/pull_time` they took a median of
250 s to load the CI image against 65 s elsewhere, and 2 of 6 never did.

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
python3 tests/mutations.py # breaks each rule in turn; a test named for it must fail
```
