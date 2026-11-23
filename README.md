# gpubox

Rent a verified GPU box from Vast.ai, and make sure it is destroyed afterwards.

Phase 1: one box for a CI job. Standard library only (Python 3.10+). The
design and its review history are in QuACK's `AI/gpubox_design.md`.

```bash
export VASTAI_API_KEY=...
gpubox up --gpu "RTX 5090" --num-gpus 2 --max-dph-per-gpu 1.25 \
    --image tridao/quack-kernels:cu13.2-DATE --namespace quack-ci --name "$RUN-$ATTEMPT"
gpubox run --script remote.sh --env SHA=$SHA --timeout-min 40
gpubox down                      # always; exit 75 means something is unresolved: read the output
gpubox reap --namespace quack-ci --max-age-min 90            # report
gpubox reap --namespace quack-ci --max-age-min 90 --destroy  # only if every box in the namespace is disposable
```

State lives in `--state-dir` (default `.gpubox`): the journal of what was
rented, a throwaway SSH key per run, and `observations.jsonl` (which machines
worked).

## What it guarantees, and what it does not

- A box is used only after it passed the health gate, every other box whose
  id is known is confirmed gone, and its id is durably journalled as the
  winner. Until then any failure or signal rolls back everything the launch
  rented; a box whose delete could not be confirmed stays in the journal for
  `down`. The one thing that may stay open beside a winner is a create that
  never gave us an id (below).
- `up` can be interrupted after the winner is journalled and still exit
  non-zero. Always run `down` after `up`, whatever it returned.
- `down` destroys every instance id in the journal. A label is only used to
  find a box whose create request got no usable answer, and then only if the
  provider's listing shows exactly one instance with that label, on the
  expected machine, with our image, started no more than 30 s before the
  attempt. That is strong evidence, not proof: an instance someone made to
  match all of it would be destroyed.
- At most `1 + --spare` boxes exist at once. A failed box keeps its slot until
  the provider confirms it is gone; a create with an unclear outcome keeps its
  slot too.
- A create whose box never shows up cannot be proven to have rented nothing.
  `down` then keeps exiting 75 until you have looked at the provider and run
  `down --abandon-unseen`.
- Every create request carries a timer that destroys the box after
  `--ttl-min`. It is started by the container's own start-up, just before
  sshd, so it does not cover a box stuck pulling the image or in that
  start-up.
- `reap --destroy` deletes by label in a namespace you declare disposable.
  It does not know or check that this launcher rented those boxes.
- Not guaranteed: a cap across separate runs; cleanup of a box whose code
  relabelled it and killed its timer when the launcher's state is also lost.
  `reap` works by label and cannot find such a box either; only a provider
  account used for nothing else, swept as a whole, closes that.

## Sharing what machines did

Set `GPUBOX_STORE_URL` and `GPUBOX_STORE_TOKEN` and gpubox shares its
observations of machines (which worked, which failed and with what reason
code; never logs or run names) through the Worker in `store/`:

- before a launch it sends what it has not sent yet and reads everyone's
  observations of the last 30 days; machines that failed for others are
  skipped like machines that failed here;
- after a launch, and when a box dies mid-run, it sends what it learned.

Both steps together may take at most 5 seconds and never fail a command. If
the store cannot be reached, the last complete copy is used and unsent rows
wait in `outbox.jsonl`. Without the two variables nothing is contacted, and
`observations.jsonl` alone is used (cache it between CI runs in that case).

## QuACK CI mapping

The flags that reproduce `tools/ci/vast/vast_ci.py` (checked against a pinned
recording of its requests in `tests/fixtures/reference_requests.json`):

```
--gpu "RTX 5090" --num-gpus 2 --max-dph-per-gpu 1.25 --capability 12.0
--min-vram-mib 30000 --min-cuda 13.2 --ttl-min 60
--must-reach https://github.com/ --must-reach https://pypi.org/simple/pip/
```

The reaper needs `--legacy-pattern 'quack-ci-\d+-\d+'` for one release, to
sweep boxes labelled by the old launcher.

## Tests

```bash
python3 -m pytest          # fake provider and fake SSH; rents nothing
python3 tests/mutations.py # breaks each safety property in turn; a test named for it must fail
```
