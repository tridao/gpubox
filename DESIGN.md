# gpubox design, revision 4

Status: third draft (2026-11-23), implemented on the branch `rev4` and not
yet run against the real provider (section 14). The first two drafts and
the first implementation were each blocked in review; section 15 lists the
findings and what was done with each. The fixes for the third review have
not themselves been reviewed. Revision 3's text stays in QuACK's
`AI/gpubox_design.md`; this document replaces its sections 4, 5, 7 (phase 2),
8 and 11 and keeps the rest by reference.

Decided by Tri on 2026-11-23:

1. Automated boxes get their own Vast subaccount.
2. A box that passed the gate is kept even if a loser's delete is not yet
   confirmed.
3. The Cloudflare reputation store is parked. Its code stays in the
   repository, unwired.

The sections on behaviour state rules, each of which a test enforces. The
earlier drafts used pseudocode instead, and both reviews found places where
the pseudocode and the prose disagreed. The code is where that gets settled.

## 1. Why a revision

Revision 3 makes a local, fsynced event log the authority on what a run
owns, and treats the label on an instance as a hint. Much of the launcher
follows from that:

- replay of the log, with rules for torn lines, damage and rotation;
- `match_lost_box`, the evidence rules for taking a labelled instance as ours;
- signals turned into flags, a thread pool, and a commit point that holds
  only if four conditions are true at once;
- four separate teardown paths (loser cleanup, rollback, `down`, `reap`).

It still does not cover the ordinary CI failure: the runner disappears and
its disk with it. Then the reaper, which works from the label, cleans up.

Revision 4 turns the two around. The tag on the box, set by the create
request, says whose it is. Local state is a snapshot of what this run asked
for and which ids it knows: enough to tear down by id while the disk lives,
not needed to tear down by tag when it does not.

## 2. Facts

Four things cannot be designed away:

1. Hosts are unreliable. A rented box may never start, or start broken.
2. A create request is not idempotent and its outcome can be unknown.
   Neither Vast nor Runpod documents an idempotency key.
3. The controller can die at any instruction, and in CI its disk dies too.
4. A leaked box is silent. It bills until someone looks.

Measured on Vast (revision 3, section 2; RTX 5090, one image):

- 4 of 14 rentals failed the gate, all with the GPU busy before our first
  CUDA context. Two of those machines passed 25 and 90 minutes later.
- A created instance is in the listing, with its label, on the first listing
  call after the create returns (3 of 3). A destroyed one is gone from the
  listing and from the lookup by id within about 5 s (3 of 3). One listing
  call takes about 2 s.
- A container that exits is restarted and keeps billing. A box can destroy
  itself with the `CONTAINER_API_KEY` Vast injects (one trial). Code on a
  box can also relabel its own instance with that key.
- A missing image tag shows in `status_msg` within 37 s.
- Rolling back two boxes took 6 to 10 s; a cancelled GitHub step gets about
  10 s before it is killed.

Three successes are evidence for the ordinary path. They are not a bound on
how late a box can appear, on how long a delete can take, or on how stale a
listing can be, and nothing below treats them as one.

Read in Vast's documentation, not tested:

- The create request takes the label. A label may be 1,024 characters.
- The v1 listing returns 25 rows a page in ascending id order, each with
  `id`, `label`, `machine_id`, `start_date`, `actual_status`, `status_msg`,
  `image_uuid`, `public_ipaddr` and, for running instances, `ports`; and a
  `total_instances` count.
- Request rates are limited per endpoint and per key; the answer is 429
  without `Retry-After`.
- API keys can be scoped to endpoint categories, which does not separate one
  key's instances from another's. A subaccount is created with
  `parent_id: "me"` and gets its own key.
- Credit is prepaid, and instances are stopped when the balance reaches
  zero. Stopped is not destroyed: a stopped instance still bills for disk.

Not known. Section 14 says which of these block which feature:

- whether a subaccount's listing, deletes and balance are separate from its
  parent's;
- whether `start_date` is set while a box is pulling its image, and whether
  anything the renter or the box does can change it later;
- whether a label sent with create is on the instance in every listing that
  shows the instance.

## 3. Principles

1. **The tag says whose a box is.** It is set by the create request and
   holds the run, the attempt and when the box expires. A box found by its
   tag can be destroyed by a process that knows nothing else.
2. **An id we know is ours until a lookup of that id says it does not
   exist.** The listing discovers boxes. It never confirms that one is gone.
3. **Choosing what to destroy and destroying it are separate.** Each command
   has its own rule for choosing. One routine deletes a set of ids and
   confirms each.
4. **Several chances to destroy a box**: the controller, a timer inside the
   box, a scheduled reaper. They all need the provider to honour a delete,
   so they are separate chances, not separate guarantees.
5. **Local state helps and is not required.** With it, `down` also reaches
   boxes that lost their tag. Without it, the tag and the reaper remain.

## 4. The run and the tag

```
run id   <namespace>.<name>.<12 hex>          quack-ci.19283746-1.3f9c2a1b5d7e
tag      <run id>.a<attempt>.<expiry>         quack-ci.19283746-1.3f9c2a1b5d7e.a2.1795400000
```

- `namespace` (`[a-z0-9][a-z0-9-]*`) and `name` (`[A-Za-z0-9][A-Za-z0-9_-]*`)
  come from the caller; CI passes its run id and attempt as the name.
- The hex digits are random, minted by `up`, so that two runs do not share
  an id even when callers reuse a name (48 bits: a collision is not
  impossible, only not worth designing for). `up` prints the run id and
  writes it to the step outputs before it sends any create.
- `a<attempt>` numbers the create requests of the run. It ties a box that
  turns up in the listing to the request that made it.
- `expiry` is ten digits of Unix time: `up`'s start plus `--ttl-min`.

Parsing is exact: five parts, each matching in full. A label that does not
parse is not a tag. Revision 3's labels (four parts) and the old launcher's
(`quack-ci-...`) do not parse.

## 5. Local state

One snapshot file per state directory, `run.json`, replaced whole at each
change (write, fsync, rename, fsync of the directory):

```
account    the provider account the run rented on, when known
run        the run id
expiry
meta       image, gpu, num_gpus
attempts   n -> {offer, machine, host, price, sent, state, id}
ids        id -> live | gone
winner     {id, attempt, host, port} once published
closing    set by down
closed     set by down when everything below held
```

An attempt is `open` from just before its create request is sent, with the
time; then `refused`, or `bound` to an id. `open` covers both "no answer
yet" and "no usable answer": in either case a box may exist that we have no
id for. `ids` is the set of every id the run knows, whatever told us about
it: the create answer, a listing during the launch, a listing during
teardown.

There is no replay. The file is read whole or not at all. Rules:

- **S1.** An attempt is in the snapshot as `open` before its create request
  is sent. If that write fails, nothing is sent.
- **S2.** An id is in `ids` before the box is used and before any delete is
  sent for it. If that write fails, the id is still destroyed from memory,
  and the command cannot end clean.
- **S3.** An id becomes `gone` only on a lookup of that id that says it does
  not exist (a 404 or an explicit null). An error, an empty listing or an
  accepted delete is not that. A lookup can be wrong, too: if a later
  listing shows a box that is recorded as `gone`, a second lookup decides,
  and a box that exists is `live` again and is destroyed. During a launch, a
  box we did not delete ourselves is not recorded as `gone` before a delete
  has been sent for it.
- **S4.** `up` and `down` hold a lock on the directory while they run. `run`
  takes none, so that `down` can end a run in progress. `up` refuses to
  start while the snapshot shows a run that is not closed.

The file is replaced by a rename. If the rename did not happen, nothing has
changed, in memory or on disk. Once it has, memory follows the file; an
error after that (the directory could not be synced, other than on a
filesystem that cannot sync directories at all) is still reported as a
failure. A snapshot is used only if it holds together: a closed run with no
live id and no open attempt, a winner that is the box of its attempt. A run
that learns of a live id is no longer closed.

A kill between a create's answer and the write of its id loses the id. The
box still carries its tag, and the attempt is still `open`, so `down` looks
for it. If the box started and changed its own label inside that window,
only the reaper finds it.

The run's SSH key lives in `keys/<run id>/` and is removed when the run is
closed. `observations.jsonl` is unchanged. A lock cannot fence a controller
on another machine: `down --run-id` from elsewhere is for a run whose
controller is dead.

## 6. Destroying

One routine, `destroy_confirmed(ids, budget)`, is used by the launch, `down`
and the reaper:

- **D1.** It sends one delete to every id before it waits for any of them.
- **D2.** It then looks each id up, repeats the delete for those still
  there, and goes on until all are gone or the budget ends. Every provider
  call inside it is asked to take no longer than what is left of the
  budget, and after the first round of deletes none is started once nothing
  is left. These are socket timeouts: the budget is what is asked for, not
  a bound on the wall clock.
- **D3.** An error on one id (delete, lookup, or recording the result) does
  not stop the others, and that id counts as not confirmed.
- **D4.** It returns the ids it could not confirm.

The listing is used only to choose. A listing is complete or it raises. To
the checks `vast.py` has today (readable rows, positive ids, a cursor on
every page, a cursor that advances) are added: no id twice,
`total_instances` equal to the rows read on whichever pages state it, and a
`label` field in every row. A row whose label is null or empty is untagged;
a row without the field fails the listing. A timeout given to a listing is
for all its pages together.

## 7. Launch

One thread. Each pass looks at the listing and at the child processes, then
decides.

**Attempts.** A bound attempt's box goes through `booting`, `probing` and
`gating` to `healthy` or `failed`. The SSH probe and the gate are child
processes with a start, a poll, a deadline and a kill; their output goes to
files, so nothing blocks on a pipe. The gate's script and the reading of its
verdict are revision 3's. The rules for reading a box's status are revision
3's: an image error, a machine error, `exited`, the start-up timeout. One is
new: a box that is not running yet and whose status and status message have
not changed for `--stall-timeout` (300 s) has failed, as `unreachable`. A box
that keeps reporting progress gets the whole start-up timeout.

**Create.** Sent once, never retried. An id comes only from an explicit yes;
`refused` only from an explicit no (the offer is dropped, nobody is blamed).
Anything else, an interrupt in the middle of the request included, leaves
the attempt `open`.

**Adoption.** A listed box whose tag names an `open` attempt of this run is
that attempt's box: the attempt is bound to its id and carries on as a
candidate. A listed box with the run's id whose attempt is already bound to
another id, or does not exist, is recorded in `ids` and destroyed.

Rules:

- **L1.** No create is sent in a pass that had no complete listing; nor
  beyond the cap; nor after the rent cut-off (60 s before the overall
  timeout); nor once any box has reported an image error. The cut-off is
  checked again after the attempt has been written to the snapshot, just
  before the request. If that write took so long that the time it recorded
  is stale (more than 15 s), nothing is sent either: `down --abandon-unseen`
  counts from that time.
- **L2.** At most `1 + spare` attempts hold a slot. An attempt holds its
  slot while it is `open`, and while it is bound to an id that is not
  `gone`. So a create with no answer holds its slot until its box shows up
  and is confirmed destroyed, or the launch ends. `max_attempts` bounds
  creates over the run.
- **L3.** A box is chosen as winner only after its gate passed, in a pass
  whose complete listing shows it and in which a lookup of its id says it is
  running, and with at least ten minutes of its lease left. Among several,
  the cheapest.
- **L4.** The hand-off is the rename that puts the winner into the snapshot.
  If `up` fails or is interrupted after it began a hand-off, it reads the
  snapshot back. A winner that is in it is kept, whatever raised. If the
  snapshot cannot be read, the box is kept too, and nothing is written, so
  that a hand-off that did happen is not erased; `down` settles it.
  Everything else the run knows is rolled back.
- **L5.** The rollback has one budget (10 s by default) that starts when the
  failure is caught. It stops the child processes, then calls
  `destroy_confirmed` with what is left. It is an attempt, not a promise:
  ids it could not confirm stay `live` in the snapshot for `down`. A child
  is started and put on the list of children with signals held back, so
  that none can be running unknown.
- **L6.** `up` exits 0 only if a winner is in the snapshot. Losers whose
  delete is not confirmed are reported as warnings and left for `down`.
- **L7.** The launch gives up, through the rollback, if no complete listing
  was obtained for 120 s.

After an image error nothing more is rented, boxes already rented get 90 s
to prove the image is fine, and otherwise the launch fails as a
configuration error, as in revision 3.

SIGINT and SIGTERM raise in the only thread, wherever it is. A second
signal ends the rollback early. `down` always follows `up`, whatever `up`
returned.

## 8. Commands

```
gpubox up     --namespace NS --name NAME --image IMG [spec flags] [--spare 1] [--ttl-min N]
gpubox run    --script FILE [--env K=V] [--timeout-min N]
gpubox down   [--run-id ID] [--abandon-unseen]
gpubox reap   --namespace NS [--destroy]
gpubox reap   --exclusive --max-age-min N [--destroy]
gpubox search [spec flags]
```

`up`, `down` and `reap` ask the provider whose key they hold. `up` records
the account in the snapshot, pinned or not. `--account-id` (or
`GPUBOX_ACCOUNT_ID`) pins it: the command then stops, before the first
create or delete, if the key is another account's. `reap --exclusive`
requires the pin.

`up` writes the run id to the step outputs before it creates the snapshot
or rents anything, and stops if that fails: it is the one output that is
not advisory, because CI's teardown step is given nothing else.

**`down`** chooses: every id in the snapshot that is not `gone`, plus every
listed box whose tag carries the run id. Ids found in the listing are
written to the snapshot before they are deleted. Rules:

- **T1.** `down` exits 0, closes the run and removes its key only if all of
  this holds: no id is left `live`; a complete listing taken after the last
  delete shows no box with the run's tag; and no attempt is `open`. A run
  that is already closed is looked at again in the same way: a box whose
  create gave no answer can show up after the run was closed.
- **T2.** With `--abandon-unseen`, an `open` attempt stops counting once a
  complete listing, taken at least 60 s after the attempt was sent, shows no
  box with its tag. `down` waits for that moment itself. It then reports
  the attempt as unseen and left to the timer and the reaper, which is not
  the same as gone. Without the flag `down` exits 75 while an attempt is
  open.
- **T3.** `down` uses a snapshot only for the run it was asked about. If
  the key belongs to another account than the one the snapshot names, that
  is an error, not a clean result. A snapshot that cannot be read is an error too;
  with `--run-id` the boxes carrying the tag are still destroyed, and the
  exit is 75.
- **T4.** `down --run-id` for a run this directory has no snapshot of works
  by tag alone, touches no snapshot and no key, and says so.

In CI the teardown step passes `--run-id` from `up`'s output and
`--abandon-unseen`, on the subaccount. An empty run id is an error, so a
job whose `up` never got as far as a run id cannot tear down someone else's.

**`run`** is revision 3's, with two changes. The snapshot knows the winner's
machine through its attempt, so a death can be recorded. And a machine is
blamed for a lost connection only if the provider still shows the instance
in a state that means it is over (`exited`, `offline`). An instance that no
longer exists was destroyed by someone (`down` here or elsewhere, the
timer, the reaper), which says nothing about the machine.

Exit codes are revision 3's.

## 9. The timer and the reaper

| Chance | Runs | Ends a box when | Does not cover |
|---|---|---|---|
| `up`'s rollback, then `down` | on the controller; in CI an `always()` step | the launch fails or the job ends | a runner that is killed; a box nobody has seen yet |
| timer in the box | in the container | the tag's expiry passes | a container that never started; code that kills the timer; a wrong clock on the box |
| `reap --exclusive`, every 30 minutes | a scheduled workflow | no valid tag, expiry passed, or too old | a box the listing does not show; a schedule that stops |

**The timer** is started first in the bootstrap, before a missing sshd is
installed. It compares the clock with the absolute expiry every 30 s, so a
container that restarts does not start counting again, and once the expiry
has passed it sends the delete every 30 s for as long as it runs, whatever
the answers say. This changes the create payload away from the pinned
recording of the old launcher, in the timer only.

**`reap --exclusive`** is for the subaccount, where every instance is
declared disposable. Rules:

- **R1.** It checks the account first and does nothing on another one.
- **R2.** It chooses an instance if its tag's expiry passed more than 60 s
  ago, or if the provider's start time for it is more than `--max-age-min`
  ago.
- **R3.** It also chooses an instance whose label is not a valid tag, but
  only if a second complete listing, at least 30 s after the first, shows
  the same id still without one. A wrong delete of a box seconds old would
  break a launch in progress. An instance that carries a tag at the second
  look is judged by that tag's expiry.
- **R4.** Without `--destroy` it prints what it chose and changes nothing.
  With it, the chosen ids go to `destroy_confirmed`.
- **R5.** It exits non-zero if a listing could not be read or a delete was
  not confirmed. If it is the second listing that fails, what the first
  chose is destroyed all the same. An instance with a valid, unexpired tag
  and no usable start time is reported: the age rule cannot apply to it.

`--max-age-min` must exceed the longest `--ttl-min` in use. A box whose
lease has ended is destroyed even if a job is still using it; that is what
a lease means, and the reason the job's time limit is kept below the TTL.

**`reap --namespace NS`** is for an account shared with other work. It
considers only instances whose tag parses to the namespace, chooses the
expired ones, and follows R4 and R5. A box whose label was changed is
invisible to it.

**What can be claimed.** No lifetime is guaranteed. On the subaccount the
reaper destroys a box at the first pass that both sees it and finds it
untagged, expired or too old, provided the delete takes effect. For a box
that is listed from the start, that pass is the first one after its expiry
(plus the 60 s margin), or after `--max-age-min` if it rewrote its tag and
the provider's start time cannot be changed from the box; how long after
depends on the schedule, which can run late. What is outside that:

- a box no listing shows cannot be found by anything we run;
- a delete that does not take effect fails the reaper workflow every half
  hour until someone looks;
- a reaper that has stopped running is noticed only from outside. The
  reaper workflow pings an external dead-man check after each successful
  pass with `--destroy` (a line in the workflow, not code here).

The subaccount's prepaid balance limits what a leak can cost in total, if
the subaccount turns out to have a balance of its own that is not topped up
automatically. It is a spending limit, not a way of destroying anything.

For QuACK CI the rate is at most $2.50 an hour for each box, with at most
two boxes holding a slot in a launch.

## 10. Labels as ownership

Revision 1 was blocked on "labels cannot establish ownership": a label can
be changed by us, by a teammate, and by code on the box. This revision
relies on the tag more than revision 3 did, so here is what each can do.

- **A box gets a run's tag that is not that run's.** Anyone with the
  account's key can do that, and so can code on a box, to its own instance.
  `down` or the reaper then destroys that box, or a launch in progress
  adopts it if its attempt is still open. So a tag is trusted exactly as far
  as everything that can write labels on the account: on the subaccount,
  CI's own boxes, all disposable. On a shared account that trust is wider
  and the consequences are real.
- **Code on the box removes or changes its tag.** `down` destroys it by its
  id from the snapshot, loser or winner. If the runner is gone, the
  exclusive reaper destroys it as untagged. On a shared account with the
  runner gone it leaks until someone looks: the residual risk revision 3
  stated, no larger.
- **Two callers use one name.** Their run ids differ, so neither's `down`
  touches the other's boxes, and a late teardown of an old run cannot hit a
  new one.

## 11. Gate and reputation

The gate's checks are revision 3's, section 6, with its rule that only a
verdict the script itself gave is recorded against a machine.

Blame is unchanged: a machine is recorded as failed only if another
candidate with the same image, request and gate version became healthy in
the same launch.

The blocklist is one pure function of `observations.jsonl`:

| Reason | Blocked for |
|---|---|
| `busy_gpu` | 2 hours |
| `unreachable`, `no_network`, `died` | 1 day |
| `runtime_error`, `wrong_hardware`, `cuda_init`, `compute_mismatch` | 14 days |

A later `healthy` row clears the first two. If fewer candidates than the cap
remain, the first two are re-admitted, soonest to expire first. Offers are
taken cheapest first, one machine per host. Offers in excluded countries
(`--exclude-country`, default `CN`) are dropped before that, on the client,
so the search request itself is unchanged. Doubling on repeats goes away.
The row format is kept, so the store can be wired back without a migration.

## 12. What the launcher asks of a provider

```python
offers(query) -> list[dict]          # candidates, in the order to try
create(offer_id, payload) -> id      # raises Refused or Unknown; sent once
instances() -> list[dict]            # the whole account, complete, or raises
instance(id) -> dict | None          # None only on an explicit "does not exist"
destroy(id) -> None                  # a box already gone is not an error
account() -> str                     # whose key this is
```

This is the Vast client as it is, plus `account`. It is not a contract for
other providers, and the launcher still reads Vast's own row fields; a
second provider is when rows get a common shape. From Runpod's
documentation, untested, the differences to design for then:

- a candidate is a data center and a GPU type; the machine is known only
  after placement, so a blocked machine is found after the pod exists;
- the tag would go in the pod's `name`, which need not be unique;
- a 400 on create covers both a bad request and no capacity, so a refusal
  cannot tell a configuration error from an empty market;
- whether a pod can terminate itself with the key it is given is unclear,
  and `--terminate-after` is reported as never enforced. Without a timer in
  the box there is one chance fewer.

## 13. What changes in the code

| Revision 3 | Revision 4 |
|---|---|
| `journal.py`: event log, replay, damage rules, rotation | `state.py`: a snapshot, the lock, tags |
| `intent`, `unknown`, `abandoned` | `open`; `--abandon-unseen` waits 60 s itself |
| `match_lost_box`: one label, machine, image, start time | the tag, which names run and attempt |
| thread pool, futures, late worker errors | one loop, children polled |
| signals as flags | `KeyboardInterrupt`, one handler that reads the snapshot |
| commit only with every loser confirmed gone | hand-off when the winner is in the snapshot; losers reported |
| `reap --legacy-pattern` | none; old labels are drained on the old account |
| `reap` by age in a namespace | by expiry in a namespace; `--exclusive` for the subaccount |
| strike doubling, store sync | fixed cooldowns; store parked |
| loser cleanup, rollback, `down`, `reap`: four delete loops | `destroy_confirmed` |

Kept: run ids with a random part; ownership of every id we know until a
lookup confirms it gone; the state lock; the cap that counts creates with no
answer; in `vast.py` the strict reading of every answer and the create
classification; SSH options; exit codes; the gate script.

Tests follow the rule numbers above. The fake provider gains a listing that
is stale while well formed, a delete that does not take effect, a box that
relabels itself, and a start time that is missing. The kill-at-each-step
tests stay, with a second process doing the teardown.

## 14. To verify with the provider

Before `reap --exclusive --destroy` is enabled anywhere:

1. **The subaccount is separate, shown in both directions.** With one box
   rented on the parent and one on the subaccount at the same time: the
   subaccount's key lists only its own; it cannot destroy the parent's; and
   `account()` returns different ids for the two keys. If the subaccount's
   listing shows the parent's box, exclusive destruction stays off and a
   separate account is needed.
2. **The label is on the instance in every listing that shows it**, from
   the first one. R3's second look covers a short gap, not a long one.
3. **`start_date`**: present while a box is loading or not; unchanged by a
   stop and start, by a relabel, and by anything the box's own key can do.
   If it can be changed from the box, the age rule does not bound a box that
   rewrites its tag.

Before relying on the balance as a limit:

4. Whether the subaccount has its own prepaid balance, whether anything
   tops it up, and what a stopped instance costs. The amount to keep there
   follows from that: concurrent launches, times two boxes, times $2.50 an
   hour, times how long a leak may last.

Shapes a detail:

5. Whether listing rows carry `actual_status`, `ports` and `public_ipaddr`.
   The implementation looks each candidate up by id for those, as revision 3
   does, and uses the listing only for ids and labels.
6. The rate limits of the listing and the lookup, to set the poll interval
   (5 s today).
7. `cancel_unavail: true` on create, documented as failing fast when an
   offer is gone.

Migration. The old launcher and its reaper keep the old key until the last
`quack-ci-` box is gone. Boxes left by revision 3's live tests, if any, are
on the main account under revision 3 labels and are checked for once by
hand.

Acceptance for QuACK CI is revision 3's section 9, with: the reaper workflow
runs `reap --exclusive` on the subaccount and pings the dead-man check;
`GPUBOX_ACCOUNT_ID` is set in both workflows; the teardown step passes the
run id and `--abandon-unseen`; the create payload equals the recording
except for the label and the timer.

## 15. What the reviews changed

First review (blocked), thirteen findings, all accepted:

| Finding | Where it went |
|---|---|
| 1. An empty listing does not confirm a known box is gone | S3, D2 |
| 2. The lifetime bound does not follow | section 9, "What can be claimed" |
| 3. A refused second run's teardown destroys the first | random run id (section 4); T3 |
| 4. Exclusive file creation does not replace the lock | S4 |
| 5. Only the winner's id survived | `ids` (section 5) |
| 6. The cap can be exceeded by several boxes | L2; attempts numbered in the tag |
| 7. The exception boundary did not implement the interruption rule | L4; S1 |
| 8. Account scope must be verified both ways and pinned before the first create | `--account-id`; section 14, item 1 |
| 9. A reaper pass can delete a launch in progress | R2 (60 s margin), R3; a missing label field fails the listing |
| 10. The loop's retries and timing were not executable as written | L1, L7; children with deadlines |
| 11. The timer in the box is weaker than stated | section 9, the timer |
| 12. `run` could not record a death | section 8, `run` |
| 13. Defer the cross-provider contract; drain old labels | section 12; section 14, migration |

Second review (blocked), ten findings, all accepted:

| Finding | Where it went |
|---|---|
| 1. The handler could roll back a published winner | L4: the handler reads the snapshot |
| 2. A clean sweep was not a clean `down` | T1: a listing after the last delete |
| 3. Ids found during teardown were not recorded | `ids`; S2 |
| 4. Recovery stated as a timed guarantee; stopping is not destroying | section 9: no lifetime is claimed, and the balance is a spending limit, conditional on item 4 of section 14 |
| 5. Bare `down` after a refused `up`; account mismatch read as clean | T3; CI passes the run id |
| 6. The rollback budget was not enforced where time is spent | D1, D2, L5 |
| 7. The two-look rule could not live in a shared sweep | principle 3; R3 |
| 8. An open attempt had no clock | `sent` (section 5); T2 |
| 9. Pseudocode and prose disagreed on reads, winner and image errors | the pseudocode is gone; L1, L3, L7 |
| 10. Trust and timer assertions overstated | sections 4, 5, 9, 10 |

Third review, of the implementation (blocked), eleven findings. The
reviewer could not run the tests. All accepted but one part:

| Finding | What was done |
|---|---|
| 1. A closed run was never looked at again, so a box that appeared after an abandoned create was never deleted | T1: `down` lists for a closed run too; a run that learns of a live id is no longer closed |
| 2. A hand-off that could not be read back was treated as no hand-off | L4: kept, and nothing written |
| 3. A run id that could not be written to the step outputs did not stop the launch | Section 8: it does now, before the snapshot exists |
| 4. A run without a pinned account stored none, so `down` with another key could read as clean | The account is always recorded and compared (T3) |
| 5. A wrong `gone` could never be corrected | S3: a listing that shows such a box leads to a second lookup; a delete is sent before a box is recorded as `gone` |
| 6. The rollback budget did not cover stopping the children, and calls went on after it was spent | L5, D2. The half-second floor on a request's timeout stays: no request completes in less |
| 7. Some damaged snapshots passed as sound; every directory-sync error was swallowed | Section 5 |
| 8. Time checks came before the work that could invalidate them | L1 (after the write), L3 (lease checked when the box is chosen). Not done: comparing the expiry in the listed tag with the snapshot's, which differ only if a label was rewritten |
| 9. Only the last page's total was checked; a listing could take a timeout per page | Section 6. L7 is unchanged: with the listing bounded, a gap can only come from failures |
| 10. A failed second look spared what the first look chose; a late tag was not judged by its expiry | R3, R5 |
| 11. A child could be running unknown after a signal; the start of a child's output was kept, not its end | L5; the last megabyte is read |
| `--max-age-min` accepted and ignored with `--namespace` | Refused |

Taken from the reviewer's answers to the open questions: `down` exits 75
for an unseen create unless `--abandon-unseen` is given; a stopped reaper is
caught by an external dead-man check.

Not done, and noted here so that it is not taken for done: a 429 from the
provider is treated like any other failed read (the pass is skipped and the
next one comes `poll` seconds later), with no longer back-off.

## 16. Open questions for Tri

1. How much credit to keep on the subaccount, once item 4 of section 14 is
   known.
2. `reap --exclusive` destroys anything untagged on that account. Confirm
   nobody will rent there by hand.
3. Which dead-man service the reaper workflow pings.
