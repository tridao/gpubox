# gpubox design, revision 5

Status (2026-11-23): implemented on the branch `rev4`. Tested against a fake
provider; individual API calls were exercised against Vast by the
measurements in `experiments/`, but `up`, `run`, `renew` and `down` have not
yet been run end to end on a real box (section 9).

Revision 4 tried to be correct against every case three reviews could
construct, and grew to about 2,750 lines. Revision 5 keeps its structure (a
tag on the box, one loop, one delete routine) and removes what only served
cases that Tri decided on 2026-11-23 to assume away. Section 2 lists those
assumptions, why each was made, and what it costs when it is false. The
history is in git and in QuACK's `AI/gpubox_design.md` (revision 3).

## 1. What it does

`gpubox up` rents a GPU box on Vast.ai for a CI job or for development:

1. Search offers; drop excluded countries and recently failed machines;
   take the cheapest, one machine per host.
2. Rent up to `1 + --spare` boxes at once (2 by default) and wait for each to
   boot, answer SSH and pass the health gate. A box that fails is destroyed
   and replaced, up to `--max-attempts` rentals (6; an offer that was already
   gone does not count).
3. The first box through the gate is the winner; the others are destroyed.
4. `gpubox run` runs a script on it; `gpubox down` destroys it.

Every box carries a lease. When it ends the box destroys itself, and a
scheduled reaper destroys whatever is left.

## 2. Assumptions

Each of these is false sometimes. The right-hand column is what happens then.

| # | Assumption | Why it was made | When it is false |
|---|---|---|---|
| A1 | Labels are trustworthy: nothing renames a box. | Guarding against it needed a dedicated account whose reaper destroys untagged boxes, with rules to avoid killing boxes seconds old. The only realistic source is hostile code approved into CI. | A renamed box is invisible to `down --run-id` and to the reaper. `down` from the original state directory still destroys it by id. Otherwise it runs until someone notices the bill. |
| A2 | The lease bounds a leak; `up` and `down` are best effort. | Tracking every rare case precisely (a create with no answer, a lookup that lies, an interrupted hand-off) cost several hundred lines to save, at worst, one box running to the end of its lease (about $2.50 for CI). | A box that `down` missed runs until its lease ends. |
| A3 | The state directory is sound. | It is written by one process under a lock; consistency checks on it found nothing in practice. | An unreadable state file: `down --run-id` works from the tag; the box's address and key are lost. |
| A4 | The listing shows a box once its create has returned. | 65 of 65 boxes in `experiments/pull_time` did. | `down` could miss a box whose id it does not hold. It lists twice, a few seconds apart, to cover a short lag; beyond that, A2. |
| A5 | The provider does not give an instance id to a second box. | `down` deletes the ids in the state file without looking at their tags, so that it reaches a box whatever its label. | A reused id would be someone else's box, deleted. |

Not assumed, because being wrong is expensive or the code is small:

- **A create may get no answer.** The attempt keeps its slot for the rest of
  the launch, in memory, so a bad minute at the provider cannot rent six
  boxes.
- **A delete is not done until a lookup of that id says the box does not
  exist.** An accepted delete, an error or an empty listing is not that.
- **The provider's answers may be malformed.** Every answer is read
  strictly; a listing is complete or it raises.
- **One host may be wrong about the image.** See section 4.
- **A box may die just after its gate.** It is looked up once more before
  the hand-off.
- **The search may return what was not asked for.** The price cap and the
  cheapest-first order are applied again locally before anything is rented.

## 3. The tag and the lease

```
run id   <namespace>.<name>.<12 random hex>      quack-ci.19283746-1.3f9c2a1b5d7e
tag      <run id>.a<attempt>.<expiry>            quack-ci.19283746-1.3f9c2a1b5d7e.a2.1795400000
```

The tag is the instance's label, set by the create request. `expiry` is Unix
time: `up`'s start plus `--ttl-min`. Parsing is exact; a label that does not
parse is not ours. The random part means two runs never share an id, so a
late teardown of one cannot touch another.

Three things end a box:

| | Runs | Acts when |
|---|---|---|
| `down` | on the controller; in CI an `always()` step | asked |
| timer in the box | in the container, started before anything else | the time in `/etc/gpubox-expiry` has passed |
| `reap --namespace NS --destroy` | a scheduled job | the expiry in the tag passed more than 60 s ago |

The timer uses the key Vast injects into each container to delete its own
instance, and repeats the delete every 30 s. The expiry file is created from
the create request unless it already exists (a container restart must not
undo a renewal); content that is not a number falls back to the original
expiry.

`gpubox renew --ttl-min N` moves the lease: it rewrites the expiry file over
SSH, then relabels the box with the new expiry, then updates the state file.
The file comes first so that if the relabel is refused, the reaper still goes
by the old expiry and the box can end early, not late. Two outcomes are not
clean: a relabel that took effect but whose answer was lost, and a state file
that could not be written afterwards. Both leave the box renewed while this
directory shows the old expiry; `renew` says so, and running it again
settles it.

What is not covered: a box whose container never starts (stuck pulling) has
no timer, so only `down` and the reaper end it; and nothing finds a box that
no listing shows.

CI and dev boxes use different namespaces, and CI its own subaccount. The
reaper only looks at one namespace, so it works the same on a shared account.
`--account-id` pins the account: with it, `up`, `down`, `renew` and `reap`
refuse a key that belongs to another one.

## 4. The launch loop

One thread. Each pass:

1. Look each candidate up by id and advance it: `booting` (wait for
   `running`), `probing` (SSH answers), `gating` (the health gate), then
   `healthy` or `failed`. Probes and gates are child processes that are
   polled, never waited for.
2. Send a delete for each failed box and look it up. It keeps its slot until
   the lookup says it is gone, so there are never more than `1 + spare`
   billable boxes that we know of.
3. If a candidate is healthy, look it up once more (still running, at least
   10 minutes of lease left) and hand over the cheapest.
4. Otherwise rent into free slots, until 60 s before `--overall-timeout`.

A box fails when: the provider reports a machine error or `exited`; it is not
reachable after `--startup-timeout` (300 s); its status has not changed for
`--stall-timeout` (300 s) while it is still loading; or the gate says so.
Stall and timeout values come from `experiments/`: pulls that finished went
silent for at most 166 s, and the ones that never finished for 600 s or more.

**Image errors.** A host that says the image cannot be pulled is a failed
box, blamed on nobody. When two different hosts have said so (boxes whose
host the offer does not name do not count as different), the launch
stops as a configuration error (exit 2). With a wrong tag that is about 40 s
in; a single misconfigured host is simply replaced. If the launch ends with
no box and any host blamed the image, that is what is reported.

A box that is running and whose sshd refuses the run's key is treated the
same way: our own sshd is started with that key, so a refusal means the image
runs something else in place of our start-up command, and waiting for the
start-up timeout cannot help.

**The hand-off** is the write that puts the winner into the state file.
Before it, a failure or a signal rolls back everything the launch rented;
after it the winner is kept. The rollback aims at 10 s (what a cancelled CI
step gets): it stops the child processes, sends every delete before waiting
for any, and gives up at the deadline. The timeouts it passes on are socket
timeouts with a floor of half a second, so 10 s is a target, not a bound.
`down` must always follow `up`.

When `up` ends, either way, it destroys the ids it holds, using at most half
its time on them, then lists once for anything else carrying the run's tag,
which is how the box of a create that got no answer is found.

## 5. Teardown

`down` works in rounds for up to a minute: a short delete-and-lookup round
for every id known so far (first those in the state file), then a listing,
whose boxes with the run's tag join the next round. A box that will not go
away therefore cannot keep one found later from being deleted. It is done
when two listings in a row, a few seconds apart, show nothing with the run's
tag (a box already confirmed gone that a lagging listing still shows does
not count as nothing). Then it exits 0, closes the run, and removes the
run's key and SSH entry.

If it cannot get there it exits 75 and prints the ids that are left, the
command to retry, the console URL, and when the boxes' own lease ends.

`down --run-id ID` for a run the directory has no state for does the same
from the tag alone. That is the CI case: a runner whose disk is gone.

## 6. Local state

`<state-dir>/run.json`, replaced whole at each change:

```
run, expiry, account, meta     what was asked for
ids                            every instance id a create returned
winner                         id, attempt, host, port, machine, host_id
closing, closed
```

Plus `keys/<run id>/` (an SSH key and known-hosts file per run), a lock
taken by `up`, `down` and `renew`, and `observations.jsonl`.

`up --ssh-config` writes a `Host <name>` block for the winner to
`~/.ssh/gpubox_config` and makes sure the first directive of `~/.ssh/config`
is `Include gpubox_config`. Nothing else in the user's file is changed; if it
is a symbolic link, its target is written. Boxes in different state
directories share these files, so changes are made under a lock of their own.

## 7. Gate and blocklist

The gate is revision 3's: GPU count, model and VRAM; required URLs
reachable; GPU idle before our first CUDA context; CUDA initialises; per GPU
a matmul checked against the CPU, and 80% of free VRAM allocated and filled,
with its first and last bytes read back.

A machine is recorded as failed only if another candidate became healthy in
the same launch. `observations.jsonl` then blocks it for 2 hours (GPU busy),
1 day (unreachable, no network, died, stalled) or 14 days (wrong hardware,
CUDA or runtime errors); a later healthy run clears the first two.

The shared reputation store (`store/`, `store.py`) is parked: kept and
tested on its own, called by no command. It will be wired back later.

## 8. What was removed from revision 4

| Removed | Covered by |
|---|---|
| `reap --exclusive`, its two-look and max-age rules | A1 |
| attempt states on disk, `--abandon-unseen`, exit 75 for a create with no answer | A2; the slot is held in memory |
| correcting a lookup that wrongly said "gone" | A2 |
| keeping a winner whose hand-off cannot be read back | A2 |
| state-file consistency checks, directory-sync error handling | A3 |
| a listing on every pass of the loop (adoption, blind timeout) | candidates are looked up by id; one listing when `up` ends |
| the 90 s grace after an image error | the two-host rule |

## 9. Not yet verified against the provider

- `PUT /instances/{id}/` with `{"label": ...}` (used by `renew`): from the
  documentation only.
- The timer in the bootstrap: its loop is tested in bash with a stand-in for
  the delete; it has never fired on a real box.
- Whether `/etc/gpubox-expiry` survives a container restart on Vast.
- Whether a Vast subaccount has its own instance list and key.
- `up`, `run`, `renew`, `down` end to end.

Seen against the real API on 2026-11-23: the strict listing (204 calls), the
account lookup, create, lookup and delete; the listing answers 429 at about
one call every 3 s, which is why the loop polls by id.
