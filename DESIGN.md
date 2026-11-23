# gpubox design

gpubox rents a GPU box on Vast.ai for a CI job or for development, checks
that it works, and makes sure it is destroyed afterwards. This document says
how, what it assumes, and what was decided against.

## 1. The problem

Four things cannot be designed away:

1. **Hosts are unreliable.** A rented box may never start, or start broken.
2. **A create request is not idempotent**, and its outcome can be unknown.
3. **The controller can die at any moment**, and in CI its disk dies too.
4. **A leaked box is silent.** It bills until someone looks.

The answers, in the same order: race a few hosts and health-check them; never
resend a create, and keep a slot for one that got no answer; put ownership on
the box itself, as a tag, so that anything can clean up; give every box a
lease that ends it without help.

## 2. What happens

`gpubox up`:

1. Search offers. Drop excluded countries, offers above the price cap and
   recently failed machines; take the cheapest first, one machine per host.
2. Rent up to `--count + --spare` boxes at once (2 by default). Each has to
   boot, answer SSH and pass the health gate. One that fails is destroyed and
   replaced, up to `--max-attempts` rentals (`2 * count + 4`, so 6 for one
   box; an offer that was already gone does not count).
3. Once `--count` boxes are through the gate, they are the winners. The
   others are destroyed. A launch that cannot get that many destroys the
   healthy ones too: all or nothing, since a job split over N boxes cannot
   use fewer.

`gpubox run` runs a script on the winner, `gpubox renew` moves the end of its
lease, `gpubox down` destroys everything the run rented, and
`gpubox reap --namespace NS` destroys boxes of a namespace whose lease has
ended.

## 3. Assumptions

Each of these is false sometimes. The last column is what happens then.

| # | Assumption | Why | When it is false |
|---|---|---|---|
| A1 | Nothing renames a box. | Guarding against it needs a dedicated account whose reaper destroys everything untagged. The realistic source is hostile code that was approved to run in CI. | A renamed box is invisible to `down --run-id` and to the reaper. `down` from the original state directory still destroys it by id. |
| A2 | The lease bounds a leak; `up` and `down` are best effort. | Chasing every rare case precisely costs far more code than the one box-lease it saves. | A box that `down` missed runs until its lease ends. |
| A3 | The state directory is sound: one writer, under a lock. | | With an unreadable state file, `down --run-id` works from the tag; the box's address and key are lost. |
| A4 | The listing shows a box once its create has returned. | Observed for every box rented while testing. | `down` could miss a box whose id it does not hold. It lists twice, seconds apart, for a short lag. |
| A5 | An instance id is never given to a second box. | `down` deletes the ids in the state file without looking at their tags, to reach a box whatever its label. | A reused id would be someone else's box. |
| A6 | Every holder of a reputation-store token is trusted. | The store only ever makes a launcher skip an offer. | A token holder can make everyone skip up to 500 machines a day. |

Not assumed, because being wrong is expensive or the check is small:

- **A create may get no answer.** The attempt keeps its slot for the rest of
  the launch, so a bad minute at the provider cannot rent six boxes.
- **A delete is done only when a lookup of that id says the box does not
  exist.** An accepted delete, an error or an empty listing is not that.
- **The provider's answers may be malformed.** Each is read strictly; a
  listing is complete or it is an error.
- **The search may return what was not asked for.** The price cap and the
  cheapest-first order are applied again locally.
- **One host may be wrong about the image** (section 5).
- **A box may die just after its gate.** It is looked up once more before
  the hand-off.

## 4. The tag and the lease

```
run id   <namespace>.<name>.<12 random hex>      quack-ci.19283746-1.3f9c2a1b5d7e
tag      <run id>.a<attempt>.<expiry>            quack-ci.19283746-1.3f9c2a1b5d7e.a2.1795400000
```

The tag is the instance's label, set by the create request. `expiry` is Unix
time: when `up` started plus `--ttl-min`. Parsing is exact; a label that does
not parse is not ours. The random part means two runs never share an id, so a
late teardown of one cannot touch another.

Three things end a box:

| | Runs | Acts when |
|---|---|---|
| `down` | on the controller; in CI an `always()` step | asked |
| timer in the box | in the container, started before anything else | the time in `/etc/gpubox-expiry` has passed |
| `reap --namespace NS --destroy` | a scheduled job | the expiry in the tag passed more than 60 s ago |

The timer uses the key Vast gives each container to delete its own instance,
and repeats the delete every 30 s. The expiry file is created from the create
request unless it already exists, so that a container restart does not undo
a renewal; content that is not a time falls back to the original expiry.

`renew` rewrites the expiry file over SSH, then relabels the box, then
updates the state file. The file comes first so that if the relabel is
refused, the reaper still goes by the old expiry and the box can end early,
not late. If the relabel took effect but its answer was lost, or the state
file could not be written, the box is renewed while the directory shows the
old expiry; `renew` says so, and running it again settles it.

Not covered: a box whose container never starts (stuck pulling its image)
has no timer, so only `down` and the reaper end it. Nothing finds a box that
no listing shows.

Use one namespace per purpose (CI, dev), since the reaper works per
namespace. `--account-id` pins the provider account: `up`, `down`, `renew`
and `reap` then refuse a key that belongs to another one.

## 5. The launch loop

One thread. Each pass:

1. Look each candidate up by id and move it on: `booting` (wait for
   `running`), `probing` (SSH answers), `gating` (the health gate), then
   `healthy` or `failed`. Probes and gates are child processes that are
   polled, never waited for.
2. Send a delete for each failed box and look it up. It keeps its slot until
   the lookup says it is gone, so there are never more than `count + spare`
   billable boxes that we know of.
3. If `--count` candidates are healthy, look them up once more (still
   running, at least 10 minutes of lease left) and hand over the cheapest
   `--count`. A healthy box waits, billing, for the others; one that dies
   meanwhile is caught by that lookup and replaced.
4. If `--count` healthy boxes can no longer be had (those healthy or on
   their way, plus as many as the offers and attempts left allow, are too
   few), give up now rather than rent or wait for nothing.
5. Otherwise rent into the free slots, until 60 s before `--overall-timeout`.

A create answered with an id an earlier create already returned is treated
like one that got no answer: it keeps its slot, and its box, if there is
one, is found by its tag.

A box fails when the provider reports a machine error or `exited`; when it
is not reachable after `--startup-timeout` (300 s, counted from the create,
so it includes the image pull); when it is still loading and its status has
not changed for `--stall-timeout` (300 s); or when the gate says so.

**The image is the problem** when a host says it cannot be pulled, or when a
box is running and its sshd refuses the run's key (our own sshd is started
with that key, so the image must be running something else). Either fails
that box, blamed on nobody. When two different hosts have shown it, the
launch stops as a configuration error (exit 2): with a wrong tag that is
about 45 s in, and one misconfigured host is simply replaced. Boxes whose
host the offer does not name do not count as different hosts. Once any box
of the launch has answered SSH with the run's key, the image is known to
work, and what hosts say about it no longer stops the launch.

**The hand-off** is the write that puts the winners into the state file, all
in one replacement of the file. Before it, a failure or a signal rolls back
what the launch rented; after it the winners are kept. The rollback aims at 10 s, which is what a cancelled CI
step gets: it stops the child processes, sends every delete before waiting
for any, and gives up at the deadline. It is an attempt, so `down` must
always follow `up`.

A healthy winner is kept even if a loser's delete is not confirmed:
destroying it would remove no leak, and would need the call that is failing.

When `up` ends, either way, it destroys the ids it holds and then lists once
for anything else with the run's tag. That is how the box of a create that
got no answer is found.

**SSH.** The bootstrap writes the run's public key to a file of its own and
starts sshd reading only that file. Root's `authorized_keys` cannot be used:
Vast replaces it with the account's registered key as the container starts.
Each run has its own key pair, removed when the run is closed.

## 5a. Jobs

A job is a script started on the box in a session of its own (`setsid`), by a
small wrapper that waits for it and writes its exit code to a file. Nothing
that happens to an SSH connection reaches it, so a dropped connection, a
killed `gpubox run` or a cancelled CI step leaves it running; `down` or the
lease end it with the box. Its directory, `/var/lib/gpubox/jobs/<id>/`, holds
the script, the log, the wrapper's pid and start time, the job's process
group, and once it has ended its exit code.

The controller follows a job with a program on the box that prints the log
from a byte offset and ends with one status line on stderr (an exit code,
`running`, `lost` or `none`). The controller counts the bytes it has; when a
connection drops without that line, it asks the provider whether the box is
still there and starts again from its offset. So every byte arrives once,
however often the connection drops.

The wrapper owns the job's end. At the timeout, kept on the box so that it
holds with nobody connected, and on `gpubox kill`, which signals the wrapper
rather than a remembered process group, it sends TERM to its child's process
group, then KILL to whatever is left after 10 s, and only then writes the exit
code (124 at the timeout). A job is `lost` when its wrapper is gone and it has
no exit code: the container restarted, or someone killed the wrapper. A
wrapper counts as alive by its pid, its start time (so a process that got its
pid after a restart is not taken for it) and not being a zombie.

The caller's variables go over SSH stdin into a 0600 file in the job's
directory. Only the wrapper's child, which then execs the script, reads them,
and deletes the file; nothing the caller passes can change what the wrapper
does. They are on no command line, here or on the box, and not in the
provider's instance settings. The start reports success only once the wrapper
has started the script; otherwise it deletes the file and fails.

The follower's status line counts only when it is whole and the follower
exited cleanly: `exit 1` cut short may have been `exit 124`. A connection that
brought output, or held for 30 s, ends an outage; `--reconnect-min` counts
from there. Job ids are the start time to the millisecond, so the newest job
is the last name.

**The container's start.** gpubox sets the container's entrypoint to bash
(Vast's `onstart` in the args launch mode, which is what its own CLI sends for
`--entrypoint`) with the bootstrap as its arguments. An image's own
ENTRYPOINT would otherwise run first, with the bootstrap as its arguments, and
one that does not end in `exec "$@"` never starts our sshd. The bootstrap also
saves the container's environment, which sessions over SSH do not inherit, to
`/etc/gpubox/env.sh` (POSIX syntax; login shells and every gpubox command on
the box load it), and installs `gpubox-destroy-self`, which writes a past time
to the expiry file: the lease timer destroys the box within 30 s.

## 5b. Interruptible boxes

`--interruptible` searches Vast's bid offers and bids `--bid-markup` times an
offer's minimum bid; the price cap and the cheapest-first order apply to the
bid plus the disk. What changes:

- An outbid box is stopped, not destroyed, and bills its disk; with its
  container stopped, its lease timer cannot fire. `down` and the reaper
  destroy it by tag whatever its state, so a scheduled `reap` is the bound.
- Being outbid is the market, not the machine: a box that exits, stalls, or
  never reaches `running` within the start-up timeout fails without blame,
  and a stopped box is never recorded as `died`.
- The job dies with its container (Vast kills it 0.1 s after SIGTERM). When
  the bid wins again the container restarts without it; `wait` sees the
  wrapper gone and reports the job `lost`.
- A resumed box may answer on another port. When SSH fails, the provider is
  asked about the box; a running box's address is taken from the answer and
  written to the state file (unless a teardown holds the directory).
- Two of our own bids on one machine outbid each other, so machines the
  account already has an instance on are skipped.

## 6. Teardown

`down` works in rounds for up to a minute: a short delete-and-lookup round
for every id known so far (first those in the state file), then a listing,
whose boxes with the run's tag join the next round. A box that will not go
away therefore cannot keep one found later from being deleted. It is done
when two listings in a row, seconds apart, show nothing with the run's tag.
Then it exits 0, closes the run, and removes the run's key and SSH entry.

Otherwise it exits 75 and prints the ids that are left, the command to
retry, where to delete them by hand, and when their lease ends.

`down --run-id ID` for a run the directory has no state of does the same
from the tag alone. That is the CI case: a runner whose disk is gone.

## 7. Local state

`<state-dir>/run.json`, replaced whole at each change:

```
run, expiry, account, meta     what was asked for
ids                            every instance id a create returned
winners                        per box: id, attempt, host, port, machine, host_id, gpu
closing, closed
```

Beside it: `keys/<run id>/` (the run's SSH key and known-hosts file), a lock
taken by `up`, `down` and `renew`, and `observations.jsonl`.

Box `i` of a run of `n` is called `<name>` if `n` is 1, else `<name>-<i>`.
A version 2 file (one `winner`) is read as a run of one box.

`up --ssh-config` writes a `Host` entry for each winner, in one block per run, to
`~/.ssh/gpubox_config` and makes sure the first directive of `~/.ssh/config`
is `Include gpubox_config`. Nothing else in that file is changed, and a
symbolic link is followed, not replaced. Boxes in different state
directories share these files, so they are changed under a lock of their own.

## 8. Health gate and blocklist

The gate runs on the box with the image's own Python: GPU count, model and
VRAM; required URLs reachable; the GPU idle before our first CUDA context.
Where the image has a CUDA build of PyTorch, also: CUDA initialises, and per
GPU a matmul checked against the CPU and 80% of free VRAM allocated and
filled. Without PyTorch, or with a CPU-only build, those are skipped: that
says nothing about the machine. A CUDA build that cannot use the GPU does,
unless the host's driver is for an older CUDA major version than the build
(a cu132 PyTorch on a 570 driver, CUDA 12.8, was seen to fail this way;
on 580 and 590 drivers, CUDA 13.0 and 13.1, it ran, and so did QuACK's
kernels). That is the caller's choice of host, not a fault of the machine:
the box fails without blame, the message says to raise `--min-cuda`, and
the launch rents no more offers whose driver is that old.
It checks the machine, not the project under test.

The caller's own checks (`--gate-script`) run after the gate passes. They are
the caller's code, so a failure blames nobody unless the script says, in the
gate's own verdict format and with one of its reason codes, that it is the
machine. That way no new reason codes reach the store, and the blame rule
below still applies: only when the launch succeeded with the same checks.

A machine is recorded as failed only if the launch succeeded (all `--count`
boxes handed over); otherwise the cause may be ours. `observations.jsonl` then
blocks it for 2 hours (GPU busy), 1 day (unreachable, no network, died,
stalled) or 14 days (wrong hardware, CUDA or runtime errors). A later
healthy run clears the first two, and they are let back in when too few
hosts are left. The 2 hours come from measurement; the other two are guesses.

## 9. Reputation store

Optional. With `GPUBOX_STORE_URL` and `GPUBOX_STORE_TOKEN` set, observations
are shared through `store/`: a Cloudflare Worker in front of a D1 database.
Clients send observations (machine id, kind, reason code); the store records
who sent each one, from the token, and when, from its own clock. The
blocklist policy stays in the client.

- It is advisory. A command waits at most 5 s in total for it, and nothing
  about it can fail the command. It can only make an offer be skipped.
- `up` and `search` read the last 30 days in one request. If that fails, the
  previous read is used.
- Rows to send wait in `observations.jsonl.outbox` until the store has them.
  A resend is harmless: a row's id is unique.
- Rows are judged per source. A machine is blocked if any source's rows block
  it, and a `healthy` row only takes back failures of its own source.
- The store limits each source to 500 rows a day and one failure per machine
  per hour.

## 10. Measured

On Vast, November 2026. Small samples, one day.

- **Loading a 1.3 GB image** (50 hosts, time from create to running): median
  69 s, 90th percentile 226 s; 46 of 50 running within 300 s. Outside
  mainland China, location made no visible difference. Advertised bandwidth
  only mattered below about 500 Mbps.
- **Mainland China**: median 250 s, and 2 of 6 hosts never finished. Hence
  the default `--exclude-country CN`.
- **Racing** (resampling those 50 hosts): the first of two is running after
  72 s on average (90th percentile 107 s), against 130 s (278 s) for one.
- **A 4 GB image** (6 hosts): 80 to 350 s where the pull finished; two
  stalled for ten minutes or more. Pulls that finished were silent for at
  most 166 s, which is where the stall rule comes from.
- **Gate failures**: in an earlier sample 4 of 14 rentals had a busy GPU
  before we touched it; two of those machines were fine 25 and 90 minutes
  later.
- **The listing** answers 429 at about one call every 3 s, which is why the
  loop polls by id.

Exercised end to end on real boxes: `up`, `run`, `renew`, `down`; the in-box
timer destroying a box 19 s after its lease ended; a container restart
(expiry, timer, address and host key kept); the reaper; a race with a loser;
an interrupt mid-launch (exit after 7 s, nothing left); a killed controller
cleaned up with `down --run-id` from an empty directory; a wrong image tag;
and a full test suite on 2x RTX 5090.

Not yet seen on a real box: a create that got no answer, a delete that did
not take, and a provider subaccount.

Also on real boxes (RTX 3060s, November 2026): an image whose ENTRYPOINT is
`ffmpeg` started our sshd, so `onstart` does replace the entrypoint; a
python:3.12-slim image passed the gate with its CUDA checks skipped; two boxes
handed over together with the spare destroyed; the image's environment in SSH
sessions; a job on each box with its index; a job followed across killed
connections with every byte once and its exit code kept; Ctrl-C leaving the
job running; the box ending a job at its timeout with nothing left of it; and
`--destroy-after` (the instance gone 26 s after the job ended).

## 11. Considered, and decided against

- **A local journal as the authority on what a run owns.** It needs replay,
  damage handling and a manual step for creates with no answer, and it is
  gone exactly when it matters most, with the CI runner's disk.
- **A dedicated account whose reaper destroys anything untagged.** It is the
  defence against a box that renames itself (A1). It needs the account to be
  used for nothing else, and rules to avoid killing boxes that are seconds
  old.
- **Listing the account on every pass of the loop.** It allows adopting the
  box of a create that got no answer mid-race, but the listing is rate
  limited; one listing when `up` ends finds that box as well.
- **Destroying the winner when a loser's delete is unconfirmed.** A cleaner
  postcondition, at the price of a failed launch and no leak removed.
- **Giving the slot of an unanswered create back after a while.** Repeated
  timeouts could then stack up unseen boxes.
- **Waiting, after one host's image error, for the others to prove it
  wrong.** The two-host rule is faster and needs no timer.
- **The provider's SDK.** It retries create requests and rewrites search and
  create payloads; the client here is one standard-library module.
- **A provider interface.** There is one provider. What the launcher needs
  from it is seven calls (`offers`, `create`, `instances`, `instance`,
  `destroy`, `relabel`, `account`), which is where a second provider would
  start.
