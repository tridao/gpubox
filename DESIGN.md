# gpubox design

gpubox rents a GPU box on Vast.ai for a CI job or for development, checks
that it works, and makes sure it is destroyed afterwards. It can also rent
from RunPod or Shadeform, or take boxes from a Slurm cluster (sections 12
to 15). This
document says how, what it assumes, and what was decided against. Where it
says Vast, read "the provider" unless those sections say otherwise.

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

1. Search offers. Drop excluded countries, offers above the price cap,
   offers that cannot cover the lease and recently failed machines; take
   the cheapest first, one machine per host.
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
  cheapest-first order are applied again locally. Vast's search requests
  `duration >= --ttl-min` (converted to seconds), then checks each offer's
  absolute `end_date` against the requested lifetime as of the search start.
  Offers with no usable end date are skipped.
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

Vast can stop a box at its contract end date, independently of our lease.
`renew` does not extend the Vast contract.

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
that box, blamed on nobody. When boxes in two different failure domains
have shown it, the launch stops as a configuration error (exit 2): with a
wrong tag that is about 45 s in, and one misconfigured host is simply
replaced. A failure domain is what an offer's box shares with others' so
that they may fail together (a broken mirror, stale credentials): a Vast
host, a VM cloud's region, a Lium node's provider. Boxes whose offer names
no domain do not count as different ones. Once any box
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
offer's minimum bid, rounded up and never below it; the price cap and the
cheapest-first order apply to the bid plus the disk, whose price is what the
offer's total says beyond its minimum bid (an offer that cannot be priced so
is not a candidate). `search` shows the same candidates. What changes:

- An outbid box is stopped, not destroyed, and bills its disk; with its
  container stopped, its lease timer cannot fire. `down` and the reaper
  destroy it by tag whatever its state, so a scheduled
  `reap --namespace NS --destroy` is the bound, and a teardown that could
  not finish says so.
- Being outbid is the market, not the machine: a box that exits, stalls, or
  never reaches `running` within the start-up timeout fails without blame; so
  does one that loses its connection while being admitted unless the provider
  then says it still runs; and a stopped box is never recorded as `died`.
- A healthy box waiting for the others may stop and resume elsewhere; the
  last look before the hand-off sees its new address and admits it again.
- The job dies with its container (Vast kills it 0.1 s after SIGTERM). When
  the bid wins again the container restarts without it; `wait` sees the
  wrapper gone and reports the job `lost`.
- A resumed box may answer on another port. When SSH fails, the provider is
  asked about the box; a running box's address is taken from the answer and
  written to the state file (unless a teardown holds the directory).
- Two of our own bids on one machine outbid each other, so machines the
  account already has an instance on are skipped (a listing row that does
  not say its machine fails the launch), and no launch races two offers of
  one machine. Two launches in different state directories that start at the
  same moment can still bid on one machine: the account's boxes are looked
  at once, when a launch starts.
- A start cut short by a stop leaves the job's variables on the box's disk:
  the bootstrap deletes any it finds when the container starts again, and a
  job directory that never got a wrapper is listed `lost` after a minute.

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

Vast responses can echo registry credentials. Errors describe the operation
and status without copying the response body. For authenticated pulls, the
launcher tracks changes to a digest of the provider's status message instead
of retaining the message in an attempt or a failure report.

The run state includes one receipt row per attempted create. The unknown row
is written before sending the request. A reply gives it an ID or a refusal;
a later listing can supply a missing ID by tag. Closing the run records known
rentals as gone, but cannot turn an unknown into a refusal. The file and its
parent directory are synced before a write returns. Tag-only cleanup cannot
reconstruct a receipt from an empty listing.

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
A version 2 file (one `winner`) is read as a run of one box. A Slurm run
says so in `meta` (`provider`, and the `slurm` settings later commands
need); its boxes' `host` is the login host, `port` the job id and
`machine` the node.

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
`--destroy-after` (the instance gone 26 s after the job ended). Through the
Python API: an interruptible box rented at 1.25 times the minimum bid; a job
killed through its wrapper (143); a job with a child that ignores SIGTERM
ended at its timeout (124), the child gone; and the box stopped through the
API, as when outbid (`actual_status` exited, `intended_status` stopped): the
wait ended as a lost box with nothing recorded as died, and once the box ran
again its job was reported lost and the saved environment was there.

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
- **An sshd of our own inside a Slurm job.** It would give a dev box on a
  cluster plain `ssh NAME`, scp and editors, as on Vast. But it needs sshd
  on the nodes, nodes reachable from wherever gpubox runs, and the job's
  environment (`CUDA_VISIBLE_DEVICES` above all) put back into every
  session; `srun --overlap` through the login host needs none of that, and
  is how a cluster's users already reach their jobs.

## 12. Providers

The launcher, `down`, `renew` and the reaper talk to a provider through a
small interface (`providers/base.py`): candidates to rent; a create that
carries the box's lease and is sent once, or sent again only where the API
collapses repeats, and whose outcome is a yes, a no, or unknown; an admission
once the box runs (what must be done before it is handed over: on Lium, its
lease set again); lookups and a listing read strictly, a delete, a relabel
that is also how a lease moves, and a way to read a row (`View`: queued,
loading, running or exited; what the provider says; where SSH reaches it). What differs between
providers and is not a call is declared by each, as class attributes, and
read through one table (`providers/__init__.py`): its API key's variable; how a box
is reached (our sshd with the run's key, or a Slurm step); how a job is
started; whether its machine ids mean the same machine from run to run, so
that observations, the blocklist and the store apply; its default spare;
how late its own lease may end a box that was never admitted; whether its
offers depend on the lease (`search --ttl-min`); and which parts of a Spec it can honour (an image, a price cap,
interruptible boxes, a registry login, countries to skip, an SSH
configuration entry, any GPU model). `up` and the command line refuse what a
provider cannot do, and insist on what it needs, before anything is rented.

The code follows that line. Everything that knows a provider is in
`providers/`: one module per provider, the interface (`base.py`), the
registry (`__init__.py`), the HTTP client they share (`httpapi.py`), and the
two kinds of box: a container the provider runs with our command as its
entrypoint (`container.py`; Vast, RunPod) and a VM that runs the image as a
container (`vmbox.py`; Shadeform, Hyperstack, Massed, Lyceum). Lium's pods
are a third kind: containers whose sshd is Lium's, which run the command a
VM box's container runs (save the environment, then wait) and are reached
as Vast's are. The rest
(`launch.py`, `state.py`, `gate.py`, `job.py`, `api.py`, `cli.py`) knows
none of them by name except to pick one, and `onbox.py` holds what every box
has whatever its provider: the expiry file, the saved environment, the
command of a container that only waits, the self-destroy helper. Adding a provider is a module in `providers/` and a
line in the registry.

A run's state file names its provider and that provider's options (never a
secret), so the commands that act on a run need neither. A box's id is a
number or a short token (RunPod's pods).

What the others were seen to do (dstack, gpuhunt, SkyPilot, November 2026):
offers come from a catalog of typed items, the lifecycle is a few calls per
provider, and optional abilities are mixins or flags. gpubox keeps that
shape. Where it differs is in what it promises about a create with no
answer and a listing that cannot be read, which none of them treat as more
than an error.

## 13. A Slurm cluster

On a cluster a box is one batch job holding an allocation: `sbatch` with
the tag as the job name, `squeue` for a lookup (`--jobs ID --states=all`)
and the listing (`--me`), `scancel`, and `scontrol update` for a relabel.
gpubox runs these on the login host through the user's own SSH
configuration. Its output goes to files, not pipes: an `ssh` that starts a
ControlMaster leaves a process behind that holds its stderr.

**What it reads.** A job is gone when Slurm says it is over (a final state,
or COMPLETING: told to end, and it cannot run again), or no longer knows
it: "Invalid job id specified" is what the controller says of a job that
ended more than MinJobAge ago. Any other failure, and an empty answer, is
an error, never evidence. `sbatch` is sent once. An explicit refusal
("Batch job submission failed: Invalid partition name specified") is our
request's fault on a cluster, so it stops the launch as a configuration
error; so does a pending job whose reason says it can never run
(PartitionTimeLimit, BadConstraints, ...). A job id is taken only from a
clean exit that prints one. Anything else, including sbatch's own trouble
reaching the controller, may have submitted a job: the attempt keeps its
slot and the job is found by its tag, as on Vast.

**The queue.** A pending job is nobody's fault. The start-up timeout counts
from when the job starts running, not from the submission, and the stall
rule does not apply; only `--overall-timeout` bounds the wait. There are no
offers: Slurm chooses the node, so each attempt is the same request, and a
node that failed the gate through its own fault is excluded (`--exclude`)
from the requests after it. `--spare` defaults to 0: a spare would hold a
share of the cluster that others could use. Observations, the blocklist
and the store are not used: node names are not Vast machine ids, and a
cluster's nodes are its operators' business.

**The box.** The batch script holds the allocation until the lease ends,
then exits, which ends the job and everything in it. As the Vast timer
does, it reads the end of the lease from a file (on the node, in
`/tmp/gpubox-<job id>`), created from the submission unless it is there,
with what is not a time read as the original expiry. Slurm's time limit is
set to the lease plus two minutes, as a backstop. `renew` rewrites the
file, then sets the job's name and time limit in one `scontrol update`; a
cluster that does not let users extend their time limit refuses that, and
the box ends at its old limit: early, not late. `gpubox-destroy-self`
writes a past time to the file. On SIGTERM (a cancel) the script removes
the directory, so the node is left clean.

**Getting in.** Every command on a box is a step of its job:
`ssh LOGIN srun --jobid=ID --overlap --nodes=1 --ntasks=1 bash -c ...`.
Slurm gives the step the job's GPUs, CPUs and environment, which is why
there is no sshd and no key of the run's own. A step's stdin is forwarded,
so scripts and secrets travel as they do on Vast. The gate is told it runs
in a share of a machine: it looks only at the GPUs in
`CUDA_VISIBLE_DEVICES` (on a cluster that does not confine devices,
`nvidia-smi` lists them all, other jobs' too) and does not judge the
node's disk.

**Jobs.** Slurm kills whatever a step started when the step ends, `setsid`
or not. So a job is started by the batch script, which lives as long as
the box: a step writes the job's directory and leaves a `start` file; the
script renames it (so a job starts once) and starts the job's wrapper.
The step waits until the file is taken, and after 30 s takes it back and
fails. A job then sees the batch step's environment, the job's own.

**Measured** on an H100 Slurm cluster (Slurm 25.11), November 2026: a box
is healthy 21 to 25 s after `up` starts when a GPU is free; a cancel
mid-queue rolls back in 2 s; a job killed controller-side carries on and
is followed to its exit code from another process; `--destroy-after` and a
renewal to one minute end the job by its own script, on time.

## 14. RunPod

A pod is a container, as on Vast, so a box is the same: our start-up
command is the pod's entrypoint, with the lease timer first and then an
sshd that takes only the run's key on the pod's published 22/tcp. RunPod
puts `runpodctl` and a key confined to the pod in every pod (seen on a
plain ubuntu image), so the timer deletes the pod by itself
(`runpodctl remove pod $RUNPOD_POD_ID`), as Vast's does with its container
key.

There are no machines to choose. The catalog lists GPU types with a list
price and stock; an offer is a GPU type in a cloud tier and, when reported,
a data center. `up` and `search` query Secure and Community separately by
default, since stock is tier-specific. The combined pool is sorted by price,
with Secure first at equal prices. Create uses the offer's exact tier so
it cannot rent a different tier at a different price. An offer is repeated
once per slot of the race because it can hold several boxes. A data center
that turns out full answers 400 ("no longer any instances available"): the
explicit no of a stale offer, which costs no attempt. A MIG slice is offered
only to whoever names it.

A pod says RUNNING before its container is up; it is running for gpubox
once it has an SSH address. The v2 API names no machine and gives no
status message, so machines are not recorded, and an image that cannot be
pulled is given up on by the stall and start-up rules rather than
recognised. A deleted pod is 404, also to a second delete. A rename does
not restart the container (a job ran on through one).

**Measured**, November 2026, secure cloud, `python:3.12-slim`: healthy 45
to 73 s after the create, once 233 s (a slow pull); a pod deleted itself
within 30 s of `--destroy-after` and 85 s after a renewal to one minute; an
interrupt mid-launch rolled back in 2.5 s; `down --run-id` from an empty
directory found the run's pods by name.

**Measured on Community Cloud**, November 2026, `python:3.12-slim`: four
single-GPU launches (RTX 3070 and RTX 3080 Ti) were healthy in 27–46 s. The
catalog gave overall stock but no data centers; placement selected by GPU
type and cloud worked. Each pod had a public `22/tcp` mapping, and SCP over
SFTP, rsync (once installed), and a detached job worked. Renewing one pod
to a one-minute lease deleted it within about 75 s. The default container
disk did not provide a `/workspace` mount. These samples do not establish
long-term uptime or guarantee a public IP. REST v2 does not offer a public-IP
placement filter; the direct SSH gate is required even when the pod reports
`RUNNING`.

Not yet: interruptible pods (v2 does not create them; its GraphQL API
does), private images (a registry credential per run, deleted with it), and
countries to skip (v2 places by data center).

## 15. Shadeform, and boxes on VMs

Shadeform rents VMs on many clouds through one API. A VM is not a
container: it boots an OS of the cloud's, with no image of ours. A box on
a VM (`providers/vmbox.py`) is still the caller's image: the VM's startup script,
run as root once the VM is up, adds the run's key for every login user and
starts the image as a container named `gpubox`, with the VM's GPUs,
network and IPC, under a command of ours that saves its environment and
waits. Everything gpubox runs on the box (the probe, the gate, jobs, `ssh`)
is SSH to the VM as the user the cloud names, with the run's key, then
`sudo docker exec` into that container. No port is opened and the image
needs no sshd; jobs, started with `setsid` in the container, outlive the
exec that started them. While the container is not up the probe fails,
and if the pull failed it says what docker said, which the launcher reads
as it reads a Vast host's status message: two VMs in different regions
saying the image cannot be pulled make it a configuration error.

An offer is an instance type in a region that has it, booted with
Shadeform's own OS image (it has docker and NVIDIA's container toolkit)
with the newest CUDA in its name, which must be at least `--min-cuda`; that
CUDA is the offer's driver version for the launcher's driver-too-old rule.
Shadeform says how long a type takes to boot (5 to 8 minutes, against a
start-up limit of 5): the launcher adds that to the start-up and stall
limits of the attempt, which are about what comes after.

The lease is the cloud's. A VM could delete itself only with the account's
key, which never goes on a box; Shadeform instead deletes a VM at its
`auto_delete` time, set at the create and moved by `renew`, with the tag,
in one update that is believed once read back. So there is no timer in the
box, and `run --destroy-after` is refused. The tag is in `tags`: a name is
at most 50 letters, digits, `_` and `-`. A deleted VM is `deleting` and
then `deleted`, or 404. `deleting` counts as gone, as Slurm's COMPLETING
does: Shadeform no longer bills it, and it cannot come back; waiting for
`deleted` would only make `down` slower. Shadeform names
no machine and has no account lookup.

**Measured**, November 2026: an A6000 on Massed Compute through Shadeform
was healthy 204 s after the create (`python:3.12-slim`); jobs, `ssh` and
`renew` worked as on Vast; a renewal to one minute was deleted by
Shadeform about 50 s after the lease ended.

Hyperstack, rented directly, uses the same box: its cloud-init user data is
the startup script, its image "with Docker" of the newest CUDA in the region
is booted, port 22 is opened without port randomization, and the tag is a
label. It keeps no expiry, so its lease is `down` and the reaper, and `up`
says so. It requires a key pair at the create, though the run's key goes in
with the script: a placeholder is imported, once per environment, whose
private half was deleted when it was made, so it lets nobody in (an earlier
run's key would let that run's holder into later VMs). Its prices are the
GPU rate of its pricebook; its disk and public IP are not in them, so the
price cap does not cover those. A VM that is DELETING still counts, as does
a Massed instance that is `terminating`: neither is known to have stopped
billing.

On every VM cloud, gpubox gives advice with the provider named (`down
--provider hyperstack ...`), and where nothing but the reaper ends a box it
says so instead of promising that the box will end itself. Calls with a
credential never follow a redirect. Known gap: where the first probe starts
the setup (Lyceum), a VM that reboots during it is not set up again; the
start-up limit then fails it and another is rented.

Massed Compute uses it too, through its MCP server: the startup script is
the instance's boot command, and the tag its name. It cannot rename an
instance, so a lease cannot be moved: providers say whether they can
(`renewable`), and `renew` is refused where not. Its login user is the
instance's own (`Ubuntu`, capital U, on its images); a VM's user is never
assumed. Measured, November 2026: an A6000 healthy 127 s after the launch.

Lyceum uses it too, though it runs nothing of ours at boot: there the first
probe that gets in sends the startup script on its stdin and starts it in a
session of its own, once (a marker file on the VM says it was). The script
is read before anything is sent to the background, which reads its stdin
from /dev/null (a live run lost its script so). Whatever the cloud, the
script writes what it does to `/var/lib/gpubox/boot.log`, whose end the
probe shows while the box is not up. Lyceum's login is `lyceum` (its docs;
root refuses the key), and its deletes answer slowly, then 500 while they
run: `down` confirms them by lookup as everywhere. Measured, November 2026:
an L40S healthy 138 s after the create.

Radiant (Ori) is not done: its API gives no prices, so the price cap could
not be kept, and its OS images have neither docker nor NVIDIA's toolkit.

The other VM clouds are to use the same box; what differs is their API, how their key gets on the VM
(an API key pair, or the startup script), and whether they have an expiry
of their own. Without one, `down` and a scheduled `reap` end their VMs.

## 16. Package releases

Hatchling builds the source distribution from the Python package, project
metadata, README and license. The wheel is built from that source
distribution, not directly from the checkout. CI and publishing use the
same checks: validate the metadata, install each artifact separately, and
check the CLI and API outside the checkout. The publishing job takes only
the checked artifacts and gets a short-lived credential through GitHub OIDC.

The one-time setup is a [pending Trusted Publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
for `gpubox` on each index. Use the GitHub repository owning this file,
workflow `publish.yml`, and environment `testpypi` on TestPyPI or `pypi` on
PyPI. Both GitHub environments allow only `v*` tags, and production requires
maintainer approval. Neither environment needs a stored API token.

For each release, set the version in `pyproject.toml`, wait for CI on `main`
and push the matching `vVERSION` tag. Dispatch **Publish gpubox** on that
tag, choosing TestPyPI first. After it succeeds, install from TestPyPI into
a fresh environment:

```bash
gh workflow run publish.yml --ref vVERSION -f index=testpypi
# After the run completes:
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps gpubox==VERSION
gpubox --help
python -c 'import gpubox.api'
```

Once the rehearsal passes, run `gh workflow run publish.yml --ref vVERSION -f index=pypi`
and approve the deployment. A version cannot be overwritten on either
index; use a new version if the artifacts need changing.

## 17. Lium

Lium is a marketplace of independent hosts, as Vast is: a node is one
machine's GPUs, listed at its provider's price per GPU (`/executors`, which
needs no key), and a pod is a container on a node. Its CPUs, memory (less
what the node keeps back, `pod_reserve`) and disk are shared out in
proportion to the GPUs rented, which is what `--min-cpus`, `--min-ram-gb`
and `--disk-gb` are judged against; seen live, a pod got exactly that
memory and the node's listed CPUs. A node listed with no minimum GPU count
rents only whole. Spot nodes, which can be taken back, are left out. A
node is offered once per box it can hold (its free GPUs, and its free
ports: each pod takes one), every node once before any again: the spares
of a race hedge against a bad machine, which two boxes on one node would
not. Nodes with frequent rent failures, a bad disk, a removal or
maintenance that starts within the lease, a reliability under
`min_reliability` or none yet, a download under the minimum, or no country
when countries are excluded are left out: a size or a time that is not
stated does not meet a minimum. The rent states the price offered, and
Lium refuses it only if the node now asks more than a cent above that; so
offers leave that cent within the cap. A node's failure domain is the
provider who runs it: two of their nodes are one witness against an image.

**The box.** A rent names a template: an image, an entrypoint and a start
command. Each attempt makes a private one-time template, which Lium
deletes with the pod (and gpubox deletes if the rent is refused). Lium
takes only one executable as the entrypoint and a start command only in
the form `bash -c '...'`, so the entrypoint is `/usr/bin/env` (which keeps
the image's own ENTRYPOINT from running our command as its arguments) and
the command is `bash -c` with the VM box's container command: save the
environment, then wait. Lium's validator then puts an sshd in the
container (installing one if the image has none) and adds the rent's keys
to root's: the run's key, and with `skip_add_agent_ssh_key` not Lium's
agent key. It adopts an sshd the image starts itself, so ours would race
it; gpubox starts none. Keys the image itself carries are not removed. The
pod shows `ssh root@HOST -p PORT` once running; anything else (a gateway's
`-J`, another user) is not how gpubox gets in, and the box is given up on.
The volume is `/root`, as Lium advises; the image's files in `/root` were
seen in it.

**The rent.** The rent carries an `Idempotency-Key`, the tag: within a day,
Lium answers the same key and body with the first answer (seen live,
`Idempotent-Replayed: true`). So unlike every other create, it is sent
again, the same body, while there is no answer (no reply, a 5xx, a
redirect) or the first is still running (`idempotency_in_progress`,
`own_rent_in_progress`), and after a 429. Once an answer was lost, only a
pod id settles the rent: a refusal that follows may be Lium's answer to the
repeat, not to the request that rented, so the attempt stays unanswered
and keeps its slot. Errors carry a stable `error.code`, which is what is
read: one about the account or the image stops the launch, since no other
node would answer otherwise. A pod id is taken only from a successful
answer, and not if a lookup shows that pod under another name.

**The lease** is Lium's: a pod's `removal_scheduled_at`. The rent carries
`termination_hours`, the lease rounded up to whole hours, as its backstop:
Lium counts it from when the pod starts running, and at that moment it
replaces any removal time set before (seen live). So the exact time is set
twice: right after the rent, which covers a pod that never starts, and
again once the pod runs, in its admission (`admit`, which the launcher
calls as soon as the pod shows an SSH address, before it probes it). A pod
whose time cannot be confirmed then is given up on and deleted, as a box
that fails its gate is. A pod left behind by a launch that died ends at the
backstop: an hour at most after the lease, plus its start (`lease_slack`,
which the advice after a failed teardown states). `renew` sets the removal
time, then the name, and believes it once both read back. Lium shows times
in UTC without an offset. Lium takes `termination_hours` up to 720, so a
longer lease is refused.

**Gone.** A deleted pod is DELETING for some 40 s, then 404 with the code
`pod_not_found`; billing stops at the delete request, so DELETING counts as
gone. A 404 without that code (a gateway's, a route's) is an error, not
absence.

**Keys.** A key acts in its workspace, and the account gpubox pins is the
user and that workspace. A key whose pod visibility is `own` lists and
looks up only the pods it rented, and nothing in the API says which kind a
key is (`/users/me` names its scopes, not its visibility). So a listing
with such a key is not the account's, and tag-only cleanup with another
key could find nothing: this is documented, not detected.

**Measured**, November 2026, `python:3.12-slim` (sshd installed by Lium):
a P40 pod healthy 49 s after the create, an RTX 4090 46 s; jobs, `ssh` and
`renew` worked; a renewal to one minute was deleted by Lium within about
60 s of the lease's end; an interrupt mid-launch rolled back in 6 s;
`down --run-id` from an empty directory found the run's pod by its name;
nothing was left after any of them, and the one-time templates were gone
with their pods.
