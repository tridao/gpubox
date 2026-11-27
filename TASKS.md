# Tasks: what a task fleet needs from gpubox

A task fleet for simulation workloads launches many single-GPU Vast boxes that run its
image's entrypoint unattended, from a private image. These tasks
close the gaps found when comparing the two. Work them in order; each is one or more commits.
`[R]` marks the ones reviewed by a second model before moving on.

## 1. Gate: CUDA checks only where PyTorch has CUDA; a custom gate hook

- [x] Torch not installed, or a CPU-only build (`torch.version.cuda is None`): skip the matmul
  and VRAM checks and say so in the report. A CUDA build that cannot initialise CUDA is
  still `cuda_init` (that is the machine). `nvidia-smi` checks always run. `GATE_VERSION` 2.
- [x] `--gate-script FILE`: run on the box after the built-in gate passes, same SSH channel,
  `--gate-timeout` (default 600 s). Exit 0 passes. To blame the machine, the script prints a
  last line `{"ok": false, "reason": "<code>", "detail": "..."}` with an existing reason code
  (`no_network`, `runtime_error`, ...); anything else fails the box without blame. No new
  reason codes, so the reputation store is unchanged.

## 2. Private registries

- [x] `--registry-user USER`, with the token read from `GPUBOX_REGISTRY_TOKEN` (never on the
  command line). Sent as Vast's `image_login`; the registry is taken from the image name.
  Use a read-only pull token: Vast and the host's docker daemon see it.
- [x] Omit provider response bodies from Vast errors; track private pull progress without logging registry status text.

## 3. N boxes in one run [R] (reviewed; checked on real boxes)

- [x] `up --count N`: one run id, one race with `N + --spare` slots, all-or-nothing (a launch
  that cannot get N healthy boxes rolls back like today's). `--max-attempts` defaults to
  `2N + 4` (6 for one box, as now).
- [x] State file v3: `winners` list (v2 files with one `winner` still load).
- [x] Names: one box is `NAME`, as now; N boxes are `NAME-0` ... `NAME-(N-1)`. `--ssh-config`
  writes one `Host` entry each. `gpubox status` prints the boxes (alias, address, lease).
  `run` takes `--box I` (`ssh`, `logs` will too); `renew` and `down` act on all of them.

## 4. Detached jobs; ssh in [R] (reviewed; checked on real boxes)

- [x] The container's environment (Docker `ENV`, Vast's variables) is saved by the bootstrap
  and loaded by every gpubox command on the box and by login shells, so `ssh` sees the
  image's `PATH`.
- [x] `run` starts the script detached on the box (`setsid`, output to a log, exit code to a
  file) and then waits by default, streaming the log. A dropped connection or a killed
  `gpubox run` does not stop the job: `run` reconnects and resumes the log from where it
  was; another process can `gpubox logs -f JOB` / `gpubox wait JOB`. `--detach` returns
  the job id at once. `--timeout-min` kills the job's process group on the box (exit 124).
- [x] `gpubox ssh [--box I] [-- CMD...]`: an interactive shell or a command, with the run's key,
  no `~/.ssh/config` needed.

## 5. A job ends its own box

Partly done: the lease timer destroys a box when its lease ends, and writing a past time to
`/etc/gpubox-expiry` would make it do so within 30 s, but nothing documents or wraps that.
- [x] Bootstrap installs `gpubox-destroy-self` (writes 0 to the expiry file; the existing timer
  does the deletion, so there is one code path). `run --destroy-after` calls it when the
  script ends, whatever its exit code.

## 6. GPU models, CPUs, RAM

- [x] `--gpu` repeatable: any of the models, cheapest first; the gate accepts any of them.
  Observations record the model actually rented.
- [x] `--min-cpus N`, `--min-ram-gb N` for the whole box (defaults as now: max(16, 8 per GPU)
  CPUs, 32 GB per GPU).

## 7. Python API [R] (reviewed; used live for task 10)

- [x] `gpubox.api`: `up(spec, ...) -> Run` (context manager: `down` on exit unless told to keep),
  `Run.boxes -> [Box]`, `Box.ssh_argv(cmd)`, `Box.start(script, env) -> Job`, `Job.wait()`,
  `Job.logs()`, `Run.renew(ttl)`, `Run.down()`, `down(run_id)`, `reap(namespace)`.
  Exceptions instead of exit codes; the CLI is a thin layer over it.

## 8. Cost accounting

Implemented: durable attempt receipts with quoted rate, identity and outcome.
Account budgets and provider invoices remain outside the library.

Not now.

## 9. Secrets

Practice now: secrets live in a manager (1Password, Vault, Doppler, a cloud secret manager,
GitHub Actions secrets) and are injected at the last moment into the process that needs them,
through a channel that does not persist: an environment variable or stdin, never a command
line (`ps` shows it) or a file that outlives the job. Prefer short-lived, narrowly scoped
credentials (OIDC federation, scoped API tokens) over long-lived keys. On a marketplace the
host operator has root on the machine, so anything sent to a box must be something you can
afford to have read: scoped, revocable, ideally short-lived.

gpubox should not talk to any one manager. It should carry a secret from the controller's
environment to the job without storing it anywhere else, so `op run -- gpubox run ...`,
GitHub secrets, Doppler etc. all work the same way:
- [x] `--env NAME` (no `=`) takes the value from gpubox's own environment, so it is not on a
  command line or in shell history. `--env-file FILE` reads `NAME=VALUE` lines.
- [x] Values travel over SSH stdin into a 0600 file in the job's directory, read by the job's
  shell and deleted as it starts; never into Vast's instance `env`, label or arguments.

## 10. Interruptible instances [R] (done, checked on a real box; see DESIGN 5b)

New problems they bring, and the answer to each:
- An outbid box is stopped, not destroyed, and Vast keeps billing its disk; its container is
  not running, so the lease timer cannot fire. -> The reaper and `down` already destroy by
  tag whatever the state; document that a scheduled `reap` is required with `--interruptible`.
- Preemption is not the machine's fault. -> A box that exits while interruptible is failed
  without blame during the launch, and no `died_after_admission` is recorded during `run`.
- A job dies with its container (Vast SIGKILLs about 0.1 s after SIGTERM), and when the bid
  wins again the container restarts without it. -> `run`/`wait` see the job gone without an
  exit file and exit 75 saying the box was preempted.
- A resumed box may come back on another port. -> `run`/`ssh` re-read the endpoint from the
  provider when a connection fails, and update the state file.
- Two bids from one account on one machine outbid each other. -> Skip machines the account
  already has an instance on.
- Price: `--interruptible` searches bid offers and bids `--bid-markup` (default 1.25) times
  the offer's minimum bid, capped at `--max-dph-per-gpu`.

## 11. Slurm clusters

A task fleet could also run on a cluster. First a general provider, then what the fleet needs.
- [x] `--provider slurm --login HOST`: a box is one batch job holding an allocation, reached
  with `srun --jobid --overlap` through the login host; the batch script keeps the lease and
  starts detached jobs; the gate looks only at the allocation's GPUs (DESIGN.md, section 13).
  Checked live on an H100 Slurm cluster.
- [ ] The image. A fleet runs a private image's entrypoint unattended. On a cluster the box
  would run it in a container: pyxis/enroot (`srun --container-image`), Apptainer, or a
  per-job rootless Docker, whichever the cluster has; `--registry-user` and
  `GPUBOX_REGISTRY_TOKEN` as on Vast, never in the job's script or environment.
- [ ] Preemptible partitions: a requeued job comes back, possibly on another node, without
  its jobs. Read it as task 10 reads an outbid box (exit 75, the box was preempted).
- [ ] Machine records for clusters: a node that keeps failing the gate, remembered across
  runs per cluster (not in the shared store, whose machine ids are Vast's).

## 12. More providers

Providers to add, in this order.
- [x] The groundwork: each provider declares what it can do (`providers/`); keys from the
  environment or that file; ids that are tokens.
- [x] RunPod (REST API v2): pods as boxes, the same bootstrap as Vast, self-deletion with the
  pod's own key. Checked live (DESIGN.md, section 14).
- [x] Include both Secure and Community RunPod offers by default, with a direct SSH gate
  for both tiers. Keep single-tier restrictions for callers that need them.
- [ ] RunPod: private images (a registry credential per run), interruptible pods (GraphQL),
  the machine id (GraphQL) for the blocklist.
- [x] Shadeform: an aggregator of VM clouds (Hyperstack, Massed Compute, ...), with a lease it
  enforces itself (`auto_delete.date_threshold`) and tags. The box is the image as a container
  on the VM, reached with SSH and `docker exec` (providers/vmbox.py). Checked live (DESIGN.md, 15).
- [ ] Shadeform: spot instances, private images (a token the startup script does not keep),
  `--ssh-config` (a ProxyCommand into the container).
- [x] Hyperstack, on the VM box; the lease is `down` and the reaper (it keeps no expiry). Not
  checked live yet.
- [x] Massed Compute (MCP), on the VM box. Checked live. No rename, so no `renew`.
- [x] Lyceum, on the VM box; the first probe sends the setup (it runs nothing at boot).
  Checked live.
- [x] Lium: a marketplace of nodes, pods as boxes with Lium's sshd, the rent retried under
  its idempotency key, the lease Lium's removal time. Checked live (DESIGN.md, 17).
- [ ] Lium: detect a key whose pod visibility is `own` (its listing is not the account's)
  once the API says which a key is; until then it is documented.
- [x] Lium: spot nodes with `--interruptible`, beside the others; a closed pod (BROKEN)
  is interrupted, not the machine's fault.
- [ ] Lium: private images (a Docker credential per run),
  node ids for the blocklist once they are seen to mean the same machine across runs.
- [x] Verda: VMs on the VM box, spot VMs with `--interruptible`, an OAuth client for a
  key, the tag as a tag. Checked live, spot and on demand (DESIGN.md, 18).
- [ ] Radiant (Ori): decided to keep a table of Ori's list prices (Oct 2026, per GPU-hour:
  H200 3.50, H100 SXM/PCIe 2.90, A100 2.74, L40S 1.55, V100S 0.95, L4 0.93, V100 0.83), since its
  API has none. Blocked: every create answers HTTP 500 with no message, so its VM fields,
  statuses and login user are unknown. On
  one shared
  base. A VM is reached as `ubuntu`
  or `root` with an SSH key registered through the API; the box runs the image with docker
  or runs in the VM; the lease is the provider's own expiry where it has one, else `down` and
  a scheduled `reap` (no account key on a box).

## 13. First PyPI release

- [x] Build and check the wheel and source distribution in CI; publish from a
  matching tag with GitHub OIDC (DESIGN.md, section 16).
- [x] Register the Trusted Publishers, rehearse `0.1.0` on TestPyPI, and publish
  the same tag to PyPI.

## 14. Vast offer lifetime

- [x] Filter Vast offers by the requested lease, and skip any whose end date
  is missing or too early. Let `search --ttl-min` preview the same filter.
