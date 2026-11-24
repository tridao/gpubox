"""The Slurm provider: its commands and how their answers are read, the launcher on a
cluster, and the whole CLI against a cluster whose jobs are processes on this machine.

`Cluster` is a fake login host: it reads the argv of the commands gpubox runs there
(sbatch, squeue, scancel, scontrol, id) and answers as Slurm would. With `local=True`
a job's batch script really runs, here, as the box; `LocalProber` runs a job's steps
in a local bash with the job's SLURM_JOB_ID, as `srun --jobid --overlap` would.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fakes import FAST, FakeCheck

from gpubox import cli, gate, job
from gpubox import launch as L
from gpubox import provider as prov
from gpubox import slurm as sl
from gpubox.provider import ConfigError
from gpubox.state import State, make_tag, parse_tag

RUN_ID = "dev.box.0123456789ab"
EXPIRY = int(time.time()) + 3600
SPEC = L.Spec(image="", gpus=(), max_dph_per_gpu=0.0)
SHIM = str(Path(__file__).parent / "bin")


class Cluster:
    """A fake Slurm cluster, as its login host's commands see it.

    jobs         id -> {state, name, node, reason, argv, script, started, limit, proc}
    pending      lookups a new job answers PENDING to before it runs
    reason       the reason a pending job gives
    fail         command ("sbatch", "squeue", "scancel", "scontrol update", ...) -> answers
                 (exit code, stdout, stderr) to give first, instead of acting
    nodes        where jobs run: the first that the job does not exclude
    local        run batch scripts as local processes (the box) in `home`
    """

    def __init__(self, *, local=False, home=None):
        self.jobs, self.calls, self.fail = {}, [], {}
        # Job ids far from anything real: a local box keeps its files in /tmp/gpubox-<id>.
        self.next = 7_000_000 + (os.getpid() % 1000) * 1000
        self.pending, self.reason = 0, "Priority"
        self.nodes = ["node1", "node2", "node3"]
        self.local, self.home = local, home

    def __call__(self, argv, stdin, timeout):
        self.calls.append((list(argv), stdin))
        cmd = " ".join(argv[:2]) if argv[0] == "scontrol" else argv[0]
        if self.fail.get(cmd):
            return self.fail[cmd].pop(0)
        self.tick()
        return getattr(self, "do_" + argv[0])(argv, stdin)

    def names(self, *wanted):
        return [a[0] if a[0] != "scontrol" else " ".join(a[:2]) for a, _ in self.calls
                if not wanted or a[0] in wanted]  # fmt: skip

    def flags(self, argv) -> dict:
        out = {}
        for a in argv[1:]:
            k, _, v = a.partition("=")
            out[k] = v
        return out

    def tick(self):
        for jid, j in self.jobs.items():
            if j["state"] == "PENDING" and j["waits"] <= 0 and j["reason"] not in sl.NEVER:
                self.start(jid)
            if j["state"] == "RUNNING" and j["proc"] is not None and j["proc"].poll() is not None:
                j["state"] = "COMPLETED"

    def start(self, jid):
        j = self.jobs[jid]
        excluded = set(filter(None, j["flags"].get("--exclude", "").split(",")))
        j["node"] = next(n for n in self.nodes if n not in excluded)
        j["state"], j["reason"], j["started"] = "RUNNING", "None", time.time()
        if self.local:
            env = {**os.environ, "SLURM_JOB_ID": str(jid), "HOME": str(self.home),
                   "PATH": ":".join([SHIM, os.environ["PATH"]])}  # fmt: skip
            j["proc"] = subprocess.Popen(
                ["bash", "-s"], stdin=subprocess.PIPE, env=env, start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )  # fmt: skip
            j["proc"].stdin.write(j["script"].encode())
            j["proc"].stdin.close()

    def kill(self, jid):
        """What Slurm's cgroup does when a job ends: everything in it goes, the jobs that
        the batch script started in sessions of their own too."""
        j = self.jobs[jid]
        box = Path(f"/tmp/gpubox-{jid}")
        groups = [j["proc"].pid] if j.get("proc") else []
        for f in box.glob("jobs/*/pid"):
            try:
                groups.append(int(f.read_text().split()[0]))
            except (OSError, ValueError, IndexError):
                pass
        for f in box.glob("jobs/*/wpid"):
            try:
                groups.append(os.getpgid(int(f.read_text().split()[0])))
            except (OSError, ValueError, IndexError):
                pass
        for g in groups:
            try:
                os.killpg(g, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if j.get("proc"):
            j["proc"].wait()
        shutil.rmtree(box, ignore_errors=True)

    def close(self):
        if self.local:
            for jid in self.jobs:
                self.kill(jid)  # also removes its directory, whatever became of the job

    # ------------------------------------------------------------ commands

    def do_id(self, argv, stdin):
        return 0, "alice\n", ""

    def do_sbatch(self, argv, stdin):
        jid, self.next = self.next, self.next + 1
        f = self.flags(argv)
        self.jobs[jid] = {
            "state": "PENDING", "name": f["--job-name"], "node": None, "reason": self.reason,
            "argv": argv, "flags": f, "script": stdin, "started": None,
            "limit": int(f["--time"]), "proc": None, "waits": self.pending,
        }  # fmt: skip
        return 0, f"{jid}\n", ""

    def row(self, jid):
        j = self.jobs[jid]
        return f"{jid}|{j['state']}|{j['node'] or ''}|{j['reason']}|{j['name']}"

    def do_squeue(self, argv, stdin):
        f = self.flags(argv)
        if "--jobs" in f:
            jid = int(f["--jobs"])
            if jid not in self.jobs:
                return 1, "", "slurm_load_jobs error: Invalid job id specified\n"
            j = self.jobs[jid]
            if j["state"] == "PENDING":
                j["waits"] -= 1
            if f["--format"] == "%M":
                used = int(time.time() - j["started"]) if j["started"] else 0
                return 0, f"{used // 60}:{used % 60:02d}\n", ""
            return 0, self.row(jid) + "\n", ""
        active = [jid for jid, j in self.jobs.items() if j["state"] in ("PENDING", "RUNNING")]
        return 0, "".join(self.row(jid) + "\n" for jid in active), ""

    def do_scancel(self, argv, stdin):
        jid = int(argv[1])
        if jid not in self.jobs:
            return (
                1,
                "",
                f"scancel: error: Kill job error on job id {jid}: Invalid job id specified\n",
            )
        j = self.jobs[jid]
        if j["state"] in ("PENDING", "RUNNING"):
            if self.local and j["state"] == "RUNNING":
                self.kill(jid)
            j["state"] = "CANCELLED"
        return 0, "", ""

    def do_scontrol(self, argv, stdin):
        if argv[1:3] == ["show", "config"]:
            return 0, "Configuration data as of ...\nClusterName             = testcluster\n", ""
        f = self.flags(argv[1:])
        j = self.jobs[int(f["JobId"])]
        j["name"], j["limit"] = f["JobName"], int(f["TimeLimit"])
        return 0, "", ""


@pytest.fixture
def cluster():
    c = Cluster()
    yield c
    c.close()


def provider(cluster, **kw):
    return sl.Slurm("login.example", runner=cluster, **kw)


# ------------------------------------------------------------------ the commands


def test_a_box_is_one_sbatch_with_the_tag_as_its_name_and_the_lease_as_its_time(cluster):
    s = provider(cluster, partition="gpu", account="lab", qos="high", constraint="h100",
                 sbatch_args=("--gres-flags=enforce-binding",))  # fmt: skip
    spec = L.Spec(image="", gpus=(), max_dph_per_gpu=0, num_gpus=2, min_cpus=16, min_ram_gb=64)
    expiry = int(time.time()) + 3600
    tag = make_tag(RUN_ID, 1, expiry)
    jid = s.create({"id": 1}, spec, tag, None, expiry, avoid={"node7", 42, "bad;name"})
    argv, script = cluster.calls[-1]
    f = cluster.flags(argv)
    assert argv[0] == "sbatch" and "--parsable" in argv and jid == cluster.next - 1
    assert f["--job-name"] == tag and f["--gres"] == "gpu:2" and f["--nodes"] == "1"
    assert 60 + sl.SLACK_MIN <= int(f["--time"]) <= 61 + sl.SLACK_MIN
    assert (f["--partition"], f["--account"], f["--qos"], f["--constraint"]) == (
        "gpu", "lab", "high", "h100",
    )  # fmt: skip
    assert f["--cpus-per-task"] == "16" and f["--mem"] == "64G"
    assert f["--exclude"] == "node7"  # only what reads as a node name
    assert argv[-1] == "--gres-flags=enforce-binding"  # the caller's own, last
    assert script == sl.batch_script(expiry)


def test_without_cpus_or_memory_asked_for_the_cluster_defaults_apply(cluster):
    provider(cluster).create({"id": 1}, SPEC, make_tag(RUN_ID, 1, EXPIRY), None, EXPIRY)
    f = cluster.flags(cluster.calls[-1][0])
    assert "--cpus-per-task" not in f and "--mem" not in f and "--partition" not in f


@pytest.mark.parametrize(
    "answer, outcome",
    [
        ((0, "123\n", ""), 123),
        ((0, "123;cluster2\n", ""), 123),
        ((0, "Welcome to the cluster!\n123\n", ""), 123),  # a chatty login shell
        ((0, "123\nand then something\n", ""), prov.CreateUnknown),
        ((0, "Submitted batch job 123\n", ""), prov.CreateUnknown),
        ((0, "", ""), prov.CreateUnknown),
        ((255, "", "ssh: connect to host login: Connection timed out"), prov.CreateUnknown),
        ((None, "", ""), prov.CreateUnknown),
        (
            (1, "", "sbatch: error: Batch job submission failed: Socket timed out on send/recv"),
            prov.CreateUnknown,
        ),
        (
            (1, "", "sbatch: error: Batch job submission failed: Invalid partition name specified"),
            ConfigError,
        ),
        ((1, "", "sbatch: error: Invalid generic resource (gres) specification"), ConfigError),
        ((127, "", "bash: sbatch: command not found"), ConfigError),
        ((1, "", "something else entirely"), prov.CreateUnknown),
    ],
)
def test_a_job_id_only_from_an_explicit_yes_and_a_refusal_is_ours(cluster, answer, outcome):
    cluster.fail["sbatch"] = [answer]
    s = provider(cluster)
    tag = make_tag(RUN_ID, 1, EXPIRY)
    if isinstance(outcome, int):
        assert s.create({"id": 1}, SPEC, tag, None, EXPIRY) == outcome
    else:
        with pytest.raises(outcome):
            s.create({"id": 1}, SPEC, tag, None, EXPIRY)


def test_sbatch_is_never_retried(cluster):
    cluster.fail["sbatch"] = [(255, "", "connection reset"), (0, "5\n", "")]
    with pytest.raises(prov.CreateUnknown):
        provider(cluster).create({"id": 1}, SPEC, make_tag(RUN_ID, 1, EXPIRY), None, EXPIRY)
    assert cluster.names("sbatch") == ["sbatch"]


def test_a_job_that_is_over_or_unknown_is_gone_and_an_error_never_is(cluster):
    s = provider(cluster)
    jid = s.create({"id": 1}, SPEC, make_tag(RUN_ID, 1, EXPIRY), None, EXPIRY)
    assert s.instance(jid)["actual_status"] == "RUNNING"
    for state in ("COMPLETING", "CANCELLED", "COMPLETED", "TIMEOUT", "NODE_FAIL"):
        cluster.jobs[jid]["state"] = state
        assert s.instance(jid) is None, state
    for state in ("PENDING", "SUSPENDED", "REQUEUED", "CONFIGURING"):
        cluster.jobs[jid]["state"] = state
        assert s.instance(jid) is not None, state
    assert s.instance(99) is None  # "Invalid job id specified": the controller forgot it
    errors = [
        (1, "", "slurm_load_jobs error: Unable to contact slurm controller (connect failure)"),
        (255, "", "ssh: Could not resolve hostname"),
        (None, "", ""),
        (0, "", ""),  # nothing: not the same as "it does not exist"
        (0, f"{jid + 1}|RUNNING|node1|None|x\n", ""),  # another job
        (0, f"{jid}|RUNNING\n", ""),  # unreadable
    ]
    for answer in errors:
        cluster.fail["squeue"] = [answer, answer]
        with pytest.raises(sl.SlurmError):
            s.instance(jid, retries=1)


def test_a_lookup_whose_connection_failed_is_tried_again(cluster):
    s = provider(cluster)
    jid = s.create({"id": 1}, SPEC, make_tag(RUN_ID, 1, EXPIRY), None, EXPIRY)
    cluster.fail["squeue"] = [(255, "", "Connection closed by remote host")]
    assert s.instance(jid)["id"] == jid


def test_the_listing_is_every_job_of_the_user_or_an_error(cluster):
    s = provider(cluster)
    tag = make_tag(RUN_ID, 1, EXPIRY)
    lines = [
        f"11|RUNNING|node1|None|{tag}",
        "12|PENDING||Priority|my|job|with|bars",  # the name is last, and may hold anything
        "13_[1-4]|PENDING||Priority|an-array",  # not ours: we never submit arrays
        "14|COMPLETING|node2|None|done",  # over
    ]
    cluster.fail["squeue"] = [(0, "\n".join(lines) + "\n", "")]
    rows = s.instances()
    assert [(r["id"], r["label"]) for r in rows] == [(11, tag), (12, "my|job|with|bars")]
    bad = [
        f"13_1|RUNNING|node1|None|{tag}",  # a tag on something we cannot have rented
        "11|RUNNING|node1",  # cut short
        "11|RUNNING|node1|None|a\n11|RUNNING|node1|None|b",  # twice
        "11|running|node1|None|a",
    ]
    for text in bad:
        cluster.fail["squeue"] = [(0, text + "\n", "")]
        with pytest.raises(sl.SlurmError):
            s.instances()
    cluster.fail["squeue"] = [(1, "", "squeue: error: Invalid user"), (1, "", "again")]
    with pytest.raises(sl.SlurmError):
        s.instances()


def test_cancelling_a_job_slurm_has_forgotten_is_not_an_error_and_a_refusal_is(cluster):
    s = provider(cluster)
    s.destroy(99)
    cluster.fail["scancel"] = [(1, "", "scancel: error: Kill job error on job id 5: Access/permission denied"), (1, "", "")]  # fmt: skip
    with pytest.raises(sl.SlurmError):
        s.destroy(5)
    cluster.fail["scancel"] = [(0, "", "scancel: error: Kill job error on job id 5: Access/permission denied")]  # fmt: skip
    with pytest.raises(sl.SlurmError):
        s.destroy(5)


def test_a_relabel_sets_the_name_and_a_time_limit_that_covers_the_new_lease(cluster):
    s = provider(cluster)
    jid = s.create({"id": 1}, SPEC, make_tag(RUN_ID, 1, EXPIRY), None, EXPIRY)
    s.instance(jid)
    cluster.jobs[jid]["started"] = time.time() - 600  # ten minutes in
    later = EXPIRY + 7200
    s.relabel(jid, make_tag(RUN_ID, 1, later))
    j = cluster.jobs[jid]
    assert parse_tag(j["name"]).expiry == later
    left = (later - time.time()) / 60
    assert 10 + left + sl.SLACK_MIN - 1 <= j["limit"] <= 10 + left + sl.SLACK_MIN + 1
    update = [a for a, _ in cluster.calls if a[:2] == ["scontrol", "update"]]
    assert len(update) == 1  # the name and the limit in one request


def test_a_time_limit_the_cluster_will_not_extend_fails_the_relabel(cluster):
    s = provider(cluster)
    jid = s.create({"id": 1}, SPEC, make_tag(RUN_ID, 1, EXPIRY), None, EXPIRY)
    cluster.fail["scontrol update"] = [(1, "", "Access/permission denied for job 4")]
    with pytest.raises(sl.SlurmError, match="permission denied"):
        s.relabel(jid, make_tag(RUN_ID, 1, EXPIRY + 7200))
    assert parse_tag(cluster.jobs[jid]["name"]).expiry == EXPIRY


def test_how_a_job_reads(cluster):
    s = provider(cluster)
    view = s.view({"id": 5, "actual_status": "RUNNING", "node": "node3", "reason": "None"})
    assert (view.state, view.endpoint, view.machine, view.msg) == (
        "running", ("login.example", 5), "node3", "",
    )  # fmt: skip
    assert s.view({"id": 5, "actual_status": "PENDING", "reason": "Priority"}).state == "queued"
    never = s.view({"id": 5, "actual_status": "PENDING", "reason": "PartitionTimeLimit"})
    assert (never.state, never.problem) == ("queued", "config")
    assert s.view({"id": 5, "actual_status": "CONFIGURING"}).state == "loading"


def test_the_account_is_the_user_on_the_cluster(cluster):
    assert provider(cluster).account() == "alice@testcluster"


def test_a_login_that_could_be_an_ssh_option_is_refused():
    for bad in ("-oProxyCommand=evil", "-q", "a b", "", "x\ny"):
        with pytest.raises(ConfigError):
            sl.Slurm(bad)
    with pytest.raises(ConfigError):
        sl.Slurm("login", sbatch_args=("not-an-option",))


def test_a_step_runs_in_the_job_with_the_boxs_variables():
    argv = sl.SlurmProber().ssh(("login.example", 42), "echo hi")
    assert argv[:2] == ["ssh", "-T"] and "BatchMode=yes" in argv and argv[-2] == "login.example"
    step = argv[-1]
    assert step.startswith("srun --jobid=42 --overlap ") and "--pty" not in step
    assert sl.BOX_ENV in step and step.endswith("echo hi'")
    shell = sl.SlurmProber().ssh(("login.example", 42), None, tty=True)
    assert shell[:2] == ["ssh", "-t"] and "--pty" in shell[-1] and "exec bash -l" in shell[-1]


def test_the_gate_looks_only_at_the_allocation(monkeypatch):
    started = []
    monkeypatch.setattr(
        sl.SlurmProber, "_start", lambda self, argv, read, **kw: started.append(argv)
    )
    sl.SlurmProber().start_gate(("login.example", 42), SPEC)
    assert '"scoped": true' in started[0][-1]


@pytest.mark.parametrize("which", ["slurm", "job"])
def test_a_process_left_holding_stderr_does_not_hold_up_a_command(tmp_path, monkeypatch, which):
    # What an ssh that starts a ControlMaster leaves behind.
    fake = tmp_path / "ssh"
    fake.write_text("#!/bin/sh\n(sleep 20 >&2 &)\necho answer\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    t0 = time.monotonic()
    if which == "slurm":
        rc, out, _ = sl.run_ssh("login.example", ["true"], None, 15)
    else:
        rc, out, _ = job.run_remote(lambda remote: ["ssh", remote], "true", timeout=15)
    assert (rc, out.strip()) == (0, "answer") and time.monotonic() - t0 < 5


def test_the_batch_script_is_bash_and_a_job_is_started_by_it():
    script = sl.batch_script(EXPIRY)
    assert subprocess.run(["bash", "-n"], input=script.encode()).returncode == 0
    remote, program = job.start_command(job.new_job_id(), "true\n", {}, spawn="runner")
    assert '"$d/start"' in program and 'setsid bash "$d/wrapper"' not in program
    assert subprocess.run(["bash", "-n"], input=program.encode()).returncode == 0


# A node's nvidia-smi where devices are not confined to the job (as on some clusters): it lists
# all eight GPUs, and GPU 5, another job's, is busy. `-i LIST` lists only those.
NODE_SMI = """#!/bin/sh
if [ $# -eq 0 ]; then echo "| Driver Version: 580.1   CUDA Version: 13.0 |"; exit 0; fi
only=
while [ $# -gt 0 ]; do [ "$1" = -i ] && only=",$2,"; shift; done
for i in 0 1 2 3 4 5 6 7; do
  case "$only" in ""|*",$i,"*) ;; *) continue;; esac
  if [ $i = 5 ]; then used=40000; util=97; else used=0; util=0; fi
  echo "$i, NVIDIA H100 80GB HBM3, 81559, $used, $util, 70.1, 580.1"
done
"""


def node_gate(tmp_path, visible, **want):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "nvidia-smi").write_text(NODE_SMI)
    (bin_dir / "nvidia-smi").chmod(0o755)
    want = {"num_gpus": 1, "gpus": [], "min_vram_mib": 0, "capability": None,
            "must_reach": [], **want}  # fmt: skip
    r = subprocess.run(
        [sys.executable, "-S", "-", json.dumps(want)],
        input=gate.GATE_SCRIPT, capture_output=True, text=True, timeout=60,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": visible},
    )  # fmt: skip
    return gate.read_gate(gate.Done(r.returncode, r.stdout, r.stderr))


def test_in_an_allocation_the_gate_looks_at_its_gpus_and_not_the_nodes(tmp_path):
    ok, reason, detail, report = node_gate(tmp_path, "3", scoped=True)
    assert (ok, reason) == (True, None), detail
    assert report["gpus"] == ["NVIDIA H100 80GB HBM3"]
    two = node_gate(tmp_path, "2,4", scoped=True, num_gpus=2, gpus=["H100"])
    assert two[:2] == (True, None), two[2]
    # Given another job's busy GPU, it says so; and it would, unscoped, count the node's.
    assert node_gate(tmp_path, "5", scoped=True)[:2] == (False, "busy_gpu")
    assert node_gate(tmp_path, "3")[:2] == (False, "wrong_hardware")


def test_a_cancelled_box_leaves_its_node_clean():
    jid = 7_999_000 + os.getpid() % 1000
    box = Path(f"/tmp/gpubox-{jid}")
    env = {**os.environ, "SLURM_JOB_ID": str(jid), "PATH": f"{SHIM}:{os.environ['PATH']}"}
    proc = subprocess.Popen(["bash", "-s"], stdin=subprocess.PIPE, env=env,
                            start_new_session=True)  # fmt: skip
    proc.stdin.write(sl.batch_script(EXPIRY).encode())
    proc.stdin.close()
    try:
        assert wait_until(lambda: (box / "expiry").exists())
        proc.terminate()  # what scancel sends first
        proc.wait(timeout=10)
        assert not box.exists()
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        shutil.rmtree(box, ignore_errors=True)


# ------------------------------------------------------------------ the launcher


class GateProber(sl.SlurmProber):
    """Steps that answer at once; the gate's verdict per node from `behaviour`."""

    def __init__(self, cluster, behaviour=None):
        super().__init__()
        self.cluster, self.behaviour = cluster, behaviour or {}
        self.ran = []

    def start_ready(self, ep):
        self.checks.append(FakeCheck(lambda: (True, "")))
        return self.checks[-1]

    def start_gate(self, ep, spec):
        node = self.cluster.jobs[ep[1]]["node"]
        verdict = {
            "busy": (False, "busy_gpu", "util 100%", {}),
        }.get(self.behaviour.get(node), (True, None, "", {"gpus": ["H100"]}))
        self.checks.append(FakeCheck(lambda: verdict))
        return self.checks[-1]

    def run(self, ep, remote, timeout=60):
        self.ran.append((ep, remote))
        return 0


def launch(tmp_path, cluster, behaviour=None, **policy_kw):
    policy = L.Policy(**{**FAST.__dict__, "spare": 0, **policy_kw})
    s = provider(cluster)
    state = State.begin(tmp_path, RUN_ID, EXPIRY, None, {"provider": "slurm"})
    return L.Launch(s, SPEC, policy, state, GateProber(cluster, behaviour)), s, state


def test_a_job_waiting_in_the_queue_is_not_failed_for_it(tmp_path, cluster):
    cluster.pending = 40  # far more lookups than the start-up timeout allows
    run, s, state = launch(tmp_path, cluster, startup_timeout=0.05, stall_timeout=0.05)
    result = run.run({})
    (w,) = result.winners
    assert w.machine_id == "node1" and cluster.names("sbatch") == ["sbatch"]
    box = state.winners[0]
    assert (box["host"], box["port"], box["machine"]) == ("login.example", w.instance_id, "node1")


def test_a_queue_that_never_lets_the_job_run_ends_the_launch_and_cancels_it(tmp_path, cluster):
    cluster.pending = 10**6
    run, s, state = launch(tmp_path, cluster, overall_timeout=0.5)
    assert run.run({}).winners == []
    assert [j["state"] for j in cluster.jobs.values()] == ["CANCELLED"]


def test_a_request_the_cluster_can_never_run_is_a_configuration_error(tmp_path, cluster):
    cluster.pending, cluster.reason = 10**6, "PartitionTimeLimit"
    run, s, state = launch(tmp_path, cluster)
    with pytest.raises(ConfigError, match="PartitionTimeLimit"):
        run.run({})
    assert [j["state"] for j in cluster.jobs.values()] == ["CANCELLED"]


def test_a_refused_submission_is_a_configuration_error_and_nothing_is_left(tmp_path, cluster):
    cluster.fail["sbatch"] = [(1, "", "sbatch: error: Batch job submission failed: Invalid qos")]
    run, s, state = launch(tmp_path, cluster)
    with pytest.raises(ConfigError, match="Invalid qos"):
        run.run({})
    assert cluster.jobs == {}


def test_a_node_that_fails_the_gate_is_excluded_from_the_next_job(tmp_path, cluster):
    run, s, state = launch(tmp_path, cluster, {"node1": "busy"})
    (w,) = run.run({}).winners
    first, second = (j for j in cluster.jobs.values())
    assert first["node"] == "node1" and first["state"] == "CANCELLED"
    assert second["flags"]["--exclude"] == "node1" and w.machine_id == "node2"


def test_down_cancels_the_runs_jobs_by_id_and_by_tag(tmp_path, cluster):
    run, s, state = launch(tmp_path, cluster)
    run.run({})
    stray = s.create({"id": 9}, SPEC, make_tag(RUN_ID, 9, EXPIRY), None, EXPIRY)  # by tag only
    other = s.create({"id": 9}, SPEC, make_tag("dev.other.0123456789ab", 1, EXPIRY), None, EXPIRY)
    assert L.down(s, state, FAST) == []
    assert cluster.jobs[stray]["state"] == "CANCELLED"
    assert cluster.jobs[other]["state"] != "CANCELLED"
    assert all(j["state"] == "CANCELLED" for k, j in cluster.jobs.items() if k != other)


def test_renew_moves_the_boxs_own_timer_then_its_name_and_time_limit(tmp_path, cluster):
    run, s, state = launch(tmp_path, cluster)
    run.run({})
    later = EXPIRY + 7200
    L.renew(s, state, run.prober, later)
    ((ep, remote),) = run.prober.ran
    assert remote == s.expiry_command(later) and str(later) in remote
    (j,) = cluster.jobs.values()
    assert parse_tag(j["name"]).expiry == later and state.expiry == later


# ------------------------------------------------------------- the whole CLI


class LocalProber(sl.SlurmProber):
    """A step of a job of the local cluster: a local bash with the job's id; the gate
    passes without looking (there is no GPU here)."""

    path = SHIM + ":" + os.environ["PATH"]
    home = "/"

    def ssh(self, ep, remote, *, connect_timeout=10, tty=None):
        inner = sl.BOX_ENV + ("exec bash" if remote is None else remote)
        env = ["env", f"SLURM_JOB_ID={ep[1]}", f"PATH={self.path}", f"HOME={self.home}"]
        return [*env, "bash", "-c", inner]

    def start_gate(self, ep, spec):
        self.checks.append(FakeCheck(lambda: (True, None, "", {"gpus": ["H100"], "cpus": 8})))
        return self.checks[-1]


@pytest.fixture
def local_cluster(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    c = Cluster(local=True, home=home)
    monkeypatch.setattr(sl, "run_ssh", lambda login, argv, stdin, t: c(argv, stdin, t))
    monkeypatch.setattr(sl, "SlurmProber", LocalProber)
    monkeypatch.setattr(LocalProber, "home", str(home))
    monkeypatch.setattr(cli, "Policy", lambda **kw: type(FAST)(**{**FAST.__dict__, **kw}))
    monkeypatch.setattr(job, "KILL_GRACE", 1)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GPUBOX_ACCOUNT_ID", raising=False)
    yield c
    c.close()


def gpubox(tmp_path, *argv):
    return cli.main([*argv, "--state-dir", str(tmp_path / "state")])


UP = ["up", "--provider", "slurm", "--login", "login.example", "--namespace", "dev",
      "--name", "box", "--ttl-min", "30"]  # fmt: skip


def wait_until(what, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if what():
            return True
        time.sleep(0.1)
    return False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shells")
def test_up_run_renew_down_on_a_cluster(tmp_path, local_cluster, capsys):
    assert gpubox(tmp_path, *UP) == 0
    state = State.load(tmp_path / "state")
    (w,) = state.winners
    jid = w["id"]
    assert state.meta["provider"] == "slurm" and state.meta["options"]["login"] == "login.example"
    assert (w["host"], w["port"], w["machine"]) == ("login.example", jid, "node1")
    box = Path(f"/tmp/gpubox-{jid}")
    assert wait_until(lambda: (box / "expiry").exists())
    assert (box / "expiry").read_text().strip() == str(state.expiry)

    gpubox(tmp_path, "status")
    out = capsys.readouterr().out
    assert f"job {jid}" in out and "on node1" in out

    # A job is started by the box's own batch script, and followed to its end.
    (tmp_path / "s.sh").write_text('echo "hello from $GPUBOX_BOX_NAME"; exit 3\n')
    assert gpubox(tmp_path, "run", "--script", str(tmp_path / "s.sh")) == 3
    assert "hello from box" in capsys.readouterr().out
    started = list(box.glob("jobs/*/started"))
    assert len(started) == 1 and not list(box.glob("jobs/*/start"))

    later_min = 120
    assert gpubox(tmp_path, "renew", "--ttl-min", str(later_min)) == 0
    renewed = State.load(tmp_path / "state").expiry
    assert (box / "expiry").read_text().strip() == str(renewed)
    j = local_cluster.jobs[jid]
    assert parse_tag(j["name"]).expiry == renewed and j["limit"] >= later_min

    assert gpubox(tmp_path, "down") == 0
    assert j["state"] == "CANCELLED" and not box.exists()
    assert State.load(tmp_path / "state").closed


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shells")
def test_a_job_can_end_its_box_on_a_cluster(tmp_path, local_cluster):
    assert gpubox(tmp_path, *UP) == 0
    (w,) = State.load(tmp_path / "state").winners
    (tmp_path / "s.sh").write_text("echo done\n")
    assert gpubox(tmp_path, "run", "--destroy-after", "--script", str(tmp_path / "s.sh")) == 0
    j = local_cluster.jobs[w["id"]]
    # The batch script sees its lease end, and exits: Slurm ends the job.
    assert wait_until(lambda: j["proc"].poll() is not None)
    assert not Path(f"/tmp/gpubox-{w['id']}").exists()
    assert gpubox(tmp_path, "down") == 0


def test_slurm_refuses_what_only_vast_has(tmp_path, local_cluster, capsys):
    for extra in (["--image", "x"], ["--max-dph-per-gpu", "1"], ["--ssh-config"],
                  ["--exclude-country", "US"]):  # fmt: skip
        assert gpubox(tmp_path, *UP, *extra) == cli.EXIT_CONFIG
        assert "not on slurm" in capsys.readouterr().out
    assert local_cluster.calls == []


def test_vast_refuses_what_only_slurm_has_and_still_needs_its_own(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VASTAI_API_KEY", "k")
    vast = ["up", "--gpu", "RTX 5090", "--max-dph-per-gpu", "1", "--image", "i",
            "--namespace", "dev", "--name", "box"]  # fmt: skip
    assert gpubox(tmp_path, *vast, "--partition", "gpu") == cli.EXIT_CONFIG
    assert "--partition: not for vast" in capsys.readouterr().out
    assert gpubox(tmp_path, *vast[:1], *vast[3:]) == cli.EXIT_CONFIG  # no --gpu
    assert "vast needs --gpu" in capsys.readouterr().out


def test_slurm_needs_a_login_host(tmp_path, local_cluster, capsys):
    assert gpubox(tmp_path, "up", "--provider", "slurm", "--namespace", "dev", "--name", "b") == 2
    assert "--login HOST" in capsys.readouterr().out


def test_down_by_run_id_alone_finds_the_jobs_by_tag(tmp_path, local_cluster):
    assert gpubox(tmp_path, *UP) == 0
    state = State.load(tmp_path / "state")
    elsewhere = tmp_path / "elsewhere"
    rc = cli.main(["down", "--run-id", state.run, "--provider", "slurm", "--login",
                   "login.example", "--state-dir", str(elsewhere)])  # fmt: skip
    assert rc == 0
    assert [j["state"] for j in local_cluster.jobs.values()] == ["CANCELLED"]


def test_reap_cancels_jobs_whose_lease_has_ended(tmp_path, local_cluster):
    s = provider(local_cluster)
    old = s.create({"id": 1}, SPEC, make_tag(RUN_ID, 1, int(time.time()) - 3600), None, EXPIRY)
    new = s.create({"id": 2}, SPEC, make_tag(RUN_ID, 2, EXPIRY), None, EXPIRY)
    argv = ["reap", "--provider", "slurm", "--login", "login.example", "--namespace", "dev",
            "--destroy"]  # fmt: skip
    assert cli.main(argv) == 0
    assert local_cluster.jobs[old]["state"] == "CANCELLED"
    assert local_cluster.jobs[new]["state"] != "CANCELLED"


def test_the_library_rents_from_a_cluster_and_tears_down_at_the_end(tmp_path, local_cluster):
    import gpubox

    where = {"provider": "slurm", "options": {"login": "login.example"}}
    spec = L.Spec(image="", gpus=(), max_dph_per_gpu=0, ttl_min=30)
    fast = L.Policy(**{**FAST.__dict__, "spare": 0})
    with gpubox.up(
        spec, namespace="dev", name="lib", policy=fast, state_dir=tmp_path / "s", **where
    ) as run:
        assert run.on_slurm and run.provider.name == "slurm"
        (box,) = run.boxes
        assert box.machine == "node1" and "job" in repr(box)
        (j,) = run.start("echo hi; exit 4\n")
        assert j.wait(quiet=True) == 4
        jid = box.instance_id
    assert local_cluster.jobs[jid]["state"] == "CANCELLED"
    with pytest.raises(ConfigError, match="not on slurm: an image"):
        gpubox.up(L.Spec(image="img", gpus=(), max_dph_per_gpu=0, ttl_min=30), namespace="dev",
                  name="lib", state_dir=tmp_path / "t", **where)  # fmt: skip
    with pytest.raises(ConfigError, match="login host"):
        gpubox.up(spec, namespace="dev", name="lib", state_dir=tmp_path / "u", provider="slurm")
