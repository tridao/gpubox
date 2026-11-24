"""A fake Vast and fake SSH for the launcher tests. No network, no GPU, no threads.

The launcher is one thread, so the fakes are plain objects: a test steers a
launch through `hooks`, functions called at the start of a provider call, which
may change the fake, raise, or send this process a signal.
"""

import fcntl
import json
import time

from gpubox import vast as v
from gpubox.launch import Launch, Policy, Spec
from gpubox.state import State, make_tag

RUN_ID = "quack-ci.test.0123456789ab"
OTHER_RUN = "quack-ci.other.89abcdef0123"  # same namespace, another run
EXPIRY = int(time.time()) + 3600
FOREIGN = (
    "vast-5090-4x-ca",
    None,
    "quack-ci-old.x.12345678.a1",  # looks like a tag, and is not one
    make_tag(OTHER_RUN, 1, EXPIRY),
)
FAST = Policy(
    poll=0.01,
    startup_timeout=0.3,
    overall_timeout=5,
    cleanup_reserve=0,
    settle=0.15,
    min_lease=1,
    rollback=0.3,
    confirm_timeout=0.3,
    confirm_poll=0.01,
    round=0.05,
    recheck=0.01,
    call_timeout=1,
)
SPEC = Spec(image="img", gpus=("RTX 5090",), max_dph_per_gpu=1.25)
REPORT = {"gpus": ["RTX 5090"], "cpus": 32, "driver": "1", "torch": "2"}


def offer(machine, host=None, dph=1.0, gpu="RTX 5090", cuda=None):
    o = {
        "id": 1000 + machine,
        "machine_id": machine,
        "host_id": host or machine,
        "dph_total": dph,
        "gpu_name": gpu,
    }
    if cuda is not None:
        o["cuda_max_good"] = cuda
    return o


class FakeVast:
    """`behaviour` maps machine_id to how a box rented on it behaves.

    ok        comes up and passes the gate
    bad_gpu   comes up, gate reports cuda_init
    busy      comes up, gate reports busy_gpu
    old_driver comes up, gate says its driver is for an older CUDA than the image needs (13)
    stuck     never leaves "loading", and its status message never changes
    pulling   never leaves "loading", but its status message changes at every look
    cdi       status message blames the machine's container runtime
    image     status message says the image cannot be pulled
    exited    container exits during bootstrap
    stale     create is refused (offer gone)
    ghost     create times out and no box exists
    lost      create times out but the box exists; listed after two listings
    lost_late as lost, but listed only once `revealed` is set
    late      its gate never finishes
    slow      passes the gate, but only after `slow_secs`
    hold      passes the gate once the test sets `release`
    vanish    passes the gate, and the instance is gone the moment it does
    api_down  rented, but the provider never answers a status query
    nokey     comes up, and its sshd refuses the run's key
    hook_fail passes the gate; the caller's gate script fails without a verdict
    hook_net  passes the gate; the caller's gate script blames the network
    hook_late passes the gate; the caller's gate script never finishes

    It is the Vast client; the launcher sees it through `v.VastProvider`. `down`, the
    reaper and the destroy helpers take it as it is, as they take any provider.
    """

    noun = "instance"

    def __init__(self, behaviour, offers=None, slow_secs=0.2):
        self.behaviour = behaviour
        self._offers = offers if offers is not None else [offer(m) for m in behaviour]
        self.live, self.destroyed, self._next = {}, [], 1
        self.creates = self.peak = 0
        self.fail_destroy = {}  # machine -> number of destroy calls that raise VastError first
        self.sticky = set()  # machines whose box survives a destroy
        self.listings_until_visible = {}
        self.revealed = False
        self.hidden = set()  # ids no listing shows
        self.unlabelled = set()  # ids listed with no label
        self.lookup_hidden = set()  # ids whose lookup says "does not exist" although they do
        self.listing_lag = 0  # listings that still show a box after it was destroyed
        self.lingering = {}  # id -> [its row, listings left]
        self.slow_secs = slow_secs
        self.release = False  # lets "hold" boxes through the gate
        self.gate_started = set()  # machines whose gate has started
        self.list_fails = 0  # number of listing calls that raise first
        self.empty_listings = 0  # number of listings that are well formed, and stale: no rows
        self.account_id = "111"
        self.queries, self.payloads = [], []  # what was asked of the provider
        self.calls = []  # (name, argument, keyword arguments) of every call, in order
        self.hooks = {}  # name -> function(argument), run at the start of that call
        for i, label in enumerate(FOREIGN):
            self.live[9000 + i] = {"id": 9000 + i, "label": label, "machine": None}

    def _enter(self, name, arg=None, kw=None):
        self.calls.append((name, arg, kw or {}))
        if name in self.hooks:
            self.hooks[name](arg)

    def ours(self):
        return sorted(i for i, b in self.live.items() if b["machine"] is not None)

    def names(self, *wanted):
        """The calls made so far, as (name, argument), optionally only some names."""
        return [(n, a) for n, a, _ in self.calls if not wanted or n in wanted]

    def account(self):
        self._enter("account")
        return self.account_id

    def offers(self, query):
        self._enter("offers")
        self.queries.append(query)
        return list(self._offers)

    def create(self, offer_id, payload):
        machine = offer_id - 1000
        self._enter("create", machine)
        kind = self.behaviour[machine]
        if kind == "stale":
            raise v.CreateRefused(400, "offer no longer available")
        if kind == "ghost":
            raise v.CreateUnknown(0, "timed out")
        self.payloads.append(payload)
        iid, self._next = self._next, self._next + 1
        self.live[iid] = {
            "id": iid,
            "label": payload["label"],
            "machine": machine,
            "start_date": time.time(),
            "image_uuid": payload.get("image"),
        }
        self.creates += 1
        self.peak = max(self.peak, len(self.ours()))
        self._enter("created", iid)  # the box exists; the answer has not been given yet
        if kind in ("lost", "lost_late"):
            self.listings_until_visible[iid] = 2
            raise v.CreateUnknown(0, "timed out")
        return iid

    def instance(self, iid, **kw):
        self._enter("instance", iid, kw)
        box = self.live.get(iid)
        if box is None or iid in self.lookup_hidden:
            return None
        kind = self.behaviour.get(box["machine"], "ok")
        if kind == "api_down":
            raise v.VastError(503, "unavailable")
        row = {"id": iid, "label": box["label"], "actual_status": "loading", "status_msg": ""}
        if kind == "cdi":
            row["status_msg"] = (
                "Error response from daemon: unresolvable CDI devices nvidia.com/gpu"
            )
        elif kind == "image":
            row["status_msg"] = "Error response from daemon: manifest unknown"
        elif kind == "exited":
            row["actual_status"] = "exited"
        elif kind == "pulling":
            row["status_msg"] = f"layer: Downloading {len(self.calls)} MB"
        elif kind != "stuck":
            row.update(
                actual_status=box.get("status", "running"),
                public_ipaddr=f"10.0.0.{box['machine']}",
                ports={"22/tcp": [{"HostPort": "2222"}]},
            )
        return row

    def instances(self, **kw):
        self._enter("instances", None, kw)
        if self.list_fails > 0:
            self.list_fails -= 1
            raise v.VastError(503, "unavailable")
        if self.empty_listings > 0:
            self.empty_listings -= 1
            return []
        rows = []
        for iid, box in self.live.items():
            kind = self.behaviour.get(box["machine"])
            if iid in self.hidden or (kind == "lost_late" and not self.revealed):
                continue
            if kind == "lost" and self.listings_until_visible.get(iid, 0) > 0:
                self.listings_until_visible[iid] -= 1
                continue
            row = {k: box.get(k) for k in ("id", "label", "start_date", "image_uuid")}
            if iid in self.unlabelled:
                row["label"] = None
            rows.append({**row, "machine_id": box["machine"], "actual_status": "running"})
        for iid, entry in list(self.lingering.items()):
            if iid not in self.live and entry[1] > 0:
                entry[1] -= 1
                rows.append(entry[0])
        return rows

    def relabel(self, iid, label):
        self._enter("relabel", iid)
        if iid not in self.live:
            raise v.VastError(404, "no such instance")
        self.live[iid]["label"] = label

    def destroy(self, iid, **kw):
        self._enter("destroy", iid, kw)
        box = self.live.get(iid)
        if box is None:
            return
        machine = box["machine"]
        if self.fail_destroy.get(machine, 0) > 0:
            self.fail_destroy[machine] -= 1
            raise v.VastError(500, "internal error")
        if machine in self.sticky:
            return
        self.live.pop(iid)
        self.destroyed.append(iid)
        if self.listing_lag:
            row = {
                "id": iid,
                "label": box["label"],
                "machine_id": machine,
                "actual_status": "exited",
            }
            self.lingering[iid] = [row, self.listing_lag]


class FileVast(FakeVast):
    """A FakeVast whose boxes live in a file, so they outlive a killed process."""

    def __init__(self, path, behaviour):
        super().__init__(behaviour)
        self.path = path
        self.live = {}
        if not path.exists():
            path.write_text(json.dumps({"next": 1, "live": {}}))

    def _load(self):
        data = json.loads(self.path.read_text())
        self._next = data["next"]
        self.live = {int(k): b for k, b in data["live"].items()}

    def _save(self):
        self.path.write_text(json.dumps({"next": self._next, "live": self.live}))

    def _with_file(self, fn, *args, **kw):
        with open(str(self.path) + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._load()
            try:
                return fn(*args, **kw)
            finally:
                self._save()

    def create(self, offer_id, payload):
        return self._with_file(super().create, offer_id, payload)

    def instance(self, iid, **kw):
        return self._with_file(super().instance, iid, **kw)

    def instances(self, **kw):
        return self._with_file(super().instances, **kw)

    def destroy(self, iid, **kw):
        return self._with_file(super().destroy, iid, **kw)

    def on_disk(self):
        """The ids of our boxes as another process would find them."""
        self._load()
        return self.ours()


class FakeCheck:
    """A child process that is done when `result()` stops returning None."""

    def __init__(self, result):
        self.result, self.killed = result, False

    def poll(self):
        return None if self.killed else self.result()

    def kill(self):
        self.killed = True


class FakeProber:
    pubkey = "ssh-ed25519 AAAAtest gpubox"

    def __init__(self, vast):
        self.vast = vast
        self.wants = []  # the spec each gate was asked to check
        self.hooked = []  # machines whose gate script was started
        self.checks = []  # every child process started
        self.ran, self.run_rc = [], 0  # commands run on a box, and what they return

    def run(self, ep, remote, timeout=60):
        self.ran.append((ep, remote))
        return self.run_rc

    def kill_all(self):
        self.vast._enter("kill_all")
        for check in self.checks:
            check.kill()

    def start_ready(self, ep):
        machine = int(ep[0].rsplit(".", 1)[1])
        refused = self.vast.behaviour.get(machine) == "nokey"
        answer = (False, f"root@{ep[0]}: Permission denied (publickey).") if refused else (True, "")
        self.checks.append(FakeCheck(lambda: answer))
        return self.checks[-1]

    def start_gate(self, ep, spec):
        vast = self.vast
        machine = int(ep[0].rsplit(".", 1)[1])
        kind = vast.behaviour[machine]
        self.wants.append(spec)
        vast.gate_started.add(machine)
        started = time.monotonic()

        def result():
            vast._enter("gate_poll", machine)  # a hook point while boxes sit in their gates
            if kind == "late" or (kind == "hold" and not vast.release):
                return None
            if kind == "slow" and time.monotonic() - started < vast.slow_secs:
                return None
            if kind == "bad_gpu":
                return False, "cuda_init", "CUDA unavailable", {}
            if kind == "busy":
                return False, "busy_gpu", "util 100%", {}
            if kind == "old_driver":
                return False, None, "driver supports CUDA 12.8", {"needs_cuda": 13}
            if kind == "vanish":
                for iid in [i for i, b in vast.live.items() if b["machine"] == machine]:
                    vast.live.pop(iid)
            return True, None, "", dict(REPORT)

        self.checks.append(FakeCheck(result))
        return self.checks[-1]

    def start_hook(self, ep, spec):
        machine = int(ep[0].rsplit(".", 1)[1])
        kind = self.vast.behaviour[machine]
        assert spec.gate_script is not None
        self.hooked.append(machine)
        answers = {
            "hook_fail": (False, None, "gate script exited 1: missing library", {}),
            "hook_net": (False, "no_network", "gate script: storage at 3 MB/s", {}),
            "hook_late": None,
        }
        answer = answers.get(kind, (True, None, "", {}))
        self.checks.append(FakeCheck(lambda: answer))
        return self.checks[-1]


def start(tmp_path, behaviour, *, vast=None, policy=FAST, spec=SPEC, expiry=EXPIRY, **policy_kw):
    """A Launch ready to run, plus its fake provider and state."""
    if policy_kw:
        policy = Policy(**{**policy.__dict__, **policy_kw})
    vast = vast or FakeVast(behaviour)
    meta = {"image": spec.image, "gpus": list(spec.gpus), "num_gpus": spec.num_gpus}
    state = State.begin(tmp_path, RUN_ID, expiry, None, meta)
    return Launch(v.VastProvider(vast), spec, policy, state, FakeProber(vast)), vast, state


def foreign_untouched(vast):
    return all(9000 + i in vast.live for i in range(len(FOREIGN)))
