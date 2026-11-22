"""A fake Vast and fake SSH for the launcher tests. No network, no GPU."""

import fcntl
import json
import threading
import time

from gpubox import vast as v
from gpubox.journal import Journal
from gpubox.launch import Launch, Policy, Spec

RUN_ID = "quack-ci.test.0123abcd"
FOREIGN = ("vast-5090-4x-ca", None, "quack-ci-old.x.12345678.a1", "quack-ci.other.89abcdef.a1")
FAST = Policy(
    poll=0.01,
    tick=0.01,
    startup_timeout=0.3,
    overall_timeout=5,
    cleanup_reserve=0,
    image_grace=2,
    reconcile_timeout=0.15,
    confirm_timeout=0.15,
    confirm_poll=0.01,
)
SPEC = Spec(image="img", gpu="RTX 5090", max_dph_per_gpu=1.25)


def offer(machine, host=None, dph=1.0):
    return {
        "id": 1000 + machine,
        "machine_id": machine,
        "host_id": host or machine,
        "dph_total": dph,
    }


class FakeVast:
    """`behaviour` maps machine_id to how a box rented on it behaves.

    ok        comes up and passes the gate
    bad_gpu   comes up, gate reports cuda_init
    busy      comes up, gate reports busy_gpu
    stuck     never leaves "loading"
    cdi       status message blames the machine's container runtime
    image     status message says the image cannot be pulled
    exited    container exits during bootstrap
    stale     create is refused (offer gone)
    ghost     create times out and no box exists
    lost      create times out but the box exists; listed after two listings
    lost_late as lost, but listed only once `reveal()` is called
    late      still in the gate when another candidate wins
    slow      passes the gate, but only after `slow_secs`
    hold      passes the gate once the test sets `release`
    api_down  rented, but the provider never answers a status query
    """

    def __init__(self, behaviour, offers=None, slow_secs=0.5):
        self.behaviour = behaviour
        self._offers = offers if offers is not None else [offer(m) for m in behaviour]
        self.live, self.destroyed, self._next = {}, [], 1
        self.creates = self.peak = 0
        self.fail_destroy = {}  # machine -> number of destroy calls that raise VastError first
        self.sticky = set()  # machines whose box survives a destroy
        self.listings_until_visible = {}
        self.revealed = False
        self.slow_secs = slow_secs
        self.release = threading.Event()  # lets "hold" boxes through the gate
        self.in_gate = {}  # machine -> Event, set when its gate starts
        self.list_fails = 0  # number of listing calls that raise first
        self.on_destroy = None  # called with the instance id before each destroy
        self.before_create = None  # called with the machine before the provider rents
        self.queries, self.payloads = [], []  # what was asked of the provider
        self.lock = threading.Lock()
        for i, label in enumerate(FOREIGN):
            self.live[9000 + i] = {"id": 9000 + i, "label": label, "machine": None}

    def ours(self):
        return sorted(i for i, b in self.live.items() if b["machine"] is not None)

    def gate_started(self, machine):
        with self.lock:
            return self.in_gate.setdefault(machine, threading.Event())

    def offers(self, query):
        self.queries.append(query)
        return list(self._offers)

    def create(self, offer_id, payload):
        machine = offer_id - 1000
        kind = self.behaviour[machine]
        if kind == "stale":
            raise v.CreateRefused(400, "offer no longer available")
        if kind == "ghost":
            raise v.CreateUnknown(0, "timed out")
        if self.before_create:
            self.before_create(machine)
        self.payloads.append(payload)
        with self.lock:
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
        if kind in ("lost", "lost_late"):
            self.listings_until_visible[iid] = 2
            raise v.CreateUnknown(0, "timed out")
        return iid

    def instance(self, iid):
        box = self.live.get(iid)
        if box is None:
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
        elif kind != "stuck":
            row.update(
                actual_status="running",
                public_ipaddr=f"10.0.0.{box['machine']}",
                ports={"22/tcp": [{"HostPort": "2222"}]},
            )
        return row

    def instances(self):
        rows = []
        if self.list_fails > 0:
            self.list_fails -= 1
            raise v.VastError(503, "unavailable")
        with self.lock:
            for iid, box in self.live.items():
                kind = self.behaviour.get(box["machine"])
                if kind == "lost_late" and not self.revealed:
                    continue
                if kind == "lost" and self.listings_until_visible.get(iid, 0) > 0:
                    self.listings_until_visible[iid] -= 1
                    continue
                row = {k: box.get(k) for k in ("id", "label", "start_date", "image_uuid")}
                rows.append({**row, "machine_id": box["machine"]})
        return rows

    def reveal(self):
        self.revealed = True

    def destroy(self, iid):
        if self.on_destroy:
            self.on_destroy(iid)
        with self.lock:
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

    def _with_file(self, fn, *args):
        with open(str(self.path) + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._load()
            try:
                return fn(*args)
            finally:
                self._save()

    def create(self, offer_id, payload):
        return self._with_file(super().create, offer_id, payload)

    def instance(self, iid):
        return self._with_file(super().instance, iid)

    def instances(self):
        return self._with_file(super().instances)

    def destroy(self, iid):
        return self._with_file(super().destroy, iid)

    def on_disk(self):
        """The ids of our boxes as another process would find them."""
        self._load()
        return self.ours()


class FakeProber:
    pubkey = "ssh-ed25519 AAAAtest gpubox"

    def __init__(self, vast):
        self.vast = vast
        self.wants = []  # the spec each gate was asked to check

    def ready(self, ep):
        return True, ""

    def gate(self, ep, spec):
        machine = int(ep[0].rsplit(".", 1)[1])
        kind = self.vast.behaviour[machine]
        self.wants.append(spec)
        self.vast.gate_started(machine).set()
        if kind == "hold" and not self.vast.release.wait(5):
            return False, None, "the test never released this gate", {}
        if kind == "bad_gpu":
            return False, "cuda_init", "CUDA unavailable", {}
        if kind == "busy":
            return False, "busy_gpu", "util 100%", {}
        if kind == "late":
            # Still in the gate when another candidate wins: the box is
            # destroyed under it and the SSH call fails.
            while any(b["machine"] == machine for b in list(self.vast.live.values())):
                time.sleep(0.01)
            return False, "unreachable", "ssh dropped during gate", {}
        if kind == "slow":
            time.sleep(self.vast.slow_secs)
        return True, None, "", {"gpus": ["RTX 5090"], "cpus": 32, "driver": "1", "torch": "2"}


def start(tmp_path, behaviour, *, vast=None, policy=FAST, spec=SPEC, **policy_kw):
    """A Launch ready to run, plus its fake provider and journal."""
    if policy_kw:
        policy = Policy(**{**policy.__dict__, **policy_kw})
    vast = vast or FakeVast(behaviour)
    journal = Journal(tmp_path)
    journal.append("run", run_id=RUN_ID, namespace="quack-ci", image=spec.image, gpu=spec.gpu)
    return Launch(vast, spec, policy, journal, FakeProber(vast), RUN_ID), vast, journal


def foreign_untouched(vast):
    return all(9000 + i in vast.live for i in range(len(FOREIGN)))


def statuses(journal):
    return {a.seq: a.status for a in journal.state().attempts.values()}
