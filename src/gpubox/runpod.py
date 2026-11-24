"""RunPod: pods through its REST API v2, as boxes.

A pod is a container, as on Vast, so a box is the same: our start-up command is the
pod's entrypoint (vast.bootstrap_script), with a lease timer first and then an sshd
that takes only the run's key, on the pod's published 22/tcp. RunPod gives every pod
`runpodctl` and a key confined to that pod (seen on a plain ubuntu image, 2026-11), so
the timer deletes the pod by itself, as Vast's does.

What differs from Vast:
  - No offers of machines: the catalog (/catalog/gpus) lists GPU types with a price
    and, per data center, how much is in stock. An offer here is a GPU type in a data
    center; RunPod picks the machine. One that is out of stock answers 400 ("no longer
    any instances available"), the explicit no that a stale Vast offer gives.
  - A pod says RUNNING before its container is up: it is running for us once it has
    an SSH address (`ssh.direct`).
  - A pod has no machine id in v2 and no status message, so machines are not recorded
    (observations, the store) and an image that cannot be pulled is not told apart
    from a slow pull: such a box is given up on by the stall and start-up rules.
  - Ids are strings, not numbers. A deleted pod is 404.
  - Not supported yet: interruptible (spot) pods, which v2 does not create; private
    images (a registry credential per run, to be deleted with it); countries to skip
    (v2 places by data center, not country).

Pricing is the catalog's list price per GPU, times the count; the price cap is applied
to it before anything is rented.
"""

from __future__ import annotations

import math
import re
import time

from . import httpapi
from . import provider as prov
from . import vast as v
from .errors import ConfigError

API_BASE = "https://api.runpod.io/v2"
MAX_PAGES = 50
# The pod deletes itself with the key RunPod gives it (RUNPOD_API_KEY, confined to the
# pod), through the runpodctl RunPod puts in every pod. The older command form: the newer
# `runpodctl pod remove` was refused (403) from inside a pod (runpodctl issue 293).
DESTROY = 'runpodctl remove pod "$RUNPOD_POD_ID"'
CLOUDS = ("SECURE", "COMMUNITY")
STATUSES = ("PROVISIONING", "STARTING", "RUNNING", "EXITED", "ERROR", "TERMINATED")
_POD_ID = re.compile(r"[a-z0-9]{6,32}")


class RunpodError(prov.ProviderError):
    pass


def Client(key: str, base: str = API_BASE, timeout: float = 20):
    """RunPod's REST API v2 (httpapi.Client): its errors are RunpodError."""
    return httpapi.Client(base, {"Authorization": f"Bearer {key}"}, RunpodError, timeout)


def _compact(text) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def gpu_matches(asked: str, *names) -> bool:
    """A model asked for, as a person writes it ("RTX 4090", "H100 SXM"), against
    RunPod's ids and names ("NVIDIA GeForce RTX 4090", "NVIDIA H100 80GB HBM3")."""
    needle = _compact(asked)
    return bool(needle) and any(needle in _compact(n) for n in names if n)


class RunPod:
    """RunPod as a provider of boxes (provider.py, providers.py)."""

    name = "runpod"
    noun = "pod"
    credential = "RUNPOD_API_KEY"
    access = "ssh"
    spawn = "setsid"
    own_sshd = True
    self_destroy = True  # with the pod's own key (DESTROY)
    lease = "box"
    observes_machines = False  # v2 names no machine
    default_spare = 1
    requires = frozenset({"image", "price_cap"})
    features = frozenset({"image", "price_cap", "ssh_config", "any_gpu"})
    searchable = True

    def __init__(self, client, *, cloud: str = "SECURE"):
        if cloud not in CLOUDS:
            raise ConfigError(f"RunPod's cloud is SECURE or COMMUNITY, not {cloud!r}")
        self.client, self.cloud = client, cloud

    @classmethod
    def from_options(cls, options: dict, key: str) -> RunPod:
        return cls(Client(key), cloud=str((options or {}).get("cloud") or "SECURE").upper())

    def options(self) -> dict:
        return {"cloud": self.cloud}

    # ------------------------------------------------------------------ offers

    def _placement(self, spec) -> dict:
        """Per-GPU host minimums, as the catalog and pod creation both take them."""
        n = spec.num_gpus
        return {
            "minVcpuCountPerGpu": math.ceil(spec.box_cpus / n),
            "minRamPerGpu": math.ceil(spec.box_ram_gb / n),
        }

    def candidates(self, spec, blocked: dict, policy) -> tuple[list[dict], str]:
        """GPU types in stock, in each data center that has them, cheapest first, within
        the price cap. Each is offered `policy.max_live` times: a data center can hold
        several boxes, and a try that finds it full costs nothing (it is refused)."""
        n = spec.num_gpus
        query = {"include": "AVAILABILITY", "product": "POD", "count": n, "cloud": self.cloud}
        if spec.min_cuda:
            query["minCudaVersion"] = f"{spec.min_cuda:.1f}"
        query.update(self._placement(spec))
        data = self.client.call("GET", "catalog/gpus", query=query)
        gpus = data.get("gpus") if isinstance(data, dict) else None
        if not isinstance(gpus, list):
            raise RunpodError(200, f"unexpected answer to the GPU catalog: {str(data)[:200]}")
        tier = self.cloud.lower()
        cap = spec.max_dph_per_gpu * n + 1e-9
        found = []
        for g in gpus:
            if not isinstance(g, dict) or not isinstance(g.get("id"), str):
                continue
            if spec.gpus and not any(gpu_matches(m, g["id"], g.get("name")) for m in spec.gpus):
                continue
            # A MIG slice is part of a GPU: only for whoever names it.
            if "MIG" in g["id"] and not any("mig" in m.lower() for m in spec.gpus):
                continue
            price = (g.get("price") or {}).get(tier)
            most = (g.get("maxCount") or {}).get(tier)
            if type(price) not in (int, float) or not 0 < price < 1e4:
                continue
            if type(most) is int and most < n:
                continue
            if g.get("availability") in (None, "NONE") or price * n > cap:
                continue
            places = [
                d["id"] for d in g.get("dataCenters") or []
                if isinstance(d, dict) and isinstance(d.get("id"), str)
                and d.get("availability") not in (None, "NONE")
            ]  # fmt: skip
            for dc in places or [None]:  # no data centers listed: let RunPod choose
                found.append((price * n, g["id"], dc))
        found.sort(key=lambda f: (f[0], f[1], f[2] or ""))
        offers = []
        for k in range(policy.max_live):
            for price, gid, dc in found:
                offers.append({
                    "id": f"{gid}@{dc or 'any'}#{k}", "machine_id": None, "host_id": None,
                    "dph_total": price, "geolocation": dc or self.cloud.lower(),
                    "gpu_name": gid, "_gpu": gid, "_dc": dc,
                })  # fmt: skip
        offers.sort(key=lambda o: o["dph_total"])  # stable: each place once before repeats
        note = (
            f"{len(found)} GPU type(s) and place(s) in stock for {spec.gpu_text} on RunPod's "
            f"{tier} cloud within ${spec.max_dph_per_gpu:g}/GPU/hr"
        )
        return offers, note

    # ------------------------------------------------------------------ boxes

    def create(self, offer: dict, spec, label: str, pubkey: str, expiry: int, avoid=()) -> str:
        """Create a pod. Sent once. An explicit no (4xx: out of stock, rejected) is
        CreateRefused, except a refused key, which is ours to fix; an id only from a 201
        that carries one; anything else may have created a pod (CreateUnknown)."""
        gpu = {"id": offer["_gpu"], "count": spec.num_gpus, **self._placement(spec)}
        if spec.min_cuda:
            gpu["minCudaVersion"] = f"{spec.min_cuda:.1f}"
        body = {
            "name": label,
            "cloud": self.cloud,
            "image": spec.image,
            "gpu": gpu,
            "disk": spec.disk_gb,
            # Our start-up command replaces the image's ENTRYPOINT, as on Vast.
            "entrypoint": ["bash"],
            "cmd": ["-c", v.bootstrap_script(pubkey, expiry, destroy=DESTROY)],
            "ports": ["22/tcp"],
            "env": {"RUNPOD_POD_USER": "0"},  # run as root, as dstack does
        }
        if offer.get("_dc"):
            body["dataCenterIds"] = [offer["_dc"]]
        try:
            data = self.client.call("POST", "pods", body=body, retries=0)
        except RunpodError as exc:
            if exc.status in (401, 403):
                raise ConfigError(f"RunPod refused the API key: {exc}") from None
            refused = 400 <= exc.status < 500 and exc.status != 408
            err = prov.CreateRefused if refused else prov.CreateUnknown
            raise err(exc.status, str(exc)) from None
        if (
            isinstance(data, dict)
            and isinstance(data.get("id"), str)
            and _POD_ID.fullmatch(data["id"])
        ):
            # The pod itself, with the name we gave it: nothing else is taken for its id.
            if data.get("name") != label:
                raise prov.CreateUnknown(201, f"create answered {data['id']} without our name")
            return data["id"]
        raise prov.CreateUnknown(200, f"create on {offer['id']}: unusable answer {str(data)[:300]}")

    def instance(self, pod_id, *, timeout=None, retries=2) -> dict | None:
        """The pod, or None when RunPod says it does not exist (404, or TERMINATED)."""
        try:
            data = self.client.call("GET", f"pods/{pod_id}", timeout=timeout, retries=retries)
        except RunpodError as exc:
            if exc.status == 404:
                return None
            raise
        row = _row(data)
        if row is None or row["id"] != pod_id:
            raise RunpodError(200, f"unexpected answer for pod {pod_id}: {str(data)[:200]}")
        return None if row["actual_status"] == "TERMINATED" else row

    def instances(self, *, timeout=None, retries=2) -> list[dict]:
        """Every pod of the account, or an error: never a partial list. A page or a row
        that cannot be read, a pod listed twice, or a cursor that does not move fails it.
        `timeout` is for the whole listing."""
        rows, seen, cursors = [], set(), set()
        deadline = None if timeout is None else time.monotonic() + timeout
        query = {"limit": 1000}
        for _ in range(MAX_PAGES):
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                raise RunpodError(0, "pod listing did not finish in time")
            data = self.client.call("GET", "pods", query=query, timeout=left, retries=retries)
            page = data.get("pods") if isinstance(data, dict) else None
            more = data.get("pagination") if isinstance(data, dict) else None
            if not isinstance(page, list) or not isinstance(more, dict):
                raise RunpodError(200, f"unexpected answer to the pod listing: {str(data)[:200]}")
            for raw in page:
                row = _row(raw)
                if row is None:
                    raise RunpodError(200, f"unreadable row in the pod listing: {str(raw)[:200]}")
                if row["id"] in seen:
                    raise RunpodError(200, f"pod {row['id']} is listed twice")
                seen.add(row["id"])
                if row["actual_status"] != "TERMINATED":
                    rows.append(row)
            if more.get("hasNextPage") is False:
                return rows
            cursor = more.get("nextCursor")
            if (
                more.get("hasNextPage") is not True
                or not isinstance(cursor, str)
                or not cursor
                or cursor in cursors
            ):
                raise RunpodError(200, f"pod listing does not advance: {str(more)[:200]}")
            cursors.add(cursor)
            query["cursor"] = cursor
        raise RunpodError(200, f"pod listing has more than {MAX_PAGES} pages")

    def destroy(self, pod_id, *, timeout=None, retries=2) -> None:
        """Ask for the pod's termination. One that is already gone (404) is not an error."""
        try:
            self.client.call("DELETE", f"pods/{pod_id}", timeout=timeout, retries=retries)
        except RunpodError as exc:
            if exc.status != 404:
                raise

    def relabel(self, pod_id, label: str) -> None:
        data = self.client.call("PATCH", f"pods/{pod_id}", body={"name": label})
        if not isinstance(data, dict) or data.get("id") != pod_id or data.get("name") != label:
            raise RunpodError(200, f"rename of pod {pod_id} not confirmed: {str(data)[:200]}")

    def account(self) -> str:
        data = self.client.call("GET", "account")
        if isinstance(data, dict) and isinstance(data.get("id"), str) and data["id"]:
            return data["id"]
        raise RunpodError(200, f"unexpected answer to the account lookup: {str(data)[:200]}")

    def view(self, row: dict) -> prov.View:
        status = row["actual_status"]
        if status in ("EXITED", "ERROR", "TERMINATED"):
            return prov.View("exited", status, "")
        ep = endpoint_of(row) if status == "RUNNING" else None
        if ep is not None:
            return prov.View("running", status, "", None, ep)
        # RUNNING before the container is up and has an address: still on its way.
        return prov.View("loading", status, "" if status != "RUNNING" else "no SSH address yet")

    def expiry_command(self, expiry: int) -> str:
        return v.set_expiry_command(expiry)

    def manual_hint(self, run_id: str, ids) -> str:
        return (
            f"Or terminate them at https://console.runpod.io/pods (their names start with {run_id})"
        )

    def describe(self, a) -> str:
        return f"${a.dph:.2f}/hr, {a.gpu}, {a.geo}"


def _row(raw) -> dict | None:
    """A pod as the launcher reads rows: the pod with `label` (its name) and
    `actual_status`, or None if it is not a pod we can read."""
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("id"), str)
        or not _POD_ID.fullmatch(raw["id"])
    ):
        return None
    name, status = raw.get("name", ...), raw.get("status")
    if name is ... or not (name is None or isinstance(name, str)) or status not in STATUSES:
        return None
    return {**raw, "label": name, "actual_status": status}


def endpoint_of(row: dict) -> tuple[str, int] | None:
    """Where the pod's sshd answers: `ssh.direct`, else the public mapping of port 22."""
    direct = (row.get("ssh") or {}).get("direct") if isinstance(row.get("ssh"), dict) else None
    candidates = []
    if isinstance(direct, dict):
        candidates.append((direct.get("host"), direct.get("port")))
    for p in (
        ((row.get("runtime") or {}).get("ports") or [])
        if isinstance(row.get("runtime"), dict)
        else []
    ):
        if isinstance(p, dict) and p.get("private") == 22:
            candidates.append((p.get("ip"), p.get("public")))
    for host, port in candidates:
        host = str(host or "").strip()
        usable = host and not host.startswith("-") and not any(c.isspace() for c in host)
        if usable and type(port) is int and 0 < port < 65536:
            return host, port
    return None
