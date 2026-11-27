"""Lium: GPU pods on independent hosts, through its REST API, as boxes.

Lium (lium.io) is a marketplace: providers list nodes (one machine's GPUs, with its CPUs,
memory and disk shared out in proportion to the GPUs rented), each at its own price per
GPU, and a pod is a container on a node. A box is such a pod, running the caller's image
under a command of ours (vmbox.CONTAINER_INIT: save the container's environment, then
wait), reached as root with SSH on the pod's published port 22.

What it maps to:
  offers     /executors: a node with enough free GPUs and ports, within the price cap,
             its share of CPUs, memory and disk large enough, its driver's CUDA new
             enough. A node that takes several boxes is offered once per slot of the
             race. Spot nodes (interruptible) are left out.
  rent       a one-time template of the image and our command (/templates; deleted
             with the pod), then /executors/{node}/rent with the run's key, the tag as
             the pod's name, and the tag as its `Idempotency-Key`: Lium answers a repeat
             with the first answer, so the rent is sent again after a lost answer, and
             once one was lost, nothing but a pod id settles it.
  look up    /pods/{id}; gone: 404 `pod_not_found`, or DELETING (billing stops at the
             delete request, and the pod cannot come back)
  list       /pods
  destroy    DELETE /pods/{id}
  relabel    /pods/{id}/schedule-removal, then the name, then read back

The sshd is Lium's: its validator installs one in the container (if the image has none)
and adds the keys of the rent, without Lium's agent key. So the image needs no sshd, a
key refused says nothing about the image (`own_sshd`), and a pod cannot delete itself:
there is no key confined to it. `run --destroy-after` is refused.

The lease is Lium's: a pod's removal time. The rent carries `termination_hours`, the
lease rounded up to whole hours, which Lium counts from when the pod starts running, and
which then replaces any removal time set before (seen live, 2026-11). So that is the
backstop a rent carries, and the exact time is set twice: right after the rent, which
covers a pod that never starts, and once it runs (`lease_on_start`: the launcher
relabels it before admitting it). `renew` moves the removal time.

A key whose pod visibility is `own` lists only the pods it rented, and nothing says which
a key is: tag-only cleanup (`down --run-id`, `reap`) needs a key that sees the account's
pods. The account is the user and its workspace.

Not yet: spot nodes (`--interruptible`), private images (a Docker credential per run),
the machine ids for the blocklist (a node's id is stable; its meaning across runs is not
yet observed).
"""

from __future__ import annotations

import datetime
import math
import re
import shlex
import time

from . import httpapi
from . import base as prov
from .base import gpu_matches
from .container import classify_status
from .vmbox import CONTAINER_INIT
from ..errors import ConfigError
from ..state import parse_tag

API_BASE = "https://lium.io/api"
STATUSES = (
    "PENDING", "RUNNING", "STOPPED", "FAILED", "DELETING", "STOP_PENDING", "START_PENDING",
    "CREATION_FAILED", "BROKEN", "REBOOT_PENDING", "REBOOT_FAILED",
)  # fmt: skip
GONE = ("DELETING",)  # no longer billed, and cannot come back
STARTING = ("PENDING", "START_PENDING", "REBOOT_PENDING")
# The image's own ENTRYPOINT would otherwise run our command as its arguments; Lium takes
# only a single executable here, and a start command only as `bash -c '...'`.
ENTRYPOINT = "/usr/bin/env"
# Refusals that do not mean nothing was rented: our own earlier request with this key may
# still be running, or have rented.
UNKNOWN = ("idempotency_in_progress", "idempotency_outcome_unknown", "own_rent_in_progress")
RENT_TRIES = 3
MAX_HOURS = 720  # the longest `termination_hours` Lium takes
PRICE_SLACK = 0.01  # Lium rents at up to this much per GPU-hour above the expected price
# Refusals that no other node would answer differently: the account, or the image.
ACCOUNT = ("unauthorized", "api_key_inactive", "insufficient_balance", "account_unfunded",
           "email_not_verified", "forbidden", "transfer_hold", "API_KEY_BUDGET_EXCEEDED")  # fmt: skip
IMAGE = ("image_reference_invalid", "image_not_found", "image_needs_registry_credentials",
         "image_registry_credentials_refused", "image_no_linux_amd64")  # fmt: skip
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# `ssh root@host -p 20010`, as Lium shows a pod reached directly.
_SSH = re.compile(
    r"ssh (?P<user>[a-z_][a-z0-9_-]{0,31})@(?P<host>[A-Za-z0-9][A-Za-z0-9.-]{0,252})"
    r"(?: -p (?P<port>[0-9]{1,5}))?"
)
KIB_PER_GB = 1024**2


class LiumError(prov.ProviderError):
    pass


def Client(key: str, base: str = API_BASE, timeout: float = 30):
    """Lium's REST API (httpapi.Client): its errors are LiumError."""
    return httpapi.Client(base, {"X-API-Key": key}, LiumError, timeout)


def rfc3339(t: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def gone(exc: LiumError) -> bool:
    """Is this Lium saying there is no such pod? A 404 from anything else (a gateway, a
    route that moved) is not."""
    return exc.status == 404 and exc.code == "pod_not_found"


def same_time(shown, t: int) -> bool:
    """Does a time Lium shows name this instant? Lium shows UTC without an offset."""
    if not isinstance(shown, str):
        return False
    try:
        when = datetime.datetime.fromisoformat(shown.strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    return abs(when.timestamp() - t) <= 1


def _number(value) -> float | None:
    if type(value) in (int, float) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        try:
            f = float(value)
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


class Lium:
    """Lium as a provider of boxes (base.py, providers/__init__.py)."""

    name = "lium"
    noun = "pod"
    credential = "LIUM_API_KEY"
    access = "ssh"  # Lium's sshd in the container, with the run's key
    spawn = "setsid"
    own_sshd = False
    self_destroy = False  # no key confined to the pod
    lease = "provider"  # the pod's removal time
    lease_on_start = True  # set again once the pod runs: Lium resets it then
    searches_lease = True  # a node's notices are read against the lease
    observes_machines = False
    default_spare = 1
    renewable = True
    requires = frozenset({"image", "price_cap"})
    features = frozenset({"image", "price_cap", "exclude_country", "ssh_config", "any_gpu"})
    searchable = True

    def __init__(self, client):
        self.client = client

    @classmethod
    def from_options(cls, options: dict, key: str) -> Lium:
        return cls(Client(key))

    def options(self) -> dict:
        return {}

    # ------------------------------------------------------------------ offers

    def candidates(self, spec, blocked: dict, policy) -> tuple[list[dict], str]:
        """Nodes that can take the box, cheapest first, within the price cap. A node is
        offered once per box it can hold, up to one per slot of the race, and every node
        once before any again: the spare boxes of a race are there for a bad machine."""
        data = self.client.call("GET", "executors")
        if not isinstance(data, list):
            raise LiumError(200, f"unexpected answer to the node listing: {str(data)[:200]}")
        horizon = time.time() + spec.ttl_min * 60
        found = [f for f in (_fit(x, spec, horizon) for x in data) if f is not None]
        found.sort(key=lambda f: (f["dph_total"], f["_node"]))
        offers = []
        for k in range(policy.max_live):
            for f in found:
                if k < f["_slots"]:
                    offers.append({**f, "id": f"{f['_node']}#{k}"})
        note = (
            f"{len(found)} node(s) for {spec.gpu_text} on Lium within "
            f"${spec.max_dph_per_gpu:g}/GPU/hr, CUDA {spec.min_cuda:g}+"
        )
        return offers, note

    # ------------------------------------------------------------------ boxes

    def create(self, offer: dict, spec, label: str, pubkey: str, expiry: int, avoid=()) -> str:
        """Rent a pod on the offer's node: a one-time template of the image, then the rent,
        sent again under the tag as its idempotency key while its answer is lost, then
        the pod's removal time."""
        node = offer["_node"]
        if node in avoid:
            raise prov.CreateRefused(0, f"node {node} failed earlier in this launch")
        hours = math.ceil((expiry - time.time()) / 3600)
        if hours > MAX_HOURS:
            raise ConfigError(f"Lium ends a pod within {MAX_HOURS} hours: lower --ttl-min")
        template = self._template(spec, label)
        body = {
            "pod_name": label,
            "template_id": template,
            "gpu_count": spec.num_gpus,
            "user_public_key": [pubkey],
            "skip_add_agent_ssh_key": True,  # not Lium's agent key
            "initial_port_count": 1,
            "enable_jupyter": False,
            "enable_volume_encryption": False,
            "enable_backups": False,
            # The backstop: whole hours from when the pod runs (see the module's docs).
            "termination_hours": max(1, hours),
            # Lium refuses the rent (409 price_changed) if the node now asks more than a
            # cent above this; the offers leave that cent within the cap.
            "expected_price_per_gpu_hour": offer["_price"],
        }
        data = self._rent(node, body, label, template, spec)
        pod = data.get("pod_id") if isinstance(data, dict) and data.get("success") is True else None
        if not (isinstance(pod, str) and _UUID.fullmatch(pod)):
            raise prov.CreateUnknown(
                200, f"rent on {offer['id']}: unusable answer {str(data)[:300]}"
            )
        try:
            row = self.instance(pod)
        except LiumError:
            return pod  # the launcher looks it up, and sets its lease once it runs
        if row is not None and row["label"] != label:
            # Not the pod we asked for, as far as we know: nothing is done to it.
            raise prov.CreateUnknown(200, f"rent answered pod {pod}, named {row['label']!r}")
        try:
            # Covers a pod that never starts; once it runs, Lium replaces it (lease_on_start).
            self._schedule(pod, expiry)
        except LiumError:
            pass  # the rent's own termination_hours bounds it
        return pod

    def _rent(self, node: str, body: dict, label: str, template: str, spec) -> dict:
        """The rent's answer. Sent again, the same body under the same key, while there is
        no answer or the request is still running; once an answer was lost, a refusal
        that follows does not show that nothing was rented."""
        lost = None
        for k in range(RENT_TRIES):
            try:
                return self.client.call(
                    "POST", f"executors/{node}/rent", body=body, retries=0,
                    headers={"Idempotency-Key": label},
                )  # fmt: skip
            except LiumError as exc:
                answered = 400 <= exc.status < 500 and exc.code not in UNKNOWN
                if not answered:
                    lost = exc
                again = not answered or exc.status == 429
                if again and k + 1 < RENT_TRIES:
                    time.sleep(2**k)
                    continue
                if lost is not None:
                    raise prov.CreateUnknown(
                        exc.status, f"rent not settled: {lost}; then {exc}"
                    ) from None
                self._drop_template(template)  # nothing was rented from it
                if exc.code in ACCOUNT or exc.status == 401:
                    raise ConfigError(f"Lium refused the rent for this account: {exc}") from None
                if exc.code in IMAGE:
                    raise ConfigError(
                        f"image {spec.image!r} cannot be used on Lium: {exc}"
                    ) from None
                raise prov.CreateRefused(exc.status, str(exc)) from None
        raise AssertionError("unreachable")

    def _template(self, spec, label: str) -> str:
        """A private template of the image and our command, deleted with its pod. Its
        refusal rents nothing; one that no other node would change stops the launch."""
        image = spec.image
        if not image or any(c.isspace() for c in image) or image.startswith("-"):
            raise ConfigError(f"not an image reference: {image!r}")
        body = {
            "name": label,
            "docker_image": image,
            "docker_image_tag": "",  # Lium splits the reference into name, tag and digest
            "docker_image_digest": "",
            "internal_ports": [22],
            "entrypoint": ENTRYPOINT,
            "startup_commands": "bash -c " + shlex.quote(CONTAINER_INIT),
            "category": "UBUNTU",
            "container_start_immediately": True,
            "description": "gpubox",
            "readme": "",
            "environment": {},
            "volumes": ["/root"],
            "is_private": True,
            "one_time_template": True,
        }
        try:
            data = self.client.call("POST", "templates", body=body, retries=0)
        except LiumError as exc:
            if exc.code in ACCOUNT or exc.status == 401:
                raise ConfigError(f"Lium refused the template for this account: {exc}") from None
            if exc.code in IMAGE or exc.code in ("bad_request", "validation_error"):
                raise ConfigError(f"Lium refused a template of {image!r}: {exc}") from None
            raise prov.CreateRefused(exc.status, f"template not created: {exc}") from None
        tid = data.get("id") if isinstance(data, dict) else None
        if not (isinstance(tid, str) and _UUID.fullmatch(tid)):
            raise prov.CreateRefused(200, f"template not created: {str(data)[:200]}")
        return tid

    def _drop_template(self, template: str) -> None:
        try:
            self.client.call("DELETE", f"templates/{template}", retries=1)
        except LiumError:
            pass  # a private template rents nothing

    def _schedule(self, pod: str, expiry: int) -> None:
        self.client.call(
            "POST", f"pods/{pod}/schedule-removal", body={"removal_scheduled_at": rfc3339(expiry)}
        )

    def instance(self, pod_id, *, timeout=None, retries=2) -> dict | None:
        """The pod, or None when Lium says it does not exist (404) or is being deleted."""
        try:
            data = self.client.call("GET", f"pods/{pod_id}", timeout=timeout, retries=retries)
        except LiumError as exc:
            if gone(exc):
                return None
            raise
        row = _row(data)
        if row is None or row["id"] != pod_id:
            raise LiumError(200, f"unexpected answer for pod {pod_id}: {str(data)[:200]}")
        return None if row["actual_status"] in GONE else row

    def instances(self, *, timeout=None, retries=2) -> list[dict]:
        """Every pod of the account that is not being deleted, or an error."""
        try:
            data = self.client.call("GET", "pods", timeout=timeout, retries=retries)
        except LiumError as exc:
            if exc.status == 402 and exc.code == "BALANCE_REQUIRED":
                return []  # said only to an account with no balance and no pods
            raise
        if not isinstance(data, list):
            raise LiumError(200, f"unexpected answer to the pod listing: {str(data)[:200]}")
        rows, seen = [], set()
        for raw in data:
            row = _row(raw)
            if row is None:
                raise LiumError(200, f"unreadable row in the pod listing: {str(raw)[:200]}")
            if row["id"] in seen:
                raise LiumError(200, f"pod {row['id']} is listed twice")
            seen.add(row["id"])
            if row["actual_status"] not in GONE:
                rows.append(row)
        return rows

    def destroy(self, pod_id, *, timeout=None, retries=2) -> None:
        """Ask for the pod's deletion. One that is already gone (404) is not an error."""
        try:
            self.client.call("DELETE", f"pods/{pod_id}", timeout=timeout, retries=retries)
        except LiumError as exc:
            if not gone(exc):
                raise

    def relabel(self, pod_id, label: str) -> None:
        """The removal time, then the name, then read back: the lease is believed moved
        only once the pod shows both. The time goes first, so that a rename that fails
        leaves the tag, which the reaper reads, with the earlier of the two."""
        tag = parse_tag(label)
        if tag is None:
            raise ValueError(f"not a tag: {label!r}")
        self._schedule(pod_id, tag.expiry)
        self.client.call("PATCH", f"pods/{pod_id}", body={"pod_name": label})
        row = self.instance(pod_id)
        if (
            row is None
            or row["label"] != label
            or not same_time(row.get("removal_scheduled_at"), tag.expiry)
        ):
            raise LiumError(200, f"update of pod {pod_id} not confirmed")

    def account(self) -> str:
        """The user and the workspace the key acts in: a key sees only its workspace's pods."""
        data = self.client.call("GET", "users/me")
        space = data.get("workspace") if isinstance(data, dict) else None
        space = space.get("id") if isinstance(space, dict) else None
        if isinstance(data.get("id") if isinstance(data, dict) else None, str) and data["id"]:
            if isinstance(space, str) and space:
                return f"{data['id']}/{space}"
        raise LiumError(200, f"unexpected answer to the account lookup: {str(data)[:200]}")

    def view(self, row: dict) -> prov.View:
        status = row["actual_status"]
        progress = (
            row.get("create_progress") if isinstance(row.get("create_progress"), dict) else {}
        )
        msg = str(progress.get("message") or row.get("phase") or "")
        if status in STARTING:
            return prov.View("loading", status, msg, classify_status(msg))
        if status != "RUNNING":
            return prov.View("exited", status, msg, classify_status(msg))
        cmd = row.get("ssh_connect_cmd")
        ep = endpoint_of(cmd)
        if ep is not None:
            return prov.View("running", status, msg, None, ep)
        if isinstance(cmd, str) and cmd.strip():
            # Through Lium's gateway (`ssh -J`), or as another user: not how gpubox gets in.
            return prov.View("exited", status, f"SSH only as {cmd[:120]!r}", "machine")
        return prov.View("loading", status, msg or "no SSH address yet")

    def expiry_command(self, expiry: int) -> str:
        return "true"  # no timer in the box: Lium keeps the lease (relabel)

    def manual_hint(self, run_id: str, ids) -> str:
        return f"Or delete them at https://lium.io/pods (their names start with {run_id})"

    def describe(self, a) -> str:
        return f"${a.dph:.2f}/hr, {a.gpu}, {a.geo}"


def _fit(x, spec, horizon: float) -> dict | None:
    """The offer a node makes for this spec (with `_slots`, the boxes it can hold), or
    None. A size or a time that is not stated does not meet a minimum."""
    if not isinstance(x, dict) or not isinstance(x.get("id"), str) or not _UUID.fullmatch(x["id"]):
        return None
    specs = x.get("specs") if isinstance(x.get("specs"), dict) else {}
    name = x.get("gpu_class") or x.get("machine_name")
    if not isinstance(name, str) or not name:
        return None
    if spec.gpus and not any(gpu_matches(m, name, x.get("machine_name")) for m in spec.gpus):
        return None
    n = spec.num_gpus
    price = _number(x.get("price_per_gpu"))
    # Lium may rent at up to a cent above the price offered (PRICE_SLACK): that cent too
    # is within the cap.
    if price is None or not 0 < price < 1e4:
        return None
    if (price + PRICE_SLACK) * n > spec.max_dph_per_gpu * n + 1e-9:
        return None
    if x.get("tier") == "spot" or specs.get("is_spot") is not False:
        return None  # interruptible
    total, free = x.get("gpu_count"), x.get("available_gpu_count")
    if type(total) is not int or type(free) is not int or not 0 < n <= free <= total:
        return None
    # A node listed with no minimum rents only whole: all its free GPUs at once.
    least = x.get("min_gpu_count_for_ssh_only_rental", x.get("min_gpu_count_for_rental"))
    if least is None:
        if n != free:
            return None
        slots = 1
    elif type(least) is not int or n < least:
        return None
    else:
        slots = free // n
    ports = x.get("free_port_count")
    if type(ports) is not int or ports < 1:
        return None
    slots = min(slots, ports)  # each box takes one (its SSH port)
    if x.get("frequent_rent_failures") is not False or x.get("disk_health_ok") is False:
        return None
    if x.get("is_slow_machine") is True:
        return None
    # A removal or a maintenance notice: only if it starts after the lease ends. A notice
    # whose start cannot be read is taken to start now.
    notice = x.get("scheduled_notice")
    if notice is not None:
        notice = notice.get("starting_at", "") if isinstance(notice, dict) else ""
    for when in (x.get("removal_at"), notice):
        if when is not None and not _after(when, horizon):
            return None
    reliability = _number(x.get("reliability_score"))  # 0-100; none yet on a new node
    if spec.min_reliability and (reliability is None or reliability < spec.min_reliability * 100):
        return None
    down = _number(x.get("effective_download_speed_mbps"))
    if down is None or down < spec.min_inet_down_mbps:
        return None
    cuda = _number(x.get("max_cuda_version"))
    if cuda is None or (spec.min_cuda and cuda < spec.min_cuda):
        return None
    loc = x.get("location") if isinstance(x.get("location"), dict) else {}
    country = loc.get("country_code") if isinstance(loc.get("country_code"), str) else ""
    if spec.exclude_countries and (
        not country or country.upper() in {c.upper() for c in spec.exclude_countries}
    ):
        return None
    # CPUs, memory (less what the node keeps back) and disk go with the share of GPUs.
    share = n / total
    cpu, ram, disk = specs.get("cpu"), specs.get("ram"), specs.get("hard_disk")
    cpus = _number((cpu or {}).get("count")) if isinstance(cpu, dict) else None
    mem = _number((ram or {}).get("total")) if isinstance(ram, dict) else None
    keep = _number((ram or {}).get("pod_reserve")) if isinstance(ram, dict) else 0.0
    room = _number((disk or {}).get("free")) if isinstance(disk, dict) else None
    if cpus is None or mem is None or room is None:
        return None
    cpus, ram_gb, disk_gb = (
        cpus * share,
        (mem - (keep or 0)) * share / KIB_PER_GB,
        room * share / KIB_PER_GB,
    )
    if cpus < spec.box_cpus or ram_gb < spec.box_ram_gb or disk_gb < spec.disk_gb:
        return None
    gpu = specs.get("gpu") if isinstance(specs.get("gpu"), dict) else {}
    details = gpu.get("details") if isinstance(gpu.get("details"), list) else []
    vram = [_number(d.get("capacity")) for d in details if isinstance(d, dict)]
    if spec.min_vram_mib and (not vram or any(v is None or v < spec.min_vram_mib for v in vram)):
        return None
    city = loc.get("city") if isinstance(loc.get("city"), str) else ""
    return {
        # Two nodes of one provider are one witness against an image (launch.py).
        "machine_id": x["id"], "host_id": _host(x),
        "dph_total": round(price * n, 6), "geolocation": ", ".join(w for w in (city, country) if w),
        "gpu_name": name, "cuda_max_good": cuda,
        "cpu_cores_effective": cpus, "cpu_ram": ram_gb * 1024,
        "_node": x["id"], "_price": price, "_slots": slots,
    }  # fmt: skip


def _after(when, horizon: float) -> bool:
    """Is this time (Lium's, UTC when it states no offset) after the horizon? Not if it
    cannot be read."""
    try:
        t = datetime.datetime.fromisoformat(str(when).strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    if t.tzinfo is None:
        t = t.replace(tzinfo=datetime.timezone.utc)
    return t.timestamp() > horizon


def _host(x: dict) -> str | None:
    """The provider behind a node, as its listing names it."""
    for key in ("provider_key", "miner_hotkey"):
        if isinstance(x.get(key), str) and x[key]:
            return x[key]
    return None


def _row(raw) -> dict | None:
    """A pod as the launcher reads rows: with `label` (its name) and `actual_status`, or
    None if it is not a pod we can read."""
    if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
        return None
    if not _UUID.fullmatch(raw["id"]):
        return None
    name, status = raw.get("pod_name", ...), raw.get("status")
    if name is ... or not (name is None or isinstance(name, str)) or status not in STATUSES:
        return None
    return {**raw, "label": name or None, "actual_status": status}


def endpoint_of(cmd) -> tuple[str, int] | None:
    """(host, port) from `ssh root@host -p port`, the command Lium shows for a pod reached
    directly; None for anything else (a gateway, another user, extra options)."""
    if not isinstance(cmd, str):
        return None
    m = _SSH.fullmatch(cmd.strip())
    if m is None or m["user"] != "root":
        return None
    port = int(m["port"] or 22)
    if not 0 < port < 65536:
        return None
    return m["host"], port
