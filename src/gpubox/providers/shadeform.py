"""Shadeform: VMs on many GPU clouds through one API, as boxes.

Shadeform rents VMs on Hyperstack, Massed Compute, Lambda, Crusoe, Nebius and others,
with one API key and one bill. A box there is a VM that runs the caller's image as a
container (vmbox.py): the VM's startup script, which Shadeform runs as root once the VM
is active, puts the run's key on the VM and starts the image; gpubox reaches the box
with SSH to the VM (its `ssh_user`) and `docker exec`.

What it maps to:
  offers     /instances/types: an instance type in a region that has it, with an
             hourly price, an OS whose name says its CUDA ("ubuntu22.04_cuda13.0_
             shade_os"), and the boot time Shadeform expects (5 to 8 minutes), which
             the launcher adds to its start-up and stall limits
  rent       /instances/create, sent once, with our tag in `tags` (a name is at most
             50 letters, digits, `_` and `-`), and `auto_delete.date_threshold` set to
             the end of the lease: the lease is Shadeform's to keep
  look up    /instances/{id}/info; gone: 404, or `deleting` (no longer billed), or
             `deleted`
  list       /instances, by pages
  destroy    /instances/{id}/delete
  relabel    /instances/{id}/update: the tag and the auto-delete time together, then
             read back

A VM cannot delete itself (that would need the account's key on it), so there is no
`--destroy-after` here, and `renew` moves Shadeform's auto-delete time. Shadeform names
no machine, so machines are not recorded; and it has no account lookup, so
`--account-id` cannot be checked. Not yet: spot instances, private images (the
registry's token would sit in the startup script, which Shadeform keeps), countries.
"""

from __future__ import annotations

import base64
import datetime
import re
import time

from . import httpapi
from . import base as prov
from . import vmbox
from ..errors import ConfigError
from .base import gpu_matches
from ..state import parse_tag

API_BASE = "https://api.shadeform.ai/v1"
MAX_PAGES = 100
PAGE = 100
GONE = ("deleting", "deleted")  # deleting: no longer billed, and will not come back
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_OS_CUDA = re.compile(r"cuda(\d+)\.(\d+)")


class ShadeformError(prov.ProviderError):
    pass


def Client(key: str, base: str = API_BASE, timeout: float = 30):
    """Shadeform's REST API (httpapi.Client): its errors are ShadeformError."""
    return httpapi.Client(base, {"X-API-KEY": key}, ShadeformError, timeout)


def os_cuda(name: str) -> float | None:
    m = _OS_CUDA.search(name)
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def best_os(options) -> tuple[str, float] | None:
    """The OS to boot: one of Shadeform's own images (they have docker and NVIDIA's
    container toolkit), the one with the newest CUDA."""
    shade = [(os_cuda(o), o) for o in options or [] if isinstance(o, str) and "shade_os" in o]
    shade = [(c, o) for c, o in shade if c is not None]
    if not shade:
        return None
    cuda, name = max(shade)
    return name, cuda


def rfc3339(t: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def short_name(label: str) -> str:
    """An instance name for people, from the tag: what Shadeform accepts."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", label)[:50]


class Shadeform:
    """Shadeform as a provider of boxes (provider.py, providers.py)."""

    name = "shadeform"
    noun = "instance"
    credential = "SHADEFORM_API_KEY"
    access = "vm"  # SSH to the VM, then docker exec (vmbox.VmProber)
    spawn = "setsid"
    own_sshd = False
    self_destroy = False
    lease = "provider"  # auto_delete
    observes_machines = False
    default_spare = 1
    renewable = True  # its lease can be moved (relabel)
    requires = frozenset({"image", "price_cap"})
    features = frozenset({"image", "price_cap", "any_gpu"})
    searchable = True
    searches_lease = False
    lease_slack = 0  # its lease ends a box at the tag's expiry

    def __init__(self, client):
        self.client = client

    @classmethod
    def from_options(cls, options: dict, key: str) -> Shadeform:
        return cls(Client(key))

    def options(self) -> dict:
        return {}

    # ------------------------------------------------------------------ offers

    def candidates(self, spec, blocked: dict, policy) -> tuple[list[dict], str]:
        """Instance types with the GPUs asked for, in each region that has them, within
        the price cap, with an OS of Shadeform's whose CUDA is at least `spec.min_cuda`.
        Each is offered once per slot of the race: a region holds more than one VM."""
        data = self.client.call(
            "GET", "instances/types", query={"num_gpus": spec.num_gpus, "available": "true"}
        )
        types = data.get("instance_types") if isinstance(data, dict) else None
        if not isinstance(types, list):
            raise ShadeformError(200, f"unexpected answer to the type listing: {str(data)[:200]}")
        cap = spec.max_dph_per_gpu * spec.num_gpus + 1e-9
        found = []
        for t in types:
            if not isinstance(t, dict) or not isinstance(t.get("cloud"), str):
                continue
            c = t.get("configuration") if isinstance(t.get("configuration"), dict) else {}
            gpu = c.get("gpu_type") or t.get("gpu_type")
            if c.get("num_gpus", t.get("num_gpus")) != spec.num_gpus or not isinstance(gpu, str):
                continue
            if spec.gpus and not any(
                gpu_matches(m, gpu, t.get("shade_instance_type")) for m in spec.gpus
            ):
                continue
            cents = t.get("hourly_price")
            if type(cents) not in (int, float) or not 0 < cents < 1e7 or cents / 100 > cap:
                continue
            if not self._fits(spec, c):
                continue
            chosen = best_os(c.get("os_options"))
            if chosen is None or (spec.min_cuda and chosen[1] < spec.min_cuda):
                continue
            boot = (t.get("boot_time") or {}).get("max_boot_in_sec")
            for a in t.get("availability") or []:
                if not isinstance(a, dict) or a.get("available") is not True:
                    continue
                if a.get("rental_type", "on_demand") != "on_demand" or not a.get("region"):
                    continue
                found.append((cents / 100, t, gpu, a["region"], chosen, boot))
        found.sort(key=lambda f: (f[0], f[1]["cloud"], f[3]))
        offers = []
        for k in range(policy.max_live):
            for price, t, gpu, region, (os_name, cuda), boot in found:
                offers.append({
                    "id": f"{t['cloud']}:{t['shade_instance_type']}:{region}#{k}",
                    "machine_id": None, "failure_domain": f"{t['cloud']}/{region}",
                    "dph_total": price, "geolocation": f"{t['cloud']} {region}",
                    "gpu_name": gpu, "cuda_max_good": cuda,
                    "boot_s": boot if type(boot) is int and 0 < boot < 7200 else 600,
                    "_cloud": t["cloud"], "_type": t["shade_instance_type"],
                    "_region": region, "_os": os_name,
                })  # fmt: skip
        offers.sort(key=lambda o: o["dph_total"])
        note = (
            f"{len(found)} instance type(s) and region(s) for {spec.gpu_text} on Shadeform "
            f"within ${spec.max_dph_per_gpu:g}/GPU/hr, CUDA {spec.min_cuda:g}+"
        )
        return offers, note

    @staticmethod
    def _fits(spec, c: dict) -> bool:
        need = [
            ("vcpus", spec.box_cpus),
            ("memory_in_gb", spec.box_ram_gb),
            ("storage_in_gb", spec.disk_gb),
        ]
        for field, least in need:  # a size that is not stated does not meet a minimum
            have = c.get(field)
            if type(have) not in (int, float) or have < least:
                return False
        vram = c.get("vram_per_gpu_in_gb")
        return not (
            spec.min_vram_mib and type(vram) in (int, float) and vram * 1024 < spec.min_vram_mib
        )

    # ------------------------------------------------------------------ boxes

    def create(self, offer: dict, spec, label: str, pubkey: str, expiry: int, avoid=()) -> str:
        """Rent a VM. Sent once. A 4xx is an explicit no (out of stock, rejected), except
        a refused key; an id only from an answer that carries one; else CreateUnknown."""
        script = vmbox.startup_script(pubkey, spec.image)
        body = {
            "cloud": offer["_cloud"],
            "region": offer["_region"],
            "shade_instance_type": offer["_type"],
            "shade_cloud": True,
            "name": short_name(label),
            "os": offer["_os"],
            "launch_configuration": {
                "type": "script",
                "script_configuration": {
                    "base64_script": base64.b64encode(script.encode()).decode()
                },
            },
            "tags": [label],
            # The lease, kept by Shadeform: it deletes the VM then, whatever else happens.
            "auto_delete": {"date_threshold": rfc3339(expiry)},
        }
        try:
            data = self.client.call("POST", "instances/create", body=body, retries=0)
        except ShadeformError as exc:
            if exc.status in (401, 403):
                raise ConfigError(f"Shadeform refused the API key: {exc}") from None
            refused = 400 <= exc.status < 500 and exc.status != 408
            raise (prov.CreateRefused if refused else prov.CreateUnknown)(
                exc.status, str(exc)
            ) from None
        if isinstance(data, dict) and isinstance(data.get("id"), str) and _ID.fullmatch(data["id"]):
            return data["id"]
        raise prov.CreateUnknown(200, f"create on {offer['id']}: unusable answer {str(data)[:300]}")

    def instance(self, instance_id, *, timeout=None, retries=2) -> dict | None:
        try:
            data = self.client.call(
                "GET", f"instances/{instance_id}/info", timeout=timeout, retries=retries
            )
        except ShadeformError as exc:
            if exc.status == 404:
                return None
            raise
        row = _row(data)
        if row is None or row["id"] != instance_id:
            raise ShadeformError(
                200, f"unexpected answer for instance {instance_id}: {str(data)[:200]}"
            )
        return None if row["actual_status"] in GONE else row

    def instances(self, *, timeout=None, retries=2) -> list[dict]:
        """Every instance of the account that is not on its way out, or an error."""
        rows, seen, totals = [], set(), set()
        deadline = None if timeout is None else time.monotonic() + timeout
        for page in range(MAX_PAGES):
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                raise ShadeformError(0, "instance listing did not finish in time")
            query = {"limit": PAGE, "offset": page * PAGE}
            data = self.client.call("GET", "instances", query=query, timeout=left, retries=retries)
            batch = data.get("instances") if isinstance(data, dict) else None
            total = data.get("total") if isinstance(data, dict) else None
            if batch is None and isinstance(data, dict) and "instances" in data:
                batch = []  # Shadeform answers null for none
            if not isinstance(batch, list) or type(total) is not int:
                raise ShadeformError(
                    200, f"unexpected answer to the instance listing: {str(data)[:200]}"
                )
            totals.add(total)
            if len(totals) > 1:  # the list changed under us: not one listing
                raise ShadeformError(200, f"instance listing states {sorted(totals)} in all")
            for raw in batch:
                row = _row(raw)
                if row is None:
                    raise ShadeformError(
                        200, f"unreadable row in the instance listing: {str(raw)[:200]}"
                    )
                if row["id"] in seen:
                    raise ShadeformError(200, f"instance {row['id']} is listed twice")
                seen.add(row["id"])
                if row["actual_status"] not in GONE:
                    rows.append(row)
            if len(seen) >= total or not batch:
                if len(seen) != total:
                    raise ShadeformError(
                        200, f"instance listing has {len(seen)} rows but states {total}"
                    )
                return rows
        raise ShadeformError(200, f"instance listing has more than {MAX_PAGES} pages")

    def destroy(self, instance_id, *, timeout=None, retries=2) -> None:
        try:
            self.client.call(
                "POST", f"instances/{instance_id}/delete", timeout=timeout, retries=retries
            )
        except ShadeformError as exc:
            if exc.status != 404:
                raise

    def relabel(self, instance_id, label: str) -> None:
        """The tag and Shadeform's auto-delete time together, then read back: the lease
        is believed moved only once the instance shows it."""
        tag = parse_tag(label)
        if tag is None:
            raise ValueError(f"not a tag: {label!r}")
        when = rfc3339(tag.expiry)
        body = {"tags": [label], "auto_delete": {"date_threshold": when}}
        self.client.call("POST", f"instances/{instance_id}/update", body=body)
        row = self.instance(instance_id)
        shown = ((row or {}).get("auto_delete") or {}).get("date_threshold")
        if row is None or row["label"] != label or not _same_time(shown, tag.expiry):
            raise ShadeformError(200, f"update of instance {instance_id} not confirmed")

    def account(self) -> str:
        raise ConfigError("Shadeform has no account lookup: --account-id cannot be checked")

    def view(self, row: dict) -> prov.View:
        status = row["actual_status"]
        msg = str(row.get("status_details") or "")
        if status in GONE:
            return prov.View("exited", status, msg)
        if status == "active":
            ep = endpoint_of(row)
            if ep is not None:
                return prov.View("running", status, msg, None, ep)
            return prov.View("loading", status, msg or "no SSH address yet")
        if status in ("pending", "creating", "pending_provider"):
            return prov.View("loading", status, msg)
        return prov.View("exited", status, msg)  # an error state

    def admit(self, instance_id, label: str) -> None:
        """Nothing to do before a box is handed over: its lease came with its create."""

    def expiry_command(self, expiry: int) -> str:
        return "true"  # no timer in the box: Shadeform keeps the lease (relabel)

    def manual_hint(self, run_id: str, ids) -> str:
        return f"Or delete them at https://platform.shadeform.ai (their tags start with {run_id})"

    def describe(self, a) -> str:
        return f"${a.dph:.2f}/hr, {a.gpu}, {a.geo}"


def _same_time(shown, expiry: int) -> bool:
    """Does an RFC 3339 time, with whatever offset it states, name this instant?"""
    if not isinstance(shown, str):
        return False
    try:
        t = datetime.datetime.fromisoformat(shown.strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    if t.tzinfo is None:
        return False
    return abs(t.timestamp() - expiry) <= 1


def _row(raw) -> dict | None:
    """An instance as the launcher reads rows, with `label` (the one tag that is ours, or
    None) and `actual_status`; None if it cannot be read, or carries two of our tags."""
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("id"), str)
        or not _ID.fullmatch(raw["id"])
    ):
        return None
    if "tags" not in raw:  # not told: not the same as no tags
        return None
    status, tags = raw.get("status"), raw["tags"]
    if not isinstance(status, str) or not status or not (tags is None or isinstance(tags, list)):
        return None
    if not all(isinstance(t, str) for t in tags or []):
        return None
    ours = [t for t in tags or [] if parse_tag(t) is not None]
    if len(ours) > 1:
        return None
    return {**raw, "label": ours[0] if ours else None, "actual_status": status}


def endpoint_of(row: dict):
    """(host, port, user) of the VM's SSH, once Shadeform says all three."""
    host, port, user = row.get("ip"), row.get("ssh_port", 22), row.get("ssh_user")
    if (
        not isinstance(host, str)
        or not host
        or host.startswith("-")
        or any(c.isspace() for c in host)
    ):
        return None
    if not isinstance(user, str) or not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", user):
        return None
    if type(port) is not int or not 0 < port < 65536:
        return None
    return host, port, user
