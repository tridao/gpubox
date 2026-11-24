"""Lyceum: its GPU VMs, rented directly, as boxes.

A box is the caller's image run as a container on the VM (vmbox.py). Lyceum installs a
public key of the caller's at the create (the run's), and runs nothing of ours at boot:
so the first probe that gets in sends the box's startup script and starts it
(vmbox.VmProber.boot).

What it maps to:
  offers     /vms/availability: hardware profiles ("h100") with a price per GPU an hour
             and the GPU counts that can be had now; the region is Lyceum's choice
  rent       POST /vms/create, sent once: the run's public key, our tag as the display
             name
  look up    /vms/{id}/status; gone: 404, or terminated
  list       /vms/list, asking for VMs in every state but terminated
  destroy    DELETE /vms/{id}
  relabel    PATCH /vms/{id} (the display name), then read back

Lyceum keeps no expiry and a VM cannot delete itself: only `down` or a scheduled `reap`
ends a VM left behind. It names no machine. Its disk is fixed (500 GB, as seen); a
`--disk-gb` above that is not offered. Not yet: spot VMs, private images.
"""

from __future__ import annotations

import re

from . import httpapi
from . import base as prov
from ..errors import ConfigError
from .base import gpu_matches

API_BASE = "https://api.lyceum.technology/api/v2/external"
GONE = ("terminated", "deleted")
DISK_GB = 500
USER = "lyceum"  # its docs: `ssh lyceum@IP`; root refuses the key (seen)
BOOT_S = 300  # "typically 1 to 3 minutes"
_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class LyceumError(prov.ProviderError):
    pass


def Client(key: str, base: str = API_BASE, timeout: float = 30):
    return httpapi.Client(base, {"Authorization": f"Bearer {key}"}, LyceumError, timeout)


class Lyceum:
    """Lyceum as a provider of boxes (provider.py, providers.py)."""

    name = "lyceum"
    noun = "VM"
    credential = "LYCEUM_API_KEY"
    access = "vm"
    startup_script = False  # nothing of ours runs at boot: the first probe starts it
    spawn = "setsid"
    own_sshd = False
    self_destroy = False
    lease = "reaper"
    renewable = True
    observes_machines = False
    default_spare = 1
    requires = frozenset({"image", "price_cap"})
    features = frozenset({"image", "price_cap", "any_gpu"})
    searchable = True

    def __init__(self, client):
        self.client = client

    @classmethod
    def from_options(cls, options: dict, key: str) -> Lyceum:
        return cls(Client(key))

    def options(self) -> dict:
        return {}

    def candidates(self, spec, blocked: dict, policy) -> tuple[list[dict], str]:
        if spec.min_cpus is not None or spec.min_ram_gb is not None:
            raise ConfigError("Lyceum does not say how many CPUs or how much RAM its VMs have")
        credits = self.client.call("GET", "billing/credits")
        left = credits.get("available_credits") if isinstance(credits, dict) else None
        if type(left) in (int, float) and left <= 0:
            raise ConfigError("Lyceum has no credits left on this account")
        data = self.client.call("GET", "vms/availability")
        profiles = data.get("available_hardware_profiles") if isinstance(data, dict) else None
        if not isinstance(profiles, list):
            raise LyceumError(200, f"unexpected answer to the availability: {str(data)[:200]}")
        n, cap = spec.num_gpus, spec.max_dph_per_gpu * spec.num_gpus + 1e-9
        found = []
        if spec.disk_gb <= DISK_GB:
            for pr in profiles:
                if not isinstance(pr, dict) or not isinstance(pr.get("hardware_profile"), str):
                    continue
                name, gpu = (
                    pr["hardware_profile"],
                    str(pr.get("gpu_type") or pr["hardware_profile"]),
                )
                unit = pr.get("price_per_hour")
                if type(unit) not in (int, float) or not 0 < unit < 1e4 or unit * n > cap:
                    continue
                if n not in (pr.get("available_gpus_per_instance") or []):
                    continue
                if spec.gpus and not any(gpu_matches(m, gpu, name) for m in spec.gpus):
                    continue
                found.append((unit * n, name, gpu))
        found.sort()
        offers = []
        for k in range(policy.max_live):
            for price, name, gpu in found:
                offers.append({
                    "id": f"{name}#{k}", "machine_id": None, "host_id": f"lyceum/{name}",
                    "dph_total": price, "geolocation": "lyceum", "gpu_name": gpu,
                    "boot_s": BOOT_S, "_profile": name, "_gpu": gpu,
                })  # fmt: skip
        offers.sort(key=lambda o: o["dph_total"])
        note = (
            f"{len(found)} hardware profile(s) available for {spec.gpu_text} on Lyceum within "
            f"${spec.max_dph_per_gpu:g}/GPU/hr"
        )
        return offers, note

    def create(self, offer: dict, spec, label: str, pubkey: str, expiry: int, avoid=()) -> str:
        """Create a VM. Sent once. A 4xx is an explicit no, except a refused key; an id
        only from an answer that carries one; anything else may have created a VM."""
        body = {
            "user_public_key": pubkey,
            "hardware_profile": offer["_profile"],
            "instance_specs": {"gpu_type": offer["_gpu"], "gpu_count": spec.num_gpus},
            "display_name": label,
            "instance_type": "on-demand",
        }
        try:
            data = self.client.call("POST", "vms/create", body=body, retries=0)
        except LyceumError as exc:
            if exc.status in (401, 403):
                raise ConfigError(f"Lyceum refused the API key: {exc}") from None
            refused = 400 <= exc.status < 500 and exc.status != 408
            raise (prov.CreateRefused if refused else prov.CreateUnknown)(
                exc.status, str(exc)
            ) from None
        vid = data.get("vm_id") if isinstance(data, dict) else None
        if isinstance(data, dict) and data.get("display_name") not in (None, label):
            raise prov.CreateUnknown(200, f"create answered VM {vid} with another name")
        if isinstance(vid, str) and _ID.fullmatch(vid):
            return vid
        raise prov.CreateUnknown(200, f"create on {offer['id']}: unusable answer {str(data)[:300]}")

    def instance(self, vm_id, *, timeout=None, retries=2) -> dict | None:
        try:
            data = self.client.call("GET", f"vms/{vm_id}/status", timeout=timeout, retries=retries)
        except LyceumError as exc:
            if exc.status == 404:
                return None
            raise
        row = _row(data)
        if row is None or row["id"] != vm_id:
            raise LyceumError(200, f"unexpected answer for VM {vm_id}: {str(data)[:200]}")
        return None if row["actual_status"] in GONE else row

    def instances(self, *, timeout=None, retries=2) -> list[dict]:
        """Every VM of the account in any state but terminated (the listing leaves some
        states out unless asked), or an error."""
        query = {
            "include_provisioning": "true", "include_ready": "true",
            "include_failed": "true", "include_terminated": "false",
        }  # fmt: skip
        data = self.client.call("GET", "vms/list", query=query, timeout=timeout, retries=retries)
        raw = data.get("vms") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            raise LyceumError(200, f"unexpected answer to the VM listing: {str(data)[:200]}")
        rows, seen = [], set()
        for item in raw:
            row = _row(item)
            if row is None:
                raise LyceumError(200, f"unreadable row in the VM listing: {str(item)[:200]}")
            if row["id"] in seen:
                raise LyceumError(200, f"VM {row['id']} is listed twice")
            seen.add(row["id"])
            if row["actual_status"] not in GONE:
                rows.append(row)
        return rows

    def destroy(self, vm_id, *, timeout=None, retries=2) -> None:
        try:
            self.client.call("DELETE", f"vms/{vm_id}", timeout=timeout, retries=retries)
        except LyceumError as exc:
            if exc.status != 404:
                raise

    def relabel(self, vm_id, label: str) -> None:
        self.client.call("PATCH", f"vms/{vm_id}", body={"display_name": label})
        row = self.instance(vm_id)
        if row is None or row["label"] != label:
            raise LyceumError(200, f"rename of VM {vm_id} not confirmed")

    def account(self) -> str:
        raise ConfigError("Lyceum has no account lookup here: --account-id cannot be checked")

    def view(self, row: dict) -> prov.View:
        status = row["actual_status"]
        if status in GONE or status in ("stopped", "error", "failed"):
            return prov.View("exited", status, "")
        ep = endpoint_of(row.get("ip_address"))
        if status in ("running", "ready") and ep is not None:
            return prov.View("running", status, "", None, (*ep, USER))
        return prov.View("loading", status, "" if ep else "no address yet")

    def expiry_command(self, expiry: int) -> str:
        return "true"

    def manual_hint(self, run_id: str, ids) -> str:
        return f"Or terminate them at https://dashboard.lyceum.technology (their names start with {run_id})"

    def describe(self, a) -> str:
        return f"${a.dph:.2f}/hr, {a.gpu}"


def endpoint_of(address) -> tuple[str, int] | None:
    """`ip_address` is "host:port" (or a host alone, port 22)."""
    if not isinstance(address, str) or not address:
        return None
    host, sep, port = address.rpartition(":")
    if not sep:
        host, port = address, "22"
    if not re.fullmatch(r"[0-9a-zA-Z.-]+", host) or host.startswith("-") or not port.isdigit():
        return None
    return (host, int(port)) if 0 < int(port) < 65536 else None


def _row(raw) -> dict | None:
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("vm_id"), str)
        or not _ID.fullmatch(raw["vm_id"])
    ):
        return None
    status = raw.get("status")
    if "display_name" not in raw or not isinstance(status, str) or not status:
        return None
    name = raw["display_name"]
    if not (name is None or isinstance(name, str)):
        return None
    return {**raw, "id": raw["vm_id"], "label": name, "actual_status": status.lower()}
