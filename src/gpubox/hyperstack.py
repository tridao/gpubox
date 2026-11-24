"""Hyperstack: its VMs, rented directly, as boxes.

A box is the caller's image run as a container on the VM (vmbox.py), as on Shadeform:
the VM's cloud-init user data is our startup script, which adds the run's key and
starts the image; gpubox reaches it as `ubuntu` with SSH and `docker exec`.

What it maps to:
  offers     /core/flavors (GPU, count, CPUs, RAM, disk, stock) priced by /pricebook
             (per GPU an hour, without disk and IP), in a region that has an image of
             Hyperstack's "with Docker" whose CUDA is at least --min-cuda
  rent       POST /core/virtual-machines, sent once: a public IP with port 22 open and
             no port randomization, our tag in `labels`
  look up    /core/virtual-machines/{id}; gone: 404, or DELETING/DELETED
  list       /core/virtual-machines, by pages
  destroy    DELETE /core/virtual-machines/{id}
  relabel    PUT .../label, then read back

Hyperstack requires a key pair at the create. The run's key goes on the VM through the
startup script, so any key pair of the environment serves; if it has none, the run's
public key is imported as `gpubox` (a public key, harmless once the run's private key is
gone).

Hyperstack keeps no expiry and a VM cannot delete itself without the account's key: only
`down`, or a scheduled `reap`, ends a VM left behind (`up` says so). Not yet: spot
flavors, private images, countries.
"""

from __future__ import annotations

import base64
import re
import time

from . import httpapi
from . import provider as prov
from . import vmbox
from .errors import ConfigError
from .runpod import gpu_matches
from .state import parse_tag

API_BASE = "https://infrahub-api.nexgencloud.com/v1"
PAGE = 100
MAX_PAGES = 100
GONE = ("DELETING", "DELETED")
KEYPAIR = "gpubox"
USER = "ubuntu"
BOOT_S = 600  # Hyperstack gives no figure; VMs were seen up in 3 to 5 minutes
_CUDA = re.compile(r"CUDA (\d+)\.(\d+)")


class HyperstackError(prov.ProviderError):
    pass


def Client(key: str, base: str = API_BASE, timeout: float = 30):
    return httpapi.Client(base, {"api_key": key}, HyperstackError, timeout)


def image_cuda(name: str) -> float | None:
    m = _CUDA.search(name or "")
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


class Hyperstack:
    """Hyperstack as a provider of boxes (provider.py, providers.py)."""

    name = "hyperstack"
    noun = "VM"
    credential = "HYPERSTACK_API_KEY"
    access = "vm"
    spawn = "setsid"
    own_sshd = False
    self_destroy = False
    lease = "reaper"
    observes_machines = False
    default_spare = 1
    renewable = True  # its lease can be moved (relabel)
    requires = frozenset({"image", "price_cap"})
    features = frozenset({"image", "price_cap", "any_gpu"})
    searchable = True

    def __init__(self, client):
        self.client = client

    @classmethod
    def from_options(cls, options: dict, key: str) -> Hyperstack:
        return cls(Client(key))

    def options(self) -> dict:
        return {}

    def _ok(self, data, what: str) -> dict:
        if not isinstance(data, dict) or data.get("status") is not True:
            raise HyperstackError(200, f"unexpected answer to {what}: {str(data)[:200]}")
        return data

    # ------------------------------------------------------------------ offers

    def _environments(self) -> dict[str, str]:
        """region -> the environment to create in (its default one if there is one)."""
        data = self._ok(self.client.call("GET", "core/environments"), "the environment listing")
        out: dict[str, str] = {}
        for e in data.get("environments") or []:
            if (
                isinstance(e, dict)
                and isinstance(e.get("name"), str)
                and isinstance(e.get("region"), str)
            ):
                if e["region"] not in out or e["name"] == f"default-{e['region']}":
                    out[e["region"]] = e["name"]
        return out

    def _images(self) -> dict[str, tuple[str, float]]:
        """region -> Hyperstack's image with Docker whose CUDA is the newest."""
        data = self._ok(self.client.call("GET", "core/images"), "the image listing")
        best: dict[str, tuple[str, float]] = {}
        for group in data.get("images") or []:
            region = group.get("region_name") if isinstance(group, dict) else None
            for img in (group.get("images") or []) if isinstance(region, str) else []:
                name = img.get("name") if isinstance(img, dict) else None
                cuda = image_cuda(name) if isinstance(name, str) and "Docker" in name else None
                if cuda is not None and (region not in best or cuda > best[region][1]):
                    best[region] = (name, cuda)
        return best

    def candidates(self, spec, blocked: dict, policy) -> tuple[list[dict], str]:
        n = spec.num_gpus
        credit = self.client.call("GET", "billing/user-credit/credit")
        if (
            isinstance(credit, dict)
            and (credit.get("data") or {}).get("can_create_instance") is False
        ):
            raise ConfigError(
                "Hyperstack will not create VMs on this account now (not enough credit)"
            )
        flavors = self._ok(self.client.call("GET", "core/flavors"), "the flavor listing")
        book = self.client.call("GET", "pricebook")
        if not isinstance(book, list):
            raise HyperstackError(200, f"unexpected answer to the pricebook: {str(book)[:200]}")
        prices = {}
        for item in book:
            try:
                prices[str(item["name"]).lower()] = float(item["value"])
            except (KeyError, TypeError, ValueError):
                continue
        envs, images = self._environments(), self._images()
        cap = spec.max_dph_per_gpu * n + 1e-9
        found = []
        for group in flavors.get("data") or []:
            for f in (group.get("flavors") or []) if isinstance(group, dict) else []:
                if (
                    not isinstance(f, dict)
                    or f.get("gpu_count") != n
                    or f.get("stock_available") is not True
                ):
                    continue
                gpu, region = f.get("gpu"), f.get("region_name")
                if not isinstance(gpu, str) or not gpu or not isinstance(f.get("name"), str):
                    continue
                if "spot" in f["name"].lower() or "spot" in gpu.lower():
                    continue  # interruptible: not yet
                if spec.gpus and not any(gpu_matches(m, gpu, f["name"]) for m in spec.gpus):
                    continue
                unit = prices.get(gpu.lower())
                if unit is None or not 0 < unit < 1e4 or unit * n > cap:
                    continue
                if region not in envs or region not in images:
                    continue
                image, cuda = images[region]
                if spec.min_cuda and cuda < spec.min_cuda:
                    continue
                if not self._fits(spec, f):
                    continue
                found.append((unit * n, f["name"], gpu, region, image, cuda))
        found.sort()
        offers = []
        for k in range(policy.max_live):
            for price, flavor, gpu, region, image, cuda in found:
                offers.append({
                    "id": f"{flavor}:{region}#{k}", "machine_id": None,
                    "host_id": f"hyperstack/{region}", "dph_total": price,
                    "geolocation": region, "gpu_name": gpu, "cuda_max_good": cuda,
                    "boot_s": BOOT_S, "_flavor": flavor, "_region": region,
                    "_env": envs[region], "_image": image,
                })  # fmt: skip
        offers.sort(key=lambda o: o["dph_total"])
        note = (
            f"{len(found)} flavor(s) in stock for {spec.gpu_text} on Hyperstack within "
            f"${spec.max_dph_per_gpu:g}/GPU/hr, CUDA {spec.min_cuda:g}+"
        )
        return offers, note

    @staticmethod
    def _fits(spec, f: dict) -> bool:
        need = (("cpu", spec.box_cpus), ("ram", spec.box_ram_gb), ("disk", spec.disk_gb))
        for field, least in need:  # a size that is not stated does not meet a minimum
            have = f.get(field)
            if type(have) not in (int, float) or have < least:
                return False
        return True

    # ------------------------------------------------------------------ boxes

    def _keypair(self, env: str, pubkey: str) -> str:
        data = self._ok(self.client.call("GET", "core/keypairs"), "the key pair listing")
        for k in data.get("keypairs") or []:
            if isinstance(k, dict) and (k.get("environment") or {}).get("name") == env:
                return str(k["name"])
        body = {"name": KEYPAIR, "environment_name": env, "public_key": pubkey}
        self._ok(
            self.client.call("POST", "core/keypairs", body=body, retries=0), "the key pair import"
        )
        return KEYPAIR

    def create(self, offer: dict, spec, label: str, pubkey: str, expiry: int, avoid=()) -> int:
        """Rent a VM. Sent once. Before the create nothing is rented, so trouble getting a
        key pair is a refusal; then a 4xx is an explicit no (out of stock, rejected),
        except a refused key; an id only from an answer with exactly one VM."""
        try:
            key_name = self._keypair(offer["_env"], pubkey)
        except HyperstackError as exc:
            raise prov.CreateRefused(
                exc.status, f"no key pair for {offer['_env']}: {exc}"
            ) from None
        script = vmbox.startup_script(pubkey, spec.image)
        body = {
            "name": re.sub(r"[^A-Za-z0-9-]", "-", label)[:50],
            "environment_name": offer["_env"],
            "image_name": offer["_image"],
            "flavor_name": offer["_flavor"],
            "key_name": key_name,
            "count": 1,
            "assign_floating_ip": True,
            "enable_port_randomization": False,  # SSH on 22, where we look for it
            "user_data": base64.b64encode(script.encode()).decode(),
            "labels": [label],
            "security_rules": [{
                "direction": "ingress", "protocol": "tcp", "ethertype": "IPv4",
                "port_range_min": 22, "port_range_max": 22, "remote_ip_prefix": "0.0.0.0/0",
            }],
        }  # fmt: skip
        try:
            data = self.client.call("POST", "core/virtual-machines", body=body, retries=0)
        except HyperstackError as exc:
            if exc.status in (401, 403):
                raise ConfigError(f"Hyperstack refused the API key: {exc}") from None
            if exc.status == 405:  # "not allowed": the account, not the offer (credit)
                raise ConfigError(f"Hyperstack would not create the VM: {exc}") from None
            refused = 400 <= exc.status < 500 and exc.status != 408
            raise (prov.CreateRefused if refused else prov.CreateUnknown)(
                exc.status, str(exc)
            ) from None
        vms = (
            data.get("instances") if isinstance(data, dict) and data.get("status") is True else None
        )
        if isinstance(vms, list) and len(vms) == 1 and isinstance(vms[0], dict):
            vid = vms[0].get("id")
            if type(vid) is int and vid > 0:
                return vid
        raise prov.CreateUnknown(200, f"create on {offer['id']}: unusable answer {str(data)[:300]}")

    def instance(self, vm_id, *, timeout=None, retries=2) -> dict | None:
        try:
            data = self.client.call(
                "GET", f"core/virtual-machines/{int(vm_id)}", timeout=timeout, retries=retries
            )
        except HyperstackError as exc:
            if exc.status == 404:
                return None
            raise
        row = (
            _row(data.get("instance"))
            if isinstance(data, dict) and data.get("status") is True
            else None
        )
        if row is None or row["id"] != vm_id:
            raise HyperstackError(200, f"unexpected answer for VM {vm_id}: {str(data)[:200]}")
        return None if row["actual_status"] in GONE else row

    def instances(self, *, timeout=None, retries=2) -> list[dict]:
        """Every VM of the account that is not on its way out, or an error. Pages are
        read until one is short; one that does not say it is the page asked for, a row
        that cannot be read, or a VM listed twice fails the whole listing."""
        rows, seen = [], set()
        deadline = None if timeout is None else time.monotonic() + timeout
        for page in range(1, MAX_PAGES + 1):
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                raise HyperstackError(0, "VM listing did not finish in time")
            data = self.client.call(
                "GET", "core/virtual-machines", query={"page": page, "pageSize": PAGE},
                timeout=left, retries=retries,
            )  # fmt: skip
            batch = (
                data.get("instances")
                if isinstance(data, dict) and data.get("status") is True
                else None
            )
            if not isinstance(batch, list) or data.get("page") not in (page, None):
                raise HyperstackError(
                    200, f"unexpected answer to the VM listing: {str(data)[:200]}"
                )
            for raw in batch:
                row = _row(raw)
                if row is None:
                    raise HyperstackError(
                        200, f"unreadable row in the VM listing: {str(raw)[:200]}"
                    )
                if row["id"] in seen:
                    raise HyperstackError(200, f"VM {row['id']} is listed twice")
                seen.add(row["id"])
                if row["actual_status"] not in GONE:
                    rows.append(row)
            if len(batch) < PAGE:
                return rows
        raise HyperstackError(200, f"VM listing has more than {MAX_PAGES} pages")

    def destroy(self, vm_id, *, timeout=None, retries=2) -> None:
        try:
            self.client.call(
                "DELETE", f"core/virtual-machines/{int(vm_id)}", timeout=timeout, retries=retries
            )
        except HyperstackError as exc:
            if exc.status != 404:
                raise

    def relabel(self, vm_id, label: str) -> None:
        self.client.call(
            "PUT", f"core/virtual-machines/{int(vm_id)}/label", body={"labels": [label]}
        )
        row = self.instance(vm_id)
        if row is None or row["label"] != label:
            raise HyperstackError(200, f"relabel of VM {vm_id} not confirmed")

    def account(self) -> str:
        data = self._ok(self.client.call("GET", "auth/me"), "the account lookup")
        email = (data.get("user") or {}).get("email")
        if isinstance(email, str) and email:
            return email
        raise HyperstackError(200, f"unexpected answer to the account lookup: {str(data)[:200]}")

    def view(self, row: dict) -> prov.View:
        status, vm = row["actual_status"], str(row.get("vm_state") or "")
        msg = vm if vm.upper() != status else ""
        if status in GONE:
            return prov.View("exited", status, msg)
        if status == "ACTIVE":
            ip = row.get("floating_ip")
            if isinstance(ip, str) and re.fullmatch(r"[0-9a-fA-F.:]+", ip):
                return prov.View("running", status, msg, None, (ip, 22, USER))
            return prov.View("loading", status, msg or "no public IP yet")
        if status in ("ERROR", "SHUTOFF", "HIBERNATED", "STOPPED", "SUSPENDED"):
            return prov.View("exited", status, msg)
        return prov.View("loading", status, msg)  # CREATING, BUILD, ...

    def expiry_command(self, expiry: int) -> str:
        return "true"  # no timer, and no expiry at Hyperstack: the tag is the lease

    def manual_hint(self, run_id: str, ids) -> str:
        return (
            f"Or delete them at https://console.hyperstack.cloud (their labels start with {run_id})"
        )

    def describe(self, a) -> str:
        return f"${a.dph:.2f}/hr, {a.gpu}, {a.geo}"


def _row(raw) -> dict | None:
    """A VM as the launcher reads rows: `label` (the one label that is ours, or None),
    `actual_status`; None if it cannot be read or carries two of our tags."""
    if not isinstance(raw, dict) or type(raw.get("id")) is not int or raw["id"] <= 0:
        return None
    if "labels" not in raw:  # not told: not the same as no labels
        return None
    status, labels = raw.get("status"), raw["labels"]
    if (
        not isinstance(status, str)
        or not status
        or not (labels is None or isinstance(labels, list))
    ):
        return None
    if not all(isinstance(t, str) for t in labels or []):
        return None
    ours = [t for t in labels or [] if parse_tag(t) is not None]
    if len(ours) > 1:
        return None
    return {**raw, "label": ours[0] if ours else None, "actual_status": status.upper()}
