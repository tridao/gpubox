"""Massed Compute: its VMs, rented directly, as boxes.

A box is the caller's image run as a container on the VM (vmbox.py). Massed is reached
through its MCP server (JSON-RPC tool calls over HTTP, answered as JSON or as server-sent
events), which is what its API keys are for.

What it maps to:
  offers     gpu_inventory_list: products ("gpu_1x_a6000") with an hourly price, specs,
             and the regions that have them in stock; booted with Massed's "Ubuntu Server
             22.04 w/ drivers" (NVIDIA drivers and docker)
  rent       instances_launch, sent once: our tag as the instance's name (any text), our
             startup script as its boot command
  look up    instances_get: an instance that is over is no longer in `runningInstances`
  list       instances_list
  destroy    instances_terminate

Massed has no way to rename an instance, so the tag, and with it the lease the reaper
reads, is fixed at the launch: `renew` is refused, and `up --ttl-min` is the lease. It
keeps no expiry and a VM cannot delete itself, so only `down` or a scheduled `reap` ends
a VM left behind. It names no machine and has no account lookup. Not yet: private
images, countries.
"""

from __future__ import annotations

import base64
import http.client
import json
import re
import time
import urllib.error
import urllib.request

from . import httpapi
from . import base as prov
from .. import gpus
from . import vmbox
from ..errors import ConfigError

URL = "https://vm.massedcompute.com/api/mcp"
# Massed's Cloudflare policy turns away urllib's own user agent.
AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/140.0 Safari/537.36"
)
IMAGE = "Ubuntu Server 22.04 w/ drivers"
GONE = ("terminated", "deleted")  # terminating is still under way
# The login user is the instance's `username` ("Ubuntu", capital U, on Massed's images:
# root and `ubuntu` are refused). A Linux user name; capitals allowed.
_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,31}")
BOOT_S = 600
# What a launch's tool error says when nothing was launched.
REFUSALS = ("capacity", "not available", "unavailable", "insufficient", "invalid", "not found",
            "no inventory", "out of stock", "billing")  # fmt: skip
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_COUNT = re.compile(r"gpu_(\d+)x_", re.I)


class MassedError(prov.ProviderError):
    pass


class Client:
    """Massed's MCP tools. A tool's answer is its `structuredContent`; a JSON-RPC error
    (the request was refused as it stood) is MassedError 400, a tool that reports an
    error 422, HTTP errors their status, no answer 0."""

    def __init__(self, key: str, url: str = URL, timeout: float = 60):
        self.key, self.url, self.timeout = key, url, timeout

    def tool(self, name: str, args: dict, *, retries: int = 2, timeout=None):
        timeout = self.timeout if timeout is None else max(0.5, min(self.timeout, timeout))
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": name, "arguments": args}}).encode()  # fmt: skip
        headers = {
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream", "User-Agent": AGENT,
        }  # fmt: skip
        for attempt in range(retries + 1):
            req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            try:
                with httpapi.OPENER.open(req, timeout=timeout) as resp:
                    answer = _parse(resp.read().decode("utf-8", errors="replace"))
                break
            except urllib.error.HTTPError as exc:
                if (exc.code == 429 or exc.code >= 500) and attempt < retries:
                    time.sleep(2**attempt)
                    continue
                raise MassedError(exc.code, f"{name}: HTTP {exc.code}") from None
            except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
                if attempt < retries:
                    time.sleep(2**attempt)
                    continue
                raise MassedError(0, f"{name}: {exc}") from None
        if not isinstance(answer, dict) or answer.get("id") != 1:
            raise MassedError(0, f"{name}: unreadable answer")
        if isinstance(answer.get("error"), dict):
            # A request refused as it stood (invalid, unknown method or parameters) is a
            # definite no; any other error is not known to be one.
            code = answer["error"].get("code")
            status = 400 if code in (-32600, -32601, -32602) else 0
            raise MassedError(status, f"{name}: {answer['error'].get('message', 'error')}")
        result = answer.get("result")
        if not isinstance(result, dict):
            raise MassedError(0, f"{name}: no result in {str(answer)[:200]}")
        if result.get("isError"):
            text = " ".join(
                str(c.get("text", "")) for c in result.get("content") or [] if isinstance(c, dict)
            )
            raise MassedError(422, f"{name}: {' '.join(text.split())[:300]}")
        return result.get("structuredContent")


def _parse(raw: str):
    """A JSON-RPC answer, sent as JSON or as server-sent events (the last data line)."""
    if raw.lstrip().startswith("{"):
        return json.loads(raw)
    data = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:")]
    if not data:
        raise ValueError("no data in the event stream")
    return json.loads(data[-1])


class Massed:
    """Massed Compute as a provider of boxes (provider.py, providers.py)."""

    name = "massed"
    noun = "instance"
    credential = "MASSED_COMPUTE_API_KEY"
    access = "vm"
    spawn = "setsid"
    own_sshd = False
    self_destroy = False
    lease = "reaper"
    renewable = False  # no rename: the tag, and the reaper's lease, is fixed at the launch
    observes_machines = False
    default_spare = 1
    requires = frozenset({"image", "price_cap"})
    features = frozenset({"image", "price_cap", "any_gpu"})
    searchable = True
    searches_lease = False
    lease_slack = 0  # its lease ends a box at the tag's expiry

    def __init__(self, client):
        self.client = client

    @classmethod
    def from_options(cls, options: dict, key: str) -> Massed:
        return cls(Client(key))

    def options(self) -> dict:
        return {}

    # ------------------------------------------------------------------ offers

    def _image_id(self) -> int:
        data = self.client.tool("images_list", {})
        for img in (data or {}).get("images") or []:
            if (
                isinstance(img, dict)
                and img.get("vm_image_name") == IMAGE
                and type(img.get("vm_image_id")) is int
            ):
                return img["vm_image_id"]
        raise MassedError(200, f"Massed has no image {IMAGE!r} any more")

    def candidates(self, spec, blocked: dict, policy) -> tuple[list[dict], str]:
        data = self.client.tool("gpu_inventory_list", {})
        inventory = (data or {}).get("gpu_inventory")
        if not isinstance(inventory, dict):
            raise MassedError(200, f"unexpected answer to the inventory: {str(data)[:200]}")
        image = self._image_id()
        cap = spec.max_dph_per_gpu * spec.num_gpus + 1e-9
        found = []
        for product, entry in inventory.items():
            t = entry.get("instance_type") if isinstance(entry, dict) else None
            if not isinstance(t, dict):
                continue
            m = _COUNT.search(product)
            if m is None or int(m.group(1)) != spec.num_gpus:
                continue
            desc = str(t.get("description") or "")
            if spec.gpus and not any(gpus.matches(g, product, desc) for g in spec.gpus):
                continue
            cents = t.get("price_cents_per_hour")
            if type(cents) not in (int, float) or not 0 < cents < 1e7 or cents / 100 > cap:
                continue
            specs = t.get("specs") if isinstance(t.get("specs"), dict) else {}
            need = (
                ("vcpu_count", spec.box_cpus),
                ("memory_gib", spec.box_ram_gb),
                ("storage_gb", spec.disk_gb),
            )
            if any(type(specs.get(k)) not in (int, float) or specs[k] < least for k, least in need):
                continue
            for r in entry.get("regions_with_capacity_available") or []:
                if isinstance(r, dict) and isinstance(r.get("name"), str):
                    found.append((cents / 100, product, desc, r["name"]))
        found.sort()
        offers = []
        for k in range(policy.max_live):
            for price, product, desc, region in found:
                offers.append({
                    "id": f"{product}:{region}#{k}", "machine_id": None,
                    "failure_domain": f"massed/{region}", "dph_total": price, "geolocation": region,
                    "gpu_name": re.sub(r"^\d+x\s*", "", desc) or product, "boot_s": BOOT_S,
                    "_product": product, "_region": region, "_image": image,
                })  # fmt: skip
        offers.sort(key=lambda o: o["dph_total"])
        note = (
            f"{len(found)} product(s) and region(s) in stock for {spec.gpu_text} on Massed "
            f"Compute within ${spec.max_dph_per_gpu:g}/GPU/hr"
        )
        return offers, note

    # ------------------------------------------------------------------ boxes

    def create(self, offer: dict, spec, label: str, pubkey: str, expiry: int, avoid=()) -> str:
        """Launch a VM. Sent once. A refused request or a tool's error is an explicit no
        (out of stock, rejected), except a refused key; an id only from an answer that
        is one; anything else may have launched a VM."""
        script = vmbox.startup_script(pubkey, spec.image)
        command = f"echo {base64.b64encode(script.encode()).decode()} | base64 -d | bash"
        args = {
            "imageId": offer["_image"], "productName": offer["_product"],
            "regionName": offer["_region"], "instanceName": label, "command": command,
        }  # fmt: skip
        try:
            data = self.client.tool("instances_launch", args, retries=0)
        except MassedError as exc:
            if exc.status in (401, 403):
                raise ConfigError(f"Massed refused the API key: {exc}") from None
            # A tool's error is a refusal only when it says so; anything else may have
            # launched a VM.
            refused = exc.status == 400 or (
                exc.status == 422 and any(w in str(exc).lower() for w in REFUSALS)
            )
            raise (prov.CreateRefused if refused else prov.CreateUnknown)(
                exc.status, str(exc)
            ) from None
        uuid = (data or {}).get("response") if isinstance(data, dict) else None
        if isinstance(uuid, str) and _UUID.fullmatch(uuid):
            return uuid
        raise prov.CreateUnknown(200, f"launch on {offer['id']}: unusable answer {str(data)[:300]}")

    def instance(self, uuid, *, timeout=None, retries=2) -> dict | None:
        """The VM, or None when Massed no longer lists it as running (it is over)."""
        data = self.client.tool("instances_get", {"uuid": uuid}, timeout=timeout, retries=retries)
        rows = data.get("runningInstances") if isinstance(data, dict) else None
        if not isinstance(rows, list) or len(rows) > 1:
            raise MassedError(200, f"unexpected answer for instance {uuid}: {str(data)[:200]}")
        if not rows:
            return None
        row = _row(rows[0])
        if row is None or row["id"] != uuid:
            raise MassedError(200, f"unexpected answer for instance {uuid}: {str(data)[:200]}")
        return None if row["actual_status"].lower() in GONE else row

    def instances(self, *, timeout=None, retries=2) -> list[dict]:
        data = self.client.tool("instances_list", {}, timeout=timeout, retries=retries)
        raw = data.get("runningInstances") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            raise MassedError(200, f"unexpected answer to the instance listing: {str(data)[:200]}")
        rows, seen = [], set()
        for item in raw:
            row = _row(item)
            if row is None:
                raise MassedError(200, f"unreadable row in the instance listing: {str(item)[:200]}")
            if row["id"] in seen:
                raise MassedError(200, f"instance {row['id']} is listed twice")
            seen.add(row["id"])
            if row["actual_status"].lower() not in GONE:
                rows.append(row)
        return rows

    def destroy(self, uuid, *, timeout=None, retries=2) -> None:
        self.client.tool(
            "instances_terminate", {"instanceUuids": [uuid]}, timeout=timeout, retries=retries
        )

    def relabel(self, uuid, label: str) -> None:
        raise ConfigError("Massed Compute cannot rename an instance: its lease is fixed")

    def account(self) -> str:
        raise ConfigError("Massed Compute has no account lookup: --account-id cannot be checked")

    def view(self, row: dict) -> prov.View:
        status = row["actual_status"]
        low = status.lower()
        if low in GONE or low in ("error", "failed", "stopped"):
            return prov.View("exited", status, "")
        ip = row.get("ip")
        booted = row.get("active") in (1, True) and row.get("os_booted") in (1, True)
        user = row.get("username")
        if not (isinstance(user, str) and _USER.fullmatch(user)):
            return prov.View("loading", status, "no login user named")
        if (
            low == "running"
            and booted
            and isinstance(ip, str)
            and re.fullmatch(r"[0-9a-fA-F.:]+", ip)
        ):
            return prov.View("running", status, "", None, (ip, 22, user))
        return prov.View("loading", status, "" if low != "running" else "booting")

    def admit(self, instance_id, label: str) -> None:
        """Nothing to do before a box is handed over: its lease came with its create."""

    def expiry_command(self, expiry: int) -> str:
        return "true"

    def manual_hint(self, run_id: str, ids) -> str:
        return (
            f"Or terminate them at https://vm.massedcompute.com (their names start with {run_id})"
        )

    def describe(self, a) -> str:
        return f"${a.dph:.2f}/hr, {a.gpu}, {a.geo}"


def _row(raw) -> dict | None:
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("uuid"), str)
        or not _UUID.fullmatch(raw["uuid"])
    ):
        return None
    name, status = raw.get("name", ...), raw.get("status")
    if (
        name is ...
        or not (name is None or isinstance(name, str))
        or not isinstance(status, str)
        or not status
    ):
        return None
    return {**raw, "id": raw["uuid"], "label": name, "actual_status": status}
