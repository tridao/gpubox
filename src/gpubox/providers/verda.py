"""Verda (formerly DataCrunch): its VMs, rented directly, as boxes.

A box is the caller's image run as a container on the VM (vmbox.py), as on Shadeform: the
create's inline startup script, which Verda runs as root on first boot, adds the run's
key and starts the image; gpubox reaches it as `ubuntu` with SSH and `docker exec`.

What it maps to:
  offers     /instance-types (GPU, count, CPUs, RAM, a price and a spot price in each
             location) where /instance-availability says the type can be had, booted
             with Verda's image "+ Docker" of the newest CUDA the type supports. With
             `--interruptible`, spot VMs too, at their price (half), beside the others.
  rent       POST /instances, sent once, with our tag as a tag (key `gpubox`), an OS
             volume of `--disk-gb` that a spot VM's end deletes with it, and the startup
             script. Its answer is the bare id. 503: no capacity there
  look up    /instances/{id}; gone: 404, or `discontinued` (a deleted or evicted spot VM,
             which the listing no longer shows), or `notfound`
  list       /instances
  destroy    PUT /instances, action `delete`, with the OS volume deleted permanently
  relabel    the new tag added, then the old one removed, then read back

Credentials are an OAuth client: VERDA_CLIENT_ID and VERDA_CLIENT_SECRET give a token
for ten minutes, which the client renews. Verda requires an SSH key at the create; the
run's key goes in with the script, so the key registered is a placeholder whose private
half was deleted when it was made (as on Hyperstack). The account is the project the
token names.

Verda keeps no expiry and a VM cannot delete itself without the account's credentials:
only `down`, or a scheduled `reap`, ends a VM left behind (`up` says so). Prices are the
VM's: the OS volume is billed apart, so the price cap does not cover it. A spot VM may be
evicted at any time without warning; it is then discontinued, and gone. Not yet: private
images.
"""

from __future__ import annotations

import base64
import json
import re
import time

from . import httpapi
from . import base as prov
from .base import number
from .. import gpus
from . import vmbox
from .vmbox import placeholder_key
from ..errors import ConfigError
from ..state import parse_tag

API_BASE = "https://api.verda.com/v1"
USER = "ubuntu"  # root refuses keys; the startup script keys every login user
BOOT_S = 120  # seen: running 17 s after the create, SSH 50 s
TAG_KEY = "gpubox"
KEY_NAME = vmbox.PLACEHOLDER
MIN_OS_GB = 50  # Verda refuses 40 ("Specified storage size is too low"); 50 was taken
# discontinued: a spot VM deleted or evicted (it stays readable for minutes, unlisted).
GONE = ("discontinued", "notfound")
STARTING = ("ordered", "new", "validating", "provisioning", "unknown")
FAILED = ("error", "installation_failed", "no_capacity", "offline")
_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_IMAGE = re.compile(r"(?P<os>\d\d\.\d\d)\.cuda(?P<maj>\d+)\.(?P<min>\d+)\.docker")


class VerdaError(prov.ProviderError):
    pass


class Client:
    """Verda's API (httpapi.Client) with a token from the client credentials, renewed
    before it runs out, and once more if a call is refused as unauthorized."""

    def __init__(self, client_id: str, secret: str, base: str = API_BASE, timeout: float = 30):
        self._creds = {"client_id": client_id, "client_secret": secret}
        self._api = httpapi.Client(base, {}, VerdaError, timeout)
        self._auth = httpapi.Client(base, {}, VerdaError, timeout)
        self._until = 0.0

    def _token(self) -> str:
        if time.monotonic() >= self._until:
            data = self._auth.call(
                "POST", "oauth2/token", body={"grant_type": "client_credentials", **self._creds}
            )
            token = data.get("access_token") if isinstance(data, dict) else None
            life = data.get("expires_in") if isinstance(data, dict) else None
            if not isinstance(token, str) or not token or type(life) not in (int, float):
                raise VerdaError(200, "unexpected answer to the token request")
            self._api.headers["Authorization"] = f"Bearer {token}"
            self._until = time.monotonic() + max(30.0, life - 60)
        return self._api.headers["Authorization"][len("Bearer ") :]

    def call(self, method, path, **kw):
        try:
            self._token()
        except VerdaError as exc:
            if exc.status in (400, 401, 403):
                raise ConfigError(f"Verda refused the client credentials: {exc}") from None
            raise
        try:
            return self._api.call(method, path, **kw)
        except VerdaError as exc:
            if exc.status != 401:
                raise
            self._until = 0.0  # refused before anything was done: once more, renewed
            self._token()
            return self._api.call(method, path, **kw)

    def project(self) -> str:
        """The project the token is for, as it says (its claims are ours, not checked)."""
        token = self._token()
        try:
            part = token.split(".")[1]
            claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
        except (IndexError, ValueError):
            claims = None
        project = claims.get("project_id") if isinstance(claims, dict) else None
        if isinstance(project, str) and project:
            return project
        raise VerdaError(200, "the token names no project")


def best_image(supported, advertised: set[str]) -> tuple[str, float] | None:
    """The image to boot: one of Verda's "+ Docker", advertised and supported by the type,
    with the newest CUDA; at a tie, the older Ubuntu, which is the one seen to work."""
    found = []
    for name in supported or []:
        m = _IMAGE.fullmatch(name) if isinstance(name, str) else None
        if m and name in advertised:
            cuda = float(f"{m['maj']}.{m['min']}")
            found.append((cuda, -float(m["os"]), name))
    if not found:
        return None
    cuda, _, name = max(found)
    return name, cuda


def _size(t: dict, field: str, key: str) -> float | None:
    v = t.get(field)
    return number(v.get(key)) if isinstance(v, dict) else None


def _vram(t: dict, n: int) -> float | None:
    """GB of each GPU: a type states the memory of all its GPUs together."""
    total = _size(t, "gpu_memory", "size_in_gigabytes")
    return total / n if total else None


def short_name(label: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", label.lower())[:60].strip("-") or "gpubox"


class Verda:
    """Verda as a provider of boxes (base.py, providers/__init__.py)."""

    name = "verda"
    noun = "instance"
    credential = ("VERDA_CLIENT_ID", "VERDA_CLIENT_SECRET")
    access = "vm"
    spawn = "setsid"
    own_sshd = False
    self_destroy = False
    lease = "reaper"
    observes_machines = False
    default_spare = 1
    renewable = True  # its lease can be moved (relabel)
    requires = frozenset({"image", "price_cap"})
    features = frozenset({"image", "price_cap", "interruptible", "exclude_country", "any_gpu"})
    searchable = True
    searches_lease = False
    lease_slack = 0  # its lease ends a box at the tag's expiry

    def __init__(self, client):
        self.client = client

    @classmethod
    def from_options(cls, options: dict, key) -> Verda:
        client_id, secret = key
        return cls(Client(client_id, secret))

    def options(self) -> dict:
        return {}

    # ------------------------------------------------------------------ offers

    def _available(self, spot: bool) -> set[tuple[str, str]]:
        data = self.client.call(
            "GET", "instance-availability", query={"is_spot": "true" if spot else "false"}
        )
        if not isinstance(data, list):
            raise VerdaError(200, f"unexpected answer to the availability: {str(data)[:200]}")
        return {
            (a["location_code"], t)
            for a in data
            if isinstance(a, dict) and isinstance(a.get("location_code"), str)
            for t in a.get("availabilities") or []
            if isinstance(t, str)
        }

    def candidates(self, spec, blocked: dict, policy) -> tuple[list[dict], str]:
        """Instance types with the GPUs asked for, in each location that has them, within
        the price cap; with `--interruptible`, spot VMs too, cheapest first (a tie to the
        VM that is not spot). Each is offered once per slot of the race."""
        n = spec.num_gpus
        types = self.client.call("GET", "instance-types")
        images = self.client.call("GET", "images")
        locations = self.client.call("GET", "locations")
        if not all(isinstance(x, list) for x in (types, images, locations)):
            raise VerdaError(200, "unexpected answer to the instance type listing")
        advertised = {
            i["image_type"] for i in images
            if isinstance(i, dict) and isinstance(i.get("image_type"), str)
            and not i.get("is_cluster")
        }  # fmt: skip
        country = {
            x["code"]: x.get("country_code") for x in locations
            if isinstance(x, dict) and isinstance(x.get("code"), str)
        }  # fmt: skip
        tiers = [False, True] if spec.interruptible else [False]
        stock = {spot: self._available(spot) for spot in tiers}
        skip = {c.upper() for c in spec.exclude_countries}
        cap = spec.max_dph_per_gpu * n + 1e-9
        found = []
        for t in types:
            if not isinstance(t, dict) or not isinstance(t.get("instance_type"), str):
                continue
            gpu = t.get("gpu") if isinstance(t.get("gpu"), dict) else {}
            model = t.get("model") or t.get("name")
            if gpu.get("number_of_gpus") != n or not isinstance(model, str):
                continue
            if model.endswith(" CC"):
                continue  # confidential computing: images of its own
            vram = _vram(t, n)
            if spec.gpus and not any(
                gpus.matches(m, model, t.get("name"), gpu.get("description"), memory_gb=vram)
                for m in spec.gpus
            ):
                continue
            if not self._fits(spec, t):
                continue
            chosen = best_image(t.get("supported_os"), advertised)
            if chosen is None or (spec.min_cuda and chosen[1] < spec.min_cuda):
                continue
            for lp in t.get("location_prices") or []:
                loc = lp.get("location_code") if isinstance(lp, dict) else None
                cc = country.get(loc)
                if not isinstance(loc, str) or not isinstance(cc, str):
                    continue
                if spec.exclude_countries and cc.upper() in skip:
                    continue
                for spot in tiers:
                    price = number(lp.get("spot_price" if spot else "price_per_hour"))
                    if price is None or not 0 < price < 1e5 or price > cap:
                        continue
                    if (loc, t["instance_type"]) in stock[spot]:
                        found.append((price, spot, t["instance_type"], loc, model, chosen))
        found.sort()
        offers = []
        for k in range(policy.max_live):
            for price, spot, itype, loc, model, (image, cuda) in found:
                tier = "spot" if spot else "on-demand"
                offers.append({
                    "id": f"{itype}:{loc}:{tier}#{k}", "machine_id": None,
                    "failure_domain": f"verda/{loc}", "dph_total": price,
                    "geolocation": f"{loc}" + (" (spot)" if spot else ""),
                    "gpu_name": model, "cuda_max_good": cuda, "boot_s": BOOT_S,
                    "_type": itype, "_location": loc, "_image": image, "_spot": spot,
                })  # fmt: skip
        note = (
            f"{len(found)} instance type(s) and location(s) for {spec.gpu_text} on Verda "
            f"within ${spec.max_dph_per_gpu:g}/GPU/hr, CUDA {spec.min_cuda:g}+"
        )
        return offers, note

    @staticmethod
    def _fits(spec, t: dict) -> bool:
        cpus, ram = _size(t, "cpu", "number_of_cores"), _size(t, "memory", "size_in_gigabytes")
        if cpus is None or ram is None or cpus < spec.box_cpus or ram < spec.box_ram_gb:
            return False  # a size that is not stated does not meet a minimum
        if spec.min_vram_mib:
            vram = _vram(t, spec.num_gpus)
            return vram is not None and vram * 1024 >= spec.min_vram_mib
        return True

    # ------------------------------------------------------------------ boxes

    def _key_id(self) -> str:
        data = self.client.call("GET", "ssh-keys")
        if not isinstance(data, list):
            raise VerdaError(200, f"unexpected answer to the SSH key listing: {str(data)[:200]}")
        for k in data:
            if isinstance(k, dict) and k.get("name") == KEY_NAME and isinstance(k.get("id"), str):
                return k["id"]
        answer = self.client.call(
            "POST", "ssh-keys", body={"name": KEY_NAME, "key": placeholder_key()},
            retries=0, text=True,
        )  # fmt: skip
        kid = answer.strip().strip('"') if isinstance(answer, str) else None
        if not kid or not _ID.fullmatch(kid):
            raise VerdaError(200, f"unexpected answer to the SSH key import: {str(answer)[:200]}")
        return kid

    def create(self, offer: dict, spec, label: str, pubkey: str, expiry: int, avoid=()) -> str:
        """Rent a VM. Sent once. Before the create nothing is rented, so trouble getting
        the SSH key is a refusal; then 503 (no capacity) or another 4xx is an explicit no,
        except one about the account; an id only from an answer that is one."""
        try:
            key_id = self._key_id()
        except VerdaError as exc:
            raise prov.CreateRefused(exc.status, f"no SSH key to create with: {exc}") from None
        body = {
            "instance_type": offer["_type"],
            "image": offer["_image"],
            "ssh_key_ids": [key_id],
            "hostname": short_name(label),
            "description": label,
            "location_code": offer["_location"],
            "is_spot": bool(offer["_spot"]),
            "tags": [{"key": TAG_KEY, "value": label}],
            "os_volume": {"name": short_name(label), "size": max(spec.disk_gb, MIN_OS_GB)},
            "startup_script": vmbox.startup_script(pubkey, spec.image),
        }
        if offer["_spot"]:  # an evicted spot VM leaves nothing to bill behind (refused
            # for an on-demand VM, whose delete names the volume)
            body["os_volume"]["on_spot_discontinue"] = "delete_permanently"
        try:
            answer = self.client.call("POST", "instances", body=body, retries=0, text=True)
        except VerdaError as exc:
            if exc.code in ("unauthorized_request", "insufficient_funds", "forbidden_action"):
                raise ConfigError(f"Verda refused the create for this account: {exc}") from None
            refused = exc.status == 503 or (400 <= exc.status < 500 and exc.status != 408)
            raise (prov.CreateRefused if refused else prov.CreateUnknown)(
                exc.status, str(exc)
            ) from None
        iid = answer.strip().strip('"') if isinstance(answer, str) else ""
        if _ID.fullmatch(iid):
            return iid
        raise prov.CreateUnknown(200, f"create on {offer['id']}: unusable answer {answer[:300]!r}")

    def instance(self, instance_id, *, timeout=None, retries=2) -> dict | None:
        try:
            data = self.client.call(
                "GET", f"instances/{instance_id}", timeout=timeout, retries=retries
            )
        except VerdaError as exc:
            if exc.status == 404:
                return None
            raise
        row = _row(data)
        if row is None or row["id"] != instance_id:
            raise VerdaError(
                200, f"unexpected answer for instance {instance_id}: {str(data)[:200]}"
            )
        return None if row["actual_status"] in GONE else row

    def instances(self, *, timeout=None, retries=2) -> list[dict]:
        """Every instance of the project that is not gone, or an error."""
        data = self.client.call("GET", "instances", timeout=timeout, retries=retries)
        if not isinstance(data, list):
            raise VerdaError(200, f"unexpected answer to the instance listing: {str(data)[:200]}")
        rows, seen = [], set()
        for raw in data:
            row = _row(raw)
            if row is None:
                raise VerdaError(200, f"unreadable row in the instance listing: {str(raw)[:200]}")
            if row["id"] in seen:
                raise VerdaError(200, f"instance {row['id']} is listed twice")
            seen.add(row["id"])
            if row["actual_status"] not in GONE:
                rows.append(row)
        return rows

    def destroy(self, instance_id, *, timeout=None, retries=2) -> None:
        """Ask for the instance's deletion, its OS volume deleted permanently. One that is
        already gone is not an error; a refusal is."""
        try:
            row = self.instance(instance_id, timeout=timeout, retries=retries)
        except VerdaError:
            row = {}  # not known: deleted without naming its volume, which goes with it
        if row is None:
            return
        body = {"action": "delete", "id": instance_id}
        vol = row.get("os_volume_id")
        if isinstance(vol, str) and vol:
            body.update(volume_ids=[vol], delete_permanently=True)
        try:
            data = self.client.call("PUT", "instances", body=body, timeout=timeout, retries=retries)
        except VerdaError as exc:
            if exc.status == 404:
                return
            raise
        for r in data if isinstance(data, list) else []:  # None: already in that state
            if isinstance(r, dict) and r.get("instanceId") == instance_id:
                if r.get("status") != "success":
                    raise VerdaError(
                        r.get("statusCode") or 200,
                        f"delete of instance {instance_id} refused: {r.get('error')}",
                    )

    def relabel(self, instance_id, label: str) -> None:
        """The new tag, then the old one removed, then read back. In between the instance
        carries two of our tags, which makes its row unreadable (a listing fails) rather
        than untagged (a box the reaper cannot see)."""
        if parse_tag(label) is None:
            raise ValueError(f"not a tag: {label!r}")
        row = self.instance(instance_id)
        if row is None:
            raise VerdaError(404, f"instance {instance_id} is gone")
        if row["label"] != label:
            self.client.call(
                "POST", f"instances/{instance_id}/tags", body={"key": TAG_KEY, "value": label}
            )
        for t in row.get("tags") or []:
            if t.get("key") == TAG_KEY and t.get("value") != label and t.get("id"):
                self.client.call("DELETE", f"instances/{instance_id}/tags/{t['id']}")
        row = self.instance(instance_id)
        if row is None or row["label"] != label:
            raise VerdaError(200, f"retag of instance {instance_id} not confirmed")

    def account(self) -> str:
        return self.client.project()

    def view(self, row: dict) -> prov.View:
        status = row["actual_status"]
        if status in GONE:
            return prov.View("exited", status, "")
        if status == "running":
            ip = row.get("ip")
            if isinstance(ip, str) and re.fullmatch(r"[0-9a-fA-F.:]+", ip):
                return prov.View("running", status, "", None, (ip, 22, USER))
            return prov.View("loading", status, "no public IP yet")
        if status in FAILED:
            return prov.View("exited", status, "")
        return prov.View("loading", status, "")  # ordered, provisioning, deleting, ...

    def admit(self, instance_id, label: str) -> None:
        """Nothing to do before a box is handed over: its lease came with its create."""

    def expiry_command(self, expiry: int) -> str:
        return "true"  # no timer, and no expiry at Verda: the tag is the lease

    def manual_hint(self, run_id: str, ids) -> str:
        return f"Or delete them at https://console.verda.com (their tags start with {run_id})"

    def describe(self, a) -> str:
        return f"${a.dph:.2f}/hr, {a.gpu}, {a.geo}"


def _row(raw) -> dict | None:
    """An instance as the launcher reads rows: `label` (the value of its one tag of ours,
    or None) and `actual_status`; None if it cannot be read or carries two of our tags."""
    if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
        return None
    if not _ID.fullmatch(raw["id"]) or "tags" not in raw:  # not told: not the same as none
        return None
    status, tags = raw.get("status"), raw["tags"]
    if not isinstance(status, str) or not status or not (tags is None or isinstance(tags, list)):
        return None
    if not all(isinstance(t, dict) for t in tags or []):
        return None
    ours = [
        t.get("value") for t in tags or []
        if t.get("key") == TAG_KEY and parse_tag(t.get("value")) is not None
    ]  # fmt: skip
    if len(ours) > 1:
        return None
    return {**raw, "label": ours[0] if ours else None, "actual_status": status.lower()}
