"""Vast.ai: the HTTP client and everything that is specific to that provider.

Standard library only. The official `vastai` SDK was considered and not used:
its high-level calls do not send the payloads below, and it retries the create
request (see the design notes). Read its source when the API changes.
"""

from __future__ import annotations

import http.client
import json
import math
import shlex
import time
import urllib.error
import urllib.parse
import urllib.request

from .. import policy as pol
from ..onbox import set_expiry_command
from . import base as prov
from . import container
from .container import classify_status

API_BASE = "https://console.vast.ai/api"
MAX_PAGES = 400  # 10,000 instances; a listing longer than this is a loop


class VastError(prov.ProviderError):
    pass


class CreateRefused(VastError, prov.CreateRefused):
    """The provider answered and did not rent a box (stale offer, bad request)."""


class CreateUnknown(VastError, prov.CreateUnknown):
    """No usable answer: a box may or may not exist."""


class Vast:
    def __init__(self, key: str, base: str = API_BASE, timeout: float = 20):
        self.key, self.base, self.timeout = key, base, timeout

    def _call(self, method, path, *, body=None, query=None, retries=2, timeout=None):
        """The parsed JSON answer, of whatever shape. Callers validate it.

        `timeout` is for one request. A caller with a deadline passes what it
        has left and `retries=0`, and does its own retrying.
        """
        timeout = self.timeout if timeout is None else max(0.5, min(self.timeout, timeout))
        url = self.base + path
        if query:
            url += "?" + urllib.parse.urlencode(
                {k: v if isinstance(v, str) else json.dumps(v) for k, v in query.items()}
            )
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Accept": "application/json",
            "User-Agent": "gpubox",
        }
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read() or b"{}")
            except urllib.error.HTTPError as exc:
                # Provider error bodies can echo a private registry login.
                exc.close()
                if (exc.code == 429 or exc.code >= 500) and attempt < retries:
                    time.sleep(2**attempt)
                    continue
                raise VastError(exc.code, f"{method} {path}: HTTP {exc.code}") from None
            except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
                # Includes a body cut short or one that is not JSON.
                if attempt < retries:
                    time.sleep(2**attempt)
                    continue
                raise VastError(0, f"{method} {path}: {type(exc).__name__}") from None
        raise AssertionError("unreachable")

    def offers(self, query: dict) -> list[dict]:
        data = self._call("POST", "/v0/bundles/", body=query)
        offers = data.get("offers") if isinstance(data, dict) else None
        if not isinstance(offers, list):
            raise VastError(200, "unexpected answer to the offer search")
        # An offer we cannot identify or price is simply not a candidate.
        return [
            o
            for o in offers
            if isinstance(o, dict)
            and type(o.get("id")) is int
            and o["id"] > 0
            and type(o.get("machine_id")) is int
            and o["machine_id"] > 0
            and type(o.get("dph_total")) in (int, float)
            and 0 < o["dph_total"] < float("inf")
        ]

    def create(self, offer_id: int, payload: dict) -> int:
        """Rent a box. Sent exactly once: a retry could rent a second one.

        Raises CreateRefused only on an explicit no (a 4xx, or `success:
        false`), and returns an id only from an explicit yes carrying one.
        Everything else, including a timeout, a 5xx, an empty or malformed
        answer, is CreateUnknown: a billable box may exist.
        """
        try:
            data = self._call("PUT", f"/v0/asks/{offer_id}/", body=payload, retries=0)
        except VastError as exc:
            refused = 400 <= exc.status < 500 and exc.status not in (408, 429)
            raise (CreateRefused if refused else CreateUnknown)(exc.status, str(exc)) from None
        if isinstance(data, dict):
            if data.get("success") is False:
                raise CreateRefused(200, f"create on offer {offer_id} refused")
            contract = data.get("new_contract")
            if data.get("success") is True and type(contract) is int and contract > 0:
                return contract
        raise CreateUnknown(200, f"create on offer {offer_id}: unusable answer")

    def account(self) -> str:
        """The id of the account this key belongs to."""
        data = self._call("GET", "/v0/users/current/")  # no `owner`: the API refuses it here
        if (
            isinstance(data, dict)
            and data.get("success", True) is True
            and type(data.get("id")) is int
            and data["id"] > 0
        ):
            return str(data["id"])
        raise VastError(200, "unexpected answer to the account lookup")

    def instance(self, instance_id: int, **call) -> dict | None:
        """The instance, or None only when the provider says it does not exist.

        Absence is a 404 or an explicit `"instances": null`. Any other answer
        that is not an instance row is an error, never evidence of absence:
        callers use None to confirm a delete and free a slot.
        """
        try:
            data = self._call("GET", f"/v0/instances/{instance_id}/", query={"owner": "me"}, **call)
        except VastError as exc:
            if exc.status == 404:
                return None
            raise
        # The real answers are {"instances": null} and {"instances": {...}}; a
        # `success` field, if there is one, must say true.
        if isinstance(data, dict) and "instances" in data and data.get("success", True) is True:
            row = data["instances"]
            if row is None:
                return None
            if isinstance(row, dict) and type(row.get("id")) is int and row["id"] == instance_id:
                return row
        raise VastError(200, f"unexpected answer for instance {instance_id}")

    def instances(self, **call) -> list[dict]:
        """Every instance of the account, or an error: never a partial list.

        Callers conclude "no box carries this tag" from this, so a page or a
        row that cannot be read fails the whole listing. So does a row
        without a `label` field (null or empty means untagged; absent means
        we were not told), an id that appears twice, and a row count that
        differs from a `total_instances` that any page states.

        A `timeout` given here is for the whole listing, all pages together.
        """
        rows, query = [], {"select_filters": {}, "order_by": [{"col": "id", "dir": "asc"}]}
        query["limit"] = 25
        seen_tokens, seen_ids, totals = set(), set(), set()
        deadline = time.monotonic() + call["timeout"] if "timeout" in call else None
        for _ in range(MAX_PAGES):
            if deadline is not None:
                call["timeout"] = deadline - time.monotonic()
                if call["timeout"] <= 0:
                    raise VastError(0, "instance listing did not finish in time")
            data = self._call("GET", "/v1/instances/", query=query, **call)
            page = data.get("instances") if isinstance(data, dict) else None
            if not isinstance(page, list) or data.get("success", True) is not True:
                raise VastError(200, "unexpected answer to the instance listing")
            for row in page:
                if not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] <= 0:
                    raise VastError(200, "unreadable row in the instance listing")
                label = row.get("label", ...)
                if label is ... or not (label is None or isinstance(label, str)):
                    raise VastError(200, f"instance {row['id']} is listed without a usable label")
                if row["id"] in seen_ids:
                    raise VastError(200, f"instance {row['id']} is listed twice")
                seen_ids.add(row["id"])
            rows.extend(page)
            if data.get("total_instances") is not None:
                totals.add(repr(data["total_instances"]))
            if "next_token" not in data:
                # The last page says `"next_token": null`. A page without the
                # field has not told us whether more follow.
                raise VastError(200, "instance listing page has no next_token")
            token = data["next_token"]
            if token is None:
                if totals - {repr(len(rows))}:
                    raise VastError(
                        200, f"instance listing has {len(rows)} rows but states a different total"
                    )
                return rows
            if not isinstance(token, str) or not token or token in seen_tokens:
                raise VastError(200, "instance listing does not advance")
            seen_tokens.add(token)
            query["after_token"] = token
        raise VastError(200, f"instance listing has more than {MAX_PAGES} pages")

    def relabel(self, instance_id: int, label: str) -> None:
        """Set an instance's label. Raises unless the provider says it did."""
        data = self._call("PUT", f"/v0/instances/{instance_id}/", body={"label": label})
        if not isinstance(data, dict) or data.get("success") is not True:
            raise VastError(200, f"relabel of instance {instance_id} refused")

    def destroy(self, instance_id: int, **call) -> None:
        """Ask for destruction. An instance that is already gone is not an error."""
        try:
            data = self._call("DELETE", f"/v0/instances/{instance_id}/", body={}, **call)
        except VastError as exc:
            if exc.status != 404:
                raise
            return
        if not isinstance(data, dict) or data.get("success", True) is not True:
            raise VastError(200, f"destroy of instance {instance_id} refused")


def offer_query(spec) -> dict:
    n = spec.num_gpus
    return {
        "gpu_name": {"in": list(spec.gpus)},
        "num_gpus": {"eq": n},
        "cuda_max_good": {"gte": spec.min_cuda},
        "reliability": {"gte": spec.min_reliability},
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "direct_port_count": {"gte": 1},
        "inet_down": {"gte": spec.min_inet_down_mbps},
        # By default, cores for a compile pool on top of one worker per GPU.
        "cpu_cores_effective": {"gte": max(16, 8 * n) if spec.min_cpus is None else spec.min_cpus},
        "cpu_ram": {"gte": (32 * n if spec.min_ram_gb is None else spec.min_ram_gb) * 1024},
        "dph_total": {"lte": spec.max_dph_per_gpu * n},
        "disk_space": {"gte": spec.disk_gb},
        # Price the offer with the disk we will actually allocate.
        "allocated_storage": spec.disk_gb,
        "type": "bid" if spec.interruptible else "on-demand",
        "order": [["dph_total", "asc"]],
        "limit": 200,
    }


# Vast gives every container CONTAINER_API_KEY, a key confined to that one
# instance, which is enough for a box to destroy itself.
SELF_DESTROY = (
    "import os, urllib.request as u; u.urlopen(u.Request("
    "'https://console.vast.ai/api/v0/instances/' + os.environ['CONTAINER_ID'] + '/', "
    "method='DELETE', data=b'{}', headers={'Content-Type': 'application/json', "
    "'Authorization': 'Bearer ' + os.environ['CONTAINER_API_KEY']}), timeout=30)"
)


DESTROY = f"python3 -c {shlex.quote(SELF_DESTROY)}"


def bootstrap_script(pubkey: str, expiry: int) -> str:
    """Vast's box: the container's start-up command (container.py), deleting itself with
    the container's own key."""
    return container.bootstrap_script(pubkey, expiry, DESTROY)


def bootstrap_command(pubkey: str, expiry: int) -> str:
    """The bootstrap as one shell command line, for a person to read or run."""
    return container.bootstrap_command(pubkey, expiry, DESTROY)


def registry_of(image: str) -> str:
    """The registry an image reference names, as docker reads it: the first path component
    if it looks like a host (has a dot or a port, or is localhost), else Docker Hub."""
    first, slash, _ = image.partition("/")
    if slash and ("." in first or ":" in first or first == "localhost"):
        return first
    return "docker.io"


def create_payload(spec, label: str, pubkey: str, expiry: int, bid: float | None = None) -> dict:
    payload = {
        "image": spec.image,
        "disk": spec.disk_gb,
        "label": label,
        "target_state": "running",
        "runtype": "args",
        "env": {"-p 22:22": "1"},
        # In the args launch mode, `onstart` replaces the image's ENTRYPOINT (as
        # `vastai create instance --entrypoint` sends it) and `args` are its arguments.
        "onstart": "bash",
        "args": ["-c", bootstrap_script(pubkey, expiry)],
    }
    if spec.interruptible:
        if bid is None:
            raise ValueError("an interruptible box needs a bid")
        payload["price"] = bid  # $/hr for the machine, the disk not included
    if spec.registry_user is not None:
        # Vast's form: `docker login` arguments, in one string.
        login = f"-u {spec.registry_user} -p {spec.registry_token} {registry_of(spec.image)}"
        payload["image_login"] = login
    return payload


def country_of(offer: dict) -> str | None:
    """The country code an offer's location ends in ("Anhui, CN" -> "CN"), if it has one."""
    geo = offer.get("geolocation")
    if not isinstance(geo, str):
        return None
    code = geo.rsplit(",", 1)[-1].strip().upper()
    return code if len(code) == 2 and code.isalpha() else None


def in_allowed_country(offers: list[dict], excluded) -> list[dict]:
    """Offers not located in an excluded country. An offer without a location is kept."""
    excluded = {c.upper() for c in excluded}
    return [o for o in offers if country_of(o) not in excluded]


def endpoint_of(instance: dict) -> tuple[str, int] | None:
    """Where the box's sshd is, or None if the row does not say so in a form we can use.
    A row we cannot read is one box's problem, never an error for the whole launch."""
    try:
        ip = str(instance["public_ipaddr"]).strip()
        port = int(instance["ports"]["22/tcp"][0]["HostPort"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    usable = ip and not ip.startswith("-") and not any(c.isspace() for c in ip)
    return (ip, port) if usable and 0 < port < 65536 else None


def priced_bids(vast, spec, found: list[dict]) -> list[dict]:
    """Interruptible offers, each with what we would bid (`_bid`, at least the minimum,
    rounded up) and what that costs an hour with the disk (`dph_total`; the disk's price
    is what the offer's total says beyond its minimum bid). An offer that cannot be priced
    so is left out, and so is any machine this account already has an instance on: two
    bids of ours on one machine outbid each other. A listing row that does not say its
    machine fails the launch rather than let such a bid through."""
    ours = set()
    for r in vast.instances():
        if type(r.get("machine_id")) is not int:
            raise VastError(200, f"instance {r.get('id')} is listed without its machine")
        ours.add(r["machine_id"])
    priced = []
    for o in found:
        min_bid, total = o.get("min_bid"), o["dph_total"]
        if type(min_bid) not in (int, float) or not 0 < min_bid <= total < 1e4:
            continue
        if o["machine_id"] in ours:
            continue
        bid = max(min_bid, math.ceil(min_bid * spec.bid_markup * 1e4) / 1e4)
        priced.append({**o, "_bid": bid, "dph_total": bid + (total - min_bid)})
    return priced


def candidates(vast, spec, blocked: dict[int, dict], want: int):
    """(offers to try, in order; the offers within the price cap; those also in an allowed
    country). What `up` rents from and `search` shows."""
    found = vast.offers(offer_query(spec))
    if spec.interruptible:
        found = priced_bids(vast, spec, found)
    # The search is asked for the price cap and the order; neither is taken
    # on trust, since what comes back is what gets rented.
    cap = spec.max_dph_per_gpu * spec.num_gpus + 1e-9
    found = sorted((o for o in found if o["dph_total"] <= cap), key=lambda o: o["dph_total"])
    allowed = in_allowed_country(found, spec.exclude_countries)
    return pol.select_offers(allowed, blocked, want), found, allowed


class VastProvider:
    """The launcher's view of Vast (provider.py): the HTTP client above, plus how offers
    are chosen and how a row is read."""

    name = "vast"
    noun = "instance"
    credential = "VASTAI_API_KEY"
    access = "ssh"
    spawn = "setsid"
    # A box runs an sshd that gpubox starts with the run's key: one that refuses the key
    # shows that the image did not run our start-up command.
    own_sshd = True
    self_destroy = True  # with the container's own key (bootstrap_script)
    lease = "box"
    observes_machines = True
    default_spare = 1
    renewable = True  # its lease can be moved (relabel)
    requires = frozenset({"image", "price_cap"})
    features = frozenset(
        {"image", "price_cap", "interruptible", "registry_login", "exclude_country", "ssh_config"}
    )
    searchable = True

    def __init__(self, client):
        self.client = client

    @classmethod
    def from_options(cls, options: dict, key: str) -> VastProvider:
        return cls(Vast(key))

    def options(self) -> dict:
        return {}

    def candidates(self, spec, blocked: dict, policy) -> tuple[list[dict], str]:
        if not spec.gpus:
            raise ValueError("a Vast search needs at least one GPU model")
        offers, found, allowed = candidates(self.client, spec, blocked, policy.max_live)
        note = (
            f"{len(offers)} candidate host(s) for {spec.gpu_text} "
            f"({len(blocked)} machine(s) blocked, {len(found) - len(allowed)} offer(s) in "
            "excluded countries)"
        )
        return offers, note

    def create(self, offer: dict, spec, label: str, pubkey: str, expiry: int, avoid=()) -> int:
        # `avoid` is for providers that choose the machine themselves; here the offer is one.
        payload = create_payload(spec, label, pubkey, expiry, bid=offer.get("_bid"))
        return self.client.create(offer["id"], payload)

    def view(self, row: dict) -> prov.View:
        status = str(row.get("actual_status") or "none")
        msg = str(row.get("status_msg") or "")
        state = {"running": "running", "exited": "exited"}.get(status, "loading")
        endpoint = endpoint_of(row) if state == "running" else None
        return prov.View(state, status, msg, classify_status(msg), endpoint)

    def instance(self, instance_id: int, **call):
        return self.client.instance(instance_id, **call)

    def instances(self, **call):
        return self.client.instances(**call)

    def destroy(self, instance_id: int, **call) -> None:
        self.client.destroy(instance_id, **call)

    def relabel(self, instance_id: int, label: str) -> None:
        self.client.relabel(instance_id, label)

    def account(self) -> str:
        return self.client.account()

    def expiry_command(self, expiry: int) -> str:
        return set_expiry_command(expiry)

    def manual_hint(self, run_id: str, ids) -> str:
        return "Or destroy them at https://cloud.vast.ai/instances/ (their labels start with the run id)"

    def describe(self, a) -> str:
        return f"${a.dph:.2f}/hr, host {a.host_id}, {a.geo}"
