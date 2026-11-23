"""Vast.ai: the HTTP client and everything that is specific to that provider.

Standard library only. The official `vastai` SDK was considered and not used:
its high-level calls do not send the payloads below, and it retries the create
request (see the design notes). Read its source when the API changes.
"""

from __future__ import annotations

import http.client
import json
import re
import shlex
import time
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://console.vast.ai/api"
MAX_PAGES = 400  # 10,000 instances; a listing longer than this is a loop

# status_msg substrings. An image problem is ours and identical on every host;
# a runtime problem belongs to the machine.
IMAGE_ERRORS = (
    "manifest unknown",
    "pull access denied",
    "repository does not exist",
    "authentication required",
)
MACHINE_ERRORS = (
    "oci runtime create failed",
    "failed to create shim",
    "nvidia-container-cli",
    "no space left on device",
)
_CDI = re.compile(r"\bCDI\b")


class VastError(RuntimeError):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


class CreateRefused(VastError):
    """The provider answered and did not rent a box (stale offer, bad request)."""


class CreateUnknown(VastError):
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
                # The status is the answer; the body is only a description and
                # reading it may fail too.
                try:
                    text = exc.read().decode(errors="replace").strip()[:500]
                except Exception:
                    text = "<unreadable error body>"
                if (exc.code == 429 or exc.code >= 500) and attempt < retries:
                    time.sleep(2**attempt)
                    continue
                raise VastError(exc.code, f"{method} {path}: HTTP {exc.code}: {text}") from None
            except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
                # Includes a body cut short or one that is not JSON.
                if attempt < retries:
                    time.sleep(2**attempt)
                    continue
                raise VastError(0, f"{method} {path}: {exc}") from None
        raise AssertionError("unreachable")

    def offers(self, query: dict) -> list[dict]:
        data = self._call("POST", "/v0/bundles/", body=query)
        offers = data.get("offers") if isinstance(data, dict) else None
        if not isinstance(offers, list):
            raise VastError(200, f"unexpected answer to the offer search: {str(data)[:200]}")
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
            refused = 400 <= exc.status < 500 and exc.status != 408
            raise (CreateRefused if refused else CreateUnknown)(exc.status, str(exc)) from None
        if isinstance(data, dict):
            if data.get("success") is False:
                raise CreateRefused(200, f"create on offer {offer_id} refused: {str(data)[:300]}")
            contract = data.get("new_contract")
            if data.get("success") is True and type(contract) is int and contract > 0:
                return contract
        raise CreateUnknown(200, f"create on offer {offer_id}: unusable answer {str(data)[:300]}")

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
        raise VastError(200, f"unexpected answer to the account lookup: {str(data)[:200]}")

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
        raise VastError(200, f"unexpected answer for instance {instance_id}: {str(data)[:200]}")

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
                raise VastError(
                    200, f"unexpected answer to the instance listing: {str(data)[:200]}"
                )
            for row in page:
                if not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] <= 0:
                    raise VastError(
                        200, f"unreadable row in the instance listing: {str(row)[:200]}"
                    )
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
                    stated = ", ".join(sorted(totals))
                    raise VastError(
                        200, f"instance listing has {len(rows)} rows but states {stated}"
                    )
                return rows
            if not isinstance(token, str) or not token or token in seen_tokens:
                raise VastError(200, f"instance listing does not advance (token {token!r})")
            seen_tokens.add(token)
            query["after_token"] = token
        raise VastError(200, f"instance listing has more than {MAX_PAGES} pages")

    def destroy(self, instance_id: int, **call) -> None:
        """Ask for destruction. An instance that is already gone is not an error."""
        try:
            data = self._call("DELETE", f"/v0/instances/{instance_id}/", body={}, **call)
        except VastError as exc:
            if exc.status != 404:
                raise
            return
        if not isinstance(data, dict) or data.get("success", True) is not True:
            raise VastError(200, f"destroy of instance {instance_id} refused: {str(data)[:200]}")


def offer_query(spec) -> dict:
    n = spec.num_gpus
    return {
        "gpu_name": {"in": [spec.gpu]},
        "num_gpus": {"eq": n},
        "cuda_max_good": {"gte": spec.min_cuda},
        "reliability": {"gte": spec.min_reliability},
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "direct_port_count": {"gte": 1},
        "inet_down": {"gte": spec.min_inet_down_mbps},
        # Cores for a compile pool on top of one worker per GPU.
        "cpu_cores_effective": {"gte": max(16, spec.min_cpus_per_gpu * n)},
        "cpu_ram": {"gte": spec.min_ram_gb_per_gpu * n * 1024},
        "dph_total": {"lte": spec.max_dph_per_gpu * n},
        "disk_space": {"gte": spec.disk_gb},
        # Price the offer with the disk we will actually allocate.
        "allocated_storage": spec.disk_gb,
        "type": "on-demand",
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


def bootstrap_command(pubkey: str, expiry: int) -> str:
    """Container command: a self-destruct timer, then sshd for our key.

    The timer is a dead-man switch that needs neither the launcher nor our API
    key. It has to destroy the instance: merely exiting does not stop the
    billing, because Vast restarts a container whose main process ends. It is
    part of the create request, so even a box whose create response we never
    saw carries it. One line, because Vast passes this as `docker run` arguments.

    It is started first, so a box stuck installing sshd is covered. It waits
    for an absolute time (the expiry in the box's tag), so a container that
    restarts does not start counting again. Once that time has passed it
    sends the delete every 30 s for as long as it runs, whatever the answers
    say: if the delete works, the loop ends with the box.
    """
    if type(expiry) is not int or not 0 < expiry < 10**10:
        raise ValueError(f"bad expiry {expiry!r}")
    watchdog = (
        f'(while :; do if [ "$(date +%s)" -ge {expiry} ]; then'
        f" python3 -c {shlex.quote(SELF_DESTROY)}; fi; sleep 30; done) >/dev/null 2>&1 &"
    )
    script = (
        watchdog
        + " "
        + "; ".join(
            [
                "set -eu",
                "if ! command -v sshd >/dev/null 2>&1; then apt-get update -qq"
                " && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq"
                " --no-install-recommends openssh-server; fi",
                "mkdir -p /run/sshd /root/.ssh",
                "chmod 700 /root/.ssh",
                f"printf '%s\\n' {shlex.quote(pubkey)} > /root/.ssh/authorized_keys",
                "chmod 600 /root/.ssh/authorized_keys",
                "ssh-keygen -A >/dev/null",
                "exec /usr/sbin/sshd -D -e"
                " -o PermitRootLogin=prohibit-password -o PasswordAuthentication=no",
            ]
        )
    )
    return "bash -c " + shlex.quote(script)


def create_payload(spec, label: str, pubkey: str, expiry: int) -> dict:
    return {
        "image": spec.image,
        "disk": spec.disk_gb,
        "label": label,
        "target_state": "running",
        "runtype": "args",
        "env": {"-p 22:22": "1"},
        "args_str": bootstrap_command(pubkey, expiry),
    }


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


def classify_status(msg: str) -> str | None:
    """'image', 'machine', or None for a status message that decides nothing."""
    low = msg.lower()
    if any(s in low for s in IMAGE_ERRORS):
        return "image"
    if any(s in low for s in MACHINE_ERRORS) or _CDI.search(msg):
        return "machine"
    return None


def endpoint_of(instance: dict) -> tuple[str, int] | None:
    ip = instance.get("public_ipaddr")
    mapped = (instance.get("ports") or {}).get("22/tcp") or []
    if ip and mapped and mapped[0].get("HostPort"):
        return str(ip).strip(), int(mapped[0]["HostPort"])
    return None
