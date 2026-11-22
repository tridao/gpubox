"""Vast.ai: the HTTP client and everything that is specific to that provider.

Standard library only. The official `vastai` SDK was considered and not used:
its high-level calls do not send the payloads below, and it retries the create
request (see the design notes). Read its source when the API changes.
"""

from __future__ import annotations

import json
import re
import shlex
import time
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://console.vast.ai/api"

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

    def _call(self, method, path, *, body=None, query=None, retries=2) -> dict:
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
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read() or b"{}")
            except urllib.error.HTTPError as exc:
                text = exc.read().decode(errors="replace").strip()[:500]
                if (exc.code == 429 or exc.code >= 500) and attempt < retries:
                    time.sleep(2**attempt)
                    continue
                raise VastError(exc.code, f"{method} {path}: HTTP {exc.code}: {text}") from None
            except (urllib.error.URLError, OSError, ValueError) as exc:
                if attempt < retries:
                    time.sleep(2**attempt)
                    continue
                raise VastError(0, f"{method} {path}: {exc}") from None
        raise AssertionError("unreachable")

    def offers(self, query: dict) -> list[dict]:
        return self._call("POST", "/v0/bundles/", body=query).get("offers") or []

    def create(self, offer_id: int, payload: dict) -> int:
        """Rent a box. Sent exactly once: a retry could rent a second one.

        Raises CreateRefused when the provider said no, CreateUnknown when we
        cannot tell (timeout, 5xx, unreadable answer). The two must not be
        confused: after CreateUnknown a billable box may exist.
        """
        try:
            data = self._call("PUT", f"/v0/asks/{offer_id}/", body=payload, retries=0)
        except VastError as exc:
            kind = CreateRefused if 400 <= exc.status < 500 else CreateUnknown
            raise kind(exc.status, str(exc)) from None
        if not data.get("success"):
            raise CreateRefused(200, f"create on offer {offer_id} refused: {str(data)[:300]}")
        if not data.get("new_contract"):
            raise CreateUnknown(200, f"create on offer {offer_id}: no id in {str(data)[:300]}")
        return int(data["new_contract"])

    def instance(self, instance_id: int) -> dict | None:
        """The instance, or None when the provider says it does not exist."""
        try:
            data = self._call("GET", f"/v0/instances/{instance_id}/", query={"owner": "me"})
        except VastError as exc:
            if exc.status == 404:
                return None
            raise
        return data.get("instances") or None

    def instances(self) -> list[dict]:
        rows, query = [], {"select_filters": {}, "order_by": [{"col": "id", "dir": "asc"}]}
        query["limit"] = 25
        while True:
            data = self._call("GET", "/v1/instances/", query=query)
            rows.extend(data.get("instances") or [])
            if not data.get("next_token"):
                return rows
            query["after_token"] = data["next_token"]

    def destroy(self, instance_id: int) -> None:
        """Ask for destruction. An instance that is already gone is not an error."""
        try:
            self._call("DELETE", f"/v0/instances/{instance_id}/", body={})
        except VastError as exc:
            if exc.status != 404:
                raise


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


def bootstrap_command(pubkey: str, ttl_min: int) -> str:
    """Container command: a self-destruct timer, then sshd for our key.

    The timer is a dead-man switch that needs neither the launcher nor our API
    key. It has to destroy the instance: merely exiting does not stop the
    billing, because Vast restarts a container whose main process ends. It is
    part of the create request, so even a box whose create response we never
    saw carries it. One line, because Vast passes this as `docker run` arguments.
    """
    watchdog = (
        f"(sleep {ttl_min}m; until python3 -c {shlex.quote(SELF_DESTROY)}; do sleep 60; done)"
        " >/dev/null 2>&1 &"
    )
    script = "; ".join(
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
            f"{watchdog} exec /usr/sbin/sshd -D -e"
            " -o PermitRootLogin=prohibit-password -o PasswordAuthentication=no",
        ]
    )
    return "bash -c " + shlex.quote(script)


def create_payload(spec, label: str, pubkey: str) -> dict:
    return {
        "image": spec.image,
        "disk": spec.disk_gb,
        "label": label,
        "target_state": "running",
        "runtype": "args",
        "env": {"-p 22:22": "1"},
        "args_str": bootstrap_command(pubkey, spec.ttl_min),
    }


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
