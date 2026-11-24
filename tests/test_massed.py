"""Massed Compute: its MCP tools, its VMs, and a launch on it, against a fake."""

import base64
import re
import time

import pytest
from fakes import FAST
from test_shadeform import VmProber

from gpubox import api
from gpubox import launch as L
from gpubox.providers import massed as ms
from gpubox.providers import base as prov
from gpubox.providers import vmbox
from gpubox.errors import ConfigError
from gpubox.state import State, make_tag

RUN_ID = "dev.ms.0123456789ab"
EXPIRY = int(time.time()) + 3600
SPEC = L.Spec(image="img:1", gpus=(), max_dph_per_gpu=1.0, min_cpus=6, min_ram_gb=32)


def product(name, desc, cents, regions, vcpu=6, ram=32, disk=256):
    return {"instance_type": {"name": name, "description": desc, "price_cents_per_hour": cents,
                              "specs": {"vcpu_count": vcpu, "memory_gib": ram, "storage_gb": disk}},
            "capacity_available": len(regions),
            "regions_with_capacity_available": [{"name": r} for r in regions]}  # fmt: skip


INVENTORY = {
    "gpu_1x_a6000_low_ram": product(
        "gpu_1x_a6000_low_ram", "1x RTX A6000 (48GB)", 55, ["us-central-2"]
    ),
    "gpu_1x_a6000": product(
        "gpu_1x_a6000", "1x RTX A6000 (48GB)", 57, ["us-central-1", "us-central-3"]
    ),
    "gpu_2x_a6000": product("gpu_2x_a6000", "2x RTX A6000 (48GB)", 114, ["us-central-1"]),
    "gpu_1x_h100": product("gpu_1x_h100", "1x H100 (80GB)", 250, ["us-central-1"]),  # above the cap
    "gpu_1x_l40": product(
        "gpu_1x_l40", "1x L40 (48GB)", 86, ["us-central-1"], ram=None
    ),  # RAM not stated
}
UUID = "00000000-0000-4000-8000-%012d"


class FakeClient:
    def __init__(self):
        self.vms, self.calls, self.fail, self.next = {}, [], {}, 0

    def tool(self, name, args, *, retries=2, timeout=None):
        self.calls.append((name, args, retries))
        if self.fail.get(name):
            answer = self.fail[name].pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        if name == "gpu_inventory_list":
            return {"gpu_inventory": INVENTORY}
        if name == "images_list":
            return {"images": [{"vm_image_id": 42, "vm_image_name": "Ubuntu Server 22 Server - Bare"},
                               {"vm_image_id": 84, "vm_image_name": ms.IMAGE}]}  # fmt: skip
        if name == "instances_launch":
            self.next += 1
            uuid = UUID % self.next
            self.vms[uuid] = {"uuid": uuid, "name": args["instanceName"], "status": "Initializing",
                              "active": 0, "os_booted": 0, "ip": "192.0.2.1", "username": "Ubuntu",
                              "args": args}  # fmt: skip
            return {"response": uuid}
        if name == "instances_get":
            vm = self.vms.get(args["uuid"])
            if vm and vm["status"] == "Initializing":  # running at the next look
                vm.update(status="Running", active=1, os_booted=1)
            return {"runningInstances": [dict(vm)] if vm else []}
        if name == "instances_list":
            return {"runningInstances": [dict(v) for v in self.vms.values()]}
        if name == "instances_terminate":
            for uuid in args["instanceUuids"]:
                self.vms.pop(uuid, None)
            return {"response": {"ok": True}}
        raise AssertionError(name)


def make():
    c = FakeClient()
    return ms.Massed(c), c


def test_offers_are_products_with_that_many_gpus_in_stock_within_the_cap_and_sizes():
    p, _ = make()
    offers, note = p.candidates(SPEC, {}, L.Policy(spare=0))
    assert [(o["_product"], o["_region"]) for o in offers] == [
        ("gpu_1x_a6000_low_ram", "us-central-2"), ("gpu_1x_a6000", "us-central-1"),
        ("gpu_1x_a6000", "us-central-3"),
    ]  # fmt: skip
    assert offers[0]["_image"] == 84 and offers[0]["gpu_name"] == "RTX A6000 (48GB)"
    assert (
        p.candidates(L.Spec(image="i", gpus=(), max_dph_per_gpu=1.0), {}, L.Policy())[0] == []
    )  # 16 CPUs


def test_a_vm_is_launched_once_with_our_tag_as_its_name_and_the_box_script_at_boot():
    p, c = make()
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    tag = make_tag(RUN_ID, 1, EXPIRY)
    uuid = p.create(offer, SPEC, tag, "ssh-ed25519 AAAA gpubox", EXPIRY)
    name, args, retries = c.calls[-1]
    assert (name, retries) == ("instances_launch", 0) and args["instanceName"] == tag
    b64 = re.fullmatch(r"echo (\S+) \| base64 -d \| bash", args["command"]).group(1)
    assert base64.b64decode(b64).decode() == vmbox.startup_script(
        "ssh-ed25519 AAAA gpubox", "img:1"
    )
    assert (
        uuid in c.vms and "sshKeys" not in args
    )  # the key goes in with the script, not the account


@pytest.mark.parametrize(
    "answer, outcome",
    [
        (ms.MassedError(422, "instances_launch: out of capacity"), prov.CreateRefused),
        (ms.MassedError(422, "instances_launch: upstream timed out"), prov.CreateUnknown),
        (ms.MassedError(400, "instances_launch: invalid"), prov.CreateRefused),
        (ms.MassedError(401, "instances_launch: HTTP 401"), ConfigError),
        (ms.MassedError(502, "instances_launch: HTTP 502"), prov.CreateUnknown),
        (ms.MassedError(0, "instances_launch: timed out"), prov.CreateUnknown),
        ({"response": "not-a-uuid"}, prov.CreateUnknown),
        (None, prov.CreateUnknown),
        ({"response": UUID % 77}, UUID % 77),
    ],
)
def test_an_instance_id_only_from_an_explicit_yes(answer, outcome):
    p, c = make()
    c.fail["instances_launch"] = [answer]
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    if isinstance(outcome, str):
        assert (
            p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "ssh-ed25519 K", EXPIRY) == outcome
        )
    else:
        with pytest.raises(outcome):
            p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "ssh-ed25519 K", EXPIRY)


def test_an_instance_no_longer_listed_is_gone_and_an_error_never_is():
    p, c = make()
    assert p.instance(UUID % 1) is None
    c.vms[UUID % 1] = {"uuid": UUID % 1, "name": None, "status": "Running", "username": "Ubuntu"}
    assert p.instance(UUID % 1)["label"] is None
    c.vms[UUID % 1]["status"] = "Terminating"  # under way: not yet gone
    assert p.instance(UUID % 1) is not None
    c.vms[UUID % 1]["status"] = "Terminated"
    assert p.instance(UUID % 1) is None
    c.vms[UUID % 1]["status"] = "Running"
    bad = [ms.MassedError(0, "x"), {"runningInstances": None}, {},
           {"runningInstances": [{"uuid": UUID % 2, "name": None, "status": "Running"}]},
           {"runningInstances": [{"uuid": UUID % 1, "status": "Running"}]},
           {"runningInstances": [dict(c.vms[UUID % 1])] * 2}]  # fmt: skip
    for answer in bad:
        c.fail["instances_get"] = [answer]
        with pytest.raises(ms.MassedError):
            p.instance(UUID % 1)


def test_the_listing_is_every_instance_or_an_error():
    p, c = make()
    tag = make_tag(RUN_ID, 1, EXPIRY)
    c.vms[UUID % 1] = {"uuid": UUID % 1, "name": tag, "status": "Initializing"}
    c.vms[UUID % 2] = {"uuid": UUID % 2, "name": "other", "status": "Terminated"}
    assert [(r["id"], r["label"]) for r in p.instances()] == [(UUID % 1, tag)]
    for answer in ({"runningInstances": [{"uuid": "x", "name": "a", "status": "Running"}]},
                   {"runningInstances": [c.vms[UUID % 1]] * 2}, {}):  # fmt: skip
        c.fail["instances_list"] = [answer]
        with pytest.raises(ms.MassedError):
            p.instances()


def test_an_instance_is_running_for_us_once_booted_with_an_ip_and_its_login_user():
    p, _ = make()
    row = ms._row({"uuid": UUID % 1, "name": None, "status": "Initializing", "active": 0,
                   "os_booted": 0, "ip": "192.0.2.2", "username": "Ubuntu"})  # fmt: skip
    assert p.view(row).state == "loading"
    row.update(actual_status="Running", active=1, os_booted=1)
    assert p.view(row).endpoint == ("192.0.2.2", 22, "Ubuntu")  # its own user, capitals and all
    row["username"] = "-oProxyCommand=x"
    assert p.view(row).state == "loading"


def test_the_mcp_answers_are_read_as_json_or_as_events():
    assert ms._parse('{"result": 1}') == {"result": 1}
    assert ms._parse('event: message\ndata: {"result": 2}\n\n') == {"result": 2}


def test_a_launch_rents_a_vm_down_terminates_it_and_renew_is_refused(tmp_path):
    p, c = make()
    state = State.begin(tmp_path, RUN_ID, EXPIRY, None, {"provider": "massed"})
    run = L.Launch(p, SPEC, L.Policy(**{**FAST.__dict__, "spare": 0}), state, VmProber())
    (w,) = run.run({}).winners
    (box,) = State.load(tmp_path).winners
    assert (box["id"], box["user"]) == (w.instance_id, "Ubuntu")
    with pytest.raises(ConfigError, match="cannot be moved"):
        api.attach(state_dir=tmp_path).renew(60)
    assert L.down(p, state, FAST) == [] and c.vms == {}


def test_only_a_request_refused_as_it_stood_is_a_definite_no(monkeypatch):
    import io

    answers = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ms.httpapi.OPENER, "open", lambda req, timeout=None: Resp(answers.pop(0)))
    client = ms.Client("k")
    for code, status in ((-32602, 400), (-32603, 0)):
        answers.append(b'{"jsonrpc": "2.0", "id": 1, "error": {"code": %d, "message": "x"}}' % code)
        with pytest.raises(ms.MassedError) as exc:
            client.tool("instances_launch", {}, retries=0)
        assert exc.value.status == status
    answers.append(b'{"jsonrpc": "2.0", "id": 7, "result": {"structuredContent": {}}}')  # not ours
    with pytest.raises(ms.MassedError):
        client.tool("instances_list", {}, retries=0)
