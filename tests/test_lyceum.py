"""Lyceum: its availability, its VMs, and a launch on it, against a fake of its API."""

import time

import pytest
from fakes import FAST
from test_shadeform import VmProber

from gpubox import launch as L
from gpubox.providers import lyceum as ly
from gpubox.providers import base as prov
from gpubox.errors import ConfigError
from gpubox.state import State, make_tag, parse_tag

RUN_ID = "dev.ly.0123456789ab"
EXPIRY = int(time.time()) + 3600
SPEC = L.Spec(image="img:1", gpus=(), max_dph_per_gpu=3.0)
VM = "00000000-0000-4000-9000-%012d"
PROFILES = [
    {
        "hardware_profile": "l40s",
        "gpu_type": "l40s",
        "price_per_hour": 2.39,
        "available_gpus_per_instance": [1, 2],
    },
    {
        "hardware_profile": "h100",
        "gpu_type": "h100",
        "price_per_hour": 6.19,
        "available_gpus_per_instance": [1],
    },
    {
        "hardware_profile": "a100",
        "gpu_type": "a100",
        "price_per_hour": 2.79,
        "available_gpus_per_instance": [4],
    },
]


class FakeApi:
    def __init__(self):
        self.vms, self.calls, self.fail, self.next, self.credits = {}, [], {}, 0, 80.0

    def call(self, method, path, *, body=None, query=None, retries=2, timeout=None):
        self.calls.append((method, path, body, query, retries))
        key = f"{method} {path.split('/')[0]}/{path.split('/')[1] if '/' in path else ''}"
        if self.fail.get(key):
            answer = self.fail[key].pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        if path == "billing/credits":
            return {"available_credits": self.credits}
        if path == "vms/availability":
            return {"available_hardware_profiles": PROFILES}
        if path == "vms/create":
            self.next += 1
            vid = VM % self.next
            self.vms[vid] = {"vm_id": vid, "status": "provisioning", "ip_address": None,
                             "display_name": body["display_name"], "body": body}  # fmt: skip
            return dict(self.vms[vid])
        if path == "vms/list":
            return {"vms": list(self.vms.values())}
        vid = path.split("/")[1]
        vm = self.vms.get(vid)
        if vm is None:
            raise ly.LyceumError(404, f"{method} /{path}: HTTP 404: not found")
        if method == "DELETE":
            del self.vms[vid]
            return {"vm_id": vid, "status": "terminated"}
        if method == "PATCH":
            vm["display_name"] = body["display_name"]
            return dict(vm)
        if vm["status"] == "provisioning":
            vm.update(status="ready", ip_address=f"192.0.2.{self.next}:22")
        return dict(vm)


def make():
    api = FakeApi()
    return ly.Lyceum(api), api


def test_offers_are_profiles_with_that_many_gpus_available_within_the_cap():
    p, api = make()
    offers, _ = p.candidates(SPEC, {}, L.Policy(spare=0))
    assert [o["_profile"] for o in offers] == ["l40s"]  # h100 over the cap, a100 not as 1x
    assert offers[0]["dph_total"] == 2.39
    big = L.Spec(image="i", gpus=(), max_dph_per_gpu=3.0, disk_gb=1000)
    assert p.candidates(big, {}, L.Policy())[0] == []  # its disk is fixed at 500 GB
    api.credits = 0
    with pytest.raises(ConfigError, match="credits"):
        p.candidates(SPEC, {}, L.Policy())


def test_a_vm_is_created_once_with_the_runs_key_and_our_tag_as_its_name():
    p, api = make()
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    tag = make_tag(RUN_ID, 1, EXPIRY)
    vid = p.create(offer, SPEC, tag, "ssh-ed25519 AAAA gpubox", EXPIRY)
    method, path, body, _, retries = api.calls[-1]
    assert (method, path, retries) == ("POST", "vms/create", 0) and vid in api.vms
    assert body["user_public_key"] == "ssh-ed25519 AAAA gpubox" and body["display_name"] == tag
    assert body["instance_specs"] == {"gpu_type": "l40s", "gpu_count": 1}


@pytest.mark.parametrize(
    "answer, outcome",
    [
        (ly.LyceumError(400, "no capacity"), prov.CreateRefused),
        (ly.LyceumError(422, "validation"), prov.CreateRefused),
        (ly.LyceumError(401, "bad key"), ConfigError),
        (ly.LyceumError(500, "boom"), prov.CreateUnknown),
        (ly.LyceumError(0, "timed out"), prov.CreateUnknown),
        ({"vm_id": "not-a-uuid"}, prov.CreateUnknown),
        ({"vm_id": VM % 9}, VM % 9),
        ({"vm_id": VM % 9, "display_name": "someone-else"}, prov.CreateUnknown),
    ],
)
def test_a_vm_id_only_from_an_explicit_yes(answer, outcome):
    p, api = make()
    api.fail["POST vms/create"] = [answer]
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    if isinstance(outcome, str):
        assert (
            p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "ssh-ed25519 K", EXPIRY) == outcome
        )
    else:
        with pytest.raises(outcome):
            p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "ssh-ed25519 K", EXPIRY)


def test_the_listing_asks_for_every_state_and_reads_strictly():
    p, api = make()
    api.vms[VM % 1] = {"vm_id": VM % 1, "status": "provisioning", "display_name": "x"}
    api.vms[VM % 2] = {"vm_id": VM % 2, "status": "terminated", "display_name": "y"}
    assert [r["id"] for r in p.instances()] == [VM % 1]
    query = api.calls[-1][3]
    assert (
        query["include_provisioning"] == query["include_ready"] == query["include_failed"] == "true"
    )
    for bad in ({"vms": [{"vm_id": VM % 1, "status": "ready"}]},  # no display_name
                {"vms": [api.vms[VM % 1]] * 2}, {}):  # fmt: skip
        api.fail["GET vms/list"] = [bad]
        with pytest.raises(ly.LyceumError):
            p.instances()


def test_a_vm_runs_for_us_as_lyceum_once_ready_with_an_address():
    p, _ = make()
    row = ly._row(
        {"vm_id": VM % 1, "status": "ready", "ip_address": "192.0.2.1:2222", "display_name": None}
    )
    assert p.view(row).endpoint == ("192.0.2.1", 2222, "lyceum")
    assert ly.endpoint_of("192.0.2.1") == ("192.0.2.1", 22) and ly.endpoint_of("-x:22") is None
    row["actual_status"] = "provisioning"
    assert p.view(row).state == "loading"


def test_a_launch_starts_the_box_through_its_first_probe_and_down_ends_it(tmp_path):
    from gpubox import api as gapi
    from gpubox.providers import vmbox

    p, api = make()
    assert not ly.Lyceum.startup_script  # the cloud runs nothing of ours at boot
    state = State.begin(tmp_path, RUN_ID, EXPIRY, None, {"provider": "lyceum"})
    run = L.Launch(p, SPEC, L.Policy(**{**FAST.__dict__, "spare": 0}), state, VmProber())
    (w,) = run.run({}).winners
    (box,) = State.load(tmp_path).winners
    assert (box["user"], box["port"]) == ("lyceum", 22)
    L.renew(p, state, run.prober, EXPIRY + 600)
    assert parse_tag(api.vms[w.instance_id]["display_name"]).expiry == EXPIRY + 600
    assert L.down(p, state, FAST) == [] and api.vms == {}
    assert gapi.vmbox is vmbox


def test_up_gives_the_probe_the_script_where_the_cloud_cannot_run_it(tmp_path, monkeypatch):
    from gpubox import api as gapi
    from gpubox.providers import vmbox

    seen = {}

    class Stop(Exception):
        pass

    def launch(provider, spec, policy, state, prober):
        seen["boot"] = prober.boot
        raise Stop

    p, _ = make()
    monkeypatch.setattr(gapi.Settings, "named_provider", lambda self: p)
    monkeypatch.setattr(gapi.L, "Launch", launch)
    spec = L.Spec(image="img:1", gpus=(), max_dph_per_gpu=3.0, ttl_min=60)
    with pytest.raises(Stop):
        gapi.up(spec, namespace="dev", name="ly", state_dir=tmp_path, provider="lyceum",
                teardown_on_failure=False)  # fmt: skip
    keys = gapi.key_dir(tmp_path, gapi.State.load(tmp_path).run)
    pub = (keys / "id_ed25519.pub").read_text().strip()
    assert seen["boot"] == vmbox.startup_script(pub, "img:1")


def test_cpu_and_ram_minimums_are_refused_since_lyceum_does_not_state_them():
    p, _ = make()
    with pytest.raises(ConfigError, match="CPUs or how much RAM"):
        p.candidates(L.Spec(image="i", gpus=(), max_dph_per_gpu=3.0, min_cpus=8), {}, L.Policy())
