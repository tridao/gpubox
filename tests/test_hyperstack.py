"""Hyperstack: its catalog, its VMs, and a launch on it, against a fake of its API."""

import base64
import time

import pytest
from fakes import FAST
from test_shadeform import VmProber

from gpubox import launch as L
from gpubox import provider as prov
from gpubox import hyperstack as hs
from gpubox import vmbox
from gpubox.errors import ConfigError
from gpubox.state import State, make_tag, parse_tag

RUN_ID = "dev.hs.0123456789ab"
EXPIRY = int(time.time()) + 3600
SPEC = L.Spec(image="img:1", gpus=(), max_dph_per_gpu=1.0)


def flavor(name, gpu, region, stock=True, n=1, **kw):
    return {"name": name, "gpu": gpu, "gpu_count": n, "region_name": region,
            "stock_available": stock, "cpu": 28, "ram": 58.0, "disk": 100, **kw}  # fmt: skip


FLAVORS = [
    flavor("n3-RTX-A6000x1", "RTX-A6000", "CANADA-1"),
    flavor("n3-L40x1-spot", "L40-spot", "CANADA-1"),
    flavor("n3-A4000x1", "A4000", "NORWAY-1"),  # Norway's newest image is CUDA 12.8
    flavor("n3-H100x1", "H100-80G-PCIe", "CANADA-1"),  # above the cap
    flavor("n3-A100x1", "A100-80G-PCIe", "CANADA-1", stock=False),
]
PRICES = [{"name": "RTX-A6000", "value": "0.500000000"}, {"name": "L40-spot", "value": "0.8"},
          {"name": "A4000", "value": "0.150000000"}, {"name": "H100-80G-PCIe", "value": "1.9"},
          {"name": "A100-80G-PCIe", "value": "1.35"}]  # fmt: skip
IMAGES = [
    {"region_name": "CANADA-1", "images": [
        {"name": "Ubuntu Server 22.04 LTS R535 CUDA 12.2 with Docker"},
        {"name": "Ubuntu Server 24.04 LTS R595 CUDA 13.2 with Docker"},
        {"name": "Ubuntu Server 24.04 LTS R595 CUDA 13.2"}]},
    {"region_name": "NORWAY-1", "images": [{"name": "Ubuntu Server 22.04 LTS R570 CUDA 12.8 with Docker"}]},
]  # fmt: skip


class FakeApi:
    def __init__(self):
        self.vms, self.calls, self.fail = {}, [], {}
        self.keypairs, self.next, self.credit = [], 1000, True

    def call(self, method, path, *, body=None, query=None, retries=2, timeout=None):
        self.calls.append((method, path, body, query, retries))
        key = (
            f"{method} {path}" if not path.startswith("core/virtual-machines/") else f"{method} vm"
        )
        if self.fail.get(key):
            answer = self.fail[key].pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        ok = {"status": True, "message": "ok"}
        if path == "billing/user-credit/credit":
            return {**ok, "data": {"credit": 5.0, "can_create_instance": self.credit}}
        if path == "core/flavors":
            return {**ok, "data": [{"flavors": FLAVORS}]}
        if path == "pricebook":
            return PRICES
        if path == "core/images":
            return {**ok, "images": IMAGES}
        if path == "core/environments":
            return {**ok, "environments": [{"name": f"default-{r}", "region": r}
                                           for r in ("CANADA-1", "NORWAY-1", "US-1")]}  # fmt: skip
        if path == "core/keypairs" and method == "GET":
            return {**ok, "keypairs": self.keypairs}
        if path == "core/keypairs":
            assert "gpubox-placeholder" in body["public_key"]  # not the run's key
            self.keypairs.append(
                {"name": body["name"], "environment": {"name": body["environment_name"]}}
            )
            return ok
        if path == "auth/me":
            return {**ok, "user": {"email": "me@example.com"}}
        if path == "core/virtual-machines" and method == "POST":
            self.next += 1
            vm = {"id": self.next, "name": body["name"], "status": "CREATING", "labels": body["labels"],
                  "floating_ip": None, "body": body}  # fmt: skip
            self.vms[vm["id"]] = vm
            return {**ok, "instances": [dict(vm)]}
        if path == "core/virtual-machines":
            rows = list(self.vms.values())
            start = (query["page"] - 1) * query["pageSize"]
            page = rows[start : start + query["pageSize"]]
            return {**ok, "page": query["page"], "count": len(page), "instances": page}
        vid = int(path.split("/")[2])
        vm = self.vms.get(vid)
        if vm is None:
            raise hs.HyperstackError(404, f"{method} /{path}: HTTP 404: not found")
        if path.endswith("/label"):
            vm["labels"] = body["labels"]
            return ok
        if method == "DELETE":
            del self.vms[vid]
            return ok
        if vm["status"] == "CREATING":
            vm.update(status="ACTIVE", vm_state="active", floating_ip=f"38.0.0.{vid % 250}")
        return {**ok, "instance": dict(vm)}


def make():
    api = FakeApi()
    return hs.Hyperstack(api), api


def test_offers_are_flavors_in_stock_in_a_region_with_a_docker_image_new_enough():
    p, _ = make()
    offers, note = p.candidates(SPEC, {}, L.Policy(spare=1))
    assert [(o["_flavor"], o["_region"]) for o in offers] == [("n3-RTX-A6000x1", "CANADA-1")] * 2
    o = offers[0]
    assert o["_image"] == "Ubuntu Server 24.04 LTS R595 CUDA 13.2 with Docker"
    assert (o["cuda_max_good"], o["dph_total"], o["_env"]) == (13.2, 0.5, "default-CANADA-1")
    older = L.Spec(image="i", gpus=(), max_dph_per_gpu=1.0, min_cuda=12.8)
    assert {o["_flavor"] for o in p.candidates(older, {}, L.Policy())[0]} == {
        "n3-RTX-A6000x1",
        "n3-A4000x1",
    }


def test_an_account_that_cannot_create_says_so_before_anything():
    p, api = make()
    api.credit = False
    with pytest.raises(ConfigError, match="credit"):
        p.candidates(SPEC, {}, L.Policy())


def test_a_vm_is_created_once_with_our_tag_the_box_script_and_ssh_on_22():
    p, api = make()
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    tag = make_tag(RUN_ID, 1, EXPIRY)
    vid = p.create(offer, SPEC, tag, "ssh-ed25519 AAAA gpubox", EXPIRY)
    method, path, body, _, retries = api.calls[-1]
    assert (method, path, retries) == ("POST", "core/virtual-machines", 0) and vid in api.vms
    assert body["labels"] == [tag] and body["count"] == 1 and body["assign_floating_ip"] is True
    assert body["enable_port_randomization"] is False and len(body["name"]) <= 50
    assert base64.b64decode(body["user_data"]).decode() == vmbox.startup_script(
        "ssh-ed25519 AAAA gpubox", "img:1"
    )
    # A placeholder whose private half is gone: never the run's key, nor an earlier run's.
    (pair,) = api.keypairs
    assert pair["name"] == "gpubox-placeholder" and pair["environment"] == {
        "name": "default-CANADA-1"
    }
    p.create(offer, SPEC, make_tag(RUN_ID, 2, EXPIRY), "ssh-ed25519 BBBB other", EXPIRY)
    assert len(api.keypairs) == 1  # one is enough: the run's key goes in with the script


@pytest.mark.parametrize(
    "answer, outcome",
    [
        (hs.HyperstackError(400, "flavor out of stock"), prov.CreateRefused),
        (hs.HyperstackError(405, "Not enough credit"), ConfigError),
        (hs.HyperstackError(401, "bad key"), ConfigError),
        (hs.HyperstackError(500, "boom"), prov.CreateUnknown),
        (hs.HyperstackError(0, "timed out"), prov.CreateUnknown),
        ({"status": True, "instances": []}, prov.CreateUnknown),
        ({"status": True, "instances": [{"id": 1}, {"id": 2}]}, prov.CreateUnknown),
        ({"status": False, "instances": [{"id": 1}]}, prov.CreateUnknown),
        ({"status": True, "instances": [{"id": 77}]}, 77),
        (
            {"status": True, "instances": [{"id": 77, "labels": ["someone-else"]}]},
            prov.CreateUnknown,
        ),
    ],
)
def test_a_vm_id_only_from_an_explicit_yes(answer, outcome):
    p, api = make()
    api.fail["POST core/virtual-machines"] = [answer]
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    if isinstance(outcome, int):
        assert (
            p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "ssh-ed25519 K", EXPIRY) == outcome
        )
    else:
        with pytest.raises(outcome):
            p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "ssh-ed25519 K", EXPIRY)


def test_no_key_pair_to_be_had_means_nothing_was_rented():
    p, api = make()
    api.fail["GET core/keypairs"] = [hs.HyperstackError(500, "down")]
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    with pytest.raises(prov.CreateRefused):
        p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "ssh-ed25519 K", EXPIRY)
    assert api.vms == {}


def test_a_vm_on_its_way_out_is_gone_and_an_error_never_is():
    p, api = make()
    assert p.instance(5) is None
    api.vms[5] = {"id": 5, "status": "ACTIVE", "labels": []}
    assert p.instance(5)["label"] is None
    api.vms[5]["status"] = "DELETED"
    assert p.instance(5) is None
    api.vms[5]["status"] = "DELETING"  # still there, and not known to have stopped billing
    assert p.instance(5) is not None
    api.vms[5]["status"] = "ACTIVE"
    for answer in (
        hs.HyperstackError(500, "x"),
        {"status": False},
        {"status": True, "instance": {"id": 6, "status": "ACTIVE"}},
    ):
        api.fail["GET vm"] = [answer]
        with pytest.raises(hs.HyperstackError):
            p.instance(5)


def test_the_listing_reads_every_page_or_fails():
    p, api = make()
    tag = make_tag(RUN_ID, 1, EXPIRY)
    for i in range(1, hs.PAGE + 6):
        api.vms[i] = {"id": i, "status": "ACTIVE", "labels": [tag] if i == 3 else None}
    rows = p.instances()
    assert len(rows) == hs.PAGE + 5 and [r["id"] for r in rows if r["label"]] == [3]
    for bad in ({"status": True, "page": 2, "instances": []},  # another page than asked for
                {"status": True, "instances": [{"id": 1, "status": "ACTIVE"}] * 2},
                {"status": False, "instances": []},
                {"status": True, "instances": [{"id": "x", "status": "ACTIVE"}]}):  # fmt: skip
        api.fail["GET core/virtual-machines"] = [bad]
        with pytest.raises(hs.HyperstackError):
            p.instances()


def test_a_relabel_is_believed_once_read_back():
    p, api = make()
    api.vms[5] = {"id": 5, "status": "ACTIVE", "labels": ["x"]}
    tag = make_tag(RUN_ID, 1, EXPIRY + 60)
    p.relabel(5, tag)
    assert api.vms[5]["labels"] == [tag]
    api.fail["PUT vm"] = [{"status": True}]
    with pytest.raises(hs.HyperstackError, match="not confirmed"):
        p.relabel(5, make_tag(RUN_ID, 1, EXPIRY + 120))


def test_a_vm_is_running_for_us_once_it_has_a_public_ip():
    p, _ = make()
    row = hs._row({"id": 5, "status": "ACTIVE", "floating_ip": None, "labels": None})
    assert p.view(row).state == "loading"
    row["floating_ip"] = "203.0.113.3"
    assert p.view(row).endpoint == ("203.0.113.3", 22, "ubuntu")
    for status, state in (("CREATING", "loading"), ("ERROR", "exited"), ("SHUTOFF", "exited")):
        row["actual_status"] = status
        assert p.view(row).state == state


def test_a_launch_rents_a_vm_hands_it_over_and_down_deletes_it_by_tag(tmp_path):
    p, api = make()
    state = State.begin(tmp_path, RUN_ID, EXPIRY, None, {"provider": "hyperstack"})
    run = L.Launch(p, SPEC, L.Policy(**{**FAST.__dict__, "spare": 0}), state, VmProber())
    (w,) = run.run({}).winners
    (box,) = State.load(tmp_path).winners
    assert (box["id"], box["user"], box["port"]) == (w.instance_id, "ubuntu", 22)
    later = EXPIRY + 600
    L.renew(p, state, run.prober, later)
    assert parse_tag(api.vms[w.instance_id]["labels"][0]).expiry == later
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    stray = p.create(offer, SPEC, make_tag(RUN_ID, 9, EXPIRY), "ssh-ed25519 K", EXPIRY)
    assert L.down(p, state, FAST) == [] and stray not in api.vms and api.vms == {}


def test_a_flavor_that_does_not_state_a_size_does_not_meet_a_minimum():
    p, _ = make()
    spec = L.Spec(image="i", gpus=(), max_dph_per_gpu=1.0)
    assert p._fits(
        spec, flavor("x", "A4000", "CANADA-1")
    )  # 28 CPUs, 58 GB: the defaults are 16, 32
    assert not p._fits(spec, {**flavor("x", "A4000", "CANADA-1"), "ram": None})
    assert not p._fits(spec, flavor("x", "A4000", "CANADA-1", cpu=8))


def test_a_listing_page_that_says_otherwise_than_it_shows_is_an_error():
    p, api = make()
    api.fail["GET core/virtual-machines"] = [
        {"status": True, "page": 1, "count": 2, "instances": []}
    ]
    with pytest.raises(hs.HyperstackError):
        p.instances()


def test_a_teardown_that_did_not_finish_says_how_to_finish_it_on_this_provider(capsys):
    p, _ = make()
    state = State.ephemeral(RUN_ID)
    state.data.update(expiry=EXPIRY)
    L.explain_leftovers(p, state, [5])
    out = capsys.readouterr().out
    assert "gpubox down --provider hyperstack --run-id" in out
    assert "gpubox reap --provider hyperstack --namespace dev --destroy" in out
    assert "destroys itself" not in out and "Nothing else ends them" in out
