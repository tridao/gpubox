"""Verda: its catalog, its VMs, its token, and a launch on it, against a fake of its API.

`FakeApi` stands in for `verda.Client`: it answers `call(method, path, ...)` as the API
does, from lists of instance types and a dict of instances, and `fail` makes chosen
calls answer otherwise first.
"""

import base64
import json
import time

import pytest
from fakes import FAST
from test_shadeform import VmProber

from gpubox import api as gapi
from gpubox import launch as L
from gpubox.errors import ConfigError
from gpubox.providers import base as prov
from gpubox.providers import verda as vd
from gpubox.state import State, make_tag, parse_tag

RUN_ID = "dev.verda.0123456789ab"
EXPIRY = int(time.time()) + 3600
SPEC = L.Spec(image="img:1", gpus=(), max_dph_per_gpu=3.0)
KEY = "ssh-ed25519 AAAA run"
DOCKER = ["24.04.cuda13.2.docker", "24.04.cuda13.4.docker", "26.04.cuda13.4.docker",
          "24.04.cuda13.4", "24.04.cuda12.9.docker"]  # fmt: skip


def uid(n: int) -> str:
    return f"00000000-0000-4000-8000-{n:012d}"


def itype(name, model, n=1, cpus=30, ram=120, vram=80, od=None, spot=None, os=DOCKER):
    """An instance type as /instance-types lists it, priced per location (per VM)."""
    prices = [{"location_code": loc, "price_per_hour": str(p), "spot_price": str(spot[loc])}
              for loc, p in (od or {}).items()]  # fmt: skip
    return {
        "instance_type": name, "model": model, "name": f"{model} 80GB",
        "gpu": {"number_of_gpus": n, "description": f"{n}x {model}"},
        "cpu": {"number_of_cores": cpus}, "memory": {"size_in_gigabytes": ram},
        "gpu_memory": {"size_in_gigabytes": vram * n}, "location_prices": prices,
        "supported_os": os,
    }  # fmt: skip


TYPES = [
    itype("1H100.80S.30V", "H100", od={"FIN-01": 5.29, "FIN-02": 5.29},
          spot={"FIN-01": 2.65, "FIN-02": 2.65}),
    itype("1A100.22V", "A100 80GB", od={"FIN-01": 2.41}, spot={"FIN-01": 1.21}),
    itype("1L40S.20V", "L40S", od={"FIN-02": 2.02}, spot={"FIN-02": 1.01}),
    itype("1RTX6000ADA.10V", "RTX 6000 Ada", cpus=10, od={"FIN-01": 1.62}, spot={"FIN-01": 0.81}),
    itype("1RTXPRO6000.30V.CC", "RTX PRO 6000 CC", od={"FIN-03": 2.92}, spot={"FIN-03": 1.46}),
    itype("2A100.44V", "A100 80GB", n=2, cpus=44, ram=240, od={"FIN-01": 4.82}, spot={"FIN-01": 2.42}),
    itype("1X.OLD", "A100 40GB", od={"FIN-01": 1.0}, spot={"FIN-01": 0.5}, os=["24.04.cuda12.9.docker"]),
    itype("CPU.4V.16G", "CPU", n=0, od={"FIN-01": 0.1}, spot={"FIN-01": 0.05}),
]  # fmt: skip
AVAILABLE = {  # (location, type) -> on demand, spot
    ("FIN-01", "1H100.80S.30V"): (False, True), ("FIN-02", "1H100.80S.30V"): (False, True),
    ("FIN-01", "1A100.22V"): (True, True), ("FIN-02", "1L40S.20V"): (True, False),
    ("FIN-01", "1RTX6000ADA.10V"): (True, True), ("FIN-03", "1RTXPRO6000.30V.CC"): (True, True),
    ("FIN-01", "1X.OLD"): (True, True), ("FIN-01", "2A100.44V"): (True, False),
}  # fmt: skip
IMAGES = [{"image_type": t, "is_cluster": False} for t in DOCKER]
LOCATIONS = [{"code": c, "country_code": "FI"} for c in ("FIN-01", "FIN-02", "FIN-03")]


def token(project="proj-1", life=600):
    claims = base64.urlsafe_b64encode(json.dumps({"project_id": project}).encode()).decode()
    return {"access_token": f"h.{claims.rstrip('=')}.s", "expires_in": life}


class FakeApi:
    def __init__(self):
        self.instances, self.keys, self.calls, self.fail = {}, [], [], {}
        self.n = 0

    def call(self, method, path, *, body=None, query=None, retries=2, timeout=None, as_text=False):
        self.calls.append((method, path, body, query, retries, as_text))
        parts = path.split("/")
        key = f"{method} {parts[0]}" + (f" {parts[-1]}" if len(parts) > 2 else "")
        if self.fail.get(key):
            answer = self.fail[key].pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        if path == "instance-types":
            return TYPES
        if path == "images":
            return IMAGES
        if path == "locations":
            return LOCATIONS
        if path == "instance-availability":
            i = 1 if query["is_spot"] == "true" else 0
            by = {}
            for (loc, t), tiers in AVAILABLE.items():
                if tiers[i]:
                    by.setdefault(loc, []).append(t)
            return [{"location_code": loc, "availabilities": ts} for loc, ts in by.items()]
        if path == "ssh-keys" and method == "GET":
            return list(self.keys)
        if path == "ssh-keys":
            assert vd.KEY_NAME in body["key"]  # not the run's key
            self.keys.append({"id": uid(800 + len(self.keys)), "name": body["name"]})
            return f'"{self.keys[-1]["id"]}"'
        if path == "instances" and method == "POST":
            self.n += 1
            iid = uid(9000 + self.n)
            self.instances[iid] = {
                "id": iid, "status": "running", "ip": f"203.0.113.{self.n}",
                "tags": [{"id": f"tag{self.n}", **body["tags"][0]}], "os_volume_id": uid(7000 + self.n),
                "is_spot": body["is_spot"], "body": body,
            }  # fmt: skip
            return iid
        if path == "instances" and method == "GET":
            return [dict(i) for i in self.instances.values() if i["status"] != "discontinued"]
        if path == "instances" and method == "PUT":
            inst = self.instances.get(body["id"])
            if inst is None:
                raise err(404, "not_found")
            inst["status"] = "discontinued" if inst["is_spot"] else "deleting"
            inst["deleted"] = body
            return [{"instanceId": body["id"], "action": "delete", "status": "success"}]
        inst = self.instances.get(parts[1])
        if inst is None:
            raise err(404, "not_found")
        if inst["status"] == "deleting" and len(parts) == 2 and inst.get("seen"):
            del self.instances[parts[1]]  # an on-demand VM is deleting, then gone
            raise err(404, "not_found")
        inst["seen"] = inst["status"] == "deleting"
        if parts[2:] == ["tags"]:
            self.n += 1
            inst["tags"].append({"id": f"tag{self.n}", **body})
            return inst["tags"][-1]
        if len(parts) == 4:
            inst["tags"] = [t for t in inst["tags"] if t["id"] != parts[3]]
            return None
        return dict(inst)

    def project(self):
        return "proj-1"


def err(status, code):
    e = vd.VerdaError(status, f"HTTP {status}: {code}")
    e.code = code
    return e


def make(api=None):
    api = api or FakeApi()
    return vd.Verda(api), api


def creates(api):
    return [c for c in api.calls if c[:2] == ("POST", "instances")]


# ------------------------------------------------------------------ offers


def test_offers_are_types_in_stock_in_a_location_cheapest_first_with_the_newest_docker_image():
    p, api = make()
    offers, note = p.candidates(SPEC, {}, L.Policy(spare=0))
    # On demand only: H100 has spot stock alone, the RTX 6000 Ada too few CPUs, the CC type
    # images of its own, the old type no CUDA 13; the 2-GPU type is not what was asked.
    assert [(o["_type"], o["_location"], o["dph_total"]) for o in offers] == [
        ("1L40S.20V", "FIN-02", 2.02), ("1A100.22V", "FIN-01", 2.41),
    ]  # fmt: skip
    o = offers[0]
    assert o["_image"] == "24.04.cuda13.4.docker" and o["cuda_max_good"] == 13.4  # a tie: 24.04
    assert o["failure_domain"] == "verda/FIN-02" and o["_spot"] is False and "Verda" in note
    assert o["boot_s"] == vd.BOOT_S


def test_spot_vms_only_for_an_interruptible_box_and_then_beside_the_others():
    p, _ = make()
    spot = L.Spec(image="i", gpus=(), max_dph_per_gpu=3.0, interruptible=True)
    offers = p.candidates(spot, {}, L.Policy(spare=0))[0]
    assert [(o["_type"], o["_location"], o["_spot"]) for o in offers] == [
        ("1A100.22V", "FIN-01", True), ("1L40S.20V", "FIN-02", False),
        ("1A100.22V", "FIN-01", False), ("1H100.80S.30V", "FIN-01", True),
        ("1H100.80S.30V", "FIN-02", True),
    ]  # fmt: skip
    assert offers[0]["geolocation"] == "FIN-01 (spot)" and offers[0]["dph_total"] == 1.21
    tie = L.Spec(image="i", gpus=("A100",), max_dph_per_gpu=1.0, interruptible=True, min_cpus=1)
    TYPES.append(itype("1A.TIE", "A100 80GB", od={"FIN-01": 1.0}, spot={"FIN-01": 1.0}))
    AVAILABLE[("FIN-01", "1A.TIE")] = (True, True)
    try:
        offers = p.candidates(tie, {}, L.Policy(spare=0))[0]
        assert [o["_spot"] for o in offers if o["_type"] == "1A.TIE"] == [False, True]
    finally:
        TYPES.pop()
        del AVAILABLE[("FIN-01", "1A.TIE")]


def test_the_models_asked_for_the_count_the_cap_and_the_countries():
    p, _ = make()
    two = L.Spec(image="i", gpus=("A100",), max_dph_per_gpu=3.0, num_gpus=2)
    assert [o["_type"] for o in p.candidates(two, {}, L.Policy(spare=0))[0]] == ["2A100.44V"]
    cheap = L.Spec(image="i", gpus=("L40S",), max_dph_per_gpu=2.0)
    assert p.candidates(cheap, {}, L.Policy())[0] == []
    away = L.Spec(image="i", gpus=(), max_dph_per_gpu=3.0, exclude_countries=("FI",))
    assert p.candidates(away, {}, L.Policy())[0] == []
    LOCATIONS[0]["country_code"] = ""  # not stated: not known to be allowed
    try:
        some = L.Spec(image="i", gpus=(), max_dph_per_gpu=3.0, exclude_countries=("US",))
        assert {o["_location"] for o in p.candidates(some, {}, L.Policy())[0]} == {"FIN-02"}
        anywhere = L.Spec(image="i", gpus=(), max_dph_per_gpu=3.0, exclude_countries=())
        assert {o["_location"] for o in p.candidates(anywhere, {}, L.Policy())[0]} >= {"FIN-01"}
    finally:
        LOCATIONS[0]["country_code"] = "FI"


def test_offers_are_cheapest_first_across_the_slots_of_a_race():
    p, _ = make()
    offers = p.candidates(SPEC, {}, L.Policy(spare=1))[0]
    assert [o["dph_total"] for o in offers] == [2.02, 2.02, 2.41, 2.41]


def test_a_type_that_does_not_state_a_size_does_not_meet_a_minimum():
    assert not vd.Verda._fits(SPEC, {"cpu": {}, "memory": {"size_in_gigabytes": 200}})
    big = L.Spec(image="i", gpus=(), max_dph_per_gpu=3.0, min_vram_mib=90 * 1024)
    assert not vd.Verda._fits(big, TYPES[0]) and vd.Verda._fits(SPEC, TYPES[0])


# ------------------------------------------------------------------ the create


def test_a_vm_is_created_once_with_our_tag_the_box_script_and_a_volume_that_goes_with_it():
    p, api = make()
    spot = L.Spec(image="img:1", gpus=("A100",), max_dph_per_gpu=3.0, interruptible=True)
    offer = p.candidates(spot, {}, L.Policy())[0][0]
    tag = make_tag(RUN_ID, 1, EXPIRY)
    iid = p.create(offer, spot, tag, "ssh-ed25519 AAAA run-key", EXPIRY)
    ((_, _, body, _, retries, as_text),) = creates(api)
    assert (retries, as_text) == (0, True) and iid in api.instances  # the answer is the bare id
    assert body["tags"] == [{"key": "gpubox", "value": tag}] and body["description"] == tag
    assert body["is_spot"] is True and body["instance_type"] == "1A100.22V"
    assert body["image"] == "24.04.cuda13.4.docker" and body["location_code"] == "FIN-01"
    assert body["os_volume"] == {"name": vd.short_name(tag), "size": 50,  # at least Verda's
                                 "on_spot_discontinue": "delete_permanently"}  # fmt: skip
    assert body["startup_script"].startswith("#!") and "run-key" in body["startup_script"]
    assert body["ssh_key_ids"] == [api.keys[0]["id"]] and "." not in body["hostname"]
    on_demand = [o for o in p.candidates(spot, {}, L.Policy())[0] if not o["_spot"]][0]
    p.create(on_demand, spot, make_tag(RUN_ID, 2, EXPIRY), KEY, EXPIRY)
    assert len(api.keys) == 1  # the placeholder is registered once
    body = creates(api)[-1][2]  # Verda refuses a spot removal policy on demand
    assert body["is_spot"] is False and "on_spot_discontinue" not in body["os_volume"]


@pytest.mark.parametrize(
    "answer, outcome",
    [
        (err(503, None), prov.CreateRefused),  # no capacity there
        (err(400, "invalid_request"), prov.CreateRefused),
        (err(402, "insufficient_funds"), ConfigError),
        (err(403, "forbidden_action"), ConfigError),
        (err(500, "server_error"), prov.CreateUnknown),
        (vd.VerdaError(0, "timed out"), prov.CreateUnknown),
        ("not an id", prov.CreateUnknown),
        ("", prov.CreateUnknown),
        (f'"{uid(5)}"', uid(5)),
        (uid(5), uid(5)),
    ],
)
def test_an_instance_id_only_from_an_explicit_yes(answer, outcome):
    p, api = make()
    api.fail["POST instances"] = [answer]
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    if isinstance(outcome, str):
        assert p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), KEY, EXPIRY) == outcome
    else:
        with pytest.raises(outcome):
            p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), KEY, EXPIRY)


def test_no_ssh_key_to_be_had_means_nothing_was_rented():
    p, api = make()
    api.fail["GET ssh-keys"] = [err(500, "server_error")]
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    with pytest.raises(prov.CreateRefused, match="SSH key"):
        p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), KEY, EXPIRY)
    assert creates(api) == []


# ------------------------------------------------------------------ the instances


def row(n, status="running", tags=None, **kw):
    tags = (
        [{"id": "t", "key": "gpubox", "value": make_tag(RUN_ID, n, EXPIRY)}]
        if tags is None
        else tags
    )
    return {
        "id": uid(n),
        "status": status,
        "ip": "203.0.113.9",
        "tags": tags,
        "os_volume_id": uid(70 + n),
        **kw,
    }


def test_a_discontinued_vm_is_gone_and_an_error_never_is():
    p, api = make()
    assert p.instance(uid(1)) is None
    api.instances[uid(1)] = row(1, is_spot=True)
    assert parse_tag(p.instance(uid(1))["label"]).attempt == 1
    for status in ("discontinued", "notfound"):  # a deleted or evicted spot VM
        api.instances[uid(1)]["status"] = status
        assert p.instance(uid(1)) is None
    api.instances[uid(1)]["status"] = "deleting"  # not known to have stopped billing
    assert p.instance(uid(1)) is not None
    two = row(1, tags=[{"id": "a", "key": "gpubox", "value": make_tag(RUN_ID, 1, EXPIRY)},
                       {"id": "b", "key": "gpubox", "value": make_tag(RUN_ID, 2, EXPIRY)}])  # fmt: skip
    for answer in (err(500, "server_error"), vd.VerdaError(0, "x"), {"id": uid(2)}, two,
                   {**row(1), "tags": "x"}, [], {k: v for k, v in row(1).items() if k != "tags"}):  # fmt: skip
        api.fail["GET instances"] = [answer]
        with pytest.raises(vd.VerdaError):
            p.instance(uid(1))


def test_the_listing_is_every_instance_or_an_error():
    p, api = make()
    api.instances = {uid(i): row(i) for i in (1, 2, 3)}
    api.instances[uid(3)]["status"] = "discontinued"
    assert [r["id"] for r in p.instances()] == [uid(1), uid(2)]
    for answer in ([row(1), row(1)], [{"id": "x"}], {"instances": []}):
        api.fail["GET instances"] = [answer]
        with pytest.raises(vd.VerdaError):
            p.instances()


def test_a_delete_takes_the_os_volume_with_it_and_a_refusal_is_an_error():
    p, api = make()
    api.instances[uid(1)] = row(1, is_spot=False)
    p.destroy(uid(1))
    assert api.instances[uid(1)]["deleted"] == {
        "action": "delete", "id": uid(1), "volume_ids": [uid(71)], "delete_permanently": True,
    }  # fmt: skip
    p.destroy(uid(2))  # gone: nothing to do
    assert [c for c in api.calls if c[0] == "PUT"] == [api.calls[1]]
    api.instances[uid(3)] = row(3, is_spot=False)
    api.fail["PUT instances"] = [[{"instanceId": uid(3), "status": "error", "error": "busy"}]]
    with pytest.raises(vd.VerdaError, match="busy"):
        p.destroy(uid(3))
    # A lookup that fails still sends the delete: the OS volume goes with it by default.
    api.fail["GET instances"] = [err(500, "server_error")]
    p.destroy(uid(3))
    assert api.instances[uid(3)]["deleted"] == {"action": "delete", "id": uid(3)}


def test_a_retag_adds_the_new_tag_before_it_removes_the_old_and_is_read_back():
    p, api = make()
    api.instances[uid(1)] = row(1)
    new = make_tag(RUN_ID, 1, EXPIRY + 600)
    p.relabel(uid(1), new)
    assert [t["value"] for t in api.instances[uid(1)]["tags"]] == [new]
    ops = [(c[0], len(c[1].split("/"))) for c in api.calls if "tags" in c[1]]
    assert ops == [("POST", 3), ("DELETE", 4)]
    api.fail["POST instances tags"] = [{}]  # said yes, did nothing
    with pytest.raises(vd.VerdaError):
        p.relabel(uid(1), make_tag(RUN_ID, 1, EXPIRY + 900))


def test_a_retag_that_did_not_take_leaves_the_old_tag():
    p, api = make()
    api.instances[uid(1)] = row(1)
    old = api.instances[uid(1)]["tags"][0]["value"]
    api.fail["POST instances tags"] = [{}]  # said yes, did nothing
    with pytest.raises(vd.VerdaError, match="not confirmed"):
        p.relabel(uid(1), make_tag(RUN_ID, 1, EXPIRY + 900))
    assert [t["value"] for t in api.instances[uid(1)]["tags"]] == [old]


def test_a_retag_cut_short_reads_as_the_earlier_lease_and_a_retag_settles_it():
    p, api = make()
    early, late = make_tag(RUN_ID, 1, EXPIRY), make_tag(RUN_ID, 1, EXPIRY + 600)
    api.instances[uid(1)] = row(1, tags=[{"id": "a", "key": "gpubox", "value": late},
                                         {"id": "b", "key": "gpubox", "value": early}])  # fmt: skip
    assert p.instance(uid(1))["label"] == early and [r["id"] for r in p.instances()] == [uid(1)]
    p.relabel(uid(1), late)
    assert [t["value"] for t in api.instances[uid(1)]["tags"]] == [late]


@pytest.mark.parametrize(
    "tags",
    [
        [{"id": "a", "key": "gpubox", "value": make_tag(RUN_ID, 1, EXPIRY)},
         {"id": "b", "key": "gpubox", "value": make_tag(RUN_ID, 2, EXPIRY)}],  # two attempts
        [{}], [{"key": "gpubox"}], [{"id": "a", "value": "x"}], ["gpubox"],
    ],
)  # fmt: skip
def test_tags_that_do_not_say_whose_it_is_make_the_row_unreadable(tags):
    assert vd._row(row(1, tags=tags)) is None
    assert vd._row(row(1, tags=[{"id": "a", "key": "team", "value": ""}]))["label"] is None


def test_only_verdas_word_that_there_is_no_such_vm_is_gone():
    p, api = make()
    for answer in (vd.VerdaError(404, "<html>"), err(404, "server_error")):
        api.fail["GET instances"] = [answer]
        with pytest.raises(vd.VerdaError):
            p.instance(uid(1))


def test_a_token_that_cannot_be_had_is_no_answer_about_the_call(monkeypatch):
    monkeypatch.setattr(vd.httpapi, "Client", FakeHttp)
    FakeHttp.log, FakeHttp.n, FakeHttp.life, FakeHttp.expired = [], 0, 600, False
    FakeHttp.token_error = vd.VerdaError(404, "no such route")
    try:
        with pytest.raises(vd.VerdaError) as exc:
            vd.Verda(vd.Client("id", "s")).instance(uid(1))
        assert exc.value.status == 0 and [p for p, _ in FakeHttp.log] == ["oauth2/token"]
    finally:
        FakeHttp.token_error = None


def test_a_vm_is_running_for_us_once_it_has_a_public_ip_reached_as_ubuntu():
    p, _ = make()
    r = vd._row(row(1))
    assert (p.view(r).state, p.view(r).endpoint) == ("running", ("203.0.113.9", 22, "ubuntu"))
    r["ip"] = None
    assert p.view(r).state == "loading"
    for status in ("ordered", "provisioning", "deleting"):
        r["actual_status"] = status
        assert p.view(r).state == "loading"
    for status in ("error", "installation_failed", "offline", "discontinued"):
        r["actual_status"] = status
        assert p.view(r).state == "exited"


# ------------------------------------------------------------------ the token


class FakeHttp:
    """httpapi.Client, as the token client and the API client both see it."""

    log = []
    token_error = None

    def __init__(self, base, headers, error, timeout=30):
        self.headers = headers

    def call(self, method, path, **kw):
        FakeHttp.log.append((path, self.headers.get("Authorization")))
        if path == "oauth2/token":
            if FakeHttp.token_error:
                raise FakeHttp.token_error
            if kw["body"]["client_secret"] != "s":
                raise vd.VerdaError(401, "unauthorized")
            FakeHttp.n += 1
            return token(life=FakeHttp.life) | {
                "access_token": f"h.{token()['access_token'].split('.')[1]}.{FakeHttp.n}"
            }
        if FakeHttp.expired:
            FakeHttp.expired = False
            raise vd.VerdaError(401, "token expired")
        return []


def test_the_token_comes_from_the_client_credentials_is_kept_and_renewed(monkeypatch):
    monkeypatch.setattr(vd.httpapi, "Client", FakeHttp)
    FakeHttp.log, FakeHttp.n, FakeHttp.life, FakeHttp.expired = [], 0, 600, False
    c = vd.Client("id", "s")
    c.call("GET", "instances")
    c.call("GET", "instances")
    assert [p for p, _ in FakeHttp.log] == ["oauth2/token", "instances", "instances"]
    FakeHttp.expired = True  # refused before anything was done: renewed, and once more
    c.call("GET", "instances")
    assert [p for p, _ in FakeHttp.log[3:]] == ["instances", "oauth2/token", "instances"]
    assert FakeHttp.log[-1][1].endswith(".2") and c.project() == "proj-1"
    FakeHttp.life = 10  # a short life is still kept for 30 s
    c._until = 0
    c.call("GET", "instances")
    c.call("GET", "instances")
    assert sum(p == "oauth2/token" for p, _ in FakeHttp.log) == 3
    with pytest.raises(ConfigError, match="client credentials"):
        vd.Client("id", "wrong").call("GET", "instances")


def test_the_client_id_and_secret_come_from_the_environment_or_the_keys_file(tmp_path, monkeypatch):
    keys = tmp_path / "keys.env"
    keys.write_text("VERDA_CLIENT_ID=the-id\nVERDA_CLIENT_SECRET='the-secret'\n")
    for name in vd.Verda.credential:
        monkeypatch.delenv(name, raising=False)
    s = gapi.Settings(state_dir=tmp_path, provider="verda", keys_file=str(keys))
    provider = s.named_provider()
    assert provider.client._creds == {"client_id": "the-id", "client_secret": "the-secret"}
    s = gapi.Settings(state_dir=tmp_path, provider="verda", keys_file=str(tmp_path / "none"))
    with pytest.raises(ConfigError, match="VERDA_CLIENT_ID"):
        s.named_provider()


# ------------------------------------------------------------------ a launch


def launch(tmp_path, spec=SPEC, **kw):
    p, api = make()
    state = State.begin(tmp_path, RUN_ID, EXPIRY, None, {"provider": "verda"})
    policy = L.Policy(**{**FAST.__dict__, **kw})
    return L.Launch(p, spec, policy, state, VmProber()), p, api, state


def test_a_launch_rents_a_vm_hands_it_over_and_down_deletes_it_by_id_and_by_tag(tmp_path):
    run, p, api, state = launch(tmp_path, spare=0)
    (w,) = run.run({}).winners
    (box,) = State.load(tmp_path).winners
    assert (box["host"], box["port"], box["user"]) == (
        api.instances[w.instance_id]["ip"],
        22,
        "ubuntu",
    )
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    stray = p.create(offer, SPEC, make_tag(RUN_ID, 9, EXPIRY), KEY, EXPIRY)
    other = p.create(offer, SPEC, make_tag("dev.other.0123456789ab", 1, EXPIRY), KEY, EXPIRY)
    assert L.down(p, state, FAST) == []
    assert w.instance_id not in api.instances and stray not in api.instances
    assert api.instances[other]["status"] == "running"


def test_renew_moves_the_tag_and_touches_nothing_on_the_box(tmp_path):
    run, p, api, state = launch(tmp_path, spare=0)
    run.run({})
    later = EXPIRY + 7200
    run.prober.run = lambda *a, **k: 255
    L.renew(p, state, run.prober, later)
    assert run.prober.ran == []
    (inst,) = api.instances.values()
    assert [parse_tag(t["value"]).expiry for t in inst["tags"]] == [later]
