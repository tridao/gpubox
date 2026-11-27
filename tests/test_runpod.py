"""RunPod: how its API v2 is asked and read, a launch on it, and its CLI options.

`FakeApi` stands in for `runpod.Client`: it answers `call(method, path, ...)` as the API
does, from a dict of pods, and `fail` makes chosen calls answer otherwise first.
"""

import time

import pytest
from fakes import FAST, FakeCheck

from gpubox import cli
from gpubox import launch as L
from gpubox.providers import base as prov
from gpubox.providers import runpod as rp
from gpubox.errors import ConfigError
from gpubox.state import State, make_tag, parse_tag

RUN_ID = "dev.pod.0123456789ab"
EXPIRY = int(time.time()) + 3600
SPEC = L.Spec(image="img:1", gpus=("RTX 4090",), max_dph_per_gpu=1.0)

CATALOG = [
    {"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "availability": "HIGH",
     "price": {"secure": 0.69, "community": 0.34}, "maxCount": {"secure": 8, "community": 8},
     "dataCenters": [{"id": "EU-RO-1", "availability": "HIGH"},
                     {"id": "US-TX-3", "availability": "LOW"},
                     {"id": "CA-MTL-1", "availability": "NONE"}]},
    {"id": "NVIDIA H100 80GB HBM3", "name": "H100 SXM", "availability": "LOW",
     "price": {"secure": 2.99}, "maxCount": {"secure": 8},
     "dataCenters": [{"id": "US-KS-2", "availability": "LOW"}]},
    {"id": "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb", "name": "MIG",
     "availability": "HIGH", "price": {"secure": 0.1}, "maxCount": {"secure": 1},
     "dataCenters": [{"id": "US-PA-1", "availability": "HIGH"}]},
    {"id": "NVIDIA RTX A6000", "name": "RTX A6000", "availability": "NONE",
     "price": {"secure": 0.49}, "maxCount": {"secure": 8}, "dataCenters": []},
]  # fmt: skip


class FakeApi:
    def __init__(self, behaviour=None):
        self.pods, self.calls, self.fail = {}, [], {}
        self.behaviour = behaviour or {}  # tier or data center -> "ok" | "full" | "noaddr"
        self.catalogs = {}  # per-tier overrides of the catalog snapshot
        self.next = 0
        self.page = 1000

    def call(self, method, path, *, body=None, query=None, retries=2, timeout=None):
        self.calls.append((method, path, body, query, retries))
        key = f"{method} {path.split('/')[0]}"
        if self.fail.get(key):
            answer = self.fail[key].pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        if path == "catalog/gpus":
            return {"gpus": self.catalogs.get(query["cloud"], CATALOG)}
        if path == "account":
            return {"id": "acct1", "email": None}
        if method == "POST" and path == "pods":
            dc = (body.get("dataCenterIds") or ["any"])[0]
            if self.behaviour.get(body["cloud"], self.behaviour.get(dc)) == "full":
                raise rp.RunpodError(
                    400, "POST /pods: HTTP 400: Bad Request: no longer any instances available"
                )
            self.next += 1
            pid = f"pod{self.next:04d}xyz"
            self.pods[pid] = {"id": pid, "name": body["name"], "status": "RUNNING", "dc": dc,
                              "ssh": {"direct": None}, "body": body}  # fmt: skip
            return dict(self.pods[pid])
        if path == "pods" and method == "GET":
            rows = list(self.pods.values())
            start = int(query.get("cursor") or 0)
            page = rows[start : start + self.page]
            more = start + self.page < len(rows)
            return {"pods": page, "pagination": {"hasNextPage": more,
                                                 "nextCursor": str(start + self.page) if more else None}}  # fmt: skip
        pid = path.split("/")[1]
        if pid not in self.pods:
            raise rp.RunpodError(404, f"{method} /{path}: HTTP 404: Not Found: pod not found")
        pod = self.pods[pid]
        if method == "GET":
            cloud = pod.get("body", {}).get("cloud")
            if self.behaviour.get(cloud, self.behaviour.get(pod.get("dc"))) != "noaddr":
                n = int(pid[3:7]) if pid.startswith("pod") else 1
                pod["ssh"] = {"direct": {"host": f"10.0.0.{n}", "port": 40000 + n}}
            return dict(pod)
        if method == "DELETE":
            del self.pods[pid]
            return None
        if method == "PATCH":
            pod.update(body)
            return dict(pod)
        raise AssertionError((method, path))


def make(api=None, **kw):
    api = api or FakeApi()
    return rp.RunPod(api, **kw), api


# ------------------------------------------------------------------ offers


def test_offers_are_gpu_types_in_stock_per_data_center_cheapest_first_within_the_cap():
    p, api = make(cloud="SECURE")
    spec = L.Spec(image="i", gpus=(), max_dph_per_gpu=3.0, num_gpus=2, min_cpus=16, min_ram_gb=100)
    offers, note = p.candidates(spec, {}, L.Policy(spare=1))
    _, path, _, query, _ = api.calls[-1]
    assert path == "catalog/gpus" and query["count"] == 2 and query["cloud"] == "SECURE"
    assert query["minCudaVersion"] == "13.0"
    assert (query["minVcpuCountPerGpu"], query["minRamPerGpu"]) == (8, 50)
    places = [(o["_gpu"], o["_dc"], o["dph_total"]) for o in offers]
    # A6000 and CA-MTL-1 are out of stock. H100 is within the cap.
    assert places[:2] == [("NVIDIA GeForce RTX 4090", "EU-RO-1", 1.38),
                          ("NVIDIA GeForce RTX 4090", "US-TX-3", 1.38)]  # fmt: skip
    assert ("NVIDIA H100 80GB HBM3", "US-KS-2", 5.98) in places
    assert all(dc != "CA-MTL-1" for _, dc, _ in places)
    assert not any("MIG" in g for g, _, _ in places)  # a slice of a GPU, not asked for
    assert len(offers) == 3 * 2  # each place offered once per slot (count + spare)
    assert [o["dph_total"] for o in offers] == sorted(o["dph_total"] for o in offers)
    assert len({o["id"] for o in offers}) == len(offers) and "RunPod" in note


def test_only_the_models_asked_for_and_never_above_the_cap():
    p, _ = make()
    offers, _ = p.candidates(SPEC, {}, L.Policy(spare=0))
    assert {o["_gpu"] for o in offers} == {"NVIDIA GeForce RTX 4090"}
    cheap = L.Spec(image="i", gpus=("H100",), max_dph_per_gpu=2.0)
    assert p.candidates(cheap, {}, L.Policy())[0] == []


def test_the_default_pool_combines_both_tiers_cheapest_first():
    p, api = make()
    offers, note = p.candidates(SPEC, {}, L.Policy(spare=0))
    assert [c[3]["cloud"] for c in api.calls] == ["SECURE", "COMMUNITY"]
    assert [(o["_cloud"], o["dph_total"]) for o in offers] == [
        ("COMMUNITY", 0.34), ("COMMUNITY", 0.34), ("SECURE", 0.69), ("SECURE", 0.69),
    ]  # fmt: skip
    assert len({o["id"] for o in offers}) == 4  # A place may appear in both tiers.
    assert "secure and community" in note


def test_availability_is_specific_to_each_tier():
    p, api = make()
    api.catalogs["COMMUNITY"] = [{**CATALOG[0], "availability": "NONE"}]
    offers, _ = p.candidates(SPEC, {}, L.Policy(spare=0))
    assert [o["_cloud"] for o in offers] == ["SECURE", "SECURE"]

    api.catalogs["COMMUNITY"] = [{**CATALOG[0], "dataCenters": []}]
    api.catalogs["SECURE"] = [{**CATALOG[0], "availability": "NONE"}]
    offers, _ = p.candidates(SPEC, {}, L.Policy(spare=0))
    assert [(o["_cloud"], o["_dc"], o["dph_total"]) for o in offers] == [("COMMUNITY", None, 0.34)]


def test_secure_is_tried_first_when_prices_tie():
    p, api = make()
    gpu = {**CATALOG[0], "price": {"secure": 0.69, "community": 0.69}, "dataCenters": []}
    api.catalogs = {cloud: [gpu] for cloud in rp.CLOUDS}
    offers, _ = p.candidates(SPEC, {}, L.Policy(spare=0))
    assert [o["_cloud"] for o in offers] == ["SECURE", "COMMUNITY"]


# ------------------------------------------------------------------ calls


def test_a_pod_is_created_once_with_our_start_up_command_and_its_tag_as_name():
    p, api = make()
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    tag = make_tag(RUN_ID, 1, EXPIRY)
    pid = p.create(offer, SPEC, tag, "ssh-ed25519 AAAA gpubox", EXPIRY)
    method, path, body, _, retries = api.calls[-1]
    assert (method, path, retries) == ("POST", "pods", 0)
    assert body["cloud"] == "COMMUNITY" and offer["_cloud"] == "COMMUNITY"
    assert body["name"] == tag and body["image"] == "img:1" and body["ports"] == ["22/tcp"]
    assert body["gpu"] == {
        "id": "NVIDIA GeForce RTX 4090",
        "count": 1,
        "minCudaVersion": "13.0",
        "minVcpuCountPerGpu": 16,
        "minRamPerGpu": 32,
    }  # the Spec's defaults
    assert body["dataCenterIds"] == ["EU-RO-1"] and body["entrypoint"] == ["bash"]
    assert rp.DESTROY in body["cmd"][1] and "ssh-ed25519 AAAA gpubox" in body["cmd"][1]
    assert pid in api.pods


@pytest.mark.parametrize(
    "answer, outcome",
    [
        (rp.RunpodError(400, "no longer any instances available"), prov.CreateRefused),
        (rp.RunpodError(422, "Request validation failed"), prov.CreateRefused),
        (rp.RunpodError(401, "Unauthorized"), ConfigError),
        (rp.RunpodError(500, "internal"), prov.CreateUnknown),
        (rp.RunpodError(0, "timed out"), prov.CreateUnknown),
        (rp.RunpodError(408, "timeout"), prov.CreateUnknown),
        ({"id": "UPPER_not_a_pod"}, prov.CreateUnknown),
        (None, prov.CreateUnknown),
        ({"id": "abc123def", "name": "someone-else"}, prov.CreateUnknown),
        ({"id": "abc123def"}, prov.CreateUnknown),  # not the pod we asked for, as far as we know
        ({"id": "abc123def", "name": make_tag(RUN_ID, 1, EXPIRY)}, "abc123def"),
    ],
)
def test_a_pod_id_only_from_an_explicit_yes(answer, outcome):
    p, api = make()
    api.fail["POST pods"] = [answer]
    offer = {"id": "x", "_gpu": "NVIDIA GeForce RTX 4090", "_dc": None, "_cloud": "SECURE"}
    if isinstance(outcome, str):
        assert p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "k", EXPIRY) == outcome
    else:
        with pytest.raises(outcome):
            p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "k", EXPIRY)


def test_a_pod_that_is_gone_is_none_and_an_error_never_is():
    p, api = make()
    assert p.instance("nosuchpod1") is None
    api.pods["abcdef1"] = {"id": "abcdef1", "name": "n", "status": "TERMINATED", "dc": "x"}
    assert p.instance("abcdef1") is None
    api.pods["abcdef1"]["status"] = "RUNNING"
    assert p.instance("abcdef1")["label"] == "n"
    for answer in (rp.RunpodError(500, "x"), rp.RunpodError(0, "x"), {"id": "other12"},
                   {"id": "abcdef1", "name": "n", "status": "WEIRD"}, {"pods": []}):  # fmt: skip
        api.fail["GET pods"] = [answer]
        with pytest.raises(rp.RunpodError):
            p.instance("abcdef1")


def test_the_listing_is_every_pod_or_an_error():
    p, api = make()
    api.page = 2
    for i in range(5):
        api.pods[f"pod{i:04d}abc"] = {"id": f"pod{i:04d}abc", "name": f"n{i}", "status": "RUNNING"}
    api.pods["pod0009abc"] = {"id": "pod0009abc", "name": None, "status": "TERMINATED"}
    rows = p.instances()
    assert [r["id"] for r in rows] == [f"pod{i:04d}abc" for i in range(5)]
    bad = [
        {"pods": [{"id": "pod0001abc", "status": "RUNNING"}], "pagination": {"hasNextPage": False}},  # no name
        {"pods": [], "pagination": {"hasNextPage": True, "nextCursor": None}},  # does not advance
        {"pods": []},  # no pagination: we were not told whether more follow
        {"pods": [{"id": "pod0001abc", "name": "a", "status": "RUNNING"}] * 2,
         "pagination": {"hasNextPage": False}},  # twice
    ]  # fmt: skip
    for answer in bad:
        api.fail["GET pods"] = [answer]
        with pytest.raises(rp.RunpodError):
            p.instances()


def test_terminating_a_pod_that_is_gone_is_not_an_error_and_a_refusal_is():
    p, api = make()
    p.destroy("nosuchpod1")
    api.fail["DELETE pods"] = [rp.RunpodError(409, "belongs to a cluster")]
    with pytest.raises(rp.RunpodError):
        p.destroy("nosuchpod1")


def test_a_rename_is_believed_only_when_the_answer_shows_it():
    p, api = make()
    api.pods["abcdef1"] = {"id": "abcdef1", "name": "old", "status": "RUNNING"}
    p.relabel("abcdef1", "new")
    assert api.pods["abcdef1"]["name"] == "new"
    api.fail["PATCH pods"] = [{"id": "abcdef1", "name": "old"}]
    with pytest.raises(rp.RunpodError):
        p.relabel("abcdef1", "newer")


def test_a_pod_is_running_for_us_once_it_has_an_ssh_address():
    p, _ = make()
    row = rp._row({"id": "abcdef1", "name": "n", "status": "RUNNING", "ssh": {"direct": None}})
    assert p.view(row).state == "loading"
    row["ssh"] = {"direct": {"host": "1.2.3.4", "port": 4022}}
    assert (p.view(row).state, p.view(row).endpoint) == ("running", ("1.2.3.4", 4022))
    row["ssh"] = {}
    row["runtime"] = {"ports": [{"private": 22, "public": 5022, "ip": "1.2.3.4"}]}
    assert p.view(row).endpoint == ("1.2.3.4", 5022)
    for status in ("EXITED", "ERROR"):
        row["status"] = row["actual_status"] = status
        assert p.view(row).state == "exited"


# ------------------------------------------------------------------ a launch


class PodProber:
    pubkey = "ssh-ed25519 AAAA gpubox"
    scoped = False

    def __init__(self):
        self.checks, self.ran = [], []

    def kill_all(self):
        for c in self.checks:
            c.kill()

    def start_ready(self, ep):
        self.checks.append(FakeCheck(lambda: (True, "")))
        return self.checks[-1]

    def start_gate(self, ep, spec):
        self.checks.append(FakeCheck(lambda: (True, None, "", {"gpus": ["RTX 4090"]})))
        return self.checks[-1]

    def run(self, ep, remote, timeout=60):
        self.ran.append((ep, remote))
        return 0


def launch(tmp_path, behaviour=None, **kw):
    p, api = make(FakeApi(behaviour))
    state = State.begin(tmp_path, RUN_ID, EXPIRY, None, {"provider": "runpod"})
    policy = L.Policy(**{**FAST.__dict__, **kw})
    return L.Launch(p, SPEC, policy, state, PodProber()), p, api, state


def test_a_launch_rents_a_pod_skips_a_full_data_center_and_keeps_the_winner(tmp_path):
    run, p, api, state = launch(tmp_path, {"EU-RO-1": "full"}, spare=0)
    result = run.run({})
    (w,) = result.winners
    assert api.pods[w.instance_id]["dc"] == "US-TX-3"
    on_disk = State.load(tmp_path)  # string ids in the state file
    assert on_disk.winners[0]["id"] == w.instance_id and on_disk.ids == [w.instance_id]
    assert on_disk.winners[0]["port"] == 40000 + int(w.instance_id[3:7])


def test_a_launch_falls_back_to_secure_when_community_has_no_direct_ssh(tmp_path):
    run, _, api, _ = launch(tmp_path, {"COMMUNITY": "noaddr"}, spare=0)
    api.catalogs["COMMUNITY"] = [{**CATALOG[0], "dataCenters": []}]
    result = run.run({})
    (winner,) = result.winners
    assert list(api.pods) == [winner.instance_id]
    assert [body["cloud"] for m, p, body, _, _ in api.calls if (m, p) == ("POST", "pods")] == [
        "COMMUNITY",
        "SECURE",
    ]


def test_down_terminates_by_id_and_by_tag_with_string_ids(tmp_path):
    run, p, api, state = launch(tmp_path)
    run.run({})
    offer = {"id": "x", "_gpu": "NVIDIA GeForce RTX 4090", "_dc": "US-TX-3", "_cloud": "SECURE"}
    stray = p.create(offer, SPEC, make_tag(RUN_ID, 9, EXPIRY), "k", EXPIRY)
    other = p.create(offer, SPEC, make_tag("dev.other.0123456789ab", 1, EXPIRY), "k", EXPIRY)
    assert L.down(p, state, FAST) == []
    assert stray not in api.pods and list(api.pods) == [other]


def test_renew_moves_the_pods_own_timer_and_its_name(tmp_path):
    run, p, api, state = launch(tmp_path, spare=0)
    run.run({})
    later = EXPIRY + 7200
    L.renew(p, state, run.prober, later)
    ((_, remote),) = run.prober.ran
    assert remote == p.expiry_command(later)
    (pod,) = api.pods.values()
    assert parse_tag(pod["name"]).expiry == later


# ------------------------------------------------------------------ the command line


@pytest.mark.parametrize("cloud, expected", [(None, rp.CLOUDS), ("secure", ("SECURE",)),
                                             ("community", ("COMMUNITY",))])  # fmt: skip
def test_runpod_search_shows_both_tiers_unless_restricted(
    tmp_path, capsys, monkeypatch, cloud, expected
):
    api = FakeApi()
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr(rp, "Client", lambda key: api)
    argv = ["search", "--provider", "runpod", "--gpu", "RTX 4090", "--max-dph-per-gpu", "1",
            "--state-dir", str(tmp_path)]  # fmt: skip
    if cloud:
        argv += ["--runpod-cloud", cloud]
    assert cli.main(argv) == 0
    assert tuple(c[3]["cloud"] for c in api.calls) == expected
    lines = capsys.readouterr().out.splitlines()[1:]
    assert [line.split("@")[-1].split("/")[0] for line in lines] == [
        tier.lower() for tier in reversed(expected) for _ in range(2)
    ]


def test_runpod_needs_an_image_and_a_price_and_refuses_what_it_cannot_do(tmp_path, capsys):
    base = ["up", "--provider", "runpod", "--namespace", "dev", "--name", "p",
            "--state-dir", str(tmp_path)]  # fmt: skip
    assert cli.main([*base, "--max-dph-per-gpu", "1"]) == cli.EXIT_CONFIG
    assert "runpod needs --image" in capsys.readouterr().out
    for extra in (["--interruptible"], ["--exclude-country", "US"], ["--partition", "x"]):
        argv = [*base, "--image", "i", "--max-dph-per-gpu", "1", *extra]
        assert cli.main(argv) == cli.EXIT_CONFIG
        assert "not on runpod" in (out := capsys.readouterr().out) or "not for runpod" in out
    assert cli.main(["up", "--gpu", "H100", "--max-dph-per-gpu", "1", "--image", "i",
                     "--namespace", "d", "--name", "p", "--runpod-cloud", "community",
                     "--state-dir", str(tmp_path)]) == cli.EXIT_CONFIG  # fmt: skip
    assert "--runpod-cloud: not for vast" in capsys.readouterr().out


def test_the_key_comes_from_the_environment_or_the_keys_file(tmp_path, monkeypatch):
    from gpubox import api

    keys = tmp_path / "keys.env"
    keys.write_text("VASTAI_API_KEY=v\nexport RUNPOD_API_KEY='r-key'\n")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    s = api.Settings(state_dir=tmp_path, provider="runpod", keys_file=str(keys))
    provider = s.named_provider()
    auth = lambda p: p.client.headers["Authorization"]
    assert isinstance(provider, rp.RunPod) and auth(provider) == "Bearer r-key"
    monkeypatch.setenv("RUNPOD_API_KEY", "from-env")
    assert auth(s.named_provider()) == "Bearer from-env"
    s = api.Settings(state_dir=tmp_path, provider="runpod", keys_file=str(tmp_path / "none"))
    monkeypatch.delenv("RUNPOD_API_KEY")
    with pytest.raises(ConfigError, match="RUNPOD_API_KEY"):
        s.named_provider()
