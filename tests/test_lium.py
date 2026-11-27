"""Lium: how its API is asked and read, a launch on it, and its CLI options.

`FakeApi` stands in for `lium.Client`: it answers `call(method, path, ...)` as the API
does, from a list of nodes and a dict of pods, and `fail` makes chosen calls answer
otherwise first.
"""

import shlex
import time

import pytest
from fakes import FAST, FakeCheck

from gpubox import cli
from gpubox import launch as L
from gpubox.errors import ConfigError
from gpubox.providers import base as prov
from gpubox.providers import httpapi
from gpubox.providers import lium as lm
from gpubox.providers.vmbox import CONTAINER_INIT
from gpubox.state import State, make_tag, parse_tag

RUN_ID = "dev.lium.0123456789ab"
EXPIRY = int(time.time()) + 3600
SPEC = L.Spec(image="img:1", gpus=("RTX 4090",), max_dph_per_gpu=1.0)
KIB = 1024**2  # per GB


def uid(n: int) -> str:
    return f"00000000-0000-4000-8000-{n:012d}"


def node(n, gpu="NVIDIA GeForce RTX 4090", price=0.35, total=1, free=1, least=1, **kw):
    """A node as /executors lists it: by default one 4090 that a gpubox Spec accepts."""
    x = {
        "id": uid(n), "machine_name": gpu, "gpu_class": gpu, "price_per_gpu": price,
        "gpu_count": total, "available_gpu_count": free, "tier": "secure",
        "min_gpu_count_for_rental": least, "min_gpu_count_for_ssh_only_rental": least,
        "frequent_rent_failures": False, "disk_health_ok": True, "is_slow_machine": False,
        "removal_at": None, "scheduled_notice": None, "reliability_score": 99.5,
        "effective_download_speed_mbps": 800.0, "max_cuda_version": 13.0,
        "free_port_count": 99,
        "location": {"country_code": "US", "city": "Austin"},
        "specs": {
            "is_spot": False,
            "cpu": {"count": 24 * total},
            "ram": {"total": (64 * total + 8) * KIB, "pod_reserve": 8 * KIB},
            "hard_disk": {"free": 500 * total * KIB},
            "gpu": {"details": [{"capacity": 24564}] * total},
        },
    }  # fmt: skip
    for k, v in kw.items():
        if k in ("is_spot", "cpu", "ram", "hard_disk", "gpu"):
            x["specs"][k] = v
        else:
            x[k] = v
    return x


NODES = [
    node(1, price=0.40),
    node(2, price=0.30, total=4, free=3, least=1),  # holds three single-GPU boxes
    node(3, price=0.10, gpu="NVIDIA H100 80GB HBM3"),  # not the model asked for
    node(4, price=1.50),  # above the cap
    node(5, price=0.20, is_spot=True),  # interruptible
    node(6, price=0.20, tier="spot"),
    node(7, price=0.20, total=2, free=2, least=None),  # rents only whole
    node(8, price=0.20, max_cuda_version=12.8),
    node(9, price=0.20, location={"country_code": "CN", "city": "x"}),
    node(10, price=0.20, reliability_score=90.0),
    node(11, price=0.20, frequent_rent_failures=True),
    node(12, price=0.20, cpu={"count": 8}),  # too few CPUs for the default 16
    node(13, price=0.20, effective_download_speed_mbps=None),
    node(14, price=0.20, scheduled_notice={"starting_at": "2026-01-01T00:00:00Z"}),
    node(15, price=0.20, free_port_count=0),
    node(16, price=0.20, ram={"total": 36 * KIB, "pod_reserve": 8 * KIB}),
    node(17, price=0.20, available_gpu_count=0),
]


class FakeApi:
    def __init__(self, nodes=None):
        self.nodes = list(NODES if nodes is None else nodes)
        self.pods, self.templates, self.calls, self.fail = {}, {}, [], {}
        self.full = set()  # nodes that refuse a rent: rented since the listing
        self.n = 0

    def call(self, method, path, *, body=None, query=None, retries=2, timeout=None, headers=None):
        self.calls.append((method, path, body, retries, headers))
        parts = path.split("/")
        key = f"{method} {parts[0]}" + (f" {parts[-1]}" if len(parts) > 2 else "")
        if len(parts) == 2:
            key = f"{method} {parts[0]} {parts[1]}"
        if self.fail.get(key):
            answer = self.fail[key].pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        if path == "executors":
            return self.nodes
        if path == "users/me":
            return {"id": uid(999), "balance": 10.0}
        if path == "templates" and method == "POST":
            self.n += 1
            self.templates[uid(5000 + self.n)] = body
            return {"id": uid(5000 + self.n), **body}
        if parts[0] == "templates" and method == "DELETE":
            self.templates.pop(parts[1], None)
            return {"success": True}
        if parts[0] == "executors" and parts[2] == "rent":
            if parts[1] in self.full:
                raise err(400, "node_unavailable")
            for p in self.pods.values():  # the same key and body: the first answer
                if p["key"] == headers["Idempotency-Key"]:
                    return {"success": True, "pod_id": p["id"]}
            self.n += 1
            pid = uid(9000 + self.n)
            self.pods[pid] = {
                "id": pid, "pod_name": body["pod_name"], "status": "RUNNING", "node": parts[1],
                "ssh_connect_cmd": f"ssh root@203.0.113.{self.n} -p {20000 + self.n}",
                "removal_scheduled_at": None, "key": headers["Idempotency-Key"], "body": body,
            }  # fmt: skip
            return {"success": True, "pod_id": pid, "phase": "queued"}
        if path == "pods" and method == "GET":
            return [dict(p) for p in self.pods.values()]
        pid = parts[1]
        if pid not in self.pods:
            raise err(404, "pod_not_found")
        pod = self.pods[pid]
        if parts[-1] == "schedule-removal":
            pod["removal_scheduled_at"] = body["removal_scheduled_at"].rstrip(
                "Z"
            )  # as Lium shows it
            return dict(pod)
        if method == "GET":
            return dict(pod)
        if method == "DELETE":
            pod["status"] = "DELETING"
            return {"success": True, "message": "Pod deletion started"}
        if method == "PATCH":
            pod["pod_name"] = body["pod_name"]
            return dict(pod)
        raise AssertionError((method, path))


def err(status, code):
    e = lm.LiumError(status, f"HTTP {status}: {code}")
    e.code = code
    return e


def make(api=None):
    api = api or FakeApi()
    return lm.Lium(api), api


def rents(api):
    return [c for c in api.calls if c[1].endswith("/rent")]


# ------------------------------------------------------------------ offers


def test_offers_are_nodes_that_can_take_the_box_cheapest_first_each_once_per_slot():
    p, api = make()
    offers, note = p.candidates(SPEC, {}, L.Policy(spare=1))
    # Node 2 holds three boxes, node 1 one; the race has two slots.
    assert [(o["_node"], o["dph_total"]) for o in offers] == [
        (uid(2), 0.30), (uid(1), 0.40), (uid(2), 0.30),
    ]  # fmt: skip
    assert len({o["id"] for o in offers}) == len(offers) and "Lium" in note
    o = offers[0]
    assert o["machine_id"] == uid(2) and o["cuda_max_good"] == 13.0
    assert o["geolocation"] == "Austin, US" and o["gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert (o["cpu_cores_effective"], o["cpu_ram"] / 1024) == (24, 64)  # a quarter of the node


def test_a_node_that_rents_only_whole_is_offered_only_for_all_its_free_gpus():
    p, _ = make(FakeApi([node(7, total=2, free=2, least=None)]))
    assert p.candidates(SPEC, {}, L.Policy())[0] == []
    two = L.Spec(image="i", gpus=("4090",), max_dph_per_gpu=1.0, num_gpus=2)
    (offer,) = p.candidates(two, {}, L.Policy(spare=3))[0]  # one box, however many slots
    assert offer["dph_total"] == 0.7


def test_what_is_not_stated_does_not_meet_a_minimum():
    nodes = [node(1, max_cuda_version=None), node(2, location={}),
             node(3, price_per_gpu="cheap"), node(4, ram=None)]  # fmt: skip
    nodes.append({**node(5), "specs": None})
    p, _ = make(FakeApi(nodes))
    assert p.candidates(SPEC, {}, L.Policy())[0] == []


def test_any_model_without_gpu_and_countries_as_asked():
    p, _ = make()
    spec = L.Spec(image="i", gpus=(), max_dph_per_gpu=1.0, exclude_countries=("US",))
    (only,) = p.candidates(spec, {}, L.Policy())[0]  # all in the US, but one in CN
    assert only["_node"] == uid(9)
    spec = L.Spec(image="i", gpus=(), max_dph_per_gpu=1.0, exclude_countries=())
    found = {o["_node"] for o in p.candidates(spec, {}, L.Policy())[0]}
    assert {uid(3), uid(9)} <= found


# ------------------------------------------------------------------ the rent


def test_a_pod_is_rented_with_a_template_of_the_image_the_runs_key_and_its_lease():
    p, api = make()
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    tag = make_tag(RUN_ID, 1, EXPIRY)
    pid = p.create(offer, SPEC, tag, "ssh-ed25519 AAAA gpubox", EXPIRY)
    (t,) = api.templates.values()
    assert t["docker_image"] == "img:1" and t["one_time_template"] and t["is_private"]
    assert t["internal_ports"] == [22] and t["entrypoint"] == "/usr/bin/env"
    assert shlex.split(t["startup_commands"]) == ["bash", "-c", CONTAINER_INIT]
    ((_, path, body, retries, headers),) = rents(api)
    assert path == f"executors/{uid(2)}/rent" and body["pod_name"] == tag
    assert body["user_public_key"] == ["ssh-ed25519 AAAA gpubox"] and body["gpu_count"] == 1
    assert body["skip_add_agent_ssh_key"] is True and body["expected_price_per_gpu_hour"] == 0.30
    assert body["template_id"] in api.templates and "termination_hours" not in body
    # Lium answers a repeat of the key with the first answer: so the rent is retried.
    assert headers == {"Idempotency-Key": tag} and retries == 2
    assert lm.same_time(api.pods[pid]["removal_scheduled_at"], EXPIRY)


def test_a_rent_repeated_under_its_key_is_the_same_pod():
    p, api = make()
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    tag = make_tag(RUN_ID, 1, EXPIRY)
    assert p.create(offer, SPEC, tag, "k", EXPIRY) == p.create(offer, SPEC, tag, "k", EXPIRY)
    assert len(api.pods) == 1


@pytest.mark.parametrize(
    "answer, outcome",
    [
        (err(400, "node_unavailable"), prov.CreateRefused),
        (err(409, "price_changed"), prov.CreateRefused),
        (err(409, "idempotency_in_progress"), prov.CreateUnknown),
        (err(409, "idempotency_outcome_unknown"), prov.CreateUnknown),
        (err(400, "own_rent_in_progress"), prov.CreateUnknown),
        (err(500, "internal_error"), prov.CreateUnknown),
        (lm.LiumError(0, "timed out"), prov.CreateUnknown),
        (err(401, "unauthorized"), ConfigError),
        (err(402, "account_unfunded"), ConfigError),
        (err(403, "insufficient_balance"), ConfigError),
        (err(400, "image_not_found"), ConfigError),
        ({"success": True}, prov.CreateUnknown),
        ({"success": False, "pod_id": uid(7)}, prov.CreateUnknown),
        ({"success": True, "pod_id": "not-a-pod"}, prov.CreateUnknown),
    ],
)
def test_a_pod_id_only_from_an_explicit_yes(answer, outcome):
    p, api = make()
    api.fail["POST executors rent"] = [answer]
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    with pytest.raises(outcome):
        p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "k", EXPIRY)
    if outcome is prov.CreateRefused:
        assert api.templates == {}  # nothing was rented from it
    elif outcome is prov.CreateUnknown and not isinstance(answer, dict):
        assert len(api.templates) == 1  # a pod may hold it


def test_a_pod_whose_lease_cannot_be_set_is_deleted_and_its_rent_counts_as_unanswered():
    p, api = make()
    api.fail["POST pods schedule-removal"] = [err(500, "internal_error")]
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    with pytest.raises(prov.CreateUnknown, match="removal time"):
        p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "k", EXPIRY)
    (pod,) = api.pods.values()
    assert pod["status"] == "DELETING"


def test_a_node_that_failed_in_this_launch_is_not_rented_again():
    p, api = make()
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    with pytest.raises(prov.CreateRefused):
        p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "k", EXPIRY, frozenset({uid(2)}))
    assert api.calls[1:] == []


def test_a_template_lium_refuses_stops_the_launch_and_a_lost_one_rents_nothing():
    p, api = make()
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    api.fail["POST templates"] = [err(400, "bad_request")]
    with pytest.raises(ConfigError):
        p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "k", EXPIRY)
    api.fail["POST templates"] = [lm.LiumError(0, "timed out")]
    with pytest.raises(prov.CreateRefused):
        p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "k", EXPIRY)
    assert rents(api) == []
    with pytest.raises(ConfigError, match="not an image"):
        p.create(offer, L.Spec(image="-x", gpus=(), max_dph_per_gpu=1), "t", "k", EXPIRY)


# ------------------------------------------------------------------ the pods


def test_a_pod_being_deleted_is_gone_and_an_error_never_is():
    p, api = make()
    assert p.instance(uid(1)) is None
    api.pods[uid(1)] = {"id": uid(1), "pod_name": "n", "status": "RUNNING"}
    assert p.instance(uid(1))["label"] == "n"
    api.pods[uid(1)]["status"] = "DELETING"  # billing stops at the delete request
    assert p.instance(uid(1)) is None
    for answer in (err(500, "internal_error"), lm.LiumError(0, "x"), {"id": uid(2)},
                   {"id": uid(1), "pod_name": "n", "status": "WEIRD"}, []):  # fmt: skip
        api.fail[f"GET pods {uid(1)}"] = [answer]
        with pytest.raises(lm.LiumError):
            p.instance(uid(1))


def test_the_listing_is_every_pod_or_an_error():
    p, api = make()
    for i in range(3):
        api.pods[uid(i)] = {"id": uid(i), "pod_name": f"n{i}", "status": "RUNNING"}
    api.pods[uid(9)] = {"id": uid(9), "pod_name": "x", "status": "DELETING"}
    assert [r["id"] for r in p.instances()] == [uid(0), uid(1), uid(2)]
    for answer in ([{"id": uid(1), "status": "RUNNING"}],  # no name
                   [{"id": uid(1), "pod_name": "a", "status": "RUNNING"}] * 2,  # twice
                   {"pods": []}, err(500, "internal_error")):  # fmt: skip
        api.fail["GET pods"] = [answer]
        with pytest.raises(lm.LiumError):
            p.instances()
    # Said only to an account with no balance and no pods.
    api.fail["GET pods"] = [err(402, "BALANCE_REQUIRED")]
    assert p.instances() == []
    api.fail["GET pods"] = [err(402, "account_unfunded")]
    with pytest.raises(lm.LiumError):
        p.instances()


def test_deleting_a_pod_that_is_gone_is_not_an_error_and_a_refusal_is():
    p, api = make()
    p.destroy(uid(1))
    api.fail[f"DELETE pods {uid(1)}"] = [err(400, "pod_state_conflict")]
    with pytest.raises(lm.LiumError):
        p.destroy(uid(1))


def test_a_relabel_moves_liums_removal_time_and_is_believed_once_read_back():
    p, api = make()
    api.pods[uid(1)] = {"id": uid(1), "pod_name": "old", "status": "RUNNING",
                        "removal_scheduled_at": None}  # fmt: skip
    tag = make_tag(RUN_ID, 1, EXPIRY + 600)
    p.relabel(uid(1), tag)
    assert api.pods[uid(1)]["pod_name"] == tag
    assert lm.same_time(api.pods[uid(1)]["removal_scheduled_at"], EXPIRY + 600)
    ops = [(c[0], c[1].split("/")[-1]) for c in api.calls]
    assert ops.index(("POST", "schedule-removal")) < ops.index(("PATCH", uid(1)))
    api.fail["POST pods schedule-removal"] = [{}]  # said yes, did nothing
    with pytest.raises(lm.LiumError, match="not confirmed"):
        p.relabel(uid(1), make_tag(RUN_ID, 1, EXPIRY + 900))


def test_liums_times_without_an_offset_are_utc():
    t = 1795502400  # 2026-11-24T06:40:00Z
    assert lm.same_time("2026-11-24T06:40:00", t) and lm.same_time("2026-11-24T06:40:00.4", t)
    assert lm.same_time("2026-11-24T08:40:00+02:00", t)
    assert not lm.same_time("2026-11-24T06:41:00", t) and not lm.same_time(None, t)


def test_a_pod_is_running_for_us_once_lium_shows_how_root_gets_in_directly():
    p, _ = make()
    row = lm._row({"id": uid(1), "pod_name": "n", "status": "PENDING", "ssh_connect_cmd": "",
                   "create_progress": {"message": "Pulling docker image img:1"}})  # fmt: skip
    assert (p.view(row).state, p.view(row).msg) == ("loading", "Pulling docker image img:1")
    row["actual_status"] = "RUNNING"
    assert p.view(row).state == "loading"  # no address yet
    row["ssh_connect_cmd"] = "ssh root@203.0.113.5 -p 20010"
    assert (p.view(row).state, p.view(row).endpoint) == ("running", ("203.0.113.5", 20010))
    for cmd in ("ssh -J root@gw.example root@10.0.0.1", "ssh ubuntu@203.0.113.5 -p 22",
                "ssh root@203.0.113.5 -p 22 -o ProxyCommand=x"):  # fmt: skip
        row["ssh_connect_cmd"] = cmd
        assert (p.view(row).state, p.view(row).problem) == ("exited", "machine")
    row["create_progress"] = {"message": "Error response from daemon: manifest unknown"}
    for status in ("PENDING", "CREATION_FAILED"):
        row["actual_status"] = status
        assert p.view(row).problem == "image"
    for status in ("FAILED", "BROKEN", "STOPPED", "REBOOT_FAILED"):
        row["actual_status"], row["create_progress"] = status, None
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


def launch(tmp_path, api=None, **kw):
    p, api = make(api)
    state = State.begin(tmp_path, RUN_ID, EXPIRY, None, {"provider": "lium"})
    policy = L.Policy(**{**FAST.__dict__, **kw})
    return L.Launch(p, SPEC, policy, state, PodProber()), p, api, state


def test_a_launch_rents_a_pod_skips_a_node_rented_since_and_keeps_the_winner(tmp_path):
    run, p, api, state = launch(tmp_path, spare=0)
    api.full.add(uid(2))
    (w,) = run.run({}).winners
    assert api.pods[w.instance_id]["node"] == uid(1)
    (box,) = State.load(tmp_path).winners
    assert box["id"] == w.instance_id and box["host"].startswith("203.0.113.")


def test_down_deletes_by_id_and_by_tag(tmp_path):
    run, p, api, state = launch(tmp_path)
    run.run({})
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    stray = p.create(offer, SPEC, make_tag(RUN_ID, 9, EXPIRY), "k", EXPIRY)
    other = p.create(offer, SPEC, make_tag("dev.other.0123456789ab", 1, EXPIRY), "k", EXPIRY)
    assert L.down(p, state, FAST) == []
    assert api.pods[stray]["status"] == "DELETING" and api.pods[other]["status"] == "RUNNING"


def test_renew_moves_liums_removal_time_and_touches_nothing_on_the_box(tmp_path):
    run, p, api, state = launch(tmp_path, spare=0)
    run.run({})
    later = EXPIRY + 7200
    run.prober.run = lambda *a, **k: 255  # the box cannot be reached: no matter, here
    L.renew(p, state, run.prober, later)
    assert run.prober.ran == []
    (pod,) = api.pods.values()
    assert parse_tag(pod["pod_name"]).expiry == later
    assert lm.same_time(pod["removal_scheduled_at"], later)


# ------------------------------------------------------------------ the command line


def test_lium_needs_an_image_and_a_price_and_refuses_what_it_cannot_do(tmp_path, capsys):
    base = ["up", "--provider", "lium", "--namespace", "dev", "--name", "p",
            "--state-dir", str(tmp_path)]  # fmt: skip
    assert cli.main([*base, "--max-dph-per-gpu", "1"]) == cli.EXIT_CONFIG
    assert "lium needs --image" in capsys.readouterr().out
    for extra in (["--interruptible"], ["--registry-user", "u"], ["--runpod-cloud", "secure"]):
        argv = [*base, "--image", "i", "--max-dph-per-gpu", "1", *extra]
        assert cli.main(argv) == cli.EXIT_CONFIG
        assert "not on lium" in (out := capsys.readouterr().out) or "not for lium" in out


def test_lium_search_lists_the_nodes_up_would_try(tmp_path, capsys, monkeypatch):
    api = FakeApi()
    monkeypatch.setenv("LIUM_API_KEY", "k")
    monkeypatch.setattr(lm, "Client", lambda key: api)
    argv = ["search", "--provider", "lium", "--gpu", "RTX 4090", "--max-dph-per-gpu", "1",
            "--state-dir", str(tmp_path)]  # fmt: skip
    assert cli.main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "2 offer(s)" and uid(2) in lines[1] and uid(1) in lines[2]


def test_the_key_goes_in_its_header_and_an_error_says_its_code():
    import http.server
    import threading

    body = '{"success": false, "error": {"code": "node_unavailable", "message": "Node is gone."}}'
    seen = []

    class Refuse(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append({k.lower(): v for k, v in self.headers.items()})
            self.rfile.read(int(self.headers["Content-Length"]))  # else the close resets it
            self.send_response(400)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Refuse)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = lm.Client("secret", base=f"http://127.0.0.1:{server.server_port}")
        with pytest.raises(lm.LiumError) as exc:
            client.call("POST", "executors/x/rent", body={}, headers={"Idempotency-Key": "t"})
    finally:
        server.shutdown()
    assert (exc.value.status, exc.value.code) == (400, "node_unavailable")
    assert seen[0]["x-api-key"] == "secret" and seen[0]["idempotency-key"] == "t"
    assert httpapi.error_code(body) == "node_unavailable"
    assert httpapi.describe_error(body) == "Node is gone.: node_unavailable"
    assert httpapi.error_code('{"code": "x"}') == "x" and httpapi.error_code("<html>") is None
