"""Shadeform, and the VM box (vmbox.py) it shares with the VM clouds to come.

`FakeApi` stands in for `shadeform.Client`: it answers `call(method, path, ...)` as the
API does, from a dict of instances; `fail` makes chosen calls answer otherwise first.
"""

import base64
import pathlib
import subprocess
import time

import pytest
from fakes import FAST, FakeCheck

from gpubox import api, gate, onbox
from gpubox import launch as L
from gpubox.providers import base as prov
from gpubox.providers import shadeform as sf
from gpubox.providers import vmbox
from gpubox.errors import ConfigError
from gpubox.state import State, make_tag, parse_tag

RUN_ID = "dev.vm.0123456789ab"
EXPIRY = int(time.time()) + 3600
SPEC = L.Spec(image="img:1", gpus=(), max_dph_per_gpu=1.0)


def itype(cloud, name, gpu, cents, regions, os=("ubuntu22.04_cuda13.0_shade_os",), **conf):
    return {
        "cloud": cloud, "shade_instance_type": name, "hourly_price": cents,
        "boot_time": {"min_boot_in_sec": 300, "max_boot_in_sec": 480},
        "configuration": {"num_gpus": 1, "gpu_type": gpu, "os_options": list(os),
                          "vcpus": 16, "memory_in_gb": 64, "storage_in_gb": 200,
                          "vram_per_gpu_in_gb": 48, **conf},
        "availability": [{"region": r, "available": ok, "rental_type": rt}
                         for r, ok, rt in regions],
    }  # fmt: skip


TYPES = [
    itype("hyperstack", "A6000", "A6000", 50, [("montreal-canada-2", True, "on_demand")],
          os=("ubuntu22.04_cuda12.8_shade_os", "ubuntu22.04")),  # CUDA 12.8 only
    itype("massedcompute", "A6000_base", "A6000", 57,
          [("kansascity-usa-1", True, "on_demand"), ("wichita-usa-1", False, "on_demand"),
           ("desmoines-usa-1", True, "spot")]),
    itype("lambdalabs", "H100", "H100", 249, [("dallas-usa-3", True, "on_demand")]),
    itype("crusoe", "L40S", "L40S", 90, [("iceland-1", True, "on_demand")], os=("ubuntu22.04",)),
    itype("vultr", "A16", "A16", 30, [("x", True, "on_demand")], storage_in_gb=10),
]  # fmt: skip


class FakeApi:
    def __init__(self):
        self.instances, self.calls, self.fail = {}, [], {}
        self.next, self.page_size = 0, 100
        self.full = set()  # regions that refuse a create
        self.boot_looks = 0  # lookups a new VM stays pending for

    def call(self, method, path, *, body=None, query=None, retries=2, timeout=None):
        self.calls.append((method, path, body, query, retries))
        key = f"{method} {path.split('/')[0]}"
        if self.fail.get(key):
            answer = self.fail[key].pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        if path == "instances/types":
            return {"instance_types": TYPES}
        if path == "instances/create":
            if body["region"] in self.full:
                raise sf.ShadeformError(409, "POST /instances/create: HTTP 409: out of capacity")
            self.next += 1
            iid = f"inst-{self.next:04d}"
            self.instances[iid] = {
                "id": iid, "name": body["name"], "status": "pending", "tags": body["tags"],
                "auto_delete": body["auto_delete"], "ip": None, "ssh_user": None, "body": body,
            }  # fmt: skip
            return {"id": iid}
        if path == "instances" and method == "GET":
            rows = [r for r in self.instances.values() if r["status"] != "deleted"]
            start = query["offset"]
            return {"instances": rows[start : start + query["limit"]], "total": len(rows)}
        iid = path.split("/")[1]
        row = self.instances.get(iid)
        if row is None or row["status"] == "deleted":
            raise sf.ShadeformError(404, f"{method} /{path}: HTTP 404: not found")
        if path.endswith("/info"):
            row["looks"] = row.get("looks", 0) + 1
            if row["status"] == "pending" and row["looks"] > self.boot_looks:  # then active
                row.update(status="active", ip=f"10.1.0.{int(iid[-4:])}", ssh_port=22,
                           ssh_user="shadeform")  # fmt: skip
            return dict(row)
        if path.endswith("/delete"):
            row["status"] = "deleting"
            return None
        if path.endswith("/update"):
            row.update(body)
            return None
        raise AssertionError((method, path))


def make():
    api_ = FakeApi()
    return sf.Shadeform(api_), api_


# ------------------------------------------------------------------ offers


def test_offers_are_types_in_regions_with_a_shade_os_new_enough_cheapest_first():
    p, api_ = make()
    offers, note = p.candidates(SPEC, {}, L.Policy(spare=1))
    assert api_.calls[0][3] == {"num_gpus": 1, "available": "true"}
    places = [(o["_cloud"], o["_region"]) for o in offers]
    # Hyperstack's OS has CUDA 12.8 (< 13.0); Crusoe's no shade_os; Lambda's H100 is over the
    # cap; Vultr's disk is too small; a spot or unavailable region is not on demand now.
    assert places == [("massedcompute", "kansascity-usa-1")] * 2  # once per slot
    o = offers[0]
    assert o["_os"] == "ubuntu22.04_cuda13.0_shade_os" and o["cuda_max_good"] == 13.0
    assert o["boot_s"] == 480 and o["dph_total"] == 0.57 and o["machine_id"] is None
    assert o["host_id"] == "massedcompute/kansascity-usa-1" and "Shadeform" in note
    older = L.Spec(image="i", gpus=("A6000",), max_dph_per_gpu=1.0, min_cuda=12.8)
    clouds = [o["_cloud"] for o in p.candidates(older, {}, L.Policy(spare=0))[0]]
    assert clouds == ["hyperstack", "massedcompute"]


def test_the_newest_cuda_of_shadeforms_own_images_is_chosen():
    assert sf.best_os(["ubuntu22.04", "ubuntu22.04_cuda12.2_shade_os",
                       "ubuntu22.04_cuda12.8_shade_os"]) == ("ubuntu22.04_cuda12.8_shade_os", 12.8)  # fmt: skip
    assert sf.best_os(["ubuntu22.04"]) is None


# ------------------------------------------------------------------ calls


def test_a_vm_is_created_once_with_our_tag_its_lease_and_the_box_script():
    p, api_ = make()
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    tag = make_tag(RUN_ID, 1, EXPIRY)
    iid = p.create(offer, SPEC, tag, "ssh-ed25519 AAAA gpubox", EXPIRY)
    method, path, body, _, retries = api_.calls[-1]
    assert (method, path, retries) == ("POST", "instances/create", 0)
    assert body["tags"] == [tag] and body["shade_cloud"] is True
    assert body["auto_delete"] == {"date_threshold": sf.rfc3339(EXPIRY)}
    assert len(body["name"]) <= 50 and set(body["name"]) <= set(
        "abcdefghijklmnopqrstuvwxyz0123456789-_"
    )
    script = base64.b64decode(
        body["launch_configuration"]["script_configuration"]["base64_script"]
    ).decode()
    assert script == vmbox.startup_script("ssh-ed25519 AAAA gpubox", "img:1")
    assert body["os"] == "ubuntu22.04_cuda13.0_shade_os" and iid in api_.instances


@pytest.mark.parametrize(
    "answer, outcome",
    [
        (sf.ShadeformError(409, "out of capacity"), prov.CreateRefused),
        (sf.ShadeformError(400, "no such cloud"), prov.CreateRefused),
        (sf.ShadeformError(401, "bad key"), ConfigError),
        (sf.ShadeformError(500, "boom"), prov.CreateUnknown),
        (sf.ShadeformError(0, "timed out"), prov.CreateUnknown),
        ({"id": "has space"}, prov.CreateUnknown),
        (None, prov.CreateUnknown),
        ({"id": "inst-9"}, "inst-9"),
    ],
)
def test_an_instance_id_only_from_an_explicit_yes(answer, outcome):
    p, api_ = make()
    api_.fail["POST instances"] = [answer]
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    if isinstance(outcome, str):
        assert (
            p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "ssh-ed25519 K", EXPIRY) == outcome
        )
    else:
        with pytest.raises(outcome):
            p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "ssh-ed25519 K", EXPIRY)


def test_an_instance_on_its_way_out_is_gone_and_an_error_never_is():
    p, api_ = make()
    assert p.instance("inst-0001") is None
    api_.instances["inst-0001"] = {"id": "inst-0001", "status": "active", "tags": None}
    assert p.instance("inst-0001")["label"] is None
    for status in sf.GONE:
        api_.instances["inst-0001"]["status"] = status
        assert p.instance("inst-0001") is None
    api_.instances["inst-0001"]["status"] = "active"
    two = [make_tag(RUN_ID, 1, EXPIRY), make_tag(RUN_ID, 2, EXPIRY)]
    for answer in (sf.ShadeformError(500, "x"), sf.ShadeformError(0, "x"), {"id": "inst-0002", "status": "active"},
                   {"id": "inst-0001", "status": "active", "tags": two}):  # fmt: skip
        api_.fail["GET instances"] = [answer]
        with pytest.raises(sf.ShadeformError):
            p.instance("inst-0001")


def test_the_listing_is_every_instance_or_an_error():
    p, api_ = make()
    api_.page_size = 2
    tag = make_tag(RUN_ID, 1, EXPIRY)
    for i in range(5):
        api_.instances[f"i{i}"] = {
            "id": f"i{i}",
            "status": "active",
            "tags": [tag] if i == 2 else ["x"],
        }
    api_.instances["i9"] = {"id": "i9", "status": "deleting", "tags": [tag]}
    rows = p.instances()
    assert [r["id"] for r in rows] == ["i0", "i1", "i2", "i3", "i4"]
    assert [r["label"] for r in rows if r["label"]] == [tag]
    bad = [
        {
            "instances": [{"id": "i0", "status": "active", "tags": None}],
            "total": 2,
        },  # total changes
        {"instances": [{"id": "i0", "tags": None}], "total": 1},  # no status
        {"instances": [{"id": "i0", "status": "active", "tags": None}] * 2, "total": 2},  # twice
        {"instances": []},  # no total
    ]
    for answer in bad:
        api_.fail["GET instances"] = [answer, {"instances": [], "total": 1}]
        with pytest.raises(sf.ShadeformError):
            p.instances()
    api_.instances.clear()
    api_.fail["GET instances"] = [{"instances": None, "total": 0}]
    assert p.instances() == []


def test_a_relabel_moves_shadeforms_own_lease_and_is_believed_once_read_back():
    p, api_ = make()
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    iid = p.create(offer, SPEC, make_tag(RUN_ID, 1, EXPIRY), "ssh-ed25519 K", EXPIRY)
    later = make_tag(RUN_ID, 1, EXPIRY + 7200)
    p.relabel(iid, later)
    row = api_.instances[iid]
    assert row["tags"] == [later] and row["auto_delete"]["date_threshold"] == sf.rfc3339(
        EXPIRY + 7200
    )
    api_.fail["POST instances"] = [None]  # accepted, but nothing changed
    with pytest.raises(sf.ShadeformError, match="not confirmed"):
        p.relabel(iid, make_tag(RUN_ID, 1, EXPIRY + 9999))


def test_an_instance_is_running_for_us_once_it_has_an_address_a_port_and_a_user():
    p, _ = make()
    row = sf._row({"id": "i1", "status": "active", "ip": None, "tags": None})
    assert p.view(row).state == "loading"
    row.update(ip="1.2.3.4", ssh_port=22, ssh_user="shadeform")
    assert p.view(row).endpoint == ("1.2.3.4", 22, "shadeform")
    for status, state in (("pending", "loading"), ("deleting", "exited"), ("error", "exited")):
        row["actual_status"] = status
        assert p.view(row).state == state


# ------------------------------------------------------------------ the VM box


def test_the_startup_script_keys_every_login_and_starts_the_image_under_our_command():
    script = vmbox.startup_script("ssh-ed25519 AAAA gpubox", "my/img:1")
    assert subprocess.run(["bash", "-n"], input=script.encode()).returncode == 0
    assert "docker run -d --name gpubox" in script and "--gpus all" in script
    assert "--entrypoint bash my/img:1 -c" in script and vmbox.PULL_ERROR in script
    assert subprocess.run(["bash", "-n", "-c", onbox.CONTAINER_INIT]).returncode == 0
    for bad in ("", "-v /:/x", "a b"):
        with pytest.raises(ValueError):
            vmbox.startup_script("ssh-ed25519 K", bad)


def test_a_vm_box_is_reached_as_its_user_then_inside_its_container(tmp_path):
    (tmp_path / "id_ed25519").write_text("")
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA gpubox\n")
    p = vmbox.VmProber(tmp_path)
    argv = p.ssh(("1.2.3.4", 2200, "shadeform"), "echo hi")
    assert "shadeform@1.2.3.4" in argv and "root@1.2.3.4" not in argv and "2200" in argv
    assert argv[-1] == vmbox.AS_ROOT + "$S docker exec -u 0 -i gpubox sh -c 'echo hi'"  # root
    assert "-F" in argv  # the user's own ssh_config does not apply, as on Vast
    assert "-it gpubox" in p.ssh(("1.2.3.4", 22, "u"), None, tty=True)[-1]
    started = []
    p._start = lambda argv, read, **kw: started.append(argv)
    p.start_ready(("1.2.3.4", 22, "u"))
    assert "docker exec -u 0 gpubox true" in started[0][-1] and vmbox.PULL_ERROR in started[0][-1]
    assert vmbox.STARTED not in started[0][-1]  # the cloud runs the script itself here


def test_where_the_cloud_cannot_run_the_script_the_first_probe_starts_it_once(tmp_path):
    import subprocess as sp

    (tmp_path / "id_ed25519").write_text("")
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA gpubox\n")
    p = vmbox.VmProber(tmp_path)
    p.boot = "echo booted\n"
    started = []
    p._start = lambda argv, read, **kw: started.append((argv, kw))
    p.start_ready(("1.2.3.4", 22, "root"))
    (_, kw), remote = started[0], started[0][0][-1]
    assert (
        kw["stdin"] == "echo booted\n" and vmbox.STARTED in remote and "setsid nohup bash" in remote
    )
    assert sp.run(["bash", "-n", "-c", remote]).returncode == 0


class VmProber:
    pubkey = "ssh-ed25519 AAAA gpubox"
    scoped = False

    def __init__(self, ready=None):
        self.checks, self.ran = [], []
        self.ready = ready or (lambda ep: (True, ""))

    def kill_all(self):
        for c in self.checks:
            c.kill()

    def start_ready(self, ep):
        self.checks.append(FakeCheck(lambda: self.ready(ep)))
        return self.checks[-1]

    def start_gate(self, ep, spec):
        self.checks.append(FakeCheck(lambda: (True, None, "", {"gpus": ["NVIDIA RTX A6000"]})))
        return self.checks[-1]

    def run(self, ep, remote, timeout=60):
        self.ran.append((ep, remote))
        return 0


def launch(tmp_path, prober=None, **kw):
    p, api_ = make()
    state = State.begin(tmp_path, RUN_ID, EXPIRY, None, {"provider": "shadeform"})
    policy = L.Policy(**{**FAST.__dict__, **kw})
    return L.Launch(p, SPEC, policy, state, prober or VmProber()), p, api_, state


def test_a_launch_rents_a_vm_and_hands_over_its_address_and_user(tmp_path):
    run, p, api_, state = launch(tmp_path, spare=0)
    (w,) = run.run({}).winners
    (box,) = State.load(tmp_path).winners
    assert (box["host"], box["port"], box["user"]) == (
        f"10.1.0.{int(w.instance_id[-4:])}",
        22,
        "shadeform",
    )
    assert w.boot_s == 480  # the expected boot is not held against the box


def test_a_long_boot_is_given_its_time_and_the_start_up_limit_counts_after_it(tmp_path):
    run, p, api_, state = launch(tmp_path, spare=0, startup_timeout=0.05, stall_timeout=0.05)
    api_.boot_looks = 30  # pending for some 0.3 s: far past the limits, well within boot_s
    assert run.run({}).winners


def test_two_vms_that_cannot_pull_the_image_make_it_a_configuration_error(tmp_path):
    TYPES.append(
        itype("massedcompute", "A6000_plus", "A6000", 57, [("desmoines-usa-1", True, "on_demand")])
    )
    try:
        said = "Error response from daemon: manifest unknown"
        run, p, api_, state = launch(tmp_path, VmProber(lambda ep: (False, said)))
        with pytest.raises(ConfigError, match="manifest unknown"):
            run.run({})
        assert all(i["status"] == "deleting" for i in api_.instances.values())
    finally:
        TYPES.pop()


def test_down_deletes_by_id_and_by_tag(tmp_path):
    run, p, api_, state = launch(tmp_path, spare=0)
    run.run({})
    offer = p.candidates(SPEC, {}, L.Policy())[0][0]
    stray = p.create(offer, SPEC, make_tag(RUN_ID, 9, EXPIRY), "ssh-ed25519 K", EXPIRY)
    other = p.create(
        offer, SPEC, make_tag("dev.other.0123456789ab", 1, EXPIRY), "ssh-ed25519 K", EXPIRY
    )
    assert L.down(p, state, FAST) == []
    assert (
        api_.instances[stray]["status"] == "deleting"
        and api_.instances[other]["status"] == "pending"
    )


def test_renew_moves_the_auto_delete_and_touches_nothing_on_the_box(tmp_path):
    run, p, api_, state = launch(tmp_path, spare=0)
    run.run({})
    later = EXPIRY + 7200
    run.prober.run = lambda *a, **k: 255  # the box cannot be reached: no matter, here
    L.renew(p, state, run.prober, later)
    assert run.prober.ran == []  # the lease is Shadeform's: nothing on the box to move
    (row,) = api_.instances.values()
    assert parse_tag(row["tags"][0]).expiry == later


def test_a_vm_box_cannot_destroy_itself_so_destroy_after_is_refused(tmp_path):
    state = State.begin(tmp_path, RUN_ID, EXPIRY, None, {"provider": "shadeform"})
    state.publish([{"id": "inst-1", "attempt": 1, "host": "h", "port": 22, "user": "u"}])
    run = api.attach(state_dir=tmp_path)
    (box,) = run.boxes
    assert box.endpoint == ("h", 22, "u")
    with pytest.raises(ConfigError, match="cannot destroy itself"):
        box.start("true\n", destroy_after=True)
    with pytest.raises(ConfigError, match="no account lookup"):
        sf.Shadeform(FakeApi()).account()
    assert gate.read_ready  # (the probe's reading is gate's, shared with every provider)


def test_unreadable_tags_make_an_unreadable_row_not_an_untagged_one():
    assert sf._row({"id": "i1", "status": "active", "tags": None})["label"] is None
    for raw in ({"id": "i1", "status": "active"}, {"id": "i1", "status": "active", "tags": [1]}):
        assert sf._row(raw) is None


def test_a_renewal_is_confirmed_only_at_the_same_instant_whatever_the_offset():
    t = 1795502400  # 2026-11-24T06:40:00Z
    assert sf._same_time("2026-11-24T06:40:00Z", t)
    assert sf._same_time("2026-11-24T08:40:00+02:00", t)
    assert not sf._same_time("2026-11-24T06:40:00+02:00", t)
    assert not sf._same_time("2026-11-24T06:40:00", t)  # no offset: not an instant


def test_the_pull_verdict_is_written_only_once_the_retries_are_spent():
    script = vmbox.startup_script("ssh-ed25519 K", "img:1")
    loop = script[
        script.index("for i in 1 2 3 4 5") : script.index(
            "done", script.index("for i in 1 2 3 4 5")
        )
    ]
    assert vmbox.PULL_ERROR not in loop and "--user 0:0" in script


def test_the_script_a_probe_sends_is_the_one_that_runs(tmp_path, monkeypatch):
    """The probe's own shell, run here: the script must arrive whole (read in the
    foreground), run once, and run detached."""
    import subprocess as sp
    import time as t

    (tmp_path / "id_ed25519").write_text("")
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA gpubox\n")
    monkeypatch.setattr(vmbox, "AS_ROOT", "S=; ")  # as root, here: no sudo
    monkeypatch.setattr(vmbox, "STARTED", str(tmp_path / "started"))
    p = vmbox.VmProber(tmp_path)
    p.boot = f"echo ran >> {tmp_path / 'ran'}\n"
    started = []
    p._start = lambda argv, read, **kw: started.append((argv[-1], kw["stdin"]))
    p.start_ready(("h", 22, "u"))
    remote, stdin = started[0]
    shim = pathlib.Path(__file__).parent / "bin"  # setsid, where the system has none
    env = {"PATH": f"{tmp_path}:/usr/bin:/bin:{shim}"}  # no docker here: the box is not up
    for _ in range(2):
        r = sp.run(["bash", "-c", remote], input=stdin, capture_output=True, text=True, env=env)
        assert r.returncode == 1
    for _ in range(50):
        if (tmp_path / "ran").exists():
            break
        t.sleep(0.1)
    assert (tmp_path / "ran").read_text() == "ran\n"  # once, though probed twice


def test_a_pull_verdict_is_the_probes_last_word(tmp_path):
    (tmp_path / "id_ed25519").write_text("")
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA gpubox\n")
    p = vmbox.VmProber(tmp_path)
    started = []
    p._start = lambda argv, read, **kw: started.append(argv[-1])
    p.start_ready(("h", 22, "u"))
    remote = started[0]
    assert remote.index(vmbox.PULL_ERROR) < remote.index(vmbox.BOOT_LOG)
    assert f"$S cat {vmbox.PULL_ERROR} >&2; exit 1" in remote
