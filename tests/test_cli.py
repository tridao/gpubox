"""The commands end to end, against the fake provider and a fake ssh."""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from fakes import FAST, OTHER_RUN, FakeCheck, FakeProber, FakeVast, foreign_untouched

from gpubox import api, cli, gate, job
from gpubox import policy as pol
from gpubox import vast as v
from gpubox.state import State, make_tag, parse_tag, state_lock

REFERENCE = json.loads((Path(__file__).parent / "fixtures/reference_requests.json").read_text())
UP = [
    "up", "--gpu", "RTX 5090", "--max-dph-per-gpu", "1.25", "--image", "img",
    "--namespace", "quack-ci", "--name", "77-1",
]  # fmt: skip
QUACK_CI = [
    "up", "--gpu", "RTX 5090", "--num-gpus", "2", "--max-dph-per-gpu", "1.25",
    "--capability", "12.0", "--min-vram-mib", "30000", "--min-cuda", "13.2", "--ttl-min", "60",
    "--must-reach", "https://github.com/", "--must-reach", "https://pypi.org/simple/pip/",
    "--image", "tridao/quack-kernels:cu13.2-26.11.22",
    "--namespace", "quack-ci", "--name", "77-1",
]  # fmt: skip


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Wire the CLI to one fake provider; returns (vast, run) where run(argv) -> exit code."""
    vast = FakeVast({1: "ok", 2: "ok"})
    vast.key_dirs, vast.probers = [], []

    def prober(key_dir, create=True):
        key_dir.mkdir(parents=True, exist_ok=True)
        (key_dir / "id_ed25519").touch()
        vast.key_dirs.append(key_dir)
        vast.probers.append(FakeProber(vast))
        return vast.probers[-1]

    monkeypatch.setenv("VASTAI_API_KEY", "k")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("GPUBOX_ACCOUNT_ID", raising=False)
    monkeypatch.setattr(cli.v, "Vast", lambda key: vast)
    monkeypatch.setattr(api, "Prober", prober)
    monkeypatch.setattr(cli, "Policy", lambda **kw: type(FAST)(**{**FAST.__dict__, **kw}))

    def run(argv):
        state = ["--state-dir", str(tmp_path)] if argv[0] != "reap" else []
        return cli.main([*argv, *state])

    run.state_dir = tmp_path
    return vast, run


def fake_ssh(monkeypatch, rc=0, timeout=False, before=None):
    """Replace ssh. `before()` runs when the connection would be made."""
    calls = []

    class FakeSsh:
        def __init__(self, key_dir, create=True):
            if not create and not (key_dir / "id_ed25519").exists():
                raise FileNotFoundError(key_dir)
            self.key_dir = key_dir

        def ssh(self, ep, remote, **kw):
            return ["ssh", str(self.key_dir), f"{ep[0]}:{ep[1]}", remote]

    def fake_run(argv, input=None, timeout=None, **kw):
        calls.append((argv, input))
        if before:
            before()
        if fake_run.timeout:
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, fake_run.rc, b"", b"")

    fake_run.rc, fake_run.timeout = rc, timeout
    monkeypatch.setattr(api, "Prober", FakeSsh)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    return calls


class LocalSsh:
    """ssh to a box that is this machine: the remote command runs in a local bash, each
    box with a jobs directory of its own. `fail` holds what the next connections, other
    than a job's start, do instead: "down" (cannot connect) or an int (the connection drops after that many
    bytes of output, before the follower can say how the job ended)."""

    root: Path
    fail: list = []
    commands: list = []

    def __init__(self, key_dir, create=True):
        if not create and not (key_dir / "id_ed25519").exists():
            raise FileNotFoundError(key_dir)

    def ssh(self, ep, remote, **kw):
        LocalSsh.commands.append(remote)
        env = ["env", f"GPUBOX_JOBS={LocalSsh.root / ep[0]}", f"PATH={LocalSsh.path}"]
        if LocalSsh.fail and remote != "bash -s":  # a job's start is not interfered with
            what = LocalSsh.fail.pop(0)
            if what == "down":
                return ["sh", "-c", "echo 'ssh: connect to host: Connection refused' >&2; exit 255"]
            return [*env, "bash", "-c", f"({remote}) 2>/dev/null | head -c {what}; exit 255"]
        return [*env, "bash", "-c", remote]


@pytest.fixture
def local_box(world, tmp_path, monkeypatch):
    """`run` and the job commands against boxes that are this machine (see LocalSsh);
    `up` keeps the fake provider's prober."""
    import shutil as sh

    LocalSsh.root, LocalSsh.fail, LocalSsh.commands = tmp_path / "boxes", [], []
    shim = [str(Path(__file__).parent / "bin")] if sh.which("setsid") is None else []
    LocalSsh.path = ":".join([*shim, os.environ["PATH"]])
    launching = api.Prober
    monkeypatch.setattr(
        api,
        "Prober",
        lambda key_dir, create=True: (launching(key_dir) if create else LocalSsh(key_dir, create)),
    )
    monkeypatch.setattr(job, "KILL_GRACE", 1)
    return LocalSsh


def box_dir(tmp_path, n=1):
    """The jobs directory of the box on machine n."""
    return tmp_path / "boxes" / f"10.0.0.{n}"


def script(tmp_path):
    (tmp_path / "s.sh").write_text("true\n")
    return ["run", "--script", str(tmp_path / "s.sh")]


def observations(tmp_path):
    rows = pol.Observations(tmp_path / "observations.jsonl").rows
    return [(o["kind"], o["reason_code"]) for o in rows]


# ------------------------------------------------------------------- up/down


def test_up_run_down(world, tmp_path, monkeypatch, capsys):
    vast, run = world
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    assert run(UP) == 0
    assert len(vast.ours()) == 1 and "READY" in capsys.readouterr().out
    run_id = State.load(tmp_path).run
    assert f"run_id={run_id}" in out.read_text()
    assert "observations_changed=true" in out.read_text()
    assert "| healthy |" in (tmp_path / "summary.md").read_text()
    box = vast.live[vast.ours()[0]]
    tag = parse_tag(box["label"])
    assert tag.run == run_id and run_id.startswith("quack-ci.77-1.")
    assert 89 * 60 < tag.expiry - time.time() <= 90 * 60  # the default --ttl-min
    assert vast.key_dirs == [tmp_path / "keys" / run_id]

    keys = []
    monkeypatch.setattr(api, "Prober", lambda key_dir, create=True: keys.append(key_dir) or
                        LocalSsh(key_dir, create))  # fmt: skip
    LocalSsh.root, LocalSsh.fail = tmp_path / "boxes", []
    LocalSsh.path = ":".join([str(Path(__file__).parent / "bin"), os.environ["PATH"]])
    (tmp_path / "s.sh").write_text('echo "hi $GREETING"; exit 7\n')
    argv = ["run", "--script", str(tmp_path / "s.sh"), "--env", "GREETING=it's $HOME; x"]
    capsys.readouterr()
    assert run(argv) == 7
    assert "hi it's $HOME; x\n" in capsys.readouterr().out
    assert set(keys) == {tmp_path / "keys" / run_id}  # this run's key, no other

    assert run(["down"]) == 0
    assert vast.ours() == [] and foreign_untouched(vast)
    assert not (tmp_path / "keys" / run_id).exists()  # nothing of this run is left to log in to
    assert run(["down"]) == 0  # idempotent


def test_the_quack_ci_flags_send_what_the_reference_launcher_sent(world):
    """Search, create and gate input for the real CI flag set, against the pinned recording."""
    vast, run = world
    vast.behaviour.update({1: "ok", 2: "ok"})
    FakeProber.pubkey, saved = "KEY", FakeProber.pubkey
    try:
        assert run(QUACK_CI) == 0
    finally:
        FakeProber.pubkey = saved
    assert vast.queries == [REFERENCE["search"]["body"]]
    from test_vast import same_but_for_the_timer

    assert vast.payloads
    for payload in vast.payloads:
        tag = parse_tag(payload["label"])
        assert 59 * 60 < tag.expiry - time.time() <= 60 * 60  # --ttl-min 60
        assert f"echo {tag.expiry} > /etc/gpubox-expiry" in payload["args"][1]  # the timer's time
        assert same_but_for_the_timer({**payload, "label": "LABEL"}, REFERENCE["create"]["body"])
    want = REFERENCE["gate"]["want"]
    for spec in vast.probers[0].wants:
        asked = (
            spec.num_gpus,
            list(spec.gpus),
            spec.min_vram_mib,
            list(spec.capability),
            list(spec.must_reach),
        )
        assert asked == tuple(
            want[k] for k in ("num_gpus", "gpus", "min_vram_mib", "capability", "must_reach")
        )
    assert vast.probers[0].wants and run(["down"]) == 0


def test_up_without_a_healthy_box_is_an_infrastructure_failure(world):
    vast, run = world
    vast.behaviour.update({1: "bad_gpu", 2: "bad_gpu"})
    assert run(UP) == cli.EXIT_INFRA
    assert vast.ours() == []
    assert run(["down"]) == 0


def test_up_refuses_a_state_directory_whose_run_still_has_boxes(world, capsys):
    vast, run = world
    assert run(UP) == 0
    creates = vast.creates
    assert run(UP) == cli.EXIT_CONFIG
    assert "gpubox down" in capsys.readouterr().out and vast.creates == creates
    assert run(["down"]) == 0
    assert run(UP) == 0  # a closed run does not block the next one
    assert len(vast.ours()) == 1
    assert len(set(vast.key_dirs)) == 2  # and the next run has a key of its own


def test_up_refuses_a_unreadable_state(world, tmp_path, capsys):
    vast, run = world
    (tmp_path / "run.json").write_text('{"version": 1, "run": "a.b.0123456789ab", "closed": tr')
    assert run(UP) == cli.EXIT_CONFIG
    assert "cannot be read" in capsys.readouterr().out
    assert vast.creates == 0


def test_the_run_id_is_published_before_anything_is_rented(world, tmp_path, monkeypatch):
    # The teardown step is given this output: it must exist whenever a box may.
    vast, run = world
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    seen = []
    vast.hooks["create"] = lambda machine: seen.append(out.read_text())
    assert run(UP) == 0
    assert seen and all("run_id=quack-ci.77-1." in text for text in seen)


def test_hosts_in_china_are_skipped_unless_asked_for(world):
    vast, run = world
    vast._offers[0]["geolocation"] = "Anhui, CN"  # machine 1, the cheapest
    vast._offers[1]["geolocation"] = "Texas, US"
    assert run([*UP, "--spare", "0"]) == 0
    assert vast.names("create") == [("create", 2)]
    assert run(["down"]) == 0
    assert run([*UP, "--spare", "0", "--exclude-country", "none"]) == 0
    assert vast.names("create")[-1] == ("create", 1)
    assert run(["down"]) == 0
    assert run([*UP, "--exclude-country", "us", "--exclude-country", "CN"]) == cli.EXIT_INFRA
    assert len(vast.names("create")) == 2  # nothing left to rent
    assert run(["down"]) == 0
    assert run([*UP, "--exclude-country", "China"]) == cli.EXIT_CONFIG


def test_up_proves_it_can_list_before_it_rents(world):
    vast, run = world
    vast.list_fails = 1
    assert run(UP) == cli.EXIT_INFRA
    assert vast.creates == 0 and vast.names("offers", "create") == []


def test_a_ttl_that_leaves_no_lease_is_refused(world):
    vast, run = world
    assert run([*UP, "--ttl-min", "10", "--overall-timeout", "900"]) == cli.EXIT_CONFIG
    assert vast.creates == 0


# ------------------------------------------------------------ the account pin


def test_up_rents_nothing_with_a_key_of_another_account(world, monkeypatch, capsys):
    vast, run = world
    assert run([*UP, "--account-id", "999"]) == cli.EXIT_CONFIG
    assert "belongs to account 111" in capsys.readouterr().out
    assert vast.creates == 0 and vast.names("instances", "offers") == []
    monkeypatch.setenv("GPUBOX_ACCOUNT_ID", "999")
    assert run(UP) == cli.EXIT_CONFIG and vast.creates == 0
    monkeypatch.setenv("GPUBOX_ACCOUNT_ID", "111")
    assert run(UP) == 0
    assert State.load(run.state_dir).account == "111"


def test_down_with_a_key_of_another_account_is_an_error_not_a_clean_result(world, capsys):
    # The other account's listing is empty of our boxes, which says nothing.
    vast, run = world
    assert run([*UP, "--account-id", "111"]) == 0
    vast.account_id = "222"
    deletes = len(vast.names("destroy"))
    assert run(["down"]) == cli.EXIT_CONFIG
    assert "another account" in capsys.readouterr().out
    assert len(vast.ours()) == 1 and len(vast.names("destroy")) == deletes
    assert not State.load(run.state_dir).closed
    vast.account_id = "111"
    assert run(["down"]) == 0 and vast.ours() == []


def test_bad_image_is_a_configuration_error(world):
    vast, run = world
    vast.behaviour.update({1: "image", 2: "image"})
    assert run(UP) == cli.EXIT_CONFIG
    assert vast.ours() == []


def test_missing_key_is_a_configuration_error(world, monkeypatch, tmp_path):
    vast, _ = world
    monkeypatch.delenv("VASTAI_API_KEY")
    argv = ["--keys-file", str(tmp_path / "none"), *UP, "--state-dir", str(tmp_path)]
    assert cli.main(argv) == cli.EXIT_CONFIG
    assert vast.creates == 0


def test_the_launch_does_not_start_if_the_run_id_cannot_be_published(world, monkeypatch, tmp_path):
    # CI's teardown step is `down --run-id <this output>`. Without the output
    # that step can do nothing, so nothing may be rented.
    vast, run = world
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "no" / "such" / "dir" / "out"))
    assert run(UP) == cli.EXIT_CONFIG
    assert vast.creates == 0 and vast.names("create") == []
    assert not (tmp_path / "run.json").exists()  # and the directory is free for the next run


def test_nothing_after_the_decision_can_fail_the_launch(world, monkeypatch, tmp_path):
    vast, run = world
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "no" / "such" / "dir" / "s.md"))
    # The run id is written; after that the output file stops being writable.
    broken = str(tmp_path / "no" / "such" / "dir" / "out")
    vast.hooks["create"] = lambda machine: monkeypatch.setenv("GITHUB_OUTPUT", broken)

    def boom(*a, **k):
        raise TypeError("len() of NoneType")

    monkeypatch.setattr(api.L, "observe", boom)
    monkeypatch.setattr(cli, "report_result", boom)
    assert run(UP) == 0 and len(vast.ours()) == 1
    assert "run_id=" in out.read_text() and "observations_changed" not in out.read_text()
    assert run(["down"]) == 0


def test_a_report_of_any_shape_can_be_printed(world, monkeypatch, capsys):
    vast, run = world
    empty = lambda self, ep, spec: FakeCheck(lambda: (True, None, "", {}))
    monkeypatch.setattr(FakeProber, "start_gate", empty)
    assert run(UP) == 0 and "READY" in capsys.readouterr().out
    assert run(["down"]) == 0


def test_a_second_command_on_a_busy_state_directory_is_refused(world, tmp_path):
    vast, run = world
    with state_lock(tmp_path):
        assert run(UP) == cli.EXIT_CONFIG
        assert run(["down"]) == cli.EXIT_CONFIG
    assert vast.creates == 0


def test_down_reports_an_unconfirmed_teardown_as_infrastructure(world):
    vast, run = world
    assert run(UP) == 0
    vast.sticky.add(vast.live[vast.ours()[0]]["machine"])
    assert run(["down"]) == cli.EXIT_INFRA
    vast.sticky.clear()
    assert run(["down"]) == 0 and vast.ours() == []


def test_a_late_teardown_of_an_earlier_run_does_not_touch_the_next_one(world, tmp_path):
    vast, run = world
    assert run(UP) == 0
    first = State.load(tmp_path).run
    assert run(["down", "--run-id", first]) == 0
    assert run(UP) == 0
    second_box, second = vast.ours(), State.load(tmp_path).run
    before = (tmp_path / "run.json").read_text()
    assert run(["down", "--run-id", first]) == 0  # the retried teardown of the first run
    assert vast.ours() == second_box
    # By tag alone, and nothing of the directory's current run is touched.
    assert (tmp_path / "run.json").read_text() == before
    assert (tmp_path / "keys" / second).exists()
    assert run(["down"]) == 0 and vast.ours() == []


@pytest.mark.parametrize("run_id", ["", "quack-ci.nope.00000000", "../../etc/passwd", "a.b.c"])
def test_down_refuses_what_is_not_a_run_id(world, tmp_path, run_id):
    # An empty one is what a CI step passes when `up` never printed a run id:
    # that job rented nothing and must not tear down whatever else is here.
    vast, run = world
    assert run(UP) == 0
    deletes = len(vast.names("destroy"))
    assert run(["down", "--run-id", run_id]) == cli.EXIT_CONFIG
    assert len(vast.ours()) == 1 and len(vast.names("destroy")) == deletes
    assert run(["down"]) == 0


def test_down_by_run_id_without_a_state_directory_works_by_tag(world, tmp_path, capsys):
    vast, run = world
    assert 9003 in vast.live  # tagged for OTHER_RUN, whose state is nowhere
    assert run(["down", "--run-id", OTHER_RUN]) == 0
    assert 9003 not in vast.live and "carries its tag" in capsys.readouterr().out
    assert not (tmp_path / "run.json").exists()


def test_a_machine_that_failed_is_skipped_by_the_next_up(world, tmp_path):
    vast, run = world
    vast.behaviour.update({1: "bad_gpu", 2: "ok", 3: "ok"})
    vast._offers = [*vast._offers, {**vast._offers[1], "id": 1003, "machine_id": 3, "host_id": 3}]
    assert run([*UP, "--spare", "0"]) == 0
    assert run(["down"]) == 0
    first = vast.creates
    assert run([*UP, "--spare", "0"]) == 0
    # Machine 1 is blocked now, so the second launch goes straight to machine 2.
    assert vast.creates == first + 1


def test_a_create_that_got_no_answer_does_not_fail_the_teardown(world, tmp_path, capsys):
    vast, run = world
    vast.behaviour.update({1: "ghost", 2: "ok"})
    assert run(UP) == 0  # a healthy winner; the ghost is a warning
    assert "got no answer to its create" in capsys.readouterr().out
    assert run(["down"]) == 0 and vast.ours() == []
    assert State.load(tmp_path).closed


def test_down_that_cannot_finish_says_what_to_do_and_keeps_the_key(
    world, tmp_path, capsys, monkeypatch
):
    vast, run = world
    assert run(UP) == 0
    run_id, iid = State.load(tmp_path).run, vast.ours()[0]
    vast.sticky.add(vast.live[iid]["machine"])
    capsys.readouterr()
    assert run(["down"]) == cli.EXIT_INFRA
    out = capsys.readouterr().out
    assert str(iid) in out and f"gpubox down --run-id {run_id}" in out and "lease ends" in out
    assert (tmp_path / "keys" / run_id).exists() and not State.load(tmp_path).closed
    vast.sticky.clear()
    assert run(["down", "--run-id", run_id]) == 0 and vast.ours() == []
    assert not (tmp_path / "keys" / run_id).exists()


def test_run_is_refused_once_the_teardown_has_begun(world, tmp_path, monkeypatch):
    vast, run = world
    assert run(UP) == 0
    vast.sticky.add(vast.live[vast.ours()[0]]["machine"])
    assert run(["down"]) == cli.EXIT_INFRA  # not finished, so the key is still there
    calls = fake_ssh(monkeypatch)
    assert run(script(tmp_path)) == cli.EXIT_CONFIG and not calls


def test_with_an_unreadable_state_file_the_run_id_still_tears_down_by_tag(world, tmp_path):
    vast, run = world
    assert run(UP) == 0
    run_id = State.load(tmp_path).run
    (tmp_path / "run.json").write_text("{ not json")
    assert run(["down"]) == cli.EXIT_CONFIG  # we cannot even tell which run this is
    assert len(vast.ours()) == 1
    assert run(["down", "--run-id", run_id]) == 0 and vast.ours() == []
    assert (tmp_path / "run.json").read_text() == "{ not json"  # and the file is left as found


# ---------------------------------------------------------------- ssh config


@pytest.fixture
def ssh_dir(tmp_path, monkeypatch):
    d = tmp_path / "home ssh"  # a space, as in "My Drive"
    monkeypatch.setenv("GPUBOX_SSH_DIR", str(d))
    return d


def test_ssh_config_adds_an_entry_for_the_box_and_down_removes_it(world, tmp_path, ssh_dir):
    vast, run = world
    ssh_dir.mkdir()
    (ssh_dir / "config").write_text("Host mine\n    HostName example.org\n")
    assert run([*UP, "--ssh-config"]) == 0
    run_id = State.load(tmp_path).run
    mine = (ssh_dir / "config").read_text()
    assert mine == "Include gpubox_config\n\nHost mine\n    HostName example.org\n"
    entry = (ssh_dir / "gpubox_config").read_text()
    keys = (tmp_path / "keys" / run_id).resolve()
    assert f"# gpubox run {run_id}\nHost 77-1\n" in entry
    assert "    HostName 10.0.0." in entry and "    Port 2222\n" in entry
    assert f'    IdentityFile "{keys}/id_ed25519"\n' in entry
    assert f'    UserKnownHostsFile "{keys}/known_hosts"\n' in entry
    assert run(["down"]) == 0
    assert (ssh_dir / "gpubox_config").read_text() == ""
    assert (ssh_dir / "config").read_text() == mine  # the user's file is not touched again


def test_what_ssh_reads_from_the_entry_is_the_box(world, tmp_path, ssh_dir):
    # Ask ssh itself what it would do: the real parser, no connection. (Our file is read
    # directly: a relative Include resolves against ~/.ssh, which a test must not touch.)
    vast, run = world
    tmp_path = tmp_path / "my state"  # a space in the key's path, as under "My Drive"
    assert cli.main([*UP, "--ssh-config", "--state-dir", str(tmp_path)]) == 0
    r = subprocess.run(
        ["ssh", "-F", str(ssh_dir / "gpubox_config"), "-G", "77-1"], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    got = dict(line.split(" ", 1) for line in r.stdout.splitlines() if " " in line)
    run_id = State.load(tmp_path).run
    assert got["hostname"].startswith("10.0.0.") and got["port"] == "2222" and got["user"] == "root"
    assert got["identityfile"] == str((tmp_path / "keys" / run_id / "id_ed25519").resolve())
    assert got["identitiesonly"] == "yes"


def test_an_include_that_sits_inside_a_host_block_does_not_count(world, ssh_dir):
    # There it only applies to that host. Ours has to be the first directive.
    vast, run = world
    ssh_dir.mkdir()
    before = "# mine\nHost other\n    Include gpubox_config\n"
    (ssh_dir / "config").write_text(before)
    assert run([*UP, "--ssh-config"]) == 0
    assert (ssh_dir / "config").read_text() == "Include gpubox_config\n\n" + before
    assert run(["down"]) == 0 and run([*UP, "--ssh-config"]) == 0
    assert (ssh_dir / "config").read_text() == "Include gpubox_config\n\n" + before  # once
    assert run(["down"]) == 0


def test_a_config_that_is_a_symbolic_link_stays_one(world, tmp_path, ssh_dir):
    # Dotfiles: the link's target is what is edited, and the link is kept.
    vast, run = world
    ssh_dir.mkdir()
    target = tmp_path / "dotfiles-ssh-config"
    target.write_text("Host mine\n    HostName example.org\n")
    (ssh_dir / "config").symlink_to(target)
    assert run([*UP, "--ssh-config"]) == 0
    assert (ssh_dir / "config").is_symlink()
    assert target.read_text().startswith("Include gpubox_config\n\nHost mine\n")
    assert run(["down"]) == 0


def test_boxes_in_different_state_directories_do_not_lose_each_others_entries(tmp_path, ssh_dir):
    # They hold different state locks, and share these files.
    import threading

    from gpubox import sshconfig

    def add(n):
        run_id = f"dev.box{n}.{n:012x}"
        sshconfig.add(run_id, [(f"box{n}", "10.0.0.1", 2000 + n)], tmp_path / "k", tmp_path / "h")

    threads = [threading.Thread(target=add, args=(n,)) for n in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    entry = (ssh_dir / "gpubox_config").read_text()
    assert all(f"Host box{n}\n" in entry for n in range(12))
    assert (ssh_dir / "config").read_text() == "Include gpubox_config\n"
    assert not [p.name for p in ssh_dir.iterdir() if "tmp" in p.name]


def test_without_the_flag_nothing_is_written_to_the_ssh_directory(world, ssh_dir):
    vast, run = world
    assert run(UP) == 0 and run(["down"]) == 0
    assert not ssh_dir.exists()


def test_a_second_box_of_the_same_name_replaces_the_first_entry(world, tmp_path, ssh_dir):
    vast, run = world
    assert run([*UP, "--ssh-config"]) == 0
    first = State.load(tmp_path).run
    other = tmp_path / "other"
    assert cli.main([*UP, "--ssh-config", "--state-dir", str(other)]) == 0
    entry = (ssh_dir / "gpubox_config").read_text()
    assert entry.count("Host 77-1\n") == 1 and first not in entry
    assert (ssh_dir / "config").read_text().count("Include gpubox_config") == 1
    assert cli.main(["down", "--state-dir", str(other)]) == 0 and run(["down"]) == 0


def test_a_failure_to_write_the_ssh_entry_does_not_cost_the_box(world, tmp_path, ssh_dir):
    vast, run = world
    ssh_dir.write_text("a file where the directory should be")
    assert run([*UP, "--ssh-config"]) == 0 and len(vast.ours()) == 1
    assert run(["down"]) == 0


# --------------------------------------------------------------------- renew


def test_renew_moves_the_lease(world, tmp_path, capsys):
    vast, run = world
    assert run(UP) == 0
    iid = vast.ours()[0]
    before = parse_tag(vast.live[iid]["label"]).expiry
    assert run(["renew", "--ttl-min", "600"]) == 0
    after = parse_tag(vast.live[iid]["label"]).expiry
    assert 598 * 60 < after - time.time() <= 600 * 60 and after > before
    assert State.load(tmp_path).expiry == after
    assert vast.probers[-1].ran[-1][1] == v.set_expiry_command(after)
    assert "lease now ends" in capsys.readouterr().out
    assert run(["down"]) == 0 and vast.ours() == []


def test_renew_with_a_key_of_another_account_changes_nothing_on_the_box(world):
    vast, run = world
    assert run([*UP, "--account-id", "111"]) == 0
    vast.account_id = "222"  # another key in this shell, and no pin
    assert run(["renew", "--ttl-min", "600"]) == cli.EXIT_CONFIG
    assert vast.probers[-1].ran == [] and vast.names("relabel") == []
    vast.account_id = "111"
    assert run(["down"]) == 0


def test_renew_needs_a_live_box(world, tmp_path):
    vast, run = world
    assert run(["renew", "--ttl-min", "60"]) == cli.EXIT_CONFIG
    assert run(UP) == 0 and run(["down"]) == 0
    assert run(["renew", "--ttl-min", "60"]) == cli.EXIT_CONFIG
    assert vast.names("relabel") == []


def test_a_renewal_that_fails_is_an_infrastructure_error(world):
    vast, run = world
    assert run(UP) == 0
    vast.hooks["relabel"] = lambda iid: (_ for _ in ()).throw(v.VastError(500, "boom"))
    assert run(["renew", "--ttl-min", "600"]) == cli.EXIT_INFRA


# ----------------------------------------------------------------------- run


def test_run_without_a_box_is_a_configuration_error(world, tmp_path):
    _, run = world
    assert run(script(tmp_path)) == cli.EXIT_CONFIG


def test_run_after_down_has_no_box_and_no_key(world, tmp_path):
    _, run = world
    assert run(UP) == 0 and run(["down"]) == 0
    assert run(script(tmp_path)) == cli.EXIT_CONFIG


def test_run_refuses_a_unreadable_state(world, tmp_path, monkeypatch):
    _, run = world
    assert run(UP) == 0
    (tmp_path / "run.json").write_text((tmp_path / "run.json").read_text()[:-20])
    calls = fake_ssh(monkeypatch)
    assert run(script(tmp_path)) == cli.EXIT_CONFIG and not calls


@pytest.mark.parametrize("run_id", ["../../outside", "/etc", "", "a.b.c", None, 7])
def test_a_run_id_is_checked_before_it_becomes_part_of_a_path(tmp_path, run_id):
    with pytest.raises(cli.StateError):
        cli.key_dir(tmp_path, run_id)
    good = "quack-ci.77-1.0123456789ab"
    assert cli.key_dir(tmp_path, good) == tmp_path / "keys" / good


def test_run_rejects_malformed_env(world, tmp_path):
    _, run = world
    assert run(UP) == 0
    for bad in ("NOVALUE", "1X=2", "A B=1"):
        assert run([*script(tmp_path), "--env", bad]) == cli.EXIT_CONFIG


def test_the_box_ends_a_job_at_its_timeout(world, tmp_path, local_box):
    _, run = world
    assert run(UP) == 0
    (tmp_path / "s.sh").write_text("echo started; sleep 30; echo never\n")
    t0 = time.monotonic()
    assert run(["run", "--script", str(tmp_path / "s.sh"), "--timeout-min", "0.02"]) == 124
    assert time.monotonic() - t0 < 10


def test_a_box_the_provider_says_is_over_is_recorded_as_died(world, tmp_path, monkeypatch):
    vast, run = world
    assert run(UP) == 0
    fake_ssh(monkeypatch, rc=255)
    vast.live[vast.ours()[0]]["status"] = "exited"
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None), ("died_after_admission", "died")]
    died = pol.Observations(tmp_path / "observations.jsonl").rows[-1]
    assert died["gpu"] == "RTX 5090" and died["gate_version"] == gate.GATE_VERSION
    assert died["machine_id"] == vast.live[vast.ours()[0]]["machine"]


def test_an_instance_that_no_longer_exists_blames_nobody(world, tmp_path, monkeypatch):
    # Someone destroyed it: a teardown elsewhere, its own timer, the reaper.
    vast, run = world
    assert run(UP) == 0
    fake_ssh(monkeypatch, rc=255)
    vast.live.pop(vast.ours()[0])
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]


def test_lost_ssh_to_a_running_box_blames_nobody(world, tmp_path, monkeypatch):
    _, run = world
    assert run(UP) == 0
    fake_ssh(monkeypatch, rc=255)
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]


def test_lost_ssh_with_the_provider_unreachable_blames_nobody(world, tmp_path, monkeypatch):
    vast, run = world
    assert run(UP) == 0
    fake_ssh(monkeypatch, rc=255)

    def outage(iid, **kw):
        raise v.VastError(503, "unavailable")

    monkeypatch.setattr(vast, "instance", outage)
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]


@pytest.mark.parametrize(
    "status,blamed",
    [("exited", True), ("offline", True), ("loading", False), (None, False), (["exited"], False)],
)
def test_only_a_terminal_provider_status_counts_as_died(
    world, tmp_path, monkeypatch, status, blamed
):
    vast, run = world
    assert run(UP) == 0
    fake_ssh(monkeypatch, rc=255)
    monkeypatch.setattr(vast, "instance", lambda iid, **kw: {"id": iid, "actual_status": status})
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert (("died_after_admission", "died") in observations(tmp_path)) is blamed


def test_our_own_down_during_run_blames_nobody(world, tmp_path, monkeypatch):
    vast, run = world
    assert run(UP) == 0
    iid = vast.ours()[0]

    def torn_down_meanwhile():
        State.load(tmp_path).set_closing()  # what `down` writes first
        vast.live[iid]["status"] = "exited"

    fake_ssh(monkeypatch, rc=255, before=torn_down_meanwhile)
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]


def test_down_that_starts_after_the_connection_dropped_is_still_not_blamed(
    world, tmp_path, monkeypatch
):
    # SSH fails for its own reasons; before we ask the provider, a teardown
    # starts. Whatever the provider then says about the box is our doing.
    vast, run = world
    assert run(UP) == 0
    iid = vast.ours()[0]
    fake_ssh(monkeypatch, rc=255)

    def over_because_of_us(instance_id, **kw):
        State.load(tmp_path).set_closing()
        return {"id": iid, "actual_status": "exited"}

    monkeypatch.setattr(vast, "instance", over_because_of_us)
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]


# ---------------------------------------------------------------------- reap


def old_box(vast, label=None):
    label = label or make_tag("quack-ci.old.0123456789ab", 1, int(time.time()) - 3600)
    vast.live[50] = {"id": 50, "label": label, "machine": 1, "start_date": time.time() - 9e4}


def test_reap_needs_explicit_permission_to_destroy(world, capsys):
    vast, run = world
    old_box(vast)
    reap = ["reap", "--namespace", "quack-ci"]
    assert run(reap) == 0
    assert 50 in vast.live and "report only" in capsys.readouterr().out
    assert vast.names("destroy") == []
    assert run([*reap, "--destroy"]) == 0
    assert 50 not in vast.live and foreign_untouched(vast)


def test_reap_reports_an_unconfirmed_delete_as_infrastructure(world):
    vast, run = world
    old_box(vast)
    vast.sticky.add(1)
    assert run(["reap", "--namespace", "quack-ci", "--destroy"]) == cli.EXIT_INFRA
    assert 50 in vast.live


def test_reap_reports_an_unreadable_listing_as_infrastructure(world):
    vast, run = world
    old_box(vast)
    vast.list_fails = 1
    assert run(["reap", "--namespace", "quack-ci", "--destroy"]) == cli.EXIT_INFRA
    assert 50 in vast.live


def test_reap_checks_the_account_when_one_is_pinned(world):
    vast, run = world
    old_box(vast)
    reap = ["reap", "--namespace", "quack-ci", "--destroy"]
    assert run([*reap, "--account-id", "999"]) == cli.EXIT_CONFIG and 50 in vast.live
    assert vast.names("instances") == []
    assert run([*reap, "--account-id", "111"]) == 0 and 50 not in vast.live


def test_reap_needs_a_namespace(world):
    vast, run = world
    with pytest.raises(SystemExit):
        run(["reap", "--destroy"])
    assert run(["reap", "--namespace", "Not A Namespace"]) == cli.EXIT_CONFIG
    assert vast.names("instances") == []


# ----------------------------------------------------------------- arguments


@pytest.mark.parametrize(
    "bad",
    [
        ["--spare", "-1"], ["--max-attempts", "0"], ["--num-gpus", "0"], ["--ttl-min", "0"],
        ["--max-dph-per-gpu", "0"], ["--max-dph-per-gpu", "nan"], ["--overall-timeout", "inf"],
        ["--min-vram-mib", "-5"],
    ],
)  # fmt: skip
def test_nonsense_arguments_are_rejected_before_anything_is_rented(world, bad):
    vast, run = world
    with pytest.raises(SystemExit) as exc:
        run([*UP, *bad])
    assert exc.value.code == 2 and vast.creates == 0


def test_bad_capability_is_a_configuration_error(world):
    vast, run = world
    assert run([*UP, "--capability", "twelve"]) == cli.EXIT_CONFIG and vast.creates == 0


# ---------------------------------------------------------------- gate script


def test_up_runs_the_gate_script_it_is_given(world, tmp_path):
    vast, run = world
    (tmp_path / "check.sh").write_text("nvidia-smi\n")
    argv = [*UP, "--gate-script", str(tmp_path / "check.sh"), "--gate-timeout", "30"]
    assert run(argv) == 0
    spec = vast.probers[0].wants[0]
    assert (spec.gate_script, spec.gate_timeout) == ("nvidia-smi\n", 30)
    assert vast.probers[0].hooked
    assert run(["down"]) == 0


def test_an_unreadable_gate_script_rents_nothing(world, tmp_path):
    vast, run = world
    assert run([*UP, "--gate-script", str(tmp_path / "missing.sh")]) == cli.EXIT_CONFIG
    assert vast.creates == 0


def test_several_models_cpus_and_ram_reach_the_search(world):
    vast, run = world
    argv = [*UP, "--gpu", "RTX 4090", "--min-cpus", "12", "--min-ram-gb", "64"]
    assert run(argv) == 0
    q = vast.queries[0]
    assert q["gpu_name"] == {"in": ["RTX 5090", "RTX 4090"]}
    assert (q["cpu_cores_effective"], q["cpu_ram"]) == ({"gte": 12}, {"gte": 64 * 1024})
    assert vast.probers[0].wants[0].gpus == ("RTX 5090", "RTX 4090")
    assert run(["down"]) == 0


def test_a_registry_login_takes_its_token_from_the_environment(world, monkeypatch):
    vast, run = world
    monkeypatch.delenv(cli.REGISTRY_TOKEN, raising=False)
    assert run([*UP, "--registry-user", "bot"]) == cli.EXIT_CONFIG
    assert vast.creates == 0
    monkeypatch.setenv(cli.REGISTRY_TOKEN, "ghp_secret")
    assert run([*UP, "--registry-user", "bot"]) == 0
    assert {p["image_login"] for p in vast.payloads} == {"-u bot -p ghp_secret docker.io"}
    assert run(["down"]) == 0


# ------------------------------------------------------------- several boxes


def test_several_boxes_are_named_listed_and_each_reachable(
    world, tmp_path, ssh_dir, capsys, local_box
):
    vast, run = world
    vast.behaviour.update({1: "ok", 2: "ok", 3: "ok"})
    vast._offers.append(dict(vast._offers[-1], id=1003, machine_id=3, host_id=3))
    assert run([*UP, "--count", "2", "--ssh-config"]) == 0
    state = State.load(tmp_path)
    assert len(state.winners) == 2 and state.meta["count"] == 2
    out = capsys.readouterr().out
    assert "77-1-0 is instance" in out and "77-1-1 is instance" in out
    entry = (ssh_dir / "gpubox_config").read_text()
    assert entry.count(f"# gpubox run {state.run}") == 1
    for i, w in enumerate(state.winners):
        r = subprocess.run(["ssh", "-F", str(ssh_dir / "gpubox_config"), "-G", f"77-1-{i}"],
                           capture_output=True, text=True)  # fmt: skip
        assert f"hostname {w['host']}" in r.stdout
    assert run(["status"]) == 0
    status = capsys.readouterr().out
    assert "2 box(es)" in status and "77-1-0" in status and "77-1-1" in status

    (tmp_path / "which.sh").write_text('echo "box $GPUBOX_BOX_INDEX of $GPUBOX_BOX_COUNT"\n')
    which = ["run", "--script", str(tmp_path / "which.sh")]
    assert run(which) == cli.EXIT_CONFIG  # which box?
    assert local_box.commands == []
    capsys.readouterr()
    assert run([*which, "--box", "1"]) == 0
    assert "box 1 of 2\n" in capsys.readouterr().out
    w = state.winners[1]
    machine = w["host"].rsplit(".", 1)[1]
    assert len(list(box_dir(tmp_path, machine).iterdir())) == 1  # on that box, no other
    assert run([*which, "--box", "2"]) == cli.EXIT_CONFIG


def test_status_of_a_directory_without_a_run(world, capsys):
    vast, run = world
    assert run(["status"]) == 0 and "no run" in capsys.readouterr().out


def test_attempts_too_few_for_the_count_rent_nothing(world):
    vast, run = world
    assert run([*UP, "--count", "3", "--max-attempts", "2"]) == cli.EXIT_CONFIG
    assert vast.creates == 0


def test_a_box_named_like_another_runs_box_replaces_only_that_entry(tmp_path, ssh_dir):
    from gpubox import sshconfig

    k, h = tmp_path / "k", tmp_path / "h"
    sshconfig.add(
        "a.shard.000000000001", [("shard-0", "10.0.0.1", 1), ("shard-1", "10.0.0.2", 2)], k, h
    )
    sshconfig.add("a.shard-0.000000000002", [("shard-0", "10.0.0.9", 9)], k, h)
    text = (ssh_dir / "gpubox_config").read_text()
    assert "Host shard-1" in text and text.count("Host shard-0") == 1 and "10.0.0.9" in text
    sshconfig.remove("a.shard-0.000000000002")
    text = (ssh_dir / "gpubox_config").read_text()
    assert "Host shard-1" in text and "Host shard-0" not in text
    sshconfig.add("a.x.000000000003", [("shard-1", "10.0.0.7", 7)], k, h)
    text = (ssh_dir / "gpubox_config").read_text()
    assert "# gpubox run a.shard.000000000001" not in text  # nothing of it left: block gone


# ------------------------------------------------------------- detached jobs


def job_script(tmp_path, text, name="job.sh"):
    (tmp_path / name).write_text(text)
    return ["run", "--script", str(tmp_path / name)]


def the_job(tmp_path, n=1):
    (d,) = box_dir(tmp_path, n).iterdir()
    return d


def test_a_dropped_connection_is_picked_up_where_it_was(world, tmp_path, local_box, capsys):
    vast, run = world
    assert run(UP) == 0
    local_box.fail = [5, "down", 3]  # drop after 5 bytes; refused; drop after 3
    lines = "".join(f"line {i}\n" for i in range(50))
    argv = job_script(tmp_path, f"printf '%s' '{lines}'; sleep 0.5; exit 3\n")
    capsys.readouterr()
    monkeypatch_retry = pytest.MonkeyPatch()
    monkeypatch_retry.setattr(job.time, "sleep", lambda s: None)
    try:
        assert run(argv) == 3
    finally:
        monkeypatch_retry.undo()
    out = capsys.readouterr().out
    assert lines in out and out.count("line ") == 50  # every byte, once


def test_a_detached_job_is_waited_for_and_read_later(world, tmp_path, local_box, capsys):
    vast, run = world
    assert run(UP) == 0
    capsys.readouterr()
    assert run([*job_script(tmp_path, "sleep 1; echo done; exit 4\n"), "--detach"]) == 0
    job_id = capsys.readouterr().out.strip().splitlines()[-1]
    assert job.JOB_ID.fullmatch(job_id)
    assert not (the_job(tmp_path) / "exit").exists()  # it runs without us
    assert run(["wait"]) == 4  # the newest job
    capsys.readouterr()
    assert run(["logs", job_id]) == 4 and "done\n" in capsys.readouterr().out
    assert run(["jobs"]) == 0 and f"{job_id} exit 4" in capsys.readouterr().out


def test_logs_of_a_running_job_print_what_there_is(world, tmp_path, local_box, capsys):
    vast, run = world
    assert run(UP) == 0
    assert run([*job_script(tmp_path, "echo early; sleep 30\n"), "--detach"]) == 0
    time.sleep(0.5)
    capsys.readouterr()
    assert run(["logs"]) == 0 and "early\n" in capsys.readouterr().out
    assert run(["jobs"]) == 0 and " running" in capsys.readouterr().out
    t0 = time.monotonic()
    assert run(["kill"]) == 0 and time.monotonic() - t0 < 10
    assert run(["wait"]) == 128 + 15  # SIGTERM


def test_a_job_whose_wrapper_died_is_lost_not_waited_for_forever(world, tmp_path, local_box):
    # As when the container restarts under it: no exit code will ever be written.
    vast, run = world
    assert run(UP) == 0
    assert run([*job_script(tmp_path, "sleep 30\n"), "--detach"]) == 0
    time.sleep(0.5)
    d = the_job(tmp_path)
    os.kill(int((d / "wpid").read_text().split()[0]), 9)
    os.killpg(int((d / "pid").read_text()), 9)
    assert run(["wait"]) == cli.EXIT_INFRA


def test_a_job_can_destroy_its_box_when_it_ends(world, tmp_path, local_box, monkeypatch):
    vast, run = world
    assert run(UP) == 0
    marker = tmp_path / "destroyed"
    helper = tmp_path / "destroy-self"
    helper.write_text(f"#!/bin/sh\ntouch {marker}\n")
    helper.chmod(0o755)
    monkeypatch.setattr(cli.v, "DESTROY_SELF", str(helper))
    assert run([*job_script(tmp_path, "exit 9\n"), "--destroy-after"]) == 9
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.exists()


def test_secrets_reach_the_job_and_nothing_else(world, tmp_path, local_box, monkeypatch, capsys):
    vast, run = world
    assert run(UP) == 0
    monkeypatch.setenv("MY_TOKEN", "s3cr3t 'quoted' $NOT")
    (tmp_path / "vars.env").write_text("# comment\nFROM_FILE=a b\nexport QUOTED='x y'\n")
    argv = [
        *job_script(tmp_path, 'echo "[$MY_TOKEN] [$FROM_FILE] [$QUOTED] [$GPUBOX_BOX_NAME]"\n'),
        "--env", "MY_TOKEN", "--env-file", str(tmp_path / "vars.env"),
    ]  # fmt: skip
    capsys.readouterr()
    assert run(argv) == 0
    assert "[s3cr3t 'quoted' $NOT] [a b] [x y] [77-1]" in capsys.readouterr().out
    assert all("s3cr3t" not in c for c in local_box.commands)  # never on a command line
    d = the_job(tmp_path)
    assert not (d / "env").exists()  # deleted once read
    assert all("s3cr3t" not in f.read_text() for f in d.iterdir() if f.name != "log")
    monkeypatch.delenv("MY_TOKEN")
    assert run([*job_script(tmp_path, "true\n"), "--env", "MY_TOKEN"]) == cli.EXIT_CONFIG


def test_a_job_on_every_box_is_followed_on_each_and_its_worst_end_returned(
    world, tmp_path, local_box, capsys
):
    vast, run = world
    vast.behaviour.update({1: "ok", 2: "ok"})
    assert run([*UP, "--count", "2"]) == 0
    capsys.readouterr()
    argv = job_script(tmp_path, 'echo "hello from $GPUBOX_BOX_INDEX"; exit $GPUBOX_BOX_INDEX\n')
    assert run([*argv, "--all"]) == 1
    out = capsys.readouterr().out
    assert "[77-1-0] hello from 0\n" in out and "[77-1-1] hello from 1\n" in out


def test_a_box_that_is_gone_ends_the_wait(world, tmp_path, local_box, monkeypatch):
    vast, run = world
    assert run(UP) == 0
    assert run([*job_script(tmp_path, "sleep 30\n"), "--detach"]) == 0
    job_id = the_job(tmp_path).name
    vast.live.pop(vast.ours()[0])  # destroyed by someone else
    local_box.fail = ["down"] * 100
    monkeypatch.setattr(job.time, "sleep", lambda s: None)
    assert run(["wait", job_id]) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]  # gone says nothing of the machine
    assert run(["wait"]) == cli.EXIT_INFRA  # which job is the newest cannot be asked either


def test_wait_without_any_job_says_so(world, tmp_path, local_box):
    vast, run = world
    assert run(UP) == 0
    assert run(["wait"]) == cli.EXIT_CONFIG
    assert run(["logs", "20260101-000000-000-abcd"]) == cli.EXIT_CONFIG
    assert run(["logs", "../../etc"]) == cli.EXIT_CONFIG


def test_ssh_runs_a_command_on_the_box(world, tmp_path, local_box):
    vast, run = world
    assert run(UP) == 0
    argv = ["ssh", "--state-dir", str(tmp_path), "--", "sh", "-c", "exit 5"]
    assert cli.main(argv) == 5
    assert local_box.commands[-1].startswith(job.LOAD_ENV)


def test_at_its_timeout_the_whole_job_is_ended_before_its_exit_code_is_written(
    world, tmp_path, local_box
):
    vast, run = world
    assert run(UP) == 0
    pids = tmp_path / "descendant.pid"
    # The script ends on SIGTERM; a child of it ignores SIGTERM and would run on.
    script = f"sh -c 'trap \"\" TERM; while :; do sleep 0.2; done' & echo $! > {pids}; wait\n"
    assert run([*job_script(tmp_path, script), "--timeout-min", "0.02"]) == 124
    descendant = int(pids.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(descendant, 0)


def test_the_jobs_variables_cannot_change_what_the_wrapper_does(world, tmp_path, local_box, capsys):
    vast, run = world
    assert run(UP) == 0
    argv = [
        *job_script(tmp_path, 'echo "[$d] [$limit] [$destroy]"; sleep 3\n'),
        "--env", "destroy=1", "--env", "limit=0", "--env", f"d={tmp_path}",
        "--timeout-min", "0.02",
    ]  # fmt: skip
    capsys.readouterr()
    assert run(argv) == 124  # its timeout held
    assert f"[{tmp_path}] [0] [1]" in capsys.readouterr().out  # the job saw its own values
    assert not (the_job(tmp_path) / "env").exists()


def test_kill_asks_the_wrapper_and_leaves_an_ended_job_alone(world, tmp_path, local_box, capsys):
    vast, run = world
    assert run(UP) == 0
    assert run(job_script(tmp_path, "exit 0\n")) == 0
    capsys.readouterr()
    assert run(["kill"]) == 0 and "already ended" in capsys.readouterr().out
    assert run([*job_script(tmp_path, "sleep 30\n", name="b.sh"), "--detach"]) == 0
    time.sleep(0.5)
    assert run(["kill"]) == 0 and "killed: exit 143" in capsys.readouterr().out
    job_id = sorted(p.name for p in box_dir(tmp_path).iterdir())[-1]
    assert (box_dir(tmp_path) / job_id / "cancelled").exists()


def test_a_start_that_the_box_cannot_carry_out_leaves_no_secrets(
    world, tmp_path, local_box, monkeypatch
):
    vast, run = world
    assert run(UP) == 0
    broken = tmp_path / "broken-bin"
    broken.mkdir()
    (broken / "setsid").write_text("#!/bin/sh\nexit 1\n")
    (broken / "setsid").chmod(0o755)
    local_box.path = f"{broken}:{local_box.path}"
    monkeypatch.setattr(job, "START_WAIT", 1)
    monkeypatch.setenv("TOKEN", "s3cr3t")
    assert run([*job_script(tmp_path, "true\n"), "--env", "TOKEN"]) == cli.EXIT_INFRA
    assert not (the_job(tmp_path) / "env").exists()


def test_a_long_line_on_one_of_several_boxes_is_shown_in_pieces(capsys):
    import threading

    out = cli.Prefixed("b", threading.Lock())
    out.write(b"x" * (3 * cli.Prefixed.LONGEST + 5))
    assert len(out.partial) == 5
    assert capsys.readouterr().out.count("[b] ") == 3


# ------------------------------------------------------------ interruptible


def bidding(vast):
    for o in vast._offers:
        o.update(min_bid=0.5, storage_total_cost=0.01)


def test_interruptible_bids_at_the_markup(world):
    vast, run = world
    bidding(vast)
    assert run([*UP, "--interruptible", "--bid-markup", "1.5"]) == 0
    assert vast.queries[0]["type"] == "bid"
    assert {p["price"] for p in vast.payloads} == {0.75}
    assert State.load(run.state_dir).meta["interruptible"] is True
    assert run(["down"]) == 0


def test_a_stopped_interruptible_box_is_not_recorded_as_died(world, tmp_path, monkeypatch, capsys):
    vast, run = world
    bidding(vast)
    assert run([*UP, "--interruptible"]) == 0
    fake_ssh(monkeypatch, rc=255)
    vast.live[vast.ours()[0]]["status"] = "exited"
    assert run(script(tmp_path)) == cli.EXIT_INFRA
    assert observations(tmp_path) == [("healthy", None)]  # outbid is not the machine
    assert "outbid" in capsys.readouterr().out
