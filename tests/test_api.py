"""The library, used as a library: against the fake provider, with jobs on this machine."""

import io
import os
from pathlib import Path

import pytest
from fakes import FAST, FakeProber, FakeVast, foreign_untouched, offer
from test_cli import LocalSsh

import gpubox
from gpubox import api, job
from gpubox.state import State

SPEC = gpubox.Spec(image="img", gpus=("RTX 5090",), max_dph_per_gpu=1.25)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A fake provider for `up` and the teardown; jobs run in a local bash (LocalSsh)."""
    import shutil

    vast = FakeVast({1: "ok", 2: "ok", 3: "ok"})
    monkeypatch.setenv("VASTAI_API_KEY", "k")
    monkeypatch.delenv("GPUBOX_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("GPUBOX_STORE_URL", raising=False)
    monkeypatch.setattr(api.v, "Vast", lambda key: vast)

    def prober(key_dir, create=True):
        if create:
            key_dir.mkdir(parents=True, exist_ok=True)
            (key_dir / "id_ed25519").touch()
            return FakeProber(vast)
        return LocalSsh(key_dir, create)

    monkeypatch.setattr(api, "Prober", prober)
    monkeypatch.setattr(job, "KILL_GRACE", 1)
    LocalSsh.root, LocalSsh.fail, LocalSsh.commands = tmp_path / "boxes", [], []
    shim = [str(Path(__file__).parent / "bin")] if shutil.which("setsid") is None else []
    LocalSsh.path = ":".join([*shim, os.environ["PATH"]])
    return vast


def launch(tmp_path, **kw):
    return gpubox.up(SPEC, namespace="lib", name="t", policy=FAST, state_dir=tmp_path, **kw)


def test_a_run_in_a_with_block_is_destroyed_when_the_block_ends(world, tmp_path):
    with launch(tmp_path, count=2) as run:
        assert [b.name for b in run.boxes] == ["t-0", "t-1"]
        assert len(world.ours()) == 2
    assert world.ours() == [] and foreign_untouched(world)
    assert State.load(tmp_path).closed


def test_even_when_the_block_raises(world, tmp_path):
    with pytest.raises(ZeroDivisionError):
        with launch(tmp_path):
            1 / 0
    assert world.ours() == []


def test_a_kept_run_outlives_the_block_and_can_be_attached_to(world, tmp_path):
    with launch(tmp_path) as run:
        run.keep()
    assert len(world.ours()) == 1
    again = gpubox.attach(tmp_path)
    assert again.id == run.id and again.box().instance_id == world.ours()[0]
    gpubox.down(state_dir=tmp_path, policy=FAST)
    assert world.ours() == []


def test_jobs_through_the_library(world, tmp_path):
    with launch(tmp_path, count=2) as run:
        jobs = run.start('echo "box $GPUBOX_BOX_INDEX"; exit $GPUBOX_BOX_INDEX\n')
        out = io.BytesIO()
        assert [o.code for o in gpubox.wait_all(jobs, output=out)] == [0, 1]
        assert b"[t-0] box 0\n" in out.getvalue() and b"[t-1] box 1\n" in out.getvalue()
        box = run.box(1)
        assert box.job().id == jobs[1].id  # the newest
        assert box.jobs() == [(jobs[1].id, "exit 1")]
        assert box.sh("echo $GPUBOX_JOBS")[1].strip().endswith("10.0.0.2")
        late = box.start("sleep 30\n")
        assert late.logs(io.BytesIO()) is None  # still running
        assert "killed" in late.kill()
        assert late.wait(io.BytesIO()) == 143


def test_a_launch_without_boxes_raises_with_how_it_went(world, tmp_path):
    world.behaviour.update({1: "bad_gpu", 2: "bad_gpu", 3: "bad_gpu"})
    with pytest.raises(gpubox.NoBoxes) as caught:
        launch(tmp_path)
    assert caught.value.result.winners == [] and world.ours() == []
    assert isinstance(caught.value, gpubox.InfraError)


def test_nothing_is_rented_if_the_run_id_cannot_be_handed_on(world, tmp_path):
    def refuse(run_id):
        raise gpubox.ConfigError("nowhere to put it")

    with pytest.raises(gpubox.ConfigError):
        launch(tmp_path, on_run_id=refuse)
    assert world.creates == 0


def test_what_the_library_says_can_go_elsewhere(world, tmp_path, capsys):
    said = []
    gpubox.set_log(said.append)
    try:
        with launch(tmp_path):
            pass
    finally:
        gpubox.set_log(None)
    assert any(m.startswith("run lib.t.") for m in said) and capsys.readouterr().out == ""


def test_a_teardown_that_cannot_finish_says_what_is_left(world, tmp_path):
    run = launch(tmp_path)
    world.sticky.add(world.live[world.ours()[0]]["machine"])
    with pytest.raises(gpubox.TeardownIncomplete) as caught:
        run.down(policy=FAST)
    assert caught.value.left == world.ours()


def test_attaching_to_nothing_is_a_state_error(world, tmp_path):
    with pytest.raises(gpubox.StateError):
        gpubox.attach(tmp_path)


def test_a_lost_job_and_a_missing_one_are_errors_of_their_kind(world, tmp_path):
    with launch(tmp_path) as run:
        box = run.box()
        with pytest.raises(gpubox.NoSuchJob):
            box.job()
        j = box.start("sleep 30\n")
        d = LocalSsh.root / "10.0.0.1" / j.id
        os.kill(int((d / "wpid").read_text().split()[0]), 9)
        os.killpg(int((d / "pid").read_text()), 9)
        with pytest.raises(gpubox.JobLost):
            j.wait(io.BytesIO())


def test_a_box_that_comes_back_elsewhere_is_followed_there(world, tmp_path):
    with launch(tmp_path) as run:
        box = run.box()
        real = world.instance

        def moved(iid, **kw):
            row = real(iid, **kw)
            if row is not None and "ports" in row:
                row["ports"] = {"22/tcp": [{"HostPort": "40001"}]}
            return row

        world.instance = moved
        assert box.watch()() is None  # running: not over
        assert box.port == 40001
        assert gpubox.attach(tmp_path).box().port == 40001  # and so is the state file


# ------------------------------------------------------- ownership, after review


def test_an_old_handle_cannot_touch_the_run_that_replaced_it(world, tmp_path):
    a = launch(tmp_path)
    a.down(policy=FAST)
    b = launch(tmp_path)
    labels = {i: world.live[i]["label"] for i in world.ours()}
    with pytest.raises(gpubox.StateError):
        a.renew(5)
    with pytest.raises(gpubox.StateError):
        a.boxes
    assert {i: world.live[i]["label"] for i in world.ours()} == labels  # b's tags as they were
    b.down(policy=FAST)


def test_a_failure_after_the_hand_off_tears_the_run_down(world, tmp_path):
    def interrupted(result, run):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        launch(tmp_path, on_result=interrupted)
    assert world.ours() == [] and State.load(tmp_path).closed


def test_a_launch_without_boxes_leaves_the_directory_ready_for_the_next(world, tmp_path):
    world.behaviour.update({1: "bad_gpu", 2: "bad_gpu", 3: "bad_gpu"})
    with pytest.raises(gpubox.NoBoxes):
        launch(tmp_path)
    world.behaviour.update({1: "ok", 2: "ok", 3: "ok"})
    with launch(tmp_path):  # no `down` needed in between
        pass


def test_a_reporter_cannot_change_the_outcome(world, tmp_path):
    def meddle(result, run):
        result.winners.clear()

    with launch(tmp_path, on_result=meddle) as run:
        assert run.boxes


def test_settings_and_keywords_are_not_mixed(world, tmp_path):
    s = gpubox.Settings(state_dir=tmp_path, api_key="secret-key")
    with pytest.raises(gpubox.ConfigError):
        gpubox.down(state_dir=tmp_path / "other", settings=s)
    with pytest.raises(gpubox.ConfigError):
        gpubox.attach(account_id="1", settings=s)
    assert "secret-key" not in repr(s)


def test_the_blocks_own_exception_wins_over_a_failed_teardown(world, tmp_path, capsys):
    with pytest.raises(ZeroDivisionError):
        with launch(tmp_path) as run:
            world.sticky.add(world.live[world.ours()[0]]["machine"])
            run.down = lambda policy=None: (_ for _ in ()).throw(gpubox.TeardownIncomplete("x"))
            1 / 0
    assert f"gpubox down --run-id {run.id}" in capsys.readouterr().out
    world.sticky.clear()
    gpubox.down(run.id, state_dir=tmp_path, policy=FAST)


def test_keep_holds_for_one_block_only(world, tmp_path):
    run = launch(tmp_path)
    with run:
        run.keep()
    assert world.ours()
    with run:
        pass
    assert world.ours() == []


def test_arguments_are_checked_like_the_command_lines(world, tmp_path):
    with pytest.raises(gpubox.ConfigError):
        gpubox.up(SPEC, namespace="lib", name="t", count=2, policy=gpubox.Policy(count=3),
                  state_dir=tmp_path)  # fmt: skip
    assert world.creates == 0
    with launch(tmp_path, count=2) as run:
        with pytest.raises(gpubox.ConfigError):
            run.renew(0)
        jobs = run.start("exit 0\n", boxes=[1])
        assert [j.box.index for j in jobs] == [1]
        with pytest.raises(gpubox.StateError):
            run.start("exit 0\n", boxes=[5])


def test_a_start_that_fails_on_one_box_hands_back_the_others(world, tmp_path):
    with launch(tmp_path, count=2) as run:
        LocalSsh.fail = []
        real = LocalSsh.ssh

        def second_box_down(self, ep, remote, **kw):
            if ep[0].endswith(".2") and remote == "bash -s":
                return ["sh", "-c", "exit 255"]
            return real(self, ep, remote, **kw)

        LocalSsh.ssh = second_box_down
        try:
            with pytest.raises(gpubox.BoxLost) as caught:
                run.start("exit 0\n")
        finally:
            LocalSsh.ssh = real
        assert [j.box.index for j in caught.value.jobs] == [0]


def test_a_death_is_told_as_it_is_recorded(world, tmp_path):
    told = []
    s = gpubox.Settings(state_dir=tmp_path, on_observation=lambda: told.append(1))
    run = gpubox.up(SPEC, namespace="lib", name="t", policy=FAST, settings=s)
    world.live[world.ours()[0]]["status"] = "exited"
    run.box()._record_if_died("exited")
    assert told == [1]
    gpubox.down(run.id, policy=FAST, settings=s)


def test_a_cancelled_follow_starts_no_new_connection():
    import threading

    cancel = threading.Event()
    cancel.set()
    end, _ = job.follow(lambda r: pytest.fail("connected"), "20261123-000000-000-abcd",
                        io.BytesIO(), cancel=cancel)  # fmt: skip
    assert end.kind == "cancelled"


def test_search_shows_the_bids_up_would_make(world, tmp_path):
    world._offers[:] = [
        {**offer(1, dph=0.11), "min_bid": 0.10},
        {**offer(2, dph=1.10), "min_bid": 1.0},
        {**offer(3, dph=0.05), "min_bid": 0.04},
    ]
    world.live[8000] = {"id": 8000, "label": None, "machine": 3}
    bids = gpubox.Spec(image="img", gpus=("RTX 5090",), max_dph_per_gpu=1.25, interruptible=True)
    offers, _ = gpubox.search(bids, state_dir=tmp_path)
    assert [(o["machine_id"], o["_bid"]) for o in offers] == [(1, 0.125)]  # 2 over cap, 3 ours
    world.live.pop(8000)


def test_a_move_that_could_not_be_written_is_written_later(world, tmp_path):
    from gpubox.state import state_lock

    with launch(tmp_path) as run:
        box = run.box()
        with state_lock(tmp_path):  # another command holds the directory
            box._moved("10.9.9.9", 4000)
        assert gpubox.attach(tmp_path).box().port != 4000
        box._moved("10.9.9.9", 4000)
        assert gpubox.attach(tmp_path).box().port == 4000


def test_naming_the_newest_job_follows_a_box_that_moved(world, tmp_path):
    with launch(tmp_path) as run:
        box = run.box()
        first = box.start("exit 0\n")
        real = world.instance

        def moved(iid, **kw):
            row = real(iid, **kw)
            if row is not None and "ports" in row:
                row["ports"] = {"22/tcp": [{"HostPort": "40001"}]}
            return row

        world.instance = moved
        LocalSsh.fail = ["down"]  # the old address no longer answers
        assert box.job().id == first.id and box.port == 40001
