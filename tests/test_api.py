"""The library, used as a library: against the fake provider, with jobs on this machine."""

import io
import os
from pathlib import Path

import pytest
from fakes import FAST, FakeProber, FakeVast, foreign_untouched
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
        assert gpubox.wait_all(jobs, output=out) == [0, 1]
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
