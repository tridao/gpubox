"""Jobs on a box: what is sent, and how the follower's answers are read."""

import io
import subprocess

import pytest

from gpubox import job


@pytest.mark.parametrize(
    "stderr,expect",
    [
        ("\nGPUBOX-JOB exit 0\n", job.End("exit", 0)),
        ("noise\nGPUBOX-JOB exit 137\n", job.End("exit", 137)),
        ("GPUBOX-JOB running\n", job.End("running")),
        ("GPUBOX-JOB lost\n", job.End("lost")),
        ("GPUBOX-JOB none\n", job.End("none")),
        ("", None),  # a dropped connection: the follower never said
        ("ssh: connect to host x port 22: Connection refused\n", None),
        ("GPUBOX-JOB exit\n", None),
        ("GPUBOX-JOB exit x\n", None),
        ("GPUBOX-JOB exit 1 2\n", None),
        ("GPUBOX-JOB maybe\n", None),
    ],
)
def test_the_followers_status_line(stderr, expect):
    assert job.parse_end(stderr) == expect


def test_job_ids_sort_by_time_and_are_checked():
    a, b = job.new_job_id(1795400000), job.new_job_id(1795400001)
    assert job.JOB_ID.fullmatch(a) and a[:15] < b[:15]
    for bad in ("../x", "", None, "20261123-000000-abcd; rm -rf /", 7):
        with pytest.raises(job.JobError):
            job.check_job_id(bad)


def sh(program, **kw):
    return subprocess.run(["bash", "-c", program], capture_output=True, text=True, **kw)


@pytest.mark.parametrize(
    "text",
    ["plain\n", "no final newline", "", "\n\n", "GPUBOX_ABC\n$HOME `x` 'q' \"d\"\\n\n", "é ✓\n"],
)
def test_a_file_is_written_exactly(tmp_path, text):
    path = tmp_path / "out"
    r = sh(job._heredoc(f'"{path}"', text))
    assert r.returncode == 0, r.stderr
    assert path.read_text() == text


def test_variables_are_sourced_back_as_given(tmp_path):
    env = {"A": "it's $HOME; `x`", "B": "line1\nline2", "C": ""}
    (tmp_path / "env").write_text(job.env_lines(env))
    r = subprocess.run(
        ["sh", "-c", f'. {tmp_path / "env"}; printf "%s|%s|%s" "$A" "$B" "$C"'],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.stdout == "it's $HOME; `x`|line1\nline2|"
    for bad in ({"1A": "x"}, {"A B": "x"}, {"A": "nul\0"}):
        with pytest.raises(job.JobError):
            job.env_lines(bad)


def test_the_start_puts_nothing_of_the_job_on_the_command_line():
    remote, stdin = job.start_command("20261123-000000-abcd", "echo SCRIPT\n", {"T": "SECRET"})
    assert remote == "bash -s"
    assert "SECRET" in stdin and "SCRIPT" in stdin


def test_variables_come_from_files_then_the_command_line(tmp_path):
    (tmp_path / "a.env").write_text("# c\n\nA=1\nexport B='two words'\nC=\"q\"\n")
    env = job.env_from(["C=cli", "D"], [str(tmp_path / "a.env")], environ={"D": "from env"})
    assert env == {"A": "1", "B": "two words", "C": "cli", "D": "from env"}
    for items, files in ((["1X=2"], []), (["UNSET"], []), ([], [str(tmp_path / "none")])):
        with pytest.raises(job.JobError):
            job.env_from(items, files, environ={})
    (tmp_path / "bad.env").write_text("no equals sign\n")
    with pytest.raises(job.JobError):
        job.env_from([], [str(tmp_path / "bad.env")], environ={})


def test_following_a_job_that_is_still_running_stops_at_what_there_is(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUBOX_JOBS", str(tmp_path))
    d = tmp_path / "20261123-000000-abcd"
    d.mkdir()
    (d / "log").write_bytes(b"partial")
    out = io.BytesIO()
    end, offset = job.follow(lambda r: ["bash", "-c", r], d.name, out, wait=False)
    assert (end, offset, out.getvalue()) == (job.End("running"), 7, b"partial")
    out = io.BytesIO()
    end, _ = job.follow(lambda r: ["bash", "-c", r], d.name, out, wait=False, offset=4)
    assert out.getvalue() == b"ial"  # from where we were
    assert job.follow(lambda r: ["bash", "-c", r], "20261123-000000-ffff", io.BytesIO())[0] == (
        job.End("none")
    )


def test_a_box_that_cannot_be_reached_is_given_up_on_after_the_patience(monkeypatch):
    monkeypatch.setattr(job.time, "sleep", lambda s: None)
    down = lambda r: ["sh", "-c", "echo 'Connection refused' >&2; exit 255"]  # noqa: E731
    t = [0.0]
    monkeypatch.setattr(job.time, "monotonic", lambda: t.__setitem__(0, t[0] + 60) or t[0])
    end, _ = job.follow(down, "20261123-000000-abcd", io.BytesIO(), patience=600)
    assert end.kind == "unreachable" and "Connection refused" in end.detail
    end, _ = job.follow(down, "20261123-000000-abcd", io.BytesIO(), still_there=lambda: False)
    assert end.kind == "gone"
