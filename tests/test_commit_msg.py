"""The commit-message hook (scripts/check_commit_msg.py)."""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "check_commit_msg.py"
_spec = importlib.util.spec_from_file_location("check_commit_msg", SCRIPT)
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

GOOD = "Save the environment so any POSIX sh can read it\n\n" + "word " * 14 + "\n"


@pytest.mark.parametrize(
    "text",
    [
        "Short subject\n",
        GOOD,
        "x" * 72 + "\n",
        "Subject\n\nSee https://example.com/" + "a" * 80 + "\n",  # a URL is not broken
        "Subject\n\n" + "a" * 90 + "\n",  # nor is one long word
        "Merge branch 'a-very-long-branch-name' of github.com:someone/somewhere into main x\n",
        "Subject\n# Please enter the commit message. " + "x" * 80 + "\n",  # a comment
        "Subject\n\nBody\n# " + cm.SCISSORS + "\n" + "diff " * 40 + "\n",  # below scissors
        "\n\nSubject\n\nBody\n\n\n",  # blank lines around it are not the message
    ],
)
def test_a_well_formed_message_passes(text):
    assert cm.problems(text) == []


@pytest.mark.parametrize(
    "text,says",
    [
        ("x" * 73 + "\n", "subject is 73 characters"),
        ("Subject\nbody right below\n", "blank line"),
        ("Subject\n\n" + "word " * 16 + "\n", "line 3 is 79 characters"),
    ],
)
def test_a_message_out_of_shape_is_refused_and_says_why(text, says):
    assert any(says in p for p in cm.problems(text))


def test_the_hook_exits_nonzero_and_explains(tmp_path):
    msg = tmp_path / "COMMIT_EDITMSG"
    msg.write_text("x" * 80 + "\n")
    r = subprocess.run([sys.executable, str(SCRIPT), str(msg)], capture_output=True, text=True)
    assert r.returncode == 1 and "at most 72" in r.stderr
    msg.write_text(GOOD)
    r = subprocess.run([sys.executable, str(SCRIPT), str(msg)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stderr == ""
