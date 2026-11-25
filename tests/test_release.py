"""Only a tag matching the distribution version can be published."""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/check-release-tag.sh"


@pytest.mark.parametrize("ref", ["refs/tags/v0.2.0", "refs/heads/v0.1.0", "refs/heads/main"])
def test_release_ref_must_match_version(ref):
    result = subprocess.run(["bash", SCRIPT, ref, "0.1.0"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "Publish from tag v0.1.0" in result.stderr


def test_matching_release_tag_is_accepted():
    subprocess.run(["bash", SCRIPT, "refs/tags/v0.1.0", "0.1.0"], check=True)
