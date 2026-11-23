"""An entry in the user's SSH configuration for a dev box, so that plain `ssh NAME`,
`scp`, `rsync` and editors work.

gpubox owns one file, `~/.ssh/gpubox_config`, with one block per run. The
user's own `~/.ssh/config` gets a single `Include gpubox_config` line at the
very top (further down it would fall inside whatever `Host` block precedes
it and apply only there); nothing else in it is changed, and if it is a
symbolic link, the file it points to is what is written. Only on request
(`up --ssh-config`); `down` removes the run's block again.

Several gpubox processes, in different state directories, share these files,
so every change is made under one lock in the SSH directory.
"""

from __future__ import annotations

import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

OWNED = "gpubox_config"
INCLUDE = f"Include {OWNED}"


def _dir() -> Path:
    return Path(os.environ.get("GPUBOX_SSH_DIR") or Path.home() / ".ssh")


@contextmanager
def _locked():
    d = _dir()
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    with open(d / (OWNED + ".lock"), "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # held for milliseconds; waiting is fine
        yield d


def _marker(run_id: str) -> str:
    return f"# gpubox run {run_id}"


def _write(path: Path, text: str) -> None:
    """Replace a file's content. A symbolic link is followed, not replaced."""
    path = Path(os.path.realpath(path))
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".gpubox-tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _blocks(text: str) -> list[list[str]]:
    """The file as blocks, each starting at a marker line (the first may have none)."""
    blocks: list[list[str]] = [[]]
    for line in text.splitlines():
        if line.startswith("# gpubox run "):
            blocks.append([])
        blocks[-1].append(line)
    return [b for b in blocks if b]


def _quoted(path) -> str:
    text = str(path)
    if '"' in text or "\n" in text:
        raise ValueError(f"cannot write this path to an SSH configuration: {text!r}")
    return f'"{text}"'


def _without_hosts(block: list[str], aliases: set[str]) -> list[str]:
    """The block without the `Host` entries for these aliases (an entry runs from its
    `Host` line to the next one); empty if no entry is left in it."""
    out, skipping = [], False
    for line in block:
        stripped = line.strip()
        if stripped.startswith("Host "):
            skipping = stripped[5:].strip() in aliases
        if not skipping:
            out.append(line)
    has_host = any(line.strip().startswith("Host ") for line in out)
    return out if has_host or not block[0].startswith("# gpubox run ") else []


def _included_at_the_top(text: str) -> bool:
    """Is our Include the first directive, so that it applies to every host?"""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped == INCLUDE
    return False


def add(run_id: str, hosts, key: Path, known_hosts: Path) -> Path:
    """Write the block for a run, a `Host` entry for each (alias, host, port) in `hosts`,
    and make sure the user's config includes our file."""
    block = [_marker(run_id)]
    for alias, host, port in hosts:
        block += [
            f"Host {alias}",
            f"    HostName {host}",
            f"    Port {port}",
            "    User root",
            f"    IdentityFile {_quoted(Path(key).resolve())}",
            "    IdentitiesOnly yes",
            f"    UserKnownHostsFile {_quoted(Path(known_hosts).resolve())}",
            "    StrictHostKeyChecking accept-new",
        ]
    aliases = set(alias for alias, _, _ in hosts)
    with _locked() as d:
        owned = d / OWNED
        old = owned.read_text() if owned.exists() else ""
        # Not this run's earlier block, and not a stale box of the same name: only that
        # box's entry goes, so another run's other boxes keep theirs.
        others = [
            kept
            for b in _blocks(old)
            if b[0] != _marker(run_id) and (kept := _without_hosts(b, aliases))
        ]
        _write(owned, "".join("\n".join(b) + "\n" for b in [*others, block]))
        config = d / "config"
        text = config.read_text() if config.exists() else ""
        if not _included_at_the_top(text):
            _write(config, f"{INCLUDE}\n\n{text}" if text else f"{INCLUDE}\n")
    return owned


def remove(run_id: str) -> None:
    """Remove a run's block, if there is one. The Include line stays: it is harmless."""
    if not (_dir() / OWNED).exists():
        return
    with _locked() as d:
        owned = d / OWNED
        blocks = _blocks(owned.read_text())
        kept = [b for b in blocks if b[0] != _marker(run_id)]
        if len(kept) != len(blocks):
            _write(owned, "".join("\n".join(b) + "\n" for b in kept))
