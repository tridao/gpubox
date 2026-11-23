"""An entry in the user's SSH configuration for a dev box, so that plain `ssh NAME`,
`scp`, `rsync` and editors work.

gpubox owns one file, `~/.ssh/gpubox_config`, with one block per run. The
user's own `~/.ssh/config` gets a single `Include gpubox_config` line, at the
top (an Include further down would fall inside whatever `Host` block precedes
it), and nothing else in it is ever changed. Only written on request
(`up --ssh-config`); `down` removes the run's block again.
"""

from __future__ import annotations

import os
from pathlib import Path

OWNED = "gpubox_config"
INCLUDE = f"Include {OWNED}"


def _dir() -> Path:
    return Path(os.environ.get("GPUBOX_SSH_DIR") or Path.home() / ".ssh")


def _marker(run_id: str) -> str:
    return f"# gpubox run {run_id}"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".gpubox-tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


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


def add(run_id: str, alias: str, host: str, port: int, key: Path, known_hosts: Path) -> Path:
    """Write the block for a run and make sure the user's config includes our file."""
    remove(run_id)
    owned = _dir() / OWNED
    old = owned.read_text() if owned.exists() else ""
    others = [b for b in _blocks(old) if not _is_host(b, alias)]  # a stale box of the same name
    block = [
        _marker(run_id),
        f"Host {alias}",
        f"    HostName {host}",
        f"    Port {port}",
        "    User root",
        f"    IdentityFile {_quoted(Path(key).resolve())}",
        "    IdentitiesOnly yes",
        f"    UserKnownHostsFile {_quoted(Path(known_hosts).resolve())}",
        "    StrictHostKeyChecking accept-new",
    ]
    _write(owned, "".join("\n".join(b) + "\n" for b in [*others, block]))
    config = _dir() / "config"
    text = config.read_text() if config.exists() else ""
    if INCLUDE not in (line.strip() for line in text.splitlines()):
        _write(config, f"{INCLUDE}\n\n{text}" if text else f"{INCLUDE}\n")
    return owned


def _is_host(block: list[str], alias: str) -> bool:
    return any(line.strip() == f"Host {alias}" for line in block)


def remove(run_id: str) -> None:
    """Remove a run's block, if there is one. The Include line stays: it is harmless."""
    owned = _dir() / OWNED
    if not owned.exists():
        return
    blocks = _blocks(owned.read_text())
    kept = [b for b in blocks if b[0] != _marker(run_id)]
    if len(kept) != len(blocks):
        _write(owned, "".join("\n".join(b) + "\n" for b in kept))
