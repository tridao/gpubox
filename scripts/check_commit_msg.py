#!/usr/bin/env python3
"""Commit-message check, run by pre-commit as a commit-msg hook.

A message is a subject of at most 72 characters, then, if there is more, a blank
line and a body wrapped at 72. A body line may be longer only if it cannot be
broken: a URL, or a single word. Git's comment lines, and everything below the
scissors line that `git commit -v` adds, are not part of the message.

Usage: check_commit_msg.py FILE (the message file git passes to the hook).
"""

from __future__ import annotations

import sys

LIMIT = 72
SCISSORS = "------------------------ >8 ------------------------"
# Subjects git writes itself, which may be long through no fault of the author.
GENERATED = ("Merge ", 'Revert "', "fixup! ", "squash! ", "amend! ")


def message_lines(text: str) -> list[str]:
    """The message as git will record it: no comment lines, nothing below the
    scissors, no trailing blank lines."""
    lines = []
    for line in text.splitlines():
        if line.startswith("#") and SCISSORS in line:
            break
        if not line.startswith("#"):
            lines.append(line.rstrip())
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0]:
        lines.pop(0)
    return lines


def unbreakable(line: str) -> bool:
    return "://" in line or " " not in line.strip()


def problems(text: str) -> list[str]:
    lines = message_lines(text)
    if not lines:
        return []  # git refuses an empty message itself
    subject, out = lines[0], []
    if not subject.startswith(GENERATED) and len(subject) > LIMIT:
        out.append(f"the subject is {len(subject)} characters; at most {LIMIT}")
    if len(lines) > 1 and lines[1]:
        out.append("the subject must be followed by a blank line, then the body")
    for n, line in enumerate(lines[2:], start=3):
        if len(line) > LIMIT and not unbreakable(line):
            out.append(f"line {n} is {len(line)} characters; wrap the body at {LIMIT}")
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_commit_msg.py MESSAGE_FILE", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8", errors="replace") as f:
        found = problems(f.read())
    for p in found:
        print(f"commit message: {p}", file=sys.stderr)
    if found:
        print(
            "A subject of at most 72 characters, a blank line, then a body wrapped at 72"
            " (see AGENTS.md).",
            file=sys.stderr,
        )
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
