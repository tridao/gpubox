# Working on gpubox

Instructions for coding agents and people working in this repository.
`CLAUDE.md` is a link to this file.

## Commit messages

A subject of at most 72 characters (aim for about 50), in the imperative,
then a blank line, then a body wrapped at 72 characters that says what
changed and why. One topic per commit. Details go in the body, never in the
subject.

```
Save the environment so any POSIX sh can read it

bash 5.2's export -p writes a value with a newline as $'...', which
dash cannot parse. Write each value single-quoted instead, and read
the file back with every sh present in the test.
```

The `commit-msg` hook (`scripts/check_commit_msg.py`) refuses a message
out of that shape. Install it once per clone: `pre-commit install`. Do not
copy the shape of older commits in the log: many predate this rule.

## Before committing

- `ruff format src tests scripts`, then commit: the hooks run ruff check and
  ruff format and fail on any change they make.
- `python3 -m pytest -q`: fake providers and fake SSH, no network, no keys.
  It takes a few minutes.
- A new guard or behaviour gets a test, and a mutation in
  `tests/mutations.py` that the test catches. Run a few with
  `python3 tests/mutations.py "part of a name"`. A mutation whose text no
  longer matches is reported STALE: update its text.
- README.md (how to use it), DESIGN.md (how it works and why) and TASKS.md
  (what is next) change with the behaviour they describe.

## Code

- Python 3.10 or later, standard library only.
- Docstrings and comments are plain declarative sentences that say what
  the code does and why.
- Everything that knows a provider is in `src/gpubox/providers/`: one
  module each, the interface (`base.py`) and the registry (`__init__.py`).
  The rest of the package names no provider except to pick one.

## Live tests

They rent real machines with real money. Take the cheapest one-GPU offer,
for a few minutes, in namespace `gpubox-test`, with a state directory
outside the repository's tracked files; end with `gpubox down`, then check
that the provider lists nothing left.

## This repository is public

Nothing internal goes into code, tests, docs or commit messages: no private
project, company or cluster names, hostnames, account or instance ids from
real runs, account balances, or keys. Test fixtures use made-up ids and the
documentation address ranges (192.0.2.0/24, 198.51.100.0/24,
203.0.113.0/24).
