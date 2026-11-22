"""Do the tests notice when a safety property is broken?

    python3 tests/mutations.py

Each entry breaks one behaviour in a scratch copy of the package and runs the
suite; the suite must fail. Run it after changing launch, journal or vast.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]

MUTATIONS = [
    # (what is broken, file, text, replacement)
    ("create is retried", "vast.py", "body=payload, retries=0)", "body=payload, retries=2)"),
    (
        "5xx on create read as refused",
        "vast.py",
        "refused = 400 <= exc.status < 500 and exc.status != 408",
        "refused = True",
    ),
    (
        "empty create answer read as refused",
        "vast.py",
        'if data.get("success") is False:\n                raise CreateRefused',
        'if not data.get("success"):\n                raise CreateRefused',
    ),
    (
        "any truthy contract id accepted",
        "vast.py",
        "type(contract) is int and contract > 0",
        "contract",
    ),
    (
        "API outage read as 'gone'",
        "vast.py",
        '            if exc.status == 404:\n                return None\n            raise\n        if isinstance(data, dict) and "instances"',
        '            return None\n        if isinstance(data, dict) and "instances"',
    ),
    (
        "malformed instance answer read as 'gone'",
        "vast.py",
        '        raise VastError(200, f"unexpected answer for instance {instance_id}',
        '        return None\n        raise VastError(200, f"unexpected answer for instance {instance_id}',
    ),
    (
        "unreadable listing page read as empty",
        "vast.py",
        '            if not isinstance(page, list) or data.get("success") is False:',
        "            if False:",
    ),
    (
        "unknown does not hold a slot",
        "journal.py",
        'LIVE = ("intent", "unknown", "created", "destroying")',
        'LIVE = ("intent", "created", "destroying")',
    ),
    (
        "no tail repair before append",
        "journal.py",
        '                    if f.read(1) != b"\\n":',
        "                    if False:",
    ),
    (
        "replay overwrites a known id",
        "journal.py",
        '            if a and a.instance_id is None:\n                a.status, a.instance_id = "created", int(e["instance_id"])\n        elif ev in ("destroying"',
        '            if a:\n                a.status, a.instance_id = "created", int(e["instance_id"])\n        elif ev in ("destroying"',
    ),
    (
        "closed believed with a live box",
        "journal.py",
        "        s.closed = s.closed and not s.live()",
        "        pass",
    ),
    (
        "any run id accepted as a file name",
        "journal.py",
        "            if not RUN_ID.fullmatch(run_id):",
        "            if False:",
    ),
    (
        "unjournalled id is admitted",
        "launch.py",
        '        self.journal.append("created", seq=a.seq, instance_id=a.instance_id)',
        '        _note(self.journal, "created", seq=a.seq, instance_id=a.instance_id)',
    ),
    (
        "candidate kept when hand-off fails",
        "launch.py",
        "            if not handed_off:\n                winner = None\n",
        "",
    ),
    (
        "unconfirmed delete frees the slot",
        "launch.py",
        '        else:\n            log(f"WARNING: instance {a.instance_id} is not confirmed destroyed")',
        '        else:\n            a.state = "destroyed"',
    ),
    ("no reconcile after unknown", "launch.py", "            self._reconcile(a)\n", ""),
    (
        "losers not destroyed at the end",
        "launch.py",
        '            if a.state in ("created", "destroying"):\n                try:\n                    self._destroy(a)',
        "            if False:\n                try:\n                    self._destroy(a)",
    ),
    (
        "renting window not enforced",
        "launch.py",
        "                and time.monotonic() < self.rent_until\n",
        "",
    ),
    (
        "API silence blamed on the machine",
        "launch.py",
        'a.reason = "unreachable" if answered else None',
        'a.reason = "unreachable"',
    ),
    ("price-less offers rented", "launch.py", "if _price(o) is not None]", "]"),
    (
        "cleanup interruptible",
        "launch.py",
        "    for sig in (signal.SIGINT, signal.SIGTERM):\n        old[sig] = signal.signal(sig, hold)",
        "    pass",
    ),
    (
        "label match ignores the machine",
        "launch.py",
        "        if machine_id is not None and machine is not None and machine != machine_id:\n            continue\n",
        "",
    ),
    (
        "label match ignores age",
        "launch.py",
        "        if started is not None and float(started) < not_before - 300:\n            continue\n",
        "",
    ),
    (
        "label match ignores the image",
        "launch.py",
        "        if image and img and img != image:\n            continue\n",
        "",
    ),
    (
        "ambiguous label match takes the first",
        "launch.py",
        "plausible[0] if len(plausible) == 1 else None",
        "plausible[0] if plausible else None",
    ),
    (
        "unseen create written off at once",
        "launch.py",
        "        elif now - a.ts > policy.settle:",
        "        elif True:",
    ),
    (
        "listing outage aborts teardown",
        "launch.py",
        '        except v.VastError as exc:\n            left += 1\n            log(f"ERROR: attempt {a.seq} ({a.label}) unresolved',
        '        except ZeroDivisionError as exc:\n            left += 1\n            log(f"ERROR: attempt {a.seq} ({a.label}) unresolved',
    ),
    (
        "blame without a winner",
        "launch.py",
        "    if not result.winner:\n        return\n    common",
        "    common",
    ),
    (
        "late loser blamed",
        "launch.py",
        "                a.ok, a.reason = False, None\n",
        "                a.ok = False\n",
    ),
    (
        "reap destroys without permission",
        "launch.py",
        "        if stale and destroy:\n            try:",
        "        if stale:\n            try:",
    ),
    (
        "namespace matched by prefix",
        "launch.py",
        "        ours = ns == namespace",
        "        ours = label.startswith(namespace)",
    ),
    (
        "legacy pattern matches current labels",
        "launch.py",
        "bool(legacy) and ns is None and",
        "bool(legacy) and",
    ),
    (
        "legacy pattern is a prefix match",
        "launch.py",
        "re.fullmatch(legacy, label)",
        "re.match(legacy, label)",
    ),
    (
        "run on a torn-down box",
        "cli.py",
        '    if state.closed or not box or box.status != "created":',
        "    if not w:",
    ),
    (
        "died recorded on a non-terminal status",
        "cli.py",
        '    if (status == "gone" or status in DEAD_STATUSES) and _ours_to_blame(',
        '    if status != "running" and _ours_to_blame(',
    ),
    (
        "own teardown blamed on the machine",
        "cli.py",
        '    return bool(attempt) and attempt.status == "created"',
        "    return True",
    ),
    (
        "down ignores the requested run",
        "cli.py",
        "journal = Journal(state_dir, run_id=args.run_id)",
        "journal = Journal(state_dir)",
    ),
    (
        "down outside the lock",
        "cli.py",
        "    with state_lock(state_dir):\n        journal = Journal(state_dir, run_id=args.run_id)",
        "    if True:\n        journal = Journal(state_dir, run_id=args.run_id)",
    ),
    ("up over a live run", "cli.py", "            if old.live():", "            if False:"),
    (
        "advisory failure fails the launch",
        "cli.py",
        "    try:\n        fn(*a, **kw)\n    except (OSError, ValueError, TypeError, KeyError) as exc:",
        "    try:\n        fn(*a, **kw)\n    except ZeroDivisionError as exc:",
    ),
    (
        "--dry-run ignored",
        "cli.py",
        "destroy=args.destroy and not args.dry_run",
        "destroy=args.destroy",
    ),
    (
        "unknown gate reason blamed",
        "gate.py",
        '            return False, None, f"health gate gave an unknown reason',
        '            return False, reason, f"health gate gave an unknown reason',
    ),
    ("ssh reads the user's config", "gate.py", '["ssh", "-F", "/dev/null", "-i"', '["ssh", "-i"'),
    (
        "healthy clears hard blocks",
        "policy.py",
        'block["soft"] = (0, None)  # it works now',
        'block["soft"] = block["hard"] = (0, None)  # it works now',
    ),
    (
        "hard blocks re-admitted",
        "policy.py",
        '.get("class") == "soft"]',
        '.get("class") in ("soft", "hard")]',
    ),
    ("free-text reasons accepted", "policy.py", "or (reason and reason not in REASONS)", ""),
    (
        "duplicate rows count twice",
        "policy.py",
        '            if row["id"] in seen:\n                continue\n',
        "",
    ),
    (
        "malformed cached rows reach the policy",
        "policy.py",
        '        if not _valid(row) or row.get("policy_version", 1) > POLICY_VERSION:',
        '        if row.get("policy_version", 1) > POLICY_VERSION:',
    ),
]


def main() -> int:
    missed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, fname, old, new in MUTATIONS:
            work = pathlib.Path(tmp) / "w"
            shutil.rmtree(work, ignore_errors=True)
            shutil.copytree(
                ROOT, work, ignore=shutil.ignore_patterns(".git", "__pycache__", ".gpubox")
            )
            path = work / "src/gpubox" / fname
            text = path.read_text()
            if text.count(old) != 1:
                print(f"STALE   {name}: text found {text.count(old)} times")
                missed += 1
                continue
            path.write_text(text.replace(old, new))
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"],
                cwd=work, capture_output=True, text=True, timeout=300,
            )  # fmt: skip
            first = next((l[7:] for l in r.stdout.splitlines() if l.startswith("FAILED")), "")
            print(
                f"{'caught ' if r.returncode else 'MISSED '} {name}"
                + (f"  <- {first[:70]}" if first else "")
            )
            missed += r.returncode == 0
    print(f"{len(MUTATIONS) - missed} of {len(MUTATIONS)} caught")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
