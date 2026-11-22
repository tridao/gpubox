"""Child process for the crash tests: launch against a file-backed fake provider, die on cue.

    python crash_child.py STATE_DIR CUE

CUE is `after:<journal event>` (die right after that record is durable),
`after:create` (the provider rented the box; die before the answer is used),
or `hold-lock` (take the state lock and wait).
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from fakes import FAST, RUN_ID, SPEC, FakeProber, FileVast  # noqa: E402

from gpubox.journal import Journal, state_lock  # noqa: E402
from gpubox.launch import Launch, Policy  # noqa: E402

state_dir, cue = Path(sys.argv[1]), sys.argv[2]
vast = FileVast(state_dir / "provider.json", {1: "ok"})
journal = Journal(state_dir)

with state_lock(state_dir):
    if cue == "hold-lock":
        print("locked", flush=True)
        time.sleep(60)
    journal.append("run", run_id=RUN_ID, namespace="quack-ci", image=SPEC.image, gpu=SPEC.gpu)
    real_append, real_create = journal.append, vast.create

    def append(ev, **fields):
        real_append(ev, **fields)
        if cue == "after:" + ev:
            os._exit(9)

    def create(offer_id, payload):
        iid = real_create(offer_id, payload)
        if cue == "after:create":
            os._exit(9)
        return iid

    journal.append, vast.create = append, create
    policy = Policy(**{**FAST.__dict__, "spare": 0})
    result = Launch(vast, SPEC, policy, journal, FakeProber(vast), RUN_ID).run({})
    print("winner", result.winner.instance_id, flush=True)
