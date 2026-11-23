"""Child process for the crash tests: launch against a file-backed fake provider, die on cue.

    python crash_child.py STATE_DIR CUE

CUE is `after:create` (the provider rented the box; die before the answer is
used), `after:id` (the id is in the state file), `after:winner` (the hand-off
is on disk), or `hold-lock` (take the state lock and wait).
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from fakes import EXPIRY, FAST, RUN_ID, SPEC, FakeProber, FileVast  # noqa: E402

from gpubox.launch import Launch, Policy  # noqa: E402
from gpubox.state import State, state_lock  # noqa: E402

state_dir, cue = Path(sys.argv[1]), sys.argv[2]
vast = FileVast(state_dir / "provider.json", {1: "ok"})

with state_lock(state_dir):
    if cue == "hold-lock":
        print("locked", flush=True)
        time.sleep(60)
    meta = {"image": SPEC.image, "gpu": SPEC.gpu, "num_gpus": SPEC.num_gpus}
    state = State.begin(state_dir, RUN_ID, EXPIRY, None, meta)

    def dying(name, method):
        def wrapped(*a, **kw):
            method(*a, **kw)
            if cue == "after:" + name:
                os._exit(9)

        return wrapped

    state.add_id = dying("id", state.add_id)
    state.publish = dying("winner", state.publish)
    if cue == "after:create":
        vast.hooks["created"] = lambda iid: (vast._save(), os._exit(9))
    policy = Policy(**{**FAST.__dict__, "spare": 0})
    result = Launch(vast, SPEC, policy, state, FakeProber(vast)).run({})
    print("winner", result.winner.instance_id, flush=True)
