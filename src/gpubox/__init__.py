"""Rent verified GPU boxes from Vast.ai, run jobs on them, and make sure they are destroyed.

The library is `gpubox.api` (its docstring has an example); the command line, `gpubox.cli`,
is a thin layer over it.
"""

from .api import Box, Job, Run, Settings, attach, down, reap, search, set_log, up, wait_all
from .errors import (
    BoxLost,
    ConfigError,
    GpuboxError,
    InfraError,
    JobLost,
    NoBoxes,
    NoSuchJob,
    TeardownIncomplete,
)
from .launch import Policy, Spec
from .state import StateError

__all__ = [
    "Box", "BoxLost", "ConfigError", "GpuboxError", "InfraError", "Job", "JobLost", "NoBoxes",
    "NoSuchJob", "Policy", "Run", "Settings", "Spec", "StateError", "TeardownIncomplete",
    "attach", "down", "reap", "search", "set_log", "up", "wait_all",
]  # fmt: skip
