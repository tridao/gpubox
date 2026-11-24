"""What can go wrong, as exceptions. The command line maps them to exit codes:
ConfigError (and StateError) to 2, InfraError to 75."""

from __future__ import annotations


class GpuboxError(RuntimeError):
    """Anything gpubox raises on purpose."""


class ConfigError(GpuboxError):
    """Our mistake (image, key, arguments): another host cannot help."""


class InfraError(GpuboxError):
    """The provider or a box did not do what was asked; another try may."""


class NoBoxes(InfraError):
    """A launch ended without the boxes asked for; everything it rented was destroyed, or
    is named in `result.left`. `result` is the launch's `launch.Result`."""

    def __init__(self, msg: str, result=None):
        super().__init__(msg)
        self.result = result


class TeardownIncomplete(InfraError):
    """`down` could not confirm that every box of the run is gone. `left` holds the ids
    still to destroy (empty if the provider could not be listed to make sure)."""

    def __init__(self, msg: str, left: list[int] | None = None):
        super().__init__(msg)
        self.left = list(left or [])


class BoxLost(InfraError):
    """A box could not be reached for longer than we were prepared to wait, or the
    provider says it is over. `status` is what the provider said, if anything."""

    def __init__(self, msg: str, status=None):
        super().__init__(msg)
        self.status = status


class JobLost(InfraError):
    """A job ended without an exit code: its box restarted, or its wrapper was killed."""


class NoSuchJob(ConfigError):
    """There is no such job on the box."""
