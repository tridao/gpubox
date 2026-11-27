"""What the launcher needs from a provider of boxes, whatever it is.

Every provider in providers/ implements it (the registry, `__init__.py`, lists them;
Vast's `vast.VastProvider` is the model). The launcher, `down`, `renew` and the reaper
see only this:

    name                      "vast" or "slurm"
    noun                      what one box is called there: "instance" or "job"
    candidates(spec, blocked, policy) -> (offers, note)
                              what may be rented, in the order to try it, and a
                              few words on how they were chosen. An offer is a
                              dict: `id`, `dph_total`, `machine_id` (None where the
                              provider names no machine), `failure_domain` (what
                              its box shares with others' so that they may fail
                              together: a Vast host, a cloud's region, a Lium node's
                              provider; None if not known), and where known
                              `geolocation`, `gpu_name`, `cuda_max_good`, `boot_s`;
                              Vast's also carry `host_id`, which observations record
    create(offer, spec, label, pubkey, expiry, avoid) -> id
                              rent one box, labelled with its tag, carrying its
                              lease. An id only from an explicit yes, CreateRefused
                              only on an explicit no, CreateUnknown when a box may
                              or may not exist; a configuration that no offer can
                              satisfy is a ConfigError. Sent once, unless the API
                              collapses repeats of it (Lium's idempotency key): then
                              it may be sent again, and once an answer was lost,
                              only an id settles it
    admit(id, label)          what must be done before a box is handed over, once
                              it runs (Lium: its lease set again); raises if it
                              cannot be, and the box is given up on
    instance(id, **call) -> row | None
                              None only when the provider says the box does not
                              exist (holds nothing, bills nothing); an error is
                              never read as that
    instances(**call) -> rows every box of the account, each with `id` and
                              `label`; complete, or an error
    view(row) -> View         a row, read the same way whatever the provider
    destroy(id, **call)       ask for destruction; an absent box is not an error
    relabel(id, label)        set the tag; raises unless it took effect. A label
                              is a tag, so this also moves the box's lease where
                              the provider keeps one (Slurm's time limit)
    account() -> str          who the credentials belong to
    expiry_command(expiry)    shell for the box: move its own lease timer
    manual_hint(run_id, ids)  where a person destroys what `down` could not
    describe(attempt)         a few words on where a box is, for people

`**call` is `timeout=` (seconds, for the whole call) and `retries=`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..errors import ConfigError, InfraError

__all__ = ["ConfigError", "CreateRefused", "CreateUnknown", "ProviderError", "View"]


class ProviderError(InfraError):
    """A provider's answer could not be had or could not be read. Never evidence that a
    box is gone. `code` is the API's own name for the error, where it states one."""

    code: str | None = None

    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


class CreateRefused(ProviderError):
    """The provider answered and did not rent a box (stale offer, bad request)."""


class CreateUnknown(ProviderError):
    """No usable answer: a box may or may not exist."""


@dataclass(frozen=True)
class View:
    """A box as the launcher reads it.

    state     queued: waiting its turn (a cluster's queue); nobody's fault, so only
                  the launch's own deadline applies, not the start-up or stall rules
              loading: on its way (an image being pulled); the stall and start-up
                  rules apply
              running: up; `endpoint` says where, once the provider says it
              exited: over, though the provider still shows it
    status    the provider's own word for its state, for messages
    msg       what the provider says about it (a status message, a queue reason)
    problem   None; "image": the image cannot be had (two hosts settle it);
              "machine": the machine is at fault; "config": our request can never
              be satisfied, so the launch stops
    endpoint  where SSH reaches it, once running: (host, port) on Vast,
              (login host, job id) on Slurm
    machine   the machine, where the row says it (a Slurm node)
    """

    state: str
    status: str = ""
    msg: str = ""
    problem: str | None = None
    endpoint: tuple | None = None
    machine: object = None


def _compact(text) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def gpu_matches(asked: str, *names) -> bool:
    """A model asked for, as a person writes it ("RTX 4090", "H100 SXM"), against
    RunPod's ids and names ("NVIDIA GeForce RTX 4090", "NVIDIA H100 80GB HBM3")."""
    needle = _compact(asked)
    return bool(needle) and any(needle in _compact(n) for n in names if n)
