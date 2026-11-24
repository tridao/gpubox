"""The providers gpubox knows, and what each can do.

Each provider class (provider.py describes the calls) also says, as class attributes:

    name, noun        "runpod", "pod"
    credential        the environment variable with its API key (None: it needs none),
                      also read from the keys file (api.Settings.keys_file)
    access            how a box is reached: "ssh" (our sshd and the run's key), "srun"
                      (a step of a Slurm job), or "vm" (SSH to a VM, then docker exec
                      into the box's container; vmbox.py)
    spawn             how a job is started on a box: "setsid", or "runner" (the box's
                      own long-lived process starts it; job.start_command)
    own_sshd          a box runs an sshd we start: one that refuses the run's key shows
                      that the image did not run our start-up command
    self_destroy      a box can end itself (its lease timer, `run --destroy-after`)
                      without an account-wide key on it
    lease             what ends a box when its lease does: "box" (its own timer),
                      "provider" (an expiry the provider keeps), or "reaper" (nothing
                      but `down` and a scheduled `reap`: `up` says so)
    observes_machines its machine ids mean the same machine from run to run, so the
                      blocklist and the reputation store apply
    default_spare     boxes raced beyond the count unless asked otherwise
    features          what of a Spec beyond the hardware it honours (FEATURES)
    requires          which of those it cannot do without ("image", "price_cap")
    renewable         its lease can be moved: `relabel` works (else `renew` is refused)
    searchable        `search` can show what `up` would try

and `from_options(options, key)` builds one from the options a run keeps in its state
file (`options()`; never a secret) and its API key.

Adding a provider: a module with such a class, listed in KINDS, and a fake of its API
that its tests drive (tests/test_runpod.py is the model); tests/test_providers.py checks
that every class declares the above.
"""

from __future__ import annotations

from . import hyperstack, massed, runpod, shadeform, slurm, vast
from .errors import ConfigError

KINDS = {
    k.name: k
    for k in (
        vast.VastProvider, slurm.Slurm, runpod.RunPod, shadeform.Shadeform, hyperstack.Hyperstack,
        massed.Massed,
    )
}  # fmt: skip

# What a Spec can ask for that not every provider can give.
FEATURES = {
    "image": "an image (--image)",
    "price_cap": "a price cap (--max-dph-per-gpu)",
    "interruptible": "an interruptible box (--interruptible)",
    "registry_login": "a registry login (--registry-user)",
    "exclude_country": "countries to skip (--exclude-country)",
    "ssh_config": "an SSH configuration entry (--ssh-config)",
    "any_gpu": "any GPU model (no --gpu)",
}


def kind(name: str):
    try:
        return KINDS[name]
    except KeyError:
        raise ConfigError(f"no such provider: {name!r} ({', '.join(KINDS)})") from None


def kind_of(meta: dict):
    """The provider a run's state file names (Vast, for a file from before there were
    others)."""
    return kind(meta.get("provider", "vast"))


def options_of(meta: dict) -> dict:
    return dict(meta.get("options") or meta.get("slurm") or {})  # "slurm": an older name


def check_spec(k, spec, *, ssh_config: bool = False, countries_given: bool = False) -> None:
    """Refuse what this provider cannot do, and insist on what it needs, before anything
    is rented. `countries_given`: the caller chose countries to skip (the default, CN,
    means nothing to a provider that has no such choice)."""
    asked = {
        "image": bool(spec.image),
        "price_cap": bool(spec.max_dph_per_gpu),
        "interruptible": spec.interruptible,
        "registry_login": spec.registry_user is not None,
        "exclude_country": countries_given,
        "ssh_config": ssh_config,
    }
    refused = [FEATURES[f] for f, on in asked.items() if on and f not in k.features]
    if refused:
        raise ConfigError(f"not on {k.name}: {', '.join(refused)}")
    needed = [FEATURES[f] for f in sorted(k.requires) if not asked.get(f)]
    if not spec.gpus and "any_gpu" not in k.features:
        needed.append("at least one GPU model (--gpu)")
    if needed:
        raise ConfigError(f"{k.name} needs {', '.join(needed)}")
