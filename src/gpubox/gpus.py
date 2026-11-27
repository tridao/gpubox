"""GPU models by one name, whatever a provider or nvidia-smi calls them.

Every provider names its GPUs its own way ("H100 SXM", "NVIDIA H100 80GB HBM3",
"1x H100 SXM5 80GB", "H100-80G-PCIe"), and nvidia-smi another. A model here is a family
("H100", "RTX4090", "RTXPRO6000") and, for the families whose form factor changes the
chip (its SMs, clocks, power, memory, NVLink), a form: "H100-SXM", "H100-PCIe",
"H100-NVL". `identify` reads a name as a model; `matches` says whether what a person
asked for (`--gpu`) is what a name says:

    H100         any H100: SXM, PCIe or NVL
    H100-SXM     only the SXM one ("H100 SXM", "h100sxm" and "H100 SXM5" too)
    A100 80GB    any A100 with 80 GB (a stated memory must agree, within 5%)
    RTX 4090     the desktop card: not the laptop chip, not the 4090 D, whatever its memory

A name that is not known as a model is matched word by word, in order, as before
(`words_match`). The gate checks what nvidia-smi says the same way, on the box: this
file is sent with it (gate.py), so it is plain Python that runs as far back as 3.7.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Families whose form factor makes a different chip, and the forms each comes in.
FORMS = {
    "H100": ("SXM", "PCIe", "NVL"),
    "H200": ("SXM", "NVL"),
    "H800": ("SXM", "PCIe"),
    "A100": ("SXM", "PCIe"),
    "A800": ("SXM", "PCIe"),
    "V100": ("SXM", "PCIe"),
    "RTXPRO6000": ("Server", "WS", "MaxQ"),
}
# The form a provider's or nvidia-smi's name means when it says none: nvidia-smi calls
# an H200 SXM "NVIDIA H200", and its NVL "NVIDIA H200 NVL".
IMPLIED = {"H200": "SXM"}

# Data-center parts, as words of a name split into letters and numbers ("L40S" is
# "l 40 s", "L40-spot" is "l 40 spot"), in this order. A "gb" after a number is memory.
_DATACENTER = [
    ("GH200", r"(?<!\d) gh 200 "), ("GB300", r"(?<!\d) gb 300 "),
    ("GB200", r"(?<!\d) gb 200 "), ("GB10", r"(?<!\d) gb 10 "),
    ("B300", r" b 300 "), ("B200", r" b 200 "), ("B100", r" b 100 "),
    ("H200", r" h 200 "), ("H100", r" h 100 "), ("H800", r" h 800 "), ("H20", r" h 20 "),
    ("A100X", r" a 100 x "), ("A100", r" a 100 "), ("A800", r" a 800 "),
    ("A40", r" a 40 "), ("A30", r" a 30 "), ("A16", r" a 16 "),
    ("A10G", r" a 10 g "), ("A10", r" a 10 "),
    ("L40S", r" l 40 s "), ("L40", r" l 40 "), ("L20", r" l 20 "), ("L4", r" l 4 "),
    ("V100S", r" v 100 s "), ("V100", r" v 100 "), ("P100", r" p 100 "), ("P40", r" p 40 "),
    ("T4", r" t 4 "),
]  # fmt: skip


@dataclass(frozen=True)
class Model:
    """A GPU model: its family, its form (None: not said), the memory a name states
    (None: not said), in GB, and its MIG slice (None: a whole GPU; "": any slice)."""

    family: str
    form: str | None = None
    memory_gb: float | None = None
    mig: str | None = None

    @property
    def name(self) -> str:
        return self.family if self.form is None else f"{self.family}-{self.form}"


def _spaced(text: str) -> str:
    """Lower case, words apart, numbers apart from letters ("RTX4070S Ti" is
    "rtx 4070 s ti", "A100-SXM4-80GB" is "a 100 sxm 4 80 gb"), a space at each end."""
    t = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    t = re.sub(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", " ", t)
    return " " + " ".join(t.split()) + " "


def _split_mig(t: str) -> tuple[str, str | None]:
    """A spaced name, and its MIG slice ("1 g 24 gb"; "" for none said; None: whole)."""
    if " mig " not in t:
        return t, None
    gpu, _, slice_ = t.partition(" mig ")
    return gpu + " ", slice_.strip()


def _laptop(t: str) -> str:
    return "Laptop" if re.search(r" (?:laptop|mobile) ", t) else ""


def _family(t: str) -> str | None:
    m = re.search(r" rtx ?pro (\d{4}) ", t)
    if m:
        return "RTXPRO" + m.group(1)
    m = re.search(r" (?:rtx )?(\d{4}) ada ", t)
    if m:
        return f"RTX{m.group(1)}Ada"
    m = re.search(r" (?:quadro ?|q )rtx (\d{4}) ", t)
    if m:
        return "QuadroRTX" + m.group(1)
    if re.search(r" titan ?rtx ", t):
        return "TitanRTX"
    if re.search(r" titan ?x ", t):
        return "TitanX"
    m = re.search(r" a (\d{4}) ", t)  # the RTX A-series: "RTX A6000", "A6000"
    if m:
        return "A" + m.group(1) + _laptop(t)
    m = re.search(r" (rtx|gtx) (\d{4})((?: (?:ti|super|s|d|tis))*) ", t)
    if m:
        suffix = " ".join(m.group(3).split()).replace("tis", "ti s").split()
        name = m.group(1).upper() + m.group(2)
        name += "Ti" * ("ti" in suffix) + "S" * bool({"s", "super"} & set(suffix))
        return name + "D" * ("d" in suffix) + _laptop(t)
    for family, pattern in _DATACENTER:
        if re.search(pattern, t):
            return family
    return None


def _form(family: str, t: str) -> str | None:
    """The form a name says, for a family whose forms are other chips. A form the family
    does not come in is kept, so that asking for it matches nothing."""
    if family == "RTXPRO6000":
        if re.search(r" max ?q ", t):
            return "MaxQ"
        if re.search(r" server | 6000 s ", t):
            return "Server"
        if re.search(r" ws | workstation ", t):
            return "WS"
        return None
    if family not in FORMS:
        return None
    if " nvl " in t:
        return "NVL"
    if " pcie " in t:
        return "PCIe"
    if " sxm " in t or (family in ("H100", "H200") and " hbm 3 " in t):
        return "SXM"
    return None


def _memory(t: str) -> float | None:
    # "80 gb", "80 g", "8 gb"; not a model's own number ("a 10 g" is the A10G).
    m = re.search(r"(?<! [a-z]) (\d{1,3}) g(?:i?b)? ", t)
    return float(m.group(1)) if m else None


def identify(*texts, implied: bool = True) -> Model | None:
    """The model these names (a provider's several names for one GPU, most telling
    first) say, or None if none is known. `implied`: a form the names leave out but
    always mean (IMPLIED); not for what a person asks, where none means any."""
    split = [_split_mig(_spaced(t)) for t in texts if t]
    family = next((f for f in (_family(t) for t, _ in split) if f), None)
    if family is None:
        return None
    named = [(t, mig) for t, mig in split if _family(t) in (family, None)]
    form = next((f for f in (_form(family, t) for t, _ in named) if f), None)
    if form is None and implied:
        form = IMPLIED.get(family)
    memory = next((m for m in (_memory(t) for t, _ in named) if m), None)
    mig = next((mig for _, mig in named if mig is not None), None)
    return Model(family, form, memory, mig)


def asked(text: str) -> Model | None:
    """What a person asked for, as a model (its form and memory only if they said them),
    or None if it names no known model."""
    return identify(text, implied=False)


def words_match(asked_text: str, *names) -> bool:
    """Every word of what was asked is in a name, in order, whatever lies between."""
    words = re.findall(r"[a-z0-9]+", str(asked_text).lower())
    if not words:
        return False
    pattern = re.compile(".*".join(map(re.escape, words)))
    return any(pattern.search(re.sub(r"[^a-z0-9]+", "", str(n).lower())) for n in names if n)


def same_memory(a: float | None, b: float | None) -> bool:
    """Do two memories agree, where both are known? Within 5%: nvidia-smi's 143771 MiB
    is an H200's 141 GB."""
    return a is None or b is None or abs(a - b) <= 0.05 * max(a, b)


def matches(asked_text: str, *names, memory_gb: float | None = None) -> bool:
    """Is a GPU these names describe what was asked for? `memory_gb`: what the GPU has,
    where it is known apart from its names (nvidia-smi's, a provider's field)."""
    want = asked(asked_text)
    if want is None:
        return words_match(asked_text, *names)
    have = identify(*names)
    if have is None or have.family != want.family:
        return False
    if want.form is not None and have.form != want.form:
        return False
    # A MIG slice is part of a GPU: only for whoever names it, and the one named.
    if (want.mig is None) != (have.mig is None):
        return False
    if want.mig and f" {want.mig} " not in f" {have.mig} ":
        return False
    return same_memory(want.memory_gb, have.memory_gb) and same_memory(want.memory_gb, memory_gb)
