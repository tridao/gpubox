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

# Data-center parts, looked for in the name with everything but letters and digits
# removed, in this order (a longer name before one it contains).
_DATACENTER = [
    ("GH200", r"gh200"), ("GB300", r"gb300"), ("GB200", r"gb200"), ("GB10", r"gb10(?!\d)"),
    ("B300", r"b300"), ("B200", r"b200"), ("B100", r"b100"),
    ("H200", r"h200"), ("H100", r"h100"), ("H800", r"h800"), ("H20", r"h20(?!\d)"),
    ("A100X", r"a100x"), ("A100", r"a100"), ("A800", r"a800"),
    ("A40", r"a40(?!\d)"), ("A30", r"a30(?!\d)"), ("A16", r"a16(?!\d)"), ("A10", r"a10(?!\d)"),
    ("L40S", r"l40s"), ("L40", r"l40(?!\d)"), ("L20", r"l20(?!\d)"), ("L4", r"l4(?!\d)"),
    ("V100S", r"v100s(?!xm)"), ("V100", r"v100"), ("P100", r"p100"), ("P40", r"p40(?!\d)"),
    ("T4", r"(?:^|tesla|nvidia)t4(?!\d)"),
]  # fmt: skip


@dataclass(frozen=True)
class Model:
    """A GPU model: its family, its form (None: not said), and the memory a name states
    (None: not said), in GB."""

    family: str
    form: str | None = None
    memory_gb: float | None = None

    @property
    def name(self) -> str:
        return self.family if self.form is None else f"{self.family}-{self.form}"


def _spaced(text: str) -> str:
    """Lower case, words apart, numbers apart from letters ("RTX4070S Ti" is
    "rtx 4070 s ti", "A100-SXM4-80GB" is "a 100 sxm 4 80 gb"), a space at each end."""
    t = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    t = re.sub(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", " ", t)
    return " " + " ".join(t.split()) + " "


def _family(text: str) -> str | None:
    t, c = _spaced(text), re.sub(r"[^a-z0-9]+", "", str(text).lower())
    m = re.search(r"rtxpro(\d{4})", c)
    if m:
        return "RTXPRO" + m.group(1)
    m = re.search(r" (?:rtx )?(\d{4}) ada ", t)
    if m:
        return f"RTX{m.group(1)}Ada"
    m = re.search(r" (?:quadro|q) rtx (\d{4}) ", t)
    if m:
        return "QuadroRTX" + m.group(1)
    if " titan rtx " in t:
        return "TitanRTX"
    if re.search(r" titan x ", t):
        return "TitanX"
    m = re.search(r" a (\d{4}) ", t)  # the RTX A-series: "RTX A6000", "A6000"
    if m:
        return "A" + m.group(1)
    m = re.search(r" (rtx|gtx) (\d{4})((?: (?:ti|super|s|d))*) ", t)
    if m:
        suffix = m.group(3).split()
        name = m.group(1).upper() + m.group(2)
        name += "Ti" * ("ti" in suffix) + "S" * bool({"s", "super"} & set(suffix))
        name += "D" * ("d" in suffix)
        return name + "Laptop" * bool(re.search(r" (?:laptop|mobile) ", t))
    for family, pattern in _DATACENTER:
        if re.search(pattern, c):
            return family
    return None


def _form(family: str, text: str) -> str | None:
    t = _spaced(text)
    if family == "RTXPRO6000":
        if re.search(r" max ?q ", t):
            return "MaxQ"
        if re.search(r" server | 6000 s ", t):
            return "Server"
        if re.search(r" ws | workstation ", t):
            return "WS"
        return None
    forms = FORMS.get(family, ())
    if "NVL" in forms and " nvl " in t:
        return "NVL"
    if re.search(r" pcie ", t):
        return "PCIe" if "PCIe" in forms else None
    if " sxm " in t or (family in ("H100", "H200") and re.search(r" hbm 3 ", t)):
        return "SXM" if "SXM" in forms else None
    return None


def _memory(text: str) -> float | None:
    m = re.search(r" (\d{2,3}) g(?:b|ib)? ", _spaced(text))
    return float(m.group(1)) if m else None


def identify(*texts, implied: bool = True) -> Model | None:
    """The model these names (a provider's several names for one GPU, most telling
    first) say, or None if none is known. `implied`: a form the names leave out but
    always mean (IMPLIED); not for what a person asks, where none means any."""
    family = next((f for f in (_family(t) for t in texts if t) if f), None)
    if family is None:
        return None
    named = [t for t in texts if t and _family(t) in (family, None)]
    form = next((f for f in (_form(family, t) for t in named) if f), None)
    if form is None and implied:
        form = IMPLIED.get(family)
    memory = next((m for m in (_memory(t) for t in named) if m), None)
    return Model(family, form, memory)


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
    return same_memory(want.memory_gb, have.memory_gb) and same_memory(want.memory_gb, memory_gb)
