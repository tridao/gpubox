"""GPU models by one name (gpus.py): what each provider's and nvidia-smi's names say, and
what a `--gpu` matches."""

import pytest

from gpubox import gpus

# (a name as some provider, or nvidia-smi, writes it; the model it is)
NAMES = [
    # H100: three chips
    ("H100 SXM", "H100-SXM"), ("NVIDIA H100 80GB HBM3", "H100-SXM"),
    ("1x H100 SXM5 80GB", "H100-SXM"), ("H100-80G-SXM5", "H100-SXM"),
    ("H100 PCIE", "H100-PCIe"), ("NVIDIA H100 PCIe", "H100-PCIe"),
    ("H100-80G-PCIe", "H100-PCIe"), ("H100 NVL", "H100-NVL"), ("NVIDIA H100 NVL", "H100-NVL"),
    ("H100", "H100"),  # no form said: not known
    # H200: nvidia-smi says "NVIDIA H200" for the SXM one
    ("H200", "H200-SXM"), ("NVIDIA H200", "H200-SXM"), ("H200 SXM5 141GB", "H200-SXM"),
    ("H200 NVL", "H200-NVL"), ("NVIDIA H200 NVL", "H200-NVL"),
    # A100
    ("A100 SXM4", "A100-SXM"), ("NVIDIA A100-SXM4-80GB", "A100-SXM"),
    ("NVIDIA A100-SXM4-40GB", "A100-SXM"), ("A100 PCIE", "A100-PCIe"),
    ("NVIDIA A100 80GB PCIe", "A100-PCIe"), ("NVIDIA A100-PCIE-40GB", "A100-PCIe"),
    ("A100-80G-PCIe", "A100-PCIe"), ("A100X", "A100X"),
    # V100 and its V100S
    ("Tesla V100-SXM2-16GB", "V100-SXM"), ("Tesla V100-PCIE-32GB", "V100-PCIe"),
    ("Tesla V100", "V100"), ("Tesla V100S", "V100S"),
    # Blackwell and the superchips
    ("B200", "B200"), ("NVIDIA B200", "B200"), ("NVIDIA B300 SXM6 AC", "B300"),
    ("GB200", "GB200"), ("GB300 SXM6 288GB", "GB300"), ("GB10", "GB10"),
    # RTX PRO 6000 Blackwell: Server, Workstation, Max-Q
    ("RTX PRO 6000 S", "RTXPRO6000-Server"), ("RTX PRO 6000 WS", "RTXPRO6000-WS"),
    ("RTX PRO 6000 Max-Q", "RTXPRO6000-MaxQ"),
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "RTXPRO6000-Server"),
    ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "RTXPRO6000-WS"),
    ("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "RTXPRO6000-MaxQ"),
    ("RTX PRO 6000 96GB", "RTXPRO6000"), ("NVIDIA RTX PRO 4500 Blackwell", "RTXPRO4500"),
    # workstation cards
    ("RTX 6000Ada", "RTX6000Ada"), ("NVIDIA RTX 6000 Ada Generation", "RTX6000Ada"),
    ("RTX 5880Ada", "RTX5880Ada"), ("RTX A6000", "A6000"), ("RTX-A6000", "A6000"),
    ("NVIDIA RTX A4500", "A4500"), ("Q RTX 6000", "QuadroRTX6000"), ("Titan RTX", "TitanRTX"),
    # data-center inference cards
    ("L40S", "L40S"), ("NVIDIA L40", "L40"), ("NVIDIA L4", "L4"), ("Tesla T4", "T4"),
    ("A10", "A10"), ("NVIDIA A40", "A40"), ("A16", "A16"), ("Tesla P40", "P40"),
    # GeForce: Ti, SUPER, D and the laptop chips are other cards
    ("RTX 4090", "RTX4090"), ("NVIDIA GeForce RTX 4090", "RTX4090"),
    ("NVIDIA GeForce RTX 4090 48GB", "RTX4090"), ("RTX 4090D", "RTX4090D"),
    ("RTX 4070S Ti", "RTX4070TiS"), ("NVIDIA GeForce RTX 4070 Ti SUPER", "RTX4070TiS"),
    ("RTX 4080S", "RTX4080S"), ("NVIDIA GeForce RTX 4080 SUPER", "RTX4080S"),
    ("RTX 3060 laptop", "RTX3060Laptop"), ("GTX 1660 S", "GTX1660S"),
    ("NVIDIA GeForce GTX 1660 Ti", "GTX1660Ti"), ("RTX 5090", "RTX5090"),
]  # fmt: skip


@pytest.mark.parametrize("name, model", NAMES)
def test_every_providers_name_and_nvidia_smis_are_read_as_one_model(name, model):
    assert gpus.identify(name).name == model


@pytest.mark.parametrize(
    "name", ["CMP 170HX", "AMD Instinct MI300X OAM", "NVIDIA Graphics Device", "", "x"]
)
def test_a_name_that_is_no_known_model_is_none(name):
    assert gpus.identify(name) is None


def test_several_names_for_one_gpu_are_read_together_the_first_naming_the_family():
    # Verda: its model, its name, its description
    model = gpus.identify("A100 40GB", "A100 SXM4 40GB", "1x A100 SXM4 40GB")
    assert (model.name, model.memory_gb) == ("A100-SXM", 40)
    # RunPod: its id, then its short name
    assert gpus.identify("NVIDIA H100 80GB HBM3", "H100 SXM").name == "H100-SXM"


@pytest.mark.parametrize(
    "asked, name, memory, yes",
    [
        ("H100", "NVIDIA H100 PCIe", None, True),  # any H100
        ("H100", "H100 NVL", None, True),
        ("H100-SXM", "NVIDIA H100 80GB HBM3", None, True),
        ("h100sxm", "H100 SXM", None, True),
        ("H100 SXM", "NVIDIA H100 PCIe", None, False),  # another chip
        ("H100-SXM", "H100", None, False),  # a form not said is not known to be SXM
        ("H200", "H200 NVL", None, True),
        ("H200-SXM", "NVIDIA H200", None, True),
        ("A100 80GB", "NVIDIA A100-SXM4-80GB", None, True),
        ("A100 80GB", "NVIDIA A100-SXM4-40GB", None, False),
        ("A100 80GB", "A100 SXM4", None, True),  # memory not said: it may be
        ("A100 80GB", "A100 SXM4", 40.0, False),  # but where it is known it must agree
        ("H200", "NVIDIA H200", 143771 / 1024, True),  # 141 GB, within 5%
        ("RTX 4090", "NVIDIA GeForce RTX 4090", 23.99, True),
        ("RTX 4090", "RTX 4090D", None, False),
        ("RTX 4090", "NVIDIA GeForce RTX 4090 Laptop GPU", None, False),
        ("RTX 4090", "NVIDIA GeForce RTX 4090 48GB", None, True),  # memory not asked
        ("RTX PRO 6000", "RTX PRO 6000 S", None, True),
        ("RTXPRO6000-WS", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", None, True),
        ("RTXPRO6000-WS", "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", None, False),
        ("RTX 6000", "RTX PRO 6000 WS", None, False),  # another card, not a word away
        ("A100", "A100X", None, False),
        ("V100", "Tesla V100S", None, False),
        ("MI300X", "AMD Instinct MI300X OAM", None, True),  # unknown: word by word
        ("", "NVIDIA H100 PCIe", None, False),
    ],
)  # fmt: skip
def test_what_a_gpu_asked_for_matches(asked, name, memory, yes):
    assert gpus.matches(asked, name, memory_gb=memory) is yes


def test_what_is_asked_says_a_form_and_a_memory_only_if_said():
    assert gpus.asked("H200").form is None and gpus.identify("H200").form == "SXM"
    assert gpus.asked("A100 80GB").memory_gb == 80 and gpus.asked("A100").memory_gb is None
    assert gpus.asked("4090") is None  # not a model's name: matched word by word
