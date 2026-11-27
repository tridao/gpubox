"""What every provider declares (providers.py), and what `up` makes of it."""

import pytest

from gpubox import api, providers
from gpubox import launch as L
from gpubox.providers import runpod as rp
from gpubox.errors import ConfigError


@pytest.mark.parametrize("name", sorted(providers.KINDS))
def test_every_provider_says_what_it_can_do(name):
    kind = providers.KINDS[name]
    assert kind.name == name and isinstance(kind.noun, str)
    assert kind.access in ("ssh", "srun", "vm") and kind.spawn in ("setsid", "runner")
    assert kind.lease in ("box", "provider", "reaper")
    assert kind.lease != "box" or kind.self_destroy  # a timer in the box must be able to
    assert type(kind.self_destroy) is bool and type(kind.observes_machines) is bool
    assert type(kind.own_sshd) is bool and type(kind.searchable) is bool
    assert type(kind.searches_lease) is bool and (kind.searchable or not kind.searches_lease)
    assert type(kind.lease_slack) is int and kind.lease_slack >= 0
    assert kind.lease == "provider" or kind.lease_slack == 0  # only a provider's lease lags
    assert kind.features <= set(providers.FEATURES) and kind.requires <= kind.features
    assert type(kind.default_spare) is int and kind.default_spare >= 0
    names = kind.credential if isinstance(kind.credential, tuple) else (kind.credential,)
    assert kind.credential is None or all(n.endswith(("_KEY", "_ID", "_SECRET")) for n in names)
    for call in ("candidates", "create", "admit", "instance", "instances", "view", "destroy",
                 "relabel", "account", "expiry_command", "manual_hint", "describe",
                 "options", "from_options"):  # fmt: skip
        assert callable(getattr(kind, call)), call


def test_up_refuses_what_a_provider_cannot_do_and_insists_on_what_it_needs():
    with pytest.raises(ConfigError, match="not on runpod: an interruptible box"):
        providers.check_spec(rp.RunPod, L.Spec(image="i", gpus=(), max_dph_per_gpu=1,
                                               interruptible=True))  # fmt: skip
    with pytest.raises(ConfigError, match="runpod needs an image"):
        providers.check_spec(rp.RunPod, L.Spec(image="", gpus=(), max_dph_per_gpu=1))
    providers.check_spec(providers.kind("slurm"), L.Spec(image="", gpus=(), max_dph_per_gpu=0))


def test_the_earlier_name_of_the_options_still_works(tmp_path):
    s = api.Settings(state_dir=tmp_path, provider="slurm", slurm={"login": "x"})
    assert s.options == {"login": "x"}
    with pytest.raises(ConfigError, match="not both"):
        api.Settings(
            state_dir=tmp_path, provider="slurm", slurm={"login": "x"}, options={"login": "y"}
        )


def test_an_authenticated_call_is_not_redirected():
    import http.server
    import threading

    from gpubox.providers import httpapi

    class Redirect(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", "http://example.invalid/steal")
            self.end_headers()

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Redirect)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        client = httpapi.Client(base, {"X-Key": "k"}, rp.RunpodError)
        with pytest.raises(rp.RunpodError) as exc:
            client.call("GET", "pods/x", retries=0)
        assert exc.value.status == 302  # never read as anything, least of all a 404
    finally:
        server.shutdown()


def test_a_model_is_matched_word_by_word_in_order():
    from gpubox.providers.base import gpu_matches

    assert gpu_matches("A100 40GB", "NVIDIA A100-SXM4-40GB")  # Verda's name, nvidia-smi's
    assert gpu_matches("RTX 4090", "NVIDIA GeForce RTX 4090") and gpu_matches("rtx4090", "RTX 4090")
    assert not gpu_matches("40GB A100", "NVIDIA A100-SXM4-40GB")
    assert not gpu_matches("H100 SXM", "NVIDIA H100 80GB HBM3") and not gpu_matches("", "x")
