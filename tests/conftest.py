import pytest


@pytest.fixture(autouse=True)
def no_real_store(monkeypatch):
    """No test may talk to, or write fake observations into, a real reputation store
    just because the developer's shell has one configured."""
    monkeypatch.delenv("GPUBOX_STORE_URL", raising=False)
    monkeypatch.delenv("GPUBOX_STORE_TOKEN", raising=False)
