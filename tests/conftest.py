"""Shared fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _gpu_broker_unset(monkeypatch):
    """A developer's ARGUS_GPU_BROKER_URL alone selects the broker (cortex 0.15): keep tests off a live iris."""
    for name in ("ARGUS_GPU_COORDINATOR", "ARGUS_GPU_BROKER_URL", "ARGUS_GPU_TENANT_MANIFEST", "ARGUS_GPU_TENANT"):
        monkeypatch.delenv(name, raising=False)
