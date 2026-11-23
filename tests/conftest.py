"""Shared fixtures."""

import pytest

# The suite-wide peer names mask falls back to (cortex's peers): a developer's shell may export them.
SUITE_PEERS = ("ARGUS_OCR_URL", "ARGUS_CURATOR_URL", "ARGUS_DARKROOM_URL", "ARGUS_XTTS_URL")


@pytest.fixture(autouse=True)
def _gpu_broker_unset(monkeypatch):
    """A developer's ARGUS_GPU_BROKER_URL alone selects the broker (cortex 0.15): keep tests off a live iris.
    Their suite file (cortex 0.16's settings) and suite-wide peer URLs are never read either."""
    monkeypatch.setenv("ARGUS_ENV_FILE", "off")
    for name in (
        "ARGUS_GPU_COORDINATOR",
        "ARGUS_GPU_BROKER_URL",
        "ARGUS_GPU_TENANT_MANIFEST",
        "ARGUS_GPU_TENANT",
        "ARGUS_PEER",
        "ARGUS_BUILD",
        *SUITE_PEERS,
    ):
        monkeypatch.delenv(name, raising=False)
