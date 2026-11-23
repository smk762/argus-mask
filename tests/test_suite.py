"""What mask takes from argus-cortex 0.16: suite-wide peer URLs, the suite file, and the /health contract."""

from __future__ import annotations

from pathlib import Path

import pytest
from argus_cortex.gpu import NullCoordinator
from argus_cortex.service import ServiceHealth
from fastapi.testclient import TestClient

from argus_mask import __version__, gpu
from argus_mask.config import (
    DEFAULT_DARKROOM_URL,
    AgeSettings,
    ConfigError,
    EditorSettings,
    PortraitSettings,
    Settings,
    SourceSettings,
)
from argus_mask.models import MASK_VERSION, Health
from argus_mask.server.app import create_app
from tests.helpers import make_face

REPO = Path(__file__).resolve().parents[1]


# ---- peer URLs: mask's variable, then the suite-wide one, then the default -------------------------


def test_mask_s_own_variable_wins_over_the_suite_wide_one(monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_CURATOR_URL", "http://suite-curator.test:8101")
    monkeypatch.setenv("ARGUS_OCR_URL", "http://suite-ocr.test:8134")
    monkeypatch.setenv("ARGUS_DARKROOM_URL", "http://suite-darkroom.test:8106")
    monkeypatch.setenv("ARGUS_MASK_CURATOR_URL", "http://own-curator.test:8101/")
    monkeypatch.setenv("ARGUS_MASK_OCR_URL", "http://own-ocr.test:8134")
    monkeypatch.setenv("ARGUS_MASK_DARKROOM_URL", "http://own-darkroom.test:8106/")
    portrait = PortraitSettings.from_env()
    assert (portrait.curator_url, portrait.ocr_url) == ("http://own-curator.test:8101", "http://own-ocr.test:8134")
    assert AgeSettings.from_env().curator_url == "http://own-curator.test:8101"
    assert SourceSettings.from_env().darkroom_url == "http://own-darkroom.test:8106"


def test_unset_falls_back_to_the_suite_wide_name_then_the_default(monkeypatch) -> None:
    for name in ("ARGUS_MASK_CURATOR_URL", "ARGUS_MASK_OCR_URL", "ARGUS_MASK_DARKROOM_URL"):
        monkeypatch.delenv(name, raising=False)
    portrait = PortraitSettings.from_env()
    assert (portrait.curator_url, portrait.ocr_url) == ("", "")  # still unset: the checks stay not_checked
    assert AgeSettings.from_env().curator_url == ""
    assert SourceSettings.from_env().darkroom_url == DEFAULT_DARKROOM_URL

    monkeypatch.setenv("ARGUS_MASK_CURATOR_URL", "  ")  # blank (compose's ${NAME:-}) is unset
    monkeypatch.setenv("ARGUS_CURATOR_URL", "http://suite-curator.test:8101")
    monkeypatch.setenv("ARGUS_OCR_URL", "http://suite-ocr.test:8134")
    monkeypatch.setenv("ARGUS_DARKROOM_URL", "http://suite-darkroom.test:8106")
    portrait = PortraitSettings.from_env()
    assert (portrait.curator_url, portrait.ocr_url) == ("http://suite-curator.test:8101", "http://suite-ocr.test:8134")
    assert AgeSettings.from_env().curator_url == "http://suite-curator.test:8101"
    assert SourceSettings.from_env().darkroom_url == "http://suite-darkroom.test:8106"


def test_off_turns_a_peer_off_for_mask_whatever_the_suite_says(monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_CURATOR_URL", "http://suite-curator.test:8101")
    monkeypatch.setenv("ARGUS_MASK_CURATOR_URL", "off")
    assert PortraitSettings.from_env().curator_url == ""


@pytest.mark.parametrize("name", ["ARGUS_MASK_OCR_URL", "ARGUS_OCR_URL"])
def test_a_bad_peer_url_is_a_config_error_naming_the_variable_never_unset(monkeypatch, name) -> None:
    monkeypatch.delenv("ARGUS_MASK_OCR_URL", raising=False)
    monkeypatch.setenv(name, "made-up-user:made-up-word@ocr.test/api")
    with pytest.raises(ConfigError) as refused:
        PortraitSettings.from_env()
    assert name in str(refused.value) and "made-up" not in str(refused.value)


def test_the_plate_command_finds_darkroom_the_same_way(monkeypatch) -> None:
    from argus_mask.cli import _darkroom_url

    monkeypatch.delenv("ARGUS_MASK_DARKROOM_URL", raising=False)
    assert _darkroom_url("") == DEFAULT_DARKROOM_URL
    monkeypatch.setenv("ARGUS_DARKROOM_URL", "http://suite-darkroom.test:8106")
    assert _darkroom_url("") == "http://suite-darkroom.test:8106"
    monkeypatch.setenv("ARGUS_MASK_DARKROOM_URL", "http://own-darkroom.test:8106")
    assert _darkroom_url("") == "http://own-darkroom.test:8106"
    assert _darkroom_url("http://given.test:8106") == "http://given.test:8106"


# ---- the TTS endpoint: mask's own variable as written, else the suite's XTTS engine ----------------


def test_the_tts_variable_is_used_as_written_and_wins(monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_XTTS_URL", "http://suite-xtts.test:8141")
    monkeypatch.setenv("ARGUS_MASK_TTS_URL", "wyoming://tts.test:10200")  # not http: never through peers
    assert Settings.from_env().tts_url == "wyoming://tts.test:10200"
    monkeypatch.setenv("ARGUS_MASK_TTS_URL", "http://tts.test:8034/synthesise/stream")
    assert Settings.from_env().tts_url == "http://tts.test:8034/synthesise/stream"


def test_an_unset_tts_falls_back_to_the_suite_s_xtts_stream(monkeypatch) -> None:
    monkeypatch.delenv("ARGUS_MASK_TTS_URL", raising=False)
    assert Settings.from_env().tts_url == ""
    monkeypatch.setenv("ARGUS_XTTS_URL", "http://suite-xtts.test:8141/")
    assert Settings.from_env().tts_url == "http://suite-xtts.test:8141/synthesise/stream"
    monkeypatch.setenv("ARGUS_MASK_TTS_URL", "off")
    assert Settings.from_env().tts_url == ""
    monkeypatch.delenv("ARGUS_MASK_TTS_URL")
    monkeypatch.setenv("ARGUS_XTTS_URL", "suite-xtts.test:8141")
    with pytest.raises(ConfigError, match="ARGUS_XTTS_URL"):
        Settings.from_env()


# ---- the suite file ---------------------------------------------------------------------------------


def test_the_broker_and_a_peer_are_read_from_the_suite_file_and_the_environment_wins(monkeypatch, tmp_path) -> None:
    suite = tmp_path / "suite.env"
    suite.write_text("ARGUS_GPU_BROKER_URL=http://127.0.0.1:9\nARGUS_CURATOR_URL=http://file-curator.test:8101\n")
    suite.chmod(0o600)
    monkeypatch.chdir(REPO)  # ./gpu-tenant.yaml
    assert isinstance(gpu.live_coordinator(), NullCoordinator)  # conftest: no suite file is read
    monkeypatch.setenv("ARGUS_ENV_FILE", str(suite))
    assert gpu.live_coordinator().name == "broker"
    assert PortraitSettings.from_env().curator_url == "http://file-curator.test:8101"
    monkeypatch.setenv("ARGUS_GPU_COORDINATOR", "none")
    monkeypatch.setenv("ARGUS_MASK_CURATOR_URL", "http://own-curator.test:8101")
    assert isinstance(gpu.live_coordinator(), NullCoordinator)
    assert PortraitSettings.from_env().curator_url == "http://own-curator.test:8101"


# ---- /health: cortex's ServiceHealth, with every field mask's own had ------------------------------


@pytest.fixture
def assets(tmp_path):
    make_face(tmp_path, "test-head")
    return tmp_path


def test_the_server_health_is_a_service_health_and_keeps_its_own_fields(assets, monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_PEER", "mask")
    with TestClient(create_app(Settings(assets_dir=assets, read_only=True))) as client:
        reply = client.get("/health")
        assert reply.status_code == 200
        body = reply.json()
        health = ServiceHealth.model_validate(body)
        assert (health.service, health.version, health.status) == ("argus-mask", __version__, "ok")
        assert health.wire == {"mask_version": MASK_VERSION} and health.read_only is True and health.peer == "mask"
        assert health.checks["driver"].ok and health.checks["driver"].required
        # Every field /health had before the contract, under the same name and shape.
        assert set(Health.model_fields) <= set(body)
        own = Health.model_validate(body)
        assert own.version == __version__ and own.read_only is True
        assert own.driver.model_dump() == {"name": "energy", "model": None, "available": True, "reason": None}
        assert (own.fps, own.faces, own.live, own.screens, own.aliases) == (30, 1, [], {}, {})
        assert "free_vram_mb" in body
        assert client.get("/health?checks=0").json()["checks"] == {}


def test_a_driver_that_is_down_is_degraded_and_still_200(assets) -> None:
    app = create_app(Settings(assets_dir=assets, driver="a2f", a2f_url="http://127.0.0.1:9"))
    with TestClient(app) as client:
        reply = client.get("/health")
        assert reply.status_code == 200
        health = ServiceHealth.model_validate(reply.json())
        assert health.status == "degraded" and health.checks["driver"].ok is False
        assert health.checks["driver"].detail is None  # why is in driver.reason, as before
        assert reply.json()["driver"]["available"] is False and "does not answer" in reply.json()["driver"]["reason"]


def test_the_editor_health_is_a_service_health_and_keeps_its_own_fields(tmp_path) -> None:
    from argus_mask.editor.app import create_editor_app

    editor = EditorSettings(blender=str(tmp_path / "no-blender"))
    with TestClient(create_editor_app(Settings(assets_dir=tmp_path, read_only=True), editor)) as client:
        reply = client.get("/health")
        assert reply.status_code == 200  # an editor that cannot build still answers
        body = reply.json()
        health = ServiceHealth.model_validate(body)
        assert (health.service, health.version, health.status) == ("argus-mask", __version__, "ok")
        assert health.capabilities == ("editor",) and health.wire == {"mask_version": MASK_VERSION}
        assert body["version"] == __version__ and body["read_only"] is True
        assert body["ready"] is False and "Blender not found" in body["reason"] and body["building"] is False


# ---- compose: the labels argus-iris's directory reads ----------------------------------------------


def test_compose_labels_the_server_and_only_the_server() -> None:
    yaml = pytest.importorskip("yaml")
    services = yaml.safe_load((REPO / "compose.yaml").read_text())["services"]
    assert services["mask"]["labels"] == {
        "argus.peer": "mask",
        "argus.service": "argus-mask",
        "argus.health": "/health",
        "argus.port": "8109",
    }
    assert services["mask"]["environment"]["ARGUS_PEER"] == "mask"
    assert "labels" not in services["a2f"]
