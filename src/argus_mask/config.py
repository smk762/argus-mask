"""Settings, read from ``ARGUS_MASK_*``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from argus_mask.drivers import DRIVERS

DEFAULT_PORT = 8109
DEFAULT_EDITOR_PORT = 8111  # 8110 is the Audio2Face server
DEFAULT_FPS = 30


class ConfigError(ValueError):
    """A setting that cannot be used."""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(f"ARGUS_MASK_{name}", default).strip()


@dataclass(frozen=True)
class Settings:
    assets_dir: Path = Path("assets")
    driver: str = "energy"
    a2f_url: str = ""
    tts_url: str = ""
    tts_voice: str = ""
    fps: int = DEFAULT_FPS
    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    read_only: bool = False
    # The Voices page's voices (EditorSettings has the same), read to refuse a consented voice whose release no
    # longer covers it, and to label what it says. A voice made elsewhere has no record here and is not checked.
    voices_dir: Path = Path("voices")

    def __post_init__(self) -> None:
        if self.driver not in DRIVERS:
            raise ConfigError(f"ARGUS_MASK_DRIVER {self.driver!r}: choose from {', '.join(DRIVERS)}")
        if not 1 <= self.fps <= 120:
            raise ConfigError(f"ARGUS_MASK_FPS {self.fps}: must be between 1 and 120")

    @classmethod
    def from_env(cls) -> Settings:
        try:
            from argus_cortex.server import env_flag  # noqa: PLC0415 - needs the [server] extra's starlette

            read_only = env_flag("ARGUS_MASK_READ_ONLY")
        except ImportError:
            read_only = _env("READ_ONLY").lower() in ("1", "true", "yes", "on")
        raw_fps = _env("FPS", str(DEFAULT_FPS))
        try:
            fps = int(raw_fps)
        except ValueError as exc:
            raise ConfigError(f"ARGUS_MASK_FPS {raw_fps!r} is not a number") from exc
        return cls(
            assets_dir=Path(_env("ASSETS_DIR", "assets")),
            driver=_env("DRIVER", "energy").lower() or "energy",
            a2f_url=_env("A2F_URL"),
            tts_url=_env("TTS_URL"),
            tts_voice=_env("TTS_VOICE"),
            fps=fps,
            cors_origins=tuple(o.strip().rstrip("/") for o in _env("CORS_ORIGINS").split(",") if o.strip()),
            read_only=read_only,
            voices_dir=Path(_env("VOICES_DIR") or "voices"),
        )


def _cache_size() -> int:
    raw = _env("BUILD_CACHE", "40")
    try:
        size = int(raw)
    except ValueError as exc:
        raise ConfigError(f"ARGUS_MASK_BUILD_CACHE {raw!r} is not a number") from exc
    if not 0 <= size <= 1000:
        raise ConfigError(f"ARGUS_MASK_BUILD_CACHE {size}: must be between 0 and 1000")
    return size


MPFB_DATA_UNDER_USER = Path("extensions/.user/user_default/mpfb/data")


@dataclass(frozen=True)
class EditorSettings:
    """The face editor's own settings. It also reads :class:`Settings` for the assets root and the guards."""

    blender: str = "blender"
    # Blender's user directory (BLENDER_USER_RESOURCES), where MPFB and its asset packs are installed.
    blender_user: Path | None = None
    # MPFB's data directory, if it is not the usual place under blender_user.
    mpfb_data: Path | None = None
    build_timeout_s: float = 300.0
    # Builds kept so that asking for one again (undo, redo) copies it instead of running Blender. 0: none.
    build_cache: int = 40
    # How the editor reaches the server (to check an exported face and send a test), and the origin a
    # screen opens the viewer at: the TLS proxy, on a LAN. The viewer's defaults to the server's.
    server_url: str = f"http://127.0.0.1:{DEFAULT_PORT}"
    viewer_url: str = ""
    # Where the Broadcast page's recordings go: an MP4 per broadcast, its WAV and take beside it.
    # Not under the assets root, where the server would take a directory for a face.
    recordings_dir: Path = Path("recordings")
    voices_dir: Path = Path("voices")  # the Voices page: checked clips, and each voice's record and source clips

    def __post_init__(self) -> None:
        for name in ("server_url", "viewer_url"):
            value = getattr(self, name)
            if value and not value.startswith(("http://", "https://")):
                raise ConfigError(f"ARGUS_MASK_{name.upper()} {value!r}: must be an http:// or https:// URL")

    @property
    def viewer(self) -> str:
        """The origin a screen opens the viewer at."""
        return (self.viewer_url or self.server_url).rstrip("/")

    @property
    def data_dir(self) -> Path | None:
        if self.mpfb_data is not None:
            return self.mpfb_data
        return self.blender_user / MPFB_DATA_UNDER_USER if self.blender_user is not None else None

    @classmethod
    def from_env(cls) -> EditorSettings:
        user = _env("BLENDER_USER") or os.environ.get("BLENDER_USER_RESOURCES", "").strip()
        data = _env("MPFB_DATA")
        raw_timeout = _env("BUILD_TIMEOUT_S", "300")
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise ConfigError(f"ARGUS_MASK_BUILD_TIMEOUT_S {raw_timeout!r} is not a number") from exc
        if not 10 <= timeout <= 3600:
            raise ConfigError(f"ARGUS_MASK_BUILD_TIMEOUT_S {timeout}: must be between 10 and 3600")
        return cls(
            blender=_env("BLENDER", "blender") or "blender",
            blender_user=Path(user) if user else None,
            mpfb_data=Path(data) if data else None,
            build_timeout_s=timeout,
            build_cache=_cache_size(),
            server_url=(_env("SERVER_URL") or f"http://127.0.0.1:{DEFAULT_PORT}").rstrip("/"),
            viewer_url=_env("VIEWER_URL").rstrip("/"),
            recordings_dir=Path(_env("RECORDINGS_DIR") or "recordings"),
            voices_dir=Path(_env("VOICES_DIR") or "voices"),
        )
