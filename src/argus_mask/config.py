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
    fps: int = DEFAULT_FPS
    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    read_only: bool = False

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
            fps=fps,
            cors_origins=tuple(o.strip().rstrip("/") for o in _env("CORS_ORIGINS").split(",") if o.strip()),
            read_only=read_only,
        )


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
        )
