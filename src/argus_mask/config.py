"""Settings, read from ``ARGUS_MASK_*``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from argus_mask.drivers import DRIVERS

DEFAULT_PORT = 8109
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
