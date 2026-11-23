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
    # Host names (no port) a page may be served from and still write to its own server: a screen opening
    # http://morpheus:8109 names morpheus here. An IP address and localhost need no entry. A name is not trusted
    # on its own, because under DNS rebinding an attacker's page sends its own name as both Origin and Host.
    trusted_hosts: tuple[str, ...] = field(default_factory=tuple)
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
            trusted_hosts=tuple(h.strip().lower() for h in _env("TRUSTED_HOSTS").split(",") if h.strip()),
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


def _number(name: str, default: str, low: float, high: float) -> float:
    raw = _env(name, default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"ARGUS_MASK_{name} {raw!r} is not a number") from exc
    if not low <= value <= high:
        raise ConfigError(f"ARGUS_MASK_{name} {value:g}: must be between {low:g} and {high:g}")
    return value


# lam/ is beside src/ in a checkout, and is not part of the package: an installed editor names it itself.
LAM_SCRIPT_IN_CHECKOUT = Path(__file__).resolve().parents[2] / "lam" / "make_heads.sh"


@dataclass(frozen=True)
class LamSettings:
    """What a guided face's LAM build needs (``editor/lam_build.py``, ``lam/README.md``). Unset, no build runs."""

    # lam/make_heads.sh: the checked pipeline, run as a process and never imported.
    script: Path | None = None
    lam_dir: Path | None = None  # the aigc3d/LAM checkout, with its model_zoo
    flame_open: Path | None = None  # flame2023_Open.pkl
    image: str = "argus-mask-lam"
    blender: str = "blender"  # Blender 4.5, for LAM's FBX-to-GLB step
    docker: str = "docker"  # make_heads.sh calls docker by that name, so no setting changes it
    # Host memory that must be free before a build, and the floor under which a running one is stopped: a build
    # on a host short of memory has taken the whole machine down.
    min_free_gb: float = 12.0
    stop_free_gb: float = 2.0
    timeout_s: float = 1800.0  # about 2 minutes a head; the shape fit adds about 10
    cache: int = 20  # heads kept, by portrait and pipeline: the same portrait gives the same head. 0: none.

    @classmethod
    def from_env(cls) -> LamSettings:
        script, lam_dir, flame = _env("LAM_SCRIPT"), _env("LAM_DIR"), _env("FLAME_OPEN")
        if script:
            found: Path | None = Path(script).expanduser()
        else:
            found = LAM_SCRIPT_IN_CHECKOUT if LAM_SCRIPT_IN_CHECKOUT.is_file() else None
        return cls(
            script=found,
            lam_dir=Path(lam_dir).expanduser() if lam_dir else None,
            flame_open=Path(flame).expanduser() if flame else None,
            image=_env("LAM_IMAGE") or "argus-mask-lam",
            blender=_env("LAM_BLENDER") or _env("BLENDER", "blender") or "blender",
            min_free_gb=_number("LAM_MIN_FREE_GB", "12", 1, 1024),
            stop_free_gb=_number("LAM_STOP_FREE_GB", "2", 0, 1024),
            timeout_s=_number("LAM_TIMEOUT_S", "1800", 60, 86400),
            cache=int(_number("LAM_CACHE", "20", 0, 1000)),
        )


# DINOv2-base (Apache-2.0), what argus-proof scores likeness with: optional identity score for a guided head (#169).
DINOV2_IN_CACHE = Path.home() / ".cache" / "argus-mask" / "dinov2-base"
# lam/likeness.py lives beside make_heads.sh; the likeness job runs it in the LAM image (MediaPipe + torch).
LIKENESS_SCRIPT_IN_CHECKOUT = Path(__file__).resolve().parents[2] / "lam" / "likeness.py"


@dataclass(frozen=True)
class LikenessSettings:
    """What a guided face's likeness step needs (``editor/likeness.py``, ``lam/likeness.py``)."""

    script: Path | None = None  # lam/likeness.py
    image: str = "argus-mask-lam"  # MediaPipe and DINOv2 run in here
    docker: str = "docker"
    dinov2: Path | None = None  # a DINOv2 checkpoint folder (model.safetensors); unset: identity is skipped
    timeout_s: float = 300.0

    @classmethod
    def from_env(cls) -> LikenessSettings:
        script, dinov2 = _env("LIKENESS_SCRIPT"), _env("DINOV2")
        if script:
            found: Path | None = Path(script).expanduser()
        else:
            found = LIKENESS_SCRIPT_IN_CHECKOUT if LIKENESS_SCRIPT_IN_CHECKOUT.is_file() else None
        if dinov2:
            weights: Path | None = Path(dinov2).expanduser()
        else:
            weights = DINOV2_IN_CACHE if (DINOV2_IN_CACHE / "model.safetensors").is_file() else None
        return cls(
            script=found,
            image=_env("LIKENESS_IMAGE") or _env("LAM_IMAGE") or "argus-mask-lam",
            dinov2=weights,
            timeout_s=_number("LIKENESS_TIMEOUT_S", "300", 30, 3600),
        )


REST_SCRIPT_IN_CHECKOUT = Path(__file__).resolve().parents[2] / "lam" / "fit_rest.py"
LANDMARKER_IN_CACHE = Path.home() / ".cache" / "argus-mask" / "face_landmarker.task"


@dataclass(frozen=True)
class RestSettings:
    """What a guided face's rest fit needs (``editor/rest_fit.py``, ``lam/fit_rest.py``)."""

    script: Path | None = None  # lam/fit_rest.py
    landmarker: Path | None = None  # MediaPipe face_landmarker.task
    timeout_s: float = 600.0  # about 3½ minutes a head on CPU
    generations: int = 20
    # uv installs pillow and mediapipe for the script when the editor's env has neither (#169).
    uv: str = "uv"

    @classmethod
    def from_env(cls) -> RestSettings:
        script, model = _env("REST_SCRIPT"), _env("LANDMARKER")
        if script:
            found: Path | None = Path(script).expanduser()
        else:
            found = REST_SCRIPT_IN_CHECKOUT if REST_SCRIPT_IN_CHECKOUT.is_file() else None
        if model:
            landmarker: Path | None = Path(model).expanduser()
        else:
            landmarker = LANDMARKER_IN_CACHE if LANDMARKER_IN_CACHE.is_file() else None
        return cls(
            script=found,
            landmarker=landmarker,
            timeout_s=_number("REST_TIMEOUT_S", "600", 60, 7200),
            generations=int(_number("REST_GENERATIONS", "20", 1, 200)),
            uv=_env("UV") or "uv",
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
