"""The wire contract: what a viewer, a take file and a face manifest carry.

A stream is one :class:`StreamHeader`, then :class:`Frame` messages, then one
:class:`StreamEnd`. A take file (``.ndjson``) is the same objects, one per line.
The version lives on the header and the manifest, not on every frame.

Audio reaches a viewer as binary messages beside the JSON ones: 8 bytes of
little-endian float64 (the chunk's start on the audio timeline, in ms), then
mono ``f32le`` PCM at ``StreamHeader.audio.sample_rate``.
"""

from __future__ import annotations

from typing import Literal

from argus_cortex.wire import make_versioned_base, render_schema
from pydantic import BaseModel, Field

# 1.0: initial contract.
MASK_VERSION = "1.0"
SUPPORTED_MASK_MAJORS: tuple[str, ...] = ("1",)
WIRE_TITLE = "argus-mask wire contract"

# A face id is a directory name under the assets root and a URL path segment.
FACE_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"

_Versioned = make_versioned_base(
    "mask_version", MASK_VERSION, SUPPORTED_MASK_MAJORS, label="mask_version", class_name="_Versioned"
)


class AudioFormat(BaseModel):
    """How PCM bytes are laid out."""

    sample_rate: int = Field(..., ge=8000, le=192000)
    channels: int = Field(1, ge=1, le=8)
    format: Literal["f32le", "s16le"] = "f32le"


class SourceHello(AudioFormat):
    """The first message a source sends on its socket: its audio format, and the idle layer's seed."""

    seed: int = Field(0, ge=0)


class DriverInfo(BaseModel):
    name: str
    model: str | None = None


class StreamHeader(_Versioned):
    """Opens a stream. ``blendshapes`` is the order of every frame's weights."""

    type: Literal["header"] = "header"
    face_id: str
    session_id: str
    fps: int = Field(..., ge=1, le=120)
    seed: int
    blendshapes: list[str]
    audio: AudioFormat
    driver: DriverInfo


class Head(BaseModel):
    """Head rotation, in degrees."""

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class Frame(BaseModel):
    """One animation frame. ``t_ms`` is its time on the audio timeline."""

    type: Literal["frame"] = "frame"
    seq: int = Field(..., ge=0)
    t_ms: int = Field(..., ge=0)
    weights: list[float]
    head: Head = Field(default_factory=Head)


class StreamEnd(BaseModel):
    """Closes a stream: the source went away and every frame has been sent."""

    type: Literal["end"] = "end"
    frames: int = Field(..., ge=0)
    duration_ms: int = Field(..., ge=0)


class Provenance(BaseModel):
    """Where a face came from. Required: a face without it is not served."""

    kind: Literal["stock", "synthetic", "curated"]
    source: str = Field(..., min_length=1)
    license: str = Field(..., min_length=1)
    notes: str = ""


class ModelFile(BaseModel):
    file: str = Field(..., pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*\.glb$")
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")


class Framing(BaseModel):
    """Where the viewer's camera sits for this head. Metres; the head's origin is the model's."""

    distance: float = Field(0.6, gt=0)
    height: float = 0.0
    fov: float = Field(30.0, gt=1, lt=120)


class FaceManifest(_Versioned):
    """``manifest.json`` in a face's directory."""

    id: str = Field(..., pattern=FACE_ID_PATTERN)
    name: str
    model: ModelFile
    blendshapes: list[str] = Field(default_factory=list)
    framing: Framing = Field(default_factory=Framing)
    provenance: Provenance


class DriverHealth(BaseModel):
    name: str
    model: str | None = None
    available: bool
    reason: str | None = None


class Health(BaseModel):
    version: str
    driver: DriverHealth
    fps: int
    free_vram_mb: int | None = None
    read_only: bool
    faces: int
    live: list[str] = Field(default_factory=list, description="Faces with a source connected")


def render_wire_schema() -> str:
    """The canonical committed-schema string (``schema`` writes it, ``schema --check`` compares)."""
    return render_schema([StreamHeader, Frame, StreamEnd, SourceHello, FaceManifest, Health], title=WIRE_TITLE)
