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
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

# 1.0: initial contract.
# 1.1: provenance kinds personal and consented, consent fields, and the derived ``cleared``.
# 1.2: ``ModelFile.format``: a face is a glTF binary (``.glb``) or a LAM Gaussian-splat head (``.zip``).
# 1.3: ``Provenance.generator`` and the derived ``disclosure``; ``SourceHello.disclose`` and
#      ``StreamHeader.disclosure`` (a real, consenting person's voice); ``ConsentVerdict``.
# 1.4: ``PlateManifest``, a backdrop made from a still image (``background.json``).
# 1.5: ``FaceManifest.framings``: the same face's other framings (waist up, full body), each its own model and camera;
#      ``Frame.cue``: a stress or a phrase edge that began on the frame, for gestures.
# 1.6: ``ModelFile.splats``: Gaussian splats bound to a mesh model's triangles (#126 route A), with the licence of
#      every input that went into them.
MASK_VERSION = "1.6"
SUPPORTED_MASK_MAJORS: tuple[str, ...] = ("1",)
WIRE_TITLE = "argus-mask wire contract"

# A face id is a directory name under the assets root and a URL path segment.
FACE_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"
# Directories in the assets root that are not faces: plates of one's own live in backgrounds/.
RESERVED_FACE_IDS: frozenset[str] = frozenset({"backgrounds"})

_Versioned = make_versioned_base(
    "mask_version", MASK_VERSION, SUPPORTED_MASK_MAJORS, label="mask_version", class_name="_Versioned"
)


class AudioFormat(BaseModel):
    """How PCM bytes are laid out."""

    sample_rate: int = Field(..., ge=8000, le=192000)
    channels: int = Field(1, ge=1, le=8)
    format: Literal["f32le", "s16le"] = "f32le"


# The label every output of a consented likeness carries. Fixed text: no record or request can change or drop it.
FACE_DISCLOSURE = "An AI avatar of a real person, shown with their consent."
VOICE_DISCLOSURE = "An AI voice of a real person, used with their consent."


class SourceHello(AudioFormat):
    """The first message a source sends on its socket: its audio format, the idle layer's seed, and whether the
    voice is a real, consenting person's (so the stream carries the disclosure label)."""

    seed: int = Field(0, ge=0)
    disclose: bool = Field(False, description="the voice is a real person's, used with their consent")


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
    disclosure: str | None = Field(
        None,
        description="the label a viewer shows while this stream plays: set when the voice is a real person's",
        exclude_if=lambda v: v is None,  # left out when unset, so a take without one reads as it did before 1.3
    )


class Head(BaseModel):
    """Head rotation, in degrees."""

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class SpeechCue(BaseModel):
    """What began in the voice on a frame (prosody.py): a stressed syllable, how strong (0..1), and a phrase's start
    or end. A screen that shows the body gestures by them; the face has already taken its nods and brows."""

    model_config = ConfigDict(extra="forbid")

    stress: float = Field(0.0, ge=0.0, le=1.0)
    phrase: Literal["start", "end"] | None = None


class Frame(BaseModel):
    """One animation frame. ``t_ms`` is its time on the audio timeline. ``cue`` is there only on a frame where a
    stress or a phrase edge began."""

    type: Literal["frame"] = "frame"
    seq: int = Field(..., ge=0)
    t_ms: int = Field(..., ge=0)
    weights: list[float]
    head: Head = Field(default_factory=Head)
    cue: SpeechCue | None = None


class StreamEnd(BaseModel):
    """Closes a stream: the source went away and every frame has been sent."""

    type: Literal["end"] = "end"
    frames: int = Field(..., ge=0)
    duration_ms: int = Field(..., ge=0)


ProvenanceKind = Literal["personal", "stock", "synthetic", "consented", "curated"]
# The kinds the project may ship, show or demo. ``personal`` is served on the user's own screens and is
# never cleared: a cartoon, a character from a film, a character LoRA. Legal for private use is the floor;
# what argus-darkroom refuses (its illegal rules) is not a face here either.
CLEARED_KINDS: frozenset[str] = frozenset({"stock", "synthetic", "consented"})
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class Generator(BaseModel):
    """How a generated face was made: the model (and its sha256), the seed and the prompt."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., min_length=1, max_length=300)
    model_sha256: str | None = Field(None, pattern=SHA256_PATTERN)
    seed: int | None = Field(None, ge=0)
    prompt: str = Field("", max_length=4000)


class Provenance(BaseModel):
    """Where a face came from. Required: a face without it is not served.

    - ``personal``: the user's own use, anything they may lawfully use privately. Needs only ``source``; not
      cleared to show or share.
    - ``stock``: a published asset under its licence. ``synthetic``: generated, with no real person or
      owned character going in.
    - ``consented``: a real person's likeness (or an owned character) under a signed release, bound by the
      release's sha256 (never the document) and the subject as the release names them.
    - ``curated``: refused until 0.3 defines what a curated face must prove.

    A ``synthetic`` (or ``personal``) face may say how it was generated (``generator``). A ``consented`` face is
    served only with a valid consent record for its release (:mod:`argus_mask.consent`), and carries the
    disclosure label.

    Unknown fields are refused, so a misspelt field cannot be dropped in silence; ``licence`` is accepted for
    ``license``, and the derived ``cleared`` and ``disclosure`` a dumped record carries are recomputed, never read.
    """

    model_config = ConfigDict(extra="forbid")

    kind: ProvenanceKind
    source: str = Field(..., min_length=1)
    license: str = ""
    notes: str = ""
    subject: str = ""
    release_sha256: str | None = Field(None, pattern=SHA256_PATTERN)
    generator: Generator | None = None

    @model_validator(mode="before")
    @classmethod
    def _read(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        data = {k: v for k, v in data.items() if k not in ("cleared", "disclosure")}
        if "licence" in data:
            licence = data.pop("licence")
            if "license" in data and data["license"] != licence:
                raise ValueError("give the licence once: `license` and `licence` disagree")
            data["license"] = licence
        return data

    @model_validator(mode="after")
    def _complete(self) -> Provenance:
        if not self.source.strip():
            raise ValueError("a face states its source: where it came from")
        if self.kind != "personal" and not self.license.strip():
            raise ValueError(f"a {self.kind} face states its licence")
        if self.kind == "consented" and not (self.release_sha256 and self.subject.strip()):
            raise ValueError("a consented face names its subject and the sha256 of the signed release")
        if self.generator is not None and self.kind not in ("synthetic", "personal"):
            raise ValueError(
                f"a {self.kind} face was not generated: only synthetic and personal faces name a generator"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cleared(self) -> bool:
        """Whether the face may be shown beyond the user's own screens: in a demo, a release, anything public."""
        return self.kind in CLEARED_KINDS

    @computed_field  # type: ignore[prop-decorator]
    @property
    def disclosure(self) -> str | None:
        """The label shown with the face: set for a real person's likeness, which is never shown unlabelled."""
        return FACE_DISCLOSURE if self.kind == "consented" else None


ModelFormat = Literal["gltf", "lam-splat"]
# The file each format comes in: a manifest whose file and format disagree is refused.
MODEL_SUFFIXES: dict[str, str] = {"gltf": ".glb", "lam-splat": ".zip"}


# Bound splats: a ceiling well above a head's (stock-01's has 149,007), so a file cannot make a screen allocate
# without bound.
MAX_BOUND_SPLATS = 1_000_000
_FILE_NAME = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}"


class Credit(BaseModel):
    """One input that went into a model, and its licence: a mesh, an environment map, a tool whose output it is."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=300)
    license: str = Field(..., min_length=1, max_length=200)
    url: str = Field("", max_length=500)


class SplatPart(BaseModel):
    """One mesh of the model, in the order the bake concatenated their vertices: what the binding indexes."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=200)
    vertices: int = Field(..., ge=1)
    triangles: int = Field(..., ge=1)


class BoundSplats(BaseModel):
    """Gaussian splats bound to the triangles of the mesh model they sit beside (#126 route A, ``bake/``).

    ``file`` is the splats at the mesh's rest pose (SPZ, version 2 or 3). ``binding`` gives each splat, in the
    SPZ's order, the triangle it rides and where on it: little-endian uint32 triangle, float32 u, float32 v (12
    bytes a splat). Triangles are counted through ``parts``, the mesh nodes in file order, so the binding holds
    for that one build of the mesh: ``mesh_sha256`` must be the model's own sha256. ``license`` is the splats',
    ``inputs`` every input that went into them with its licence.
    """

    model_config = ConfigDict(extra="forbid")

    file: str = Field(..., pattern=_FILE_NAME + r"\.spz$")
    sha256: str = Field(..., pattern=SHA256_PATTERN)
    binding: str = Field(..., pattern=_FILE_NAME + r"\.bin$")
    binding_sha256: str = Field(..., pattern=SHA256_PATTERN)
    count: int = Field(..., ge=1, le=MAX_BOUND_SPLATS)
    mesh_sha256: str = Field(..., pattern=SHA256_PATTERN)
    parts: list[SplatPart] = Field(..., min_length=1, max_length=64)
    license: str = Field(..., min_length=1, max_length=200)
    inputs: list[Credit] = Field(..., min_length=1, max_length=32)

    @property
    def triangles(self) -> int:
        return sum(p.triangles for p in self.parts)


class ModelFile(BaseModel):
    """A face's model, in its directory: its file name, sha256 and format.

    - ``gltf``: a glTF 2.0 binary (``.glb``), morph targets named for the ARKit shapes. The default, so a
      manifest from before 1.2 reads as it did.
    - ``lam-splat``: a LAM Gaussian-splat head (``.zip``): one folder holding ``skin.glb`` (the FLAME mesh and
      its ARKit shapes), ``offset.ply`` (the splats), ``animation.glb``, ``vertex_order.json`` and optionally
      ``iris_occlusion.json``.

    A ``gltf`` model may carry ``splats`` (:class:`BoundSplats`): files beside it that a screen able to draw
    them shows riding the mesh; one that is not shows the mesh.
    """

    file: str = Field(..., pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*\.(glb|zip)$")
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    format: ModelFormat = Field("gltf", description="gltf: a .glb; lam-splat: a LAM Gaussian-splat head, a .zip")
    splats: BoundSplats | None = None

    @model_validator(mode="after")
    def _agrees(self) -> ModelFile:
        suffix = MODEL_SUFFIXES[self.format]
        if not self.file.endswith(suffix):
            raise ValueError(f"a {self.format} model is a {suffix} file, not {self.file}")
        if self.splats is not None:
            if self.format != "gltf":
                raise ValueError("only a mesh (gltf) model carries bound splats")
            if self.splats.mesh_sha256 != self.sha256:
                raise ValueError("the splats were bound to another build of the mesh: bake them again")
        return self

    @property
    def files(self) -> list[str]:
        """Every file this model is: the model's own, then its splats' and their binding."""
        return [self.file, *((self.splats.file, self.splats.binding) if self.splats else ())]


class Framing(BaseModel):
    """Where the viewer's camera sits for this head. Metres; the head's origin is the model's."""

    distance: float = Field(0.6, gt=0)
    height: float = 0.0
    fov: float = Field(30.0, gt=1, lt=120)


# How much of the body a model shows (#62): the head (and shoulders: ``model``), the waist up, or all of it.
FramingName = Literal["head", "waist", "full"]
FRAMINGS: tuple[str, ...] = ("head", "waist", "full")
BodyFraming = Literal["waist", "full"]


class FramedModel(BaseModel):
    """Another framing of the same face: its own model (a rigged ``.glb`` from the same preset) and camera."""

    model_config = ConfigDict(extra="forbid")

    model: ModelFile
    framing: Framing


class FaceManifest(_Versioned):
    """``manifest.json`` in a face's directory.

    ``model`` and ``framing`` are the face's head, which every screen can show. ``framings`` holds the same face at
    other framings (``waist``, ``full``), each a model of its own beside it; a screen asks for one with ``?framing=``.
    """

    id: str = Field(..., pattern=FACE_ID_PATTERN)
    name: str
    model: ModelFile
    blendshapes: list[str] = Field(default_factory=list)
    framing: Framing = Field(default_factory=Framing)
    framings: dict[BodyFraming, FramedModel] = Field(default_factory=dict)
    provenance: Provenance

    @model_validator(mode="after")
    def _framings(self) -> FaceManifest:
        files = [name for m in (self.model, *(f.model for f in self.framings.values())) for name in m.files]
        if len(set(files)) != len(files):
            raise ValueError("each framing is its own model file, and splats and binding files of their own")
        if self.framings and (
            self.model.format != "gltf" or any(f.model.format != "gltf" for f in self.framings.values())
        ):
            raise ValueError("framings are built meshes (gltf): a splat head is a head only")
        return self

    def framed(self, name: str) -> tuple[ModelFile, Framing] | None:
        """The model and camera for framing *name*: the head's for ``head``, None if the face has no such framing."""
        if name == "head":
            return self.model, self.framing
        found = self.framings.get(name)  # type: ignore[call-overload]
        return (found.model, found.framing) if found else None


class DriverHealth(BaseModel):
    name: str
    model: str | None = None
    available: bool
    reason: str | None = None


class ViewerStatus(BaseModel):
    """What a viewer says about itself on its socket, when it connects and whenever it changes.

    State, not media: whether it can play sound yet (a browser needs one click or key press first),
    and how it shows the face. The server keeps the latest per viewer and reports it on ``/health``.
    """

    model_config = ConfigDict(extra="ignore")  # a newer viewer may say more

    type: Literal["status"] = "status"
    audio: bool = Field(False, description="Sound is unlocked: the screen will be heard")
    display: Literal["flat", "hld", "go"] = "flat"
    kiosk: bool = False
    sound: Literal["local", "external"] = Field(
        "local",
        description="local: the screen plays the session's sound; external: it plays none (house speakers, "
        "Snapcast, play it) and only moves, on the session's timeline (viewer ?audio=external)",
    )


class Screen(BaseModel):
    """A viewer that has said what it is: one entry of ``Health.screens``."""

    display: str
    audio_ready: bool = Field(
        ...,
        description="Ready to speak in sync: with sound local, its sound is unlocked (it will be heard); with sound "
        "external, it is watching and moves on the session's timeline, no sound of its own needed",
    )
    kiosk: bool = False
    sound: Literal["local", "external"] = "local"


class Health(BaseModel):
    version: str
    driver: DriverHealth
    fps: int
    free_vram_mb: int | None = None
    read_only: bool
    faces: int
    live: list[str] = Field(default_factory=list, description="Faces with a source connected")
    screens: dict[str, list[Screen]] = Field(
        default_factory=dict,
        description="Per face (or alias, by the name the viewers used), the viewers that have reported their state",
    )
    aliases: dict[str, str] = Field(
        default_factory=dict, description="Stable names for faces: alias -> the face id it stands for now"
    )


class RestSet(BaseModel):
    """A named baseline expression: what a face holds at rest and speaks from (``GET /face/{id}/rest``).

    The viewer adds ``weights`` to every frame (the mouth's at half while speaking, so the mouth stays the
    driver's) and, at rest, moves the face by ``life``: 0 still, 1 the default, up to 2.
    """

    name: str = Field(..., pattern=FACE_ID_PATTERN)
    weights: dict[str, float] = Field(default_factory=dict, description="ARKit blendshape name -> weight, 0 to 1")
    life: float = Field(1.0, ge=0.0, le=2.0)


# A backdrop's name: a URL parameter (``?scene=``) and a directory name.
SCENE_NAME_PATTERN = r"^[a-z][a-z0-9-]{0,31}$"
# What a plate's files may be called: a plain PNG in the plate's own directory, never a path.
PLATE_FILE_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}\.png$"


class PlateImage(BaseModel):
    """One image of a plate, in its directory: the file, its sha256, and its size in pixels."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(..., pattern=PLATE_FILE_PATTERN)
    sha256: str = Field(..., pattern=SHA256_PATTERN)
    width: int = Field(..., gt=0, le=8192)
    height: int = Field(..., gt=0, le=8192)


class PlateEffects(BaseModel):
    """How much each built-in effect moves the still, 0 (off) to 1. All are functions of the scene's time.

    - ``drift``: the camera drifts slowly across the plate; with a depth map, near things move more (parallax).
    - ``sway``: leaves and fronds move in a breeze, where ``sway_mask`` is light, or by depth without one.
    - ``dapple``: light through leaves drifts over the scene.
    - ``pulse``: lights breathe, where ``light_mask`` is light.
    - ``soft``: how far out of focus the plate is, so it sits behind the face.
    - ``exposure``: the plate's brightness, so it sits behind the face (below 1 darkens).
    """

    model_config = ConfigDict(extra="forbid")

    drift: float = Field(0.5, ge=0.0, le=1.0)
    sway: float = Field(0.0, ge=0.0, le=1.0)
    dapple: float = Field(0.0, ge=0.0, le=1.0)
    pulse: float = Field(0.0, ge=0.0, le=1.0)
    soft: float = Field(0.4, ge=0.0, le=1.0)
    exposure: float = Field(0.8, ge=0.1, le=2.0)


class DarkroomDepth(BaseModel):
    """The depth map darkroom derived from the plate's record (``POST /depth/{record_id}``)."""

    model: str = Field(..., min_length=1)
    model_license: str = Field(..., min_length=1)
    sha256: str = Field(..., pattern=SHA256_PATTERN)
    workflow: str | None = None


class DarkroomRender(BaseModel):
    """The argus-darkroom generation a plate came from: its record, and what was asked for.

    Darkroom keeps the full record (the guard's verdicts included); this is what a reader of the plate needs to
    find it again and to judge it. ``plate.sha256`` is the record's image sha256: the plate is darkroom's bytes.
    """

    record_id: str = Field(..., min_length=1, max_length=200)
    darkroom_version: str = ""
    outcome: Literal["passed"] = Field("passed", description="only a released image becomes a plate")
    enforcing_policy: str = Field(..., min_length=1)
    checkpoint: str = Field(..., min_length=1)
    seed: int = Field(..., ge=0)
    prompt: str = Field(..., min_length=1, max_length=4000)
    negative_prompt: str = Field("", max_length=4000)
    workflow: str | None = None
    sampler: str | None = None
    scheduler: str | None = None
    steps: int | None = None
    cfg: float | None = None
    guidance: float | None = None
    depth: DarkroomDepth | None = None


class PlateManifest(_Versioned):
    """``background.json`` in a plate's directory: a still image, animated by the viewer's built-in effects.

    The plate is the picture; ``depth`` (near is light) gives parallax and places the sway; ``sway_mask`` and
    ``light_mask`` (light is on) say where leaves move and lights pulse. Every image is the plate's size.
    Provenance is required, as for a face: a plate without it is not shown, and only a cleared one may ship.
    """

    name: str = Field(..., pattern=SCENE_NAME_PATTERN)
    title: str = Field(..., min_length=1, max_length=80)
    kind: Literal["plate"] = "plate"
    tint: str = Field("#ffffff", pattern=r"^#[0-9a-fA-F]{6}$", description="the scene's light, for the rim lights")
    plate: PlateImage
    depth: PlateImage | None = None
    sway_mask: PlateImage | None = None
    light_mask: PlateImage | None = None
    effects: PlateEffects = Field(default_factory=PlateEffects)
    provenance: Provenance
    darkroom: DarkroomRender | None = None

    @model_validator(mode="after")
    def _agrees(self) -> PlateManifest:
        images = {"depth": self.depth, "sway_mask": self.sway_mask, "light_mask": self.light_mask}
        for role, image in images.items():
            if image is not None and (image.width, image.height) != (self.plate.width, self.plate.height):
                raise ValueError(
                    f"{role} is {image.width}x{image.height}, the plate {self.plate.width}x{self.plate.height}"
                )
        files = [i.file for i in (self.plate, *images.values()) if i is not None]
        if len(set(files)) != len(files):
            raise ValueError("each image of a plate is its own file")
        if self.provenance.kind == "curated":
            raise ValueError("a curated plate is refused until curated is defined")
        if self.provenance.kind == "consented":
            raise ValueError("a plate shows no one: consented is for a person's likeness")
        if self.darkroom is not None:
            generator = self.provenance.generator
            if generator is None or generator.model != self.darkroom.checkpoint or generator.seed != self.darkroom.seed:
                raise ValueError("a plate from darkroom names darkroom's checkpoint and seed as its generator")
            if self.darkroom.depth is not None and (
                self.depth is None or self.depth.sha256 != self.darkroom.depth.sha256
            ):
                raise ValueError("the depth map is not the one darkroom made")
        return self


class SceneInfo(BaseModel):
    """A backdrop a screen can show (``GET /scenes``): a built-in scene, or a plate in the assets."""

    name: str = Field(..., pattern=SCENE_NAME_PATTERN)
    title: str
    kind: Literal["procedural", "plate"]
    built_in: bool
    cleared: bool = Field(..., description="whether it may be shown beyond the user's own screens")


class FaceScene(BaseModel):
    """The backdrop a face shows by default (``GET /face/{id}/scene``): a scene name, or null for a plain one."""

    scene: str | None = Field(None, pattern=SCENE_NAME_PATTERN)


def render_wire_schema() -> str:
    """The canonical committed-schema string (``schema`` writes it, ``schema --check`` compares)."""
    from argus_mask.consent import ConsentVerdict  # noqa: PLC0415 - consent imports this module

    return render_schema(
        [
            StreamHeader,
            Frame,
            StreamEnd,
            SourceHello,
            FaceManifest,
            Health,
            ViewerStatus,
            RestSet,
            ConsentVerdict,
            PlateManifest,
            SceneInfo,
            FaceScene,
        ],
        title=WIRE_TITLE,
    )
