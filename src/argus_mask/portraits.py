"""Source portraits for a guided face (#167): style templates, darkroom record checks, and provenance kind.

``lam/portraits.py`` is the CLI on top of this. The editor's source step uses the same templates and checks so a
batch from the CLI and one from the editor agree on what is ``synthetic``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from argus_mask.backgrounds import KNOWN_LICENSES, POLICY, PlateError

# FLUX.1 schnell: the one checkpoint known to clear a synthetic face (docs/synthetic-faces.md).
CHECKPOINT = "flux/flux1-schnell-fp8-e4m3fn.safetensors"
SAMPLING = {"sampler": "euler", "scheduler": "simple", "steps": 4, "cfg": 1.0, "guidance": 3.5}
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1280

# What LAM needs of a portrait: square on, neutral, flat light (a smile or a shadow is baked into the head).
TEMPLATE = (
    "close-up studio portrait photograph of {subject}, head and shoulders, facing the camera squarely, head level, "
    "calm neutral expression, mouth closed and relaxed, not smiling, eyes open looking into the lens, "
    "very soft flat lighting from a large softbox directly in front, face evenly lit, no shadows on the face, "
    "plain dark grey background, sharp focus, natural skin texture, 85mm lens, f/8"
)
# An idealised, painted look: LAM softens what it is given, and a smooth stylised face survives that where a
# photograph's skin and shading come out heavier and older (#151). Still a text prompt only, so still synthetic.
STYLISED = (
    "digital painting portrait of {subject}, idealized smooth flawless skin, clear bright eyes, refined features, "
    "head and shoulders, facing the camera squarely, head level, calm neutral expression, lips closed, "
    "eyes looking into the lens, soft even front lighting, plain dark grey background, highly detailed, sharp focus"
)
# Between the two: realistic proportions and light, rendered-smooth skin; the look LAM keeps (#151).
SEMIREAL = (
    "semi-realistic 3D render portrait of {subject}, realistic proportions, smooth even skin, natural colouring, "
    "clear eyes, head and shoulders, facing the camera squarely, head level, calm neutral expression, lips closed, "
    "eyes looking into the lens, soft even front lighting, plain grey background, clean image, sharp focus"
)
# Realistic character art: a photograph's proportions and light, an idealised painted finish (#151).
CHARACTER = (
    "highly realistic digital art portrait of {subject}, realistic adult proportions, flawless smooth skin, "
    "subtle natural makeup, clear detailed eyes, head and shoulders, facing the camera squarely, head level, "
    "calm neutral expression, lips closed, eyes looking into the lens, soft even front lighting, plain grey background, "
    "cinematic character art, sharp focus"
)
TEMPLATES: dict[str, str] = {
    "photo": TEMPLATE,
    "stylised": STYLISED,
    "semireal": SEMIREAL,
    "character": CHARACTER,
}
StyleName = Literal["photo", "stylised", "semireal", "character"]
DEFAULT_STYLE: StyleName = "character"

# A portrait upload is at most this big; open_image still caps pixels (cortex limits).
MAX_PORTRAIT_BYTES = 8 * 1024 * 1024
MAX_PORTRAIT_PIXELS = 40_000_000
PORTRAIT_FORMATS = ("JPEG", "PNG", "WEBP")

# Candidate id for a seed: ``s501``. Matches FACE_ID_PATTERN.
_SEED_CANDIDATE = re.compile(r"^s(\d+)$")


class PortraitError(Exception):
    """A source portrait could not be made or accepted, with the reason."""


def seeds(text: str) -> list[int]:
    """``501-508`` or ``501,503,507``."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        first, _, last = part.partition("-")
        try:
            start = int(first)
            stop = int(last or first)
        except ValueError as exc:
            raise PortraitError(f"{text!r} is not a seed list: use 501-508 or 501,503") from exc
        if stop < start:
            raise PortraitError(f"{text!r}: the range ends before it starts")
        if stop - start > 64:
            raise PortraitError(f"{text!r}: at most 64 seeds in one batch")
        out.extend(range(start, stop + 1))
    if not out:
        raise PortraitError("give at least one seed")
    if len(out) != len(set(out)):
        raise PortraitError(f"{text!r}: each seed once")
    return out


def candidate_for_seed(seed: int) -> str:
    return f"s{seed}"


def seed_of_candidate(candidate: str) -> int | None:
    found = _SEED_CANDIDATE.fullmatch(candidate)
    return int(found.group(1)) if found else None


def prompt_for(subject: str, style: str = DEFAULT_STYLE) -> str:
    """The full prompt around *subject* for *style*; refuses an unknown style."""
    template = TEMPLATES.get(style)
    if template is None:
        raise PortraitError(f"{style!r} is not a portrait style: {', '.join(sorted(TEMPLATES))}")
    text = subject.strip()
    if not text:
        raise PortraitError("the subject says who: age (18 or over), hair, clothes; no names")
    return template.format(subject=text)


def subject_line(
    *,
    age: int,
    hair: str = "",
    clothes: str = "",
    eyes: str = "",
    description: str = "",
) -> str:
    """Compose the subject clause from the structured fields the editor takes."""
    if age < 18:
        raise PortraitError(f"age {age}: a source portrait is 18 or over")
    if age > 120:
        raise PortraitError(f"age {age}: not a plausible adult age")
    parts = [f"an adult in their {age}s" if age % 10 == 0 and age < 100 else f"an adult aged {age}"]
    if hair.strip():
        parts.append(f"{hair.strip()} hair" if "hair" not in hair.lower() else hair.strip())
    if eyes.strip():
        parts.append(f"{eyes.strip()} eyes" if "eye" not in eyes.lower() else eyes.strip())
    if clothes.strip():
        wearing = clothes.strip()
        parts.append(wearing if wearing.lower().startswith(("wearing", "in a", "in an")) else f"wearing {wearing}")
    if description.strip():
        parts.append(description.strip())
    return ", ".join(parts)


def license_for_checkpoint(checkpoint: str, stated: str = "") -> str:
    """The licence a synthetic portrait from *checkpoint* is under."""
    if stated.strip():
        return stated.strip()
    name = checkpoint.rsplit("/", 1)[-1].lower()
    for fragment, licence in KNOWN_LICENSES.items():
        if fragment in name:
            return licence
    known = ", ".join(f"{k} ({v})" for k, v in KNOWN_LICENSES.items())
    raise PortraitError(
        f"{checkpoint}: its output licence is not known here; state it after checking the model's terms, "
        f"or use a known one: {known}"
    )


def generate_body(
    prompt: str,
    seed: int,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    policy: str = POLICY,
    checkpoint: str = CHECKPOINT,
    init_image: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The JSON body darkroom's ``/generate/stream`` takes for one portrait (or a restyle with *init_image*)."""
    body: dict[str, Any] = {
        "user_prompt": prompt,
        "negative_prompt": "",
        "expand": False,
        "seed": seed,
        "enforcing_policy": policy,
        "checkpoint": checkpoint,
        "sampling": {**SAMPLING, "width": width, "height": height},
        "loras": [],
        "advisory_policies": [],
    }
    if init_image is not None:
        body["init_image"] = init_image
    return body


def image_sha256(record: dict) -> str | None:
    """The sha256 darkroom recorded for the image, if any."""
    image = record.get("image") or {}
    sha = image.get("sha256") if isinstance(image, dict) else None
    return str(sha) if isinstance(sha, str) and len(sha) == 64 else None


def names_loras(record: dict) -> bool:
    asked = record.get("request") or {}
    loras = asked.get("loras") if isinstance(asked, dict) else None
    return bool(loras)


def refusal(record: dict) -> str | None:
    """Why this record's image is not a cleared *synthetic* source, or None.

    A record that names a LoRA is still usable as ``personal`` (see :func:`kind_from_record`); this refusal is
    for the generate path, which only keeps ``synthetic``.
    """
    if record.get("outcome") != "passed":
        return f"darkroom: {record.get('outcome', 'no record')} ({record.get('outcome_reason') or 'no reason given'})"
    if names_loras(record):
        return "the record names a LoRA: the face would be personal, not synthetic"
    return None


def check_source_record(
    record: dict,
    image: bytes,
    *,
    policy: str = POLICY,
    expect_prompt: str | None = None,
    expect_seed: int | None = None,
    expect_checkpoint: str | None = None,
) -> str:
    """Check a darkroom record for use as a source portrait; return the provenance kind.

    Always: outcome ``passed``, the enforcing *policy*, the image's sha256. When *expect_** are set (a generate
    or restyle we asked for), the prompt, seed and checkpoint must match too — the same bar as a plate's
    :func:`~argus_mask.backgrounds._check_record`, without requiring a negative-prompt dance for FLUX.
    A record that names a LoRA is ``personal``, never ``synthetic``.
    """
    rid = str(record.get("id") or "")
    outcome = record.get("outcome")
    if outcome != "passed":
        reason = record.get("outcome_reason") or ""
        raise PortraitError(f"darkroom record {rid}: {outcome}{': ' + reason if reason else ''}")
    guard = record.get("guard") or {}
    found_policy = guard.get("enforcing_policy") if isinstance(guard, dict) else None
    if found_policy != policy:
        raise PortraitError(f"darkroom record {rid}: enforced {found_policy!r}, not {policy!r}")
    sha = image_sha256(record)
    if sha is None:
        raise PortraitError(f"darkroom record {rid}: no image sha256")
    if hashlib.sha256(image).hexdigest() != sha:
        raise PortraitError(f"darkroom record {rid}: the image does not match the record's sha256")
    asked = record.get("request") or {}
    if not isinstance(asked, dict):
        raise PortraitError(f"darkroom record {rid}: no request")
    if expect_prompt is not None:
        if asked.get("expanded_prompt"):
            raise PortraitError(f"darkroom record {rid}: the prompt was expanded")
        if asked.get("final_prompt") != expect_prompt:
            raise PortraitError(f"darkroom record {rid}: darkroom rendered a different prompt")
    if expect_seed is not None and asked.get("seed") != expect_seed:
        raise PortraitError(f"darkroom record {rid}: seed {asked.get('seed')}, not {expect_seed}")
    if expect_checkpoint is not None:
        checkpoint = (asked.get("checkpoint") or {}).get("name") if isinstance(asked.get("checkpoint"), dict) else None
        if checkpoint is None and isinstance(asked.get("checkpoint"), str):
            checkpoint = asked.get("checkpoint")
        if checkpoint != expect_checkpoint:
            raise PortraitError(f"darkroom record {rid}: checkpoint {checkpoint!r}, not {expect_checkpoint!r}")
    return "personal" if names_loras(record) else "synthetic"


def kind_from_record(record: dict, image: bytes, *, policy: str = POLICY) -> str:
    """``synthetic`` or ``personal`` for a picked darkroom record (no expect_* — we did not ask for it)."""
    return check_source_record(record, image, policy=policy)


def as_plate_error(exc: PlateError) -> PortraitError:
    return PortraitError(str(exc))
