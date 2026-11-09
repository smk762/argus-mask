"""Take files: a performance on disk, one JSON object per line.

The first line is the :class:`~argus_mask.models.StreamHeader`, then one
:class:`~argus_mask.models.Frame` per line, then the
:class:`~argus_mask.models.StreamEnd`. A take and its audio replay in a viewer
with no driver running.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from argus_mask.models import Frame, StreamEnd, StreamHeader


class TakeError(ValueError):
    """A take file that cannot be read."""


def line(message: StreamHeader | Frame | StreamEnd) -> str:
    """One message as it goes on the wire and into a take: compact JSON, keys in model order."""
    return json.dumps(message.model_dump(mode="json"), separators=(",", ":"))


def render_take(header: StreamHeader, frames: Iterable[Frame], end: StreamEnd) -> str:
    return "\n".join([line(header), *(line(f) for f in frames), line(end)]) + "\n"


def read_take(path: str | Path) -> tuple[StreamHeader, list[Frame], StreamEnd | None]:
    header: StreamHeader | None = None
    frames: list[Frame] = []
    end: StreamEnd | None = None
    with Path(path).open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                data = json.loads(raw)
                kind = data.get("type") if isinstance(data, dict) else None
                if kind == "header" and header is None:
                    header = StreamHeader.model_validate(data)
                elif kind == "frame" and header is not None:
                    frames.append(Frame.model_validate(data))
                elif kind == "end" and header is not None:
                    end = StreamEnd.model_validate(data)
                else:
                    raise TakeError(f"{path}:{number}: unexpected {kind!r} message")
            except (ValueError, RuntimeError) as exc:
                if isinstance(exc, TakeError):
                    raise
                raise TakeError(f"{path}:{number}: {exc}") from exc
    if header is None:
        raise TakeError(f"{path}: no header line")
    return header, frames, end
