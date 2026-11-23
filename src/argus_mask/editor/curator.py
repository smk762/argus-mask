"""argus-curator's face_source preset for a guided portrait (#168).

Upload → scan → ``POST /scan/{id}/select`` with ``face_source``. Each image left out names which check failed
(``face-source:<check>``). Mask maps those reasons onto the portrait step's checks; it does not re-implement them.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request


class CuratorError(Exception):
    """Curator could not run face_source, with the reason."""


@dataclass(frozen=True)
class FaceSourceRow:
    """One image under a face_source select: kept (empty reason) or left out with why."""

    name: str  # the file stem we uploaded (a candidate id)
    keep_reason: str
    rank: int | None = None


@dataclass(frozen=True)
class AgeReading:
    """Apparent age from curator's ``POST /faces`` (MiVOLO), for one image."""

    years: float
    source: str  # e.g. mivolo
    faces: int


@dataclass
class CuratorClient:
    """The few curator routes a guided portrait / age step needs. *opener* is injectable for tests."""

    url: str
    timeout: float = 120.0
    opener: Any = request.urlopen  # noqa: ANN401 - urllib opener or a test double

    def age(self, image: bytes) -> AgeReading:
        """Apparent age of the primary face in *image* (``POST /faces``, memory only)."""
        if not self.url.startswith(("http://", "https://")):
            raise CuratorError(f"ARGUS_MASK_CURATOR_URL {self.url!r}: must be an http(s) URL")
        if not image:
            raise CuratorError("no image for age")
        payload = self._call("POST", "/faces", image, content_type="image/png")
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise CuratorError("curator /faces returned non-JSON") from exc
        faces = data.get("faces")
        if not isinstance(faces, list) or not faces:
            raise CuratorError("curator found no face to age")
        source = str(data.get("source") or "mivolo")
        # Largest face by bbox area is the primary (curator may return several).
        best: dict | None = None
        best_area = -1.0
        for face in faces:
            if not isinstance(face, dict):
                continue
            years = face.get("age_estimate_years")
            if not isinstance(years, (int, float)):
                continue
            bbox = face.get("bbox") or [0, 0, 0, 0]
            try:
                x, y, w, h = (float(bbox[i]) for i in range(4))
                area = abs(w * h)
            except (TypeError, ValueError, IndexError):
                area = 0.0
            if area >= best_area:
                best_area = area
                best = face
        if best is None:
            raise CuratorError("curator returned faces with no age")
        return AgeReading(years=float(best["age_estimate_years"]), source=source, faces=len(faces))

    def face_source(self, images: dict[str, bytes], *, folder: str | None = None) -> list[FaceSourceRow]:
        """Run face_source on *images* (name → PNG/JPEG bytes). Returns one row per name."""
        if not images:
            return []
        if not self.url.startswith(("http://", "https://")):
            raise CuratorError(f"ARGUS_MASK_CURATOR_URL {self.url!r}: must be an http(s) URL")
        folder = folder or f"mask-guided/{uuid.uuid4().hex[:12]}"
        self._upload(folder, images)
        scan_id = self._scan(folder)
        rows = self._select(scan_id)
        by_stem = {Path_stem(r["rel_path"]): r for r in rows}
        out: list[FaceSourceRow] = []
        for name in images:
            found = by_stem.get(name)
            if found is None:
                out.append(FaceSourceRow(name=name, keep_reason="face-source:missing"))
            else:
                out.append(
                    FaceSourceRow(
                        name=name,
                        keep_reason=str(found.get("keep_reason") or ""),
                        rank=found.get("rank") if isinstance(found.get("rank"), int) else None,
                    )
                )
        return out

    def _upload(self, folder: str, images: dict[str, bytes]) -> None:
        boundary = f"----mask{uuid.uuid4().hex}"
        parts: list[bytes] = []
        for name, data in images.items():
            filename = f"{name}.png"
            parts.append(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
                    f"Content-Type: image/png\r\n\r\n"
                ).encode()
                + data
                + b"\r\n"
            )
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="folder"\r\n\r\n{folder}\r\n--{boundary}--\r\n'
            ).encode()
        )
        body = b"".join(parts)
        self._call(
            "POST",
            "/upload",
            body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )

    def _scan(self, folder: str) -> str:
        payload = self._call("POST", "/scan/folder", json.dumps({"path": folder}).encode())
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise CuratorError("curator scan returned non-JSON") from exc
        scan_id = data.get("scan_id") or data.get("id")
        if not isinstance(scan_id, str) or not scan_id:
            raise CuratorError("curator scan returned no scan_id")
        return scan_id

    def _select(self, scan_id: str) -> list[dict]:
        body = json.dumps({"face_source": {}, "rank": "identity", "min_score": 0.0}).encode()
        payload = self._call("POST", f"/scan/{parse.quote(scan_id)}/select", body)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise CuratorError("curator select returned non-JSON") from exc
        rows = data.get("rows")
        if not isinstance(rows, list):
            raise CuratorError("curator select returned no rows")
        return [r for r in rows if isinstance(r, dict)]

    def _call(self, method: str, path: str, body: bytes, *, content_type: str = "application/json") -> bytes:
        req = request.Request(
            f"{self.url.rstrip('/')}{path}",
            data=body,
            method=method,
            headers={"Content-Type": content_type},
        )
        try:
            with self.opener(req, timeout=self.timeout) as reply:
                return reply.read()
        except error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise CuratorError(f"curator {method} {path}: {exc.code} {detail}") from exc
        except OSError as exc:
            raise CuratorError(f"curator is unreachable at {self.url}: {exc}") from exc


def Path_stem(rel_path: str) -> str:
    """``a/b/c1.png`` → ``c1``."""
    name = rel_path.rsplit("/", 1)[-1]
    if "." in name:
        return name.rsplit(".", 1)[0]
    return name


# face_source keep_reason → (check name, verdict). Hard failures refuse; pose/quality advise.
FACE_SOURCE_CHECKS: dict[str, tuple[str, str]] = {
    "face-source:no-face": ("identity", "refused"),
    "face-source:faces": ("identity", "refused"),
    "face-source:age": ("adult", "refused"),
    "face-source:no-age": ("adult", "refused"),
    "face-source:face-size": ("framing", "refused"),
    "face-source:yaw": ("frontal", "advisory"),
    "face-source:pitch": ("frontal", "advisory"),
    "face-source:roll": ("frontal", "advisory"),
    "face-source:face-sharpness": ("sharpness", "advisory"),
    "face-source:eyes": ("eyes", "advisory"),
    "face-source:mouth": ("eyes", "advisory"),  # mouth-open sits with eyes/pose quality from curator
    "face-source:identity": ("identity", "advisory"),
    "face-source:identity-distance": ("identity", "advisory"),
    "face-source:missing": ("identity", "refused"),
}

CURATOR_CHECK_NAMES = ("identity", "adult", "frontal", "sharpness", "eyes")
