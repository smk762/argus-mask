"""PCM in, mono float32 out, at the rate the driver wants.

Three pieces: :class:`PcmDecoder` turns a source's byte chunks into mono
float32 (a chunk may end mid-sample; the remainder is carried), :class:`Resampler`
is a streaming resampler, and :func:`read_wav` loads a file for the offline
commands. No audio is ever written to disk here.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from argus_mask.models import AudioFormat

DRIVER_RATE = 16000  # what every driver is fed

_SAMPLE_BYTES = {"f32le": 4, "s16le": 2}


class AudioError(ValueError):
    """Audio that cannot be read: a bad WAV, an unsupported encoding."""


def _to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    if channels == 1:
        return samples
    return samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)


class PcmDecoder:
    """Interleaved PCM bytes to mono float32 in [-1, 1], across chunk boundaries."""

    def __init__(self, fmt: AudioFormat) -> None:
        self.fmt = fmt
        self._frame_bytes = _SAMPLE_BYTES[fmt.format] * fmt.channels
        self._rest = b""

    def feed(self, data: bytes) -> np.ndarray:
        buf = self._rest + data
        usable = len(buf) - len(buf) % self._frame_bytes
        self._rest = buf[usable:]
        if not usable:
            return np.zeros(0, dtype=np.float32)
        if self.fmt.format == "f32le":
            samples = np.frombuffer(buf[:usable], dtype="<f4").astype(np.float32)
        else:
            samples = np.frombuffer(buf[:usable], dtype="<i2").astype(np.float32) / 32768.0
        # A NaN or an out-of-range float would reach the viewer's speakers and the driver's maths.
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(_to_mono(samples, self.fmt.channels), -1.0, 1.0)


class Resampler:
    """Streaming mono resampler. The same rate in and out passes audio through."""

    def __init__(self, in_rate: int, out_rate: int = DRIVER_RATE) -> None:
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._stream = None
        if in_rate != out_rate:
            import soxr  # noqa: PLC0415 - only when a rate actually changes

            self._stream = soxr.ResampleStream(in_rate, out_rate, 1, dtype="float32", quality="HQ")

    def feed(self, samples: np.ndarray, *, last: bool = False) -> np.ndarray:
        if self._stream is None:
            return samples
        return self._stream.resample_chunk(samples, last=last)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """A WAV file as ``(mono float32, sample_rate)``.

    Reads integer PCM (8, 16, 24, 32 bit) and IEEE float (32, 64 bit), which the
    standard library's ``wave`` does not.
    """
    return decode_wav(Path(path).read_bytes(), str(path))


def decode_wav(data: bytes, path: str = "the audio") -> tuple[np.ndarray, int]:
    """A WAV file's bytes as ``(mono float32, sample_rate)``; *path* names it in errors. See `read_wav`."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError(f"{path}: not a RIFF/WAVE file")
    fmt: tuple[int, int, int, int] | None = None  # (encoding, channels, rate, bits)
    pcm: bytes | None = None
    pos = 12
    while pos + 8 <= len(data):
        chunk_id, size = data[pos : pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8 : pos + 8 + size]
        if chunk_id == b"fmt " and len(body) >= 16:
            encoding, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", body)
            if encoding == 0xFFFE and len(body) >= 26:  # WAVE_FORMAT_EXTENSIBLE: the real code is in the GUID
                encoding = struct.unpack_from("<H", body, 24)[0]
            fmt = (encoding, channels, rate, bits)
        elif chunk_id == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise AudioError(f"{path}: no fmt or data chunk")
    encoding, channels, rate, bits = fmt
    if channels < 1 or rate < 1:
        raise AudioError(f"{path}: {channels} channels at {rate} Hz")
    width = bits // 8
    pcm = pcm[: len(pcm) - len(pcm) % (width * channels)] if width else b""
    if encoding == 1 and bits == 8:
        samples = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif encoding == 1 and bits == 16:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    elif encoding == 1 and bits == 24:
        raw = np.frombuffer(pcm, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        samples = (value - ((value & 0x800000) << 1)).astype(np.float32) / 8388608.0
    elif encoding == 1 and bits == 32:
        samples = (np.frombuffer(pcm, dtype="<i4").astype(np.float64) / 2147483648.0).astype(np.float32)
    elif encoding == 3 and bits == 32:
        samples = np.frombuffer(pcm, dtype="<f4").astype(np.float32)
    elif encoding == 3 and bits == 64:
        samples = np.frombuffer(pcm, dtype="<f8").astype(np.float32)
    else:
        raise AudioError(f"{path}: unsupported WAV encoding {encoding} at {bits} bits")
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(_to_mono(samples, channels), -1.0, 1.0), rate
