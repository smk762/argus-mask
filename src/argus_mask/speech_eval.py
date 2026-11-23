"""Speech eval for a LAM head (#171): lip gap over a fixed take, against a reference head.

#161 measured the same Audio2Face take with MediaPipe: lip-parting shapes held under speech left the
lips closed on 42% of frames with 52 direction reversals; dropping those shapes under speech raised
closed to 62% and cut reversals to 38 (in line with ``splat-01``). This module scores a gap series and
flags a stutter or an oversized mouth from those numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Eye-spacing units: below this the mouth reads closed (#161's measure).
CLOSED_GAP = 0.04

# Absolute floors from #161's before/after midpoints.
STUTTER_CLOSED_BELOW = 0.50
STUTTER_REVERSALS_ABOVE = 45

# Relative to the reference head (splat-01 / the after measure).
STUTTER_CLOSED_DROP = 0.12
STUTTER_REVERSAL_RISE = 10
OVERSIZE_P95_RATIO = 1.35

# Baked #161 "after" / splat-01-like reference when no head is measured.
REF_CLOSED = 0.62
REF_REVERSALS = 38
REF_OPENING_P95 = 0.12


@dataclass(frozen=True)
class LipMetrics:
    closed_share: float
    reversals: int
    opening_p95: float
    frames: int

    def as_data(self) -> dict[str, Any]:
        return {
            "closed_share": round(self.closed_share, 4),
            "reversals": int(self.reversals),
            "opening_p95": round(self.opening_p95, 4),
            "frames": int(self.frames),
        }


@dataclass(frozen=True)
class SpeechReport:
    """Candidate vs reference, with stutter / oversized flags."""

    candidate: LipMetrics
    reference: LipMetrics
    stutter: bool
    oversized: bool
    reasons: tuple[str, ...]

    def as_data(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.as_data(),
            "reference": self.reference.as_data(),
            "stutter": self.stutter,
            "oversized": self.oversized,
            "reasons": list(self.reasons),
        }


def metrics_from_gaps(gaps: list[float] | tuple[float, ...], *, closed_below: float = CLOSED_GAP) -> LipMetrics:
    """Share closed, direction reversals, and opening at p95 from a lip-gap series."""
    if not gaps:
        raise ValueError("need at least one lip-gap sample")
    series = [max(0.0, float(g)) for g in gaps]
    closed = sum(1 for g in series if g < closed_below) / len(series)
    deltas = [series[i] - series[i - 1] for i in range(1, len(series))]
    reversals = 0
    last_sign = 0
    for d in deltas:
        sign = 1 if d > 1e-6 else -1 if d < -1e-6 else 0
        if sign == 0:
            continue
        if last_sign != 0 and sign != last_sign:
            reversals += 1
        last_sign = sign
    ordered = sorted(series)
    # Nearest-rank p95.
    idx = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return LipMetrics(
        closed_share=float(closed),
        reversals=int(reversals),
        opening_p95=float(ordered[idx]),
        frames=len(series),
    )


def reference_baseline() -> LipMetrics:
    """#161's after numbers (splat-01-like), used when no reference head is measured."""
    return LipMetrics(
        closed_share=REF_CLOSED,
        reversals=REF_REVERSALS,
        opening_p95=REF_OPENING_P95,
        frames=0,
    )


def score(candidate: LipMetrics, reference: LipMetrics | None = None) -> SpeechReport:
    """Flag stutter and oversized mouth against *reference* (default: #161 after)."""
    ref = reference or reference_baseline()
    reasons: list[str] = []
    stutter = False
    if candidate.closed_share < STUTTER_CLOSED_BELOW:
        stutter = True
        reasons.append(f"lips closed on {candidate.closed_share:.0%} of frames (under {STUTTER_CLOSED_BELOW:.0%})")
    if candidate.reversals > STUTTER_REVERSALS_ABOVE:
        stutter = True
        reasons.append(f"{candidate.reversals} direction reversals (over {STUTTER_REVERSALS_ABOVE})")
    if candidate.closed_share + STUTTER_CLOSED_DROP < ref.closed_share:
        stutter = True
        reasons.append(
            f"closed share {candidate.closed_share:.0%} is {ref.closed_share - candidate.closed_share:.0%} "
            f"below the reference ({ref.closed_share:.0%})"
        )
    if candidate.reversals >= ref.reversals + STUTTER_REVERSAL_RISE:
        stutter = True
        reasons.append(
            f"{candidate.reversals} reversals vs reference {ref.reversals} (+{candidate.reversals - ref.reversals})"
        )
    oversized = False
    if ref.opening_p95 > 1e-9 and candidate.opening_p95 > ref.opening_p95 * OVERSIZE_P95_RATIO:
        oversized = True
        reasons.append(
            f"opening p95 {candidate.opening_p95:.3f} is "
            f"{candidate.opening_p95 / ref.opening_p95:.2f}× the reference ({ref.opening_p95:.3f})"
        )
    return SpeechReport(
        candidate=candidate,
        reference=ref,
        stutter=stutter,
        oversized=oversized,
        reasons=tuple(reasons),
    )


def synthetic_gaps(*, stutter: bool, frames: int = 120) -> list[float]:
    """A gap series that scores like #161 before (*stutter*) or after (clean)."""
    import math

    gaps: list[float] = []
    # Stutter: stay open more often, chatter the direction every few frames.
    # Clean: mostly closed with smooth open–close cycles.
    for i in range(frames):
        t = i / 30.0
        if stutter:
            base = 0.06 + 0.05 * abs(math.sin(t * 9.0))
            # Jab the gap so deltas reverse often.
            jab = 0.04 * (1 if (i // 2) % 2 == 0 else -1)
            gaps.append(max(0.0, base + jab))
        else:
            # Soft syllables: open briefly, mostly under CLOSED_GAP.
            cycle = (i % 20) / 20.0
            if 0.2 < cycle < 0.55:
                gaps.append(0.02 + 0.10 * math.sin((cycle - 0.2) / 0.35 * math.pi))
            else:
                gaps.append(0.01 + 0.005 * abs(math.sin(t * 2.0)))
    return gaps
