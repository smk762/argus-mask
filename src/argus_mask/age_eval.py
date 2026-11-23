"""Age read for a LAM head (#172): apparent age of the head against its portrait.

#151 found photograph portraits come out of LAM about ten years older; character art did not.
This scores the two ages and flags when the head reads much older. It is a likeness score, not the
18-or-over gate (that one is darkroom's and curator's, on the portrait, before any build).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Head − portrait years that counts as "much older" (#151: photo ≈ +10, character ≈ 0).
OLDER_YEARS = 7.0


@dataclass(frozen=True)
class AgeReport:
    """Portrait age, head age, their difference, and whether the head reads older."""

    portrait_years: float
    head_years: float
    delta_years: float  # head − portrait
    older: bool
    threshold: float
    source: str

    def as_data(self) -> dict[str, Any]:
        return {
            "portrait_years": round(self.portrait_years, 1),
            "head_years": round(self.head_years, 1),
            "delta_years": round(self.delta_years, 1),
            "older": self.older,
            "threshold": self.threshold,
            "source": self.source,
        }


def score(
    portrait_years: float,
    head_years: float,
    *,
    threshold: float = OLDER_YEARS,
    source: str = "mivolo",
) -> AgeReport:
    """Flag when the head reads at least *threshold* years older than the portrait."""
    delta = float(head_years) - float(portrait_years)
    return AgeReport(
        portrait_years=float(portrait_years),
        head_years=float(head_years),
        delta_years=delta,
        older=delta >= threshold,
        threshold=float(threshold),
        source=source,
    )
