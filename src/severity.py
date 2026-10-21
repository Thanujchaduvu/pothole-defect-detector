"""Severity scoring: turns detection boxes into a coverage % and a severity label."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# (upper bound in %, label). Bands follow the assignment table.
BANDS = [(3.0, "LOW"), (10.0, "MEDIUM"), (20.0, "HIGH")]


def coverage_sum(boxes: Iterable[Sequence[float]], width: int, height: int) -> float:
    """Assignment formula: sum of box areas / image area (can exceed 100% with overlaps)."""
    total = 0.0
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, x1), min(width, x2)))
        y1, y2 = sorted((max(0, y1), min(height, y2)))
        total += max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return 100.0 * total / (width * height)


def coverage_union(boxes: Iterable[Sequence[float]], width: int, height: int) -> float:
    """Overlap-safe coverage: paints boxes on a binary mask and counts filled pixels."""
    mask = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        x1, x2 = int(max(0, round(x1))), int(min(width, round(x2)))
        y1, y2 = int(max(0, round(y1))), int(min(height, round(y2)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return 100.0 * float(mask.mean())


def severity_from_coverage(coverage_pct: float, has_detection: bool = True) -> str:
    """Map coverage % to NONE / LOW / MEDIUM / HIGH / CRITICAL."""
    if not has_detection:
        return "NONE"
    for upper, label in BANDS:
        if coverage_pct <= upper:
            return label
    return "CRITICAL"


def score(boxes: Sequence[Sequence[float]], width: int, height: int, mode: str = "union") -> dict:
    """Return {'coverage_pct', 'severity', 'n_detections'} for a list of xyxy boxes."""
    if mode not in ("union", "sum"):
        raise ValueError("mode must be 'union' or 'sum'")
    fn = coverage_union if mode == "union" else coverage_sum
    cov = fn(boxes, width, height) if len(boxes) else 0.0
    return {
        "coverage_pct": cov,
        "severity": severity_from_coverage(cov, has_detection=len(boxes) > 0),
        "n_detections": len(boxes),
    }
