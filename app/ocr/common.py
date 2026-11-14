"""Shared data types for OCR engines."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Line:
    text: str
    box: list[float]                 # axis-aligned [x0, y0, x1, y1] in original image pixels
    score: float | None = None
    rect: tuple | None = None        # rotated rectangle (cx, cy, w, h, angle_deg), w along the text; for deskew


@dataclass
class PageResult:
    lines: list[Line]
    width: int
    height: int
    raw: str | None = None           # engine raw output when the engine has one
    layout_reliable: bool = False    # whether line positions can be trusted for blank-line/indent restoration
    warnings: list[str] = field(default_factory=list)
    boxes: list[Line] = field(default_factory=list)   # raw detection boxes (before merging into lines)
    skew_deg: float = 0.0            # page tilt removed before line grouping; `lines` boxes are deskewed


class OcrError(Exception):
    """Local recognition failed (engine error / OOM / unreadable image)."""
