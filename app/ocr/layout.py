"""Layout restoration: positioned text lines -> essay text with paragraphs, blank lines and indentation.

The per-line record (text + box + score) is always kept in ocr.json; this module only builds the transcription:
 - visual lines of one paragraph are joined with a space; a line-end hyphen after a letter is a word break
   ("h-" + "igher" -> "higher") and is removed (same rule as fast_ocr.join_lines);
 - a new paragraph starts when
     * the vertical gap to the previous line is > blank_line_factor x median line pitch (also emits a blank line),
     * the line is indented by > 1.5 x median character width from the typical left edge, or
     * the previous line is short: it stops before 60% of the typical width, or it ends with sentence punctuation
       and stops more than max(2.5 chars, 8% of the width) before the typical right edge;
 - tiny fragments (height < 0.7 x median line height and at most 3 characters: stray marks, small insertions)
   are kept inline in the current paragraph and never start a paragraph;
 - an indented paragraph keeps 2 spaces per 1.5 character widths (max 8);
 - pages are concatenated; across a page break only the indent / short-line rules apply.
One paragraph per output line; paragraphs separated by "\n" (plus an empty line where a gap was seen).
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .common import Line

HYPHENS = "-‐‑–—一－_~"


SKEW_MIN_DEG = 0.3
SKEW_MAX_DEG = 30.0


def estimate_skew(boxes: list[Line]) -> float:
    """Page tilt in degrees (image coordinates, positive = lines fall to the right): median angle of the wide
    horizontal text boxes (w >= 3h). 0 when fewer than 3 such boxes."""
    angs = [b.rect[4] for b in boxes
            if b.rect and b.rect[2] >= 3 * b.rect[3] and abs(b.rect[4]) <= SKEW_MAX_DEG]
    return median(angs) if len(angs) >= 3 else 0.0


def deskew(boxes: list[Line], width: float, height: float, skew: float) -> list[tuple]:
    """(box, text, score) items with every box rotated by -skew about the page centre (upright boxes rebuilt from
    the rotated rectangles), so reading order and left edges are not distorted by a tilted photo."""
    if abs(skew) < SKEW_MIN_DEG:
        return [(b.box, b.text, b.score) for b in boxes]
    import math
    phi = math.radians(-skew)
    c, s = math.cos(phi), math.sin(phi)
    ox, oy = width / 2, height / 2
    out = []
    for b in boxes:
        if b.rect:
            cx, cy, w, h, ang = b.rect
        else:
            cx, cy = (b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2
            w, h, ang = b.box[2] - b.box[0], b.box[3] - b.box[1], 0.0
        dx, dy = cx - ox, cy - oy
        nx, ny = ox + dx * c - dy * s, oy + dx * s + dy * c
        r = math.radians(ang - skew)
        bw = abs(w * math.cos(r)) + abs(h * math.sin(r))
        bh = abs(w * math.sin(r)) + abs(h * math.cos(r))
        out.append(([nx - bw / 2, ny - bh / 2, nx + bw / 2, ny + bh / 2], b.text, b.score))
    return out


def group_boxes(boxes: list[Line], width: float, height: float) -> tuple[list[Line], float]:
    """Deskew (when the photo is tilted) and merge raw detection boxes into visual lines. Returns (lines, skew)."""
    skew = estimate_skew(boxes)
    if abs(skew) < SKEW_MIN_DEG:
        skew = 0.0
    return group_lines(deskew(boxes, width, height, skew)), skew


def group_lines(boxes_texts: list[tuple[list[float], str, float | None]]) -> list[Line]:
    """Merge detection boxes that sit on the same visual line (left to right, joined by a space)."""
    items = sorted((t for t in boxes_texts if t[1].strip()), key=lambda t: (t[0][1] + t[0][3]) / 2)
    rows: list[list[tuple]] = []
    for it in items:
        b = it[0]
        cy, h = (b[1] + b[3]) / 2, max(1.0, b[3] - b[1])
        placed = False
        for row in reversed(rows[-3:]):
            rb = [x[0] for x in row]
            r0, r1 = min(x[1] for x in rb), max(x[3] for x in rb)
            rh = median(max(1.0, x[3] - x[1]) for x in rb)
            overlap = min(b[3], r1) - max(b[1], r0)
            if overlap > 0.5 * min(h, rh) and r0 <= cy <= r1:
                row.append(it)
                placed = True
                break
        if not placed:
            rows.append([it])
    out: list[Line] = []
    for row in rows:
        row.sort(key=lambda t: t[0][0])
        text = " ".join(t[1].strip() for t in row)
        x0 = min(t[0][0] for t in row); y0 = min(t[0][1] for t in row)
        x1 = max(t[0][2] for t in row); y1 = max(t[0][3] for t in row)
        sc = [t[2] for t in row if t[2] is not None]
        out.append(Line(text, [x0, y0, x1, y1], sum(sc) / len(sc) if sc else None))
    out.sort(key=lambda l: (l.box[1] + l.box[3]) / 2)
    return out


_SENTENCE_END = '.!?"\'”’)'


@dataclass
class _Flag:
    text: str
    gap_before: bool
    indent: str
    short: bool
    tiny: bool = False


def _page_flags(lines: list[Line], blank_line_factor: float) -> list[_Flag]:
    if not lines:
        return []
    heights = [max(1.0, l.box[3] - l.box[1]) for l in lines]
    mh = median(heights)
    tiny = [h < 0.7 * mh and len(l.text.strip()) <= 3 for l, h in zip(lines, heights)]
    body = [l for l, t in zip(lines, tiny) if not t] or list(lines)
    if body:     # only fragments BETWEEN body lines count as tiny; a footer/header ("共 页") stays a line of its own
        top, bottom = body[0].box[1], body[-1].box[3]
        tiny = [t and top < (l.box[1] + l.box[3]) / 2 < bottom for l, t in zip(lines, tiny)]
        body = [l for l, t in zip(lines, tiny) if not t] or list(lines)
    centers = [(l.box[1] + l.box[3]) / 2 for l in body]
    pitches = [b - a for a, b in zip(centers, centers[1:]) if b > a]
    pitch = median(pitches) if pitches else 0.0
    char_ws = [(l.box[2] - l.box[0]) / max(1, len(l.text)) for l in body if len(l.text) >= 4]
    cw = median(char_ws) if char_ws else 0.0
    lefts = sorted(l.box[0] for l in body)
    rights = sorted(l.box[2] for l in body)
    left = lefts[len(lefts) // 4]
    right = rights[(3 * len(rights)) // 4]
    width = max(1.0, right - left)
    out = []
    prev_center = None
    for l, is_tiny in zip(lines, tiny):
        if is_tiny:
            out.append(_Flag(l.text.strip(), False, "", False, True))
            continue
        c = (l.box[1] + l.box[3]) / 2
        gap = prev_center is not None and pitch > 0 and (c - prev_center) > blank_line_factor * pitch
        prev_center = c
        off = l.box[0] - left
        indent = " " * min(8, int(off / (1.5 * cw)) * 2) if cw > 0 and off > 1.5 * cw else ""
        t = l.text.rstrip()
        ends_hyphen = len(t) > 1 and t[-1] in HYPHENS and t[-2].isalpha()
        stop = right - l.box[2]
        short = len(body) > 1 and not ends_hyphen and (
            l.box[2] < left + 0.6 * width
            or (bool(t) and t[-1] in _SENTENCE_END and stop > max(2.5 * cw, 0.08 * width)))
        out.append(_Flag(l.text.strip(), gap, indent, short))
    return out


def _join(a: str, b: str) -> str:
    if not a:
        return b
    if len(a) > 1 and a[-1] in HYPHENS and a[-2].isalpha():
        return a[:-1].rstrip() + b
    return f"{a} {b}"


def restore_pages(pages: list[list[Line]], blank_line_factor: float = 1.6) -> str:
    paras: list[tuple[bool, str, str]] = []           # (blank line before, indent, text)
    prev: _Flag | None = None
    for lines in pages:
        flags = _page_flags(lines, blank_line_factor)
        for k, f in enumerate(flags):
            if not f.text:
                continue
            if f.tiny and paras:                      # stray mark / small insertion: inline, state unchanged
                b, ind, t = paras[-1]
                paras[-1] = (b, ind, _join(t, f.text))
                continue
            gap = f.gap_before if k > 0 else False
            new = prev is None or gap or bool(f.indent) or prev.short
            if new:
                paras.append((gap and bool(paras), f.indent, f.text))
            else:
                b, ind, t = paras[-1]
                paras[-1] = (b, ind, _join(t, f.text))
            prev = f
    out: list[str] = []
    for blank, ind, t in paras:
        if blank:
            out.append("")
        out.append(ind + t)
    return "\n".join(out)


def restore_text(lines: list[Line], blank_line_factor: float = 1.6) -> str:
    return restore_pages([lines], blank_line_factor)


def plain_join(pages: list[list[Line]]) -> str:
    """Positions not reliable: keep the engine's line order, one line per line."""
    return "\n".join(l.text for lines in pages for l in lines if l.text.strip())
