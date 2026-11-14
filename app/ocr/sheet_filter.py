"""Remove the answer sheet's printed matter (headers, instructions, margin notes, barcodes and their digits,
page numbers) from a recognised page, using blank-sheet templates.

Works on the RAW detection boxes (before they are merged into lines), so a printed label and handwriting on the
same row can be separated. Nothing is dropped silently: every excluded box is returned with its reason and is
stored in ocr.json. When in doubt a box is KEPT -- losing a line of the essay is worse than leaving a printed
line in the transcription.

Steps for one photographed page:
 1. candidates: photo boxes whose text matches a template line (normalised text, similarity >= 0.85, both
    texts >= 4 characters). A word may match several template lines and vice versa.
 2. alignment (robust): every pair of well-separated candidates proposes a similarity transform (scale,
    rotation, shift) from template pixels to photo pixels; the proposal that most candidates agree with
    (position within a tolerance AND matching box size) wins and is refined by least squares on those inliers.
    A student who writes a printed word ("Name") somewhere in the essay therefore cannot pull the alignment
    off. Phone photos are cropped / shifted / slightly rotated, so positions are never compared without it.
    The template page with the most inliers is used.
 3. exclusion reasons
      template_text    matches template text and sits where the template has it (aligned); a long line
                       (>= 8 characters) may be a little off position. Without alignment only long matches go,
                       short ones ("Name", "Date") could be the student's own words.
      template_region  aligned, lies mostly (>= 60 % of its area) inside where the template has printed text,
                       and has no real letters (barcode digits, page numbers that differ from sheet to sheet) or
                       partly matches that printed text ("Page 3" vs "Page 2"), or is vertical text on a vertical
                       printed area (margin notes such as "DO NOT WRITE IN THIS MARGIN", which the recogniser may
                       read differently on the blank and the filled sheet). Horizontal handwriting with words is
                       never removed by position alone.
      outside_body     no real letters (digits / symbols / barcode stripes read as "IlI|") and above or below
                       every line that does contain words -- the fallback when alignment is not possible
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median

import numpy as np

from .common import Line

MATCH = 0.85
LONG_TEXT = 8
REGION_OVERLAP = 0.6
REGION_TEXT_SIM = 0.6
VERTICAL = 2.5                     # box height / width at which text counts as vertical (margin notes)
MAX_CANDIDATES = 40
_STRIPES = set("Il1|!i")


def norm_text(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def has_words(s: str) -> bool:
    """True when the text contains real letters (barcode stripes misread as I/l/1/| do not count)."""
    letters = [c for c in s if c.isalpha() and c not in _STRIPES]
    return len(letters) >= 2


@dataclass
class Transform:
    a: float
    b: float
    tx: float
    ty: float

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return self.a * x - self.b * y + self.tx, self.b * x + self.a * y + self.ty

    @property
    def scale(self) -> float:
        return math.hypot(self.a, self.b)

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.b, self.a))


def _center(box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _sane(t: Transform) -> bool:
    return 0.2 <= t.scale <= 5.0 and abs(t.angle_deg) <= 20


def _from_two(p1, p2, q1, q2) -> Transform | None:
    """Exact similarity mapping p1->q1, p2->q2 (complex numbers: q = z*p + t)."""
    dp = complex(*p2) - complex(*p1)
    if abs(dp) < 1e-6:
        return None
    z = (complex(*q2) - complex(*q1)) / dp
    t = complex(*q1) - z * complex(*p1)
    return Transform(z.real, z.imag, t.real, t.imag)


def _fit(pairs: list[tuple[list, list]]) -> Transform | None:
    """Least squares on box centres (the centre of an axis-aligned box of a rotated rectangle is exact)."""
    if len(pairs) < 2:
        return None
    rows, rhs = [], []
    for tb, pb in pairs:
        (x, y), (u, v) = _center(tb), _center(pb)
        rows.append([x, -y, 1, 0]); rhs.append(u)
        rows.append([y, x, 0, 1]); rhs.append(v)
    sol, *_ = np.linalg.lstsq(np.asarray(rows, float), np.asarray(rhs, float), rcond=None)
    return Transform(*map(float, sol))


def _map_box(t: Transform, box, pad: float = 0.0) -> list[float]:
    x0, y0, x1, y1 = box
    pts = [t.apply(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]


def _offset(t: Transform, tb, pb) -> float:
    return math.dist(t.apply(*_center(tb)), _center(pb))


def _consistent(t: Transform, tb, pb, tol: float) -> bool:
    """The photo box is where the template box lands, with about the same size."""
    if _offset(t, tb, pb) > tol:
        return False
    m = _map_box(t, tb)
    rw = (pb[2] - pb[0]) / max(1.0, m[2] - m[0])
    rh = (pb[3] - pb[1]) / max(1.0, m[3] - m[1])
    return 0.6 <= rw <= 1.6 and 0.5 <= rh <= 2.0


def estimate_transform(pairs: list[tuple[list, list]], tol: float) -> Transform | None:
    """pairs: (template box, photo box) candidates, possibly with wrong matches. tol: allowed position error in
    photo pixels. None when the candidates do not give a trustworthy alignment."""
    pairs = pairs[:MAX_CANDIDATES]
    n = len(pairs)
    if n < 2:
        return None
    spread = 4 * tol
    tc = [_center(tb) for tb, _ in pairs]
    pc = [_center(pb) for _, pb in pairs]
    best: list[int] = []
    best_err = math.inf
    for a in range(n):
        for b in range(a + 1, n):
            if math.dist(pc[a], pc[b]) < spread or math.dist(tc[a], tc[b]) < 1e-6:
                continue
            t = _from_two(tc[a], tc[b], pc[a], pc[b])
            if t is None or not _sane(t):
                continue
            inl = [k for k in range(n) if _consistent(t, pairs[k][0], pairs[k][1], tol)]
            if len(inl) < 2:
                continue
            err = sum(_offset(t, *pairs[k]) for k in inl)
            if len(inl) > len(best) or (len(inl) == len(best) and err < best_err):
                best, best_err = inl, err
    if len(best) < 2:
        return None
    t = _fit([pairs[k] for k in best])
    for _ in range(2):                                    # refine: re-collect inliers with the fitted transform
        inl = [k for k in range(n) if _consistent(t, pairs[k][0], pairs[k][1], tol)]
        if len(inl) < 2:
            return None
        t = _fit([pairs[k] for k in inl]) or t
    if not _sane(t):
        return None
    inl = [k for k in range(n) if _consistent(t, pairs[k][0], pairs[k][1], tol)]
    if len(inl) < 2 or max(math.dist(pc[i], pc[j]) for i in inl for j in inl) < spread:
        return None
    if median(_offset(t, *pairs[k]) for k in inl) > 0.8 * tol:
        return None
    return t


def _vertical(box) -> bool:
    return (box[3] - box[1]) >= VERTICAL * max(1.0, box[2] - box[0])


def _overlap_frac(box, region) -> float:
    w = min(box[2], region[2]) - max(box[0], region[0])
    h = min(box[3], region[3]) - max(box[1], region[1])
    if w <= 0 or h <= 0:
        return 0.0
    area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
    return w * h / area


def _candidates(boxes: list[Line], tlines: list[dict]) -> list[tuple[float, int, int]]:
    """All text matches (similarity, photo index, template index), longest / best first."""
    out = []
    pn = [norm_text(b.text) for b in boxes]
    for j, tl in enumerate(tlines):
        tn = norm_text(tl["text"])
        if len(tn) < 4:
            continue
        for i, p in enumerate(pn):
            if len(p) < 4 or abs(len(p) - len(tn)) > max(3, 0.4 * len(tn)):
                continue
            s = _sim(p, tn)
            if s >= MATCH:
                out.append((s, i, j))
    out.sort(key=lambda c: (-len(norm_text(tlines[c[2]]["text"])), -c[0], c[1], c[2]))
    return out


def _one_to_one(cand) -> int:
    used_i, used_j, n = set(), set(), 0
    for _s, i, j in sorted(cand, reverse=True):
        if i not in used_i and j not in used_j:
            used_i.add(i); used_j.add(j); n += 1
    return n


def _tolerance(boxes: list[Line], line_h: float) -> float:
    x0 = min(b.box[0] for b in boxes); y0 = min(b.box[1] for b in boxes)
    x1 = max(b.box[2] for b in boxes); y1 = max(b.box[3] for b in boxes)
    return max(line_h, 0.015 * math.hypot(x1 - x0, y1 - y0))


def filter_page(boxes: list[Line], template_pages: list[dict]) -> tuple[list[Line], list[dict], dict]:
    """template_pages: [{"page_id", "lines": [{"text", "box"}]}] (template boxes in the blank photo's pixels).
    Returns (kept boxes, excluded [{text, box, score, reason}], info)."""
    info = {"template_page": None, "anchors": 0, "aligned": False}
    if not boxes:
        return [], [], info
    line_h = median(max(1.0, b.box[3] - b.box[1]) for b in boxes)
    tol = _tolerance(boxes, line_h)
    # choose the template page: most aligned inliers, then most text matches
    best = None
    for tp in template_pages:
        tl = tp.get("lines") or []
        if not tl:
            continue
        cand = _candidates(boxes, tl)
        t = estimate_transform([(tl[j]["box"], boxes[i].box) for _s, i, j in cand], tol)
        inl = 0 if t is None else sum(1 for _s, i, j in cand[:MAX_CANDIDATES]
                                      if _consistent(t, tl[j]["box"], boxes[i].box, tol))
        key = (inl, _one_to_one(cand))
        if best is None or key > best[0]:
            best = (key, tp, cand, t)
    excluded: dict[int, str] = {}
    if best is not None:
        (inl, matches), tp, cand, t = best
        tl = tp["lines"]
        info["template_page"] = tp.get("page_id")
        info["anchors"] = inl if t is not None else matches
        info["aligned"] = t is not None
        if t is not None:
            info["scale"], info["angle_deg"] = round(t.scale, 4), round(t.angle_deg, 2)
        for _s, i, j in cand:
            if i in excluded:
                continue
            tb, pb = tl[j]["box"], boxes[i].box
            long_enough = len(norm_text(tl[j]["text"])) >= LONG_TEXT
            if t is None:
                hit = long_enough
            else:
                hit = _consistent(t, tb, pb, tol) or (long_enough and _offset(t, tb, pb) <= 4 * tol)
            if hit:
                excluded[i] = "template_text"
        if t is not None:
            regions = []
            for x in tl:
                pad = 0.25 * max(1.0, (x["box"][3] - x["box"][1]) * t.scale)
                regions.append((_map_box(t, x["box"], pad), norm_text(x["text"]), _vertical(x["box"])))
            for i, b in enumerate(boxes):
                if i in excluded:
                    continue
                bn = norm_text(b.text)
                for r, rn, rv in regions:
                    if _overlap_frac(b.box, r) >= REGION_OVERLAP and \
                            (not has_words(b.text) or _sim(bn, rn) >= REGION_TEXT_SIM or (rv and _vertical(b.box))):
                        excluded[i] = "template_region"
                        break
    # fallback: letter-less boxes outside the essay body (barcode digits / page numbers when not aligned)
    worded = [b for i, b in enumerate(boxes) if i not in excluded and has_words(b.text)]
    if worded:
        top = min(b.box[1] for b in worded)
        bottom = max(b.box[3] for b in worded)
        for i, b in enumerate(boxes):
            if i in excluded or has_words(b.text):
                continue
            cy = (b.box[1] + b.box[3]) / 2
            if cy < top or cy > bottom:
                excluded[i] = "outside_body"
    kept = [b for i, b in enumerate(boxes) if i not in excluded]
    out = [{"text": boxes[i].text, "box": [round(v, 1) for v in boxes[i].box],
            "score": None if boxes[i].score is None else round(boxes[i].score, 4), "reason": r}
           for i, r in sorted(excluded.items())]
    return kept, out, info
