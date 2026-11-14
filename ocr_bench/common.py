"""Shared helpers for the OCR speed benchmark: test cases, metrics, fast DB post-processing, crops, CTC decode.

Only numpy + PIL (no OpenCV / scipy / pyclipper), so it runs in the existing .venv as-is.
"""
from __future__ import annotations

import json
import math
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent          # writing_modify/
DATA = ROOT / "test_data"
DET_DIR = ROOT / "PP-OCRv6_medium_det_safetensors"
REC_DIR = ROOT / "PP-OCRv6_medium_rec_safetensors"
HY_DIR = ROOT / "hunyuan_OCR"
OUT = Path(__file__).resolve().parent / "results"
CACHE = Path(__file__).resolve().parent / "cache"


# ----------------------------------------------------------------------------------------------------------------
# test data
# ----------------------------------------------------------------------------------------------------------------
def parse_gt(raw: str) -> tuple[str, dict[str, int]]:
    lines = raw.splitlines()
    body, rest = [], ""
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith("##") or s.startswith("expected"):
            rest = "\n".join(lines[i:])
            break
        body.append(l)
    exp = {k: int(v) for k, v in re.findall(r'"([^"]+)"\s*:\s*(\d+)', rest)}
    return "\n".join(body).strip(), exp


def load_image(path: Path) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


def load_cases() -> list[dict]:
    cases = []
    for d in sorted(p for p in DATA.iterdir() if p.is_dir()):
        pages = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        gt, exp = parse_gt((d / "text.txt").read_text(encoding="utf-8"))
        cases.append({"name": d.name, "pages": pages, "images": [load_image(p) for p in pages], "gt": gt,
                      "expected": exp})
    return cases


# ----------------------------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------------------------
HYPHENS = "-‐‑–—一－_~"


def join_lines(lines: list[str]) -> str:
    """Join OCR lines into running text, removing end-of-line hyphenation ("u-" + "nderstand")."""
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if out and out[-1] in HYPHENS and len(out) > 1 and out[-2].isalpha():
            out = out[:-1].rstrip() + ln
        elif out:
            out += " " + ln
        else:
            out = ln
    return out


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower().replace("’", "'"))


def edit_distance(a, b) -> int:
    """Levenshtein distance, numpy row-vectorised (works for str or list of tokens)."""
    if isinstance(a, str):
        A = np.frombuffer(a.encode("utf-32-le"), dtype=np.uint32)
        B = np.frombuffer(b.encode("utf-32-le"), dtype=np.uint32)
    else:
        voc = {t: i for i, t in enumerate(set(a) | set(b))}
        A = np.array([voc[t] for t in a], dtype=np.int64)
        B = np.array([voc[t] for t in b], dtype=np.int64)
    n, m = len(A), len(B)
    if n == 0 or m == 0:
        return max(n, m)
    ar = np.arange(m + 1)
    prev = ar.copy()
    for i in range(1, n + 1):
        sub = prev[:-1] + (B != A[i - 1])
        tmp = np.empty(m + 1, dtype=np.int64)
        tmp[0] = i
        tmp[1:] = np.minimum(prev[1:] + 1, sub)
        prev = np.minimum.accumulate(tmp - ar) + ar          # insertion chain: row[j] = min_k tmp[k] + (j-k)
    return int(prev[-1])


def score_text(pred: str, gt: str, expected: dict[str, int]) -> dict:
    pt, gt_t = tokens(pred), tokens(gt)
    wer = edit_distance(pt, gt_t) / max(1, len(gt_t))
    pc, gc = " ".join(pt), " ".join(gt_t)
    cer = edit_distance(pc, gc) / max(1, len(gc))
    hit = tot = 0
    for w, c in expected.items():
        k = pt.count(w.lower())
        hit += min(k, c)
        tot += c
    return {"wer": wer, "cer": cer, "typo_hit": hit, "typo_total": tot}


def summarize(case_scores: list[dict]) -> dict:
    n = len(case_scores)
    return {"wer": sum(s["wer"] for s in case_scores) / n, "cer": sum(s["cer"] for s in case_scores) / n,
            "typo_hit": sum(s["typo_hit"] for s in case_scores), "typo_total": sum(s["typo_total"] for s in case_scores)}


# ----------------------------------------------------------------------------------------------------------------
# reading order: boxes -> lines
# ----------------------------------------------------------------------------------------------------------------
def order_lines(items: list[tuple[list[float], str]]) -> list[str]:
    """items: [(x0,y0,x1,y1), text]. Group boxes whose vertical centres overlap into one line, left to right."""
    items = [it for it in items if it[1].strip()]
    items.sort(key=lambda it: (it[0][1] + it[0][3]) / 2)
    lines: list[list] = []
    for box, text in items:
        cy, h = (box[1] + box[3]) / 2, box[3] - box[1]
        for ln in lines[-3:]:
            ly0, ly1 = ln[0]
            lh = ly1 - ly0
            ov = min(box[3], ly1) - max(box[1], ly0)
            if ov > 0.5 * min(h, lh) and ly0 <= cy <= ly1:
                ln[1].append((box, text))
                ln[0] = (min(ly0, box[1]), max(ly1, box[3]))
                break
        else:
            lines.append([(box[1], box[3]), [(box, text)]])
    lines.sort(key=lambda ln: (ln[0][0] + ln[0][1]) / 2)
    return [" ".join(t for _, t in sorted(ln[1], key=lambda bt: bt[0][0])) for ln in lines]


# ----------------------------------------------------------------------------------------------------------------
# fast DB post-processing (run-length connected components, numpy only)
# ----------------------------------------------------------------------------------------------------------------
def _runs(mask: np.ndarray):
    H, W = mask.shape
    pad = np.zeros((H, W + 2), np.int8)
    pad[:, 1:-1] = mask
    d = np.diff(pad, axis=1)
    sr, sc = np.nonzero(d == 1)
    er, ec = np.nonzero(d == -1)        # ec is exclusive end
    return sr, sc, ec


def _label_runs(sr, sc, ec, W):
    """8-connected labelling of horizontal runs; returns component id per run."""
    n = len(sr)
    stride = W + 2
    skey = sr.astype(np.int64) * stride + sc
    ekey = sr.astype(np.int64) * stride + ec
    prev_row = (sr.astype(np.int64) - 1) * stride
    lo = np.searchsorted(ekey, prev_row + sc, side="left")        # runs in row-1 whose end >= my start
    hi = np.searchsorted(skey, prev_row + ec, side="right")       # ... and whose start <= my end
    cnt = np.maximum(hi - lo, 0)
    cnt[sr == 0] = 0
    a = np.repeat(np.arange(n), cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    b = np.repeat(lo, cnt) + off
    lab = np.arange(n)
    if len(a):
        while True:
            la, lb = lab[a], lab[b]
            if np.array_equal(la, lb):
                break
            np.minimum.at(lab, np.maximum(la, lb), np.minimum(la, lb))
            while True:                                            # pointer jumping
                nl = lab[lab]
                if np.array_equal(nl, lab):
                    break
                lab = nl
    _, comp = np.unique(lab, return_inverse=True)
    return comp


def _hull(pts: np.ndarray) -> np.ndarray:
    pts = np.unique(pts, axis=0)
    if len(pts) <= 2:
        return pts
    P = pts[np.lexsort((pts[:, 1], pts[:, 0]))].tolist()

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) -
                                   (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h
    lower, upper = half(P), half(P[::-1])
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def min_area_rect(pts: np.ndarray):
    """Rotating calipers over convex hull. Returns (center, (w,h) along (u,v) axes, u_axis)."""
    hull = _hull(pts)
    if len(hull) < 3:
        c = pts.mean(0)
        return c, (float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))), np.array([1.0, 0.0])
    e = np.roll(hull, -1, axis=0) - hull
    ang = np.arctan2(e[:, 1], e[:, 0]) % (np.pi / 2)
    ang = np.unique(np.round(ang, 6))
    u = np.stack([np.cos(ang), np.sin(ang)], 1)                  # (k,2)
    v = np.stack([-u[:, 1], u[:, 0]], 1)
    pu, pv = hull @ u.T, hull @ v.T                             # (n,k)
    area = np.ptp(pu, 0) * np.ptp(pv, 0)
    k = int(np.argmin(area))
    u0, u1, v0, v1 = pu[:, k].min(), pu[:, k].max(), pv[:, k].min(), pv[:, k].max()
    c = u[k] * (u0 + u1) / 2 + v[k] * (v0 + v1) / 2
    return c, (u1 - u0, v1 - v0), u[k]


def _rect_points(c, wh, ax):
    ay = np.array([-ax[1], ax[0]])
    w, h = wh
    pts = np.array([c - ax * w / 2 - ay * h / 2, c + ax * w / 2 - ay * h / 2,
                    c + ax * w / 2 + ay * h / 2, c - ax * w / 2 + ay * h / 2])
    return order_quad(pts)


def order_quad(pts: np.ndarray) -> np.ndarray:
    """tl, tr, br, bl (same convention as PaddleOCR get_mini_boxes)."""
    p = pts[np.argsort(pts[:, 0], kind="stable")]
    left, right = p[:2], p[2:]
    tl, bl = (left[0], left[1]) if left[0, 1] <= left[1, 1] else (left[1], left[0])
    tr, br = (right[0], right[1]) if right[0, 1] <= right[1, 1] else (right[1], right[0])
    return np.array([tl, tr, br, bl])


def _quad_score(prob: np.ndarray, quad: np.ndarray) -> float:
    H, W = prob.shape
    x0, x1 = int(max(0, math.floor(quad[:, 0].min()))), int(min(W - 1, math.ceil(quad[:, 0].max())))
    y0, y1 = int(max(0, math.floor(quad[:, 1].min()))), int(min(H - 1, math.ceil(quad[:, 1].max())))
    if x1 < x0 or y1 < y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    inside = np.ones(yy.shape, bool)
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        cross = (q[0] - p[0]) * (yy - p[1]) - (q[1] - p[1]) * (xx - p[0])
        inside &= cross >= -0.5 * math.hypot(q[0] - p[0], q[1] - p[1])
    sub = prob[y0:y1 + 1, x0:x1 + 1]
    return float(sub[inside].mean()) if inside.any() else 0.0


def db_postprocess(prob: np.ndarray, src_w: int, src_h: int, thresh=0.2, box_thresh=0.45, unclip_ratio=1.4,
                   min_size=3, max_candidates=3000) -> list[np.ndarray]:
    """prob: (H,W) map of the det input size. Returns quads (4,2) in source-image pixel coords."""
    H, W = prob.shape
    mask = prob > thresh
    sr, sc, ec = _runs(mask)
    if len(sr) == 0:
        return []
    comp = _label_runs(sr, sc, ec, W)
    order = np.argsort(comp, kind="stable")
    comp_s = comp[order]
    bounds = np.flatnonzero(np.diff(comp_s)) + 1
    groups = np.split(order, bounds)
    sx, sy = src_w / W, src_h / H
    boxes = []
    for g in groups[:max_candidates]:
        npx = int((ec[g] - sc[g]).sum())
        if npx < 4:
            continue
        pts = np.concatenate([np.stack([sc[g], sr[g]], 1), np.stack([ec[g] - 1, sr[g]], 1)]).astype(np.float64)
        c, (w, h), ax = min_area_rect(pts)
        if min(w, h) < min_size:
            continue
        quad = _rect_points(c, (w, h), ax)
        if _quad_score(prob, quad) < box_thresh:
            continue
        d = w * h * unclip_ratio / (2 * (w + h))
        w2, h2 = w + 2 * d, h + 2 * d
        if min(w2, h2) < min_size + 2:
            continue
        q = _rect_points(c, (w2, h2), ax)
        q[:, 0] = np.clip(q[:, 0] * sx, 0, src_w)
        q[:, 1] = np.clip(q[:, 1] * sy, 0, src_h)
        boxes.append(q)
    return boxes


# ----------------------------------------------------------------------------------------------------------------
# det pre-processing, crops and rec pre/post-processing
# ----------------------------------------------------------------------------------------------------------------
DET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)          # applied to B,G,R (Paddle reads BGR)
DET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def det_size(w: int, h: int, max_side: int) -> tuple[int, int]:
    r = min(1.0, max_side / max(w, h))
    return max(32, int(round(w * r / 32)) * 32), max(32, int(round(h * r / 32)) * 32)


def det_preprocess(im: Image.Image, size: tuple[int, int], pad_to: tuple[int, int] | None = None) -> np.ndarray:
    """-> (1,3,H,W) float32 BGR normalised. If pad_to is given, the image is resized to `size` and zero-padded."""
    r = im.resize(size, Image.BILINEAR)
    a = np.asarray(r, dtype=np.float32)[:, :, ::-1]               # RGB -> BGR
    a = (a * (1 / 255.0) - DET_MEAN) / DET_STD
    a = np.ascontiguousarray(a.transpose(2, 0, 1))[None]
    if pad_to is not None and (pad_to[0] != size[0] or pad_to[1] != size[1]):
        out = np.zeros((1, 3, pad_to[1], pad_to[0]), np.float32)
        out[:, :, : size[1], : size[0]] = a
        return out
    return a


REC_H = 48


def crop_for_rec(im: Image.Image, quad: np.ndarray, rec_h: int = REC_H, max_w: int = 3200) -> np.ndarray:
    """Perspective-ish crop of the quad straight to height rec_h (crop+resize in one PIL transform).
    Returns HxWx3 uint8 RGB."""
    tl, tr, br, bl = quad
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    w, h = max(2.0, w), max(2.0, h)
    rotate = h / w >= 1.5
    if rotate:                                                    # vertical text: sample so it comes out horizontal
        tl, tr, br, bl = bl, tl, tr, br
        w, h = h, w
    out_w = int(min(max_w, max(1, math.ceil(rec_h * w / h))))
    data = tuple(float(v) for v in (*tl, *bl, *br, *tr))
    crop = im.transform((out_w, rec_h), Image.QUAD, data, Image.BILINEAR)
    return np.asarray(crop)


def rec_batch(crops: list[np.ndarray], min_w: int = 320, align: int = 1, fixed_w: int | None = None) -> np.ndarray:
    """Stack HxWx3 uint8 RGB crops -> (B,3,48,Wmax) float32 BGR in [-1,1], right-padded with 0 (Paddle style)."""
    W = fixed_w or max(min_w, max(c.shape[1] for c in crops))
    if align > 1:
        W = int(math.ceil(W / align) * align)
    out = np.zeros((len(crops), 3, REC_H, W), np.float32)
    for i, c in enumerate(crops):
        a = c[:, :W, ::-1].astype(np.float32) * (2 / 255.0) - 1.0
        out[i, :, :, : a.shape[1]] = a.transpose(2, 0, 1)
    return out


_CHARS: list[str] | None = None


def charset() -> list[str]:
    global _CHARS
    if _CHARS is None:
        _CHARS = json.loads((REC_DIR / "preprocessor_config.json").read_text(encoding="utf-8"))["character_list"]
    return _CHARS


def ctc_decode(idx: np.ndarray, prob: np.ndarray) -> list[tuple[str, float]]:
    """idx/prob: (B,T) argmax and max-prob."""
    chars = charset()
    res = []
    for i in range(idx.shape[0]):
        s = idx[i]
        keep = np.ones(len(s), bool)
        keep[1:] = s[1:] != s[:-1]
        keep &= s != 0
        sel = s[keep]
        res.append(("".join(chars[j] for j in sel), float(prob[i][keep].mean()) if keep.any() else 0.0))
    return res


# ----------------------------------------------------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------------------------------------------------
class Timer:
    def __init__(self):
        self.t: dict[str, float] = {}

    @contextmanager
    def __call__(self, key: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[key] = self.t.get(key, 0.0) + time.perf_counter() - t0


def save_json(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    return p
