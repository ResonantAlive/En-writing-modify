"""Baseline: PP-OCRv6 medium det+rec through transformers' own processors/models (same pipeline as app/ocr/paddle_engine.py,
copied here so the benchmark folder is self-contained).

  python bench_hf_baseline.py --det-max-side 1280      # what the app does today
  python bench_hf_baseline.py --det-max-side 0         # processor default (full resolution)
"""
from __future__ import annotations

import argparse
import math
import statistics
import time
from collections import deque

import numpy as np
import torch
from PIL import Image

import common as C


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    comps = []
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        q = deque([(y0, x0)])
        pts = []
        while q:
            y, x = q.popleft()
            pts.append((y, x))
            if y > 0 and mask[y - 1, x] and not seen[y - 1, x]:
                seen[y - 1, x] = True; q.append((y - 1, x))
            if y + 1 < h and mask[y + 1, x] and not seen[y + 1, x]:
                seen[y + 1, x] = True; q.append((y + 1, x))
            if x > 0 and mask[y, x - 1] and not seen[y, x - 1]:
                seen[y, x - 1] = True; q.append((y, x - 1))
            if x + 1 < w and mask[y, x + 1] and not seen[y, x + 1]:
                seen[y, x + 1] = True; q.append((y, x + 1))
        comps.append(np.asarray(pts, dtype=np.int32))
    return comps


def db_boxes(prob, thresh=0.2, box_thresh=0.45, unclip_ratio=1.4, min_size=3, max_candidates=3000, stride=2):
    h, w = prob.shape
    hs, ws = h // stride, w // stride
    small = prob[: hs * stride, : ws * stride].reshape(hs, stride, ws, stride).max(axis=(1, 3))
    mask = small > thresh
    out = []
    for pts in _components(mask)[:max_candidates]:
        if len(pts) < 4:
            continue
        ys = pts[:, 0] * stride + stride / 2
        xs = pts[:, 1] * stride + stride / 2
        score = float(small[pts[:, 0], pts[:, 1]].mean())
        if score < box_thresh:
            continue
        P = np.stack([xs, ys], 1).astype(np.float64)
        c = P.mean(0)
        evals, evecs = np.linalg.eigh(np.cov((P - c).T))
        ax = evecs[:, 1]
        if ax[0] < 0:
            ax = -ax
        ang = math.atan2(ax[1], ax[0])
        if abs(math.degrees(ang)) > 30 or evals[1] < 1.2 * evals[0]:
            ax = np.array([1.0, 0.0])
        ay = np.array([-ax[1], ax[0]])
        u, v = (P - c) @ ax, (P - c) @ ay
        u0, u1, v0, v1 = u.min() - stride / 2, u.max() + stride / 2, v.min() - stride / 2, v.max() + stride / 2
        bw, bh = u1 - u0, v1 - v0
        if min(bw, bh) < min_size:
            continue
        d = bw * bh * unclip_ratio / (2 * (bw + bh))
        u0 -= d; u1 += d; v0 -= d; v1 += d
        out.append(np.array([c + ax * u0 + ay * v0, c + ax * u1 + ay * v0, c + ax * u1 + ay * v1, c + ax * u0 + ay * v1]))
    return out


def crop_quad(im, quad):
    tl, tr, br, bl = quad
    w = max(2, int(round(np.linalg.norm(tr - tl))))
    h = max(2, int(round(np.linalg.norm(bl - tl))))
    crop = im.transform((w, h), Image.QUAD, tuple(float(v) for v in (*tl, *bl, *br, *tr)), Image.BICUBIC)
    if h >= 1.5 * w:
        crop = crop.rotate(90, expand=True)
    return crop


class HFPipeline:
    def __init__(self, det_max_side=1280, rec_batch=16):
        from transformers import (PPOCRV5ServerDetImageProcessor, PPOCRV6MediumDetForObjectDetection,
                                  PPOCRV6SmallRecForTextRecognition, PPOCRV6SmallRecImageProcessor)
        self.det = PPOCRV6MediumDetForObjectDetection.from_pretrained(str(C.DET_DIR)).eval()
        self.det_proc = PPOCRV5ServerDetImageProcessor.from_pretrained(str(C.DET_DIR))
        self.rec = PPOCRV6SmallRecForTextRecognition.from_pretrained(str(C.REC_DIR)).eval()
        self.rec_proc = PPOCRV6SmallRecImageProcessor.from_pretrained(str(C.REC_DIR))
        self.det_max_side, self.rec_batch = det_max_side, rec_batch

    @torch.no_grad()
    def __call__(self, im, T: C.Timer):
        W, H = im.size
        with T("det_pre"):
            dim = im
            if self.det_max_side and max(W, H) > self.det_max_side:
                k = self.det_max_side / max(W, H)
                dim = im.resize((max(32, round(W * k)), max(32, round(H * k))), Image.BILINEAR)
            inp = self.det_proc(images=dim, return_tensors="pt")
        with T("det_infer"):
            prob = self.det(pixel_values=inp["pixel_values"]).last_hidden_state[0, 0].numpy()
        with T("det_post"):
            ph, pw = prob.shape
            boxes = db_boxes(prob)
        with T("crop"):
            sx, sy = W / pw, H / ph
            crops, meta = [], []
            for quad in boxes:
                q = quad * np.array([sx, sy])
                crops.append(crop_quad(im, q))
                meta.append([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()])
        texts = [""] * len(crops)
        order = sorted(range(len(crops)), key=lambda i: crops[i].size[0] / max(1, crops[i].size[1]))
        for k in range(0, len(order), self.rec_batch):
            idx = order[k:k + self.rec_batch]
            with T("rec_pre"):
                inp = self.rec_proc(images=[crops[i] for i in idx], return_tensors="pt")
            with T("rec_infer"):
                out = self.rec(pixel_values=inp["pixel_values"])
            with T("rec_post"):
                for i, d in zip(idx, self.rec_proc.post_process_text_recognition(out)):
                    texts[i] = d["text"]
        return C.order_lines(list(zip(meta, texts))), {"det_shape": list(inp_shape(inp)), "n_boxes": len(boxes)}


def inp_shape(inp):
    return tuple(inp["pixel_values"].shape) if "pixel_values" in inp else ()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-max-side", type=int, default=1280)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.perf_counter()
    pipe = HFPipeline(det_max_side=a.det_max_side)
    load_s = time.perf_counter() - t0
    cases = C.load_cases()
    pages = [(c, i, im) for c in cases for i, im in enumerate(c["images"])]
    # warm-up on one page
    pipe(pages[0][2], C.Timer())
    per_run, stage_tot, last = [], {}, {}
    for r in range(a.runs):
        tt = 0.0
        for c, i, im in pages:
            T = C.Timer()
            t = time.perf_counter()
            lines, info = pipe(im, T)
            dt = time.perf_counter() - t
            tt += dt
            for k, v in T.t.items():
                stage_tot[k] = stage_tot.get(k, 0) + v
            last[(c["name"], i)] = lines
        per_run.append(tt / len(pages))
    scores = []
    for c in cases:
        text = C.join_lines([l for i in range(len(c["images"])) for l in last[(c["name"], i)]])
        scores.append({"case": c["name"], **C.score_text(text, c["gt"], c["expected"]), "text": text})
    summ = C.summarize(scores)
    n = a.runs * len(pages)
    res = {"name": f"hf_baseline_det{a.det_max_side}{a.tag}", "load_s": load_s, "sec_per_page": statistics.median(per_run),
           "runs": per_run, "stages_per_page": {k: v / n for k, v in stage_tot.items()}, "accuracy": summ,
           "cases": scores, "threads": torch.get_num_threads()}
    C.save_json(res["name"] + ".json", res)
    print(f"{res['name']}: {res['sec_per_page']:.3f} s/page  load {load_s:.1f}s  "
          f"WER {summ['wer']:.4f} CER {summ['cer']:.4f} typos {summ['typo_hit']}/{summ['typo_total']}")
    print("  stages/page:", {k: round(v, 3) for k, v in res["stages_per_page"].items()})


if __name__ == "__main__":
    main()
