"""Optimised PP-OCRv6 medium det+rec pipeline, CPU / FP32 only, no resolution reduction.

Pre-processing follows the model's own config: det limit_type=min 736 / max_side 4000, rounded to /32
(for these A4 scans that is full resolution, 2112x2976). Crops are rectified (bicubic) at native resolution and
resized to height 48 keeping aspect ratio (bilinear, no antialias = cv2.INTER_LINEAR), right-padded like PaddleOCR.

What is optimised is only how well the hardware is used:
  * kernel back-end (torch eager / torch jit-frozen / OpenVINO CPU)
  * thread count & scheduling (hybrid P/E cores)
  * rec batching (sort by width -> little padding, batch size tuned for FLOP efficiency)
  * numpy run-length DB post-processing instead of per-pixel Python BFS
  * page-level parallelism (several OpenVINO streams, each working on its own page)

  python ppocr_cpu.py --backend ov --workers 1 --rec-bs 8
"""
from __future__ import annotations

import argparse
import math
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

import common as C
import engines as E


def official_det_size(w: int, h: int, limit=736, max_side=4000) -> tuple[int, int]:
    r = limit / min(w, h) if min(w, h) < limit else 1.0
    rw, rh = int(w * r), int(h * r)
    if max(rw, rh) > max_side:
        k = max_side / max(rw, rh)
        rw, rh = int(rw * k), int(rh * k)
    return max(32, int(round(rw / 32) * 32)), max(32, int(round(rh / 32) * 32))


def rectify(im: Image.Image, quad: np.ndarray) -> Image.Image:
    """PaddleOCR get_rotate_crop_image: crop the quad to an upright image at native resolution (bicubic)."""
    tl, tr, br, bl = quad
    w = int(max(np.linalg.norm(tl - tr), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(tl - bl), np.linalg.norm(tr - br)))
    w, h = max(1, w), max(1, h)
    crop = im.transform((w, h), Image.QUAD, tuple(float(v) for v in (*tl, *bl, *br, *tr)), Image.BICUBIC)
    if h / w >= 1.5:
        crop = crop.transpose(Image.ROTATE_90)
    return crop


def resize_h48(crop: Image.Image, max_w: int = 3200) -> np.ndarray:
    """Height 48 keeping ratio, bilinear without antialias (matches cv2.INTER_LINEAR / transformers antialias=False)."""
    w, h = crop.size
    nw = int(min(max_w, max(1, math.ceil(48 * w / h))))
    sx, sy = w / nw, h / 48
    r = crop.transform((nw, 48), Image.AFFINE, (sx, 0, 0, 0, sy, 0), Image.BILINEAR)
    return np.asarray(r)


class Pipeline:
    def __init__(self, det, rec, rec_bs: int = 8, det_size_fn=official_det_size):
        self.det, self.rec, self.rec_bs, self.det_size_fn = det, rec, rec_bs, det_size_fn

    def det_stage(self, im: Image.Image, T: C.Timer):
        W, H = im.size
        with T("det_pre"):
            size = self.det_size_fn(W, H)
            x = C.det_preprocess(im, size)
        with T("det_infer"):
            prob = self.det(x)
        with T("det_post"):
            quads = C.db_postprocess(prob, W, H)
        with T("crop"):
            crops = [resize_h48(rectify(im, q)) for q in quads]
        boxes = [[float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]
                 for q in quads]
        return boxes, crops, size

    def rec_stage(self, crops: list[np.ndarray], T: C.Timer) -> list[str]:
        texts = [""] * len(crops)
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1])
        bs = self.rec_bs or max(1, len(crops))                # 0 = all lines of the page in one batch
        for k in range(0, len(order), bs):
            idx = order[k:k + bs]
            with T("rec_pre"):
                x = C.rec_batch([crops[i] for i in idx])
            with T("rec_infer"):
                prob, am = self.rec(x)
            with T("rec_post"):
                for i, (t, _s) in zip(idx, C.ctc_decode(am, prob)):
                    texts[i] = t
        return texts

    def page(self, im: Image.Image, T: C.Timer | None = None) -> list[str]:
        T = T or C.Timer()
        boxes, crops, _ = self.det_stage(im, T)
        texts = self.rec_stage(crops, T)
        return C.order_lines(list(zip(boxes, texts)))


def build(backend: str, threads: int = 0, rec_bs: int = 8, hint: str = "LATENCY", streams: int | None = None,
          core_type: str | None = None):
    if backend.startswith("torch"):
        import torch
        if threads:
            torch.set_num_threads(threads)
        if backend == "torch-best":                           # best torch combo from cpu_mfu_bench
            det, rec = E.TorchBackend("det", channels_last=True), E.TorchBackend("rec", mode="jit")
        else:
            mode = "jit" if backend == "torch-jit" else "eager"
            det, rec = E.TorchBackend("det", mode=mode), E.TorchBackend("rec", mode=mode)
    else:
        kw = dict(hint=hint, threads=threads, precision="f32", streams=streams)
        det, rec = E.OVBackend("det", "CPU", **kw), E.OVBackend("rec", "CPU", **kw)
    return Pipeline(det, rec, rec_bs=rec_bs)


class MultiStream:
    """N workers, each owns a det+rec infer request on the same compiled models (OpenVINO streams)."""

    def __init__(self, base: Pipeline, workers: int):
        self.base, self.workers = base, workers
        self.local = threading.local()

    def _pipe(self):
        p = getattr(self.local, "p", None)
        if p is None:
            b = self.base
            det = _ReqWrap(b.det, "det")
            rec = _ReqWrap(b.rec, "rec")
            p = self.local.p = Pipeline(det, rec, rec_bs=b.rec_bs, det_size_fn=b.det_size_fn)
        return p

    def run(self, ims: list[Image.Image]):
        def one(im):
            T = C.Timer()
            return self._pipe().page(im, T), T
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(one, ims))


class _ReqWrap:
    def __init__(self, ovb: E.OVBackend, kind: str):
        self.req = ovb.cm.create_infer_request()
        self.kind = kind

    def __call__(self, x):
        self.req.infer({0: x})
        if self.kind == "det":
            return self.req.get_output_tensor(0).data[0].copy()
        return self.req.get_output_tensor(0).data.copy(), self.req.get_output_tensor(1).data.copy()


def evaluate(results: dict, cases) -> tuple[dict, list]:
    scores = []
    for c in cases:
        text = C.join_lines([l for i in range(len(c["images"])) for l in results[(c["name"], i)]])
        scores.append({"case": c["name"], **C.score_text(text, c["gt"], c["expected"]), "text": text})
    return C.summarize(scores), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ov", choices=["ov", "torch", "torch-jit", "torch-best"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--rec-bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1, help=">1: pages in parallel (OpenVINO streams)")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    t0 = time.perf_counter()
    if a.workers > 1:
        assert a.backend == "ov"
        base = build("ov", threads=a.threads, rec_bs=a.rec_bs, hint="THROUGHPUT", streams=a.workers)
        runner = MultiStream(base, a.workers)
    else:
        base = build(a.backend, threads=a.threads, rec_bs=a.rec_bs)
        runner = None
    load_s = time.perf_counter() - t0
    cases = C.load_cases()
    pages = [(c["name"], i, im) for c in cases for i, im in enumerate(c["images"])]

    seen = set()
    for _, _, im in pages:                                    # warm-up once per distinct page size
        if im.size not in seen:
            seen.add(im.size)
            base.page(im)
    per_run, stage_tot, out = [], {}, {}
    for _ in range(a.runs):
        t = time.perf_counter()
        if runner is None:
            for name, i, im in pages:
                T = C.Timer()
                out[(name, i)] = base.page(im, T)
                for k, v in T.t.items():
                    stage_tot[k] = stage_tot.get(k, 0) + v
        else:
            res = runner.run([im for _, _, im in pages])
            for (name, i, _), (lines, T) in zip(pages, res):
                out[(name, i)] = lines
                for k, v in T.t.items():
                    stage_tot[k] = stage_tot.get(k, 0) + v
        per_run.append((time.perf_counter() - t) / len(pages))
    summ, scores = evaluate(out, cases)
    n = a.runs * len(pages)
    name = f"cpu_{a.backend}_w{a.workers}_t{a.threads or 'auto'}_bs{a.rec_bs}{a.tag}"
    res = {"name": name, "load_s": load_s, "sec_per_page": statistics.median(per_run), "runs": per_run,
           "stages_per_page_sum_over_workers": {k: v / n for k, v in stage_tot.items()}, "accuracy": summ,
           "cases": scores, "lines": {f"{k[0]}/{k[1]}": v for k, v in out.items()}}
    C.save_json(name + ".json", res)
    print(f"{name}: {res['sec_per_page']:.3f} s/page (wall, 8 pages)  load {load_s:.1f}s  WER {summ['wer']:.4f} "
          f"CER {summ['cer']:.4f} typos {summ['typo_hit']}/{summ['typo_total']}")
    print("  stages/page:", {k: round(v, 3) for k, v in res["stages_per_page_sum_over_workers"].items()})


if __name__ == "__main__":
    main()
