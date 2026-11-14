"""Check that every optimised back-end gives the same numbers as plain torch FP32 on real pages
(det prob map diff, box count, rec argmax sequences / decoded strings).

  python verify_equiv.py torch-cl torch-jit ov
"""
from __future__ import annotations

import sys

import numpy as np

import common as C
import engines as E
import ppocr_cpu as P


def backends(name):
    if name == "torch-cl":
        return E.TorchBackend("det", channels_last=True), E.TorchBackend("rec", channels_last=True)
    if name == "torch-jit":
        return E.TorchBackend("det", mode="jit"), E.TorchBackend("rec", mode="jit")
    if name == "ov":
        return E.OVBackend("det", "CPU", precision="f32"), E.OVBackend("rec", "CPU", precision="f32")
    raise ValueError(name)


def main():
    names = sys.argv[1:] or ["torch-cl", "torch-jit", "ov"]
    cases = C.load_cases()
    pages = [cases[0]["images"][0], cases[3]["images"][0]]          # photo (1280x1696) + A4 scan (2112x2976)
    ref_det, ref_rec = E.TorchBackend("det"), E.TorchBackend("rec")
    ref = []
    for im in pages:
        size = P.official_det_size(*im.size)
        x = C.det_preprocess(im, size)
        prob = ref_det(x)
        quads = C.db_postprocess(prob, *im.size)
        crops = [P.resize_h48(P.rectify(im, q)) for q in quads]
        crops.sort(key=lambda c: c.shape[1])
        xr = C.rec_batch(crops)
        rp, ri = ref_rec(xr)
        ref.append((x, prob, len(quads), xr, rp, ri))
    for name in names:
        det, rec = backends(name)
        for k, (x, prob, nq, xr, rp, ri) in enumerate(ref):
            p2 = det(x)
            nq2 = len(C.db_postprocess(p2, *pages[k].size))
            p_r, i_r = rec(xr)
            same_seq = int((i_r == ri).all(axis=1).sum())
            t1 = [t for t, _ in C.ctc_decode(ri, rp)]
            t2 = [t for t, _ in C.ctc_decode(i_r, p_r)]
            print(f"{name:10s} page{k}: det max|dprob| {np.abs(p2 - prob).max():.2e}  boxes {nq2} vs {nq}  "
                  f"rec argmax identical rows {same_seq}/{len(ri)}  strings identical "
                  f"{sum(a == b for a, b in zip(t1, t2))}/{len(t1)}  max|dprob| {np.abs(p_r - rp).max():.2e}",
                  flush=True)


if __name__ == "__main__":
    main()
