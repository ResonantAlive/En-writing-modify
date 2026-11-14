"""CPU-only kernel efficiency: achieved GFLOPS / MFU per back-end, thread count and batch size (FP32 only).

  python cpu_mfu_bench.py torch-eager torch-cl torch-jit ov-lat ov-tp ...  [--threads 4,8]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import common as C
import engines as E
import mfu

DET_GF_PER_PX = 2715.5 / (2976 * 2112)          # GFLOPs per det input pixel (linear in H*W, measured in mfu.py)
REC_GF_PER_COL = 36.4 / 1024                    # GFLOPs per rec input column (linear in W)

DET_SHAPES = [(1, 3, 2976, 2112), (1, 3, 1696, 1280)]
REC_SHAPES = [(1, 3, 48, 1024), (6, 3, 48, 1024), (12, 3, 48, 1024), (28, 3, 48, 1024)]


def gflops(kind, shape):
    if kind == "det":
        return DET_GF_PER_PX * shape[2] * shape[3]
    return REC_GF_PER_COL * shape[0] * shape[3]


def bench(fn, x, n=3):
    fn(x)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn(x)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def make(name, kind, threads):
    if name.startswith("torch"):
        torch.set_num_threads(threads)
        return {"torch-eager": lambda: E.TorchBackend(kind),
                "torch-cl": lambda: E.TorchBackend(kind, channels_last=True),
                "torch-jit": lambda: E.TorchBackend(kind, mode="jit"),
                "torch-jit-cl": lambda: E.TorchBackend(kind, mode="jit", channels_last=True)}[name]()
    hint = "THROUGHPUT" if name == "ov-tp" else "LATENCY"
    return E.OVBackend(kind, "CPU", hint=hint, threads=threads, precision="f32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--threads", default="8")
    ap.add_argument("--kinds", default="det,rec")
    a = ap.parse_args()
    peak = max(mfu.gemm_peak(n=6144, iters=5) for _ in range(2))
    print(f"sgemm peak now: {peak:.0f} GFLOPS", flush=True)
    rows = []
    for name in a.names:
        for th in [int(t) for t in a.threads.split(",")]:
            for kind in a.kinds.split(","):
                be = make(name, kind, th)
                for shape in (DET_SHAPES if kind == "det" else REC_SHAPES):
                    x = np.random.rand(*shape).astype(np.float32)
                    t = bench(be, x)
                    g = gflops(kind, shape) / t
                    rows.append({"name": name, "threads": th, "kind": kind, "shape": shape, "ms": t * 1000,
                                 "gflops": g, "mfu": g / peak})
                    print(f"{name:13s} t={th} {kind} {str(shape):20s} {t*1000:8.0f} ms  {g:6.0f} GFLOPS  "
                          f"MFU {g/peak:5.1%}", flush=True)
                del be
    C.OUT.mkdir(exist_ok=True)
    p = C.OUT / "cpu_mfu.jsonl"
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, "peak": peak}) + "\n")


if __name__ == "__main__":
    main()
