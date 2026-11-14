"""CPU peak FLOPs (sgemm) + model FLOPs per input shape, used to turn latencies into MFU.

  python mfu.py            -> prints peak GFLOPS and per-shape GFLOPs, writes results/flops.json
"""
from __future__ import annotations

import time

import torch
from torch.utils.flop_counter import FlopCounterMode

import common as C
import engines as E


def gemm_peak(n=4096, iters=10, threads=None):
    if threads:
        torch.set_num_threads(threads)
    a, b = torch.randn(n, n), torch.randn(n, n)
    for _ in range(3):
        a @ b
    t = time.perf_counter()
    for _ in range(iters):
        a @ b
    dt = (time.perf_counter() - t) / iters
    return 2 * n ** 3 / dt / 1e9


def model_flops(kind, shape):
    m = E.load_torch(kind)
    x = torch.zeros(shape)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x)
    return fc.get_total_flops()


if __name__ == "__main__":
    nt = torch.get_num_threads()
    peak = max(gemm_peak() for _ in range(3))
    peak1 = gemm_peak(threads=1)
    torch.set_num_threads(nt)
    print(f"sgemm peak: {peak:.1f} GFLOPS ({nt} threads), {peak1:.1f} GFLOPS (1 thread)")
    res = {"peak_gflops": peak, "peak_gflops_1t": peak1, "threads": nt, "flops": {}}
    for kind, shape in [("det", (1, 3, 2976, 2112)), ("det", (1, 3, 1696, 1280)), ("rec", (1, 3, 48, 1024)),
                        ("rec", (1, 3, 48, 320))]:
        f = model_flops(kind, shape)
        res["flops"][f"{kind}{list(shape)}"] = f
        print(f"{kind} {shape}: {f/1e9:.1f} GFLOPs")
    C.save_json("flops.json", res)
