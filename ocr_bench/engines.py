"""Model back-ends for PP-OCRv6 det/rec: plain torch, torch-jit-frozen, OpenVINO (CPU / GPU / NPU).

All back-ends expose  det(x[1,3,H,W]) -> prob[H,W]   and   rec(x[B,3,48,W]) -> (maxprob[B,T], argmax[B,T]).
The argmax is folded into the rec graph so the 18710-class softmax never leaves the device.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

import common as C


class DetWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(pixel_values=x).last_hidden_state[:, 0]


class RecWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        p = self.m(pixel_values=x).last_hidden_state
        prob, idx = p.max(dim=-1)
        return prob, idx.to(torch.int32)


def load_torch(kind: str) -> nn.Module:
    from transformers import PPOCRV6MediumDetForObjectDetection, PPOCRV6SmallRecForTextRecognition
    if kind == "det":
        return DetWrap(PPOCRV6MediumDetForObjectDetection.from_pretrained(str(C.DET_DIR)).eval()).eval()
    return RecWrap(PPOCRV6SmallRecForTextRecognition.from_pretrained(str(C.REC_DIR)).eval()).eval()


EXAMPLE = {"det": (1, 3, 1280, 896), "rec": (4, 3, 48, 960)}


# ----------------------------------------------------------------------------------------------------------------
class TorchBackend:
    def __init__(self, kind: str, mode: str = "eager", channels_last: bool = False):
        self.kind = kind
        m = load_torch(kind)
        self.cl = channels_last
        if channels_last:
            m = m.to(memory_format=torch.channels_last)
        if mode == "jit":
            ex = torch.zeros(EXAMPLE[kind])
            with torch.inference_mode(False), torch.no_grad():
                tm = torch.jit.trace(m, ex, check_trace=False, strict=False)
                tm = torch.jit.optimize_for_inference(torch.jit.freeze(tm.eval()))
            m = tm
        self.m = m

    def __call__(self, x: np.ndarray):
        t = torch.from_numpy(x)
        if self.cl:
            t = t.contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            out = self.m(t)
        if self.kind == "det":
            return out[0].numpy()
        return out[0].numpy(), out[1].numpy()


# ----------------------------------------------------------------------------------------------------------------
def ir_path(kind: str) -> Path:
    return C.CACHE / f"{kind}.xml"


def export_ir(kind: str, force: bool = False) -> Path:
    """Convert the torch model to OpenVINO IR once (dynamic H/W for det, dynamic B/W for rec); weights kept FP32."""
    import openvino as ov
    p = ir_path(kind)
    if p.exists() and not force:
        return p
    C.CACHE.mkdir(parents=True, exist_ok=True)
    m = load_torch(kind)
    ex = torch.zeros(EXAMPLE[kind])
    with torch.no_grad():
        om = ov.convert_model(m, example_input=ex)
    if kind == "det":
        om.reshape({om.inputs[0]: ov.PartialShape([1, 3, -1, -1])})
    else:
        om.reshape({om.inputs[0]: ov.PartialShape([-1, 3, 48, -1])})
    ov.save_model(om, str(p), compress_to_fp16=False)          # keep exact FP32 weights (no quality change)
    return p


_CORE = None


def core():
    global _CORE
    if _CORE is None:
        import openvino as ov
        _CORE = ov.Core()
        C.CACHE.mkdir(parents=True, exist_ok=True)
        _CORE.set_property({"CACHE_DIR": str(C.CACHE / "ov_cache")})
    return _CORE


class OVBackend:
    """device: CPU / GPU / NPU. static: fixed input shape (required for NPU, often faster on GPU).
    hint: LATENCY / THROUGHPUT. precision: None (device default) / f32 / f16 / bf16."""

    def __init__(self, kind: str, device: str = "CPU", static: tuple | None = None, hint: str = "LATENCY",
                 precision: str | None = None, threads: int = 0, streams: int | None = None):
        import openvino as ov
        self.kind = kind
        model = core().read_model(str(export_ir(kind)))
        if static is not None:
            model.reshape({model.inputs[0]: ov.PartialShape(list(static))})
        cfg = {"PERFORMANCE_HINT": hint}
        if precision:
            cfg["INFERENCE_PRECISION_HINT"] = precision
        if threads and device == "CPU":
            cfg["INFERENCE_NUM_THREADS"] = threads
        if streams is not None:
            cfg["NUM_STREAMS"] = streams
        self.static = static
        self.device = device
        self.cm = core().compile_model(model, device, cfg)
        self.req = self.cm.create_infer_request()

    def __call__(self, x: np.ndarray):
        self.req.infer({0: x})
        if self.kind == "det":
            return self.req.get_output_tensor(0).data[0].copy()
        return self.req.get_output_tensor(0).data.copy(), self.req.get_output_tensor(1).data.copy()

    # async helpers (several requests in flight, used for rec throughput)
    def infer_many(self, xs: list[np.ndarray], n_req: int = 2):
        import openvino as ov
        q = ov.AsyncInferQueue(self.cm, n_req)
        out = [None] * len(xs)

        def cb(req, i):
            out[i] = (req.get_output_tensor(0).data.copy(), req.get_output_tensor(1).data.copy())
        q.set_callback(cb)
        for i, x in enumerate(xs):
            q.start_async({0: x}, i)
        q.wait_all()
        return out
