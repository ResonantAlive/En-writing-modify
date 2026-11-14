"""PP-OCRv6 medium engine: thin wrapper over fast_ocr.FastPPOCR (OpenVINO CPU FP32 by default, torch optional).

Quality rules (see ocr_bench/README.md): detection runs at the model's own preprocessing size (no downscaling),
FP32 only, recognition one line at a time. Line grouping / layout restoration is done by app.ocr.layout.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from . import layout
from .common import Line, OcrError, PageResult

log = logging.getLogger("wm.ocr.paddle")

DET_DIR = "PP-OCRv6_medium_det_safetensors"
REC_DIR = "PP-OCRv6_medium_rec_safetensors"
NAME = "paddle"


class PaddleEngine:
    name = NAME

    def __init__(self, root: Path, *, backend: str = "openvino", ir_dir: Path | None = None, threads: int = 0):
        self.root = Path(root)
        self.backend = backend
        self.ir_dir = Path(ir_dir) if ir_dir else self.root / "data" / "ov_models"
        self.threads = int(threads)
        self.ocr = None
        self._load_lock = threading.Lock()
        self.versions: dict = {}

    def load(self) -> None:
        with self._load_lock:
            if self.ocr is not None:
                return
            from .fast_ocr import FastPPOCR
            # first use with the openvino backend exports the IR once (needs torch + transformers, ~30-45 s)
            self.ocr = FastPPOCR(model_root=self.root, backend=self.backend, ir_dir=self.ir_dir, threads=self.threads)
            self.versions = _versions(self.backend)

    def warmup(self) -> None:
        self.load()
        self.ocr.warmup()

    def info(self) -> dict:
        return {"local_engine": NAME, "name": "PP-OCRv6_medium (det+rec)",
                "version": self.versions,
                "params": {"backend": self.backend, "precision": "fp32", "det_resize": "model default (no downscale)",
                           "rec_batch": 1, "threads": self.threads or "all"}}

    def recognize(self, im) -> PageResult:
        self.load()
        try:
            boxes = self.ocr.recognize_boxes(im)
        except (RuntimeError, MemoryError, ValueError, OSError) as e:
            raise OcrError(f"paddle recognition failed: {type(e).__name__}") from e
        raw = [Line(b.text, list(b.box), b.score, getattr(b, "rect", None)) for b in boxes]
        lines, skew = layout.group_boxes(raw, im.width, im.height)
        return PageResult(lines=lines, width=im.width, height=im.height, layout_reliable=True, boxes=raw,
                          skew_deg=skew)


def _versions(backend: str) -> dict:
    out = {"backend": backend}
    try:
        if backend == "openvino":
            import openvino
            out["openvino"] = openvino.__version__
        else:
            import torch
            import transformers
            out["torch"], out["transformers"] = torch.__version__, transformers.__version__
    except Exception:
        pass
    return out
