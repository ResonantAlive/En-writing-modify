"""Copied from ocr_bench/fast_ocr.py (the benchmarked version); only the default paths differ.

PP-OCRv6 medium 快速推理：CPU / FP32 / OpenVINO（推荐方案的单文件版本，可直接拷走使用）。

不缩图、不量化：检测按模型自带配置预处理（短边 >=736、长边 <=4000、取整到 32），识别每次一行。
运行时只依赖 numpy、Pillow、openvino；torch + transformers 只在第一次导出模型时需要
（backend="torch" 时运行也需要）。

    from fast_ocr import FastPPOCR
    ocr = FastPPOCR()                          # 第一次会导出 OpenVINO 模型到 ./cache（约 45 秒）
    text = ocr.recognize_text("page_1.png")    # 整页文字（行尾连字符已拼回）
    lines = ocr.recognize("page_1.png")        # [Line(text, box, score), ...] 按阅读顺序

命令行：
    python fast_ocr.py page_1.png page_2.png            # 多页按顺序拼成一篇
    python fast_ocr.py page_1.png --lines --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = HERE.parent.parent           # writing_modify/：放着两个 PP-OCRv6 模型目录
DET_DIRNAME = "PP-OCRv6_medium_det_safetensors"
REC_DIRNAME = "PP-OCRv6_medium_rec_safetensors"
DEFAULT_IR_DIR = DEFAULT_MODEL_ROOT / "data" / "ov_models"   # 导出的 OpenVINO 模型（data/ 已 gitignore）

REC_H = 48
DET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # 作用在 B,G,R 上（Paddle 按 BGR 读图）
DET_STD = np.array([0.229, 0.224, 0.225], np.float32)
HYPHENS = "-‐‑–—一－_~"


@dataclass
class Line:
    text: str
    box: tuple[float, float, float, float]       # x0, y0, x1, y1，原图像素坐标
    score: float                                  # 字符平均置信度
    rect: tuple | None = None                     # 旋转矩形 (cx, cy, w, h, angle°)，w 沿文字方向；版面纠斜用


# =====================================================================================================
# 模型导出（只需一次；需要 torch + transformers）
# =====================================================================================================
def export_openvino(model_root: Path | str = DEFAULT_MODEL_ROOT, ir_dir: Path | str = DEFAULT_IR_DIR,
                    force: bool = False) -> Path:
    """把 transformers 的 PP-OCRv6 det/rec 转成 OpenVINO IR。权重保持 FP32（compress_to_fp16=False）。"""
    model_root, ir_dir = Path(model_root), Path(ir_dir)
    if not force and (ir_dir / "det.xml").exists() and (ir_dir / "rec.xml").exists():
        return ir_dir
    import openvino as ov
    import torch
    from transformers import PPOCRV6MediumDetForObjectDetection, PPOCRV6SmallRecForTextRecognition

    class Det(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):                     # -> 概率图 (1,H,W)
            return self.m(pixel_values=x).last_hidden_state[:, 0]

    class Rec(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):                     # -> (最大概率, 字符 id)，argmax 放进图里
            p = self.m(pixel_values=x).last_hidden_state
            prob, idx = p.max(dim=-1)
            return prob, idx.to(torch.int32)

    ir_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("det", Det(PPOCRV6MediumDetForObjectDetection.from_pretrained(str(model_root / DET_DIRNAME)).eval()),
         (1, 3, 1280, 896), [1, 3, -1, -1]),
        ("rec", Rec(PPOCRV6SmallRecForTextRecognition.from_pretrained(str(model_root / REC_DIRNAME)).eval()),
         (4, 3, 48, 960), [-1, 3, 48, -1]),
    ]
    for name, module, example, dyn in jobs:
        with torch.no_grad():
            om = ov.convert_model(module.eval(), example_input=torch.zeros(example))
        om.reshape({om.inputs[0]: ov.PartialShape(dyn)})
        ov.save_model(om, str(ir_dir / f"{name}.xml"), compress_to_fp16=False)
    return ir_dir


# =====================================================================================================
# 推理后端
# =====================================================================================================
class _OpenVINORunner:
    def __init__(self, ir_dir: Path, threads: int = 0):
        import openvino as ov
        core = ov.Core()
        core.set_property({"CACHE_DIR": str(ir_dir / "ov_cache")})
        # 显式 f32：ARM（含 Apple Silicon）上 OpenVINO CPU 默认会用 f16 计算
        cfg = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"}
        if threads:
            cfg["INFERENCE_NUM_THREADS"] = threads
        self._det = core.compile_model(core.read_model(str(ir_dir / "det.xml")), "CPU", cfg).create_infer_request()
        self._rec = core.compile_model(core.read_model(str(ir_dir / "rec.xml")), "CPU", cfg).create_infer_request()

    def det(self, x: np.ndarray) -> np.ndarray:
        self._det.infer({0: x})
        return self._det.get_output_tensor(0).data[0].copy()

    def rec(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._rec.infer({0: x})
        return self._rec.get_output_tensor(0).data.copy(), self._rec.get_output_tensor(1).data.copy()


class _TorchRunner:
    """torch 里最快的组合：检测 channels_last，识别 jit 冻结。给 OpenVINO 不理想的平台（如 Mac）做对照。"""

    def __init__(self, model_root: Path, threads: int = 0):
        import torch
        from transformers import PPOCRV6MediumDetForObjectDetection, PPOCRV6SmallRecForTextRecognition
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self._det = PPOCRV6MediumDetForObjectDetection.from_pretrained(
            str(model_root / DET_DIRNAME)).eval().to(memory_format=torch.channels_last)
        rec = PPOCRV6SmallRecForTextRecognition.from_pretrained(str(model_root / REC_DIRNAME)).eval()

        class Rec(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                prob, idx = self.m(pixel_values=x).last_hidden_state.max(dim=-1)
                return prob, idx.to(torch.int32)

        with torch.no_grad():
            tm = torch.jit.freeze(torch.jit.trace(Rec(rec).eval(), torch.zeros(4, 3, 48, 960), check_trace=False,
                                                  strict=False).eval())
            try:
                tm = torch.jit.optimize_for_inference(tm)
            except Exception:                     # 个别平台没有 oneDNN，退回只冻结
                pass
        self._rec = tm

    def det(self, x: np.ndarray) -> np.ndarray:
        t = self.torch.from_numpy(x).contiguous(memory_format=self.torch.channels_last)
        with self.torch.no_grad():
            return self._det(pixel_values=t).last_hidden_state[0, 0].numpy()

    def rec(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self.torch.no_grad():
            p, i = self._rec(self.torch.from_numpy(x))
        return p.numpy(), i.numpy()


# =====================================================================================================
# 对外接口
# =====================================================================================================
class FastPPOCR:
    def __init__(self, model_root: Path | str = DEFAULT_MODEL_ROOT, backend: str = "openvino",
                 ir_dir: Path | str = DEFAULT_IR_DIR, threads: int = 0):
        """backend: "openvino"（推荐）或 "torch"。threads=0 表示用全部核心。"""
        self.model_root, self.ir_dir = Path(model_root), Path(ir_dir)
        if backend == "openvino":
            export_openvino(self.model_root, self.ir_dir)
            self._r = _OpenVINORunner(self.ir_dir, threads)
        elif backend == "torch":
            self._r = _TorchRunner(self.model_root, threads)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        cfg = json.loads((self.model_root / REC_DIRNAME / "preprocessor_config.json").read_text(encoding="utf-8"))
        self._chars: list[str] = cfg["character_list"]
        self._lock = threading.Lock()             # 一个实例同一时刻只跑一页（推理本身已占满所有核心）

    def warmup(self, size: tuple[int, int] = (2100, 2970)) -> None:
        """服务启动时调用一次：用一张画了字的白纸跑一遍，把首次推理的一次性开销（约 3 秒）提前付掉。"""
        from PIL import ImageDraw
        im = Image.new("RGB", size, (255, 255, 255))
        d = ImageDraw.Draw(im)
        for k in range(6):
            d.text((size[0] // 10, size[1] // 10 + k * size[1] // 12), "warm up the text recognizer " * 3,
                   fill=(0, 0, 0), font_size=max(12, size[0] // 40))
        self.recognize_boxes(im)

    def recognize_boxes(self, image) -> list[Line]:
        """逐个检测框的识别结果（未合并成行），顺序为检测顺序。"""
        im = load_image(image)
        W, H = im.size
        with self._lock:
            prob = self._r.det(det_preprocess(im, det_size(W, H)))
            out = []
            for q in db_postprocess(prob, W, H):
                crop = rectify(im, q)
                # 竖排文字可能自下而上或自上而下：两个方向都识别，取置信度高的（只有竖框多一次识别）
                cands = [crop, crop.transpose(Image.ROTATE_180)] if is_vertical(q) else [crop]
                text, score = "", -1.0
                for c in cands:
                    p, i = self._r.rec(rec_input(resize_h48(c)))
                    t, s = ctc_decode(i[0], p[0], self._chars)
                    if s > score:
                        text, score = t, s
                if text.strip():
                    box = (float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max()))
                    out.append(Line(text, box, score, quad_rect(q)))
        return out

    def recognize(self, image) -> list[Line]:
        """按阅读顺序合并成行。"""
        return group_lines(self.recognize_boxes(image))

    def recognize_text(self, image) -> str:
        return join_lines([ln.text for ln in self.recognize(image)])

    def recognize_pages(self, images) -> str:
        """多页按顺序拼成一篇（跨页的断词也会拼回）。"""
        return join_lines([ln.text for im in images for ln in self.recognize(im)])


# =====================================================================================================
# 预处理 / 后处理（numpy + PIL，不依赖 OpenCV）
# =====================================================================================================
def load_image(image) -> Image.Image:
    """路径、PIL.Image 或 RGB uint8 数组 -> RGB PIL.Image（按 EXIF 转正，透明底合成白底）。"""
    if isinstance(image, (str, Path)):
        im = Image.open(image)
    elif isinstance(image, np.ndarray):
        im = Image.fromarray(image)
    else:
        im = image
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


def det_size(w: int, h: int, limit: int = 736, max_side: int = 4000) -> tuple[int, int]:
    """模型自带的 DetResizeForTest（limit_type=min）：短边不足 736 才放大，长边封顶 4000，取整到 32。"""
    r = limit / min(w, h) if min(w, h) < limit else 1.0
    rw, rh = int(w * r), int(h * r)
    if max(rw, rh) > max_side:
        k = max_side / max(rw, rh)
        rw, rh = int(rw * k), int(rh * k)
    return max(32, int(round(rw / 32) * 32)), max(32, int(round(rh / 32) * 32))


def det_preprocess(im: Image.Image, size: tuple[int, int]) -> np.ndarray:
    a = np.asarray(im.resize(size, Image.BILINEAR), dtype=np.float32)[:, :, ::-1]   # RGB -> BGR
    a = (a * (1 / 255.0) - DET_MEAN) / DET_STD
    return np.ascontiguousarray(a.transpose(2, 0, 1))[None]


def _runs(mask):
    H, W = mask.shape
    pad = np.zeros((H, W + 2), np.int8)
    pad[:, 1:-1] = mask
    d = np.diff(pad, axis=1)
    sr, sc = np.nonzero(d == 1)
    _, ec = np.nonzero(d == -1)
    return sr, sc, ec


def _label_runs(sr, sc, ec, W):
    """横向游程的 8 连通标记，返回每个游程所属的连通域编号。"""
    n = len(sr)
    stride = W + 2
    skey = sr.astype(np.int64) * stride + sc
    ekey = sr.astype(np.int64) * stride + ec
    prev_row = (sr.astype(np.int64) - 1) * stride
    lo = np.searchsorted(ekey, prev_row + sc, side="left")
    hi = np.searchsorted(skey, prev_row + ec, side="right")
    cnt = np.maximum(hi - lo, 0)
    cnt[sr == 0] = 0
    a = np.repeat(np.arange(n), cnt)
    b = np.repeat(lo, cnt) + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    lab = np.arange(n)
    if len(a):
        while True:
            la, lb = lab[a], lab[b]
            if np.array_equal(la, lb):
                break
            np.minimum.at(lab, np.maximum(la, lb), np.minimum(la, lb))
            while True:
                nl = lab[lab]
                if np.array_equal(nl, lab):
                    break
                lab = nl
    return np.unique(lab, return_inverse=True)[1]


def _hull(pts):
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


def _min_area_rect(pts):
    hull = _hull(pts)
    if len(hull) < 3:
        return pts.mean(0), (float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))), np.array([1.0, 0.0])
    e = np.roll(hull, -1, axis=0) - hull
    ang = np.unique(np.round(np.arctan2(e[:, 1], e[:, 0]) % (np.pi / 2), 6))
    u = np.stack([np.cos(ang), np.sin(ang)], 1)
    v = np.stack([-u[:, 1], u[:, 0]], 1)
    pu, pv = hull @ u.T, hull @ v.T
    k = int(np.argmin(np.ptp(pu, 0) * np.ptp(pv, 0)))
    u0, u1, v0, v1 = pu[:, k].min(), pu[:, k].max(), pv[:, k].min(), pv[:, k].max()
    return u[k] * (u0 + u1) / 2 + v[k] * (v0 + v1) / 2, (u1 - u0, v1 - v0), u[k]


def _order_quad(pts):
    p = pts[np.argsort(pts[:, 0], kind="stable")]
    left, right = p[:2], p[2:]
    tl, bl = (left[0], left[1]) if left[0, 1] <= left[1, 1] else (left[1], left[0])
    tr, br = (right[0], right[1]) if right[0, 1] <= right[1, 1] else (right[1], right[0])
    return np.array([tl, tr, br, bl])


def _rect_points(c, wh, ax):
    ay = np.array([-ax[1], ax[0]])
    w, h = wh
    return _order_quad(np.array([c - ax * w / 2 - ay * h / 2, c + ax * w / 2 - ay * h / 2,
                                 c + ax * w / 2 + ay * h / 2, c - ax * w / 2 + ay * h / 2]))


def _quad_score(prob, quad):
    H, W = prob.shape
    x0, x1 = int(max(0, math.floor(quad[:, 0].min()))), int(min(W - 1, math.ceil(quad[:, 0].max())))
    y0, y1 = int(max(0, math.floor(quad[:, 1].min()))), int(min(H - 1, math.ceil(quad[:, 1].max())))
    if x1 < x0 or y1 < y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    inside = np.ones(yy.shape, bool)
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        inside &= (q[0] - p[0]) * (yy - p[1]) - (q[1] - p[1]) * (xx - p[0]) >= -0.5 * math.hypot(q[0] - p[0],
                                                                                                    q[1] - p[1])
    sub = prob[y0:y1 + 1, x0:x1 + 1]
    return float(sub[inside].mean()) if inside.any() else 0.0


def db_postprocess(prob: np.ndarray, src_w: int, src_h: int, thresh=0.2, box_thresh=0.45, unclip_ratio=1.4,
                   min_size=3, max_candidates=3000) -> list[np.ndarray]:
    """DB 后处理（参数取自模型 inference.yml）。返回原图坐标的四边形 (4,2)，顺序 左上/右上/右下/左下。"""
    H, W = prob.shape
    sr, sc, ec = _runs(prob > thresh)
    if len(sr) == 0:
        return []
    comp = _label_runs(sr, sc, ec, W)
    order = np.argsort(comp, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(comp[order])) + 1)
    sx, sy = src_w / W, src_h / H
    boxes = []
    for g in groups[:max_candidates]:
        if int((ec[g] - sc[g]).sum()) < 4:
            continue
        pts = np.concatenate([np.stack([sc[g], sr[g]], 1), np.stack([ec[g] - 1, sr[g]], 1)]).astype(np.float64)
        c, (w, h), ax = _min_area_rect(pts)
        if min(w, h) < min_size or _quad_score(prob, _rect_points(c, (w, h), ax)) < box_thresh:
            continue
        d = w * h * unclip_ratio / (2 * (w + h))
        if min(w + 2 * d, h + 2 * d) < min_size + 2:
            continue
        q = _rect_points(c, (w + 2 * d, h + 2 * d), ax)
        q[:, 0] = np.clip(q[:, 0] * sx, 0, src_w)
        q[:, 1] = np.clip(q[:, 1] * sy, 0, src_h)
        boxes.append(q)
    return boxes


def _quad_wh(quad: np.ndarray) -> tuple[int, int]:
    tl, tr, br, bl = quad
    w = max(1, int(max(np.linalg.norm(tl - tr), np.linalg.norm(br - bl))))
    h = max(1, int(max(np.linalg.norm(tl - bl), np.linalg.norm(tr - br))))
    return w, h


def is_vertical(quad: np.ndarray) -> bool:
    w, h = _quad_wh(quad)
    return h / w >= 1.5


def quad_rect(quad: np.ndarray) -> tuple[float, float, float, float, float]:
    """(cx, cy, w, h, angle°)：w 为沿文字方向的边长；角度为图像坐标（y 向下）中文字方向的倾角，
    横排约 0（右端偏下为正），竖排约 ±90。"""
    tl, tr, br, bl = quad
    top, side = tr - tl, bl - tl
    wt, ws = float(np.linalg.norm(top)), float(np.linalg.norm(side))
    cx, cy = (float(v) for v in quad.mean(axis=0))
    if wt >= ws:
        return cx, cy, wt, ws, math.degrees(math.atan2(float(top[1]), float(top[0])))
    return cx, cy, ws, wt, math.degrees(math.atan2(float(side[1]), float(side[0])))


def rectify(im: Image.Image, quad: np.ndarray) -> Image.Image:
    """按四边形在原分辨率上矫正裁剪（双三次），竖排文字转成横排。"""
    tl, tr, br, bl = quad
    w = max(1, int(max(np.linalg.norm(tl - tr), np.linalg.norm(br - bl))))
    h = max(1, int(max(np.linalg.norm(tl - bl), np.linalg.norm(tr - br))))
    crop = im.transform((w, h), Image.QUAD, tuple(float(v) for v in (*tl, *bl, *br, *tr)), Image.BICUBIC)
    return crop.transpose(Image.ROTATE_90) if h / w >= 1.5 else crop


def resize_h48(crop: Image.Image, max_w: int = 3200) -> np.ndarray:
    """缩放到高 48、保持宽高比；双线性不抗锯齿（与 cv2.INTER_LINEAR 一致）。"""
    w, h = crop.size
    nw = int(min(max_w, max(1, math.ceil(REC_H * w / h))))
    return np.asarray(crop.transform((nw, REC_H), Image.AFFINE, (w / nw, 0, 0, 0, h / REC_H, 0), Image.BILINEAR))


def rec_input(crop: np.ndarray, min_w: int = 320) -> np.ndarray:
    """(48,W,3) RGB uint8 -> (1,3,48,max(W,320)) BGR，[-1,1]，右侧补 0（PaddleOCR 做法）。"""
    W = max(min_w, crop.shape[1])
    out = np.zeros((1, 3, REC_H, W), np.float32)
    out[0, :, :, : crop.shape[1]] = (crop[:, :, ::-1].astype(np.float32) * (2 / 255.0) - 1.0).transpose(2, 0, 1)
    return out


def ctc_decode(idx: np.ndarray, prob: np.ndarray, chars: list[str]) -> tuple[str, float]:
    keep = np.ones(len(idx), bool)
    keep[1:] = idx[1:] != idx[:-1]
    keep &= idx != 0
    return "".join(chars[j] for j in idx[keep]), (float(prob[keep].mean()) if keep.any() else 0.0)


def group_lines(items: list[Line]) -> list[Line]:
    """把纵向中心重叠的检测框归为一行，行内从左到右，行间从上到下。"""
    items = sorted((it for it in items if it.text.strip()), key=lambda it: (it.box[1] + it.box[3]) / 2)
    rows: list[list] = []
    for it in items:
        x0, y0, x1, y1 = it.box
        cy, h = (y0 + y1) / 2, y1 - y0
        for row in rows[-3:]:
            ly0, ly1 = row[0]
            if min(y1, ly1) - max(y0, ly0) > 0.5 * min(h, ly1 - ly0) and ly0 <= cy <= ly1:
                row[1].append(it)
                row[0] = (min(ly0, y0), max(ly1, y1))
                break
        else:
            rows.append([(y0, y1), [it]])
    rows.sort(key=lambda r: (r[0][0] + r[0][1]) / 2)
    out = []
    for _, members in rows:
        members.sort(key=lambda it: it.box[0])
        out.append(Line(" ".join(m.text for m in members),
                        (min(m.box[0] for m in members), min(m.box[1] for m in members),
                         max(m.box[2] for m in members), max(m.box[3] for m in members)),
                        float(np.mean([m.score for m in members]))))
    return out


def join_lines(lines: list[str]) -> str:
    """行拼成连续文字，去掉行尾断词连字符（"u-" + "nderstand" -> "understand"）。"""
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if out and out[-1] in HYPHENS and len(out) > 1 and out[-2].isalpha():
            out = out[:-1].rstrip() + ln
        else:
            out = f"{out} {ln}" if out else ln
    return out


# =====================================================================================================
def main():
    ap = argparse.ArgumentParser(description="PP-OCRv6 medium on CPU (OpenVINO FP32)")
    ap.add_argument("images", nargs="+", help="一篇作文的各页，按顺序")
    ap.add_argument("--backend", default="openvino", choices=["openvino", "torch"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    ap.add_argument("--lines", action="store_true", help="逐行打印（含坐标和置信度）")
    ap.add_argument("--json", help="把逐行结果写到这个 JSON 文件")
    a = ap.parse_args()

    t0 = time.perf_counter()
    ocr = FastPPOCR(a.model_root, backend=a.backend, threads=a.threads)
    print(f"[load {time.perf_counter() - t0:.1f}s]")
    pages = []
    for p in a.images:
        t = time.perf_counter()
        lines = ocr.recognize(p)
        print(f"[{p}: {len(lines)} lines, {time.perf_counter() - t:.2f}s]")
        pages.append({"image": p, "lines": [asdict(ln) for ln in lines]})
        if a.lines:
            for ln in lines:
                print(f"  {ln.score:.3f} {tuple(round(v) for v in ln.box)} {ln.text}")
    print(join_lines([ln["text"] for pg in pages for ln in pg["lines"]]))
    if a.json:
        Path(a.json).write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
