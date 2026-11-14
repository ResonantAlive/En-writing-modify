"""Local OCR service: environment check, global single-engine FIFO queue, per-subtask OCR (writes ocr.json)."""
from __future__ import annotations

import importlib.util
import logging
import threading
import time
from collections import deque
from pathlib import Path

from ..batches import paths, store
from ..config import PROJECT_ROOT
from ..core import atomic_io, clock
from ..staging import images as imgmod
from . import layout
from .common import OcrError, PageResult

log = logging.getLogger("wm.ocr")

ENGINES = ("paddle",)                     # only PP-OCRv6 (decided 2026-11-14; HunyuanOCR autocorrects spelling)
_REQUIRED_DIRS = {"paddle": ("PP-OCRv6_medium_det_safetensors", "PP-OCRv6_medium_rec_safetensors")}


class OcrFailed(Exception):
    pass


class OcrEmpty(Exception):
    pass


# ------------------------------------------------------------------ environment
def model_root(cfg) -> Path:
    r = cfg["ocr"].get("model_root")
    return Path(r) if r else PROJECT_ROOT


_env_cache: dict = {"t": 0.0, "v": None}


def ir_dir(cfg) -> Path:
    r = cfg["ocr"].get("ir_dir")
    return Path(r) if r else cfg.data_dir / "ov_models"


def _ir_ready(cfg) -> bool:
    d = ir_dir(cfg)
    return all((d / f).is_file() for f in ("det.xml", "det.bin", "rec.xml", "rec.bin"))


def engine_status(cfg, engine: str) -> tuple[bool, str | None]:
    if engine not in ENGINES:
        return False, f"unknown local engine '{engine}'"
    root = model_root(cfg)
    backend = cfg["ocr"].get("backend", "openvino")
    if backend == "openvino":
        need = ["openvino"] + ([] if _ir_ready(cfg) else ["torch", "transformers"])   # export needs torch once
    else:
        need = ["torch", "transformers"]
    for mod in need:
        if importlib.util.find_spec(mod) is None:
            return False, f"python package '{mod}' is not installed in the project environment"
    for d in _REQUIRED_DIRS[engine]:
        if not (root / d).is_dir():
            return False, f"model folder '{d}' not found under {root}"
    return True, None


def local_env_ok(cfg) -> tuple[bool, str | None]:
    """True when the default engine can run. Cached for 30 s; written to data/state/env_check.json."""
    now = time.monotonic()
    if _env_cache["v"] is not None and now - _env_cache["t"] < 30:
        return _env_cache["v"]
    engines = {e: engine_status(cfg, e) for e in ENGINES}
    default = cfg["default_local_engine"]
    ok, reason = engines.get(default, (False, "unknown default engine"))
    try:
        atomic_io.write_json_atomic(cfg.state_dir / "env_check.json", {
            "checked_at": clock.iso(),
            "engines": {e: {"ok": v[0], "reason": v[1]} for e, v in engines.items()}})
    except OSError:
        pass
    _env_cache["t"], _env_cache["v"] = now, (ok, reason)
    return ok, reason


def engine_ok(cfg, engine: str) -> tuple[bool, str | None]:
    return engine_status(cfg, engine)


# ------------------------------------------------------------------ global FIFO queue + engine cache
class _FifoLock:
    def __init__(self):
        self._cv = threading.Condition()
        self._q: deque = deque()
        self._busy = False

    def __enter__(self):
        t = object()
        with self._cv:
            self._q.append(t)
            while self._busy or self._q[0] is not t:
                self._cv.wait()
            self._q.popleft()
            self._busy = True

    def __exit__(self, *a):
        with self._cv:
            self._busy = False
            self._cv.notify_all()

    def waiting(self) -> int:
        with self._cv:
            return len(self._q)


_QUEUE = _FifoLock()
_engine = {"name": None, "obj": None}


def _get_engine(cfg, name: str):
    """Only one engine is resident at a time: switching drops the previous one (memory)."""
    if _engine["name"] == name and _engine["obj"] is not None:
        return _engine["obj"]
    _engine["obj"] = None
    _engine["name"] = None
    import gc
    gc.collect()
    if name != "paddle":
        raise OcrFailed(f"unknown engine {name}")
    from .paddle_engine import PaddleEngine
    ocr = cfg["ocr"]
    obj = PaddleEngine(model_root(cfg), backend=ocr.get("backend", "openvino"), ir_dir=ir_dir(cfg),
                       threads=int(ocr.get("threads", 0)))
    obj.load()
    _engine["name"], _engine["obj"] = name, obj
    return obj


def recognize_images(cfg, engine_name: str, im_list) -> tuple[list[PageResult], dict]:
    """Serialised through the global queue. Returns (pages, engine info)."""
    with _QUEUE:
        try:
            eng = _get_engine(cfg, engine_name)
            pages = [eng.recognize(im) for im in im_list]
            return pages, eng.info()
        except OcrError as e:
            raise OcrFailed(str(e)) from e
        except (ImportError, OSError, RuntimeError) as e:
            raise OcrFailed(f"{type(e).__name__}: engine could not run") from e


# ------------------------------------------------------------------ per subtask
def assemble(pages: list[PageResult], blank_line_factor: float = 1.6) -> str:
    if all(p.layout_reliable for p in pages):
        return layout.restore_pages([p.lines for p in pages], blank_line_factor).strip()
    return layout.plain_join([p.lines for p in pages]).strip()


def warmup_async(cfg) -> threading.Thread | None:
    """Load (and on first use export) the default engine in the background, through the queue, so the first
    essay does not pay the start-up cost. Called by the launcher only (never by tests / create_app)."""
    ok, reason = local_env_ok(cfg)
    if not ok:
        log.info("local OCR warm-up skipped: %s", reason)
        return None

    def run() -> None:
        try:
            with _QUEUE:
                eng = _get_engine(cfg, cfg["default_local_engine"])
                eng.warmup()
            log.info("local OCR engine ready")
        except Exception:
            log.exception("local OCR warm-up failed")
    t = threading.Thread(target=run, name="ocr-warmup", daemon=True)
    t.start()
    return t


def _set_state(cfg, sid: str, state: str) -> None:
    def fn(m: dict) -> None:
        m["local_ocr_state"] = state
    try:
        store.update_meta(cfg, sid, fn)
    except Exception:
        log.warning("could not update local_ocr_state of %s", sid)


def recognize_one(cfg, im) -> tuple[PageResult, dict]:
    """One image with the default engine, through the global queue (used for blank answer sheets)."""
    pages, info = recognize_images(cfg, cfg["default_local_engine"], [im])
    return pages[0], info


def apply_sheet_filter(pages: list[PageResult], template_pages: list[dict]) -> list[dict]:
    """Remove printed answer-sheet matter in place (pages[k].lines rebuilt from the kept raw boxes).
    Returns per page {"excluded": [...], "info": {...}}."""
    from . import sheet_filter
    out = []
    for pg in pages:
        raw = pg.boxes or pg.lines
        kept, excluded, info = sheet_filter.filter_page(raw, template_pages)
        if excluded:
            pg.boxes = kept
            pg.lines, pg.skew_deg = layout.group_boxes(kept, pg.width, pg.height)
        out.append({"excluded": excluded, "info": info})
    return out


def _sheet_key(snap: dict | None) -> dict | None:
    return {"id": snap.get("id"), "page_ids": list(snap.get("page_ids") or [])} if snap else None


def run_for_subtask(cfg, sid: str, engine_name: str, sheet: dict | None = None) -> dict:
    """Blocking. Reuses an existing ocr.json produced by the same engine and answer-sheet template; otherwise
    recognises every original photo (EXIF-oriented), removes the template's printed matter and writes ocr.json.
    Raises OcrFailed / OcrEmpty."""
    sdir = paths.subtask_dir(cfg, sid)
    p = sdir / "ocr.json"
    old = atomic_io.read_json(p, None)
    if isinstance(old, dict) and old.get("transcription_source") == "local_ocr" \
            and (old.get("engine") or {}).get("local_engine") == engine_name \
            and old.get("sheet_template") == _sheet_key(sheet):
        if not (old.get("transcription") or "").strip():
            raise OcrEmpty("no text recognised")
        return old
    meta = store.read_meta(cfg, sid)
    if meta is None or not meta.get("images"):
        raise OcrFailed("no images")
    _set_state(cfg, sid, "running")
    try:
        ims = [imgmod.load_oriented(sdir / im["original"]) for im in meta["images"]]
    except Exception as e:
        _set_state(cfg, sid, "failed")
        raise OcrFailed(f"image unreadable: {type(e).__name__}") from e
    try:
        pages, info = recognize_images(cfg, engine_name, ims)
    except OcrFailed:
        _set_state(cfg, sid, "failed")
        raise
    finally:
        for im in ims:
            im.close()
    filt = [{"excluded": [], "info": None}] * len(pages)
    if sheet:
        from ..sheets import store as sheets
        filt = apply_sheet_filter(pages, sheets.pages_for(cfg, sheet))
    text = assemble(pages, float(cfg["layout"]["blank_line_factor"]))
    doc = {"schema_version": 1, "mode": "local_ocr", "created_at": clock.iso(),
           "transcription": text, "transcription_source": "local_ocr",
           "sheet_template": _sheet_key(sheet),
           "pages": [{"index": k, "image_ref": {"path": im["original"], "sha256": im.get("sha256"),
                                                "mime": im.get("mime")},
                      "layout_reliable": pg.layout_reliable, "width": pg.width, "height": pg.height,
                      "skew_deg": round(pg.skew_deg, 2),
                      "lines": [{"text": l.text, "box": [round(v, 1) for v in l.box],
                                 "score": None if l.score is None else round(l.score, 4)} for l in pg.lines],
                      "excluded": f["excluded"], "sheet_match": f["info"],
                      "raw": pg.raw, "warnings": pg.warnings}
                     for k, (im, pg, f) in enumerate(zip(meta["images"], pages, filt), start=1)],
           "engine": info, "warnings": [w for pg in pages for w in pg.warnings]}
    atomic_io.write_json_atomic(p, doc)
    _set_state(cfg, sid, "done" if text.strip() else "failed")
    if not text.strip():
        raise OcrEmpty("no text recognised")
    return doc
