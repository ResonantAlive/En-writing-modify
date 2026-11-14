"""Answer-sheet templates: blank sheets uploaded once, recognised once, reused to remove printed matter.

Storage:
  data/app/sheet_templates.json            metadata + recognised printed lines of every blank page
  data/app/sheet_templates/<tid>/<pid>.<ext>   the uploaded blank-sheet photos (originals)

A template is a growing set of pages (cover page, inner pages, ... of the same booklet). Pages are recognised
in the background through the global OCR queue (status pending -> ready | failed). There is no deletion in v1:
a wrong page is switched off with `disabled` (reversible). Batches pin the page ids they used at creation.
Visibility follows prompt templates: owner, root, or shared (sharing is a root action).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import BinaryIO, Iterable

from ..auth.deps import Principal
from ..config import AppConfig
from ..core import atomic_io, clock
from ..core.ids import new_id
from ..core.locks import file_lock
from ..errors import AppError, NotFound
from ..staging import images

log = logging.getLogger("wm.sheets")

MAX_NAME = 100
MAX_PAGES = 20
CHUNK = 1 << 20
ESSAY_LIKE_LETTERS = 1500        # a "blank" page with this much text is probably a written essay


def _path(cfg: AppConfig) -> Path:
    return cfg.app_dir / "sheet_templates.json"


def files_dir(cfg: AppConfig, tid: str) -> Path:
    return cfg.app_dir / "sheet_templates" / tid


def _read(cfg: AppConfig) -> dict:
    d = atomic_io.read_json(_path(cfg), None) or {}
    d.setdefault("schema_version", 1)
    d.setdefault("templates", [])
    return d


def _mutate(cfg: AppConfig, fn):
    with file_lock(cfg.lock_path("sheet_templates")):
        data = _read(cfg)
        res = fn(data)
        atomic_io.write_json_atomic(_path(cfg), data)
    return res


def _find(data: dict, tid: str) -> dict | None:
    return next((t for t in data["templates"] if t["id"] == tid), None)


def get(cfg: AppConfig, tid: str) -> dict | None:
    return _find(_read(cfg), tid)


def visible(p: Principal, t: dict) -> bool:
    return p.is_root or t.get("owner_id") == p.user_id or bool(t.get("shared"))


def can_edit(p: Principal, t: dict) -> bool:
    return p.is_root or t.get("owner_id") == p.user_id


def get_visible(cfg: AppConfig, p: Principal, tid: str) -> dict:
    t = get(cfg, tid)
    if t is None or not visible(p, t):
        raise NotFound()
    return t


def _counts(t: dict) -> dict:
    c = {"total": 0, "ready": 0, "pending": 0, "failed": 0, "disabled": 0}
    for pg in t["pages"]:
        c["total"] += 1
        if pg.get("disabled"):
            c["disabled"] += 1
        else:
            c[pg["status"]] = c.get(pg["status"], 0) + 1
    return c


def usable_pages(t: dict) -> list[dict]:
    return [pg for pg in t["pages"] if pg["status"] == "ready" and not pg.get("disabled")]


def summary(t: dict) -> dict:
    c = _counts(t)
    return {"id": t["id"], "name": t["name"], "owner_id": t["owner_id"], "shared": bool(t.get("shared")),
            "created_at": t["created_at"], "updated_at": t["updated_at"], "pages": c,
            "usable": c["ready"] > 0}


def public_page(pg: dict) -> dict:
    ocr = pg.get("ocr") or {}
    return {"page_id": pg["page_id"], "orig_name": pg.get("orig_name"), "status": pg["status"],
            "error": pg.get("error"), "disabled": bool(pg.get("disabled")), "width": pg.get("width"),
            "height": pg.get("height"), "created_at": pg["created_at"], "warnings": ocr.get("warnings", []),
            "lines": [{"text": l["text"], "box": l["box"]} for l in ocr.get("lines", [])]}


def detail(t: dict) -> dict:
    out = summary(t)
    out["page_list"] = [public_page(pg) for pg in t["pages"]]
    return out


def list_visible(cfg: AppConfig, p: Principal) -> list[dict]:
    return [summary(t) for t in _read(cfg)["templates"] if visible(p, t)]


def _name(v) -> str:
    if not isinstance(v, str) or not v.strip():
        raise AppError("validation_error", "name is required", 422, {"field": "name"})
    v = v.strip()
    if len(v) > MAX_NAME:
        raise AppError("validation_error", f"name is at most {MAX_NAME} characters", 422, {"field": "name"})
    return v


def create(cfg: AppConfig, p: Principal, name) -> dict:
    name = _name(name)
    now = clock.iso()
    t = {"id": new_id("sheet"), "name": name, "owner_id": p.user_id, "shared": False, "created_at": now,
         "updated_at": now, "pages": []}

    def fn(data):
        data["templates"].append(t)
    _mutate(cfg, fn)
    return t


def rename(cfg: AppConfig, p: Principal, tid: str, name) -> dict:
    name = _name(name)

    def fn(data):
        t = _find(data, tid)
        if t is None or not visible(p, t):
            raise NotFound()
        if not can_edit(p, t):
            raise AppError("forbidden", "only the owner can change this template", 403)
        t["name"], t["updated_at"] = name, clock.iso()
        return t
    return _mutate(cfg, fn)


def set_shared(cfg: AppConfig, tid: str, shared: bool) -> dict:
    def fn(data):
        t = _find(data, tid)
        if t is None:
            raise NotFound()
        t["shared"], t["updated_at"] = bool(shared), clock.iso()
        return t
    return _mutate(cfg, fn)


def set_disabled(cfg: AppConfig, p: Principal, tid: str, pid: str, disabled: bool) -> dict:
    def fn(data):
        t = _find(data, tid)
        if t is None or not visible(p, t):
            raise NotFound()
        if not can_edit(p, t):
            raise AppError("forbidden", "only the owner can change this template", 403)
        pg = next((x for x in t["pages"] if x["page_id"] == pid), None)
        if pg is None:
            raise NotFound()
        pg["disabled"], t["updated_at"] = bool(disabled), clock.iso()
        return t
    return _mutate(cfg, fn)


def retry_page(cfg: AppConfig, p: Principal, tid: str, pid: str) -> dict:
    def fn(data):
        t = _find(data, tid)
        if t is None or not visible(p, t):
            raise NotFound()
        if not can_edit(p, t):
            raise AppError("forbidden", "only the owner can change this template", 403)
        pg = next((x for x in t["pages"] if x["page_id"] == pid), None)
        if pg is None:
            raise NotFound()
        if pg["status"] != "failed":
            raise AppError("not_failed", "only a failed page can be retried", 409)
        pg["status"], pg["error"] = "pending", None
        return t
    t = _mutate(cfg, fn)
    kick(cfg)
    return t


def image_path(cfg: AppConfig, p: Principal, tid: str, pid: str) -> tuple[Path, str]:
    t = get_visible(cfg, p, tid)
    pg = next((x for x in t["pages"] if x["page_id"] == pid), None)
    if pg is None:
        raise NotFound()
    path = files_dir(cfg, tid) / pg["file"]
    if not path.is_file():
        raise NotFound()
    return path, pg.get("mime") or "application/octet-stream"


def add_pages(cfg: AppConfig, p: Principal, tid: str,
              items: Iterable[tuple[str, BinaryIO]]) -> tuple[dict, list[dict]]:
    """Store blank-sheet photos (validated like essay uploads) and queue them for recognition."""
    t = get_visible(cfg, p, tid)
    if not can_edit(p, t):
        raise AppError("forbidden", "only the owner can add pages", 403)
    d = files_dir(cfg, tid)
    d.mkdir(parents=True, exist_ok=True)
    cap = int(cfg["max_file_mb"]) * 1024 * 1024
    ready, rejected = [], []
    room = MAX_PAGES - len(t["pages"])
    for name, stream in items:
        name = (name or "")[:512]
        if len(ready) >= room:
            rejected.append({"name": name, "code": "too_many_pages", "message": f"At most {MAX_PAGES} pages"})
            continue
        head = stream.read(CHUNK)
        sniffed = images.sniff_type(head[:64])
        if sniffed is None:
            rejected.append({"name": name, "code": "unsupported_type",
                             "message": "Unsupported file type (allowed: JPEG, PNG, WebP, HEIC, HEIF)"})
            continue
        mime, ext = sniffed
        pid = new_id("pg")
        dst = d / f"{pid}.{ext}"
        size, too_big = 0, False
        with open(dst, "wb") as out:
            chunk = head
            while chunk:
                size += len(chunk)
                if size > cap:
                    too_big = True
                    break
                out.write(chunk)
                chunk = stream.read(CHUNK)
        if too_big:
            dst.unlink(missing_ok=True)
            rejected.append({"name": name, "code": "file_too_large",
                             "message": f"File exceeds the {cfg['max_file_mb']} MB limit"})
            continue
        try:
            with images.load_oriented(dst) as im:
                w, h = im.size
        except Exception:
            dst.unlink(missing_ok=True)
            rejected.append({"name": name, "code": "corrupt_image", "message": "The image could not be decoded"})
            continue
        ready.append({"page_id": pid, "orig_name": name, "file": f"{pid}.{ext}", "mime": mime,
                      "sha256": images.sha256_file(dst), "width": w, "height": h, "status": "pending",
                      "error": None, "disabled": False, "created_at": clock.iso(), "ocr": None})

    def fn(data):
        cur = _find(data, tid)
        for pg in ready:
            if len(cur["pages"]) >= MAX_PAGES:
                (d / pg["file"]).unlink(missing_ok=True)
                rejected.append({"name": pg["orig_name"], "code": "too_many_pages",
                                 "message": f"At most {MAX_PAGES} pages"})
            else:
                cur["pages"].append(pg)
        if ready:
            cur["updated_at"] = clock.iso()
        return cur
    t = _mutate(cfg, fn)
    if ready:
        kick(cfg)
    return t, rejected


# ------------------------------------------------------------------ background recognition
_worker = {"thread": None}
_worker_lock = threading.Lock()


def _next_pending(cfg: AppConfig) -> tuple[str, dict] | None:
    for t in _read(cfg)["templates"]:
        for pg in t["pages"]:
            if pg["status"] == "pending":
                return t["id"], pg
    return None


def _set_result(cfg: AppConfig, tid: str, pid: str, **fields) -> None:
    def fn(data):
        t = _find(data, tid)
        pg = next((x for x in t["pages"] if x["page_id"] == pid), None) if t else None
        if pg is not None:
            pg.update(fields)
            t["updated_at"] = clock.iso()
    _mutate(cfg, fn)


def recognize_page(cfg: AppConfig, tid: str, pg: dict) -> None:
    from ..ocr import service
    path = files_dir(cfg, tid) / pg["file"]
    try:
        with images.load_oriented(path) as im:
            res, info = service.recognize_one(cfg, im)
    except service.OcrFailed as e:
        _set_result(cfg, tid, pg["page_id"], status="failed", error=str(e)[:200])
        return
    except Exception as e:                                       # unreadable file etc.
        _set_result(cfg, tid, pg["page_id"], status="failed", error=f"{type(e).__name__}"[:200])
        return
    raw = res.boxes or res.lines
    lines = [{"text": l.text, "box": [round(v, 1) for v in l.box],
              "score": None if l.score is None else round(l.score, 4)} for l in raw if l.text.strip()]
    warnings = []
    if not lines:
        warnings.append("no_printed_text")
    if sum(c.isalpha() for l in lines for c in l["text"]) >= ESSAY_LIKE_LETTERS:
        warnings.append("looks_like_written_page")
    _set_result(cfg, tid, pg["page_id"], status="ready", error=None,
                ocr={"engine": info, "lines": lines, "warnings": warnings, "recognized_at": clock.iso()})


def _run(cfg: AppConfig) -> None:
    try:
        while True:
            nxt = _next_pending(cfg)
            if nxt is None:
                return
            recognize_page(cfg, *nxt)
    except Exception:
        log.exception("sheet template recognition worker stopped")
    finally:
        with _worker_lock:
            _worker["thread"] = None
        if _next_pending(cfg) is not None:          # pages added while finishing
            kick(cfg)


def kick(cfg: AppConfig) -> None:
    """Start the background worker if pending pages exist and it is not running."""
    with _worker_lock:
        th = _worker["thread"]
        if th is not None and th.is_alive():
            return
        if _next_pending(cfg) is None:
            return
        th = threading.Thread(target=_run, args=(cfg,), name="sheet-ocr", daemon=True)
        _worker["thread"] = th
        th.start()


def wait_idle(timeout: float = 30.0) -> bool:
    """Tests: wait for the worker to finish."""
    th = _worker["thread"]
    if th is None:
        return True
    th.join(timeout)
    return not th.is_alive()


# ------------------------------------------------------------------ batch side
def snapshot_for_batch(cfg: AppConfig, p: Principal, tid) -> dict:
    """Validate the template chosen at batch creation and pin its usable pages."""
    if not isinstance(tid, str) or not tid:
        raise AppError("sheet_template_required",
                       "local recognition needs an answer-sheet template (upload a blank sheet first)", 422)
    t = get(cfg, tid)
    if t is None or not visible(p, t):
        raise AppError("invalid_sheet_template", "answer-sheet template not found", 422)
    pages = usable_pages(t)
    if not pages:
        c = _counts(t)
        msg = ("the blank sheets of this template are still being recognised" if c["pending"]
               else "this template has no recognised blank sheet")
        raise AppError("sheet_template_not_ready", msg, 409, {"pages": c})
    return {"id": t["id"], "name": t["name"], "page_ids": [pg["page_id"] for pg in pages]}


def pages_for(cfg: AppConfig, snap: dict | None) -> list[dict]:
    """Recognised printed lines of the pinned pages: [{"page_id", "lines"}]."""
    if not snap:
        return []
    t = get(cfg, snap.get("id"))
    if t is None:
        return []
    want = set(snap.get("page_ids") or [])
    return [{"page_id": pg["page_id"], "lines": (pg.get("ocr") or {}).get("lines", [])}
            for pg in t["pages"] if pg["page_id"] in want and pg["status"] == "ready"]
