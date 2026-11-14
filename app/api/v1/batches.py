"""Batch API: precheck / create / list / detail / control / SSE progress / original photo download /
teacher review / summary result."""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from typing import Optional

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from ...auth.audit import audit
from ...auth.deps import Principal, cfg_of, client_ip, require_user
from ...batches import create as create_mod
from ...batches import executor as executor_mod
from ...batches import breaker as breaker_mod
from ...batches import paths, results, review as review_mod, revisions, store
from ...core import atomic_io
from ...core.paths import safe_join
from ...errors import AppError, NotFound
from . import admin

router = APIRouter(prefix="/batches", tags=["batches"])

POLL_S = 1.0            # SSE state poll interval
KEEPALIVE_S = 15.0
FORCE_REFRESH_S = 5.0

# owner account disabled -> pause that owner's running batches (stay paused after re-enable)
if store.pause_owner_batches not in admin.on_user_disabled:
    admin.on_user_disabled.append(store.pause_owner_batches)


class ConfigIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    provider_id: Optional[str] = None
    model: Optional[str] = None
    params: Optional[dict] = None
    template_id: Optional[str] = None
    template_version: Optional[int] = None
    prompt_text: Optional[str] = None
    new_template_name: Optional[str] = None
    auto_append_fewshot: Optional[bool] = None
    third_party_ack: Optional[bool] = None


class CreateIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    upload_id: Optional[str] = None
    topic: Optional[str] = None
    rubric: Optional[str] = None
    genre: Optional[str] = None
    full_score: Optional[float] = None
    route: Optional[str] = None
    provider_id: Optional[str] = None
    model: Optional[str] = None
    template_id: Optional[str] = None
    template_version: Optional[int] = None
    vlm_transcription_choice: Optional[str] = None
    local_engine: Optional[str] = None
    sheet_template_id: Optional[str] = None
    fewshot_scope: Optional[str] = None
    zero_shot_approved: Optional[bool] = None
    third_party_ack: Optional[bool] = None
    grouping: Optional[dict] = None


def _load(cfg, p: Principal, task_code: str) -> dict:
    """Batch the caller may see, else 404 (never reveals other accounts' batches)."""
    b = store.read_batch(cfg, task_code)
    if b is None or (b.get("owner_id") != p.user_id and not p.is_root):
        raise NotFound()
    return b


def _tpl_keys(cfg, rev: dict) -> tuple[str, str]:
    """score_key / criteria_key of the template version pinned by a revision (defaults while unavailable)."""
    try:
        from ...prompts import store as pstore
        t = rev.get("template") or {}
        v = pstore.get_version(cfg, t.get("id"), t.get("version"))
        return v.get("score_key") or "判分", v.get("criteria_key") or "判据"
    except Exception:
        return "判分", "判据"


def _rev_summary(r: dict) -> dict:
    return {"rev": r.get("rev"), "at": r.get("at"), "provider_id": r.get("provider_id"),
            "provider_alias": r.get("provider_alias"), "model": r.get("model"),
            "template": r.get("template"), "reason": r.get("reason")}


def _summary(cfg, b: dict) -> dict:
    return store.summary(b, store.counts(cfg, b["task_code"]))


@router.post("/precheck")
def precheck(body: CreateIn, request: Request, p: Principal = Depends(require_user)):
    return create_mod.precheck(cfg_of(request), p, body.model_dump())


@router.post("", status_code=201)
@router.post("/", status_code=201, include_in_schema=False)
def create(body: CreateIn, request: Request, p: Principal = Depends(require_user)):
    return create_mod.create_batch(cfg_of(request), p, body.model_dump())


@router.get("")
@router.get("/", include_in_schema=False)
def list_batches(request: Request, all: bool = False, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    everyone = all and p.is_root
    rows = [b for b in store.iter_batches(cfg) if everyone or b.get("owner_id") == p.user_id]
    rows.sort(key=lambda b: (b.get("created_at", ""), b["task_code"]), reverse=True)
    return [_summary(cfg, b) for b in rows]


@router.get("/{task_code}")
def detail(task_code: str, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    b = _load(cfg, p, task_code)
    out = _summary(cfg, b)
    rev = b["config_revisions"][-1]
    out.update({
        "rubric": b.get("rubric"), "full_score": b.get("full_score"), "language": b.get("language"),
        "local_engine": b.get("local_engine"), "vlm_transcription_choice": b.get("vlm_transcription_choice"),
        "sheet_template": b.get("sheet_template"),
        "grouping": b.get("grouping"), "fewshot_scope": b.get("fewshot_scope"),
        "fewshot_ids": b.get("fewshot_ids"), "zero_shot_approved": b.get("zero_shot_approved"),
        "api_error_streak": b.get("api_error_streak"),
        "breaker": {"tripped_reason": (b.get("breaker") or {}).get("tripped_reason")},
        "config": {"rev": rev["rev"], "provider_id": rev.get("provider_id"), "model": rev.get("model"),
                   "template": rev.get("template")},
        "config_revisions": [_rev_summary(r) for r in b["config_revisions"]],
        "timeline": b.get("timeline", []),
    })
    out["score_key"], out["criteria_key"] = _tpl_keys(cfg, rev)
    return out


@router.get("/{task_code}/subtasks")
def subtasks(task_code: str, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    _load(cfg, p, task_code)
    out = []
    for m in store.read_metas(cfg, task_code):
        att, ai_score = review_mod.ai_snapshot(cfg, m["subtask_id"], m)
        rv = review_mod.latest(review_mod.read(cfg, m["subtask_id"])) if m.get("review_status") == "reviewed" else None
        out.append({"subtask_id": m["subtask_id"], "index": m.get("index"), "status": m["status"],
                    "fail_reason": m.get("fail_reason"), "review_status": m.get("review_status"),
                    "photos": len(m.get("images", [])), "warnings": m.get("warnings", []),
                    "ai_score": ai_score, "teacher_score": rv.get("final_score") if rv else None,
                    "adopted_ai": bool(rv.get("adopted_ai")) if rv else None,
                    "review_outdated": review_mod.outdated(rv, att)})
    return out


# ---- control
def _act(action: str):
    def handler(task_code: str, request: Request, p: Principal = Depends(require_user)):
        cfg = cfg_of(request)
        _load(cfg, p, task_code)
        b = store.user_action(cfg, task_code, action, p.user_id)
        if action == "cancel":
            executor_mod.abort_batch(cfg, task_code)
            results.generate_if_final(cfg, task_code, "cancelled")
        executor_mod.poke(cfg)
        return _summary(cfg, b)
    return handler


for _a in ("pause", "resume", "cancel", "rerun-failed"):
    router.post(f"/{{task_code}}/{_a}")(_act(_a))


@router.post("/{task_code}/config")
def change_config(task_code: str, body: ConfigIn, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    _load(cfg, p, task_code)
    rev = revisions.change_config(cfg, p, task_code, body.model_dump(exclude_unset=True))
    executor_mod.poke(cfg)
    return rev


@router.get("/{task_code}/failures")
def failures(task_code: str, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    b = _load(cfg, p, task_code)
    reasons: dict[str, int] = {}
    for m in store.read_metas(cfg, task_code):
        if m.get("status") == "failed":
            r = m.get("fail_reason") or "unknown"
        elif m.get("status") == "api_error":
            r = "api_error"
        else:
            continue
        reasons[r] = reasons.get(r, 0) + 1
    br = b.get("breaker") or {}
    return {"pause_reason": b.get("pause_reason"), "samples": br.get("samples") or [], "reasons": reasons}


def _sub(cfg, p: Principal, task_code: str, subtask_id: str) -> dict:
    _load(cfg, p, task_code)
    if not paths.valid_subtask_id(subtask_id) or subtask_id[:12] != task_code:
        raise NotFound()
    m = store.read_meta(cfg, subtask_id)
    if m is None:
        raise NotFound()
    return m


@router.get("/{task_code}/subtasks/{subtask_id}")
def subtask_detail(task_code: str, subtask_id: str, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    meta = _sub(cfg, p, task_code, subtask_id)
    sdir = paths.subtask_dir(cfg, subtask_id)
    attempts = []
    for n in breaker_mod.attempt_numbers(cfg, subtask_id):
        r = atomic_io.read_json(sdir / "llm" / f"{n:03d}.response.json", None)
        if not isinstance(r, dict):
            continue
        err = r.get("error")
        attempts.append({
            "attempt": r.get("attempt", n),
            "error_code": r.get("error_code") or (err.get("code") if isinstance(err, dict) else None),
            "classification": r.get("classification"), "http_status": r.get("http_status"),
            "latency_ms": r.get("latency_ms"), "usage": r.get("usage")})
    return {"meta": meta, "ocr": atomic_io.read_json(sdir / "ocr.json", None),
            "grading": atomic_io.read_json(sdir / "grading.json", None), "attempts": attempts,
            "review": atomic_io.read_json(sdir / "review.json", None)}


_N = re.compile(r"^\d{1,3}$")


@router.get("/{task_code}/subtasks/{subtask_id}/attempts/{n}")
def attempt_detail(task_code: str, subtask_id: str, n: str, request: Request,
                   p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    _sub(cfg, p, task_code, subtask_id)
    if not _N.match(n) or int(n) < 1:
        raise NotFound()
    d = safe_join(paths.subtask_dir(cfg, subtask_id), "llm")
    req = atomic_io.read_json(d / f"{int(n):03d}.request.json", None)
    resp = atomic_io.read_json(d / f"{int(n):03d}.response.json", None)
    if req is None and resp is None:
        raise NotFound()
    return {"request": req, "response": resp}


@router.post("/{task_code}/subtasks/{subtask_id}/regrade")
def regrade(task_code: str, subtask_id: str, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    _sub(cfg, p, task_code, subtask_id)
    res = store.regrade_subtask(cfg, task_code, subtask_id, p.user_id)
    executor_mod.poke(cfg)
    out = {"task_code": task_code, "subtask_id": subtask_id, "subtask_status": "queued",
           "status": res["status"]}
    if res["queued_until_resume"]:
        out["note"] = "queued_until_resume"
    return out


# ---- teacher review / summary
@router.post("/{task_code}/subtasks/{subtask_id}/review")
def submit_review(task_code: str, subtask_id: str, request: Request, body: dict = Body(...),
                  p: Principal = Depends(require_user)):
    """{"adopt_ai": true} or {"final_score": n, "items": [{"name", "content"}, ...]} (items optional)."""
    cfg = cfg_of(request)
    _sub(cfg, p, task_code, subtask_id)
    res = review_mod.submit(cfg, p, task_code, subtask_id, body)
    rev = res["revision"]
    audit(cfg, p.user_id, "review.submit", subtask_id,
          {"rev": rev["rev"], "adopted_ai": rev["adopted_ai"], "items": len(rev["items"] or [])},
          client_ip(request), p.role)
    results.generate_if_final(cfg, task_code, "review")
    store.bump(cfg, task_code)
    return {"review": res["review"], "revision": rev}


@router.get("/{task_code}/subtasks/{subtask_id}/review")
def get_review(task_code: str, subtask_id: str, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    meta = _sub(cfg, p, task_code, subtask_id)
    att, score = review_mod.ai_snapshot(cfg, subtask_id, meta)
    return {"review": review_mod.read(cfg, subtask_id), "ai_attempt": att, "ai_score": score,
            "can_review": meta.get("status") in review_mod.REVIEWABLE, "limits": cfg["review"]}


@router.get("/{task_code}/result")
def get_result(task_code: str, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    _load(cfg, p, task_code)
    return results.read_or_generate(cfg, task_code)


@router.post("/{task_code}/result/regenerate")
def regenerate_result(task_code: str, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    b = _load(cfg, p, task_code)
    if b.get("status") not in results.FINAL:
        raise AppError("result_not_ready", "the summary can be generated once the batch is completed or cancelled",
                       409, {"status": b.get("status")})
    doc = results.generate(cfg, task_code, "manual")
    audit(cfg, p.user_id, "result.regenerate", task_code, None, client_ip(request), p.role)
    return doc


# ---- SSE
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def progress_stream(cfg, task_code: str, is_disconnected, poll_s: float | None = None,
                          keepalive_s: float | None = None):
    """Yields SSE text: `event: progress` on every change, keepalive comments otherwise."""
    poll = POLL_S if poll_s is None else poll_s
    keep = KEEPALIVE_S if keepalive_s is None else keepalive_s
    last = None
    last_ver = -1
    since_keep = 0.0
    since_force = 0.0
    while True:
        if await is_disconnected():
            return
        ver = store.version(cfg, task_code)
        if last is None or ver != last_ver or since_force >= FORCE_REFRESH_S:
            snap = await asyncio.to_thread(store.progress, cfg, task_code)
            last_ver = ver
            since_force = 0.0
            if snap is None:
                return
            if snap != last:
                last = snap
                since_keep = 0.0
                yield _sse("progress", snap)
        if since_keep >= keep:
            since_keep = 0.0
            yield ": keepalive\n\n"
        await asyncio.sleep(poll)
        since_keep += poll
        since_force += poll


@router.get("/{task_code}/events")
def events(task_code: str, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    _load(cfg, p, task_code)
    return StreamingResponse(progress_stream(cfg, task_code, request.is_disconnected),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---- originals
_K = re.compile(r"^[1-9]\d{0,3}$")


def _image_entry(cfg, p: Principal, task_code: str, subtask_id: str, k: str):
    _load(cfg, p, task_code)
    if not paths.valid_subtask_id(subtask_id) or subtask_id[:12] != task_code or not _K.match(k):
        raise NotFound()
    m = store.read_meta(cfg, subtask_id)
    if m is None:
        raise NotFound()
    for im in m.get("images", []):
        if im.get("index") == int(k):
            return paths.subtask_dir(cfg, subtask_id), im
    raise NotFound()


_PRIVATE = {"Cache-Control": "private, max-age=3600"}


@router.get("/{task_code}/subtasks/{subtask_id}/image/{k}")
def original(task_code: str, subtask_id: str, k: str, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    sdir, im = _image_entry(cfg, p, task_code, subtask_id, k)
    f = safe_join(sdir, im["original"])
    if not f.is_file():
        raise NotFound()
    return FileResponse(f, media_type=im.get("mime") or "application/octet-stream", headers=_PRIVATE)


def _make_thumb(src, dst) -> None:
    try:
        from ...staging import images
        images.make_thumbnail(src, dst, 320)
        return
    except (ImportError, AttributeError):
        pass
    from PIL import Image, ImageOps                       # stub-tolerant fallback
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except Exception:
        pass
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((320, 320))
        im.save(dst, "JPEG", quality=80)


@router.get("/{task_code}/subtasks/{subtask_id}/image/{k}/thumb")
def original_thumb(task_code: str, subtask_id: str, k: str, request: Request,
                   p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    sdir, im = _image_entry(cfg, p, task_code, subtask_id, k)
    src = safe_join(sdir, im["original"])
    if not src.is_file():
        raise NotFound()
    dst = safe_join(sdir, "image", "derived", f"thumb_{int(k)}.jpg")
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            _make_thumb(src, tmp)
            os.replace(tmp, dst)
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise AppError("thumbnail_failed", "could not generate thumbnail", 500)
    return FileResponse(dst, media_type="image/jpeg", headers=_PRIVATE)
