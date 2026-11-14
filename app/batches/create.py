"""Batch creation: validation, grouping resolution, numbering, moving originals out of staging.

Commit protocol: allocate numbers -> create dirs -> MOVE originals -> write every meta.json ->
write batch.json LAST (commit point) -> release staging. A directory without batch.json is a
half-created batch and is deleted by startup recovery (its number is never reused).
"""
from __future__ import annotations

import logging
import math
import os
import shutil
from pathlib import Path

from ..core import atomic_io, clock, counters
from ..errors import AppError
from ..staging import uploads
from . import hooks, paths, store

log = logging.getLogger("wm.batches")

ROUTES = ("local_ocr", "vlm")
CHOICES = ("local_transcription", "skip_fewshot", "history_fewshot")
ENGINES = ("paddle",)
SCOPES = ("own", "all")


def _bad(field: str, msg: str) -> AppError:
    return AppError("validation_error", msg, 422, {"field": field})


def _text(body: dict, field: str, strict: bool) -> str:
    v = body.get(field)
    if v is None or (isinstance(v, str) and not v.strip()):
        if strict:
            raise _bad(field, f"{field} is required")
        return ""
    if not isinstance(v, str):
        raise _bad(field, f"{field} must be a string")
    return v.strip()


def _write_batch_json(path: Path, data: dict) -> None:   # separate function so tests can simulate a crash
    atomic_io.write_json_atomic(path, data)


class Prepared:
    pass


def _prepare(cfg, user, body: dict, *, strict: bool) -> Prepared:
    p = Prepared()
    upload_id = body.get("upload_id")
    if not isinstance(upload_id, str) or not upload_id:
        raise _bad("upload_id", "upload_id is required")
    p.upload_id = upload_id
    p.topic = _text(body, "topic", strict)
    p.rubric = _text(body, "rubric", strict)
    p.genre = _text(body, "genre", strict)
    fs = body.get("full_score")
    if strict:
        if isinstance(fs, bool) or not isinstance(fs, (int, float)) or not math.isfinite(fs) or fs <= 0:
            raise _bad("full_score", "full_score must be a number > 0")
    p.full_score = fs
    p.route = body.get("route")
    if p.route not in ROUTES:
        raise AppError("invalid_route", "route must be local_ocr or vlm", 422)
    choice = body.get("vlm_transcription_choice")
    if p.route == "vlm":
        if choice is None:
            if strict:
                raise AppError("transcription_choice_required",
                               "choose how handwriting is transcribed for the vlm route", 422)
        elif choice not in CHOICES:
            raise AppError("invalid_transcription_choice", f"vlm_transcription_choice must be one of {CHOICES}", 422)
    else:
        choice = None
    p.choice = choice
    p.needs_local = p.route == "local_ocr" or choice == "local_transcription"
    engine = body.get("local_engine") or cfg["default_local_engine"]
    if p.needs_local:
        if engine not in ENGINES:
            raise AppError("invalid_local_engine", f"local_engine must be one of {ENGINES}", 422)
        p.local_engine = engine
    else:
        p.local_engine = None
    p.provider_id = body.get("provider_id") or ""
    if strict:
        hooks.validate_provider_and_template(cfg, user, body)       # may fill defaults (model) into body
    p.model = body.get("model")
    p.template_id = body.get("template_id") or ""
    p.template_version = body.get("template_version")
    p.scope = body.get("fewshot_scope") or "own"
    if p.scope not in SCOPES:
        raise AppError("invalid_fewshot_scope", "fewshot_scope must be own or all", 422)
    p.grouping_req = body.get("grouping") if isinstance(body.get("grouping"), dict) else {}
    return p


def _local_env(cfg, p) -> tuple[bool, str | None]:
    if not p.needs_local:
        return True, None
    ok, reason = hooks.local_env_ok(cfg)
    if ok and p.local_engine:
        from ..ocr import service            # the chosen engine must be usable too, not only the default one
        if hooks.local_env_ok is service.local_env_ok:
            ok, reason = service.engine_status(cfg, p.local_engine)
    return bool(ok), reason


def _draft(p) -> dict:
    return {"topic": p.topic, "rubric": p.rubric, "genre": p.genre, "full_score": p.full_score, "route": p.route,
            "provider_id": p.provider_id, "model": p.model}


def precheck(cfg, user, body: dict) -> dict:
    """Side-effect free; reports instead of rejecting (apart from malformed input / 404)."""
    p = _prepare(cfg, user, body, strict=False)
    hooks.require_visible_refs(cfg, user, body)
    ok, reason = hooks.local_env_ok(cfg)
    from ..staging import grouping
    manifest = uploads.get_manifest(cfg, user.user_id, p.upload_id)
    result = grouping.resolve_groups(cfg, manifest, dict(p.grouping_req))
    if p.scope == "all" and not user.is_root:
        ids = []
    else:
        ids = hooks.find_fewshot(cfg, user.user_id, p.scope, _draft(p)) or []
    approved = body.get("zero_shot_approved") is True
    sheet_ok, sheet_reason = True, None
    if p.needs_local:
        try:
            _sheet_snapshot(cfg, user, body)
        except AppError as e:
            sheet_ok, sheet_reason = False, e.code
    return {
        "sheet_template_ok": sheet_ok, "sheet_template_reason": sheet_reason,
        "fewshot_available": bool(ids), "fewshot_count": len(ids),
        "needs_third_party_ack": bool(hooks.needs_third_party_ack(cfg, user, p.provider_id)),
        "local_env_ok": bool(ok), "local_env_reason": reason,
        "zero_shot_consent_required": (not ids) and (not approved) and p.choice != "skip_fewshot",
        "grouping": result,
    }


def _order_value(f: dict, basis: str, position: int):
    if basis == "filename":
        return f.get("name_seq")
    if basis == "exif":
        return f.get("exif_time")
    if basis == "mtime":
        return f.get("client_last_modified")
    if basis == "manual_user":
        return position
    return None


def create_batch(cfg, user, body: dict) -> dict:
    p = _prepare(cfg, user, body, strict=True)
    ok, reason = _local_env(cfg, p)
    if not ok:
        raise AppError("local_env_unavailable", reason or "local OCR environment is unavailable", 409)
    p.sheet = _sheet_snapshot(cfg, user, body) if p.needs_local else None
    from ..staging import grouping
    manifest = uploads.get_manifest(cfg, user.user_id, p.upload_id)          # 404 if missing / not owner
    result = grouping.resolve_groups(cfg, manifest, dict(p.grouping_req))
    if not result.get("can_submit"):
        raise AppError("grouping_invalid", "the photo grouping is not valid", 422, result.get("problems"))
    if result.get("mode") == "manual" and p.grouping_req.get("confirmed") is not True:
        raise AppError("grouping_unconfirmed", "confirm the manual grouping before creating the batch", 422,
                       result.get("problems"))
    if hooks.needs_third_party_ack(cfg, user, p.provider_id) and body.get("third_party_ack") is not True:
        raise AppError("third_party_ack_required",
                       "acknowledge that essay images are sent to a third-party provider", 409)
    if p.scope == "all" and not user.is_root:
        raise AppError("forbidden", "fewshot_scope=all is only available to root", 403)
    if p.choice == "skip_fewshot":
        fewshot_ids: list[str] = []
        approved = True
    else:
        fewshot_ids = list(hooks.find_fewshot(cfg, user.user_id, p.scope, _draft(p)) or [])
        approved = body.get("zero_shot_approved") is True
        if not fewshot_ids and not approved:
            raise AppError("zero_shot_consent_required",
                           "no few-shot examples found; confirm grading without examples", 409)
    revision = hooks.build_revision(cfg, user, _revision_body(body, p), fewshot_ids, 1, "initial")
    if body.get("third_party_ack") is True:
        _record_ack(cfg, user, p.provider_id)
    return _commit(cfg, user, body, p, manifest, result, fewshot_ids, approved, revision)


def _sheet_snapshot(cfg, user, body: dict) -> dict:
    """Local recognition needs an answer-sheet template (blank sheets) to remove printed matter."""
    from ..sheets import store as sheets
    return sheets.snapshot_for_batch(cfg, user, body.get("sheet_template_id"))


def _revision_body(body: dict, p) -> dict:
    # topic/rubric/genre/full_score are needed to render {{TOPIC}} etc. into the system prompt
    return {"provider_id": p.provider_id, "model": p.model, "template_id": p.template_id,
            "template_version": p.template_version, "params": body.get("params") or {},
            "topic": p.topic, "rubric": p.rubric, "genre": p.genre, "full_score": p.full_score}


def _record_ack(cfg, user, provider_id: str) -> None:
    """The caller confirmed the third-party provider: remember it (idempotent). Degrades silently while
    the providers module is not available or the provider is unknown."""
    try:
        from ..providers import store as pstore
        from ..providers import third_party
        prov = pstore.get_provider(cfg, provider_id)
        if prov is not None:
            third_party.record_ack(cfg, user.user_id, prov)
    except (ImportError, AttributeError):
        pass


def _commit(cfg, user, body, p, manifest, result, fewshot_ids, approved, revision) -> dict:
    files = {f["file_id"]: f for f in manifest["files"]}
    essays = result["essays"]
    mode = result["mode"]
    basis = "manual_user" if mode == "manual" else ("single" if mode == "single" else result["order_basis"])
    task_code, sub_ids = counters.allocate_batch(cfg, len(essays))
    bdir = paths.batch_dir(cfg, task_code)
    moved: list[tuple[Path, Path]] = []
    position = 0
    try:
        bdir.mkdir(parents=True, exist_ok=False)
        grouping_out = []
        for sid, essay in zip(sub_ids, essays):
            sdir = paths.subtask_dir(cfg, sid)
            (sdir / "image").mkdir(parents=True, exist_ok=True)
            images = []
            gl = []
            for k, fid in enumerate(essay, start=1):
                position += 1
                f = files[fid]
                ext = str(f.get("ext") or "").lstrip(".").lower() or "bin"
                rel = f"image/original_{k}.{ext}"
                src = uploads.file_abspath(cfg, p.upload_id, fid)
                dst = sdir / rel
                try:
                    os.replace(src, dst)
                except OSError:
                    shutil.move(str(src), str(dst))
                moved.append((src, dst))
                images.append({"index": k, "orig_name": f.get("orig_name"), "original": rel,
                               "sha256": f.get("sha256"), "mime": f.get("mime"), "width": f.get("width"),
                               "height": f.get("height"), "order_basis": basis,
                               "order_value": _order_value(f, basis, position),
                               "client_last_modified": f.get("client_last_modified"),
                               "exif_time": f.get("exif_time"), "derived": []})
                gl.append({"file_id": fid, "orig_name": f.get("orig_name")})
            grouping_out.append(gl)
            meta = {"schema_version": 1, "subtask_id": sid, "task_code": task_code,
                    "index": int(sid[12:]), "owner_id": user.user_id, "status": "queued", "fail_reason": None,
                    "attempts": 0, "config_rev": None, "review_status": "none", "warnings": [],
                    "images": images, "local_ocr_state": "n/a", "timeline": []}
            store.append_timeline(meta, "created", None, {})
            atomic_io.write_json_atomic(sdir / "meta.json", meta)
        now = clock.iso()
        batch = {
            "schema_version": 1, "task_code": task_code, "owner_id": user.user_id, "created_at": now,
            "topic": p.topic, "rubric": p.rubric, "genre": p.genre, "full_score": p.full_score,
            "language": "en", "route": p.route, "local_engine": p.local_engine,
            "sheet_template": getattr(p, "sheet", None),
            "vlm_transcription_choice": p.choice,
            "grouping": {"multi_photo": bool(p.grouping_req.get("multi_photo")),
                         "order_basis": result.get("order_basis"), "mode": mode,
                         "essays": grouping_out, "confirmed": p.grouping_req.get("confirmed") is True},
            "fewshot_scope": p.scope, "fewshot_ids": fewshot_ids, "zero_shot_approved": approved,
            "status": "queued", "pause_reason": None, "api_error_streak": 0,
            "breaker": {"tripped_reason": None, "samples": []},
            "config_revisions": [revision],
            "timeline": [],
        }
        store.append_timeline(batch, "created", user.user_id, {"subtasks": len(sub_ids)})
        _write_batch_json(bdir / "batch.json", batch)           # commit point
    except Exception:
        # roll back: put originals back into staging and drop the half-created dir
        for src, dst in reversed(moved):
            try:
                os.replace(dst, src)
            except OSError:
                log.error("could not restore a staged original during rollback of %s", task_code)
        shutil.rmtree(bdir, ignore_errors=True)
        raise
    store.track(cfg, task_code, "queued")
    store.bump(cfg, task_code)
    try:
        uploads.release(cfg, p.upload_id)
    except Exception:
        log.exception("staging release failed for %s", task_code)
    try:
        from . import executor
        executor.poke(cfg)
    except Exception:
        pass
    return {"task_code": task_code, "status": "queued", "subtask_ids": sub_ids}
