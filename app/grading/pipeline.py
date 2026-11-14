"""Subtask processor (VLM route) + hook registration.

One LLM call per essay when nothing fails (the VLM reads the images directly; there is NO separate
transcription call). Every attempt is persisted as llm/NNN.request.json (before sending) and
llm/NNN.response.json (after). Class table: see classify.py.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import re
from pathlib import Path

from ..batches import executor, hooks, paths, store
from ..core import atomic_io, clock
from . import classify, fewshot, preprocess, revision
from .parse import parse_response

log = logging.getLogger("wm.grading")

EMPTY = "<empty>"
_ATT = re.compile(r"^(\d{3,})\.(request|response)\.json$")
_NOT_COUNTED = ("interrupted", "cancelled")


# ------------------------------------------------------------------ registration
def register() -> None:
    executor.set_processor(process)
    hooks.find_fewshot = fewshot.find_fewshot
    hooks.build_revision = revision.make_revision
    from ..ocr import service as ocr_service
    hooks.local_env_ok = ocr_service.local_env_ok


# ------------------------------------------------------------------ attempt files
def _llm_dir(sdir: Path) -> Path:
    return sdir / "llm"


def scan_attempts(sdir: Path) -> tuple[int, list[dict]]:
    """(max attempt number used by any request/response file, responses of NON-interrupted attempts in
    numeric order). Interrupted / cancelled attempts (response error code) never count."""
    d = _llm_dir(sdir)
    nums: set[int] = set()
    try:
        for name in os.listdir(d):
            m = _ATT.match(name)
            if m:
                nums.add(int(m.group(1)))
    except FileNotFoundError:
        pass
    counted = []
    for n in sorted(nums):
        r = atomic_io.read_json(d / f"{n:03d}.response.json", None)
        if not isinstance(r, dict):
            continue
        err = r.get("error")
        code = r.get("error_code") or (err.get("code") if isinstance(err, dict) else None)
        if code in _NOT_COUNTED or r.get("classification") in _NOT_COUNTED:
            continue
        r.setdefault("attempt", n)
        counted.append(r)
    return (max(nums) if nums else 0), counted


# ------------------------------------------------------------------ helpers
def _rel(sdir: Path, p: Path) -> str:
    return p.relative_to(sdir).as_posix()


def _load_images(sdir: Path, entries: list[dict]) -> list[dict]:
    out = []
    n = len(entries)
    for k, ent in enumerate(entries, start=1):
        data = (sdir / ent["path"]).read_bytes()
        out.append({"page": f"{k}/{n}", "k": k, "n": n, "path": ent["path"], "mime": "image/jpeg",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "b64": base64.b64encode(data).decode("ascii")})
    return out


def _text_messages(system: str, essay: str, retry_note: str | None) -> tuple[list, list]:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": essay}]
    if retry_note:
        msgs.append({"role": "user", "content": retry_note})
    return msgs, [dict(m) for m in msgs]


def _messages(system: str, imgs: list[dict], retry_note: str | None) -> tuple[list, list]:
    """-> (messages actually sent, messages as stored: image data replaced by image_ref)."""
    sent_parts, saved_parts = [], []
    for im in imgs:
        text = {"type": "text", "text": f"第{im['k']}页/共{im['n']}页"}
        sent_parts += [text, {"type": "image_url", "image_url": {"url": f"data:{im['mime']};base64,{im['b64']}"}}]
        saved_parts += [text, {"type": "image_ref", "path": im["path"], "sha256": im["sha256"],
                               "mime": im["mime"], "page": im["page"]}]
    sent = [{"role": "system", "content": system}, {"role": "user", "content": sent_parts}]
    saved = [{"role": "system", "content": system}, {"role": "user", "content": saved_parts}]
    if retry_note:
        extra = {"role": "user", "content": retry_note}
        sent.append(extra)
        saved.append(dict(extra))
    return sent, saved


def _retry_note(code: str, detail: str | None, keys: dict) -> str:
    req = "、".join(keys["required_keys"])
    return (f"你上一次的回复无法被系统接受（{code}：{detail or ''}）。请重新回答：只输出一个JSON对象，"
            f"不要输出任何其他文字；必须包含字段：{req}；其中“{keys['score_key']}”必须是0到"
            f"{keys['full_score']:g}之间的数字。")


def _template_keys(cfg, rev: dict, full_score) -> dict:
    keys = {"score_key": "判分", "score_aliases": [], "criteria_key": "判据",
            "required_keys": ["判分", "判据"], "full_score": float(full_score or 100)}
    try:
        from ..prompts import store as pstore
        t = rev.get("template") or {}
        v = pstore.get_version(cfg, t.get("id"), t.get("version"))
        for k in ("score_key", "score_aliases", "criteria_key", "required_keys"):
            if v.get(k) is not None:
                keys[k] = v[k]
    except Exception:
        log.warning("template version unavailable, default keys used")
    return keys


def _find_rev(batch: dict, rev_no: int) -> dict | None:
    revs = batch.get("config_revisions") or []
    return next((r for r in revs if r.get("rev") == rev_no), None) or (revs[rev_no - 1] if 0 < rev_no <= len(revs) else None)


async def _report_breaker(cfg, task_code: str, cls: str) -> None:
    try:
        from ..batches import breaker
    except ImportError:
        return
    try:
        await asyncio.to_thread(breaker.record_api_call, cfg, task_code, classify.api_outcome_for_breaker(cls))
    except Exception:
        log.exception("breaker report failed")


def _write_ocr_once(sdir: Path, entries: list[dict]) -> None:
    p = sdir / "ocr.json"
    if p.exists():
        return
    # EXTENSION POINT (M4): when vlm_transcription_choice == "local_transcription" the local OCR result
    # (transcription text, lines, engine info, transcription_source "local_ocr") is produced before the
    # LLM call and written here instead of this "<empty>" placeholder.
    doc = {"schema_version": 1, "mode": "vlm", "created_at": clock.iso(),
           "transcription": EMPTY, "transcription_source": "none",
           "pages": [{"index": k, "image_ref": {"path": e["path"], "sha256": e["sha256"], "mime": "image/jpeg"},
                      "layout_reliable": False, "lines": []} for k, e in enumerate(entries, start=1)],
           "engine": {"local_engine": None, "name": None, "version": None, "params": None},
           "warnings": []}
    atomic_io.write_json_atomic(p, doc)


def _meta_note(cfg, sid: str, event: str, detail: dict, config_rev: int | None = None,
               attempts: int | None = None) -> None:
    def fn(m: dict) -> None:
        if config_rev is not None:
            m["config_rev"] = config_rev
        if attempts is not None:
            m["llm_attempts"] = attempts
        store.append_timeline(m, event, None, detail)
    try:
        store.update_meta(cfg, sid, fn)
    except Exception:
        log.warning("could not update meta of %s", sid)


def _ensure_run_base(cfg, sid: str, counted: int) -> int:
    def fn(m: dict) -> int:
        if m.get("run_base") is None:
            m["run_base"] = counted
        return int(m["run_base"])
    return store.update_meta(cfg, sid, fn)


def _set_run_base(cfg, sid: str, value: int) -> None:
    def fn(m: dict) -> None:
        m["run_base"] = value
    store.update_meta(cfg, sid, fn)


def _detail_for_raw(raw) -> str:
    base = classify.error_detail_for(raw) or ""
    extra = getattr(raw, "error_detail", None)
    return f"{base}: {extra}" if extra and base else (base or extra or "")


# ------------------------------------------------------------------ processing
async def process(ctx) -> executor.Outcome:
    return await _process_llm(ctx)


async def _process_llm(ctx) -> executor.Outcome:
    cfg, sid = ctx.cfg, ctx.subtask_id
    tc = sid[:12]
    sdir = paths.subtask_dir(cfg, sid)
    batch = ctx.batch
    rev = _find_rev(batch, ctx.config_rev)
    if rev is None:
        batch = store.read_batch(cfg, tc) or batch
        rev = _find_rev(batch, ctx.config_rev)
    if rev is None:
        return executor.Outcome("failed", "provider_unavailable")
    from ..providers import client, store as provstore
    provider = provstore.get_provider(cfg, rev.get("provider_id"))
    if provider is None or provider.get("deleted"):
        return executor.Outcome("failed", "provider_unavailable")

    text_route = batch.get("route") == "local_ocr"
    if text_route or batch.get("vlm_transcription_choice") == "local_transcription":
        # local recognition happens BEFORE any LLM call; ocr.json is written by the OCR service
        from ..ocr import service as ocr_service
        try:
            ocr_doc = await asyncio.to_thread(ocr_service.run_for_subtask, cfg, sid,
                                              batch.get("local_engine") or cfg["default_local_engine"],
                                              batch.get("sheet_template"))
        except ocr_service.OcrEmpty:
            return executor.Outcome("failed", "ocr_empty")
        except ocr_service.OcrFailed as e:
            return executor.Outcome("failed", "ocr_failed", [str(e)[:200]])
        essay_text = ocr_doc.get("transcription") or ""
    else:
        essay_text = None
    try:
        if text_route:
            entries = []
        else:
            entries = await asyncio.to_thread(preprocess.prepare_vlm_images, cfg, sid)
            await asyncio.to_thread(_write_ocr_once, sdir, entries)
    except preprocess.PreprocessError:
        return executor.Outcome("failed", "preprocess_failed")
    except OSError:
        return executor.Outcome("failed", "preprocess_failed")

    keys = _template_keys(cfg, rev, batch.get("full_score"))
    retry_max = max(0, int(cfg["llm"]["retry_max"]))
    budget = 1 + retry_max

    max_n, counted = await asyncio.to_thread(scan_attempts, sdir)
    run_base = await asyncio.to_thread(_ensure_run_base, cfg, sid, len(counted))
    if len(counted) - run_base >= budget and (not counted or counted[-1].get("classification") != "A"):
        # stale budget (not a format-retry situation): start a fresh budget
        run_base = len(counted)
        await asyncio.to_thread(_set_run_base, cfg, sid, run_base)
    window = counted[run_base:]
    last_code, last_detail = None, None
    regex_hit = None                       # (attempt, score, content)
    for r in window:
        if r.get("classification") == "A":
            last_code, last_detail = r.get("error_code"), r.get("error_detail")
        if r.get("parse_level") == "regex" and isinstance(r.get("regex_score"), (int, float)):
            regex_hit = (r["attempt"], float(r["regex_score"]), r.get("content") or "")
    used = len(window)
    llm = cfg["llm"]
    params = dict(rev.get("params") or {})
    model = rev.get("model") or provider.get("default_model")
    d = _llm_dir(sdir)
    d.mkdir(parents=True, exist_ok=True)
    retry_note = _retry_note(last_code, last_detail, keys) if used > 0 and last_code else None

    while used < budget:
        n = max_n + 1
        max_n = n
        imgs = await asyncio.to_thread(_load_images, sdir, entries)
        provider = provstore.get_provider(cfg, rev.get("provider_id"))        # current key (rotation)
        if provider is None or provider.get("deleted"):
            return executor.Outcome("failed", "provider_unavailable")
        if text_route:
            sent, saved = _text_messages(rev.get("rendered_prompt") or "", essay_text or "", retry_note)
        else:
            sent, saved = _messages(rev.get("rendered_prompt") or "", imgs, retry_note)
        atomic_io.write_json_atomic(d / f"{n:03d}.request.json", {
            "attempt": n, "created_at": clock.iso(),
            "provider_alias": provider.get("alias") or rev.get("provider_alias"),
            "key_last4": (provider.get("api_key") or "")[-4:], "model": model, "params": params,
            "config_rev": ctx.config_rev, "template": rev.get("template"),
            "fewshot_ids": list(batch.get("fewshot_ids") or []), "messages": saved})
        started = clock.iso()
        raw = await client.post_chat(provider, model, sent, params, float(llm["timeout_s"]))
        cls = classify.classify_call(raw)
        await _report_breaker(cfg, tc, cls)
        del sent, imgs

        resp = {"attempt": n, "started_at": started, "latency_ms": raw.latency_ms,
                "http_status": raw.http_status,
                "raw": raw.body if raw.body is not None else raw.text,
                "content": None, "thinking": None, "parsed": None, "parse_level": None,
                "usage": None, "error": None, "classification": cls, "error_code": None, "error_detail": None}
        pr = None
        if cls == "ok":
            resp["usage"] = client.usage(raw.body) if raw.body is not None else None
            if raw.body is not None:
                content, thinking, finish = client.extract_message(raw.body)
            else:
                content, thinking, finish = "", None, None
            resp["content"], resp["thinking"] = content, thinking
            if raw.body is None and raw.text.strip():
                code, detail = "json_parse_failed", "response body is not JSON"
            elif finish == "content_filter":
                code, detail = "refusal_or_safety", "finish_reason=content_filter"
            else:
                pr = parse_response(content, score_key=keys["score_key"], score_aliases=keys["score_aliases"],
                                    criteria_key=keys["criteria_key"], required_keys=keys["required_keys"],
                                    full_score=keys["full_score"])
                code, detail = pr.error_code, pr.error_detail
                if code and finish == "length" and code in ("json_parse_failed", "missing_required_field"):
                    detail = f"{detail} (output truncated: finish_reason=length)"
            if pr is not None:
                resp["parse_level"] = pr.level
                resp["parsed"] = pr.obj
                if pr.level == "regex" and pr.score is not None:
                    resp["regex_score"] = pr.score
            if code:
                resp["classification"] = cls = "A"
                resp["error_code"], resp["error_detail"] = code, detail
                resp["error"] = {"code": code, "detail": detail}
        elif cls == "B":
            resp["error_code"] = "api_error"
            resp["error_detail"] = _detail_for_raw(raw)
            resp["error"] = {"code": "api_error", "detail": resp["error_detail"]}
        else:
            resp["error_code"] = "request_rejected"
            resp["error_detail"] = _detail_for_raw(raw)
            resp["error"] = {"code": "request_rejected", "detail": resp["error_detail"]}
        atomic_io.write_json_atomic(d / f"{n:03d}.response.json", resp)
        used += 1
        await asyncio.to_thread(_meta_note, cfg, sid, "llm_attempt",
                                {"attempt": n, "classification": cls, "error_code": resp["error_code"]},
                                ctx.config_rev, run_base + used)

        if cls == "B":
            return executor.Outcome("api_error", "api_error")
        if cls == "C":
            return executor.Outcome("failed", "request_rejected")
        if cls == "ok":
            doc = {"schema_version": 1, "attempt": n, "parse_level": pr.level, "result": pr.obj,
                   "score": pr.score, "warnings": [], "raw_text": None, "adopted_at": clock.iso()}
            atomic_io.write_json_atomic(sdir / "grading.json", doc)
            return executor.Outcome("done")
        # class A
        last_code, last_detail = resp["error_code"], resp["error_detail"]
        if resp.get("regex_score") is not None:
            regex_hit = (n, float(resp["regex_score"]), resp["content"] or "")
        retry_note = _retry_note(last_code, last_detail, keys)

    # budget exhausted
    if regex_hit is not None:
        n, score, text = regex_hit
        doc = {"schema_version": 1, "attempt": n, "parse_level": "regex_fallback",
               "result": {keys["score_key"]: score}, "score": score, "warnings": ["regex_fallback"],
               "raw_text": text, "adopted_at": clock.iso()}
        atomic_io.write_json_atomic(sdir / "grading.json", doc)
        return executor.Outcome("done", None, ["regex_fallback"])
    return executor.Outcome("failed", last_code or "json_parse_failed")
