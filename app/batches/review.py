"""Teacher review: review.json is append-only (revisions[]); the newest revision is the one in force.

A revision is either "adopt the AI result" (adopted_ai: true, final score = AI score, no items copied) or a
teacher score in [0, full_score] with optional items (name + content). Essays whose AI grading failed can be
scored by hand (no AI snapshot, diff null). The AI's own files (grading.json, llm/) are never modified.

Limits (settings "review"): at most 200 items, item content <= 10000 and name <= 100 characters, all counted in
Unicode code points (Python len); the request body has its own cap (max_body_mb, default 8 MB) enforced by the
body-limit middleware.
"""
from __future__ import annotations

import math

from ..auth.deps import Principal
from ..core import atomic_io, clock
from ..errors import AppError, NotFound
from . import paths, store

REVIEWABLE = ("done", "failed", "api_error")


def _num(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def ai_snapshot(cfg, subtask_id: str, meta: dict) -> tuple[int | None, float | None]:
    """(attempt, score) of the AI result currently in force; (None, None) when the AI grading did not succeed."""
    if meta.get("status") != "done":
        return None, None
    g = atomic_io.read_json(paths.subtask_dir(cfg, subtask_id) / "grading.json", None)
    if not isinstance(g, dict):
        return None, None
    score = _num(g.get("score"))
    return (g.get("attempt") if score is not None else None), score


def _bad(code: str, msg: str, detail: dict | None = None) -> AppError:
    return AppError(code, msg, 422, detail)


def _text_ok(s: str) -> bool:
    return not any(0xD800 <= ord(c) <= 0xDFFF for c in s)      # lone surrogates cannot be stored as UTF-8


def validate_items(cfg, items) -> list[dict] | None:
    """None / [] -> no items. Otherwise [{"name", "content"}] within the configured limits (code points)."""
    if items is None:
        return None
    if not isinstance(items, list):
        raise _bad("invalid_items", "items must be a list of {name, content}")
    lim = cfg["review"]
    max_items, max_chars, max_name = int(lim["max_items"]), int(lim["max_item_chars"]), int(lim["max_name_chars"])
    if len(items) > max_items:
        raise _bad("too_many_items", f"at most {max_items} items", {"max": max_items, "got": len(items)})
    out = []
    for k, it in enumerate(items):
        if not isinstance(it, dict):
            raise _bad("invalid_item", f"item {k + 1} must be an object with name and content", {"index": k})
        name = it.get("name")
        name = "" if name is None else name
        content = it.get("content")
        if not isinstance(name, str) or not isinstance(content, str):
            raise _bad("invalid_item", f"item {k + 1}: name and content must be text", {"index": k})
        if not _text_ok(name) or not _text_ok(content):
            raise _bad("invalid_item", f"item {k + 1} contains invalid characters", {"index": k})
        if len(name) > max_name:
            raise _bad("item_name_too_long", f"item {k + 1}: name is longer than {max_name} characters",
                       {"index": k, "max": max_name, "got": len(name)})
        if len(content) > max_chars:
            raise _bad("item_too_long", f"item {k + 1}: content is longer than {max_chars} characters",
                       {"index": k, "max": max_chars, "got": len(content)})
        if not content.strip():
            raise _bad("item_empty", f"item {k + 1}: content is empty", {"index": k})
        out.append({"name": name, "content": content})
    return out or None


def read(cfg, subtask_id: str) -> dict | None:
    doc = atomic_io.read_json(paths.subtask_dir(cfg, subtask_id) / "review.json", None)
    return doc if isinstance(doc, dict) else None


def outdated(rv: dict | None, ai_attempt) -> bool:
    """The AI result was regraded after this review (the review was made against another attempt)."""
    return bool(rv) and rv.get("ai_attempt") != ai_attempt


def latest(doc: dict | None) -> dict | None:
    revs = (doc or {}).get("revisions")
    return revs[-1] if isinstance(revs, list) and revs else None


def submit(cfg, p: Principal, task_code: str, subtask_id: str, body: dict) -> dict:
    """Append one revision. Returns {"review": whole document, "revision": the new entry}."""
    if not isinstance(body, dict):
        raise _bad("validation_error", "request body must be a JSON object")
    adopt = body.get("adopt_ai", False)
    if not isinstance(adopt, bool):
        raise _bad("validation_error", "adopt_ai must be true or false", {"field": "adopt_ai"})
    items_in = body.get("items")
    if adopt and items_in:
        raise _bad("adopt_with_items", "adopting the AI result copies no items; submit a teacher score instead")
    items = None if adopt else validate_items(cfg, items_in)
    score_in = body.get("final_score")
    if not paths.valid_subtask_id(subtask_id) or subtask_id[:12] != task_code:
        raise NotFound()

    def fn(b: dict) -> dict:
        m = atomic_io.read_json(store.meta_json(cfg, subtask_id), None)
        if m is None:
            raise NotFound()
        if m.get("status") not in REVIEWABLE:
            raise AppError("review_not_allowed", f"a subtask that is {m.get('status')} cannot be reviewed", 409,
                           {"status": m.get("status")})
        full = _num(b.get("full_score"))
        ai_attempt, ai_score = ai_snapshot(cfg, subtask_id, m)
        if adopt:
            if ai_score is None:
                raise AppError("no_ai_result", "there is no successful AI result to adopt", 409)
            final = ai_score
        else:
            final = _num(score_in)
            if final is None:
                raise _bad("final_score_required", "final_score is required (a number)", {"field": "final_score"})
            if final < 0 or (full is not None and final > full):
                raise _bad("score_out_of_range", f"final_score must be between 0 and {b.get('full_score')}",
                           {"min": 0, "max": b.get("full_score"), "got": final})
        sdir = paths.subtask_dir(cfg, subtask_id)
        doc = read(cfg, subtask_id) or {"schema_version": 1, "task_code": task_code, "subtask_id": subtask_id,
                                        "revisions": []}
        if not isinstance(doc.get("revisions"), list):
            raise AppError("review_corrupt", "review.json is not readable; fix it on the server", 500)
        rev = {"rev": len(doc["revisions"]) + 1, "at": clock.iso(), "teacher_id": p.user_id,
               "final_score": final, "items": items, "adopted_ai": adopt,
               "ai_attempt": ai_attempt, "ai_score": ai_score,
               "diff": None if ai_score is None else round(final - ai_score, 6),
               "full_score": b.get("full_score"), "config_rev": m.get("config_rev")}
        doc["revisions"].append(rev)
        atomic_io.write_json_atomic(sdir / "review.json", doc)
        m["review_status"] = "reviewed"
        store.append_timeline(m, "reviewed", p.user_id, {"rev": rev["rev"], "adopted_ai": adopt})
        atomic_io.write_json_atomic(store.meta_json(cfg, subtask_id), m)
        return {"review": doc, "revision": rev, "batch_status": b.get("status")}

    return store.update_batch(cfg, task_code, fn)
