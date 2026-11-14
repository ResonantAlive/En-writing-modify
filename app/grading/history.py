"""Read-only view over data/history for few-shot selection and prompt-assist.

Per-subtask files read (all optional): meta.json, ocr.json (transcription), grading.json (result),
llm/NNN.response.json (parsed result of a specific attempt), review.json (the teacher review).

review.json (written by the review milestone) is tolerated in two shapes:
  {"final_score","items","adopted_ai","ai_attempt",...}                     (latest review inline)
  {"revisions":[{...same...}, ...]}  or  {"history":[...]}                   (latest = last entry)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

from ..batches import paths
from ..core import atomic_io

log = logging.getLogger("wm.grading")

_ATTEMPT_RE = re.compile(r"^(\d{3,})\.response\.json$")


def latest_review(doc) -> dict | None:
    if not isinstance(doc, dict):
        return None
    for key in ("revisions", "history", "reviews"):
        lst = doc.get(key)
        if isinstance(lst, list):
            lst = [x for x in lst if isinstance(x, dict)]
            return dict(lst[-1]) if lst else None
    if any(k in doc for k in ("final_score", "items", "adopted_ai")):
        return dict(doc)
    return None


def attempt_parsed(sdir: Path, n) -> dict | None:
    """`parsed` of attempt n (the AI result object), or None."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return None
    doc = _read(sdir / "llm" / f"{n:03d}.response.json")
    p = doc.get("parsed") if isinstance(doc, dict) else None
    return p if isinstance(p, dict) else None


def _read(path: Path):
    try:
        return atomic_io.read_json(path, None)
    except Exception:
        return None


def _template_for(batch: dict, meta: dict) -> dict | None:
    revs = batch.get("config_revisions") or []
    if not revs:
        return None
    want = meta.get("config_rev")
    rev = next((r for r in revs if r.get("rev") == want), None) or revs[-1]
    t = rev.get("template")
    if isinstance(t, dict) and t.get("id"):
        return {"id": t.get("id"), "version": t.get("version")}
    return None


def load_record(batch: dict, sdir: Path) -> dict | None:
    """Full (internal) record of one subtask directory; None when unreadable."""
    meta = _read(sdir / "meta.json")
    if not isinstance(meta, dict):
        return None
    ocr = _read(sdir / "ocr.json")
    grading = _read(sdir / "grading.json")
    review_doc = _read(sdir / "review.json")
    rv = latest_review(review_doc)
    transcription = ocr.get("transcription") if isinstance(ocr, dict) else None
    ai_result = grading.get("result") if isinstance(grading, dict) and isinstance(grading.get("result"), dict) else None
    review = None
    if rv is not None:
        review = {"final_score": rv.get("final_score"), "items": rv.get("items"),
                  "adopted_ai": bool(rv.get("adopted_ai"))}
    return {
        "task_code": batch.get("task_code"), "subtask_id": meta.get("subtask_id"),
        "owner_id": batch.get("owner_id") or meta.get("owner_id"),
        "topic": batch.get("topic"), "rubric": batch.get("rubric"), "genre": batch.get("genre"),
        "full_score": batch.get("full_score"),
        "transcription": transcription if isinstance(transcription, str) else "",
        "ai_result": ai_result, "review": review,
        "template": _template_for(batch, meta),
        # internal extras (stripped by iter_records)
        "_dir": sdir, "_created_at": batch.get("created_at") or "",
        "_review_at": (rv or {}).get("saved_at") or (rv or {}).get("at") or "",
        "_ai_attempt": (rv or {}).get("ai_attempt"),
    }


def iter_records_ext(cfg, owner_id: str | None) -> Iterator[dict]:
    for d in paths.iter_task_dirs(cfg):
        try:
            batch = atomic_io.read_json(d / "batch.json", None)
        except Exception:
            continue
        if not isinstance(batch, dict):
            continue
        if owner_id is not None and batch.get("owner_id") != owner_id:
            continue
        for sd in paths._subdirs(d, paths.SUB_RE):
            try:
                rec = load_record(batch, sd)
            except Exception:
                log.warning("unreadable history directory skipped: %s", sd.name)
                continue
            if rec is not None:
                yield rec


def iter_records(cfg, owner_id: str | None) -> Iterator[dict]:
    """Public record shape (no paths, no internals, never thinking / images / keys)."""
    for rec in iter_records_ext(cfg, owner_id):
        yield {k: v for k, v in rec.items() if not k.startswith("_")}


def get_record_ext(cfg, subtask_id: str) -> dict | None:
    try:
        sdir = paths.subtask_dir(cfg, subtask_id)
        batch = atomic_io.read_json(paths.batch_dir(cfg, subtask_id[:12]) / "batch.json", None)
    except Exception:
        return None
    if not isinstance(batch, dict) or not sdir.is_dir():
        return None
    try:
        return load_record(batch, sdir)
    except Exception:
        return None
