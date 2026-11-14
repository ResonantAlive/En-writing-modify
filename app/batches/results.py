"""Batch summary result-<task_code>.json (one file per batch, next to batch.json).

Generated when the batch becomes completed or cancelled, again after every teacher review of such a batch, and
on request. Before each overwrite the previous file is copied to result_backup/result-<task_code>.<stamp>.json;
the newest `result_backups_keep` (default 20) backups are kept.

Per subtask: final score and its source (teacher | ai | null), AI score (current grading), teacher score, diff
(as recorded by the latest review, i.e. against the AI score the teacher saw), `review_outdated` (the essay was
regraded after that review), status and the relative paths of its files. Status:
  done            has a teacher review (including hand scoring after a failed AI grading)
  pending_review  AI grading succeeded, no review yet
  failed          AI failed / api_error, no review
  cancelled       cancelled, no review
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from ..core import atomic_io, clock
from ..errors import AppError
from . import paths, review, store

log = logging.getLogger("wm.batches")

FINAL = ("completed", "cancelled")
BATCH_FIELDS = ("task_code", "owner_id", "created_at", "status", "topic", "rubric", "genre", "full_score",
                "language", "route", "local_engine", "vlm_transcription_choice", "fewshot_ids",
                "zero_shot_approved", "sheet_template")


def result_path(cfg, task_code: str) -> Path:
    return paths.batch_dir(cfg, task_code) / f"result-{task_code}.json"


def backup_dir(cfg, task_code: str) -> Path:
    return paths.batch_dir(cfg, task_code) / "result_backup"


def _files(bdir: Path, sid: str, meta: dict) -> dict:
    sdir = bdir / sid
    rel = lambda p: p.relative_to(bdir).as_posix()
    out = {"meta": f"{sid}/meta.json",
           "images": [f"{sid}/{im['original']}" for im in meta.get("images", []) if im.get("original")]}
    for name in ("ocr", "grading", "review"):
        if (sdir / f"{name}.json").is_file():
            out[name] = f"{sid}/{name}.json"
    llm = sdir / "llm"
    out["llm"] = sorted(rel(p) for p in llm.glob("*.json")) if llm.is_dir() else []
    return out


def _row(cfg, bdir: Path, meta: dict) -> dict:
    sid = meta["subtask_id"]
    ai_attempt, ai_score = review.ai_snapshot(cfg, sid, meta)
    rv = review.latest(review.read(cfg, sid))
    teacher = rv.get("final_score") if rv else None
    if rv:
        status = "done"
    elif meta.get("status") == "cancelled":
        status = "cancelled"
    elif ai_score is not None:
        status = "pending_review"
    else:
        status = "failed"
    if teacher is not None:
        final, source = teacher, "teacher"
    elif ai_score is not None and status == "pending_review":
        final, source = ai_score, "ai"
    else:
        final, source = None, None
    return {"subtask_id": sid, "index": meta.get("index"), "status": status,
            "final_score": final, "score_source": source, "ai_score": ai_score, "teacher_score": teacher,
            "diff": rv.get("diff") if rv else None,
            "adopted_ai": bool(rv.get("adopted_ai")) if rv else None, "review_rev": rv.get("rev") if rv else None,
            "review_ai_attempt": rv.get("ai_attempt") if rv else None, "review_outdated": review.outdated(rv, ai_attempt),
            "ai_attempt": ai_attempt, "config_rev": meta.get("config_rev"),
            "subtask_status": meta.get("status"), "fail_reason": meta.get("fail_reason"),
            "photos": len(meta.get("images", [])), "files": _files(bdir, sid, meta)}


def build(cfg, task_code: str, trigger: str) -> dict:
    b = store.require_batch(cfg, task_code)
    bdir = paths.batch_dir(cfg, task_code)
    rows = [_row(cfg, bdir, m) for m in store.read_metas(cfg, task_code)]
    rows.sort(key=lambda r: (r["index"] is None, r["index"] or 0, r["subtask_id"]))
    counts = {k: sum(1 for r in rows if r["status"] == k) for k in ("done", "pending_review", "failed", "cancelled")}
    finals = [r["final_score"] for r in rows if r["final_score"] is not None]
    diffs = [abs(r["diff"]) for r in rows if r["diff"] is not None]
    revs = b.get("config_revisions") or []
    return {"schema_version": 1, "task_code": task_code, "generated_at": clock.iso(), "trigger": trigger,
            "batch": {k: b.get(k) for k in BATCH_FIELDS} | {"latest_config_rev": revs[-1].get("rev") if revs else None},
            "counts": counts | {"total": len(rows)},
            "stats": {"scored": len(finals), "mean_final_score": round(sum(finals) / len(finals), 4) if finals else None,
                      "reviewed": counts["done"],
                      "mean_abs_diff": round(sum(diffs) / len(diffs), 4) if diffs else None},
            "subtasks": rows}


def _backup(cfg, task_code: str, cur: Path) -> None:
    d = backup_dir(cfg, task_code)
    d.mkdir(parents=True, exist_ok=True)
    stamp = clock.now().strftime("%Y%m%dT%H%M%S%f")
    dst = d / f"result-{task_code}.{stamp}.json"
    k = 1
    while dst.exists():
        dst = d / f"result-{task_code}.{stamp}-{k}.json"
        k += 1
    atomic_io.write_bytes_atomic(dst, atomic_io.read_bytes(cur))
    keep = max(1, int(cfg["result_backups_keep"]))
    olds = sorted(d.glob(f"result-{task_code}.*.json"), key=lambda p: (p.stat().st_mtime_ns, p.name))
    for p in olds[:-keep]:
        try:
            p.unlink()
        except OSError:
            log.warning("could not remove old result backup %s", p.name)


def generate(cfg, task_code: str, trigger: str = "manual") -> dict:
    """Write result-<task_code>.json (backing up the previous one). Consistent snapshot under the batch lock."""
    with store.locked(cfg, task_code):
        doc = build(cfg, task_code, trigger)
        p = result_path(cfg, task_code)
        if p.is_file():
            _backup(cfg, task_code, p)
        atomic_io.write_json_atomic(p, doc)
    return doc


def generate_if_final(cfg, task_code: str, trigger: str) -> dict | None:
    b = store.read_batch(cfg, task_code)
    if b is None or b.get("status") not in FINAL:
        return None
    return generate(cfg, task_code, trigger)


def read_or_generate(cfg, task_code: str) -> dict:
    """GET: the stored summary; generated on the spot for a finished batch that has none yet (e.g. a restart
    happened between completion and generation)."""
    doc = atomic_io.read_json(result_path(cfg, task_code), None)
    if isinstance(doc, dict):
        return doc
    doc = generate_if_final(cfg, task_code, "on_demand")
    if doc is None:
        raise AppError("result_not_ready", "the summary is generated when the batch is completed or cancelled", 409)
    return doc


_threads: list[threading.Thread] = []


def _generate_quietly(cfg, task_code: str, trigger: str) -> None:
    try:
        generate(cfg, task_code, trigger)
    except Exception:
        log.exception("result generation failed for %s", task_code)


def on_batch_finished(cfg, task_code: str) -> None:
    """Executor hook (batch became completed): generate in a background thread so a large batch never stalls the
    executor loop. A GET that arrives first generates on demand (both run under the batch lock)."""
    th = threading.Thread(target=_generate_quietly, args=(cfg, task_code, "completed"),
                          name=f"result-{task_code}", daemon=True)
    _threads[:] = [t for t in _threads if t.is_alive()] + [th]
    th.start()


def wait_idle(timeout: float = 10.0) -> None:
    """Tests: wait for background generations."""
    for t in list(_threads):
        t.join(timeout)
