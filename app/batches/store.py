"""Read / update helpers for batch.json and meta.json, plus the batch state machine.

All read-modify-write goes through update_batch / update_meta, which hold an in-process lock and the
file lock `batch-<task_code>` (the lock is per batch: batch.json and every meta.json of that batch share it).
The mutate callback must NOT call back into update_* (no nesting).
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from ..core import atomic_io, clock
from ..core.locks import file_lock
from ..errors import AppError, NotFound
from . import paths

SUBTASK_TERMINAL = {"done", "failed", "api_error", "cancelled"}
COUNT_KEYS = ("queued", "running", "done", "failed", "api_error", "cancelled")
PAUSED = ("paused_user", "paused_circuit_breaker", "paused_api_errors")

_guard = threading.Lock()
_tlocks: dict[str, threading.Lock] = {}
_versions: dict[str, int] = {}
_active: dict[str, set[str]] = {}          # data_dir -> task codes whose batch is queued/running


def _key(cfg, task_code: str) -> str:
    return f"{cfg.data_dir}|{task_code}"


def _tlock(key: str) -> threading.Lock:
    with _guard:
        lk = _tlocks.get(key)
        if lk is None:
            lk = _tlocks[key] = threading.Lock()
        return lk


@contextmanager
def locked(cfg, task_code: str):
    with _tlock(_key(cfg, task_code)):
        with file_lock(cfg.lock_path(f"batch-{task_code}")):
            yield


def bump(cfg, task_code: str) -> None:
    with _guard:
        k = _key(cfg, task_code)
        _versions[k] = _versions.get(k, 0) + 1


def version(cfg, task_code: str) -> int:
    with _guard:
        return _versions.get(_key(cfg, task_code), 0)


# ---- active set (batches the executor must look at)
def active_codes(cfg) -> list[str]:
    with _guard:
        return sorted(_active.get(str(cfg.data_dir), ()))


def track(cfg, task_code: str, status: str) -> None:
    with _guard:
        s = _active.setdefault(str(cfg.data_dir), set())
        (s.add if status in ("queued", "running") else s.discard)(task_code)


def reset_active(cfg, codes) -> None:
    with _guard:
        _active[str(cfg.data_dir)] = set(codes)


# ---- timeline
def append_timeline(doc: dict, event: str, by: str | None = None, detail: dict | None = None) -> None:
    """Batch timeline entries carry `by`; subtask entries omit it (pass by=None)."""
    e: dict[str, Any] = {"at": clock.iso(), "event": event}
    if by is not None:
        e["by"] = by
    e["detail"] = detail or {}
    doc.setdefault("timeline", []).append(e)


# ---- paths
def batch_json(cfg, task_code: str) -> Path:
    return paths.batch_dir(cfg, task_code) / "batch.json"


def meta_json(cfg, subtask_id: str) -> Path:
    return paths.subtask_dir(cfg, subtask_id) / "meta.json"


# ---- reads
def read_batch(cfg, task_code: str) -> dict | None:
    if not paths.valid_task_code(task_code):
        return None
    return atomic_io.read_json(batch_json(cfg, task_code), None)


def require_batch(cfg, task_code: str) -> dict:
    b = read_batch(cfg, task_code)
    if b is None:
        raise NotFound()
    return b


def read_meta(cfg, subtask_id: str) -> dict | None:
    if not paths.valid_subtask_id(subtask_id):
        return None
    return atomic_io.read_json(meta_json(cfg, subtask_id), None)


def list_subtask_ids(cfg, task_code: str) -> list[str]:
    return [p.name for p in paths._subdirs(paths.batch_dir(cfg, task_code), paths.SUB_RE)]


def read_metas(cfg, task_code: str) -> list[dict]:
    out = []
    for sid in list_subtask_ids(cfg, task_code):
        m = read_meta(cfg, sid)
        if m is not None:
            out.append(m)
    return out


def counts(cfg, task_code: str) -> dict:
    c = {k: 0 for k in COUNT_KEYS}
    for m in read_metas(cfg, task_code):
        s = m.get("status")
        if s in c:
            c[s] += 1
    c["total"] = sum(c[k] for k in COUNT_KEYS)
    return c


def iter_batches(cfg):
    """Yield every committed batch.json (dicts)."""
    for d in paths.iter_task_dirs(cfg):
        b = atomic_io.read_json(d / "batch.json", None)
        if b is not None:
            yield b


# ---- writes
def update_batch(cfg, task_code: str, fn: Callable[[dict], Any]) -> Any:
    """fn(batch) mutates in place under the lock; if fn raises nothing is written. Returns fn's result."""
    with locked(cfg, task_code):
        b = read_batch(cfg, task_code)
        if b is None:
            raise NotFound()
        res = fn(b)
        atomic_io.write_json_atomic(batch_json(cfg, task_code), b)
    track(cfg, task_code, b["status"])
    bump(cfg, task_code)
    return res


def update_meta(cfg, subtask_id: str, fn: Callable[[dict], Any]) -> Any:
    task_code = subtask_id[:12]
    with locked(cfg, task_code):
        m = read_meta(cfg, subtask_id)
        if m is None:
            raise NotFound()
        res = fn(m)
        atomic_io.write_json_atomic(meta_json(cfg, subtask_id), m)
    bump(cfg, task_code)
    return res


# ---- views
def summary(batch: dict, cnt: dict) -> dict:
    return {"task_code": batch["task_code"], "status": batch["status"], "pause_reason": batch.get("pause_reason"),
            "topic": batch.get("topic"), "genre": batch.get("genre"), "route": batch.get("route"),
            "created_at": batch.get("created_at"), "owner_id": batch.get("owner_id"), "counts": cnt}


def progress(cfg, task_code: str) -> dict | None:
    b = read_batch(cfg, task_code)
    if b is None:
        return None
    return {"task_code": task_code, "status": b["status"], "pause_reason": b.get("pause_reason"),
            "counts": counts(cfg, task_code)}


# ---- state machine
def invalid(frm: str, action: str) -> AppError:
    return AppError("invalid_transition", f"cannot {action} a batch that is {frm}", 409,
                    {"from": frm, "action": action})


def set_status(cfg, task_code: str, action: str, new: str, *, allowed_from, by: str,
               reason: str | None = None, event: str | None = None) -> dict:
    """Atomic checked transition; appends a timeline entry. Raises invalid_transition (409)."""
    def fn(b: dict) -> dict:
        if b["status"] not in allowed_from:
            raise invalid(b["status"], action)
        old = b["status"]
        b["status"] = new
        b["pause_reason"] = reason if new in PAUSED else None
        if new == "running" and old == "paused_api_errors":
            b["api_error_streak"] = 0               # the user acknowledged the outage: start counting afresh
        if new == "running" and old in ("paused_api_errors", "paused_circuit_breaker"):
            b["breaker"] = {"tripped_reason": None, "samples": []}
        append_timeline(b, event or f"status:{new}", by, {"from": old, "to": new, "reason": reason})
        return b
    return update_batch(cfg, task_code, fn)


def requeue_subtasks(cfg, task_code: str, statuses: set[str], event: str = "requeued") -> int:
    n = 0
    for sid in list_subtask_ids(cfg, task_code):
        def fn(m: dict) -> bool:
            if m.get("status") in statuses:
                frm = m["status"]
                m["status"] = "queued"
                m["fail_reason"] = None
                m["run_base"] = None                  # fresh retry budget for the re-run
                append_timeline(m, event, None, {"from": frm})
                return True
            return False
        if update_meta(cfg, sid, fn):
            n += 1
    return n


def cancel_subtasks(cfg, task_code: str, event: str = "cancelled") -> list[str]:
    """queued/running -> cancelled. Returns the ids that were running."""
    was_running: list[str] = []
    for sid in list_subtask_ids(cfg, task_code):
        def fn(m: dict) -> None:
            if m.get("status") in ("queued", "running"):
                if m["status"] == "running":
                    was_running.append(sid)
                frm = m["status"]
                m["status"] = "cancelled"
                append_timeline(m, event, None, {"from": frm})
        update_meta(cfg, sid, fn)
    return was_running


def user_action(cfg, task_code: str, action: str, by: str) -> dict:
    """pause | resume | cancel | rerun-failed. Returns the updated batch."""
    cur = require_batch(cfg, task_code)["status"]
    if action == "pause":
        return set_status(cfg, task_code, action, "paused_user", allowed_from=("queued", "running"),
                          by=by, reason="user", event="paused")
    if action == "resume":
        return set_status(cfg, task_code, action, "running", allowed_from=PAUSED, by=by, event="resumed")
    if action == "cancel":
        b = set_status(cfg, task_code, action, "cancelled",
                       allowed_from=("queued", "running", "completed") + PAUSED, by=by, event="cancelled")
        cancel_subtasks(cfg, task_code)
        return b
    if action == "rerun-failed":
        if cur not in PAUSED + ("completed",):
            raise invalid(cur, action)
        c = counts(cfg, task_code)
        if c["failed"] + c["api_error"] == 0:
            raise AppError("nothing_to_rerun", "no failed subtasks to rerun", 409)
        # requeue first: a running batch with no queued subtask would be completed by the executor
        requeue_subtasks(cfg, task_code, {"failed", "api_error"}, "rerun_failed")
        return set_status(cfg, task_code, action, "running", allowed_from=PAUSED + ("completed",),
                          by=by, event="rerun_failed")
    raise NotFound()


def pause_batch(cfg, task_code: str, status: str, reason: str | None, by: str = "system",
                event: str = "paused", detail: dict | None = None) -> bool:
    """Thread-safe, synchronous pause of a queued/running batch (no new dispatch; in-flight subtasks finish).
    Works from worker threads and (briefly blocking) from the event loop. False if it was not pausable."""
    if status not in PAUSED:
        raise ValueError(f"not a paused status: {status}")
    try:
        set_status(cfg, task_code, "pause", status, allowed_from=("queued", "running"), by=by,
                   reason=reason, event=event)
    except AppError:
        return False
    return True


def regrade_subtask(cfg, task_code: str, subtask_id: str, by: str) -> dict:
    """done/failed/api_error -> queued with a fresh retry budget. One atomic step under the batch lock:
    completed batch -> running; paused batch stays paused (the subtask waits for resume); cancelled -> 409.
    Returns {"status": batch status, "queued_until_resume": bool}."""
    if not paths.valid_subtask_id(subtask_id) or subtask_id[:12] != task_code:
        raise NotFound()

    def fn(b: dict) -> dict:
        m = atomic_io.read_json(meta_json(cfg, subtask_id), None)
        if m is None:
            raise NotFound()
        if b["status"] == "cancelled":
            raise invalid(b["status"], "regrade")
        if m.get("status") not in ("done", "failed", "api_error"):
            raise AppError("invalid_transition", f"cannot regrade a subtask that is {m.get('status')}", 409,
                           {"from": m.get("status"), "action": "regrade"})
        frm = m["status"]
        m["status"] = "queued"
        m["fail_reason"] = None
        m["run_base"] = None
        append_timeline(m, "regrade", None, {"from": frm})
        atomic_io.write_json_atomic(meta_json(cfg, subtask_id), m)
        if b["status"] == "completed":
            b["status"] = "running"
            b["pause_reason"] = None
            append_timeline(b, "regrade", by, {"subtask_id": subtask_id, "from": "completed", "to": "running"})
        else:
            append_timeline(b, "regrade", by, {"subtask_id": subtask_id})
        return {"status": b["status"], "queued_until_resume": b["status"] in PAUSED}
    return update_batch(cfg, task_code, fn)


def pause_owner_batches(cfg, owner_id: str) -> list[str]:
    """Owner account disabled: queued/running batches -> paused_user / owner_disabled."""
    done = []
    for b in list(iter_batches(cfg)):
        if b.get("owner_id") != owner_id or b.get("status") not in ("queued", "running"):
            continue
        try:
            set_status(cfg, b["task_code"], "pause", "paused_user", allowed_from=("queued", "running"),
                       by="system", reason="owner_disabled", event="paused")
            done.append(b["task_code"])
        except AppError:
            pass
    return done
