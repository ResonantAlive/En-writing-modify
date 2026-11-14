"""Startup recovery (run once, before the executor dispatches anything).

- task directories without batch.json are half-created batches -> deleted (their numbers stay burnt,
  the counter is persisted separately and never rolls back)
- subtasks left `running` -> `queued` (cancelled batches: -> `cancelled`)
- attempts with a request.json but no response.json get an `interrupted` response.json
- queued/running batches resume automatically; paused_* stay paused
"""
from __future__ import annotations

import logging
import os
import shutil

from ..core import atomic_io, clock
from ..errors import NotFound
from . import paths, store

log = logging.getLogger("wm.batches")


def close_open_attempts(cfg, subtask_id: str, code: str = "interrupted") -> list[int]:
    """Every llm/NNN.request.json without a NNN.response.json gets a response.json recording that the
    attempt never finished (`interrupted` after a crash/shutdown, `cancelled` after a user cancel). Such
    attempts never count toward the retry budget or any breaker counter. Returns the attempt numbers."""
    d = paths.subtask_dir(cfg, subtask_id) / "llm"
    try:
        names = set(os.listdir(d))
    except (FileNotFoundError, NotFound):
        return []
    closed = []
    for name in sorted(names):
        if len(name) == 16 and name[:3].isdigit() and name.endswith(".request.json"):
            n = int(name[:3])
            if f"{n:03d}.response.json" in names:
                continue
            atomic_io.write_json_atomic(d / f"{n:03d}.response.json", {
                "attempt": n, "at": clock.iso(), "error": {"code": code}, "classification": code,
                "error_code": code})
            closed.append(n)
    return closed


def recover(cfg) -> dict:
    removed: list[str] = []
    requeued = 0
    closed = 0
    active: list[str] = []
    for d in list(paths.iter_task_dirs(cfg)):
        if not (d / "batch.json").is_file():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
            continue
        b = atomic_io.read_json(d / "batch.json", None)
        if b is None:
            continue
        tc = b["task_code"]
        cancelled = b.get("status") == "cancelled"
        for sid in store.list_subtask_ids(cfg, tc):
            try:
                closed += len(close_open_attempts(cfg, sid, "cancelled" if cancelled else "interrupted"))
            except Exception:
                log.exception("could not close open attempts of %s", sid)

            def fn(m: dict) -> bool:
                if m.get("status") == "running":
                    m["status"] = "cancelled" if cancelled else "queued"
                    store.append_timeline(m, "recovered_after_restart", None, {"to": m["status"]})
                    return True
                return False
            if store.update_meta(cfg, sid, fn) and not cancelled:
                requeued += 1
        if b.get("status") in ("queued", "running"):
            active.append(tc)
    store.reset_active(cfg, active)
    if removed or requeued or closed:
        log.warning("startup recovery: removed %d half-created batch dir(s), requeued %d subtask(s), "
                    "closed %d interrupted attempt(s)", len(removed), requeued, closed)
    return {"removed": removed, "requeued": requeued, "active": active, "interrupted_attempts": closed}
