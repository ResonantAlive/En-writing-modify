"""Circuit breakers.

1. API-error streak (`record_api_call`): ONE counter per batch, updated under the batch lock in the order the
   calls complete, persisted in batch.json (`api_error_streak`, survives restarts). 2xx clears it, an API
   error (class B) adds one, a rejected request (class C, "other") neither adds nor clears. At
   settings.breaker.api_error_threshold (3) the batch is paused as paused_api_errors, atomically with the
   counter update (so concurrent callers cannot both trip it; in-flight subtasks simply finish).
2. Failure ratio (`evaluate_ratio`): over ALL terminal subtasks of the batch (done / failed / api_error;
   cancelled excluded). `done` (including regex_fallback) and `api_error` are never failures but are in the
   denominator. min_processed (10) reached and a failure reason with count/processed strictly > ratio (0.3)
   => paused_circuit_breaker, pause_reason = that reason code.

Both are plain synchronous functions: call them from a worker thread (asyncio.to_thread) or any thread.
Never logs essay text / model output.
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter

from ..core import atomic_io
from ..errors import NotFound
from . import paths, store

log = logging.getLogger("wm.breaker")

# fail reasons of the A (format), C (rejected) and D (local) classes that count toward the ratio breaker
RATIO_REASONS = frozenset({
    "json_parse_failed", "missing_required_field", "score_out_of_range", "empty_response", "refusal_or_safety",
    "request_rejected", "ocr_failed", "ocr_empty", "preprocess_failed", "provider_unavailable"})
OUTCOMES = ("ok", "api_error", "other")
SNIPPET = 500
MAX_SAMPLES = 5


def _settings(cfg) -> dict:
    try:
        b = cfg["breaker"]
        return {"min_processed": int(b["min_processed"]), "ratio": float(b["ratio"]),
                "threshold": int(b["api_error_threshold"])}
    except Exception:
        return {"min_processed": 10, "ratio": 0.3, "threshold": 3}


def _poke(cfg) -> None:
    try:
        from . import executor
        executor.poke(cfg)
    except Exception:
        pass


# ---- attempt files
def llm_dir(cfg, subtask_id: str):
    return paths.subtask_dir(cfg, subtask_id) / "llm"


def attempt_numbers(cfg, subtask_id: str) -> list[int]:
    out = set()
    try:
        for name in os.listdir(llm_dir(cfg, subtask_id)):
            if len(name) > 4 and name[:3].isdigit() and name[3:] in (".request.json", ".response.json"):
                out.add(int(name[:3]))
    except FileNotFoundError:
        pass
    return sorted(out)


def _is_interrupted(r: dict) -> bool:
    err = r.get("error")
    code = r.get("error_code") or (err.get("code") if isinstance(err, dict) else None)
    return code in ("interrupted", "cancelled")


def _last_response(cfg, subtask_id: str) -> tuple[int | None, dict | None]:
    for n in reversed(attempt_numbers(cfg, subtask_id)):
        r = atomic_io.read_json(llm_dir(cfg, subtask_id) / f"{n:03d}.response.json", None)
        if isinstance(r, dict) and not _is_interrupted(r):
            return n, r
    return None, None


def _snippet(resp: dict | None) -> str:
    if not resp:
        return ""
    raw = resp.get("content")
    if not raw:
        raw = resp.get("raw")
    if raw is not None and not isinstance(raw, str):
        try:
            raw = json.dumps(raw, ensure_ascii=False)
        except Exception:
            raw = ""
    return (raw or "")[:SNIPPET]


# ---- 1. consecutive API errors
def _api_error_samples(cfg, task_code: str) -> list[dict]:
    """Most recent API-error attempts of the batch (from the saved response files)."""
    found = []
    for sid in store.list_subtask_ids(cfg, task_code):
        d = llm_dir(cfg, sid)
        for n in attempt_numbers(cfg, sid):
            f = d / f"{n:03d}.response.json"
            try:
                mt = f.stat().st_mtime
            except OSError:
                continue
            r = atomic_io.read_json(f, None)
            if isinstance(r, dict) and (r.get("error_code") == "api_error"
                                        or r.get("classification") in ("B", "api_error")):
                found.append((mt, sid, n, r))
    found.sort(key=lambda x: x[0], reverse=True)
    out = []
    for _, sid, n, r in found[:MAX_SAMPLES]:
        snippet = f"http {r.get('http_status')}" if r.get("http_status") else "no response"
        if r.get("error_detail"):
            snippet += f": {r['error_detail']}"
        out.append({"subtask_id": sid, "fail_reason": "api_error", "attempt": n, "snippet": snippet[:SNIPPET]})
    return out


def record_api_call(cfg, task_code: str, outcome: str, **_ignored) -> bool:
    """Record one finished API call. True if THIS call tripped the breaker (batch paused_api_errors)."""
    if outcome not in OUTCOMES:
        outcome = "other"
    threshold = _settings(cfg)["threshold"]
    tripped = {"v": False}

    def fn(b: dict) -> None:
        streak = int(b.get("api_error_streak") or 0)
        if outcome == "ok":
            streak = 0
        elif outcome == "api_error":
            streak += 1
        b["api_error_streak"] = streak
        if outcome == "api_error" and streak >= threshold and b.get("status") in ("queued", "running"):
            old = b["status"]
            b["status"] = "paused_api_errors"
            b["pause_reason"] = "api_errors"
            b["breaker"] = {"tripped_reason": "api_errors", "samples": []}
            store.append_timeline(b, "paused", "breaker", {"from": old, "to": "paused_api_errors",
                                                           "reason": "api_errors", "streak": streak})
            tripped["v"] = True

    try:
        store.update_batch(cfg, task_code, fn)
    except NotFound:
        return False
    if tripped["v"]:
        try:      # samples are filled in right after the (atomic) pause; they are informational only
            samples = _api_error_samples(cfg, task_code)

            def put(b: dict) -> None:
                if b.get("status") == "paused_api_errors":
                    b.setdefault("breaker", {})["samples"] = samples
            store.update_batch(cfg, task_code, put)
        except Exception:
            log.exception("could not store api-error samples for %s", task_code)
        log.warning("batch %s paused: %d consecutive API errors", task_code, threshold)
        _poke(cfg)
    return tripped["v"]


# ---- 2. failure ratio
def evaluate_ratio(cfg, task_code: str) -> bool:
    """Call after every subtask reaches a terminal state. True if the batch was paused by this call."""
    st = _settings(cfg)
    b = store.read_batch(cfg, task_code)
    if b is None or b.get("status") not in ("queued", "running"):
        return False
    metas = store.read_metas(cfg, task_code)
    processed = [m for m in metas if m.get("status") in ("done", "failed", "api_error")]
    n = len(processed)
    if n < st["min_processed"]:
        return False
    reasons = Counter(m.get("fail_reason") for m in processed
                      if m.get("status") == "failed" and m.get("fail_reason") in RATIO_REASONS)
    if not reasons:
        return False
    reason, count = max(sorted(reasons.items()), key=lambda kv: kv[1])
    if not (count / n > st["ratio"]):
        return False
    samples = []
    for m in processed:
        if m.get("status") == "failed" and m.get("fail_reason") == reason:
            attempt, resp = _last_response(cfg, m["subtask_id"])
            samples.append({"subtask_id": m["subtask_id"], "fail_reason": reason,
                            "attempt": attempt, "snippet": _snippet(resp)})
            if len(samples) >= MAX_SAMPLES:
                break
    tripped = {"v": False}

    def fn(bb: dict) -> None:
        if bb.get("status") not in ("queued", "running"):
            return
        old = bb["status"]
        bb["status"] = "paused_circuit_breaker"
        bb["pause_reason"] = reason
        bb["breaker"] = {"tripped_reason": reason, "samples": samples}
        store.append_timeline(bb, "paused", "breaker", {"from": old, "to": "paused_circuit_breaker",
                                                        "reason": reason, "failed": count, "processed": n})
        tripped["v"] = True
    store.update_batch(cfg, task_code, fn)
    if tripped["v"]:
        log.warning("batch %s paused: %d/%d subtasks failed with %s", task_code, count, n, reason)
        _poke(cfg)
    return tripped["v"]
