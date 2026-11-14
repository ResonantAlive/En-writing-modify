"""M3 (K): circuit breakers, config revisions, recovery of interrupted attempts, regrade / rerun, failures and
subtask detail endpoints. Fake processors only; no network, no key."""
from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from app.batches import breaker, hooks, paths, recover, store
from app.batches import create as create_mod
from app.batches import executor as ex_mod
from app.config import load_config
from app.core import atomic_io, clock
from tests.test_batches import (API, Env, H, _direct_cfg, _env_with, _isolate, _unbind_logging,  # noqa: F401
                                api_create, env, env1, env2, err, post_action, stage, status_of,
                                sub_statuses, body, wait_for, Gate, force)

Outcome = ex_mod.Outcome


@pytest.fixture
def env3(tmp_path):
    clock.set_now_fn(None)
    e, ctx, old = _env_with(tmp_path, {"concurrency": 3})
    yield e
    ctx.__exit__(None, None, None)
    ex_mod.Executor.TICK_S = old
    _unbind_logging()


# --------------------------------------------------------------------------- helpers
def mk_batch(cfg, user, n: int, **over) -> dict:
    up = stage(cfg, user.user_id, [f"{i}.jpg" for i in range(n)])[0]
    return create_mod.create_batch(cfg, user, body(up, **over))


def set_sub(cfg, sid: str, status: str, reason: str | None = None, **extra) -> None:
    def fn(m):
        m["status"] = status
        m["fail_reason"] = reason
        m.update(extra)
    store.update_meta(cfg, sid, fn)


def running(cfg, tc: str) -> None:
    store.update_batch(cfg, tc, lambda b: b.update(status="running"))


def batch_of(cfg, tc: str) -> dict:
    return store.read_batch(cfg, tc)


def write_attempt(cfg, sid: str, n: int, request=True, response=None) -> None:
    d = paths.subtask_dir(cfg, sid) / "llm"
    d.mkdir(parents=True, exist_ok=True)
    if request:
        atomic_io.write_json_atomic(d / f"{n:03d}.request.json", {"attempt": n, "model": "m1"})
    if response is not None:
        atomic_io.write_json_atomic(d / f"{n:03d}.response.json", response)


def read_resp(cfg, sid: str, n: int):
    return atomic_io.read_json(paths.subtask_dir(cfg, sid) / "llm" / f"{n:03d}.response.json", None)


# --------------------------------------------------------------------------- ratio breaker (unit)
def test_ratio_trips_over_30_percent_with_samples(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    r = mk_batch(cfg, user, 12)
    tc, sids = r["task_code"], r["subtask_ids"]
    running(cfg, tc)
    for i, sid in enumerate(sids[:10]):
        set_sub(cfg, sid, "failed" if i < 4 else "done", "json_parse_failed" if i < 4 else None)
    write_attempt(cfg, sids[0], 1, response={"attempt": 1, "content": "x" * 900, "error_code": "json_parse_failed"})
    assert breaker.evaluate_ratio(cfg, tc) is True                    # 4/10 = 40% > 30%
    b = batch_of(cfg, tc)
    assert b["status"] == "paused_circuit_breaker" and b["pause_reason"] == "json_parse_failed"
    assert b["breaker"]["tripped_reason"] == "json_parse_failed"
    s = b["breaker"]["samples"]
    assert len(s) == 4 and {x["subtask_id"] for x in s} == set(sids[:4])
    first = next(x for x in s if x["subtask_id"] == sids[0])
    assert first["attempt"] == 1 and first["snippet"] == "x" * 500 and first["fail_reason"] == "json_parse_failed"
    ev = [e for e in b["timeline"] if e["event"] == "paused"][-1]
    assert ev["by"] == "breaker" and ev["detail"]["reason"] == "json_parse_failed"
    # already paused: never triggers twice
    assert breaker.evaluate_ratio(cfg, tc) is False
    assert len([e for e in batch_of(cfg, tc)["timeline"] if e["event"] == "paused"]) == 1


def test_ratio_samples_capped_at_five(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    r = mk_batch(cfg, user, 10)
    running(cfg, r["task_code"])
    for sid in r["subtask_ids"]:
        set_sub(cfg, sid, "failed", "empty_response")
    assert breaker.evaluate_ratio(cfg, r["task_code"])
    assert len(batch_of(cfg, r["task_code"])["breaker"]["samples"]) == 5


def test_ratio_exactly_30_percent_does_not_trip(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    r = mk_batch(cfg, user, 10)
    running(cfg, r["task_code"])
    for i, sid in enumerate(r["subtask_ids"]):
        set_sub(cfg, sid, "failed" if i < 3 else "done", "score_out_of_range" if i < 3 else None)
    assert breaker.evaluate_ratio(cfg, r["task_code"]) is False
    assert batch_of(cfg, r["task_code"])["status"] == "running"


def test_ratio_needs_min_processed(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    r = mk_batch(cfg, user, 12)
    running(cfg, r["task_code"])
    for sid in r["subtask_ids"][:9]:
        set_sub(cfg, sid, "failed", "refusal_or_safety")                # 9 of 9 failed but < 10 processed
    assert breaker.evaluate_ratio(cfg, r["task_code"]) is False
    set_sub(cfg, r["subtask_ids"][9], "failed", "refusal_or_safety")    # 10th terminal subtask
    assert breaker.evaluate_ratio(cfg, r["task_code"]) is True
    assert batch_of(cfg, r["task_code"])["pause_reason"] == "refusal_or_safety"


def test_ratio_cancelled_not_processed_done_and_api_error_never_failures(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    r = mk_batch(cfg, user, 14)
    tc, sids = r["task_code"], r["subtask_ids"]
    running(cfg, tc)
    # 10 processed: 6 api_error + 3 done(regex_fallback) + 1 failed  -> nothing counts as a failure streak
    for sid in sids[:6]:
        set_sub(cfg, sid, "api_error", "api_error")
    for sid in sids[6:9]:
        set_sub(cfg, sid, "done", None, warnings=["regex_fallback"])
    set_sub(cfg, sids[9], "failed", "json_parse_failed")
    for sid in sids[10:]:
        set_sub(cfg, sid, "cancelled")                                   # excluded from the denominator
    assert breaker.evaluate_ratio(cfg, tc) is False
    # different reasons never add up: 3 reasons x 3 of 10
    for sid, reason in zip(sids[:9], ["empty_response"] * 3 + ["ocr_failed"] * 3 + ["request_rejected"] * 3):
        set_sub(cfg, sid, "failed", reason)
    set_sub(cfg, sids[9], "done")
    assert breaker.evaluate_ratio(cfg, tc) is False                      # 30% each, not strictly more
    # non-ratio reasons (internal_error) never trip
    for sid in sids[:10]:
        set_sub(cfg, sid, "failed", "internal_error")
    assert breaker.evaluate_ratio(cfg, tc) is False
    assert batch_of(cfg, tc)["status"] == "running"


def test_ratio_after_resume_uses_all_terminal_subtasks(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    r = mk_batch(cfg, user, 12)
    tc, sids = r["task_code"], r["subtask_ids"]
    running(cfg, tc)
    for i, sid in enumerate(sids[:10]):
        set_sub(cfg, sid, "failed" if i < 5 else "done", "json_parse_failed" if i < 5 else None)
    assert breaker.evaluate_ratio(cfg, tc)
    store.user_action(cfg, tc, "resume", "u")
    b = batch_of(cfg, tc)
    assert b["status"] == "running" and b["breaker"] == {"tripped_reason": None, "samples": []}
    assert breaker.evaluate_ratio(cfg, tc) is True                        # same formula over all terminal ones
    store.user_action(cfg, tc, "resume", "u")
    for sid in sids[:5]:                                                 # config fixed: failures re-run fine
        set_sub(cfg, sid, "done")
    assert breaker.evaluate_ratio(cfg, tc) is False


def test_ratio_breaker_via_executor_stops_dispatch(env1):
    seen = []

    async def proc(ctx):
        seen.append(ctx.subtask_id)
        if len(seen) <= 5:
            return Outcome("failed", "json_parse_failed")
        return Outcome("done")

    ex_mod.set_processor(proc)
    tc = api_create(env1, "alice", names=tuple(f"{i}.jpg" for i in range(12)))
    wait_for(lambda: status_of(env1, "alice", tc)["status"] == "paused_circuit_breaker")
    time.sleep(0.4)
    assert len(seen) == 10                                                # 5/10 trips at the 10th terminal subtask
    d = status_of(env1, "alice", tc)
    assert d["pause_reason"] == "json_parse_failed"
    f = env1.clients["alice"].get(f"{API}/batches/{tc}/failures").json()
    assert f["pause_reason"] == "json_parse_failed" and f["reasons"] == {"json_parse_failed": 5}
    assert len(f["samples"]) == 5 and set(f["samples"][0]) == {"subtask_id", "fail_reason", "attempt", "snippet"}
    assert sub_statuses(env1, tc).count("queued") == 2
    # resume: remaining two run, 5/12 > 30% re-trips only if still over; here batch completes or re-pauses
    assert post_action(env1, "alice", tc, "resume").status_code == 200
    wait_for(lambda: status_of(env1, "alice", tc)["status"] in ("paused_circuit_breaker", "completed"))


def test_ratio_not_triggered_by_regex_fallback_done(env1):
    async def proc(ctx):
        return Outcome("done", None, ["regex_fallback"])

    ex_mod.set_processor(proc)
    tc = api_create(env1, "alice", names=tuple(f"{i}.jpg" for i in range(11)))
    wait_for(lambda: status_of(env1, "alice", tc)["status"] == "completed")
    assert status_of(env1, "alice", tc)["breaker"]["tripped_reason"] is None


# --------------------------------------------------------------------------- API error streak (unit)
def test_three_consecutive_api_errors_pause_immediately(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    r = mk_batch(cfg, user, 4)
    tc = r["task_code"]
    running(cfg, tc)
    assert breaker.record_api_call(cfg, tc, "api_error") is False
    assert breaker.record_api_call(cfg, tc, "api_error") is False
    assert batch_of(cfg, tc)["api_error_streak"] == 2
    assert breaker.record_api_call(cfg, tc, "api_error") is True
    b = batch_of(cfg, tc)
    assert b["status"] == "paused_api_errors" and b["pause_reason"] == "api_errors"
    assert b["api_error_streak"] == 3 and b["breaker"]["tripped_reason"] == "api_errors"
    assert breaker.record_api_call(cfg, tc, "api_error") is False         # already paused: no second trip
    assert len([e for e in batch_of(cfg, tc)["timeline"] if e["event"] == "paused"]) == 1


def test_error_error_ok_error_error_does_not_trip(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    tc = mk_batch(cfg, user, 2)["task_code"]
    running(cfg, tc)
    for o in ("api_error", "api_error", "ok", "api_error", "api_error"):
        assert breaker.record_api_call(cfg, tc, o) is False
    b = batch_of(cfg, tc)
    assert b["status"] == "running" and b["api_error_streak"] == 2


def test_other_class_neither_increments_nor_clears(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    tc = mk_batch(cfg, user, 2)["task_code"]
    running(cfg, tc)
    breaker.record_api_call(cfg, tc, "api_error")
    breaker.record_api_call(cfg, tc, "other")
    assert batch_of(cfg, tc)["api_error_streak"] == 1
    breaker.record_api_call(cfg, tc, "api_error")
    breaker.record_api_call(cfg, tc, "other")
    assert batch_of(cfg, tc)["api_error_streak"] == 2
    assert batch_of(cfg, tc)["status"] == "running"
    assert breaker.record_api_call(cfg, tc, "api_error") is True          # errors separated by C still consecutive
    assert breaker.record_api_call(cfg, tc, "bogus") is False             # unknown value == other


def test_concurrent_threads_share_one_counter(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    cfg.settings["breaker"]["api_error_threshold"] = 1000
    tc = mk_batch(cfg, user, 2)["task_code"]
    running(cfg, tc)
    n = 40
    barrier = threading.Barrier(n)

    def call():
        barrier.wait()
        breaker.record_api_call(cfg, tc, "api_error")
    ts = [threading.Thread(target=call) for _ in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert batch_of(cfg, tc)["api_error_streak"] == n                    # no lost update


def test_concurrent_exactly_one_call_trips(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    tc = mk_batch(cfg, user, 2)["task_code"]
    running(cfg, tc)
    n = 12
    barrier = threading.Barrier(n)
    results = []

    def call():
        barrier.wait()
        results.append(breaker.record_api_call(cfg, tc, "api_error"))
    ts = [threading.Thread(target=call) for _ in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert results.count(True) == 1
    assert batch_of(cfg, tc)["status"] == "paused_api_errors"
    assert [e["event"] for e in batch_of(cfg, tc)["timeline"]].count("paused") == 1


def test_asyncio_gather_through_to_thread(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    tc = mk_batch(cfg, user, 2)["task_code"]
    running(cfg, tc)

    async def go():
        return await asyncio.gather(*[asyncio.to_thread(breaker.record_api_call, cfg, tc, "api_error")
                                      for _ in range(6)])
    res = asyncio.run(go())
    assert res.count(True) == 1 and batch_of(cfg, tc)["status"] == "paused_api_errors"


def test_ok_clears_and_resume_resets_streak(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    tc = mk_batch(cfg, user, 2)["task_code"]
    running(cfg, tc)
    for _ in range(3):
        breaker.record_api_call(cfg, tc, "api_error")
    assert batch_of(cfg, tc)["status"] == "paused_api_errors"
    store.user_action(cfg, tc, "resume", "u")
    b = batch_of(cfg, tc)
    assert b["status"] == "running" and b["api_error_streak"] == 0 and b["breaker"]["samples"] == []
    assert breaker.record_api_call(cfg, tc, "api_error") is False         # needs 3 fresh consecutive errors


def test_api_error_samples_from_response_files(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    r = mk_batch(cfg, user, 3)
    tc = r["task_code"]
    running(cfg, tc)
    for i, sid in enumerate(r["subtask_ids"]):
        write_attempt(cfg, sid, 1, response={"attempt": 1, "http_status": 503, "error_code": "api_error",
                                             "classification": "B", "error_detail": "upstream down"})
        breaker.record_api_call(cfg, tc, "api_error")
    s = batch_of(cfg, tc)["breaker"]["samples"]
    assert len(s) == 3 and all("503" in x["snippet"] and x["fail_reason"] == "api_error" for x in s)


def test_streak_persists_across_restart_and_paused_stays_paused(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    tc = mk_batch(cfg, user, 3)["task_code"]
    running(cfg, tc)
    breaker.record_api_call(cfg, tc, "api_error")
    breaker.record_api_call(cfg, tc, "api_error")
    cfg2 = load_config(tmp_path / "data")                                 # "restart"
    recover.recover(cfg2)
    assert batch_of(cfg2, tc)["api_error_streak"] == 2
    assert breaker.record_api_call(cfg2, tc, "api_error") is True
    cfg3 = load_config(tmp_path / "data")
    recover.recover(cfg3)
    b = batch_of(cfg3, tc)
    assert b["status"] == "paused_api_errors" and b["api_error_streak"] == 3 and tc not in store.active_codes(cfg3)


def test_api_error_pause_from_worker_threads_in_running_executor(env3):
    """3 in-flight subtasks fail with API errors concurrently: batch pauses, in-flight finish and are saved,
    nothing new is dispatched."""
    cfg = env3.cfg
    started = []
    lock = threading.Lock()
    tc_box = {}

    async def proc(ctx):
        with lock:
            started.append(ctx.subtask_id)
        while len(started) < 3:
            await asyncio.sleep(0.01)
        await asyncio.to_thread(breaker.record_api_call, cfg, ctx.subtask_id[:12], "api_error")
        await asyncio.sleep(0.4)                                          # keep the slot busy: no new dispatch
        return Outcome("api_error", "api_error")

    ex_mod.set_processor(proc)
    tc = api_create(env3, "alice", names=tuple(f"{i}.jpg" for i in range(6)))
    wait_for(lambda: status_of(env3, "alice", tc)["status"] == "paused_api_errors")
    d = status_of(env3, "alice", tc)
    assert d["pause_reason"] == "api_errors" and d["api_error_streak"] == 3
    wait_for(lambda: sub_statuses(env3, tc).count("api_error") == 3)
    time.sleep(0.4)
    assert len(started) == 3 and sub_statuses(env3, tc).count("queued") == 3
    assert status_of(env3, "alice", tc)["status"] == "paused_api_errors"


# --------------------------------------------------------------------------- interrupted attempts / recovery
def test_recovery_writes_interrupted_and_does_not_count(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    r = mk_batch(cfg, user, 2)
    tc, (s1, s2) = r["task_code"], r["subtask_ids"]
    store.update_batch(cfg, tc, lambda b: b.update(status="running"))
    store.update_meta(cfg, s1, lambda m: m.update(status="running"))
    write_attempt(cfg, s1, 1, response={"attempt": 1, "error_code": "empty_response", "classification": "A"})
    write_attempt(cfg, s1, 2)                                            # request but no response: crash
    write_attempt(cfg, s2, 1, response={"attempt": 1, "error_code": None})
    info = recover.recover(cfg)
    assert info["interrupted_attempts"] == 1
    resp = read_resp(cfg, s1, 2)
    assert resp["attempt"] == 2 and resp["error"] == {"code": "interrupted"}
    assert resp["classification"] == "interrupted" and resp["error_code"] == "interrupted"
    assert read_resp(cfg, s1, 1)["error_code"] == "empty_response"      # untouched
    assert (paths.subtask_dir(cfg, s1) / "llm" / "002.request.json").is_file()
    assert store.read_meta(cfg, s1)["status"] == "queued"
    assert breaker.attempt_numbers(cfg, s1) == [1, 2]                   # next attempt number is 3
    assert batch_of(cfg, tc)["api_error_streak"] == 0                    # no breaker counter moved
    assert breaker._last_response(cfg, s1)[0] == 1                       # interrupted is not "the last failure"
    assert recover.recover(cfg)["interrupted_attempts"] == 0             # idempotent


def test_shutdown_writes_interrupted_and_restart_uses_next_attempt(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    r = mk_batch(cfg, user, 1)
    tc, sid = r["task_code"], r["subtask_ids"][0]
    box = {"n": 0}

    async def proc(ctx):
        n = box["n"] = box["n"] + 1
        nums = breaker.attempt_numbers(ctx.cfg, ctx.subtask_id)
        att = (max(nums) if nums else 0) + 1
        write_attempt(ctx.cfg, ctx.subtask_id, att)
        if n == 1:
            await asyncio.sleep(30)
        write_attempt(ctx.cfg, ctx.subtask_id, att, request=False, response={"attempt": att, "error_code": None})
        return Outcome("done")

    async def run(wait_done):
        ex_mod.Executor.TICK_S = 0.05
        ex_mod.set_processor(proc)
        e = ex_mod.Executor(cfg)
        await e.start()
        for _ in range(300):
            if wait_done():
                break
            await asyncio.sleep(0.02)
        await e.stop()
    try:
        asyncio.run(run(lambda: box["n"] == 1))
        assert read_resp(cfg, sid, 1)["error_code"] == "interrupted"
        assert store.read_meta(cfg, sid)["status"] == "queued"
        asyncio.run(run(lambda: batch_of(cfg, tc)["status"] == "completed"))
    finally:
        ex_mod.Executor.TICK_S = 0.5
    assert breaker.attempt_numbers(cfg, sid) == [1, 2]
    assert read_resp(cfg, sid, 2)["error_code"] is None
    assert batch_of(cfg, tc)["api_error_streak"] == 0


def test_cancel_marks_open_attempt_cancelled(env1):
    cfg = env1.cfg
    started = []

    async def proc(ctx):
        write_attempt(ctx.cfg, ctx.subtask_id, 1)
        started.append(ctx.subtask_id)
        await asyncio.sleep(30)

    ex_mod.set_processor(proc)
    tc = api_create(env1, "alice", names=("a.jpg", "b.jpg"))
    wait_for(lambda: started)
    assert post_action(env1, "alice", tc, "cancel").status_code == 200
    wait_for(lambda: read_resp(cfg, started[0], 1) is not None)
    r = read_resp(cfg, started[0], 1)
    assert r["error"] == {"code": "cancelled"} and r["classification"] == "cancelled"
    assert sub_statuses(env1, tc) == ["cancelled", "cancelled"]


# --------------------------------------------------------------------------- config revisions
def test_config_revision_flow_only_unfinished_use_new_revision(env1):
    gate = Gate()
    used = {}

    async def proc(ctx):
        used[ctx.subtask_id] = ctx.config_rev
        if len(used) == 2:
            await gate.wait()
        return Outcome("done")

    ex_mod.set_processor(proc)
    tc = api_create(env1, "alice", names=tuple(f"{i}.jpg" for i in range(4)))
    wait_for(lambda: len(used) == 2)
    assert post_action(env1, "alice", tc, "pause").status_code == 200    # in-flight (#2) finishes, no more dispatch
    gate.ev.set()
    sids = store.list_subtask_ids(env1.cfg, tc)
    wait_for(lambda: sub_statuses(env1, tc)[:2] == ["done", "done"])
    c = env1.clients["alice"]
    r = c.post(f"{API}/batches/{tc}/config", json={"model": "m2", "params": {"temperature": 0.5}}, headers=H(c))
    assert r.status_code == 200, r.text
    rev = r.json()
    assert rev["rev"] == 2 and rev["reason"] == "user_change" and rev["model"] == "m2"
    assert rev["params"] == {"temperature": 0.5} and rev["provider_id"] == "p1"
    assert rev["template"] == {"id": "t1", "version": 1}                  # unchanged template stays pinned
    assert post_action(env1, "alice", tc, "resume").status_code == 200
    wait_for(lambda: status_of(env1, "alice", tc)["status"] == "completed")
    assert [used[s] for s in sids] == [1, 1, 2, 2]
    assert [m["config_rev"] for m in store.read_metas(env1.cfg, tc)] == [1, 1, 2, 2]
    d = status_of(env1, "alice", tc)
    assert [x["rev"] for x in d["config_revisions"]] == [1, 2]
    assert "rendered_prompt" not in json.dumps(d["config_revisions"]) and d["config_revisions"][1]["model"] == "m2"
    assert d["score_key"] and d["criteria_key"]
    assert "config_changed" in [e["event"] for e in d["timeline"]]


def test_config_failed_subtasks_rerun_with_new_revision_on_completed_batch(env1):
    used = []

    async def proc(ctx):
        used.append((ctx.subtask_id, ctx.config_rev))
        return Outcome("failed", "ocr_failed") if len(used) == 2 else Outcome("done")

    ex_mod.set_processor(proc)
    tc = api_create(env1, "alice", names=("a.jpg", "b.jpg", "c.jpg"))
    wait_for(lambda: status_of(env1, "alice", tc)["status"] == "completed")
    c = env1.clients["alice"]
    assert c.post(f"{API}/batches/{tc}/config", json={"model": "m9"}, headers=H(c)).json()["rev"] == 2
    assert post_action(env1, "alice", tc, "rerun-failed").status_code == 200
    wait_for(lambda: status_of(env1, "alice", tc)["status"] == "completed")
    assert [m["config_rev"] for m in store.read_metas(env1.cfg, tc)] == [1, 2, 1]
    assert sub_statuses(env1, tc) == ["done", "done", "done"]


def test_config_only_in_paused_or_completed_and_owner_only(env):
    tc = api_create(env, "alice")
    c = env.clients["alice"]
    err(c.post(f"{API}/batches/{tc}/config", json={"model": "m2"}, headers=H(c)), 409, "invalid_transition")
    force(env, tc, "paused_user", "user")
    err(env.clients["bob"].post(f"{API}/batches/{tc}/config", json={"model": "m2"},
                                headers=H(env.clients["bob"])), 404, "not_found")
    assert c.post(f"{API}/batches/{tc}/config", json={}, headers=H(c)).json()["rev"] == 2   # no-change revision ok
    r = env.clients["rooty"].post(f"{API}/batches/{tc}/config", json={"model": "m3"}, headers=H(env.clients["rooty"]))
    assert r.status_code == 200 and r.json()["rev"] == 3
    err(c.post(f"{API}/batches/{tc}/config", json={"params": "x"}, headers=H(c)), 422, "validation_error")


def test_config_new_provider_validation_and_third_party_ack(env, monkeypatch):
    from app.errors import NotFound
    tc = api_create(env, "alice")
    force(env, tc, "paused_user", "user")
    c = env.clients["alice"]
    seen = []

    def validate(cfg, user, b):
        seen.append(dict(b))
        if b["provider_id"] == "bad":
            raise NotFound("provider_not_found")
        if not b.get("model"):
            b["model"] = "default-of-" + b["provider_id"]
    monkeypatch.setattr(hooks, "validate_provider_and_template", validate)
    monkeypatch.setattr(hooks, "needs_third_party_ack", lambda cfg, user, pid: pid == "p2")
    err(c.post(f"{API}/batches/{tc}/config", json={"provider_id": "bad"}, headers=H(c)), 404, "not_found")
    assert seen[-1]["route"] == "vlm" and seen[-1]["template_id"] == "t1"
    err(c.post(f"{API}/batches/{tc}/config", json={"provider_id": "p2"}, headers=H(c)), 409,
        "third_party_ack_required")
    assert len(status_of(env, "alice", tc)["config_revisions"]) == 1      # nothing appended on failure
    from app.providers import store as pstore, third_party
    acked = []
    monkeypatch.setattr(pstore, "get_provider", lambda cfg, pid: {"id": pid, "base_url": "https://x.example/v1"})
    monkeypatch.setattr(third_party, "record_ack", lambda cfg, uid, prov: acked.append((uid, prov["id"])))
    r = c.post(f"{API}/batches/{tc}/config", json={"provider_id": "p2", "third_party_ack": True}, headers=H(c))
    assert r.status_code == 200, r.text
    assert r.json()["provider_id"] == "p2" and r.json()["model"] == "default-of-p2"   # new provider => its default
    assert acked == [(env.uid("alice"), "p2")]


def test_create_records_third_party_ack_and_revision_via_hook(env, monkeypatch):
    from app.providers import store as pstore, third_party
    acked = []
    built = []
    monkeypatch.setattr(pstore, "get_provider", lambda cfg, pid: {"id": pid, "base_url": "https://x.example/v1"})
    monkeypatch.setattr(third_party, "record_ack", lambda cfg, uid, prov: acked.append((uid, prov["id"])))
    monkeypatch.setattr(hooks, "needs_third_party_ack", lambda cfg, user, pid: True)

    def build(cfg, user, b, fewshot_ids, rev, reason, template_override=None):
        built.append((dict(b), list(fewshot_ids), rev, reason))
        return {"rev": rev, "at": clock.iso(), "provider_id": b["provider_id"], "provider_alias": "alias",
                "model": b["model"], "params": {}, "template": {"id": b["template_id"], "version": 7},
                "rendered_prompt": "SYSTEM", "reason": reason}
    monkeypatch.setattr(hooks, "build_revision", build)
    monkeypatch.setattr(hooks, "find_fewshot", lambda cfg, uid, scope, draft: ["000000000000_00001"])
    c = env.clients["alice"]
    up, _, _ = stage(env.cfg, env.uid("alice"), ["a.jpg"])
    err(c.post(f"{API}/batches", json=body(up, vlm_transcription_choice="history_fewshot"), headers=H(c)),
        409, "third_party_ack_required")
    assert acked == [] and built == []
    r = c.post(f"{API}/batches", json=body(up, vlm_transcription_choice="history_fewshot", third_party_ack=True),
               headers=H(c))
    assert r.status_code == 201, r.text
    assert acked == [(env.uid("alice"), "p1")]
    assert built[0][1:] == (["000000000000_00001"], 1, "initial") and built[0][0]["model"] == "m1"
    b = store.read_batch(env.cfg, r.json()["task_code"])
    assert b["config_revisions"][0]["rendered_prompt"] == "SYSTEM" and b["config_revisions"][0]["rev"] == 1
    assert b["fewshot_ids"] == ["000000000000_00001"]


def test_default_build_revision_is_placeholder(env):
    tc = api_create(env, "alice")
    rev = store.read_batch(env.cfg, tc)["config_revisions"][0]
    assert rev["rev"] == 1 and rev["reason"] == "initial" and rev["rendered_prompt"] == ""
    assert rev["provider_id"] == "p1" and rev["model"] == "m1" and rev["template"] == {"id": "t1", "version": 1}
    assert sorted(rev) == ["at", "model", "params", "provider_alias", "provider_id", "reason", "rendered_prompt",
                           "rev", "template"]


def test_config_with_real_providers_and_prompts(env):
    """Integration with P's real stores: prompt_text saving rules + third-party ack of a NEW provider."""
    pytest.importorskip("app.providers.store")
    pytest.importorskip("app.prompts.store")
    from app.providers import third_party
    third_party.register_hooks()
    c, root = env.clients["alice"], env.clients["rooty"]
    local = c.post(f"{API}/providers", json={"alias": "local", "base_url": "http://127.0.0.1:9/v1",
                                             "api_key": "sk-test-xxxx", "default_model": "m-local"}, headers=H(c))
    far = c.post(f"{API}/providers", json={"alias": "far", "base_url": "https://api.example.com/v1",
                                           "api_key": "sk-test-yyyy", "default_model": "m-far"}, headers=H(c))
    assert local.status_code in (200, 201) and far.status_code in (200, 201), (local.text, far.text)
    lid, fid = local.json()["id"], far.json()["id"]
    txt = "Grade. {{FEWSHOT}}"
    mine = c.post(f"{API}/prompts", json={"name": "mine", "text": txt, "input_mode": "image"}, headers=H(c)).json()
    theirs = root.post(f"{API}/prompts", json={"name": "rootp", "text": txt, "input_mode": "image"}, headers=H(root)).json()
    assert root.post(f"{API}/prompts/{theirs['id']}/share", json={"shared": True}, headers=H(root)).status_code == 200
    text_tpl = c.post(f"{API}/prompts", json={"name": "texty", "text": txt, "input_mode": "text"}, headers=H(c)).json()
    up, _, _ = stage(env.cfg, env.uid("alice"), ["a.jpg"])
    r = c.post(f"{API}/batches", json=body(up, provider_id=lid, model=None, template_id=mine["id"],
                                           template_version=1), headers=H(c))
    assert r.status_code == 201, r.text
    tc = r.json()["task_code"]
    assert store.read_batch(env.cfg, tc)["config_revisions"][0]["model"] == "m-local"
    force(env, tc, "paused_user", "user")
    # new third-party provider: ack needed
    err(c.post(f"{API}/batches/{tc}/config", json={"provider_id": fid}, headers=H(c)), 409,
        "third_party_ack_required")
    # template of the wrong mode is rejected
    err(c.post(f"{API}/batches/{tc}/config", json={"template_id": text_tpl["id"]}, headers=H(c)), 422,
        "template_mode_mismatch")
    r = c.post(f"{API}/batches/{tc}/config", json={"provider_id": fid, "third_party_ack": True,
                                                  "prompt_text": "Edited. {{FEWSHOT}}"}, headers=H(c))
    assert r.status_code == 200, r.text
    rev = r.json()
    assert rev["provider_id"] == fid and rev["model"] == "m-far"
    assert rev["template"] == {"id": mine["id"], "version": 2}            # own template: new version, same id
    assert c.get(f"{API}/providers/{fid}/ack-status").json()["acked"] is True
    # a shared template of another user: edited text becomes a NEW template with the parent recorded
    r = c.post(f"{API}/batches/{tc}/config", json={"template_id": theirs["id"], "prompt_text": "Mine now. {{FEWSHOT}}",
                                                  "new_template_name": "forked"}, headers=H(c))
    assert r.status_code == 200, r.text
    t = r.json()["template"]
    assert t["id"] not in (mine["id"], theirs["id"]) and t["version"] == 1
    from app.prompts import store as pstore
    rec = pstore.get_template(env.cfg, t["id"])
    assert rec["parent_id"] == theirs["id"] and rec["name"] == "forked" and rec["owner_id"] == env.uid("alice")
    assert pstore.get_version(env.cfg, theirs["id"])["text"] == txt       # the original is untouched
    # missing placeholder => P's validation error surfaces and nothing is appended
    n = len(store.read_batch(env.cfg, tc)["config_revisions"])
    err(c.post(f"{API}/batches/{tc}/config", json={"prompt_text": "no placeholder"}, headers=H(c)), 422,
        "fewshot_placeholder_missing")
    assert len(store.read_batch(env.cfg, tc)["config_revisions"]) == n
    # api key never in any batch response
    assert "sk-test" not in json.dumps(status_of(env, "alice", tc))


# --------------------------------------------------------------------------- regrade / rerun
def test_regrade_completed_batch_requeues_and_resets_run_base(env1):
    runs = []

    async def proc(ctx):
        runs.append((ctx.subtask_id, ctx.meta.get("run_base")))
        return Outcome("done")

    ex_mod.set_processor(proc)
    tc = api_create(env1, "alice", names=("a.jpg", "b.jpg"))
    wait_for(lambda: status_of(env1, "alice", tc)["status"] == "completed")
    sid = store.list_subtask_ids(env1.cfg, tc)[0]
    store.update_meta(env1.cfg, sid, lambda m: m.update(run_base=2))
    c = env1.clients["alice"]
    r = c.post(f"{API}/batches/{tc}/subtasks/{sid}/regrade", headers=H(c))
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "running" and "note" not in r.json()
    wait_for(lambda: len(runs) == 3 and status_of(env1, "alice", tc)["status"] == "completed")
    assert runs[2] == (sid, None)                                        # fresh retry budget
    m = store.read_meta(env1.cfg, sid)
    assert m["status"] == "done" and m["run_base"] is None
    assert "regrade" in [e["event"] for e in m["timeline"]]


def test_regrade_paused_stays_paused_until_resume(env1):
    tc = api_create(env1, "alice", names=("a.jpg", "b.jpg"))
    sid = store.list_subtask_ids(env1.cfg, tc)[0]
    set_sub(env1.cfg, sid, "failed", "ocr_failed")
    force(env1, tc, "paused_circuit_breaker", "ocr_failed")
    c = env1.clients["alice"]
    r = c.post(f"{API}/batches/{tc}/subtasks/{sid}/regrade", headers=H(c))
    assert r.status_code == 200 and r.json()["note"] == "queued_until_resume"
    assert r.json()["status"] == "paused_circuit_breaker"
    assert status_of(env1, "alice", tc)["status"] == "paused_circuit_breaker"
    assert store.read_meta(env1.cfg, sid)["status"] == "queued"


def test_regrade_validation(env1):
    tc = api_create(env1, "alice", names=("a.jpg", "b.jpg"))
    s1, s2 = store.list_subtask_ids(env1.cfg, tc)
    c = env1.clients["alice"]
    url = lambda s: f"{API}/batches/{tc}/subtasks/{s}/regrade"
    err(c.post(url(s1), headers=H(c)), 409, "invalid_transition")        # queued subtask
    set_sub(env1.cfg, s1, "done")
    err(env1.clients["bob"].post(url(s1), headers=H(env1.clients["bob"])), 404, "not_found")
    err(c.post(f"{API}/batches/{tc}/subtasks/{tc}99999/regrade", headers=H(c)), 404, "not_found")
    err(c.post(f"{API}/batches/{tc}/subtasks/zzz/regrade", headers=H(c)), 404, "not_found")
    assert post_action(env1, "alice", tc, "cancel").status_code == 200
    err(c.post(url(s1), headers=H(c)), 409, "invalid_transition")        # cancelled batch is final


def test_rerun_failed_resets_run_base(env):
    tc = api_create(env, "alice", names=("a.jpg", "b.jpg", "c.jpg"))
    sids = store.list_subtask_ids(env.cfg, tc)
    set_sub(env.cfg, sids[0], "failed", "ocr_failed", run_base=3)
    set_sub(env.cfg, sids[1], "api_error", "api_error", run_base=1)
    set_sub(env.cfg, sids[2], "done", run_base=1)
    force(env, tc, "paused_user", "user")
    assert post_action(env, "alice", tc, "rerun-failed").status_code == 200
    ms = {m["subtask_id"]: m for m in store.read_metas(env.cfg, tc)}
    assert ms[sids[0]]["run_base"] is None and ms[sids[1]]["run_base"] is None
    assert ms[sids[2]]["run_base"] == 1 or ms[sids[2]]["status"] == "done"   # done subtasks untouched


# --------------------------------------------------------------------------- failures / detail endpoints
def test_failures_endpoint_shape_and_access(env):
    tc = api_create(env, "alice", names=("a.jpg", "b.jpg", "c.jpg"))
    s = store.list_subtask_ids(env.cfg, tc)
    set_sub(env.cfg, s[0], "failed", "empty_response")
    set_sub(env.cfg, s[1], "failed", "empty_response")
    set_sub(env.cfg, s[2], "api_error", "api_error")
    c = env.clients["alice"]
    j = c.get(f"{API}/batches/{tc}/failures").json()
    assert set(j) == {"pause_reason", "samples", "reasons"}
    assert j["reasons"] == {"empty_response": 2, "api_error": 1} and j["samples"] == [] and j["pause_reason"] is None
    assert env.clients["bob"].get(f"{API}/batches/{tc}/failures").status_code == 404
    assert env.clients["rooty"].get(f"{API}/batches/{tc}/failures").status_code == 200


def test_subtask_detail_and_attempt_endpoints(env):
    tc = api_create(env, "alice", names=("a.jpg",))
    sid = store.list_subtask_ids(env.cfg, tc)[0]
    sdir = paths.subtask_dir(env.cfg, sid)
    atomic_io.write_json_atomic(sdir / "ocr.json", {"mode": "vlm", "transcription": "<empty>"})
    atomic_io.write_json_atomic(sdir / "grading.json", {"schema_version": 1, "result": {"判分": 7}})
    atomic_io.write_json_atomic(sdir / "review.json", {"final_score": 8})
    write_attempt(env.cfg, sid, 1, response={"attempt": 1, "http_status": 200, "latency_ms": 42,
                                            "classification": "ok", "error_code": "json_parse_failed",
                                            "usage": {"prompt": 1, "completion": 2, "total": 3}})
    write_attempt(env.cfg, sid, 2, response={"attempt": 2, "http_status": 200, "latency_ms": 7,
                                            "classification": "ok", "error_code": None, "usage": None})
    write_attempt(env.cfg, sid, 3, response={"attempt": 3, "error": {"code": "interrupted"},
                                            "classification": "interrupted", "error_code": "interrupted"})
    c = env.clients["alice"]
    j = c.get(f"{API}/batches/{tc}/subtasks/{sid}").json()
    assert set(j) == {"meta", "ocr", "grading", "attempts", "review"}
    assert j["meta"]["subtask_id"] == sid and j["ocr"]["mode"] == "vlm" and j["grading"]["result"] == {"判分": 7}
    assert j["review"] == {"final_score": 8}
    assert [a["attempt"] for a in j["attempts"]] == [1, 2, 3]
    assert j["attempts"][0] == {"attempt": 1, "error_code": "json_parse_failed", "classification": "ok",
                                "http_status": 200, "latency_ms": 42, "usage": {"prompt": 1, "completion": 2, "total": 3}}
    assert j["attempts"][2]["error_code"] == "interrupted"
    a = c.get(f"{API}/batches/{tc}/subtasks/{sid}/attempts/2").json()
    assert a["request"] == {"attempt": 2, "model": "m1"} and a["response"]["latency_ms"] == 7
    assert c.get(f"{API}/batches/{tc}/subtasks/{sid}/attempts/9").status_code == 404
    assert c.get(f"{API}/batches/{tc}/subtasks/{sid}/attempts/abc").status_code == 404
    assert c.get(f"{API}/batches/{tc}/subtasks/{sid}/attempts/0").status_code == 404
    assert c.get(f"{API}/batches/{tc}/subtasks/{sid}/attempts/..%2F..").status_code == 404
    bob = env.clients["bob"]
    assert bob.get(f"{API}/batches/{tc}/subtasks/{sid}").status_code == 404
    assert bob.get(f"{API}/batches/{tc}/subtasks/{sid}/attempts/1").status_code == 404
    assert env.clients["rooty"].get(f"{API}/batches/{tc}/subtasks/{sid}").status_code == 200


def test_subtask_detail_without_files(env):
    tc = api_create(env, "alice")
    sid = store.list_subtask_ids(env.cfg, tc)[0]
    j = env.clients["alice"].get(f"{API}/batches/{tc}/subtasks/{sid}").json()
    assert j["ocr"] is None and j["grading"] is None and j["review"] is None and j["attempts"] == []
