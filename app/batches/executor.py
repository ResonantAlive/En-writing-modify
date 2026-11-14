"""Background batch executor (runs in the app's event loop).

The processor (M3 registers the real one with set_processor) is `async def fn(ctx) -> Outcome`.
Without a registered processor nothing is dispatched and batches stay queued.
State lives on disk (batch.json / meta.json via app.batches.store); the loop re-reads it every tick, so
pause/resume/cancel from the API simply take effect on the next tick (poke() makes that immediate).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from ..errors import AppError
from . import breaker, recover, store

log = logging.getLogger("wm.executor")

OUTCOME_STATUSES = {"done", "failed", "api_error"}


@dataclass
class Outcome:
    status: str                      # done | failed | api_error
    fail_reason: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ProcessCtx:
    cfg: object
    batch: dict                      # snapshot
    subtask_id: str
    meta: dict                       # snapshot (status already `running`)
    config_rev: int


_processor: Callable[[ProcessCtx], Awaitable[Outcome]] | None = None
on_batch_finished: list[Callable] = []           # fn(cfg, task_code), called when a batch becomes completed
_LIVE: list["Executor"] = []


def set_processor(fn):
    """Register (or clear with None) the subtask processor. Returns the previous one."""
    global _processor
    prev, _processor = _processor, fn
    return prev


def get_processor():
    return _processor


def poke(cfg) -> None:
    """Thread-safe: ask live executor(s) for this data dir to re-scan now."""
    for e in list(_LIVE):
        if str(e.cfg.data_dir) == str(cfg.data_dir):
            e.wake_threadsafe()


def abort_batch(cfg, task_code: str) -> None:
    """Thread-safe: cancel in-flight subtasks of a batch (after the API marked them cancelled)."""
    for e in list(_LIVE):
        if str(e.cfg.data_dir) == str(cfg.data_dir):
            e.abort_threadsafe(task_code)


class Executor:
    TICK_S = 0.5

    def __init__(self, cfg):
        self.cfg = cfg
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        self._main: asyncio.Task | None = None
        self._inflight: dict[str, asyncio.Task] = {}
        self._aborted: set[str] = set()
        self._stopping = False
        self._last_tc: str | None = None

    # ---- lifecycle
    async def start(self) -> None:
        await asyncio.to_thread(recover.recover, self.cfg)
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self._stopping = False
        self._main = asyncio.create_task(self._run(), name="wm-executor")
        _LIVE.append(self)

    async def stop(self) -> None:
        self._stopping = True
        if self in _LIVE:
            _LIVE.remove(self)
        if self._main:
            self._main.cancel()
            try:
                await self._main
            except (asyncio.CancelledError, Exception):
                pass
            self._main = None
        tasks = list(self._inflight.values())
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ---- wakeups
    async def wake(self) -> None:
        if self._wake:
            self._wake.set()

    def wake_threadsafe(self) -> None:
        if self._loop and self._wake and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._wake.set)
            except RuntimeError:
                pass

    def abort_threadsafe(self, task_code: str) -> None:
        if self._loop and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._abort, task_code)
            except RuntimeError:
                pass

    def _abort(self, task_code: str) -> None:
        for sid, t in list(self._inflight.items()):
            if sid[:12] == task_code:
                self._aborted.add(sid)
                t.cancel()

    # ---- API for M3
    async def pause(self, task_code: str, status: str, reason: str | None) -> bool:
        """Pause a queued/running batch (in-flight subtasks finish). False if it was not pausable."""
        if status not in store.PAUSED:
            raise ValueError(f"not a paused status: {status}")

        def do() -> bool:
            try:
                store.set_status(self.cfg, task_code, "pause", status, allowed_from=("queued", "running"),
                                 by="executor", reason=reason, event="paused")
                return True
            except AppError:
                return False
        return await asyncio.to_thread(do)

    # ---- main loop
    def _concurrency(self) -> int:
        try:
            return max(1, int(self.cfg["concurrency"]))
        except Exception:
            return 1

    async def _run(self) -> None:
        while not self._stopping:
            self._wake.clear()
            try:
                free = self._concurrency() - len(self._inflight)
                ctxs = await asyncio.to_thread(self._plan, set(self._inflight), max(free, 0))
                for ctx in ctxs:
                    self._inflight[ctx.subtask_id] = asyncio.create_task(self._run_one(ctx))
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("executor tick failed")
            try:
                await asyncio.wait_for(self._wake.wait(), self.TICK_S)
            except asyncio.TimeoutError:
                pass

    def _plan(self, inflight: set[str], free: int) -> list[ProcessCtx]:
        """Sync (runs in a worker thread): batch queued->running, completion, and pick subtasks to start."""
        cfg = self.cfg
        if _processor is None:
            return []
        queues: dict[str, list[str]] = {}
        for tc in store.active_codes(cfg):
            b = store.read_batch(cfg, tc)
            if b is None:
                store.track(cfg, tc, "gone")
                continue
            if b["status"] == "queued":
                try:
                    b = store.set_status(cfg, tc, "start", "running", allowed_from=("queued",),
                                         by="executor", event="started")
                except AppError:
                    continue
            if b["status"] != "running":
                store.track(cfg, tc, b["status"])
                continue
            metas = store.read_metas(cfg, tc)
            if not any(m.get("status") in ("queued", "running") for m in metas):
                self._complete(tc)
                continue
            queues[tc] = [m["subtask_id"] for m in metas
                          if m.get("status") == "queued" and m["subtask_id"] not in inflight]
        order = sorted(queues)
        if self._last_tc in order:               # keep rotating across ticks (fair with concurrency=1)
            i = order.index(self._last_tc) + 1
            order = order[i:] + order[:i]
        elif self._last_tc:
            order = [c for c in order if c > self._last_tc] + [c for c in order if c <= self._last_tc]
        picked: list[tuple[str, str]] = []
        while free > 0 and any(queues.values()):
            for tc in order:
                if free > 0 and queues[tc]:
                    picked.append((tc, queues[tc].pop(0)))
                    self._last_tc = tc
                    free -= 1
        out = []
        for tc, sid in picked:
            batch = store.read_batch(cfg, tc)
            if batch is None or batch["status"] != "running":
                continue
            rev = batch["config_revisions"][-1]["rev"]

            def start(m: dict, rev=rev):
                if m.get("status") != "queued":
                    return None
                m["status"] = "running"
                m["attempts"] = int(m.get("attempts", 0)) + 1
                m["config_rev"] = rev
                store.append_timeline(m, "started", None, {"attempt": m["attempts"]})
                return dict(m)
            snap = store.update_meta(cfg, sid, start)
            if snap is not None:
                out.append(ProcessCtx(cfg, batch, sid, snap, rev))
        return out

    def _complete(self, tc: str) -> None:
        try:
            store.set_status(self.cfg, tc, "complete", "completed", allowed_from=("running",),
                             by="executor", event="completed")
        except AppError:
            return
        for hook in list(on_batch_finished):
            try:
                hook(self.cfg, tc)
            except Exception:
                log.exception("on_batch_finished hook failed")

    async def _run_one(self, ctx: ProcessCtx) -> None:
        sid = ctx.subtask_id
        outcome: Outcome | None = None
        try:
            outcome = await _processor(ctx) if _processor else Outcome("failed", "internal_error")
            if not isinstance(outcome, Outcome) or outcome.status not in OUTCOME_STATUSES:
                outcome = Outcome("failed", "internal_error")
        except asyncio.CancelledError:
            if sid in self._aborted:
                self._aborted.discard(sid)           # user cancel: API already marked the subtask cancelled
                self._close_attempts(sid, "cancelled")
                self._finish(sid, None)
            else:                                   # shutdown: put it back, restart resumes it
                self._close_attempts(sid, "interrupted")
                self._requeue(sid)
                raise
            return
        except Exception as e:
            log.error("processor raised %s for subtask %s", type(e).__name__, sid)
            outcome = Outcome("failed", "internal_error")
        finally:
            self._inflight.pop(sid, None)
        try:
            await asyncio.to_thread(self._finish, sid, outcome)
        except Exception:
            log.exception("could not record outcome of %s", sid)
        try:                                        # failure-ratio breaker over all terminal subtasks
            await asyncio.to_thread(breaker.evaluate_ratio, self.cfg, sid[:12])
        except Exception:
            log.exception("ratio breaker evaluation failed for %s", sid[:12])
        await self.wake()

    def _finish(self, sid: str, outcome: Outcome | None) -> None:
        def fn(m: dict) -> None:
            if m.get("status") != "running":
                return                                # cancelled meanwhile: keep it
            if outcome is None:
                m["status"] = "cancelled"
                store.append_timeline(m, "cancelled", None, {"in_flight": True})
                return
            m["status"] = outcome.status
            m["fail_reason"] = outcome.fail_reason if outcome.status != "done" else None
            if outcome.warnings:
                m["warnings"] = list(m.get("warnings", [])) + list(outcome.warnings)
            store.append_timeline(m, outcome.status, None, {"fail_reason": m["fail_reason"]})
        store.update_meta(self.cfg, sid, fn)

    def _close_attempts(self, sid: str, code: str) -> None:
        try:
            recover.close_open_attempts(self.cfg, sid, code)
        except Exception:
            log.exception("could not close open attempts of %s", sid)

    def _requeue(self, sid: str) -> None:
        def fn(m: dict) -> None:
            if m.get("status") == "running":
                m["status"] = "queued"
                store.append_timeline(m, "interrupted_by_shutdown", None, {})
        try:
            store.update_meta(self.cfg, sid, fn)
        except Exception:
            pass
