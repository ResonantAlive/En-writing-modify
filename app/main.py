"""FastAPI application factory."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

import asyncio

from .api.v1 import admin, auth, batches, prompts as prompts_api, providers as providers_api, tokens, uploads as uploads_api
from .api.v1 import sheets as sheets_api
from .auth.invite import ensure_invite_codes
from .batches import executor as executor_mod, results as results_mod
from .batches.executor import Executor
from .core.bodylimit import BodyLimit
from .grading import pipeline
from .providers import third_party
from .staging import uploads
from .config import PROJECT_ROOT, AppConfig, load_config
from .core.locks import ExecutorLock
from .errors import AppError
from .logging_setup import setup_logging

log = logging.getLogger("wm.main")


def create_app(data_dir: Path | str | None = None, cfg: AppConfig | None = None,
               acquire_executor_lock: bool = True, enable_grading: bool = True) -> FastAPI:
    """enable_grading=False: do not register the real grading processor (tests that drive the executor
    with their own fake processor pass False)."""
    cfg = cfg or load_config(data_dir)
    warnings = cfg.check_network()          # raises StartupError -> refuse to start
    setup_logging(cfg)
    third_party.register_hooks()            # provider/template validation + third-party ack hooks
    if results_mod.on_batch_finished not in executor_mod.on_batch_finished:
        executor_mod.on_batch_finished.append(results_mod.on_batch_finished)   # result-<task_code>.json

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        lock = None
        if acquire_executor_lock:
            lock = ExecutorLock(cfg.lock_path("executor"))
            lock.acquire()                  # raises ExecutorLockError if another process holds it
            app.state.executor_lock = lock
        if ensure_invite_codes(cfg):
            msg = f"invite codes saved to {cfg.app_dir / 'invite_codes.json'}"
            log.info(msg)
            print(msg)
        for w in warnings:
            log.warning(w)
            print(w)
        executor = None
        cleaner = None
        if lock:   # only the process holding the executor lock runs batches
            executor = Executor(cfg)
            app.state.executor = executor
            if enable_grading:
                pipeline.register()         # sets the real subtask processor (and grading hooks)
                from .sheets import store as sheets_store
                sheets_store.kick(cfg)      # blank sheets left pending by a restart
            await executor.start()

            async def _clean_loop():
                while True:
                    try:
                        await asyncio.to_thread(uploads.cleanup_expired, cfg)
                    except Exception:
                        log.exception("staging cleanup failed")
                    await asyncio.sleep(1800)
            cleaner = asyncio.create_task(_clean_loop())
        try:
            yield
        finally:
            if cleaner:
                cleaner.cancel()
            if executor:
                await executor.stop()
            if lock:
                lock.release()

    app = FastAPI(title="writing_modify", version="0.1.0", lifespan=lifespan)
    app.state.cfg = cfg

    if cfg.net["cors_origins"]:
        app.add_middleware(CORSMiddleware, allow_origins=cfg.net["cors_origins"], allow_credentials=True,
                           allow_methods=["*"], allow_headers=["*"])

    app.add_middleware(BodyLimit, cfg=cfg)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        return resp

    @app.exception_handler(AppError)
    async def _app_error(_: Request, e: AppError):
        return JSONResponse(e.body(), status_code=e.status)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, e: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed", 401: "unauthenticated", 403: "forbidden",
                413: "body_too_large"}.get(
            e.status_code, "http_error")
        return JSONResponse({"code": code, "message": str(e.detail), "detail": None}, status_code=e.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError):
        errs = [{"loc": x.get("loc"), "msg": x.get("msg"), "type": x.get("type")} for x in e.errors()]
        return JSONResponse({"code": "validation_error", "message": "invalid request", "detail": errs},
                            status_code=422)

    for r in (auth.router, tokens.router, admin.router, uploads_api.router, batches.router,
              providers_api.router, prompts_api.router, sheets_api.router):
        app.include_router(r, prefix="/api/v1")

    front = PROJECT_ROOT / "frontend"
    if (front / "index.html").exists():
        app.mount("/", StaticFiles(directory=front, html=True), name="frontend")   # build output only, never data/
    return app
