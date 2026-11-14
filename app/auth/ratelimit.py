"""Brute-force protection persisted in data/state/auth_state.json.
scope 'login'  : key = username_lower|ip
scope 'invite' : key = ip
Uses the real wall clock (time.time) so it is independent of the configurable app clock."""
from __future__ import annotations

import time

from ..config import AppConfig
from ..core import atomic_io
from ..core.locks import file_lock
from ..errors import AppError

_time = time.time
_STALE_S = 3600
_MAX_ENTRIES = 5000


def _path(cfg: AppConfig):
    return cfg.state_dir / "auth_state.json"


def _load(cfg: AppConfig) -> dict:
    d = atomic_io.read_json(_path(cfg), None) or {}
    d.setdefault("schema_version", 1)
    d.setdefault("login", {})
    d.setdefault("invite", {})
    return d


def _prune(d: dict, now: float) -> None:
    for scope in ("login", "invite"):
        m = d[scope]
        for k in [k for k, e in m.items()
                  if e.get("locked_until_ts", 0) < now and now - e.get("last_fail_ts", 0) > _STALE_S]:
            del m[k]
        if len(m) > _MAX_ENTRIES:
            for k in sorted(m, key=lambda k: m[k].get("last_fail_ts", 0))[: len(m) - _MAX_ENTRIES]:
                del m[k]


def ensure_not_locked(cfg: AppConfig, scope: str, key: str) -> None:
    e = _load(cfg)[scope].get(key)
    if e and e.get("locked_until_ts", 0) > _time():
        raise AppError("too_many_attempts", "too many failed attempts, try again later", 429)


def record_failure(cfg: AppConfig, scope: str, key: str) -> None:
    lim = cfg["lockout"]
    with file_lock(cfg.lock_path("auth_state")):
        d = _load(cfg)
        now = _time()
        e = d[scope].get(key) or {"fails": 0}
        if e.get("locked_until_ts", 0) and e["locked_until_ts"] <= now:
            e = {"fails": 0}                      # lock expired: start over
        if now - e.get("last_fail_ts", now) > _STALE_S:
            e["fails"] = 0
        e["fails"] += 1
        e["last_fail_ts"] = now
        if e["fails"] >= lim["fails"]:
            e["locked_until_ts"] = now + lim["minutes"] * 60
        d[scope][key] = e
        _prune(d, now)
        atomic_io.write_json_atomic(_path(cfg), d)


def record_success(cfg: AppConfig, scope: str, key: str) -> None:
    with file_lock(cfg.lock_path("auth_state")):
        d = _load(cfg)
        if key in d[scope]:
            del d[scope][key]
            atomic_io.write_json_atomic(_path(cfg), d)
