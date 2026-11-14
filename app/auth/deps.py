"""Request authentication: Bearer token or session cookie; CSRF for cookie-authenticated writes.
Effective role = min(token role, account's current role)."""
from __future__ import annotations

import hmac
import ipaddress
from dataclasses import dataclass

from fastapi import Request, Response

from ..config import AppConfig
from ..errors import AppError
from . import users as U

SESSION_COOKIE = "wm_session"
CSRF_COOKIE = "wm_csrf"
CSRF_HEADER = "x-csrf-token"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


@dataclass
class Principal:
    user_id: str
    username: str
    account_role: str
    role: str                  # effective role
    via: str                   # 'bearer' | 'cookie'
    token_id: str | None = None

    @property
    def is_root(self) -> bool:
        return self.role == "root"


def cfg_of(request: Request) -> AppConfig:
    return request.app.state.cfg


def client_ip(request: Request) -> str:
    cfg = cfg_of(request)
    peer = request.client.host if request.client else "unknown"
    trusted = cfg.net["trusted_proxies"]
    if peer in trusted:
        xff = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        for hop in reversed(xff):               # rightmost non-trusted hop is the real client
            if hop not in trusted:
                try:
                    ipaddress.ip_address(hop)
                    return hop
                except ValueError:
                    break
    return peer


def set_auth_cookies(response: Response, cfg: AppConfig, token: str, csrf: str) -> None:
    age = int(cfg["session_days"] * 86400)
    kw = dict(max_age=age, samesite="lax", secure=cfg.secure_cookies, path="/")
    response.set_cookie(SESSION_COOKIE, token, httponly=True, **kw)
    response.set_cookie(CSRF_COOKIE, csrf, httponly=False, **kw)


def clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")


def _unauth() -> AppError:
    return AppError("unauthenticated", "authentication required", 401)


def require_user(request: Request) -> Principal:
    cfg = cfg_of(request)
    auth = request.headers.get("authorization")
    if auth:
        scheme, _, raw = auth.partition(" ")
        if scheme.lower() != "bearer" or not raw.strip():
            raise _unauth()
        found = U.resolve_token(cfg, raw.strip())
        if not found:
            raise _unauth()
        user, tok = found
        return Principal(user["id"], user["username"], user["role"],
                         U.min_role(tok["role"], user["role"]), "bearer", tok["id"])
    raw = request.cookies.get(SESSION_COOKIE)
    found = U.resolve_session(cfg, raw) if raw else None
    if not found:
        raise _unauth()
    user, sess = found
    if request.method not in SAFE_METHODS:
        hdr = request.headers.get(CSRF_HEADER, "")
        ck = request.cookies.get(CSRF_COOKIE, "")
        if not hdr or not hmac.compare_digest(hdr, ck) or not hmac.compare_digest(hdr, sess["csrf"]):
            raise AppError("csrf_failed", "missing or invalid CSRF token", 403)
    return Principal(user["id"], user["username"], user["role"], user["role"], "cookie")


def require_root(request: Request) -> Principal:
    p = require_user(request)
    if not p.is_root:
        raise AppError("forbidden", "root role required", 403)
    return p
