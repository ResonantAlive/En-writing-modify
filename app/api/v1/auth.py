from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel

from ...auth import invite, ratelimit
from ...auth import users as U
from ...auth.audit import audit
from ...auth.deps import (CSRF_COOKIE, SESSION_COOKIE, Principal, cfg_of, clear_auth_cookies,
                          client_ip, require_user, set_auth_cookies)
from ...errors import AppError

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterIn(BaseModel):
    invite_code: str
    username: str
    password: str


class LoginIn(BaseModel):
    username: str
    password: str


@router.post("/register", status_code=201)
def register(body: RegisterIn, request: Request):
    cfg, ip = cfg_of(request), client_ip(request)
    ratelimit.ensure_not_locked(cfg, "invite", ip)
    role = invite.role_for_code(cfg, body.invite_code)
    if role is None:
        ratelimit.record_failure(cfg, "invite", ip)
        raise AppError("invalid_invite", "invalid invite code", 403)
    user = U.register(cfg, role, body.username, body.password)
    ratelimit.record_success(cfg, "invite", ip)
    audit(cfg, user["id"], "auth.register", user["id"], {"role": role}, ip, role)
    return user


@router.post("/login")
def login(body: LoginIn, request: Request, response: Response):
    cfg, ip = cfg_of(request), client_ip(request)
    key = f"{body.username.lower()[:64]}|{ip}"
    ratelimit.ensure_not_locked(cfg, "login", key)
    user = U.authenticate(cfg, body.username, body.password)
    if not user:
        ratelimit.record_failure(cfg, "login", key)
        audit(cfg, None, "auth.login_failed", None, {"username": body.username[:64]}, ip)
        raise AppError("invalid_credentials", "invalid username or password", 401)
    ratelimit.record_success(cfg, "login", key)
    token, csrf = U.create_session(cfg, user["id"], ip)
    set_auth_cookies(response, cfg, token, csrf)
    audit(cfg, user["id"], "auth.login", user["id"], {}, ip, user["role"])
    return {"user": U.public_user(user), "csrf_token": csrf}


@router.post("/logout")
def logout(request: Request, response: Response, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    raw = request.cookies.get(SESSION_COOKIE)
    if raw:
        U.delete_session(cfg, raw)
    clear_auth_cookies(response)
    audit(cfg, p.user_id, "auth.logout", p.user_id, {}, client_ip(request), p.role)
    return {"ok": True}


@router.get("/me")
def me(request: Request, p: Principal = Depends(require_user)):
    return {"id": p.user_id, "username": p.username, "account_role": p.account_role,
            "role": p.role, "via": p.via, "csrf_token": request.cookies.get(CSRF_COOKIE)}
