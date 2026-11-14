from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from ...auth import invite
from ...auth import users as U
from ...auth.audit import audit
from ...auth.deps import Principal, cfg_of, client_ip, require_root
from ...errors import AppError

router = APIRouter(prefix="/admin", tags=["admin"])

# Hooks other modules (executor, M2+) register to react to an account being disabled
# (running batches of that owner -> paused_user / owner_disabled).
on_user_disabled: list = []


class RoleIn(BaseModel):
    role: str


class ResetIn(BaseModel):
    which: str = "both"      # root | user | both


@router.get("/users")
def users(request: Request, p: Principal = Depends(require_root)):
    return U.list_users(cfg_of(request))


@router.post("/users/{user_id}/disable")
def disable(user_id: str, request: Request, p: Principal = Depends(require_root)):
    cfg = cfg_of(request)
    res = U.set_disabled(cfg, user_id, True)
    for hook in on_user_disabled:
        hook(cfg, user_id)
    audit(cfg, p.user_id, "admin.user_disable", user_id, {}, client_ip(request), p.role)
    return res


@router.post("/users/{user_id}/enable")
def enable(user_id: str, request: Request, p: Principal = Depends(require_root)):
    cfg = cfg_of(request)
    res = U.set_disabled(cfg, user_id, False)
    audit(cfg, p.user_id, "admin.user_enable", user_id, {}, client_ip(request), p.role)
    return res


@router.post("/users/{user_id}/role")
def role(user_id: str, body: RoleIn, request: Request, p: Principal = Depends(require_root)):
    cfg = cfg_of(request)
    res = U.set_role(cfg, user_id, body.role)
    audit(cfg, p.user_id, "admin.role_change", user_id, {"role": body.role}, client_ip(request), p.role)
    return res


@router.post("/invite-codes/reset")
def reset_invites(body: ResetIn, request: Request, p: Principal = Depends(require_root)):
    cfg = cfg_of(request)
    if body.which not in ("root", "user", "both"):
        raise AppError("invalid_which", "which must be root, user or both", 422)
    invite.reset_codes(cfg, body.which)
    audit(cfg, p.user_id, "admin.invite_reset", None, {"which": body.which}, client_ip(request), p.role)
    path = cfg.app_dir / "invite_codes.json"
    # Deliberately does NOT return the new codes: read them on the server.
    return {"ok": True, "message": f"reset; new code(s) written on the server to {path}"}


@router.get("/tokens")
def all_tokens(request: Request, p: Principal = Depends(require_root)):
    return U.list_tokens(cfg_of(request), None)


@router.delete("/tokens/{token_id}")
def revoke_any(token_id: str, request: Request, p: Principal = Depends(require_root)):
    cfg = cfg_of(request)
    rec = U.revoke_token(cfg, token_id, only_owner=None)
    audit(cfg, p.user_id, "admin.token_revoke", token_id, {"owner": rec.get("owner_id")}, client_ip(request), p.role)
    return rec
