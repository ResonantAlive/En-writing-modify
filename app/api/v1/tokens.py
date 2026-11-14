from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from ...auth import users as U
from ...auth.audit import audit
from ...auth.deps import Principal, cfg_of, client_ip, require_user
from ...errors import AppError

router = APIRouter(prefix="/tokens", tags=["tokens"])


class TokenIn(BaseModel):
    name: str
    role: str = "user"
    expires_in_days: float | None = None


@router.post("", status_code=201)
def create(body: TokenIn, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    if body.role not in U.ROLE_RANK:
        raise AppError("invalid_role", "role must be root or user", 422)
    if U.ROLE_RANK[body.role] > U.ROLE_RANK[p.role]:
        raise AppError("role_exceeds_creator", "token role cannot exceed your own role", 403)
    if body.expires_in_days is not None and body.expires_in_days <= 0:
        raise AppError("invalid_expiry", "expires_in_days must be positive", 422)
    raw, rec = U.create_token(cfg, p.user_id, body.name, body.role, body.expires_in_days)
    audit(cfg, p.user_id, "token.create", rec["id"], {"role": body.role, "name": rec["name"]},
          client_ip(request), p.role)
    return {**rec, "token": raw}      # plaintext shown exactly once


@router.get("")
def list_mine(request: Request, p: Principal = Depends(require_user)):
    return U.list_tokens(cfg_of(request), p.user_id)


@router.delete("/{token_id}")
def revoke(token_id: str, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    rec = U.revoke_token(cfg, token_id, only_owner=p.user_id)    # other accounts' tokens -> 404
    audit(cfg, p.user_id, "token.revoke", token_id, {}, client_ip(request), p.role)
    return rec
