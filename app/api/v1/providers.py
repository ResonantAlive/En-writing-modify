"""Providers API. API keys are write-only: responses carry key_last4 only."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from ...auth.audit import audit
from ...auth.deps import Principal, cfg_of, client_ip, require_user
from ...errors import NotFound
from ...providers import store, third_party

router = APIRouter(prefix="/providers", tags=["providers"])


class ProviderIn(BaseModel):
    alias: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    default_model: Optional[str] = None
    shared: Optional[bool] = None


def _visible(cfg, p: Principal, provider_id: str) -> dict:
    if not store.can_use(cfg, p, provider_id):
        raise NotFound()
    return store.get_provider(cfg, provider_id)


@router.get("")
@router.get("/", include_in_schema=False)
def list_providers(request: Request, p: Principal = Depends(require_user)):
    return store.list_visible(cfg_of(request), p)


@router.post("", status_code=201)
@router.post("/", status_code=201, include_in_schema=False)
def create_provider(body: ProviderIn, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    rec = store.create(cfg, p, body.model_dump())
    audit(cfg, p.user_id, "provider.create", rec["id"], {"base_url": rec["base_url"], "shared": rec["shared"]},
          client_ip(request), p.role)
    return store.public_view(rec)


@router.patch("/{provider_id}")
def patch_provider(provider_id: str, body: ProviderIn, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    rec = store.update(cfg, p, provider_id, body.model_dump(exclude_none=True))
    changed = sorted(k for k, v in body.model_dump().items() if v is not None)
    audit(cfg, p.user_id, "provider.update", provider_id, {"fields": changed}, client_ip(request), p.role)
    return store.public_view(rec)


@router.delete("/{provider_id}", status_code=204)
def delete_provider(provider_id: str, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    store.soft_delete(cfg, p, provider_id)
    audit(cfg, p.user_id, "provider.delete", provider_id, {}, client_ip(request), p.role)


@router.post("/{provider_id}/ack")
def ack_provider(provider_id: str, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    prov = _visible(cfg, p, provider_id)
    third_party.record_ack(cfg, p.user_id, prov)
    audit(cfg, p.user_id, "provider.ack", provider_id, {"host": third_party.host(prov)},
          client_ip(request), p.role)
    return {"is_third_party": bool(prov["is_third_party"]), "acked": not third_party.needs_ack(cfg, p.user_id, prov)
            or not prov["is_third_party"], "host": third_party.host(prov)}


@router.get("/{provider_id}/ack-status")
def ack_status(provider_id: str, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    prov = _visible(cfg, p, provider_id)
    return {"is_third_party": bool(prov["is_third_party"]),
            "acked": not third_party.needs_ack(cfg, p.user_id, prov),
            "host": third_party.host(prov)}
