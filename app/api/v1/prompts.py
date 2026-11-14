"""Prompt templates API. Versions are immutable; builtin templates are read-only."""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ...auth.audit import audit
from ...auth.deps import Principal, cfg_of, client_ip, require_root, require_user
from ...errors import AppError, NotFound
from ...prompts import assist as assist_mod
from ...prompts import store
from ...providers import store as pstore
from ...providers import third_party

router = APIRouter(prefix="/prompts", tags=["prompts"])


class VersionIn(BaseModel):
    text: Optional[str] = None
    input_mode: Optional[str] = None
    score_key: Optional[str] = None
    score_aliases: Optional[list] = None
    criteria_key: Optional[str] = None
    required_keys: Optional[list] = None
    max_chars: Optional[int] = None
    auto_append_fewshot: Optional[bool] = None


class TemplateIn(VersionIn):
    name: Optional[str] = None
    parent_id: Optional[str] = None
    parent_version: Optional[int] = None


class ShareIn(BaseModel):
    shared: bool


class AssistIn(BaseModel):
    provider_id: str
    requirement: str = Field(min_length=1, max_length=20000)
    include_history: bool = False
    history_scope: Literal["own", "all"] = "own"
    topic: Optional[str] = None
    model: Optional[str] = None
    third_party_ack: Optional[bool] = None


def _view(cfg, p: Principal, tid: str) -> dict:
    if not store.can_use(cfg, p, tid):
        raise NotFound()
    return store.get_template(cfg, tid)


@router.get("")
@router.get("/", include_in_schema=False)
def list_templates(request: Request, input_mode: Optional[Literal["text", "image"]] = None,
                   p: Principal = Depends(require_user)):
    return store.list_visible(cfg_of(request), p, input_mode)


# NOTE: declared before /{template_id} routes so "assist" is never taken for an id.
@router.post("/assist")
def assist(body: AssistIn, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    if body.history_scope == "all" and not p.is_root:
        raise AppError("forbidden", "history_scope=all is only available to root", 403)
    if not pstore.can_use(cfg, p, body.provider_id):
        raise NotFound()
    prov = pstore.get_provider(cfg, body.provider_id)
    if third_party.needs_ack(cfg, p.user_id, prov):
        if body.third_party_ack is not True:
            raise AppError("third_party_ack_required",
                           "acknowledge that data is sent to a third-party provider", 409,
                           {"host": third_party.host(prov)})
        third_party.record_ack(cfg, p.user_id, prov)
    owner = None if body.history_scope == "all" else p.user_id
    text = assist_mod.draft(cfg, prov, body.model or prov["default_model"], body.requirement, owner,
                            body.include_history, body.topic)
    return {"draft": text}


@router.post("", status_code=201)
@router.post("/", status_code=201, include_in_schema=False)
def create_template(body: TemplateIn, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    rec = store.create_template(cfg, p, body.model_dump())
    audit(cfg, p.user_id, "prompt.create", rec["id"], {"parent_id": rec["parent_id"]}, client_ip(request), p.role)
    return rec


@router.get("/{template_id}")
def get_template(template_id: str, request: Request, p: Principal = Depends(require_user)):
    return _view(cfg_of(request), p, template_id)


@router.post("/{template_id}/versions", status_code=201)
def add_version(template_id: str, body: VersionIn, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    t, ver = store.add_version(cfg, p, template_id, body.model_dump())
    if t["owner_id"] != p.user_id:               # root editing someone else's template
        audit(cfg, p.user_id, "prompt.root_edit", template_id,
              {"owner_id": t["owner_id"], "new_version": ver["version"]}, client_ip(request), p.role)
    return t


@router.post("/{template_id}/share")
def share_template(template_id: str, body: ShareIn, request: Request, p: Principal = Depends(require_root)):
    cfg = cfg_of(request)
    t = store.set_shared(cfg, template_id, body.shared)
    audit(cfg, p.user_id, "prompt.share", template_id, {"shared": body.shared}, client_ip(request), p.role)
    return store.summary(t)


@router.delete("/{template_id}", status_code=204)
def delete_template(template_id: str, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    t = store.soft_delete(cfg, p, template_id)
    if t["owner_id"] != p.user_id:
        audit(cfg, p.user_id, "prompt.root_delete", template_id, {"owner_id": t["owner_id"]},
              client_ip(request), p.role)
