"""Answer-sheet templates API: blank sheets that teach local OCR which text is printed on the paper."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...auth.audit import audit
from ...auth.deps import Principal, cfg_of, client_ip, require_root, require_user
from ...errors import AppError
from ...sheets import store

router = APIRouter(prefix="/sheet-templates", tags=["sheet-templates"])

MAX_FILES_PER_REQUEST = 10


class CreateIn(BaseModel):
    name: str


class RenameIn(BaseModel):
    name: str


class ShareIn(BaseModel):
    shared: bool


class PageIn(BaseModel):
    disabled: Optional[bool] = None


@router.get("")
@router.get("/", include_in_schema=False)
def list_templates(request: Request, p: Principal = Depends(require_user)):
    return store.list_visible(cfg_of(request), p)


@router.post("", status_code=201)
@router.post("/", status_code=201, include_in_schema=False)
def create_template(body: CreateIn, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    t = store.create(cfg, p, body.name)
    audit(cfg, p.user_id, "sheet_template.create", t["id"], None, client_ip(request), p.role)
    return store.detail(t)


@router.get("/{tid}")
def get_template(tid: str, request: Request, p: Principal = Depends(require_user)):
    return store.detail(store.get_visible(cfg_of(request), p, tid))


@router.patch("/{tid}")
def rename_template(tid: str, body: RenameIn, request: Request, p: Principal = Depends(require_user)):
    return store.detail(store.rename(cfg_of(request), p, tid, body.name))


@router.post("/{tid}/share")
def share_template(tid: str, body: ShareIn, request: Request, p: Principal = Depends(require_root)):
    cfg = cfg_of(request)
    t = store.set_shared(cfg, tid, body.shared)
    audit(cfg, p.user_id, "sheet_template.share", tid, {"shared": body.shared}, client_ip(request), p.role)
    return store.detail(t)


@router.post("/{tid}/pages")
def add_pages(tid: str, request: Request, files: list[UploadFile] = File(...),
              p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    if len(files) > MAX_FILES_PER_REQUEST:
        raise AppError("too_many_files", f"at most {MAX_FILES_PER_REQUEST} files per request", 422)
    try:
        t, rejected = store.add_pages(cfg, p, tid, [(f.filename or "", f.file) for f in files])
    finally:
        for f in files:
            f.file.close()
    added = len(files) - len(rejected)
    audit(cfg, p.user_id, "sheet_template.add_pages", tid, {"added": added}, client_ip(request), p.role)
    return {"template": store.detail(t), "rejected": rejected}


@router.patch("/{tid}/pages/{pid}")
def update_page(tid: str, pid: str, body: PageIn, request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    if body.disabled is None:
        raise AppError("validation_error", "nothing to change", 422)
    t = store.set_disabled(cfg, p, tid, pid, body.disabled)
    audit(cfg, p.user_id, "sheet_template.page_disabled" if body.disabled else "sheet_template.page_enabled",
          tid, {"page_id": pid}, client_ip(request), p.role)
    return store.detail(t)


@router.post("/{tid}/pages/{pid}/retry")
def retry_page(tid: str, pid: str, request: Request, p: Principal = Depends(require_user)):
    return store.detail(store.retry_page(cfg_of(request), p, tid, pid))


@router.get("/{tid}/pages/{pid}/image")
def page_image(tid: str, pid: str, request: Request, p: Principal = Depends(require_user)):
    path, mime = store.image_path(cfg_of(request), p, tid, pid)
    return FileResponse(path, media_type=mime, headers={"Cache-Control": "private, max-age=3600"})
