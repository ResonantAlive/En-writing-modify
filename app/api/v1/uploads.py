"""Upload staging API (owner-only; root gets no special access)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse

from ...auth.deps import Principal, cfg_of, require_user
from ...staging import grouping, images, uploads as S

router = APIRouter(prefix="/uploads", tags=["uploads"])


def _int_or_none(v) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


@router.get("/limits")
def limits(request: Request, p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    from ...batches import hooks   # looked up at call time (M4 replaces local_env_ok)
    env_ok, env_reason = hooks.local_env_ok(cfg)
    return {"max_files_per_batch": cfg["max_files_per_batch"], "max_file_mb": cfg["max_file_mb"],
            "max_pages_per_essay": cfg["max_pages_per_essay"], "allowed_types": images.ALLOWED_MIMES,
            "default_local_engine": cfg["default_local_engine"],
            "local_env_ok": env_ok, "local_env_reason": env_reason}


@router.post("")
@router.post("/", include_in_schema=False)
def upload(request: Request, files: list[UploadFile] = File(...),
           client_last_modified: list[str] = Form(default=[]),
           upload_id: Optional[str] = Form(default=None),
           p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    items = [(f.filename or "", f.file,
              _int_or_none(client_last_modified[i]) if i < len(client_last_modified) else None)
             for i, f in enumerate(files)]
    try:
        m, rejected = S.add_files(cfg, p.user_id, upload_id or None, items)
    finally:
        for f in files:
            f.file.close()
    return {"upload_id": m["upload_id"], "files": S.public_manifest(m)["files"], "rejected": rejected}


@router.get("/{upload_id}")
def get_upload(upload_id: str, request: Request, p: Principal = Depends(require_user)):
    return S.public_manifest(S.get_manifest(cfg_of(request), p.user_id, upload_id))


@router.get("/{upload_id}/files/{file_id}/thumb")
def thumb(upload_id: str, file_id: str, request: Request, p: Principal = Depends(require_user)):
    path = S.thumb_abspath(cfg_of(request), p.user_id, upload_id, file_id)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@router.delete("/{upload_id}/files/{file_id}")
def delete_file(upload_id: str, file_id: str, request: Request, p: Principal = Depends(require_user)):
    S.delete_file(cfg_of(request), p.user_id, upload_id, file_id)
    return {"ok": True}


@router.delete("/{upload_id}")
def delete_upload(upload_id: str, request: Request, p: Principal = Depends(require_user)):
    S.delete_upload(cfg_of(request), p.user_id, upload_id)
    return {"ok": True}


@router.post("/{upload_id}/grouping/preview")
def preview(upload_id: str, request: Request, body: dict = Body(default={}),
            p: Principal = Depends(require_user)):
    cfg = cfg_of(request)
    m = S.get_manifest(cfg, p.user_id, upload_id)
    return grouping.resolve_groups(cfg, m, body)
