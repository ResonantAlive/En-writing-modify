"""Upload staging store: data/staging/<upload_id>/{manifest.json, files/, thumbs/}. Strictly owner-only."""
from __future__ import annotations

import hashlib
import logging
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterable

from ..config import AppConfig
from ..core import atomic_io, clock
from ..core.ids import new_id
from ..core.locks import file_lock
from ..core.paths import safe_join
from ..errors import NotFound
from . import grouping, images

log = logging.getLogger("wm.staging")

_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")
MAX_NAME_CHARS = 512
CHUNK = 1 << 20


def _check_id(value: str) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise NotFound()
    return value


def upload_dir(cfg: AppConfig, upload_id: str) -> Path:
    return safe_join(cfg.staging_dir, _check_id(upload_id))


def _lock(cfg: AppConfig, upload_id: str):
    return file_lock(cfg.lock_path(f"staging-{upload_id}"))


def _read_manifest(cfg: AppConfig, upload_id: str) -> dict:
    m = atomic_io.read_json(upload_dir(cfg, upload_id) / "manifest.json", None)
    if not m:
        raise NotFound()
    return m


def get_manifest(cfg: AppConfig, owner_id: str, upload_id: str) -> dict:
    """NotFound if the upload is missing OR belongs to someone else (even root)."""
    m = _read_manifest(cfg, upload_id)
    if m.get("owner_id") != owner_id:
        raise NotFound()
    return m


def _find(m: dict, file_id: str) -> dict:
    _check_id(file_id)
    for f in m["files"]:
        if f["file_id"] == file_id:
            return f
    raise NotFound()


def file_abspath(cfg: AppConfig, upload_id: str, file_id: str) -> Path:
    m = _read_manifest(cfg, upload_id)
    f = _find(m, file_id)
    return safe_join(upload_dir(cfg, upload_id), "files", f"{f['file_id']}.{f['ext']}")


def thumb_abspath(cfg: AppConfig, owner_id: str, upload_id: str, file_id: str) -> Path:
    f = _find(get_manifest(cfg, owner_id, upload_id), file_id)
    p = safe_join(upload_dir(cfg, upload_id), f["thumb"])
    if not p.is_file():
        raise NotFound()
    return p


def release(cfg: AppConfig, upload_id: str) -> None:
    """Delete the whole staging dir (after the batch has been created)."""
    shutil.rmtree(upload_dir(cfg, upload_id), ignore_errors=True)


def public_file(upload_id: str, f: dict) -> dict:
    return {
        "file_id": f["file_id"], "orig_name": f["orig_name"], "mime": f["mime"], "size": f["size"],
        "width": f["width"], "height": f["height"],
        "thumb_url": f"/api/v1/uploads/{upload_id}/files/{f['file_id']}/thumb",
        "name_seq": f.get("name_seq"), "exif_time": f.get("exif_time"),
        "client_last_modified": f.get("client_last_modified"),
    }


def public_manifest(m: dict) -> dict:
    return {"upload_id": m["upload_id"], "created_at": m["created_at"],
            "files": [public_file(m["upload_id"], f) for f in m["files"]]}


def create_upload(cfg: AppConfig, owner_id: str) -> dict:
    upload_id = new_id("up")
    d = upload_dir(cfg, upload_id)
    (d / "files").mkdir(parents=True, exist_ok=True)
    (d / "thumbs").mkdir(parents=True, exist_ok=True)
    m = {"schema_version": 1, "upload_id": upload_id, "owner_id": owner_id,
         "created_at": clock.iso(), "files": []}
    atomic_io.write_json_atomic(d / "manifest.json", m)
    return m


def _rejected(name: str, code: str, message: str) -> dict:
    return {"orig_name": name, "code": code, "message": message}


def _discard(*paths: Path) -> None:
    for p in paths:
        try:
            p.unlink()
        except OSError:
            pass


def _ingest_one(cfg: AppConfig, d: Path, name: str, stream: BinaryIO, lm: int | None):
    """Stream one file to disk and validate it. Returns (entry, None) or (None, rejection)."""
    cap = int(cfg["max_file_mb"]) * 1024 * 1024
    head = stream.read(CHUNK)
    sniffed = images.sniff_type(head[:64])
    if sniffed is None:
        return None, _rejected(name, "unsupported_type",
                               "Unsupported file type (allowed: JPEG, PNG, WebP, HEIC, HEIF)")
    mime, ext = sniffed
    fid = new_id("f")
    dst = d / "files" / f"{fid}.{ext}"
    thumb = d / "thumbs" / f"{fid}.jpg"
    h = hashlib.sha256()
    size = 0
    with open(dst, "wb") as out:
        chunk = head
        while chunk:
            size += len(chunk)
            if size > cap:
                out.close()
                _discard(dst)
                return None, _rejected(name, "file_too_large",
                                       f"File exceeds the {cfg['max_file_mb']} MB limit")
            h.update(chunk)
            out.write(chunk)
            chunk = stream.read(CHUNK)
    try:
        w, hgt = images.make_thumbnail(dst, thumb)
        exif_time = images.exif_datetime(dst)
    except Exception:
        _discard(dst, thumb)
        return None, _rejected(name, "corrupt_image", "The image could not be decoded")
    return {
        "file_id": fid, "orig_name": name, "ext": ext, "mime": mime, "size": size,
        "sha256": h.hexdigest(), "width": w, "height": hgt, "thumb": f"thumbs/{fid}.jpg",
        "name_seq": grouping.name_seq(name), "exif_time": exif_time, "client_last_modified": lm,
    }, None


def add_files(cfg: AppConfig, owner_id: str, upload_id: str | None,
              items: Iterable[tuple[str, BinaryIO, int | None]]) -> tuple[dict, list[dict]]:
    """Create or append to an upload. Returns (manifest, rejected). Invalid files are rejected individually."""
    m = get_manifest(cfg, owner_id, upload_id) if upload_id else create_upload(cfg, owner_id)
    upload_id = m["upload_id"]
    d = upload_dir(cfg, upload_id)
    max_files = int(cfg["max_files_per_batch"])
    existing = len(m["files"])
    ready: list[dict] = []
    rejected: list[dict] = []
    for name, stream, lm in items:
        name = (name or "")[:MAX_NAME_CHARS]
        if existing + len(ready) >= max_files:
            rejected.append(_rejected(name, "too_many_files", f"At most {max_files} files per batch"))
            continue
        entry, rej = _ingest_one(cfg, d, name, stream, lm)
        if rej:
            rejected.append(rej)
        else:
            ready.append(entry)
    # commit under the lock (re-read: another request may have appended meanwhile)
    with _lock(cfg, upload_id):
        cur = _read_manifest(cfg, upload_id)
        for e in ready:
            if len(cur["files"]) >= max_files:
                _discard(d / "files" / f"{e['file_id']}.{e['ext']}", d / e["thumb"])
                rejected.append(_rejected(e["orig_name"], "too_many_files",
                                          f"At most {max_files} files per batch"))
            else:
                cur["files"].append(e)
        atomic_io.write_json_atomic(d / "manifest.json", cur)
    return cur, rejected


def delete_file(cfg: AppConfig, owner_id: str, upload_id: str, file_id: str) -> None:
    get_manifest(cfg, owner_id, upload_id)
    d = upload_dir(cfg, upload_id)
    with _lock(cfg, upload_id):
        cur = _read_manifest(cfg, upload_id)
        f = _find(cur, file_id)
        cur["files"] = [x for x in cur["files"] if x["file_id"] != file_id]
        atomic_io.write_json_atomic(d / "manifest.json", cur)
    _discard(d / "files" / f"{f['file_id']}.{f['ext']}", d / f["thumb"])


def delete_upload(cfg: AppConfig, owner_id: str, upload_id: str) -> None:
    get_manifest(cfg, owner_id, upload_id)
    release(cfg, upload_id)


def cleanup_expired(cfg: AppConfig) -> int:
    """Delete staging dirs older than settings.staging_ttl_hours (manifest created_at OR dir mtime)."""
    ttl = float(cfg["staging_ttl_hours"]) * 3600
    removed = 0
    if not cfg.staging_dir.is_dir():
        return 0
    for d in list(cfg.staging_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            expired = (time.time() - d.stat().st_mtime) > ttl
            if not expired:
                m = atomic_io.read_json(d / "manifest.json", None)
                ca = m.get("created_at") if isinstance(m, dict) else None
                if ca:
                    try:
                        expired = (clock.now() - datetime.fromisoformat(ca)).total_seconds() > ttl
                    except (ValueError, TypeError):
                        pass
            if expired:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except Exception:
            log.exception("staging cleanup failed for one directory")
    return removed
