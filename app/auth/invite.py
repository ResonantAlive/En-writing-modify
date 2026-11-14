"""Invite codes: generated once at first start, stored in data/app/invite_codes.json.
Never logged, never returned by any API."""
from __future__ import annotations

import base64
import hmac
import secrets

from ..config import AppConfig
from ..core import atomic_io, clock
from ..core.locks import file_lock


def _new_code() -> str:
    raw = base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")  # 160 bits
    return "-".join(raw[i:i + 4] for i in range(0, len(raw), 4))


def _path(cfg: AppConfig):
    return cfg.app_dir / "invite_codes.json"


def ensure_invite_codes(cfg: AppConfig) -> bool:
    """Create the file if missing. Existing file is never overwritten. Returns True if created."""
    with file_lock(cfg.lock_path("invite")):
        if _path(cfg).exists():
            return False
        atomic_io.write_json_atomic(_path(cfg), {
            "schema_version": 1, "root_invite_code": _new_code(),
            "user_invite_code": _new_code(), "created_at": clock.iso()})
        return True


def _norm(code: str) -> str:
    return "".join((code or "").split()).upper()


def role_for_code(cfg: AppConfig, code: str) -> str | None:
    data = atomic_io.read_json(_path(cfg), None)
    if not data:
        return None
    given = _norm(code).encode()
    root_ok = hmac.compare_digest(given, _norm(data["root_invite_code"]).encode())
    user_ok = hmac.compare_digest(given, _norm(data["user_invite_code"]).encode())
    return "root" if root_ok else "user" if user_ok else None


def reset_codes(cfg: AppConfig, which: str) -> None:
    if which not in ("root", "user", "both"):
        raise ValueError(which)
    with file_lock(cfg.lock_path("invite")):
        data = atomic_io.read_json(_path(cfg), None) or {"schema_version": 1}
        if which in ("root", "both"):
            data["root_invite_code"] = _new_code()
        if which in ("user", "both"):
            data["user_invite_code"] = _new_code()
        data["created_at"] = clock.iso()
        atomic_io.write_json_atomic(_path(cfg), data)
