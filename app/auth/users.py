"""users.json store: accounts, sessions, API tokens. All mutations are lock + atomic write."""
from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timedelta
from typing import Any, Callable

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from ..config import AppConfig
from ..core import atomic_io, clock
from ..core.ids import new_id
from ..core.locks import file_lock
from ..errors import AppError, NotFound

ROLE_RANK = {"user": 1, "root": 2}
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{3,32}$")
_hasher = PasswordHasher()          # argon2id by default
_DUMMY_HASH = _hasher.hash("dummy-password-for-timing")


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def min_role(a: str, b: str) -> str:
    return a if ROLE_RANK[a] <= ROLE_RANK[b] else b


def _path(cfg: AppConfig):
    return cfg.app_dir / "users.json"


def read_users(cfg: AppConfig) -> dict:
    d = atomic_io.read_json(_path(cfg), None) or {}
    d.setdefault("schema_version", 1)
    d.setdefault("users", [])
    return d


def mutate(cfg: AppConfig, fn: Callable[[dict], Any]) -> Any:
    with file_lock(cfg.lock_path("users")):
        data = read_users(cfg)
        res = fn(data)           # fn may raise to abort without writing
        atomic_io.write_json_atomic(_path(cfg), data)
    return res


def _find(data: dict, user_id: str) -> dict | None:
    return next((u for u in data["users"] if u["id"] == user_id), None)


def public_user(u: dict) -> dict:
    return {"id": u["id"], "username": u["username"], "role": u["role"],
            "disabled": u["disabled"], "created_at": u["created_at"]}


def _parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


def _expired(ts: str | None) -> bool:
    d = _parse(ts)
    return bool(d and d <= clock.now())


def _due(cfg: AppConfig, last: str | None) -> bool:
    d = _parse(last)
    return d is None or (clock.now() - d).total_seconds() >= cfg["last_used_write_interval_s"]


# ---- accounts -------------------------------------------------------------------
def validate_username(name: str) -> None:
    if not _USERNAME_RE.match(name or ""):
        raise AppError("invalid_username", "username: 3-32 chars of letters, digits, _ or -", 422)


def validate_password(pw: str) -> None:
    if not 8 <= len(pw or "") <= 256:
        raise AppError("invalid_password", "password must be 8-256 characters", 422)


def register(cfg: AppConfig, role: str, username: str, password: str) -> dict:
    validate_username(username)
    validate_password(password)
    pw_hash = _hasher.hash(password)          # slow; outside the lock
    low = username.lower()

    def fn(data: dict) -> dict:
        if any(u["username_lower"] == low for u in data["users"]):
            raise AppError("username_taken", "username already taken", 409)
        u = {"id": new_id("u"), "username": username, "username_lower": low, "role": role,
             "password_hash": pw_hash, "disabled": False, "created_at": clock.iso(),
             "tp_ack": [], "sessions": [], "tokens": []}
        data["users"].append(u)
        return public_user(u)
    return mutate(cfg, fn)


def authenticate(cfg: AppConfig, username: str, password: str) -> dict | None:
    low = (username or "").lower()
    u = next((u for u in read_users(cfg)["users"] if u["username_lower"] == low), None)
    try:
        _hasher.verify(u["password_hash"] if u else _DUMMY_HASH, password or "")
    except (VerificationError, InvalidHashError):
        return None
    return u if u and not u["disabled"] else None


def _enabled_roots(data: dict, excluding: str | None = None) -> int:
    return sum(1 for u in data["users"] if u["role"] == "root" and not u["disabled"] and u["id"] != excluding)


def set_disabled(cfg: AppConfig, user_id: str, disabled: bool) -> dict:
    def fn(data: dict) -> dict:
        u = _find(data, user_id)
        if not u:
            raise NotFound()
        if disabled and u["role"] == "root" and not u["disabled"] and _enabled_roots(data, user_id) == 0:
            raise AppError("last_root", "cannot disable the last enabled root account", 409)
        u["disabled"] = disabled
        if disabled:
            u["sessions"] = []                  # sessions die immediately; tokens fail the disabled check
        return public_user(u)
    return mutate(cfg, fn)


def set_role(cfg: AppConfig, user_id: str, role: str) -> dict:
    if role not in ROLE_RANK:
        raise AppError("invalid_role", "role must be root or user", 422)

    def fn(data: dict) -> dict:
        u = _find(data, user_id)
        if not u:
            raise NotFound()
        if role == "user" and u["role"] == "root" and not u["disabled"] and _enabled_roots(data, user_id) == 0:
            raise AppError("last_root", "cannot demote the last enabled root account", 409)
        u["role"] = role
        return public_user(u)
    return mutate(cfg, fn)


# ---- sessions -------------------------------------------------------------------
def create_session(cfg: AppConfig, user_id: str, ip: str | None) -> tuple[str, str]:
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(24)
    now = clock.now()

    def fn(data: dict) -> None:
        u = _find(data, user_id)
        u["sessions"] = [s for s in u["sessions"] if not _expired(s["expires_at"])]
        u["sessions"].append({"id_hash": sha256(token), "csrf": csrf, "created_at": clock.iso(now),
                              "expires_at": clock.iso(now + timedelta(days=cfg["session_days"])),
                              "last_active": clock.iso(now), "ip": ip})
    mutate(cfg, fn)
    return token, csrf


def resolve_session(cfg: AppConfig, raw: str) -> tuple[dict, dict] | None:
    h = sha256(raw)
    for u in read_users(cfg)["users"]:
        for s in u["sessions"]:
            if s["id_hash"] == h:
                if _expired(s["expires_at"]) or u["disabled"]:
                    return None
                if _due(cfg, s["last_active"]):
                    def fn(data: dict) -> None:
                        uu = _find(data, u["id"])
                        for ss in uu["sessions"]:
                            if ss["id_hash"] == h:
                                ss["last_active"] = clock.iso()
                    mutate(cfg, fn)
                return u, s
    return None


def delete_session(cfg: AppConfig, raw: str) -> None:
    h = sha256(raw)

    def fn(data: dict) -> None:
        for u in data["users"]:
            u["sessions"] = [s for s in u["sessions"] if s["id_hash"] != h]
    mutate(cfg, fn)


# ---- API tokens -----------------------------------------------------------------
def create_token(cfg: AppConfig, user_id: str, name: str, role: str, expires_days: float | None) -> tuple[str, dict]:
    if role not in ROLE_RANK:
        raise AppError("invalid_role", "role must be root or user", 422)
    name = (name or "").strip()
    if not 1 <= len(name) <= 100:
        raise AppError("invalid_name", "token name must be 1-100 characters", 422)
    raw = "wm_" + secrets.token_urlsafe(32)
    now = clock.now()
    rec = {"id": new_id("tok"), "name": name, "hash": sha256(raw), "role": role, "last4": raw[-4:],
           "created_at": clock.iso(now),
           "expires_at": clock.iso(now + timedelta(days=expires_days)) if expires_days else None,
           "revoked": False, "last_used": None}

    def fn(data: dict) -> None:
        _find(data, user_id)["tokens"].append(rec)
    mutate(cfg, fn)
    return raw, public_token(rec)


def public_token(t: dict, owner: dict | None = None) -> dict:
    out = {k: t[k] for k in ("id", "name", "role", "last4", "created_at", "expires_at", "revoked", "last_used")}
    if owner:
        out["owner_id"], out["owner_username"] = owner["id"], owner["username"]
    return out


def resolve_token(cfg: AppConfig, raw: str) -> tuple[dict, dict] | None:
    h = sha256(raw)
    for u in read_users(cfg)["users"]:
        for t in u["tokens"]:
            if t["hash"] == h:
                if t["revoked"] or _expired(t["expires_at"]) or u["disabled"]:
                    return None
                if _due(cfg, t["last_used"]):
                    def fn(data: dict) -> None:
                        for tt in _find(data, u["id"])["tokens"]:
                            if tt["hash"] == h:
                                tt["last_used"] = clock.iso()
                    mutate(cfg, fn)
                return u, t
    return None


def revoke_token(cfg: AppConfig, token_id: str, only_owner: str | None) -> dict:
    def fn(data: dict) -> dict:
        for u in data["users"]:
            if only_owner and u["id"] != only_owner:
                continue
            for t in u["tokens"]:
                if t["id"] == token_id:
                    t["revoked"] = True
                    return public_token(t, u)
        raise NotFound()
    return mutate(cfg, fn)


def list_tokens(cfg: AppConfig, owner_id: str | None) -> list[dict]:
    out = []
    for u in read_users(cfg)["users"]:
        if owner_id and u["id"] != owner_id:
            continue
        out += [public_token(t, u) for t in u["tokens"]]
    return out


def list_users(cfg: AppConfig) -> list[dict]:
    return [public_user(u) for u in read_users(cfg)["users"]]
