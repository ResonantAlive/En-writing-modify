"""data/app/providers.json: LLM provider records. The plaintext api_key lives ONLY here and is never
returned by any function named public_*/list_visible (only key_last4)."""
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from ..auth.deps import Principal
from ..config import AppConfig, is_loopback
from ..core import atomic_io, clock
from ..core.ids import new_id
from ..core.locks import file_lock
from ..errors import AppError, NotFound


def _path(cfg: AppConfig):
    return cfg.app_dir / "providers.json"


def _read(cfg: AppConfig) -> dict:
    d = atomic_io.read_json(_path(cfg), None) or {}
    d.setdefault("schema_version", 1)
    d.setdefault("providers", [])
    return d


def _mutate(cfg: AppConfig, fn):
    with file_lock(cfg.lock_path("providers")):
        data = _read(cfg)
        res = fn(data)
        atomic_io.write_json_atomic(_path(cfg), data)
    return res


def host_of(base_url: str) -> str:
    try:
        return (urlsplit(base_url).hostname or "").lower()
    except ValueError:
        return ""


def is_third_party_url(base_url: str) -> bool:
    """Third party unless the host is localhost / a loopback address (127.0.0.0/8, ::1)."""
    host = host_of(base_url)
    return not (host and is_loopback(host))


def _bad(field: str, msg: str) -> AppError:
    return AppError("validation_error", msg, 422, {"field": field})


def _clean_str(v: Any, field: str, maxlen: int) -> str:
    if not isinstance(v, str) or not v.strip():
        raise _bad(field, f"{field} is required")
    v = v.strip()
    if len(v) > maxlen:
        raise _bad(field, f"{field} too long")
    return v


def _clean_url(v: Any) -> str:
    v = _clean_str(v, "base_url", 500)
    try:
        sp = urlsplit(v)
        host = sp.hostname
    except ValueError:
        raise _bad("base_url", "base_url is not a valid URL")
    if sp.scheme not in ("http", "https") or not host:
        raise _bad("base_url", "base_url must be an http(s) URL with a host")
    return v


def get_provider(cfg: AppConfig, provider_id: str) -> dict | None:
    """Internal: includes api_key, no visibility check, excludes soft-deleted."""
    for p in _read(cfg)["providers"]:
        if p["id"] == provider_id and not p.get("deleted"):
            return p
    return None


def key_last4(p: dict) -> str:
    return (p.get("api_key") or "")[-4:]


def public_view(p: dict) -> dict:
    return {"id": p["id"], "alias": p["alias"], "base_url": p["base_url"], "key_last4": key_last4(p),
            "default_model": p["default_model"], "owner_id": p["owner_id"], "shared": bool(p["shared"]),
            "is_third_party": bool(p["is_third_party"]), "created_at": p["created_at"],
            "updated_at": p["updated_at"]}


def _visible(principal: Principal, p: dict) -> bool:
    return principal.is_root or p["owner_id"] == principal.user_id or bool(p.get("shared"))


def can_use(cfg: AppConfig, principal: Principal, provider_id: str) -> bool:
    p = get_provider(cfg, provider_id)
    return bool(p and _visible(principal, p))


def list_visible(cfg: AppConfig, principal: Principal) -> list[dict]:
    return [public_view(p) for p in _read(cfg)["providers"]
            if not p.get("deleted") and _visible(principal, p)]


def create(cfg: AppConfig, principal: Principal, body: dict) -> dict:
    alias = _clean_str(body.get("alias"), "alias", 100)
    base_url = _clean_url(body.get("base_url"))
    key = _clean_str(body.get("api_key"), "api_key", 2000)
    model = _clean_str(body.get("default_model"), "default_model", 200)
    shared = body.get("shared") is True
    if shared and not principal.is_root:
        raise AppError("forbidden", "only root may share a provider", 403)
    now = clock.iso()
    rec = {"id": new_id("prov", 4), "alias": alias, "base_url": base_url, "api_key": key,
           "default_model": model, "owner_id": principal.user_id, "shared": shared,
           "is_third_party": is_third_party_url(base_url), "created_at": now, "updated_at": now,
           "deleted": False}

    def fn(data):
        data["providers"].append(rec)
    _mutate(cfg, fn)
    return rec


def _editable(principal: Principal, p: dict) -> bool:
    return principal.is_root or p["owner_id"] == principal.user_id


def update(cfg: AppConfig, principal: Principal, provider_id: str, body: dict) -> dict:
    def fn(data):
        p = next((x for x in data["providers"] if x["id"] == provider_id and not x.get("deleted")), None)
        if not p or not _visible(principal, p):
            raise NotFound()
        if not _editable(principal, p):
            raise AppError("forbidden", "shared providers can only be edited by their owner or root", 403)
        if body.get("alias") is not None:
            p["alias"] = _clean_str(body["alias"], "alias", 100)
        if body.get("base_url") is not None:
            p["base_url"] = _clean_url(body["base_url"])
            p["is_third_party"] = is_third_party_url(p["base_url"])
        if body.get("api_key") is not None:
            p["api_key"] = _clean_str(body["api_key"], "api_key", 2000)
        if body.get("default_model") is not None:
            p["default_model"] = _clean_str(body["default_model"], "default_model", 200)
        if body.get("shared") is not None:
            want = body["shared"] is True
            if want != bool(p["shared"]):
                if not principal.is_root:
                    raise AppError("forbidden", "only root may change sharing", 403)
                if p["owner_id"] != principal.user_id:
                    raise AppError("forbidden", "root can only share providers it owns", 403)
            p["shared"] = want
        p["updated_at"] = clock.iso()
        return dict(p)
    return _mutate(cfg, fn)


def soft_delete(cfg: AppConfig, principal: Principal, provider_id: str) -> None:
    def fn(data):
        p = next((x for x in data["providers"] if x["id"] == provider_id and not x.get("deleted")), None)
        if not p or not _visible(principal, p):
            raise NotFound()
        if not _editable(principal, p):
            raise AppError("forbidden", "shared providers can only be deleted by their owner or root", 403)
        p["deleted"] = True
        p["updated_at"] = clock.iso()
    _mutate(cfg, fn)
