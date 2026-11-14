"""data/app/prompts.json: prompt templates with immutable versions. Builtin templates are code-defined
(app/prompts/builtin/*.json, read-only, visible to everyone)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from ..auth.deps import Principal
from ..config import AppConfig
from ..core import atomic_io, clock
from ..core.ids import new_id
from ..core.locks import file_lock
from ..errors import AppError, NotFound

BUILTIN_DIR = Path(__file__).resolve().parent / "builtin"      # tests may monkeypatch
FEWSHOT = "{{FEWSHOT}}"
MODES = ("text", "image")


def _path(cfg: AppConfig):
    return cfg.app_dir / "prompts.json"


def _read(cfg: AppConfig) -> dict:
    d = atomic_io.read_json(_path(cfg), None) or {}
    d.setdefault("schema_version", 1)
    d.setdefault("templates", [])
    return d


def _mutate(cfg: AppConfig, fn):
    with file_lock(cfg.lock_path("prompts")):
        data = _read(cfg)
        res = fn(data)
        atomic_io.write_json_atomic(_path(cfg), data)
    return res


def builtins() -> list[dict]:
    out = []
    d = Path(BUILTIN_DIR)
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        t = json.loads(f.read_text("utf-8"))
        t["builtin"] = True
        t.setdefault("owner_id", None)
        t["shared"] = True
        t.setdefault("deleted", False)
        t.setdefault("parent_id", None)
        t.setdefault("parent_version", None)
        out.append(t)
    return out


def get_template(cfg: AppConfig, tid: str) -> dict | None:
    """Raw template (deleted ones included, so pinned batches keep working); no visibility check."""
    for t in builtins():
        if t["id"] == tid:
            return t
    for t in _read(cfg)["templates"]:
        if t["id"] == tid:
            return t
    return None


def get_version(cfg: AppConfig, tid: str, version: int | None = None) -> dict:
    t = get_template(cfg, tid)
    if not t or not t["versions"]:
        raise NotFound("template not found")
    if version is None:
        return t["versions"][-1]
    for v in t["versions"]:
        if v["version"] == version:
            return v
    raise NotFound("template version not found")


def _visible(principal: Principal, t: dict) -> bool:
    return (principal.is_root or t.get("builtin") or t.get("owner_id") == principal.user_id
            or bool(t.get("shared")))


def can_use(cfg: AppConfig, principal: Principal, tid: str) -> bool:
    t = get_template(cfg, tid)
    return bool(t and not t.get("deleted") and _visible(principal, t))


def summary(t: dict) -> dict:
    v = t["versions"][-1]
    return {"id": t["id"], "name": t["name"], "builtin": bool(t.get("builtin")), "owner_id": t.get("owner_id"),
            "shared": bool(t.get("shared")), "latest_version": v["version"], "input_mode": v["input_mode"],
            "parent_id": t.get("parent_id"), "updated_at": v["created_at"]}


def list_visible(cfg: AppConfig, principal: Principal, input_mode: str | None = None) -> list[dict]:
    out = []
    for t in builtins() + _read(cfg)["templates"]:
        if t.get("deleted") or not t["versions"] or not _visible(principal, t):
            continue
        s = summary(t)
        if input_mode and s["input_mode"] != input_mode:
            continue
        out.append(s)
    return out


# ---- validation / building a version ------------------------------------------------
def _bad(field: str, msg: str) -> AppError:
    return AppError("validation_error", msg, 422, {"field": field})


def _str_list(v: Any, field: str) -> list[str]:
    if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
        raise _bad(field, f"{field} must be a list of non-empty strings")
    return [x.strip() for x in v]


def build_version(cfg: AppConfig, body: dict, *, base: dict | None, user_id: str, version: int) -> dict:
    """Validate the body (fields omitted fall back to `base`, the previous version) and return a new
    immutable version record."""
    def pick(k, default=None):
        if body.get(k) is not None:
            return body[k]
        if base is not None and k in base:
            return base[k]
        return default

    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise _bad("text", "text is required")
    mode = pick("input_mode")
    if mode not in MODES:
        raise AppError("invalid_input_mode", "input_mode must be text or image", 422)
    if base is not None and mode != base["input_mode"]:
        raise AppError("input_mode_immutable", "a template's input_mode cannot change between versions", 422)
    score_key = pick("score_key", "判分")
    criteria_key = pick("criteria_key", "判据")
    for k, v in (("score_key", score_key), ("criteria_key", criteria_key)):
        if not isinstance(v, str) or not v.strip():
            raise _bad(k, f"{k} must be a non-empty string")
    score_key, criteria_key = score_key.strip(), criteria_key.strip()
    aliases = _str_list(pick("score_aliases", []), "score_aliases")
    if body.get("required_keys") is not None:
        required = _str_list(body["required_keys"], "required_keys")
    elif base is not None and body.get("score_key") is None and body.get("criteria_key") is None:
        required = list(base["required_keys"])
    else:
        required = [score_key, criteria_key]
    if score_key not in required:
        raise AppError("score_key_not_required", "score_key must be listed in required_keys", 422)
    limit = cfg["prompt"]["max_chars"]
    max_chars = pick("max_chars", limit)
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 1 <= max_chars <= limit:
        raise _bad("max_chars", f"max_chars must be an integer in 1..{limit}")
    if FEWSHOT not in text:
        if body.get("auto_append_fewshot") is True:
            text = text.rstrip() + "\n\n" + FEWSHOT
        else:
            raise AppError("fewshot_placeholder_missing",
                           "the prompt must contain {{FEWSHOT}}; resend with auto_append_fewshot=true to append it",
                           422)
    if len(text) > max_chars:                     # str length == Unicode code points
        raise AppError("prompt_too_long", f"prompt is {len(text)} characters; limit is {max_chars}", 422,
                       {"length": len(text), "max_chars": max_chars})
    return {"version": version, "created_at": clock.iso(), "created_by": user_id, "text": text,
            "input_mode": mode, "score_key": score_key, "score_aliases": aliases,
            "criteria_key": criteria_key, "required_keys": required, "max_chars": max_chars}


def _clean_name(name: Any) -> str:
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 100:
        raise _bad("name", "name is required (1-100 characters)")
    return name.strip()


def create_template(cfg: AppConfig, principal: Principal, body: dict) -> dict:
    name = _clean_name(body.get("name"))
    parent_id, parent_version = body.get("parent_id"), body.get("parent_version")
    if parent_id is not None:
        if not isinstance(parent_id, str) or not can_use(cfg, principal, parent_id):
            raise NotFound("parent template not found")
        pv = get_version(cfg, parent_id, parent_version) if parent_version is not None else get_version(cfg, parent_id)
        parent_version = pv["version"]
    else:
        parent_version = None
    ver = build_version(cfg, body, base=None, user_id=principal.user_id, version=1)
    rec = {"id": new_id("tpl", 4), "name": name, "builtin": False, "owner_id": principal.user_id,
           "shared": False, "parent_id": parent_id, "parent_version": parent_version, "deleted": False,
           "created_at": ver["created_at"], "versions": [ver]}
    _mutate(cfg, lambda data: data["templates"].append(rec))
    return rec


def add_version(cfg: AppConfig, principal: Principal, tid: str, body: dict) -> tuple[dict, dict]:
    """Returns (template, new_version). builtin => 409; non-owner non-root => 404."""
    def fn(data):
        t = next((x for x in data["templates"] if x["id"] == tid), None)
        if t is None:
            if any(b["id"] == tid for b in builtins()):
                raise AppError("builtin_readonly", "builtin templates are read-only", 409,
                               {"hint": "save as a new template"})
            raise NotFound()
        if t.get("deleted") or not (principal.is_root or t["owner_id"] == principal.user_id):
            raise NotFound()
        ver = build_version(cfg, body, base=t["versions"][-1], user_id=principal.user_id,
                            version=t["versions"][-1]["version"] + 1)
        t["versions"].append(ver)
        return copy.deepcopy(t), ver
    return _mutate(cfg, fn)


def set_shared(cfg: AppConfig, tid: str, shared: bool) -> dict:
    def fn(data):
        t = next((x for x in data["templates"] if x["id"] == tid and not x.get("deleted")), None)
        if t is None:
            if any(b["id"] == tid for b in builtins()):
                raise AppError("builtin_readonly", "builtin templates are read-only", 409)
            raise NotFound()
        t["shared"] = bool(shared)
        return copy.deepcopy(t)
    return _mutate(cfg, fn)


def soft_delete(cfg: AppConfig, principal: Principal, tid: str) -> dict:
    def fn(data):
        t = next((x for x in data["templates"] if x["id"] == tid), None)
        if t is None:
            if any(b["id"] == tid for b in builtins()):
                raise AppError("builtin_readonly", "builtin templates cannot be deleted", 409)
            raise NotFound()
        if t.get("deleted") or not (principal.is_root or t["owner_id"] == principal.user_id):
            raise NotFound()
        t["deleted"] = True
        return copy.deepcopy(t)
    return _mutate(cfg, fn)
