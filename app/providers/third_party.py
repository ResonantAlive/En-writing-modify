"""Third-party provider checks (essay images leave the machine) + hook registration."""
from __future__ import annotations

from ..auth import users as U
from ..core import clock
from ..errors import AppError, NotFound
from . import store

_MODE_FOR_ROUTE = {"vlm": "image", "local_ocr": "text"}


def _norm(url: str) -> str:
    return (url or "").strip().rstrip("/")


def host(provider: dict) -> str:
    return store.host_of(provider["base_url"])


def has_ack(cfg, user_id: str, provider: dict) -> bool:
    u = next((x for x in U.read_users(cfg)["users"] if x["id"] == user_id), None)
    if not u:
        return False
    return any(a.get("provider_id") == provider["id"] and _norm(a.get("base_url")) == _norm(provider["base_url"])
               for a in u.get("tp_ack", []))


def needs_ack(cfg, user_id: str, provider: dict) -> bool:
    return bool(provider.get("is_third_party")) and not has_ack(cfg, user_id, provider)


def record_ack(cfg, user_id: str, provider: dict) -> None:
    """Idempotent; a stale entry for the same provider (old base_url) is replaced."""
    if not provider.get("is_third_party") or has_ack(cfg, user_id, provider):
        return

    def fn(data):
        u = next((x for x in data["users"] if x["id"] == user_id), None)
        if not u:
            raise NotFound()
        acks = [a for a in u.setdefault("tp_ack", []) if a.get("provider_id") != provider["id"]]
        acks.append({"provider_id": provider["id"], "base_url": provider["base_url"], "at": clock.iso()})
        u["tp_ack"] = acks
    U.mutate(cfg, fn)


def _needs_third_party_ack_hook(cfg, user, provider_id: str) -> bool:
    p = store.get_provider(cfg, provider_id) if provider_id else None
    if not p or not store.can_use(cfg, user, provider_id):     # someone else's provider: same as a missing one
        return False
    return needs_ack(cfg, getattr(user, "user_id", user), p)


def _bad(field: str, msg: str) -> AppError:
    return AppError("validation_error", msg, 422, {"field": field})


def _validate_provider_and_template_hook(cfg, user, body: dict) -> None:
    from ..prompts import store as pstore
    pid = body.get("provider_id")
    if not isinstance(pid, str) or not pid:
        raise _bad("provider_id", "provider_id is required")
    prov = store.get_provider(cfg, pid)
    if not prov or not store.can_use(cfg, user, pid):
        raise NotFound("provider_not_found")
    tid = body.get("template_id")
    if not isinstance(tid, str) or not tid:
        raise _bad("template_id", "template_id is required")
    if not pstore.can_use(cfg, user, tid):
        raise NotFound("template_not_found")
    tv = body.get("template_version")
    if tv is not None and (isinstance(tv, bool) or not isinstance(tv, int)):
        raise _bad("template_version", "template_version must be an integer")
    try:
        ver = pstore.get_version(cfg, tid, tv)
    except NotFound:
        raise _bad("template_version", "template version does not exist")
    want = _MODE_FOR_ROUTE.get(body.get("route"))
    if want and ver["input_mode"] != want:
        raise AppError("template_mode_mismatch",
                       f"route {body.get('route')} needs a template with input_mode {want}", 422,
                       {"route": body.get("route"), "template_input_mode": ver["input_mode"]})
    m = body.get("model")
    if not isinstance(m, str) or not m.strip():
        body["model"] = prov["default_model"]     # in place: callers should read body["model"] afterwards


def _require_visible_refs_hook(cfg, user, body: dict) -> None:
    """Precheck reports problems instead of rejecting, but a provider / template the caller cannot use is a 404
    exactly as in create: someone else's id must look like a missing one and none of its properties (e.g.
    whether it is a third-party host) may leak through the report. Empty ids are fine (form not filled in yet)."""
    from ..prompts import store as pstore
    pid, tid = body.get("provider_id"), body.get("template_id")
    if isinstance(pid, str) and pid and not store.can_use(cfg, user, pid):
        raise NotFound("provider_not_found")
    if isinstance(tid, str) and tid and not pstore.can_use(cfg, user, tid):
        raise NotFound("template_not_found")


def register_hooks() -> None:
    from ..batches import hooks
    hooks.needs_third_party_ack = _needs_third_party_ack_hook
    hooks.validate_provider_and_template = _validate_provider_and_template_hook
    hooks.require_visible_refs = _require_visible_refs_hook
