"""Config revisions: pin provider / model / params / template version and the fully rendered system prompt."""
from __future__ import annotations

from ..core import clock
from ..errors import AppError, NotFound
from . import fewshot

_RESERVED_PARAMS = {"model", "messages", "stream", "stream_options"}


def _principal_id(p) -> str | None:
    return p if isinstance(p, str) or p is None else getattr(p, "user_id", None)


def _batch_fields(cfg, body: dict) -> dict:
    """topic/rubric/genre/full_score used for rendering. `body` normally carries them (creation); a config
    change may pass only {"task_code": ...}, then they come from batch.json."""
    out = {k: body.get(k) for k in ("topic", "rubric", "genre", "full_score")}
    if any(v is None for v in out.values()) and body.get("task_code"):
        from ..batches import store
        b = store.read_batch(cfg, body["task_code"]) or {}
        for k in out:
            if out[k] is None:
                out[k] = b.get(k)
    return out


def make_revision(cfg, principal_or_user_id, body: dict, fewshot_ids: list[str], rev: int, reason: str,
                  template_override: dict | None = None) -> dict:
    from ..prompts import render, store as pstore
    from ..providers import store as provstore

    provider_id = body.get("provider_id")
    provider = provstore.get_provider(cfg, provider_id) if provider_id else None
    if provider is None:
        raise NotFound("provider not found")
    tpl = template_override or {"id": body.get("template_id"), "version": body.get("template_version")}
    if not tpl.get("id"):
        raise AppError("template_not_found", "template_id is required", 404)
    version = pstore.get_version(cfg, tpl["id"], tpl.get("version"))      # NotFound when missing
    llm = cfg["llm"]
    params = {"temperature": llm["temperature"], "max_tokens": llm["max_tokens"]}
    extra = body.get("params")
    if isinstance(extra, dict):
        params.update({k: v for k, v in extra.items() if k not in _RESERVED_PARAMS})
    fs_text = fewshot.build_fewshot_text(cfg, fewshot_ids or [], version.get("score_key"),
                                         version.get("criteria_key"))
    rendered = render.render_system_prompt(version, _batch_fields(cfg, body), fs_text)
    return {
        "rev": rev, "at": clock.iso(),
        "provider_id": provider["id"], "provider_alias": provider.get("alias"),
        "provider_key_last4": (provider.get("api_key") or "")[-4:],
        "model": body.get("model") or provider.get("default_model"),
        "params": params,
        "template": {"id": tpl["id"], "version": version["version"]},
        "rendered_prompt": rendered, "reason": reason,
    }
