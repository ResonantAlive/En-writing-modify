"""Config revisions: change provider / model / params / prompt template of a paused or finished batch.

A new revision (reason "user_change") is appended to batch.json `config_revisions`; the executor always
dispatches queued subtasks with the LATEST revision, so only unfinished and re-run (failed / api_error /
regraded) subtasks use it. `done` subtasks keep theirs (meta.config_rev records what each really used).
The few-shot set is not recomputed: the same fewshot_ids are re-rendered into the new prompt.
"""
from __future__ import annotations

from ..errors import AppError
from . import hooks, store

ALLOWED_STATUSES = store.PAUSED + ("completed",)


def _bad(field: str, msg: str) -> AppError:
    return AppError("validation_error", msg, 422, {"field": field})


def _opt_str(body: dict, field: str) -> str | None:
    v = body.get(field)
    if v is None or v == "":
        return None
    if not isinstance(v, str):
        raise _bad(field, f"{field} must be a string")
    return v.strip() or None


def _save_prompt_text(cfg, user, tid: str, tver: int | None, text: str, body: dict) -> tuple[str, int]:
    """Own template -> new version of the same id; builtin / someone else's -> NEW template with parent
    recorded. Returns (template_id, version)."""
    try:
        from ..prompts import store as pstore
    except ImportError:
        raise AppError("prompts_unavailable", "prompt templates are not available", 503)
    t = pstore.get_template(cfg, tid)
    if t is None:
        from ..errors import NotFound
        raise NotFound("template_not_found")
    base = pstore.get_version(cfg, tid, tver)
    fields = {"text": text, "input_mode": base["input_mode"], "score_key": base.get("score_key"),
              "score_aliases": base.get("score_aliases") or [], "criteria_key": base.get("criteria_key"),
              "required_keys": base.get("required_keys"), "max_chars": base.get("max_chars"),
              "auto_append_fewshot": body.get("auto_append_fewshot") is True}
    if not t.get("builtin") and t.get("owner_id") == user.user_id and not t.get("deleted"):
        tpl, ver = pstore.add_version(cfg, user, tid, fields)
        return tpl["id"], ver["version"]
    name = _opt_str(body, "new_template_name") or f"{t['name']} (edited)"
    rec = pstore.create_template(cfg, user, {**fields, "name": name, "parent_id": tid,
                                             "parent_version": base["version"]})
    return rec["id"], rec["versions"][-1]["version"]


def change_config(cfg, user, task_code: str, body: dict) -> dict:
    b = store.require_batch(cfg, task_code)
    if b["status"] not in ALLOWED_STATUSES:
        raise AppError("invalid_transition", f"cannot change the config of a batch that is {b['status']}", 409,
                       {"from": b["status"], "action": "config"})
    cur = b["config_revisions"][-1]
    provider_id = _opt_str(body, "provider_id") or cur.get("provider_id") or ""
    provider_changed = provider_id != (cur.get("provider_id") or "")
    model = _opt_str(body, "model") or (None if provider_changed else cur.get("model"))
    cur_tpl = cur.get("template") or {}
    tid = _opt_str(body, "template_id")
    tv = body.get("template_version")
    if tv is not None and (isinstance(tv, bool) or not isinstance(tv, int)):
        raise _bad("template_version", "template_version must be an integer")
    if tid is None or tid == cur_tpl.get("id"):
        tid = cur_tpl.get("id") or ""
        if tv is None:
            tv = cur_tpl.get("version")
    params = dict(cur.get("params") or {})
    if body.get("params") is not None:
        if not isinstance(body["params"], dict):
            raise _bad("params", "params must be an object")
        params.update(body["params"])
    text = body.get("prompt_text")
    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise _bad("prompt_text", "prompt_text must be a non-empty string")

    # provider / template usability + mode vs route (+ model default of a new provider)
    vbody = {"provider_id": provider_id, "model": model, "template_id": tid, "template_version": tv,
             "route": b.get("route")}
    hooks.validate_provider_and_template(cfg, user, vbody)
    model = vbody.get("model")
    if hooks.needs_third_party_ack(cfg, user, provider_id):
        if body.get("third_party_ack") is not True:
            raise AppError("third_party_ack_required",
                           "acknowledge that essay images are sent to a third-party provider", 409)
    if body.get("third_party_ack") is True:
        try:
            from ..providers import store as pstore
            from ..providers import third_party
            prov = pstore.get_provider(cfg, provider_id)
            if prov is not None:
                third_party.record_ack(cfg, user.user_id, prov)
        except (ImportError, AttributeError):
            pass

    if text is not None:
        tid, tv = _save_prompt_text(cfg, user, tid, tv, text, body)

    rbody = {"provider_id": provider_id, "model": model, "params": params, "template_id": tid,
             "template_version": tv, "route": b.get("route")}
    for _ in range(5):
        b = store.require_batch(cfg, task_code)
        rev_no = len(b["config_revisions"]) + 1
        rev = hooks.build_revision(cfg, user, rbody, list(b.get("fewshot_ids") or []), rev_no, "user_change")

        def fn(bb: dict):
            if bb["status"] not in ALLOWED_STATUSES:
                raise AppError("invalid_transition", f"cannot change the config of a batch that is "
                               f"{bb['status']}", 409, {"from": bb["status"], "action": "config"})
            if len(bb["config_revisions"]) + 1 != rev_no:
                return None                                 # someone else added a revision: rebuild
            bb["config_revisions"].append(rev)
            store.append_timeline(bb, "config_changed", user.user_id, {
                "rev": rev_no, "provider_id": rev.get("provider_id"), "model": rev.get("model"),
                "template": rev.get("template")})
            return rev
        res = store.update_batch(cfg, task_code, fn)
        if res is not None:
            return res
    raise AppError("conflict", "the config was changed concurrently; retry", 409)
