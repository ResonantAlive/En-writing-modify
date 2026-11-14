"""Extension points replaced by M3/M4. Always looked up at call time (`hooks.local_env_ok(...)`)
so they can be monkeypatched."""
from __future__ import annotations


def local_env_ok(cfg) -> tuple[bool, str | None]:
    return False, "local OCR environment check not implemented yet"


def find_fewshot(cfg, owner_id: str, scope: str, draft: dict) -> list[str]:
    return []


def needs_third_party_ack(cfg, user, provider_id: str) -> bool:
    return False


def validate_provider_and_template(cfg, user, body: dict) -> None:
    return None


def require_visible_refs(cfg, user, body: dict) -> None:
    """Precheck (report-only): a provider / template id the caller cannot use must 404 like in create."""
    return None


def build_revision(cfg, user, body: dict, fewshot_ids: list, rev: int, reason: str,
                   template_override: dict | None = None) -> dict:
    """Build one config revision. Default = placeholder (no rendered prompt); grading.revision replaces it."""
    from ..core import clock
    tpl = template_override or {"id": body.get("template_id") or "", "version": body.get("template_version")}
    return {"rev": rev, "at": clock.iso(), "provider_id": body.get("provider_id") or "", "provider_alias": None,
            "model": body.get("model"), "params": dict(body.get("params") or {}),
            "template": {"id": tpl.get("id"), "version": tpl.get("version")},
            "rendered_prompt": "", "reason": reason}


_HOOK_NAMES = ("local_env_ok", "find_fewshot", "needs_third_party_ack", "validate_provider_and_template",
               "require_visible_refs", "build_revision")
DEFAULTS = {n: globals()[n] for n in _HOOK_NAMES}


def reset_defaults() -> None:
    """Restore the default implementations (tests)."""
    g = globals()
    for n, fn in DEFAULTS.items():
        g[n] = fn
