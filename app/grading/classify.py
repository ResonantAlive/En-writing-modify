"""Call classification (transport / HTTP level). Content problems (class A) come from parsing.

  ok  : HTTP 2xx (content is judged by parse_response; class "A" = 2xx with unusable content, retried)
  B   : api_error, never retried: timeout / connect / network errors, HTTP 401 402 403 404 408 429, any 5xx,
        any 3xx, any other non-2xx that is not 4xx (1xx, odd codes)
  C   : request_rejected, never retried: every OTHER 4xx (400 405 409 413 415 418 422 ...)
"""
from __future__ import annotations

B_STATUSES = {401, 402, 403, 404, 408, 429}


def classify_call(raw) -> str:
    if getattr(raw, "error_kind", None):
        return "B"
    st = getattr(raw, "http_status", None)
    if st is None:
        return "B"
    if 200 <= st < 300:
        return "ok"
    if st in B_STATUSES:
        return "B"
    if 400 <= st < 500:
        return "C"
    return "B"


def api_outcome_for_breaker(cls: str) -> str:
    """ok/A (any 2xx, even with unusable content) -> "ok" (resets the streak); B -> "api_error";
    C -> "other" (neither counts nor clears)."""
    if cls in ("ok", "A"):
        return "ok"
    if cls == "B":
        return "api_error"
    return "other"


def error_code_for(cls: str) -> str | None:
    return {"B": "api_error", "C": "request_rejected"}.get(cls)


def error_detail_for(raw) -> str | None:
    kind = getattr(raw, "error_kind", None)
    if kind:
        return f"{kind}"
    st = getattr(raw, "http_status", None)
    if st is None:
        return "no response"
    if 200 <= st < 300:
        return None
    return f"http {st}"
