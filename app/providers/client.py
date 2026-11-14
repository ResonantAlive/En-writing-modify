"""OpenAI-compatible chat-completions client. No retries, no redirects, never logs. The API key is
only placed in the Authorization header and scrubbed from any error text."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from . import store


@dataclass
class RawCall:
    http_status: int | None
    body: dict | None
    text: str
    latency_ms: int
    error_kind: str | None = None        # "timeout" | "connect" | "network" | None
    error_detail: str | None = None


def _scrub(s: str, key: str) -> str:
    if key:
        s = s.replace(key, "***")
    return re.sub(r"(?i)bearer\s+\S+", "Bearer ***", s)[:500]


async def post_chat(provider: dict, model: str, messages: list, params: dict, timeout_s: float,
                    client: httpx.AsyncClient | None = None) -> RawCall:
    key = provider.get("api_key") or ""
    url = provider["base_url"].rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, **(params or {})}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    own = client is None
    if own:
        client = httpx.AsyncClient(follow_redirects=False, timeout=timeout_s,
                                   trust_env=store.is_third_party_url(provider["base_url"]))
    t0 = time.monotonic()

    def ms() -> int:
        return int((time.monotonic() - t0) * 1000)
    try:
        resp = await client.post(url, json=payload, headers=headers, timeout=timeout_s,
                                 follow_redirects=False)
    except httpx.TimeoutException as e:
        return RawCall(None, None, "", ms(), "timeout", _scrub(f"{type(e).__name__}: {e}", key))
    except httpx.ConnectError as e:
        return RawCall(None, None, "", ms(), "connect", _scrub(f"{type(e).__name__}: {e}", key))
    except Exception as e:          # CancelledError is a BaseException and propagates
        return RawCall(None, None, "", ms(), "network", _scrub(f"{type(e).__name__}: {e}", key))
    finally:
        if own:
            await client.aclose()
    text = resp.text
    body = None
    try:
        j = resp.json()
        body = j if isinstance(j, dict) else None
    except Exception:
        body = None
    return RawCall(resp.status_code, body, text, ms())


_THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.S)


def extract_message(body: Any) -> tuple[str, str | None, str | None]:
    """-> (content, reasoning, finish_reason)."""
    try:
        choice = body["choices"][0]
        msg = choice.get("message") or {}
    except Exception:
        return "", None, None
    finish = choice.get("finish_reason")
    c = msg.get("content")
    if isinstance(c, list):
        c = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c
                    if not isinstance(x, dict) or x.get("type", "text") == "text")
    if not isinstance(c, str):
        c = ""
    reasoning = None
    for k in ("reasoning_content", "reasoning", "thinking"):
        v = msg.get(k)
        if isinstance(v, str) and v.strip():
            reasoning = v
            break
    m = _THINK.match(c)
    if m:
        inner = m.group(1).strip()
        c = c[m.end():]
        reasoning = f"{reasoning}\n{inner}" if reasoning and inner else (reasoning or inner or None)
    return c, reasoning, finish if isinstance(finish, str) else None


def usage(body: Any) -> dict:
    u = body.get("usage") if isinstance(body, dict) else None
    u = u if isinstance(u, dict) else {}

    def n(k):
        v = u.get(k)
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0
    p, c = n("prompt_tokens"), n("completion_tokens")
    t = n("total_tokens") or p + c
    return {"prompt": p, "completion": c, "total": t}
