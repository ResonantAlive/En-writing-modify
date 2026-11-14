"""Model-output parsing chain.

Levels (the first one that yields a JSON object wins):
  direct   json.loads(whole text)
  fenced   a ```json ... ``` block (any fenced block that parses)
  balanced largest bracket-balanced {...} slice that json.loads accepts (prefix / suffix prose, nested braces
           inside strings are handled)
  repair   json-repair (trailing commas, single quotes, comments, raw newlines, True/None, truncation);
           ALL fields of the repaired object are kept
  regex    only a score could be found ("判分: 7"): never a full result -> `missing_required_field` unless the
           template only requires the score key
After an object exists it is validated (required keys, numeric score within [0, full_score]).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - import guard only
    import json_repair as _json_repair
except Exception:  # pragma: no cover
    _json_repair = None


@dataclass
class ParseResult:
    level: str | None = None            # direct|fenced|balanced|repair|regex|None
    obj: dict | None = None
    score: float | None = None
    error_code: str | None = None
    error_detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.error_code is None


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n?(.*?)```", re.S)
_REFUSAL = re.compile(
    r"(i\s*(?:'m|am)?\s*(?:sorry|unable)|i\s*can(?:'|’)?t\s+(?:help|assist|comply|provide|grade|do)|"
    r"i\s*cannot\s+(?:help|assist|comply|provide|grade|do)|i\s*won(?:'|’)?t|as an ai|"
    r"against (?:my|our) (?:policy|policies|guidelines)|content policy|"
    r"抱歉|对不起|无法(?:协助|提供|评分|完成|帮助)|不能(?:协助|提供|帮助)|无法满足|违反.{0,6}(?:政策|规定))",
    re.I)
_NUM = r"(-?\d+(?:\.\d+)?)"


def _loads_dict(s: str) -> dict | None:
    try:
        v = json.loads(s)
    except (ValueError, RecursionError):
        return None
    return v if isinstance(v, dict) else None


def _balanced_spans(text: str, max_spans: int = 300) -> tuple[list[str], int]:
    """Outermost bracket-balanced {...} spans (string aware). Returns (spans that json.loads accepts,
    length of the longest outermost span that does NOT parse). Inner objects of a broken span are never
    taken on their own (that would silently drop the other fields)."""
    good: list[str] = []
    broken = 0
    i, n, count = 0, len(text), 0
    while count < max_spans:
        i = text.find("{", i)
        if i < 0:
            break
        count += 1
        depth, j, in_str, esc, end = 0, i, False, False, -1
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
            j += 1
        if end < 0:                       # unbalanced (truncated): leave to repair
            broken = max(broken, n - i)
            break
        cand = text[i:end + 1]
        if _loads_dict(cand) is not None:
            good.append(cand)
        else:
            broken = max(broken, len(cand))
        i = end + 1
    return good, broken


def _repair_obj(text: str) -> dict | None:
    if _json_repair is None or "{" not in text:
        return None
    fences = [m.group(2) for m in _FENCE.finditer(text) if "{" in m.group(2)]
    sources = fences + [text[text.index("{"):]]
    best: dict | None = None
    for src in sources:
        try:
            v = _json_repair.repair_json(src, return_objects=True)
        except Exception:
            continue
        if isinstance(v, list):
            v = max((x for x in v if isinstance(x, dict)), key=lambda d: len(d), default=None)
        if isinstance(v, dict) and v and (best is None or len(v) > len(best)):
            best = v
    return best


def extract_object(text: str) -> tuple[str | None, dict | None]:
    t = text.strip().lstrip("﻿")
    obj = _loads_dict(t)
    if obj is not None:
        return "direct", obj
    for m in _FENCE.finditer(t):
        obj = _loads_dict(m.group(2).strip())
        if obj is not None:
            return "fenced", obj
    cands, broken = _balanced_spans(t)
    if cands:
        best = max(cands, key=len)
        if len(best) >= broken:
            return "balanced", json.loads(best)
    obj = _repair_obj(t)
    if obj is not None:
        return "repair", obj
    return None, None


def _to_number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
    elif isinstance(v, str):
        try:
            f = float(v.strip())
        except ValueError:
            return None
    else:
        return None
    return f if math.isfinite(f) else None


def _find_score_key(obj: dict, score_key: str, aliases: list[str]) -> str | None:
    for k in [score_key, *aliases]:
        if k in obj:
            return k
    return None


def _regex_score(text: str, keys: list[str]) -> float | None | bool:
    """float = number found; None = no key found. (illegal range is judged by the caller)"""
    for k in keys:
        if not k:
            continue
        m = re.search(r"[\"'“]?" + re.escape(k) + r"[\"'”]?\s*[:：=]\s*[\"'“]?\s*" + _NUM, text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def parse_response(text: str, *, score_key: str, score_aliases: list[str] | None = None,
                   criteria_key: str = "", required_keys: list[str] | None = None,
                   full_score: float = 100) -> ParseResult:
    aliases = list(score_aliases or [])
    required = list(required_keys) if required_keys is not None else [score_key, criteria_key]
    required = [k for k in required if k]
    if text is None or not str(text).strip():
        return ParseResult(error_code="empty_response", error_detail="the model returned no content")
    text = str(text)

    level, obj = extract_object(text)
    if obj is not None:
        return _validate(level, obj, score_key, aliases, required, full_score)

    # level 4: regex score only
    val = _regex_score(text, [score_key, *aliases])
    if val is not None:
        legal = 0 <= val <= full_score
        only_score = required == [score_key]
        if legal and only_score:
            return ParseResult("regex", {score_key: val}, val)
        if not legal:
            return ParseResult("regex", None, None, "score_out_of_range",
                               f"score {val} is outside [0, {full_score}]")
        return ParseResult("regex", {score_key: val}, val, "missing_required_field",
                           "only a score could be extracted; required fields are missing: "
                           + ", ".join(k for k in required if k != score_key))
    if _REFUSAL.search(text):
        return ParseResult(error_code="refusal_or_safety", error_detail="the model refused to answer")
    return ParseResult(error_code="json_parse_failed", error_detail="no JSON object found in the reply")


def _validate(level: str | None, obj: dict, score_key: str, aliases: list[str], required: list[str],
              full_score: float) -> ParseResult:
    missing = []
    for k in required:
        if k == score_key:
            if _find_score_key(obj, score_key, aliases) is None:
                missing.append(k)
        elif k not in obj:
            missing.append(k)
    sk = _find_score_key(obj, score_key, aliases)
    if missing:
        return ParseResult(level, obj, None, "missing_required_field", "missing: " + ", ".join(missing))
    score = None
    if sk is not None:
        score = _to_number(obj[sk])
        if score is None:
            return ParseResult(level, obj, None, "missing_required_field",
                               f"{sk} is not a number")
        if not (0 <= score <= full_score):
            return ParseResult(level, obj, None, "score_out_of_range",
                               f"score {score:g} is outside [0, {full_score:g}]")
        if sk != score_key:                  # normalise: expose the score under the canonical key too
            obj = {**obj, score_key: obj[sk]}
    return ParseResult(level, obj, score, None, None)
