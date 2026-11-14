"""System-prompt rendering: all placeholders are replaced in ONE regex pass, so substituted values
(e.g. a rubric that literally contains {{FEWSHOT}}) are never re-scanned."""
from __future__ import annotations

import re

_PH = re.compile(r"\{\{(TOPIC|RUBRIC|GENRE|FULL_SCORE|FEWSHOT)\}\}")


def _num(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return "" if v is None else str(v)


def render_system_prompt(version: dict, batch: dict, fewshot_text: str | None) -> str:
    vals = {"TOPIC": batch.get("topic") or "", "RUBRIC": batch.get("rubric") or "",
            "GENRE": batch.get("genre") or "", "FULL_SCORE": _num(batch.get("full_score")),
            "FEWSHOT": fewshot_text or ""}
    return _PH.sub(lambda m: vals[m.group(1)], version["text"])
