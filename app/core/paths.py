"""Path safety helpers."""
from __future__ import annotations

import re
from pathlib import Path

from ..errors import NotFound

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_join(base: Path, *parts: str) -> Path:
    """Join and require the result to stay inside base (no traversal); else 404."""
    base = Path(base).resolve()
    p = base.joinpath(*parts).resolve()
    if p != base and base not in p.parents:
        raise NotFound()
    return p


def sanitize_filename(name: str, max_len: int = 120) -> str:
    name = _ILLEGAL.sub("_", name or "").strip(" .")
    name = name.replace("..", "_")
    if not name:
        name = "file"
    stem = name.split(".")[0].upper()
    if stem in _RESERVED:
        name = "_" + name
    return name[:max_len]
