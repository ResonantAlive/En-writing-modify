"""Application logging with defensive secret redaction (keys/tokens/invite codes/Bearer)."""
from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler

from .config import AppConfig

_PATTERNS = [
    (re.compile(r"wm_[A-Za-z0-9_\-]{20,}"), "wm_***"),
    (re.compile(r"(?i)bearer\s+\S+"), "Bearer ***"),
    (re.compile(r"\b[A-Z2-7]{4}(?:-[A-Z2-7]{4}){7}\b"), "****-INVITE-****"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "sk-***"),
]


class RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for pat, rep in _PATTERNS:
            msg = pat.sub(rep, msg)
        record.msg, record.args = msg, ()
        return True


def setup_logging(cfg: AppConfig) -> None:
    root = logging.getLogger("wm")
    root.setLevel(logging.INFO)
    if any(getattr(h, "_wm", False) for h in root.handlers):
        return
    h = RotatingFileHandler(cfg.app_log_dir / "app.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    h.addFilter(RedactFilter())
    h._wm = True  # type: ignore[attr-defined]
    root.addHandler(h)
