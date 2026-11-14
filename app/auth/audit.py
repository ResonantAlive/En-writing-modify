"""Append-only audit log: data/logs/audit/audit-YYYYMMDD.jsonl. Never put secrets in `detail`."""
from __future__ import annotations

from typing import Any

from ..config import AppConfig
from ..core import atomic_io, clock


def audit(cfg: AppConfig, actor: str | None, action: str, target: str | None = None,
          detail: dict[str, Any] | None = None, ip: str | None = None, role: str | None = None) -> None:
    now = clock.now()
    rec = {"at": clock.iso(now), "actor": actor, "role": role, "ip": ip,
           "action": action, "target": target, "detail": detail or {}}
    path = cfg.audit_dir / f"audit-{now.strftime('%Y%m%d')}.jsonl"
    atomic_io.append_jsonl(path, rec, cfg.lock_path("audit"))
