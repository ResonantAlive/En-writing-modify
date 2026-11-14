"""Configuration: data/app/settings.json merged over defaults."""
from __future__ import annotations

import copy
import ipaddress
import os
from pathlib import Path
from typing import Any

from .core import atomic_io, clock
from .errors import StartupError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "schema_version": 1,
    "default_local_engine": "paddle",
    "timezone": "local",
    "concurrency": 4,
    "max_files_per_batch": 100,
    "max_file_mb": 20,
    "max_pages_per_essay": 6,
    "staging_ttl_hours": 24,
    "fewshot_k": 3,
    "fewshot_token_budget": 6000,
    "vlm": {"max_edge": 2048, "quality": 90, "max_request_mb": 18},
    "breaker": {"min_processed": 10, "ratio": 0.3, "api_error_threshold": 3},
    "layout": {"line_merge_overlap": 0.5, "blank_line_factor": 1.6, "space_px": 12},
    "review": {"max_items": 200, "max_item_chars": 10000, "max_name_chars": 100, "max_body_mb": 8},
    "llm": {"timeout_s": 120, "max_tokens": 8192, "temperature": 0.0, "retry_max": 3},
    "prompt": {"max_chars": 20000},
    # local OCR = PP-OCRv6 only. backend: "openvino" (default, FP32) or "torch"; threads 0 = all cores;
    # ir_dir: exported OpenVINO model (default data/ov_models). No downscaling / no INT8 (quality rule).
    "ocr": {"model_root": None, "backend": "openvino", "ir_dir": None, "threads": 0},
    "result_backups_keep": 20,
    "max_json_mb": 2,              # body limit of every non-multipart request (the review endpoint: review.max_body_mb)
    "session_days": 7,
    "lockout": {"fails": 5, "minutes": 10},
    "last_used_write_interval_s": 300,
    "replace_retry_total_s": 2.0,
    "network": {
        "host": "127.0.0.1",
        "port": 8000,
        "https": False,            # True = served over HTTPS (own TLS or via HTTPS reverse proxy)
        "ssl_certfile": None,
        "ssl_keyfile": None,
        "allow_insecure_lan": False,
        "trusted_proxies": [],     # only these peers may set X-Forwarded-For
        "cors_origins": [],
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


class AppConfig:
    def __init__(self, data_dir: Path, settings: dict):
        self.data_dir = Path(data_dir)
        self.settings = settings
        self.app_dir = self.data_dir / "app"
        self.state_dir = self.data_dir / "state"
        self.locks_dir = self.state_dir / "locks"
        self.staging_dir = self.data_dir / "staging"
        self.logs_dir = self.data_dir / "logs"
        self.app_log_dir = self.logs_dir / "app"
        self.audit_dir = self.logs_dir / "audit"
        self.history_dir = self.data_dir / "history"

    def ensure_dirs(self) -> None:
        for d in (self.app_dir, self.state_dir, self.locks_dir, self.staging_dir,
                  self.app_log_dir, self.audit_dir, self.history_dir):
            d.mkdir(parents=True, exist_ok=True)

    def lock_path(self, name: str) -> Path:
        return self.locks_dir / f"{name}.lock"

    def __getitem__(self, key: str) -> Any:
        return self.settings[key]

    @property
    def net(self) -> dict:
        return self.settings["network"]

    @property
    def secure_cookies(self) -> bool:
        """Cookies get `Secure` whenever the service is not loopback-only."""
        return not is_loopback(self.net["host"])

    def check_network(self) -> list[str]:
        """Refuse to start on non-loopback without HTTPS unless explicitly allowed.
        Returns warnings (printed prominently by the launcher)."""
        n = self.net
        warnings: list[str] = []
        if not is_loopback(n["host"]) and not n["https"]:
            if not n["allow_insecure_lan"]:
                raise StartupError(
                    f"listening on non-loopback address {n['host']!r} requires HTTPS "
                    "(network.https=true via own TLS or an HTTPS reverse proxy); "
                    "set network.allow_insecure_lan=true to override (NOT recommended)"
                )
            warnings.append(
                "!!! WARNING: allow_insecure_lan=true -- serving over plain HTTP on a non-loopback "
                "address. Passwords and student data cross the network UNENCRYPTED. !!!"
            )
        return warnings


def load_config(data_dir: Path | str | None = None) -> AppConfig:
    data_dir = Path(data_dir or os.environ.get("WM_DATA_DIR") or PROJECT_ROOT / "data")
    path = data_dir / "app" / "settings.json"
    user = atomic_io.read_json(path, None)
    settings = _merge(DEFAULTS, user or {})
    cfg = AppConfig(data_dir, settings)
    cfg.ensure_dirs()
    if user is None:
        atomic_io.write_json_atomic(path, settings)
    atomic_io.configure(settings["replace_retry_total_s"])
    clock.configure(settings["timezone"])
    return cfg
