"""Unified error type; all API errors serialize as {code, message, detail}."""
from __future__ import annotations

from typing import Any


class AppError(Exception):
    def __init__(self, code: str, message: str = "", status: int = 400, detail: Any = None):
        super().__init__(message or code)
        self.code = code
        self.message = message or code
        self.status = status
        self.detail = detail

    def body(self) -> dict:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class NotFound(AppError):
    def __init__(self, message: str = "not found"):
        super().__init__("not_found", message, 404)


class LimitExceeded(AppError):
    def __init__(self, code: str, message: str):
        super().__init__(code, message, 409)


class StartupError(RuntimeError):
    """Raised when the service must refuse to start."""
