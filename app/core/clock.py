"""Time source. Local timezone by default; overridable for tests (clock rollback)."""
from __future__ import annotations

from datetime import datetime, tzinfo
from typing import Callable, Optional

_tz: Optional[tzinfo] = None
_now_fn: Optional[Callable[[], datetime]] = None


def configure(tzname: str | None) -> None:
    global _tz
    if not tzname or tzname == "local":
        _tz = None
        return
    try:
        from zoneinfo import ZoneInfo
        _tz = ZoneInfo(tzname)
    except Exception as e:  # on Windows named zones need the `tzdata` package
        raise ValueError(f"invalid timezone {tzname!r} (on Windows install `tzdata`): {e}")


def set_now_fn(fn: Optional[Callable[[], datetime]]) -> None:
    global _now_fn
    _now_fn = fn


def now() -> datetime:
    if _now_fn is not None:
        return _now_fn()
    return datetime.now(_tz) if _tz else datetime.now().astimezone()


def iso(dt: datetime | None = None) -> str:
    return (dt or now()).isoformat(timespec="seconds")


def date_str(dt: datetime | None = None) -> str:
    """DDMMYYYY"""
    return (dt or now()).strftime("%d%m%Y")
