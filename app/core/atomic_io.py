"""Atomic file IO. Windows-aware: os.replace retries on PermissionError."""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

_replace_total_s = 2.0


def configure(replace_retry_total_s: float) -> None:
    global _replace_total_s
    _replace_total_s = float(replace_retry_total_s)


def _replace(src: Path, dst: Path) -> None:
    deadline = time.monotonic() + _replace_total_s
    delay = 0.01
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.2)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def write_json_atomic(path: Path, obj: Any) -> None:
    write_bytes_atomic(path, dumps(obj).encode("utf-8"))


def read_bytes(path: Path) -> bytes:
    """Read fully then close at once; retry briefly if another process is replacing the file."""
    deadline = time.monotonic() + _replace_total_s
    while True:
        try:
            with open(path, "rb") as f:
                return f.read()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        raw = read_bytes(Path(path))
    except FileNotFoundError:
        return default
    return json.loads(raw.decode("utf-8"))


def append_jsonl(path: Path, obj: Any, lock_path: Path | None = None) -> None:
    """Append one JSON line (lock serializes writers across processes)."""
    from .locks import file_lock
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    lp = lock_path or path.with_name(path.name + ".lock")
    with file_lock(lp):
        with open(path, "ab") as f:
            f.write(line.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
