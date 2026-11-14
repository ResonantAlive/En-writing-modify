"""History directory layout: history/YYYY/MM/DDMMYYYY/<task_code>/<subtask_id>/ (derived from ids only)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

from ..errors import NotFound

TASK_RE = re.compile(r"^\d{12}$")
SUB_RE = re.compile(r"^\d{17}$")


def valid_task_code(task_code: str) -> bool:
    return isinstance(task_code, str) and bool(TASK_RE.match(task_code))


def valid_subtask_id(subtask_id: str) -> bool:
    return isinstance(subtask_id, str) and bool(SUB_RE.match(subtask_id))


def batch_dir(cfg, task_code: str) -> Path:
    if not valid_task_code(task_code):
        raise NotFound()
    d = task_code[:8]                       # DDMMYYYY
    return Path(cfg.history_dir) / d[4:8] / d[2:4] / d / task_code


def subtask_dir(cfg, subtask_id: str) -> Path:
    if not valid_subtask_id(subtask_id):
        raise NotFound()
    return batch_dir(cfg, subtask_id[:12]) / subtask_id


def find_batch_dir(cfg, task_code: str) -> Path | None:
    """The batch directory if it exists and is committed (has batch.json), else None."""
    if not valid_task_code(task_code):
        return None
    d = batch_dir(cfg, task_code)
    return d if (d / "batch.json").is_file() else None


def _subdirs(p: Path, rx: re.Pattern) -> list[Path]:
    try:
        with os.scandir(p) as it:
            return sorted((Path(e.path) for e in it if e.is_dir() and rx.match(e.name)), key=lambda x: x.name)
    except FileNotFoundError:
        return []


_Y, _M, _D = re.compile(r"^\d{4}$"), re.compile(r"^\d{2}$"), re.compile(r"^\d{8}$")


def iter_task_dirs(cfg) -> Iterator[Path]:
    """Every task directory (committed or not) under history/."""
    for y in _subdirs(Path(cfg.history_dir), _Y):
        for m in _subdirs(y, _M):
            for d in _subdirs(m, _D):
                yield from _subdirs(d, TASK_RE)
