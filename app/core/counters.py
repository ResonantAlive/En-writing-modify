"""Task/subtask numbering. Atomic and cross-process safe (lock + persisted counter).

task_code  = DDMMYYYY + 4-digit daily sequence   (12 chars)
subtask_id = task_code + 5-digit sub sequence    (17 chars)
Counters store the max allocated sequence PER DATE, so a clock rollback resumes from the
saved max of that date and can never duplicate an issued number. Never wraps around.
"""
from __future__ import annotations

from ..config import AppConfig
from ..errors import LimitExceeded
from . import atomic_io, clock
from .locks import file_lock

MAX_TASKS_PER_DAY = 9999
MAX_SUBTASKS_PER_TASK = 99999


def subtask_id(task_code: str, index: int) -> str:
    if not 1 <= index <= MAX_SUBTASKS_PER_TASK:
        raise LimitExceeded("subtask_limit_exceeded", f"a batch holds at most {MAX_SUBTASKS_PER_TASK} essays")
    return f"{task_code}{index:05d}"


def allocate_batch(cfg: AppConfig, n_subtasks: int) -> tuple[str, list[str]]:
    """Reserve the next task_code for today plus n_subtasks ids (1-based, in order)."""
    if n_subtasks < 1:
        raise LimitExceeded("empty_batch", "a batch needs at least one essay")
    if n_subtasks > MAX_SUBTASKS_PER_TASK:
        raise LimitExceeded("subtask_limit_exceeded", f"a batch holds at most {MAX_SUBTASKS_PER_TASK} essays")
    path = cfg.state_dir / "counters.json"
    with file_lock(cfg.lock_path("counters")):
        data = atomic_io.read_json(path, None) or {"schema_version": 1, "task_seq": {}}
        day = clock.date_str()
        seq = int(data["task_seq"].get(day, 0)) + 1
        if seq > MAX_TASKS_PER_DAY:
            raise LimitExceeded("task_limit_exceeded", f"at most {MAX_TASKS_PER_DAY} batches per day")
        data["task_seq"][day] = seq
        atomic_io.write_json_atomic(path, data)
    task_code = f"{day}{seq:04d}"
    return task_code, [subtask_id(task_code, i) for i in range(1, n_subtasks + 1)]
