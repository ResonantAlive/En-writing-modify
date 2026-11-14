"""File locks: short critical sections + the process-lifetime executor lock."""
from __future__ import annotations

from pathlib import Path

from filelock import FileLock, Timeout

from ..errors import StartupError


def file_lock(lock_path: Path, timeout: float = 30.0) -> FileLock:
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(lock_path), timeout=timeout)


class ExecutorLockError(StartupError):
    pass


class ExecutorLock:
    """Held for the whole process lifetime; a second process cannot acquire it."""

    def __init__(self, lock_path: Path):
        lock_path = Path(lock_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(lock_path), timeout=0)
        self.path = lock_path

    def acquire(self) -> None:
        try:
            self._lock.acquire(timeout=0)
        except Timeout:
            raise ExecutorLockError(
                f"executor lock {self.path} is held by another process; "
                "only one service process may run (refusing to start)"
            )

    def release(self) -> None:
        if self._lock.is_locked:
            self._lock.release()
