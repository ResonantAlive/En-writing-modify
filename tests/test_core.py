import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from app.config import DEFAULTS, load_config
from app.core import atomic_io, clock, counters
from app.core.locks import ExecutorLock, ExecutorLockError
from app.core.paths import safe_join, sanitize_filename
from app.errors import LimitExceeded, NotFound, StartupError

ROOT = str(Path(__file__).resolve().parent.parent)
PY = sys.executable


@pytest.fixture(autouse=True)
def _restore_io_cfg():
    yield
    atomic_io.configure(2.0)


# ---------------- 1. atomic_io ----------------
def test_roundtrip_chinese(tmp_path):
    p = tmp_path / "a" / "x.json"
    obj = {"åå­—": "å¼ ä¸‰", "ä½œæ–‡": "ä»Šå¤©å¤©æ°”å¾ˆå¥½", "n": [1, 2]}
    atomic_io.write_json_atomic(p, obj)
    assert atomic_io.read_json(p) == obj
    raw = p.read_bytes()
    assert "å¼ ä¸‰".encode("utf-8") in raw
    assert b"\\u" not in raw
    assert [x.name for x in p.parent.iterdir()] == ["x.json"]


def test_read_json_default(tmp_path):
    assert atomic_io.read_json(tmp_path / "nope.json", "d") == "d"


def test_replace_retried_then_succeeds(tmp_path, monkeypatch):
    real = os.replace
    calls = {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] < 4:
            raise PermissionError("locked")
        return real(a, b)

    monkeypatch.setattr(atomic_io.os, "replace", flaky)
    atomic_io.configure(2.0)
    p = tmp_path / "f.json"
    atomic_io.write_json_atomic(p, {"ok": 1})
    assert calls["n"] == 4
    assert atomic_io.read_json(p) == {"ok": 1}
    assert [x.name for x in tmp_path.iterdir()] == ["f.json"]


def test_persistent_permission_error_raises(tmp_path, monkeypatch):
    def boom(a, b):
        raise PermissionError("locked")

    monkeypatch.setattr(atomic_io.os, "replace", boom)
    atomic_io.configure(0.3)
    p = tmp_path / "f.json"
    t0 = time.monotonic()
    with pytest.raises(PermissionError):
        atomic_io.write_json_atomic(p, {"a": 1})
    el = time.monotonic() - t0
    assert 0.25 <= el < 2.0
    assert list(tmp_path.iterdir()) == []  # tmp cleaned up


def test_crash_mid_write_keeps_old_file(tmp_path, monkeypatch):
    p = tmp_path / "f.json"
    atomic_io.write_json_atomic(p, {"v": "oldæ—§"})

    def crash(fd):
        raise OSError("disk died")

    monkeypatch.setattr(atomic_io.os, "fsync", crash)
    with pytest.raises(OSError):
        atomic_io.write_json_atomic(p, {"v": "new"})
    monkeypatch.undo()
    assert atomic_io.read_json(p) == {"v": "oldæ—§"}
    assert [x.name for x in tmp_path.iterdir()] == ["f.json"]


def test_append_jsonl_threads(tmp_path):
    p = tmp_path / "log.jsonl"
    errs = []

    def w(i):
        try:
            for j in range(20):
                atomic_io.append_jsonl(p, {"i": i, "j": j, "t": "ä¸­æ–‡"})
        except Exception as e:  # pragma: no cover
            errs.append(e)

    ts = [threading.Thread(target=w, args=(i,)) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs
    lines = p.read_bytes().decode("utf-8").splitlines()
    assert len(lines) == 160
    parsed = [json.loads(l) for l in lines]
    assert len({(d["i"], d["j"]) for d in parsed}) == 160


# ---------------- 2. counters ----------------
def fixed(y, m, d):
    clock.set_now_fn(lambda: datetime(y, m, d, 10, 0, 0))


def test_format_and_order(cfg):
    fixed(2026, 3, 5)
    code, subs = counters.allocate_batch(cfg, 3)
    assert code == "050320260001" and len(code) == 12
    assert subs == [code + "00001", code + "00002", code + "00003"]
    assert all(len(s) == 17 for s in subs)


def test_same_day_increments_and_new_date_resets(cfg):
    fixed(2026, 3, 5)
    assert counters.allocate_batch(cfg, 1)[0] == "050320260001"
    assert counters.allocate_batch(cfg, 1)[0] == "050320260002"
    fixed(2026, 3, 6)
    assert counters.allocate_batch(cfg, 1)[0] == "060320260001"


def test_clock_rollback_continues_from_saved_max(cfg):
    fixed(2026, 3, 5)
    counters.allocate_batch(cfg, 1)
    counters.allocate_batch(cfg, 1)
    fixed(2026, 3, 6)
    counters.allocate_batch(cfg, 1)
    fixed(2026, 3, 5)  # rollback
    assert counters.allocate_batch(cfg, 1)[0] == "050320260003"
    data = json.loads((cfg.state_dir / "counters.json").read_text("utf-8"))
    assert data["task_seq"] == {"05032026": 3, "06032026": 1}


def test_task_limit_no_wrap(cfg):
    fixed(2026, 3, 5)
    atomic_io.write_json_atomic(cfg.state_dir / "counters.json",
                                {"schema_version": 1, "task_seq": {"05032026": 9999}})
    for _ in range(2):
        with pytest.raises(LimitExceeded) as e:
            counters.allocate_batch(cfg, 1)
        assert e.value.code == "task_limit_exceeded"
    data = atomic_io.read_json(cfg.state_dir / "counters.json")
    assert data["task_seq"]["05032026"] == 9999
    fixed(2026, 3, 6)  # other day still fine
    assert counters.allocate_batch(cfg, 1)[0] == "060320260001"


def test_subtask_limit_does_not_consume(cfg):
    fixed(2026, 3, 5)
    with pytest.raises(LimitExceeded) as e:
        counters.allocate_batch(cfg, 100000)
    assert e.value.code == "subtask_limit_exceeded"
    assert not (cfg.state_dir / "counters.json").exists()
    assert counters.allocate_batch(cfg, 1)[0] == "050320260001"
    with pytest.raises(LimitExceeded):
        counters.subtask_id("050320260001", 100000)
    with pytest.raises(LimitExceeded):
        counters.subtask_id("050320260001", 0)
    assert counters.subtask_id("050320260001", 99999).endswith("99999")


def test_empty_batch_rejected(cfg):
    with pytest.raises(LimitExceeded) as e:
        counters.allocate_batch(cfg, 0)
    assert e.value.code == "empty_batch"


def test_restart_safe(cfg):
    fixed(2026, 3, 5)
    counters.allocate_batch(cfg, 1)
    counters.allocate_batch(cfg, 1)
    cfg2 = load_config(cfg.data_dir)
    assert counters.allocate_batch(cfg2, 1)[0] == "050320260003"


WORKER = textwrap.dedent("""
    import sys, json
    sys.path.insert(0, {root!r})
    from pathlib import Path
    from app.config import load_config
    from app.core import counters
    cfg = load_config(Path({data!r}))
    out = [counters.allocate_batch(cfg, 2)[0] for _ in range(25)]
    print(json.dumps(out))
""")


def test_concurrent_processes_and_threads_unique(cfg):
    procs = [subprocess.Popen([PY, "-I", "-c", WORKER.format(root=ROOT, data=str(cfg.data_dir))],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for _ in range(4)]
    results = []
    lock = threading.Lock()

    def tw():
        for _ in range(5):
            c = counters.allocate_batch(cfg, 1)[0]
            with lock:
                results.append(c)

    tl = [threading.Thread(target=tw) for _ in range(4)]
    [t.start() for t in tl]
    [t.join() for t in tl]
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
        results += json.loads(out.strip().splitlines()[-1])
    assert len(results) == 120
    assert len(set(results)) == 120
    assert {int(c[8:]) for c in results} == set(range(1, 121))


# ---------------- 3. ExecutorLock ----------------
HOLDER = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, {root!r})
    from app.core.locks import ExecutorLock
    l = ExecutorLock({lock!r}); l.acquire()
    print("READY", flush=True)
    time.sleep({sleep})
""")


def start_holder(lock_path, sleep=60):
    p = subprocess.Popen([PY, "-I", "-c", HOLDER.format(root=ROOT, lock=str(lock_path), sleep=sleep)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "READY"
    return p


def kill_tree(p):
    """The venv python.exe is a launcher that spawns the real interpreter as a child on
    Windows; killing only the parent leaves the child holding the lock."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    else:
        p.kill()
    p.wait()


def test_executor_lock_same_process(tmp_path):
    a = ExecutorLock(tmp_path / "e.lock")
    b = ExecutorLock(tmp_path / "e.lock")
    a.acquire()
    try:
        with pytest.raises(ExecutorLockError):
            b.acquire()
    finally:
        a.release()
    b.acquire()
    b.release()


def test_executor_lock_error_is_startup_error():
    assert issubclass(ExecutorLockError, StartupError)


def test_executor_lock_subprocess(tmp_path):
    lp = tmp_path / "state" / "e.lock"
    h = start_holder(lp, sleep=60)
    try:
        lk = ExecutorLock(lp)
        with pytest.raises(ExecutorLockError):
            lk.acquire()
    finally:
        kill_tree(h)
    lk = ExecutorLock(lp)
    lk.acquire()
    lk.release()


def test_executor_lock_released_on_holder_exit(tmp_path):
    lp = tmp_path / "e.lock"
    h = start_holder(lp, sleep=1)
    h.wait(timeout=30)
    lk = ExecutorLock(lp)
    lk.acquire()
    lk.release()


def test_create_app_lifespan_refuses_when_locked(cfg):
    from fastapi.testclient import TestClient
    from app.main import create_app
    h = start_holder(cfg.lock_path("executor"))
    try:
        app = create_app(cfg.data_dir)
        with pytest.raises(ExecutorLockError):
            with TestClient(app):
                pass
    finally:
        kill_tree(h)
    app = create_app(cfg.data_dir)
    with TestClient(app):
        pass  # now acquires fine
    lk = ExecutorLock(cfg.lock_path("executor"))  # released afterwards
    lk.acquire()
    lk.release()


# ---------------- 4. config ----------------
def mk(tmp_path, **net):
    d = tmp_path / "d"
    (d / "app").mkdir(parents=True, exist_ok=True)
    (d / "app" / "settings.json").write_text(json.dumps({"network": net}), "utf-8")
    return load_config(d)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]", "127.5.5.5"])
def test_loopback_ok(tmp_path, host):
    c = mk(tmp_path, host=host)
    assert c.check_network() == []
    assert c.secure_cookies is False


def test_non_loopback_http_refused(tmp_path):
    c = mk(tmp_path, host="0.0.0.0")
    with pytest.raises(StartupError):
        c.check_network()
    assert c.secure_cookies is True


def test_non_loopback_allow_insecure_warns(tmp_path):
    c = mk(tmp_path, host="0.0.0.0", allow_insecure_lan=True)
    w = c.check_network()
    assert len(w) == 1 and "WARNING" in w[0]


def test_non_loopback_https_ok(tmp_path):
    c = mk(tmp_path, host="0.0.0.0", https=True)
    assert c.check_network() == []
    assert c.secure_cookies is True


def test_hostname_not_loopback(tmp_path):
    c = mk(tmp_path, host="example.com")
    with pytest.raises(StartupError):
        c.check_network()


def test_settings_merge(tmp_path):
    d = tmp_path / "d"
    (d / "app").mkdir(parents=True)
    (d / "app" / "settings.json").write_text(
        json.dumps({"concurrency": 9, "vlm": {"quality": 50}, "network": {"port": 9000}}), "utf-8")
    c = load_config(d)
    assert c["concurrency"] == 9
    assert c["vlm"]["quality"] == 50 and c["vlm"]["max_edge"] == 2048
    assert c.net["port"] == 9000 and c.net["host"] == "127.0.0.1"
    assert c["session_days"] == 7
    assert DEFAULTS["concurrency"] == 4 and DEFAULTS["vlm"]["quality"] == 90  # defaults not mutated


def test_defaults_written_on_first_load(tmp_path):
    c = load_config(tmp_path / "d")
    assert atomic_io.read_json(c.app_dir / "settings.json")["concurrency"] == 4


def test_load_config_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WM_DATA_DIR", str(tmp_path / "envd"))
    c = load_config()
    assert c.data_dir == tmp_path / "envd"


def test_create_app_refuses_non_loopback_http(tmp_path):
    from app.main import create_app
    d = tmp_path / "d"
    (d / "app").mkdir(parents=True)
    (d / "app" / "settings.json").write_text(json.dumps({"network": {"host": "0.0.0.0"}}), "utf-8")
    with pytest.raises(StartupError):
        create_app(d)


# ---------------- 5. paths ----------------
def test_safe_join(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    assert safe_join(base, "a", "b.txt") == (base / "a" / "b.txt").resolve()
    assert safe_join(base) == base.resolve()
    for bad in (("..", "x"), ("a", "..", "..", "x"), ("../x",), ("..\\x",), (str(tmp_path),)):
        with pytest.raises(NotFound):
            safe_join(base, *bad)
    (tmp_path / "base2").mkdir()
    with pytest.raises(NotFound):  # sibling sharing a name prefix
        safe_join(base, "..", "base2")


def test_safe_join_absolute_part(tmp_path):
    base = tmp_path / "b"
    base.mkdir()
    with pytest.raises(NotFound):
        safe_join(base, "C:\\Windows\\win.ini" if os.name == "nt" else "/etc/passwd")


def test_sanitize_filename():
    assert sanitize_filename('a<b>c:d"e/f\\g|h?i*j.txt') == "a_b_c_d_e_f_g_h_i_j.txt"
    out = sanitize_filename("../../etc/passwd")
    assert "/" not in out and ".." not in out
    assert sanitize_filename("") == "file"
    assert sanitize_filename(None) == "file"
    assert sanitize_filename("  . . ") == "file"
    assert sanitize_filename("ä½œæ–‡1.jpg") == "ä½œæ–‡1.jpg"
    assert sanitize_filename("a\x00b\x1f.png") == "a_b_.png"
    assert len(sanitize_filename("x" * 500 + ".png")) == 120


@pytest.mark.parametrize("n", ["CON", "con.txt", "NUL.tar.gz", "COM1", "lpt9.png", "aux"])
def test_sanitize_reserved(n):
    out = sanitize_filename(n)
    assert out.startswith("_") and out[1:] == n


def test_sanitize_not_reserved():
    assert sanitize_filename("CONSOLE.txt") == "CONSOLE.txt"

