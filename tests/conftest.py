import pytest

from app.config import load_config
from app.core import clock


@pytest.fixture
def cfg(tmp_path):
    return load_config(tmp_path / "data")


@pytest.fixture(autouse=True)
def _reset_clock():
    yield
    clock.set_now_fn(None)


@pytest.fixture(autouse=True)
def _no_real_ocr_models(tmp_path, monkeypatch):
    """Tests never load the real PP-OCRv6 models: the model root points at an empty folder (OCR tests use a fake
    engine). Without this a local-OCR test would export ~160 MB of OpenVINO IR into the temp dir."""
    from app.ocr import service
    monkeypatch.setattr(service, "model_root", lambda cfg: tmp_path / "no_models")
    service._env_cache.update(t=0.0, v=None)
    service._engine.update(name=None, obj=None)
    yield
    service._env_cache.update(t=0.0, v=None)
    service._engine.update(name=None, obj=None)


@pytest.fixture(autouse=True)
def _no_shipped_builtin_templates(tmp_path_factory, monkeypatch):
    """The shipped builtin prompt templates (app/prompts/builtin/) are hidden from tests by default so template
    lists are deterministic; tests that need them use the `builtin_dir` fixture (tests/test_prompts.py)."""
    import app.prompts.store as tstore
    empty = tmp_path_factory.getbasetemp() / "no_builtin_templates"        # outside each test's tmp_path
    empty.mkdir(exist_ok=True)
    monkeypatch.setattr(tstore, "BUILTIN_DIR", empty)


@pytest.fixture(autouse=True)
def _wait_result_threads():
    """result-<task_code>.json is generated in a background thread after completion; finish it before teardown."""
    yield
    from app.batches import results
    results.wait_idle(10)
