"""M2 batches: creation flow, numbering, recovery, access control, state machine, executor, SSE."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import batches as batches_api
from app.auth.deps import Principal
from tests.sheet_fixtures import make_sheet
from app.batches import create as create_mod
from app.batches import executor as ex_mod
from app.batches import hooks, paths, recover, store
from app.config import load_config
from app.core import atomic_io, clock
from app.errors import AppError
from app.main import create_app
from app.staging import uploads
from tests.fixtures_images import make_image

PW = "correct-horse-battery"
API = "/api/v1"


# --------------------------------------------------------------------------- fixtures / helpers
@pytest.fixture(autouse=True)
def _isolate():
    ex_mod.set_processor(None)
    ex_mod.on_batch_finished.clear()
    hooks.reset_defaults()
    yield
    ex_mod.set_processor(None)
    ex_mod.on_batch_finished.clear()
    hooks.reset_defaults()


def _unbind_logging():
    root = logging.getLogger("wm")
    for h in list(root.handlers):
        if getattr(h, "_wm", False):
            root.removeHandler(h)
            h.close()


class Env:
    def __init__(self, data, app):
        self.data, self.app = data, app
        self.cfg = app.state.cfg
        self.users: dict[str, dict] = {}
        self.clients: dict[str, TestClient] = {}

    def _codes(self):
        return json.loads((self.data / "app" / "invite_codes.json").read_text("utf-8"))

    def add_user(self, name: str, kind: str = "user") -> TestClient:
        c0 = TestClient(self.app)
        code = self._codes()["root_invite_code" if kind == "root" else "user_invite_code"]
        r = c0.post(f"{API}/auth/register", json={"invite_code": code, "username": name, "password": PW})
        assert r.status_code == 201, r.text
        self.users[name] = r.json()
        c = TestClient(self.app)
        r = c.post(f"{API}/auth/login", json={"username": name, "password": PW})
        assert r.status_code == 200, r.text
        c.csrf = r.json()["csrf_token"]
        self.clients[name] = c
        return c

    def uid(self, name: str) -> str:
        return self.users[name]["id"]

    def principal(self, name: str) -> Principal:
        u = self.users[name]
        return Principal(u["id"], name, u["role"], u["role"], "cookie")


def _make_env(tmp_path, settings=None):
    data = tmp_path / "data"
    if settings:
        (data / "app").mkdir(parents=True, exist_ok=True)
        (data / "app" / "settings.json").write_text(json.dumps(settings), "utf-8")
    _unbind_logging()
    app = create_app(data, enable_grading=False)        # no real processor; these tests drive fake ones
    hooks.reset_defaults()                              # create_app registers provider hooks; use the defaults
    return data, app


def _env_with(tmp_path, settings=None, tick=0.05):
    data, app = _make_env(tmp_path, settings)
    old = ex_mod.Executor.TICK_S
    ex_mod.Executor.TICK_S = tick
    ctx = TestClient(app)
    ctx.__enter__()
    e = Env(data, app)
    e.add_user("rooty", "root")
    e.add_user("alice")
    e.add_user("bob")
    return e, ctx, old


@pytest.fixture
def env(tmp_path):
    clock.set_now_fn(None)
    e, ctx, old = _env_with(tmp_path)
    yield e
    ctx.__exit__(None, None, None)
    ex_mod.Executor.TICK_S = old
    _unbind_logging()


@pytest.fixture
def env2(tmp_path):
    """concurrency = 2"""
    clock.set_now_fn(None)
    e, ctx, old = _env_with(tmp_path, {"concurrency": 2})
    yield e
    ctx.__exit__(None, None, None)
    ex_mod.Executor.TICK_S = old
    _unbind_logging()


@pytest.fixture
def env1(tmp_path):
    """concurrency = 1"""
    clock.set_now_fn(None)
    e, ctx, old = _env_with(tmp_path, {"concurrency": 1})
    yield e
    ctx.__exit__(None, None, None)
    ex_mod.Executor.TICK_S = old
    _unbind_logging()


def H(c):
    return {"X-CSRF-Token": c.csrf}


def jpg(i: int = 0) -> bytes:
    return make_image("jpeg", size=(40 + i, 20), color=(10 * (i % 20), 30, 30))


def stage(cfg, owner_id: str, names: list[str], exif: list[str | None] | None = None,
          mtimes: list[int | None] | None = None) -> tuple[str, dict, list[bytes]]:
    """Real staging dir (written by app.staging.uploads). Returns (upload_id, manifest, bytes list)."""
    blobs = [make_image("jpeg", size=(40 + i, 20), color=(9 * i % 250, 40, 90),
                        exif_time=(exif[i] if exif else None)) for i in range(len(names))]
    items = [(n, io.BytesIO(b), (mtimes[i] if mtimes else None)) for i, (n, b) in enumerate(zip(names, blobs))]
    m, rej = uploads.add_files(cfg, owner_id, None, items)
    assert not rej, rej
    return m["upload_id"], m, blobs


def body(upload_id: str, **over) -> dict:
    b = {"upload_id": upload_id, "topic": "My summer", "rubric": "content 10, language 10", "genre": "narrative",
         "full_score": 20, "route": "vlm", "provider_id": "p1", "model": "m1", "template_id": "t1",
         "template_version": 1, "vlm_transcription_choice": "skip_fewshot",
         "grouping": {"multi_photo": False}}
    b.update(over)
    return b


def staging_exists(cfg, upload_id: str) -> bool:
    return (cfg.staging_dir / upload_id / "manifest.json").exists()


def wait_for(pred, timeout=10.0, step=0.02):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("timed out waiting for condition")


def api_create(env: Env, who: str, names=("a.jpg",), **over) -> str:
    c = env.clients[who]
    up, _, _ = stage(env.cfg, env.uid(who), list(names))
    r = c.post(f"{API}/batches", json=body(up, **over), headers=H(c))
    assert r.status_code == 201, r.text
    return r.json()["task_code"]


def status_of(env: Env, who: str, tc: str) -> dict:
    r = env.clients[who].get(f"{API}/batches/{tc}")
    assert r.status_code == 200, r.text
    return r.json()


def post_action(env, who, tc, action):
    c = env.clients[who]
    return c.post(f"{API}/batches/{tc}/{action}", headers=H(c))


def err(r, status, code):
    assert r.status_code == status, r.text
    j = r.json()
    assert set(j) == {"code", "message", "detail"}
    assert j["code"] == code, j
    return j


def sub_statuses(env, tc) -> list[str]:
    return [m["status"] for m in store.read_metas(env.cfg, tc)]


def timeline_events(b: dict) -> list[str]:
    return [e["event"] for e in b["timeline"]]


# --------------------------------------------------------------------------- creation
def test_create_passes_topic_rubric_genre_score_to_revision_builder(env, monkeypatch):
    """Regression (found by the live DeepSeek test): the initial revision's prompt was rendered with empty
    {{TOPIC}}/{{RUBRIC}}/{{GENRE}}/{{FULL_SCORE}} because create_batch did not pass them on."""
    seen = {}
    orig = hooks.build_revision

    def spy(cfg, user, rbody, fewshot_ids, rev, reason, template_override=None):
        seen.update(rbody)
        return orig(cfg, user, rbody, fewshot_ids, rev, reason, template_override)
    monkeypatch.setattr(hooks, "build_revision", spy)
    api_create(env, "alice", topic="Topic X", rubric="Rubric Y", genre="letter", full_score=15)
    assert (seen["topic"], seen["rubric"], seen["genre"], seen["full_score"]) == ("Topic X", "Rubric Y", "letter", 15)


def test_create_basic_numbering_files_and_layout(env):
    cfg = env.cfg
    up, man, blobs = stage(cfg, env.uid("alice"), ["IMG_1.jpg", "IMG_2.jpg", "IMG_3.jpg"])
    c = env.clients["alice"]
    r = c.post(f"{API}/batches", json=body(up), headers=H(c))
    assert r.status_code == 201, r.text
    j = r.json()
    tc, sids = j["task_code"], j["subtask_ids"]
    assert j["status"] == "queued" and len(tc) == 12 and tc[:8] == clock.date_str()
    assert sids == [f"{tc}{i:05d}" for i in (1, 2, 3)]
    d = cfg.history_dir / tc[4:8] / tc[2:4] / tc[:8] / tc
    assert d == paths.batch_dir(cfg, tc) and paths.find_batch_dir(cfg, tc) == d
    b = json.loads((d / "batch.json").read_text("utf-8"))
    assert b["owner_id"] == env.uid("alice") and b["status"] == "queued" and b["language"] == "en"
    assert b["zero_shot_approved"] is True and b["vlm_transcription_choice"] == "skip_fewshot"
    assert b["local_engine"] is None and b["pause_reason"] is None
    assert b["config_revisions"][0]["rev"] == 1 and b["config_revisions"][0]["template"] == {"id": "t1", "version": 1}
    assert b["timeline"][0]["event"] == "created"
    assert [g[0]["orig_name"] for g in b["grouping"]["essays"]] == ["IMG_1.jpg", "IMG_2.jpg", "IMG_3.jpg"]
    for i, sid in enumerate(sids):                       # subtask order = essay order
        meta = json.loads((d / sid / "meta.json").read_text("utf-8"))
        assert meta["index"] == i + 1 and meta["status"] == "queued" and meta["owner_id"] == env.uid("alice")
        im = meta["images"][0]
        f = d / sid / im["original"]
        assert im["original"] == "image/original_1.jpg"
        data = f.read_bytes()
        assert data == blobs[i]                           # byte identical
        assert hashlib.sha256(data).hexdigest() == im["sha256"]
        assert im["order_basis"] == "single" and im["orig_name"] == f"IMG_{i + 1}.jpg"
    assert not staging_exists(cfg, up) and not (cfg.staging_dir / up).exists()


def test_multi_photo_per_n_and_manual_confirmation(env):
    cfg = env.cfg
    c = env.clients["alice"]
    names = [f"IMG_{i}.jpg" for i in (1, 2, 3, 4)]
    up, man, blobs = stage(cfg, env.uid("alice"), names)
    g = {"multi_photo": True, "mode": "per_n", "n": 2}
    r = c.post(f"{API}/batches", json=body(up, grouping=g), headers=H(c))
    assert r.status_code == 201, r.text
    tc = r.json()["task_code"]
    assert len(r.json()["subtask_ids"]) == 2
    metas = store.read_metas(cfg, tc)
    assert [im["orig_name"] for im in metas[0]["images"]] == ["IMG_1.jpg", "IMG_2.jpg"]
    assert [im["original"] for im in metas[1]["images"]] == ["image/original_1.jpg", "image/original_2.jpg"]
    assert metas[0]["images"][0]["order_basis"] == "filename" and metas[0]["images"][0]["order_value"] == [1]
    assert store.read_batch(cfg, tc)["grouping"]["mode"] == "per_n"

    # manual: must be confirmed
    up2, man2, _ = stage(cfg, env.uid("alice"), ["a.jpg", "b.jpg", "c.jpg"])
    fids = [f["file_id"] for f in man2["files"]]
    gm = {"multi_photo": True, "mode": "manual", "essays": [[fids[2], fids[0]], [fids[1]]]}
    j = err(c.post(f"{API}/batches", json=body(up2, grouping=gm), headers=H(c)), 422, "grouping_unconfirmed")
    assert staging_exists(cfg, up2)
    r = c.post(f"{API}/batches", json=body(up2, grouping={**gm, "confirmed": True}), headers=H(c))
    assert r.status_code == 201, r.text
    metas = store.read_metas(cfg, r.json()["task_code"])
    assert [im["orig_name"] for im in metas[0]["images"]] == ["c.jpg", "a.jpg"]       # user's order kept
    assert metas[0]["images"][0]["order_basis"] == "manual_user"


def test_grouping_invalid_keeps_staging(env):
    cfg = env.cfg
    c = env.clients["alice"]
    up, man, _ = stage(cfg, env.uid("alice"), ["a.jpg", "b.jpg"])
    fids = [f["file_id"] for f in man["files"]]
    gm = {"multi_photo": True, "mode": "manual", "essays": [[fids[0]]], "confirmed": True}   # b unassigned
    j = err(c.post(f"{API}/batches", json=body(up, grouping=gm), headers=H(c)), 422, "grouping_invalid")
    assert any(p["type"] == "unassigned_file" for p in j["detail"])
    assert staging_exists(cfg, up)
    assert (cfg.state_dir / "counters.json").exists() is False       # no number burnt


def test_transcription_choice_required_via_api(env):
    cfg = env.cfg
    c = env.clients["alice"]
    up, _, _ = stage(cfg, env.uid("alice"), ["a.jpg"])
    b = body(up)
    b.pop("vlm_transcription_choice")
    err(c.post(f"{API}/batches", json=b, headers=H(c)), 422, "transcription_choice_required")
    b["vlm_transcription_choice"] = None
    err(c.post(f"{API}/batches", json=b, headers=H(c)), 422, "transcription_choice_required")
    b["vlm_transcription_choice"] = "nonsense"
    err(c.post(f"{API}/batches", json=b, headers=H(c)), 422, "invalid_transcription_choice")
    assert staging_exists(cfg, up)


def test_field_validation_and_route(env):
    cfg = env.cfg
    c = env.clients["alice"]
    up, _, _ = stage(cfg, env.uid("alice"), ["a.jpg"])
    for field, val in (("topic", ""), ("topic", "   "), ("rubric", None), ("genre", ""), ("full_score", 0),
                       ("full_score", -3), ("full_score", None)):
        b = body(up)
        b[field] = val
        err(c.post(f"{API}/batches", json=b, headers=H(c)), 422, "validation_error")
    err(c.post(f"{API}/batches", json=body(up, full_score="abc"), headers=H(c)), 422, "validation_error")
    err(c.post(f"{API}/batches", json=body(up, route="carrier_pigeon"), headers=H(c)), 422, "invalid_route")
    err(c.post(f"{API}/batches", json=body(up, route=None), headers=H(c)), 422, "invalid_route")
    assert staging_exists(cfg, up)


def test_local_env_unavailable_and_engine(env, monkeypatch):
    cfg = env.cfg
    c = env.clients["alice"]
    up, _, _ = stage(cfg, env.uid("alice"), ["a.jpg"])
    # default hook: local environment not implemented -> 409
    err(c.post(f"{API}/batches", json=body(up, route="local_ocr", vlm_transcription_choice=None),
               headers=H(c)), 409, "local_env_unavailable")
    err(c.post(f"{API}/batches", json=body(up, vlm_transcription_choice="local_transcription"),
               headers=H(c)), 409, "local_env_unavailable")
    # vlm w/o local transcription does not need the local env
    assert c.post(f"{API}/batches", json=body(up), headers=H(c)).status_code == 201
    up, _, _ = stage(cfg, env.uid("alice"), ["a.jpg"])
    monkeypatch.setattr(hooks, "local_env_ok", lambda cfg: (True, None))      # looked up at call time
    err(c.post(f"{API}/batches", json=body(up, route="local_ocr", local_engine="tesseract"),
               headers=H(c)), 422, "invalid_local_engine")
    # local recognition needs an answer-sheet template
    err(c.post(f"{API}/batches", json=body(up, route="local_ocr", zero_shot_approved=True), headers=H(c)),
        422, "sheet_template_required")
    sid = make_sheet(cfg, env.uid("alice"))
    r = c.post(f"{API}/batches", json=body(up, route="local_ocr", vlm_transcription_choice="history_fewshot",
                                           zero_shot_approved=True, sheet_template_id=sid), headers=H(c))
    assert r.status_code == 201, r.text
    b = store.read_batch(cfg, r.json()["task_code"])
    assert b["local_engine"] == "paddle" and b["route"] == "local_ocr" and b["vlm_transcription_choice"] is None
    assert b["sheet_template"] == {"id": sid, "name": "IG 作文纸", "page_ids": ["pg_sheet_t1_0"]}
    assert c.get(f"{API}/batches/{b['task_code']}", headers=H(c)).json()["sheet_template"]["id"] == sid
    up, _, _ = stage(cfg, env.uid("alice"), ["a.jpg"])
    err(c.post(f"{API}/batches", json=body(up, vlm_transcription_choice="local_transcription", local_engine="hunyuan",
                                           zero_shot_approved=True, sheet_template_id=sid), headers=H(c)),
        422, "invalid_local_engine")
    r = c.post(f"{API}/batches", json=body(up, vlm_transcription_choice="local_transcription",
                                           zero_shot_approved=True, sheet_template_id=sid), headers=H(c))
    assert r.status_code == 201
    assert store.read_batch(cfg, r.json()["task_code"])["local_engine"] == "paddle"


def test_sheet_template_rules_at_batch_creation(env, monkeypatch):
    cfg = env.cfg
    c = env.clients["alice"]
    monkeypatch.setattr(hooks, "local_env_ok", lambda cfg: (True, None))
    up, _, _ = stage(cfg, env.uid("alice"), ["a.jpg"])
    lb = lambda **kw: body(up, route="local_ocr", zero_shot_approved=True, **kw)
    # vlm without local transcription needs no template
    vlm_up, _, _ = stage(cfg, env.uid("alice"), ["b.jpg"])
    assert c.post(f"{API}/batches", json=body(vlm_up), headers=H(c)).status_code == 201
    err(c.post(f"{API}/batches", json=lb(sheet_template_id="sheet_nope"), headers=H(c)), 422, "invalid_sheet_template")
    other = make_sheet(cfg, env.uid("bob"), tid="sheet_bob")                       # someone else's -> not visible
    err(c.post(f"{API}/batches", json=lb(sheet_template_id=other), headers=H(c)), 422, "invalid_sheet_template")
    pend = make_sheet(cfg, env.uid("alice"), tid="sheet_pend", status="pending")
    err(c.post(f"{API}/batches", json=lb(sheet_template_id=pend), headers=H(c)), 409, "sheet_template_not_ready")
    off = make_sheet(cfg, env.uid("alice"), tid="sheet_off", disabled=True)
    err(c.post(f"{API}/batches", json=lb(sheet_template_id=off), headers=H(c)), 409, "sheet_template_not_ready")
    shared = make_sheet(cfg, env.uid("bob"), tid="sheet_shared", shared=True)    # shared by root -> usable
    pre = c.post(f"{API}/batches/precheck", json=lb(sheet_template_id=pend), headers=H(c)).json()
    assert pre["sheet_template_ok"] is False and pre["sheet_template_reason"] == "sheet_template_not_ready"
    pre = c.post(f"{API}/batches/precheck", json=lb(sheet_template_id=shared), headers=H(c)).json()
    assert pre["sheet_template_ok"] is True
    r = c.post(f"{API}/batches", json=lb(sheet_template_id=shared), headers=H(c))
    assert r.status_code == 201, r.text


def test_third_party_ack(env, monkeypatch):
    cfg = env.cfg
    c = env.clients["alice"]
    seen = []
    monkeypatch.setattr(hooks, "needs_third_party_ack", lambda cfg, user, pid: seen.append(pid) or True)
    up, _, _ = stage(cfg, env.uid("alice"), ["a.jpg"])
    err(c.post(f"{API}/batches", json=body(up), headers=H(c)), 409, "third_party_ack_required")
    assert seen == ["p1"] and staging_exists(cfg, up)
    pc = c.post(f"{API}/batches/precheck", json=body(up), headers=H(c)).json()
    assert pc["needs_third_party_ack"] is True
    assert c.post(f"{API}/batches", json=body(up, third_party_ack=True), headers=H(c)).status_code == 201


def test_zero_shot_consent_flow(env, monkeypatch):
    cfg = env.cfg
    c = env.clients["alice"]
    up, _, _ = stage(cfg, env.uid("alice"), ["a.jpg", "b.jpg"])
    b = body(up, vlm_transcription_choice="history_fewshot")
    err(c.post(f"{API}/batches", json=b, headers=H(c)), 409, "zero_shot_consent_required")
    assert staging_exists(cfg, up) and len(list((cfg.staging_dir / up / "files").iterdir())) == 2
    r = c.post(f"{API}/batches", json={**b, "zero_shot_approved": True}, headers=H(c))   # SAME upload_id
    assert r.status_code == 201, r.text
    assert store.read_batch(cfg, r.json()["task_code"])["zero_shot_approved"] is True
    assert not staging_exists(cfg, up)
    # skip_fewshot records approval without asking
    up, _, _ = stage(cfg, env.uid("alice"), ["a.jpg"])
    r = c.post(f"{API}/batches", json=body(up, vlm_transcription_choice="skip_fewshot", zero_shot_approved=None),
               headers=H(c))
    assert r.status_code == 201
    sb = store.read_batch(cfg, r.json()["task_code"])
    assert sb["zero_shot_approved"] is True and sb["fewshot_ids"] == []
    # few-shot available: no consent needed, ids recorded, hook receives scope/owner/draft
    calls = []
    monkeypatch.setattr(hooks, "find_fewshot",
                        lambda cfg, owner, scope, draft: calls.append((owner, scope, draft["topic"])) or ["fs1", "fs2"])
    up, _, _ = stage(cfg, env.uid("alice"), ["a.jpg"])
    pc = c.post(f"{API}/batches/precheck", json=b | {"upload_id": up}, headers=H(c)).json()
    assert pc["fewshot_available"] and pc["fewshot_count"] == 2 and pc["zero_shot_consent_required"] is False
    r = c.post(f"{API}/batches", json=b | {"upload_id": up}, headers=H(c))
    assert r.status_code == 201
    fb = store.read_batch(cfg, r.json()["task_code"])
    assert fb["fewshot_ids"] == ["fs1", "fs2"] and fb["fewshot_scope"] == "own"
    assert calls[-1] == (env.uid("alice"), "own", "My summer")


def test_fewshot_scope_all_root_only(env, monkeypatch):
    cfg = env.cfg
    monkeypatch.setattr(hooks, "find_fewshot", lambda cfg, owner, scope, draft: ["x"])
    c = env.clients["alice"]
    up, _, _ = stage(cfg, env.uid("alice"), ["a.jpg"])
    err(c.post(f"{API}/batches", json=body(up, fewshot_scope="all"), headers=H(c)), 403, "forbidden")
    assert staging_exists(cfg, up)
    err(c.post(f"{API}/batches", json=body(up, fewshot_scope="weird"), headers=H(c)), 422, "invalid_fewshot_scope")
    rc = env.clients["rooty"]
    up, _, _ = stage(cfg, env.uid("rooty"), ["a.jpg"])
    r = rc.post(f"{API}/batches", json=body(up, fewshot_scope="all"), headers=H(rc))
    assert r.status_code == 201
    assert store.read_batch(cfg, r.json()["task_code"])["fewshot_scope"] == "all"


def test_staging_is_owner_only_even_for_root(env):
    cfg = env.cfg
    up, _, _ = stage(cfg, env.uid("alice"), ["a.jpg"])
    for who in ("bob", "rooty"):
        c = env.clients[who]
        err(c.post(f"{API}/batches", json=body(up), headers=H(c)), 404, "not_found")
        err(c.post(f"{API}/batches/precheck", json=body(up), headers=H(c)), 404, "not_found")
    assert staging_exists(cfg, up)
    c = env.clients["alice"]
    err(c.post(f"{API}/batches", json=body("up_doesnotexist"), headers=H(c)), 404, "not_found")
    err(c.post(f"{API}/batches", json=body("../../etc"), headers=H(c)), 404, "not_found")


def test_precheck_has_no_side_effects(env):
    cfg = env.cfg
    c = env.clients["alice"]
    up, _, _ = stage(cfg, env.uid("alice"), ["IMG_1.jpg", "IMG_2.jpg", "IMG_3.jpg"])
    b = body(up, grouping={"multi_photo": True, "mode": "per_n", "n": 2})
    b.pop("vlm_transcription_choice")        # precheck is lenient: UI checks before the user chose
    r = c.post(f"{API}/batches/precheck", json=b, headers=H(c))
    assert r.status_code == 200, r.text
    j = r.json()
    assert set(j) == {"fewshot_available", "fewshot_count", "needs_third_party_ack", "local_env_ok",
                      "local_env_reason", "zero_shot_consent_required", "grouping", "sheet_template_ok",
                      "sheet_template_reason"}
    assert j["local_env_ok"] is False and "not implemented" in j["local_env_reason"]
    assert j["zero_shot_consent_required"] is True and j["grouping"]["mode"] == "manual"   # 3 % 2 != 0
    assert staging_exists(cfg, up) and not (cfg.state_dir / "counters.json").exists()
    assert list(store.iter_batches(cfg)) == []


def test_originals_move_and_sha_verified(env):
    cfg = env.cfg
    up, man, blobs = stage(cfg, env.uid("alice"), ["x.jpg", "y.jpg"])
    sha_by_name = {f["orig_name"]: f["sha256"] for f in man["files"]}
    for f in man["files"]:
        assert (cfg.staging_dir / up / "files" / f"{f['file_id']}.{f['ext']}").exists()
    c = env.clients["alice"]
    tc = c.post(f"{API}/batches", json=body(up), headers=H(c)).json()["task_code"]
    for m in store.read_metas(cfg, tc):
        im = m["images"][0]
        assert im["sha256"] == sha_by_name[im["orig_name"]]
        assert hashlib.sha256((paths.subtask_dir(cfg, m["subtask_id"]) / im["original"]).read_bytes()).hexdigest() \
            == im["sha256"]


# --------------------------------------------------------------------------- numbering / crash recovery
def _direct_cfg(tmp_path):
    cfg = load_config(tmp_path / "data")
    return cfg, Principal("u_direct", "direct", "user", "user", "cookie")


def test_100_concurrent_creations_unique_codes(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    ups = [stage(cfg, user.user_id, ["a.jpg", "b.jpg"])[0] for _ in range(100)]

    def go(up):
        return create_mod.create_batch(cfg, user, body(up))

    with ThreadPoolExecutor(16) as pool:
        res = list(pool.map(go, ups))
    codes = [r["task_code"] for r in res]
    assert len(set(codes)) == 100
    assert sorted(codes) == [f"{clock.date_str()}{i:04d}" for i in range(1, 101)]
    for r in res:
        assert len(r["subtask_ids"]) == 2 and r["subtask_ids"][0] == r["task_code"] + "00001"
        assert store.counts(cfg, r["task_code"])["total"] == 2
    assert len(list(store.iter_batches(cfg))) == 100


class Crash(BaseException):
    pass


def test_crash_before_batch_json_is_recovered_and_number_not_reused(tmp_path, monkeypatch):
    cfg, user = _direct_cfg(tmp_path)
    ok = create_mod.create_batch(cfg, user, body(stage(cfg, user.user_id, ["a.jpg"])[0]))
    up, _, _ = stage(cfg, user.user_id, ["a.jpg", "b.jpg"])

    def boom(path, data):
        raise Crash()
    monkeypatch.setattr(create_mod, "_write_batch_json", boom)
    with pytest.raises(Crash):
        create_mod.create_batch(cfg, user, body(up))
    monkeypatch.undo()
    crashed = f"{clock.date_str()}0002"
    d = paths.batch_dir(cfg, crashed)
    assert d.is_dir() and not (d / "batch.json").exists()          # half-created
    assert (d / f"{crashed}00001" / "meta.json").exists()
    assert paths.find_batch_dir(cfg, crashed) is None
    out = recover.recover(cfg)                                       # startup recovery
    assert out["removed"] == [crashed] and not d.exists()
    assert paths.batch_dir(cfg, ok["task_code"]).is_dir()            # committed batch untouched
    nxt = create_mod.create_batch(cfg, user, body(stage(cfg, user.user_id, ["a.jpg"])[0]))
    assert nxt["task_code"] == f"{clock.date_str()}0003"             # 0002 never reused


def test_recovery_runs_on_executor_start(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    d = paths.batch_dir(cfg, f"{clock.date_str()}0042")
    (d / f"{d.name}00001").mkdir(parents=True)
    (d / f"{d.name}00001" / "meta.json").write_text("{}", "utf-8")

    async def go():
        e = ex_mod.Executor(cfg)
        await e.start()
        await e.stop()
    asyncio.run(go())
    assert not d.exists()


def test_failure_rolls_back_and_staging_reusable(tmp_path, monkeypatch):
    cfg, user = _direct_cfg(tmp_path)
    up, man, blobs = stage(cfg, user.user_id, ["a.jpg", "b.jpg"])

    def boom(path, data):
        raise RuntimeError("disk full")
    monkeypatch.setattr(create_mod, "_write_batch_json", boom)
    with pytest.raises(RuntimeError):
        create_mod.create_batch(cfg, user, body(up))
    monkeypatch.undo()
    assert not paths.batch_dir(cfg, f"{clock.date_str()}0001").exists()
    for f, blob in zip(man["files"], blobs):                        # originals back in staging
        assert (cfg.staging_dir / up / "files" / f"{f['file_id']}.{f['ext']}").read_bytes() == blob
    r = create_mod.create_batch(cfg, user, body(up))                 # same upload id works again
    assert r["task_code"] == f"{clock.date_str()}0002"


# --------------------------------------------------------------------------- access control, downloads
def test_cross_account_access_is_404_root_sees_all(env):
    tc = api_create(env, "alice", names=("a.jpg", "b.jpg"))
    sid = tc + "00001"
    b = env.clients["bob"]
    for path in ("", "/subtasks", "/events", f"/subtasks/{sid}/image/1", f"/subtasks/{sid}/image/1/thumb"):
        err(b.get(f"{API}/batches/{tc}{path}"), 404, "not_found")
    for a in ("pause", "resume", "cancel", "rerun-failed"):
        err(post_action(env, "bob", tc, a), 404, "not_found")
    assert status_of(env, "alice", tc)["status"] == "queued"         # untouched
    assert b.get(f"{API}/batches").json() == []
    assert b.get(f"{API}/batches?all=true").json() == []             # all=true is root only
    assert [x["task_code"] for x in env.clients["alice"].get(f"{API}/batches").json()] == [tc]
    r = env.clients["rooty"]
    assert r.get(f"{API}/batches/{tc}").json()["owner_id"] == env.uid("alice")
    assert r.get(f"{API}/batches/{tc}/subtasks").status_code == 200
    assert r.get(f"{API}/batches/{tc}/subtasks/{sid}/image/1").status_code == 200
    assert r.get(f"{API}/batches").json() == []                      # root's own list excludes others
    assert [x["task_code"] for x in r.get(f"{API}/batches?all=true").json()] == [tc]
    assert post_action(env, "rooty", tc, "pause").status_code == 200  # root may control
    assert TestClient(env.app).get(f"{API}/batches").status_code == 401
    err(b.get(f"{API}/batches/999999999999"), 404, "not_found")
    err(b.get(f"{API}/batches/abc"), 404, "not_found")


def test_list_detail_subtasks_shape(env):
    t1 = api_create(env, "alice", names=("a.jpg", "b.jpg"))
    t2 = api_create(env, "alice")
    lst = env.clients["alice"].get(f"{API}/batches").json()
    assert [x["task_code"] for x in lst] == [t2, t1]                 # newest first
    row = lst[1]
    assert set(row) == {"task_code", "status", "pause_reason", "topic", "genre", "route", "created_at",
                        "owner_id", "counts"}
    assert row["counts"] == {"queued": 2, "running": 0, "done": 0, "failed": 0, "api_error": 0,
                             "cancelled": 0, "total": 2}
    d = status_of(env, "alice", t1)
    assert d["topic"] == "My summer" and d["counts"]["total"] == 2 and d["owner_id"] == env.uid("alice")
    assert d["timeline"][0]["event"] == "created" and "rendered_prompt" not in json.dumps(d)
    subs = env.clients["alice"].get(f"{API}/batches/{t1}/subtasks").json()
    assert [s["subtask_id"] for s in subs] == [t1 + "00001", t1 + "00002"]
    assert subs[0] == {"subtask_id": t1 + "00001", "index": 1, "status": "queued", "fail_reason": None,
                       "review_status": "none", "photos": 1, "warnings": [], "ai_score": None,
                       "teacher_score": None, "adopted_ai": None, "review_outdated": False}


def test_original_and_thumb_download_and_traversal(env):
    cfg = env.cfg
    up, man, blobs = stage(cfg, env.uid("alice"), ["a.jpg", "b.jpg"])
    c = env.clients["alice"]
    tc = c.post(f"{API}/batches", json=body(up), headers=H(c)).json()["task_code"]
    sid = tc + "00002"
    r = c.get(f"{API}/batches/{tc}/subtasks/{sid}/image/1")
    assert r.status_code == 200 and r.content == blobs[1] and r.headers["content-type"] == "image/jpeg"
    assert "private" in r.headers["cache-control"]
    r = c.get(f"{API}/batches/{tc}/subtasks/{sid}/image/1/thumb")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    from PIL import Image
    assert max(Image.open(io.BytesIO(r.content)).size) <= 320
    assert (paths.subtask_dir(cfg, sid) / "image" / "derived" / "thumb_1.jpg").is_file()
    assert c.get(f"{API}/batches/{tc}/subtasks/{sid}/image/1/thumb").status_code == 200     # cached path
    bad = [f"{sid}/image/2", f"{sid}/image/0", f"{sid}/image/01", f"{sid}/image/x", f"{sid}/image/..%2F..%2Fmeta.json",
           f"{sid}/image/1%2F..%2F..%2Fmeta.json", f"..%2F{sid}/image/1", f"{sid}%2F..%2F{tc}00001/image/1",
           "00000000000000000/image/1", f"{tc[:11]}9{sid[12:]}/image/1", f"{sid}/image/1.jpg",
           f"{sid}/image/1/../../meta.json", f"../../../../{sid}/image/1"]
    for p in bad:
        for suffix in ("", "/thumb"):
            r = c.get(f"{API}/batches/{tc}/subtasks/{p}{suffix}")
            assert r.status_code == 404, (p, suffix, r.status_code)
            assert b"manifest" not in r.content and b"owner_id" not in r.content
    # tampered meta pointing outside the subtask dir is refused
    def evil(m):
        m["images"][0]["original"] = "../../batch.json"
    store.update_meta(cfg, sid, evil)
    assert c.get(f"{API}/batches/{tc}/subtasks/{sid}/image/1").status_code == 404
    assert c.get(f"{API}/batches/{tc}/subtasks/{sid}/image/1/thumb").status_code == 404
    # subtask belonging to a different batch is not reachable through this batch code
    other = api_create(env, "alice")
    assert c.get(f"{API}/batches/{tc}/subtasks/{other}00001/image/1").status_code == 404


# --------------------------------------------------------------------------- state machine
def force(env, tc, status, reason=None):
    def fn(b):
        b["status"] = status
        b["pause_reason"] = reason
    store.update_batch(env.cfg, tc, fn)


def test_state_machine_user_actions(env):
    tc = api_create(env, "alice", names=("a.jpg", "b.jpg"))
    inv = lambda r: err(r, 409, "invalid_transition")
    # queued: only pause / cancel
    inv(post_action(env, "alice", tc, "resume"))
    inv(post_action(env, "alice", tc, "rerun-failed"))
    r = post_action(env, "alice", tc, "pause")
    assert r.status_code == 200 and r.json()["status"] == "paused_user" and r.json()["pause_reason"] == "user"
    inv(post_action(env, "alice", tc, "pause"))
    r = post_action(env, "alice", tc, "resume")
    assert r.json()["status"] == "running" and r.json()["pause_reason"] is None
    inv(post_action(env, "alice", tc, "resume"))
    inv(post_action(env, "alice", tc, "rerun-failed"))                # running: not allowed
    assert post_action(env, "alice", tc, "pause").status_code == 200  # running -> paused_user
    for st in ("paused_circuit_breaker", "paused_api_errors"):
        force(env, tc, st, "x")
        inv(post_action(env, "alice", tc, "pause"))
        r = post_action(env, "alice", tc, "resume")
        assert r.status_code == 200 and r.json()["status"] == "running" and r.json()["pause_reason"] is None
    r = post_action(env, "alice", tc, "cancel")
    assert r.json()["status"] == "cancelled"
    assert sub_statuses(env, tc) == ["cancelled", "cancelled"]
    for a in ("pause", "resume", "cancel", "rerun-failed"):
        inv(post_action(env, "alice", tc, a))
    b = status_of(env, "alice", tc)
    ev = timeline_events(b)
    assert ev == ["created", "paused", "resumed", "paused", "resumed", "resumed", "cancelled"][:0] or \
        ev[0] == "created" and ev.count("paused") == 2 and ev.count("resumed") == 3 and ev[-1] == "cancelled"
    assert all(e["by"] == env.uid("alice") for e in b["timeline"][1:])
    assert all(any(t["event"] == "cancelled" for t in m["timeline"]) for m in store.read_metas(env.cfg, tc))


def test_cancel_from_every_state_and_completed_transitions(env):
    for st in ("queued", "running", "paused_user", "paused_circuit_breaker", "paused_api_errors", "completed"):
        tc = api_create(env, "alice")
        force(env, tc, st, "x" if st.startswith("paused") else None)
        assert post_action(env, "alice", tc, "cancel").json()["status"] == "cancelled", st
    # completed: pause/resume invalid
    tc = api_create(env, "alice")
    force(env, tc, "completed")
    err(post_action(env, "alice", tc, "pause"), 409, "invalid_transition")
    err(post_action(env, "alice", tc, "resume"), 409, "invalid_transition")
    err(post_action(env, "alice", tc, "rerun-failed"), 409, "nothing_to_rerun")


def test_rerun_failed_requeues_and_completed_to_running(env):
    tc = api_create(env, "alice", names=("a.jpg", "b.jpg", "c.jpg"))
    sids = store.list_subtask_ids(env.cfg, tc)
    for sid, st in zip(sids, ("done", "failed", "api_error")):
        store.update_meta(env.cfg, sid, lambda m, st=st: m.update(status=st, fail_reason="boom" if st != "done" else None))
    force(env, tc, "completed")
    r = post_action(env, "alice", tc, "rerun-failed")
    assert r.status_code == 200 and r.json()["status"] == "running"
    assert sub_statuses(env, tc) == ["done", "queued", "queued"]
    assert store.read_metas(env.cfg, tc)[1]["fail_reason"] is None
    assert "rerun_failed" in timeline_events(status_of(env, "alice", tc))
    # rerun from a paused batch also works
    store.update_meta(env.cfg, sids[1], lambda m: m.update(status="failed"))
    force(env, tc, "paused_user", "user")
    assert post_action(env, "alice", tc, "rerun-failed").json()["status"] == "running"
    assert sub_statuses(env, tc) == ["done", "queued", "queued"]


# --------------------------------------------------------------------------- executor
class Gate:
    """Cross-thread gate usable from the app's event loop."""
    def __init__(self):
        self.ev = threading.Event()

    async def wait(self):
        while not self.ev.is_set():
            await asyncio.sleep(0.01)


def test_no_processor_nothing_dispatched(env):
    tc = api_create(env, "alice", names=("a.jpg", "b.jpg"))
    time.sleep(0.5)
    assert status_of(env, "alice", tc)["status"] == "queued"
    assert sub_statuses(env, tc) == ["queued", "queued"]


def test_executor_concurrency_limit_and_completion_and_outcomes(env2):
    state = {"active": 0, "max": 0, "seen": []}
    finished = []
    ex_mod.on_batch_finished.append(lambda cfg, tc: finished.append(tc))

    async def proc(ctx):
        state["active"] += 1
        state["max"] = max(state["max"], state["active"])
        state["seen"].append((ctx.subtask_id, ctx.config_rev, ctx.meta["status"], ctx.batch["task_code"]))
        await asyncio.sleep(0.08)
        state["active"] -= 1
        i = int(ctx.subtask_id[12:])
        if i == 1:
            return ex_mod.Outcome("done", warnings=["low contrast"])
        if i == 2:
            return ex_mod.Outcome("failed", "unreadable")
        if i == 3:
            return ex_mod.Outcome("api_error", "provider 503")
        if i == 4:
            raise RuntimeError("secret essay text must not be logged")
        if i == 5:
            return "not an outcome"
        return ex_mod.Outcome("done")

    ex_mod.set_processor(proc)
    tc = api_create(env2, "alice", names=tuple(f"p{i}.jpg" for i in range(7)))
    wait_for(lambda: status_of(env2, "alice", tc)["status"] == "completed")
    assert state["max"] == 2                                          # settings.concurrency honoured
    assert len(state["seen"]) == 7 and all(s[1] == 1 and s[2] == "running" for s in state["seen"])
    metas = store.read_metas(env2.cfg, tc)
    assert [m["status"] for m in metas] == ["done", "failed", "api_error", "failed", "failed", "done", "done"]
    assert metas[0]["warnings"] == ["low contrast"] and metas[1]["fail_reason"] == "unreadable"
    assert metas[2]["fail_reason"] == "provider 503"
    assert metas[3]["fail_reason"] == "internal_error" and metas[4]["fail_reason"] == "internal_error"
    assert all(m["attempts"] == 1 and m["config_rev"] == 1 for m in metas)
    assert [e["event"] for e in metas[0]["timeline"]] == ["created", "started", "done"]
    d = status_of(env2, "alice", tc)
    assert d["counts"] == {"queued": 0, "running": 0, "done": 3, "failed": 3, "api_error": 1,
                           "cancelled": 0, "total": 7}
    ev = timeline_events(d)
    assert ev[:3] == ["created", "started", "completed"]
    wait_for(lambda: finished == [tc])               # hooks run right after the status change
    # secrets not logged
    logs = "\n".join(f.read_text("utf-8") for f in (env2.data / "logs" / "app").glob("app.log*"))
    assert "secret essay text" not in logs
    # completed -> running via rerun-failed; only failed/api_error rerun
    ex_mod.set_processor(lambda ctx: _ok(ctx))
    assert post_action(env2, "alice", tc, "rerun-failed").json()["status"] == "running"
    wait_for(lambda: status_of(env2, "alice", tc)["status"] == "completed")
    assert set(sub_statuses(env2, tc)) == {"done"}
    assert [m["attempts"] for m in store.read_metas(env2.cfg, tc)] == [1, 2, 2, 2, 2, 1, 1]
    wait_for(lambda: finished == [tc, tc])


async def _ok(ctx):
    return ex_mod.Outcome("done")


def test_round_robin_across_batches(env1):
    order = []

    async def proc(ctx):
        order.append(ctx.batch["task_code"])
        await asyncio.sleep(0.03)
        return ex_mod.Outcome("done")

    t1 = api_create(env1, "alice", names=("a.jpg",) * 3)
    t2 = api_create(env1, "bob", names=("a.jpg",) * 3)
    ex_mod.set_processor(proc)
    wait_for(lambda: status_of(env1, "alice", t1)["status"] == "completed"
             and status_of(env1, "bob", t2)["status"] == "completed")
    assert order[:4] == [t1, t2, t1, t2]


def test_pause_lets_in_flight_finish_then_resume(env1):
    gate = Gate()
    started = []

    async def proc(ctx):
        started.append(ctx.subtask_id)
        if len(started) == 1:
            await gate.wait()
        return ex_mod.Outcome("done")

    ex_mod.set_processor(proc)
    tc = api_create(env1, "alice", names=("a.jpg", "b.jpg", "c.jpg"))
    wait_for(lambda: len(started) == 1)
    assert post_action(env1, "alice", tc, "pause").json()["status"] == "paused_user"
    gate.ev.set()
    wait_for(lambda: sub_statuses(env1, tc)[0] == "done")             # in-flight saved
    time.sleep(0.4)
    assert len(started) == 1 and sub_statuses(env1, tc) == ["done", "queued", "queued"]
    assert status_of(env1, "alice", tc)["status"] == "paused_user"
    assert post_action(env1, "alice", tc, "resume").json()["status"] == "running"
    wait_for(lambda: status_of(env1, "alice", tc)["status"] == "completed")
    assert sub_statuses(env1, tc) == ["done"] * 3


def test_cancel_aborts_in_flight(env1):
    state = {"cancelled": False}
    started = []

    async def proc(ctx):
        started.append(ctx.subtask_id)
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        return ex_mod.Outcome("done")

    ex_mod.set_processor(proc)
    tc = api_create(env1, "alice", names=("a.jpg", "b.jpg"))
    wait_for(lambda: len(started) == 1)
    assert post_action(env1, "alice", tc, "cancel").json()["status"] == "cancelled"
    wait_for(lambda: state["cancelled"])
    assert sub_statuses(env1, tc) == ["cancelled", "cancelled"]
    time.sleep(0.3)
    assert len(started) == 1 and status_of(env1, "alice", tc)["status"] == "cancelled"
    assert env1.app.state.executor._inflight == {}


def test_executor_pause_api_for_m3(env1):
    gate = Gate()
    started = []

    async def proc(ctx):
        started.append(1)
        await gate.wait()
        return ex_mod.Outcome("done")

    ex_mod.set_processor(proc)
    tc = api_create(env1, "alice", names=("a.jpg", "b.jpg"))
    wait_for(lambda: len(started) == 1)
    e = env1.app.state.executor
    fut = asyncio.run_coroutine_threadsafe(e.pause(tc, "paused_circuit_breaker", "too many failures"), e._loop)
    assert fut.result(5) is True
    d = status_of(env1, "alice", tc)
    assert d["status"] == "paused_circuit_breaker" and d["pause_reason"] == "too many failures"
    fut = asyncio.run_coroutine_threadsafe(e.pause(tc, "paused_user", "again"), e._loop)
    assert fut.result(5) is False                                     # already paused
    gate.ev.set()
    wait_for(lambda: sub_statuses(env1, tc)[0] == "done")
    time.sleep(0.3)
    assert sub_statuses(env1, tc)[1] == "queued"


# --------------------------------------------------------------------------- restart
def test_restart_recovery_with_fresh_executor(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    cfg.settings["concurrency"] = 4
    mk = lambda n: create_mod.create_batch(cfg, user, body(stage(cfg, user.user_id, [f"{i}.jpg" for i in range(n)])[0]))
    run, paused, cancelled = mk(3), mk(2), mk(1)
    # simulate a crash: subtask 1 of each batch was `running`
    for r in (run, paused, cancelled):
        store.update_meta(cfg, r["subtask_ids"][0], lambda m: m.update(status="running", attempts=1))
    store.update_batch(cfg, run["task_code"], lambda b: b.update(status="running"))
    store.update_batch(cfg, paused["task_code"], lambda b: b.update(status="paused_user", pause_reason="user"))
    store.update_batch(cfg, cancelled["task_code"], lambda b: b.update(status="cancelled"))
    processed = []

    async def proc(ctx):
        processed.append(ctx.subtask_id)
        return ex_mod.Outcome("done")

    async def go():
        ex_mod.Executor.TICK_S = 0.05
        ex_mod.set_processor(proc)
        e = ex_mod.Executor(cfg)                                      # "restart": fresh executor, same data dir
        await e.start()
        for _ in range(200):
            if store.read_batch(cfg, run["task_code"])["status"] == "completed":
                break
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.3)
        await e.stop()
    try:
        asyncio.run(go())
    finally:
        ex_mod.Executor.TICK_S = 0.5
    assert sorted(processed) == sorted(run["subtask_ids"])            # running one was requeued and re-run
    assert store.read_batch(cfg, run["task_code"])["status"] == "completed"
    assert [m["attempts"] for m in store.read_metas(cfg, run["task_code"])] == [2, 1, 1]
    pb = store.read_batch(cfg, paused["task_code"])
    assert pb["status"] == "paused_user" and pb["pause_reason"] == "user"      # stays paused
    assert sub_statuses(type("E", (), {"cfg": cfg}), paused["task_code"]) == ["queued", "queued"]
    assert any(e["event"] == "recovered_after_restart" for e in store.read_metas(cfg, paused["task_code"])[0]["timeline"])
    assert store.read_batch(cfg, cancelled["task_code"])["status"] == "cancelled"
    assert store.read_metas(cfg, cancelled["task_code"])[0]["status"] == "cancelled"


def test_shutdown_requeues_in_flight(tmp_path):
    cfg, user = _direct_cfg(tmp_path)
    r = create_mod.create_batch(cfg, user, body(stage(cfg, user.user_id, ["a.jpg"])[0]))
    started = []

    async def proc(ctx):
        started.append(1)
        await asyncio.sleep(30)

    async def go():
        ex_mod.Executor.TICK_S = 0.05
        ex_mod.set_processor(proc)
        e = ex_mod.Executor(cfg)
        await e.start()
        for _ in range(200):
            if started:
                break
            await asyncio.sleep(0.02)
        await e.stop()
    try:
        asyncio.run(go())
    finally:
        ex_mod.Executor.TICK_S = 0.5
    assert store.read_metas(cfg, r["task_code"])[0]["status"] == "queued"


# --------------------------------------------------------------------------- owner disabled
def test_owner_disabled_pauses_running_batches(env1):
    gate = Gate()
    started = []

    async def proc(ctx):
        started.append(ctx.subtask_id)
        await gate.wait()
        return ex_mod.Outcome("done")

    ex_mod.set_processor(proc)
    ta = api_create(env1, "alice", names=("a.jpg", "b.jpg"))
    tb = api_create(env1, "bob", names=("a.jpg",))
    wait_for(lambda: status_of(env1, "alice", ta)["status"] == "running")
    ex_mod.set_processor(None)                       # keep bob's batch queued
    r = env1.clients["rooty"].post(f"{API}/admin/users/{env1.uid('alice')}/disable", headers=H(env1.clients["rooty"]))
    assert r.status_code == 200, r.text
    b = store.read_batch(env1.cfg, ta)
    assert b["status"] == "paused_user" and b["pause_reason"] == "owner_disabled"
    assert b["timeline"][-1]["event"] == "paused" and b["timeline"][-1]["by"] == "system"
    assert store.read_batch(env1.cfg, tb)["status"] in ("queued", "running")      # other owners unaffected
    gate.ev.set()
    wait_for(lambda: sub_statuses(env1, ta)[0] == "done")                          # in-flight finished
    rc = env1.clients["rooty"]
    assert rc.post(f"{API}/admin/users/{env1.uid('alice')}/enable", headers=H(rc)).status_code == 200
    time.sleep(0.4)
    b = store.read_batch(env1.cfg, ta)
    assert b["status"] == "paused_user" and b["pause_reason"] == "owner_disabled"  # stays paused
    assert sub_statuses(env1, ta)[1] == "queued"
    assert env1.clients["rooty"].post(f"{API}/batches/{ta}/resume", headers=H(rc)).json()["status"] == "running"


def test_owner_disabled_pauses_queued_batches_too(env):
    ta = api_create(env, "alice")
    rc = env.clients["rooty"]
    assert rc.post(f"{API}/admin/users/{env.uid('alice')}/disable", headers=H(rc)).status_code == 200
    b = store.read_batch(env.cfg, ta)
    assert b["status"] == "paused_user" and b["pause_reason"] == "owner_disabled"


# --------------------------------------------------------------------------- SSE
def test_sse_progress_stream_emits_on_change(env):
    tc = api_create(env, "alice", names=("a.jpg", "b.jpg"))
    cfg = env.cfg

    async def go():
        async def never():
            return False
        gen = batches_api.progress_stream(cfg, tc, never, poll_s=0.01, keepalive_s=0.05)
        first = await asyncio.wait_for(gen.__anext__(), 2)
        assert first.startswith("event: progress\ndata: ")
        d1 = json.loads(first.split("data: ", 1)[1])
        assert d1["task_code"] == tc and d1["status"] == "queued" and d1["counts"]["total"] == 2
        # nothing changes -> only keepalive comments
        ka = await asyncio.wait_for(gen.__anext__(), 2)
        assert ka.startswith(":")
        # change: pause the batch, then finish a subtask
        await asyncio.to_thread(store.user_action, cfg, tc, "pause", "x")
        ev = None
        for _ in range(20):
            chunk = await asyncio.wait_for(gen.__anext__(), 2)
            if chunk.startswith("event: progress"):
                ev = json.loads(chunk.split("data: ", 1)[1])
                break
        assert ev and ev["status"] == "paused_user" and ev["pause_reason"] == "user"
        await asyncio.to_thread(store.update_meta, cfg, tc + "00001", lambda m: m.update(status="done"))
        for _ in range(20):
            chunk = await asyncio.wait_for(gen.__anext__(), 2)
            if chunk.startswith("event: progress"):
                ev = json.loads(chunk.split("data: ", 1)[1])
                break
        assert ev["counts"]["done"] == 1 and ev["counts"]["queued"] == 1
        await gen.aclose()

        # stops when the client disconnects
        flag = {"gone": False}

        async def gone():
            return flag["gone"]
        gen = batches_api.progress_stream(cfg, tc, gone, poll_s=0.01)
        await gen.__anext__()
        flag["gone"] = True
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(gen.__anext__(), 2)
    asyncio.run(go())


def test_sse_endpoint_route_registered_and_content_type(env):
    tc = api_create(env, "alice")
    assert TestClient(env.app).get(f"{API}/batches/{tc}/events").status_code == 401
    # The endpoint itself returns a StreamingResponse with the SSE media type.
    from fastapi.responses import StreamingResponse
    from starlette.requests import Request
    cfg = env.cfg
    scope = {"type": "http", "method": "GET", "headers": [], "app": env.app, "path": "/", "query_string": b""}
    p = env.principal("alice")
    resp = batches_api.events(tc, Request(scope), p)
    assert isinstance(resp, StreamingResponse) and resp.media_type == "text/event-stream"
    with pytest.raises(AppError):
        batches_api.events(tc, Request(scope), env.principal("bob"))
