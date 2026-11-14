"""M5: teacher review (review.json), summary result-<task_code>.json, request body limits, and the end-to-end
acceptance flow on the real grading pipeline (mock LLM over HTTP, fake local OCR engine, no real key)."""
from __future__ import annotations

import io
import json
import time

import pytest
from fastapi.testclient import TestClient

from app.batches import executor as ex_mod
from app.batches import hooks, paths, results, store
from app.core import clock
from app.main import create_app
from app.ocr import service
from app.ocr.common import Line, PageResult
from tests.fixtures_images import make_image
from tests.mock_llm_server import CRITERIA, SCORE, mock_llm  # noqa: F401
from tests.sheet_fixtures import make_sheet
from tests.test_batches import API, Env, H, _isolate, _unbind_logging, wait_for  # noqa: F401

T_TEXT = "Grade the essay. Topic {{TOPIC}} Rubric {{RUBRIC}} Genre {{GENRE}} Full {{FULL_SCORE}}\n{{FEWSHOT}}"
T_IMAGE = "Grade the photographed essay. {{TOPIC}} {{RUBRIC}} {{GENRE}} {{FULL_SCORE}}\n{{FEWSHOT}}"


class FakeEngine:
    """Local OCR stand-in: one essay per page, text chosen by image width."""
    def __init__(self):
        self.calls = 0

    def recognize(self, im):
        self.calls += 1
        lines = [Line("Dear Tom,", [100, 100, 400, 140], 0.99),
                 Line(f"I recieve your leter number {im.size[0]} and I am hapy.", [100, 200, 1500, 240], 0.98),
                 Line("Best wishes, Lily", [100, 300, 600, 340], 0.99)]
        return PageResult(lines=lines, width=im.size[0], height=im.size[1], layout_reliable=True, boxes=lines)

    def info(self):
        return {"local_engine": "paddle", "name": "fake", "version": {}, "params": {}}


@pytest.fixture
def app_env(tmp_path, mock_llm, monkeypatch):  # noqa: F811
    """Real app with the real grading pipeline; users rooty (root), alice, bob."""
    clock.set_now_fn(None)
    _unbind_logging()
    data = tmp_path / "data"
    app = create_app(data, enable_grading=True)
    old_tick = ex_mod.Executor.TICK_S
    ex_mod.Executor.TICK_S = 0.05
    ctx = TestClient(app)
    ctx.__enter__()
    eng = FakeEngine()
    monkeypatch.setattr(service, "_get_engine", lambda cfg, name: eng)
    monkeypatch.setattr(service, "engine_status", lambda cfg, name: (True, None))
    monkeypatch.setattr(hooks, "local_env_ok", lambda cfg: (True, None))
    e = Env(data, app)
    for name, kind in (("rooty", "root"), ("alice", "user"), ("bob", "user")):
        e.add_user(name, kind)
    e.mock, e.engine = mock_llm, eng
    yield e
    ctx.__exit__(None, None, None)
    ex_mod.Executor.TICK_S = old_tick
    _unbind_logging()


def setup_account(e, who: str, *, mode="text"):
    c = e.clients[who]
    prov = c.post(f"{API}/providers", json={"alias": f"mock-{who}", "base_url": e.mock.base_url,
                                            "api_key": "sk-test-0000-1111", "default_model": "m1"}, headers=H(c))
    assert prov.status_code == 201, prov.text
    tpl = c.post(f"{API}/prompts", json={"name": f"tpl-{mode}", "input_mode": mode,
                                         "text": T_TEXT if mode == "text" else T_IMAGE}, headers=H(c))
    assert tpl.status_code == 201, tpl.text
    sheet = make_sheet(e.cfg, e.uid(who), tid=f"sheet_{who}")
    return prov.json()["id"], tpl.json()["id"], sheet


def upload(c, n: int, base_w: int = 300) -> str:
    files = [("files", (f"p{i + 1}.jpg", io.BytesIO(make_image("jpeg", size=(base_w + i, 200))), "image/jpeg"))
             for i in range(n)]
    r = c.post(f"{API}/uploads", files=files, data={"client_last_modified": [""] * n}, headers=H(c))
    assert r.status_code == 200, r.text
    return r.json()["upload_id"]


def create_batch(e, who, ids, n=2, route="local_ocr", base_w=300, **over) -> str:
    prov, tpl, sheet = ids
    c = e.clients[who]
    body = {"upload_id": upload(c, n, base_w), "topic": "A letter to Tom", "rubric": "content 10, language 10",
            "genre": "letter", "full_score": 20, "route": route, "provider_id": prov, "model": "m1",
            "template_id": tpl, "template_version": None, "grouping": {"multi_photo": False},
            "vlm_transcription_choice": None if route == "local_ocr" else "skip_fewshot",
            "sheet_template_id": sheet if route == "local_ocr" else None}
    body.update(over)
    r = c.post(f"{API}/batches", json=body, headers=H(c))
    assert r.status_code == 201, r.text
    return r.json()["task_code"]


def wait_done(e, who, tc):
    c = e.clients[who]
    wait_for(lambda: c.get(f"{API}/batches/{tc}", headers=H(c)).json()["status"] == "completed", timeout=30)
    results.wait_idle()


def subs(e, who, tc):
    c = e.clients[who]
    return c.get(f"{API}/batches/{tc}/subtasks", headers=H(c)).json()


def review(e, who, tc, sid, body, raw: bytes | None = None):
    c = e.clients[who]
    url = f"{API}/batches/{tc}/subtasks/{sid}/review"
    if raw is not None:
        return c.post(url, content=raw, headers={**H(c), "Content-Type": "application/json"})
    return c.post(url, json=body, headers=H(c))


def err(r, status, code):
    assert r.status_code == status, r.text
    assert r.json()["code"] == code, r.json()
    return r.json()


def backups(e, tc):
    d = results.backup_dir(e.cfg, tc)
    return sorted(d.glob("*.json")) if d.is_dir() else []


# ------------------------------------------------------------------ end-to-end acceptance
def test_e2e_review_result_and_fewshot_from_reviews(app_env):
    e = app_env
    a, b = e.clients["alice"], e.clients["bob"]
    ids_a = setup_account(e, "alice")
    # batch A: local OCR route, two essays, the second one fails at the provider (C class -> no retry)
    e.mock.script(lambda req, n: "http:400" if "number 301" in json.dumps(req["body"]) else "ok_json")
    tc = create_batch(e, "alice", ids_a, n=2, zero_shot_approved=True)
    wait_done(e, "alice", tc)
    s1, s2 = sorted(subs(e, "alice", tc), key=lambda s: s["index"])
    assert s1["status"] == "done" and s1["ai_score"] == SCORE and s1["teacher_score"] is None
    assert s2["status"] == "failed" and s2["ai_score"] is None
    # summary generated on completion
    res = a.get(f"{API}/batches/{tc}/result", headers=H(a)).json()
    assert res["trigger"] == "completed" and res["counts"] == {"done": 0, "pending_review": 1, "failed": 1,
                                                                "cancelled": 0, "total": 2}
    r1 = res["subtasks"][0]
    assert r1["status"] == "pending_review" and r1["final_score"] == SCORE and r1["score_source"] == "ai"
    bdir = paths.batch_dir(e.cfg, tc)
    for rel in [r1["files"]["meta"], r1["files"]["ocr"], r1["files"]["grading"], *r1["files"]["images"],
                *r1["files"]["llm"]]:
        assert (bdir / rel).is_file(), rel
    assert res["batch"]["topic"] == "A letter to Tom" and res["batch"]["full_score"] == 20
    assert backups(e, tc) == []
    sid1, sid2 = s1["subtask_id"], s2["subtask_id"]
    grading_before = (bdir / sid1 / "grading.json").read_bytes()

    # one-click adopt
    r = review(e, "alice", tc, sid1, {"adopt_ai": True})
    assert r.status_code == 200, r.text
    rev = r.json()["revision"]
    assert rev["rev"] == 1 and rev["adopted_ai"] is True and rev["final_score"] == SCORE and rev["items"] is None
    assert rev["ai_score"] == SCORE and rev["diff"] == 0 and rev["teacher_id"] == e.uid("alice")
    assert rev["ai_attempt"] == 1
    res = a.get(f"{API}/batches/{tc}/result", headers=H(a)).json()
    assert res["trigger"] == "review" and res["counts"]["done"] == 1 and len(backups(e, tc)) == 1
    # override with items (append-only: revision 1 untouched, AI files untouched)
    items = [{"name": "内容", "content": "TEACHER-ITEM-XYZ 要点齐全"}, {"name": "", "content": "语言有错误"}]
    r = review(e, "alice", tc, sid1, {"final_score": 15.5, "items": items})
    assert r.status_code == 200, r.text
    doc = r.json()["review"]
    assert [x["rev"] for x in doc["revisions"]] == [1, 2]
    assert doc["revisions"][0] == rev
    assert doc["revisions"][1]["diff"] == 15.5 - SCORE and doc["revisions"][1]["items"] == items
    assert (bdir / sid1 / "grading.json").read_bytes() == grading_before
    on_disk = json.loads((bdir / sid1 / "review.json").read_text("utf-8"))
    assert on_disk == doc and on_disk["revisions"][0] == rev
    # failed essay: hand scoring (no items -> score only, diff null); adopting is impossible
    err(review(e, "alice", tc, sid2, {"adopt_ai": True}), 409, "no_ai_result")
    r = review(e, "alice", tc, sid2, {"final_score": 12})
    assert r.status_code == 200 and r.json()["revision"]["diff"] is None and r.json()["revision"]["items"] is None
    res = a.get(f"{API}/batches/{tc}/result", headers=H(a)).json()
    assert res["counts"] == {"done": 2, "pending_review": 0, "failed": 0, "cancelled": 0, "total": 2}
    x1, x2 = res["subtasks"]
    assert (x1["final_score"], x1["score_source"], x1["teacher_score"], x1["ai_score"], x1["diff"]) == \
        (15.5, "teacher", 15.5, SCORE, 15.5 - SCORE)
    assert (x2["final_score"], x2["score_source"], x2["ai_score"], x2["diff"]) == (12, "teacher", None, None)
    assert res["stats"]["reviewed"] == 2 and res["stats"]["mean_abs_diff"] == 15.5 - SCORE
    assert len(backups(e, tc)) == 3
    assert subs(e, "alice", tc)[0]["review_status"] == "reviewed"
    # validation
    err(review(e, "alice", tc, sid1, {"final_score": 20.5}), 422, "score_out_of_range")
    err(review(e, "alice", tc, sid1, {"final_score": -1}), 422, "score_out_of_range")
    err(review(e, "alice", tc, sid1, {"items": items}), 422, "final_score_required")
    err(review(e, "alice", tc, sid1, {"final_score": True}), 422, "final_score_required")
    err(review(e, "alice", tc, sid1, {"adopt_ai": True, "items": items}), 422, "adopt_with_items")
    err(review(e, "alice", tc, sid1, {"final_score": 3, "items": [{"name": "x", "content": "  "}]}), 422, "item_empty")
    err(review(e, "alice", tc, sid1, {"final_score": 3, "items": "text"}), 422, "invalid_items")
    assert len(json.loads((bdir / sid1 / "review.json").read_text("utf-8"))["revisions"]) == 2
    # manual regeneration
    assert a.post(f"{API}/batches/{tc}/result/regenerate", headers=H(a)).json()["trigger"] == "manual"
    assert len(backups(e, tc)) == 4
    audit = "".join(p.read_text("utf-8") for p in e.cfg.audit_dir.glob("*.jsonl"))
    assert audit.count('"review.submit"') == 3 and '"result.regenerate"' in audit
    assert "TEACHER-ITEM-XYZ" not in audit

    # few-shot from the reviews: batch B of alice uses them; bob never sees them
    pre = a.post(f"{API}/batches/precheck", json={"upload_id": upload(a, 1), "topic": "A letter to Tom",
                                                  "rubric": "content 10, language 10", "genre": "letter",
                                                  "full_score": 20, "route": "local_ocr", "provider_id": ids_a[0],
                                                  "model": "m1", "template_id": ids_a[1],
                                                  "sheet_template_id": ids_a[2],
                                                  "grouping": {"multi_photo": False}}, headers=H(a)).json()
    assert pre["fewshot_available"] is True and pre["fewshot_count"] == 2
    e.mock.script(lambda req, n: "ok_json")
    tc_b = create_batch(e, "alice", ids_a, n=1, base_w=500)
    wait_done(e, "alice", tc_b)
    bb = store.read_batch(e.cfg, tc_b)
    assert sorted(bb["fewshot_ids"]) == sorted([sid1, sid2])
    sid_b = store.list_subtask_ids(e.cfg, tc_b)[0]
    req = json.loads((paths.subtask_dir(e.cfg, sid_b) / "llm" / "001.request.json").read_text("utf-8"))
    system = req["messages"][0]["content"]
    assert "TEACHER-ITEM-XYZ" in system and '"判分": 15.5' in system and '"判分": 12' in system
    assert "number 300" in system                                   # the shot's transcription
    assert req["fewshot_ids"] == bb["fewshot_ids"] and req["config_rev"] == 1 and req["model"] == "m1"
    assert req["provider_alias"] == "mock-alice" and req["template"]["id"] == ids_a[1]
    resp = json.loads((paths.subtask_dir(e.cfg, sid_b) / "llm" / "001.response.json").read_text("utf-8"))
    assert resp.get("latency_ms") is not None and resp.get("usage") is not None
    # bob: no access to alice's few-shot records
    ids_b = setup_account(e, "bob")
    pre_b = b.post(f"{API}/batches/precheck", json={"upload_id": upload(b, 1), "topic": "A letter to Tom",
                                                    "rubric": "content 10, language 10", "genre": "letter",
                                                    "full_score": 20, "route": "local_ocr", "provider_id": ids_b[0],
                                                    "model": "m1", "template_id": ids_b[1],
                                                    "sheet_template_id": ids_b[2],
                                                    "grouping": {"multi_photo": False}}, headers=H(b)).json()
    assert pre_b["fewshot_count"] == 0 and pre_b["zero_shot_consent_required"] is True
    tc_bob = create_batch(e, "bob", ids_b, n=1, base_w=700, zero_shot_approved=True)
    wait_done(e, "bob", tc_bob)
    assert store.read_batch(e.cfg, tc_bob)["fewshot_ids"] == []
    sid_bob = store.list_subtask_ids(e.cfg, tc_bob)[0]
    req_bob = (paths.subtask_dir(e.cfg, sid_bob) / "llm" / "001.request.json").read_text("utf-8")
    assert "number 700" in req_bob
    assert "TEACHER-ITEM-XYZ" not in req_bob
    assert "number 300" not in req_bob and "number 301" not in req_bob
    # bob cannot review / read alice's essays or summary
    err(review(e, "bob", tc, sid1, {"final_score": 1}), 404, "not_found")
    err(b.get(f"{API}/batches/{tc}/result", headers=H(b)), 404, "not_found")
    # root can review anyone's essay (teacher_id = root)
    root = e.clients["rooty"]
    r = root.post(f"{API}/batches/{tc}/subtasks/{sid2}/review", json={"final_score": 11}, headers=H(root))
    assert r.status_code == 200 and r.json()["revision"]["teacher_id"] == e.uid("rooty")


def test_review_states_cancel_summary_and_backup_rotation(app_env, monkeypatch):
    e = app_env
    a = e.clients["alice"]
    ids = setup_account(e, "alice", mode="image")
    e.mock.script(lambda req, n: "delay:5")          # keeps the essays in flight
    tc = create_batch(e, "alice", ids, n=2, route="vlm")
    sid = store.list_subtask_ids(e.cfg, tc)[0]
    wait_for(lambda: any(s["status"] == "running" for s in subs(e, "alice", tc)))
    err(a.get(f"{API}/batches/{tc}/result", headers=H(a)), 409, "result_not_ready")
    err(a.post(f"{API}/batches/{tc}/result/regenerate", headers=H(a)), 409, "result_not_ready")
    st = {s["subtask_id"]: s["status"] for s in subs(e, "alice", tc)}
    err(review(e, "alice", tc, sid, {"final_score": 3}), 409, "review_not_allowed")
    assert st[sid] in ("queued", "running")
    # cancel -> summary generated immediately, everything cancelled
    assert a.post(f"{API}/batches/{tc}/cancel", headers=H(a)).status_code == 200
    res = json.loads(results.result_path(e.cfg, tc).read_text("utf-8"))
    assert res["trigger"] == "cancelled" and res["counts"]["cancelled"] == 2
    err(review(e, "alice", tc, sid, {"final_score": 3}), 409, "review_not_allowed")
    # backups keep only the newest N
    e.cfg.settings["result_backups_keep"] = 3
    for _ in range(6):
        assert a.post(f"{API}/batches/{tc}/result/regenerate", headers=H(a)).status_code == 200
    assert len(backups(e, tc)) == 3
    # a missing summary of a finished batch is generated on demand
    results.result_path(e.cfg, tc).unlink()
    assert a.get(f"{API}/batches/{tc}/result", headers=H(a)).json()["trigger"] == "on_demand"


# ------------------------------------------------------------------ limits (code points, body size)
def _done_essay(e):
    ids = setup_account(e, "alice", mode="image")
    e.mock.script(lambda req, n: "ok_json")
    tc = create_batch(e, "alice", ids, n=1, route="vlm")
    wait_done(e, "alice", tc)
    return tc, store.list_subtask_ids(e.cfg, tc)[0]


def test_review_item_limits_count_code_points(app_env):
    e = app_env
    tc, sid = _done_essay(e)
    zh = "中" * 10000
    emoji = "👍😀" * 2500 + "评" * 5000            # 10000 code points, 15000 UTF-16 units (JS length)
    assert len(emoji) == 10000
    items = [{"name": "名" * 100, "content": zh}] * 199 + [{"name": "表情", "content": emoji}]
    raw = json.dumps({"final_score": 10, "items": items}, ensure_ascii=False).encode("utf-8")
    assert 5.5e6 < len(raw) < 8 * 1024 * 1024             # ~6 MB of UTF-8 passes the 8 MB review limit
    r = review(e, "alice", tc, sid, None, raw=raw)
    assert r.status_code == 200, r.text[:300]
    saved = json.loads((paths.subtask_dir(e.cfg, sid) / "review.json").read_text("utf-8"))["revisions"][-1]
    assert len(saved["items"]) == 200 and saved["items"][-1]["content"] == emoji
    # one over each limit
    too_many = json.dumps({"final_score": 10, "items": [{"name": "a", "content": "b"}] * 201}).encode()
    j = err(review(e, "alice", tc, sid, None, raw=too_many), 422, "too_many_items")
    assert j["detail"] == {"max": 200, "got": 201}
    long1 = json.dumps({"final_score": 10, "items": [{"name": "a", "content": "😀" * 10001}]},
                       ensure_ascii=False).encode()
    j = err(review(e, "alice", tc, sid, None, raw=long1), 422, "item_too_long")
    assert j["detail"]["got"] == 10001 and j["detail"]["index"] == 0
    err(review(e, "alice", tc, sid, {"final_score": 1, "items": [{"name": "n" * 101, "content": "x"}]}),
        422, "item_name_too_long")
    err(review(e, "alice", tc, sid, None, raw=b'{"final_score": 1, "items": [{"name": "a", "content": "\\ud800"}]}'),
        422, "invalid_item")
    # body caps: review 8 MB, every other JSON request 2 MB
    big = b'{"final_score": 1, "items": [], "pad": "' + b"x" * (8 * 1024 * 1024) + b'"}'
    err(review(e, "alice", tc, sid, None, raw=big), 413, "body_too_large")
    c = e.clients["alice"]
    r = c.post(f"{API}/prompts", content=b'{"name": "x", "text": "' + b"y" * (2 * 1024 * 1024 + 10) + b'"}',
               headers={**H(c), "Content-Type": "application/json"})
    err(r, 413, "body_too_large")
    assert len(json.loads((paths.subtask_dir(e.cfg, sid) / "review.json").read_text("utf-8"))["revisions"]) == 1


def test_review_get_endpoint_and_csrf(app_env):
    e = app_env
    tc, sid = _done_essay(e)
    c = e.clients["alice"]
    g = c.get(f"{API}/batches/{tc}/subtasks/{sid}/review", headers=H(c)).json()
    assert g["review"] is None and g["ai_score"] == SCORE and g["can_review"] is True
    assert g["limits"]["max_items"] == 200
    # cookie write without CSRF token is refused; Bearer works without it
    r = c.post(f"{API}/batches/{tc}/subtasks/{sid}/review", json={"adopt_ai": True})
    assert r.status_code == 403
    tok = c.post(f"{API}/tokens", json={"name": "t", "role": "user"}, headers=H(c)).json()["token"]
    r = TestClient(e.app).post(f"{API}/batches/{tc}/subtasks/{sid}/review", json={"adopt_ai": True},
                               headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200, r.text
    d = c.get(f"{API}/batches/{tc}/subtasks/{sid}", headers=H(c)).json()
    assert d["review"]["revisions"][0]["adopted_ai"] is True
    assert subs(e, "alice", tc)[0]["review_outdated"] is False


def test_regrade_after_review_keeps_review_and_flags_it(app_env):
    e = app_env
    tc, sid = _done_essay(e)
    c = e.clients["alice"]
    assert review(e, "alice", tc, sid, {"final_score": 9}).json()["revision"]["diff"] == 9 - SCORE
    assert c.post(f"{API}/batches/{tc}/subtasks/{sid}/regrade", headers=H(c)).status_code == 200
    wait_done(e, "alice", tc)
    s = subs(e, "alice", tc)[0]
    assert s["review_status"] == "reviewed" and s["teacher_score"] == 9 and s["review_outdated"] is True
    row = c.get(f"{API}/batches/{tc}/result", headers=H(c)).json()["subtasks"][0]
    assert (row["status"], row["final_score"], row["diff"], row["ai_attempt"], row["review_ai_attempt"],
            row["review_outdated"]) == ("done", 9, 9 - SCORE, 2, 1, True)
    # reviewing again against the new attempt clears the flag
    assert review(e, "alice", tc, sid, {"adopt_ai": True}).json()["revision"]["ai_attempt"] == 2
    assert subs(e, "alice", tc)[0]["review_outdated"] is False
