"""Batch common-problems report: what is sent to the model, normalisation of its answer, storage / backups,
errors, third-party acknowledgement and cross-account access. Mock LLM only."""
from __future__ import annotations

import json

import pytest

from app.batches import paths, report, results, store
from tests.mock_llm_server import mock_llm  # noqa: F401
from tests.test_batches import API, H
from tests.test_m5 import app_env, create_batch, err, review, setup_account, subs, wait_done  # noqa: F401

MARK = "教学助手"        # the report system prompt (grading prompts never contain it)


def is_report(req) -> bool:
    msgs = req["body"].get("messages") or []
    return bool(msgs) and MARK in str(msgs[0].get("content"))


def report_requests(e):
    return [r for r in e.mock.chat_requests() if is_report(r)]


def answer(obj) -> dict:
    return {"kind": "content", "content": json.dumps(obj, ensure_ascii=False)}


GOOD = {"概述": "整体中等，时态问题突出。",
        "共性问题": [
            {"问题": "拼写错误", "典型表现": "recieve", "涉及作文": [2], "教学建议": "讲评常见拼写"},
            {"问题": "时态不一致", "典型表现": "过去式混用", "涉及作文": [1, 2, 3, 9], "教学建议": "专项练习"},
            {"问题": "编号不存在的问题", "涉及作文": [9]},
            "not an object",
        ],
        "整体优点": ["切题", 5, "结构清楚"]}


def script(report_reply=None, grading_first_fails=False):
    def fn(req, n):
        if is_report(req):
            return report_reply if report_reply is not None else answer(GOOD)
        return "http:400" if grading_first_fails and n == 1 else "ok_json"
    return fn


def finished_batch(e, who="alice", n=3, ids=None, fail_one=True):
    ids = ids or setup_account(e, who, mode="image")
    e.mock.script(script(grading_first_fails=fail_one))
    tc = create_batch(e, who, ids, n=n, route="vlm")
    wait_done(e, who, tc)
    return tc, ids


def post_report(e, who, tc, body=None):
    c = e.clients[who]
    return c.post(f"{API}/batches/{tc}/report", json=body or {}, headers=H(c))


def used_numbers(e, who, tc):
    return [i for i, s in enumerate(subs(e, who, tc), 1) if s["status"] == "done"]


def test_report_content_sent_and_normalised(app_env):
    e = app_env
    a = e.clients["alice"]
    tc, ids = finished_batch(e)
    rows = subs(e, "alice", tc)
    used = used_numbers(e, "alice", tc)
    assert len(used) == 2 and [s["status"] for s in rows].count("failed") == 1
    # teacher reviews one essay with items (replaces the AI text) and overrides the score
    rev_no = used[0]
    assert review(e, "alice", tc, rows[rev_no - 1]["subtask_id"],
                  {"final_score": 15.5, "items": [{"name": "语言", "content": "TEACHER-ITEM-XYZ"}]}).status_code == 200
    g = a.get(f"{API}/batches/{tc}/report", headers=H(a)).json()
    assert g["report"] is None and g["ready"] is True and g["default_provider_id"] == ids[0]
    before = len(report_requests(e))
    r = post_report(e, "alice", tc)
    assert r.status_code == 200, r.text
    doc = r.json()
    assert len(report_requests(e)) == before + 1                         # exactly one model call
    # ---- what was sent
    msgs = report_requests(e)[-1]["body"]["messages"]
    user = msgs[1]["content"]
    assert "A letter to Tom" in user and "content 10, language 10" in user and "满分：20" in user
    assert "TEACHER-ITEM-XYZ" in user and '"最终分": 15.5' in user and '"分数来源": "老师"' in user
    assert user.count("整体不错，注意时态一致。") == 1                  # only the unreviewed essay's AI comment
    assert "另有 1 篇没有可用结果" in user
    blob = json.dumps(msgs, ensure_ascii=False)
    for sid in store.list_subtask_ids(e.cfg, tc):
        assert sid not in blob and tc not in blob
    assert "data:image" not in blob and "base64" not in blob and "sk-test" not in blob
    assert e.uid("alice") not in blob
    # ---- normalisation: unknown numbers dropped, counts computed here, sorted by count
    valid = set(used)
    want_t = sorted(valid & {1, 2, 3})
    probs = doc["problems"]
    assert [p["problem"] for p in probs][0] == "时态不一致"
    assert probs[0]["essays"] == want_t and probs[0]["count"] == len(want_t)
    assert probs[0]["share"] == round(len(want_t) / 2, 4)
    names = [p["problem"] for p in probs]
    assert "编号不存在的问题" not in names
    if 2 in valid:
        assert "拼写错误" in names
    assert set(doc["warnings"]) == {"unknown_essay_numbers", "problem_without_essays", "problem_not_object"}
    assert doc["strengths"] == ["切题", "结构清楚"] and doc["summary"].startswith("整体中等")
    assert doc["inputs"] == {"essays_total": 3, "essays_used": 2, "reviewed": 1,
                             "excluded": {"failed": 1, "cancelled": 0}, "chars": doc["inputs"]["chars"],
                             "includes_teacher_review": True}
    assert [x["no"] for x in doc["essay_index"]] == used
    assert doc["model"] == {"provider_id": ids[0], "provider_alias": "mock-alice", "name": "m1"}
    # ---- stored next to result-<tc>.json; GET returns it; secrets and keys never in it
    path = report.report_path(e.cfg, tc)
    assert path.parent == results.result_path(e.cfg, tc).parent and path.name == f"report-{tc}.json"
    assert json.loads(path.read_text("utf-8")) == doc
    assert a.get(f"{API}/batches/{tc}/report", headers=H(a)).json()["report"] == doc
    assert "sk-test" not in path.read_text("utf-8")
    audit = "".join(p.read_text("utf-8") for p in e.cfg.audit_dir.glob("*.jsonl"))
    assert '"report.generate"' in audit and "TEACHER-ITEM-XYZ" not in audit and "时态不一致" not in audit
    # ---- regenerate: previous report backed up, failed attempts leave the stored report untouched
    bdir = paths.batch_dir(e.cfg, tc) / "report_backup"
    assert not bdir.exists()
    assert post_report(e, "alice", tc).status_code == 200
    assert len(list(bdir.glob(f"report-{tc}.*.json"))) == 1
    kept = path.read_bytes()
    e.mock.script(script(report_reply={"kind": "not_json"}))
    j = err(post_report(e, "alice", tc), 502, "report_parse_failed")
    assert j["detail"]["reason"] == "no_json_object"
    e.mock.script(script(report_reply=answer({"概述": "x"})))
    assert err(post_report(e, "alice", tc), 502, "report_parse_failed")["detail"]["reason"] == "missing_problem_list"
    e.mock.script(script(report_reply="http:500"))
    err(post_report(e, "alice", tc), 502, "report_failed")
    e.mock.script(script(report_reply={"kind": "empty"}))
    err(post_report(e, "alice", tc), 502, "report_failed")
    assert path.read_bytes() == kept and len(list(bdir.glob("*.json"))) == 1


def test_report_not_ready_no_data_and_limits(app_env):
    e = app_env
    ids = setup_account(e, "alice", mode="image")
    e.mock.script(lambda req, n: "delay:5")
    tc = create_batch(e, "alice", ids, n=2, route="vlm")
    err(post_report(e, "alice", tc), 409, "report_not_ready")
    a = e.clients["alice"]
    assert a.get(f"{API}/batches/{tc}/report", headers=H(a)).json()["ready"] is False
    assert a.post(f"{API}/batches/{tc}/cancel", headers=H(a)).status_code == 200
    j = err(post_report(e, "alice", tc), 409, "report_no_data")
    assert j["detail"]["excluded"]["cancelled"] == 2
    assert report_requests(e) == [] and not report.report_path(e.cfg, tc).exists()
    # too much text for one report -> 422, nothing sent
    tc2, _ = finished_batch(e, fail_one=False)
    e.cfg.settings["report"]["max_total_chars"] = 500
    j = err(post_report(e, "alice", tc2), 422, "report_too_large")
    assert j["detail"]["essays"] == 3 and j["detail"]["chars"] > 500 and j["detail"]["max_chars"] == 500
    assert report_requests(e) == []
    # per-essay text cap: a long teacher item is truncated in the prompt, the stored review is not
    e.cfg.settings["report"].update(max_total_chars=150000, max_essay_chars=200)
    sid = store.list_subtask_ids(e.cfg, tc2)[0]
    assert review(e, "alice", tc2, sid, {"final_score": 3, "items": [{"name": "n", "content": "长" * 5000}]}).status_code == 200
    assert post_report(e, "alice", tc2).status_code == 200
    user = report_requests(e)[-1]["body"]["messages"][1]["content"]
    assert "内容过长，已截断" in user and "长" * 300 not in user
    # concurrent generation for the same batch is refused
    with report._lock:
        report._inflight.add(tc2)
    try:
        err(post_report(e, "alice", tc2), 409, "report_in_progress")
    finally:
        report._inflight.discard(tc2)


def test_report_provider_choice_ack_and_isolation(app_env):
    e = app_env
    a, b = e.clients["alice"], e.clients["bob"]
    tc, ids = finished_batch(e, fail_one=False)
    # a provider that is not local needs the account's acknowledgement before anything is sent
    ext = a.post(f"{API}/providers", json={"alias": "ext", "base_url": "https://api.example.com/v1",
                                           "api_key": "sk-test-9999-8888", "default_model": "m9"}, headers=H(a))
    assert ext.status_code == 201, ext.text
    n_before = len(e.mock.chat_requests())
    j = err(post_report(e, "alice", tc, {"provider_id": ext.json()["id"]}), 409, "third_party_ack_required")
    assert j["detail"]["host"] == "api.example.com"
    assert len(e.mock.chat_requests()) == n_before
    # explicit provider / model of the owner's own
    prov2 = a.post(f"{API}/providers", json={"alias": "mock-2", "base_url": e.mock.base_url,
                                             "api_key": "sk-test-1234-5678", "default_model": "m2"}, headers=H(a)).json()
    doc = post_report(e, "alice", tc, {"provider_id": prov2["id"]}).json()
    assert doc["model"] == {"provider_id": prov2["id"], "provider_alias": "mock-2", "name": "m2"}
    assert post_report(e, "alice", tc, {"model": "m7"}).json()["model"]["name"] == "m7"
    assert report_requests(e)[-1]["body"]["model"] == "m7"
    err(post_report(e, "alice", tc, {"provider_id": 5}), 422, "validation_error")
    err(post_report(e, "alice", tc, {"provider_id": "nope"}), 404, "not_found")
    # bob: cannot read or generate for alice's batch, nor use alice's provider for his own batch
    err(b.get(f"{API}/batches/{tc}/report", headers=H(b)), 404, "not_found")
    err(post_report(e, "bob", tc), 404, "not_found")
    ids_b = setup_account(e, "bob", mode="image")
    tc_b, _ = finished_batch(e, who="bob", ids=ids_b, fail_one=False)
    n = len(report_requests(e))
    err(post_report(e, "bob", tc_b, {"provider_id": ids[0]}), 404, "not_found")
    assert len(report_requests(e)) == n
    doc_b = post_report(e, "bob", tc_b).json()
    assert doc_b["inputs"]["essays_used"] == 3 and doc_b["model"]["provider_alias"] == "mock-bob"
    assert "TEACHER" not in json.dumps(report_requests(e)[-1]["body"]) and tc not in json.dumps(report_requests(e)[-1]["body"])
