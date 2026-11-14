import asyncio
import base64
import hashlib
import json
import os

import pytest

from app.batches import executor, hooks, paths, store
from app.batches.executor import ProcessCtx
from app.core import atomic_io
from app.grading import pipeline, revision
from app.providers import store as provstore
from tests.fixtures_images import make_image
from tests.mock_llm_server import mock_llm  # noqa: F401  (fixture)

KEY = "sk-test-xxxx"
TC = "141120260001"
SID = TC + "00001"
TEMPLATE_TEXT = "你是评分老师。题目：{{TOPIC}}\n标准：{{RUBRIC}}\n体裁：{{GENRE}} 满分：{{FULL_SCORE}}\n{{FEWSHOT}}"


@pytest.fixture(autouse=True)
def _clean_hooks():
    prev = executor.get_processor()
    yield
    hooks.reset_defaults()
    executor.set_processor(prev)


def write_provider(cfg, base_url, key=KEY, deleted=False):
    atomic_io.write_json_atomic(cfg.app_dir / "providers.json", {"schema_version": 1, "providers": [{
        "id": "prov_1", "alias": "mock", "base_url": base_url, "api_key": key, "default_model": "mock-model",
        "owner_id": "u1", "shared": False, "is_third_party": False, "created_at": "x", "updated_at": "x",
        "deleted": deleted}]})


def write_template(cfg, **over):
    v = {"version": 1, "created_at": "x", "created_by": "u1", "text": TEMPLATE_TEXT, "input_mode": "image",
         "score_key": "判分", "score_aliases": [], "criteria_key": "判据", "required_keys": ["判分", "判据"],
         "max_chars": 20000}
    v.update(over)
    atomic_io.write_json_atomic(cfg.app_dir / "prompts.json", {"schema_version": 1, "templates": [{
        "id": "tpl_1", "name": "t", "builtin": False, "owner_id": "u1", "shared": False, "parent_id": None,
        "parent_version": None, "deleted": False, "created_at": "x", "versions": [v]}]})


def setup(cfg, mock, *, pages=1, route="vlm", size=(300, 200), fewshot_ids=(), choice="skip_fewshot",
          body_extra=None, tpl_over=None):
    write_provider(cfg, mock.base_url if mock else "http://127.0.0.1:9/v1")
    write_template(cfg, **(tpl_over or {}))
    body = {"provider_id": "prov_1", "template_id": "tpl_1", "template_version": None, "topic": "My Trip",
            "rubric": "Rubric-XYZ", "genre": "narrative", "full_score": 10, **(body_extra or {})}
    rev = revision.make_revision(cfg, "u1", body, list(fewshot_ids), 1, "initial")
    bdir = paths.batch_dir(cfg, TC)
    sdir = bdir / SID
    (sdir / "image").mkdir(parents=True, exist_ok=True)
    images = []
    for k in range(1, pages + 1):
        data = make_image("jpeg", size, color=(10 * k, 100, 200))
        (sdir / "image" / f"original_{k}.jpg").write_bytes(data)
        images.append({"index": k, "orig_name": f"p{k}.jpg", "original": f"image/original_{k}.jpg",
                       "sha256": hashlib.sha256(data).hexdigest(), "mime": "image/jpeg", "derived": []})
    meta = {"schema_version": 1, "subtask_id": SID, "task_code": TC, "index": 1, "owner_id": "u1",
            "status": "running", "fail_reason": None, "attempts": 1, "config_rev": 1, "review_status": "none",
            "warnings": [], "images": images, "local_ocr_state": "n/a", "timeline": []}
    batch = {"schema_version": 1, "task_code": TC, "owner_id": "u1", "created_at": "2026-11-14T10:00:00",
             "topic": "My Trip", "rubric": "Rubric-XYZ", "genre": "narrative", "full_score": 10, "language": "en",
             "route": route, "local_engine": None, "vlm_transcription_choice": choice,
             "fewshot_ids": list(fewshot_ids), "status": "running", "pause_reason": None, "api_error_streak": 0,
             "breaker": {"tripped_reason": None, "samples": []}, "config_revisions": [rev], "timeline": []}
    atomic_io.write_json_atomic(sdir / "meta.json", meta)
    atomic_io.write_json_atomic(bdir / "batch.json", batch)
    return ProcessCtx(cfg, batch, SID, meta, 1), sdir, bdir


def run(ctx):
    return asyncio.run(pipeline.process(ctx))


def rd(p):
    return json.loads(open(p, encoding="utf-8").read())


def streak(cfg):
    return store.read_batch(cfg, TC)["api_error_streak"]


def meta_of(cfg):
    return store.read_meta(cfg, SID)


def all_text(root):
    out = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".json"):
                out.append(open(os.path.join(dp, f), encoding="utf-8").read())
    return "\n".join(out)


# ------------------------------------------------------------------ happy path
def test_vlm_happy_path_single_call(cfg, mock_llm):
    ctx, sdir, bdir = setup(cfg, mock_llm)
    mock_llm.script(["ok_json"])
    out = run(ctx)
    assert (out.status, out.fail_reason, out.warnings) == ("done", None, [])

    reqs = mock_llm.chat_requests()
    assert len(reqs) == 1                                     # exactly one call, no transcription call
    assert reqs[0]["headers"]["Authorization"] == f"Bearer {KEY}"
    body = reqs[0]["body"]
    assert body["model"] == "mock-model" and body["temperature"] == 0.0 and body["max_tokens"] == 8192
    sysm, user = body["messages"]
    assert sysm["role"] == "system" and "Rubric-XYZ" in sysm["content"] and "My Trip" in sysm["content"]
    assert "{{" not in sysm["content"]
    assert user["content"][0] == {"type": "text", "text": "第1页/共1页"}
    url = user["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    sent_bytes = base64.b64decode(url.split(",", 1)[1])

    req = rd(sdir / "llm" / "001.request.json")
    ref = req["messages"][1]["content"][1]
    assert ref["type"] == "image_ref" and ref["page"] == "1/1" and ref["mime"] == "image/jpeg"
    assert ref["sha256"] == hashlib.sha256(sent_bytes).hexdigest()
    assert (sdir / ref["path"]).read_bytes() == sent_bytes
    assert req["messages"][0] == sysm and req["messages"][1]["content"][0] == user["content"][0]
    assert "base64" not in json.dumps(req)
    assert req["attempt"] == 1 and req["key_last4"] == "xxxx" and req["provider_alias"] == "mock"
    assert req["model"] == "mock-model" and req["config_rev"] == 1 and req["template"] == {"id": "tpl_1", "version": 1}
    assert req["fewshot_ids"] == [] and "headers" not in req
    assert req["params"] == {"temperature": 0.0, "max_tokens": 8192}

    resp = rd(sdir / "llm" / "001.response.json")
    assert resp["attempt"] == 1 and resp["http_status"] == 200 and resp["classification"] == "ok"
    assert resp["error"] is None and resp["error_code"] is None
    assert resp["parse_level"] == "direct" and resp["parsed"]["判分"] == 7
    assert resp["usage"] == {"prompt": 100, "completion": 20, "total": 120}
    assert resp["thinking"] is None and isinstance(resp["raw"], dict) and resp["latency_ms"] >= 0

    g = rd(sdir / "grading.json")
    assert g["schema_version"] == 1 and g["attempt"] == 1 and g["parse_level"] == "direct"
    assert g["result"]["判分"] == 7 and g["score"] == 7 and g["warnings"] == [] and g["raw_text"] is None
    assert g["adopted_at"]

    ocr = rd(sdir / "ocr.json")
    assert ocr["mode"] == "vlm" and ocr["transcription"] == "<empty>" and ocr["transcription_source"] == "none"
    assert ocr["pages"][0]["layout_reliable"] is False and ocr["pages"][0]["lines"] == []
    assert ocr["pages"][0]["image_ref"]["sha256"] == ref["sha256"]
    assert ocr["engine"]["local_engine"] is None and ocr["warnings"] == []

    assert KEY not in all_text(bdir) and "api_key" not in all_text(bdir) and "Bearer" not in all_text(bdir)
    assert streak(cfg) == 0
    m = meta_of(cfg)
    assert m["config_rev"] == 1 and m["llm_attempts"] == 1
    assert any(e["event"] == "llm_attempt" for e in m["timeline"])
    d = m["images"][0]["derived"]
    assert len(d) == 1 and d[0]["purpose"] == "vlm" and d[0]["path"] == "image/derived/vlm_1.jpg"
    assert d[0]["sha256"] == ref["sha256"]


def test_multi_page_order_and_labels(cfg, mock_llm):
    ctx, sdir, _ = setup(cfg, mock_llm, pages=3)
    run(ctx)
    parts = mock_llm.chat_requests()[0]["body"]["messages"][1]["content"]
    assert [p.get("text") for p in parts if p["type"] == "text"] == ["第1页/共3页", "第2页/共3页", "第3页/共3页"]
    assert [p["type"] for p in parts] == ["text", "image_url"] * 3
    refs = [p for p in rd(sdir / "llm" / "001.request.json")["messages"][1]["content"] if p["type"] == "image_ref"]
    for p, r in zip([p for p in parts if p["type"] == "image_url"], refs):
        assert hashlib.sha256(base64.b64decode(p["image_url"]["url"].split(",")[1])).hexdigest() == r["sha256"]


def test_few_shot_text_is_in_system_prompt_and_no_local_ocr_text(cfg, mock_llm):
    from tests.test_fewshot import seed, sub, rv
    ids = seed(cfg, "u1", "131120260001", [sub(text="FEWSHOT-ESSAY-TEXT", review=rv(9))], topic="Old", rubric="OldR")
    ctx, sdir, _ = setup(cfg, mock_llm, fewshot_ids=ids)
    run(ctx)
    system = mock_llm.chat_requests()[0]["body"]["messages"][0]["content"]
    assert "FEWSHOT-ESSAY-TEXT" in system
    for m in mock_llm.chat_requests()[0]["body"]["messages"][1:]:
        assert isinstance(m["content"], list) and all(p["type"] in ("text", "image_url") for p in m["content"])
    assert rd(sdir / "llm" / "001.request.json")["fewshot_ids"] == ids


# ------------------------------------------------------------------ retries / A class
def test_format_error_retries_with_note_and_no_thinking(cfg, mock_llm):
    ctx, sdir, _ = setup(cfg, mock_llm)
    mock_llm.script(["reasoning_bad", "fenced_json"])
    out = run(ctx)
    assert out.status == "done"
    reqs = mock_llm.chat_requests()
    assert len(reqs) == 2
    second = reqs[1]["body"]["messages"]
    assert len(second) == 3 and second[2]["role"] == "user" and "json_parse_failed" in second[2]["content"]
    assert "SECRET-THINKING" not in json.dumps(second, ensure_ascii=False)      # thinking never fed back
    assert "这不是JSON" not in json.dumps(second, ensure_ascii=False)
    r1 = rd(sdir / "llm" / "001.response.json")
    assert r1["classification"] == "A" and r1["error_code"] == "json_parse_failed" and r1["error"]["code"]
    assert "SECRET-THINKING" in r1["thinking"] and "SECRET-THINKING" not in (r1["content"] or "")
    g = rd(sdir / "grading.json")
    assert g["attempt"] == 2 and g["parse_level"] == "fenced"
    assert "SECRET-THINKING" not in json.dumps(g, ensure_ascii=False)
    assert "thinking" not in json.dumps(rd(sdir / "llm" / "002.request.json"))


@pytest.mark.parametrize("beh,code", [("not_json", "json_parse_failed"), ("empty", "empty_response"),
                                      ("refusal", "refusal_or_safety"), ("missing_field", "missing_required_field"),
                                      ("score_out_of_range", "score_out_of_range")])
def test_exhausted_a_class_fails_with_last_code_after_4_attempts(cfg, mock_llm, beh, code):
    ctx, sdir, _ = setup(cfg, mock_llm)
    mock_llm.script([beh] * 10)
    out = run(ctx)
    assert (out.status, out.fail_reason) == ("failed", code)
    assert len(mock_llm.chat_requests()) == 4
    assert sorted(p.name for p in (sdir / "llm").iterdir()) == [
        f"{n:03d}.{k}.json" for n in range(1, 5) for k in ("request", "response")]
    assert not (sdir / "grading.json").exists()
    assert streak(cfg) == 0                               # 2xx with bad content resets, never counts


def test_content_filter_finish_reason(cfg, mock_llm):
    ctx, sdir, _ = setup(cfg, mock_llm)
    mock_llm.script([{"kind": "content", "content": "", "finish_reason": "content_filter"}, "ok_json"])
    assert run(ctx).status == "done"
    assert rd(sdir / "llm" / "001.response.json")["error_code"] == "refusal_or_safety"


def test_regex_only_retried_then_adopted_as_fallback(cfg, mock_llm):
    ctx, sdir, _ = setup(cfg, mock_llm)
    mock_llm.script(["regex_only"] * 10)
    out = run(ctx)
    assert (out.status, out.warnings) == ("done", ["regex_fallback"])
    assert len(mock_llm.chat_requests()) == 4             # retried, not adopted immediately
    r1 = rd(sdir / "llm" / "001.response.json")
    assert r1["error_code"] == "missing_required_field" and r1["parse_level"] == "regex" and r1["classification"] == "A"
    g = rd(sdir / "grading.json")
    assert g["parse_level"] == "regex_fallback" and g["score"] == 7 and g["raw_text"] == "判分: 7"
    assert g["warnings"] == ["regex_fallback"] and g["result"] == {"判分": 7.0} and g["attempt"] == 4


def test_regex_then_good_is_not_a_fallback(cfg, mock_llm):
    ctx, sdir, _ = setup(cfg, mock_llm)
    mock_llm.script(["regex_only", "ok_json"])
    out = run(ctx)
    assert out.status == "done" and out.warnings == []
    assert rd(sdir / "grading.json")["parse_level"] == "direct"


def test_regex_fallback_uses_earlier_attempt_when_last_is_other_error(cfg, mock_llm):
    ctx, sdir, _ = setup(cfg, mock_llm)
    mock_llm.script(["regex_only", "not_json", "not_json", "not_json"])
    out = run(ctx)
    assert out.status == "done" and out.warnings == ["regex_fallback"]
    assert rd(sdir / "grading.json")["attempt"] == 1


def test_single_retry_max_setting(cfg, mock_llm):
    cfg.settings["llm"]["retry_max"] = 1
    ctx, sdir, _ = setup(cfg, mock_llm)
    mock_llm.script(["not_json"] * 5)
    assert run(ctx).status == "failed"
    assert len(mock_llm.chat_requests()) == 2


def test_json_repair_trailing_comma_end_to_end(cfg, mock_llm):
    ctx, sdir, _ = setup(cfg, mock_llm)
    mock_llm.script(["trailing_comma"])
    assert run(ctx).status == "done"
    g = rd(sdir / "grading.json")
    assert g["parse_level"] == "repair" and g["result"]["评语"] == "好" and g["result"]["判据"][0]["body"] == "切题"


@pytest.mark.parametrize("beh,level", [("fenced_json", "fenced"), ("text_around", "balanced"),
                                       ("reasoning_content", "direct"), ("think_tags", "direct")])
def test_thinking_variants_stored_separately(cfg, mock_llm, beh, level):
    ctx, sdir, _ = setup(cfg, mock_llm)
    mock_llm.script([beh])
    assert run(ctx).status == "done"
    resp = rd(sdir / "llm" / "001.response.json")
    assert resp["parse_level"] == level
    if beh in ("reasoning_content", "think_tags"):
        assert "SECRET-THINKING" in resp["thinking"] and "SECRET-THINKING" not in resp["content"]
        assert "SECRET-THINKING" not in (sdir / "grading.json").read_text("utf-8")


# ------------------------------------------------------------------ B / C
@pytest.mark.parametrize("beh", ["http:500", "http:503", "http:429", "http:401", "http:404", "http:408",
                                 "http:302", "connection_drop"])
def test_b_class_no_retry_api_error(cfg, mock_llm, beh):
    ctx, sdir, _ = setup(cfg, mock_llm)
    mock_llm.script([beh, "ok_json"])
    out = run(ctx)
    assert (out.status, out.fail_reason) == ("api_error", "api_error")
    assert len(mock_llm.chat_requests()) == 1
    r = rd(sdir / "llm" / "001.response.json")
    assert r["classification"] == "B" and r["error_code"] == "api_error" and r["error_detail"]
    if beh.startswith("http:"):
        assert r["http_status"] == int(beh[5:]) and beh[5:] in r["error_detail"]
    assert streak(cfg) == 1
    assert not (sdir / "grading.json").exists()


def test_timeout_is_b_class(cfg, mock_llm):
    cfg.settings["llm"]["timeout_s"] = 0.3
    ctx, sdir, _ = setup(cfg, mock_llm)
    mock_llm.script([{"kind": "delay", "seconds": 3, "then": "ok_json"}])
    out = run(ctx)
    assert out.status == "api_error"
    r = rd(sdir / "llm" / "001.response.json")
    assert r["classification"] == "B" and "timeout" in r["error_detail"] and r["http_status"] is None
    assert streak(cfg) == 1


def test_connect_refused_is_b_class(cfg):
    ctx, sdir, _ = setup(cfg, None)                       # nothing listens on port 9
    out = run(ctx)
    assert out.status == "api_error"
    assert rd(sdir / "llm" / "001.response.json")["classification"] == "B"


@pytest.mark.parametrize("st", [400, 413, 422, 418, 405, 409, 415])
def test_c_class_failed_request_rejected(cfg, mock_llm, st):
    ctx, sdir, bdir = setup(cfg, mock_llm)
    b = store.read_batch(cfg, TC)
    b["api_error_streak"] = 2
    atomic_io.write_json_atomic(bdir / "batch.json", b)
    mock_llm.script([f"http:{st}", "ok_json"])
    out = run(ctx)
    assert (out.status, out.fail_reason) == ("failed", "request_rejected")
    assert len(mock_llm.chat_requests()) == 1
    r = rd(sdir / "llm" / "001.response.json")
    assert r["classification"] == "C" and r["error_code"] == "request_rejected" and str(st) in r["error_detail"]
    assert streak(cfg) == 2                               # C neither counts nor clears


def test_any_2xx_resets_streak_even_with_bad_content(cfg, mock_llm):
    ctx, sdir, bdir = setup(cfg, mock_llm)
    b = store.read_batch(cfg, TC)
    b["api_error_streak"] = 2
    atomic_io.write_json_atomic(bdir / "batch.json", b)
    cfg.settings["llm"]["retry_max"] = 0
    mock_llm.script(["not_json"])
    assert run(ctx).status == "failed"
    assert streak(cfg) == 0


def test_breaker_called_in_completion_order_with_outcomes(cfg, mock_llm, monkeypatch):
    from app.batches import breaker
    calls = []
    monkeypatch.setattr(breaker, "record_api_call", lambda c, tc, o: calls.append((tc, o)) or False)
    ctx, *_ = setup(cfg, mock_llm)
    mock_llm.script(["not_json", "ok_json"])
    run(ctx)
    assert calls == [(TC, "ok"), (TC, "ok")]
    calls.clear()
    ctx, *_ = setup(cfg, mock_llm)
    mock_llm.script(["http:500"])
    run(ctx)
    ctx, *_ = setup(cfg, mock_llm)
    mock_llm.script(["http:400"])
    run(ctx)
    assert calls == [(TC, "api_error"), (TC, "other")]


# ------------------------------------------------------------------ attempt accounting
def _fake_attempt(sdir, n, code, cls="A"):
    d = sdir / "llm"
    d.mkdir(exist_ok=True)
    atomic_io.write_json_atomic(d / f"{n:03d}.request.json", {"attempt": n})
    atomic_io.write_json_atomic(d / f"{n:03d}.response.json", {
        "attempt": n, "classification": cls, "error_code": code, "error_detail": "x" if code else None,
        "error": {"code": code} if code else None, "content": "junk"})


def test_interrupted_attempts_do_not_count_and_numbering_continues(cfg, mock_llm):
    ctx, sdir, _ = setup(cfg, mock_llm)
    for n in (1, 2, 3):
        _fake_attempt(sdir, n, "interrupted", "interrupted")
    atomic_io.write_json_atomic(sdir / "llm" / "004.request.json", {"attempt": 4})      # request without response
    mock_llm.script(["not_json"] * 10)
    out = run(ctx)
    assert out.status == "failed"
    assert len(mock_llm.chat_requests()) == 4                                            # full fresh budget
    names = sorted(p.name for p in (sdir / "llm").iterdir())
    assert "008.response.json" in names and "009.request.json" not in names
    assert meta_of(cfg)["run_base"] == 0


def test_prior_attempts_before_run_base_do_not_consume_budget(cfg, mock_llm):
    ctx, sdir, _ = setup(cfg, mock_llm)
    for n in (1, 2, 3, 4):
        _fake_attempt(sdir, n, "json_parse_failed")
    mock_llm.script(["not_json"] * 10)
    assert run(ctx).status == "failed"
    assert meta_of(cfg)["run_base"] == 4
    assert len(mock_llm.chat_requests()) == 4              # (run_base was absent -> fresh budget)


def test_resume_continues_the_same_budget_and_sends_previous_error(cfg, mock_llm):
    ctx, sdir, bdir = setup(cfg, mock_llm)
    for n in (1, 2):
        _fake_attempt(sdir, n, "json_parse_failed")
    _fake_attempt(sdir, 3, "interrupted", "interrupted")
    m = meta_of(cfg)
    m["run_base"] = 0
    atomic_io.write_json_atomic(sdir / "meta.json", m)
    mock_llm.script(["not_json"] * 10)
    out = run(ctx)
    assert (out.status, out.fail_reason) == ("failed", "json_parse_failed")
    reqs = mock_llm.chat_requests()
    assert len(reqs) == 2                                   # 2 counted attempts already used of 4
    assert "json_parse_failed" in reqs[0]["body"]["messages"][-1]["content"]      # note from the last A attempt
    assert sorted(p.name for p in (sdir / "llm").iterdir())[-1] == "005.response.json"


def test_exhausted_before_start_finalises_without_calls(cfg, mock_llm):
    ctx, sdir, _ = setup(cfg, mock_llm)
    for n in (1, 2, 3):
        _fake_attempt(sdir, n, "missing_required_field")
    atomic_io.write_json_atomic(sdir / "llm" / "004.request.json", {"attempt": 4})
    _fake_attempt(sdir, 4, "empty_response")
    m = meta_of(cfg)
    m["run_base"] = 0
    atomic_io.write_json_atomic(sdir / "meta.json", m)
    out = run(ctx)
    assert (out.status, out.fail_reason) == ("failed", "empty_response")
    assert mock_llm.chat_requests() == []


# ------------------------------------------------------------------ misc routes / failures
def test_local_ocr_route_engine_unavailable_never_calls_llm(cfg, mock_llm):
    # model folders missing (conftest points the model root at an empty dir) -> ocr_failed, no LLM call
    ctx, sdir, _ = setup(cfg, mock_llm, route="local_ocr", choice=None)
    ctx.batch["local_engine"] = "paddle"
    out = run(ctx)
    assert (out.status, out.fail_reason) == ("failed", "ocr_failed")
    assert out.warnings and mock_llm.chat_requests() == []
    assert not (sdir / "llm").exists()


def test_provider_deleted_or_missing(cfg, mock_llm):
    ctx, sdir, _ = setup(cfg, mock_llm)
    write_provider(cfg, mock_llm.base_url, deleted=True)
    out = run(ctx)
    assert (out.status, out.fail_reason) == ("failed", "provider_unavailable")
    assert mock_llm.chat_requests() == []
    os.remove(cfg.app_dir / "providers.json")
    assert run(ctx).fail_reason == "provider_unavailable"


def test_current_key_read_at_call_time(cfg, mock_llm):
    ctx, *_ = setup(cfg, mock_llm)
    write_provider(cfg, mock_llm.base_url, key="sk-test-rotated")
    run(ctx)
    assert mock_llm.chat_requests()[0]["headers"]["Authorization"] == "Bearer sk-test-rotated"


def test_preprocess_failed_on_corrupt_image(cfg, mock_llm):
    ctx, sdir, _ = setup(cfg, mock_llm)
    (sdir / "image" / "original_1.jpg").write_bytes(b"not an image")
    out = run(ctx)
    assert (out.status, out.fail_reason) == ("failed", "preprocess_failed")
    assert mock_llm.chat_requests() == []


def test_preprocess_resizes_and_steps_down_to_fit(cfg, mock_llm):
    import io
    import random
    from PIL import Image
    ctx, sdir, _ = setup(cfg, mock_llm, pages=2)
    rnd = random.Random(1)
    for k in (1, 2):
        im = Image.frombytes("RGB", (1600, 1200), bytes(rnd.getrandbits(8) for _ in range(1600 * 1200 * 3)))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=95)
        (sdir / "image" / f"original_{k}.jpg").write_bytes(buf.getvalue())
    cfg.settings["vlm"].update({"max_edge": 1500, "quality": 90, "max_request_mb": 0.8})
    assert run(ctx).status == "done"
    m = meta_of(cfg)
    total = 0
    for img in m["images"]:
        vl = [d for d in img["derived"] if d["purpose"] == "vlm"]
        assert len(vl) == 1
        p = sdir / vl[0]["path"]
        total += p.stat().st_size
        assert hashlib.sha256(p.read_bytes()).hexdigest() == vl[0]["sha256"]
        assert max(vl[0]["params"]["width"], vl[0]["params"]["height"]) < 1500 and "_e" in vl[0]["path"]
    assert total <= 0.8 * 1024 * 1024
    left = sorted(p.name for p in (sdir / "image" / "derived").iterdir())
    assert len(left) == 2                                  # stale iterations removed
    sent = mock_llm.chat_requests()[0]["body"]["messages"][1]["content"]
    refs = [p for p in rd(sdir / "llm" / "001.request.json")["messages"][1]["content"] if p["type"] == "image_ref"]
    for part, ref in zip([p for p in sent if p["type"] == "image_url"], refs):
        assert hashlib.sha256(base64.b64decode(part["image_url"]["url"].split(",")[1])).hexdigest() == ref["sha256"]


def test_derived_reused_on_second_run(cfg, mock_llm):
    ctx, sdir, _ = setup(cfg, mock_llm)
    run(ctx)
    p = sdir / "image" / "derived" / "vlm_1.jpg"
    mt = p.stat().st_mtime_ns
    run(ctx)
    assert p.stat().st_mtime_ns == mt
    assert len(mock_llm.chat_requests()) == 2 and len(rd(sdir / "ocr.json")["pages"]) == 1


def test_cancellation_propagates_and_writes_no_response(cfg, mock_llm):
    ctx, sdir, _ = setup(cfg, mock_llm)
    mock_llm.script([{"kind": "delay", "seconds": 10, "then": "ok_json"}])

    async def go():
        t = asyncio.create_task(pipeline.process(ctx))
        for _ in range(100):
            await asyncio.sleep(0.05)
            if (sdir / "llm" / "001.request.json").exists() and mock_llm.requests:
                break
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t

    asyncio.run(go())
    assert (sdir / "llm" / "001.request.json").exists()
    assert not (sdir / "llm" / "001.response.json").exists()


# ------------------------------------------------------------------ revision + register
def test_make_revision_shape_and_pinning(cfg):
    from tests.test_fewshot import seed, sub, rv
    ids = seed(cfg, "u1", "131120260001", [sub(text="SHOT-TEXT", review=rv(9))])
    write_provider(cfg, "http://127.0.0.1:1/v1")
    write_template(cfg)
    body = {"provider_id": "prov_1", "template_id": "tpl_1", "topic": "TT", "rubric": "RR", "genre": "g",
            "full_score": 10.0, "params": {"temperature": 0.5, "top_p": 0.9, "model": "evil", "messages": []}}
    rev = revision.make_revision(cfg, "u1", body, ids, 3, "user_change")
    assert rev["rev"] == 3 and rev["reason"] == "user_change" and rev["at"]
    assert rev["provider_id"] == "prov_1" and rev["provider_alias"] == "mock" and rev["provider_key_last4"] == "xxxx"
    assert rev["model"] == "mock-model"                         # provider default
    assert rev["params"] == {"temperature": 0.5, "max_tokens": 8192, "top_p": 0.9}
    assert rev["template"] == {"id": "tpl_1", "version": 1}
    assert "SHOT-TEXT" in rev["rendered_prompt"] and "满分：10" in rev["rendered_prompt"]
    assert KEY not in json.dumps(rev)
    rev2 = revision.make_revision(cfg, "u1", {**body, "model": "m2"}, [], 1, "initial")
    assert rev2["model"] == "m2" and "SHOT-TEXT" not in rev2["rendered_prompt"]
    assert "{{FEWSHOT}}" not in rev2["rendered_prompt"]
    ov = revision.make_revision(cfg, "u1", body, [], 2, "user_change", {"id": "tpl_1", "version": 1})
    assert ov["template"]["version"] == 1
    from app.errors import NotFound
    with pytest.raises(NotFound):
        revision.make_revision(cfg, "u1", {**body, "provider_id": "nope"}, [], 1, "initial")


def test_register_sets_processor_and_hooks_only_on_register(cfg):
    assert hooks.find_fewshot is hooks.DEFAULTS["find_fewshot"]
    executor.set_processor(None)
    pipeline.register()
    assert executor.get_processor() is pipeline.process
    from app.grading import fewshot
    assert hooks.find_fewshot is fewshot.find_fewshot and hooks.build_revision is revision.make_revision
