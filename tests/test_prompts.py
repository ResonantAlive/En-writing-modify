"""M3 prompt templates: versioning, builtin read-only, validation, render, assist endpoint."""
from __future__ import annotations

import json
import shutil
import sys
import types

import pytest

import app.prompts.store as tstore
from app.prompts.render import render_system_prompt
from tests.test_providers import API, EXT, KEY, Env, Mock, err, env, mock  # noqa: F401  (fixtures)

from app.config import PROJECT_ROOT

DRAFTS = PROJECT_ROOT / "app" / "prompts" / "builtin"          # approved 2026-11-14, shipped
T = "Grade {{TOPIC}} / {{RUBRIC}} / {{GENRE}} / {{FULL_SCORE}}\n{{FEWSHOT}}"


def mk(env, who, **over):
    body = {"name": "tpl", "text": T, "input_mode": "text", **over}
    return env.post(who, "/prompts", json=body)


@pytest.fixture
def builtin_dir(tmp_path, monkeypatch):
    d = tmp_path / "builtin"
    d.mkdir()
    for f in DRAFTS.glob("*.json"):
        shutil.copy(f, d / f.name)
    monkeypatch.setattr(tstore, "BUILTIN_DIR", d)
    return d


# ------------------------------------------------------------------ shipped builtin templates
def test_builtin_templates_ship_and_are_valid():
    shipped = sorted(p.name for p in (PROJECT_ROOT / "app" / "prompts" / "builtin").iterdir() if p.name != ".gitkeep")
    assert shipped == ["image_v1.json", "text_v1.json"]
    assert not (PROJECT_ROOT / "app" / "prompts" / "builtin_drafts").exists()
    ids = set()
    for f in sorted(DRAFTS.glob("*.json")):
        t = json.loads(f.read_text("utf-8"))
        ids.add(t["id"])
        v = t["versions"][0]
        assert t["builtin"] is True and v["version"] == 1
        assert v["score_key"] == "判分" and v["criteria_key"] == "判据"
        assert v["required_keys"] == ["判分", "判据"] and v["max_chars"] == 20000
        assert {"分数", "得分", "score"} <= set(v["score_aliases"])
        assert len(v["text"]) <= 20000
        for ph in ("{{TOPIC}}", "{{RUBRIC}}", "{{GENRE}}", "{{FULL_SCORE}}", "{{FEWSHOT}}"):
            assert ph in v["text"]
        for key in ("判分", "判据", "高光点", "错误点", "错误类型", "词数是否符合要求",
                    "用词是否符合文体要求", "格式是否符合文体要求"):
            assert key in v["text"]
        assert "OCR" not in v["text"] and "忽略" not in v["text"] and "ignore" not in v["text"].lower()
        assert v["input_mode"] == ("text" if t["id"] == "builtin_text" else "image")
    assert ids == {"builtin_text", "builtin_image"}
    img = json.loads((DRAFTS / "image_v1.json").read_text("utf-8"))["versions"][0]["text"]
    txt = json.loads((DRAFTS / "text_v1.json").read_text("utf-8"))["versions"][0]["text"]
    assert "第k页/共n页" in img and "手写" in img and "文本" in txt


def test_no_builtins_by_default(env):
    assert env.get("alice", "/prompts").json() == []


# ------------------------------------------------------------------ builtin read-only
def test_builtin_readonly_and_save_as(env, builtin_dir):
    lst = env.get("alice", "/prompts").json()
    assert {x["id"] for x in lst} == {"builtin_text", "builtin_image"} and all(x["builtin"] for x in lst)
    assert [x["id"] for x in env.get("alice", "/prompts?input_mode=image").json()] == ["builtin_image"]
    full = env.get("bob", "/prompts/builtin_text").json()
    assert full["versions"][0]["input_mode"] == "text"
    for who in ("alice", "rooty"):
        r = env.post(who, "/prompts/builtin_text/versions", json={"text": T})
        err(r, 409, "builtin_readonly")
        assert r.json()["detail"]["hint"]
        err(env.delete(who, "/prompts/builtin_text"), 409, "builtin_readonly")
    err(env.post("rooty", "/prompts/builtin_text/share", json={"shared": False}), 409, "builtin_readonly")
    # save as new template with parent link
    r = mk(env, "alice", name="mine", parent_id="builtin_text", parent_version=1)
    assert r.status_code == 201
    t = r.json()
    assert t["parent_id"] == "builtin_text" and t["parent_version"] == 1 and t["builtin"] is False
    assert t["owner_id"] == env.uid("alice") and t["id"] != "builtin_text"
    # parent_version defaults to the parent's latest; unknown parent => 404
    assert mk(env, "alice", parent_id="builtin_image").json()["parent_version"] == 1
    assert mk(env, "alice", parent_id="tpl_nope").status_code == 404
    # parent that the caller cannot see => 404
    assert mk(env, "bob", parent_id=t["id"]).status_code == 404


def test_builtin_usable_by_get_version_and_can_use(env, builtin_dir):
    assert tstore.can_use(env.cfg, env.principal("alice"), "builtin_image")
    assert tstore.get_version(env.cfg, "builtin_image", None)["input_mode"] == "image"
    assert tstore.get_version(env.cfg, "builtin_image", 1)["version"] == 1
    with pytest.raises(Exception):
        tstore.get_version(env.cfg, "builtin_image", 2)


# ------------------------------------------------------------------ versions
def test_versions_immutable_and_owner_rules(env):
    t = mk(env, "alice", name="a").json()
    assert t["id"].startswith("tpl_") and len(t["versions"]) == 1
    v1 = dict(t["versions"][0])
    assert v1["version"] == 1 and v1["created_by"] == env.uid("alice")
    assert v1["score_key"] == "判分" and v1["criteria_key"] == "判据" and v1["required_keys"] == ["判分", "判据"]
    assert v1["max_chars"] == 20000 and v1["score_aliases"] == []
    r = env.post("alice", f"/prompts/{t['id']}/versions", json={"text": "v2 {{FEWSHOT}}", "score_aliases": ["s"]})
    assert r.status_code == 201
    j = r.json()
    assert [v["version"] for v in j["versions"]] == [1, 2] and j["versions"][0] == v1     # v1 untouched
    assert j["versions"][1]["score_aliases"] == ["s"] and j["versions"][1]["input_mode"] == "text"
    assert j["versions"][1]["score_key"] == "判分"                                        # inherited
    # input_mode cannot change
    err(env.post("alice", f"/prompts/{t['id']}/versions", json={"text": T, "input_mode": "image"}), 422,
        "input_mode_immutable")
    # no PUT/PATCH on versions
    assert env.patch("alice", f"/prompts/{t['id']}/versions", json={"text": "x"}).status_code in (404, 405)
    # others cannot touch (404), even for a shared template
    err(env.post("bob", f"/prompts/{t['id']}/versions", json={"text": T}), 404, "not_found")
    env.post("rooty", f"/prompts/{t['id']}/share", json={"shared": True})
    err(env.post("bob", f"/prompts/{t['id']}/versions", json={"text": T}), 404, "not_found")
    err(env.delete("bob", f"/prompts/{t['id']}"), 404, "not_found")
    assert env.get("bob", f"/prompts/{t['id']}").status_code == 200
    # get_version pinned vs latest
    assert tstore.get_version(env.cfg, t["id"], 1)["text"] == T
    assert tstore.get_version(env.cfg, t["id"], None)["text"].startswith("v2")


def test_visibility_and_share(env):
    ta = mk(env, "alice", name="a").json()
    ids = lambda who: {x["id"] for x in env.get(who, "/prompts").json()}   # noqa: E731
    assert ids("alice") == {ta["id"]} and ids("bob") == set() and ids("rooty") == {ta["id"]}
    err(env.get("bob", f"/prompts/{ta['id']}"), 404, "not_found")
    assert env.get("rooty", f"/prompts/{ta['id']}").status_code == 200
    err(env.post("alice", f"/prompts/{ta['id']}/share", json={"shared": True}), 403, "forbidden")
    r = env.post("rooty", f"/prompts/{ta['id']}/share", json={"shared": True})
    assert r.status_code == 200 and r.json()["shared"] is True
    assert ids("bob") == {ta["id"]}
    assert tstore.can_use(env.cfg, env.principal("bob"), ta["id"])
    env.post("rooty", f"/prompts/{ta['id']}/share", json={"shared": False})
    assert ids("bob") == set()
    assert env.post("rooty", "/prompts/tpl_nope/share", json={"shared": True}).status_code == 404
    assert env.get("alice", "/prompts?input_mode=image").json() == []
    assert env.get("alice", "/prompts?input_mode=bogus").status_code == 422
    row = env.get("alice", "/prompts").json()[0]
    assert set(row) == {"id", "name", "builtin", "owner_id", "shared", "latest_version", "input_mode",
                        "parent_id", "updated_at"} and row["latest_version"] == 1 and "text" not in row


def test_root_edit_creates_version_and_audit(env):
    t = mk(env, "alice", name="a").json()
    r = env.post("rooty", f"/prompts/{t['id']}/versions", json={"text": "root edit {{FEWSHOT}}"})
    assert r.status_code == 201
    j = r.json()
    assert j["owner_id"] == env.uid("alice") and len(j["versions"]) == 2
    assert j["versions"][1]["created_by"] == env.uid("rooty") and j["versions"][0]["text"] == T
    recs = [json.loads(line) for f in env.cfg.audit_dir.glob("*.jsonl") for line in f.read_text("utf-8").splitlines()]
    hit = [x for x in recs if x["action"] == "prompt.root_edit"]
    assert len(hit) == 1 and hit[0]["actor"] == env.uid("rooty") and hit[0]["target"] == t["id"]
    assert hit[0]["detail"] == {"owner_id": env.uid("alice"), "new_version": 2}
    # editing own template is not a root-edit audit event
    own = mk(env, "rooty", name="r").json()
    env.post("rooty", f"/prompts/{own['id']}/versions", json={"text": T})
    recs = [json.loads(line) for f in env.cfg.audit_dir.glob("*.jsonl") for line in f.read_text("utf-8").splitlines()]
    assert len([x for x in recs if x["action"] == "prompt.root_edit"]) == 1


def test_soft_delete_keeps_versions(env):
    t = mk(env, "alice").json()
    env.post("alice", f"/prompts/{t['id']}/versions", json={"text": "two {{FEWSHOT}}"})
    assert env.delete("alice", f"/prompts/{t['id']}").status_code == 204
    assert env.get("alice", "/prompts").json() == []
    assert env.get("alice", f"/prompts/{t['id']}").status_code == 404
    assert not tstore.can_use(env.cfg, env.principal("alice"), t["id"])
    # data and every version are still there (pinned batches keep working)
    raw = json.loads((env.cfg.app_dir / "prompts.json").read_text("utf-8"))["templates"][0]
    assert raw["deleted"] is True and [v["version"] for v in raw["versions"]] == [1, 2]
    assert tstore.get_version(env.cfg, t["id"], 1)["text"] == T
    assert tstore.get_version(env.cfg, t["id"], None)["text"].startswith("two")
    err(env.post("alice", f"/prompts/{t['id']}/versions", json={"text": T}), 404, "not_found")
    assert env.delete("alice", f"/prompts/{t['id']}").status_code == 404
    # root deleting another user's template is audited
    t2 = mk(env, "alice").json()
    assert env.delete("rooty", f"/prompts/{t2['id']}").status_code == 204
    recs = [json.loads(line) for f in env.cfg.audit_dir.glob("*.jsonl") for line in f.read_text("utf-8").splitlines()]
    assert any(x["action"] == "prompt.root_delete" for x in recs)


# ------------------------------------------------------------------ validation
def test_fewshot_placeholder_validation(env):
    r = mk(env, "alice", text="no placeholder here")
    err(r, 422, "fewshot_placeholder_missing")
    r = mk(env, "alice", text="no placeholder here", auto_append_fewshot=True)
    assert r.status_code == 201
    assert r.json()["versions"][0]["text"] == "no placeholder here\n\n{{FEWSHOT}}"
    tid = r.json()["id"]
    err(env.post("alice", f"/prompts/{tid}/versions", json={"text": "still none"}), 422, "fewshot_placeholder_missing")
    ok = env.post("alice", f"/prompts/{tid}/versions", json={"text": "v2", "auto_append_fewshot": True})
    assert ok.json()["versions"][1]["text"] == "v2\n\n{{FEWSHOT}}"
    # already present => never appended twice
    r = mk(env, "alice", text="a {{FEWSHOT}} b", auto_append_fewshot=True)
    assert r.json()["versions"][0]["text"] == "a {{FEWSHOT}} b"


def test_prompt_too_long_counts_code_points(env):
    ph = "{{FEWSHOT}}"
    ok = mk(env, "alice", text=ph + "😀" * 4, max_chars=15)           # 15 code points (19 UTF-16 units)
    assert ok.status_code == 201, ok.text
    r = mk(env, "alice", text=ph + "😀" * 5, max_chars=15)           # 16
    err(r, 422, "prompt_too_long")
    assert r.json()["detail"] == {"length": 16, "max_chars": 15}
    # default limit comes from settings (20000)
    assert mk(env, "alice", text=ph + "a" * (20000 - 11)).status_code == 201
    err(mk(env, "alice", text=ph + "a" * (20000 - 10)), 422, "prompt_too_long")
    # auto-appended placeholder counts too
    err(mk(env, "alice", text="a" * 20000, auto_append_fewshot=True), 422, "prompt_too_long")
    # max_chars itself is bounded by the setting
    err(mk(env, "alice", max_chars=20001), 422, "validation_error")
    err(mk(env, "alice", max_chars=0), 422, "validation_error")
    # new version inherits max_chars
    tid = ok.json()["id"]
    err(env.post("alice", f"/prompts/{tid}/versions", json={"text": ph + "😀" * 5}), 422, "prompt_too_long")


def test_other_validation(env):
    err(mk(env, "alice", input_mode="audio"), 422, "invalid_input_mode")
    err(mk(env, "alice", input_mode=None), 422, "invalid_input_mode")
    err(mk(env, "alice", name="  "), 422, "validation_error")
    err(mk(env, "alice", text=""), 422, "validation_error")
    err(mk(env, "alice", score_key="分", required_keys=["判据"]), 422, "score_key_not_required")
    r = mk(env, "alice", score_key="score", criteria_key="why")
    assert r.json()["versions"][0]["required_keys"] == ["score", "why"]
    r = mk(env, "alice", score_key="s", required_keys=["s"], score_aliases=["a", "b"])
    assert r.json()["versions"][0]["required_keys"] == ["s"] and r.json()["versions"][0]["score_aliases"] == ["a", "b"]
    err(mk(env, "alice", score_aliases=["ok", 3]), 422, "validation_error")
    assert TestClientless(env)


def TestClientless(env):          # unauthenticated access is refused
    from fastapi.testclient import TestClient
    c = TestClient(env.app)
    return c.get(f"{API}/prompts").status_code == 401 and c.post(f"{API}/prompts/assist", json={}).status_code == 401


# ------------------------------------------------------------------ render
def test_render_single_pass():
    v = {"text": "R:{{RUBRIC}}|T:{{TOPIC}}|G:{{GENRE}}|S:{{FULL_SCORE}}|F:{{FEWSHOT}}|{{UNKNOWN}}"}
    b = {"topic": "t {{RUBRIC}}", "rubric": "see {{FEWSHOT}} and {{TOPIC}}", "genre": "letter", "full_score": 15}
    out = render_system_prompt(v, b, "SHOT {{TOPIC}}")
    assert out == "R:see {{FEWSHOT}} and {{TOPIC}}|T:t {{RUBRIC}}|G:letter|S:15|F:SHOT {{TOPIC}}|{{UNKNOWN}}"
    assert render_system_prompt(v, {**b, "full_score": 12.5}, None).count("S:12.5") == 1
    assert "F:|" in render_system_prompt(v, b, "")
    assert "F:|" in render_system_prompt(v, b, None)
    assert render_system_prompt({"text": "x {{FEWSHOT}}"}, {}, "A$1\\g<0>") == "x A$1\\g<0>"       # no regex-escape bugs


def test_render_real_draft_roundtrip():
    v = json.loads((DRAFTS / "text_v1.json").read_text("utf-8"))["versions"][0]
    out = render_system_prompt(v, {"topic": "My Hometown", "rubric": "content 10", "genre": "essay",
                                   "full_score": 25.0}, "")
    assert "My Hometown" in out and "{{" not in out and "25" in out and "25.0" not in out


# ------------------------------------------------------------------ assist
@pytest.fixture
def history(monkeypatch):
    try:
        import app.grading.history as h
    except ImportError:
        import app.grading as pkg
        h = types.ModuleType("app.grading.history")
        monkeypatch.setitem(sys.modules, "app.grading.history", h)
        monkeypatch.setattr(pkg, "history", h, raising=False)
    calls = []
    recs = [{"task_code": f"T{i}", "subtask_id": f"S{i}", "owner_id": "u", "topic": f"topic{i}", "rubric": "r",
             "genre": "g", "full_score": 10, "transcription": f"essay text {i}", "ai_result": {"判分": 5},
             "review": {"final_score": 6, "items": [], "adopted_ai": False, "reviewer_secret": "no"},
             "template": {"id": "t", "version": 1}, "thinking": "PRIVATE-THINKING",
             "images": ["x.jpg"], "api_key": "sk-leak"} for i in range(8)]

    def fake(cfg, owner_id):
        calls.append(owner_id)
        return iter(recs)
    monkeypatch.setattr(h, "iter_records", fake, raising=False)
    return calls


def assist(env, who, mock_, **over):
    p = over.pop("provider", None) or env.mkprov(who, base_url=mock_.url)
    body = {"provider_id": p["id"], "requirement": "A strict marker for letters", **over}
    return env.post(who, "/prompts/assist", json=body)


def test_assist_basic_no_history(env, mock, history):
    mock.script = [{"content": "  DRAFT TEXT {{FEWSHOT}} "}]
    r = assist(env, "alice", mock, topic="My trip")
    assert r.status_code == 200 and r.json() == {"draft": "DRAFT TEXT {{FEWSHOT}}"}
    req = mock.requests[0]
    assert req["body"]["model"] == "m-default" and req["headers"]["Authorization"] == f"Bearer {KEY}"
    msgs = req["body"]["messages"]
    assert msgs[0]["role"] == "system" and "{{FEWSHOT}}" in msgs[0]["content"]
    assert "A strict marker for letters" in msgs[1]["content"] and "My trip" in msgs[1]["content"]
    assert "essay text" not in json.dumps(msgs) and history == []
    assert KEY not in r.text


def test_assist_with_history_fields_and_scope(env, mock, history):
    r = assist(env, "alice", mock, include_history=True)
    assert r.status_code == 200
    assert history == [env.uid("alice")]
    user_msg = mock.requests[0]["body"]["messages"][1]["content"]
    assert user_msg.count('"task_code"') == 5                   # at most 5 records
    assert "essay text 0" in user_msg and "topic4" in user_msg and "topic5" not in user_msg
    for forbidden in ("PRIVATE-THINKING", "images", "x.jpg", "sk-leak", "api_key", "reviewer_secret", "owner_id"):
        assert forbidden not in user_msg
    # users may not ask for all
    err(assist(env, "alice", mock, include_history=True, history_scope="all"), 403, "forbidden")
    # root may
    p = env.mkprov("rooty", base_url=mock.url)
    assert assist(env, "rooty", mock, provider=p, include_history=True, history_scope="all").status_code == 200
    assert history[-1] is None
    assert assist(env, "rooty", mock, provider=p, include_history=True).status_code == 200
    assert history[-1] == env.uid("rooty")
    assert len(mock.requests) == 3                               # the 403 never reached the provider


def test_assist_history_module_missing_is_tolerated(env, mock, monkeypatch):
    import app.prompts.assist as a
    monkeypatch.setitem(sys.modules, "app.grading.history", None)       # import fails
    assert a._history(env.cfg, "u") == []
    assert assist(env, "alice", mock, include_history=True).status_code == 200


def test_assist_access_and_ack(env, mock, history):
    other = env.mkprov("bob", base_url=mock.url)
    err(assist(env, "alice", mock, provider=other), 404, "not_found")          # not visible
    err(env.post("alice", "/prompts/assist", json={"provider_id": "prov_nope", "requirement": "x"}), 404, "not_found")
    # unacked third-party => 409 and nothing is sent anywhere
    p = env.mkprov("alice", base_url=EXT)
    r = env.post("alice", "/prompts/assist", json={"provider_id": p["id"], "requirement": "x"})
    err(r, 409, "third_party_ack_required")
    assert r.json()["detail"] == {"host": "api.example.com"}
    assert mock.requests == []
    # explicit ack in the request records the ack (the call itself fails: example.com is never contacted
    # because we point the provider back to the mock first)
    env.post("alice", f"/providers/{p['id']}/ack")
    env.patch("alice", f"/providers/{p['id']}", json={"base_url": mock.url})        # local now => no ack needed
    assert env.post("alice", "/prompts/assist", json={"provider_id": p["id"], "requirement": "x"}).status_code == 200
    # validation
    assert env.post("alice", "/prompts/assist", json={"provider_id": p["id"], "requirement": ""}).status_code == 422
    assert env.post("alice", "/prompts/assist", json={"provider_id": p["id"], "requirement": "x",
                                                      "history_scope": "weird"}).status_code == 422


def test_assist_third_party_ack_in_body_records_ack(env, mock, history, monkeypatch):
    from app.providers import client as llm
    seen = []

    async def fake_post(provider, model, messages, params, timeout_s, client=None):
        seen.append(provider["base_url"])
        return llm.RawCall(200, {"choices": [{"message": {"content": "D {{FEWSHOT}}"}}]}, "", 1)
    monkeypatch.setattr(llm, "post_chat", fake_post)          # never touch the network
    p = env.mkprov("alice", base_url=EXT)
    r = env.post("alice", "/prompts/assist", json={"provider_id": p["id"], "requirement": "x",
                                                   "third_party_ack": True})
    assert r.status_code == 200 and seen == [EXT]
    assert env.get("alice", f"/providers/{p['id']}/ack-status").json()["acked"] is True
    assert env.post("alice", "/prompts/assist", json={"provider_id": p["id"], "requirement": "y"}).status_code == 200


def test_assist_provider_failures(env, mock, history):
    mock.script = [{"status": 500, "body": b"nope"}]
    r = assist(env, "alice", mock)
    err(r, 502, "assist_failed")
    assert r.json()["detail"]["http_status"] == 500 and KEY not in r.text
    mock.script = [{"content": "   "}]
    err(assist(env, "alice", mock), 502, "assist_failed")
    p = env.mkprov("alice", base_url="http://127.0.0.1:1/v1")
    r = env.post("alice", "/prompts/assist", json={"provider_id": p["id"], "requirement": "x"})
    err(r, 502, "assist_failed")
    assert r.json()["detail"]["error_kind"] == "connect" and KEY not in r.text


def test_assist_route_not_shadowed_by_id(env, mock, history):
    # POST /prompts/assist is the assist endpoint, not a template id
    assert assist(env, "alice", mock).status_code == 200
    assert env.get("alice", "/prompts/assist").status_code == 404
