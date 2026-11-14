"""M3 providers: key masking, visibility, third-party detection + ack flow, validate hook, HTTP client."""
from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from app.auth.deps import Principal
from app.batches import hooks
from app.errors import AppError, NotFound
from app.main import create_app
from app.prompts import store as tstore
from app.providers import client as llm
from app.providers import store as pstore
from app.providers import third_party

PW = "correct-horse-battery"
API = "/api/v1"
KEY = "sk-test-SECRETKEY1234"
EXT = "https://api.example.com/v1"


# ------------------------------------------------------------------ fixtures
def _unbind_logging():
    root = logging.getLogger("wm")
    for h in list(root.handlers):
        if getattr(h, "_wm", False):
            root.removeHandler(h)
            h.close()


class Env:
    def __init__(self, data, app):
        self.data, self.app, self.cfg = data, app, app.state.cfg
        self.users: dict[str, dict] = {}
        self.clients: dict[str, TestClient] = {}

    def add_user(self, name, kind="user") -> TestClient:
        codes = json.loads((self.data / "app" / "invite_codes.json").read_text("utf-8"))
        c0 = TestClient(self.app)
        r = c0.post(f"{API}/auth/register", json={
            "invite_code": codes["root_invite_code" if kind == "root" else "user_invite_code"],
            "username": name, "password": PW})
        assert r.status_code == 201, r.text
        self.users[name] = r.json()
        c = TestClient(self.app)
        r = c.post(f"{API}/auth/login", json={"username": name, "password": PW})
        assert r.status_code == 200, r.text
        c.csrf = r.json()["csrf_token"]
        self.clients[name] = c
        return c

    def uid(self, n):
        return self.users[n]["id"]

    def principal(self, n) -> Principal:
        u = self.users[n]
        return Principal(u["id"], n, u["role"], u["role"], "cookie")

    def h(self, n):
        return {"X-CSRF-Token": self.clients[n].csrf}

    def post(self, who, url, **kw):
        return self.clients[who].post(f"{API}{url}", headers=self.h(who), **kw)

    def patch(self, who, url, **kw):
        return self.clients[who].patch(f"{API}{url}", headers=self.h(who), **kw)

    def delete(self, who, url):
        return self.clients[who].delete(f"{API}{url}", headers=self.h(who))

    def get(self, who, url, **kw):
        return self.clients[who].get(f"{API}{url}", **kw)

    def mkprov(self, who, base_url="http://127.0.0.1:9/v1", key=KEY, **kw) -> dict:
        body = {"alias": "p", "base_url": base_url, "api_key": key, "default_model": "m-default", **kw}
        r = self.post(who, "/providers", json=body)
        assert r.status_code == 201, r.text
        return r.json()


@pytest.fixture
def env(tmp_path):
    saved = {k: getattr(hooks, k) for k in ("needs_third_party_ack", "validate_provider_and_template")}
    _unbind_logging()
    data = tmp_path / "data"
    app = create_app(data, acquire_executor_lock=False, enable_grading=False)
    with TestClient(app):
        e = Env(data, app)
        e.add_user("rooty", "root")
        e.add_user("alice")
        e.add_user("bob")
        yield e
    _unbind_logging()
    for k, v in saved.items():
        setattr(hooks, k, v)


def err(r, status, code):
    assert r.status_code == status, r.text
    j = r.json()
    assert set(j) == {"code", "message", "detail"}
    assert j["code"] == code, j


# ------------------------------------------------------------------ mock server
class Mock:
    def __init__(self):
        self.requests: list[dict] = []
        self.script = []              # callables / dicts consumed in order; default => ok
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n)
                outer.requests.append({"path": self.path, "headers": dict(self.headers),
                                       "body": json.loads(raw) if raw else None})
                beh = outer.script.pop(0) if outer.script else {}
                if beh.get("delay"):
                    time.sleep(beh["delay"])
                status = beh.get("status", 200)
                if status == 302:
                    self.send_response(302)
                    self.send_header("Location", "/elsewhere")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = beh.get("body") or {"choices": [{"message": {"content": beh.get("content", "ok")},
                                                        "finish_reason": "stop"}],
                                           "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/v1"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def mock():
    m = Mock()
    yield m
    m.close()


def run(coro):
    return asyncio.run(coro)


def prov(url, key=KEY):
    return {"id": "prov_x", "base_url": url, "api_key": key}


# ------------------------------------------------------------------ masking
def _all_data_text(env, skip=("providers.json",)):
    out = {}
    for f in env.data.rglob("*"):
        if f.is_file() and f.name not in skip:
            try:
                out[str(f)] = f.read_bytes().decode("utf-8", "replace")
            except OSError:
                pass
    return out


def test_key_masked_everywhere(env):
    p = env.mkprov("alice", alias="mine")
    assert KEY not in json.dumps(p) and p["key_last4"] == KEY[-4:] and "api_key" not in p
    r = env.get("alice", "/providers")
    assert KEY not in r.text and r.json()[0]["key_last4"] == "1234"
    # root view (all providers) is masked too
    r = env.get("rooty", "/providers")
    assert any(x["id"] == p["id"] for x in r.json()) and KEY not in r.text
    # patch response, ack, ack-status
    r = env.patch("alice", f"/providers/{p['id']}", json={"alias": "x", "api_key": "sk-test-NEWKEY9999"})
    assert r.status_code == 200 and "NEWKEY" not in r.text and r.json()["key_last4"] == "9999"
    assert "api_key" not in env.post("alice", f"/providers/{p['id']}/ack").text
    # errors never echo the key
    r = env.post("alice", "/providers", json={"alias": "a", "base_url": "ftp://x", "api_key": KEY,
                                              "default_model": "m"})
    err(r, 422, "validation_error")
    assert KEY not in r.text
    r = env.post("alice", "/providers", json={"alias": "", "base_url": "http://127.0.0.1/v1", "api_key": KEY,
                                              "default_model": "m"})
    assert r.status_code == 422 and KEY not in r.text
    r = env.post("alice", "/providers", json={"alias": 5, "api_key": KEY})
    assert r.status_code == 422 and KEY not in r.text
    # nothing but providers.json holds the key: logs, audit, users, ...
    for path, text in _all_data_text(env).items():
        assert KEY not in text and "NEWKEY9999" not in text, path
    assert (env.data / "app" / "providers.json").exists()
    assert list((env.data / "logs" / "audit").glob("*.jsonl"))


def test_key_not_in_audit_or_logs_after_delete(env):
    p = env.mkprov("alice")
    assert env.delete("alice", f"/providers/{p['id']}").status_code == 204
    for path, text in _all_data_text(env, skip=()).items():
        if path.endswith("providers.json"):
            continue
        assert KEY not in text, path


# ------------------------------------------------------------------ visibility
def test_visibility_rules(env):
    pa = env.mkprov("alice", alias="alice-p")
    pr = env.mkprov("rooty", alias="shared-p", shared=True)
    rn = env.mkprov("rooty", alias="root-private")
    ids = lambda who: {x["id"] for x in env.get(who, "/providers").json()}      # noqa: E731
    assert ids("alice") == {pa["id"], pr["id"]}
    assert ids("bob") == {pr["id"]}
    assert ids("rooty") == {pa["id"], pr["id"], rn["id"]}
    # other user's non-shared => 404 on every verb
    for who in ("bob",):
        assert env.patch(who, f"/providers/{pa['id']}", json={"alias": "z"}).status_code == 404
        assert env.delete(who, f"/providers/{pa['id']}").status_code == 404
        assert env.post(who, f"/providers/{pa['id']}/ack").status_code == 404
        assert env.get(who, f"/providers/{pa['id']}/ack-status").status_code == 404
    err(env.patch("bob", f"/providers/{rn['id']}", json={"alias": "z"}), 404, "not_found")
    # shared one: visible, not editable / deletable by users
    err(env.patch("bob", f"/providers/{pr['id']}", json={"alias": "z"}), 403, "forbidden")
    err(env.delete("bob", f"/providers/{pr['id']}"), 403, "forbidden")
    assert env.get("bob", f"/providers/{pr['id']}/ack-status").status_code == 200
    # only root can share
    err(env.post("alice", "/providers", json={"alias": "a", "base_url": "http://localhost:1/v1",
                                              "api_key": KEY, "default_model": "m", "shared": True}), 403, "forbidden")
    err(env.patch("alice", f"/providers/{pa['id']}", json={"shared": True}), 403, "forbidden")
    assert env.patch("rooty", f"/providers/{rn['id']}", json={"shared": True}).json()["shared"] is True
    assert rn["id"] in ids("bob")
    # root may edit another user's provider (rotate key) and delete it
    assert env.patch("rooty", f"/providers/{pa['id']}", json={"api_key": "sk-test-ROOTSET0000"}).status_code == 200
    assert pstore.get_provider(env.cfg, pa["id"])["api_key"].endswith("0000")
    assert env.delete("alice", f"/providers/{pa['id']}").status_code == 204
    assert pa["id"] not in ids("alice") and pa["id"] not in ids("rooty")
    assert pstore.get_provider(env.cfg, pa["id"]) is None
    # unauthenticated
    assert TestClient(env.app).get(f"{API}/providers").status_code == 401


def test_patch_validation_and_defaults(env):
    p = env.mkprov("alice")
    err(env.patch("alice", f"/providers/{p['id']}", json={"base_url": "not a url"}), 422, "validation_error")
    err(env.patch("alice", f"/providers/{p['id']}", json={"api_key": "  "}), 422, "validation_error")
    # omitted fields keep their value
    r = env.patch("alice", f"/providers/{p['id']}", json={"default_model": "m2"}).json()
    assert r["default_model"] == "m2" and r["base_url"] == p["base_url"]
    assert pstore.get_provider(env.cfg, p["id"])["api_key"] == KEY


# ------------------------------------------------------------------ third party
@pytest.mark.parametrize("url,third", [
    ("http://localhost:8000/v1", False), ("http://127.0.0.1:8000/v1", False), ("http://127.5.5.5/v1", False),
    ("http://[::1]:8000/v1", False), ("https://api.example.com/v1", True), ("http://10.0.0.5:8000/v1", True),
    ("https://localhost.evil.com/v1", True), ("http://0.0.0.0:80/v1", True)])
def test_third_party_detection(env, url, third):
    p = env.mkprov("alice", base_url=url)
    assert p["is_third_party"] is third
    st = env.get("alice", f"/providers/{p['id']}/ack-status").json()
    assert st["is_third_party"] is third and st["acked"] is (not third)


def test_ack_flow_per_account_and_base_url_change(env):
    p = env.mkprov("rooty", base_url=EXT, shared=True)
    pid = p["id"]
    st = env.get("alice", f"/providers/{pid}/ack-status").json()
    assert st == {"is_third_party": True, "acked": False, "host": "api.example.com"}
    pa, pb = env.principal("alice"), env.principal("bob")
    assert hooks.needs_third_party_ack(env.cfg, pa, pid) is True
    r = env.post("alice", f"/providers/{pid}/ack")
    assert r.status_code == 200 and r.json()["acked"] is True
    assert env.post("alice", f"/providers/{pid}/ack").status_code == 200       # idempotent
    acks = [u for u in json.loads((env.data / "app" / "users.json").read_text("utf-8"))["users"]
            if u["id"] == env.uid("alice")][0]["tp_ack"]
    assert len(acks) == 1 and acks[0]["provider_id"] == pid and acks[0]["base_url"] == EXT and acks[0]["at"]
    assert hooks.needs_third_party_ack(env.cfg, pa, pid) is False
    assert hooks.needs_third_party_ack(env.cfg, pb, pid) is True               # per account
    assert env.get("bob", f"/providers/{pid}/ack-status").json()["acked"] is False
    # base_url change => re-ack needed; other provider unaffected
    assert env.patch("rooty", f"/providers/{pid}", json={"base_url": "https://other.example.org/v1"}).status_code == 200
    assert hooks.needs_third_party_ack(env.cfg, pa, pid) is True
    assert env.get("alice", f"/providers/{pid}/ack-status").json()["host"] == "other.example.org"
    env.post("alice", f"/providers/{pid}/ack")
    assert hooks.needs_third_party_ack(env.cfg, pa, pid) is False
    acks = [u for u in json.loads((env.data / "app" / "users.json").read_text("utf-8"))["users"]
            if u["id"] == env.uid("alice")][0]["tp_ack"]
    assert len(acks) == 1 and acks[0]["base_url"] == "https://other.example.org/v1"
    # trailing slash is not a different endpoint
    assert third_party.needs_ack(env.cfg, env.uid("alice"), {**pstore.get_provider(env.cfg, pid),
                                                              "base_url": "https://other.example.org/v1/"}) is False
    # moving to localhost => no ack needed anymore
    env.patch("rooty", f"/providers/{pid}", json={"base_url": "http://127.0.0.1:1/v1"})
    assert hooks.needs_third_party_ack(env.cfg, pb, pid) is False


def test_ack_noop_for_local_and_unknown_provider(env):
    p = env.mkprov("alice")
    r = env.post("alice", f"/providers/{p['id']}/ack")
    assert r.status_code == 200 and r.json()["is_third_party"] is False and r.json()["acked"] is True
    users = json.loads((env.data / "app" / "users.json").read_text("utf-8"))["users"]
    assert all(u["tp_ack"] == [] for u in users)
    assert env.post("alice", "/providers/prov_nope/ack").status_code == 404
    assert hooks.needs_third_party_ack(env.cfg, env.principal("alice"), "prov_nope") is False


# ------------------------------------------------------------------ validate hook
def _tpl(env, who, mode="image", name="t"):
    r = env.post(who, "/prompts", json={"name": name, "text": "x {{FEWSHOT}}", "input_mode": mode})
    assert r.status_code == 201, r.text
    return r.json()


def test_validate_hook(env):
    pa = env.mkprov("alice", default_model="alice-model")
    ti, tt = _tpl(env, "alice", "image"), _tpl(env, "alice", "text")
    al = env.principal("alice")
    v = hooks.validate_provider_and_template
    base = {"provider_id": pa["id"], "template_id": ti["id"], "route": "vlm"}
    b = dict(base)
    assert v(env.cfg, al, b) is None
    assert b["model"] == "alice-model"                       # default filled in place
    b = {**base, "model": "custom"}
    v(env.cfg, al, b)
    assert b["model"] == "custom"
    # missing / unknown ids
    for bad in ({**base, "provider_id": ""}, {**base, "template_id": None}):
        with pytest.raises(AppError) as e:
            v(env.cfg, al, dict(bad))
        assert e.value.status == 422
    for bad in ({**base, "provider_id": "prov_nope"}, {**base, "template_id": "tpl_nope"}):
        with pytest.raises(NotFound):
            v(env.cfg, al, dict(bad))
    # other user's provider / template => NotFound
    bob = env.principal("bob")
    with pytest.raises(NotFound):
        v(env.cfg, bob, dict(base))
    pb = env.mkprov("bob")
    with pytest.raises(NotFound):
        v(env.cfg, bob, {"provider_id": pb["id"], "template_id": ti["id"], "route": "vlm"})
    # mode mismatch
    with pytest.raises(AppError) as e:
        v(env.cfg, al, {**base, "template_id": tt["id"]})
    assert (e.value.status, e.value.code) == (422, "template_mode_mismatch")
    with pytest.raises(AppError) as e:
        v(env.cfg, al, {**base, "route": "local_ocr"})
    assert e.value.code == "template_mode_mismatch"
    v(env.cfg, al, {**base, "route": "local_ocr", "template_id": tt["id"]})
    # version must exist
    v(env.cfg, al, {**base, "template_version": 1})
    with pytest.raises(AppError) as e:
        v(env.cfg, al, {**base, "template_version": 7})
    assert e.value.status == 422
    # soft-deleted provider / template
    assert env.delete("alice", f"/prompts/{ti['id']}").status_code == 204
    with pytest.raises(NotFound):
        v(env.cfg, al, dict(base))
    ti2 = _tpl(env, "alice", "image")
    env.delete("alice", f"/providers/{pa['id']}")
    with pytest.raises(NotFound):
        v(env.cfg, al, {**base, "template_id": ti2["id"]})


def test_hooks_registered_and_wired_into_batch_create(env):
    assert hooks.needs_third_party_ack.__module__ == "app.providers.third_party"
    assert hooks.validate_provider_and_template.__module__ == "app.providers.third_party"
    p = env.mkprov("alice", base_url=EXT)
    r = env.post("alice", "/batches/precheck", json={"upload_id": "u_none", "route": "vlm", "provider_id": p["id"]})
    assert r.status_code == 404                              # (upload does not exist) - hook import is sane


# ------------------------------------------------------------------ client
def test_post_chat_ok_and_headers(mock):
    mock.script = [{"content": "hello"}]
    raw = run(llm.post_chat(prov(mock.url + "/"), "m1", [{"role": "user", "content": "hi"}],
                            {"temperature": 0.0, "max_tokens": 5}, 5))
    assert raw.http_status == 200 and raw.error_kind is None and raw.latency_ms >= 0
    assert raw.body["choices"][0]["message"]["content"] == "hello" and raw.text
    req = mock.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"]["Authorization"] == f"Bearer {KEY}"
    assert req["body"] == {"model": "m1", "messages": [{"role": "user", "content": "hi"}],
                           "temperature": 0.0, "max_tokens": 5}
    assert llm.usage(raw.body) == {"prompt": 3, "completion": 2, "total": 5}


def test_post_chat_redirect_not_followed(mock):
    mock.script = [{"status": 302}]
    raw = run(llm.post_chat(prov(mock.url), "m", [], {}, 5))
    assert raw.http_status == 302 and raw.error_kind is None and len(mock.requests) == 1


def test_post_chat_http_error_keeps_text(mock):
    mock.script = [{"status": 500, "body": b"upstream exploded"}]
    raw = run(llm.post_chat(prov(mock.url), "m", [], {}, 5))
    assert raw.http_status == 500 and raw.body is None and raw.text == "upstream exploded"
    mock.script = [{"status": 429, "body": {"error": {"message": "slow down"}}}]
    raw = run(llm.post_chat(prov(mock.url), "m", [], {}, 5))
    assert raw.http_status == 429 and raw.body["error"]["message"] == "slow down"


def test_post_chat_timeout(mock):
    mock.script = [{"delay": 1.5}]
    raw = run(llm.post_chat(prov(mock.url), "m", [], {}, 0.3))
    assert raw.error_kind == "timeout" and raw.http_status is None and raw.body is None
    assert KEY not in (raw.error_detail or "")


def test_post_chat_connect_error_and_no_key_leak():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    raw = run(llm.post_chat(prov(f"http://127.0.0.1:{port}/v1"), "m", [], {}, 15))
    assert raw.error_kind == "connect" and raw.http_status is None
    assert KEY not in (raw.error_detail or "")
    # scrubbing itself
    assert KEY not in llm._scrub(f"boom {KEY} Authorization: Bearer {KEY}", KEY)
    # an invalid URL is a network-class error and must not leak the key either
    raw = run(llm.post_chat(prov("http://", KEY), "m", [], {}, 15))
    assert raw.error_kind in ("network", "connect") and KEY not in (raw.error_detail or "")


def test_post_chat_cancellation_propagates(mock):
    mock.script = [{"delay": 1.0}]

    async def go():
        t = asyncio.create_task(llm.post_chat(prov(mock.url), "m", [], {}, 5))
        await asyncio.sleep(0.2)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
    run(go())


def test_post_chat_uses_given_client(mock):
    import httpx

    async def go():
        async with httpx.AsyncClient() as c:
            r1 = await llm.post_chat(prov(mock.url), "m", [], {}, 5, client=c)
            r2 = await llm.post_chat(prov(mock.url), "m", [], {}, 5, client=c)     # not closed by post_chat
            return r1, r2
    r1, r2 = run(go())
    assert r1.http_status == r2.http_status == 200


def test_extract_message_variants():
    def body(msg, finish="stop"):
        return {"choices": [{"message": msg, "finish_reason": finish}]}
    assert llm.extract_message(body({"content": "abc"})) == ("abc", None, "stop")
    assert llm.extract_message(body({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})) \
        == ("ab", None, "stop")
    assert llm.extract_message(body({"content": "x", "reasoning_content": "why"}, "length")) == ("x", "why", "length")
    assert llm.extract_message(body({"content": "x", "reasoning": "r2"}))[1] == "r2"
    assert llm.extract_message(body({"content": "x", "thinking": "t3"}))[1] == "t3"
    c, r, _ = llm.extract_message(body({"content": "<think>\nplan it\n</think>\n{\"a\": 1}"}))
    assert c == '{"a": 1}' and r == "plan it"
    c, r, _ = llm.extract_message(body({"content": "<think>t</think>answer", "reasoning_content": "rc"}))
    assert c == "answer" and "rc" in r and "t" in r
    assert llm.extract_message(body({"content": None})) == ("", None, "stop")
    for junk in (None, {}, {"choices": []}, {"choices": [None]}, "str", {"choices": [{"message": None}]}):
        c, r, f = llm.extract_message(junk)
        assert c == "" and r is None
    assert llm.usage({}) == {"prompt": 0, "completion": 0, "total": 0}
    assert llm.usage({"usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 9}})["total"] == 9
