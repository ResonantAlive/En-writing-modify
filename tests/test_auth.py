"""Auth / invite / token / admin / audit acceptance tests (self-contained fixtures)."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.auth import ratelimit
from app.core import clock
from app.main import create_app

PW = "correct-horse-battery"
API = "/api/v1"


class RecClient(TestClient):
    """TestClient that records every response body + headers into a shared list."""
    sink: list

    def request(self, *a, **kw):
        r = super().request(*a, **kw)
        self.sink.append(r.text + "\n" + str(dict(r.headers)))
        return r


class Env:
    def __init__(self, data: "Path", app):  # noqa: F821
        self.data = data
        self.app = app
        self.bodies: list[str] = []
        self.client = self.new_client()

    def new_client(self) -> RecClient:
        c = RecClient(self.app)
        c.sink = self.bodies
        return c

    # --- files
    @property
    def app_dir(self):
        return self.data / "app"

    def codes(self) -> dict:
        return json.loads((self.app_dir / "invite_codes.json").read_text("utf-8"))

    def users_json(self) -> dict:
        return json.loads((self.app_dir / "users.json").read_text("utf-8"))

    def auth_state(self) -> dict:
        return json.loads((self.data / "state" / "auth_state.json").read_text("utf-8"))

    def audit_records(self) -> list[dict]:
        out = []
        for f in sorted((self.data / "logs" / "audit").glob("audit-*.jsonl")):
            out += [json.loads(l) for l in f.read_text("utf-8").splitlines() if l.strip()]
        return out

    def audit_text(self) -> str:
        return "\n".join(f.read_text("utf-8") for f in (self.data / "logs" / "audit").glob("audit-*.jsonl"))

    def log_text(self) -> str:
        return "\n".join(f.read_text("utf-8") for f in (self.data / "logs" / "app").glob("app.log*"))

    # --- actions
    def register(self, kind: str, name: str, pw: str = PW, client=None):
        code = self.codes()["root_invite_code" if kind == "root" else "user_invite_code"]
        return (client or self.client).post(f"{API}/auth/register",
                                            json={"invite_code": code, "username": name, "password": pw})

    def make_user(self, kind: str, name: str, pw: str = PW) -> dict:
        r = self.register(kind, name, pw)
        assert r.status_code == 201, r.text
        return r.json()

    def login(self, name: str, pw: str = PW) -> RecClient:
        c = self.new_client()
        r = c.post(f"{API}/auth/login", json={"username": name, "password": pw})
        assert r.status_code == 200, r.text
        c.csrf = r.json()["csrf_token"]
        return c

    @staticmethod
    def h(c) -> dict:
        return {"X-CSRF-Token": c.csrf}

    def make_token(self, c, name="t", role="user", **extra) -> tuple[str, dict]:
        r = c.post(f"{API}/tokens", json={"name": name, "role": role, **extra}, headers=self.h(c))
        assert r.status_code == 201, r.text
        j = r.json()
        return j["token"], j

    def bearer(self, tok: str) -> dict:
        return {"Authorization": f"Bearer {tok}"}


def _build(tmp_path, monkeypatch, settings: dict | None = None, name="data"):
    data = tmp_path / name
    if settings is not None:
        (data / "app").mkdir(parents=True, exist_ok=True)
        (data / "app" / "settings.json").write_text(json.dumps(settings), "utf-8")
    # logging handler is process-global and bound to the first data dir: rebind to this one
    root = logging.getLogger("wm")
    for h in list(root.handlers):
        if getattr(h, "_wm", False):
            root.removeHandler(h)
            h.close()
    return data, create_app(data)


@pytest.fixture
def env(tmp_path, monkeypatch):
    clock.set_now_fn(None)
    data, app = _build(tmp_path, monkeypatch)
    with TestClient(app):                      # runs lifespan (invite codes, executor lock)
        e = Env(data, app)
        yield e
    clock.set_now_fn(None)
    root = logging.getLogger("wm")
    for h in list(root.handlers):
        if getattr(h, "_wm", False):
            root.removeHandler(h)
            h.close()


@pytest.fixture
def root_env(env):
    """env with a root account 'rooty' logged in."""
    env.make_user("root", "rooty")
    env.rc = env.login("rooty")
    return env


def _err(r, status, code):
    assert r.status_code == status, r.text
    j = r.json()
    assert set(j) == {"code", "message", "detail"}
    assert j["code"] == code
    return j


# ======================================================================== invite codes
def test_invite_file_created_and_not_overwritten(tmp_path, monkeypatch):
    clock.set_now_fn(None)
    data, app = _build(tmp_path, monkeypatch)
    assert not (data / "app" / "invite_codes.json").exists()
    with TestClient(app):
        pass
    p = data / "app" / "invite_codes.json"
    first = p.read_text("utf-8")
    j = json.loads(first)
    assert j["root_invite_code"] and j["user_invite_code"] and j["root_invite_code"] != j["user_invite_code"]
    data2, app2 = _build(tmp_path, monkeypatch)
    assert data2 == data
    with TestClient(app2):
        pass
    assert p.read_text("utf-8") == first


def test_user_code_gives_user_root_code_gives_root(env):
    assert env.make_user("user", "bob")["role"] == "user"
    assert env.make_user("root", "alice")["role"] == "root"


def test_wrong_invite_403_generic(env):
    r = env.client.post(f"{API}/auth/register",
                        json={"invite_code": "AAAA-BBBB", "username": "zed", "password": PW})
    j = _err(r, 403, "invalid_invite")
    assert "root" not in j["message"].lower() and "user" not in j["message"].lower()


def test_invite_codes_never_leak(env):
    codes = env.codes()
    secret = [codes["root_invite_code"], codes["user_invite_code"]]
    env.make_user("root", "rooty")
    env.make_user("user", "bob")
    rc = env.login("rooty")
    env.client.post(f"{API}/auth/register", json={"invite_code": "WRONG", "username": "x1x", "password": PW})
    rc.get(f"{API}/admin/users")
    rc.post(f"{API}/admin/invite-codes/reset", json={"which": "both"}, headers=env.h(rc))
    new = env.codes()
    secret += [new["root_invite_code"], new["user_invite_code"]]
    blob = "\n".join(env.bodies) + env.log_text() + env.audit_text()
    for s in secret:
        assert s not in blob
        assert s.replace("-", "") not in blob
    # sanity: logs were actually produced in this data dir
    assert "invite codes saved to" in env.log_text() or env.audit_text()
    assert json.dumps(secret) not in blob


def test_reset_endpoint_root_only_and_response(root_env):
    e = root_env
    e.make_user("user", "bob")
    uc = e.login("bob")
    _err(uc.post(f"{API}/admin/invite-codes/reset", json={"which": "both"}, headers=e.h(uc)), 403, "forbidden")
    r = e.rc.post(f"{API}/admin/invite-codes/reset", json={"which": "both"}, headers=e.h(e.rc))
    assert r.status_code == 200
    new = e.codes()
    assert new["root_invite_code"] not in r.text and new["user_invite_code"] not in r.text
    assert "invite_codes.json" in r.json()["message"]
    _err(e.rc.post(f"{API}/admin/invite-codes/reset", json={"which": "nope"}, headers=e.h(e.rc)),
         422, "invalid_which")


def test_reset_old_invalid_new_works_accounts_unaffected(root_env):
    e = root_env
    e.make_user("user", "bob")
    old = e.codes()
    e.rc.post(f"{API}/admin/invite-codes/reset", json={"which": "both"}, headers=e.h(e.rc))
    new = e.codes()
    assert new["user_invite_code"] != old["user_invite_code"]
    assert new["root_invite_code"] != old["root_invite_code"]
    c = e.new_client()
    for k in ("user_invite_code", "root_invite_code"):
        r = c.post(f"{API}/auth/register", json={"invite_code": old[k], "username": "late", "password": PW})
        _err(r, 403, "invalid_invite")
    r = e.register("user", "newbie", client=c)
    assert r.status_code == 201 and r.json()["role"] == "user"
    r = e.register("root", "newroot", client=c)
    assert r.status_code == 201 and r.json()["role"] == "root"
    e.login("bob")          # existing accounts unaffected
    e.login("rooty")


def test_reset_only_root_keeps_user_code(root_env):
    e = root_env
    old = e.codes()
    e.rc.post(f"{API}/admin/invite-codes/reset", json={"which": "root"}, headers=e.h(e.rc))
    new = e.codes()
    assert new["user_invite_code"] == old["user_invite_code"]
    assert new["root_invite_code"] != old["root_invite_code"]
    e.rc.post(f"{API}/admin/invite-codes/reset", json={"which": "user"}, headers=e.h(e.rc))
    newer = e.codes()
    assert newer["root_invite_code"] == new["root_invite_code"]
    assert newer["user_invite_code"] != new["user_invite_code"]


def test_invite_bruteforce_429(env):
    for _ in range(5):
        r = env.client.post(f"{API}/auth/register",
                            json={"invite_code": "NOPE", "username": "zzz1", "password": PW})
        assert r.status_code == 403
    # even the right code is now refused
    r = env.register("user", "zzz1")
    _err(r, 429, "too_many_attempts")


# ======================================================================== username / password
@pytest.mark.parametrize("name", ["ab", "a" * 33, "bad name", "bad.name", "名字名字", "a@b", ""])
def test_username_invalid(env, name):
    r = env.register("user", name)
    assert r.status_code == 422
    assert set(r.json()) == {"code", "message", "detail"}


@pytest.mark.parametrize("name", ["abc", "a" * 32, "A_b-9"])
def test_username_valid(env, name):
    assert env.register("user", name).status_code == 201


def test_username_case_insensitive_unique(env):
    assert env.register("user", "Alice").status_code == 201
    _err(env.register("user", "alice"), 409, "username_taken")
    _err(env.register("root", "ALICE"), 409, "username_taken")


@pytest.mark.parametrize("pw,ok", [("a" * 7, False), ("a" * 8, True), ("a" * 256, True), ("a" * 257, False)])
def test_password_length(env, pw, ok):
    r = env.register("user", "pwuser", pw)
    assert r.status_code == (201 if ok else 422)


def test_password_stored_argon2id(env):
    env.make_user("user", "bob", "my-secret-pass-123")
    raw = (env.app_dir / "users.json").read_text("utf-8")
    assert "my-secret-pass-123" not in raw
    assert env.users_json()["users"][0]["password_hash"].startswith("$argon2id$")


# ======================================================================== login / lockout
def test_login_failure_message_identical(env):
    env.make_user("user", "bob")
    c = env.new_client()
    r1 = c.post(f"{API}/auth/login", json={"username": "ghost", "password": "whatever123"})
    r2 = c.post(f"{API}/auth/login", json={"username": "bob", "password": "wrong-password"})
    assert r1.status_code == r2.status_code == 401
    assert r1.json() == r2.json()
    assert r1.json()["code"] == "invalid_credentials"


def test_login_lockout_after_5_even_with_correct_password(env):
    env.make_user("user", "bob")
    c = env.new_client()
    for _ in range(5):
        assert c.post(f"{API}/auth/login", json={"username": "bob", "password": "bad-password"}).status_code == 401
    _err(c.post(f"{API}/auth/login", json={"username": "bob", "password": PW}), 429, "too_many_attempts")
    st = env.auth_state()
    (key, entry), = st["login"].items()
    assert key == "bob|testclient" and entry["fails"] == 5 and entry["locked_until_ts"] > 0


def test_lock_expiry(env, monkeypatch):
    env.make_user("user", "bob")
    c = env.new_client()
    for _ in range(5):
        c.post(f"{API}/auth/login", json={"username": "bob", "password": "bad-password"})
    assert c.post(f"{API}/auth/login", json={"username": "bob", "password": PW}).status_code == 429
    import time
    real = time.time
    monkeypatch.setattr(ratelimit, "_time", lambda: real() + 11 * 60)   # lock is 10 min
    assert c.post(f"{API}/auth/login", json={"username": "bob", "password": PW}).status_code == 200


def test_success_resets_counter(env):
    env.make_user("user", "bob")
    c = env.new_client()
    for _ in range(3):
        c.post(f"{API}/auth/login", json={"username": "bob", "password": "bad-password"})
    assert env.auth_state()["login"]["bob|testclient"]["fails"] == 3
    assert c.post(f"{API}/auth/login", json={"username": "bob", "password": PW}).status_code == 200
    assert "bob|testclient" not in env.auth_state()["login"]
    for _ in range(4):      # would have locked if counter had not been reset (3 + 4 >= 5)
        assert c.post(f"{API}/auth/login", json={"username": "bob", "password": "bad-password"}).status_code == 401
    assert env.auth_state()["login"]["bob|testclient"]["fails"] == 4


# ======================================================================== cookies / CSRF
def test_cookie_flags(env):
    env.make_user("user", "bob")
    r = env.new_client().post(f"{API}/auth/login", json={"username": "bob", "password": PW})
    cookies = r.headers.get_list("set-cookie")
    sess = next(c for c in cookies if c.startswith("wm_session="))
    csrf = next(c for c in cookies if c.startswith("wm_csrf="))
    assert "httponly" in sess.lower()
    assert "samesite=lax" in sess.lower()
    assert "secure" not in sess.lower()
    assert "httponly" not in csrf.lower()
    assert "secure" not in csrf.lower()


def test_csrf_enforced_for_cookie_auth(env):
    env.make_user("user", "bob")
    c = env.login("bob")
    body = {"name": "t", "role": "user"}
    _err(c.post(f"{API}/tokens", json=body), 403, "csrf_failed")
    _err(c.post(f"{API}/tokens", json=body, headers={"X-CSRF-Token": "wrong"}), 403, "csrf_failed")
    assert c.get(f"{API}/auth/me").status_code == 200          # GET needs none
    assert c.post(f"{API}/tokens", json=body, headers=env.h(c)).status_code == 201


def test_bearer_post_needs_no_csrf(env):
    env.make_user("user", "bob")
    c = env.login("bob")
    tok, _ = env.make_token(c)
    plain = env.new_client()
    r = plain.post(f"{API}/tokens", json={"name": "x"}, headers=env.bearer(tok))
    assert r.status_code == 201
    assert plain.get(f"{API}/auth/me", headers=env.bearer(tok)).json()["via"] == "bearer"


def test_logout_invalidates_session(env):
    env.make_user("user", "bob")
    c = env.login("bob")
    sess = c.cookies.get("wm_session")
    assert sess
    assert c.post(f"{API}/auth/logout", headers=env.h(c)).status_code == 200
    stale = env.new_client()
    stale.cookies.set("wm_session", sess)
    _err(stale.get(f"{API}/auth/me"), 401, "unauthenticated")
    assert env.users_json()["users"][0]["sessions"] == []


# ======================================================================== tokens
def test_token_plaintext_only_at_creation(env):
    env.make_user("user", "bob")
    c = env.login("bob")
    tok, created = env.make_token(c, "mine")
    assert tok.startswith("wm_") and created["last4"] == tok[-4:]
    lst = c.get(f"{API}/tokens")
    me = c.get(f"{API}/auth/me")
    assert tok not in lst.text and tok not in me.text
    assert all("token" not in t and "hash" not in t for t in lst.json())
    raw = (env.app_dir / "users.json").read_text("utf-8")
    assert tok not in raw
    assert hashlib.sha256(tok.encode()).hexdigest() in raw
    assert tok not in env.log_text() + env.audit_text()


def test_token_role_limits(env):
    env.make_user("user", "bob")
    env.make_user("root", "rooty")
    bc, rc = env.login("bob"), env.login("rooty")
    r = bc.post(f"{API}/tokens", json={"name": "t", "role": "root"}, headers=env.h(bc))
    _err(r, 403, "role_exceeds_creator")
    assert env.make_token(rc, "r", "root")[1]["role"] == "root"
    assert env.make_token(rc, "u", "user")[1]["role"] == "user"
    _err(rc.post(f"{API}/tokens", json={"name": "t", "role": "god"}, headers=env.h(rc)), 422, "invalid_role")
    _err(rc.post(f"{API}/tokens", json={"name": "t", "expires_in_days": 0}, headers=env.h(rc)), 422, "invalid_expiry")


def test_effective_role_is_min(env):
    env.make_user("root", "root1")
    env.make_user("root", "root2")
    c1, c2 = env.login("root1"), env.login("root2")
    tok, _ = env.make_token(c1, "rt", "root")
    plain = env.new_client()
    assert plain.get(f"{API}/admin/users", headers=env.bearer(tok)).status_code == 200
    uid1 = next(u["id"] for u in c2.get(f"{API}/admin/users").json() if u["username"] == "root1")
    assert c2.post(f"{API}/admin/users/{uid1}/role", json={"role": "user"}, headers=env.h(c2)).status_code == 200
    _err(plain.get(f"{API}/admin/users", headers=env.bearer(tok)), 403, "forbidden")
    me = plain.get(f"{API}/auth/me", headers=env.bearer(tok)).json()
    assert me["role"] == "user" and me["account_role"] == "user"


def test_disable_kills_sessions_and_tokens_enable_restores_tokens_only(root_env):
    e = root_env
    ub = e.make_user("user", "bob")
    bc = e.login("bob")
    tok, _ = e.make_token(bc)
    plain = e.new_client()
    assert plain.get(f"{API}/auth/me", headers=e.bearer(tok)).status_code == 200
    assert e.rc.post(f"{API}/admin/users/{ub['id']}/disable", headers=e.h(e.rc)).status_code == 200
    _err(bc.get(f"{API}/auth/me"), 401, "unauthenticated")
    _err(plain.get(f"{API}/auth/me", headers=e.bearer(tok)), 401, "unauthenticated")
    # cannot log in while disabled
    _err(e.new_client().post(f"{API}/auth/login", json={"username": "bob", "password": PW}), 401, "invalid_credentials")
    assert e.rc.post(f"{API}/admin/users/{ub['id']}/enable", headers=e.h(e.rc)).status_code == 200
    assert plain.get(f"{API}/auth/me", headers=e.bearer(tok)).status_code == 200      # token restored
    assert bc.get(f"{API}/auth/me").status_code == 401                                # session not restored


def test_revoke_rules(root_env):
    e = root_env
    e.make_user("user", "bob")
    e.make_user("user", "carol")
    bc, cc = e.login("bob"), e.login("carol")
    btok, brec = e.make_token(bc, "b")
    ctok, crec = e.make_token(cc, "c")
    # user cannot revoke another account's token
    _err(bc.delete(f"{API}/tokens/{crec['id']}", headers=e.h(bc)), 404, "not_found")
    assert e.new_client().get(f"{API}/auth/me", headers=e.bearer(ctok)).status_code == 200
    # revoke own
    r = bc.delete(f"{API}/tokens/{brec['id']}", headers=e.h(bc))
    assert r.status_code == 200 and r.json()["revoked"] is True
    _err(e.new_client().get(f"{API}/auth/me", headers=e.bearer(btok)), 401, "unauthenticated")
    # root lists all (incl. owners) and revokes any
    allt = e.rc.get(f"{API}/admin/tokens").json()
    assert {t["id"] for t in allt} >= {brec["id"], crec["id"]}
    assert {t["owner_username"] for t in allt} >= {"bob", "carol"}
    assert e.rc.delete(f"{API}/admin/tokens/{crec['id']}", headers=e.h(e.rc)).status_code == 200
    _err(e.new_client().get(f"{API}/auth/me", headers=e.bearer(ctok)), 401, "unauthenticated")
    _err(e.rc.delete(f"{API}/admin/tokens/tok_nonexistent", headers=e.h(e.rc)), 404, "not_found")
    # non-root cannot use admin listing
    _err(bc.get(f"{API}/admin/tokens"), 403, "forbidden")


def test_expired_token_rejected(env):
    env.make_user("user", "bob")
    c = env.login("bob")
    tok, _ = env.make_token(c, "exp", expires_in_days=1)
    plain = env.new_client()
    assert plain.get(f"{API}/auth/me", headers=env.bearer(tok)).status_code == 200
    base = datetime.now().astimezone()
    clock.set_now_fn(lambda: base + timedelta(days=2))
    _err(plain.get(f"{API}/auth/me", headers=env.bearer(tok)), 401, "unauthenticated")


def test_last_used_throttled(env):
    env.make_user("user", "bob")
    c = env.login("bob")
    tok, rec = env.make_token(c)
    assert rec["last_used"] is None
    plain = env.new_client()
    t0 = datetime.now().astimezone()
    now = {"t": t0}
    clock.set_now_fn(lambda: now["t"])

    def last_used():
        return env.users_json()["users"][0]["tokens"][0]["last_used"]

    assert plain.get(f"{API}/auth/me", headers=env.bearer(tok)).status_code == 200
    first = last_used()
    assert first == clock.iso(t0)
    mtime = (env.app_dir / "users.json").stat().st_mtime_ns
    now["t"] = t0 + timedelta(seconds=30)
    assert plain.get(f"{API}/auth/me", headers=env.bearer(tok)).status_code == 200
    assert last_used() == first                                     # throttled: no write
    assert (env.app_dir / "users.json").stat().st_mtime_ns == mtime
    now["t"] = t0 + timedelta(seconds=400)
    plain.get(f"{API}/auth/me", headers=env.bearer(tok))
    assert last_used() == clock.iso(now["t"])                       # interval elapsed: written


# ======================================================================== admin
def test_non_root_forbidden_on_admin(env):
    env.make_user("user", "bob")
    bc = env.login("bob")
    for method, path, kw in [("get", "/admin/users", {}), ("get", "/admin/tokens", {}),
                             ("post", "/admin/users/x/disable", {}), ("post", "/admin/users/x/enable", {}),
                             ("post", "/admin/users/x/role", {"json": {"role": "root"}}),
                             ("post", "/admin/invite-codes/reset", {"json": {"which": "both"}}),
                             ("delete", "/admin/tokens/x", {})]:
        r = getattr(bc, method)(API + path, headers=env.h(bc), **kw)
        _err(r, 403, "forbidden")
    _err(env.new_client().get(f"{API}/admin/users"), 401, "unauthenticated")


def test_last_root_protection(env):
    r1 = env.make_user("root", "root1")
    c1 = env.login("root1")
    _err(c1.post(f"{API}/admin/users/{r1['id']}/disable", headers=env.h(c1)), 409, "last_root")
    _err(c1.post(f"{API}/admin/users/{r1['id']}/role", json={"role": "user"}, headers=env.h(c1)), 409, "last_root")
    r2 = env.make_user("root", "root2")
    # another root exists: allowed
    assert c1.post(f"{API}/admin/users/{r2['id']}/disable", headers=env.h(c1)).status_code == 200
    # r2 is disabled now, r1 is again the last enabled root
    _err(c1.post(f"{API}/admin/users/{r1['id']}/role", json={"role": "user"}, headers=env.h(c1)), 409, "last_root")
    assert c1.post(f"{API}/admin/users/{r2['id']}/enable", headers=env.h(c1)).status_code == 200
    assert c1.post(f"{API}/admin/users/{r1['id']}/role", json={"role": "user"}, headers=env.h(c1)).status_code == 200
    _err(c1.post(f"{API}/admin/users/nope/disable", headers=env.h(c1)), 403, "forbidden")   # r1 now plain user


def test_disable_second_root_then_self_blocked_unknown_user_404(root_env):
    e = root_env
    _err(e.rc.post(f"{API}/admin/users/u_missing/disable", headers=e.h(e.rc)), 404, "not_found")
    _err(e.rc.post(f"{API}/admin/users/u_missing/role", json={"role": "bogus"}, headers=e.h(e.rc)), 422, "invalid_role")


# ======================================================================== audit
def test_audit_records_and_no_secrets(root_env):
    e = root_env
    ub = e.make_user("user", "bob")
    bc = e.login("bob")
    e.new_client().post(f"{API}/auth/login", json={"username": "bob", "password": "wrong-password-xyz"})
    tok, rec = e.make_token(bc, "k")
    bc.delete(f"{API}/tokens/{rec['id']}", headers=e.h(bc))
    e.rc.post(f"{API}/admin/invite-codes/reset", json={"which": "user"}, headers=e.h(e.rc))
    e.rc.post(f"{API}/admin/users/{ub['id']}/role", json={"role": "root"}, headers=e.h(e.rc))
    e.rc.post(f"{API}/admin/users/{ub['id']}/role", json={"role": "user"}, headers=e.h(e.rc))
    actions = {r["action"] for r in e.audit_records()}
    assert {"auth.register", "auth.login", "auth.login_failed", "token.create", "token.revoke",
            "admin.invite_reset", "admin.role_change"} <= actions
    assert any(r["action"] == "auth.register" and r["detail"] == {"role": "user"} for r in e.audit_records())
    assert any(r["action"] == "admin.role_change" and r["target"] == ub["id"] for r in e.audit_records())
    files = list((e.data / "logs" / "audit").glob("audit-*.jsonl"))
    assert files and all(len(f.stem) == len("audit-YYYYMMDD") for f in files)
    codes = e.codes()
    text = e.audit_text()
    for secret in (PW, "wrong-password-xyz", tok, codes["root_invite_code"], codes["user_invite_code"]):
        assert secret not in text


# ======================================================================== error format
def test_unified_error_format(env):
    _err(env.new_client().get(f"{API}/auth/me"), 401, "unauthenticated")
    j = env.new_client().get(f"{API}/no/such/route")
    _err(j, 404, "not_found")
    # malformed / missing body -> 422
    r = env.client.post(f"{API}/auth/login", json={"username": "x"})
    j = _err(r, 422, "validation_error")
    assert isinstance(j["detail"], list) and j["detail"]
    # bad bearer scheme / token -> 401
    _err(env.new_client().get(f"{API}/auth/me", headers={"Authorization": "Basic abc"}), 401, "unauthenticated")
    _err(env.new_client().get(f"{API}/auth/me", headers={"Authorization": "Bearer wm_bogus"}), 401, "unauthenticated")
    # app-level 404 via revoke
    env.make_user("user", "bob")
    bc = env.login("bob")
    _err(bc.delete(f"{API}/tokens/tok_nope", headers=env.h(bc)), 404, "not_found")
    _err(env.client.post(f"{API}/auth/register",
                         json={"invite_code": env.codes()["user_invite_code"], "username": "x", "password": PW}),
         422, "invalid_username")


# ======================================================================== trusted proxy
def _fail(c, n, xff=None, name="bob", pw="bad-password"):
    codes = []
    for i in range(n):
        h = {"X-Forwarded-For": xff(i) if callable(xff) else xff} if xff else {}
        codes.append(c.post(f"{API}/auth/login", json={"username": name, "password": pw}, headers=h).status_code)
    return codes


def test_trusted_proxy_uses_xff(tmp_path, monkeypatch):
    clock.set_now_fn(None)
    data, app = _build(tmp_path, monkeypatch, {"network": {"trusted_proxies": ["testclient"]}})
    with TestClient(app):
        e = Env(data, app)
        e.make_user("user", "bob")
        c = e.new_client()
        assert _fail(c, 5, "1.2.3.4") == [401] * 5
        keys = list(e.auth_state()["login"])
        assert keys == ["bob|1.2.3.4"]
        # same real client IP via XFF is locked even with the right password
        h = {"X-Forwarded-For": "1.2.3.4"}
        assert c.post(f"{API}/auth/login", json={"username": "bob", "password": PW}, headers=h).status_code == 429
        # a different client behind the proxy is unaffected
        h2 = {"X-Forwarded-For": "5.6.7.8"}
        assert c.post(f"{API}/auth/login", json={"username": "bob", "password": PW}, headers=h2).status_code == 200
        # rightmost untrusted hop is used: a client-supplied leftmost value cannot dodge the lock
        h3 = {"X-Forwarded-For": "9.9.9.9, 1.2.3.4"}
        assert c.post(f"{API}/auth/login", json={"username": "bob", "password": PW}, headers=h3).status_code == 429
    clock.set_now_fn(None)


def test_untrusted_peer_ignores_xff(env):
    env.make_user("user", "bob")
    c = env.new_client()
    codes = _fail(c, 5, lambda i: f"10.0.0.{i + 1}")
    assert codes == [401] * 5
    assert list(env.auth_state()["login"]) == ["bob|testclient"]
    r = c.post(f"{API}/auth/login", json={"username": "bob", "password": PW},
               headers={"X-Forwarded-For": "7.7.7.7"})
    _err(r, 429, "too_many_attempts")
