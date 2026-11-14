"""Permission matrix (spec section 7): every route that takes an object id is enumerated from the app's OpenAPI
schema and called by another account with alice's real ids. Expected: 404 (or 403 for root-only routes), with a
response identical to the one for an id that does not exist, so a foreign object is indistinguishable from a
missing one. Plus: ids smuggled in request bodies, list filtering, root access, root's user-role token and
unauthenticated access. A new route with an id parameter fails `test_matrix_covers_every_id_route` until it is
added here."""
from __future__ import annotations

import io
import json
import re

import pytest
from fastapi.testclient import TestClient

from app.batches import paths, store
from app.sheets import store as sheets_store
from app.staging import uploads
from tests.fixtures_images import make_image
from tests.mock_llm_server import mock_llm  # noqa: F401
from tests.test_batches import API, H
from tests.test_m5 import T_TEXT, app_env, create_batch, review, setup_account, upload, wait_done  # noqa: F401

ROOT_ONLY = re.compile(r"^/api/v1/admin/|/share$")
NO_AUTH = {("POST", "/api/v1/auth/login"), ("POST", "/api/v1/auth/register"), ("POST", "/api/v1/auth/logout")}
STREAMING = {("GET", "/api/v1/batches/{task_code}/events")}          # SSE: never opened by the owner here

BODIES = {
    ("POST", "/api/v1/batches/{task_code}/config"): {"model": "m2"},
    ("POST", "/api/v1/batches/{task_code}/subtasks/{subtask_id}/review"): {"final_score": 1},
    ("POST", "/api/v1/prompts/{template_id}/versions"): {"text": T_TEXT + " v2"},
    ("POST", "/api/v1/prompts/{template_id}/share"): {"shared": True},
    ("PATCH", "/api/v1/providers/{provider_id}"): {"default_model": "hijack"},
    ("PATCH", "/api/v1/sheet-templates/{tid}"): {"name": "hijack"},
    ("PATCH", "/api/v1/sheet-templates/{tid}/pages/{pid}"): {"disabled": True},
    ("POST", "/api/v1/sheet-templates/{tid}/share"): {"shared": True},
    ("POST", "/api/v1/uploads/{upload_id}/grouping/preview"): {},
    ("POST", "/api/v1/admin/users/{user_id}/role"): {"role": "root"},
}
MULTIPART = {("POST", "/api/v1/sheet-templates/{tid}/pages")}


def ghost(v: str) -> str:
    """A well-formed id that does not exist: same shape, last character changed."""
    last = v[-1]
    if last.isdigit():
        repl = "8" if last != "8" else "7"
    elif last.isupper():
        repl = "Q" if last != "Q" else "R"
    else:
        repl = "q" if last != "q" else "r"
    return v[:-1] + repl


def id_routes(app) -> list[tuple[str, str]]:
    out = []
    for path, ops in app.openapi()["paths"].items():
        if "{" not in path:
            continue
        out += [(m.upper(), path) for m in ops]
    return sorted(out)


def call(c, method, tmpl, ids, headers):
    url = tmpl.format(**ids)
    if (method, tmpl) in MULTIPART:
        files = [("files", ("blank.png", io.BytesIO(make_image("png", size=(400, 300))), "image/png"))]
        return c.request(method, url, files=files, headers=headers)
    body = BODIES.get((method, tmpl))
    if method in ("POST", "PATCH", "PUT"):
        return c.request(method, url, json=body if body is not None else {}, headers=headers)
    return c.request(method, url, headers=headers)


def norm(r):
    try:
        j = r.json()
    except ValueError:
        return r.status_code, r.text
    if isinstance(j, dict):
        j = {k: v for k, v in j.items() if k not in ("request_id",)}
    return r.status_code, j


@pytest.fixture
def world(app_env):
    """alice owns one of everything; bob and the root account (rooty) exist with their own provider."""
    e = app_env
    a = e.clients["alice"]
    ids_a = setup_account(e, "alice")
    e.mock.script(lambda req, n: "ok_json")
    tc = create_batch(e, "alice", ids_a, n=1, zero_shot_approved=True)
    wait_done(e, "alice", tc)
    sid = store.list_subtask_ids(e.cfg, tc)[0]
    assert review(e, "alice", tc, sid, {"adopt_ai": True}).status_code == 200
    up = upload(a, 2)
    file_id = a.get(f"{API}/uploads/{up}", headers=H(a)).json()["files"][0]["file_id"]
    tok = a.post(f"{API}/tokens", json={"name": "alice-cli", "role": "user"}, headers=H(a)).json()
    pages = a.get(f"{API}/sheet-templates/{ids_a[2]}", headers=H(a)).json()["page_list"]
    img = sheets_store.files_dir(e.cfg, ids_a[2]) / f"{pages[0]['page_id']}.jpg"     # fixture pages have no file
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(make_image("jpeg", size=(1100, 2000)))
    e.alice_ids = {"task_code": tc, "subtask_id": sid, "n": "1", "k": "1",
                   "template_id": ids_a[1], "provider_id": ids_a[0], "tid": ids_a[2], "pid": pages[0]["page_id"],
                   "token_id": tok["id"], "upload_id": up, "file_id": file_id, "user_id": e.uid("alice")}
    e.ids_a = ids_a
    e.ids_b = setup_account(e, "bob")
    b = e.clients["bob"]
    e.bob_bearer = {"Authorization": "Bearer " + b.post(f"{API}/tokens", json={"name": "bob-cli", "role": "user"},
                                                        headers=H(b)).json()["token"]}
    return e


def snapshot(e):
    """Everything a foreign write could have touched."""
    ids = e.alice_ids
    sdir = paths.subtask_dir(e.cfg, ids["subtask_id"])
    files = [paths.batch_dir(e.cfg, ids["task_code"]) / "batch.json", sdir / "meta.json", sdir / "review.json",
             e.cfg.app_dir / "providers.json", e.cfg.app_dir / "prompts.json", e.cfg.app_dir / "sheet_templates.json"]
    snap = {p.name: p.read_bytes() for p in files}                 # all exist (KeyError/FileNotFound otherwise)
    users = json.loads((e.cfg.app_dir / "users.json").read_text("utf-8"))["users"]
    snap["alice"] = next(u for u in users if u["id"] == e.uid("alice"))      # bob's own token use is recorded
    up = uploads.upload_dir(e.cfg, ids["upload_id"])
    snap["staging"] = sorted(str(p.relative_to(up)) for p in up.rglob("*"))
    assert snap["staging"]
    return snap


def test_matrix_covers_every_id_route(world):
    e = world
    params = set()
    for _, tmpl in id_routes(e.app):
        params |= set(re.findall(r"{(\w+)}", tmpl))
    assert params <= set(e.alice_ids), params - set(e.alice_ids)


def test_foreign_ids_are_indistinguishable_from_missing(world):
    e = world
    b = e.clients["bob"]
    missing = {k: ghost(v) for k, v in e.alice_ids.items()}
    before = snapshot(e)
    checked = 0
    for method, tmpl in id_routes(e.app):
        expected = 403 if ROOT_ONLY.search(tmpl) else 404
        for label, client, headers in (("cookie", b, H(b)), ("bearer", TestClient(e.app), e.bob_bearer)):
            foreign = call(client, method, tmpl, e.alice_ids, headers)
            ghosted = call(client, method, tmpl, missing, headers)
            assert foreign.status_code == expected, (label, method, tmpl, foreign.status_code, foreign.text[:200])
            assert norm(foreign) == norm(ghosted), (label, method, tmpl, norm(foreign), norm(ghosted))
            if expected == 404:
                assert foreign.json()["code"] == "not_found"
            checked += 1
    assert checked == 2 * len(id_routes(e.app)) >= 80
    assert snapshot(e) == before                                  # nothing of alice's changed


def test_owner_reaches_every_id_route_read(world):
    """Non-vacuity: the ids used above are real; alice (owner) gets 200 on every read route."""
    e = world
    a = e.clients["alice"]
    for method, tmpl in id_routes(e.app):
        if method != "GET" or ROOT_ONLY.search(tmpl) or (method, tmpl) in STREAMING:
            continue
        r = call(a, method, tmpl, e.alice_ids, H(a))
        assert r.status_code == 200, (tmpl, r.status_code, r.text[:200])


def test_foreign_ids_in_request_bodies(world):
    e = world
    b = e.clients["bob"]
    prov_a, tpl_a, sheet_a = e.ids_a
    prov_b, tpl_b, sheet_b = e.ids_b
    up_b = upload(b, 1)
    base = {"upload_id": up_b, "topic": "t", "rubric": "r", "genre": "letter", "full_score": 20,
            "route": "local_ocr", "provider_id": prov_b, "model": "m1", "template_id": tpl_b,
            "sheet_template_id": sheet_b, "grouping": {"multi_photo": False}, "zero_shot_approved": True}
    for field, foreign in (("upload_id", e.alice_ids["upload_id"]), ("provider_id", prov_a),
                           ("template_id", tpl_a), ("sheet_template_id", sheet_a)):
        for route in ("batches/precheck", "batches"):
            r = b.post(f"{API}/{route}", json=base | {field: foreign}, headers=H(b))
            g = b.post(f"{API}/{route}", json=base | {field: ghost(foreign)}, headers=H(b))
            if route == "batches/precheck" and field == "sheet_template_id":
                # precheck reports sheet problems instead of rejecting; foreign == missing
                assert r.status_code == 200 and r.json()["sheet_template_ok"] is False, r.text[:200]
            elif field == "sheet_template_id":
                assert r.status_code == 422 and r.json()["code"] == "invalid_sheet_template", r.text[:200]
            else:
                assert r.status_code == 404, (route, field, r.status_code, r.text[:200])
                assert r.json()["code"] == "not_found"
            assert norm(r) == norm(g), (route, field, norm(r), norm(g))
    assert [x["owner_id"] for x in store.iter_batches(e.cfg)] == [e.uid("alice")]       # nothing created for bob
    r = b.post(f"{API}/prompts/assist", json={"provider_id": prov_a, "requirement": "write a template"},
               headers=H(b))
    assert r.status_code == 404
    # bob's own batch: switching it to alice's provider / template is refused the same way
    e.mock.script(lambda req, n: "ok_json")
    tc_b = create_batch(e, "bob", e.ids_b, n=1, base_w=700, zero_shot_approved=True)
    wait_done(e, "bob", tc_b)
    for body in ({"provider_id": prov_a}, {"template_id": tpl_a}):
        r = b.post(f"{API}/batches/{tc_b}/config", json=body, headers=H(b))
        assert r.status_code == 404, (body, r.status_code, r.text[:200])
    assert len(store.read_batch(e.cfg, tc_b)["config_revisions"]) == 1
    # the alice upload is still alice's and unused
    a = e.clients["alice"]
    assert a.get(f"{API}/uploads/{e.alice_ids['upload_id']}", headers=H(a)).status_code == 200


def test_lists_only_show_own_objects(world):
    e = world
    b = e.clients["bob"]
    ids = e.alice_ids
    assert ids["task_code"] not in json.dumps(b.get(f"{API}/batches", headers=H(b)).json())
    assert ids["task_code"] not in json.dumps(b.get(f"{API}/batches?all=true", headers=H(b)).json())
    assert ids["provider_id"] not in json.dumps(b.get(f"{API}/providers", headers=H(b)).json())
    assert ids["template_id"] not in json.dumps(b.get(f"{API}/prompts", headers=H(b)).json())
    assert ids["tid"] not in json.dumps(b.get(f"{API}/sheet-templates", headers=H(b)).json())
    assert ids["token_id"] not in json.dumps(b.get(f"{API}/tokens", headers=H(b)).json())
    # alice's own lists do show them (and never the provider key)
    a = e.clients["alice"]
    provs = a.get(f"{API}/providers", headers=H(a)).json()
    assert ids["provider_id"] in json.dumps(provs) and "sk-test-0000-1111" not in json.dumps(provs)


def test_root_access_and_roots_user_token(world):
    e = world
    root = e.clients["rooty"]
    ids = e.alice_ids
    tc, sid = ids["task_code"], ids["subtask_id"]
    for url in (f"/batches/{tc}", f"/batches/{tc}/subtasks", f"/batches/{tc}/subtasks/{sid}",
                f"/batches/{tc}/subtasks/{sid}/review", f"/batches/{tc}/result", f"/batches/{tc}/subtasks/{sid}/image/1",
                f"/prompts/{ids['template_id']}", f"/sheet-templates/{ids['tid']}",
                f"/providers/{ids['provider_id']}/ack-status"):
        assert root.get(API + url, headers=H(root)).status_code == 200, url
    assert tc in json.dumps(root.get(f"{API}/batches?all=true", headers=H(root)).json())
    assert tc not in json.dumps(root.get(f"{API}/batches", headers=H(root)).json())      # default: own only
    # staging uploads and personal tokens stay private even for root
    assert root.get(f"{API}/uploads/{ids['upload_id']}", headers=H(root)).status_code == 404
    assert root.delete(f"{API}/tokens/{ids['token_id']}", headers=H(root)).status_code == 404
    # root's user-role token acts as an ordinary user
    tok = root.post(f"{API}/tokens", json={"name": "root-as-user", "role": "user"}, headers=H(root)).json()["token"]
    rt = TestClient(e.app)
    hdr = {"Authorization": f"Bearer {tok}"}
    assert rt.get(f"{API}/batches?all=true", headers=hdr).json() == []
    for url in (f"/batches/{tc}", f"/batches/{tc}/result", f"/prompts/{ids['template_id']}",
                f"/sheet-templates/{ids['tid']}", f"/providers/{ids['provider_id']}/ack-status"):
        r = rt.get(API + url, headers=hdr)
        assert r.status_code == 404, (url, r.status_code)
    assert rt.post(f"{API}/batches/{tc}/subtasks/{sid}/review", json={"final_score": 1},
                   headers=hdr).status_code == 404
    assert rt.get(f"{API}/admin/users", headers=hdr).status_code == 403
    assert rt.post(f"{API}/tokens", json={"name": "x", "role": "root"}, headers=hdr).status_code == 403


def test_unauthenticated_gets_401_everywhere(world):
    e = world
    anon = TestClient(e.app)
    n = 0
    for path, ops in e.app.openapi()["paths"].items():
        for m in ops:
            method = m.upper()
            if (method, path) in NO_AUTH:
                continue
            ids = {k: v for k, v in e.alice_ids.items()}
            r = call(anon, method, path, ids, {})
            assert r.status_code == 401, (method, path, r.status_code, r.text[:200])
            n += 1
    assert n >= 60
