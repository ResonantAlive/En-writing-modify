"""Staging / images / grouping tests (real files, real HTTP via TestClient)."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core import clock
from app.main import create_app
from app.staging import grouping, images, uploads as S
from tests.fixtures_images import make_image

PW = "correct-horse-battery"
API = "/api/v1"
UP = f"{API}/uploads"


class Env:
    def __init__(self, data, app):
        self.data, self.app = data, app

    def codes(self):
        return json.loads((self.data / "app" / "invite_codes.json").read_text("utf-8"))

    def login(self, kind: str, name: str):
        c = TestClient(self.app)
        code = self.codes()["root_invite_code" if kind == "root" else "user_invite_code"]
        r = c.post(f"{API}/auth/register", json={"invite_code": code, "username": name, "password": PW})
        assert r.status_code == 201, r.text
        r = c.post(f"{API}/auth/login", json={"username": name, "password": PW})
        assert r.status_code == 200, r.text
        c.h = {"X-CSRF-Token": r.json()["csrf_token"]}
        c.uid = r.json().get("user", {}).get("id") or r.json().get("id")
        return c


@pytest.fixture
def mk(tmp_path):
    stack = ExitStack()
    n = [0]

    def make(settings: dict | None = None) -> Env:
        n[0] += 1
        data = tmp_path / f"data{n[0]}"
        if settings is not None:
            (data / "app").mkdir(parents=True, exist_ok=True)
            (data / "app" / "settings.json").write_text(json.dumps(settings), "utf-8")
        root = logging.getLogger("wm")
        for h in list(root.handlers):
            if getattr(h, "_wm", False):
                root.removeHandler(h)
                h.close()
        app = create_app(data)
        stack.enter_context(TestClient(app))
        return Env(data, app)

    yield make
    stack.close()
    clock.set_now_fn(None)
    root = logging.getLogger("wm")
    for h in list(root.handlers):
        if getattr(h, "_wm", False):
            root.removeHandler(h)
            h.close()


@pytest.fixture
def env(mk):
    return mk()


@pytest.fixture
def alice(env):
    return env.login("user", "alice")


def files_arg(*items):
    """items: (name, bytes)"""
    return [("files", (n, io.BytesIO(b), "application/octet-stream")) for n, b in items]


def upload(c, *items, lm=None, upload_id=None):
    data = {}
    if lm is not None:
        data["client_last_modified"] = [str(x) if x is not None else "" for x in lm]
    if upload_id:
        data["upload_id"] = upload_id
    return c.post(UP, files=files_arg(*items), data=data, headers=c.h)


def staging_dir(env, uid):
    return env.data / "staging" / uid


# ---------------------------------------------------------------- type detection

def test_sniff_ignores_extension(alice, env):
    png = make_image("png")
    r = upload(alice, ("photo.txt", png), ("fake.jpg", b"just some text, not an image at all"))
    assert r.status_code == 200, r.text
    j = r.json()
    assert [f["mime"] for f in j["files"]] == ["image/png"]
    assert j["files"][0]["orig_name"] == "photo.txt"
    assert len(j["rejected"]) == 1
    assert j["rejected"][0]["orig_name"] == "fake.jpg" and j["rejected"][0]["code"] == "unsupported_type"
    # rejected bytes never stored
    stored = list((staging_dir(env, j["upload_id"]) / "files").iterdir())
    assert len(stored) == 1 and stored[0].suffix == ".png"


@pytest.mark.parametrize("fmt,mime,ext", [("jpeg", "image/jpeg", "jpg"), ("png", "image/png", "png"),
                                          ("webp", "image/webp", "webp"), ("heic", "image/heic", "heic")])
def test_allowed_formats(alice, env, fmt, mime, ext):
    data = make_image(fmt, size=(64, 32))
    assert images.sniff_type(data[:32]) == (mime, ext)
    r = upload(alice, (f"x.{ext}", data))
    j = r.json()
    assert not j["rejected"], j
    f = j["files"][0]
    assert f["mime"] == mime and (f["width"], f["height"]) == (64, 32) and f["size"] == len(data)


def test_sniff_heif_and_negative():
    assert images.sniff_type(b"\x00\x00\x00\x18ftypmif1\x00\x00\x00\x00") == ("image/heif", "heif")
    assert images.sniff_type(b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00") is None
    assert images.sniff_type(b"GIF89a....") is None
    assert images.sniff_type(b"") is None
    assert images.sniff_type(b"RIFF\x00\x00\x00\x00WAVE") is None


def test_heic_thumbnail_is_jpeg(alice, env):
    r = upload(alice, ("IMG_0001.HEIC", make_image("heic", size=(600, 300))))
    f = r.json()["files"][0]
    assert (f["width"], f["height"]) == (600, 300)
    t = alice.get(f["thumb_url"])
    assert t.status_code == 200 and t.headers["content-type"] == "image/jpeg"
    assert "private" in t.headers["cache-control"]
    assert t.content[:3] == b"\xff\xd8\xff"
    im = Image.open(io.BytesIO(t.content))
    assert im.format == "JPEG" and max(im.size) == 320 and im.size == (320, 160)


def test_exif_oriented_thumbnail_and_dims(alice):
    data = make_image("jpeg", size=(80, 40), orientation=6)
    f = upload(alice, ("rot.jpg", data)).json()["files"][0]
    assert (f["width"], f["height"]) == (40, 80)          # oriented (display) dimensions
    im = Image.open(io.BytesIO(alice.get(f["thumb_url"]).content)).convert("RGB")
    assert im.size == (40, 80)
    # marker (blue, top-left in raw pixels) must end up top-right after rotation
    r, g, b = im.getpixel((35, 5))
    assert b > 150 and r < 100
    r, g, b = im.getpixel((4, 5))
    assert r > 150 and b < 100


def test_exif_time_extracted(alice):
    data = make_image("jpeg", exif_time="2026:11:13 10:45:12")
    f = upload(alice, ("a.jpg", data)).json()["files"][0]
    assert f["exif_time"] == "2026-11-13T10:45:12"
    g = upload(alice, ("b.jpg", make_image("jpeg"))).json()["files"][0]
    assert g["exif_time"] is None


def test_exif_time_png_and_webp(alice):
    for fmt in ("png", "webp"):
        f = upload(alice, (f"a.{fmt}", make_image(fmt, exif_time="2026:01:02 03:04:05"))).json()["files"][0]
        assert f["exif_time"] == "2026-01-02T03:04:05", fmt


# ---------------------------------------------------------------- caps / corrupt

def test_per_file_size_cap(mk):
    env = mk({"max_file_mb": 1})
    c = env.login("user", "alice")
    big = make_image("png")[:200] + b"\x00" * (1024 * 1024 + 10)   # valid magic, > 1 MB
    ok = make_image("jpeg")
    j = upload(c, ("big.png", big), ("ok.jpg", ok)).json()
    assert [f["orig_name"] for f in j["files"]] == ["ok.jpg"]
    assert j["rejected"][0]["code"] == "file_too_large"
    assert not any(p.name.startswith("f_") and p.stat().st_size > 1024 * 1024
                   for p in (staging_dir(env, j["upload_id"]) / "files").iterdir())
    assert len(list((staging_dir(env, j["upload_id"]) / "files").iterdir())) == 1


def test_count_cap_across_appends(mk):
    env = mk({"max_files_per_batch": 3})
    c = env.login("user", "alice")
    j = upload(c, *[(f"{i}.png", make_image("png")) for i in range(5)]).json()
    assert len(j["files"]) == 3
    assert [r["code"] for r in j["rejected"]] == ["too_many_files"] * 2
    j2 = upload(c, ("more.png", make_image("png")), upload_id=j["upload_id"]).json()
    assert len(j2["files"]) == 3 and j2["rejected"][0]["code"] == "too_many_files"
    # delete one, then there is room again
    fid = j["files"][0]["file_id"]
    assert c.delete(f"{UP}/{j['upload_id']}/files/{fid}", headers=c.h).json() == {"ok": True}
    j3 = upload(c, ("more.png", make_image("png")), upload_id=j["upload_id"]).json()
    assert len(j3["files"]) == 3 and not j3["rejected"]


def test_corrupt_image_rejected_individually(alice, env):
    good = make_image("jpeg")
    corrupt_png = make_image("png")[:30]                 # valid magic, truncated body
    corrupt_jpg = good[:20] + b"\x00" * 50
    j = upload(alice, ("c1.png", corrupt_png), ("g.jpg", good), ("c2.jpg", corrupt_jpg)).json()
    assert [f["orig_name"] for f in j["files"]] == ["g.jpg"]
    assert {r["orig_name"]: r["code"] for r in j["rejected"]} == {"c1.png": "corrupt_image",
                                                                  "c2.jpg": "corrupt_image"}
    d = staging_dir(env, j["upload_id"])
    assert len(list((d / "files").iterdir())) == 1 and len(list((d / "thumbs").iterdir())) == 1


def test_limits_endpoint(alice):
    j = alice.get(f"{UP}/limits").json()
    assert {k: j[k] for k in ("max_files_per_batch", "max_file_mb", "max_pages_per_essay",
                              "allowed_types", "default_local_engine")} == {
        "max_files_per_batch": 100, "max_file_mb": 20, "max_pages_per_essay": 6,
        "allowed_types": ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"],
        "default_local_engine": "paddle"}
    assert isinstance(j["local_env_ok"], bool) and "local_env_reason" in j


def test_requires_auth(env):
    c = TestClient(env.app)
    assert c.get(f"{UP}/limits").status_code == 401
    assert c.get(f"{UP}/up_abc").status_code == 401


# ---------------------------------------------------------------- originals / names

def test_original_byte_identical(alice, env):
    items = [("a.jpg", make_image("jpeg", exif_time="2026:11:13 10:45:12")),
             ("b.png", make_image("png")), ("c.webp", make_image("webp")), ("d.heic", make_image("heic"))]
    j = upload(alice, *items).json()
    m = json.loads((staging_dir(env, j["upload_id"]) / "manifest.json").read_text("utf-8"))
    assert m["schema_version"] == 1 and m["owner_id"] and m["upload_id"] == j["upload_id"]
    for (name, data), f in zip(items, m["files"]):
        p = S.file_abspath(env.app.state.cfg, j["upload_id"], f["file_id"])
        assert p.read_bytes() == data
        assert hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"] == hashlib.sha256(data).hexdigest()
        assert f["size"] == len(data) and f["thumb"] == f"thumbs/{f['file_id']}.jpg"
        assert (staging_dir(env, j["upload_id"]) / f["thumb"]).is_file()
    assert [f["orig_name"] for f in m["files"]] == [n for n, _ in items]       # upload order


def test_orig_name_is_data_only(alice, env):
    evil = ["<img src=x onerror=alert(1)>.jpg", "..\\..\\evil.jpg", "../../etc/passwd.png",
            "a:b*c?d'e|f;g.jpg", "作文 第1页 😀.JPG", "CON.png", "x" * 50 + ".jpg"]
    j = upload(alice, *[(n, make_image("jpeg")) for n in evil]).json()
    assert [f["orig_name"] for f in j["files"]] == evil
    again = alice.get(f"{UP}/{j['upload_id']}").json()
    assert [f["orig_name"] for f in again["files"]] == evil                      # verbatim round trip
    d = staging_dir(env, j["upload_id"])
    names = {p.name for p in (d / "files").iterdir()}
    assert all(n.startswith("f_") and n.endswith(".jpg") for n in names) and len(names) == len(evil)
    # nothing escaped the upload dir
    assert not (env.data / "evil.jpg").exists()
    assert {p.name for p in (env.data / "staging").iterdir()} == {j["upload_id"]}


def test_name_seq_in_response(alice):
    j = upload(alice, ("IMG_20261113_104512.jpg", make_image("jpeg")), ("scan.jpg", make_image("jpeg"))).json()
    assert j["files"][0]["name_seq"] == [20261113, 104512] and j["files"][1]["name_seq"] is None


def test_client_last_modified_matching(alice):
    j = upload(alice, ("a.jpg", make_image("jpeg")), ("b.jpg", make_image("jpeg")),
               ("c.jpg", make_image("jpeg")), lm=[1731494712000, None, "garbage"]).json()
    assert [f["client_last_modified"] for f in j["files"]] == [1731494712000, None, None]


def test_append_to_existing_upload(alice, env):
    j = upload(alice, ("a.jpg", make_image("jpeg"))).json()
    j2 = upload(alice, ("b.png", make_image("png")), upload_id=j["upload_id"]).json()
    assert j2["upload_id"] == j["upload_id"]
    assert [f["orig_name"] for f in j2["files"]] == ["a.jpg", "b.png"]
    assert [f["orig_name"] for f in alice.get(f"{UP}/{j['upload_id']}").json()["files"]] == ["a.jpg", "b.png"]


# ---------------------------------------------------------------- ownership / traversal

def test_owner_only_even_for_root(env, alice):
    root = env.login("root", "rooty")
    bob = env.login("user", "bob")
    j = upload(alice, ("a.jpg", make_image("jpeg")), ("b.jpg", make_image("jpeg"))).json()
    uid, fid = j["upload_id"], j["files"][0]["file_id"]
    for other in (bob, root):
        assert other.get(f"{UP}/{uid}").status_code == 404
        assert other.get(f"{UP}/{uid}/files/{fid}/thumb").status_code == 404
        assert other.delete(f"{UP}/{uid}/files/{fid}", headers=other.h).status_code == 404
        assert other.delete(f"{UP}/{uid}", headers=other.h).status_code == 404
        assert other.post(f"{UP}/{uid}/grouping/preview", json={"multi_photo": False},
                          headers=other.h).status_code == 404
        r = upload(other, ("z.jpg", make_image("jpeg")), upload_id=uid)
        assert r.status_code == 404
        body = r.json()
        assert body["code"] == "not_found"
    # nothing changed
    assert len(alice.get(f"{UP}/{uid}").json()["files"]) == 2
    # 404 body is identical for missing and foreign uploads
    assert bob.get(f"{UP}/{uid}").json() == bob.get(f"{UP}/up_doesnotexist").json()
    # get_manifest enforces it at the Python level too
    cfg = env.app.state.cfg
    S.get_manifest(cfg, json.loads((staging_dir(env, uid) / "manifest.json").read_text())["owner_id"], uid)
    from app.errors import NotFound
    with pytest.raises(NotFound):
        S.get_manifest(cfg, "someone-else", uid)


def test_path_traversal_ids_rejected(alice, env):
    j = upload(alice, ("a.jpg", make_image("jpeg"))).json()
    uid, fid = j["upload_id"], j["files"][0]["file_id"]
    cfg = env.app.state.cfg
    from app.errors import NotFound
    for bad in ["..", "../data", "a/b", "a\\b", "", ".", "x.y", "up_1%2f..", "..\\..", "a b"]:
        with pytest.raises(NotFound):
            S.upload_dir(cfg, bad)
        with pytest.raises(NotFound):
            S.file_abspath(cfg, uid, bad)
    for bad in ["%2e%2e", "..%2f..%2fapp", "up_x%00", "%2e%2e%5c%2e%2e"]:
        assert alice.get(f"{UP}/{bad}").status_code == 404
        assert alice.get(f"{UP}/{uid}/files/{bad}/thumb").status_code == 404
        # some encodings collapse into a path that only has GET routes (405): still nothing is deleted
        assert alice.delete(f"{UP}/{uid}/files/{bad}", headers=alice.h).status_code in (404, 405)
        assert alice.delete(f"{UP}/{bad}", headers=alice.h).status_code in (404, 405)
        assert alice.post(f"{UP}/{bad}/grouping/preview", json={}, headers=alice.h).status_code in (404, 405)
        assert upload(alice, ("a.jpg", make_image("jpeg")), upload_id=bad).status_code == 404
    assert (staging_dir(env, uid) / "manifest.json").is_file()
    assert (env.data / "app").is_dir()


def test_delete_file_and_upload(alice, env):
    j = upload(alice, ("a.jpg", make_image("jpeg")), ("b.jpg", make_image("jpeg"))).json()
    uid = j["upload_id"]
    f0 = j["files"][0]
    d = staging_dir(env, uid)
    assert alice.delete(f"{UP}/{uid}/files/{f0['file_id']}", headers=alice.h).json() == {"ok": True}
    assert [f["orig_name"] for f in alice.get(f"{UP}/{uid}").json()["files"]] == ["b.jpg"]
    assert len(list((d / "files").iterdir())) == 1 and len(list((d / "thumbs").iterdir())) == 1
    assert alice.get(f0["thumb_url"]).status_code == 404
    assert alice.delete(f"{UP}/{uid}/files/{f0['file_id']}", headers=alice.h).status_code == 404
    assert alice.delete(f"{UP}/{uid}", headers=alice.h).json() == {"ok": True}
    assert not d.exists() and alice.get(f"{UP}/{uid}").status_code == 404


def test_release(alice, env):
    uid = upload(alice, ("a.jpg", make_image("jpeg"))).json()["upload_id"]
    S.release(env.app.state.cfg, uid)
    assert not staging_dir(env, uid).exists()
    S.release(env.app.state.cfg, uid)        # idempotent


def test_writes_need_csrf(alice):
    r = alice.post(UP, files=files_arg(("a.jpg", make_image("jpeg"))))
    assert r.status_code == 403


# ---------------------------------------------------------------- cleanup

def test_cleanup_expired_only_expired(alice, env):
    cfg = env.app.state.cfg
    a = upload(alice, ("a.jpg", make_image("jpeg"))).json()["upload_id"]   # fresh
    b = upload(alice, ("b.jpg", make_image("jpeg"))).json()["upload_id"]   # old by mtime
    c = upload(alice, ("c.jpg", make_image("jpeg"))).json()["upload_id"]   # old by manifest created_at
    old = time.time() - 25 * 3600
    os.utime(staging_dir(env, b), (old, old))
    mp = staging_dir(env, c) / "manifest.json"
    m = json.loads(mp.read_text("utf-8"))
    m["created_at"] = clock.iso(clock.now().replace(year=2020))
    mp.write_text(json.dumps(m), "utf-8")
    assert S.cleanup_expired(cfg) == 2
    assert staging_dir(env, a).is_dir() and not staging_dir(env, b).exists() and not staging_dir(env, c).exists()
    assert S.cleanup_expired(cfg) == 0
    assert alice.get(f"{UP}/{a}").status_code == 200


def test_cleanup_respects_ttl_setting(mk):
    env = mk({"staging_ttl_hours": 1})
    c = env.login("user", "alice")
    keep = upload(c, ("a.jpg", make_image("jpeg"))).json()["upload_id"]
    drop = upload(c, ("a.jpg", make_image("jpeg"))).json()["upload_id"]
    t = time.time() - 2 * 3600
    os.utime(staging_dir(env, drop), (t, t))
    assert S.cleanup_expired(env.app.state.cfg) == 1
    assert staging_dir(env, keep).exists() and not staging_dir(env, drop).exists()


# ---------------------------------------------------------------- grouping (pure)

def mf(*specs):
    """specs: dict(name=..., exif=..., lm=...) -> manifest"""
    files = []
    for i, s in enumerate(specs):
        files.append({"file_id": s.get("id", f"f{i}"), "orig_name": s.get("name", f"x{i}.jpg"),
                      "name_seq": grouping.name_seq(s.get("name", "")), "exif_time": s.get("exif"),
                      "client_last_modified": s.get("lm")})
    return {"upload_id": "up_1", "files": files}


def ptypes(res):
    return sorted(p["type"] for p in res["problems"])


def test_name_seq():
    assert grouping.name_seq("IMG_20261113_104512.jpg") == [20261113, 104512]
    assert grouping.name_seq("IMG_1234.HEIC") == [1234]
    assert grouping.name_seq("scan.jpg") is None
    assert grouping.name_seq("a1b22.c3.png") == [1, 22, 3]       # only the last extension is dropped
    assert grouping.name_seq("C:\\pics\\IMG_7.jpg") == [7]
    assert grouping.name_seq("") is None


def test_single_mode(cfg):
    m = mf(dict(name="b.jpg"), dict(name="a.jpg"), dict(name="c.jpg"))
    r = grouping.resolve_groups(cfg, m, {"multi_photo": False})
    assert r["mode"] == "single" and r["order_basis"] == "single"
    assert r["essays"] == [["f0"], ["f1"], ["f2"]] and r["ordered_file_ids"] == ["f0", "f1", "f2"]
    assert r["can_submit"] and not r["needs_confirmation"] and r["problems"] == []
    assert r["summary"] == {"essays": 3, "photos": 3}
    # per_n etc. ignored when multi_photo is false
    r2 = grouping.resolve_groups(cfg, m, {"multi_photo": False, "mode": "per_n", "n": 2})
    assert r2["essays"] == r["essays"]


def test_single_mode_empty_upload(cfg):
    r = grouping.resolve_groups(cfg, mf(), {"multi_photo": False})
    assert not r["can_submit"] and "empty_batch" in ptypes(r)


def test_filename_natural_sort(cfg):
    m = mf(dict(name="IMG_10.jpg"), dict(name="IMG_9.jpg"), dict(name="IMG_100.jpg"), dict(name="IMG_2.jpg"))
    r = grouping.resolve_groups(cfg, m, {"multi_photo": True, "mode": "per_n", "n": 2})
    assert r["order_bases"]["filename"] == {"usable": True, "reason": None}
    assert r["order_basis"] == "filename"
    assert r["ordered_file_ids"] == ["f3", "f1", "f0", "f2"]          # 2, 9, 10, 100
    assert r["essays"] == [["f3", "f1"], ["f0", "f2"]] and r["mode"] == "per_n" and r["can_submit"]
    assert r["order_values"]["f1"] == [9]


def test_filename_timestamp_style(cfg):
    m = mf(dict(name="IMG_20261113_104512.jpg"), dict(name="IMG_20261113_104500.jpg"),
           dict(name="IMG_20261112_235959.jpg"))
    r = grouping.resolve_groups(cfg, m, {"multi_photo": True, "mode": "per_n", "n": 1})
    assert r["ordered_file_ids"] == ["f2", "f1", "f0"]
    assert r["order_values"]["f2"] == [20261112, 235959]


def test_filename_duplicates_or_missing_unusable_falls_to_exif(cfg):
    m = mf(dict(name="IMG_1.jpg", exif="2026-01-01T10:00:02"), dict(name="IMG_1.png", exif="2026-01-01T10:00:01"),
           dict(name="IMG_2.jpg", exif="2026-01-01T10:00:03"))
    r = grouping.resolve_groups(cfg, m, {"multi_photo": True, "mode": "per_n", "n": 3})
    assert r["order_bases"]["filename"] == {"usable": False, "reason": "duplicate_values"}
    assert r["order_bases"]["exif"]["usable"] and r["order_basis"] == "exif"
    assert r["ordered_file_ids"] == ["f1", "f0", "f2"]
    m2 = mf(dict(name="IMG_1.jpg", lm=3), dict(name="scan.jpg", lm=1), dict(name="IMG_3.jpg", lm=2))
    r2 = grouping.resolve_groups(cfg, m2, {"multi_photo": True, "mode": "per_n", "n": 3})
    assert r2["order_bases"]["filename"] == {"usable": False, "reason": "missing_values"}
    assert r2["order_basis"] == "mtime" and r2["ordered_file_ids"] == ["f1", "f2", "f0"]


def test_exif_missing_or_duplicate_unusable(cfg):
    m = mf(dict(name="a.jpg", exif="2026-01-01T10:00:00"), dict(name="b.jpg", exif=None),
           dict(name="c.jpg", exif="2026-01-01T10:00:02"))
    assert grouping.resolve_groups(cfg, m, {"multi_photo": True})["order_bases"]["exif"]["usable"] is False
    m = mf(dict(name="a.jpg", exif="2026-01-01T10:00:00"), dict(name="b.jpg", exif="2026-01-01T10:00:00"))
    assert grouping.resolve_groups(cfg, m, {"multi_photo": True})["order_bases"]["exif"]["reason"] == "duplicate_values"


def test_mtime_basis(cfg):
    m = mf(dict(name="a.jpg", lm=300), dict(name="b.jpg", lm=100), dict(name="c.jpg", lm=200))
    r = grouping.resolve_groups(cfg, m, {"multi_photo": True, "mode": "per_n", "n": 3})
    assert r["order_basis"] == "mtime" and r["ordered_file_ids"] == ["f1", "f2", "f0"]
    assert r["essays"] == [["f1", "f2", "f0"]]


def test_priority_filename_over_exif_over_mtime(cfg):
    m = mf(dict(name="IMG_2.jpg", exif="2026-01-01T10:00:01", lm=1), dict(name="IMG_1.jpg", exif="2026-01-01T10:00:02", lm=2))
    r = grouping.resolve_groups(cfg, m, {"multi_photo": True, "mode": "per_n", "n": 2})
    assert r["order_basis"] == "filename" and r["ordered_file_ids"] == ["f1", "f0"]
    assert all(r["order_bases"][b]["usable"] for b in ("filename", "exif", "mtime"))


def test_forced_basis_override(cfg):
    m = mf(dict(name="IMG_2.jpg", exif="2026-01-01T10:00:01", lm=20), dict(name="IMG_1.jpg", exif="2026-01-01T10:00:02", lm=10))
    base = {"multi_photo": True, "mode": "per_n", "n": 2}
    assert grouping.resolve_groups(cfg, m, {**base, "order_basis": "exif"})["ordered_file_ids"] == ["f0", "f1"]
    assert grouping.resolve_groups(cfg, m, {**base, "order_basis": "mtime"})["ordered_file_ids"] == ["f1", "f0"]
    r = grouping.resolve_groups(cfg, m, {**base, "order_basis": "filename"})
    assert r["order_basis"] == "filename" and r["problems"] == []


def test_forced_unusable_basis(cfg):
    m = mf(dict(name="IMG_2.jpg", lm=20), dict(name="IMG_1.jpg", lm=10))      # no exif
    r = grouping.resolve_groups(cfg, m, {"multi_photo": True, "mode": "per_n", "n": 2, "order_basis": "exif"})
    assert "basis_unusable" in ptypes(r)
    assert r["order_basis"] == "filename"                                      # fell back to first usable
    assert r["can_submit"]                                                     # informational only
    m = mf(dict(name="a.jpg"), dict(name="b.jpg"))                             # nothing usable at all
    r = grouping.resolve_groups(cfg, m, {"multi_photo": True, "mode": "per_n", "n": 2, "order_basis": "mtime"})
    assert {"basis_unusable", "no_reliable_order"} <= set(ptypes(r)) and r["order_basis"] == "none"


@pytest.mark.parametrize("n,expect", [(2, [["f0", "f1"], ["f2", "f3"], ["f4", "f5"]]),
                                      (3, [["f0", "f1", "f2"], ["f3", "f4", "f5"]]),
                                      (6, [["f0", "f1", "f2", "f3", "f4", "f5"]]),
                                      (1, [[f"f{i}"] for i in range(6)])])
def test_per_n(cfg, n, expect):
    m = mf(*[dict(name=f"IMG_{i}.jpg") for i in range(6)])
    r = grouping.resolve_groups(cfg, m, {"multi_photo": True, "mode": "per_n", "n": n})
    assert r["mode"] == "per_n" and r["essays"] == expect and r["can_submit"] and not r["needs_confirmation"]
    assert r["summary"] == {"essays": len(expect), "photos": 6}


def test_per_n_custom_4(cfg):
    m = mf(*[dict(name=f"IMG_{i}.jpg") for i in range(8)])
    r = grouping.resolve_groups(cfg, m, {"multi_photo": True, "mode": "per_n", "n": 4})
    assert [len(e) for e in r["essays"]] == [4, 4] and r["can_submit"]


def test_per_n_not_divisible_switches_to_manual(cfg):
    m = mf(*[dict(name=f"IMG_{i}.jpg") for i in range(7)])
    r = grouping.resolve_groups(cfg, m, {"multi_photo": True, "mode": "per_n", "n": 3})
    assert "not_divisible" in ptypes(r)
    assert r["mode"] == "manual" and r["needs_confirmation"]
    assert [len(e) for e in r["essays"]] == [3, 3, 1]                          # closest plan, prefilled
    assert r["order_basis"] == "filename" and r["can_submit"]
    assert r["summary"] == {"essays": 3, "photos": 7}


def test_per_n_over_max_pages(cfg):
    m = mf(*[dict(name=f"IMG_{i}.jpg") for i in range(8)])
    r = grouping.resolve_groups(cfg, m, {"multi_photo": True, "mode": "per_n", "n": 8})
    assert "too_many_pages" in ptypes(r) and not r["can_submit"]


def test_per_n_invalid_n(cfg):
    from app.errors import AppError
    m = mf(dict(name="IMG_1.jpg"))
    for n in (None, 0, -1, "2", True):
        with pytest.raises(AppError) as e:
            grouping.resolve_groups(cfg, m, {"multi_photo": True, "mode": "per_n", "n": n})
        assert e.value.status == 422


def test_no_reliable_order_forces_manual(cfg):
    m = mf(dict(name="a.jpg"), dict(name="b.jpg"), dict(name="c.jpg"))
    r = grouping.resolve_groups(cfg, m, {"multi_photo": True, "mode": "per_n", "n": 3})
    assert r["order_basis"] == "none" and r["mode"] == "manual" and r["needs_confirmation"]
    assert "no_reliable_order" in ptypes(r)
    assert r["ordered_file_ids"] == ["f0", "f1", "f2"] and r["essays"] == [["f0"], ["f1"], ["f2"]]
    assert all(not b["usable"] for b in r["order_bases"].values())


def test_manual_mixed_group_sizes(cfg):
    m = mf(*[dict(name=f"IMG_{i}.jpg") for i in range(6)])
    essays = [["f3"], ["f1", "f0"], ["f5", "f4", "f2"]]
    r = grouping.resolve_groups(cfg, m, {"multi_photo": True, "mode": "manual", "essays": essays})
    assert r["mode"] == "manual" and r["order_basis"] == "manual_user" and r["essays"] == essays
    assert r["can_submit"] and r["needs_confirmation"] and r["problems"] == []
    assert r["summary"] == {"essays": 3, "photos": 6}
    assert r["ordered_file_ids"] == ["f3", "f1", "f0", "f5", "f4", "f2"]
    assert r["order_values"]["f3"] == 1 and r["order_values"]["f2"] == 6


def test_manual_without_essays_prefills_in_order(cfg):
    m = mf(dict(name="IMG_3.jpg"), dict(name="IMG_1.jpg"), dict(name="IMG_2.jpg"))
    r = grouping.resolve_groups(cfg, m, {"multi_photo": True, "mode": "manual"})
    assert r["essays"] == [["f1"], ["f2"], ["f0"]] and r["order_basis"] == "filename" and r["can_submit"]


def test_manual_problems(cfg):
    m = mf(*[dict(name=f"IMG_{i}.jpg") for i in range(8)])
    ids = [f"f{i}" for i in range(8)]
    req = lambda e: {"multi_photo": True, "mode": "manual", "essays": e}
    r = grouping.resolve_groups(cfg, m, req([ids[:4], [], ids[4:]]))
    assert ptypes(r) == ["empty_group"] and not r["can_submit"]
    assert r["problems"][0]["essay_index"] == 1
    r = grouping.resolve_groups(cfg, m, req([ids[:7], ids[7:]]))                 # 7 > max 6
    assert ptypes(r) == ["too_many_pages"] and not r["can_submit"] and r["problems"][0]["essay_index"] == 0
    r = grouping.resolve_groups(cfg, m, req([ids[:3], ids[3:7]]))                # f7 missing
    assert ptypes(r) == ["unassigned_file"] and r["problems"][0]["detail"] == "f7" and not r["can_submit"]
    r = grouping.resolve_groups(cfg, m, req([ids[:4], ids[3:]]))                 # f3 twice
    assert ptypes(r) == ["duplicate_file"] and not r["can_submit"]
    r = grouping.resolve_groups(cfg, m, req([ids[:4], ids[4:] + ["f999"]]))
    assert ptypes(r) == ["unknown_file"] and not r["can_submit"]
    r = grouping.resolve_groups(cfg, m, req([]))
    assert "empty_batch" in ptypes(r) and not r["can_submit"]
    r = grouping.resolve_groups(cfg, m, req([ids[:4], ids[4:]]))
    assert r["can_submit"]


def test_manual_with_unreliable_order_still_ok(cfg):
    m = mf(dict(name="a.jpg"), dict(name="b.jpg"))
    r = grouping.resolve_groups(cfg, m, {"multi_photo": True, "mode": "manual", "essays": [["f1", "f0"]]})
    assert r["can_submit"] and r["order_basis"] == "manual_user" and "no_reliable_order" in ptypes(r)


def test_max_pages_setting_respected(mk):
    env = mk({"max_pages_per_essay": 2})
    m = mf(*[dict(name=f"IMG_{i}.jpg") for i in range(3)])
    r = grouping.resolve_groups(env.app.state.cfg, m,
                                {"multi_photo": True, "mode": "manual", "essays": [["f0", "f1", "f2"]]})
    assert ptypes(r) == ["too_many_pages"]


# ---------------------------------------------------------------- grouping via HTTP

def test_preview_endpoint_end_to_end(alice):
    items = [(f"IMG_{i}.jpg", make_image("jpeg", exif_time=f"2026:11:13 10:0{i}:00")) for i in (3, 1, 2, 10)]
    j = upload(alice, *items, lm=[4, 3, 2, 1]).json()
    uid = j["upload_id"]
    ids = [f["file_id"] for f in j["files"]]
    r = alice.post(f"{UP}/{uid}/grouping/preview", json={"multi_photo": True, "mode": "per_n", "n": 2},
                   headers=alice.h)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["order_basis"] == "filename"
    assert res["ordered_file_ids"] == [ids[1], ids[2], ids[0], ids[3]]
    assert res["can_submit"] and res["summary"] == {"essays": 2, "photos": 4}
    r = alice.post(f"{UP}/{uid}/grouping/preview",
                   json={"multi_photo": True, "mode": "per_n", "n": 2, "order_basis": "mtime"}, headers=alice.h)
    assert r.json()["ordered_file_ids"] == [ids[3], ids[2], ids[1], ids[0]]
    r = alice.post(f"{UP}/{uid}/grouping/preview", json={"multi_photo": True, "mode": "per_n", "n": 0},
                   headers=alice.h)
    assert r.status_code == 422 and r.json()["code"] == "invalid_grouping"
    r = alice.post(f"{UP}/{uid}/grouping/preview", json={"multi_photo": False}, headers=alice.h)
    assert r.json()["mode"] == "single" and len(r.json()["essays"]) == 4
