"""Answer-sheet templates: printed-matter filter, template store, API, and the OCR service path."""
from __future__ import annotations

import io
import math

import pytest

from app.ocr import service, sheet_filter
from app.ocr.common import Line, OcrError, PageResult
from app.sheets import store as sheets
from tests.fixtures_images import make_image
from tests.sheet_fixtures import make_sheet
from tests.test_batches import API, H, _isolate, env, err  # noqa: F401
from tests.test_grading_pipeline import SID, meta_of, rd, setup


@pytest.fixture(autouse=True)
def _reset_engine():
    yield
    sheets.wait_idle(10)
    service._engine.update(name=None, obj=None)
    service._env_cache.update(t=0.0, v=None)


# ------------------------------------------------------------------ synthetic sheet
TEMPLATE = [
    {"text": "Cambridge International Education", "box": [100, 40, 900, 80]},
    {"text": "Write your answer in the space below", "box": [100, 120, 800, 150]},
    {"text": "Name", "box": [100, 180, 180, 210]},
    {"text": "0500/01/O/N/26", "box": [100, 1900, 400, 1930]},
    {"text": "* 0123456789 *", "box": [650, 1900, 1000, 1930]},
    {"text": "2", "box": [540, 1960, 560, 1990]},
    {"text": "[Turn over", "box": [850, 1960, 1000, 1990]},
]
ESSAY = ["My summer holiday was grate fun.", "We went to the beach every day.",
         "In conclusion I recieve a lot of joy."]


def tf(box, s=1.25, deg=2.0, tx=-30.0, ty=55.0):
    """Template pixels -> photo pixels (axis-aligned box of the transformed rectangle, like the detector's)."""
    a, b = s * math.cos(math.radians(deg)), s * math.sin(math.radians(deg))
    x0, y0, x1, y1 = box
    pts = [(a * x - b * y + tx, b * x + a * y + ty) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    return [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]


def photo(**kw):
    """A filled sheet: printed matter (two lines read slightly differently, new barcode digits and page number),
    a handwritten name next to the printed label, and three essay lines."""
    t = {x["text"]: x["box"] for x in TEMPLATE}
    boxes = [
        Line("Cambridge Intemational Education", tf(t["Cambridge International Education"], **kw), 0.98),
        Line("Write your answer in the space below", tf(t["Write your answer in the space below"], **kw), 0.99),
        Line("Name", tf(t["Name"], **kw), 0.99),
        Line("Lily Chen", tf([200, 178, 420, 212], **kw), 0.9),
        Line("0500/01/O/N/26", tf(t["0500/01/O/N/26"], **kw), 0.99),
        Line("* 9876543210 *", tf(t["* 0123456789 *"], **kw), 0.97),
        Line("3", tf(t["2"], **kw), 0.9),
        Line("Turn over", tf(t["[Turn over"], **kw), 0.99),
    ]
    for k, s in enumerate(ESSAY):
        boxes.append(Line(s, tf([100, 300 + 70 * k, 950, 345 + 70 * k], **kw), 0.9))
    return boxes


def reasons(excl):
    return {e["text"]: e["reason"] for e in excl}


# ------------------------------------------------------------------ filter
@pytest.mark.parametrize("deg", [0.0, 2.0, -4.0])
def test_filter_removes_printed_matter_keeps_handwriting(deg):
    kept, excl, info = sheet_filter.filter_page(photo(deg=deg), [{"page_id": "pg1", "lines": TEMPLATE}])
    assert info["aligned"] and info["template_page"] == "pg1" and info["anchors"] >= 5
    assert abs(info["scale"] - 1.25) < 0.05 and abs(info["angle_deg"] - deg) < 1.5
    r = reasons(excl)
    assert r["Cambridge Intemational Education"] == "template_text"
    assert r["Name"] == "template_text"                         # short label: removed because it is in place
    assert r["0500/01/O/N/26"] == "template_text" and r["Turn over"] == "template_text"
    assert r["* 9876543210 *"] == "template_region"             # different barcode digits, same place
    assert r["3"] == "template_region"                          # page number differs from the blank sheet
    assert [b.text for b in kept] == ["Lily Chen"] + ESSAY
    for e in excl:
        assert set(e) == {"text", "box", "score", "reason"}


def test_filter_picks_best_template_page():
    other = {"page_id": "cover", "lines": [{"text": "Candidate declaration and signature", "box": [50, 50, 900, 90]}]}
    _, excl, info = sheet_filter.filter_page(photo(), [other, {"page_id": "inner", "lines": TEMPLATE}])
    assert info["template_page"] == "inner" and len(excl) == 7


def test_filter_short_template_word_in_essay_is_kept():
    boxes = photo() + [Line("Name", tf([300, 600, 380, 640]), 0.9)]      # the student writes "Name" in the body
    kept, excl, _ = sheet_filter.filter_page(boxes, [{"page_id": "p", "lines": TEMPLATE}])
    assert sum(1 for b in kept if b.text == "Name") == 1
    assert sum(1 for e in excl if e["text"] == "Name") == 1


def test_filter_without_alignment_only_long_text_and_letterless_edges():
    # no consistent second anchor -> no alignment; the long header goes, the short label stays (could be the
    # student's own word); the bogus tiny template box cannot produce an alignment
    tpl = [{"text": "Cambridge International Education", "box": [100, 40, 900, 80]}, {"text": "Name", "box": [0, 0, 1, 1]}]
    boxes = [Line("Cambridge International Education", [90, 30, 910, 75]),
             Line("Name", [100, 100, 180, 130]),
             Line("Dear Tom, I am writing to you.", [100, 200, 900, 240]),
             Line("2026", [100, 260, 200, 300]),                  # digits inside the body: kept
             Line("It was great.", [100, 320, 600, 360]),
             Line("IlI|lIIl|I", [100, 1800, 600, 1840]),          # barcode stripes read as letters
             Line("9 780000 123456", [100, 1850, 600, 1890])]
    kept, excl, info = sheet_filter.filter_page(boxes, [{"page_id": "p", "lines": tpl}])
    assert not info["aligned"] and info["anchors"] == 2
    assert reasons(excl) == {"Cambridge International Education": "template_text",
                             "IlI|lIIl|I": "outside_body", "9 780000 123456": "outside_body"}
    assert [b.text for b in kept] == ["Name", "Dear Tom, I am writing to you.", "2026", "It was great."]


def test_filter_no_template_lines_and_no_boxes():
    boxes = [Line("Hello there friend", [0, 0, 500, 40])]
    kept, excl, info = sheet_filter.filter_page(boxes, [{"page_id": "p", "lines": []}])
    assert kept == boxes and excl == [] and not info["aligned"]
    assert sheet_filter.filter_page([], [{"page_id": "p", "lines": TEMPLATE}]) == ([], [], {
        "template_page": None, "anchors": 0, "aligned": False})


def test_estimate_transform_recovers_rejects_outlier_and_refuses_weak_input():
    tb = [x["box"] for x in TEMPLATE if len(sheet_filter.norm_text(x["text"])) >= 4]
    pairs = [(b, tf(b, s=0.8, deg=-3, tx=40, ty=-20)) for b in tb]
    t = sheet_filter.estimate_transform(pairs, 30)
    assert abs(t.scale - 0.8) < 0.03 and abs(t.angle_deg + 3) < 1 and abs(t.tx - 40) < 15
    bad = pairs + [([100, 1000, 300, 1030], [800, 100, 1000, 130])]       # one wrong match
    t2 = sheet_filter.estimate_transform(bad, 30)
    assert t2 is not None and abs(t2.scale - 0.8) < 0.05
    close = [([100, 40, 300, 80], [100, 40, 300, 80]), ([310, 40, 500, 80], [310, 40, 500, 80])]
    assert sheet_filter.estimate_transform(close, 60) is None            # anchors too close together
    assert sheet_filter.estimate_transform(pairs[:1], 30) is None
    flipped = [(b, [2000 - p[2], 3000 - p[3], 2000 - p[0], 3000 - p[1]]) for b, p in pairs]   # 180° turn
    assert sheet_filter.estimate_transform(flipped, 30) is None


def test_has_words_ignores_barcode_stripes():
    assert not sheet_filter.has_words("IlI|lI1!") and not sheet_filter.has_words("0500 1234")
    assert sheet_filter.has_words("Dear Tom") and sheet_filter.has_words("go")


# ------------------------------------------------------------------ store / API
def fake_recognizer(monkeypatch, lines_by_size=None, fail_sizes=()):
    """service.recognize_one replacement keyed by image width (tests make images of distinct widths)."""
    calls = []

    def rec(cfg, im):
        calls.append(im.size)
        if im.size[0] in fail_sizes:
            raise service.OcrFailed("engine broke")
        texts = (lines_by_size or {}).get(im.size[0], ["Cambridge International Education", "0500/01/O/N/26"])
        boxes = [Line(t, [10, 10 + 30 * i, 200, 30 + 30 * i], 0.99) for i, t in enumerate(texts)]
        return PageResult(lines=boxes, width=im.size[0], height=im.size[1], boxes=boxes), \
            {"local_engine": "paddle", "name": "fake"}
    monkeypatch.setattr(service, "recognize_one", rec)
    return calls


def jpg(w):
    return make_image("jpeg", size=(w, 60))


def upload(c, tid, files):
    return c.post(f"{API}/sheet-templates/{tid}/pages", headers=H(c),
                  files=[("files", (n, io.BytesIO(b), "application/octet-stream")) for n, b in files])


def test_template_create_upload_recognise_and_reuse(env, monkeypatch):
    calls = fake_recognizer(monkeypatch, {101: []})
    c = env.clients["alice"]
    err(c.post(f"{API}/sheet-templates", json={"name": "  "}, headers=H(c)), 422, "validation_error")
    assert c.post(f"{API}/sheet-templates", json={"name": "IG"}).status_code == 403      # CSRF
    r = c.post(f"{API}/sheet-templates", json={"name": " IG 0500 作文纸 "}, headers=H(c))
    assert r.status_code == 201, r.text
    t = r.json()
    tid = t["id"]
    assert t["name"] == "IG 0500 作文纸" and t["usable"] is False and t["page_list"] == []
    r = upload(c, tid, [("blank1.jpg", jpg(100)), ("notes.txt", b"hello world, not an image"),
                        ("empty.jpg", jpg(101))])
    assert r.status_code == 200, r.text
    j = r.json()
    assert [x["code"] for x in j["rejected"]] == ["unsupported_type"]
    assert len(j["template"]["page_list"]) == 2
    assert sheets.wait_idle(10)
    d = c.get(f"{API}/sheet-templates/{tid}", headers=H(c)).json()
    assert d["usable"] is True and d["pages"]["ready"] == 2 and len(calls) == 2
    p1, p2 = d["page_list"]
    assert p1["status"] == "ready" and p1["lines"][0]["text"] == "Cambridge International Education"
    assert p1["warnings"] == [] and p2["warnings"] == ["no_printed_text"]
    assert p1["orig_name"] == "blank1.jpg" and p1["width"] == 100
    # files on disk under data/app/sheet_templates/<tid>/ ; image endpoint for the owner only
    assert len(list(sheets.files_dir(env.cfg, tid).iterdir())) == 2
    img = c.get(f"{API}/sheet-templates/{tid}/pages/{p1['page_id']}/image", headers=H(c))
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg" and img.content[:2] == b"\xff\xd8"
    # later: more blank pages to the same template
    r = upload(c, tid, [("blank2.jpg", jpg(102))])
    assert sheets.wait_idle(10)
    d = c.get(f"{API}/sheet-templates/{tid}", headers=H(c)).json()
    assert d["pages"]["total"] == 3 and d["pages"]["ready"] == 3
    # rename, list
    assert c.patch(f"{API}/sheet-templates/{tid}", json={"name": "IG new"}, headers=H(c)).json()["name"] == "IG new"
    lst = c.get(f"{API}/sheet-templates", headers=H(c)).json()
    assert [x["id"] for x in lst] == [tid] and "page_list" not in lst[0]


def test_template_access_control_and_sharing(env, monkeypatch):
    fake_recognizer(monkeypatch)
    a, b, root = env.clients["alice"], env.clients["bob"], env.clients["rooty"]
    tid = a.post(f"{API}/sheet-templates", json={"name": "A"}, headers=H(a)).json()["id"]
    upload(a, tid, [("x.jpg", jpg(100))])
    assert sheets.wait_idle(10)
    pid = a.get(f"{API}/sheet-templates/{tid}", headers=H(a)).json()["page_list"][0]["page_id"]
    # bob: not visible -> 404 everywhere
    assert b.get(f"{API}/sheet-templates", headers=H(b)).json() == []
    err(b.get(f"{API}/sheet-templates/{tid}", headers=H(b)), 404, "not_found")
    err(b.patch(f"{API}/sheet-templates/{tid}", json={"name": "x"}, headers=H(b)), 404, "not_found")
    err(upload(b, tid, [("y.jpg", jpg(103))]), 404, "not_found")
    err(b.patch(f"{API}/sheet-templates/{tid}/pages/{pid}", json={"disabled": True}, headers=H(b)), 404, "not_found")
    err(b.get(f"{API}/sheet-templates/{tid}/pages/{pid}/image", headers=H(b)), 404, "not_found")
    # sharing is a root action
    assert a.post(f"{API}/sheet-templates/{tid}/share", json={"shared": True}, headers=H(a)).status_code == 403
    assert root.get(f"{API}/sheet-templates/{tid}", headers=H(root)).status_code == 200
    r = root.post(f"{API}/sheet-templates/{tid}/share", json={"shared": True}, headers=H(root))
    assert r.status_code == 200 and r.json()["shared"] is True
    # shared: bob can see and use it, not change it
    assert [x["id"] for x in b.get(f"{API}/sheet-templates", headers=H(b)).json()] == [tid]
    assert b.get(f"{API}/sheet-templates/{tid}/pages/{pid}/image", headers=H(b)).status_code == 200
    err(b.patch(f"{API}/sheet-templates/{tid}", json={"name": "x"}, headers=H(b)), 403, "forbidden")
    err(upload(b, tid, [("y.jpg", jpg(103))]), 403, "forbidden")
    err(b.patch(f"{API}/sheet-templates/{tid}/pages/{pid}", json={"disabled": True}, headers=H(b)), 403, "forbidden")
    err(root.post(f"{API}/sheet-templates/sheet_nope/share", json={"shared": True}, headers=H(root)), 404, "not_found")


def test_template_failed_page_retry_disable_and_limits(env, monkeypatch):
    calls = fake_recognizer(monkeypatch, fail_sizes=(100,))
    c = env.clients["alice"]
    tid = c.post(f"{API}/sheet-templates", json={"name": "A"}, headers=H(c)).json()["id"]
    upload(c, tid, [("x.jpg", jpg(100))])
    assert sheets.wait_idle(10)
    pg = c.get(f"{API}/sheet-templates/{tid}", headers=H(c)).json()["page_list"][0]
    assert pg["status"] == "failed" and "engine broke" in pg["error"]
    pid = pg["page_id"]
    fake_recognizer(monkeypatch)                                         # engine fixed
    r = c.post(f"{API}/sheet-templates/{tid}/pages/{pid}/retry", headers=H(c))
    assert r.status_code == 200
    assert sheets.wait_idle(10)
    d = c.get(f"{API}/sheet-templates/{tid}", headers=H(c)).json()
    assert d["page_list"][0]["status"] == "ready" and d["usable"]
    err(c.post(f"{API}/sheet-templates/{tid}/pages/{pid}/retry", headers=H(c)), 409, "not_failed")
    # disable / enable (no deletion in v1)
    d = c.patch(f"{API}/sheet-templates/{tid}/pages/{pid}", json={"disabled": True}, headers=H(c)).json()
    assert d["usable"] is False and d["pages"]["disabled"] == 1 and d["page_list"][0]["disabled"] is True
    err(c.patch(f"{API}/sheet-templates/{tid}/pages/{pid}", json={}, headers=H(c)), 422, "validation_error")
    d = c.patch(f"{API}/sheet-templates/{tid}/pages/{pid}", json={"disabled": False}, headers=H(c)).json()
    assert d["usable"] is True
    err(c.patch(f"{API}/sheet-templates/{tid}/pages/pg_nope", json={"disabled": True}, headers=H(c)), 404, "not_found")
    # limits
    err(upload(c, tid, [(f"{i}.jpg", jpg(110 + i)) for i in range(11)]), 422, "too_many_files")
    monkeypatch.setattr(sheets, "MAX_PAGES", 3)
    j = upload(c, tid, [("a.jpg", jpg(120)), ("b.jpg", jpg(121)), ("c.jpg", jpg(122))]).json()
    assert [x["code"] for x in j["rejected"]] == ["too_many_pages"] and len(j["template"]["page_list"]) == 3
    assert sheets.wait_idle(10)


def test_template_essay_like_blank_page_warns(env, monkeypatch):
    long = ["This is a very long handwritten line full of letters " * 3] * 12
    fake_recognizer(monkeypatch, {100: long})
    c = env.clients["alice"]
    tid = c.post(f"{API}/sheet-templates", json={"name": "A"}, headers=H(c)).json()["id"]
    upload(c, tid, [("x.jpg", jpg(100))])
    assert sheets.wait_idle(10)
    assert c.get(f"{API}/sheet-templates/{tid}", headers=H(c)).json()["page_list"][0]["warnings"] == [
        "looks_like_written_page"]


def test_template_logs_and_responses_carry_no_text_content(env, monkeypatch):
    fake_recognizer(monkeypatch, {100: ["SECRET-PRINTED-LINE-XYZ"]})
    c = env.clients["alice"]
    tid = c.post(f"{API}/sheet-templates", json={"name": "A"}, headers=H(c)).json()["id"]
    upload(c, tid, [("x.jpg", jpg(100))])
    assert sheets.wait_idle(10)
    logs = "".join(p.read_text("utf-8", errors="replace") for p in env.cfg.logs_dir.rglob("*") if p.is_file())
    assert "SECRET-PRINTED-LINE-XYZ" not in logs


# ------------------------------------------------------------------ service: ocr.json with a template
class BoxEngine:
    def __init__(self, boxes):
        self.boxes, self.calls = boxes, 0

    def recognize(self, im):
        self.calls += 1
        from app.ocr import layout
        b = list(self.boxes)
        return PageResult(lines=layout.group_lines([(l.box, l.text, l.score) for l in b]), width=im.size[0],
                          height=im.size[1], layout_reliable=True, boxes=b)

    def info(self):
        return {"local_engine": "paddle", "name": "fake", "version": {}, "params": {}}


def test_run_for_subtask_applies_sheet_template(cfg, monkeypatch):
    _, sdir, _ = setup(cfg, None, pages=1, route="local_ocr", choice=None)
    eng = BoxEngine(photo())
    monkeypatch.setattr(service, "_get_engine", lambda cfg, name: eng)
    tid = make_sheet(cfg, "u1", lines=TEMPLATE)
    snap = {"id": tid, "name": "IG", "page_ids": [f"pg_{tid}_0"]}
    doc = service.run_for_subtask(cfg, SID, "paddle", snap)
    text = doc["transcription"]
    assert "Cambridge" not in text and "0500" not in text and "9876543210" not in text and "Turn over" not in text
    assert "Lily Chen" in text and "grate fun" in text and "recieve" in text
    pg = rd(sdir / "ocr.json")["pages"][0]
    assert {e["reason"] for e in pg["excluded"]} == {"template_text", "template_region"} and len(pg["excluded"]) == 7
    assert pg["sheet_match"]["aligned"] is True and pg["sheet_match"]["template_page"] == f"pg_{tid}_0"
    assert doc["sheet_template"] == {"id": tid, "page_ids": [f"pg_{tid}_0"]}
    assert meta_of(cfg)["local_ocr_state"] == "done"
    service.run_for_subtask(cfg, SID, "paddle", snap)                     # same template -> reused
    assert eng.calls == 1
    service.run_for_subtask(cfg, SID, "paddle", None)                     # different template key -> redone
    assert eng.calls == 2
    assert "Cambridge" in rd(sdir / "ocr.json")["transcription"]


def test_recognize_one_uses_default_engine_and_queue(cfg, monkeypatch):
    eng = BoxEngine([Line("Name", [0, 0, 10, 10])])
    seen = []
    monkeypatch.setattr(service, "_get_engine", lambda cfg, name: seen.append(name) or eng)
    res, info = service.recognize_one(cfg, __import__("PIL.Image", fromlist=["Image"]).new("RGB", (20, 20)))
    assert seen == ["paddle"] and res.boxes[0].text == "Name" and info["local_engine"] == "paddle"


def test_recognize_one_engine_error_is_ocr_failed(cfg, monkeypatch):
    class Bad(BoxEngine):
        def recognize(self, im):
            raise OcrError("x")
    monkeypatch.setattr(service, "_get_engine", lambda cfg, name: Bad([]))
    from PIL import Image
    with pytest.raises(service.OcrFailed):
        service.recognize_one(cfg, Image.new("RGB", (20, 20)))


def test_filter_long_printed_line_copied_into_essay_is_kept_when_aligned():
    copied = Line("Write your answer in the space below", tf([100, 700, 800, 740]), 0.9)
    kept, excl, info = sheet_filter.filter_page(photo() + [copied], [{"page_id": "p", "lines": TEMPLATE}])
    assert info["aligned"]
    assert [b.text for b in kept].count("Write your answer in the space below") == 1
    assert reasons(excl)["Write your answer in the space below"] == "template_text"     # the printed one


@pytest.mark.parametrize("seed", range(6))
def test_filter_robust_to_detector_jitter_and_tilt(seed):
    import random
    rnd = random.Random(seed)
    deg = rnd.uniform(-8, 8)
    s = rnd.uniform(0.7, 2.2)
    boxes = [Line(b.text, [v + rnd.uniform(-6, 6) for v in b.box], b.score)
             for b in photo(s=s, deg=deg, tx=rnd.uniform(-80, 80), ty=rnd.uniform(-80, 80))]
    kept, excl, info = sheet_filter.filter_page(boxes, [{"page_id": "p", "lines": TEMPLATE}])
    assert info["aligned"], info
    assert [b.text for b in kept] == ["Lily Chen"] + ESSAY
    assert len(excl) == 7



def test_filter_vertical_margin_note_read_differently_is_removed():
    """The blank's vertical margin note was misread; on the photo it reads correctly -> removed by position."""
    tpl = TEMPLATE + [{"text": "OIGIN", "box": [1020, 300, 1060, 1500]}]
    boxes = photo() + [Line("DO NOT WRITE IN THIS MARGIN", tf([1018, 310, 1062, 1490]), 0.95),
                       Line("Hello", tf([980, 1700, 1100, 1740]), 0.9)]     # horizontal word: not vertical
    kept, excl, info = sheet_filter.filter_page(boxes, [{"page_id": "p", "lines": tpl}])
    assert info["aligned"]
    assert reasons(excl)["DO NOT WRITE IN THIS MARGIN"] == "template_region"
    assert "Hello" in [b.text for b in kept]


# ------------------------------------------------------------------ deskew (tilted photos)
from app.ocr import fast_ocr, layout  # noqa: E402


def _tilted_lines(deg, texts, x0=200, y0=300, pitch=100, w=1800, h=50, short=()):
    """Raw boxes of a page rotated by `deg` (image coords): AABB + rotated rectangle, like the detector."""
    a = math.radians(deg)
    ox, oy = 1240, 1754
    out = []
    for k, t in enumerate(texts):
        ww = w * 0.35 if k in short else w
        cx, cy = x0 + ww / 2, y0 + pitch * k + h / 2
        dx, dy = cx - ox, cy - oy
        rx, ry = ox + dx * math.cos(a) - dy * math.sin(a), oy + dx * math.sin(a) + dy * math.cos(a)
        bw = abs(ww * math.cos(a)) + abs(h * math.sin(a))
        bh = abs(ww * math.sin(a)) + abs(h * math.cos(a))
        out.append(Line(t, [rx - bw / 2, ry - bh / 2, rx + bw / 2, ry + bh / 2], 0.9, (rx, ry, ww, h, deg)))
    return out


def test_group_boxes_deskews_tilted_page_reading_order():
    texts = [f"line number {k} of the essay text" for k in range(8)]
    boxes = _tilted_lines(4.0, texts, short=(3, 6))
    plain = [l.text for l in layout.group_lines([(b.box, b.text, b.score) for b in boxes])]
    assert plain != texts                                      # tilt breaks naive ordering / merges rows
    lines, skew = layout.group_boxes(boxes, 2480, 3508)
    assert abs(skew - 4.0) < 0.01 and [l.text for l in lines] == texts
    # deskewed left edges line up again (no false indents)
    lefts = [l.box[0] for l in lines]
    assert max(lefts) - min(lefts) < 5


def test_group_boxes_small_or_unknown_tilt_is_untouched():
    boxes = [Line("a b c d e f", [100, 100, 900, 140]), Line("g h i j k l", [100, 200, 900, 240])]
    lines, skew = layout.group_boxes(boxes, 1000, 1000)
    assert skew == 0.0 and [l.box for l in lines] == [[100, 100, 900, 140], [100, 200, 900, 240]]
    nearly = _tilted_lines(0.1, ["one two three four", "five six seven", "eight nine ten"])
    assert layout.group_boxes(nearly, 2480, 3508)[1] == 0.0


def test_estimate_skew_ignores_vertical_and_short_boxes():
    boxes = _tilted_lines(-2.0, ["aaaa aaaa aaaa", "bbbb bbbb bbbb", "cccc cccc cccc"])
    boxes += [Line("MARGIN", [10, 10, 50, 900], 0.9, (30, 455, 890, 40, 90.0)),
              Line("x", [0, 0, 20, 20], 0.9, (10, 10, 20, 20, 45.0))]
    assert abs(layout.estimate_skew(boxes) + 2.0) < 1e-6


def test_quad_rect_and_vertical_detection():
    import numpy as np
    a = math.radians(3)
    c, s_ = math.cos(a), math.sin(a)
    pts = np.array([[0, 0], [400, 0], [400, 40], [0, 40]], float)
    rot = np.array([[x * c - y * s_ + 100, x * s_ + y * c + 50] for x, y in pts])
    q = fast_ocr._order_quad(rot)
    cx, cy, w, h, ang = fast_ocr.quad_rect(q)
    assert abs(w - 400) < 1 and abs(h - 40) < 1 and abs(ang - 3) < 0.1
    assert not fast_ocr.is_vertical(q)
    vq = fast_ocr._order_quad(np.array([[0, 0], [40, 0], [40, 600], [0, 600]], float))
    assert fast_ocr.is_vertical(vq) and abs(abs(fast_ocr.quad_rect(vq)[4]) - 90) < 0.1
