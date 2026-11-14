import json
import threading
import time

import numpy as np
import pytest

from app.batches import executor, hooks
from app.ocr import layout, service
from app.ocr.common import Line, OcrError, PageResult
from app.ocr import fast_ocr
from tests.mock_llm_server import mock_llm  # noqa: F401
from tests.test_grading_pipeline import SID, TC, meta_of, rd, run, setup


@pytest.fixture(autouse=True)
def _clean():
    prev = executor.get_processor()
    yield
    hooks.reset_defaults()
    executor.set_processor(prev)
    service._engine.update(name=None, obj=None)
    service._env_cache.update(t=0.0, v=None)


# ------------------------------------------------------------------ layout
def L(text, x0, y0, x1, y1):
    return Line(text, [x0, y0, x1, y1])


def full(text, row, x0=100, x1=900):
    return L(text, x0, 100 + 60 * row, x1, 140 + 60 * row)


def test_group_lines_merges_words_on_one_row_in_x_order():
    items = [([200, 10, 260, 40], "world", 0.9), ([10, 12, 120, 42], "hello", 0.8), ([10, 100, 100, 130], "next", 0.7)]
    out = layout.group_lines(items)
    assert [l.text for l in out] == ["hello world", "next"]


def test_restore_joins_paragraph_lines_and_dehyphenates():
    lines = [full("Nowadays there is many debate about free h-", 0), full("igher education for all studens and the", 1),
             full("goverment pays.", 2, x1=400)]
    assert layout.restore_text(lines) == "Nowadays there is many debate about free higher education for all studens and the goverment pays."


def test_restore_paragraph_breaks_indent_and_blank_line():
    lines = [full("First paragraph line one goes all the way", 0), full("and ends early.", 1, x1=420),
             full("Second paragraph begins at the margin and", 2), full("continues here to the right edge of page", 3),
             full("Third one is indented and runs full width", 4, x0=170), full("to the end.", 5, x1=380),
             L("Fourth after a big gap runs the full width", 100, 100 + 60 * 8, 900, 140 + 60 * 8)]
    rows = layout.restore_text(lines).split("\n")
    assert rows[0] == "First paragraph line one goes all the way and ends early."   # short line ends paragraph
    assert rows[1] == "Second paragraph begins at the margin and continues here to the right edge of page"
    assert rows[2].startswith("  ") and rows[2].strip().startswith("Third one is indented")   # indent kept
    assert rows[3] == ""                                                    # vertical gap -> blank line
    assert rows[4] == "Fourth after a big gap runs the full width"
    assert len(rows) == 5


def test_restore_real_hyphen_mid_line_is_kept():
    lines = [full("a well-educated population may take better", 0), full("care.", 1, x1=200)]
    assert layout.restore_text(lines) == "a well-educated population may take better care."


def test_restore_pages_continue_paragraph_across_page_break():
    p1 = [full("The essay starts on the first page and runs", 0), full("to the very end of the first sheet so it", 1)]
    p2 = [full("continues on the second page here.", 0, x1=500), full("A new paragraph is indented on page two", 1, x0=170)]
    rows = layout.restore_pages([p1, p2]).split("\n")
    assert rows[0].endswith("first sheet so it continues on the second page here.")
    assert rows[1].strip().startswith("A new paragraph") and rows[1].startswith("  ")


def test_restore_tiny_fragment_stays_inline():
    # real case (test1): a 24 px high "f" between two 40 px lines must not start an indented paragraph
    lines = [full("but it should not be com", 0), L("f", 240, 143, 265, 167),
             full("pletely free without any responsibility and", 1), full("more words here.", 2, x1=400)]
    out = layout.restore_text(lines)
    assert "\n" not in out and out.startswith("but it should not be com f pletely")


def test_restore_sentence_end_moderately_short_line_breaks_paragraph():
    # real case (test1): line ends with "." about 12% of the width before the right edge -> new paragraph
    lines = [full("This can teach them responsibility and the", 0),
             full("quality of education stays high.", 1, x1=800),
             full("In conclusion, although free education has", 2), full("many benefits.", 3, x1=400)]
    rows = layout.restore_text(lines).split("\n")
    assert len(rows) == 2 and rows[1].startswith("In conclusion")


def test_restore_line_without_punctuation_slightly_short_continues():
    lines = [full("a line that wraps because the next word is", 0, x1=820),
             full("long enough to move to the following line.", 1)]
    assert "\n" not in layout.restore_text(lines)


def test_restore_small_footer_below_text_is_its_own_line():
    # real case (test1): the answer sheet's printed footer is small but sits below the essay -> not inline
    lines = [full("Some paragraph that runs the full width of the", 0), full("page and then ends here.", 1, x1=400),
             L("共 页", 700, 400, 800, 425)]
    rows = layout.restore_text(lines).split("\n")
    assert rows[0] == "Some paragraph that runs the full width of the page and then ends here."
    assert rows[-1].strip() == "共 页" and rows[0] != rows[-1]       # a line of its own


def test_restore_empty_and_single():
    assert layout.restore_text([]) == ""
    assert layout.restore_text([L("only", 0, 0, 10, 10)]) == "only"


# ------------------------------------------------------------------ fast_ocr helpers (no models needed)
def test_det_size_keeps_original_resolution():
    assert fast_ocr.det_size(2112, 2976) == (2112, 2976)          # no downscaling (quality rule)
    assert fast_ocr.det_size(2100, 2970) == (2112, 2976)          # only rounded to a multiple of 32
    w, h = fast_ocr.det_size(500, 700)                            # short side < 736 -> enlarged
    assert min(w, h) >= 736 - 32


def test_db_postprocess_two_bars_in_source_coordinates():
    p = np.zeros((200, 400), dtype=np.float32)
    p[30:50, 20:300] = 0.9
    p[100:122, 40:360] = 0.95
    p[160:162, 10:12] = 0.9                                       # speck below min size
    boxes = fast_ocr.db_postprocess(p, 800, 400)                  # source image is 2x the map
    assert len(boxes) == 2
    for q in boxes:
        assert q.shape == (4, 2) and q[:, 0].max() <= 800 and q[:, 1].max() <= 400


def test_db_postprocess_low_score_dropped():
    p = np.zeros((100, 100), dtype=np.float32)
    p[10:30, 10:80] = 0.3                                         # above thresh 0.2, below box_thresh 0.45
    assert fast_ocr.db_postprocess(p, 100, 100) == []


def test_ctc_decode_collapses_repeats_and_blanks():
    chars = ["blank", "a", "b"]
    text, score = fast_ocr.ctc_decode(np.array([1, 1, 0, 1, 2, 2, 0]), np.array([.9, .9, .5, .8, .7, .7, .5]), chars)
    assert text == "aab" and 0.7 < score < 0.9


# ------------------------------------------------------------------ service
class FakeEngine:
    def __init__(self, text="Dear Tom,\n\nI recieve your letter.", fail=False, reliable=False):
        self.text, self.fail, self.reliable, self.calls = text, fail, reliable, 0

    def recognize(self, im):
        self.calls += 1
        if self.fail:
            raise OcrError("boom")
        lines = [Line(t, [0, 40 * i, 100, 40 * i + 30]) for i, t in enumerate(self.text.split("\n"))]
        return PageResult(lines=lines, width=im.size[0], height=im.size[1], raw=self.text,
                          layout_reliable=self.reliable)

    def info(self):
        return {"local_engine": "paddle", "name": "fake", "version": {}, "params": {}}


def _patch_engine(monkeypatch, eng):
    monkeypatch.setattr(service, "_get_engine", lambda cfg, name: eng)


def test_run_for_subtask_writes_ocr_json_and_reuses(cfg, monkeypatch):
    ctx, sdir, _ = setup(cfg, None, pages=2, route="local_ocr", choice=None)
    eng = FakeEngine()
    _patch_engine(monkeypatch, eng)
    doc = service.run_for_subtask(cfg, SID, "paddle")
    assert doc["transcription_source"] == "local_ocr" and "recieve" in doc["transcription"]
    assert eng.calls == 2 and len(doc["pages"]) == 2
    assert doc["pages"][0]["image_ref"]["path"] == "image/original_1.jpg"
    assert rd(sdir / "ocr.json")["engine"]["local_engine"] == "paddle"
    assert meta_of(cfg)["local_ocr_state"] == "done"
    service.run_for_subtask(cfg, SID, "paddle")                      # reuse, no new recognition
    assert eng.calls == 2


def test_run_for_subtask_failure_and_empty(cfg, monkeypatch):
    setup(cfg, None, route="local_ocr", choice=None)
    _patch_engine(monkeypatch, FakeEngine(fail=True))
    with pytest.raises(service.OcrFailed):
        service.run_for_subtask(cfg, SID, "paddle")
    assert meta_of(cfg)["local_ocr_state"] == "failed"
    _patch_engine(monkeypatch, FakeEngine(text="   "))
    with pytest.raises(service.OcrEmpty):
        service.run_for_subtask(cfg, SID, "paddle")


def test_queue_is_serial_and_fifo():
    order, active, peak = [], [0], [0]

    def job(i):
        with service._QUEUE:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            order.append(i)
            time.sleep(0.05)
            active[0] -= 1

    ts = []
    for i in range(5):
        t = threading.Thread(target=job, args=(i,))
        t.start()
        ts.append(t)
        time.sleep(0.01)
    for t in ts:
        t.join()
    assert peak[0] == 1 and order == [0, 1, 2, 3, 4]


def test_env_check_reports_missing_model_dir(cfg, tmp_path):
    cfg.settings["ocr"]["model_root"] = str(tmp_path)
    ok, reason = service.engine_status(cfg, "paddle")
    assert not ok and "model folder" in reason
    assert service.engine_status(cfg, "hunyuan") == (False, "unknown local engine 'hunyuan'")
    service._env_cache.update(t=0.0, v=None)
    assert service.local_env_ok(cfg)[0] is False
    assert (cfg.state_dir / "env_check.json").exists()


# ------------------------------------------------------------------ pipeline integration
def test_route1_sends_ocr_text_not_images(cfg, mock_llm, monkeypatch):
    ctx, sdir, _ = setup(cfg, mock_llm, pages=2, route="local_ocr", choice=None,
                         tpl_over={"input_mode": "text"})
    ctx.batch["local_engine"] = "paddle"
    _patch_engine(monkeypatch, FakeEngine(text="Dear Tom,\nI recieve it."))
    mock_llm.script(["ok_json"])
    out = run(ctx)
    assert out.status == "done"
    reqs = mock_llm.chat_requests()
    assert len(reqs) == 1
    user = reqs[0]["body"]["messages"][1]
    assert isinstance(user["content"], str)                          # plain text, no image parts
    assert user["content"].count("recieve") == 2                     # two pages, same fake text each
    assert "data:image" not in json.dumps(reqs[0]["body"])
    saved = rd(sdir / "llm" / "001.request.json")
    assert saved["messages"][1]["content"] == user["content"]
    assert rd(sdir / "ocr.json")["mode"] == "local_ocr"


def test_route1_ocr_failure_never_calls_llm(cfg, mock_llm, monkeypatch):
    ctx, _, _ = setup(cfg, mock_llm, route="local_ocr", choice=None, tpl_over={"input_mode": "text"})
    ctx.batch["local_engine"] = "paddle"
    _patch_engine(monkeypatch, FakeEngine(fail=True))
    out = run(ctx)
    assert (out.status, out.fail_reason) == ("failed", "ocr_failed")
    assert mock_llm.chat_requests() == []


def test_route1_empty_text_is_ocr_empty(cfg, mock_llm, monkeypatch):
    ctx, _, _ = setup(cfg, mock_llm, route="local_ocr", choice=None, tpl_over={"input_mode": "text"})
    ctx.batch["local_engine"] = "paddle"
    _patch_engine(monkeypatch, FakeEngine(text=""))
    out = run(ctx)
    assert (out.status, out.fail_reason) == ("failed", "ocr_empty")
    assert mock_llm.chat_requests() == []


def test_vlm_local_transcription_keeps_images_and_saves_ocr(cfg, mock_llm, monkeypatch):
    ctx, sdir, _ = setup(cfg, mock_llm, route="vlm", choice="local_transcription")
    ctx.batch["local_engine"] = "paddle"
    _patch_engine(monkeypatch, FakeEngine(text="Hello there"))
    mock_llm.script(["ok_json"])
    assert run(ctx).status == "done"
    user = mock_llm.chat_requests()[0]["body"]["messages"][1]
    assert isinstance(user["content"], list) and user["content"][1]["type"] == "image_url"   # VLM still sees images
    doc = rd(sdir / "ocr.json")
    assert doc["transcription"] == "Hello there" and doc["transcription_source"] == "local_ocr"
