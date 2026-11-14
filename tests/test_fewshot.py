import json

from app.batches import paths
from app.core import atomic_io
from app.grading import fewshot, history
from app.grading.revision import make_revision  # noqa: F401  (import check)

AI = {"判分": 6, "判据": [{"name": "ai-内容", "body": "ai body"}], "评语": "ai comment"}


def seed(cfg, owner, code, subs, *, topic="T", rubric="R", genre="narrative", full=10, created="2026-11-01T10:00:00",
         tpl=("tpl_a", 1)):
    """subs: list of dict(transcription, ai(dict|None), review(dict|None), attempts({n: parsed}), thinking)"""
    bdir = paths.batch_dir(cfg, code)
    bdir.mkdir(parents=True, exist_ok=True)
    atomic_io.write_json_atomic(bdir / "batch.json", {
        "schema_version": 1, "task_code": code, "owner_id": owner, "created_at": created, "topic": topic,
        "rubric": rubric, "genre": genre, "full_score": full, "status": "completed",
        "config_revisions": [{"rev": 1, "template": {"id": tpl[0], "version": tpl[1]}}]})
    ids = []
    for i, s in enumerate(subs, start=1):
        sid = f"{code}{i:05d}"
        ids.append(sid)
        sd = bdir / sid
        sd.mkdir(exist_ok=True)
        atomic_io.write_json_atomic(sd / "meta.json", {"subtask_id": sid, "task_code": code, "owner_id": owner,
                                                       "status": "done", "config_rev": 1})
        if s.get("transcription") is not None:
            atomic_io.write_json_atomic(sd / "ocr.json", {"transcription": s["transcription"]})
        if s.get("ai") is not None:
            atomic_io.write_json_atomic(sd / "grading.json", {"result": s["ai"], "score": 1})
        for n, parsed in (s.get("attempts") or {}).items():
            atomic_io.write_json_atomic(sd / "llm" / f"{n:03d}.response.json",
                                        {"attempt": n, "parsed": parsed, "thinking": "SECRET-THINK"})
        if s.get("review") is not None:
            atomic_io.write_json_atomic(sd / "review.json", s["review"])
    return ids


def rv(score=None, items=None, adopted=False, ai_attempt=None, at="2026-11-02T10:00:00"):
    return {"final_score": score, "items": items, "adopted_ai": adopted, "ai_attempt": ai_attempt, "saved_at": at}


def sub(text="my essay text", ai=AI, review=None, **kw):
    return {"transcription": text, "ai": ai, "review": review if review is not None else rv(8), **kw}


def draft(**kw):
    return {"topic": "T", "rubric": "R", "genre": "narrative", **kw}


def test_isolation_by_owner(cfg):
    a = seed(cfg, "u_a", "141120260001", [sub()])
    b = seed(cfg, "u_b", "141120260002", [sub()])
    assert fewshot.select_shots(cfg, "u_a", "own", draft(), set(), 3, 6000) == a
    assert fewshot.select_shots(cfg, "u_b", "own", draft(), set(), 3, 6000) == b
    assert set(fewshot.select_shots(cfg, "u_root", "all", draft(), set(), 3, 6000)) == set(a + b)
    assert fewshot.select_shots(cfg, "u_c", "own", draft(), set(), 3, 6000) == []
    assert [r["owner_id"] for r in history.iter_records(cfg, "u_a")] == ["u_a"]
    assert len(list(history.iter_records(cfg, None))) == 2


def test_exclusions_batch_and_subtask(cfg):
    ids = seed(cfg, "u", "141120260001", [sub(), sub()])
    other = seed(cfg, "u", "141120260002", [sub()])
    got = fewshot.select_shots(cfg, "u", "own", draft(task_code="141120260001"), set(), 5, 6000)
    assert got == other
    got = fewshot.select_shots(cfg, "u", "own", draft(), {"141120260001"}, 5, 6000)
    assert got == other
    # regrade: the subtask itself AND its whole batch are excluded
    got = fewshot.select_shots(cfg, "u", "own", draft(subtask_id=ids[0]), set(), 5, 6000)
    assert got == other
    got = fewshot.select_shots(cfg, "u", "own", draft(), set(), 5, 6000, exclude_subtask_ids={ids[0]})
    assert ids[0] not in got and ids[1] in got


def test_only_reviewed_nonempty_transcription_and_genre(cfg):
    seed(cfg, "u", "141120260001", [
        {"transcription": "x", "ai": AI, "review": None},              # no review
        sub(text="<empty>"), sub(text="   "),
        {"transcription": None, "ai": AI, "review": rv(5)},             # no ocr.json
        sub(text=" <empty> "),
    ])
    good = seed(cfg, "u", "141120260002", [sub()])
    seed(cfg, "u", "141120260003", [sub()], genre="argumentative")
    assert fewshot.select_shots(cfg, "u", "own", draft(), set(), 9, 6000) == good


def test_standard_answer_adopted_override_noai(cfg):
    items = [{"name": "我的内容", "body": "老师写的"}]
    ids = seed(cfg, "u", "141120260001", [
        sub(review=rv(9, items, adopted=True, ai_attempt=2), attempts={1: {"判分": 1, "判据": []}, 2: AI}),
        sub(review=rv(9, items)),                                    # override with items
        sub(review=rv(4, None)),                                     # override score only: keep AI criteria
        sub(ai=None, review=rv(3, items)),                           # no AI result
        sub(ai=None, review=rv(None, None)),                         # nothing usable
    ])
    recs = {r["subtask_id"]: r for r in history.iter_records_ext(cfg, "u")}

    adopted = fewshot.build_standard_answer(recs[ids[0]])
    assert adopted == AI                                              # the attempt named in ai_attempt
    ov = fewshot.build_standard_answer(recs[ids[1]])
    assert ov["判分"] == 9 and ov["判据"] == items and ov["评语"] == "ai comment"
    ov2 = fewshot.build_standard_answer(recs[ids[2]])
    assert ov2["判分"] == 4 and ov2["判据"] == AI["判据"]
    noai = fewshot.build_standard_answer(recs[ids[3]])
    assert noai == {"判分": 3, "判据": items}
    assert fewshot.build_standard_answer(recs[ids[4]]) is None


def test_text_contains_no_thinking_and_has_fields(cfg):
    ids = seed(cfg, "u", "141120260001", [
        sub(text="Dear Tom, my essay", review=rv(9, None, adopted=True, ai_attempt=1), attempts={1: AI})],
        topic="My Trip", rubric="Rubric X")
    txt = fewshot.build_fewshot_text(cfg, ids)
    assert "SECRET-THINK" not in txt
    assert "My Trip" in txt and "Rubric X" in txt and "Dear Tom, my essay" in txt
    ans = json.loads(txt.split("标准答案：\n", 1)[1])
    assert ans == AI
    assert fewshot.build_fewshot_text(cfg, []) == ""
    assert fewshot.build_fewshot_text(cfg, ["999999999999 99999"]) == ""


def test_rekey_to_target_template_keys(cfg):
    ids = seed(cfg, "u", "141120260001", [sub(review=rv(7, [{"name": "n", "body": "b"}]))])
    txt = fewshot.build_fewshot_text(cfg, ids, "score", "criteria")
    ans = json.loads(txt.split("标准答案：\n", 1)[1])
    assert ans["score"] == 7 and ans["criteria"] == [{"name": "n", "body": "b"}] and "判分" not in ans


def test_k_and_budget(cfg):
    for i in range(5):
        seed(cfg, "u", f"1411202600{i:02d}", [sub(text="w " * 50)], created=f"2026-11-0{i + 1}T10:00:00")
    assert len(fewshot.select_shots(cfg, "u", "own", draft(), set(), 3, 100000)) == 3
    assert len(fewshot.select_shots(cfg, "u", "own", draft(), set(), 99, 100000)) == 5
    assert fewshot.select_shots(cfg, "u", "own", draft(), set(), 0, 100000) == []
    # budget: one shot costs ~ (len/3.5) tokens; allow roughly two
    recs = list(history.iter_records_ext(cfg, "u"))
    ans = fewshot.build_standard_answer(recs[0])
    cost = fewshot.approx_tokens(fewshot.shot_text(recs[0], ans, 1))
    got = fewshot.select_shots(cfg, "u", "own", draft(), set(), 99, cost * 2 + 5)
    assert len(got) == 2
    assert fewshot.select_shots(cfg, "u", "own", draft(), set(), 99, cost - 1) == []
    # defaults come from settings
    assert len(fewshot.find_fewshot(cfg, "u", "own", draft())) == 3


def test_big_shot_skipped_but_small_later_kept(cfg):
    seed(cfg, "u", "141120260001", [sub(text="small")], created="2026-11-01T10:00:00")
    seed(cfg, "u", "141120260002", [sub(text="x" * 5000)], created="2026-11-05T10:00:00")     # newest, huge
    got = fewshot.select_shots(cfg, "u", "own", draft(), set(), 3, 300)
    assert got == ["14112026000100001"]


def test_priority_topic_then_template_then_newest(cfg):
    old_same = seed(cfg, "u", "141120260001", [sub()], created="2026-11-01T10:00:00")
    new_other = seed(cfg, "u", "141120260002", [sub()], topic="Other", created="2026-11-09T10:00:00")
    mid_same = seed(cfg, "u", "141120260003", [sub()], created="2026-11-05T10:00:00", tpl=("tpl_b", 1))
    # same topic first (newest of them first), different topic last
    got = fewshot.select_shots(cfg, "u", "own", draft(), set(), 5, 100000)
    assert got == mid_same + old_same + new_other
    # whitespace/case-insensitive topic+rubric hash
    got = fewshot.select_shots(cfg, "u", "own", draft(topic="  t ", rubric="r"), set(), 5, 100000)
    assert got == mid_same + old_same + new_other
    # tie on topic -> same template id first
    got = fewshot.select_shots(cfg, "u", "own", draft(template_id="tpl_a"), set(), 5, 100000)
    assert got == old_same + mid_same + new_other


def test_history_record_shape(cfg):
    seed(cfg, "u", "141120260001", [sub(review=rv(8, [{"name": "a", "body": "b"}], True, 1))], topic="TT")
    rec = next(history.iter_records(cfg, "u"))
    assert set(rec) == {"task_code", "subtask_id", "owner_id", "topic", "rubric", "genre", "full_score",
                        "transcription", "ai_result", "review", "template"}
    assert rec["review"] == {"final_score": 8, "items": [{"name": "a", "body": "b"}], "adopted_ai": True}
    assert rec["template"] == {"id": "tpl_a", "version": 1}
    assert rec["ai_result"] == AI


def test_unreadable_dirs_skipped(cfg):
    ids = seed(cfg, "u", "141120260001", [sub(), sub()])
    (paths.subtask_dir(cfg, ids[0]) / "meta.json").write_text("{not json", encoding="utf-8")
    assert [r["subtask_id"] for r in history.iter_records(cfg, "u")] == [ids[1]]
