"""Few-shot selection and rendering. Text only (no images), never includes model thinking.

Eligibility: record belongs to owner (scope "all" = every account; root-only is enforced by the caller),
has a teacher review, transcription not empty / "<empty>", same genre as the draft (different genre never
used), not in the excluded batches / subtasks.
Ranking: (same genre & same normalised topic+rubric hash) first, then same genre; ties -> same template id
first; then newest. Token budget: approx tokens = ceil(len/3.5); shots that do not fit the remaining budget
are skipped.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from . import history

EMPTY = "<empty>"


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def topic_hash(topic, rubric) -> str:
    return hashlib.sha1((_norm(topic) + "\n" + _norm(rubric)).encode("utf-8")).hexdigest()


def approx_tokens(text: str) -> int:
    return math.ceil(len(text) / 3.5)


def _has_transcription(rec: dict) -> bool:
    t = (rec.get("transcription") or "").strip()
    return bool(t) and t != EMPTY


def _teacher_items(rv: dict):
    items = rv.get("items")
    return items if isinstance(items, list) and items else None


def _ai_for_review(rec: dict) -> dict | None:
    """AI result the review refers to: the attempt named in review.ai_attempt, else grading.json."""
    n = rec.get("_ai_attempt")
    if n is not None and rec.get("_dir") is not None:
        p = history.attempt_parsed(rec["_dir"], n)
        if p is not None:
            return p
    return rec.get("ai_result")


def build_standard_answer(rec: dict, score_key: str = "判分", criteria_key: str = "判据",
                          src_score_key: str | None = None, src_criteria_key: str | None = None) -> dict | None:
    """The 'correct' answer a shot teaches, or None if there is no usable review.
    adopted_ai -> the AI result; override -> AI result with score_key = teacher score and criteria_key =
    teacher items (if any); no AI result -> {score_key: teacher score, criteria_key: items}."""
    rv = rec.get("review")
    if not rv:
        return None
    sk0 = src_score_key or score_key
    ck0 = src_criteria_key or criteria_key
    ai = _ai_for_review(rec)
    items = _teacher_items(rv)
    score = rv.get("final_score")
    if rv.get("adopted_ai") and isinstance(ai, dict):
        out = _rekey(ai, sk0, ck0, score_key, criteria_key)
        return out
    if isinstance(ai, dict):
        out = _rekey(ai, sk0, ck0, score_key, criteria_key)
        if score is not None:
            out[score_key] = score
        if items is not None:
            out[criteria_key] = items
        return out
    if score is None:
        return None
    return {score_key: score, criteria_key: items if items is not None else []}


def _rekey(ai: dict, sk0: str, ck0: str, sk: str, ck: str) -> dict:
    out = dict(ai)
    if sk0 != sk and sk0 in out and sk not in out:
        out = {(sk if k == sk0 else k): v for k, v in out.items()}
    if ck0 != ck and ck0 in out and ck not in out:
        out = {(ck if k == ck0 else k): v for k, v in out.items()}
    return out


def shot_text(rec: dict, answer: dict, index: int | None = None) -> str:
    head = f"### 示例{index}" if index else "### 示例"
    ans = json.dumps(answer, ensure_ascii=False)
    return (f"{head}\n题目：{rec.get('topic') or ''}\n评分标准：{rec.get('rubric') or ''}\n"
            f"作文转写：\n{(rec.get('transcription') or '').strip()}\n标准答案：\n{ans}")


def _template_keys(cfg, tpl: dict | None) -> tuple[str | None, str | None]:
    if not tpl or not tpl.get("id"):
        return None, None
    try:
        from ..prompts import store as pstore
        v = pstore.get_version(cfg, tpl["id"], tpl.get("version"))
        if isinstance(v, dict):
            return v.get("score_key"), v.get("criteria_key")
    except Exception:
        pass
    return None, None


def _keys_for(cfg, rec: dict, score_key, criteria_key) -> tuple[str, str, str | None, str | None]:
    s0, c0 = _template_keys(cfg, rec.get("template"))
    return (score_key or s0 or "判分", criteria_key or c0 or "判据", s0 or "判分", c0 or "判据")


def _eligible(rec: dict, genre: str, exclude_codes: set, exclude_subs: set) -> bool:
    if rec.get("task_code") in exclude_codes or rec.get("subtask_id") in exclude_subs:
        return False
    if not rec.get("review") or not _has_transcription(rec):
        return False
    return _norm(rec.get("genre")) == genre


def select_shots(cfg, owner_id: str, scope: str, draft: dict, exclude_task_codes=None,
                 k: int | None = None, token_budget: int | None = None,
                 exclude_subtask_ids=None) -> list[str]:
    k = int(cfg["fewshot_k"] if k is None else k)
    budget = int(cfg["fewshot_token_budget"] if token_budget is None else token_budget)
    if k <= 0:
        return []
    exclude_codes = set(exclude_task_codes or ())
    exclude_subs = set(exclude_subtask_ids or ())
    if draft.get("task_code"):
        exclude_codes.add(draft["task_code"])
    if draft.get("subtask_id"):                 # regrade: the subtask itself AND its batch
        exclude_subs.add(draft["subtask_id"])
        exclude_codes.add(str(draft["subtask_id"])[:12])
    exclude_codes |= set(draft.get("exclude_task_codes") or ())
    genre = _norm(draft.get("genre"))
    want_hash = topic_hash(draft.get("topic"), draft.get("rubric"))
    want_tpl = draft.get("template_id")
    tkeys = (draft.get("score_key"), draft.get("criteria_key"))
    recs = []
    for rec in history.iter_records_ext(cfg, None if scope == "all" else owner_id):
        if scope != "all" and rec.get("owner_id") != owner_id:
            continue
        if _eligible(rec, genre, exclude_codes, exclude_subs):
            recs.append(rec)

    def rank(rec: dict):
        same_topic = topic_hash(rec.get("topic"), rec.get("rubric")) == want_hash
        same_tpl = bool(want_tpl) and (rec.get("template") or {}).get("id") == want_tpl
        when = rec.get("_review_at") or rec.get("_created_at") or ""
        return (0 if same_topic else 1, 0 if same_tpl else 1, _neg(when), _neg(rec.get("subtask_id") or ""))

    recs.sort(key=rank)
    out: list[str] = []
    remaining = budget
    for rec in recs:
        if len(out) >= k:
            break
        sk, ck, s0, c0 = _keys_for(cfg, rec, *tkeys)
        ans = build_standard_answer(rec, sk, ck, s0, c0)
        if ans is None:
            continue
        cost = approx_tokens(shot_text(rec, ans, len(out) + 1))
        if cost > remaining:
            continue
        remaining -= cost
        out.append(rec["subtask_id"])
    return out


class _neg:
    """Sort key wrapper: descending order for strings inside an otherwise ascending tuple."""
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def __lt__(self, o):
        return self.v > o.v

    def __eq__(self, o):
        return self.v == o.v


def build_fewshot_text(cfg, shot_ids, score_key: str | None = None, criteria_key: str | None = None) -> str:
    """Rendered few-shot block for the given subtask ids ("" when none / unreadable). Optional keys: the
    target template's score/criteria keys (the standard answers are re-keyed to them)."""
    parts = []
    for sid in shot_ids or []:
        rec = history.get_record_ext(cfg, sid)
        if rec is None or not rec.get("review") or not _has_transcription(rec):
            continue
        sk, ck, s0, c0 = _keys_for(cfg, rec, score_key, criteria_key)
        ans = build_standard_answer(rec, sk, ck, s0, c0)
        if ans is None:
            continue
        parts.append(shot_text(rec, ans, len(parts) + 1))
    return "\n\n".join(parts)


def find_fewshot(cfg, owner_id: str, scope: str, draft: dict) -> list[str]:
    """Hook signature used by app.batches.hooks.find_fewshot."""
    return select_shots(cfg, owner_id, scope, draft or {}, set(), None, None)
