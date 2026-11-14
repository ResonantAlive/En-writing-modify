"""Batch common-problems report: report-<task_code>.json (next to result-<task_code>.json).

One on-demand model call per generation. What is sent to the provider (nothing else):
  * topic, rubric ("judging rules"), genre, full score of the batch;
  * per essay that has a usable result: its number in the essay list, the final score and its source, and the
    judgement text: the AI's parsed result (criteria + comment), or - when the teacher reviewed it with items -
    the teacher's items instead of the AI text (the teacher's final score always wins);
  * how many essays have no usable result (failed / cancelled, not included).
Never sent: essay photos, local OCR transcriptions, account / task / subtask ids, keys.

Counts are computed here from the essay numbers the model lists (numbers outside the batch are dropped), not taken
from the model's own arithmetic. The previous report is copied to report_backup/ before an overwrite.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading

from ..core import atomic_io, clock
from ..errors import AppError, NotFound
from ..grading import parse
from ..providers import client as llm
from ..providers import store as pstore
from ..providers import third_party
from . import paths, results, review, store

log = logging.getLogger("wm.batches")

PROMPT_VERSION = 1
SYSTEM = """你是英语作文阅卷老师的教学助手。老师给你一个批次的批改汇总：作文题目、评分标准（判分细则）、满分，以及每篇作文的编号、最终分和评价（AI 的判据与评语；老师复核过并写了意见的作文，给出的是老师的意见）。你看不到学生的作文原文。
请据此找出这个批次里学生集中出现的问题，供老师备课讲评。要求：
1. 只根据给出的判据、评语、意见和分数归纳，不要编造；评价里没有提到的内容不要推测。
2. 每个问题必须列出涉及的作文编号，编号只能使用下面出现的“编号”。至少两篇作文出现的问题才算共性问题；如果本批次只有一两篇作文，可以列出各自的主要问题。
3. 共性问题按涉及的作文数从多到少排列，最多 8 条。每条写清：问题是什么、典型表现、给老师的讲评或教学建议。
4. 只输出一个 JSON 对象，不要代码围栏，不要任何解释，格式如下：
{"概述": "一两句话概括整体情况", "共性问题": [{"问题": "...", "典型表现": "...", "涉及作文": [1, 3], "教学建议": "..."}], "整体优点": ["..."]}"""

_inflight: set[str] = set()
_lock = threading.Lock()


def report_path(cfg, task_code: str):
    return paths.batch_dir(cfg, task_code) / f"report-{task_code}.json"


def read(cfg, task_code: str) -> dict | None:
    doc = atomic_io.read_json(report_path(cfg, task_code), None)
    return doc if isinstance(doc, dict) else None


def _fit(value, limit: int):
    s = json.dumps(value, ensure_ascii=False)
    return value if len(s) <= limit else {"（内容过长，已截断）": s[:limit]}


def collect(cfg, task_code: str) -> tuple[list[dict], list[str], dict]:
    """-> (essays sent, their subtask ids in the same order, stats). Raises report_no_data / report_too_large."""
    lim = int(cfg["report"]["max_essay_chars"])
    items, sids = [], []
    excluded = {"failed": 0, "cancelled": 0}
    metas = store.read_metas(cfg, task_code)
    for no, m in enumerate(metas, 1):                     # numbering = row number in the essay list
        sid = m["subtask_id"]
        rv = review.latest(review.read(cfg, sid))
        _att, ai_score = review.ai_snapshot(cfg, sid, m)
        if rv is None and ai_score is None:
            excluded["cancelled" if m.get("status") == "cancelled" else "failed"] += 1
            continue
        it: dict = {"编号": no, "最终分": rv["final_score"] if rv else ai_score, "分数来源": "老师" if rv else "AI"}
        t_items = rv.get("items") if rv else None
        if ai_score is not None:
            it["AI判分"] = ai_score
        if t_items:
            it["老师意见"] = _fit([{"名称": x.get("name", ""), "内容": x.get("content", "")} for x in t_items], lim)
        elif ai_score is not None:
            g = atomic_io.read_json(paths.subtask_dir(cfg, sid) / "grading.json", None) or {}
            res = g.get("result")
            it["AI评价"] = _fit(res if isinstance(res, dict) else {"仅提取到分数": g.get("score")}, lim)
        items.append(it)
        sids.append(sid)
    if not items:
        raise AppError("report_no_data", "no essay of this batch has a usable result yet", 409, {"excluded": excluded})
    stats = {"essays_total": len(metas), "essays_used": len(items),
             "reviewed": sum(1 for it in items if it["分数来源"] == "老师"), "excluded": excluded}
    return items, sids, stats


def build_messages(b: dict, items: list[dict], stats: dict) -> list[dict]:
    missing = stats["excluded"]["failed"] + stats["excluded"]["cancelled"]
    user = (f"作文题目：\n{b.get('topic') or ''}\n\n评分标准（判分细则）：\n{b.get('rubric') or ''}\n\n"
            f"体裁：{b.get('genre') or ''}；满分：{b.get('full_score')}\n\n"
            f"本批次共 {stats['essays_total']} 篇作文，其中 {stats['essays_used']} 篇有批改结果（见下）"
            + (f"，另有 {missing} 篇没有可用结果，未包含。" if missing else "。")
            + "\n\n各篇批改结果（JSON）：\n" + json.dumps(items, ensure_ascii=False, indent=1))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _text(v, cap: int) -> str:
    return v.strip()[:cap] if isinstance(v, str) else ""


def _numbers(v, valid: set[int]) -> tuple[list[int], bool]:
    """Essay numbers the model listed -> (valid sorted unique numbers, whether anything was dropped)."""
    if not isinstance(v, list):
        return [], bool(v)
    out, dropped = set(), False
    for x in v:
        n = None
        if isinstance(x, bool):
            pass
        elif isinstance(x, int):
            n = x
        elif isinstance(x, float) and x == int(x):
            n = int(x)
        elif isinstance(x, str) and x.strip().isdigit():
            n = int(x.strip())
        if n in valid:
            out.add(n)
        else:
            dropped = True
    return sorted(out), dropped


def normalize(obj: dict, valid: set[int], max_problems: int) -> dict | None:
    raw = obj.get("共性问题")
    if not isinstance(raw, list):
        return None
    warnings, problems = [], []
    for p in raw:
        if not isinstance(p, dict):
            warnings.append("problem_not_object")
            continue
        nos, dropped = _numbers(p.get("涉及作文"), valid)
        if dropped:
            warnings.append("unknown_essay_numbers")
        name = _text(p.get("问题"), 300)
        if not name or not nos:
            warnings.append("problem_without_essays")
            continue
        problems.append({"problem": name, "pattern": _text(p.get("典型表现"), 2000), "essays": nos,
                         "count": len(nos), "share": round(len(nos) / len(valid), 4),
                         "advice": _text(p.get("教学建议"), 2000)})
    problems.sort(key=lambda x: (-x["count"], x["problem"]))
    strengths = [_text(x, 500) for x in obj.get("整体优点", []) if isinstance(x, str) and x.strip()] \
        if isinstance(obj.get("整体优点"), list) else []
    return {"summary": _text(obj.get("概述"), 4000), "problems": problems[:max_problems], "strengths": strengths[:8],
            "warnings": sorted(set(warnings))}


def _provider(cfg, p, b: dict, body: dict) -> tuple[dict, str]:
    for k in ("provider_id", "model"):
        if body.get(k) is not None and not isinstance(body[k], str):
            raise AppError("validation_error", f"{k} must be a string", 422, {"field": k})
    revs = b.get("config_revisions") or []
    cur = revs[-1] if revs else {}
    pid = body.get("provider_id") or cur.get("provider_id") or ""
    if not pid or not pstore.can_use(cfg, p, pid):
        raise NotFound("provider_not_found")
    prov = pstore.get_provider(cfg, pid)
    if prov is None:
        raise NotFound("provider_not_found")
    model = (body.get("model") or "").strip() or (cur.get("model") if pid == cur.get("provider_id") else None) \
        or prov["default_model"]
    return prov, model


def generate(cfg, p, task_code: str, body: dict) -> dict:
    """Build the report with one provider call. body: {provider_id?, model?, third_party_ack?}."""
    b = store.require_batch(cfg, task_code)
    if b.get("status") not in results.FINAL:
        raise AppError("report_not_ready", "the report can be generated once the batch is completed or cancelled",
                       409, {"status": b.get("status")})
    prov, model = _provider(cfg, p, b, body or {})
    if third_party.needs_ack(cfg, p.user_id, prov) and (body or {}).get("third_party_ack") is not True:
        raise AppError("third_party_ack_required", "acknowledge that data is sent to a third-party provider", 409,
                       {"host": third_party.host(prov)})
    with _lock:
        if task_code in _inflight:
            raise AppError("report_in_progress", "a report for this batch is already being generated", 409)
        _inflight.add(task_code)
    try:
        items, sids, stats = collect(cfg, task_code)
        messages = build_messages(b, items, stats)
        chars = sum(len(m["content"]) for m in messages)
        cap = int(cfg["report"]["max_total_chars"])
        if chars > cap:
            raise AppError("report_too_large", "too much text for one report", 422,
                           {"essays": stats["essays_used"], "chars": chars, "max_chars": cap})
        if (body or {}).get("third_party_ack") is True:
            third_party.record_ack(cfg, p.user_id, prov)
        llm_cfg = cfg["llm"]
        params = {"temperature": llm_cfg["temperature"], "max_tokens": llm_cfg["max_tokens"]}
        raw = asyncio.run(llm.post_chat(prov, model, messages, params, float(llm_cfg["timeout_s"])))
        if raw.error_kind or raw.http_status is None or not 200 <= raw.http_status < 300:
            raise AppError("report_failed", "the provider call failed", 502,
                           {"http_status": raw.http_status, "error_kind": raw.error_kind})
        content, _r, _f = llm.extract_message(raw.body)
        if not content.strip():
            raise AppError("report_failed", "the provider returned an empty answer", 502,
                           {"http_status": raw.http_status, "error_kind": None})
        _s, obj = parse.extract_object(content)
        norm = normalize(obj, {it["编号"] for it in items}, int(cfg["report"]["max_problems"])) if obj else None
        if norm is None:
            raise AppError("report_parse_failed", "the model's answer is not a valid report; try again", 502,
                           {"reason": "no_json_object" if obj is None else "missing_problem_list"})
        doc = {"schema_version": 1, "task_code": task_code, "generated_at": clock.iso(), "generated_by": p.user_id,
               "prompt_version": PROMPT_VERSION,
               "model": {"provider_id": prov["id"], "provider_alias": prov.get("alias"), "name": model},
               "inputs": {**stats, "chars": chars, "includes_teacher_review": True},
               **norm, "essay_index": [{"no": it["编号"], "subtask_id": sid} for it, sid in zip(items, sids)]}
        with store.locked(cfg, task_code):
            path = report_path(cfg, task_code)
            if path.is_file():
                results.backup_file(cfg, task_code, path, "report_backup", "report")
            atomic_io.write_json_atomic(path, doc)
        return doc
    finally:
        with _lock:
            _inflight.discard(task_code)
