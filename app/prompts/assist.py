"""Prompt-writing assistant: asks a provider to draft a grading prompt from the user's requirement
(plus optionally a few of the caller's own history records). The result is only a draft."""
from __future__ import annotations

import asyncio
import json

from ..errors import AppError
from ..providers import client as llm

MAX_RECORDS = 5
_FIELDS = ("task_code", "subtask_id", "topic", "rubric", "genre", "full_score", "transcription",
           "ai_result", "review", "template")

SYSTEM = """你是提示词工程师，帮助英语作文阅卷老师为大模型编写“作文评分系统提示词”。
请根据老师的需求写出一份完整的系统提示词草稿，直接输出提示词正文，不要附加解释，不要使用代码围栏。
提示词必须满足：
1. 用中文写指令；学生作文是英文。
2. 只能依据 {{TOPIC}}（题目）、{{RUBRIC}}（评分标准）、{{GENRE}}（文体）、{{FULL_SCORE}}（满分）这四个占位符给出的信息评分，占位符原样保留，不要替换成具体内容。
3. 包含一行占位符 {{FEWSHOT}}，放在“参考示例”之前的引导语之后；该占位符可能被替换为空。
4. 规定严格的 JSON 输出格式，并说明每个字段的含义。
5. 不得修改、补全或猜测学生原文。"""


def _slim(rec: dict) -> dict:
    out = {k: rec.get(k) for k in _FIELDS if k in rec}
    t = out.get("transcription")
    if isinstance(t, str) and len(t) > 3000:
        out["transcription"] = t[:3000]
    rv = out.get("review")
    if isinstance(rv, dict):
        out["review"] = {k: rv.get(k) for k in ("final_score", "items", "adopted_ai")}
    return out


def _history(cfg, owner_id: str | None) -> list[dict]:
    try:
        from ..grading import history          # G's module; may not exist yet
        it = history.iter_records(cfg, owner_id)
    except Exception:
        return []
    reviewed, others = [], []
    try:
        for i, r in enumerate(it):
            if i >= 300:
                break
            if not isinstance(r, dict):
                continue
            (reviewed if r.get("review") else others).append(r)
            if len(reviewed) >= MAX_RECORDS:
                break
    except Exception:
        pass
    return [_slim(r) for r in (reviewed + others)[:MAX_RECORDS]]


def build_messages(requirement: str, records: list[dict], topic: str | None) -> list[dict]:
    parts = [f"老师的需求：\n{requirement}"]
    if topic:
        parts.append(f"作文题目（仅供参考）：\n{topic}")
    if records:
        parts.append("以下是老师账号下的历史批改记录（供你参考评分风格，不必逐字引用）：\n"
                     + json.dumps(records, ensure_ascii=False, indent=1))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def draft(cfg, provider: dict, model: str, requirement: str, owner_id: str | None,
          include_history: bool, topic: str | None) -> str:
    records = _history(cfg, owner_id) if include_history else []
    msgs = build_messages(requirement, records, topic)
    llm_cfg = cfg["llm"]
    params = {"temperature": llm_cfg["temperature"], "max_tokens": llm_cfg["max_tokens"]}
    raw = asyncio.run(llm.post_chat(provider, model, msgs, params, float(llm_cfg["timeout_s"])))
    if raw.error_kind or raw.http_status is None or not 200 <= raw.http_status < 300:
        raise AppError("assist_failed", "the provider call failed", 502,
                       {"http_status": raw.http_status, "error_kind": raw.error_kind})
    content, _, _ = llm.extract_message(raw.body)
    content = content.strip()
    if not content:
        raise AppError("assist_failed", "the provider returned an empty answer", 502,
                       {"http_status": raw.http_status, "error_kind": None})
    return content
