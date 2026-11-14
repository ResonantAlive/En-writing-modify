// Pure helpers for the teacher review form and the batch summary (no DOM; tested with node).
// Lengths are counted in Unicode code points ([...s].length), exactly like the server (emoji = 1, not 2).
import { cpLen } from "./render_model.mjs";

export const DEFAULT_LIMITS = Object.freeze({ max_items: 200, max_item_chars: 10000, max_name_chars: 100, max_body_mb: 8 });

/** Server error codes of POST .../review -> Chinese message. */
export const REVIEW_ERROR = Object.freeze({
  too_many_items: "条目过多", item_too_long: "条目内容过长", item_name_too_long: "条目名称过长", item_empty: "条目内容不能为空",
  invalid_item: "条目格式不正确", invalid_items: "条目格式不正确", final_score_required: "请填写教师分数",
  score_out_of_range: "分数超出范围", adopt_with_items: "一键采纳时不能附带条目", no_ai_result: "这篇作文没有可采纳的 AI 结果",
  review_not_allowed: "这篇作文当前不能复核（尚未批改完成或已取消）", body_too_large: "提交内容过大",
});

export function reviewErrorText(err) {
  const code = err && err.code;
  const base = Object.prototype.hasOwnProperty.call(REVIEW_ERROR, code) ? REVIEW_ERROR[code] : null;
  if (!base) return null;
  const d = (err && err.detail) || {};
  const at = Number.isInteger(d.index) ? `（第 ${d.index + 1} 条）` : "";
  const lim = d.max != null ? `：最多 ${d.max}，当前 ${d.got}` : "";
  return base + at + lim;
}

/**
 * Validate the form. score: raw input string; items: [{name, content}] raw strings.
 * Rows where both name and content are blank are dropped. Returns {errors: string[], body}.
 */
export function buildReviewBody(score, items, fullScore, limits = DEFAULT_LIMITS) {
  const L = { ...DEFAULT_LIMITS, ...(limits || {}) };
  const errors = [];
  const raw = String(score ?? "").trim();
  const s = raw === "" ? NaN : Number(raw);
  if (raw === "") errors.push("请填写教师分数");
  else if (!Number.isFinite(s)) errors.push("分数必须是数字");
  else if (s < 0 || (Number.isFinite(Number(fullScore)) && s > Number(fullScore))) errors.push(`分数必须在 0 到 ${fullScore} 之间`);
  const kept = [];
  (Array.isArray(items) ? items : []).forEach((it) => {
    const name = String((it && it.name) ?? "").trim();
    const content = String((it && it.content) ?? "");
    if (!name && !content.trim()) return;
    kept.push({ name, content });
  });
  if (kept.length > L.max_items) errors.push(`条目最多 ${L.max_items} 条（当前 ${kept.length} 条）`);
  kept.forEach((it, i) => {
    if (!it.content.trim()) errors.push(`第 ${i + 1} 条：内容不能为空`);
    const nl = cpLen(it.name), cl = cpLen(it.content);
    if (nl > L.max_name_chars) errors.push(`第 ${i + 1} 条：名称 ${nl} 字，最多 ${L.max_name_chars} 字`);
    if (cl > L.max_item_chars) errors.push(`第 ${i + 1} 条：内容 ${cl} 字，最多 ${L.max_item_chars} 字`);
  });
  const body = { final_score: s };
  if (kept.length) body.items = kept;
  return { errors, body };
}

export const fmtScore = (v) => (v == null || !Number.isFinite(Number(v)) ? "—" : String(Math.round(Number(v) * 1e4) / 1e4));
export const fmtDiff = (v) => (v == null ? "—" : (Number(v) > 0 ? "+" : "") + fmtScore(v));

/** One line describing a revision (newest first lists). */
export function revisionLine(rev) {
  const how = rev.adopted_ai ? "一键采纳 AI" : rev.ai_score == null ? "人工评分（AI 未给出分数）" : "教师改分";
  const items = Array.isArray(rev.items) && rev.items.length ? ` · ${rev.items.length} 条意见` : "";
  return `第 ${rev.rev} 版 · ${how} · 最终 ${fmtScore(rev.final_score)}${rev.ai_score != null ? ` · AI ${fmtScore(rev.ai_score)} · 差值 ${fmtDiff(rev.diff)}` : ""}${items}`;
}

export const RESULT_STATUS = Object.freeze({ done: "已复核", pending_review: "待复核", failed: "失败", cancelled: "已取消" });
export const RESULT_TRIGGER = Object.freeze({ completed: "批次完成时", cancelled: "取消时", review: "复核后", manual: "手动", on_demand: "查看时补" });

/** Summary numbers for the panel, in display order. */
export function summaryRows(doc) {
  const c = (doc && doc.counts) || {}, st = (doc && doc.stats) || {};
  return [
    ["作文总数", c.total ?? 0], ["已复核", c.done ?? 0], ["待复核", c.pending_review ?? 0], ["失败", c.failed ?? 0],
    ["已取消", c.cancelled ?? 0], ["平均最终分", fmtScore(st.mean_final_score)], ["复核与 AI 平均差（绝对值）", fmtScore(st.mean_abs_diff)],
  ];
}
