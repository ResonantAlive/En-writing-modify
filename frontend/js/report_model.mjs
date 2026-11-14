// Pure helpers for the batch common-problems report (no DOM; tested with node).

export const REPORT_ERROR = Object.freeze({
  report_not_ready: "批次完成或取消后才能生成报告", report_in_progress: "这个批次的报告正在生成，请稍等",
  report_no_data: "这个批次还没有可用的批改结果", report_parse_failed: "模型返回的内容不是有效的报告，请重新生成",
  report_failed: "调用服务商失败，请检查服务商和模型后重试", provider_not_found: "找不到所选服务商",
});

/** Chinese message for a server error of the report endpoints, or null for codes handled elsewhere. */
export function reportErrorText(err) {
  const code = err && err.code;
  if (code === "report_too_large") {
    const d = (err && err.detail) || {};
    return `内容太多，一次发不完：${d.essays ?? "?"} 篇共 ${d.chars ?? "?"} 字，上限 ${d.max_chars ?? "?"} 字`;
  }
  if (code === "report_failed" && err.detail && err.detail.http_status) return `${REPORT_ERROR.report_failed}（HTTP ${err.detail.http_status}）`;
  return Object.prototype.hasOwnProperty.call(REPORT_ERROR, code) ? REPORT_ERROR[code] : null;
}

export const REPORT_WARNING = Object.freeze({
  unknown_essay_numbers: "模型列出了本批次里不存在的作文编号，已忽略",
  problem_without_essays: "有的问题没有对应的作文编号，已丢弃",
  problem_not_object: "模型输出里有格式不对的条目，已忽略",
});

export const reportWarningText = (code) => (Object.prototype.hasOwnProperty.call(REPORT_WARNING, code) ? REPORT_WARNING[code] : String(code));

/** [1, 3, 5] -> "作文 1、3、5" (numbers are the row numbers of the essay list). */
export const essayNos = (nos) => (Array.isArray(nos) && nos.length ? "作文 " + nos.join("、") : "");

/** 0.3333 -> "33.3%" */
export const pct = (share) => (share != null && share !== "" && Number.isFinite(Number(share)) ? `${Math.round(Number(share) * 1000) / 10}%` : "—");

/** One line about how the report was made. */
export function reportMeta(doc) {
  const i = (doc && doc.inputs) || {}, m = (doc && doc.model) || {};
  const ex = i.excluded || {};
  const left = [ex.failed ? `失败 ${ex.failed}` : "", ex.cancelled ? `已取消 ${ex.cancelled}` : ""].filter(Boolean).join("、");
  return `${m.provider_alias || ""} / ${m.name || ""} 生成 · 纳入 ${i.essays_used ?? 0} / ${i.essays_total ?? 0} 篇（其中 ${i.reviewed ?? 0} 篇按老师复核）`
    + (left ? ` · 未纳入：${left}` : "");
}
