// Single lookup table: raw reason / error / warning codes -> short Chinese labels. Unknown codes fall back to the raw code.
// Pure module (no DOM) so it can be unit-tested with node.

export const REASON_LABEL = {
  // grading / parsing
  json_parse_failed: "模型输出不是有效的 JSON",
  missing_required_field: "模型输出缺少必需字段",
  missing_values: "模型输出缺少必需字段",
  score_out_of_range: "分数超出范围",
  empty_response: "模型返回为空",
  refusal_or_safety: "模型拒绝回答（安全策略）",
  regex_fallback: "仅用正则提取到分数，需教师仔细复核",
  // api
  api_error: "接口调用出错（超时 / 网络 / 限流 / 鉴权 / 服务端错误）",
  request_rejected: "请求被服务商拒绝（参数或格式不被接受）",
  provider_unavailable: "服务商不可用（已删除或无权使用）",
  timeout: "请求超时", connect: "无法连接服务商", network: "网络错误",
  // local pipeline
  ocr_failed: "本地识别失败",
  ocr_empty: "本地识别结果为空",
  preprocess_failed: "照片预处理失败",
  interrupted: "被中断（服务重启）",
  cancelled: "已取消",
  // misc
  third_party_ack_required: "需要确认第三方服务",
  zero_shot_consent_required: "需要确认无范例评分",
  local_env_unavailable: "本地识别环境不可用",
  template_mode_mismatch: "模板类型与处理路线不匹配",
  fewshot_placeholder_missing: "提示词缺少 {{FEWSHOT}} 占位符",
  prompt_too_long: "提示词过长",
  builtin_readonly: "内置模板为只读",
  provider_not_found: "服务商不存在",
  circuit_breaker: "失败比例过高",
  api_errors: "接口连续出错",
};

export function reasonLabel(code) {
  if (code == null || code === "") return "";
  const k = String(code);
  return Object.prototype.hasOwnProperty.call(REASON_LABEL, k) ? REASON_LABEL[k] : k;
}

/** Plain-Chinese pause explanation, derived from status / pause_reason. */
export function pauseExplanation(status, reason) {
  if (status === "paused_api_errors") return "接口连续出错，已自动暂停。请检查服务商的网址、密钥、额度或网络，或更换服务商 / 模型后继续。";
  if (status === "paused_circuit_breaker") {
    const why = reason ? `（主要原因：${reasonLabel(reason)}）` : "";
    return `失败比例过高，已自动暂停${why}。system prompt 构建、模型选择或本地识别可能有问题，请查看失败样例后调整再继续。`;
  }
  if (status === "paused_user") return "已由用户暂停。";
  return "";
}

export const TP_HOST_FALLBACK = "（未知主机）";
