// DOM rendering of a grading result / attempt, driven by the pure model in render_model.mjs.
// Every string goes through textContent (h() text nodes); no HTML strings anywhere.
import { h } from "./dom.js";
import { buildNode, describeMessages, prominentWarnings } from "./render_model.mjs";
import { reasonLabel } from "./labels.js";

function renderNodeDom(n) {
  switch (n.kind) {
    case "null": return h("span", { class: "rt-null muted", text: "（空）" });
    case "truncated": return h("span", { class: "rt-null muted", text: "（层级过深，已省略）" });
    case "number": return h("span", { class: "rt-num", text: String(n.value) });
    case "boolean": return h("span", { class: "rt-num", text: n.value ? "是" : "否" });
    case "string": return h("div", { class: "rt-str", text: n.value });
    case "array":
      if (!n.items.length) return h("span", { class: "rt-null muted", text: "（无）" });
      return h("ul", { class: "rt-list" }, n.items.map((it) => h("li", null, renderNodeDom(it))),
        n.omitted ? h("li", { class: "muted", text: `…另有 ${n.omitted} 项未显示` }) : null);
    case "object":
      return h("div", { class: "rt-obj" }, n.entries.map((e) => {
        const inline = e.node.kind === "number" || e.node.kind === "boolean" || e.node.kind === "null" || (e.node.kind === "string" && !e.node.value.includes("\n") && [...e.node.value].length <= 60);
        const cls = "rt-entry" + (e.role ? " rt-" + e.role : "") + (inline ? " rt-inline" : "");
        return h("section", { class: cls },
          h("h4", { class: "rt-key" }, e.key, e.role === "score" ? h("span", { class: "badge badge-info", text: "分数" }) : e.role === "criteria" ? h("span", { class: "badge", text: "判据" }) : null),
          h("div", { class: "rt-val" }, renderNodeDom(e.node)));
      }), n.omitted ? h("div", { class: "muted", text: `…另有 ${n.omitted} 个字段未显示` }) : null);
    default: return h("span", { text: "" });
  }
}

/** Render any JSON value recursively by key name; scoreKey / criteriaKey (top level) are highlighted. */
export function renderGrading(value, { scoreKey, criteriaKey } = {}) {
  return h("div", { class: "rt-root" }, renderNodeDom(buildNode(value, { scoreKey, criteriaKey })));
}

export function warningsView(warnings) {
  const w = prominentWarnings(warnings);
  const out = [];
  if (w.regexFallback) {
    out.push(h("div", { class: "alert alert-warn alert-strong", role: "alert" },
      h("strong", { text: "需要教师仔细复核：" }),
      h("span", { text: " 模型输出格式不合规，系统仅用正则从原文中提取了分数（没有完整的判据）。请对照作文逐项确认后再采用。" })));
  }
  if (w.others.length) out.push(h("div", { class: "alert alert-warn", role: "status" }, h("ul", null, w.others.map((x) => h("li", { text: reasonLabel(x) + (reasonLabel(x) !== x ? `（${x}）` : "") })))));
  return out;
}

/** Attempt viewer: request (messages, images as refs only) and response. */
export function attemptView(data, { scoreKey, criteriaKey } = {}) {
  const req = (data && data.request) || {}, res = (data && data.response) || {};
  const msgs = describeMessages(req.messages);
  const metaReq = { ...req }; delete metaReq.messages;
  const resView = { ...res }; for (const k of ["raw", "parsed", "content", "thinking"]) delete resView[k]; // shown separately above
  const sect = (title, body, open) => h("details", { class: "att-sect", open: !!open }, h("summary", { text: title }), body);
  return h("div", { class: "attempt-view" },
    sect("请求（图片仅显示引用）", h("div", null,
      h("div", { class: "rt-root" }, renderNodeDom(buildNode(metaReq, {}))),
      msgs.map((m) => h("div", { class: "msg" },
        h("div", { class: "msg-role", text: m.role }),
        m.parts.map((p) => h("pre", { class: "msg-part" + (p.kind === "image_ref" ? " msg-img" : ""), text: p.text }))))), true),
    sect("响应", h("div", null,
      res.error_code ? h("div", { class: "alert alert-error", text: `错误：${reasonLabel(res.error_code)}${res.error_detail ? "（" + res.error_detail + "）" : ""}` }) : null,
      res.parsed ? h("div", null, h("h4", { text: "解析结果" }), renderGrading(res.parsed, { scoreKey, criteriaKey })) : null,
      res.content ? h("div", null, h("h4", { text: "模型原始输出" }), h("pre", { class: "msg-part", text: String(res.content) })) : null,
      res.thinking ? h("details", null, h("summary", { text: "思考过程" }), h("pre", { class: "msg-part", text: String(res.thinking) })) : null,
      h("div", { class: "rt-root" }, renderNodeDom(buildNode(resView, {})))), true),
    res.raw !== undefined ? sect("原始响应体", h("pre", { class: "msg-part", text: typeof res.raw === "string" ? res.raw : JSON.stringify(res.raw, null, 2) }), false) : null);
}
