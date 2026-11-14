// Batch common-problems report panel: one model call per generation (see app/batches/report.py).
// All dynamic text goes through textContent (h()); nothing here builds HTML or hrefs from data.
import { api } from "./api.js";
import { h, mount, fmtTime } from "./dom.js";
import { errorBox, spinner, toast, confirmModal } from "./ui.js";
import { preflightAck, confirmForProvider } from "./thirdparty.js";
import { downloadJson } from "./review.js";
import { reportErrorText, reportWarningText, reportMeta, essayNos, pct } from "./report_model.mjs";

const SENT = "发送给模型的内容：作文题目、评分标准（判分细则）、满分，以及每篇作文的编号、最终分和评价（AI 的判据与评语；老师复核并写了意见的作文改为老师的意见）。不发送作文原文和照片。";
const friendly = (e) => { const t = reportErrorText(e); return t ? { ...e, message: t } : e; };

export function reportPanel(code) {
  const root = h("div", { class: "card" });
  const msg = h("div"), out = h("div");
  let info = null, providers = [], busy = false;
  const provSel = h("select", { "aria-label": "服务商" });
  const model = h("input", { type: "text", "aria-label": "模型" });
  const btn = h("button", { type: "button", class: "btn btn-primary", onclick: () => generate() });

  function syncModel() { const p = providers.find((x) => x.id === provSel.value); if (p) model.value = p.id === info.default_provider_id && info.default_model ? info.default_model : (p.default_model || ""); }
  provSel.addEventListener("change", syncModel);

  async function generate() {
    if (busy) return;
    const prov = providers.find((x) => x.id === provSel.value);
    if (!prov) { mount(msg, errorBox({ message: "请先选择服务商", code: "form_incomplete" })); return; }
    if (info.report && !(await confirmModal("重新生成报告", "将再调用一次模型，旧报告会自动备份。确定吗？", "重新生成"))) return;
    mount(msg);
    const body = { provider_id: prov.id };
    if (model.value.trim()) body.model = model.value.trim();
    const pre = await preflightAck(prov, "题目、评分标准和各篇作文的评价文字（可能包含学生原话）会被发送到该主机。");
    if (!pre.ok) return;
    if (pre.ack) body.third_party_ack = true;
    busy = true; btn.disabled = true; btn.textContent = "生成中…（调用一次模型，可能要几十秒）";
    try {
      const send = async (b) => {
        try { return await api.makeReport(code, b); }
        catch (e) {
          if (e.code === "third_party_ack_required" && !b.third_party_ack && (await confirmForProvider(prov, "题目、评分标准和各篇作文的评价文字会被发送到该主机。"))) return api.makeReport(code, { ...b, third_party_ack: true });
          throw e;
        }
      };
      info = { ...info, report: await send(body) };
      toast("共性问题报告已生成", "ok");
      paint();
    } catch (e) { mount(msg, errorBox(friendly(e))); }
    finally { busy = false; btn.disabled = false; paintBtn(); }
  }

  const paintBtn = () => { btn.textContent = info && info.report ? "重新生成报告" : "生成共性问题报告"; };

  function paintReport() {
    const d = info.report;
    if (!d) { mount(out, h("p", { class: "muted small", text: "还没有生成过报告。" })); return; }
    mount(out,
      h("div", { class: "muted small", text: `${fmtTime(d.generated_at)} · ${reportMeta(d)}` }),
      d.summary ? h("p", { class: "pre-wrap", text: d.summary }) : null,
      d.problems.length ? h("div", { class: "report-problems" }, d.problems.map((p, i) => h("div", { class: "report-problem" },
        h("div", { class: "row" }, h("strong", { text: `${i + 1}. ${p.problem}` }), h("span", { class: "badge badge-info", text: `${p.count} 篇 · ${pct(p.share)}` })),
        p.pattern ? h("div", { class: "pre-wrap", text: p.pattern }) : null,
        h("div", { class: "muted small", text: essayNos(p.essays) }),
        p.advice ? h("div", { class: "pre-wrap" }, h("strong", { text: "讲评建议：" }), p.advice) : null)))
        : h("p", { class: "muted", text: "模型没有归纳出需要讲评的共性问题。" }),
      d.strengths.length ? h("div", null, h("h3", { text: "整体优点" }), h("ul", null, d.strengths.map((s) => h("li", { class: "pre-wrap", text: s })))) : null,
      d.warnings.length ? h("div", { class: "alert alert-warn small", role: "status" }, d.warnings.map((w) => h("div", { text: reportWarningText(w) }))) : null,
      h("p", { class: "muted small", text: "涉及篇数由系统按模型列出的作文编号统计（编号对应下方作文列表的行号），模型可能漏列，请结合作文列表核对。报告基于生成时的复核结果，之后有改动请重新生成。" }),
      h("div", { class: "row" }, h("button", { type: "button", class: "btn btn-small", text: "下载报告 JSON", onclick: () => downloadJson(`report-${String(d.task_code).replace(/[^A-Za-z0-9_-]/g, "_")}.json`, d) })));
  }

  function paint() {
    paintBtn(); paintReport();
    mount(root, h("h2", { text: "共性问题报告" }),
      h("p", { class: "muted small", text: "让模型汇总这个批次里学生集中出现的问题，供讲评使用。每次生成调用一次模型。" }),
      h("p", { class: "muted small", text: SENT }),
      h("div", { class: "row" }, provSel, model, btn), msg, out);
  }

  async function load() {
    mount(root, h("h2", { text: "共性问题报告" }), spinner());
    try {
      [info, providers] = await Promise.all([api.getReport(code), api.listProviders()]);
      if (!info.ready) { mount(root); return; }
      mount(provSel, providers.map((p) => h("option", { value: p.id, text: `${p.alias}${p.is_third_party ? "（第三方）" : ""}` })));
      provSel.value = providers.some((p) => p.id === info.default_provider_id) ? info.default_provider_id : (providers[0] && providers[0].id) || "";
      syncModel(); paint();
    } catch (e) { mount(root, h("h2", { text: "共性问题报告" }), errorBox(e)); }
  }
  load();
  return { el: root, reload: load };
}
