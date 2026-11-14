// Teacher review panel for one essay and the batch summary panel (spec 4.6 / 4.7).
// Every save appends a revision to review.json; earlier revisions and the AI result files are never changed.
// All dynamic text goes through textContent (h()); the download link is a locally created blob: URL.
import { api } from "./api.js";
import { h, mount, fmtTime } from "./dom.js";
import { errorBox, spinner, toast, confirmModal } from "./ui.js";
import { cpLen } from "./render_model.mjs";
import { DEFAULT_LIMITS, buildReviewBody, reviewErrorText, revisionLine, fmtScore, summaryRows, RESULT_TRIGGER } from "./review_model.mjs";

const friendly = (e) => { const t = reviewErrorText(e); return t ? { ...e, message: t } : e; };

function counter(input, max) {
  const el = h("span", { class: "muted small counter" });
  const sync = () => { const n = cpLen(input.value); el.textContent = `${n} / ${max}`; el.classList.toggle("over", n > max); };
  input.addEventListener("input", sync); sync();
  return el;
}

/**
 * Build once per essay and re-attach on repaint (the subtask table is redrawn on progress events; typed text must
 * survive that). onSaved(revision) lets the page refresh the list and the summary.
 */
export function reviewPanel({ code, sid, fullScore, onSaved }) {
  const root = h("div", { class: "review-panel" });
  const histHost = h("div"), formHost = h("div"), msg = h("div");
  let info = null, limits = DEFAULT_LIMITS, rows = [], busy = false;
  const scoreIn = h("input", { type: "number", min: "0", max: String(fullScore), step: "any", inputmode: "decimal", class: "score-input", aria: { label: "教师分数" } });
  const itemsHost = h("div", { class: "review-items" });
  const addBtn = h("button", { type: "button", class: "btn btn-small", text: "＋ 添加意见条目", onclick: () => { addRow({}); syncAdd(); } });
  const adoptBtn = h("button", { type: "button", class: "btn btn-primary", text: "一键采纳 AI 结果", onclick: () => submit(true) });
  const saveBtn = h("button", { type: "button", class: "btn btn-primary", text: "保存教师评分", onclick: () => submit(false) });

  function syncAdd() { addBtn.disabled = rows.length >= limits.max_items; addBtn.title = addBtn.disabled ? `最多 ${limits.max_items} 条` : ""; }
  function addRow(it) {
    const name = h("input", { type: "text", value: it.name || "", placeholder: "名称（可选，如“内容”“语言”）", aria: { label: "条目名称" } });
    const content = h("textarea", { rows: "3", placeholder: "意见内容", aria: { label: "条目内容" } });
    content.value = it.content || "";
    const row = { name, content, el: null };
    row.el = h("div", { class: "review-item" },
      h("div", { class: "row" }, name, counter(name, limits.max_name_chars),
        h("button", { type: "button", class: "btn btn-small btn-danger", text: "删除", aria: { label: "删除此条目" }, onclick: () => { rows = rows.filter((r) => r !== row); row.el.remove(); syncAdd(); } })),
      content, counter(content, limits.max_item_chars));
    rows.push(row); itemsHost.appendChild(row.el);
  }
  function fillFrom(rev) {
    rows = []; itemsHost.replaceChildren();
    scoreIn.value = rev && rev.final_score != null ? String(rev.final_score) : "";
    for (const it of (rev && rev.items) || []) addRow(it);
    syncAdd();
  }

  async function submit(adopt) {
    if (busy) return;
    mount(msg);
    let body;
    if (adopt) {
      const nonEmpty = rows.some((r) => r.name.value.trim() || r.content.value.trim());
      if (nonEmpty && !(await confirmModal("一键采纳", "一键采纳只记录 AI 分数，不保存下方填写的意见条目。确定采纳吗？", "采纳"))) return;
      body = { adopt_ai: true };
    } else {
      const v = buildReviewBody(scoreIn.value, rows.map((r) => ({ name: r.name.value, content: r.content.value })), fullScore, limits);
      if (v.errors.length) { mount(msg, h("div", { class: "alert alert-error", role: "alert" }, v.errors.slice(0, 8).map((t) => h("div", { text: t })))); return; }
      body = v.body;
    }
    busy = true; adoptBtn.disabled = saveBtn.disabled = true;
    try {
      const r = await api.submitReview(code, sid, body);
      info = { ...info, review: r.review };
      toast(`已保存复核（第 ${r.revision.rev} 版）`, "ok");
      paintHistory();
      if (adopt) fillFrom(r.revision);
      if (onSaved) onSaved(r.revision);
    } catch (e) { mount(msg, errorBox(friendly(e))); }
    finally { busy = false; adoptBtn.disabled = !(info && info.ai_score != null); saveBtn.disabled = false; }
  }

  function paintHistory() {
    const revs = (info && info.review && info.review.revisions) || [];
    if (!revs.length) { mount(histHost, h("p", { class: "muted small", text: "尚未复核。" })); return; }
    const last = revs[revs.length - 1];
    mount(histHost,
      h("div", { class: "review-current" },
        h("div", null, h("strong", { text: `当前最终分：${fmtScore(last.final_score)} / ${fmtScore(fullScore)}` }),
          h("span", { class: "muted small", text: ` · ${fmtTime(last.at)}` })),
        h("div", { class: "muted small", text: revisionLine(last) }),
        last.ai_attempt !== info.ai_attempt ? h("div", { class: "alert alert-warn small", role: "status",
          text: `这次复核针对的是${last.ai_attempt != null ? `第 ${last.ai_attempt} 次` : "失败时的"} AI 结果；之后已重新批改${info.ai_attempt != null ? `（当前为第 ${info.ai_attempt} 次）` : ""}，可以再次复核。` }) : null,
        Array.isArray(last.items) && last.items.length ? h("ul", { class: "review-item-list" }, last.items.map((it) =>
          h("li", null, it.name ? h("strong", { text: it.name + "：" }) : null, h("span", { class: "pre-wrap", text: it.content })))) : null),
      revs.length > 1 ? h("details", null, h("summary", { text: `复核历史（${revs.length} 版，旧版本保留不变）` }),
        h("ol", { class: "review-history", reversed: true }, revs.slice().reverse().map((rv) => h("li", null,
          h("div", { text: revisionLine(rv) }),
          h("div", { class: "muted small", text: `${fmtTime(rv.at)} · 复核人 ${rv.teacher_id}${rv.config_rev != null ? ` · 配置第 ${rv.config_rev} 版` : ""}` }),
          Array.isArray(rv.items) && rv.items.length ? h("details", null, h("summary", { text: "意见条目" }),
            h("ul", null, rv.items.map((it) => h("li", null, it.name ? h("strong", { text: it.name + "：" }) : null, h("span", { class: "pre-wrap", text: it.content }))))) : null)))) : null);
  }

  function paintForm() {
    const ai = info.ai_score;
    if (!info.can_review) { mount(formHost, h("p", { class: "muted small", text: "这篇作文当前不能复核（尚未批改完成或已取消）。" })); return; }
    adoptBtn.disabled = ai == null;
    mount(formHost,
      h("div", { class: "muted small", text: ai != null ? `AI 判分：${fmtScore(ai)} / ${fmtScore(fullScore)}（第 ${info.ai_attempt} 次调用）` : "AI 没有给出分数，可直接人工评分（差值记为空）。" }),
      h("div", { class: "row" }, h("label", { class: "inline-label", text: "教师分数" }), scoreIn, h("span", { class: "muted small", text: `0 – ${fmtScore(fullScore)}` })),
      h("div", { class: "muted small", text: `意见条目可选：最多 ${limits.max_items} 条，每条内容最多 ${limits.max_item_chars} 字、名称最多 ${limits.max_name_chars} 字（按字符计，emoji 算 1 个）。` }),
      itemsHost, addBtn, msg,
      h("div", { class: "row" }, ai != null ? adoptBtn : null, saveBtn));
  }

  async function load() {
    mount(root, spinner());
    try {
      info = await api.getReview(code, sid);
      limits = { ...DEFAULT_LIMITS, ...(info.limits || {}) };
      const revs = (info.review && info.review.revisions) || [];
      fillFrom(revs[revs.length - 1] || null);
      paintHistory(); paintForm();
      mount(root, h("h3", { text: "教师复核" }), histHost, formHost);
    } catch (e) { mount(root, errorBox(e)); }
  }
  load();
  return root;
}

/** Summary panel of a finished (completed / cancelled) batch. */
export function resultPanel(code) {
  const root = h("div", { class: "card" });
  const msg = h("div");
  let doc = null;
  async function load() {
    try { doc = await api.getResult(code); paint(); }
    catch (e) { if (e.code === "result_not_ready") mount(root); else mount(root, h("h2", { text: "批次汇总" }), errorBox(e)); }
  }
  function download() {
    if (!doc) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url; a.download = `result-${String(doc.task_code).replace(/[^A-Za-z0-9_-]/g, "_")}.json`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function regen(e) {
    e.target.disabled = true; mount(msg);
    try { doc = await api.regenerateResult(code); toast("汇总已重新生成（旧文件已备份）", "ok"); paint(); }
    catch (err) { mount(msg, errorBox(err)); e.target.disabled = false; }
  }
  function paint() {
    mount(root, h("h2", { text: "批次汇总" }),
      h("p", { class: "muted small", text: `result-${doc.task_code}.json · ${RESULT_TRIGGER[doc.trigger] || doc.trigger}生成于 ${fmtTime(doc.generated_at)}。每次复核后自动更新，覆盖前自动备份（保留最近 20 份）。` }),
      h("dl", { class: "kv" }, summaryRows(doc).flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: String(v) })])),
      msg,
      h("div", { class: "row" },
        h("button", { type: "button", class: "btn btn-small", text: "下载汇总 JSON", onclick: download }),
        h("button", { type: "button", class: "btn btn-small", text: "重新生成", onclick: regen })));
  }
  load();
  return { el: root, reload: load };
}
