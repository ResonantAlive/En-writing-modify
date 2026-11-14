// #/batches/<code> : batch detail with per-subtask results, attempts, pause panel and config changes.
import { api, batchEventsUrl, subtaskImageUrl } from "./api.js";
import { h, mount, fmtTime, field } from "./dom.js";
import { errorBox, spinner, statusLabel, confirmModal, openLightbox, toast } from "./ui.js";
import { progressBar, actionButtons, PAUSE_REASON } from "./batches.js";
import { reasonLabel, pauseExplanation } from "./labels.js";
import { renderGrading, warningsView, attemptView } from "./result_view.js";
import { preflightAck, confirmForProvider } from "./thirdparty.js";
import { prominentWarnings } from "./render_model.mjs";
import { reviewPanel, resultPanel } from "./review.js";
import { fmtScore } from "./review_model.mjs";

const EXCLUDE_REASON = {
  template_text: "与答题纸印刷文字一致", template_region: "位于答题纸印刷区域（条形码 / 页码等）", outside_body: "正文以外的数字 / 符号",
};

/** What the answer-sheet template removed from the local recognition, page by page (plain text only). */
function excludedView(ocr) {
  const pages = Array.isArray(ocr && ocr.pages) ? ocr.pages : [];
  const total = pages.reduce((n, p) => n + (Array.isArray(p.excluded) ? p.excluded.length : 0), 0);
  if (!ocr || !ocr.sheet_template) return null;
  return h("details", null,
    h("summary", { text: `已去除的答题纸印刷内容（${total} 处）` }),
    pages.map((p) => {
      const m = p.sheet_match || {};
      const ex = Array.isArray(p.excluded) ? p.excluded : [];
      return h("div", { class: "excluded-page" },
        h("div", { class: "muted small", text: `第 ${p.index} 页：${m.aligned ? `已与模板对齐（匹配 ${m.anchors} 处）` : m.template_page ? "未能对齐，仅按文字匹配去除" : "没有匹配到模板"}` }),
        ex.length ? h("ul", { class: "excluded-lines" }, ex.map((e) => h("li", null,
          h("span", { class: "excluded-text mono", text: e.text }), h("span", { class: "muted small", text: ` · ${EXCLUDE_REASON[e.reason] || e.reason}` }))))
          : h("div", { class: "muted small", text: "（无）" }));
    }));
}

const REV_REASON = { initial: "创建时", user_change: "手动调整", auto: "自动" };
const isPaused = (s) => s === "paused_circuit_breaker" || s === "paused_api_errors";

export function renderBatchDetail(root, ctx, code) {
  const errHost = h("div"), headHost = h("div"), pauseHost = h("div"), cfgHost = h("div"), revHost = h("div"), subHost = h("div"), resHost = h("div");
  let es = null, alive = true, batch = null, subTimer = null, subs = [], lastStatus = null, result = null;
  const expanded = new Set(), cache = new Map(); // sid -> {status, detail}
  const panels = new Map(); // sid -> {status, el}: review forms are built once so typing survives table repaints

  function panelFor(s) {
    const p = panels.get(s.subtask_id);
    if (p && p.status === s.status) return p.el;
    const el = reviewPanel({ code, sid: s.subtask_id, fullScore: batch ? batch.full_score : null,
      onSaved: () => { loadSubs(); if (result) result.reload(); } });
    panels.set(s.subtask_id, { status: s.status, el });
    return el;
  }
  function paintResult() {
    const fin = batch && (batch.status === "completed" || batch.status === "cancelled");
    if (!fin) { result = null; mount(resHost); return; }
    if (result) result.reload(); else { result = resultPanel(code); mount(resHost, result.el); }
  }

  const showErr = (e) => mount(errHost, errorBox(e));
  const keys = () => ({ scoreKey: batch && batch.score_key, criteriaKey: batch && batch.criteria_key });
  const curRev = () => { const r = (batch && batch.config_revisions) || []; return r[r.length - 1] || null; };

  async function loadBatch() {
    try {
      batch = await api.getBatch(code);
      if (!alive) return;
      mount(errHost);
      paintHead(); paintPause(); paintRevs(); paintResult();
      lastStatus = batch.status;
      if (isPaused(batch.status)) loadFailures();
    } catch (e) { showErr(e); }
  }
  async function loadSubs() {
    try { subs = await api.subtasks(code); if (alive) paintSubs(); } catch (e) { mount(subHost, errorBox(e)); }
  }
  const reloadAll = () => { loadBatch(); loadSubs(); };

  // ---------- head ----------
  function paintHead() {
    const b = batch;
    const rev = curRev();
    mount(headHost, h("div", { class: "card" },
      h("div", { class: "batch-head" }, h("h2", { text: b.task_code }),
        h("span", { class: "badge st-" + b.status, text: statusLabel(b.status) + (b.pause_reason ? `（${PAUSE_REASON[b.pause_reason] || reasonLabel(b.pause_reason)}）` : "") })),
      h("dl", { class: "kv" },
        ...[["题目", b.topic], ["体裁", b.genre], ["满分", b.full_score], ["路线", b.route === "vlm" ? "视觉模型" : b.route === "local_ocr" ? "本地识别" : b.route],
          ["答题纸模板", b.sheet_template ? `${b.sheet_template.name}（${(b.sheet_template.page_ids || []).length} 页空白纸）` : null],
          ["当前配置", rev ? `${rev.provider_alias || ""} / ${rev.model || ""} / 模板 ${rev.template ? rev.template.id + " v" + rev.template.version : ""}（第 ${rev.rev} 版）` : ""],
          ["创建时间", fmtTime(b.created_at)], ["所有者", b.owner_id]].filter(([, v]) => v != null && v !== "").flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: String(v) })])),
      progressBar(b.counts),
      h("div", { class: "row" },
        isPaused(b.status) ? null : actionButtons(b, reloadAll, showErr),
        (isPaused(b.status) || b.status === "completed") ? h("button", { type: "button", class: "btn btn-small", text: "调整服务商 / 模型 / 提示词", onclick: () => openConfig({ promptOpen: false }) }) : null)));
  }

  // ---------- pause panel ----------
  let failures = null;
  async function loadFailures() {
    try { failures = await api.batchFailures(code); if (alive) paintPause(); } catch (e) { showErr(e); }
  }
  function paintPause() {
    const b = batch;
    if (!b || !isPaused(b.status)) { mount(pauseHost); failures = null; return; }
    const reasons = failures && failures.reasons ? Object.entries(failures.reasons).sort((a, c) => c[1] - a[1]) : [];
    const samples = (failures && failures.samples) || [];
    mount(pauseHost, h("div", { class: "card pause-panel" },
      h("h2", { text: b.status === "paused_api_errors" ? "已暂停：接口连续出错" : "已暂停：失败比例过高" }),
      h("div", { class: "alert alert-warn", role: "status", text: pauseExplanation(b.status, (failures && failures.pause_reason) || b.pause_reason) }),
      b.status === "paused_api_errors" && b.api_error_streak ? h("p", { class: "muted small", text: `当前连续接口错误：${b.api_error_streak} 次` }) : null,
      reasons.length ? h("div", null, h("h3", { text: "失败原因统计" }),
        h("ul", null, reasons.map(([r, n]) => h("li", { text: `${reasonLabel(r)}${reasonLabel(r) !== r ? `（${r}）` : ""}：${n} 篇` })))) : null,
      samples.length ? h("div", null, h("h3", { text: "失败样例" }),
        samples.map((s) => h("details", { class: "att-sect" },
          h("summary", { text: `${s.subtask_id || ""} · ${reasonLabel(s.fail_reason)}${s.attempt != null ? ` · 第 ${s.attempt} 次尝试` : ""}` }),
          s.snippet ? h("pre", { class: "msg-part", text: String(s.snippet) }) : h("div", { class: "muted small", text: "（无模型输出片段）" })))) : null,
      h("div", { class: "row" },
        h("button", { type: "button", class: "btn btn-primary", text: "继续", onclick: (e) => doAction(e, "resume") }),
        h("button", { type: "button", class: "btn", text: "更换服务商 / 模型", onclick: () => openConfig({ promptOpen: false, resume: true }) }),
        h("button", { type: "button", class: "btn", text: "编辑提示词后继续", onclick: () => openConfig({ promptOpen: true, resume: true }) }),
        h("button", { type: "button", class: "btn btn-danger", text: "取消批次", onclick: (e) => doAction(e, "cancel", "取消后排队中的作文将不再处理，且无法恢复。确定取消吗？") }))));
  }

  async function doAction(e, action, confirmText) {
    if (confirmText && !(await confirmModal("取消批次", confirmText, "取消批次"))) return;
    e.target.disabled = true;
    try { await api.batchAction(code, action); mount(errHost); mount(cfgHost); reloadAll(); }
    catch (err) { showErr(err); e.target.disabled = false; }
  }

  // ---------- config change ----------
  async function openConfig({ promptOpen, resume }) {
    mount(cfgHost, spinner());
    const rev = curRev();
    let providers = [], templates = [];
    const mode = batch.route === "vlm" ? "image" : "text";
    try { [providers, templates] = await Promise.all([api.listProviders(), api.listPrompts(mode)]); }
    catch (e) { mount(cfgHost); showErr(e); return; }
    if (!alive) return;
    const msg = h("div");
    const provSel = h("select", { "aria-label": "服务商" }, h("option", { value: "", text: `保持当前（${(rev && rev.provider_alias) || "?"}）` }),
      providers.map((p) => h("option", { value: p.id, text: `${p.alias}（默认：${p.default_model || "?"}）${p.is_third_party ? " · 第三方" : ""}` })));
    const model = h("input", { type: "text", value: (rev && rev.model) || "" });
    provSel.addEventListener("change", () => { const p = providers.find((x) => x.id === provSel.value); if (p) model.value = p.default_model || ""; });
    const tplSel = h("select", { "aria-label": "模板" }, templates.map((t) => h("option", { value: t.id, text: `${t.name}（v${t.latest_version}）` })));
    if (rev && rev.template) tplSel.value = rev.template.id;
    const editBox = h("input", { type: "checkbox", checked: !!promptOpen });
    const ta = h("textarea", { rows: "16", class: "mono prompt-text", "aria-label": "提示词正文", spellcheck: "false" });
    const newName = h("input", { type: "text", maxlength: "80", placeholder: "留空则自动命名为“原名称（edited）”" });
    const promptBox = h("div", { hidden: !promptOpen }, field("提示词正文", ta, "保存时若你无权修改原模板（内置或他人的），会自动另存为你的新模板。必须包含 {{FEWSHOT}}。").row,
      field("另存为新模板时的名称（可选）", newName).row);
    let loadedText = null, loadedFor = null;
    async function loadText() {
      const tid = tplSel.value; if (!tid) return;
      const key = tid + "@" + (rev && rev.template && rev.template.id === tid ? rev.template.version : "latest");
      if (key === loadedFor) return;
      ta.disabled = true; ta.value = "加载中…";
      try {
        const t = await api.getPrompt(tid);
        const want = rev && rev.template && rev.template.id === tid ? rev.template.version : Math.max(...t.versions.map((v) => v.version));
        const v = t.versions.find((x) => x.version === want) || t.versions[t.versions.length - 1];
        loadedText = v.text; loadedFor = key; ta.value = v.text; ta.disabled = false;
      } catch (e) { loadedText = null; loadedFor = null; ta.value = ""; mount(msg, errorBox(e)); }
    }
    const syncEdit = () => { promptBox.hidden = !editBox.checked; if (editBox.checked) loadText(); };
    editBox.addEventListener("change", syncEdit);
    tplSel.addEventListener("change", () => { loadedFor = null; if (editBox.checked) loadText(); });
    const contBox = resume ? h("input", { type: "checkbox", checked: true }) : null;

    async function save() {
      mount(msg);
      const body = {};
      if (provSel.value) body.provider_id = provSel.value;
      if (model.value.trim() && (provSel.value || model.value.trim() !== (rev && rev.model))) body.model = model.value.trim();
      if (tplSel.value && !(rev && rev.template && rev.template.id === tplSel.value)) body.template_id = tplSel.value;
      if (editBox.checked && loadedText != null && ta.value !== loadedText) {
        if (!ta.value.trim()) { mount(msg, errorBox({ message: "提示词正文不能为空", code: "form_incomplete" })); return; }
        body.prompt_text = ta.value;
        if (!body.template_id) body.template_id = tplSel.value;
        if (newName.value.trim()) body.new_template_name = newName.value.trim();
      }
      const changed = Object.keys(body).length > 0;
      saveBtn.disabled = true;
      try {
        if (changed) {
          const prov = body.provider_id ? providers.find((p) => p.id === body.provider_id) : null;
          if (prov) {
            const pre = await preflightAck(prov, "作文内容（文字或照片）会被发送到该主机。");
            if (!pre.ok) return;
            if (pre.ack) body.third_party_ack = true;
          }
          const send = async (b) => {
            try { return await api.batchConfig(code, b); }
            catch (e) {
              if (e.code === "fewshot_placeholder_missing") {
                if (await confirmModal("缺少 {{FEWSHOT}}", "提示词里没有 {{FEWSHOT}}，是否在末尾自动追加后保存？", "自动追加")) return api.batchConfig(code, { ...b, auto_append_fewshot: true });
              }
              if (e.code === "third_party_ack_required" && prov && !b.third_party_ack && (await confirmForProvider(prov, "作文内容（文字或照片）会被发送到该主机。"))) return api.batchConfig(code, { ...b, third_party_ack: true });
              throw e;
            }
          };
          await send(body);
          toast("配置已更新（只影响未完成和失败的作文）", "ok");
        } else if (!(contBox && contBox.checked)) { mount(msg, errorBox({ message: "没有任何改动", code: "no_change" })); return; }
        if (contBox && contBox.checked) await api.batchAction(code, "resume");
        mount(cfgHost); reloadAll();
      } catch (e) { mount(msg, errorBox(e)); }
      finally { saveBtn.disabled = false; }
    }
    const saveBtn = h("button", { type: "button", class: "btn btn-primary", text: resume ? "保存并继续" : "保存配置", onclick: save });
    mount(cfgHost, h("div", { class: "card" },
      h("h2", { text: "调整批改配置" }),
      h("p", { class: "muted small", text: "新配置只用于尚未完成和失败的作文；已完成的作文保留原来的配置，可单独“重新批改”。" }),
      field("服务商", provSel).row, field("模型", model).row, field("提示词模板", tplSel).row,
      h("label", { class: "check" }, editBox, "编辑提示词正文"), promptBox,
      contBox ? h("label", { class: "check" }, contBox, "保存后立即继续") : null,
      msg,
      h("div", { class: "row" }, saveBtn, h("button", { type: "button", class: "btn", text: "关闭", onclick: () => mount(cfgHost) }))));
    if (promptOpen) loadText();
  }

  // ---------- revisions ----------
  function paintRevs() {
    const revs = (batch && batch.config_revisions) || [];
    if (revs.length < 1) { mount(revHost); return; }
    mount(revHost, h("div", { class: "card" }, h("details", null, h("summary", { text: `配置修订记录（${revs.length}）` }),
      h("div", { class: "table-wrap" }, h("table", { class: "table" },
        h("thead", null, h("tr", null, ["版本", "时间", "服务商", "模型", "模板", "原因"].map((t) => h("th", { text: t })))),
        h("tbody", null, revs.map((r) => h("tr", null,
          h("td", { text: String(r.rev) }), h("td", { class: "small", text: fmtTime(r.at) }), h("td", { text: r.provider_alias || "" }), h("td", { text: r.model || "" }),
          h("td", { class: "small", text: r.template ? `${r.template.id} v${r.template.version}` : "" }), h("td", { text: REV_REASON[r.reason] || reasonLabel(r.reason) })))))))));
  }

  // ---------- subtasks ----------
  function paintSubs() {
    if (!subs.length) { mount(subHost, h("p", { class: "muted", text: "没有子任务" })); return; }
    const rows = [];
    subs.forEach((s, idx) => {
      const n = Number(s.photos) || 0;
      const items = Array.from({ length: n }, (_, i) => ({ src: subtaskImageUrl(code, s.subtask_id, i + 1, false), caption: `${s.subtask_id} 第 ${i + 1} 张` }));
      const thumbs = items.map((it, i) => h("button", { type: "button", class: "thumb-btn", aria: { label: `查看第 ${i + 1} 张大图` }, onclick: () => openLightbox(items, i) },
        h("img", { src: subtaskImageUrl(code, s.subtask_id, i + 1, true), alt: "", loading: "lazy" })));
      const ws = Array.isArray(s.warnings) ? s.warnings : [];
      const regex = prominentWarnings(ws).regexFallback;
      const open = expanded.has(s.subtask_id);
      const canView = s.status === "done" || s.status === "failed" || s.status === "api_error";
      const canRegrade = canView && batch && batch.status !== "cancelled";
      const reviewed = s.review_status === "reviewed";
      rows.push(h("tr", null,
        h("td", { text: `${idx + 1}（${s.subtask_id}）` }), h("td", null, h("div", { class: "thumb-strip" }, thumbs)),
        h("td", null, h("span", { class: "badge st-" + s.status, text: statusLabel(s.status) }), regex ? h("span", { class: "badge badge-warn", text: "需仔细复核" }) : null,
          reviewed && s.review_outdated ? h("span", { class: "badge badge-warn", text: "复核后已重新批改" }) : null,
          reviewed ? h("span", { class: "badge badge-ok", text: s.adopted_ai ? "已复核（采纳 AI）" : "已复核" }) :canView ? h("span", { class: "badge", text: s.status === "done" ? "待复核" : "可人工评分" }) : null),
        h("td", { class: "score-cell", text: fmtScore(s.ai_score) }),
        h("td", { class: "score-cell" + (reviewed ? " score-final" : ""), text: fmtScore(s.teacher_score) }),
        h("td", { class: "small", text: [s.fail_reason ? reasonLabel(s.fail_reason) : "", ...ws.filter((w) => w !== "regex_fallback").map(reasonLabel)].filter(Boolean).join("；") }),
        h("td", null, h("div", { class: "row" },
          canView ? h("button", { type: "button", class: "btn btn-small", text: open ? "收起" : reviewed ? "查看 / 修改复核" : "查看结果 / 复核", onclick:() => { if (open) expanded.delete(s.subtask_id); else expanded.add(s.subtask_id); paintSubs(); } }) : null,
          canRegrade ? h("button", { type: "button", class: "btn btn-small", text: "重新批改", onclick: (e) => regrade(e, s) }) : null))));
      if (open) {
        const host = h("div", { class: "sub-detail" });
        rows.push(h("tr", { class: "detail-row" }, h("td", { colspan: "7" }, host)));
        showDetail(s, host);
      }
    });
    mount(subHost, h("div", { class: "table-wrap" }, h("table", { class: "table" },
      h("thead", null, h("tr", null, ["作文", "照片", "状态", "AI 分", "教师分", "原因 / 提示", "操作"].map((t) => h("th", { text: t })))),
      h("tbody", null, rows))));
  }

  async function regrade(e, s) {
    if (!(await confirmModal("重新批改", `将重新批改 ${s.subtask_id}，并覆盖当前结果（历史尝试记录保留）。确定吗？`, "重新批改"))) return;
    e.target.disabled = true;
    try {
      const r = await api.regrade(code, s.subtask_id);
      cache.delete(s.subtask_id); panels.delete(s.subtask_id);
      toast(r && JSON.stringify(r).includes("queued_until_resume") ? "已排入队列，批次继续后开始批改" : "已重新排队批改", "ok");
      reloadAll();
    } catch (err) { showErr(err); e.target.disabled = false; }
  }

  async function showDetail(s, host) {
    const c = cache.get(s.subtask_id);
    if (c && c.status === s.status) { paintDetail(host, s, c.detail); return; }
    mount(host, spinner());
    try {
      const d = await api.subtaskDetail(code, s.subtask_id);
      cache.set(s.subtask_id, { status: s.status, detail: d });
      if (alive && host.isConnected) paintDetail(host, s, d);
    } catch (e) { if (host.isConnected) mount(host, errorBox(e)); }
  }

  function paintDetail(host, s, d) {
    const k = keys();
    const g = d.grading, attempts = d.attempts || [];
    const gWarn = g && Array.isArray(g.warnings) ? g.warnings : [];
    const metaWarn = d.meta && Array.isArray(d.meta.warnings) ? d.meta.warnings : [];
    const allWarn = [...new Set([...gWarn, ...metaWarn, ...(g && g.parse_level === "regex_fallback" ? ["regex_fallback"] : [])])];
    const attemptHost = h("div");
    const ocr = d.ocr && typeof d.ocr.transcription === "string" && d.ocr.transcription.trim() && d.ocr.transcription !== "<empty>" ? d.ocr.transcription : null;
    mount(host,
      warningsView(allWarn),
      s.fail_reason && !g ? h("div", { class: "alert alert-error", text: `失败原因：${reasonLabel(s.fail_reason)}${reasonLabel(s.fail_reason) !== s.fail_reason ? `（${s.fail_reason}）` : ""}` }) : null,
      g && g.result != null ? h("div", null,
        h("h3", { text: `批改结果${g.parse_level ? `（解析方式：${g.parse_level}）` : ""}` }),
        renderGrading(g.result, k)) : (g && g.score != null ? h("div", null, h("h3", { text: "仅提取到分数" }), renderGrading({ [k.scoreKey || "score"]: g.score }, k), g.raw_text ? h("details", null, h("summary", { text: "模型原始输出" }), h("pre", { class: "msg-part", text: String(g.raw_text) })) : null) : null),
      ocr ? h("details", null, h("summary", { text: "本地识别文字" }), h("pre", { class: "msg-part", text: ocr })) : null,
      excludedView(d.ocr),
      panelFor(s),
      h("h3", { text: `调用记录（${attempts.length}）` }),
      attempts.length ? h("div", { class: "table-wrap" }, h("table", { class: "table" },
        h("thead", null, h("tr", null, ["#", "结果", "分类", "HTTP", "耗时", "用量", ""].map((t) => h("th", { text: t })))),
        h("tbody", null, attempts.map((a) => h("tr", null,
          h("td", { text: String(a.attempt) }),
          h("td", { text: a.error_code ? reasonLabel(a.error_code) : "成功" }),
          h("td", { text: a.classification || "" }),
          h("td", { text: a.http_status != null ? String(a.http_status) : "" }),
          h("td", { text: a.latency_ms != null ? `${a.latency_ms} ms` : "" }),
          h("td", { class: "small", text: a.usage ? `${a.usage.prompt ?? "?"}+${a.usage.completion ?? "?"}` : "" }),
          h("td", null, h("button", { type: "button", class: "btn btn-small", text: "查看请求 / 响应", onclick: async (e) => {
            e.target.disabled = true; mount(attemptHost, spinner());
            try { const ad = await api.attemptDetail(code, s.subtask_id, a.attempt); mount(attemptHost, attemptView(ad, k)); }
            catch (err) { mount(attemptHost, errorBox(err)); }
            e.target.disabled = false;
          } }))))))) : h("p", { class: "muted small", text: "没有调用记录" }),
      attemptHost);
  }

  mount(root, h("section", { class: "page" },
    h("div", { class: "row" }, h("button", { type: "button", class: "btn btn-small", text: "← 返回批次列表", onclick: () => ctx.go("#/batches") })),
    errHost, headHost, pauseHost, cfgHost, resHost, revHost, h("div", { class: "card" }, h("h2", { text: "作文列表" }), subHost)));
  mount(headHost, spinner()); mount(subHost, spinner());
  loadBatch(); loadSubs();
  if (typeof EventSource !== "undefined") {
    es = new EventSource(batchEventsUrl(code));
    let last = 0;
    es.addEventListener("progress", (ev) => {
      let d; try { d = JSON.parse(ev.data); } catch { return; }
      if (!batch) return;
      Object.assign(batch, { status: d.status, pause_reason: d.pause_reason, counts: d.counts });
      paintHead();
      if (d.status !== lastStatus) { lastStatus = d.status; loadBatch(); } // pause / resume: refetch full detail + failures
      const now = Date.now();
      if (now - last > 1500) { last = now; loadSubs(); } else { clearTimeout(subTimer); subTimer = setTimeout(loadSubs, 1600); }
    });
  }
  return () => { alive = false; if (es) es.close(); clearTimeout(subTimer); };
}
