import { api, batchEventsUrl, subtaskImageUrl } from "./api.js";
import { h, mount, fmtTime } from "./dom.js";
import { reasonLabel } from "./labels.js";
import { errorBox, spinner, statusLabel, confirmModal, openLightbox } from "./ui.js";

const ACTIVE = new Set(["queued", "running"]);
export const PAUSE_REASON = { owner_disabled: "账号被禁用", user: "用户暂停", api_errors: "接口连续出错", circuit_breaker: "失败比例过高" };

function seg(cls, pct) { const e = h("span", { class: cls }); e.style.width = pct + "%"; return e; }

export function progressBar(counts) {
  const total = counts && counts.total ? counts.total : 0;
  const part = (k) => (total ? ((counts[k] || 0) / total) * 100 : 0);
  const done = (counts && counts.done) || 0, bad = ((counts && counts.failed) || 0) + ((counts && counts.api_error) || 0);
  return h("div", { class: "progress-wrap" },
    h("div", { class: "progress", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": String(total), "aria-valuenow": String(done + bad) },
      seg("p-done", part("done")), seg("p-bad", part("failed") + part("api_error")), seg("p-run", part("running"))),
    h("div", { class: "muted small", text: total ? `完成 ${done} / ${total}` + (bad ? `，失败 ${bad}` : "") + ((counts.running || 0) ? `，处理中 ${counts.running}` : "") : "无子任务" }));
}

export function actionButtons(b, onDone, onError) {
  const s = b.status;
  const mk = (label, action, opts = {}) => h("button", { type: "button", class: "btn btn-small " + (opts.danger ? "btn-danger" : ""), text: label,
    onclick: async (e) => {
      if (opts.confirm && !(await confirmModal(label, opts.confirm, label))) return;
      e.target.disabled = true;
      try { onDone(await api.batchAction(b.task_code, action), action); } catch (err) { onError(err); e.target.disabled = false; }
    } });
  const out = [];
  if (s === "running" || s === "queued") out.push(mk("暂停", "pause"));
  if (String(s).startsWith("paused")) out.push(mk("继续", "resume"));
  const c = b.counts || {};
  if ((c.failed || 0) + (c.api_error || 0) > 0 && s !== "cancelled") out.push(mk("重跑失败项", "rerun-failed"));
  if (s !== "cancelled" && s !== "completed") out.push(mk("取消", "cancel", { danger: true, confirm: "取消后排队中的作文将不再处理，且无法恢复。确定取消吗？" }));
  return out;
}

export function renderBatchList(root, ctx) {
  const errHost = h("div"), listHost = h("div");
  const sources = new Map(); // task_code -> EventSource
  let pollTimer = null, alive = true, all = false;
  const rows = new Map();   // task_code -> {batch, refs}

  function closeAll() { for (const es of sources.values()) es.close(); sources.clear(); clearInterval(pollTimer); }

  function watch(b) {
    if (!ACTIVE.has(b.status) || sources.has(b.task_code) || sources.size >= 12 || typeof EventSource === "undefined") return;
    const es = new EventSource(batchEventsUrl(b.task_code));
    sources.set(b.task_code, es);
    es.addEventListener("progress", (ev) => {
      let d; try { d = JSON.parse(ev.data); } catch { return; }
      const r = rows.get(b.task_code); if (!r) return;
      Object.assign(r.batch, { status: d.status, pause_reason: d.pause_reason, counts: d.counts });
      paintRow(r);
      if (!ACTIVE.has(d.status)) { es.close(); sources.delete(b.task_code); }
    });
    es.onerror = () => { /* EventSource retries on its own; the poll below is the safety net */ };
  }

  function paintRow(r) {
    const b = r.batch;
    r.status.textContent = statusLabel(b.status) + (b.pause_reason ? `（${PAUSE_REASON[b.pause_reason] || reasonLabel(b.pause_reason)}）` : "");
    r.status.className = "badge st-" + b.status;
    mount(r.progress, progressBar(b.counts));
    mount(r.actions, actionButtons(b, () => load(), (e) => mount(errHost, errorBox(e))));
  }

  async function load() {
    mount(errHost);
    try {
      const list = await api.listBatches(all);
      if (!alive) return;
      rows.clear();
      if (!list.length) { mount(listHost, h("div", { class: "card empty" }, h("p", { text: "还没有批次。" }), h("button", { type: "button", class: "btn btn-primary", text: "去上传", onclick: () => ctx.go("#/upload") }))); return; }
      const items = list.map((b) => {
        const r = { batch: b, status: h("span"), progress: h("div"), actions: h("div", { class: "row" }) };
        rows.set(b.task_code, r);
        paintRow(r);
        return h("article", { class: "card batch-card" },
          h("div", { class: "batch-head" },
            h("a", { class: "batch-code", tabindex: "0", role: "link", text: b.task_code, onclick: () => ctx.go("#/batches/" + encodeURIComponent(b.task_code)),
              onkeydown: (e) => { if (e.key === "Enter") ctx.go("#/batches/" + encodeURIComponent(b.task_code)); } }), r.status),
          h("div", { class: "muted small", text: [b.topic, b.genre, b.route === "vlm" ? "视觉模型" : b.route === "local_ocr" ? "本地识别" : b.route, fmtTime(b.created_at), all && b.owner_id ? "用户 " + b.owner_id : ""].filter(Boolean).join(" · ") }),
          r.progress, r.actions);
      });
      mount(listHost, items);
      for (const b of list) watch(b);
    } catch (e) { mount(listHost); mount(errHost, errorBox(e)); }
  }

  const allBox = h("input", { type: "checkbox", onchange: () => { all = allBox.checked; closeAll(); load(); startPoll(); } });
  function startPoll() {
    clearInterval(pollTimer);
    pollTimer = setInterval(async () => { // safety net for dropped SSE: refresh active rows' status
      for (const [code, r] of rows) if (ACTIVE.has(r.batch.status) && !sources.has(code)) { load(); return; }
    }, 8000);
  }
  mount(root, h("section", { class: "page" }, h("h1", { text: "批次" }),
    ctx.isRoot ? h("label", { class: "check" }, allBox, "显示所有用户的批次") : null, errHost, listHost));
  mount(listHost, spinner());
  load(); startPoll();
  return () => { alive = false; closeAll(); };
}
