// Shared UI pieces: toast, modal, lightbox, error boxes, status labels.
import { h, mount, safeUrl } from "./dom.js";

export const STATUS_LABEL = {
  queued: "排队中", running: "处理中", completed: "已完成", cancelled: "已取消",
  paused_user: "已暂停", paused_circuit_breaker: "熔断暂停", paused_api_errors: "接口错误暂停",
  done: "完成", failed: "失败", api_error: "接口错误",
};
export const statusLabel = (s) => STATUS_LABEL[s] || String(s ?? "");

const PROBLEM_TEXT = {
  empty_group: "存在空的作文分组", too_many_pages: "某篇作文的照片数超过上限", unassigned_file: "有照片未分配到任何作文",
  duplicate_file: "有照片被重复分配", unknown_file: "包含未知照片", not_divisible: "照片总数不能被 N 整除，已切换为手动整理",
  no_reliable_order: "没有可靠的排序依据，请手动整理", basis_unusable: "所选排序依据不可用", empty_batch: "没有可提交的照片",
};
export const problemText = (p) => (PROBLEM_TEXT[p.type] || p.type) + (p.essay_index != null ? `（作文 ${p.essay_index + 1}）` : "");

/** Plain-text description of an ApiError (message, plus problems for grouping_invalid). */
export function errorLines(err) {
  const lines = [err && err.message ? String(err.message) : "发生未知错误"];
  if (err && Array.isArray(err.detail)) {
    for (const d of err.detail.slice(0, 10)) {
      if (d && typeof d === "object" && d.type) lines.push("· " + problemText(d) + (d.detail ? `：${d.detail}` : ""));
    }
  }
  return lines;
}

export function errorBox(err, extraClass = "") {
  const lines = errorLines(err);
  return h("div", { class: "alert alert-error " + extraClass, role: "alert" },
    lines.map((l, i) => h("div", { class: i ? "alert-sub" : "", text: l })),
    err && err.code ? h("div", { class: "alert-code", text: `错误代码：${err.code}` }) : null);
}

export function infoBox(text, kind = "info") { return h("div", { class: "alert alert-" + kind, role: kind === "error" ? "alert" : "status", text }); }

let toastHost = null;
export function toast(text, kind = "info") {
  if (!toastHost) { toastHost = h("div", { class: "toasts", "aria-live": "polite" }); document.body.appendChild(toastHost); }
  const t = h("div", { class: "toast toast-" + kind, text });
  toastHost.appendChild(t);
  setTimeout(() => t.remove(), kind === "error" ? 6000 : 3000);
}

// ---- focus-trapping modal ----
function trapFocus(root, e) {
  if (e.key !== "Tab") return;
  const f = [...root.querySelectorAll("button, input, select, textarea, [tabindex]:not([tabindex='-1'])")].filter((x) => !x.disabled);
  if (!f.length) return;
  const first = f[0], last = f[f.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
}

/** modal({title, body: Node|string, buttons:[{label, primary, value}]}) -> Promise<value|null> */
export function modal({ title, body, buttons }) {
  return new Promise((resolve) => {
    const prev = document.activeElement;
    const done = (v) => { document.removeEventListener("keydown", onKey, true); overlay.remove(); if (prev && prev.focus) prev.focus(); resolve(v); };
    const onKey = (e) => {
      if (e.key === "Escape") { e.stopPropagation(); done(null); } else trapFocus(dlg, e);
    };
    const btns = (buttons || [{ label: "确定", primary: true, value: true }]).map((b) =>
      h("button", { type: "button", class: "btn " + (b.primary ? "btn-primary" : ""), text: b.label, onclick: () => done(b.value) }));
    const dlg = h("div", { class: "modal", role: "dialog", "aria-modal": "true", "aria-label": title },
      h("h3", { text: title }),
      h("div", { class: "modal-body" }, typeof body === "string" ? h("p", { text: body }) : body),
      h("div", { class: "modal-actions" }, btns));
    const overlay = h("div", { class: "overlay", onclick: (e) => { if (e.target === overlay) done(null); } }, dlg);
    document.body.appendChild(overlay);
    document.addEventListener("keydown", onKey, true);
    (btns[btns.length - 1] || dlg).focus();
  });
}

export const confirmModal = (title, text, okLabel = "确定") =>
  modal({ title, body: text, buttons: [{ label: "取消", value: false }, { label: okLabel, primary: true, value: true }] }).then((v) => v === true);

// ---- lightbox ----
/** openLightbox([{src, caption}], startIndex) ; arrows / Esc ; click backdrop closes */
export function openLightbox(items, start = 0) {
  if (!items.length) return;
  let i = Math.max(0, Math.min(items.length - 1, start));
  const prev = document.activeElement;
  const img = h("img", { class: "lb-img", alt: "" });
  const cap = h("div", { class: "lb-cap" });
  const show = () => {
    const it = items[i];
    const s = safeUrl(it.src);
    if (s) img.setAttribute("src", s); else img.removeAttribute("src");
    img.alt = it.caption || "";
    cap.textContent = `${i + 1} / ${items.length}  ${it.caption || ""}`;
    prevBtn.disabled = i === 0; nextBtn.disabled = i === items.length - 1;
  };
  const go = (d) => { i = Math.max(0, Math.min(items.length - 1, i + d)); show(); };
  const close = () => { document.removeEventListener("keydown", onKey, true); box.remove(); if (prev && prev.focus) prev.focus(); };
  const onKey = (e) => {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); e.stopPropagation(); go(-1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); e.stopPropagation(); go(1); }
    else trapFocus(box, e);
  };
  const prevBtn = h("button", { type: "button", class: "btn lb-nav", text: "‹", aria: { label: "上一张" }, onclick: () => go(-1) });
  const nextBtn = h("button", { type: "button", class: "btn lb-nav", text: "›", aria: { label: "下一张" }, onclick: () => go(1) });
  const closeBtn = h("button", { type: "button", class: "btn lb-close", text: "关闭 ✕", onclick: close });
  const box = h("div", { class: "lightbox", role: "dialog", "aria-modal": "true", "aria-label": "照片预览", onclick: (e) => { if (e.target === box || e.target === img) close(); } },
    h("div", { class: "lb-bar" }, cap, closeBtn),
    h("div", { class: "lb-stage" }, prevBtn, img, nextBtn));
  document.body.appendChild(box);
  document.addEventListener("keydown", onKey, true);
  show();
  closeBtn.focus();
}

export function spinner(text) { return h("div", { class: "loading" }, h("span", { class: "spin", "aria-hidden": "true" }), text || "加载中…"); }

export function setPage(root, ...nodes) { mount(root, ...nodes); }
