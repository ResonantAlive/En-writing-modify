// Manual grouping editor (DOM layer). All state transitions live in group_model.mjs (pure, unit-tested).
// Drag uses pointer events (mouse / pen: drag the whole thumbnail; touch: drag the grip handle so the page can still scroll).
// Keyboard: select + toolbar buttons, Alt+Arrow to move the focused photo, Space select, Enter enlarge, Ctrl+Z / Ctrl+Y.
import * as M from "./group_model.mjs";
import { h, mount } from "./dom.js";
import { openLightbox, problemText } from "./ui.js";

const DRAG_THRESHOLD = 6;

/**
 * createGroupEditor({ maxPages, onChange(essays), getBaseOrder() -> ids|null })
 * returns { el, setFiles(files[]), setEssays(essays, {resetHistory}), getEssays(), setProblems(problems), destroy() }
 */
export function createGroupEditor({ maxPages = 0, onChange = () => {}, getBaseOrder = () => null }) {
  let files = new Map();
  let hist = M.createHistory([]);
  let selected = new Set();
  let serverProblems = [];
  let focusId = null;
  let drag = null;           // active drag session
  let pendingRender = false;
  let justDragged = false;
  let autoN = 2;
  let max = maxPages;

  const el = h("div", { class: "group-editor" });
  const essays = () => hist.present;

  // ---------- state transitions ----------
  function apply(next, keepFocus) {
    const nh = M.commit(hist, next);
    if (nh === hist) return;
    hist = nh;
    if (keepFocus) focusId = keepFocus;
    changed();
  }
  function changed() { pruneSelection(); render(); onChange(M.clone(essays())); }
  function pruneSelection() {
    const all = new Set(M.flatten(essays()));
    for (const id of [...selected]) if (!all.has(id)) selected.delete(id);
  }
  const sel = () => M.inReadingOrder(essays(), [...selected]);

  function doUndo() { const n = M.undo(hist); if (n !== hist) { hist = n; changed(); } }
  function doRedo() { const n = M.redo(hist); if (n !== hist) { hist = n; changed(); } }

  // ---------- rendering ----------
  function problemsFor(i) {
    const out = [];
    const e = essays()[i];
    if (!e.length) out.push("这一篇是空的");
    else if (max > 0 && e.length > max) out.push(`超过每篇上限 ${max} 张`);
    for (const p of serverProblems) {
      if (p.essay_index === i && !(p.type === "too_many_pages" && max > 0 && e.length > max) && !(p.type === "empty_group" && !e.length)) out.push(problemText({ type: p.type, essay_index: null }));
    }
    return out;
  }

  function flatIndex(id) { return M.flatten(essays()).indexOf(id); }

  function openBig(id) {
    const items = [];
    essays().forEach((e, ei) => e.forEach((fid, k) => {
      const f = files.get(fid);
      items.push({ src: f ? f.thumb_url : "", caption: `作文 ${ei + 1} · 第 ${k + 1} 张 · ${f ? f.orig_name : ""}` });
    }));
    openLightbox(items, flatIndex(id));
  }

  function thumbEl(id, ei, k, len) {
    const f = files.get(id) || { orig_name: "（未知文件）", thumb_url: "" };
    const isSel = selected.has(id);
    const selBtn = h("button", { type: "button", class: "sel", "aria-pressed": String(isSel), aria: { label: `${isSel ? "取消选择" : "选择"}照片 ${f.orig_name}` }, text: isSel ? "✓" : "",
      onclick: (e) => { e.stopPropagation(); toggleSel(id); } });
    const t = h("div", { class: "thumb" + (isSel ? " selected" : ""), tabindex: "0", role: "group", dataset: { fid: id },
      "aria-label": `作文 ${ei + 1} 第 ${k + 1} 张，共 ${len} 张：${f.orig_name}。空格选择，回车放大，Alt加方向键移动` },
    h("img", { src: f.thumb_url, alt: "", draggable: "false" }),
    h("span", { class: "idx", text: String(k + 1) }), selBtn,
    h("span", { class: "grip", title: "按住拖动", "aria-hidden": "true", text: "⠿" }),
    h("span", { class: "fname", text: f.orig_name }));
    t.addEventListener("pointerdown", (e) => onPointerDown(e, id, t));
    t.addEventListener("click", (e) => {
      if (justDragged || e.target.closest("button")) return;
      if (e.ctrlKey || e.metaKey || e.shiftKey) toggleSel(id); else openBig(id);
    });
    t.addEventListener("keydown", (e) => onThumbKey(e, id));
    t.addEventListener("focus", () => { focusId = id; });
    return t;
  }

  function toggleSel(id) {
    if (selected.has(id)) selected.delete(id); else selected.add(id);
    focusId = id; render();
  }

  function render() {
    if (drag && drag.active) { pendingRender = true; return; }
    pendingRender = false;
    const es = essays();
    const nPhotos = M.countPhotos(es);
    const localBad = es.some((_, i) => problemsFor(i).length);
    const selIds = sel();

    const btn = (text, onclick, disabled, extra = "", label) => h("button", { type: "button", class: "btn btn-small " + extra, text, disabled, onclick, aria: label ? { label } : null });
    const nInput = h("input", { type: "number", min: "1", max: "50", value: String(autoN), class: "n-input", "aria-label": "每篇照片数 N" });
    const autoBtn = btn(`按 ${autoN} 张自动分组`, () => {
      const base = getBaseOrder() || M.flatten(es);
      apply(M.autoGroup(base, autoN));
    });
    nInput.addEventListener("input", () => {
      const v = Math.max(1, Math.min(50, Math.floor(Number(nInput.value)) || 1));
      autoN = v; autoBtn.textContent = `按 ${v} 张自动分组`;
    });

    const toolbar = h("div", { class: "ge-toolbar", role: "toolbar", "aria-label": "分组工具" },
      btn("↶ 撤销", doUndo, !M.canUndo(hist), "", "撤销（Ctrl+Z）"), btn("↷ 重做", doRedo, !M.canRedo(hist), "", "重做（Ctrl+Y）"),
      h("span", { class: "sep" }),
      btn("合并为一篇", () => apply(M.mergeSelected(es, selIds)), selIds.length < 2),
      btn("拆成单张", () => apply(M.splitSelectedToSingles(es, selIds)), selIds.length < 1),
      btn("移到新的一篇", () => apply(M.moveToNewEssay(es, selIds, es.length)), selIds.length < 1),
      btn("清除选择", () => { selected.clear(); render(); }, selIds.length < 1),
      h("span", { class: "sep" }),
      h("label", { class: "n-label" }, "每", nInput, "张"), autoBtn,
      btn("全部拆成单张", () => apply(M.singles(getBaseOrder() || M.flatten(es)))));

    const summary = h("div", { class: "ge-summary", "aria-live": "polite" },
      h("strong", { text: `共 ${es.length} 篇作文、${nPhotos} 张照片` }),
      selIds.length ? h("span", { class: "muted", text: `　已选 ${selIds.length} 张` }) : null,
      localBad ? h("span", { class: "bad", text: "　有分组需要调整（见红色标记）" }) : null);

    const list = h("div", { class: "ge-list" });
    es.forEach((e, ei) => {
      if (ei > 0) {
        list.appendChild(h("div", { class: "essay-sep" }, h("button", { type: "button", class: "merge-btn", text: "⤒ 合并上下两篇 ⤓",
          aria: { label: `把作文 ${ei} 与作文 ${ei + 1} 合并` }, onclick: () => apply(M.mergeAdjacent(es, ei - 1)) })));
      }
      const probs = problemsFor(ei);
      const thumbs = h("div", { class: "thumbs" });
      e.forEach((id, k) => {
        if (k > 0) thumbs.appendChild(h("button", { type: "button", class: "gap-btn", text: "✂", title: "在这里拆开为两篇",
          aria: { label: `在第 ${k} 张和第 ${k + 1} 张之间拆分作文 ${ei + 1}` }, onclick: () => apply(M.splitAt(es, ei, k)) }));
        thumbs.appendChild(thumbEl(id, ei, k, e.length));
      });
      const allSel = e.length > 0 && e.every((id) => selected.has(id));
      list.appendChild(h("article", { class: "essay" + (probs.length ? " invalid" : ""), dataset: { ei } },
        h("div", { class: "essay-head" },
          h("h3", { text: `作文 ${ei + 1}（${e.length} 张）` }),
          probs.length ? h("span", { class: "essay-problem", role: "alert", text: probs.join("；") }) : null,
          h("button", { type: "button", class: "btn btn-small", text: allSel ? "取消全选" : "全选本篇",
            onclick: () => { for (const id of e) { if (allSel) selected.delete(id); else selected.add(id); } render(); } })),
        thumbs));
    });
    const newZone = h("div", { class: "newzone", dataset: { newzone: "1" }, text: "把照片拖到这里，新建一篇作文" });
    const ae = document.activeElement;
    const tbIdx = ae && ae.closest && ae.closest(".ge-toolbar") && el.contains(ae) ? [...el.querySelectorAll(".ge-toolbar button")].indexOf(ae) : -1;
    mount(el, toolbar, summary, list, newZone);
    if (tbIdx >= 0 && !restoreFocus) { const nb = toolbar.querySelectorAll("button")[tbIdx]; if (nb) nb.focus({ preventScroll: true }); }
    if (focusId && restoreFocus) {
      const t = el.querySelector(`.thumb[data-fid="${CSS.escape(focusId)}"]`);
      if (t) t.focus({ preventScroll: true });
    }
  }
  let restoreFocus = false; // only steal focus back after keyboard interactions inside the editor

  // ---------- keyboard ----------
  function onThumbKey(e, id) {
    restoreFocus = true;
    const es = essays();
    const ids = [...el.querySelectorAll(".thumb")].map((t) => t.dataset.fid);
    const pos = ids.indexOf(id);
    const loc = M.locate(es, id);
    if (!loc) return;
    if (e.key === " " || e.key === "Spacebar") { e.preventDefault(); toggleSel(id); return; }
    if (e.key === "Enter") { e.preventDefault(); openBig(id); return; }
    if (e.altKey && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
      e.preventDefault();
      const t = M.stepTarget(es, id, e.key === "ArrowLeft" ? "left" : "right");
      if (t) apply(M.movePhotos(es, [id], t.essay, t.index), id);
      return;
    }
    if (e.altKey && e.key === "ArrowUp") {
      e.preventDefault();
      if (loc.essay > 0) apply(M.movePhotos(es, [id], loc.essay - 1, es[loc.essay - 1].length), id);
      else if (es[0].length > 1) apply(M.moveToNewEssay(es, [id], 0), id);
      return;
    }
    if (e.altKey && e.key === "ArrowDown") {
      e.preventDefault();
      if (loc.essay < es.length - 1) apply(M.movePhotos(es, [id], loc.essay + 1, 0), id);
      else if (es[loc.essay].length > 1) apply(M.moveToNewEssay(es, [id], es.length), id);
      return;
    }
    if (!e.altKey && !e.ctrlKey && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
      e.preventDefault();
      const nid = ids[pos + (e.key === "ArrowLeft" ? -1 : 1)];
      if (nid) el.querySelector(`.thumb[data-fid="${CSS.escape(nid)}"]`).focus();
    }
  }
  function onDocKey(e) {
    if (!el.isConnected) return;
    const tg = e.target;
    if (tg && (tg.tagName === "INPUT" || tg.tagName === "TEXTAREA" || tg.tagName === "SELECT" || tg.isContentEditable)) return;
    if (document.querySelector(".lightbox, .overlay")) return;
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && k === "z" && !e.shiftKey) { e.preventDefault(); restoreFocus = false; doUndo(); }
    else if ((e.ctrlKey || e.metaKey) && (k === "y" || (k === "z" && e.shiftKey))) { e.preventDefault(); restoreFocus = false; doRedo(); }
  }
  document.addEventListener("keydown", onDocKey);
  el.addEventListener("pointerdown", () => { restoreFocus = false; }, true);

  // ---------- pointer drag ----------
  function onPointerDown(e, id, thumb) {
    if (e.button !== undefined && e.button !== 0) return;
    if (e.target.closest("button")) return;
    if (e.pointerType === "touch" && !e.target.closest(".grip")) return; // touch: page scroll unless grip is used
    if (drag) return;
    const ids = selected.has(id) ? sel() : [id];
    drag = { id, ids, pointerId: e.pointerId, sx: e.clientX, sy: e.clientY, x: e.clientX, y: e.clientY, active: false, ghost: null, drop: null, raf: 0, thumb };
    try { thumb.setPointerCapture(e.pointerId); } catch { /* not fatal */ }
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerCancel);
    window.addEventListener("keydown", onDragKey, true);
  }
  function onPointerMove(e) {
    if (!drag || e.pointerId !== drag.pointerId) return;
    drag.x = e.clientX; drag.y = e.clientY;
    if (!drag.active) {
      if (Math.hypot(drag.x - drag.sx, drag.y - drag.sy) < DRAG_THRESHOLD) return;
      activateDrag();
    }
    e.preventDefault();
    moveGhost(); updateDrop(); scrollTick();
  }
  function activateDrag() {
    drag.active = true;
    document.body.classList.add("dragging");
    const f = files.get(drag.id);
    drag.ghost = h("div", { class: "drag-ghost" }, h("img", { src: f ? f.thumb_url : "", alt: "" }), drag.ids.length > 1 ? h("span", { class: "count", text: String(drag.ids.length) }) : null);
    document.body.appendChild(drag.ghost);
    for (const i of drag.ids) { const t = el.querySelector(`.thumb[data-fid="${CSS.escape(i)}"]`); if (t) t.classList.add("dragging"); }
  }
  function moveGhost() { drag.ghost.style.transform = `translate(${drag.x + 10}px, ${drag.y + 10}px)`; }

  function clearIndicators() {
    for (const n of el.querySelectorAll(".drop-before, .drop-after, .drop-target")) n.classList.remove("drop-before", "drop-after", "drop-target");
  }
  function insertionIndex(card, x, y) {
    const ts = [...card.querySelectorAll(".thumb")];
    if (!ts.length) return { index: 0, ref: null, after: false };
    let best = 0, bestD = Infinity, after = false;
    ts.forEach((t, i) => {
      const r = t.getBoundingClientRect();
      const dx = x - (r.left + r.width / 2), dy = (y - (r.top + r.height / 2)) * 2;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; after = x > r.left + r.width / 2; }
    });
    return { index: best + (after ? 1 : 0), ref: ts[best], after };
  }
  function updateDrop() {
    clearIndicators();
    drag.drop = null;
    const under = document.elementFromPoint(drag.x, drag.y);
    if (!under || !el.contains(under)) return;
    const zone = under.closest(".newzone");
    if (zone) { zone.classList.add("drop-target"); drag.drop = { type: "new", at: essays().length }; return; }
    const card = under.closest(".essay");
    if (!card) return;
    const ei = Number(card.dataset.ei);
    const ins = insertionIndex(card, drag.x, drag.y);
    card.classList.add("drop-target");
    if (ins.ref) ins.ref.classList.add(ins.after ? "drop-after" : "drop-before");
    drag.drop = { type: "essay", essay: ei, index: ins.index };
  }
  function scrollTick() {
    cancelAnimationFrame(drag.raf);
    const edge = 70;
    const dy = drag.y < edge ? -Math.ceil((edge - drag.y) / 5) : drag.y > window.innerHeight - edge ? Math.ceil((drag.y - (window.innerHeight - edge)) / 5) : 0;
    if (!dy) return;
    const d = drag;
    d.raf = requestAnimationFrame(() => { if (drag !== d) return; window.scrollBy(0, dy); updateDrop(); scrollTick(); });
  }
  function endDrag() {
    if (!drag) return;
    cancelAnimationFrame(drag.raf);
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);
    window.removeEventListener("pointercancel", onPointerCancel);
    window.removeEventListener("keydown", onDragKey, true);
    if (drag.ghost) drag.ghost.remove();
    document.body.classList.remove("dragging");
    clearIndicators();
    for (const t of el.querySelectorAll(".thumb.dragging")) t.classList.remove("dragging");
    try { drag.thumb.releasePointerCapture(drag.pointerId); } catch { /* ignore */ }
  }
  function onPointerUp(e) {
    if (!drag || e.pointerId !== drag.pointerId) return;
    const d = drag;
    if (d.active) { updateDrop(); }
    const drop = d.active ? d.drop : null;
    const wasActive = d.active;
    d.active = false;
    endDrag(); drag = null;
    if (wasActive) { justDragged = true; setTimeout(() => { justDragged = false; }, 0); }
    if (drop) {
      const es = essays();
      apply(drop.type === "new" ? M.moveToNewEssay(es, d.ids, drop.at) : M.movePhotos(es, d.ids, drop.essay, drop.index), d.id);
    } else if (pendingRender) render();
  }
  function onPointerCancel(e) {
    if (!drag || e.pointerId !== drag.pointerId) return;
    const was = drag.active; drag.active = false; endDrag(); drag = null;
    if (was) { justDragged = true; setTimeout(() => { justDragged = false; }, 0); }
    if (pendingRender) render();
  }
  function onDragKey(e) {
    if (e.key === "Escape" && drag) { e.preventDefault(); e.stopPropagation(); const was = drag.active; drag.active = false; endDrag(); drag = null; if (was) { justDragged = true; setTimeout(() => { justDragged = false; }, 0); } if (pendingRender) render(); }
  }

  // ---------- public API ----------
  render();
  return {
    el,
    setFiles(list) { files = new Map(list.map((f) => [f.file_id, f])); render(); },
    setMaxPages(n) { max = n; render(); },
    setEssays(next, { resetHistory = false } = {}) {
      if (resetHistory) { hist = M.createHistory(next); selected.clear(); }
      else hist = M.commit(hist, next);
      pruneSelection(); render();
    },
    getEssays() { return M.clone(essays()); },
    hasEdits() { return M.canUndo(hist); },
    setProblems(p) { const n = Array.isArray(p) ? p : []; if (JSON.stringify(n) === JSON.stringify(serverProblems)) return; serverProblems = n; render(); },
    /** Local instant validity (empty / too many); the server result stays authoritative. */
    localProblems() { return M.validate(essays(), max); },
    destroy() { document.removeEventListener("keydown", onDocKey); if (drag) { drag.active = false; endDrag(); drag = null; } },
  };
}
