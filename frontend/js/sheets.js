// #/sheets and #/sheets/<id> : answer-sheet templates. Blank sheets (no handwriting) are uploaded once; the printed
// text on them (headers, instructions, barcode digits, page numbers) is recognised and later removed from essays
// recognised with local OCR. A template can grow (more pages later) and be reused by every batch.
import { api, sheetPageImageUrl } from "./api.js";
import { h, mount, field, fmtTime } from "./dom.js";
import { errorBox, spinner, toast, openLightbox } from "./ui.js";

export const SHEET_WARNING = {
  no_printed_text: "这一页没有识别到印刷文字，对过滤没有帮助，可以停用。",
  looks_like_written_page: "这一页文字很多，像是写过字的作文纸。模板必须是空白纸，否则作文内容可能被当成印刷文字去掉，建议停用这一页并重新上传空白纸。",
};
const PAGE_STATUS = { pending: "识别中", ready: "可用", failed: "识别失败" };
const REJECT_TEXT = {
  unsupported_type: "不支持的文件类型", file_too_large: "文件过大", corrupt_image: "图片无法读取", too_many_pages: "页数已达上限",
};

export function sheetStateText(t) {
  const c = t.pages || {};
  const parts = [`可用 ${c.ready || 0}`];
  if (c.pending) parts.push(`识别中 ${c.pending}`);
  if (c.failed) parts.push(`失败 ${c.failed}`);
  if (c.disabled) parts.push(`停用 ${c.disabled}`);
  return parts.join(" · ");
}

export function renderSheets(root, ctx, tid) {
  return tid ? renderDetail(root, ctx, tid) : renderList(root, ctx);
}

const editable = (ctx, t) => ctx.isRoot || t.owner_id === ctx.user.id;

function intro() {
  return h("div", { class: "muted small" },
    h("p", { text: "使用本地识别前，请先上传空白答题纸（同款纸张、没有写字）。系统会记下纸上印刷的文字——标题、说明、页边提示、条形码数字、页码等，识别作文时自动去掉这些内容。" }),
    h("p", { text: "一套答题纸有多页（封面、内页……）时，每种页面各拍一张空白的。模板可以反复使用，之后也能继续补充新的空白页。拍摄时尽量拍全整张纸、光线均匀。" }));
}

// ------------------------------------------------------------------ list
function renderList(root, ctx) {
  let alive = true;
  const errHost = h("div"), listHost = h("div"), formHost = h("div");

  async function load() {
    mount(errHost);
    try {
      const items = await api.listSheets();
      if (!alive) return;
      if (!items.length) { mount(listHost, h("div", { class: "card empty" }, h("p", { text: "还没有答题纸模板。点击“新建模板”，再上传空白答题纸照片。" }))); return; }
      mount(listHost, h("div", { class: "grid-cards" }, items.map((t) => h("article", { class: "card" },
        h("div", { class: "batch-head" },
          h("h3", { text: t.name }),
          h("div", { class: "row" },
            t.usable ? h("span", { class: "badge badge-ok", text: "可用" }) : h("span", { class: "badge badge-warn", text: (t.pages && t.pages.pending) ? "识别中" : "无可用页" }),
            t.shared ? h("span", { class: "badge badge-info", text: "已共享" }) : null,
            t.owner_id !== ctx.user.id ? h("span", { class: "badge", text: ctx.isRoot ? `用户 ${t.owner_id}` : "他人共享（只读）" }) : null)),
        h("dl", { class: "kv" },
          h("dt", { text: "页面" }), h("dd", { text: sheetStateText(t) }),
          h("dt", { text: "更新时间" }), h("dd", { text: fmtTime(t.updated_at) })),
        h("div", { class: "row" }, h("button", { type: "button", class: "btn btn-small", text: editable(ctx, t) ? "打开 / 上传空白页" : "查看", onclick: () => ctx.go("#/sheets/" + encodeURIComponent(t.id)) }))))));
    } catch (e) { mount(listHost); mount(errHost, errorBox(e)); }
  }

  function openForm() {
    const name = h("input", { type: "text", maxlength: "100", placeholder: "例如：IGCSE 0500 答题纸" });
    const msg = h("div");
    const save = h("button", { type: "button", class: "btn btn-primary", text: "创建", onclick: async () => {
      mount(msg);
      if (!name.value.trim()) { mount(msg, errorBox({ message: "请填写模板名称", code: "form_incomplete" })); return; }
      save.disabled = true;
      try { const t = await api.createSheet(name.value.trim()); toast("已创建，请上传空白答题纸", "ok"); ctx.go("#/sheets/" + encodeURIComponent(t.id)); }
      catch (e) { save.disabled = false; mount(msg, errorBox(e)); }
    } });
    mount(formHost, h("form", { class: "card", novalidate: true, onsubmit: (e) => { e.preventDefault(); save.click(); } },
      h("h2", { text: "新建答题纸模板" }), field("模板名称", name).row, msg,
      h("div", { class: "row" }, save, h("button", { type: "button", class: "btn", text: "取消", onclick: () => mount(formHost) }))));
    name.focus();
  }

  mount(root, h("section", { class: "page" },
    h("div", { class: "row between" }, h("h1", { text: "答题纸模板" }), h("button", { type: "button", class: "btn btn-primary", text: "新建模板", onclick: openForm })),
    intro(), errHost, formHost, listHost));
  mount(listHost, spinner());
  load();
  return () => { alive = false; };
}

// ------------------------------------------------------------------ detail
function renderDetail(root, ctx, tid) {
  let alive = true, timer = null, t = null, built = false;
  const errHost = h("div"), headHost = h("div"), statusHost = h("div"), upHost = h("div"), pagesHost = h("div");
  const openLines = new Set();        // page ids whose "printed text" list is expanded (kept across refreshes)

  async function load() {
    clearTimeout(timer);
    try {
      t = await api.getSheet(tid);
      if (!alive) return;
      mount(errHost);
      if (!built) { built = true; paintHead(); paintUpload(editable(ctx, t)); }   // inputs are never rebuilt
      paintStatus(); paintPages(editable(ctx, t));
      if (t.pages && t.pages.pending) timer = setTimeout(load, 2000);      // recognition runs in the background
    } catch (e) {
      if (!alive) return;
      if (!built) { mount(headHost); mount(pagesHost); }       // first load failed (e.g. 404); later: keep the page
      mount(errHost, errorBox(e));
      if (built && t && t.pages && t.pages.pending) timer = setTimeout(load, 5000);
    }
  }

  function paintHead() {
    const can = editable(ctx, t);
    const title = h("h2", { text: t.name });
    const nameIn = h("input", { type: "text", maxlength: "100", value: t.name, "aria-label": "模板名称" });
    const renameBtn = h("button", { type: "button", class: "btn btn-small", text: "改名", onclick: async () => {
      if (!nameIn.value.trim()) return;
      renameBtn.disabled = true;
      try { const r = await api.renameSheet(tid, nameIn.value.trim()); title.textContent = r.name; toast("已改名", "ok"); }
      catch (e) { mount(errHost, errorBox(e)); }
      finally { renameBtn.disabled = false; }
    } });
    const share = ctx.isRoot ? (() => {
      const box = h("input", { type: "checkbox", checked: !!t.shared, onchange: async () => {
        box.disabled = true;
        try { await api.shareSheet(tid, box.checked); toast(box.checked ? "已共享给所有用户" : "已取消共享", "ok"); }
        catch (e) { box.checked = !box.checked; mount(errHost, errorBox(e)); }
        finally { box.disabled = false; }
      } });
      return h("label", { class: "check small" }, box, "共享给所有人（其他用户可选用，不能修改）");
    })() : null;
    mount(headHost, h("div", { class: "card" },
      h("div", { class: "batch-head" }, title, statusHost),
      can ? h("div", { class: "row" }, nameIn, renameBtn) : h("p", { class: "muted small", text: "这是他人共享的模板，只能查看和选用。" }),
      share));
  }

  function paintStatus() {
    mount(statusHost, h("div", { class: "row" },
      t.usable ? h("span", { class: "badge badge-ok", text: "可用于批改" }) : h("span", { class: "badge badge-warn", text: t.pages && t.pages.pending ? "识别中…" : "还没有可用的空白页" }),
      h("span", { class: "muted small", text: `页面：${sheetStateText(t)} · 更新于 ${fmtTime(t.updated_at)}` })));
  }

  function paintUpload(can) {
    if (!can) { mount(upHost); return; }
    const input = h("input", { type: "file", multiple: true, accept: "image/jpeg,image/png,image/webp,image/heic,image/heif,.heic,.heif" });
    const msg = h("div");
    const btn = h("button", { type: "button", class: "btn btn-primary", text: "上传空白页", onclick: async () => {
      mount(msg);
      const files = [...(input.files || [])];
      if (!files.length) { mount(msg, errorBox({ message: "请先选择空白答题纸照片", code: "form_incomplete" })); return; }
      if (files.length > 10) { mount(msg, errorBox({ message: "每次最多上传 10 张", code: "too_many_files" })); return; }
      btn.disabled = true; mount(msg, spinner("上传中…"));
      try {
        const r = await api.uploadSheetPages(tid, files);
        input.value = "";
        const rej = r.rejected || [];
        mount(msg, rej.length ? h("div", { class: "alert alert-warn", role: "status" },
          h("div", { text: `${files.length - rej.length} 张已上传，${rej.length} 张未接收：` }),
          h("ul", null, rej.map((x) => h("li", { text: `${x.name || "（未命名）"}：${REJECT_TEXT[x.code] || x.message || x.code}` }))))
          : h("div", { class: "alert", role: "status", text: `${files.length} 张已上传，正在识别印刷文字…` }));
        await load();
      } catch (e) { mount(msg, errorBox(e)); }
      finally { btn.disabled = false; }
    } });
    mount(upHost, h("div", { class: "card" },
      h("h3", { text: "上传空白答题纸" }),
      h("p", { class: "muted small", text: "只上传没有写字的空白纸。每次最多 10 张，一个模板最多 20 页。识别在后台进行，完成后即可在上传作文时选用。" }),
      h("div", { class: "row" }, input, btn), msg));
  }

  function paintPages(can) {
    const list = t.page_list || [];
    if (!list.length) { mount(pagesHost, h("div", { class: "card empty" }, h("p", { text: "还没有空白页。" }))); return; }
    const imgs = list.map((p, k) => ({ src: sheetPageImageUrl(tid, p.page_id), caption: `第 ${k + 1} 页 · ${p.orig_name || ""}` }));
    mount(pagesHost, h("div", { class: "sheet-pages" }, list.map((p, k) => {
      const status = p.disabled ? "已停用" : (PAGE_STATUS[p.status] || p.status);
      const cls = p.disabled ? "badge" : p.status === "ready" ? "badge badge-ok" : p.status === "failed" ? "badge badge-bad" : "badge badge-warn";
      return h("article", { class: "card sheet-page" + (p.disabled ? " is-disabled" : "") },
        h("button", { type: "button", class: "sheet-thumb", "aria-label": `查看第 ${k + 1} 页大图`, onclick: () => openLightbox(imgs, k) },
          h("img", { src: imgs[k].src, alt: "", loading: "lazy" })),
        h("div", { class: "sheet-page-body" },
          h("div", { class: "row between" }, h("strong", { text: `第 ${k + 1} 页` }), h("span", { class: cls, text: status })),
          h("div", { class: "muted small", text: `${p.orig_name || ""}${p.width ? ` · ${p.width}×${p.height}` : ""} · ${fmtTime(p.created_at)}` }),
          p.status === "failed" && p.error ? h("div", { class: "alert alert-error small", text: `识别失败：${p.error}` }) : null,
          (p.warnings || []).map((w) => h("div", { class: "alert alert-warn small", role: "status", text: SHEET_WARNING[w] || w })),
          p.status === "ready" ? h("details", { open: openLines.has(p.page_id), ontoggle: (e) => { if (e.target.open) openLines.add(p.page_id); else openLines.delete(p.page_id); } },
            h("summary", { text: `识别到的印刷文字（${(p.lines || []).length} 处）` }),
            h("ul", { class: "sheet-lines" }, (p.lines || []).map((l) => h("li", { class: "mono small", text: l.text })))) : null,
          can ? h("div", { class: "row" },
            p.status === "failed" ? h("button", { type: "button", class: "btn btn-small", text: "重新识别", onclick: (e) => act(e, () => api.retrySheetPage(tid, p.page_id), "已重新排队") }) : null,
            h("button", { type: "button", class: "btn btn-small", text: p.disabled ? "启用" : "停用",
              onclick: (e) => act(e, () => api.setSheetPage(tid, p.page_id, !p.disabled), p.disabled ? "已启用" : "已停用（之后创建的批次不再使用这一页）") })) : null));
    })));
  }

  async function act(e, fn, okText) {
    e.target.disabled = true;
    try { await fn(); toast(okText, "ok"); await load(); }
    catch (err) { e.target.disabled = false; mount(errHost, errorBox(err)); }
  }

  mount(root, h("section", { class: "page" },
    h("div", { class: "row" }, h("button", { type: "button", class: "btn btn-small", text: "← 返回模板列表", onclick: () => ctx.go("#/sheets") })),
    h("h1", { text: "答题纸模板" }), intro(), errHost, headHost, upHost, pagesHost));
  mount(headHost, spinner());
  load();
  return () => { alive = false; clearTimeout(timer); };
}
