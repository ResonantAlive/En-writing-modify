// Upload page: file upload, grouping (single / per-N / manual), batch creation form.
import { api } from "./api.js";
import { h, mount, field, debounce, fmtSize } from "./dom.js";
import { errorBox, infoBox, modal, confirmModal, openLightbox, problemText, spinner, toast } from "./ui.js";
import { createGroupEditor } from "./group.js";
import * as M from "./group_model.mjs";
import { hostOf } from "./render_model.mjs";
import { preflightAck, confirmForProvider } from "./thirdparty.js";
import { sheetStateText } from "./sheets.js";

const CHUNK = 6;
const SHEET_REASON = {
  sheet_template_required: "本地识别需要选择答题纸模板（先到“答题纸”页上传空白纸）",
  invalid_sheet_template: "找不到所选答题纸模板",
  sheet_template_not_ready: "所选答题纸模板还没有识别完成的空白页，请稍候或到“答题纸”页查看",
};
const BASIS_LABEL = { filename: "文件名中的编号", exif: "拍摄时间（EXIF）", mtime: "文件修改时间" };

export function renderUpload(root, ctx) {
  const S = {
    limits: null, uploadId: null, files: [], rejected: [], uploading: null,
    multi: false, basis: null, mode: "per_n", n: 2, nCustom: false,
    preview: null, previewPending: false, previewErr: null, baseOrder: null,
    manualInit: false, confirmed: false, localOk: null, localReason: null, precheck: null, submitting: false,
    providers: [], templates: [], tplMode: null, sheets: [],
  };
  let alive = true, seq = 0;
  const editor = createGroupEditor({
    maxPages: 0,
    getBaseOrder: () => S.baseOrder,
    onChange: () => { S.confirmed = false; if (confirmBox) confirmBox.checked = false; schedulePreview(); paintAction(); },
  });
  let confirmBox = null;

  // ---------- hosts ----------
  const limitsHost = h("div"), dropHost = h("div"), msgHost = h("div"), listHost = h("div"),
    groupHost = h("div"), previewHost = h("div"), actionHost = h("div"), formHost = h("div"), submitMsg = h("div");

  // ---------- upload ----------
  const fileInput = h("input", { type: "file", multiple: true, class: "visually-hidden", tabindex: "-1", "aria-label": "选择照片",
    onchange: () => { const l = [...fileInput.files]; fileInput.value = ""; addFiles(l); } });

  async function addFiles(list) {
    if (!list.length || S.uploading || !S.limits) return;
    const L = S.limits, problems = [];
    let ok = [];
    const room = Math.max(0, (L.max_files_per_batch || 0) - S.files.length);
    for (const f of list) {
      if (L.max_file_mb && f.size > L.max_file_mb * 1048576) problems.push({ orig_name: f.name, message: `文件过大（${fmtSize(f.size)}，上限 ${L.max_file_mb} MB）` });
      else if (ok.length >= room) problems.push({ orig_name: f.name, message: `超出单批照片数量上限（${L.max_files_per_batch} 张）` });
      else ok.push(f);
    }
    S.rejected = problems;
    S.uploading = { done: 0, total: ok.length }; paintDrop(); paintMsg();
    try {
      for (let i = 0; i < ok.length && alive; i += CHUNK) {
        const chunk = ok.slice(i, i + CHUNK);
        const res = await api.uploadFiles(chunk, chunk.map((f) => f.lastModified), S.uploadId);
        S.uploadId = res.upload_id;
        S.files = res.files;   // server returns ALL files of the upload, not only this request's
        for (const r of res.rejected || []) S.rejected.push({ orig_name: r.orig_name, message: r.message || r.code });
        S.uploading.done = Math.min(ok.length, i + chunk.length);
        paintDrop(); paintList();
      }
    } catch (e) { S.uploadError = e; }
    S.uploading = null;
    if (!alive) return;
    paintDrop(); paintMsg(S.uploadError); S.uploadError = null;
    filesChanged();
  }

  async function removeFile(f) {
    try { await api.deleteUploadFile(S.uploadId, f.file_id); } catch (e) { mount(msgHost, errorBox(e)); return; }
    S.files = S.files.filter((x) => x.file_id !== f.file_id);
    filesChanged();
  }
  async function clearAll() {
    if (!S.uploadId) return;
    if (!(await confirmModal("清空照片", "将删除已上传但尚未创建批次的所有照片，确定吗？", "清空"))) return;
    try { await api.deleteUpload(S.uploadId); } catch (e) { mount(msgHost, errorBox(e)); return; }
    S.uploadId = null; S.files = []; S.rejected = []; S.preview = null; S.baseOrder = null; S.manualInit = false; S.mode = "per_n";
    editor.setEssays([], { resetHistory: true });
    filesChanged();
  }

  function filesChanged() {
    editor.setFiles(S.files);
    if (S.manualInit) editor.setEssays(M.reconcile(editor.getEssays(), S.files.map((f) => f.file_id)));
    S.confirmed = false;
    paintList(); paintGroup(); paintMsg();
    if (!S.files.length) { S.preview = null; S.previewPending = false; seq++; paintPreview(); paintAction(); return; }
    schedulePreview();
  }

  // ---------- grouping preview ----------
  function groupingReq(forOrdering) {
    if (!S.multi) return { multi_photo: false, order_basis: null, mode: null, n: null, essays: null };
    const base = { multi_photo: true, order_basis: S.basis };
    if (S.mode === "per_n") return { ...base, mode: "per_n", n: S.n, essays: null };
    return { ...base, mode: "manual", n: null, essays: forOrdering || !S.manualInit ? null : editor.getEssays() };
  }

  const runPreview = debounce(async () => {
    if (!alive || !S.uploadId || !S.files.length) return;
    const my = ++seq;
    const req = groupingReq(false);
    try {
      const r = await api.groupingPreview(S.uploadId, req);
      if (my !== seq || !alive) return;
      S.previewErr = null; S.previewPending = false; S.preview = r;
      if (req.essays == null && Array.isArray(r.ordered_file_ids)) S.baseOrder = r.ordered_file_ids;
      if (S.multi) {
        if (r.mode === "manual" && S.mode === "per_n") { S.mode = "manual"; S.manualInit = true; editor.setEssays(r.essays, { resetHistory: true }); S.confirmed = false; paintGroup(); }
        else if (S.mode === "manual" && !S.manualInit) { S.manualInit = true; editor.setEssays(r.essays, { resetHistory: true }); }
        const key = JSON.stringify([r.order_bases, r.order_basis]);
        if (key !== S.basesKey) { S.basesKey = key; paintGroup(); }
      }
      editor.setProblems(r.problems);
    } catch (e) {
      if (my !== seq) return;
      S.previewPending = false; S.previewErr = e; S.preview = null;
    }
    paintPreview(); paintAction();
  }, 350);

  function schedulePreview() { S.previewPending = true; paintAction(); runPreview(); }

  // ---------- painting ----------
  function paintDrop() {
    const busy = S.uploading;
    mount(dropHost, h("div", { class: "dropzone", role: "button", tabindex: "0", "aria-label": "拖入或选择作文照片",
      onclick: () => !busy && fileInput.click(),
      onkeydown: (e) => { if ((e.key === "Enter" || e.key === " ") && !busy) { e.preventDefault(); fileInput.click(); } },
      ondragover: (e) => { if (e.dataTransfer && [...e.dataTransfer.types].includes("Files")) { e.preventDefault(); e.currentTarget.classList.add("over"); } },
      ondragleave: (e) => e.currentTarget.classList.remove("over"),
      ondrop: (e) => { e.preventDefault(); e.currentTarget.classList.remove("over"); if (e.dataTransfer) addFiles([...e.dataTransfer.files]); } },
    busy ? spinner(`上传中 ${S.uploading.done} / ${S.uploading.total}…`)
      : [h("strong", { text: "把作文照片拖到这里，或点击选择" }),
        h("div", { class: "muted small", text: S.limits ? `JPG / PNG / WebP / HEIC，每张不超过 ${S.limits.max_file_mb} MB，最多 ${S.limits.max_files_per_batch} 张` : "" })]));
  }

  function paintMsg(err) {
    const out = [];
    if (err) out.push(errorBox(err));
    if (S.rejected.length) {
      out.push(h("div", { class: "alert alert-warn", role: "alert" },
        h("strong", { text: `${S.rejected.length} 个文件未被接收：` }),
        h("ul", null, S.rejected.slice(0, 30).map((r) => h("li", { text: `${r.orig_name}：${r.message}` }))),
        h("button", { type: "button", class: "btn btn-small", text: "知道了", onclick: () => { S.rejected = []; paintMsg(); } })));
    }
    mount(msgHost, out);
  }

  function paintList() {
    if (!S.files.length) { mount(listHost); return; }
    mount(listHost, h("div", { class: "card" },
      h("div", { class: "row between" }, h("strong", { text: `已上传 ${S.files.length} 张` }), h("button", { type: "button", class: "btn btn-small btn-danger", text: "全部清空", onclick: clearAll })),
      h("div", { class: "file-grid" }, S.files.map((f, i) => h("div", { class: "file-card" },
        h("button", { type: "button", class: "thumb-btn", aria: { label: `放大查看 ${f.orig_name}` }, onclick: () => openLightbox(S.files.map((x) => ({ src: x.thumb_url, caption: x.orig_name })), i) },
          h("img", { src: f.thumb_url, alt: "", loading: "lazy" })),
        h("div", { class: "fname", title: f.orig_name, text: f.orig_name }),
        h("button", { type: "button", class: "btn btn-small", text: "删除", aria: { label: `删除 ${f.orig_name}` }, onclick: () => removeFile(f) }))))));
  }

  function paintGroup() {
    if (!S.files.length) { mount(groupHost); return; }
    const multiBox = h("input", { type: "checkbox", checked: S.multi, onchange: () => {
      S.multi = multiBox.checked; S.confirmed = false; S.basis = null;
      if (!S.multi) { S.manualInit = false; S.mode = "per_n"; }
      paintGroup(); schedulePreview();
    } });
    const parts = [h("label", { class: "check big" }, multiBox, "一篇作文可能包含多张照片")];
    if (S.multi) {
      const bases = (S.preview && S.preview.order_bases) || {};
      const cur = S.preview && S.preview.order_basis;
      const basisSel = h("select", { "aria-label": "排序依据", onchange: () => { S.basis = basisSel.value || null; S.confirmed = false; schedulePreview(); } },
        h("option", { value: "", text: "自动选择（推荐）" }),
        ["filename", "exif", "mtime"].map((k) => h("option", { value: k, text: BASIS_LABEL[k] + (bases[k] && !bases[k].usable ? "（不可用）" : ""), disabled: !!(bases[k] && !bases[k].usable), selected: S.basis === k })));
      basisSel.value = S.basis || "";
      const basisText = !S.preview ? "" : cur === "none" ? "没有可靠的排序依据，请手动整理" : cur && BASIS_LABEL[cur] ? `当前按「${BASIS_LABEL[cur]}」排序` : "";
      const unusable = ["filename", "exif", "mtime"].filter((k) => bases[k] && !bases[k].usable && bases[k].reason).map((k) => `${BASIS_LABEL[k]}：${bases[k].reason}`);
      parts.push(h("div", { class: "card inner" },
        field("照片排序依据", basisSel, basisText).row,
        unusable.length ? h("ul", { class: "muted small" }, unusable.map((u) => h("li", { text: u }))) : null,
        h("fieldset", { class: "radio-group" }, h("legend", { text: "分组方式" }),
          modeRadio("per_n", "每 N 张为一篇"),
          S.mode === "per_n" ? nChooser() : null,
          modeRadio("manual", "手动整理（拖动、拆分、合并）"))));
      if (S.mode === "manual") {
        parts.push(h("div", { class: "card inner" }, h("p", { class: "muted small", text: "拖动缩略图调整顺序或移到其他作文；点击缩略图放大；勾选多张后可合并 / 拆开。触屏请按住 ⠿ 拖动。" }), editor.el));
      }
    }
    mount(groupHost, parts);
  }

  function modeRadio(value, label) {
    const r = h("input", { type: "radio", name: "gmode", value, checked: S.mode === value, onchange: async () => {
      if (S.mode === value) return;
      if (value === "per_n" && S.manualInit && editor.hasEdits() && !(await confirmModal("放弃手动整理？", "切换回“每 N 张为一篇”会丢弃你在手动模式下的调整。", "切换"))) { r.checked = false; paintGroup(); return; }
      S.mode = value; S.confirmed = false;
      if (value === "manual") {
        S.manualInit = true;
        const es = S.preview && S.preview.essays && S.preview.essays.length ? S.preview.essays : M.singles(S.baseOrder || S.files.map((f) => f.file_id));
        editor.setEssays(M.reconcile(es, S.files.map((f) => f.file_id)), { resetHistory: true });
      } else S.manualInit = false;
      paintGroup(); schedulePreview();
    } });
    return h("label", { class: "check" }, r, label);
  }

  function nChooser() {
    const sel = h("select", { "aria-label": "每篇照片数" }, [2, 3].map((v) => h("option", { value: String(v), text: `${v} 张` })), h("option", { value: "custom", text: "自定义…" }));
    const custom = h("input", { type: "number", min: "1", max: "50", class: "n-input", value: String(S.n), "aria-label": "自定义每篇照片数", hidden: !S.nCustom });
    sel.value = S.nCustom ? "custom" : String(S.n);
    sel.addEventListener("change", () => {
      if (sel.value === "custom") { S.nCustom = true; custom.hidden = false; custom.focus(); }
      else { S.nCustom = false; custom.hidden = true; S.n = Number(sel.value); schedulePreview(); }
    });
    custom.addEventListener("input", () => { const v = Math.floor(Number(custom.value)); if (v >= 1) { S.n = Math.min(v, 50); schedulePreview(); } });
    return h("div", { class: "row indent" }, "每", sel, custom, "为一篇");
  }

  function paintPreview() {
    const nodes = [];
    if (S.previewErr) nodes.push(errorBox(S.previewErr));
    const r = S.preview;
    if (r) {
      const notes = (r.problems || []).filter((p) => ["not_divisible", "no_reliable_order", "basis_unusable"].includes(p.type));
      for (const p of notes) nodes.push(infoBox(problemText(p) + (p.detail ? `（${p.detail}）` : ""), "warn"));
      if (!S.multi || S.mode !== "manual") nodes.push(staticEssays(r));
    }
    mount(previewHost, nodes);
  }

  function staticEssays(r) {
    const byId = new Map(S.files.map((f) => [f.file_id, f]));
    const flat = M.flatten(r.essays || []);
    return h("div", { class: "card inner" },
      h("div", { class: "ge-summary" }, h("strong", { text: `共 ${r.summary ? r.summary.essays : (r.essays || []).length} 篇作文、${r.summary ? r.summary.photos : flat.length} 张照片` })),
      h("div", { class: "static-list" }, (r.essays || []).map((e, ei) => {
        const bad = (r.problems || []).some((p) => p.essay_index === ei);
        return h("div", { class: "essay static" + (bad ? " invalid" : "") },
          h("div", { class: "essay-head" }, h("h3", { text: `作文 ${ei + 1}（${e.length} 张）` })),
          h("div", { class: "thumbs" }, e.map((id) => {
            const f = byId.get(id);
            return h("button", { type: "button", class: "thumb static-thumb", aria: { label: `放大 ${f ? f.orig_name : ""}` },
              onclick: () => openLightbox(flat.map((x) => ({ src: byId.get(x) ? byId.get(x).thumb_url : "", caption: byId.get(x) ? byId.get(x).orig_name : "" })), flat.indexOf(id)) },
            h("img", { src: f ? f.thumb_url : "", alt: "" }));
          })));
      })));
  }

  // ---------- action (submit) ----------
  function isManualFinal() { return S.multi && (S.mode === "manual" || (S.preview && S.preview.mode === "manual")); }

  function submitReasons() {
    const out = [];
    if (S.uploading) out.push("照片仍在上传");
    if (!S.files.length) { out.push("请先上传照片"); return out; }
    if (S.previewPending) out.push("正在校验分组…");
    if (S.previewErr) out.push("分组校验失败：" + S.previewErr.message);
    const r = S.preview;
    if (r && !r.can_submit) {
      const seen = new Set();
      for (const p of r.problems || []) { if (["not_divisible", "no_reliable_order", "basis_unusable"].includes(p.type)) continue; const t = problemText(p); if (!seen.has(t)) { seen.add(t); out.push(t); } }
      if (!out.length) out.push("分组暂不可提交");
    }
    if (S.multi && S.mode === "manual") for (const p of editor.localProblems()) out.push(problemText(p));
    if (!S.previewPending && r && isManualFinal() && !S.confirmed) out.push("请勾选“我已核对分组无误”");
    if (S.submitting) out.push("正在提交…");
    return [...new Set(out)];
  }

  function paintAction() {
    const reasons = submitReasons();
    const manual = isManualFinal();
    confirmBox = h("input", { type: "checkbox", checked: S.confirmed, onchange: () => { S.confirmed = confirmBox.checked; paintAction(); } });
    const btn = h("button", { type: "button", class: "btn btn-primary btn-big", text: "创建批次", disabled: reasons.length > 0 || !!S.submitting, onclick: submit });
    mount(actionHost,
      S.files.length && S.preview && manual ? h("label", { class: "check confirm" }, confirmBox, "我已核对分组无误，按此分组创建批次") : null,
      reasons.length ? h("ul", { class: "reasons", "aria-live": "polite" }, reasons.map((x) => h("li", { text: x }))) : null,
      btn);
  }

  // ---------- create form ----------
  const F = {};
  function buildForm() {
    F.topic = h("input", { type: "text", maxlength: "500" });
    F.rubric = h("textarea", { rows: "5", placeholder: "评分标准 / 评分细则" });
    F.genre = h("input", { type: "text", maxlength: "100", placeholder: "例如：议论文 / 应用文" });
    F.full = h("input", { type: "number", min: "1", step: "any", value: "100" });
    F.route = {};
    const routeRadio = (v, label) => { const i = h("input", { type: "radio", name: "route", value: v, onchange: syncForm }); F.route[v] = i; return h("label", { class: "check" }, i, label); };
    F.routeLocalLabel = routeRadio("local_ocr", "本地识别（先在本机识别文字）");
    F.routeVlmLabel = routeRadio("vlm", "视觉模型（把照片发送给模型）");
    F.provider = h("select", { "aria-label": "服务商" }, h("option", { value: "", text: "加载中…" }));
    F.model = h("input", { type: "text", placeholder: "默认使用所选服务商的默认模型" });
    F.template = h("select", { "aria-label": "提示词模板" }, h("option", { value: "", text: "请先选择处理路线" }));
    F.tplHint = h("div", { class: "hint" });
    F.provHint = h("div", { class: "hint" });
    F.provider.addEventListener("change", () => {
      const p = S.providers.find((x) => x.id === F.provider.value);
      if (p) F.model.value = p.default_model || "";
      F.provHint.textContent = p ? (p.is_third_party ? `第三方服务（${hostOf(p.base_url)}），首次使用需确认` : "本机服务") : "";
    });
    F.template.addEventListener("change", () => {
      const t = S.templates.find((x) => x.id === F.template.value);
      F.tplHint.textContent = t ? `将使用 v${t.latest_version}（创建时固定该版本）${t.builtin ? " · 内置" : ""}` : "";
    });
    F.choice = {};
    const choiceRadio = (v, label) => { const i = h("input", { type: "radio", name: "vchoice", value: v, onchange: syncForm }); F.choice[v] = i; return h("label", { class: "check" }, i, label); };
    F.choiceLocalLabel = choiceRadio("local_transcription", "先用本地模型转写文字");
    F.choiceBox = h("fieldset", { class: "radio-group", hidden: true }, h("legend", { text: "视觉模型处理方式（必选）" }),
      F.choiceLocalLabel, choiceRadio("skip_fewshot", "直接发送照片，不使用历史范例"), choiceRadio("history_fewshot", "直接发送照片，并使用历史范例"));
    F.engine = h("select", null, h("option", { value: "paddle", text: "PP-OCRv6（本地）" }));   // only one local engine
    F.engineRow = field("本地识别引擎", F.engine);
    F.localNote = h("div", { class: "alert alert-warn", hidden: true, role: "status" });
    F.sheet = h("select", { "aria-label": "答题纸模板" }, h("option", { value: "", text: "加载中…" }));
    F.sheetHint = h("div", { class: "hint" });
    F.sheetRow = h("div", null, field("答题纸模板（本地识别必选）", F.sheet).row, F.sheetHint,
      h("button", { type: "button", class: "btn btn-small", text: "管理答题纸模板 / 上传空白纸", onclick: () => ctx.go("#/sheets") }));
    F.sheet.addEventListener("change", () => { paintSheetHint(); try { if (F.sheet.value) localStorage.setItem("wm_last_sheet", F.sheet.value); } catch { /* storage unavailable */ } });
    F.scope = h("select", null, h("option", { value: "own", text: "仅我自己的范例" }), h("option", { value: "all", text: "所有人的范例（管理员）" }));
    F.scopeRow = field("范例范围", F.scope);
    F.scopeRow.row.hidden = !ctx.isRoot;
    F.info = h("div");
    const onForm = debounce(silentPrecheck, 700);
    for (const e of [F.topic, F.rubric, F.genre, F.full, F.provider, F.model, F.template, F.engine, F.sheet, F.scope]) { e.addEventListener("input", onForm); e.addEventListener("change", onForm); }
    mount(formHost, h("form", { class: "card", novalidate: true, onsubmit: (e) => e.preventDefault() },
      h("h2", { text: "批改设置" }),
      field("作文题目", F.topic).row, field("评分标准", F.rubric).row, field("体裁", F.genre).row, field("满分", F.full).row,
      h("fieldset", { class: "radio-group" }, h("legend", { text: "处理路线" }), F.routeLocalLabel, F.routeVlmLabel),
      F.localNote,
      field("服务商", F.provider).row, F.provHint,
      field("模型", F.model, "已按服务商默认模型预填，可修改").row, field("提示词模板", F.template).row, F.tplHint,
      F.choiceBox, F.engineRow.row, F.sheetRow, F.scopeRow.row, F.info));
    syncForm();
  }

  const route = () => (F.route.local_ocr.checked ? "local_ocr" : F.route.vlm.checked ? "vlm" : null);
  const choice = () => Object.keys(F.choice).find((k) => F.choice[k].checked) || null;
  const needsEngine = () => route() === "local_ocr" || (route() === "vlm" && choice() === "local_transcription");

  function syncForm() {
    const localBlocked = S.localOk === false;
    F.route.local_ocr.disabled = localBlocked; F.choice.local_transcription.disabled = localBlocked;
    if (localBlocked) {
      if (F.route.local_ocr.checked) F.route.local_ocr.checked = false;
      if (F.choice.local_transcription.checked) F.choice.local_transcription.checked = false;
    }
    F.localNote.hidden = !localBlocked;
    F.localNote.textContent = localBlocked ? `本地识别暂不可用：${S.localReason || "本地环境未就绪"}。请改用视觉模型。` : "";
    F.choiceBox.hidden = route() !== "vlm";
    F.engineRow.row.hidden = true;      // single local engine: nothing to choose
    F.sheetRow.hidden = !needsEngine();
    loadTemplates();
  }

  // answer-sheet templates: printed text of the blank sheet is removed from the local recognition
  async function loadSheets() {
    try {
      S.sheets = await api.listSheets();
      if (!alive) return;
      const usable = S.sheets.filter((t) => t.usable);
      mount(F.sheet, S.sheets.length
        ? [h("option", { value: "", text: "请选择答题纸模板" }), S.sheets.map((t) => h("option", { value: t.id, disabled: !t.usable,
          text: `${t.name}（${t.usable ? `可用 ${t.pages.ready} 页` : t.pages && t.pages.pending ? "识别中" : "无可用页"}）${t.shared && t.owner_id !== ctx.user.id ? " · 共享" : ""}` }))]
        : h("option", { value: "", text: "还没有模板，请先上传空白答题纸" }));
      let last = null;
      try { last = localStorage.getItem("wm_last_sheet"); } catch { /* storage unavailable */ }
      if (last && usable.some((t) => t.id === last)) F.sheet.value = last;
      else if (usable.length === 1) F.sheet.value = usable[0].id;
      paintSheetHint();
    } catch (e) { mount(F.sheet, h("option", { value: "", text: "答题纸模板加载失败" })); mount(submitMsg, errorBox(e)); }
  }
  function paintSheetHint() {
    const t = (S.sheets || []).find((x) => x.id === F.sheet.value);
    F.sheetHint.textContent = t ? `识别时会去掉这套答题纸上的印刷文字（标题、说明、条形码数字、页码等）· ${sheetStateText(t)}`
      : "本地识别需要先上传同款空白答题纸，用来去掉纸上的印刷文字。";
  }

  // template list is filtered by route: vlm => image templates, local_ocr => text templates
  async function loadTemplates() {
    const r = route(), mode = r === "vlm" ? "image" : r === "local_ocr" ? "text" : null;
    if (mode === S.tplMode) return;
    S.tplMode = mode;
    if (!mode) { S.templates = []; mount(F.template, h("option", { value: "", text: "请先选择处理路线" })); F.tplHint.textContent = ""; return; }
    mount(F.template, h("option", { value: "", text: "加载中…" }));
    try {
      const list = await api.listPrompts(mode);
      if (!alive || S.tplMode !== mode) return;
      S.templates = list;
      mount(F.template, list.length
        ? [h("option", { value: "", text: "请选择模板" }), list.map((t) => h("option", { value: t.id, text: `${t.name}（v${t.latest_version}${t.builtin ? "，内置" : ""}）` }))]
        : h("option", { value: "", text: "没有可用模板，请先到“提示词”页新建" }));
      if (list.length === 1) F.template.value = list[0].id;
      F.template.dispatchEvent(new Event("change"));
    } catch (e) { S.tplMode = null; mount(F.template, h("option", { value: "", text: "模板加载失败" })); mount(submitMsg, errorBox(e)); }
  }

  async function loadProviders() {
    try {
      S.providers = await api.listProviders();
      if (!alive) return;
      mount(F.provider, S.providers.length
        ? [h("option", { value: "", text: "请选择服务商" }), S.providers.map((p) => h("option", { value: p.id, text: `${p.alias}（默认模型：${p.default_model || "?"}）${p.is_third_party ? " · 第三方" : ""}${p.shared && p.owner_id !== ctx.user.id ? " · 共享" : ""}` }))]
        : h("option", { value: "", text: "没有可用服务商，请先到“服务商”页添加" }));
      if (S.providers.length === 1) { F.provider.value = S.providers[0].id; F.provider.dispatchEvent(new Event("change")); }
    } catch (e) { mount(F.provider, h("option", { value: "", text: "服务商加载失败" })); mount(submitMsg, errorBox(e)); }
  }
  const selectedProvider = () => S.providers.find((x) => x.id === F.provider.value) || { id: F.provider.value, alias: "", base_url: "" };

  function buildBody(extra) {
    const g = groupingReq(false);
    if (isManualFinal()) { g.mode = "manual"; g.essays = S.mode === "manual" && S.manualInit ? editor.getEssays() : (S.preview && S.preview.essays) || null; g.confirmed = !!S.confirmed; }
    const r = route();
    return {
      upload_id: S.uploadId, topic: F.topic.value.trim(), rubric: F.rubric.value.trim(), genre: F.genre.value.trim(),
      full_score: Number(F.full.value), route: r, provider_id: F.provider.value.trim(), model: F.model.value.trim() || null,
      template_id: F.template.value, template_version: null,
      vlm_transcription_choice: r === "vlm" ? choice() : null,
      local_engine: needsEngine() ? F.engine.value : null,
      sheet_template_id: needsEngine() ? (F.sheet.value || null) : null,
      fewshot_scope: ctx.isRoot ? F.scope.value : null,
      zero_shot_approved: null, third_party_ack: null, grouping: g, ...extra,
    };
  }

  function applyPrecheck(p) {
    S.precheck = p;
    S.localOk = p.local_env_ok; S.localReason = p.local_env_reason || null;
    syncForm();
    const notes = [];
    if (p.fewshot_available) notes.push(`可使用的历史范例：${p.fewshot_count} 条`);
    const warn = [];
    if (needsEngine() && p.sheet_template_ok === false && F.sheet.value) warn.push(SHEET_REASON[p.sheet_template_reason] || "所选答题纸模板暂不可用");
    mount(F.info, notes.map((t) => h("div", { class: "muted small", text: t })), warn.map((t) => h("div", { class: "alert alert-warn small", role: "status", text: t })));
  }

  async function silentPrecheck() {
    if (!alive || !S.uploadId || !S.files.length || !route()) return;
    try { applyPrecheck(await api.precheck(buildBody())); } catch { /* incomplete form: ignore, real errors shown on submit */ }
  }

  function clientValidate(b) {
    const miss = [];
    if (!b.topic) miss.push("作文题目"); if (!b.rubric) miss.push("评分标准"); if (!b.genre) miss.push("体裁");
    if (!(b.full_score > 0)) miss.push("满分（需大于 0）");
    if (!b.route) miss.push("处理路线");
    if (b.route === "vlm" && !b.vlm_transcription_choice) miss.push("视觉模型处理方式");
    if (!b.provider_id) miss.push("服务商");
    if (!b.template_id) miss.push("提示词模板");
    if (needsEngine() && !b.sheet_template_id) miss.push("答题纸模板（先到“答题纸”页上传空白纸）");
    return miss;
  }

  async function submit() {
    if (S.submitting) return;
    mount(submitMsg);
    const body = buildBody();
    const miss = clientValidate(body);
    if (miss.length) { mount(submitMsg, errorBox({ message: "请补全：" + miss.join("、"), code: "form_incomplete" })); return; }
    S.submitting = true; paintAction();
    try {
      const p = await api.precheck(body);
      applyPrecheck(p);
      if (needsEngine() && p.local_env_ok === false) throw { message: `本地识别环境不可用：${p.local_env_reason || ""}`, code: "local_env_unavailable" };
      if (needsEngine() && p.sheet_template_ok === false) throw { message: SHEET_REASON[p.sheet_template_reason] || "所选答题纸模板暂不可用", code: p.sheet_template_reason || "sheet_template_not_ready" };
      if (p.grouping) { S.preview = p.grouping; editor.setProblems(p.grouping.problems); }
      if (p.grouping && !p.grouping.can_submit) throw { message: "分组存在问题，无法提交", code: "grouping_invalid", detail: p.grouping.problems };
      if (p.needs_third_party_ack && !(await askThirdParty())) return;
      if (p.needs_third_party_ack) body.third_party_ack = true;
      else {
        const pre = await preflightAck(selectedProvider(), "作文照片 / 转写文字会被发送到该主机。");
        if (!pre.ok) return;
        if (pre.ack) body.third_party_ack = true;
      }
      if (p.zero_shot_consent_required) { if (!(await askZeroShot())) return; body.zero_shot_approved = true; }
      await create(body);
    } catch (e) {
      mount(submitMsg, errorBox(e && e.message ? e : { message: "提交失败" }));
    } finally { S.submitting = false; if (alive) paintAction(); }
  }

  async function create(body, retried) {
    try {
      const res = await api.createBatch(body);
      toast("批次已创建：" + res.task_code, "ok");
      ctx.go("#/batches/" + encodeURIComponent(res.task_code));
    } catch (e) {
      if (!retried && e.code === "third_party_ack_required" && (await askThirdParty())) return create({ ...body, third_party_ack: true }, true);
      if (!retried && e.code === "zero_shot_consent_required" && (await askZeroShot())) return create({ ...body, zero_shot_approved: true }, true);
      throw e;
    }
  }
  const askThirdParty = () => confirmForProvider(selectedProvider(), "作文照片 / 转写文字会被发送到该主机。");
  const askZeroShot = () => confirmModal("没有可用的历史范例", "当前没有可用的历史范例，模型将在没有范例参考的情况下评分，结果可能不够稳定。确认继续吗？", "确认，无范例评分");

  // ---------- boot ----------
  mount(root, h("section", { class: "page" }, h("h1", { text: "上传作文照片" }), limitsHost, fileInput, dropHost, msgHost, listHost, groupHost, previewHost,
    formHost, submitMsg, actionHost));
  paintDrop(); paintAction();
  buildForm();
  loadProviders();
  loadSheets();
  mount(limitsHost, spinner());
  api.limits().then((L) => {
    S.limits = L; mount(limitsHost);
    editor.setMaxPages(Number(L.max_pages_per_essay) || 0);
    if (L.default_local_engine) F.engine.value = L.default_local_engine;
    if (L.local_env_ok === false) { S.localOk = false; S.localReason = L.local_env_reason || null; syncForm(); }
    fileInput.accept = (L.allowed_types || []).join(",") + ",.heic,.heif";
    paintDrop();
  }).catch((e) => mount(limitsHost, errorBox(e)));

  const onWinDrop = (e) => { if (e.dataTransfer && [...(e.dataTransfer.types || [])].includes("Files")) e.preventDefault(); };
  window.addEventListener("dragover", onWinDrop); window.addEventListener("drop", onWinDrop);
  return () => { alive = false; runPreview.cancel(); editor.destroy(); window.removeEventListener("dragover", onWinDrop); window.removeEventListener("drop", onWinDrop); };
}
