// #/prompts : prompt templates (versions are immutable; edit = new version or new template), plus LLM-assisted drafts.
import { api } from "./api.js";
import { h, mount, field, fmtTime } from "./dom.js";
import { errorBox, spinner, toast, confirmModal, modal } from "./ui.js";
import { cpLen } from "./render_model.mjs";
import { preflightAck, confirmForProvider } from "./thirdparty.js";
import { reasonLabel } from "./labels.js";

const MODE_LABEL = { text: "文字（本地识别后）", image: "图片（视觉模型）" };

export function renderPrompts(root, ctx) {
  const errHost = h("div"), listHost = h("div"), detailHost = h("div"), assistHost = h("div");
  let alive = true, filter = "", items = [], openId = null;

  const mine = (t) => t.owner_id === ctx.user.id;
  const canEditInPlace = (t) => !t.builtin && (mine(t) || ctx.isRoot);

  async function loadList() {
    mount(errHost);
    try { items = await api.listPrompts(filter || undefined); } catch (e) { mount(listHost); mount(errHost, errorBox(e)); return; }
    if (alive) paintList();
  }

  function paintList() {
    const sel = h("select", { "aria-label": "按类型筛选", onchange: () => { filter = sel.value; loadList(); } },
      h("option", { value: "", text: "全部类型" }), h("option", { value: "text", text: "文字模式" }), h("option", { value: "image", text: "图片模式" }));
    sel.value = filter;
    mount(listHost, h("div", { class: "card" },
      h("div", { class: "row between" }, h("h2", { text: "提示词模板" }), h("div", { class: "row" }, "类型", sel)),
      items.length ? h("div", { class: "table-wrap" }, h("table", { class: "table" },
        h("thead", null, h("tr", null, ["名称", "类型", "最新版本", "状态", "更新时间", ""].map((t) => h("th", { text: t })))),
        h("tbody", null, items.map((t) => h("tr", { class: t.id === openId ? "row-active" : "" },
          h("td", { text: t.name }),
          h("td", { text: MODE_LABEL[t.input_mode] || t.input_mode }),
          h("td", { text: "v" + t.latest_version }),
          h("td", null, h("div", { class: "row" },
            t.builtin ? h("span", { class: "badge", text: "内置" }) : null,
            t.shared ? h("span", { class: "badge badge-info", text: "已共享" }) : null,
            !t.builtin && !mine(t) ? h("span", { class: "badge", text: ctx.isRoot ? `用户 ${t.owner_id}` : "他人共享" }) : null)),
          h("td", { class: "small", text: fmtTime(t.updated_at) }),
          h("td", null, h("button", { type: "button", class: "btn btn-small", text: "查看 / 编辑", onclick: () => openTemplate(t.id) })))))))
        : h("p", { class: "muted", text: "没有符合条件的模板。内置模板审核通过前，请先通过下方“LLM 辅助起草”或手动新建模板。" }),
      h("div", { class: "row" }, h("button", { type: "button", class: "btn", text: "新建空白模板", onclick: () => openNew() }))));
  }

  // ---------- detail / editor ----------
  async function openTemplate(id) {
    openId = id; paintList();
    mount(detailHost, spinner());
    let t;
    try { t = await api.getPrompt(id); } catch (e) { mount(detailHost, errorBox(e)); return; }
    if (!alive) return;
    paintDetail(t);
    detailHost.scrollIntoView && detailHost.scrollIntoView({ block: "nearest" });
  }

  function openNew(seed) {
    openId = null; paintList();
    paintDetail(null, seed);
  }

  function paintDetail(t, seed) {
    const isNew = !t;
    const versions = isNew ? [] : [...t.versions].sort((a, b) => b.version - a.version);
    let cur = isNew ? { text: (seed && seed.text) || "", input_mode: (seed && seed.input_mode) || "image", score_key: "判分", score_aliases: [], criteria_key: "判据", required_keys: ["判分", "判据"], max_chars: 20000 } : versions[0];
    const msg = h("div");
    const verSel = h("select", { "aria-label": "版本", onchange: () => { cur = versions.find((v) => String(v.version) === verSel.value); fillFrom(cur); setEditing(false); } },
      versions.map((v) => h("option", { value: String(v.version), text: `v${v.version}  ${fmtTime(v.created_at)}` })));
    const nameIn = h("input", { type: "text", maxlength: "80", value: isNew ? (seed && seed.name) || "" : `${t.name}（编辑）`, "aria-label": "新模板名称" });
    const modeSel = h("select", { "aria-label": "输入类型" }, h("option", { value: "text", text: MODE_LABEL.text }), h("option", { value: "image", text: MODE_LABEL.image }));
    const ta = h("textarea", { rows: "18", class: "mono prompt-text", "aria-label": "提示词正文", spellcheck: "false" });
    const counter = h("div", { class: "hint" });
    const keys = h("dl", { class: "kv" });
    let editing = isNew, asNewMode = false;
    const nameRow = isNew ? null : field("新模板名称", nameIn).row;
    if (nameRow) nameRow.hidden = true;

    const fillFrom = (v) => {
      ta.value = v.text; modeSel.value = v.input_mode || "image";
      mount(keys,
        ...[["判分字段", v.score_key], ["判据字段", v.criteria_key], ["必需字段", (v.required_keys || []).join("、")], ["分数别名", (v.score_aliases || []).join("、") || "（无）"], ["长度上限", v.max_chars]]
          .flatMap(([k, val]) => [h("dt", { text: k }), h("dd", { text: String(val ?? "") })]));
      updCount();
    };
    const updCount = () => {
      const n = cpLen(ta.value), max = Number(cur.max_chars) || 0;
      counter.textContent = `${n} / ${max} 字符` + (ta.value.includes("{{FEWSHOT}}") ? "" : "  ·  缺少 {{FEWSHOT}} 占位符");
      counter.className = "hint" + (max && n > max ? " bad" : "");
    };
    ta.addEventListener("input", updCount);

    const buttons = h("div", { class: "row" });
    function setEditing(on) {
      editing = on;
      ta.readOnly = !on;
      if (nameRow) nameRow.hidden = !(on && asNewMode);
      paintButtons();
    }

    function bodyFor(extra) {
      return { text: ta.value, input_mode: isNew ? modeSel.value : cur.input_mode, score_key: cur.score_key, score_aliases: cur.score_aliases || [], criteria_key: cur.criteria_key,
        required_keys: cur.required_keys || [cur.score_key, cur.criteria_key], max_chars: cur.max_chars, ...extra };
    }

    async function doSave(kind) {
      mount(msg);
      if (!ta.value.trim()) { mount(msg, errorBox({ message: "提示词正文不能为空", code: "form_incomplete" })); return; }
      if (kind === "new" && !nameIn.value.trim()) { mount(msg, errorBox({ message: "请填写新模板名称", code: "form_incomplete" })); return; }
      const send = (extra) => {
        if (kind === "version") return api.addPromptVersion(t.id, bodyFor(extra));
        const parent = isNew ? {} : { parent_id: t.id, parent_version: Number(verSel.value) };
        return api.createPrompt(bodyFor({ name: nameIn.value.trim(), ...parent, ...extra }));
      };
      let saved;
      try {
        try { saved = await send({}); }
        catch (e) {
          if (e.code === "fewshot_placeholder_missing") {
            const choice = await modal({ title: "缺少 {{FEWSHOT}} 占位符",
              body: "提示词里没有 {{FEWSHOT}}，历史范例就无法插入。是否在末尾自动追加？",
              buttons: [{ label: "取消保存", value: "cancel" }, { label: "自动追加 {{FEWSHOT}}", primary: true, value: "append" }] });
            if (choice !== "append") { mount(msg, errorBox({ message: "未保存：" + reasonLabel("fewshot_placeholder_missing"), code: e.code })); return; }
            saved = await send({ auto_append_fewshot: true });
          } else throw e;
        }
      } catch (e) {
        mount(msg, errorBox(e.code === "prompt_too_long" ? { ...e, message: `${e.message || reasonLabel(e.code)}（当前 ${cpLen(ta.value)} 字符，上限 ${cur.max_chars}）` } : e));
        return;
      }
      toast(kind === "version" ? "已保存为新版本" : "已保存为新模板", "ok");
      await loadList();
      await openTemplate((saved && saved.id) || (t && t.id));
    }

    function paintButtons() {
      const out = [];
      if (isNew) {
        out.push(h("button", { type: "button", class: "btn btn-primary", text: "保存为新模板", onclick: () => doSave("new") }));
      } else if (!editing) {
        out.push(h("button", { type: "button", class: "btn", text: canEditInPlace(t) ? "编辑（保存为新版本）" : "基于此版本另存为新模板", onclick: () => { asNewMode = !canEditInPlace(t); setEditing(true); } }));
        if (canEditInPlace(t)) out.push(h("button", { type: "button", class: "btn btn-small", text: "另存为新模板", onclick: () => { asNewMode = true; setEditing(true); } }));
      } else {
        if (canEditInPlace(t) && !asNewMode) out.push(h("button", { type: "button", class: "btn btn-primary", text: "保存为新版本", onclick: () => doSave("version") }));
        else out.push(h("button", { type: "button", class: "btn btn-primary", text: "保存为新模板", onclick: () => doSave("new") }));
        out.push(h("button", { type: "button", class: "btn", text: "放弃修改", onclick: () => { fillFrom(cur); asNewMode = false; setEditing(false); mount(msg); } }));
      }
      if (!isNew && ctx.isRoot && !t.builtin) {
        const box = h("input", { type: "checkbox", checked: !!t.shared, onchange: async () => {
          box.disabled = true;
          try { await api.sharePrompt(t.id, box.checked); toast(box.checked ? "已共享给所有人" : "已取消共享", "ok"); await loadList(); await openTemplate(t.id); }
          catch (e) { box.checked = !box.checked; box.disabled = false; mount(msg, errorBox(e)); }
        } });
        out.push(h("label", { class: "check small" }, box, "共享给所有人"));
      }
      if (!isNew && canEditInPlace(t)) {
        out.push(h("button", { type: "button", class: "btn btn-small btn-danger", text: "删除模板", onclick: async () => {
          if (!(await confirmModal("删除模板", `确定删除“${t.name}”吗？已创建的批次仍保留其使用的版本。`, "删除"))) return;
          try { await api.deletePrompt(t.id); toast("已删除", "ok"); openId = null; mount(detailHost); await loadList(); } catch (e) { mount(msg, errorBox(e)); }
        } }));
      }
      mount(buttons, out);
    }

    const modeRow = field("输入类型", modeSel).row; modeRow.hidden = !isNew;
    mount(detailHost, h("div", { class: "card" },
      h("div", { class: "batch-head" },
        h("h2", { text: isNew ? "新建模板" : t.name }),
        isNew ? null : h("div", { class: "row" }, t.builtin ? h("span", { class: "badge", text: "内置（只读）" }) : null, "版本", verSel)),
      !isNew && t.builtin ? h("p", { class: "muted small", text: "内置模板不能直接修改；可以基于它另存为你自己的新模板。" }) : null,
      isNew ? field("模板名称", nameIn).row : nameRow,
      modeRow,
      field("提示词正文", ta, "可用占位符：{{TOPIC}} {{RUBRIC}} {{GENRE}} {{FULL_SCORE}} {{FEWSHOT}}；正文必须包含 {{FEWSHOT}}").row,
      counter, keys, msg, buttons));
    fillFrom(cur);
    setEditing(isNew);
  }

  // ---------- LLM-assisted draft ----------
  function paintAssist() {
    const provSel = h("select", { "aria-label": "服务商" }, h("option", { value: "", text: "加载中…" }));
    const req = h("textarea", { rows: "5", placeholder: "描述你想要的提示词：例如“批改初二记叙文，按内容/结构/语言/书写四项给分，分项要有评语”", "aria-label": "需求描述" });
    const topic = h("input", { type: "text", maxlength: "500", placeholder: "（可选）作文题目，帮助模型起草", "aria-label": "题目" });
    const hist = h("input", { type: "checkbox" });
    const scope = h("select", { "aria-label": "历史范围" }, h("option", { value: "own", text: "仅我自己的批改记录" }), h("option", { value: "all", text: "所有人的批改记录" }));
    const scopeRow = field("历史范围", scope, "最多使用 5 条历史记录（题目、评分标准、转写、AI 结果与教师复核）；不包含照片和思考过程。").row;
    scopeRow.hidden = true;
    hist.addEventListener("change", () => { scopeRow.hidden = !hist.checked || !ctx.isRoot; });
    const msg = h("div"), out = h("div");
    let provs = [];
    api.listProviders().then((l) => {
      provs = l;
      mount(provSel, l.length ? l.map((p) => h("option", { value: p.id, text: `${p.alias}（${p.default_model || "?"}）${p.is_third_party ? " · 第三方" : ""}` })) : h("option", { value: "", text: "没有可用服务商，请先到“服务商”页添加" }));
    }).catch((e) => mount(msg, errorBox(e)));

    const go = h("button", { type: "button", class: "btn btn-primary", text: "生成草稿", onclick: async () => {
      mount(msg); mount(out);
      const p = provs.find((x) => x.id === provSel.value);
      if (!p) { mount(msg, errorBox({ message: "请先选择服务商", code: "form_incomplete" })); return; }
      if (!req.value.trim()) { mount(msg, errorBox({ message: "请填写需求描述", code: "form_incomplete" })); return; }
      go.disabled = true;
      try {
        const pre = await preflightAck(p, "需求描述" + (hist.checked ? "以及最多 5 条历史批改记录（含学生作文转写文字）" : "") + "会被发送到该主机。");
        if (!pre.ok) return;
        const body = { provider_id: p.id, requirement: req.value.trim(), include_history: hist.checked, history_scope: ctx.isRoot && hist.checked ? scope.value : "own" };
        if (topic.value.trim()) body.topic = topic.value.trim();
        if (pre.ack) body.third_party_ack = true;
        let res;
        try { res = await api.assistPrompt(body); }
        catch (e) {
          if (e.code === "third_party_ack_required" && (await confirmForProvider(p, "需求描述（及所选历史记录）会被发送到该主机。"))) res = await api.assistPrompt({ ...body, third_party_ack: true });
          else throw e;
        }
        paintDraft(String((res && res.draft) || ""));
      } catch (e) { mount(msg, errorBox(e)); }
      finally { go.disabled = false; }
    } });

    function paintDraft(text) {
      const ta = h("textarea", { rows: "16", class: "mono prompt-text", "aria-label": "草稿" }); ta.value = text;
      mount(out, h("div", { class: "card inner" },
        h("h3", { text: "草稿（请检查并修改后再保存，模型输出不保证可用）" }), ta,
        h("div", { class: "row" },
          h("button", { type: "button", class: "btn btn-primary", text: "用此草稿新建模板…", onclick: () => { openNew({ text: ta.value, name: "AI 草稿" }); detailHost.scrollIntoView && detailHost.scrollIntoView({ block: "start" }); } }))));
    }

    mount(assistHost, h("div", { class: "card" },
      h("h2", { text: "LLM 辅助起草" }),
      h("p", { class: "muted small", text: "让一个模型根据你的需求起草提示词。草稿不会自动保存，也不会自动用于批改。" }),
      field("服务商", provSel).row, field("需求", req).row, field("题目（可选）", topic).row,
      h("label", { class: "check" }, hist, "参考我的历史批改记录"), scopeRow, msg, h("div", { class: "row" }, go), out));
  }

  mount(root, h("section", { class: "page" }, h("h1", { text: "提示词模板" }), errHost, listHost, detailHost, assistHost));
  mount(listHost, spinner());
  loadList(); paintAssist();
  return () => { alive = false; };
}
