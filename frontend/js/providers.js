// #/providers : model providers (alias, base URL, key, default model). Keys are write-only; only last4 is ever shown.
import { api } from "./api.js";
import { h, mount, field } from "./dom.js";
import { errorBox, spinner, toast, confirmModal } from "./ui.js";
import { hostOf } from "./render_model.mjs";

export function renderProviders(root, ctx) {
  const errHost = h("div"), listHost = h("div"), formHost = h("div");
  let alive = true, items = [];

  const mine = (p) => p.owner_id === ctx.user.id;
  const canEdit = (p) => mine(p) || ctx.isRoot;

  async function load() {
    mount(errHost);
    try { items = await api.listProviders(); } catch (e) { mount(listHost); mount(errHost, errorBox(e)); return; }
    if (alive) paintList();
  }

  function paintList() {
    if (!items.length) { mount(listHost, h("div", { class: "card empty" }, h("p", { text: "还没有服务商。点击“添加服务商”填写网址、密钥和默认模型。" }))); return; }
    mount(listHost, h("div", { class: "grid-cards" }, items.map((p) => {
      const editable = canEdit(p);
      return h("article", { class: "card" },
        h("div", { class: "batch-head" },
          h("h3", { text: p.alias }),
          h("div", { class: "row" },
            p.is_third_party ? h("span", { class: "badge badge-warn", text: "第三方" }) : h("span", { class: "badge", text: "本机" }),
            p.shared ? h("span", { class: "badge badge-info", text: "已共享" }) : null,
            !mine(p) ? h("span", { class: "badge", text: ctx.isRoot ? `用户 ${p.owner_id}` : "他人共享（只读）" }) : null)),
        h("dl", { class: "kv" },
          h("dt", { text: "主机" }), h("dd", { class: "mono", text: hostOf(p.base_url) }),
          h("dt", { text: "接口地址" }), h("dd", { class: "mono", text: p.base_url }),
          h("dt", { text: "密钥" }), h("dd", { class: "mono", text: p.key_last4 ? `••••${p.key_last4}` : "（无）" }),
          h("dt", { text: "默认模型" }), h("dd", { text: p.default_model || "" })),
        h("div", { class: "row" },
          editable ? h("button", { type: "button", class: "btn btn-small", text: "编辑", onclick: () => openForm(p) }) : null,
          ctx.isRoot && editable ? shareToggle(p) : null,
          editable ? h("button", { type: "button", class: "btn btn-small btn-danger", text: "删除", onclick: () => remove(p) }) : null));
    })));
  }

  function shareToggle(p) {
    const box = h("input", { type: "checkbox", checked: !!p.shared, onchange: async () => {
      box.disabled = true;
      try { await api.updateProvider(p.id, { shared: box.checked }); toast(box.checked ? "已共享给所有用户" : "已取消共享", "ok"); await load(); }
      catch (e) { box.checked = !box.checked; box.disabled = false; mount(errHost, errorBox(e)); }
    } });
    return h("label", { class: "check small" }, box, "共享给所有人（共享后其他用户可使用但看不到密钥）");
  }

  async function remove(p) {
    if (!(await confirmModal("删除服务商", `确定删除“${p.alias}”吗？已创建的批次不受影响，但无法再选择它。`, "删除"))) return;
    try { await api.deleteProvider(p.id); toast("已删除", "ok"); await load(); } catch (e) { mount(errHost, errorBox(e)); }
  }

  function openForm(p) {
    const editing = !!p;
    const alias = h("input", { type: "text", maxlength: "60", value: editing ? p.alias : "" });
    const base = h("input", { type: "text", maxlength: "300", placeholder: "例如 https://api.example.com/v1", value: editing ? p.base_url : "", autocomplete: "off" });
    const key = h("input", { type: "password", autocomplete: "off", placeholder: editing ? `留空表示不修改（当前以 ${p.key_last4 || "…"} 结尾）` : "API Key" });
    const model = h("input", { type: "text", maxlength: "120", value: editing ? p.default_model || "" : "" });
    const shared = ctx.isRoot ? h("input", { type: "checkbox", checked: editing && !!p.shared }) : null;
    const msg = h("div");
    const thirdNote = h("div", { class: "alert alert-warn", hidden: true, role: "status" });
    const syncNote = () => {
      const v = base.value.trim();
      const local = /^https?:\/\/(localhost|127\.\d+\.\d+\.\d+|\[::1\])([:/]|$)/i.test(v);
      thirdNote.hidden = !v || local;
      thirdNote.textContent = thirdNote.hidden ? "" : `该地址（${hostOf(v)}）会被视为第三方服务：首次使用时会要求你确认作文内容将发送到该主机。`;
    };
    base.addEventListener("input", syncNote); syncNote();
    const save = h("button", { type: "button", class: "btn btn-primary", text: "保存", onclick: async () => {
      mount(msg);
      const body = { alias: alias.value.trim(), base_url: base.value.trim(), default_model: model.value.trim() };
      if (key.value) body.api_key = key.value;
      if (shared) body.shared = shared.checked;
      const miss = [];
      if (!body.alias) miss.push("别名"); if (!body.base_url) miss.push("接口地址"); if (!body.default_model) miss.push("默认模型");
      if (!editing && !body.api_key) miss.push("API Key");
      if (miss.length) { mount(msg, errorBox({ message: "请补全：" + miss.join("、"), code: "form_incomplete" })); return; }
      save.disabled = true;
      try {
        if (editing) await api.updateProvider(p.id, body); else await api.createProvider(body);
        key.value = "";
        toast("已保存", "ok"); mount(formHost); await load();
      } catch (e) { mount(msg, errorBox(e)); save.disabled = false; }
    } });
    mount(formHost, h("form", { class: "card", novalidate: true, onsubmit: (e) => e.preventDefault() },
      h("h2", { text: editing ? `编辑服务商：${p.alias}` : "添加服务商" }),
      field("别名", alias).row, field("接口地址（base URL）", base, "会在其后追加 /chat/completions").row, thirdNote,
      field("API Key", key, "仅保存在服务器上，之后只会显示末尾 4 位").row,
      field("默认模型", model).row,
      shared ? h("label", { class: "check" }, shared, "共享给所有人（仅管理员）") : null,
      msg,
      h("div", { class: "row" }, save, h("button", { type: "button", class: "btn", text: "取消", onclick: () => mount(formHost) }))));
    alias.focus();
  }

  mount(root, h("section", { class: "page" },
    h("div", { class: "row between" }, h("h1", { text: "服务商" }), h("button", { type: "button", class: "btn btn-primary", text: "添加服务商", onclick: () => openForm(null) })),
    h("p", { class: "muted small", text: "服务商是批改所用的大模型接口（OpenAI 兼容）。密钥不会再次显示。" }),
    errHost, formHost, listHost));
  mount(listHost, spinner());
  load();
  return () => { alive = false; };
}
