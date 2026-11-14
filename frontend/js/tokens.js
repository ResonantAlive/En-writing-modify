import { api } from "./api.js";
import { h, mount, field, fmtTime } from "./dom.js";
import { errorBox, spinner, toast, confirmModal } from "./ui.js";

export function tokenTable(tokens, { onRevoke, showOwner }) {
  if (!tokens.length) return h("p", { class: "muted", text: "暂无令牌" });
  return h("div", { class: "table-wrap" }, h("table", { class: "table" },
    h("thead", null, h("tr", null, ["名称", showOwner ? "所有者" : null, "角色", "尾号", "创建时间", "过期时间", "最近使用", "状态", ""].filter((x) => x !== null).map((t) => h("th", { text: t })))),
    h("tbody", null, tokens.map((t) => h("tr", null,
      h("td", { text: t.name }),
      showOwner ? h("td", { text: t.owner_username || t.owner_id || "" }) : null,
      h("td", { text: t.role }), h("td", { text: "…" + (t.last4 || "") }),
      h("td", { text: fmtTime(t.created_at) }), h("td", { text: t.expires_at ? fmtTime(t.expires_at) : "永不" }),
      h("td", { text: t.last_used ? fmtTime(t.last_used) : "从未" }),
      h("td", null, h("span", { class: "badge " + (t.revoked ? "badge-bad" : "badge-ok"), text: t.revoked ? "已吊销" : "有效" })),
      h("td", null, t.revoked ? null : h("button", { type: "button", class: "btn btn-small btn-danger", text: "吊销", onclick: () => onRevoke(t) })))))));
}

export function renderTokens(root, ctx) {
  const listHost = h("div"), msgHost = h("div"), createdHost = h("div");
  async function load() {
    mount(listHost, spinner());
    try {
      const tokens = await api.listTokens();
      mount(listHost, tokenTable(tokens, { onRevoke: async (t) => {
        if (!(await confirmModal("吊销令牌", `确定吊销令牌“${t.name}”吗？使用它的程序将立即失效。`, "吊销"))) return;
        try { await api.revokeToken(t.id); toast("已吊销"); load(); } catch (e) { mount(msgHost, errorBox(e)); }
      } }));
    } catch (e) { mount(listHost, errorBox(e)); }
  }

  const name = h("input", { type: "text", maxlength: "100", required: true });
  const role = h("select", null, h("option", { value: "user", text: "user（普通）" }), ctx.isRoot ? h("option", { value: "root", text: "root（管理员）" }) : null);
  const days = h("input", { type: "number", min: "0.1", step: "any", placeholder: "留空表示永不过期" });
  const form = h("form", { class: "card", novalidate: true, onsubmit: async (e) => {
    e.preventDefault(); mount(msgHost); mount(createdHost);
    if (!name.value.trim()) { mount(msgHost, errorBox({ message: "请填写令牌名称" })); return; }
    const body = { name: name.value.trim(), role: role.value };
    if (days.value.trim() !== "") {
      const d = Number(days.value);
      if (!(d > 0)) { mount(msgHost, errorBox({ message: "有效天数必须大于 0" })); return; }
      body.expires_in_days = d;
    }
    try {
      const rec = await api.createToken(body);
      name.value = ""; days.value = "";
      const out = h("input", { type: "text", readonly: true, class: "mono", value: rec.token, "aria-label": "新令牌明文" });
      mount(createdHost, h("div", { class: "alert alert-warn", role: "status" },
        h("strong", { text: "请立即复制令牌，关闭页面后将无法再次查看。" }),
        h("div", { class: "row" }, out,
          h("button", { type: "button", class: "btn", text: "复制", onclick: async () => {
            try { await navigator.clipboard.writeText(rec.token); toast("已复制"); } catch { out.select(); toast("请按 Ctrl+C 复制", "info"); }
          } }))));
      out.select();
      load();
    } catch (err) { mount(msgHost, errorBox(err)); }
  } },
  h("h2", { text: "创建令牌" }),
  field("名称", name).row, field("角色", role).row, field("有效天数（可选）", days).row,
  h("div", { class: "actions" }, h("button", { type: "submit", class: "btn btn-primary", text: "创建" })));

  mount(root, h("section", { class: "page" }, h("h1", { text: "API 令牌" }), form, msgHost, createdHost,
    h("div", { class: "card" }, h("h2", { text: "我的令牌" }), listHost)));
  load();
}
