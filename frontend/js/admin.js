import { api } from "./api.js";
import { h, mount, fmtTime } from "./dom.js";
import { errorBox, spinner, toast, confirmModal, infoBox } from "./ui.js";
import { tokenTable } from "./tokens.js";

export function renderAdmin(root, ctx) {
  const msgHost = h("div"), usersHost = h("div"), tokensHost = h("div"), inviteHost = h("div");

  async function loadUsers() {
    mount(usersHost, spinner());
    try {
      const users = await api.adminUsers();
      mount(usersHost, h("div", { class: "table-wrap" }, h("table", { class: "table" },
        h("thead", null, h("tr", null, ["用户名", "角色", "状态", "创建时间", "操作"].map((t) => h("th", { text: t })))),
        h("tbody", null, users.map((u) => {
          const self = u.id === ctx.user.id;
          const act = async (fn, okMsg) => { mount(msgHost); try { await fn(); toast(okMsg); loadUsers(); } catch (e) { mount(msgHost, errorBox(e)); } };
          const roleSel = h("select", { "aria-label": `${u.username} 的角色`, disabled: self, onchange: async () => {
            const nr = roleSel.value;
            if (!(await confirmModal("修改角色", `将 ${u.username} 的角色改为 ${nr}？`))) { roleSel.value = u.role; return; }
            act(() => api.adminRole(u.id, nr), "角色已更新");
          } }, h("option", { value: "user", text: "user", selected: u.role === "user" }), h("option", { value: "root", text: "root", selected: u.role === "root" }));
          roleSel.value = u.role;
          return h("tr", null, h("td", { text: u.username + (self ? "（我）" : "") }), h("td", null, roleSel),
            h("td", null, h("span", { class: "badge " + (u.disabled ? "badge-bad" : "badge-ok"), text: u.disabled ? "已禁用" : "正常" })),
            h("td", { text: fmtTime(u.created_at) }),
            h("td", null, u.disabled
              ? h("button", { type: "button", class: "btn btn-small", text: "启用", onclick: () => act(() => api.adminEnable(u.id), "已启用") })
              : h("button", { type: "button", class: "btn btn-small btn-danger", text: "禁用", disabled: self, onclick: async () => {
                if (await confirmModal("禁用账号", `禁用 ${u.username} 后其运行中的批次将被暂停，确定吗？`, "禁用")) act(() => api.adminDisable(u.id), "已禁用");
              } })));
        })))));
    } catch (e) { mount(usersHost, errorBox(e)); }
  }

  async function loadTokens() {
    mount(tokensHost, spinner());
    try {
      const toks = await api.adminTokens();
      mount(tokensHost, tokenTable(toks, { showOwner: true, onRevoke: async (t) => {
        if (!(await confirmModal("吊销令牌", `吊销 ${t.owner_username || ""} 的令牌“${t.name}”？`, "吊销"))) return;
        try { await api.adminRevokeToken(t.id); toast("已吊销"); loadTokens(); } catch (e) { mount(msgHost, errorBox(e)); }
      } }));
    } catch (e) { mount(tokensHost, errorBox(e)); }
  }

  const which = h("select", { "aria-label": "重置范围" },
    h("option", { value: "both", text: "两种邀请码" }), h("option", { value: "user", text: "仅普通用户邀请码" }), h("option", { value: "root", text: "仅管理员邀请码" }));
  mount(inviteHost, h("div", { class: "row" }, which, h("button", { type: "button", class: "btn btn-danger", text: "重置邀请码", onclick: async () => {
    if (!(await confirmModal("重置邀请码", "旧邀请码将立即失效，确定重置吗？", "重置"))) return;
    mount(msgHost);
    try { await api.adminResetInvites(which.value); mount(msgHost, infoBox("新邀请码已写入服务器文件，请到服务器查看", "info")); }
    catch (e) { mount(msgHost, errorBox(e)); }
  } })));

  mount(root, h("section", { class: "page" }, h("h1", { text: "管理" }), msgHost,
    h("div", { class: "card" }, h("h2", { text: "用户" }), usersHost),
    h("div", { class: "card" }, h("h2", { text: "邀请码" }), inviteHost),
    h("div", { class: "card" }, h("h2", { text: "所有令牌" }), tokensHost)));
  loadUsers(); loadTokens();
}
