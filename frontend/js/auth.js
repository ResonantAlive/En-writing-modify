import { api } from "./api.js";
import { h, mount, field } from "./dom.js";
import { errorBox } from "./ui.js";

export function renderLogin(root, { onLoggedIn }) {
  let tab = "login";
  const errHost = h("div");
  const body = h("div");
  const tabBtn = (id, text) => h("button", { type: "button", role: "tab", class: "tab", text, "aria-selected": String(tab === id), onclick: () => { tab = id; draw(); } });

  function draw() {
    mount(errHost);
    const username = h("input", { type: "text", autocomplete: "username", required: true, maxlength: "64" });
    const password = h("input", { type: "password", autocomplete: tab === "login" ? "current-password" : "new-password", required: true });
    const invite = h("input", { type: "text", autocomplete: "off", required: true });
    const submit = h("button", { type: "submit", class: "btn btn-primary", text: tab === "login" ? "登录" : "注册并登录" });
    const form = h("form", { class: "card form-narrow", novalidate: true,
      onsubmit: async (e) => {
        e.preventDefault(); mount(errHost);
        if (!username.value.trim() || !password.value || (tab === "register" && !invite.value.trim())) {
          mount(errHost, errorBox({ message: "请填写所有必填项" })); return;
        }
        submit.disabled = true;
        try {
          if (tab === "register") await api.register(invite.value.trim(), username.value.trim(), password.value);
          const res = await api.login(username.value.trim(), password.value);
          onLoggedIn(res.user, res.csrf_token);
        } catch (err) { mount(errHost, errorBox(err)); submit.disabled = false; }
      } },
      field("用户名", username).row, field("密码", password).row,
      tab === "register" ? field("邀请码", invite, "邀请码由管理员提供").row : null,
      h("div", { class: "actions" }, submit));
    mount(body,
      h("div", { class: "tabs", role: "tablist" }, tabBtn("login", "登录"), tabBtn("register", "注册")),
      errHost, form);
    username.focus();
  }
  draw();
  mount(root, h("section", { class: "page page-narrow" }, h("h1", { text: "欢迎使用" }), body));
}
