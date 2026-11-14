import { api, setCsrf, setUnauthorizedHandler } from "./api.js";
import { h, mount } from "./dom.js";
import { errorBox, spinner } from "./ui.js";
import { renderLogin } from "./auth.js";
import { renderUpload } from "./upload.js";
import { renderBatchList } from "./batches.js";
import { renderBatchDetail } from "./batch_detail.js";
import { renderProviders } from "./providers.js";
import { renderPrompts } from "./prompts.js";
import { renderTokens } from "./tokens.js";
import { renderAdmin } from "./admin.js";
import { renderSheets } from "./sheets.js";

const nav = document.getElementById("nav");
const root = document.getElementById("view");
const session = { user: null };
let cleanup = null;
let renderSeq = 0;

function go(hash) { if (location.hash === hash) route(); else location.hash = hash; }

export async function refreshSession() {
  try {
    const me = await api.me();
    session.user = me; setCsrf(me.csrf_token);
  } catch { session.user = null; }
  return session.user;
}

export function setUser(u, csrf) { session.user = u; setCsrf(csrf); }

setUnauthorizedHandler(() => {
  if (session.user) { session.user = null; renderNav(); }
  if (!location.hash.startsWith("#/login")) go("#/login");
});

const isRoot = () => session.user && session.user.role === "root";

function renderNav() {
  const cur = location.hash || "";
  const link = (hash, text) => h("a", { class: "nav-link", href: undefined, role: "link", tabindex: "0", "aria-current": cur.startsWith(hash) ? "page" : null, text,
    onclick: () => go(hash), onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(hash); } } });
  const u = session.user;
  mount(nav,
    h("span", { class: "brand", text: "作文批改" }),
    u ? h("div", { class: "nav-links" }, link("#/upload", "上传"), link("#/batches", "批次"), link("#/providers", "服务商"), link("#/prompts", "提示词"), link("#/sheets", "答题纸"), link("#/tokens", "API 令牌"), isRoot() ? link("#/admin", "管理") : null) : null,
    u ? h("div", { class: "nav-user" },
      h("span", { class: "who", text: `${u.username}（${u.role === "root" ? "管理员" : "用户"}）` }),
      h("button", { type: "button", class: "btn btn-small", text: "退出登录", onclick: async () => {
        try { await api.logout(); } catch { /* cookie may already be gone */ }
        session.user = null; renderNav(); go("#/login");
      } })) : null);
}

async function route() {
  const seq = ++renderSeq;
  if (cleanup) { try { cleanup(); } catch { /* ignore */ } cleanup = null; }
  const hash = location.hash || "#/";
  if (!session.user) {
    mount(root, spinner());
    await refreshSession();
    if (seq !== renderSeq) return;
  }
  renderNav();
  const parts = hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  const page = parts[0] || "";
  if (!session.user) {
    if (page !== "login") return go("#/login");
    cleanup = renderLogin(root, { onLoggedIn: (user, csrf) => { setUser(user, csrf); renderNav(); go("#/upload"); } }) || null;
    return;
  }
  if (page === "login" || page === "") return go("#/upload");
  const ctx = { user: session.user, go, isRoot: isRoot() };
  try {
    if (page === "upload") cleanup = renderUpload(root, ctx);
    else if (page === "batches" && parts[1]) cleanup = renderBatchDetail(root, ctx, parts[1]);
    else if (page === "batches") cleanup = renderBatchList(root, ctx);
    else if (page === "providers") cleanup = renderProviders(root, ctx);
    else if (page === "prompts") cleanup = renderPrompts(root, ctx);
    else if (page === "tokens") cleanup = renderTokens(root, ctx);
    else if (page === "sheets") cleanup = renderSheets(root, ctx, parts[1]);
    else if (page === "admin") {
      if (!isRoot()) return mount(root, h("div", { class: "alert alert-error", role: "alert", text: "仅管理员可访问此页面" }));
      cleanup = renderAdmin(root, ctx);
    } else mount(root, h("div", { class: "alert alert-error", role: "alert", text: "页面不存在" }));
  } catch (e) {
    mount(root, errorBox({ message: "页面加载失败：" + (e && e.message), code: "ui_error" }));
  }
}

window.addEventListener("hashchange", route);
route();
