// Safe DOM helpers. Dynamic data is only ever applied via textContent / setAttribute of vetted attributes.
// There is deliberately no HTML-string API here.

const SAFE_URL = /^(\/api\/[A-Za-z0-9_\-./?=&%]*|blob:[^\s]*)$/;

export function safeUrl(u) {
  return typeof u === "string" && SAFE_URL.test(u) && !u.includes("..") ? u : "";
}

/**
 * h("div", {class:"x", onclick: fn, dataset:{a:1}, aria:{label:"y"}}, child, "text", ...)
 * Strings become text nodes. null/false/undefined children are skipped. Arrays are flattened.
 */
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = String(v);
      else if (k === "dataset") for (const [dk, dv] of Object.entries(v)) el.dataset[dk] = String(dv);
      else if (k === "aria") for (const [ak, av] of Object.entries(v)) el.setAttribute("aria-" + ak, String(av));
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "src") { const s = safeUrl(v); if (s) el.setAttribute("src", s); }
      else if (k === "href") { /* never set hrefs from data */ }
      else if (k === "value") el.value = v;
      else if (k === "checked" || k === "disabled" || k === "hidden" || k === "required" || k === "selected" || k === "multiple") el[k] = !!v;
      else if (v === true) el.setAttribute(k, "");
      else el.setAttribute(k, String(v));
    }
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children) {
    if (c == null || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else if (c instanceof Node) el.appendChild(c);
    else el.appendChild(document.createTextNode(String(c)));
  }
}

export function clear(el) { el.replaceChildren(); return el; }
export function mount(el, ...children) { el.replaceChildren(); append(el, children); return el; }

export function fmtTime(s) {
  if (!s) return "";
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? String(s) : d.toLocaleString("zh-CN", { hour12: false });
}

export function fmtSize(n) {
  if (!Number.isFinite(n)) return "";
  if (n < 1024) return n + " B";
  if (n < 1048576) return (n / 1024).toFixed(0) + " KB";
  return (n / 1048576).toFixed(1) + " MB";
}

export function debounce(fn, ms) {
  let t = null;
  const d = (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  d.cancel = () => clearTimeout(t);
  return d;
}

/** Labelled form field row: returns {row, input}. */
export function field(labelText, input, hint) {
  const id = "f" + Math.random().toString(36).slice(2, 9);
  input.id = id;
  const row = h("div", { class: "field" }, h("label", { for: id, text: labelText }), input, hint ? h("div", { class: "hint", text: hint }) : null);
  return { row, input };
}
