// Single gateway for every network call: request() (JSON / multipart / blobless).
// Errors are normalised to ApiError {status, code, message, detail}; messages are plain text.

const BASE = "/api/v1";
let csrfFallback = null;
let onUnauthorized = null;

export class ApiError extends Error {
  constructor(status, code, message, detail) {
    super(message || code || "error");
    this.status = status; this.code = code || "error"; this.detail = detail;
  }
}

export function setCsrf(t) { if (t) csrfFallback = t; }
export function setUnauthorizedHandler(fn) { onUnauthorized = fn; }

function csrfToken() {
  const m = document.cookie.split("; ").find((c) => c.startsWith("wm_csrf="));
  if (m) { try { return decodeURIComponent(m.slice(8)); } catch { return m.slice(8); } }
  return csrfFallback;
}

function normaliseError(status, body) {
  if (body && typeof body === "object") {
    if (typeof body.code === "string") return new ApiError(status, body.code, String(body.message || body.code), body.detail);
    if (Array.isArray(body.detail)) { // framework validation errors
      const msg = body.detail.map((d) => (d && d.msg ? `${(d.loc || []).slice(1).join(".")}: ${d.msg}` : String(d))).join("; ");
      return new ApiError(status, "validation_error", msg, body.detail);
    }
    if (typeof body.detail === "string") return new ApiError(status, "error", body.detail);
  }
  return new ApiError(status, "http_" + status, status ? `请求失败（HTTP ${status}）` : "无法连接服务器", null);
}

/**
 * request(method, path, {json, form, query, signal, raw})
 * path is relative to /api/v1 (e.g. "/auth/me").
 */
export async function request(method, path, opts = {}) {
  const headers = { Accept: "application/json" };
  const init = { method, credentials: "same-origin", headers, signal: opts.signal };
  if (method !== "GET" && method !== "HEAD") {
    const t = csrfToken();
    if (t) headers["X-CSRF-Token"] = t;
  }
  if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; init.body = JSON.stringify(opts.json); }
  else if (opts.form) init.body = opts.form;
  let url = BASE + path;
  if (opts.query) {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(opts.query)) if (v != null) q.set(k, String(v));
    const s = q.toString();
    if (s) url += "?" + s;
  }
  let res;
  try { res = await fetch(url, init); }
  catch (e) {
    if (e && e.name === "AbortError") throw e;
    throw new ApiError(0, "network_error", "无法连接服务器，请检查网络后重试", null);
  }
  let body = null;
  const text = await res.text();
  if (text) { try { body = JSON.parse(text); } catch { body = null; } }
  if (!res.ok) {
    const err = normaliseError(res.status, body);
    if (res.status === 401 && !opts.noAuthRedirect && onUnauthorized) onUnauthorized(err);
    throw err;
  }
  return body;
}

const get = (p, o) => request("GET", p, o);
const post = (p, json, o) => request("POST", p, { ...o, json: json === undefined ? {} : json });
const del = (p, o) => request("DELETE", p, o);

export const api = {
  // auth
  me: () => get("/auth/me", { noAuthRedirect: true }),
  login: (username, password) => request("POST", "/auth/login", { json: { username, password }, noAuthRedirect: true }),
  register: (invite_code, username, password) => request("POST", "/auth/register", { json: { invite_code, username, password }, noAuthRedirect: true }),
  logout: () => post("/auth/logout"),
  // tokens
  listTokens: () => get("/tokens"),
  createToken: (body) => post("/tokens", body),
  revokeToken: (id) => del("/tokens/" + encodeURIComponent(id)),
  // admin
  adminUsers: () => get("/admin/users"),
  adminDisable: (id) => post(`/admin/users/${encodeURIComponent(id)}/disable`),
  adminEnable: (id) => post(`/admin/users/${encodeURIComponent(id)}/enable`),
  adminRole: (id, role) => post(`/admin/users/${encodeURIComponent(id)}/role`, { role }),
  adminResetInvites: (which) => post("/admin/invite-codes/reset", { which }),
  adminTokens: () => get("/admin/tokens"),
  adminRevokeToken: (id) => del("/admin/tokens/" + encodeURIComponent(id)),
  // uploads
  limits: () => get("/uploads/limits"),
  uploadFiles: (files, lastModifieds, uploadId, signal) => {
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    for (const lm of lastModifieds) form.append("client_last_modified", lm == null ? "" : String(lm));
    if (uploadId) form.append("upload_id", uploadId);
    return request("POST", "/uploads", { form, signal });
  },
  getUpload: (id) => get("/uploads/" + encodeURIComponent(id)),
  deleteUploadFile: (id, fid) => del(`/uploads/${encodeURIComponent(id)}/files/${encodeURIComponent(fid)}`),
  deleteUpload: (id) => del("/uploads/" + encodeURIComponent(id)),
  groupingPreview: (id, req, signal) => request("POST", `/uploads/${encodeURIComponent(id)}/grouping/preview`, { json: req, signal }),
  // batches
  precheck: (body) => post("/batches/precheck", body),
  createBatch: (body) => post("/batches", body),
  listBatches: (all) => get("/batches", { query: all ? { all: "true" } : undefined }),
  getBatch: (code) => get("/batches/" + encodeURIComponent(code)),
  subtasks: (code) => get(`/batches/${encodeURIComponent(code)}/subtasks`),
  batchAction: (code, action) => post(`/batches/${encodeURIComponent(code)}/${action}`),
  batchFailures: (code) => get(`/batches/${encodeURIComponent(code)}/failures`),
  batchConfig: (code, body) => post(`/batches/${encodeURIComponent(code)}/config`, body),
  subtaskDetail: (code, sid) => get(`/batches/${encodeURIComponent(code)}/subtasks/${encodeURIComponent(sid)}`),
  attemptDetail: (code, sid, n) => get(`/batches/${encodeURIComponent(code)}/subtasks/${encodeURIComponent(sid)}/attempts/${Number(n)}`),
  regrade: (code, sid) => post(`/batches/${encodeURIComponent(code)}/subtasks/${encodeURIComponent(sid)}/regrade`),
  getReview: (code, sid) => get(`/batches/${encodeURIComponent(code)}/subtasks/${encodeURIComponent(sid)}/review`),
  submitReview: (code, sid, body) => post(`/batches/${encodeURIComponent(code)}/subtasks/${encodeURIComponent(sid)}/review`, body),
  getResult: (code) => get(`/batches/${encodeURIComponent(code)}/result`),
  regenerateResult: (code) => post(`/batches/${encodeURIComponent(code)}/result/regenerate`),
  // providers
  listProviders: () => get("/providers"),
  createProvider: (body) => post("/providers", body),
  updateProvider: (id, body) => request("PATCH", "/providers/" + encodeURIComponent(id), { json: body }),
  deleteProvider: (id) => del("/providers/" + encodeURIComponent(id)),
  ackProvider: (id) => post(`/providers/${encodeURIComponent(id)}/ack`),
  ackStatus: (id) => get(`/providers/${encodeURIComponent(id)}/ack-status`),
  // prompt templates
  listPrompts: (inputMode) => get("/prompts", { query: inputMode ? { input_mode: inputMode } : undefined }),
  getPrompt: (id) => get("/prompts/" + encodeURIComponent(id)),
  createPrompt: (body) => post("/prompts", body),
  addPromptVersion: (id, body) => post(`/prompts/${encodeURIComponent(id)}/versions`, body),
  sharePrompt: (id, shared) => post(`/prompts/${encodeURIComponent(id)}/share`, { shared: !!shared }),
  deletePrompt: (id) => del("/prompts/" + encodeURIComponent(id)),
  assistPrompt: (body) => post("/prompts/assist", body),
  // answer-sheet templates (blank sheets whose printed text is removed from local OCR)
  listSheets: () => get("/sheet-templates"),
  getSheet: (id) => get("/sheet-templates/" + encodeURIComponent(id)),
  createSheet: (name) => post("/sheet-templates", { name }),
  renameSheet: (id, name) => request("PATCH", "/sheet-templates/" + encodeURIComponent(id), { json: { name } }),
  shareSheet: (id, shared) => post(`/sheet-templates/${encodeURIComponent(id)}/share`, { shared: !!shared }),
  uploadSheetPages: (id, files) => {
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    return request("POST", `/sheet-templates/${encodeURIComponent(id)}/pages`, { form });
  },
  setSheetPage: (id, pid, disabled) => request("PATCH", `/sheet-templates/${encodeURIComponent(id)}/pages/${encodeURIComponent(pid)}`, { json: { disabled: !!disabled } }),
  retrySheetPage: (id, pid) => post(`/sheet-templates/${encodeURIComponent(id)}/pages/${encodeURIComponent(pid)}/retry`),
};

export const sheetPageImageUrl = (id, pid) =>
  `${BASE}/sheet-templates/${encodeURIComponent(id)}/pages/${encodeURIComponent(pid)}/image`;
export const batchEventsUrl = (code) => `${BASE}/batches/${encodeURIComponent(code)}/events`;
export const subtaskImageUrl = (code, sid, k, thumb) =>
  `${BASE}/batches/${encodeURIComponent(code)}/subtasks/${encodeURIComponent(sid)}/image/${Number(k)}${thumb ? "/thumb" : ""}`;
