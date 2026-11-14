// Pure model for rendering grading results recursively BY KEY NAME (no DOM here).
// buildNode(value, {scoreKey, criteriaKey}) -> node tree:
//   {kind:"object", entries:[{key, role:"score"|"criteria"|null, node}]}
//   {kind:"array", items:[node]} | {kind:"string"|"number"|"boolean"|"null", value} | {kind:"truncated"}
// Roles are only assigned on top-level (depth 0) keys.

export const MAX_DEPTH = 12;
export const MAX_ITEMS = 500;

export function buildNode(value, opts = {}, depth = 0) {
  if (value === null || value === undefined) return { kind: "null" };
  if (depth > MAX_DEPTH) return { kind: "truncated" };
  if (Array.isArray(value)) {
    return { kind: "array", items: value.slice(0, MAX_ITEMS).map((v) => buildNode(v, opts, depth + 1)), omitted: Math.max(0, value.length - MAX_ITEMS) };
  }
  switch (typeof value) {
    case "object": {
      const keys = Object.keys(value);
      const entries = keys.slice(0, MAX_ITEMS).map((key) => ({
        key,
        role: depth === 0 ? (key === opts.scoreKey ? "score" : key === opts.criteriaKey ? "criteria" : null) : null,
        node: buildNode(value[key], opts, depth + 1),
      }));
      // score first, criteria second, rest in original order (stable)
      const rank = (e) => (e.role === "score" ? 0 : e.role === "criteria" ? 1 : 2);
      entries.sort((a, b) => rank(a) - rank(b));
      return { kind: "object", entries, omitted: Math.max(0, keys.length - MAX_ITEMS) };
    }
    case "string": return { kind: "string", value };
    case "number": return { kind: "number", value };
    case "boolean": return { kind: "boolean", value };
    default: return { kind: "string", value: String(value) };
  }
}

/** Code-point length (not UTF-16 units). */
export const cpLen = (s) => [...String(s ?? "")].length;

/** Warnings that need prominent teacher attention. */
export function prominentWarnings(warnings) {
  const list = Array.isArray(warnings) ? warnings.map(String) : [];
  return { regexFallback: list.includes("regex_fallback"), others: list.filter((w) => w !== "regex_fallback") };
}

/** Summary of an images-as-refs request: replace any image payload with a ref description. Never returns data URIs. */
export function describeContentPart(part) {
  if (!part || typeof part !== "object") return { kind: "text", text: String(part ?? "") };
  if (part.type === "image_ref") return { kind: "image_ref", text: `[图片引用] ${part.path || ""}${part.page ? `（${part.page}）` : ""}${part.sha256 ? ` sha256=${String(part.sha256).slice(0, 12)}…` : ""}` };
  if (part.type === "image_url") return { kind: "image_ref", text: "[图片已省略]" };
  if (part.type === "text") return { kind: "text", text: String(part.text ?? "") };
  return { kind: "text", text: JSON.stringify(part) };
}

export function describeMessages(messages) {
  return (Array.isArray(messages) ? messages : []).map((m) => ({
    role: String((m && m.role) || "?"),
    parts: Array.isArray(m && m.content) ? m.content.map(describeContentPart) : [describeContentPart({ type: "text", text: m && m.content })],
  }));
}

/** Pick the latest template version number from a template record. */
export function latestVersion(tpl) {
  const vs = (tpl && tpl.versions) || [];
  return vs.reduce((m, v) => Math.max(m, Number(v.version) || 0), 0);
}

/** Extract host for display from a base_url; falls back to the raw string. */
export function hostOf(url) {
  try { return new URL(url).host; } catch { return String(url || ""); }
}
