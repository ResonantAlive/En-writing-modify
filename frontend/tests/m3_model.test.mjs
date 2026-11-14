import assert from "node:assert/strict";
import { reasonLabel, pauseExplanation, REASON_LABEL } from "../js/labels.js";
import { buildNode, cpLen, prominentWarnings, describeMessages, latestVersion, hostOf } from "../js/render_model.mjs";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

let n = 0;
const t = (name, fn) => { try { fn(); n++; } catch (e) { console.error("FAIL:", name); throw e; } };

t("labels: known and fallback", () => {
  assert.equal(reasonLabel("json_parse_failed"), REASON_LABEL.json_parse_failed);
  assert.equal(reasonLabel("missing_values"), REASON_LABEL.missing_required_field);
  for (const c of ["request_rejected", "api_error", "ocr_failed", "refusal_or_safety", "empty_response", "score_out_of_range"]) assert.notEqual(reasonLabel(c), c);
  assert.equal(reasonLabel("totally_new_code"), "totally_new_code");
  assert.equal(reasonLabel(""), ""); assert.equal(reasonLabel(null), "");
  assert.equal(reasonLabel("constructor"), "constructor"); // no prototype leakage
  assert.equal(reasonLabel("__proto__"), "__proto__");
});

t("pause explanation text", () => {
  assert.ok(pauseExplanation("paused_circuit_breaker", "json_parse_failed").includes("system prompt 构建、模型选择或本地识别可能有问题"));
  assert.ok(pauseExplanation("paused_api_errors").includes("接口连续出错"));
  assert.equal(pauseExplanation("running"), "");
});

t("recursive model: roles, ordering, nesting", () => {
  const v = { 评语: "好", 判据: [{ name: "内容", body: "充实" }], 判分: 8, extra: { a: [1, 2, { b: "x" }], c: null, d: true } };
  const m = buildNode(v, { scoreKey: "判分", criteriaKey: "判据" });
  assert.equal(m.kind, "object");
  assert.deepEqual(m.entries.map((e) => e.key), ["判分", "判据", "评语", "extra"]);
  assert.equal(m.entries[0].role, "score"); assert.equal(m.entries[1].role, "criteria"); assert.equal(m.entries[2].role, null);
  assert.equal(m.entries[1].node.kind, "array");
  assert.equal(m.entries[1].node.items[0].entries[0].node.value, "内容");
  const extra = m.entries[3].node;
  assert.equal(extra.entries[0].node.items[2].entries[0].node.value, "x");
  assert.equal(extra.entries[1].node.kind, "null"); assert.equal(extra.entries[2].node.kind, "boolean");
});

t("roles only at top level; strings kept verbatim (no HTML interpretation)", () => {
  const m = buildNode({ 判分: 1, inner: { 判分: 2 } }, { scoreKey: "判分", criteriaKey: "判据" });
  const inner = m.entries.find((e) => e.key === "inner").node;
  assert.equal(inner.entries[0].role, null);
  const s = buildNode("<img src=x onerror=alert(1)>", {});
  assert.equal(s.kind, "string"); assert.equal(s.value, "<img src=x onerror=alert(1)>");
});

t("depth and size limits", () => {
  let deep = "leaf"; for (let i = 0; i < 40; i++) deep = [deep];
  let node = buildNode(deep, {}); let d = 0; while (node.kind === "array") { node = node.items[0]; d++; }
  assert.equal(node.kind, "truncated"); assert.ok(d <= 14);
  const big = buildNode(Array.from({ length: 2000 }, (_, i) => i), {});
  assert.equal(big.items.length, 500); assert.equal(big.omitted, 1500);
});

t("code point counting", () => {
  assert.equal(cpLen("a😀b"), 3); assert.equal("a😀b".length, 4); assert.equal(cpLen(null), 0);
});

t("warnings", () => {
  assert.deepEqual(prominentWarnings(["regex_fallback", "x"]), { regexFallback: true, others: ["x"] });
  assert.deepEqual(prominentWarnings(null), { regexFallback: false, others: [] });
});

t("describeMessages never leaks image payloads", () => {
  const out = describeMessages([
    { role: "system", content: "sys" },
    { role: "user", content: [{ type: "text", text: "第1页/共1页" }, { type: "image_ref", path: "derived/vlm_1.jpg", sha256: "abcdef0123456789ff", page: "1/1" },
      { type: "image_url", image_url: { url: "data:image/jpeg;base64,AAAA" } }] },
  ]);
  assert.equal(out[0].parts[0].text, "sys");
  assert.equal(out[1].parts[1].kind, "image_ref");
  assert.ok(out[1].parts[1].text.includes("derived/vlm_1.jpg"));
  assert.ok(!JSON.stringify(out).includes("base64"));
});

t("latestVersion / hostOf", () => {
  assert.equal(latestVersion({ versions: [{ version: 1 }, { version: 3 }, { version: 2 }] }), 3);
  assert.equal(latestVersion(null), 0);
  assert.equal(hostOf("https://api.example.com:8443/v1"), "api.example.com:8443");
  assert.equal(hostOf("not a url"), "not a url");
});

t("static safety scan: no HTML-injection APIs in frontend/js", () => {
  const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "js");
  const bad = /\b(innerHTML|outerHTML|insertAdjacentHTML|document\.write|eval\s*\(|new Function)\b/;
  for (const f of readdirSync(dir)) {
    if (!/\.(js|mjs)$/.test(f)) continue;
    const src = readFileSync(path.join(dir, f), "utf8").split("\n").filter((l) => !l.trim().startsWith("//")).join("\n");
    assert.ok(!bad.test(src), "forbidden API in " + f);
  }
});

console.log(`m3_model tests passed: ${n}`);
