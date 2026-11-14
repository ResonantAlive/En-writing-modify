import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { buildReviewBody, reviewErrorText, revisionLine, summaryRows, fmtScore, fmtDiff, DEFAULT_LIMITS } from "../js/review_model.mjs";

let n = 0;
const t = (name, fn) => { try { fn(); n++; } catch (e) { console.error("FAIL:", name); throw e; } };

t("score: required, numeric, within [0, full]", () => {
  assert.deepEqual(buildReviewBody("", [], 20).errors, ["请填写教师分数"]);
  assert.deepEqual(buildReviewBody("abc", [], 20).errors, ["分数必须是数字"]);
  assert.equal(buildReviewBody("20.5", [], 20).errors.length, 1);
  assert.equal(buildReviewBody("-1", [], 20).errors.length, 1);
  const ok = buildReviewBody(" 15.5 ", [], 20);
  assert.deepEqual(ok, { errors: [], body: { final_score: 15.5 } });          // no items key when none
  assert.deepEqual(buildReviewBody("0", [], 20).body, { final_score: 0 });
  assert.deepEqual(buildReviewBody("20", [], 20).errors, []);
});

t("items: blank rows dropped, content required, names trimmed", () => {
  const r = buildReviewBody("10", [{ name: "", content: "  " }, { name: " 内容 ", content: "要点齐全\n" }, { name: "x", content: "" }], 20);
  assert.equal(r.errors.length, 1);
  assert.match(r.errors[0], /第 2 条：内容不能为空/);
  const ok = buildReviewBody("10", [{ name: " 内容 ", content: "要点齐全\n" }, {}], 20);
  assert.deepEqual(ok.body.items, [{ name: "内容", content: "要点齐全\n" }]);
});

t("limits count code points like the server (emoji = 1)", () => {
  const emoji = "👍😀".repeat(2500) + "评".repeat(5000);
  assert.equal(emoji.length, 15000);                                          // UTF-16 units
  const items = Array.from({ length: 199 }, () => ({ name: "名".repeat(100), content: "中".repeat(10000) })).concat([{ name: "表情", content: emoji }]);
  assert.deepEqual(buildReviewBody("10", items, 20).errors, []);
  assert.equal(buildReviewBody("10", items.concat([{ name: "", content: "x" }]), 20).errors[0], "条目最多 200 条（当前 201 条）");
  assert.match(buildReviewBody("10", [{ name: "", content: "😀".repeat(10001) }], 20).errors[0], /内容 10001 字，最多 10000 字/);
  assert.match(buildReviewBody("10", [{ name: "n".repeat(101), content: "x" }], 20).errors[0], /名称 101 字，最多 100 字/);
  assert.deepEqual(buildReviewBody("10", [{ name: "", content: "x".repeat(5) }], 20, { max_item_chars: 4 }).errors.length, 1);
  assert.equal(DEFAULT_LIMITS.max_items, 200);
});

t("server error codes become Chinese messages with index / limits", () => {
  assert.equal(reviewErrorText({ code: "too_many_items", detail: { max: 200, got: 201 } }), "条目过多：最多 200，当前 201");
  assert.equal(reviewErrorText({ code: "item_too_long", detail: { index: 0, max: 10000, got: 10001 } }), "条目内容过长（第 1 条）：最多 10000，当前 10001");
  assert.equal(reviewErrorText({ code: "no_ai_result" }), "这篇作文没有可采纳的 AI 结果");
  assert.equal(reviewErrorText({ code: "unknown_code" }), null);
  assert.equal(reviewErrorText({ code: "constructor" }), null);              // no prototype leakage
  assert.equal(reviewErrorText(null), null);
});

t("revision line and summary rows", () => {
  assert.equal(revisionLine({ rev: 1, adopted_ai: true, final_score: 7, ai_score: 7, diff: 0, items: null }), "第 1 版 · 一键采纳 AI · 最终 7 · AI 7 · 差值 0");
  assert.equal(revisionLine({ rev: 2, adopted_ai: false, final_score: 15.5, ai_score: 7, diff: 8.5, items: [{}, {}] }), "第 2 版 · 教师改分 · 最终 15.5 · AI 7 · 差值 +8.5 · 2 条意见");
  assert.equal(revisionLine({ rev: 1, adopted_ai: false, final_score: 12, ai_score: null, diff: null, items: null }), "第 1 版 · 人工评分（AI 未给出分数） · 最终 12");
  assert.equal(fmtScore(null), "—"); assert.equal(fmtScore(1 / 3), "0.3333"); assert.equal(fmtDiff(-1.5), "-1.5"); assert.equal(fmtDiff(null), "—");
  const rows = summaryRows({ counts: { done: 2, pending_review: 1, failed: 0, cancelled: 0, total: 3 }, stats: { mean_final_score: 12.25, mean_abs_diff: null } });
  assert.deepEqual(rows.map((r) => r[1]), [3, 2, 1, 0, 0, "12.25", "—"]);
  assert.equal(summaryRows(null)[0][1], 0);
});

t("review UI: blob download only, no hrefs from data, code-point counters", () => {
  const src = readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "js", "review.js"), "utf8");
  assert.ok(src.includes("URL.createObjectURL(new Blob("));
  assert.equal((src.match(/\.href\s*=/g) || []).length, 1);                 // the single blob: link
  assert.ok(src.includes("a.href = url;"));
  assert.ok(src.includes("cpLen(input.value)"));
});

console.log(`m5_model tests passed: ${n}`);
