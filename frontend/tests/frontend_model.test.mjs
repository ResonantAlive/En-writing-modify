import assert from "node:assert/strict";
import * as M from "../js/group_model.mjs";

const E = (s) => s.split("|").map((g) => (g ? g.split("") : [])); // "ab|c" -> [[a,b],[c]]
const S = (es) => es.map((e) => e.join("")).join("|");
let n = 0;
const t = (name, fn) => { fn(); n++; };

t("reorder within essay (move forward adjusts index)", () => {
  assert.equal(S(M.movePhoto(E("abc|d"), "a", 0, 3)), "bca|d");
  assert.equal(S(M.movePhoto(E("abc|d"), "c", 0, 0)), "cab|d");
  assert.equal(S(M.movePhoto(E("abc|d"), "a", 0, 1)), "abc|d"); // gap right after itself = no-op
});
t("move between essays; empty source is removed", () => {
  assert.equal(S(M.movePhoto(E("ab|c"), "c", 0, 1)), "acb");
  assert.equal(S(M.movePhoto(E("ab|c|d"), "c", 2, 1)), "ab|dc");
});
t("move multiple keeps reading order", () => {
  assert.equal(S(M.movePhotos(E("abc|de"), ["e", "a"], 1, 1)), "bc|dae");
});
t("move to new essay", () => {
  assert.equal(S(M.moveToNewEssay(E("ab|c"), ["a"], 2)), "b|c|a");
  assert.equal(S(M.moveToNewEssay(E("ab|c"), ["c"], 0)), "c|ab");
  assert.equal(S(M.moveToNewEssay(E("ab|c"), ["c"], 1)), "ab|c");
});
t("split at gap and merge adjacent", () => {
  assert.equal(S(M.splitAt(E("abc|d"), 0, 1)), "a|bc|d");
  assert.equal(S(M.splitAt(E("abc|d"), 0, 0)), "abc|d");
  assert.equal(S(M.splitAt(E("abc|d"), 0, 3)), "abc|d");
  assert.equal(S(M.mergeAdjacent(E("a|b|c"), 0)), "ab|c");
  assert.equal(S(M.mergeAdjacent(E("a|b|c"), 2)), "a|b|c");
});
t("merge selected", () => {
  assert.equal(S(M.mergeSelected(E("a|b|c|d"), ["d", "b"])), "a|bd|c");
  assert.equal(S(M.mergeSelected(E("ab|c"), ["b", "c"])), "a|bc");
  assert.equal(S(M.mergeSelected(E("ab|c"), ["a", "b"])), "ab|c");
  assert.equal(S(M.mergeSelected(E("ab|c"), ["a"])), "ab|c");
});
t("split selected into singles keeps order", () => {
  assert.equal(S(M.splitSelectedToSingles(E("abcd|e"), ["b"])), "a|b|cd|e");
  assert.equal(S(M.splitSelectedToSingles(E("abc"), ["a", "b", "c"])), "a|b|c");
  assert.equal(S(M.splitSelectedToSingles(E("abc|d"), [])), "abc|d");
});
t("auto group", () => {
  assert.equal(S(M.autoGroup(["a", "b", "c", "d", "e"], 2)), "ab|cd|e");
  assert.equal(S(M.autoGroup(["a", "b"], 0)), "a|b");
  assert.equal(S(M.singles(["a", "b"])), "a|b");
});
t("reconcile / removeFile / validate", () => {
  assert.equal(S(M.reconcile(E("ab|x|c"), ["a", "b", "c", "d"])), "ab|c|d");
  assert.equal(S(M.reconcile(E("aa|b"), ["a", "b"])), "a|b");
  assert.equal(S(M.removeFile(E("ab|c"), "c")), "ab");
  const p = M.validate([[], ["a", "b", "c"], ["d"]], 2);
  assert.deepEqual(p.map((x) => [x.type, x.essay_index]), [["empty_group", 0], ["too_many_pages", 1]]);
});
t("stepTarget (keyboard moves)", () => {
  const es = E("ab|c");
  assert.deepEqual(M.stepTarget(es, "b", "left"), { essay: 0, index: 0 });
  assert.deepEqual(M.stepTarget(es, "a", "right"), { essay: 0, index: 2 });
  assert.deepEqual(M.stepTarget(es, "b", "right"), { essay: 1, index: 0 });
  assert.deepEqual(M.stepTarget(es, "c", "left"), { essay: 0, index: 2 });
  assert.equal(M.stepTarget(es, "a", "left"), null);
  assert.equal(S(M.movePhoto(es, "a", 0, 2)), "ba|c");
  assert.equal(S(M.movePhoto(es, "b", 1, 0)), "a|bc");
  assert.equal(S(M.movePhoto(es, "c", 0, 2)), "abc");
});
t("immutability", () => {
  const es = E("ab|c");
  const snap = JSON.stringify(es);
  M.movePhoto(es, "a", 1, 0); M.mergeSelected(es, ["a", "c"]); M.splitAt(es, 0, 1);
  M.moveToNewEssay(es, ["a"], 0); M.splitSelectedToSingles(es, ["a"]);
  assert.equal(JSON.stringify(es), snap);
});
t("history undo/redo", () => {
  let h = M.createHistory(E("a|b|c"));
  h = M.commit(h, E("ab|c"));
  h = M.commit(h, E("ab|c")); // identical -> no new entry
  assert.equal(h.past.length, 1);
  h = M.commit(h, E("abc"));
  h = M.undo(h); assert.equal(S(h.present), "ab|c");
  h = M.undo(h); assert.equal(S(h.present), "a|b|c");
  h = M.undo(h); assert.equal(S(h.present), "a|b|c");
  assert.ok(!M.canUndo(h) && M.canRedo(h));
  h = M.redo(h); assert.equal(S(h.present), "ab|c");
  h = M.commit(h, E("a|bc")); // new edit clears redo
  assert.ok(!M.canRedo(h));
  let x = M.createHistory(E("a|b"), 2);
  for (let i = 0; i < 5; i++) x = M.commit(x, [["a"], ["b"], [String(i)]]);
  assert.equal(x.past.length, 2);
});
t("conservation: no op loses or duplicates photos", () => {
  let es = E("abc|de|f|g");
  const all = M.flatten(es).sort().join("");
  const ops = [
    (e) => M.movePhotos(e, ["a", "f"], 1, 1), (e) => M.mergeSelected(e, ["g", "b"]),
    (e) => M.splitSelectedToSingles(e, ["c", "d"]), (e) => M.moveToNewEssay(e, ["e"], 0),
    (e) => M.splitAt(e, 0, 1), (e) => M.mergeAdjacent(e, 0),
  ];
  for (let r = 0; r < 20; r++) {
    for (const op of ops) {
      es = op(es);
      assert.equal(M.flatten(es).sort().join(""), all);
      assert.ok(es.every((g) => g.length));
    }
  }
});
console.log(`ok - ${n} test groups passed`);
