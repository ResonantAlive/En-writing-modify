// Pure grouping state model (no DOM). An "essays" value is an array of arrays of file ids.
// All functions are immutable: they return new arrays and never mutate their inputs.

export const clone = (essays) => essays.map((e) => e.slice());
export const flatten = (essays) => essays.flat();
export const countPhotos = (essays) => essays.reduce((n, e) => n + e.length, 0);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const NEW = Object.freeze({ marker: "new" });

export function equal(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].length !== b[i].length) return false;
    for (let j = 0; j < a[i].length; j++) if (a[i][j] !== b[i][j]) return false;
  }
  return true;
}

export const prune = (essays) => essays.filter((e) => e.length > 0);

/** Selected ids in global reading order (unknown ids ignored). */
export function inReadingOrder(essays, ids) {
  const set = new Set(ids);
  const out = [];
  for (const e of essays) for (const id of e) if (set.has(id)) out.push(id);
  return out;
}

/** Locate a file: {essay, index} or null. */
export function locate(essays, id) {
  for (let i = 0; i < essays.length; i++) {
    const j = essays[i].indexOf(id);
    if (j >= 0) return { essay: i, index: j };
  }
  return null;
}

/**
 * Move photos to essay `toEssay` at gap `toIndex` (0..len of that essay, measured BEFORE removal).
 * Works for reorder (same essay) and cross-essay moves. Emptied source essays are removed.
 */
export function movePhotos(essays, ids, toEssay, toIndex) {
  const moved = inReadingOrder(essays, ids);
  const target = essays[toEssay];
  if (!moved.length || !target) return clone(essays);
  const set = new Set(moved);
  const ti0 = clamp(toIndex, 0, target.length);
  const ti = ti0 - target.slice(0, ti0).filter((id) => set.has(id)).length;
  const out = essays.map((e) => e.filter((id) => !set.has(id)));
  out[toEssay].splice(ti, 0, ...moved);
  return prune(out);
}

export const movePhoto = (essays, id, toEssay, toIndex) => movePhotos(essays, [id], toEssay, toIndex);

/** Move photos into a brand-new essay placed before original essay index `at` (len = at the end). */
export function moveToNewEssay(essays, ids, at) {
  const moved = inReadingOrder(essays, ids);
  if (!moved.length) return clone(essays);
  const set = new Set(moved);
  const work = essays.map((e) => e.filter((id) => !set.has(id)));
  work.splice(clamp(at, 0, work.length), 0, NEW);
  return work.filter((e) => e === NEW || e.length > 0).map((e) => (e === NEW ? moved.slice() : e));
}

/** Split essay `ei` into two at gap `gap` (1..len-1). */
export function splitAt(essays, ei, gap) {
  const e = essays[ei];
  if (!e || gap <= 0 || gap >= e.length) return clone(essays);
  const out = clone(essays);
  out.splice(ei, 1, e.slice(0, gap), e.slice(gap));
  return out;
}

/** Merge essay `ei` with the next one. */
export function mergeAdjacent(essays, ei) {
  if (ei < 0 || ei >= essays.length - 1) return clone(essays);
  const out = clone(essays);
  out.splice(ei, 2, essays[ei].concat(essays[ei + 1]));
  return out;
}

/** Merge the selected photos into one essay, placed at the first selected photo's essay. */
export function mergeSelected(essays, ids) {
  const moved = inReadingOrder(essays, ids);
  if (moved.length < 2) return clone(essays);
  const set = new Set(moved);
  const firstIdx = essays.findIndex((e) => e.some((id) => set.has(id)));
  const work = essays.map((e) => e.filter((id) => !set.has(id)));
  // keep reading order: if unselected photos precede the first selected one, the merged essay goes after them
  const firstPos = essays[firstIdx].findIndex((id) => set.has(id));
  work.splice(firstPos > 0 ? firstIdx + 1 : firstIdx, 0, NEW);
  return work.filter((e) => e === NEW || e.length > 0).map((e) => (e === NEW ? moved : e));
}

/** Every selected photo becomes its own essay, in place; unselected neighbours keep their grouping. */
export function splitSelectedToSingles(essays, ids) {
  const set = new Set(ids);
  const out = [];
  for (const e of essays) {
    let run = [];
    for (const id of e) {
      if (set.has(id)) {
        if (run.length) out.push(run);
        run = [];
        out.push([id]);
      } else run.push(id);
    }
    if (run.length) out.push(run);
  }
  return out;
}

export function autoGroup(orderedIds, n) {
  const size = Math.max(1, Math.floor(Number(n)) || 1);
  const out = [];
  for (let i = 0; i < orderedIds.length; i += size) out.push(orderedIds.slice(i, i + size));
  return out;
}

export const singles = (orderedIds) => orderedIds.map((id) => [id]);

/** Make essays consistent with the current file id list: drop unknown/duplicate, append missing as singles. */
export function reconcile(essays, allIds) {
  const known = new Set(allIds);
  const seen = new Set();
  const out = [];
  for (const e of essays) {
    const kept = [];
    for (const id of e) if (known.has(id) && !seen.has(id)) { seen.add(id); kept.push(id); }
    if (kept.length) out.push(kept);
  }
  for (const id of allIds) if (!seen.has(id)) { seen.add(id); out.push([id]); }
  return out;
}

export function removeFile(essays, id) {
  return prune(essays.map((e) => e.filter((x) => x !== id)));
}

/** Client-side instant validation (server preview stays authoritative). */
export function validate(essays, maxPages) {
  const problems = [];
  essays.forEach((e, i) => {
    if (e.length === 0) problems.push({ type: "empty_group", essay_index: i, detail: "empty" });
    else if (maxPages > 0 && e.length > maxPages) problems.push({ type: "too_many_pages", essay_index: i, detail: `${e.length} > ${maxPages}` });
  });
  return problems;
}

/** Keyboard step: gap (in movePhotos terms) one position left/right, crossing essay borders; null at the ends. */
export function stepTarget(essays, id, dir) {
  const loc = locate(essays, id);
  if (!loc) return null;
  const len = essays[loc.essay].length;
  if (dir === "left") {
    if (loc.index > 0) return { essay: loc.essay, index: loc.index - 1 };
    if (loc.essay > 0) return { essay: loc.essay - 1, index: essays[loc.essay - 1].length };
    return null;
  }
  if (dir === "right") {
    if (loc.index < len - 1) return { essay: loc.essay, index: loc.index + 2 };
    if (loc.essay < essays.length - 1) return { essay: loc.essay + 1, index: 0 };
    return null;
  }
  return null;
}

// ---- undo / redo history (immutable) ----
export function createHistory(initial, limit = 100) {
  return { present: clone(initial), past: [], future: [], limit };
}
export function commit(h, next) {
  if (equal(h.present, next)) return h;
  return { ...h, present: clone(next), past: [...h.past, h.present].slice(-h.limit), future: [] };
}
export const canUndo = (h) => h.past.length > 0;
export const canRedo = (h) => h.future.length > 0;
export function undo(h) {
  if (!canUndo(h)) return h;
  return { ...h, present: h.past[h.past.length - 1], past: h.past.slice(0, -1), future: [h.present, ...h.future] };
}
export function redo(h) {
  if (!canRedo(h)) return h;
  return { ...h, present: h.future[0], past: [...h.past, h.present], future: h.future.slice(1) };
}
/** Replace content and clear history. */
export const resetHistory = (h, essays) => createHistory(essays, h.limit);
