"""Pure grouping logic: decide the photo order and how photos are split into essays."""
from __future__ import annotations

import re

from ..errors import AppError

BASES = ("filename", "exif", "mtime")
_INT = re.compile(r"\d+")


def name_seq(orig_name: str) -> list[int] | None:
    """All integer groups of the file-name stem (natural-sort key); None if there are none."""
    base = re.split(r"[\/]", orig_name or "")[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    nums = [int(x) for x in _INT.findall(stem)]
    return nums or None


def _values(basis: str, files: list[dict]) -> list:
    out = []
    for f in files:
        if basis == "filename":
            v = f.get("name_seq")
            v = tuple(v) if v else None
        elif basis == "exif":
            v = f.get("exif_time") or None
        else:
            v = f.get("client_last_modified")
            v = v if isinstance(v, int) and not isinstance(v, bool) else None
        out.append(v)
    return out


def _usable(vals: list) -> tuple[bool, str | None]:
    if not vals:
        return False, "no_files"
    if any(v is None for v in vals):
        return False, "missing_values"
    if len(set(vals)) != len(vals):
        return False, "duplicate_values"
    return True, None


def _jsonable(v):
    return list(v) if isinstance(v, tuple) else v


def _chunks(ids: list[str], n: int) -> list[list[str]]:
    return [ids[i:i + n] for i in range(0, len(ids), n)]


def _validate(cfg, essays: list, all_ids: list[str]) -> list[dict]:
    problems: list[dict] = []
    max_pages = int(cfg["max_pages_per_essay"])
    known = set(all_ids)
    seen: set[str] = set()
    if not all_ids or not essays:
        problems.append({"type": "empty_batch", "essay_index": None, "detail": "no photos to submit"})
    for i, essay in enumerate(essays):
        if not essay:
            problems.append({"type": "empty_group", "essay_index": i, "detail": f"essay {i + 1} has no photos"})
            continue
        if len(essay) > max_pages:
            problems.append({"type": "too_many_pages", "essay_index": i,
                             "detail": f"essay {i + 1} has {len(essay)} photos (max {max_pages})"})
        for fid in essay:
            if fid not in known:
                problems.append({"type": "unknown_file", "essay_index": i, "detail": str(fid)})
            elif fid in seen:
                problems.append({"type": "duplicate_file", "essay_index": i, "detail": fid})
            else:
                seen.add(fid)
    for fid in all_ids:
        if fid not in seen:
            problems.append({"type": "unassigned_file", "essay_index": None, "detail": fid})
    return problems


_BLOCKING = {"empty_group", "too_many_pages", "unassigned_file", "duplicate_file", "unknown_file", "empty_batch"}


def resolve_groups(cfg, manifest: dict, req: dict) -> dict:
    req = req or {}
    files = manifest.get("files", [])
    upload_ids = [f["file_id"] for f in files]
    multi = bool(req.get("multi_photo"))

    bases: dict[str, dict] = {}
    vals_by: dict[str, dict[str, object]] = {}
    for b in BASES:
        vals = _values(b, files)
        ok, reason = _usable(vals)
        bases[b] = {"usable": ok, "reason": reason}
        vals_by[b] = dict(zip(upload_ids, vals))

    problems: list[dict] = []

    def result(basis, ordered, mode, essays, order_values):
        problems.extend(_validate(cfg, essays, upload_ids))
        return {
            "order_bases": bases, "order_basis": basis, "ordered_file_ids": ordered, "mode": mode,
            "essays": essays, "problems": problems,
            "can_submit": not any(p["type"] in _BLOCKING for p in problems),
            "needs_confirmation": mode == "manual",
            "summary": {"essays": len(essays), "photos": sum(len(e) for e in essays)},
            "order_values": order_values,
        }

    if not multi:
        return result("single", list(upload_ids), "single", [[i] for i in upload_ids],
                      {i: 1 for i in upload_ids})

    # --- choose the order basis
    requested = req.get("order_basis")
    chosen = None
    if requested in BASES:
        if bases[requested]["usable"]:
            chosen = requested
        else:
            problems.append({"type": "basis_unusable", "essay_index": None,
                             "detail": f"{requested}: {bases[requested]['reason']}"})
    if chosen is None:
        chosen = next((b for b in BASES if bases[b]["usable"]), None)
    if chosen is None:
        problems.append({"type": "no_reliable_order", "essay_index": None,
                         "detail": "no order basis is usable for every photo"})
        ordered = list(upload_ids)
        basis_name = "none"
        order_values = {i: None for i in upload_ids}
    else:
        ordered = sorted(upload_ids, key=lambda i: vals_by[chosen][i])
        basis_name = chosen
        order_values = {i: _jsonable(vals_by[chosen][i]) for i in upload_ids}

    mode = req.get("mode")
    raw_essays = req.get("essays")

    if mode == "per_n" and chosen is not None:
        n = req.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise AppError("invalid_grouping", "per_n grouping needs an integer n >= 1", 422)
        essays = _chunks(ordered, n)
        if len(ordered) % n != 0:
            problems.append({"type": "not_divisible", "essay_index": None,
                             "detail": f"{len(ordered)} photos cannot be split evenly into groups of {n}"})
            return result(basis_name, ordered, "manual", essays, order_values)
        return result(basis_name, ordered, "per_n", essays, order_values)

    # manual (explicitly requested, forced by missing order, or no mode given)
    if isinstance(raw_essays, list) and mode in ("manual", None, "per_n"):
        essays = [[x if isinstance(x, str) else str(x) for x in (e if isinstance(e, list) else [])]
                  for e in raw_essays]
        flat = [x for e in essays for x in e]
        pos = {}
        for k, x in enumerate(flat, 1):
            pos.setdefault(x, k)
        return result("manual_user", flat, "manual", essays, {i: pos.get(i) for i in upload_ids})
    return result(basis_name, ordered, "manual", [[i] for i in ordered], order_values)
