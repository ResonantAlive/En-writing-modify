"""VLM image preparation: derived/vlm_k.jpg per original photo, request size capped."""
from __future__ import annotations

import logging
from pathlib import Path

from ..batches import paths, store
from ..staging import images

log = logging.getLogger("wm.grading")

MAX_ITERATIONS = 8


class PreprocessError(Exception):
    pass


def _name(k: int, edge: int | None, q: int | None, first: bool) -> str:
    return f"vlm_{k}.jpg" if first else f"vlm_{k}_e{edge}_q{q}.jpg"


def _existing(sdir: Path, meta: dict, limit: int) -> list[dict] | None:
    """Reuse the previous derived set when every file exists, matches its sha256 and fits the limit."""
    out, total = [], 0
    for img in meta.get("images", []):
        ent = next((d for d in img.get("derived", []) if d.get("purpose") == "vlm"), None)
        if not ent:
            return None
        p = sdir / ent["path"]
        try:
            if not p.is_file() or images.sha256_file(p) != ent.get("sha256"):
                return None
            total += p.stat().st_size
        except OSError:
            return None
        out.append(ent)
    if not out or total > limit:
        return None
    return out


def prepare_vlm_images(cfg, subtask_id: str) -> list[dict]:
    """Returns one derived entry per original photo (in order): {"path","purpose","sha256","params"} with path
    relative to the subtask directory. Updates meta.images[k].derived. Raises PreprocessError."""
    sdir = paths.subtask_dir(cfg, subtask_id)
    meta = store.read_meta(cfg, subtask_id)
    if meta is None or not meta.get("images"):
        raise PreprocessError("no images")
    vlm = cfg["vlm"]
    limit = int(float(vlm["max_request_mb"]) * 1024 * 1024)
    reuse = _existing(sdir, meta, limit)
    if reuse is not None:
        return reuse
    edge, quality = int(vlm["max_edge"]), int(vlm["quality"])
    first = True
    entries: list[dict] = []
    for _ in range(MAX_ITERATIONS):
        entries = []
        total = 0
        longest = 0
        try:
            for k, img in enumerate(meta["images"], start=1):
                rel = f"image/derived/{_name(k, edge, quality, first)}"
                dst = sdir / rel
                src = sdir / img["original"]
                params = images.preprocess_for_vlm(src, dst, edge, quality)
                total += dst.stat().st_size
                longest = max(longest, params["width"], params["height"])
                entries.append({"path": rel, "purpose": "vlm", "sha256": images.sha256_file(dst), "params": params})
        except Exception as e:
            raise PreprocessError(f"{type(e).__name__}") from e
        if total <= limit:
            break
        first = False
        edge = max(64, int(min(edge, longest) * 0.85))
        quality = max(50, quality - 10)
    _commit(cfg, subtask_id, sdir, entries)
    return entries


def _commit(cfg, subtask_id: str, sdir: Path, entries: list[dict]) -> None:
    keep = {e["path"] for e in entries}
    stale: list[str] = []

    def fn(m: dict) -> None:
        for img, ent in zip(m["images"], entries):
            old = [d for d in img.get("derived", []) if d.get("purpose") == "vlm"]
            stale.extend(d["path"] for d in old if d.get("path") not in keep)
            img["derived"] = [d for d in img.get("derived", []) if d.get("purpose") != "vlm"] + [ent]
    store.update_meta(cfg, subtask_id, fn)
    for rel in stale:
        try:
            (sdir / rel).unlink()
        except OSError:
            pass
    # files written by an abandoned iteration that were never recorded
    d = sdir / "image" / "derived"
    if d.is_dir():
        for f in d.glob("vlm_*.jpg"):
            if f"image/derived/{f.name}" not in keep:
                try:
                    f.unlink()
                except OSError:
                    pass
