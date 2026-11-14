"""Write answer-sheet templates straight into data/app/sheet_templates.json (no OCR needed)."""
from __future__ import annotations

from app.core import atomic_io
from app.sheets import store as sheets

DEFAULT_LINES = [
    {"text": "Cambridge IGCSE English Language", "box": [100, 40, 900, 80], "score": 0.99},
    {"text": "DO NOT WRITE IN THIS MARGIN", "box": [20, 300, 60, 1500], "score": 0.98},
    {"text": "0500/01/O/N/26", "box": [100, 1900, 400, 1930], "score": 0.99},
    {"text": "9 780000 123456", "box": [700, 1900, 1000, 1930], "score": 0.97},
    {"text": "Turn over", "box": [850, 1960, 1000, 1990], "score": 0.99},
]


def make_sheet(cfg, owner_id: str, *, tid: str = "sheet_t1", name: str = "IG 作文纸", status: str = "ready",
               lines=None, shared: bool = False, disabled: bool = False, pages: int = 1) -> str:
    data = atomic_io.read_json(sheets._path(cfg), None) or {"schema_version": 1, "templates": []}
    pgs = []
    for k in range(pages):
        pgs.append({"page_id": f"pg_{tid}_{k}", "orig_name": f"blank{k}.jpg", "file": f"pg_{tid}_{k}.jpg",
                    "mime": "image/jpeg", "sha256": "0" * 64, "width": 1100, "height": 2000, "status": status,
                    "error": None, "disabled": disabled, "created_at": "2026-11-14T10:00:00",
                    "ocr": ({"engine": {}, "lines": list(DEFAULT_LINES if lines is None else lines),
                             "warnings": [], "recognized_at": "x"} if status == "ready" else None)})
    data["templates"].append({"id": tid, "name": name, "owner_id": owner_id, "shared": shared,
                              "created_at": "2026-11-14T10:00:00", "updated_at": "2026-11-14T10:00:00",
                              "pages": pgs})
    atomic_io.write_json_atomic(sheets._path(cfg), data)
    return tid
