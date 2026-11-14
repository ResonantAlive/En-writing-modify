"""Answer-sheet template check: recognise blank sheet(s) and filled photo(s) with the app's local OCR and show
what the template filter removes.

Usage:
    python tools/sheet_check.py --blank blank_p1.jpg [--blank blank_p2.jpg ...] photo1.jpg [photo2.jpg ...]
        [--out sheet_check_out]

For every photo it prints the alignment (scale / rotation / matched anchors), every removed box with its reason
and the final transcription. With --out the same is written as JSON (one file per photo). Uses exactly the app's
path (PaddleEngine -> sheet_filter -> layout restoration); nothing under data/ is touched except the exported
OpenVINO model in data/ov_models (created on first use).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blank", action="append", required=True, help="blank answer-sheet photo (repeatable)")
    ap.add_argument("photos", nargs="+", help="filled answer-sheet photos")
    ap.add_argument("--backend", default="openvino", choices=["openvino", "torch"])
    ap.add_argument("--out", default=None, help="folder for JSON results")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from app.config import load_config
    from app.ocr import service
    from app.ocr.paddle_engine import PaddleEngine
    from app.staging.images import load_oriented

    cfg = load_config(ROOT / "data")
    eng = PaddleEngine(service.model_root(cfg), backend=a.backend, ir_dir=service.ir_dir(cfg))
    t0 = time.perf_counter()
    eng.load()
    eng.warmup()
    print(f"engine ready in {time.perf_counter() - t0:.1f}s")

    template_pages = []
    for k, b in enumerate(a.blank, 1):
        with load_oriented(Path(b)) as im:
            t0 = time.perf_counter()
            res = eng.recognize(im)
        lines = [{"text": l.text, "box": list(l.box)} for l in (res.boxes or res.lines) if l.text.strip()]
        template_pages.append({"page_id": f"blank{k}", "lines": lines})
        print(f"\n[blank {k}] {b}: {len(lines)} printed text boxes ({time.perf_counter() - t0:.1f}s)")
        for l in lines:
            print(f"    {l['text']}")

    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    for p in a.photos:
        with load_oriented(Path(p)) as im:
            t0 = time.perf_counter()
            res = eng.recognize(im)
            dt = time.perf_counter() - t0
        t1 = time.perf_counter()
        f = service.apply_sheet_filter([res], template_pages)[0]        # the app's own path
        ft = time.perf_counter() - t1
        excluded, info, kept = f["excluded"], f["info"], res.boxes
        text = service.assemble([res], float(cfg["layout"]["blank_line_factor"]))
        print(f"\n[photo] {p}: OCR {dt:.1f}s, filter {ft * 1000:.0f} ms, page tilt {res.skew_deg:.2f}°")
        print(f"    template page: {info['template_page']}, aligned: {info['aligned']}, anchors: {info['anchors']}"
              + (f", scale {info['scale']}, rotation {info['angle_deg']}°" if info["aligned"] else ""))
        print(f"    removed {len(excluded)} box(es):")
        for e in excluded:
            print(f"      - [{e['reason']}] {e['text']}")
        print("    transcription:")
        for ln in text.splitlines():
            print(f"      | {ln}")
        if out:
            (out / (Path(p).stem + ".json")).write_text(json.dumps(
                {"photo": str(p), "info": info, "excluded": excluded, "transcription": text,
                 "kept": [{"text": b.text, "box": list(b.box)} for b in kept]}, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
