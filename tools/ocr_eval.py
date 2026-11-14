"""OCR evaluation tool: run the app's local OCR (PP-OCRv6) on a folder of labelled essay photos.

Layout of the data folder (default: ./test_data):
    <case>/page_1.png|jpg ...   photos in page order
    <case>/text.txt             ground-truth essay (one paragraph per line); optional trailing section:
                                    ##错误词            (marker line, optional)
                                    expected = {"goverment": 5, ...}   misspelled word -> count in the essay
                                    注：...             (free notes, ignored)

Metrics per case:
    CER, WER          edit distance against the ground truth (whitespace-normalised)
    faithfulness      share of the student's misspellings that survive recognition verbatim -- an engine that
                      "autocorrects" scores low; this matters most for grading
    paragraphs        paragraph count of the restored transcription vs. ground truth (layout restoration)
    seconds           wall time per case (model load / warm-up excluded)

Usage:
    python tools/ocr_eval.py [--data test_data] [--cases test1,test2] [--backend openvino|torch] [--out eval_out]

Uses exactly the app's code path (PaddleEngine -> layout restoration). Nothing here touches data/ except the
exported OpenVINO model in data/ov_models (created on first use). Results go to --out.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_EXPECTED = re.compile(r'"([^"]+)"\s*:\s*(\d+)')


def parse_truth(path: Path) -> tuple[str, dict[str, int]]:
    raw = path.read_text(encoding="utf-8-sig")
    body: list[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if s.startswith("##") or s.startswith("expected"):
            break
        body.append(ln)
    text = "\n".join(body).strip()
    m = re.search(r"expected\s*=\s*\{(.*?)\}", raw, re.S)
    exp = {k: int(v) for k, v in _EXPECTED.findall(m.group(1))} if m else {}
    return text, exp


def edit_distance(a: list, b: list) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def cer_wer(truth: str, pred: str) -> tuple[float, float]:
    t, p = norm_ws(truth), norm_ws(pred)
    cer = edit_distance(list(t), list(p)) / max(1, len(t))
    tw, pw = t.split(), p.split()
    return cer, edit_distance(tw, pw) / max(1, len(tw))


def faithfulness(expected: dict[str, int], pred: str) -> tuple[int, int, dict]:
    have: dict[str, int] = {}
    for w in re.findall(r"[A-Za-z']+", pred.lower()):
        have[w] = have.get(w, 0) + 1
    kept = total = 0
    detail = {}
    for w, n in expected.items():
        k = min(n, have.get(w.lower(), 0))
        kept += k
        total += n
        detail[w] = f"{k}/{n}"
    return kept, total, detail


def paragraphs(s: str) -> int:
    return len([ln for ln in s.splitlines() if ln.strip()])


def page_files(case_dir: Path) -> list[Path]:
    imgs = [p for p in case_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".heic")]
    return sorted(imgs, key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "test_data"))
    ap.add_argument("--cases", default="")
    ap.add_argument("--backend", default="openvino", choices=["openvino", "torch"])
    ap.add_argument("--out", default=str(ROOT / "eval_out"))
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from app.ocr.paddle_engine import PaddleEngine
    from app.ocr.service import assemble
    from app.staging.images import load_oriented

    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [c for c in sorted(data.iterdir()) if c.is_dir() and (c / "text.txt").exists()]
    if a.cases:
        want = set(a.cases.split(","))
        cases = [c for c in cases if c.name in want]
    eng = PaddleEngine(ROOT, backend=a.backend, ir_dir=ROOT / "data" / "ov_models")
    t0 = time.time()
    eng.warmup()
    print(f"[load+warm-up {time.time() - t0:.1f}s] {eng.info()['version']}", flush=True)
    results = []
    for c in cases:
        truth, expected = parse_truth(c / "text.txt")
        try:
            t = time.time()
            pages = [eng.recognize(load_oriented(p)) for p in page_files(c)]
            secs = time.time() - t
        except Exception as e:                                # report and continue with the next case
            print(f"{c.name}: FAILED {type(e).__name__}: {e}")
            continue
        pred = assemble(pages)
        cer, wer = cer_wer(truth, pred)
        kept, total, detail = faithfulness(expected, pred)
        row = {"case": c.name, "pages": len(pages), "cer": round(cer, 4), "wer": round(wer, 4),
               "misspellings_kept": kept, "misspellings_total": total,
               "faithfulness": round(kept / total, 3) if total else None,
               "paragraphs_truth": paragraphs(truth), "paragraphs_pred": paragraphs(pred),
               "seconds": round(secs, 1), "seconds_per_page": round(secs / max(1, len(pages)), 2), "detail": detail}
        results.append(row)
        (out / f"{c.name}.txt").write_text(pred, encoding="utf-8")
        print(f"{c.name}: CER {cer:.4f} WER {wer:.4f} misspellings kept {kept}/{total} "
              f"paragraphs {row['paragraphs_pred']}/{row['paragraphs_truth']} {secs:.1f}s ({len(pages)} pages)",
              flush=True)
    (out / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if results:
        n_pages = sum(r["pages"] for r in results)
        k = sum(r["misspellings_kept"] for r in results)
        t = sum(r["misspellings_total"] for r in results)
        print(f"== mean CER {sum(r['cer'] for r in results) / len(results):.4f}  "
              f"mean WER {sum(r['wer'] for r in results) / len(results):.4f}  misspellings kept {k}/{t}  "
              f"{sum(r['seconds'] for r in results) / n_pages:.2f} s/page over {n_pages} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
