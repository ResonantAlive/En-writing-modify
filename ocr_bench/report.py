"""Collect results/*.json into one markdown table (results/SUMMARY.md)."""
from __future__ import annotations

import json

import common as C

def main():
    rows = []
    for p in sorted(C.OUT.glob("*.json")):
        if p.name in ("flops.json",) or p.name.startswith("hunyuan"):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if "sec_per_page" not in r:
            continue
        a = r["accuracy"]
        st = r.get("stages_per_page") or r.get("stages_per_page_sum_over_workers") or {}
        det = st.get("det_infer", 0)
        rec = st.get("rec_infer", 0)
        other = sum(v for k, v in st.items() if k not in ("det_infer", "rec_infer"))
        rows.append((r["sec_per_page"], r["name"], det, rec, other, a))
    rows.sort(key=lambda x: -x[0])
    base = next((r[0] for r in rows if r[1] == "hf_baseline_det0"), max(r[0] for r in rows) if rows else 1)
    out = ["| 配置 | 秒/页 | 加速 | det 推理 | rec 推理 | 其余(前后处理) | WER | CER | 错词保留 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for sp, name, det, rec, other, a in rows:
        label = name + ("（检测降到1280，降质量，不采用）" if name == "hf_baseline_det1280" else "")
        out.append(f"| {label} | {sp:.2f} | {base/sp:.2f}x | {det:.2f} | {rec:.2f} | {other:.2f} | {a['wer']:.4f} | "
                   f"{a['cer']:.4f} | {a['typo_hit']}/{a['typo_total']} |")
    hy = []
    for p in sorted(C.OUT.glob("hunyuan*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        for pg in r["pages"]:
            hy.append(f"| {r['name']} | {pg['page']} | {pg['total_s']:.1f} | {pg['vision_s']:.1f} | "
                      f"{pg['prefill_s_incl_vision']:.1f} | {pg['decode_s']:.1f} | {pg['new_tokens']} | "
                      f"{pg['decode_tok_per_s']:.1f} |")
    if hy:
        out += ["", "| 混元配置 | 页 | 总秒 | 视觉编码 | 首 token(含视觉) | 解码 | 新 token | tok/s |",
                "|---|---|---|---|---|---|---|---|"] + hy
    text = "\n".join(out)
    (C.OUT / "SUMMARY.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
