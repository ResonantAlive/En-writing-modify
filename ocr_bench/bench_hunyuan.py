"""HunyuanOCR on CPU (FP32, model's own image preprocessing, greedy decoding). Reports vision / prefill / decode split.

  python bench_hunyuan.py --attn sdpa --pages test1/0 --prompt plain
"""
from __future__ import annotations

import argparse
import time

import torch
from transformers import LogitsProcessor, LogitsProcessorList

import common as C

PROMPTS = {
    "plain": "Extract all the text in the image, keep the line breaks and paragraphs of the original.",
    "spot": "Detect and recognize text in the image, and output the text coordinates in a formatted manner.",
}


class StepClock(LogitsProcessor):
    """Records a timestamp per generated token; optionally switches thread count once decoding starts
    (prefill/vision are compute-bound, decode is memory-bound and may prefer fewer, faster cores)."""

    def __init__(self, decode_threads: int = 0):
        self.ts = []
        self.decode_threads = decode_threads

    def __call__(self, input_ids, scores):
        if not self.ts and self.decode_threads:
            torch.set_num_threads(self.decode_threads)
        self.ts.append(time.perf_counter())
        return scores


def load(attn: str, dtype: str):
    from transformers import AutoImageProcessor, AutoTokenizer, HunYuanVLForConditionalGeneration, HunYuanVLProcessor
    p = str(C.HY_DIR)
    tok = AutoTokenizer.from_pretrained(p)
    for attr, tid in (("image_start_token", 120118), ("image_end_token", 120119), ("image_token", 120120)):
        setattr(tok, attr, tok.convert_ids_to_tokens(tid))
        setattr(tok, attr + "_id", tid)
    proc = HunYuanVLProcessor(image_processor=AutoImageProcessor.from_pretrained(p), tokenizer=tok)
    model = HunYuanVLForConditionalGeneration.from_pretrained(p, attn_implementation=attn,
                                                              dtype=getattr(torch, dtype)).eval()
    return proc, model


def run_one(proc, model, im, prompt, max_new_tokens, decode_threads=0):
    T = C.Timer()
    with T("preprocess"):
        msgs = [{"role": "user", "content": [{"type": "image", "image": "x"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if proc.image_start_token + proc.image_token not in text:
            text = text.replace(proc.image_token, proc.image_start_token + proc.image_token + proc.image_end_token, 1)
        inputs = proc(text=[text], images=im, padding=True, return_tensors="pt")
    vis = {}

    def pre(_m, _a):
        vis["t0"] = time.perf_counter()

    def post(_m, _a, _o):
        vis["t1"] = time.perf_counter()
    h1 = model.model.vision_tower.register_forward_pre_hook(pre)
    h2 = model.model.vision_tower.register_forward_hook(post)
    n0 = torch.get_num_threads()
    clk = StepClock(decode_threads)
    t0 = time.perf_counter()
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             logits_processor=LogitsProcessorList([clk]))
    t_end = time.perf_counter()
    h1.remove(); h2.remove()
    torch.set_num_threads(n0)
    out = ids[0][inputs["input_ids"].shape[1]:]
    n_new = len(out)
    raw = proc.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    first = clk.ts[0] if clk.ts else t_end
    stats = {
        "preprocess_s": T.t["preprocess"],
        "image_px": list(inputs["image_grid_thw"][0].tolist()) if "image_grid_thw" in inputs else None,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "vision_s": vis.get("t1", 0) - vis.get("t0", 0),
        "prefill_s_incl_vision": first - t0,
        "decode_s": t_end - first,
        "new_tokens": n_new,
        "decode_tok_per_s": (n_new - 1) / max(1e-9, t_end - first),
        "total_s": t_end - t0 + T.t["preprocess"],
    }
    return raw, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--pages", default="test1/0", help="comma list case/page_index, or 'all'")
    ap.add_argument("--prompt", default="plain", choices=list(PROMPTS))
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--decode-threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    t = time.perf_counter()
    proc, model = load(a.attn, a.dtype)
    load_s = time.perf_counter() - t
    cases = {c["name"]: c for c in C.load_cases()}
    sel = [(n, i) for n, c in cases.items() for i in range(len(c["images"]))] if a.pages == "all" else \
        [(p.split("/")[0], int(p.split("/")[1])) for p in a.pages.split(",")]
    results = []
    for n, i in sel:
        raw, st = run_one(proc, model, cases[n]["images"][i], PROMPTS[a.prompt], a.max_new_tokens, a.decode_threads)
        sc = C.score_text(C.join_lines(raw.split("\n")), cases[n]["gt"], cases[n]["expected"]) \
            if len(cases[n]["images"]) == 1 else None
        results.append({"page": f"{n}/{i}", **st, "score_single_page_case": sc, "raw": raw})
        print(f"{n}/{i}: total {st['total_s']:.1f}s  vision {st['vision_s']:.1f}s  prefill(+vis) "
              f"{st['prefill_s_incl_vision']:.1f}s  decode {st['decode_s']:.1f}s for {st['new_tokens']} tok "
              f"({st['decode_tok_per_s']:.1f} tok/s)  grid {st['image_px']} prompt_tok {st['prompt_tokens']}"
              + (f"  WER {sc['wer']:.3f} typos {sc['typo_hit']}/{sc['typo_total']}" if sc else ""), flush=True)
    name = f"hunyuan_{a.attn}_{a.dtype}_{a.prompt}_t{a.threads or 'auto'}_dt{a.decode_threads or 'same'}{a.tag}"
    C.save_json(name + ".json", {"name": name, "load_s": load_s, "pages": results})


if __name__ == "__main__":
    main()
