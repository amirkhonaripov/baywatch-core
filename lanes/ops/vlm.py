"""Qwen3-VL zero-shot bay-state answers with per-state yes-probabilities (from answer-token logprobs).

Runs in the separate venv env/ops-vlm (vLLM pins its own torch):
  $SEEN_ROOT/env/ops-vlm/bin/python -m lanes.ops.vlm --items items.parquet --out out.parquet [--crop]
items: columns item_id, path (relative to SEEN_DATA or absolute) and optional roi_x1..roi_y2 (normalised).
Used two ways: (1) baseline on the hand-labelled CCTV frames ("why not just call a big model"),
(2) teacher labels for training frames that have no human/derived label for a state (verified on a sample).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from seen.paths import DATA

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
KEYS = ["car_present", "raised", "wheel_off", "hood_open", "person_at_car", "person_under_car", "work", "washing"]
PROMPT = """This is a frame from a camera in a car service workshop (or a still photo of a car).
Answer about the main car in the image (the largest or most central car, van, SUV or pickup).
Reply with exactly these 8 lines, each in the form `name: yes` or `name: no`, nothing else:
car_present: is a car present in the image?
raised: is the car lifted off the floor on a lift or a jack (its wheels not standing on the ground)?
wheel_off: is at least one wheel removed from the car?
hood_open: is the car's bonnet (hood) open?
person_at_car: is a person standing, crouching or kneeling right next to the car or touching it?
person_under_car: is a person underneath the car?
work: is someone actively working on the car (hands or tools on it, handling a wheel or a part of it)?
washing: is the car being washed (water, foam or a pressure washer on it)?"""


def _img(path: str, roi=None, margin: float = 0.1, min_side: int = 448) -> Image.Image:
    p = Path(path)
    im = Image.open(p if p.is_absolute() else DATA / p).convert("RGB")
    if roi is not None:
        w, h = im.size
        x1, y1, x2, y2 = roi
        dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
        im = im.crop((int(max(0, x1 - dx) * w), int(max(0, y1 - dy) * h), int(min(1, x2 + dx) * w), int(min(1, y2 + dy) * h)))
    s = min_side / min(im.size)
    if s > 1:
        im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.Resampling.BICUBIC)
    return im


def _data_url(im: Image.Image) -> str:
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _parse(out) -> dict[str, float]:
    """Walk generated tokens; after each `key:` read P(yes) / (P(yes) + P(no)) from the top logprobs."""
    probs: dict[str, float] = {}
    text = ""
    toks = out.outputs[0]
    for tid, lp in zip(toks.token_ids, toks.logprobs or []):
        stripped = text.rstrip()
        pending = next((k for k in KEYS if k not in probs and stripped.endswith(k + ":")), None)
        if pending is not None:
            py = pn = -math.inf
            for cand in lp.values():
                t = (cand.decoded_token or "").strip().lower()
                if t == "yes":
                    py = max(py, cand.logprob)
                elif t == "no":
                    py, pn = py, max(pn, cand.logprob)
            if py > -math.inf or pn > -math.inf:
                m = max(py, pn)
                ey, en = math.exp(py - m) if py > -math.inf else 0.0, math.exp(pn - m) if pn > -math.inf else 0.0
                probs[pending] = ey / (ey + en)
        text += lp[tid].decoded_token if tid in lp else ""
    # fallback for anything the logprob walk missed: parse the text
    for line in toks.text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip().strip("`").lower()
            if k in KEYS and k not in probs and v.strip().lower().startswith(("yes", "no")):
                probs[k] = 1.0 if v.strip().lower().startswith("yes") else 0.0
    return probs


def _hf_parse(text_tokens: list[str], step_probs: list[dict[str, float]], text: str) -> dict[str, float]:
    """Same walk as _parse for the transformers backend: step_probs[i] = {'yes': p, 'no': p} at generated step i."""
    probs: dict[str, float] = {}
    acc = ""
    for tok, sp in zip(text_tokens, step_probs):
        stripped = acc.rstrip()
        pending = next((k for k in KEYS if k not in probs and stripped.endswith(k + ":")), None)
        if pending is not None and (sp["yes"] + sp["no"]) > 0:
            probs[pending] = sp["yes"] / (sp["yes"] + sp["no"])
        acc += tok
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip().strip("`").lower()
            if k in KEYS and k not in probs and v.strip().lower().startswith(("yes", "no")):
                probs[k] = 1.0 if v.strip().lower().startswith("yes") else 0.0
    return probs


def run_hf(items: pd.DataFrame, out: Path, crop: bool, batch: int, timing_n: int) -> None:
    """transformers backend (env/main). Slower than vLLM; reported latency/cost are therefore upper bounds."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    t_load = time.time()
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 256, max_pixels=640 * 640)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    cold = time.time() - t_load
    tok = proc.tokenizer
    yes_ids = {i for t in ("yes", " yes", "Yes", " Yes") for i in tok.encode(t, add_special_tokens=False)[:1]}
    no_ids = {i for t in ("no", " no", "No", " No") for i in tok.encode(t, add_special_tokens=False)[:1]}

    def one_batch(rows):
        ims, texts = [], []
        for r in rows:
            roi = (r.roi_x1, r.roi_y1, r.roi_x2, r.roi_y2) if crop and hasattr(r, "roi_x1") and not pd.isna(r.roi_x1) else None
            ims.append(_img(r.path, roi))
            msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            texts.append(proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False))
        enc = proc(text=texts, images=ims, return_tensors="pt", padding=True).to("cuda:0")
        with torch.inference_mode():
            g = model.generate(**enc, max_new_tokens=64, do_sample=False, return_dict_in_generate=True, output_scores=True)
        seqs = g.sequences[:, enc["input_ids"].shape[1]:]
        probs = [torch.softmax(sc.float(), -1).cpu() for sc in g.scores]
        res = []
        for b in range(len(rows)):
            ids = seqs[b].tolist()
            toks = [tok.decode([i]) for i in ids]
            sps = [{"yes": float(sum(probs[t][b, i] for i in yes_ids)), "no": float(sum(probs[t][b, i] for i in no_ids))}
                   for t in range(len(ids))]
            text = tok.decode(ids, skip_special_tokens=True)
            res.append((_hf_parse(toks, sps, text), text, int(enc["attention_mask"][b].sum()), len(ids)))
        return res

    rows = list(items.itertuples(index=False))
    lat = []
    if timing_n:
        one_batch(rows[:1])
        for r in rows[:timing_n]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            one_batch([r])
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
    out_rows = []
    t0 = time.time()
    n_in = n_out = 0
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        for r, (p, text, ni, no) in zip(chunk, one_batch(chunk)):
            n_in += ni
            n_out += no
            out_rows.append({"item_id": r.item_id, **{f"p_{k}": p.get(k, np.nan) for k in KEYS}, "raw": text})
        print(s + len(chunk), len(rows), f"{time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0
    pd.DataFrame(out_rows).to_parquet(out)
    stats = dict(model=MODEL, engine="transformers", n=len(rows), batch=batch, wall_s=wall,
                 frames_per_s=len(rows) / max(wall, 1e-9),
                 single_latency_ms_p50=float(np.percentile(lat, 50)) if lat else None,
                 single_latency_ms_p95=float(np.percentile(lat, 95)) if lat else None,
                 mean_prompt_tokens=n_in / max(len(rows), 1), mean_output_tokens=n_out / max(len(rows), 1), cold_start_s=cold,
                 missing=int(pd.DataFrame(out_rows)[[f"p_{k}" for k in KEYS]].isna().sum().sum()),
                 gpu=torch.cuda.get_device_name(0))
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=1))
    print("VLM_DONE", json.dumps(stats))


def run(items: pd.DataFrame, out: Path, crop: bool, batch: int, gpu_mem: float, timing_n: int) -> None:
    from vllm import LLM, SamplingParams

    t_load = time.time()
    llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=gpu_mem, limit_mm_per_prompt={"image": 1},
              enable_prefix_caching=True, mm_processor_kwargs={"max_pixels": 640 * 640, "min_pixels": 256 * 256}, seed=0)
    cold = time.time() - t_load
    sp = SamplingParams(temperature=0.0, max_tokens=64, logprobs=8)

    def msgs(r):
        roi = (r.roi_x1, r.roi_y1, r.roi_x2, r.roi_y2) if crop and hasattr(r, "roi_x1") and not pd.isna(r.roi_x1) else None
        im = _img(r.path, roi)
        return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _data_url(im)}},
                                             {"type": "text", "text": PROMPT}]}]

    rows = list(items.itertuples(index=False))
    res = []
    # single-request latency on the first timing_n items (batch 1, warm)
    lat = []
    if timing_n:
        llm.chat([msgs(rows[0])], sp, use_tqdm=False)
        for r in rows[:timing_n]:
            m = msgs(r)
            t0 = time.perf_counter()
            llm.chat([m], sp, use_tqdm=False)
            lat.append((time.perf_counter() - t0) * 1000)
    t0 = time.time()
    n_tok_in = n_tok_out = 0
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        outs = llm.chat([msgs(r) for r in chunk], sp, use_tqdm=False)
        for r, o in zip(chunk, outs):
            p = _parse(o)
            n_tok_in += len(o.prompt_token_ids or [])
            n_tok_out += len(o.outputs[0].token_ids)
            res.append({"item_id": r.item_id, **{f"p_{k}": p.get(k, np.nan) for k in KEYS}, "raw": o.outputs[0].text})
        print(s + len(chunk), len(rows), f"{time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0
    pd.DataFrame(res).to_parquet(out)
    stats = dict(model=MODEL, engine="vllm", n=len(rows), batch=batch, wall_s=wall, frames_per_s=len(rows) / max(wall, 1e-9),
                 single_latency_ms_p50=float(np.percentile(lat, 50)) if lat else None,
                 single_latency_ms_p95=float(np.percentile(lat, 95)) if lat else None,
                 mean_prompt_tokens=n_tok_in / max(len(rows), 1), mean_output_tokens=n_tok_out / max(len(rows), 1),
                 cold_start_s=cold, missing=int(pd.DataFrame(res)[[f"p_{k}" for k in KEYS]].isna().sum().sum()))
    import torch

    stats["gpu"] = torch.cuda.get_device_name(0)
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=1))
    print("VLM_DONE", json.dumps(stats))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gpu-mem", type=float, default=0.40)
    ap.add_argument("--timing-n", type=int, default=0)
    ap.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--model", default=MODEL, help="e.g. Qwen/Qwen3-VL-8B-Instruct-FP8 (fits the NVMe budget; L4-deployable)")
    a = ap.parse_args()
    MODEL = a.model
    if a.engine == "hf":
        run_hf(pd.read_parquet(a.items), Path(a.out), a.crop, a.batch, a.timing_n)
    else:
        run(pd.read_parquet(a.items), Path(a.out), a.crop, a.batch, a.gpu_mem, a.timing_n)
