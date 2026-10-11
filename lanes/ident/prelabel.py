"""Pre-label real-Uzbek plate crops with two independent readers, then render contact sheets for visual verification.

Reader 1: pretrained fast-plate-ocr cct-s-v2-global (NOT our fine-tuned model -> no anchoring on the model under test),
          with a 4-way rotation search (crops come in arbitrary phone orientation).
Reader 2: Qwen3-VL-4B-Instruct on the upright crop (transformers, bf16, GPU1).
Outputs (data/prep/ident/uz_real/prelabel_<split>.jsonl): id, crop, rot (ccw degrees to make it upright), ocr_text/conf, vlm_text, agree.
Sheets (data/prep/ident/review/<split>_NNN.jpg): 30 crops each, upright, with both proposals under every crop.

  python -m lanes.ident.prelabel --split test --stage read
  python -m lanes.ident.prelabel --split test --stage sheets
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from lanes.ident import uz_format as uz
from lanes.ident.ocr_onnx import PlateOcrOnnx, default_providers
from lanes.ident.reader import PlateReader
from lanes.ident.tools.sheet import font
from seen.paths import RAW, fast_dir

FAST = fast_dir("ident")
UZ = FAST / "uz_real"
REV = FAST / "review"
PROMPT = ("This is a close-up of an Uzbekistan vehicle licence plate. Read the characters of the plate number. Format: a 2-digit region code "
          "(in a small box at the left), then either 1 letter + 3 digits + 2 letters, or 3 digits + 3 letters. Letters are Latin A-Z. "
          "Ignore the flag and the small 'UZ'. Answer with the plate characters only, no spaces, upper case (example: 01A123BC). "
          "If the plate is unreadable, answer UNREADABLE.")


def crops_for(split: str):
    recs = [json.loads(l) for l in (UZ / "records.jsonl").read_text().splitlines()]
    out = []
    for r in recs:
        if r["split"] != split:
            continue
        for c in r.get("crops", []):
            out.append({"id": r["id"], "crop": c, "orig": r["orig"]})
    return out


def blue_right_minus_left(img: np.ndarray) -> float:
    """Orientation cue: the flag/UZ block at the RIGHT of an upright Uzbek plate is blue/white/green; the region box at the left is black (or green for EVs).
    Returns (blue mass in the right 22 %) - (blue mass in the left 22 %), normalised."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] >= 98) & (hsv[..., 0] <= 128) & (hsv[..., 1] > 70) & (hsv[..., 2] > 60)).astype(np.float32)
    w = m.shape[1]
    k = max(1, int(w * 0.22))
    return float(m[:, w - k:].mean() - m[:, :k].mean())


def pick_rotation(img: np.ndarray, fallback: int) -> tuple[int, float]:
    h, w = img.shape[:2]
    ks = [0, 2] if w >= h * 1.15 else [1, 3] if h >= w * 1.15 else [0, 1, 2, 3]
    sc = {k: blue_right_minus_left(np.ascontiguousarray(np.rot90(img, k))) for k in ks}
    best = max(sc, key=sc.get)
    if len(ks) == 2:
        other = [k for k in ks if k != best][0]
        if sc[best] - sc[other] < 0.02:  # no flag cue (grey/IR image): keep the reader's choice if compatible
            fb = (fallback // 90) % 4
            return (fb if fb in ks else best) * 90, 0.0
        return best * 90, sc[best] - sc[other]
    return best * 90, sc[best]


def read_stage(split: str, device: str, use_vlm: bool, limit: int | None, ocr_dir: str | None = None, mode: str = "argmax", tag: str = ""):
    items = crops_for(split)
    if limit:
        items = items[:limit]
    if ocr_dir:
        ocr = PlateOcrOnnx(Path(ocr_dir) / "model.onnx", Path(ocr_dir) / "plate_config.yaml", providers=default_providers(device))
    else:
        ocr = PlateOcrOnnx(RAW / "ident_pretrained" / "cct_s_v2_global.onnx", RAW / "ident_pretrained" / "cct_s_v2_global_plate_config.yaml", providers=default_providers(device))
    rd = PlateReader(ocr, mode=mode, orient=True)
    imgs = [cv2.cvtColor(cv2.imread(str(UZ / "crops" / it["crop"])), cv2.COLOR_BGR2RGB) for it in items]
    res = []
    for i in range(0, len(imgs), 128):
        res += rd.read(imgs[i:i + 128])
    if ocr_dir:  # a Uzbek-trained reader: its own orientation search is reliable, use it as is
        for it, r in zip(items, res):
            it.update({"rot": r.rot, "rot_cue": 0.0, "rot_reader": r.rot, "ocr_text": r.text, "ocr_conf": round(r.conf, 4), "ocr_valid": r.valid})
    else:
        for it, r, im in zip(items, res, imgs):
            rot, cue = pick_rotation(im, r.rot)
            it.update({"rot": rot, "rot_cue": round(cue, 4), "rot_reader": r.rot})
        up = [np.ascontiguousarray(np.rot90(im, it["rot"] // 90)) for it, im in zip(items, imgs)]
        rd2 = PlateReader(ocr, mode="argmax", orient=False)
        res2 = []
        for i in range(0, len(up), 128):
            res2 += rd2.read(up[i:i + 128])
        for it, r in zip(items, res2):
            it.update({"ocr_text": r.text, "ocr_conf": round(r.conf, 4)})
    if use_vlm:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        mid = "Qwen/Qwen3-VL-4B-Instruct"
        proc = AutoProcessor.from_pretrained(mid)
        model = Qwen3VLForConditionalGeneration.from_pretrained(mid, dtype=torch.bfloat16, device_map=device).eval()
        proc.tokenizer.padding_side = "left"
        B = 16
        for i in range(0, len(items), B):
            batch = items[i:i + B]
            msgs = []
            for it, img in zip(batch, imgs[i:i + B]):
                up = np.rot90(img, it["rot"] // 90)
                pil = Image.fromarray(np.ascontiguousarray(up))
                if pil.width < 256:  # upscale small crops so the vision encoder sees enough pixels
                    s = 256 / pil.width
                    pil = pil.resize((256, max(32, int(pil.height * s))), Image.BICUBIC)
                msgs.append([{"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": PROMPT}]}])
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=16, do_sample=False)
            txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            for it, t in zip(batch, txt):
                t = re.sub(r"[^0-9A-Z]", "", t.upper())
                it["vlm_text"] = t
            print("vlm", i + len(batch), "/", len(items), flush=True)
    for it in items:
        it["agree"] = bool(it.get("vlm_text") and uz.normalise(it["ocr_text"]) == it["vlm_text"])
    out_p = UZ / f"prelabel_{split}{tag}.jsonl"
    out_p.write_text("".join(json.dumps(it) + "\n" for it in items))
    n_ag = sum(it["agree"] for it in items)
    print(f"{split}: {len(items)} crops; readers agree on {n_ag}")


def sheets_stage(split: str, per_sheet: int = 30, cols: int = 3, corr_path: str | None = None, blind: bool = False, prefix: str = ""):
    """Contact sheets (upright crops). With ``corr_path`` the manual rotation corrections are applied; ``blind`` hides the model proposals."""
    items = [json.loads(l) for l in (UZ / f"prelabel_{split}.jsonl").read_text().splitlines()]
    corr = json.loads(Path(corr_path).read_text()) if corr_path else {}
    REV.mkdir(parents=True, exist_ok=True)
    CW, CH, CAP = (780, 150, 22) if blind else (520, 112, 34)
    f = font(14 if blind else 12)
    for s in range(0, len(items), per_sheet):
        chunk = items[s:s + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * CW, rows * (CH + CAP)), "white")
        d = ImageDraw.Draw(sheet)
        for j, it in enumerate(chunk):
            im = cv2.cvtColor(cv2.imread(str(UZ / "crops" / it["crop"])), cv2.COLOR_BGR2RGB)
            rot = (it.get("rot", 0) + int(corr.get(str(s + j), {}).get("rot_add", 0))) % 360
            im = np.ascontiguousarray(np.rot90(im, rot // 90))
            pil = Image.fromarray(im)
            sc = min((CW - 6) / pil.width, CH / pil.height)
            pil = pil.resize((max(8, int(pil.width * sc)), max(8, int(pil.height * sc))), Image.LANCZOS)
            x, y = (j % cols) * CW, (j // cols) * (CH + CAP)
            sheet.paste(pil, (x + 3, y + 2))
            d.rectangle([x, y, x + CW - 1, y + CH + CAP - 1], outline=(200, 200, 200))
            if blind:
                d.text((x + 4, y + CH + 3), f"#{s + j}", fill=(0, 0, 160), font=f)
            else:
                d.text((x + 4, y + CH + 1), f"#{s + j}  O:{it['ocr_text']}", fill=(0, 0, 160), font=f)
                d.text((x + 4, y + CH + 16), f"     V:{it.get('vlm_text', '')}", fill=(160, 0, 0), font=f)
        sheet.save(REV / f"{prefix}{split}_{s // per_sheet:03d}.jpg", quality=92)
    print("sheets:", (len(items) + per_sheet - 1) // per_sheet)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--stage", choices=["read", "sheets"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ocr-dir", default=None, help="artifact-style dir (model.onnx + plate_config.yaml) of our fine-tuned OCR; default = pretrained baseline")
    ap.add_argument("--mode", default="argmax", choices=["argmax", "constrained", "auto"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--corr", default=None)
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--per-sheet", type=int, default=30)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    if a.stage == "read":
        read_stage(a.split, a.device, not a.no_vlm, a.limit, a.ocr_dir, a.mode, a.tag)
    else:
        sheets_stage(a.split, per_sheet=a.per_sheet, cols=a.cols, corr_path=a.corr, blind=a.blind, prefix=a.prefix)
