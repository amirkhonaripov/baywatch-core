"""Qwen3-VL reader (second, independent reader). Used (a) to cross-check the verified real-UZ test labels and (b) for pseudo-labels.

  python -m lanes.ident.vlm_reader labels --out lanes/ident/labels/uz_real_test_vlm.jsonl      # test crops, upright via our verified rotations
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from lanes.ident import uz_format as uz
from seen.paths import fast_dir

UZD = fast_dir("ident") / "uz_real"
PROMPT = ("This is a close-up of an Uzbekistan vehicle licence plate. Read the characters of the plate number. Format: a 2-digit region code "
          "(in a small box at the left), then either 1 letter + 3 digits + 2 letters, or 3 digits + 3 letters. Letters are Latin A-Z. "
          "Ignore the flag and the small 'UZ'. Answer with the plate characters only, no spaces, upper case (example: 01A123BC). "
          "If the plate is unreadable, answer UNREADABLE.")
MID = "Qwen/Qwen3-VL-4B-Instruct"


class QwenReader:
    def __init__(self, device="cuda:0", model_id=MID):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.proc.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device).eval()

    def read(self, imgs: list[np.ndarray], batch: int = 16, min_w: int = 256) -> list[str]:
        out = []
        for i in range(0, len(imgs), batch):
            msgs = []
            for img in imgs[i:i + batch]:
                pil = Image.fromarray(np.ascontiguousarray(img))
                if pil.width < min_w:
                    s = min_w / pil.width
                    pil = pil.resize((min_w, max(32, int(pil.height * s))), Image.BICUBIC)
                msgs.append([{"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": PROMPT}]}])
            inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(**inp, max_new_tokens=16, do_sample=False)
            txt = self.proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            out += [re.sub(r"[^0-9A-Z]", "", t.upper()) for t in txt]
            print("vlm", len(out), "/", len(imgs), flush=True)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["labels"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=str(Path(__file__).parent / "labels" / "uz_real_test_labels.jsonl"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.labels).read_text().splitlines()]
    imgs = []
    for r in rows:
        im = cv2.cvtColor(cv2.imread(str(UZD / "crops" / r["crop"])), cv2.COLOR_BGR2RGB)
        imgs.append(np.rot90(im, r["rot"] // 90))
    rd = QwenReader(a.device)
    res = rd.read(imgs)
    agree = 0
    with open(a.out, "w") as f:
        for r, t in zip(rows, res):
            ok = r["readable"] and t == r["text"]
            agree += ok
            f.write(json.dumps({"idx": r["idx"], "label": r["text"], "vlm": t, "agree": bool(ok), "readable": r["readable"]}) + "\n")
    n = sum(r["readable"] for r in rows)
    print(f"VLM agrees with the verified label on {agree}/{n} readable crops")


if __name__ == "__main__":
    main()
