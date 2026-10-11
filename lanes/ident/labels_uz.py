"""Real-UZ OCR test labels: apply the visual-verification corrections, write the test pack, build the human spot-check page.

Provenance (declared in docs/lanes/ident_notes.md): every crop was pre-labelled by two independent readers (pretrained fast-plate-ocr
cct-s-v2-global with a flag-colour orientation cue; Qwen3-VL-4B-Instruct) and then verified crop by crop by Claude (an AI) on contact sheets;
corrections live in lanes/ident/labels/uz_real_test_corrections.json. The labels are NOT yet human-verified: docs/lanes/ident_uz_review.html
is the page for the human spot-check (>= 20 % of crops).

  python -m lanes.ident.labels_uz build   -> $SEEN_FAST/ident/ocr/test_uz_real.pkl (upright crops, verified text) and test_uz_real_raw.pkl (raw orientation)
                                             + lanes/ident/labels/uz_real_test_labels.jsonl (final labels, one row per crop, incl. proposals)
  python -m lanes.ident.labels_uz html    -> docs/lanes/ident_uz_review.html (static, base64 thumbnails)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from lanes.ident import uz_format as uz
from lanes.ident.data.pack import save_test_pack
from seen.paths import fast_dir

FAST = fast_dir("ident")
UZ = FAST / "uz_real"
LAB = Path(__file__).parent / "labels"


def load_items():
    return [json.loads(l) for l in (UZ / "prelabel_test.jsonl").read_text().splitlines()]


def final_rows():
    items = load_items()
    corr = json.loads((LAB / "uz_real_test_corrections.json").read_text())
    vf = UZ / "test_vlm.jsonl"
    vlm = {r["idx"]: r["vlm"] for r in map(json.loads, vf.read_text().splitlines())} if vf.exists() else {}
    for i, it in enumerate(items):
        if i in vlm:
            it["vlm_text"] = vlm[i]
    rows = []
    for i, it in enumerate(items):
        c = corr.get(str(i), {})
        rot = (it["rot"] + int(c.get("rot_add", 0))) % 360
        prop_ocr = uz.normalise(it.get("ocr_text", ""))
        prop_vlm = uz.normalise(it.get("vlm_text", ""))
        text = uz.normalise(c.get("text", "")) if "text" in c else (prop_vlm if c.get("use") == "vlm" else prop_ocr)
        rows.append({"idx": i, "id": it["id"], "crop": it["crop"], "rot": rot, "rot_proposed": it["rot"], "text": text, "ocr_proposal": prop_ocr, "vlm_proposal": prop_vlm,
                     "changed_vs_ocr": text != prop_ocr, "changed_vs_vlm": (text != prop_vlm) if prop_vlm else None, "rot_changed": rot != it["rot"],
                     "flag": c.get("flag", ""), "ev_green_region": bool(c.get("ev", False)), "readable": c.get("flag", "") not in ("unreadable", "partial"),
                     "verified_by": "claude (AI), pending human spot-check"})
    return rows


def build():
    rows = final_rows()
    LAB.mkdir(parents=True, exist_ok=True)
    (LAB / "uz_real_test_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    imgs_up, imgs_raw, texts, regs, srcs = [], [], [], [], []
    for r in rows:
        if not r["readable"]:
            continue
        raw = cv2.cvtColor(cv2.imread(str(UZ / "crops" / r["crop"])), cv2.COLOR_BGR2RGB)
        up = np.ascontiguousarray(np.rot90(raw, r["rot"] // 90))
        imgs_up.append(up)
        imgs_raw.append(raw)
        texts.append(r["text"])
        p = uz.parse(r["text"])
        regs.append(uz.UZ_EV if (r["ev_green_region"] and p.valid) else p.format if p.valid else "unknown")
        srcs.append(r["id"])
    save_test_pack("test_uz_real", imgs_up, texts, regs, srcs)
    save_test_pack("test_uz_real_raw", imgs_raw, texts, regs, srcs)
    n = len(rows)
    print(f"{n} crops, {len(texts)} readable; valid UZ layout: {sum(uz.parse(t).valid for t in texts)}; "
          f"OCR proposal changed: {sum(r['changed_vs_ocr'] for r in rows)}; rot changed: {sum(r['rot_changed'] for r in rows)}")


def html():
    rows = final_rows()
    items = []
    for r in rows:
        raw = cv2.cvtColor(cv2.imread(str(UZ / "crops" / r["crop"])), cv2.COLOR_BGR2RGB)
        up = np.ascontiguousarray(np.rot90(raw, r["rot"] // 90))
        im = Image.fromarray(up)
        im.thumbnail((300, 90))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        items.append((r, base64.b64encode(buf.getvalue()).decode()))
    cells = []
    for r, b64 in items:
        flag = f" <i>[{r['flag']}]</i>" if r["flag"] else ""
        cells.append(f'<div class="c"><img src="data:image/jpeg;base64,{b64}" alt="crop {r["idx"]}"><div><b>#{r["idx"]}</b> {uz.display(r["text"]) or "(none)"}{flag}</div>'
                     f'<div class="s">OCR: {r["ocr_proposal"]} | VLM: {r["vlm_proposal"] or "-"}</div></div>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Uzbek plate OCR test labels (spot-check)</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#111}} .g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:10px}}
.c{{border:1px solid #ccc;padding:6px;border-radius:6px}} .c img{{display:block;margin-bottom:4px;background:#eee}} .s{{color:#555;font-size:12px;font-family:monospace}}
@media (prefers-color-scheme:dark){{body{{background:#111;color:#eee}} .c{{border-color:#444}} .s{{color:#aaa}}}}</style></head><body>
<h1>Real Uzbek plate OCR test set: labels to spot-check</h1>
<p><b>Internal review page, not for the deck or UI</b> (real plates of private vehicles). {len(rows)} crops from the held-out originals of the Roboflow "Uzbek_Plates" set (CC BY 4.0).
Crops are shown upright. The bold text is the label used for the accuracy numbers. Labels were pre-labelled by two models (fast-plate-ocr cct-s-v2-global and Qwen3-VL-4B, shown in grey)
and <b>verified by Claude (an AI) crop by crop; they are pending human spot-check</b>. Please check at least 20 % (60 crops) and report disagreements.</p>
<div class="g">{''.join(cells)}</div></body></html>"""
    out = Path("docs/lanes/ident_uz_review.html")
    out.write_text(page)
    print("wrote", out, len(page) // 1024, "KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "html"])
    a = ap.parse_args()
    {"build": build, "html": html}[a.cmd]()
