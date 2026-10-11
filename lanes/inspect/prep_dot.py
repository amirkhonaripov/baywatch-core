"""Prepare DOT-code data (lane D, task 3b).

1. Character detector  : Roboflow TyreAI `tyreai/ocr-yphpv` v3 (CC BY label, 4,054 images, 39 character classes, close-ups of
   embossed sidewall text). Pool train+valid, re-split 80/5/15 (train/val/test) by pHash cluster so near-duplicates do not
   straddle splits.
2. DOT-region detector : single class "dot_code". Train on Roboflow `dots-tracking/tire-dot-finder` (83) + the 'dot' class of
   `kumbhar-saket/tyre-sidewall-text-label` (train split only); TEST on `subhams-workspace-l47ut/dot-code-6shg0` (97, a different
   uploader = cross-source). Near-duplicates of test images are removed from training.

Run: python -P -m lanes.inspect.prep_dot
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from lanes.inspect import common as C
from seen.paths import RAW

OUT = C.FAST_I / "dot"                 # NVMe (the HDD copy under data/prep/inspect/dot is the pre-migration original)


def _split_dirs(root: Path) -> list[tuple[str, Path, Path]]:
    out = []
    for s in ["train", "valid", "test"]:
        if (root / s / "images").exists():
            out.append((s, root / s / "images", root / s / "labels"))
    return out


def prep_chars() -> dict:
    root = RAW / "inspect_rf_tyreai_ocr"
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    items = []
    for s, idir, ldir in _split_dirs(root):
        for ip in C.list_images(idir):
            lp = ldir / (ip.stem + ".txt")
            if lp.exists():
                items.append((ip, lp))
    cl = C.group_near_dups([str(i[0]) for i in items])
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for (ip, lp), c in zip(items, cl):
        u = C.seeded_fraction(f"dotchars-{c}")
        splits["test" if u < 0.15 else "val" if u < 0.20 else "train"].append((ip, lp))
    for s, lst in splits.items():
        di, dl = C.ensure(OUT / "chars" / "images" / s), C.ensure(OUT / "chars" / "labels" / s)
        for ip, lp in lst:
            shutil.copy(ip, di / ip.name)
            shutil.copy(lp, dl / lp.name)
    (OUT / "chars" / "data.yaml").write_text(
        f"path: {OUT / 'chars'}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {json.dumps(str(n))}\n" for i, n in enumerate(names)))
    (OUT / "chars" / "train_files.txt").write_text("\n".join(str(i[0]) for i in splits["train"] + splits["val"]))
    return {"n_items": len(items), **{k: len(v) for k, v in splits.items()}, "clusters": len(set(cl))}


def prep_region() -> dict:
    """One-class DOT region detector; yolov8 boxes (detection) -> train on two sources, test on a third."""
    tr_items, te_items = [], []  # (img, label_lines)
    # sources for training
    r1 = RAW / "inspect_rf_dot_finder"
    for s, idir, ldir in _split_dirs(r1):
        for ip in C.list_images(idir):
            lp = ldir / (ip.stem + ".txt")
            if lp.exists():
                tr_items.append((ip, [ln for ln in lp.read_text().splitlines() if ln.strip()], "dot_finder"))
    r2 = RAW / "inspect_rf_sidewall_text"
    names2 = yaml.safe_load((r2 / "data.yaml").read_text())["names"]
    dot_idx = names2.index("dot")
    for s, idir, ldir in _split_dirs(r2):
        if s != "train":
            continue
        for ip in C.list_images(idir):
            lp = ldir / (ip.stem + ".txt")
            if not lp.exists():
                continue
            lines = []
            for ln in lp.read_text().splitlines():
                t = ln.split()
                if len(t) >= 5 and int(t[0]) == dot_idx:
                    lines.append("0 " + " ".join(t[1:5]) if len(t) == 5 else "0 " + " ".join(t[1:]))
            if lines:
                tr_items.append((ip, lines, "sidewall_text"))
    # test source
    r3 = RAW / "inspect_rf_dot_code"
    for s, idir, ldir in _split_dirs(r3):
        for ip in C.list_images(idir):
            lp = ldir / (ip.stem + ".txt")
            if lp.exists():
                te_items.append((ip, [ln for ln in lp.read_text().splitlines() if ln.strip()], "dot_code"))
    tw = C.near_dup_matches([str(i[0]) for i in tr_items], [str(i[0]) for i in te_items])
    tr_items = [i for i in tr_items if str(i[0]) not in tw]
    cl = C.group_near_dups([str(i[0]) for i in tr_items])
    val_idx = {k for k, c in enumerate(cl) if C.seeded_fraction(f"dotreg-{c}") < 0.12}
    for tag, lst in [("train", [i for k, i in enumerate(tr_items) if k not in val_idx]),
                     ("val", [i for k, i in enumerate(tr_items) if k in val_idx]), ("test", te_items)]:
        di, dl = C.ensure(OUT / "region" / "images" / tag), C.ensure(OUT / "region" / "labels" / tag)
        for ip, lines, src in lst:
            stem = f"{src}__{ip.stem}"[:150]
            shutil.copy(ip, di / f"{stem}{ip.suffix}")
            # all region labels become class 0; keep polygons' bounding box only if label has >5 values
            out = []
            for ln in lines:
                t = ln.split()
                if len(t) == 5:
                    out.append("0 " + " ".join(t[1:]))
                elif len(t) > 5:  # polygon -> box
                    v = [float(x) for x in t[1:]]
                    xs, ys = v[0::2], v[1::2]
                    out.append(f"0 {(min(xs)+max(xs))/2:.6f} {(min(ys)+max(ys))/2:.6f} {max(xs)-min(xs):.6f} {max(ys)-min(ys):.6f}")
            (dl / f"{stem}.txt").write_text("\n".join(out))
    (OUT / "region" / "data.yaml").write_text(
        f"path: {OUT / 'region'}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: dot_code\n")
    (OUT / "region" / "train_files.txt").write_text("\n".join(str(i[0]) for i in tr_items))
    return {"train_val": len(tr_items), "test": len(te_items), "removed_train_near_dup_of_test": len(tw)}


def migrate_from_hdd() -> None:
    """One-off: copy the already prepared DOT sets from the HDD workspace to the NVMe and rewrite the yaml paths."""
    src = C.PREP_I / "dot"
    n = C.copy_tree_threaded(src, OUT)
    for sub in ("chars", "region"):
        y = OUT / sub / "data.yaml"
        t = y.read_text().replace(str(src / sub), str(OUT / sub))
        y.write_text(t)
        for f in ("train_files.txt",):
            q = OUT / sub / f
            if q.exists():
                q.write_text(q.read_text().replace(str(RAW), "RAW"))
    print("migrated", n, "files to", OUT)


if __name__ == "__main__":
    import sys
    if "--migrate" in sys.argv:
        migrate_from_hdd()
        raise SystemExit(0)
    info = {"chars": prep_chars(), "region": prep_region()}
    C.write_json(info, OUT / "splits.json")
    print(info)
