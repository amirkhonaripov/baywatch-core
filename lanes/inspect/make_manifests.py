"""Write the lane-D held-out test sets as eval/manifests-style JSONL (one file per component) for Lane G.

Schema (eval/manifest.py): {id, path, source, dataset, split, phash, label}. `path` is the absolute NVMe path of the staged copy
(the pHash is taken on that copy, as for the leakage audit). split = "test" for every item. One JSONL per component:
  inspect_damage.jsonl   all damage test sets held out from EVERY lane-D model (in-source and cross-source), `source` = test-set name
  inspect_damage_A_only.jsonl  cardd_trainpart: held out only from model A (model B trains on it)
  inspect_parts.jsonl    all parts test sets
  inspect_tyre.jsonl     tyre condition test sets
  inspect_dot.jsonl      DOT character + region test sets

    python -P -m lanes.inspect.make_manifests --out runs/inspect/manifests
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lanes.inspect import common as C


def _damage_like(root: Path, component: str, only: str | None = None, exclude: str | None = None) -> list[dict]:
    rows = []
    for gt in sorted((root / "eval").glob("*/gt.json")):
        name = gt.parent.name
        if (only and name != only) or (exclude and name == exclude):
            continue
        recs = C.read_json(gt)
        hashes = C.hash_with_flip([r["path"] for r in recs])
        for r in recs:
            h = hashes.get(r["path"])
            rows.append({"id": f"inspect_{component}:{name}/{r['id']}", "path": r["path"], "source": f"inspect/{name}", "dataset": name,
                         "split": "test", "phash": h[0] if h else None,
                         "label": [{"cls": g["cls"], "box": [round(v, 1) for v in g["box"]]} for g in r["gt"]]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.RUNS_I / "manifests"))
    a = ap.parse_args()
    out = C.ensure(Path(a.out))
    # cardd_trainpart = CarDD images that model B TRAINS on; it is a held-out test only for model A -> its own manifest
    sets = {"damage": _damage_like(C.FAST_I / "damage", "damage", exclude="cardd_trainpart"),
            "damage_A_only": _damage_like(C.FAST_I / "damage", "damage", only="cardd_trainpart"),
            "parts": _damage_like(C.FAST_I / "parts", "parts")}
    # tyre: preds csv of the tyre run lists the test files + labels
    tyre = []
    for csvf in sorted(C.RUNS_I.glob("tyre_*/preds_*.csv")):
        key = csvf.stem.replace("preds_", "")
        lines = csvf.read_text().splitlines()[1:]
        paths = [ln.split(",")[0] for ln in lines]
        hashes = C.hash_with_flip(paths)
        for ln in lines:
            p, lab, orig, _ = ln.split(",")
            h = hashes.get(p)
            tyre.append({"id": f"inspect_tyre:{key}/{Path(p).name}", "path": p, "source": f"inspect/tyre/{key}", "dataset": key, "split": "test",
                         "phash": h[0] if h else None, "label": lab})
    sets["tyre"] = tyre
    dot = []
    for sub in ("chars", "region"):
        d = C.FAST_I.parent.parent  # not used; DOT prepared data lives on the HDD workspace
    from lanes.inspect.prep_dot import OUT as DOT_OUT
    for sub in ("chars", "region"):
        imgs = sorted((DOT_OUT / sub / "images" / "test").glob("*.*"))
        hashes = C.hash_with_flip([str(p) for p in imgs])
        for p in imgs:
            h = hashes.get(str(p))
            dot.append({"id": f"inspect_dot:{sub}/{p.name}", "path": str(p), "source": f"inspect/dot_{sub}", "dataset": f"dot_{sub}", "split": "test",
                        "phash": h[0] if h else None, "label": (DOT_OUT / sub / "labels" / "test" / (p.stem + ".txt")).read_text().count("\n") + 1})
    sets["dot"] = dot
    for comp, rows in sets.items():
        f = out / f"inspect_{comp}.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(comp, len(rows), "->", f)


if __name__ == "__main__":
    main()
