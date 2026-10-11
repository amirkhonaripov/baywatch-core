"""Effective sample size of every lane-D test set: images vs pHash near-duplicate clusters (Hamming <= 8, incl. mirrored).
Image-level CIs assume independent images; where clusters << images the cluster-level numbers are the honest ones.

    python -P -m lanes.inspect.cluster_stats   -> runs/inspect/cluster_stats.json
"""
from __future__ import annotations

from pathlib import Path

from lanes.inspect import common as C


def main() -> None:
    out = {}
    sets = {}
    for kind in ("damage", "parts"):
        for gt in sorted((C.FAST_I / kind / "eval").glob("*/gt.json")):
            sets[f"{kind}/{gt.parent.name}"] = [r["path"] for r in C.read_json(gt)]
    from lanes.inspect.prep_dot import OUT as DOT
    for sub in ("chars", "region"):
        sets[f"dot/{sub}_test"] = [str(p) for p in sorted((DOT / sub / "images" / "test").glob("*.*"))]
    for csvf in sorted(C.RUNS_I.glob("tyre_dino_lr/preds_*.csv")):
        sets[f"tyre/{csvf.stem.replace('preds_', '')}"] = [ln.split(",")[0] for ln in csvf.read_text().splitlines()[1:]]
    for name, paths in sets.items():
        cl = C.group_near_dups(paths)
        out[name] = {"images": len(paths), "clusters": len(set(cl)), "largest_cluster": max(__import__("collections").Counter(cl).values()) if cl else 0}
        print(name, out[name], flush=True)
    C.write_json(out, C.RUNS_I / "cluster_stats.json")


if __name__ == "__main__":
    main()
