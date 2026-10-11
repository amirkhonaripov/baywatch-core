"""Run Lane G's leakage audit (eval.manifest.leakage_audit) for lane D against OUR held-out manifests (until G adopts them in
eval/manifests/). Needs lanes.inspect.make_manifests to have run.

    python -P -m lanes.inspect.audit --manifests runs/inspect/manifests --out runs/inspect/leakage_audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import eval.manifest as M
from lanes.inspect import common as C

PAIRS = {  # component manifest -> training lists that must not overlap it
    "inspect_damage": ["damage/train_files_A.txt", "damage/train_files_B.txt"],
    "inspect_parts": ["parts/train_files.txt"],
    "inspect_tyre": ["../../../mnt_runs/tyre_dino_lr/train_files.txt"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default=str(C.RUNS_I / "manifests"))
    ap.add_argument("--out", default=str(C.RUNS_I / "leakage_audit.json"))
    a = ap.parse_args()
    M.MANIFEST_DIR = Path(a.manifests)  # our manifests, not eval/manifests
    runs = {
        "inspect_damage": [C.FAST_I / "damage" / "train_files_A.txt", C.FAST_I / "damage" / "train_files_B.txt"],
        "inspect_parts": [C.FAST_I / "parts" / "train_files.txt"],
        "inspect_tyre": [C.RUNS_I / "tyre_dino_lr" / "train_files.txt"],
    }
    from lanes.inspect.prep_dot import OUT as DOT
    runs["inspect_dot"] = [DOT / "chars" / "train_files.txt", DOT / "region" / "train_files.txt"]
    report = {}
    for comp, lists in runs.items():
        for tl in lists:
            if not Path(tl).exists():
                continue
            if comp == "inspect_dot":  # DOT train lists name RAW paths; point them at the staged train images instead
                sub = "chars" if "chars" in str(tl) else "region"
                items = [str(p) for p in sorted((DOT / sub / "images" / "train").glob("*.*"))] + [str(p) for p in sorted((DOT / sub / "images" / "val").glob("*.*"))]
                rep = M.leakage_audit([comp], items)
            else:
                comps = [comp] + (["inspect_damage_A_only"] if comp == "inspect_damage" and tl.name == "train_files_A.txt" else [])
                rep = M.leakage_audit(comps, tl)
            key = f"{comp} vs {Path(tl).parent.name}/{Path(tl).name}"
            report[key] = {"verdict": rep["verdict"], **{k: v for k, v in rep["components"][comp].items() if k != "examples"},
                           "examples": rep["components"][comp]["examples"][:5]}
            if "inspect_damage_A_only" in rep["components"]:
                report[key]["A_only_component"] = {k: v for k, v in rep["components"]["inspect_damage_A_only"].items() if k != "examples"}
            print(key, report[key]["verdict"], {k: report[key][k] for k in ("n_test", "n_train", "path_overlap", "near_dup_strict_train_files", "near_dup_loose_train_files") if k in report[key]} if "n_train" in report[key] else "", flush=True)
    C.write_json(report, Path(a.out))


if __name__ == "__main__":
    main()
