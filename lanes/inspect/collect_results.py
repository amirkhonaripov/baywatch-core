"""Collect every component_results_*.json under runs/inspect/<exp>/ into runs/inspect/component_results.json (a list of
ComponentResult, contracts/metrics.schema.json) and validate required fields.

    python -P -m lanes.inspect.collect_results [--exps dmg_A,dmg_B,...]   then  scripts/sync.sh pull runs/inspect/component_results.json docs/lanes/inspect_results.json
"""
from __future__ import annotations

import argparse
import math

from lanes.inspect import common as C

REQ = ["id", "name", "lane", "data_kind", "dataset", "n", "metrics"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", default="")
    a = ap.parse_args()
    keep = set(filter(None, a.exps.split(",")))
    out, seen = [], set()
    for f in sorted(C.RUNS_I.glob("*/component_results_*.json")):
        if keep and f.parent.name not in keep:
            continue
        for r in C.read_json(f):
            miss = [k for k in REQ if k not in r]
            assert not miss, f"{f}: missing {miss}"
            assert r["lane"] == "inspect" and r["data_kind"] in ("real", "synthetic"), f"{f}: bad lane/data_kind"
            assert r["id"] not in seen, f"duplicate id {r['id']}"
            seen.add(r["id"])
            for m in r["metrics"]:
                assert math.isfinite(m["value"]), f"{r['id']}: non-finite {m['name']}"
                if "ci_low" in m:
                    assert m["ci_low"] <= m["ci_high"] + 1e-12, f"{r['id']}: inverted CI for {m['name']}"
            out.append(r)
    C.write_json(out, C.RUNS_I / "component_results.json")
    print(len(out), "component results ->", C.RUNS_I / "component_results.json")


if __name__ == "__main__":
    main()
