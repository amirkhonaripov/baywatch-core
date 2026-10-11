"""Merge every runs/flow/*/component_results.json (pulled from the box) + runs/flow/*/speed_*.json into docs/lanes/flow_results.json.

    rsync -a --include='*/' --include='component_results.json' --include='speed_*.json' --include='table.md' --include='*.json' --exclude='*' \
        rtx:/mnt/bigdrive/amirkhon/seen/runs/flow/ <local runs dir>
    python -m lanes.flow.collect <local runs dir>
Real and synthetic results are never blended: ``data_kind`` is kept per entry (all flow results are real data).
"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(__file__).resolve().parents[2] / "docs" / "lanes" / "flow_results.json"
comps, speed = [], []
SKIP = {"router_so400m", "blur_smoke", "ft_ndispark", "zs_detrac_partial"}  # ft_ndispark pools NDISPark train images the fine-tune saw; others are superseded
seen_ids = set()
for f in sorted(root.glob("*/component_results.json")):
    if f.parent.name in SKIP:
        continue
    for c in json.loads(f.read_text()):
        if c["id"] not in seen_ids:  # re-evaluations of the same model on the same data repeat ids across experiment folders
            seen_ids.add(c["id"])
            comps.append(c)
for f in sorted(root.glob("*/speed_*.json")):
    speed += [s for s in json.loads(f.read_text()) if s.get("stage") != "parity"]
ids = [c["id"] for c in comps]
assert len(ids) == len(set(ids)), "duplicate component ids"
out.write_text(json.dumps({"components": comps, "speed": speed}, indent=1))
print("wrote", out, len(comps), "components,", len(speed), "speed rows")
