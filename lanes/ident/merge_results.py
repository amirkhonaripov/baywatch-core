"""Merge every ComponentResult list under runs/ident/*/component_results*.json into one file (box) -> copy to docs/lanes/ident_results.json (Mac).

  python -m lanes.ident.merge_results            # writes runs/ident/component_results_all.json
Validates the entries against contracts/metrics.schema.json (jsonschema, if installed) and refuses duplicate ids.
"""
from __future__ import annotations

import json
from pathlib import Path

from seen.paths import RUNS


def main():
    root = RUNS / "ident"
    comps: dict[str, dict] = {}
    for f in sorted(root.glob("*/component_results*.json")):
        if f.parent.name in ("ocr_smoke", "det_smoke"):
            continue
        for c in json.loads(f.read_text()):
            c.setdefault("notes", "")
            c["notes"] = (c["notes"] + f" [run: {f.parent.name}]").strip()
            if c["id"] in comps:
                raise SystemExit(f"duplicate id {c['id']} in {f}")
            comps[c["id"]] = c
    out = root / "component_results_all.json"
    out.write_text(json.dumps(list(comps.values()), indent=1))
    try:
        import jsonschema
        schema = json.loads((Path(__file__).resolve().parents[2] / "contracts" / "metrics.schema.json").read_text())
        sub = {"$schema": schema["$schema"], "$defs": schema["$defs"], "type": "array", "items": {"$ref": "#/$defs/ComponentResult"}}
        jsonschema.validate(list(comps.values()), sub)
        print("schema ok")
    except ImportError:
        print("jsonschema not installed; schema not checked")
    print(len(comps), "component results ->", out)


if __name__ == "__main__":
    main()
