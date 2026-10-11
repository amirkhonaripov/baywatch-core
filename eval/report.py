"""Collect every lane's results + Lane G's own measurements into docs/eval_report.{json,md}.

Inputs (all optional; missing ones are listed in the report, never invented):
  docs/lanes/*_results.json        list[ComponentResult] per lane (lanes write these; G's jury components too)
  docs/lanes/eval_speed.json       list[SpeedResult]            (eval/bench.py)
  docs/lanes/eval_cost.json        list[CostLine]               (eval/cost.py)
  docs/lanes/eval_jury.json        JurySet                      (eval/jury_run.py)
  docs/lanes/eval_failures.json    list[FailureCase]            (eval/failures.py)
  docs/lanes/eval_hardware.json    {name: description}          (eval/bench.py)
  docs/lanes/*_sources.md          licence tables per lane

Outputs: docs/eval_report.json (validated against contracts/metrics.schema.json), docs/eval_report.md,
web/public/eval_report.json (+ web/public/lanes/eval_gallery/ copy of the blurred failure gallery).

    python -m eval.report [--root <dir with docs/lanes overrides>] [--out <output root>]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SCHEMA = REPO / "contracts" / "metrics.schema.json"
LANES = ["ident", "flow", "ops", "inspect", "sim", "router", "e2e"]
LANE_DIRS = {"ident": "A (identification)", "flow": "B (flow)", "ops": "C (bay states / operations)",
             "inspect": "D (inspection)", "sim": "E (scenario engine / synthetic)", "router": "router", "e2e": "end-to-end"}
OWN = {"speed": "eval_speed.json", "cost": "eval_cost.json", "jury": "eval_jury.json",
       "failures": "eval_failures.json", "hardware": "eval_hardware.json", "status": "eval_run_status.json",
       "leakage": "eval_leakage.json"}


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text())


def _find(name: str, roots: list[Path]) -> Path | None:
    """Last root wins (``--root`` overrides the repo copy)."""
    hit = None
    for r in roots:
        p = r / "docs" / "lanes" / name
        if p.exists():
            hit = p
    return hit


def _validator():
    import jsonschema

    schema = json.loads(SCHEMA.read_text())
    return schema, jsonschema.Draft202012Validator


def _validate_def(item: Any, defname: str) -> list[str]:
    import jsonschema

    schema, V = _validator()
    sub = {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{defname}"}
    return [e.message for e in V(sub, format_checker=jsonschema.FormatChecker()).iter_errors(item)]


def collect(roots: list[Path]) -> tuple[dict, list[str], dict]:
    """Return (report dict, problems, provenance)."""
    problems: list[str] = []
    prov: dict[str, Any] = {"component_files": {}, "lanes_present": [], "lanes_missing": []}
    comps: list[dict] = []
    files: dict[str, Path] = {}
    for r in roots:
        for p in sorted((r / "docs" / "lanes").glob("*_results.json")):
            files[p.name] = p
    for name, p in sorted(files.items()):
        try:
            data = _load_json(p)
        except Exception as e:
            problems.append(f"{name}: unreadable JSON ({e})")
            continue
        if isinstance(data, dict):
            data = data.get("components", [data])
        ok = 0
        for i, item in enumerate(data):
            errs = _validate_def(item, "ComponentResult")
            if errs:
                problems.append(f"{name}[{i}] ({item.get('id', '?') if isinstance(item, dict) else '?'}): " + "; ".join(errs[:3]))
                continue
            bad = [m["name"] for m in item["metrics"] if not math.isfinite(m["value"])]
            if bad:
                problems.append(f"{name}[{i}] {item['id']}: non-finite metric(s) {bad}")
                continue
            comps.append({**item, "_file": name})
            ok += 1
        prov["component_files"][name] = ok
    present = sorted({c["lane"] for c in comps})
    prov["lanes_present"] = present
    prov["lanes_missing"] = [l for l in ["ident", "flow", "ops", "inspect", "sim"] if l not in present]

    report: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "components": []}
    seen_ids = set()
    for c in comps:
        cid = c["id"]
        if cid in seen_ids:
            problems.append(f"duplicate component id {cid} (from {c['_file']}); kept the first")
            continue
        seen_ids.add(cid)
        report["components"].append({k: v for k, v in c.items() if not k.startswith("_")})

    p = _find(OWN["hardware"], roots)
    if p:
        report["hardware"] = {k: str(v) for k, v in _load_json(p).items()}
    for key, defname in (("speed", "SpeedResult"), ("cost", "CostLine"), ("failures", "FailureCase")):
        if key == "speed":  # eval_speed.json (box) + eval_speed_<where>.json (e.g. _modal)
            found: dict[str, Path] = {}
            for r in roots:
                for q in sorted((r / "docs" / "lanes").glob("eval_speed*.json")):
                    found[q.name] = q
            paths = [found[k] for k in sorted(found)]
        else:
            p = _find(OWN[key], roots)
            paths = [p] if p else []
        if not paths:
            prov[f"{key}_missing"] = True
            continue
        items = []
        for p in paths:
            for i, it in enumerate(_load_json(p)):
                errs = _validate_def(it, defname)
                if errs:
                    problems.append(f"{p.name}[{i}]: " + "; ".join(errs[:3]))
                else:
                    items.append(it)
        report[key] = items
    p = _find("eval_bench_details.json", roots)
    if p:
        try:
            prov["bench_hw"] = _load_json(p).get("details", {}).get("hardware", {})
        except Exception:
            pass
    for key in ("status", "leakage"):
        p = _find(OWN[key], roots)
        if p:
            try:
                prov[key] = _load_json(p)
            except Exception:
                pass
    p = _find(OWN["jury"], roots)
    if p:
        jury = _load_json(p)
        errs = _validate_def(jury, "JurySet")
        if errs:
            problems.append(f"{p.name}: " + "; ".join(errs[:3]))
        else:
            report["jury_set"] = jury
    else:
        prov["jury_missing"] = True
    return report, problems, prov


def validate_report(report: dict) -> list[str]:
    import jsonschema

    schema, V = _validator()
    errs = [f"{list(e.path)}: {e.message}" for e in V(schema, format_checker=jsonschema.FormatChecker()).iter_errors(report)]
    try:
        from seen.contracts.metrics import EvalReport

        EvalReport.model_validate(report)
    except ImportError:
        pass
    except Exception as e:  # pydantic ValidationError
        errs.append(f"pydantic: {e}")
    return errs


# ------------------------------------------------------------------------------------------------ licences
def licence_table(roots: list[Path]) -> dict[str, list[tuple[str, str, str]]]:
    """Parse markdown tables in docs/lanes/*_sources.md: {lane: [(name, licence, used_for)]}."""
    out: dict[str, list[tuple[str, str, str]]] = {}
    files: dict[str, Path] = {}
    for r in roots:
        for p in (r / "docs" / "lanes").glob("*_sources.md"):
            files[p.name] = p
    for name, p in sorted(files.items()):
        lane = name.replace("_sources.md", "")
        rows: list[tuple[str, str, str]] = []
        header: list[str] | None = None
        for line in p.read_text().splitlines():
            if not line.strip().startswith("|"):
                header = None
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if header is None:
                header = [c.lower() for c in cells]
                continue
            if all(set(c) <= set("-: ") for c in cells):
                continue
            def col(*keys):
                for i, h in enumerate(header or []):
                    if any(k in h for k in keys) and i < len(cells):
                        return cells[i]
                return ""
            if any("licen" in h for h in header):
                rows.append((col("name", "dataset", "model", "source"), col("licen"), col("use", "purpose")))
        out[lane] = rows
    return out


# ------------------------------------------------------------------------------------------------ rendering
def _fmt(v: float, unit: str | None) -> str:
    if unit in ("fraction", "ratio") and -1.5 <= v <= 1.5:
        return f"{100 * v:.1f}%"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.3g}"


def fmt_metric(m: dict) -> str:
    u = m.get("unit")
    s = _fmt(m["value"], u)
    if "ci_low" in m and "ci_high" in m:
        s += f" [{_fmt(m['ci_low'], u)}, {_fmt(m['ci_high'], u)}]"
    if u and u not in ("fraction", "ratio"):
        s += f" {u}"
    return s


def _esc(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _confusion_md(conf: dict) -> list[str]:
    labels = conf.get("labels") or []
    mat = conf.get("matrix") or []
    if not labels or not mat:
        return []
    out = ["", "| truth \\ pred | " + " | ".join(map(_esc, labels)) + " |", "|---" * (len(labels) + 1) + "|"]
    for l, row in zip(labels, mat):
        out.append(f"| **{_esc(l)}** | " + " | ".join(str(x) for x in row) + " |")
    return out + [""]


def _components_md(comps: list[dict]) -> list[str]:
    out: list[str] = []
    if not comps:
        return ["_No results in this section yet._", ""]
    by_lane: dict[str, list[dict]] = {}
    for c in comps:
        by_lane.setdefault(c["lane"], []).append(c)
    for lane in [l for l in LANES if l in by_lane]:
        out += [f"### Lane {LANE_DIRS.get(lane, lane)}", ""]
        for c in by_lane[lane]:
            out += [f"#### {_esc(c['name'])} (`{c['id']}`)", ""]
            meta = [f"dataset **{_esc(c['dataset'])}**", f"domain {_esc(c.get('domain') or 'not stated')}",
                    f"split {_esc(c.get('split') or 'not stated')}", f"n = **{c['n']}**",
                    f"model {_esc(c.get('model') or 'not stated')}"]
            out += ["; ".join(meta) + ".", "", "| metric | value [95% CI] |", "|---|---|"]
            out += [f"| {_esc(m['name'])} | {fmt_metric(m)} |" for m in c["metrics"]]
            if c.get("confusion"):
                out += ["", "Confusion matrix (rows = truth):"] + _confusion_md(c["confusion"])
            if c.get("predictions_csv"):
                out += ["", f"Per-sample predictions: `{c['predictions_csv']}`"]
            if c.get("notes"):
                out += ["", f"Notes: {_esc(c['notes'])}"]
            out.append("")
    return out


def _speed_md(speed: list[dict]) -> list[str]:
    if not speed:
        return ["_Not measured yet (run `python -m eval.bench` on the box)._", ""]
    out = ["| stage | hardware | runtime | batch | p50 ms | p95 ms | fps | cold start s | notes |",
           "|---|---|---|---|---|---|---|---|---|"]
    for s in speed:
        f = lambda k, d=1: (f"{s[k]:.{d}f}" if isinstance(s.get(k), (int, float)) else "")  # noqa: E731
        out.append(f"| {_esc(s['stage'])} | {_esc(s['hardware'])} | {_esc(s.get('runtime', ''))} | {s.get('batch', '')} | "
                   f"{f('p50_ms')} | {f('p95_ms')} | {f('fps')} | {f('cold_start_s', 2)} | {_esc(s.get('notes', ''))} |")
    return out + [""]


def _cost_md(cost: list[dict]) -> list[str]:
    if not cost:
        return ["_Not computed yet (run `python -m eval.cost`)._", ""]
    out = ["| item | value | unit | assumptions |", "|---|---|---|---|"]
    for c in cost:
        v = c["value"]
        vs = (f"{v:.6f}".rstrip("0") if abs(v) < 0.01 else f"{v:,.4f}") if abs(v) < 1 else f"{v:,.2f}"
        out.append(f"| {_esc(c['item'])} | {vs} | {_esc(c['unit'])} | {_esc(c.get('assumptions', ''))} |")
    return out + [""]


def _jury_md(j: dict | None) -> list[str]:
    if not j:
        return ["_Jury set not run yet (`python -m eval.jury_run`)._", ""]
    out = [f"n = **{j.get('n')}** inputs (Wikimedia Commons, licence per file in `eval/jury/jury.jsonl`). "
           "Expected behaviour was labelled by the Lane G agent from the images (single annotator).", ""]
    summ = j.get("summary") or {}
    if summ:
        out += ["| summary | value [95% CI] | n |", "|---|---|---|"]
        done = set()
        for k, v in summ.items():
            if k in done or k.endswith(("_ci_low", "_ci_high", "_n")) or "_n[" in k:
                continue
            lo, hi, n = summ.get(f"{k}_ci_low"), summ.get(f"{k}_ci_high"), summ.get(f"{k}_n")
            if k.startswith("scene_recall["):
                n = summ.get(k.replace("scene_recall[", "scene_recall_n["))
            is_frac = lo is not None or k.startswith(("scene_recall", "route_accuracy", "visit_ok", "visit_iff"))
            if isinstance(v, float) and is_frac:
                val = f"{100 * v:.1f}%" + (f" [{100 * lo:.1f}, {100 * hi:.1f}]" if lo is not None else "")
            elif isinstance(v, float):
                val = f"{v:,.1f}"
            else:
                val = str(v)
            out.append(f"| {_esc(k)} | {val} | {n if n is not None else ''} |")
        out.append("")
    oc = j.get("outcomes") or []
    if oc:
        cols = ["id", "category", "kind", "exp_scene", "got_scene", "exp_vehicle", "visit", "exp_plate", "plate_read",
                "route_ok", "visit_ok", "status", "latency_ms"]
        out += ["<details><summary>Per-input outcomes</summary>", "", "| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
        for o in oc:
            row = []
            for c in cols:
                v = o.get(c, "")
                row.append(f"{v:.0f}" if isinstance(v, float) else _esc(v))
            out.append("| " + " | ".join(row) + " |")
        out += ["", "</details>", ""]
    return out


def _failures_md(fails: list[dict]) -> list[str]:
    if not fails:
        return ["_No failure cases collected yet._", ""]
    out = ["All images are face-blurred (seen.blur + an extra YuNet face pass in eval/failures.py) before they are stored.", ""]
    by: dict[str, list[dict]] = {}
    for f in fails:
        by.setdefault(f["component"], []).append(f)
    for comp, items in sorted(by.items()):
        out += [f"### {comp}", ""]
        for f in items:
            img = f"![{_esc(f['id'])}]({f['image_uri']})" if f.get("image_uri") else ""
            cat = f" _{_esc(f['category'])}_" if f.get("category") else ""
            out += [f"- **{_esc(f['id'])}**{cat}: {_esc(f['description'])}", f"  {img}" if img else ""]
        out.append("")
    return out


def render_md(report: dict, problems: list[str], prov: dict, licences: dict) -> str:
    comps = report.get("components", [])
    real = [c for c in comps if c["data_kind"] == "real"]
    synth = [c for c in comps if c["data_kind"] == "synthetic"]
    L = ["# Seen: evaluation report", "",
         f"Generated {report['generated_at']} by `python -m eval.run_all` (Lane G, independent of the trainers). "
         "Every number below is produced by a script in this repo; nothing is typed in by hand. "
         "Real-data and synthetic-data results are kept in separate sections and are never averaged together. "
         "Intervals are 95%: Wilson for proportions, percentile bootstrap over images/clips for everything else.", ""]
    if report.get("hardware"):
        L += ["**Hardware:** " + "; ".join(f"{k}: {v}" for k, v in report["hardware"].items()), ""]
    L += ["## Coverage", "",
          f"- Lanes with results: {', '.join(prov['lanes_present']) or 'none yet'}.",
          f"- Lanes with no results yet: {', '.join(prov['lanes_missing']) or 'none'}.",
          f"- Result files read: " + (", ".join(f"`{k}` ({v} valid)" for k, v in prov["component_files"].items()) or "none") + ".", ""]
    st = prov.get("status")
    if isinstance(st, dict):
        L += ["Last `eval.run_all` steps:", "", "| step | status |", "|---|---|"]
        L += [f"| {_esc(k)} | {_esc(v)} |" for k, v in st.items() if k != "generated_at"] + [""]
    lk = prov.get("leakage")
    if isinstance(lk, list) and lk:
        L += ["Leakage audit (training lists vs held-out manifests, pHash <= 4 = near-identical):", "",
              "| lane/exp | n train | verdict | details |", "|---|---|---|---|"]
        for r in lk:
            det = "; ".join(f"{c}: path {v.get('path_overlap')}, near-dup {v.get('near_dup_strict_train_files')}"
                            for c, v in r.get("components", {}).items())
            L.append(f"| {_esc(r['lane'])}/{_esc(r['exp'])} | {r['n_train']} | {r['verdict']} | {_esc(det)} |")
        L.append("")
    L += ["## 1. Real-data results", "", "Each component is tested on real public data held out by source "
          "(manifests in `eval/manifests/`). Domain is stated per row: most tests are not service-bay CCTV.", ""]
    L += _components_md(real)
    L += ["## 2. Synthetic-data results (rendered or simulated; not evidence of real-world accuracy)", ""]
    L += _components_md(synth)
    L += ["## 3. Speed", ""]
    bh = prov.get("bench_hw") or {}
    if bh:
        L += [f"Box measurement conditions: IO pressure at start {bh.get('io_pressure', 'n/a')}; GPU sharing "
              f"{bh.get('gpu_contention') or 'not recorded for this run (box GPU1 is shared with Blender renders and other lanes; treat latency as an upper bound and cameras per GPU as a lower bound)'}.", ""]
    L += ["Runtime rows compare backends within one run; across runs the shared GPU makes absolute numbers vary. "
          "Edge-box (RTX 4060 Ti) numbers in the cost table are spec-ratio estimates, not measurements.", ""]
    L += _speed_md(report.get("speed", []))
    L += ["## 4. Cost", ""] + _cost_md(report.get("cost", []))
    L += ["## 5. Jury-style test set", ""] + _jury_md(report.get("jury_set"))
    L += ["## 6. Failure gallery", ""] + _failures_md(report.get("failures", []))
    L += ["## 7. Datasets and licences", "",
          "Several datasets are non-commercial or research-only: fine for this prototype, a paid pilot needs "
          "re-licensing or own pilot data. Public datasets contain real faces and plates; no identifiable face is "
          "shown in this report.", ""]
    ds = sorted({(c["lane"], c["dataset"], c["data_kind"]) for c in comps})
    if ds:
        L += ["| lane | dataset (as reported) | kind |", "|---|---|---|"] + [f"| {a} | {_esc(b)} | {k} |" for a, b, k in ds] + [""]
    for lane, rows in sorted(licences.items()):
        L += [f"**{lane}** (`docs/lanes/{lane}_sources.md`):", ""]
        if rows:
            L += ["| name | licence | used for |", "|---|---|---|"] + [f"| {_esc(a)} | {_esc(b)} | {_esc(c)} |" for a, b, c in rows]
        else:
            L += ["_No licence table found in this file._"]
        L.append("")
    if not licences:
        L += ["_No `docs/lanes/*_sources.md` files yet._", ""]
    if problems:
        L += ["## 8. Collection problems (entries skipped)", ""] + [f"- {_esc(p)}" for p in problems] + [""]
    return "\n".join(L).rstrip() + "\n"


def build(roots: list[Path], out: Path, web: bool = True) -> dict:
    report, problems, prov = collect(roots)
    errs = validate_report(report)
    if errs:
        raise SystemExit("eval_report.json fails the schema:\n" + "\n".join(errs[:20]))
    lic = licence_table(roots)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    js = json.dumps(report, indent=2, ensure_ascii=False)
    (out / "docs" / "eval_report.json").write_text(js)
    (out / "docs" / "eval_report.md").write_text(render_md(report, problems, prov, lic))
    if web:
        wp = out / "web" / "public"
        wp.mkdir(parents=True, exist_ok=True)
        (wp / "eval_report.json").write_text(js)
        gal = None
        for r in roots:
            if (r / "docs" / "lanes" / "eval_gallery").is_dir():
                gal = r / "docs" / "lanes" / "eval_gallery"
        if gal is not None:
            dst = wp / "lanes" / "eval_gallery"
            if dst.resolve() != gal.resolve():
                shutil.copytree(gal, dst, dirs_exist_ok=True)
    return {"components": len(report["components"]), "problems": problems, **prov}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=[], help="extra roots whose docs/lanes override the repo's")
    ap.add_argument("--out", default=str(REPO))
    ap.add_argument("--no-web", action="store_true")
    a = ap.parse_args(argv)
    roots = [REPO] + [Path(r) for r in a.root]
    info = build(roots, Path(a.out), web=not a.no_web)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
