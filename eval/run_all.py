"""Regenerate every Lane G output that can be produced right now; skip what is missing and say why.

On the box (GPU1):   scripts/box.sh --gpu 1 -- python -m eval.run_all
From the Mac:        python eval/run_all.py --remote          (sync -> run on box GPU1 -> pull -> rebuild report locally)
Options: --quick (short benchmark), --skip bench,jury,audit,cost,failures,report

Steps
 1. jury media present (fetch if missing) and eval/manifests/jury.jsonl written
 2. leakage audit of every runs/<lane>/<exp>/train_files.txt against all held-out manifests
 3. speed benchmark (eval/bench.py)          needs CUDA
 4. jury set through the full pipeline       needs CUDA
 5. cost model from the measured speed
 6. failure gallery (face-blurred)
 7. report: docs/eval_report.{json,md} + web/public/eval_report.json
Status of every step goes to docs/lanes/eval_run_status.json and into the report.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
BOX_ROOT = "/mnt/bigdrive/amirkhon/seen"


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def step_jury_media(out: Path) -> str:
    """Fetch missing media, stage to NVMe, write the held-out manifest under ``out`` (never into code/ on the box)."""
    from eval.jury.common import jury_items, local_path
    from eval.jury.fetch import fetch, stage_fast, write_manifest

    items = jury_items()
    msg = ""
    if any(not local_path(i).exists() for i in items):
        res = fetch(items)
        if res["failed"]:
            msg = f"partial: {res['present']}/{res['total']} media present; failed {res['failed'][:3]}; "
    staged = stage_fast(items)
    dst = out / "eval" / "manifests" / "jury.jsonl"
    write_manifest(items, dst)
    return msg + f"{len(items)} items; {staged}; manifest -> {dst} (pull into eval/manifests/)"


# Candidate pools other lanes draw from (not necessarily training lists). A jury item found in any of them is
# reported as "seen by a lane" and excluded from the clean jury subset. Globs are relative to SEEN_DATA.
POOL_GLOBS = ["prep/*/router/records_*.jsonl", "raw/ops_commons/meta.csv", "prep/ops/**/records*.jsonl",
              "prep/*/commons*/*.jsonl"]


def _pool_titles(p: Path) -> set[str]:
    import csv

    titles = set()
    try:
        if p.suffix == ".csv":
            for r in csv.DictReader(p.open()):
                t = r.get("title") or ""
                if t:
                    titles.add(t if t.startswith("File:") else "File:" + t)
        else:
            for line in p.read_text().splitlines():
                if line.strip():
                    t = json.loads(line).get("title") or ""
                    if t:
                        titles.add(t if t.startswith("File:") else "File:" + t)
    except Exception:
        pass
    return titles


def step_pools(out: Path) -> str:
    """Exact Commons-title overlap between the jury set and other lanes' data pools (cheap; no image reads)."""
    from eval.jury.common import jury_items
    from seen.paths import DATA, RUNS

    jt = {i["title"]: i["id"] for i in jury_items() if i["kind"] != "video_frame"}
    frames_of = {}
    for i in jury_items():
        if i["kind"] == "video_frame":
            frames_of.setdefault(i["frame_of"], []).append(i["id"])
    rows = []
    for g in POOL_GLOBS:
        for p in sorted(DATA.glob(g)):
            hit = sorted(jt[t] for t in _pool_titles(p) & set(jt))
            hit += [f for h in list(hit) for f in frames_of.get(h, [])]
            rows.append({"pool": str(p.relative_to(DATA)), "jury_ids": sorted(set(hit))})
    d = RUNS / "eval" / "leakage"
    d.mkdir(parents=True, exist_ok=True)
    (d / "jury_pool_overlap.json").write_text(json.dumps(rows, indent=2))
    (out / "docs" / "lanes" / "eval_jury_pool_overlap.json").write_text(json.dumps(rows, indent=2))
    n = len({j for r in rows for j in r["jury_ids"]})
    return f"ok: {len(rows)} pools scanned; {n} jury items appear in some lane pool (excluded from the clean subset)"


def step_adopt(out: Path) -> str:
    """Adopt lane-proposed held-out lists from docs/lanes/<lane>_manifests/*.jsonl into out/eval/manifests/."""
    from eval.manifest import adopt

    files = sorted(REPO.glob("docs/lanes/*_manifests/*.jsonl"))
    if not files:
        return "skipped: no docs/lanes/<lane>_manifests/*.jsonl proposed yet"
    rows = adopt(files, out / "eval" / "manifests")
    (out / "docs" / "lanes" / "eval_adopted_manifests.json").write_text(json.dumps(rows, indent=2))
    return f"ok: {sum(r['status'] == 'adopted' for r in rows)}/{len(rows)} adopted"


def step_audit(out: Path, max_files: int = 30000) -> str:
    from eval.manifest import components, leakage_audit
    from seen.paths import RUNS

    lists = sorted(p for p in RUNS.glob("*/*/train_files.txt") if p.parts[-3] != "eval")
    too_big = []
    for p in list(lists):
        n = sum(1 for _ in p.open())
        if n > max_files:
            too_big.append(f"{p.parts[-3]}/{p.parts[-2]} ({n} files)")
            lists.remove(p)
    comps = components()
    if not lists:
        (out / "docs" / "lanes" / "eval_leakage.json").write_text("[]")
        return "skipped: no runs/<lane>/<exp>/train_files.txt found yet"
    rows = []
    for p in lists:
        rep = leakage_audit(comps, p)
        lane, exp = p.parts[-3], p.parts[-2]
        dst = RUNS / "eval" / "leakage" / f"{lane}_{exp}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(rep, indent=2))
        rows.append({"lane": lane, "exp": exp, "n_train": rep["n_train"], "verdict": rep["verdict"],
                     "components": {c: {k: v for k, v in r.items() if k != "examples"} for c, r in rep["components"].items()},
                     "report": str(dst)})
    (out / "docs" / "lanes" / "eval_leakage.json").write_text(json.dumps(rows, indent=2))
    bad = [f"{r['lane']}/{r['exp']}" for r in rows if r["verdict"] != "PASS"]
    skipped = f"; not audited (> {max_files} files, crop lists): {too_big}" if too_big else ""
    head = f"ok: {len(rows)} training lists audited against {comps}"
    return (f"{head}; FAIL: {bad}" if bad else f"{head}, all PASS") + skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--remote", action="store_true", help="run from the Mac: sync, run on box GPU1, pull, report")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--audit-max-files", type=int, default=30000)
    ap.add_argument("--bench-args", default="", help="extra args for eval.bench, e.g. '--cold-repeats 1 --workers 1,2'")
    a = ap.parse_args(argv)
    if a.remote:
        return remote(a)
    from seen.paths import RUNS

    out = Path(a.out) if a.out else RUNS / "eval" / "latest"
    (out / "docs" / "lanes").mkdir(parents=True, exist_ok=True)
    skip = set(filter(None, a.skip.split(",")))
    status: dict[str, str] = {}
    prev_status: dict = {}
    try:
        prev_status = json.loads((out / "docs" / "lanes" / "eval_run_status.json").read_text())
    except Exception:
        pass
    cuda = _cuda()

    def run(name, fn):
        if name in skip:
            prev = prev_status.get(name)
            status[name] = f"not re-run (--skip); previous run: {prev}" if prev and not str(prev).startswith("not re-run") \
                else (prev or "skipped: --skip")
            return
        t0 = time.time()
        print(f"\n===== {name}", flush=True)
        try:
            status[name] = (fn() or "ok") + f" ({time.time() - t0:.0f}s)"
        except SystemExit as e:
            status[name] = f"FAILED: exit {e.code} ({time.time() - t0:.0f}s)"
        except Exception as e:
            traceback.print_exc()
            status[name] = f"FAILED: {type(e).__name__}: {str(e)[:300]}"
        print(f"----- {name}: {status[name]}", flush=True)

    run("jury_media", lambda: step_jury_media(out))
    run("adopt", lambda: step_adopt(out))
    run("pools", lambda: step_pools(out))
    run("audit", lambda: step_audit(out, a.audit_max_files))
    if cuda:
        from eval import bench, jury_run

        run("jury", lambda: "ok" if jury_run.main(["--out", str(out)]) == 0 else "FAILED")
        bargs = ["--out", str(out)] + (["--quick"] if a.quick else []) + a.bench_args.split()
        run("bench", lambda: "ok" if bench.main(bargs) == 0 else "FAILED")
    else:
        status["bench"] = status["jury"] = "skipped: no CUDA on this machine (run on the box GPU1 or Modal L4)"
    from eval import cost, failures, report

    run("cost", lambda: "ok" if cost.main(["--root", str(out), "--out", str(out)]) == 0 else "FAILED")
    run("failures", lambda: f"{len(failures.collect([REPO, out], out))} cases")
    lanes_found = sorted(p.parts[-2] for p in (REPO / "lanes").glob("*/predict.py"))
    status["lane_predictors_present"] = ", ".join(lanes_found) or "none yet"
    (out / "docs" / "lanes" / "eval_run_status.json").write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cuda": cuda, **status}, indent=2))
    def _report():
        info = report.build([REPO, out], out)
        return (f"ok: {info['components']} components from {info['component_files']}, "
                f"{len(info['problems'])} collection problems; lanes missing: {info['lanes_missing']}")

    run("report", _report)
    (out / "docs" / "lanes" / "eval_run_status.json").write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cuda": cuda, **status}, indent=2))
    print(json.dumps(status, indent=2))
    return 0 if not any(v.startswith("FAILED") for v in status.values()) else 1


def remote(a) -> int:
    """Mac side: push code, run on the box, pull outputs into the repo, rebuild the report with the Mac's lane files."""
    sh = lambda *c: subprocess.run(list(c), cwd=REPO, check=False)  # noqa: E731
    if sh("scripts/sync.sh").returncode != 0:
        print("sync failed (another lane may be syncing); retrying once")
        time.sleep(5)
        if sh("scripts/sync.sh").returncode != 0:
            return 1
    out_box = f"{BOX_ROOT}/runs/eval/latest"
    args = ["scripts/box.sh", "--gpu", a.gpu, "--", "python", "-m", "eval.run_all", "--out", out_box]
    if a.quick:
        args.append("--quick")
    if a.skip:
        args += ["--skip", a.skip]
    if a.bench_args:
        args += ["--bench-args", a.bench_args]
    rc = sh(*args).returncode
    box = os.environ.get("SEEN_BOX", "rtx")
    for sub in ("docs/", "web/", "eval/manifests/"):
        sh("rsync", "-az", f"{box}:{out_box}/{sub}", f"{REPO}/{sub}")
    from eval import report

    report.build([REPO], REPO)
    print(f"box run exit code {rc}; outputs pulled into docs/ and web/public/")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
