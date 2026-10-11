"""Download Roboflow Universe datasets for lane D with the `roboflow` package.

    scripts/box.sh -- bash -c 'set -a; source /mnt/bigdrive/amirkhon/seen/.env; set +a; python -P -m lanes.inspect.fetch_roboflow dot parts_khpi'

The API key is read from the environment (ROBOFLOW_API_KEY) and is never printed, logged or written anywhere: all output
of the roboflow package is captured and scrubbed before it is shown.
Roboflow licence labels are the uploader's, provenance of the images is usually undocumented (see docs/lanes/inspect_sources.md).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time

from seen.paths import RAW

# target -> (workspace, project, dir name, export format)
TARGETS = {
    "dot": ("tyreai", "ocr-yphpv", "inspect_rf_tyreai_ocr", "yolov8"),
    "dot_finder": ("dots-tracking", "tire-dot-finder", "inspect_rf_dot_finder", "yolov8"),
    "dot_code": ("subhams-workspace-l47ut", "dot-code-6shg0", "inspect_rf_dot_code", "yolov8"),
    "sidewall": ("kumbhar-saket", "tyre-sidewall-text-label", "inspect_rf_sidewall_text", "yolov8"),
    "parts_khpi": ("khpi", "car-segmentation-ga8mc", "inspect_rf_khpi_parts", "yolov8"),
    "cardd_rf": ("car-damage-detection-cardd", "car-damage-severity-detection-cardd", "inspect_rf_cardd", "yolov8"),
}


def scrub(s: str, key: str) -> str:
    return s.replace(key, "***") if key else s


def main(targets: list[str]) -> None:
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        raise SystemExit("ROBOFLOW_API_KEY not in environment (source the box .env first)")
    from roboflow import Roboflow

    rf = Roboflow(api_key=key)
    for t in targets:
        ws, proj, name, fmt = TARGETS[t]
        out = RAW / name
        if (out / ".done").exists():
            print(t, "already done"); continue
        if out.exists() and not any(out.iterdir()):
            out.rmdir()  # roboflow's download() silently does nothing when the target dir already exists
        buf = io.StringIO()
        info = {}
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                p = rf.workspace(ws).project(proj)
                vs = p.versions()
                nums = sorted(int(str(v.version).split("/")[-1]) for v in vs)
                info["versions"] = nums
                cand = nums[::-1] if nums else list(range(12, 0, -1))  # versions() is empty for some Universe projects
                ver = None
                for cv in cand:
                    try:
                        p.version(cv).download(fmt, location=str(out), overwrite=True)
                    except Exception:  # noqa: BLE001
                        continue
                    if out.exists() and sum(1 for _ in out.rglob("*") if _.is_file()) >= 10:
                        ver = cv
                        break
                    if out.exists() and not any(out.iterdir()):
                        out.rmdir()
                if ver is None:
                    raise RuntimeError("no downloadable version found")
            n_files = sum(1 for _ in out.rglob("*") if _.is_file()) if out.exists() else 0
            if n_files < 10:
                raise RuntimeError(f"download produced {n_files} files; roboflow said: {scrub(buf.getvalue()[-300:], key)}")
            info["files"] = n_files
            info.update({"workspace": ws, "project": proj, "version": ver, "format": fmt, "time": time.strftime("%F %T")})
            (out / ".done").write_text(json.dumps(info))
            print(t, "ok", info)
        except Exception as e:  # noqa: BLE001
            print(t, "FAILED", scrub(f"{type(e).__name__}: {e}", key))
        tail = scrub(buf.getvalue(), key)
        print(tail[-600:])


if __name__ == "__main__":
    main(sys.argv[1:])
