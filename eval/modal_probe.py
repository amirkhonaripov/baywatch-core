"""Measure the *deployed* Seen API on Modal (L4) with a handful of photo requests: cold vs warm, end to end.

    python -m eval.modal_probe --url https://sanjar--seen-api.modal.run --n 8 --images-dir <dir of jury photos>

Budget-aware (demo budget $30): one /health call, then ``n`` sequential /analyze/image calls with real jury photos
(vehicle scenes). The first call after >= 2.5 min of idle is normally a cold start (the app scales to zero; the
deployed scaledown window is 120 s). Reported as SpeedResult rows in docs/lanes/eval_speed_modal.json:
  - client end-to-end (from this machine, includes upload and internet), first call and warm calls
  - server-side pipeline total (AnalysisResult.timings_ms.total) on the L4 = the GPU-side cost driver
Never sends anything but public Commons photos from the jury set.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from eval.stats import percentiles

REPO = Path(__file__).resolve().parents[1]


def _ctx():
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _get(url: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "SeenEval/0.1"}), timeout=timeout,
                                context=_ctx()) as r:
        return json.loads(r.read())


def _post_image(url: str, path: Path, timeout: int = 200) -> dict:
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": f"multipart/form-data; boundary={b}", "User-Agent": "SeenEval/0.1"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
        return json.loads(r.read())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", default=str(REPO))
    a = ap.parse_args(argv)
    from eval.jury.common import jury_items

    base = a.url.rstrip("/")
    items = [i for i in jury_items() if i["kind"] == "image" and i["expected"]["vehicle"] and not i["expected"]["hard"]]
    paths = [Path(a.images_dir) / i["file"] for i in items if (Path(a.images_dir) / i["file"]).exists()][: a.n]
    t0 = time.time()
    health = _get(f"{base}/health")
    health_s = time.time() - t0
    calls = []
    for p in paths:
        t = time.perf_counter()
        try:
            res = _post_image(f"{base}/analyze/image", p)
            calls.append({"file": p.name, "client_ms": (time.perf_counter() - t) * 1000,
                          "server_total_ms": res.get("timings_ms", {}).get("total"),
                          "modal_roundtrip_ms": res.get("timings_ms", {}).get("modal_roundtrip"),
                          "scene": res.get("route", {}).get("scene"), "visit": bool(res.get("visit")),
                          "model_versions": res.get("model_versions")})
        except Exception as e:
            calls.append({"file": p.name, "client_ms": (time.perf_counter() - t) * 1000, "error": f"{type(e).__name__}: {e}"})
        print(json.dumps(calls[-1])[:300], flush=True)
    ok = [c for c in calls if "error" not in c]
    hw = f"NVIDIA {health.get('gpu', {}).get('type', 'L4')} (Modal, deployed {base.split('//')[-1]})"
    rows = []
    if ok:
        first, warm = ok[0], ok[1:]
        cw = percentiles([c["client_ms"] for c in warm]) if warm else None
        sv = percentiles([c["server_total_ms"] for c in warm if c.get("server_total_ms") is not None]) if warm else None
        rows.append({"stage": "deployed API /analyze/image, first call (client end-to-end)", "hardware": hw,
                     "runtime": "as deployed", "batch": 1, "p50_ms": round(first["client_ms"], 1),
                     "cold_start_s": round(first["client_ms"] / 1000, 2),
                     "notes": f"first request after the probe started; health said last GPU load at "
                              f"{health.get('last_gpu_load_at')}, pipeline_load_s {health.get('pipeline_load_s')}. "
                              "Cold only if no container was warm (compare modal_roundtrip of the first and later calls in eval_modal_probe.json). Client = the developer Mac over the public internet."})
        if cw:
            rows.append({"stage": "deployed API /analyze/image, warm (client end-to-end)", "hardware": hw,
                         "runtime": "as deployed", "batch": 1, "p50_ms": round(cw["p50"], 1), "p95_ms": round(cw["p95"], 1),
                         "notes": f"n={cw['n']} sequential photo requests incl. upload and internet round trip"})
        if sv and sv.get("n"):
            rows.append({"stage": "pipeline total (1 frame, analyze()) on Modal, server-side", "hardware": hw,
                         "runtime": "as deployed", "batch": 1, "p50_ms": round(sv["p50"], 1), "p95_ms": round(sv["p95"], 1),
                         "notes": f"AnalysisResult.timings_ms.total on the GPU container, n={sv['n']}; models "
                                  f"{ok[-1].get('model_versions')}"})
    rec = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "url": base, "health": health,
           "health_s": health_s, "calls": calls, "speed": rows}
    d = Path(a.out) / "docs" / "lanes"
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval_speed_modal.json").write_text(json.dumps(rows, indent=2))
    (d / "eval_modal_probe.json").write_text(json.dumps(rec, indent=2, default=str))
    print(json.dumps(rows, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
