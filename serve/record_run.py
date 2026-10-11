"""Save a REAL API result as a "recorded backup run" for the web app (used when the network fails on stage).

Two modes (both talk to a running API, e.g. serve.local through the ssh tunnel):
    python -m serve.record_run --api http://localhost:8008 --image web/public/samples/uz_plate_van.jpg \
        --id uz_plate_van --title "Car photo, Uzbek plate" --work-order oil_filter_change,wash,cabin_filter --simulate
    python -m serve.record_run --api http://localhost:8008 --result-json result.json --id detrac_video \
        --title "Traffic video (browser-sampled)" --input "samples/detrac_MVI_39051_12s.mp4 sampled in the browser"

Writes web/public/samples/recorded/<id>/result.json + evidence/*.jpg and upserts web/public/samples/recorded/index.json.
The result JSON is stored exactly as the API returned it (no edits); the web app labels it "recorded, not live".
Uses only the standard library.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web" / "public"
REC = WEB / "samples" / "recorded"


def post_image(api: str, image: Path, work_order: list[str], simulate: bool) -> dict:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []

    def field(name: str, value: str) -> None:
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())

    ctype = mimetypes.guess_type(image.name)[0] or "image/jpeg"
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{image.name}\"\r\n"
                 f"Content-Type: {ctype}\r\n\r\n".encode() + image.read_bytes() + b"\r\n")
    if work_order:
        field("work_order", json.dumps({"id": "WO-RECORDED-1", "lines": work_order}))
    field("simulate", "true" if simulate else "false")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(f"{api}/analyze/image", data=b"".join(parts), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def evidence_uris(res: dict) -> set[str]:
    uris = {f.get("frame_uri") for f in res.get("frames", [])}
    for v in (res.get("visit"), res.get("simulated_continuation")):
        for e in (v or {}).get("events", []):
            uris.add((e.get("evidence") or {}).get("frame_uri"))
    return {u for u in uris if u and u.startswith("evidence/")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", required=True)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--result-json", type=Path)
    ap.add_argument("--id", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--input", default=None, help="human description of the input")
    ap.add_argument("--input-licence", default=None)
    ap.add_argument("--captured-with", default="serve.local on the GPU box (RTX PRO 5000, GPU1), baseline pretrained models")
    ap.add_argument("--work-order", default="")
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()

    api = args.api.rstrip("/")
    if args.image:
        res = post_image(api, args.image, [c for c in args.work_order.split(",") if c], args.simulate)
    elif args.result_json:
        res = json.loads(args.result_json.read_text())
    else:
        raise SystemExit("give --image or --result-json")

    out = REC / args.id
    (out / "evidence").mkdir(parents=True, exist_ok=True)
    for uri in sorted(evidence_uris(res)):
        with urllib.request.urlopen(f"{api}/{uri}", timeout=60) as r:
            (out / uri).write_bytes(r.read())
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False))

    idx_path = REC / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else []
    meta = {
        "id": args.id,
        "title": args.title,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "captured_with": args.captured_with,
        "input": args.input or (f"samples/{args.image.name}" if args.image else "browser upload"),
        "input_licence": args.input_licence,
        "file": f"samples/recorded/{args.id}/result.json",
        "base": f"samples/recorded/{args.id}/",
    }
    idx = [m for m in idx if m["id"] != args.id] + [meta]
    idx_path.write_text(json.dumps(idx, indent=1, ensure_ascii=False) + "\n")
    print(f"recorded {args.id}: {len(evidence_uris(res))} evidence images, visit={bool(res.get('visit'))} -> {out}")


if __name__ == "__main__":
    main()
