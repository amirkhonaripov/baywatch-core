"""Run the Seen API locally (same FastAPI app as Modal, in-process pipeline).

On the box (GPU1):
    scripts/box.sh --gpu 1 -- python -m serve.local --port 8008
From the Mac:
    ssh -N -L 8008:localhost:8008 rtx      # then http://localhost:8008/health
    cd web && VITE_API_BASE=http://localhost:8008 npm run dev

Evidence JPEGs (already face-blurred by the pipeline) go to $SEEN_EVIDENCE_DIR or <SEEN_ROOT>/runs/serve/evidence
(never into code/, which the next sync deletes).
"""
from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path


def default_evidence_dir() -> Path:
    env = os.environ.get("SEEN_EVIDENCE_DIR")
    if env:
        return Path(env)
    from seen.paths import RUNS

    p = RUNS / "serve" / "evidence"
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except OSError:
        return Path(tempfile.gettempdir()) / "seen-evidence"


def build_app(device: str = "cuda:0", evidence_dir: Path | None = None, cors_origins: list[str] | None = None):
    """Load the pipeline in this (main) thread, then return the FastAPI app."""
    from seen.pipeline import default_pipeline
    from serve.app import LocalBackend, create_app

    backend = LocalBackend(lambda: default_pipeline(device=device), evidence_dir or default_evidence_dir(), eager=False)
    backend.load_sync()
    if backend.health()["status"] != "ok":
        raise SystemExit(f"pipeline failed to load: {backend.health()['error']}")
    logging.getLogger("seen.serve").info("pipeline ready in %.1fs: %s", backend.load_s, backend.health()["model_versions"])
    return create_app(backend, cors_origins=cors_origins)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback only; use an ssh tunnel)")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--device", default="cuda:0", help="torch device passed to Predictor.load()")
    ap.add_argument("--evidence-dir", type=Path, default=None)
    ap.add_argument("--cors-origin", action="append", default=None,
                    help="extra allowed origin (github.io and localhost are always allowed)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    app = build_app(args.device, args.evidence_dir, args.cors_origin)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
