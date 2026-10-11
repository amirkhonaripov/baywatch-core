"""FastAPI app implementing contracts/api.openapi.yaml.

The app is backend-agnostic: ``create_app(backend)`` takes anything that implements ``Backend``.
  * serve.local      -> LocalBackend (pipeline in-process, MemoryJobStore, LocalEvidenceStore)
  * serve.modal_app  -> ModalBackend (GPU class via .remote()/.spawn(), modal.Dict jobs, Volume evidence)

Endpoints: GET /health, GET /catalogue, POST /analyze/image, POST /analyze/frames, GET /jobs/{id}, GET /evidence/{name}
Limits (serve.core): <=120 frames, <=10 MB per image, <=60 s of video, <=80 MB per request.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Protocol

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from serve import core

log = logging.getLogger("seen.serve")

DEFAULT_CORS_REGEX = r"https://[A-Za-z0-9-]+\.github\.io|http://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?"


class Backend(Protocol):
    name: str

    def health(self, warm: bool = False) -> dict: ...

    def analyze_image(self, image: bytes, opts: dict) -> dict: ...

    def submit_frames(self, job_id: str, images: list[bytes], timestamps: list[float], opts: dict) -> None: ...

    def job(self, job_id: str) -> dict | None: ...

    def evidence(self, name: str) -> bytes | None: ...


# ---- local backend ----------------------------------------------------------------------------------------

class LocalBackend:
    """Runs the pipeline in this process. One GPU worker thread; requests are serialised on a lock."""

    name = "local"

    def __init__(self, pipeline_factory: Callable[[], Any], evidence_dir: Path, eager: bool = True):
        self._factory = pipeline_factory
        self._pipeline: Any = None
        self._load_error: str | None = None
        self._ready = threading.Event()
        self._gpu = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="seen-job")
        self.jobs = core.MemoryJobStore()
        self.store = core.LocalEvidenceStore(evidence_dir)
        self.loaded_at: float | None = None
        self.load_s: float | None = None
        self._started = eager
        if eager:
            threading.Thread(target=self._load, daemon=True).start()

    def load_sync(self) -> None:
        """Load the pipeline in the calling (main) thread. serve.local does this before starting uvicorn: heavy
        imports (torch, rfdetr, onnxruntime) in a background thread can deadlock with imports in request threads."""
        self._started = True
        self._load()

    def _load(self) -> None:
        t0 = time.perf_counter()
        try:
            self._pipeline = self._factory()
            self.load_s = time.perf_counter() - t0
            self.loaded_at = time.time()
            log.info("pipeline loaded in %.1fs: %s", self.load_s, self._pipeline.model_versions)
        except Exception as e:  # noqa: BLE001
            self._load_error = f"{type(e).__name__}: {e}"
            log.error("pipeline failed to load:\n%s", traceback.format_exc())
        finally:
            self._ready.set()

    def _get(self) -> Any:
        with self._gpu:
            if not self._ready.is_set() and not self._started:
                self._started = True
                threading.Thread(target=self._load, daemon=True).start()
        self._ready.wait(timeout=600)
        if self._pipeline is None:
            raise RuntimeError(f"pipeline not available: {self._load_error or 'still loading'}")
        return self._pipeline

    def health(self, warm: bool = False) -> dict:
        status = "ok" if self._pipeline is not None else ("error" if self._load_error else "loading")
        return {
            "status": status,
            "backend": self.name,
            "api_version": core.API_VERSION,
            "model_versions": self._pipeline.model_versions if self._pipeline is not None else {},
            "gpu": core.gpu_info(),
            "pipeline_load_s": self.load_s,
            "error": self._load_error,
            "limits": core.limits(),
        }

    def analyze_image(self, image: bytes, opts: dict) -> dict:
        pipe = self._get()
        with self._gpu:
            return core.run_analysis(pipe, [image], [0.0], opts, self.store)

    def submit_frames(self, job_id: str, images: list[bytes], timestamps: list[float], opts: dict) -> None:
        self.jobs.put(job_id, {"job_id": job_id, "status": "queued", "progress": 0.0})
        self._pool.submit(self._run_job, job_id, images, timestamps, opts)

    def _run_job(self, job_id: str, images: list[bytes], timestamps: list[float], opts: dict) -> None:
        def progress(p: float, _stage: str) -> None:
            self.jobs.put(job_id, {"job_id": job_id, "status": "running", "progress": round(p, 3)})

        try:
            progress(0.01, "starting")
            pipe = self._get()
            with self._gpu:
                res = core.run_analysis(pipe, images, timestamps, opts, self.store, progress)
            self.jobs.put(job_id, {"job_id": job_id, "status": "done", "progress": 1.0, "result": res})
        except core.InputError as e:
            self.jobs.put(job_id, {"job_id": job_id, "status": "error", "error": str(e)})
        except Exception as e:  # noqa: BLE001
            log.error("job %s failed:\n%s", job_id, traceback.format_exc())
            self.jobs.put(job_id, {"job_id": job_id, "status": "error", "error": f"analysis failed: {type(e).__name__}: {e}"})

    def job(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    def evidence(self, name: str) -> bytes | None:
        return self.store.read(name)


# ---- app --------------------------------------------------------------------------------------------------

def _read_upload(f: UploadFile, label: str) -> bytes:
    # Read at most limit+1 bytes so an oversized part is rejected without loading it all.
    data = f.file.read(core.MAX_IMAGE_BYTES + 1)
    core.check_image_bytes(data, label)
    return data


def create_app(backend: Backend, *, cors_origins: list[str] | None = None, cors_regex: str | None = DEFAULT_CORS_REGEX,
               disabled: bool | None = None) -> FastAPI:
    app = FastAPI(title="Seen API", version=core.API_VERSION,
                  description="Inference API for Seen (see contracts/api.openapi.yaml). Every result carries its source, "
                              "confidence, model and model version.")
    app.add_middleware(CORSMiddleware, allow_origins=cors_origins or [], allow_origin_regex=cors_regex,
                       allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"], max_age=3600)
    kill = disabled if disabled is not None else os.environ.get("SEEN_DISABLED", "0") == "1"

    @app.exception_handler(core.InputError)
    async def _input_error(_req: Request, exc: core.InputError):  # noqa: ANN202
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    def _guard() -> None:
        if kill:
            raise HTTPException(503, "The demo API is paused by the operator (spend limit). Use the recorded backup run.")

    @app.exception_handler(core.Unavailable)
    async def _unavailable(_req: Request, exc: core.Unavailable):  # noqa: ANN202
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.get("/health")
    def health(warm: bool = False) -> dict:
        """warm=1 asks the backend to start a GPU container in the background (Modal; rate-limited)."""
        h = backend.health(warm=warm)
        h["disabled"] = kill
        return h

    @app.get("/catalogue")
    def catalogue() -> dict:
        return core.load_catalogue_json()

    @app.post("/analyze/image")
    def analyze_image(file: UploadFile = File(...), zones: str | None = Form(None), work_order: str | None = Form(None),
                      simulate: str | None = Form(None)) -> JSONResponse:
        _guard()
        data = _read_upload(file, file.filename or "file")
        opts = core.parse_options(zones, work_order, simulate)
        try:
            res = backend.analyze_image(data, opts)
        except (core.InputError, core.Unavailable):
            raise
        except Exception as e:  # noqa: BLE001
            log.error("analyze_image failed:\n%s", traceback.format_exc())
            raise HTTPException(500, f"analysis failed: {type(e).__name__}: {e}") from e
        return JSONResponse(res)

    @app.post("/analyze/frames")
    def analyze_frames(files: list[UploadFile] = File(...), timestamps: str = Form(...), zones: str | None = Form(None),
                       work_order: str | None = Form(None), simulate: str | None = Form(None)) -> dict:
        _guard()
        if not files:
            raise core.InputError("no frames uploaded")
        if len(files) > core.MAX_FRAMES:
            raise core.InputError(f"{len(files)} frames is more than the {core.MAX_FRAMES}-frame limit", 413)
        ts = core.parse_timestamps(timestamps, len(files))
        opts = core.parse_options(zones, work_order, simulate)
        images, total = [], 0
        for i, f in enumerate(files):
            b = _read_upload(f, f"frame {i}")
            total += len(b)
            if total > core.MAX_TOTAL_BYTES:
                raise core.InputError(f"request larger than {core.MAX_TOTAL_BYTES // (1024 * 1024)} MB", 413)
            images.append(b)
        job_id = uuid.uuid4().hex
        try:
            backend.submit_frames(job_id, images, ts, opts)
        except Exception as e:  # noqa: BLE001
            log.error("submit failed:\n%s", traceback.format_exc())
            raise HTTPException(500, f"could not start job: {type(e).__name__}: {e}") from e
        return {"job_id": job_id, "status": "queued", "progress": 0.0}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        if not core.JOB_ID_RE.match(job_id):
            raise HTTPException(404, "unknown job")
        rec = backend.job(job_id)
        if rec is None:
            raise HTTPException(404, "unknown job (jobs expire; local jobs are lost on restart)")
        return {k: v for k, v in rec.items() if k in ("job_id", "status", "progress", "result", "error")}

    @app.get("/evidence/{name}")
    def evidence(name: str) -> Response:
        if not core.EVIDENCE_NAME_RE.match(name):
            raise HTTPException(404, "not found")
        data = backend.evidence(name)
        if data is None:
            raise HTTPException(404, "not found")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400, immutable"})

    return app
