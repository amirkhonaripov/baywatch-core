"""Seen API on Modal: a CPU web function (FastAPI, uploads/polling/catalogue/evidence) + one GPU class (pipeline).

    modal run    serve/modal_app.py           # smoke test: one GPU call on a tiny image, prints the summary, exits
    modal serve  serve/modal_app.py           # temporary URL while it runs (Ctrl+C stops it)
    modal deploy serve/modal_app.py           # permanent URL https://<workspace>--seen-api.modal.run

Spend controls (env vars read at deploy time):
    SEEN_MODAL_GPU=L4 (or T4)  SEEN_MODAL_MAX_GPU=2  SEEN_MODAL_SCALEDOWN=120  SEEN_DISABLED=1 (kill switch -> HTTP 503)
    min_containers is always 0 (scale to zero). SEEN_MODAL_SNAPSHOT=0 disables the CPU memory snapshot.
State:
    modal.Dict  "seen-jobs"      job records, progress, cached GPU info
    modal.Volume "seen-artifacts" model weights: artifacts/<lane>/<model>/..., cache/{roboflow,fast-plate-ocr,open-image-models,hf}
    modal.Volume "seen-evidence"  face-blurred evidence JPEGs written by the GPU class
Weights are pushed from the box with serve/upload_artifacts.py.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import modal

APP_NAME = os.environ.get("SEEN_MODAL_APP", "seen")
GPU = os.environ.get("SEEN_MODAL_GPU", "L4")
VLM_GPU = os.environ.get("SEEN_MODAL_VLM_GPU", "L4")  # Qwen3-VL-8B bf16 (~17 GB) in its own container
MAX_GPU = int(os.environ.get("SEEN_MODAL_MAX_GPU", "2"))
SCALEDOWN = int(os.environ.get("SEEN_MODAL_SCALEDOWN", "120"))
SNAPSHOT = os.environ.get("SEEN_MODAL_SNAPSHOT", "1") == "1"
DISABLED = os.environ.get("SEEN_DISABLED", "0")
CORS_ORIGINS = os.environ.get("SEEN_CORS_ORIGINS", "")  # comma-separated extra origins; *.github.io + localhost always allowed
TORCH_INDEX = os.environ.get("SEEN_TORCH_INDEX", "https://download.pytorch.org/whl/cu130")
REPO = Path(__file__).resolve().parent.parent

# Pinned to the versions in the box env (env/main) so Modal runs the same code paths we test locally.
TORCH = ["torch==2.14.1", "torchvision==0.29.1"]
GPU_DEPS = [
    "numpy==2.5.2", "pillow==12.3.0", "pydantic==2.14.0", "scipy==1.18.1",
    "opencv-python-headless==5.0.0.93", "onnxruntime-gpu==1.31.0", "onnx==1.23.2",
    "rfdetr==1.11.2", "supervision==0.30.9", "fast-alpr==0.4.0", "fast-plate-ocr==1.1.0", "open-image-models==0.6.0",
    "transformers==5.19.0", "accelerate==1.15.0", "pandas==3.0.6", "pyarrow==26.0.0", "timm==1.0.30", "open-clip-torch==3.3.0", "ultralytics==8.4.175",
    "fastapi==0.143.0",  # serve.core imports nothing from it, but keeps `serve` importable in the GPU image
]
WEB_DEPS = ["fastapi==0.143.0", "starlette==1.7.0", "python-multipart==0.0.32", "pydantic==2.14.0", "pillow==12.3.0", "numpy==2.5.2"]

ART_MOUNT, EV_MOUNT = "/artifacts", "/evidence"
artifacts_vol = modal.Volume.from_name("seen-artifacts", create_if_missing=True)
evidence_vol = modal.Volume.from_name("seen-evidence", create_if_missing=True)
jobs = modal.Dict.from_name("seen-jobs", create_if_missing=True)

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*WEB_DEPS)
    .env({"SEEN_DISABLED": DISABLED, "SEEN_CORS_ORIGINS": CORS_ORIGINS, "SEEN_ROOT": "/tmp/seen"})
    .add_local_python_source("seen", "serve")
    .add_local_dir(REPO / "contracts", "/root/contracts")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(*TORCH, index_url=TORCH_INDEX)
    .uv_pip_install(*GPU_DEPS)
    .env({
        "SEEN_ROOT": "/tmp/seen",  # never write runs/ into the weights volume
        "SEEN_ARTIFACTS": f"{ART_MOUNT}/artifacts",
        "RF_HOME": f"{ART_MOUNT}/cache/roboflow",
        "ROBOFLOW_HOME": f"{ART_MOUNT}/cache/roboflow",
        "HF_HOME": f"{ART_MOUNT}/cache/hf",
        "SEEN_GPU": GPU,
        "SEEN_INSPECTION": os.environ.get("SEEN_INSPECTION", "0"),
    })
    .add_local_python_source("seen", "lanes", "sim", "serve", "eval")
    .add_local_dir(REPO / "contracts", "/root/contracts")
)

app = modal.App(APP_NAME)


def _link_caches() -> None:
    """fast-plate-ocr / open-image-models cache under ~/.cache; point them at the weights volume."""
    home = Path.home() / ".cache"
    home.mkdir(parents=True, exist_ok=True)
    for name in ("fast-plate-ocr", "open-image-models"):
        target = Path(ART_MOUNT) / "cache" / name
        target.mkdir(parents=True, exist_ok=True)
        link = home / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target)


class VolumeEvidenceStore:
    """Evidence JPEGs on the seen-evidence Volume (GPU side). Commit after each analysis."""

    def __init__(self, root: str):
        self.root = Path(root)

    def write(self, name: str, rgb) -> str:
        from serve import core

        if not core.EVIDENCE_NAME_RE.match(name):
            raise ValueError(f"bad evidence name {name!r}")
        (self.root / name).write_bytes(core.encode_jpeg(rgb))
        return core.evidence_uri(name)

    def read(self, name: str) -> bytes | None:
        p = self.root / name
        return p.read_bytes() if p.is_file() else None


@app.cls(image=gpu_image, gpu=GPU, volumes={ART_MOUNT: artifacts_vol, EV_MOUNT: evidence_vol}, min_containers=0,
         max_containers=MAX_GPU, scaledown_window=SCALEDOWN, timeout=300, enable_memory_snapshot=SNAPSHOT)
class Analyzer:
    @modal.enter(snap=True)
    def preload(self) -> None:
        # Heavy imports only (no CUDA init): captured in the CPU memory snapshot to cut cold starts.
        import numpy  # noqa: F401
        import PIL.Image  # noqa: F401
        import torch  # noqa: F401

        import seen.pipeline  # noqa: F401
        import serve.core  # noqa: F401

    @modal.enter(snap=False)
    def load(self) -> None:
        from seen.pipeline import default_pipeline
        from serve import core

        t0 = time.perf_counter()
        _link_caches()
        before = {p for p in Path(ART_MOUNT, "cache").rglob("*") if p.is_file()}
        self.pipeline = default_pipeline(device="cuda:0", vlm=RemoteVLM(), inspection=os.environ.get("SEEN_INSPECTION", "0") == "1")
        VLM().warm.spawn()  # start the VLM container in parallel so the first escalation does not wait for it
        # One throw-away inference so CUDA kernels / ORT sessions are initialised before the first real request.
        import numpy as np

        from seen.predictor import Frame

        self.pipeline.analyze([Frame(index=0, t_s=0.0, image=np.zeros((480, 640, 3), np.uint8))])
        self.load_s = time.perf_counter() - t0
        self.store = VolumeEvidenceStore(EV_MOUNT)
        self._commit_lock = threading.Lock()
        if {p for p in Path(ART_MOUNT, "cache").rglob("*") if p.is_file()} - before:
            artifacts_vol.commit()  # weights downloaded on this cold start: keep them for the next one
        info = {"model_versions": self.pipeline.model_versions, "gpu": core.gpu_info(), "pipeline_load_s": round(self.load_s, 2),
                "loaded_at": time.time()}
        # JSON round-trip: Dict values are pickled, and the torch-free web image must be able to read them.
        self.info_ = json.loads(json.dumps(info, default=str))
        jobs.put("info:gpu", self.info_)

    @modal.method()
    def analyze(self, images: list[bytes], timestamps: list[float], opts: dict, job_id: str | None = None) -> dict:
        from serve import core

        def progress(p: float, stage: str) -> None:
            if job_id:
                jobs.put(f"progress:{job_id}", {"status": "running", "progress": round(p, 3), "stage": stage})

        res = core.run_analysis(self.pipeline, images, timestamps, opts, self.store, progress)
        # The web function reads evidence through the Volume API, which only sees committed files. Commit in the
        # background so the JSON result is not delayed (~0.5 s); /evidence retries for a few seconds meanwhile.
        threading.Thread(target=self._commit, daemon=True).start()
        return res

    def _commit(self) -> None:
        with self._commit_lock:
            evidence_vol.commit()

    @modal.exit()
    def flush(self) -> None:
        if hasattr(self, "_commit_lock"):
            with self._commit_lock:
                evidence_vol.commit()

    @modal.method()
    def info(self) -> dict:
        return self.info_


class RemoteVLM:
    """lanes.ops.cascade.VLMClient that sends JPEG crops to the VLM GPU class (used inside Analyzer)."""

    name = "qwen3-vl-8b-zeroshot"
    version = "Qwen3-VL-8B-Instruct@modal"

    def answer(self, images) -> list[dict]:
        from lanes.ops.cascade import jpeg_bytes

        return VLM().answer.remote([jpeg_bytes(im) for im in images])


@app.cls(image=gpu_image, gpu=VLM_GPU, volumes={ART_MOUNT: artifacts_vol}, min_containers=0, max_containers=1,
         scaledown_window=SCALEDOWN, timeout=600)
class VLM:
    @modal.enter()
    def load(self) -> None:
        from lanes.ops.cascade import LocalQwen

        t0 = time.perf_counter()
        before = {p for p in Path(ART_MOUNT, "cache").rglob("*") if p.is_file()}
        self.q = LocalQwen(device="cuda:0")
        self.q.load()
        self.load_s = time.perf_counter() - t0
        if {p for p in Path(ART_MOUNT, "cache").rglob("*") if p.is_file()} - before:
            artifacts_vol.commit()  # Qwen weights downloaded on first start: keep them on the volume

    @modal.method()
    def warm(self) -> float:
        return self.load_s

    @modal.method()
    def answer(self, jpegs: list[bytes]) -> list[dict]:
        import io

        import numpy as np
        from PIL import Image

        ims = [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in jpegs]
        return self.q.answer(ims) if ims else []


class ModalBackend:
    """serve.app.Backend on Modal primitives (runs inside the CPU web function)."""

    name = "modal"
    IMAGE_WAIT_S = 140  # Modal web endpoints time out synchronous requests at ~150 s

    def health(self, warm: bool = False) -> dict:
        from serve import core

        try:
            info = jobs.get("info:gpu") or {}
        except Exception:  # noqa: BLE001  (unreadable entry from an older deploy)
            info = {}
        if warm:
            last = jobs.get("warm:last") or 0
            if time.time() - last > 60:  # at most one warm-up call per minute
                jobs.put("warm:last", time.time())
                Analyzer().info.spawn()
        return {
            "status": "ok",
            "backend": self.name,
            "api_version": core.API_VERSION,
            "model_versions": info.get("model_versions", {}),
            "gpu": {"type": GPU, **(info.get("gpu") or {})},
            "pipeline_load_s": info.get("pipeline_load_s"),
            "last_gpu_load_at": info.get("loaded_at"),
            "limits": core.limits(),
            "error": None,
        }

    def analyze_image(self, image: bytes, opts: dict) -> dict:
        from serve import core

        t0 = time.perf_counter()
        call = Analyzer().analyze.spawn([image], [0.0], opts, None)
        try:
            res = call.get(timeout=self.IMAGE_WAIT_S)
            res["timings_ms"]["modal_roundtrip"] = (time.perf_counter() - t0) * 1000
            return res
        except TimeoutError as e:  # builtin and modal.exception.TimeoutError
            raise core.Unavailable("the GPU did not answer within 140 s (cold start); please retry") from e

    def submit_frames(self, job_id: str, images: list[bytes], timestamps: list[float], opts: dict) -> None:
        jobs.put(f"job:{job_id}", {"job_id": job_id, "status": "queued", "created": time.time()})
        call = Analyzer().analyze.spawn(images, timestamps, opts, job_id)
        jobs.put(f"job:{job_id}", {"job_id": job_id, "status": "queued", "created": time.time(), "call_id": call.object_id})

    def job(self, job_id: str) -> dict | None:
        rec = jobs.get(f"job:{job_id}")
        if rec is None:
            return None
        if rec.get("status") in ("done", "error") or "call_id" not in rec:
            return rec
        call = modal.FunctionCall.from_id(rec["call_id"])
        try:
            res = call.get(timeout=0)
        except modal.exception.OutputExpiredError:
            rec = {"job_id": job_id, "status": "error", "error": "result expired"}
            jobs.put(f"job:{job_id}", rec)
            return rec
        except TimeoutError:
            p = jobs.get(f"progress:{job_id}")
            if p:
                return {"job_id": job_id, "status": "running", "progress": p.get("progress", 0.0)}
            return {"job_id": job_id, "status": "queued", "progress": 0.0}
        except Exception as e:  # noqa: BLE001  (remote exception re-raised here)
            msg = str(e) if e.__class__.__name__ == "InputError" else f"analysis failed: {type(e).__name__}: {e}"
            rec = {"job_id": job_id, "status": "error", "error": msg}
            jobs.put(f"job:{job_id}", rec)
            return rec
        rec = {"job_id": job_id, "status": "done", "progress": 1.0, "result": res}
        jobs.put(f"job:{job_id}", rec)
        return rec

    def evidence(self, name: str) -> bytes | None:
        # The GPU container commits evidence in the background right after returning a result: retry briefly.
        for attempt in range(8):
            try:
                return b"".join(evidence_vol.read_file(name))
            except Exception:  # noqa: BLE001  (not committed yet / missing)
                if attempt < 7:
                    time.sleep(0.5)
        return None


@app.function(image=web_image, min_containers=0, max_containers=2, scaledown_window=300, timeout=600)
@modal.concurrent(max_inputs=32)
@modal.asgi_app(label=f"{APP_NAME}-api")
def api():
    from serve.app import create_app

    extra = [o.strip() for o in os.environ.get("SEEN_CORS_ORIGINS", "").split(",") if o.strip()]
    return create_app(ModalBackend(), cors_origins=extra or None)


@app.local_entrypoint()
def smoke(image: str = "") -> None:
    """One GPU call: `modal run serve/modal_app.py [--image path.jpg]`. Prints route, visit and timings."""
    import io
    import json

    if image:
        data = Path(image).read_bytes()
    else:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (320, 240), (90, 120, 150)).save(buf, format="JPEG")
        data = buf.getvalue()
    t0 = time.perf_counter()
    res = Analyzer().analyze.remote([data], [0.0], {"zones": None, "work_order": None, "simulate": False}, None)
    wall = time.perf_counter() - t0
    print(json.dumps({"route": res["route"], "visit": bool(res.get("visit")), "timings_ms": res["timings_ms"],
                      "model_versions": res["model_versions"], "warnings": res.get("warnings"), "wall_s": round(wall, 2)}, indent=1))
    print(json.dumps(Analyzer().info.remote(), indent=1))
