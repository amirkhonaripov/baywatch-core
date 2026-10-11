"""Framework-free helpers shared by serve.app (FastAPI), serve.local and serve.modal_app.

Nothing here imports FastAPI or Modal, so the GPU worker and the CPU web function can both use it.
Every number returned to clients comes from ``seen.pipeline`` -- this module only validates input,
decodes images and serialises the AnalysisResult.
"""
from __future__ import annotations

import io
import json
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Protocol

# ---- limits (keep in sync with web/src/lib/limits.ts and docs/lanes/app_notes.md) -----------------------
MAX_FRAMES = 120  # 60 s at 2 fps
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # per uploaded image
MAX_TOTAL_BYTES = 80 * 1024 * 1024  # per /analyze/frames request
MAX_VIDEO_S = 3600.0  # last timestamp allowed: covers timelapse/CCTV stage footage; cost is bounded by the 120-frame cap
MAX_SIDE = 1920  # server-side downscale of the long side before inference
API_VERSION = "0.1.0"

EVIDENCE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}\.jpg$")
JOB_ID_RE = re.compile(r"^[a-f0-9]{8,40}$")


def limits() -> dict[str, float]:
    return {"max_frames": MAX_FRAMES, "max_image_bytes": MAX_IMAGE_BYTES, "max_total_bytes": MAX_TOTAL_BYTES,
            "max_video_s": MAX_VIDEO_S, "max_side_px": MAX_SIDE}


class InputError(ValueError):
    """Client-side problem with the request (mapped to HTTP 400, or 413 when too large)."""

    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


class Unavailable(RuntimeError):
    """Backend temporarily unable to answer (e.g. GPU cold start exceeded the wait); mapped to HTTP 503."""


# ---- request parsing -------------------------------------------------------------------------------------

def parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def parse_options(zones: str | None, work_order: str | None, simulate: Any) -> dict:
    """Validate the optional JSON form fields against the contracts. Returns JSON-able dicts."""
    from seen.contracts.events import Zone
    from seen.contracts.workorder import WorkOrder

    out: dict[str, Any] = {"zones": None, "work_order": None, "simulate": parse_bool(simulate)}
    if zones not in (None, "", "null", "[]"):
        try:
            raw = json.loads(zones)
            if not isinstance(raw, list):
                raise ValueError("zones must be a JSON array")
            parsed = [Zone.model_validate(z) for z in raw]
        except Exception as e:  # noqa: BLE001
            raise InputError(f"invalid zones: {e}") from e
        for z in parsed:
            pts = z.polygon or []
            if pts and len(pts) < 3:
                raise InputError(f"zone {z.id}: polygon needs at least 3 points")
            for x, y in pts:
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    raise InputError(f"zone {z.id}: polygon points must be normalised to 0..1")
        out["zones"] = [z.model_dump(mode="json", exclude_none=True) for z in parsed]
    if work_order not in (None, "", "null"):
        try:
            wo = WorkOrder.model_validate(json.loads(work_order))
        except Exception as e:  # noqa: BLE001
            raise InputError(f"invalid work_order: {e}") from e
        if len(wo.lines) > 30:
            raise InputError("work_order: at most 30 lines")
        out["work_order"] = wo.model_dump(mode="json", exclude_none=True)
    return out


def parse_timestamps(raw: str, n: int) -> list[float]:
    try:
        ts = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        raise InputError(f"timestamps must be a JSON array of seconds: {e}") from e
    if not isinstance(ts, list) or not all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in ts):
        raise InputError("timestamps must be a JSON array of numbers")
    if len(ts) != n:
        raise InputError(f"got {n} files but {len(ts)} timestamps")
    ts = [float(t) for t in ts]
    if any(t < 0 for t in ts):
        raise InputError("timestamps must be >= 0")
    if any(b < a for a, b in zip(ts, ts[1:])):
        raise InputError("timestamps must be non-decreasing (same order as files)")
    if ts and ts[-1] > MAX_VIDEO_S + 0.5:
        raise InputError(f"video longer than {MAX_VIDEO_S:.0f} s is not accepted (last timestamp {ts[-1]:.1f} s)", 413)
    return ts


_MAGIC = ((b"\xff\xd8\xff", "jpeg"), (b"\x89PNG\r\n\x1a\n", "png"))


def check_image_bytes(b: bytes, label: str = "file") -> str:
    if not b:
        raise InputError(f"{label}: empty upload")
    if len(b) > MAX_IMAGE_BYTES:
        raise InputError(f"{label}: {len(b) / 1e6:.1f} MB is larger than the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit", 413)
    for magic, kind in _MAGIC:
        if b.startswith(magic):
            return kind
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "webp"
    if b[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1", b"ftyphevc"):
        raise InputError(f"{label}: HEIC is not supported; the web app converts photos to JPEG before upload")
    raise InputError(f"{label}: not a JPEG/PNG/WebP image")


def decode_image(b: bytes) -> tuple[Any, tuple[int, int], tuple[int, int]]:
    """Bytes -> (RGB uint8 HxWx3, original (w,h), used (w,h)). Applies EXIF orientation and caps the long side."""
    import numpy as np
    from PIL import Image, ImageOps

    try:
        im = Image.open(io.BytesIO(b))
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise InputError(f"could not decode image: {e}") from e
    orig = im.size
    if max(orig) > MAX_SIDE:
        s = MAX_SIDE / max(orig)
        im = im.resize((max(1, round(orig[0] * s)), max(1, round(orig[1] * s))), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8).copy(), orig, im.size


def encode_jpeg(rgb: Any, quality: int = 85) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# ---- stores ----------------------------------------------------------------------------------------------

class EvidenceStore(Protocol):
    def write(self, name: str, rgb: Any) -> str:
        """Store an already face-blurred RGB frame; return its URI (relative to the API base)."""
        ...

    def read(self, name: str) -> bytes | None: ...


def evidence_uri(name: str) -> str:
    return f"evidence/{name}"


class LocalEvidenceStore:
    """Evidence JPEGs in a local directory (serve.local). Modal uses a Volume-backed store instead."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, rgb: Any) -> str:
        if not EVIDENCE_NAME_RE.match(name):
            raise ValueError(f"bad evidence name {name!r}")
        (self.root / name).write_bytes(encode_jpeg(rgb))
        return evidence_uri(name)

    def read(self, name: str) -> bytes | None:
        if not EVIDENCE_NAME_RE.match(name):
            return None
        p = self.root / name
        return p.read_bytes() if p.is_file() else None


class JobStore(Protocol):
    def get(self, job_id: str) -> dict | None: ...

    def put(self, job_id: str, rec: dict) -> None: ...


class MemoryJobStore:
    """Thread-safe in-memory job store (serve.local). Keeps the most recent ``cap`` jobs."""

    def __init__(self, cap: int = 500):
        self._d: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self.cap = cap

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            rec = self._d.get(job_id)
            return dict(rec) if rec is not None else None

    def put(self, job_id: str, rec: dict) -> None:
        with self._lock:
            self._d[job_id] = dict(rec)
            self._d.move_to_end(job_id)
            while len(self._d) > self.cap:
                self._d.popitem(last=False)


# ---- the one analysis entry point -------------------------------------------------------------------------

ProgressFn = Callable[[float, str], None]


def run_analysis(pipeline: Any, images: list[bytes], timestamps: list[float], opts: dict,
                 evidence: EvidenceStore | None, progress: ProgressFn | None = None) -> dict:
    """Decode frames, run ``pipeline.analyze`` and return the AnalysisResult as a JSON-able dict."""
    from seen.contracts.events import Zone
    from seen.contracts.workorder import WorkOrder
    from seen.predictor import Frame

    t0 = time.perf_counter()
    frames, notes = [], []
    for i, (b, t) in enumerate(zip(images, timestamps)):
        rgb, orig, used = decode_image(b)
        if orig != used and not notes:
            notes.append(f"input downscaled from {orig[0]}x{orig[1]} to {used[0]}x{used[1]} before inference")
        frames.append(Frame(index=i, t_s=float(t), image=rgb))
        if progress and (i % 10 == 0):
            progress(0.05 + 0.15 * (i + 1) / len(images), "decoding")
    decode_ms = (time.perf_counter() - t0) * 1000
    if progress:
        progress(0.2, "running models")
    zones = [Zone.model_validate(z) for z in (opts.get("zones") or [])] or None
    wo = WorkOrder.model_validate(opts["work_order"]) if opts.get("work_order") else None
    res = pipeline.analyze(frames, zones=zones, work_order=wo, simulate=bool(opts.get("simulate")),
                           evidence_writer=evidence.write if evidence is not None else None)
    out = res.model_dump(mode="json", exclude_none=True)
    out.setdefault("timings_ms", {})["decode"] = decode_ms
    out["timings_ms"]["server_total"] = (time.perf_counter() - t0) * 1000
    if notes:
        out["warnings"] = [*(out.get("warnings") or []), *notes]
    if progress:
        progress(1.0, "done")
    return out


def gpu_info() -> dict[str, Any]:
    """Best-effort accelerator description for /health. Never raises and never imports anything: importing torch
    from a request thread while another thread is importing it can deadlock, so only already-loaded modules are used."""
    import sys

    info: dict[str, Any] = {}
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            info["torch"] = str(torch.__version__)
            info["cuda_available"] = bool(torch.cuda.is_available())
            if info["cuda_available"]:
                info["name"] = torch.cuda.get_device_name(0)
        except Exception as e:  # noqa: BLE001
            info["torch_error"] = str(e)[:200]
    ort = sys.modules.get("onnxruntime")
    if ort is not None:
        try:
            info["onnxruntime_providers"] = ort.get_available_providers()
        except Exception:  # noqa: BLE001
            pass
    return info


def load_catalogue_json() -> dict:
    from seen.workorder import load_catalogue

    return load_catalogue().model_dump(mode="json", exclude_none=True)
