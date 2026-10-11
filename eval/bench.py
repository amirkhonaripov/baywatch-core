"""Speed benchmark: per-stage latency, runtimes (PyTorch / ONNX Runtime CUDA / TensorRT FP16), cold vs warm, and
cameras per GPU from N concurrent simulated camera streams. Lane G; never imports lane training code.

    # box GPU1 (writes runs/eval/bench/<stamp>/bench.json and docs/lanes/eval_speed.json under --out)
    scripts/box.sh --gpu 1 -- python -m eval.bench --out /mnt/bigdrive/amirkhon/seen/runs/eval/latest
    # quick smoke run
    python -m eval.bench --quick
    # Modal L4 later: same module, label the hardware
    python -m eval.bench --hardware-label "NVIDIA L4 (Modal)" --out /tmp/out

What is measured
- Each predictor through its ``Predictor.predict`` (what the pipeline calls), on real images from the jury set
  resized to 1280x720 (camera-like), warm p50/p95 over many calls, after warm-up.
- The full pipeline ``seen.pipeline.default_pipeline().analyze([frame])`` (router, visit engine, matcher included).
- RF-DETR detector in three runtimes with an accuracy-parity check against the PyTorch output on the same images.
- fast-alpr (plate detector + OCR) with ONNX Runtime CPU / CUDA / TensorRT providers, parity = same plate texts.
- Cold start: a fresh Python process (imports + CUDA init + model load + first inference), weights already cached
  on disk. Modal cold start also includes container start; measure that separately on Modal.
- Streams: N cameras emit frames at a fixed rate (gate 5 fps, bay 1 fps), a pool of workers runs the pipeline,
  frames older than the deadline are dropped (skip-to-latest policy). The largest N with <=1 % drops and p95
  end-to-end latency under the deadline = cameras per GPU. Frames are pre-decoded in RAM (decode cost excluded).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from eval.stats import average_precision, percentiles

REPO = Path(__file__).resolve().parents[1]
LANE_PREDICTORS = [("ident", "get_predictor"), ("flow", "get_predictor"), ("flow", "get_router"),
                   ("ops", "get_predictor"), ("inspect", "get_predictor")]  # lanes/<lane>/predict.py factories
RES_720P = (1280, 720)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def hardware_label() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
            return f"{name} (CUDA_VISIBLE_DEVICES={vis})"
    except Exception:
        pass
    import platform

    return f"CPU {platform.processor() or platform.machine()}"


def hardware_info() -> dict[str, str]:
    info = {"gpu": hardware_label()}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        if out:
            info["nvidia_smi"] = " | ".join(out)
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_runtime"] = str(torch.version.cuda)
    except Exception:
        pass
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
    except Exception:
        pass
    info["cpu_threads"] = str(os.cpu_count())
    info["io_pressure"] = io_pressure()
    info["gpu_contention"] = gpu_contention()
    return info


def gpu_contention() -> str:
    """Other compute processes and utilisation on the visible GPU (GPU1 on the box is shared with Blender and lanes)."""
    try:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        q = subprocess.run(["nvidia-smi", "-i", vis, "--query-gpu=uuid,utilization.gpu,memory.used", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        uuid, util, mem = [x.strip() for x in q.split(",")]
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.splitlines()
        others = [a for a in apps if a.startswith(uuid) and str(os.getpid()) not in a]
        return f"{len(others)} other processes on the GPU, utilisation {util}, memory used {mem}"
    except Exception as e:
        return f"n/a ({type(e).__name__})"


def io_pressure() -> str:
    """Linux PSI for IO at measurement time; the shared HDD was IO-saturated on 10 Oct (cold start / load times suffer)."""
    try:
        lines = Path("/proc/pressure/io").read_text().split("\n")
        full = [l for l in lines if l.startswith("full")]
        return full[0].replace("full ", "full: ") if full else "n/a"
    except OSError:
        return "n/a"


# ------------------------------------------------------------------------------------------------ inputs
def load_frames(n: int = 24, size: tuple[int, int] = RES_720P, vehicle_only: bool = True) -> tuple[list[np.ndarray], str]:
    """Real images from the jury set (vehicle scenes), resized to ``size``. Returns (images RGB, provenance)."""
    import cv2

    from eval.jury.common import jury_items, local_path

    imgs, used = [], []
    try:
        items = jury_items()
    except Exception:
        items = []
    for it in items:
        if it.get("kind") != "image" or (vehicle_only and not it.get("expected", {}).get("vehicle")):
            continue
        p = local_path(it)
        if not p.exists():
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        imgs.append(cv2.resize(im, size, interpolation=cv2.INTER_AREA)[:, :, ::-1].copy())
        used.append(it["id"])
        if len(imgs) >= n:
            break
    if imgs:
        return imgs, f"{len(imgs)} jury images with vehicles, resized to {size[0]}x{size[1]}"
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8) for _ in range(n)], \
        "RANDOM NOISE frames (jury images not found): latency of empty scenes only"


def _frames(imgs):
    from seen.predictor import Frame

    return [Frame(index=0, t_s=0.0, image=im) for im in imgs]


# ----------------------------------------------------------------------------------------------- timing
def time_calls(fn, inputs: list, iters: int, warmup: int = 3) -> list[float]:
    for x in inputs[:warmup]:
        fn(x)
    out = []
    k = 0
    while len(out) < iters:
        x = inputs[k % len(inputs)]
        t0 = time.perf_counter()
        fn(x)
        out.append((time.perf_counter() - t0) * 1000)
        k += 1
    return out


def speed_row(stage: str, hw: str, runtime: str, lat: list[float], batch: int = 1, notes: str = "",
              cold_start_s: float | None = None) -> dict:
    p = percentiles(lat)
    row = {"stage": stage, "hardware": hw, "runtime": runtime, "batch": batch, "p50_ms": round(p["p50"], 2),
           "p95_ms": round(p["p95"], 2), "fps": round(1000.0 * batch / p["p50"], 2), "notes": notes}
    if cold_start_s is not None:
        row["cold_start_s"] = round(cold_start_s, 2)
    return row


def ort_providers_of(obj: Any, depth: int = 4, seen: set | None = None) -> list[list[str]]:
    """Find onnxruntime sessions inside an object and return their active providers (what really ran)."""
    import onnxruntime as ort

    seen = seen if seen is not None else set()
    if depth < 0 or id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, ort.InferenceSession):
        return [obj.get_providers()]
    out = []
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, (int, float, str, bytes, np.ndarray)):
                continue
            out += ort_providers_of(v, depth - 1, seen)
    return out


# ------------------------------------------------------------------------------------ RF-DETR via ONNX
COCO_KEEP_IDS = None


def _coco_keep():
    global COCO_KEEP_IDS
    if COCO_KEEP_IDS is None:
        from rfdetr.assets.coco_classes import COCO_CLASSES

        from seen.baseline import COCO_KEEP

        COCO_KEEP_IDS = {int(k): COCO_KEEP[v] for k, v in COCO_CLASSES.items() if v in COCO_KEEP}
    return COCO_KEEP_IDS


def export_rfdetr_onnx(out_dir: Path, size: str = "base") -> Path:
    p = out_dir / f"rfdetr-{size}.onnx"
    if p.exists():
        return p
    from rfdetr import RFDETRBase, RFDETRLarge

    m = (RFDETRLarge if size == "large" else RFDETRBase)()
    out_dir.mkdir(parents=True, exist_ok=True)
    got = Path(m.export(output_dir=str(out_dir), format="onnx", verbose=False))
    if got != p:
        got.rename(p)
    return p


class RFDETROnnx:
    """RF-DETR COCO through ONNX Runtime with the same pre/post-processing as rfdetr.predict (bench only)."""

    version = "pretrained-onnx"

    def __init__(self, onnx_path: Path, providers: list, threshold: float = 0.4, name: str = "rfdetr-coco-onnx"):
        self.path, self.providers, self.threshold, self.name = onnx_path, providers, threshold, name
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def load(self, device: str = "cuda:0") -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(self.path), so, providers=self.providers)
        inp = self.sess.get_inputs()[0]
        self.input_name, self.res = inp.name, int(inp.shape[2])
        self.active = self.sess.get_providers()

    def raw(self, image: np.ndarray):
        import cv2

        h, w = image.shape[:2]
        x = cv2.resize(image, (self.res, self.res), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = (x.transpose(2, 0, 1)[None] - self.mean) / self.std
        boxes, logits = self.sess.run(None, {self.input_name: np.ascontiguousarray(x, dtype=np.float32)})
        prob = 1 / (1 + np.exp(-logits[0].astype(np.float32)))  # (300, 91)
        flat = prob.reshape(-1)
        k = min(300, flat.size)
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[np.argsort(-flat[idx])]
        scores = flat[idx]
        q, cls = idx // prob.shape[1], idx % prob.shape[1]
        cx, cy, bw, bh = boxes[0][q].T
        xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
        return xyxy, scores, cls

    def predict(self, frames, context=None):
        from seen.predictor import Det, empty_preds

        keep = _coco_keep()
        out = empty_preds(frames)
        for f, p in zip(frames, out):
            t0 = time.perf_counter()
            xyxy, scores, cls = self.raw(f.image)
            h, w = f.image.shape[:2]
            for (x1, y1, x2, y2), s, c in zip(xyxy, scores, cls):
                if s < self.threshold:
                    break
                name = keep.get(int(c))
                if name:
                    p.detections.append(Det(cls=name, confidence=float(s), bbox=(x1 / w, y1 / h, x2 / w, y2 / h)))
            p.timings_ms["vehicle_detector"] = (time.perf_counter() - t0) * 1000
        return out


def _dets(preds) -> list[tuple]:
    return [(d.cls, d.confidence, *d.bbox) for d in preds[0].detections]


def parity(ref: list[list[tuple]], test: list[list[tuple]], classes=("car", "truck", "bus", "motorcycle", "person")) -> dict:
    """Treat the PyTorch output as ground truth: AP50 of the other runtime against it + mean |dconf| of matches."""
    from eval.stats import iou_matrix

    images = [{"gt": [(d[0], *d[2:6]) for d in r], "pred": list(t)} for r, t in zip(ref, test)]
    aps = [average_precision(images, c, 0.5) for c in classes]
    aps = [a for a in aps if np.isfinite(a)]
    dconf, n_ref, n_match = [], 0, 0
    for r, t in zip(ref, test):
        n_ref += len(r)
        for c in classes:
            rb = [d for d in r if d[0] == c]
            tb = [d for d in t if d[0] == c]
            if not rb or not tb:
                continue
            iou = iou_matrix(np.array([d[2:6] for d in rb]), np.array([d[2:6] for d in tb]))
            for i in range(len(rb)):
                j = int(np.argmax(iou[i]))
                if iou[i, j] >= 0.5:
                    n_match += 1
                    dconf.append(abs(rb[i][1] - tb[j][1]))
    return {"parity_map50_vs_torch": float(np.mean(aps)) if aps else float("nan"),
            "matched_fraction": n_match / n_ref if n_ref else float("nan"),
            "mean_abs_dconf": float(np.mean(dconf)) if dconf else float("nan"), "n_images": len(ref)}


TRT_DIRS = [os.environ.get("SEEN_TRT_LIBS", ""),
            os.path.join(os.environ.get("SEEN_FAST", "/home/amirkhon/seen_fast"), "eval", "trt", "tensorrt_libs"),
            "/mnt/bigdrive/amirkhon/seen/env/eval-trt/tensorrt_libs"]  # TensorRT 10.15 (cu13): libnvinfer + sm120/ptx builder resources


def trt_available() -> tuple[bool, str]:
    """ORT's TensorRT EP dlopens libnvinfer.so.10. We keep TensorRT 10.x libs out of env/main (env/eval-trt) and
    preload them by absolute path with RTLD_GLOBAL so the EP finds them already loaded."""
    import ctypes

    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # CUDA / cuDNN from the nvidia-* wheels torch already installed
    except Exception:
        pass
    errs = []
    for d in filter(None, TRT_DIRS):
        p = Path(d)
        if not (p / "libnvinfer.so.10").exists():
            errs.append(f"{d}: no libnvinfer.so.10")
            continue
        try:
            for name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
                if (p / name).exists():
                    ctypes.CDLL(str(p / name), mode=ctypes.RTLD_GLOBAL)
            return True, f"TensorRT libs preloaded from {d}"
        except OSError as e:
            errs.append(f"{d}: {e}")
    try:
        ctypes.CDLL("libnvinfer.so.10", mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL("libnvonnxparser.so.10", mode=ctypes.RTLD_GLOBAL)
        return True, "libnvinfer.so.10 on the loader path"
    except OSError as e:
        errs.append(str(e))
    return False, "TensorRT 10 libs not available: " + "; ".join(errs)


def bench_rfdetr_runtimes(imgs, hw, iters, work: Path, log) -> tuple[list[dict], dict]:
    from seen.baseline import PretrainedDetector

    rows, par = [], {}
    frames = [[f] for f in _frames(imgs)]
    torch_det = PretrainedDetector()
    torch_det.load("cuda:0")
    lat = time_calls(lambda fr: torch_det.predict(fr), frames, iters)
    rows.append(speed_row("vehicle_detector (RF-DETR base, 560px)", hw, "PyTorch (rfdetr.predict, fp32)", lat,
                          notes="reference for parity"))
    ref = [_dets(torch_det.predict(fr)) for fr in frames]
    log(f"torch rfdetr p50 {rows[-1]['p50_ms']} ms")
    try:  # rfdetr's own optimised path: traced (torchscript) + FP16
        import torch

        t0 = time.perf_counter()
        torch_det.model.inference(compile=True, batch_size=1, dtype=torch.float16)
        torch_det.predict(frames[0])
        build_s = time.perf_counter() - t0
        lat = time_calls(lambda fr: torch_det.predict(fr), frames, iters)
        pr = parity(ref, [_dets(torch_det.predict(fr)) for fr in frames])
        par["PyTorch fp16 (rfdetr.inference, torchscript)"] = {**pr, "first_call_s": round(build_s, 2)}
        rows.append(speed_row("vehicle_detector (RF-DETR base, 560px)", hw, "PyTorch fp16 (rfdetr.inference, torchscript)", lat,
                              notes=f"parity vs PyTorch fp32 on {pr['n_images']} images: mAP50 {pr['parity_map50_vs_torch']:.3f}, "
                                    f"matched {pr['matched_fraction']:.3f}, mean |dconf| {pr['mean_abs_dconf']:.4f}; trace+cast {build_s:.1f}s"))
        log(f"torch fp16 p50 {rows[-1]['p50_ms']} ms parity {pr}")
    except Exception as e:
        par["PyTorch fp16 (rfdetr.inference, torchscript)"] = f"failed: {type(e).__name__}: {str(e)[:200]}"
    del torch_det
    try:
        onnx_path = export_rfdetr_onnx(work / "export")
    except Exception as e:
        log(f"ONNX export failed: {e}")
        return rows, {"onnx_export_error": str(e)}
    configs = [("ONNX Runtime CUDA EP (fp32)", ["CUDAExecutionProvider", "CPUExecutionProvider"])]
    ok, why = trt_available()
    if ok:
        def trt(tag: str, **opt):
            cache = work / f"trt_cache_{tag}"
            cache.mkdir(parents=True, exist_ok=True)
            return [("TensorrtExecutionProvider", {"trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache),
                                                   "trt_timing_cache_enable": True, "trt_timing_cache_path": str(cache), **opt}),
                    "CUDAExecutionProvider", "CPUExecutionProvider"]

        configs.append(("ONNX Runtime TensorRT EP (fp32/tf32)", trt("fp32")))
        configs.append(("ONNX Runtime TensorRT EP (fp16)", trt("fp16", trt_fp16_enable=True)))
        configs.append(("ONNX Runtime TensorRT EP (fp16, LayerNorm in fp32)",
                        trt("fp16ln", trt_fp16_enable=True, trt_layer_norm_fp32_fallback=True)))
    else:
        par["tensorrt"] = why
    for label, prov in configs:
        try:
            m = RFDETROnnx(onnx_path, prov)
            t0 = time.perf_counter()
            m.load()
            m.predict(frames[0])  # engine build for TRT happens here
            build_s = time.perf_counter() - t0
            if "TensorRT" in label and m.active[0] != "TensorrtExecutionProvider":
                par[label] = f"TensorRT EP did not activate (active {m.active})"
                continue
            lat = time_calls(lambda fr: m.predict(fr), frames, iters)
            test = [_dets(m.predict(fr)) for fr in frames]
            pr = parity(ref, test)
            par[label] = {**pr, "active_providers": m.active, "first_call_s": round(build_s, 2)}
            rows.append(speed_row("vehicle_detector (RF-DETR base, 560px)", hw, label, lat,
                                  notes=f"parity vs PyTorch on {pr['n_images']} images: mAP50 {pr['parity_map50_vs_torch']:.3f}, "
                                        f"matched {pr['matched_fraction']:.3f}, mean |dconf| {pr['mean_abs_dconf']:.4f}; "
                                        f"first call incl. session/engine build {build_s:.1f}s"))
            log(f"{label} p50 {rows[-1]['p50_ms']} ms parity {pr}")
        except Exception as e:
            par[label] = f"failed: {type(e).__name__}: {e}"
            log(f"{label} failed: {e}")
    return rows, par


def bench_alpr_runtimes(imgs, hw, iters, work: Path, log) -> tuple[list[dict], dict]:
    from fast_alpr import ALPR

    rows, par = [], {}
    bgr = [np.ascontiguousarray(im[:, :, ::-1]) for im in imgs]
    # The reference is the CUDA EP (fp32). TensorRT is tried last because loading its libraries changes what the
    # library-default provider list resolves to in this process.
    configs = [("ONNX Runtime CUDA EP (fp32, reference)", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
               ("library default providers (as in seen.baseline)", None),
               ("ONNX Runtime CPU EP", ["CPUExecutionProvider"]),
               ("ONNX Runtime TensorRT EP (fp16)", "TRT")]
    ref_texts = None
    for label, prov in configs:
        try:
            if prov == "TRT":
                ok, why = trt_available()
                if not ok:
                    par[label] = why
                    continue
                cache = work / "trt_cache_alpr"
                cache.mkdir(parents=True, exist_ok=True)
                prov = [("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                                       "trt_engine_cache_path": str(cache)}),
                        "CUDAExecutionProvider", "CPUExecutionProvider"]
            kw = {} if prov is None else {"detector_providers": prov, "ocr_providers": prov}
            a = ALPR(**kw)
            t0 = time.perf_counter()
            a.predict(bgr[0])
            first = time.perf_counter() - t0
            lat = time_calls(lambda x: a.predict(x), bgr, iters)
            texts = [sorted((r.ocr.text if r.ocr else "") for r in a.predict(x)) for x in bgr]
            n_plates = sum(len(t) for t in texts)
            if ref_texts is None:
                ref_texts = texts
            same = sum(t == r for t, r in zip(texts, ref_texts))
            active = ort_providers_of(a)
            par[label] = {"active_providers": active, "same_texts_as_reference": f"{same}/{len(texts)} images",
                          "plates_found": n_plates, "first_call_s": round(first, 2)}
            rows.append(speed_row("plate (fast-alpr detector + OCR, per frame)", hw, label, lat,
                                  notes=f"{n_plates} plates on {len(bgr)} images; identical plate texts to the CUDA fp32 "
                                        f"reference on {same}/{len(texts)} images; active providers {active}"))
            log(f"alpr {label} p50 {rows[-1]['p50_ms']} ms; {par[label]}")
        except Exception as e:
            par[label] = f"failed: {type(e).__name__}: {e}"
            log(f"alpr {label} failed: {e}")
    return rows, par


def bench_lane_predictors(imgs, hw, iters, log) -> tuple[list[dict], dict]:
    """Each lane's get_predictor() (if importable and its artifacts load), given baseline detector context."""
    from seen.baseline import PretrainedDetector

    rows, status = [], {}
    frames = [[f] for f in _frames(imgs)]
    ctx_det = None
    for lane, factory in LANE_PREDICTORS:
        mod = f"lanes.{lane}.predict"
        key = f"{lane}.{factory}"
        try:
            m = importlib.import_module(mod)
            pred = getattr(m, factory)()
            pred.load("cuda:0")
        except Exception as e:
            status[key] = f"skipped: {type(e).__name__}: {str(e)[:200]}"
            continue
        try:
            if ctx_det is None:
                ctx_det = PretrainedDetector()
                ctx_det.load("cuda:0")
                ctxs = [ctx_det.predict(fr) for fr in frames]
            pairs = list(zip(frames, ctxs))
            lat = time_calls(lambda fc: pred.predict(fc[0], context=fc[1]), pairs, iters)
            rows.append(speed_row(f"{key}: {getattr(pred, 'name', lane)}", hw, "as shipped by lane", lat,
                                  notes=f"version {getattr(pred, 'version', '?')}; context = baseline RF-DETR detections"))
            status[key] = "ok"
        except Exception as e:
            status[key] = f"predict failed: {type(e).__name__}: {str(e)[:200]}"
        finally:
            del pred
    return rows, status


def gpu_mem_mb() -> float | None:
    """GPU memory held by this process (torch + ONNX Runtime), from nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            pid, mem = [x.strip() for x in line.split(",")[:2]]
            if int(pid) == os.getpid():
                return float(mem)
    except Exception:
        pass
    return None


def bench_pipeline(imgs, hw, iters, log) -> tuple[list[dict], dict]:
    from seen.pipeline import default_pipeline

    pipe = default_pipeline("cuda:0")
    frames = [[f] for f in _frames(imgs)]
    stage_lat: dict[str, list[float]] = {}
    totals = []

    def run(fr):
        r = pipe.analyze(fr)
        return r

    for fr in frames[:3]:
        run(fr)
    k = 0
    while len(totals) < iters:
        r = run(frames[k % len(frames)])
        for s, v in r.timings_ms.items():
            stage_lat.setdefault(s, []).append(v)
        totals.append(r.timings_ms["total"])
        k += 1
    mem = gpu_mem_mb()
    rows = [speed_row("pipeline total (1 frame, analyze())", hw, "as deployed (seen.pipeline.default_pipeline)", totals,
                      notes=f"models: {pipe.model_versions}; GPU memory held by the process after warm-up: "
                            f"{mem if mem is not None else 'n/a'} MiB")]
    for s, v in stage_lat.items():
        if s != "total":
            rows.append(speed_row(f"pipeline stage: {s}", hw, "as deployed", v, notes="from AnalysisResult.timings_ms"))
    return rows, {"model_versions": pipe.model_versions, "gpu_mem_mib": mem}


# ------------------------------------------------------------------------------------------- cold start
def cold_probe() -> None:
    """Run in a fresh process: time imports, model load and the first inference of the default pipeline."""
    t_start = float(os.environ.get("SEEN_T0", time.time()))
    t0 = time.time()
    import torch  # noqa: F401

    from seen.pipeline import default_pipeline
    from seen.predictor import Frame

    t_imp = time.time()
    pipe = default_pipeline("cuda:0")
    t_load = time.time()
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    imgs, _ = load_frames(1)
    pipe.analyze([Frame(index=0, t_s=0.0, image=imgs[0] if imgs else img)])
    t_first = time.time()
    pipe.analyze([Frame(index=0, t_s=0.0, image=imgs[0] if imgs else img)])
    t_second = time.time()
    print("COLD_PROBE " + json.dumps({"interpreter_start_s": t0 - t_start, "imports_s": t_imp - t0,
                                       "model_load_s": t_load - t_imp, "first_inference_s": t_first - t_load,
                                       "second_inference_s": t_second - t_first, "total_s": t_first - t_start}))


def bench_cold(hw: str, repeats: int, log) -> tuple[list[dict], list[dict]]:
    runs = []
    for _ in range(repeats):
        env = {**os.environ, "SEEN_T0": str(time.time())}
        try:
            r = subprocess.run([sys.executable, "-m", "eval.bench", "--cold-probe"], capture_output=True, text=True,
                               env=env, cwd=str(REPO), timeout=2400)
        except subprocess.TimeoutExpired:
            log("cold probe timed out after 2400 s")
            continue
        line = [l for l in r.stdout.splitlines() if l.startswith("COLD_PROBE ")]
        if not line:
            log(f"cold probe failed: {r.stderr[-800:]}")
            continue
        runs.append(json.loads(line[0][len("COLD_PROBE "):]))
        log(f"cold probe {runs[-1]}")
    if not runs:
        return [], []
    tot = [r["total_s"] for r in runs]
    med = lambda k: float(np.median([r[k] for r in runs]))  # noqa: E731
    row = {"stage": "pipeline cold start (new process to first result)", "hardware": hw, "runtime": "as deployed",
           "batch": 1, "p50_ms": round(med("first_inference_s") * 1000, 1), "cold_start_s": round(float(np.median(tot)), 2),
           "notes": f"median of {len(runs)} fresh processes, weights cached on local disk: interpreter {med('interpreter_start_s'):.1f}s, "
                    f"imports {med('imports_s'):.1f}s, model load {med('model_load_s'):.1f}s, first inference "
                    f"{med('first_inference_s'):.2f}s (p50_ms = first inference), second inference {med('second_inference_s') * 1000:.0f} ms. "
                    "Excludes container start and weight download."}
    return [row], runs


# ---------------------------------------------------------------------------------------------- streams
@dataclass
class StreamResult:
    n_cams: int
    fps: float
    workers: int
    offered: int
    processed: int
    dropped: int
    lat_p50_ms: float
    lat_p95_ms: float
    achieved_fps: float
    ok: bool


def stream_sim(pipes: list, frames: list[np.ndarray], n_cams: int, fps: float, duration_s: float,
               deadline_s: float, seed: int = 0) -> StreamResult:
    """Open-loop simulation: N cameras at ``fps`` each; one worker thread per pipeline instance."""
    from seen.predictor import Frame

    rng = random.Random(seed)
    q: queue.Queue = queue.Queue()
    lat: list[float] = []
    dropped = [0]
    processed = [0]
    lock = threading.Lock()
    stop = threading.Event()
    period = 1.0 / fps
    start = time.perf_counter() + 0.2
    events = []
    for c in range(n_cams):
        off = rng.random() * period
        t = off
        while t < duration_s:
            events.append((start + t, c))
            t += period
    events.sort()

    def producer():
        for t_due, c in events:
            d = t_due - time.perf_counter()
            if d > 0:
                time.sleep(d)
            q.put((t_due, c))
        stop.set()

    def worker(pipe):
        k = 0
        while not (stop.is_set() and q.empty()):
            try:
                t_due, c = q.get(timeout=0.05)
            except queue.Empty:
                continue
            if time.perf_counter() - t_due > deadline_s:
                with lock:
                    dropped[0] += 1
                continue
            img = frames[(c * 7 + k) % len(frames)]
            k += 1
            pipe.analyze([Frame(index=0, t_s=0.0, image=img)])
            done = time.perf_counter()
            with lock:
                processed[0] += 1
                lat.append((done - t_due) * 1000)

    th = [threading.Thread(target=worker, args=(p,), daemon=True) for p in pipes]
    for t in th:
        t.start()
    t_begin = time.perf_counter()
    producer()
    for t in th:
        t.join(timeout=deadline_s + 30)
    wall = time.perf_counter() - t_begin
    lat_late = [l for l in lat if l > deadline_s * 1000]
    p = percentiles(lat) if lat else {"p50": float("inf"), "p95": float("inf")}
    offered = len(events)
    bad = dropped[0] + len(lat_late)
    ok = offered > 0 and bad / offered <= 0.01 and p["p95"] <= deadline_s * 1000
    return StreamResult(n_cams, fps, len(pipes), offered, processed[0], dropped[0], round(p["p50"], 1), round(p["p95"], 1),
                        round(processed[0] / max(wall, 1e-9), 2), bool(ok))


def max_cameras(pipes, frames, fps, deadline_s, duration_s, n_max: int, log) -> tuple[int, list[StreamResult]]:
    trials: list[StreamResult] = []
    lo, hi, n = 0, None, 1
    while n <= n_max:
        r = stream_sim(pipes, frames, n, fps, duration_s, deadline_s)
        trials.append(r)
        log(f"  streams {n} x {fps} fps, {len(pipes)} workers -> ok={r.ok} drop={r.dropped}/{r.offered} p95={r.lat_p95_ms} ms")
        if r.ok:
            lo = n
            n *= 2
        else:
            hi = n
            break
    if hi is None:
        return lo, trials
    while hi - lo > 1 and hi - lo > max(1, lo // 8):
        mid = (lo + hi) // 2
        r = stream_sim(pipes, frames, mid, fps, duration_s, deadline_s)
        trials.append(r)
        log(f"  streams {mid} x {fps} fps -> ok={r.ok} drop={r.dropped}/{r.offered} p95={r.lat_p95_ms} ms")
        if r.ok:
            lo = mid
        else:
            hi = mid
    return lo, trials


def bench_streams(imgs, hw, duration_s, worker_counts, log, n_max: int = 256) -> tuple[list[dict], list[dict]]:
    from seen.pipeline import default_pipeline

    profiles = [("gate", 5.0, 1.0), ("bay", 1.0, 2.0)]  # name, fps per camera, deadline s
    rows, trials_out = [], []
    pipes_all = []
    for w in worker_counts:
        while len(pipes_all) < w:
            pipes_all.append(default_pipeline("cuda:0"))
        pipes = pipes_all[:w]
        for p in pipes:  # warm
            from seen.predictor import Frame

            p.analyze([Frame(index=0, t_s=0.0, image=imgs[0])])
        for name, fps, deadline in profiles:
            log(f"stream search: {name} @ {fps} fps, {w} workers")
            n, trials = max_cameras(pipes, imgs, fps, deadline, duration_s, n_max, log)
            best = [t for t in trials if t.n_cams == n]
            trials_out += [{"profile": name, **t.__dict__} for t in trials]
            if n == 0:
                rows.append({"stage": f"streams: {name} cameras @ {fps:g} fps per camera", "hardware": hw,
                             "runtime": f"default pipeline x {w} worker(s)", "batch": 1, "p50_ms": trials[0].lat_p50_ms,
                             "p95_ms": trials[0].lat_p95_ms, "fps": 0.0,
                             "notes": "could not sustain even 1 camera within the deadline"})
                continue
            b = best[-1]
            rows.append({"stage": f"streams: {name} cameras @ {fps:g} fps per camera", "hardware": hw,
                         "runtime": f"default pipeline x {w} worker thread(s)", "batch": 1,
                         "p50_ms": b.lat_p50_ms, "p95_ms": b.lat_p95_ms, "fps": b.achieved_fps,
                         "notes": f"max sustained cameras = {n} (<=1% dropped/late, p95 end-to-end <= {deadline:g}s, "
                                  f"{duration_s:g}s per trial; frames pre-decoded 1280x720, decode excluded)"})
    return rows, trials_out


# ------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output root (docs/lanes/eval_speed.json is written under it)")
    ap.add_argument("--hardware-label", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--stream-seconds", type=float, default=20.0)
    ap.add_argument("--workers", default="1,2,4")
    ap.add_argument("--skip", default="", help="comma list of: runtimes,alpr,lanes,pipeline,cold,streams")
    ap.add_argument("--cold-probe", action="store_true")
    ap.add_argument("--cold-repeats", type=int, default=2)
    a = ap.parse_args(argv)
    if a.cold_probe:
        cold_probe()
        return 0
    from seen.paths import RUNS

    out_root = Path(a.out) if a.out else RUNS / "eval" / "latest"
    work = RUNS / "eval" / "bench"
    run_dir = work / _now()
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = (run_dir / "bench.log").open("a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    skip = set(filter(None, a.skip.split(",")))
    iters = 20 if a.quick else a.iters
    stream_s = 6.0 if a.quick else a.stream_seconds
    workers = [1] if a.quick else [int(x) for x in a.workers.split(",")]
    hw = a.hardware_label or hardware_label()
    imgs, prov = load_frames(8 if a.quick else 24)
    log(f"hardware: {hw}; inputs: {prov}; IO pressure {io_pressure()}")
    rows: list[dict] = []
    extra: dict[str, Any] = {"inputs": prov, "hardware": hardware_info()}
    # Order matters: the runtime comparisons preload TensorRT libs, after which fast-alpr's library-default providers
    # would switch to the TensorRT EP. Everything "as deployed" therefore runs first.
    steps = [("pipeline", lambda: bench_pipeline(imgs, hw, iters, log)),
             ("lanes", lambda: bench_lane_predictors(imgs, hw, iters, log)),
             ("cold", lambda: bench_cold(hw, 1 if a.quick else a.cold_repeats, log)),
             ("streams", lambda: bench_streams(imgs, hw, stream_s, workers, log)),
             ("alpr", lambda: bench_alpr_runtimes(imgs, hw, iters, work, log)),
             ("runtimes", lambda: bench_rfdetr_runtimes(imgs, hw, iters, work, log))]
    for name, fn in steps:
        if name in skip:
            continue
        log(f"== {name}")
        try:
            r, info = fn()
            rows += r
            extra[name] = info
        except Exception as e:
            import traceback

            log(f"{name} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
            extra[name] = f"failed: {type(e).__name__}: {e}"
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    if "noise" in prov.lower():
        for r in rows:
            r["notes"] = (r.get("notes", "") + " | INPUT WAS RANDOM NOISE").strip(" |")
    extra["io_pressure_end"] = io_pressure()
    extra["gpu_contention_end"] = gpu_contention()
    shared = f"GPU sharing at start: {extra['hardware'].get('gpu_contention')}; at end: {extra['gpu_contention_end']}."
    for r in rows:
        r["notes"] = (r.get("notes", "") + " " + shared).strip()
    for r in rows:
        if r["stage"].startswith("pipeline cold start"):
            r["notes"] += f" IO pressure at start of bench: {extra['hardware'].get('io_pressure')} (shared HDD; model load is IO-bound)."
    res = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "speed": rows, "details": extra}
    (run_dir / "bench.json").write_text(json.dumps(res, indent=2, default=str))
    d = out_root / "docs" / "lanes"
    d.mkdir(parents=True, exist_ok=True)
    if skip and (d / "eval_speed.json").exists():  # partial re-run: keep rows of the steps we skipped
        key = lambda r: (r["stage"], r.get("runtime"), r["hardware"])  # noqa: E731
        new = {key(r) for r in rows}
        old = [r for r in json.loads((d / "eval_speed.json").read_text()) if key(r) not in new]
        old_steps = {"runtimes": "vehicle_detector", "alpr": "plate (fast-alpr"}
        old = [r for r in old if not any(n not in skip and r["stage"].startswith(p) for n, p in old_steps.items())]
        rows = old + rows
        try:
            prev = json.loads((d / "eval_bench_details.json").read_text()).get("details", {})
            extra = {**prev, **{k: v for k, v in extra.items() if k not in skip}}
        except Exception:
            pass
        res = {**res, "speed": rows, "details": extra}
    (d / "eval_speed.json").write_text(json.dumps(rows, indent=2))
    hwj = d / "eval_hardware.json"
    hwd = json.loads(hwj.read_text()) if hwj.exists() else {}
    hwd.update({"bench": hw, **{f"bench_{k}": v for k, v in extra["hardware"].items()}})
    hwj.write_text(json.dumps(hwd, indent=2))
    (d / "eval_bench_details.json").write_text(json.dumps(res, indent=2, default=str))
    log(f"wrote {run_dir / 'bench.json'} and {d / 'eval_speed.json'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
