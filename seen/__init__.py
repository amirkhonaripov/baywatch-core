"""Seen: service-visit understanding from cameras."""
__version__ = "0.1.0"


def _preload_cuda_libs() -> None:
    """onnxruntime-gpu needs cuBLAS/cuDNN from the nvidia-* wheels; importing torch first loads them into the process."""
    try:
        import torch  # noqa: F401

        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:  # noqa: BLE001
        pass


_preload_cuda_libs()
