"""Backend selection: CuPy (GPU) if available, otherwise NumPy (CPU).

Both libraries share the same array API, so the aligner code is written once.
"""
import numpy as np

try:
    import cupy as cp
    try:
        cp.cuda.runtime.getDeviceCount()
        _GPU_OK = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        _GPU_OK = False
except ImportError:  # CuPy not installed
    cp = None
    _GPU_OK = False


def get_backend(device: str = "auto"):
    """Return (xp, name). device: 'auto' | 'gpu' | 'cpu'."""
    if device == "cpu":
        return np, "cpu (numpy)"
    if device == "gpu":
        if not _GPU_OK:
            raise RuntimeError("GPU requested but CuPy/CUDA device not available.")
        return cp, "gpu (cupy)"
    if _GPU_OK:
        return cp, "gpu (cupy)"
    return np, "cpu (numpy)"


def to_numpy(arr):
    """Copy an array back to host memory regardless of backend."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)
