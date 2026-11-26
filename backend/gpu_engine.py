"""GPU scan engine (Numba CUDA) with honest hardware detection.

* One CUDA thread evaluates one window (guide vs protospacer mismatch count + PAM check).
* The reference (both strands) is copied to the GPU ONCE per reference and reused for all guides
  (no repeated host<->device transfer of the reference).  Per scan only the small guide / PAM
  tables go up and one uint8 per window comes back.  This is NOT zero-copy.
* If no real CUDA GPU is present the engine reports unavailable; the app then uses the CPU.
* Numba's CUDA *simulator* (NUMBA_ENABLE_CUDASIM=1) is detected and reported as "simulated"
  - it is used only by the unit tests to verify kernel logic and is never presented as a GPU.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from backend.cpu_engine import PAM_FAIL, ScanResult
from backend.pam_detector import pam_lut, site_layout

_cuda = None
_import_error = None
try:                                   # pragma: no cover - depends on installation
    from numba import cuda as _cuda    # type: ignore
except Exception as exc:               # noqa: BLE001
    _import_error = f"{type(exc).__name__}: {exc}"


@dataclass
class GPUInfo:
    available: bool = False
    simulated: bool = False
    device_name: str = "None"
    compute_capability: str = ""
    total_memory_mb: float = 0.0
    free_memory_mb: float = 0.0
    reason: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _is_simulator() -> bool:
    try:
        from numba import config as ncfg
        return bool(getattr(ncfg, "ENABLE_CUDASIM", False))
    except Exception:  # noqa: BLE001
        return os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"


def detect_gpu() -> GPUInfo:
    """Detect a real CUDA device.  Never raises."""
    if _cuda is None:
        return GPUInfo(reason=f"Numba CUDA is not importable ({_import_error}). Using CPU.")
    sim = _is_simulator()
    try:
        if not _cuda.is_available():
            return GPUInfo(simulated=sim, reason="No CUDA-capable NVIDIA GPU / driver detected. "
                           "Running in CPU-only mode.")
    except Exception as exc:  # noqa: BLE001
        return GPUInfo(reason=f"CUDA detection failed ({type(exc).__name__}: {exc}). Using CPU.")
    if sim:
        return GPUInfo(available=False, simulated=True, device_name="CUDA simulator (not a GPU)",
                       reason="Numba CUDA simulator active - not a real GPU. Using CPU.")
    try:
        dev = _cuda.get_current_device()
        name = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
        cc = ".".join(str(x) for x in dev.compute_capability)
        free_b, total_b = _cuda.current_context().get_memory_info()
        return GPUInfo(True, False, name, cc, total_b / 2**20, free_b / 2**20, "CUDA GPU ready")
    except Exception as exc:  # noqa: BLE001
        return GPUInfo(reason=f"CUDA initialisation failed ({type(exc).__name__}: {exc}). Using CPU.")


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #
if _cuda is not None:
    @_cuda.jit
    def _scan_kernel(seq, guide, pam_lut_d, spacer_off, pam_off, require_pam, out):
        i = _cuda.grid(1)
        if i >= out.shape[0]:
            return
        mm = 0
        for k in range(guide.shape[0]):
            if seq[i + spacer_off + k] != guide[k]:      # N (=4) never equals a guide base
                mm += 1
        if require_pam:
            for k in range(pam_lut_d.shape[0]):
                if pam_lut_d[k, seq[i + pam_off + k]] == 0:
                    out[i] = 255
                    return
        out[i] = mm if mm < 255 else 254


class GPUScanner:
    """Device-resident reference scanner."""
    name = "gpu"

    def __init__(self, fwd: np.ndarray, rev: np.ndarray, threads_per_block: int = 256):
        info = detect_gpu()
        if not info.available and not (_cuda is not None and info.simulated and _is_simulator()):
            raise RuntimeError(info.reason or "GPU unavailable")
        self.tpb = threads_per_block
        self.info = info
        self._mem_before = self._free_mb()
        t0 = time.perf_counter()
        self.d_fwd = _cuda.to_device(np.ascontiguousarray(fwd))
        self.d_rev = _cuda.to_device(np.ascontiguousarray(rev))
        _cuda.synchronize()
        self.upload_s = time.perf_counter() - t0
        self.n_fwd, self.n_rev = len(fwd), len(rev)
        self.device_mem_used_mb = max(0.0, self._mem_before - self._free_mb())

    @staticmethod
    def _free_mb() -> float:
        try:
            return _cuda.current_context().get_memory_info()[0] / 2**20
        except Exception:  # noqa: BLE001
            return 0.0

    def _run(self, d_seq, n, guide_d, lut_d, layout, require_pam):
        nw = n - layout["window"] + 1
        if nw <= 0:
            return np.zeros(0, dtype=np.uint8)
        d_out = _cuda.device_array(nw, dtype=np.uint8)
        blocks = (nw + self.tpb - 1) // self.tpb
        _scan_kernel[blocks, self.tpb](d_seq, guide_d, lut_d, layout["spacer_off"],
                                       layout["pam_off"], bool(require_pam), d_out)
        return d_out

    def scan(self, guide_codes: np.ndarray, system, require_pam: bool = True) -> ScanResult:
        layout = site_layout(system, len(guide_codes))
        lut = pam_lut(system.pam)
        _cuda.synchronize()
        t0 = time.perf_counter()
        guide_d = _cuda.to_device(np.ascontiguousarray(guide_codes))
        lut_d = _cuda.to_device(np.ascontiguousarray(lut))
        k0 = time.perf_counter()
        d_a = self._run(self.d_fwd, self.n_fwd, guide_d, lut_d, layout, require_pam)
        d_b = self._run(self.d_rev, self.n_rev, guide_d, lut_d, layout, require_pam)
        _cuda.synchronize()
        k1 = time.perf_counter()
        a = d_a.copy_to_host() if not isinstance(d_a, np.ndarray) else d_a
        b = d_b.copy_to_host() if not isinstance(d_b, np.ndarray) else d_b
        t1 = time.perf_counter()
        return ScanResult(a, b, t1 - t0, k1 - k0, "gpu")

    def close(self):
        self.d_fwd = self.d_rev = None
