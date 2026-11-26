"""Engine selection with automatic CPU fallback."""
from __future__ import annotations

from dataclasses import dataclass

from backend.cpu_engine import CPUScanner
from backend.gpu_engine import GPUScanner, detect_gpu


@dataclass
class ScannerChoice:
    scanner: object
    requested: str
    used: str                 # "cpu" or "gpu"
    fallback_reason: str = ""


def create_scanner(fwd, rev, mode: str = "auto") -> ScannerChoice:
    """mode: 'auto' (GPU if available), 'gpu' (GPU, fall back with a message), 'cpu'."""
    mode = mode.lower()
    if mode == "cpu":
        return ScannerChoice(CPUScanner(fwd, rev), mode, "cpu")
    info = detect_gpu()
    if not info.available:
        reason = info.reason if mode == "gpu" else ""
        return ScannerChoice(CPUScanner(fwd, rev), mode, "cpu", reason or "GPU not available")
    try:
        return ScannerChoice(GPUScanner(fwd, rev), mode, "gpu")
    except Exception as exc:  # noqa: BLE001  (e.g. out of GPU memory)
        return ScannerChoice(CPUScanner(fwd, rev), mode, "cpu",
                             f"GPU initialisation failed ({type(exc).__name__}: {exc}); CPU fallback used.")


def scan_with_fallback(choice: ScannerChoice, guide_codes, system, require_pam, fwd, rev):
    """Run one scan; if the GPU fails mid-run, transparently redo it on the CPU."""
    try:
        return choice.scanner.scan(guide_codes, system, require_pam)
    except Exception as exc:  # noqa: BLE001
        if choice.used != "gpu":
            raise
        choice.fallback_reason = f"GPU error during scan ({type(exc).__name__}: {exc}); CPU fallback used."
        choice.used = "cpu"
        choice.scanner = CPUScanner(fwd, rev)
        return choice.scanner.scan(guide_codes, system, require_pam)
