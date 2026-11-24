"""CPU scan engine (NumPy, vectorised) + a slow pure-Python reference used by the tests.

A "scan" evaluates EVERY window of the reference on one strand and returns, per window start,
the number of mismatches between the guide and the protospacer (255 = PAM requirement failed).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from backend.pam_detector import pam_lut, site_layout

PAM_FAIL = 255


@dataclass
class ScanResult:
    mm_fwd: np.ndarray        # uint8 mismatches per window start, forward strand
    mm_rev: np.ndarray        # uint8 mismatches per window start, reverse-complement strand
    elapsed_s: float          # wall time of the scan (compute + any transfers for GPU)
    kernel_s: float | None = None   # pure compute time (GPU kernels only)
    engine: str = "cpu"


def scan_strand_numpy(codes: np.ndarray, guide: np.ndarray, layout: dict, lut: np.ndarray,
                      require_pam: bool) -> np.ndarray:
    W, so, po = layout["window"], layout["spacer_off"], layout["pam_off"]
    nw = len(codes) - W + 1
    if nw <= 0:
        return np.zeros(0, dtype=np.uint8)
    mm = np.zeros(nw, dtype=np.uint8)
    for k in range(len(guide)):                      # one vectorised pass per guide position
        mm += (codes[so + k: so + k + nw] != guide[k])
    if require_pam:
        ok = np.ones(nw, dtype=bool)
        for k in range(lut.shape[0]):
            ok &= lut[k][codes[po + k: po + k + nw]].astype(bool)
        mm[~ok] = PAM_FAIL
    return mm


class CPUScanner:
    """Holds prepared reference arrays; interface shared with GPUScanner."""
    name = "cpu"

    def __init__(self, fwd: np.ndarray, rev: np.ndarray):
        self.fwd, self.rev = fwd, rev

    def scan(self, guide_codes: np.ndarray, system, require_pam: bool = True) -> ScanResult:
        layout = site_layout(system, len(guide_codes))
        lut = pam_lut(system.pam)
        t0 = time.perf_counter()
        a = scan_strand_numpy(self.fwd, guide_codes, layout, lut, require_pam)
        b = scan_strand_numpy(self.rev, guide_codes, layout, lut, require_pam)
        dt = time.perf_counter() - t0
        return ScanResult(a, b, dt, None, "cpu")

    def close(self):
        pass


def scan_strand_python(codes, guide, layout, lut, require_pam) -> list:
    """Slow, obviously-correct reference implementation (used to validate the fast engines)."""
    W, so, po = layout["window"], layout["spacer_off"], layout["pam_off"]
    out = []
    for i in range(len(codes) - W + 1):
        mm = sum(1 for k in range(len(guide)) if codes[i + so + k] != guide[k])
        ok = all(lut[k][codes[i + po + k]] for k in range(lut.shape[0]))
        out.append(PAM_FAIL if (require_pam and not ok) else mm)
    return out
