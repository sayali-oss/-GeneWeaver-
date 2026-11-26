"""GPU tests.  Real-GPU tests auto-skip without CUDA.  Kernel LOGIC is verified through Numba's
CUDA simulator in a subprocess (this verifies correctness, not performance - and the simulator is
never reported as a GPU)."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import config
from backend.gpu_engine import detect_gpu
from backend.scanner import create_scanner, scan_with_fallback

ROOT = Path(__file__).resolve().parents[1]
GPU = detect_gpu()


def test_detection_never_raises_and_is_honest():
    info = detect_gpu()
    assert isinstance(info.available, bool)
    if not info.available:
        assert info.reason and info.device_name in ("None", "CUDA simulator (not a GPU)")


def test_cpu_fallback_when_gpu_unavailable():
    fwd = np.zeros(100, dtype=np.uint8); rev = fwd.copy()
    ch = create_scanner(fwd, rev, "gpu")
    if not GPU.available:
        assert ch.used == "cpu" and ch.fallback_reason          # clear message, no crash
    else:
        assert ch.used == "gpu"
    ch.scanner.close()


def test_auto_and_cpu_modes():
    fwd = np.zeros(100, dtype=np.uint8)
    assert create_scanner(fwd, fwd, "cpu").used == "cpu"
    assert create_scanner(fwd, fwd, "auto").used in ("cpu", "gpu")


def test_runtime_gpu_failure_falls_back_to_cpu():
    class Boom:
        name = "gpu"
        def scan(self, *a): raise MemoryError("out of device memory")
        def close(self): pass
    from backend.scanner import ScannerChoice
    fwd = np.random.default_rng(0).integers(0, 4, 200, dtype=np.uint8)
    ch = ScannerChoice(Boom(), "gpu", "gpu")
    out = scan_with_fallback(ch, fwd[:20], config.CRISPR_SYSTEMS["SpCas9"], True, fwd, fwd)
    assert ch.used == "cpu" and "fallback" in ch.fallback_reason.lower() and out.engine == "cpu"


def test_kernel_matches_cpu_in_cuda_simulator():
    code = textwrap.dedent("""
        import sys; sys.path.insert(0, %r)
        import numpy as np, config
        from backend.cpu_engine import CPUScanner
        from backend.gpu_engine import GPUScanner, detect_gpu
        assert detect_gpu().simulated and not detect_gpu().available   # simulator is never a "GPU"
        rng = np.random.default_rng(3)
        for system in config.CRISPR_SYSTEMS.values():
            fwd = rng.integers(0, 5, 1500, dtype=np.uint8); rev = rng.integers(0, 5, 1500, dtype=np.uint8)
            guide = rng.integers(0, 4, system.default_guide_len, dtype=np.uint8)
            for req in (True, False):
                a = CPUScanner(fwd, rev).scan(guide, system, req)
                b = GPUScanner(fwd, rev, threads_per_block=64).scan(guide, system, req)
                assert np.array_equal(a.mm_fwd, b.mm_fwd) and np.array_equal(a.mm_rev, b.mm_rev), (system.key, req)
        print("KERNEL_OK")
    """) % str(ROOT)
    env = dict(os.environ, NUMBA_ENABLE_CUDASIM="1")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert "KERNEL_OK" in out.stdout, out.stdout + out.stderr


@pytest.mark.skipif(not GPU.available, reason="no real CUDA GPU in this environment")
def test_real_gpu_equals_cpu():
    from backend.benchmark import make_random_codes
    from backend.cpu_engine import CPUScanner
    from backend.gpu_engine import GPUScanner
    fwd, rev = make_random_codes(200_000, 5)
    guide = np.random.default_rng(1).integers(0, 4, 20, dtype=np.uint8)
    sp = config.CRISPR_SYSTEMS["SpCas9"]
    a, b = CPUScanner(fwd, rev).scan(guide, sp, True), GPUScanner(fwd, rev).scan(guide, sp, True)
    assert np.array_equal(a.mm_fwd, b.mm_fwd) and np.array_equal(a.mm_rev, b.mm_rev)
