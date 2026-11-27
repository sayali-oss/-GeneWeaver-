"""CPU vs GPU benchmarking on synthetic (seeded, reproducible) random reference sequences.

Every number is measured at run time.  Identical inputs are given to both engines and their
outputs are compared element-by-element (mismatch counts are integers -> tolerance is exactly 0).
If no real CUDA GPU exists only the CPU rows are produced; speed-up is then reported as N/A.
"""
from __future__ import annotations

import platform
import statistics
import threading
import time
import tracemalloc
from datetime import datetime

import numpy as np
import psutil

import config
from backend.cpu_engine import CPUScanner
from backend.gpu_engine import GPUScanner, detect_gpu
from backend.pam_detector import site_layout


def make_random_codes(n: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    fwd = rng.integers(0, 4, size=n, dtype=np.uint8)
    rev = (3 - fwd[::-1]).astype(np.uint8)         # complement (A<->T, C<->G) then reverse
    return fwd, rev


class _GPUUtilSampler:
    """Polls NVML utilisation in a thread (needs optional nvidia-ml-py / pynvml)."""
    def __init__(self):
        self.samples, self._stop, self._t = [], threading.Event(), None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nv = pynvml
            self.ok = True
        except Exception:  # noqa: BLE001
            self.ok = False

    def __enter__(self):
        if self.ok:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self._nv.nvmlDeviceGetUtilizationRates(self._h).gpu)
            except Exception:  # noqa: BLE001
                break
            time.sleep(0.01)

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=1)

    def mean(self):
        return round(float(np.mean(self.samples)), 1) if self.samples else None


def _stats(times: list) -> dict:
    return {"mean_s": statistics.fmean(times), "min_s": min(times), "max_s": max(times),
            "std_s": statistics.pstdev(times) if len(times) > 1 else 0.0, "total_s": sum(times)}


def _run_engine(make_scanner, guides, system, require_pam, repeats, warmup, windows_per_pass):
    """Measure one engine on one reference.  Returns (row, outputs_of_last_run)."""
    scanner = make_scanner()
    try:
        for _ in range(warmup):                          # JIT compile / cache warm-up (not timed)
            scanner.scan(guides[0], system, require_pam)
        times, kernel, outs = [], [], None
        proc = psutil.Process()
        cpu0, w0 = sum(proc.cpu_times()[:2]), time.perf_counter()
        with _GPUUtilSampler() as sampler:
            for _ in range(repeats):
                t0 = time.perf_counter()
                outs = [scanner.scan(g, system, require_pam) for g in guides]
                times.append(time.perf_counter() - t0)
                if outs[0].kernel_s is not None:
                    kernel.append(sum(o.kernel_s for o in outs))
        wall = time.perf_counter() - w0
        cpu_util = (sum(proc.cpu_times()[:2]) - cpu0) / wall * 100 if wall > 0 else 0.0

        # memory: separate (untimed) run
        tracemalloc.start()
        scanner.scan(guides[0], system, require_pam)
        peak = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()

        st = _stats(times)
        work = windows_per_pass * len(guides)            # guide-vs-site comparisons per run
        row = {**st, "repeats": repeats, "warmup_runs": warmup,
               "windows_per_s": work / st["mean_s"], "kernel_mean_s": statistics.fmean(kernel) if kernel else None,
               "peak_host_mem_mb": round(peak, 2), "process_cpu_percent": round(cpu_util, 1),
               "gpu_util_percent": sampler.mean(),
               "gpu_mem_mb": round(getattr(scanner, "device_mem_used_mb", 0.0), 2) if scanner.name == "gpu" else None,
               "upload_s": getattr(scanner, "upload_s", None)}
        return row, outs
    finally:
        scanner.close()


def run_benchmark(sizes=None, n_guides: int = 4, repeats: int = config.BENCH_REPEATS,
                  system_key: str = config.DEFAULT_SYSTEM, require_pam: bool = True,
                  warmup: int = 1, progress=None, save: bool = True) -> dict:
    sizes = list(sizes or config.BENCH_SIZES_BP)
    system = config.CRISPR_SYSTEMS[system_key]
    gpu = detect_gpu()
    guide_len = system.default_guide_len
    rng = np.random.default_rng(config.BENCH_SEED)
    guides = [rng.integers(0, 4, size=guide_len, dtype=np.uint8) for _ in range(n_guides)]
    layout = site_layout(system, guide_len)
    rows, notes = [], []
    for si, n in enumerate(sizes):
        if progress:
            progress(si / len(sizes), f"Benchmarking {n:,} bp…")
        fwd, rev = make_random_codes(n, config.BENCH_SEED + n)
        wpp = 2 * (n - layout["window"] + 1)
        cpu_row, cpu_out = _run_engine(lambda: CPUScanner(fwd, rev), guides, system, require_pam,
                                       repeats, warmup, wpp)
        cpu_row.update({"size_bp": n, "engine": "cpu", "n_guides": n_guides,
                        "throughput_mbp_s": n * 2 * n_guides / cpu_row["mean_s"] / 1e6, "validated": True,
                        "speedup": None})
        rows.append(cpu_row)
        if gpu.available:
            try:
                g_row, g_out = _run_engine(lambda: GPUScanner(fwd, rev), guides, system, require_pam,
                                           repeats, max(1, warmup), wpp)
                same = all(np.array_equal(a.mm_fwd, b.mm_fwd) and np.array_equal(a.mm_rev, b.mm_rev)
                           for a, b in zip(cpu_out, g_out))
                g_row.update({"size_bp": n, "engine": "gpu", "n_guides": n_guides,
                              "throughput_mbp_s": n * 2 * n_guides / g_row["mean_s"] / 1e6,
                              "validated": bool(same),
                              "speedup": cpu_row["mean_s"] / g_row["mean_s"]})
                rows.append(g_row)
                if not same:
                    notes.append(f"WARNING: CPU and GPU outputs differ at {n:,} bp.")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"GPU run failed at {n:,} bp ({type(exc).__name__}: {exc}); CPU results kept.")
    if not gpu.available:
        notes.append("No real CUDA GPU detected: CPU-only benchmark. GPU speed-up is not available - "
                     "no GPU numbers were estimated or simulated.")
    if progress:
        progress(1.0, "Benchmark complete")
    bench = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "system": system.key, "n_guides": n_guides, "repeats": repeats, "sizes_bp": sizes,
        "gpu_used": any(r["engine"] == "gpu" for r in rows), "gpu": gpu.to_dict(),
        "hardware": {"platform": platform.platform(), "python": platform.python_version(),
                     "cpu_logical_cores": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)},
        "method": ("Synthetic random reference (seed %d), %d random %d-nt guides per run, both strands scanned, "
                   "PAM check %s. Timings are wall-clock per run after %d warm-up run(s); GPU timing includes "
                   "guide upload + kernels + result download but excludes the one-time reference upload "
                   "(reported separately as upload_s)." % (config.BENCH_SEED, n_guides, guide_len,
                                                          "on" if require_pam else "off", warmup)),
        "notes": notes, "rows": rows,
    }
    if save:
        try:
            from database import database as db
            db.save_benchmark(bench)
        except Exception as exc:  # noqa: BLE001
            bench["save_error"] = str(exc)
    return bench
