import json
import numpy as np
from backend.benchmark import run_benchmark, make_random_codes


def test_random_codes_reproducible_and_complementary():
    a, ar = make_random_codes(1000, 1); b, _ = make_random_codes(1000, 1)
    assert np.array_equal(a, b) and np.array_equal(ar, 3 - a[::-1])


def test_benchmark_runs_and_is_measured():
    b = run_benchmark(sizes=[20_000, 60_000], n_guides=2, repeats=3, save=False)
    cpu = [r for r in b["rows"] if r["engine"] == "cpu"]
    assert len(cpu) == 2 and all(r["mean_s"] > 0 and r["min_s"] <= r["mean_s"] <= r["max_s"] for r in cpu)
    assert all(r["validated"] and r["throughput_mbp_s"] > 0 and r["peak_host_mem_mb"] > 0 for r in cpu)
    if not b["gpu_used"]:                        # CPU-only machine: nothing GPU-related may be fabricated
        assert all(r["engine"] == "cpu" for r in b["rows"]) and b["notes"]
    json.dumps(b)                                # fully serialisable


def test_benchmark_persisted(tmp_path):
    from database import database as db
    b = run_benchmark(sizes=[10_000], n_guides=1, repeats=3, save=False)
    db.save_benchmark(b, tmp_path / "b.db")
    assert db.latest_benchmark(tmp_path / "b.db")["sizes_bp"] == [10_000]
