"""End-to-end analysis pipeline: validate -> preprocess -> scan (GPU/CPU) -> PAM -> candidates."""
from __future__ import annotations

import time
import uuid
from datetime import datetime

import numpy as np

import config
from backend import pam_detector
from backend.gpu_engine import detect_gpu
from backend.off_target_detector import extract_candidates
from backend.preprocessing import dna_to_rna, prepare_reference, encode
from backend.scanner import create_scanner, scan_with_fallback
from backend.sequence_validator import validate_config, composition


class AnalysisConfigError(ValueError):
    """Raised for invalid analysis configurations (message lists every problem)."""


def run_analysis(ref_records, guide_records, system_key: str = config.DEFAULT_SYSTEM,
                 max_mismatches: int = 4, require_pam: bool = True, engine_mode: str = "auto",
                 name: str = "", scheme: config.AlignmentScheme = config.DEFAULT_SCHEME,
                 progress=None, save: bool = True) -> dict:
    t_start = time.perf_counter()
    system = config.CRISPR_SYSTEMS.get(system_key)
    if system is None:
        raise AnalysisConfigError(f"Unknown CRISPR system '{system_key}'.")
    errs = validate_config([len(g.sequence) for g in guide_records],
                           [len(r.sequence) for r in ref_records], system, max_mismatches)
    if errs:
        raise AnalysisConfigError(" ".join(errs))

    def tick(frac, msg):
        if progress:
            progress(min(frac, 1.0), msg)

    gpu = detect_gpu()
    prep_log, refs_info, guides_info = [], [], []
    tick(0.02, "Preprocessing sequences…")
    t0 = time.perf_counter()
    prepared = []
    for r in ref_records:
        p = prepare_reference(r)
        prepared.append(p)
        refs_info.append({"id": r.id, "length": p.length, "composition": r.info or composition(r.sequence),
                          "preprocessing": p.log, "warnings": r.warnings})
    guide_codes = []
    for g in guide_records:
        guide_codes.append(encode(g.sequence))
        guides_info.append({"id": g.id, "dna": g.sequence, "rna": dna_to_rna(g.sequence),
                            "length": len(g.sequence), "composition": g.info or composition(g.sequence),
                            "original": g.original})
    t_prep = time.perf_counter() - t0

    candidates, pam_summary, pam_sites = [], [], []
    truncated_any = False
    scan_s = kernel_s = align_s = 0.0
    windows = 0
    engine_used, fallback = "cpu", ""
    total_jobs = max(1, len(prepared) * len(guide_records))
    job = 0
    for ref in prepared:
        pam_summary.append({"reference": ref.id, **pam_detector.count_pam_sites(ref, system)})
        sites = pam_detector.find_pam_sites(ref, system)
        pam_sites.extend(sites[:2000])
        choice = create_scanner(ref.fwd, ref.rev, engine_mode)
        try:
            for g, gc in zip(guide_records, guide_codes):
                tick(0.1 + 0.85 * job / total_jobs, f"Scanning {ref.id} with {g.id} ({choice.used.upper()})…")
                scan = scan_with_fallback(choice, gc, system, require_pam, ref.fwd, ref.rev)
                scan_s += scan.elapsed_s
                kernel_s += scan.kernel_s or 0.0
                windows += len(scan.mm_fwd) + len(scan.mm_rev)
                t1 = time.perf_counter()
                cands, trunc = extract_candidates(ref, g.id, g.sequence, gc, scan, system,
                                                  max_mismatches, require_pam, scheme)
                align_s += time.perf_counter() - t1
                truncated_any |= trunc
                candidates.extend(cands)
                job += 1
        finally:
            engine_used, fallback = choice.used, choice.fallback_reason
            choice.scanner.close()

    # global ranking per guide across ALL references (extract_candidates ranks per reference only)
    from backend import scoring
    by_guide = {}
    for c in candidates:
        by_guide.setdefault(c["guide_id"], []).append(c)
    candidates = []
    for gid in [g.id for g in guide_records]:
        for rank, c in enumerate(sorted(by_guide.get(gid, []), key=scoring.rank_key), 1):
            c["rank"] = rank
            candidates.append(c)
    perfect = [c for c in candidates if c["classification"] == "Perfect-match target"]
    offs = [c for c in candidates if c["mismatches"] > 0 and c["pam_ok"]]
    offs_nopam = [c for c in candidates if c["mismatches"] > 0 and not c["pam_ok"]]
    aln_scores = [c["alignment_score"] for c in candidates if c["alignment_score"] is not None]
    total_s = time.perf_counter() - t_start
    tick(1.0, "Done")

    results = {
        "id": uuid.uuid4().hex[:12],
        "name": name or f"Analysis {datetime.now():%Y-%m-%d %H:%M:%S}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"system": system.key, "pam": system.pam, "pam_side": system.pam_side,
                   "max_mismatches": max_mismatches, "require_pam": require_pam,
                   "engine_requested": engine_mode, "scheme": vars(scheme) if hasattr(scheme, "__dict__") else {},
                   "seed_len": system.seed_len},
        "engine": {"requested": engine_mode, "used": engine_used, "gpu_available": gpu.available,
                   "device": gpu.device_name, "fallback_reason": fallback, "gpu_info": gpu.to_dict()},
        "references": refs_info, "guides": guides_info,
        "pam_summary": pam_summary, "pam_sites": pam_sites,
        "candidates": candidates,
        "summary": {
            "n_references": len(refs_info), "n_guides": len(guides_info),
            "total_bp": sum(r["length"] for r in refs_info),
            "perfect_targets": len(perfect), "off_targets": len(offs),
            "off_targets_no_pam": len(offs_nopam), "total_candidates": len(candidates),
            "avg_alignment_score": round(float(np.mean(aln_scores)), 2) if aln_scores else 0.0,
            "truncated": truncated_any,
        },
        "timings": {"preprocess_s": round(t_prep, 4), "scan_s": round(scan_s, 4),
                    "kernel_s": round(kernel_s, 4) if engine_used == "gpu" else None,
                    "alignment_s": round(align_s, 4), "total_s": round(total_s, 4),
                    "windows_scanned": windows,
                    "windows_per_s": round(windows / scan_s) if scan_s > 0 else None},
        "disclaimer": config.DISCLAIMER,
        "saved": False,
    }
    if save:
        try:
            from database import database as db
            db.save_analysis(results)
            results["saved"] = True
        except Exception as exc:  # noqa: BLE001 - never lose results because of DB trouble
            results["save_error"] = f"{type(exc).__name__}: {exc}"
    return results
