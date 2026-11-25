"""Candidate (on-/off-target) extraction from scan results, with scoring and alignments."""
from __future__ import annotations

import numpy as np

import config
from backend import scoring
from backend.alignment_engine import needleman_wunsch, smith_waterman
from backend.pam_detector import check_pam, site_layout
from backend.preprocessing import decode, strand_to_forward


def extract_candidates(ref, guide_id: str, guide_seq: str, guide_codes: np.ndarray, scan,
                       system: config.CrisprSystem, max_mm: int, require_pam: bool,
                       scheme: config.AlignmentScheme = config.DEFAULT_SCHEME,
                       max_candidates: int = config.MAX_CANDIDATES_PER_GUIDE,
                       max_alignments: int = config.MAX_ALIGNMENTS_PER_GUIDE) -> tuple:
    """Turn per-window mismatch arrays into ranked candidate dicts.

    Returns (candidates, truncated_flag).
    """
    L = len(guide_seq)
    lay = site_layout(system, L)
    so, po, P = lay["spacer_off"], lay["pam_off"], len(system.pam)
    raw = []
    for strand, arr, mm in (("+", ref.fwd, scan.mm_fwd), ("-", ref.rev, scan.mm_rev)):
        idx = np.nonzero(mm <= max_mm)[0]
        for i in idx:
            raw.append((int(mm[i]), strand, int(i), arr))
    truncated = len(raw) > max_candidates
    if truncated:
        raw.sort(key=lambda x: (x[0], x[1], x[2]))
        raw = raw[:max_candidates]

    cands = []
    for _, strand, i, arr in raw:
        target = decode(arr[i + so:i + so + L])
        pam_seq = decode(arr[i + po:i + po + P])
        mm_pos = [k + 1 for k in range(L) if target[k] != guide_seq[k] or target[k] == "N"]
        pam_ok = check_pam(pam_seq, system.pam)
        if require_pam and not pam_ok:
            continue
        s, e = strand_to_forward(i + so, L, ref.length, strand)
        ps, pe = strand_to_forward(i + po, P, ref.length, strand)
        seed = scoring.seed_positions(L, system)
        n_mm = len(mm_pos)
        if n_mm == 0 and pam_ok:
            cls = "Perfect-match target"
        elif n_mm == 0:
            cls = "Perfect match, no PAM"
        else:
            cls = "Off-target candidate"
        cands.append({
            "guide_id": guide_id, "guide_seq": guide_seq, "reference": ref.id, "strand": strand,
            "start": s + 1, "end": e, "pam_start": ps + 1, "pam_end": pe, "pam_seq": pam_seq,
            "pam_ok": bool(pam_ok), "mismatches": n_mm, "mismatch_positions": mm_pos,
            "seed_mismatches": sum(1 for p in mm_pos if p in seed),
            "similarity_score": scoring.similarity_score(mm_pos, L, system),
            "target_seq": target, "classification": cls,
            "alignment": None, "sw_score": None, "alignment_score": None, "explanation": "",
            "_i": i,
        })
    cands.sort(key=scoring.rank_key)

    flank = config.ALIGNMENT_FLANK
    for rank, c in enumerate(cands, 1):
        c["rank"] = rank
        c["explanation"] = scoring.explain(c, system)
        if rank <= max_alignments:
            arr = ref.fwd if c["strand"] == "+" else ref.rev
            i = c.pop("_i")
            a0, a1 = max(0, i + so - flank), min(len(arr), i + so + L + flank)
            region = decode(arr[a0:a1])
            nw = needleman_wunsch(guide_seq, c["target_seq"], scheme)
            sw = smith_waterman(guide_seq, region, scheme)
            c["alignment"] = nw.to_dict()
            c["alignment_score"] = nw.score
            c["sw_score"] = sw.score
        else:
            c.pop("_i", None)
    return cands, truncated
