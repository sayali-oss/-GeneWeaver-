"""PAM detection for configurable CRISPR systems (both strands)."""
from __future__ import annotations

import numpy as np

import config
from backend.preprocessing import PreparedReference, strand_to_forward


def pam_lut(pam: str) -> np.ndarray:
    """(len(pam), 5) uint8 lookup: lut[k, base_code] = 1 if the base satisfies pattern[k].

    Column 4 (N / unknown base) is always 0, so unknown bases never satisfy a PAM.
    """
    lut = np.zeros((len(pam), 5), dtype=np.uint8)
    for k, ch in enumerate(pam.upper()):
        allowed = config.IUPAC[ch]
        for code, base in enumerate("ACGT"):
            if base in allowed:
                lut[k, code] = 1
    return lut


def pam_match_array(codes: np.ndarray, pam: str) -> np.ndarray:
    """Boolean ok[i] = pattern matches codes[i : i+len(pam)]  (length n-P+1)."""
    P = len(pam)
    nw = len(codes) - P + 1
    if nw <= 0:
        return np.zeros(0, dtype=bool)
    lut = pam_lut(pam)
    ok = np.ones(nw, dtype=bool)
    for k in range(P):
        ok &= lut[k][codes[k:k + nw]].astype(bool)
    return ok


def check_pam(pam_seq: str, pam_pattern: str) -> bool:
    """True if the literal PAM sequence satisfies the IUPAC pattern."""
    if len(pam_seq) != len(pam_pattern):
        return False
    return all(b in config.IUPAC[p] for b, p in zip(pam_seq.upper(), pam_pattern.upper()))


def find_pam_sites(ref: PreparedReference, system: config.CrisprSystem) -> list:
    """All PAM occurrences on both strands (1-based forward coordinates)."""
    sites = []
    P = len(system.pam)
    for strand, arr in (("+", ref.fwd), ("-", ref.rev)):
        for i in np.nonzero(pam_match_array(arr, system.pam))[0]:
            s, e = strand_to_forward(int(i), P, ref.length, strand)
            seq = "".join("ACGTN"[c] for c in arr[i:i + P])
            sites.append({"reference": ref.id, "strand": strand, "start": s + 1, "end": e,
                          "pam": seq})
    sites.sort(key=lambda d: d["start"])
    return sites


def count_pam_sites(ref: PreparedReference, system: config.CrisprSystem) -> dict:
    return {"+": int(pam_match_array(ref.fwd, system.pam).sum()),
            "-": int(pam_match_array(ref.rev, system.pam).sum()),
            "pam": system.pam, "pam_len": len(system.pam)}


def site_layout(system: config.CrisprSystem, guide_len: int) -> dict:
    """Offsets of protospacer and PAM inside one scan window."""
    P = len(system.pam)
    if system.pam_side == "3prime":
        return {"spacer_off": 0, "pam_off": guide_len, "window": guide_len + P}
    return {"spacer_off": P, "pam_off": 0, "window": guide_len + P}
