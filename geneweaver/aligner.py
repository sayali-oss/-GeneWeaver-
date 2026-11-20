"""Core engine: mismatch-tolerant guide search with PAM check on both strands.

Idea: for a genome array G and guide g (length L), the mismatch count at every
start position i is  sum_j [G[i+j] != g[j]].  We compute this for ALL positions
at once with L vectorised comparisons - each one is a data-parallel operation,
which is exactly what a GPU is good at.
"""
from dataclasses import dataclass
import numpy as np

from .backend import get_backend, to_numpy
from .sequence import encode, reverse_complement, decode
from .scoring import off_target_score

PAM_LEN = 3  # NGG


@dataclass
class Hit:
    guide: str
    chrom: str
    position: int      # 0-based start of the protospacer on the forward strand
    strand: str        # '+' or '-'
    mismatches: int
    mm_positions: tuple
    site: str          # genomic protospacer (strand-oriented) + PAM
    score: float


def _mismatch_counts(xp, genome, guide, chunk=50_000_000):
    """Yield (offset, mismatch_count_array, valid_mask) over genome chunks."""
    L = len(guide)
    n = genome.shape[0]
    span = L + PAM_LEN
    if n < span:
        return
    start = 0
    while start < n - span + 1:
        end = min(start + chunk, n)
        seg = genome[start:end]
        m = seg.shape[0] - span + 1
        if m <= 0:
            break
        mism = xp.zeros(m, dtype=xp.uint8)
        for j in range(L):
            mism += (seg[j:j + m] != guide[j]).astype(xp.uint8)
        # PAM = N G G right after the protospacer
        pam_ok = (seg[L + 1:L + 1 + m] == 2) & (seg[L + 2:L + 2 + m] == 2)
        # windows containing ambiguous bases (N=4) are skipped
        clean = xp.ones(m, dtype=bool)
        for j in range(span):
            clean &= seg[j:j + m] < 4
        yield start, mism, pam_ok & clean
        start += m



def _scan_strand(xp, genome_dev, guide_np, guide_dev, max_mm, strand, chrom, glen, guide_str):
    hits = []
    L = len(guide_np)
    g_host = to_numpy(genome_dev) if False else None  # (kept for clarity; not used)
    for start, mism, ok in _mismatch_counts(xp, genome_dev, guide_dev):
        cand = xp.nonzero((mism <= max_mm) & ok)[0]
        cand = to_numpy(cand)
        mism_h = to_numpy(mism[cand]) if cand.size else []
        for idx, k in zip(cand, mism_h):
            i = int(start + idx)
            window = to_numpy(genome_dev[i:i + L + PAM_LEN])
            mmpos = tuple(int(p) for p in np.nonzero(window[:L] != guide_np)[0])
            pos = i if strand == "+" else glen - (i + L + PAM_LEN)
            hits.append(Hit(guide_str, chrom, pos, strand, int(k), mmpos,
                            decode(window[:L]) + " " + decode(window[L:]),
                            off_target_score(mmpos, int(k))))
    return hits


def search_guide(guide: str, genome_seq: str, chrom: str = "chr", max_mismatches: int = 3,
                 device: str = "auto"):
    """Find all NGG-adjacent sites within `max_mismatches` of `guide` on both strands."""
    xp, _ = get_backend(device)
    guide_np = encode(guide)
    if (guide_np > 3).any():
        raise ValueError("Guide must contain only A/C/G/T.")
    g_fwd = encode(genome_seq)
    g_rev = reverse_complement(g_fwd)
    guide_dev = xp.asarray(guide_np)
    hits = []
    for strand, arr in (("+", g_fwd), ("-", g_rev)):
        hits += _scan_strand(xp, xp.asarray(arr), guide_np, guide_dev, max_mismatches,
                             strand, chrom, len(g_fwd), guide.upper())
    hits.sort(key=lambda h: (h.mismatches, -h.score, h.chrom, h.position))
    return hits


def search_guides(guides, fasta_records, max_mismatches=3, device="auto"):
    """Search many guides against many FASTA records [(name, seq), ...]."""
    out = {}
    for gd in guides:
        res = []
        for name, seq in fasta_records:
            res += search_guide(gd, seq, name, max_mismatches, device)
        res.sort(key=lambda h: (h.mismatches, -h.score))
        out[gd] = res
    return out


def specificity_score(hits) -> float:
    """Guide specificity 0-100 (higher = safer). Excludes the on-target exact hit."""
    off = [h for h in hits if h.mismatches > 0]
    total = sum(h.score for h in off)
    return round(10000.0 / (100.0 + total), 2)
