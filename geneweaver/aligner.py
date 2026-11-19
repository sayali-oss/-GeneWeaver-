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
