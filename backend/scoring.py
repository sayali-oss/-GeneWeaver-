"""Transparent candidate scoring.

IMPORTANT: this is a documented HEURISTIC for ranking candidate sites by sequence similarity.
It is NOT a validated off-target probability (unlike e.g. CFD or MIT specificity models).

Similarity score (0-100)
    weight(position) = SEED_WEIGHT (2.0) for the PAM-proximal seed region, otherwise 1.0
    score = 100 * (1 - sum(weight of mismatched positions) / sum(weight of all positions))
A perfect match scores 100.  PAM compatibility is reported separately and used first when ranking.

Ranking order (best first):
    1. PAM-compatible sites before non-PAM sites
    2. fewer total mismatches
    3. fewer seed-region mismatches
    4. higher similarity score
    5. reference name / position (for a deterministic order)
"""
from __future__ import annotations

import config

SEED_WEIGHT = 2.0
NONSEED_WEIGHT = 1.0


def position_weights(guide_len: int, system: config.CrisprSystem) -> list:
    """Weight per guide position (index 0 = guide 5' end)."""
    seed = min(system.seed_len, guide_len)
    w = [NONSEED_WEIGHT] * guide_len
    rng = range(guide_len - seed, guide_len) if system.pam_side == "3prime" else range(0, seed)
    for i in rng:
        w[i] = SEED_WEIGHT
    return w


def seed_positions(guide_len: int, system: config.CrisprSystem) -> set:
    """1-based guide positions belonging to the PAM-proximal seed."""
    seed = min(system.seed_len, guide_len)
    if system.pam_side == "3prime":
        return set(range(guide_len - seed + 1, guide_len + 1))
    return set(range(1, seed + 1))


def similarity_score(mismatch_positions, guide_len: int, system: config.CrisprSystem) -> float:
    """mismatch_positions are 1-based guide positions."""
    w = position_weights(guide_len, system)
    penalty = sum(w[p - 1] for p in mismatch_positions)
    return round(100.0 * (1.0 - penalty / sum(w)), 2)


def rank_key(c: dict):
    return (0 if c["pam_ok"] else 1, c["mismatches"], c["seed_mismatches"],
            -c["similarity_score"], c["reference"], c["start"])


def explain(c: dict, system: config.CrisprSystem) -> str:
    seed = system.seed_len
    return (f"{c['mismatches']} mismatch(es) at guide position(s) {c['mismatch_positions'] or '-'}; "
            f"{c['seed_mismatches']} in the {seed}-nt PAM-proximal seed (weight {SEED_WEIGHT}x), "
            f"others weight {NONSEED_WEIGHT}x -> similarity {c['similarity_score']}/100. "
            f"PAM {c['pam_seq']} {'matches' if c['pam_ok'] else 'does NOT match'} {system.pam}. "
            "Computational candidate prediction only.")
