"""Generate the reproducible SYNTHETIC sample dataset (data/sample_*.fasta + sample_truth.json).

Source: random sequences from numpy.random.default_rng(seed=7) with known target sites planted
at recorded positions.  Limitation: synthetic, not a real genome - random sequence has no
biological repeat structure.  The ground truth lets the tests verify that the engine recovers
every planted site.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.preprocessing import reverse_complement  # noqa: E402

rng = np.random.default_rng(7)
rand = lambda n: "".join(rng.choice(list("ACGT"), n))            # noqa: E731
mutate = lambda s, pos: "".join((("ACGT"[("ACGT".index(c) + 1) % 4]) if (i + 1) in pos else c)  # noqa: E731
                                for i, c in enumerate(s))

g1, g2, g3 = rand(20), rand(20), rand(20)
refs = {"synthetic_chrA": list(rand(20000)), "synthetic_chrB": list(rand(12000))}
truth = []


def plant(ref, start0, guide, pam, strand, mm_pos):
    site = mutate(guide, set(mm_pos)) + pam            # protospacer + PAM, 5'->3' on its own strand
    if strand == "-":
        site = reverse_complement(site)
    refs[ref][start0:start0 + len(site)] = list(site)
    truth.append({"reference": ref, "guide": {g1: "guide_1", g2: "guide_2"}[guide], "strand": strand,
                  "start": start0 + 1, "end": start0 + len(site), "mismatches": len(mm_pos),
                  "mismatch_positions": sorted(mm_pos), "pam": pam})


plant("synthetic_chrA", 3000, g1, "TGG", "+", [])                 # perfect target, forward
plant("synthetic_chrA", 9000, g1, "AGG", "-", [3, 8])             # 2 mismatches, reverse strand
plant("synthetic_chrA", 15000, g1, "CGG", "+", [2, 5, 11, 18])    # 4 mismatches
plant("synthetic_chrB", 5000, g1, "GGG", "+", [4, 9, 14])         # 3 mismatches
plant("synthetic_chrB", 2000, g2, "CGG", "-", [])                 # perfect target, reverse strand
plant("synthetic_chrA", 12000, g2, "AGG", "+", [1, 6])            # 2 mismatches (non-seed)
plant("synthetic_chrB", 8000, g1, "TAG", "+", [20])               # 1 mismatch, NO PAM (TAG != NGG)

def wrap(s, w=70):
    return "\n".join(s[i:i + w] for i in range(0, len(s), w))

(ROOT / "data").mkdir(exist_ok=True)
with open(ROOT / "data/sample_reference.fasta", "w") as f:
    for k, v in refs.items():
        f.write(f">{k} synthetic reference with planted CRISPR sites\n{wrap(''.join(v))}\n")
with open(ROOT / "data/sample_guides.fasta", "w") as f:
    for k, v in (("guide_1", g1), ("guide_2", g2), ("guide_3_no_site", g3)):
        f.write(f">{k}\n{v}\n")
(ROOT / "data/sample_truth.json").write_text(json.dumps(
    {"guides": {"guide_1": g1, "guide_2": g2, "guide_3_no_site": g3}, "planted": truth}, indent=2))
print("sample data written;", len(truth), "planted sites")
