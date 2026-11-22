"""Sequence preprocessing: numeric encoding, reverse complement, reference preparation.

Encoding: A=0 C=1 G=2 T=3 N=4.  Both strands are prepared once per reference so that the
scan engines treat every window identically (reverse-strand search = forward search on the
reverse complement, with coordinates mapped back).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_COMP = str.maketrans("ACGTN", "TGCAN")
_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    _LUT[ord(_b)] = _i
_DECODE = np.array(list("ACGTN"))


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def rna_to_dna(seq: str) -> str:
    return seq.upper().replace("U", "T")


def dna_to_rna(seq: str) -> str:
    return seq.upper().replace("T", "U")


def encode(seq: str) -> np.ndarray:
    """String -> uint8 codes (A0 C1 G2 T3 N4)."""
    return _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def decode(codes) -> str:
    return "".join(_DECODE[np.asarray(codes, dtype=np.int64)])


@dataclass
class PreparedReference:
    id: str
    sequence: str
    fwd: np.ndarray          # encoded forward strand
    rev: np.ndarray          # encoded reverse-complement strand
    length: int
    log: list


def prepare_reference(record) -> PreparedReference:
    seq = record.sequence
    fwd = encode(seq)
    rev = encode(reverse_complement(seq))
    log = list(record.log) + ["Encoded forward strand (A0 C1 G2 T3 N4)",
                              "Generated reverse-complement strand"]
    return PreparedReference(record.id, seq, fwd, rev, len(seq), log)


def strand_to_forward(offset: int, length: int, n: int, strand: str) -> tuple:
    """Map [offset, offset+length) on a strand array to 0-based half-open forward coordinates."""
    if strand == "+":
        return offset, offset + length
    return n - offset - length, n - offset
