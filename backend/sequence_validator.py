"""Input validation for reference sequences and guide RNAs."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import config

_DNA = set("ACGTN")
_IUPAC_EXTRA = set("RYSWKMBDHV")


@dataclass
class ValidationResult:
    valid: bool
    cleaned: str = ""
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    log: list = field(default_factory=list)     # preprocessing operations applied


def clean_sequence(raw: str, kind: str = "reference") -> ValidationResult:
    """Normalise and validate one nucleotide sequence.

    * strips whitespace/digits (GenBank-style numbering), upper-cases
    * converts RNA (U) to DNA (T)
    * reference: IUPAC ambiguity codes -> N (warned); anything else is an error
    * guide: only A/C/G/T/U allowed (no N)
    """
    res = ValidationResult(valid=False)
    if raw is None or not str(raw).strip():
        res.errors.append(f"The {kind} sequence is empty.")
        return res

    seq = str(raw)
    stripped = re.sub(r"[\s\d]", "", seq)
    if stripped != seq:
        res.log.append("Removed whitespace/digits")
    seq = stripped.upper()
    if seq != stripped:
        res.log.append("Converted to upper case")
    if "U" in seq:
        res.log.append(f"Converted {seq.count('U')} RNA 'U' to DNA 'T'")
        seq = seq.replace("U", "T")

    chars = set(seq)
    if kind == "guide":
        bad = chars - set("ACGT")
        if bad:
            res.errors.append(f"Guide contains invalid characters: {', '.join(sorted(bad))} "
                              "(only A, C, G, T/U allowed).")
    else:
        extra = chars & _IUPAC_EXTRA
        if extra:
            n = sum(seq.count(c) for c in extra)
            res.warnings.append(f"{n} IUPAC ambiguity codes ({', '.join(sorted(extra))}) converted to N.")
            res.log.append(f"Converted {n} ambiguity codes to N")
            seq = re.sub("[RYSWKMBDHV]", "N", seq)
            chars = set(seq)
        bad = chars - _DNA
        if bad:
            res.errors.append(f"Reference contains invalid characters: {', '.join(sorted(bad))}.")

    if not seq and not res.errors:
        res.errors.append(f"The {kind} sequence is empty after cleaning.")
    if res.errors:
        res.cleaned = seq
        return res

    if kind == "guide":
        if not (config.MIN_GUIDE_LEN <= len(seq) <= config.MAX_GUIDE_LEN):
            res.errors.append(f"Guide length {len(seq)} is outside the supported range "
                              f"{config.MIN_GUIDE_LEN}-{config.MAX_GUIDE_LEN} nt.")
    else:
        if seq.count("N") / len(seq) > 0.5:
            res.warnings.append("More than 50% of the reference is N (unknown bases).")

    res.cleaned = seq
    res.valid = not res.errors
    return res


def composition(seq: str) -> dict:
    """Nucleotide counts and GC content (GC% is computed over A/C/G/T only)."""
    counts = {b: seq.count(b) for b in "ACGTN"}
    acgt = sum(counts[b] for b in "ACGT")
    gc = (counts["G"] + counts["C"]) / acgt * 100 if acgt else 0.0
    return {"length": len(seq), "counts": counts, "gc_percent": round(gc, 2)}


def validate_config(guide_lens, ref_lengths, system, max_mismatches) -> list:
    """Return a list of configuration errors (empty list = OK)."""
    errs = []
    if system.key not in config.CRISPR_SYSTEMS:
        errs.append(f"Unknown CRISPR system '{system.key}'.")
    if not (0 <= max_mismatches <= config.MAX_MISMATCHES_LIMIT):
        errs.append(f"Max mismatches must be between 0 and {config.MAX_MISMATCHES_LIMIT}.")
    if not ref_lengths:
        errs.append("At least one reference sequence is required.")
    if not guide_lens:
        errs.append("At least one guide sequence is required.")
    if len(guide_lens) > config.MAX_GUIDES:
        errs.append(f"At most {config.MAX_GUIDES} guides are supported per run.")
    if sum(ref_lengths) > config.MAX_REFERENCE_BP:
        errs.append(f"Total reference length {sum(ref_lengths):,} bp exceeds the limit "
                    f"of {config.MAX_REFERENCE_BP:,} bp.")
    pam_len = len(system.pam)
    for g in guide_lens:
        if ref_lengths and min(ref_lengths) < g + pam_len:
            errs.append(f"A reference is shorter than guide + PAM ({g + pam_len} nt).")
            break
    return errs
