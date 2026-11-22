"""FASTA / raw-sequence loading (Biopython)."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

from Bio import SeqIO

from backend.sequence_validator import clean_sequence, composition

ALLOWED_EXT = (".fasta", ".fa", ".fna", ".fas", ".txt")


@dataclass
class SeqRecordLite:
    id: str
    description: str
    original: str            # untouched input (preserved)
    sequence: str            # cleaned/normalised sequence
    log: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


@dataclass
class LoadResult:
    records: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.records) and not self.errors


def _decode(data) -> str:
    if isinstance(data, bytes):
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return data.decode("latin-1")
    return str(data)


def load_fasta_text(data, kind: str = "reference", default_id: str = "sequence") -> LoadResult:
    """Parse FASTA text/bytes.  Text without a '>' header is treated as one raw sequence."""
    out = LoadResult()
    text = _decode(data)
    if not text.strip():
        out.errors.append("The input is empty.")
        return out
    if "\x00" in text:
        out.errors.append("The file looks binary/corrupted (contains NUL bytes).")
        return out

    if ">" not in text:                       # raw pasted sequence
        raw_records = [(default_id, "", text)]
    else:
        if not text.lstrip().startswith(">"):
            out.errors.append("Corrupted FASTA: content appears before the first '>' header line.")
            return out
        try:
            raw_records = [(r.id, r.description, str(r.seq))
                           for r in SeqIO.parse(io.StringIO(text), "fasta")]
        except Exception as exc:  # noqa: BLE001 - Biopython raises several types
            out.errors.append(f"Could not parse FASTA: {exc}")
            return out
        if not raw_records:
            out.errors.append("No FASTA records found.")
            return out

    seen = {}
    for rid, desc, raw in raw_records:
        rid = rid or default_id
        seen[rid] = seen.get(rid, 0) + 1
        uid = rid if seen[rid] == 1 else f"{rid}_{seen[rid]}"
        res = clean_sequence(raw, kind)
        if not res.valid:
            out.errors.extend(f"[{uid}] {e}" for e in res.errors)
            continue
        out.records.append(SeqRecordLite(uid, desc, raw, res.cleaned, res.log, res.warnings,
                                         composition(res.cleaned)))
        out.warnings.extend(f"[{uid}] {w}" for w in res.warnings)
    return out


def load_fasta_file(path, kind: str = "reference") -> LoadResult:
    p = Path(path)
    if not p.exists():
        return LoadResult(errors=[f"File not found: {p}"])
    if p.suffix.lower() not in ALLOWED_EXT:
        return LoadResult(errors=[f"Unsupported file type '{p.suffix}'. Use .fasta, .fa or .fna."])
    try:
        return load_fasta_text(p.read_bytes(), kind, default_id=p.stem)
    except OSError as exc:
        return LoadResult(errors=[f"Could not read {p.name}: {exc}"])


def load_guides_text(data) -> LoadResult:
    """Guides may be FASTA, or one sequence per line (optionally 'name SEQ')."""
    text = _decode(data)
    if ">" in text:
        return load_fasta_text(text, kind="guide", default_id="guide")
    out = LoadResult()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        out.errors.append("No guide sequences provided.")
        return out
    for i, ln in enumerate(lines, 1):
        parts = ln.replace(",", " ").split()
        name, seq = (parts[0], parts[1]) if len(parts) >= 2 else (f"guide_{i}", parts[0])
        res = clean_sequence(seq, "guide")
        if not res.valid:
            out.errors.extend(f"[{name}] {e}" for e in res.errors)
            continue
        out.records.append(SeqRecordLite(name, "", seq, res.cleaned, res.log, res.warnings,
                                         composition(res.cleaned)))
    return out
