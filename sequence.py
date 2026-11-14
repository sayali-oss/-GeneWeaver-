"""Sequence I/O and encoding. A,C,G,T -> 0,1,2,3 ; anything else (N...) -> 4."""
import numpy as np

_LUT = np.full(256, 4, dtype=np.uint8)
for i, b in enumerate("ACGT"):
    _LUT[ord(b)] = i
    _LUT[ord(b.lower())] = i
_COMP = np.array([3, 2, 1, 0, 4], dtype=np.uint8)  # A<->T, C<->G, N stays N


def encode(seq: str) -> np.ndarray:
    return _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def decode(arr) -> str:
    return "".join("ACGTN"[int(x)] for x in arr)


def reverse_complement(arr: np.ndarray) -> np.ndarray:
    return _COMP[arr[::-1]]


def read_fasta(path: str):
    """Yield (name, sequence_string) for each record in a FASTA file."""
    name, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks)
