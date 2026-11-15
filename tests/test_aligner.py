import random
from geneweaver.aligner import search_guide
from geneweaver.sequence import encode, reverse_complement, decode

G = "GACGCATAAAGATGAGACGC"


def rc(s):
    return s[::-1].translate(str.maketrans("ACGT", "TGCA"))


def test_encode_roundtrip():
    assert decode(encode("ACGTN")) == "ACGTN"


def test_revcomp():
    assert decode(reverse_complement(encode("AACG"))) == "CGTT"


def test_exact_forward_hit():
    seq = "T" * 50 + G + "TGG" + "A" * 50
    hits = search_guide(G, seq, device="cpu", max_mismatches=0)
    assert len(hits) == 1 and hits[0].position == 50 and hits[0].strand == "+"


def test_mismatch_and_pam_required():
    mut = G[:4] + ("A" if G[4] != "A" else "C") + G[5:]
    good = "T" * 30 + mut + "AGG" + "T" * 30
    bad_pam = "T" * 30 + mut + "ATG" + "T" * 30
    assert len(search_guide(G, good, device="cpu", max_mismatches=1)) == 1
    assert len(search_guide(G, bad_pam, device="cpu", max_mismatches=1)) == 0


def test_reverse_strand_coordinates():
    seq = "T" * 40 + rc(G + "CGG") + "A" * 40
    hits = search_guide(G, seq, device="cpu", max_mismatches=0)
    assert len(hits) == 1 and hits[0].strand == "-" and hits[0].position == 40


def test_matches_naive_on_random():
    random.seed(3)
    seq = "".join(random.choices("ACGT", k=20000))
    exp = 0
    for i in range(len(seq) - 22):
        if seq[i + 21:i + 23] == "GG" and sum(a != b for a, b in zip(G, seq[i:i + 20])) <= 6:
            exp += 1
    fwd = [h for h in search_guide(G, seq, device="cpu", max_mismatches=6) if h.strand == "+"]
    assert len(fwd) == exp
