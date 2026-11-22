import config
from backend.alignment_engine import needleman_wunsch, smith_waterman, Alignment

S = config.AlignmentScheme(match=1, mismatch=-1, gap=-1)


def test_nw_textbook_score():
    # classic example GCATGCU vs GATTACA: optimal score 0 with +1/-1/-1
    a = needleman_wunsch("GCATGCT", "GATTACA", S)
    assert a.score == 0 and len(a.query_aln) == len(a.target_aln)


def test_nw_exact_and_one_mismatch():
    g = "GACGCATAAAGATGAGACGC"
    e = needleman_wunsch(g, g)
    assert e.mismatches == 0 and e.identity == 100.0 and e.score == 20 * 2
    t = g[:5] + ("A" if g[5] != "A" else "C") + g[6:]
    m = needleman_wunsch(g, t)
    assert m.mismatches == 1 and m.mismatch_positions == [6] and m.gaps == 0


def test_nw_multiple_mismatches_positions():
    g = "ACGTACGTACGTACGTACGT"
    t = list(g); t[1] = "T"; t[10] = "A"; t[19] = "C"
    a = needleman_wunsch(g, "".join(t))
    assert a.mismatch_positions == [2, 11, 20]


def test_nw_gap_detection():
    a = needleman_wunsch("ACGTACGTAC", "ACGTAGTAC")           # one deletion in target
    assert a.gaps == 1 and "-" in a.target_aln


def test_sw_local():
    a = smith_waterman("TTGACGGATT", "CCCCCTGACGGACCCCC", config.AlignmentScheme(2, -3, -5))
    assert a.score > 0 and a.query_aln == a.target_aln and a.target_end - a.target_start == len(a.target_aln)
    assert a.length == len(a.midline)


def test_sw_no_similarity():
    a = smith_waterman("AAAA", "CCCC")
    assert a.score == 0 and a.length == 0


def test_alignment_dataclass_roundtrip():
    a = needleman_wunsch("ACGT", "ACGA")
    assert Alignment(**a.to_dict()).score == a.score


def test_not_substring_search():
    # a substring search would find nothing here; real alignment still scores the near match
    a = smith_waterman("GATTACAGATTACA", "GATTTCAGATTACA")
    assert a.score > 20 and a.mismatches >= 1
