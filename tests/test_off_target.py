import numpy as np
import pytest

import config
from backend.cpu_engine import CPUScanner, scan_strand_python, PAM_FAIL
from backend.off_target_detector import extract_candidates
from backend.pam_detector import pam_lut, site_layout
from backend.pipeline import run_analysis, AnalysisConfigError
from backend.preprocessing import prepare_reference, encode, reverse_complement
from backend.sequence_loader import load_fasta_text, load_guides_text

SP = config.CRISPR_SYSTEMS["SpCas9"]
G = "GACGCATAAAGATGAGACGC"
FLANK = "ACACACTTGTCTCTGAGAGTCATCATCA"


def run(seq, guide=G, mm=4, pam=True):
    refs = load_fasta_text(f">r\n{seq}").records
    gs = load_guides_text(f"g {guide}").records
    return run_analysis(refs, gs, "SpCas9", mm, pam, "cpu", save=False)


def mut(s, pos):
    return "".join(("ACGT"[("ACGT".index(c) + 1) % 4] if i + 1 in pos else c) for i, c in enumerate(s))


def test_exact_match_forward():
    r = run(FLANK + G + "TGG" + FLANK)
    c = r["candidates"][0]
    assert c["mismatches"] == 0 and c["strand"] == "+" and c["start"] == len(FLANK) + 1
    assert c["pam_seq"] == "TGG" and c["classification"] == "Perfect-match target"
    assert r["summary"]["perfect_targets"] == 1


def test_exact_match_reverse_strand_coordinates():
    site = G + "AGG"
    seq = FLANK + reverse_complement(site) + FLANK
    c = run(seq)["candidates"][0]
    assert c["strand"] == "-" and c["mismatches"] == 0 and c["pam_seq"] == "AGG"
    # protospacer occupies the 3' end of the forward-strand footprint (PAM is at the left)
    assert c["start"] == len(FLANK) + 3 + 1 and c["end"] == len(FLANK) + 23


def test_one_and_multiple_mismatches():
    one = run(FLANK + mut(G, {7}) + "CGG" + FLANK)["candidates"][0]
    assert one["mismatches"] == 1 and one["mismatch_positions"] == [7]
    multi = run(FLANK + mut(G, {2, 9, 15, 20}) + "AGG" + FLANK)["candidates"][0]
    assert multi["mismatches"] == 4 and multi["mismatch_positions"] == [2, 9, 15, 20]
    assert multi["seed_mismatches"] == 3          # seed = guide positions 9-20 -> 9, 15, 20


def test_missing_pam_excluded_or_flagged():
    seq = FLANK + G + "TAG" + FLANK
    assert run(seq)["candidates"] == []
    flagged = run(seq, pam=False)["candidates"]
    assert flagged and not flagged[0]["pam_ok"] and flagged[0]["classification"] == "Perfect match, no PAM"


def test_multiple_sites_and_ranking():
    seq = FLANK + mut(G, {3, 5}) + "AGG" + FLANK + G + "CGG" + FLANK + mut(G, {1}) + "TGG" + FLANK
    c = run(seq)["candidates"]
    assert [x["mismatches"] for x in c] == [0, 1, 2]
    assert [x["rank"] for x in c] == [1, 2, 3]
    assert c[0]["similarity_score"] > c[1]["similarity_score"] > c[2]["similarity_score"]


def test_no_matching_site():
    r = run("ACACACACAGTGTGTGTGACACACACACAGTGTGTGTGAC" * 3)
    assert r["candidates"] == [] and r["summary"]["total_candidates"] == 0


def test_mismatch_limit_respected():
    seq = FLANK + mut(G, {1, 4, 8, 12, 16}) + "AGG" + FLANK
    assert run(seq, mm=4)["candidates"] == [] and len(run(seq, mm=5)["candidates"]) == 1


def test_numpy_matches_python_reference():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 5, 3000, dtype=np.uint8)           # includes N (=4)
    guide = encode(G)
    lay, lut = site_layout(SP, 20), pam_lut("NGG")
    from backend.cpu_engine import scan_strand_numpy
    for req in (True, False):
        fast = scan_strand_numpy(codes, guide, lay, lut, req)
        slow = scan_strand_python(codes.tolist(), guide.tolist(), lay, lut, req)
        assert fast.tolist() == slow


def test_cas12a_5prime_pam():
    cas = config.CRISPR_SYSTEMS["Cas12a"]
    g23 = "GACGCATAAAGATGAGACGCTTA"
    seq = FLANK + "TTTA" + g23 + FLANK
    refs = load_fasta_text(f">r\n{seq}").records
    r = run_analysis(refs, load_guides_text(f"g {g23}").records, "Cas12a", 2, True, "cpu", save=False)
    c = r["candidates"][0]
    assert c["pam_seq"] == "TTTA" and c["mismatches"] == 0 and c["start"] == len(FLANK) + 5


def test_planted_sites_all_recovered(sample):
    refs, guides, truth = sample
    res = run_analysis(refs.records, guides.records, "SpCas9", 4, True, "cpu", save=False)
    found = {(c["guide_id"], c["reference"], c["strand"], c["start"], c["end"], c["mismatches"],
              tuple(c["mismatch_positions"])) for c in res["candidates"]}
    for p in truth["planted"]:
        key = (p["guide"], p["reference"], p["strand"], None, None, p["mismatches"], tuple(p["mismatch_positions"]))
        if p["pam"].endswith("GG"):
            assert any(f[:3] == key[:3] and f[5:] == key[5:] for f in found), p
        else:                                              # TAG -> must NOT be reported (PAM required)
            assert not any(f[1] == p["reference"] and f[5] == p["mismatches"] and f[0] == p["guide"]
                           and abs(f[3] - p["start"]) < 3 for f in found)
    assert not any(c["guide_id"] == "guide_3_no_site" for c in res["candidates"])


def test_config_errors_raise():
    refs = load_fasta_text(">r\nACGT" * 20).records
    gs = load_guides_text(f"g {G}").records
    with pytest.raises(AnalysisConfigError):
        run_analysis(refs, gs, "SpCas9", 99, True, "cpu", save=False)
    with pytest.raises(AnalysisConfigError):
        run_analysis(load_fasta_text(">r\nACGT").records, gs, "SpCas9", 3, True, "cpu", save=False)


def test_rank_is_global_per_guide_across_references(sample):
    refs, guides, _ = sample
    res = run_analysis(refs.records, guides.records, "SpCas9", 4, True, "cpu", save=False)
    for gid in ("guide_1", "guide_2"):
        cs = [c for c in res["candidates"] if c["guide_id"] == gid]
        assert [c["rank"] for c in cs] == list(range(1, len(cs) + 1))
        assert [c["mismatches"] for c in cs] == sorted(c["mismatches"] for c in cs)
