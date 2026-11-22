import pytest
from backend.sequence_loader import load_fasta_file, load_fasta_text, load_guides_text
from backend.sequence_validator import clean_sequence, composition, validate_config
import config


def test_valid_fasta_file(sample):
    refs, _, _ = sample
    assert refs.ok and len(refs.records) == 2
    assert refs.records[0].info["length"] == 20000
    assert 0 < refs.records[0].info["gc_percent"] < 100


def test_composition_counts():
    c = composition("AACGTTN")
    assert c["counts"] == {"A": 2, "C": 1, "G": 1, "T": 2, "N": 1} and c["gc_percent"] == round(2 / 6 * 100, 2)


def test_invalid_fasta_content_before_header():
    r = load_fasta_text("ACGT\n>seq\nACGT")
    assert not r.ok and "before the first" in r.errors[0]


def test_invalid_fasta_binary():
    assert not load_fasta_text(b"\x00\x01>abc").ok


def test_missing_file(tmp_path):
    r = load_fasta_file(tmp_path / "nope.fasta")
    assert not r.ok and "not found" in r.errors[0]


def test_wrong_extension(tmp_path):
    p = tmp_path / "x.pdf"; p.write_text(">a\nACGT")
    assert "Unsupported" in load_fasta_file(p).errors[0]


def test_empty_sequence():
    assert not load_fasta_text("   \n").ok
    assert not load_fasta_text(">a\n").ok or load_fasta_text(">a\n").errors
    assert not clean_sequence("", "guide").valid


def test_invalid_characters():
    r = clean_sequence("ACGTXZ1!", "reference")
    assert not r.valid and "invalid" in r.errors[0].lower()
    assert not clean_sequence("ACGTNACGTACGTACGTACG", "guide").valid     # N not allowed in guides


def test_rna_and_ambiguity_handling():
    g = clean_sequence("GACGCAUAAAGAUGAGACGC", "guide")
    assert g.valid and "U" not in g.cleaned and g.log
    r = clean_sequence("ACGTRYACGT", "reference")
    assert r.valid and r.cleaned == "ACGTNNACGT" and r.warnings


def test_raw_and_multi_fasta_and_duplicate_ids():
    assert load_fasta_text("ACGT ACGT").records[0].sequence == "ACGTACGT"
    r = load_fasta_text(">a\nACGT\n>a\nTTTT\n")
    assert [x.id for x in r.records] == ["a", "a_2"]


def test_original_preserved():
    r = load_fasta_text(">a\nacg t\n")
    assert r.records[0].original.strip() != r.records[0].sequence and r.records[0].sequence == "ACGT"


def test_guide_length_limits():
    assert not load_guides_text("ACGT").ok
    assert load_guides_text("g1 GACGCATAAAGATGAGACGC").ok


def test_config_validation():
    sp = config.CRISPR_SYSTEMS["SpCas9"]
    assert validate_config([20], [100], sp, 3) == []
    assert validate_config([20], [], sp, 3)
    assert validate_config([20], [100], sp, 99)
    assert validate_config([20], [10], sp, 3)          # reference shorter than guide+PAM
    assert validate_config([], [100], sp, 3)
