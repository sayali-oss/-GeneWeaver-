import config
from backend.pam_detector import check_pam, find_pam_sites, count_pam_sites, site_layout
from backend.preprocessing import prepare_reference, reverse_complement
from backend.sequence_loader import load_fasta_text

SP = config.CRISPR_SYSTEMS["SpCas9"]


def ref(seq):
    return prepare_reference(load_fasta_text(f">x\n{seq}").records[0])


def test_ngg_matching():
    assert check_pam("AGG", "NGG") and check_pam("TGG", "NGG") and check_pam("CGG", "NGG")
    assert not check_pam("TAG", "NGG") and not check_pam("AGA", "NGG")


def test_iupac_systems():
    assert check_pam("ACGAAT", "NNGRRT") and not check_pam("ACGCCT", "NNGRRT")
    assert check_pam("TTTA", "TTTV") and not check_pam("TTTT", "TTTV")


def test_forward_and_reverse_pam_sites():
    r = ref("AAAAAAAAAATGGAAAAAAAAAA")           # TGG forward at 1-based 11-13
    sites = find_pam_sites(r, SP)
    fwd = [s for s in sites if s["strand"] == "+"]
    assert any(s["start"] == 11 and s["pam"] == "TGG" for s in fwd)
    r2 = ref(reverse_complement("AAAAAAAAAATGGAAAAAAAAAA"))   # same site now on the reverse strand
    rev = [s for s in find_pam_sites(r2, SP) if s["strand"] == "-"]
    assert any(s["pam"] == "TGG" for s in rev)


def test_missing_pam():
    r = ref("ACACACACACACACACACACACAC")
    assert count_pam_sites(r, SP)["+"] == 0


def test_layouts():
    assert site_layout(SP, 20) == {"spacer_off": 0, "pam_off": 20, "window": 23}
    cas12 = config.CRISPR_SYSTEMS["Cas12a"]
    assert site_layout(cas12, 23) == {"spacer_off": 4, "pam_off": 0, "window": 27}


def test_n_never_satisfies_pam():
    r = ref("AAAAAAAAAANGGAAAAA")
    assert all(s["start"] != 11 for s in find_pam_sites(r, SP) if s["strand"] == "+")
