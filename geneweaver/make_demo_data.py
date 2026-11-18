"""Create a synthetic genome with planted on-/off-target sites for demos & tests."""
import random

GUIDE = "GACGCATAAAGATGAGACGC"


def mutate(s, positions):
    s = list(s)
    for p in positions:
        s[p] = random.choice([b for b in "ACGT" if b != s[p]])
    return "".join(s)


def revcomp(s):
    return s[::-1].translate(str.maketrans("ACGT", "TGCA"))


def build(n=200_000, seed=7, path="data/demo_genome.fa"):
    random.seed(seed)
    g = list("".join(random.choices("ACGT", k=n)))
    plants = [(1000, GUIDE, "+"), (50000, mutate(GUIDE, [3]), "+"),
              (90000, mutate(GUIDE, [5, 12]), "-"), (150000, mutate(GUIDE, [0, 8, 17]), "+")]
    for pos, site, strand in plants:
        full = site + "TGG"
        if strand == "-":
            full = revcomp(full)
        g[pos:pos + 23] = full
    seq = "".join(g)
    with open(path, "w") as fh:
        fh.write(">chrDemo\n")
        for i in range(0, len(seq), 70):
            fh.write(seq[i:i + 70] + "\n")
    open("data/guides.txt", "w").write(GUIDE + "\n")
    return plants


if __name__ == "__main__":
    print(build())
