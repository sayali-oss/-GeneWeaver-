"""Genuine dynamic-programming alignment: Needleman-Wunsch (global) and Smith-Waterman (local).

Both use a linear gap penalty and full traceback.  Tie-breaking order during traceback is
diagonal > up (gap in target) > left (gap in query), which makes results deterministic.
Smith-Waterman starts the traceback at the FIRST cell (row-major) holding the maximum score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import config

_DIAG, _UP, _LEFT, _STOP = 1, 2, 3, 0


@dataclass
class Alignment:
    algorithm: str
    score: int
    query_aln: str
    target_aln: str
    midline: str                 # '|' match, 'x' mismatch, ' ' gap
    query_start: int             # 0-based, half-open
    query_end: int
    target_start: int
    target_end: int
    matches: int
    mismatches: int
    gaps: int
    length: int
    identity: float              # matches / alignment length * 100
    mismatch_positions: list     # 1-based positions in the QUERY (guide) of mismatched columns

    def to_dict(self) -> dict:
        return asdict(self)


def _build(algorithm, score, qa, ta, qs, qe, ts, te) -> Alignment:
    mid, mm_pos, matches, mism, gaps = [], [], 0, 0, 0
    qpos = qs
    for a, b in zip(qa, ta):
        if a == "-" or b == "-":
            mid.append(" ")
            gaps += 1
            if a != "-":
                qpos += 1
            continue
        qpos += 1
        if a == b:
            mid.append("|")
            matches += 1
        else:
            mid.append("x")
            mism += 1
            mm_pos.append(qpos)           # qpos is now the 1-based query index of this column
    length = len(qa)
    ident = round(matches / length * 100, 2) if length else 0.0
    return Alignment(algorithm, int(score), qa, ta, "".join(mid), qs, qe, ts, te,
                     matches, mism, gaps, length, ident, mm_pos)


def _sub(a: str, b: str, s: config.AlignmentScheme) -> int:
    # N never counts as a match
    return s.match if (a == b and a != "N") else s.mismatch


def needleman_wunsch(query: str, target: str, scheme: config.AlignmentScheme = config.DEFAULT_SCHEME) -> Alignment:
    """Global alignment of query vs target."""
    q, t = query.upper(), target.upper()
    n, m = len(q), len(t)
    H = [[0] * (m + 1) for _ in range(n + 1)]
    P = [[_STOP] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        H[i][0] = i * scheme.gap
        P[i][0] = _UP
    for j in range(1, m + 1):
        H[0][j] = j * scheme.gap
        P[0][j] = _LEFT
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = H[i - 1][j - 1] + _sub(q[i - 1], t[j - 1], scheme)
            u = H[i - 1][j] + scheme.gap
            l = H[i][j - 1] + scheme.gap
            best = max(d, u, l)
            H[i][j] = best
            P[i][j] = _DIAG if best == d else (_UP if best == u else _LEFT)
    qa, ta = [], []
    i, j = n, m
    while i > 0 or j > 0:
        p = P[i][j]
        if p == _DIAG:
            qa.append(q[i - 1]); ta.append(t[j - 1]); i -= 1; j -= 1
        elif p == _UP:
            qa.append(q[i - 1]); ta.append("-"); i -= 1
        else:
            qa.append("-"); ta.append(t[j - 1]); j -= 1
    return _build("Needleman-Wunsch", H[n][m], "".join(reversed(qa)), "".join(reversed(ta)), 0, n, 0, m)


def smith_waterman(query: str, target: str, scheme: config.AlignmentScheme = config.DEFAULT_SCHEME) -> Alignment:
    """Local alignment of query vs target."""
    q, t = query.upper(), target.upper()
    n, m = len(q), len(t)
    H = [[0] * (m + 1) for _ in range(n + 1)]
    P = [[_STOP] * (m + 1) for _ in range(n + 1)]
    best, bi, bj = 0, 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = H[i - 1][j - 1] + _sub(q[i - 1], t[j - 1], scheme)
            u = H[i - 1][j] + scheme.gap
            l = H[i][j - 1] + scheme.gap
            v = max(0, d, u, l)
            H[i][j] = v
            if v == 0:
                P[i][j] = _STOP
            elif v == d:
                P[i][j] = _DIAG
            elif v == u:
                P[i][j] = _UP
            else:
                P[i][j] = _LEFT
            if v > best:
                best, bi, bj = v, i, j
    if best == 0:
        return _build("Smith-Waterman", 0, "", "", 0, 0, 0, 0)
    qa, ta = [], []
    i, j = bi, bj
    while i > 0 and j > 0 and H[i][j] > 0:
        p = P[i][j]
        if p == _DIAG:
            qa.append(q[i - 1]); ta.append(t[j - 1]); i -= 1; j -= 1
        elif p == _UP:
            qa.append(q[i - 1]); ta.append("-"); i -= 1
        elif p == _LEFT:
            qa.append("-"); ta.append(t[j - 1]); j -= 1
        else:
            break
    return _build("Smith-Waterman", best, "".join(reversed(qa)), "".join(reversed(ta)), i, bi, j, bj)


def alignment_to_text(aln: Alignment, width: int = 60) -> str:
    """Plain-text alignment block (for reports)."""
    lines = []
    for k in range(0, max(aln.length, 1), width):
        lines.append(f"Guide  {aln.query_aln[k:k + width]}")
        lines.append(f"       {aln.midline[k:k + width]}")
        lines.append(f"Target {aln.target_aln[k:k + width]}")
    return "\n".join(lines)


def alignment_to_html(aln: Alignment) -> str:
    """Coloured HTML alignment (matches green, mismatches red, gaps amber)."""
    def span(ch, cls):
        return f'<span class="aln-{cls}">{ch}</span>'
    top, mid, bot = [], [], []
    for a, m, b in zip(aln.query_aln, aln.midline, aln.target_aln):
        cls = "match" if m == "|" else ("mismatch" if m == "x" else "gap")
        top.append(span(a, cls)); mid.append(span(m if m != " " else "&nbsp;", cls)); bot.append(span(b, cls))
    return ('<div class="aln-block"><div><b>Guide&nbsp;&nbsp;</b>' + "".join(top) + "</div>"
            '<div><b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b>' + "".join(mid) + "</div>"
            '<div><b>Target&nbsp;</b>' + "".join(bot) + "</div></div>")
