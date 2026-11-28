"""CSV / JSON / PDF report generation from real analysis and benchmark results."""
from __future__ import annotations

import csv
import io
import json
from collections import Counter

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)
from xml.sax.saxutils import escape

import config
from backend.alignment_engine import alignment_to_text

CSV_COLUMNS = ["rank", "guide_id", "reference", "strand", "start", "end", "pam_seq", "pam_ok",
               "mismatches", "mismatch_positions", "seed_mismatches", "similarity_score",
               "alignment_score", "sw_score", "classification", "target_seq", "explanation"]


def candidates_csv(results: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for c in results["candidates"]:
        row = []
        for k in CSV_COLUMNS:
            v = c.get(k)
            row.append(";".join(map(str, v)) if isinstance(v, list) else ("" if v is None else v))
        w.writerow(row)
    return buf.getvalue()


def results_json(results: dict) -> str:
    return json.dumps(results, indent=2)


def benchmark_csv(bench: dict) -> str:
    cols = ["size_bp", "engine", "n_guides", "repeats", "mean_s", "min_s", "max_s", "std_s", "total_s",
            "kernel_mean_s", "throughput_mbp_s", "windows_per_s", "speedup", "peak_host_mem_mb",
            "gpu_mem_mb", "process_cpu_percent", "gpu_util_percent", "validated"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in bench["rows"]:
        w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])
    return buf.getvalue()


def benchmark_json(bench: dict) -> str:
    return json.dumps(bench, indent=2)


def _mismatch_chart(results) -> Drawing:
    counts = Counter(c["mismatches"] for c in results["candidates"])
    keys = sorted(counts) or [0]
    d = Drawing(400, 160)
    ch = VerticalBarChart()
    ch.x, ch.y, ch.width, ch.height = 40, 25, 330, 110
    ch.data = [[counts.get(k, 0) for k in keys]]
    ch.categoryAxis.categoryNames = [str(k) for k in keys]
    ch.valueAxis.valueMin = 0
    ch.valueAxis.valueMax = max(max(ch.data[0]), 1) + 1
    ch.valueAxis.valueStep = max(1, (ch.valueAxis.valueMax // 5) or 1)
    ch.bars[0].fillColor = colors.HexColor("#0F766E")
    d.add(ch)
    return d


def analysis_pdf(results: dict, bench: dict | None = None, max_rows: int = 40, max_alignments: int = 8) -> bytes:
    """Build the PDF report; returns bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm,
                            title=f"GeneWeaver report - {results['name']}")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=colors.HexColor("#0F766E"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#134E4A"))
    small = ParagraphStyle("small", parent=ss["BodyText"], fontSize=8, leading=10)
    story = [Paragraph("GeneWeaver Analysis Report", h1),
             Paragraph(config.APP_SUBTITLE, ss["Heading3"]),
             Paragraph(f"<b>Analysis:</b> {escape(results['name'])} &nbsp; <b>ID:</b> {results['id']} "
                       f"&nbsp; <b>Date:</b> {results['timestamp']}", ss["BodyText"]), Spacer(1, 6)]

    def table(data, widths=None, font=8):
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), font),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0FDFA")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t

    s, e, cfg, tm = results["summary"], results["engine"], results["config"], results["timings"]
    story += [Paragraph("1. Summary", h2), table([
        ["Metric", "Value"],
        ["Reference sequences / total bp", f"{s['n_references']} / {s['total_bp']:,}"],
        ["Guide sequences", s["n_guides"]],
        ["Perfect-match targets (PAM-compatible)", s["perfect_targets"]],
        ["Off-target candidates (PAM-compatible)", s["off_targets"]],
        ["Average alignment score", s["avg_alignment_score"]],
        ["CRISPR system / PAM", f"{cfg['system']} / {cfg['pam']} ({cfg['pam_side']})"],
        ["Max mismatches / PAM required", f"{cfg['max_mismatches']} / {cfg['require_pam']}"],
        ["Engine used", f"{e['used'].upper()}" + (f"  ({e['device']})" if e["used"] == "gpu" else "  (CPU-only mode)")],
        ["Scan time / total time", f"{tm['scan_s']} s / {tm['total_s']} s"],
        ["Windows scanned", f"{tm['windows_scanned']:,}"]], [80 * mm, 95 * mm], 9)]
    if e["fallback_reason"]:
        story.append(Paragraph(f"Engine note: {escape(e['fallback_reason'])}", small))

    story += [Spacer(1, 8), Paragraph("2. Candidate distribution by mismatch count", h2), _mismatch_chart(results)]

    story += [Paragraph("3. Ranked candidates", h2)]
    rows = [["#", "Guide", "Ref", "Str", "Start-End", "PAM", "MM", "Seed", "Sim.", "Aln"]]
    for c in results["candidates"][:max_rows]:
        rows.append([c["rank"], c["guide_id"], c["reference"], c["strand"], f"{c['start']}-{c['end']}",
                     c["pam_seq"], c["mismatches"], c["seed_mismatches"], c["similarity_score"],
                     c["alignment_score"]])
    if len(rows) == 1:
        story.append(Paragraph("No candidate sites were found with the chosen settings.", ss["BodyText"]))
    else:
        story.append(table(rows, [8*mm, 28*mm, 33*mm, 8*mm, 28*mm, 12*mm, 8*mm, 10*mm, 12*mm, 10*mm], 7))
        if len(results["candidates"]) > max_rows:
            story.append(Paragraph(f"Showing {max_rows} of {len(results['candidates'])} candidates; "
                                   "see the CSV for the complete list.", small))

    story += [Paragraph("4. Detailed alignments (top candidates)", h2)]
    shown = 0
    for c in results["candidates"]:
        if not c.get("alignment") or shown >= max_alignments:
            continue
        from backend.alignment_engine import Alignment
        aln = Alignment(**c["alignment"])
        story.append(Paragraph(f"<b>#{c['rank']}</b> {escape(c['guide_id'])} on {escape(c['reference'])} "
                               f"({c['strand']}) {c['start']}-{c['end']} PAM {c['pam_seq']} - "
                               f"{c['mismatches']} mismatch(es) at {c['mismatch_positions'] or '-'}; "
                               f"identity {aln.identity}%", small))
        story.append(Preformatted(alignment_to_text(aln), ParagraphStyle("mono", fontName="Courier", fontSize=7.5, leading=9)))
        shown += 1

    if bench:
        story += [PageBreak(), Paragraph("5. CPU vs GPU benchmark", h2),
                  Paragraph(escape(bench["method"]), small), Spacer(1, 4)]
        rows = [["Size (bp)", "Engine", "Mean (s)", "Min (s)", "Max (s)", "Mbp/s", "Speedup", "Valid"]]
        for r in bench["rows"]:
            rows.append([f"{r['size_bp']:,}", r["engine"].upper(), f"{r['mean_s']:.4f}", f"{r['min_s']:.4f}",
                         f"{r['max_s']:.4f}", f"{r['throughput_mbp_s']:.1f}",
                         f"{r['speedup']:.2f}x" if r.get("speedup") else "n/a", "yes" if r["validated"] else "NO"])
        story.append(table(rows, None, 8))
        for n in bench.get("notes", []):
            story.append(Paragraph(escape(n), small))

    story += [Spacer(1, 12), Paragraph("Scientific disclaimer", h2), Paragraph(config.DISCLAIMER, ss["BodyText"])]
    doc.build(story)
    return buf.getvalue()
