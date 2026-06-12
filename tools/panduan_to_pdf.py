#!/usr/bin/env python3
"""
panduan_to_pdf.py — render PANDUAN_ANOTASI_MANUAL.md (atau MD lain) ke PDF rapi
berbasis tabel, untuk dibaca/anotasi di tablet (mis. iPad).

Pakai:
    .venv/bin/python tools/panduan_to_pdf.py docs/PANDUAN_ANOTASI_MANUAL.md docs/PANDUAN_ANOTASI_MANUAL.pdf

Kata kekuatan KUAT/SEDANG/LEMAH otomatis diberi warna (hijau/amber/merah).
Mendukung subset Markdown yang dipakai dokumen: heading #/##/###, tabel GFM,
blockquote >, bullet -, fenced code ```, serta inline **bold** *italic* `code`.
"""
import os
import re
import sys

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Preformatted, KeepTogether)

# ── Palet ───────────────────────────────────────────────────────────────────
C_KUAT   = colors.HexColor("#15803d")   # hijau
C_SEDANG = colors.HexColor("#b45309")   # amber
C_LEMAH  = colors.HexColor("#b91c1c")   # merah
C_HEAD   = colors.HexColor("#1f2937")   # header tabel
C_HEADTX = colors.white
C_ROW    = colors.HexColor("#f3f4f6")   # baris selang-seling
C_GRID   = colors.HexColor("#d1d5db")
C_QUOTE  = colors.HexColor("#fef3c7")   # latar blockquote
C_QBAR   = colors.HexColor("#f59e0b")
C_CODE   = colors.HexColor("#0f172a")
C_CODEBG = colors.HexColor("#f1f5f9")
C_TITLE  = colors.HexColor("#7c2d12")

AVAIL_W = A4[0] - 32 * mm   # lebar konten (margin kiri+kanan 16mm)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(s: str) -> str:
    """Markdown inline → markup reportlab (<b>,<i>,<font>). Warnai kata kekuatan."""
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
    # warna kata kekuatan (token utuh, boleh diikuti * atau panah)
    def warna(m):
        kata = m.group(0)
        c = {"KUAT": C_KUAT, "SEDANG": C_SEDANG, "LEMAH": C_LEMAH}[m.group(1)]
        return f'<font color="#{c.hexval()[2:]}"><b>{kata}</b></font>'
    s = re.sub(r"\b(KUAT|SEDANG|LEMAH)\b\*?", warna, s)
    return s


def plain_len(s: str) -> int:
    return len(re.sub(r"[*`#>|]", "", s))


def make_styles():
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica",
                          fontSize=9.3, leading=13, spaceAfter=4)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                        fontSize=16, leading=20, textColor=C_TITLE, spaceBefore=10, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                        fontSize=13, leading=17, textColor=colors.HexColor("#1e3a8a"),
                        spaceBefore=12, spaceAfter=4)
    h3 = ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold",
                        fontSize=11, leading=14, textColor=colors.HexColor("#374151"),
                        spaceBefore=8, spaceAfter=3)
    cell = ParagraphStyle("cell", parent=body, fontSize=8.6, leading=11, spaceAfter=0)
    cellh = ParagraphStyle("cellh", parent=cell, textColor=C_HEADTX, fontName="Helvetica-Bold")
    quote = ParagraphStyle("quote", parent=body, fontSize=8.8, leading=12,
                           leftIndent=8, textColor=colors.HexColor("#3f2d00"))
    return dict(body=body, h1=h1, h2=h2, h3=h3, cell=cell, cellh=cellh, quote=quote)


def build_table(rows, st):
    header, *bodyrows = rows
    ncol = len(header)
    # bobot lebar kolom dari panjang konten (clamp biar 1 kolom tak mendominasi)
    weights = []
    for c in range(ncol):
        mx = max([plain_len(header[c])] + [plain_len(r[c]) for r in bodyrows if c < len(r)] + [4])
        weights.append(min(mx, 42))
    tot = sum(weights) or 1
    colw = [max(34, AVAIL_W * w / tot) for w in weights]
    # skala agar pas lebar
    scale = AVAIL_W / sum(colw)
    colw = [w * scale for w in colw]

    data = [[Paragraph(inline(c), st["cellh"]) for c in header]]
    for r in bodyrows:
        r = (r + [""] * ncol)[:ncol]
        data.append([Paragraph(inline(c), st["cell"]) for c in r])

    t = Table(data, colWidths=colw, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), C_HEAD),
        ("GRID", (0, 0), (-1, -1), 0.5, C_GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), C_ROW))
    t.setStyle(TableStyle(style))
    return t


def parse(md_path, st):
    with open(md_path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    flow = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.rstrip()

        if not s.strip():
            i += 1
            continue

        if s.strip() == "---":          # pemisah horizontal → sedikit ruang
            flow.append(Spacer(1, 4))
            i += 1
            continue

        if s.startswith("```"):         # blok kode
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            pre = Preformatted("\n".join(buf), ParagraphStyle(
                "code", fontName="Courier", fontSize=8, leading=10, textColor=C_CODE))
            box = Table([[pre]], colWidths=[AVAIL_W])
            box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C_CODEBG),
                                     ("BOX", (0, 0), (-1, -1), 0.5, C_GRID),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                     ("TOPPADDING", (0, 0), (-1, -1), 6),
                                     ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            flow.append(box)
            continue

        if s.lstrip().startswith("|") and "|" in s:   # tabel
            tbl = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                row = lines[i].strip().strip("|")
                cells = [c.strip() for c in row.split("|")]
                if not re.match(r"^[\s:\-]+$", "".join(cells)):   # lewati baris ---
                    tbl.append(cells)
                i += 1
            if tbl:
                flow.append(build_table(tbl, st))
                flow.append(Spacer(1, 6))
            continue

        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            lvl = len(m.group(1))
            flow.append(Paragraph(inline(m.group(2)), st[f"h{lvl}"]))
            i += 1
            continue

        if s.lstrip().startswith(">"):   # blockquote (gabung baris berurutan)
            buf = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            para = Paragraph(inline(" ".join(b for b in buf if b.strip())), st["quote"])
            box = Table([[para]], colWidths=[AVAIL_W])
            box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C_QUOTE),
                                     ("LINEBEFORE", (0, 0), (0, -1), 3, C_QBAR),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                     ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                                     ("TOPPADDING", (0, 0), (-1, -1), 4),
                                     ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
            flow.append(box)
            flow.append(Spacer(1, 3))
            continue

        if re.match(r"^\s*[-*]\s+", s):   # bullet (gabung berurutan)
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                txt = re.sub(r"^\s*[-*]\s+", "", lines[i])
                flow.append(Paragraph("•&nbsp;&nbsp;" + inline(txt),
                                      ParagraphStyle("li", parent=st["body"],
                                                     leftIndent=10, spaceAfter=2)))
                i += 1
            continue

        flow.append(Paragraph(inline(s), st["body"]))   # paragraf biasa
        i += 1
    return flow


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#9ca3af"))
    canvas.drawCentredString(A4[0] / 2, 10 * mm,
                             f"Panduan Anotasi Manual  ·  hal. {doc.page}")
    canvas.restoreState()


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "docs/PANDUAN_ANOTASI_MANUAL.md"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".pdf"
    st = make_styles()
    doc = SimpleDocTemplate(out, pagesize=A4,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=16 * mm,
                            title="Panduan Anotasi Manual")
    doc.build(parse(src, st), onFirstPage=footer, onLaterPages=footer)
    print(f"PDF dibuat: {out}  ({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
