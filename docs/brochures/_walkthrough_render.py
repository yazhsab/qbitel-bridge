"""
Shared renderer for QBITEL BPO walkthrough PDFs.

Reads a markdown file and emits a styled PDF using ReportLab with the same
brand palette and page furniture as build_bpo_pdf.py. The two thin wrappers
(build_bpo_product_owner_pdf.py, build_bpo_sales_manager_pdf.py) supply
cover-page text and output paths.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

# ─── Brand palette (kept identical to build_bpo_pdf.py) ──────────────────────
NAVY = HexColor('#0D1B3E')
TEAL = HexColor('#008B9A')
TEAL_DARK = HexColor('#006B7A')
GOLD = HexColor('#F0A500')
LIGHT_BG = HexColor('#F4F7FA')
MID_GREY = HexColor('#5A6A7A')
DARK_TEXT = HexColor('#1A1A2E')
TABLE_ALT = HexColor('#EAF3F8')
WHITE_C = HexColor('#FFFFFF')
LIGHT_NAVY = HexColor('#1A2D5A')

PAGE_W, PAGE_H = letter
MARGIN = 0.85 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


# ─── Configurable cover ──────────────────────────────────────────────────────

@dataclass
class CoverSpec:
    """Per-document cover-page configuration."""
    title_line_1: str          # "QBITEL"
    title_line_2: str          # "BRIDGE"
    subtitle: str              # "PRODUCT OWNER WALKTHROUGH"
    tagline: str               # one-line italic tagline
    metric_boxes: Sequence[tuple]  # 3 tuples of (big, small) for the upper row
    metric_boxes_2: Sequence[tuple]  # 3 tuples for the lower row
    footer_pillars: str        # bottom-left bold line
    version_line: str          # "Version 1.0 | February 2026 ..."
    header_label: str          # right-of-logo header label


# ─── Custom flowables ────────────────────────────────────────────────────────

class SectionHeader(Flowable):
    """Navy section header with gold + teal accents — matches build_bpo_pdf.py."""

    def __init__(self, title, subtitle=None, width=None):
        super().__init__()
        self.title = title
        self.subtitle = subtitle
        self.w = width or CONTENT_W
        self.h = 52 if subtitle else 38

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(NAVY)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        c.setFillColor(GOLD)
        c.rect(0, 0, 6, self.h, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(self.w - 4, 0, 4, self.h, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 13)
        title_y = self.h - 22 if self.subtitle else (self.h - 16) / 2 + 4
        c.drawString(16, title_y, self.title.upper())
        if self.subtitle:
            c.setFillColor(TEAL)
            c.setFont('Helvetica-Oblique', 9)
            c.drawString(16, 8, self.subtitle)


class CalloutBox(Flowable):
    """Quote/callout box (used for blockquotes in the markdown)."""

    def __init__(self, text_lines, width=None, bg=LIGHT_BG):
        super().__init__()
        self.text_lines = text_lines
        self.w = width or CONTENT_W
        self.bg = bg
        self.h = len(text_lines) * 14 + 24

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(self.bg)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(0, 0, 5, self.h, fill=1, stroke=0)
        c.setFillColor(GOLD)
        c.setFont('Helvetica-Bold', 14)
        c.drawString(14, self.h - 20, '◈')
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Oblique', 9.5)
        y = self.h - 18
        for line in self.text_lines:
            c.drawString(30, y, line)
            y -= 14


# ─── Page furniture ──────────────────────────────────────────────────────────

def make_draw_page(header_label: str):
    """Build a page-decoration callback bound to a per-doc header label."""

    def draw_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - 0.45 * inch, PAGE_W, 0.45 * inch, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.rect(0, PAGE_H - 0.45 * inch - 3, PAGE_W, 3, fill=1, stroke=0)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica-Bold', 10)
        canvas.drawString(MARGIN, PAGE_H - 0.32 * inch, 'QBITEL BRIDGE')
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(TEAL)
        canvas.drawString(MARGIN + 1.1 * inch, PAGE_H - 0.32 * inch, header_label)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 8)
        page_str = f'Page {doc.page}'
        pw = canvas.stringWidth(page_str, 'Helvetica', 8)
        canvas.drawString(PAGE_W - MARGIN - pw, PAGE_H - 0.32 * inch, page_str)
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, PAGE_W, 0.4 * inch, fill=1, stroke=0)
        canvas.setFillColor(TEAL)
        canvas.rect(0, 0.4 * inch, PAGE_W, 2, fill=1, stroke=0)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 7.5)
        canvas.drawString(
            MARGIN,
            0.15 * inch,
            'Confidential — For Authorized Recipients Only  |  © 2026 QBITEL. All Rights Reserved.',
        )
        contact_str = 'enterprise@qbitel.com  |  bridge.qbitel.com'
        cw = canvas.stringWidth(contact_str, 'Helvetica', 7.5)
        canvas.drawString(PAGE_W - MARGIN - cw, 0.15 * inch, contact_str)
        canvas.restoreState()

    return draw_page


def make_draw_cover(spec: CoverSpec):
    """Build a cover-page decoration callback bound to a per-doc spec."""

    def draw_cover(canvas, doc):
        canvas.saveState()
        w, h = PAGE_W, PAGE_H

        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, w, h, fill=1, stroke=0)

        # Gold + teal corner accents
        canvas.setFillColor(GOLD)
        p = canvas.beginPath()
        p.moveTo(w * 0.55, h)
        p.lineTo(w, h)
        p.lineTo(w, h * 0.65)
        p.close()
        canvas.drawPath(p, fill=1, stroke=0)
        canvas.setFillColor(TEAL)
        p2 = canvas.beginPath()
        p2.moveTo(w * 0.72, h)
        p2.lineTo(w, h)
        p2.lineTo(w, h * 0.8)
        p2.close()
        canvas.drawPath(p2, fill=1, stroke=0)

        # Bottom strip
        canvas.setFillColor(TEAL_DARK)
        canvas.rect(0, 0, w, 1.6 * inch, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.rect(0, 1.6 * inch, w, 5, fill=1, stroke=0)

        # Titles
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica-Bold', 52)
        canvas.drawString(MARGIN, h * 0.68, spec.title_line_1)
        canvas.setFillColor(TEAL)
        canvas.drawString(MARGIN, h * 0.68 - 58, spec.title_line_2)
        canvas.setFillColor(GOLD)
        canvas.rect(MARGIN, h * 0.68 - 68, 3.4 * inch, 5, fill=1, stroke=0)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica-Bold', 17)
        canvas.drawString(MARGIN, h * 0.68 - 100, spec.subtitle)
        canvas.setFillColor(HexColor('#CCDDE8'))
        canvas.setFont('Helvetica-Oblique', 13)
        canvas.drawString(MARGIN, h * 0.68 - 128, spec.tagline)

        # Upper metric row
        box_w = (CONTENT_W - 2 * 0.15 * inch) / 3
        bx_start = MARGIN
        by = h * 0.38
        bh = 1.0 * inch
        for i, (big, small) in enumerate(spec.metric_boxes):
            bx = bx_start + i * (box_w + 0.15 * inch)
            bg_c = TEAL if i == 1 else LIGHT_NAVY
            canvas.setFillColor(bg_c)
            canvas.roundRect(bx, by, box_w, bh, 6, fill=1, stroke=0)
            canvas.setFillColor(GOLD)
            canvas.rect(bx, by + bh - 5, box_w, 5, fill=1, stroke=0)
            canvas.setFillColor(GOLD if i == 1 else WHITE_C)
            canvas.setFont('Helvetica-Bold', 20)
            tw = canvas.stringWidth(big, 'Helvetica-Bold', 20)
            canvas.drawString(bx + (box_w - tw) / 2, by + bh - 30, big)
            canvas.setFillColor(WHITE_C)
            canvas.setFont('Helvetica', 8)
            for j, line in enumerate(small.split('\n')):
                lw = canvas.stringWidth(line, 'Helvetica', 8)
                canvas.drawString(bx + (box_w - lw) / 2, by + bh - 46 - j * 12, line)

        # Lower metric row
        by2 = h * 0.28
        for i, (big, small) in enumerate(spec.metric_boxes_2):
            bx = bx_start + i * (box_w + 0.15 * inch)
            canvas.setFillColor(LIGHT_BG)
            canvas.roundRect(bx, by2, box_w, 0.85 * inch, 6, fill=1, stroke=0)
            canvas.setFillColor(NAVY)
            canvas.setFont('Helvetica-Bold', 18)
            tw = canvas.stringWidth(big, 'Helvetica-Bold', 18)
            canvas.drawString(bx + (box_w - tw) / 2, by2 + 0.55 * inch, big)
            canvas.setFillColor(MID_GREY)
            canvas.setFont('Helvetica', 7.5)
            for j, line in enumerate(small.split('\n')):
                lw = canvas.stringWidth(line, 'Helvetica', 7.5)
                canvas.drawString(bx + (box_w - lw) / 2, by2 + 0.35 * inch - j * 11, line)

        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica-Bold', 10)
        canvas.drawString(MARGIN, 0.85 * inch, spec.footer_pillars)
        canvas.setFont('Helvetica', 8.5)
        canvas.setFillColor(HexColor('#AABBCC'))
        canvas.drawString(MARGIN, 0.55 * inch, spec.version_line)

        canvas.restoreState()

    return draw_cover


# ─── Paragraph styles ────────────────────────────────────────────────────────

def get_styles():
    return {
        'body': ParagraphStyle(
            'body', fontName='Helvetica', fontSize=10, leading=14,
            textColor=DARK_TEXT, spaceAfter=6, spaceBefore=0, alignment=TA_LEFT,
        ),
        'h2': ParagraphStyle(
            'h2', fontName='Helvetica-Bold', fontSize=14, leading=18,
            textColor=NAVY, spaceAfter=6, spaceBefore=10,
        ),
        'h3': ParagraphStyle(
            'h3', fontName='Helvetica-Bold', fontSize=11.5, leading=15,
            textColor=NAVY, spaceAfter=4, spaceBefore=10,
        ),
        'h4': ParagraphStyle(
            'h4', fontName='Helvetica-Bold', fontSize=10.5, leading=13,
            textColor=TEAL_DARK, spaceAfter=3, spaceBefore=8,
        ),
        'bullet': ParagraphStyle(
            'bullet', fontName='Helvetica', fontSize=9.5, leading=13,
            textColor=DARK_TEXT, spaceAfter=2, leftIndent=14, bulletIndent=2,
        ),
        'table_header': ParagraphStyle(
            'th', fontName='Helvetica-Bold', fontSize=9,
            textColor=WHITE_C, leading=12,
        ),
        'table_cell': ParagraphStyle(
            'tc', fontName='Helvetica', fontSize=9,
            textColor=DARK_TEXT, leading=12,
        ),
        'table_cell_bold': ParagraphStyle(
            'tcb', fontName='Helvetica-Bold', fontSize=9,
            textColor=NAVY, leading=12,
        ),
        'caption': ParagraphStyle(
            'caption', fontName='Helvetica-Oblique', fontSize=8,
            textColor=MID_GREY, alignment=TA_CENTER, spaceAfter=4,
        ),
        'code': ParagraphStyle(
            'code', fontName='Courier', fontSize=7.5, leading=9.5,
            textColor=DARK_TEXT, backColor=LIGHT_BG,
            leftIndent=8, rightIndent=8,
            spaceAfter=6, spaceBefore=4,
            borderPadding=6,
        ),
    }


# ─── Inline markdown → ReportLab markup ──────────────────────────────────────

_BOLD = re.compile(r'\*\*([^*]+)\*\*')
_ITAL = re.compile(r'(?<!\*)\*([^*]+)\*(?!\*)')
_INLINE_CODE = re.compile(r'`([^`]+)`')
_LINK = re.compile(r'\[([^\]]+)\]\([^)]+\)')


def _inline(text: str) -> str:
    # Escape ampersands and angle brackets for ReportLab markup
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = _LINK.sub(r'\1', text)                       # drop URL, keep label
    text = _BOLD.sub(r'<b>\1</b>', text)
    text = _ITAL.sub(r'<i>\1</i>', text)
    text = _INLINE_CODE.sub(r'<font name="Courier">\1</font>', text)
    return text


# ─── Markdown → flowables ────────────────────────────────────────────────────

def parse_markdown(md: str, S) -> List:
    """Parse a markdown string into a list of ReportLab flowables."""
    flows: List = []
    lines = md.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.rstrip()

        # Skip leading H1 (the title — already on the cover)
        if stripped.startswith('# ') and not flows:
            i += 1
            continue

        # Skip horizontal rules — used as section dividers, the section headers
        # already provide visual separation.
        if stripped.strip() == '---':
            i += 1
            continue

        # Blank line
        if not stripped.strip():
            i += 1
            continue

        # Fenced code block (```) — render as preformatted monospace block.
        # Used for ASCII architecture / flow diagrams.
        if stripped.lstrip().startswith('```'):
            i += 1
            block_lines: list[str] = []
            while i < n and not lines[i].lstrip().startswith('```'):
                # Preserve the raw line; escape for markup safety.
                raw = lines[i].rstrip('\n')
                raw = (raw.replace('&', '&amp;')
                           .replace('<', '&lt;')
                           .replace('>', '&gt;')
                           .replace(' ', '&nbsp;'))
                block_lines.append(raw)
                i += 1
            if i < n:
                i += 1  # skip closing ```
            if block_lines:
                code_html = '<br/>'.join(block_lines)
                flows.append(Paragraph(code_html, S['code']))
            continue

        # Headings
        if stripped.startswith('## '):
            title = stripped[3:].strip()
            # Section break before major sections
            flows.append(Spacer(1, 6))
            flows.append(SectionHeader(title))
            flows.append(Spacer(1, 8))
            i += 1
            continue
        if stripped.startswith('### '):
            flows.append(Paragraph(_inline(stripped[4:].strip()), S['h3']))
            i += 1
            continue
        if stripped.startswith('#### '):
            flows.append(Paragraph(_inline(stripped[5:].strip()), S['h4']))
            i += 1
            continue

        # Blockquote → callout box (collect consecutive '> ' lines)
        if stripped.startswith('>'):
            quote_lines = []
            while i < n and lines[i].lstrip().startswith('>'):
                ln = lines[i].lstrip()[1:].lstrip()
                quote_lines.append(_inline(ln))
                i += 1
            # CalloutBox renders raw text — strip markup for safety
            plain = [re.sub(r'<[^>]+>', '', x) for x in quote_lines]
            # Wrap long lines for the fixed-width callout
            wrapped: list[str] = []
            for p in plain:
                while len(p) > 92:
                    cut = p.rfind(' ', 0, 92)
                    if cut == -1:
                        cut = 92
                    wrapped.append(p[:cut])
                    p = p[cut + 1:]
                if p:
                    wrapped.append(p)
            if wrapped:
                flows.append(CalloutBox(wrapped))
                flows.append(Spacer(1, 6))
            continue

        # Table (markdown pipe table)
        if stripped.startswith('|') and i + 1 < n and re.match(r'^\|[\s\-:|]+\|\s*$', lines[i + 1]):
            header_cells = [c.strip() for c in stripped.strip('|').split('|')]
            i += 2  # skip header + separator
            rows = []
            while i < n and lines[i].lstrip().startswith('|'):
                row_cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                rows.append(row_cells)
                i += 1
            flows.append(_make_table(header_cells, rows, S))
            flows.append(Spacer(1, 6))
            continue

        # Bullet list
        if re.match(r'^\s*[-*]\s', stripped):
            while i < n and re.match(r'^\s*[-*]\s', lines[i]):
                raw = re.sub(r'^\s*[-*]\s', '', lines[i]).strip()
                # Strip "[ ] " or "[x] " checkbox marker if present
                raw = re.sub(r'^\[[ xX]\]\s*', '', raw)
                flows.append(Paragraph('• ' + _inline(raw), S['bullet']))
                i += 1
            flows.append(Spacer(1, 3))
            continue

        # Ordered list
        if re.match(r'^\s*\d+\.\s', stripped):
            while i < n and re.match(r'^\s*\d+\.\s', lines[i]):
                raw = re.sub(r'^\s*\d+\.\s', '', lines[i]).strip()
                flows.append(Paragraph('▸ ' + _inline(raw), S['bullet']))
                i += 1
            flows.append(Spacer(1, 3))
            continue

        # Default: paragraph (collect lines until blank / structural)
        para_lines = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].rstrip()
            if (not nxt.strip()
                    or nxt.startswith('#')
                    or nxt.startswith('|')
                    or nxt.startswith('>')
                    or nxt.strip() == '---'
                    or re.match(r'^\s*[-*]\s', nxt)
                    or re.match(r'^\s*\d+\.\s', nxt)):
                break
            para_lines.append(nxt)
            i += 1
        flows.append(Paragraph(_inline(' '.join(para_lines)), S['body']))

    return flows


# ─── Table builder ───────────────────────────────────────────────────────────

def _make_table(headers, rows, S):
    n_cols = len(headers)
    # Even column widths by default; ReportLab will wrap on Paragraph cells
    col_w = [CONTENT_W / n_cols] * n_cols

    data = [[Paragraph(_inline(h), S['table_header']) for h in headers]]
    for row in rows:
        # Pad/truncate to header width to be defensive
        cells = list(row) + [''] * (n_cols - len(row))
        cells = cells[:n_cols]
        styled = []
        for ci, cell in enumerate(cells):
            style = S['table_cell_bold'] if ci == 0 else S['table_cell']
            styled.append(Paragraph(_inline(cell), style))
        data.append(styled)

    n_rows = len(data)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE_C),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        *[
            ('ROWBACKGROUND', (0, ri), (-1, ri), TABLE_ALT if ri % 2 == 1 else WHITE_C)
            for ri in range(1, n_rows)
        ],
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#C8D8E8')),
        ('LINEBELOW', (0, -1), (-1, -1), 1, TEAL),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ])
    tbl = Table(data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(style)
    return tbl


# ─── Public build entry point ────────────────────────────────────────────────

def build_walkthrough_pdf(*, markdown_path: str, output_path: str, cover: CoverSpec) -> str:
    """Build the walkthrough PDF from a markdown file + cover spec."""
    from reportlab.platypus import NextPageTemplate

    with open(markdown_path, 'r', encoding='utf-8') as f:
        md = f.read()

    doc = BaseDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.65 * inch, bottomMargin=0.6 * inch,
    )
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H, id='cover')
    inner_frame = Frame(
        MARGIN, 0.6 * inch,
        PAGE_W - 2 * MARGIN, PAGE_H - 0.65 * inch - 0.6 * inch,
        id='inner',
    )
    cover_tmpl = PageTemplate(id='Cover', frames=[cover_frame], onPage=make_draw_cover(cover))
    inner_tmpl = PageTemplate(
        id='Inner', frames=[inner_frame], onPage=make_draw_page(cover.header_label),
    )
    doc.addPageTemplates([cover_tmpl, inner_tmpl])

    S = get_styles()
    story: list = [NextPageTemplate('Inner'), PageBreak()]
    story.extend(parse_markdown(md, S))

    doc.build(story)
    return output_path
