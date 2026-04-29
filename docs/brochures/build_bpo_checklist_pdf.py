"""
Build QBITEL Bridge BPO Deployment & Delivery Checklist - Professional PDF
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, NextPageTemplate

# ─── Brand Colors ─────────────────────────────────────────────────────────────
NAVY       = HexColor('#0D1B3E')
TEAL       = HexColor('#008B9A')
TEAL_DARK  = HexColor('#006B7A')
TEAL_LIGHT = HexColor('#E0F4F6')
GOLD       = HexColor('#F0A500')
GOLD_LIGHT = HexColor('#FEF6E0')
LIGHT_BG   = HexColor('#F4F7FA')
MID_GREY   = HexColor('#5A6A7A')
DARK_TEXT  = HexColor('#1A1A2E')
TABLE_ALT  = HexColor('#EAF3F8')
WHITE_C    = HexColor('#FFFFFF')
LIGHT_NAVY = HexColor('#1A2D5A')
GREEN      = HexColor('#2E8B57')
GREEN_LIGHT= HexColor('#F0FFF4')
RED_SOFT   = HexColor('#8B1A1A')
ORANGE     = HexColor('#E07B00')

PAGE_W, PAGE_H = letter
MARGIN = 0.8 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


# ─── Custom Flowables ─────────────────────────────────────────────────────────

class PhaseHeader(Flowable):
    """Colored phase header with phase number and title."""
    def __init__(self, phase_num, title, duration, owner, width=None):
        super().__init__()
        self.phase_num = phase_num
        self.title = title
        self.duration = duration
        self.owner = owner
        self.w = width or CONTENT_W
        self.h = 50

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Background
        c.setFillColor(NAVY)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        # Gold left badge
        c.setFillColor(GOLD)
        c.rect(0, 0, 58, self.h, fill=1, stroke=0)
        # Teal right accent
        c.setFillColor(TEAL)
        c.rect(self.w - 5, 0, 5, self.h, fill=1, stroke=0)
        # Phase label
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 7)
        c.drawString(6, self.h - 14, 'PHASE')
        c.setFont('Helvetica-Bold', 22)
        pw = c.stringWidth(self.phase_num, 'Helvetica-Bold', 22)
        c.drawString(29 - pw / 2, self.h - 38, self.phase_num)
        # Title
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 13)
        c.drawString(68, self.h - 20, self.title.upper())
        # Duration and owner badges
        c.setFillColor(TEAL)
        dur_label = f'⏱  {self.duration}'
        dw = c.stringWidth(dur_label, 'Helvetica', 8) + 14
        c.roundRect(68, 8, dw, 16, 3, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica', 8)
        c.drawString(75, 13, dur_label)
        # Owner badge
        c.setFillColor(LIGHT_NAVY)
        ow = c.stringWidth(f'Owner: {self.owner}', 'Helvetica', 8) + 14
        c.roundRect(68 + dw + 8, 8, ow, 16, 3, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica', 8)
        c.drawString(68 + dw + 15, 13, f'Owner: {self.owner}')


class ChecklistSection(Flowable):
    """A checklist section header."""
    def __init__(self, number, title, width=None):
        super().__init__()
        self.number = number
        self.title = title
        self.w = width or CONTENT_W
        self.h = 28

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(TEAL_LIGHT)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(0, 0, 4, self.h, fill=1, stroke=0)
        c.setFillColor(TEAL_DARK)
        c.setFont('Helvetica-Bold', 10)
        c.drawString(12, self.h - 18, f'{self.number}  —  {self.title}')


class CheckItem(Flowable):
    """A single checklist item with checkbox."""
    def __init__(self, text, sub_items=None, indent=0, highlight=None, width=None):
        super().__init__()
        self.text = text
        self.sub_items = sub_items or []
        self.indent = indent
        self.highlight = highlight  # 'critical', 'warning', None
        self.w = width or CONTENT_W
        self.line_h = 14
        # Wrap text
        self._lines = self._wrap(text, self.w - 40 - indent)
        sub_lines = sum(len(self._wrap(s, self.w - 60 - indent)) for s in self.sub_items)
        self.h = len(self._lines) * self.line_h + sub_lines * 13 + 6

    def _wrap(self, text, max_w):
        import reportlab.pdfgen.canvas as _c
        # Approximate: 6 chars per em at 9pt
        chars_per_line = int(max_w / 5.5)
        words = text.split()
        lines, current = [], ''
        for word in words:
            test = (current + ' ' + word).strip()
            if len(test) <= chars_per_line:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [text]

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        y = self.h - self.line_h
        x_start = 10 + self.indent

        # Highlight background
        if self.highlight == 'critical':
            c.setFillColor(HexColor('#FFF8E8'))
            c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
            c.setFillColor(GOLD)
            c.rect(0, 0, 3, self.h, fill=1, stroke=0)
        elif self.highlight == 'warning':
            c.setFillColor(HexColor('#F8F0FF'))
            c.rect(0, 0, self.w, self.h, fill=1, stroke=0)

        # Checkbox
        c.setStrokeColor(TEAL)
        c.setLineWidth(1.2)
        c.roundRect(x_start, y - 2, 10, 10, 1.5, fill=0, stroke=1)

        # Text
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 9)
        for i, line in enumerate(self._lines):
            c.drawString(x_start + 15, y - i * self.line_h, line)
        y -= len(self._lines) * self.line_h

        # Sub-items
        for sub in self.sub_items:
            sub_lines = self._wrap(sub, self.w - 70 - self.indent)
            c.setFillColor(TEAL)
            c.circle(x_start + 22, y + 3, 2, fill=1, stroke=0)
            c.setFillColor(MID_GREY)
            c.setFont('Helvetica', 8.5)
            for j, sl in enumerate(sub_lines):
                c.drawString(x_start + 28, y - j * 13, sl)
            y -= len(sub_lines) * 13


class TableCheckItem(Flowable):
    """A table-style checklist for structured data."""
    def __init__(self, headers, rows, col_widths, width=None):
        super().__init__()
        self.headers = headers
        self.rows = rows
        self.col_widths = col_widths
        self.w = width or CONTENT_W

    def wrap(self, avw, avh):
        return self.w, (len(self.rows) + 1) * 20 + 2

    def draw(self):
        c = self.canv
        n = len(self.col_widths)
        row_h = 18
        total_h = (len(self.rows) + 1) * row_h
        y = total_h

        # Header
        x = 0
        for i, (h, w) in enumerate(zip(self.headers, self.col_widths)):
            c.setFillColor(NAVY)
            c.rect(x, y - row_h, w * inch, row_h, fill=1, stroke=0)
            c.setFillColor(WHITE_C)
            c.setFont('Helvetica-Bold', 8)
            c.drawString(x + 5, y - row_h + 5, h)
            x += w * inch

        y -= row_h
        for ri, row in enumerate(self.rows):
            x = 0
            bg = TABLE_ALT if ri % 2 == 0 else WHITE_C
            c.setFillColor(bg)
            c.rect(0, y - row_h, sum(w * inch for w in self.col_widths), row_h, fill=1, stroke=0)
            for ci, (cell, w) in enumerate(zip(row, self.col_widths)):
                c.setFillColor(NAVY if ci == 0 else DARK_TEXT)
                c.setFont('Helvetica-Bold' if ci == 0 else 'Helvetica', 8)
                c.drawString(x + 5, y - row_h + 5, str(cell))
                # Checkbox in last col if it's '☐'
                if str(cell) in ('☐', '□', '[ ]'):
                    c.setStrokeColor(TEAL)
                    c.setLineWidth(1)
                    c.roundRect(x + 5, y - row_h + 4, 9, 9, 1, fill=0, stroke=1)
                x += w * inch
            # Border
            c.setStrokeColor(HexColor('#C8D8E8'))
            c.setLineWidth(0.3)
            c.line(0, y - row_h, sum(w * inch for w in self.col_widths), y - row_h)
            y -= row_h


class SignOffBlock(Flowable):
    """Sign-off block with signature lines."""
    def __init__(self, signatories, width=None):
        super().__init__()
        self.signatories = signatories
        self.w = width or CONTENT_W
        self.h = len(signatories) * 36 + 16

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(LIGHT_BG)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        c.setFillColor(GOLD)
        c.rect(0, 0, 4, self.h, fill=1, stroke=0)

        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 9.5)
        c.drawString(12, self.h - 14, 'SIGN-OFF REQUIRED')

        y = self.h - 30
        half = self.w / 2
        for i, sig in enumerate(self.signatories):
            col = i % 2
            row = i // 2
            x = 12 + col * half
            base_y = y - row * 36
            c.setFillColor(NAVY)
            c.setFont('Helvetica-Bold', 8.5)
            c.drawString(x, base_y, sig)
            c.setStrokeColor(TEAL)
            c.setLineWidth(0.75)
            c.line(x, base_y - 14, x + half - 24, base_y - 14)
            c.setFillColor(MID_GREY)
            c.setFont('Helvetica', 7.5)
            c.drawString(x, base_y - 24, 'Signature ________________________   Date ___________')


class PerformanceTargetBlock(Flowable):
    """Performance targets table."""
    def __init__(self, rows, width=None):
        super().__init__()
        self.rows = rows
        self.w = width or CONTENT_W
        self.row_h = 18
        self.h = (len(rows) + 1) * self.row_h + 4

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        rh = self.row_h
        total_h = self.h
        # Header
        cols = [2.4 * inch, 1.6 * inch, 1.5 * inch, 0.9 * inch]
        headers = ['Metric', 'Target', 'Measured', 'Met']
        x = 0
        y = total_h
        for h, cw in zip(headers, cols):
            c.setFillColor(NAVY)
            c.rect(x, y - rh, cw, rh, fill=1, stroke=0)
            c.setFillColor(WHITE_C)
            c.setFont('Helvetica-Bold', 8)
            c.drawString(x + 5, y - rh + 5, h)
            x += cw
        y -= rh

        for ri, row in enumerate(self.rows):
            bg = TABLE_ALT if ri % 2 == 0 else WHITE_C
            c.setFillColor(bg)
            c.rect(0, y - rh, sum(cols), rh, fill=1, stroke=0)
            x = 0
            for ci, (cell, cw) in enumerate(zip(row, cols)):
                if ci == 3:  # Checkbox column
                    c.setStrokeColor(TEAL)
                    c.setLineWidth(1)
                    c.roundRect(x + 8, y - rh + 4, 10, 10, 1.5, fill=0, stroke=1)
                else:
                    c.setFillColor(NAVY if ci == 0 else (TEAL_DARK if ci == 1 else MID_GREY))
                    c.setFont('Helvetica-Bold' if ci == 0 else 'Helvetica', 8)
                    c.drawString(x + 5, y - rh + 5, str(cell))
                x += cw
            c.setStrokeColor(HexColor('#C8D8E8'))
            c.setLineWidth(0.3)
            c.line(0, y - rh, sum(cols), y - rh)
            y -= rh


# ─── Page Callbacks ───────────────────────────────────────────────────────────

def draw_cover(canvas, doc):
    canvas.saveState()
    w, h = PAGE_W, PAGE_H

    # Full navy
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)

    # Gold triangle accent
    canvas.setFillColor(GOLD)
    p = canvas.beginPath()
    p.moveTo(w * 0.45, h)
    p.lineTo(w, h)
    p.lineTo(w, h * 0.55)
    p.close()
    canvas.drawPath(p, fill=1, stroke=0)

    # Teal overlay triangle
    canvas.setFillColor(TEAL)
    p2 = canvas.beginPath()
    p2.moveTo(w * 0.65, h)
    p2.lineTo(w, h)
    p2.lineTo(w, h * 0.74)
    p2.close()
    canvas.drawPath(p2, fill=1, stroke=0)

    # Checklist icon (bottom right corner decoration)
    canvas.setFillColor(TEAL_DARK)
    canvas.rect(0, 0, w, 1.5 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, 1.5 * inch, w, 4, fill=1, stroke=0)

    # Left bold bar
    canvas.setFillColor(GOLD)
    canvas.rect(0, 0, 8, h, fill=1, stroke=0)

    # Document type chip
    canvas.setFillColor(TEAL)
    canvas.roundRect(MARGIN + 4, h * 0.84, 2.6 * inch, 22, 4, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(MARGIN + 16, h * 0.84 + 7, 'DEPLOYMENT & DELIVERY CHECKLIST')

    # Main title
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 46)
    canvas.drawString(MARGIN + 4, h * 0.73, 'QBITEL')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica-Bold', 46)
    canvas.drawString(MARGIN + 4, h * 0.73 - 54, 'BRIDGE')

    # Gold underline
    canvas.setFillColor(GOLD)
    canvas.rect(MARGIN + 4, h * 0.73 - 62, 3.4 * inch, 4, fill=1, stroke=0)

    # Subtitle
    canvas.setFillColor(HexColor('#CCDDE8'))
    canvas.setFont('Helvetica', 12)
    canvas.drawString(MARGIN + 4, h * 0.73 - 86, 'BPO & Call Center Industry — End-to-End Deployment Guide')

    # Phase overview boxes
    phases = [
        ('11', 'Deployment\nPhases'),
        ('4–6h', 'Core Activation\nWindow'),
        ('100%', 'Zero Downtime\nDeployment'),
        ('30d', 'Post Go-Live\nMonitoring'),
    ]
    bw = (CONTENT_W - 0.45 * inch) / 4
    by = h * 0.48
    bh = 0.82 * inch
    for i, (num, label) in enumerate(phases):
        bx = MARGIN + i * (bw + 0.15 * inch)
        bg = TEAL if i in (0, 2) else LIGHT_NAVY
        canvas.setFillColor(bg)
        canvas.roundRect(bx, by, bw, bh, 5, fill=1, stroke=0)
        canvas.setFillColor(GOLD if bg == LIGHT_NAVY else WHITE_C)
        canvas.setFont('Helvetica-Bold', 20)
        nw = canvas.stringWidth(num, 'Helvetica-Bold', 20)
        canvas.drawString(bx + (bw - nw) / 2, by + bh - 32, num)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 8)
        for j, ln in enumerate(label.split('\n')):
            lw = canvas.stringWidth(ln, 'Helvetica', 8)
            canvas.drawString(bx + (bw - lw) / 2, by + bh - 48 - j * 11, ln)

    # Phase list
    phase_list = [
        'Phase 0: Pre-Engagement & Scoping',
        'Phase 1: AI Protocol Discovery',
        'Phase 2: Infrastructure Readiness',
        'Phase 3: Security & Compliance Setup',
        'Phase 4: Protocol Protection Activation',
        'Phase 5: Integration Deployment',
        'Phase 6: Automation Recipe Execution',
        'Phase 7: Monitoring & Alerting Setup',
        'Phase 8: Compliance Validation',
        'Phase 9: User Acceptance Testing',
        'Phase 10: Go-Live & Handover',
        'Phase 11: Post-Go-Live Monitoring',
    ]
    canvas.setFillColor(LIGHT_BG)
    canvas.roundRect(MARGIN + 4, h * 0.29, CONTENT_W, h * 0.15, 4, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica-Bold', 7.5)
    canvas.drawString(MARGIN + 12, h * 0.29 + h * 0.15 - 12, 'CHECKLIST PHASES:')
    col_w = CONTENT_W / 2
    for i, ph in enumerate(phase_list):
        col = i % 2
        row = i // 2
        x = MARGIN + 12 + col * col_w
        y_pos = h * 0.29 + h * 0.15 - 24 - row * 12
        canvas.setFillColor(GOLD)
        canvas.circle(x + 2, y_pos + 3.5, 2, fill=1, stroke=0)
        canvas.setFillColor(DARK_TEXT)
        canvas.setFont('Helvetica', 8)
        canvas.drawString(x + 9, y_pos, ph)

    # Bottom strip
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(MARGIN + 4, 0.82 * inch, 'Version 1.0  |  February 2026  |  Confidential — Authorized Deployment Teams Only')
    canvas.setFillColor(HexColor('#AABBCC'))
    canvas.setFont('Helvetica', 8.5)
    canvas.drawString(MARGIN + 4, 0.54 * inch, 'enterprise@qbitel.com  |  bridge.qbitel.com')

    canvas.restoreState()


def draw_page(canvas, doc):
    canvas.saveState()
    # Header
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 0.42 * inch, PAGE_W, 0.42 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, PAGE_H - 0.42 * inch - 3, PAGE_W, 3, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 9.5)
    canvas.drawString(MARGIN, PAGE_H - 0.29 * inch, 'QBITEL BRIDGE')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica', 9)
    canvas.drawString(MARGIN + 1.05 * inch, PAGE_H - 0.29 * inch,
                      'BPO Deployment & Delivery Checklist')
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 8)
    pg = f'Page {doc.page}'
    pw = canvas.stringWidth(pg, 'Helvetica', 8)
    canvas.drawString(PAGE_W - MARGIN - pw, PAGE_H - 0.29 * inch, pg)

    # Footer
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, 0.38 * inch, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0.38 * inch, PAGE_W, 2, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 7.5)
    canvas.drawString(MARGIN, 0.14 * inch,
                      'Confidential — For Authorized Deployment Teams Only  |  © 2026 QBITEL. All Rights Reserved.')
    contact = 'enterprise@qbitel.com  |  bridge.qbitel.com'
    cw = canvas.stringWidth(contact, 'Helvetica', 7.5)
    canvas.drawString(PAGE_W - MARGIN - cw, 0.14 * inch, contact)
    canvas.restoreState()


# ─── Helper Styles ────────────────────────────────────────────────────────────

def get_styles():
    S = {}
    S['body'] = ParagraphStyle('body', fontName='Helvetica', fontSize=9.5,
                                leading=14, textColor=DARK_TEXT, spaceAfter=5)
    S['note'] = ParagraphStyle('note', fontName='Helvetica-Oblique', fontSize=8.5,
                                leading=13, textColor=MID_GREY, spaceAfter=5)
    S['callout'] = ParagraphStyle('callout', fontName='Helvetica-Bold', fontSize=9.5,
                                   leading=14, textColor=NAVY, spaceAfter=5)
    S['table_hdr'] = ParagraphStyle('th', fontName='Helvetica-Bold', fontSize=8.5,
                                     textColor=WHITE_C)
    S['table_cell'] = ParagraphStyle('tc', fontName='Helvetica', fontSize=8.5,
                                      textColor=DARK_TEXT)
    S['table_bold'] = ParagraphStyle('tb', fontName='Helvetica-Bold', fontSize=8.5,
                                      textColor=NAVY)
    return S


def sp(n=6):
    return Spacer(1, n)


def check(text, indent=0, highlight=None, subs=None):
    return CheckItem(text, sub_items=subs or [], indent=indent, highlight=highlight)


def section(num, title):
    return ChecklistSection(num, title)


def make_checklist_table(headers, rows, col_widths, S):
    """Standard table for checklist tables."""
    data = [[Paragraph(h, S['table_hdr']) for h in headers]]
    for ri, row in enumerate(rows):
        data.append([
            Paragraph(str(c), S['table_bold'] if ci == 0 else S['table_cell'])
            for ci, c in enumerate(row)
        ])
    n = len(data)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE_C),
        *[('ROWBACKGROUND', (0, i), (-1, i), TABLE_ALT if i % 2 == 1 else WHITE_C)
          for i in range(1, n)],
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#C8D8E8')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ])
    tbl = Table(data, colWidths=[w * inch for w in col_widths])
    tbl.setStyle(style)
    return tbl


# ─── Build PDF ────────────────────────────────────────────────────────────────

def build_pdf():
    out = 'docs/brochures/QBITEL_BPO_Deployment_Checklist.pdf'
    doc = BaseDocTemplate(
        out, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.62 * inch, bottomMargin=0.56 * inch,
    )
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H, id='cover')
    inner_frame = Frame(MARGIN, 0.56 * inch,
                        PAGE_W - 2 * MARGIN, PAGE_H - 0.62 * inch - 0.56 * inch, id='inner')
    doc.addPageTemplates([
        PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover),
        PageTemplate(id='Inner', frames=[inner_frame], onPage=draw_page),
    ])

    S = get_styles()
    story = []
    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())

    # ── OVERVIEW TABLE ─────────────────────────────────────────────────────────
    story.append(Paragraph('<b>DEPLOYMENT OVERVIEW</b>',
                           ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=13,
                                          textColor=NAVY, spaceAfter=6)))
    story.append(Paragraph(
        'End-to-end delivery checklist for deploying QBITEL Bridge in a BPO/Call Center environment. '
        'Track every phase from pre-sales discovery through post-go-live validation.',
        S['note']))
    story.append(sp(6))

    story.append(make_checklist_table(
        ['Phase', 'Duration', 'Owner', 'Status'],
        [
            ['Phase 0: Pre-Engagement & Scoping', '1–2 days', 'Sales + Presales', '☐ Pending'],
            ['Phase 1: AI Protocol Discovery', '2–4 hours', 'QBITEL AI Engine', '☐ Pending'],
            ['Phase 2: Infrastructure Readiness', '1–2 days', 'Customer IT + QBITEL', '☐ Pending'],
            ['Phase 3: Security & Compliance Setup', '4–6 hours', 'QBITEL + Compliance', '☐ Pending'],
            ['Phase 4: Protocol Protection Activation', '2–4 hours', 'QBITEL Engine', '☐ Pending'],
            ['Phase 5: Integration Deployment', '4–8 hours', 'QBITEL + Customer IT', '☐ Pending'],
            ['Phase 6: Automation Recipe Execution', '2–3 hours', 'Zero-Touch Orchestrator', '☐ Pending'],
            ['Phase 7: Monitoring & Alerting Setup', '1–2 hours', 'QBITEL + SOC', '☐ Pending'],
            ['Phase 8: Compliance Validation', '2–4 hours', 'Compliance Officer', '☐ Pending'],
            ['Phase 9: User Acceptance Testing', '1 day', 'QA + Operations', '☐ Pending'],
            ['Phase 10: Go-Live & Handover', '2–4 hours', 'All Teams', '☐ Pending'],
            ['Phase 11: Post-Go-Live Monitoring', '30 days', 'QBITEL CSM', '☐ Pending'],
        ],
        [3.0, 1.1, 1.9, 1.0],
        S
    ))
    story.append(sp(8))
    story.append(Paragraph(
        '★  Total Activation Window: <b>4–6 hours (zero downtime)</b>  |  '
        'Full Validation & Hardening: <b>5–10 business days</b>',
        ParagraphStyle('bold_note', fontName='Helvetica-Bold', fontSize=9,
                       textColor=TEAL, spaceAfter=0)))
    story.append(PageBreak())

    # ── PHASE 0 ────────────────────────────────────────────────────────────────
    story.append(PhaseHeader('0', 'Pre-Engagement & Scoping', '1–2 Days', 'Sales + Presales'))
    story.append(sp(8))

    story.append(KeepTogether([
        section('0.1', 'Customer Information Gathering'), sp(4),
        check('BPO type: ☐ Financial Services  ☐ Healthcare  ☐ General CS  ☐ Remote Workforce  ☐ Multi-Tenant'),
        check('Total concurrent agent seat count recorded'),
        check('Number of sites/locations and offshore agent countries documented'),
        check('Remote agent percentage and locations captured'),
        check('Annual carrier invoice reviewed for toll fraud baseline'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('0.2', 'Existing Infrastructure Inventory'), sp(4),
        check('PBX/ACD platform(s) identified', subs=[
            '☐ Avaya Aura/CM  ☐ Cisco CUCM  ☐ Genesys Cloud  ☐ Asterisk/FreePBX',
            '☐ Mitel  ☐ BroadWorks  ☐ RingCentral  ☐ Other']),
        check('CRM platform(s) identified: Salesforce / Zendesk / ServiceNow / Dynamics 365'),
        check('WFM platform(s): NICE / Verint / Aspect / Calabrio / Genesys WFM'),
        check('Call recording platform(s) identified and recording storage path confirmed'),
        check('Mainframe terminal access: ☐ IBM TN3270e  ☐ IBM TN5250  ☐ None'),
        check('IVR/self-service platform identified and payment call flows documented'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('0.3', 'Compliance Requirements'), sp(4),
        check('Compliance frameworks applicable confirmed', subs=[
            '☐ PCI-DSS 4.0  ☐ TCPA  ☐ HIPAA  ☐ HITECH  ☐ SOC 2 Type II',
            '☐ GDPR  ☐ SOX  ☐ GLBA  ☐ FCA/MiFID II']),
        check('Upcoming audit dates noted and timeline confirmed with QBITEL'),
        check('Per-tenant compliance requirements captured for multi-tenant BPOs'),
        check('Recording retention periods confirmed per client contract', highlight='critical'),
        check('DPA / BAA requirements confirmed with legal team', highlight='critical'),
        check('Data residency requirements identified (EU, US, APAC)'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('0.4', 'Network & Access Pre-Requisites'), sp(4),
        check('SPAN port / network tap access confirmed with IT team', highlight='critical'),
        check('Required firewall ports reviewed and approved', subs=[
            'SIP: 5060 (UDP/TCP), 5061 (TLS), 5062 (PQC-TLS)',
            'RTP: 16384–32767 (UDP) | Management: 443, 8443',
            'HSM: 1792 (if hardware HSM)']),
        check('Network diagram obtained and voice VLAN/segmentation documented'),
        check('Change management window confirmed with operations team'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('0.5', 'Stakeholder Sign-Off — Phase 0'),
        sp(6),
        SignOffBlock([
            'CTO / IT Director', 'CISO / Security Lead',
            'Chief Compliance Officer', 'Contact Center Director',
            'QBITEL Delivery Lead', 'Customer Executive Sponsor',
        ]),
        sp(6)
    ]))

    story.append(PageBreak())

    # ── PHASE 1 ────────────────────────────────────────────────────────────────
    story.append(PhaseHeader('1', 'AI Protocol Discovery', '2–4 Hours', 'QBITEL AI Engine'))
    story.append(sp(4))
    story.append(Paragraph('Passive phase — zero traffic impact. No call disruption at any step.',
                           S['note']))
    story.append(sp(8))

    story.append(KeepTogether([
        section('1.1', 'Network Tap Deployment'), sp(4),
        check('SPAN port configured on voice/data switch', highlight='critical'),
        check('Passive network tap placed at SIP trunk boundary'),
        check('QBITEL Sensor VM deployed, powered on, and connectivity confirmed'),
        check('Tap verified as read-only — no traffic interference confirmed'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('1.2', 'Discovery Phase Execution'), sp(4),
        check('Discovery phase started in QBITEL Management Console'),
        check('Statistical analysis completed (5–10 sec) — entropy + byte frequency analysis'),
        check('ML classification completed (10–20 sec) — 89%+ accuracy target'),
        check('Grammar learning completed (1–2 min) — PCFG + Transformer semantic analysis'),
        check('Parser generation completed (30–60 sec) — 50,000+ msg/sec parsers'),
        check('Full discovery report reviewed and approved by customer IT team'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('1.3', 'Protocol Discovery Results'), sp(4),
        check('SIP (port 5060) — session count recorded'),
        check('SIP-TLS (5061) and SIP-PQC-TLS (5062) status confirmed'),
        check('RTP/SRTP concurrent stream count recorded'),
        check('DTMF (RFC 2833/4733) relay paths identified'),
        check('TN3270e / TN5250 sessions — agent count recorded'),
        check('CTI (TSAPI / CSTA / Cisco Finesse) events discovered'),
        check('IVR (VoiceXML / MRCP / CCXML) traffic identified'),
        check('Undocumented/unexpected protocols flagged and risk-assessed', highlight='warning'),
        check('Unencrypted PII-carrying streams flagged and documented', highlight='critical'),
        sp(6)
    ]))

    story.append(PageBreak())

    # ── PHASE 2 ────────────────────────────────────────────────────────────────
    story.append(PhaseHeader('2', 'Infrastructure Readiness', '1–2 Days', 'Customer IT + QBITEL'))
    story.append(sp(8))

    story.append(section('2.1', 'QBITEL Engine Server Requirements'))
    story.append(sp(4))
    story.append(make_checklist_table(
        ['Component', 'Minimum Spec', 'Recommended', '☐ Ready'],
        [
            ['API Servers (x3)', '4 CPU, 16GB RAM', '8 CPU, 32GB RAM', '☐'],
            ['PostgreSQL Primary', '8 CPU, 32GB, 500GB SSD', '16 CPU, 64GB, 1TB NVMe', '☐'],
            ['PostgreSQL Replica', '8 CPU, 32GB, 500GB SSD', 'Same as primary', '☐'],
            ['Redis Cluster (x3)', '2 CPU, 8GB RAM', '4 CPU, 16GB RAM', '☐'],
            ['Load Balancer (HA x2)', '2 CPU, 4GB RAM', '4 CPU, 8GB RAM', '☐'],
            ['Monitoring Stack', '4 CPU, 8GB RAM', '8 CPU, 16GB RAM', '☐'],
            ['QBITEL Sensor (per site)', '2 CPU, 8GB RAM', '4 CPU, 16GB RAM', '☐'],
        ],
        [2.8, 1.6, 1.6, 0.6],
        S
    ))
    story.append(sp(8))

    story.append(KeepTogether([
        section('2.2', 'Core Infrastructure Setup'), sp(4),
        check('OS confirmed: Ubuntu 22.04 LTS or RHEL 8+ on all servers'),
        check('Python 3.10+ / Docker 20.10+ / Kubernetes 1.25+ installed'),
        check('NTP synchronized across all servers'),
        check('DNS resolution working for all internal hostnames'),
        check('Storage throughput tested: minimum 500 MB/s for recording encryption workloads'),
        check('PostgreSQL 15+ with primary-replica replication confirmed', highlight='critical'),
        check('Automated backup schedule: every 6 hours, backup restoration tested (RTO: 30 min)'),
        check('Database encryption at rest (AES-256-GCM) enabled', highlight='critical'),
        check('QBITEL schema migration run: alembic upgrade head'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('2.3', 'HSM & Cryptographic Setup (Financial / Payment BPOs)'), sp(4),
        check('HSM provisioned: FIPS 140-3 Level 3 configuration', highlight='critical'),
        check('ML-KEM (Kyber) and ML-DSA (Dilithium) algorithms loaded to HSM'),
        check('HSM network connectivity to QBITEL Engine verified'),
        check('HSM backup key escrow configured and documented'),
        check('Software HSM approved for non-financial BPOs (if applicable)'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('2.4', 'On-Premise AI / LLM Setup'), sp(4),
        check('Ollama installed and running on QBITEL Engine server'),
        check('LLM model selected and downloaded', subs=[
            '☐ Llama 3.2 8B (standard)  ☐ Llama 3.2 70B (enterprise)',
            '☐ Mixtral 8x7B (multilingual BPOs)  ☐ Qwen 2.5 (APAC)']),
        check('LLM response time tested: <5 seconds for threat narrative generation'),
        check('Air-gapped mode confirmed (if required) — no internet connectivity after setup'),
        sp(6)
    ]))

    story.append(PageBreak())

    # ── PHASE 3 ────────────────────────────────────────────────────────────────
    story.append(PhaseHeader('3', 'Security & Compliance Configuration', '4–6 Hours', 'QBITEL + Compliance'))
    story.append(sp(8))

    story.append(KeepTogether([
        section('3.1', 'Security Policy Selection (select one)'), sp(4),
        check('☐ Financial Services BPO Policy — PCI-DSS 4.0 + SOX + GLBA (CRITICAL, 256-bit, 10yr retention)', highlight='critical'),
        check('☐ Healthcare BPO Policy — HIPAA + HITECH (CRITICAL, 256-bit, 6yr retention)', highlight='critical'),
        check('☐ General Customer Service Policy — GDPR + SOC 2 (ENHANCED, 192-bit, 3yr retention)'),
        check('☐ Remote Workforce Policy — Enhanced endpoint security (ENHANCED, TOTP + biometric MFA)'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('3.2', 'PQC Key Generation'), sp(4),
        check('ML-KEM-512 key pairs — Voice signaling & CTI (latency target: <50ms)', highlight='critical'),
        check('ML-KEM-768 key pairs — Agent desktop, terminal emulation, remote access'),
        check('ML-KEM-1024 key pairs — Payment processing and call recording', highlight='critical'),
        check('ML-DSA-65 / ML-DSA-87 signature keys generated'),
        check('Falcon-512 compact signature keys for bandwidth-constrained paths'),
        check('All keys stored in HSM — key rotation schedule configured'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('3.3', 'Authentication & Access Control'), sp(4),
        check('JWT secret configured (minimum 32 chars, cryptographically random)', highlight='critical'),
        check('MFA enforced for all QBITEL Management Console admin accounts'),
        check('RBAC roles configured: Admin, Security Analyst, Compliance Viewer, Read-Only'),
        check('Service account credentials rotated from defaults', highlight='critical'),
        check('SSH key-only access configured — password auth disabled'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('3.4', 'Multi-Tenant Configuration (if applicable)'), sp(4),
        check('Per-tenant cryptographic key isolation confirmed', highlight='critical'),
        check('Per-tenant compliance policy applied independently'),
        check('Per-tenant audit trail segregation verified'),
        check('Cross-tenant data access tested and confirmed blocked'),
        sp(6)
    ]))

    story.append(PageBreak())

    # ── PHASE 4 ────────────────────────────────────────────────────────────────
    story.append(PhaseHeader('4', 'Protocol Protection Activation', '2–4 Hours', 'QBITEL Engine'))
    story.append(sp(4))
    story.append(Paragraph('Activation is per-trunk and staged. Voice calls continue uninterrupted throughout.',
                           S['note']))
    story.append(sp(8))

    story.append(KeepTogether([
        section('4.1', 'SIP Voice Signaling Protection'), sp(4),
        check('First test trunk selected for initial activation', highlight='critical'),
        check('SIP-PQC-TLS activated on port 5062 (hybrid mode with X25519)'),
        check('SIP signaling latency measured: _____ ms (target: <50ms p95)'),
        check('Call setup test (10 calls): all succeeded'),
        check('Remaining trunks activated one-by-one with 30-minute monitoring between each'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('4.2', 'RTP Media Encryption'), sp(4),
        check('SRTP-PQC activated: ML-KEM-512 + AES-256-GCM'),
        check('RTP overhead measured: _____ ms (target: <2ms)', highlight='critical'),
        check('MOS score before: _____  |  MOS score after: _____ (must be equal or better)', highlight='critical'),
        check('Packet loss and jitter confirmed — no regression'),
        check('Codec compatibility: ☐ G.711  ☐ G.729  ☐ G.722  ☐ Opus'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('4.3', 'DTMF Masking Activation'), sp(4),
        check('Masking mode selected: ☐ CLAMP  ☐ FLAT_TONE  ☐ SILENCE  ☐ REPLACE'),
        check('DTMF masking activated on all IVR payment paths', highlight='critical'),
        check('Masking latency measured: _____ ms (target: <5ms)'),
        check('Test payment call: Card digits NOT audible in agent headset ☐', highlight='critical'),
        check('Test payment call: Card digits NOT present in recording ☐', highlight='critical'),
        check('Test payment call: Payment gateway received DTMF digits correctly ☐'),
        check('PAN detection engine activated — tested with Luhn-valid test card numbers'),
        check('Agent screen masking: Last 4 digits only display confirmed'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('4.4', 'Call Recording Encryption'), sp(4),
        check('Recording encryption: ML-KEM-1024 + AES-256-GCM activated', highlight='critical'),
        check('Recording throughput tested: _____ concurrent streams (target: 10,000+)'),
        check('Auto pause-on-payment: triggered, event logged (timestamp, agent_id, call_id)'),
        check('Auto resume after timeout: working, event logged'),
        check('Recording tamper detection: encrypted file unreadable without QBITEL key'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('4.5', 'TN3270e / TN5250 Terminal Protection'), sp(4),
        check('TN3270e PQC tunnel wrapper activated'),
        check('TN5250 PQC tunnel wrapper activated (if applicable)'),
        check('Terminal session latency: _____ ms (target: <300ms)'),
        check('Agents confirmed: no visible change to terminal emulator'),
        check('Session audit logging confirmed active'),
        sp(6)
    ]))

    story.append(PageBreak())

    # ── PHASE 5 ────────────────────────────────────────────────────────────────
    story.append(PhaseHeader('5', 'Integration Deployment', '4–8 Hours', 'QBITEL + Customer IT'))
    story.append(sp(8))

    story.append(KeepTogether([
        section('5.1', 'PBX / ACD Integration'), sp(4),
        check('Avaya Aura: TSAPI/DMCC with PQC tunnel — call flow test (100 calls): ☐ Pass'),
        check('Cisco CUCM: CTI-OS / Finesse API with PQC tunnel — call flow test: ☐ Pass'),
        check('Genesys Cloud: REST API with PQC-TLS — OAuth token secured — test: ☐ Pass'),
        check('Asterisk/FreePBX: AMI/ARI with PQC tunnel — dialplan compatibility: ☐ Pass'),
        check('PBX admin credentials secured in vault'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('5.2', 'CRM & WFM Integration'), sp(4),
        check('CRM API connected with PII masking and data classification active'),
        check('CRM data access audit logging confirmed'),
        check('Rate limiting on bulk API queries enabled (prevent data exfiltration)', highlight='critical'),
        check('WFM bridge configured and schedule enforcement confirmed'),
        check('CRM screen pop test from agent desktop: ☐ Pass'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('5.3', 'SIEM / SOC Integration'), sp(4),
        check('CEF/syslog output configured and test events received in SIEM: ☐ Pass'),
        check('QBITEL event fields verified: call_id, agent_id, tenant_id, trunk_id, fraud_type'),
        check('SOAR webhook configured — test trigger executed: ☐ Pass'),
        sp(6)
    ]))

    story.append(PageBreak())

    # ── PHASE 6 ────────────────────────────────────────────────────────────────
    story.append(PhaseHeader('6', 'Automation Recipe Execution', '2–3 Hours', 'Zero-Touch Orchestrator'))
    story.append(sp(4))
    story.append(Paragraph('Run via QBITEL Zero-Touch Orchestrator. Each recipe must complete with 100% step success.',
                           S['note']))
    story.append(sp(8))

    recipes = [
        ('6.1', 'Recipe 1: PCI-DSS Voice Compliance', [
            'Step 1: DTMF masking configured on all IVR paths (CLAMP mode)',
            'Step 2: Recording pause/resume enabled (auto-pause on payment detection)',
            'Step 3: PAN detection engine deployed (credit/debit card detection)',
            'Step 4: Agent screen masking configured (last 4 digits only)',
            'Step 5: PCI scope management activated (auto-descope)',
            'Step 6: PCI-DSS SAQ-D compliance report generated',
        ]),
        ('6.2', 'Recipe 2: Toll Fraud Prevention', [
            'Step 1: Premium rate number database loaded (200+ countries)',
            'Step 2: IRSF detection rules configured (block + alert)',
            'Step 3: Velocity/volume alerting set (max 20 intl/hour, 5 calls/min)',
            'Step 4: Automatic call blocking enabled (premium numbers, spoofed CLI)',
            'Step 5: Off-hours monitoring configured (block intl after business hours)',
            'Step 6: Toll fraud prevention validated with test scenarios',
        ]),
        ('6.3', 'Recipe 3: Remote Workforce Security', [
            'Step 1: PQC key pairs generated for remote agents (ML-KEM-768, HSM-stored)',
            'Step 2: VPN-less PQC tunnels configured (pqc-wireguard protocol)',
            'Step 3: Device posture checking deployed (OS, antivirus, disk encryption)',
            'Step 4: Geo-fencing rules configured for approved countries',
            'Step 5: Screen watermarking deployed (agent ID + timestamp)',
            'Step 6: Network risk assessment configured (WiFi, router security)',
        ]),
        ('6.4', 'Recipe 4: Quantum-Safe Voice Upgrade', [
            'Step 1: ML-KEM-768 key pairs generated (10 pairs for voice infrastructure)',
            'Step 2: SIP-PQC-TLS configured on port 5062 (hybrid with X25519)',
            'Step 3: SRTP-PQC enabled for media (AES-256-GCM + key rotation)',
            'Step 4: Recording keys wrapped with PQC (ML-KEM-1024)',
            'Step 5: HSM configuration updated (ML-KEM and ML-DSA enabled)',
            'Step 6: End-to-end quantum-safe voice validation completed',
        ]),
        ('6.5', 'Recipe 5: Full Compliance Suite', [
            'Step 1: PCI-DSS 4.0 voice controls deployed (Req 3.3, 3.4, 3.5, 8.3, 10.2)',
            'Step 2: TCPA consent management configured (tracking, opt-out, DNC)',
            'Step 3: SOC 2 Type II monitoring deployed',
            'Step 4: GDPR recording consent controls activated',
            'Step 5: Automated compliance reporting deployed (monthly reports)',
            'Step 6: HIPAA PHI protection activated (healthcare BPOs only)',
        ]),
    ]

    for num, title, steps in recipes:
        items = [section(num, title), sp(4)]
        for step in steps:
            items.append(check(f'☐  {step}'))
        items.append(check('All steps: ☐ 100% success  |  Failed steps: ___________', highlight='critical'))
        items.append(sp(8))
        story.append(KeepTogether(items))

    story.append(KeepTogether([
        section('6.6', 'Zero-Touch Orchestrator Final Validation'), sp(4),
        check('All 5 recipes completed with 100% step success', highlight='critical'),
        check('Orchestrator confidence scores ≥ 0.95 for all auto-executed actions'),
        check('Risk score post-deployment: _____ / 100 (target: <20)', highlight='critical'),
        check('Security score post-deployment: _____ / 100 (target: >80)'),
        check('Remaining manual gaps documented with remediation plan'),
        sp(6)
    ]))

    story.append(PageBreak())

    # ── PHASE 7 ────────────────────────────────────────────────────────────────
    story.append(PhaseHeader('7', 'Monitoring & Alerting Setup', '1–2 Hours', 'QBITEL + SOC Team'))
    story.append(sp(8))

    story.append(KeepTogether([
        section('7.1', 'Key BPO Metrics Confirmed in Prometheus'), sp(4),
        check('Active agent sessions count'),
        check('Concurrent encrypted call streams'),
        check('DTMF masking events per hour'),
        check('Toll fraud blocks per hour and fraud type distribution'),
        check('PQC encryption latency (p50, p95, p99) — all within targets'),
        check('Autonomous response event count and confidence scores'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('7.2', 'Grafana Dashboards Active'), sp(4),
        check('Voice Quality panel: MOS, latency, packet loss'),
        check('Fraud Prevention panel: blocked calls, fraud types, estimated savings'),
        check('PCI Compliance panel: DTMF masking events, scope tracking'),
        check('Agent Security panel: DLP events, posture compliance percentage'),
        check('Autonomous Response panel: actions taken, confidence scores, SOC queue'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('7.3', 'Critical Alerting Rules Configured'), sp(4),
        check('P1 — Voice quality degradation (MOS drop >10%) → PagerDuty + SMS', highlight='critical'),
        check('P1 — Autonomous response affecting live calls → PagerDuty + Call', highlight='critical'),
        check('P1 — QBITEL Engine outage → PagerDuty + Call', highlight='critical'),
        check('P2 — Toll fraud rate spike (>5x baseline) → Email + Slack'),
        check('P2 — PCI DTMF masking failure → Email + Slack + Compliance team', highlight='critical'),
        check('P3 — Certificate expiry <30 days → Email'),
        check('Alert escalation matrix documented: L1 (auto) → L2 (SOC) → L3 (QBITEL 24/7)'),
        sp(6)
    ]))

    story.append(PageBreak())

    # ── PHASE 8 ────────────────────────────────────────────────────────────────
    story.append(PhaseHeader('8', 'Compliance Validation', '2–4 Hours', 'Compliance Officer'))
    story.append(sp(8))

    story.append(KeepTogether([
        section('8.1', 'PCI-DSS Voice Controls Validation'), sp(4),
        check('10 test payment calls — card digits NOT present in recordings ☐', highlight='critical'),
        check('PAN detection — Luhn-valid test card numbers blocked across all streams ☐', highlight='critical'),
        check('Agent screen masking — card number masked to last 4 digits ☐'),
        check('Recording auto-pause on payment: triggered and logged ☐'),
        check('PCI scope reduction report generated and reviewed by CCO'),
        check('Compliance report generation time: _____ min (target: <10)'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('8.2', 'Toll Fraud Validation'), sp(4),
        check('10 test calls to premium-rate prefixes: ALL blocked ☐', highlight='critical'),
        check('3 test Wangiri callback patterns: detected and blocked ☐'),
        check('2 test calls with spoofed CLI: anomaly flagged ☐'),
        check('100 legitimate calls: zero false positives ☐', highlight='critical'),
        check('Fraud detection latency: _____ sec (target: <1 second)'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('8.3', 'Autonomous Response Validation'), sp(4),
        check('Simulated SIP injection attack: blocked <1 second, full audit trail ☐', highlight='critical'),
        check('LLM threat narrative generated: plain-language explanation received ☐'),
        check('Automated action reasoning chain logged with confidence score ☐'),
        check('Emergency stop tested: all autonomous actions frozen instantly ☐', highlight='critical'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('8.4', 'Compliance Reports Generated'), sp(4),
        check('PCI-DSS 4.0 compliance report ☐ Generated in _____ min'),
        check('HIPAA Technical Safeguards evidence package ☐ Generated / ☐ N/A'),
        check('SOC 2 Type II evidence package ☐ Generated'),
        check('GDPR compliance report ☐ Generated / ☐ N/A'),
        check('SOX recording integrity report ☐ Generated / ☐ N/A'),
        check('All reports reviewed and approved by Compliance Officer'),
        sp(6)
    ]))

    story.append(PageBreak())

    # ── PHASE 9 ────────────────────────────────────────────────────────────────
    story.append(PhaseHeader('9', 'User Acceptance Testing', '1 Day', 'QA + Operations'))
    story.append(sp(8))

    story.append(KeepTogether([
        section('9.1', 'Voice Quality UAT'), sp(4),
        check('100-call voice quality test with live agents', highlight='critical'),
        check('MOS scores: Pre _____ | Post _____ (must be equal or better)', highlight='critical'),
        check('Agent-reported call quality: No degradation noted ☐'),
        check('DTMF tone quality (payment flow): No degradation for callers ☐'),
        check('Transfer / hold / conference call flows: All working ☐'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('9.2', 'Agent & Operations UAT'), sp(4),
        check('Agents can log in and work normally — zero disruption confirmed ☐'),
        check('TN3270e / mainframe access working as before ☐'),
        check('CRM screen pop working correctly ☐'),
        check('Payment call flow tested end-to-end with live agent ☐', highlight='critical'),
        check('DLP policy does not block normal agent workflows ☐'),
        check('Supervisor monitoring, barge-in, and whisper working ☐'),
        check('Compliance dashboard accessible to operations and compliance teams ☐'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('9.3', 'UAT Sign-Off'),
        sp(6),
        SignOffBlock([
            'Contact Center Director', 'Operations Lead',
            'Security Team Lead', 'Compliance Officer',
        ]),
        sp(6)
    ]))

    story.append(PageBreak())

    # ── PHASE 10 ───────────────────────────────────────────────────────────────
    story.append(PhaseHeader('10', 'Go-Live & Handover', '2–4 Hours', 'All Teams'))
    story.append(sp(8))

    story.append(KeepTogether([
        section('10.1', 'Pre-Go-Live Final Checks (Day Before)'), sp(4),
        check('Code freeze on QBITEL Engine in customer environment'),
        check('Full staging environment validation completed'),
        check('Load testing completed at projected peak call volume'),
        check('Final security scan: ☐ No critical findings', highlight='critical'),
        check('Rollback procedure tested and confirmed working', highlight='critical'),
        check('Emergency contact list distributed to all teams'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('10.2', 'Go-Live Day Execution'), sp(4),
        check('All monitoring dashboards open and green before activation', highlight='critical'),
        check('QBITEL support team on standby for first 4 hours'),
        check('30-minute post-activation health check completed', subs=[
            'Call success rate ≥99.9% baseline  |  DTMF masking active on all payment paths',
            'Toll fraud monitoring active  |  Recording encryption active  |  No P1 alerts']),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('10.3', 'Runbook & Documentation Handover'), sp(4),
        check('QBITEL Engine administration runbook delivered'),
        check('Incident response playbooks delivered', subs=[
            'Toll fraud response procedure',
            'DTMF masking failure recovery',
            'Remote agent quarantine procedure',
            'PCI audit evidence retrieval procedure']),
        check('Rollback procedure document delivered'),
        check('Compliance report generation guide delivered'),
        check('Emergency stop procedure documented and posted'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('10.4', 'Training Delivery Completed'), sp(4),
        check('SOC team: QBITEL security event response training (2 hours) ☐'),
        check('Operations team: Compliance dashboard and reporting (1 hour) ☐'),
        check('IT admin team: QBITEL Engine administration (2 hours) ☐'),
        check('Management briefing: Autonomous response and audit trail (1 hour) ☐'),
        check('Agent briefing (DLP policy only): What changed and why (30 min) ☐'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('10.5', 'Go-Live Sign-Off — All Parties Required'),
        sp(6),
        SignOffBlock([
            'Engineering Lead', 'CISO / Security Lead',
            'IT / DevOps Lead', 'Compliance Officer',
            'QBITEL Delivery Lead', 'Customer Executive Sponsor',
        ]),
        sp(6)
    ]))

    story.append(PageBreak())

    # ── PHASE 11 ───────────────────────────────────────────────────────────────
    story.append(PhaseHeader('11', 'Post-Go-Live Monitoring', '30 Days', 'QBITEL CSM'))
    story.append(sp(8))

    story.append(KeepTogether([
        section('11.1', 'Week 1 — Stabilization'), sp(4),
        check('Daily health check calls (QBITEL CSM + Customer IT): ☐ Day 1  ☐ Day 2  ☐ Day 3  ☐ Day 4  ☐ Day 5'),
        check('False positive review — toll fraud false positives: _____ (tune if >0)'),
        check('False positive review — DLP false positives: _____ (tune if >0)'),
        check('All autonomous response actions reviewed and confirmed appropriate'),
        check('AI model baseline learning confirmed: environment baseline established'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('11.2', 'Week 2 — Optimization'), sp(4),
        check('Toll fraud detection rate reviewed and baseline set'),
        check('PQC encryption performance baseline confirmed — no latency regressions'),
        check('First compliance report generated and reviewed by Compliance Officer'),
        check('First PCI-DSS scope reduction calculation delivered to CCO'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('11.3', 'Weeks 3–4 — Hardening'), sp(4),
        check('Encryption key rotation cycle confirmed active', highlight='critical'),
        check('DR drill completed — backup restoration tested'),
        check('Security posture re-assessed by Zero-Touch Orchestrator'),
        check('Risk score at 30 days: _____ / 100 (target: <15)', highlight='critical'),
        check('Security score at 30 days: _____ / 100 (target: >85)'),
        check('Monthly compliance reports generated for all applicable frameworks'),
        sp(6)
    ]))

    story.append(KeepTogether([
        section('11.4', '30-Day Business Review'), sp(4),
        check('Toll fraud losses prevented: $ ___________'),
        check('Fraud events detected and blocked: ___________'),
        check('PCI-DSS audit scope reduction confirmed: _____ %'),
        check('Autonomous threat resolution rate: _____ % (target: ≥78%)'),
        check('False positive rate: _____ % (target: <1%)'),
        check('Agent complaints or workflow issues: ___________'),
        check('30-day executive review meeting completed ☐'),
        sp(6)
    ]))

    story.append(PageBreak())

    # ── APPENDIX A: PERFORMANCE TARGETS ───────────────────────────────────────
    story.append(Paragraph('APPENDIX A — Performance Targets',
                           ParagraphStyle('app', fontName='Helvetica-Bold', fontSize=13,
                                          textColor=NAVY, spaceAfter=6, spaceBefore=0)))
    story.append(sp(4))
    story.append(PerformanceTargetBlock([
        ['Voice PQC encryption overhead', '<2ms', '___ ms', '☐'],
        ['DTMF masking latency', '<5ms', '___ ms', '☐'],
        ['SIP signaling processing', '<10ms p95', '___ ms', '☐'],
        ['TN3270e session latency', '<300ms', '___ ms', '☐'],
        ['Toll fraud detection', '<1 second', '___ sec', '☐'],
        ['PAN detection latency', '<50ms', '___ ms', '☐'],
        ['Recording encryption throughput', '10,000+ streams', '___ streams', '☐'],
        ['Compliance report generation', '<10 minutes', '___ min', '☐'],
        ['Autonomous threat resolution', '≥78%', '___ %', '☐'],
        ['Call success rate post-deployment', '≥99.9% baseline', '___ %', '☐'],
        ['Full deployment time', '4–6 hours', '___ hours', '☐'],
    ]))
    story.append(sp(14))

    # ── APPENDIX B: SUPPORT CONTACTS ──────────────────────────────────────────
    story.append(Paragraph('APPENDIX B — Support & Escalation Contacts',
                           ParagraphStyle('app', fontName='Helvetica-Bold', fontSize=13,
                                          textColor=NAVY, spaceAfter=6)))
    story.append(sp(4))
    story.append(make_checklist_table(
        ['Priority', 'Trigger', 'Contact', 'SLA'],
        [
            ['P1 Critical', 'Voice degradation, live call impact, Engine outage', 'enterprise@qbitel.com + 24/7 phone', '30 min (Global) / 1hr (Ent)'],
            ['P2 High', 'Fraud spike, masking failure, compliance alert', 'enterprise@qbitel.com', '4 hours'],
            ['P3 Warning', 'Certificate expiry, capacity, config drift', 'enterprise@qbitel.com', '1 business day'],
            ['General', 'Questions, optimization, reporting', 'bridge.qbitel.com support portal', '2 business days'],
        ],
        [0.9, 2.8, 2.0, 1.3],
        S
    ))
    story.append(sp(14))

    # ── CLOSING ────────────────────────────────────────────────────────────────
    contact_data = [
        [Paragraph('<b><font color="#008B9A">✉</font>  enterprise@qbitel.com</b>',
                   ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=9.5, textColor=WHITE_C)),
         Paragraph('<b><font color="#008B9A">⊕</font>  bridge.qbitel.com</b>',
                   ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=9.5, textColor=WHITE_C)),
         Paragraph('<b><font color="#008B9A">◉</font>  24/7 Support Available</b>',
                   ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=9.5, textColor=WHITE_C))],
        [Paragraph('Enterprise & Deployment', ParagraphStyle('cs', fontName='Helvetica', fontSize=8, textColor=TEAL)),
         Paragraph('Portal & Documentation', ParagraphStyle('cs', fontName='Helvetica', fontSize=8, textColor=TEAL)),
         Paragraph('P1 Critical Issues', ParagraphStyle('cs', fontName='Helvetica', fontSize=8, textColor=TEAL))],
    ]
    ctbl = Table(contact_data, colWidths=[CONTENT_W / 3] * 3)
    ctbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), NAVY),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, TEAL),
    ]))
    story.append(ctbl)
    story.append(sp(12))
    story.append(Paragraph(
        '<i>QBITEL Bridge — Because the Human API Deserves Quantum-Safe Protection.</i>',
        ParagraphStyle('closing', fontName='Helvetica-BoldOblique', fontSize=12,
                       textColor=NAVY, alignment=TA_CENTER, spaceAfter=6)
    ))
    story.append(Paragraph(
        'Version 1.0  |  February 2026  |  Confidential — For Authorized Deployment Teams Only',
        ParagraphStyle('ver', fontName='Helvetica', fontSize=8,
                       textColor=MID_GREY, alignment=TA_CENTER)
    ))

    doc.build(story)
    print(f'PDF saved: {out}')
    return out


if __name__ == '__main__':
    build_pdf()
