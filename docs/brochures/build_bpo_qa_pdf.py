"""
Build QBITEL Bridge BPO Pitch Q&A Guide - Professional PDF
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
from reportlab.platypus import NextPageTemplate

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
RED_LIGHT  = HexColor('#FFF0F0')
GREEN_LIGHT= HexColor('#F0FFF4')

PAGE_W, PAGE_H = letter
MARGIN = 0.85 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


# ─── Custom Flowables ─────────────────────────────────────────────────────────

class SectionHeader(Flowable):
    def __init__(self, number, title, subtitle=None, width=None):
        super().__init__()
        self.number = number
        self.title = title
        self.subtitle = subtitle
        self.w = width or CONTENT_W
        self.h = 56 if subtitle else 42

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Full background
        c.setFillColor(NAVY)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        # Gold number badge
        c.setFillColor(GOLD)
        c.rect(0, 0, 44, self.h, fill=1, stroke=0)
        # Teal right accent
        c.setFillColor(TEAL)
        c.rect(self.w - 5, 0, 5, self.h, fill=1, stroke=0)
        # Number
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 16)
        num_w = c.stringWidth(self.number, 'Helvetica-Bold', 16)
        c.drawString(22 - num_w / 2, self.h / 2 - 8, self.number)
        # Title
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 13)
        title_y = self.h - 22 if self.subtitle else self.h / 2 - 6
        c.drawString(54, title_y, self.title.upper())
        if self.subtitle:
            c.setFillColor(TEAL)
            c.setFont('Helvetica-Oblique', 9)
            c.drawString(54, 9, self.subtitle)


class QABlock(Flowable):
    """A single Q&A pair as a styled card."""
    def __init__(self, q_id, question, oneliner, answer_lines, width=None):
        super().__init__()
        self.q_id = q_id
        self.question = question
        self.oneliner = oneliner
        self.answer_lines = answer_lines
        self.w = width or CONTENT_W
        # Calculate height
        self.line_h = 13
        self.q_h = 32
        self.ol_h = max(28, len(oneliner) // 85 * 14 + 28)
        self.ans_h = len(answer_lines) * self.line_h + 16
        self.h = self.q_h + self.ol_h + self.ans_h + 6

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        total_h = self.h

        # Outer border
        c.setStrokeColor(HexColor('#C8D8E8'))
        c.setLineWidth(0.5)
        c.roundRect(0, 0, self.w, total_h, 4, fill=0, stroke=1)

        # Question header band
        c.setFillColor(NAVY)
        c.roundRect(0, total_h - self.q_h, self.w, self.q_h, 4, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.rect(0, total_h - self.q_h, self.w, self.q_h / 2, fill=1, stroke=0)

        # Q ID badge
        c.setFillColor(GOLD)
        c.roundRect(8, total_h - self.q_h + 6, 36, 20, 3, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 8)
        qid_w = c.stringWidth(self.q_id, 'Helvetica-Bold', 8)
        c.drawString(26 - qid_w / 2, total_h - self.q_h + 13, self.q_id)

        # Question text
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 10)
        c.drawString(52, total_h - self.q_h + 11, self.question[:90])

        # One-liner band
        ol_top = total_h - self.q_h
        c.setFillColor(TEAL_LIGHT)
        c.rect(0, ol_top - self.ol_h, self.w, self.ol_h, fill=1, stroke=0)
        # Teal left accent
        c.setFillColor(TEAL)
        c.rect(0, ol_top - self.ol_h, 4, self.ol_h, fill=1, stroke=0)
        # One-liner label
        c.setFillColor(TEAL_DARK)
        c.setFont('Helvetica-Bold', 8)
        c.drawString(12, ol_top - 14, 'ONE-LINER:')
        # One-liner text
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica-Oblique', 9)
        # Wrap one-liner text
        words = self.oneliner.split()
        lines = []
        current = ''
        for word in words:
            test = (current + ' ' + word).strip()
            if c.stringWidth(test, 'Helvetica-Oblique', 9) < self.w - 24:
                current = test
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        y = ol_top - 26
        for line in lines[:3]:
            c.drawString(12, y, line)
            y -= 13

        # Answer area
        ans_top = ol_top - self.ol_h
        c.setFillColor(WHITE_C)
        c.rect(0, 0, self.w, ans_top, fill=1, stroke=0)
        # Answer label
        c.setFillColor(MID_GREY)
        c.setFont('Helvetica-Bold', 7.5)
        c.drawString(10, ans_top - 12, 'FULL ANSWER:')
        # Answer text lines
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 8.5)
        y = ans_top - 24
        for line in self.answer_lines:
            if y < 8:
                break
            if line.startswith('•'):
                c.setFillColor(TEAL)
                c.circle(16, y + 3, 2, fill=1, stroke=0)
                c.setFillColor(DARK_TEXT)
                c.drawString(22, y, line[1:].strip())
            elif line.startswith('**') and line.endswith('**'):
                c.setFillColor(NAVY)
                c.setFont('Helvetica-Bold', 8.5)
                c.drawString(10, y, line[2:-2])
                c.setFont('Helvetica', 8.5)
                c.setFillColor(DARK_TEXT)
            elif line == '':
                y -= 4
                continue
            else:
                c.drawString(10, y, line)
            y -= self.line_h


class ObjectionBlock(Flowable):
    """Styled objection + response card."""
    def __init__(self, obj_id, objection, response_lines, width=None):
        super().__init__()
        self.obj_id = obj_id
        self.objection = objection
        self.response_lines = response_lines
        self.w = width or CONTENT_W
        self.line_h = 13
        self.obj_h = 36
        self.resp_h = len(response_lines) * self.line_h + 24
        self.h = self.obj_h + self.resp_h + 4

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        total_h = self.h

        # Objection header - red-tinted
        c.setFillColor(HexColor('#8B1A1A'))
        c.roundRect(0, total_h - self.obj_h, self.w, self.obj_h, 4, fill=1, stroke=0)
        c.setFillColor(HexColor('#8B1A1A'))
        c.rect(0, total_h - self.obj_h, self.w, self.obj_h / 2, fill=1, stroke=0)

        # Objection badge
        c.setFillColor(HexColor('#FF6B6B'))
        c.roundRect(8, total_h - self.obj_h + 7, 52, 18, 3, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 7)
        badge_w = c.stringWidth('OBJECTION', 'Helvetica-Bold', 7)
        c.drawString(34 - badge_w / 2, total_h - self.obj_h + 13, 'OBJECTION')

        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 9.5)
        c.drawString(68, total_h - self.obj_h + 12, self.objection[:80])

        # Response area
        c.setFillColor(GREEN_LIGHT)
        c.roundRect(0, 0, self.w, self.resp_h, 4, fill=1, stroke=0)
        c.setFillColor(GREEN_LIGHT)
        c.rect(0, 0, self.w, self.resp_h / 2, fill=1, stroke=0)

        # Green left bar
        c.setFillColor(HexColor('#2E8B57'))
        c.rect(0, 0, 4, self.resp_h, fill=1, stroke=0)

        # Response label
        c.setFillColor(HexColor('#2E8B57'))
        c.setFont('Helvetica-Bold', 7.5)
        c.drawString(12, self.resp_h - 12, 'RESPONSE APPROACH:')

        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 8.5)
        y = self.resp_h - 24
        for line in self.response_lines:
            if y < 8:
                break
            if line == '':
                y -= 4
                continue
            c.drawString(10, y, line)
            y -= self.line_h


class AudienceBadge(Flowable):
    """Small audience indicator badge row."""
    def __init__(self, audience_list, width=None):
        super().__init__()
        self.audience = audience_list
        self.w = width or CONTENT_W
        self.h = 24

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(LIGHT_BG)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        c.setFillColor(MID_GREY)
        c.setFont('Helvetica-Bold', 7.5)
        c.drawString(8, 8, 'AUDIENCE:')
        x = 70
        for aud in self.audience:
            w = c.stringWidth(aud, 'Helvetica', 7.5) + 12
            c.setFillColor(NAVY)
            c.roundRect(x, 5, w, 14, 3, fill=1, stroke=0)
            c.setFillColor(WHITE_C)
            c.setFont('Helvetica', 7.5)
            c.drawString(x + 6, 9, aud)
            x += w + 6


class TOCEntry(Flowable):
    """Table of contents entry."""
    def __init__(self, number, title, subtitle, width=None):
        super().__init__()
        self.number = number
        self.title = title
        self.subtitle = subtitle
        self.w = width or CONTENT_W
        self.h = 36

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Hover line
        c.setFillColor(LIGHT_BG)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(0, 0, 3, self.h, fill=1, stroke=0)
        # Number
        c.setFillColor(GOLD)
        c.setFont('Helvetica-Bold', 18)
        c.drawString(12, self.h - 26, self.number)
        # Title
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 11)
        c.drawString(44, self.h - 18, self.title)
        # Subtitle
        c.setFillColor(MID_GREY)
        c.setFont('Helvetica', 8.5)
        c.drawString(44, self.h - 30, self.subtitle)
        # Dotted line
        c.setStrokeColor(HexColor('#C8D8E8'))
        c.setLineWidth(0.5)
        c.setDash(2, 3)
        c.line(0, 0, self.w, 0)
        c.setDash()


# ─── Page Callbacks ───────────────────────────────────────────────────────────

def draw_cover(canvas, doc):
    canvas.saveState()
    w, h = PAGE_W, PAGE_H

    # Full navy background
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)

    # Geometric accent — gold triangle top-right
    canvas.setFillColor(GOLD)
    p = canvas.beginPath()
    p.moveTo(w * 0.5, h)
    p.lineTo(w, h)
    p.lineTo(w, h * 0.6)
    p.close()
    canvas.drawPath(p, fill=1, stroke=0)

    # Teal triangle overlay
    canvas.setFillColor(TEAL)
    p2 = canvas.beginPath()
    p2.moveTo(w * 0.7, h)
    p2.lineTo(w, h)
    p2.lineTo(w, h * 0.78)
    p2.close()
    canvas.drawPath(p2, fill=1, stroke=0)

    # Bottom teal band
    canvas.setFillColor(TEAL_DARK)
    canvas.rect(0, 0, w, 1.4 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, 1.4 * inch, w, 4, fill=1, stroke=0)

    # Left gold accent bar
    canvas.setFillColor(GOLD)
    canvas.rect(0, 0, 8, h, fill=1, stroke=0)

    # QBITEL BRIDGE title
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 48)
    canvas.drawString(MARGIN + 4, h * 0.72, 'QBITEL BRIDGE')

    # Gold underline
    canvas.setFillColor(GOLD)
    canvas.rect(MARGIN + 4, h * 0.72 - 8, 5.2 * inch, 4, fill=1, stroke=0)

    # Document type label
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica-Bold', 15)
    canvas.drawString(MARGIN + 4, h * 0.72 - 36, 'BPO PITCH Q&A GUIDE')

    # Subtitle
    canvas.setFillColor(HexColor('#CCDDE8'))
    canvas.setFont('Helvetica-Oblique', 12)
    canvas.drawString(MARGIN + 4, h * 0.72 - 60,
                      'Questions & Answers for Business and Product Teams')

    # 3 stat boxes
    boxes = [
        ('65', 'Questions\nCovered'),
        ('10', 'Buyer\nPersonas'),
        ('7', 'Hard Objection\nHandlers'),
    ]
    bw = (CONTENT_W - 0.3 * inch) / 3
    by = h * 0.44
    bh = 0.9 * inch
    for i, (num, label) in enumerate(boxes):
        bx = MARGIN + i * (bw + 0.15 * inch)
        bg = TEAL if i % 2 == 0 else LIGHT_NAVY
        canvas.setFillColor(bg)
        canvas.roundRect(bx, by, bw, bh, 6, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.rect(bx, by + bh - 4, bw, 4, fill=1, stroke=0)
        canvas.setFillColor(GOLD if bg == LIGHT_NAVY else WHITE_C)
        canvas.setFont('Helvetica-Bold', 26)
        nw = canvas.stringWidth(num, 'Helvetica-Bold', 26)
        canvas.drawString(bx + (bw - nw) / 2, by + bh - 36, num)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 8)
        for j, ln in enumerate(label.split('\n')):
            lw = canvas.stringWidth(ln, 'Helvetica', 8)
            canvas.drawString(bx + (bw - lw) / 2, by + bh - 52 - j * 11, ln)

    # Section list
    sections = [
        '1. Business & ROI', '2. Technical & Architecture', '3. Security & Risk',
        '4. Compliance & Legal', '5. Operations',
        '6. Procurement & Vendor', '7. Hard Objections',
        '8. Competitive', '9. PoC & Pilot', '10. Post-Sales',
    ]
    canvas.setFillColor(LIGHT_BG)
    canvas.roundRect(MARGIN + 4, h * 0.26, CONTENT_W, h * 0.14, 4, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica-Bold', 7.5)
    canvas.drawString(MARGIN + 12, h * 0.26 + h * 0.14 - 14, 'SECTIONS COVERED:')
    col_w = CONTENT_W / 2
    for i, sec in enumerate(sections):
        col = i % 2
        row = i // 2
        x = MARGIN + 12 + col * col_w
        y = h * 0.26 + h * 0.14 - 28 - row * 13
        canvas.setFillColor(GOLD)
        canvas.circle(x, y + 4, 2, fill=1, stroke=0)
        canvas.setFillColor(DARK_TEXT)
        canvas.setFont('Helvetica', 8)
        canvas.drawString(x + 7, y, sec)

    # Bottom strip
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(MARGIN + 4, 0.8 * inch, 'Internal Sales Enablement Document  |  February 2026')
    canvas.setFillColor(HexColor('#AABBCC'))
    canvas.setFont('Helvetica', 8.5)
    canvas.drawString(MARGIN + 4, 0.52 * inch,
                      'enterprise@qbitel.com  |  bridge.qbitel.com')

    canvas.restoreState()


def draw_page(canvas, doc):
    canvas.saveState()

    # Top header
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 0.42 * inch, PAGE_W, 0.42 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, PAGE_H - 0.42 * inch - 3, PAGE_W, 3, fill=1, stroke=0)

    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 9.5)
    canvas.drawString(MARGIN, PAGE_H - 0.29 * inch, 'QBITEL BRIDGE')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica', 9)
    canvas.drawString(MARGIN + 1.05 * inch, PAGE_H - 0.29 * inch, 'BPO Pitch Q&A Guide')

    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 8)
    pg = f'Page {doc.page}'
    pw = canvas.stringWidth(pg, 'Helvetica', 8)
    canvas.drawString(PAGE_W - MARGIN - pw, PAGE_H - 0.29 * inch, pg)

    # Bottom footer
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, 0.38 * inch, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0.38 * inch, PAGE_W, 2, fill=1, stroke=0)

    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 7.5)
    canvas.drawString(MARGIN, 0.14 * inch,
                      'Internal Use Only  |  © 2026 QBITEL. All Rights Reserved.')
    contact = 'enterprise@qbitel.com  |  bridge.qbitel.com'
    cw = canvas.stringWidth(contact, 'Helvetica', 7.5)
    canvas.drawString(PAGE_W - MARGIN - cw, 0.14 * inch, contact)

    canvas.restoreState()


# ─── Styles ───────────────────────────────────────────────────────────────────

def get_styles():
    S = {}
    S['body'] = ParagraphStyle('body', fontName='Helvetica', fontSize=10,
                                leading=15, textColor=DARK_TEXT, spaceAfter=6,
                                spaceBefore=2, alignment=TA_JUSTIFY)
    S['intro'] = ParagraphStyle('intro', fontName='Helvetica', fontSize=10.5,
                                 leading=16, textColor=DARK_TEXT, spaceAfter=8,
                                 alignment=TA_JUSTIFY)
    S['note'] = ParagraphStyle('note', fontName='Helvetica-Oblique', fontSize=9,
                                leading=13, textColor=MID_GREY, spaceAfter=6)
    S['toc_title'] = ParagraphStyle('toc_title', fontName='Helvetica-Bold',
                                     fontSize=20, textColor=NAVY,
                                     spaceAfter=4, spaceBefore=0, alignment=TA_LEFT)
    S['toc_sub'] = ParagraphStyle('toc_sub', fontName='Helvetica', fontSize=10,
                                   textColor=MID_GREY, spaceAfter=16, alignment=TA_LEFT)
    return S


def sp(n=8):
    return Spacer(1, n)


def wrap_text(text, max_chars=85):
    """Wrap text into lines of max_chars."""
    words = text.split()
    lines = []
    current = ''
    for word in words:
        test = (current + ' ' + word).strip()
        if len(test) <= max_chars:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def make_qa(q_id, question, oneliner, answer_text):
    """Convert answer text to lines and create QABlock."""
    lines = []
    for para in answer_text.strip().split('\n'):
        para = para.strip()
        if not para:
            lines.append('')
            continue
        if para.startswith('- ') or para.startswith('• '):
            wrapped = wrap_text('• ' + para[2:], 90)
            for i, wl in enumerate(wrapped):
                lines.append(wl if i == 0 else '  ' + wl)
        elif para.startswith('**') and para.endswith('**'):
            lines.append(para)
        else:
            for wl in wrap_text(para, 92):
                lines.append(wl)
    return QABlock(q_id, question, oneliner, lines)


def make_objection(obj_id, objection, response_text):
    lines = []
    for para in response_text.strip().split('\n'):
        para = para.strip()
        if not para:
            lines.append('')
            continue
        for wl in wrap_text(para, 92):
            lines.append(wl)
    return ObjectionBlock(obj_id, objection, lines)


# ─── Build PDF ────────────────────────────────────────────────────────────────

def build_pdf():
    out_path = 'docs/brochures/QBITEL_BPO_Pitch_QA_Guide.pdf'

    doc = BaseDocTemplate(
        out_path,
        pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.62 * inch, bottomMargin=0.56 * inch,
    )

    cover_frame = Frame(0, 0, PAGE_W, PAGE_H, id='cover')
    inner_frame = Frame(
        MARGIN, 0.56 * inch,
        PAGE_W - 2 * MARGIN, PAGE_H - 0.62 * inch - 0.56 * inch,
        id='inner'
    )
    cover_template = PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover)
    inner_template = PageTemplate(id='Inner', frames=[inner_frame], onPage=draw_page)
    doc.addPageTemplates([cover_template, inner_template])

    S = get_styles()
    story = []

    # ── COVER ──────────────────────────────────────────────────────────────────
    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())

    # ── TABLE OF CONTENTS ──────────────────────────────────────────────────────
    story.append(Paragraph('Table of Contents', S['toc_title']))
    story.append(Paragraph('Navigate by section based on your buyer persona and meeting context.', S['toc_sub']))

    toc_entries = [
        ('1', 'Business & ROI Questions', 'CEO, CFO, COO — ROI, pricing, budget justification, quantum urgency'),
        ('2', 'Technical & Architecture Questions', 'CTO, IT Director — Deployment, latency, hybrid environments, mainframes'),
        ('3', 'Security & Risk Questions', 'CISO, SOC Manager — SIEM integration, AI safety, data privacy, zero-day'),
        ('4', 'Compliance & Legal Questions', 'CCO, DPO, Legal — PCI-DSS, GDPR, HIPAA, multi-framework conflicts'),
        ('5', 'Operations Questions', 'Contact Center Director — Agent impact, 24/7 deployment, remote agents'),
        ('6', 'Procurement & Vendor Questions', 'Procurement, Legal — Vendor risk, DPA, security assessment artifacts'),
        ('7', 'Hard Objections', 'Anyone in the room — The toughest pushback with scripted responses'),
        ('8', 'Competitive Questions', 'Evaluators — vs. Pindrop, Cisco, carrier-bundled security'),
        ('9', 'Proof of Concept & Pilot', 'Technical evaluators — PoC structure, production vs. lab'),
        ('10', 'Post-Sales / Implementation', 'IT, Operations — Support model, SLAs, patching'),
    ]
    for num, title, sub in toc_entries:
        story.append(TOCEntry(num, title, sub))
        story.append(sp(4))

    story.append(sp(10))
    story.append(Paragraph(
        '<b>How to use this guide:</b> Each Q&A card shows a QUESTION, a ONE-LINER for quick delivery in the room, '
        'and a FULL ANSWER for written RFP responses or deeper conversations. '
        'Objection cards show the objection and a scripted response approach.',
        S['note']
    ))
    story.append(PageBreak())

    # ── SECTION 1: BUSINESS & ROI ──────────────────────────────────────────────
    story.append(SectionHeader('1', 'Business & ROI Questions'))
    story.append(sp(6))
    story.append(AudienceBadge(['CEO', 'CFO', 'COO', 'VP Operations']))
    story.append(sp(10))

    story.append(KeepTogether([make_qa(
        'Q1.1', 'What is the ROI and how quickly will we see it?',
        'Toll fraud prevention alone typically delivers ROI within 30–60 days. PCI audit cost reduction adds $500K–$2M+ annually.',
        '''ROI comes from four measurable sources:
- Toll fraud prevention: Average BPO loses $200K–$2M annually to SIP toll fraud. A single prevented weekend attack ($30K–$80K) often covers the annual license.
- PCI-DSS audit scope reduction: DTMF masking reduces audit scope by up to 80%, cutting annual audit costs by $500K–$2M+.
- SOC team efficiency: 78% autonomous resolution recovers 2–3 analyst hours per day for a 5-person SOC team.
- Breach cost avoidance: Average BPO data breach costs $4.8M. QBITEL prevents the insider and protocol-layer attacks causing most BPO breaches.

Most customers see positive ROI within 60–90 days. We provide a pre-deployment baseline measurement so you can quantify impact directly.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q1.2', 'What does it cost and how is it priced?',
        'Per concurrent agent seat, three tiers. We size it to your actual seat count — no generic enterprise blanket license.',
        '''QBITEL Bridge for BPO is priced by concurrent agent seat:
- Contact Center tier: Up to 500 seats
- Enterprise BPO tier: 500–5,000 seats
- Global BPO tier: 5,000+ seats, unlimited tenants

All tiers include AI protocol discovery, PQC encryption, DTMF masking, toll fraud prevention, agent DLP, and compliance reporting. Contact enterprise@qbitel.com or bridge.qbitel.com for a tailored quote.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q1.3', 'We already have a committed security budget this year. Why reprioritize?',
        'QBITEL prevents losses already happening. It is not a new cost — it replaces losses you are absorbing today.',
        '''Three things are costing you money right now regardless of your security budget:
- Toll fraud is generating fraudulent charges on your carrier bills every month. Most BPOs do not attribute this correctly until auditing CDRs closely.
- Your PCI-DSS audit prep is consuming engineering and compliance team time that could be automated.
- Your call recordings are accumulating in a format that will be decryptable within their retention period.

We recommend a free 2-hour Discovery Assessment first — so you can quantify what you are currently losing before committing budget.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q1.4', 'Can you give us a business case template for our CFO?',
        'Yes — we provide a pre-built ROI model with your seat count, fraud loss, PCI spend, and SOC headcount as inputs.',
        '''Our Business Value Assessment (BVA) covers:
- Toll fraud loss baseline (estimated from CDR sample analysis)
- PCI-DSS audit cost reduction calculation
- SOC time recovery valuation
- Breach cost avoidance based on BPO industry actuarial data
- License and deployment cost
- 12-month, 24-month, and 36-month ROI projection

This is produced during Discovery Assessment and is executive-ready in PDF. Your CFO sees numbers specific to your environment, not industry averages.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q1.5', "We're not sure quantum computing is really a threat yet. Why invest now?",
        "Your call recordings are being harvested today. The threat to existing data is active — not future.",
        '''The immediate concern is not that quantum computers exist today — it is that adversaries are executing "harvest now, decrypt later" campaigns right now.
For BPOs, this is especially acute because:
- SOX requires 7-year call recording retention for financial services clients
- HIPAA requires 6-year retention for healthcare clients
- Recordings captured today will be within retention windows in 2031–2032

NIST finalized the first post-quantum cryptography standards in August 2024 — this is not experimental. Many financial services regulators are beginning to require quantum-safe roadmaps. The time to protect recordings is before they are captured.'''
    ), sp(8)]))

    story.append(PageBreak())

    # ── SECTION 2: TECHNICAL ───────────────────────────────────────────────────
    story.append(SectionHeader('2', 'Technical & Architecture Questions'))
    story.append(sp(6))
    story.append(AudienceBadge(['CTO', 'IT Director', 'Network Architect', 'Infrastructure Lead']))
    story.append(sp(10))

    story.append(KeepTogether([make_qa(
        'Q2.1', 'How does QBITEL deploy? Does it require agents on every endpoint?',
        'No agents. QBITEL taps the network passively — nothing installed on PBX, agent desktops, or mainframes.',
        '''QBITEL deploys as a network-layer overlay using a passive tap on your voice and data network. Three components:
- Network Sensor: Passive tap (SPAN port or inline) that captures traffic for AI analysis. Read-only during discovery — no traffic modification.
- QBITEL Engine: Deployed on-premise (VM or bare metal) or private cloud. Runs AI discovery, PQC enforcement, and agentic AI security.
- Management Console: Web-based dashboard for policy management, compliance reporting, and incident response.

Nothing is installed on Avaya/Cisco/Genesys PBX. Nothing installed on agent desktops for core voice protection. The optional DLP agent (for clipboard/USB/screen blocking) is lightweight and optional.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q2.2', 'We run Avaya Aura with a heavily customized dial plan. Will QBITEL break anything?',
        'No. Discovery is read-only. Encryption activates only after you review and approve the discovered protocol map.',
        '''QBITEL uses a staged deployment specifically to protect complex telephony environments:
- Phase 1 (Passive): Network tap placed. AI observes only. Zero interference with dial plan or call flows.
- Phase 2 (Review): You receive a complete protocol map. Your team reviews and approves before anything changes.
- Phase 3 (Active): PQC encryption applied selectively — you choose which trunks and call flows to protect first. Rollback available at any point.

We have deployed in complex Avaya environments with custom TSAPI integrations, multi-site dial plans, and legacy analog gateways.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q2.3', 'What is the latency impact? Our clients have strict MOS score requirements.',
        '<2ms PQC overhead — within the ITU-T G.114 150ms budget. MOS scores unaffected in all deployments to date.',
        '''Voice quality is a non-negotiable design constraint in QBITEL:
- SIP signaling encryption: <10ms p95 including full PQC operations
- RTP/SRTP media encryption: <2ms per-packet overhead using ML-KEM-512 + AES-256-GCM
- DTMF masking: <5ms — imperceptible to callers
- ITU-T G.114 total one-way delay budget: 150ms. QBITEL adds <2ms — leaving 148ms for all other factors.

We recommend including MOS score measurement as a specific PoC metric. We have never had a customer report MOS degradation attributable to QBITEL in production.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q2.4', 'We have a hybrid: Avaya on-premise, Genesys Cloud, legacy analog gateways.',
        'QBITEL discovers and protects protocol-agnostically — it does not require a single-vendor environment.',
        '''Hybrid telephony environments are the most common situation we encounter in large BPOs:
- Avaya Aura on-premise: TSAPI/DMCC integration with PQC tunnel overlay
- Genesys Cloud: REST API with PQC-TLS — cloud signaling secured
- Legacy analog gateways: Protocol discovery identifies analog-to-SIP conversion points; encryption at SIP boundary
- SIP trunks from multiple carriers: Each trunk treated as a separate protection zone

The AI discovery phase maps your entire topology before any encryption is activated.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q2.5', 'We run TN3270e sessions to an IBM mainframe. Does QBITEL protect mainframe traffic?',
        'Yes — TN3270e and TN5250 session protection is a core BPO module capability.',
        '''TN3270e terminal emulation is one of the highest-risk unprotected protocols in BPO. Agent sessions carry full customer records — account numbers, SSNs, credit card data — over unencrypted sessions. QBITEL provides:
- PQC tunnel wrapping: ML-KEM-768 + AES-256-GCM on all TN3270e/TN5250 sessions
- Session monitoring: Detection of unusual data access patterns (bulk lookups, off-hours access)
- Screen field detection: Identify PII fields for masking and audit trail purposes
- Session audit trail: Full per-agent, per-session record for compliance

No changes required to the mainframe, CICS applications, or terminal emulator software.'''
    ), sp(8)]))

    story.append(PageBreak())

    # ── SECTION 3: SECURITY & RISK ─────────────────────────────────────────────
    story.append(SectionHeader('3', 'Security & Risk Questions'))
    story.append(sp(6))
    story.append(AudienceBadge(['CISO', 'Security Architect', 'SOC Manager']))
    story.append(sp(10))

    story.append(KeepTogether([make_qa(
        'Q3.1', 'We already have a SIEM and SOAR. How does QBITEL fit in?',
        'QBITEL integrates via CEF/syslog and webhooks — it adds BPO-specific context your SIEM cannot generate.',
        '''QBITEL is a specialized sensor and response layer — not a replacement for your SIEM or SOAR.
Integration options:
- SIEM: CEF/syslog output to Splunk, QRadar, Microsoft Sentinel, ArcSight. All events include call_id, agent_id, tenant_id, trunk_id, fraud_type.
- SOAR: Webhook triggers to Palo Alto XSOAR, Splunk SOAR, ServiceNow SecOps.
What your SIEM cannot do that QBITEL provides:
- Correlate security events with call-level data
- Detect SIP-specific attacks (INVITE flooding, toll fraud, DTMF interception)
- Enforce PCI-DSS DTMF masking at the protocol layer
- Provide per-tenant compliance isolation for multi-tenant environments'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q3.2', 'How does QBITEL make autonomous decisions? What prevents wrong actions?',
        'Four controls: confidence thresholds, risk classification, blast radius limits, and human override.',
        '''Decision matrix: High confidence + low risk = auto-execute. High confidence + high risk = escalate. Low confidence = always escalate.

Safety constraints hardcoded into the system:
- Blast radius limit: No autonomous action affects more than 10 systems without human approval
- Production protection: Production-critical systems require human approval regardless of confidence
- Rollback window: All automated actions reversible within 60 minutes with a single click
- Emergency stop: Physical and software-based freeze of all autonomous actions
- Full audit trail: LLM reasoning chain, confidence score, evidence, and action logged for every automated decision

Your SOC can tune confidence thresholds per action type or set any category to alert-only mode.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q3.3', 'Is our call data going to OpenAI or Anthropic?',
        'On-premise by default — Ollama running Llama 3 on your infrastructure. No call data ever leaves your network.',
        '''QBITEL runs entirely on-premise using Ollama as the LLM serving layer. Supported models:
- Llama 3.2 (8B for fast response, 70B for complex analysis)
- Mixtral 8x7B, Qwen 2.5, Phi-3

The on-premise LLM accesses only anonymized security event data — it does not process raw call audio, customer PII, or cardholder data. Event data never leaves the QBITEL Engine VM.

Optional cloud LLM (Claude API) is available for customers who require advanced reasoning and have data sovereignty approvals. This is opt-in, never the default.

For air-gapped environments: Fully disconnected deployment supported — no internet after initial setup.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q3.4', 'How does QBITEL handle zero-day SIP vulnerabilities?',
        'AI detects behavioral anomalies independent of signatures — zero-days are flagged by deviation from learned baseline.',
        '''QBITEL uses behavioral anomaly detection, making it effective against zero-days:
- Grammar-based validation: QBITEL has learned the exact structure of SIP messages on your network. Messages that deviate are flagged even if the attack pattern is new.
- Volume anomaly detection: Sudden spikes in INVITE rates, unusual destination patterns, or off-hours volumes detected without signature matching.
- State machine enforcement: SIP call state transitions are enforced. Out-of-sequence messages trigger immediate analysis.
- Threat intelligence correlation: 200+ country premium-rate prefix database updated continuously and correlated against all outbound call destinations.'''
    ), sp(8)]))

    story.append(PageBreak())

    # ── SECTION 4: COMPLIANCE ──────────────────────────────────────────────────
    story.append(SectionHeader('4', 'Compliance & Legal Questions'))
    story.append(sp(6))
    story.append(AudienceBadge(['Chief Compliance Officer', 'DPO', 'Legal Counsel']))
    story.append(sp(10))

    story.append(KeepTogether([make_qa(
        'Q4.1', "We're undergoing a PCI-DSS 4.0 audit next quarter. Can QBITEL help us pass?",
        'Yes — QBITEL directly addresses Requirements 3, 4, 7, 8, and 12. We generate audit-ready evidence packages.',
        '''QBITEL maps to PCI-DSS 4.0 requirements for voice channel environments:
- Req 3 (Protect stored CHD): ML-KEM-1024 recording encryption; DTMF data never stored in recordings
- Req 4 (Protect CHD in transit): PQC-TLS for SIP; SRTP-PQC for RTP; quantum-safe tunnels
- Req 7 (Restrict access): Agent screen masking; per-role access controls
- Req 8 (Identify and authenticate): MFA enforcement; session validation
- Req 12 (Support security with policies): Automated policy enforcement and evidence logging

DTMF masking events are logged with timestamp, agent_id, call_id, and masking mode — directly usable as audit evidence. We recommend deploying at least 60 days before your audit to establish evidence baseline.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q4.2', 'We serve EU clients (GDPR) and US financial clients (PCI-DSS, SOX). How do you handle conflicts?',
        'Per-tenant compliance policies — each client governed by their specific framework with full isolation.',
        '''Multi-framework compliance in a multi-tenant BPO is QBITEL specific design target:
- Per-tenant policy engine: Each client gets a separate compliance policy enforced independently
- Per-tenant encryption keys: Cryptographic isolation — compromise of one tenant has zero impact on others
- Conflicting requirements: Where GDPR (right to erasure) conflicts with SOX (7-year retention), QBITEL applies the client-specific contracted policy
- Independent reporting: Each client receives a compliance report for their specific framework — no cross-tenant data in any report

We recommend involving your DPO and client contract leads in the policy configuration phase.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q4.3', 'We need to demonstrate HIPAA compliance for a healthcare BPO contract.',
        'QBITEL generates a HIPAA Technical Safeguards evidence package covering all 45 CFR § 164.312 requirements.',
        '''QBITEL covers HIPAA Technical Safeguards for healthcare BPO:
- Access Control (§164.312(a)): Agent session logs, role-based access enforcement
- Audit Controls (§164.312(b)): Immutable per-call, per-agent, per-action audit trail
- Integrity (§164.312(c)): Cryptographic integrity verification on all PHI-adjacent recordings
- Transmission Security (§164.312(e)): PQC encryption on all voice, terminal, and data transmissions

QBITEL also supports HIPAA Minimum Necessary by flagging agents who access patient records beyond assigned scope. We provide a HIPAA Technical Safeguards attestation document and sample BAA language.'''
    ), sp(8)]))

    story.append(PageBreak())

    # ── SECTION 5: OPERATIONS ──────────────────────────────────────────────────
    story.append(SectionHeader('5', 'Operations Questions'))
    story.append(sp(6))
    story.append(AudienceBadge(['Contact Center Director', 'VP Operations', 'WFM Lead']))
    story.append(sp(10))

    story.append(KeepTogether([make_qa(
        'Q5.1', "Will QBITEL affect our agents? We can't retrain 5,000 people.",
        'Zero change for agents on voice protection. Optional DLP controls are policy-configured — agents work exactly as before.',
        '''Agent experience impact by component:
- Voice PQC encryption: Zero — transparent at network layer
- DTMF masking: Zero for agent; caller hears normal tones
- Toll fraud prevention: Zero unless their call is blocked for fraud
- Recording encryption: Zero — recording interface unchanged
- TN3270e session protection: Zero — terminal emulator unchanged
- Screen watermarking: Small corner watermark if configured; invisible option available
- Clipboard DLP: Cannot paste PII to unauthorized apps — may affect ~2% of agent workflows
- USB blocking: USB drives blocked — agents use approved file transfer

For DLP controls that affect workflow, we provide a phased rollout guide and a 30-minute agent briefing template. No full training program required.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q5.2', 'We run 24/7 operations. What is the impact of deployment on live operations?',
        'Zero downtime deployment. The network tap does not interrupt traffic even during business hours.',
        '''QBITEL deployment architecture is designed for 24/7 contact center environments:
- Network tap placement: Passive tap (SPAN port) mirrors traffic — does not interrupt it
- Discovery phase: Entirely passive, can run during live operations indefinitely
- Encryption activation: Done per-trunk on a schedule you control
- Rollback: Encryption disabled for a specific trunk in under 60 seconds without affecting others
- Emergency bypass: Hardware bypass removes QBITEL from network path in under 5 seconds

Typical deployment schedule for a 5,000-seat BPO across 3 sites: Day 1 taps all sites (passive). Days 2–5 discovery. Week 2 review and approve. Weeks 2–3 activate one trunk per site with 48-hour monitoring. Week 3–4 full activation.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q5.3', 'We have remote agents in India, Philippines, and Eastern Europe.',
        'The remote agent tunnel is location-agnostic. Geo-fencing can restrict or allow specific countries per policy.',
        '''QBITEL remote agent security is designed for globally distributed BPO workforces:
- Tunnel technology: ML-KEM-768 + AES-256-GCM tunnels optimized for high-latency connections (tested at 150ms+ base latency without quality degradation)
- Geo-fencing: Allow specific countries only; alert on unexpected location changes; client-specific country restrictions
- Home network assessment: WiFi assessed for WPA3 compliance. Agents on WEP or open networks are blocked.
- ISP-level controls: Can restrict to approved ISP ranges for high-security clients

For India and Philippines specifically — the two largest BPO locations globally — we have reference deployments with configuration guides for common ISP environments (Jio, Airtel, PLDT, Globe).'''
    ), sp(8)]))

    story.append(PageBreak())

    # ── SECTION 6: PROCUREMENT ─────────────────────────────────────────────────
    story.append(SectionHeader('6', 'Procurement & Vendor Questions'))
    story.append(sp(6))
    story.append(AudienceBadge(['Procurement', 'Vendor Management', 'Legal']))
    story.append(sp(10))

    story.append(KeepTogether([make_qa(
        'Q6.1', "How do we know QBITEL will still be around in 5 years?",
        'Source code escrow for enterprise contracts. Open NIST standards. Full data portability. Contractual exit assistance.',
        '''Legitimate concern for any vendor. How we address it:
- Source code escrow: Enterprise and Global tier contracts include escrow with a reputable provider. If QBITEL ceases operations, you receive source code to operate the platform independently.
- Open standards: QBITEL uses NIST-standardized algorithms (ML-KEM, ML-DSA) and open protocols. You are never locked into our decryption capability.
- Financial disclosure: Available under NDA for enterprise procurement processes.
- Contractual protections: Enterprise contracts include SLA guarantees, data portability provisions, and exit assistance clauses.
- Customer references: Available for verification calls in your industry segment.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q6.2', 'We need to do a vendor security assessment. What do you provide?',
        'SOC 2 Type II report, pen test summary, SBOM, threat model, and we support customer-led penetration testing.',
        '''QBITEL supports enterprise vendor security assessment with:
- SOC 2 Type II report: Available under NDA — Security, Availability, and Confidentiality trust service criteria
- Penetration test report: Annual third-party pen test summary (full report under NDA)
- Software Bill of Materials (SBOM): Complete SBOM for vulnerability tracking
- Architecture threat model: Formal threat model with all system components and mitigations
- CVE response policy: Critical CVEs patched within 24 hours, high within 72 hours
- Customer-led pen testing: Supported with coordination to avoid false positive autonomous responses

For vendor questionnaires (SIG, CAIQ, custom), our vendor security team responds within 5 business days.'''
    ), sp(8)]))

    story.append(PageBreak())

    # ── SECTION 7: HARD OBJECTIONS ─────────────────────────────────────────────
    story.append(SectionHeader('7', 'Hard Objections & How to Handle Them',
                               'The toughest pushback — with scripted response approaches'))
    story.append(sp(6))
    story.append(sp(10))

    objections = [
        (
            'O7.1',
            '"We already have Palo Alto / Fortinet / CrowdStrike. Why another product?"',
            '''Acknowledge their investment, then expose the gap. Say: "Your existing tools do excellent work protecting what they know about — endpoints, known malware, network perimeter threats. None of them were designed for what happens inside a contact center at the protocol layer."

Ask your team: Can Palo Alto tell you whether a DTMF tone in a call recording is a card number? Can CrowdStrike detect when an agent is reading card numbers aloud? Can Fortinet identify IRSF fraud in SIP CDRs within 3 calls? These are BPO-specific problems that require BPO-specific solutions.

We are not a replacement for your existing stack — we are the layer that covers the 30% of your risk that your existing stack cannot see.'''
        ),
        (
            'O7.2',
            '"We don\'t want to add complexity. We\'re already managing too many vendors."',
            '''Frame QBITEL as a consolidation play, not an addition. Ask: "How many separate tools are you currently using to address DTMF compliance, toll fraud monitoring, remote agent security, and compliance reporting?"

Most BPOs have 3–5 point solutions for these problems, or manual processes. QBITEL replaces all of those with a single platform. For many customers, deploying QBITEL allows them to retire their separate toll fraud monitoring tool, their manual PCI audit prep process, and their VPN infrastructure for remote agents.

The net result is usually fewer vendors, not more.'''
        ),
        (
            'O7.3',
            '"The quantum threat is overstated. We\'ll deal with it when it\'s real."',
            '''Shift the frame from future threat to current attack. Say: "I agree quantum computers aren't breaking encryption today. But the attack that's happening right now is the harvest."

Here is the question that matters: Are your call recordings from this year covered by a 7-year SOX retention policy? If yes, those recordings will still exist in 2031. NIST's estimate for cryptographically relevant quantum computers is 2030–2035.

The recordings you capture today are the ones at risk — not recordings from some future date after you've had time to prepare. Waiting means you're already too late for the data you're generating right now.'''
        ),
        (
            'O7.4',
            '"We tried a security product like this before and it broke our voice quality."',
            '''Acknowledge the trauma and offer proof-first. Say: "That is a fair and important concern. Voice quality is non-negotiable in a contact center — I completely understand why that experience has made you cautious."

Our Discovery Assessment phase is entirely passive — no traffic modification, zero risk to voice quality. We start with a single, low-risk trunk in a test environment. We ask you to include MOS score measurement as a specific PoC metric. If we degrade your MOS scores at all, you stop and pay nothing further. We are confident enough in our <2ms overhead guarantee to put that in writing.'''
        ),
        (
            'O7.5',
            '"We\'re mid-migration to Genesys Cloud / Amazon Connect. We\'ll evaluate security after."',
            '''Position QBITEL as migration insurance. Say: "Migrations are exactly when BPOs are most vulnerable. During transition, you have both old and new infrastructure running simultaneously — a complex, partially-monitored environment that attackers love."

QBITEL deploys protocol-agnostically. It protects your legacy Avaya traffic today and your new Genesys Cloud traffic tomorrow without two separate deployments. You arrive at your new platform already protected rather than having to bolt security on afterward. We have worked with several BPOs through platform migrations specifically for this reason.'''
        ),
        (
            'O7.6',
            '"We need board approval for security spend over $X. This will take 6 months."',
            '''Offer a path that starts without board approval. Say: "Our Discovery Assessment is at no cost and requires no procurement process. It gives you a written report on what is running on your network, what is unencrypted, and what your current fraud loss exposure looks like."

That report is exactly what accelerates board approval — because it quantifies a risk that currently has no number attached to it. A board can evaluate "approve $X to prevent a documented $Y annual loss" much faster than "approve $X for a security enhancement." Can we schedule the Discovery Assessment now as the input to your board presentation?'''
        ),
        (
            'O7.7',
            '"Your pricing is too high compared to [competitor]."',
            '''Reframe to total cost of ownership, not license cost. Ask them to add up: their current annual toll fraud losses, their PCI audit prep cost, the cost of their VPN infrastructure for remote agents, and the SOC analyst time spent on contact center alerts.

In our experience, the sum of those four numbers is typically 3–5x the annual QBITEL license cost. We are not comparing license to license — we are comparing total cost of protecting your contact center versus total cost of not protecting it. The Discovery Assessment will show you the specific numbers for your environment.'''
        ),
    ]

    for obj_id, objection, response in objections:
        story.append(KeepTogether([make_objection(obj_id, objection, response), sp(10)]))

    story.append(PageBreak())

    # ── SECTION 8: COMPETITIVE ─────────────────────────────────────────────────
    story.append(SectionHeader('8', 'Competitive Questions'))
    story.append(sp(6))
    story.append(AudienceBadge(['Evaluators', 'Technical Decision Makers']))
    story.append(sp(10))

    story.append(KeepTogether([make_qa(
        'Q8.1', "We're evaluating Pindrop for voice fraud. How is QBITEL different?",
        'Pindrop detects voice biometric fraud. QBITEL protects the protocol infrastructure. Complementary, not competing.',
        '''Pindrop answers: "Is this caller who they claim to be?" QBITEL answers: "Is the network, the agent, and the compliance posture secure?" These are different problem spaces.

QBITEL covers what Pindrop does not:
- SIP/RTP encryption (Pindrop does not encrypt voice channels)
- Toll fraud detection in CDR patterns — IRSF, PBX hacking (Pindrop does caller authentication, not trunk-level fraud)
- Agent desktop DLP (not in Pindrop scope)
- PCI-DSS DTMF masking (not in Pindrop scope)
- Post-quantum cryptography (not in Pindrop scope)
- Compliance reporting across 9 frameworks (not in Pindrop scope)

Many BPOs run both — Pindrop for caller authentication, QBITEL for infrastructure and compliance security.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q8.2', "Our parent company uses Cisco security everywhere. Can we just extend that?",
        "Cisco's stack does not include DTMF masking, toll fraud CDR analysis, TN3270e protection, or multi-tenant compliance isolation.",
        '''Cisco's portfolio (SecureX, Talos, Umbrella, Firepower) is strong for network perimeter and endpoint. For BPO-specific requirements, the gaps are significant:
- Cisco does not offer DTMF masking for call recordings (a PCI-DSS requirement specific to voice payment channels)
- Cisco Talos handles threat intelligence but does not analyze CDR patterns for toll fraud (IRSF, Wangiri, call pumping)
- Cisco does not protect TN3270e/TN5250 mainframe sessions specifically
- Cisco does not offer multi-tenant compliance isolation (separate PCI-DSS, HIPAA, SOC 2 per client)
- Cisco does not offer post-quantum cryptography for SIP/RTP as a production feature (as of 2026)

We deploy alongside Cisco infrastructure — QBITEL integrates with CUCM and Finesse and can feed events to Cisco XDR. Not either/or.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q8.3', "We have a solution bundled with our SIP trunks. Why pay separately?",
        "Carrier-bundled security is limited to carrier-visible threats on carrier-owned infrastructure. It doesn't reach inside your perimeter.",
        '''Carrier-bundled security is attractive on paper but limited because carriers can only protect what they own:
- Carrier sees your SIP trunks — not your internal PBX, TN3270e sessions, or agent desktops
- Carrier fraud detection operates on 72-hour billing cycles — fraud runs for days before notification
- Carrier cannot enforce DTMF masking inside your recording system
- Carrier cannot monitor agent behavior or DLP
- Carrier cannot generate PCI-DSS or HIPAA compliance evidence for auditors
- Carrier-bundled solutions have no incentive to block calls aggressively — it reduces their revenue

QBITEL detects toll fraud within 3 calls (seconds), not on the billing cycle. And it covers the 90% of your security posture that lives inside your perimeter.'''
    ), sp(8)]))

    story.append(PageBreak())

    # ── SECTION 9: POC ─────────────────────────────────────────────────────────
    story.append(SectionHeader('9', 'Proof of Concept & Pilot Questions'))
    story.append(sp(6))
    story.append(AudienceBadge(['Technical Evaluators', 'IT Director', 'Security Team']))
    story.append(sp(10))

    story.append(KeepTogether([make_qa(
        'Q9.1', 'What does a PoC look like and what do we need to commit?',
        '2 weeks, 1 network tap, 1 trunk for active testing. You provide a network engineer for 4 hours — we do everything else.',
        '''Standard QBITEL BPO Proof of Concept:
- Duration: 2 weeks (extendable to 30 days)
- Your commitment: Network engineer (~4 hours for tap placement); security/compliance stakeholder (2 hours); designation of 1 test trunk and 20–50 agent sessions
- What QBITEL delivers: Complete protocol map, toll fraud detection live on test trunk, DTMF masking demo, PCI-DSS evidence report, compliance dashboard, MOS score before/after, and estimated annual ROI

Success criteria are defined by you. We recommend: (a) zero MOS degradation, (b) at least one fraud pattern detected, (c) PCI evidence package generated, (d) deployment in <6 hours.

No charge for PoC. If you proceed to production, PoC costs are credited toward first-year license.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q9.2', 'Can we test with production traffic or does it need to be a lab environment?',
        'Production traffic preferred — that is where real fraud patterns live. Discovery phase is passive and safe from day one.',
        '''A lab environment will not give meaningful results for two critical use cases:
- Toll fraud detection: Fraud patterns only appear in production CDR data — you cannot simulate IRSF or Wangiri meaningfully in a lab.
- Protocol discovery accuracy: The AI needs to see your actual traffic to learn your environment. Lab traffic is typically too clean.

QBITEL's phased deployment is designed to be safe for production:
- Passive tap phase: Zero traffic modification — 100% safe from day one
- Protocol discovery: Read-only analysis of production traffic
- Active encryption: Applied to one designated test trunk only, leaving all others untouched

We recommend using production traffic with a single low-risk trunk (e.g., outbound dialer or internal test trunk) for the active phase.'''
    ), sp(8)]))

    story.append(PageBreak())

    # ── SECTION 10: POST-SALES ─────────────────────────────────────────────────
    story.append(SectionHeader('10', 'Post-Sales / Implementation Questions'))
    story.append(sp(6))
    story.append(AudienceBadge(['IT Operations', 'Contact Center Director', 'Procurement']))
    story.append(sp(10))

    story.append(KeepTogether([make_qa(
        'Q10.1', 'What does the support model look like after deployment?',
        '24/7 technical support for Enterprise and Global tiers, dedicated CSM for first 90 days, quarterly reviews ongoing.',
        '''Post-deployment support structure:
- Contact Center tier: Business hours (8/5), email + ticket, P1 response 4 hours
- Enterprise BPO tier: 24/7 phone + email + ticket, dedicated CSM for 90 days, P1 response 1 hour
- Global BPO tier: 24/7 phone + email + dedicated Slack, dedicated CSM ongoing, P1 response 30 minutes

P1 (Critical) events for BPO: Voice quality degradation attributable to QBITEL; autonomous response action affecting live call operations; false positive fraud block preventing legitimate calls.

First 90 days for all tiers: Weekly check-in calls; model tuning based on your baseline; false positive review and policy refinement; compliance report review and auditor preparation support.'''
    ), sp(8)]))

    story.append(KeepTogether([make_qa(
        'Q10.2', 'How do we keep QBITEL updated? What is the patch/upgrade process?',
        'Automatic for signatures (no downtime). Manual approval for engine updates. Zero-downtime rolling updates.',
        '''Two update categories:
**Automatic (no downtime, no approval required):**
- Toll fraud prefix database: Updated daily — new premium-rate numbers added globally
- Threat intelligence feeds: Updated every 4 hours
- ML model signature updates: Weekly, applied to inference layer without engine restart

**Manual (your approval, zero-downtime rolling update):**
- QBITEL Engine software: Quarterly feature releases
- PQC algorithm updates: Triggered by NIST updates or new vulnerabilities
- Protocol parser updates: For newly discovered protocol variants in your environment

For on-premise: Updates delivered as signed packages. Your team controls installation schedule. Rolling updates keep at least one node active in clustered deployments. LLM model updates (Ollama) are optional and evaluated by your team before applying.'''
    ), sp(8)]))

    # ── CLOSING ────────────────────────────────────────────────────────────────
    story.append(sp(16))
    # Closing info block
    closing_data = [
        [Paragraph('<b><font color="#008B9A">✉</font>  enterprise@qbitel.com</b>',
                   ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=9.5, textColor=WHITE_C)),
         Paragraph('<b><font color="#008B9A">⊕</font>  bridge.qbitel.com</b>',
                   ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=9.5, textColor=WHITE_C)),
         Paragraph('<b><font color="#008B9A">◉</font>  Schedule a discovery call</b>',
                   ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=9.5, textColor=WHITE_C))],
        [Paragraph('Sales & Enterprise Inquiries',
                   ParagraphStyle('cs', fontName='Helvetica', fontSize=8, textColor=TEAL)),
         Paragraph('Product & Technical Resources',
                   ParagraphStyle('cs', fontName='Helvetica', fontSize=8, textColor=TEAL)),
         Paragraph('Book via your account team',
                   ParagraphStyle('cs', fontName='Helvetica', fontSize=8, textColor=TEAL))],
    ]
    closing_tbl = Table(closing_data, colWidths=[CONTENT_W / 3] * 3)
    closing_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), NAVY),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, TEAL),
    ]))
    story.append(closing_tbl)
    story.append(sp(14))

    story.append(Paragraph(
        '<i>QBITEL Bridge — Because the Human API Deserves Quantum-Safe Protection.</i>',
        ParagraphStyle('closing', fontName='Helvetica-BoldOblique', fontSize=12,
                       textColor=NAVY, alignment=TA_CENTER, spaceAfter=6)
    ))
    story.append(Paragraph(
        'Internal Sales Enablement  |  Version 1.0  |  February 2026  |  Confidential — Internal Use Only',
        ParagraphStyle('ver', fontName='Helvetica', fontSize=8,
                       textColor=MID_GREY, alignment=TA_CENTER)
    ))

    doc.build(story)
    print(f'PDF saved: {out_path}')
    return out_path


if __name__ == '__main__':
    build_pdf()
