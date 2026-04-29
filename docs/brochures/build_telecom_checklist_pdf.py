"""
Build QBITEL Bridge Telecom Deployment Checklist - Professional PDF
10-phase deployment checklist for Telecommunications & 5G Networks
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

# Phase color palette
PHASE_COLORS = [
    HexColor('#0D1B3E'),  # 0: Pre-Engagement
    HexColor('#006B7A'),  # 1: Protocol Discovery
    HexColor('#005F6B'),  # 2: Infrastructure Readiness
    HexColor('#007A5E'),  # 3: Security Policy Config
    HexColor('#004F6B'),  # 4: Signaling Protection
    HexColor('#0D3B6E'),  # 5: 5G Core Security
    HexColor('#2E8B57'),  # 6: Fraud Detection
    HexColor('#1A5276'),  # 7: IoT Gateway
    HexColor('#5A6A7A'),  # 8: Monitoring
    HexColor('#6B5B00'),  # 9: Compliance Validation
    HexColor('#8B4513'),  # 10: Go-Live
]


class PhaseHeader(Flowable):
    def __init__(self, phase_num, title, duration, owner, color=None, width=None):
        super().__init__()
        self.phase_num = phase_num
        self.title = title
        self.duration = duration
        self.owner = owner
        self.color = color or NAVY
        self.w = width or CONTENT_W
        self.h = 52

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(self.color)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        # Phase badge
        c.setFillColor(GOLD)
        c.rect(0, 0, 52, self.h, fill=1, stroke=0)
        c.setFillColor(self.color)
        c.setFont('Helvetica-Bold', 9)
        c.drawCentredString(26, self.h - 18, 'PHASE')
        c.setFont('Helvetica-Bold', 20)
        c.drawCentredString(26, self.h - 36, str(self.phase_num))
        # Title
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 13)
        c.drawString(62, self.h - 22, self.title.upper())
        # Duration and owner tags
        c.setFillColor(TEAL)
        c.roundRect(62, 8, 90, 16, 3, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica', 8)
        c.drawString(68, 13, self.duration)
        c.setFillColor(GOLD)
        c.roundRect(160, 8, 110, 16, 3, fill=1, stroke=0)
        c.setFillColor(self.color)
        c.setFont('Helvetica-Bold', 8)
        c.drawString(166, 13, self.owner)
        # Right accent
        c.setFillColor(TEAL)
        c.rect(self.w - 4, 0, 4, self.h, fill=1, stroke=0)


class ChecklistItem(Flowable):
    def __init__(self, text, category='', critical=False, width=None):
        super().__init__()
        self.text = text
        self.category = category
        self.critical = critical
        self.w = width or CONTENT_W
        lines = max(1, len(text) // 90 + 1)
        self.h = lines * 13 + 22

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Background
        bg = GOLD_LIGHT if self.critical else WHITE_C
        c.setFillColor(bg)
        c.roundRect(0, 0, self.w, self.h, 3, fill=1, stroke=0)
        # Left indicator
        indicator_color = GOLD if self.critical else TEAL
        c.setFillColor(indicator_color)
        c.rect(0, 0, 4, self.h, fill=1, stroke=0)
        # Checkbox
        c.setStrokeColor(MID_GREY)
        c.setLineWidth(1)
        c.setFillColor(WHITE_C)
        c.roundRect(12, self.h / 2 - 7, 14, 14, 2, fill=1, stroke=1)
        # Critical star
        if self.critical:
            c.setFillColor(GOLD)
            c.setFont('Helvetica-Bold', 10)
            c.drawString(30, self.h / 2 - 5, '*')
        # Category badge
        if self.category:
            cat_w = c.stringWidth(self.category, 'Helvetica-Bold', 7) + 8
            c.setFillColor(TEAL_LIGHT if not self.critical else GOLD_LIGHT)
            c.roundRect(self.w - cat_w - 8, self.h / 2 - 8, cat_w, 16, 3, fill=1, stroke=0)
            c.setFillColor(TEAL_DARK if not self.critical else ORANGE)
            c.setFont('Helvetica-Bold', 7)
            c.drawString(self.w - cat_w - 4, self.h / 2 - 3, self.category)
        # Item text
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica-Bold' if self.critical else 'Helvetica', 9.5)
        words = self.text.split()
        line = ''
        y = self.h - 15
        x_start = 42
        cat_reserve = (c.stringWidth(self.category, 'Helvetica-Bold', 7) + 18) if self.category else 0
        for word in words:
            test = line + ' ' + word if line else word
            max_w = self.w - x_start - cat_reserve - 12
            if c.stringWidth(test, 'Helvetica-Bold' if self.critical else 'Helvetica', 9.5) < max_w:
                line = test
            else:
                c.drawString(x_start, y, line)
                y -= 13
                line = word
                x_start = 42
                cat_reserve = 0
        if line:
            c.drawString(x_start, y, line)


class DeliverableBox(Flowable):
    def __init__(self, items, title='Phase Deliverables', width=None):
        super().__init__()
        self.items = items
        self.title = title
        self.w = width or CONTENT_W
        self.h = 28 + len(items) * 16 + 8

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(LIGHT_BG)
        c.roundRect(0, 0, self.w, self.h, 4, fill=1, stroke=0)
        c.setFillColor(TEAL_DARK)
        c.roundRect(0, self.h - 24, self.w, 24, 4, fill=1, stroke=0)
        c.rect(0, self.h - 30, self.w, 6, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 9)
        c.drawString(10, self.h - 16, self.title.upper())
        y = self.h - 38
        for item in self.items:
            c.setFillColor(TEAL)
            c.circle(18, y + 4, 3, fill=1, stroke=0)
            c.setFillColor(DARK_TEXT)
            c.setFont('Helvetica', 9)
            c.drawString(28, y, item)
            y -= 16


class RACIRow(Flowable):
    def __init__(self, activity, responsible, accountable, consulted, informed, width=None):
        super().__init__()
        self.activity = activity
        self.responsible = responsible
        self.accountable = accountable
        self.consulted = consulted
        self.informed = informed
        self.w = width or CONTENT_W
        lines = max(1, len(activity) // 40 + 1)
        self.h = lines * 13 + 12

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        cols = [self.w * 0.38, self.w * 0.155, self.w * 0.155, self.w * 0.155, self.w * 0.155]
        x = 0
        data = [self.activity, self.responsible, self.accountable, self.consulted, self.informed]
        colors_map = {'R': GREEN, 'A': NAVY, 'C': TEAL, 'I': MID_GREY, '-': LIGHT_BG}
        for i, (col_w, val) in enumerate(zip(cols, data)):
            if i == 0:
                c.setFillColor(WHITE_C)
                c.rect(x, 0, col_w, self.h, fill=1, stroke=0)
                c.setFillColor(DARK_TEXT)
                c.setFont('Helvetica', 8.5)
                # Wrap text
                words = val.split()
                line = ''
                y = self.h - 12
                for word in words:
                    test = line + ' ' + word if line else word
                    if c.stringWidth(test, 'Helvetica', 8.5) < col_w - 10:
                        line = test
                    else:
                        c.drawString(x + 6, y, line)
                        y -= 12
                        line = word
                if line:
                    c.drawString(x + 6, y, line)
            else:
                bg = colors_map.get(val, LIGHT_BG)
                c.setFillColor(bg)
                c.rect(x, 0, col_w, self.h, fill=1, stroke=0)
                text_color = WHITE_C if val in ('R', 'A', 'C', 'I') else MID_GREY
                c.setFillColor(text_color)
                c.setFont('Helvetica-Bold', 10)
                c.drawCentredString(x + col_w / 2, self.h / 2 - 5, val)
            # Grid line
            c.setStrokeColor(MID_GREY)
            c.setLineWidth(0.3)
            c.line(x, 0, x, self.h)
            x += col_w
        c.line(x, 0, x, self.h)
        c.line(0, 0, self.w, 0)
        c.line(0, self.h, self.w, self.h)


def draw_cover(canvas, doc):
    canvas.saveState()
    w, h = PAGE_W, PAGE_H
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, h - 8, w, 8, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, h - 14, w * 0.5, 4, fill=1, stroke=0)
    # Decorative elements
    for i, (px, py, r, alpha) in enumerate([
        (w * 0.82, h * 0.6, 200, 1),
        (w * 0.82, h * 0.6, 150, 1),
        (w * 0.82, h * 0.6, 110, 0),
    ]):
        if alpha:
            canvas.setFillColor(HexColor('#1A2D5A') if i == 0 else HexColor('#0F2248'))
            canvas.circle(px, py, r, fill=1, stroke=0)
        else:
            canvas.setStrokeColor(TEAL)
            canvas.setLineWidth(2)
            canvas.circle(px, py, r, fill=0, stroke=1)
    # Phase count badge
    canvas.setFillColor(GOLD)
    canvas.circle(w * 0.82, h * 0.6, 70, fill=1, stroke=0)
    canvas.setFillColor(NAVY)
    canvas.setFont('Helvetica-Bold', 24)
    canvas.drawCentredString(w * 0.82, h * 0.6 + 6, '11')
    canvas.setFont('Helvetica', 9)
    canvas.drawCentredString(w * 0.82, h * 0.6 - 14, 'Phases')
    # Title block
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 28)
    canvas.drawString(0.75 * inch, h - 1.2 * inch, 'QBITEL BRIDGE')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica', 13)
    canvas.drawString(0.75 * inch, h - 1.55 * inch, 'Telecommunications & 5G Networks')
    canvas.setFillColor(GOLD)
    canvas.rect(0.75 * inch, h - 1.75 * inch, 3.5 * inch, 3, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 30)
    canvas.drawString(0.75 * inch, h - 2.55 * inch, 'DEPLOYMENT')
    canvas.setFont('Helvetica-Bold', 28)
    canvas.setFillColor(GOLD)
    canvas.drawString(0.75 * inch, h - 2.95 * inch, 'CHECKLIST')
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 12)
    canvas.drawString(0.75 * inch, h - 3.3 * inch, 'Phases 0-10: Pre-Engagement to Go-Live')
    canvas.setFillColor(HexColor('#A0B8CC'))
    canvas.setFont('Helvetica', 10)
    canvas.drawString(0.75 * inch, h - 3.6 * inch, 'SS7 Signaling  |  5G Core  |  IoT/mMTC  |  Fraud  |  Compliance')
    # Phase timeline visual
    phases = ['PRE', 'DISC', 'INFRA', 'POL', 'SIG', '5G', 'FRAUD', 'IOT', 'MON', 'COMP', 'GO']
    phase_colors_cover = [HexColor('#0D1B3E'), HexColor('#006B7A'), HexColor('#005F6B'),
                          HexColor('#007A5E'), HexColor('#004F6B'), HexColor('#0D3B6E'),
                          HexColor('#2E8B57'), HexColor('#1A5276'), HexColor('#5A6A7A'),
                          HexColor('#6B5B00'), HexColor('#8B4513')]
    box_w = (w - 1.5 * inch) / len(phases)
    for i, (ph, col) in enumerate(zip(phases, phase_colors_cover)):
        bx = 0.75 * inch + i * box_w
        by = h * 0.24
        canvas.setFillColor(col)
        canvas.roundRect(bx, by, box_w - 4, 32, 3, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.setFont('Helvetica-Bold', 7)
        canvas.drawCentredString(bx + (box_w - 4) / 2, by + 20, str(i))
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 7)
        canvas.drawCentredString(bx + (box_w - 4) / 2, by + 8, ph)
    # Footer
    canvas.setFillColor(HexColor('#060E20'))
    canvas.rect(0, 0, w, 0.7 * inch, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0.7 * inch, w, 2, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 9)
    canvas.drawString(0.75 * inch, 0.28 * inch, 'enterprise@qbitel.com')
    canvas.setFillColor(TEAL)
    canvas.drawString(2.5 * inch, 0.28 * inch, '|')
    canvas.setFillColor(WHITE_C)
    canvas.drawString(2.7 * inch, 0.28 * inch, 'https://bridge.qbitel.com')
    canvas.setFillColor(MID_GREY)
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(w - 0.75 * inch, 0.28 * inch, 'IMPLEMENTATION GUIDE -- CONFIDENTIAL')
    canvas.restoreState()


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 0.55 * inch, PAGE_W, 0.55 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, PAGE_H - 0.55 * inch - 3, PAGE_W, 3, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(MARGIN, PAGE_H - 0.35 * inch, 'QBITEL BRIDGE')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica', 9)
    canvas.drawString(MARGIN + 1.05 * inch, PAGE_H - 0.35 * inch,
                      'TELECOM DEPLOYMENT CHECKLIST')
    canvas.setFillColor(HexColor('#A0B8CC'))
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.35 * inch, 'enterprise@qbitel.com')
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, 0.55 * inch, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0.55 * inch, PAGE_W, 2, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 8)
    canvas.drawString(MARGIN, 0.22 * inch, 'QBITEL BRIDGE -- TELECOM DEPLOYMENT')
    canvas.setFillColor(MID_GREY)
    canvas.drawRightString(PAGE_W - MARGIN, 0.22 * inch, f'Page {doc.page} | Confidential')
    canvas.restoreState()


def make_styles():
    body = ParagraphStyle('body', fontName='Helvetica', fontSize=9.5,
                          textColor=DARK_TEXT, spaceAfter=6, leading=14, alignment=TA_JUSTIFY)
    intro = ParagraphStyle('intro', fontName='Helvetica', fontSize=10,
                           textColor=MID_GREY, spaceAfter=8, leading=15, alignment=TA_JUSTIFY)
    note = ParagraphStyle('note', fontName='Helvetica-Oblique', fontSize=8.5,
                          textColor=MID_GREY, spaceAfter=6, leading=13)
    critical_note = ParagraphStyle('crit', fontName='Helvetica-Bold', fontSize=8.5,
                                   textColor=ORANGE, spaceAfter=6)
    return body, intro, note, critical_note


def build_doc(output_path):
    doc = BaseDocTemplate(
        output_path, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN
    )
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H,
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
                        id='cover')
    inner_frame = Frame(MARGIN, 0.7 * inch, CONTENT_W, PAGE_H - MARGIN - 0.7 * inch,
                        id='inner')
    doc.addPageTemplates([
        PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover),
        PageTemplate(id='Inner', frames=[inner_frame], onPage=draw_page),
    ])
    story = []
    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())

    body, intro, note, crit_note = make_styles()

    # ─── Intro Overview Table ───────────────────────────────────────────────
    overview_data = [
        [Paragraph('<b>PHASE</b>', ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=8.5, textColor=WHITE_C)),
         Paragraph('<b>NAME</b>', ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=8.5, textColor=WHITE_C)),
         Paragraph('<b>DURATION</b>', ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=8.5, textColor=WHITE_C)),
         Paragraph('<b>OWNER</b>', ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=8.5, textColor=WHITE_C)),
         Paragraph('<b>ITEMS</b>', ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=8.5, textColor=WHITE_C))],
        ['0', 'Pre-Engagement', '1 week', 'QBITEL SE + Operator CISO', '8'],
        ['1', 'Protocol Discovery', '1-2 weeks', 'QBITEL SE + Network Ops', '9'],
        ['2', 'Infrastructure Readiness', '1-2 weeks', 'Operator Infra + QBITEL', '8'],
        ['3', 'Security Policy Configuration', '1 week', 'Operator Security + QBITEL', '8'],
        ['4', 'Signaling Protection', '2-3 weeks', 'Signaling Ops + QBITEL', '9'],
        ['5', '5G Core Security', '2 weeks', 'Core Network Ops + QBITEL', '9'],
        ['6', 'Fraud Detection Activation', '1-2 weeks', 'Fraud Team + QBITEL', '8'],
        ['7', 'IoT Gateway Protection', '1-2 weeks', 'IoT/Data Team + QBITEL', '7'],
        ['8', 'Monitoring & Alerting', '1 week', 'NOC + QBITEL', '7'],
        ['9', 'Compliance Validation', '1 week', 'Compliance + Legal', '8'],
        ['10', 'Go-Live & Handover', '3-5 days', 'Joint QBITEL + Operator', '6'],
    ]
    for i in range(1, len(overview_data)):
        overview_data[i] = [
            Paragraph(str(overview_data[i][0]), ParagraphStyle('c', fontName='Helvetica-Bold',
                       fontSize=12, textColor=GOLD, alignment=TA_CENTER)),
            Paragraph(str(overview_data[i][1]), ParagraphStyle('c2', fontName='Helvetica-Bold',
                       fontSize=8.5, textColor=DARK_TEXT)),
            Paragraph(str(overview_data[i][2]), ParagraphStyle('c3', fontName='Helvetica', fontSize=8.5)),
            Paragraph(str(overview_data[i][3]), ParagraphStyle('c4', fontName='Helvetica', fontSize=8)),
            Paragraph(str(overview_data[i][4]), ParagraphStyle('c5', fontName='Helvetica-Bold',
                       fontSize=9, textColor=TEAL, alignment=TA_CENTER)),
        ]
    ov_table = Table(overview_data, colWidths=[CONTENT_W * 0.07, CONTENT_W * 0.28,
                                               CONTENT_W * 0.15, CONTENT_W * 0.37, CONTENT_W * 0.13])
    ov_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('GRID', (0, 0), (-1, -1), 0.4, MID_GREY),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (0, -1), LIGHT_NAVY),
    ]))
    story.append(Paragraph('Deployment Overview', ParagraphStyle('title', fontName='Helvetica-Bold',
                  fontSize=16, textColor=NAVY, spaceAfter=8)))
    story.append(Paragraph(
        'This checklist guides the end-to-end deployment of QBITEL Bridge across a telecommunications '
        'operator network -- from initial pre-engagement assessment through to full go-live and operational '
        'handover. Each phase includes specific technical items, deliverables, and RACI assignments. '
        'Items marked with * are critical path -- they must be completed before the next phase begins.',
        intro))
    story.append(Paragraph('<b>* Critical items are marked in gold and must be signed off before phase gate.</b>',
                           crit_note))
    story.append(ov_table)
    story.append(Spacer(1, 14))
    story.append(PageBreak())

    # =========================================================================
    # PHASE 0: Pre-Engagement
    # =========================================================================
    story.append(PhaseHeader(0, 'Pre-Engagement', '1 week', 'QBITEL SE + Operator CISO',
                              color=PHASE_COLORS[0]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'The pre-engagement phase establishes the scope, objectives, and success criteria for the QBITEL '
        'Bridge deployment. No network changes occur in this phase. Output is a signed Statement of Work '
        'and a completed Threat Exposure Assessment Report.',
        intro))
    items_0 = [
        ('Complete network topology questionnaire -- RAN, core, interconnect, roaming, and cloud inventory', 'PLANNING', True),
        ('Identify all SS7 interconnect points -- E1/T1 and SIGTRAN/M3UA links to roaming partners and IPX hubs', 'SS7', True),
        ('Document existing SS7 signaling firewall vendor, version, and current rule policy scope', 'SS7', False),
        ('Map 5G deployment status -- NSA vs SA, NFs deployed, slices active, vendor ecosystem', '5G', True),
        ('Identify regulatory obligations -- NIS2, FCC reporting, GSMA FS.19 assessment deadlines', 'COMPLIANCE', True),
        ('Obtain fraud loss profile -- IRSF, SIM swap, bypass fraud -- 12 months of historical data', 'FRAUD', False),
        ('Define VIP subscriber categories requiring enhanced SS7 location protection', 'SS7', False),
        ('Execute NDA and initiate Statement of Work review and approval', 'ADMIN', True),
    ]
    for text, cat, critical in items_0:
        story.append(KeepTogether([ChecklistItem(text, cat, critical), Spacer(1, 4)]))
    story.append(Spacer(1, 6))
    story.append(DeliverableBox([
        'Signed NDA and Statement of Work',
        'Network Topology Map (RAN, Core, Interconnect)',
        'Threat Exposure Assessment Report (confidential)',
        'Regulatory Obligation Register',
        'Fraud Loss Profile -- 12 month baseline',
    ]))
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # =========================================================================
    # PHASE 1: Protocol Discovery
    # =========================================================================
    story.append(PhaseHeader(1, 'Protocol Discovery', '1-2 weeks', 'QBITEL SE + Network Ops',
                              color=PHASE_COLORS[1]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'Passive protocol discovery establishes a complete inventory of signaling traffic, protocol versions, '
        'and current security posture without any changes to live network traffic flows. QBITEL Bridge '
        'passive taps are installed and configured for monitoring-only mode.',
        intro))
    items_1 = [
        ('Install QBITEL Bridge passive taps on SS7/SIGTRAN links -- monitoring only, no inline enforcement', 'SS7', True),
        ('Run 72-hour SS7 MAP traffic capture and baseline analysis -- document normal roaming patterns', 'SS7', True),
        ('Enumerate all SS7 MAP message types in active use -- SRI, ATI, PSI, UL, CLR, ISD', 'SS7', False),
        ('Capture and analyze Diameter S6a/S6d/S9 traffic -- identify authentication vector exchange patterns', 'DIAMETER', True),
        ('Map 5G SBI interface traffic -- identify all NF-to-NF API call patterns and volumes', '5G', True),
        ('Enumerate active network slices (S-NSSAI) and document per-slice traffic profiles', '5G', False),
        ('Capture SIP trunk traffic -- identify IRSF-risk destinations and anomalous call patterns', 'SIP', False),
        ('Document all IoT APNs, NB-IoT/LTE-M device populations, and current security policies', 'IOT', False),
        ('Generate Protocol Inventory Report with quantum vulnerability risk scoring per interface', 'COMPLIANCE', True),
    ]
    for text, cat, critical in items_1:
        story.append(KeepTogether([ChecklistItem(text, cat, critical), Spacer(1, 4)]))
    story.append(Spacer(1, 6))
    story.append(DeliverableBox([
        'SS7 MAP Traffic Baseline Report',
        'Protocol Inventory -- all interfaces with quantum vulnerability scores',
        'Active Attack Campaign Report (detected during passive monitoring)',
        '5G Slice Inventory and Traffic Profile',
        'IoT APN and Device Population Report',
    ]))
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # =========================================================================
    # PHASE 2: Infrastructure Readiness
    # =========================================================================
    story.append(PhaseHeader(2, 'Infrastructure Readiness', '1-2 weeks', 'Operator Infra + QBITEL',
                              color=PHASE_COLORS[2]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'Infrastructure readiness prepares the hardware, HSM, network integration, and software environment '
        'for QBITEL Bridge deployment. No security policy changes are active at this stage.',
        intro))
    items_2 = [
        ('Provision QBITEL Bridge server hardware -- 2-socket x86 with AVX-512, minimum 64GB RAM per node', 'HARDWARE', True),
        ('Install and initialize Hardware Security Modules (HSM) -- Thales Luna or Entrust nShield', 'HSM', True),
        ('Perform HSM key ceremony -- establish PQC root key hierarchy under dual-control procedures', 'HSM', True),
        ('Configure HSM cluster replication for geographic redundancy -- minimum 2 sites', 'HSM', True),
        ('Deploy QBITEL Bridge software -- containerized (Kubernetes) or bare-metal per architecture decision', 'DEPLOY', False),
        ('Configure network taps / inline deployment ports for SS7 and SIP interfaces', 'NETWORK', False),
        ('Validate network management connectivity -- NETCONF/YANG for Nokia, RESTCONF for Ericsson', 'INTEGRATION', False),
        ('Configure QBITEL Bridge management console access -- RBAC, MFA, audit logging', 'SECURITY', True),
    ]
    for text, cat, critical in items_2:
        story.append(KeepTogether([ChecklistItem(text, cat, critical), Spacer(1, 4)]))
    story.append(Spacer(1, 6))
    story.append(DeliverableBox([
        'HSM Installation and Key Ceremony Report',
        'QBITEL Bridge Cluster Deployment Verification',
        'Network Integration Test Report',
        'Management Console Access and RBAC Configuration',
        'Infrastructure Readiness Sign-off (joint QBITEL + Operator)',
    ]))
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # =========================================================================
    # PHASE 3: Security Policy Configuration
    # =========================================================================
    story.append(PhaseHeader(3, 'Security Policy Configuration', '1 week', 'Operator Security + QBITEL',
                              color=PHASE_COLORS[3]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'Security policy configuration establishes the cryptographic policy framework, subscriber data '
        'classification, and protocol-specific security rules before any enforcement is activated. '
        'All policies are reviewed and approved by the operator security team before phase gate.',
        intro))
    items_3 = [
        ('Define subscriber data classification tiers -- standard, VIP, critical government, MVNO', 'POLICY', True),
        ('Configure PQC algorithm policy per network domain -- choose ML-KEM-768 vs 1024 per interface', 'CRYPTO', True),
        ('Define SS7 MAP filtering policy -- alert-only vs inline block, per attack category', 'SS7', True),
        ('Configure Diameter S6a roaming partner trust profiles -- whitelist legitimate partners', 'DIAMETER', True),
        ('Define 5G network slice security policies -- per S-NSSAI PQC key hierarchy and isolation rules', '5G', True),
        ('Configure SIP trunk IRSF blocking policy -- thresholds, automatic trunk suspension rules', 'SIP', False),
        ('Define IoT security profiles per APN -- NB-IoT, LTE-M, industrial IIoT enhanced profiles', 'IOT', False),
        ('Configure fraud detection sensitivity -- precision vs recall tradeoff per fraud category', 'FRAUD', False),
    ]
    for text, cat, critical in items_3:
        story.append(KeepTogether([ChecklistItem(text, cat, critical), Spacer(1, 4)]))
    story.append(Spacer(1, 6))
    story.append(DeliverableBox([
        'Security Policy Framework Document (operator-approved)',
        'PQC Algorithm Selection Matrix per Network Domain',
        'SS7/Diameter Filtering Policy Rulebook',
        '5G Slice Security Policy Configuration',
        'Fraud Detection Sensitivity Configuration',
    ]))
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # =========================================================================
    # PHASE 4: Signaling Protection
    # =========================================================================
    story.append(PhaseHeader(4, 'Signaling Protection', '2-3 weeks', 'Signaling Ops + QBITEL',
                              color=PHASE_COLORS[4]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'Signaling protection activates inline SS7 MAP filtering, Diameter security, and SIP trunk '
        'hardening in production. Activation follows a phased approach: alert-only for 72 hours, '
        'graduated enforcement, then full blocking on confirmed attack patterns.',
        intro))
    items_4 = [
        ('Activate SS7 MAP AnyTimeInterrogation (ATI) filtering -- alert-only mode for 72 hours', 'SS7', True),
        ('Review ATI alert queue -- validate detection accuracy, tune false positive threshold', 'SS7', True),
        ('Enable SS7 MAP ATI inline blocking for high-confidence attack signatures', 'SS7', True),
        ('Activate SS7 MAP SendRoutingInfo (SRI) anomaly detection -- roaming partner behavior profiling', 'SS7', False),
        ('Enable SS7 MAP ProvideSubscriberInfo (PSI) and UpdateLocation filtering', 'SS7', False),
        ('Activate PQC authentication vector wrapping on HLR/HSS MAP interfaces', 'SS7', True),
        ('Configure Diameter S6a Cancel Location Request (CLR) anomaly detection', 'DIAMETER', False),
        ('Enable Diameter Insert Subscriber Data (ISD) integrity verification', 'DIAMETER', False),
        ('Activate SIP trunk IRSF detection -- alert-only for 48 hours before enabling blocking', 'SIP', True),
    ]
    for text, cat, critical in items_4:
        story.append(KeepTogether([ChecklistItem(text, cat, critical), Spacer(1, 4)]))
    story.append(Spacer(1, 6))
    story.append(DeliverableBox([
        'SS7 MAP Filtering Activation Report -- 72hr alert-only results',
        'False Positive Analysis and Threshold Tuning Report',
        'Inline Blocking Activation Sign-off',
        'PQC Authentication Vector Wrapping Verification',
        'Diameter Security Activation Report',
    ]))
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # =========================================================================
    # PHASE 5: 5G Core Security
    # =========================================================================
    story.append(PhaseHeader(5, '5G Core Security', '2 weeks', 'Core Network Ops + QBITEL',
                              color=PHASE_COLORS[5]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        '5G core security activation deploys PQC-hybrid mTLS on all SBI interfaces, enforces slice '
        'cryptographic isolation, and activates NF integrity verification across AMF, SMF, UPF, '
        'AUSF, UDM, and NRF. Deployment follows one NF at a time to minimize risk.',
        intro))
    items_5 = [
        ('Deploy QBITEL Bridge sidecar proxies on AMF pods -- verify PQC hybrid mTLS on N1/N2', '5G', True),
        ('Activate SUPI/SUCI integrity verification on AMF -- alert on unencrypted SUPI exposure', '5G', True),
        ('Deploy sidecar proxies on SMF -- activate N4/PFCP session binding integrity verification', '5G', True),
        ('Enable GTP-TEID integrity monitoring on UPF -- detect tunnel redirect attempts', '5G', False),
        ('Deploy sidecar on AUSF -- activate PQC wrapping of HXRES* authentication vectors', '5G', True),
        ('Enable UDM subscriber credential quantum-hardening -- encrypt UDR subscriber data at rest', '5G', True),
        ('Activate NRF service registration integrity -- detect rogue NF registration attempts', '5G', False),
        ('Configure per-slice PQC key hierarchies -- validate slice isolation enforcement', '5G', True),
        ('Verify SEPP N32 interface PQC hybrid key exchange for inter-PLMN roaming security', '5G', False),
    ]
    for text, cat, critical in items_5:
        story.append(KeepTogether([ChecklistItem(text, cat, critical), Spacer(1, 4)]))
    story.append(Spacer(1, 6))
    story.append(DeliverableBox([
        '5G NF Sidecar Deployment Verification Report',
        'SBI mTLS PQC Key Exchange Validation',
        'Slice Isolation Test Results (cross-slice attack simulation)',
        'SUPI Protection Audit Report',
        'UDM/UDR Quantum-Hardening Verification',
    ]))
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # =========================================================================
    # PHASE 6: Fraud Detection Activation
    # =========================================================================
    story.append(PhaseHeader(6, 'Fraud Detection Activation', '1-2 weeks', 'Fraud Team + QBITEL',
                              color=PHASE_COLORS[6]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'Fraud detection activation initializes machine learning models with operator-specific CDR history, '
        'configures real-time IRSF blocking, and integrates with existing fraud management systems. '
        'Models are validated against known fraud events before blocking is enabled.',
        intro))
    items_6 = [
        ('Load 6-12 months of historical CDR data for ML model training and calibration', 'FRAUD', True),
        ('Validate ML model accuracy against known historical IRSF events -- target 99.9%+ detection rate', 'FRAUD', True),
        ('Configure IRSF number range intelligence feeds -- GSMA, CFCA, operator-specific IPRN lists', 'FRAUD', True),
        ('Activate real-time IRSF call blocking -- test on low-risk SIP trunks first', 'FRAUD', True),
        ('Enable wangiri one-ring fraud detection and automatic callback blocking', 'FRAUD', False),
        ('Configure SIM swap anomaly detection -- HLR/HSS update pattern monitoring', 'FRAUD', False),
        ('Enable bypass fraud / SIM box detection -- RF signature and call pattern analysis', 'FRAUD', False),
        ('Integrate fraud event feed with existing FMS (Subex/TEOCO/Syniverse) via REST API', 'INTEGRATION', True),
    ]
    for text, cat, critical in items_6:
        story.append(KeepTogether([ChecklistItem(text, cat, critical), Spacer(1, 4)]))
    story.append(Spacer(1, 6))
    story.append(DeliverableBox([
        'ML Model Training and Validation Report',
        'IRSF Detection Accuracy Benchmark Results',
        'Real-time Blocking Activation Report',
        'FMS Integration Test Confirmation',
        'First Month Fraud Prevention Impact Report',
    ]))
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # =========================================================================
    # PHASE 7: IoT Gateway Protection
    # =========================================================================
    story.append(PhaseHeader(7, 'IoT Gateway Protection', '1-2 weeks', 'IoT/Data Team + QBITEL',
                              color=PHASE_COLORS[7]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'IoT gateway protection activates network-layer PQC for NB-IoT, LTE-M, and eMTC device populations '
        'without requiring any firmware updates to constrained devices. Botnet detection and automatic '
        'quarantine are configured per device category.',
        intro))
    items_7 = [
        ('Configure IoT PQC gateway per NB-IoT and LTE-M APN -- verify device traffic forwarding', 'IOT', True),
        ('Activate device behavioral fingerprinting -- establish baseline per device type and APN', 'IOT', True),
        ('Enable IoT botnet C2 detection -- configure automatic quarantine APN for compromised devices', 'IOT', True),
        ('Configure smart metering security profile -- tamper evidence binding for meter readings', 'IOT', False),
        ('Enable GSMA IoT Security Guidelines compliance checks per device category', 'COMPLIANCE', False),
        ('Configure IIoT enhanced security profiles for critical infrastructure APNs', 'IOT', False),
        ('Verify eSIM/iSIM provisioning integration -- PQC key material in profile download', 'IOT', False),
    ]
    for text, cat, critical in items_7:
        story.append(KeepTogether([ChecklistItem(text, cat, critical), Spacer(1, 4)]))
    story.append(Spacer(1, 6))
    story.append(DeliverableBox([
        'IoT Gateway PQC Activation Report',
        'Device Behavioral Baseline Report',
        'Botnet Detection Validation Results',
        'IoT Security Coverage Report (% devices protected)',
    ]))
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # =========================================================================
    # PHASE 8: Monitoring & Alerting
    # =========================================================================
    story.append(PhaseHeader(8, 'Monitoring & Alerting', '1 week', 'NOC + QBITEL',
                              color=PHASE_COLORS[8]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'Monitoring and alerting configures carrier-grade NOC dashboards, regulatory reporting feeds, '
        'and integration with existing SIEM and OSS systems. All KPIs are validated against GSMA '
        'fraud reporting standards and operator-specific SLA requirements.',
        intro))
    items_8 = [
        ('Configure Prometheus/OpenTelemetry metrics export from all QBITEL Bridge nodes', 'MONITORING', True),
        ('Deploy pre-built Grafana dashboards -- SS7 attacks, fraud events, PQC ops, slice security', 'MONITORING', True),
        ('Configure GSMA fraud KPI reporting -- IRSF block rate, SS7 attack volume, location tracking blocks', 'COMPLIANCE', True),
        ('Integrate QBITEL Bridge event stream with SIEM (Splunk/Elastic/QRadar)', 'SIEM', False),
        ('Configure PagerDuty/OpsGenie alerting for critical security events', 'ALERTING', False),
        ('Enable NIS2 incident detection reporting -- configure 24/72-hour notification workflows', 'COMPLIANCE', True),
        ('Configure FCC SS7 monitoring monthly report automation', 'COMPLIANCE', False),
    ]
    for text, cat, critical in items_8:
        story.append(KeepTogether([ChecklistItem(text, cat, critical), Spacer(1, 4)]))
    story.append(Spacer(1, 6))
    story.append(DeliverableBox([
        'NOC Dashboard Deployment and Acceptance',
        'GSMA KPI Reporting Configuration Validation',
        'SIEM Integration Test Report',
        'Alerting Runbook (escalation procedures)',
    ]))
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # =========================================================================
    # PHASE 9: Compliance Validation
    # =========================================================================
    story.append(PhaseHeader(9, 'Compliance Validation', '1 week', 'Compliance + Legal',
                              color=PHASE_COLORS[9]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'Compliance validation generates formal evidence packages for all applicable regulatory frameworks. '
        'QBITEL Bridge automated report generation is configured and validated against each framework '
        'requirement. Output is a compliance evidence package ready for regulator submission.',
        intro))
    items_9 = [
        ('Generate 3GPP TS 33.501 compliance evidence -- SBI mTLS, SUPI protection, slice security controls', 'COMPLIANCE', True),
        ('Generate GSMA FS.19 quantum readiness evidence package -- algorithm inventory, PQC deployment proof', 'COMPLIANCE', True),
        ('Generate GSMA FS.11 SS7 monitoring compliance report -- attack detection and blocking statistics', 'COMPLIANCE', True),
        ('Validate NIS2 Directive compliance evidence -- security measures, incident notification workflows', 'COMPLIANCE', True),
        ('Generate FCC SS7 remediation evidence report for carrier regulatory filing', 'COMPLIANCE', False),
        ('Conduct NESAS security assurance evidence review with QBITEL Bridge security team', 'COMPLIANCE', False),
        ('Legal review of all compliance documentation before submission', 'LEGAL', True),
        ('Obtain operator CISO sign-off on compliance evidence package', 'ADMIN', True),
    ]
    for text, cat, critical in items_9:
        story.append(KeepTogether([ChecklistItem(text, cat, critical), Spacer(1, 4)]))
    story.append(Spacer(1, 6))
    story.append(DeliverableBox([
        '3GPP TS 33.501 Compliance Evidence Package',
        'GSMA FS.19 Quantum Readiness Certification Evidence',
        'GSMA FS.11 SS7 Security Compliance Report',
        'NIS2 Directive Security Measures Documentation',
        'CISO-Signed Compliance Evidence Package',
    ]))
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # =========================================================================
    # PHASE 10: Go-Live & Handover
    # =========================================================================
    story.append(PhaseHeader(10, 'Go-Live & Handover', '3-5 days', 'Joint QBITEL + Operator',
                              color=PHASE_COLORS[10]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'Go-Live confirms full production readiness and transfers day-to-day operational responsibility '
        'to the operator NOC and security teams, supported by QBITEL Bridge 24/7 monitoring and '
        'ongoing threat intelligence services.',
        intro))
    items_10 = [
        ('Conduct final end-to-end SLA validation -- PQC ops/sec, SS7 block latency, availability', 'SLA', True),
        ('Execute joint penetration test simulation -- SS7 location tracking, 5G slice bypass attempt', 'SECURITY', True),
        ('Complete operator NOC runbook training -- alerting procedures, escalation paths, QBITEL Bridge controls', 'TRAINING', True),
        ('Transfer primary operational responsibility to operator NOC with QBITEL Bridge 24/7 monitoring backup', 'HANDOVER', True),
        ('Schedule quarterly security review calendar -- threat intelligence briefings, model updates', 'ONGOING', False),
        ('Issue Go-Live Certificate and final project documentation package', 'ADMIN', True),
    ]
    for text, cat, critical in items_10:
        story.append(KeepTogether([ChecklistItem(text, cat, critical), Spacer(1, 4)]))
    story.append(Spacer(1, 6))
    story.append(DeliverableBox([
        'Final SLA Validation Report',
        'Penetration Test Simulation Results',
        'Operator NOC Runbook (QBITEL Bridge operations)',
        'Go-Live Certificate signed by both parties',
        'Project Closure and Lessons Learned Report',
    ]))
    story.append(Spacer(1, 14))

    # ─── RACI Matrix ────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph('RACI Responsibility Matrix', ParagraphStyle('rtitle', fontName='Helvetica-Bold',
                  fontSize=14, textColor=NAVY, spaceAfter=6)))
    story.append(Paragraph(
        'R = Responsible (does the work)  |  A = Accountable (signs off)  |  '
        'C = Consulted (provides input)  |  I = Informed (kept updated)',
        ParagraphStyle('rlegend', fontName='Helvetica', fontSize=9, textColor=MID_GREY, spaceAfter=10)))

    # RACI header
    raci_header = [
        Paragraph('<b>Activity</b>', ParagraphStyle('rh', fontName='Helvetica-Bold', fontSize=8.5, textColor=WHITE_C)),
        Paragraph('<b>QBITEL</b>', ParagraphStyle('rh', fontName='Helvetica-Bold', fontSize=8.5, textColor=WHITE_C, alignment=TA_CENTER)),
        Paragraph('<b>Operator NOC</b>', ParagraphStyle('rh', fontName='Helvetica-Bold', fontSize=8.5, textColor=WHITE_C, alignment=TA_CENTER)),
        Paragraph('<b>Security Team</b>', ParagraphStyle('rh', fontName='Helvetica-Bold', fontSize=8.5, textColor=WHITE_C, alignment=TA_CENTER)),
        Paragraph('<b>Compliance</b>', ParagraphStyle('rh', fontName='Helvetica-Bold', fontSize=8.5, textColor=WHITE_C, alignment=TA_CENTER)),
    ]
    raci_rows = [
        ('Network topology and protocol discovery', 'R', 'C', 'I', 'I'),
        ('HSM key ceremony', 'R', 'I', 'A', 'I'),
        ('SS7 MAP filtering policy definition', 'C', 'I', 'A', 'C'),
        ('SS7 inline blocking activation', 'R', 'C', 'A', 'I'),
        ('5G SBI sidecar proxy deployment', 'R', 'C', 'I', 'I'),
        ('Slice security policy configuration', 'C', 'I', 'A', 'C'),
        ('Fraud detection ML model training', 'R', 'I', 'C', 'I'),
        ('FMS integration configuration', 'R', 'C', 'C', 'I'),
        ('IoT gateway PQC activation', 'R', 'C', 'I', 'I'),
        ('NOC dashboard configuration', 'R', 'A', 'C', 'I'),
        ('Regulatory compliance evidence generation', 'R', 'I', 'C', 'A'),
        ('Go-Live acceptance sign-off', 'C', 'I', 'A', 'C'),
        ('Ongoing 24/7 monitoring', 'R', 'C', 'I', 'I'),
        ('Quarterly security reviews', 'R', 'I', 'A', 'C'),
    ]
    raci_data = [raci_header]
    for row in raci_rows:
        raci_data.append([
            Paragraph(row[0], ParagraphStyle('rc', fontName='Helvetica', fontSize=8.5, textColor=DARK_TEXT)),
            Paragraph(row[1], ParagraphStyle('rv', fontName='Helvetica-Bold', fontSize=11,
                       textColor=GREEN if row[1]=='R' else NAVY if row[1]=='A' else TEAL if row[1]=='C' else MID_GREY,
                       alignment=TA_CENTER)),
            Paragraph(row[2], ParagraphStyle('rv2', fontName='Helvetica-Bold', fontSize=11,
                       textColor=GREEN if row[2]=='R' else NAVY if row[2]=='A' else TEAL if row[2]=='C' else MID_GREY,
                       alignment=TA_CENTER)),
            Paragraph(row[3], ParagraphStyle('rv3', fontName='Helvetica-Bold', fontSize=11,
                       textColor=GREEN if row[3]=='R' else NAVY if row[3]=='A' else TEAL if row[3]=='C' else MID_GREY,
                       alignment=TA_CENTER)),
            Paragraph(row[4], ParagraphStyle('rv4', fontName='Helvetica-Bold', fontSize=11,
                       textColor=GREEN if row[4]=='R' else NAVY if row[4]=='A' else TEAL if row[4]=='C' else MID_GREY,
                       alignment=TA_CENTER)),
        ])
    raci_table = Table(raci_data, colWidths=[CONTENT_W * 0.40, CONTENT_W * 0.15,
                                              CONTENT_W * 0.15, CONTENT_W * 0.15, CONTENT_W * 0.15])
    raci_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('GRID', (0, 0), (-1, -1), 0.4, MID_GREY),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(raci_table)
    story.append(Spacer(1, 14))

    # ─── Contact footer ─────────────────────────────────────────────────────
    contact_data = [[
        Paragraph('<b>Enterprise Sales</b><br/>enterprise@qbitel.com',
                  ParagraphStyle('cf', fontName='Helvetica', fontSize=10,
                                 textColor=DARK_TEXT, alignment=TA_CENTER)),
        Paragraph('<b>Technical Pre-Sales</b><br/>https://bridge.qbitel.com',
                  ParagraphStyle('cf2', fontName='Helvetica', fontSize=10,
                                 textColor=DARK_TEXT, alignment=TA_CENTER)),
        Paragraph('<b>Implementation Support</b><br/>24/7 deployment assistance',
                  ParagraphStyle('cf3', fontName='Helvetica', fontSize=10,
                                 textColor=DARK_TEXT, alignment=TA_CENTER)),
    ]]
    ct = Table(contact_data, colWidths=[CONTENT_W / 3] * 3)
    ct.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, MID_GREY),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(ct)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        'Securing the networks that secure the world.',
        ParagraphStyle('tagline', fontName='Helvetica-Oblique', fontSize=11,
                       textColor=TEAL, alignment=TA_CENTER)))

    doc.build(story)
    print(f'Checklist PDF written to {output_path}')


if __name__ == '__main__':
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'QBITEL_Telecom_Deployment_Checklist.pdf')
    build_doc(out)
