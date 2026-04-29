"""
Build QBITEL Bridge BPO Marketing Pitch - Professional PDF
Uses ReportLab for full layout/design control.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import Color, HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate

# ─── Brand Colors ─────────────────────────────────────────────────────────────
NAVY       = HexColor('#0D1B3E')
TEAL       = HexColor('#008B9A')
TEAL_DARK  = HexColor('#006B7A')
GOLD       = HexColor('#F0A500')
LIGHT_BG   = HexColor('#F4F7FA')
MID_GREY   = HexColor('#5A6A7A')
DARK_TEXT  = HexColor('#1A1A2E')
TABLE_ALT  = HexColor('#EAF3F8')
WHITE_C    = HexColor('#FFFFFF')
LIGHT_NAVY = HexColor('#1A2D5A')

PAGE_W, PAGE_H = letter
MARGIN = 0.85 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


# ─── Custom Flowables ─────────────────────────────────────────────────────────

class ColorBar(Flowable):
    """A full-width colored horizontal bar."""
    def __init__(self, height, color, width=None):
        super().__init__()
        self.bar_height = height
        self.color = color
        self.bar_width = width or CONTENT_W

    def wrap(self, avail_w, avail_h):
        return self.bar_width, self.bar_height

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.bar_width, self.bar_height, fill=1, stroke=0)


class SectionHeader(Flowable):
    """Navy section header with gold accent left bar."""
    def __init__(self, title, subtitle=None, width=None):
        super().__init__()
        self.title = title
        self.subtitle = subtitle
        self.w = width or CONTENT_W
        self.h = 52 if subtitle else 40

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Background
        c.setFillColor(NAVY)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        # Gold left accent
        c.setFillColor(GOLD)
        c.rect(0, 0, 6, self.h, fill=1, stroke=0)
        # Teal right accent
        c.setFillColor(TEAL)
        c.rect(self.w - 4, 0, 4, self.h, fill=1, stroke=0)
        # Title
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 13)
        title_y = self.h - 22 if self.subtitle else (self.h - 16) / 2 + 4
        c.drawString(16, title_y, self.title.upper())
        if self.subtitle:
            c.setFillColor(TEAL)
            c.setFont('Helvetica-Oblique', 9)
            c.drawString(16, 8, self.subtitle)


class StatBlock(Flowable):
    """A row of stat boxes."""
    def __init__(self, stats, width=None, height=70, bg=NAVY):
        super().__init__()
        self.stats = stats  # list of (big, small)
        self.w = width or CONTENT_W
        self.h = height
        self.bg = bg

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        n = len(self.stats)
        box_w = self.w / n
        for i, (big, small) in enumerate(self.stats):
            x = i * box_w
            bg = TEAL if i % 2 == 0 else NAVY
            c.setFillColor(bg)
            c.rect(x, 0, box_w, self.h, fill=1, stroke=0)
            # Big number
            c.setFillColor(GOLD if bg == NAVY else WHITE_C)
            c.setFont('Helvetica-Bold', 22)
            text_w = c.stringWidth(big, 'Helvetica-Bold', 22)
            c.drawString(x + (box_w - text_w) / 2, self.h - 34, big)
            # Small label
            c.setFillColor(WHITE_C)
            c.setFont('Helvetica', 8)
            for j, line in enumerate(small.split('\n')):
                lw = c.stringWidth(line, 'Helvetica', 8)
                c.drawString(x + (box_w - lw) / 2, self.h - 50 - j * 11, line)


class ScenarioBox(Flowable):
    """Scenario card with teal header band."""
    def __init__(self, label, title, body_lines, width=None):
        super().__init__()
        self.label = label
        self.title = title
        self.body_lines = body_lines
        self.w = width or CONTENT_W
        self.line_h = 13
        self.body_h = len(body_lines) * self.line_h + 20
        self.h = 34 + self.body_h

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Body background
        c.setFillColor(LIGHT_BG)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        # Header band
        c.setFillColor(TEAL_DARK)
        c.rect(0, self.h - 34, self.w, 34, fill=1, stroke=0)
        # Gold label
        c.setFillColor(GOLD)
        c.setFont('Helvetica-Bold', 8)
        c.drawString(10, self.h - 16, self.label + '  ›')
        # Title
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 11)
        label_w = c.stringWidth(self.label + '  ›', 'Helvetica-Bold', 8)
        c.drawString(10 + label_w + 8, self.h - 14, self.title)
        # Body text
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 9)
        y = self.h - 34 - 16
        for line in self.body_lines:
            c.drawString(12, y, line)
            y -= self.line_h
        # Left accent bar
        c.setFillColor(TEAL)
        c.rect(0, 0, 4, self.h - 34, fill=1, stroke=0)


class CalloutBox(Flowable):
    """Highlighted callout quote box."""
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


# ─── Page Template ────────────────────────────────────────────────────────────

def draw_page(canvas, doc):
    """Header and footer for every page."""
    canvas.saveState()

    # Top header strip
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 0.45 * inch, PAGE_W, 0.45 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, PAGE_H - 0.45 * inch - 3, PAGE_W, 3, fill=1, stroke=0)

    # Header text
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(MARGIN, PAGE_H - 0.32 * inch, 'QBITEL BRIDGE')
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(TEAL)
    canvas.drawString(MARGIN + 1.1 * inch, PAGE_H - 0.32 * inch,
                      'BPO & Call Center Security Platform')

    # Page number (right)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 8)
    page_str = f'Page {doc.page}'
    pw = canvas.stringWidth(page_str, 'Helvetica', 8)
    canvas.drawString(PAGE_W - MARGIN - pw, PAGE_H - 0.32 * inch, page_str)

    # Bottom footer strip
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, 0.4 * inch, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0.4 * inch, PAGE_W, 2, fill=1, stroke=0)

    canvas.setFillColor(MID_GREY)
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(WHITE_C)
    canvas.drawString(MARGIN, 0.15 * inch,
                      'Confidential — For Authorized Recipients Only  |  © 2026 QBITEL. All Rights Reserved.')
    canvas.setFont('Helvetica', 7.5)
    contact_str = 'enterprise@qbitel.com  |  bridge.qbitel.com'
    cw = canvas.stringWidth(contact_str, 'Helvetica', 7.5)
    canvas.drawString(PAGE_W - MARGIN - cw, 0.15 * inch, contact_str)

    canvas.restoreState()


def draw_cover(canvas, doc):
    """Full-bleed cover page."""
    canvas.saveState()
    w, h = PAGE_W, PAGE_H

    # Full navy background
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)

    # Gold diagonal accent block (top-right)
    canvas.setFillColor(GOLD)
    p = canvas.beginPath()
    p.moveTo(w * 0.55, h)
    p.lineTo(w, h)
    p.lineTo(w, h * 0.65)
    p.close()
    canvas.drawPath(p, fill=1, stroke=0)

    # Teal diagonal accent (mid-right)
    canvas.setFillColor(TEAL)
    p2 = canvas.beginPath()
    p2.moveTo(w * 0.72, h)
    p2.lineTo(w, h)
    p2.lineTo(w, h * 0.8)
    p2.close()
    canvas.drawPath(p2, fill=1, stroke=0)

    # Light teal accent strip (bottom)
    canvas.setFillColor(TEAL_DARK)
    canvas.rect(0, 0, w, 1.6 * inch, fill=1, stroke=0)

    # Gold bottom accent line
    canvas.setFillColor(GOLD)
    canvas.rect(0, 1.6 * inch, w, 5, fill=1, stroke=0)

    # QBITEL BRIDGE — Main title
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 52)
    canvas.drawString(MARGIN, h * 0.68, 'QBITEL')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica-Bold', 52)
    canvas.drawString(MARGIN, h * 0.68 - 58, 'BRIDGE')

    # Gold underline
    canvas.setFillColor(GOLD)
    canvas.rect(MARGIN, h * 0.68 - 68, 3.4 * inch, 5, fill=1, stroke=0)

    # Subtitle
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 17)
    canvas.drawString(MARGIN, h * 0.68 - 100,
                      'BPO & CALL CENTER SECURITY PLATFORM')

    # Tagline
    canvas.setFillColor(HexColor('#CCDDE8'))
    canvas.setFont('Helvetica-Oblique', 13)
    canvas.drawString(MARGIN, h * 0.68 - 128,
                      'Quantum-Safe Security for the Human API')

    # Key metrics — 3 white boxes
    metrics = [
        ('4–6 Hours', 'Full Deployment\nZero Downtime'),
        ('78%', 'Autonomous Threat\nResolution'),
        ('9 Frameworks', 'Compliance Automated'),
    ]
    box_w = (CONTENT_W - 2 * 0.15 * inch) / 3
    bx_start = MARGIN
    by = h * 0.38
    bh = 1.0 * inch

    for i, (big, small) in enumerate(metrics):
        bx = bx_start + i * (box_w + 0.15 * inch)
        # Box background
        bg_c = TEAL if i == 1 else LIGHT_NAVY
        canvas.setFillColor(bg_c)
        canvas.roundRect(bx, by, box_w, bh, 6, fill=1, stroke=0)
        # Gold accent top
        canvas.setFillColor(GOLD)
        canvas.rect(bx, by + bh - 5, box_w, 5, fill=1, stroke=0)
        # Big number
        canvas.setFillColor(GOLD if i == 1 else WHITE_C)
        canvas.setFont('Helvetica-Bold', 20)
        tw = canvas.stringWidth(big, 'Helvetica-Bold', 20)
        canvas.drawString(bx + (box_w - tw) / 2, by + bh - 30, big)
        # Small label
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 8)
        for j, line in enumerate(small.split('\n')):
            lw = canvas.stringWidth(line, 'Helvetica', 8)
            canvas.drawString(bx + (box_w - lw) / 2, by + bh - 46 - j * 12, line)

    # Second row of stats
    metrics2 = [
        ('<2ms', 'Voice Encryption Overhead'),
        ('89%+', 'Protocol Discovery Accuracy'),
        ('$10B+', 'Annual Industry Fraud Loss\nQBITEL Prevents'),
    ]
    by2 = h * 0.28
    for i, (big, small) in enumerate(metrics2):
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

    # Bottom info strip
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(MARGIN, 0.85 * inch,
                      'AI-Powered  •  Quantum-Safe  •  Zero Disruption  •  Deployed in Hours')
    canvas.setFont('Helvetica', 8.5)
    canvas.setFillColor(HexColor('#AABBCC'))
    canvas.drawString(MARGIN, 0.55 * inch,
                      'Confidential Marketing Document  |  Version 1.0  |  February 2026')

    canvas.restoreState()


# ─── Style Definitions ────────────────────────────────────────────────────────

def get_styles():
    styles = {}

    styles['body'] = ParagraphStyle(
        'body', fontName='Helvetica', fontSize=10, leading=15,
        textColor=DARK_TEXT, spaceAfter=8, spaceBefore=2,
        alignment=TA_JUSTIFY
    )
    styles['body_left'] = ParagraphStyle(
        'body_left', fontName='Helvetica', fontSize=10, leading=15,
        textColor=DARK_TEXT, spaceAfter=8, spaceBefore=2,
        alignment=TA_LEFT
    )
    styles['subsection'] = ParagraphStyle(
        'subsection', fontName='Helvetica-Bold', fontSize=12, leading=16,
        textColor=NAVY, spaceAfter=5, spaceBefore=14,
        borderPad=0
    )
    styles['bullet'] = ParagraphStyle(
        'bullet', fontName='Helvetica', fontSize=9.5, leading=14,
        textColor=DARK_TEXT, spaceAfter=3, spaceBefore=0,
        leftIndent=14, bulletIndent=0
    )
    styles['table_header'] = ParagraphStyle(
        'table_header', fontName='Helvetica-Bold', fontSize=9,
        textColor=WHITE_C, leading=12
    )
    styles['table_cell'] = ParagraphStyle(
        'table_cell', fontName='Helvetica', fontSize=9,
        textColor=DARK_TEXT, leading=12
    )
    styles['table_cell_bold'] = ParagraphStyle(
        'table_cell_bold', fontName='Helvetica-Bold', fontSize=9,
        textColor=NAVY, leading=12
    )
    styles['caption'] = ParagraphStyle(
        'caption', fontName='Helvetica-Oblique', fontSize=8,
        textColor=MID_GREY, spaceAfter=4, alignment=TA_CENTER
    )

    return styles


# ─── Table Builder ────────────────────────────────────────────────────────────

def make_table(headers, rows, col_widths, styles_dict):
    s = styles_dict
    data = []

    # Header row
    data.append([Paragraph(h, s['table_header']) for h in headers])

    # Data rows
    for ri, row in enumerate(rows):
        cells = []
        for ci, cell_text in enumerate(row):
            style = s['table_cell_bold'] if ci == 0 else s['table_cell']
            cells.append(Paragraph(cell_text, style))
        data.append(cells)

    n_rows = len(data)
    n_cols = len(headers)

    style = TableStyle([
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE_C),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ROWBACKGROUND', (0, 0), (-1, 0), NAVY),
        # Alternating rows
        *[('ROWBACKGROUND', (0, i), (-1, i), TABLE_ALT if i % 2 == 1 else WHITE_C)
          for i in range(1, n_rows)],
        # Padding
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        # Grid
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#C8D8E8')),
        ('LINEABOVE', (0, 0), (-1, 0), 0, WHITE_C),
        ('LINEBELOW', (0, -1), (-1, -1), 1, TEAL),
        # Vertical alignment
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ])

    tbl = Table(data, colWidths=[w * inch for w in col_widths])
    tbl.setStyle(style)
    return tbl


def sp(n=8):
    return Spacer(1, n)


def subsection_para(num, title, styles):
    return Paragraph(
        f'<font color="#F0A500"><b>{num}</b></font>  '
        f'<font color="#0D1B3E"><b>{title}</b></font>',
        styles['subsection']
    )


# ─── Main Build Function ───────────────────────────────────────────────────────

def build_pdf():
    out_path = 'docs/brochures/QBITEL_Bridge_BPO_Marketing_Pitch.pdf'

    # Use BaseDocTemplate for multi-template support (cover vs inner pages)
    doc = BaseDocTemplate(
        out_path,
        pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.65 * inch, bottomMargin=0.6 * inch,
    )

    cover_frame = Frame(0, 0, PAGE_W, PAGE_H, id='cover')
    inner_frame = Frame(
        MARGIN, 0.6 * inch,
        PAGE_W - 2 * MARGIN, PAGE_H - 0.65 * inch - 0.6 * inch,
        id='inner'
    )

    cover_template = PageTemplate(id='Cover', frames=[cover_frame],
                                  onPage=draw_cover)
    inner_template = PageTemplate(id='Inner', frames=[inner_frame],
                                  onPage=draw_page)
    doc.addPageTemplates([cover_template, inner_template])

    S = get_styles()
    story = []

    # ── COVER PAGE ─────────────────────────────────────────────────────────────
    # Cover is drawn entirely by draw_cover onPage callback.
    # We just need a PageBreak to advance to the next page after cover.
    from reportlab.platypus import NextPageTemplate
    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())

    # ── SECTION 1: THE PROBLEM ─────────────────────────────────────────────────
    story.append(SectionHeader(
        '1. The Problem BPOs Cannot Ignore',
        'Three converging threats exposing contact center infrastructure'
    ))
    story.append(sp(8))

    story.append(Paragraph(
        'Business Process Outsourcing operations are the frontline of global commerce — processing millions of '
        'sensitive customer interactions daily. Credit card payments, patient records, insurance details, and '
        'financial disputes flow through infrastructure built decades ago, protected by encryption standards '
        'that were never designed for today\'s threat landscape.',
        S['body']
    ))

    threats = [
        ('<b><font color="#008B9A">Quantum Harvest Attacks:</font></b>',
         'Nation-state adversaries capture encrypted call recordings today to decrypt when quantum '
         'computers arrive (5–10 years). SOX 7-year retention means recordings from this week will '
         'still exist when quantum decryption is viable.'),
        ('<b><font color="#008B9A">Toll Fraud — $10B Annual Industry Loss:</font></b>',
         'A single compromised PBX trunk can generate $50,000 in fraudulent premium-rate calls over '
         'one weekend. Traditional detection happens 72 hours later on the carrier invoice.'),
        ('<b><font color="#008B9A">Remote Workforce Security Gap:</font></b>',
         '60%+ of BPO agents work from home on consumer-grade networks. One compromised home router '
         'is all a threat actor needs to intercept agent sessions and exfiltrate PII undetected.'),
    ]
    for label, text in threats:
        story.append(Paragraph(f'▪  {label} {text}', S['bullet']))
        story.append(sp(4))

    story.append(sp(6))
    story.append(CalloutBox([
        'Traditional response: $5M+ PBX replacement, 12–18 months of migration risk, weeks of agent retraining.',
        'QBITEL response: 4–6 hour network-layer deployment. Nothing replaced. Nothing disrupted.',
    ]))
    story.append(sp(10))

    story.append(make_table(
        ['Challenge', 'Industry Impact', 'QBITEL Approach'],
        [
            ['Legacy PBX encryption', 'Unencrypted SIP carries 70%+ of global call traffic', 'PQC overlay — no hardware changes'],
            ['TN3270e terminal sessions', 'Mainframe access with zero session protection', 'PQC tunnel wrapping — transparent'],
            ['Payment DTMF exposure', '50M+ daily voice payment transactions at risk', 'Automated DTMF masking in <5ms'],
            ['Remote agent risk', '60%+ agents on consumer-grade home networks', 'VPN-less PQC tunnels — instant'],
            ['Toll fraud', '$10B+ annual industry loss', 'AI detection in <1 second'],
            ['Insider data theft', '34% of BPO breaches are insider threats', 'DLP enforcement at kernel level'],
            ['Multi-tenant compliance', 'Separate audits per enterprise client', 'Isolated policies — one platform'],
        ],
        [2.2, 2.3, 2.5],
        S
    ))

    story.append(PageBreak())

    # ── SECTION 2: HOW IT WORKS ────────────────────────────────────────────────
    story.append(SectionHeader(
        '2. QBITEL Bridge — How It Works',
        'AI discovers. PQC encrypts. Agentic AI defends.'
    ))
    story.append(sp(8))

    story.append(Paragraph(
        'QBITEL Bridge deploys as a network-layer security overlay in 4–6 hours. No PBX replacement. '
        'No agent retraining. No downtime. In three automated phases, your entire contact center '
        'environment is protected by NIST Level 5 post-quantum cryptography.',
        S['body']
    ))

    story.append(subsection_para('Phase 1', 'AI Protocol Discovery (2–4 Hours)', S))
    story.append(make_table(
        ['Discovery Phase', 'Duration', 'What Happens'],
        [
            ['Statistical Analysis', '5–10 sec', 'Entropy, byte frequency, binary vs. text classification'],
            ['ML Classification', '10–20 sec', 'CNN + BiLSTM — 89%+ protocol family accuracy'],
            ['Grammar Learning', '1–2 min', 'PCFG inference + Transformer semantic learning'],
            ['Parser Generation', '30–60 sec', 'Auto-generate parsers at 50,000+ msg/sec'],
            ['Adaptive Learning', 'Continuous', 'Error analysis and grammar refinement'],
        ],
        [2.2, 1.2, 3.6],
        S
    ))
    story.append(sp(8))

    story.append(subsection_para('Phase 2', 'Post-Quantum Encryption (1 Hour)', S))
    story.append(make_table(
        ['Subdomain', 'Algorithm', 'Latency', 'Use Case'],
        [
            ['Voice Signaling', 'ML-KEM-512 + Falcon-512', '<50ms', 'SIP/SDP encryption'],
            ['Voice Media (RTP)', 'ML-KEM-512 + Falcon-512', '<20ms', 'Real-time media encryption'],
            ['Payment Processing', 'ML-KEM-1024 + ML-DSA-87', '<100ms', 'PCI-DSS Level 1'],
            ['Call Recording', 'ML-KEM-1024 + ML-DSA-87', '<1s', 'Quantum-safe long-term storage'],
            ['Remote Agent Tunnel', 'ML-KEM-768 + ML-DSA-65', '<200ms', 'VPN-less WFH access'],
            ['Terminal Emulation', 'ML-KEM-768 + ML-DSA-65', '<300ms', 'TN3270e/TN5250 sessions'],
        ],
        [2.0, 2.2, 0.9, 1.9],
        S
    ))
    story.append(sp(8))

    story.append(subsection_para('Phase 3', 'Agentic AI Security — 78% Autonomous Response', S))
    story.append(Paragraph(
        'QBITEL\'s agentic AI monitors all protected traffic continuously. 78% of security events are handled '
        'without human intervention. Every automated action generates a plain-language LLM-powered narrative — '
        'not an alert code. All LLM reasoning runs on-premise (Ollama/Llama 3) — no customer data leaves your network.',
        S['body']
    ))

    story.append(make_table(
        ['Threat Event', 'Autonomous Response', 'Time to Resolution'],
        [
            ['SIP injection attack', 'Block source, alert NOC, preserve evidence', '<1 second'],
            ['Terminal session hijacking', 'Terminate session, force re-auth', '<2 seconds'],
            ['Bulk customer data access', 'Rate-limit, supervisor notification', '<5 seconds'],
            ['Recording tampering', 'Integrity alert, evidence locked', '<1 second'],
            ['Rogue remote agent device', 'Quarantine endpoint, suspend sessions', '<10 seconds'],
            ['Toll fraud pattern detected', 'Block trunk, alert, forensics preserved', '<1 second'],
        ],
        [2.6, 2.8, 1.6],
        S
    ))

    story.append(PageBreak())

    # ── SECTION 3: CAPABILITIES ────────────────────────────────────────────────
    story.append(SectionHeader(
        '3. Capability Deep Dives',
        'Six purpose-built security modules for contact center environments'
    ))
    story.append(sp(8))

    story.append(subsection_para('3.1', 'PCI-DSS Voice Compliance — Up to 80% Scope Reduction', S))
    story.append(Paragraph(
        'Every system that touches cardholder data falls within PCI-DSS audit scope — recordings, agent desktops, '
        'CRM integrations. QBITEL eliminates cardholder data from your environment before it can be stored or accessed.',
        S['body']
    ))
    story.append(make_table(
        ['PCI Control', 'What QBITEL Does', 'Compliance Impact'],
        [
            ['DTMF Masking', 'Card digits clamped in headset AND recording (<5ms)', 'CHD never reaches agent or recording'],
            ['Auto Pause/Resume', 'Recording pauses on payment detection automatically', 'Recording system exits PCI scope'],
            ['PAN Detection', 'Real-time Luhn validation across all data streams', 'Accidental PAN caught immediately'],
            ['Screen Masking', 'Last 4 digits only on agent desktop', 'Agent screen removed from scope'],
            ['Recording Encryption', 'ML-KEM-1024 for archive', 'Safe against quantum decryption'],
            ['Scope Tracking', 'Auto PCI scope per call/agent/tenant', 'Continuous evidence — not just at audit'],
        ],
        [1.9, 2.8, 2.3],
        S
    ))
    story.append(sp(8))

    story.append(subsection_para('3.2', 'Toll Fraud Prevention — 10 Pattern Types, <1 Second Detection', S))
    story.append(make_table(
        ['Fraud Type', 'Attack Pattern', 'QBITEL Response'],
        [
            ['IRSF', 'Premium-rate calls — 200+ country database', 'Block <1 second; forensics preserved'],
            ['PBX Hacking', 'Unauthorized trunk access; off-hours spikes', 'Trunk quarantined; NOC alerted'],
            ['Wangiri', 'Missed call callback to premium numbers', 'Pattern detected after 3 calls; blocked'],
            ['Call Transfer Fraud', 'Transfer to premium-rate destinations', 'Whitelist validation on every transfer'],
            ['Call Pumping', 'Artificially extended calls; revenue share abuse', 'Duration anomaly; call terminated'],
            ['Bypass Fraud', 'SIM box termination; CLI manipulation', 'CLI anomaly detected; call blocked'],
        ],
        [1.8, 2.4, 2.8],
        S
    ))
    story.append(sp(8))

    story.append(subsection_para('3.3', 'Agent Desktop DLP — Six Exfiltration Channels Blocked', S))
    story.append(make_table(
        ['Threat Vector', 'Exfiltration Method', 'QBITEL Defense'],
        [
            ['Clipboard', 'Copy-paste SSN/card numbers to personal files', 'Kernel-level PII clipboard block'],
            ['Screen Capture', 'Screenshot customer data', 'Block PrintScreen + third-party tools'],
            ['USB', 'Copy data to USB drive', 'USB storage blocked; events logged'],
            ['Email/Chat', 'Email PII to personal accounts', 'Outbound PII pattern monitoring'],
            ['Voice Reading', 'Read card numbers aloud during calls', 'Speech analytics detect CHD reading'],
            ['Screen Scraping', 'Automated tools scraping agent desktop', 'eBPF-based scraping detection'],
        ],
        [1.5, 2.4, 3.1],
        S
    ))
    story.append(Paragraph(
        '<b>Forensic Watermarking:</b> Every agent screen carries an invisible watermark with agent ID and '
        'session timestamp — enabling post-incident attribution to the exact agent, session, and moment.',
        S['body_left']
    ))

    story.append(PageBreak())

    # ── SECTION 4: COMPLIANCE ──────────────────────────────────────────────────
    story.append(SectionHeader(
        '4. Compliance Coverage',
        'Nine frameworks automated. Reports generated in under 10 minutes.'
    ))
    story.append(sp(8))

    story.append(make_table(
        ['Framework', 'BPO Application', 'What QBITEL Automates'],
        [
            ['PCI-DSS 4.0', 'Voice payment processing', 'DTMF masking, recording encryption, agent controls, scope reduction'],
            ['TCPA', 'Outbound calling', 'Consent tracking, DNC list enforcement, time-of-day restrictions'],
            ['HIPAA', 'Healthcare BPO', 'PHI encryption, minimum necessary access, 6-year audit retention'],
            ['SOC 2 Type II', 'Service organizations', 'Continuous monitoring, automated evidence, real-time alerting'],
            ['GDPR', 'EU customer data', 'Recording consent, DSAR processing, retention/deletion automation'],
            ['SOX', 'Financial services', 'Recording integrity, tamper-evident audit trails, 7-year retention'],
            ['GLBA', 'Financial data', 'Data classification, access controls, breach notification'],
            ['FCA/MiFID II', 'UK/EU financial recording', 'All-call recording, quantum-safe encryption, retention'],
            ['NIST PQC', 'Quantum-safe transition', 'ML-KEM + ML-DSA across all voice and data channels'],
        ],
        [1.5, 1.8, 3.7],
        S
    ))
    story.append(sp(6))
    story.append(CalloutBox([
        'Compliance reports generated on-demand in under 10 minutes per framework.',
        'Blockchain-backed audit trails for tamper evidence. Multi-tenant isolation ensures',
        'each enterprise client receives fully independent compliance reporting.',
    ]))

    # ── SECTION 5: REAL-WORLD SCENARIOS ───────────────────────────────────────
    story.append(sp(10))
    story.append(SectionHeader(
        '5. Real-World Scenarios',
        'Proven results across financial services, healthcare, and multi-tenant BPO environments'
    ))
    story.append(sp(10))

    scenarios = [
        ('SCENARIO A', 'Financial Services BPO — 5,000 Seats', [
            'Challenge: Fortune 500 bank BPO handles credit card disputes over legacy Avaya infrastructure.',
            'PCI-DSS audit scope covers the entire contact center. Annual audit cost: $2.3M.',
            '',
            'QBITEL: Network tap in 30 min. AI discovers SIP + TN3270e in 2 hours. PQC-SRTP and DTMF',
            'masking active before end of business. Agents work uninterrupted. PCI scope reduced 80%.',
            'Annual audit cost reduced by $1.7M.',
        ]),
        ('SCENARIO B', 'Healthcare BPO — 2,000 Remote Agents', [
            'Challenge: Remote agents handle patient scheduling and insurance verification from home networks.',
            'HIPAA requires audit trails and PHI encryption — with zero visibility into agent home environments.',
            '',
            'QBITEL: VPN-less quantum-safe tunnels deployed to 2,000 agents. Continuous endpoint compliance',
            '(WPA3, disk encryption, antivirus). PHI exfiltration monitoring across all channels.',
            'First HIPAA audit: zero findings.',
        ]),
        ('SCENARIO C', 'Toll Fraud — The $50,000 Weekend Attack', [
            'Challenge: Contact center discovers $50,000 in fraudulent Caribbean premium-rate calls on Monday',
            'morning. Attack ran all weekend through one compromised SIP trunk.',
            '',
            'With QBITEL: IRSF pattern detected after the 3rd call. Trunk isolated automatically. NOC alerted.',
            'Forensic evidence preserved. Total loss: $200 vs. $50,000 without QBITEL.',
        ]),
        ('SCENARIO D', 'Multi-Tenant BPO — Banking, Healthcare, Retail', [
            'Challenge: One BPO facility, three enterprise clients with different compliance frameworks.',
            'Three separate annual audits. Separate infrastructure per client is cost-prohibitive.',
            '',
            'QBITEL: Per-tenant cryptographic key isolation. Per-tenant compliance: PCI-DSS, HIPAA, SOC 2.',
            'Three independent compliance reports — generated on-demand in under 10 minutes each.',
        ]),
    ]

    for label, title, lines in scenarios:
        story.append(ScenarioBox(label, title, lines))
        story.append(sp(10))

    story.append(PageBreak())

    # ── SECTION 6: INTEGRATION ─────────────────────────────────────────────────
    story.append(SectionHeader(
        '6. Integration',
        'Works with everything you already have — nothing replaced'
    ))
    story.append(sp(8))

    story.append(subsection_para('6.1', 'PBX & Telephony Platforms', S))
    story.append(make_table(
        ['Platform', 'Integration Method', 'Infrastructure Change'],
        [
            ['Avaya Aura / CM', 'TSAPI/DMCC with PQC tunnel', 'None'],
            ['Cisco CUCM', 'CTI-OS / Finesse API + PQC', 'None'],
            ['Genesys Cloud', 'REST API with PQC-TLS', 'None'],
            ['Asterisk / FreePBX', 'AMI/ARI with PQC tunnel', 'None'],
            ['Legacy PBX (any)', 'Protocol-level overlay', 'None — fully agnostic'],
        ],
        [2.2, 2.8, 2.0],
        S
    ))
    story.append(sp(8))

    story.append(subsection_para('6.2', 'Deployment — 4 Steps to Full Protection', S))
    story.append(make_table(
        ['Step', 'Duration', 'Activity'],
        [
            ['1. Network Tap', '30 minutes', 'Non-invasive tap on voice/data network — passive observation'],
            ['2. Protocol Discovery', '2–4 hours', 'AI identifies all protocols, traffic patterns, data flows'],
            ['3. Encryption Activation', '1 hour', 'PQC activated for all discovered protocols automatically'],
            ['4. Policy Deployment', '30 minutes', 'BPO security policies configured and validated'],
            ['Total', '4–6 hours', 'Full quantum-safe protection. Zero downtime. Nothing replaced.'],
        ],
        [1.8, 1.2, 4.0],
        S
    ))

    # ── SECTION 7: PERFORMANCE ─────────────────────────────────────────────────
    story.append(sp(10))
    story.append(SectionHeader(
        '7. Performance Specifications',
        'Enterprise scale — zero compromise on voice quality'
    ))
    story.append(sp(8))

    story.append(make_table(
        ['Metric', 'Performance', 'Business Impact'],
        [
            ['Voice PQC overhead', '<2ms', 'Inaudible — within ITU-T G.114 budget'],
            ['DTMF masking latency', '<5ms', 'Callers cannot detect payment flow delay'],
            ['Toll fraud detection', '<1 second', 'Fraud stopped before meaningful loss'],
            ['Concurrent agent sessions', '20,000+', 'Enterprise scale from day one'],
            ['Recording encryption', '10,000+ streams', 'Entire recording estate simultaneously'],
            ['Autonomous threat resolution', '78% no-touch', 'SOC handles exceptions, not routine alerts'],
            ['PAN detection', '<50ms', 'Real-time Luhn validation on all streams'],
            ['Compliance report generation', '<10 minutes', 'On-demand for any framework, any client'],
        ],
        [2.4, 1.6, 3.0],
        S
    ))

    # ── SECTION 8: WHY QBITEL ─────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(SectionHeader(
        '8. Why QBITEL Bridge',
        'The only platform purpose-built for BPO contact center security'
    ))
    story.append(sp(8))

    story.append(subsection_para('8.1', 'vs. Traditional Security Vendors', S))
    story.append(make_table(
        ['Capability', 'Traditional Vendors', 'QBITEL Bridge'],
        [
            ['Legacy protocol support', 'Known protocols only', 'AI discovers unknown/undocumented protocols'],
            ['Quantum cryptography', 'Not available', 'NIST Level 5 — ML-KEM-1024 + ML-DSA-87'],
            ['Deployment', 'Weeks to months', '4–6 hours, zero downtime'],
            ['BPO-specific controls', 'Generic security policies', 'DTMF masking, toll fraud, DLP, multi-tenant'],
            ['Threat response', 'Alert-based, SOC review', '78% autonomous LLM reasoning'],
            ['Compliance', 'Manual evidence collection', '9 frameworks automated, <10 min reports'],
            ['Air-gapped AI', 'Cloud-dependent', 'On-premise Ollama — no data egress'],
        ],
        [2.2, 2.0, 2.8],
        S
    ))
    story.append(sp(10))

    story.append(subsection_para('8.2', 'vs. PBX Replacement', S))
    story.append(make_table(
        ['Factor', 'PBX Replacement', 'QBITEL Bridge'],
        [
            ['Cost', '$5M+ hardware + migration + training', 'Fraction of replacement cost'],
            ['Downtime', 'Weeks of migration risk', 'Zero — 4–6 hour deployment'],
            ['Timeline', '12–18 months to production', 'Full protection same business day'],
            ['Quantum-readiness', 'Depends on vendor roadmap', 'NIST Level 5 from day one'],
            ['Agent impact', 'Full retraining required', 'Zero — agents notice nothing'],
        ],
        [1.8, 2.6, 2.6],
        S
    ))

    # ── SECTION 9: NEXT STEPS ──────────────────────────────────────────────────
    story.append(sp(12))
    story.append(SectionHeader(
        '9. Next Steps',
        'Full quantum-safe protection in under one week'
    ))
    story.append(sp(10))

    next_steps = [
        ('STEP 1', 'Discovery Assessment — Free, 2 Hours', [
            'Passive network tap deployed in your environment.',
            'Report delivered showing: which protocols are running, what is unencrypted,',
            'and where your PCI/HIPAA compliance scope currently sits.',
            'No commitment. No infrastructure changes.',
        ]),
        ('STEP 2', 'Proof of Concept — 2 Weeks', [
            'Full QBITEL Bridge deployment against your production traffic.',
            'Real threats. Live compliance reporting. Toll fraud monitoring.',
            'ROI measured against your current state baseline — before you sign.',
        ]),
        ('STEP 3', 'Production Deployment — 4–6 Hours', [
            'Zero-downtime deployment. All protocols protected. All agents covered.',
            'All 9 compliance frameworks active. Full documentation, runbooks,',
            'and dedicated support for the first 90 days.',
        ]),
    ]

    for label, title, lines in next_steps:
        story.append(ScenarioBox(label, title, lines))
        story.append(sp(10))

    # Contact info
    story.append(sp(6))
    contact_data = [
        [Paragraph('<b><font color="#008B9A">✉</font>  enterprise@qbitel.com</b>', S['table_header']),
         Paragraph('<b><font color="#008B9A">⊕</font>  bridge.qbitel.com</b>', S['table_header']),
         Paragraph('<b><font color="#008B9A">◉</font>  Contact your account team</b>', S['table_header'])],
        [Paragraph('Email', ParagraphStyle('sm', fontName='Helvetica', fontSize=8, textColor=TEAL)),
         Paragraph('Website', ParagraphStyle('sm', fontName='Helvetica', fontSize=8, textColor=TEAL)),
         Paragraph('Schedule a call', ParagraphStyle('sm', fontName='Helvetica', fontSize=8, textColor=TEAL))],
    ]
    contact_tbl = Table(contact_data, colWidths=[CONTENT_W / 3] * 3)
    contact_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), NAVY),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, TEAL),
    ]))
    story.append(contact_tbl)
    story.append(sp(14))

    # Closing
    story.append(Paragraph(
        '<i>QBITEL Bridge — Because the Human API Deserves Quantum-Safe Protection.</i>',
        ParagraphStyle('closing', fontName='Helvetica-BoldOblique', fontSize=12,
                       textColor=NAVY, alignment=TA_CENTER, spaceAfter=6)
    ))
    story.append(Paragraph(
        'Version 1.0  |  February 2026  |  Confidential — For Authorized Recipients Only',
        ParagraphStyle('ver', fontName='Helvetica', fontSize=8,
                       textColor=MID_GREY, alignment=TA_CENTER)
    ))

    doc.build(story)
    print(f'PDF saved: {out_path}')
    return out_path


if __name__ == '__main__':
    build_pdf()
