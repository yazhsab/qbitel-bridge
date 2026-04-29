"""Build QBITEL Bridge Automotive & Connected Vehicles Marketing Pitch - PDF"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, NextPageTemplate

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

def sp(n): return Spacer(1, n)


class ColorBar(Flowable):
    def __init__(self, color, height=4, width=None):
        Flowable.__init__(self)
        self.color = color
        self.bar_height = height
        self.bar_width = width or CONTENT_W

    def wrap(self, availW, availH):
        return (self.bar_width, self.bar_height)

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.bar_width, self.bar_height, fill=1, stroke=0)


class SectionHeader(Flowable):
    def __init__(self, title, subtitle='', width=None):
        Flowable.__init__(self)
        self.title = title
        self.subtitle = subtitle
        self.h = 48 if subtitle else 36
        self.w = width or CONTENT_W

    def wrap(self, availW, availH):
        return (self.w, self.h)

    def draw(self):
        c = self.canv
        w, h = self.w, self.h
        # Navy background
        c.setFillColor(NAVY)
        c.rect(0, 0, w, h, fill=1, stroke=0)
        # Gold left accent bar (6px)
        c.setFillColor(GOLD)
        c.rect(0, 0, 6, h, fill=1, stroke=0)
        # Teal right accent bar (4px)
        c.setFillColor(TEAL)
        c.rect(w - 4, 0, 4, h, fill=1, stroke=0)
        # Title text
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 11)
        if self.subtitle:
            c.drawString(14, h - 18, self.title)
            c.setFillColor(HexColor('#A8C4CC'))
            c.setFont('Helvetica', 8.5)
            c.drawString(14, h - 32, self.subtitle)
        else:
            c.drawString(14, h / 2 - 5, self.title)


class StatBlock(Flowable):
    def __init__(self, stats, width=None):
        Flowable.__init__(self)
        self.stats = stats  # list of (value, label)
        self.w = width or CONTENT_W
        self.h = 72

    def wrap(self, availW, availH):
        return (self.w, self.h)

    def draw(self):
        c = self.canv
        n = len(self.stats)
        col_w = self.w / n
        for i, (val, label) in enumerate(self.stats):
            x = i * col_w
            # Navy cell background
            c.setFillColor(NAVY)
            c.rect(x + 2, 2, col_w - 4, self.h - 4, fill=1, stroke=0)
            # Gold value
            c.setFillColor(GOLD)
            c.setFont('Helvetica-Bold', 15)
            vw = c.stringWidth(val, 'Helvetica-Bold', 15)
            c.drawString(x + col_w / 2 - vw / 2, self.h * 0.5, val)
            # White label
            c.setFillColor(WHITE_C)
            c.setFont('Helvetica', 8)
            lw = c.stringWidth(label, 'Helvetica', 8)
            c.drawString(x + col_w / 2 - lw / 2, self.h * 0.25, label)


class ScenarioBox(Flowable):
    def __init__(self, label, title, lines, width=None):
        Flowable.__init__(self)
        self.label = label
        self.title = title
        self.lines = lines
        self.w = width or CONTENT_W
        self.line_h = 14
        self.header_h = 30
        self.h = self.header_h + len(lines) * self.line_h + 16

    def wrap(self, availW, availH):
        return (self.w, self.h)

    def draw(self):
        c = self.canv
        w, h = self.w, self.h
        # Teal header
        c.setFillColor(TEAL)
        c.rect(0, h - self.header_h, w, self.header_h, fill=1, stroke=0)
        # Label badge
        c.setFillColor(NAVY)
        c.rect(0, h - self.header_h, 58, self.header_h, fill=1, stroke=0)
        c.setFillColor(GOLD)
        c.setFont('Helvetica-Bold', 8)
        lw = c.stringWidth(self.label, 'Helvetica-Bold', 8)
        c.drawString(29 - lw / 2, h - self.header_h + 10, self.label)
        # Title in header
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 9)
        c.drawString(66, h - self.header_h + 10, self.title)
        # White body
        c.setFillColor(LIGHT_BG)
        c.rect(0, 0, w, h - self.header_h, fill=1, stroke=0)
        # Border
        c.setStrokeColor(TEAL)
        c.setLineWidth(0.5)
        c.rect(0, 0, w, h, fill=0, stroke=1)
        # Bullet lines
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 8.5)
        for i, line in enumerate(self.lines):
            y = h - self.header_h - 12 - i * self.line_h
            c.drawString(12, y, u'\u2022 ' + line)


class CalloutBox(Flowable):
    def __init__(self, text, width=None):
        Flowable.__init__(self)
        self.text = text
        self.w = width or CONTENT_W
        self.h = 52

    def wrap(self, availW, availH):
        return (self.w, self.h)

    def draw(self):
        c = self.canv
        # Light background
        c.setFillColor(LIGHT_BG)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        # Gold left bar (6px)
        c.setFillColor(GOLD)
        c.rect(0, 0, 6, self.h, fill=1, stroke=0)
        # Text
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 8.5)
        lines = self.text.split('\n')
        for i, line in enumerate(lines[:3]):
            c.drawString(14, self.h - 16 - i * 14, line)


def get_styles():
    return {
        'body': ParagraphStyle('body', fontName='Helvetica', fontSize=8.5,
                               leading=12, textColor=DARK_TEXT, spaceAfter=4),
        'h1': ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=16,
                             leading=20, textColor=NAVY, spaceAfter=8),
        'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=12,
                             leading=16, textColor=TEAL, spaceAfter=6),
        'h3': ParagraphStyle('h3', fontName='Helvetica-Bold', fontSize=10,
                             leading=14, textColor=NAVY, spaceAfter=4),
        'table_cell': ParagraphStyle('table_cell', fontName='Helvetica', fontSize=8,
                                     leading=11, textColor=DARK_TEXT),
        'table_header': ParagraphStyle('table_header', fontName='Helvetica-Bold', fontSize=8,
                                       leading=11, textColor=WHITE_C),
        'bullet': ParagraphStyle('bullet', fontName='Helvetica', fontSize=8.5,
                                 leading=12, textColor=DARK_TEXT, leftIndent=12,
                                 firstLineIndent=-10, spaceAfter=2),
    }


def draw_page(canvas, doc):
    canvas.saveState()
    # NAVY header strip 30pt
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 30, PAGE_W, 30, fill=1, stroke=0)
    # Title white 9pt left
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(MARGIN, PAGE_H - 20, 'QBITEL BRIDGE \u2014 AUTOMOTIVE & CONNECTED VEHICLES')
    # Page number right
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 20, f'Page {doc.page}')
    # TEAL footer strip
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, PAGE_W, 22, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 7.5)
    canvas.drawString(MARGIN, 7, 'Confidential \u2014 For Authorized Recipients Only  |  \u00a9 2026 QBITEL. All Rights Reserved.')
    contact_str = 'enterprise@qbitel.com  |  bridge.qbitel.com'
    cw = canvas.stringWidth(contact_str, 'Helvetica', 7.5)
    canvas.drawString(PAGE_W - MARGIN - cw, 7, contact_str)
    canvas.restoreState()


def draw_cover(canvas, doc):
    canvas.saveState()
    # Full NAVY background
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # TEAL decorative top strip
    canvas.setFillColor(TEAL)
    canvas.rect(0, PAGE_H * 0.55, PAGE_W, PAGE_H * 0.45, fill=1, stroke=0)
    # Gold accent line
    canvas.setFillColor(GOLD)
    canvas.rect(0, PAGE_H * 0.55, PAGE_W, 4, fill=1, stroke=0)
    # "QBITEL BRIDGE" top
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 28)
    canvas.drawString(MARGIN, PAGE_H * 0.55 + PAGE_H * 0.45 * 0.75, 'QBITEL BRIDGE')
    # Subtitle in GOLD
    canvas.setFillColor(GOLD)
    canvas.setFont('Helvetica-Bold', 18)
    canvas.drawString(MARGIN, PAGE_H * 0.55 + PAGE_H * 0.45 * 0.55, 'Automotive & Connected Vehicles')
    # Main title white
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 20)
    canvas.drawString(MARGIN, PAGE_H * 0.55 + PAGE_H * 0.45 * 0.35, 'Quantum-Safe Security for the')
    canvas.drawString(MARGIN, PAGE_H * 0.55 + PAGE_H * 0.45 * 0.2, 'Connected Vehicle Ecosystem')
    # 4 stat boxes in NAVY section
    stats = [('<5ms', 'V2X Verification'), ('1,500+', 'msg/sec Batch'), ('72%', 'Smaller Certs'), ('UNECE WP.29', 'Ready')]
    box_w = (PAGE_W - 2 * MARGIN - 30) / 4
    for i, (val, lbl) in enumerate(stats):
        x = MARGIN + i * (box_w + 10)
        y = 0.12 * PAGE_H
        canvas.setFillColor(LIGHT_NAVY)
        canvas.roundRect(x, y, box_w, 0.1 * PAGE_H, 6, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.setFont('Helvetica-Bold', 14)
        vw = canvas.stringWidth(val, 'Helvetica-Bold', 14)
        canvas.drawString(x + box_w / 2 - vw / 2, y + 0.1 * PAGE_H * 0.55, val)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 8)
        lw = canvas.stringWidth(lbl, 'Helvetica', 8)
        canvas.drawString(x + box_w / 2 - lw / 2, y + 0.1 * PAGE_H * 0.25, lbl)
    # Bottom contact strip
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, PAGE_W, 0.08 * PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 9)
    canvas.drawString(MARGIN, 0.04 * PAGE_H, 'enterprise@qbitel.com  |  bridge.qbitel.com')
    canvas.restoreState()


def build_doc(output_path):
    doc = BaseDocTemplate(output_path, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN)
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id='cover')
    inner_frame = Frame(MARGIN, 0.7 * inch, CONTENT_W,
        PAGE_H - MARGIN - 0.7 * inch, id='inner')
    cover_template = PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover)
    inner_template = PageTemplate(id='Inner', frames=[inner_frame], onPage=draw_page)
    doc.addPageTemplates([cover_template, inner_template])
    S = get_styles()
    story = []
    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())

    # Executive summary callout
    story.append(SectionHeader('Executive Summary', 'Quantum-safe V2X security for the connected vehicle ecosystem'))
    story.append(sp(8))
    story.append(StatBlock([
        ('<5ms', 'V2X Verification'),
        ('1,500+', 'msg/sec Batch'),
        ('72%', 'Smaller Certs'),
        ('$10-30', 'Per Vehicle/Year'),
        ('54', 'WP.29 Countries'),
    ]))
    story.append(sp(10))
    story.append(Paragraph(
        'With 1.4 billion vehicles on roads globally and fleet lifecycles stretching 15-20 years into the future, '
        'vehicles sold today will still be operating in 2040-2045 — exactly when quantum computers are projected to '
        'break classical ECDSA cryptography. QBITEL Bridge delivers post-quantum V2X authentication, fleet-wide OTA '
        'PQC migration, and automated UNECE WP.29 compliance evidence in a single integrated platform.',
        S['body']))
    story.append(sp(12))

    # Section 1: The Connected Vehicle Security Crisis
    story.append(SectionHeader('1. The Connected Vehicle Security Crisis', 'Three threats that cannot be ignored'))
    story.append(sp(8))
    # threat stats table
    threats = [
        ['Threat', 'Description', 'Impact'],
        ['V2X Spoofing', '$20 SDR device can inject false V2X messages, trigger emergency braking', 'Accidents, gridlock, platoon attacks'],
        ['Quantum Fleet Lifecycle', '15-20yr vehicle lifetime extends past quantum threat horizon (2040-2045)', 'Fleet-wide key compromise'],
        ['UNECE WP.29 R155', 'Mandatory cybersecurity management in 54 countries — penalties include market withdrawal', 'Type approval revocation'],
    ]
    tbl = Table(threats, colWidths=[CONTENT_W * 0.2, CONTENT_W * 0.5, CONTENT_W * 0.3])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY), ('TEXTCOLOR', (0, 0), (-1, 0), WHITE_C),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_BG), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, TABLE_ALT]),
        ('FONTSIZE', (0, 1), (-1, -1), 8), ('GRID', (0, 0), (-1, -1), 0.5, MID_GREY),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6), ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)
    story.append(sp(12))

    # Section 2: 7 Capabilities
    caps = [
        ('2. V2X Message Authentication', 'IEEE 1609.2 + PQC \u2014 Cryptographic truth for every vehicle message',
         [('Algorithm', 'ML-KEM-768 (key encap) + ML-DSA-65 + Falcon-512 (signatures)'),
          ('Compatibility', 'Backward compatible with existing RSU infrastructure'),
          ('Coverage', 'BSM, SPaT, MAP, TIM, EVA message types'),
          ('Latency', '<5ms end-to-end signature verification')]),
        ('3. Implicit Certificate Compression', '72% smaller than standard Dilithium \u2014 built for V2X bandwidth',
         [('Size', '666 bytes (Falcon-512) vs 2,420 bytes (Dilithium)'),
          ('Throughput', '1,500+ msg/sec batch verification with SIMD'),
          ('Encoding', 'X9.62 point compression + dictionary-based'),
          ('Compatibility', 'IEEE 1609.2 extension field encoding')]),
        ('4. Fleet-Wide OTA PQC Migration', 'Staged rollout with automatic rollback \u2014 zero fleet disruption',
         [('Rollout stages', 'Canary (10 vehicles) \u2192 1% \u2192 10% \u2192 100%'),
          ('Rollback', 'Automatic on >0.1% error rate at any stage'),
          ('Delta updates', 'Cryptographic diff reduces OTA payload 80%'),
          ('Verification', 'TPM-attested post-install verification')]),
        ('5. SCMS Integration & Pseudonym Management', 'Full Security Credential Management System compatibility',
         [('Pseudonyms', 'Short-lived certificates (1 week) for privacy'),
          ('Misbehavior', 'Automated detection and reporting to SCMS'),
          ('Revocation', 'CRL distribution in <100ms to all RSUs'),
          ('Provisioning', 'Bulk enrollment: 10,000 vehicles/hour')]),
        ('6. Autonomous Vehicle Platooning Security', 'Authenticated convoy commands \u2014 rogue vehicles cannot join',
         [('Leader verification', 'Platoon leader PQC identity binding'),
          ('Command auth', 'Every following distance/brake command signed'),
          ('Rogue detection', '<100ms detection of unauthorized convoy member'),
          ('Failsafe', 'Graceful platoon disbanding on auth failure')]),
        ('7. ISO/SAE 21434 & UNECE WP.29 Compliance', 'Complete compliance evidence for type approval',
         [('TARA', 'Automated threat analysis and risk assessment'),
          ('Evidence', 'Type approval security validation report'),
          ('CSMS', 'Cybersecurity Management System documentation'),
          ('R156', 'Software update management system compliance')]),
    ]
    for title, subtitle, rows in caps:
        story.append(SectionHeader(title, subtitle))
        story.append(sp(6))
        tdata = [['Attribute', 'Value']] + [[r[0], r[1]] for r in rows]
        t = Table(tdata, colWidths=[CONTENT_W * 0.3, CONTENT_W * 0.7])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), TEAL), ('TEXTCOLOR', (0, 0), (-1, 0), WHITE_C),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, LIGHT_BG]),
            ('FONTSIZE', (0, 1), (-1, -1), 8), ('GRID', (0, 0), (-1, -1), 0.5, MID_GREY),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(sp(10))

    # Compliance table
    story.append(SectionHeader('8. Compliance Coverage', 'All major automotive cybersecurity frameworks'))
    story.append(sp(6))
    comp_data = [['Framework', 'Coverage', 'Key Requirement'],
        ['UNECE WP.29 R155', 'Full', 'Cybersecurity Management System'],
        ['UNECE WP.29 R156', 'Full', 'Software Update Management'],
        ['ISO/SAE 21434', 'Full', 'TARA, security validation'],
        ['IEEE 1609.2', 'Extended (PQC)', 'V2X security services'],
        ['SAE J3061', 'Aligned', 'Cyber-physical systems guidebook'],
        ['NIST FIPS 203/204', 'Native', 'ML-KEM, ML-DSA algorithms'],
        ['SAE J2735', 'Compatible', 'DSRC/C-V2X message sets'],
    ]
    ct = Table(comp_data, colWidths=[CONTENT_W * 0.3, CONTENT_W * 0.15, CONTENT_W * 0.55])
    ct.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY), ('TEXTCOLOR', (0, 0), (-1, 0), WHITE_C),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, TABLE_ALT]),
        ('FONTSIZE', (0, 1), (-1, -1), 8), ('GRID', (0, 0), (-1, -1), 0.5, MID_GREY),
        ('LEFTPADDING', (0, 0), (-1, -1), 6), ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(ct)
    story.append(sp(12))

    # Performance specs
    story.append(SectionHeader('9. Performance Specifications'))
    story.append(sp(6))
    perf = [['Metric', 'Requirement', 'QBITEL Performance'],
        ['V2X verification latency', '<10ms', '<5ms'],
        ['Batch throughput', '500+ msg/sec', '1,500+ msg/sec'],
        ['Certificate size', '<2KB', '666 bytes (Falcon-512)'],
        ['OTA delivery time', '<24 hours', '<4 hours (delta)'],
        ['False positive rate', '<0.001%', '<0.0001%'],
        ['Fleet coverage stages', 'N/A', 'Canary \u2192 1% \u2192 10% \u2192 100%'],
        ['Cost per vehicle/year', '$50-200 (classical)', '$10-30 (QBITEL PQC)'],
    ]
    pt = Table(perf, colWidths=[CONTENT_W * 0.35, CONTENT_W * 0.25, CONTENT_W * 0.4])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY), ('TEXTCOLOR', (0, 0), (-1, 0), WHITE_C),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, TABLE_ALT]),
        ('FONTSIZE', (0, 1), (-1, -1), 8), ('GRID', (0, 0), (-1, -1), 0.5, MID_GREY),
        ('LEFTPADDING', (0, 0), (-1, -1), 6), ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(pt)
    story.append(sp(12))

    # Customer Scenarios
    story.append(SectionHeader('10. Customer Scenarios'))
    story.append(sp(8))
    scenarios = [
        ('OEM', 'Global OEM \u2014 Fleet-Wide PQC Migration',
         ['Challenge: 10M vehicle fleet, classical ECDSA expiring before 2040',
          'Solution: QBITEL staged OTA rollout with SCMS integration',
          'Result: UNECE WP.29 type approval, <$30/vehicle/year cost',
          'Timeline: 12-month full fleet migration, zero downtime']),
        ('TIER 1', 'Tier 1 Supplier \u2014 V2X Module Security',
         ['Challenge: Authentication chip integration for next-gen V2X modules',
          'Solution: Falcon-512 implicit certificates, SIMD batch verification',
          'Result: 1,500+ msg/sec, 72% smaller certificates vs Dilithium',
          'Timeline: 6-month silicon bring-up + validation']),
        ('CITY', 'Smart City \u2014 V2I Infrastructure Security',
         ['Challenge: 500 RSUs with unauthenticated V2X broadcast',
          'Solution: RSU firmware update, traffic signal anti-spoofing',
          'Result: Cryptographic V2I authentication, intersection safety',
          'Timeline: 3-month RSU upgrade, zero road closure']),
    ]
    for label, title, lines in scenarios:
        story.append(ScenarioBox(label, title, lines))
        story.append(sp(8))

    # Next Steps
    story.append(SectionHeader('11. Next Steps'))
    story.append(sp(8))
    nextsteps = [
        ('STEP 1', 'V2X Security Assessment (Week 1-2)',
         ['Passive V2X traffic capture and protocol analysis',
          'Current SCMS architecture review',
          'UNECE WP.29 gap assessment']),
        ('STEP 2', 'Proof of Concept (Month 1-2)',
         ['1,000-vehicle pilot deployment',
          'SCMS integration and certificate provisioning',
          'Performance benchmark vs requirements']),
        ('STEP 3', 'Fleet Migration Planning (Month 3)',
         ['Staged OTA rollout roadmap',
          'Rollback procedure documentation',
          'Type approval evidence package']),
    ]
    for label, title, lines in nextsteps:
        story.append(ScenarioBox(label, title, lines))
        story.append(sp(8))

    # Contact
    story.append(sp(6))
    contact_data = [
        [Paragraph('<b>enterprise@qbitel.com</b>', ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=10, textColor=GOLD)),
         Paragraph('<b>bridge.qbitel.com</b>', ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=10, textColor=GOLD)),
         Paragraph('<b>Contact your account team</b>', ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=10, textColor=GOLD))],
        [Paragraph('Email', ParagraphStyle('cl', fontName='Helvetica', fontSize=8, textColor=WHITE_C)),
         Paragraph('Website', ParagraphStyle('cl', fontName='Helvetica', fontSize=8, textColor=WHITE_C)),
         Paragraph('Schedule a call', ParagraphStyle('cl', fontName='Helvetica', fontSize=8, textColor=WHITE_C))],
    ]
    contact_tbl = Table(contact_data, colWidths=[CONTENT_W / 3] * 3)
    contact_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), NAVY),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, TEAL),
    ]))
    story.append(contact_tbl)

    doc.build(story)


if __name__ == '__main__':
    import os
    os.chdir('/Users/prabakarankannan/qbitel')
    build_doc('docs/brochures/QBITEL_Bridge_Automotive_Marketing_Pitch.pdf')
    print('PDF saved: docs/brochures/QBITEL_Bridge_Automotive_Marketing_Pitch.pdf')
