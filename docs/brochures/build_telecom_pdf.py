"""
Build QBITEL Bridge Telecom Marketing Pitch - Professional PDF
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
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, NextPageTemplate

# Brand Colors
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


class ColorBar(Flowable):
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


class StatBlock(Flowable):
    def __init__(self, stats, width=None, height=70, bg=NAVY):
        super().__init__()
        self.stats = stats
        self.w = width or CONTENT_W
        self.h = height
        self.bg = bg

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(self.bg)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        n = len(self.stats)
        col_w = self.w / n
        for i, (big, small) in enumerate(self.stats):
            x = i * col_w
            if i > 0:
                c.setStrokeColor(TEAL)
                c.setLineWidth(0.5)
                c.line(x, 10, x, self.h - 10)
            c.setFillColor(GOLD)
            c.setFont('Helvetica-Bold', 18)
            c.drawCentredString(x + col_w / 2, self.h - 32, big)
            c.setFillColor(WHITE_C)
            c.setFont('Helvetica', 8)
            c.drawCentredString(x + col_w / 2, self.h - 48, small)


class ScenarioBox(Flowable):
    def __init__(self, title, org, challenge, solution, outcome, width=None):
        super().__init__()
        self.title = title
        self.org = org
        self.challenge = challenge
        self.solution = solution
        self.outcome = outcome
        self.w = width or CONTENT_W
        self.h = 160

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(LIGHT_BG)
        c.roundRect(0, 0, self.w, self.h, 6, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.roundRect(0, self.h - 30, self.w, 30, 6, fill=1, stroke=0)
        c.rect(0, self.h - 44, self.w, 14, fill=1, stroke=0)
        c.setFillColor(GOLD)
        c.rect(0, self.h - 30, 6, 30, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 11)
        c.drawString(14, self.h - 19, self.title)
        y = self.h - 48
        for label, text in [('ORG:', self.org), ('CHALLENGE:', self.challenge),
                            ('SOLUTION:', self.solution), ('OUTCOME:', self.outcome)]:
            c.setFillColor(TEAL_DARK)
            c.setFont('Helvetica-Bold', 8)
            c.drawString(8, y, label)
            c.setFillColor(DARK_TEXT)
            c.setFont('Helvetica', 8)
            # Wrap text manually
            words = text.split()
            line = ''
            x_off = 75
            line_h = 11
            for word in words:
                test = line + ' ' + word if line else word
                if c.stringWidth(test, 'Helvetica', 8) < self.w - x_off - 8:
                    line = test
                else:
                    c.drawString(x_off, y, line)
                    y -= line_h
                    line = word
                    x_off = 8
            if line:
                c.drawString(x_off, y, line)
                x_off = 8
            y -= 14


class CalloutBox(Flowable):
    def __init__(self, text, bg=None, border_color=None, width=None, height=50):
        super().__init__()
        self.text = text
        self.bg = bg or LIGHT_BG
        self.border_color = border_color or TEAL
        self.w = width or CONTENT_W
        self.h = height

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(self.bg)
        c.roundRect(0, 0, self.w, self.h, 4, fill=1, stroke=0)
        c.setFillColor(self.border_color)
        c.rect(0, 0, 5, self.h, fill=1, stroke=0)
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica-Oblique', 10)
        words = self.text.split()
        line = ''
        y = self.h - 18
        for word in words:
            test = line + ' ' + word if line else word
            if c.stringWidth(test, 'Helvetica-Oblique', 10) < self.w - 22:
                line = test
            else:
                c.drawString(14, y, line)
                y -= 14
                line = word
        if line:
            c.drawString(14, y, line)


def draw_cover(canvas, doc):
    canvas.saveState()
    w, h = PAGE_W, PAGE_H
    # Full-bleed navy background
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    # Teal accent stripe top
    canvas.setFillColor(TEAL)
    canvas.rect(0, h - 8, w, 8, fill=1, stroke=0)
    # Gold diagonal accent
    canvas.setFillColor(GOLD)
    canvas.rect(0, h - 14, w * 0.4, 4, fill=1, stroke=0)
    # Large decorative circle
    canvas.setFillColor(HexColor('#1A2D5A'))
    canvas.circle(w * 0.85, h * 0.72, 180, fill=1, stroke=0)
    canvas.setFillColor(HexColor('#0F2248'))
    canvas.circle(w * 0.85, h * 0.72, 140, fill=1, stroke=0)
    # Inner teal ring
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(2)
    canvas.circle(w * 0.85, h * 0.72, 110, fill=0, stroke=1)
    # QBITEL wordmark
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 28)
    canvas.drawString(0.75 * inch, h - 1.2 * inch, 'QBITEL BRIDGE')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica', 13)
    canvas.drawString(0.75 * inch, h - 1.55 * inch, 'Quantum-Safe Infrastructure Security')
    # Gold divider
    canvas.setFillColor(GOLD)
    canvas.rect(0.75 * inch, h - 1.75 * inch, 3.5 * inch, 3, fill=1, stroke=0)
    # Main title
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 36)
    canvas.drawString(0.75 * inch, h - 2.6 * inch, 'TELECOMMUNICATIONS')
    canvas.setFont('Helvetica-Bold', 34)
    canvas.setFillColor(GOLD)
    canvas.drawString(0.75 * inch, h - 3.1 * inch, '& 5G NETWORKS')
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 14)
    canvas.drawString(0.75 * inch, h - 3.5 * inch, 'Carrier-Grade Post-Quantum Cryptographic Security')
    canvas.setFont('Helvetica', 12)
    canvas.setFillColor(HexColor('#A0B8CC'))
    canvas.drawString(0.75 * inch, h - 3.8 * inch, 'SS7/Diameter Protection  |  5G Core Security  |  IoT at Scale')
    # Stats row
    stats_y = h * 0.30
    stats = [
        ('150,000+', 'PQC ops/sec'),
        ('99.999%', 'Availability SLA'),
        ('<1 sec', 'SS7 Block Speed'),
        ('5G Slice', 'Ready'),
    ]
    box_w = (w - 1.5 * inch) / len(stats)
    for i, (big, small) in enumerate(stats):
        bx = 0.75 * inch + i * box_w
        canvas.setFillColor(HexColor('#1A2D5A'))
        canvas.roundRect(bx, stats_y, box_w - 8, 72, 5, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.setFont('Helvetica-Bold', 16)
        canvas.drawCentredString(bx + (box_w - 8) / 2, stats_y + 42, big)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 9)
        canvas.drawCentredString(bx + (box_w - 8) / 2, stats_y + 26, small)
    # Bottom contact bar
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
    canvas.drawRightString(w - 0.75 * inch, 0.28 * inch, 'CONFIDENTIAL — FOR AUTHORIZED RECIPIENTS ONLY')
    canvas.restoreState()


def draw_page(canvas, doc):
    canvas.saveState()
    # Header
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 0.55 * inch, PAGE_W, 0.55 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, PAGE_H - 0.55 * inch - 3, PAGE_W, 3, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(MARGIN, PAGE_H - 0.35 * inch, 'QBITEL BRIDGE')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica', 9)
    canvas.drawString(MARGIN + 1.05 * inch, PAGE_H - 0.35 * inch, 'TELECOMMUNICATIONS & 5G NETWORKS')
    canvas.setFillColor(HexColor('#A0B8CC'))
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.35 * inch, 'enterprise@qbitel.com')
    # Footer
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, 0.55 * inch, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0.55 * inch, PAGE_W, 2, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 8)
    canvas.drawString(MARGIN, 0.22 * inch, 'QBITEL BRIDGE — TELECOMMUNICATIONS & 5G')
    canvas.setFillColor(MID_GREY)
    canvas.drawRightString(PAGE_W - MARGIN, 0.22 * inch,
                           f'Page {doc.page} | Confidential')
    canvas.restoreState()


def build_styles():
    styles = {}
    styles['h1'] = ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=16,
                                    textColor=NAVY, spaceAfter=10, spaceBefore=16)
    styles['h2'] = ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=13,
                                    textColor=TEAL_DARK, spaceAfter=8, spaceBefore=12)
    styles['h3'] = ParagraphStyle('h3', fontName='Helvetica-Bold', fontSize=11,
                                    textColor=NAVY, spaceAfter=6, spaceBefore=10)
    styles['body'] = ParagraphStyle('body', fontName='Helvetica', fontSize=9.5,
                                      textColor=DARK_TEXT, spaceAfter=6, leading=14,
                                      alignment=TA_JUSTIFY)
    styles['bullet'] = ParagraphStyle('bullet', fontName='Helvetica', fontSize=9.5,
                                        textColor=DARK_TEXT, spaceAfter=4, leading=13,
                                        leftIndent=16, bulletIndent=4)
    styles['caption'] = ParagraphStyle('caption', fontName='Helvetica-Oblique', fontSize=8,
                                         textColor=MID_GREY, spaceAfter=4, alignment=TA_CENTER)
    styles['table_hdr'] = ParagraphStyle('table_hdr', fontName='Helvetica-Bold', fontSize=8.5,
                                           textColor=WHITE_C)
    styles['table_cell'] = ParagraphStyle('table_cell', fontName='Helvetica', fontSize=8.5,
                                            textColor=DARK_TEXT, leading=12)
    styles['contact'] = ParagraphStyle('contact', fontName='Helvetica-Bold', fontSize=10,
                                         textColor=TEAL, spaceAfter=4, alignment=TA_CENTER)
    return styles


def build_doc(output_path):
    doc = BaseDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN
    )
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H,
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
                        id='cover')
    inner_frame = Frame(MARGIN, 0.7 * inch, CONTENT_W, PAGE_H - MARGIN - 0.7 * inch,
                        id='inner')
    cover_template = PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover)
    inner_template = PageTemplate(id='Inner', frames=[inner_frame], onPage=draw_page)
    doc.addPageTemplates([cover_template, inner_template])

    S = build_styles()
    story = []
    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())

    # --- Executive Summary ---
    story.append(SectionHeader('Executive Summary',
                               'Carrier-Grade Quantum-Safe Security for Global Telecommunications'))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        'The global telecommunications industry operates the most critical digital infrastructure on earth — '
        'carrying voice, data, financial transactions, emergency services, and government communications for '
        'billions of people. Yet the protocols underpinning this infrastructure were designed in an era when '
        'security was an afterthought, and the quantum threat is transforming yesterday\'s theoretical risks '
        'into today\'s operational emergencies.', S['body']))
    story.append(Spacer(1, 6))
    story.append(StatBlock([
        ('150,000+', 'PQC ops/sec'),
        ('99.999%', 'Availability'),
        ('<1 sec', 'SS7 Block'),
        ('5G Slice', 'Ready'),
    ]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        '<b>QBITEL Bridge</b> delivers carrier-grade post-quantum cryptographic (PQC) security purpose-built '
        'for telecommunications networks: protecting SS7/Diameter signaling, 5G core network slices, SIP/VoIP '
        'infrastructure, IoT mass-device gateways, and subscriber data at 150,000+ cryptographic operations '
        'per second with 99.999% availability — without requiring updates to a single subscriber device.', S['body']))
    story.append(Paragraph(
        'In 2024 alone, 850 million SS7 attacks were recorded globally. International Revenue Share Fraud (IRSF) '
        'costs the industry more than $10 billion annually. And the quantum threat to subscriber databases — '
        'holding biometric, financial, and location data for billions of people — is no longer measured in '
        'decades but in years.', S['body']))
    story.append(CalloutBox(
        'QBITEL Bridge integrates directly into existing signaling infrastructure, 5G core network functions, and '
        'fraud management systems — providing a unified cryptographic defense layer that scales from regional MVNOs '
        'to the largest Tier-1 MNOs on the planet.',
        bg=LIGHT_BG, border_color=GOLD, height=60))
    story.append(Spacer(1, 12))

    # --- Telecom Security Paradox ---
    story.append(SectionHeader('The Telecom Security Paradox',
                               'Networks that connect the world are themselves vulnerable'))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        'Telecommunications carriers face a fundamental paradox: the very protocols that enabled global '
        'connectivity were engineered for reliability and interoperability — not security. SS7, the signaling '
        'protocol suite that routes calls and SMS for billions of subscribers worldwide, was designed in 1975 '
        'with zero authentication mechanisms. Any operator connected to the global SS7 network can potentially '
        'send any message on behalf of any other operator.', S['body']))
    story.append(Paragraph(
        'This is not a theoretical concern. SS7 attack kits are commercially available on dark web forums for '
        'as little as $500 per session. A sophisticated adversary can track the location of a VIP subscriber, '
        'intercept SMS-based two-factor authentication codes, forward calls silently, or deny service to a '
        'target — all from a laptop, anywhere in the world, exploiting protocols your network uses every second.',
        S['body']))
    story.append(Paragraph(
        'Meanwhile, the rollout of 5G networks has tripled the attack surface compared to 4G. The 5G '
        'Service-Based Architecture (SBA) introduces HTTP/2-based interfaces between network functions — '
        'bringing web-application vulnerabilities into the core of carrier infrastructure. Network slicing '
        'enables unprecedented service differentiation, but each slice boundary is a potential lateral '
        'movement vector.', S['body']))
    story.append(Paragraph(
        'And looming over all of this is the quantum threat. Nation-state adversaries are harvesting encrypted '
        'subscriber data today using harvest-now, decrypt-later strategies — collecting authentication vectors, '
        'location histories, and communications metadata that will become readable the moment a '
        'cryptographically-relevant quantum computer comes online.', S['body']))
    story.append(Spacer(1, 12))

    # --- Three Critical Threats ---
    story.append(SectionHeader('Three Critical Threats Facing Telecom Operators'))
    story.append(Spacer(1, 8))

    threat_data = [
        [Paragraph('THREAT', S['table_hdr']),
         Paragraph('VECTOR', S['table_hdr']),
         Paragraph('IMPACT', S['table_hdr']),
         Paragraph('QBITEL BRIDGE RESPONSE', S['table_hdr'])],
        [Paragraph('SS7/Diameter Legacy', S['table_cell']),
         Paragraph('MAP/ISUP/TCAP exploits, no authentication, globally accessible', S['table_cell']),
         Paragraph('850M attacks/year, location tracking $500/target, SMS MFA bypass', S['table_cell']),
         Paragraph('Real-time MAP filtering, PQC auth vector wrapping, <1 sec block', S['table_cell'])],
        [Paragraph('5G Hyperscale Surface', S['table_cell']),
         Paragraph('SBI HTTP/2 APIs, cross-slice lateral movement, PFCP manipulation', S['table_cell']),
         Paragraph('3x larger than 4G, core NF compromise exposes millions of subscribers', S['table_cell']),
         Paragraph('Per-slice PQC policies, SBI mTLS, NF integrity verification', S['table_cell'])],
        [Paragraph('Quantum Subscriber Data', S['table_cell']),
         Paragraph('Harvest-now decrypt-later on HLR/HSS, Ki keys, RAND/SRES vectors', S['table_cell']),
         Paragraph('Billions of subscriber records exposed, GDPR/NIS2 fines up to 4% revenue', S['table_cell']),
         Paragraph('ML-KEM/ML-DSA at network layer, no device update needed, crypto-agile', S['table_cell'])],
    ]
    threat_table = Table(threat_data, colWidths=[CONTENT_W * 0.18, CONTENT_W * 0.27,
                                                  CONTENT_W * 0.27, CONTENT_W * 0.28])
    threat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE_C),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('GRID', (0, 0), (-1, -1), 0.4, MID_GREY),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(threat_table)
    story.append(Spacer(1, 14))

    # --- Threat Details ---
    story.append(Paragraph('<b>Threat 1: SS7/Diameter Legacy Protocol Exposure</b>', S['h2']))
    story.append(Paragraph(
        'SS7 (Signaling System No. 7), including MAP, ISUP, and TCAP protocols, underpins virtually all mobile '
        'voice and SMS services globally. Despite being designed in 1975 with no authentication, it remains the '
        'backbone of inter-carrier communication. Diameter, its 4G-era successor, improved some aspects but '
        'inherited fundamental trust model weaknesses.', S['body']))
    for bullet in [
        '<b>Location tracking</b> of any mobile subscriber is available for purchase at $500 per target',
        '<b>SMS interception</b> enables bypass of SMS-based MFA across banking, government, and enterprise applications',
        '<b>Call diversion</b> allows silent call forwarding to adversary-controlled numbers',
        '<b>850M+ attacks recorded</b> in 2024 — averaging nearly 100 SS7 attacks per second globally',
        '<b>Subscriber denial of service</b> attacks can silence journalists, activists, and government officials',
    ]:
        story.append(Paragraph(f'\u2022 {bullet}', S['bullet']))
    story.append(Spacer(1, 8))

    story.append(Paragraph('<b>Threat 2: 5G Hyperscale Attack Surface</b>', S['h2']))
    story.append(Paragraph(
        '5G networks introduce unprecedented complexity. The 5G core Service-Based Architecture (SBA) relies on '
        'HTTP/2 RESTful APIs between network functions — AMF, SMF, UPF, AUSF, UDM, PCF — connected via the N2, '
        'N3, N4, N6, and SBI interfaces. Each interface is a potential attack vector. Each network slice has '
        'different security requirements that a one-size-fits-all approach cannot address.', S['body']))
    for bullet in [
        'A compromised 5G core network function can expose subscriber session data for <b>millions of users simultaneously</b>',
        'Cross-slice attacks allow lateral movement from low-security IoT slice to high-security emergency-services slice',
        'PFCP manipulation on N4 interface can redirect user-plane traffic at carrier scale',
        '5G core has <b>3x more attack surface</b> than 4G, with larger proportion internet-adjacent',
    ]:
        story.append(Paragraph(f'\u2022 {bullet}', S['bullet']))
    story.append(Spacer(1, 8))

    story.append(Paragraph('<b>Threat 3: Quantum Threat to Subscriber Databases</b>', S['h2']))
    story.append(Paragraph(
        'Mobile network operators maintain some of the most sensitive databases on earth: subscriber identity '
        '(IMSI/MSISDN), authentication credentials (Ki keys, RAND/SRES vectors), location history, communication '
        'metadata, biometric data for identity verification, and financial information for billing. This data is '
        'protected today by classical cryptography — AES-128 for 4G subscriber authentication, RSA and ECDH for '
        'key exchange — all of which quantum computers will break.', S['body']))
    for bullet in [
        'Nation-state adversaries harvesting encrypted subscriber data today for <b>quantum-era decryption later</b>',
        'Authentication credentials from quantum-decrypted HLR/HSS enable subscriber impersonation at scale',
        'GDPR/NIS2 penalties for subscriber data breach can reach <b>4% of global annual turnover</b>',
        'GSMA FS.19 has flagged quantum vulnerability as a critical long-term risk requiring immediate roadmap',
    ]:
        story.append(Paragraph(f'\u2022 {bullet}', S['bullet']))
    story.append(Spacer(1, 12))
    story.append(PageBreak())

    # --- Seven Capabilities ---
    story.append(SectionHeader('QBITEL Bridge — Seven Core Capabilities',
                               'Full-spectrum protection across every telecom protocol and domain'))
    story.append(Spacer(1, 10))

    caps = [
        ('01', 'SS7/Diameter Protocol Security',
         'Real-time MAP filtering blocks location tracking, SMS interception, and call diversion attacks '
         'in under 1 second. PQC-wrapped authentication vectors protect HLR/HSS subscriber data. '
         'Covers E1/T1 legacy and SIGTRAN/M3UA IP-encapsulated SS7. Immutable blockchain audit trails '
         'for GSMA FS.11/FS.19 compliance.'),
        ('02', '5G Core Network Slice Protection',
         'Per-slice PQC policy enforcement for eMBB, URLLC, and mMTC slices with independent key '
         'hierarchies. mTLS with PQC hybrid key exchange on all SBI interfaces. AMF/SMF/UPF/AUSF/UDM/NRF '
         'integrity verification. NSSAI manipulation detection and slice admission control.'),
        ('03', 'SIP/VoIP Fraud Prevention',
         'Real-time IRSF detection with sub-second blocking of calls to international premium rate numbers. '
         'Wangiri one-ring fraud detection. SIP INVITE flood rate limiting for 100,000+ concurrent calls. '
         'ML-DSA-signed SIP trunk authentication. Compatible with Cisco CUBE, Ribbon SBC, Oracle ACME Packet.'),
        ('04', 'IoT/mMTC Mass Device Security',
         'Network-layer PQC protection for 50M+ IoT devices — no firmware update required. ML-KEM optimized '
         'for NB-IoT, LTE-M, and eMTC constrained devices. Real-time IoT botnet detection with automatic '
         'quarantine. GSMA IoT security guidelines and ETSI EN 303 645 alignment.'),
        ('05', 'Carrier-Grade PQC at Scale',
         '150,000+ PQC operations per second per node with hardware acceleration. ML-KEM-768/1024, '
         'ML-DSA-65/87, SLH-DSA-SHAKE-256 support. FIPS 140-3 Level 3 HSM integration. Linear horizontal '
         'scaling. Sub-millisecond cryptographic latency invisible to subscriber experience.'),
        ('06', 'Autonomous Fraud Detection',
         'Graph neural network models trained on inter-carrier signaling graphs detect coordinated attacks '
         'across multiple operators. 500+ behavioral features per subscriber. Federated learning preserves '
         'data sovereignty. False positive rate below 0.01% — carrier NOC ready. Covers IRSF, SIM swap, '
         'bypass fraud, wangiri, roaming fraud, and PBX hacking.'),
        ('07', 'Regulatory Compliance Automation',
         'Continuous automated generation of 3GPP TS 33.501, GSMA FS.19/FS.11, NESAS, NIS2, FCC/CISA, '
         'ETSI NFV-SEC, BEREC, and GDPR compliance evidence. Daily cryptographic logs with HSM attestation. '
         'NIS2 72-hour incident reporting. Monthly SS7 attack blocking reports in GSMA format.'),
    ]

    for num, title, desc in caps:
        cap_data = [[
            Paragraph(f'<font color="#F0A500"><b>{num}</b></font>', ParagraphStyle(
                'cn', fontName='Helvetica-Bold', fontSize=20, textColor=GOLD)),
            Paragraph(f'<b>{title}</b><br/><font size="9">{desc}</font>', ParagraphStyle(
                'cd', fontName='Helvetica', fontSize=9.5, textColor=DARK_TEXT, leading=14))
        ]]
        cap_table = Table(cap_data, colWidths=[0.55 * inch, CONTENT_W - 0.55 * inch])
        cap_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
            ('LEFTPADDING', (0, 0), (0, 0), 10),
            ('LEFTPADDING', (1, 0), (1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LINEAFTER', (0, 0), (0, 0), 3, TEAL),
            ('LINEBEFORE', (0, 0), (0, 0), 3, GOLD),
        ]))
        story.append(KeepTogether([cap_table, Spacer(1, 6)]))
    story.append(Spacer(1, 12))
    story.append(PageBreak())

    # --- Compliance Coverage ---
    story.append(SectionHeader('Compliance Coverage Matrix',
                               'Automated evidence generation for every major telecom regulatory framework'))
    story.append(Spacer(1, 8))
    comp_data = [
        [Paragraph(h, S['table_hdr']) for h in ['Regulation', 'Scope', 'Key Requirements', 'QBITEL Bridge Status']],
        ['3GPP TS 33.501', '5G security', 'SBI mTLS, SUPI protection, slice security', 'Full coverage'],
        ['GSMA FS.19', 'PQC readiness', 'Algorithm migration, subscriber protection', 'Full coverage'],
        ['GSMA FS.11', 'SS7 security', 'MAP filtering, location privacy', 'Full coverage'],
        ['NESAS / SCAS', 'Equipment assurance', 'Security test evidence', 'Automated evidence'],
        ['NIS2 Directive', 'EU critical infra', 'Incident reporting, security measures', 'Automated reporting'],
        ['FCC SS7 Action', 'US carriers', 'SS7 monitoring, remediation', 'Full coverage'],
        ['ETSI NFV-SEC', 'Virtual NFs', 'vNF integrity, isolation', 'Full coverage'],
        ['BEREC Security', 'EU telecom', 'Availability, integrity measures', 'KPI automation'],
        ['GDPR Article 32', 'EU data protection', 'Encryption of subscriber data', 'PQC encryption'],
    ]
    for i in range(1, len(comp_data)):
        comp_data[i] = [Paragraph(str(v), S['table_cell']) for v in comp_data[i]]
    comp_table = Table(comp_data, colWidths=[CONTENT_W * 0.2, CONTENT_W * 0.18,
                                              CONTENT_W * 0.37, CONTENT_W * 0.25])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('GRID', (0, 0), (-1, -1), 0.4, MID_GREY),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 14))

    # --- Integration Ecosystem ---
    story.append(SectionHeader('Integration Ecosystem',
                               'Nokia, Ericsson, Open RAN, Amdocs, Oracle Communications — and more'))
    story.append(Spacer(1, 8))
    int_data = [
        [Paragraph(h, S['table_hdr']) for h in ['Vendor / Platform', 'Integration Points', 'Deployment Mode']],
        ['Nokia', 'CloudBand, AVP telemetry, NetAct/1Network Manager, Bell Labs PQC research', 'Cloud-native, containerized'],
        ['Ericsson', 'Cloud Manager, ERIC-OSS, Radio System O-RAN xApp, UDM/AUSF PQC', 'Cloud-native, hybrid'],
        ['Open RAN / Free5GC', 'O-RAN O1/A1/E2, Free5GC, Open5GS, OpenShift Telco, ONAP', 'Kubernetes-native'],
        ['Amdocs', 'CES fraud alerts, Revenue Management, Network Cloud OSS northbound API', 'API integration'],
        ['Oracle Communications', 'Diameter Signaling Router, Session Border Controller, PCRF/PCF', 'Inline / API'],
        ['Subex / TEOCO', 'Fraud Management System REST API, real-time event feed, revenue impact', 'API integration'],
        ['Syniverse', 'Roaming hub integrity, inter-carrier settlement, fraud data sharing', 'Federated'],
    ]
    for i in range(1, len(int_data)):
        int_data[i] = [Paragraph(str(v), S['table_cell']) for v in int_data[i]]
    int_table = Table(int_data, colWidths=[CONTENT_W * 0.22, CONTENT_W * 0.52, CONTENT_W * 0.26])
    int_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('GRID', (0, 0), (-1, -1), 0.4, MID_GREY),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(int_table)
    story.append(Spacer(1, 14))
    story.append(PageBreak())

    # --- Performance Specs ---
    story.append(SectionHeader('Performance Specifications',
                               'Carrier-grade SLAs engineered for the world\u2019s most demanding networks'))
    story.append(Spacer(1, 8))
    perf_data = [
        [Paragraph(h, S['table_hdr']) for h in ['Metric', 'Specification', 'Notes']],
        ['PQC Operations/Second', '150,000+', 'Per node, hardware-accelerated'],
        ['SS7 Attack Block Latency', '<1 second', 'From detection to blocking action'],
        ['SIP Call Processing', '100,000+ concurrent', 'No measurable latency addition'],
        ['5G Slice Policies', 'Unlimited', 'Per-slice, per-subscriber granularity'],
        ['Subscriber Capacity', '500M+', 'With horizontal node scaling'],
        ['Availability SLA', '99.999%', 'Five-nines, active-active clustering'],
        ['Cryptographic Latency', '<1ms', 'Sub-millisecond, invisible to subscriber'],
        ['Fraud Detection Latency', '<500ms', 'Real-time blocking capability'],
        ['False Positive Rate', '<0.01%', 'AI-tuned, carrier NOC ready'],
        ['HSM Key Operations/sec', '10,000+', 'Hardware-bound, FIPS 140-3 L3'],
    ]
    for i in range(1, len(perf_data)):
        perf_data[i] = [Paragraph(str(v), S['table_cell']) for v in perf_data[i]]
    perf_table = Table(perf_data, colWidths=[CONTENT_W * 0.32, CONTENT_W * 0.28, CONTENT_W * 0.40])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('GRID', (0, 0), (-1, -1), 0.4, MID_GREY),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 14))

    # --- Competitive Differentiation ---
    story.append(SectionHeader('Competitive Differentiation',
                               'Why QBITEL Bridge — not point solutions'))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        'The telecom security market is fragmented across specialist vendors: signaling firewall vendors '
        '(ISMS, Cellusys, Mobileum), fraud management systems (Subex, TEOCO, Syniverse), network security '
        'vendors (Palo Alto, Fortinet), and emerging PQC vendors. Each solves part of the problem. '
        'QBITEL Bridge is the only platform that addresses all dimensions simultaneously.', S['body']))
    diff_points = [
        ('Unified Platform',
         'Protocol security + PQC + fraud detection in one integrated platform — eliminating integration '
         'complexity, data silos, and coverage gaps of point-solution architectures'),
        ('True Carrier-Grade Performance',
         '150,000+ ops/sec PQC with 99.999% availability — not lab-benchmark figures that fall apart under real traffic'),
        ('No Device Updates Required',
         'Network-layer PQC protection means 500M+ legacy devices are protected without a single firmware update or SIM swap'),
        ('Full Protocol Stack Coverage',
         'From 1975-era SS7 through Diameter, SIP, GTP, PFCP, and 5G SBI — single policy framework and unified audit trail'),
        ('Compliance Automation',
         '3GPP, GSMA, NIS2, and FCC-ready documentation generated automatically — eliminating hundreds of audit hours per cycle'),
        ('Any-Environment Deployment',
         'On-premises, private cloud, AWS/Azure/GCP, and hybrid — with Kubernetes-native telco cloud containers'),
    ]
    diff_data = [[Paragraph('<b>DIFFERENTIATOR</b>', S['table_hdr']),
                  Paragraph('<b>CAPABILITY</b>', S['table_hdr'])]]
    for title, desc in diff_points:
        diff_data.append([Paragraph(f'<b>{title}</b>', S['table_cell']),
                          Paragraph(desc, S['table_cell'])])
    diff_table = Table(diff_data, colWidths=[CONTENT_W * 0.28, CONTENT_W * 0.72])
    diff_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('GRID', (0, 0), (-1, -1), 0.4, MID_GREY),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(diff_table)
    story.append(Spacer(1, 14))
    story.append(PageBreak())

    # --- Customer Scenarios ---
    story.append(SectionHeader('Customer Scenarios',
                               'Real outcomes for Tier-1 MNOs, MVNOs, and telecom equipment vendors'))
    story.append(Spacer(1, 10))

    scenarios = [
        ('Scenario 1: Tier-1 MNO — Quantum-Safe Subscriber Database Protection',
         'European Tier-1 MNO, 80M subscribers, 12 countries',
         'Nation-state adversary harvesting SS7 signaling data for quantum-era decryption. '
         'NIS2 compliance deadline. GSMA FS.19 quantum readiness assessment due.',
         'PQC wrapping of SS7 MAP auth vectors. Retrospective analysis found 127 undetected '
         'VIP surveillance campaigns. 5G SA core with ML-DSA SBI. Automated FS.19 documentation.',
         'GSMA FS.19 certified in 14 weeks. All SS7 surveillance blocked. NIS2 evidence '
         'provided to regulator. EUR 40M in projected fines avoided.'),
        ('Scenario 2: MVNO — IRSF Revenue Protection',
         'UK MVNO, 4M subscribers, operating over Tier-1 host MNO',
         'EUR 2.3M/month IRSF losses. Fraud tools only flagging after call completion — '
         'too late for real-time blocking. Contractual liability to host MNO for fraud traffic costs.',
         'SIP trunk IRSF detection engine. ML models trained on 6 months of CDR history. '
         'Sub-second blocking of IRSF calls. Wangiri detection. Automatic trunk suspension.',
         'IRSF reduced from EUR 2.3M/month to EUR 15K/month — 99.3% reduction in 8 weeks. ROI in 60 days.'),
        ('Scenario 3: Equipment Vendor — PQC-Ready Product Portfolio',
         'Tier-2 SBC and signaling gateway vendor, global MNO customer base',
         'MNO RFPs requiring PQC capability. Engineering team estimated 18 months for native PQC. '
         'Risk of losing major contracts worth tens of millions of euros.',
         'QBITEL Bridge PQC SDK integrated into SBC firmware as OEM module. '
         'Pre-integrated PQC key management. Co-branded joint MNO sales enablement.',
         'Three major MNO contracts won citing PQC differentiation — EUR 47M total value. '
         'Time-to-market reduced from 18 months to 12 weeks.'),
    ]

    for title, org, challenge, solution, outcome in scenarios:
        scen_data = [
            [Paragraph(f'<b>{title}</b>', ParagraphStyle('sh', fontName='Helvetica-Bold',
                        fontSize=10, textColor=WHITE_C))],
            [[
                Table([
                    [Paragraph('<b>Organization</b>', S['table_hdr']),
                     Paragraph('<b>Challenge</b>', S['table_hdr']),
                     Paragraph('<b>Solution</b>', S['table_hdr']),
                     Paragraph('<b>Outcome</b>', S['table_hdr'])],
                    [Paragraph(org, S['table_cell']),
                     Paragraph(challenge, S['table_cell']),
                     Paragraph(solution, S['table_cell']),
                     Paragraph(outcome, S['table_cell'])],
                ], colWidths=[CONTENT_W * 0.2, CONTENT_W * 0.25, CONTENT_W * 0.28, CONTENT_W * 0.27],
                   style=TableStyle([
                       ('BACKGROUND', (0, 0), (-1, 0), TEAL_DARK),
                       ('BACKGROUND', (0, 1), (-1, 1), LIGHT_BG),
                       ('GRID', (0, 0), (-1, -1), 0.4, MID_GREY),
                       ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                       ('TOPPADDING', (0, 0), (-1, -1), 6),
                       ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                       ('LEFTPADDING', (0, 0), (-1, -1), 6),
                   ]))
            ]]
        ]
        outer = Table([
            [Paragraph(f'<b>{title}</b>', ParagraphStyle('sh2', fontName='Helvetica-Bold',
                        fontSize=10, textColor=WHITE_C, spaceAfter=0))],
        ], colWidths=[CONTENT_W])
        outer.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), NAVY),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        inner_tbl = Table([
            [Paragraph('<b>Organization</b>', S['table_hdr']),
             Paragraph('<b>Challenge</b>', S['table_hdr']),
             Paragraph('<b>Solution</b>', S['table_hdr']),
             Paragraph('<b>Outcome</b>', S['table_hdr'])],
            [Paragraph(org, S['table_cell']),
             Paragraph(challenge, S['table_cell']),
             Paragraph(solution, S['table_cell']),
             Paragraph(outcome, S['table_cell'])],
        ], colWidths=[CONTENT_W * 0.2, CONTENT_W * 0.25, CONTENT_W * 0.28, CONTENT_W * 0.27])
        inner_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), TEAL_DARK),
            ('BACKGROUND', (0, 1), (-1, 1), LIGHT_BG),
            ('GRID', (0, 0), (-1, -1), 0.4, MID_GREY),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(KeepTogether([outer, inner_tbl, Spacer(1, 12)]))

    story.append(PageBreak())

    # --- Deployment Timeline ---
    story.append(SectionHeader('Deployment Timeline',
                               'From discovery to full carrier-grade protection'))
    story.append(Spacer(1, 8))
    timeline_data = [
        [Paragraph(h, S['table_hdr']) for h in ['Week', 'Phase', 'Key Deliverable']],
        ['1-2', 'Discovery & Assessment', 'Network topology map, protocol inventory, SS7/Diameter threat profile'],
        ['3-4', 'Architecture Design', 'Integration architecture, PQC policy framework, HSM design'],
        ['5-6', 'Infrastructure Provisioning', 'HSM installation, network taps, pilot environment setup'],
        ['7-9', 'Protocol Security Configuration', 'SS7 MAP filtering rules, Diameter policies, SIP trunk hardening'],
        ['10-11', '5G Core Integration', 'SBI mTLS/PQC, slice policies, AMF/SMF/UPF/NRF protection activation'],
        ['12-13', 'Fraud Detection Activation', 'ML model initialization, IRSF rules, FMS integration, wangiri'],
        ['14-15', 'Monitoring & Alerting', 'NOC dashboards, GSMA KPI reporting, NIS2 regulatory feeds'],
        ['16', 'Go-Live & Acceptance', 'Full production cutover, SLA validation, runbook handover'],
    ]
    for i in range(1, len(timeline_data)):
        timeline_data[i] = [Paragraph(str(v), S['table_cell']) for v in timeline_data[i]]
    tl_table = Table(timeline_data, colWidths=[CONTENT_W * 0.1, CONTENT_W * 0.32, CONTENT_W * 0.58])
    tl_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('GRID', (0, 0), (-1, -1), 0.4, MID_GREY),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(tl_table)
    story.append(Spacer(1, 6))
    story.append(CalloutBox(
        'MVNO Accelerated Deployment: 6 weeks to full SIP fraud protection and compliance reporting, '
        'using pre-integrated connectors for major host MNO platforms.',
        bg=LIGHT_BG, border_color=GOLD, height=44))
    story.append(Spacer(1, 14))

    # --- Next Steps ---
    story.append(SectionHeader('Next Steps',
                               'Start your quantum-safe telecom journey today'))
    story.append(Spacer(1, 8))
    steps = [
        ('Step 1', 'Threat Exposure Assessment (Complimentary)',
         'Passive analysis of SS7 and Diameter signaling traffic — identifying active attack campaigns, '
         'protocol anomalies, and quantum vulnerability exposure with no network changes. '
         'Confidential executive briefing within 5 business days.'),
        ('Step 2', 'PQC Readiness Assessment',
         'Structured assessment of your cryptographic posture across all network domains — mapping current '
         'algorithm usage, identifying quantum-vulnerable systems, and producing a prioritized migration '
         'roadmap aligned with GSMA FS.19 requirements.'),
        ('Step 3', '30-Day Pilot Program',
         'Live pilot on a defined network segment — signaling link, 5G core slice, or SIP trunk group — '
         'demonstrating real attack detection, fraud prevention impact, and compliance evidence generation '
         'with your actual traffic.'),
        ('Step 4', 'Full Network Deployment',
         'Full deployment with QBITEL Bridge-certified implementation support, 24/7 NOC monitoring, '
         'ongoing threat intelligence feeds, and continuous compliance automation.'),
    ]
    step_data = [[Paragraph('<b>STEP</b>', S['table_hdr']),
                  Paragraph('<b>ACTIVITY</b>', S['table_hdr']),
                  Paragraph('<b>DESCRIPTION</b>', S['table_hdr'])]]
    for s, t, d in steps:
        step_data.append([Paragraph(f'<b>{s}</b>', S['table_cell']),
                          Paragraph(f'<b>{t}</b>', S['table_cell']),
                          Paragraph(d, S['table_cell'])])
    step_table = Table(step_data, colWidths=[CONTENT_W * 0.1, CONTENT_W * 0.3, CONTENT_W * 0.6])
    step_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('GRID', (0, 0), (-1, -1), 0.4, MID_GREY),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(step_table)
    story.append(Spacer(1, 16))

    # --- Contact ---
    story.append(ColorBar(3, GOLD))
    story.append(Spacer(1, 10))
    contact_data = [[
        Paragraph('<b>Enterprise Sales</b><br/>enterprise@qbitel.com', ParagraphStyle(
            'ct', fontName='Helvetica', fontSize=10, textColor=DARK_TEXT, alignment=TA_CENTER)),
        Paragraph('<b>Technical Pre-Sales</b><br/>https://bridge.qbitel.com', ParagraphStyle(
            'ct2', fontName='Helvetica', fontSize=10, textColor=DARK_TEXT, alignment=TA_CENTER)),
        Paragraph('<b>Partner Program</b><br/>OEM, VAR & SI inquiries welcome', ParagraphStyle(
            'ct3', fontName='Helvetica', fontSize=10, textColor=DARK_TEXT, alignment=TA_CENTER)),
    ]]
    ct_table = Table(contact_data, colWidths=[CONTENT_W / 3] * 3)
    ct_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
        ('GRID', (0, 0), (-1, -1), 0.5, MID_GREY),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(ct_table)
    story.append(Spacer(1, 8))
    story.append(ColorBar(3, NAVY))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        'Securing the networks that secure the world.',
        ParagraphStyle('tagline', fontName='Helvetica-Oblique', fontSize=11,
                       textColor=TEAL, alignment=TA_CENTER)))

    doc.build(story)
    print(f"PDF written to {output_path}")


if __name__ == '__main__':
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'QBITEL_Bridge_Telecom_Marketing_Pitch.pdf')
    build_doc(out)
