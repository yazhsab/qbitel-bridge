"""
Build QBITEL Bridge Aviation & Aerospace Marketing Pitch - Professional PDF
Uses ReportLab for full layout/design control.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, NextPageTemplate

# ── Brand Colors ──────────────────────────────────────────────────────────────
NAVY      = HexColor('#0D1B3E')
TEAL      = HexColor('#008B9A')
TEAL_DARK = HexColor('#006B7A')
GOLD      = HexColor('#F0A500')
LIGHT_BG  = HexColor('#F4F7FA')
MID_GREY  = HexColor('#5A6A7A')
DARK_TEXT = HexColor('#1A1A2E')
TABLE_ALT = HexColor('#EAF3F8')
WHITE_C   = HexColor('#FFFFFF')
LIGHT_NAVY= HexColor('#1A2D5A')
RED_DARK  = HexColor('#8B0000')
GREEN_DARK= HexColor('#006400')

PAGE_W, PAGE_H = letter
MARGIN    = 0.85 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


# ── Custom Flowables ──────────────────────────────────────────────────────────

class ColorBar(Flowable):
    def __init__(self, height=4, color=TEAL):
        super().__init__()
        self.bar_h = height
        self.color = color
    def wrap(self, avw, avh):
        self._w = avw
        return avw, self.bar_h
    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self._w, self.bar_h, stroke=0, fill=1)


class SectionHeader(Flowable):
    def __init__(self, title, subtitle=''):
        super().__init__()
        self.title    = title
        self.subtitle = subtitle
        self.height   = 52 if subtitle else 40
    def wrap(self, avw, avh):
        self._w = avw
        return avw, self.height
    def draw(self):
        c = self.canv
        w, h = self._w, self.height
        # Navy background
        c.setFillColor(NAVY)
        c.rect(0, 0, w, h, stroke=0, fill=1)
        # Gold left accent bar
        c.setFillColor(GOLD)
        c.rect(0, 0, 5, h, stroke=0, fill=1)
        # Teal right accent bar
        c.setFillColor(TEAL)
        c.rect(w - 5, 0, 5, h, stroke=0, fill=1)
        # Title text
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 13)
        c.drawString(14, h - 22, self.title)
        if self.subtitle:
            c.setFont('Helvetica', 9)
            c.setFillColor(TEAL)
            c.drawString(14, h - 38, self.subtitle)


class StatBlock(Flowable):
    def __init__(self, stats):
        super().__init__()
        self.stats = stats  # list of (value, label) tuples, max 4
        self.height = 70
    def wrap(self, avw, avh):
        self._w = avw
        return avw, self.height
    def draw(self):
        c = self.canv
        n = len(self.stats)
        col_w = self._w / n
        for i, (val, lbl) in enumerate(self.stats):
            x = i * col_w
            # Alternating navy / light_navy
            bg = NAVY if i % 2 == 0 else LIGHT_NAVY
            c.setFillColor(bg)
            c.rect(x, 0, col_w - 2, self.height, stroke=0, fill=1)
            # Gold value
            c.setFillColor(GOLD)
            c.setFont('Helvetica-Bold', 16)
            c.drawCentredString(x + col_w / 2, self.height - 28, val)
            # White label
            c.setFillColor(WHITE_C)
            c.setFont('Helvetica', 7)
            words = lbl.split()
            line1 = ' '.join(words[:3])
            line2 = ' '.join(words[3:]) if len(words) > 3 else ''
            c.drawCentredString(x + col_w / 2, self.height - 42, line1)
            if line2:
                c.drawCentredString(x + col_w / 2, self.height - 52, line2)


class ScenarioBox(Flowable):
    def __init__(self, label, title, lines):
        super().__init__()
        self.label  = label
        self.title  = title
        self.lines  = lines
        self.height = 30 + len(lines) * 16 + 10
    def wrap(self, avw, avh):
        self._w = avw
        return avw, self.height
    def draw(self):
        c = self.canv
        w, h = self._w, self.height
        # Teal header bar
        c.setFillColor(TEAL)
        c.rect(0, h - 30, w, 30, stroke=0, fill=1)
        # Gold label pill
        c.setFillColor(GOLD)
        c.roundRect(8, h - 24, 55, 18, 4, stroke=0, fill=1)
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 7)
        c.drawCentredString(35, h - 16, self.label)
        # White title
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 10)
        c.drawString(72, h - 19, self.title)
        # White body bg
        c.setFillColor(WHITE_C)
        c.rect(0, 0, w, h - 30, stroke=0, fill=1)
        # Light border
        c.setStrokeColor(TEAL)
        c.setLineWidth(1)
        c.rect(0, 0, w, h, stroke=1, fill=0)
        # Body lines
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 8.5)
        y = h - 46
        for line in self.lines:
            c.drawString(10, y, line)
            y -= 16


class CalloutBox(Flowable):
    def __init__(self, text, icon='!'):
        super().__init__()
        self.text = text
        self.icon = icon
        self.height = 50
    def wrap(self, avw, avh):
        self._w = avw
        return avw, self.height
    def draw(self):
        c = self.canv
        w, h = self._w, self.height
        c.setFillColor(LIGHT_BG)
        c.rect(0, 0, w, h, stroke=0, fill=1)
        c.setFillColor(GOLD)
        c.rect(0, 0, 6, h, stroke=0, fill=1)
        c.setFillColor(GOLD)
        c.setFont('Helvetica-Bold', 14)
        c.drawString(14, h - 22, self.icon)
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 9)
        # Wrap text
        words = self.text.split()
        line, lines = '', []
        for w_tok in words:
            test = (line + ' ' + w_tok).strip()
            if c.stringWidth(test, 'Helvetica', 9) < w - 50:
                line = test
            else:
                lines.append(line)
                line = w_tok
        lines.append(line)
        y = h - 18
        for ln in lines[:3]:
            c.drawString(34, y, ln)
            y -= 13


# ── Page Callbacks ────────────────────────────────────────────────────────────

def draw_cover(canvas, doc):
    canvas.saveState()
    w, h = PAGE_W, PAGE_H
    # Full navy background
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, h, stroke=0, fill=1)

    # Teal diagonal accent band
    from reportlab.graphics.shapes import Polygon
    canvas.setFillColor(TEAL_DARK)
    p = canvas.beginPath()
    p.moveTo(0, h * 0.55)
    p.lineTo(w * 0.6, h * 0.72)
    p.lineTo(w * 0.6, h * 0.68)
    p.lineTo(0, h * 0.51)
    p.close()
    canvas.drawPath(p, stroke=0, fill=1)

    # Gold accent strip at top
    canvas.setFillColor(GOLD)
    canvas.rect(0, h - 10, w, 10, stroke=0, fill=1)

    # Teal strip below gold
    canvas.setFillColor(TEAL)
    canvas.rect(0, h - 18, w, 8, stroke=0, fill=1)

    # QBITEL BRIDGE wordmark
    canvas.setFillColor(GOLD)
    canvas.setFont('Helvetica-Bold', 28)
    canvas.drawString(MARGIN, h - 80, 'QBITEL BRIDGE')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica-Bold', 13)
    canvas.drawString(MARGIN, h - 100, 'AVIATION & AEROSPACE SECURITY PLATFORM')

    # Divider
    canvas.setStrokeColor(GOLD)
    canvas.setLineWidth(2)
    canvas.line(MARGIN, h - 110, w - MARGIN, h - 110)

    # Main headline
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 22)
    canvas.drawString(MARGIN, h - 160, 'Quantum-Safe Air Traffic')
    canvas.drawString(MARGIN, h - 188, 'and Avionics Protection')

    # Subheadline
    canvas.setFillColor(TABLE_ALT)
    canvas.setFont('Helvetica', 11)
    canvas.drawString(MARGIN, h - 218, 'ADS-B Authentication  |  Bandwidth-Optimized PQC  |  DO-326A Compliance')
    canvas.drawString(MARGIN, h - 234, 'ARINC 653 Security Partition  |  Ground-Only Deployment')

    # Key threat statement
    canvas.setFillColor(GOLD)
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(MARGIN, h - 270, 'The safest industry in the world has no authentication on its')
    canvas.drawString(MARGIN, h - 284, 'primary surveillance system. QBITEL Bridge changes that.')

    # Aircraft silhouette decorative lines
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(0.5)
    for i in range(6):
        y_pos = h * 0.3 + i * 20
        canvas.line(MARGIN, y_pos, w - MARGIN, y_pos + 5)

    # Bottom stat boxes: 4 key metrics
    stats = [
        ('180K+', 'Daily Unauthenticated ADS-B Messages'),
        ('$20', 'Cost to Spoof Aircraft Position'),
        ('60-80%', 'PQC Signature Compression'),
        ('Zero', 'Aircraft Changes Required'),
    ]
    box_w = (w - 2 * MARGIN - 9) / 4
    y_box = MARGIN + 50
    for i, (val, lbl) in enumerate(stats):
        x = MARGIN + i * (box_w + 3)
        bg = LIGHT_NAVY if i % 2 == 0 else TEAL_DARK
        canvas.setFillColor(bg)
        canvas.roundRect(x, y_box, box_w, 65, 4, stroke=0, fill=1)
        canvas.setFillColor(GOLD)
        canvas.setFont('Helvetica-Bold', 18)
        canvas.drawCentredString(x + box_w / 2, y_box + 38, val)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 7)
        words = lbl.split()
        line1 = ' '.join(words[:3])
        line2 = ' '.join(words[3:]) if len(words) > 3 else ''
        canvas.drawCentredString(x + box_w / 2, y_box + 22, line1)
        if line2:
            canvas.drawCentredString(x + box_w / 2, y_box + 10, line2)

    # DO-326A badge
    canvas.setFillColor(TEAL)
    canvas.roundRect(w - MARGIN - 130, y_box + 75, 130, 22, 4, stroke=0, fill=1)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 8)
    canvas.drawCentredString(w - MARGIN - 65, y_box + 84, 'DO-326A Evidence Ready')

    # Contact strip at very bottom
    canvas.setFillColor(TEAL_DARK)
    canvas.rect(0, 0, w, MARGIN + 42, stroke=0, fill=1)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 9)
    canvas.drawString(MARGIN, 22, 'enterprise@qbitel.com   |   https://bridge.qbitel.com')

    canvas.restoreState()


def draw_page(canvas, doc):
    canvas.saveState()
    w, h = PAGE_W, PAGE_H
    # Navy header strip
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 0.45 * inch, w, 0.45 * inch, stroke=0, fill=1)
    canvas.setFillColor(GOLD)
    canvas.rect(0, h - 0.48 * inch, w, 3, stroke=0, fill=1)
    # Header text
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(MARGIN, h - 0.3 * inch, 'QBITEL BRIDGE')
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(TEAL)
    canvas.drawString(MARGIN + 1.1 * inch, h - 0.3 * inch, '— Aviation & Aerospace Security Platform')
    # Page number (right)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 8)
    page_num = doc.page
    canvas.drawRightString(w - MARGIN, h - 0.3 * inch, f'Page {page_num}')

    # Teal footer
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, w, 0.55 * inch, stroke=0, fill=1)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 7.5)
    canvas.drawCentredString(w / 2, 0.32 * inch, 'QBITEL BRIDGE — AVIATION & AEROSPACE')
    canvas.setFont('Helvetica', 7)
    canvas.drawString(MARGIN, 0.16 * inch, 'enterprise@qbitel.com  |  https://bridge.qbitel.com')
    canvas.drawRightString(w - MARGIN, 0.16 * inch, 'Confidential — Not for Public Distribution')
    canvas.restoreState()


# ── Paragraph Styles ──────────────────────────────────────────────────────────

def make_styles():
    styles = {}
    styles['body'] = ParagraphStyle('body',
        fontName='Helvetica', fontSize=9.5, leading=15,
        textColor=DARK_TEXT, spaceAfter=6, alignment=TA_JUSTIFY)
    styles['body_l'] = ParagraphStyle('body_l',
        fontName='Helvetica', fontSize=9.5, leading=15,
        textColor=DARK_TEXT, spaceAfter=4, alignment=TA_LEFT)
    styles['h2'] = ParagraphStyle('h2',
        fontName='Helvetica-Bold', fontSize=13, leading=17,
        textColor=NAVY, spaceBefore=14, spaceAfter=6)
    styles['h3'] = ParagraphStyle('h3',
        fontName='Helvetica-Bold', fontSize=11, leading=15,
        textColor=TEAL_DARK, spaceBefore=10, spaceAfter=4)
    styles['h4'] = ParagraphStyle('h4',
        fontName='Helvetica-Bold', fontSize=9.5, leading=13,
        textColor=NAVY, spaceBefore=6, spaceAfter=3)
    styles['bullet'] = ParagraphStyle('bullet',
        fontName='Helvetica', fontSize=9, leading=14,
        textColor=DARK_TEXT, leftIndent=14, spaceAfter=3,
        bulletIndent=4, bulletFontName='Helvetica', bulletFontSize=9)
    styles['caption'] = ParagraphStyle('caption',
        fontName='Helvetica-Oblique', fontSize=8, leading=11,
        textColor=MID_GREY, spaceAfter=4, alignment=TA_CENTER)
    styles['label'] = ParagraphStyle('label',
        fontName='Helvetica-Bold', fontSize=8, leading=11,
        textColor=TEAL_DARK, spaceAfter=2)
    styles['contact'] = ParagraphStyle('contact',
        fontName='Helvetica-Bold', fontSize=10, leading=15,
        textColor=NAVY, spaceAfter=4, alignment=TA_CENTER)
    styles['quote'] = ParagraphStyle('quote',
        fontName='Helvetica-Oblique', fontSize=10, leading=15,
        textColor=TEAL_DARK, leftIndent=20, rightIndent=20,
        spaceBefore=6, spaceAfter=8, alignment=TA_JUSTIFY)
    return styles


def tbl_style(has_header=True):
    cmds = [
        ('BACKGROUND',  (0, 0), (-1, 0 if has_header else -1), NAVY if has_header else LIGHT_BG),
        ('TEXTCOLOR',   (0, 0), (-1, 0), WHITE_C if has_header else DARK_TEXT),
        ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, 0), 8),
        ('FONTNAME',    (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',    (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('TEXTCOLOR',   (0, 1), (-1, -1), DARK_TEXT),
        ('GRID',        (0, 0), (-1, -1), 0.4, MID_GREY),
        ('TOPPADDING',  (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING',(0, 0), (-1, -1), 6),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
    ]
    return TableStyle(cmds)


def sp(n=6):
    return Spacer(1, n)


# ── Content Builders ──────────────────────────────────────────────────────────

def build_executive_summary(story, S):
    story.append(SectionHeader('EXECUTIVE SUMMARY',
        'The Safest Industry in the World Has No Authentication on Its Primary Surveillance System'))
    story.append(sp(10))
    story.append(Paragraph(
        'Aviation has achieved an extraordinary safety record through decades of rigorous engineering, '
        'redundant systems, independent oversight, and a culture of relentless improvement. Yet this '
        'same industry — which mandates triple-redundant hydraulics and requires independent verification '
        'of every line of flight-critical software — broadcasts the precise position of every aircraft '
        'on earth over an open radio channel with no authentication, no encryption, and no mechanism '
        'to distinguish a real aircraft from a spoofed one.', S['body']))
    story.append(Paragraph(
        'ADS-B (Automatic Dependent Surveillance-Broadcast) is the backbone of modern air traffic '
        'surveillance, mandated by the FAA, EASA, ICAO, and virtually every civil aviation authority '
        'worldwide. It was designed as an open broadcast system in the 1990s and 2000s when '
        'cryptographic authentication of 1090 MHz broadcasts was considered impractical. '
        'Today, a $20 software-defined radio is all that is required to inject ghost aircraft '
        'into the global air traffic picture.', S['body']))

    story.append(sp(8))
    # Key stats block
    story.append(StatBlock([
        ('180K+', 'Daily ADS-B Messages — Zero Authentication'),
        ('$20', 'Cost to Spoof Aircraft Position with SDR'),
        ('60-80%', 'PQC Signature Compression Achieved'),
        ('36 Wks', 'Typical Full Deployment Timeline'),
    ]))
    story.append(sp(10))

    story.append(Paragraph('What QBITEL Bridge Delivers', S['h3']))
    deliverables = [
        ('Cryptographic ADS-B Authentication',
         'Ground-receiver-level authentication with zero aircraft modifications or DO-178C recertification.'),
        ('Bandwidth-Optimized PQC',
         'Post-quantum cryptography compressed 60-80% to fit LDACS (600 bps) and ACARS (2.4 kbps) constraints.'),
        ('AI-Driven Spoofing Detection',
         'Multilateration cross-validation detects ghost aircraft within 10 seconds of injection.'),
        ('ARINC 653 Security Partition',
         'Isolated airborne security partition for IMA platforms — DAL-D certified, no safety partition impact.'),
        ('DO-326A/DO-178C Evidence Generation',
         'Complete Secure Development Assurance documentation accepted by FAA and EASA.'),
        ('ATC Network Quantum Hardening',
         'Quantum-safe VPN mesh protecting EUROCONTROL, NATS, FAA SWIM infrastructure.'),
    ]
    for title, body in deliverables:
        story.append(Paragraph(f'<b>{title}</b> — {body}', S['bullet']))
    story.append(sp(8))
    story.append(CalloutBox(
        'NIST finalized PQC standards August 2024. U.S. government mandates PQC migration by 2035. '
        'Aircraft certified today will operate until 2060-2065. Act now.', icon='Q'))


def build_threats(story, S):
    story.append(PageBreak())
    story.append(SectionHeader('THE THREE CRITICAL THREATS',
        'ADS-B Spoofing | Bandwidth-Constrained Quantum Exposure | Certification Barrier'))
    story.append(sp(10))

    # Threat 1
    story.append(Paragraph('Threat 1: ADS-B Spoofing and Ghost Aircraft', S['h2']))
    story.append(ColorBar(3, GOLD))
    story.append(sp(6))
    story.append(Paragraph(
        'ADS-B OUT requires every aircraft above certain altitudes to continuously broadcast its ICAO '
        '24-bit address, callsign, GPS position, altitude, velocity, and emergency status on 1090 MHz — '
        'a frequency accessible to any radio receiver. There is no authentication field in the ADS-B '
        'message format. Any transmitter broadcasting correctly formatted messages will be accepted by '
        'ground stations and displayed on ATC radar screens.', S['body']))

    story.append(Paragraph('Attack Vector', S['h4']))
    attack_steps = [
        'Monitor legitimate ADS-B transmissions to understand traffic patterns',
        'Generate spoofed ADS-B messages for non-existent ghost aircraft ($20 SDR + open-source software)',
        'Inject false position updates for real aircraft — position falsification',
        'Create phantom traffic that triggers TCAS resolution advisories in real aircraft',
        'Gradually shift an aircraft\'s apparent position to induce air traffic controller errors',
    ]
    for step in attack_steps:
        story.append(Paragraph(f'• {step}', S['bullet']))

    story.append(sp(6))
    # Threat impact table
    tdata = [
        ['Threat Vector', 'Attack Cost', 'Current Detectability', 'QBITEL Response'],
        ['ADS-B ghost aircraft injection', '$20 SDR', 'None — no authentication', 'Detection <10 sec'],
        ['ADS-B position falsification', '$20 SDR', 'None', 'Detection <30 sec'],
        ['ACARS message forgery', '$500 SDR+decoder', 'None', 'Real-time PQC auth'],
        ['CPDLC clearance spoofing', 'Sophisticated', 'Limited', 'PQC + session binding'],
        ['Harvest-now-decrypt-later', 'Passive collection', 'None', 'PQC eliminates threat'],
    ]
    t = Table(tdata, colWidths=[CONTENT_W * 0.30, CONTENT_W * 0.15,
                                 CONTENT_W * 0.28, CONTENT_W * 0.27])
    t.setStyle(tbl_style())
    story.append(t)
    story.append(sp(12))

    # Threat 2
    story.append(Paragraph('Threat 2: Bandwidth-Constrained Quantum Exposure', S['h2']))
    story.append(ColorBar(3, GOLD))
    story.append(sp(6))
    story.append(Paragraph(
        'Standard post-quantum signatures are enormous by aviation standards. A raw ML-DSA-65 '
        '(Dilithium-3) signature is 3,309 bytes. At 2,400 bps ACARS, this requires 11 seconds '
        'to transmit — an eternity for time-sensitive operational communications. Aviation data '
        'links operate at bandwidths that would seem impossibly restrictive to enterprise IT professionals.', S['body']))

    bw_data = [
        ['Data Link', 'Bandwidth', 'Primary Use', 'Raw PQC Overhead', 'Compressed Overhead'],
        ['LDACS (next-gen)', '600 bps–2.4 kbps', 'Air-ground digital', 'Too large', '<15% Falcon-512'],
        ['VHF ACARS', '2,400 bps', 'Operational comms', '1,100%', '<8% session cache'],
        ['HF ACARS', '300–1,800 bps', 'Oceanic comms', 'Too large', '<12% session-based'],
        ['SATCOM Iridium', '600 bps–128 kbps', 'Global coverage', '1,100%', '<8% with caching'],
        ['SATCOM Inmarsat', '1.2–432 kbps', 'Broadband ops', '252%', '<3% full PQC'],
        ['ATC Networks', 'Gbps Ethernet', 'Ground systems', '<0.1%', 'Not required'],
    ]
    t2 = Table(bw_data, colWidths=[CONTENT_W * 0.18, CONTENT_W * 0.17,
                                    CONTENT_W * 0.18, CONTENT_W * 0.20, CONTENT_W * 0.27])
    t2.setStyle(tbl_style())
    story.append(t2)
    story.append(sp(6))
    story.append(Paragraph(
        'QBITEL Bridge achieves 60-80% signature compression through algorithm-specific lossless '
        'compression exploiting lattice signature structure, context-aware delta encoding, session '
        'key amortization, hierarchical signing with time-bounded session certificates, and selective '
        'field authentication. Compressed Falcon-512 fits LDACS frame constraints. '
        'Compressed Dilithium-3 adds less than 0.8 seconds latency on 2,400 bps ACARS.', S['body']))
    story.append(sp(12))

    # Threat 3
    story.append(Paragraph('Threat 3: The DO-178C Certification Barrier', S['h2']))
    story.append(ColorBar(3, GOLD))
    story.append(sp(6))
    story.append(Paragraph(
        'Aviation software certification under DO-178C is the most rigorous software quality process '
        'in any industry. For Design Assurance Level A (catastrophic failure conditions), DO-178C '
        'requires full requirements traceability, Modified Condition/Decision Coverage (MC/DC) for '
        'every decision point, independent verification of every test procedure, and comprehensive '
        'documentation of every design decision. The cost of DO-178C Level A certification for a '
        'new avionics function ranges from $50M to $500M per aircraft type.', S['body']))
    story.append(Paragraph(
        'This creates a security paradox: the most safety-conscious industry in the world is '
        'structurally prevented from rapidly deploying security updates. Boeing 737 MAX aircraft '
        'delivered in 2024 will likely still be in service in 2055-2060. The cryptographic '
        'algorithms protecting their communications must remain secure for their entire operational life.', S['body']))
    story.append(Paragraph(
        '<b>QBITEL\'s Response:</b> The primary deployment architecture is <b>ground-only</b> — '
        'no changes to certified airborne software, eliminating the DO-178C recertification '
        'requirement entirely. For airborne deployment, QBITEL provides a DAL-D ARINC 653 '
        'security partition with complete DO-178C evidence generation.', S['body']))


def build_capabilities(story, S):
    story.append(PageBreak())
    story.append(SectionHeader('SEVEN CORE CAPABILITIES',
        'ADS-B Auth | Bandwidth PQC | LDACS/ACARS/SATCOM | ARINC 653 | Ground-Only | DO-326A | ATC Networks'))
    story.append(sp(10))

    caps = [
        ('1', 'ADS-B Authentication and Anti-Spoofing',
         [('Layer 1 — ICAO Address Binding',
           'Aircraft ICAO 24-bit addresses cryptographically bound to operator certificates in distributed aviation PKI.'),
          ('Layer 2 — Message Authentication Codes',
           'MACs computed and verified at ground stations via transponder-adjacent hardware — no airborne software certification.'),
          ('Layer 3 — Behavioral Authentication',
           'AI engine performs behavioral authentication for non-participating aircraft, verifying position consistency with performance envelopes.'),
          ('Layer 4 — Cross-Source Validation',
           'ADS-B cross-validated against SSR Mode S radar, MLAT calculations, and historical flight profile data.'),
         ],
         [('Auth Latency', '<50 ms'), ('Throughput', '500K+ msg/hr'), ('False Positive', '<0.001%'), ('Detect Rate', '>99.7%')]),

        ('2', 'Bandwidth-Optimized PQC for Constrained Data Links',
         [('Algorithm-Specific Compression',
           'Exploit mathematical structure of lattice signatures for 60-80% lossless compression unique to each algorithm.'),
          ('Context-Aware Delta Encoding',
           'Aviation messages follow predictable patterns; QBITEL transmits cryptographic deltas rather than full signatures.'),
          ('Session Key Amortization',
           'Single expensive PQC handshake; subsequent messages use lightweight symmetric MACs on the same link.'),
          ('Hierarchical Signing',
           'Ground infrastructure signs time-bounded session certificates for rapid per-message authentication without per-message PQC overhead.'),
         ],
         [('Dilithium-3 Reduction', '60-80%'), ('Falcon-512 Size', '359-538B'), ('ACARS Overhead', '<0.8 sec'), ('LDACS Compliance', 'Yes')]),

        ('3', 'LDACS / ACARS / SATCOM Security',
         [('LDACS (Next-Gen Air-Ground)',
           'Quantum-safe key establishment, CPDLC message authentication, ATC clearance protection, privacy for L-band transmissions.'),
          ('VHF/HF ACARS (Legacy)',
           'Compressed PQC authentication, end-to-end integrity, non-repudiation for maintenance write-ups and MEL deferrals.'),
          ('SATCOM (Iridium and Inmarsat)',
           'Bandwidth-adaptive PQC selection, hybrid encryption for VoIP SATCOM, ACARS-over-SATCOM (AOS) stream protection.'),
          ('CPDLC Protection',
           'Controller-Pilot Data Link Communications secured against clearance spoofing with session-bound PQC authentication.'),
         ],
         [('Link Types Secured', '6+'), ('Bandwidth Adapt', 'Yes'), ('AOS Support', 'Yes'), ('VoIP Encrypt', 'Yes')]),

        ('4', 'ARINC 653 Security Partition (Airborne)',
         [('Partition Isolation',
           'Zero shared memory with flight-critical partitions. Strict time allocation cannot delay flight-critical processing.'),
          ('Partition Functions',
           'Local key storage, PQC computation, authentication verification, security event logging, certificate management.'),
          ('DO-178C DAL-D Certification',
           'Partition failure cannot propagate to safety-critical systems due to ARINC 653 isolation guarantees.'),
          ('Evidence Package',
           'Complete HLR/LLR, Software Architecture, source code traceability, unit tests, MC/DC coverage, tool qualification.'),
         ],
         [('IMA Platforms', '787, A380, A350'), ('DAL Level', 'DAL-D'), ('Isolation', 'ARINC 653'), ('Cert Evidence', 'Full DO-178C')]),

        ('5', 'Ground-System-Only Deployment',
         [('No Aircraft Changes',
           'ADS-B spoofing attacks occur at the ground receiver. Authentication at the ground receiver intercepts spoofed messages without aircraft involvement.'),
          ('No DO-178C Recertification',
           'Ground-only deployment triggers no airborne software change — the most expensive and time-consuming aspect of aviation security.'),
          ('Complete Coverage',
           'ADS-B authentication, ATC network security, ACARS ground side, SATCOM ground side, airport surface surveillance — all without aircraft modification.'),
          ('Immediate Deployment',
           'Ground-only deployment can be initiated within weeks. No aircraft access, maintenance slots, or airworthiness approvals required.'),
         ],
         [('Aircraft Changes', 'None'), ('Recertification', 'None'), ('Deploy Time', 'Weeks'), ('ADS-B Coverage', 'Full')]),

        ('6', 'DO-326A / DO-178C Compliance Evidence Generation',
         [('DO-326A Documentation',
           'Aircraft Security Log (ASL), Security Target Analysis (STA), Security Development Analysis (SDA), Security Assessment Report.'),
          ('DO-178C Evidence (Airborne)',
           'Software Plans, Development artifacts (HLR/LLR, Architecture, Design, Source Code), Verification (tests, results, MC/DC coverage).'),
          ('FAA and EASA Alignment',
           'Evidence structured for FAA Aircraft Certification Service Issue Papers and EASA Certification Review Items (CRIs).'),
          ('Ongoing Compliance',
           'Continuous evidence generation throughout system lifecycle. Configuration management integration ensures evidence currency.'),
         ],
         [('DO-326A Docs', 'Full ASL/STA/SDA'), ('DO-178C Evidence', 'Complete'), ('FAA Alignment', 'Yes'), ('EASA Alignment', 'Yes')]),

        ('7', 'ATC Network Quantum Hardening',
         [('SWIM Network Protection',
           'Quantum-safe VPN mesh for FAA SWIM NAS and EUROCONTROL SWIM B2B APIs — securing NextGen/SESAR information sharing.'),
          ('Inter-Facility Communications',
           'Quantum-safe encryption for AIDC (ATS Inter-facility Data Communications) and OLDI (On-Line Data Interchange) links.'),
          ('AFTN/AMHS Gateway Security',
           'Quantum-safe wrapping for Aeronautical Fixed Telecommunication Network and Aeronautical Message Handling System traffic.'),
          ('HSM and SIEM Integration',
           'Hardware Security Modules for ATC cryptographic key storage. SIEM integration with aviation-specific threat intelligence feeds.'),
         ],
         [('Facilities Protected', 'ACC, TRACON, Tower'), ('HSM Integration', 'Yes'), ('SIEM Integration', 'Yes'), ('SWIM Support', 'Yes')]),
    ]

    for num, title, details, metrics in caps:
        story.append(KeepTogether([
            Paragraph(f'Capability {num}: {title}', S['h2']),
            ColorBar(3, TEAL),
            sp(6),
        ]))
        for sub_title, sub_body in details:
            story.append(Paragraph(f'<b>{sub_title}:</b> {sub_body}', S['body_l']))
        story.append(sp(4))
        # Metrics row
        metric_data = [['Metric', 'Value'] * 2]
        row = []
        for k, v in metrics:
            row.extend([k, v])
        metric_data.append(row)
        mt = Table([list(map(str, row)) for row in metric_data],
                   colWidths=[CONTENT_W * 0.17, CONTENT_W * 0.20,
                              CONTENT_W * 0.17, CONTENT_W * 0.20])
        mt.setStyle(TableStyle([
            ('BACKGROUND',  (0, 0), (-1, 0), TEAL),
            ('TEXTCOLOR',   (0, 0), (-1, 0), WHITE_C),
            ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',    (0, 0), (-1, -1), 7.5),
            ('FONTNAME',    (0, 1), (-1, -1), 'Helvetica'),
            ('BACKGROUND',  (0, 1), (-1, 1), TABLE_ALT),
            ('GRID',        (0, 0), (-1, -1), 0.3, MID_GREY),
            ('TOPPADDING',  (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(mt)
        story.append(sp(14))


def build_compliance(story, S):
    story.append(PageBreak())
    story.append(SectionHeader('COMPLIANCE COVERAGE',
        'DO-326A | DO-178C | FAA AC 119-1 | EASA CS-STAN | ICAO Annex 10 | NIST FIPS 203/204/205'))
    story.append(sp(10))
    story.append(Paragraph(
        'QBITEL Bridge provides complete regulatory compliance coverage for aviation security certification '
        'across all major jurisdictions and standards bodies. The following table maps each applicable '
        'standard to QBITEL\'s coverage scope and the specific evidence artifacts generated.', S['body']))
    story.append(sp(8))
    comp_data = [
        ['Regulation / Standard', 'Scope', 'QBITEL Coverage', 'Evidence Provided'],
        ['DO-326A (ED-202A)', 'Airworthiness Security Process', 'Full', 'ASL, STA, SDA, SAR'],
        ['DO-356A (ED-203A)', 'Security Methods and Considerations', 'Full', 'Method selection docs'],
        ['DO-178C (ED-12C)', 'Airborne Software Quality', 'Partition-scoped', 'Full Plan/Dev/Verify'],
        ['DO-254 (ED-80)', 'Airborne Electronic Hardware', 'Interface documentation', 'HW-SW interface docs'],
        ['FAA AC 119-1', 'Aircraft Network Security Program', 'Full', 'ANSP documentation'],
        ['EASA CS-STAN', 'Standard Changes and Repairs', 'Applicable standards', 'Change justification'],
        ['ICAO Annex 10', 'Aeronautical Telecommunications', 'Alignment', 'Technical compliance'],
        ['EUROCAE ED-205', 'Aviation Wireless Cyber Security', 'Full', 'Compliance mapping'],
        ['RTCA SC-216 / DO-260C', 'ADS-B MOPS', 'Security extensions', 'Technical interface doc'],
        ['NIST FIPS 203 / 204 / 205', 'Post-Quantum Cryptography Stds', 'Full implementation', 'Algorithm cert docs'],
        ['NIST SP 800-208', 'PQC Key Management', 'Full', 'Key management procedures'],
        ['Common Criteria EAL4+', 'Security Product Evaluation', 'Full CC package', 'CC evaluation docs'],
    ]
    t = Table(comp_data, colWidths=[CONTENT_W * 0.25, CONTENT_W * 0.28,
                                     CONTENT_W * 0.18, CONTENT_W * 0.29])
    t.setStyle(tbl_style())
    story.append(t)
    story.append(sp(10))
    story.append(CalloutBox(
        'All compliance documentation is generated automatically by BRIDGE-AV-CERT and is '
        'formatted for direct submission to FAA, EASA, CAAC, and national aviation authorities.', icon='C'))


def build_integration(story, S):
    story.append(PageBreak())
    story.append(SectionHeader('INTEGRATION ECOSYSTEM',
        'Honeywell | Thales | Collins Aerospace | SITA | NATS | FAA/EUROCONTROL'))
    story.append(sp(10))

    partners = [
        ('Avionics OEM Partners', [
            ('Honeywell Aerospace',
             'Connected Aircraft platform integration. APEX IMA platform compatibility. '
             'Primus avionics suite integration. HTR system interfaces for spoofing alert correlation.'),
            ('Thales Avionics',
             'IMA platform (TopTech) security partition integration. FLYSMART+ operational support '
             'system security. AVMS network protection. TCAS/ACAS integration for spoofing alerts.'),
            ('Collins Aerospace',
             'Pro Line Fusion avionics suite integration. GLOBALink SATCOM security enhancement. '
             'MultiScan weather radar data link protection. ARINC 429 interface security gateway.'),
        ]),
        ('Air Navigation Service Providers', [
            ('NATS (UK)',
             'NERC interface and iCAS integration. Swanwick and Prestwick Area Control Centre '
             'deployment support. NERC engineering coordination for receiver network authentication.'),
            ('EUROCONTROL',
             'SWIM Yellow Profile security extension. B2B API quantum protection. '
             'CFMU network security and Network Manager interface.'),
            ('FAA (USA)',
             'SWIM NAS integration for NextGen. TFMS (Traffic Flow Management System) network '
             'protection. STARS (Standard Terminal Automation Replacement System) security overlay.'),
        ]),
        ('Communication and Data Service Providers', [
            ('SITA',
             'ACARS quantum-safe upgrade path for AviNet. Global aviation network security for '
             'WorldTracer and baggage system infrastructure. Airport community systems protection.'),
            ('ARINC (Collins Aerospace)',
             'GlobalLink SATCOM security enhancement. ARINC CDN (Content Delivery Network) '
             'protection. Ground network quantum hardening for ARINC 429/629/717 gateways.'),
            ('Inmarsat and VSAT Providers',
             'SwiftBroadband quantum-safe session protection. Classic Aero and Swift64 '
             'backward-compatible security. GX Aviation (Ka-band) full PQC implementation.'),
        ]),
    ]

    for section_title, items in partners:
        story.append(Paragraph(section_title, S['h3']))
        story.append(ColorBar(2, TEAL))
        story.append(sp(6))
        for partner, desc in items:
            story.append(Paragraph(f'<b>{partner}:</b> {desc}', S['body_l']))
        story.append(sp(8))


def build_deployment(story, S):
    story.append(PageBreak())
    story.append(SectionHeader('DEPLOYMENT TIMELINE AND PERFORMANCE',
        '36-Week Deployment | 99.999% Availability | <50ms Authentication Latency'))
    story.append(sp(10))

    story.append(Paragraph('Phased Deployment Timeline', S['h3']))
    phases = [
        ('Phase 0', 'Safety Assessment and Gap Analysis (Weeks 1-4)',
         'DO-326A threat assessment initiation. Protocol inventory and security gap analysis. '
         'ANSP/CAA notification and regulatory engagement. ADS-B receiver network survey. '
         'Bandwidth measurement and PQC compression requirement determination.'),
        ('Phase 1', 'Ground Infrastructure Deployment (Weeks 5-12)',
         'QBITEL Bridge ground node deployment at primary ATC facilities. ADS-B authentication '
         'infrastructure at receiver sites. PKI certificate authority establishment. '
         'ATC network security assessment and quantum-safe VPN deployment.'),
        ('Phase 2', 'Authentication Network Enrollment (Weeks 13-20)',
         'Aircraft operator enrollment in authentication PKI. Ground receiver software update '
         'for authentication verification. ATC display integration for authentication status. '
         'ACARS gateway quantum protection at AOCs. SATCOM ground infrastructure enhancement.'),
        ('Phase 3', 'AI Spoofing Detection Activation (Weeks 21-28)',
         'Multilateration receiver network integration and calibration. AI model training on '
         'local traffic patterns and aircraft performance data. False positive tuning and '
         'threshold optimization. Controller training and HMI familiarization.'),
        ('Phase 4', 'Full Operational Status (Weeks 29-36)',
         'Authentication verification mandatory for enrolled aircraft. Spoofing detection '
         'alerts integrated into ATC workflow. DO-326A compliance documentation completed '
         'and submitted. SOC handover and monitoring activation.'),
    ]
    for phase_num, phase_title, phase_body in phases:
        story.append(KeepTogether([
            ScenarioBox(phase_num, phase_title, [
                phase_body[:80] + '...' if len(phase_body) > 80 else phase_body,
            ]),
            sp(6),
        ]))

    story.append(sp(10))
    story.append(Paragraph('Performance Specifications', S['h3']))
    story.append(ColorBar(3, GOLD))
    story.append(sp(8))

    # Authentication Performance
    story.append(Paragraph('Authentication Performance', S['h4']))
    auth_data = [
        ['Metric', 'Specification', 'Notes'],
        ['ADS-B Authentication Latency', '<50 ms', 'End-to-end, ground receiver to ATC display'],
        ['Message Throughput', '500,000+ msg/hour', 'Per QBITEL Bridge node'],
        ['False Positive Rate', '<0.001%', 'Fewer than 1 in 100,000 authenticated messages'],
        ['Spoofing Detection Rate', '>99.7%', 'Systematic spoofing attacks'],
        ['Ghost Aircraft Detection', '<10 seconds', 'Time from injection to ATC alert'],
        ['Position Falsification Detection', '<30 seconds', 'Gradual position shift attacks'],
        ['System Availability', '99.999%', 'Five nines — matching ATC reliability standards'],
        ['Automatic Failover Time', '<500 ms', 'Failover to secondary node'],
    ]
    t = Table(auth_data, colWidths=[CONTENT_W * 0.35, CONTENT_W * 0.25, CONTENT_W * 0.40])
    t.setStyle(tbl_style())
    story.append(t)
    story.append(sp(8))

    # Cryptographic Performance
    story.append(Paragraph('Cryptographic Algorithm Performance', S['h4']))
    crypto_data = [
        ['Algorithm', 'Sign Time', 'Verify Time', 'Raw Size', 'Compressed'],
        ['ML-DSA-65 (Dilithium-3)', '<10 ms', '<5 ms', '3,309 B', '659-1,317 B (60-80% reduction)'],
        ['Falcon-512', '<20 ms', '<2 ms', '897 B', '359-538 B (40-60% reduction)'],
        ['ML-KEM-768', '<1 ms', '<1 ms', '1,088 B', 'N/A (KEM)'],
        ['SLH-DSA-128f', '<50 ms', '<5 ms', '17,088 B', '6,835 B (stateless hash-based)'],
        ['AES-256-GCM', '<0.1 ms/KB', '<0.1 ms/KB', 'N/A', 'N/A (symmetric)'],
    ]
    t2 = Table(crypto_data, colWidths=[CONTENT_W * 0.23, CONTENT_W * 0.12,
                                        CONTENT_W * 0.12, CONTENT_W * 0.12, CONTENT_W * 0.41])
    t2.setStyle(tbl_style())
    story.append(t2)


def build_competitive(story, S):
    story.append(PageBreak())
    story.append(SectionHeader('COMPETITIVE DIFFERENTIATION',
        'vs. Radar-Only | vs. ICAO Working Groups | vs. Classical Vendors | vs. IT-Adapted Solutions'))
    story.append(sp(10))

    comparisons = [
        ('vs. Radar-Only Security (SSR Mode S)',
         'Argument: "We have secondary surveillance radar. We do not need ADS-B authentication."',
         'Reality: SSR Mode S provides valuable cross-checking but has no global oceanic coverage, '
         'is subject to electronic warfare and interference, provides lower resolution altitude '
         'encoding, and does not protect commercial flight tracking aggregators used by airlines '
         'and passengers. QBITEL Bridge adds AI-driven multilateration cross-validation that '
         'correlates ADS-B with SSR data in real time — enhancing rather than replacing radar.',
         TEAL),
        ('vs. ICAO Working Groups (Wait for Standardization)',
         'Argument: "We will wait for ICAO to standardize ADS-B authentication."',
         'Reality: ICAO ADS-B authentication working groups have been active for over a decade '
         'with no standardized solution deployed. Standardization timelines typically extend '
         '5-10 years from initial proposal to mandated implementation. During that period, '
         'the threat environment continues to evolve and quantum computers advance. '
         'QBITEL Bridge provides operational security now while positioning customers for '
         'seamless compliance with future ICAO mandates. Early adopters gain operational '
         'experience and certified infrastructure when standards mandate implementation.',
         TEAL_DARK),
        ('vs. Classical Security Vendors (RSA/ECDSA)',
         'Argument: "Our current cryptographic systems are adequate."',
         'Reality: RSA/ECDSA-based systems will be broken by quantum computers estimated '
         '2030-2040. Aircraft deployed today will still be flying then. "Crypto agility" '
         'promises from legacy vendors require software updates — and in airborne systems, '
         'software updates require DO-178C recertification. QBITEL Bridge implements '
         'NIST-standardized PQC algorithms (FIPS 203, 204, 205) from day one, eliminating '
         'the need for an emergency cryptographic migration that would require recertification '
         'of deployed airborne systems.',
         NAVY),
        ('vs. IT-Adapted Enterprise Security Solutions',
         'Argument: "We can adapt our existing enterprise security tools for aviation."',
         'Reality: General IT security vendors adapting enterprise products to aviation '
         'have no design experience for 600 bps bandwidth constraints, no understanding of '
         'DO-178C/DO-326A certification requirements, cannot generate aviation regulatory '
         'compliance evidence, have no experience integrating with ASTERIX, SDPS, and SWIM '
         'protocols, and have no operational precedent in safety-critical environments. '
         'QBITEL Bridge was architected for aviation from day one — not retrofitted from '
         'enterprise IT.',
         MID_GREY),
    ]

    for title, objection, reality, color in comparisons:
        story.append(Paragraph(title, S['h3']))
        story.append(ColorBar(2, color))
        story.append(sp(4))
        story.append(Paragraph(f'<i>{objection}</i>', S['quote']))
        story.append(Paragraph(reality, S['body']))
        story.append(sp(10))


def build_scenarios(story, S):
    story.append(PageBreak())
    story.append(SectionHeader('CUSTOMER SCENARIOS',
        'Air Navigation Service Provider | Aircraft OEM | Airline CISO'))
    story.append(sp(10))

    scenarios = [
        ('Scenario 1', 'Air Navigation Service Provider (ANSP)',
         ['Profile: Major ANSP — 25+ ATC facilities, 3,000+ daily flights, NextGen/SESAR modernization',
          'Challenge: 180K+ daily unauthenticated ADS-B messages, SWIM connections over internet infrastructure',
          'QBITEL: ADS-B auth at 47 receiver sites, quantum-safe VPN mesh, AI spoofing detection, SOC',
          'Outcome: National airspace ADS-B authentication within 6 months, <0.001% false positive rate',
          'Timeline: 28 weeks to Full Operational Status | Contact: enterprise@qbitel.com']),
        ('Scenario 2', 'Aircraft OEM — New Aircraft Program',
         ['Profile: Commercial OEM, new program, EIS 2027, service life to 2060-2065',
          'Challenge: 40-year security horizon, DO-178C scope minimization, multi-authority evidence',
          'QBITEL: ARINC 653 partition design from program inception, DAL-D DO-178C evidence package',
          'Outcome: Security at certification baseline (no STC), PQC comms from EIS, 40-year roadmap',
          'Timeline: Aligned with aircraft program schedule | Contact: certification@qbitel.com']),
        ('Scenario 3', 'Airline CISO — Operations Security Program',
         ['Profile: 400+ aircraft, 120+ destinations, 6 continents, diverse fleet B737/777/787, A320/330/350',
          'Challenge: FAA/EASA/CAA compliance, ACARS/SATCOM unsecured, harvest-now-decrypt-later threat',
          'QBITEL: AOC security infrastructure, ACARS gateway auth, SATCOM session PQC, ADS-B monitoring',
          'Outcome: All ACARS authenticated within 20 weeks, oceanic SATCOM quantum-safe, regulatory compliance',
          'Timeline: 20 weeks | Contact: enterprise@qbitel.com']),
    ]

    for label, title, lines in scenarios:
        story.append(ScenarioBox(label, title, lines))
        story.append(sp(10))


def build_contact(story, S):
    story.append(PageBreak())
    story.append(SectionHeader('NEXT STEPS AND CONTACT',
        'Schedule a Briefing | Request Assessment | Initiate DO-326A Gap Analysis'))
    story.append(sp(10))

    story.append(Paragraph('Why Act Now', S['h2']))
    reasons = [
        ('The Quantum Clock Is Running',
         'NIST finalized PQC standards August 2024. The U.S. government mandates PQC migration by 2035. '
         'Aircraft certified today will operate until 2060. The time to implement quantum-safe aviation '
         'security is at the beginning of new programs and network modernization initiatives — not when '
         'a regulatory mandate forces emergency implementation.'),
        ('The Threat Is Present, Not Future',
         'ADS-B spoofing with $20 hardware is not theoretical. It is a documented, demonstrated, '
         'currently-achievable attack. Every day without authentication is a day the aviation system '
         'operates on trust rather than cryptographic verification.'),
        ('Early Movers Gain Regulatory Advantage',
         'Aviation regulatory processes reward early engagement. Organizations beginning DO-326A '
         'compliance documentation today will be positioned ahead of competitors when authorities '
         'mandate aviation cybersecurity certification. QBITEL Bridge provides the compliance '
         'evidence infrastructure to support that advantage.'),
    ]
    for title, body in reasons:
        story.append(Paragraph(f'<b>{title}:</b> {body}', S['body']))
    story.append(sp(10))

    story.append(Paragraph('Immediate Actions Available', S['h3']))
    actions = [
        ('For Air Navigation Service Providers',
         'Schedule technical briefing | Request ADS-B vulnerability assessment | Initiate DO-326A gap analysis | Request non-operational demonstration'),
        ('For Aircraft OEMs and Avionics Integrators',
         'ARINC 653 partition architecture review | DO-178C evidence scope discussion | FAA/EASA Issue Paper strategy | Bandwidth optimization review'),
        ('For Airlines',
         'Airline CISO quantum threat briefing | ACARS/SATCOM security assessment | Regulatory compliance alignment review | Fleet security roadmap'),
        ('For Military and Defense Aviation',
         'Contact QBITEL defense aviation team | DoD PQC mandate alignment | ADS-B military airspace coexistence planning'),
    ]
    for audience, action_list in actions:
        story.append(Paragraph(f'<b>{audience}:</b> {action_list}', S['bullet']))
    story.append(sp(12))

    # Contact box
    story.append(ColorBar(4, GOLD))
    story.append(sp(8))
    story.append(Paragraph('Contact QBITEL Bridge Aviation Practice', S['contact']))
    story.append(sp(4))

    contact_data = [
        ['Enterprise Sales and Technical Inquiries', 'Aviation Compliance and Certification', 'SOC (24/7 Monitoring)'],
        ['enterprise@qbitel.com', 'certification@qbitel.com', 'soc@qbitel.com'],
        ['https://bridge.qbitel.com', 'DO-326A and DO-178C evidence support', '24/7 aviation security monitoring'],
    ]
    ct = Table(contact_data, colWidths=[CONTENT_W / 3] * 3)
    ct.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0), NAVY),
        ('BACKGROUND',  (0, 1), (-1, 1), TEAL),
        ('BACKGROUND',  (0, 2), (-1, 2), TABLE_ALT),
        ('TEXTCOLOR',   (0, 0), (-1, 1), WHITE_C),
        ('TEXTCOLOR',   (0, 2), (-1, 2), DARK_TEXT),
        ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',    (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',    (0, 0), (-1, -1), 9),
        ('ALIGN',       (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING',  (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 8),
        ('GRID',        (0, 0), (-1, -1), 0.5, WHITE_C),
    ]))
    story.append(ct)
    story.append(sp(12))
    story.append(ColorBar(4, GOLD))
    story.append(sp(8))
    story.append(Paragraph(
        'QBITEL Bridge — Securing Aviation\'s Future, Today', S['contact']))
    story.append(Paragraph(
        'Classification: Commercial — Not for Public Distribution  |  '
        'Version 2025.1 — Aviation and Aerospace Vertical  |  © 2025 QBITEL',
        ParagraphStyle('footer_note', fontName='Helvetica-Oblique',
                       fontSize=7.5, leading=11, textColor=MID_GREY,
                       alignment=TA_CENTER)))


# ── Main Build Function ────────────────────────────────────────────────────────

def build_doc(output_path):
    doc = BaseDocTemplate(
        output_path, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN)

    cover_frame = Frame(0, 0, PAGE_W, PAGE_H,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id='cover')
    inner_frame = Frame(MARGIN, 0.7 * inch, CONTENT_W,
        PAGE_H - MARGIN - 0.7 * inch, id='inner')

    cover_template = PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover)
    inner_template = PageTemplate(id='Inner', frames=[inner_frame], onPage=draw_page)
    doc.addPageTemplates([cover_template, inner_template])

    S = make_styles()
    story = []
    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())

    build_executive_summary(story, S)
    build_threats(story, S)
    build_capabilities(story, S)
    build_compliance(story, S)
    build_integration(story, S)
    build_deployment(story, S)
    build_competitive(story, S)
    build_scenarios(story, S)
    build_contact(story, S)

    doc.build(story)
    print(f"PDF written: {output_path}")


if __name__ == '__main__':
    import os
    out = '/Users/prabakarankannan/qbitel/docs/brochures/QBITEL_Bridge_Aviation_Marketing_Pitch.pdf'
    build_doc(out)
    size = os.path.getsize(out)
    print(f"File size: {size:,} bytes ({size/1024:.1f} KB)")
