"""
Build QBITEL Bridge Banking & Financial Services Marketing Pitch - Professional PDF
Uses ReportLab for full layout/design control.
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
        n = len(self.stats)
        box_w = self.w / n
        for i, (big, small) in enumerate(self.stats):
            x = i * box_w
            bg = TEAL if i % 2 == 0 else NAVY
            c.setFillColor(bg)
            c.rect(x, 0, box_w, self.h, fill=1, stroke=0)
            c.setFillColor(GOLD if bg == NAVY else WHITE_C)
            c.setFont('Helvetica-Bold', 20)
            tw = c.stringWidth(big, 'Helvetica-Bold', 20)
            c.drawString(x + (box_w - tw) / 2, self.h - 32, big)
            c.setFillColor(WHITE_C)
            c.setFont('Helvetica', 8)
            for j, line in enumerate(small.split('\n')):
                lw = c.stringWidth(line, 'Helvetica', 8)
                c.drawString(x + (box_w - lw) / 2, self.h - 48 - j * 11, line)


class ScenarioBox(Flowable):
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
        c.setFillColor(LIGHT_BG)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        c.setFillColor(TEAL_DARK)
        c.rect(0, self.h - 34, self.w, 34, fill=1, stroke=0)
        c.setFillColor(GOLD)
        c.setFont('Helvetica-Bold', 8)
        c.drawString(10, self.h - 16, self.label + '  >')
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 11)
        label_w = c.stringWidth(self.label + '  >', 'Helvetica-Bold', 8)
        c.drawString(10 + label_w + 8, self.h - 14, self.title)
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 9)
        y = self.h - 34 - 16
        for line in self.body_lines:
            c.drawString(12, y, line)
            y -= self.line_h
        c.setFillColor(TEAL)
        c.rect(0, 0, 4, self.h - 34, fill=1, stroke=0)


class CalloutBox(Flowable):
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
        c.drawString(14, self.h - 20, '*')
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Oblique', 9.5)
        y = self.h - 18
        for line in self.text_lines:
            c.drawString(30, y, line)
            y -= 14


def draw_cover(canvas, doc):
    canvas.saveState()
    w, h = PAGE_W, PAGE_H
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, h * 0.6, w, h * 0.4, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    p = canvas.beginPath()
    p.moveTo(w * 0.5, h)
    p.lineTo(w, h)
    p.lineTo(w, h * 0.72)
    p.close()
    canvas.drawPath(p, fill=1, stroke=0)
    canvas.setFillColor(TEAL_DARK)
    canvas.rect(0, 0, w, 1.8 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, 1.8 * inch, w, 5, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 28)
    canvas.drawString(MARGIN, h * 0.88, 'QBITEL BRIDGE')
    canvas.setFillColor(GOLD)
    canvas.setFont('Helvetica-Bold', 18)
    canvas.drawString(MARGIN, h * 0.83, 'Banking & Financial Services Security')
    canvas.setFillColor(GOLD)
    canvas.rect(MARGIN, h * 0.823, 4 * inch, 3, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 30)
    canvas.drawString(MARGIN, h * 0.72, 'Quantum-Safe Protection for')
    canvas.setFont('Helvetica-Bold', 30)
    canvas.drawString(MARGIN, h * 0.72 - 38, 'Global Payment Infrastructure')
    canvas.setFillColor(HexColor('#CCDDE8'))
    canvas.setFont('Helvetica-Oblique', 13)
    canvas.drawString(MARGIN, h * 0.72 - 76,
                      'Securing ISO 8583  |  SWIFT/FedWire  |  COBOL Mainframes  |  FIX Trading')
    stats = [
        ('10,000+', 'TPS', 'Encryption'),
        ('<50ms', 'End-to-End', 'Latency'),
        ('78%', 'Autonomous', 'Response'),
        ('PCI-DSS', '4.0 Ready', 'Certified'),
    ]
    box_w = (w - 2 * MARGIN) / 4
    box_h = 0.9 * inch
    box_y = 1.85 * inch
    colors_box = [NAVY, LIGHT_NAVY, NAVY, LIGHT_NAVY]
    for i, (val, l1, l2) in enumerate(stats):
        bx = MARGIN + i * box_w
        canvas.setFillColor(colors_box[i])
        canvas.rect(bx, box_y, box_w - 4, box_h, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.setFont('Helvetica-Bold', 16)
        vw = canvas.stringWidth(val, 'Helvetica-Bold', 16)
        canvas.drawString(bx + (box_w - 4 - vw) / 2, box_y + box_h - 26, val)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 8)
        l1w = canvas.stringWidth(l1, 'Helvetica', 8)
        canvas.drawString(bx + (box_w - 4 - l1w) / 2, box_y + box_h - 42, l1)
        l2w = canvas.stringWidth(l2, 'Helvetica', 8)
        canvas.drawString(bx + (box_w - 4 - l2w) / 2, box_y + box_h - 54, l2)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 9)
    canvas.drawString(MARGIN, 0.7 * inch, 'enterprise@qbitel.com  |  https://bridge.qbitel.com')
    canvas.setFillColor(HexColor('#AABBCC'))
    canvas.setFont('Helvetica', 8)
    canvas.drawString(MARGIN, 0.5 * inch, 'Confidential - For Authorized Recipients Only  |  (c) 2026 QBITEL.')
    canvas.restoreState()


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 0.45 * inch, PAGE_W, 0.45 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, PAGE_H - 0.45 * inch - 3, PAGE_W, 3, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(MARGIN, PAGE_H - 0.31 * inch, 'QBITEL BRIDGE')
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(TEAL)
    canvas.drawString(MARGIN + 1.1 * inch, PAGE_H - 0.31 * inch, 'BANKING & FINANCIAL SERVICES')
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 8)
    page_str = 'Page %d' % doc.page
    pw = canvas.stringWidth(page_str, 'Helvetica', 8)
    canvas.drawString(PAGE_W - MARGIN - pw, PAGE_H - 0.31 * inch, page_str)
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, 0.4 * inch, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0.4 * inch, PAGE_W, 2, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 7.5)
    canvas.drawString(MARGIN, 0.15 * inch,
                      'Confidential - For Authorized Recipients Only  |  (c) 2026 QBITEL. All Rights Reserved.')
    contact_str = 'enterprise@qbitel.com  |  bridge.qbitel.com'
    cw = canvas.stringWidth(contact_str, 'Helvetica', 7.5)
    canvas.drawString(PAGE_W - MARGIN - cw, 0.15 * inch, contact_str)
    canvas.restoreState()


def make_styles():
    styles = {}
    styles['body'] = ParagraphStyle('body', fontName='Helvetica', fontSize=9.5,
                                     leading=15, textColor=DARK_TEXT, spaceAfter=6,
                                     alignment=TA_JUSTIFY)
    styles['bullet'] = ParagraphStyle('bullet', fontName='Helvetica', fontSize=9.5,
                                       leading=14, textColor=DARK_TEXT, leftIndent=14,
                                       firstLineIndent=0, spaceAfter=4)
    styles['h2'] = ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=12,
                                   leading=16, textColor=NAVY, spaceAfter=6, spaceBefore=10)
    styles['h3'] = ParagraphStyle('h3', fontName='Helvetica-Bold', fontSize=10.5,
                                   leading=14, textColor=TEAL_DARK, spaceAfter=4, spaceBefore=8)
    styles['table_h'] = ParagraphStyle('table_h', fontName='Helvetica-Bold', fontSize=8.5,
                                        leading=12, textColor=WHITE_C, alignment=TA_CENTER)
    styles['table_c'] = ParagraphStyle('table_c', fontName='Helvetica', fontSize=8.5,
                                        leading=12, textColor=DARK_TEXT)
    return styles


def tbl_style(header_rows=1):
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, header_rows - 1), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, header_rows - 1), WHITE_C),
        ('FONTNAME', (0, 0), (-1, header_rows - 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, header_rows - 1), 8.5),
        ('ALIGN', (0, 0), (-1, header_rows - 1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, header_rows), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('FONTNAME', (0, header_rows), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, header_rows), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.4, MID_GREY),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
    ])


def build_story(styles):
    S = []

    def sp(n=8):
        S.append(Spacer(1, n))

    def hdr(title, subtitle=None):
        S.append(SectionHeader(title, subtitle))
        sp(10)

    def para(text, style='body'):
        S.append(Paragraph(text, styles[style]))

    def bul(text):
        S.append(Paragraph('  * ' + text, styles['bullet']))

    # Executive Summary
    hdr('Executive Summary', 'Quantum-Safe Protection for Global Payment Infrastructure')
    para('The global banking system processes over $5 trillion in daily transactions across payment rails '
         'built on protocols designed decades before quantum computing existed. ISO 8583 card networks, '
         'SWIFT MT/MX wire transfers, COBOL mainframes, and FIX trading systems form the backbone of global '
         'finance - and every one carries quantum-vulnerable cryptography that nation-state adversaries are '
         'harvesting today for decryption tomorrow.')
    sp(6)
    para('<b>QBITEL Bridge</b> is the only protocol-aware, quantum-safe security platform purpose-built for '
         'banking infrastructure. It delivers post-quantum encryption at <b>10,000+ transactions per second</b> '
         'with <b>sub-50ms latency</b>, integrates natively with HSMs from Thales Luna, AWS CloudHSM, and '
         'Futurex, and provides <b>78% autonomous incident response</b> without touching your core banking systems.')
    sp(10)
    S.append(StatBlock([
        ('$6.2B', 'Projected quantum-related\nbanking losses by 2030'),
        ('80%', 'Payment volume on\nquantum-vulnerable rails'),
        ('<50ms', 'PQC encryption\nlatency at 10K+ TPS'),
        ('78%', 'Autonomous\nincident response'),
    ]))
    sp(10)
    bul('<b>$6.2 billion</b> in quantum-related banking losses projected by 2030 (NIST/McKinsey)')
    bul('<b>80%</b> of global payment volume traverses quantum-vulnerable legacy systems')
    bul('<b>DORA</b> (EU Digital Operational Resilience Act) ICT risk frameworks - mandatory January 2025')
    bul('<b>PCI-DSS 4.0</b> requires cryptographic agility and quantum-readiness assessments')
    bul('<b>Harvest-now-decrypt-later</b> attacks on SWIFT traffic confirmed by intelligence agencies')
    sp(8)
    para('QBITEL Bridge enables your institution to achieve quantum-safe compliance, protect payment rails '
         'in production, and modernize security posture - all without disrupting the 15-20 year system '
         'lifecycles that define banking infrastructure.')

    S.append(PageBreak())

    # Banking Crisis
    hdr('The Banking Crisis No CISO Can Ignore', 'Three existential threats to global payment infrastructure')
    para('<b>Threat 1: Harvest-Now-Decrypt-Later on Payment Rails</b>', 'h2')
    para('Nation-state actors - confirmed by NSA, GCHQ, and ENISA advisories - are systematically '
         'harvesting encrypted SWIFT MT103 wire transfers, ISO 8583 card authorization flows, and '
         'FedWire/ACH batch files today. Their strategy: store the ciphertext, wait for quantum computers '
         'to mature (5-10 years), then decrypt trillions in historical transaction records.')
    sp(6)
    S.append(CalloutBox([
        'IBM, Google, and IonQ roadmaps target fault-tolerant quantum systems within 5-10 years.',
        'SWIFT regulatory archive requirements: 5-7 years. The harvest window is already open.',
        'Average cost of a major payment breach: $50 million - quantum decryption has no precedent.',
    ]))
    sp(8)
    para('<b>Threat 2: Mainframe Inter-System Unencrypted Communications</b>', 'h2')
    para('IBM z/OS mainframes running COBOL/CICS/DB2 process the majority of global banking transactions. '
         'Inter-LPAR communications, TN3270e terminal sessions, MQ messaging between CICS regions, and '
         'DB2 DRDA database connections often traverse unencrypted or weakly encrypted channels - even in 2026. '
         'Traditional network security tools cannot inspect these because they do not understand EBCDIC '
         'encoding, CICS transaction flows, or DB2 package authentication.')
    sp(8)
    para('<b>Threat 3: COBOL Legacy Protocol Vulnerabilities</b>', 'h2')
    para('The average age of COBOL applications in production banking systems is 45 years. Architected before '
         'TLS existed, before IPv6, and before cryptographic agility. QBITEL research identified three vulnerability '
         'categories in production banking environments:')
    sp(4)
    bul('<b>Replay attacks</b> on fixed-format ISO 8583 messages lacking nonce or timestamp fields')
    bul('<b>MITM exposure</b> on TN3270e sessions between branch teller systems and mainframe CICS')
    bul('<b>Weak key derivation</b> in legacy DES/3DES implementations in payment processing COBOL modules')

    S.append(PageBreak())

    # Platform Overview
    hdr('QBITEL Bridge for Banking', 'Protocol-aware | Hardware-backed | Quantum-safe')
    para('QBITEL Bridge is deployed as a transparent proxy layer between existing banking infrastructure '
         'components. No modifications to core banking applications, no changes to SWIFT connectivity, '
         'no alterations to payment card network integrations.')
    sp(10)
    arch_data = [
        [Paragraph('Component', styles['table_h']),
         Paragraph('Capability', styles['table_h']),
         Paragraph('Banking Protocol Coverage', styles['table_h'])],
        ['Protocol Discovery Engine', '2-4 hour full inventory', 'ISO 8583, ISO 20022, SWIFT MT/MX, FIX, TN3270e, ACH, FedWire, SEPA, CHIPS'],
        ['PQC Cryptographic Engine', 'NIST FIPS 203/204/205', 'ML-KEM (Kyber), ML-DSA (Dilithium), SPHINCS+'],
        ['HSM Integration Layer', 'PKCS#11 + native APIs', 'Thales Luna 7, AWS CloudHSM, Azure MHSM, Futurex, Utimaco'],
        ['Autonomous Response Fabric', '78% zero-touch response', 'Payment fraud, SWIFT anomaly, mainframe intrusion playbooks'],
        ['Compliance Automation', 'Real-time evidence', 'PCI-DSS 4.0, DORA, Basel III/IV, SOX, GDPR, BCBS 239, SWIFT CSP'],
    ]
    arch_tbl = Table(arch_data, colWidths=[CONTENT_W * 0.22, CONTENT_W * 0.22, CONTENT_W * 0.56])
    arch_tbl.setStyle(tbl_style())
    S.append(arch_tbl)
    sp(10)
    S.append(CalloutBox([
        'Deployment: On-premises appliance | Private cloud (VMware/OpenStack) | Public cloud (AWS/Azure/GCP) BYOK | Hybrid multi-site',
    ]))

    S.append(PageBreak())

    # Capability 1
    hdr('Capability 1: ISO 8583 / ISO 20022 Payment Rail Protection',
        'Field-level PQC encryption at 10,000+ TPS for card payment networks')
    para('ISO 8583 is the dominant message standard for card payment authorization, clearing, and settlement - '
         'used by Visa, Mastercard, Amex, and domestic card schemes globally. ISO 20022 is the next-generation '
         'XML-based standard adopted by SWIFT, TARGET2, CHIPS, and FedNow.')
    sp(6)
    para('<b>The problem:</b> Both standards define message structure but not cryptographic protection. '
         'ISO 8583 messages typically use PIN block encryption for PIN fields only - the remaining 128 data '
         'elements including PANs, amounts, merchant codes, and authorization codes traverse in cleartext '
         'or with legacy RSA-1024/2048 transport encryption.')
    sp(8)
    cap1_data = [
        [Paragraph('Protection Layer', styles['table_h']),
         Paragraph('QBITEL Bridge Capability', styles['table_h']),
         Paragraph('Standard Preserved', styles['table_h'])],
        ['ISO 8583 Field Encryption', 'Field-level ML-KEM for PAN, expiry, CVV2, amount', 'Full bit-map integrity'],
        ['Protocol-Aware Inspection', 'Parse bit maps at 10,000+ TPS without buffering', 'Zero message modification'],
        ['ISO 20022 XML Signing', 'ML-DSA on pacs.008, camt.053, pain.001 message types', 'XSD schema compliance'],
        ['PIN Block Modernization', 'TDES to AES-256 with PQC key wrapping (ISO 9564)', 'PIN block format preserved'],
        ['Cryptographic Agility', 'Algorithm swap without application code change', 'Scheme format integrity maintained'],
    ]
    cap1_tbl = Table(cap1_data, colWidths=[CONTENT_W * 0.28, CONTENT_W * 0.45, CONTENT_W * 0.27])
    cap1_tbl.setStyle(tbl_style())
    S.append(cap1_tbl)
    sp(10)
    S.append(StatBlock([
        ('10K+', 'ISO 8583 TPS\nwith full PQC active'),
        ('<50ms', 'End-to-end\nencryption latency'),
        ('128', 'ISO 8583 fields\nprotected selectively'),
        ('Zero', 'Application code\nchanges required'),
    ], height=65))

    S.append(PageBreak())

    # Capability 2
    hdr('Capability 2: SWIFT / Wire Transfer Security',
        'Post-quantum message signing for MT103, MT202, SWIFT MX, FedWire, CHIPS')
    para('SWIFT is the messaging backbone for international wire transfers, correspondent banking, and '
         'securities settlement. SWIFT MT103, MT202, and ISO 20022 MX messages carry instructions for '
         'trillions in daily interbank flows. SWIFT MX migration is underway and mandatory for most corridors.')
    sp(6)
    S.append(CalloutBox([
        'Bangladesh Bank 2016: $81M stolen via fraudulent MT103 insertions - no message-level signing.',
        'SWIFT regulatory archives carry RSA/AES encryption: quantum computers will break this.',
        'SWIFT CSP 2025 mandates controls but does not yet mandate post-quantum cryptography.',
    ]))
    sp(8)
    swift_data = [
        [Paragraph('SWIFT CSP Control', styles['table_h']),
         Paragraph('QBITEL Bridge Capability', styles['table_h']),
         Paragraph('Algorithm', styles['table_h'])],
        ['MT/MX Message Authentication', 'ML-DSA signature on every outbound SWIFT message', 'FIPS 204 ML-DSA'],
        ['BIC-to-BIC Key Exchange', 'ML-KEM replacing RSA-2048 key transport', 'FIPS 203 ML-KEM'],
        ['FedWire/Fedline Protection', 'PQC-wrapped Fedline Advantage connections', 'AES-256-GCM + ML-KEM'],
        ['CHIPS Authentication', 'PQC key wrapping for CHIPS participants', 'ML-KEM + ML-DSA'],
        ['SWIFT Anomaly Detection', 'ML model trained on 50M+ SWIFT message patterns', 'Real-time inference'],
        ['CSP 1.1 Environment', 'Protocol-aware network segmentation', 'Zero-trust micro-segmentation'],
    ]
    swift_tbl = Table(swift_data, colWidths=[CONTENT_W * 0.33, CONTENT_W * 0.45, CONTENT_W * 0.22])
    swift_tbl.setStyle(tbl_style())
    S.append(swift_tbl)

    S.append(PageBreak())

    # Capability 3
    hdr('Capability 3: COBOL / Mainframe Legacy Shield',
        'Zero code changes. TN3270e PQC proxy, CICS signing, DB2 DRDA upgrade.')
    para('IBM z/OS mainframes running COBOL/CICS/DB2 process an estimated <b>95% of ATM transactions</b> and '
         '<b>80% of in-person card swipes</b> globally. With 15-20 year deployment lifecycles, these systems '
         'cannot be replaced without multi-year transformation programs. QBITEL Bridge brings quantum-safe '
         'security to the mainframe without touching a single line of COBOL.')
    sp(8)
    main_data = [
        [Paragraph('Component', styles['table_h']),
         Paragraph('Zero-Downtime', styles['table_h']),
         Paragraph('Performance Impact', styles['table_h']),
         Paragraph('Code Change', styles['table_h'])],
        ['TN3270e Sessions', 'YES', '<5ms per session', 'None'],
        ['CICS Transactions', 'YES', '<2ms per transaction', 'None'],
        ['DB2 DRDA Connections', 'YES', '<8ms per query', 'None'],
        ['IBM MQ Messages', 'YES', '<3ms per message', 'None'],
        ['VSAM File Encryption', 'Scheduled window', '<1% MIPS overhead', 'None'],
        ['IMS DL/I Calls', 'YES', '<4ms per call', 'None'],
        ['AS/400 (IBM i)', 'YES', '<6ms per session', 'None'],
    ]
    main_tbl = Table(main_data, colWidths=[CONTENT_W * 0.31, CONTENT_W * 0.18, CONTENT_W * 0.26, CONTENT_W * 0.25])
    main_tbl.setStyle(tbl_style())
    S.append(main_tbl)

    S.append(PageBreak())

    # Capability 4
    hdr('Capability 4: Trading Protocol Security (FIX / FpML)',
        'ML-DSA message signing for FIX sessions at <50 microseconds latency')
    para('Electronic trading relies on FIX (Financial Information eXchange) protocol for order routing, '
         'execution confirmation, and market data between buy-side, sell-side, ECNs, and exchanges. '
         'FIX 4.2 through FIX 5.0/FIXT 1.1 carries equity, fixed income, FX, and derivatives orders globally. '
         'FpML covers OTC derivatives confirmation under EMIR/Dodd-Frank.')
    sp(6)
    fix_data = [
        [Paragraph('FIX/FpML Protection', styles['table_h']),
         Paragraph('Technical Detail', styles['table_h'])],
        ['FIX Session Authentication', 'Replace Tag 96 password with ML-DSA signing on every message'],
        ['Order Message Signing', 'NewOrderSingle (35=D), OrderCancel (35=F), ExecutionReport (35=8)'],
        ['Replay Prevention', 'Bind ML-DSA signatures to MsgSeqNum - cryptographic replay block'],
        ['FpML Encryption', 'ML-KEM + AES-256-GCM for OTC confirmation payloads (EMIR compliant)'],
        ['Market Data Integrity', 'Sign MarketDataSnapshotFullRefresh (35=W) from exchange feeds'],
        ['Latency Target', '<50 microseconds added latency - DPDK-accelerated processing'],
    ]
    fix_tbl = Table(fix_data, colWidths=[CONTENT_W * 0.38, CONTENT_W * 0.62])
    fix_tbl.setStyle(tbl_style())
    S.append(fix_tbl)
    sp(10)
    S.append(CalloutBox([
        'Red team finding: FIX sequence number replay allows order book manipulation.',
        'QBITEL Bridge: ML-DSA signing deployed in 48 hours, <50us latency impact, zero code changes.',
    ]))

    S.append(PageBreak())

    # Capabilities 5-7
    hdr('Capabilities 5-7: Cloud Security | Compliance | Fraud Analytics',
        'Enterprise-wide PQC coverage across cloud, compliance, and fraud prevention')
    para('<b>Capability 5: Cloud Migration Security</b>', 'h2')
    para('QBITEL Bridge enables BYOK post-quantum keys into AWS CloudHSM, Azure Managed HSM, and GCP Cloud HSM. '
         'ML-KEM outer-layer wraps all cloud-native encryption operations. Multi-cloud key synchronization '
         'with GDPR/DORA data residency enforcement.')
    sp(6)
    cloud_data = [
        [Paragraph('Provider', styles['table_h']),
         Paragraph('HSM Product', styles['table_h']),
         Paragraph('Integration', styles['table_h']),
         Paragraph('PQC Support', styles['table_h'])],
        ['AWS', 'CloudHSM (Luna HSM 7)', 'PKCS#11 + JCA/JCE', 'ML-KEM, ML-DSA via QBITEL proxy'],
        ['Azure', 'Managed HSM (Marvell)', 'Azure MHSM REST API', 'ML-KEM, ML-DSA via QBITEL proxy'],
        ['GCP', 'Cloud HSM (Marvell)', 'PKCS#11 + KMS API', 'ML-KEM, ML-DSA via QBITEL proxy'],
        ['On-prem', 'Thales Luna Network HSM 7', 'PKCS#11 + REST', 'Native + QBITEL extensions'],
        ['On-prem', 'Futurex Vectera Plus', 'PKCS#11 + FXAPI', 'Native + QBITEL extensions'],
    ]
    cloud_tbl = Table(cloud_data, colWidths=[CONTENT_W * 0.1, CONTENT_W * 0.3, CONTENT_W * 0.28, CONTENT_W * 0.32])
    cloud_tbl.setStyle(tbl_style())
    S.append(cloud_tbl)
    sp(10)
    para('<b>Capability 6: Autonomous Compliance - PCI-DSS 4.0 / DORA / Basel III</b>', 'h2')
    para('Automates evidence collection for PCI-DSS 4.0 (Requirements 3, 4, 6, 8, 10, 12), '
         'DORA (Articles 9, 10, 11, 17, 26, 28), Basel III/IV operational risk data, SOX ITGC controls, '
         'GDPR data-by-default encryption, BCBS 239 data lineage, SWIFT CSP 2025, and NY DFS Part 500.')
    sp(8)
    para('<b>Capability 7: Real-Time Fraud Analytics</b>', 'h2')
    para('Protocol-native feature extraction from ISO 8583 messages: 200+ features including BIN geography, '
         'merchant category cross-reference, amount velocity, and field consistency scoring - all at '
         'sub-50ms inference integrated directly into the authorization path. SWIFT MT103 anomaly detection '
         'covers beneficiary manipulation, amount rounding (BEC hallmark), and off-hours timing patterns. '
         'Federated learning trains models without sharing raw transaction data (GDPR compliant).')

    S.append(PageBreak())

    # Compliance Table
    hdr('Compliance Coverage', 'PCI-DSS 4.0 | DORA | Basel III/IV | SOX | GDPR | BCBS 239 | SWIFT CSP')
    comp_data = [
        [Paragraph('Regulation', styles['table_h']),
         Paragraph('QBITEL Bridge Coverage', styles['table_h']),
         Paragraph('Evidence Generated', styles['table_h'])],
        ['PCI-DSS 4.0', 'Requirements 3, 4, 6, 8, 10, 12', 'SAQ-D, ROC evidence packages'],
        ['DORA (EU) 2022/2554', 'Articles 9, 10, 11, 17, 26, 28', 'ICT risk register, incident reports'],
        ['Basel III/IV', 'Operational risk data collection', 'AMA data feeds, KRI dashboards'],
        ['SOX Section 404', 'IT general controls (ITGC)', 'ITGC control evidence'],
        ['GDPR Articles 25/32', 'Data-by-default encryption', 'ROPA entries, DPA evidence'],
        ['BCBS 239', 'Risk data aggregation reporting', 'Data lineage maps'],
        ['SWIFT CSP 2025', 'Mandatory + advisory controls', 'CSP attestation package'],
        ['NIST FIPS 140-3', 'Level 3 HSM validation', 'CMVP certificates'],
        ['ISO 27001:2022', 'Annex A cryptographic controls', 'ISMS evidence'],
        ['NY DFS Part 500', 'Encryption and CISO reporting', 'Part 500 attestation'],
    ]
    comp_tbl = Table(comp_data, colWidths=[CONTENT_W * 0.25, CONTENT_W * 0.4, CONTENT_W * 0.35])
    comp_tbl.setStyle(tbl_style())
    S.append(comp_tbl)

    S.append(PageBreak())

    # Integration Ecosystem
    hdr('Integration Ecosystem', 'Core banking | Payment infrastructure | Security stack')
    para('<b>Core Banking Systems</b>', 'h2')
    cb_data = [
        [Paragraph('Vendor', styles['table_h']),
         Paragraph('Product', styles['table_h']),
         Paragraph('Integration Method', styles['table_h']),
         Paragraph('Deploy Time', styles['table_h'])],
        ['Temenos', 'T24 / Transact', 'REST API + message broker', '2-4 weeks'],
        ['Finastra', 'Fusion / Kondor', 'SWIFT gateway integration', '3-5 weeks'],
        ['FIS', 'Modern Banking Platform', 'ISO 20022 proxy', '2-3 weeks'],
        ['Fiserv', 'DNA / Premier', 'JDBC/ODBC intercept', '1-2 weeks'],
        ['Oracle', 'FLEXCUBE', 'MQ + REST integration', '3-4 weeks'],
        ['SAP', 'SAP Banking Services', 'RFC/BAPI interception', '2-4 weeks'],
        ['Infosys', 'Finacle', 'REST + SWIFT gateway', '2-3 weeks'],
    ]
    cb_tbl = Table(cb_data, colWidths=[CONTENT_W * 0.18, CONTENT_W * 0.28, CONTENT_W * 0.34, CONTENT_W * 0.2])
    cb_tbl.setStyle(tbl_style())
    S.append(cb_tbl)
    sp(10)
    para('<b>Payment Infrastructure</b>', 'h2')
    pay_data = [
        [Paragraph('System', styles['table_h']),
         Paragraph('Protocol', styles['table_h']),
         Paragraph('Notes', styles['table_h'])],
        ['Visa DPS / VisaNet', 'ISO 8583', 'Transparent proxy - zero application change'],
        ['Mastercard', 'ISO 8583 / ISO 20022', 'Field-level encryption, scheme format preserved'],
        ['SWIFT Alliance', 'MT/MX (ISO 20022)', 'SWIFT API Gateway, CSP-aligned'],
        ['FedNow', 'ISO 20022', 'Real-time PQC on Fed connectivity layer'],
        ['FedWire', 'Fedline protocol', 'FIPS 140-3 compliant TLS + PQC wrapper'],
        ['CHIPS', 'Proprietary', 'PQC key wrapping for CHIPS participants'],
        ['SEPA (EPC)', 'ISO 20022 SCT/SCT Inst', 'GDPR-compliant cross-border PQC'],
        ['ACH / NACHA', 'NACHA format', 'File-level encryption, batch + real-time'],
    ]
    pay_tbl = Table(pay_data, colWidths=[CONTENT_W * 0.25, CONTENT_W * 0.25, CONTENT_W * 0.5])
    pay_tbl.setStyle(tbl_style())
    S.append(pay_tbl)

    S.append(PageBreak())

    # Deployment Timeline
    hdr('Deployment Timeline', '4-phase deployment: 16 weeks from passive tap to full enterprise PQC')
    phases = [
        ('Phase 1', 'Protocol Discovery & Risk Assessment', 'Weeks 1-2', [
            'Deploy passive protocol tap on SPAN/mirror ports - zero production impact',
            'Automated discovery: ISO 8583, SWIFT, FIX, TN3270e, and proprietary protocols',
            'Cryptographic inventory: identify all RSA, ECC, DES/3DES deployments',
            'Quantum risk scoring: rank protocols by breach impact and vulnerability timeline',
            'Deliverable: Protocol Risk Register with prioritized remediation roadmap',
        ]),
        ('Phase 2', 'Infrastructure Readiness & HSM Provisioning', 'Weeks 3-5', [
            'HSM cluster provisioning: Thales Luna / AWS CloudHSM / Azure MHSM',
            'Network segmentation: PCI cardholder data environment isolation',
            'QBITEL Bridge appliance deployment (physical or virtual)',
            'Integration testing with core banking system (read-only / passive mode)',
            'Deliverable: QBITEL Bridge operational in passive monitoring mode',
        ]),
        ('Phase 3', 'Payment Rail Protection', 'Weeks 6-10', [
            'ISO 8583 field-level encryption activation (non-business-hours cutover)',
            'SWIFT MT/MX PQC wrapping activation - zero SWIFT downtime',
            'FedWire/ACH session protection and performance validation',
            '10,000 TPS load test + <50ms latency verification',
            'Deliverable: Payment rails protected with PQC; fraud analytics baseline operational',
        ]),
        ('Phase 4', 'Mainframe, Legacy & Full Compliance', 'Weeks 11-16', [
            'TN3270e PQC proxy activation for branch teller network',
            'COBOL/CICS transaction signing and DB2 DRDA PQC upgrade',
            'FIX/FpML trading protocol security activation',
            'DORA ICT risk register + PCI-DSS 4.0 evidence package completion',
            'Go-live sign-off, SOC integration, runbook handover',
        ]),
    ]
    phase_colors = [TEAL_DARK, NAVY, TEAL_DARK, NAVY]
    for (ph, title, timing, items), color in zip(phases, phase_colors):
        phase_data = [
            [Paragraph('<b>%s: %s</b>' % (ph, title),
                       ParagraphStyle('ph', fontName='Helvetica-Bold', fontSize=10, textColor=WHITE_C)),
             Paragraph(timing, ParagraphStyle('tm', fontName='Helvetica-Bold', fontSize=10,
                       textColor=GOLD, alignment=TA_RIGHT))],
        ]
        phase_tbl = Table(phase_data, colWidths=[CONTENT_W * 0.75, CONTENT_W * 0.25])
        phase_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), color),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        S.append(phase_tbl)
        items_data = [[Paragraph('  * ' + item, styles['body'])] for item in items]
        items_tbl = Table(items_data, colWidths=[CONTENT_W])
        items_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        S.append(items_tbl)
        sp(8)

    S.append(PageBreak())

    # Performance Specs
    hdr('Performance Specifications', 'Validated under sustained load with ML-KEM and ML-DSA active')
    perf_data = [
        [Paragraph('Metric', styles['table_h']),
         Paragraph('Specification', styles['table_h']),
         Paragraph('Test Condition', styles['table_h'])],
        ['ISO 8583 throughput', '10,000+ TPS sustained', '60-minute load test, ML-KEM active'],
        ['SWIFT MT103 latency', '<8ms added latency', 'ML-DSA signing, full HSM custody'],
        ['FIX message latency', '<50 microseconds added', 'FIX 5.0 NewOrderSingle with signing'],
        ['TN3270e session setup', '<5ms per session', '2,000 concurrent sessions'],
        ['Protocol discovery', '2-4 hours complete', '100+ protocol types on 10Gbps link'],
        ['Autonomous response rate', '78% zero-touch', 'Banking-tuned incident playbooks'],
        ['System availability', '99.999% (5 nines)', 'Active-active HA cluster'],
        ['HSM operations', '20,000 ops/sec', 'Thales Luna Network HSM 7'],
        ['Key rotation', 'Zero-downtime', 'Automated with HSM custody'],
        ['Failover time', '<30 seconds', 'Active-passive with synchronous replication'],
        ['Audit log throughput', '100,000 events/sec', 'Signed, tamper-evident logs'],
        ['Compliance reporting', 'Real-time', 'Continuous evidence collection'],
    ]
    perf_tbl = Table(perf_data, colWidths=[CONTENT_W * 0.32, CONTENT_W * 0.33, CONTENT_W * 0.35])
    perf_tbl.setStyle(tbl_style())
    S.append(perf_tbl)

    S.append(PageBreak())

    # Competitive Differentiation
    hdr('Competitive Differentiation', 'Why QBITEL Bridge is the only banking-native PQC platform')
    comp_diff = [
        ('vs. Traditional HSM Vendors (Thales, Utimaco, Futurex)',
         'HSMs provide cryptographic operations but have zero protocol intelligence. An HSM cannot '
         'parse ISO 8583 bit maps, inspect SWIFT message fields, or identify COBOL data structures. '
         'QBITEL Bridge uses HSMs as the cryptographic backend and adds protocol intelligence, '
         'autonomous response, and compliance automation on top.',
         'QBITEL: Protocol-aware + HSM-backed = the only combination addressing both protocol and cryptographic security.'),
        ('vs. Cloud-Native Encryption (AWS KMS, Azure Key Vault, GCP KMS)',
         'Cloud KMS services use RSA-2048 and ECC P-256 for key transport - both quantum-vulnerable. '
         'Post-quantum support in cloud KMS is in beta with limited algorithm coverage. Cloud KMS '
         'cannot protect on-premises mainframes, SWIFT connectivity, or ISO 8583 card networks.',
         'QBITEL: Cloud-agnostic, mainframe-capable, fully PQC-compliant today - not on a roadmap.'),
        ('vs. Network Security Platforms (Palo Alto, Fortinet, Cisco)',
         'Next-generation firewalls operate on IP/TCP layers and cannot decrypt banking-specific '
         'protocols without breaking transaction flows. They provide TLS inspection for web traffic '
         'but have no concept of ISO 8583 field structures or SWIFT message types.',
         'QBITEL: Banking-protocol-native inspection at layers 5-7 - not just network-layer perimeter.'),
        ('vs. Payment Security Specialists (Voltage, P2PE vendors)',
         'Point-to-point encryption vendors focus on PAN protection at the point of sale. They do not '
         'address SWIFT security, mainframe communications, trading protocol integrity, or post-quantum '
         'key exchange for wire transfer systems.',
         'QBITEL: Complete ecosystem - ISO 8583, SWIFT, FedWire, COBOL mainframes, FIX trading, cloud.'),
    ]
    for title, body, advantage in comp_diff:
        S.append(KeepTogether([
            Paragraph('<b>%s</b>' % title, styles['h2']),
            Paragraph(body, styles['body']),
            CalloutBox([advantage]),
            Spacer(1, 8),
        ]))

    S.append(PageBreak())

    # Customer Scenarios
    hdr('Customer Scenarios', 'Real deployments across Tier-1 global bank, regional bank, and investment bank')
    scenarios = [
        ('Scenario 1', 'Tier-1 Global Bank - DORA Compliance Under Time Pressure', [
            'Challenge: EU universal bank with 90-day deadline to demonstrate DORA remediation to regulator.',
            'Discovery: 340 protocol flows found in 72 hours - 23 SWIFT paths, 4 card networks, 12 TN3270e sessions.',
            'Deployment: ML-KEM activated on SWIFT connections in Week 3 - zero SWIFT connectivity interruption.',
            'Compliance: DORA ICT risk register auto-populated; incident engine connected to Splunk in Week 4.',
            'Outcome: DORA evidence package delivered 45 days ahead of regulatory deadline. ECB-accepted.',
            'Metrics: 340 protocols in 72h | DORA package in 45 days | Zero SWIFT downtime',
        ]),
        ('Scenario 2', 'Regional US Bank - PCI-DSS 4.0 Quantum Readiness', [
            'Challenge: $15B asset community bank - QSA finding: RSA-1024 in ISO 8583 acquirer connection.',
            'Discovery: Passive tap on card network connections revealed legacy RSA-1024 key exchange.',
            'Deployment: ML-KEM replacement activated in 48 hours with zero card network downtime.',
            'Compliance: PCI-DSS 4.0 evidence auto-generated for Requirements 3, 4, 8, and 10.',
            'Outcome: RSA-1024 critical finding closed. QSA ROC evidence package auto-generated. Zero findings.',
            'Metrics: Critical finding remediated in 48h | PCI-DSS ROC in 30 days | QSA findings closed',
        ]),
        ('Scenario 3', 'Investment Bank - FIX Trading Protocol Security', [
            'Challenge: Red team critical finding - FIX 4.2 sequence number replay allows order book manipulation.',
            'Discovery: FIX session monitoring activated in 4 hours via passive tap. Replay attack confirmed.',
            'Deployment: ML-DSA signing on all FIX sessions in 48 hours. Zero trading system code changes.',
            'Security: Sequence number binding to ML-DSA signature prevents replay cryptographically.',
            'Outcome: Critical finding remediated. <50 microseconds latency impact. EMIR FpML compliance.',
            'Metrics: 48h remediation | <50us latency | Zero code changes | EMIR FpML compliance',
        ]),
    ]
    for label, title, lines in scenarios:
        S.append(KeepTogether([
            ScenarioBox(label, title, lines),
            Spacer(1, 10),
        ]))

    S.append(PageBreak())

    # Next Steps
    hdr('Next Steps', 'Protocol discovery to full compliance in 16 weeks')
    steps = [
        ('Protocol Discovery Session',
         'Deploy passive tap. Deliver complete banking protocol inventory and quantum risk assessment '
         'within 48 hours. Zero production impact, no agents, no code changes.'),
        ('Compliance Gap Assessment',
         'Map current cryptographic posture against PCI-DSS 4.0, DORA, and SWIFT CSP 2025. '
         'Identify highest-priority gaps with remediation roadmap.'),
        ('Technical Deep-Dive',
         '90-minute session for CISO, Head of Payments, and Head of Architecture. '
         'Protocol-specific deployment design for your ISO 8583, SWIFT, and mainframe topology.'),
        ('30-Day Proof of Value',
         'Production POV: protocol discovery, PQC activation on one payment rail, '
         'PCI-DSS 4.0 compliance gap report, DORA ICT risk register seed, performance benchmarking. '
         'No long-term commitment. Most clients achieve ROI within Q1.'),
    ]
    for i, (title, body) in enumerate(steps, 1):
        step_data = [
            [Paragraph(str(i), ParagraphStyle('sn', fontName='Helvetica-Bold', fontSize=18,
                       textColor=GOLD, alignment=TA_CENTER)),
             Paragraph('<b>%s</b><br/>%s' % (title, body), styles['body'])],
        ]
        step_tbl = Table(step_data, colWidths=[0.5 * inch, CONTENT_W - 0.5 * inch])
        step_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), NAVY),
            ('BACKGROUND', (1, 0), (1, 0), LIGHT_BG),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))
        S.append(step_tbl)
        sp(6)

    S.append(PageBreak())

    # Contact
    hdr('Contact QBITEL Banking Practice', 'enterprise@qbitel.com | https://bridge.qbitel.com')
    S.append(Spacer(1, 20))
    S.append(StatBlock([
        ('NIST', 'FIPS 140-3\nLevel 3'),
        ('PCI-DSS', 'QSA Partner\nProgram'),
        ('SWIFT', 'Service Bureau\nPartner'),
        ('SOC 2', 'Type II\nCertified'),
    ], height=70))
    sp(20)
    contact_data = [
        [Paragraph('Enterprise Inquiries', styles['table_h']),
         Paragraph('Platform Portal', styles['table_h']),
         Paragraph('Banking Practice', styles['table_h'])],
        [Paragraph('<b>enterprise@qbitel.com</b>',
                   ParagraphStyle('c1', fontName='Helvetica-Bold', fontSize=11, textColor=TEAL, alignment=TA_CENTER)),
         Paragraph('<b>https://bridge.qbitel.com</b>',
                   ParagraphStyle('c2', fontName='Helvetica-Bold', fontSize=11, textColor=TEAL, alignment=TA_CENTER)),
         Paragraph('CISO Briefings | Technical POV\nCompliance Assessment | Board Decks',
                   ParagraphStyle('c3', fontName='Helvetica', fontSize=9, textColor=DARK_TEXT, alignment=TA_CENTER))],
    ]
    contact_tbl = Table(contact_data, colWidths=[CONTENT_W / 3, CONTENT_W / 3, CONTENT_W / 3])
    contact_tbl.setStyle(tbl_style())
    S.append(contact_tbl)
    sp(20)
    S.append(CalloutBox([
        'QBITEL Bridge - Quantum-Safe. Protocol-Aware. Banking-Ready.',
        'Protecting ISO 8583 | SWIFT MT/MX | COBOL Mainframes | FIX Trading | Real-Time Payments',
        '(c) 2026 QBITEL. All Rights Reserved. Confidential - For Authorized Recipients Only.',
    ]))
    return S


def build_pdf():
    import os
    output = '/Users/prabakarankannan/qbitel/docs/brochures/QBITEL_Bridge_Banking_Marketing_Pitch.pdf'
    doc = BaseDocTemplate(
        output,
        pagesize=letter,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
    )
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H, leftPadding=0, rightPadding=0,
                        topPadding=0, bottomPadding=0, id='cover')
    content_frame = Frame(MARGIN, 0.55 * inch, CONTENT_W, PAGE_H - 1.1 * inch, id='content')
    cover_template = PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover)
    content_template = PageTemplate(id='Content', frames=[content_frame], onPage=draw_page)
    doc.addPageTemplates([cover_template, content_template])
    styles = make_styles()
    story = [NextPageTemplate('Content'), PageBreak()] + build_story(styles)
    doc.build(story)
    print('PDF written: ' + output)

if __name__ == '__main__':
    build_pdf()
