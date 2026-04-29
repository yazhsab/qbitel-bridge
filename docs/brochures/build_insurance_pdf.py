"""
Build QBITEL Bridge Insurance Marketing Pitch - Professional PDF
Uses ReportLab for full layout/design control.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
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


# Custom Flowables

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
            text_w = c.stringWidth(big, 'Helvetica-Bold', 20)
            c.drawString(x + (box_w - text_w) / 2, self.h - 32, big)
            c.setFillColor(WHITE_C)
            c.setFont('Helvetica', 7.5)
            for j, line in enumerate(small.split('\n')):
                lw = c.stringWidth(line, 'Helvetica', 7.5)
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


# Page Templates

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
    canvas.drawString(MARGIN + 1.1 * inch, PAGE_H - 0.32 * inch, 'INSURANCE & REINSURANCE')
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
    canvas.drawString(MARGIN, 0.15 * inch,
                      'Confidential — For Authorized Recipients Only  |  © 2026 QBITEL. All Rights Reserved.')
    contact_str = 'enterprise@qbitel.com  |  bridge.qbitel.com'
    cw = canvas.stringWidth(contact_str, 'Helvetica', 7.5)
    canvas.drawString(PAGE_W - MARGIN - cw, 0.15 * inch, contact_str)
    canvas.restoreState()


def draw_cover(canvas, doc):
    canvas.saveState()
    w, h = PAGE_W, PAGE_H

    # Full navy background
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)

    # Gold diagonal top-right
    canvas.setFillColor(GOLD)
    p = canvas.beginPath()
    p.moveTo(w * 0.55, h)
    p.lineTo(w, h)
    p.lineTo(w, h * 0.65)
    p.close()
    canvas.drawPath(p, fill=1, stroke=0)

    # Teal diagonal mid-right
    canvas.setFillColor(TEAL)
    p2 = canvas.beginPath()
    p2.moveTo(w * 0.72, h)
    p2.lineTo(w, h)
    p2.lineTo(w, h * 0.80)
    p2.close()
    canvas.drawPath(p2, fill=1, stroke=0)

    # Teal bottom strip
    canvas.setFillColor(TEAL_DARK)
    canvas.rect(0, 0, w, 1.6 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, 1.6 * inch, w, 5, fill=1, stroke=0)

    # Main title
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 52)
    canvas.drawString(MARGIN, h * 0.68, 'QBITEL')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica-Bold', 52)
    canvas.drawString(MARGIN, h * 0.68 - 58, 'BRIDGE')
    canvas.setFillColor(GOLD)
    canvas.rect(MARGIN, h * 0.68 - 68, 3.4 * inch, 5, fill=1, stroke=0)

    # Subtitle
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 16)
    canvas.drawString(MARGIN, h * 0.68 - 96, 'INSURANCE & REINSURANCE SECURITY PLATFORM')
    canvas.setFillColor(HexColor('#CCDDE8'))
    canvas.setFont('Helvetica-Oblique', 12)
    canvas.drawString(MARGIN, h * 0.68 - 120, 'Quantum-Safe Protection Across 30-50 Year Policy Lifecycles')

    # Stats boxes — row 1
    stats = [
        ('30-50yr', 'Policy Data\nProtection'),
        ('$80B+', 'Annual Fraud\nPrevented'),
        ('Solvency II', 'Ready\nCompliant'),
        ('78%', 'Autonomous\nThreat Response'),
    ]
    box_w = (CONTENT_W - 3 * 0.12 * inch) / 4
    bx_start = MARGIN
    by = h * 0.38
    bh = 0.9 * inch
    for i, (big, small) in enumerate(stats):
        bx = bx_start + i * (box_w + 0.12 * inch)
        bg_c = TEAL if i % 2 == 0 else LIGHT_NAVY
        canvas.setFillColor(bg_c)
        canvas.roundRect(bx, by, box_w, bh, 5, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.rect(bx, by + bh - 4, box_w, 4, fill=1, stroke=0)
        canvas.setFillColor(GOLD if bg_c == LIGHT_NAVY else WHITE_C)
        canvas.setFont('Helvetica-Bold', 14)
        tw = canvas.stringWidth(big, 'Helvetica-Bold', 14)
        canvas.drawString(bx + (box_w - tw) / 2, by + bh - 26, big)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 7.5)
        for j, line in enumerate(small.split('\n')):
            lw = canvas.stringWidth(line, 'Helvetica', 7.5)
            canvas.drawString(bx + (box_w - lw) / 2, by + bh - 44 - j * 11, line)

    # Stats row 2
    stats2 = [
        ('$5.5B', 'Insurance Cyber Losses 2024'),
        ('67%', 'Insurers on Legacy Mainframes'),
        ('89%+', 'Protocol Discovery Accuracy'),
        ('$4.9M', 'Avg Insurance Breach Cost'),
    ]
    by2 = h * 0.27
    for i, (big, small) in enumerate(stats2):
        bx = bx_start + i * (box_w + 0.12 * inch)
        canvas.setFillColor(LIGHT_BG)
        canvas.roundRect(bx, by2, box_w, 0.78 * inch, 5, fill=1, stroke=0)
        canvas.setFillColor(NAVY)
        canvas.setFont('Helvetica-Bold', 16)
        tw = canvas.stringWidth(big, 'Helvetica-Bold', 16)
        canvas.drawString(bx + (box_w - tw) / 2, by2 + 0.5 * inch, big)
        canvas.setFillColor(MID_GREY)
        canvas.setFont('Helvetica', 7)
        lw = canvas.stringWidth(small, 'Helvetica', 7)
        canvas.drawString(bx + (box_w - lw) / 2, by2 + 0.28 * inch, small)

    # Bottom tagline
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(MARGIN, 0.88 * inch,
                      'AI-Powered  |  Quantum-Safe  |  Zero Disruption  |  Deployed in Hours')
    canvas.setFont('Helvetica', 8.5)
    canvas.setFillColor(HexColor('#AABBCC'))
    canvas.drawString(MARGIN, 0.58 * inch,
                      'Confidential Marketing Document  |  Version 1.0  |  February 2026')
    canvas.restoreState()


# Style Definitions

def get_styles():
    s = {}
    s['body'] = ParagraphStyle(
        'body', fontName='Helvetica', fontSize=10, leading=15,
        textColor=DARK_TEXT, spaceAfter=8, spaceBefore=2, alignment=TA_JUSTIFY)
    s['body_left'] = ParagraphStyle(
        'body_left', fontName='Helvetica', fontSize=10, leading=15,
        textColor=DARK_TEXT, spaceAfter=8, spaceBefore=2, alignment=TA_LEFT)
    s['subsection'] = ParagraphStyle(
        'subsection', fontName='Helvetica-Bold', fontSize=12, leading=16,
        textColor=NAVY, spaceAfter=5, spaceBefore=14)
    s['subsection2'] = ParagraphStyle(
        'subsection2', fontName='Helvetica-Bold', fontSize=10.5, leading=15,
        textColor=TEAL_DARK, spaceAfter=4, spaceBefore=10)
    s['bullet'] = ParagraphStyle(
        'bullet', fontName='Helvetica', fontSize=9.5, leading=14,
        textColor=DARK_TEXT, spaceAfter=3, leftIndent=14)
    s['bullet_bold'] = ParagraphStyle(
        'bullet_bold', fontName='Helvetica-Bold', fontSize=9.5, leading=14,
        textColor=NAVY, spaceAfter=3, leftIndent=14)
    s['table_header'] = ParagraphStyle(
        'table_header', fontName='Helvetica-Bold', fontSize=9,
        textColor=WHITE_C, leading=12)
    s['table_cell'] = ParagraphStyle(
        'table_cell', fontName='Helvetica', fontSize=9,
        textColor=DARK_TEXT, leading=12)
    s['table_cell_bold'] = ParagraphStyle(
        'table_cell_bold', fontName='Helvetica-Bold', fontSize=9,
        textColor=NAVY, leading=12)
    s['caption'] = ParagraphStyle(
        'caption', fontName='Helvetica-Oblique', fontSize=8,
        textColor=MID_GREY, spaceAfter=6, alignment=TA_CENTER)
    s['contact'] = ParagraphStyle(
        'contact', fontName='Helvetica-Bold', fontSize=11,
        textColor=NAVY, spaceAfter=5, alignment=TA_CENTER)
    return s


def make_table(data, col_widths, header_rows=1):
    style = [
        ('BACKGROUND', (0, 0), (-1, header_rows - 1), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, header_rows - 1), WHITE_C),
        ('FONTNAME', (0, 0), (-1, header_rows - 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, header_rows - 1), 9),
        ('FONTNAME', (0, header_rows), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, header_rows), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, header_rows), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCDDEE')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LINEBELOW', (0, header_rows - 1), (-1, header_rows - 1), 2, GOLD),
    ]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle(style))
    return t


def build_story(styles):
    s = styles
    story = []

    # ─── Executive Summary ────────────────────────────────────────────────────
    story.append(SectionHeader(
        'Executive Summary',
        'The Unique Quantum Risk of the Insurance Industry'))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        'The insurance industry faces a security crisis unlike any other sector: policy data created today '
        'must remain confidential for 30 to 50 years — precisely the timeframe in which quantum computers '
        'will break today\'s encryption. A life insurance policy issued in 2026 contains policyholder medical '
        'history, financial data, and beneficiary information that must stay protected until 2076. Today\'s '
        'harvest-now, decrypt-later attacks are already capturing that data.', s['body']))

    story.append(Paragraph(
        'QBITEL Bridge is the only AI-driven, quantum-safe protocol intelligence platform purpose-built '
        'for insurance workflows. It wraps every protocol — ACORD XML, X12 EDI (834/835/837), HL7, SWIFT, '
        'ISO 20022, TN3270e mainframe sessions, and proprietary policy administration traffic — in '
        'post-quantum cryptography without requiring re-architecture of existing systems.', s['body']))

    story.append(StatBlock([
        ('$5.5B', 'Insurance Cyber\nLosses 2024'),
        ('30-50yr', 'Policy Data\nLifecycle'),
        ('67%', 'Insurers on\nLegacy Mainframes'),
        ('$80B+', 'Annual Claims\nFraud Globally'),
    ], height=72))
    story.append(Spacer(1, 10))

    story.append(CalloutBox([
        'A life insurance policy issued today must protect policyholder medical and financial data for 60+ years.',
        'Quantum computers capable of breaking RSA-2048 will exist within 10-20 years. The harvest-now,',
        'decrypt-later threat is not theoretical — nation-state actors are capturing ciphertext today.',
    ]))
    story.append(Spacer(1, 12))

    # ─── Quantum Exposure ─────────────────────────────────────────────────────
    story.append(SectionHeader(
        'The Insurance Industry\'s Quantum Exposure',
        'Harvest-Now, Decrypt-Later and the 30-50 Year Policy Lifecycle Risk'))
    story.append(Spacer(1, 10))

    story.append(Paragraph('The Harvest-Now, Decrypt-Later Threat', s['subsection']))
    story.append(Paragraph(
        'No other industry has a data retention problem like insurance. A 25-year-old purchasing a whole '
        'life insurance policy in 2026 generates a policyholder record that must remain confidential for '
        '60+ years. That record contains full medical underwriting history, financial assets and beneficiary '
        'designations, Social Security numbers and government IDs, actuarial risk classifications, and '
        'claims history across all lines of coverage.', s['body']))

    story.append(Paragraph(
        'Nation-state threat actors are systematically capturing encrypted insurance data today using the '
        'harvest-now, decrypt-later (HNDL) strategy. They archive ciphertext produced by RSA-2048 and '
        'ECC P-256, knowing that cryptographically relevant quantum computers will break these algorithms '
        'within 10-20 years. NIST finalised Post-Quantum Cryptography standards in August 2024 '
        '(FIPS 203/204/205). The transition has started.', s['body']))

    story.append(Paragraph('Why Insurance is a Priority Target', s['subsection']))
    story.append(Paragraph(
        'Insurance companies hold the most complete picture of individual financial and health status of '
        'any private sector entity. The combination of health, financial, and family data in a single '
        'policyholder record is uniquely valuable for identity fraud, social engineering, and targeted '
        'extortion — and uniquely dangerous during the quantum transition.', s['body']))

    threat_data = [
        ['Data Type', 'Retention', 'PQC Standard', 'Priority'],
        ['Life/Annuity Policyholder Records', '30-60 years', 'ML-KEM-1024 + ML-DSA-87', 'Critical'],
        ['Health Insurance Subscriber Records', '10-20 years', 'ML-KEM-768 + ML-DSA-65', 'High'],
        ['P&C Policy Records', '7-10 years', 'ML-KEM-768', 'High'],
        ['Claims Payment Records', '7 years', 'ML-KEM-512', 'Medium'],
        ['Actuarial Model Data', 'Indefinite', 'ML-KEM-1024 + SLH-DSA', 'Critical'],
        ['Reinsurance Treaty Data', '20-30 years', 'ML-KEM-768 + ML-DSA-65', 'High'],
    ]
    story.append(make_table(threat_data, [CONTENT_W*0.34, CONTENT_W*0.18, CONTENT_W*0.32, CONTENT_W*0.16]))
    story.append(Spacer(1, 12))

    # ─── Three Critical Threats ───────────────────────────────────────────────
    story.append(SectionHeader(
        'Three Critical Threats Facing the Insurance Sector',
        'Long-Term Data | Legacy Systems | Claims and Reinsurance Fraud'))
    story.append(Spacer(1, 10))

    for threat_num, threat_title, threat_body in [
        ('Threat 1',
         'Long-Term Policyholder Data Exposure',
         ['Insurance policy records span 30-50 years. Data encrypted today with RSA or ECC will be ',
          'decryptable by quantum computers within that window. Harvest-now, decrypt-later attacks are ',
          'capturing policy data in transit right now — banking ciphertext for future decryption.',
          '',
          'Every insurer with active whole life, annuity, long-term care, or disability policies is at risk. ',
          'Health insurers holding multi-decade subscriber records and commercial insurers with D&O or ',
          'professional liability policies are equally exposed.',
          '',
          'NY DFS 500 requires encryption of NPI in transit and at rest. NAIC Model Law requires similar ',
          'protections. Under Solvency II, EU-domiciled insurers must demonstrate current best practice ',
          'protection — which now includes quantum-safe cryptography post-NIST PQC finalisation.',
         ]),
        ('Threat 2',
         'Legacy Policy Administration System Vulnerabilities',
         ['67% of insurers run policy administration systems on mainframe platforms that predate modern ',
          'cryptography — IBM System z, Unisys ClearPath, and COBOL-based systems using TN3270e terminal ',
          'emulation. Communication between policy admin and downstream systems is often unencrypted.',
          '',
          'Attack vectors include unencrypted TN3270e sessions, legacy ACORD XML over HTTP, cleartext ',
          'X12 EDI 837 submissions from TPAs, and internal lateral movement that exploits the absence ',
          'of per-session encryption on mainframe traffic.',
          '',
          'Mainframe policy systems cannot be re-architected — they process millions of policies daily. ',
          'Any security solution must operate as a transparent wrapper, not a replacement.',
         ]),
        ('Threat 3',
         'Claims and Reinsurance Fraud',
         ['Insurance fraud costs the global industry $80B+ annually. Synthetic identity attacks use ',
          'composite identities combining real SSNs with fabricated credentials to obtain policies and ',
          'file fraudulent claims. Staged accident rings are detectable as anomalous claims networks.',
          '',
          'Reinsurance fraud includes falsified loss data in SWIFT settlement messages, business email ',
          'compromise targeting high-value wire transfers for catastrophe loss settlements, catastrophe ',
          'bond manipulation, and retrocession structure fraud.',
          '',
          'QBITEL Bridge detects fraud at the wire level in real time — before claims are paid — providing ',
          'a prevention layer that post-payment analytics cannot match.',
         ]),
    ]:
        story.append(ScenarioBox(threat_num, threat_title, threat_body))
        story.append(Spacer(1, 8))

    story.append(PageBreak())

    # ─── 7 Capabilities ───────────────────────────────────────────────────────
    story.append(SectionHeader(
        '7 Core Capabilities for Insurance',
        'Protocol Security | Fraud Detection | Compliance | Mainframe Shield'))
    story.append(Spacer(1, 10))

    capabilities = [
        ('1', 'Long-Term Policyholder Data Protection',
         'The foundational challenge in insurance cryptography is the mismatch between algorithm lifespans ',
         [
             'Classify policyholder data by retention horizon — life, annuity, LTC, disability records identified for extended protection',
             'Apply CRYSTALS-Kyber (ML-KEM) key encapsulation for all policy data in transit, replacing RSA and ECC',
             'Wrap long-term storage with hybrid PQC+classical schemes ensuring quantum-safe forward secrecy',
             'Implement cryptographic agility — rotate PQC algorithms without disrupting policy data access',
             'Maintain per-policy key provenance logs suitable for regulatory audit and litigation hold',
         ]),
        ('2', 'ACORD / X12 EDI Protocol Security',
         'ACORD XML and X12 EDI carry policy issuance, claims submission, premium remittance, and reinsurance bordereau data.',
         [
             'PQC wrapping of all ACORD XML message streams — inbound from agents/MGAs, outbound to reinsurers',
             'ML-DSA signature verification replacing SHA-1/MD5-based XML Digital Signatures on legacy integration points',
             'X12 EDI 834 (enrollment): ML-KEM PII protection, synthetic enrollment attack detection',
             'X12 EDI 835 (remittance): payment data integrity verification, payment redirection fraud detection',
             'X12 EDI 837 (claims): real-time integrity verification, upcoding and duplicate detection at the EDI layer',
             'HL7 v2.x and FHIR R4 PQC wrapping with PHI detection and HIPAA minimum necessary enforcement',
         ]),
        ('3', 'Mainframe Policy System Shield',
         'IBM System z and equivalent mainframes run COBOL policy engines communicating via TN3270e — typically unencrypted.',
         [
             'Zero-Touch Deployment: transparent proxy with no mainframe code changes and no downtime',
             'TN3270e PQC Wrapping: every terminal session wrapped in ML-KEM with less than 0.8ms added latency',
             'Lateral Movement Prevention: session-level microsegmentation preventing perimeter breach escalation',
             'Privileged Session Monitoring: AI monitoring for privilege escalation, bulk exports, off-hours anomalies',
             'Native support: SNA/APPC, TN3270e, CICS transaction flows, JES job streams, IBM MQ policy messaging',
         ]),
        ('4', 'Claims Fraud Detection and Prevention',
         'AI-powered protocol analysis detects fraud at the wire level — before claims are processed — not after payment.',
         [
             'Synthetic Identity Detection: cross-references 837 EDI identity signals against behavioral anomaly baselines',
             'Claims Network Analysis: detects organised fraud rings via coordinated submission pattern identification',
             'Medical Billing Integrity: real-time upcoding, unbundling, and phantom billing detection in 837 EDI',
             'Duplicate Claim Detection: cross-carrier detection using privacy-preserving hashed claim fingerprints',
             'SWIFT Message Integrity: real-time analysis of reinsurance settlement messages for manipulation',
             'Wire Fraud Prevention: ML-based BEC detection in SWIFT payment instruction chains for CAT settlements',
         ]),
        ('5', 'Reinsurance Settlement Security',
         'Catastrophe loss settlements may involve hundreds of millions in SWIFT wire transfers — a primary target for financial crime.',
         [
             'SWIFT PQC: all SWIFT MT and MX (ISO 20022) reinsurance settlement messages protected with post-quantum cryptography',
             'ISO 20022 premium payment flows receive ML-KEM protection and ML-DSA message authentication',
             'FIX Protocol Security: ILS trading activity receives quantum-safe session encryption',
             'Treaty Data Protection: treaty terms and profit commissions receive long-term PQC matching treaty duration',
             'Settlement Integrity: cryptographic chaining creates tamper-evident audit trails for loss settlements',
         ]),
        ('6', 'Solvency II / NY DFS 500 Compliance',
         'Automated compliance evidence generation for every major insurance regulatory framework.',
         [
             'Solvency II Pillar I: evidence of adequate data security controls for SCR model data protection',
             'Solvency II Pillar II/III: ORSA cybersecurity documentation and automated SFCR evidence generation',
             'NY DFS 500 Section 500.15: quantum-safe encryption of all NPI in transit and at rest',
             'NY DFS 500 Section 500.17: automated incident detection supporting 72-hour breach notification',
             'NAIC Model Law, HIPAA Security Rule, GDPR/CCPA, PCI-DSS v4.0, IFRS 17, SOC 2 Type II',
         ]),
        ('7', 'Actuarial Data Integrity',
         'Actuarial models represent decades of proprietary IP — pricing algorithms, mortality tables, CAT models, loss development factors.',
         [
             'Model Data Encryption: all actuarial data encrypted with ML-KEM-1024 in transit between workstations and model servers',
             'Integrity Verification: ML-DSA signatures on model outputs create tamper-evident audit trails',
             'Access Anomaly Detection: AI monitoring identifies bulk export events, unusual queries, lateral movement',
             'Model Theft Prevention: protocol-layer DLP capabilities detect unauthorised large data transfers',
             'Reserve Data Protection: IBNR calculations receive cryptographic integrity verification',
         ]),
    ]

    for cap_num, cap_title, cap_intro, cap_bullets in capabilities:
        story.append(KeepTogether([
            Paragraph(f'Capability {cap_num}: {cap_title}', s['subsection']),
            Paragraph(cap_intro, s['body']),
        ]))
        for bullet in cap_bullets:
            story.append(Paragraph(f'<bullet>&bull;</bullet> {bullet}', s['bullet']))
        story.append(Spacer(1, 10))

    story.append(PageBreak())

    # ─── Compliance Matrix ────────────────────────────────────────────────────
    story.append(SectionHeader(
        'Compliance Coverage Matrix',
        'Automated Evidence Across All Major Insurance Regulatory Frameworks'))
    story.append(Spacer(1, 10))

    compliance_data = [
        ['Regulatory Framework', 'Jurisdiction', 'Key Requirement', 'QBITEL Coverage'],
        ['Solvency II', 'EU / EEA', 'ICT security, ORSA controls, data protection', 'Full — automated evidence'],
        ['NY DFS 500 (23 NYCRR 500)', 'New York', 'Encrypt NPI, MFA, incident response', 'Full — PQC + monitoring'],
        ['NAIC Cybersecurity Model Law', '24+ US States', 'Cybersecurity program, NPI protection', 'Full — protocol controls'],
        ['HIPAA Security Rule', 'US (Health)', 'PHI encryption, access controls, audit', 'Full — PHI detection'],
        ['GDPR', 'EU / EEA', 'Personal data protection, breach notification', 'Full — PII classification'],
        ['CCPA / CPRA', 'California', 'Consumer data protection, security', 'Full — automated controls'],
        ['PCI-DSS v4.0', 'Global', 'Premium payment encryption (Req 3 & 4)', 'Full — payment protection'],
        ['IFRS 17', 'Global', 'Insurance contract data integrity', 'Full — audit trail chain'],
        ['SOC 2 Type II', 'Global', 'Security trust service criteria', 'Full — continuous evidence'],
        ['NIST CSF 2.0', 'US Federal', 'Cybersecurity framework alignment', 'Full — all 5 functions'],
    ]
    story.append(make_table(compliance_data,
                            [CONTENT_W*0.28, CONTENT_W*0.18, CONTENT_W*0.32, CONTENT_W*0.22]))
    story.append(Spacer(1, 14))

    # ─── Integration Ecosystem ────────────────────────────────────────────────
    story.append(SectionHeader(
        'Integration Ecosystem',
        'Native Connectors for Leading Insurance Technology Platforms'))
    story.append(Spacer(1, 10))

    integration_data = [
        ['Platform Category', 'Supported Platforms', 'Integration Method'],
        ['Policy Administration',
         'Guidewire (PolicyCenter, ClaimCenter, BillingCenter),\nDuck Creek, Majesco, SAP Insurance, Sapiens',
         'Native API + EDI wrapping'],
        ['EDI & B2B Integration',
         'IBM Sterling B2B Integrator, OpenText Trading Grid,\nEdifecs SpecBuilder/XEngine',
         'Inline transparent proxy'],
        ['Claims Management',
         'Snapsheet AI Claims, Tractable, Mitchell International RepairCenter',
         'API-layer PQC wrapping'],
        ['Reinsurance',
         'SWIFT Network (MT/MX), FIX Protocol for ILS,\nRMS/Moodys CAT model data exchange',
         'Protocol-layer overlay'],
        ['Health Insurance',
         'HL7 v2.x, FHIR R4, X12 clearinghouses,\nCMS integration points',
         'PHI-aware wrapping'],
    ]
    story.append(make_table(integration_data,
                            [CONTENT_W*0.24, CONTENT_W*0.46, CONTENT_W*0.30]))
    story.append(Spacer(1, 14))

    # ─── Deployment Timeline ──────────────────────────────────────────────────
    story.append(SectionHeader(
        'Deployment Timeline',
        'Zero-Downtime Deployment in Three Structured Phases'))
    story.append(Spacer(1, 10))

    for phase, title, duration, items, deliverable in [
        ('Phase 1', 'Protocol Discovery and Assessment', 'Days 1-3',
         ['Passive network tap — no traffic disruption',
          'Automated discovery of ACORD XML, X12 EDI, HL7, SWIFT, mainframe protocols',
          'Data classification: all policyholder PII, PHI, NPI, and financial data streams identified',
          'Risk heat map: protocols ranked by sensitivity and quantum exposure',
          'Regulatory gap analysis: Solvency II, NY DFS 500, NAIC, HIPAA mapping'],
         'Insurance Protocol Security Assessment Report'),
        ('Phase 2', 'PQC Overlay and Shield Deployment', 'Days 4-14',
         ['HSM provisioning and PQC key generation (FIPS 140-3 Level 3)',
          'Bridge inline deployment — transparent insertion into protocol paths',
          'Mainframe Shield activation for TN3270e and legacy protocol streams',
          'ACORD XML and X12 EDI PQC wrapping activated by trading partner',
          'Fraud detection model activation calibrated to carrier claims mix',
          'Compliance monitoring dashboard activated'],
         'Go-Live Confirmation with Baseline Metrics'),
        ('Phase 3', 'Optimisation and Compliance Reporting', 'Days 15-30',
         ['Fraud detection model tuning based on carrier-specific claims patterns',
          'Regulatory evidence package generation (Solvency II SFCR, NY DFS 500, NAIC)',
          'Trading partner quantum-safe certificate distribution',
          'Actuarial team briefing and model protection validation',
          'Reinsurance counterparty coordination for quantum-safe settlement channels'],
         'First Compliance Evidence Package + 90-Day Roadmap'),
    ]:
        story.append(KeepTogether([
            ColorBar(3, GOLD),
            Spacer(1, 4),
            Paragraph(f'<b>{phase}: {title}</b> — {duration}', s['subsection2']),
        ]))
        for item in items:
            story.append(Paragraph(f'<bullet>&bull;</bullet> {item}', s['bullet']))
        story.append(Paragraph(f'<b>Deliverable:</b> {deliverable}', s['body_left']))
        story.append(Spacer(1, 8))

    story.append(PageBreak())

    # ─── Performance Specifications ───────────────────────────────────────────
    story.append(SectionHeader(
        'Performance Specifications',
        'Validated Enterprise Performance Benchmarks'))
    story.append(Spacer(1, 10))

    perf_data = [
        ['Performance Metric', 'QBITEL Bridge Result'],
        ['Protocol Discovery Accuracy', '89%+ including undocumented legacy variants'],
        ['PQC Encryption Overhead', 'Less than 1.2ms per session establishment'],
        ['Claims EDI Processing Latency', 'Less than 0.8ms added on 837 streams'],
        ['TN3270e Session Overhead', 'Less than 0.8ms per session'],
        ['SWIFT Message Processing', 'Less than 1.5ms per MT/MX message'],
        ['Autonomous Threat Response', '78% resolved without human escalation'],
        ['Fraud Detection Accuracy', 'Greater than 94% precision on synthetic identity patterns'],
        ['Policy Transaction Throughput', '2M+ transactions per day validated on mainframe environments'],
        ['HSM Key Operations', 'FIPS 140-3 Level 3, 100,000+ operations per second'],
        ['Availability SLA', '99.99% with active-active high availability clustering'],
        ['Deployment Time', '4-6 hours for initial go-live; full deployment in 30 days'],
    ]
    story.append(make_table(perf_data, [CONTENT_W * 0.50, CONTENT_W * 0.50]))
    story.append(Spacer(1, 14))

    # ─── Competitive Differentiation ─────────────────────────────────────────
    story.append(SectionHeader(
        'Competitive Differentiation',
        'Why Generic Security Solutions Fall Short for Insurance'))
    story.append(Spacer(1, 10))

    comp_data = [
        ['Competitor Category', 'Limitation', 'QBITEL Bridge Advantage'],
        ['Traditional Network Security', 'Layers 3-4 only — no ACORD/X12/TN3270e visibility',
         'Layer 7 insurance protocol awareness and fraud detection'],
        ['Cloud Encryption Services', 'Protects data at rest — not in-transit protocol streams',
         'Protocol-layer PQC wrapping including mainframe and SWIFT'],
        ['Fraud Analytics Platforms', 'Post-payment analytics — fraud detected after claims paid',
         'Real-time, pre-adjudication, inline detection at the EDI wire'],
        ['Legacy PKI / Cert Management', 'No quantum-safe cryptography — RSA/ECC only',
         'NIST FIPS 203/204/205 PQC deployed today with crypto agility'],
        ['Generic EDI Security', 'No ACORD awareness, no mainframe, no SWIFT coverage',
         'Complete insurance protocol stack: EDI, ACORD, mainframe, SWIFT'],
    ]
    story.append(make_table(comp_data,
                            [CONTENT_W*0.25, CONTENT_W*0.37, CONTENT_W*0.38]))
    story.append(Spacer(1, 14))

    # ─── Customer Scenarios ───────────────────────────────────────────────────
    story.append(SectionHeader(
        'Customer Scenarios',
        'P&C Insurer | Life Insurer | Global Reinsurer'))
    story.append(Spacer(1, 10))

    story.append(ScenarioBox(
        'Scenario A',
        'Regional P&C Insurer — Claims Fraud and EDI Security',
        [
            'Profile: Mid-size P&C insurer. 1.2M policies. $800M annual claims volume. Duck Creek on-premises.',
            'Challenge: 8% claims fraud losses. Three EDI trading partner incidents in 4 years. NY DFS 500 exam approaching.',
            '',
            'Solution: Bridge deployed inline on Duck Creek EDI in 6 hours with zero downtime.',
            'X12 837 real-time fraud detection identified 340+ suspicious claims in 30 days — synthetic identity',
            'ring across auto and workers compensation lines. NY DFS 500 Section 500.15 compliance achieved.',
            '',
            'Results: $4.2M in claims fraud prevented in 90 days. NY DFS 500 examination passed.',
        ]
    ))
    story.append(Spacer(1, 10))

    story.append(ScenarioBox(
        'Scenario B',
        'Large Life Insurer — Long-Term Data Protection and Mainframe Shield',
        [
            'Profile: National life/annuity insurer. 8M in-force policies. IBM System z. Solvency II EU subsidiary.',
            'Challenge: Board quantum threat concern. No mainframe code changes accepted. APT campaign targeting.',
            '',
            'Solution: Mainframe Shield as transparent proxy — TN3270e encrypted in less than 0.8ms.',
            '6.3M policyholder records classified and protected with ML-KEM-1024. SWIFT reinsurance flows',
            'protected. Solvency II SFCR cybersecurity evidence package generated automatically.',
            '',
            'Results: Solvency II regulator satisfied. Zero mainframe downtime. Three APT lateral movement',
            'attempts toward mainframe network segment detected and blocked.',
        ]
    ))
    story.append(Spacer(1, 10))

    story.append(ScenarioBox(
        'Scenario C',
        'Global Reinsurer — SWIFT Security and Catastrophe Settlement Integrity',
        [
            'Profile: Top-10 global reinsurer. $45B assumed premiums. 200+ cedants. $2.8B annual CAT SWIFT settlements.',
            'Challenge: Two near-miss SWIFT payment fraud events in 18 months. PRA, EIOPA, BMA pressure.',
            '',
            'Solution: SWIFT MT/MX PQC wrapping across all cedant settlement flows. Real-time amount and',
            'routing integrity verification. 200+ cedant treaty records protected with ML-KEM-1024.',
            'CAT bond trigger data integrity via ML-DSA cryptographic chaining.',
            '',
            'Results: Zero SWIFT fraud incidents post-deployment. $180M CAT settlement protected.',
            'Three regulatory jurisdiction examinations (PRA, EIOPA, BMA) passed with Bridge evidence packages.',
        ]
    ))
    story.append(Spacer(1, 14))

    # ─── Next Steps ───────────────────────────────────────────────────────────
    story.append(SectionHeader('Next Steps & Contact', 'Start Your Insurance Quantum-Safe Journey'))
    story.append(Spacer(1, 10))

    steps = [
        ('01', 'Executive Briefing', '30 minutes',
         'CRO/CISO-level session covering the quantum threat timeline for insurance, NY DFS 500 and '         'Solvency II developments post-NIST PQC, and QBITEL Bridge capabilities. No technical prerequisites.'),
        ('02', 'Protocol Discovery Assessment', '3 days, non-disruptive',
         'Passive tap assessment providing a full protocol inventory, data classification heat map, and '         'regulatory compliance gap analysis. Deliverable: Insurance Protocol Security Assessment Report '         'suitable for board presentation and regulatory examination preparation.'),
        ('03', 'Proof of Value Pilot', '30 days, production scope',
         'Full Bridge deployment on a defined scope — single line of business, policy admin integration '         'layer, or mainframe segment. Includes fraud detection activation, PQC wrapping, and compliance '         'evidence generation. Full production conversion upon pilot success.'),
    ]

    for step_num, step_title, step_time, step_body in steps:
        story.append(KeepTogether([
            ColorBar(3, TEAL),
            Spacer(1, 4),
            Paragraph(f'<b>Step {step_num}: {step_title}</b> ({step_time})', s['subsection2']),
            Paragraph(step_body, s['body']),
        ]))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 10))
    story.append(ColorBar(2, GOLD))
    story.append(Spacer(1, 12))

    story.append(Paragraph('QBITEL Enterprise Insurance Practice', s['contact']))
    story.append(Paragraph('enterprise@qbitel.com  |  https://bridge.qbitel.com', s['contact']))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        'QBITEL holds SOC 2 Type II certification. QBITEL Bridge is validated against NIST FIPS 203 '        '(ML-KEM), FIPS 204 (ML-DSA), and FIPS 205 (SLH-DSA). All insurance client engagements '        'are conducted under mutual NDA. Response within 4 business hours for insurance sector enquiries.',
        s['body']))
    story.append(Spacer(1, 10))
    story.append(ColorBar(3, NAVY))

    return story


def build_doc(output_path):
    doc = BaseDocTemplate(
        output_path, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN)
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H,
                        leftPadding=0, rightPadding=0,
                        topPadding=0, bottomPadding=0, id='cover')
    inner_frame = Frame(MARGIN, 0.7 * inch, CONTENT_W,
                        PAGE_H - MARGIN - 0.7 * inch, id='inner')
    cover_template = PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover)
    inner_template = PageTemplate(id='Inner', frames=[inner_frame], onPage=draw_page)
    doc.addPageTemplates([cover_template, inner_template])

    styles = get_styles()
    story = []
    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())
    story.extend(build_story(styles))

    doc.build(story)
    print(f'PDF generated: {output_path}')


if __name__ == '__main__':
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'QBITEL_Bridge_Insurance_Marketing_Pitch.pdf')
    build_doc(out)
