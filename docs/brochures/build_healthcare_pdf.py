"""
Build QBITEL Bridge Healthcare & Medical Devices Marketing Pitch - Professional PDF
Uses ReportLab for full layout/design control.
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
            c.drawString(x + (box_w - text_w) / 2, self.h - 34, big)
            c.setFillColor(WHITE_C)
            c.setFont('Helvetica', 8)
            for j, line in enumerate(small.split('\n')):
                lw = c.stringWidth(line, 'Helvetica', 8)
                c.drawString(x + (box_w - lw) / 2, self.h - 50 - j * 11, line)


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
        label_w = c.stringWidth(self.label + '  >', 'Helvetica-Bold', 8)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 11)
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
        c.drawString(14, self.h - 20, '+')
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Oblique', 9.5)
        y = self.h - 18
        for line in self.text_lines:
            c.drawString(30, y, line)
            y -= 14


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
    canvas.drawString(MARGIN + 1.1 * inch, PAGE_H - 0.32 * inch,
                      'HEALTHCARE & MEDICAL DEVICES')
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
                      'Confidential -- For Authorized Recipients Only  |  (c) 2025 QBITEL. All Rights Reserved.')
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

    # Gold diagonal accent (top-right)
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

    # Teal dark bottom strip
    canvas.setFillColor(TEAL_DARK)
    canvas.rect(0, 0, w, 1.6 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, 1.6 * inch, w, 5, fill=1, stroke=0)

    # QBITEL BRIDGE title
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
                      'HEALTHCARE & MEDICAL DEVICES')

    # Tagline
    canvas.setFillColor(HexColor('#CCDDE8'))
    canvas.setFont('Helvetica-Oblique', 13)
    canvas.drawString(MARGIN, h * 0.68 - 128,
                      'Post-Quantum Security for Clinical Networks and Medical Devices')

    # 4 key metrics boxes
    metrics = [
        ('6.2', 'Avg Device\nVulnerabilities'),
        ('<1ms', 'Crypto\nOverhead'),
        ('Zero', 'FDA\nRecertification'),
        ('<10min', 'HIPAA Audit\nReport'),
    ]
    box_w = (CONTENT_W - 3 * 0.12 * inch) / 4
    bx_start = MARGIN
    by = h * 0.36
    bh = 1.0 * inch

    for i, (big, small) in enumerate(metrics):
        bx = bx_start + i * (box_w + 0.12 * inch)
        bg_c = TEAL if i % 2 == 0 else LIGHT_NAVY
        canvas.setFillColor(bg_c)
        canvas.roundRect(bx, by, box_w, bh, 6, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.rect(bx, by + bh - 5, box_w, 5, fill=1, stroke=0)
        canvas.setFillColor(GOLD if bg_c == LIGHT_NAVY else WHITE_C)
        canvas.setFont('Helvetica-Bold', 19)
        tw = canvas.stringWidth(big, 'Helvetica-Bold', 19)
        canvas.drawString(bx + (box_w - tw) / 2, by + bh - 30, big)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 7.5)
        for j, line in enumerate(small.split('\n')):
            lw = canvas.stringWidth(line, 'Helvetica', 7.5)
            canvas.drawString(bx + (box_w - lw) / 2, by + bh - 46 - j * 11, line)

    # Second stat row
    metrics2 = [
        ('$10.9M', 'Avg Healthcare\nBreach Cost'),
        ('18 Months', 'Avg Time to\nDetect Breach'),
        ('100%', 'PHI Transmissions\nVulnerable Today'),
        ('$1.3B', 'HIPAA Fines\nin 2024'),
    ]
    by2 = h * 0.26
    for i, (big, small) in enumerate(metrics2):
        bx = bx_start + i * (box_w + 0.12 * inch)
        canvas.setFillColor(LIGHT_BG)
        canvas.roundRect(bx, by2, box_w, 0.85 * inch, 6, fill=1, stroke=0)
        canvas.setFillColor(NAVY)
        canvas.setFont('Helvetica-Bold', 14)
        tw = canvas.stringWidth(big, 'Helvetica-Bold', 14)
        canvas.drawString(bx + (box_w - tw) / 2, by2 + 0.56 * inch, big)
        canvas.setFillColor(MID_GREY)
        canvas.setFont('Helvetica', 7)
        for j, line in enumerate(small.split('\n')):
            lw = canvas.stringWidth(line, 'Helvetica', 7)
            canvas.drawString(bx + (box_w - lw) / 2, by2 + 0.32 * inch - j * 10, line)

    # Bottom tag line
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(MARGIN, 0.85 * inch,
                      'Non-Invasive  *  Zero Firmware Changes  *  Zero FDA Recertification  *  HIPAA Automated')
    canvas.setFont('Helvetica', 8.5)
    canvas.setFillColor(HexColor('#AABBCC'))
    canvas.drawString(MARGIN, 0.55 * inch,
                      'Confidential Marketing Document  |  Version 1.0  |  2025')
    canvas.restoreState()


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
        textColor=NAVY, spaceAfter=5, spaceBefore=14
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


def make_table(headers, rows, col_widths, s):
    data = [[Paragraph(h, s['table_header']) for h in headers]]
    for ri, row in enumerate(rows):
        cells = []
        for ci, cell_text in enumerate(row):
            style = s['table_cell_bold'] if ci == 0 else s['table_cell']
            cells.append(Paragraph(cell_text, style))
        data.append(cells)
    n_rows = len(data)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE_C),
        *[('ROWBACKGROUND', (0, i), (-1, i), TABLE_ALT if i % 2 == 1 else WHITE_C)
          for i in range(1, n_rows)],
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#C8D8E8')),
        ('LINEBELOW', (0, -1), (-1, -1), 1, TEAL),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ])
    tbl = Table(data, colWidths=[w * inch for w in col_widths])
    tbl.setStyle(style)
    return tbl


def sp(n=8):
    return Spacer(1, n)


def sub(num, title, S):
    return Paragraph(
        f'<font color="#F0A500"><b>{num}</b></font>  '
        f'<font color="#0D1B3E"><b>{title}</b></font>',
        S['subsection']
    )


def build_pdf():
    out_path = 'docs/brochures/QBITEL_Bridge_Healthcare_Marketing_Pitch.pdf'

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

    cover_template = PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover)
    inner_template = PageTemplate(id='Inner', frames=[inner_frame], onPage=draw_page)
    doc.addPageTemplates([cover_template, inner_template])

    S = get_styles()
    story = []

    # Cover page - drawn by callback, just switch template
    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())

    # SECTION 1: Executive Summary
    story.append(SectionHeader(
        '1. Executive Summary',
        'The healthcare security crisis demands immediate action'
    ))
    story.append(sp(8))
    story.append(CalloutBox([
        'The average connected medical device carries 6.2 vulnerabilities.',
        'PHI sells for $250-$1,000/record on dark web markets.',
        'A single healthcare breach costs $10.9M on average.',
        'QBITEL Bridge closes every gap -- without touching a single line of device firmware.',
    ]))
    story.append(sp(10))
    story.append(Paragraph(
        'Healthcare is the most targeted sector in cybersecurity - and the least protected at the device layer. '
        'Clinical networks connect thousands of FDA-cleared devices running decade-old software, exchanging the '
        'most sensitive data in existence across protocols designed before modern threats existed. Today, 100% '
        'of PHI transmissions are vulnerable to harvest-now-decrypt-later quantum attacks.',
        S['body']
    ))
    story.append(sp(6))
    story.append(StatBlock([
        ('6.2', 'Avg Device\nVulnerabilities'),
        ('<1ms', 'Crypto\nOverhead'),
        ('Zero', 'FDA\nRecertification'),
        ('<10min', 'HIPAA Audit\nReport'),
    ]))
    story.append(sp(10))

    story.append(sub('1.1', 'What QBITEL Bridge Delivers', S))
    bullets = [
        'Zero-touch device protection for every connected medical device on your network',
        'Post-quantum HL7/FHIR/DICOM security for all clinical data flows',
        'HIPAA compliance automation with audit reports in under 10 minutes',
        'Less than 1ms cryptographic overhead -- invisible to clinicians and patients',
        '100% device coverage regardless of device age, manufacturer, or embedded OS',
        'No FDA recertification triggered -- ever',
    ]
    for b in bullets:
        story.append(Paragraph(f'<bullet>&bull;</bullet> {b}', S['bullet']))
    story.append(PageBreak())

    # SECTION 2: Healthcare Security Crisis
    story.append(SectionHeader(
        '2. The Healthcare Security Crisis',
        'Three converging threats your organization cannot ignore'
    ))
    story.append(sp(8))

    story.append(sub('2.1', 'Threat 1: The Connected Device Explosion', S))
    story.append(Paragraph(
        'Modern hospitals operate 10,000 to 50,000 connected medical devices: infusion pumps, patient monitors, '
        'imaging systems, ventilators, implantable device programmers, and hundreds of specialized clinical '
        'instruments. Each device is a potential attack vector with no endpoint protection possible.',
        S['body']
    ))
    story.append(make_table(
        ['Threat Indicator', 'Statistic', 'Business Impact'],
        [
            ['Avg vulnerabilities per device', '6.2 (Claroty 2024)', '$10.9M avg breach cost'],
            ['Devices on end-of-life OS', '53% -- no patch path', 'Permanently unpatched attack surface'],
            ['Orgs with device security incident', '89% in past 2 years', 'Near-certainty of compromise'],
            ['Avg breach detection time', '18 months', 'Adversary dwell time exceeds policy'],
            ['PHI transmissions vulnerable', '100% today', 'Total quantum harvest exposure'],
        ],
        [2.3, 1.8, 2.9],
        S
    ))
    story.append(sp(10))

    story.append(sub('2.2', 'Threat 2: The Quantum PHI Harvest Threat', S))
    story.append(Paragraph(
        'PHI retains its value for decades: genetic data, chronic condition histories, insurance details, '
        'and identity information remain exploitable across a patient\'s lifetime. Nation-state adversaries '
        'are actively harvesting encrypted PHI today using harvest-now-decrypt-later strategies.',
        S['body']
    ))
    threat_items = [
        ('<b><font color="#008B9A">$250-$1,000 per PHI record:</font></b>',
         'PHI commands the highest price of any data type on dark web markets -- 10x financial records.'),
        ('<b><font color="#008B9A">RSA-2048 breaks in approximately 8 hours</font></b>',
         'on a cryptographically relevant quantum computer. NIST projects availability within 10-15 years.'),
        ('<b><font color="#008B9A">30-50 year PHI exposure window:</font></b>',
         'Genetic records and chronic disease histories stored today will be decryptable within the quantum timeline.'),
        ('<b><font color="#008B9A">$1.3B in HIPAA fines in 2024:</font></b>',
         'Regulators are escalating enforcement. Post-quantum compliance expectations are forming now.'),
    ]
    for label, text in threat_items:
        story.append(Paragraph(f'<bullet>&bull;</bullet> {label} {text}', S['bullet']))
        story.append(sp(3))
    story.append(sp(8))

    story.append(sub('2.3', 'Threat 3: The FDA Recertification Barrier', S))
    story.append(Paragraph(
        'FDA-cleared medical devices cannot be modified without triggering recertification. '
        'Recertification costs $500K to $2M per device class and takes 12-36 months. '
        'This regulatory constraint has left security teams unable to deploy endpoint protection '
        'on the very devices most at risk. QBITEL Bridge solves this problem categorically.',
        S['body']
    ))
    story.append(make_table(
        ['Barrier', 'Cost/Time', 'QBITEL Solution'],
        [
            ['FDA recertification per device class', '$500K-$2M, 12-36 months', 'Non-invasive wrapper -- zero recertification'],
            ['Endpoint agent installation', 'Triggers recertification', 'Network-layer protection -- no agent needed'],
            ['Firmware modification', 'FDA 510(k) re-submission', 'Zero firmware changes -- ever'],
            ['Legacy OS devices', 'No patch path available', '100% coverage regardless of OS version'],
        ],
        [2.4, 2.0, 2.6],
        S
    ))
    story.append(PageBreak())

    # SECTION 3: Seven Deep-Dive Capabilities
    story.append(SectionHeader(
        '3. Seven Deep-Dive Capabilities',
        'Purpose-built for the unique security requirements of clinical environments'
    ))
    story.append(sp(8))

    story.append(sub('3.1', 'Non-Invasive Medical Device Shield', S))
    story.append(Paragraph(
        'Bridge deploys a network-layer wrapper that operates entirely outside the device boundary. '
        'Traffic from protected devices is intercepted at the network switch level, encrypted using '
        'ML-KEM-512, and forwarded through an authenticated quantum-safe channel -- all before '
        'it reaches the clinical LAN. No firmware. No agents. No recertification.',
        S['body']
    ))
    story.append(make_table(
        ['Technical Element', 'Specification', 'Clinical Benefit'],
        [
            ['Interception method', 'IEEE 802.1X network-layer', 'Zero device modification required'],
            ['Key encapsulation', 'ML-KEM-512 (FIPS 203)', 'Less than 1ms establishment overhead'],
            ['Key storage', 'HSM-backed -- never in RAM', 'Quantum-safe key protection'],
            ['Device fingerprinting', 'Automated baseline ML', '100% device inventory visibility'],
            ['Device coverage', '100% -- all manufacturers/OS', 'Devices from 1995 onward protected'],
            ['VLAN compatibility', 'Full clinical VLAN support', 'Existing segmentation preserved'],
        ],
        [2.2, 2.2, 2.6],
        S
    ))
    story.append(sp(8))

    story.append(sub('3.2', 'HL7/FHIR Secure Interoperability', S))
    story.append(Paragraph(
        'Bridge\'s Protocol Security Layer wraps every HL7 v2/v3 and FHIR R4 communication in '
        'post-quantum encryption while maintaining complete protocol fidelity. ADT notifications, '
        'lab results, medication orders, and FHIR REST APIs are all protected transparently.',
        S['body']
    ))
    fhir_items = [
        'Full HL7 v2.x message parsing: ADT, ORM, ORU, MDM, RAS, MFN, SIU segments',
        'FHIR R4 resource-level encryption with SMART on FHIR token binding',
        'ML-DSA digital signatures on every clinical message for non-repudiation',
        'Sub-5ms end-to-end encryption overhead on standard HL7 message sizes',
        'Lossless protocol translation: Mirth Connect, Rhapsody, InterSystems Ensemble',
        'Every HL7/FHIR transaction logged with quantum-safe tamper-evident audit trail',
    ]
    for item in fhir_items:
        story.append(Paragraph(f'<bullet>&bull;</bullet> {item}', S['bullet']))
    story.append(sp(8))

    story.append(sub('3.3', 'DICOM Imaging Protection', S))
    story.append(Paragraph(
        'Medical imaging represents the largest PHI data volume in healthcare. DICOM files -- '
        'CT scans, MRI images, X-rays, ultrasounds, and pathology slides -- are transmitted '
        'using protocols designed in 1993. Bridge applies post-quantum encryption to all '
        'DICOM traffic with zero impact on image rendering latency.',
        S['body']
    ))
    story.append(make_table(
        ['DICOM Operation', 'Bridge Protection', 'Latency Overhead'],
        [
            ['C-STORE (image transfer)', 'ML-KEM-512 encrypted', 'Less than 2ms'],
            ['C-FIND (query)', 'Authenticated query channel', 'Less than 1ms'],
            ['C-MOVE (retrieve)', 'Encrypted transfer tunnel', 'Less than 2ms'],
            ['WADO-RS (web access)', 'HTTPS + PQC overlay', 'Less than 3ms'],
            ['Modality Worklist (MWL)', 'Signed + encrypted', 'Less than 1ms'],
        ],
        [2.2, 2.4, 2.4],
        S
    ))
    story.append(sp(8))

    story.append(sub('3.4', 'Lightweight PQC for Constrained Medical Devices', S))
    story.append(Paragraph(
        'Many medical devices operate with less than 64KB RAM. Standard post-quantum algorithms '
        'are too resource-intensive for these environments. Bridge implements ML-KEM-512 with '
        'all cryptographic workload offloaded to Bridge edge nodes -- the protected device '
        'handles less than 200 bytes of protocol framing overhead.',
        S['body']
    ))
    story.append(make_table(
        ['Device Class', 'RAM Profile', 'Bridge Mode', 'Battery Impact'],
        [
            ['High-capability (imaging, lab)', 'Greater than 1GB', 'Full PQC on-device assist', 'Negligible'],
            ['Mid-range (monitors, infusion)', '1-256MB', 'Hybrid edge offload', 'Less than 0.5%'],
            ['Constrained (wearables, sensors)', 'Less than 64KB', 'Full edge offload', 'Less than 0.1%'],
            ['Implantable programmers', 'Less than 16KB', 'Protocol proxy mode', 'Zero on device'],
        ],
        [2.2, 1.4, 1.8, 1.6],
        S
    ))
    story.append(sp(8))

    story.append(sub('3.5', 'Battery-Aware Cryptographic Scheduling', S))
    story.append(Paragraph(
        'Bridge\'s Battery-Aware Scheduler dynamically adjusts cryptographic operations based on '
        'device battery state, transmission priority, and clinical urgency classification. '
        'Critical alerts always transmit immediately. Routine telemetry batches during optimal windows.',
        S['body']
    ))
    battery_items = [
        'Critical alerts (arrhythmia, hypoxia, low battery): Immediate full-PQC -- no delay, no exception',
        'Routine vitals (normal range): Scheduled batch transmission during optimal battery windows',
        'Background telemetry: Compressed encrypted batches during charging or high-battery periods',
        'Emergency override: Clinical staff can force immediate full transmission of any data class',
        'Ambulatory cardiac monitors: Less than 4 additional hours battery drain per week',
        'Wireless infusion pumps: Less than 1% battery impact on 72-hour battery life',
    ]
    for item in battery_items:
        story.append(Paragraph(f'<bullet>&bull;</bullet> {item}', S['bullet']))
    story.append(sp(8))

    story.append(sub('3.6', 'HIPAA/FDA Compliance Automation', S))
    story.append(Paragraph(
        'Bridge\'s Compliance Engine automates the generation of HIPAA audit reports, FDA '
        'cybersecurity documentation, and HITRUST CSF evidence packages. Audit reports '
        'generated in less than 10 minutes -- compared to industry average of 40+ hours.',
        S['body']
    ))
    story.append(make_table(
        ['Compliance Function', 'Traditional Effort', 'With QBITEL Bridge'],
        [
            ['HIPAA 164.312 audit report', '40-160 hours/quarter', 'Less than 10 minutes, on-demand'],
            ['PHI access audit trail', 'Manual log review', 'Automated, tamper-evident'],
            ['FDA SBOM generation', 'Weeks of manual work', 'Automated per device class'],
            ['HITRUST CSF evidence', 'Multiple person-weeks', 'Continuous automated collection'],
            ['Breach notification prep', 'Days of investigation', '15-minute detection + report'],
            ['BAA tracking', 'Spreadsheet-based', 'Automated execution and tracking'],
        ],
        [2.4, 1.8, 2.8],
        S
    ))
    story.append(sp(8))

    story.append(sub('3.7', 'Clinical Network Anomaly Detection', S))
    story.append(Paragraph(
        'Bridge\'s ML-powered anomaly detection engine builds behavioral baselines for every '
        'protected device and alerts on deviations that indicate compromise, lateral movement, '
        'or data exfiltration. Compromised devices are automatically quarantined in VLAN '
        'isolation within 30 seconds.',
        S['body']
    ))
    anomaly_items = [
        'Per-device communication baseline: destination, volume, frequency, protocol',
        'Anomalous destination detection: device communicating to new IP or domain',
        'PHI volume anomalies: unusual data extraction from imaging or EHR systems',
        'Lateral movement detection: device attempting to reach segments outside its role',
        'Ransomware precursor detection: reconnaissance patterns, credential access attempts',
        'SIEM integration: Splunk, Microsoft Sentinel, IBM QRadar, Palo Alto XSIAM',
    ]
    for item in anomaly_items:
        story.append(Paragraph(f'<bullet>&bull;</bullet> {item}', S['bullet']))
    story.append(PageBreak())

    # SECTION 4: Compliance Coverage
    story.append(SectionHeader(
        '4. Compliance Coverage',
        'Automated evidence across every healthcare regulatory framework'
    ))
    story.append(sp(8))
    story.append(make_table(
        ['Regulation / Standard', 'QBITEL Bridge Coverage', 'Automation Level'],
        [
            ['HIPAA Security Rule (45 CFR 164.312)', 'Full technical safeguards', 'Automated audit reports'],
            ['HIPAA Breach Notification Rule', 'Breach detection + notification', '15-minute detection'],
            ['HITRUST CSF v11', 'Full control mapping', 'Continuous evidence collection'],
            ['FDA 21 CFR Part 11', 'Electronic records + signatures', 'Automated'],
            ['FDA Cybersecurity Guidance (Oct 2023)', 'SBOM + post-market surveillance', 'Automated documentation'],
            ['SOC 2 Type II', 'Security + Availability trust services', 'Continuous monitoring'],
            ['GDPR (cross-border PHI)', 'Encryption + data subject rights', 'Automated'],
            ['NIST CSF 2.0', 'Identify, Protect, Detect, Respond, Recover', 'Full mapping'],
            ['IEC 62443', 'Industrial/medical device cybersecurity', 'Network segmentation controls'],
            ['ISO 27001', 'Information security management', 'Evidence package'],
        ],
        [2.5, 2.3, 2.2],
        S
    ))
    story.append(sp(10))
    story.append(CalloutBox([
        'QBITEL Bridge is a HIPAA-compliant Business Associate. We execute BAAs with all healthcare customers.',
        'Every Bridge deployment includes a dedicated HIPAA compliance dashboard and automated audit trail.',
        'First healthcare vendor to achieve quantum-safe + HITRUST CSF dual certification.',
    ]))
    story.append(PageBreak())

    # SECTION 5: Integration Ecosystem
    story.append(SectionHeader(
        '5. Integration Ecosystem',
        'Works with everything your clinical environment already runs'
    ))
    story.append(sp(8))

    story.append(sub('5.1', 'Electronic Health Record Systems', S))
    story.append(make_table(
        ['EHR Platform', 'Integration Method', 'Capabilities'],
        [
            ['Epic Systems', 'SMART on FHIR + Interconnect API', 'MyChart, Interconnect, Cosmos'],
            ['Oracle Cerner', 'Millennium API + CareAware', 'Device integration, HL7 feeds'],
            ['MEDITECH Expanse', 'FHIR R4 + MAGIC legacy', 'Expanse web, legacy MAGIC support'],
            ['athenahealth', 'athenaNet API', 'Cloud EHR, revenue cycle protection'],
            ['Allscripts/Veradigm', 'Sunrise Clinical Manager', 'Professional EHR, analytics'],
        ],
        [1.8, 2.2, 3.0],
        S
    ))
    story.append(sp(8))

    story.append(sub('5.2', 'Medical Device Manufacturers', S))
    story.append(make_table(
        ['Manufacturer', 'Device Categories', 'Integration Level'],
        [
            ['GE Healthcare', 'Imaging, patient monitoring, ECG', 'Native protocol adapter'],
            ['Philips', 'IntelliVue monitoring, imaging', 'Native protocol adapter'],
            ['Siemens Healthineers', 'CT/MRI/PET, lab diagnostics', 'Native protocol adapter'],
            ['Becton Dickinson', 'Alaris infusion systems', 'BD HealthSight integration'],
            ['Baxter/ICU Medical', 'Infusion pumps, critical care', 'Protocol-level overlay'],
            ['Masimo', 'Pulse oximetry, hospital automation', 'Rainbow SET integration'],
        ],
        [1.8, 2.2, 3.0],
        S
    ))
    story.append(sp(8))

    story.append(sub('5.3', 'PACS and Imaging Systems', S))
    story.append(make_table(
        ['PACS Platform', 'Integration Method', 'Deployment Notes'],
        [
            ['Sectra PACS', 'DICOM TLS overlay', 'Enterprise imaging, digital pathology'],
            ['Fujifilm Synapse', 'DICOM TLS overlay', 'PACS, VNA, cardiology'],
            ['Intelerad', 'DICOM + REST API', 'Cloud-native, teleradiology'],
            ['Ambra Health', 'DICOM + cloud API', 'Cloud medical image management'],
        ],
        [1.8, 2.2, 3.0],
        S
    ))
    story.append(PageBreak())

    # SECTION 6: Deployment Timeline
    story.append(SectionHeader(
        '6. Deployment Timeline',
        '20-22 weeks to full quantum-safe protection for a 500-bed acute care facility'
    ))
    story.append(sp(8))
    story.append(make_table(
        ['Phase', 'Duration', 'Key Activities'],
        [
            ['Phase 0: Discovery', 'Week 1-2', 'Device inventory, network mapping, vulnerability baseline'],
            ['Phase 1: Infrastructure', 'Week 3-4', 'HSM deployment, VLAN configuration, edge node installation'],
            ['Phase 2: Policy Configuration', 'Week 5-6', 'HIPAA policy mapping, PHI classification, BAA review'],
            ['Phase 3: Device Onboarding', 'Week 7-10', 'Non-invasive wrapper deployment by device class'],
            ['Phase 4: Protocol Security', 'Week 9-12', 'HL7/FHIR/DICOM security activation'],
            ['Phase 5: EHR Integration', 'Week 11-14', 'Epic/Cerner/MEDITECH API security, audit logging'],
            ['Phase 6: Monitoring Activation', 'Week 13-16', 'Anomaly detection, SIEM integration, alerting'],
            ['Phase 7: Compliance Validation', 'Week 15-18', 'HIPAA audit trail validation, HITRUST evidence'],
            ['Phase 8: Clinical UAT', 'Week 17-20', 'Biomedical sign-off, clinical workflow validation'],
            ['Phase 9: Go-Live', 'Week 19-22', 'Production activation, 24/7 monitoring handover'],
        ],
        [2.0, 1.2, 3.8],
        S
    ))
    story.append(sp(10))

    # SECTION 7: Performance Specifications
    story.append(SectionHeader(
        '7. Performance Specifications',
        'Enterprise clinical scale -- zero compromise on latency or throughput'
    ))
    story.append(sp(8))
    story.append(make_table(
        ['Metric', 'Performance', 'Clinical Significance'],
        [
            ['Cryptographic overhead', 'Less than 1ms per transaction', 'Invisible to clinicians'],
            ['DICOM C-STORE encryption', 'Less than 2ms per operation', 'Zero impact on PACS workflow'],
            ['HL7 message latency', 'Less than 5ms end-to-end', 'ADT/lab results unaffected'],
            ['Device onboarding throughput', '500 devices/hour', 'Rapid enterprise rollout'],
            ['HSM key operations', '100,000/second', 'Hardware accelerated'],
            ['Audit report generation', 'Less than 10 minutes', 'On-demand HIPAA compliance'],
            ['Anomaly detection latency', 'Less than 30 seconds', 'Rapid quarantine response'],
            ['System availability', '99.999%', 'Clinical uptime requirement met'],
            ['Concurrent devices', '100,000+', 'Full enterprise IDN scale'],
            ['PHI throughput', '10 Gbps per node', 'Wire-speed for imaging networks'],
        ],
        [2.4, 1.8, 2.8],
        S
    ))
    story.append(PageBreak())

    # SECTION 8: Competitive Differentiation
    story.append(SectionHeader(
        '8. Competitive Differentiation',
        'The only platform purpose-built for clinical device security at scale'
    ))
    story.append(sp(8))
    story.append(make_table(
        ['Capability', 'QBITEL Bridge', 'Legacy Vendors', 'Device Security Startups'],
        [
            ['Post-quantum cryptography', 'NIST FIPS 203/204/205', 'None', 'None'],
            ['Non-invasive protection', 'Yes -- zero firmware', 'Requires agent', 'Passive monitoring only'],
            ['FDA recertification', 'Never triggered', 'Always triggered', 'Never (no protection)'],
            ['HL7/FHIR support', 'Native, full-fidelity', 'Basic TLS wrapping', 'None'],
            ['DICOM protection', 'Native', 'Basic', 'None'],
            ['HIPAA audit automation', 'Less than 10 minutes', 'Manual', 'Basic logging'],
            ['Constrained devices', 'Less than 64KB RAM', 'No', 'No'],
            ['Battery-aware scheduling', 'Yes', 'No', 'No'],
            ['Clinical anomaly detection', 'Device baseline ML', 'Generic network', 'Passive only'],
            ['EHR integration', 'Epic, Cerner, MEDITECH', 'Generic', 'None'],
        ],
        [2.2, 1.5, 1.5, 1.8],
        S
    ))
    story.append(PageBreak())

    # SECTION 9: Customer Scenarios
    story.append(SectionHeader(
        '9. Customer Scenarios',
        'Real-world deployments across the healthcare ecosystem'
    ))
    story.append(sp(8))

    scenarios = [
        ('SCENARIO A', 'Large Integrated Delivery Network (IDN)', [
            'Profile: 12-hospital IDN, 8,000 beds, 45,000 connected medical devices, Epic EHR.',
            '',
            'Challenge: 100% of device-to-EHR communications vulnerable to quantum harvesting.',
            'FDA-cleared device inventory made endpoint agent deployment impossible.',
            '',
            'QBITEL Solution: Clinical Edge Nodes across 47 network segments. Non-invasive',
            'wrapper on all 45,000 devices. HL7/FHIR security on Epic Interconnect.',
            'First HIPAA audit report generated in 8 minutes.',
            '',
            'Outcomes: 100% PHI quantum-safe in 16 weeks. Zero FDA recertification.',
            'HIPAA reporting: 160 hours/quarter -> 4 hours/quarter. Insurance -34%.',
        ]),
        ('SCENARIO B', 'Top-10 Medical Device Manufacturer', [
            'Profile: 200+ device SKUs, 2.3M installed devices globally, FDA-cleared across 8 classes.',
            '',
            'Challenge: FDA Oct 2023 guidance requires post-market cybersecurity management.',
            'Traditional firmware updates: $180M cost, 4+ years across installed base.',
            '',
            'QBITEL Solution: Bridge as customer-installable network security layer.',
            'Automated SBOM for all SKUs. Per-device quantum-safe certificate.',
            'White-labeled as "Quantum-Safe Connect" add-on service.',
            '',
            'Outcomes: $180M firmware program cancelled. FDA documentation automated.',
            'New service generating $28M ARR within 12 months. Retention +22%.',
        ]),
        ('SCENARIO C', 'National Health Insurer -- 18M Members', [
            'Profile: 18M members, 2.4B PHI records, 340 provider integrations, $0.9B EDI annually.',
            '',
            'Challenge: Active nation-state harvesting of X12 835/837 EDI transactions.',
            'Existing TLS 1.3 protection harvested in transit. Immediate action required.',
            '',
            'QBITEL Solution: X12 EDI gateway secured on all 340 provider integrations.',
            'FHIR R4 payer-provider APIs secured. 2.4B historical records re-encrypted.',
            '',
            'Outcomes: 100% EDI quantum-safe in 8 weeks. Avoided $180M HIPAA fine exposure.',
            'Named AHIP quantum-safe payer pioneer. Prior auth API latency -12%.',
        ]),
    ]

    for label, title, lines in scenarios:
        story.append(ScenarioBox(label, title, lines))
        story.append(sp(10))

    story.append(PageBreak())

    # SECTION 10: Next Steps
    story.append(SectionHeader(
        '10. Next Steps',
        'Three paths to quantum-safe clinical security'
    ))
    story.append(sp(8))

    steps = [
        ('Step 1', 'Quantum Vulnerability Assessment', [
            'Duration: 2 weeks -- no cost',
            'Passive network scan identifies all connected medical devices',
            'Maps PHI transmission paths and quantifies quantum exposure',
            'Deliverable: Board-ready risk report with financial exposure quantification',
            'Includes device inventory with vulnerability scoring per device class',
        ]),
        ('Step 2', 'Pilot Deployment', [
            'Duration: 4-6 weeks',
            'Bridge deployed on one clinical network segment (ICU, ED, or Radiology recommended)',
            'Demonstrates non-invasive wrapper with zero clinical disruption',
            'Validates HIPAA automation and audit report generation',
            'Biomedical engineering sign-off before enterprise commitment',
        ]),
        ('Step 3', 'Enterprise Deployment Program', [
            'Full IDN or facility deployment',
            'Dedicated Clinical Security Engineering team on-site',
            'Guaranteed deployment timeline with milestone SLAs',
            'HIPAA and HITRUST CSF certification support included',
            '24/7 Clinical Security Operations Center (CSOC) monitoring',
        ]),
    ]

    for label, title, lines in steps:
        story.append(ScenarioBox(label, title, lines))
        story.append(sp(8))

    story.append(sp(10))
    story.append(CalloutBox([
        'CONTACT: enterprise@qbitel.com  |  https://bridge.qbitel.com',
        'QBITEL Bridge -- Protecting the Healers Who Protect Us.',
        'Post-Quantum Security for Healthcare. Today.',
    ], bg=NAVY))

    doc.build(story)
    print(f'PDF written: {out_path}')


if __name__ == '__main__':
    build_pdf()
