"""Build QBITEL Bridge Defense & Sovereign Networks Marketing Pitch - PDF"""
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
    def __init__(self, height, color, width=None):
        super().__init__()
        self.bar_height = height
        self.color = color
        self.bar_width = width if width else CONTENT_W

    def wrap(self, avw, avh):
        return self.bar_width, self.bar_height

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.bar_width, self.bar_height, fill=1, stroke=0)


class SectionHeader(Flowable):
    def __init__(self, title, subtitle=None, width=None):
        super().__init__()
        self.title = title
        self.subtitle = subtitle
        self.w = width if width else CONTENT_W
        self.h = 52 if subtitle else 40

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Navy background
        c.setFillColor(NAVY)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        # Gold left bar (6px)
        c.setFillColor(GOLD)
        c.rect(0, 0, 6, self.h, fill=1, stroke=0)
        # Teal right bar (4px)
        c.setFillColor(TEAL)
        c.rect(self.w - 4, 0, 4, self.h, fill=1, stroke=0)
        # Title text
        c.setFillColor(WHITE_C)
        c.setFont("Helvetica-Bold", 13)
        if self.subtitle:
            c.drawString(14, self.h - 22, self.title)
            c.setFillColor(HexColor('#A0B4C8'))
            c.setFont("Helvetica", 9)
            c.drawString(14, 8, self.subtitle)
        else:
            c.drawString(14, (self.h - 13) / 2, self.title)


class StatBlock(Flowable):
    def __init__(self, stats):
        super().__init__()
        self.stats = stats  # list of (value, label) tuples

    def wrap(self, avw, avh):
        return CONTENT_W, 60

    def draw(self):
        c = self.canv
        n = len(self.stats)
        col_w = CONTENT_W / n
        c.setFillColor(NAVY)
        c.rect(0, 0, CONTENT_W, 60, fill=1, stroke=0)
        for i, (val, label) in enumerate(self.stats):
            x = i * col_w
            # Teal divider (except first)
            if i > 0:
                c.setFillColor(TEAL)
                c.rect(x, 8, 1, 44, fill=1, stroke=0)
            # Gold value
            c.setFillColor(GOLD)
            c.setFont("Helvetica-Bold", 11)
            val_w = c.stringWidth(val, "Helvetica-Bold", 11)
            c.drawString(x + (col_w - val_w) / 2, 34, val)
            # White label
            c.setFillColor(WHITE_C)
            c.setFont("Helvetica", 7)
            lbl_w = c.stringWidth(label, "Helvetica", 7)
            c.drawString(x + (col_w - lbl_w) / 2, 16, label)


class ScenarioBox(Flowable):
    def __init__(self, label, title, lines, width=None):
        super().__init__()
        self.label = label
        self.title = title
        self.lines = lines
        self.w = width if width else CONTENT_W
        # Calculate height: teal header (36) + lines
        line_h = 14
        self.h = 36 + len(lines) * line_h + 16

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Teal header
        c.setFillColor(TEAL)
        c.rect(0, self.h - 36, self.w, 36, fill=1, stroke=0)
        # Light border
        c.setStrokeColor(TEAL)
        c.setLineWidth(1)
        c.rect(0, 0, self.w, self.h - 36, fill=0, stroke=1)
        # Light bg body
        c.setFillColor(LIGHT_BG)
        c.rect(1, 1, self.w - 2, self.h - 37, fill=1, stroke=0)
        # Navy label badge
        c.setFillColor(NAVY)
        c.rect(10, self.h - 30, 60, 20, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont("Helvetica-Bold", 7)
        lw = c.stringWidth(self.label, "Helvetica-Bold", 7)
        c.drawString(10 + (60 - lw) / 2, self.h - 23, self.label)
        # White title
        c.setFillColor(WHITE_C)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(80, self.h - 24, self.title)
        # Body lines
        c.setFillColor(DARK_TEXT)
        c.setFont("Helvetica", 8)
        y = self.h - 36 - 16
        for line in self.lines:
            if y < 6:
                break
            c.drawString(12, y, line)
            y -= 14


class CalloutBox(Flowable):
    def __init__(self, text, icon='►', width=None):
        super().__init__()
        self.text = text
        self.icon = icon
        self.w = width if width else CONTENT_W
        # Calculate height based on text wrapping
        chars_per_line = int((self.w - 30) / 6.5)
        words = text.split()
        lines = []
        current = ""
        for w in words:
            if len(current) + len(w) + 1 <= chars_per_line:
                current = current + " " + w if current else w
            else:
                if current:
                    lines.append(current)
                current = w
        if current:
            lines.append(current)
        self.text_lines = lines
        self.h = max(36, len(lines) * 14 + 16)

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Light background
        c.setFillColor(LIGHT_BG)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        # Gold left bar
        c.setFillColor(GOLD)
        c.rect(0, 0, 5, self.h, fill=1, stroke=0)
        # Icon
        c.setFillColor(GOLD)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(12, self.h / 2 - 5, self.icon)
        # Text lines
        c.setFillColor(DARK_TEXT)
        c.setFont("Helvetica", 9)
        y_start = self.h - 14
        if len(self.text_lines) > 1:
            y_start = self.h / 2 + (len(self.text_lines) * 14) / 2 - 10
        for i, line in enumerate(self.text_lines):
            c.drawString(26, y_start - i * 14, line)


def get_styles():
    styles = {}
    styles['body'] = ParagraphStyle(
        'body', fontName='Helvetica', fontSize=9, leading=14,
        textColor=DARK_TEXT, spaceAfter=6, alignment=TA_JUSTIFY
    )
    styles['body_small'] = ParagraphStyle(
        'body_small', fontName='Helvetica', fontSize=8, leading=12,
        textColor=DARK_TEXT, spaceAfter=4, alignment=TA_LEFT
    )
    styles['h1'] = ParagraphStyle(
        'h1', fontName='Helvetica-Bold', fontSize=16, leading=20,
        textColor=NAVY, spaceAfter=8, spaceBefore=12
    )
    styles['h2'] = ParagraphStyle(
        'h2', fontName='Helvetica-Bold', fontSize=12, leading=16,
        textColor=TEAL, spaceAfter=6, spaceBefore=8
    )
    styles['h3'] = ParagraphStyle(
        'h3', fontName='Helvetica-Bold', fontSize=10, leading=14,
        textColor=NAVY, spaceAfter=4, spaceBefore=6
    )
    styles['table_header'] = ParagraphStyle(
        'table_header', fontName='Helvetica-Bold', fontSize=8, leading=11,
        textColor=WHITE_C, alignment=TA_CENTER
    )
    styles['table_cell'] = ParagraphStyle(
        'table_cell', fontName='Helvetica', fontSize=8, leading=11,
        textColor=DARK_TEXT, alignment=TA_LEFT
    )
    styles['table_cell_center'] = ParagraphStyle(
        'table_cell_center', fontName='Helvetica', fontSize=8, leading=11,
        textColor=DARK_TEXT, alignment=TA_CENTER
    )
    styles['bullet'] = ParagraphStyle(
        'bullet', fontName='Helvetica', fontSize=9, leading=13,
        textColor=DARK_TEXT, spaceAfter=3, leftIndent=14,
        bulletIndent=4, bulletText='•'
    )
    styles['callout'] = ParagraphStyle(
        'callout', fontName='Helvetica-Oblique', fontSize=9, leading=13,
        textColor=NAVY, spaceAfter=4, leftIndent=10
    )
    styles['caption'] = ParagraphStyle(
        'caption', fontName='Helvetica', fontSize=7, leading=10,
        textColor=MID_GREY, spaceAfter=2, alignment=TA_CENTER
    )
    return styles


def draw_page(canvas, doc):
    canvas.saveState()
    # Header strip
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 30, PAGE_W, 30, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(MARGIN, PAGE_H - 19, "QBITEL BRIDGE — DEFENSE & SOVEREIGN NETWORKS")
    canvas.setFont("Helvetica", 8)
    page_text = f"Page {doc.page}"
    pw = canvas.stringWidth(page_text, "Helvetica", 8)
    canvas.drawString(PAGE_W - MARGIN - pw, PAGE_H - 19, page_text)
    # Teal footer strip
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, PAGE_W, 22, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(MARGIN, 7, "Confidential — For Authorized Recipients Only  |  © 2026 QBITEL")
    contact = "enterprise@qbitel.com  |  bridge.qbitel.com"
    cw = canvas.stringWidth(contact, "Helvetica", 7)
    canvas.drawString(PAGE_W - MARGIN - cw, 7, contact)
    canvas.restoreState()


def draw_cover(canvas, doc):
    canvas.saveState()
    # Full-bleed navy background
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    # Teal decorative strip top ~40%
    strip_h = PAGE_H * 0.40
    canvas.setFillColor(TEAL_DARK)
    canvas.rect(0, PAGE_H - strip_h, PAGE_W, strip_h, fill=1, stroke=0)

    # Gold accent bar
    canvas.setFillColor(GOLD)
    canvas.rect(0, PAGE_H - strip_h - 4, PAGE_W, 4, fill=1, stroke=0)

    # QBITEL BRIDGE product name (top area)
    canvas.setFillColor(WHITE_C)
    canvas.setFont("Helvetica-Bold", 28)
    canvas.drawString(MARGIN, PAGE_H - 90, "QBITEL BRIDGE")

    canvas.setFillColor(GOLD)
    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(MARGIN, PAGE_H - 118, "Defense & Sovereign Networks")

    # Decorative line under subtitle
    canvas.setStrokeColor(GOLD)
    canvas.setLineWidth(2)
    canvas.line(MARGIN, PAGE_H - 128, MARGIN + 3 * inch, PAGE_H - 128)

    # Main title
    canvas.setFillColor(WHITE_C)
    canvas.setFont("Helvetica-Bold", 22)
    title_y = PAGE_H - strip_h + 80
    canvas.drawString(MARGIN, title_y, "Quantum-Safe Security for")
    canvas.drawString(MARGIN, title_y - 30, "National Defense Infrastructure")

    # Subtitle
    canvas.setFillColor(HexColor('#A0C0D0'))
    canvas.setFont("Helvetica", 10)
    sub_y = title_y - 62
    canvas.drawString(MARGIN, sub_y,
        "Air-Gapped Sovereign AI  |  CNSA 2.0 / NIST Level 5 PQC  |  CMMC 2.0 Level 3 Automation")
    canvas.drawString(MARGIN, sub_y - 14,
        "Zero Cloud Dependency  |  Legacy System Modernization  |  TPM 2.0 Hardware Security")

    # 4 stat boxes at bottom
    box_y = 1.5 * inch
    box_w = (PAGE_W - 2 * MARGIN - 3 * 8) / 4
    box_h = 70
    stats = [
        ("NIST Level 5", "PQC Algorithm Grade"),
        ("Air-Gapped", "Deployment Model"),
        ("CMMC 2.0 L3", "Compliance Ready"),
        ("Zero Cloud", "Full Sovereignty"),
    ]
    for i, (val, label) in enumerate(stats):
        bx = MARGIN + i * (box_w + 8)
        # Navy rounded rect
        canvas.setFillColor(LIGHT_NAVY)
        canvas.roundRect(bx, box_y, box_w, box_h, 4, fill=1, stroke=0)
        # Gold accent top bar
        canvas.setFillColor(GOLD)
        canvas.rect(bx, box_y + box_h - 4, box_w, 4, fill=1, stroke=0)
        # Value text
        canvas.setFillColor(GOLD)
        canvas.setFont("Helvetica-Bold", 11)
        vw = canvas.stringWidth(val, "Helvetica-Bold", 11)
        canvas.drawString(bx + (box_w - vw) / 2, box_y + 32, val)
        # Label text
        canvas.setFillColor(WHITE_C)
        canvas.setFont("Helvetica", 8)
        lw = canvas.stringWidth(label, "Helvetica", 8)
        canvas.drawString(bx + (box_w - lw) / 2, box_y + 14, label)

    # Bottom teal strip
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, PAGE_W, 40, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(MARGIN, 14, "enterprise@qbitel.com  |  https://bridge.qbitel.com")
    canvas.setFont("Helvetica-Bold", 9)
    ver_text = "© 2026 QBITEL  |  Confidential"
    vw = canvas.stringWidth(ver_text, "Helvetica-Bold", 9)
    canvas.drawString(PAGE_W - MARGIN - vw, 14, ver_text)

    canvas.restoreState()


def build_table(data, col_widths, header_bg=NAVY, alt_bg=TABLE_ALT, S=None):
    if S is None:
        S = get_styles()
    formatted = []
    for r, row in enumerate(data):
        fmt_row = []
        for cell in row:
            if r == 0:
                fmt_row.append(Paragraph(str(cell), S['table_header']))
            else:
                fmt_row.append(Paragraph(str(cell), S['table_cell']))
        formatted.append(fmt_row)

    tbl = Table(formatted, colWidths=col_widths, repeatRows=1)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), header_bg),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE_C),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, alt_bg]),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCCCCC')),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ])
    tbl.setStyle(style)
    return tbl


def build_doc(output_path):
    doc = BaseDocTemplate(
        output_path, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN
    )
    cover_frame = Frame(
        0, 0, PAGE_W, PAGE_H,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        id='cover'
    )
    inner_frame = Frame(
        MARGIN, 0.7 * inch, CONTENT_W,
        PAGE_H - MARGIN - 0.7 * inch,
        id='inner'
    )
    cover_template = PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover)
    inner_template = PageTemplate(id='Inner', frames=[inner_frame], onPage=draw_page)
    doc.addPageTemplates([cover_template, inner_template])

    S = get_styles()
    story = []

    # Cover page placeholder
    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────
    # SECTION 1 — EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────────────────────
    story.append(SectionHeader("EXECUTIVE SUMMARY",
        "Sovereign Security for the Defense Enterprise"))
    story.append(sp(10))

    story.append(Paragraph(
        "The United States defense industrial base faces an existential cryptographic threat: nation-state "
        "adversaries are systematically harvesting classified communications today, storing petabytes of "
        "encrypted traffic with the explicit intent to decrypt it once quantum computing reaches sufficient "
        "scale. The NSA recognized this threat in 2022 when it published CNSA 2.0, mandating a complete "
        "transition to post-quantum cryptography by 2030 for all National Security Systems.",
        S['body']))

    story.append(Paragraph(
        "QBITEL Bridge is the only enterprise security platform engineered from the ground up for sovereign, "
        "air-gapped defense environments. Three non-negotiable design principles: zero cloud dependency "
        "(all AI inference runs on-premise via Ollama with no external API calls), NIST Level 5 post-quantum "
        "cryptography natively implemented (ML-KEM-1024 for key encapsulation, ML-DSA-87 for digital "
        "signatures), and comprehensive CMMC 2.0 Level 3 automation covering all 110 NIST SP 800-171 "
        "practices.",
        S['body']))

    story.append(Paragraph(
        "The threat landscape is unambiguous: 91% of defense contractors are targeted by nation-state actors "
        "annually, classified intellectual property valued at over $600 billion is at risk, and the average "
        "APT dwell time exceeds 200 days. QBITEL Bridge addresses each dimension simultaneously — detecting "
        "APTs, protecting CUI with quantum-safe encryption, automating CMMC compliance, and wrapping legacy "
        "COBOL/CICS systems — all without a single outbound internet connection.",
        S['body']))

    story.append(sp(8))
    story.append(StatBlock([
        ("91%", "Defense Contractors\nNation-State Targeted"),
        ("$600B+", "Classified IP\nAt Risk"),
        ("200+ Days", "Avg APT Dwell\nTime in Defense Nets"),
        ("2030", "CNSA 2.0 Transition\nDeadline (NSA)"),
        ("110/110", "NIST 800-171\nPractices Automated"),
    ]))
    story.append(sp(12))

    # ─────────────────────────────────────────────────────────────
    # SECTION 2 — THREAT LANDSCAPE
    # ─────────────────────────────────────────────────────────────
    story.append(SectionHeader("THE DEFENSE CYBER THREAT LANDSCAPE",
        "Nation-State Threats Require Nation-State-Grade Defenses"))
    story.append(sp(10))

    story.append(Paragraph("<b>Threat 1: The Quantum Harvest-Now-Decrypt-Later Attack</b>", S['h3']))
    story.append(Paragraph(
        "Nation-state intelligence services are conducting systematic interception and archival of encrypted "
        "classified communications. The objective: when cryptographically-relevant quantum computers (CRQCs) "
        "become available, all currently-encrypted traffic becomes readable. RSA-2048 and ECC-256 — the "
        "cryptographic foundations of virtually all current defense network security — can be broken by a "
        "CRQC running Shor's algorithm in hours.",
        S['body']))
    story.append(CalloutBox(
        "CRITICAL: Every classified email, encrypted file transfer, and VPN session secured with classical "
        "cryptography that has been intercepted is vulnerable to future quantum decryption. "
        "Act before the quantum transition — not after.",
        icon='⚠'))
    story.append(sp(8))

    story.append(Paragraph("<b>Threat 2: Supply Chain Compromise</b>", S['h3']))
    story.append(Paragraph(
        "SolarWinds demonstrated that nation-state actors can compromise the software supply chain upstream "
        "and achieve simultaneous access to thousands of targets. For the DIB, this risk is compounded by "
        "the complexity of the defense contracting ecosystem: a single prime may have hundreds of Tier 1/2 "
        "suppliers with varying cybersecurity postures and direct CUI access.",
        S['body']))
    story.append(sp(6))

    story.append(Paragraph("<b>Threat 3: Legacy System Exposure</b>", S['h3']))
    story.append(Paragraph(
        "The U.S. military operates some of the oldest computing infrastructure on earth. Legacy COBOL "
        "command-and-control systems, TN3270e mainframe terminals, and aging SS7-based military "
        "communications infrastructure cannot be patched to support post-quantum cryptography. Yet they "
        "handle some of the most sensitive data in the defense enterprise — all without modern "
        "cryptographic protections.",
        S['body']))

    story.append(sp(8))

    # Threat stats table
    threat_data = [
        ["Threat Vector", "Scale", "Current Exposure", "QBITEL Bridge Response"],
        ["Harvest-Now-Decrypt-Later", "State-level resources", "All classical crypto",
         "ML-KEM-1024 forward secrecy"],
        ["APT Lateral Movement", "200+ day avg dwell", "91% of DIB targeted",
         "78% autonomous response"],
        ["Supply Chain Compromise", "220,000+ DIB companies", "Tier 2/3 weak links",
         "Zero-trust contractor access"],
        ["Legacy System Exploitation", "30+ year system lifecycles", "No PQC capability",
         "Transparent PQC proxy wrap"],
        ["Insider Threat", "Privileged access abuse", "CUI exfiltration risk",
         "Behavioral AI + immutable audit"],
    ]
    story.append(build_table(threat_data,
        [1.4*inch, 1.3*inch, 1.4*inch, 2.3*inch], S=S))
    story.append(sp(12))

    # ─────────────────────────────────────────────────────────────
    # SECTION 3 — 7 CAPABILITIES
    # ─────────────────────────────────────────────────────────────
    story.append(SectionHeader("CAPABILITY 1: AIR-GAPPED SOVEREIGN DEPLOYMENT",
        "Zero Cloud Dependency — Complete Operational Sovereignty"))
    story.append(sp(10))
    story.append(Paragraph(
        "Defense environments have a fundamental requirement that disqualifies the vast majority of "
        "commercial security products: complete operational independence from external networks and cloud "
        "infrastructure. QBITEL Bridge was designed air-gapped-first. The platform's AI inference engine "
        "runs on-premise using Ollama with no external API calls — zero data egress, zero cloud dependency.",
        S['body']))

    cap1_data = [
        ["Component", "Implementation", "Validation"],
        ["AI Inference Engine", "Ollama on-premise (no external calls)", "Network capture: 0 outbound"],
        ["Threat Intelligence", "Offline update packages (USB/media)", "Cryptographic pkg verification"],
        ["Compliance Engine", "All 110 practices checked locally", "No cloud API dependencies"],
        ["Key Management", "On-premise HSM (FIPS 140-3 L3)", "Air-gap verified key gen"],
        ["Management Plane", "Out-of-band network + MFA", "Zero internet connectivity"],
        ["Software Updates", "Signed air-gap-safe packages", "TPM integrity verification"],
    ]
    story.append(build_table(cap1_data, [1.8*inch, 2.5*inch, 2.1*inch], S=S))
    story.append(sp(12))

    story.append(SectionHeader("CAPABILITY 2: CNSA 2.0 / NIST LEVEL 5 PQC",
        "NSA-Recommended Post-Quantum Cryptography — Native Implementation"))
    story.append(sp(10))
    story.append(Paragraph(
        "QBITEL Bridge implements the complete NSA CNSA 2.0 algorithm suite natively, including hybrid "
        "classical/PQC key exchange for the transition period. ML-KEM-1024 provides NIST Level 5 key "
        "encapsulation (the highest security level defined by NIST), and ML-DSA-87 provides quantum-safe "
        "digital signatures. All algorithms are implemented using NIST-validated cryptographic modules.",
        S['body']))

    pqc_data = [
        ["Algorithm", "Type", "NIST Level", "CNSA 2.0", "Use Case"],
        ["ML-KEM-1024", "Key Encapsulation", "Level 5", "Approved", "Key exchange, session setup"],
        ["ML-DSA-87", "Digital Signature", "Level 5", "Approved", "Auth, code signing"],
        ["Falcon-1024", "Digital Signature", "Level 5", "Approved", "Bandwidth-constrained"],
        ["SPHINCS+-256s", "Signature (stateless)", "Level 5", "Approved", "Long-term archives"],
        ["AES-256-GCM", "Symmetric (hybrid)", "Classical", "Retained", "Data encryption"],
        ["SHA-3-512", "Hash (hybrid)", "Classical", "Retained", "Integrity verification"],
    ]
    story.append(build_table(pqc_data,
        [1.3*inch, 1.3*inch, 0.9*inch, 0.8*inch, 2.1*inch], S=S))
    story.append(sp(12))

    story.append(SectionHeader("CAPABILITY 3: CLASSIFIED CUI PROTECTION",
        "NIST SP 800-171/172 — Complete Controlled Unclassified Information Safeguarding"))
    story.append(sp(10))
    story.append(Paragraph(
        "QBITEL Bridge provides end-to-end CUI protection: automated discovery across all 125 CUI "
        "categories, quantum-safe encryption at rest and in transit, need-to-know access control with "
        "continuous PQC-authenticated verification, and blockchain-backed immutable audit trails that "
        "satisfy DoD assessor requirements.",
        S['body']))

    cui_data = [
        ["CUI Protection Layer", "Mechanism", "NIST 800-171 Practices"],
        ["Discovery & Classification", "ML-based content scanning (125 categories)", "AC.1.001, MP.2.120"],
        ["Encryption at Rest", "AES-256-GCM with ML-KEM-1024 key wrap", "SC.3.177, SC.3.187"],
        ["Encryption in Transit", "TLS 1.3 + ML-KEM hybrid key exchange", "SC.3.185, SC.3.190"],
        ["Access Control", "ABAC + PQC-signed credentials + TPM attestation", "AC.2.005, AC.2.006"],
        ["Immutable Audit Trail", "Blockchain-linked, tamper-evident logging", "AU.2.041, AU.2.042"],
        ["Data Labeling", "Automated CUI marking per ISOO guidance", "MP.2.120, AC.1.002"],
    ]
    story.append(build_table(cui_data, [1.7*inch, 2.3*inch, 2.4*inch], S=S))
    story.append(sp(12))

    story.append(PageBreak())

    story.append(SectionHeader("CAPABILITY 4: DEFENSE CONTRACTOR NETWORK SECURITY",
        "CMMC 2.0 Zero-Trust Architecture for the Defense Industrial Base"))
    story.append(sp(10))
    story.append(Paragraph(
        "QBITEL Bridge implements a zero-trust architecture for defense contractor networks that eliminates "
        "implicit trust, enforces cryptographic identity verification for every access request, and provides "
        "continuous behavioral monitoring that detects APT lateral movement — even when adversaries have "
        "compromised legitimate contractor credentials.",
        S['body']))

    zt_data = [
        ["Zero Trust Pillar", "QBITEL Bridge Implementation", "DoD ZTA Alignment"],
        ["Identity", "PQC-signed CAC/PKI + behavioral scoring", "NIST ZTA Pillar 1"],
        ["Device", "TPM 2.0 attestation + health posture", "NIST ZTA Pillar 2"],
        ["Network", "Micro-segmentation + encrypted east-west", "NIST ZTA Pillar 3"],
        ["Application", "Zero-trust app access + RBAC enforcement", "NIST ZTA Pillar 4"],
        ["Data", "CUI tagging + quantum-safe encryption", "NIST ZTA Pillar 5"],
        ["Visibility", "78% autonomous response + SIEM integration", "NIST ZTA Pillar 6"],
        ["Automation", "SOAR playbooks + auto-containment", "NIST ZTA Pillar 7"],
    ]
    story.append(build_table(zt_data, [1.6*inch, 2.5*inch, 2.3*inch], S=S))
    story.append(sp(12))

    story.append(SectionHeader("CAPABILITY 5: LEGACY MILITARY SYSTEM MODERNIZATION",
        "Quantum-Safe Protection for 30-Year-Old Infrastructure — Zero Downtime"))
    story.append(sp(10))
    story.append(Paragraph(
        "QBITEL Bridge deploys transparent protocol proxies that intercept and re-encrypt legacy protocol "
        "traffic with post-quantum cryptographic protection — without modifying the legacy systems or "
        "incurring any downtime. Legacy COBOL/CICS applications, TN3270e mainframe sessions, and aging "
        "military protocols continue operating exactly as before.",
        S['body']))

    legacy_data = [
        ["Legacy Protocol", "Wrapping Method", "Modification Required", "Performance Impact"],
        ["TN3270e / TN3270", "Transparent PQC proxy (ARP intercept)", "None", "<2ms added latency"],
        ["COBOL/CICS ISC", "PQC transport wrapper (MQ/TCP/IP)", "None", "<1ms added latency"],
        ["SNA/APPN", "Protocol-aware security overlay", "None", "<2ms added latency"],
        ["Military SS7", "Signaling monitoring + anomaly detect", "None", "Passive monitoring"],
        ["Tactical Mesh", "PQC-secured mesh relay proxy", "None", "<3ms added latency"],
        ["RS-232/Serial (legacy)", "Serial-to-PQC bridge appliance", "None", "<5ms added latency"],
    ]
    story.append(build_table(legacy_data,
        [1.4*inch, 1.9*inch, 1.5*inch, 1.6*inch], S=S))
    story.append(sp(12))

    story.append(SectionHeader("CAPABILITY 6: CMMC 2.0 COMPLIANCE AUTOMATION",
        "Complete 110-Practice Automation — C3PAO-Ready Evidence in <15 Minutes"))
    story.append(sp(10))
    story.append(Paragraph(
        "QBITEL Bridge automates the entire CMMC 2.0 evidence collection and documentation process, "
        "reducing the time required to generate a complete assessment package from months to under 15 "
        "minutes. Continuous monitoring ensures compliance status is maintained between assessments.",
        S['body']))

    cmmc_data = [
        ["CMMC Capability", "Automation Level", "Output"],
        ["110-Practice Assessment", "Fully automated, continuous", "Real-time compliance score"],
        ["Evidence Collection", "Automated config snapshots + logs", "Formatted evidence package"],
        ["POA&M Management", "Auto-generated with risk priority", "POA&M document (Word/PDF)"],
        ["System Security Plan", "Auto-populated from discovery", "SSP artifacts for C3PAO"],
        ["C3PAO Evidence Package", "One-click generation", "Complete package in <15 min"],
        ["800-172 Enhanced (Level 3)", "24 additional practices covered", "APT countermeasure evidence"],
        ["Continuous Monitoring", "Daily scoring + drift alerts", "Automated compliance reports"],
    ]
    story.append(build_table(cmmc_data, [2.0*inch, 2.0*inch, 2.4*inch], S=S))
    story.append(sp(12))

    story.append(SectionHeader("CAPABILITY 7: TPM-BOUND HARDWARE SECURITY",
        "Hardware Root of Trust — Cryptographic Proof of Platform Integrity"))
    story.append(sp(10))
    story.append(Paragraph(
        "QBITEL Bridge's hardware security integration establishes a TPM 2.0-based hardware root of trust "
        "that cannot be compromised by software attacks. Keys are sealed to specific TPM PCR values and "
        "can only be unsealed when the platform is in a known-good state — preventing extraction via "
        "rootkits, firmware implants, or offline attacks.",
        S['body']))

    tpm_data = [
        ["TPM Feature", "Implementation", "Security Benefit"],
        ["Measured Boot", "PCR-recorded boot chain measurement", "Detects firmware/OS tampering"],
        ["Key Sealing", "Keys bound to PCR values", "Prevents offline key extraction"],
        ["Remote Attestation", "TPM attestation quotes", "Proves platform integrity remotely"],
        ["FIPS 140-3 HSM", "Thales Luna / equivalent", "Root CA key hardware protection"],
        ["Secure Enclave", "Intel TDX / AMD SEV-SNP", "Isolated key operation execution"],
        ["Anti-Tamper", "Physical intrusion detection", "Auto key destruction on tamper"],
    ]
    story.append(build_table(tpm_data, [1.7*inch, 2.3*inch, 2.4*inch], S=S))
    story.append(sp(12))

    # ─────────────────────────────────────────────────────────────
    # COMPLIANCE COVERAGE
    # ─────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(SectionHeader("COMPLIANCE COVERAGE",
        "Comprehensive Defense & Intelligence Community Framework Coverage"))
    story.append(sp(10))

    comp_data = [
        ["Framework", "Coverage", "Details"],
        ["NIST SP 800-171", "110/110 practices", "Full CUI protection automation"],
        ["NIST SP 800-172", "Enhanced controls", "APT-resistant measures, CMMC L3"],
        ["CMMC 2.0 Level 3", "Complete", "Automated evidence, C3PAO-ready"],
        ["DISA STIGs", "Aligned", "Automated hardening and verification"],
        ["FedRAMP High", "Compatible", "Control mapping and evidence support"],
        ["DoD IL4 / IL5", "Fully supported", "IL-specific security controls"],
        ["DoD IL6", "Architecture aligned", "On-premise sovereign deployment"],
        ["NSA CNSA 2.0", "Native implementation", "ML-KEM-1024, ML-DSA-87 deployed"],
        ["DoD Zero Trust Architecture", "All 7 pillars", "Identity, Device, Network, App, Data"],
        ["ITAR", "Compliant", "NIST-standard algos (not ITAR-controlled)"],
        ["EAR", "EAR99 classification", "PQC algorithms fully exportable"],
        ["FISMA High", "Aligned", "NIST RMF control implementation"],
    ]
    story.append(build_table(comp_data, [2.0*inch, 1.4*inch, 3.0*inch], S=S))
    story.append(sp(12))

    # ─────────────────────────────────────────────────────────────
    # INTEGRATION ECOSYSTEM
    # ─────────────────────────────────────────────────────────────
    story.append(SectionHeader("INTEGRATION ECOSYSTEM",
        "Seamless Integration with Defense Security Infrastructure"))
    story.append(sp(10))

    integ_data = [
        ["Integration Category", "Supported Products / Protocols", "Integration Type"],
        ["SIEM", "Splunk ES, IBM QRadar, ArcSight, Elastic", "Certified connectors, CEF/LEEF"],
        ["SOAR", "Palo Alto XSOAR, Splunk SOAR, ServiceNow SecOps", "Bidirectional REST API"],
        ["Identity (IAM)", "DoD PKI/CAC, Active Directory, CyberArk", "PKCS#11, SAML 2.0 + PQC"],
        ["DoD Infrastructure", "DISA ACAS, HBSS, DIBNet portal", "Automated reporting APIs"],
        ["Cloud (IL4/IL5)", "Azure Government, AWS GovCloud", "On-premise bridge connector"],
        ["Classified Networks", "SIPRNet, JWICS (air-gapped)", "On-premise only, no cloud"],
        ["Ticketing/ITSM", "ServiceNow, Jira (on-premise)", "REST API, webhook"],
    ]
    story.append(build_table(integ_data, [1.6*inch, 2.4*inch, 2.4*inch], S=S))
    story.append(sp(12))

    # ─────────────────────────────────────────────────────────────
    # DEPLOYMENT TIMELINE
    # ─────────────────────────────────────────────────────────────
    story.append(SectionHeader("DEPLOYMENT TIMELINE",
        "Structured 12-Week Deployment — ATO-Process Compatible"))
    story.append(sp(10))

    timeline_data = [
        ["Phase", "Duration", "Key Activities", "Owner"],
        ["Phase 1: Air-Gap Verification\n& Sovereign AI Setup",
         "Weeks 1-2",
         "Network isolation verification, on-premise server provisioning,\nOllama install, zero-connection validation",
         "IT/Infrastructure"],
        ["Phase 2: PQC Key Ceremony\n& TPM Binding",
         "Weeks 3-4",
         "Formal key ceremony (3+ custodians), root CA gen in HSM,\nTPM 2.0 key sealing on all nodes",
         "Crypto Officer"],
        ["Phase 3: CUI Protection\n& Legacy Wrapping",
         "Weeks 5-8",
         "CUI discovery scan, PQC encryption policy application,\nTN3270e/COBOL proxy deployment, regression testing",
         "ISSO + App Team"],
        ["Phase 4: CMMC Evidence\n& ATO Validation",
         "Weeks 9-12",
         "110-practice evidence run, POA&M generation, SSP population,\nC3PAO package, red team, ATO assembly",
         "Compliance + AO"],
    ]
    story.append(build_table(timeline_data, [1.6*inch, 0.8*inch, 3.0*inch, 1.0*inch], S=S))
    story.append(sp(12))

    # ─────────────────────────────────────────────────────────────
    # PERFORMANCE SPECIFICATIONS
    # ─────────────────────────────────────────────────────────────
    story.append(SectionHeader("PERFORMANCE SPECIFICATIONS",
        "Mission-Grade Performance — No Operational Penalty for Security"))
    story.append(sp(10))

    perf_data = [
        ["Metric", "Specification", "Conditions"],
        ["Encryption throughput", "50,000+ ops/sec", "ML-KEM-1024 on 32-core server, air-gapped"],
        ["Key operation latency", "<10ms (p99)", "TPM-bound key operations"],
        ["System availability", "99.999% (5 nines)", "Active/passive HA cluster"],
        ["Autonomous threat response", "78% without human approval", "Policy-governed automated response"],
        ["Audit trail integrity", "Blockchain-backed, immutable", "Cryptographically linked entries"],
        ["CMMC evidence generation", "<15 minutes", "Full 110-practice package"],
        ["Legacy proxy overhead", "<2ms additional latency", "TN3270e and COBOL wrappers"],
        ["CUI discovery rate", "500,000+ docs/hour", "Parallel ML classification"],
        ["False positive rate", "<0.3%", "Production-tuned defense models"],
        ["Recovery time objective", "<4 hours", "Active/passive HA warm standby"],
    ]
    story.append(build_table(perf_data, [2.2*inch, 1.8*inch, 2.4*inch], S=S))
    story.append(sp(12))

    # ─────────────────────────────────────────────────────────────
    # COMPETITIVE DIFFERENTIATION
    # ─────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(SectionHeader("COMPETITIVE DIFFERENTIATION",
        "Why QBITEL Bridge — Versus Every Alternative"))
    story.append(sp(10))

    comp_diff_data = [
        ["Dimension", "NSA Type 1 Devices", "Commercial PQC Vendors",
         "Cloud Platforms", "QBITEL Bridge"],
        ["Air-Gap Support", "Yes (point-to-point)", "No", "No", "Yes (native)"],
        ["Sovereign AI", "N/A", "No", "No", "Yes (Ollama)"],
        ["CNSA 2.0 Native", "Yes (certified)", "Partial (libraries)", "No", "Yes (full suite)"],
        ["CMMC Automation", "No", "No", "Limited", "Yes (110 practices)"],
        ["Legacy System Wrapping", "Limited (inline only)", "No", "No", "Yes (transparent proxy)"],
        ["Behavioral Anomaly Detection", "No", "No", "Yes (cloud-only)", "Yes (on-premise AI)"],
        ["TPM 2.0 Key Sealing", "No", "No", "No", "Yes"],
        ["78% Autonomous Response", "No", "No", "Partial", "Yes"],
        ["Single Platform", "No (single function)", "No (point solution)", "Partial", "Yes"],
    ]
    story.append(build_table(comp_diff_data,
        [1.4*inch, 1.0*inch, 1.1*inch, 1.0*inch, 1.1*inch], S=S))
    story.append(sp(12))

    # ─────────────────────────────────────────────────────────────
    # CUSTOMER SCENARIOS
    # ─────────────────────────────────────────────────────────────
    story.append(SectionHeader("CUSTOMER SCENARIOS",
        "Real-World Defense Deployment Use Cases"))
    story.append(sp(10))

    story.append(ScenarioBox(
        "SCENARIO A",
        "DoD Prime Contractor — CMMC 2.0 Level 3 Compliance",
        [
            "Situation: $2B prime, 3,000 employees, 200+ subcontractors, 47 open POA&M items, C3PAO",
            "assessment in 6 months. Legacy IBM mainframe processing CUI.",
            "",
            "Solution: Air-gapped QBITEL Bridge across 15 CUI sites. Automated CMMC evidence reduces",
            "47 POA&M to 8 within 30 days. TN3270e proxy wraps mainframe CUI. Zero-trust applied to",
            "200-node subcontractor network. C3PAO package generated in <15 min.",
            "",
            "Outcome: CMMC 2.0 Level 3 certification in 4 months. Contract retained. 91% reduction",
            "in manual compliance effort. Complete quantum-safe protection of all CUI.",
        ]
    ))
    story.append(sp(8))

    story.append(ScenarioBox(
        "SCENARIO B",
        "Intelligence Community Network — TS/SCI Protection",
        [
            "Situation: IC program office, classified network with TS/SCI data, 1980s legacy mainframes,",
            "modern analyst workstations. Red team found APT undetected for 60 days.",
            "",
            "Solution: Air-gapped deployment (zero cloud, classification-compliant). ML-KEM-1024 +",
            "ML-DSA-87 across all boundaries. Mainframe wrapped with PQC proxy (zero code changes).",
            "Sovereign AI (Ollama) with IC-calibrated behavioral baselines.",
            "",
            "Outcome: Simulated APT detected in 4 hours (vs. 60-day baseline). Classified data",
            "protected vs. quantum decryption. Legacy systems unchanged. Full ATO documentation.",
        ]
    ))
    story.append(sp(8))

    story.append(ScenarioBox(
        "SCENARIO C",
        "Defense Manufacturer — Supply Chain Security",
        [
            "Situation: Tier 1 manufacturer, 85 Tier 2/3 suppliers across 12 allied nations.",
            "Threat intel: 3 supplier networks compromised by nation-state actors.",
            "",
            "Solution: Zero-trust contractor portal with CMMC compliance verification. All CUI shared",
            "with suppliers encrypted with ML-KEM-1024. Behavioral anomaly detection on all supplier",
            "access. Automated DIBNet compromise reporting.",
            "",
            "Outcome: 3 compromised networks detected and isolated in 24 hours. Zero CUI exfiltrated.",
            "Complete supply chain audit trail. Prime contract relationship protected.",
        ]
    ))
    story.append(sp(12))

    # ─────────────────────────────────────────────────────────────
    # NEXT STEPS
    # ─────────────────────────────────────────────────────────────
    story.append(SectionHeader("NEXT STEPS",
        "Engage QBITEL Bridge for Your Defense Environment"))
    story.append(sp(10))

    next_data = [
        ["Engagement Option", "Description", "Duration", "Cost"],
        ["Classified Technical Briefing",
         "Cleared QBITEL team delivers architecture deep-dive (TS/SCI by arrangement)",
         "2 hours", "No charge"],
        ["CMMC Readiness Assessment",
         "Gap analysis vs. 110 NIST 800-171 practices, prioritized remediation roadmap",
         "3-5 days", "No charge"],
        ["Air-Gapped Proof of Concept",
         "QBITEL Bridge deployed in customer-controlled air-gapped environment for 30 days",
         "30 days", "By arrangement"],
        ["Procurement Support",
         "Contract vehicle identification, SOW development, ATO documentation support",
         "As needed", "Per contract"],
    ]
    story.append(build_table(next_data, [1.6*inch, 2.8*inch, 0.8*inch, 1.2*inch], S=S))
    story.append(sp(12))

    # ─────────────────────────────────────────────────────────────
    # CONTACT
    # ─────────────────────────────────────────────────────────────
    contact_data = [
        ["Enterprise Inquiries", "QBITEL Bridge Platform", "Defense Practice"],
        ["enterprise@qbitel.com", "https://bridge.qbitel.com",
         "Cleared personnel available\nfor classified engagements"],
    ]
    tbl = Table(contact_data, colWidths=[CONTENT_W/3, CONTENT_W/3, CONTENT_W/3])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('BACKGROUND', (0, 1), (-1, 1), LIGHT_NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), GOLD),
        ('TEXTCOLOR', (0, 1), (-1, 1), WHITE_C),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, 1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, TEAL),
    ]))
    story.append(tbl)
    story.append(sp(8))

    story.append(Paragraph(
        "QBITEL Bridge — Defense &amp; Sovereign Networks | "
        "Quantum-Safe Security for National Defense Infrastructure | "
        "© 2026 QBITEL. All rights reserved. Confidential — For Authorized Recipients Only.",
        S['caption']
    ))

    doc.build(story)


if __name__ == '__main__':
    import os
    os.makedirs('docs/brochures', exist_ok=True)
    out = 'docs/brochures/QBITEL_Bridge_Defense_Marketing_Pitch.pdf'
    build_doc(out)
    print(f'PDF saved: {out}')
