"""Build QBITEL Bridge Automotive Deployment Checklist - PDF"""
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
CRITICAL_BG = HexColor('#FFF0F0')
CRITICAL_BORDER = HexColor('#CC0000')
PASS_BG    = HexColor('#F0FFF4')
PASS_BORDER = HexColor('#1A6B3A')
WARNING_BG  = HexColor('#FFFBF0')
WARNING_BORDER = HexColor('#CC7700')
PAGE_W, PAGE_H = letter
MARGIN = 0.85 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

def sp(n): return Spacer(1, n)


class PhaseHeader(Flowable):
    """Phase header: gold badge + navy bg + teal right accent + duration/owner badges."""
    def __init__(self, phase_num, title, duration, owner, width=None):
        Flowable.__init__(self)
        self.phase_num = str(phase_num)
        self.title = title
        self.duration = duration
        self.owner = owner
        self.w = width or CONTENT_W
        self.h = 50

    def wrap(self, availW, availH):
        return (self.w, self.h)

    def draw(self):
        c = self.canv
        w, h = self.w, self.h
        # Navy background
        c.setFillColor(NAVY)
        c.rect(0, 0, w, h, fill=1, stroke=0)
        # Teal right accent (5px)
        c.setFillColor(TEAL)
        c.rect(w - 5, 0, 5, h, fill=1, stroke=0)
        # Gold left badge (58px)
        c.setFillColor(GOLD)
        c.rect(0, 0, 58, h, fill=1, stroke=0)
        # "PHASE" label
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 7)
        pw = c.stringWidth('PHASE', 'Helvetica-Bold', 7)
        c.drawString(29 - pw / 2, h - 14, 'PHASE')
        # Phase number
        c.setFont('Helvetica-Bold', 18)
        nw = c.stringWidth(self.phase_num, 'Helvetica-Bold', 18)
        c.drawString(29 - nw / 2, h - 34, self.phase_num)
        # Title
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 11)
        c.drawString(68, h - 18, self.title)
        # Duration badge
        c.setFillColor(TEAL)
        c.roundRect(68, 6, 110, 16, 4, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica', 7.5)
        c.drawString(74, 10, self.duration)
        # Owner badge
        c.setFillColor(LIGHT_NAVY)
        owner_text = 'Owner: ' + self.owner
        ow = c.stringWidth(owner_text, 'Helvetica', 7.5) + 10
        c.roundRect(186, 6, ow, 16, 4, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.drawString(192, 10, owner_text)


class CheckItem(Flowable):
    """Checkbox item with optional sub-items and highlight."""
    def __init__(self, text, sub_items=None, highlight=None, width=None):
        Flowable.__init__(self)
        self.text = text
        self.sub_items = sub_items or []
        self.highlight = highlight  # None, 'critical', 'pass', 'warning'
        self.w = width or CONTENT_W
        self.line_h = 14
        self.sub_line_h = 13
        self.padding = 6
        # Calculate height
        self.h = self.line_h + len(self.sub_items) * self.sub_line_h + self.padding * 2

    def wrap(self, availW, availH):
        return (self.w, self.h)

    def draw(self):
        c = self.canv
        w, h = self.w, self.h
        # Background based on highlight
        if self.highlight == 'critical':
            c.setFillColor(CRITICAL_BG)
            border_color = CRITICAL_BORDER
        elif self.highlight == 'pass':
            c.setFillColor(PASS_BG)
            border_color = PASS_BORDER
        elif self.highlight == 'warning':
            c.setFillColor(WARNING_BG)
            border_color = WARNING_BORDER
        else:
            c.setFillColor(WHITE_C)
            border_color = MID_GREY
        c.rect(0, 0, w, h, fill=1, stroke=0)
        # Bottom border line
        c.setStrokeColor(HexColor('#E0E8EE'))
        c.setLineWidth(0.5)
        c.line(0, 0, w, 0)
        # Checkbox square
        checkbox_x = 8
        checkbox_y = h - self.padding - self.line_h + 2
        c.setStrokeColor(NAVY)
        c.setLineWidth(1)
        c.rect(checkbox_x, checkbox_y, 10, 10, fill=0, stroke=1)
        # Highlight indicator strip on left
        if self.highlight:
            c.setFillColor(border_color)
            c.rect(0, 0, 3, h, fill=1, stroke=0)
        # Main text
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 8.5)
        text_x = checkbox_x + 16
        text_y = h - self.padding - self.line_h + 3
        # Word wrap text
        words = self.text.split()
        lines = []
        line = ''
        max_w = w - text_x - 8
        for word in words:
            test = (line + ' ' + word).strip()
            if c.stringWidth(test, 'Helvetica', 8.5) < max_w:
                line = test
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        c.drawString(text_x, text_y, lines[0] if lines else self.text)
        # Sub-items
        for i, sub in enumerate(self.sub_items):
            sub_y = text_y - (i + 1) * self.sub_line_h
            sub_cb_x = checkbox_x + 20
            # Sub-checkbox
            c.setStrokeColor(TEAL)
            c.setLineWidth(0.75)
            c.rect(sub_cb_x, sub_y - 1, 8, 8, fill=0, stroke=1)
            c.setFillColor(MID_GREY)
            c.setFont('Helvetica', 7.5)
            c.drawString(sub_cb_x + 12, sub_y, sub)


def get_styles():
    return {
        'body': ParagraphStyle('body', fontName='Helvetica', fontSize=8.5,
                               leading=12, textColor=DARK_TEXT, spaceAfter=4),
        'h1': ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=16,
                             leading=20, textColor=NAVY, spaceAfter=8),
        'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=12,
                             leading=16, textColor=TEAL, spaceAfter=6),
        'caption': ParagraphStyle('caption', fontName='Helvetica', fontSize=7.5,
                                  leading=10, textColor=MID_GREY, spaceAfter=2),
        'signoff': ParagraphStyle('signoff', fontName='Helvetica', fontSize=8,
                                  leading=12, textColor=DARK_TEXT),
    }


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 30, PAGE_W, 30, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(MARGIN, PAGE_H - 20, 'QBITEL BRIDGE \u2014 AUTOMOTIVE DEPLOYMENT CHECKLIST')
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 20, f'Page {doc.page}')
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
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, PAGE_H * 0.55, PAGE_W, PAGE_H * 0.45, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, PAGE_H * 0.55, PAGE_W, 4, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 28)
    canvas.drawString(MARGIN, PAGE_H * 0.55 + PAGE_H * 0.45 * 0.75, 'QBITEL BRIDGE')
    canvas.setFillColor(GOLD)
    canvas.setFont('Helvetica-Bold', 18)
    canvas.drawString(MARGIN, PAGE_H * 0.55 + PAGE_H * 0.45 * 0.55, 'Automotive & Connected Vehicles')
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 20)
    canvas.drawString(MARGIN, PAGE_H * 0.55 + PAGE_H * 0.45 * 0.38, 'Deployment Checklist')
    canvas.setFont('Helvetica', 12)
    canvas.drawString(MARGIN, PAGE_H * 0.55 + PAGE_H * 0.45 * 0.24, '10 Phases | 135-Day Deployment | UNECE WP.29 Ready')
    # Stats
    stats = [('10', 'Phases'), ('135', 'Days'), ('70+', 'Checklist Items'), ('WP.29', 'Compliant')]
    box_w = (PAGE_W - 2 * MARGIN - 30) / 4
    for i, (val, lbl) in enumerate(stats):
        x = MARGIN + i * (box_w + 10)
        y = 0.12 * PAGE_H
        canvas.setFillColor(LIGHT_NAVY)
        canvas.roundRect(x, y, box_w, 0.1 * PAGE_H, 6, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.setFont('Helvetica-Bold', 16)
        vw = canvas.stringWidth(val, 'Helvetica-Bold', 16)
        canvas.drawString(x + box_w / 2 - vw / 2, y + 0.1 * PAGE_H * 0.55, val)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 8)
        lw = canvas.stringWidth(lbl, 'Helvetica', 8)
        canvas.drawString(x + box_w / 2 - lw / 2, y + 0.1 * PAGE_H * 0.25, lbl)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, PAGE_W, 0.08 * PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 9)
    canvas.drawString(MARGIN, 0.04 * PAGE_H, 'enterprise@qbitel.com  |  bridge.qbitel.com')
    canvas.restoreState()


def signoff_block(role_str):
    """Return a signoff line paragraph."""
    return Paragraph(
        f'<b>Sign-off:</b> {role_str} _________________ Date _____________',
        ParagraphStyle('so', fontName='Helvetica', fontSize=8, textColor=DARK_TEXT,
                       borderColor=GOLD, borderWidth=0.5, borderPadding=4,
                       backColor=LIGHT_BG))


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

    # Introduction
    story.append(Paragraph('QBITEL Bridge Automotive Deployment Checklist', S['h1']))
    story.append(Paragraph(
        'This checklist covers all 10 phases of a QBITEL Bridge automotive deployment, '
        'from initial WP.29 gap assessment through operational go-live. Each phase includes '
        'sign-off requirements for the responsible owner. Total deployment timeline: 135 days.',
        S['body']))
    story.append(sp(8))

    # Legend
    legend_data = [
        ['Highlight', 'Meaning'],
        ['Standard item', 'Normal deployment task'],
        ['Critical item', 'Blocks progression to next phase if incomplete'],
        ['Pass/Fail item', 'Has binary pass/fail acceptance criteria'],
        ['Warning item', 'Requires monitoring but does not block progression'],
    ]
    legend_tbl = Table(legend_data, colWidths=[CONTENT_W * 0.3, CONTENT_W * 0.7])
    legend_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY), ('TEXTCOLOR', (0, 0), (-1, 0), WHITE_C),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 8), ('GRID', (0, 0), (-1, -1), 0.5, MID_GREY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, LIGHT_BG]),
        ('LEFTPADDING', (0, 0), (-1, -1), 6), ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(legend_tbl)
    story.append(sp(12))

    # =========================================================================
    # PHASE 0: Pre-Engagement & WP.29 Gap Assessment
    # =========================================================================
    story.append(PhaseHeader('0', 'Pre-Engagement & WP.29 Gap Assessment', 'Days 1-5', 'Automotive CISO'))
    story.append(sp(4))
    phase0_items = [
        ('Identify OEM cybersecurity lead and SCMS provider contact', None, 'critical'),
        ('Complete UNECE WP.29 R155/R156 gap analysis', None, 'critical'),
        ('Inventory current V2X deployment (RSU count, OBU count, SCMS provider)', None, None),
        ('Review existing ISO/SAE 21434 TARA documentation', None, None),
        ('Identify fleet size and OTA platform capabilities', None, None),
        ('Define PQC algorithm selection (Falcon-512 vs Falcon-1024 vs hybrid)', None, 'warning'),
    ]
    for text, subs, hl in phase0_items:
        story.append(CheckItem(text, subs, hl))
    story.append(sp(6))
    story.append(signoff_block('Automotive CISO'))
    story.append(sp(12))

    # =========================================================================
    # PHASE 1: V2X Protocol Discovery & Analysis
    # =========================================================================
    story.append(PhaseHeader('1', 'V2X Protocol Discovery & Analysis', 'Days 6-12', 'V2X Architect'))
    story.append(sp(4))
    phase1_items = [
        ('Deploy V2X passive capture on target RSU corridors', None, 'critical'),
        ('Enumerate all V2X message types in production (BSM, SPaT, MAP, TIM, EVA)', None, None),
        ('Audit current certificate chain (IEEE 1609.2 compliance)', None, 'critical'),
        ('Measure baseline V2X message rates (avg/peak msg/sec)', None, None),
        ('Assess DSRC vs C-V2X split in deployment', None, None),
        ('Document legacy RSU firmware versions', None, 'warning'),
    ]
    for text, subs, hl in phase1_items:
        story.append(CheckItem(text, subs, hl))
    story.append(sp(6))
    story.append(signoff_block('V2X Architect'))
    story.append(sp(12))

    # =========================================================================
    # PHASE 2: SCMS & PKI Infrastructure Readiness
    # =========================================================================
    story.append(PhaseHeader('2', 'SCMS & PKI Infrastructure Readiness', 'Days 13-20', 'PKI Team'))
    story.append(sp(4))
    phase2_items = [
        ('Confirm SCMS provider API access (CAMP or OBS)', None, 'critical'),
        ('Provision Falcon-512 root CA in HSM (FIPS 140-3 Level 3)', None, 'critical'),
        ('Generate pseudonym certificate pool (initial batch: 52 weeks x fleet size)', None, 'critical'),
        ('Configure CRL distribution infrastructure (<100ms target)', None, 'critical'),
        ('Test bulk enrollment pipeline (10,000 vehicles/hour throughput)', None, 'pass'),
        ('Establish misbehavior reporting channel to SCMS', None, None),
    ]
    for text, subs, hl in phase2_items:
        story.append(CheckItem(text, subs, hl))
    story.append(sp(6))
    story.append(signoff_block('PKI Lead'))
    story.append(sp(12))

    # =========================================================================
    # PHASE 3: Cryptographic Architecture Validation
    # =========================================================================
    story.append(PhaseHeader('3', 'Cryptographic Architecture Validation', 'Days 21-28', 'Crypto Team'))
    story.append(sp(4))
    phase3_items = [
        ('Benchmark Falcon-512 verification on target ECU hardware', None, 'critical'),
        ('Validate <5ms V2X verification latency (IEEE 1609.2 requirement: <10ms)', None, 'pass'),
        ('Test batch verification throughput at peak density (>=1,500 msg/sec)', None, 'pass'),
        ('Validate 666-byte implicit certificate fits DSRC frame size', None, 'pass'),
        ('Test hybrid ECDSA+Falcon-512 dual-signing mode', None, 'critical'),
        ('Benchmark OTA update signing (ML-DSA-65) performance', None, None),
    ]
    for text, subs, hl in phase3_items:
        story.append(CheckItem(text, subs, hl))
    story.append(sp(6))
    story.append(signoff_block('Crypto Lead'))
    story.append(sp(12))

    # =========================================================================
    # PHASE 4: RSU Infrastructure Deployment
    # =========================================================================
    story.append(PhaseHeader('4', 'RSU Infrastructure Deployment', 'Days 29-40', 'Infrastructure'))
    story.append(sp(4))
    phase4_items = [
        ('Update RSU firmware to QBITEL V2X authentication module', None, 'critical'),
        ('Configure Falcon-512 certificate verification on each RSU', None, 'critical'),
        ('Deploy CRL update mechanism on all RSUs', None, 'critical'),
        ('Test RSU-to-RSU certificate exchange', None, 'pass'),
        ('Validate RSU backward compatibility with legacy OBUs (ECDSA fallback)', None, 'critical'),
        ('Monitor RSU V2X authentication success rate (target: >99.99%)', None, 'warning'),
    ]
    for text, subs, hl in phase4_items:
        story.append(CheckItem(text, subs, hl))
    story.append(sp(6))
    story.append(signoff_block('Infrastructure Lead'))
    story.append(sp(12))

    # =========================================================================
    # PHASE 5: OBU (Vehicle) Certificate Provisioning
    # =========================================================================
    story.append(PhaseHeader('5', 'OBU (Vehicle) Certificate Provisioning', 'Days 41-55', 'OTA Team'))
    story.append(sp(4))
    phase5_items = [
        ('Generate Falcon-512 key pairs for canary fleet (10 vehicles)', None, 'critical'),
        ('Provision pseudonym certificate pools via SCMS API', None, 'critical'),
        ('Deploy QBITEL V2X library via OTA to canary fleet', None, 'critical'),
        ('Validate canary fleet V2X authentication success rate', None, 'pass'),
        ('Monitor for regressions vs. baseline (V2X message rate, latency)', None, 'warning'),
        ('Approve canary -> 1% fleet expansion', None, 'critical'),
    ]
    for text, subs, hl in phase5_items:
        story.append(CheckItem(text, subs, hl))
    story.append(sp(6))
    story.append(signoff_block('OTA Lead'))
    story.append(sp(12))

    # =========================================================================
    # PHASE 6: Staged Fleet OTA Rollout
    # =========================================================================
    story.append(PhaseHeader('6', 'Staged Fleet OTA Rollout', 'Days 56-90', 'Fleet Operations'))
    story.append(sp(4))
    phase6_items = [
        ('Deploy to 1% of fleet (monitor 24h, error rate <0.1%)',
         ['Confirm automatic rollback is armed', 'Verify delta update delivery <4 hours'], 'critical'),
        ('Deploy to 10% of fleet (monitor 48h, error rate <0.1%)',
         ['Geographic diversity check', 'Latency regression check at 10% scale'], 'critical'),
        ('Deploy to 50% of fleet (monitor 72h)',
         ['Peak traffic load test during deployment', 'SCMS pseudonym pool adequacy check'], 'warning'),
        ('Deploy to 100% of fleet',
         ['Final coverage verification', 'All vehicles reporting authentication success'], 'critical'),
        ('Verify TPM attestation on all updated vehicles', None, 'pass'),
        ('Generate fleet coverage report for WP.29 documentation', None, 'critical'),
    ]
    for text, subs, hl in phase6_items:
        story.append(CheckItem(text, subs, hl))
    story.append(sp(6))
    story.append(signoff_block('Fleet Operations Lead'))
    story.append(sp(12))

    # =========================================================================
    # PHASE 7: Misbehavior Detection Activation
    # =========================================================================
    story.append(PhaseHeader('7', 'Misbehavior Detection Activation', 'Days 91-100', 'Security Team'))
    story.append(sp(4))
    phase7_items = [
        ('Enable V2X anomaly detection on all RSUs', None, 'critical'),
        ('Configure physics-based validation (speed/position plausibility)', None, 'critical'),
        ('Set misbehavior reporting threshold and SCMS integration', None, None),
        ('Test spoofing detection with simulation (pass: detection <100ms)',
         ['Simulate 100 spoofed BSMs', 'Verify all 100 rejected within 100ms', 'Verify zero false positives on legitimate messages'], 'pass'),
        ('Validate false positive rate (<0.0001% of legitimate messages)', None, 'pass'),
        ('Configure P1 alert: Active spoofing detected -> immediate SOC notification', None, 'critical'),
    ]
    for text, subs, hl in phase7_items:
        story.append(CheckItem(text, subs, hl))
    story.append(sp(6))
    story.append(signoff_block('Security Lead'))
    story.append(sp(12))

    # =========================================================================
    # PHASE 8: UNECE WP.29 / ISO 21434 Evidence Generation
    # =========================================================================
    story.append(PhaseHeader('8', 'UNECE WP.29 / ISO 21434 Evidence Generation', 'Days 101-115', 'Compliance'))
    story.append(sp(4))
    phase8_items = [
        ('Generate automated TARA covering V2X attack surfaces', None, 'critical'),
        ('Produce security validation test results', None, 'critical'),
        ('Generate CSMS documentation for WP.29 R155 submission', None, 'critical'),
        ('Produce R156 SUMS documentation and OTA audit logs', None, 'critical'),
        ('Prepare fleet coverage certificate for type approval', None, 'critical'),
        ('Review all evidence with legal/regulatory team', None, 'warning'),
    ]
    for text, subs, hl in phase8_items:
        story.append(CheckItem(text, subs, hl))
    story.append(sp(6))
    story.append(signoff_block('Compliance Lead'))
    story.append(sp(12))

    # =========================================================================
    # PHASE 9: Performance & Safety Validation
    # =========================================================================
    story.append(PhaseHeader('9', 'Performance & Safety Validation', 'Days 116-125', 'Validation Team'))
    story.append(sp(4))
    phase9_items = [
        ('V2X latency test: 100% messages <5ms (pass/fail)',
         ['Test at low density (rural: 50 msg/sec)', 'Test at high density (urban: 1,000 msg/sec)', 'All results must be <5ms'], 'pass'),
        ('Batch throughput test: >=1,500 msg/sec at peak density (pass/fail)', None, 'pass'),
        ('Spoofing detection test: 0/100 simulated spoofed messages accepted (pass/fail)',
         ['Inject 100 crafted spoofed BSMs', 'All must be rejected', 'Zero legitimate messages rejected'], 'pass'),
        ('OTA rollback test: automatic rollback triggered correctly (pass/fail)',
         ['Inject artificial error rate >0.1%', 'Confirm rollback triggers automatically', 'Confirm rollback completes within 30 minutes'], 'pass'),
        ('Hybrid mode test: ECDSA legacy compatibility verified (pass/fail)',
         ['Legacy RSU sees valid ECDSA signature', 'Upgraded RSU verifies both ECDSA and Falcon-512'], 'pass'),
    ]
    for text, subs, hl in phase9_items:
        story.append(CheckItem(text, subs, hl))
    story.append(sp(6))
    story.append(signoff_block('Validation Lead'))
    story.append(sp(12))

    # =========================================================================
    # PHASE 10: Operational Handover & Go-Live
    # =========================================================================
    story.append(PhaseHeader('10', 'Operational Handover & Go-Live', 'Days 126-135', 'CTO / CISO'))
    story.append(sp(4))
    phase10_items = [
        ('Configure fleet health monitoring dashboard', None, 'critical'),
        ('Establish pseudonym rotation automation (weekly)', None, 'critical'),
        ('Set up 24/7 NOC alerting for V2X anomalies', None, 'critical'),
        ('Complete OEM security team training (V2X security operations)',
         ['V2X authentication monitoring', 'Misbehavior detection response', 'OTA rollback procedures'], None),
        ('Publish WP.29 CSMS evidence package', None, 'critical'),
        ('Execute go-live sign-off', None, 'critical'),
    ]
    for text, subs, hl in phase10_items:
        story.append(CheckItem(text, subs, hl))
    story.append(sp(6))
    story.append(signoff_block('CTO'))
    story.append(sp(4))
    story.append(signoff_block('CISO'))
    story.append(sp(12))

    # =========================================================================
    # APPENDIX A: Performance Targets
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph('Appendix A: Performance Targets', S['h1']))
    story.append(sp(6))
    perf_data = [
        ['Metric', 'Minimum Requirement', 'QBITEL Target', 'Status'],
        ['V2X verification latency', '<10ms', '<5ms', 'EXCEEDS'],
        ['Batch throughput', '500 msg/sec', '1,500 msg/sec', 'EXCEEDS'],
        ['Certificate size', '<2KB', '666 bytes', 'EXCEEDS'],
        ['CRL distribution to RSUs', '<1 second', '<100ms', 'EXCEEDS'],
        ['Fleet OTA coverage', '100%', 'Staged 12 months', 'COMPLIANT'],
        ['False positive rate', '<0.001%', '<0.0001%', 'EXCEEDS'],
        ['SCMS bulk enrollment', 'N/A', '10,000 vehicles/hour', 'EXCEEDS'],
        ['Cost per vehicle/year', '$50-200 (classical)', '$10-30 (PQC)', 'LOWER COST'],
    ]
    pt = Table(perf_data, colWidths=[CONTENT_W * 0.33, CONTENT_W * 0.22, CONTENT_W * 0.25, CONTENT_W * 0.2])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY), ('TEXTCOLOR', (0, 0), (-1, 0), WHITE_C),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, 0), 8.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, TABLE_ALT]),
        ('FONTSIZE', (0, 1), (-1, -1), 8), ('GRID', (0, 0), (-1, -1), 0.5, MID_GREY),
        ('LEFTPADDING', (0, 0), (-1, -1), 6), ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TEXTCOLOR', (3, 1), (3, -1), HexColor('#1A6B3A')),
        ('FONTNAME', (3, 1), (3, -1), 'Helvetica-Bold'),
    ]))
    story.append(pt)
    story.append(sp(16))

    # =========================================================================
    # APPENDIX B: Rollback Procedures
    # =========================================================================
    story.append(Paragraph('Appendix B: Rollback Procedures', S['h1']))
    story.append(sp(6))
    story.append(Paragraph('<b>Automatic Rollback Trigger Conditions</b>', S['h2']))
    story.append(sp(4))
    auto_triggers = [
        ('Error rate >0.1% in any deployment stage', None, 'critical'),
        ('V2X authentication failure rate >0.01% sustained for >5 minutes', None, 'critical'),
        ('TPM attestation failure on >10 vehicles in a batch', None, 'critical'),
        ('OTA delivery failure rate >5% in any cohort', None, 'warning'),
        ('V2X message latency >8ms (80% of the 10ms budget) on >1% of messages', None, 'warning'),
    ]
    for text, subs, hl in auto_triggers:
        story.append(CheckItem(text, subs, hl))
    story.append(sp(10))
    story.append(Paragraph('<b>Manual Rollback Steps</b>', S['h2']))
    story.append(sp(4))
    manual_steps = [
        ('Access fleet management console at https://bridge.qbitel.com/fleet', None, None),
        ('Navigate to Fleet > OTA Management > Rollback', None, None),
        ('Select affected vehicle cohort (by region, VIN range, or deployment batch)', None, None),
        ('Confirm rollback to previous firmware version', None, 'critical'),
        ('Monitor rollback completion rate (target: 100% within 30 minutes)', None, None),
        ('Verify TPM attestation on all rolled-back vehicles', None, 'pass'),
        ('File incident report to QBITEL NOC: noc@qbitel.com', None, None),
    ]
    for text, subs, hl in manual_steps:
        story.append(CheckItem(text, subs, hl))
    story.append(sp(10))
    story.append(Paragraph('<b>Escalation Contacts</b>', S['h2']))
    story.append(sp(4))
    esc_data = [
        ['Severity', 'Contact', 'Response Time'],
        ['P1 - Active V2X spoofing', 'QBITEL NOC: noc@qbitel.com | +1-800-QBITEL1', '<15 minutes'],
        ['P2 - OTA rollback failure', 'QBITEL Automotive Engineering: auto@qbitel.com', '<1 hour'],
        ['P3 - Performance degradation', 'enterprise@qbitel.com', '<4 hours'],
        ['General enquiries', 'enterprise@qbitel.com | bridge.qbitel.com', 'Next business day'],
    ]
    et = Table(esc_data, colWidths=[CONTENT_W * 0.25, CONTENT_W * 0.5, CONTENT_W * 0.25])
    et.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY), ('TEXTCOLOR', (0, 0), (-1, 0), WHITE_C),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, 0), 8.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BG, TABLE_ALT]),
        ('FONTSIZE', (0, 1), (-1, -1), 8), ('GRID', (0, 0), (-1, -1), 0.5, MID_GREY),
        ('LEFTPADDING', (0, 0), (-1, -1), 6), ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TEXTCOLOR', (0, 1), (0, 1), CRITICAL_BORDER),
        ('FONTNAME', (0, 1), (0, 1), 'Helvetica-Bold'),
    ]))
    story.append(et)
    story.append(sp(12))

    # Contact footer
    contact_data = [
        [Paragraph('<b>enterprise@qbitel.com</b>',
                   ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=10, textColor=GOLD)),
         Paragraph('<b>bridge.qbitel.com</b>',
                   ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=10, textColor=GOLD)),
         Paragraph('<b>noc@qbitel.com (24/7)</b>',
                   ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=10, textColor=GOLD))],
        [Paragraph('Sales & Licensing', ParagraphStyle('cl', fontName='Helvetica', fontSize=8, textColor=WHITE_C)),
         Paragraph('Portal & Documentation', ParagraphStyle('cl', fontName='Helvetica', fontSize=8, textColor=WHITE_C)),
         Paragraph('NOC Support', ParagraphStyle('cl', fontName='Helvetica', fontSize=8, textColor=WHITE_C))],
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
    build_doc('docs/brochures/QBITEL_Automotive_Deployment_Checklist.pdf')
    print('PDF saved: docs/brochures/QBITEL_Automotive_Deployment_Checklist.pdf')
