"""
Build QBITEL Bridge Insurance Deployment Checklist - Professional PDF
10-phase deployment checklist for Insurance & Reinsurance environments.
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

# Brand Colors
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
ORANGE     = HexColor('#E07B00')

PAGE_W, PAGE_H = letter
MARGIN = 0.8 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


# Custom Flowables

class PhaseHeader(Flowable):
    def __init__(self, phase_num, title, duration, owner, width=None):
        super().__init__()
        self.phase_num = phase_num
        self.title = title
        self.duration = duration
        self.owner = owner
        self.w = width or CONTENT_W
        self.h = 50

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(NAVY)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        # Phase number badge
        badge_colors = [GOLD, TEAL, HexColor('#8B5E00'), HexColor('#006B7A'),
                        HexColor('#1A4D8F'), GOLD, TEAL, NAVY, HexColor('#2E8B57'),
                        GOLD, TEAL]
        badge_idx = self.phase_num % len(badge_colors)
        c.setFillColor(badge_colors[badge_idx])
        c.rect(0, 0, 52, self.h, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 8)
        label = f'PHASE'
        lw = c.stringWidth(label, 'Helvetica-Bold', 8)
        c.drawString(26 - lw / 2, self.h - 18, label)
        c.setFont('Helvetica-Bold', 16)
        nw = c.stringWidth(str(self.phase_num), 'Helvetica-Bold', 16)
        c.drawString(26 - nw / 2, self.h - 36, str(self.phase_num))
        # Title
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 12)
        c.drawString(62, self.h - 20, self.title.upper())
        # Duration and owner
        c.setFillColor(TEAL)
        c.setFont('Helvetica', 8)
        info = f'Duration: {self.duration}   |   Owner: {self.owner}'
        c.drawString(62, 10, info)
        # Right accent
        c.setFillColor(GOLD)
        c.rect(self.w - 3, 0, 3, self.h, fill=1, stroke=0)


class CheckItem(Flowable):
    def __init__(self, category, text, priority='Required', width=None):
        super().__init__()
        self.category = category
        self.text = text
        self.priority = priority
        self.w = width or CONTENT_W
        self.line_h = 13
        lines = max(1, (len(text) // 85) + 1)
        self.h = lines * self.line_h + 18

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Background
        priority_bgs = {
            'Critical': HexColor('#FFF8E8'),
            'Required': WHITE_C,
            'Recommended': HexColor('#F0FFF4'),
            'Optional': LIGHT_BG,
        }
        c.setFillColor(priority_bgs.get(self.priority, WHITE_C))
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        # Bottom border
        c.setStrokeColor(HexColor('#CCDDEE'))
        c.setLineWidth(0.5)
        c.line(0, 0, self.w, 0)
        # Checkbox
        c.setStrokeColor(TEAL)
        c.setLineWidth(1)
        c.rect(8, self.h / 2 - 7, 14, 14, fill=0, stroke=1)
        # Category badge
        cat_colors = {
            'Security': NAVY,
            'Compliance': TEAL_DARK,
            'Technical': HexColor('#1A4D8F'),
            'Operational': HexColor('#5A6A7A'),
            'Executive': HexColor('#8B5E00'),
            'Actuarial': HexColor('#006B7A'),
            'Fraud': HexColor('#8B1A1A'),
            'Reinsurance': HexColor('#1A5E8B'),
            'Claims': HexColor('#2E5E2E'),
            'Data': HexColor('#5A1A8B'),
        }
        cat_col = cat_colors.get(self.category, MID_GREY)
        cat_w = max(50, len(self.category) * 5.5 + 8)
        c.setFillColor(cat_col)
        c.roundRect(28, self.h / 2 - 7, cat_w, 14, 3, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 6.5)
        cw = c.stringWidth(self.category.upper(), 'Helvetica-Bold', 6.5)
        c.drawString(28 + cat_w / 2 - cw / 2, self.h / 2 - 1, self.category.upper())
        # Priority indicator
        pri_cols = {
            'Critical': HexColor('#C0392B'),
            'Required': TEAL,
            'Recommended': GREEN,
            'Optional': MID_GREY,
        }
        p_col = pri_cols.get(self.priority, TEAL)
        pri_x = self.w - 70
        c.setFillColor(p_col)
        c.roundRect(pri_x, self.h / 2 - 7, 65, 14, 3, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 6.5)
        pw = c.stringWidth(self.priority.upper(), 'Helvetica-Bold', 6.5)
        c.drawString(pri_x + 32 - pw / 2, self.h / 2 - 1, self.priority.upper())
        # Item text
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 8.5)
        text_x = 28 + max(50, len(self.category) * 5.5 + 8) + 8
        text_w = pri_x - text_x - 8
        # Simple text wrapping
        words = self.text.split()
        line_chars = int(text_w / 5.2)
        lines = []
        cur = ''
        for w in words:
            if len(cur) + len(w) + 1 <= line_chars:
                cur = (cur + ' ' + w).strip()
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        y = self.h / 2 + (len(lines) - 1) * self.line_h / 2 - 4
        for line in lines:
            c.drawString(text_x, y, line)
            y -= self.line_h


class MilestoneBlock(Flowable):
    def __init__(self, title, criteria_lines, width=None):
        super().__init__()
        self.title = title
        self.criteria_lines = criteria_lines
        self.w = width or CONTENT_W
        self.line_h = 13
        self.h = len(criteria_lines) * self.line_h + 38

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(GREEN_LIGHT)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        c.setFillColor(GREEN)
        c.rect(0, self.h - 28, self.w, 28, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 10)
        c.drawString(10, self.h - 18, f'MILESTONE: {self.title}')
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 8.5)
        y = self.h - 44
        for line in self.criteria_lines:
            c.drawString(14, y, f'  {line}')
            y -= self.line_h
        c.setFillColor(GREEN)
        c.rect(0, 0, 4, self.h - 28, fill=1, stroke=0)


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
    canvas.drawString(MARGIN + 1.1 * inch, PAGE_H - 0.32 * inch,
                      'INSURANCE & REINSURANCE — DEPLOYMENT CHECKLIST')
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
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    p = canvas.beginPath()
    p.moveTo(w * 0.58, h); p.lineTo(w, h); p.lineTo(w, h * 0.66)
    p.close()
    canvas.drawPath(p, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    p2 = canvas.beginPath()
    p2.moveTo(w * 0.74, h); p2.lineTo(w, h); p2.lineTo(w, h * 0.80)
    p2.close()
    canvas.drawPath(p2, fill=1, stroke=0)
    canvas.setFillColor(TEAL_DARK)
    canvas.rect(0, 0, w, 1.5 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, 1.5 * inch, w, 5, fill=1, stroke=0)

    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 46)
    canvas.drawString(MARGIN, h * 0.70, 'QBITEL')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica-Bold', 46)
    canvas.drawString(MARGIN, h * 0.70 - 52, 'BRIDGE')
    canvas.setFillColor(GOLD)
    canvas.rect(MARGIN, h * 0.70 - 62, 3.0 * inch, 4, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 14)
    canvas.drawString(MARGIN, h * 0.70 - 90, 'INSURANCE & REINSURANCE')
    canvas.setFont('Helvetica-Bold', 18)
    canvas.drawString(MARGIN, h * 0.70 - 116, 'DEPLOYMENT CHECKLIST')
    canvas.setFillColor(HexColor('#CCDDE8'))
    canvas.setFont('Helvetica-Oblique', 11)
    canvas.drawString(MARGIN, h * 0.70 - 142,
                      '11 Phases | Zero Downtime | Solvency II / NY DFS 500 Ready')

    # Phase timeline indicators
    phases = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
    box_w = CONTENT_W / len(phases) - 2
    by = h * 0.36
    for i, p in enumerate(phases):
        bx = MARGIN + i * (box_w + 2)
        bg = TEAL if i % 3 == 0 else (GOLD if i % 3 == 1 else LIGHT_NAVY)
        canvas.setFillColor(bg)
        canvas.roundRect(bx, by, box_w, 0.5 * inch, 3, fill=1, stroke=0)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica-Bold', 8)
        pw2 = canvas.stringWidth(p, 'Helvetica-Bold', 8)
        canvas.drawString(bx + box_w / 2 - pw2 / 2, by + 16, p)

    # Stats
    stats = [
        ('11', 'Deployment\nPhases'),
        ('100+', 'Checklist\nItems'),
        ('30 Days', 'To Full\nDeployment'),
        ('0', 'Downtime\nRequired'),
    ]
    box_w2 = (CONTENT_W - 3 * 0.12 * inch) / 4
    by2 = h * 0.24
    bh = 0.85 * inch
    for i, (big, small) in enumerate(stats):
        bx = MARGIN + i * (box_w2 + 0.12 * inch)
        bg_c = TEAL if i % 2 == 0 else LIGHT_NAVY
        canvas.setFillColor(bg_c)
        canvas.roundRect(bx, by2, box_w2, bh, 5, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.rect(bx, by2 + bh - 4, box_w2, 4, fill=1, stroke=0)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica-Bold', 18)
        tw = canvas.stringWidth(big, 'Helvetica-Bold', 18)
        canvas.drawString(bx + (box_w2 - tw) / 2, by2 + bh - 28, big)
        canvas.setFont('Helvetica', 7.5)
        for j, line in enumerate(small.split('\n')):
            lw = canvas.stringWidth(line, 'Helvetica', 7.5)
            canvas.drawString(bx + (box_w2 - lw) / 2, by2 + bh - 46 - j * 11, line)

    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 9.5)
    canvas.drawString(MARGIN, 0.82 * inch,
                      'For CISO, CRO, Compliance, Actuarial, and Technology Teams')
    canvas.setFont('Helvetica', 8.5)
    canvas.setFillColor(HexColor('#AABBCC'))
    canvas.drawString(MARGIN, 0.55 * inch,
                      'Confidential Internal Document  |  Version 1.0  |  February 2026')
    canvas.restoreState()


# Style Definitions

def get_styles():
    s = {}
    s['body'] = ParagraphStyle(
        'body', fontName='Helvetica', fontSize=9.5, leading=14,
        textColor=DARK_TEXT, spaceAfter=6, alignment=TA_JUSTIFY)
    s['note'] = ParagraphStyle(
        'note', fontName='Helvetica-Oblique', fontSize=8.5, leading=12,
        textColor=MID_GREY, spaceAfter=5, alignment=TA_LEFT,
        leftIndent=8)
    s['legend_title'] = ParagraphStyle(
        'legend_title', fontName='Helvetica-Bold', fontSize=9,
        textColor=NAVY, spaceAfter=4)
    return s


def phase_block(story, phase_num, title, duration, owner, items, milestone_title, milestone_criteria, styles):
    """Render a complete phase: header + checklist items + milestone."""
    story.append(KeepTogether([
        PhaseHeader(phase_num, title, duration, owner),
        Spacer(1, 2),
    ]))
    for category, text, priority in items:
        story.append(CheckItem(category, text, priority))
    story.append(Spacer(1, 6))
    story.append(MilestoneBlock(milestone_title, milestone_criteria))
    story.append(Spacer(1, 12))


def build_story(styles):
    story = []

    # Legend
    legend_data = [
        [Paragraph('Priority Legend', styles['legend_title']),
         Paragraph('CRITICAL: Must complete before proceeding', styles['note']),
         Paragraph('REQUIRED: Mandatory for go-live', styles['note']),
         Paragraph('RECOMMENDED: Best practice', styles['note']),
         Paragraph('OPTIONAL: Available capability', styles['note']),
        ],
    ]
    legend_tbl = Table([[Paragraph('Priority Legend', styles['legend_title']),
                         Paragraph('CRITICAL: Must complete before proceeding', styles['note']),
                         Paragraph('REQUIRED: Mandatory for go-live', styles['note']),
                         Paragraph('RECOMMENDED: Best practice', styles['note']),
                         Paragraph('OPTIONAL: Available capability', styles['note']),
                         ]],
                       colWidths=[CONTENT_W * 0.18, CONTENT_W * 0.21,
                                  CONTENT_W * 0.21, CONTENT_W * 0.20, CONTENT_W * 0.20])
    legend_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCDDEE')),
    ]))
    story.append(legend_tbl)
    story.append(Spacer(1, 12))

    # ─── Phase 0: Pre-Engagement ───────────────────────────────────────────────
    phase_block(story, 0, 'Pre-Engagement', 'Week -2 to -1', 'CRO + CISO + Compliance',
        [
            ('Executive', 'Obtain CRO and CISO joint sign-off on QBITEL Bridge deployment scope and authority', 'Critical'),
            ('Executive', 'Confirm board-level awareness of quantum threat to long-term policyholder data (life, annuity, LTC lines)', 'Critical'),
            ('Compliance', 'Identify all applicable regulatory frameworks: NY DFS 500, Solvency II, NAIC Model Law, HIPAA, GDPR/CCPA, PCI-DSS, IFRS 17', 'Critical'),
            ('Compliance', 'Notify regulatory counsel of planned PQC deployment to support proactive regulatory communication', 'Recommended'),
            ('Compliance', 'Review current NY DFS 500 Section 500.15 encryption compliance status and identify known gaps', 'Critical'),
            ('Data', 'Initiate data classification review: identify all policy lines with 10+ year data retention requirements', 'Critical'),
            ('Data', 'Engage Chief Actuary to identify actuarial model data flows requiring priority protection', 'Required'),
            ('Technical', 'Inventory all connected systems: policy admin, claims, EDI clearinghouses, reinsurance portals, trading partners', 'Required'),
            ('Technical', 'Identify all EDI trading partner connections with current encryption level documentation', 'Required'),
            ('Security', 'Review most recent penetration test and vulnerability assessment results for protocol coverage gaps', 'Recommended'),
            ('Executive', 'Assign project sponsor (recommended: CRO or CISO), project manager, and technical lead', 'Required'),
            ('Operational', 'Schedule Phase 1 deployment window — no disruption to live insurance operations required', 'Required'),
        ],
        'Pre-Engagement Complete',
        [
            'CRO and CISO have signed deployment scope document',
            'Applicable regulatory frameworks documented and compliance gaps identified',
            'Data classification scope defined — all 10+ year retention data types identified',
            'System inventory complete and project team assigned',
        ],
        styles)

    # ─── Phase 1: Protocol Discovery ───────────────────────────────────────────
    phase_block(story, 1, 'Protocol Discovery', 'Days 1-3', 'CISO + Network Engineering',
        [
            ('Technical', 'Deploy QBITEL Bridge passive network tap — zero traffic disruption, read-only monitoring mode', 'Critical'),
            ('Technical', 'Configure tap points: policy admin network interface, EDI gateway, reinsurance portal DMZ, mainframe network segment', 'Critical'),
            ('Security', 'Initiate passive ACORD XML protocol discovery — identify all ACORD message types in use (AL3, ACORD 103, 165, etc.)', 'Required'),
            ('Security', 'Initiate X12 EDI passive tap — identify all transaction sets: 834, 835, 837P, 837I, 837D, 270/271, 276/277', 'Required'),
            ('Security', 'Identify all HL7 v2.x and FHIR R4 message flows for health insurance and group life lines', 'Required'),
            ('Security', 'Discover SWIFT MT and MX message flows — identify all reinsurance settlement and premium payment channels', 'Required'),
            ('Security', 'Map mainframe TN3270e session topology — identify all connected terminal emulation clients and servers', 'Critical'),
            ('Security', 'Identify all ISO 20022 premium payment message flows (pacs.008, pacs.009, camt messages)', 'Required'),
            ('Security', 'Discover FIX protocol sessions for ILS trading desk if applicable', 'Optional'),
            ('Technical', 'Map all proprietary policy admin integration protocols — document undocumented protocol variants', 'Required'),
            ('Data', 'Classify discovered protocol streams by data sensitivity: NPI, PHI, PII, financial, actuarial, reinsurance treaty', 'Critical'),
            ('Compliance', 'Map discovered protocol streams to regulatory control requirements (NY DFS 500, Solvency II, HIPAA, NAIC)', 'Required'),
            ('Security', 'Generate risk heat map — protocols ranked by sensitivity and quantum exposure', 'Required'),
            ('Compliance', 'Produce Insurance Protocol Security Assessment Report for CRO/CISO review', 'Critical'),
        ],
        'Protocol Discovery Complete',
        [
            'All insurance protocol types discovered and catalogued (ACORD, X12 EDI, HL7, SWIFT, TN3270e)',
            'Data classification complete — all NPI, PHI, PII flows identified and mapped',
            'Risk heat map produced and presented to CRO/CISO',
            'Insurance Protocol Security Assessment Report delivered',
        ],
        styles)

    story.append(PageBreak())

    # ─── Phase 2: Infrastructure Readiness ────────────────────────────────────
    phase_block(story, 2, 'Infrastructure Readiness', 'Days 3-7', 'CISO + Infrastructure + Procurement',
        [
            ('Security', 'Provision FIPS 140-3 Level 3 HSMs for PQC key generation and storage — minimum two for HA', 'Critical'),
            ('Technical', 'Configure HSM high-availability clustering with key replication — test failover before deployment', 'Critical'),
            ('Technical', 'Provision QBITEL Bridge appliances (physical or virtual) for inline deployment at each tap point', 'Required'),
            ('Technical', 'Configure active-active HA clustering for Bridge appliances — verify 99.99% availability design', 'Required'),
            ('Security', 'Implement network segmentation for mainframe policy admin traffic — isolate TN3270e network segment', 'Critical'),
            ('Technical', 'Establish Bridge management network — segregated from policy data network paths', 'Required'),
            ('Compliance', 'Enable comprehensive audit logging for claims database access — retain minimum 7 years per NAIC Model Law', 'Critical'),
            ('Security', 'Configure SIEM integration for Bridge security events — connect to existing SOC tooling', 'Required'),
            ('Technical', 'Validate network bandwidth and latency baseline for mainframe and EDI networks before Bridge inline insertion', 'Required'),
            ('Operational', 'Confirm change management process for Bridge deployment — schedule maintenance windows if required', 'Required'),
            ('Security', 'Prepare certificate authority infrastructure for quantum-safe certificate issuance to trading partners', 'Recommended'),
            ('Compliance', 'Document infrastructure design for Solvency II ORSA and NY DFS 500 Section 500.16 evidence', 'Required'),
        ],
        'Infrastructure Ready',
        [
            'FIPS 140-3 Level 3 HSMs provisioned and HA cluster tested',
            'Bridge appliances provisioned and HA design validated',
            'Network segmentation for mainframe implemented',
            'Audit logging enabled and SIEM integration confirmed',
        ],
        styles)

    # ─── Phase 3: Policy Configuration ────────────────────────────────────────
    phase_block(story, 3, 'Policy Configuration', 'Days 6-9', 'CISO + Compliance + Actuarial',
        [
            ('Security', 'Configure PQC encryption tiers by data type: ML-KEM-1024 for life/annuity/actuarial, ML-KEM-768 for health/reinsurance, ML-KEM-512 for P&C/claims', 'Critical'),
            ('Compliance', 'Configure Solvency II resilience policy: define ICT security controls per EIOPA Guidelines on ICT Security', 'Critical'),
            ('Compliance', 'Configure NY DFS 500 alignment policy: Section 500.15 encryption enforcement, Section 500.12 authentication, Section 500.17 notification', 'Critical'),
            ('Compliance', 'Configure NAIC Model Law cybersecurity program documentation — define NPI scope and protection controls', 'Required'),
            ('Compliance', 'Configure HIPAA Security Rule controls: PHI encryption (164.312(a)(2)(iv)), transmission security (164.312(e)(1)), audit logging (164.312(b))', 'Required'),
            ('Security', 'Define key rotation schedule by data classification tier — align to policy retention horizon and regulatory requirements', 'Critical'),
            ('Security', 'Configure cryptographic agility settings — define algorithm upgrade triggers and rollback procedures', 'Required'),
            ('Actuarial', 'Configure actuarial data protection policy — define model data classification and ML-KEM-1024 enforcement scope', 'Required'),
            ('Data', 'Configure GDPR/CCPA data subject rights support — enable cryptographic key deletion for erasure requests', 'Required'),
            ('Fraud', 'Define fraud detection threshold parameters — configure sensitivity levels for synthetic identity, upcoding, and ring detection', 'Required'),
            ('Reinsurance', 'Configure reinsurance data protection policy — define treaty data ML-KEM tier and SWIFT integrity verification parameters', 'Required'),
            ('Compliance', 'Configure compliance evidence vault — set retention periods per regulatory requirement (7 years minimum US, 10 years EU)', 'Critical'),
        ],
        'Policy Configuration Complete',
        [
            'PQC encryption tiers configured by insurance data type',
            'Solvency II, NY DFS 500, and NAIC policies configured',
            'HIPAA Security Rule controls configured for health insurance lines',
            'Fraud detection thresholds set and reviewed with SIU team',
        ],
        styles)

    story.append(PageBreak())

    # ─── Phase 4: Policyholder Data Protection ────────────────────────────────
    phase_block(story, 4, 'Policyholder Data Protection', 'Days 8-12', 'CISO + Data Privacy + Actuarial',
        [
            ('Security', 'Activate PQC wrapping for all life and whole life policy data in transit — apply ML-KEM-1024 to all in-scope protocol streams', 'Critical'),
            ('Security', 'Activate PQC wrapping for annuity and variable annuity policy data — include surrender value and beneficiary data flows', 'Critical'),
            ('Security', 'Activate PQC wrapping for long-term care insurance data — include care assessment, provider billing (837I), and eligibility flows', 'Critical'),
            ('Security', 'Activate PQC wrapping for disability income policy data — include earnings records and medical certification flows', 'Required'),
            ('Security', 'Activate PQC wrapping for group life and group health data — include X12 834 enrollment and 837 claims flows', 'Required'),
            ('Data', 'Classify all PII in active policy data flows — verify all Name, SSN, DOB, address, and beneficiary data is in-scope for PQC', 'Critical'),
            ('Data', 'Classify all PHI in health insurance data flows — verify HIPAA minimum necessary enforcement is active', 'Critical'),
            ('Compliance', 'Activate consent record integrity verification — apply ML-DSA signatures to all policyholder consent records', 'Required'),
            ('Data', 'Enable GDPR Article 32 data protection controls for EU policyholder data — confirm cryptographic key isolation by jurisdiction', 'Required'),
            ('Data', 'Enable CCPA/CPRA controls for California policyholder data — confirm right-to-erasure key deletion workflow', 'Required'),
            ('Security', 'Verify per-policy key provenance logging is active — confirm retention period aligns to policy duration', 'Critical'),
            ('Actuarial', 'Activate ML-DSA integrity signing for actuarial model data flows — include mortality tables, lapse assumptions, CAT model outputs', 'Critical'),
        ],
        'Policyholder Data Protection Active',
        [
            'ML-KEM-1024 active on all life, annuity, LTC, and actuarial data flows',
            'PHI and PII classification complete and in-scope for PQC wrapping',
            'GDPR/CCPA key isolation and consent integrity controls active',
            'Per-policy key provenance logging confirmed operational',
        ],
        styles)

    # ─── Phase 5: EDI & Protocol Security ─────────────────────────────────────
    phase_block(story, 5, 'EDI & Protocol Security', 'Days 10-14', 'CISO + Integration Engineering',
        [
            ('Security', 'Activate ACORD XML PQC wrapping for all inbound agent and MGA submissions — include ACORD AL3, 103, 165 message types', 'Critical'),
            ('Security', 'Activate ACORD XML PQC wrapping for all outbound reinsurance bordereau and data exchange flows', 'Critical'),
            ('Security', 'Activate ML-DSA signature verification for ACORD XML — replace SHA-1/MD5 XML Digital Signatures on legacy integration points', 'Required'),
            ('Security', 'Activate X12 EDI 834 PQC wrapping for all enrollment/disenrollment flows — apply ML-KEM to all member PII', 'Required'),
            ('Security', 'Activate X12 EDI 835 PQC wrapping for all remittance advice flows — enable payment redirection fraud detection', 'Required'),
            ('Security', 'Activate X12 EDI 837P/837I/837D PQC wrapping for all claims submissions — enable real-time fraud detection', 'Critical'),
            ('Security', 'Activate HL7 v2.x and FHIR R4 PQC wrapping for health insurance data exchanges — enable PHI boundary detection', 'Required'),
            ('Security', 'Activate ISO 20022 PQC wrapping for premium payment message flows — protect pacs.008 and pacs.009 messages', 'Required'),
            ('Technical', 'Notify all EDI trading partners of PQC session wrapping deployment — provide hybrid certificate package to PQC-capable partners', 'Required'),
            ('Security', 'Validate PQC wrapping with each major EDI trading partner type: clearinghouses, TPAs, reinsurance portals', 'Required'),
            ('Fraud', 'Verify fraud detection is active on all 837 claim submission streams — confirm SIU escalation workflow integration', 'Critical'),
            ('Compliance', 'Generate first EDI security compliance report — document all trading partner encryption status for NY DFS 500 evidence', 'Required'),
        ],
        'EDI & Protocol Security Active',
        [
            'ACORD XML PQC wrapping active for all agent and reinsurance flows',
            'X12 EDI 834/835/837 PQC wrapping active for all trading partners',
            'HL7 and ISO 20022 PQC wrapping active',
            'Real-time fraud detection confirmed active on 837 claim streams',
        ],
        styles)

    story.append(PageBreak())

    # ─── Phase 6: Mainframe Policy System Shield ───────────────────────────────
    phase_block(story, 6, 'Mainframe Policy System Shield', 'Days 12-16', 'CISO + Mainframe Team',
        [
            ('Technical', 'Position Bridge transparent proxy in front of mainframe network interface — confirm zero COBOL code changes required', 'Critical'),
            ('Security', 'Activate TN3270e PQC session wrapping — apply ML-KEM to all terminal emulation sessions between mainframe and connected clients', 'Critical'),
            ('Security', 'Activate SNA/APPC session encryption for any SNA-based mainframe communication still in use', 'Required'),
            ('Security', 'Activate CICS transaction flow PQC wrapping for policy enquiry and update transactions', 'Required'),
            ('Security', 'Activate IBM MQ policy messaging encryption — protect all message queue traffic between mainframe and distributed systems', 'Required'),
            ('Security', 'Activate JES job entry stream monitoring — detect bulk policy data exports from batch job submissions', 'Required'),
            ('Security', 'Configure lateral movement prevention — enforce microsegmentation on all mainframe network segment traffic', 'Critical'),
            ('Security', 'Activate privileged session monitoring — enable AI detection for off-hours access, bulk exports, privilege escalation on mainframe sessions', 'Critical'),
            ('Technical', 'Verify TN3270e session latency overhead is under 0.8ms — confirm no SLA impact on policy processing', 'Required'),
            ('Technical', 'Verify mainframe transaction throughput is unaffected — run benchmark against pre-deployment baseline', 'Critical'),
            ('Operational', 'Confirm mainframe operations team has Bridge bypass procedure for emergency use — document and restrict access', 'Required'),
            ('Compliance', 'Document mainframe PQC deployment for Solvency II ORSA and NY DFS 500 Section 500.15 evidence', 'Required'),
        ],
        'Mainframe Shield Active',
        [
            'TN3270e PQC wrapping active with confirmed latency under 0.8ms',
            'Mainframe throughput benchmark confirmed — no performance degradation',
            'Lateral movement prevention and privileged session monitoring active',
            'Zero COBOL changes made — confirmed by mainframe team',
        ],
        styles)

    # ─── Phase 7: Fraud Detection Integration ─────────────────────────────────
    phase_block(story, 7, 'Fraud Detection Integration', 'Days 14-18', 'CISO + SIU + Claims Operations',
        [
            ('Fraud', 'Activate synthetic identity detection on all X12 837 claim submission streams — calibrate to carrier-specific SSN/DOB/name baseline', 'Critical'),
            ('Fraud', 'Activate claims network analysis — configure multi-party fraud ring detection parameters with SIU team input', 'Critical'),
            ('Fraud', 'Activate medical billing integrity detection — configure CPT/ICD upcoding thresholds by specialty and provider type', 'Required'),
            ('Fraud', 'Activate duplicate claim detection — configure cross-carrier claim fingerprint hashing and deduplication threshold', 'Required'),
            ('Fraud', 'Activate policy application fraud detection — configure ACORD XML new business anomaly detection for synthetic applications', 'Required'),
            ('Claims', 'Integrate fraud detection alerts with claims management system (Guidewire ClaimCenter, Duck Creek Claims, or equivalent)', 'Critical'),
            ('Claims', 'Configure SIU team escalation workflow — define alert priority levels and automated claim hold procedures', 'Critical'),
            ('Reinsurance', 'Activate SWIFT message integrity verification — configure settlement amount tolerance and routing anomaly thresholds', 'Critical'),
            ('Reinsurance', 'Activate bordereau validation — configure expected loss range parameters per treaty type for anomaly detection', 'Required'),
            ('Reinsurance', 'Activate wire fraud prevention for reinsurance SWIFT payments — configure BEC indicator detection and settlement hold workflow', 'Critical'),
            ('Fraud', 'Run initial fraud detection calibration — review first 48 hours of detection output with SIU team and tune thresholds', 'Required'),
            ('Fraud', 'Confirm false positive rate is acceptable — target less than 2% false positive rate on clean claim populations', 'Required'),
        ],
        'Fraud Detection Active',
        [
            'Synthetic identity and claims ring detection active and calibrated',
            'Medical billing integrity detection tuned to carrier specialty mix',
            'SIU escalation workflow confirmed end-to-end',
            'SWIFT integrity verification and wire fraud prevention active',
        ],
        styles)

    story.append(PageBreak())

    # ─── Phase 8: Compliance Validation ───────────────────────────────────────
    phase_block(story, 8, 'Compliance Validation', 'Days 18-24', 'CRO + Compliance + Legal',
        [
            ('Compliance', 'Generate Solvency II SFCR cybersecurity evidence package — include EIOPA ICT Guidelines control coverage report', 'Critical'),
            ('Compliance', 'Generate Solvency II ORSA cybersecurity control documentation — include quantitative risk reduction evidence', 'Required'),
            ('Compliance', 'Generate NY DFS 500 Section 500.15 encryption compliance evidence — document all NPI-in-transit encryption coverage', 'Critical'),
            ('Compliance', 'Generate NY DFS 500 Section 500.17 incident response capability evidence — document autonomous detection and notification workflows', 'Required'),
            ('Compliance', 'Generate NAIC Model Law cybersecurity program documentation — format for state insurance department examination', 'Required'),
            ('Compliance', 'Generate HIPAA Security Rule compliance evidence — include 164.312(a)(2)(iv) and 164.312(e)(1) encryption documentation', 'Required'),
            ('Compliance', 'Generate GDPR Article 32 compliance evidence — document encryption adequacy for EU policyholder data and DPA notification readiness', 'Required'),
            ('Compliance', 'Generate CCPA/CPRA compliance evidence — document California consumer data protection controls and right-to-erasure capability', 'Required'),
            ('Compliance', 'Generate PCI-DSS v4.0 Requirement 3 and 4 evidence for premium payment card data flows', 'Recommended'),
            ('Compliance', 'Generate IFRS 17 data integrity evidence — document ML-DSA integrity verification for actuarial data used in contract liability measurement', 'Recommended'),
            ('Compliance', 'Review all compliance evidence packages with regulatory counsel — confirm readiness for examination', 'Critical'),
            ('Compliance', 'Submit proactive regulatory communication to NY DFS / EIOPA supervisor if required by regulatory relationship', 'Optional'),
        ],
        'Compliance Validation Complete',
        [
            'Solvency II SFCR and ORSA evidence packages reviewed and approved by Compliance',
            'NY DFS 500 evidence package reviewed and approved by legal counsel',
            'NAIC, HIPAA, GDPR/CCPA evidence packages complete',
            'All evidence packages stored in Bridge compliance vault with integrity signing',
        ],
        styles)

    # ─── Phase 9: UAT & Performance Testing ───────────────────────────────────
    phase_block(story, 9, 'UAT & Performance Testing', 'Days 22-28', 'CISO + IT + Operations + Actuarial',
        [
            ('Technical', 'Run end-to-end claims processing test: submit synthetic X12 837 test claims through full workflow and verify latency under SLA', 'Critical'),
            ('Technical', 'Run end-to-end policy issuance test: submit ACORD XML test application and verify PQC wrapping active throughout lifecycle', 'Critical'),
            ('Technical', 'Run mainframe policy query test: verify TN3270e sessions function normally with PQC wrapping active and under 0.8ms overhead', 'Critical'),
            ('Technical', 'Run EDI trading partner integration test: verify each major trading partner can exchange data with Bridge wrapping active', 'Required'),
            ('Technical', 'Run SWIFT reinsurance message test: verify ML-KEM wrapping and integrity verification on test settlement messages', 'Required'),
            ('Fraud', 'Run fraud detection test: inject known synthetic identity and upcoding test cases and verify detection and SIU escalation', 'Critical'),
            ('Actuarial', 'Run actuarial model data flow test: verify ML-DSA integrity signing is active on actuarial model outputs and detectable on consumption', 'Required'),
            ('Security', 'Run HSM failover test: verify key operations continue seamlessly during HSM HA failover', 'Required'),
            ('Security', 'Run Bridge HA failover test: verify policy transactions continue with zero disruption during Bridge appliance failover', 'Required'),
            ('Compliance', 'Validate compliance evidence generation: generate test evidence packages for each regulatory framework and verify completeness', 'Required'),
            ('Technical', 'Run 24-hour production load test: verify performance metrics under realistic insurance transaction volumes', 'Required'),
            ('Operational', 'Conduct tabletop incident response exercise: test Bridge-assisted detection-to-response workflow with SOC team', 'Recommended'),
        ],
        'UAT & Performance Testing Complete',
        [
            'End-to-end claims, policy, and mainframe workflows tested and confirmed under SLA',
            'Fraud detection test cases passed — SIU escalation workflow confirmed',
            'HSM and Bridge HA failover tests passed',
            '24-hour load test completed with no performance issues',
        ],
        styles)

    story.append(PageBreak())

    # ─── Phase 10: Go-Live & Handover ─────────────────────────────────────────
    phase_block(story, 10, 'Go-Live & Handover', 'Days 28-30', 'CISO + CRO + Operations',
        [
            ('Operational', 'Execute go-live sign-off: CRO and CISO formal sign-off on production Bridge deployment', 'Critical'),
            ('Security', 'Switch Bridge from monitoring-only to active inline enforcement mode on all configured protocol streams', 'Critical'),
            ('Operational', 'Confirm all trading partner EDI connections are operational with PQC wrapping active', 'Critical'),
            ('Security', 'Confirm mainframe TN3270e Shield is active and monitoring in production — verify zero performance impact', 'Critical'),
            ('Fraud', 'Confirm fraud detection is active in production mode — verify SIU team is monitoring alerts', 'Critical'),
            ('Reinsurance', 'Confirm SWIFT integrity verification is active for all reinsurance settlement channels', 'Critical'),
            ('Technical', 'Activate 24/7 QBITEL SOC monitoring for production insurance environment', 'Required'),
            ('Compliance', 'Generate Day 1 production compliance snapshot — first evidence package from production Bridge deployment', 'Required'),
            ('Operational', 'Brief insurance operations team on Bridge monitoring dashboards and alert escalation procedures', 'Required'),
            ('Actuarial', 'Brief actuarial team on model data integrity monitoring — confirm alerts for integrity verification failures', 'Required'),
            ('Operational', 'Document runbook for Bridge operations — include incident response, trading partner on-boarding, and algorithm update procedures', 'Required'),
            ('Compliance', 'Schedule first 90-day compliance review — plan quarterly compliance evidence package generation cadence', 'Required'),
            ('Executive', 'Provide CRO/Board quantum security posture report — document quantum exposure eliminated, compliance status, and fraud detection baseline', 'Required'),
            ('Operational', 'Assign Named Customer Success Manager from QBITEL — schedule 30-day check-in and quarterly business reviews', 'Required'),
        ],
        'Go-Live Complete — QBITEL Bridge Production Active',
        [
            'CRO and CISO sign-off confirmed — Bridge in production enforcement mode',
            'All protocol streams PQC-protected: ACORD, X12 EDI, HL7, SWIFT, TN3270e',
            'Fraud detection, compliance automation, and SWIFT integrity verification active',
            'Day 1 compliance evidence package generated and archived',
            'Insurance operations and actuarial teams briefed — runbook distributed',
        ],
        styles)

    # Summary table
    summary_data = [
        ['Phase', 'Title', 'Duration', 'Owner', 'Key Outcome'],
        ['Phase 0', 'Pre-Engagement', 'Week -2 to -1', 'CRO + CISO', 'Scope, regulatory mapping, data classification'],
        ['Phase 1', 'Protocol Discovery', 'Days 1-3', 'CISO + Network', 'Full insurance protocol inventory'],
        ['Phase 2', 'Infrastructure', 'Days 3-7', 'CISO + Infra', 'HSM, HA, network segmentation'],
        ['Phase 3', 'Policy Config', 'Days 6-9', 'CISO + Compliance', 'PQC tiers, Solvency II/NY DFS 500 policies'],
        ['Phase 4', 'Policyholder Data', 'Days 8-12', 'CISO + Data Privacy', 'Life/annuity/LTC/health data PQC active'],
        ['Phase 5', 'EDI & Protocol', 'Days 10-14', 'CISO + Integration', 'ACORD/X12/HL7/ISO 20022 protection'],
        ['Phase 6', 'Mainframe Shield', 'Days 12-16', 'CISO + Mainframe', 'TN3270e PQC, zero COBOL changes'],
        ['Phase 7', 'Fraud Detection', 'Days 14-18', 'CISO + SIU', 'Claims fraud, SWIFT wire fraud prevention'],
        ['Phase 8', 'Compliance', 'Days 18-24', 'CRO + Compliance', 'Solvency II/NY DFS 500/NAIC evidence'],
        ['Phase 9', 'UAT & Testing', 'Days 22-28', 'CISO + IT + Ops', 'End-to-end validation, load testing'],
        ['Phase 10', 'Go-Live', 'Days 28-30', 'CISO + CRO', 'Production enforcement active'],
    ]
    tbl = Table(summary_data,
                colWidths=[CONTENT_W*0.10, CONTENT_W*0.18, CONTENT_W*0.16,
                           CONTENT_W*0.16, CONTENT_W*0.40])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE_C),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 7.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#CCDDEE')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LINEBELOW', (0, 0), (-1, 0), 2, GOLD),
    ]))
    story.append(Paragraph('Deployment Summary',
                           ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=12,
                                          textColor=NAVY, spaceAfter=8, spaceBefore=10)))
    story.append(tbl)
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        'QBITEL Enterprise Insurance Practice  |  enterprise@qbitel.com  |  bridge.qbitel.com',
        ParagraphStyle('ctr', fontName='Helvetica-Bold', fontSize=10,
                       textColor=NAVY, alignment=TA_CENTER)))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        'Confidential — For Authorized Recipients Only  |  Copyright 2026 QBITEL. All Rights Reserved.',
        ParagraphStyle('ctr2', fontName='Helvetica', fontSize=8,
                       textColor=MID_GREY, alignment=TA_CENTER)))

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
                       'QBITEL_Insurance_Deployment_Checklist.pdf')
    build_doc(out)
