"""
Build QBITEL Aviation Deployment Checklist PDF
11 phases (0-10) covering the complete QBITEL Bridge aviation deployment lifecycle
plus appendices for performance SLAs, rollback procedures, and escalation.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (Paragraph, Spacer, Table, TableStyle,
                                 PageBreak, KeepTogether)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, NextPageTemplate

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
GREEN_OK  = HexColor('#2D7D46')
AMBER     = HexColor('#D97706')

PAGE_W, PAGE_H = letter
MARGIN    = 0.85 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


class PhaseHeader(Flowable):
    """Full-width phase header with phase number, title, duration, and responsible party."""
    def __init__(self, phase_num, title, duration, owner, color=NAVY):
        super().__init__()
        self.phase_num = phase_num
        self.title     = title
        self.duration  = duration
        self.owner     = owner
        self.color     = color
        self.height    = 50
    def wrap(self, avw, avh):
        self._w = avw
        return avw, self.height
    def draw(self):
        c = self.canv
        w, h = self._w, self.height
        # Background
        c.setFillColor(self.color)
        c.rect(0, 0, w, h, stroke=0, fill=1)
        # Gold left accent
        c.setFillColor(GOLD)
        c.rect(0, 0, 6, h, stroke=0, fill=1)
        # Phase number badge
        c.setFillColor(GOLD)
        c.roundRect(12, 10, 52, 30, 5, stroke=0, fill=1)
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 8)
        c.drawCentredString(38, h - 20, 'PHASE')
        c.setFont('Helvetica-Bold', 14)
        c.drawCentredString(38, h - 34, str(self.phase_num))
        # Title
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 11)
        c.drawString(76, h - 20, self.title)
        # Duration and owner
        c.setFillColor(TEAL)
        c.setFont('Helvetica', 8)
        c.drawString(76, h - 34, f'Duration: {self.duration}   |   Owner: {self.owner}')


class CheckItem(Flowable):
    """A single checklist item with checkbox, item text, and optional sub-detail."""
    def __init__(self, number, text, detail='', priority='REQUIRED', category=''):
        super().__init__()
        self.number   = number
        self.text     = text
        self.detail   = detail
        self.priority = priority
        self.category = category
        self.height   = 28 if not detail else 42
    def wrap(self, avw, avh):
        self._w = avw
        return avw, self.height
    def draw(self):
        c = self.canv
        w, h = self._w, self.height
        # Background — alternating
        bg = WHITE_C if self.number % 2 == 0 else LIGHT_BG
        c.setFillColor(bg)
        c.rect(0, 0, w, h, stroke=0, fill=1)
        # Checkbox
        c.setStrokeColor(TEAL_DARK)
        c.setLineWidth(1.2)
        c.rect(8, h - 20, 14, 14, stroke=1, fill=0)
        # Item number
        c.setFillColor(MID_GREY)
        c.setFont('Helvetica', 7.5)
        c.drawCentredString(15, h - 29, str(self.number))
        # Priority badge
        pcolor = NAVY if self.priority == 'REQUIRED' else (AMBER if self.priority == 'RECOMMENDED' else TEAL)
        c.setFillColor(pcolor)
        badge_text = self.priority[:3]
        c.roundRect(26, h - 20, 24, 12, 2, stroke=0, fill=1)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 6)
        c.drawCentredString(38, h - 13, badge_text)
        # Category tag
        if self.category:
            c.setFillColor(TABLE_ALT)
            cat_w = c.stringWidth(self.category, 'Helvetica', 6.5) + 8
            c.roundRect(54, h - 20, cat_w, 12, 2, stroke=0, fill=1)
            c.setFillColor(TEAL_DARK)
            c.setFont('Helvetica', 6.5)
            c.drawString(58, h - 13, self.category)
            text_x = 58 + cat_w
        else:
            text_x = 56
        # Main text
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 9)
        txt = self.text
        if c.stringWidth(txt, 'Helvetica', 9) > w - text_x - 6:
            while c.stringWidth(txt + '...', 'Helvetica', 9) > w - text_x - 6 and txt:
                txt = txt[:-1]
            txt = txt + '...'
        c.drawString(text_x, h - 14, txt)
        # Detail text
        if self.detail:
            c.setFillColor(MID_GREY)
            c.setFont('Helvetica-Oblique', 7.5)
            det = self.detail
            if c.stringWidth(det, 'Helvetica-Oblique', 7.5) > w - text_x - 6:
                while c.stringWidth(det + '...', 'Helvetica-Oblique', 7.5) > w - text_x - 6 and det:
                    det = det[:-1]
                det = det + '...'
            c.drawString(text_x, h - 28, det)
        # Bottom border
        c.setStrokeColor(TABLE_ALT)
        c.setLineWidth(0.4)
        c.line(0, 0, w, 0)


def chk(num, text, detail='', priority='REQUIRED', category=''):
    return KeepTogether([CheckItem(num, text, detail, priority, category), Spacer(1, 2)])


def draw_cover(canvas, doc):
    canvas.saveState()
    w, h = PAGE_W, PAGE_H
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, h, stroke=0, fill=1)
    # Teal accent
    canvas.setFillColor(TEAL_DARK)
    p = canvas.beginPath()
    p.moveTo(0, h * 0.45)
    p.lineTo(w, h * 0.55)
    p.lineTo(w, h * 0.50)
    p.lineTo(0, h * 0.40)
    p.close()
    canvas.drawPath(p, stroke=0, fill=1)
    # Gold top strip
    canvas.setFillColor(GOLD)
    canvas.rect(0, h - 10, w, 10, stroke=0, fill=1)
    canvas.setFillColor(TEAL)
    canvas.rect(0, h - 18, w, 8, stroke=0, fill=1)
    # Wordmark
    canvas.setFillColor(GOLD)
    canvas.setFont('Helvetica-Bold', 26)
    canvas.drawString(MARGIN, h - 78, 'QBITEL BRIDGE')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica-Bold', 11)
    canvas.drawString(MARGIN, h - 98, 'AVIATION & AEROSPACE — DEPLOYMENT CHECKLIST')
    canvas.setStrokeColor(GOLD)
    canvas.setLineWidth(2)
    canvas.line(MARGIN, h - 108, w - MARGIN, h - 108)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 19)
    canvas.drawString(MARGIN, h - 148, '11-Phase Implementation Guide')
    canvas.setFillColor(TABLE_ALT)
    canvas.setFont('Helvetica', 10)
    phases = [
        'Phase 0: Pre-Engagement Safety Assessment',
        'Phase 1: Protocol Discovery and Analysis',
        'Phase 2: Ground Infrastructure Readiness',
        'Phase 3: Security Architecture Design',
        'Phase 4: ADS-B Authentication Deployment',
        'Phase 5: Data Link Protection (LDACS/ACARS/SATCOM)',
        'Phase 6: AI Spoofing Detection Activation',
        'Phase 7: ATC System Integration',
        'Phase 8: DO-326A Evidence Generation',
        'Phase 9: Validation and Testing',
        'Phase 10: Operational Handover',
        'Appendix: Performance SLAs, Rollback, Escalation',
    ]
    y_s = h - 175
    for ph in phases:
        canvas.drawString(MARGIN + 15, y_s, ph)
        y_s -= 17
    # Stats row
    stats = [('180K+', 'Daily Unauthenticated ADS-B'),
             ('36 Wks', 'Full Deployment Timeline'),
             ('99.999%', 'Availability SLA Target'),
             ('Zero', 'Aircraft Changes Required')]
    box_w = (w - 2 * MARGIN - 9) / 4
    y_box = MARGIN + 50
    for i, (val, lbl) in enumerate(stats):
        x = MARGIN + i * (box_w + 3)
        bg = LIGHT_NAVY if i % 2 == 0 else TEAL_DARK
        canvas.setFillColor(bg)
        canvas.roundRect(x, y_box, box_w, 60, 4, stroke=0, fill=1)
        canvas.setFillColor(GOLD)
        canvas.setFont('Helvetica-Bold', 16)
        canvas.drawCentredString(x + box_w / 2, y_box + 34, val)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 7)
        words = lbl.split()
        line1 = ' '.join(words[:2])
        line2 = ' '.join(words[2:]) if len(words) > 2 else ''
        canvas.drawCentredString(x + box_w / 2, y_box + 19, line1)
        if line2:
            canvas.drawCentredString(x + box_w / 2, y_box + 8, line2)
    # Footer
    canvas.setFillColor(TEAL_DARK)
    canvas.rect(0, 0, w, MARGIN + 42, stroke=0, fill=1)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 9)
    canvas.drawString(MARGIN, 22, 'enterprise@qbitel.com   |   https://bridge.qbitel.com')
    canvas.restoreState()


def draw_page(canvas, doc):
    canvas.saveState()
    w, h = PAGE_W, PAGE_H
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 0.45 * inch, w, 0.45 * inch, stroke=0, fill=1)
    canvas.setFillColor(GOLD)
    canvas.rect(0, h - 0.48 * inch, w, 3, stroke=0, fill=1)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(MARGIN, h - 0.3 * inch, 'QBITEL BRIDGE')
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(TEAL)
    canvas.drawString(MARGIN + 1.1 * inch, h - 0.3 * inch, '— Aviation Deployment Checklist')
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(w - MARGIN, h - 0.3 * inch, f'Page {doc.page}')
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, w, 0.55 * inch, stroke=0, fill=1)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 7.5)
    canvas.drawCentredString(w / 2, 0.32 * inch, 'QBITEL BRIDGE — AVIATION & AEROSPACE DEPLOYMENT CHECKLIST')
    canvas.setFont('Helvetica', 7)
    canvas.drawString(MARGIN, 0.16 * inch, 'enterprise@qbitel.com  |  https://bridge.qbitel.com')
    canvas.drawRightString(w - MARGIN, 0.16 * inch, 'Confidential — Not for Public Distribution')
    canvas.restoreState()


def make_styles():
    styles = {}
    styles['body'] = ParagraphStyle('body',
        fontName='Helvetica', fontSize=9, leading=14,
        textColor=DARK_TEXT, spaceAfter=4)
    styles['note'] = ParagraphStyle('note',
        fontName='Helvetica-Oblique', fontSize=8, leading=12,
        textColor=MID_GREY, spaceAfter=4, leftIndent=10)
    styles['h3'] = ParagraphStyle('h3',
        fontName='Helvetica-Bold', fontSize=10, leading=14,
        textColor=TEAL_DARK, spaceBefore=8, spaceAfter=4)
    styles['cat'] = ParagraphStyle('cat',
        fontName='Helvetica-Bold', fontSize=8, leading=11,
        textColor=NAVY, spaceAfter=2, leftIndent=8)
    return styles


def sp(n=6):
    return Spacer(1, n)


def section_divider(story, label):
    story.append(sp(4))
    story.append(Paragraph(label, ParagraphStyle('div',
        fontName='Helvetica-Bold', fontSize=8, textColor=TEAL_DARK,
        spaceBefore=6, spaceAfter=3, leftIndent=0)))


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

    # ─── HOW TO USE ──────────────────────────────────────────────────────────
    story.append(Paragraph('How to Use This Checklist', ParagraphStyle('ht',
        fontName='Helvetica-Bold', fontSize=12, textColor=NAVY, spaceAfter=6)))
    legend_data = [
        ['Priority', 'Badge', 'Meaning'],
        ['REQUIRED', 'REQ', 'Must be completed before proceeding to next phase'],
        ['RECOMMENDED', 'REC', 'Strongly recommended — omission requires documented justification'],
        ['OPTIONAL', 'OPT', 'Optional enhancement — complete if applicable to deployment'],
    ]
    lt = Table(legend_data, colWidths=[CONTENT_W * 0.22, CONTENT_W * 0.10, CONTENT_W * 0.68])
    lt.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR',   (0, 0), (-1, 0), WHITE_C),
        ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',    (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',    (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, LIGHT_BG]),
        ('GRID',        (0, 0), (-1, -1), 0.3, MID_GREY),
        ('TOPPADDING',  (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(lt)
    story.append(sp(10))

    # ─── PHASE 0: Pre-Engagement ──────────────────────────────────────────────
    story.append(PhaseHeader(0, 'PRE-ENGAGEMENT SAFETY ASSESSMENT',
        '4 weeks', 'QBITEL Aviation Practice Lead + Customer Safety Team', NAVY))
    story.append(sp(6))

    section_divider(story, 'Safety Team Briefing')
    story.append(chk(1, 'Conduct QBITEL aviation security briefing for customer safety team',
        'Include SMS coordinator, chief engineer, and airworthiness authority liaison', 'REQUIRED', 'SAFETY'))
    story.append(chk(2, 'Review existing safety case and safety management system documentation',
        'Identify existing security-relevant assumptions in safety case', 'REQUIRED', 'SAFETY'))
    story.append(chk(3, 'Confirm QBITEL deployment has zero impact on existing safety cases',
        'Ground-only: no impact. Airborne: ARINC 653 isolation assessment required', 'REQUIRED', 'SAFETY'))
    story.append(chk(4, 'Document safety team acceptance sign-off for deployment scope',
        'Formal acceptance required before any infrastructure work begins', 'REQUIRED', 'SAFETY'))

    section_divider(story, 'DO-326A Gap Assessment')
    story.append(chk(5, 'Initiate DO-326A Aircraft Security Log (ASL) with baseline security posture',
        'ASL must capture security assumptions at deployment start for traceability', 'REQUIRED', 'DO-326A'))
    story.append(chk(6, 'Conduct DO-326A gap analysis against existing security documentation',
        'Compare current state against DO-326A requirements and identify gaps', 'REQUIRED', 'DO-326A'))
    story.append(chk(7, 'Define security requirements derived from threat analysis (Security Target Analysis)',
        'STA must cover ADS-B, ACARS, SATCOM, ATC network, and any airborne elements', 'REQUIRED', 'DO-326A'))
    story.append(chk(8, 'Confirm scope: ground-only vs. ground + airborne partition deployment',
        'Airborne deployment triggers DO-178C scope and STC/CS-STAN assessment', 'REQUIRED', 'SCOPE'))

    section_divider(story, 'Airspace Authority Notification')
    story.append(chk(9, 'Notify national aviation authority (FAA/EASA/CAA) of planned security deployment',
        'Use AC 119-1 (FAA) or AMC 20-42 (EASA) notification procedure', 'REQUIRED', 'REGULATORY'))
    story.append(chk(10, 'Notify ANSP of planned ground receiver infrastructure changes',
        'ADS-B receiver modifications require ANSP coordination', 'REQUIRED', 'REGULATORY'))
    story.append(chk(11, 'Confirm no NOTAM requirements for ground infrastructure deployment',
        'Ground-only deployment typically does not require NOTAMs', 'RECOMMENDED', 'REGULATORY'))
    story.append(chk(12, 'Initiate FAA Issue Paper or EASA CRI if airborne deployment is in scope',
        'Early engagement with authority is critical for airborne certification timeline', 'REQUIRED', 'REGULATORY'))

    section_divider(story, 'Preliminary Technical Assessment')
    story.append(chk(13, 'Conduct ADS-B receiver network survey — inventory all receiver sites',
        'Document receiver location, hardware, firmware, and network connectivity', 'REQUIRED', 'TECHNICAL'))
    story.append(chk(14, 'Measure ADS-B message volumes at representative receiver sites',
        'Required for authentication node sizing and throughput planning', 'REQUIRED', 'TECHNICAL'))
    story.append(chk(15, 'Assess existing ATC network topology for QBITEL integration points',
        'Document ASTERIX feeds, SDPS connections, SWIM interfaces', 'REQUIRED', 'TECHNICAL'))

    # ─── PHASE 1: Protocol Discovery ──────────────────────────────────────────
    story.append(PageBreak())
    story.append(PhaseHeader(1, 'PROTOCOL DISCOVERY AND ANALYSIS',
        '4 weeks', 'QBITEL Technical Team + ANSP Engineering', LIGHT_NAVY))
    story.append(sp(6))

    section_divider(story, 'ADS-B Receiver Analysis')
    story.append(chk(16, 'Deploy passive ADS-B receiver tap at representative sites (minimum 5)',
        'Record and analyze 72 hours of ADS-B traffic for baseline characterization', 'REQUIRED', 'ADS-B'))
    story.append(chk(17, 'Characterize ADS-B message volume, message rate, and ICAO address distribution',
        'Required for AI model training dataset and authentication enrollment planning', 'REQUIRED', 'ADS-B'))
    story.append(chk(18, 'Identify existing MLAT receiver infrastructure and integration capability',
        'MLAT integration significantly enhances spoofing detection accuracy', 'RECOMMENDED', 'ADS-B'))
    story.append(chk(19, 'Analyze ADS-B message format distribution (ADS-B OUT types 0-31)',
        'Ensure QBITEL authentication handles all message type formats in use', 'REQUIRED', 'ADS-B'))
    story.append(chk(20, 'Identify any non-standard or military ADS-B traffic in airspace',
        'Military and government aircraft may require special handling configuration', 'RECOMMENDED', 'ADS-B'))

    section_divider(story, 'ACARS and Data Link Analysis')
    story.append(chk(21, 'Inventory all ACARS VHF ground stations in scope',
        'Document frequency, ground station hardware, and network connectivity', 'REQUIRED', 'ACARS'))
    story.append(chk(22, 'Monitor and catalog ACARS message types and volumes for 72 hours',
        'Required for ACARS gateway sizing and authentication overhead assessment', 'REQUIRED', 'ACARS'))
    story.append(chk(23, 'Assess HF ACARS infrastructure if oceanic routes are in scope',
        'HF ACARS requires separate gateway configuration for low-bandwidth constraints', 'RECOMMENDED', 'ACARS'))
    story.append(chk(24, 'Survey SATCOM ground earth stations (Iridium, Inmarsat) if in scope',
        'SATCOM security requires gateway deployment at each ground earth station', 'RECOMMENDED', 'SATCOM'))

    section_divider(story, 'ARINC 429 Bus Analysis (Airborne Deployments Only)')
    story.append(chk(25, 'Map ARINC 429 bus topology for target aircraft type(s)',
        'Required for ARINC 653 security partition I/O specification', 'REQUIRED', 'AIRBORNE'))
    story.append(chk(26, 'Identify transponder ARINC 429 output bus for TAU installation',
        'TAU connects as passive tap on transponder output bus', 'REQUIRED', 'AIRBORNE'))
    story.append(chk(27, 'Review IMA platform ARINC 653 partition allocation and available time budget',
        'Security partition time budget must not affect safety-critical partition guarantees', 'REQUIRED', 'AIRBORNE'))

    # ─── PHASE 2: Ground Infrastructure Readiness ────────────────────────────
    story.append(PageBreak())
    story.append(PhaseHeader(2, 'GROUND INFRASTRUCTURE READINESS',
        '4 weeks', 'QBITEL Infrastructure Team + Customer IT/Network Team', TEAL_DARK))
    story.append(sp(6))

    section_divider(story, 'ATC Network Segmentation')
    story.append(chk(28, 'Assess ATC network architecture for QBITEL node placement',
        'Authentication nodes placed between receiver feeds and SDPS/display systems', 'REQUIRED', 'NETWORK'))
    story.append(chk(29, 'Implement network segmentation separating authentication infrastructure from ATC safety systems',
        'QBITEL nodes must be network-isolated from safety-critical ATC automation', 'REQUIRED', 'NETWORK'))
    story.append(chk(30, 'Deploy N+1 redundant QBITEL authentication nodes at each primary ATC facility',
        'Minimum 2 nodes per facility; 3 for major Area Control Centres', 'REQUIRED', 'NETWORK'))
    story.append(chk(31, 'Establish out-of-band management network for QBITEL infrastructure',
        'Management access must not share paths with operational data flows', 'REQUIRED', 'NETWORK'))
    story.append(chk(32, 'Deploy quantum-safe VPN between ATC facilities (QBITEL Bridge QS-VPN)',
        'PQC-secured VPN mesh for inter-facility coordination and QBITEL management', 'REQUIRED', 'NETWORK'))

    section_divider(story, 'Hardware Security Module (HSM) Deployment')
    story.append(chk(33, 'Procure FIPS 140-3 Level 3 HSMs for root key storage',
        'Minimum 2 HSMs per deployment (primary + backup) in geographically separate facilities', 'REQUIRED', 'PKI'))
    story.append(chk(34, 'Initialize HSMs and generate root key material under dual-control ceremony',
        'Key ceremony must be documented for DO-326A ASL. Witnesses required.', 'REQUIRED', 'PKI'))
    story.append(chk(35, 'Configure HSM for QBITEL authentication workloads — Falcon-512 and ML-DSA acceleration',
        'Verify HSM performance meets QBITEL throughput requirements under load', 'REQUIRED', 'PKI'))

    section_divider(story, 'Certificate Authority Infrastructure')
    story.append(chk(36, 'Deploy QBITEL Aviation PKI root CA (offline HSM-backed)',
        'Root CA must remain offline except during subordinate CA issuance ceremonies', 'REQUIRED', 'PKI'))
    story.append(chk(37, 'Deploy online subordinate CAs for aircraft enrollment and receiver authentication',
        'Separate subordinate CAs for aircraft identity and receiver infrastructure', 'REQUIRED', 'PKI'))
    story.append(chk(38, 'Establish CRL (Certificate Revocation List) and OCSP distribution points',
        'Emergency revocation must propagate within 60 seconds to all ground stations', 'REQUIRED', 'PKI'))
    story.append(chk(39, 'Configure certificate lifecycle management for automatic renewal',
        'Auto-renewal initiated 90 days before expiration; completed 30 days before', 'REQUIRED', 'PKI'))
    story.append(chk(40, 'Test PKI federation with any partner ANSP PKI infrastructure',
        'Cross-certification required for international airspace coverage', 'RECOMMENDED', 'PKI'))

    # ─── PHASE 3: Security Architecture ─────────────────────────────────────
    story.append(PageBreak())
    story.append(PhaseHeader(3, 'SECURITY ARCHITECTURE DESIGN',
        '4 weeks', 'QBITEL Security Architects + Customer Chief Engineer', NAVY))
    story.append(sp(6))

    section_divider(story, 'Deployment Model Decision')
    story.append(chk(41, 'Confirm final deployment scope: ground-only vs. ground + airborne',
        'Document decision rationale in DO-326A ASL for regulatory record', 'REQUIRED', 'ARCHITECTURE'))
    story.append(chk(42, 'Define authentication enrollment targets: which aircraft types and operators',
        'Prioritize high-frequency operators and aircraft with significant spoofing risk', 'REQUIRED', 'ARCHITECTURE'))
    story.append(chk(43, 'Define behavioral authentication thresholds for non-enrolled aircraft',
        'False positive rate target, detection sensitivity, and alert escalation thresholds', 'REQUIRED', 'ARCHITECTURE'))
    story.append(chk(44, 'Design ATC display integration for authentication status indicators',
        'HMI design reviewed with controllers and human factors specialists', 'REQUIRED', 'ARCHITECTURE'))

    section_divider(story, 'ARINC 653 Partition Design (Airborne Only)')
    story.append(chk(45, 'Define ARINC 653 security partition time budget and memory allocation',
        'Security partition time budget must not affect safety-critical partition guarantees', 'REQUIRED', 'AIRBORNE'))
    story.append(chk(46, 'Design security partition interfaces — ARINC 429 in, ARINC 429 out',
        'Interface specification reviewed by aircraft OEM avionics engineering', 'REQUIRED', 'AIRBORNE'))
    story.append(chk(47, 'Define security partition health monitoring interface to IMA health management',
        'Partition health status reported per ARINC 653 health monitoring API', 'REQUIRED', 'AIRBORNE'))
    story.append(chk(48, 'Complete DAL-D safety assessment and coordinate with OEM safety team',
        'Confirm DAL-D is appropriate given partition isolation characteristics', 'REQUIRED', 'AIRBORNE'))

    section_divider(story, 'Bandwidth and Compression Configuration')
    story.append(chk(49, 'Configure PQC algorithm selection per link type based on Phase 1 measurements',
        'LDACS: Falcon-512 compressed. VHF ACARS: Falcon-512 session. HF: session-based MAC.', 'REQUIRED', 'PQC'))
    story.append(chk(50, 'Set session key cache TTL based on operational patterns',
        'Default: 24 hours or 1,000 messages. Adjust for high-frequency operators.', 'REQUIRED', 'PQC'))
    story.append(chk(51, 'Validate compressed PQC sizes against link frame capacity for all link types',
        'Verify no link type exceeds overhead targets from Section 4 specifications', 'REQUIRED', 'PQC'))

    # ─── PHASE 4: ADS-B Authentication Deployment ────────────────────────────
    story.append(PageBreak())
    story.append(PhaseHeader(4, 'ADS-B AUTHENTICATION DEPLOYMENT',
        '6 weeks', 'QBITEL Field Engineering + ANSP Ground Operations', LIGHT_NAVY))
    story.append(sp(6))

    section_divider(story, 'Cryptographic Binding Infrastructure')
    story.append(chk(52, 'Deploy QBITEL authentication nodes at all ADS-B receiver sites in scope',
        'Authentication nodes installed in-line between receiver feed and network', 'REQUIRED', 'ADS-B-AUTH'))
    story.append(chk(53, 'Configure cryptographic MAC verification with PKI integration at each node',
        'Verify node can validate MACs for enrolled aircraft in less than 50ms', 'REQUIRED', 'ADS-B-AUTH'))
    story.append(chk(54, 'Deploy transponder-adjacent authentication units (TAUs) on first enrolled aircraft',
        'TAU installation during scheduled maintenance — estimated 4-8 hours per aircraft', 'REQUIRED', 'ADS-B-AUTH'))
    story.append(chk(55, 'Enroll first aircraft operator cohort in aviation PKI',
        'Issue enrollment certificates, configure aircraft ICAO address binding', 'REQUIRED', 'ADS-B-AUTH'))

    section_divider(story, 'Message Authentication Code (MAC) Integration')
    story.append(chk(56, 'Verify MAC computation and verification for enrolled aircraft at each receiver site',
        'Perform end-to-end MAC test flights with known enrolled aircraft', 'REQUIRED', 'ADS-B-AUTH'))
    story.append(chk(57, 'Configure authentication status pass-through to ASTERIX and SDPS feeds',
        'Authentication status bits embedded in ASTERIX records for ATC display', 'REQUIRED', 'ADS-B-AUTH'))
    story.append(chk(58, 'Verify ATC display shows authentication status for enrolled aircraft',
        'Controller display shows verified/unverified/anomaly status correctly', 'REQUIRED', 'ADS-B-AUTH'))
    story.append(chk(59, 'Validate graceful degradation — authentication failure does not block ATC data',
        'Simulate authentication node failure; confirm ADS-B data continues unaffected', 'REQUIRED', 'ADS-B-AUTH'))

    section_divider(story, 'PKI and Certificate Lifecycle Test')
    story.append(chk(60, 'Test certificate revocation propagation: measure time from revoke to ground station awareness',
        'Revocation must propagate to all stations within 60 seconds target', 'REQUIRED', 'PKI'))
    story.append(chk(61, 'Test certificate renewal process: renew an enrolled aircraft certificate without interruption',
        'Renewal must complete without affecting authentication continuity', 'REQUIRED', 'PKI'))

    # ─── PHASE 5: Data Link Protection ───────────────────────────────────────
    story.append(PageBreak())
    story.append(PhaseHeader(5, 'DATA LINK PROTECTION — LDACS / ACARS / SATCOM',
        '4 weeks', 'QBITEL Data Link Team + Airline Operations / ANSP', TEAL_DARK))
    story.append(sp(6))

    section_divider(story, 'LDACS PQC Configuration')
    story.append(chk(62, 'Deploy QBITEL LDACS security gateway at ground station infrastructure',
        'LDACS gateway intercepts, authenticates, and forwards data link traffic', 'REQUIRED', 'LDACS'))
    story.append(chk(63, 'Configure Falcon-512 compressed PQC for LDACS data sessions',
        'Verify compressed signature fits within LDACS frame structure (600 bps minimum)', 'REQUIRED', 'LDACS'))
    story.append(chk(64, 'Test CPDLC authentication through LDACS security layer',
        'Send test CPDLC clearances and verify authentication and integrity end-to-end', 'REQUIRED', 'LDACS'))
    story.append(chk(65, 'Validate LDACS session continuity across aircraft movement between LDACS cells',
        'Session handover must maintain authentication without visible delay', 'REQUIRED', 'LDACS'))

    section_divider(story, 'ACARS Gateway Quantum Protection')
    story.append(chk(66, 'Deploy QBITEL ACARS authentication gateway at all AOC ACARS uplink points',
        'Gateway performs MAC authentication on all ACARS uplinks and downlinks', 'REQUIRED', 'ACARS'))
    story.append(chk(67, 'Configure VHF ACARS session-based PQC (Falcon-512 compressed)',
        'Verify session establishment completes within 3 ACARS message exchange cycles', 'REQUIRED', 'ACARS'))
    story.append(chk(68, 'Configure HF ACARS low-bandwidth authentication mode',
        'Session-based MAC with 32-byte HMAC. Verify overhead under 12% at 1,200 bps.', 'RECOMMENDED', 'ACARS'))
    story.append(chk(69, 'Test ACARS authentication with representative message types (OOOI, M-PDU, weather)',
        'Verify authentication does not exceed 0.8-second latency overhead on VHF', 'REQUIRED', 'ACARS'))
    story.append(chk(70, 'Enable non-repudiation logging for safety-critical ACARS messages',
        'Maintenance write-ups and MEL deferrals logged with cryptographic proof of origin', 'REQUIRED', 'ACARS'))

    section_divider(story, 'SATCOM Session Security')
    story.append(chk(71, 'Deploy QBITEL SATCOM security gateway at Iridium and Inmarsat ground earth stations',
        'Gateway wraps SATCOM sessions with PQC-derived session keys', 'RECOMMENDED', 'SATCOM'))
    story.append(chk(72, 'Configure adaptive PQC selection based on SATCOM link rate',
        'Iridium (600 bps): session MAC. Inmarsat SB-S (10.5 kbps): full Falcon-512.', 'RECOMMENDED', 'SATCOM'))
    story.append(chk(73, 'Test ACARS-over-SATCOM (AOS) authentication for oceanic routes',
        'Verify authentication continuity during oceanic flight (Iridium and Inmarsat)', 'RECOMMENDED', 'SATCOM'))

    # ─── PHASE 6: Spoofing Detection Activation ───────────────────────────────
    story.append(PageBreak())
    story.append(PhaseHeader(6, 'AI SPOOFING DETECTION ACTIVATION',
        '6 weeks', 'QBITEL AI/ML Team + ANSP Operations', NAVY))
    story.append(sp(6))

    section_divider(story, 'AI Multilateration Cross-Validation')
    story.append(chk(74, 'Integrate QBITEL AI engine with MLAT receiver network feeds',
        'Minimum 4 receiver stations per coverage zone for MLAT accuracy', 'REQUIRED', 'AI-DETECT'))
    story.append(chk(75, 'Collect 4 weeks of baseline ADS-B+MLAT correlated data for AI model training',
        'Training dataset must include diverse aircraft types, weather, and traffic conditions', 'REQUIRED', 'AI-DETECT'))
    story.append(chk(76, 'Train AI behavioral models on local traffic patterns and aircraft performance data',
        'Model must be trained per aircraft type using locally representative training data', 'REQUIRED', 'AI-DETECT'))
    story.append(chk(77, 'Validate AI model performance against known legitimate anomalies',
        'Use historical data with known emergency maneuvers, weather deviations, equipment issues', 'REQUIRED', 'AI-DETECT'))

    section_divider(story, 'Alert Threshold Configuration')
    story.append(chk(78, 'Configure alert thresholds for position discontinuity detection',
        'Threshold based on aircraft performance envelope for each enrolled aircraft type', 'REQUIRED', 'AI-DETECT'))
    story.append(chk(79, 'Configure MLAT-vs-ADS-B position discrepancy alert thresholds',
        'Start conservative (high threshold) and tune based on false positive rate observations', 'REQUIRED', 'AI-DETECT'))
    story.append(chk(80, 'Configure velocity and altitude violation detection thresholds',
        'Velocity and rate-of-climb limits derived from aircraft type performance data', 'REQUIRED', 'AI-DETECT'))
    story.append(chk(81, 'Set alert escalation levels: Advisory / Warning / Alert for ATC display',
        'Advisory: anomaly detected. Warning: probable spoofing. Alert: confirmed attack.', 'REQUIRED', 'AI-DETECT'))

    section_divider(story, 'False Positive Rate Validation')
    story.append(chk(82, 'Operate AI detection in advisory-only mode for 2 weeks',
        'Record all detections; classify as true positive, false positive, or uncertain', 'REQUIRED', 'AI-DETECT'))
    story.append(chk(83, 'Measure false positive rate against target of less than 0.001%',
        'Adjust thresholds if false positive rate exceeds target before operational activation', 'REQUIRED', 'AI-DETECT'))
    story.append(chk(84, 'Conduct spoofing simulation exercises to validate detection rate',
        'Use QBITEL spoofing simulation tool in controlled, coordinated environment', 'REQUIRED', 'AI-DETECT'))
    story.append(chk(85, 'Document detection rate and false positive rate for DO-326A SDA',
        'Measured performance must be captured in Security Development Assurance documentation', 'REQUIRED', 'DO-326A'))

    # ─── PHASE 7: ATC System Integration ─────────────────────────────────────
    story.append(PageBreak())
    story.append(PhaseHeader(7, 'ATC SYSTEM INTEGRATION',
        '4 weeks', 'QBITEL Integration Team + ANSP Systems Engineering', LIGHT_NAVY))
    story.append(sp(6))

    section_divider(story, 'Radar Correlation Integration')
    story.append(chk(86, 'Integrate QBITEL radar correlation engine with SSR Mode S radar feeds',
        'Ingest SSR data for real-time ADS-B vs. radar position correlation', 'REQUIRED', 'ATC-INT'))
    story.append(chk(87, 'Configure radar-ADS-B discrepancy alert thresholds',
        'Discrepancy threshold accounts for GPS vs. SSR accuracy differences', 'REQUIRED', 'ATC-INT'))
    story.append(chk(88, 'Validate correlation engine performance with known radar-ADS-B offset data',
        'Use historical data from aircraft with known GPS-SSR offsets for calibration', 'REQUIRED', 'ATC-INT'))

    section_divider(story, 'Flight Data Processor Integration')
    story.append(chk(89, 'Integrate authentication status with Flight Data Processor (FDP) track records',
        'Authentication status appended to FDP track data for correlation with flight plan data', 'REQUIRED', 'ATC-INT'))
    story.append(chk(90, 'Verify authentication status displayed on controller working positions',
        'HMI verification with at least 3 controller working positions across different display types', 'REQUIRED', 'ATC-INT'))
    story.append(chk(91, 'Test ASTERIX output with authentication status bits to all downstream consumers',
        'Verify all ASTERIX consumers (airline flight tracking, airport systems) receive status', 'RECOMMENDED', 'ATC-INT'))

    section_divider(story, 'Safety Alert Integration')
    story.append(chk(92, 'Integrate spoofing alerts with ATC safety net systems',
        'Spoofing alerts coordinated with STCA, MSAW, and other safety net systems', 'REQUIRED', 'ATC-INT'))
    story.append(chk(93, 'Verify spoofing alerts do not generate spurious STCA advisories',
        'STCA must filter alerts for tracks flagged as spoofed/unverified', 'REQUIRED', 'ATC-INT'))
    story.append(chk(94, 'Conduct integrated test with safety nets: confirm expected alert behavior',
        'Simulated spoofing injection: verify correct alert hierarchy and suppression', 'REQUIRED', 'ATC-INT'))

    # ─── PHASE 8: DO-326A Evidence Generation ────────────────────────────────
    story.append(PageBreak())
    story.append(PhaseHeader(8, 'DO-326A EVIDENCE GENERATION',
        '4 weeks', 'QBITEL Compliance Team + Customer Airworthiness Team', TEAL_DARK))
    story.append(sp(6))

    section_divider(story, 'Security Target Analysis (STA)')
    story.append(chk(95, 'Complete Security Target Analysis covering all threat vectors in scope',
        'STA must cover ADS-B spoofing, data link attacks, network intrusion, insider threats', 'REQUIRED', 'DO-326A'))
    story.append(chk(96, 'Document threat likelihood and impact assessments for each threat vector',
        'Likelihood and impact based on operational environment, not generic assessments', 'REQUIRED', 'DO-326A'))
    story.append(chk(97, 'Derive security requirements from STA threat analysis',
        'Each threat must have at least one security requirement that mitigates it', 'REQUIRED', 'DO-326A'))

    section_divider(story, 'Security Development Assurance (SDA)')
    story.append(chk(98, 'Complete Security Development Analysis demonstrating requirements are implemented',
        'SDA must trace each security requirement to design decisions and verification evidence', 'REQUIRED', 'DO-326A'))
    story.append(chk(99, 'Include penetration test results in SDA as verification evidence',
        'Third-party penetration test recommended for ATC-critical deployments', 'RECOMMENDED', 'DO-326A'))
    story.append(chk(100, 'Include ADS-B spoofing simulation results in SDA',
        'Detection rate and false positive rate measurements from Phase 6', 'REQUIRED', 'DO-326A'))

    section_divider(story, 'Aircraft Security Log (ASL)')
    story.append(chk(101, 'Update ASL with all security-relevant decisions made during deployment',
        'ASL captures design decisions, assumptions, and accepted residual risks', 'REQUIRED', 'DO-326A'))
    story.append(chk(102, 'Obtain airworthiness authority acceptance of ASL (if airborne deployment)',
        'Ground-only: ASL submitted to authority as notification. Airborne: formal acceptance required.', 'REQUIRED', 'DO-326A'))

    section_divider(story, 'DO-178C Evidence (Airborne Partition Only)')
    story.append(chk(103, 'Deliver complete DO-178C Level D evidence package to OEM/STC applicant',
        'Software Plans (4), Development artifacts (4), Verification artifacts (4)', 'REQUIRED', 'DO-178C'))
    story.append(chk(104, 'Coordinate DO-178C evidence review with FAA DER or EASA DOA',
        'Designated Engineering Representative or Design Organization Approval review', 'REQUIRED', 'DO-178C'))

    # ─── PHASE 9: Validation & Testing ───────────────────────────────────────
    story.append(PageBreak())
    story.append(PhaseHeader(9, 'VALIDATION AND TESTING',
        '4 weeks', 'QBITEL QA Team + ANSP Safety Review Board', NAVY))
    story.append(sp(6))

    section_divider(story, 'Spoofing Simulation Exercises')
    story.append(chk(105, 'Conduct controlled ADS-B ghost aircraft injection test at non-operational receiver',
        'Coordinated exercise with ANSP, using QBITEL simulation tool on isolated receiver', 'REQUIRED', 'VALIDATION'))
    story.append(chk(106, 'Conduct ADS-B position falsification test for enrolled aircraft',
        'Inject false position data; verify detection within 30-second target', 'REQUIRED', 'VALIDATION'))
    story.append(chk(107, 'Conduct ACARS message forgery simulation at AOC test environment',
        'Inject forged ACARS message; verify real-time authentication rejection', 'REQUIRED', 'VALIDATION'))
    story.append(chk(108, 'Test network intrusion simulation against ATC quantum-safe VPN',
        'Third-party penetration test of VPN and SWIM API security layer', 'RECOMMENDED', 'VALIDATION'))

    section_divider(story, 'False Positive Rate Final Validation')
    story.append(chk(109, 'Measure operational false positive rate over 30-day monitoring period',
        'Target: less than 0.001%. Document and investigate all false positives.', 'REQUIRED', 'VALIDATION'))
    story.append(chk(110, 'Confirm false positive rate does not adversely affect controller workload',
        'Controller workload assessment with operational air traffic controllers', 'REQUIRED', 'VALIDATION'))

    section_divider(story, 'Safety Review Board')
    story.append(chk(111, 'Present DO-326A evidence package to internal Safety Review Board',
        'Board includes safety team, airworthiness representative, and operations management', 'REQUIRED', 'SAFETY'))
    story.append(chk(112, 'Obtain Safety Review Board acceptance before operational activation',
        'Board acceptance documented as formal record in DO-326A ASL', 'REQUIRED', 'SAFETY'))
    story.append(chk(113, 'Submit DO-326A compliance package to national aviation authority',
        'FAA: Issue Paper resolution. EASA: CRI closure. National CAA: equivalent process.', 'REQUIRED', 'REGULATORY'))

    # ─── PHASE 10: Operational Handover ───────────────────────────────────────
    story.append(PageBreak())
    story.append(PhaseHeader(10, 'OPERATIONAL HANDOVER',
        '4 weeks', 'QBITEL Customer Success + ANSP Operations Management', LIGHT_NAVY))
    story.append(sp(6))

    section_divider(story, 'ANS Operator Training')
    story.append(chk(114, 'Complete QBITEL Bridge administrator training for ANS technical staff',
        'System administration, monitoring, alert management, and key lifecycle management', 'REQUIRED', 'TRAINING'))
    story.append(chk(115, 'Complete ATC controller familiarization training',
        'Authentication status indicators, alert response procedures, reporting protocols', 'REQUIRED', 'TRAINING'))
    story.append(chk(116, 'Complete SOC operator training for QBITEL monitoring workflows',
        'Alert triage, incident classification, escalation procedures, regulatory reporting', 'REQUIRED', 'TRAINING'))
    story.append(chk(117, 'Deliver operator documentation: system manual, quick reference cards, procedures',
        'All documentation reviewed and approved by ANSP operations management', 'REQUIRED', 'TRAINING'))

    section_divider(story, 'Safety Case Approval')
    story.append(chk(118, 'Obtain formal safety case acceptance from national aviation authority',
        'Ground-only: authority notification with no-objection response. Airborne: formal acceptance.', 'REQUIRED', 'SAFETY'))
    story.append(chk(119, 'Update operational procedures and ATC controller instructions',
        'Safety-case-approved procedures published and effective before operational activation', 'REQUIRED', 'SAFETY'))
    story.append(chk(120, 'Confirm no outstanding DO-326A evidence items or authority queries',
        'All authority questions resolved before declaring Full Operational Status', 'REQUIRED', 'REGULATORY'))

    section_divider(story, 'Continuous Monitoring Activation')
    story.append(chk(121, 'Activate QBITEL SOC monitoring for full operational coverage',
        '24/7 monitoring with aviation-specific threat intelligence and alert triage', 'REQUIRED', 'OPERATIONS'))
    story.append(chk(122, 'Configure automated regulatory incident reporting',
        'Reporting templates compliant with FAA, EASA, and national CAA requirements', 'REQUIRED', 'OPERATIONS'))
    story.append(chk(123, 'Establish monthly security posture review cadence with ANSP safety team',
        'Monthly report includes detection statistics, false positive rate, and threat intelligence', 'REQUIRED', 'OPERATIONS'))
    story.append(chk(124, 'Schedule annual DO-326A compliance review aligned with SMS audit cycle',
        'Annual review updates ASL, reviews threat landscape, and updates STA if required', 'REQUIRED', 'DO-326A'))
    story.append(chk(125, 'Declare Full Operational Status (FOS) with documented authority confirmation',
        'FOS declaration recorded in DO-326A ASL and communicated to all stakeholders', 'REQUIRED', 'OPERATIONS'))

    # ─── APPENDIX ─────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph('APPENDIX: PERFORMANCE SLAs, ROLLBACK, AND ESCALATION',
        ParagraphStyle('app_title', fontName='Helvetica-Bold', fontSize=13,
                       textColor=NAVY, spaceAfter=8)))

    story.append(Paragraph('Performance SLA Targets', ParagraphStyle('app_h',
        fontName='Helvetica-Bold', fontSize=10, textColor=TEAL_DARK, spaceAfter=4)))
    sla_data = [
        ['Metric', 'Target', 'Measurement Period', 'Remedy if Breached'],
        ['System Availability', '99.999% (5 min/year)', 'Monthly', 'Service credit 10% monthly fee'],
        ['P1 Incident Response', '<15 minutes', 'Per incident', 'Service credit 5% monthly fee'],
        ['On-Site Escalation', '<4 hours', 'Per P1 incident', 'Service credit 5% monthly fee'],
        ['Auth Latency (ADS-B)', '<50 ms end-to-end', 'Weekly average', 'Investigation and remediation'],
        ['False Positive Rate', '<0.001%', 'Monthly average', 'Threshold re-tuning within 5 days'],
        ['Revocation Propagation', '<60 seconds', 'Per event', 'Emergency process review'],
        ['AI Model Accuracy', '>99.7% systematic', 'Quarterly', 'Model retraining within 30 days'],
        ['Software Patch Deployment', '<72 hours critical', 'Per vulnerability', 'Emergency maintenance window'],
    ]
    st = Table(sla_data, colWidths=[CONTENT_W * 0.27, CONTENT_W * 0.22,
                                     CONTENT_W * 0.22, CONTENT_W * 0.29])
    st.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR',   (0, 0), (-1, 0), WHITE_C),
        ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',    (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',    (0, 0), (-1, -1), 7.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('GRID',        (0, 0), (-1, -1), 0.3, MID_GREY),
        ('TOPPADDING',  (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(st)
    story.append(sp(10))

    story.append(Paragraph('Rollback Procedures', ParagraphStyle('app_h',
        fontName='Helvetica-Bold', fontSize=10, textColor=TEAL_DARK, spaceAfter=4)))
    rollback_data = [
        ['Trigger', 'Rollback Action', 'Time to Rollback', 'Authority Required'],
        ['Authentication node failure', 'Automatic failover to standby node', '<500 ms', 'None (automated)'],
        ['False positive rate >0.01%', 'Revert to advisory-only mode', '<30 minutes', 'ANSP Ops Manager'],
        ['Safety net conflict detected', 'Suspend authentication display', '<5 minutes', 'ANSP Safety Manager'],
        ['PKI infrastructure failure', 'Revert to behavioral-only mode', '<60 seconds', 'ANSP Ops Manager'],
        ['Authority instruction to suspend', 'Full system bypass/pass-through', '<5 minutes', 'ANSP Accountable Mgr'],
        ['Major software update failure', 'Rollback to previous certified version', '<2 hours', 'QBITEL Change Board'],
    ]
    rt = Table(rollback_data, colWidths=[CONTENT_W * 0.27, CONTENT_W * 0.30,
                                          CONTENT_W * 0.18, CONTENT_W * 0.25])
    rt.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0), TEAL_DARK),
        ('TEXTCOLOR',   (0, 0), (-1, 0), WHITE_C),
        ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',    (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',    (0, 0), (-1, -1), 7.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('GRID',        (0, 0), (-1, -1), 0.3, MID_GREY),
        ('TOPPADDING',  (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(rt)
    story.append(sp(10))

    story.append(Paragraph('Escalation Matrix', ParagraphStyle('app_h',
        fontName='Helvetica-Bold', fontSize=10, textColor=TEAL_DARK, spaceAfter=4)))
    esc_data = [
        ['Priority', 'Definition', 'QBITEL Response', 'Customer Contact', 'Authority Notification'],
        ['P1 — Critical', 'Auth system down or safety impact', '<15 min / 24x7 SOC', 'ANSP Ops Director', 'Immediate if safety impact'],
        ['P2 — High', 'Detection rate degraded or high FP rate', '<1 hour / 24x7 SOC', 'ANSP IT Manager', 'Within 24 hours if required'],
        ['P3 — Medium', 'Non-critical feature issue', '<4 hours / business hours', 'Technical lead', 'Not required'],
        ['P4 — Low', 'Enhancement or cosmetic issue', '<2 business days', 'Account manager', 'Not required'],
        ['Security Incident', 'Confirmed spoofing attack detected', '<5 min / SOC escalation', 'ANSP Safety Manager', 'Per national authority requirements'],
    ]
    et = Table(esc_data, colWidths=[CONTENT_W * 0.14, CONTENT_W * 0.25,
                                     CONTENT_W * 0.22, CONTENT_W * 0.19, CONTENT_W * 0.20])
    et.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR',   (0, 0), (-1, 0), WHITE_C),
        ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',    (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',    (0, 0), (-1, -1), 7),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('GRID',        (0, 0), (-1, -1), 0.3, MID_GREY),
        ('TOPPADDING',  (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(et)
    story.append(sp(12))

    story.append(Paragraph(
        'QBITEL Bridge — Aviation and Aerospace Deployment Checklist  |  '
        'enterprise@qbitel.com  |  https://bridge.qbitel.com',
        ParagraphStyle('footer_note', fontName='Helvetica-Oblique',
                       fontSize=8, textColor=MID_GREY, alignment=TA_CENTER)))

    doc.build(story)
    print(f"PDF written: {output_path}")


if __name__ == '__main__':
    import os
    out = '/Users/prabakarankannan/qbitel/docs/brochures/QBITEL_Aviation_Deployment_Checklist.pdf'
    build_doc(out)
    size = os.path.getsize(out)
    print(f"File size: {size:,} bytes ({size/1024:.1f} KB)")
