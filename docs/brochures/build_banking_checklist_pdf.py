"""
Build QBITEL Bridge Banking Deployment Checklist - Professional PDF
10 phases + appendices for Banking & Financial Services deployment
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
RED_SOFT   = HexColor('#8B1A1A')
ORANGE     = HexColor('#E07B00')

PAGE_W, PAGE_H = letter
MARGIN = 0.8 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


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
        c.setFillColor(GOLD)
        c.rect(0, 0, 56, self.h, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(self.w - 4, 0, 4, self.h, fill=1, stroke=0)

        # Phase number
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 10)
        pn = self.phase_num
        pw = c.stringWidth(pn, 'Helvetica-Bold', 10)
        c.drawString(28 - pw / 2, self.h - 20, pn)
        c.setFillColor(NAVY)
        c.setFont('Helvetica', 7)
        lw = c.stringWidth('PHASE', 'Helvetica', 7)
        c.drawString(28 - lw / 2, self.h - 32, 'PHASE')

        # Title
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 12)
        c.drawString(66, self.h - 22, self.title.upper())

        # Duration and owner
        c.setFillColor(TEAL)
        c.setFont('Helvetica', 8)
        meta = '%s  |  Owner: %s' % (self.duration, self.owner)
        c.drawString(66, self.h - 36, meta)


class CheckItem(Flowable):
    def __init__(self, text, sub_items=None, priority='REQUIRED', width=None):
        super().__init__()
        self.text = text
        self.sub_items = sub_items or []
        self.priority = priority
        self.w = width or CONTENT_W
        self._calc_h()

    def _calc_h(self):
        chars_per_line = int((self.w - 70) / 5.5)
        main_lines = max(1, len(self.text) // chars_per_line + 1)
        self.main_h = main_lines * 13 + 20
        self.sub_h = 0
        for sub in self.sub_items:
            sub_lines = max(1, len(sub) // (chars_per_line + 8) + 1)
            self.sub_h += sub_lines * 12 + 6
        self.h = self.main_h + self.sub_h + (4 if self.sub_items else 0)

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv

        # Priority color
        priority_colors = {
            'REQUIRED': NAVY,
            'RECOMMENDED': TEAL_DARK,
            'OPTIONAL': MID_GREY,
            'CRITICAL': HexColor('#8B0000'),
        }
        pcol = priority_colors.get(self.priority, NAVY)

        # Background
        c.setFillColor(WHITE_C)
        c.rect(0, self.sub_h, self.w, self.main_h, fill=1, stroke=0)
        # Left accent
        c.setFillColor(pcol)
        c.rect(0, self.sub_h, 3, self.main_h, fill=1, stroke=0)
        # Checkbox
        c.setStrokeColor(MID_GREY)
        c.setFillColor(WHITE_C)
        c.rect(10, self.sub_h + (self.main_h - 14) / 2, 14, 14, fill=1, stroke=1)
        # Priority badge
        c.setFillColor(pcol)
        c.roundRect(self.w - 80, self.sub_h + (self.main_h - 16) / 2, 70, 16, 4, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 7)
        bw = c.stringWidth(self.priority, 'Helvetica-Bold', 7)
        c.drawString(self.w - 80 + (70 - bw) / 2, self.sub_h + (self.main_h - 16) / 2 + 5, self.priority)
        # Text
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 9)
        text_x = 32
        text_w = self.w - 90
        words = self.text.split()
        line = ''
        lines_out = []
        for word in words:
            test = (line + ' ' + word).strip()
            if c.stringWidth(test, 'Helvetica', 9) < text_w:
                line = test
            else:
                if line:
                    lines_out.append(line)
                line = word
        if line:
            lines_out.append(line)
        y_start = self.sub_h + self.main_h - 14
        for ln in lines_out:
            c.drawString(text_x, y_start, ln)
            y_start -= 13

        # Sub items
        if self.sub_items:
            c.setFillColor(LIGHT_BG)
            c.rect(0, 0, self.w, self.sub_h, fill=1, stroke=0)
            c.setFillColor(TEAL)
            c.rect(0, 0, 3, self.sub_h, fill=1, stroke=0)
            c.setFillColor(DARK_TEXT)
            c.setFont('Helvetica', 8)
            y_sub = self.sub_h - 12
            for sub in self.sub_items:
                c.setFillColor(TEAL)
                c.circle(24, y_sub + 3, 2, fill=1, stroke=0)
                c.setFillColor(DARK_TEXT)
                words2 = sub.split()
                line2 = ''
                lines2 = []
                text_w2 = self.w - 50
                for word in words2:
                    test2 = (line2 + ' ' + word).strip()
                    if c.stringWidth(test2, 'Helvetica', 8) < text_w2:
                        line2 = test2
                    else:
                        if line2:
                            lines2.append(line2)
                        line2 = word
                if line2:
                    lines2.append(line2)
                for ln2 in lines2:
                    c.drawString(32, y_sub, ln2)
                    y_sub -= 12

        # Bottom separator
        c.setStrokeColor(LIGHT_BG)
        c.setLineWidth(0.5)
        c.line(0, 0, self.w, 0)


def draw_cover(canvas, doc):
    canvas.saveState()
    w, h = PAGE_W, PAGE_H
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, h * 0.58, w, h * 0.42, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    p = canvas.beginPath()
    p.moveTo(w * 0.55, h)
    p.lineTo(w, h)
    p.lineTo(w, h * 0.75)
    p.close()
    canvas.drawPath(p, fill=1, stroke=0)
    canvas.setFillColor(TEAL_DARK)
    canvas.rect(0, 0, w, 2 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, 2 * inch, w, 5, fill=1, stroke=0)

    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 28)
    canvas.drawString(MARGIN, h * 0.88, 'QBITEL BRIDGE')
    canvas.setFillColor(GOLD)
    canvas.setFont('Helvetica-Bold', 16)
    canvas.drawString(MARGIN, h * 0.83, 'Banking & Financial Services')
    canvas.setFillColor(GOLD)
    canvas.rect(MARGIN, h * 0.823, 3.5 * inch, 3, fill=1, stroke=0)

    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 30)
    canvas.drawString(MARGIN, h * 0.72, 'DEPLOYMENT CHECKLIST')
    canvas.setFont('Helvetica-Bold', 18)
    canvas.drawString(MARGIN, h * 0.72 - 36, 'Banking & Financial Services')
    canvas.setFillColor(HexColor('#CCDDE8'))
    canvas.setFont('Helvetica-Oblique', 12)
    canvas.drawString(MARGIN, h * 0.72 - 62,
                      '10 Phases | 16 Weeks | ISO 8583 | SWIFT | COBOL | FIX | PCI-DSS 4.0 | DORA')

    # Phase summary boxes
    phases = [('0-2', 'Discover\n& Plan'), ('3-5', 'Build\nInfra'), ('6-10', 'Protect\nPayments'), ('11-16', 'Go\nLive')]
    box_w = (w - 2 * MARGIN) / 4
    box_h = inch
    box_y = 2.1 * inch
    cols = [NAVY, LIGHT_NAVY, NAVY, LIGHT_NAVY]
    for i, (wks, lbl) in enumerate(phases):
        bx = MARGIN + i * box_w
        canvas.setFillColor(cols[i])
        canvas.rect(bx, box_y, box_w - 4, box_h, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.setFont('Helvetica-Bold', 13)
        ww = canvas.stringWidth('WK ' + wks, 'Helvetica-Bold', 13)
        canvas.drawString(bx + (box_w - 4 - ww) / 2, box_y + box_h - 26, 'WK ' + wks)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 8)
        for j, ln in enumerate(lbl.split('\n')):
            lw = canvas.stringWidth(ln, 'Helvetica', 8)
            canvas.drawString(bx + (box_w - 4 - lw) / 2, box_y + box_h - 44 - j * 13, ln)

    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 9)
    canvas.drawString(MARGIN, 0.8 * inch, 'enterprise@qbitel.com  |  https://bridge.qbitel.com')
    canvas.setFillColor(HexColor('#AABBCC'))
    canvas.setFont('Helvetica', 8)
    canvas.drawString(MARGIN, 0.55 * inch, 'Confidential - For Implementation Teams Only  |  (c) 2026 QBITEL.')
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
    canvas.drawString(MARGIN + 1.1 * inch, PAGE_H - 0.31 * inch, 'BANKING DEPLOYMENT CHECKLIST')
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
                      'Confidential - For Implementation Teams Only  |  (c) 2026 QBITEL. All Rights Reserved.')
    contact_str = 'enterprise@qbitel.com  |  bridge.qbitel.com'
    cw = canvas.stringWidth(contact_str, 'Helvetica', 7.5)
    canvas.drawString(PAGE_W - MARGIN - cw, 0.15 * inch, contact_str)
    canvas.restoreState()


def make_styles():
    styles = {}
    styles['body'] = ParagraphStyle('body', fontName='Helvetica', fontSize=9,
                                     leading=13, textColor=DARK_TEXT, spaceAfter=4)
    styles['note'] = ParagraphStyle('note', fontName='Helvetica-Oblique', fontSize=8.5,
                                     leading=12, textColor=MID_GREY, spaceAfter=4, leftIndent=10)
    styles['h3'] = ParagraphStyle('h3', fontName='Helvetica-Bold', fontSize=10,
                                   leading=14, textColor=NAVY, spaceAfter=4, spaceBefore=8)
    styles['table_h'] = ParagraphStyle('table_h', fontName='Helvetica-Bold', fontSize=8,
                                        leading=11, textColor=WHITE_C, alignment=TA_CENTER)
    styles['table_c'] = ParagraphStyle('table_c', fontName='Helvetica', fontSize=8,
                                        leading=11, textColor=DARK_TEXT)
    return styles


def tbl_style(header_rows=1):
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, header_rows - 1), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, header_rows - 1), WHITE_C),
        ('FONTNAME', (0, 0), (-1, header_rows - 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, header_rows - 1), 8),
        ('ALIGN', (0, 0), (-1, header_rows - 1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, header_rows), (-1, -1), [WHITE_C, TABLE_ALT]),
        ('FONTNAME', (0, header_rows), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, header_rows), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.4, MID_GREY),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ])


def phase_block(S, ph_num, title, duration, owner, items, styles, note=None):
    S.append(KeepTogether([
        PhaseHeader(ph_num, title, duration, owner),
        Spacer(1, 4),
    ]))
    if note:
        S.append(Paragraph(note, styles['note']))
        S.append(Spacer(1, 4))
    for text, subs, priority in items:
        S.append(KeepTogether([
            CheckItem(text, subs, priority),
            Spacer(1, 3),
        ]))
    S.append(Spacer(1, 10))


def build_story(styles):
    S = []

    def sp(n=8):
        S.append(Spacer(1, n))

    # Phase 0: Pre-Engagement & Scoping
    phase_block(S, '0', 'Pre-Engagement & Scoping', 'Week 0 (Pre-Start)', 'Account Executive + Solutions Architect', [
        ('Confirm executive sponsor: CISO or CTO with authority to approve scope and budget.',
         ['Identify CISO, CTO, Head of Payments, Head of Compliance as primary stakeholders',
          'Confirm board or ExCo awareness of quantum security initiative',
          'Assign internal QBITEL Bridge project owner (minimum 0.5 FTE)'],
         'CRITICAL'),
        ('Define engagement scope: which payment rails, protocols, and data centers are in scope for Phase 1.',
         ['List all ISO 8583 card scheme connections (Visa, Mastercard, domestic)',
          'List all SWIFT connectivity points (Alliance Access, Alliance Gateway)',
          'Identify mainframe environments (z/OS version, CICS/IMS, DB2/IMS DB)',
          'Identify FIX trading environments if applicable'],
         'REQUIRED'),
        ('Conduct infrastructure audit: collect existing network diagrams, HSM inventory, and protocol documentation.',
         ['Request existing network diagrams for payment DMZ and mainframe LAN',
          'Collect HSM inventory: vendor, model, firmware version, partition structure',
          'Gather SWIFT CSP last attestation report',
          'Collect most recent PCI-DSS QSA report or SAQ'],
         'REQUIRED'),
        ('Complete compliance gap assessment: PCI-DSS 4.0, DORA, SWIFT CSP 2025.',
         ['Map current cryptographic implementations against PCI-DSS 4.0 Requirements 3, 4, 12.3.3',
          'Assess DORA Article 9/10/11 current posture',
          'Review SWIFT CSP mandatory controls status'],
         'REQUIRED'),
        ('Obtain data center access approvals for passive tap deployment.',
         ['Submit change request for SPAN/mirror port configuration on payment network switches',
          'Obtain network security team sign-off for passive monitoring'],
         'REQUIRED'),
        ('Define success criteria and KPIs for the engagement.',
         ['Agree on TPS throughput target (default: 10,000 TPS)',
          'Agree on latency budget (default: <50ms)',
          'Agree on compliance evidence package format and target date'],
         'REQUIRED'),
        ('Complete QBITEL Bridge master service agreement and data processing agreement.',
         ['Legal review of MSA, DPA, SLA schedules',
          'GDPR data processing agreement for any EU customer data in scope',
          'Escrow arrangement documentation (if required)'],
         'REQUIRED'),
        ('Schedule kickoff workshop with all stakeholders.',
         ['2-day on-site workshop: architecture review, protocol discovery approach, risk register seed'],
         'REQUIRED'),
    ], styles, 'Phase 0 must be completed before any on-site deployment begins. All access approvals must be in writing.')

    S.append(PageBreak())

    # Phase 1: Protocol Discovery
    phase_block(S, '1', 'Protocol Discovery', 'Weeks 1-2', 'Solutions Architect + Client Network Team', [
        ('Configure SPAN/mirror ports on payment network core switches for passive tap.',
         ['Payment DMZ switch: configure SPAN for all ports carrying ISO 8583 traffic',
          'Mainframe LAN switch: configure SPAN for TN3270e and DRDA traffic',
          'SWIFT zone switch: configure SPAN for SWIFT Alliance Access connections',
          'Trading LAN switch: configure SPAN for FIX session ports (if in scope)'],
         'REQUIRED'),
        ('Deploy QBITEL Bridge Discovery Appliance in passive-only mode.',
         ['Rack-mount or VM deployment in out-of-band monitoring VLAN',
          'Verify no traffic modification capability - discovery appliance is receive-only',
          'Configure packet capture storage: minimum 2TB for 48-hour capture baseline'],
         'REQUIRED'),
        ('Execute automated ISO 8583 and ISO 20022 protocol fingerprinting.',
         ['Confirm ISO 8583 bit-map detection across all card scheme connections',
          'Identify ISO 20022 message types in use (pacs.008, camt.053, pain.001)',
          'Map acquirer-to-issuer and processor-to-scheme message flows',
          'Identify any legacy ISO 8583 v1987 vs v1993 vs v2003 versions in use'],
         'REQUIRED'),
        ('Execute automated SWIFT MT/MX protocol fingerprinting.',
         ['Identify SWIFT MT103, MT202, MT515, MT940, MT950 message flows',
          'Confirm FIN framing vs MX ISO 20022 identification',
          'Map SWIFT Alliance Access connectivity paths',
          'Identify any proprietary SWIFT extensions in use'],
         'REQUIRED'),
        ('Execute TN3270e and mainframe protocol discovery.',
         ['Identify TN3270e sessions: source IP, CICS region, transaction codes',
          'Identify DB2 DRDA connection sources and database names',
          'Identify IBM MQ queue managers and channels carrying sensitive data',
          'Map IMS Connect connection paths if applicable'],
         'REQUIRED'),
        ('Execute FIX protocol discovery if trading in scope.',
         ['Identify FIX versions (4.2, 4.4, 5.0) and CompIDs in use',
          'Map FIX sessions to ECN/exchange counterparties',
          'Identify FpML connections if OTC derivatives in scope'],
         'RECOMMENDED'),
        ('Generate Protocol Risk Register.',
         ['Rank all discovered protocols by quantum vulnerability (RSA key size, cipher suite)',
          'Assign breach impact score (payment volume, customer PII, regulatory sensitivity)',
          'Calculate quantum vulnerability timeline per protocol',
          'Produce prioritized remediation roadmap'],
         'REQUIRED'),
        ('Present Protocol Risk Register to CISO and Head of Payments.',
         ['Schedule executive briefing: 60-minute presentation of discovery findings',
          'Agree remediation priority order for Phase 3',
          'Obtain sign-off on Phase 2 scope'],
         'REQUIRED'),
    ], styles, 'Protocol discovery is passive - zero traffic modification. SPAN configuration is the only change required to production infrastructure.')

    S.append(PageBreak())

    # Phase 2: Infrastructure Readiness
    phase_block(S, '2', 'Infrastructure Readiness', 'Weeks 3-5', 'Solutions Architect + Client Infrastructure Team', [
        ('Provision HSM cluster (minimum 2 HSMs for HA).',
         ['Option A - Thales Luna Network HSM 7: rack, cable, initialize, partition for QBITEL Bridge',
          'Option B - AWS CloudHSM: provision cluster in payment VPC, configure PKCS#11 client',
          'Option C - Azure Managed HSM: provision in Azure payment subscription',
          'Generate root key material within HSM boundary - never exported'],
         'REQUIRED'),
        ('Configure HSM network partitions and access controls.',
         ['Create dedicated QBITEL Bridge application partition',
          'Configure HSM admin credentials in HSM-backed vault (not plaintext)',
          'Enable FIPS 140-3 Level 3 operation mode',
          'Configure HSM audit logging to tamper-evident syslog'],
         'REQUIRED'),
        ('Deploy QBITEL Bridge main appliances (active-active HA pair).',
         ['Rack-mount or VM deployment in payment DMZ',
          'Configure active-active clustering with synchronous state replication',
          'Verify PKCS#11 connectivity to HSM cluster',
          'Configure management interface with RBAC and MFA'],
         'REQUIRED'),
        ('Configure network segmentation for PCI cardholder data environment.',
         ['Verify QBITEL Bridge placement between card network gateway and core banking',
          'Confirm no direct cardholder data flows bypass QBITEL Bridge',
          'Configure ACLs to restrict QBITEL Bridge management access',
          'Document network topology changes for PCI-DSS scoping'],
         'REQUIRED'),
        ('Enable database audit logging for DB2 and IMS DB.',
         ['Enable DB2 audit policy for QBITEL Bridge service account connections',
          'Configure DB2 DRDA logging for connection authentication events',
          'Enable IMS Connect audit for remote connection events if applicable'],
         'RECOMMENDED'),
        ('Configure SIEM integration (Splunk / QRadar / Sentinel).',
         ['Install QBITEL Bridge Splunk app or configure CEF/LEEF output',
          'Configure QBITEL Bridge event forwarding to SIEM HEC/syslog',
          'Validate PCI-DSS and DORA dashboard data flow',
          'Test alert delivery to SOC ticketing system'],
         'REQUIRED'),
        ('Conduct integration testing in passive (read-only) mode.',
         ['Verify QBITEL Bridge receives all protocol flows from SPAN',
          'Verify HSM operations are functional (key generation, PKCS#11 test)',
          'Verify SIEM receives events from QBITEL Bridge',
          'Verify compliance dashboards populate correctly'],
         'REQUIRED'),
        ('Staff training: operations team, SOC analysts, compliance team.',
         ['QBITEL Bridge operator training: 1-day hands-on session',
          'SOC analyst training: alert triage, playbook execution, escalation',
          'Compliance team training: evidence portal, report generation'],
         'REQUIRED'),
    ], styles, 'Phase 2 introduces QBITEL Bridge into the network but in passive mode only. No traffic modification until Phase 3 activation.')

    S.append(PageBreak())

    # Phase 3: Security Policy Configuration
    phase_block(S, '3', 'Security Policy Configuration', 'Weeks 4-5 (parallel with Phase 2)', 'Solutions Architect + Compliance Team', [
        ('Configure PCI-DSS 4.0 security zones and data classification policies.',
         ['Define cardholder data environment (CDE) scope in QBITEL Bridge policy engine',
          'Configure field-level sensitivity classification for ISO 8583 (PAN, CVV2, track data)',
          'Set encryption policy: ML-KEM-768 minimum for all CDE traffic',
          'Enable PCI-DSS compliance evidence collection (Requirements 3, 4, 8, 10)'],
         'REQUIRED'),
        ('Configure DORA ICT resilience policies.',
         ['Define DORA major incident thresholds (transaction failure rate, latency SLA breach)',
          'Configure automatic DORA incident classification (critical, significant, major)',
          'Set DORA incident notification contacts (CISO, compliance, regulator email)',
          'Enable DORA Article 26 resilience testing schedule'],
         'REQUIRED'),
        ('Configure SWIFT CSP security policies.',
         ['Set SWIFT environment boundary policy: only SWIFT Alliance Access IPs in SWIFT zone',
          'Configure SWIFT message signing policy: ML-DSA on all outbound MT/MX',
          'Enable SWIFT anomaly detection model with institution-specific baseline',
          'Set SWIFT CSP control evidence collection schedule'],
         'REQUIRED'),
        ('Configure cryptographic agility policies.',
         ['Document all active cipher suites in QBITEL Bridge policy registry (PCI-DSS 12.3.3)',
          'Set algorithm deprecation schedule: RSA-2048 replacement timeline',
          'Enable automatic algorithm upgrade notification',
          'Configure quantum risk scoring threshold for mandatory escalation'],
         'REQUIRED'),
        ('Configure autonomous incident response playbooks.',
         ['ISO 8583 velocity anomaly: auto-block + SOC alert',
          'SWIFT BIC deviation: auto-hold + CISO notification',
          'TN3270e unauthorized session: auto-terminate + audit log',
          'FIX sequence replay detected: auto-block + trading desk alert',
          'HSM tamper event: full system lockdown + emergency escalation'],
         'REQUIRED'),
        ('Conduct policy review with compliance team and legal.',
         ['Review autonomous response actions for legal and regulatory acceptability',
          'Confirm GDPR data minimization in logging configuration',
          'Confirm DORA incident classification thresholds with compliance officer'],
         'REQUIRED'),
    ], styles, 'Security policies must be reviewed and approved by compliance and legal before Phase 4 activation begins.')

    S.append(PageBreak())

    # Phase 4: Payment Rail Protection
    phase_block(S, '4', 'Payment Rail Protection', 'Weeks 6-10', 'Solutions Architect + Head of Payments + Network Team', [
        ('Activate ISO 8583 field-level encryption (scheduled maintenance window).',
         ['Pre-activation test: verify ISO 8583 proxy in pass-through mode with zero errors',
          'Cutover window: activate ML-KEM field encryption during off-peak (Sunday 02:00-04:00)',
          'Post-activation validation: end-to-end ISO 8583 test transaction through all schemes',
          'Monitor for 24 hours before confirming production stability'],
         'CRITICAL'),
        ('Validate Visa and Mastercard scheme format compliance post-activation.',
         ['Run Visa Base I test suite against QBITEL Bridge proxy',
          'Run Mastercard Global Clearing test suite',
          'Confirm ISO 8583 MTI and bit-map structure unchanged',
          'Confirm PIN block encryption compatibility with payment HSM'],
         'REQUIRED'),
        ('Activate SWIFT MT/MX PQC message signing.',
         ['Pre-activation: test ML-DSA signing on SWIFT test message with Alliance Access',
          'Coordinate with SWIFT Service Bureau for signing capability notification',
          'Activate ML-DSA signing on outbound SWIFT MT103, MT202, MT515',
          'Activate ML-KEM key exchange for SWIFT Alliance Access connectivity',
          'Monitor SWIFT NAK rates for 24 hours post-activation'],
         'CRITICAL'),
        ('Activate FedWire and FedNow protection.',
         ['Notify Federal Reserve operational contact of PQC capability activation',
          'Activate PQC-enhanced TLS for Fedline Advantage connections',
          'Activate ML-DSA signing for FedNow ISO 20022 pacs.008 messages',
          'Validate FedNow 10-second processing SLA is maintained'],
         'REQUIRED'),
        ('Activate ACH/NACHA file-level encryption.',
         ['Configure SFTP/MFT integration for PQC-protected NACHA file transmission',
          'Activate AES-256-GCM encryption with ML-KEM key exchange for ACH files',
          'Validate NACHA format integrity post-encryption with ACH operator',
          'Test both same-day ACH and standard ACH batch processing'],
         'REQUIRED'),
        ('Activate CHIPS PQC authentication if applicable.',
         ['Coordinate with The Clearing House for PQC capability activation',
          'Activate ML-KEM key wrapping for CHIPS participant authentication'],
         'RECOMMENDED'),
        ('Conduct 10,000 TPS performance validation under load.',
         ['Run 60-minute sustained load test at 10,000 TPS with ML-KEM active',
          'Measure p50, p95, p99 latency distribution - target p99 <50ms',
          'Measure HSM operation rate - target <20,000 ops/sec (within Luna HSM 7 spec)',
          'Confirm zero transaction errors during load test',
          'Document performance results for PCI-DSS evidence'],
         'REQUIRED'),
        ('Train fraud analytics model on 90-day historical transaction data.',
         ['Load 90-day historical ISO 8583 transaction sample (anonymized)',
          'Train fraud detection model with institution-specific baseline',
          'Set fraud alert thresholds with Head of Fraud Risk',
          'Enable real-time fraud analytics in advisory mode (no auto-block) for first 30 days'],
         'RECOMMENDED'),
    ], styles, 'All payment rail activation steps require a pre-approved change request and out-of-hours maintenance window.')

    S.append(PageBreak())

    # Phase 5: Mainframe & Legacy Shield
    phase_block(S, '5', 'Mainframe & Legacy Shield', 'Weeks 8-12', 'Solutions Architect + Mainframe Team', [
        ('Deploy TN3270e PQC proxy for branch teller network.',
         ['Configure QBITEL Bridge TN3270e proxy between teller LAN and CICS region',
          'Test with 10 pilot teller workstations before full rollout',
          'Verify CICS transaction codes function correctly through PQC proxy',
          'Verify 3270 screen rendering is unchanged for teller operators',
          'Rollout to all branch teller workstations in regional batches'],
         'REQUIRED'),
        ('Deploy CICS transaction signing for external-facing transactions.',
         ['Identify CICS transactions handling external payment data',
          'Configure QBITEL Bridge CICS gateway interception for target transactions',
          'Test COMMAREA data integrity through signing layer',
          'Monitor CICS response times for <2ms overhead compliance'],
         'REQUIRED'),
        ('Activate DB2 DRDA PQC upgrade.',
         ['Configure QBITEL Bridge DRDA proxy between distributed application servers and DB2',
          'Test DB2 application connectivity through DRDA proxy (JDBC test suite)',
          'Verify query response times are within <8ms overhead budget',
          'Monitor DB2 connection pool behavior through proxy'],
         'REQUIRED'),
        ('Activate IBM MQ message signing.',
         ['Configure QBITEL Bridge MQ interception for queues carrying payment data',
          'Define MQ signing policy: which queue managers and channels require ML-DSA',
          'Test MQ message put/get through signing layer',
          'Verify MQ message format and MQMD headers are preserved'],
         'REQUIRED'),
        ('Deploy AS/400 (IBM i) protection if applicable.',
         ['Configure TN5250e PQC proxy for IBM i terminal sessions',
          'Configure QBITEL Bridge for IBM i socket and JDBC connections',
          'Test RPG application connectivity through PQC proxy'],
         'RECOMMENDED'),
        ('Validate mainframe performance: MIPS and response time.',
         ['Confirm MIPS consumption increase is <0.1% (zero direct mainframe impact)',
          'Confirm TN3270e session setup latency increase is <5ms',
          'Confirm CICS transaction response time increase is <2ms',
          'Confirm DB2 query response time increase is <8ms',
          'Document performance validation for compliance evidence'],
         'REQUIRED'),
        ('VSAM file encryption (scheduled maintenance window).',
         ['Identify VSAM files containing sensitive payment data',
          'Schedule VSAM encryption activation during batch window',
          'Test COBOL program VSAM I/O through encryption layer',
          'Verify JCL and utility access to VSAM files is maintained'],
         'RECOMMENDED'),
    ], styles, 'Mainframe deployments are zero code change. All interception occurs at network and gateway layers outside z/OS.')

    S.append(PageBreak())

    # Phase 6: Integration Deployment
    phase_block(S, '6', 'Integration Deployment', 'Weeks 6-12 (parallel)', 'Solutions Architect + Integration Team', [
        ('Complete core banking system integration (Temenos/Finastra/FIS/Fiserv/Oracle).',
         ['Configure REST API integration for core banking event feeds',
          'Test payment message flow from core banking through QBITEL Bridge to payment gateway',
          'Verify account balance inquiry flows are protected',
          'Validate core banking vendor test certification (if required by contract)'],
         'REQUIRED'),
        ('Complete HSM integration validation.',
         ['Run PKCS#11 compliance test suite against HSM cluster',
          'Verify ML-KEM key generation within HSM boundary',
          'Verify ML-DSA signing operations with HSM-stored keys',
          'Test HSM failover: pull primary HSM network cable, verify secondary takes over',
          'Verify key rotation completes without payment transaction interruption'],
         'REQUIRED'),
        ('Complete SIEM/SOAR integration and playbook validation.',
         ['Validate all QBITEL Bridge alert types are received in SIEM',
          'Run playbook tests: inject test ISO 8583 anomaly, verify auto-response',
          'Run playbook tests: inject test SWIFT anomaly, verify alert chain',
          'Confirm ticket creation in ServiceNow/JIRA for all P0/P1 alerts',
          'Confirm DORA incident classification is correct for each alert type'],
         'REQUIRED'),
        ('Complete fraud analytics integration with fraud operations team.',
         ['Deliver fraud analytics dashboard training to Head of Fraud Risk',
          'Configure fraud alert routing to fraud operations email/SMS',
          'Set dispute management system integration for auto-flagged transactions',
          'Define escalation path for fraud ML model threshold tuning'],
         'RECOMMENDED'),
        ('Complete identity and access management (IAM) integration.',
         ['Configure CyberArk/BeyondTrust for QBITEL Bridge privileged access',
          'Integrate QBITEL Bridge management with corporate SSO (Okta/Azure AD)',
          'Configure QBITEL Bridge RBAC: Admin, Operator, Auditor, Read-Only roles',
          'Enable MFA requirement for all QBITEL Bridge management access'],
         'REQUIRED'),
        ('Complete compliance platform integration.',
         ['Configure GRC platform (RSA Archer/ServiceNow GRC/MetricStream) data feeds',
          'Validate PCI-DSS evidence package population in GRC',
          'Validate DORA ICT risk register population',
          'Configure compliance report scheduling (daily, weekly, monthly)'],
         'REQUIRED'),
    ], styles)

    S.append(PageBreak())

    # Phase 7: Monitoring & Alerting
    phase_block(S, '7', 'Monitoring & Alerting', 'Weeks 10-13', 'Solutions Architect + SOC Team', [
        ('Configure real-time transaction anomaly monitoring.',
         ['Set ISO 8583 velocity thresholds per card BIN and merchant category',
          'Set SWIFT MT103 amount deviation alerts (>3 standard deviations from 90-day baseline)',
          'Set TN3270e concurrent session anomaly alerts (>N sessions per operator ID)',
          'Set FIX sequence gap and replay alerts'],
         'REQUIRED'),
        ('Configure PCI-DSS compliance monitoring dashboards.',
         ['Enable real-time PCI-DSS Requirement 3, 4, 8, 10 control status dashboard',
          'Configure encryption coverage report: % of CDE traffic with active PQC',
          'Set alert for any detected cleartext cardholder data in monitored segments',
          'Configure daily PCI-DSS compliance summary email to CISO'],
         'REQUIRED'),
        ('Configure DORA incident reporting automation.',
         ['Enable DORA major incident auto-detection based on agreed thresholds',
          'Configure automated DORA incident report draft generation',
          'Set 4-hour notification SLA for DORA major incident classification',
          'Test DORA incident end-to-end from detection to report generation'],
         'REQUIRED'),
        ('Configure HSM health monitoring.',
         ['Enable HSM capacity alerts (>80% operation utilization)',
          'Enable key expiration alerts (30-day advance warning)',
          'Enable HSM tamper event escalation to CISO and HSM vendor',
          'Configure HSM firmware version monitoring'],
         'REQUIRED'),
        ('Configure cryptographic agility monitoring.',
         ['Enable alerts for detection of deprecated cipher suites (DES, 3DES, RC4, MD5)',
          'Enable alerts for RSA-1024 key detection in any protocol stream',
          'Monitor algorithm usage distribution report: % traffic on PQC vs classical'],
         'REQUIRED'),
        ('Conduct NOC team tabletop exercise.',
         ['Scenario 1: ISO 8583 payment rail attack - verify detection and response',
          'Scenario 2: SWIFT anomaly (potential fraud) - verify detection and escalation',
          'Scenario 3: HSM hardware failure - verify HA failover and payment continuity',
          'Scenario 4: DORA major incident - verify report generation and regulator notification'],
         'REQUIRED'),
    ], styles, 'Monitoring must be operational for minimum 2 weeks before Phase 8 compliance validation begins.')

    S.append(PageBreak())

    # Phase 8: Compliance Validation
    phase_block(S, '8', 'Compliance Validation', 'Weeks 12-14', 'Solutions Architect + Compliance Team + QSA', [
        ('PCI-DSS 4.0 evidence package preparation.',
         ['Requirement 3.5: Key management evidence (HSM custody, rotation logs)',
          'Requirement 4.2: Strong cryptography evidence (ML-KEM/ML-DSA deployment)',
          'Requirement 6.4: Web application protection evidence',
          'Requirement 8.6: MFA management evidence',
          'Requirement 10.2: Audit log evidence (tamper-evident, signed)',
          'Requirement 12.3.3: Cryptographic inventory and agility roadmap document'],
         'REQUIRED'),
        ('Provide QSA with read-only dashboard access for evidence review.',
         ['Create QSA-role user account with read-only compliance portal access',
          'Provide QSA with 90-day historical compliance dashboard data',
          'Schedule QSA walkthrough of QBITEL Bridge architecture (2 hours)'],
         'REQUIRED'),
        ('DORA ICT resilience test execution (Article 26).',
         ['Execute threat-led penetration test of ISO 8583 payment rail (QBITEL red team)',
          'Execute SWIFT message manipulation simulation',
          'Execute mainframe TN3270e session interception simulation',
          'Document resilience test results in DORA format',
          'Deliver DORA Article 26 test report to compliance officer'],
         'REQUIRED'),
        ('SWIFT CSP attestation package preparation.',
         ['Generate SWIFT CSP control evidence report from QBITEL Bridge compliance portal',
          'Map QBITEL Bridge controls to SWIFT CSP 2025 mandatory and advisory controls',
          'Prepare CSP self-attestation supporting documentation'],
         'REQUIRED'),
        ('DORA ICT risk register validation.',
         ['Review auto-populated DORA ICT risk register for completeness',
          'Validate ICT third-party risk entries for QBITEL Bridge and other ICT providers',
          'Obtain compliance officer sign-off on DORA risk register'],
         'REQUIRED'),
        ('Basel III operational risk data validation.',
         ['Verify AMA loss data collection is operational',
          'Validate KRI feeds to operational risk platform',
          'Confirm BCBS 239 data lineage documentation for payment data flows'],
         'RECOMMENDED'),
        ('Internal audit review.',
         ['Schedule internal audit review of QBITEL Bridge deployment',
          'Provide internal audit with SOC 2 Type II report and FIPS 140-3 certificate',
          'Facilitate internal audit access to compliance evidence portal'],
         'REQUIRED'),
    ], styles)

    S.append(PageBreak())

    # Phase 9: UAT & Performance Testing
    phase_block(S, '9', 'UAT & Performance Testing', 'Weeks 14-15', 'Solutions Architect + QA + Payment Operations', [
        ('Execute full 10,000 TPS sustained load test.',
         ['Run 60-minute sustained load at 10,000 TPS with all PQC protections active',
          'Measure p50/p95/p99 latency at peak load - target p99 <50ms',
          'Measure error rate - target 0 transaction errors',
          'Measure HSM operation utilization - target <80% capacity',
          'Document results and sign off with Head of Payments'],
         'REQUIRED'),
        ('Validate <50ms latency SLA across all protocol streams.',
         ['ISO 8583 field-level encryption: measure end-to-end latency for 1,000 sample transactions',
          'SWIFT MT103 with ML-DSA signing: measure added latency',
          'TN3270e session setup: measure time-to-screen for 100 sessions',
          'DB2 DRDA query: measure query response time delta vs baseline',
          'FIX order message signing: measure microsecond-level latency impact'],
         'REQUIRED'),
        ('Execute HA failover test.',
         ['With active payment traffic flowing, physically disconnect primary QBITEL Bridge node',
          'Verify failover completes in <30 seconds',
          'Verify zero ISO 8583 transaction errors during failover',
          'Verify SWIFT connectivity maintained during failover',
          'Restore primary node and verify re-synchronization'],
         'REQUIRED'),
        ('Execute HSM failover test.',
         ['With payment traffic flowing, disconnect primary HSM',
          'Verify QBITEL Bridge fails over to secondary HSM automatically',
          'Verify payment processing continues without interruption',
          'Verify HSM key material is consistent on secondary'],
         'REQUIRED'),
        ('Execute bypass mode test.',
         ['Simulate QBITEL Bridge software failure (kill primary process)',
          'Verify bypass mode activates within configured timeout',
          'Verify payment traffic continues in bypass mode (unencrypted pass-through)',
          'Verify alert generated for bypass mode activation',
          'Verify QBITEL Bridge resumes active mode after process restart'],
         'REQUIRED'),
        ('Execute fraud analytics validation.',
         ['Inject 100 synthetic fraudulent ISO 8583 transactions',
          'Verify fraud detection model flags >90% of injected fraud',
          'Verify false positive rate <0.5% on legitimate transaction baseline',
          'Validate fraud alert delivery to fraud operations team'],
         'RECOMMENDED'),
        ('User acceptance sign-off from all stakeholders.',
         ['Head of Payments: payment rail protection and performance sign-off',
          'CISO: security controls and compliance evidence sign-off',
          'Head of Compliance: PCI-DSS and DORA evidence package sign-off',
          'Head of Technology: infrastructure and SLA sign-off',
          'Internal Audit: control evidence sign-off'],
         'REQUIRED'),
    ], styles)

    S.append(PageBreak())

    # Phase 10: Go-Live & Handover
    phase_block(S, '10', 'Go-Live & Handover', 'Week 16', 'QBITEL NOC + Client Operations Team', [
        ('Execute go-live change request and CAB approval.',
         ['Submit formal change request for production go-live',
          'Obtain Change Advisory Board (CAB) approval',
          'Notify all stakeholders of go-live date and maintenance window',
          'Prepare rollback procedure document'],
         'REQUIRED'),
        ('Activate all remaining protocol protections in sequence.',
         ['Final activation checklist walkthrough with QBITEL engineer',
          'Confirm all payment rails are in active PQC mode (not pass-through)',
          'Confirm all SWIFT connections are in signed mode',
          'Confirm all TN3270e sessions are in encrypted mode'],
         'REQUIRED'),
        ('Deliver operations runbook to client NOC team.',
         ['QBITEL Bridge startup/shutdown procedures',
          'HA failover procedures (manual and automatic)',
          'HSM key rotation procedures',
          'Incident response playbooks (P0 through P3)',
          'DORA incident classification and reporting procedures',
          'PCI-DSS evidence generation procedures',
          'Vendor escalation contacts and SLA reference'],
         'REQUIRED'),
        ('Establish 24/7 NOC support handover.',
         ['QBITEL NOC contact numbers and escalation path',
          'Client NOC primary and secondary contacts registered with QBITEL',
          'PagerDuty/OpsGenie integration for P0 alert escalation',
          'Regular NOC sync schedule (weekly for first 3 months)'],
         'REQUIRED'),
        ('Deliver training to all operational teams.',
         ['QBITEL Bridge operator certification: 1-day final assessment',
          'SOC analyst refresher: 4-hour session on production alert handling',
          'Compliance team: 2-hour session on evidence portal and report generation',
          'Executive dashboard: 1-hour session for CISO/CTO on management reporting'],
         'REQUIRED'),
        ('Complete project sign-off and handover documentation.',
         ['Project completion certificate signed by client project owner',
          'As-built architecture diagram delivered and verified',
          'All change requests closed in client ITSM system',
          'Warranty period start date confirmed (default: 90 days)'],
         'REQUIRED'),
        ('Schedule 30-day and 90-day post-go-live health checks.',
         ['30-day check: performance review, compliance evidence review, alert tuning',
          '90-day check: fraud model performance review, compliance evidence package review'],
         'REQUIRED'),
    ], styles, 'Go-live requires signed UAT approval from all Phase 9 stakeholders before proceeding.')

    S.append(PageBreak())

    # Appendix A: Performance SLAs
    S.append(Paragraph('Appendix A: Performance SLAs', ParagraphStyle('app_h', fontName='Helvetica-Bold',
                        fontSize=14, textColor=NAVY, spaceAfter=8)))
    sla_data = [
        [Paragraph('Metric', styles['table_h']),
         Paragraph('SLA Target', styles['table_h']),
         Paragraph('Measurement Method', styles['table_h']),
         Paragraph('Breach Consequence', styles['table_h'])],
        ['System availability', '99.999% (5 nines)', 'QBITEL NOC uptime monitoring', 'Credit: 10x monthly subscription'],
        ['ISO 8583 throughput', '10,000 TPS minimum', 'Monthly load test', 'Credit: 5x daily subscription'],
        ['ISO 8583 p99 latency', '<50ms end-to-end', 'Continuous p99 measurement', 'Credit: 2x daily subscription'],
        ['SWIFT added latency', '<8ms per message', 'SWIFT message timing logs', 'Credit: 2x daily subscription'],
        ['FIX added latency', '<50 microseconds', 'FIX session timing', 'Credit: 2x daily subscription'],
        ['TN3270e session setup', '<5ms added', 'Session timing logs', 'Notification only'],
        ['HA failover time', '<30 seconds', 'Quarterly failover test', 'Credit: 5x daily subscription'],
        ['P0 incident response', '<5 min engineer engaged', 'NOC escalation logs', 'Credit: 10x daily subscription'],
        ['Key rotation', 'Zero downtime', 'Rotation event logs', 'Credit: 5x daily subscription'],
    ]
    sla_tbl = Table(sla_data, colWidths=[CONTENT_W * 0.22, CONTENT_W * 0.18, CONTENT_W * 0.3, CONTENT_W * 0.3])
    sla_tbl.setStyle(tbl_style())
    S.append(sla_tbl)
    sp(12)

    # Appendix B: Rollback Procedures
    S.append(Paragraph('Appendix B: Rollback Procedures', ParagraphStyle('app_h', fontName='Helvetica-Bold',
                        fontSize=14, textColor=NAVY, spaceAfter=8)))
    rb_data = [
        [Paragraph('Scenario', styles['table_h']),
         Paragraph('Rollback Action', styles['table_h']),
         Paragraph('Time to Rollback', styles['table_h']),
         Paragraph('Payment Impact', styles['table_h'])],
        ['ISO 8583 PQC failure', 'Set payment-rail policy to PASSTHROUGH mode via CLI', '<30 seconds', 'Zero - pass-through maintains connectivity'],
        ['SWIFT signing failure', 'Disable ML-DSA signing policy for SWIFT zone', '<30 seconds', 'SWIFT messages continue unsigned'],
        ['TN3270e proxy failure', 'Set TN3270e policy to DIRECT CONNECT mode', '<60 seconds', 'Teller sessions reconnect directly to CICS'],
        ['Full QBITEL Bridge failure', 'Remove inline deployment; bypass with patch cable', '<5 minutes', 'Full bypass - no PQC protection'],
        ['HSM failure (primary)', 'Automatic failover to secondary HSM', '<10 seconds', 'Zero - automatic'],
        ['HSM failure (both)', 'Activate emergency key cache from encrypted backup', '<2 minutes', 'Degraded - local key cache only'],
    ]
    rb_tbl = Table(rb_data, colWidths=[CONTENT_W * 0.25, CONTENT_W * 0.35, CONTENT_W * 0.18, CONTENT_W * 0.22])
    rb_tbl.setStyle(tbl_style())
    S.append(rb_tbl)
    sp(12)

    # Appendix C: Escalation Matrix
    S.append(Paragraph('Appendix C: Escalation Matrix', ParagraphStyle('app_h', fontName='Helvetica-Bold',
                        fontSize=14, textColor=NAVY, spaceAfter=8)))
    esc_data = [
        [Paragraph('Incident Priority', styles['table_h']),
         Paragraph('Definition', styles['table_h']),
         Paragraph('Response Time', styles['table_h']),
         Paragraph('QBITEL Escalation', styles['table_h']),
         Paragraph('Client Escalation', styles['table_h'])],
        ['P0 - Critical', 'Active payment rail attack or complete system failure',
         '<5 min engineer', 'NOC Lead + VP Engineering', 'CISO + CTO + Head of Payments'],
        ['P1 - High', 'SWIFT anomaly, significant performance degradation, HSM alarm',
         '<15 min engineer', 'Senior NOC Engineer', 'CISO + Head of Payments'],
        ['P2 - Medium', 'Compliance violation, non-critical alert, key rotation issue',
         '<1 hour response', 'NOC Engineer', 'Head of Compliance + IT Manager'],
        ['P3 - Low', 'Informational alerts, performance optimization, feature requests',
         'Next business day', 'Support Engineer', 'IT Operations Manager'],
        ['Regulatory', 'DORA major incident, PCI-DSS breach notification required',
         'Immediate parallel', 'QBITEL Compliance + NOC', 'CISO + CCO + Legal + Regulator'],
    ]
    esc_tbl = Table(esc_data, colWidths=[CONTENT_W * 0.15, CONTENT_W * 0.27, CONTENT_W * 0.14, CONTENT_W * 0.22, CONTENT_W * 0.22])
    esc_tbl.setStyle(tbl_style())
    S.append(esc_tbl)
    sp(12)

    return S


def build_pdf():
    import os
    output = '/Users/prabakarankannan/qbitel/docs/brochures/QBITEL_Banking_Deployment_Checklist.pdf'
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
