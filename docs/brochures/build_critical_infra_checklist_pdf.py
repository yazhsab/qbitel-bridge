"""Build QBITEL Bridge Critical Infrastructure Deployment Checklist - PDF"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
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
RED_HL     = HexColor('#C0392B')
GREEN_HL   = HexColor('#27AE60')
ORANGE_HL  = HexColor('#E67E22')
LIGHT_RED  = HexColor('#FADBD8')
LIGHT_GREEN= HexColor('#D5F5E3')
LIGHT_ORG  = HexColor('#FDEBD0')

PAGE_W, PAGE_H = letter
MARGIN = 0.85 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

def sp(n): return Spacer(1, n)


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
        # Navy background
        c.setFillColor(NAVY)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)

        # Gold left badge (58px wide)
        c.setFillColor(GOLD)
        c.rect(0, 0, 58, self.h, fill=1, stroke=0)

        # PHASE label in badge
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 6.5)
        lw = c.stringWidth('PHASE', 'Helvetica-Bold', 6.5)
        c.drawString(29 - lw/2, self.h - 14, 'PHASE')

        # Phase number large
        c.setFont('Helvetica-Bold', 20)
        nw = c.stringWidth(str(self.phase_num), 'Helvetica-Bold', 20)
        c.drawString(29 - nw/2, self.h - 36, str(self.phase_num))

        # Teal right accent (5px)
        c.setFillColor(TEAL)
        c.rect(self.w - 5, 0, 5, self.h, fill=1, stroke=0)

        # Title in white
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 11)
        c.drawString(70, self.h - 20, self.title)

        # Duration badge
        dur_text = self.duration
        dur_w = c.stringWidth(dur_text, 'Helvetica', 7.5) + 12
        dur_x = 70

        c.setFillColor(TEAL)
        c.roundRect(dur_x, 8, dur_w, 16, 3, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica', 7.5)
        c.drawString(dur_x + 6, 13, dur_text)

        # Owner badge
        own_text = 'Owner: ' + self.owner
        own_w = c.stringWidth(own_text, 'Helvetica', 7.5) + 12
        own_x = dur_x + dur_w + 8

        c.setFillColor(LIGHT_NAVY)
        c.roundRect(own_x, 8, own_w, 16, 3, fill=1, stroke=0)
        c.setFillColor(GOLD)
        c.setFont('Helvetica', 7.5)
        c.drawString(own_x + 6, 13, own_text)


class CheckItem(Flowable):
    def __init__(self, text, sub_items=None, highlight=None, width=None):
        super().__init__()
        self.text = text
        self.sub_items = sub_items or []
        self.highlight = highlight  # None, 'critical', 'pass', 'warning'
        self.w = width or CONTENT_W
        # Calculate height
        chars_per_line = int((self.w - 55) / 5.2)
        main_lines = max(1, len(text) // chars_per_line + 1)
        sub_h = len(self.sub_items) * 16
        self.h = main_lines * 14 + sub_h + 12

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv

        # Background color based on highlight
        if self.highlight == 'critical':
            bg = LIGHT_RED
            border = RED_HL
        elif self.highlight == 'pass':
            bg = LIGHT_GREEN
            border = GREEN_HL
        elif self.highlight == 'warning':
            bg = LIGHT_ORG
            border = ORANGE_HL
        else:
            bg = WHITE_C
            border = MID_GREY

        c.setFillColor(bg)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)

        # Left accent stripe for highlights
        if self.highlight:
            c.setFillColor(border)
            c.rect(0, 0, 3, self.h, fill=1, stroke=0)

        # Checkbox (10x10, navy border)
        box_x = 10
        box_y = self.h - 18
        c.setStrokeColor(NAVY)
        c.setFillColor(WHITE_C)
        c.rect(box_x, box_y, 10, 10, fill=1, stroke=1)

        # Text to right of checkbox
        text_x = 28
        text_y = self.h - 16
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica-Bold' if self.highlight == 'critical' else 'Helvetica', 8.5)

        # Word wrap main text
        words = self.text.split()
        lines = []
        current = ''
        max_chars = int((self.w - text_x - 8) / 5.0)
        for word in words:
            test = (current + ' ' + word).strip()
            if len(test) <= max_chars:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

        for i, line in enumerate(lines):
            y = text_y - i * 13
            if y < (len(self.sub_items) * 16 + 4):
                break
            c.drawString(text_x, y, line)

        # Highlight label
        if self.highlight == 'critical':
            label = 'CRITICAL'
            c.setFillColor(RED_HL)
        elif self.highlight == 'pass':
            label = 'REQUIRED'
            c.setFillColor(GREEN_HL)
        elif self.highlight == 'warning':
            label = 'IMPORTANT'
            c.setFillColor(ORANGE_HL)
        else:
            label = None

        if label:
            lw = c.stringWidth(label, 'Helvetica-Bold', 6.5)
            lx = self.w - lw - 12
            c.roundRect(lx - 4, box_y, lw + 8, 10, 2, fill=1, stroke=0)
            c.setFillColor(WHITE_C)
            c.setFont('Helvetica-Bold', 6.5)
            c.drawString(lx, box_y + 2, label)

        # Sub-items indented
        if self.sub_items:
            c.setFont('Helvetica', 7.5)
            c.setFillColor(MID_GREY)
            base_y = len(self.sub_items) * 16 + 2
            for j, sub in enumerate(self.sub_items):
                sy = base_y - j * 16
                # Small dash bullet
                c.drawString(32, sy, '-  ' + sub)

        # Bottom border
        c.setStrokeColor(MID_GREY)
        c.setLineWidth(0.3)
        c.line(0, 0, self.w, 0)


class SignoffBlock(Flowable):
    def __init__(self, roles, width=None):
        super().__init__()
        self.roles = roles
        self.w = width or CONTENT_W
        self.h = 16 + len(roles) * 28

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(NAVY)
        c.rect(0, self.h - 16, self.w, 16, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 8)
        c.drawString(10, self.h - 11, 'PHASE SIGN-OFF — REQUIRED BEFORE PROCEEDING TO NEXT PHASE')

        for i, role in enumerate(self.roles):
            y = self.h - 16 - (i + 1) * 28
            c.setFillColor(LIGHT_BG)
            c.rect(0, y, self.w, 26, fill=1, stroke=0)
            c.setStrokeColor(MID_GREY)
            c.setLineWidth(0.5)
            c.rect(0, y, self.w, 26, fill=0, stroke=1)
            c.setFillColor(DARK_TEXT)
            c.setFont('Helvetica-Bold', 8)
            c.drawString(10, y + 16, role + ':')
            # Signature line
            sig_x = 10 + c.stringWidth(role + ':', 'Helvetica-Bold', 8) + 10
            c.setStrokeColor(MID_GREY)
            c.line(sig_x, y + 16, self.w * 0.55, y + 16)
            c.setFont('Helvetica', 7.5)
            c.drawString(self.w * 0.57, y + 16, 'Date:')
            c.line(self.w * 0.57 + 35, y + 16, self.w - 10, y + 16)
            c.setFillColor(MID_GREY)
            c.setFont('Helvetica', 6.5)
            c.drawString(10, y + 5, 'Signature')


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H-30, PAGE_W, 30, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C); canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(MARGIN, PAGE_H-20, 'QBITEL BRIDGE - CRITICAL INFRASTRUCTURE DEPLOYMENT CHECKLIST')
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(PAGE_W-MARGIN, PAGE_H-20, f'Page {doc.page}')
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, PAGE_W, 22, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C); canvas.setFont('Helvetica', 7.5)
    canvas.drawString(MARGIN, 7, 'Confidential - For Authorized Recipients Only  |  (c) 2026 QBITEL. All Rights Reserved.')
    contact_str = 'enterprise@qbitel.com  |  bridge.qbitel.com'
    cw = canvas.stringWidth(contact_str, 'Helvetica', 7.5)
    canvas.drawString(PAGE_W-MARGIN-cw, 7, contact_str)
    canvas.restoreState()


def draw_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY); canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(TEAL); canvas.rect(0, PAGE_H*0.55, PAGE_W, PAGE_H*0.45, fill=1, stroke=0)
    canvas.setFillColor(GOLD); canvas.rect(0, PAGE_H*0.55, PAGE_W, 4, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C); canvas.setFont('Helvetica-Bold', 28)
    canvas.drawString(MARGIN, PAGE_H*0.55 + PAGE_H*0.45*0.78, 'QBITEL BRIDGE')
    canvas.setFillColor(GOLD); canvas.setFont('Helvetica-Bold', 16)
    canvas.drawString(MARGIN, PAGE_H*0.55 + PAGE_H*0.45*0.62, 'Critical Infrastructure & ICS/SCADA')
    canvas.setFillColor(WHITE_C); canvas.setFont('Helvetica-Bold', 22)
    canvas.drawString(MARGIN, PAGE_H*0.55 + PAGE_H*0.45*0.44, 'Deployment Checklist')
    canvas.setFont('Helvetica', 13)
    canvas.drawString(MARGIN, PAGE_H*0.55 + PAGE_H*0.45*0.30, '10 Phases | 120-Day Deployment | Zero Downtime')
    canvas.drawString(MARGIN, PAGE_H*0.55 + PAGE_H*0.45*0.18, 'Safety-First | NERC CIP | IEC 62443 | IEC 61508 Compatible')
    stats = [('10', 'Phases'), ('120', 'Days'), ('0', 'Downtime'), ('IEC 61508', 'Safe')]
    box_w = (PAGE_W - 2*MARGIN - 30) / 4
    for i, (val, lbl) in enumerate(stats):
        x = MARGIN + i*(box_w+10); y = 0.12*PAGE_H
        canvas.setFillColor(LIGHT_NAVY); canvas.roundRect(x, y, box_w, 0.1*PAGE_H, 6, fill=1, stroke=0)
        canvas.setFillColor(GOLD); canvas.setFont('Helvetica-Bold', 14)
        vw = canvas.stringWidth(val, 'Helvetica-Bold', 14)
        canvas.drawString(x + box_w/2 - vw/2, y + 0.1*PAGE_H*0.55, val)
        canvas.setFillColor(WHITE_C); canvas.setFont('Helvetica', 8)
        lw = canvas.stringWidth(lbl, 'Helvetica', 8)
        canvas.drawString(x + box_w/2 - lw/2, y + 0.1*PAGE_H*0.25, lbl)
    canvas.setFillColor(TEAL); canvas.rect(0, 0, PAGE_W, 0.08*PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C); canvas.setFont('Helvetica', 9)
    canvas.drawString(MARGIN, 0.04*PAGE_H, 'enterprise@qbitel.com  |  bridge.qbitel.com')
    canvas.restoreState()


def build_checklist(output_path):
    import os; os.chdir('/Users/prabakarankannan/qbitel')
    doc = BaseDocTemplate(output_path, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN)
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id='cover')
    inner_frame = Frame(MARGIN, 0.7*inch, CONTENT_W, PAGE_H - MARGIN - 0.7*inch, id='inner')
    cover_template = PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover)
    inner_template = PageTemplate(id='Inner', frames=[inner_frame], onPage=draw_page)
    doc.addPageTemplates([cover_template, inner_template])

    story = []
    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())

    # Introduction
    body_style = ParagraphStyle('body', fontName='Helvetica', fontSize=9, leading=14, textColor=DARK_TEXT, spaceAfter=6)
    h2_style = ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=11, textColor=NAVY, spaceBefore=8, spaceAfter=4)

    story.append(Paragraph('DEPLOYMENT OVERVIEW', h2_style))
    story.append(Paragraph(
        'This checklist guides the deployment of QBITEL Bridge across critical infrastructure environments. '
        'It is structured in 10 phases spanning 120 days, designed for zero operational downtime. '
        'Each phase requires explicit sign-off from designated roles before proceeding. '
        'Safety system boundaries (IEC 61508 SIL 3/4) are formally documented in Phase 4 '
        'and require Safety Officer written approval before any active protection is enabled.', body_style))
    story.append(sp(6))

    # Phase summary table
    phase_summary = [
        ['Phase', 'Name', 'Duration', 'Key Milestone'],
        ['0', 'Pre-Engagement & Safety Officer Alignment', 'Days 1-5', 'Safety boundaries documented'],
        ['1', 'Passive OT Protocol Discovery', 'Days 6-14', 'Asset inventory complete'],
        ['2', 'Zone & Conduit Mapping (IEC 62443)', 'Days 15-22', 'Zone model approved'],
        ['3', 'Infrastructure Readiness', 'Days 23-32', 'HSM and Ollama deployed'],
        ['4', 'Safety System Identification & Exclusion', 'Days 33-38', 'Safety Officer sign-off'],
        ['5', 'PLC Command Authentication Rollout', 'Days 39-55', 'Auth active on all PLCs'],
        ['6', 'Protocol Protection Activation', 'Days 56-68', 'All protocols protected'],
        ['7', 'NERC CIP Control Mapping', 'Days 69-80', 'Compliance mapped'],
        ['8', 'Monitoring & Alerting Configuration', 'Days 81-90', 'SOC dashboard live'],
        ['9', 'Compliance Validation', 'Days 91-105', 'Evidence package approved'],
        ['10', 'Operational Handover', 'Days 106-120', 'Go-live sign-off'],
    ]
    pt = Table(phase_summary, colWidths=[CONTENT_W*0.07, CONTENT_W*0.42, CONTENT_W*0.18, CONTENT_W*0.33])
    pt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),NAVY), ('TEXTCOLOR',(0,0),(-1,0),WHITE_C),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'), ('FONTSIZE',(0,0),(-1,0),8),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[LIGHT_BG,TABLE_ALT]),
        ('FONTSIZE',(0,1),(-1,-1),7.5), ('GRID',(0,0),(-1,-1),0.5,MID_GREY),
        ('VALIGN',(0,0),(-1,-1),'TOP'), ('LEFTPADDING',(0,0),(-1,-1),5),
        ('TOPPADDING',(0,0),(-1,-1),4), ('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))
    story.append(pt)
    story.append(sp(12))

    # =========================================================
    # PHASE 0
    # =========================================================
    story.append(PhaseHeader('0', 'Pre-Engagement & Safety Officer Alignment', 'Days 1-5', 'Project Manager / CISO'))
    story.append(sp(6))

    phase0_items = [
        ('Identify Asset Owner, Operations Manager, Safety Officer, and NERC CIP Compliance Lead', [], 'critical'),
        ('Complete OT asset inventory (preliminary) from existing documentation', ['Review engineering drawings, P&ID diagrams, network diagrams', 'Identify known PLCs, RTUs, HMIs, historians, SCADA servers'], None),
        ('Review existing NERC CIP AOP/CIP documentation and compliance status', ['Identify current CIP-002 BES Cyber System classification', 'Review last NERC CIP audit findings and remediation status'], None),
        ('Identify safety system boundaries (SIL 3/4 exclusion zones)', ['Document all IEC 61508 certified SIS systems by tag number', 'Identify SIS network segments for passive-only designation'], 'critical'),
        ('Map IT/OT network topology (logical and physical)', ['Document existing firewall rules and DMZ architecture', 'Identify all remote access paths into OT network'], None),
        ('Schedule change windows for active protection phases (Phases 5, 6)', ['Confirm maintenance window schedule for next 90 days', 'Document change freeze periods that must be avoided'], 'warning'),
        ('Obtain project charter approval and stakeholder sign-off', [], 'critical'),
    ]

    for text, subs, hl in phase0_items:
        story.append(CheckItem(text, subs, hl))
        story.append(sp(3))

    story.append(sp(6))
    story.append(SignoffBlock(['Asset Owner', 'Operations Manager', 'Safety Officer', 'NERC CIP Compliance Lead']))
    story.append(sp(12))

    # =========================================================
    # PHASE 1
    # =========================================================
    story.append(PhaseHeader('1', 'Passive OT Protocol Discovery', 'Days 6-14', 'OT Security Engineer'))
    story.append(sp(6))

    phase1_items = [
        ('Deploy passive network tap on target OT segment(s) - read-only, no traffic injection', ['Verify tap is receive-only by design (no transmit capability)', 'Confirm no impact on existing network traffic before proceeding'], 'critical'),
        ('Run QBITEL discovery on target OT segments', ['Modbus TCP/RTU discovery and device enumeration', 'DNP3 master/outstation mapping', 'IEC 61850 GOOSE/SV identification', 'OPC UA server/client mapping'], None),
        ('Generate asset inventory report', ['Every PLC, RTU, HMI, historian, engineering workstation', 'Protocol version, firmware (where detectable), communication partners'], None),
        ('Identify unencrypted and unauthenticated protocol flows', ['Flag all Modbus without authentication', 'Flag all DNP3 without SAv5', 'Flag IEC 61850 GOOSE without HMAC'], 'warning'),
        ('Map BES Cyber Systems per NERC CIP CIP-002', ['Draft BES Cyber System classification for each discovered asset', 'Submit to NERC CIP Compliance Lead for review'], None),
        ('Validate no impact on SCADA polling rates', ['Monitor SCADA polling latency before and during tap deployment', 'Confirm no polling timeouts or communication errors'], 'critical'),
        ('Produce vulnerability assessment and risk prioritization', [], None),
    ]

    for text, subs, hl in phase1_items:
        story.append(CheckItem(text, subs, hl))
        story.append(sp(3))

    story.append(sp(6))
    story.append(SignoffBlock(['OT Security Engineer', 'Operations Manager', 'NERC CIP Compliance Lead']))
    story.append(sp(12))

    # =========================================================
    # PHASE 2
    # =========================================================
    story.append(PhaseHeader('2', 'Zone & Conduit Mapping (IEC 62443)', 'Days 15-22', 'OT Security Architect'))
    story.append(sp(6))

    phase2_items = [
        ('Classify all assets into IEC 62443 security zones', ['Safety Zone: SIS systems (SIL 3/4) - passive only', 'Control Zone: PLCs, RTUs, field controllers', 'Supervisory Zone: SCADA servers, HMIs, historians', 'Enterprise Zone: business network, vendor access'], 'critical'),
        ('Define conduit security requirements per zone boundary', ['Control Zone <-> Supervisory Zone: authenticated, encrypted', 'Supervisory Zone <-> Enterprise: DMZ, strictly filtered', 'Any zone -> Safety Zone: passive monitoring only, no active enforcement'], None),
        ('Document trust boundaries for NERC CIP ESP mapping', ['Map IEC 62443 zones to NERC CIP ESP boundaries', 'Identify all ESP ingress/egress points'], None),
        ('Identify high-risk conduits requiring priority protection', ['Internet-facing vendor remote access connections', 'Historian data links to corporate network', 'Engineering workstation connections to control zone'], 'warning'),
        ('Design DMZ architecture for IT/OT boundary', ['Define allowed data flows through DMZ', 'Identify historian replication and SCADA-to-corporate data flows'], None),
        ('Produce Zone and Conduit design document for approval', [], None),
    ]

    for text, subs, hl in phase2_items:
        story.append(CheckItem(text, subs, hl))
        story.append(sp(3))

    story.append(sp(6))
    story.append(SignoffBlock(['OT Security Architect', 'Operations Manager', 'CISO']))
    story.append(sp(12))

    # =========================================================
    # PHASE 3
    # =========================================================
    story.append(PhaseHeader('3', 'Infrastructure Readiness', 'Days 23-32', 'OT Infrastructure Engineer'))
    story.append(sp(6))

    phase3_items = [
        ('Provision on-premise HSM (Thales Luna Network HSM or equivalent, FIPS 140-3 Level 3)', ['Complete HSM installation and initialization', 'Conduct air-gapped key ceremony with dual-person integrity', 'Test HSM connectivity from QBITEL management server'], 'critical'),
        ('Deploy air-gapped Ollama LLM instance on management server', ['Install Ollama and download required models to local storage', 'Verify no external API calls (monitor network traffic)', 'Test threat analysis queries against local model'], None),
        ('Configure historian server integration', ['OSIsoft PI System: configure AF server connection and tag subscription', 'GE Proficy or Honeywell PHD: configure data feed for physics model', 'Validate process data is flowing to QBITEL physics engine'], None),
        ('Set up OT jump server with MFA for privileged access', ['Deploy dedicated jump server in OT DMZ', 'Configure MFA (hardware token preferred for OT environments)', 'Document access procedure and emergency bypass'], 'warning'),
        ('Verify air-gap integrity on OT network', ['Confirm no internet connectivity on OT network segments', 'Verify no unauthorized wireless access points', 'Document all authorized remote access paths with approval'], 'critical'),
        ('Validate management server meets QBITEL performance requirements', [], None),
    ]

    for text, subs, hl in phase3_items:
        story.append(CheckItem(text, subs, hl))
        story.append(sp(3))

    story.append(sp(6))
    story.append(SignoffBlock(['OT Infrastructure Engineer', 'CISO', 'Operations Manager']))
    story.append(sp(12))

    # =========================================================
    # PHASE 4
    # =========================================================
    story.append(PhaseHeader('4', 'Safety System Identification & Exclusion', 'Days 33-38', 'Safety Officer (REQUIRED)'))
    story.append(sp(6))

    phase4_items = [
        ('Formally document SIS boundary (IEC 61508 SIL 3/4 certified systems)', ['List all SIS systems by tag number, vendor, SIL rating, and network segment', 'Include: Emergency Shutdown Systems, Fire and Gas Systems, High Integrity Pressure Protection', 'Cross-reference with process safety documentation (HAZOP, SIL assessment)'], 'critical'),
        ('Configure QBITEL Safety Exclusion Zones (passive-only for all SIS segments)', ['Configure each SIS network segment as Exclusion Zone in QBITEL policy engine', 'Verify no active protection rules apply to any SIS network segment', 'Test that no QBITEL traffic is generated toward SIS network segments'], 'critical'),
        ('Test that no QBITEL active protection affects SIS timing', ['Monitor SIS communication timing before and after QBITEL tap deployment', 'Confirm no change in SIS polling latency or communication patterns', 'Document timing validation results for Safety Officer review'], 'critical'),
        ('Configure SIS anomaly escalation to human operators only', ['Any SIS anomaly: escalate to Safety Officer and Operations Manager', 'Confirm no automated response actions are configured for SIS segments', 'Test escalation path with simulated anomaly event'], None),
        ('Review QBITEL configuration against IEC 61508 requirements', ['Confirm QBITEL does not modify SIS firmware or configuration', 'Confirm QBITEL operates outside the SIS safety certification boundary', 'Document configuration for Safety Officer review'], None),
        ('Obtain Safety Officer written approval for SIS boundary configuration', ['Safety Officer reviews and signs SIS Exclusion Zone configuration document', 'Document approval in project record', 'MANDATORY GATE: Active protection phases cannot proceed without this sign-off'], 'critical'),
    ]

    for text, subs, hl in phase4_items:
        story.append(CheckItem(text, subs, hl))
        story.append(sp(3))

    story.append(sp(6))
    story.append(SignoffBlock(['Safety Officer (MANDATORY)', 'Operations Manager', 'OT Security Engineer']))
    story.append(sp(12))

    # =========================================================
    # PHASE 5
    # =========================================================
    story.append(PhaseHeader('5', 'PLC Command Authentication Rollout', 'Days 39-55', 'OT Security Engineer'))
    story.append(sp(6))

    phase5_items = [
        ('Deploy ML-DSA-65 signing on Modbus masters (non-SIS first, per change window)', ['Start with lowest-risk, non-critical OT segment', 'Follow change management procedure for each segment activation', 'Maintain manual rollback capability throughout rollout'], 'warning'),
        ('Test command authentication on isolated test PLC before production rollout', ['Validate ML-DSA-65 signed command accepted by verification layer', 'Validate unsigned command rejected and SOC alert generated', 'Validate replayed command rejected'], 'critical'),
        ('Validate <1ms overhead on production control loops', ['Measure SCADA polling latency before activation', 'Measure SCADA polling latency after activation', 'Confirm <1ms overhead and <100us jitter on all measured loops'], 'critical'),
        ('Roll out command authentication to remaining PLCs/RTUs by zone', ['Control Zone PLCs: activate per zone following change window schedule', 'RTUs (DNP3): activate on each master/outstation pair', 'Engineering workstations: enforce certificate-based authentication'], None),
        ('Test replay attack prevention (timestamp/nonce mechanism)', ['Capture a valid signed command', 'Replay the captured command after the nonce window', 'Confirm replay is rejected and SOC alert generated'], None),
        ('Validate physics-aware anomaly detection baseline established', ['Confirm historian data is feeding physics model', 'Verify at least 72 hours of baseline data collected', 'Test that a simulated physics violation generates an alert'], None),
        ('Produce rollout completion report documenting all activated segments', [], None),
    ]

    for text, subs, hl in phase5_items:
        story.append(CheckItem(text, subs, hl))
        story.append(sp(3))

    story.append(sp(6))
    story.append(SignoffBlock(['OT Security Engineer', 'Operations Manager', 'NERC CIP Compliance Lead']))
    story.append(sp(12))

    # =========================================================
    # PHASE 6
    # =========================================================
    story.append(PhaseHeader('6', 'Protocol Protection Activation', 'Days 56-68', 'OT Security Engineer'))
    story.append(sp(6))

    phase6_items = [
        ('Enable IEC 61850 GOOSE/SV authentication (IEC 62351)', ['Configure ML-DSA-65 GOOSE authentication per substation/segment', 'Validate <100us overhead on GOOSE messages (well within 4ms protection requirement)', 'Test protection relay operation with authenticated GOOSE traffic'], 'critical'),
        ('Enable DNP3 Secure Authentication v5 wrapping where native SAv5 not available', ['Configure QBITEL DNP3 SAv5 proxy on master stations', 'Validate all outstation communication continues normally', 'Test unauthorized DNP3 command rejection'], None),
        ('Enable OPC UA security profile (SignAndEncrypt mode)', ['Configure QBITEL OPC UA security proxy for legacy clients', 'Validate all OPC UA client connections continue to function', 'Confirm OPC UA data flows are encrypted and authenticated'], None),
        ('Configure BACnet/IP and EtherNet/IP protection', ['Enable BACnet anomaly detection and command filtering', 'Enable EtherNet/IP CIP messaging authentication where feasible', 'Validate Rockwell/Allen-Bradley PLC communication unaffected'], None),
        ('Validate all protocol changes against SCADA polling requirements', ['Run 24-hour soak test on each protocol after activation', 'Monitor for any polling timeouts, communication errors, or latency increases', 'Confirm SCADA operator reports no issues'], 'critical'),
        ('Document protocol protection configuration baseline', [], None),
    ]

    for text, subs, hl in phase6_items:
        story.append(CheckItem(text, subs, hl))
        story.append(sp(3))

    story.append(sp(6))
    story.append(SignoffBlock(['OT Security Engineer', 'SCADA Engineer', 'Operations Manager']))
    story.append(sp(12))

    # =========================================================
    # PHASE 7
    # =========================================================
    story.append(PhaseHeader('7', 'NERC CIP Control Mapping', 'Days 69-80', 'NERC CIP Compliance Lead'))
    story.append(sp(6))

    phase7_items = [
        ('Map QBITEL capabilities to CIP-005 (Electronic Security Perimeter)', ['Document ESP boundary monitoring coverage', 'Map zone/conduit configuration to CIP-005 requirements', 'Verify all ESP ingress/egress points are monitored'], None),
        ('Map to CIP-007 (System Security Management - ports, patches, access)', ['Document port and service monitoring coverage', 'Map security event logging to CIP-007 requirements', 'Verify failed authentication events are captured and alerted'], None),
        ('Map to CIP-010 (Change Management - baseline monitoring)', ['Configure CIP-010 baseline deviation detection', 'Test that new unauthorized device triggers CIP-010 alert', 'Document configuration baseline and deviation thresholds'], None),
        ('Map to CIP-011 (Information Protection - encryption)', ['Document PQC encryption coverage for BES Cyber System communications', 'Map historian data protection to CIP-011 requirements', 'Document any unprotected flows with risk justification'], None),
        ('Configure automated CIP evidence collection for all mapped requirements', ['Verify telemetry is being collected for each CIP requirement', 'Test evidence package generation for spot-check of data quality', 'Configure evidence retention period per NERC CIP requirements (6 years)'], 'warning'),
        ('Test NERC CIP report generation (<10 minutes)', ['Trigger report generation and measure time to completion', 'Review report structure and evidence quality with Compliance Lead', 'Confirm report format meets auditor expectations'], None),
        ('Produce NERC CIP control mapping document for audit file', [], None),
    ]

    for text, subs, hl in phase7_items:
        story.append(CheckItem(text, subs, hl))
        story.append(sp(3))

    story.append(sp(6))
    story.append(SignoffBlock(['NERC CIP Compliance Lead', 'CISO', 'External NERC CIP Reviewer (if available)']))
    story.append(sp(12))

    # =========================================================
    # PHASE 8
    # =========================================================
    story.append(PhaseHeader('8', 'Monitoring & Alerting Configuration', 'Days 81-90', 'OT SOC Lead'))
    story.append(sp(6))

    phase8_items = [
        ('Configure OT SOC dashboard (Grafana/Splunk or equivalent)', ['Deploy OT-specific dashboard with protocol health, alert counts, compliance status', 'Configure role-based access: operator view, compliance view, executive view', 'Integrate with existing SIEM (Splunk ES, IBM QRadar, or Microsoft Sentinel)'], None),
        ('Configure P1 alert: Nation-state TTP or PLC command injection', ['Trigger: Any command injection attempt, unauthorized unsigned PLC command', 'Response: Immediate notification to Operations Manager and CISO', 'SLA: Notification within 60 seconds of detection'], 'critical'),
        ('Configure P2 alert: Unauthorized device on OT network', ['Trigger: New device not in approved asset inventory communicates on OT network', 'Response: Block device (if non-SIS) and notify OT SOC within 5 minutes', 'SLA: 15-minute response requirement'], 'warning'),
        ('Configure P3 alert: NERC CIP deviation detected', ['Trigger: Any deviation from documented CIP baseline (ports, configs, access)', 'Response: Notify NERC CIP Compliance Lead and document in incident record', 'SLA: 4-hour remediation or documented exception'], None),
        ('Enable physics-aware anomaly alerts', ['Configure threshold alerts for sensor value violations of physics models', 'Tune thresholds based on 72-hour baseline data (minimize false positives)', 'Verify SIS anomalies escalate to Safety Officer only (no automated block)'], 'warning'),
        ('Configure NIS2 incident reporting automation (EU operators)', ['Configure NIS2 24-hour early warning report template', 'Configure NIS2 72-hour incident report template', 'Test report generation with simulated incident scenario'], None),
        ('Conduct 48-hour monitoring validation with OT SOC team', [], None),
    ]

    for text, subs, hl in phase8_items:
        story.append(CheckItem(text, subs, hl))
        story.append(sp(3))

    story.append(sp(6))
    story.append(SignoffBlock(['OT SOC Lead', 'Operations Manager', 'CISO']))
    story.append(sp(12))

    # =========================================================
    # PHASE 9
    # =========================================================
    story.append(PhaseHeader('9', 'Compliance Validation', 'Days 91-105', 'NERC CIP Compliance Lead'))
    story.append(sp(6))

    phase9_items = [
        ('Generate complete NERC CIP evidence package (all applicable CIP standards)', ['CIP-002: BES Cyber System inventory and classification', 'CIP-005: ESP monitoring evidence and boundary documentation', 'CIP-007: System security management evidence (ports, patches, access)', 'CIP-010: Change management baseline and deviation log', 'CIP-011: Information protection and encryption evidence'], 'critical'),
        ('Conduct IEC 62443 SL-A (achieved security level) assessment', ['Assess each zone against its target Security Level (SL-T)', 'Document gap between SL-A and SL-T with remediation plan', 'Update zone/conduit documentation with assessed security levels'], None),
        ('Generate NIS2 incident reporting test (EU operators)', ['Simulate a qualifying incident and generate 24h and 72h reports', 'Review report content against NIS2 requirements', 'Confirm report submission pathway is operational'], None),
        ('Validate audit trail completeness (no gaps in evidence record)', ['Verify log retention covers required period (NERC CIP: 6 years)', 'Confirm no gaps in continuous monitoring evidence', 'Validate timestamping and chain of custody for all evidence'], 'warning'),
        ('Conduct TSA Pipeline Security compliance review (pipeline operators)', ['Map QBITEL capabilities to TSA cybersecurity directives', 'Generate TSA-formatted compliance evidence package', 'Validate incident reporting pathway to TSA'], None),
        ('External reviewer sign-off on evidence quality', ['Engage qualified NERC CIP subject matter expert for evidence review', 'Address any reviewer findings before final sign-off', 'Document reviewer credentials and review methodology'], 'critical'),
    ]

    for text, subs, hl in phase9_items:
        story.append(CheckItem(text, subs, hl))
        story.append(sp(3))

    story.append(sp(6))
    story.append(SignoffBlock(['NERC CIP Compliance Lead', 'External NERC CIP Reviewer', 'CISO']))
    story.append(sp(12))

    # =========================================================
    # PHASE 10
    # =========================================================
    story.append(PhaseHeader('10', 'Operational Handover', 'Days 106-120', 'OT SOC Lead / Operations Manager'))
    story.append(sp(6))

    phase10_items = [
        ('OT operator training: monitoring, alerting, and incident response (8 hours)', ['Alert triage: P1, P2, P3 response procedures', 'SOC dashboard operation and report generation', 'Incident response workflow: contain, investigate, report'], 'warning'),
        ('ISSO/CISO training: compliance reporting and evidence management', ['NERC CIP evidence package generation and review', 'IEC 62443 assessment update procedure', 'NIS2 and TSA incident reporting workflows'], None),
        ('Establish 30-day hypercare period with QBITEL engineering support', ['Daily QBITEL engineer available during hypercare', 'Define escalation path for any operational issues', 'Document any tuning or configuration adjustments during hypercare'], None),
        ('Define steady-state NOC/SOC handover procedure', ['Transition from hypercare to steady-state support tier', 'Define on-call rotation and escalation contacts', 'Establish change management procedure for ongoing QBITEL updates'], None),
        ('Document escalation procedures: engineering, safety officer, NERC CIP', ['Engineering escalation: equipment issues, timing violations', 'Safety Officer escalation: any SIS anomaly or boundary change', 'NERC CIP escalation: compliance deviation or audit notice'], None),
        ('Validate complete deployment against Phase 0 scope', ['Confirm all intended OT segments are protected', 'Confirm NERC CIP evidence is complete and current', 'Confirm physics model baseline is established for all protected segments'], None),
        ('FINAL GO-LIVE SIGN-OFF: All stakeholders required', [], 'critical'),
    ]

    for text, subs, hl in phase10_items:
        story.append(CheckItem(text, subs, hl))
        story.append(sp(3))

    story.append(sp(6))
    story.append(SignoffBlock(['Operations Manager', 'CISO', 'Safety Officer', 'NERC CIP Compliance Lead', 'Asset Owner']))
    story.append(sp(12))

    story.append(PageBreak())

    # =========================================================
    # APPENDIX A: Performance SLAs
    # =========================================================
    story.append(Paragraph('APPENDIX A: Performance SLAs', h2_style))
    story.append(sp(6))

    sla_data = [
        ['Metric', 'SLA Value', 'Measurement Method', 'Remediation SLA'],
        ['PQC overhead', '<1ms per transaction', 'Network tap latency measurement', '4 hours'],
        ['Timing jitter', '<100us', 'Precision timing measurement', '4 hours'],
        ['System availability', '99.999% (5 nines)', 'Uptime monitoring', 'Immediate (bypass activates)'],
        ['False positive rate', '<0.001%', 'Alert review and classification', '24 hours'],
        ['Alert delivery: P1', '<60 seconds', 'Alert generation timestamp', 'Immediate investigation'],
        ['Alert delivery: P2', '<5 minutes', 'Alert generation timestamp', '15-minute response'],
        ['Alert delivery: P3', '<30 minutes', 'Alert generation timestamp', '4-hour remediation'],
        ['NERC CIP report', '<10 minutes', 'Report generation timestamp', '1 hour'],
        ['Discovery (new tap)', '<4 hours', 'Discovery completion timestamp', '24 hours'],
        ['Threat intel update', 'Weekly (air-gap transfer)', 'Update receipt verification', '48 hours'],
    ]
    st = Table(sla_data, colWidths=[CONTENT_W*0.25, CONTENT_W*0.2, CONTENT_W*0.3, CONTENT_W*0.25])
    st.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),NAVY), ('TEXTCOLOR',(0,0),(-1,0),WHITE_C),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'), ('FONTSIZE',(0,0),(-1,0),8),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[LIGHT_BG,TABLE_ALT]),
        ('FONTSIZE',(0,1),(-1,-1),7.5), ('GRID',(0,0),(-1,-1),0.5,MID_GREY),
        ('VALIGN',(0,0),(-1,-1),'TOP'), ('LEFTPADDING',(0,0),(-1,-1),5),
        ('TOPPADDING',(0,0),(-1,-1),4), ('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))
    story.append(st)
    story.append(sp(16))

    # =========================================================
    # APPENDIX B: Rollback Procedures
    # =========================================================
    story.append(Paragraph('APPENDIX B: Rollback Procedures', h2_style))
    story.append(sp(6))
    story.append(Paragraph(
        'QBITEL deployment is designed to be fully reversible at each phase. '
        'The following emergency procedures enable rapid rollback without operational impact.', body_style))
    story.append(sp(6))

    rollback_items = [
        ('Emergency Passthrough Mode: Bypass QBITEL authentication in <60 seconds',
         ['Access QBITEL management console > Emergency > Passthrough Mode',
          'Enter dual-authorization codes (two operators required)',
          'All traffic passes unimpeded - QBITEL continues passive monitoring only',
          'SOC alert generated and CISO notified automatically',
          'Passthrough mode logged with timestamp and operator credentials'],
         'critical'),
        ('Phase 5/6 Rollback: Disable PLC command authentication per zone',
         ['Access QBITEL policy engine > Zone Configuration',
          'Set target zone to Monitoring Only mode',
          'Authentication bypass takes effect immediately (no traffic interruption)',
          'SCADA polling resumes in unauthenticated mode - same as pre-deployment state',
          'Document rollback event in change management system'],
         'warning'),
        ('Safety Override Procedure: Emergency SIS communication path',
         ['SIS communication paths are NEVER blocked by QBITEL under any circumstances',
          'If SIS communication appears affected, immediately activate Passthrough Mode',
          'Notify Safety Officer immediately',
          'Do not attempt to diagnose QBITEL configuration while SIS event is in progress',
          'QBITEL support: 24/7 emergency line available during hypercare and post-deployment'],
         'critical'),
        ('Complete QBITEL Removal: If deployment must be fully reversed',
         ['Activate Emergency Passthrough Mode',
          'Remove QBITEL appliances from network path (tap remains for monitoring)',
          'Network returns to pre-deployment state immediately',
          'No QBITEL configuration changes affect OT devices - removal is clean',
          'Notify QBITEL account team for support and next steps'],
         None),
    ]

    for text, subs, hl in rollback_items:
        story.append(CheckItem(text, subs, hl))
        story.append(sp(6))

    story.append(sp(12))
    story.append(Paragraph(
        'For 24/7 emergency support during deployment, contact: enterprise@qbitel.com | bridge.qbitel.com',
        ParagraphStyle('footer', fontName='Helvetica-Bold', fontSize=9, textColor=NAVY)))

    doc.build(story)


if __name__ == '__main__':
    build_checklist('docs/brochures/QBITEL_CriticalInfra_Deployment_Checklist.pdf')
    print('PDF saved: docs/brochures/QBITEL_CriticalInfra_Deployment_Checklist.pdf')
