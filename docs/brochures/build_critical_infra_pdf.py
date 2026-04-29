"""Build QBITEL Bridge Critical Infrastructure & ICS/SCADA Marketing Pitch - PDF"""
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
        self.bar_height = height; self.color = color; self.bar_width = width or CONTENT_W
    def wrap(self, avw, avh): return self.bar_width, self.bar_height
    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.bar_width, self.bar_height, fill=1, stroke=0)


class SectionHeader(Flowable):
    def __init__(self, title, subtitle=None, width=None):
        super().__init__()
        self.title = title; self.subtitle = subtitle; self.w = width or CONTENT_W
        self.h = 52 if subtitle else 40
    def wrap(self, avw, avh): return self.w, self.h
    def draw(self):
        c = self.canv
        c.setFillColor(NAVY); c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        c.setFillColor(GOLD); c.rect(0, 0, 6, self.h, fill=1, stroke=0)
        c.setFillColor(TEAL); c.rect(self.w-4, 0, 4, self.h, fill=1, stroke=0)
        c.setFillColor(WHITE_C); c.setFont('Helvetica-Bold', 13)
        ty = self.h - 22 if self.subtitle else (self.h-16)/2 + 4
        c.drawString(16, ty, self.title.upper())
        if self.subtitle:
            c.setFillColor(TEAL); c.setFont('Helvetica', 9)
            c.drawString(16, 8, self.subtitle)


class StatBlock(Flowable):
    def __init__(self, stats, width=None):
        super().__init__()
        self.stats = stats; self.w = width or CONTENT_W; self.h = 60
    def wrap(self, avw, avh): return self.w, self.h
    def draw(self):
        c = self.canv; n = len(self.stats); cw = self.w / n
        c.setFillColor(NAVY); c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        for i, (val, lbl) in enumerate(self.stats):
            x = i * cw
            if i > 0:
                c.setFillColor(TEAL); c.rect(x, 8, 1, self.h-16, fill=1, stroke=0)
            c.setFillColor(GOLD); c.setFont('Helvetica-Bold', 14)
            vw = c.stringWidth(val, 'Helvetica-Bold', 14)
            c.drawString(x + cw/2 - vw/2, self.h*0.55, val)
            c.setFillColor(WHITE_C); c.setFont('Helvetica', 8)
            lw = c.stringWidth(lbl, 'Helvetica', 8)
            c.drawString(x + cw/2 - lw/2, self.h*0.2, lbl)


class ScenarioBox(Flowable):
    def __init__(self, label, title, lines, width=None):
        super().__init__()
        self.label = label; self.title = title; self.lines = lines
        self.w = width or CONTENT_W; self.h = 28 + len(lines) * 16 + 10
    def wrap(self, avw, avh): return self.w, self.h
    def draw(self):
        c = self.canv
        c.setFillColor(TEAL); c.rect(0, self.h-28, self.w, 28, fill=1, stroke=0)
        c.setFillColor(NAVY); c.setFont('Helvetica-Bold', 9)
        c.drawString(10, self.h-18, self.label)
        c.setFillColor(WHITE_C); c.setFont('Helvetica-Bold', 10)
        c.drawString(70, self.h-18, self.title)
        c.setFillColor(LIGHT_BG); c.rect(0, 0, self.w, self.h-28, fill=1, stroke=0)
        c.setFillColor(DARK_TEXT); c.setFont('Helvetica', 8.5)
        for i, line in enumerate(self.lines):
            y = self.h - 44 - i*16
            c.drawString(12, y, '• ' + line)


class CalloutBox(Flowable):
    def __init__(self, text, icon='>', width=None):
        super().__init__()
        self.text = text; self.icon = icon; self.w = width or CONTENT_W; self.h = 36
    def wrap(self, avw, avh): return self.w, self.h
    def draw(self):
        c = self.canv
        c.setFillColor(GOLD); c.rect(0, 0, 6, self.h, fill=1, stroke=0)
        c.setFillColor(LIGHT_BG); c.rect(6, 0, self.w-6, self.h, fill=1, stroke=0)
        c.setFillColor(DARK_TEXT); c.setFont('Helvetica', 9)
        c.drawString(18, self.h*0.4, self.text)


def get_styles():
    return {
        'body': ParagraphStyle('body', fontName='Helvetica', fontSize=9, leading=14, textColor=DARK_TEXT, spaceAfter=6),
        'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=11, textColor=NAVY, spaceBefore=8, spaceAfter=4),
        'h3': ParagraphStyle('h3', fontName='Helvetica-Bold', fontSize=9.5, textColor=TEAL_DARK, spaceBefore=6, spaceAfter=3),
        'bullet': ParagraphStyle('bullet', fontName='Helvetica', fontSize=9, leading=13, leftIndent=12, textColor=DARK_TEXT),
        'table_cell': ParagraphStyle('tc', fontName='Helvetica', fontSize=8.5, leading=12, textColor=DARK_TEXT),
        'table_header': ParagraphStyle('th', fontName='Helvetica-Bold', fontSize=9, textColor=WHITE_C),
    }


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H-30, PAGE_W, 30, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C); canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(MARGIN, PAGE_H-20, 'QBITEL BRIDGE - CRITICAL INFRASTRUCTURE & ICS/SCADA')
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
    canvas.drawString(MARGIN, PAGE_H*0.55 + PAGE_H*0.45*0.75, 'QBITEL BRIDGE')
    canvas.setFillColor(GOLD); canvas.setFont('Helvetica-Bold', 18)
    canvas.drawString(MARGIN, PAGE_H*0.55 + PAGE_H*0.45*0.55, 'Critical Infrastructure & ICS/SCADA')
    canvas.setFillColor(WHITE_C); canvas.setFont('Helvetica-Bold', 20)
    canvas.drawString(MARGIN, PAGE_H*0.55 + PAGE_H*0.45*0.35, 'Quantum-Safe Security for')
    canvas.drawString(MARGIN, PAGE_H*0.55 + PAGE_H*0.45*0.2, 'Operational Technology Networks')
    stats = [('<1ms', 'PQC Overhead'), ('<100us', 'Timing Jitter'), ('99.999%', 'Availability'), ('NERC CIP', 'Compliant')]
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


def build_doc(output_path):
    import os; os.chdir('/Users/prabakarankannan/qbitel')
    doc = BaseDocTemplate(output_path, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN)
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id='cover')
    inner_frame = Frame(MARGIN, 0.7*inch, CONTENT_W, PAGE_H - MARGIN - 0.7*inch, id='inner')
    cover_template = PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover)
    inner_template = PageTemplate(id='Inner', frames=[inner_frame], onPage=draw_page)
    doc.addPageTemplates([cover_template, inner_template])
    S = get_styles()
    story = []
    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())

    # Executive Summary
    story.append(SectionHeader('Executive Summary', 'The OT security imperative in the quantum era'))
    story.append(sp(8))
    story.append(Paragraph(
        'Operational technology networks controlling power grids, water treatment plants, pipelines, and '
        'manufacturing facilities have become the primary target for nation-state cyberattacks. '
        'The Colonial Pipeline attack cost $4.4 billion. Ukrainian power grid attacks left millions without electricity. '
        'Volt Typhoon has pre-positioned in US critical infrastructure. 13 OT attacks occur globally every single day.',
        S['body']))
    story.append(Paragraph(
        'Legacy OT protocols were never designed for security. Modbus, invented in 1979, has zero authentication. '
        'DNP3 and IEC 61850 similarly lack cryptographic protection. Any device on an OT network can send arbitrary '
        'commands to PLCs controlling breakers, chemical dosing systems, or pipeline compressors. '
        'Nation-states exploit this daily.',
        S['body']))
    story.append(Paragraph(
        'QBITEL Bridge addresses the root cause: the absence of cryptographic authentication at the OT protocol level. '
        'Our approach starts passive - a read-only tap that discovers all devices and protocols without operational risk - '
        'then layers deterministic post-quantum cryptography with <1ms overhead and <100us jitter, '
        'ensuring real-time control loops and safety-instrumented systems are never compromised.',
        S['body']))
    story.append(sp(12))

    # Section 1: OT Security Imperative
    story.append(SectionHeader('1. The OT/ICS Security Imperative', 'Operational technology is now the primary nation-state target'))
    story.append(sp(8))
    story.append(StatBlock([('13/day', 'OT Attacks Globally'), ('$4.7B', 'Avg Grid Outage Cost'), ('20-40yr', 'OT Equipment Lifecycle'), ('$1M/day', 'NERC CIP Violation Fine')]))
    story.append(sp(8))
    threats_data = [
        ['Threat', 'Root Cause', 'Consequence'],
        ['Modbus/DNP3 No Auth', 'Designed 1979-1990 with zero security', 'Any device can command PLCs - grid shutdown, water contamination'],
        ['Safety-Security Tension', 'Cannot take critical systems offline to patch', 'Traditional security tools cause the outages they aim to prevent'],
        ['Quantum OT Lifecycle', 'OT equipment lasts 20-40 years past quantum horizon', 'Classical encryption protecting grid today broken by 2060'],
    ]
    tbl = Table(threats_data, colWidths=[CONTENT_W*0.22, CONTENT_W*0.33, CONTENT_W*0.45])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),NAVY), ('TEXTCOLOR',(0,0),(-1,0),WHITE_C),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'), ('FONTSIZE',(0,0),(-1,0),9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[LIGHT_BG,TABLE_ALT]),
        ('FONTSIZE',(0,1),(-1,-1),8), ('GRID',(0,0),(-1,-1),0.5,MID_GREY),
        ('VALIGN',(0,0),(-1,-1),'TOP'), ('LEFTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),5), ('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    story.append(tbl)
    story.append(sp(12))

    # Capabilities
    caps = [
        ('2. Passive OT Protocol Discovery', 'Zero disruption - network tap only, read-only',
         [('Method', 'Passive network tap - no active probing, no traffic injection'),
          ('Coverage', 'Modbus, DNP3, IEC 61850, OPC UA, BACnet, EtherNet/IP, PROFINET'),
          ('Time to discover', '2-4 hours vs 6-12 months manual inventory'),
          ('Output', 'Asset inventory, protocol map, vulnerability assessment')]),
        ('3. Deterministic PQC for Real-Time SCADA', 'IEC 61508 compliant - no timing violations',
         [('Overhead', '<1ms PQC processing per transaction'),
          ('Jitter', '<100us - meets SIL 3/4 real-time requirements'),
          ('Algorithms', 'ML-KEM-768 (key encap), ML-DSA-65 (auth), AES-256-GCM'),
          ('Degradation', 'Graceful: if PQC unavailable, traffic passes with alert')]),
        ('4. PLC Command Authentication', 'Every command cryptographically signed - injection impossible',
         [('Signing', 'Each PLC WRITE command signed with ML-DSA-65'),
          ('Verification', 'RTU/PLC validates signature before executing command'),
          ('Replay protection', 'Timestamp + nonce - replayed commands rejected'),
          ('Key management', 'HSM-backed, air-gapped key ceremony')]),
        ('5. Safety-Instrumented System Protection', 'Safety-first - SIS ALWAYS escalates to humans',
         [('Boundary', 'SIS systems explicitly excluded from autonomous action'),
          ('SIL compliance', 'Passive monitoring only for SIL 3/4 loops'),
          ('Human-in-loop', 'Any SIS anomaly: immediate operator notification, no auto-action'),
          ('IEC 61508', 'Full compliance - no modification of safety-certified code')]),
        ('6. Air-Gapped Sovereign Deployment', 'No internet connectivity required',
         [('AI inference', 'Ollama on-premise - no cloud LLM calls'),
          ('Threat intel', 'Offline update via removable media (air-gap transfer)'),
          ('HSM', 'On-premise Thales Luna or equivalent - never cloud'),
          ('Compliance', 'Fully sovereign - no data leaves the facility')]),
        ('7. NERC CIP / IEC 62443 Automation', 'Automated compliance evidence - audit-ready in <10 minutes',
         [('NERC CIP', 'CIP-002 through CIP-014 continuous monitoring'),
          ('IEC 62443', 'Zone/conduit mapping, SL-T security level assessment'),
          ('NIS2', 'EU incident reporting automation'),
          ('Reports', 'Audit-ready evidence packages in <10 minutes')]),
        ('8. Physics-Aware Anomaly Detection', 'Cross-validates sensor readings against physical laws',
         [('Validation', 'Temperature, pressure, flow readings validated against physics models'),
          ('False positives', '<0.001% - physics constraints eliminate noise'),
          ('Detection', 'Command anomalies, sensor spoofing, rogue devices'),
          ('Response', 'Autonomous block (safe) or escalate (safety-critical)')]),
    ]
    for title, subtitle, rows in caps:
        story.append(SectionHeader(title, subtitle))
        story.append(sp(6))
        tdata = [['Attribute', 'Value']] + [[r[0], r[1]] for r in rows]
        t = Table(tdata, colWidths=[CONTENT_W*0.28, CONTENT_W*0.72])
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),TEAL), ('TEXTCOLOR',(0,0),(-1,0),WHITE_C),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'), ('FONTSIZE',(0,0),(-1,0),8),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE_C,LIGHT_BG]),
            ('FONTSIZE',(0,1),(-1,-1),8), ('GRID',(0,0),(-1,-1),0.5,MID_GREY),
            ('VALIGN',(0,0),(-1,-1),'TOP'), ('LEFTPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),4), ('BOTTOMPADDING',(0,0),(-1,-1),4),
        ]))
        story.append(t)
        story.append(sp(10))

    # Compliance table
    story.append(SectionHeader('9. Compliance Coverage'))
    story.append(sp(6))
    comp_data = [['Framework','Coverage','Key Requirement'],
        ['NERC CIP (CIP-002 to CIP-014)','Continuous','ESP, BES Cyber System protection'],
        ['IEC 62443','Full SL-T','Zone/conduit, security levels'],
        ['NIST SP 800-82','Mapped','ICS security controls'],
        ['NIS2 Directive','Reporting','EU critical infrastructure'],
        ['TSA Pipeline Security','Compliant','Pipeline cybersecurity mandates'],
        ['IEC 61508','Compatible','Functional safety (SIL 3/4)'],
        ['IEC 62351','Power systems','GOOSE, SV authentication'],
    ]
    ct = Table(comp_data, colWidths=[CONTENT_W*0.32, CONTENT_W*0.15, CONTENT_W*0.53])
    ct.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),NAVY), ('TEXTCOLOR',(0,0),(-1,0),WHITE_C),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'), ('FONTSIZE',(0,0),(-1,0),9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[LIGHT_BG,TABLE_ALT]),
        ('FONTSIZE',(0,1),(-1,-1),8), ('GRID',(0,0),(-1,-1),0.5,MID_GREY),
        ('LEFTPADDING',(0,0),(-1,-1),6), ('TOPPADDING',(0,0),(-1,-1),4), ('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))
    story.append(ct)
    story.append(sp(12))

    # Performance specs
    story.append(SectionHeader('10. Performance Specifications'))
    story.append(sp(6))
    perf = [['Metric','Value'],
        ['PQC overhead','<1ms per transaction'],
        ['Timing jitter','<100us (SIL 3/4 compliant)'],
        ['System availability','99.999% (five nines)'],
        ['False positive rate','<0.001% (physics-aware)'],
        ['Protocol discovery','2-4 hours (passive tap)'],
        ['NERC CIP report','<10 minutes automated'],
        ['Deployment','Zero downtime, passive-first'],
    ]
    pt = Table(perf, colWidths=[CONTENT_W*0.4, CONTENT_W*0.6])
    pt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),NAVY), ('TEXTCOLOR',(0,0),(-1,0),WHITE_C),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'), ('FONTSIZE',(0,0),(-1,0),9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[LIGHT_BG,TABLE_ALT]),
        ('FONTSIZE',(0,1),(-1,-1),8), ('GRID',(0,0),(-1,-1),0.5,MID_GREY),
        ('LEFTPADDING',(0,0),(-1,-1),6), ('TOPPADDING',(0,0),(-1,-1),4), ('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))
    story.append(pt)
    story.append(sp(12))

    # Customer Scenarios
    story.append(SectionHeader('11. Customer Scenarios'))
    story.append(sp(8))
    scenarios = [
        ('POWER', 'Power Utility - Grid Substation Protection',
         ['Challenge: 500 substations with IEC 61850 GOOSE messages unauthenticated',
          'Solution: Passive discovery + IEC 62351 PQC authentication',
          'Result: NERC CIP CIP-007 compliance, zero downtime deployment',
          'Timeline: 8-week rollout across all substations']),
        ('WATER', 'Water Treatment - SCADA Command Authentication',
         ['Challenge: DNP3 SCADA with no authentication - chemical dosing at risk',
          'Solution: ML-DSA-65 command signing on all DNP3 masters',
          'Result: Command injection impossible, physics anomaly detection active',
          'Timeline: 4-week deployment, single change window']),
        ('PIPELINE', 'Pipeline Operator - TSA Mandate Compliance',
         ['Challenge: TSA Pipeline Security Directive deadline, Modbus unprotected',
          'Solution: Passive Modbus discovery + PQC wrapping + TSA evidence',
          'Result: TSA mandate compliant, fraction of $4.4B Colonial Pipeline precedent',
          'Timeline: 6-week deployment, TSA evidence package in 2 weeks']),
    ]
    for label, title, lines in scenarios:
        story.append(ScenarioBox(label, title, lines))
        story.append(sp(8))

    # Next steps
    story.append(SectionHeader('12. Next Steps'))
    story.append(sp(8))
    nextsteps = [
        ('STEP 1', 'Passive OT Discovery Assessment (2 weeks, zero disruption)',
         ['Deploy passive network tap on target OT segment', 'Enumerate all protocols, devices, and communication patterns', 'Produce asset inventory and vulnerability report']),
        ('STEP 2', 'NERC CIP / IEC 62443 Gap Analysis',
         ['Map current controls against NERC CIP requirements', 'Identify high-priority gaps in ESP and command authentication', 'Produce remediation roadmap with estimated effort']),
        ('STEP 3', 'Proof of Concept - Single Substation or Plant Segment',
         ['Deploy PLC command authentication on target segment', 'Validate <1ms overhead and zero timing violations', 'Demonstrate NERC CIP evidence generation']),
    ]
    for label, title, lines in nextsteps:
        story.append(ScenarioBox(label, title, lines))
        story.append(sp(8))

    # Contact
    story.append(sp(6))
    contact_data = [
        [Paragraph('<b>enterprise@qbitel.com</b>', ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=10, textColor=GOLD)),
         Paragraph('<b>bridge.qbitel.com</b>', ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=10, textColor=GOLD)),
         Paragraph('<b>Contact your account team</b>', ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=10, textColor=GOLD))],
        [Paragraph('Email', ParagraphStyle('cl', fontName='Helvetica', fontSize=8, textColor=WHITE_C)),
         Paragraph('Website', ParagraphStyle('cl', fontName='Helvetica', fontSize=8, textColor=WHITE_C)),
         Paragraph('Schedule a call', ParagraphStyle('cl', fontName='Helvetica', fontSize=8, textColor=WHITE_C))],
    ]
    ctbl = Table(contact_data, colWidths=[CONTENT_W/3]*3)
    ctbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),NAVY), ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'), ('TOPPADDING',(0,0),(-1,-1),10),
        ('BOTTOMPADDING',(0,0),(-1,-1),10), ('GRID',(0,0),(-1,-1),1,TEAL),
    ]))
    story.append(ctbl)
    doc.build(story)


if __name__ == '__main__':
    build_doc('docs/brochures/QBITEL_Bridge_CriticalInfra_Marketing_Pitch.pdf')
    print('PDF saved: docs/brochures/QBITEL_Bridge_CriticalInfra_Marketing_Pitch.pdf')
