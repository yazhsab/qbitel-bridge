"""
Build QBITEL Bridge Healthcare Deployment Checklist - Professional PDF
10-phase deployment checklist for Healthcare & Medical Devices.
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
GOLD       = HexColor('#F0A500')
LIGHT_BG   = HexColor('#F4F7FA')
MID_GREY   = HexColor('#5A6A7A')
DARK_TEXT  = HexColor('#1A1A2E')
TABLE_ALT  = HexColor('#EAF3F8')
WHITE_C    = HexColor('#FFFFFF')
LIGHT_NAVY = HexColor('#1A2D5A')
GREEN_C    = HexColor('#2D7A3A')
GREEN_LIGHT= HexColor('#F0FFF4')
AMBER_C    = HexColor('#D97706')
AMBER_LIGHT= HexColor('#FFFBEB')

PAGE_W, PAGE_H = letter
MARGIN = 0.85 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


class PhaseHeader(Flowable):
    def __init__(self, phase_num, phase_name, duration, owner, width=None):
        super().__init__()
        self.phase_num = phase_num
        self.phase_name = phase_name
        self.duration = duration
        self.owner = owner
        self.w = width or CONTENT_W
        self.h = 50

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Background gradient simulation
        c.setFillColor(NAVY)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        # Phase number badge
        c.setFillColor(GOLD)
        c.roundRect(8, 8, 36, 34, 4, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 14)
        num_str = str(self.phase_num)
        num_w = c.stringWidth(num_str, 'Helvetica-Bold', 14)
        c.drawString(26 - num_w / 2, 21, num_str)
        # Phase name
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 13)
        c.drawString(54, 30, f'PHASE {self.phase_num}: {self.phase_name.upper()}')
        # Duration and owner pills
        c.setFillColor(TEAL)
        c.roundRect(54, 10, 80, 16, 3, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica', 8)
        c.drawString(59, 14, self.duration)
        c.setFillColor(LIGHT_NAVY)
        c.roundRect(142, 10, self.w - 152, 16, 3, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica', 8)
        c.drawString(148, 14, f'Owner: {self.owner}')
        # Right teal accent
        c.setFillColor(TEAL)
        c.rect(self.w - 4, 0, 4, self.h, fill=1, stroke=0)


class ChecklistItem(Flowable):
    def __init__(self, number, task, owner_tag, priority='normal', notes='', width=None):
        super().__init__()
        self.number = number
        self.task = task
        self.owner_tag = owner_tag
        self.priority = priority
        self.notes = notes
        self.w = width or CONTENT_W
        task_lines = self._count_lines(task, 'Helvetica', 9.5, self.w - 120)
        notes_lines = self._count_lines(notes, 'Helvetica-Oblique', 8, self.w - 120) if notes else 0
        self.h = max(28, task_lines * 13 + (notes_lines * 11 if notes else 0) + 14)

    def _count_lines(self, text, font, size, max_w):
        from reportlab.pdfbase.pdfmetrics import stringWidth
        if not text:
            return 0
        words = text.split()
        lines = 1
        current = ''
        for word in words:
            test = current + (' ' if current else '') + word
            if stringWidth(test, font, size) <= max_w:
                current = test
            else:
                lines += 1
                current = word
        return lines

    def _wrap_text(self, text, font, size, max_w):
        from reportlab.pdfbase.pdfmetrics import stringWidth
        if not text:
            return []
        words = text.split()
        lines = []
        current = ''
        for word in words:
            test = current + (' ' if current else '') + word
            if stringWidth(test, font, size) <= max_w:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or ['']

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Row background
        bg = AMBER_LIGHT if self.priority == 'critical' else (GREEN_LIGHT if self.priority == 'verify' else WHITE_C)
        c.setFillColor(bg)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        # Checkbox
        c.setStrokeColor(TEAL)
        c.setLineWidth(1.5)
        c.roundRect(8, self.h / 2 - 9, 16, 16, 2, fill=0, stroke=1)
        # Item number
        c.setFillColor(MID_GREY)
        c.setFont('Helvetica', 7.5)
        c.drawString(30, self.h / 2 - 4, f'{self.number}.')
        # Task text
        task_lines = self._wrap_text(self.task, 'Helvetica', 9.5, self.w - 120)
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 9.5)
        start_y = self.h - 14
        if self.notes:
            note_lines = self._wrap_text(self.notes, 'Helvetica-Oblique', 8, self.w - 120)
            total_h = len(task_lines) * 13 + len(note_lines) * 11
            start_y = self.h - (self.h - total_h) / 2 - 4
        for i, line in enumerate(task_lines):
            c.drawString(48, start_y - i * 13, line)
        if self.notes:
            note_lines = self._wrap_text(self.notes, 'Helvetica-Oblique', 8, self.w - 120)
            note_y = start_y - len(task_lines) * 13
            c.setFillColor(MID_GREY)
            c.setFont('Helvetica-Oblique', 8)
            for i, line in enumerate(note_lines):
                c.drawString(48, note_y - i * 11 + 2, line)
        # Owner tag
        tag_color = TEAL if 'Security' in self.owner_tag or 'CISO' in self.owner_tag else (
            AMBER_C if 'Clinical' in self.owner_tag or 'Biomedical' in self.owner_tag else NAVY)
        tag_w = c.stringWidth(self.owner_tag, 'Helvetica-Bold', 7) + 10
        c.setFillColor(tag_color)
        c.roundRect(self.w - tag_w - 4, self.h / 2 - 8, tag_w, 16, 3, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 7)
        c.drawString(self.w - tag_w + 1, self.h / 2 - 4, self.owner_tag)
        # Priority indicator
        if self.priority == 'critical':
            c.setFillColor(AMBER_C)
            c.rect(0, 0, 4, self.h, fill=1, stroke=0)
        elif self.priority == 'verify':
            c.setFillColor(GREEN_C)
            c.rect(0, 0, 4, self.h, fill=1, stroke=0)
        else:
            c.setFillColor(TEAL)
            c.rect(0, 0, 3, self.h, fill=1, stroke=0)
        # Bottom divider
        c.setStrokeColor(HexColor('#E0EAF0'))
        c.setLineWidth(0.5)
        c.line(4, 0, self.w, 0)


class SummaryBox(Flowable):
    def __init__(self, phase_num, title, item_count, critical_count, width=None):
        super().__init__()
        self.phase_num = phase_num
        self.title = title
        self.item_count = item_count
        self.critical_count = critical_count
        self.w = width or CONTENT_W
        self.h = 42

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(TEAL)
        c.roundRect(0, 0, self.w, self.h, 4, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 10)
        c.drawString(14, 26, f'Phase {self.phase_num} Complete: {self.title}')
        c.setFont('Helvetica', 8.5)
        c.drawString(14, 12, f'{self.item_count} checklist items  |  {self.critical_count} critical items  |  Sign-off required before advancing to next phase')
        # Checkbox for sign-off
        c.setStrokeColor(WHITE_C)
        c.setLineWidth(1.5)
        c.roundRect(self.w - 90, 12, 14, 14, 2, fill=0, stroke=1)
        c.setFont('Helvetica', 8)
        c.drawString(self.w - 72, 15, 'Sign-off:_____________')


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 0.45 * inch, PAGE_W, 0.45 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, PAGE_H - 0.45 * inch - 3, PAGE_W, 3, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(MARGIN, PAGE_H - 0.32 * inch, 'QBITEL BRIDGE')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica', 9)
    canvas.drawString(MARGIN + 1.1 * inch, PAGE_H - 0.32 * inch,
                      'Healthcare Deployment Checklist')
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
                      'Confidential -- For Authorized Implementation Teams Only  |  (c) 2025 QBITEL Technologies')
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
    p.moveTo(w * 0.58, h)
    p.lineTo(w, h)
    p.lineTo(w, h * 0.68)
    p.close()
    canvas.drawPath(p, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    p2 = canvas.beginPath()
    p2.moveTo(w * 0.74, h)
    p2.lineTo(w, h)
    p2.lineTo(w, h * 0.80)
    p2.close()
    canvas.drawPath(p2, fill=1, stroke=0)
    canvas.setFillColor(TEAL_DARK)
    canvas.rect(0, 0, w, 1.5 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, 1.5 * inch, w, 5, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 42)
    canvas.drawString(MARGIN, h * 0.70, 'QBITEL BRIDGE')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica-Bold', 20)
    canvas.drawString(MARGIN, h * 0.70 - 40, 'HEALTHCARE DEPLOYMENT CHECKLIST')
    canvas.setFillColor(GOLD)
    canvas.rect(MARGIN, h * 0.70 - 50, 4.8 * inch, 4, fill=1, stroke=0)
    canvas.setFillColor(HexColor('#CCDDE8'))
    canvas.setFont('Helvetica-Oblique', 12)
    canvas.drawString(MARGIN, h * 0.70 - 74,
                      '10-Phase Implementation Guide -- Healthcare & Medical Devices')
    # Phase overview boxes
    phases = [
        ('0', 'Pre-Engage'), ('1', 'Discovery'), ('2', 'Infra'), ('3', 'Policy'),
        ('4', 'Devices'), ('5', 'Protocols'), ('6', 'EHR/HIS'), ('7', 'Monitoring'),
        ('8', 'Compliance'), ('9', 'UAT'), ('10', 'Go-Live'),
    ]
    box_w = (CONTENT_W - 10 * 0.06 * inch) / 11
    by = h * 0.38
    for i, (num, name) in enumerate(phases):
        bx = MARGIN + i * (box_w + 0.06 * inch)
        c_bg = TEAL if i % 2 == 0 else LIGHT_NAVY
        canvas.setFillColor(c_bg)
        canvas.roundRect(bx, by, box_w, 0.75 * inch, 4, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.rect(bx, by + 0.75 * inch - 3, box_w, 3, fill=1, stroke=0)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica-Bold', 12)
        nw = canvas.stringWidth(num, 'Helvetica-Bold', 12)
        canvas.drawString(bx + (box_w - nw) / 2, by + 0.48 * inch, num)
        canvas.setFont('Helvetica', 6)
        nw2 = canvas.stringWidth(name, 'Helvetica', 6)
        canvas.drawString(bx + (box_w - nw2) / 2, by + 0.22 * inch, name)
    # Stats row
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(MARGIN, h * 0.28,
                      '11 Phases  |  200+ Checklist Items  |  20-22 Week Timeline  |  Zero Clinical Disruption Guaranteed')
    canvas.setFillColor(HexColor('#AABBCC'))
    canvas.setFont('Helvetica', 8.5)
    canvas.drawString(MARGIN, h * 0.24,
                      'Color Legend:   Gold bar = Critical item   Teal bar = Standard item   Green bar = Verification item')
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(MARGIN, 0.80 * inch,
                      'Facility: _________________________________   Project Lead: _________________________   Start Date: ___________')
    canvas.setFillColor(HexColor('#AABBCC'))
    canvas.setFont('Helvetica', 8.5)
    canvas.drawString(MARGIN, 0.50 * inch,
                      'Confidential Implementation Document  |  Version 1.0  |  2025  |  QBITEL Technologies')
    canvas.restoreState()


def sp(n=8):
    return Spacer(1, n)


def build_checklist_pdf():
    out_path = 'docs/brochures/QBITEL_Healthcare_Deployment_Checklist.pdf'

    doc = BaseDocTemplate(
        out_path, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.65 * inch, bottomMargin=0.6 * inch,
    )
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H, id='cover')
    inner_frame = Frame(
        MARGIN, 0.6 * inch,
        PAGE_W - 2 * MARGIN, PAGE_H - 0.65 * inch - 0.6 * inch,
        id='inner'
    )
    doc.addPageTemplates([
        PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover),
        PageTemplate(id='Inner', frames=[inner_frame], onPage=draw_page),
    ])

    story = []
    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())

    # Legend
    S_note = ParagraphStyle('note', fontName='Helvetica-Oblique', fontSize=8.5,
                            textColor=MID_GREY, leading=12, spaceAfter=4)
    S_head = ParagraphStyle('head', fontName='Helvetica-Bold', fontSize=11,
                            textColor=NAVY, leading=14, spaceAfter=6)
    story.append(Paragraph('HOW TO USE THIS CHECKLIST', S_head))
    story.append(Paragraph(
        'Work through each phase sequentially. Obtain sign-off at the end of each phase before advancing. '
        'Gold left bar = Critical item requiring CISO or Compliance Officer sign-off. '
        'Teal left bar = Standard item (IT/Security team sign-off). '
        'Green left bar = Verification item (requires testing evidence). '
        'Owner tags: Security = Security Team, Clinical = Clinical Informatics, Biomedical = Biomedical Engineering.',
        S_note
    ))
    story.append(sp(10))

    # ── PHASE 0: Pre-Engagement ────────────────────────────────────────────────
    story.append(PhaseHeader(0, 'Pre-Engagement', 'Week 1-2',
                             'CISO + Project Manager'))
    story.append(sp(4))
    phase0_items = [
        ('0.1', 'Identify executive sponsor (CISO or VP of IT) with budget authority', 'CISO', 'critical', ''),
        ('0.2', 'Assemble core project team: CISO, CIO delegate, HIPAA Officer, Biomedical Engineering lead', 'CISO', 'critical', ''),
        ('0.3', 'Schedule kickoff meeting with QBITEL Clinical Security Engineering team', 'PM', 'normal', ''),
        ('0.4', 'Review existing medical device inventory for completeness (biomedical asset register)', 'Biomedical', 'normal', ''),
        ('0.5', 'Identify all clinical network segments and VLAN architecture documentation', 'Network', 'normal', ''),
        ('0.6', 'Collect existing HIPAA Security Rule gap assessment (if available)', 'Compliance', 'normal', ''),
        ('0.7', 'Identify all EHR and clinical application vendors (Epic, Cerner, etc.)', 'IT', 'normal', ''),
        ('0.8', 'Obtain existing network diagrams including clinical device VLANs', 'Network', 'normal', ''),
        ('0.9', 'Brief Legal team on QBITEL BAA requirements -- obtain preliminary approval', 'Legal', 'critical', ''),
        ('0.10', 'Notify FDA Cybersecurity point of contact if facility is research/regulated', 'Compliance', 'normal', ''),
        ('0.11', 'Schedule Biomedical Engineering availability for device inventory phase', 'Biomedical', 'normal', ''),
        ('0.12', 'Identify Change Control Board (CAB) process for network changes', 'IT', 'normal', ''),
    ]
    for num, task, owner, priority, notes in phase0_items:
        story.append(ChecklistItem(num, task, owner, priority, notes))
    story.append(sp(6))
    story.append(SummaryBox(0, 'Pre-Engagement', 12, 3))
    story.append(PageBreak())

    # ── PHASE 1: Medical Device Discovery ─────────────────────────────────────
    story.append(PhaseHeader(1, 'Medical Device Discovery', 'Week 1-2',
                             'Security Team + Biomedical Engineering'))
    story.append(sp(4))
    phase1_items = [
        ('1.1', 'Deploy QBITEL passive network sensors on clinical network segments', 'Security', 'normal', 'Passive only -- no traffic injection, no device impact'),
        ('1.2', 'Validate sensor placement covers all clinical VLANs: ICU, ED, OR, Radiology, Pharmacy, Med-Surg', 'Network', 'critical', ''),
        ('1.3', 'Run 48-hour passive discovery scan to observe all device communication patterns', 'Security', 'normal', 'Do not disturb clinical operations'),
        ('1.4', 'Export device inventory report and compare to biomedical asset register', 'Biomedical', 'verify', 'Expect 15-20% devices not in asset register'),
        ('1.5', 'Classify all discovered devices by class: imaging, monitoring, infusion, lab, other', 'Biomedical', 'normal', ''),
        ('1.6', 'Identify devices on end-of-life operating systems (Windows XP, CE, embedded Linux)', 'Security', 'critical', ''),
        ('1.7', 'Document all communication protocols observed per device class', 'Security', 'normal', 'HL7, DICOM, proprietary, etc.'),
        ('1.8', 'Identify devices with external internet connectivity (even occasional)', 'Security', 'critical', ''),
        ('1.9', 'Map all HL7 interface connections to EHR integration engine', 'IT', 'normal', ''),
        ('1.10', 'Map all DICOM connections to PACS and modality worklist servers', 'IT', 'normal', ''),
        ('1.11', 'Identify devices with default or shared credentials (vulnerability scan results)', 'Security', 'critical', ''),
        ('1.12', 'Generate vulnerability score report per device class for executive review', 'Security', 'verify', 'Board-ready risk report delivered by QBITEL'),
        ('1.13', 'Biomedical Engineering review of device inventory for accuracy', 'Biomedical', 'verify', ''),
        ('1.14', 'Document devices requiring Protocol Proxy Mode (legacy/implantable programmer)', 'Security', 'normal', ''),
    ]
    for num, task, owner, priority, notes in phase1_items:
        story.append(ChecklistItem(num, task, owner, priority, notes))
    story.append(sp(6))
    story.append(SummaryBox(1, 'Medical Device Discovery', 14, 5))
    story.append(PageBreak())

    # ── PHASE 2: Infrastructure Readiness ─────────────────────────────────────
    story.append(PhaseHeader(2, 'Infrastructure Readiness', 'Week 3-4',
                             'Network + Security Team'))
    story.append(sp(4))
    phase2_items = [
        ('2.1', 'Procure and rack QBITEL Clinical Edge Node hardware in primary data center', 'IT', 'normal', 'FIPS 140-3 Level 3 HSM included in appliance'),
        ('2.2', 'Procure and rack redundant (hot-standby) Clinical Edge Node', 'IT', 'normal', ''),
        ('2.3', 'Configure Bridge management VLAN (isolated from clinical VLANs)', 'Network', 'normal', ''),
        ('2.4', 'Configure 10GbE or 25GbE uplinks from Edge Nodes to core switching', 'Network', 'normal', ''),
        ('2.5', 'Configure network SPAN/tap ports on all clinical access switches', 'Network', 'normal', 'Passive tap -- no traffic disruption'),
        ('2.6', 'Implement VLAN segmentation per clinical area if not already in place', 'Network', 'critical', 'ICU, ED, OR, Radiology, Pharmacy each in isolated VLANs'),
        ('2.7', 'Configure HSM initialization and key generation on primary Edge Node', 'Security', 'critical', 'Requires QBITEL HSM Administrator ceremony'),
        ('2.8', 'Synchronize HSM state to hot-standby Edge Node', 'Security', 'critical', ''),
        ('2.9', 'Configure firewall rules for Bridge management traffic', 'Security', 'normal', 'Port list provided by QBITEL'),
        ('2.10', 'Test fail-open configuration for life-critical device VLANs (ICU, CCU, OR)', 'Security', 'verify', 'CRITICAL: Life-safety devices must fail-open'),
        ('2.11', 'Configure NTP synchronization for audit log timestamping', 'IT', 'normal', ''),
        ('2.12', 'Validate storage capacity for 90-day audit log retention', 'IT', 'normal', ''),
        ('2.13', 'Configure backup and disaster recovery for Edge Node configuration', 'IT', 'normal', ''),
        ('2.14', 'Submit network changes through Change Control Board (CAB)', 'IT', 'critical', ''),
    ]
    for num, task, owner, priority, notes in phase2_items:
        story.append(ChecklistItem(num, task, owner, priority, notes))
    story.append(sp(6))
    story.append(SummaryBox(2, 'Infrastructure Readiness', 14, 5))
    story.append(PageBreak())

    # ── PHASE 3: Policy Configuration ─────────────────────────────────────────
    story.append(PhaseHeader(3, 'Policy Configuration', 'Week 5-6',
                             'HIPAA Officer + Security Team'))
    story.append(sp(4))
    phase3_items = [
        ('3.1', 'Execute QBITEL Business Associate Agreement (BAA) -- Legal sign-off', 'Legal', 'critical', ''),
        ('3.2', 'Configure HIPAA Security Rule policy mapping in Bridge Compliance Engine', 'Compliance', 'critical', '45 CFR 164.312 all subsections'),
        ('3.3', 'Define PHI data classification tiers (high, medium, standard)', 'Compliance', 'normal', ''),
        ('3.4', 'Configure PHI transmission logging policies per data classification', 'Security', 'normal', ''),
        ('3.5', 'Map device classes to clinical urgency tiers (critical, routine, background)', 'Clinical', 'critical', 'Life-safety devices always Priority 0'),
        ('3.6', 'Configure Battery-Aware Scheduler policies for wireless device classes', 'Security', 'normal', ''),
        ('3.7', 'Define anomaly detection thresholds per device class', 'Security', 'normal', 'Use QBITEL recommended baselines initially'),
        ('3.8', 'Configure VLAN quarantine automation for compromised device response', 'Security', 'critical', 'Test in lab environment before production'),
        ('3.9', 'Define incident notification escalation tree (Security > CISO > CEO path)', 'Security', 'normal', ''),
        ('3.10', 'Configure audit log retention policy (90-day minimum, 1-year recommended)', 'Compliance', 'normal', ''),
        ('3.11', 'Define access control policies for Bridge management console', 'Security', 'normal', ''),
        ('3.12', 'Configure role-based access for Compliance dashboard (auditor read-only)', 'Compliance', 'normal', ''),
    ]
    for num, task, owner, priority, notes in phase3_items:
        story.append(ChecklistItem(num, task, owner, priority, notes))
    story.append(sp(6))
    story.append(SummaryBox(3, 'Policy Configuration', 12, 5))
    story.append(PageBreak())

    # ── PHASE 4: Medical Device Protection ────────────────────────────────────
    story.append(PhaseHeader(4, 'Medical Device Protection', 'Week 7-10',
                             'Security Team + Biomedical Engineering'))
    story.append(sp(4))
    phase4_items = [
        ('4.1', 'Begin device onboarding with lowest-risk device class (lab analyzers recommended)', 'Security', 'normal', 'Do not start with life-critical devices'),
        ('4.2', 'Deploy non-invasive wrapper on laboratory device VLAN -- validate no disruption', 'Security', 'verify', '48-hour monitoring period before expanding'),
        ('4.3', 'Deploy wrapper on pharmacy automation devices', 'Security', 'normal', ''),
        ('4.4', 'Deploy wrapper on administrative medical devices (nurse call, wandering)', 'Security', 'normal', ''),
        ('4.5', 'Deploy wrapper on radiology ancillary devices (workstations, printers)', 'Security', 'normal', ''),
        ('4.6', 'Deploy wrapper on patient monitoring devices (Med-Surg) -- notify unit managers', 'Security', 'critical', 'Notify nursing leadership before activation'),
        ('4.7', 'Deploy wrapper on infusion pump fleet -- validate alarm transmission unaffected', 'Biomedical', 'critical', 'Test critical alarm delivery in monitored environment'),
        ('4.8', 'Deploy wrapper on ICU monitoring devices -- Biomedical Engineering on-site', 'Biomedical', 'critical', 'ICU deployment requires Biomedical presence'),
        ('4.9', 'Deploy wrapper on ED monitoring devices', 'Biomedical', 'critical', ''),
        ('4.10', 'Deploy wrapper on OR-connected devices -- coordinate with OR leadership', 'Clinical', 'critical', 'OR blackout periods: no changes during surgical lists'),
        ('4.11', 'Deploy wrapper on imaging modalities (CT, MRI, X-ray, Ultrasound)', 'Biomedical', 'normal', ''),
        ('4.12', 'Configure Protocol Proxy Mode for implantable device programmers', 'Security', 'normal', ''),
        ('4.13', 'Validate 100% device coverage in device inventory dashboard', 'Security', 'verify', ''),
        ('4.14', 'Biomedical Engineering sign-off on all device class deployments', 'Biomedical', 'critical', ''),
        ('4.15', 'Run 72-hour stability monitoring across all protected device VLANs', 'Security', 'verify', ''),
    ]
    for num, task, owner, priority, notes in phase4_items:
        story.append(ChecklistItem(num, task, owner, priority, notes))
    story.append(sp(6))
    story.append(SummaryBox(4, 'Medical Device Protection', 15, 7))
    story.append(PageBreak())

    # ── PHASE 5: HL7/FHIR/DICOM Security ──────────────────────────────────────
    story.append(PhaseHeader(5, 'HL7/FHIR/DICOM Security', 'Week 9-12',
                             'IT + Integration Team'))
    story.append(sp(4))
    phase5_items = [
        ('5.1', 'Inventory all HL7 v2 interface connections from EHR integration engine', 'IT', 'normal', ''),
        ('5.2', 'Deploy Bridge HL7 Protocol Security Layer on integration engine outputs', 'IT', 'normal', 'Mirth Connect, Rhapsody, or Ensemble'),
        ('5.3', 'Validate HL7 ADT message delivery to downstream systems (post-Bridge)', 'IT', 'verify', 'Test ADT A01-A40 message types'),
        ('5.4', 'Validate HL7 ORM/ORU lab order and result message delivery', 'IT', 'verify', ''),
        ('5.5', 'Deploy Bridge FHIR R4 security layer on all FHIR API endpoints', 'IT', 'normal', ''),
        ('5.6', 'Validate SMART on FHIR token issuance and validation (post-Bridge)', 'IT', 'verify', ''),
        ('5.7', 'Configure DICOM TLS overlay on all imaging modality connections', 'IT', 'normal', ''),
        ('5.8', 'Validate DICOM C-STORE operations to PACS (post-Bridge)', 'Radiology', 'verify', 'Validate with one study from each modality type'),
        ('5.9', 'Validate DICOM C-FIND and C-MOVE retrieval operations', 'Radiology', 'verify', ''),
        ('5.10', 'Validate Modality Worklist (MWL) delivery to imaging modalities', 'Radiology', 'verify', ''),
        ('5.11', 'Deploy X12 EDI security layer on clearinghouse connections (if applicable)', 'IT', 'normal', '837/835 transaction flows'),
        ('5.12', 'Configure ML-DSA digital signatures on all outbound HL7 clinical messages', 'Security', 'normal', 'Non-repudiation for clinical messages'),
        ('5.13', 'Validate DICOM web services (WADO-RS, STOW-RS) -- post-Bridge', 'IT', 'verify', ''),
        ('5.14', 'Run 48-hour protocol monitoring to confirm zero message delivery failures', 'IT', 'verify', ''),
    ]
    for num, task, owner, priority, notes in phase5_items:
        story.append(ChecklistItem(num, task, owner, priority, notes))
    story.append(sp(6))
    story.append(SummaryBox(5, 'HL7/FHIR/DICOM Security', 14, 2))
    story.append(PageBreak())

    # ── PHASE 6: EHR/HIS Integration ──────────────────────────────────────────
    story.append(PhaseHeader(6, 'EHR/HIS Integration', 'Week 11-14',
                             'IT + Clinical Informatics'))
    story.append(sp(4))
    phase6_items = [
        ('6.1', 'Configure Bridge connector for Epic Interconnect API security', 'IT', 'normal', ''),
        ('6.2', 'Validate Epic MyChart FHIR API calls through Bridge (post-activation)', 'IT', 'verify', ''),
        ('6.3', 'Configure Epic Cosmos data pipeline security (if applicable)', 'IT', 'normal', ''),
        ('6.4', 'Configure Bridge connector for Cerner Millennium API (if applicable)', 'IT', 'normal', ''),
        ('6.5', 'Configure Bridge connector for MEDITECH Expanse (if applicable)', 'IT', 'normal', ''),
        ('6.6', 'Enable PHI access audit logging in Bridge for all EHR API calls', 'Compliance', 'critical', ''),
        ('6.7', 'Configure patient consent tracking for FHIR data sharing (if applicable)', 'Compliance', 'normal', 'Required for Da Vinci implementation guides'),
        ('6.8', 'Validate clinical analytics platform data feeds through Bridge', 'IT', 'verify', 'Health Catalyst, Arcadia, or equivalent'),
        ('6.9', 'Configure HIPAA audit trail for all PHI access events', 'Compliance', 'critical', ''),
        ('6.10', 'Validate audit trail completeness with Compliance Officer', 'Compliance', 'verify', ''),
        ('6.11', 'Generate first HIPAA compliance report from Bridge Compliance Engine', 'Compliance', 'verify', 'Validate report covers all 45 CFR 164.312 requirements'),
        ('6.12', 'Review HIPAA report with HIPAA Security Officer -- obtain sign-off', 'Compliance', 'critical', ''),
    ]
    for num, task, owner, priority, notes in phase6_items:
        story.append(ChecklistItem(num, task, owner, priority, notes))
    story.append(sp(6))
    story.append(SummaryBox(6, 'EHR/HIS Integration', 12, 4))
    story.append(PageBreak())

    # ── PHASE 7: Monitoring & Alerting ────────────────────────────────────────
    story.append(PhaseHeader(7, 'Monitoring & Alerting', 'Week 13-16',
                             'Security Team + SIEM Team'))
    story.append(sp(4))
    phase7_items = [
        ('7.1', 'Activate clinical anomaly detection on all protected device VLANs', 'Security', 'normal', ''),
        ('7.2', 'Establish device communication baselines (allow 7-day learning period)', 'Security', 'normal', 'Do not tune thresholds during learning period'),
        ('7.3', 'Configure PHI access volume anomaly alerts', 'Security', 'normal', ''),
        ('7.4', 'Configure device-to-new-destination alerts', 'Security', 'normal', ''),
        ('7.5', 'Configure lateral movement detection (cross-VLAN anomalies)', 'Security', 'critical', ''),
        ('7.6', 'Configure VLAN quarantine automation and test in lab (non-production)', 'Security', 'verify', 'Validate automated quarantine does not affect other devices'),
        ('7.7', 'Integrate Bridge alerts with SIEM (Splunk, Sentinel, QRadar)', 'Security', 'normal', ''),
        ('7.8', 'Configure HIPAA breach detection alerts (PHI exfiltration indicators)', 'Compliance', 'critical', ''),
        ('7.9', 'Configure ransomware precursor detection alerts', 'Security', 'critical', ''),
        ('7.10', 'Test alert delivery to on-call security team (end-to-end test)', 'Security', 'verify', ''),
        ('7.11', 'Configure Biomedical Engineering work order integration for quarantined devices', 'Biomedical', 'normal', ''),
        ('7.12', 'Configure executive security digest reports (weekly/monthly)', 'Security', 'normal', ''),
        ('7.13', 'Run tabletop exercise simulating compromised infusion pump detection', 'Security', 'verify', ''),
        ('7.14', 'Document SOC runbook for medical device security incidents', 'Security', 'normal', ''),
    ]
    for num, task, owner, priority, notes in phase7_items:
        story.append(ChecklistItem(num, task, owner, priority, notes))
    story.append(sp(6))
    story.append(SummaryBox(7, 'Monitoring & Alerting', 14, 4))
    story.append(PageBreak())

    # ── PHASE 8: Compliance Validation ────────────────────────────────────────
    story.append(PhaseHeader(8, 'Compliance Validation', 'Week 15-18',
                             'Compliance Officer + Security Team'))
    story.append(sp(4))
    phase8_items = [
        ('8.1', 'Generate full HIPAA 45 CFR 164.312 audit report -- validate all controls covered', 'Compliance', 'critical', ''),
        ('8.2', 'Review HIPAA audit report with external HIPAA counsel', 'Legal', 'critical', ''),
        ('8.3', 'Submit HIPAA audit report to Compliance Committee for approval', 'Compliance', 'critical', ''),
        ('8.4', 'Generate FDA cybersecurity documentation (SBOM) for all protected device classes', 'Compliance', 'normal', ''),
        ('8.5', 'Review FDA documentation with Biomedical Engineering and Risk Management', 'Biomedical', 'normal', ''),
        ('8.6', 'Generate HITRUST CSF evidence package for all applicable controls', 'Compliance', 'normal', 'If HITRUST certification in scope'),
        ('8.7', 'Review HITRUST evidence with HITRUST assessor (if applicable)', 'Compliance', 'verify', ''),
        ('8.8', 'Generate SOC 2 Type II evidence for Security and Availability trust services', 'Compliance', 'normal', ''),
        ('8.9', 'Validate breach detection capability: simulate PHI exfiltration event', 'Security', 'verify', 'Confirm 15-minute detection SLA'),
        ('8.10', 'Test HIPAA breach notification report generation (less than 10 minutes)', 'Compliance', 'verify', ''),
        ('8.11', 'Validate 21 CFR Part 11 electronic signature compliance for audit records', 'Compliance', 'normal', ''),
        ('8.12', 'Document post-quantum cryptography algorithm implementation for compliance records', 'Security', 'normal', 'FIPS 203/204/205 documentation'),
        ('8.13', 'Obtain Compliance Officer sign-off on all compliance documentation', 'Compliance', 'critical', ''),
    ]
    for num, task, owner, priority, notes in phase8_items:
        story.append(ChecklistItem(num, task, owner, priority, notes))
    story.append(sp(6))
    story.append(SummaryBox(8, 'Compliance Validation', 13, 6))
    story.append(PageBreak())

    # ── PHASE 9: Clinical UAT ──────────────────────────────────────────────────
    story.append(PhaseHeader(9, 'Clinical User Acceptance Testing', 'Week 17-20',
                             'Clinical Informatics + Biomedical Engineering'))
    story.append(sp(4))
    phase9_items = [
        ('9.1', 'Brief clinical leadership (CNO, CMO, department heads) on Bridge deployment', 'Clinical', 'critical', ''),
        ('9.2', 'Define clinical test scenarios per department (ICU, ED, OR, Radiology)', 'Clinical', 'normal', ''),
        ('9.3', 'Conduct ICU clinical workflow validation -- nursing and intensivist sign-off', 'Clinical', 'critical', 'All alarm delivery tested'),
        ('9.4', 'Conduct ED clinical workflow validation', 'Clinical', 'critical', ''),
        ('9.5', 'Conduct OR clinical workflow validation -- Anesthesia and OR nursing sign-off', 'Clinical', 'critical', 'Coordinate with OR schedule; no active cases'),
        ('9.6', 'Conduct Radiology workflow validation -- validate PACS performance unchanged', 'Radiology', 'verify', 'Timing comparison: pre- and post-Bridge image load times'),
        ('9.7', 'Conduct Pharmacy automation workflow validation', 'Clinical', 'normal', ''),
        ('9.8', 'Conduct Laboratory workflow validation', 'Clinical', 'normal', ''),
        ('9.9', 'Validate infusion pump alarm delivery in clinical test environment', 'Biomedical', 'critical', 'Test all alarm types: occlusion, air, low battery, dose limit'),
        ('9.10', 'Validate ventilator alarm delivery', 'Biomedical', 'critical', ''),
        ('9.11', 'Validate cardiac monitor alarm delivery (ICU and telemetry)', 'Biomedical', 'critical', ''),
        ('9.12', 'Document any clinical workflow concerns and resolve before go-live', 'Clinical', 'critical', ''),
        ('9.13', 'Obtain written sign-off from Biomedical Engineering Director', 'Biomedical', 'critical', ''),
        ('9.14', 'Obtain written sign-off from Chief Nursing Officer or delegate', 'Clinical', 'critical', ''),
        ('9.15', 'Obtain written sign-off from Chief Medical Officer or delegate', 'Clinical', 'critical', ''),
    ]
    for num, task, owner, priority, notes in phase9_items:
        story.append(ChecklistItem(num, task, owner, priority, notes))
    story.append(sp(6))
    story.append(SummaryBox(9, 'Clinical UAT', 15, 9))
    story.append(PageBreak())

    # ── PHASE 10: Go-Live & Handover ───────────────────────────────────────────
    story.append(PhaseHeader(10, 'Go-Live & Handover', 'Week 19-22',
                             'All Stakeholders'))
    story.append(sp(4))
    phase10_items = [
        ('10.1', 'Confirm all Phase 0-9 sign-offs complete before scheduling go-live', 'PM', 'critical', ''),
        ('10.2', 'Schedule go-live during low-census period (weekend, holiday week)', 'PM', 'normal', ''),
        ('10.3', 'Notify all department heads and clinical staff of go-live date/time', 'PM', 'normal', ''),
        ('10.4', 'Confirm QBITEL Clinical Security Engineering team on-site for go-live', 'PM', 'critical', ''),
        ('10.5', 'Confirm Biomedical Engineering on-site during go-live window', 'Biomedical', 'critical', ''),
        ('10.6', 'Activate production Bridge configuration on primary Edge Node', 'Security', 'critical', ''),
        ('10.7', 'Monitor all clinical device VLANs for first 4 hours post-activation', 'Security', 'verify', ''),
        ('10.8', 'Validate anomaly detection is active and generating baseline alerts', 'Security', 'verify', ''),
        ('10.9', 'Confirm SIEM integration is receiving Bridge events', 'Security', 'verify', ''),
        ('10.10', 'Conduct 24-hour post-go-live review call with clinical leadership', 'PM', 'normal', ''),
        ('10.11', 'Handover Bridge management to internal security team (with training)', 'Security', 'critical', ''),
        ('10.12', 'Confirm 24/7 CSOC monitoring is active (QBITEL CSOC)', 'Security', 'normal', ''),
        ('10.13', 'Schedule 30-day post-deployment review meeting', 'PM', 'normal', ''),
        ('10.14', 'Generate first production HIPAA compliance report for Compliance Officer', 'Compliance', 'verify', ''),
        ('10.15', 'Submit Bridge deployment documentation to CISO and Board security committee', 'Security', 'critical', ''),
        ('10.16', 'Notify cyber insurance carrier of quantum-safe deployment for premium review', 'Legal', 'normal', 'Premium reduction typically 20-35%'),
    ]
    for num, task, owner, priority, notes in phase10_items:
        story.append(ChecklistItem(num, task, owner, priority, notes))
    story.append(sp(6))
    story.append(SummaryBox(10, 'Go-Live & Handover', 16, 7))
    story.append(PageBreak())

    # Summary page
    S_big = ParagraphStyle('big', fontName='Helvetica-Bold', fontSize=16, textColor=NAVY,
                           leading=20, spaceAfter=10, alignment=TA_CENTER)
    S_med = ParagraphStyle('med', fontName='Helvetica', fontSize=10, textColor=DARK_TEXT,
                           leading=14, spaceAfter=6, alignment=TA_CENTER)
    story.append(Paragraph('DEPLOYMENT COMPLETE', S_big))
    story.append(Paragraph(
        'Congratulations. Your organization has achieved quantum-safe PHI protection '
        'for all connected medical devices, clinical protocols, and EHR integrations.',
        S_med
    ))
    story.append(sp(12))

    summary_data = [
        ['Phase', 'Name', 'Items', 'Critical', 'Sign-off'],
        ['0', 'Pre-Engagement', '12', '3', '___________'],
        ['1', 'Medical Device Discovery', '14', '5', '___________'],
        ['2', 'Infrastructure Readiness', '14', '5', '___________'],
        ['3', 'Policy Configuration', '12', '5', '___________'],
        ['4', 'Medical Device Protection', '15', '7', '___________'],
        ['5', 'HL7/FHIR/DICOM Security', '14', '2', '___________'],
        ['6', 'EHR/HIS Integration', '12', '4', '___________'],
        ['7', 'Monitoring & Alerting', '14', '4', '___________'],
        ['8', 'Compliance Validation', '13', '6', '___________'],
        ['9', 'Clinical UAT', '15', '9', '___________'],
        ['10', 'Go-Live & Handover', '16', '7', '___________'],
        ['TOTAL', '', '151+', '57', '___________'],
    ]

    S_th = ParagraphStyle('th', fontName='Helvetica-Bold', fontSize=9, textColor=WHITE_C)
    S_td = ParagraphStyle('td', fontName='Helvetica', fontSize=9, textColor=DARK_TEXT)
    S_td_b = ParagraphStyle('tdb', fontName='Helvetica-Bold', fontSize=9, textColor=NAVY)

    tbl_data = [[Paragraph(h, S_th) for h in summary_data[0]]]
    for ri, row in enumerate(summary_data[1:]):
        sty = S_td_b if ri == len(summary_data) - 2 else S_td
        tbl_data.append([Paragraph(c, sty) for c in row])

    tbl = Table(tbl_data, colWidths=[0.6*inch, 2.8*inch, 0.7*inch, 0.7*inch, 1.5*inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        *[('ROWBACKGROUND', (0, i), (-1, i), TABLE_ALT if i % 2 == 1 else WHITE_C)
          for i in range(1, len(tbl_data))],
        ('ROWBACKGROUND', (0, len(tbl_data)-1), (-1, len(tbl_data)-1), LIGHT_NAVY),
        ('TEXTCOLOR', (0, len(tbl_data)-1), (-1, len(tbl_data)-1), WHITE_C),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#C8D8E8')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(tbl)
    story.append(sp(16))

    story.append(Paragraph('QBITEL BRIDGE -- Protecting the Healers Who Protect Us', S_big))
    story.append(Paragraph('enterprise@qbitel.com  |  https://bridge.qbitel.com', S_med))

    doc.build(story)
    print(f'Checklist PDF written: {out_path}')


if __name__ == '__main__':
    build_checklist_pdf()
