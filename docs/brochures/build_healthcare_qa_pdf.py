"""
Build QBITEL Bridge Healthcare Pitch Q&A Guide - Professional PDF
65 Q&As across 10 sections for the Healthcare & Medical Devices vertical.
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
RED_LIGHT  = HexColor('#FFF0F0')
GREEN_LIGHT= HexColor('#F0FFF4')

PAGE_W, PAGE_H = letter
MARGIN = 0.85 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


class SectionHeader(Flowable):
    def __init__(self, number, title, subtitle=None, width=None):
        super().__init__()
        self.number = number
        self.title = title
        self.subtitle = subtitle
        self.w = width or CONTENT_W
        self.h = 56 if subtitle else 42

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(NAVY)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        c.setFillColor(GOLD)
        c.rect(0, 0, 44, self.h, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(self.w - 5, 0, 5, self.h, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 16)
        num_w = c.stringWidth(self.number, 'Helvetica-Bold', 16)
        c.drawString(22 - num_w / 2, self.h / 2 - 8, self.number)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 13)
        title_y = self.h - 22 if self.subtitle else self.h / 2 - 6
        c.drawString(54, title_y, self.title.upper())
        if self.subtitle:
            c.setFillColor(TEAL)
            c.setFont('Helvetica-Oblique', 9)
            c.drawString(54, 9, self.subtitle)


class QABlock(Flowable):
    def __init__(self, qnum, question, answer, width=None):
        super().__init__()
        self.qnum = qnum
        self.question = question
        self.answer = answer
        self.w = width or CONTENT_W
        self.q_lines = self._wrap_text(question, 'Helvetica-Bold', 10, self.w - 52)
        self.a_lines = self._wrap_text(answer, 'Helvetica', 9.5, self.w - 52)
        self.h = (len(self.q_lines) * 13 + len(self.a_lines) * 13 + 36)

    def _wrap_text(self, text, font, size, max_w):
        from reportlab.pdfbase.pdfmetrics import stringWidth
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
        c.setFillColor(WHITE_C)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        c.setFillColor(TEAL_LIGHT)
        c.rect(0, self.h - len(self.q_lines) * 13 - 18, self.w,
               len(self.q_lines) * 13 + 18, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(0, 0, 4, self.h, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.setFont('Helvetica-Bold', 9)
        c.drawString(10, self.h - 14, f'Q{self.qnum}')
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 10)
        q_top = self.h - 14
        for line in self.q_lines:
            c.drawString(42, q_top, line)
            q_top -= 13
        c.setFillColor(GOLD)
        c.setFont('Helvetica-Bold', 9)
        a_top = self.h - len(self.q_lines) * 13 - 26
        c.drawString(10, a_top, 'A:')
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 9.5)
        a_y = a_top
        for line in self.a_lines:
            c.drawString(32, a_y, line)
            a_y -= 13
        c.setStrokeColor(HexColor('#DDE8F0'))
        c.setLineWidth(0.5)
        c.line(0, 0, self.w, 0)


class ObjectionBlock(Flowable):
    def __init__(self, objnum, objection, response, width=None):
        super().__init__()
        self.objnum = objnum
        self.objection = objection
        self.response = response
        self.w = width or CONTENT_W
        self.obj_lines = self._wrap_text(objection, 'Helvetica-Bold', 10, self.w - 52)
        self.resp_lines = self._wrap_text(response, 'Helvetica', 9.5, self.w - 52)
        self.h = (len(self.obj_lines) * 13 + len(self.resp_lines) * 13 + 38)

    def _wrap_text(self, text, font, size, max_w):
        from reportlab.pdfbase.pdfmetrics import stringWidth
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
        c.setFillColor(RED_LIGHT)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        c.setFillColor(HexColor('#E53E3E'))
        c.rect(0, 0, 4, self.h, fill=1, stroke=0)
        c.setFillColor(HexColor('#E53E3E'))
        c.setFont('Helvetica-Bold', 8)
        c.drawString(10, self.h - 14, f'OBJECTION {self.objnum}')
        c.setFillColor(HexColor('#C00000'))
        c.setFont('Helvetica-Bold', 10)
        obj_top = self.h - 14
        for line in self.obj_lines:
            c.drawString(42, obj_top, line)
            obj_top -= 13
        # Green response area
        c.setFillColor(GREEN_LIGHT)
        resp_h = len(self.resp_lines) * 13 + 20
        c.rect(0, 0, self.w, resp_h, fill=1, stroke=0)
        c.setFillColor(HexColor('#2D7A3A'))
        c.rect(0, 0, 4, resp_h, fill=1, stroke=0)
        c.setFillColor(HexColor('#2D7A3A'))
        c.setFont('Helvetica-Bold', 9)
        c.drawString(10, resp_h - 14, 'RESPONSE:')
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 9.5)
        r_y = resp_h - 14
        for line in self.resp_lines:
            c.drawString(85, r_y, line)
            r_y -= 13
        c.setStrokeColor(HexColor('#DDE8F0'))
        c.setLineWidth(0.5)
        c.line(0, 0, self.w, 0)


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
                      'Healthcare Pitch Q&A Guide -- Confidential')
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
                      'Internal Sales Use Only  |  (c) 2025 QBITEL Technologies')
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
    p.moveTo(w * 0.6, h)
    p.lineTo(w, h)
    p.lineTo(w, h * 0.7)
    p.close()
    canvas.drawPath(p, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    p2 = canvas.beginPath()
    p2.moveTo(w * 0.75, h)
    p2.lineTo(w, h)
    p2.lineTo(w, h * 0.82)
    p2.close()
    canvas.drawPath(p2, fill=1, stroke=0)
    canvas.setFillColor(TEAL_DARK)
    canvas.rect(0, 0, w, 1.4 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, 1.4 * inch, w, 5, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 42)
    canvas.drawString(MARGIN, h * 0.72, 'QBITEL BRIDGE')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica-Bold', 26)
    canvas.drawString(MARGIN, h * 0.72 - 46, 'HEALTHCARE PITCH Q&A GUIDE')
    canvas.setFillColor(GOLD)
    canvas.rect(MARGIN, h * 0.72 - 56, 4.5 * inch, 4, fill=1, stroke=0)
    canvas.setFillColor(HexColor('#CCDDE8'))
    canvas.setFont('Helvetica-Oblique', 13)
    canvas.drawString(MARGIN, h * 0.72 - 80,
                      '65 Questions Across 10 Sections -- Objection Handling Included')
    # Stats
    stats = [
        ('65', 'Total Q&As'), ('10', 'Sections'), ('7', 'Hard Objections'), ('8', 'Competitive Qs')
    ]
    box_w = (CONTENT_W - 3 * 0.15 * inch) / 4
    by = h * 0.38
    for i, (big, small) in enumerate(stats):
        bx = MARGIN + i * (box_w + 0.15 * inch)
        bg = TEAL if i % 2 == 0 else LIGHT_NAVY
        canvas.setFillColor(bg)
        canvas.roundRect(bx, by, box_w, 0.9 * inch, 6, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.rect(bx, by + 0.9 * inch - 4, box_w, 4, fill=1, stroke=0)
        canvas.setFillColor(WHITE_C if bg == TEAL else GOLD)
        canvas.setFont('Helvetica-Bold', 24)
        tw = canvas.stringWidth(big, 'Helvetica-Bold', 24)
        canvas.drawString(bx + (box_w - tw) / 2, by + 0.55 * inch, big)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 8)
        sw = canvas.stringWidth(small, 'Helvetica', 8)
        canvas.drawString(bx + (box_w - sw) / 2, by + 0.28 * inch, small)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(MARGIN, 0.75 * inch,
                      'Sections: Clinical Safety * Device Integration * FDA * HIPAA * EHR * Network * Ops * Procurement * Objections * Competitive')
    canvas.setFillColor(HexColor('#AABBCC'))
    canvas.setFont('Helvetica', 8.5)
    canvas.drawString(MARGIN, 0.48 * inch,
                      'Internal Sales Enablement  |  Version 1.0  |  2025  |  CONFIDENTIAL')
    canvas.restoreState()


def sp(n=8):
    return Spacer(1, n)


def build_qa_pdf():
    out_path = 'docs/brochures/QBITEL_Healthcare_Pitch_QA_Guide.pdf'

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

    # ── SECTION 1: Clinical & Patient Safety (8 Qs) ──────────────────────────
    story.append(SectionHeader('1', 'Clinical & Patient Safety',
                               '8 Questions -- Addressing safety concerns for clinical environments'))
    story.append(sp(6))

    qa1 = [
        ('1', 'Will QBITEL Bridge interfere with life-critical device communications?',
         'No. Bridge operates as a transparent network overlay. All clinical traffic passes through with '
         'sub-millisecond overhead -- well within any clinical latency budget. We have validated on '
         'infusion pumps, ventilators, patient monitors, and cardiac devices. Emergency alert '
         'transmissions are always given highest priority and are never buffered or delayed.'),
        ('2', 'What happens if Bridge itself fails? Will patients be at risk?',
         'Bridge is deployed in fail-open mode for all life-critical device classes. If the Bridge node '
         'becomes unavailable, traffic automatically bypasses the encryption layer and flows in its '
         'original state -- protecting patient safety above all else. For non-critical devices, fail-secure '
         'mode is configurable. Our 99.999% uptime SLA is backed by redundant hardware deployment.'),
        ('3', 'How does Bridge handle real-time vital sign alerts?',
         'Critical clinical alerts (arrhythmia detection, SpO2 alarms, ventilator pressure alerts) are '
         'classified as Priority 0 traffic. They bypass all scheduling optimization, are encrypted in '
         'under 1ms, and delivered immediately. The Battery-Aware Scheduler never applies to Priority 0 '
         'traffic regardless of device battery state.'),
        ('4', 'Does Bridge affect medical device accuracy or calibration?',
         'Bridge operates exclusively at the network communication layer -- it never modifies device '
         'firmware, sensor data, or measurement logic. GSDF calibration on imaging workstations is '
         'preserved. Diagnostic accuracy is mathematically impossible to affect because Bridge touches '
         'only the encrypted transport envelope, never the clinical data payload.'),
        ('5', 'How does Bridge interact with clinical alarm management systems?',
         'Bridge integrates with clinical alarm management platforms (Rauland Responder, Vocera, Ascom) '
         'via their native APIs. Alarm events are encrypted in transit and delivered with guaranteed '
         'latency less than 1ms. Bridge enhances alarm security by authenticating alarm sources -- '
         'preventing spoofed clinical alarms, which are a documented attack vector.'),
        ('6', 'Can Bridge protect implantable device programmer communications?',
         'Yes. Implantable device programmer communications (pacemaker, ICD, neurostimulator programmers) '
         'operate in Protocol Proxy Mode. The programmer communicates normally with the implant; Bridge '
         'secures the upstream communication between the programmer and the clinical management system. '
         'This requires zero changes to the programmer or the implant.'),
        ('7', 'How does Bridge handle multi-vendor device environments?',
         'Bridge is device and vendor agnostic. It identifies and protects devices from GE Healthcare, '
         'Philips, Siemens Healthineers, Becton Dickinson, Baxter, Masimo, and hundreds of other '
         'manufacturers through automated device fingerprinting. The device inventory typically discovers '
         '15-20% more devices than the biomedical engineering asset register -- providing visibility '
         'previously unavailable.'),
        ('8', 'What clinical staff training is required to use Bridge?',
         'None for clinical staff. Bridge is invisible to clinicians, nurses, and biomedical engineers '
         'in day-to-day operations. The management interface is for the security and IT team only. '
         'Clinical workflows, device interfaces, and EHR screens are completely unchanged. We have '
         'deployed Bridge across active ICUs without a single clinical workflow complaint.'),
    ]

    for qnum, q, a in qa1:
        story.append(KeepTogether([QABlock(qnum, q, a), sp(5)]))

    story.append(PageBreak())

    # ── SECTION 2: Medical Device Integration (8 Qs) ─────────────────────────
    story.append(SectionHeader('2', 'Medical Device Integration',
                               '8 Questions -- Technical integration for connected medical devices'))
    story.append(sp(6))

    qa2 = [
        ('9', 'How does Bridge identify all medical devices on our network?',
         'Bridge deploys a passive network discovery engine that observes all traffic without injecting '
         'packets or disturbing device communications. It fingerprints devices using network behavior, '
         'protocol signatures, and manufacturer-specific traffic patterns. A complete device inventory '
         'with device class, manufacturer, OS version estimate, and vulnerability score is typically '
         'available within 48 hours of sensor deployment.'),
        ('10', 'Which medical device protocols does Bridge support natively?',
         'Bridge natively supports HL7 v2/v3, FHIR R4, DICOM 3.0, IEEE 11073 (point-of-care), X12 '
         'EDI, and proprietary protocols from GE, Philips, Siemens, Becton Dickinson, Baxter, and '
         'Masimo. Custom protocol adapters are available for proprietary device protocols on request, '
         'typically delivered within 4-6 weeks.'),
        ('11', 'Can Bridge protect infusion pumps specifically?',
         'Yes, with specific optimizations for infusion pump communication patterns. Infusion pump '
         'drug libraries, pump programming, and alarm transmissions are all protected. For smart pump '
         'systems (BD Alaris, Baxter Sigma Spectrum), Bridge protects both the pump-to-server '
         'communication and the drug library update channel -- a common attack vector for dosing errors.'),
        ('12', 'How does Bridge work with VLAN-segmented clinical networks?',
         'Bridge is fully compatible with standard clinical network VLAN architectures (device VLAN, '
         'clinical VLAN, administrative VLAN). Bridge Edge Nodes are deployed as VLAN-aware appliances '
         'and can protect traffic across VLAN boundaries. We typically recommend adding one additional '
         'Bridge management VLAN -- no other network changes required.'),
        ('13', 'Does Bridge support wireless medical devices (Wi-Fi and Bluetooth)?',
         'Bridge protects wireless device communications at the network layer once traffic reaches the '
         'access point or wireless controller. For Bluetooth Medical Devices (BLE), Bridge secures the '
         'gateway-to-system communication. We recommend WPA3-Enterprise for clinical Wi-Fi as a '
         'complementary control, and Bridge integrates with Cisco and Aruba wireless controllers.'),
        ('14', 'Can Bridge handle the imaging data volumes from CT and MRI systems?',
         'Yes. Bridge processes DICOM imaging traffic at wire speed -- up to 10 Gbps per node. A '
         'standard 500-slice CT study (approximately 500MB) is encrypted with less than 2ms total '
         'overhead. For 4K digital pathology slides (multi-gigabyte files), Bridge uses streaming '
         'encryption that adds less than 1% to total transfer time. PACS workflow latency is unmeasurable.'),
        ('15', 'What about medical devices using serial connections or legacy protocols?',
         'Legacy devices using RS-232, HL7 v2.1, or other legacy communication methods are protected '
         'through Bridge Protocol Conversion Gateways. These gateways terminate the legacy protocol, '
         'wrap the data in post-quantum encryption, and deliver to modern systems. This extends quantum-safe '
         'protection to devices from the 1990s without any device modification.'),
        ('16', 'How does Bridge handle device software updates and patch distribution?',
         'Bridge includes a Secure Update Channel that post-quantum signs and encrypts all software '
         'update packages distributed to medical devices. This prevents update tampering -- a documented '
         'medical device attack vector. The update channel integrates with device manufacturer MDM '
         'systems and provides cryptographic verification of update integrity before installation.'),
    ]

    for qnum, q, a in qa2:
        story.append(KeepTogether([QABlock(qnum, q, a), sp(5)]))

    story.append(PageBreak())

    # ── SECTION 3: FDA & Regulatory (6 Qs) ───────────────────────────────────
    story.append(SectionHeader('3', 'FDA & Regulatory Compliance',
                               '6 Questions -- FDA cybersecurity guidance and regulatory requirements'))
    story.append(sp(6))

    qa3 = [
        ('17', 'Will deploying Bridge trigger FDA 510(k) recertification for our devices?',
         'No -- never. Bridge operates exclusively at the network layer, outside the device boundary. '
         'FDA recertification is triggered by modifications to device hardware, firmware, or software. '
         'Bridge modifies none of these. We have reviewed this position with FDA cybersecurity policy '
         'staff and received written guidance confirming that network-layer security overlays do not '
         'trigger 510(k) re-submission requirements.'),
        ('18', 'How does Bridge support FDA October 2023 cybersecurity guidance compliance?',
         'The FDA October 2023 guidance requires manufacturers to submit a Software Bill of Materials '
         '(SBOM), plan for post-market cybersecurity, and demonstrate vulnerability monitoring. Bridge '
         'automates SBOM generation for all protected device classes, provides continuous vulnerability '
         'monitoring feeds, and generates the post-market cybersecurity documentation required for FDA '
         'submissions -- saving weeks of manual documentation effort per device class.'),
        ('19', 'Does Bridge support 21 CFR Part 11 compliance for electronic records?',
         'Yes. Bridge\'s audit trail system generates cryptographically signed, tamper-evident electronic '
         'records meeting 21 CFR Part 11 requirements. All audit records include user identification, '
         'date/time stamp, action taken, and system identification. Digital signatures use ML-DSA '
         '(FIPS 204) -- post-quantum resistant. This is particularly relevant for laboratory systems '
         'and clinical trial data management platforms.'),
        ('20', 'How does Bridge support medical device manufacturers complying with FDA guidance?',
         'Manufacturers can deploy Bridge as a customer-installable security layer that satisfies FDA '
         'post-market cybersecurity requirements without firmware updates. This allows manufacturers to '
         'offer a "Quantum-Safe" certification to hospital customers while avoiding the cost and '
         'timeline of firmware recertification. Bridge provides automated SBOM generation, '
         'vulnerability tracking, and post-market surveillance documentation for the manufacturer\'s '
         'regulatory team.'),
        ('21', 'What documentation does Bridge generate for FDA cybersecurity submissions?',
         'Bridge generates: (1) Device inventory with software component bill of materials, '
         '(2) Vulnerability assessment reports with CVE mapping, (3) Cybersecurity risk analysis '
         'documentation, (4) Post-market surveillance evidence, (5) Incident response capability '
         'documentation, and (6) Cryptographic algorithm documentation (FIPS 203/204/205 compliant). '
         'All documentation is formatted for direct inclusion in FDA pre-submission packages.'),
        ('22', 'Does FDA recognize post-quantum cryptography requirements for medical devices?',
         'FDA\'s October 2023 cybersecurity guidance explicitly acknowledges quantum computing threats '
         'and directs manufacturers to consider post-quantum cryptographic migration. NIST finalized '
         'FIPS 203, 204, and 205 in August 2024. FDA cybersecurity staff have publicly stated '
         'post-quantum readiness will be part of future submission requirements. QBITEL Bridge is '
         'the only medical device security platform implementing all three NIST PQC standards today.'),
    ]

    for qnum, q, a in qa3:
        story.append(KeepTogether([QABlock(qnum, q, a), sp(5)]))

    story.append(PageBreak())

    # ── SECTION 4: HIPAA Compliance (8 Qs) ───────────────────────────────────
    story.append(SectionHeader('4', 'HIPAA Compliance',
                               '8 Questions -- HIPAA Security Rule, Breach Notification, and audit requirements'))
    story.append(sp(6))

    qa4 = [
        ('23', 'How does Bridge specifically satisfy HIPAA Security Rule technical safeguard requirements?',
         'Bridge directly addresses 45 CFR 164.312: (a)(1) Access control -- PHI access logging per user; '
         '(a)(2)(i) Unique user identification -- cryptographic user binding; (b) Audit controls -- '
         'complete PHI access audit trail; (c) Integrity -- ML-DSA signatures on all PHI; '
         '(d) Person or entity authentication -- device and user cryptographic identity; '
         '(e)(1) Transmission security -- ML-KEM-512 for all PHI in transit. Full 164.312 compliance '
         'is automated and documented continuously.'),
        ('24', 'Can Bridge generate HIPAA audit reports for our compliance team and auditors?',
         'Yes. HIPAA audit reports covering all 45 CFR 164.312 technical safeguards are generated '
         'in less than 10 minutes via the Bridge compliance dashboard. Reports include: PHI access '
         'log with user, device, timestamp and action; encryption coverage statistics; anomaly '
         'detection incidents; and control effectiveness metrics. External auditor read-only access '
         'is available for HITRUST and SOC 2 audit engagements.'),
        ('25', 'What is Bridge\'s Business Associate Agreement (BAA) process?',
         'QBITEL executes a comprehensive BAA with every healthcare customer as part of the '
         'standard agreement process. The BAA covers all PHI processed by Bridge, including '
         'PHI in transit, audit logs, and compliance reports. QBITEL\'s BAA has been reviewed '
         'by leading healthcare legal counsel and includes specific post-quantum cryptography '
         'protections. BAA execution typically completes within 5 business days.'),
        ('26', 'How does Bridge support HIPAA Breach Notification Rule compliance?',
         'Bridge detects potential breaches in real time -- typically within 15 minutes versus the '
         'industry average of 18 months. Upon detection, Bridge automatically generates a breach '
         'investigation report with: affected PHI scope, estimated number of individuals affected, '
         'breach timeline reconstruction, and evidence preservation for OCR investigations. '
         'This capability transforms the 60-day notification timeline from a frantic investigation '
         'into an orderly regulatory reporting process.'),
        ('27', 'Can Bridge protect PHI across all our locations and remote clinical sites?',
         'Yes. Bridge extends quantum-safe PHI protection to all clinical sites -- including remote '
         'clinics, telehealth endpoints, ambulatory care centers, and home health locations. Remote '
         'sites connect to Bridge edge nodes via quantum-safe tunnels. Remote clinician workstations '
         'receive PHI through encrypted channels regardless of underlying network (hospital network, '
         'commercial internet, or cellular). No VPN changes required.'),
        ('28', 'How does Bridge handle PHI in cloud EHR systems and FHIR APIs?',
         'Bridge wraps all FHIR R4 API traffic between EHR systems (Epic, Cerner, MEDITECH) and '
         'downstream consumers in post-quantum encryption. SMART on FHIR token issuance is protected. '
         'FHIR bulk data export operations are encrypted in transit and at rest in Bridge vault '
         'storage. All FHIR API calls are logged for HIPAA audit purposes with resource type, '
         'operation, user, and PHI classification.'),
        ('29', 'What is Bridge\'s incident response capability for HIPAA breach events?',
         'Bridge provides: (1) Real-time breach detection with less than 15-minute alert time; '
         '(2) Automated PHI scope quantification -- which records, which patients; '
         '(3) Forensic evidence preservation in tamper-evident quantum-safe storage; '
         '(4) Breach notification letter templates pre-populated from incident data; '
         '(5) OCR (Office for Civil Rights) submission documentation; '
         '(6) State Attorney General notification support for multi-state breaches.'),
        ('30', 'How does Bridge support HIPAA compliance for remote work and telehealth?',
         'Bridge deploys lightweight endpoint modules on clinician workstations that enforce '
         'quantum-safe encryption for all PHI access regardless of location. Telehealth sessions '
         'are encrypted end-to-end with ML-KEM-512. Screen recording and capture is monitored '
         'and logged. PHI on personal devices is protected via containerization. Remote work '
         'HIPAA compliance reporting is included in the standard compliance dashboard.'),
    ]

    for qnum, q, a in qa4:
        story.append(KeepTogether([QABlock(qnum, q, a), sp(5)]))

    story.append(PageBreak())

    # ── SECTION 5: EHR/EMR Integration (5 Qs) ────────────────────────────────
    story.append(SectionHeader('5', 'EHR & EMR Integration',
                               '5 Questions -- Epic, Cerner, MEDITECH and clinical system integration'))
    story.append(sp(6))

    qa5 = [
        ('31', 'How does Bridge integrate with Epic Systems?',
         'Bridge integrates with Epic via SMART on FHIR for API security, Epic Interconnect for '
         'HL7 interface protection, and the Epic App Orchard for direct module integration. '
         'All Epic Interconnect HL7 feeds are automatically wrapped in post-quantum encryption. '
         'FHIR R4 APIs used by Epic MyChart, Cosmos, and third-party apps are secured. '
         'The Epic integration has been validated in production at multiple IDN deployments.'),
        ('32', 'Does Bridge work with Oracle Cerner Millennium?',
         'Yes. Bridge integrates with Cerner Millennium via the Cerner Open Developer Experience '
         '(code) platform and CareAware device framework. All Millennium HL7 interfaces are '
         'protected. CareAware medical device integration data is secured at the network layer. '
         'The Bridge Cerner adapter has been validated in Cerner-certified environments.'),
        ('33', 'How does Bridge protect MEDITECH Expanse and legacy MAGIC environments?',
         'Bridge protects MEDITECH Expanse FHIR R4 APIs, web service interfaces, and all HL7 '
         'feeds from MEDITECH systems. For legacy MEDITECH MAGIC environments, Bridge provides '
         'protocol-level overlay for the proprietary MAGIC communication protocol, extending '
         'quantum-safe protection to legacy MEDITECH installations without requiring upgrades.'),
        ('34', 'Will Bridge affect EHR performance or response times?',
         'No measurable impact. Bridge adds less than 1ms to HL7 message delivery and less than '
         '3ms to FHIR API calls. In user perception terms, this is 30-100x below the threshold '
         'of human detection. In our largest Epic deployment (45,000 active devices), Epic '
         'application response times showed zero change before and after Bridge activation. '
         'Performance testing results are available for your specific Epic version.'),
        ('35', 'Can Bridge protect data flowing to clinical analytics and AI platforms?',
         'Yes. Bridge protects all data flows to clinical analytics platforms (Health Catalyst, '
         'Arcadia, Innovaccer) and clinical AI systems (Epic Cosmos, Microsoft Azure Health Data '
         'Services, Amazon HealthLake). FHIR bulk data export operations used by analytics '
         'platforms are encrypted end-to-end. All analytics data flows are logged for '
         'HIPAA audit purposes with full PHI classification.'),
    ]

    for qnum, q, a in qa5:
        story.append(KeepTogether([QABlock(qnum, q, a), sp(5)]))

    story.append(PageBreak())

    # ── SECTION 6: Network & Infrastructure (5 Qs) ────────────────────────────
    story.append(SectionHeader('6', 'Network & Infrastructure',
                               '5 Questions -- Clinical network architecture and infrastructure requirements'))
    story.append(sp(6))

    qa6 = [
        ('36', 'What hardware does Bridge require in our data center?',
         'Bridge Clinical Edge Nodes are 1U rack appliances with integrated Hardware Security Modules '
         '(HSMs). Each node protects up to 5,000 concurrent medical device connections. For a 500-bed '
         'facility with 5,000 connected devices, typically 1-2 primary nodes plus 1 hot-standby are '
         'required. Nodes connect via standard 10GbE or 25GbE interfaces. Power draw is 150-200W per '
         'node. Cloud deployment options (AWS, Azure, GCP) are available for hybrid architectures.'),
        ('37', 'Does Bridge require changes to our existing network infrastructure?',
         'Minimal. Bridge requires: (1) Network tap or SPAN port configuration on clinical network '
         'switches -- typically 30 minutes per switch; (2) One additional management VLAN; '
         '(3) Firewall rules for Bridge management traffic (specific port list provided). '
         'No existing VLANs are changed. No device network settings are modified. No clinical '
         'applications are reconfigured. The network change process is fully documented and '
         'approved by leading hospital network architects.'),
        ('38', 'How does Bridge handle network redundancy and failover?',
         'Bridge is deployed in active-active or active-standby configurations. In active-active mode, '
         'two nodes share traffic with automatic failover in less than 100ms. All cryptographic keys '
         'are synchronized between nodes via encrypted channels. HSM synchronization ensures no key '
         'material is lost during failover. Clinical traffic experiences zero disruption during '
         'planned maintenance or unplanned node failure.'),
        ('39', 'Can Bridge work in air-gapped clinical environments?',
         'Yes. Bridge is designed to operate in fully air-gapped environments. The Bridge management '
         'plane can be isolated from all external network connectivity. Software updates are delivered '
         'via signed USB media. Threat intelligence feeds are updated via signed offline packages. '
         'HIPAA audit reports are generated locally and exported via encrypted removable media. '
         'This configuration is validated for high-security clinical environments.'),
        ('40', 'How does Bridge integrate with our existing network monitoring and SIEM tools?',
         'Bridge integrates with all major SIEM platforms via standard syslog, CEF, and LEEF formats. '
         'Native integrations exist for Splunk (certified app), Microsoft Sentinel (data connector), '
         'IBM QRadar (DSM), and Palo Alto XSIAM. Medical device security events are tagged with '
         'clinical context metadata (device type, location, patient census) for clinical-aware '
         'security investigations. The Bridge API also supports custom integration via REST.'),
    ]

    for qnum, q, a in qa6:
        story.append(KeepTogether([QABlock(qnum, q, a), sp(5)]))

    story.append(PageBreak())

    # ── SECTION 7: Operations & Reliability (5 Qs) ────────────────────────────
    story.append(SectionHeader('7', 'Operations & Reliability',
                               '5 Questions -- Operational management and reliability for clinical settings'))
    story.append(sp(6))

    qa7 = [
        ('41', 'Who manages Bridge after deployment? Is specialized expertise needed?',
         'Bridge is managed through a web-based console requiring no specialized cryptography '
         'knowledge. Day-to-day operations are handled by your existing security or IT operations '
         'team. Bridge automates key management, certificate rotation, and policy updates. '
         'QBITEL provides 24/7 Clinical Security Operations Center (CSOC) support for all '
         'enterprise customers, with clinical-context-aware analysts who understand healthcare '
         'workflows and device behavior.'),
        ('42', 'What is Bridge\'s SLA for uptime and availability?',
         'Bridge carries a 99.999% uptime SLA (less than 5 minutes downtime per year) for '
         'enterprise deployments. This is backed by: redundant hardware deployment, automated '
         'failover in less than 100ms, HSM-synchronized key backup, and 24/7 CSOC monitoring. '
         'Planned maintenance windows are scheduled during low-census periods and completed '
         'with zero clinical traffic disruption. SLA credits apply for any breach of the '
         'uptime guarantee.'),
        ('43', 'How does Bridge handle software updates without disrupting clinical operations?',
         'Bridge uses blue-green deployment for all software updates. The update is applied to '
         'the standby node, validated, then traffic is seamlessly shifted with less than 100ms '
         'interruption. The original node is then updated and returned to standby. This process '
         'is fully automated and can be scheduled for off-peak hours. Clinical staff never '
         'experience a service interruption. Software updates include new threat intelligence, '
         'protocol adapters, and cryptographic algorithm updates.'),
        ('44', 'What visibility does Bridge provide into medical device security posture?',
         'Bridge provides a real-time Clinical Security Dashboard showing: device inventory with '
         'security score per device, PHI transmission encryption coverage, anomaly detection '
         'alerts by clinical area, compliance control status across HIPAA/HITRUST, active '
         'threats and investigations, and trend analytics. Executive summary reports are '
         'generated on-demand for CISO, CIO, and Board-level reporting. The dashboard is '
         'available via web browser with role-based access control.'),
        ('45', 'How does Bridge handle key rotation and cryptographic key management?',
         'Bridge uses automated key rotation with configurable schedules (daily, weekly, monthly '
         'per policy). ML-KEM-512 session keys are ephemeral -- unique per session, providing '
         'perfect forward secrecy. Long-term identity keys are stored in HSMs and rotated '
         'annually. Key rotation is zero-downtime and transparent to devices. All key operations '
         'are logged for compliance purposes. FIPS 140-3 Level 3 HSMs are standard on all '
         'Bridge Clinical Edge Node deployments.'),
    ]

    for qnum, q, a in qa7:
        story.append(KeepTogether([QABlock(qnum, q, a), sp(5)]))

    story.append(PageBreak())

    # ── SECTION 8: Procurement (5 Qs) ─────────────────────────────────────────
    story.append(SectionHeader('8', 'Procurement & Commercial',
                               '5 Questions -- Pricing, contracting, and procurement considerations'))
    story.append(sp(6))

    qa8 = [
        ('46', 'How is QBITEL Bridge priced?',
         'Bridge is priced on a per-protected-device basis with tiered volume pricing. Healthcare '
         'pricing tiers: Starter (up to 1,000 devices), Professional (1,000-10,000 devices), '
         'Enterprise (10,000-50,000 devices), and IDN (50,000+ devices with custom pricing). '
         'All tiers include unlimited HL7/FHIR/DICOM traffic, HIPAA compliance automation, '
         '24/7 CSOC support, and software updates. Multi-year agreements include significant '
         'discounts. Contact enterprise@qbitel.com for a tailored quote.'),
        ('47', 'Is Bridge available through healthcare GPO contracts?',
         'QBITEL Bridge is available through Premier, Vizient, and HealthTrust Performance Group '
         '(HPG) contract vehicles. GPO members receive preferred pricing and streamlined '
         'contracting. For IDN-level deployments, QBITEL also supports custom enterprise '
         'agreements with multi-facility volume pricing. We work with all major healthcare '
         'legal and procurement frameworks including standard BAA terms and HIPAA Business '
         'Associate requirements.'),
        ('48', 'What does the implementation cost include?',
         'The QBITEL Bridge enterprise agreement includes: (1) Hardware Clinical Edge Nodes '
         'leased or purchased; (2) Professional Services for discovery, configuration, and '
         'deployment; (3) Integration with EHR platforms (Epic/Cerner/MEDITECH); '
         '(4) Biomedical engineering coordination and UAT support; '
         '(5) HIPAA/HITRUST certification support; (6) 24/7 CSOC monitoring for year one; '
         '(7) Quarterly compliance review meetings. Typical total implementation cost for '
         'a 500-bed facility is included in the enterprise agreement.'),
        ('49', 'Can we pilot Bridge before committing to an enterprise deployment?',
         'Yes. QBITEL offers a structured 6-week pilot program for qualifying healthcare '
         'organizations. The pilot covers one clinical area (ICU, ED, or Radiology typically) '
         'with 200-500 devices. The pilot includes full deployment, HIPAA compliance '
         'demonstration, and clinical validation. Pilot outcomes provide the evidence base '
         'for board and leadership approval of enterprise deployment. Pilot pricing is fixed '
         'and fully credited against the enterprise agreement upon conversion.'),
        ('50', 'What is the ROI case for Bridge?',
         'A typical 500-bed facility ROI: (1) Avoided breach cost: $10.9M average x probability '
         'reduction; (2) HIPAA fine avoidance: $1.3B industry total in 2024 -- average fine '
         '$2.2M; (3) Compliance labor savings: 120-160 hours/quarter x $150/hr = $72-96K/year; '
         '(4) Cyber insurance premium reduction: 20-35% reduction upon quantum-safe attestation '
         '(validated with major healthcare insurers); (5) FDA recertification avoided: $500K-$2M '
         'per device class. Full ROI model available on request.'),
    ]

    for qnum, q, a in qa8:
        story.append(KeepTogether([QABlock(qnum, q, a), sp(5)]))

    story.append(PageBreak())

    # ── SECTION 9: Hard Objections (7) ────────────────────────────────────────
    story.append(SectionHeader('9', 'Hard Objections',
                               '7 Objection Handlers -- The toughest pushback you will face'))
    story.append(sp(6))

    objections = [
        ('1', 'Our devices are FDA-cleared, we cannot modify them.',
         'Correct -- and Bridge never modifies them. Bridge operates entirely at the network layer, '
         'outside the device boundary. No firmware is changed, no software is installed, no device '
         'configuration is altered. We have reviewed this architecture with FDA cybersecurity policy '
         'staff and have written guidance confirming this does not trigger recertification. Your '
         'FDA clearances remain completely intact. Bridge is the only solution that solves the '
         'medical device security problem without requiring a single device modification.'),
        ('2', 'HIPAA does not require quantum-safe cryptography.',
         'HIPAA requires "reasonable and appropriate" encryption for PHI in transit -- a standard '
         'that evolves with the threat landscape. NIST finalized post-quantum standards in August '
         '2024. HHS has publicly acknowledged quantum threats to healthcare PHI. OCR guidance on '
         'encryption standards is updated to reflect NIST recommendations. More practically: PHI '
         'from your organization is being harvested today for future quantum decryption. The breach '
         'happens now. The HIPAA fine will come when quantum decryption makes it discoverable. '
         'Proactive protection is the only defense.'),
        ('3', 'We have cyber insurance. A breach is covered.',
         'Three points: First, cyber insurance carriers are now excluding quantum-harvest events '
         'from standard policies -- the exclusion language appeared in 2024 renewal cycles. '
         'Second, insurance does not cover the full $10.9M average healthcare breach cost -- '
         'reputational damage, patient churn, and regulatory remediation are typically uninsured. '
         'Third, premiums are rising 40-60% annually for healthcare -- demonstrating quantum-safe '
         'posture to your carrier has produced 20-35% premium reductions for QBITEL customers. '
         'Bridge pays for itself in insurance savings alone for many customers.'),
        ('4', 'Our EHR vendor (Epic/Cerner) handles security.',
         'EHR vendors provide application-level security for their platforms -- they do not and '
         'cannot protect the thousands of medical devices that feed data into those platforms. '
         'Epic, for example, explicitly documents that device security is the customer\'s '
         'responsibility. The attack surface for PHI breach is primarily the device-to-EHR '
         'communication layer and the legacy device inventory -- not the EHR application itself. '
         'Bridge complements EHR security by protecting the data before it reaches the EHR.'),
        ('5', 'This is too complex for our IT team to manage.',
         'Bridge is designed for healthcare IT teams, not cryptography researchers. The management '
         'console requires no cryptographic knowledge. Policies are configured using healthcare '
         'terminology (device class, clinical area, PHI classification) -- not encryption '
         'parameters. QBITEL\'s 24/7 CSOC handles all expert-level security operations. The '
         'most complex task your IT team performs is reviewing the daily security digest email. '
         'We have successfully deployed Bridge in 12-person IT departments serving 1,500-bed facilities.'),
        ('6', 'We are a small clinic or physician practice -- we are not a target.',
         'Small healthcare organizations are disproportionately targeted because they have valuable '
         'PHI with weak security. The OCR breach portal shows hundreds of small practice breaches '
         'annually -- including practices with fewer than 10 employees. Average HIPAA fine for '
         'small practices: $100K-$500K. Average breach remediation: $200K-$500K. For a small '
         'clinic, one breach is existential. QBITEL offers right-sized pricing for small practices '
         'through our Starter tier, making enterprise-grade protection accessible at clinic scale.'),
        ('7', 'Our systems are air-gapped. We do not have internet exposure.',
         'Air-gapping reduces but does not eliminate risk. Healthcare air-gap breaches are documented: '
         'USB-delivered malware (Stuxnet methodology), supply chain attacks on device software '
         'updates, compromised vendor maintenance laptops, and insider threats all bypass air-gaps. '
         'More importantly, air-gaps in healthcare are rarely absolute -- imaging, laboratory, and '
         'pharmacy systems typically have at least one internet-adjacent connection. Bridge protects '
         'air-gapped environments fully, including offline update delivery and local compliance reporting.'),
    ]

    for onum, obj, resp in objections:
        story.append(KeepTogether([ObjectionBlock(onum, obj, resp), sp(6)]))

    story.append(PageBreak())

    # ── SECTION 10: Competitive (8 Qs) ────────────────────────────────────────
    story.append(SectionHeader('10', 'Competitive Questions',
                               '8 Questions -- How Bridge compares to alternatives'))
    story.append(sp(6))

    qa10 = [
        ('58', 'How does Bridge compare to Claroty, Medigate, or Armis for medical device security?',
         'Claroty, Medigate (now Microsoft Defender for IoT), and Armis are visibility and monitoring '
         'platforms -- they detect threats but do not encrypt or prevent them. They see an infusion '
         'pump communicating to an unauthorized server; they alert. Bridge encrypts the infusion pump '
         'communication so the attacker cannot read the intercepted data AND alerts on the anomaly. '
         'For the quantum threat specifically, monitoring platforms offer zero protection -- only '
         'encryption prevents harvest-now-decrypt-later attacks.'),
        ('59', 'We already have Palo Alto or Cisco for network security. Why do we need Bridge?',
         'Palo Alto and Cisco provide excellent perimeter and east-west network security -- but they '
         'do not implement post-quantum cryptography, and they cannot protect medical device '
         'communications without triggering FDA recertification concerns. Bridge is a complementary '
         'layer that adds post-quantum encryption to device communications, HL7/FHIR protocol '
         'security, and HIPAA-specific compliance automation. Bridge integrates with Palo Alto '
         'XSIAM and Cisco SecureX, providing clinical-context threat data to your existing SOC.'),
        ('60', 'Can we just upgrade to TLS 1.3 everywhere instead of buying Bridge?',
         'TLS 1.3 is excellent classical cryptography, but it uses ECDH and RSA key exchange -- '
         'both broken by quantum computers. TLS 1.3 does not protect against harvest-now-decrypt-later '
         'attacks on your PHI. Additionally, TLS 1.3 cannot be deployed on most medical devices '
         'without firmware updates -- which trigger FDA recertification. Bridge implements '
         'ML-KEM-512 (post-quantum) key exchange as a replacement for ECDH, providing quantum '
         'resistance that TLS 1.3 alone cannot deliver.'),
        ('61', 'What about using a VPN for medical device security?',
         'Traditional VPNs use classical cryptography (RSA, ECDH) that is vulnerable to quantum '
         'harvest attacks. More fundamentally, medical devices cannot install VPN clients without '
         'FDA recertification. Bridge provides the equivalent of a VPN for medical devices -- '
         'encrypted tunnels for all device traffic -- without any client installation on the '
         'protected devices. Bridge also provides clinical-specific capabilities (HL7/FHIR/DICOM '
         'security, HIPAA audit automation) that no VPN product offers.'),
        ('62', 'How does Bridge compare to Microsoft Defender for IoT (formerly Medigate)?',
         'Microsoft Defender for IoT is a passive monitoring and visibility solution -- it observes '
         'medical device traffic but does not encrypt it. It identifies vulnerabilities but cannot '
         'remediate them without device modification. Bridge actively encrypts all medical device '
         'communications and provides automated remediation through VLAN quarantine. For HIPAA '
         'compliance, Bridge generates automated audit reports; Defender for IoT does not. '
         'The two tools can complement each other -- Defender for visibility, Bridge for active protection.'),
        ('63', 'A competitor claims they can also do post-quantum encryption. How do we compare?',
         'Verify three things: (1) Which NIST standard? FIPS 203 (ML-KEM), 204 (ML-DSA), or 205 '
         '(SLH-DSA)? Generic "post-quantum" claims may refer to experimental algorithms not yet '
         'standardized. QBITEL implements all three NIST finalized standards. (2) Can it protect '
         'medical devices without FDA recertification? If their solution requires device software '
         'changes, the answer is no. (3) Does it include HL7/FHIR/DICOM-native protocol security '
         'and HIPAA audit automation? Bridge is the only platform purpose-built for clinical environments.'),
        ('64', 'Can we build this ourselves using open-source PQC libraries?',
         'Technically possible but practically inadvisable for healthcare. Building a production-grade '
         'PQC implementation requires: cryptographic engineering expertise, FIPS 140-3 validated '
         'HSM integration, healthcare protocol expertise (HL7/FHIR/DICOM parsing), HIPAA audit '
         'framework development, clinical network deployment expertise, and 24/7 operational '
         'support. Estimated build cost: $5-15M over 3-4 years. QBITEL delivers this in 20 weeks '
         'at a fraction of the build cost, with proven production deployments.'),
        ('65', 'What happens if NIST changes the post-quantum standards?',
         'NIST finalized FIPS 203, 204, and 205 in August 2024 after 8 years of evaluation -- '
         'these are not experimental algorithms. NIST has committed to a multi-year stability '
         'period for these standards. If NIST releases revised algorithms (FIPS 205 SLH-DSA '
         'already provides a hash-based backup), Bridge delivers cryptographic agility -- '
         'the ability to swap algorithms without device changes or recertification. Cryptographic '
         'agility is a built-in Bridge capability and a key differentiator from point solutions.'),
    ]

    for qnum, q, a in qa10:
        story.append(KeepTogether([QABlock(qnum, q, a), sp(5)]))

    story.append(PageBreak())

    # Back cover summary
    from reportlab.platypus.flowables import Flowable as F
    story.append(Spacer(1, 0.3 * inch))

    S_body = ParagraphStyle('b', fontName='Helvetica', fontSize=10, leading=15,
                            textColor=DARK_TEXT, spaceAfter=8, alignment=TA_CENTER)
    S_head = ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=14, leading=18,
                            textColor=NAVY, spaceAfter=8, alignment=TA_CENTER)
    story.append(Paragraph('QBITEL BRIDGE HEALTHCARE', S_head))
    story.append(Paragraph('65 Q&As. 10 Sections. Every clinical objection answered.', S_body))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph('enterprise@qbitel.com  |  https://bridge.qbitel.com', S_body))

    doc.build(story)
    print(f'QA PDF written: {out_path}')


if __name__ == '__main__':
    build_qa_pdf()
