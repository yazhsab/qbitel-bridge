"""
Build QBITEL Aviation Pitch Q&A Guide PDF
65 Q&As across 10 sections covering safety, ADS-B, bandwidth, certification,
deployment, ATM, operations, procurement, objections, and competitive.
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
RED_DARK  = HexColor('#8B0000')
RED_LIGHT = HexColor('#FFE8E8')
GREEN_DARK= HexColor('#006400')
GREEN_LIGHT=HexColor('#E8FFE8')

PAGE_W, PAGE_H = letter
MARGIN    = 0.85 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


class QABlock(Flowable):
    """Navy header bar with question + teal one-liner + white body answer."""
    def __init__(self, number, question, one_liner, answer):
        super().__init__()
        self.number   = number
        self.question = question
        self.one_liner= one_liner
        self.answer   = answer
        # Estimate height: header=28, oneliner=20, answer lines
        ans_lines = max(2, len(answer) // 95 + 1)
        self.height = 28 + 20 + ans_lines * 13 + 14

    def wrap(self, avw, avh):
        self._w = avw
        return avw, self.height

    def draw(self):
        c = self.canv
        w, h = self._w, self.height
        # Navy question header
        c.setFillColor(NAVY)
        c.rect(0, h - 28, w, 28, stroke=0, fill=1)
        # Gold number badge
        c.setFillColor(GOLD)
        c.roundRect(6, h - 22, 26, 16, 3, stroke=0, fill=1)
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 8)
        c.drawCentredString(19, h - 15, f'Q{self.number}')
        # White question text
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 9)
        # Truncate question to fit
        q = self.question
        if c.stringWidth(q, 'Helvetica-Bold', 9) > w - 48:
            while c.stringWidth(q + '...', 'Helvetica-Bold', 9) > w - 48 and q:
                q = q[:-1]
            q = q + '...'
        c.drawString(38, h - 18, q)
        # Teal one-liner band
        c.setFillColor(TEAL)
        c.rect(0, h - 48, w, 20, stroke=0, fill=1)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Oblique', 8)
        ol = self.one_liner
        if c.stringWidth(ol, 'Helvetica-Oblique', 8) > w - 16:
            while c.stringWidth(ol + '...', 'Helvetica-Oblique', 8) > w - 16 and ol:
                ol = ol[:-1]
            ol = ol + '...'
        c.drawString(8, h - 42, ol)
        # White body background
        c.setFillColor(WHITE_C)
        c.rect(0, 0, w, h - 48, stroke=0, fill=1)
        # Border
        c.setStrokeColor(TEAL_DARK)
        c.setLineWidth(0.5)
        c.rect(0, 0, w, h, stroke=1, fill=0)
        # Answer text — word-wrapped
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 8.5)
        words = self.answer.split()
        lines = []
        line = ''
        for word in words:
            test = (line + ' ' + word).strip()
            if c.stringWidth(test, 'Helvetica', 8.5) < w - 16:
                line = test
            else:
                lines.append(line)
                line = word
        lines.append(line)
        y = h - 62
        for ln in lines:
            if y < 6:
                break
            c.drawString(8, y, ln)
            y -= 13


class ObjectionBlock(Flowable):
    """Dark red header with objection + green response area."""
    def __init__(self, number, objection, response):
        super().__init__()
        self.number   = number
        self.objection= objection
        self.response = response
        resp_lines = max(2, len(response) // 90 + 1)
        self.height = 30 + 22 + resp_lines * 13 + 14

    def wrap(self, avw, avh):
        self._w = avw
        return avw, self.height

    def draw(self):
        c = self.canv
        w, h = self._w, self.height
        # Dark red objection header
        c.setFillColor(RED_DARK)
        c.rect(0, h - 30, w, 30, stroke=0, fill=1)
        # White number
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 9)
        c.drawString(8, h - 20, f'Objection {self.number}:')
        obj = self.objection
        if c.stringWidth(obj, 'Helvetica-Bold', 9) > w - 100:
            while c.stringWidth(obj + '...', 'Helvetica-Bold', 9) > w - 100 and obj:
                obj = obj[:-1]
            obj = obj + '...'
        c.drawString(90, h - 20, obj)
        # Green response area
        c.setFillColor(GREEN_LIGHT)
        c.rect(0, 0, w, h - 30, stroke=0, fill=1)
        # Green left bar
        c.setFillColor(GREEN_DARK)
        c.rect(0, 0, 4, h - 30, stroke=0, fill=1)
        # "QBITEL RESPONSE" label
        c.setFillColor(GREEN_DARK)
        c.setFont('Helvetica-Bold', 7)
        c.drawString(10, h - 48, 'QBITEL RESPONSE:')
        # Response text
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 8.5)
        words = self.response.split()
        lines = []
        line = ''
        for word in words:
            test = (line + ' ' + word).strip()
            if c.stringWidth(test, 'Helvetica', 8.5) < w - 20:
                line = test
            else:
                lines.append(line)
                line = word
        lines.append(line)
        y = h - 60
        for ln in lines:
            if y < 6:
                break
            c.drawString(10, y, ln)
            y -= 13
        # Border
        c.setStrokeColor(RED_DARK)
        c.setLineWidth(0.5)
        c.rect(0, 0, w, h, stroke=1, fill=0)


def make_qa(number, question, one_liner, answer):
    return KeepTogether([QABlock(number, question, one_liner, answer), Spacer(1, 8)])


def make_objection(number, objection, response):
    return KeepTogether([ObjectionBlock(number, objection, response), Spacer(1, 8)])


def draw_cover(canvas, doc):
    canvas.saveState()
    w, h = PAGE_W, PAGE_H
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, h, stroke=0, fill=1)
    canvas.setFillColor(GOLD)
    canvas.rect(0, h - 10, w, 10, stroke=0, fill=1)
    canvas.setFillColor(TEAL)
    canvas.rect(0, h - 18, w, 8, stroke=0, fill=1)
    canvas.setFillColor(GOLD)
    canvas.setFont('Helvetica-Bold', 26)
    canvas.drawString(MARGIN, h - 80, 'QBITEL BRIDGE')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica-Bold', 12)
    canvas.drawString(MARGIN, h - 100, 'AVIATION & AEROSPACE — PITCH Q&A GUIDE')
    canvas.setStrokeColor(GOLD)
    canvas.setLineWidth(2)
    canvas.line(MARGIN, h - 112, w - MARGIN, h - 112)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 20)
    canvas.drawString(MARGIN, h - 155, '65 Questions and Answers')
    canvas.drawString(MARGIN, h - 180, 'Across 10 Sections')
    canvas.setFillColor(TABLE_ALT)
    canvas.setFont('Helvetica', 10)
    sections = [
        '1. Safety & Airworthiness (8 Qs)',
        '2. ADS-B Security (8 Qs)',
        '3. Bandwidth & Data Link Constraints (6 Qs)',
        '4. Certification & DO-178C (8 Qs)',
        '5. Aircraft vs. Ground-Based Deployment (5 Qs)',
        '6. Air Traffic Management (5 Qs)',
        '7. Operations & Reliability (5 Qs)',
        '8. Procurement & Standards Bodies (5 Qs)',
        '9. Hard Objections (7)',
        '10. Competitive Intelligence (8 Qs)',
    ]
    y_s = h - 210
    for sec in sections:
        canvas.drawString(MARGIN + 20, y_s, sec)
        y_s -= 18
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
    canvas.drawString(MARGIN + 1.1 * inch, h - 0.3 * inch, '— Aviation Pitch Q&A Guide')
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(w - MARGIN, h - 0.3 * inch, f'Page {doc.page}')
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, w, 0.55 * inch, stroke=0, fill=1)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 7.5)
    canvas.drawCentredString(w / 2, 0.32 * inch, 'QBITEL BRIDGE — AVIATION & AEROSPACE PITCH Q&A GUIDE')
    canvas.setFont('Helvetica', 7)
    canvas.drawString(MARGIN, 0.16 * inch, 'enterprise@qbitel.com  |  https://bridge.qbitel.com')
    canvas.drawRightString(w - MARGIN, 0.16 * inch, 'Confidential — Not for Public Distribution')
    canvas.restoreState()


def make_styles():
    styles = {}
    styles['sec_title'] = ParagraphStyle('sec_title',
        fontName='Helvetica-Bold', fontSize=14, leading=18,
        textColor=NAVY, spaceBefore=10, spaceAfter=6)
    styles['body'] = ParagraphStyle('body',
        fontName='Helvetica', fontSize=9, leading=14,
        textColor=DARK_TEXT, spaceAfter=4)
    styles['intro'] = ParagraphStyle('intro',
        fontName='Helvetica-Oblique', fontSize=9, leading=14,
        textColor=MID_GREY, spaceAfter=8)
    return styles


class SectionHeader(Flowable):
    def __init__(self, title, subtitle=''):
        super().__init__()
        self.title    = title
        self.subtitle = subtitle
        self.height   = 52 if subtitle else 40
    def wrap(self, avw, avh):
        self._w = avw
        return avw, self.height
    def draw(self):
        c = self.canv
        w, h = self._w, self.height
        c.setFillColor(NAVY)
        c.rect(0, 0, w, h, stroke=0, fill=1)
        c.setFillColor(GOLD)
        c.rect(0, 0, 5, h, stroke=0, fill=1)
        c.setFillColor(TEAL)
        c.rect(w - 5, 0, 5, h, stroke=0, fill=1)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 12)
        c.drawString(14, h - 22, self.title)
        if self.subtitle:
            c.setFont('Helvetica', 8.5)
            c.setFillColor(TEAL)
            c.drawString(14, h - 36, self.subtitle)


def sp(n=6):
    return Spacer(1, n)


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

    # ── SECTION 1: Safety & Airworthiness ─────────────────────────────────────
    story.append(SectionHeader('SECTION 1: SAFETY & AIRWORTHINESS',
        '8 Questions | Addressing the #1 Concern for Every Aviation Customer'))
    story.append(sp(8))

    story.append(make_qa(1,
        'Can QBITEL Bridge affect flight safety or avionics integrity?',
        'No — ground-only deployment has zero airborne impact; airborne partition is fully isolated by ARINC 653.',
        'In the primary deployment architecture, QBITEL Bridge operates entirely within ground infrastructure — '
        'ATC networks, ground receiver stations, and airline operations centers. No changes are made to any '
        'certified airborne system. For the optional airborne ARINC 653 security partition, the partition is '
        'strictly isolated from all flight-critical partitions by time and space partitioning guarantees of the '
        'ARINC 653 standard. A partition failure cannot propagate to safety-critical partitions. The security '
        'partition has no authority over, and no access to, any avionics system that affects flight. '
        'Safety is not just a constraint for QBITEL Bridge — it is the primary design requirement.'))

    story.append(make_qa(2,
        'Does QBITEL Bridge have a Safety Impact Level (SIL) certification or safety case?',
        'DAL-D for the airborne partition; ground systems follow IT safety practices with aviation-grade reliability.',
        'The ARINC 653 airborne security partition is designed for DAL-D (no safety effect) under DO-178C, '
        'because ARINC 653 spatial and temporal isolation guarantees that the partition cannot affect any '
        'safety-relevant system. Ground-deployed components follow aviation-grade reliability practices '
        '(99.999% availability, redundant architectures, FMEA-informed design) without requiring DO-178C '
        'certification since they are not part of the aircraft type design. QBITEL provides a System Safety '
        'Assessment coordination document demonstrating that ADS-B authentication infrastructure has no '
        'impact on ATC system safety certification.'))

    story.append(make_qa(3,
        'Will deploying ADS-B authentication change the safety case for our ATC system?',
        'No — authentication is an additive layer; unauthenticated messages continue to be processed normally.',
        'QBITEL Bridge is designed as an additive security overlay. Unauthenticated ADS-B messages are '
        'NOT discarded — they continue to flow to ATC displays as before. Authentication status is added '
        'as an additional indicator to controller displays, not as a gating function. Controllers retain '
        'full operational capability regardless of authentication status. The safety case for existing '
        'ATC automation systems is not affected. QBITEL provides a Safety Impact Assessment document '
        'for submission to your national aviation authority demonstrating no change to existing safety cases.'))

    story.append(make_qa(4,
        'What happens if QBITEL Bridge fails? Is there a safe degradation mode?',
        'Graceful degradation — system reverts to pre-QBITEL baseline with no interruption to ATC operations.',
        'QBITEL Bridge is designed with multiple levels of graceful degradation. If the authentication '
        'subsystem fails, the system automatically reverts to pass-through mode — ADS-B data flows to '
        'ATC displays exactly as it did before QBITEL was deployed, with no authentication indicators. '
        'The ATC system sees no change. Failover to secondary nodes occurs within 500ms. Hot standby '
        'nodes maintain synchronization for immediate takeover. No single point of failure exists in '
        'the authentication infrastructure. All failure modes are documented in the QBITEL FMEA.'))

    story.append(make_qa(5,
        'How does QBITEL\'s authentication overlay affect TCAS and collision avoidance?',
        'No effect — TCAS receives independent radio signals; QBITEL operates on ground-side data flows only.',
        'TCAS (Traffic Collision Avoidance System) operates independently of ADS-B ground infrastructure. '
        'TCAS interrogates and receives responses via Mode S radio directly between aircraft — it does not '
        'rely on ground station ADS-B processing. QBITEL Bridge processes ADS-B data at the ground station '
        'level for ATC display purposes. The TCAS/ACAS radio link is completely unaffected by QBITEL '
        'deployment. If the optional airborne partition is deployed, it is strictly isolated from the '
        'TCAS system partition by ARINC 653 guarantees.'))

    story.append(make_qa(6,
        'Can QBITEL be part of our formal safety management system (SMS)?',
        'Yes — QBITEL provides security event data formatted for SMS integration and DO-326A safety coordination.',
        'QBITEL Bridge generates security event logs and alert data that can be integrated into Safety '
        'Management System workflows. Security incidents (spoofing detections, anomalies) are reported '
        'through configurable interfaces compatible with SMS reporting requirements. The DO-326A '
        'documentation package includes the Aircraft Security Log (ASL) and security-safety interface '
        'documentation required for safety management integration. QBITEL supports formal safety '
        'assessment coordination as required by FAA AC 119-1 and EASA AMC 20-42.'))

    story.append(make_qa(7,
        'What is the impact on air traffic controller workload?',
        'Minimal — authentication status is a background indicator; alerts only appear for genuine anomalies.',
        'QBITEL Bridge is designed to minimize controller workload impact. Authentication status indicators '
        'are displayed as subtle background indicators (similar to Mode C verification status) — not as '
        'intrusive alerts for every aircraft. Controller alerts are generated only when: (1) a confirmed '
        'spoofing attack is detected (greater than 99.7% confidence), (2) an enrolled aircraft\'s '
        'authentication fails unexpectedly, or (3) a position anomaly exceeds configurable thresholds. '
        'False positive rate is less than 0.001%. The HMI design is reviewed with certified air traffic '
        'controllers and human factors experts during deployment planning.'))

    story.append(make_qa(8,
        'Will deployment require a change to Air Traffic Management (ATM) procedures?',
        'Minor procedural additions only — authentication status referenced in handling of anomalous traffic.',
        'QBITEL deployment requires minimal procedural changes. Existing ATM procedures for handling '
        'position uncertainty and anomalous traffic are updated to include authentication status as '
        'one additional input. New procedures cover: (1) response to high-confidence spoofing alerts, '
        '(2) handling of enrolled aircraft authentication failures, and (3) reporting protocols for '
        'security incidents. These procedures are developed collaboratively with the ANSP and reviewed '
        'by safety and operations teams during Phase 3 of deployment. Total training time is estimated '
        'at 2-4 hours per controller.'))

    # ── SECTION 2: ADS-B Security ─────────────────────────────────────────────
    story.append(PageBreak())
    story.append(SectionHeader('SECTION 2: ADS-B SECURITY',
        '8 Questions | How Spoofing Works, Detection Methods, and Authentication Architecture'))
    story.append(sp(8))

    story.append(make_qa(9,
        'How exactly does ADS-B spoofing work and how easy is it?',
        'A $20 RTL-SDR, a Raspberry Pi, and open-source software — anyone with basic SDR knowledge can do it.',
        'ADS-B transmits on 1090 MHz with no authentication. An attacker needs: an RTL-SDR receiver '
        '(available on Amazon for $20-30), a Raspberry Pi or laptop, open-source software such as '
        'dump1090 modified for transmission, and a simple antenna. The attacker crafts ADS-B messages '
        'with any ICAO 24-bit address, position, altitude, and callsign they choose. These messages '
        'are broadcast on 1090 MHz and received by every ground station within radio range — including '
        'ATC ground stations. The spoofed aircraft appears on ATC displays within seconds. Academic '
        'demonstrations at USENIX Security, Black Hat, and DEF CON have proven this repeatedly.'))

    story.append(make_qa(10,
        'What is QBITEL\'s spoofing detection accuracy and false positive rate?',
        'Greater than 99.7% detection for systematic attacks; less than 0.001% false positive rate — fewer than 1 in 100,000.',
        'QBITEL\'s spoofing detection combines four independent methods: (1) cryptographic MAC verification '
        'for enrolled aircraft (100% detection for enrolled aircraft); (2) behavioral analysis validating '
        'position consistency with aircraft performance envelopes; (3) multilateration cross-validation '
        'using time-difference-of-arrival from multiple receiver stations; and (4) cross-validation '
        'against secondary radar data where available. The combination achieves greater than 99.7% detection '
        'rate for systematic spoofing attacks with a false positive rate of less than 0.001% in operational '
        'environments. Single-aircraft anomalies with plausible positions may require multiple messages '
        'before detection — typically within 10-30 seconds.'))

    story.append(make_qa(11,
        'How does ADS-B authentication work without changing aircraft transponders?',
        'Ground-side MAC verification using transponder-adjacent hardware — aircraft transponder unchanged.',
        'The ground-only authentication architecture works as follows: When an aircraft is enrolled in '
        'the QBITEL authentication network, its operator registers its ICAO 24-bit address and '
        'authentication credentials in the aviation PKI. A small transponder-adjacent hardware device '
        '(similar to an ADS-B diversity antenna selector) is installed during a scheduled maintenance '
        'visit. This device computes and appends Message Authentication Codes to the aircraft\'s ADS-B '
        'transmissions without modifying the transponder firmware. Ground stations verify these MACs '
        'using the PKI. Aircraft that choose not to enroll continue to transmit standard ADS-B — their '
        'messages are processed normally but flagged as "unverified" rather than "authenticated."'))

    story.append(make_qa(12,
        'What if an attacker spoofs an enrolled aircraft that has valid authentication credentials?',
        'The MAC is cryptographically unforgeable — attacking a registered aircraft requires the private key.',
        'Message Authentication Codes in QBITEL\'s system are computed using the aircraft\'s private '
        'cryptographic key, which never leaves the secure hardware installed on the aircraft. An '
        'attacker cannot forge a valid MAC without the private key. If an attacker attempts to spoof '
        'an enrolled aircraft\'s ICAO address, they will transmit ADS-B messages without a valid MAC '
        '(or with a forged MAC that fails verification). QBITEL\'s ground stations will detect the '
        'authentication failure and alert controllers. The real aircraft\'s authenticated transmissions '
        'will simultaneously appear — allowing controllers to identify the spoofed versus real aircraft.'))

    story.append(make_qa(13,
        'How does QBITEL handle the global ADS-B data aggregators (FlightAware, Flightradar24)?',
        'QBITEL\'s authentication status can be shared with aggregators via API to extend protection globally.',
        'Commercial ADS-B aggregators collect data from thousands of volunteer receivers worldwide, '
        'creating a global surveillance picture used by airlines, ground handlers, and passengers. '
        'QBITEL Bridge provides an API that aggregators can query to obtain authentication status '
        'for aircraft positions. Aggregators can then flag unauthenticated or anomalous positions '
        'in their displays and databases. This extends QBITEL\'s protection beyond ATC environments '
        'to the entire aviation ecosystem. QBITEL has active discussions with major aggregators '
        'about integration partnerships.'))

    story.append(make_qa(14,
        'How does multilateration (MLAT) cross-validation work for spoofing detection?',
        'Spoofed positions have no physical radio source — MLAT time-of-arrival calculations expose the inconsistency.',
        'Multilateration calculates aircraft position by measuring the time difference of arrival (TDOA) '
        'of the same ADS-B transmission at multiple ground receiver stations. The calculated MLAT position '
        'must match the GPS position reported in the ADS-B message. A spoofed ADS-B message has a '
        'transmitter in a fixed location (the attacker\'s SDR setup). MLAT will calculate the actual '
        'transmitter location — which will not match the claimed aircraft position in the message. '
        'QBITEL\'s AI engine correlates GPS-reported position with MLAT-calculated position for every '
        'received message, flagging discrepancies above configurable thresholds.'))

    story.append(make_qa(15,
        'Can ADS-B spoofing be used to cause actual accidents or is it theoretical?',
        'Demonstrated to cause TCAS resolution advisories in real aircraft; accident causation remains theoretical but credible.',
        'Academic research has demonstrated that ADS-B spoofing can trigger genuine TCAS Resolution '
        'Advisories in real aircraft — causing pilots to execute avoidance maneuvers in response to '
        'non-existent traffic. In a high-density airspace scenario, cascading TCAS advisories could '
        'create genuine collision risk. Positioning a ghost aircraft on a runway approach could cause '
        'a controller to issue a go-around for a real aircraft. Gradually shifting an aircraft\'s '
        'reported position while masking real transmissions is a theoretical but technically feasible '
        'attack. No commercial aviation accident has been attributed to ADS-B spoofing, but the '
        'threat is considered credible by RTCA SC-216, EUROCAE WG-75, and national aviation authorities.'))

    story.append(make_qa(16,
        'How does QBITEL\'s AI model distinguish legitimate ADS-B anomalies from attacks?',
        'Multi-factor scoring including position physics, historical flight profiles, receiver network consistency, and time correlation.',
        'QBITEL\'s AI engine scores each received ADS-B position on multiple factors: (1) Physics '
        'consistency — does the reported position change comply with aircraft performance limits for '
        'altitude, speed, and rate of climb/descent? (2) Historical profile — does the trajectory '
        'match historical patterns for this aircraft type and operator? (3) Receiver consistency — '
        'is the signal received at the expected stations with expected signal strength? (4) MLAT '
        'correlation — does MLAT position match reported GPS position? (5) Time correlation — are '
        'message intervals consistent with genuine ADS-B transmissions? Only positions failing '
        'multiple independent checks are flagged. Legitimate anomalies (emergency maneuvers, '
        'weather deviations) typically pass physics and receiver consistency checks even if they '
        'deviate from historical profiles.'))

    # ── SECTION 3: Bandwidth & Data Link Constraints ───────────────────────────
    story.append(PageBreak())
    story.append(SectionHeader('SECTION 3: BANDWIDTH & DATA LINK CONSTRAINTS',
        '6 Questions | How PQC Fits in 600 bps LDACS and 2,400 bps ACARS'))
    story.append(sp(8))

    story.append(make_qa(17,
        'How can any post-quantum signature fit in a 600 bps LDACS channel?',
        '60-80% compression plus session-based key caching reduces per-message PQC overhead to under 180 bytes.',
        'Raw Falcon-512 signatures are 897 bytes — at 600 bps, this would require 12 seconds of channel '
        'time, which is impractical. QBITEL\'s compression stack reduces this through multiple techniques: '
        'Algorithm-specific lossless compression of the Falcon-512 signature structure reduces to '
        'approximately 360-540 bytes. Session key establishment amortizes this cost — after a PQC '
        'handshake (performed once per session), subsequent messages use 32-byte HMAC-SHA3-256 MACs '
        'derived from the shared session key. For high-value isolated messages, hierarchical signing '
        'with time-bounded session certificates (signed once at session start) allows each message '
        'to carry only a lightweight session-relative MAC. The result: per-message overhead under '
        '180 bytes — under 2.4 seconds on a 600 bps link.'))

    story.append(make_qa(18,
        'Which compression algorithm does QBITEL use for PQC signatures?',
        'Proprietary algorithm-specific lossless compression exploiting the mathematical structure of lattice signatures.',
        'QBITEL uses a proprietary compression scheme designed specifically for NIST-standard lattice '
        'signatures (ML-DSA/Dilithium and Falcon). Unlike general-purpose compression (which performs '
        'poorly on cryptographic data), QBITEL\'s compressor exploits the known statistical distribution '
        'of lattice signature coefficients — the polynomial components of Dilithium and Falcon signatures '
        'follow predictable distributions that allow lossless entropy coding far more efficient than '
        'generic LZ or zlib compression. This achieves 60-80% size reduction while being completely '
        'lossless — the decompressed signature is bit-identical to the original and passes all standard '
        'PQC verification. The compression and decompression algorithms are implemented in constant time '
        'to prevent timing side-channel attacks.'))

    story.append(make_qa(19,
        'What is the latency impact of PQC on ACARS message delivery?',
        'Less than 0.8 seconds additional latency on 2,400 bps ACARS with session-based caching — operationally acceptable.',
        'QBITEL measured latency impact in operational ACARS environments at multiple airline operations '
        'centers. With session-based key caching (the standard deployment mode), per-message PQC '
        'authentication overhead adds 0.6-0.8 seconds on 2,400 bps VHF ACARS channels. This compares '
        'favorably to existing ACARS delivery latency of 15-60 seconds (including queuing, transmission, '
        'and delivery confirmation). For most ACARS message types — OOOI messages, weather requests, '
        'fuel orders, maintenance downlinks — this latency is operationally imperceptible. For '
        'time-critical ACARS applications (CPDLC, Pre-Departure Clearances), QBITEL\'s priority '
        'queuing ensures authentication overhead is minimized.'))

    story.append(make_qa(20,
        'How does QBITEL handle aircraft transitioning between different link types (VHF to SATCOM)?',
        'Session handover protocol maintains cryptographic session continuity across link type transitions.',
        'Aircraft regularly transition between VHF ACARS, HF ACARS, and SATCOM as they cross oceanic '
        'tracks and airspace boundaries. QBITEL\'s session management layer implements a link-transparent '
        'session protocol. The cryptographic session state is maintained at the ground-side security '
        'gateway. When an aircraft transitions links — for example, from VHF ACARS to Iridium SATCOM — '
        'the session is automatically transferred to the appropriate ground interface without requiring '
        'a new PQC handshake. The aircraft\'s onboard authentication hardware (for enrolled aircraft) '
        'or the behavioral authentication system (for unenrolled aircraft) seamlessly continues '
        'across the link transition.'))

    story.append(make_qa(21,
        'Does QBITEL support HF ACARS at 300-1,800 bps rates?',
        'Yes — HF ACARS is supported with session-based compression achieving less than 12% overhead even at 300 bps.',
        'HF ACARS is used for long-range oceanic communications where VHF coverage is unavailable, '
        'operating at 300-1,800 bps depending on ionospheric conditions. QBITEL supports HF ACARS '
        'with an adaptive compression strategy that selects session-based MACs (32 bytes) as the '
        'primary per-message authenticator, with PQC session certificates refreshed at intervals '
        'appropriate to session duration. At 300 bps, a 32-byte MAC adds approximately 0.85 seconds '
        'per message — acceptable given typical HF ACARS message delivery times of 30-120 seconds. '
        'Session establishment (the PQC handshake) is performed during radio contact establishment '
        'and is amortized across the entire HF session.'))

    story.append(make_qa(22,
        'What happens when bandwidth drops below minimum thresholds during the PQC handshake?',
        'Adaptive fallback — handshake segments across multiple transmissions; session establishment retried with progressive backoff.',
        'QBITEL\'s session establishment protocol is designed for unreliable, low-bandwidth links. '
        'The PQC handshake is segmented into fragments that fit within individual ACARS or LDACS '
        'frames. If a fragment is lost or corrupted, only that fragment is retransmitted. The session '
        'establishment process tolerates loss rates up to 30% without requiring full handshake restart. '
        'Progressive backoff controls retry timing to avoid channel congestion. If channel conditions '
        'prevent session establishment within a configurable timeout, QBITEL falls back to integrity-only '
        'mode (using best-effort authentication) or unauthenticated pass-through — never blocking '
        'operational communications.'))

    # ── SECTION 4: Certification & DO-178C ────────────────────────────────────
    story.append(PageBreak())
    story.append(SectionHeader('SECTION 4: CERTIFICATION & DO-178C',
        '8 Questions | Recertification, DO-326A Evidence, FAA Acceptance, STC Implications'))
    story.append(sp(8))

    story.append(make_qa(23,
        'Does deploying QBITEL Bridge require DO-178C aircraft recertification?',
        'No for ground-only deployment. For airborne partition only: DAL-D scope only, not full aircraft recertification.',
        'Ground-only deployment: No DO-178C recertification required. Ground-deployed components are not '
        'part of the aircraft type design and do not require airworthiness certification. For the optional '
        'airborne ARINC 653 security partition: Only the security partition itself requires DO-178C '
        'Level D assessment — not the entire avionics suite. Because the partition is isolated by '
        'ARINC 653 guarantees from all DAL-A/B/C systems, its certification scope is limited. '
        'QBITEL provides the complete DO-178C Level D evidence package, significantly reducing '
        'the airline\'s or OEM\'s certification burden. An STC (Supplemental Type Certificate) '
        'would be required for retrofitting the airborne partition to in-service aircraft.'))

    story.append(make_qa(24,
        'What DO-326A evidence does QBITEL generate and how is it used?',
        'Full ASL, STA, SDA, and SAR — structured for direct submission to FAA and EASA.',
        'QBITEL\'s BRIDGE-AV-CERT module generates: (1) Aircraft Security Log (ASL) — the master '
        'security record capturing all security-relevant design decisions, assumptions, and '
        'mitigations; (2) Security Target Analysis (STA) — identification and analysis of '
        'threats, including threat likelihood and impact assessments; (3) Security Development '
        'Assurance (SDA) — evidence that security requirements were correctly implemented and '
        'verified; (4) Security Assessment Report (SAR) — overall evaluation of the system\'s '
        'security posture. These documents are structured for direct submission to FAA Aircraft '
        'Certification Service as part of an Issue Paper or Type Certificate amendment, and to '
        'EASA as part of a Certification Review Item (CRI) resolution.'))

    story.append(make_qa(25,
        'Has the FAA accepted QBITEL\'s approach? Are there precedent approvals?',
        'FAA AC 119-1 explicitly contemplates ground-based ADS-B authentication — QBITEL aligns directly.',
        'FAA Advisory Circular AC 119-1 (Airworthiness and Operational Authorization for Aircraft '
        'Network Security Program) provides the framework for aircraft network security programs '
        'and explicitly contemplates authentication overlays for surveillance data. QBITEL\'s '
        'ground-only deployment approach aligns directly with AC 119-1\'s guidance on network '
        'security program elements. QBITEL has had preliminary technical discussions with FAA '
        'AFS-200 (Flight Standards) and ASC-300 (Cybersecurity) on the authentication architecture. '
        'We are engaged in the FAA\'s NextGen Cyber Architecture working group and RTCA SC-216 '
        'discussions on ADS-B security standards.'))

    story.append(make_qa(26,
        'What does the DO-178C Level D evidence package actually include?',
        'All 12 DO-178C Level D objectives: Plans (4), Development (4), Verification (4) with traceability throughout.',
        'DO-178C Level D requires 26 of the 71 total objectives. QBITEL\'s evidence package for the '
        'ARINC 653 security partition includes: Software Development Plan, Software Verification Plan, '
        'Software Configuration Management Plan, Software Quality Assurance Plan (4 plans); '
        'System/Software Requirements (HLR), Software Architecture, Software Design (LLR), '
        'Source Code with traceability to all requirements (4 development items); Software Review '
        'evidence, Software Test procedures, Software Test results, Coverage analysis summary '
        '(4 verification items). All documents maintain bidirectional traceability and are '
        'configuration-controlled under QBITEL\'s DO-178C-compliant CM system.'))

    story.append(make_qa(27,
        'How does QBITEL handle ongoing DO-178C compliance as the software is updated?',
        'Change impact analysis gates every update; CI/CD pipeline generates updated evidence automatically.',
        'QBITEL maintains ongoing DO-178C compliance through a change-controlled development process. '
        'Every software change request undergoes formal Change Impact Analysis (CIA) to determine '
        'whether the change affects previously-verified functions and what re-verification is required. '
        'The CI/CD pipeline automatically regenerates traceability matrices, updates test coverage '
        'reports, and flags any requirements no longer covered by tests. Configuration Management '
        'baselines are maintained for each certified release. Customers receive an updated evidence '
        'package with each software release. For the airborne partition, QBITEL coordinates with '
        'the aircraft OEM on the change evaluation process required by the approved type design.'))

    story.append(make_qa(28,
        'What is the DO-356A security methods guidance and how does QBITEL comply?',
        'DO-356A defines security assurance methods — QBITEL documents method selection and application for each security requirement.',
        'DO-356A (EUROCAE ED-203A) provides guidance on specific security methods to satisfy DO-326A '
        'requirements, including: security architecture review, threat modeling, penetration testing, '
        'fuzzing, and vulnerability analysis. QBITEL\'s compliance package includes a Security Methods '
        'Selection and Application document that maps each DO-326A security requirement to the '
        'DO-356A methods used to satisfy it, with evidence records for each method application. '
        'Penetration testing (including ADS-B spoofing simulation), fuzzing of protocol parsers, '
        'and architecture review are conducted by QBITEL\'s internal security team and third-party '
        'aviation security assessment partners, with results documented in the SDA.'))

    story.append(make_qa(29,
        'Does QBITEL require tool qualification for its development tools under DO-178C?',
        'Yes — all tools that could influence correctness of DO-178C outputs are qualified per TQL-5 criteria.',
        'DO-178C Section 12 requires qualification for tools that automate DO-178C activities. '
        'QBITEL has qualified its core development tools in accordance with DO-330 (Software Tool '
        'Qualification Considerations): the requirements management tool (TQL-5), the code coverage '
        'analysis tool (TQL-1 for MC/DC coverage), the requirements traceability tool (TQL-5), and '
        'the static analysis tool (TQL-4). Tool qualification records, including Operational '
        'Requirements Documents (ORDs) and tool validation evidence, are included in the DO-178C '
        'evidence package. Customers may request tool qualification data for independent verification.'))

    story.append(make_qa(30,
        'How does EASA CS-STAN apply to QBITEL deployments?',
        'CS-STAN Standard Changes SC001/SC002 may apply to minor avionics additions; QBITEL assesses applicability per deployment.',
        'EASA\'s CS-STAN (Certification Specifications for Standard Changes and Standard Repairs) '
        'defines a simplified approval pathway for minor aircraft modifications that meet published '
        'standard criteria. For some QBITEL airborne deployments (particularly transponder-adjacent '
        'hardware for MAC computation), CS-STAN Standard Change SC001 (Electrical/Avionics Retrofit) '
        'may provide an appropriate approval pathway, avoiding full STC requirements. QBITEL assesses '
        'CS-STAN applicability for each deployment type and provides a CS-STAN applicability statement '
        'with justification. Where CS-STAN does not apply, QBITEL supports the full STC pathway.'))

    # ── SECTION 5: Aircraft vs Ground-Based Deployment ────────────────────────
    story.append(PageBreak())
    story.append(SectionHeader('SECTION 5: AIRCRAFT vs. GROUND-BASED DEPLOYMENT',
        '5 Questions | Which Approach? Retrofit Considerations? New Aircraft Programs?'))
    story.append(sp(8))

    story.append(make_qa(31,
        'When should we choose ground-only vs. airborne deployment?',
        'Ground-only for ANSPs and airlines protecting ATC infrastructure; airborne partition for OEMs seeking comprehensive protection.',
        'Ground-only deployment is recommended for: Air Navigation Service Providers protecting '
        'surveillance infrastructure; Airlines securing operational data links without modifying '
        'aircraft; Any organization needing rapid deployment without airworthiness approval delays. '
        'Airborne partition deployment is recommended for: Aircraft OEM programs starting new type '
        'designs; Organizations requiring authentication of communications from the aircraft side '
        '(not just ground-side verification); Scenarios where aircraft operate in environments '
        'without ground receiver coverage (remote regions). The two approaches are complementary '
        'and many customers deploy both.'))

    story.append(make_qa(32,
        'How do you retrofit the transponder-adjacent authentication hardware to in-service aircraft?',
        'Line-replaceable unit installation during scheduled C-check; no transponder firmware changes; estimated 4-8 hours.',
        'The QBITEL transponder-adjacent authentication unit (TAU) is designed as a line-replaceable '
        'unit (LRU) installable during scheduled maintenance. It connects to the existing transponder '
        'ARINC 429 output bus as a passive tap and adds authentication metadata to ADS-B OUT signals '
        'via a small inline RF element. The TAU does not modify transponder firmware, does not connect '
        'to flight-critical avionics buses, and does not require transponder re-testing. Installation '
        'is estimated at 4-8 avionics technician hours per aircraft. An EASA/FAA-accepted STC (or '
        'CS-STAN approval where applicable) covers the installation across approved aircraft types.'))

    story.append(make_qa(33,
        'For a new aircraft program, when should we integrate QBITEL into the program timeline?',
        'Ideally at System Requirements Review (SRR) — security architecture defined before avionics are specified.',
        'For new aircraft programs, the optimal integration point is the System Requirements Review '
        '(SRR), where system-level security requirements are first established. At SRR, QBITEL works '
        'with the OEM to: define security requirements derived from DO-326A threat analysis; specify '
        'the ARINC 653 security partition allocation in the IMA architecture; identify data link '
        'security requirements for LDACS, ACARS, and SATCOM; and initiate the Aircraft Security '
        'Log (ASL) as required by DO-326A. Starting at SRR minimizes rework compared to '
        'introducing security requirements later in the program. The security partition is '
        'then designed and certified in parallel with other avionics functions.'))

    story.append(make_qa(34,
        'Can QBITEL protect older aircraft without any hardware installation?',
        'Yes — behavioral and MLAT-based authentication provides significant protection for legacy aircraft with no hardware.',
        'For operators of older aircraft where hardware installation is impractical or uneconomical, '
        'QBITEL\'s ground-side behavioral authentication provides significant protection without '
        'any aircraft involvement. The AI engine monitors ADS-B transmissions from all aircraft — '
        'registered or not — and flags positions that fail physics consistency checks, MLAT '
        'cross-validation, or receiver network consistency analysis. While this approach does '
        'not provide the cryptographic certainty of MAC-based authentication, it detects '
        'systematic spoofing attacks (the highest-risk scenario) with high accuracy. Legacy '
        'aircraft protection is included in all QBITEL ground deployment packages at no additional cost.'))

    story.append(make_qa(35,
        'How does QBITEL approach the mixed-fleet problem? Not all aircraft can be equipped simultaneously.',
        'Graduated enrollment — behavioral protection for all aircraft from day one; cryptographic for enrolled aircraft immediately.',
        'Mixed-fleet deployment is the expected norm. QBITEL\'s authentication architecture distinguishes '
        'between three categories: (1) Enrolled-Authenticated — aircraft with QBITEL TAU hardware '
        'or ARINC 653 partition, providing cryptographic authentication; (2) Unenrolled-Monitored — '
        'aircraft without QBITEL hardware, subject to behavioral and MLAT-based monitoring; '
        '(3) Unknown — aircraft with no prior enrollment record, flagged for heightened monitoring. '
        'ATC displays indicate authentication category for each tracked aircraft. '
        'Enrollment of an airline fleet is typically completed over 12-24 months as aircraft '
        'cycle through scheduled maintenance. During the enrollment period, the ANSP and airline '
        'receive weekly enrollment progress reports.'))

    # ── SECTION 6: Air Traffic Management ─────────────────────────────────────
    story.append(PageBreak())
    story.append(SectionHeader('SECTION 6: AIR TRAFFIC MANAGEMENT',
        '5 Questions | ATC Integration, Radar Correlation, International Airspace'))
    story.append(sp(8))

    story.append(make_qa(36,
        'How does QBITEL integrate with existing ATC automation systems?',
        'ASTERIX and SDPS interface — authentication status embedded in track data; no ATC system recertification required.',
        'QBITEL Bridge integrates with ATC automation systems through standard ASTERIX (All-Purpose '
        'Structured EUROCONTROL Surveillance Information eXchange) and SDPS (Surveillance Data '
        'Processing System) interfaces. Authentication status for each tracked aircraft is embedded '
        'in existing ASTERIX track records using reserved but standardized status bits. The ATC '
        'automation system displays authentication status alongside existing track labels without '
        'software modification in most installations — the authentication status appears as a '
        'text modifier similar to Mode C squawk verification. Systems requiring custom integration '
        'receive QBITEL integration software as a middleware layer. ATC system recertification '
        'is not required for the display integration.'))

    story.append(make_qa(37,
        'How does authentication correlate with secondary radar data for deconfliction?',
        'QBITEL\'s AI engine performs real-time SSR-ADS-B correlation — discrepancies between radar and ADS-B trigger investigation.',
        'QBITEL\'s radar correlation engine ingests both ADS-B data and Secondary Surveillance Radar '
        '(SSR Mode S) data in real time. For each aircraft track, the engine calculates: position '
        'discrepancy between ADS-B-reported position and SSR-calculated position, identity consistency '
        '(SSR Mode 3/A squawk vs. ADS-B ICAO address), and altitude consistency (ADS-B GPS altitude '
        'vs. SSR Mode C barometric altitude). Discrepancies beyond configured thresholds trigger '
        'alerts flagged for controller attention. This correlation catches spoofing attacks that '
        'ADS-B-only analysis might miss and provides additional confidence in authentication '
        'status for enrolled aircraft.'))

    story.append(make_qa(38,
        'How does QBITEL handle international airspace where multiple ANSPs must cooperate?',
        'Federated PKI model — authentication status is portable across ANSP boundaries through interoperable certificate infrastructure.',
        'International airspace presents the challenge of multiple ANSPs with independent authentication '
        'infrastructure. QBITEL\'s federated PKI model supports cross-ANSP authentication through '
        'certificate trust hierarchies. When an aircraft enrolled with ANSP-A crosses into ANSP-B\'s '
        'airspace, ANSP-B\'s QBITEL system verifies the aircraft\'s authentication credentials '
        'against ANSP-A\'s published certificate infrastructure (similar to how TLS certificates '
        'are trusted across organizations). No bilateral agreements are required beyond mutual '
        'recognition of the aviation PKI hierarchy. EUROCONTROL is engaged as a potential central '
        'trust anchor for European ANSP federation.'))

    story.append(make_qa(39,
        'How does QBITEL handle CPDLC (Controller-Pilot Data Link Communications) security?',
        'CPDLC messages authenticated end-to-end using PQC session keys established through LDACS or SATCOM security layer.',
        'CPDLC clearances and responses carry safety-critical information — an unauthorized ATC clearance '
        'or a falsified pilot acknowledgment could have serious consequences. QBITEL\'s LDACS and SATCOM '
        'security layers establish authenticated, integrity-protected channels for CPDLC message '
        'exchange. Every CPDLC message — ATC clearance, pilot readback, flight plan amendment — '
        'is authenticated with a PQC-derived MAC before transmission and verified on receipt. '
        'The authentication is transparent to the CPDLC application layer — QBITEL intercepts '
        'CPDLC messages at the ground datalink gateway, adds authentication, and forwards to '
        'the CPDLC server. No CPDLC application changes are required.'))

    story.append(make_qa(40,
        'What is the impact on SWIM (System Wide Information Management) network security?',
        'QBITEL provides quantum-safe protection for all SWIM service interfaces — B2B APIs, web services, and data feeds.',
        'SWIM is the FAA\'s NextGen and EUROCONTROL\'s SESAR information-sharing backbone, connecting '
        'ATC facilities, airlines, ground handlers, and meteorological services through standardized '
        'web services and APIs. These interfaces are increasingly critical for ATC operations but '
        'were not designed with quantum-safe cryptography. QBITEL Bridge protects SWIM through: '
        'quantum-safe TLS 1.3 for all SWIM web service connections; PQC-signed data records for '
        'flight data, trajectory, and weather information; quantum-safe authentication for SWIM '
        'service publishers and subscribers; and integrity protection for SWIM data in transit '
        'and at rest. SWIM protection is deployed at ATC facility gateways without modification '
        'to existing SWIM application software.'))

    # ── SECTION 7: Operations & Reliability ────────────────────────────────────
    story.append(PageBreak())
    story.append(SectionHeader('SECTION 7: OPERATIONS & RELIABILITY',
        '5 Questions | 99.999% Uptime, Failover, Maintenance Windows, SOC'))
    story.append(sp(8))

    story.append(make_qa(41,
        'How does QBITEL achieve 99.999% uptime for ATC-grade reliability?',
        'N+1 redundant node architecture with hot standby, geographic distribution, and <500ms automatic failover.',
        'QBITEL Bridge\'s availability architecture includes: N+1 redundant authentication nodes '
        'at each deployment site — if the primary node fails, the standby takes over within 500ms '
        'with no interruption to ATC data flow; geographic redundancy with primary and secondary '
        'data centers for ground-side cryptographic services; stateless authentication verification '
        '— any node can verify any aircraft\'s authentication without session affinity; distributed '
        'PKI with multiple certificate validation endpoints; and watchdog monitoring with automatic '
        'restart for node-level software failures. Planned maintenance can be performed with zero '
        'downtime using rolling updates across the redundant nodes. 99.999% availability has been '
        'demonstrated in 12-month pilot deployments.'))

    story.append(make_qa(42,
        'How does QBITEL handle maintenance windows without interrupting ATC operations?',
        'Zero-downtime rolling maintenance — nodes updated sequentially while remaining nodes handle full load.',
        'QBITEL Bridge updates and maintenance are performed using a rolling update pattern. '
        'In an N+1 configuration: (1) the standby node is updated and verified first; '
        '(2) traffic is switched to the updated standby; (3) the original primary is updated '
        '(now acting as standby); (4) normal N+1 operation resumes. Total planned maintenance '
        'time from start to finish is typically 2-4 hours, with zero impact on ATC operations '
        'throughout. For major version upgrades, QBITEL coordinates a maintenance window with '
        'the ANSP operations team to schedule during low-traffic periods (typically overnight), '
        'even though zero-downtime operation is available.'))

    story.append(make_qa(43,
        'What Security Operations Center (SOC) capabilities does QBITEL provide for aviation customers?',
        '24/7 aviation-specific SOC with spoofing alert triage, threat intelligence, and regulatory incident reporting.',
        'QBITEL\'s aviation SOC provides: 24/7 monitoring of authentication anomalies and spoofing '
        'alerts across all deployed receiver networks; aviation-specific threat intelligence including '
        'known SDR attack signatures, identified threat actor TTPs, and global ADS-B anomaly feeds; '
        'alert triage with trained aviation security analysts — distinguishing genuine attacks from '
        'equipment malfunctions; incident response support including evidence collection for regulatory '
        'reporting; monthly threat reports formatted for ANSP safety management system integration; '
        'and coordination with EUROCONTROL CERT, FAA Cyber Security, and national aviation authority '
        'reporting requirements.'))

    story.append(make_qa(44,
        'How are cryptographic keys managed and rotated in an operational aviation environment?',
        'Automated key lifecycle management with HSM-backed key storage — zero operational impact during routine rotation.',
        'QBITEL\'s key management system handles the full certificate and key lifecycle: '
        'Aircraft enrollment certificates have configurable validity periods (1-3 years typical); '
        'Renewal is initiated 90 days before expiration and completed 30 days before expiration '
        'through an automated renewal process requiring no aircraft downtime; Session keys are '
        'rotated at configurable intervals (default: every 24 hours or 1,000 messages, whichever '
        'comes first); Ground infrastructure keys are stored in FIPS 140-3 Level 3 Hardware '
        'Security Modules; Key ceremonies for root CA operations are documented and auditable '
        'under DO-326A ASL requirements. Emergency key revocation (for compromised aircraft '
        'credentials) propagates to all ground stations within 60 seconds.'))

    story.append(make_qa(45,
        'How does QBITEL handle the special reliability requirements of oceanic and remote airspace?',
        'Satellite-linked receiver network with independent authentication processing at each station — no central point of failure.',
        'Oceanic airspace presents unique challenges: no radar backup, limited SATCOM bandwidth, '
        'and extreme distances. QBITEL\'s oceanic deployment model distributes authentication '
        'processing to individual receiver stations — each station performs independent '
        'behavioral authentication without requiring real-time connection to central servers. '
        'Local authentication decisions are made autonomously and synchronized with central '
        'systems when connectivity permits. For SATCOM-based ACARS/CPDLC communications, '
        'authentication processing occurs at the SATCOM ground earth station. Geographic '
        'distribution of processing nodes eliminates dependence on any single network path '
        'for oceanic authentication services.'))

    # ── SECTION 8: Procurement & Standards Bodies ──────────────────────────────
    story.append(PageBreak())
    story.append(SectionHeader('SECTION 8: PROCUREMENT & STANDARDS BODIES',
        '5 Questions | ICAO, EUROCONTROL, FAA Procurement, Timeline, Framework Contracts'))
    story.append(sp(8))

    story.append(make_qa(46,
        'Is QBITEL recognized by ICAO, EUROCONTROL, or RTCA?',
        'QBITEL participates in RTCA SC-216, EUROCAE WG-75, and has presented technical approach to EUROCONTROL CERT.',
        'QBITEL is an active participant in: RTCA SC-216 (ADS-B Systems) — the committee responsible '
        'for ADS-B MOPS (Minimum Operational Performance Standards); EUROCAE WG-75 (ADS-B) — the '
        'European parallel to RTCA SC-216; and EUROCAE WG-72 (Information Security). QBITEL has '
        'presented its bandwidth-optimized PQC approach to the EUROCAE information security working '
        'group and received favorable technical feedback. QBITEL has engaged with EUROCONTROL CERT '
        '(Computer Emergency Response Team) on the ADS-B threat landscape. ICAO engagement is ongoing '
        'through national delegations. QBITEL\'s technical approach is aligned with the direction of '
        'ICAO\'s Aviation Cyber Security Action Plan.'))

    story.append(make_qa(47,
        'What is the typical procurement timeline for a national ANSP deployment?',
        'Initial contract to Full Operational Status in 36-48 weeks — shorter than most security infrastructure procurements.',
        'QBITEL\'s procurement and deployment timeline for a national ANSP: Contract award to '
        'project initiation: 2-4 weeks (QBITEL maintains deployment readiness). Phase 0 Safety '
        'Assessment: 4 weeks. Phase 1 Ground Infrastructure (first 5 sites): 8 weeks. Phase 2 '
        'Authentication Enrollment (all sites): 8 weeks. Phase 3 AI Detection Training and '
        'Tuning: 8 weeks. Phase 4 Full Operational Status: final 8 weeks. Total: 36-40 weeks '
        'from contract to FOS. The 36-week timeline is achievable because QBITEL\'s ground-only '
        'primary deployment requires no aircraft access, no airworthiness approval delays, '
        'and no ATC system software recertification.'))

    story.append(make_qa(48,
        'Does QBITEL offer framework contracts compatible with government procurement requirements?',
        'Yes — pre-competed framework agreements available in multiple jurisdictions including UK G-Cloud and EU framework structures.',
        'QBITEL maintains framework agreements structured for government procurement in multiple '
        'jurisdictions: UK Crown Commercial Service G-Cloud (Technology Products and Services); '
        'EU public procurement framework compatible with Directive 2014/24/EU; U.S. Federal '
        'procurement framework compatible with FAR Part 12 (Commercial Items) and GSA Schedule; '
        'NATO NSPA framework for member nation defense aviation procurement. These framework '
        'agreements allow ANSPs and national aviation authorities to procure QBITEL Bridge '
        'services without full competitive tender in applicable circumstances. QBITEL can '
        'also support sole-source justification documentation for urgent security requirements.'))

    story.append(make_qa(49,
        'How does QBITEL price its aviation deployments?',
        'Modular pricing by capability tier: ground receiver authentication, ATC network security, airline ops, airborne partition.',
        'QBITEL\'s aviation pricing is structured around four capability modules, each priced '
        'independently: (1) Ground Receiver Authentication — per-receiver-site subscription '
        'covering ADS-B authentication infrastructure and AI spoofing detection; (2) ATC Network '
        'Security — per-facility subscription covering quantum-safe VPN, SWIM protection, '
        'and ATC network hardening; (3) Airline Operations — per-aircraft-type subscription '
        'covering ACARS, SATCOM, and AOC ground infrastructure security; (4) Airborne Partition — '
        'one-time certification package plus per-aircraft-type maintenance subscription. '
        'Volume discounts apply for large receiver networks and fleet sizes. '
        'Contact enterprise@qbitel.com for a tailored proposal.'))

    story.append(make_qa(50,
        'What support and SLA commitments does QBITEL offer for ATC-critical deployments?',
        'Platinum SLA: 99.999% uptime commitment, 15-minute response, 4-hour on-site escalation, 24/7 aviation SOC.',
        'QBITEL\'s Platinum Aviation SLA for ATC-critical deployments includes: 99.999% monthly '
        'uptime guarantee with financial remedies for breach; 15-minute response time for '
        'P1 incidents (authentication system degradation); 4-hour on-site escalation for '
        'hardware failures at ATC facilities; 24/7 aviation-specific SOC staffed by aviation '
        'security analysts; dedicated Technical Account Manager with aviation background; '
        'annual security posture review aligned with ANSP safety management cycle; and '
        'regulatory reporting support for security incident notifications to national aviation '
        'authorities. SLA terms are designed to align with ANSP contractual obligations '
        'to their national aviation authority.'))

    # ── SECTION 9: Hard Objections ─────────────────────────────────────────────
    story.append(PageBreak())
    story.append(SectionHeader('SECTION 9: HARD OBJECTIONS',
        '7 Critical Objections with QBITEL Responses'))
    story.append(sp(8))

    story.append(make_objection(1,
        '"ADS-B spoofing has never caused an actual accident."',
        'You are correct that no commercial aviation accident has been officially attributed to '
        'ADS-B spoofing — yet. However, the same was true of runway incursions before GPS-enabled '
        'ground guidance systems were mandated. Aviation safety operates on the principle of '
        'eliminating known risks before accidents occur, not after. ADS-B spoofing has been '
        'demonstrated to cause genuine TCAS Resolution Advisories in real aircraft. The '
        'unauthenticated broadcast architecture is a documented, well-understood vulnerability '
        'that requires only $20 and freely available software to exploit. The question is not '
        'whether ADS-B spoofing will eventually contribute to an incident — it is whether '
        'your organization will have closed that vulnerability before or after that event.'))

    story.append(make_objection(2,
        '"DO-178C recertification is too expensive — we cannot afford to touch airborne software."',
        'QBITEL agrees completely. That is why our primary deployment architecture requires '
        'zero changes to any certified airborne software. Our ground-only deployment provides '
        'ADS-B authentication, ATC network quantum hardening, and ACARS/SATCOM security '
        'without touching a single line of aircraft software. The DO-178C certification '
        'capability exists for organizations that choose airborne deployment — typically new '
        'aircraft program OEMs who are already in a certification cycle. For everyone else, '
        'ground-only deployment closes the most critical vulnerabilities at a fraction of '
        'the cost and with no certification risk.'))

    story.append(make_objection(3,
        '"ICAO has not mandated ADS-B authentication yet. We will wait for the mandate."',
        'Waiting for an ICAO mandate is a reasonable position for commodity IT procurement. '
        'For aviation security, it is a calculated risk decision. ICAO working groups on '
        'ADS-B authentication have been meeting for over a decade without producing a deployed '
        'standard. The mandate, when it comes, will have a compliance deadline — and the '
        'implementation timeline will be measured in years, not months. Organizations that '
        'begin now will have operational experience, trained staff, and certified infrastructure '
        'when the mandate arrives. Organizations that wait will face emergency implementation '
        'under time pressure. QBITEL\'s architecture is designed to evolve with ICAO standards — '
        'your investment is protected regardless of how the mandate is written.'))

    story.append(make_objection(4,
        '"Our aircraft are too old to upgrade — we cannot install new hardware."',
        'Legacy aircraft require no hardware installation for QBITEL\'s protection. Our '
        'ground-based behavioral authentication and MLAT cross-validation protects all aircraft '
        'in your airspace regardless of their age or equipment. The spoofing detection AI '
        'monitors ADS-B transmissions from every aircraft — registered or not — and flags '
        'positions that fail physics consistency, MLAT correlation, or receiver network '
        'consistency checks. For your ground infrastructure (ATC networks, ACARS gateways, '
        'SATCOM ground earth stations), QBITEL provides full quantum-safe protection with '
        'no aircraft involvement. Legacy aircraft protection is included in all ground '
        'deployment packages at no additional cost.'))

    story.append(make_objection(5,
        '"We cannot risk any latency on safety-critical communications systems."',
        'QBITEL agrees that safety-critical systems cannot accept latency. That is why our '
        'authentication layer is designed with zero latency for safety-critical paths. '
        'ADS-B data flows to ATC displays with authentication status — the authentication '
        'process does not gate or delay ATC display updates. Unauthenticated messages are '
        'displayed immediately with an "unverified" indicator. Authenticated messages are '
        'displayed with a "verified" indicator after MAC verification (less than 50ms). '
        'CPDLC and safety-critical ACARS messages use pre-established session keys that '
        'add less than 1ms of authentication overhead. No safety-critical message is '
        'ever blocked or delayed by QBITEL authentication processing.'))

    story.append(make_objection(6,
        '"Ground radar backup makes ADS-B security unnecessary."',
        'Ground radar provides valuable redundancy and we strongly support maintaining radar '
        'coverage. However, radar backup does not address ADS-B security for several reasons: '
        'Oceanic airspace has no radar coverage — ADS-B is the only surveillance source. '
        'Airport surface surveillance (A-SMGCS) relies on ADS-B without radar backup. '
        'Commercial ADS-B aggregators used by airlines, ground handlers, and passenger apps '
        'have no radar backup. SSR radar altitude encoding is lower resolution than ADS-B GPS altitude. '
        'A sophisticated spoofing attack that maintains consistency between ADS-B and SSR '
        '(by spoofing at the SSR Mode C altitude) could deceive both systems simultaneously. '
        'QBITEL enhances radar-ADS-B correlation, making both data sources more reliable.'))

    story.append(make_objection(7,
        '"We need EUROCAE/RTCA standards for ADS-B authentication before we can deploy."',
        'Standards development and operational deployment are not mutually exclusive. QBITEL\'s '
        'architecture is designed to align with the direction of RTCA SC-216 and EUROCAE WG-75 '
        'standards work. The ground-based authentication approach QBITEL implements is consistent '
        'with the authentication architectures under discussion in these working groups. '
        'Importantly, standards bodies explicitly encourage pre-standard operational trials '
        'to inform standard development — your deployment would contribute operational data '
        'to the standards process. QBITEL participates in both SC-216 and WG-75 and '
        'commits to updating its architecture to align with final standards at no '
        'additional cost to deployed customers.'))

    # ── SECTION 10: Competitive ────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(SectionHeader('SECTION 10: COMPETITIVE INTELLIGENCE',
        '8 Questions | vs. Radar-Only, vs. Working Groups, vs. Wait-and-See, vs. IT Vendors'))
    story.append(sp(8))

    story.append(make_qa(51,
        'How does QBITEL compare to pure radar-based solutions for ADS-B security?',
        'Radar validates but does not authenticate — QBITEL provides cryptographic certainty radar cannot offer.',
        'Radar-based cross-validation (the standard approach for ADS-B anomaly detection) works by '
        'comparing ADS-B-reported positions with radar-calculated positions. This detects gross '
        'position falsification but has significant limitations: (1) No oceanic or remote coverage; '
        '(2) Lower resolution than ADS-B GPS altitude; (3) A sophisticated attack maintaining '
        'ADS-B/radar consistency passes detection; (4) Radar does not authenticate identity — '
        'a spoofed ICAO address presenting consistent position data is indistinguishable. '
        'QBITEL provides cryptographic authentication that detects identity fraud regardless '
        'of position consistency, plus behavioral analysis for non-enrolled aircraft. '
        'The combination of QBITEL plus radar cross-validation is more secure than either alone.'))

    story.append(make_qa(52,
        'What if a customer says they will just wait for ICAO to solve the problem?',
        'ICAO is working on the problem — but waiting is itself a risk management decision with known consequences.',
        'Respect the customer\'s position, then help them understand the risk calculation: '
        'ICAO ADS-B authentication discussions began in earnest around 2012. Twelve years later, '
        'no standard has been deployed. The ICAO process — consensus-based, multi-nation, technically '
        'conservative — takes time. When a standard is published, implementation mandates will '
        'follow years later. Any organization waiting for the mandate will face: compressed '
        'implementation timeline with no operational experience; potential emergency certification '
        'costs if mandate includes airborne requirements; competitors who have gained operational '
        'advantage. QBITEL can be deployed, operated, and refined while standards are developed — '
        'and is committed to standards alignment at no customer cost.'))

    story.append(make_qa(53,
        'How does QBITEL compare to building an in-house ADS-B security solution?',
        'In-house development lacks aviation PKI, DO-326A expertise, receiver network, and PQC compression — 3-5 year build.',
        'Organizations that have considered in-house development typically discover: Building '
        'DO-326A-compliant security documentation requires specialist expertise not available '
        'in most IT security teams; Aviation PKI infrastructure for aircraft identity management '
        'requires trust anchors, cross-certification, and certificate lifecycle management '
        'at global scale; Multilateration receiver network integration requires relationships '
        'with receiver network operators (OpenSky, commercial networks); PQC compression for '
        'aviation bandwidths is a specialized research problem QBITEL has solved over 3+ years; '
        'ATC system integration (ASTERIX, SDPS, SWIM) requires deep aviation domain knowledge. '
        'Estimated in-house build time: 3-5 years. QBITEL deploys in 36 weeks.'))

    story.append(make_qa(54,
        'What vendors are competing with QBITEL in the aviation security space?',
        'No direct competitors offer the combination of ground-only ADS-B authentication, bandwidth-optimized PQC, and DO-326A evidence.',
        'The aviation security market is fragmented across several approaches: Traditional '
        'aviation cybersecurity consultancies (Thales, SITA Cybersecurity, Airbus CyberSecurity) '
        'offer assessment and advisory services but not operational ADS-B authentication solutions; '
        'Academic research groups (OpenSky Network, TU Kaiserslautern, ETH Zurich) have published '
        'ADS-B authentication research but do not offer commercial products; IT security vendors '
        '(Palo Alto, Fortinet) offer aviation-adjacent network security but not ADS-B authentication '
        'or aviation PQC; ICAO working groups are producing standards but not products. '
        'QBITEL is the only commercial vendor offering integrated ADS-B authentication plus '
        'bandwidth-optimized PQC plus DO-326A compliance evidence as a unified platform.'))

    story.append(make_qa(55,
        'How does QBITEL handle the objection that "classical cryptography is good enough for now"?',
        'It may be "good enough for now" — but aircraft certified today will operate past "now," into the quantum era.',
        '"Good enough for now" is the right framing. Classical cryptography (AES-256, SHA-384) is '
        'not broken by quantum computers and provides adequate symmetric security. RSA and ECDSA '
        '(used for key exchange and digital signatures) will be broken by quantum computers '
        'estimated 2030-2040. The key question is: what are you securing, and for how long? '
        'Aircraft certified today have 35-40-year service lives. Communications infrastructure '
        'for those aircraft must remain secure throughout. If you deploy classical PKI today '
        'for aircraft that will fly until 2060, you will need an emergency migration in the '
        '2030s — requiring airborne software updates (DO-178C recertification) for every '
        'aircraft type. QBITEL eliminates that emergency by implementing PQC now.'))

    story.append(make_qa(56,
        'An airline security team says "we have bigger priorities." How do you respond?',
        'Acknowledge priorities, then show how QBITEL addresses regulatory risk, operational risk, and board-level reporting simultaneously.',
        'Airline CISOs are dealing with ransomware, IT/OT separation, passenger data protection, '
        'and regulatory compliance across multiple jurisdictions simultaneously. QBITEL is not '
        'asking to replace those priorities. The question is: what happens when a regulator asks '
        'about ADS-B security posture? What happens when an ACARS message forging incident '
        'affects operations? What happens when the "harvest now, decrypt later" threat to '
        'operational data becomes a board-level conversation? QBITEL provides: regulatory '
        'compliance documentation for FAA, EASA, and national authority requirements; '
        'measurable operational risk reduction for ACARS and SATCOM; and quantifiable '
        'cyber risk metrics for board reporting. These outcomes align with existing priorities '
        'rather than competing with them.'))

    story.append(make_qa(57,
        'What is QBITEL\'s response if a customer has just had a cybersecurity assessment that did not mention ADS-B?',
        'Most aviation cybersecurity assessments focus on IT/OT separation and network security — ADS-B RF layer is a blind spot.',
        'Aviation cybersecurity assessments typically focus on: aircraft network connectivity '
        '(cabin vs. avionics network separation), airline IT infrastructure (reservation systems, '
        'baggage systems, crew systems), and ATC network security (SWIM, radar data distribution). '
        'ADS-B security at the RF layer is rarely included because it requires specialized RF '
        'security expertise that is uncommon in general cybersecurity consultancies. The absence '
        'of ADS-B in an assessment does not mean it was evaluated and found acceptable — '
        'it most likely means it was not in scope. QBITEL offers a complimentary ADS-B '
        'vulnerability assessment as a first engagement activity.'))

    story.append(make_qa(58,
        'How does QBITEL differentiate from academic ADS-B security research (OpenSky, TU Kaiserslautern)?',
        'Academic research identifies and quantifies threats; QBITEL provides operational deployment with ATC integration and compliance documentation.',
        'Academic research groups have made enormous contributions to understanding ADS-B '
        'vulnerabilities. The OpenSky Network (ETH Zurich, TU Kaiserslautern, University of '
        'Kaiserslautern) has published seminal papers on ADS-B spoofing detection, multilateration, '
        'and authentication architectures. QBITEL has built on this research foundation. The '
        'difference is operational readiness: academic research demonstrates concepts; QBITEL '
        'provides production-grade software with ATC ASTERIX/SDPS integration, aviation-grade '
        'reliability (99.999%), DO-326A compliance documentation, 24/7 SOC support, and '
        'legal SLA commitments. QBITEL maintains collaborative relationships with key academic '
        'groups and supports their research through data sharing agreements.'))

    doc.build(story)
    print(f"PDF written: {output_path}")


if __name__ == '__main__':
    import os
    out = '/Users/prabakarankannan/qbitel/docs/brochures/QBITEL_Aviation_Pitch_QA_Guide.pdf'
    build_doc(out)
    size = os.path.getsize(out)
    print(f"File size: {size:,} bytes ({size/1024:.1f} KB)")
