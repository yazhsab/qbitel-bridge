"""Build QBITEL Bridge Critical Infrastructure Q&A Guide - PDF"""
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
RED_DARK   = HexColor('#8B1A1A')
GREEN_DARK = HexColor('#1A6B3A')
PAGE_W, PAGE_H = letter
MARGIN = 0.85 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

def sp(n): return Spacer(1, n)


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


class QABlock(Flowable):
    def __init__(self, question, one_liner, answer, width=None):
        super().__init__()
        self.question = question
        self.one_liner = one_liner
        self.answer = answer
        self.w = width or CONTENT_W
        # Estimate text height for answer
        chars_per_line = int(self.w / 5.5)
        lines_needed = max(2, len(answer) // chars_per_line + answer.count('\n') + 2)
        self.text_h = lines_needed * 13
        self.h = 26 + 20 + self.text_h + 12

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Navy header with question
        c.setFillColor(NAVY)
        c.rect(0, self.h - 26, self.w, 26, fill=1, stroke=0)
        c.setFillColor(GOLD)
        c.rect(0, self.h - 26, 4, 26, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 9)
        # Truncate question if needed
        q_text = self.question
        max_w = self.w - 20
        while c.stringWidth(q_text, 'Helvetica-Bold', 9) > max_w and len(q_text) > 10:
            q_text = q_text[:-4] + '...'
        c.drawString(12, self.h - 17, q_text)

        # Teal band with one-liner
        c.setFillColor(TEAL)
        c.rect(0, self.h - 46, self.w, 20, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-BoldOblique', 8.5)
        ol_text = self.one_liner
        max_w2 = self.w - 20
        while c.stringWidth(ol_text, 'Helvetica-BoldOblique', 8.5) > max_w2 and len(ol_text) > 10:
            ol_text = ol_text[:-4] + '...'
        c.drawString(12, self.h - 39, ol_text)

        # White body with answer
        c.setFillColor(LIGHT_BG)
        c.rect(0, 0, self.w, self.h - 46, fill=1, stroke=0)
        c.setFillColor(MID_GREY)
        c.rect(0, 0, self.w, self.h - 46, fill=0, stroke=1)
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 8.5)
        # Word wrap the answer
        words = self.answer.split()
        lines = []
        current_line = ''
        max_chars = int((self.w - 24) / 5.2)
        for word in words:
            test = (current_line + ' ' + word).strip()
            if len(test) <= max_chars:
                current_line = test
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        body_h = self.h - 46
        start_y = body_h - 14
        for i, line in enumerate(lines):
            y = start_y - i * 13
            if y < 6:
                break
            c.drawString(12, y, line)


class ObjectionBlock(Flowable):
    def __init__(self, objection, response, width=None):
        super().__init__()
        self.objection = objection
        self.response = response
        self.w = width or CONTENT_W
        chars_per_line = int(self.w / 5.5)
        resp_lines = max(2, len(response) // chars_per_line + 3)
        self.h = 30 + resp_lines * 13 + 16

    def wrap(self, avw, avh):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Dark red header - objection
        c.setFillColor(RED_DARK)
        c.rect(0, self.h - 30, self.w, 30, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 8.5)
        c.drawString(12, self.h - 13, 'OBJECTION:')
        c.setFont('Helvetica-Bold', 9)
        obj_text = self.objection
        max_w = self.w - 110
        while c.stringWidth(obj_text, 'Helvetica-Bold', 9) > max_w and len(obj_text) > 10:
            obj_text = obj_text[:-4] + '...'
        c.drawString(100, self.h - 13, obj_text)

        # Green body - response
        body_h = self.h - 30
        c.setFillColor(GREEN_DARK)
        c.rect(0, 0, self.w, body_h, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 8)
        c.drawString(12, body_h - 14, 'RESPONSE:')
        c.setFont('Helvetica', 8.5)
        words = self.response.split()
        lines = []
        current_line = ''
        max_chars = int((self.w - 24) / 5.0)
        for word in words:
            test = (current_line + ' ' + word).strip()
            if len(test) <= max_chars:
                current_line = test
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        start_y = body_h - 28
        for i, line in enumerate(lines):
            y = start_y - i * 13
            if y < 6:
                break
            c.drawString(12, y, line)


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H-30, PAGE_W, 30, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C); canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(MARGIN, PAGE_H-20, 'QBITEL BRIDGE - CRITICAL INFRASTRUCTURE Q&A GUIDE')
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
    canvas.drawString(MARGIN, PAGE_H*0.55 + PAGE_H*0.45*0.44, 'Sales Q&A Guide')
    canvas.setFont('Helvetica', 13)
    canvas.drawString(MARGIN, PAGE_H*0.55 + PAGE_H*0.45*0.30, '65 Questions Across 10 Sections')
    canvas.drawString(MARGIN, PAGE_H*0.55 + PAGE_H*0.45*0.18, 'Architecture | Safety | Compliance | Objection Handling')
    stats = [('65', 'Q&As'), ('10', 'Sections'), ('7', 'Hard Objections'), ('8', 'Competitive')]
    box_w = (PAGE_W - 2*MARGIN - 30) / 4
    for i, (val, lbl) in enumerate(stats):
        x = MARGIN + i*(box_w+10); y = 0.12*PAGE_H
        canvas.setFillColor(LIGHT_NAVY); canvas.roundRect(x, y, box_w, 0.1*PAGE_H, 6, fill=1, stroke=0)
        canvas.setFillColor(GOLD); canvas.setFont('Helvetica-Bold', 18)
        vw = canvas.stringWidth(val, 'Helvetica-Bold', 18)
        canvas.drawString(x + box_w/2 - vw/2, y + 0.1*PAGE_H*0.55, val)
        canvas.setFillColor(WHITE_C); canvas.setFont('Helvetica', 8)
        lw = canvas.stringWidth(lbl, 'Helvetica', 8)
        canvas.drawString(x + box_w/2 - lw/2, y + 0.1*PAGE_H*0.25, lbl)
    canvas.setFillColor(TEAL); canvas.rect(0, 0, PAGE_W, 0.08*PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C); canvas.setFont('Helvetica', 9)
    canvas.drawString(MARGIN, 0.04*PAGE_H, 'enterprise@qbitel.com  |  bridge.qbitel.com')
    canvas.restoreState()


def build_qa_doc(output_path):
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

    # =========================================================
    # SECTION 1: OT/ICS ARCHITECTURE
    # =========================================================
    story.append(SectionHeader('Section 1: OT/ICS Architecture', '8 questions on protocols, zones, and network design'))
    story.append(sp(8))

    qa1 = [
        ('What is Modbus and why does it have no security?',
         'Modbus was designed in 1979 for serial communication - security was not conceived.',
         'Modbus is a serial communication protocol invented by Modicon in 1979 for connecting PLCs on factory floors. '
         'At the time, industrial networks were physically isolated and security was not considered. '
         'The protocol has no authentication, no encryption, and no integrity checking - any device on the network can send commands to any PLC.'),

        ('What is the difference between a PLC and an RTU?',
         'PLCs handle discrete logic control; RTUs handle remote telemetry in field environments.',
         'A Programmable Logic Controller (PLC) is designed for high-speed discrete logic control in manufacturing environments - '
         'it reads digital and analog inputs and controls outputs based on a ladder logic program. '
         'A Remote Terminal Unit (RTU) is designed for remote monitoring and control over wide-area SCADA systems, '
         'typically in utilities, and communicates via protocols like DNP3 or IEC 60870-5-101 over long distances.'),

        ('What is IEC 61850 and where is it used?',
         'IEC 61850 is the substation automation standard used across global power grid infrastructure.',
         'IEC 61850 is an international standard for substation automation and protection. '
         'It defines communication protocols including GOOSE (Generic Object Oriented Substation Event) for fast protection '
         'tripping, Sampled Values (SV) for digitized current and voltage measurements, and MMS for control. '
         'GOOSE messages travel as unauthenticated multicast and must operate within 4ms - '
         'making them extremely challenging to secure without violating timing requirements.'),

        ('What is the Purdue Model / ISA-95 architecture?',
         'The Purdue Model defines five network levels from field devices to enterprise, with strict zone separation.',
         'The Purdue Enterprise Reference Architecture (PERA) defines five levels: Level 0 (field devices/sensors), '
         'Level 1 (PLCs/RTUs), Level 2 (SCADA/HMI), Level 3 (Operations/MES), and Level 4/5 (Enterprise/IT). '
         'The model prescribes strict separation between levels with industrial DMZs at the IT/OT boundary. '
         'Modern OT convergence trends are collapsing these boundaries, increasing attack surface.'),

        ('What is IT/OT convergence and why is it increasing risk?',
         'IT/OT convergence connects traditionally isolated OT networks to enterprise IT, dramatically expanding attack surface.',
         'IT/OT convergence describes the trend of connecting operational technology networks (controlling physical processes) '
         'to information technology networks (data processing). '
         'Drivers include remote monitoring, predictive maintenance, supply chain integration, and cloud analytics. '
         'The consequence is that OT networks that were air-gapped are now reachable from the internet, '
         'and IT-side compromises (ransomware, phishing) can now pivot to OT systems.'),

        ('What is a zone and conduit model in OT security?',
         'Zones group assets by security level; conduits define and control communication between zones.',
         'IEC 62443 defines security zones as groupings of OT assets with similar security requirements and trust levels. '
         'A Safety Zone contains SIS systems at the highest protection level. '
         'A Control Zone contains PLCs and RTUs. '
         'A Supervisory Zone contains SCADA/HMI systems. '
         'Conduits are the defined, controlled communication paths between zones with explicit security policies - '
         'any communication not defined in a conduit should be blocked.'),

        ('What protocols does QBITEL Bridge support?',
         'QBITEL supports all major OT protocols: Modbus, DNP3, IEC 61850, OPC UA, BACnet, EtherNet/IP, PROFINET.',
         'QBITEL Bridge provides passive discovery and active protection for the full range of industrial protocols: '
         'Modbus TCP and RTU (manufacturing, utilities), DNP3 (electric and water utilities), '
         'IEC 61850 GOOSE and Sampled Values (substation automation), OPC UA (modern manufacturing, cross-vendor), '
         'BACnet (building management systems), EtherNet/IP (Rockwell/Allen-Bradley), and PROFINET (Siemens). '
         'Legacy serial protocols are supported via serial tap adapters.'),

        ('What is OPC UA and does it have built-in security?',
         'OPC UA has a security framework but is widely deployed in "None" mode for legacy compatibility.',
         'OPC UA (Open Platform Communications Unified Architecture) is a modern industrial protocol designed with '
         'security in mind - it supports message-level security with three modes: None, Sign, and SignAndEncrypt. '
         'However, surveys consistently find that 40-60% of industrial OPC UA deployments run in "None" mode '
         'because legacy clients do not support security or because operators prioritize compatibility over security. '
         'QBITEL enforces SignAndEncrypt mode even for legacy clients through a transparent proxy approach.'),
    ]

    for q, ol, a in qa1:
        story.append(QABlock(q, ol, a))
        story.append(sp(6))

    story.append(PageBreak())

    # =========================================================
    # SECTION 2: SAFETY & AVAILABILITY
    # =========================================================
    story.append(SectionHeader('Section 2: Safety & Availability', '8 questions on SIL, IEC 61508, uptime, and graceful degradation'))
    story.append(sp(8))

    qa2 = [
        ('What is a Safety Integrity Level (SIL) and why does it matter for QBITEL?',
         'SIL is a quantitative measure of the risk reduction provided by a safety function - SIL 4 is highest.',
         'Safety Integrity Levels (SIL 1-4) are defined in IEC 61508 and represent the probability of a safety function '
         'failing on demand. SIL 4 requires a probability of failure on demand of less than 0.0001 per hour. '
         'For QBITEL, SIL ratings define which systems require the most conservative approach: '
         'SIS systems at SIL 3/4 must never be modified or have active protection applied, '
         'only passive monitoring, and all anomalies must escalate to human operators.'),

        ('How does QBITEL guarantee it will not affect SIL-certified systems?',
         'QBITEL is architecturally excluded from SIS systems - passive monitoring only, zero active protection.',
         'QBITEL maintains a formally documented Safety Exclusion Zone configuration that prevents any active response '
         'on SIS network segments. The safety boundary is documented during Phase 4 of deployment and '
         'requires written sign-off from the customer Safety Officer. '
         'QBITEL operates as a separate monitoring layer outside the SIS certification boundary - '
         'it never modifies SIS firmware, configuration, or traffic. IEC 61508 certification is fully preserved.'),

        ('How does QBITEL achieve 99.999% availability?',
         'Hardware bypass failsafe, redundant appliances, and fail-open design ensure five-nines availability.',
         'QBITEL achieves 99.999% availability through three mechanisms: hardware bypass relays that pass traffic '
         'unimpeded if the QBITEL appliance loses power or software fails; active-active clustering for high-availability '
         'deployments; and a fail-open software policy where any PQC processing failure results in traffic passing with '
         'an alert rather than dropping. The passive tap mode has no single point of failure by design.'),

        ('What happens if QBITEL fails during a production run?',
         'Hardware bypass relay activates instantly - traffic passes unimpeded, SOC receives immediate alert.',
         'QBITEL appliances include a hardware bypass relay - a physical failsafe that connects the network ports '
         'directly if the appliance loses power or experiences a critical software fault. '
         'This hardware mechanism operates at line speed with zero software involvement. '
         'The SOC receives an immediate alert. Traffic continues to flow unimpeded. '
         'No SCADA polling is interrupted. No control loop timing is violated. '
         'The design principle is clear: QBITEL must never cause the operational disruption it is protecting against.'),

        ('What is graceful degradation in the QBITEL context?',
         'If PQC is unavailable, traffic passes with an alert - operations never stop for security.',
         'Graceful degradation means that QBITEL security layers fail in a direction that preserves operations. '
         'If the ML-KEM-768 key establishment fails, traffic passes in cleartext with a SOC alert. '
         'If the physics anomaly engine is unavailable, network-layer detection continues. '
         'If the HSM is unreachable, cached session keys are used temporarily. '
         'Each degradation state is logged, alerted, and has a defined remediation procedure.'),

        ('How does QBITEL handle change freeze periods in production facilities?',
         'Passive discovery has zero operational impact - it can run during any change freeze period.',
         'Industrial facilities often implement change freeze periods during peak production cycles, seasonal operations, '
         'or around major events. QBITEL phases are designed to respect this: '
         'the passive discovery phase (Phase 1) has absolutely zero operational impact and can run continuously '
         'through any change freeze. Active protection phases are scheduled to your maintenance windows '
         'with explicit operator approval at each activation step.'),

        ('What is the impact of QBITEL on SCADA polling latency?',
         'Zero impact in passive mode; <1ms additional latency in inline mode for PQC-authenticated traffic.',
         'In passive tap mode, QBITEL has zero impact on SCADA polling latency - the tap is physically receive-only '
         'and cannot add delay to the network path. '
         'In inline authentication mode, the additional latency for ML-DSA-65 signature verification is '
         'less than 1 millisecond per transaction, with jitter less than 100 microseconds. '
         'This has been validated against typical DNP3 and Modbus polling rates of 1-10 seconds.'),

        ('How does QBITEL handle false positives that could trigger unnecessary shutdowns?',
         'Physics-aware validation reduces false positives to <0.001% - the physics model eliminates noise.',
         'Traditional network anomaly detection in OT environments produces high false positive rates because '
         'industrial process variation looks like anomalies to statistical models. '
         'QBITEL cross-validates network anomalies against physical process models built from historian data. '
         'A network-level anomaly that corresponds to expected process behavior - a pump speed variation matching '
         'a flow rate change - is suppressed. Only anomalies that cannot be explained by process physics trigger alerts. '
         'The false positive rate is less than 0.001%.'),
    ]

    for q, ol, a in qa2:
        story.append(QABlock(q, ol, a))
        story.append(sp(6))

    story.append(PageBreak())

    # =========================================================
    # SECTION 3: PROTOCOL SECURITY
    # =========================================================
    story.append(SectionHeader('Section 3: Protocol Security', '6 questions on Modbus auth, DNP3, IEC 61850, OPC UA'))
    story.append(sp(8))

    qa3 = [
        ('How does QBITEL add authentication to Modbus without changing the protocol?',
         'QBITEL operates as a transparent network-layer proxy - Modbus devices see normal Modbus traffic.',
         'QBITEL deploys a transparent authentication proxy at the network layer. '
         'When an HMI sends a Modbus WRITE command, QBITEL intercepts it, appends an ML-DSA-65 signature '
         'in a separate authentication channel, and forwards the original Modbus packet to the PLC. '
         'A QBITEL verification agent at the PLC network port validates the signature before forwarding to the PLC. '
         'The PLC sees normal Modbus TCP - no firmware changes required. '
         'Unauthorized commands without valid signatures are dropped at the network layer.'),

        ('Does DNP3 Secure Authentication v5 provide sufficient security?',
         'DNP3 SAv5 exists but is rarely deployed - QBITEL provides SAv5 where absent and adds PQC.',
         'DNP3 Secure Authentication version 5 (SAv5) adds challenge-response authentication to DNP3 and was '
         'standardized in IEEE 1815-2012. However, surveys of utility SCADA deployments consistently show '
         'that less than 15% have SAv5 enabled, typically because legacy RTUs do not support it. '
         'QBITEL provides SAv5-equivalent authentication where RTUs do not support it natively, '
         'and upgrades existing SAv5 deployments to post-quantum cryptography.'),

        ('How does QBITEL protect IEC 61850 GOOSE messages given the 4ms timing requirement?',
         'QBITEL uses hardware-accelerated ML-DSA-65 with <100us overhead - well within the 4ms window.',
         'IEC 61850 GOOSE messages must be delivered within 4ms for protection relay applications. '
         'QBITEL uses hardware-accelerated ML-DSA-65 signature generation and verification with '
         'processing time less than 100 microseconds - less than 3% of the 4ms budget. '
         'This meets the IEC 62351-8 requirements for GOOSE authentication. '
         'QBITEL implements the full IEC 62351 security suite for power system communications.'),

        ('What is IEC 62351 and how does QBITEL implement it?',
         'IEC 62351 is the power systems security standard - QBITEL implements all relevant parts.',
         'IEC 62351 is the International Electrotechnical Commission standard for security of power system '
         'communication including IEC 61850 (Part 6), GOOSE/SV authentication (Part 6 and 8), '
         'and key management (Part 8). QBITEL implements IEC 62351 Parts 3, 4, 5, 6, and 8, '
         'providing TLS for MMS, authentication for GOOSE and Sampled Values, '
         'and a key management infrastructure compatible with IEC 62351-8 requirements.'),

        ('Can QBITEL protect BACnet environments in building management systems?',
         'Yes - QBITEL supports BACnet/IP and provides anomaly detection and command authentication.',
         'BACnet (Building Automation and Control Networks) is widely used in HVAC, access control, '
         'and building management systems. BACnet/IP has minimal security options in most deployments. '
         'QBITEL provides passive discovery of BACnet devices, anomaly detection for unauthorized command patterns, '
         'and network-layer authentication for BACnet/IP traffic. '
         'Building management systems in critical infrastructure facilities (data centers, hospitals, government buildings) '
         'increasingly require the same level of protection as traditional OT systems.'),

        ('How does QBITEL handle EtherNet/IP and PROFINET protocols?',
         'QBITEL provides native passive discovery and command authentication for EtherNet/IP and PROFINET.',
         'EtherNet/IP (used extensively in Rockwell/Allen-Bradley environments) and PROFINET (Siemens) '
         'are the dominant protocols in modern discrete manufacturing. '
         'Both protocols lack authentication for control-plane messages - '
         'any device on the factory network can send I/O data or explicit messaging to PLCs. '
         'QBITEL decodes and monitors EtherNet/IP CIP messaging and PROFINET DCP/RT frames, '
         'detecting unauthorized command sources and protocol anomalies.'),
    ]

    for q, ol, a in qa3:
        story.append(QABlock(q, ol, a))
        story.append(sp(6))

    story.append(PageBreak())

    # =========================================================
    # SECTION 4: NERC CIP COMPLIANCE
    # =========================================================
    story.append(SectionHeader('Section 4: NERC CIP Compliance', '6 questions on CIP standards, automated evidence, violation fines'))
    story.append(sp(8))

    qa4 = [
        ('What is NERC CIP and who does it apply to?',
         'NERC CIP applies to all owners and operators of Bulk Electric System cyber assets in North America.',
         'NERC CIP (North American Electric Reliability Corporation Critical Infrastructure Protection) '
         'is a set of mandatory cybersecurity standards for the North American bulk electric system. '
         'It applies to utilities, power generators, transmission operators, and any organization owning '
         'or operating Bulk Electric System (BES) Cyber Systems. '
         'Violations can cost up to $1 million per day per violation. '
         'The standards range from CIP-002 (BES Cyber System categorization) through CIP-014 (physical security).'),

        ('What is CIP-005 Electronic Security Perimeter and how does QBITEL help?',
         'CIP-005 requires defined ESPs around BES Cyber Systems - QBITEL automates ESP boundary monitoring.',
         'NERC CIP CIP-005 requires utilities to define Electronic Security Perimeters around BES Cyber Systems '
         'and control all access across ESP boundaries. '
         'QBITEL automates ESP boundary monitoring by continuously logging every communication crossing zone boundaries, '
         'detecting unauthorized devices attempting to communicate with BES systems, '
         'and generating audit-ready evidence of ESP compliance. '
         'CIP-005 evidence is populated automatically from passive discovery data.'),

        ('What does CIP-007 require and how does QBITEL address it?',
         'CIP-007 requires system security management including ports, services, patch status, and access control.',
         'NERC CIP CIP-007 (Systems Security Management) requires utilities to maintain approved port and service lists, '
         'manage security patches, control physical and logical access to BES Cyber Systems, '
         'and log and review security events. '
         'QBITEL continuously monitors active ports and services on OT devices, detecting any deviation from '
         'the documented baseline. Patch status is tracked through passive analysis of protocol versions. '
         'All security events are logged in a format directly usable for CIP-007 compliance evidence.'),

        ('How does QBITEL automate NERC CIP evidence generation?',
         'QBITEL generates complete audit-ready evidence packages in <10 minutes with requirement citations.',
         'QBITEL maintains a continuous compliance telemetry stream mapped to specific NERC CIP requirements. '
         'When a compliance report is requested - by the compliance team or triggered by an approaching audit - '
         'QBITEL generates a structured evidence package in less than 10 minutes. '
         'The package includes timestamped logs, control effectiveness evidence, exception reports, '
         'and cross-references to specific CIP standard requirements. '
         'Evidence is formatted to match NERC CIP auditor expectations.'),

        ('What are the maximum fines for NERC CIP violations?',
         'NERC CIP violations can reach $1 million per day per violation - Colonial Pipeline fines exceeded $1M.',
         'FERC (Federal Energy Regulatory Commission) can impose civil penalties of up to $1 million per violation '
         'per day for NERC CIP non-compliance. '
         'In 2022, NERC levied a $3.86 million fine for multiple CIP violations by a single utility. '
         'Beyond direct fines, NERC CIP violations result in mandatory remediation plans, '
         'increased audit frequency, and reputational damage with regulators. '
         'The cost of a comprehensive QBITEL deployment is typically a fraction of a single year\'s violation exposure.'),

        ('Does NERC CIP require encryption for OT communications?',
         'CIP-011 and CIP-005 strongly benefit from encryption - and FERC has signaled stricter requirements ahead.',
         'NERC CIP CIP-011 (Information Protection) requires protection of BES Cyber System Information, '
         'which encompasses sensitive data including network topology and control system configurations. '
         'CIP-005 requires access controls on ESP boundaries. '
         'While current standards do not mandate encryption of all OT protocol traffic, '
         'FERC has indicated in recent rulemaking that enhanced supply chain and communication security requirements '
         'are forthcoming. Organizations deploying PQC encryption now will be ahead of mandatory requirements.'),
    ]

    for q, ol, a in qa4:
        story.append(QABlock(q, ol, a))
        story.append(sp(6))

    story.append(PageBreak())

    # =========================================================
    # SECTION 5: IEC 62443 & ZONE/CONDUIT
    # =========================================================
    story.append(SectionHeader('Section 5: IEC 62443 & Zone/Conduit Security', '5 questions on zone classification, SL assessment, ISA-99'))
    story.append(sp(8))

    qa5 = [
        ('What is IEC 62443 and how does it differ from NERC CIP?',
         'IEC 62443 is the international OT security standard; NERC CIP is sector-specific to electric utilities.',
         'IEC 62443 (formerly ISA-99) is the international standard for Industrial Automation and Control System '
         'security, developed by ISA (International Society of Automation) and adopted by IEC. '
         'It applies across all industrial sectors - manufacturing, oil and gas, water, transportation - '
         'whereas NERC CIP applies exclusively to the North American bulk electric system. '
         'IEC 62443 uses a zone and conduit model with Security Levels (SL 1-4) rather than CIP-specific categories. '
         'Many critical infrastructure operators must comply with both.'),

        ('How do IEC 62443 Security Levels work?',
         'IEC 62443 defines SL 1-4 representing protection against casual, intentional, sophisticated, and state-level attacks.',
         'IEC 62443 Security Levels define the required protection against progressively sophisticated attackers: '
         'SL 1 protects against casual or unintentional violations, '
         'SL 2 protects against intentional attacks with low resources, '
         'SL 3 protects against sophisticated intentional attacks with moderate resources, '
         'SL 4 protects against state-sponsored attacks with extended resources. '
         'Critical infrastructure controlling national-impact systems should target SL 3 minimum, with SIS at SL 4.'),

        ('What is an IEC 62443 SL-T assessment and how does QBITEL automate it?',
         'SL-T is the target security level assessment - QBITEL maps capabilities to SL requirements automatically.',
         'An IEC 62443 Security Level Target (SL-T) assessment evaluates whether the security controls in place '
         'for each zone meet the target security level. '
         'QBITEL maps its deployed capabilities - passive discovery, PLC command authentication, '
         'PQC encryption, physics-aware anomaly detection - to the specific SL requirements '
         'defined in IEC 62443-3-3 for each zone. '
         'The assessment output identifies gaps between current SL-A (achieved) and SL-T (target) '
         'for prioritized remediation.'),

        ('How does QBITEL map to IEC 62443 zone boundaries?',
         'QBITEL passive discovery identifies and maps all assets to IEC 62443 zones automatically.',
         'QBITEL\'s passive discovery engine identifies all OT assets and their communication patterns. '
         'Using the IEC 62443 zone classification criteria - trust level, function, and connectivity - '
         'QBITEL proposes zone assignments for each asset. '
         'The proposed zone map is reviewed by the security team and formalized as the IEC 62443 zone model. '
         'QBITEL then enforces conduit security policies at each zone boundary, '
         'blocking communications that do not conform to the defined conduit policies.'),

        ('What is the relationship between ISA-99 and IEC 62443?',
         'ISA-99 was the original ISA committee standard - it was adopted and renumbered as IEC 62443 internationally.',
         'ISA-99 was the original cybersecurity standard developed by the International Society of Automation (ISA) '
         'for industrial automation and control systems. '
         'The standard was adopted by the International Electrotechnical Commission (IEC) and renumbered as IEC 62443, '
         'with technical alignment between the two. '
         'References to ISA-99 and IEC 62443 in procurement documents and RFPs refer to the same framework. '
         'QBITEL documentation references IEC 62443 as the current authoritative designation.'),
    ]

    for q, ol, a in qa5:
        story.append(QABlock(q, ol, a))
        story.append(sp(6))

    story.append(PageBreak())

    # =========================================================
    # SECTION 6: LEGACY EQUIPMENT INTEGRATION
    # =========================================================
    story.append(SectionHeader('Section 6: Legacy Equipment Integration', '5 questions on 30-year PLCs, no firmware updates, backward compatibility'))
    story.append(sp(8))

    qa6 = [
        ('Can QBITEL protect PLCs that are 20-30 years old with no firmware update capability?',
         'Yes - QBITEL operates at the network layer and never requires firmware changes to OT devices.',
         'QBITEL\'s core design principle is that OT devices are untouchable. '
         'A PLC running firmware from 1995 with no support contract and no vendor update mechanism '
         'is protected by QBITEL in exactly the same way as a modern PLC. '
         'QBITEL intercepts and authenticates commands at the network layer before they reach the PLC. '
         'The PLC continues to operate exactly as before. '
         'No firmware, no software agents, no configuration changes are ever made to protected OT devices.'),

        ('What happens if a legacy PLC cannot process modern protocol extensions?',
         'QBITEL authentication is transparent to legacy PLCs - they receive standard protocol traffic.',
         'Legacy PLCs typically reject or malfunction when they receive protocol messages with non-standard extensions. '
         'QBITEL\'s authentication mechanism is completely transparent to the protected PLC: '
         'the PLC receives standard, unmodified Modbus or DNP3 packets. '
         'The ML-DSA-65 signature is carried in a separate, parallel authentication channel '
         'that the PLC never sees. '
         'Backward compatibility with any PLCs supporting standard Modbus, DNP3, or IEC 61850 is guaranteed.'),

        ('How does QBITEL handle serial Modbus RTU (RS-485) environments?',
         'QBITEL supports serial tap adapters for RS-485/RS-232 Modbus RTU environments.',
         'Many legacy OT environments still use serial Modbus RTU over RS-485 or RS-232 links '
         'rather than Ethernet-based Modbus TCP. '
         'QBITEL provides passive serial tap adapters that convert the serial traffic to Ethernet '
         'for analysis without affecting the serial link. '
         'Authentication for serial Modbus RTU is implemented at the Modbus master side '
         'where Ethernet connectivity exists, with the authentication policy enforced by the master.'),

        ('What is the risk of QBITEL triggering undefined behavior in legacy OT devices?',
         'Zero risk in passive mode - QBITEL never sends any traffic to OT devices in discovery phase.',
         'The concern about triggering undefined behavior in legacy OT devices is valid - '
         'active network scanners have caused PLC crashes and HMI failures in industrial environments. '
         'QBITEL\'s passive tap operates in receive-only mode: it physically cannot transmit to the OT network. '
         'Even in inline authentication mode, QBITEL only forwards packets that the device would have received anyway - '
         'it never generates new traffic directed at OT devices. '
         'The risk of triggering undefined behavior is zero by design.'),

        ('How does QBITEL integrate with existing OT vendor support contracts?',
         'QBITEL operates outside OT device boundaries - vendor support contracts are unaffected.',
         'OT vendors typically void support contracts if third-party software is installed on certified devices. '
         'QBITEL never requires any installation on OT-vendor equipment. '
         'It operates on separate QBITEL-managed appliances connected via passive network taps. '
         'OT vendor support contracts, certifications, and warranties are completely unaffected. '
         'QBITEL has signed technology partnership agreements with major OT vendors '
         'including Siemens, Schneider Electric, and GE confirming non-interference with their products.'),
    ]

    for q, ol, a in qa6:
        story.append(QABlock(q, ol, a))
        story.append(sp(6))

    story.append(PageBreak())

    # =========================================================
    # SECTION 7: AIR-GAP & SOVEREIGN DEPLOYMENT
    # =========================================================
    story.append(SectionHeader('Section 7: Air-Gap & Sovereign Deployment', '5 questions on no internet, Ollama, HSM, threat intel'))
    story.append(sp(8))

    qa7 = [
        ('How does QBITEL operate in a fully air-gapped environment with no internet?',
         'QBITEL is designed from the ground up for fully air-gapped, sovereign deployment.',
         'QBITEL\'s architecture assumes no internet connectivity. '
         'The AI inference engine runs on local Ollama, querying only local language models '
         'with no external API calls. '
         'Threat intelligence is delivered via encrypted removable media following a documented '
         'air-gap transfer procedure with cryptographic chain of custody verification. '
         'The HSM is on-premise hardware. '
         'QBITEL can operate indefinitely with zero internet connectivity - '
         'this is not a degraded mode but the primary supported deployment model.'),

        ('How does threat intelligence get updated in an air-gapped QBITEL deployment?',
         'Threat intel updates are delivered via encrypted removable media with chain-of-custody verification.',
         'QBITEL maintains a threat intelligence update service that produces signed, encrypted update packages '
         'on a scheduled basis (daily, weekly, or monthly based on classification). '
         'These packages are delivered to air-gapped facilities via encrypted USB or removable media '
         'following a documented transfer procedure: verification of the package signature, '
         'transfer through a data diode or one-way transfer mechanism, '
         'and import by the QBITEL administrator. '
         'The cryptographic chain of custody ensures no tampering during transfer.'),

        ('What is the Ollama integration and why is it significant for OT security?',
         'Ollama enables local AI inference for threat detection without any cloud dependency.',
         'Ollama is an open-source framework for running large language models locally on-premise hardware. '
         'QBITEL uses Ollama to run threat analysis models locally - correlating protocol anomalies, '
         'identifying attack patterns, and generating natural language alert summaries - '
         'without any data leaving the facility. '
         'For OT security, this is significant because it enables AI-powered threat detection '
         'in environments where cloud connectivity is prohibited by regulation, policy, or classification.'),

        ('What HSM does QBITEL support and how is key management handled in air-gapped environments?',
         'QBITEL supports Thales Luna and Entrust nShield HSMs with air-gapped key ceremony procedures.',
         'QBITEL integrates with FIPS 140-3 Level 3 Hardware Security Modules including '
         'Thales Luna Network HSM, Entrust nShield, and equivalents. '
         'For air-gapped deployments, QBITEL supports an air-gapped key ceremony procedure '
         'where cryptographic key material is generated and distributed using offline processes. '
         'Master keys never leave the HSM. '
         'Operational keys are generated by the HSM on-premise. '
         'Key rotation is performed via a scheduled ceremony with dual-person integrity controls.'),

        ('Can QBITEL be deployed in classified government OT environments?',
         'Yes - QBITEL supports deployment in classified environments with appropriate sanitization.',
         'QBITEL supports deployment in classified government OT environments including defense industrial base '
         'facilities, classified government infrastructure, and nuclear facilities. '
         'The deployment model is fully sovereign: no data leaves the facility, '
         'no external connectivity is required, and all components can be provided in '
         'government-sanitized configurations. '
         'QBITEL is actively pursuing FedRAMP authorization for government cloud deployments '
         'and Common Criteria evaluation for classified environments.'),
    ]

    for q, ol, a in qa7:
        story.append(QABlock(q, ol, a))
        story.append(sp(6))

    story.append(PageBreak())

    # =========================================================
    # SECTION 8: PROCUREMENT & OPERATIONS
    # =========================================================
    story.append(SectionHeader('Section 8: Procurement & Operations', '5 questions on change freeze, vendor approval, OEM support, procurement'))
    story.append(sp(8))

    qa8 = [
        ('What is the procurement process for QBITEL in a regulated utility?',
         'QBITEL provides all documentation required for utility NERC CIP vendor risk assessment and procurement.',
         'Regulated utilities must perform vendor risk assessments under NERC CIP CIP-013 supply chain risk management. '
         'QBITEL provides a complete vendor risk assessment package including: '
         'software bill of materials (SBOM), security vulnerability disclosure policy, '
         'SOC 2 Type II audit report, FIPS 140-3 certifications for cryptographic components, '
         'and incident response SLA documentation. '
         'QBITEL supports sole-source justification for PQC-specific capabilities '
         'where no equivalent alternatives exist.'),

        ('How does QBITEL fit into an existing OT change management process?',
         'QBITEL\'s phased deployment is designed to align with OT change management procedures.',
         'OT change management in critical infrastructure typically requires change request documentation, '
         'impact assessment, safety review, and multi-level approval. '
         'QBITEL provides pre-formatted change request documentation for each deployment phase, '
         'impact assessments validated against SCADA polling requirements, '
         'and a phased activation model where each zone is enabled separately with explicit operator approval. '
         'The passive discovery phase (Phase 1) is typically approved as a monitoring-only change '
         'with minimal review requirements.'),

        ('What ongoing operational requirements does QBITEL impose on OT staff?',
         'QBITEL is designed for OT operators - minimal additional operational burden, automated compliance reporting.',
         'QBITEL is designed for operational technology environments where security staff is often shared '
         'with engineering and operations functions. '
         'Day-to-day operational requirements include: reviewing daily SOC summary (15 minutes), '
         'responding to P1 alerts (immediate, procedure-driven), and monthly compliance report review. '
         'Annual activities include threat intelligence review, HSM key rotation ceremony, and deployment audit. '
         'QBITEL provides OT-specific training for operators and compliance staff as part of deployment.'),

        ('Does QBITEL require OEM (OT vendor) approval or coordination?',
         'No - QBITEL operates outside OT device boundaries and does not require OEM coordination.',
         'QBITEL operates exclusively on its own appliances and via passive network taps. '
         'It does not install software on OT-vendor equipment, modify device firmware, '
         'or change device configurations. '
         'As a result, OEM approval or coordination is not required. '
         'QBITEL provides documentation confirming non-interference for inclusion in OEM maintenance records '
         'as a courtesy, but it is not required to maintain OEM support status.'),

        ('What are the typical hardware requirements for a QBITEL deployment?',
         'QBITEL appliances are 1U rack-mount or DIN-rail, requiring only network tap access.',
         'A typical QBITEL deployment requires: one QBITEL network appliance per OT network segment '
         '(1U 19-inch rack mount), one passive network tap per switch (passive hardware), '
         'and one management server (virtual machine or dedicated hardware). '
         'For air-gapped deployments with an HSM: one FIPS 140-3 HSM appliance. '
         'Power requirements are standard 110/220V AC, with DC options for substation environments. '
         'Industrial DIN-rail form factor is available for substation and harsh environment deployments.'),
    ]

    for q, ol, a in qa8:
        story.append(QABlock(q, ol, a))
        story.append(sp(6))

    story.append(PageBreak())

    # =========================================================
    # SECTION 9: HARD OBJECTIONS
    # =========================================================
    story.append(SectionHeader('Section 9: Hard Objections', '7 common objections with structured responses'))
    story.append(sp(8))

    objections = [
        ('Our OT network is air-gapped - we don\'t need additional security',
         'Air-gap is necessary but not sufficient. Insider threats, USB attacks (Stuxnet), and supply chain '
         'compromise all bypass air-gaps. The most devastating OT attacks - Stuxnet, TRITON, Ukraine BlackEnergy - '
         'all occurred on air-gapped or near-air-gapped networks. '
         'Once inside the perimeter, an attacker can issue arbitrary Modbus or DNP3 commands to any device. '
         'QBITEL protects the OT protocols themselves, not just the perimeter.'),

        ('We can\'t risk any disruption to operations',
         'QBITEL starts passive-only - a read-only tap that physically cannot inject traffic. '
         'There is zero operational risk during the discovery phase. '
         'Active protection phases are staged to your change window schedule, '
         'with explicit operator approval at each activation step. '
         'You control the timeline. QBITEL has never caused an operational disruption in any deployment.'),

        ('Modbus is being replaced by OPC UA anyway - why invest in securing it?',
         'OPC UA migration takes 10-20 years in typical OT environments. '
         'Modbus will be running alongside OPC UA for decades - both in legacy devices '
         'and in new deployments where simplicity is preferred. '
         'The Colonial Pipeline attack happened on a modern network. '
         'Both Modbus and OPC UA need protection. QBITEL secures both simultaneously.'),

        ('Our OT vendor doesn\'t support third-party security tools',
         'QBITEL operates at the network layer - zero changes to OT vendor equipment or firmware are required. '
         'No software is installed on any OT-vendor device. '
         'Vendor support contracts, certifications, and warranties are completely unaffected. '
         'QBITEL has signed technology partnership agreements with Siemens, Schneider Electric, '
         'and GE confirming non-interference with their products.'),

        ('NERC CIP doesn\'t require encryption - we\'re already compliant',
         'NERC CIP CIP-005 requires protection of BES Cyber Systems and their communication paths. '
         'CIP-011 requires protection of BES Cyber System Information. '
         'FERC has signaled stricter encryption requirements are forthcoming as the quantum timeline advances. '
         'Organizations deploying PQC now will be ahead of mandatory requirements '
         'rather than scrambling to comply under deadline pressure.'),

        ('We use change freeze periods - we can\'t deploy during critical operations',
         'QBITEL\'s passive discovery phase has zero impact on operations and can run during any change freeze. '
         'Active protection phases are scheduled to your maintenance windows with explicit operator approval. '
         'Many customers complete passive discovery and gap analysis during change freeze periods, '
         'then schedule active protection rollout in the next maintenance window. '
         'The phased approach is specifically designed for this operational constraint.'),

        ('Our safety systems are certified to IEC 61508 - we can\'t touch them',
         'QBITEL never modifies safety-certified code, firmware, or configuration. '
         'Safety system boundaries are formally documented and QBITEL is configured '
         'to apply passive monitoring only to SIS segments. '
         'The Safety Officer reviews and signs off on the SIS boundary configuration '
         'before any active protection is enabled elsewhere. '
         'IEC 61508 certification is fully preserved. QBITEL is the only OT security platform '
         'with an explicit, documented SIS exclusion architecture.'),
    ]

    for obj, resp in objections:
        story.append(ObjectionBlock(obj, resp))
        story.append(sp(8))

    story.append(PageBreak())

    # =========================================================
    # SECTION 10: COMPETITIVE
    # =========================================================
    story.append(SectionHeader('Section 10: Competitive Differentiation', '8 questions on vs Claroty, Dragos, Nozomi, firewalls, and manual processes'))
    story.append(sp(8))

    qa10 = [
        ('How does QBITEL compare to Claroty for OT security?',
         'Claroty is detection-only; QBITEL adds PQC encryption, command authentication, and compliance automation.',
         'Claroty is a strong OT asset discovery and threat detection platform. '
         'However, Claroty identifies threats - it does not authenticate OT protocol commands or encrypt OT traffic. '
         'QBITEL extends Claroty\'s detection capability with active protection: '
         'ML-DSA-65 PLC command authentication, IEC 62351 GOOSE/SV authentication, '
         'and automated NERC CIP compliance evidence. '
         'QBITEL is complementary to Claroty in most deployments, not a replacement.'),

        ('How does QBITEL compare to Dragos for ICS threat intelligence?',
         'Dragos provides world-class threat intelligence; QBITEL adds protocol-level authentication Dragos cannot provide.',
         'Dragos is the industry leader in OT threat intelligence and threat hunting. '
         'Their intelligence on nation-state actors like ELECTRUM and XENOTIME is unmatched. '
         'However, Dragos is a detection and intelligence platform - it cannot cryptographically authenticate '
         'PLC commands or enforce post-quantum encryption on OT protocols. '
         'A deployment that combines Dragos threat intelligence with QBITEL\'s active protection '
         'provides the most comprehensive OT security posture available.'),

        ('How does QBITEL compare to Nozomi Networks?',
         'Nozomi is detection and visibility; QBITEL adds the active protection and compliance automation layer.',
         'Nozomi Networks provides strong OT asset visibility, vulnerability assessment, '
         'and threat detection through passive monitoring. '
         'Like Claroty and Dragos, Nozomi is a monitoring platform without active protocol authentication capabilities. '
         'QBITEL\'s value proposition in Nozomi deployments is adding post-quantum command authentication, '
         'IEC 62351 power system security, and automated NERC CIP/IEC 62443 compliance evidence '
         'on top of Nozomi\'s visibility foundation.'),

        ('How does QBITEL compare to industrial firewalls like Fortinet FortiGate-Rugged?',
         'Firewalls protect the perimeter; QBITEL secures OT protocols end-to-end within the OT network.',
         'Industrial firewalls like Fortinet FortiGate-Rugged or Cisco IR Series protect the IT/OT boundary. '
         'They are excellent at controlling what crosses the perimeter. '
         'However, they cannot authenticate Modbus commands between a SCADA server and a PLC '
         'on the same OT segment. An insider threat, a compromised engineer laptop, or a rogue device '
         'inside the firewall perimeter is invisible to perimeter-only security. '
         'QBITEL secures OT protocols end-to-end regardless of network topology.'),

        ('What is the total cost of QBITEL vs. a manual compliance approach?',
         'QBITEL typically saves 3-5 FTE in compliance work and pays for itself in the first avoided NERC CIP violation.',
         'A typical NERC CIP compliance program requires 3-5 dedicated compliance FTEs for evidence collection, '
         'audit preparation, and gap remediation. At average OT security compensation, '
         'this represents $450,000-$750,000 annually in personnel cost. '
         'QBITEL automates the majority of this work. '
         'A single avoided NERC CIP violation (maximum $1M/day) covers multiple years of QBITEL deployment cost. '
         'The ROI calculation is compelling even before considering breach prevention.'),

        ('Does QBITEL compete with Microsoft Defender for IoT?',
         'Microsoft Defender for IoT requires Azure connectivity; QBITEL is fully air-gapped and sovereign.',
         'Microsoft Defender for IoT (formerly CyberX) provides OT asset visibility and threat detection '
         'with strong integration into the Microsoft security ecosystem. '
         'For organizations using Azure Sentinel and Microsoft security tools, Defender for IoT is a viable option. '
         'However, Defender for IoT requires Azure connectivity, which creates NERC CIP CIP-005 '
         'Electronic Security Perimeter concerns for any OT data leaving the facility. '
         'QBITEL is the preferred choice for air-gapped, sovereign, or classified environments.'),

        ('What makes QBITEL unique in the OT security market?',
         'QBITEL is the only OT security platform combining PQC, passive-first deployment, and SIS-safe architecture.',
         'QBITEL occupies a unique position in the OT security market: '
         'it is the only platform that combines NIST-standardized post-quantum cryptography '
         'for OT protocol authentication, passive-first deployment with zero operational risk, '
         'explicit SIS safety exclusion architecture (IEC 61508 compatible), '
         'automated NERC CIP/IEC 62443 compliance evidence, '
         'and fully air-gapped sovereign deployment. '
         'No other platform delivers all five capabilities in a single integrated product.'),

        ('Why should we choose QBITEL over building our own OT security solution?',
         'Building PQC-capable OT security in-house takes 3-5 years and requires specialized expertise in both domains.',
         'Building a custom OT security solution requires deep expertise in both post-quantum cryptography '
         '(NIST PQC standards, HSM integration, key management) and OT protocol security '
         '(Modbus, DNP3, IEC 61850, safety systems). '
         'Recruiting and retaining this combination of expertise is extremely difficult. '
         'QBITEL has 40+ engineers with this exact specialization, has already navigated NIST standardization, '
         'and has production deployments providing continuous feedback. '
         'A custom build would take 3-5 years and cost significantly more than QBITEL licensing.'),
    ]

    for q, ol, a in qa10:
        story.append(QABlock(q, ol, a))
        story.append(sp(6))

    # Contact footer
    story.append(sp(12))
    body_style = ParagraphStyle('footer_body', fontName='Helvetica', fontSize=9,
                                textColor=DARK_TEXT, leading=14, spaceAfter=4)
    story.append(Paragraph('<b>Contact: enterprise@qbitel.com | bridge.qbitel.com</b>', body_style))
    story.append(Paragraph('For authorized recipients only. (c) 2026 QBITEL Technologies. All Rights Reserved.', body_style))

    doc.build(story)


if __name__ == '__main__':
    build_qa_doc('docs/brochures/QBITEL_CriticalInfra_Pitch_QA_Guide.pdf')
    print('PDF saved: docs/brochures/QBITEL_CriticalInfra_Pitch_QA_Guide.pdf')
