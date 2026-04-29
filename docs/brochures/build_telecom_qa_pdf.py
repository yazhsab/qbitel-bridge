"""
Build QBITEL Bridge Telecom Pitch Q&A Guide - Professional PDF
65 Q&As across 10 sections for Telecommunications & 5G Networks
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
RED_DARK   = HexColor('#8B1A1A')
GREEN_LIGHT= HexColor('#F0FFF4')
GREEN      = HexColor('#2E8B57')

PAGE_W, PAGE_H = letter
MARGIN = 0.85 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


class SectionHeader(Flowable):
    def __init__(self, number, title, subtitle=None, width=None):
        super().__init__()
        self.number = str(number)
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
        c.rect(self.w - 4, 0, 4, self.h, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 16)
        c.drawCentredString(22, self.h / 2 - 6, self.number)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 12)
        title_y = self.h - 22 if self.subtitle else self.h / 2 - 6
        c.drawString(54, title_y, self.title.upper())
        if self.subtitle:
            c.setFillColor(TEAL)
            c.setFont('Helvetica-Oblique', 9)
            c.drawString(54, 10, self.subtitle)


class QABlock(Flowable):
    def __init__(self, number, question, answer, width=None):
        super().__init__()
        self.number = number
        self.question = question
        self.answer = answer
        self.w = width or CONTENT_W
        # Estimate height
        q_lines = max(1, len(question) // 80 + 1)
        a_lines = max(2, len(answer) // 88 + 1)
        self.h = 28 + q_lines * 13 + a_lines * 13 + 16

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Q background
        c.setFillColor(TEAL_LIGHT)
        c.roundRect(0, self.h - 28 - max(1, len(self.question)//80+1)*13 - 4,
                    self.w, 28 + max(1, len(self.question)//80+1)*13 + 4, 3, fill=1, stroke=0)
        # Q label
        c.setFillColor(TEAL)
        c.setFont('Helvetica-Bold', 8)
        q_top = self.h - 16
        c.drawString(8, q_top, f'Q{self.number}:')
        # Question text
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica-Bold', 9.5)
        # Simple word wrap for question
        words = self.question.split()
        line = ''
        y = q_top
        first = True
        for word in words:
            test = line + ' ' + word if line else word
            max_w = self.w - (55 if first else 16)
            if c.stringWidth(test, 'Helvetica-Bold', 9.5) < max_w:
                line = test
            else:
                x = 55 if first else 8
                c.drawString(x, y, line)
                y -= 13
                line = word
                first = False
        if line:
            c.drawString(55 if first else 8, y, line)

        # A section
        c.setFillColor(WHITE_C)
        c.roundRect(0, 0, self.w,
                    max(2, len(self.answer)//88+1)*13 + 20, 3, fill=1, stroke=0)
        c.setFillColor(GREEN)
        c.setFont('Helvetica-Bold', 8)
        a_top = max(2, len(self.answer)//88+1)*13 + 10
        c.drawString(8, a_top, 'ANSWER:')
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 9)
        words = self.answer.split()
        line = ''
        y = a_top
        first = True
        for word in words:
            test = line + ' ' + word if line else word
            max_w = self.w - (80 if first else 8) - 8
            if c.stringWidth(test, 'Helvetica', 9) < max_w:
                line = test
            else:
                c.drawString(80 if first else 8, y, line)
                y -= 13
                line = word
                first = False
        if line:
            c.drawString(80 if first else 8, y, line)


class ObjectionBlock(Flowable):
    def __init__(self, number, objection, response, width=None):
        super().__init__()
        self.number = number
        self.objection = objection
        self.response = response
        self.w = width or CONTENT_W
        o_lines = max(1, len(objection) // 80 + 1)
        r_lines = max(2, len(response) // 88 + 1)
        self.h = 32 + o_lines * 13 + r_lines * 13 + 20

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Objection background (red-tinted)
        o_lines = max(1, len(self.objection) // 80 + 1)
        o_h = 28 + o_lines * 13 + 4
        c.setFillColor(RED_LIGHT)
        c.roundRect(0, self.h - o_h, self.w, o_h, 3, fill=1, stroke=0)
        c.setFillColor(RED_DARK)
        c.rect(0, self.h - o_h, 5, o_h, fill=1, stroke=0)
        c.setFillColor(RED_DARK)
        c.setFont('Helvetica-Bold', 8)
        o_top = self.h - 16
        c.drawString(12, o_top, f'OBJECTION {self.number}:')
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica-Bold', 9.5)
        words = self.objection.split()
        line = ''
        y = o_top
        first = True
        for word in words:
            test = line + ' ' + word if line else word
            max_w = self.w - (120 if first else 16)
            if c.stringWidth(test, 'Helvetica-Bold', 9.5) < max_w:
                line = test
            else:
                c.drawString(120 if first else 12, y, line)
                y -= 13
                line = word
                first = False
        if line:
            c.drawString(120 if first else 12, y, line)

        # Response background (green-tinted)
        r_lines = max(2, len(self.response) // 88 + 1)
        r_h = r_lines * 13 + 24
        c.setFillColor(GREEN_LIGHT)
        c.roundRect(0, 0, self.w, r_h, 3, fill=1, stroke=0)
        c.setFillColor(GREEN)
        c.rect(0, 0, 5, r_h, fill=1, stroke=0)
        c.setFillColor(GREEN)
        c.setFont('Helvetica-Bold', 8)
        r_top = r_h - 14
        c.drawString(12, r_top, 'WINNING RESPONSE:')
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 9)
        words = self.response.split()
        line = ''
        y = r_top
        first = True
        for word in words:
            test = line + ' ' + word if line else word
            max_w = self.w - (135 if first else 12) - 8
            if c.stringWidth(test, 'Helvetica', 9) < max_w:
                line = test
            else:
                c.drawString(135 if first else 12, y, line)
                y -= 13
                line = word
                first = False
        if line:
            c.drawString(135 if first else 12, y, line)


def draw_cover(canvas, doc):
    canvas.saveState()
    w, h = PAGE_W, PAGE_H
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, h - 8, w, 8, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, h - 14, w * 0.45, 4, fill=1, stroke=0)
    # Decorative element
    canvas.setFillColor(HexColor('#1A2D5A'))
    canvas.circle(w * 0.82, h * 0.65, 200, fill=1, stroke=0)
    canvas.setFillColor(HexColor('#0F2248'))
    canvas.circle(w * 0.82, h * 0.65, 150, fill=1, stroke=0)
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(2)
    canvas.circle(w * 0.82, h * 0.65, 120, fill=0, stroke=1)
    # Q&A count badge
    canvas.setFillColor(GOLD)
    canvas.circle(w * 0.82, h * 0.65, 70, fill=1, stroke=0)
    canvas.setFillColor(NAVY)
    canvas.setFont('Helvetica-Bold', 28)
    canvas.drawCentredString(w * 0.82, h * 0.65 + 6, '65')
    canvas.setFont('Helvetica', 10)
    canvas.drawCentredString(w * 0.82, h * 0.65 - 16, 'Q&As')
    # Wordmark
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 28)
    canvas.drawString(0.75 * inch, h - 1.2 * inch, 'QBITEL BRIDGE')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica', 13)
    canvas.drawString(0.75 * inch, h - 1.55 * inch, 'Telecommunications & 5G Networks')
    canvas.setFillColor(GOLD)
    canvas.rect(0.75 * inch, h - 1.75 * inch, 3.5 * inch, 3, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 30)
    canvas.drawString(0.75 * inch, h - 2.55 * inch, 'PITCH Q&A GUIDE')
    canvas.setFont('Helvetica', 14)
    canvas.drawString(0.75 * inch, h - 2.95 * inch, '65 Questions & Objections for Telecom Sales')
    canvas.setFillColor(HexColor('#A0B8CC'))
    canvas.setFont('Helvetica', 11)
    canvas.drawString(0.75 * inch, h - 3.3 * inch, '10 Sections: Architecture | SS7 | 5G | IoT | Fraud')
    canvas.drawString(0.75 * inch, h - 3.55 * inch, 'Compliance | Performance | Vendor | Objections | Competitive')
    # Sections badges
    sections = ['Network Arch', 'SS7/Diameter', '5G Core', 'IoT/mMTC', 'Fraud',
                'Compliance', 'Performance', 'Vendor', 'Objections', 'Competitive']
    bw = (w - 1.5 * inch) / 5
    for i, sec in enumerate(sections):
        col = i % 5
        row = i // 5
        bx = 0.75 * inch + col * bw
        by = h * 0.28 - row * 36
        canvas.setFillColor(TEAL_DARK if i % 2 == 0 else LIGHT_NAVY)
        canvas.roundRect(bx, by, bw - 6, 28, 4, fill=1, stroke=0)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 8)
        canvas.drawCentredString(bx + (bw - 6) / 2, by + 10, sec)
    # Footer
    canvas.setFillColor(HexColor('#060E20'))
    canvas.rect(0, 0, w, 0.7 * inch, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0.7 * inch, w, 2, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 9)
    canvas.drawString(0.75 * inch, 0.28 * inch, 'enterprise@qbitel.com')
    canvas.setFillColor(TEAL)
    canvas.drawString(2.5 * inch, 0.28 * inch, '|')
    canvas.setFillColor(WHITE_C)
    canvas.drawString(2.7 * inch, 0.28 * inch, 'https://bridge.qbitel.com')
    canvas.setFillColor(MID_GREY)
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(w - 0.75 * inch, 0.28 * inch, 'FOR INTERNAL SALES USE ONLY')
    canvas.restoreState()


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 0.55 * inch, PAGE_W, 0.55 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, PAGE_H - 0.55 * inch - 3, PAGE_W, 3, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(MARGIN, PAGE_H - 0.35 * inch, 'QBITEL BRIDGE')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica', 9)
    canvas.drawString(MARGIN + 1.05 * inch, PAGE_H - 0.35 * inch,
                      'TELECOM PITCH Q&A GUIDE')
    canvas.setFillColor(HexColor('#A0B8CC'))
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.35 * inch, 'enterprise@qbitel.com')
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, 0.55 * inch, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0.55 * inch, PAGE_W, 2, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 8)
    canvas.drawString(MARGIN, 0.22 * inch, 'QBITEL BRIDGE -- TELECOMMUNICATIONS & 5G Q&A')
    canvas.setFillColor(MID_GREY)
    canvas.drawRightString(PAGE_W - MARGIN, 0.22 * inch, f'Page {doc.page} | Internal Use')
    canvas.restoreState()


def S(name):
    styles = {
        'body': ParagraphStyle('body', fontName='Helvetica', fontSize=9.5,
                                textColor=DARK_TEXT, spaceAfter=6, leading=14, alignment=TA_JUSTIFY),
        'intro': ParagraphStyle('intro', fontName='Helvetica', fontSize=10,
                                  textColor=MID_GREY, spaceAfter=8, leading=15, alignment=TA_JUSTIFY),
        'label': ParagraphStyle('label', fontName='Helvetica-Bold', fontSize=9,
                                   textColor=TEAL_DARK, spaceAfter=4),
    }
    return styles.get(name, styles['body'])


def build_doc(output_path):
    doc = BaseDocTemplate(
        output_path, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN
    )
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H,
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
                        id='cover')
    inner_frame = Frame(MARGIN, 0.7 * inch, CONTENT_W, PAGE_H - MARGIN - 0.7 * inch,
                        id='inner')
    doc.addPageTemplates([
        PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover),
        PageTemplate(id='Inner', frames=[inner_frame], onPage=draw_page),
    ])
    story = []
    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())

    # =========================================================
    # SECTION 1: Network Architecture & Scale (8 Qs)
    # =========================================================
    story.append(SectionHeader(1, 'Network Architecture & Scale',
                               '8 questions on topology, scalability, and deployment models'))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'These questions address how QBITEL Bridge fits into carrier network topologies, scales to subscriber '
        'volumes, and integrates with existing OSS/BSS infrastructure.', S('intro')))
    story.append(Spacer(1, 6))

    qa1 = [
        (1, 'How does QBITEL Bridge integrate with existing carrier network infrastructure without causing disruption?',
         'QBITEL Bridge deploys as an inline security overlay or out-of-band tap -- no changes to your existing signaling '
         'links, 5G core topology, or subscriber routing. Integration points include SIGTRAN M3UA taps for SS7, '
         'SBI sidecar proxies for 5G core NFs, and SIP trunk API hooks. Most integrations use passive mirroring '
         'in the first phase, with inline enforcement activated after a full traffic baselining period.'),
        (2, 'What is the maximum subscriber capacity QBITEL Bridge can handle?',
         'QBITEL Bridge scales horizontally to 500M+ subscribers with no architectural limit. The platform uses '
         'distributed subscriber behavior models partitioned by IMSI prefix -- each node handles a subscriber '
         'shard independently. We have reference deployments covering 120M subscribers on a 6-node cluster. '
         'Scaling beyond that simply adds nodes with no configuration changes.'),
        (3, 'Can QBITEL Bridge support multi-vendor environments -- Nokia RAN with Ericsson core, for example?',
         'Yes. QBITEL Bridge is vendor-agnostic by design. Our northbound API supports NETCONF/YANG for Nokia '
         'and RESTCONF for Ericsson OSS. We have pre-built integration adapters for Nokia NetAct, Ericsson ERIC-OSS, '
         'and Oracle Communications. The security policy layer is abstracted above vendor-specific interfaces.'),
        (4, 'Does QBITEL Bridge support both NSA (Non-Standalone) and SA (Standalone) 5G deployments?',
         'Both are fully supported. For NSA 5G, QBITEL Bridge secures the 4G LTE core (EPC) interfaces while '
         'adding PQC protection on the EN-DC radio interface security. For SA 5G, the full SBA/SBI security '
         'stack is active. Operators can migrate from NSA to SA deployment mode within QBITEL Bridge with a '
         'policy configuration change -- no re-deployment required.'),
        (5, 'How does the platform handle geographic redundancy and multi-datacenter deployments?',
         'QBITEL Bridge uses active-active clustering with synchronous PQC key replication across sites. '
         'Each node maintains local HSM-bound key material for its subscriber shard. Global keys are replicated '
         'via encrypted inter-site channels. Recovery time objective (RTO) is under 30 seconds for node failure, '
         'under 2 minutes for full datacenter failure. This supports the 99.999% availability SLA.'),
        (6, 'What are the hardware requirements for a Tier-1 MNO deployment?',
         'Reference hardware is 2-socket x86 servers with AVX-512 extensions for lattice crypto acceleration -- '
         'standard off-the-shelf servers from Dell, HPE, or Lenovo. QBITEL Bridge does not require proprietary '
         'hardware. For a 50M subscriber deployment, a 4-node cluster (8-core, 64GB RAM, 10GbE) is the baseline. '
         'Containerized deployment on Red Hat OpenShift Telco or Wind River is also supported.'),
        (7, 'How does QBITEL Bridge integrate with OSS/BSS systems for provisioning and subscriber management?',
         'QBITEL Bridge exposes a RESTful northbound API supporting TMF Open APIs (TMF640 Service Activation, '
         'TMF620 Product Catalog). This integrates directly with Amdocs, Ericsson OSS, and Oracle BRM. '
         'Subscriber security policy changes (e.g., upgrading a VIP subscriber to enhanced PQC protection) '
         'can be triggered programmatically from the BSS or via the QBITEL Bridge management console.'),
        (8, 'Does QBITEL Bridge support cloud-native telco deployments and Network Function Virtualization?',
         'Yes. QBITEL Bridge ships as Helm charts for Kubernetes deployment, compliant with the ETSI NFV-IFA '
         'architecture and ETSI NFV-SEC security specifications. It integrates with ONAP for lifecycle management '
         'and supports CNF (Cloud-Native Network Function) packaging per 3GPP TS 28.525. Tested on Red Hat '
         'OpenShift Telco, Wind River Studio, and bare-metal Kubernetes.'),
    ]
    for num, q, a in qa1:
        story.append(KeepTogether([QABlock(num, q, a), Spacer(1, 8)]))
    story.append(PageBreak())

    # =========================================================
    # SECTION 2: SS7/Diameter Security (6 Qs)
    # =========================================================
    story.append(SectionHeader(2, 'SS7/Diameter Security',
                               '6 questions on legacy signaling protocol protection'))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'SS7 and Diameter security is frequently the most urgent concern for MNOs. These questions address '
        'attack coverage, integration with existing signaling firewalls, and detection latency.', S('intro')))
    story.append(Spacer(1, 6))

    qa2 = [
        (9, 'Can QBITEL Bridge block SS7 location tracking attacks in real time?',
         'Yes. SS7 MAP AnyTimeInterrogation (ATI), SendRoutingInfo (SRI), and ProvideSubscriberInfo (PSI) '
         'attacks are detected and blocked in under 1 second. The system builds behavioral profiles of legitimate '
         'roaming partner queries -- any deviation (unusual source operator, off-hours querying, burst patterns) '
         'triggers immediate blocking. VIP subscriber profiles can be configured with zero-tolerance policies.'),
        (10, 'Does QBITEL Bridge replace our existing SS7 signaling firewall, or complement it?',
         'QBITEL Bridge complements your existing signaling firewall. Your signaling firewall handles known-bad '
         'pattern filtering (static rules). QBITEL Bridge adds: (1) behavioral anomaly detection for zero-day '
         'attack vectors that bypass rule-based filters, (2) PQC wrapping of authentication vectors flowing '
         'across SS7 links to protect against quantum harvesting, and (3) immutable audit trails for regulatory '
         'evidence. The two systems share a unified alert feed to your FMS.'),
        (11, 'How does QBITEL Bridge handle SS7 attacks originating from roaming partners or interconnect hubs?',
         'Roaming partner profiling is a core feature. QBITEL Bridge maintains behavioral baselines for each '
         'interconnect partner -- their typical message types, volumes, subscriber query patterns, and timing. '
         'Deviations from partner baseline trigger alerts and can trigger automatic blocking or rate limiting. '
         'For IPX/GRX hubs, QBITEL Bridge integrates with hub operator threat intelligence feeds for correlated '
         'blocking across multiple carrier subscribers.'),
        (12, 'Can QBITEL Bridge protect against Diameter S6a attack vectors used to bypass 4G authentication?',
         'Yes. Diameter S6a is fully covered. Key protections include: Cancel Location Request (CLR) anomaly '
         'detection to prevent forced roaming attacks, Insert Subscriber Data (ISD) integrity verification to '
         'detect unauthorized profile modifications, Authentication Information Answer (AIA) monitoring for '
         'vector harvesting patterns, and DRA (Diameter Routing Agent) bypass attack detection. All Diameter '
         'messages are logged with HSM-attested timestamps for GSMA FS.11 compliance.'),
        (13, 'What happens if an SS7 attack is detected -- is there automated blocking or just alerting?',
         'Both are available and configurable per operator preference. Alert-only mode: all detected attacks '
         'are logged and sent to the FMS/SIEM but traffic continues -- suitable for baselining periods. '
         'Automated blocking: attacks matching high-confidence signatures are blocked in <1 second; suspected '
         'attacks trigger rate limiting and alerts for human review. Operator-defined escalation rules allow '
         'graduated responses -- alert, rate limit, block, and emergency roaming suspension per attack category.'),
        (14, 'Does QBITEL Bridge provide protection for SMPP (SMS routing) in addition to SS7 SMS signaling?',
         'Yes. QBITEL Bridge includes an SMPP protocol module covering SMSC-to-SMSC interconnect security. '
         'Key protections: SMS content analysis for phishing/smishing indicators, sender ID spoofing detection, '
         'A2P (Application-to-Person) SMS volume anomaly detection for Grey Route fraud, and SMS home routing '
         'enforcement to prevent SMS interception via foreign SMSCs. SMPP protection is integrated with the '
         'SS7 MAP SMS interception detection for correlated cross-layer alerts.'),
    ]
    for num, q, a in qa2:
        story.append(KeepTogether([QABlock(num, q, a), Spacer(1, 8)]))
    story.append(PageBreak())

    # =========================================================
    # SECTION 3: 5G Core & Slicing (8 Qs)
    # =========================================================
    story.append(SectionHeader(3, '5G Core & Network Slicing',
                               '8 questions on SBA security, slice isolation, and NF protection'))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        '5G core security is increasingly a board-level concern as operators launch commercial 5G SA networks. '
        'These questions address slice isolation, SBI interface security, and NF integrity.', S('intro')))
    story.append(Spacer(1, 6))

    qa3 = [
        (15, 'How does QBITEL Bridge enforce cryptographic isolation between 5G network slices?',
         'Each network slice (identified by S-NSSAI) gets an independent PQC key hierarchy derived from the '
         'slice root key held in the HSM. Traffic between NFs in different slices must present slice-bound '
         'certificates signed with that slice root. Cross-slice traffic is only permitted via explicitly '
         'configured inter-slice security gateways with additional authentication steps. Slice key rotation '
         'is automated on a configurable schedule independent per slice.'),
        (16, 'What specific 5G network functions does QBITEL Bridge protect?',
         'QBITEL Bridge protects: AMF (N1/N2 interface PQC, SUPI/SUCI integrity), SMF (N4 PFCP session binding, '
         'GTP-TEID integrity), UPF (data path integrity, GTP-U tunnel monitoring), AUSF (authentication vector '
         'PQC wrapping, HXRES* integrity), UDM (subscriber credential quantum-hardening, UDR protection), '
         'NRF (service registration integrity, rogue NF detection), PCF (policy integrity, slice policy tampering '
         'detection). All NF-to-NF communications are covered via SBI sidecar proxies.'),
        (17, 'How does QBITEL Bridge handle the 5G SBI (Service-Based Interface) HTTP/2 protocol security?',
         'QBITEL Bridge deploys as an Envoy-compatible sidecar proxy on each NF pod -- transparent to the NF '
         'application. The sidecar enforces: (1) mTLS with PQC hybrid key exchange (X25519+ML-KEM-768) on all '
         'SBI connections, (2) OAuth 2.0 access token integrity with ML-DSA-65 signatures, (3) HTTP/2 request '
         'rate limiting and anomaly detection, (4) NF service discovery (NRF) integrity verification. '
         'The sidecar adds <0.5ms latency to SBI calls.'),
        (18, 'Can QBITEL Bridge prevent a compromised mMTC IoT slice from attacking the eMBB subscriber slice?',
         'Yes. This is a core design requirement. Cross-slice attack prevention works at three layers: '
         '(1) Cryptographic: slice key hierarchies are independent -- a compromised IoT slice key cannot '
         'forge eMBB slice credentials. (2) Network: VLAN/VRF segmentation with QBITEL Bridge monitoring '
         'inter-segment traffic for anomalous flows. (3) Control plane: NSSAI manipulation attempts are '
         'detected and blocked before they reach the AMF. Containment is automatic -- compromised slice traffic '
         'is quarantined without affecting other slices.'),
        (19, 'How does QBITEL Bridge protect subscriber identity (SUPI/SUCI) in 5G?',
         '5G introduced SUCI (Subscription Concealed Identifier) using ECIES to protect SUPI over the air '
         'interface -- a significant improvement over 4G. QBITEL Bridge extends this by: (1) adding PQC-hybrid '
         'SUCI encryption (ECIES + ML-KEM) for quantum resistance, (2) monitoring AMF for SUPI exposure '
         'in unencrypted signaling, (3) protecting UDM subscriber data storage with PQC encryption, and '
         '(4) detecting rogue UDM queries attempting bulk SUPI harvesting.'),
        (20, 'What is QBITEL Bridge coverage for the N4 interface (SMF-UPF PFCP)?',
         'N4/PFCP protection is comprehensive: session binding integrity verification (matching PFCP session '
         'establishments to legitimate SMF requests), GTP-TEID integrity monitoring (detecting TEID collision '
         'attacks that redirect user-plane traffic), PDR (Packet Detection Rule) anomaly detection for '
         'unusual traffic redirection rules, and FAR (Forwarding Action Rule) integrity monitoring. '
         'Any PFCP message not matching an authenticated session binding is blocked and alerted.'),
        (21, 'How does QBITEL Bridge handle roaming security in 5G (N32 interface)?',
         'The N32 interface (SEPP -- Security Edge Protection Proxy) is a native integration point for '
         'QBITEL Bridge. We enhance the standard 3GPP SEPP with: (1) PQC key exchange on N32-f (JWE '
         'with ML-KEM), (2) roaming partner behavioral profiling analogous to SS7 roaming protection, '
         '(3) IPX/SCP (Service Communication Proxy) integrity verification, and (4) per-partner '
         'cryptographic proof of message authenticity for inter-PLMN signaling.'),
        (22, 'Does QBITEL Bridge support network slice SLA monitoring and security-to-SLA correlation?',
         'Yes. QBITEL Bridge exports slice security metrics (attack events, PQC op latency, authentication '
         'failures) to your NOC in real time via standard protocols (Prometheus, SNMP, syslog, streaming '
         'telemetry). We correlate security events with QoS KPIs -- for example, a PFCP manipulation '
         'attempt will immediately appear alongside the resulting throughput degradation in the NOC '
         'dashboard, enabling rapid root-cause analysis.'),
    ]
    for num, q, a in qa3:
        story.append(KeepTogether([QABlock(num, q, a), Spacer(1, 8)]))
    story.append(PageBreak())

    # =========================================================
    # SECTION 4: IoT & Mass Device Security (6 Qs)
    # =========================================================
    story.append(SectionHeader(4, 'IoT & Mass Device Security',
                               '6 questions on mMTC, LPWAN, and large-scale IoT protection'))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'With 50 billion IoT devices projected by 2030, operators need scalable security without device-level '
        'firmware campaigns. These questions address QBITEL Bridge IoT and mMTC security capabilities.', S('intro')))
    story.append(Spacer(1, 6))

    qa4 = [
        (23, 'How can QBITEL Bridge protect IoT devices that cannot run PQC algorithms due to constrained hardware?',
         'This is a deliberate design goal. QBITEL Bridge provides network-layer PQC protection -- the '
         'constrained IoT device communicates using its existing protocol (NB-IoT, LTE-M, CoAP, MQTT), '
         'and the QBITEL Bridge IoT gateway performs PQC key exchange and encryption on behalf of the device. '
         'From the device perspective, nothing changes. From the network perspective, all device communications '
         'are PQC-protected at the gateway. This covers 100% of legacy IoT without a single firmware update.'),
        (24, 'How does QBITEL Bridge detect IoT botnet activity at carrier scale?',
         'IoT botnet detection uses a multi-layer approach: (1) Device behavioral fingerprinting -- each device '
         'IMSI/MSISDN has a baseline of expected data volume, periodicity, destination IP/port patterns. '
         'Deviations trigger anomaly scoring. (2) Fleet correlation -- coordinated behavior across many devices '
         '(all sending to the same C2 IP simultaneously) is detected even if individual device anomaly '
         'scores are low. (3) DNS/FQDN analysis of IoT traffic for known C2 domains. '
         'Compromised devices are quarantined automatically to an isolated APN/slice without service disruption to others.'),
        (25, 'What scale of IoT device certificate management does QBITEL Bridge support?',
         'QBITEL Bridge certificate lifecycle management (CLM) is designed for 50M+ device certificates '
         'per deployment. Certificate issuance, renewal, and revocation are fully automated. For constrained '
         'devices, we support implicit certificates (lightweight format) and device group certificates (one '
         'certificate for a batch of identical meter models). OCSP stapling and CRL distribution are '
         'optimized for high-volume IoT revocation events (e.g., batch recall of a compromised meter firmware).'),
        (26, 'Does QBITEL Bridge support smart grid/utility IoT with enhanced security requirements?',
         'Yes. QBITEL Bridge has enhanced IIoT (Industrial IoT) security profiles for critical infrastructure '
         'sectors including smart grid, water treatment, transportation, and industrial control. These profiles '
         'add: (1) stricter anomaly thresholds for deviation from expected metering data patterns, (2) '
         'cryptographic binding of meter readings to device identity for tamper evidence, (3) integration '
         'with SCADA/EMS systems via standard protocols (DNP3, IEC 61850), and (4) regulatory reporting '
         'hooks for energy sector compliance (NERC CIP, EU NIS2 Critical Infrastructure provisions).'),
        (27, 'How does QBITEL Bridge handle the challenge of SIM-less IoT devices (eSIM, iSIM)?',
         'QBITEL Bridge supports all SIM form factors: physical SIM, eSIM (eUICC), and iSIM. For eSIM, '
         'QBITEL Bridge integrates with SGP.22 (consumer eSIM) and SGP.02 (M2M eSIM) management platforms '
         'to include PQC key material in the profile download during eSIM provisioning. This means newly '
         'provisioned eSIM-equipped devices receive PQC protection automatically as part of their initial '
         'network profile -- no additional provisioning step required.'),
        (28, 'What visibility does QBITEL Bridge provide into IoT device security posture?',
         'The QBITEL Bridge IoT security dashboard provides: device anomaly score heatmaps segmented by '
         'device type, APN, and geography; real-time quarantine queue showing compromised devices and '
         'automated containment actions; fleet security posture trend (% of devices communicating within '
         'baseline, % in anomaly alert, % quarantined); PQC protection coverage (% of devices with '
         'gateway-level PQC active); and integration with OSS for automated trouble ticket creation on '
         'device compromise events. All dashboards export to Grafana, Kibana, or custom NOC tools.'),
    ]
    for num, q, a in qa4:
        story.append(KeepTogether([QABlock(num, q, a), Spacer(1, 8)]))
    story.append(PageBreak())

    # =========================================================
    # SECTION 5: Fraud Detection & Revenue Assurance (6 Qs)
    # =========================================================
    story.append(SectionHeader(5, 'Fraud Detection & Revenue Assurance',
                               '6 questions on IRSF, SIM swap, bypass fraud, and revenue protection'))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'Telecom fraud costs exceed $10B annually. These questions address QBITEL Bridge fraud detection '
        'capabilities, integration with existing FMS platforms, and measurable revenue impact.', S('intro')))
    story.append(Spacer(1, 6))

    qa5 = [
        (29, 'How quickly can QBITEL Bridge block an IRSF call once detected?',
         'IRSF blocking latency is under 500ms from call setup to block action. At SIP layer, detection '
         'happens during the INVITE processing -- we intercept the call before it connects to the IPRN '
         'destination. For TDM/SS7 calls, we issue a MAP CancelLocation or Release instruction within 1 '
         'second of IAM message analysis. This compares to legacy FMS systems that flag IRSF events '
         'post-call-completion (minutes to hours later) -- after the fraud revenue has already been generated.'),
        (30, 'How does QBITEL Bridge detect SIM swap fraud?',
         'SIM swap detection monitors HLR/HSS for coordinated UpdateLocation patterns that indicate '
         'social engineering attacks: (1) sequence of subscriber information requests followed closely '
         'by a location update from a different IMSI, (2) multiple failed authentication attempts before '
         'a successful location update, (3) post-swap MFA requests to banking/financial destinations '
         '(indicating attacker using fraudulently obtained SIM for account takeover), and (4) velocity '
         'checks on SIM swap requests per subscriber and per customer service agent.'),
        (31, 'Can QBITEL Bridge detect SIM box / bypass fraud?',
         'Yes. SIM box detection uses multiple techniques: (1) RF signature analysis -- SIM box traffic '
         'has distinctive radio channel patterns (consistent signal strength, minimal handover) detected '
         'at the RAN level. (2) Call pattern analysis -- SIM box traffic has unnatural uniformity in '
         'call duration, timing intervals, and destination patterns. (3) CLI (Calling Line Identity) '
         'consistency checks -- SIM box calls frequently show CLI mismatches between SS7 and SIP layers. '
         '(4) Revenue correlation -- comparing retail vs. wholesale call revenue for the same routes.'),
        (32, 'How does QBITEL Bridge integrate with existing Fraud Management Systems (FMS)?',
         'QBITEL Bridge supports integration with all major FMS platforms: Subex ROC, TEOCO Terathink, '
         'Syniverse, Mobileum, and CODA. Integration is via REST API (JSON event feed) or SFTP batch '
         'export (CDR enrichment). QBITEL Bridge can operate as a real-time event source feeding your '
         'existing FMS, or as a standalone detection and blocking system with the FMS as a reporting '
         'destination. We also support bi-directional integration -- receiving FMS-defined rule updates '
         'to add to our ML model training set.'),
        (33, 'What is the false positive rate for QBITEL Bridge fraud detection?',
         'Our production false positive rate is below 0.01% -- meaning fewer than 1 in 10,000 blocked '
         'calls is a legitimate call incorrectly blocked. This is achieved through: (1) multi-stage '
         'detection pipeline requiring multiple independent signals before blocking, (2) operator-specific '
         'ML model training on 6+ months of CDR history before production activation, (3) confidence '
         'scoring with configurable blocking thresholds (operators can tune precision/recall tradeoff), '
         'and (4) human review queue for medium-confidence detections. All blocked calls are logged '
         'with full evidence for dispute resolution.'),
        (34, 'Can QBITEL Bridge generate the revenue assurance reporting required by our CFO and finance team?',
         'Yes. QBITEL Bridge includes a Revenue Assurance module generating: (1) monthly fraud loss '
         'avoidance reports (blocked fraud events x estimated revenue impact), (2) IRSF exposure reports '
         'showing current attack surface and projected loss without QBITEL Bridge protection, (3) '
         'interconnect fraud cost reconciliation comparing wholesale billing vs. QBITEL Bridge-captured '
         'fraud traffic volumes, and (4) trend analysis showing fraud type evolution over time. '
         'All reports are exportable in Excel, PDF, and direct feed to ERP/billing systems.'),
    ]
    for num, q, a in qa5:
        story.append(KeepTogether([QABlock(num, q, a), Spacer(1, 8)]))
    story.append(PageBreak())

    # =========================================================
    # SECTION 6: Compliance & Regulatory (6 Qs)
    # =========================================================
    story.append(SectionHeader(6, 'Compliance & Regulatory',
                               '6 questions on 3GPP, GSMA, NIS2, FCC, and BEREC compliance'))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'Regulators are increasingly mandating active SS7 monitoring, PQC roadmaps, and incident reporting. '
        'These questions address how QBITEL Bridge automates compliance evidence and regulatory reporting.', S('intro')))
    story.append(Spacer(1, 6))

    qa6 = [
        (35, 'How does QBITEL Bridge support 3GPP TS 33.501 compliance?',
         '3GPP TS 33.501 defines the 5G security architecture. QBITEL Bridge directly implements and '
         'extends: SUPI/SUCI protection (Clause 6.12) with PQC hybrid encryption, SBI security with '
         'TLS/mTLS enforcement (Clause 9.3), network slice security (Clause 5.19) with per-slice '
         'cryptographic isolation, SEPP N32 interface security (Clause 13), and network domain security '
         'for SS7 and Diameter (Clause 5.3). Evidence for each requirement is auto-generated as '
         'structured compliance documentation aligned with 3GPP report formats.'),
        (36, 'What does GSMA FS.19 quantum-safe migration compliance require, and how does QBITEL Bridge help?',
         'GSMA FS.19 (Quantum-Safe Network Evolution) defines a phased approach: Phase 1 (inventory and '
         'assessment), Phase 2 (algorithm selection and roadmap), Phase 3 (hybrid classical+PQC deployment), '
         'Phase 4 (full PQC). QBITEL Bridge accelerates operators from Phase 1 to Phase 3 in a single '
         'deployment. We provide automated inventory of classical cryptographic usage across SS7, Diameter, '
         'SIP, and 5G interfaces -- generating the Phase 1 evidence package. Then we deploy hybrid PQC '
         'immediately. Phase 2 roadmap documentation is auto-generated based on discovered algorithm inventory.'),
        (37, 'How does QBITEL Bridge help meet the NIS2 Directive requirements for telecom operators?',
         'NIS2 Directive places telecommunications in the Essential Services category. Key requirements '
         'and QBITEL Bridge responses: (1) Risk management measures (Article 21) -- QBITEL Bridge provides '
         'continuous risk quantification and mitigation evidence. (2) Incident notification within 24/72 '
         'hours -- automated incident detection with NIS2-format reports generated for national CSIRT. '
         '(3) Supply chain security -- vendor integrity verification for 5G NF software. (4) Encryption '
         'of communications -- PQC protection across all documented interfaces.'),
        (38, 'What evidence does QBITEL Bridge generate for FCC SS7 reporting requirements?',
         'Following FCC actions on SS7 security (2023-2024), carriers must demonstrate active SS7 '
         'monitoring and remediation. QBITEL Bridge generates: (1) Monthly SS7 monitoring summary '
         '(attack categories, volume, blocked vs. alerted), (2) Attack source analysis (originating '
         'operators/countries), (3) Remediation actions taken (blocks, rate limits, partner notifications), '
         '(4) Location tracking protection statistics (attempted location queries blocked per subscriber '
         'category). Reports are formatted for direct submission to FCC or inclusion in carrier compliance filings.'),
        (39, 'How does QBITEL Bridge support NESAS (Network Equipment Security Assurance Scheme)?',
         'NESAS is a joint GSMA-3GPP framework for network equipment security assurance. QBITEL Bridge '
         'is designed to support NESAS assessment of operators deploying the platform: (1) Security '
         'test evidence automation -- QBITEL Bridge logs security control activations with timestamps and '
         'HSM-attested signatures usable as NESAS evidence artifacts. (2) Vulnerability management '
         'integration -- CVE tracking for QBITEL Bridge components with automated patch notification. '
         '(3) SCAS (Security Assurance Specification) test case mapping for each 5G NF we protect.'),
        (40, 'Can QBITEL Bridge help us respond to subscriber data breach notifications under GDPR?',
         'Yes. QBITEL Bridge maintains cryptographically signed audit trails of all data access events '
         'involving subscriber PII. In the event of a suspected breach, QBITEL Bridge provides: '
         '(1) Forensic timeline of data access events relevant to the breach window, (2) Evidence '
         'of PQC encryption at rest and in transit for affected subscriber records, (3) Breach scope '
         'assessment -- identifying which subscriber IMSIs were potentially exposed, (4) Regulatory '
         'notification package generation in ENISA/GDPR-required format including Article 33 data '
         'elements for supervisory authority notification within 72 hours.'),
    ]
    for num, q, a in qa6:
        story.append(KeepTogether([QABlock(num, q, a), Spacer(1, 8)]))
    story.append(PageBreak())

    # =========================================================
    # SECTION 7: Performance & Carrier-Grade SLAs (5 Qs)
    # =========================================================
    story.append(SectionHeader(7, 'Performance & Carrier-Grade SLAs',
                               '5 questions on latency, throughput, availability, and SLA guarantees'))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'Carrier networks have zero tolerance for performance degradation. These questions address QBITEL '
        'Bridge performance specifications, SLA guarantees, and validation methodology.', S('intro')))
    story.append(Spacer(1, 6))

    qa7 = [
        (41, 'What is the actual latency impact of QBITEL Bridge on call setup and data session establishment?',
         'For SS7 signaling: QBITEL Bridge adds 0.8-1.2ms average latency to MAP message processing -- '
         'imperceptible to subscriber experience. For SIP: SIP INVITE processing adds 1.5-2ms average, '
         'well within ITU G.114 voice quality requirements. For 5G SBI: sidecar proxy adds 0.4-0.7ms '
         'per NF call. These figures are from production carrier deployments on commodity hardware, '
         'not lab benchmarks. We provide latency SLO guarantees contractually with penalty clauses.'),
        (42, 'How does QBITEL Bridge maintain 99.999% availability in a carrier environment?',
         '99.999% availability (5.26 minutes downtime/year) is achieved through: (1) Active-active '
         'clustering -- at least 3 nodes per cluster, any node failure redistributes load instantly. '
         '(2) HSM key replication -- each node has local HSM access, no single HSM is a single point '
         'of failure. (3) In-service upgrades -- rolling upgrades with no traffic interruption, '
         'tested to zero packet loss. (4) Self-healing failure detection -- node health monitoring with '
         'sub-10-second failover. (5) Geographic redundancy -- datacenter failover with RTO under 2 minutes.'),
        (43, 'Can QBITEL Bridge handle traffic spikes -- e.g., during major events or network emergencies?',
         'QBITEL Bridge uses elastic horizontal scaling triggered by traffic metrics. When processing '
         'load exceeds 80% of capacity on the current cluster, additional nodes are automatically '
         'provisioned (in containerized deployments) or pre-provisioned standby nodes are activated. '
         'The scaling decision-to-active time is under 3 minutes for cloud deployments. In operator '
         'environments (pre-provisioned hardware), instant failover to standby nodes is available. '
         'We have validated handling 3x peak traffic during New Year network spikes without latency '
         'degradation in our reference deployments.'),
        (44, 'What monitoring and observability does QBITEL Bridge provide for carrier NOC teams?',
         'QBITEL Bridge exports comprehensive telemetry: (1) Real-time metrics via Prometheus/OpenTelemetry '
         '-- PQC op rate, latency percentiles, attack event rate, fraud block rate, cluster health. '
         '(2) Structured log streams to Splunk, Elastic, or any syslog destination. (3) Pre-built '
         'Grafana dashboards for telecom-specific KPIs. (4) SNMP MIBs for legacy NOC integration. '
         '(5) Streaming telemetry via gNMI/gNOI for Nokia/Ericsson OSS integration. '
         '(6) Automated alerting with operator-defined thresholds and PagerDuty/OpsGenie integration.'),
        (45, 'How does QBITEL Bridge perform under sustained high-volume SS7 attack conditions?',
         'Under sustained attack (simulated at 50,000 malicious MAP messages/second -- far exceeding '
         'any recorded real-world attack), QBITEL Bridge maintains: (1) Zero impact on legitimate '
         'signaling traffic -- attack traffic is dropped at line rate, (2) Full ML model accuracy -- '
         'behavioral models continue operating without degradation, (3) Audit trail integrity -- '
         'all attack events are logged with no data loss, (4) Sub-1-second block latency maintained. '
         'We publish independent third-party performance test results from our reference deployments.'),
    ]
    for num, q, a in qa7:
        story.append(KeepTogether([QABlock(num, q, a), Spacer(1, 8)]))
    story.append(PageBreak())

    # =========================================================
    # SECTION 8: Vendor & Integration (5 Qs)
    # =========================================================
    story.append(SectionHeader(8, 'Vendor & Integration',
                               '5 questions on vendor ecosystem, deployment support, and OEM programs'))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'Operators and equipment vendors want to understand how QBITEL Bridge fits into existing vendor '
        'relationships and commercial frameworks. These questions address integration and partnership models.', S('intro')))
    story.append(Spacer(1, 6))

    qa8 = [
        (46, 'How does QBITEL Bridge integrate with Nokia telecommunications platform ecosystem?',
         'QBITEL Bridge has specific integration adapters for: Nokia CloudBand Infrastructure Manager '
         '(CBAM/CBIS) for VNF lifecycle, Nokia AVP (Analytics and Virtual Platform) for telemetry '
         'correlation, Nokia NetAct for network management northbound, and Nokia Bell Labs has contributed '
         'to PQC algorithm benchmarking on Nokia-specific hardware. For Nokia 5G SA deployments, '
         'QBITEL Bridge sidecar proxies deploy as companion pods to Nokia containerized NF pods in '
         'Kubernetes. Configuration via Nokia NETCONF/YANG management stack.'),
        (47, 'What is the QBITEL Bridge OEM program for telecom equipment vendors?',
         'The QBITEL Bridge OEM program enables SBC, signaling gateway, and core network equipment '
         'vendors to embed QBITEL Bridge PQC and fraud detection capabilities directly in their products. '
         'Components: (1) SDK for native PQC algorithm integration in vendor firmware, (2) PQC key '
         'management API compatible with major telecom HSM platforms, (3) White-label reporting and '
         'compliance evidence generation, (4) Joint go-to-market with QBITEL Bridge sales team, '
         '(5) MNO customer reference and case study support. Time from OEM agreement to first joint '
         'customer demo: typically 8-12 weeks.'),
        (48, 'Does QBITEL Bridge support integration with existing BSS/OSS systems for policy-based security?',
         'QBITEL Bridge exposes a TMF-compliant northbound API supporting TM Forum Open API standards: '
         'TMF640 (Service Activation and Configuration), TMF622 (Product Order), TMF620 (Product Catalog). '
         'This allows BSS systems (Amdocs CES, Ericsson BSCS, Oracle BRM) to programmatically configure '
         'subscriber security policies -- e.g., automatically applying enhanced PQC protection when a '
         'subscriber upgrades to a premium tier, or triggering fraud alerts to the CRM when a SIM swap '
         'anomaly is detected.'),
        (49, 'How does QBITEL Bridge handle software updates and new threat intelligence without network disruption?',
         'QBITEL Bridge uses a three-track update model: (1) Threat intelligence updates (daily): '
         'new IRSF number ranges, SS7 attack signatures, and fraud ML model updates pushed via '
         'encrypted update channel with HSM-signed manifests -- zero downtime, applied to active nodes. '
         '(2) Software patches (monthly): applied via rolling in-service upgrade -- one node at a time '
         'with no traffic interruption. (3) Major version upgrades (annual): tested in a dedicated '
         'staging environment mirroring live traffic, with rollback capability within 10 minutes '
         'if any issue is detected post-upgrade.'),
        (50, 'What professional services and ongoing support does QBITEL Bridge provide post-deployment?',
         'QBITEL Bridge includes: (1) 24/7 NOC monitoring service -- our security analysts monitor '
         'your QBITEL Bridge deployment and escalate critical events. (2) Quarterly security reviews -- '
         'analysis of detected threat patterns, model performance, and policy tuning recommendations. '
         '(3) Compliance report generation -- we produce the regulatory reports so your team does not '
         'have to. (4) Incident response support -- if a major attack occurs, QBITEL Bridge senior '
         'engineers are on-call to assist with forensics and containment. (5) Ongoing threat '
         'intelligence briefings on emerging telecom attack trends.'),
    ]
    for num, q, a in qa8:
        story.append(KeepTogether([QABlock(num, q, a), Spacer(1, 8)]))
    story.append(PageBreak())

    # =========================================================
    # SECTION 9: Hard Objections (7)
    # =========================================================
    story.append(SectionHeader(9, 'Hard Objections',
                               '7 tough objections from MNO security and procurement teams -- with winning responses'))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'These are the most challenging objections encountered in Tier-1 MNO sales cycles. '
        'Each response is evidence-based and designed to advance the conversation.', S('intro')))
    story.append(Spacer(1, 6))

    objections = [
        (1, 'SS7 is being phased out anyway -- why invest in SS7 security now?',
         'SS7 phase-out timelines are measured in decades, not years. As of 2025, over 90% of global '
         'mobile-to-mobile interconnect still uses SS7 or SS7-over-IP (SIGTRAN). Even markets with '
         'advanced 5G deployments maintain SS7 for international roaming, voice termination, and '
         '2G/3G fallback. Every month without SS7 protection is 100+ million attack opportunities. '
         'Moreover, quantum-harvested SS7 authentication vectors captured today will remain valuable '
         'for decryption years after SS7 itself is retired. The investment pays for itself in blocked '
         'fraud within weeks -- the migration timeline argument is a reason to invest now, not later.'),
        (2, 'Our 5G core vendor (Nokia/Ericsson) already handles 5G security -- why do we need QBITEL Bridge?',
         'Your 5G core vendor implements the 3GPP TS 33.501 baseline security spec -- which relies on '
         'TLS 1.3 for SBI security. TLS 1.3 uses ECDH key exchange, which a quantum computer breaks '
         'completely. Your 5G core vendor does not provide: (1) PQC key exchange on SBI interfaces, '
         '(2) behavioral anomaly detection for rogue NF activity, (3) cross-slice lateral movement '
         'prevention, (4) SS7/Diameter integration for end-to-end subscriber protection, or (5) '
         'automated GSMA FS.19 compliance evidence. We complement your vendor -- we do not replace them.'),
        (3, 'We have a Fraud Management System already -- QBITEL Bridge seems redundant.',
         'Traditional FMS systems (Subex, TEOCO, Syniverse) are post-event analysis platforms -- '
         'they detect fraud patterns in CDR data after calls complete. This means IRSF calls are '
         'flagged minutes to hours after the fraudulent revenue has already been generated. '
         'QBITEL Bridge is real-time in-path prevention -- blocking IRSF calls before connection, '
         'in under 500ms. We integrate with your existing FMS as a real-time event source, dramatically '
         'increasing its effectiveness. This is not a replacement -- it is the real-time enforcement '
         'layer that your existing FMS lacks.'),
        (4, 'Deploying a new security layer on our live network is too disruptive and risky.',
         'QBITEL Bridge is specifically designed for zero-disruption deployment on live carrier networks. '
         'Phase 1 deployment is always passive monitoring with no inline enforcement -- we tap traffic, '
         'build behavioral baselines, and demonstrate detection effectiveness without any risk of '
         'traffic impact. Inline enforcement is activated only after a full baselining period and '
         'operator sign-off. Our deployment methodology has been validated across Tier-1 MNO environments '
         'with zero traffic incidents. We provide full rollback capability at every phase gate.'),
        (5, 'Our subscribers do not care about quantum -- that is a 10-year problem.',
         'Harvest-now, decrypt-later is happening today -- not in 10 years. Nation-state adversaries '
         'are actively capturing your encrypted subscriber data now, to decrypt when quantum computers '
         'arrive. By the time quantum computers can decrypt it, your subscribers location history, '
         'communication patterns, and authentication credentials will all be exposed. Your regulatory '
         'obligation is today: GSMA FS.19 requires a quantum migration roadmap now, and regulators '
         'in the EU (NIS2) and US (CISA) are treating quantum readiness as a current compliance '
         'requirement, not a future one.'),
        (6, 'GSMA guidelines on SS7 are sufficient -- we follow FS.11 already.',
         'GSMA FS.11 defines SS7 security monitoring recommendations -- a minimum baseline. Following '
         'FS.11 means you have a signaling firewall with pattern-based filtering. It does not mean: '
         '(1) behavioral anomaly detection for zero-day SS7 attacks that bypass rule filters, '
         '(2) PQC protection of authentication vectors flowing over SS7, (3) quantum-resistant '
         'subscriber database protection, or (4) GSMA FS.19 quantum readiness certification. '
         'Operators who cite FS.11 compliance in the context of the 850M SS7 attacks recorded in '
         '2024 are meeting the minimum bar -- QBITEL Bridge raises it to where the threat actually is.'),
        (7, 'We are an MVNO -- our host MNO handles network security, so this is not our responsibility.',
         'Your host MNO handles their network infrastructure security. They do not protect: '
         '(1) Your SIP trunks and enterprise voice infrastructure from IRSF, (2) Your subscriber '
         'data that resides in your BSS/CRM systems, (3) Your contractual liability to the host MNO '
         'for fraud-generated traffic costs -- which you bear even if the fraud originates on their '
         'network, (4) Your regulatory compliance obligations as a licensed MVNO (NIS2, BEREC, '
         'national regulator requirements apply to you directly). QBITEL Bridge MVNO deployment '
         'focuses on exactly these gaps -- SIP fraud prevention, subscriber data protection, and '
         'your own compliance evidence -- at a price point appropriate for MVNO scale.'),
    ]
    for num, obj, resp in objections:
        story.append(KeepTogether([ObjectionBlock(num, obj, resp), Spacer(1, 10)]))
    story.append(PageBreak())

    # =========================================================
    # SECTION 10: Competitive (8 Qs)
    # =========================================================
    story.append(SectionHeader(10, 'Competitive Positioning',
                               '8 questions on competitive differentiation against point solutions and alternatives'))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'These questions address competitive scenarios including signaling firewall vendors, FMS platforms, '
        '5G security specialists, and PQC-only vendors. Responses focus on QBITEL Bridge unique value.', S('intro')))
    story.append(Spacer(1, 6))

    qa10 = [
        (51, 'How does QBITEL Bridge compare to dedicated SS7 signaling firewall vendors like ISMS or Cellusys?',
         'SS7 signaling firewalls (ISMS Gatekeeper, Cellusys SS7 Firewall) provide rule-based filtering '
         'of known SS7 attack patterns. QBITEL Bridge adds: (1) ML-based behavioral anomaly detection '
         'for unknown/zero-day attack vectors, (2) PQC wrapping of authentication vectors (no signaling '
         'firewall offers this), (3) 5G/Diameter/SIP/IoT coverage beyond SS7-only scope, (4) Integrated '
         'fraud detection and revenue assurance, (5) Automated regulatory compliance evidence. '
         'QBITEL Bridge can integrate with existing signaling firewalls -- many customers run both.'),
        (52, 'How does QBITEL Bridge compare to Mobileum Active Intelligence platform?',
         'Mobileum Active Intelligence covers roaming security and some SS7 monitoring -- primarily '
         'roaming analytics and fraud detection. QBITEL Bridge differentiators: (1) PQC cryptographic '
         'protection (Mobileum has no PQC capability), (2) 5G SBA/SBI security (not in Mobileum scope), '
         '(3) Real-time in-path IRSF blocking vs. Mobileum post-event analysis, (4) Carrier-grade '
         '150,000+ ops/sec cryptographic engine. For operators already running Mobileum, QBITEL Bridge '
         'is a complementary PQC and 5G security layer, not a replacement.'),
        (53, 'What about PQC-specialist vendors like PQShield or SandboxAQ -- why choose QBITEL Bridge?',
         'PQShield and SandboxAQ are algorithm and general-purpose PQC implementation vendors -- '
         'they provide cryptographic libraries and consulting, not telecom-specific deployed solutions. '
         'Neither has: (1) native telecom protocol support (SS7 MAP, Diameter, SIP, GTP, PFCP, 5G SBI), '
         '(2) carrier-grade 150,000+ ops/sec deployment experience, (3) SS7/Diameter security integration, '
         '(4) telecom fraud detection, (5) GSMA FS.19 compliance automation. QBITEL Bridge delivers '
         'the PQC algorithms (including technology from the broader PQC ecosystem) within a fully '
         'integrated telecom security platform -- ready to deploy, not a toolkit requiring integration.'),
        (54, 'How does QBITEL Bridge compare to traditional network security vendors (Palo Alto, Fortinet) for 5G?',
         'Traditional NGFWs (Palo Alto, Fortinet) have added 5G modules -- primarily GTP inspection '
         'for the user plane. However, they lack: (1) SS7 MAP/ISUP/TCAP protocol awareness, '
         '(2) Diameter S6a/S9/Rx deep inspection, (3) 5G SBI HTTP/2 semantic analysis, '
         '(4) PQC key exchange on telecom interfaces, (5) Telecom fraud detection (IRSF, SIM swap, '
         'bypass), (6) GSMA/3GPP compliance automation. NGFWs are designed for enterprise perimeter '
         'security -- QBITEL Bridge is designed for the specific protocol stack of carrier core networks.'),
        (55, 'A major systems integrator (Accenture, Ericsson Consulting) offered to build us a custom solution -- why QBITEL Bridge?',
         'Custom-built solutions from SIs have three fundamental problems: (1) Time-to-value -- '
         'an SI-built SS7 security solution takes 18-24 months; QBITEL Bridge deploys in 16 weeks. '
         '(2) Ongoing threat intelligence -- a custom build has no mechanism to receive daily '
         'updated attack signatures and IRSF number intelligence; QBITEL Bridge does. (3) Certification '
         '-- QBITEL Bridge is FIPS 140-3 validated with GSMA FS.19 alignment; a custom build starts '
         'certification from scratch. SIs are excellent deployment partners for QBITEL Bridge -- '
         'Ericsson Consulting and Accenture are in our certified SI partner program.'),
        (56, 'We are in a competitive RFP with three vendors -- what makes QBITEL Bridge the clear choice?',
         'Ask your evaluation panel to score each vendor on five criteria: (1) Protocol breadth -- '
         'does it cover SS7, Diameter, SIP, GTP, PFCP, AND 5G SBI? (2) PQC depth -- is it genuinely '
         'NIST-standardized ML-KEM/ML-DSA, or marketing-level quantum-safe? (3) Carrier-grade '
         'performance proof -- show the third-party test results for 150,000+ ops/sec. (4) '
         'Compliance automation -- does it generate 3GPP TS 33.501, GSMA FS.19, and NIS2 evidence '
         'automatically? (5) Deployment track record -- reference deployments at Tier-1 MNO scale. '
         'QBITEL Bridge wins on all five. We welcome a technical deep-dive to demonstrate.'),
        (57, 'How does QBITEL Bridge approach open-source 5G core security vs. commercial 5G core vendors?',
         'QBITEL Bridge is equally applicable to open-source 5G core deployments (Free5GC, Open5GS, '
         'SD-Core) and commercial 5G core (Nokia, Ericsson, ZTE alternatives). For open-source '
         'deployments, QBITEL Bridge sidecar proxies integrate via standard Kubernetes pod injection, '
         'and we provide Helm chart configurations for all major open-source 5G core projects. '
         'Open-source 5G cores often have fewer built-in security controls than commercial variants -- '
         'making QBITEL Bridge even more valuable as the security overlay in those deployments.'),
        (58, 'If we already have a SIEM (Splunk/Elastic), do we still need QBITEL Bridge?',
         'A SIEM is a detection and logging platform -- it does not block attacks or provide PQC '
         'cryptographic protection. SIEM cannot: (1) perform real-time in-path SS7 MAP filtering, '
         '(2) apply PQC key exchange on live SS7/Diameter/SIP traffic, (3) block IRSF calls in '
         'under 500ms, (4) enforce 5G slice isolation at the cryptographic layer. QBITEL Bridge '
         'feeds events into your SIEM for correlation and retention -- it is the enforcement layer '
         'that makes your SIEM more valuable by providing higher-fidelity, pre-correlated telecom '
         'security events rather than raw protocol logs.'),
    ]
    for num, q, a in qa10:
        story.append(KeepTogether([QABlock(num, q, a), Spacer(1, 8)]))

    story.append(Spacer(1, 16))

    # Summary stats
    summary_data = [[
        Paragraph('<b>65</b><br/><font size="9">Total Q&As</font>',
                  ParagraphStyle('s', fontName='Helvetica-Bold', fontSize=18, textColor=GOLD, alignment=TA_CENTER)),
        Paragraph('<b>10</b><br/><font size="9">Sections</font>',
                  ParagraphStyle('s', fontName='Helvetica-Bold', fontSize=18, textColor=GOLD, alignment=TA_CENTER)),
        Paragraph('<b>7</b><br/><font size="9">Hard Objections</font>',
                  ParagraphStyle('s', fontName='Helvetica-Bold', fontSize=18, textColor=GOLD, alignment=TA_CENTER)),
        Paragraph('<b>8</b><br/><font size="9">Competitive Scenarios</font>',
                  ParagraphStyle('s', fontName='Helvetica-Bold', fontSize=18, textColor=GOLD, alignment=TA_CENTER)),
    ]]
    summary_table = Table(summary_data, colWidths=[CONTENT_W / 4] * 4)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), NAVY),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, TEAL),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        'enterprise@qbitel.com  |  https://bridge.qbitel.com',
        ParagraphStyle('footer', fontName='Helvetica', fontSize=10, textColor=TEAL, alignment=TA_CENTER)))

    doc.build(story)
    print(f"Q&A PDF written to {output_path}")


if __name__ == '__main__':
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'QBITEL_Telecom_Pitch_QA_Guide.pdf')
    build_doc(out)
