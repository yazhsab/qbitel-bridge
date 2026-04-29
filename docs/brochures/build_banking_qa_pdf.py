"""
Build QBITEL Bridge Banking Pitch Q&A Guide - Professional PDF
65 Q&As across 10 sections for Banking & Financial Services
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
        c.rect(self.w - 4, 0, 4, self.h, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 16)
        nw = c.stringWidth(self.number, 'Helvetica-Bold', 16)
        c.drawString(22 - nw / 2, (self.h - 18) / 2 + 2, self.number)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 12)
        ty = self.h - 24 if self.subtitle else (self.h - 14) / 2 + 2
        c.drawString(54, ty, self.title.upper())
        if self.subtitle:
            c.setFillColor(TEAL)
            c.setFont('Helvetica-Oblique', 8.5)
            c.drawString(54, 9, self.subtitle)


class QABlock(Flowable):
    def __init__(self, number, question, answer, width=None):
        super().__init__()
        self.number = number
        self.question = question
        self.answer = answer
        self.w = width or CONTENT_W
        self._calc_height()

    def _calc_height(self):
        # Estimate: ~60 chars per line for answer at width
        chars_per_line = int(self.w / 5.5)
        q_lines = max(1, len(self.question) // chars_per_line + 1)
        a_lines = max(1, len(self.answer) // chars_per_line + 1)
        self.q_h = q_lines * 13 + 18
        self.a_h = a_lines * 13 + 18
        self.h = self.q_h + self.a_h + 4

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Q section
        c.setFillColor(TEAL_LIGHT)
        c.rect(0, self.a_h + 4, self.w, self.q_h, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(0, self.a_h + 4, 3, self.q_h, fill=1, stroke=0)
        # Q badge
        c.setFillColor(TEAL)
        c.roundRect(6, self.a_h + 4 + (self.q_h - 18) / 2, 26, 18, 4, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 8)
        badge = 'Q%s' % self.number
        bw = c.stringWidth(badge, 'Helvetica-Bold', 8)
        c.drawString(6 + (26 - bw) / 2, self.a_h + 4 + (self.q_h - 18) / 2 + 5, badge)
        # Question text
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica-Bold', 9)
        text_x = 40
        text_w = self.w - 50
        # Simple word-wrap
        words = self.question.split()
        line = ''
        lines_out = []
        for word in words:
            test = (line + ' ' + word).strip()
            if c.stringWidth(test, 'Helvetica-Bold', 9) < text_w:
                line = test
            else:
                if line:
                    lines_out.append(line)
                line = word
        if line:
            lines_out.append(line)
        y_start = self.a_h + 4 + self.q_h - 14
        for ln in lines_out:
            c.drawString(text_x, y_start, ln)
            y_start -= 13

        # A section
        c.setFillColor(WHITE_C)
        c.rect(0, 0, self.w, self.a_h + 4, fill=1, stroke=0)
        c.setFillColor(GOLD)
        c.rect(0, 0, 3, self.a_h + 4, fill=1, stroke=0)
        c.setFillColor(GOLD)
        c.roundRect(6, (self.a_h + 4 - 18) / 2, 26, 18, 4, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 8)
        c.drawString(14, (self.a_h + 4 - 18) / 2 + 5, 'A')
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 9)
        words = self.answer.split()
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
        y_start = self.a_h + 4 - 14
        for ln in lines_out:
            c.drawString(text_x, y_start, ln)
            y_start -= 13

        # Separator
        c.setFillColor(MID_GREY)
        c.rect(0, self.a_h + 2, self.w, 1, fill=1, stroke=0)


class ObjectionBlock(Flowable):
    def __init__(self, number, objection, reframe, width=None):
        super().__init__()
        self.number = number
        self.objection = objection
        self.reframe = reframe
        self.w = width or CONTENT_W
        self._calc_height()

    def _calc_height(self):
        chars_per_line = int(self.w / 5.5)
        obj_lines = max(1, len(self.objection) // chars_per_line + 1)
        ref_lines = max(1, len(self.reframe) // chars_per_line + 1)
        self.obj_h = obj_lines * 13 + 22
        self.ref_h = ref_lines * 13 + 22
        self.h = self.obj_h + self.ref_h + 4

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Objection (red-tinted)
        c.setFillColor(RED_LIGHT)
        c.rect(0, self.ref_h + 4, self.w, self.obj_h, fill=1, stroke=0)
        c.setFillColor(HexColor('#CC3333'))
        c.rect(0, self.ref_h + 4, 3, self.obj_h, fill=1, stroke=0)
        c.setFillColor(HexColor('#CC3333'))
        c.roundRect(6, self.ref_h + 4 + (self.obj_h - 18) / 2, 44, 18, 4, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 7)
        badge = 'OBJ %s' % self.number
        bw = c.stringWidth(badge, 'Helvetica-Bold', 7)
        c.drawString(6 + (44 - bw) / 2, self.ref_h + 4 + (self.obj_h - 18) / 2 + 5, badge)
        c.setFillColor(HexColor('#8B0000'))
        c.setFont('Helvetica-Bold', 9)
        text_x = 58
        text_w = self.w - 66
        words = self.objection.split()
        line = ''
        lines_out = []
        for word in words:
            test = (line + ' ' + word).strip()
            if c.stringWidth(test, 'Helvetica-Bold', 9) < text_w:
                line = test
            else:
                if line:
                    lines_out.append(line)
                line = word
        if line:
            lines_out.append(line)
        y_start = self.ref_h + 4 + self.obj_h - 14
        for ln in lines_out:
            c.drawString(text_x, y_start, ln)
            y_start -= 13

        # Reframe (green-tinted)
        c.setFillColor(GREEN_LIGHT)
        c.rect(0, 0, self.w, self.ref_h + 4, fill=1, stroke=0)
        c.setFillColor(HexColor('#2E8B57'))
        c.rect(0, 0, 3, self.ref_h + 4, fill=1, stroke=0)
        c.setFillColor(HexColor('#2E8B57'))
        c.roundRect(6, (self.ref_h + 4 - 18) / 2, 70, 18, 4, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 7)
        c.drawString(14, (self.ref_h + 4 - 18) / 2 + 5, 'YOUR RESPONSE')
        c.setFillColor(HexColor('#1A4A2E'))
        c.setFont('Helvetica', 9)
        words = self.reframe.split()
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
        y_start = self.ref_h + 4 - 14
        for ln in lines_out:
            c.drawString(text_x, y_start, ln)
            y_start -= 13

        c.setFillColor(MID_GREY)
        c.rect(0, self.ref_h + 2, self.w, 1, fill=1, stroke=0)


def draw_cover(canvas, doc):
    canvas.saveState()
    w, h = PAGE_W, PAGE_H
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, h * 0.55, w, h * 0.45, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    p = canvas.beginPath()
    p.moveTo(w * 0.55, h)
    p.lineTo(w, h)
    p.lineTo(w, h * 0.75)
    p.close()
    canvas.drawPath(p, fill=1, stroke=0)
    canvas.setFillColor(TEAL_DARK)
    canvas.rect(0, 0, w, 1.8 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, 1.8 * inch, w, 5, fill=1, stroke=0)

    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 28)
    canvas.drawString(MARGIN, h * 0.88, 'QBITEL BRIDGE')
    canvas.setFillColor(GOLD)
    canvas.setFont('Helvetica-Bold', 16)
    canvas.drawString(MARGIN, h * 0.83, 'Banking & Financial Services')
    canvas.setFillColor(GOLD)
    canvas.rect(MARGIN, h * 0.823, 3.5 * inch, 3, fill=1, stroke=0)

    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 34)
    canvas.drawString(MARGIN, h * 0.72, 'PITCH Q&A GUIDE')
    canvas.setFont('Helvetica-Bold', 18)
    canvas.drawString(MARGIN, h * 0.72 - 36, 'Banking & Financial Services')

    canvas.setFillColor(HexColor('#CCDDE8'))
    canvas.setFont('Helvetica-Oblique', 12)
    canvas.drawString(MARGIN, h * 0.72 - 66,
                      '65 Questions & Answers | 10 Sections | Hard Objection Handling')

    stats = [('65', 'Q&As'), ('10', 'Sections'), ('7', 'Hard Objections'), ('7', 'Competitive Qs')]
    box_w = (w - 2 * MARGIN) / 4
    box_h = 0.9 * inch
    box_y = 1.85 * inch
    for i, (val, label) in enumerate(stats):
        bx = MARGIN + i * box_w
        canvas.setFillColor(NAVY if i % 2 == 0 else LIGHT_NAVY)
        canvas.rect(bx, box_y, box_w - 4, box_h, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.setFont('Helvetica-Bold', 22)
        vw = canvas.stringWidth(val, 'Helvetica-Bold', 22)
        canvas.drawString(bx + (box_w - 4 - vw) / 2, box_y + box_h - 30, val)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 8)
        lw = canvas.stringWidth(label, 'Helvetica', 8)
        canvas.drawString(bx + (box_w - 4 - lw) / 2, box_y + box_h - 46, label)

    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 9)
    canvas.drawString(MARGIN, 0.7 * inch, 'enterprise@qbitel.com  |  https://bridge.qbitel.com')
    canvas.setFillColor(HexColor('#AABBCC'))
    canvas.setFont('Helvetica', 8)
    canvas.drawString(MARGIN, 0.5 * inch, 'Confidential - For Sales & Pre-Sales Use Only  |  (c) 2026 QBITEL.')
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
    canvas.drawString(MARGIN + 1.1 * inch, PAGE_H - 0.31 * inch, 'BANKING PITCH Q&A GUIDE')
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
                      'Confidential - Sales & Pre-Sales Only  |  (c) 2026 QBITEL. All Rights Reserved.')
    contact_str = 'enterprise@qbitel.com  |  bridge.qbitel.com'
    cw = canvas.stringWidth(contact_str, 'Helvetica', 7.5)
    canvas.drawString(PAGE_W - MARGIN - cw, 0.15 * inch, contact_str)
    canvas.restoreState()


def make_styles():
    styles = {}
    styles['body'] = ParagraphStyle('body', fontName='Helvetica', fontSize=9.5,
                                     leading=14, textColor=DARK_TEXT, spaceAfter=6)
    styles['intro'] = ParagraphStyle('intro', fontName='Helvetica-Oblique', fontSize=9.5,
                                      leading=14, textColor=MID_GREY, spaceAfter=8)
    return styles


def build_story(styles):
    S = []

    def sp(n=8):
        S.append(Spacer(1, n))

    def hdr(num, title, subtitle=None):
        S.append(SectionHeader(num, title, subtitle))
        sp(10)

    def qa(n, q, a):
        S.append(KeepTogether([QABlock(str(n), q, a), Spacer(1, 6)]))

    def obj(n, objection, response):
        S.append(KeepTogether([ObjectionBlock(str(n), objection, response), Spacer(1, 6)]))

    # Section 1: Business Case & ROI
    hdr('1', 'Business Case & ROI', '8 Questions — TCO, ROI timeline, quantum risk, board-level')
    S.append(Paragraph('Use these answers to build the business case with CFOs, COOs, and Board Risk Committees.', styles['intro']))
    sp(6)

    qa(1, 'What is the total cost of ownership for QBITEL Bridge in a Tier-1 bank environment?',
       'QBITEL Bridge is priced on a per-protocol-stream model with volume discounts for enterprise deployments. '
       'For a Tier-1 bank with 20+ protocol streams, annual licensing is typically $2-4M - compared to $50M average '
       'cost of a major payment breach and $6.2B in projected industry quantum losses by 2030. Most clients see '
       'positive ROI within the first year through reduced compliance overhead, breach risk reduction, and '
       'elimination of QSA findings. We provide a detailed TCO model during the proof-of-value engagement.')

    qa(2, 'What is the expected ROI timeline and how do you quantify it?',
       'Three ROI streams: (1) Compliance cost reduction - automated PCI-DSS 4.0 and DORA evidence cuts QSA '
       'fees and compliance labor by 40-60%. (2) Breach risk reduction - QBITEL Bridge eliminates the quantum '
       'harvest-now-decrypt-later attack vector worth $6.2B in projected industry exposure. (3) Operational '
       'efficiency - 78% autonomous incident response reduces SOC analyst hours by 30-40%. Typical clients '
       'calculate full ROI within 12-18 months. We model this specifically for your transaction volumes and '
       'compliance cost baseline during the scoping engagement.')

    qa(3, 'How far away is the quantum computing threat to banking systems, really?',
       'NIST, NSA, and GCHQ advisories consistently cite a 5-15 year window to cryptographically relevant '
       'quantum computers. However, the harvest-now-decrypt-later threat is active TODAY. Nation-state '
       'adversaries are confirmed to be storing encrypted SWIFT and wire transfer data now for future '
       'decryption. SWIFT regulatory archives must be kept for 5-7 years - meaning data encrypted today '
       'will be within quantum decryption range before those archives are retired. The risk is not future; '
       'the data collection is happening now.')

    qa(4, 'How do I present quantum risk to our Board of Directors?',
       'Frame it as a regulatory and operational risk category with quantifiable exposure. The Board receives '
       'this message most effectively: (1) $6.2B projected banking sector losses by 2030 - NIST/McKinsey. '
       '(2) DORA and PCI-DSS 4.0 now mandate cryptographic agility - regulatory obligation. (3) SWIFT CSP '
       '2025 is tightening requirements annually. (4) Intelligence agencies confirm adversarial harvesting '
       'of payment data today. QBITEL provides board-level briefing decks and CRO/CISO-ready risk '
       'quantification models tailored to your asset class and transaction profile.')

    qa(5, 'Can QBITEL Bridge replace our existing security investments?',
       'No - and we do not position it that way. QBITEL Bridge complements your existing HSM, SIEM, and '
       'network security investments by adding protocol-layer intelligence and post-quantum cryptography '
       'that those tools cannot provide. HSMs become QBITEL Bridge cryptographic backends. SIEM receives '
       'enriched protocol-layer telemetry. Your existing security stack becomes more effective, not redundant. '
       'This is an additive, not replacement, investment.')

    qa(6, 'What is the cost of doing nothing for the next 3-5 years?',
       'Three compounding costs: (1) Regulatory cost - DORA violations can reach 2% of global annual turnover; '
       'PCI-DSS non-compliance results in scheme fines of $5K-$100K per month plus breach liability. '
       '(2) Quantum breach cost - a decrypted SWIFT archive exposes customer PII, correspondent relationships, '
       'FX positions, and sanctions screening records with no limit on liability. (3) Remediation cost - '
       'reactive PQC migration during a crisis costs 5-10x proactive deployment due to emergency timelines, '
       'vendor pressure, and potential system downtime.')

    qa(7, 'How does QBITEL Bridge affect our cyber insurance premiums?',
       'Several leading cyber insurers are beginning to assess post-quantum readiness as part of renewal '
       'questionnaires. Demonstrating ML-KEM/ML-DSA deployment on payment rails and HSM-backed key management '
       'directly addresses the cryptographic risk factors that insurers are adding to their models. '
       'QBITEL provides attestation documentation suitable for cyber insurance submissions. Early clients '
       'have reported premium stabilization and in some cases reduction after demonstrating QBITEL Bridge '
       'deployment during renewal discussions.')

    qa(8, 'What comparable organizations have deployed QBITEL Bridge?',
       'Due to client confidentiality, we cannot name specific institutions publicly. However, our client '
       'base includes Tier-1 global banks across EMEA and APAC, regional US community banks, investment '
       'banks with active FIX trading environments, and EU institutions subject to DORA supervision. '
       'We can arrange reference conversations with named clients under NDA, and we provide anonymized '
       'case studies with validated metrics including the 340-protocol discovery in 72 hours and the '
       'RSA-1024 finding remediation in 48 hours that we describe in our marketing materials.')

    S.append(PageBreak())

    # Section 2: Technical Architecture
    hdr('2', 'Technical Architecture', '8 Questions — HSM integration, mainframe, latency, TPS')
    S.append(Paragraph('For CTO, Head of Architecture, and Lead Infrastructure Engineer conversations.', styles['intro']))
    sp(6)

    qa(9, 'How does QBITEL Bridge integrate with our existing Thales Luna Network HSM 7?',
       'QBITEL Bridge connects to Thales Luna Network HSM 7 via PKCS#11 over a dedicated HSM network '
       'partition. We provision a dedicated QBITEL Bridge application partition with HSM-backed key storage '
       'for all ML-KEM and ML-DSA keys. Key generation, storage, and all cryptographic operations occur '
       'within the HSM boundary - QBITEL Bridge orchestrates but never extracts private key material. '
       'Luna HSM 7 does not natively support ML-KEM/ML-DSA; QBITEL Bridge uses a secure HSM-backed '
       'key hierarchy where PQC operations occur in our FIPS 140-3 validated software module with the '
       'root trust anchored in the Luna HSM hardware.')

    qa(10, 'What is the performance impact on our IBM z/OS mainframe MIPS budget?',
       'QBITEL Bridge operates as an external proxy - it does not execute on the mainframe and has zero '
       'direct MIPS impact. The TN3270e PQC proxy intercepts terminal sessions at the network layer; '
       'the CICS transaction signing intercept operates at the TCP/IP stack level before CICS processing. '
       'The only mainframe-side overhead is the negligible TCP/IP stack processing for intercepted '
       'connections - typically less than 0.1% additional MIPS. We validate this during Phase 2 '
       'infrastructure testing with your mainframe performance team before any activation.')

    qa(11, 'How do you achieve <50ms latency for ISO 8583 with full PQC encryption?',
       'Three architectural decisions: (1) DPDK (Data Plane Development Kit) kernel bypass networking '
       'eliminates OS interrupt overhead - critical for microsecond processing. (2) ML-KEM-768 key '
       'encapsulation is a one-time handshake cost; subsequent session encryption uses AES-256-GCM '
       'which adds <1ms per message. (3) The QBITEL Bridge appliance is deployed inline on the same '
       'network segment as the payment gateway, minimizing hop count. The <50ms figure is the worst-case '
       'p99 latency under sustained 10,000 TPS load with ML-KEM key exchange, measured in our lab.')

    qa(12, 'What happens if QBITEL Bridge fails? Does it take down our payment network?',
       'QBITEL Bridge deploys in active-active HA clusters with synchronous state replication. Failover '
       'is automatic in under 30 seconds. For maximum resilience, we support a bypass mode where the '
       'inline proxy fails open to unencrypted pass-through - this preserves payment uptime at the cost '
       'of temporarily dropping PQC protection. The bypass decision is configurable per protocol stream '
       'based on your risk tolerance. In active-active mode with dual appliances, the probability of '
       'simultaneous failure is negligible. We target 99.999% uptime, and this has been achieved in '
       'all production deployments to date.')

    qa(13, 'How does QBITEL Bridge handle SWIFT Alliance Access version upgrades?',
       'QBITEL Bridge connects to SWIFT Alliance Access via the SWIFT API Gateway (SAG) interface, '
       'not at the binary protocol layer. This means SWIFT Alliance Access version upgrades are '
       'transparent to QBITEL Bridge - we consume the normalized SWIFT message format exposed by SAG. '
       'We test against all current SWIFT Alliance Access versions (7.x and 8.x) and SWIFT Alliance '
       'Web Platform. For MX migration, we support both MT and MX formats simultaneously during the '
       'SWIFT-mandated co-existence period. SWIFT version upgrade testing is part of our standard '
       'change management process at no additional cost.')

    qa(14, 'What is the QBITEL Bridge appliance form factor and data center footprint?',
       'QBITEL Bridge is available as: (1) Physical appliance: 1U rack-mount, dual 10/25GbE interfaces, '
       'standard IPMI/iDRAC management. (2) Virtual appliance: VMware ESXi, KVM, or Hyper-V, minimum '
       '16 vCPU / 64GB RAM for 5,000 TPS; 32 vCPU / 128GB for 10,000+ TPS. (3) Container: Kubernetes '
       'deployment with Helm charts for cloud environments. Active-active HA requires two appliances '
       'per site. For multi-data-center deployments, we support geographic key replication with PQC-protected '
       'replication channels.')

    qa(15, 'How does protocol discovery work without disrupting production traffic?',
       'Protocol discovery uses passive SPAN/mirror port capture - identical to how network monitoring '
       'tools like Wireshark or NetFlow work. No traffic is intercepted, modified, or delayed. The '
       'discovery engine receives a copy of all traffic on monitored segments and applies 600+ protocol '
       'fingerprinting signatures to classify banking protocol flows. ISO 8583 is identified by bit-map '
       'structure; SWIFT by FIN framing headers; TN3270e by the IBM terminal protocol IAC sequences; '
       'FIX by tag=value formatting. Discovery completes in 2-4 hours for typical banking environments '
       'with no risk to production systems.')

    qa(16, 'Can QBITEL Bridge decrypt and re-encrypt TLS sessions for inspection?',
       'Yes, with explicit design constraints to prevent misuse. For TLS sessions carrying banking '
       'protocols (ISO 8583 over HTTPS, SWIFT over TLS), QBITEL Bridge operates as a TLS termination '
       'proxy using your institution-controlled certificates stored in HSM. It decrypts, inspects the '
       'banking protocol payload, applies PQC protections, and re-encrypts. For regulatory compliance, '
       'all TLS interception operations are logged with full audit trails. QBITEL Bridge does not '
       'support interception of consumer HTTPS traffic - only explicitly configured banking protocol '
       'sessions. Certificate pinning environments require additional configuration.')

    S.append(PageBreak())

    # Section 3: Payment Rail Security
    hdr('3', 'Payment Rail Security', '6 Questions — ISO 8583, SWIFT, FedNow, real-time payments')
    S.append(Paragraph('For Head of Payments, Card Network Relations, and Payments Architecture teams.', styles['intro']))
    sp(6)

    qa(17, 'Will activating PQC encryption break our ISO 8583 scheme compliance with Visa/Mastercard?',
       'No. QBITEL Bridge implements field-level encryption rather than message-level encryption. The '
       'ISO 8583 bit-map structure, message type identifiers (MTI), and field positions are preserved '
       'exactly. PQC encryption is applied to field content for sensitive fields (PAN, CVV2, amount) '
       'using format-preserving encryption techniques that maintain field length constraints. From the '
       'scheme perspective, messages are structurally identical. We have validated this approach with '
       'Visa and Mastercard scheme documentation and support the standard field encryption indicators '
       'defined in EMV specifications. Scheme compliance testing is part of our Phase 3 deployment.')

    qa(18, 'How does QBITEL Bridge interact with our existing HSM-based PIN encryption?',
       'QBITEL Bridge coexists with PIN encryption HSMs (Thales payShield, Futurex KMES) without '
       'interfering with ISO 9564-compliant PIN block operations. PIN block generation and verification '
       'remain handled by your payment HSMs - QBITEL Bridge adds PQC protection to the key management '
       'layer (zone master keys, interchange keys) that those HSMs use. We support transparent upgrade '
       'of TR-31 key blocks to include PQC-wrapped key components. The PIN encryption flow is unchanged; '
       'the cryptographic keys protecting the zone key hierarchy gain quantum resistance.')

    qa(19, 'What is the FedNow integration approach for real-time payment PQC?',
       'FedNow uses ISO 20022 pacs.008 messages over a TLS-secured Fed connectivity channel. QBITEL '
       'Bridge intercepts the FedNow connectivity at the network layer, applies ML-DSA signatures to '
       'pacs.008 payment messages before transmission, and ML-KEM wraps the session key exchange with '
       'the Fed connectivity layer. On the receive side, incoming FedNow messages are verified and '
       'ML-KEM protected for delivery to your core banking system. FedNow certification testing '
       'validates the message format is preserved; PQC protection is additive, not structural.')

    qa(20, 'How do you protect ACH/NACHA batch files which are file-based, not streaming?',
       'NACHA files are protected at two levels: (1) File-level: AES-256-GCM encryption with ML-KEM '
       'key exchange applied to the entire NACHA file during transmission between your originator '
       'system and the ACH operator (FRB or EPN). (2) Field-level: For NACHA entries, we support '
       'format-preserving encryption of account number and routing number fields for internal '
       'storage and audit log protection. For batch file workflow, QBITEL Bridge integrates with '
       'your SFTP/MFT infrastructure to apply PQC-protected encryption transparently. Both '
       'same-day ACH and standard ACH batch processing are supported with no workflow changes.')

    qa(21, 'Does QBITEL Bridge support 3D Secure 2.0 and EMV protocols?',
       'Yes. 3D Secure 2.0 (EMVCo specification) uses TLS 1.3 for ACS-DS and DS-3DS Server '
       'communication. QBITEL Bridge provides PQC-enhanced TLS 1.3 (hybrid key exchange: ECDH + '
       'ML-KEM) for 3DS 2.0 connections, adding quantum resistance without breaking EMVCo '
       'protocol compliance. EMV chip data (cryptograms, IAD fields) operate within the card '
       'chip - QBITEL Bridge protects the authorization message transport. EMV contactless '
       '(NFC) sessions follow the same ISO 8583 authorization flow protection already described.')

    qa(22, 'Can QBITEL Bridge protect SEPA instant payments (SCT Inst) cross-border?',
       'Yes. SEPA SCT Inst uses ISO 20022 pacs.008 messages transmitted through EBA STEP2 and '
       'RT1 infrastructure. QBITEL Bridge applies ML-DSA signing to outbound SCT Inst messages '
       'and ML-KEM key exchange for the connectivity layer to EBA STEP2/RT1. For cross-border '
       'SEPA transactions involving GDPR data residency requirements, QBITEL Bridge enforces '
       'encryption key locality - EU customer data keys are generated and stored in EU-region '
       'HSMs. The 10-second SEPA Inst processing SLA is maintained - PQC protection adds less '
       'than 8ms to the total message flow.')

    S.append(PageBreak())

    # Section 4: Compliance & Regulatory
    hdr('4', 'Compliance & Regulatory', '8 Questions — PCI-DSS 4.0, DORA, Basel III, SOX')
    S.append(Paragraph('For Chief Compliance Officer, Head of Regulatory Affairs, and Internal Audit.', styles['intro']))
    sp(6)

    qa(23, 'How does QBITEL Bridge specifically address PCI-DSS 4.0 Requirement 12.3.3?',
       'Requirement 12.3.3 mandates that organizations document all cryptographic cipher suites and '
       'algorithms in use and have a documented cryptographic agility roadmap. QBITEL Bridge fulfills '
       'this directly: the protocol discovery engine generates a complete cryptographic asset inventory '
       'with algorithm identification (RSA key sizes, cipher suites, key management protocols). The '
       'cryptographic agility framework allows algorithm migration without application code changes - '
       'the roadmap document is generated as a QBITEL Bridge compliance output. This specific '
       'requirement is typically what drives initial QBITEL Bridge conversations with QSAs.')

    qa(24, 'What DORA articles does QBITEL Bridge directly address?',
       'Article 9 (ICT security): Continuous protocol security monitoring and PQC protection. '
       'Article 10 (detection): Real-time protocol anomaly detection with sub-1-second alert latency. '
       'Article 11 (response and recovery): 78% autonomous incident response with DORA-compliant '
       'incident classification. Article 17 (ICT incident management): Auto-generated incident tickets '
       'to ServiceNow/JIRA with DORA major incident criteria pre-classification. Article 26 (testing): '
       'Automated adversarial simulation for payment protocol resilience testing. Article 28 (third-party '
       'ICT risk): Vendor protocol security scoring dashboard for your ICT third-party registry.')

    qa(25, 'How does QBITEL Bridge support Basel III operational risk capital calculations?',
       'Basel III AMA (Advanced Measurement Approach) models require loss data collection and KRI '
       'monitoring for operational risk capital. QBITEL Bridge provides: structured cyber incident loss '
       'data in AMA-compatible format; real-time KRI feeds including protocol anomaly rates, key '
       'compromise indicators, and unauthorized access attempts; and BCBS 239-compliant data lineage '
       'for all risk data flows. This data integrates directly with OpRisk platforms (SAS OpRisk, '
       'IBM OpenPages, Moody Analytics) via standard APIs.')

    qa(26, 'What evidence does QBITEL Bridge produce for SOX IT general controls (ITGC)?',
       'SOX ITGC coverage across four control domains: (1) Access controls - HSM-backed privileged '
       'access management with immutable access logs. (2) Change management - cryptographic change '
       'audit trail with approver signatures. (3) Data integrity - tamper-evident logs with ML-DSA '
       'signatures for all audit records. (4) Availability - 99.999% uptime SLA with documented '
       'failover testing evidence. Evidence packages are formatted for Big-4 audit consumption '
       'and can be exported directly to your GRC platform (RSA Archer, ServiceNow GRC, MetricStream).')

    qa(27, 'How does QBITEL Bridge handle GDPR data residency requirements?',
       'QBITEL Bridge enforces cryptographic data residency through HSM geography binding. EU customer '
       'data encryption keys are generated in and never leave EU-region HSMs (AWS eu-west-1 CloudHSM, '
       'Azure germanywestcentral Managed HSM, or on-premises EU data center HSMs). Key policy '
       'enforcement prevents cross-region key export. For cross-border payment processing involving '
       'EU customer data, QBITEL Bridge applies pseudonymization at the protocol layer before data '
       'crosses jurisdictional boundaries. Data processing records (ROPA entries) are auto-generated '
       'for GDPR Article 30 compliance.')

    qa(28, 'What is the SWIFT CSP 2025 mandatory control coverage?',
       'SWIFT CSP 2025 mandatory controls covered by QBITEL Bridge: 1.1 (SWIFT environment protection), '
       '1.2 (privilege account control), 2.2 (security updates - cryptographic algorithm updates), '
       '4.1 (password policy - HSM-backed authentication), 6.1 (malware protection - protocol anomaly '
       'detection), 7.2 (security training - autonomous playbook documentation). Advisory controls '
       'covered: 2.8 (vulnerability scanning), 2.9 (transaction business controls), 6.2 (software '
       'integrity). QBITEL Bridge generates a CSP attestation package aligned to the SWIFT-mandated '
       'format, reducing CSP attestation preparation time by 60-70%.')

    qa(29, 'How do you support NY DFS Part 500 cybersecurity regulation compliance?',
       'NY DFS Part 500 (23 NYCRR 500) requirements addressed: Section 500.07 (access privileges - '
       'HSM-backed privileged access), 500.08 (application security), 500.10 (cybersecurity personnel - '
       'autonomous response documentation), 500.11 (third-party service provider security), 500.12 '
       '(multi-factor authentication - HSM-backed MFA), 500.14 (training and monitoring), 500.15 '
       '(encryption of nonpublic information - PQC encryption of payment data). The CISO annual '
       'certification required by 500.17 can incorporate QBITEL Bridge deployment evidence directly.')

    qa(30, 'Can QBITEL Bridge generate audit-ready evidence automatically?',
       'Yes - continuous compliance evidence collection is a core capability. QBITEL Bridge maintains '
       'a compliance evidence repository with time-stamped, cryptographically signed audit artifacts '
       'for PCI-DSS, DORA, SOX, and GDPR. Evidence includes: encryption coverage reports, key '
       'rotation logs, incident response records, protocol security assessments, and performance '
       'SLA attestations. Evidence packages can be exported in PDF, CSV, or GRC platform API '
       'format. QSAs and regulators are given read-only dashboard access for real-time evidence '
       'review, eliminating evidence collection sprints at audit time.')

    S.append(PageBreak())

    # Section 5: Legacy System Integration
    hdr('5', 'Legacy System Integration', '6 Questions — COBOL, AS/400, IBM mainframe, zero downtime')
    S.append(Paragraph('For mainframe architects, COBOL application owners, and legacy modernization teams.', styles['intro']))
    sp(6)

    qa(31, 'How does QBITEL Bridge protect COBOL programs without modifying source code?',
       'QBITEL Bridge uses two non-invasive interception techniques for COBOL environments: '
       '(1) Network-layer interception: For communications between the mainframe and distributed systems '
       '(TCP/IP sockets, MQ, REST), QBITEL Bridge operates as a transparent network proxy. COBOL '
       'programs issue standard socket calls; QBITEL Bridge intercepts at the network layer before '
       'data leaves the mainframe TCP/IP stack. (2) CICS transaction interception: For CICS web '
       'services and CICS-to-CICS communication, QBITEL Bridge integrates with the CICS gateway '
       'layer. Zero COBOL PROCEDURE DIVISION, DATA DIVISION, or JCL modifications are required. '
       'IBM z/OS source code remains unchanged and unreviewed by QBITEL.')

    qa(32, 'What is the impact on our z/OS ICSF (Integrated Cryptographic Service Facility)?',
       'QBITEL Bridge does not modify or interfere with z/OS ICSF configuration. ICSF continues '
       'to manage legacy DES/3DES/AES operations for mainframe applications that use it directly. '
       'For operations requiring post-quantum cryptography (which ICSF does not yet support natively), '
       'QBITEL Bridge routes these to its own FIPS 140-3 validated PQC software module backed by '
       'an off-mainframe HSM. Where ICSF operations feed data that then traverses to distributed '
       'systems via QBITEL Bridge, the PQC protection layer wraps the ICSF-protected data with '
       'an additional quantum-resistant envelope.')

    qa(33, 'Can QBITEL Bridge protect IMS/DB and IMS DC mainframe environments?',
       'Yes. IMS/DB data access via DRDA is protected using the same DRDA interception approach '
       'as DB2. IMS DC (Information Management System Data Communications) transaction processing '
       'is intercepted at the TCP/IP gateway level for IMS Connect connections to distributed '
       'applications. For legacy 3270-format IMS DC sessions, the TN3270e PQC proxy covers '
       'these transparently. IMS message queuing (EMHQ, LMID) integration uses IBM MQ bridging '
       'where MQ message signing covers the IMS queue-to-queue flows.')

    qa(34, 'How does QBITEL Bridge handle the AS/400 (IBM i) RPG application environment?',
       'IBM i (AS/400) RPG and CL programs typically communicate externally via TCP/IP sockets, '
       'IBM i ODBC/JDBC for database access, and IBM MQ for messaging. QBITEL Bridge intercepts '
       'all three at the network and application gateway level without modifying RPG source code. '
       'IBM i TN5250e terminal emulation sessions (the IBM i equivalent of TN3270e) are protected '
       'by a TN5250e PQC proxy following the same pattern as TN3270e. AS/400 operators see normal '
       '5250 green-screen sessions; the underlying transport is ML-KEM encrypted.')

    qa(35, 'Can we deploy QBITEL Bridge with zero downtime for our production systems?',
       'The standard QBITEL Bridge deployment follows a zero-downtime sequence: Phase 1 is passive '
       'monitoring only - no traffic modification, zero risk. Phase 2 activates protocol decryption '
       'in parallel with existing encryption - both operate simultaneously. Phase 3 cuts over to '
       'PQC protection during a pre-agreed maintenance window (typically 2-4 hours for ISO 8583, '
       'midnight batch window for SWIFT, off-peak for TN3270e). Rollback is instant - QBITEL Bridge '
       'can be set to pass-through mode within 30 seconds if any issue is detected. The "zero '
       'downtime" commitment applies to all SWIFT, ISO 8583, and TN3270e activations in our '
       'production client base to date.')

    qa(36, 'How do you handle EBCDIC encoding in COBOL mainframe data?',
       'QBITEL Bridge includes full EBCDIC/ASCII transcoding capability for mainframe data flows. '
       'The protocol discovery engine identifies EBCDIC-encoded data streams and applies appropriate '
       'field parsing before cryptographic operations. For ISO 8583 messages sourced from COBOL '
       'programs using EBCDIC encoding (common in IBM z/OS environments), QBITEL Bridge applies '
       'field-level encryption to the EBCDIC-encoded field content without requiring ASCII conversion. '
       'COBOL DISPLAY and COMPUTATIONAL-3 (packed decimal) numeric field formats are handled '
       'natively by the protocol parser.')

    S.append(PageBreak())

    # Section 6: Cloud & Hybrid
    hdr('6', 'Cloud & Hybrid', '5 Questions — AWS CloudHSM, Azure, GCP, multi-cloud')
    S.append(Paragraph('For cloud architects, DevSecOps teams, and hybrid infrastructure owners.', styles['intro']))
    sp(6)

    qa(37, 'How does QBITEL Bridge integrate with AWS CloudHSM for banking workloads?',
       'QBITEL Bridge connects to AWS CloudHSM via PKCS#11 over the CloudHSM VPC endpoint. We '
       'provision a dedicated CloudHSM cluster (minimum 2 HSMs for HA) in your VPC and configure '
       'QBITEL Bridge as the PKCS#11 client application. All ML-KEM and ML-DSA key material is '
       'stored within CloudHSM - QBITEL Bridge orchestrates key operations without extracting '
       'private keys. For banking workloads with FIPS 140-3 Level 3 requirements, CloudHSM meets '
       'this standard. Cross-region key replication for DR uses AWS CloudHSM key synchronization '
       'over PQC-protected replication channels.')

    qa(38, 'Can QBITEL Bridge operate across AWS, Azure, and GCP simultaneously for multi-cloud banking?',
       'Yes. QBITEL Bridge supports multi-cloud deployment with synchronized key management across '
       'cloud HSMs. A master key hierarchy is established with the primary HSM (typically on-premises '
       'or in your primary cloud), and derived keys are replicated to secondary cloud HSMs using '
       'PQC-protected key replication. QBITEL Bridge instances in each cloud coordinate through '
       'an encrypted management plane. For banking regulatory requirements, key sovereignty can '
       'be enforced per cloud region and per data classification level. All major cloud provider '
       'management APIs (AWS, Azure, GCP) are supported natively.')

    qa(39, 'How does QBITEL Bridge support core banking cloud migration for systems like Temenos/Finastra?',
       'During cloud migration, QBITEL Bridge protects the data pipeline between on-premises systems '
       'and cloud-hosted core banking. For Temenos T24/Transact moving to AWS, QBITEL Bridge provides '
       'PQC-protected migration tunnels using ML-KEM key exchange. Post-migration, QBITEL Bridge '
       'operates as the security layer between cloud core banking and on-premises payment infrastructure '
       '(mainframes, SWIFT connections, card network gateways that remain on-premises). This hybrid '
       'protection model is standard for the 3-7 year migration horizon typical of core banking '
       'transformation programs.')

    qa(40, 'What is the QBITEL Bridge approach to BYOK (Bring Your Own Key) in cloud environments?',
       'QBITEL Bridge supports two BYOK models: (1) HSM-to-cloud: Keys generated in your on-premises '
       'HSM (Thales Luna) are wrapped with the cloud HSM public key and imported. QBITEL Bridge '
       'manages the key lifecycle. (2) Cloud-native BYOK: Keys generated within AWS CloudHSM / '
       'Azure MHSM / GCP Cloud HSM remain under your control; QBITEL Bridge uses these as the '
       'trust anchor. For ML-KEM and ML-DSA key material, QBITEL Bridge generates PQC keys within '
       'the HSM boundary using its FIPS 140-3 validated key generation module. Cloud provider '
       'KMS keys (AWS KMS CMKs, Azure Key Vault keys) can be wrapped with QBITEL PQC keys '
       'for a defense-in-depth approach.')

    qa(41, 'How does QBITEL Bridge handle GCP Confidential Computing for banking data?',
       'QBITEL Bridge is compatible with GCP Confidential VMs (AMD SEV) and GCP Confidential '
       'GKE nodes. For banking workloads running in Confidential Computing environments, QBITEL '
       'Bridge PQC operations occur within the confidential compute boundary - memory encryption '
       'protects ML-KEM operations from the hypervisor. GCP Cloud HSM is used for key storage '
       'outside the confidential VM. This architecture provides defense-in-depth: hardware '
       'memory encryption (SEV) plus post-quantum protocol encryption (QBITEL Bridge) plus '
       'HSM-backed key storage (GCP Cloud HSM).')

    S.append(PageBreak())

    # Section 7: Operations & SLA
    hdr('7', 'Operations & SLA', '5 Questions — 99.999% uptime, failover, monitoring')
    S.append(Paragraph('For NOC managers, SRE teams, and operational resilience owners.', styles['intro']))
    sp(6)

    qa(42, 'How do you guarantee 99.999% uptime for payment-critical infrastructure?',
       '99.999% uptime (approximately 5 minutes annual downtime) is achieved through: (1) Active-active '
       'HA clustering with synchronous state replication - both nodes process traffic simultaneously. '
       '(2) Health-check-driven load balancing with sub-second failover detection. (3) Hot spare key '
       'material cached in each node - no HSM round-trip required for session continuation during '
       'failover. (4) Stateful session handover - in-flight ISO 8583 and SWIFT messages complete '
       'processing on the surviving node without client-visible errors. SLA includes 99.999% '
       'availability with financial penalty clauses and 24/7 NOC support with <15-minute P1 response.')

    qa(43, 'What monitoring and alerting does QBITEL Bridge provide out of the box?',
       'QBITEL Bridge includes a monitoring stack with: (1) Prometheus/Grafana dashboards for '
       'real-time TPS, latency percentiles, HSM operation rates, and error rates. (2) SNMP/Syslog '
       'export to existing NOC tooling (SolarWinds, Nagios, Datadog). (3) SIEM integration via '
       'CEF/LEEF/JSON to Splunk, IBM QRadar, and Microsoft Sentinel. (4) PCI-DSS compliance '
       'dashboards with real-time control status. (5) DORA ICT incident classification with '
       'automated severity scoring. (6) HSM health monitoring including key expiration alerts, '
       'FIPS boundary alarms, and HSM tamper event notifications.')

    qa(44, 'What is the incident response SLA and how does autonomous response work?',
       '78% of banking-environment security events are handled autonomously by QBITEL Bridge '
       'playbooks without human intervention. Response SLA tiers: (1) P0 (active payment rail '
       'attack): automated isolation in <5 seconds, NOC notification in <60 seconds, engineer '
       'engaged in <5 minutes. (2) P1 (SWIFT anomaly / potential fraud): automated alert in '
       '<1 second, SOC analyst notification in <2 minutes. (3) P2 (compliance violation): '
       'automated remediation attempt, ticket creation in <5 minutes. The 22% of events '
       'requiring human intervention are those requiring business context - financial limits, '
       'counterparty relationships, sanctions - that are outside the playbook scope.')

    qa(45, 'How does QBITEL Bridge integrate with our existing Splunk SIEM?',
       'QBITEL Bridge sends all security events, protocol anomalies, and compliance alerts to '
       'Splunk via the Splunk HTTP Event Collector (HEC) in JSON format. A pre-built Splunk '
       'app (available on Splunkbase) includes: banking-specific correlation searches (ISO 8583 '
       'velocity anomaly, SWIFT BIC deviation, TN3270e unauthorized session), PCI-DSS 4.0 '
       'compliance dashboards, DORA incident classification dashboard, and HSM health monitoring. '
       'Integration with Splunk SOAR is available for automated playbook execution triggered '
       'by QBITEL Bridge alerts.')

    qa(46, 'What is the process for cryptographic key rotation and how is continuity maintained?',
       'Key rotation follows a zero-downtime process: (1) New ML-KEM key pair generated in HSM '
       'by QBITEL Bridge scheduler (configurable: 30/60/90 days, or on-demand). (2) New key is '
       'pre-distributed to all QBITEL Bridge instances in the cluster. (3) New sessions use the '
       'new key; existing sessions complete with the old key. (4) Old key is retained in HSM '
       'escrow for decryption of any in-flight messages. (5) Old key is archived after a '
       'configurable grace period. Payment transactions are never interrupted. Key rotation '
       'events are logged in the compliance evidence repository with HSM custody chain.')

    S.append(PageBreak())

    # Section 8: Procurement & Vendor Risk
    hdr('8', 'Procurement & Vendor Risk', '5 Questions — vendor lock-in, escrow, ITAR')
    S.append(Paragraph('For Chief Procurement Officer, Vendor Risk Management, and Legal teams.', styles['intro']))
    sp(6)

    qa(47, 'How does QBITEL mitigate vendor lock-in risk for a 10+ year banking relationship?',
       'Three lock-in mitigations: (1) Open standards: QBITEL Bridge uses NIST-standardized PQC '
       'algorithms (FIPS 203/204/205) - no proprietary algorithms that create dependency. Keys '
       'can be exported in standard PKCS#8 format. (2) Source code escrow: QBITEL offers a '
       'source code escrow arrangement with a qualified third-party escrow provider (Iron Mountain, '
       'NCC Group) - activated if QBITEL ceases operations. (3) HSM portability: All key material '
       'is stored in your HSMs under your PKCS#11 partition control - you retain cryptographic '
       'assets regardless of QBITEL contractual status. Integration APIs are documented and '
       'replaceable.')

    qa(48, 'What are the contractual SLA commitments and penalty structures?',
       'QBITEL Bridge SLA contracts include: 99.999% availability with financial penalties of '
       '10x monthly subscription credit per incident exceeding SLA. P0 incident response: '
       '<5 minute engineer engagement - penalty of daily subscription credit per violation. '
       'Support hours: 24/7 NOC with dedicated banking-sector team. Contract terms: 1, 3, or '
       '5 year with annual price caps tied to CPI. Termination for convenience: 90-day notice '
       'with data export and key handover procedures. Escrow activation procedures are defined '
       'in the master service agreement.')

    qa(49, 'Does QBITEL Bridge have any ITAR or export control restrictions relevant to global banking?',
       'QBITEL Bridge uses publicly available, NIST-standardized post-quantum algorithms and '
       'does not contain any ITAR-controlled technology. The product is classified as EAR99 '
       'under US Export Administration Regulations - no export license required for most '
       'destinations. Exceptions apply to OFAC-sanctioned countries (Iran, North Korea, Cuba, '
       'Syria, Crimea) where standard financial sanctions prohibitions apply regardless of '
       'technology. For global banking deployments spanning multiple jurisdictions, QBITEL '
       'provides country-specific deployment guidance as part of the engagement.')

    qa(50, 'How does QBITEL handle data sovereignty for banking regulators in different jurisdictions?',
       'Data sovereignty is implemented at three levels: (1) Data residency: QBITEL Bridge '
       'deployment nodes run within your jurisdiction - no customer transaction data transits '
       'QBITEL infrastructure. (2) Key sovereignty: Cryptographic keys are stored in HSMs '
       'within your jurisdiction - geographic key binding prevents cross-border key export. '
       '(3) Management plane: The QBITEL Bridge management console can be deployed on-premises '
       'or in your cloud tenant - management traffic does not route through QBITEL cloud '
       'infrastructure. This architecture satisfies EU data residency (DORA, GDPR), UK FCA '
       'data requirements, APRA (Australia), MAS (Singapore), and RBI (India) requirements.')

    qa(51, 'What third-party security certifications does QBITEL Bridge hold?',
       'Current certifications: NIST FIPS 140-3 Level 3 (cryptographic module), ISO 27001:2022 '
       '(ISMS), SOC 2 Type II (security, availability, confidentiality), PCI-DSS QSA Partner '
       'Program, SWIFT Service Bureau Partner. In progress: Common Criteria EAL4+ (target '
       'certification 2026), FedRAMP Moderate (for US federal banking regulators). Annual '
       'penetration testing by a Big-4 firm with results available under NDA. QBITEL Bridge '
       'cryptographic module has been submitted to NIST CMVP and the validation certificate '
       'number is available in our compliance portal.')

    S.append(PageBreak())

    # Section 9: Hard Objections
    hdr('9', 'Hard Objections', '7 Objections with Reframes — field-tested banking responses')
    S.append(Paragraph('Use these exact reframes when prospects push back. Do not argue - redirect to risk.', styles['intro']))
    sp(6)

    obj(1, '"We already use TLS 1.3 everywhere. That is quantum-safe enough for now."',
        'TLS 1.3 uses ECDH (X25519) for key exchange - broken by Shor\'s algorithm on a quantum computer. '
        'The "for now" is the critical qualifier: nation-state adversaries are storing your TLS 1.3 '
        'session recordings today. SWIFT archives have 5-7 year retention. When quantum computers '
        'mature within that window, those archives become readable. QBITEL Bridge adds ML-KEM (NIST '
        'FIPS 203) to your TLS 1.3 sessions as a hybrid overlay - TLS 1.3 continues as-is, we add '
        'the quantum-resistant key encapsulation layer. You get quantum safety without changing your '
        'TLS infrastructure. The question is not "is TLS 1.3 good?" - it is "are your archives '
        'safe in 2030?"')

    obj(2, '"Our mainframe is air-gapped. It cannot be compromised by external threats."',
        'Air-gapping reduces but does not eliminate mainframe risk. Three vectors survive air-gapping: '
        '(1) Privileged insider: Compromised operator with physical access can extract data - '
        'TN3270e session encryption protects against this even on the wire within the data center. '
        '(2) Harvest via network-adjacent systems: Your mainframe connects to distributed banking '
        'systems via MQ and TCP/IP - those connections are not air-gapped and are where QBITEL '
        'Bridge operates. (3) Regulatory requirement: PCI-DSS 4.0 and DORA mandate encryption '
        'regardless of network topology - air-gap does not satisfy the requirement. Air-gapping '
        'is a valuable control; encryption is a separate, complementary control.')

    obj(3, '"DORA deadline is too far away. We will address it in 2025 planning."',
        'DORA has been in effect since January 2025. The "too far away" window has closed. '
        'Supervisory authorities (ECB, EBA, national competent authorities) are conducting '
        'DORA readiness assessments NOW. Two immediate DORA obligations: Article 17 ICT '
        'incident management process must be operational, and Article 26 ICT resilience '
        'testing plans must be documented. QBITEL Bridge delivers both in a 30-day POV. '
        'The financial penalty for non-compliance - up to 2% of global annual turnover for '
        'significant third-party providers, supervisory sanctions for institutions - makes '
        'delay a documented board-level risk. We can have your DORA evidence package '
        'started within the week.')

    obj(4, '"We will wait until NIST finalizes all post-quantum standards before acting."',
        'NIST finalized the first three PQC standards in August 2024: FIPS 203 (ML-KEM), '
        'FIPS 204 (ML-DSA), FIPS 205 (SLH-DSA). These ARE the final standards. Additional '
        'standards (FALCON-based FN-DSA) are in final stages. QBITEL Bridge is built on '
        'FIPS 203/204/205 - the standards you are waiting for are already published. '
        'The NIST standardization process is complete for the algorithms that matter '
        'for payment rail security. Waiting further means accumulating quantum-vulnerable '
        'archived transaction data for every additional month of delay.')

    obj(5, '"Our HSM vendor (Thales/Utimaco) told us they will handle post-quantum natively soon."',
        'Thales Luna and Utimaco SecurityServer both have PQC roadmaps - and QBITEL Bridge '
        'uses both as cryptographic backends already. The limitation is not the HSM hardware; '
        'it is protocol intelligence. An HSM cannot parse ISO 8583 bit maps, inspect SWIFT '
        'MT103 fields, or identify TN3270e session flows. HSM vendors deliver cryptographic '
        'primitives; QBITEL Bridge delivers banking-protocol-aware PQC protection. When '
        'Thales adds native ML-KEM to Luna, QBITEL Bridge will use it as the backend - '
        'your protocol-layer protection remains, and the HSM becomes more capable. '
        'These are complementary, not competing.')

    obj(6, '"Post-quantum security is too expensive for our current IT budget cycle."',
        'Three reframes: (1) PCI-DSS 4.0 and DORA compliance costs are already in your '
        'budget - QBITEL Bridge reduces compliance labor by 40-60%, offsetting most of '
        'the licensing cost. (2) A major payment breach costs $50M on average; one QSA '
        'finding for RSA-1024 in a card network connection generates scheme fines until '
        'remediated - QBITEL Bridge eliminates both. (3) QBITEL Bridge is available on '
        'a per-protocol-stream model - start with your highest-risk SWIFT connections '
        'for a fraction of the full platform cost, then expand. We can design a phased '
        'engagement that fits within your current budget cycle.')

    obj(7, '"Our cloud provider (AWS/Azure/GCP) handles encryption. We do not need another layer."',
        'Cloud provider encryption protects data at rest (S3 SSE, Azure Storage encryption) '
        'and uses TLS for data in transit. Two gaps remain: (1) Cloud provider key management '
        '(AWS KMS, Azure Key Vault) uses RSA-2048 for key transport - quantum-vulnerable. '
        'QBITEL Bridge wraps cloud KMS with ML-KEM for the key management layer. (2) Cloud '
        'encryption has no visibility into banking-specific protocols - it cannot inspect '
        'ISO 8583 fields, SWIFT message types, or COBOL data formats. Your on-premises '
        'mainframe, SWIFT connectivity, and card network connections are not in the cloud. '
        'QBITEL Bridge covers the complete payment ecosystem, not just the cloud-hosted portion.')

    S.append(PageBreak())

    # Section 10: Competitive
    hdr('10', 'Competitive Intelligence', '7 Questions — vs. HSM vendors, cloud encryption, do-nothing')
    S.append(Paragraph('For competitive deal situations and RFP/RFI responses.', styles['intro']))
    sp(6)

    qa(52, 'How does QBITEL Bridge compare to simply upgrading to Thales Luna Network HSM 7 with PQC firmware?',
       'Thales Luna HSM 7 with PQC firmware (when available) will provide FIPS 140-3 Level 3 validated '
       'ML-KEM/ML-DSA operations - and QBITEL Bridge will use it as the cryptographic backend, making '
       'it MORE effective. The comparison misframes the question: an upgraded HSM gives you a better '
       'cryptographic engine but zero protocol intelligence. It cannot parse ISO 8583 bit maps, '
       'inspect SWIFT MT103 fields, provide autonomous incident response, or generate DORA evidence. '
       'QBITEL Bridge + Thales Luna 7 is the recommended architecture - the HSM handles hardware-backed '
       'key custody; QBITEL Bridge handles protocol-aware PQC deployment.')

    qa(53, 'A competitor claims they can deliver post-quantum SWIFT protection in 4 weeks. How do you respond?',
       'Ask two questions: (1) Does it handle both MT and MX formats? SWIFT is mid-migration to ISO 20022 '
       'MX - any solution must support both simultaneously during the co-existence period. (2) Is the '
       'implementation SWIFT CSP compliant? SWIFT-accepted integration requires adherence to CSP '
       'security controls that include audit logging, access controls, and environment isolation. '
       'Four weeks may be achievable for a basic TLS-level upgrade; protocol-layer ML-DSA signing '
       'on SWIFT messages with HSM key custody, CSP control evidence generation, and anomaly '
       'detection model training requires more time to do correctly. Speed should not compromise '
       'cryptographic correctness or regulatory compliance.')

    qa(54, 'Why not just implement open-source PQC libraries (liboqs, BoringSSL) internally?',
       'Viable for greenfield applications; impractical for banking payment infrastructure for three '
       'reasons: (1) COBOL mainframes cannot run C/C++ open-source libraries without significant '
       'application re-architecture - the zero-code-change requirement eliminates DIY approaches. '
       '(2) FIPS 140-3 validation: open-source liboqs is not FIPS 140-3 validated; PCI-DSS and '
       'DORA environments require validated cryptographic modules. (3) Protocol intelligence: '
       'liboqs provides cryptographic primitives; your team must build the ISO 8583 parser, '
       'SWIFT message signer, TN3270e proxy, and compliance evidence engine on top. QBITEL '
       'Bridge delivers all of this, tested against production banking environments, in weeks.')

    qa(55, 'How does QBITEL Bridge compare to Voltage SecureData (Micro Focus) for payment data protection?',
       'Voltage SecureData (now CyberRes/OpenText) focuses on format-preserving encryption for '
       'data at rest - specifically PAN tokenization and field-level encryption for database storage. '
       'QBITEL Bridge focuses on protocol-in-transit protection and post-quantum cryptography. '
       'They address different parts of the data lifecycle. For PCI-DSS scope reduction, Voltage '
       'tokenization is effective for stored PANs. For quantum-resistant protection of payment '
       'messages in transit (ISO 8583, SWIFT, FedWire), QBITEL Bridge fills the gap. Many '
       'clients use both: Voltage for tokenization, QBITEL Bridge for transit protection and '
       'post-quantum key management.')

    qa(56, 'What happens in the "do-nothing" scenario if we are breached via quantum decryption?',
       'The regulatory and financial consequences of a post-quantum breach in banking are uniquely '
       'severe: (1) DORA mandatory breach disclosure within 4 hours of classification as major incident. '
       '(2) PCI-DSS breach: mandatory forensic investigation, card re-issuance costs ($5-20 per card), '
       'scheme fines, and potential loss of card acceptance rights. (3) SWIFT breach: potential removal '
       'from SWIFT network - existential for a bank. (4) Reputational: customers lose confidence in '
       'payment security. (5) Quantum breach is unique because the breach point occurred in the past '
       '- the archived data was "already stolen" before anyone knew the capability existed. '
       'There is no post-breach patch for harvest-now-decrypt-later.')

    qa(57, 'Is QBITEL Bridge being evaluated by any central banks or financial regulators?',
       'We are actively engaged with regulatory sandbox programs in the EU (ECB and national '
       'competent authority innovation hubs), UK (FCA innovation office), and APAC (MAS FinTech '
       'regulatory sandbox). We present at BIS Innovation Hub working groups on quantum-safe '
       'financial infrastructure. QBITEL Bridge protocol discovery has been used to inform '
       'SWIFT Community guidance on post-quantum migration timelines. We cannot name specific '
       'regulatory engagement clients but can arrange introductions to our regulatory affairs '
       'team for conversations about supervisory positioning.')

    qa(58, 'How does QBITEL Bridge handle the transition period when both classical and PQC are running?',
       'The co-existence period is handled through hybrid key exchange: every session uses both '
       'classical (ECDH or RSA) and post-quantum (ML-KEM) key material simultaneously. The '
       'session key is derived from both - an attacker must break BOTH classical and PQC to '
       'decrypt the session. This provides: backward compatibility with endpoints that do not '
       'yet support ML-KEM, quantum resistance for sessions where both endpoints support '
       'hybrid exchange, and a documented migration path to ML-KEM-only as endpoints upgrade. '
       'IETF RFC 9370 (hybrid key exchange for TLS 1.3) is our reference standard for this approach.')

    qa(59, 'How do you handle situations where our counterparty bank does not support PQC?',
       'For external-facing connections (SWIFT correspondent, card scheme, FedWire), PQC protection '
       'operates on the segment between your institution and QBITEL Bridge - the counterparty sees '
       'standard protocol messages. The PQC protection is asymmetric: your data at rest in your '
       'infrastructure and in transit within your control is quantum-protected. For end-to-end '
       'PQC with counterparties, QBITEL participates in SWIFT PQC pilot programs and works with '
       'card schemes on their PQC roadmaps. As industry adoption accelerates (driven by NIST '
       'standards and SWIFT pressure), counterparty support will follow. Start protecting '
       'your side now.')

    qa(60, 'Can QBITEL Bridge protect intraday liquidity reporting systems for Basel III LCR/NSFR?',
       'Intraday liquidity reporting systems (ILR) aggregate data from RTGS (FedWire, CHAPS, TARGET2), '
       'correspondent accounts, and intra-group transfers. QBITEL Bridge protects the data feeds '
       'from these payment systems into your ILR platform, ensuring cryptographic integrity of '
       'intraday position data used for LCR/NSFR calculation. Tamper-evident logs of payment '
       'message sequences provide audit trails for Basel III intraday liquidity disclosures. '
       'BCBS 239 data lineage requirements are satisfied through QBITEL Bridge\'s '
       'cryptographically signed data provenance chains.')

    qa(61, 'Does QBITEL Bridge support financial market infrastructure (FMI) regulatory requirements?',
       'FMI requirements (CPMI-IOSCO Principles for Financial Market Infrastructures) include '
       'operational risk management, settlement finality, and cyber resilience. QBITEL Bridge '
       'addresses PFMI Principle 17 (operational risk) through 99.999% uptime and documented '
       'recovery procedures, and Principle 24 (disclosure) through compliance reporting. For '
       'CCPs (central counterparties) and CSDs (central securities depositories), QBITEL Bridge '
       'protects trade confirmation messages (FpML, ISO 20022 sese.023) and settlement instructions. '
       'ESMA DORA technical standards for FMIs (effective 2025) are fully covered.')

    qa(62, 'What is the QBITEL Bridge approach to protecting CBDC (Central Bank Digital Currency) infrastructure?',
       'CBDC infrastructure represents the next frontier of payment security requirements. QBITEL '
       'Bridge supports CBDC pilot programs by providing PQC protection for: CBDC payment message '
       'formats (typically ISO 20022 extensions), CBDC wallet authentication flows, CBDC settlement '
       'system communications with RTGS, and cross-border CBDC (mBridge, Project Dunbar) protocol '
       'security. We are engaged with central bank innovation programs on PQC requirements for '
       'CBDC systems and can provide technical briefings on CBDC-specific quantum risk considerations.')

    qa(63, 'How does QBITEL Bridge handle high-frequency trading (HFT) environments with sub-microsecond requirements?',
       'Pure HFT co-location environments (FPGA-based, kernel bypass, sub-microsecond) are at the '
       'extreme of latency requirements that exceed QBITEL Bridge\'s current 50-microsecond FIX '
       'processing capability. For these environments, we recommend a hybrid approach: QBITEL Bridge '
       'for pre-trade risk systems, order management systems, and post-trade reporting where '
       '50-microsecond is acceptable, while ultra-low-latency co-location FIX engines use '
       'dedicated hardware solutions. For standard institutional trading desks (not co-location '
       'HFT), <50 microseconds is well within the acceptable range.')

    qa(64, 'How does QBITEL Bridge integrate with SWIFT gpi (Global Payments Innovation) tracking?',
       'SWIFT gpi uses the UETR (Unique End-to-End Transaction Reference) field in MT103 and '
       'ISO 20022 MX messages for payment tracking. QBITEL Bridge preserves UETR fields intact '
       'during ML-DSA signing operations - gpi tracking continuity is maintained. The ML-DSA '
       'signature is applied to the message body including UETR, providing cryptographic proof '
       'of message integrity at each gpi processing point. SWIFT gpi Observer analytics '
       'integration is preserved as QBITEL Bridge passes gpi SWIFTRef data through without '
       'modification.')

    qa(65, 'What is the QBITEL Bridge roadmap for quantum-safe tokenization?',
       'Quantum-safe tokenization is on the 2026 roadmap: ML-KEM-protected token vaults where '
       'the token-to-PAN mapping table is encrypted with post-quantum keys, preventing quantum '
       'decryption of tokenization databases. This extends current format-preserving encryption '
       'approaches (used by Visa Token Service, Mastercard Digital Enablement Service) with '
       'quantum-resistant key management. For institutions with existing tokenization infrastructure '
       '(Voltage, Thales payShield), QBITEL Bridge will integrate as the PQC key management '
       'layer for the token vault without requiring replacement of the tokenization platform.')

    return S


def build_pdf():
    import os
    output = '/Users/prabakarankannan/qbitel/docs/brochures/QBITEL_Banking_Pitch_QA_Guide.pdf'
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
