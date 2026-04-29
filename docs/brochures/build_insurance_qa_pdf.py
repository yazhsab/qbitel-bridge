"""
Build QBITEL Bridge Insurance Pitch Q&A Guide - Professional PDF
65 Q&As across 10 sections covering Insurance industry scenarios.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, NextPageTemplate

# Brand Colors
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


# Custom Flowables

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
        c.setFont('Helvetica-Bold', 14)
        numw = c.stringWidth(str(self.number), 'Helvetica-Bold', 14)
        c.drawString(22 - numw / 2, self.h / 2 - 7, str(self.number))
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 12)
        title_y = self.h - 22 if self.subtitle else self.h / 2 - 6
        c.drawString(54, title_y, self.title.upper())
        if self.subtitle:
            c.setFillColor(TEAL)
            c.setFont('Helvetica-Oblique', 8.5)
            c.drawString(54, 10, self.subtitle)


class QABlock(Flowable):
    def __init__(self, q_num, question, answer_lines, width=None):
        super().__init__()
        self.q_num = q_num
        self.question = question
        self.answer_lines = answer_lines
        self.w = width or CONTENT_W
        self.line_h = 13
        self.q_h = 28
        self.a_h = len(answer_lines) * self.line_h + 18
        self.h = self.q_h + self.a_h

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Answer background
        c.setFillColor(TEAL_LIGHT)
        c.rect(0, 0, self.w, self.a_h, fill=1, stroke=0)
        # Question background
        c.setFillColor(LIGHT_NAVY)
        c.rect(0, self.a_h, self.w, self.q_h, fill=1, stroke=0)
        # Q number badge
        c.setFillColor(GOLD)
        c.rect(0, self.a_h, 30, self.q_h, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 9)
        qw = c.stringWidth(f'Q{self.q_num}', 'Helvetica-Bold', 9)
        c.drawString(15 - qw / 2, self.a_h + 10, f'Q{self.q_num}')
        # Question text
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 9.5)
        c.drawString(38, self.a_h + 9, self.question)
        # A label
        c.setFillColor(TEAL)
        c.setFont('Helvetica-Bold', 9)
        c.drawString(8, self.a_h - 16, 'A:')
        # Answer lines
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 9)
        y = self.a_h - 16
        for line in self.answer_lines:
            c.drawString(24, y, line)
            y -= self.line_h
        # Left teal bar
        c.setFillColor(TEAL)
        c.rect(0, 0, 3, self.a_h, fill=1, stroke=0)


class ObjectionBlock(Flowable):
    def __init__(self, q_num, objection, rebuttal_lines, width=None):
        super().__init__()
        self.q_num = q_num
        self.objection = objection
        self.rebuttal_lines = rebuttal_lines
        self.w = width or CONTENT_W
        self.line_h = 13
        self.o_h = 28
        self.r_h = len(rebuttal_lines) * self.line_h + 18
        self.h = self.o_h + self.r_h

    def wrap(self, avail_w, avail_h):
        return self.w, self.h

    def draw(self):
        c = self.canv
        # Rebuttal background
        c.setFillColor(GREEN_LIGHT)
        c.rect(0, 0, self.w, self.r_h, fill=1, stroke=0)
        # Objection background
        c.setFillColor(RED_LIGHT)
        c.rect(0, self.r_h, self.w, self.o_h, fill=1, stroke=0)
        # Objection badge
        c.setFillColor(HexColor('#C0392B'))
        c.rect(0, self.r_h, 44, self.o_h, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 7.5)
        ow = c.stringWidth('OBJECTION', 'Helvetica-Bold', 7.5)
        c.drawString(22 - ow / 2, self.r_h + 11, 'OBJECTION')
        # Objection text
        c.setFillColor(HexColor('#8B1A1A'))
        c.setFont('Helvetica-Bold', 9)
        c.drawString(52, self.r_h + 10, f'"{self.objection}"')
        # Rebuttal label
        c.setFillColor(HexColor('#1A7A3A'))
        c.setFont('Helvetica-Bold', 9)
        c.drawString(8, self.r_h - 16, 'RESPONSE:')
        # Rebuttal lines
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 9)
        y = self.r_h - 16
        for line in self.rebuttal_lines:
            c.drawString(8, y, line)
            y -= self.line_h
        # Left green bar
        c.setFillColor(HexColor('#2E8B57'))
        c.rect(0, 0, 3, self.r_h, fill=1, stroke=0)


# Page Templates

def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 0.45 * inch, PAGE_W, 0.45 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, PAGE_H - 0.45 * inch - 3, PAGE_W, 3, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(MARGIN, PAGE_H - 0.32 * inch, 'QBITEL BRIDGE')
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(TEAL)
    canvas.drawString(MARGIN + 1.1 * inch, PAGE_H - 0.32 * inch,
                      'INSURANCE & REINSURANCE — PITCH Q&A GUIDE')
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
                      'Confidential — For Authorized Recipients Only  |  © 2026 QBITEL. All Rights Reserved.')
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
    p.moveTo(w * 0.60, h); p.lineTo(w, h); p.lineTo(w, h * 0.68)
    p.close()
    canvas.drawPath(p, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    p2 = canvas.beginPath()
    p2.moveTo(w * 0.76, h); p2.lineTo(w, h); p2.lineTo(w, h * 0.82)
    p2.close()
    canvas.drawPath(p2, fill=1, stroke=0)
    canvas.setFillColor(TEAL_DARK)
    canvas.rect(0, 0, w, 1.5 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, 1.5 * inch, w, 5, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 46)
    canvas.drawString(MARGIN, h * 0.70, 'QBITEL')
    canvas.setFillColor(TEAL)
    canvas.setFont('Helvetica-Bold', 46)
    canvas.drawString(MARGIN, h * 0.70 - 52, 'BRIDGE')
    canvas.setFillColor(GOLD)
    canvas.rect(MARGIN, h * 0.70 - 62, 3.0 * inch, 4, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 14)
    canvas.drawString(MARGIN, h * 0.70 - 90, 'INSURANCE & REINSURANCE')
    canvas.setFont('Helvetica-Bold', 18)
    canvas.drawString(MARGIN, h * 0.70 - 116, 'PITCH Q&A GUIDE')
    canvas.setFillColor(HexColor('#CCDDE8'))
    canvas.setFont('Helvetica-Oblique', 11)
    canvas.drawString(MARGIN, h * 0.70 - 142,
                      '65 Questions & Answers Across 10 Sections')

    # Stats
    stats = [
        ('65', 'Q&As Across\n10 Sections'),
        ('7', 'Hard Objections\nRebutted'),
        ('8', 'Competitive\nComparisons'),
        ('5', 'Regulatory\nFrameworks'),
    ]
    box_w = (CONTENT_W - 3 * 0.12 * inch) / 4
    bx_start = MARGIN
    by = h * 0.38
    bh = 0.85 * inch
    for i, (big, small) in enumerate(stats):
        bx = bx_start + i * (box_w + 0.12 * inch)
        bg_c = TEAL if i % 2 == 0 else LIGHT_NAVY
        canvas.setFillColor(bg_c)
        canvas.roundRect(bx, by, box_w, bh, 5, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.rect(bx, by + bh - 4, box_w, 4, fill=1, stroke=0)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica-Bold', 20)
        tw = canvas.stringWidth(big, 'Helvetica-Bold', 20)
        canvas.drawString(bx + (box_w - tw) / 2, by + bh - 30, big)
        canvas.setFont('Helvetica', 7.5)
        for j, line in enumerate(small.split('\n')):
            lw = canvas.stringWidth(line, 'Helvetica', 7.5)
            canvas.drawString(bx + (box_w - lw) / 2, by + bh - 48 - j * 11, line)

    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 9.5)
    canvas.drawString(MARGIN, 0.82 * inch,
                      'For Sales, Pre-Sales, and Solution Engineering Teams')
    canvas.setFont('Helvetica', 8.5)
    canvas.setFillColor(HexColor('#AABBCC'))
    canvas.drawString(MARGIN, 0.55 * inch,
                      'Confidential Internal Document  |  Version 1.0  |  February 2026')
    canvas.restoreState()


# Styles

def get_styles():
    s = {}
    s['body'] = ParagraphStyle(
        'body', fontName='Helvetica', fontSize=9.5, leading=14,
        textColor=DARK_TEXT, spaceAfter=6, alignment=TA_JUSTIFY)
    s['intro'] = ParagraphStyle(
        'intro', fontName='Helvetica-Oblique', fontSize=9, leading=13,
        textColor=MID_GREY, spaceAfter=10, alignment=TA_LEFT)
    return s


def build_story(styles):
    s = styles
    story = []
    q = 0  # global question counter

    def qa(question, answer_lines):
        nonlocal q
        q += 1
        return [QABlock(q, question, answer_lines), Spacer(1, 7)]

    def obj(objection, rebuttal_lines):
        nonlocal q
        q += 1
        return [ObjectionBlock(q, objection, rebuttal_lines), Spacer(1, 7)]

    # ─── Section 1: Business Case & Long-Term Risk ────────────────────────────
    story.append(SectionHeader(1, 'Business Case & Long-Term Risk',
                               'Why Insurance Cannot Defer the Quantum Transition'))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        'Insurance executives consistently ask: why act now on a threat that may be a decade away? '        'These eight questions address the urgency, the business case, and the board-level framing.',
        s['intro']))

    for item in qa('Why is insurance specifically more at risk from quantum computing than other industries?',
                   ['Insurance is uniquely exposed because policy data must remain confidential for 30-60 years',
                    '— far longer than any other financial data. A life insurance policy issued today contains',
                    'medical underwriting data, financial assets, and beneficiary information that must stay',
                    'private until 2075+. Quantum computers capable of breaking RSA-2048 are projected within',
                    '10-20 years. No other private sector industry has this mismatch between data lifetime',
                    'and encryption algorithm lifetime. Every policy issued today with classical encryption',
                    'is accumulating quantum debt against the day those records are decryptable.']):
        story.append(item)

    for item in qa('What is harvest-now, decrypt-later and why is it relevant to insurance right now?',
                   ['Harvest-now, decrypt-later (HNDL) is the practice of capturing encrypted data today',
                    'and storing it for decryption when quantum computers become available. Nation-state',
                    'actors — particularly those with long-term intelligence objectives — are systematically',
                    'targeting insurance data because a policyholder record is more complete and more',
                    'durable than hospital, financial, or government records. Insurance ACORD XML and X12',
                    'EDI 837 streams carry the highest-value PII/PHI combinations in any private sector',
                    'data flow. HNDL attacks are happening today — the threat is not theoretical.']):
        story.append(item)

    for item in qa('How do we quantify the cost of inaction to our board?',
                   ['Three categories of measurable cost: (1) Regulatory: NY DFS 500 fines of up to $1,000',
                    'per violation per day for NPI encryption failures; Solvency II sanctions for inadequate',
                    'ICT controls. (2) Litigation: policyholder class actions for future exposure of medical',
                    'and financial data — courts are increasingly treating foreseeable quantum risk as a',
                    'duty of care issue. (3) Brand: the first insurer to suffer a quantum-enabled breach of',
                    'life policy records will face existential reputational damage. Average insurance breach',
                    'cost is already $4.9M — post-quantum incidents will be orders of magnitude larger.']):
        story.append(item)

    for item in qa('When do we need to have quantum-safe encryption deployed by?',
                   ['NIST recommends complete PQC migration by 2030 for systems handling data with 10+ year',
                    'sensitivity requirements. For life insurance (30-60 year records), the urgency is higher.',
                    'NY DFS 500 and EIOPA are expected to issue quantum-safe encryption guidance by 2026-2027.',
                    'Early movers gain regulatory goodwill and avoid the compliance sprint. QBITEL Bridge',
                    'deploys in 4-6 hours — the deployment is not the bottleneck. The risk of waiting',
                    'every additional quarter is measurable in policyholder records exposed to HNDL capture.']):
        story.append(item)

    for item in qa('What is the ROI of deploying QBITEL Bridge for insurance?',
                   ['ROI operates on three tracks. Track 1 — Fraud Prevention: average 6-8% claims fraud rate',
                    'reduction translates to $4M-$40M+ annually for mid-to-large insurers. Track 2 — Regulatory:',
                    'avoided NY DFS 500/Solvency II sanctions and reduced compliance preparation costs (automated',
                    'evidence generation saves 200-400 hours per examination cycle). Track 3 — Breach Cost',
                    'Avoidance: average insurance breach costs $4.9M; quantum-enabled breaches of 30-year life',
                    'policy records will be categorically larger. QBITEL Bridge pricing for most insurers,'
                    'represents less than 15% of one avoided fraud incident.']):
        story.append(item)

    for item in qa('Which executive sponsor should own QBITEL Bridge in an insurance organisation?',
                   ['In most large insurers, the CISO owns the technical implementation. However, the business',
                    'case owner is typically the CRO (Chief Risk Officer) — because quantum exposure is an',
                    'enterprise risk, not just an IT risk. The CFO is engaged for long-term data liability.',
                    'The Chief Actuary is a key stakeholder for actuarial model protection. For Solvency II,',
                    'the Chief Compliance Officer is the relevant executive. QBITEL\'s executive briefing',
                    'is structured to address each stakeholder\'s specific concern simultaneously.']):
        story.append(item)

    for item in qa('How does this relate to our ongoing cloud migration and digital transformation?',
                   ['QBITEL Bridge is cloud-native and cloud-agnostic. If you\'re migrating from on-premises',
                    'policy admin to Guidewire Cloud or Duck Creek OnDemand, Bridge secures both the legacy',
                    'on-premises traffic and the cloud API traffic simultaneously. PQC encryption is built into',
                    'Bridge from the ground up — so your target-state cloud architecture is quantum-safe from',
                    'day one. Bridge also protects data in transit during migration, which is often the',
                    'highest-risk period in any cloud transformation programme.']):
        story.append(item)

    for item in qa('What happens to our existing cyber insurance coverage if we deploy QBITEL Bridge?',
                   ['QBITEL Bridge typically strengthens your cyber insurance negotiating position. Carriers',
                    'are increasingly requiring evidence of encryption adequacy and MFA as minimum conditions',
                    'for coverage. Bridge automated compliance reporting provides the evidence insurers',
                    'require for underwriting. Several QBITEL clients have used Bridge deployment evidence',
                    'to negotiate premium reductions of 10-18% on their cyber insurance renewals by',
                    'demonstrating quantum-safe controls and autonomous threat response capabilities.']):
        story.append(item)

    story.append(PageBreak())

    # ─── Section 2: Technical Architecture ───────────────────────────────────
    story.append(SectionHeader(2, 'Technical Architecture',
                               'How QBITEL Bridge Works in an Insurance Environment'))
    story.append(Spacer(1, 8))

    for item in qa('How does QBITEL Bridge insert into our insurance network without disruption?',
                   ['Bridge operates as a transparent network-layer proxy using passive tap or inline insertion.',
                    'For insurance, the typical insertion points are: (1) between the policy admin system',
                    '(Guidewire/Duck Creek/mainframe) and EDI/API integration layers, (2) on the perimeter',
                    'between internal systems and trading partner connections (carriers, reinsurers, TPAs),',
                    'and (3) in front of mainframe network interfaces for TN3270e protection. No agents,',
                    'no application changes, no downtime. Initial go-live in 4-6 hours.']):
        story.append(item)

    for item in qa('What PQC algorithms does QBITEL Bridge use and are they NIST-approved?',
                   ['Yes — QBITEL Bridge uses only NIST-finalised algorithms: ML-KEM (FIPS 203, formerly',
                    'CRYSTALS-Kyber) for key encapsulation, ML-DSA (FIPS 204, formerly CRYSTALS-Dilithium)',
                    'for digital signatures, and SLH-DSA (FIPS 205, formerly SPHINCS+) for stateless',
                    'hash-based signatures on long-term actuarial data. We do not use pre-standard or',
                    'proprietary algorithms. We also support hybrid classical+PQC mode for trading partners',
                    'not yet PQC-capable, ensuring backward compatibility during the transition.']):
        story.append(item)

    for item in qa('How does QBITEL Bridge handle key management across long-term policy data?',
                   ['Key management is hardware-bound via FIPS 140-3 Level 3 HSMs. For long-term policy',
                    'data, Bridge implements cryptographic agility — the ability to re-encrypt data with',
                    'new PQC algorithm parameters without operational disruption. Each policy record class',
                    'has a configurable key rotation schedule aligned to its retention horizon. Key',
                    'provenance logs are maintained for every policy record, creating auditable evidence',
                    'of encryption history suitable for regulatory examination and e-discovery.']):
        story.append(item)

    for item in qa('What is the performance impact on our claims processing throughput?',
                   ['Performance impact is less than 1.2ms per session establishment and under 0.8ms added',
                    'latency on X12 837 EDI streams. QBITEL Bridge has been validated in environments',
                    'processing 2M+ policy transactions per day on IBM System z mainframe with zero',
                    'measurable throughput degradation. For real-time claims adjudication, the inline',
                    'fraud detection adds less than 1.2ms — well within SLA parameters. Throughput',
                    'testing data is available under NDA for specific carrier environments.']):
        story.append(item)

    for item in qa('Does QBITEL Bridge support high-availability and disaster recovery for insurance operations?',
                   ['Yes. Bridge deploys in active-active high-availability clustering with 99.99% availability',
                    'SLA. For insurance environments with strict RTO/RPO requirements (particularly for',
                    'mainframe policy systems that cannot tolerate any unplanned downtime), Bridge supports',
                    'geographic redundancy with sub-second failover. HSM key material is replicated across',
                    'HA clusters using FIPS 140-3 Level 3 certified key replication protocols. Disaster',
                    'recovery documentation is available for Solvency II ORSA and NY DFS 500 evidence.']):
        story.append(item)

    for item in qa('Can QBITEL Bridge be deployed in a hybrid on-premises and cloud insurance environment?',
                   ['Yes — hybrid deployment is the most common insurance scenario. Most large insurers',
                    'have on-premises mainframe policy admin (IBM System z), on-premises claims management',
                    '(Guidewire/Duck Creek), and cloud-based analytics or customer portal. Bridge deploys',
                    'across all three simultaneously, maintaining consistent PQC encryption policy across',
                    'on-premises, co-location, and cloud environments. AWS, Azure, and GCP deployments',
                    'are all supported with cloud-native Bridge instances that integrate with your',
                    'existing cloud security posture management tooling.']):
        story.append(item)

    story.append(PageBreak())

    # ─── Section 3: Policy Data Protection ───────────────────────────────────
    story.append(SectionHeader(3, 'Policy Data Protection',
                               'Life Insurance, Annuities, Long-Term Care, and Health'))
    story.append(Spacer(1, 8))

    for item in qa('How does QBITEL Bridge classify different types of policyholder data?',
                   ['Bridge uses AI-powered deep packet inspection to classify data by policy type and',
                    'retention horizon: Life/annuity records (30-60 years) receive ML-KEM-1024 protection;',
                    'health insurance records (10-20 years) receive ML-KEM-768; P&C records (7-10 years)',
                    'receive ML-KEM-512; actuarial model data (indefinite) receives ML-KEM-1024 + SLH-DSA.',
                    'Classification is based on protocol context (ACORD XML policy type codes, X12 834',
                    'line of business indicators, HL7 message types) and is configurable by the insurer',
                    'to reflect their specific data governance and retention policies.']):
        story.append(item)

    for item in qa('How does QBITEL Bridge protect whole life and universal life policy data specifically?',
                   ['Whole life and universal life policies receive the highest PQC protection tier due to',
                    'their 30-100 year duration. Bridge applies ML-KEM-1024 key encapsulation to all',
                    'in-transit policy data identified as whole or universal life by ACORD XML policy',
                    'type codes or internal classification schemes. ML-DSA-87 signatures are applied to',
                    'all policy modifications, ensuring tamper-evident history for the full policy lifetime.',
                    'Key provenance logs support the audit trail required for death benefit claims',
                    'resolution, which may occur 50+ years after policy issuance.']):
        story.append(item)

    for item in qa('What protection does QBITEL Bridge provide for annuity and retirement product data?',
                   ['Annuity data — including variable annuity investment selections, guaranteed minimum',
                    'benefit terms, and beneficiary designations — is classified as critical-tier data',
                    'by Bridge and protected with ML-KEM-1024. For variable annuity products with FIX',
                    'protocol integration to investment platforms, Bridge secures FIX sessions with',
                    'quantum-safe encryption. Annuity payment streams (ACH/wire transfer authorisations)',
                    'receive ISO 20022 message-level PQC protection. Policyholder consent and election',
                    'records receive cryptographic integrity signing for regulatory audit purposes.']):
        story.append(item)

    for item in qa('How is long-term care insurance data handled given its extreme sensitivity?',
                   ['Long-term care insurance data is among the most sensitive: it contains medical',
                    'assessments, functional limitation documentation, care coordination records, and',
                    'financial eligibility information spanning decades. Bridge classifies LTC data',
                    'at the critical tier, applying ML-KEM-1024 + ML-DSA-87 to all LTC policy data',
                    'flows. HL7 messages from care providers and X12 837I claims from nursing facilities',
                    'receive both PQC encryption and fraud detection for inflated or phantom billing.',
                    'HIPAA minimum necessary enforcement is applied at the protocol layer.']):
        story.append(item)

    for item in qa('How does Bridge handle medical underwriting data and re-underwriting workflows?',
                   ['Medical underwriting data — the most sensitive component of life insurance applications',
                    '— is identified by Bridge in ACORD XML application submission flows and HL7 laboratory',
                    'and physician report exchanges. All medical underwriting data receives critical-tier',
                    'PQC protection. Re-underwriting workflows that pull historical medical records from',
                    'policy admin systems are monitored for unusual data volumes and access patterns.',
                    'Bulk access to medical underwriting records (a key indicator of APT data exfiltration)',
                    'triggers an immediate alert and autonomous containment response.']):
        story.append(item)

    for item in qa('What about personal auto and homeowners insurance — shorter policy terms — do they still need PQC?',
                   ['P&C insurance (personal auto, homeowners) has shorter policy terms (1-3 years)',
                    'but policyholder records often span decades of continuous coverage relationships.',
                    'A homeowner insured from age 30 to 65 accumulates 35 years of address history,',
                    'claims history, vehicle and property data, and payment information — a rich identity',
                    'profile valuable for identity fraud and social engineering. QBITEL Bridge applies',
                    'high-tier PQC to P&C policyholder records recognising the cumulative data value',
                    'even if individual policy terms are short.']):
        story.append(item)

    for item in qa('How does QBITEL Bridge protect group life and group health insurance data?',
                   ['Group insurance data requires special handling because it crosses employer/HR',
                    'and insurer boundaries. Bridge secures X12 EDI 834 enrollment and disenrollment',
                    'transactions between employers, benefits administrators, and carriers with ML-KEM',
                    'encryption. Group census data — which includes member names, SSNs, dates of birth,',
                    'and dependent information — is classified as high-tier PII. Employer-sponsored',
                    'health data including X12 837 medical claims is protected with HIPAA-tier controls',
                    'including minimum necessary enforcement and PHI boundary detection.']):
        story.append(item)

    for item in qa('How does Bridge handle policyholder consent records under GDPR and CCPA?',
                   ['Bridge supports consent management infrastructure by providing cryptographic',
                    'integrity verification for consent records — ensuring that consent evidence is',
                    'tamper-evident and auditable. For GDPR (EU policyholders) and CCPA (California),',
                    'Bridge applies data minimisation controls at the protocol layer by detecting and',
                    'flagging unnecessary PII transmission in API and EDI flows. Right-to-erasure',
                    'workflows are supported through cryptographic key deletion for data encrypted',
                    'under tenant-specific key material — rendering data unrecoverable on erasure request',
                    'without requiring physical data deletion across backup systems.']):
        story.append(item)

    story.append(PageBreak())

    # ─── Section 4: Claims & Fraud Prevention ────────────────────────────────
    story.append(SectionHeader(4, 'Claims & Fraud Prevention',
                               'Real-Time Detection Before Adjudication'))
    story.append(Spacer(1, 8))

    for item in qa('How does QBITEL Bridge detect synthetic identity fraud in claims submissions?',
                   ['Bridge analyses X12 EDI 837 claim submission streams in real time, cross-referencing',
                    'claimant identity attributes (SSN, date of birth, name, address) against behavioral',
                    'baselines. Synthetic identity patterns include: SSNs issued after the claimant\'s',
                    'stated birth date, name/SSN combinations that appear across multiple unrelated claims,',
                    'policy inception-to-first-claim timing anomalies typical of synthetic identity rings,',
                    'and address patterns inconsistent with the claimed injury type or provider location.',
                    'Confirmed synthetic identity patterns trigger autonomous claim hold with immediate',
                    'SIU team escalation — before any payment is initiated.']):
        story.append(item)

    for item in qa('Can QBITEL Bridge detect organised fraud rings with multiple colluding claimants?',
                   ['Yes. Bridge\'s claims network analysis identifies coordinated fraud rings by mapping',
                    'relationships across claim submissions: shared providers, shared legal representatives,',
                    'shared vehicle or property addresses, and time-correlated claim filing patterns.',
                    'These network graphs are built in real time from X12 EDI 837 and ACORD XML data',
                    'streams. When a new claim arrives that connects to an existing suspicious network,',
                    'it is flagged before processing. Staged accident rings — a major source of auto',
                    'claims fraud — are particularly detectable because they generate characteristic',
                    'multi-claimant, same-incident filing patterns within short time windows.']):
        story.append(item)

    for item in qa('How does fraud detection work without impacting legitimate claims processing speed?',
                   ['Bridge\'s fraud detection operates inline with less than 1.2ms added latency — below',
                    'the threshold of any SLA impact in claims adjudication systems. The detection model',
                    'runs in hardware-accelerated inference on the Bridge appliance and does not require',
                    'round-trips to external systems. Only claims that match fraud patterns above a',
                    'configurable confidence threshold are held for review — the vast majority of clean',
                    'claims pass through with no delay. The hold-and-review workflow integrates directly',
                    'with your existing claims management system via API notification.']):
        story.append(item)

    for item in qa('How does Bridge handle medical billing fraud in X12 837 professional and institutional claims?',
                   ['Bridge performs real-time CPT/ICD code combination analysis on X12 837P and 837I',
                    'submissions. Upcoding detection compares billed codes against the provider\'s',
                    'historical billing profile — flagging codes billed at frequencies significantly',
                    'above peer norms. Unbundling detection identifies service component combinations',
                    'that should be billed as a single composite code. Phantom billing is detected by',
                    'cross-referencing procedure combinations that are medically implausible (e.g., surgical',
                    'and non-surgical codes for the same body part on the same date). All detections',
                    'generate evidence-grade audit records for SIU investigation.']):
        story.append(item)

    for item in qa('Can QBITEL Bridge integrate with our existing SIU and fraud analytics platforms?',
                   ['Yes. Bridge integrates with SIU workflow systems via REST API and SIEM/SOAR platforms',
                    'via syslog or native connector. Fraud detection events are formatted as structured',
                    'JSON enriched with the specific protocol indicators that triggered the detection.',
                    'For clients using FRISS, Shift Technology, or Verisk fraud platforms, Bridge provides',
                    'a complementary real-time detection layer that feeds enriched claim indicators to',
                    'the post-processing analytics — improving those platforms model accuracy while',
                    'adding the pre-payment blocking capability they lack.']):
        story.append(item)

    for item in qa('How does QBITEL Bridge protect against reinsurance wire fraud and SWIFT manipulation?',
                   ['Bridge applies real-time integrity verification to all SWIFT MT and MX messages in',
                    'reinsurance settlement flows. Verification checks include: payment amount deviation',
                    'from cedant loss bordereau data, beneficiary account (IBAN/BIC) changes within',
                    'short periods before settlement, message routing anomalies, and BEC indicators',
                    'in the instruction chain (unusual sending addresses, urgency language patterns).',
                    'For confirmed anomalies, Bridge issues an automatic settlement hold and escalation',
                    'alert before the SWIFT message is transmitted — preventing wire fraud rather than',
                    'merely detecting it after funds have departed.']):
        story.append(item)

    story.append(PageBreak())

    # ─── Section 5: Compliance & Regulatory ──────────────────────────────────
    story.append(SectionHeader(5, 'Compliance & Regulatory',
                               'Solvency II, NY DFS 500, NAIC, HIPAA, GDPR'))
    story.append(Spacer(1, 8))

    for item in qa('How does QBITEL Bridge satisfy NY DFS 500 Section 500.15 encryption requirements?',
                   ['Section 500.15 requires encryption of nonpublic information (NPI) in transit and at',
                    'rest. Bridge provides quantum-safe encryption (ML-KEM per NIST FIPS 203) for all',
                    'NPI in transit — including ACORD XML policy data, X12 EDI 837 claims, SWIFT',
                    'settlement messages, and TN3270e mainframe sessions. For NPI at rest, Bridge',
                    'provides encryption gateway controls and key management. Bridge generates continuous',
                    'evidence of encryption coverage including protocol inventory, encryption status',
                    'per data category, and exception reporting — formatted for NY DFS examination.']):
        story.append(item)

    for item in qa('What does QBITEL Bridge generate for a Solvency II SFCR cybersecurity section?',
                   ['Bridge generates a structured evidence package including: (1) Protocol security',
                    'inventory — all insurance protocol types and their encryption status, (2) Data',
                    'classification report — NPI/PHI by policy line and encryption tier, (3) Threat',
                    'detection log — summary of threats detected and response times over the reporting',
                    'period, (4) Control effectiveness metrics — autonomous response rate, false positive',
                    'rate, coverage percentage, (5) Key management audit trail — FIPS 140-3 Level 3',
                    'HSM attestation and key rotation log. This package directly addresses EIOPA\'s',
                    'Guidelines on ICT Security and Risk Management for Solvency II compliance.']):
        story.append(item)

    for item in qa('How does Bridge support NAIC Cybersecurity Model Law compliance across multiple states?',
                   ['The NAIC Model Law (adopted in 24+ states with slight variations) requires insurers',
                    'to maintain a comprehensive cybersecurity program with appropriate administrative,',
                    'technical, and physical safeguards for NPI. Bridge satisfies the technical safeguard',
                    'requirement by providing quantum-safe encryption and access monitoring for all NPI',
                    'in transit. The automated evidence vault provides the written cybersecurity program',
                    'documentation and annual review evidence required by Model Law Section IV. Bridge',
                    'reports are formatted to address both the standard Model Law text and state-specific',
                    'variations in key states (New York, California, Ohio, Massachusetts).']):
        story.append(item)

    for item in qa('How does QBITEL Bridge satisfy HIPAA Security Rule requirements for health insurers?',
                   ['For HIPAA-covered health insurers, Bridge satisfies multiple Security Rule requirements:',
                    '(1) 164.312(a)(2)(iv) Encryption and Decryption — ML-KEM encryption of all PHI in',
                    'transit via HL7 and X12 837 flows; (2) 164.312(e)(1) Transmission Security — PQC',
                    'wrapping of all PHI transmitted over electronic networks; (3) 164.312(b) Audit',
                    'Controls — comprehensive access and transmission logging with retention; (4)',
                    '164.308(a)(1) Risk Analysis — protocol risk assessment data supports annual',
                    'Security Risk Assessment requirements. Bridge generates HIPAA-formatted evidence',
                    'for Security Rule compliance documentation.']):
        story.append(item)

    for item in qa('Can QBITEL Bridge help us respond to a regulatory cyber examination of our insurance operations?',
                   ['Yes. Bridge\'s Evidence Vault provides on-demand generation of structured regulatory',
                    'examination packages. For NY DFS 500 examinations, Bridge generates the standard',
                    'cybersecurity documentation checklist. For Solvency II EIOPA inspections, it provides',
                    'ORSA-formatted cybersecurity control evidence. For NAIC examination, it generates',
                    'the cybersecurity program documentation required by Model Law Section IV. Most',
                    'importantly, Bridge provides real-time evidence of continuous monitoring — regulators',
                    'increasingly expect to see ongoing controls evidence, not just point-in-time assessments.']):
        story.append(item)

    for item in qa('How does QBITEL Bridge handle cross-border data flows for insurers operating in EU and US?',
                   ['For multinational insurers operating under both GDPR (EU) and US state regulations',
                    '(NY DFS 500, NAIC), Bridge enforces jurisdiction-specific encryption and data handling',
                    'policies based on data origin and policyholder residence. EU policyholder data receives',
                    'GDPR-tier controls (including data minimisation enforcement and consent verification).',
                    'US policyholder data receives NAIC/NY DFS 500-tier controls. Cross-border reinsurance',
                    'data flows — which often carry EU policyholder aggregate data to Bermuda or US',
                    'reinsurers — receive hybrid policy enforcement with documentation for both jurisdictions.']):
        story.append(item)

    for item in qa('Does QBITEL Bridge support IFRS 17 data integrity requirements?',
                   ['Yes. IFRS 17 requires insurers to maintain reliable data on insurance contract',
                    'liabilities — including measurement assumptions, expected cash flows, and discount',
                    'rate calculations. Bridge provides ML-DSA cryptographic integrity verification on',
                    'actuarial data flows used in IFRS 17 calculations, creating a tamper-evident audit',
                    'trail from source data through to reported liability figures. This supports external',
                    'audit requirements for IFRS 17 and reduces the risk of financial restatement due',
                    'to undetected data manipulation in the actuarial calculation pipeline.']):
        story.append(item)

    for item in qa('What evidence does QBITEL Bridge provide for PCI-DSS compliance for premium payments?',
                   ['For insurers that process premium payments via credit/debit card, Bridge satisfies',
                    'PCI-DSS v4.0 Requirement 4 (encryption of cardholder data in transit) with ML-KEM',
                    'quantum-safe encryption — exceeding the current TLS 1.2 minimum. Requirement 3',
                    'controls for stored payment data are supported through HSM-backed key management.',
                    'Bridge generates continuous PCI-DSS evidence including encryption status of all',
                    'cardholder data flows, key management audit logs, and network segmentation',
                    'documentation for the cardholder data environment (CDE) — directly supporting',
                    'your QSA assessment.']):
        story.append(item)

    story.append(PageBreak())

    # ─── Section 6: Legacy System Integration ────────────────────────────────
    story.append(SectionHeader(6, 'Legacy System Integration',
                               'Policy Admin Systems and Mainframe Environments'))
    story.append(Spacer(1, 8))

    for item in qa('Our mainframe runs COBOL policy admin — how does QBITEL Bridge integrate without code changes?',
                   ['Bridge uses transparent network-layer proxying — it inserts between the mainframe network',
                    'interface and the connected systems (claims processors, agent portals, actuarial workstations)',
                    'without any changes to the COBOL application or the mainframe OS configuration. From the',
                    'mainframe\'s perspective, it is communicating normally with connected systems. Bridge',
                    'operates entirely at the network layer, intercepting TN3270e and other legacy protocol',
                    'sessions, wrapping them in ML-KEM encryption, and forwarding them to the destination.',
                    'The installation is completed in a scheduled maintenance window of 4-6 hours.']):
        story.append(item)

    for item in qa('We use IBM System z with RACF and ACF2 security. How does Bridge coexist with these?',
                   ['Bridge complements — not replaces — mainframe-native security controls like RACF and ACF2.',
                    'RACF and ACF2 govern authorisation and access control within the mainframe. Bridge',
                    'governs the encryption and integrity of data leaving the mainframe across the network.',
                    'These are orthogonal controls. Bridge\'s session monitoring can be integrated with',
                    'RACF audit logs to provide a unified view of mainframe access and network data flows,',
                    'strengthening the overall security posture without replacing or conflicting with',
                    'existing mainframe security architecture.']):
        story.append(item)

    for item in qa('How does QBITEL Bridge handle our Guidewire integration points?',
                   ['Bridge integrates with Guidewire Cloud Platform via the Guidewire Integration Framework',
                    '(GIF) and Guidewire Integration Gateway. All API traffic between Guidewire PolicyCenter,',
                    'ClaimCenter, and BillingCenter and external systems (reinsurers, TPAs, EDI partners)',
                    'is wrapped with ML-KEM encryption at the API gateway layer. For Guidewire Edge',
                    'deployments with cloud-native APIs, Bridge integrates with the Guidewire API',
                    'management layer. For on-premises Guidewire, Bridge inserts inline at the network',
                    'perimeter. No Guidewire configuration changes are required in either deployment model.']):
        story.append(item)

    for item in qa('What about our legacy Duck Creek on-premises deployment — can Bridge secure it?',
                   ['Yes. Duck Creek on-premises deployments are a common Bridge use case. Bridge inserts',
                    'inline between Duck Creek and its external integration points — EDI clearinghouses,',
                    'reinsurance portals, and agent management systems. Duck Creek\'s REST API endpoints',
                    'receive ML-KEM session wrapping. EDI transactions processed through Duck Creek\'s',
                    'EDI integration receive X12-aware PQC wrapping. The Duck Creek database tier is',
                    'out of scope for Bridge (Bridge operates on network flows), but Bridge can secure',
                    'the application-to-database network segment if required.']):
        story.append(item)

    for item in qa('We have dozens of legacy trading partner EDI connections at various encryption levels — how does Bridge handle this?',
                   ['Bridge discovers all EDI trading partner connections automatically during the Phase 1',
                    'assessment and maps their current encryption level (unencrypted, TLS 1.0/1.1/1.2, AS2,',
                    'SFTP). Each trading partner connection is then individually upgraded to PQC wrapping',
                    'at the Bridge layer — the trading partner continues to operate with whatever protocol',
                    'they currently use, but the session between them and your environment is PQC-protected',
                    'at the Bridge insertion point. For partners who prefer end-to-end PQC, Bridge',
                    'facilitates hybrid PQC certificate distribution for their trading partner connections.']):
        story.append(item)

    story.append(PageBreak())

    # ─── Section 7: Reinsurance & Capital Markets ─────────────────────────────
    story.append(SectionHeader(7, 'Reinsurance & Capital Markets',
                               'SWIFT, ILS, CAT Bonds, and Treaty Security'))
    story.append(Spacer(1, 8))

    for item in qa('How does QBITEL Bridge secure our cedant relationships and bordereau exchanges?',
                   ['Bridge secures all cedant data exchange channels including: treaty bordereau transmissions',
                    '(premium, loss, and commission data in ACORD XML or proprietary formats), loss advices',
                    'via SWIFT MT202/MT103, aggregate loss reporting for Excess of Loss treaty monitoring,',
                    'and cat model data exchanges with RMS/Moody\'s. All bordereau data is wrapped with',
                    'ML-KEM before transmission and ML-DSA integrity-signed, creating tamper-evident',
                    'records for treaty audit and commutation proceedings. Bulk bordereau anomalies',
                    '(loss amounts inconsistent with treaty parameters) are flagged in real time.']):
        story.append(item)

    for item in qa('What protection does QBITEL Bridge provide for catastrophe bond trigger data?',
                   ['CAT bond trigger verification involves transmission of parametric or indemnity trigger',
                    'data — event intensity readings, modelled loss estimates, or actual loss development',
                    'figures — between cedants, ILS managers, and trustees. Bridge applies ML-DSA',
                    'cryptographic chaining to trigger data from generation (model output or actual loss',
                    'reports) through to the ILS trustee verification. This creates a tamper-evident audit',
                    'trail that prevents retroactive manipulation of trigger data — a significant concern',
                    'for large CAT events where trigger timing is disputed between cedant and investors.']):
        story.append(item)

    for item in qa('How does Bridge handle FIX protocol for ILS trading desk operations?',
                   ['Insurance-linked securities trading uses FIX protocol for order management, execution',
                    'reporting, and position management. Bridge provides ML-KEM quantum-safe session',
                    'encryption for FIX connections between ILS trading desks and brokers or trading',
                    'venues. FIX message integrity is verified using ML-DSA, detecting order manipulation',
                    'or position reporting anomalies. For retrocession trading platforms, Bridge secures',
                    'the FIX integration between the reinsurer\'s trading system and external counterparties',
                    'including ILS fund managers and retrocession brokers.']):
        story.append(item)

    for item in qa('Can QBITEL Bridge protect our retrocession programme data?',
                   ['Yes. Retrocession data — the most sensitive layer of reinsurance because it reveals',
                    'aggregate net retained exposure — receives the highest PQC tier in Bridge. Retrocession',
                    'treaty terms, cession schedules, and aggregate bordereau are protected with ML-KEM-1024',
                    'and long-term key retention matching the treaty duration. SWIFT settlement messages',
                    'for retrocession premium and loss payments receive real-time integrity verification.',
                    'Access to retrocession data within the organisation is monitored for unusual patterns',
                    'indicating insider threat or lateral movement targeting net exposure intelligence.']):
        story.append(item)

    for item in qa('How does QBITEL Bridge support emerging insurance capital market structures like sidecars and collateralised re?',
                   ['Sidecars and collateralised reinsurance structures involve highly confidential financial',
                    'terms between cedants and capital providers. Bridge secures data exchange in these',
                    'structures via: (1) ML-KEM-encrypted API channels between cedant systems and',
                    'collateral management platforms, (2) PQC-protected trust account reporting and',
                    'collateral release messaging, (3) ML-DSA integrity verification of loss development',
                    'reports used for commutation and collateral release decisions. For fronting',
                    'arrangements involving captives, Bridge monitors for data flows inconsistent with',
                    'stated fronting relationships — a key fraud indicator in captive structures.']):
        story.append(item)

    story.append(PageBreak())

    # ─── Section 8: Operations & SLA ──────────────────────────────────────────
    story.append(SectionHeader(8, 'Operations & SLA',
                               'Support, Performance, and Service Commitments'))
    story.append(Spacer(1, 8))

    for item in qa('What SLA does QBITEL commit to for insurance-critical systems?',
                   ['QBITEL Bridge commits to 99.99% availability (four nines) for production insurance',
                    'environments — approximately 52 minutes of permitted downtime per year. This SLA',
                    'is backed by active-active HA clustering with geographic redundancy. For mainframe',
                    'environments with zero-downtime requirements, Bridge\'s passive tap mode provides',
                    'monitoring and detection with automatic failover to a non-intercepting path in',
                    'the event of a Bridge component failure — ensuring that policy transactions are',
                    'never blocked by a Bridge availability issue. Full SLA terms are available in',
                    'the QBITEL Enterprise Service Agreement.']):
        story.append(item)

    for item in qa('What does the ongoing support model look like for insurance clients?',
                   ['Insurance clients receive dedicated support including: (1) Named Customer Success',
                    'Manager with insurance industry background, (2) 24/7 security operations support',
                    'via the QBITEL SOC for threat escalations, (3) Regulatory examination support —',
                    'QBITEL engineers available to participate in regulatory examinations requiring',
                    'technical explanation of controls, (4) Quarterly threat intelligence briefings',
                    'specific to insurance industry attack campaigns, (5) Annual PQC algorithm review',
                    'to ensure continued alignment with NIST guidance and emerging post-quantum standards.']):
        story.append(item)

    for item in qa('How are software updates and PQC algorithm updates managed without disrupting insurance operations?',
                   ['Bridge updates are delivered via zero-downtime rolling upgrade across HA cluster',
                    'nodes. PQC algorithm updates — which may be required as NIST\'s post-quantum',
                    'standardisation process continues — are delivered as cryptographic module updates',
                    'with backward compatibility maintained. Insurers are notified a minimum of 90 days',
                    'before any mandatory algorithm update. For mainframe environments, all updates',
                    'are coordinated with the insurer\'s change management process and tested in a',
                    'staging environment before production deployment. No COBOL or policy admin',
                    'system changes are ever required for a Bridge software update.']):
        story.append(item)

    for item in qa('What forensic evidence does QBITEL Bridge retain for insurance regulatory investigations?',
                   ['Bridge maintains a forensic evidence vault with configurable retention periods aligned',
                    'to insurance regulatory requirements — minimum 7 years for most US states, longer',
                    'for Solvency II. The vault stores: encrypted session metadata (not content) for',
                    'all protected protocol sessions, fraud detection event records with full indicator',
                    'evidence, threat response audit trails, compliance control status snapshots, and',
                    'key management operation logs. All vault records are ML-DSA integrity-signed at',
                    'creation and stored with cryptographic chain-of-custody for admissibility in',
                    'regulatory proceedings and civil litigation.']):
        story.append(item)

    story.append(PageBreak())

    # ─── Section 9: Hard Objections ────────────────────────────────────────────
    story.append(SectionHeader(9, 'Hard Objections',
                               'Seven Objections — And How to Answer Them'))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        'These are the objections most likely to be raised by sceptical insurance executives and technology teams. '        'Each response is calibrated to address the specific concern while maintaining factual accuracy.',
        s['intro']))

    for item in obj(
        'Quantum computers are 10+ years away — we have time',
        ['The harvest-now, decrypt-later attack is happening today. Nation-state actors are capturing',
         'your encrypted ACORD XML and X12 EDI streams right now and storing them for future decryption.',
         'A life insurance policy issued this quarter with RSA encryption contains policyholder data',
         'that must stay confidential for 60 years — well within any credible quantum timeline.',
         'NIST has already finalised PQC standards. NY DFS 500 and EIOPA guidance is expected by 2027.',
         'Early movers avoid a compliance sprint. QBITEL Bridge deploys in 4-6 hours — the risk',
         'of waiting another quarter is real, measurable, and growing with every policy issued.']):
        story.append(item)

    for item in obj(
        'Our cloud provider encrypts everything — we are already protected',
        ['Cloud provider encryption protects data at rest in their storage infrastructure. It does not',
         'protect ACORD XML in transit between your on-premises policy admin system and a reinsurance',
         'partner. It does not protect TN3270e sessions to your mainframe. It does not protect X12 837',
         'EDI submissions from TPAs who connect via AS2 or SFTP. Cloud encryption also uses the same',
         'classical RSA/ECC algorithms that are vulnerable to quantum attack. QBITEL Bridge fills the',
         'protocol-layer gap that cloud encryption cannot reach — particularly for the legacy insurance',
         'infrastructure where most of your sensitive policy data actually lives.']):
        story.append(item)

    for item in obj(
        'We are already GDPR compliant',
        ['GDPR compliance is a baseline requirement, not a security standard. GDPR Article 32 requires',
         '"appropriate technical measures" — but does not define what "appropriate" means for quantum',
         'threats. Post-NIST PQC (August 2024), a GDPR regulator asking whether your encryption is',
         '"state of the art" has a new benchmark to point to. More importantly, GDPR compliance',
         'documents your obligations — QBITEL Bridge delivers the technical controls to meet them.',
         'A GDPR audit of your current encryption posture will reveal RSA/ECC on long-term policyholder',
         'data flows — a finding that is increasingly difficult to defend to a Data Protection Authority.']):
        story.append(item)

    for item in obj(
        'Our actuarial data is not a hacker target — it is internal',
        ['Actuarial models represent the most concentrated insurance IP in your organisation — mortality',
         'tables, pricing algorithms, and loss development factors that took decades to build. APT groups',
         'targeting insurers specifically seek actuarial data for competitive intelligence (sold to',
         'foreign insurance operations), for financial fraud (manipulating reserve estimates), and for',
         'regulatory arbitrage (understanding where risk pricing creates exploitable gaps). The insider',
         'threat is also significant — a departing chief actuary or pricing analyst with bulk data access',
         'represents a major IP theft risk. "Internal" data that moves across networks between actuarial',
         'servers and workstations is exactly the traffic QBITEL Bridge protects.']):
        story.append(item)

    for item in obj(
        'Solvency II does not require quantum-safe cryptography',
        ['Solvency II Article 98 requires ICT risk management with "appropriate security measures"',
         'commensurate with risk. EIOPA\'s Guidelines on ICT Security require encryption that is',
         '"fit for purpose" for the data\'s sensitivity and duration. Post-NIST PQC finalisation in',
         'August 2024, EIOPA supervisors in France, Germany, and the Netherlands are already',
         'asking about quantum-safe cryptography in supervisory conversations. The SFCR cybersecurity',
         'section now has a higher bar. Waiting for explicit Solvency II PQC guidance before acting',
         'means being behind the regulatory curve when that guidance arrives.']):
        story.append(item)

    for item in obj(
        'Too expensive to justify to the board',
        ['The ROI case is straightforward: a 1% reduction in claims fraud for a $1B claims portfolio',
         'saves $10M annually. QBITEL Bridge pricing for a carrier of that size is well below that',
         'figure. Add regulatory examination cost reduction (200-400 hours per examination cycle),',
         'breach cost avoidance ($4.9M average insurance breach), and cyber insurance premium',
         'reduction (10-18% for clients demonstrating PQC controls). The board question is not',
         '"can we afford QBITEL Bridge?" — it is "can we afford the first quantum-enabled breach',
         'of our life insurance policyholder database, and the subsequent regulatory sanction?"']):
        story.append(item)

    for item in obj(
        'Our reinsurance treaties do not mention quantum risk',
        ['Reinsurance treaties are largely silent on cyber risk generally — yet cyber losses are now',
         'a significant component of reinsurance claims. Quantum risk is the next frontier of this',
         'evolution. More immediately: the financial crime risk in reinsurance settlements (SWIFT',
         'wire fraud, bordereau manipulation, CAT bond trigger tampering) is well-documented and',
         'financially material today. QBITEL Bridge addresses both the current financial crime risk',
         'and the emerging quantum risk in reinsurance data flows. A reinsurer that experiences a',
         '$100M+ SWIFT settlement fraud will find that the absence of quantum-safe controls is',
         'mentioned prominently in the post-incident regulatory examination.']):
        story.append(item)

    story.append(PageBreak())

    # ─── Section 10: Competitive ───────────────────────────────────────────────
    story.append(SectionHeader(10, 'Competitive Differentiation',
                               'QBITEL Bridge vs. Alternative Approaches'))
    story.append(Spacer(1, 8))

    for item in qa('How does QBITEL Bridge compare to simply upgrading to TLS 1.3 across all systems?',
                   ['TLS 1.3 is a significant improvement over TLS 1.2 but still uses classical RSA and',
                    'ECC key exchange — both vulnerable to quantum attack. TLS 1.3 also does not address',
                    'TN3270e mainframe sessions (no TLS support), legacy ACORD XML over HTTP between older',
                    'trading partners, or SWIFT message-level security (which is separate from transport',
                    'layer). QBITEL Bridge wraps all protocol types — including those that cannot be',
                    'upgraded to TLS — with NIST-standardised PQC and provides fraud detection that',
                    'TLS upgrades cannot offer. TLS 1.3 is a floor requirement; PQC is the ceiling needed',
                    'for 30-60 year policy data protection.']):
        story.append(item)

    for item in qa('How does QBITEL Bridge compare to traditional data loss prevention (DLP) solutions?',
                   ['Traditional DLP operates on endpoint agents and email/web gateways — it does not',
                    'inspect ACORD XML, X12 EDI, SWIFT, or TN3270e protocol streams. DLP detects data',
                    'leaving the organisation but does not encrypt it, does not detect protocol-layer',
                    'fraud, and does not provide quantum-safe cryptography. QBITEL Bridge operates at',
                    'the network layer on industry-specific protocols that DLP tools have no awareness',
                    'of. Bridge also provides positive security (PQC encryption) rather than just',
                    'blocking — enabling secure data flows rather than just detecting insecure ones.']):
        story.append(item)

    for item in qa('How does QBITEL Bridge compare to FRISS or Shift Technology for fraud prevention?',
                   ['FRISS and Shift Technology are post-payment fraud analytics platforms — they identify',
                    'fraud patterns after claims are submitted and during adjudication. QBITEL Bridge',
                    'detects fraud at the wire level in real time, before adjudication. The platforms are',
                    'complementary: Bridge provides the pre-payment blocking layer, while FRISS/Shift',
                    'provide the deeper post-payment analytics and investigation workflow. Bridge also',
                    'enriches the data available to fraud analytics platforms by providing structured',
                    'protocol-layer indicators that improve their model accuracy. Many QBITEL clients',
                    'use both Bridge and a fraud analytics platform in combination.']):
        story.append(item)

    for item in qa('How does QBITEL Bridge compare to Palo Alto Networks or Fortinet for network security?',
                   ['Palo Alto and Fortinet are excellent perimeter security platforms operating at',
                    'network Layers 3-4 with application-layer signatures. They have no insurance-specific',
                    'protocol awareness — they cannot parse ACORD XML claim data, inspect X12 837 code',
                    'combinations for upcoding, or validate SWIFT reinsurance settlement amounts.',
                    'They also do not provide quantum-safe cryptography on session streams. QBITEL',
                    'Bridge complements perimeter security by operating at Layer 7 on insurance-specific',
                    'protocols and providing PQC encryption — filling the gap that even the best',
                    'perimeter security platforms cannot address.']):
        story.append(item)

    for item in qa('Why not build quantum-safe encryption in-house as part of our application modernisation?',
                   ['In-house PQC implementation risks: (1) Implementation errors in PQC are cryptographically',
                    'catastrophic and difficult to detect — side-channel attacks on ML-KEM implementations',
                    'have been demonstrated in academic literature; (2) Coverage — in-house efforts typically',
                    'cover new application code but miss legacy mainframe sessions, trading partner EDI,',
                    'and SWIFT channels; (3) Timeline — application modernisation programmes run 3-7 years;',
                    'HNDL attacks are active today; (4) Maintenance — NIST is expected to revise PQC',
                    'parameters; in-house code requires continuous expert maintenance. QBITEL Bridge',
                    'provides validated, HSM-backed PQC in 4-6 hours with ongoing maintenance included.']):
        story.append(item)

    for item in qa('How does QBITEL Bridge compare to IBM\'s quantum-safe offerings for System z?',
                   ['IBM offers quantum-safe features natively on System z (z16 and later) for z/OS',
                    'applications that have been updated to use IBM\'s PQC libraries. This is valuable',
                    'but has two constraints: (1) it requires COBOL/PL/I application code changes to',
                    'invoke the PQC libraries — most legacy policy admin applications cannot be modified;',
                    '(2) it only protects data within the z/OS environment, not the network sessions',
                    'leaving the mainframe to connected systems. QBITEL Bridge protects the network',
                    'layer without any mainframe code changes, and covers all connected systems',
                    '(claims, analytics, agents, reinsurance) not just the mainframe itself.']):
        story.append(item)

    for item in qa('What makes QBITEL Bridge specifically better for insurance than a generic PQC VPN?',
                   ['A generic PQC VPN encrypts the tunnel between network endpoints — it has no visibility',
                    'into what is inside the tunnel. It cannot detect a fraudulent X12 837 claim inside',
                    'an encrypted VPN tunnel, cannot classify ACORD XML by policy type for tiered PQC',
                    'protection, and cannot verify SWIFT message integrity. QBITEL Bridge is protocol-aware:',
                    'it understands ACORD XML, X12 EDI, HL7, SWIFT, and TN3270e at the payload level.',
                    'This protocol intelligence enables both the security outcomes (fraud detection,',
                    'integrity verification, data classification) and the compliance outputs (regulatory',
                    'evidence by protocol type and data category) that a generic VPN cannot provide.']):
        story.append(item)

    for item in qa('Are there other insurance companies already using QBITEL Bridge?',
                   ['Yes. QBITEL Bridge is deployed at regional P&C carriers, large life and annuity',
                    'insurers, health insurance organisations, and global reinsurers. Specific client',
                    'references are available under NDA to qualified prospects who have signed a',
                    'mutual non-disclosure agreement. We are also able to facilitate peer conversations',
                    'between prospective clients and existing clients at the CISO or CRO level upon',
                    'request. Client case studies (anonymised by default, named with client permission)',
                    'are available in the QBITEL Bridge Insurance Reference Pack.']):
        story.append(item)

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width=CONTENT_W, thickness=2, color=GOLD))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        'QBITEL Enterprise Insurance Practice  |  enterprise@qbitel.com  |  bridge.qbitel.com',
        ParagraphStyle('ctr', fontName='Helvetica-Bold', fontSize=10,
                       textColor=NAVY, alignment=TA_CENTER)))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'Confidential — For Authorized Recipients Only  |  Copyright 2026 QBITEL. All Rights Reserved.',
        ParagraphStyle('ctr2', fontName='Helvetica', fontSize=8,
                       textColor=MID_GREY, alignment=TA_CENTER)))

    return story


def build_doc(output_path):
    doc = BaseDocTemplate(
        output_path, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN)
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H,
                        leftPadding=0, rightPadding=0,
                        topPadding=0, bottomPadding=0, id='cover')
    inner_frame = Frame(MARGIN, 0.7 * inch, CONTENT_W,
                        PAGE_H - MARGIN - 0.7 * inch, id='inner')
    cover_template = PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover)
    inner_template = PageTemplate(id='Inner', frames=[inner_frame], onPage=draw_page)
    doc.addPageTemplates([cover_template, inner_template])

    styles = get_styles()
    story = []
    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())
    story.extend(build_story(styles))

    doc.build(story)
    print(f'PDF generated: {output_path}')


if __name__ == '__main__':
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'QBITEL_Insurance_Pitch_QA_Guide.pdf')
    build_doc(out)
