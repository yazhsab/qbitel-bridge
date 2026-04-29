"""Build QBITEL Bridge Automotive & Connected Vehicles Q&A Guide - PDF"""
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
DARK_RED   = HexColor('#8B1A1A')
GREEN      = HexColor('#1A6B3A')
LIGHT_RED  = HexColor('#FFF0F0')
LIGHT_GREEN = HexColor('#F0FFF4')
PAGE_W, PAGE_H = letter
MARGIN = 0.85 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

def sp(n): return Spacer(1, n)


class QABlock(Flowable):
    """Navy header (question), teal one-liner band, white body (full answer)."""
    def __init__(self, question, one_liner, full_answer, width=None):
        Flowable.__init__(self)
        self.question = question
        self.one_liner = one_liner
        self.full_answer = full_answer
        self.w = width or CONTENT_W
        self.header_h = 26
        self.one_liner_h = 20
        # Estimate body height from text length
        chars_per_line = int(self.w * 0.14)
        lines = max(2, len(full_answer) // chars_per_line + 1)
        self.body_h = lines * 13 + 12
        self.h = self.header_h + self.one_liner_h + self.body_h

    def wrap(self, availW, availH):
        return (self.w, self.h)

    def draw(self):
        c = self.canv
        w, h = self.w, self.h
        # Navy header
        c.setFillColor(NAVY)
        c.rect(0, h - self.header_h, w, self.header_h, fill=1, stroke=0)
        c.setFillColor(GOLD)
        c.rect(0, h - self.header_h, 4, self.header_h, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 8.5)
        # Truncate question if too long
        q = self.question
        max_w = w - 20
        while c.stringWidth(q, 'Helvetica-Bold', 8.5) > max_w and len(q) > 10:
            q = q[:-4] + '...'
        c.drawString(10, h - self.header_h + 9, q)
        # Teal one-liner band
        teal_y = h - self.header_h - self.one_liner_h
        c.setFillColor(TEAL)
        c.rect(0, teal_y, w, self.one_liner_h, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 8)
        ol = self.one_liner
        while c.stringWidth(ol, 'Helvetica-Bold', 8) > max_w and len(ol) > 10:
            ol = ol[:-4] + '...'
        c.drawString(10, teal_y + 6, ol)
        # White body
        body_y = 0
        c.setFillColor(WHITE_C)
        c.rect(0, body_y, w, self.body_h, fill=1, stroke=0)
        c.setStrokeColor(MID_GREY)
        c.setLineWidth(0.5)
        c.rect(0, 0, w, h, fill=0, stroke=1)
        # Wrap and render full answer text
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 8)
        # Simple word wrap
        words = self.full_answer.split()
        lines = []
        line = ''
        for word in words:
            test = (line + ' ' + word).strip()
            if c.stringWidth(test, 'Helvetica', 8) < w - 20:
                line = test
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        y_start = self.body_h - 12
        for i, ln in enumerate(lines):
            y = y_start - i * 13
            if y < 4:
                break
            c.drawString(10, y, ln)


class ObjectionBlock(Flowable):
    """Dark red header (objection) + green response area."""
    def __init__(self, objection, response, width=None):
        Flowable.__init__(self)
        self.objection = objection
        self.response = response
        self.w = width or CONTENT_W
        self.header_h = 28
        chars_per_line = int(self.w * 0.14)
        lines = max(2, len(response) // chars_per_line + 1)
        self.body_h = lines * 13 + 14
        self.h = self.header_h + self.body_h

    def wrap(self, availW, availH):
        return (self.w, self.h)

    def draw(self):
        c = self.canv
        w, h = self.w, self.h
        # Dark red header
        c.setFillColor(DARK_RED)
        c.rect(0, h - self.header_h, w, self.header_h, fill=1, stroke=0)
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 8)
        c.drawString(10, h - self.header_h + 10, 'OBJECTION:')
        c.setFont('Helvetica', 8)
        obj = self.objection
        max_w = w - 100
        while c.stringWidth(obj, 'Helvetica', 8) > max_w and len(obj) > 10:
            obj = obj[:-4] + '...'
        c.drawString(82, h - self.header_h + 10, obj)
        # Green response area
        c.setFillColor(LIGHT_GREEN)
        c.rect(0, 0, w, self.body_h, fill=1, stroke=0)
        c.setFillColor(GREEN)
        c.rect(0, 0, 5, self.body_h, fill=1, stroke=0)
        c.setStrokeColor(GREEN)
        c.setLineWidth(0.5)
        c.rect(0, 0, w, h, fill=0, stroke=1)
        # Response label
        c.setFillColor(GREEN)
        c.setFont('Helvetica-Bold', 7.5)
        c.drawString(12, self.body_h - 12, 'RESPONSE:')
        # Response text
        c.setFillColor(DARK_TEXT)
        c.setFont('Helvetica', 8)
        words = self.response.split()
        lines = []
        line = ''
        for word in words:
            test = (line + ' ' + word).strip()
            if c.stringWidth(test, 'Helvetica', 8) < w - 22:
                line = test
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        y_start = self.body_h - 25
        for i, ln in enumerate(lines):
            y = y_start - i * 13
            if y < 4:
                break
            c.drawString(12, y, ln)


class TOCEntry(Flowable):
    """Section divider with number and title."""
    def __init__(self, number, title, width=None):
        Flowable.__init__(self)
        self.number = number
        self.title = title
        self.w = width or CONTENT_W
        self.h = 40

    def wrap(self, availW, availH):
        return (self.w, self.h)

    def draw(self):
        c = self.canv
        w, h = self.w, self.h
        # Full navy background
        c.setFillColor(NAVY)
        c.rect(0, 0, w, h, fill=1, stroke=0)
        # Gold number badge
        c.setFillColor(GOLD)
        c.rect(0, 0, 50, h, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 16)
        nw = c.stringWidth(self.number, 'Helvetica-Bold', 16)
        c.drawString(25 - nw / 2, h / 2 - 7, self.number)
        # Teal right accent
        c.setFillColor(TEAL)
        c.rect(w - 5, 0, 5, h, fill=1, stroke=0)
        # Title
        c.setFillColor(WHITE_C)
        c.setFont('Helvetica-Bold', 12)
        c.drawString(62, h / 2 - 5, self.title)


def get_styles():
    return {
        'body': ParagraphStyle('body', fontName='Helvetica', fontSize=8.5,
                               leading=12, textColor=DARK_TEXT, spaceAfter=4),
        'h1': ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=16,
                             leading=20, textColor=NAVY, spaceAfter=8),
        'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=12,
                             leading=16, textColor=TEAL, spaceAfter=6),
        'h3': ParagraphStyle('h3', fontName='Helvetica-Bold', fontSize=10,
                             leading=14, textColor=NAVY, spaceAfter=4),
        'table_cell': ParagraphStyle('table_cell', fontName='Helvetica', fontSize=8,
                                     leading=11, textColor=DARK_TEXT),
        'table_header': ParagraphStyle('table_header', fontName='Helvetica-Bold', fontSize=8,
                                       leading=11, textColor=WHITE_C),
        'bullet': ParagraphStyle('bullet', fontName='Helvetica', fontSize=8.5,
                                 leading=12, textColor=DARK_TEXT, leftIndent=12,
                                 firstLineIndent=-10, spaceAfter=2),
    }


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 30, PAGE_W, 30, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(MARGIN, PAGE_H - 20, 'QBITEL BRIDGE \u2014 AUTOMOTIVE Q&A GUIDE')
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 20, f'Page {doc.page}')
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, PAGE_W, 22, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 7.5)
    canvas.drawString(MARGIN, 7, 'Confidential \u2014 For Authorized Recipients Only  |  \u00a9 2026 QBITEL. All Rights Reserved.')
    contact_str = 'enterprise@qbitel.com  |  bridge.qbitel.com'
    cw = canvas.stringWidth(contact_str, 'Helvetica', 7.5)
    canvas.drawString(PAGE_W - MARGIN - cw, 7, contact_str)
    canvas.restoreState()


def draw_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, PAGE_H * 0.55, PAGE_W, PAGE_H * 0.45, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, PAGE_H * 0.55, PAGE_W, 4, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 28)
    canvas.drawString(MARGIN, PAGE_H * 0.55 + PAGE_H * 0.45 * 0.75, 'QBITEL BRIDGE')
    canvas.setFillColor(GOLD)
    canvas.setFont('Helvetica-Bold', 18)
    canvas.drawString(MARGIN, PAGE_H * 0.55 + PAGE_H * 0.45 * 0.55, 'Automotive & Connected Vehicles')
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica-Bold', 20)
    canvas.drawString(MARGIN, PAGE_H * 0.55 + PAGE_H * 0.45 * 0.38, 'Sales Q&A Guide')
    canvas.setFont('Helvetica', 12)
    canvas.drawString(MARGIN, PAGE_H * 0.55 + PAGE_H * 0.45 * 0.24, '65 Questions Across 10 Sections')
    # Stats
    stats = [('65', 'Q&As'), ('10', 'Sections'), ('7', 'Objections'), ('PQC', 'Ready')]
    box_w = (PAGE_W - 2 * MARGIN - 30) / 4
    for i, (val, lbl) in enumerate(stats):
        x = MARGIN + i * (box_w + 10)
        y = 0.12 * PAGE_H
        canvas.setFillColor(LIGHT_NAVY)
        canvas.roundRect(x, y, box_w, 0.1 * PAGE_H, 6, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.setFont('Helvetica-Bold', 16)
        vw = canvas.stringWidth(val, 'Helvetica-Bold', 16)
        canvas.drawString(x + box_w / 2 - vw / 2, y + 0.1 * PAGE_H * 0.55, val)
        canvas.setFillColor(WHITE_C)
        canvas.setFont('Helvetica', 8)
        lw = canvas.stringWidth(lbl, 'Helvetica', 8)
        canvas.drawString(x + box_w / 2 - lw / 2, y + 0.1 * PAGE_H * 0.25, lbl)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, PAGE_W, 0.08 * PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(WHITE_C)
    canvas.setFont('Helvetica', 9)
    canvas.drawString(MARGIN, 0.04 * PAGE_H, 'enterprise@qbitel.com  |  bridge.qbitel.com')
    canvas.restoreState()


def build_doc(output_path):
    doc = BaseDocTemplate(output_path, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN)
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id='cover')
    inner_frame = Frame(MARGIN, 0.7 * inch, CONTENT_W,
        PAGE_H - MARGIN - 0.7 * inch, id='inner')
    cover_template = PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover)
    inner_template = PageTemplate(id='Inner', frames=[inner_frame], onPage=draw_page)
    doc.addPageTemplates([cover_template, inner_template])
    story = []
    story.append(NextPageTemplate('Inner'))
    story.append(PageBreak())

    # =========================================================================
    # SECTION 1: V2X Architecture & Safety
    # =========================================================================
    story.append(TOCEntry('1', 'V2X Architecture & Safety'))
    story.append(sp(8))

    s1_qas = [
        ('What V2X standards does QBITEL support?',
         'Supports IEEE 1609.2 with PQC extension',
         'Full support for DSRC and C-V2X (PC5/Uu), all SAE J2735 message types, IEEE 1609.2 security with Falcon-512/ML-DSA-65 extension. Backward compatible with existing RSU deployments.'),
        ('How does QBITEL protect V2X message integrity?',
         'Cryptographic MAC on every V2X message',
         'Each BSM, SPaT, MAP, and TIM message is signed with a Falcon-512 short signature. Receivers verify authenticity before acting on the message — a spoofed message is cryptographically rejected.'),
        ('Can QBITEL prevent false collision warnings?',
         'Yes — spoofed BSMs are rejected by signature verification',
         'Without authentication, any $20 SDR can broadcast fake Basic Safety Messages triggering emergency braking. QBITEL\'s per-message signature makes spoofed messages detectable within <5ms.'),
        ('What happens if a vehicle\'s private key is compromised?',
         'Pseudonym revocation within 100ms fleet-wide',
         'SCMS pseudonym architecture means each vehicle uses short-lived certificates (1 week). Compromise triggers automated misbehavior reporting and CRL distribution within 100ms.'),
        ('How does platooning security work?',
         'Every convoy command is authenticated — rogue vehicles cannot join',
         'Platoon leader identity is bound to a long-term PQC certificate. Every following-distance and brake command is signed. Unauthorized vehicles attempting to join are detected in <100ms.'),
        ('Does QBITEL protect C-V2X as well as DSRC?',
         'Yes — protocol-agnostic V2X security layer',
         'QBITEL operates at the security service layer above the physical layer, supporting both DSRC (802.11p) and C-V2X (LTE-V2X, 5G-NR-V2X) deployments equally.'),
        ('What is the impact on V2X latency?',
         '<5ms overhead — well within the 10ms safety requirement',
         'The Falcon-512 implicit certificate scheme adds <5ms to message processing. IEEE 1609.2 requires <10ms for safety-critical messages. QBITEL meets this with 2x margin.'),
        ('How does QBITEL handle high-density intersections?',
         'Batch verification at 1,500+ msg/sec handles peak traffic',
         'Dense urban intersections can generate 1,000+ simultaneous V2X messages per second. QBITEL\'s SIMD-accelerated batch verification handles 1,500+ msg/sec with <5ms per-message latency.'),
    ]
    for q, ol, fa in s1_qas:
        story.append(QABlock(q, ol, fa))
        story.append(sp(6))

    # =========================================================================
    # SECTION 2: Cryptographic Performance
    # =========================================================================
    story.append(sp(8))
    story.append(TOCEntry('2', 'Cryptographic Performance'))
    story.append(sp(8))

    s2_qas = [
        ('Which PQC algorithms are used for V2X?',
         'Falcon-512 for signatures (small), ML-KEM-768 for key encap',
         'Falcon-512 produces 666-byte signatures — the most compact NIST-standardized post-quantum signature. ML-KEM-768 handles key encapsulation for secure channel establishment.'),
        ('Why Falcon-512 instead of ML-DSA (Dilithium)?',
         '72% smaller signatures — critical for bandwidth-constrained V2X',
         'ML-DSA signatures are 2,420 bytes, which exceeds V2X DSRC bandwidth. Falcon-512 at 666 bytes fits comfortably while providing equivalent security.'),
        ('What is the batch verification architecture?',
         'SIMD-parallelized multi-signature verification',
         'QBITEL uses AVX2/NEON SIMD instructions to batch-verify groups of Falcon-512 signatures simultaneously. This achieves 1,500+ verifications/second vs ~200/sec sequential.'),
        ('How large are the certificates?',
         '666 bytes total (Falcon-512 implicit)',
         'Standard IEEE 1609.2 certificates with Dilithium would be ~3KB. QBITEL\'s implicit certificate scheme with Falcon-512 reduces this to 666 bytes — fitting easily in a single V2X frame.'),
        ('What is the hybrid transition approach?',
         'Dual-mode: classical ECDSA + Falcon-512 hybrid during migration',
         'During the transition period, QBITEL vehicles sign messages with both ECDSA (for legacy RSUs) and Falcon-512 (for quantum-safe RSUs). Legacy vehicles see a valid classical signature; upgraded infrastructure verifies both.'),
        ('What NIST security level does this achieve?',
         'NIST Level 1 (Falcon-512) — equivalent to AES-128 quantum security',
         'Falcon-512 achieves NIST PQC Level 1, providing 128-bit post-quantum security. For high-security V2I infrastructure, Falcon-1024 (Level 5) is also supported.'),
        ('How does key generation work for a large fleet?',
         'Bulk provisioning 10,000 vehicles/hour via SCMS API',
         'QBITEL integrates with SCMS providers (CAMP, OBS) for bulk pseudonym certificate generation. Falcon-512 key generation takes <1ms per vehicle on standard hardware.'),
        ('What is the OTA update signing architecture?',
         'ML-DSA-65 signed firmware packages with TPM verification',
         'All OTA firmware updates are signed with ML-DSA-65 (NIST Level 3). Post-installation, TPM 2.0 attestation verifies the update hash before activation.'),
    ]
    for q, ol, fa in s2_qas:
        story.append(QABlock(q, ol, fa))
        story.append(sp(6))

    # =========================================================================
    # SECTION 3: Fleet OTA & Migration
    # =========================================================================
    story.append(sp(8))
    story.append(TOCEntry('3', 'Fleet OTA & Migration'))
    story.append(sp(8))

    s3_qas = [
        ('How does staged rollout work?',
         'Canary \u2192 1% \u2192 10% \u2192 100% with automatic rollback',
         'Each stage monitors error rates for 24 hours. If >0.1% of vehicles report issues, rollback is automatically triggered. Full fleet migration completes in 12 months with zero forced downtime.'),
        ('What if an OTA update fails mid-delivery?',
         'Automatic rollback to last known good state',
         'Failed updates trigger immediate rollback to the previous firmware version, with TPM attestation verifying the rollback completed. The affected vehicle is flagged for investigation.'),
        ('How are delta updates generated?',
         'Binary diff algorithm reduces payload 80%',
         'QBITEL\'s OTA platform generates binary diffs between firmware versions, reducing the update payload by ~80%. A 500MB full update becomes a ~100MB delta.'),
        ('Can QBITEL update legacy vehicles without V2X capability?',
         'Yes — supports cellular and WiFi OTA delivery',
         'For vehicles without V2X (pre-2020 fleet), QBITEL delivers PQC updates via cellular (4G/5G) or dealership WiFi hotspots, using the same staged rollout and TPM verification.'),
        ('How long does full fleet migration take?',
         '12 months for 10M vehicles with staged rollout',
         'Week 1-2: 10-vehicle canary. Month 1: 1% (100K vehicles). Month 3: 10% (1M). Month 12: 100% complete. Timeline scales linearly with fleet size.'),
        ('What is the rollback procedure?',
         'Automated: error rate trigger. Manual: fleet management console',
         'Automatic rollback triggers on error rate threshold. Manual rollback available via fleet management console for any subset of vehicles within 5 minutes.'),
    ]
    for q, ol, fa in s3_qas:
        story.append(QABlock(q, ol, fa))
        story.append(sp(6))

    # =========================================================================
    # SECTION 4: UNECE WP.29 & ISO 21434
    # =========================================================================
    story.append(sp(8))
    story.append(TOCEntry('4', 'UNECE WP.29 & ISO/SAE 21434'))
    story.append(sp(8))

    s4_qas = [
        ('What does UNECE WP.29 R155 require?',
         'Cybersecurity Management System (CSMS) with ongoing monitoring',
         'R155 requires OEMs to have a certified CSMS covering the full vehicle lifecycle. QBITEL provides the technical security controls and documentation evidence required for type approval.'),
        ('How does QBITEL support the TARA process?',
         'Automated threat analysis covering V2X, OTA, telematics attack vectors',
         'QBITEL\'s AI engine generates TARA documentation covering all V2X attack surfaces: spoofing, replay, relay, DoS, and supply chain attacks. Evidence is formatted for type approval submission.'),
        ('What evidence does QBITEL produce for type approval?',
         'Security validation report, TARA, CSMS documentation',
         'QBITEL automatically generates all required ISO 21434 artifacts: threat catalog, risk assessment, security goals, cybersecurity concept, and validation test results.'),
        ('Does QBITEL support UN R156 software update requirements?',
         'Yes — OTA management system with full audit trail',
         'R156 requires a Software Update Management System (SUMS). QBITEL\'s OTA platform provides cryptographically signed update records, rollback logs, and fleet coverage reports.'),
        ('Which countries mandate UNECE WP.29?',
         '54 countries including EU, Japan, South Korea, UK',
         'WP.29 is mandatory for new type approvals in the EU (July 2022), Japan, South Korea, and 51 other UN members. China and US have equivalent programs under development.'),
        ('What is the type approval timeline impact?',
         '2-3 months vs 6-12 months without QBITEL',
         'Manual TARA and security validation typically takes 6-12 months. QBITEL\'s automated evidence generation reduces this to 2-3 months.'),
        ('Does QBITEL work with existing CSMS systems?',
         'Yes — integrates as a technical control provider',
         'QBITEL integrates with established CSMS frameworks (Upstream Security, Argus, Cymotive) as a technical control layer, supplementing their monitoring with V2X-specific PQC authentication.'),
        ('How does QBITEL handle ongoing WP.29 monitoring requirements?',
         'Continuous monitoring dashboard with automated incident reporting',
         'R155 requires ongoing monitoring throughout vehicle lifetime. QBITEL\'s fleet health dashboard tracks V2X authentication failures, misbehavior reports, and OTA update status continuously.'),
    ]
    for q, ol, fa in s4_qas:
        story.append(QABlock(q, ol, fa))
        story.append(sp(6))

    # =========================================================================
    # SECTION 5: SCMS & Certificate Management
    # =========================================================================
    story.append(sp(8))
    story.append(TOCEntry('5', 'SCMS & Certificate Management'))
    story.append(sp(8))

    s5_qas = [
        ('What SCMS providers does QBITEL integrate with?',
         'CAMP, OnBoard Security, and custom SCMS via API',
         'QBITEL provides pre-built connectors for CAMP (North America) and OBS (international), plus a RESTful API for custom SCMS integrations.'),
        ('How does pseudonym certificate rotation work?',
         'Weekly rotation with overlap period for seamless transition',
         'Pseudonym certificates expire weekly. QBITEL pre-provisions the next week\'s certificates 48 hours in advance, ensuring seamless rotation with no connectivity gap.'),
        ('How does misbehavior detection work?',
         'Anomaly detection flags vehicles broadcasting invalid messages',
         'QBITEL cross-validates V2X messages against GPS, radar, and LIDAR data. Messages inconsistent with physics (impossible speed, wrong location) trigger misbehavior reports to the SCMS.'),
        ('How fast is certificate revocation?',
         'CRL distributed to all RSUs within 100ms',
         'When SCMS revokes a pseudonym, QBITEL distributes the updated Certificate Revocation List (CRL) to all connected RSUs within 100ms via the V2X management channel.'),
        ('What privacy protections are in place for pseudonyms?',
         'Pseudonym unlinkability — observer cannot correlate vehicle trips',
         'Each vehicle uses a pool of 20+ pseudonyms per week, randomly rotated. No observer can link pseudonym changes to a specific vehicle, protecting location privacy.'),
    ]
    for q, ol, fa in s5_qas:
        story.append(QABlock(q, ol, fa))
        story.append(sp(6))

    # =========================================================================
    # SECTION 6: Autonomous & Connected Vehicles
    # =========================================================================
    story.append(sp(8))
    story.append(TOCEntry('6', 'Autonomous & Connected Vehicles'))
    story.append(sp(8))

    s6_qas = [
        ('How does QBITEL secure AV sensor fusion?',
         'Authenticates all V2X inputs to the sensor fusion stack',
         'Autonomous vehicles fuse V2X data with LIDAR/radar/camera. QBITEL ensures all V2X inputs are cryptographically authenticated before entering the fusion stack, preventing spoofed sensor data injection.'),
        ('Does QBITEL support geofencing security?',
         'Yes — geofence boundary messages are authenticated',
         'Smart infrastructure uses V2X to define geofenced zones (low-speed areas, pedestrian zones). QBITEL authenticates all geofence activation messages to prevent malicious zone creation.'),
        ('How is remote diagnostic security handled?',
         'UDS sessions over authenticated, PQC-encrypted channels',
         'Unified Diagnostic Services (UDS) for remote diagnostics are tunneled through QBITEL\'s ML-KEM-768 encrypted channel, preventing unauthorized diagnostic access or ECU reprogramming.'),
        ('What about vehicle-to-cloud (V2C) security?',
         'All telematics and OTA channels use ML-KEM-768',
         'Vehicle-to-cloud communication (telemetry, navigation, OTA) is protected with ML-KEM-768 key encapsulation, ensuring collected data cannot be decrypted by future quantum computers.'),
        ('How does QBITEL protect electric vehicle charging?',
         'ISO 15118 PLC communication PQC-wrapped',
         'EV charging uses ISO 15118 Power Line Communication for smart charging negotiation. QBITEL wraps these sessions with PQC encryption to prevent charging spoofing and payment fraud.'),
    ]
    for q, ol, fa in s6_qas:
        story.append(QABlock(q, ol, fa))
        story.append(sp(6))

    # =========================================================================
    # SECTION 7: Supply Chain Security
    # =========================================================================
    story.append(sp(8))
    story.append(TOCEntry('7', 'Supply Chain Security'))
    story.append(sp(8))

    s7_qas = [
        ('How does QBITEL protect Tier 1/2 supplier ECUs?',
         'Hardware-bound PQC identity for every ECU',
         'Each ECU is provisioned with a hardware-bound Falcon-512 identity certificate during manufacturing. ECU communications are authenticated against this identity, preventing counterfeit ECU injection.'),
        ('How are ECU firmware supply chain attacks detected?',
         'Code signing with ML-DSA-65 + TPM attestation',
         'All ECU firmware is signed by the manufacturer with ML-DSA-65. TPM 2.0 attestation verifies the signature chain from silicon to software before any ECU boots.'),
        ('Can QBITEL detect counterfeit parts?',
         'Cryptographic part authentication via ECU identity',
         'Counterfeit ECUs lack valid QBITEL identity certificates. When a counterfeit ECU attempts to communicate on the CAN bus, it fails authentication and triggers an alert.'),
        ('How does QBITEL secure the OEM-supplier data exchange?',
         'PQC-encrypted data rooms for design files and firmware',
         'Sensitive design files, calibration data, and firmware shared between OEMs and suppliers are protected with ML-KEM-1024 encryption, resistant to industrial espionage via quantum decryption.'),
        ('What is the software supply chain bill of materials (SBOM) capability?',
         'Automated SBOM generation with cryptographic integrity',
         'QBITEL generates cryptographically signed SBOMs for all vehicle software components, enabling rapid vulnerability assessment when new CVEs are published.'),
    ]
    for q, ol, fa in s7_qas:
        story.append(QABlock(q, ol, fa))
        story.append(sp(6))

    # =========================================================================
    # SECTION 8: OEM & Tier 1 Integration
    # =========================================================================
    story.append(sp(8))
    story.append(TOCEntry('8', 'OEM & Tier 1 Integration'))
    story.append(sp(8))

    s8_qas = [
        ('How long does OEM integration take?',
         '3-6 months for full V2X stack integration',
         'Phase 1 (Month 1): SCMS API integration and certificate provisioning. Phase 2 (Month 2-3): V2X authentication library integration. Phase 3 (Month 4-6): OTA platform and fleet management.'),
        ('Does QBITEL require hardware changes to existing vehicles?',
         'No — software-only for existing fleet; hardware-optimized for new platforms',
         'Existing vehicles receive QBITEL via OTA software update. New vehicle platforms can integrate Falcon-512 hardware acceleration (NXP S32G, Qualcomm SA8195P) for optimal performance.'),
        ('How does QBITEL integrate with existing V2X stacks?',
         'SDK with API wrapper for all major V2X stacks',
         'QBITEL provides a C/C++ SDK with wrappers for Qualcomm V2X, NXP RoadLINK, Autotalks CRATON2, and Continental MK5/MK6 platforms.'),
        ('What is the licensing model?',
         'Per-vehicle annual subscription + SCMS integration fee',
         'QBITEL is priced at $10-30/vehicle/year depending on fleet size and feature set. Volume discounts apply for fleets >1M vehicles. SCMS integration is a one-time setup fee.'),
        ('What support does QBITEL provide during integration?',
         'Dedicated automotive engineering team + 24/7 NOC',
         'QBITEL provides a dedicated automotive security engineering team during integration, plus 24/7 NOC support for production deployments.'),
    ]
    for q, ol, fa in s8_qas:
        story.append(QABlock(q, ol, fa))
        story.append(sp(6))

    # =========================================================================
    # SECTION 9: Hard Objections
    # =========================================================================
    story.append(sp(8))
    story.append(TOCEntry('9', 'Hard Objections & Responses'))
    story.append(sp(8))

    objections = [
        ('"V2X adoption is too early for quantum concerns"',
         'V2X deployment is accelerating (DSRC mandates in EU infrastructure, C-V2X in new vehicles). Vehicles built today have 15-20 year lifecycles. The harvest-now-decrypt-later threat means vehicle location and telematics data collected today can be decrypted by 2040 quantum computers — within the vehicle\'s lifetime.'),
        ('"Our HSM vendor handles this"',
         'HSM vendors manage keys but don\'t provide V2X protocol integration, implicit certificate compression, or SCMS connectivity. QBITEL provides the full stack: V2X authentication library, SCMS integration, OTA pipeline, and compliance evidence — not just key storage.'),
        ('"UNECE WP.29 doesn\'t mandate PQC yet"',
         'WP.29 R155 requires cybersecurity management throughout the vehicle\'s life. Vehicles approved today will still be sold and driven in 2040-2045. Using cryptography that will be broken within the vehicle\'s compliance window is a systematic risk that type approval authorities are beginning to scrutinize.'),
        ('"OTA updates risk bricking vehicles"',
         'QBITEL\'s staged rollout (canary \u2192 1% \u2192 10% \u2192 100%) with automatic rollback eliminates this risk. Each stage is monitored for 24 hours. If any issues arise, rollback is automated. Our OTA platform has a 99.999% success rate across >5M vehicle updates.'),
        ('"Classical ECDSA meets today\'s requirements"',
         'It meets today\'s requirements but not tomorrow\'s. A vehicle sold in 2026 with ECDSA will be on the road in 2045, when quantum computers are projected to break ECDSA in hours. The cost to migrate the fleet mid-lifecycle is 10-50x higher than building in PQC from the start.'),
        ('"Our SCMS provider will add PQC"',
         'SCMS providers manage certificate issuance, but the V2X authentication library running in the vehicle is the OEM\'s responsibility. QBITEL integrates with your existing SCMS (CAMP, OBS) while providing the vehicle-side PQC authentication stack.'),
        ('"Too expensive to upgrade a 10M vehicle fleet"',
         'At $10-30/vehicle/year, a 10M vehicle fleet costs $100-300M/year — compared to a single major V2X spoofing incident causing recalls, liability, and brand damage estimated at $500M+. The cost of NOT protecting the fleet is orders of magnitude higher.'),
    ]
    for obj, resp in objections:
        story.append(ObjectionBlock(obj, resp))
        story.append(sp(8))

    # =========================================================================
    # SECTION 10: Competitive Positioning
    # =========================================================================
    story.append(sp(8))
    story.append(TOCEntry('10', 'Competitive Positioning'))
    story.append(sp(8))

    s10_qas = [
        ('How does QBITEL compare to classical PKI (current V2X)?',
         'QBITEL adds PQC to existing PKI; classical PKI vulnerable to HNDL attacks',
         'QBITEL adds post-quantum protection to existing PKI infrastructure. Classical ECDSA-based PKI is vulnerable to harvest-now-decrypt-later attacks given 15-20yr vehicle lifetimes extending past the quantum threat horizon.'),
        ('How does QBITEL compare to Argus/Karamba (in-vehicle security)?',
         'Argus/Karamba protect in-vehicle networks; QBITEL protects V2X — complementary',
         'Argus Cyber Security and Karamba Security protect in-vehicle networks (CAN bus, ECU intrusion). QBITEL protects V2X communications (V2V, V2I). These are non-overlapping, complementary security layers.'),
        ('How does QBITEL compare to cloud-native encryption (AWS IoT, Azure)?',
         'Cloud covers V2C; QBITEL covers V2X — architecturally different problem',
         'Cloud providers encrypt vehicle-to-cloud (V2C) channels. They cannot protect V2X (peer-to-peer, <5ms, broadcast). QBITEL covers the V2X security layer that cloud providers architecturally cannot reach.'),
        ('What about waiting for the IEEE 1609.2 PQC standard?',
         'IEEE working group is years away; QBITEL implements PQC today with migration path',
         'The IEEE 1609.2 PQC working group is years from publication. Vehicles being designed now have 2025-2028 production starts and 15-20yr lifecycle commitments. QBITEL\'s NIST FIPS 203/204-aligned implementation provides migration path to final standard.'),
        ('How does QBITEL compare to HSM vendors (Thales, Infineon)?',
         'HSMs manage keys; QBITEL provides V2X integration + SCMS + OTA + compliance',
         'HSMs manage cryptographic keys and hardware acceleration. QBITEL provides V2X protocol integration, SCMS connectivity, OTA pipeline, and compliance evidence generation. These are complementary, not competitive.'),
        ('How does QBITEL compare to SCMS providers?',
         'SCMS manages cert issuance; QBITEL provides vehicle-side auth + SCMS integration',
         'SCMS providers (CAMP, OBS) manage certificate issuance and revocation. QBITEL provides the vehicle-side authentication library and integrates with existing SCMS investments. Works with, not against, your SCMS.'),
        ('Does QBITEL have automotive-specific certifications?',
         'ISO/SAE 21434 validated, NIST FIPS 203/204 compliant, WP.29 evidence generation',
         'QBITEL implements ISO/SAE 21434-validated security controls, NIST FIPS 203/204 standardized algorithms (ML-KEM, ML-DSA), and generates UNECE WP.29 R155 type approval evidence packages.'),
        ('What is QBITEL\'s automotive roadmap?',
         'IEEE 1609.2 PQC standard adoption, 5G-NR-V2X, AV sensor fusion auth in 2026',
         'QBITEL\'s 2026 roadmap includes: IEEE 1609.2 PQC standard adoption when published, 5G-NR-V2X enhanced security, autonomous vehicle sensor fusion authentication, and ISO 15118-3 EV charging PQC integration.'),
    ]
    for q, ol, fa in s10_qas:
        story.append(QABlock(q, ol, fa))
        story.append(sp(6))

    # Contact footer
    story.append(sp(12))
    contact_data = [
        [Paragraph('<b>enterprise@qbitel.com</b>',
                   ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=10, textColor=GOLD)),
         Paragraph('<b>bridge.qbitel.com</b>',
                   ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=10, textColor=GOLD)),
         Paragraph('<b>Contact your account team</b>',
                   ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=10, textColor=GOLD))],
        [Paragraph('Email', ParagraphStyle('cl', fontName='Helvetica', fontSize=8, textColor=WHITE_C)),
         Paragraph('Website', ParagraphStyle('cl', fontName='Helvetica', fontSize=8, textColor=WHITE_C)),
         Paragraph('Schedule a call', ParagraphStyle('cl', fontName='Helvetica', fontSize=8, textColor=WHITE_C))],
    ]
    contact_tbl = Table(contact_data, colWidths=[CONTENT_W / 3] * 3)
    contact_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), NAVY),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, TEAL),
    ]))
    story.append(contact_tbl)

    doc.build(story)


if __name__ == '__main__':
    import os
    os.chdir('/Users/prabakarankannan/qbitel')
    build_doc('docs/brochures/QBITEL_Automotive_Pitch_QA_Guide.pdf')
    print('PDF saved: docs/brochures/QBITEL_Automotive_Pitch_QA_Guide.pdf')
