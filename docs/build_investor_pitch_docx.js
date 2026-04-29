const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, TabStopType, TabStopPosition,
} = require("docx");

// ── Color Palette (matching the PDF) ──
const BRAND_DARK = "1B2A4A";
const BRAND_BLUE = "2E75B6";
const BRAND_LIGHT = "D5E8F0";
const BRAND_ACCENT = "E8792F";
const BRAND_GREEN = "2E8B57";
const BRAND_TEAL = "1A8A7D";
const GRAY_LIGHT = "F2F2F2";
const WHITE = "FFFFFF";
const RED = "CC0000";

// ── Helpers ──
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

function heading(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({ heading: level, children: [new TextRun(text)] });
}
function para(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    alignment: opts.align || AlignmentType.LEFT,
    children: [new TextRun({ text, font: "Arial", size: 22, ...opts })],
  });
}
function boldPara(text, opts = {}) { return para(text, { bold: true, ...opts }); }
function spacer() { return new Paragraph({ spacing: { after: 60 }, children: [] }); }

function headerCell(text, width, opts = {}) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    shading: { fill: BRAND_DARK, type: ShadingType.CLEAR },
    margins: cellMargins,
    children: [new Paragraph({ alignment: AlignmentType.LEFT, children: [new TextRun({ text, bold: true, color: WHITE, font: "Arial", size: 20 })] })],
  });
}
function dataCell(text, width, opts = {}) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    shading: opts.shading ? { fill: opts.shading, type: ShadingType.CLEAR } : undefined,
    margins: cellMargins,
    children: [new Paragraph({ alignment: opts.align || AlignmentType.LEFT, children: [new TextRun({ text: String(text), font: "Arial", size: 20, bold: opts.bold || false, color: opts.color || "000000" })] })],
  });
}

function makeTable(headers, rows, colWidths) {
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({ children: headers.map((h, i) => headerCell(h, colWidths[i])) }),
      ...rows.map((row, ri) =>
        new TableRow({
          children: row.map((cell, ci) => {
            const isTotal = row[0]?.toString().includes("TOTAL") || row[0]?.toString().includes("Total");
            return dataCell(cell, colWidths[ci], {
              shading: isTotal ? "E8E8E8" : (ri % 2 === 1 ? GRAY_LIGHT : undefined),
              bold: isTotal,
            });
          }),
        })
      ),
    ],
  });
}

// Callout box (like the blue boxes in the PDF)
function calloutBox(title, body) {
  const noBorder = { style: BorderStyle.NONE, size: 0 };
  const leftBorder = { style: BorderStyle.SINGLE, size: 12, color: BRAND_BLUE };
  const boxBorders = { top: noBorder, bottom: noBorder, right: noBorder, left: leftBorder };
  const boxMargins = { top: 120, bottom: 120, left: 200, right: 200 };
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [new TableRow({
      children: [new TableCell({
        borders: boxBorders,
        width: { size: 9360, type: WidthType.DXA },
        shading: { fill: "EDF4F9", type: ShadingType.CLEAR },
        margins: boxMargins,
        children: [
          new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: title, bold: true, color: BRAND_BLUE, font: "Arial", size: 22 })] }),
          new Paragraph({ children: [new TextRun({ text: body, font: "Arial", size: 20, color: "333333" })] }),
        ],
      })],
    })],
  });
}

// Metric highlight row (like the PDF hero metrics)
function metricRow(metrics) {
  const colW = Math.floor(9360 / metrics.length);
  const noBorder = { style: BorderStyle.NONE, size: 0 };
  const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: "DDDDDD" };
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: metrics.map(() => colW),
    rows: [new TableRow({
      children: metrics.map((m, i) => new TableCell({
        borders: { top: thinBorder, bottom: thinBorder, left: i === 0 ? thinBorder : noBorder, right: i === metrics.length - 1 ? thinBorder : noBorder },
        shading: { fill: m.bg || GRAY_LIGHT, type: ShadingType.CLEAR },
        margins: { top: 120, bottom: 120, left: 80, right: 80 },
        width: { size: colW, type: WidthType.DXA },
        children: [
          new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: m.value, bold: true, color: m.color || BRAND_BLUE, font: "Arial", size: 36 })] }),
          new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 }, children: [new TextRun({ text: m.label, bold: true, color: BRAND_DARK, font: "Arial", size: 18 })] }),
          new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: m.sub || "", color: "888888", font: "Arial", size: 16 })] }),
        ],
      })),
    })],
  });
}

const numbering = {
  config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
    ]},
    { reference: "numbers", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
    ]},
  ],
};
function bullet(text, level = 0) {
  return new Paragraph({ numbering: { reference: "bullets", level }, spacing: { after: 80 }, children: [new TextRun({ text, font: "Arial", size: 22 })] });
}
function numberedItem(text) {
  return new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text, font: "Arial", size: 22 })] });
}

// ══════════════════════════════════════════════════════════
// COVER PAGE
// ══════════════════════════════════════════════════════════
function coverPage() {
  return [
    spacer(), spacer(), spacer(), spacer(), spacer(), spacer(), spacer(),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: "QBITEL BRIDGE", font: "Arial", size: 80, bold: true, color: BRAND_DARK })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "SERIES A INVESTOR PITCH", font: "Arial", size: 32, color: BRAND_BLUE })] }),
    spacer(),
    new Paragraph({ alignment: AlignmentType.CENTER, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BRAND_BLUE, space: 4 } }, children: [] }),
    spacer(),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "Securing the Legacy Foundations of the Modern World", font: "Arial", size: 26, italics: true, color: "444444" })] }),
    spacer(), spacer(),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "Series A Target: \u20B920 Crore", font: "Arial", size: 28, bold: true, color: BRAND_DARK })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 }, children: [new TextRun({ text: "March 2026 | CONFIDENTIAL", font: "Arial", size: 22, color: "666666" })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "All figures in Indian Rupees (\u20B9) | India-First Go-to-Market", font: "Arial", size: 22, bold: true, color: BRAND_ACCENT })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Capital-efficient India strategy with 87% lower costs than US equivalent", font: "Arial", size: 20, color: "888888" })] }),
    spacer(), spacer(), spacer(), spacer(), spacer(),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "QBITEL Technologies Private Limited", font: "Arial", size: 20, color: "999999" })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Confidential \u2014 Do Not Distribute", font: "Arial", size: 18, color: "999999" })] }),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ══════════════════════════════════════════════════════════
// TABLE OF CONTENTS PAGE
// ══════════════════════════════════════════════════════════
function tocPage() {
  const items = [
    ["1.", "Executive Summary", "3"],
    ["2.", "The Market Crisis", "5"],
    ["3.", "Our Solution: The QBITEL Bridge Platform", "7"],
    ["4.", "Technology Deep Dive", "9"],
    ["5.", "Competitive Landscape", "11"],
    ["6.", "Business Model", "12"],
    ["7.", "Financial Projections (INR)", "14"],
    ["8.", "Use of Funds", "16"],
    ["9.", "Go-to-Market Strategy (India-First)", "17"],
    ["10.", "Product & Growth Roadmap", "19"],
    ["11.", "Team", "20"],
    ["12.", "The Ask", "21"],
  ];
  return [
    heading("Table of Contents"),
    new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: RED, space: 4 } }, children: [] }),
    spacer(),
    ...items.map(([num, title, pg]) =>
      new Paragraph({
        spacing: { after: 120 },
        tabStops: [{ type: TabStopType.RIGHT, position: 9360 }],
        children: [
          new TextRun({ text: `${num}  ${title}`, font: "Arial", size: 24 }),
          new TextRun({ text: `\t${pg}`, font: "Arial", size: 24, color: BRAND_BLUE }),
        ],
      })
    ),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ══════════════════════════════════════════════════════════
// MAIN CONTENT
// ══════════════════════════════════════════════════════════
const children = [
  ...coverPage(),
  ...tocPage(),

  // ═══ 1. EXECUTIVE SUMMARY ═══
  heading("1. Executive Summary"),
  spacer(),
  para("QBITEL Bridge is the only open-source platform that discovers unknown legacy protocols with AI, encrypts them with post-quantum cryptography, and defends them autonomously\u2014without replacing a single line of existing code."),
  spacer(),

  metricRow([
    { value: "\u20B925L Cr+", label: "Daily Transactions at Risk", sub: "Banking sector alone", color: RED, bg: "FFF5F5" },
    { value: "60%", label: "Fortune 500 on Legacy Systems", sub: "20-40 year old infrastructure", color: BRAND_ACCENT, bg: "FFF8F0" },
    { value: "5\u201310yr", label: "Quantum Threat Timeline", sub: "Encryption broken by 2033", color: BRAND_TEAL, bg: "F0FAF8" },
    { value: "35L+", label: "Cybersecurity Talent Gap", sub: "Unfilled roles globally", color: BRAND_DARK, bg: GRAY_LIGHT },
  ]),
  spacer(),

  heading("Three Converging Crises Create an Urgent, Durable Opportunity", HeadingLevel.HEADING_2),
  bullet("The Legacy Crisis: 60% of Fortune 500 companies run critical operations on 20-40 year old systems. Competitors treat manual reverse engineering as a \u20B91.7-8.4 Crore cost; QBITEL automates the process in hours, capturing this as pure margin."),
  bullet("The Quantum Threat: Within 5-10 years, quantum computers will render today\u2019s encryption (RSA, ECDSA) obsolete. Adversaries are currently harvesting data to decrypt later\u2014data stolen today is a breach tomorrow."),
  bullet("The Speed Gap: We face a human-scale defence against a machine-scale threat landscape. Human security teams take 65 minutes to respond to threats that compromise networks in milliseconds."),
  spacer(),

  calloutBox(
    "Investment Thesis",
    "QBITEL sits at the intersection of three secular trends\u2014legacy modernisation, post-quantum cryptography, and AI-driven security automation\u2014each independently a multi-billion dollar market. Our platform addresses all three simultaneously with a non-invasive, open-source approach that enterprises trust."
  ),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ INVESTMENT SNAPSHOT ═══
  heading("Investment Snapshot", HeadingLevel.HEADING_2),
  spacer(),
  makeTable(
    ["Parameter", "Details"],
    [
      ["Funding Round", "Series A Preferred Equity"],
      ["Amount Sought", "\u20B920 Crore"],
      ["Pre-Money Valuation", "\u20B960 Crore"],
      ["Post-Money Valuation", "\u20B980 Crore"],
      ["Investor Ownership", "25%"],
      ["Runway to Breakeven", "19 months"],
      ["Year 1 Target ARR", "\u20B97.2 Crore (6 customers)"],
      ["Year 2 Target ARR", "\u20B919.2 Crore (16 customers)"],
      ["Year 3 Target ARR", "\u20B940 Crore (36 customers)"],
      ["Gross Margin Target (Year 3)", "77%+"],
      ["Year 2 EBITDA", "+\u20B93.64 Crore (profitable)"],
      ["Target Close", "Q2 2026"],
      ["Minimum Cheque", "\u20B91 Crore"],
    ],
    [4000, 5360]
  ),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 2. THE MARKET CRISIS ═══
  heading("2. The Market Crisis: A Multi-Trillion Rupee Vulnerability"),
  spacer(),
  para("The global economy rests on ageing foundations that are architecturally incapable of resisting modern machine-speed attacks or the impending quantum decryption threat."),
  spacer(),

  makeTable(
    ["Sector", "The Risk", "Economic Impact"],
    [
      ["Banking & Finance", "92% of top 100 banks rely on undocumented COBOL mainframes for core operations. No native encryption, no API interfaces, no modern monitoring.", "\u20B925+ lakh crore in daily transactions at risk. \u20B942 lakh/hour downtime exposure. Systemic contagion risk."],
      ["Healthcare", "Medical devices average 6.2 unpatched vulnerabilities. Firmware updates blocked by \u20B91.7 Crore FDA recertification costs per device.", "\u20B910,920 Crore in HIPAA fines (2024). Patient health records valued at \u20B921,000-84,000 per record on dark web."],
      ["Manufacturing / Industrial", "SCADA/ICS systems lack native encryption. Patching risks catastrophic safety failures and extended production halts.", "Multi-crore production stoppages. Critical infrastructure blackouts. Potential for physical safety incidents."],
    ],
    [2000, 3860, 3500]
  ),
  spacer(),

  heading("The Quantum Countdown", HeadingLevel.HEADING_2),
  para("NIST finalised post-quantum cryptography standards (FIPS 203/204) in 2024, marking the official start of the compliance clock. The timeline is unambiguous:"),
  spacer(),
  makeTable(
    ["Timeline", "What Happens"],
    [
      ["2025\u20132026", "Critical liability window. Adversaries actively harvest encrypted data for future decryption. NIST FIPS 203/204 standards finalised. RBI IT Master Direction mandates quantum-readiness."],
      ["2027\u20132029", "Mandatory compliance enforcement begins. PCI-DSS 4.0 and EU DORA quantum requirements take effect. SEBI, IRDAI, TRAI deadlines."],
      ["2030\u20132033", "Quantum computers reach 1,000+ logical qubits. RSA-2048 and ECDSA encryption broken. All unprotected data retroactively compromised."],
    ],
    [2000, 7360]
  ),
  spacer(),

  heading("India-Specific Regulatory Catalysts (Mandatory Spend)", HeadingLevel.HEADING_2),
  makeTable(
    ["Regulation", "Sector", "Deadline", "Spend Driver"],
    [
      ["RBI IT Master Direction 2024", "Banking", "April 2025", "Quantum-readiness for all scheduled commercial banks"],
      ["CERT-In Incident Reporting", "All Critical Infra", "Active Now", "6-hour mandatory incident reporting"],
      ["SEBI Cybersecurity Circular", "Capital Markets", "2025", "Stock exchanges, brokers, depositories"],
      ["IRDAI Cybersecurity Guidelines", "Insurance", "2025", "Insurer cybersecurity compliance"],
      ["TRAI Telecom Security Rules", "Telecom", "2025", "Operator network security mandates"],
      ["ABDM Data Security Standards", "Healthcare", "2025", "Hospital and health data standards"],
      ["MeitY DPDP Act", "All Sectors", "2025-2026", "Data privacy and protection compliance"],
    ],
    [2800, 1800, 1200, 3560]
  ),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 3. OUR SOLUTION ═══
  heading("3. Our Solution: The QBITEL Bridge Platform"),
  spacer(),
  para("QBITEL Bridge provides a non-invasive overlay that modernises legacy environments through a zero-trust, autonomous, five-stage workflow. No rip-and-replace. No downtime. No custom code."),
  spacer(),

  makeTable(
    ["#", "Stage", "Capability", "Outcome"],
    [
      ["1", "DISCOVER", "AI listens to network traffic and reverse-engineers proprietary protocols\u2014zero documentation required.", "Complete protocol dictionary in 2\u20134 hours vs. 6\u201312 months manually. Cost: ~\u20B942L vs. \u20B94.2-16.8 Crore."],
      ["2", "PROTECT", "Communications wrapped in NIST Level 5 post-quantum encryption (ML-KEM / Kyber-1024) at network layer.", "Quantum-immune posture achieved in hours. Less than 1ms latency overhead. Immediate protection against harvest-now-decrypt-later."],
      ["3", "TRANSLATE", "Legacy protocols auto-converted to modern REST APIs and SDKs in 6 programming languages.", "Developer velocity accelerated from months to minutes. Unlocks \u20B94.2\u201342 Crore stalled digital transformation projects."],
      ["4", "COMPLY", "Audit-ready compliance reports for 9 major frameworks (PCI-DSS 4.0, HIPAA, SOC 2, NIST CSF, etc.) generated in under 10 minutes using blockchain-backed evidence.", "Eliminates weeks of manual evidence gathering. Reduces audit preparation costs by 80%+."],
      ["5", "OPERATE", "Agentic AI security engine autonomously handles 78% of attacks with sub-10-second response time and \u20B90.84 cost per event.", "Solves the 35 lakh-role global security talent gap. 390x faster response than human SOC teams."],
    ],
    [400, 1200, 3960, 3800]
  ),
  spacer(),

  calloutBox(
    "Key Differentiator: Non-Invasive Architecture",
    "QBITEL Bridge operates entirely at the network layer as a transparent proxy. Legacy systems see no changes to their interfaces, code, or configuration. This eliminates the primary barrier to legacy security upgrades\u2014operational risk\u2014and reduces deployment timelines from years to weeks."
  ),
  spacer(),

  heading("Seven Industry Verticals", HeadingLevel.HEADING_2),
  makeTable(
    ["Vertical", "Target Systems", "Key Protocols", "India Market Driver"],
    [
      ["Banking & Finance", "COBOL mainframes, core banking, SWIFT", "ISO 8583, SWIFT MT/MX, FIX", "RBI IT Master Direction 2024"],
      ["Healthcare", "Medical devices, EHR systems", "HL7, FHIR R4, DICOM, IEEE 11073", "ABDM Data Security Standards"],
      ["Critical Infrastructure", "Power grids, water, oil/gas, SCADA", "Modbus, DNP3, IEC 61850, OPC UA", "CERT-In / CERC mandates"],
      ["Automotive", "V2X security, connected vehicles, CAN", "IEEE 1609.2, CAN, C-V2X", "UNECE WP.29 R155/R156"],
      ["Aviation", "ADS-B, ACARS, flight systems", "ARINC 429/653, LDACS, ADS-B", "DGCA / EASA requirements"],
      ["Telecommunications", "SS7, 5G core, SIP/VoIP", "Diameter, SS7 MAP, SIP", "TRAI Security Rules"],
      ["BPO & Call Centres", "Legacy PBX, terminal access", "TN3270e, SIP, DTMF", "PCI-DSS voice compliance"],
    ],
    [2000, 2360, 2500, 2500]
  ),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 4. TECHNOLOGY DEEP DIVE ═══
  heading("4. Technology Deep Dive"),
  spacer(),

  heading("AI-Powered Protocol Discovery: Replacing \u20B916.8 Crore with \u20B942 Lakh", HeadingLevel.HEADING_2),
  makeTable(
    ["Metric", "Traditional Reverse Engineering", "QBITEL AI Discovery"],
    [
      ["Time to Completion", "6\u201312 months", "2\u20134 hours"],
      ["Cost Per Protocol", "\u20B94.2\u201316.8 Crore", "~\u20B942 Lakh"],
      ["Accuracy", "Variable (human error)", "89%+ (consistent & validated)"],
      ["Required Expertise", "Rare protocol engineers", "Any standard developer"],
      ["Scalability", "Linear cost growth", "Near-zero marginal cost"],
    ],
    [2500, 3430, 3430]
  ),
  spacer(),

  heading("Post-Quantum Cryptography Stack (NIST FIPS 203/204)", HeadingLevel.HEADING_2),
  bullet("ML-KEM (Kyber-1024): Quantum-safe key exchange using Module Lattice-Based Key Encapsulation. Provides NIST Security Level 5 protection against both classical and quantum adversaries."),
  bullet("ML-DSA (Dilithium-5): Digital signatures that remain valid even when quantum computers break RSA and ECDSA. Ensures document and transaction integrity post-quantum."),
  bullet("Invisible Performance: Protection delivered at wire-speed via Rust/DPDK implementation with <1ms overhead. Zero impact on existing legacy control loops or transaction latency."),
  spacer(),

  heading("Performance-Driven Polyglot Architecture", HeadingLevel.HEADING_2),
  makeTable(
    ["Layer", "Language", "Framework", "Technical Rationale"],
    [
      ["Data Plane", "Rust", "DPDK", "Memory-safe, wire-speed processing. Achieves <1ms encryption overhead without garbage collection pauses."],
      ["AI Engine", "Python", "PyTorch / LangGraph", "Access to the full AI/ML ecosystem. Semantic protocol learning via transformer-based architectures."],
      ["Control Plane", "Go", "gRPC", "High-concurrency service orchestration. Native gRPC support for efficient microservice communication."],
      ["UI Console", "TypeScript", "React", "Type-safe enterprise dashboards with real-time telemetry and threat visualisation."],
    ],
    [1800, 1500, 2200, 3860]
  ),
  spacer(),

  calloutBox(
    "Air-Gap Advantage",
    "Unlike cloud-dependent incumbents (CrowdStrike, Palo Alto), QBITEL Bridge operates 100% on-premises using open-source LLMs (Ollama/vLLM). This is a non-negotiable requirement for regulated sectors\u2014banking, defence (DRDO/ISRO), critical infrastructure\u2014that prohibit external data transmission."
  ),
  spacer(),

  heading("Performance Benchmarks", HeadingLevel.HEADING_2),
  makeTable(
    ["Capability", "Metric", "QBITEL Performance", "Industry Comparison"],
    [
      ["Protocol Discovery", "Time to results", "2\u20134 hours", "6\u201312 months (manual)"],
      ["PQC Encryption", "Per-message overhead", "<1ms", "10\u201350ms (software PQC)"],
      ["Streaming Pipeline", "Throughput", "1,00,000+ msg/sec", "Industry-leading"],
      ["Security Response", "Decision time", "<1 second", "65\u2013140 min (manual SOC)"],
      ["Autonomy Rate", "Auto-handled attacks", "78%", "0% (competitor playbooks)"],
      ["Compliance Reports", "Generation time", "<10 minutes", "Days\u2013weeks (manual)"],
      ["Platform Availability", "Uptime target", "99.999%", "99.99% (typical SaaS)"],
      ["Cost per Event", "Security event cost", "\u20B90.84", "\u20B9840\u20134,200 (traditional)"],
    ],
    [2200, 2000, 2560, 2600]
  ),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 5. COMPETITIVE LANDSCAPE ═══
  heading("5. Competitive Landscape"),
  spacer(),
  para("QBITEL protects the systems that market incumbents are architecturally incapable of reaching. We do not compete on features\u2014we compete on existence."),
  spacer(),

  makeTable(
    ["Capability", "CrowdStrike / EDR", "Claroty / Dragos", "Nozomi / Armis", "QBITEL Bridge"],
    [
      ["Legacy COBOL / Mainframe Support", "\u2717", "\u2717", "\u2717", "\u2713"],
      ["SCADA / ICS Protection", "Partial", "\u2713", "\u2713", "\u2713"],
      ["Post-Quantum Cryptography", "\u2717", "\u2717", "\u2717", "\u2713 NIST L5"],
      ["AI Protocol Discovery", "\u2717", "\u2717", "\u2717", "\u2713 Autonomous"],
      ["100% Air-Gapped / On-Premises", "\u2717", "Partial", "Partial", "\u2713 Full"],
      ["Auto Compliance Reporting", "Partial", "Partial", "Partial", "\u2713 9 Frameworks"],
      ["Autonomous Threat Response", "Partial", "\u2717", "\u2717", "\u2713 78% Auto"],
      ["Open-Source Core (Apache 2.0)", "\u2717", "\u2717", "\u2717", "\u2713"],
    ],
    [2400, 1500, 1500, 1500, 2460]
  ),
  spacer(),

  heading("Why Incumbents Cannot Catch Up", HeadingLevel.HEADING_2),
  bullet("Architectural lock-in: CrowdStrike and Palo Alto require OS-level agents\u2014incompatible with COBOL mainframes and embedded SCADA controllers by definition."),
  bullet("Missing quantum roadmap: No major incumbent has filed patents or published roadmaps for post-quantum cryptography integration as of Q4 2025."),
  bullet("Open-source trust deficit: Regulated industries demand auditable code. Proprietary black-box security is increasingly unacceptable in banking and critical infrastructure procurement."),
  bullet("No autonomous discovery: Existing OT vendors (Claroty, Dragos) do asset discovery of known protocols; QBITEL discovers and learns unknown, undocumented protocols."),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 6. BUSINESS MODEL ═══
  heading("6. Business Model: The Open-Source Flywheel"),
  spacer(),
  para("We leverage Apache 2.0 licensing to build the institutional trust required for critical infrastructure procurement, while capturing value through a robust enterprise subscription and marketplace ecosystem."),
  spacer(),

  makeTable(
    ["Revenue Stream", "Annual Value", "Model", "Notes"],
    [
      ["Core Platform Subscription", "\u20B91.2 Crore ARR/site", "SaaS \u2013 Annual", "Includes enterprise SLAs, priority support, dedicated engineering. Primary revenue driver."],
      ["LLM Feature Bundle", "+\u20B930 Lakh ARR/site", "Add-on Module", "Advanced autonomous reasoning and natural language threat investigation."],
      ["Quantum Readiness Sprint", "\u20B950 Lakh/engagement", "Professional Services", "Fixed-fee migration project to full post-quantum posture. High-margin, repeatable."],
      ["Marketplace Protocol Adapters", "30% platform fee", "Marketplace", "70% revenue share to creators. 1,000+ adapters projected by 2027. GMV target: \u20B950 Crore by 2028."],
    ],
    [2200, 2200, 1800, 3160]
  ),
  spacer(),

  heading("Unit Economics", HeadingLevel.HEADING_2),
  metricRow([
    { value: "\u20B91.5Cr", label: "ARR Per Enterprise Site", sub: "Core + LLM Bundle + PS", color: BRAND_DARK, bg: GRAY_LIGHT },
    { value: "77%", label: "Gross Margin (Yr 3)", sub: "Software-driven model", color: BRAND_GREEN, bg: "F0FFF4" },
    { value: "\u20B90.84", label: "Autonomous Event Cost", sub: "vs. \u20B9840-4,200 SOC cost", color: BRAND_TEAL, bg: "F0FAF8" },
    { value: "7.4x", label: "LTV:CAC Ratio", sub: "World-class unit economics", color: BRAND_ACCENT, bg: "FFF8F0" },
  ]),
  spacer(),

  calloutBox(
    "Open-Source as a Moat, Not a Liability",
    "Apache 2.0 licensing is the procurement unlock for critical infrastructure. Government agencies (GeM), regulated banks (RBI), and hospital networks (ABDM) require source code auditability. Our open-source core generates community-driven protocol adapters, competitive intelligence on threats, and institutional trust\u2014all of which accelerate our enterprise sales cycle."
  ),
  spacer(),

  heading("India Pricing (Converted to INR)", HeadingLevel.HEADING_2),
  makeTable(
    ["Tier", "Target Customer", "Subscription", "LLM Bundle", "PS (Year 1)", "Total Year 1 TCV"],
    [
      ["Enterprise", "Tier-1 Banks, Large Hospitals, Refineries", "\u20B91.20 Cr", "\u20B930L", "\u20B950L", "\u20B92.00 Cr"],
      ["Mid-Market", "PSBs, Mid-size Hospitals, Utilities", "\u20B980L", "\u20B920L", "\u20B930L", "\u20B91.30 Cr"],
      ["SME / Starter", "Small Banks, Clinics, BPO Centres", "\u20B950L", "\u20B915L", "\u20B920L", "\u20B985L"],
      ["Government / PSU", "Defence, DRDO, ISRO, PSUs (GeM)", "\u20B91.00 Cr", "\u20B925L", "\u20B940L", "\u20B91.65 Cr"],
    ],
    [1400, 2200, 1300, 1200, 1300, 1960]
  ),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 7. FINANCIAL PROJECTIONS ═══
  heading("7. Financial Projections"),
  spacer(),
  para("Three-year projections reflect a capital-efficient, land-and-expand SaaS model targeting regulated enterprise verticals with long contract durations and high switching costs."),
  spacer(),

  makeTable(
    ["Metric", "Year 1 (FY2026-27)", "Year 2 (FY2027-28)", "Year 3 (FY2028-29)"],
    [
      ["Enterprise Customer Count", "6", "16", "36 (30 India + 6 Intl)"],
      ["Ending ARR", "\u20B97.2 Crore", "\u20B919.2 Crore", "\u20B940 Crore"],
      ["Total Revenue", "\u20B94.68 Crore", "\u20B920 Crore", "\u20B942 Crore"],
      ["Gross Margin", "66%", "71%", "77%"],
      ["EBITDA", "(\u20B97.08 Cr)", "+\u20B93.64 Cr", "+\u20B911 Cr"],
      ["EBITDA Margin", "N/A (pre-scale)", "+18.2%", "+26.2%"],
      ["Net Revenue Retention", "120%", "128%", "135%"],
      ["Countries / Verticals", "1 / 3", "3 / 5", "5 / 7"],
    ],
    [2800, 2187, 2186, 2187]
  ),
  spacer(),

  heading("Key Assumptions", HeadingLevel.HEADING_2),
  bullet("Year 1 anchored by 3 Tier-1 banking clients from closed beta (LOIs signed) plus 3 healthcare/industrial customers from GA launch."),
  bullet("Average contract value: \u20B91.2 Crore ARR per site (Core subscription). LLM bundle and professional services are additional."),
  bullet("Gross margin expansion driven by automation reducing support headcount requirements as AI autonomous response handles increasing share of events."),
  bullet("Marketplace revenue modelled conservatively at 30% platform fee on independently submitted adapters; excludes proprietary adapter revenue."),
  spacer(),

  calloutBox(
    "ARR Growth: \u20B97.2 Cr \u2192 \u20B940 Cr in 36 Months",
    "This trajectory is supported by: (1) contracted pipeline from banking closed beta, (2) RBI/SEBI/CERT-In regulatory mandates forcing quantum migrations by 2025\u20132027, and (3) India-first cost advantage enabling 87% lower burn rate than US competitors."
  ),
  spacer(),

  heading("Year-over-Year P&L Summary (INR)", HeadingLevel.HEADING_2),
  makeTable(
    ["Line Item", "FY2026-27", "FY2027-28", "FY2028-29"],
    [
      ["Subscription Revenue", "\u20B93.06 Cr", "\u20B912.48 Cr", "\u20B926.64 Cr"],
      ["LLM Bundle Revenue", "\u20B972 Lakh", "\u20B93.36 Cr", "\u20B97.56 Cr"],
      ["Professional Services", "\u20B972 Lakh", "\u20B93.20 Cr", "\u20B95.40 Cr"],
      ["Managed Detection & MDR", "\u20B918 Lakh", "\u20B996 Lakh", "\u20B92.40 Cr"],
      ["Total Revenue", "\u20B94.68 Cr", "\u20B920.00 Cr", "\u20B942.00 Cr"],
      ["Cost of Revenue (COGS)", "(\u20B91.60 Cr)", "(\u20B95.84 Cr)", "(\u20B99.84 Cr)"],
      ["Gross Profit", "\u20B93.08 Cr", "\u20B914.16 Cr", "\u20B932.16 Cr"],
      ["Operating Expenses", "(\u20B910.16 Cr)", "(\u20B910.52 Cr)", "(\u20B921.16 Cr)"],
      ["EBITDA", "(\u20B97.08 Cr)", "+\u20B93.64 Cr", "+\u20B911.00 Cr"],
    ],
    [3000, 2120, 2120, 2120]
  ),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 8. USE OF FUNDS ═══
  heading("8. Use of Funds"),
  spacer(),
  para("QBITEL is raising \u20B920 Crore in Series A funding to achieve product completion, regulatory certifications, and India-first market leadership in quantum-safe legacy security. India\u2019s 87% cost advantage over the US means \u20B920 Crore delivers the same output as $18M+ raised by a Silicon Valley competitor."),
  spacer(),

  makeTable(
    ["%", "Category", "Amount (INR)", "Specific Investments"],
    [
      ["40%", "Engineering & Product", "\u20B98.0 Crore", "Finalise autonomous AI engine; expand discovery accuracy to 95%+; PQC domain optimisation; complete SDK for 6 languages; harden air-gap deployment."],
      ["25%", "Go-to-Market", "\u20B95.0 Crore", "6 Account Executives, 4 Solution Engineers, VP Sales/Marketing hire; vertical marketing for banking, healthcare, energy; events & content."],
      ["20%", "Certifications & Partnerships", "\u20B94.0 Crore", "CERT-In empanelment; SOC 2 Type II; ISO 27001; NIST FIPS validation; HSM integration; GeM listing; SI partnerships (TCS, Wipro, Infosys)."],
      ["15%", "Operations & Working Capital", "\u20B93.0 Crore", "Office (Bengaluru/Hyderabad), IT infrastructure, legal, finance, insurance, contingency reserve."],
    ],
    [500, 2000, 1600, 5260]
  ),
  spacer(),

  metricRow([
    { value: "\u20B920Cr", label: "Total Series A Raise", sub: "India-first capital efficiency", color: BRAND_DARK, bg: GRAY_LIGHT },
    { value: "22mo", label: "Runway", sub: "Breakeven at Month 19", color: BRAND_BLUE, bg: BRAND_LIGHT },
    { value: "\u20B919.2Cr", label: "Expected ARR at Y2 End", sub: "16 enterprise customers", color: BRAND_GREEN, bg: "F0FFF4" },
    { value: "71%+", label: "Target Gross Margin", sub: "Software-driven expansion", color: BRAND_ACCENT, bg: "FFF8F0" },
  ]),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 9. GO-TO-MARKET ═══
  heading("9. Go-to-Market Strategy (India-First)"),
  spacer(),

  heading("Why India First?", HeadingLevel.HEADING_2),
  bullet("87% lower engineering costs: \u20B912 Crore vs \u20B975 Crore (US equivalent) for a 20-person engineering team"),
  bullet("World-class cryptography and AI talent at 5-6x lower cost than Silicon Valley"),
  bullet("Government 'Make in India' preference: GeM procurement, PMGDISHA, defence procurement policy"),
  bullet("Air-gapped on-premise AI requirement: Cloud tools barred from DRDO/ISRO/defence environments"),
  bullet("CERT-In empanelment: Mandatory for all government cybersecurity contracts"),
  bullet("Data residency: DPDP Act requires Indian data storage\u2014cloud-only vendors are disadvantaged"),
  spacer(),

  heading("Sales Motion (Proven 5-Step Process)", HeadingLevel.HEADING_2),
  numberedItem("Free Quantum Risk Assessment (48-hour passive network tap) \u2014 creates board-level urgency"),
  numberedItem("Compliance Gap Briefing (60 min for CISO/CTO/Board) \u2014 quantifies regulatory exposure"),
  numberedItem("Technical Deep-Dive (90 min with architects) \u2014 live demonstration of AI discovery"),
  numberedItem("30-Day Paid Proof of Value (\u20B921 Lakh / ~$25K) on one network segment"),
  numberedItem("Annual Subscription (\u20B954.6L\u20131.01 Cr ARR per site) with 3-year preferred terms"),
  spacer(),

  heading("Channel Strategy", HeadingLevel.HEADING_2),
  makeTable(
    ["Channel", "Year 1 Mix", "Year 2 Mix", "Key Partners"],
    [
      ["Direct Enterprise Sales", "70%", "50%", "In-house AE team (10 reps)"],
      ["Government (GeM/PSU)", "15%", "20%", "CERT-In empanelment, GeM listing, DPP compliance"],
      ["MSSP Partners", "10%", "20%", "Tata Communications, Wipro Cyber Defence, Airtel Secure"],
      ["System Integrators", "5%", "10%", "TCS, Infosys, HCL, Tech Mahindra, Deloitte"],
    ],
    [2200, 1200, 1200, 4760]
  ),
  spacer(),

  heading("Target Customer Pipeline (Year 1)", HeadingLevel.HEADING_2),
  makeTable(
    ["#", "Segment", "Target Profile", "Est. ARR (INR)", "Status"],
    [
      ["1", "Banking (Private)", "ICICI / HDFC / Kotak / Axis Bank", "\u20B91.50 Cr", "Closed Beta"],
      ["2", "Banking (PSB)", "SBI / PNB / Bank of Baroda", "\u20B91.20 Cr", "POV Proposal"],
      ["3", "Healthcare", "Apollo / Fortis / Max / Narayana", "\u20B91.00 Cr", "Initial Discussion"],
      ["4", "Critical Infra", "NTPC / PowerGrid / Adani Power", "\u20B91.50 Cr", "RFP Response"],
      ["5", "Defence / PSU", "DRDO / ISRO / BEL (via GeM)", "\u20B91.00 Cr", "GeM Registration"],
      ["6", "BPO / Telecom", "TCS BPS / Wipro BPS / Genpact", "\u20B91.00 Cr", "Proof of Concept"],
    ],
    [400, 1500, 2960, 1700, 2800]
  ),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 10. ROADMAP ═══
  heading("10. Product & Growth Roadmap"),
  spacer(),
  makeTable(
    ["Phase", "Timeline", "Milestones"],
    [
      ["Foundation", "Q4 2025", "Protocol discovery MVP delivered. Tier-1 banking closed beta launched with 3 anchor customers. SOC 2 Type I audit initiated. Core team of 18 engineers onboarded."],
      ["General Availability", "Q1 2026", "Platform GA release. SOC 2 Type I certification achieved. First 6 enterprise contracts executed. Marketplace launched with 50 founding protocol adapters."],
      ["Vertical Expansion", "Q3 2026", "Specialised energy and healthcare modules released. Marketplace reaches 500+ adapters. First CERT-In empanelment. Second sales cohort closes 10 net-new logos."],
      ["Federal & Scale", "Q1 2027", "CERT-In empanelment granted. ISO 27001 certified. Marketplace reaches 1,000+ adapters. 16 enterprise customers active. \u20B919.2 Cr ARR. EBITDA positive (+\u20B93.64 Cr)."],
      ["Market Leadership", "2028", "Multi-vertical leadership. 36 customers (30 India + 6 international). \u20B940 Cr ARR at 77% gross margin. International expansion to UAE/Singapore. Exploration of strategic M&A or NSE SME IPO path."],
    ],
    [1800, 1200, 6360]
  ),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 11. TEAM ═══
  heading("11. Team"),
  spacer(),
  makeTable(
    ["Role", "Count", "Key Expertise"],
    [
      ["CTO / Co-Founder", "1", "Cryptography PhD, NIST PQC standardisation contributor"],
      ["CEO / Co-Founder", "1", "Enterprise sales, cybersecurity strategy, ex-Big 4"],
      ["VP Engineering", "1", "Distributed systems, Rust/Go/Python polyglot, ex-Google/Microsoft"],
      ["Senior AI/ML Engineers", "4", "LLM fine-tuning, protocol discovery, anomaly detection, IIT/IISc"],
      ["PQC Engineers", "3", "Kyber, Dilithium, HSM integration, threshold cryptography"],
      ["Security Researchers", "4", "MITRE ATT&CK, threat hunting, vulnerability research, CERT-In"],
      ["Platform Engineers", "4", "Kubernetes, eBPF, Istio, DPDK, Kafka, cloud-native"],
      ["Enterprise Sales", "3", "Banking, healthcare, government verticals, enterprise SaaS"],
      ["Solution Engineers", "2", "Pre-sales, POV execution, customer onboarding"],
      ["QA / DevSecOps", "3", "CI/CD, SAST/DAST, compliance testing, SOC 2 readiness"],
    ],
    [2800, 800, 5760]
  ),
  spacer(),
  heading("Board Composition (Post-Investment)", HeadingLevel.HEADING_2),
  bullet("2 Founders (CEO + CTO)"),
  bullet("2 Lead Investor nominees"),
  bullet("1 Independent Director (industry veteran)"),
  new Paragraph({ children: [new PageBreak()] }),

  // ═══ 12. THE ASK ═══
  heading("12. The Ask"),
  spacer(), spacer(),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "We Are Raising", font: "Arial", size: 24, color: BRAND_BLUE })] }),
  spacer(),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "\u20B920 Crore \u2014 Series A", font: "Arial", size: 44, bold: true, color: BRAND_DARK })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "To lead the quantum-safe modernisation market and protect the infrastructure the world runs on.", font: "Arial", size: 24, italics: true, color: "444444" })] }),
  spacer(),

  calloutBox(
    "Why Now?",
    "The \u20B920 Crore Series A is structured to reach EBITDA breakeven by Month 19 and \u20B919.2 Cr ARR by Year 2 end. Three catalysts make this the optimal investment window: (1) NIST standards finalised in 2024 have started enterprise procurement cycles; (2) RBI IT Master Direction mandates quantum-readiness for all scheduled banks; (3) India\u2019s 87% cost advantage means we reach profitability before needing a Series B."
  ),
  spacer(),

  heading("Series A Investor Rights & Structure", HeadingLevel.HEADING_2),
  bullet("Preferred equity with standard Series A protective provisions."),
  bullet("Board composition: 2 Founders + 2 Lead Investors + 1 Independent Director."),
  bullet("Pro-rata rights through Series B for investors committing \u20B92 Crore+."),
  bullet("1x Non-Participating Liquidation Preference."),
  bullet("Weighted Average Broad-Based Anti-Dilution."),
  bullet("15% ESOP pool (post-money, fully diluted)."),
  bullet("Target close: Q2 2026. Minimum cheque: \u20B91 Crore."),
  spacer(),

  heading("Exit Scenarios (Investor Returns)", HeadingLevel.HEADING_2),
  makeTable(
    ["Scenario", "Exit Valuation (INR)", "Revenue Multiple", "MOIC", "IRR"],
    [
      ["Conservative (Strategic Acq.)", "\u20B9300 Crore", "7.5x Yr 3 ARR", "2.5x", "~40%"],
      ["Base Case (Strategic Acq.)", "\u20B9600 Crore", "15x Yr 3 ARR", "5.4x", "~70%"],
      ["Bull Case (NSE IPO / Premium)", "\u20B9900 Crore", "22.5x Yr 3 ARR", "8.1x", "~95%"],
    ],
    [2500, 2200, 1800, 1000, 1860]
  ),
  spacer(),
  boldPara("Strategic Acquirers: Palo Alto Networks, CrowdStrike, IBM, Cisco, Thales Group, TCS, Wipro, Infosys"),
  spacer(), spacer(),

  new Paragraph({ alignment: AlignmentType.CENTER, border: { top: { style: BorderStyle.SINGLE, size: 4, color: BRAND_BLUE, space: 8 } }, children: [] }),
  spacer(),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "The window to become quantum-safe before adversaries exploit harvested data is closing.", font: "Arial", size: 24, bold: true, color: BRAND_DARK })] }),
  spacer(),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "QBITEL Bridge is how the world\u2019s most critical systems survive the next decade.", font: "Arial", size: 22, color: "444444" })] }),
  spacer(),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "This document is confidential and intended solely for prospective accredited investors.", font: "Arial", size: 18, italics: true, color: "999999" })] }),
];

// ── Build Document ──
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: BRAND_DARK },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BRAND_BLUE, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: BRAND_BLUE },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 } },
    ],
  },
  numbering,
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [
          new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: BRAND_BLUE, space: 4 } },
            children: [new TextRun({ text: "QBITEL BRIDGE  |  SERIES A  |  CONFIDENTIAL", font: "Arial", size: 16, bold: true, color: "666666" })],
          }),
        ],
      }),
    },
    footers: {
      default: new Footer({
        children: [
          new Paragraph({
            border: { top: { style: BorderStyle.SINGLE, size: 2, color: BRAND_BLUE, space: 4 } },
            children: [
              new TextRun({ text: "Confidential \u2014 Do Not Distribute", font: "Arial", size: 14, color: "999999" }),
              new TextRun({ text: "\tPage ", font: "Arial", size: 14, color: "999999" }),
              new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 14, color: "999999" }),
            ],
            tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
          }),
        ],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/Users/prabakarankannan/qbitel/docs/QBITEL_Investor_Pitch_2026_INR.docx", buffer);
  console.log("Investor Pitch document created successfully!");
});
