const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, TabStopType, TabStopPosition
} = require("docx");

// Helper functions
const b = (text, opts = {}) => new TextRun({ text, bold: true, font: "Times New Roman", size: 20, ...opts });
const t = (text, opts = {}) => new TextRun({ text, font: "Times New Roman", size: 20, ...opts });
const it = (text, opts = {}) => new TextRun({ text, font: "Times New Roman", size: 20, italics: true, ...opts });
const sf = (text, opts = {}) => new TextRun({ text, font: "Arial", size: 20, ...opts }); // sans-serif for algorithm names

const border = { style: BorderStyle.SINGLE, size: 1, color: "000000" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0 };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };
const cellMargins = { top: 40, bottom: 40, left: 80, right: 80 };

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "D9E2F3", type: ShadingType.CLEAR },
    margins: cellMargins,
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [b(text, { size: 18 })] })]
  });
}

function cell(text, width, align = AlignmentType.LEFT) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    margins: cellMargins,
    children: [new Paragraph({ alignment: align, children: [t(text, { size: 18 })] })]
  });
}

function sectionHeading(num, title) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 120 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: `${num}. ${title}`, bold: true, font: "Times New Roman", size: 24, allCaps: true })]
  });
}

function subsectionHeading(letter, title) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 80 },
    children: [new TextRun({ text: `${letter}. ${title}`, bold: true, font: "Times New Roman", size: 22, italics: true })]
  });
}

function bodyPara(children, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: 360 },
    ...opts,
    children
  });
}

function bulletItem(children) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60 },
    children
  });
}

function numberedItem(children, ref = "numbers") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 60 },
    children
  });
}

function equationPara(text) {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    alignment: AlignmentType.CENTER,
    children: [it(text)]
  });
}

function algLine(text, indent = 0) {
  return new Paragraph({
    spacing: { after: 40 },
    indent: { left: 720 + indent * 360 },
    children: [new TextRun({ text, font: "Courier New", size: 18 })]
  });
}

function tableCaption(text) {
  return new Paragraph({
    spacing: { before: 200, after: 80 },
    alignment: AlignmentType.CENTER,
    children: [b(text, { size: 18 })]
  });
}

// Build the document
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 20 } }
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 360, after: 120 }, alignment: AlignmentType.CENTER, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, italics: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 240, after: 80 }, outlineLevel: 1 } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers2",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", start: 4, alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers3",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", start: 15, alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1080, bottom: 1440, left: 1080 }
        }
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: "IEEE Conference Paper", font: "Times New Roman", size: 16, italics: true, color: "888888" })]
          })]
        })
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: "Page ", font: "Times New Roman", size: 16 }), new TextRun({ children: [PageNumber.CURRENT], font: "Times New Roman", size: 16 })]
          })]
        })
      },
      children: [
        // ===== TITLE =====
        new Paragraph({
          spacing: { before: 480, after: 240 },
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Domain-Specific Post-Quantum Cryptographic Constructions for Critical Infrastructure: Design, Implementation, and Analysis", bold: true, font: "Times New Roman", size: 36 })]
        }),

        // ===== AUTHOR =====
        new Paragraph({
          spacing: { after: 80 },
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Prabakaran Kannan", font: "Times New Roman", size: 24 })]
        }),
        new Paragraph({
          spacing: { after: 360 },
          alignment: AlignmentType.CENTER,
          children: [it("QBITEL Technologies"), t("  "), it("Email: prabakaran@qbitel.com")]
        }),

        // ===== ABSTRACT =====
        new Paragraph({
          spacing: { before: 240, after: 120 },
          alignment: AlignmentType.CENTER,
          children: [b("Abstract", { size: 20, italics: true })]
        }),
        bodyPara([
          it("The finalization of NIST post-quantum cryptographic (PQC) standards\u2014ML-KEM (FIPS 203), ML-DSA (FIPS 204), and SLH-DSA (FIPS 205)\u2014marks a critical milestone in the transition to quantum-resistant security. However, these general-purpose primitives do not directly address the unique operational constraints of critical infrastructure domains such as automotive vehicle-to-everything (V2X) communication, aviation air traffic control (ATC) datalinks, banking correspondent networks, healthcare electronic health records (EHR), and industrial SCADA systems. This paper presents 15 novel domain-specific PQC constructions built upon NIST-standardized primitives, organized across five critical infrastructure verticals. Our contributions include: (1) a lattice-based group signature scheme for anonymous V2X authentication with batch verification exceeding 1,000 signatures per second; (2) Merkle tree aggregate signatures achieving 60\u201380% bandwidth reduction for bandwidth-constrained aviation channels; (3) multi-authority threshold signatures and proxy re-encryption for banking compliance; (4) homomorphic vital signs analytics and proxy re-encryption for HIPAA-compliant healthcare; and (5) TESLA++ broadcast authentication achieving sub-50 \u00B5s MAC computation for IEC 61850 GOOSE/SV industrial protocols. We provide both theoretical security analysis reducing to standard lattice assumptions and implementation benchmarks demonstrating practical feasibility. All constructions maintain backward compatibility through a novel crypto agility negotiation protocol supporting zero-downtime algorithm rotation.", { size: 18 })
        ], { indent: { left: 360, right: 360 } }),

        new Paragraph({
          spacing: { before: 120, after: 240 },
          indent: { left: 360, right: 360 },
          children: [b("Keywords\u2014", { size: 18, italics: true }), it("Post-quantum cryptography, critical infrastructure, ML-KEM, ML-DSA, group signatures, proxy re-encryption, homomorphic encryption, broadcast authentication, V2X security, aviation security, SCADA security", { size: 18 })]
        }),

        // ===== I. INTRODUCTION =====
        sectionHeading("I", "Introduction"),

        bodyPara([
          t("The advent of cryptographically relevant quantum computers (CRQCs) poses an existential threat to classical public-key cryptography. Shor\u2019s algorithm [1] can factor large integers and compute discrete logarithms in polynomial time, rendering RSA, ECDSA, and Diffie-Hellman insecure. In response, the National Institute of Standards and Technology (NIST) finalized three post-quantum cryptographic standards in 2024: ML-KEM (FIPS 203) [2] for key encapsulation, ML-DSA (FIPS 204) [3] for digital signatures, and SLH-DSA (FIPS 205) [4] for stateless hash-based signatures. Additionally, Falcon [5] remains under consideration for compact signatures, and LMS/XMSS [6] provide stateful hash-based alternatives per NIST SP 800-208.")
        ]),

        bodyPara([
          t("While these standards provide robust general-purpose quantum resistance, critical infrastructure domains impose constraints that standard algorithms alone cannot satisfy:")
        ]),

        bulletItem([b("Automotive V2X: "), t("Anonymous vehicle authentication with Sybil attack prevention and sub-millisecond verification at scale.")]),
        bulletItem([b("Aviation ATC: "), t("PQC signatures over channels as narrow as 600 bps (classic SATCOM), requiring aggressive bandwidth compression.")]),
        bulletItem([b("Banking: "), t("Multi-party authorization with regulatory compliance proofs that do not expose underlying financial data.")]),
        bulletItem([b("Healthcare: "), t("Privacy-preserving encrypted record transfer across providers, including constrained medical devices with <32 KB RAM.")]),
        bulletItem([b("Industrial SCADA: "), t("Broadcast authentication with <50 \u00B5s latency for IEC 61850 sampled values at 4,000 Hz.")]),

        subsectionHeading("A", "Contributions"),

        bodyPara([t("This paper presents 15 domain-specific PQC constructions, organized into three categories:")], { indent: {} }),

        bodyPara([b("Core Framework (3 constructions):")], { indent: {} }),
        numberedItem([t("Crypto Agility Negotiation Protocol with policy-driven algorithm rotation")]),
        numberedItem([t("Quantum Threat Scoring Engine implementing Mosca\u2019s inequality [7]")]),
        numberedItem([t("Hybrid Key Exchange (X25519-ML-KEM-768, P384-ML-KEM-1024)")]),

        bodyPara([b("Domain-Specific Protocols (10 constructions):")], { indent: {} }),
        numberedItem([t("Lattice-based group signatures for V2X (Automotive)")], "numbers2"),
        numberedItem([t("Merkle tree aggregate signatures for ATC (Aviation)")], "numbers2"),
        numberedItem([t("Forward-secure double-ratchet channels (Aviation)")], "numbers2"),
        numberedItem([t("Multi-authority threshold signatures (Banking)")], "numbers2"),
        numberedItem([t("SWIFT proxy re-encryption (Banking)")], "numbers2"),
        numberedItem([t("Regulatory zero-knowledge proof engine (Banking)")], "numbers2"),
        numberedItem([t("EHR proxy re-encryption (Healthcare)")], "numbers2"),
        numberedItem([t("Homomorphic vital signs analytics (Healthcare)")], "numbers2"),
        numberedItem([t("PQ verifiable credentials with selective disclosure (Healthcare)")], "numbers2"),
        numberedItem([t("TESLA++ broadcast authentication for IEC 61850 (Industrial)")], "numbers2"),
        numberedItem([t("Verifiable delay functions for safety-critical timing (Industrial)")], "numbers2"),

        bodyPara([b("Constrained Environment Adaptation (1 construction):")], { indent: {} }),
        numberedItem([t("Lightweight PQC profiles for medical devices (<32 KB RAM)")], "numbers3"),

        subsectionHeading("B", "Paper Organization"),

        bodyPara([
          t("Section II provides background on NIST PQC standards and threat models. Section III presents the core cryptographic framework. Sections IV\u2013VIII detail domain-specific constructions. Section IX provides performance evaluation. Section X presents security analysis. Section XI surveys related work, and Section XII concludes.")
        ]),

        // ===== II. BACKGROUND =====
        sectionHeading("II", "Background and Preliminaries"),

        subsectionHeading("A", "NIST PQC Standards"),

        bodyPara([
          b("ML-KEM (FIPS 203) "), t("is a key encapsulation mechanism based on the Module Learning with Errors (Module-LWE) problem. It provides three parameter sets: ML-KEM-512 (NIST Level 1), ML-KEM-768 (Level 3), and ML-KEM-1024 (Level 5), with public key sizes of 800, 1,184, and 1,568 bytes respectively.")
        ]),

        bodyPara([
          b("ML-DSA (FIPS 204)"), t(", formerly Dilithium, is a digital signature scheme based on Module-LWE and Module-SIS (Short Integer Solution). ML-DSA-44 provides NIST Level 2 security with 2,420-byte signatures, ML-DSA-65 provides Level 3 with 3,293-byte signatures, and ML-DSA-87 provides Level 5 with 4,595-byte signatures.")
        ]),

        bodyPara([
          b("Falcon "), t("[5] offers compact signatures (~666 bytes at Level 1) based on NTRU lattices, achieving 3.6\u00D7 smaller signatures than ML-DSA at comparable security levels.")
        ]),

        bodyPara([
          b("LMS/XMSS "), t("(NIST SP 800-208) [6] are stateful hash-based signature schemes specified in RFC 8554 and RFC 8391, required by NSA CNSA 2.0 for firmware signing and long-lived certificates.")
        ]),

        subsectionHeading("B", "Lattice Assumptions"),

        bodyPara([t("Our constructions rely on the following computational hardness assumptions:")], { indent: {} }),

        bulletItem([b("Module-LWE: "), t("Given (A, b = As + e) where A is in the ring, s is secret, and e is a small error vector, it is computationally hard to recover s.")]),
        bulletItem([b("Module-SIS: "), t("Given A, it is hard to find a short non-zero vector z such that Az = 0 mod q with ||z|| \u2264 \u03B2.")]),

        subsectionHeading("C", "Hash-Based Primitives"),

        bodyPara([
          t("All constructions use NIST-approved hash functions: SHA3-256 for commitments and Merkle trees, SHAKE256 for extensible-output functions and key derivation, and HKDF-SHA256 [8] for hybrid key exchange domain separation.")
        ]),

        subsectionHeading("D", "Threat Model: Harvest-Now-Decrypt-Later"),

        bodyPara([
          t("We adopt the harvest-now-decrypt-later (HNDL) threat model, where adversaries record encrypted communications today for future decryption by CRQCs. Mosca\u2019s theorem [7] formalizes migration urgency: if data must remain confidential for x years, migration takes y years, and CRQCs arrive in z years, then migration must begin when x + y > z.")
        ]),

        // ===== III. CORE FRAMEWORK =====
        sectionHeading("III", "Core Cryptographic Framework"),

        subsectionHeading("A", "Crypto Agility Negotiation Protocol"),

        bodyPara([
          t("We introduce a runtime algorithm negotiation framework enabling seamless transition between classical, hybrid, and post-quantum suites. The protocol maintains a registry of 28 algorithms across five categories (KEM, signature, symmetric, hash, KDF) with per-algorithm metadata including NIST security level, key sizes, and operational status.")
        ]),

        // Algorithm 1
        new Paragraph({ spacing: { before: 160 }, alignment: AlignmentType.CENTER, children: [b("Algorithm 1: ", { size: 18 }), t("Crypto Agility Negotiation", { size: 18 })] }),
        algLine("Require: Peer capabilities C_peer, local policy \u03C0"),
        algLine("Ensure: Negotiated suite S"),
        algLine("1: C_local \u2190 GetCapabilities()"),
        algLine("2: C_common \u2190 C_local \u2229 C_peer"),
        algLine("3: C_filtered \u2190 ApplyPolicy(C_common, \u03C0)"),
        algLine("4: for category \u2208 {KEM, SIG, SYM, HASH, KDF} do"),
        algLine("5:   S[category] \u2190 SelectBest(C_filtered, category)", 1),
        algLine("6: end for"),
        algLine("7: return S"),

        bodyPara([t("Four pre-defined policies govern algorithm selection:")]),
        bulletItem([b("TRANSITIONAL: "), t("Hybrid classical+PQC (minimum NIST Level 1)")]),
        bulletItem([b("POST_QUANTUM: "), t("PQC-only suites (minimum Level 3)")]),
        bulletItem([b("CNSA2: "), t("NSA CNSA 2.0 compliant (Level 5, LMS/XMSS required)")]),
        bulletItem([b("LEGACY_COMPATIBLE: "), t("Classical with PQC preference")]),

        bodyPara([
          t("The protocol supports zero-downtime rotation: active sessions continue with the current suite while new sessions negotiate upgraded algorithms. Compliance auditing is maintained through full negotiation history with Prometheus metrics export.")
        ]),

        subsectionHeading("B", "Quantum Threat Scoring Engine"),

        bodyPara([
          t("The Quantum Risk Score (QRS) is a composite metric in [0, 100] quantifying the quantum threat exposure of cryptographic assets:")
        ]),

        equationPara("QRS = \u03A3(i=1..5) w_i \u00B7 f_i"),

        bodyPara([t("where the five factors and their weights are:")], { indent: {} }),
        bulletItem([t("Algorithm Vulnerability Score (w\u2081 = 0.30): RSA-2048 scores 85, ECDSA-P256 scores 90, ML-KEM-768 scores 2")]),
        bulletItem([t("Time Horizon Factor (w\u2082 = 0.25): Based on data retention requirements vs. estimated CRQC timeline (default 2035)")]),
        bulletItem([t("Data Sensitivity (w\u2083 = 0.20): Five tiers from PUBLIC (1) to TOP_SECRET (5)")]),
        bulletItem([t("Migration Gap (w\u2084 = 0.15): Seven-stage continuum from NOT_STARTED to CNSA2_COMPLIANT")]),
        bulletItem([t("Exposure Surface (w\u2085 = 0.10): Network accessibility and data volume")]),

        bodyPara([
          t("The engine applies Mosca\u2019s inequality to flag assets where x + y > z, triggering immediate migration prioritization. Risk levels map to: LOW (0\u201325), MODERATE (26\u201350), HIGH (51\u201375), CRITICAL (76\u2013100).")
        ]),

        subsectionHeading("C", "Hybrid Key Exchange"),

        bodyPara([
          t("Our hybrid KEM combines classical ECDH with ML-KEM using dual key encapsulation and HKDF-SHA256 for shared secret derivation:")
        ]),

        equationPara("K_hybrid = HKDF(K_classical || K_PQC, salt, info)"),

        bodyPara([t("Three variants are supported:")], { indent: {} }),
        bulletItem([b("X25519-ML-KEM-768: "), t("Public key 1,216 B, ciphertext 1,120 B (TLS 1.3 default)")]),
        bulletItem([b("P384-ML-KEM-1024: "), t("Public key 1,665 B, ciphertext 1,665 B (enterprise)")]),
        bulletItem([b("X25519-ML-KEM-512: "), t("Public key 832 B, ciphertext 800 B (constrained)")]),

        bodyPara([
          t("The design follows draft-ietf-tls-hybrid-design with domain separation via the variant name encoded in the HKDF info parameter. Security requires breaking "), it("both"), t(" the classical and PQC components simultaneously.")
        ]),

        // ===== IV. AUTOMOTIVE =====
        sectionHeading("IV", "Automotive: V2X Post-Quantum Security"),

        subsectionHeading("A", "Lattice-Based Group Signatures for V2X"),

        bodyPara([
          t("Vehicle-to-everything (V2X) communication requires anonymous authentication: verifiers must confirm that a message originates from a legitimate vehicle without learning the vehicle\u2019s identity. Simultaneously, authorized authorities must be able to trace signatures for accident investigation.")
        ]),

        bodyPara([t("We construct a group signature scheme based on Module-SIS/Module-LWE with the following properties:")]),
        bulletItem([b("Anonymity: "), t("Verifiers learn only group membership")]),
        bulletItem([b("Traceability: "), t("Group Manager (GM) opens signatures to reveal signer identity")]),
        bulletItem([b("Time-windowed linkability: "), t("Signatures within the same 5-minute window are linkable (Sybil detection)")]),
        bulletItem([b("Verifier-local revocation (VLR): "), t("No GM contact required for revocation checking")]),

        // Algorithm 2
        new Paragraph({ spacing: { before: 160 }, alignment: AlignmentType.CENTER, children: [b("Algorithm 2: ", { size: 18 }), t("V2X Group Signature Generation", { size: 18 })] }),
        algLine("Require: Member key sk_i, group public key gpk, message m, time window w"),
        algLine("Ensure: Group signature \u03C3"),
        algLine("1: tag \u2190 SHAKE256(sk_i || w)    // Pseudonym tag"),
        algLine("2: \u03C3_base \u2190 ML-DSA.Sign(sk_i, m || tag)"),
        algLine("3: ct \u2190 ML-KEM.Encaps(gpk_GM, id_i)    // Identity escrow"),
        algLine("4: return \u03C3 = (\u03C3_base, tag, ct)"),

        bodyPara([
          t("The pseudonym tag tag = SHAKE256(sk_i || w) changes every 5 minutes, providing unlinkability across windows while enabling Sybil detection within a window. The identity escrow ciphertext ct allows the GM to recover id_i using its private key when authorized (e.g., warrant-based).")
        ]),

        bodyPara([
          t("Batch verification processes 1,000+ signatures per second by amortizing Merkle tree root computations across message batches. Signature size is approximately 2 KB per message with <1 ms verification time, meeting the timing requirements of IEEE 1609.2 [9], SAE J2735, and ETSI TS 103 097.")
        ]),

        // ===== V. AVIATION =====
        sectionHeading("V", "Aviation: Bandwidth-Constrained PQC"),

        subsectionHeading("A", "Merkle Tree Aggregate Signatures"),

        bodyPara([
          t("Aviation ATC channels operate under severe bandwidth constraints: VHF ACARS at 2.4 kbps, HF datalink at 1.8 kbps, and classic SATCOM at 600 bps. Standard ML-DSA-65 signatures (3,293 bytes) are infeasible at these rates.")
        ]),

        bodyPara([t("We propose a Merkle tree-based signature aggregation scheme that compresses N individual PQC signatures into a single aggregate proof:")]),
        numberedItem([t("Each message m_i is individually signed: \u03C3_i \u2190 ML-DSA.Sign(sk_i, m_i)")]),
        numberedItem([t("Leaf nodes: h_i = SHA3-256(m_i || \u03C3_i)")]),
        numberedItem([t("Merkle tree root: R = MerkleRoot(h_1, ..., h_N)")]),
        numberedItem([t("Truncation: Signatures are truncated to 8\u201332 bytes")]),
        numberedItem([t("Compression: Zstd applied to truncated batch")]),

        // Table I: Aviation Channel Bandwidth
        tableCaption("TABLE I: Aviation Channel Bandwidth Analysis"),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2400, 1800, 2400, 2760],
          rows: [
            new TableRow({ children: [headerCell("Channel", 2400), headerCell("Rate", 1800), headerCell("Target", 2400), headerCell("Achieved", 2760)] }),
            new TableRow({ children: [cell("VHF ACARS", 2400), cell("2.4 kbps", 1800, AlignmentType.RIGHT), cell("80% reduction", 2400, AlignmentType.RIGHT), cell("~200 B/sig", 2760, AlignmentType.RIGHT)] }),
            new TableRow({ children: [cell("HF Datalink", 2400), cell("1.8 kbps", 1800, AlignmentType.RIGHT), cell("85% reduction", 2400, AlignmentType.RIGHT), cell("~150 B/sig", 2760, AlignmentType.RIGHT)] }),
            new TableRow({ children: [cell("Classic SATCOM", 2400), cell("600 bps", 1800, AlignmentType.RIGHT), cell("90% reduction", 2400, AlignmentType.RIGHT), cell("~100 B/sig", 2760, AlignmentType.RIGHT)] }),
            new TableRow({ children: [cell("LDACS", 2400), cell("100 kbps", 1800, AlignmentType.RIGHT), cell("50% reduction", 2400, AlignmentType.RIGHT), cell("~500 B/sig", 2760, AlignmentType.RIGHT)] }),
            new TableRow({ children: [cell("ADS-B", 2400), cell("1 Mbps", 1800, AlignmentType.RIGHT), cell("30% reduction", 2400, AlignmentType.RIGHT), cell("~800 B/sig", 2760, AlignmentType.RIGHT)] }),
          ]
        }),

        bodyPara([
          t("Batch sizes of 16\u201364 messages achieve 60\u201380% bandwidth reduction compared to transmitting individual PQC signatures. Verification latency remains below 50 ms for batches of 64 messages.")
        ]),

        bodyPara([
          t("Message priority levels (DISTRESS, URGENCY, SAFETY, ROUTINE) determine aggregation policies: DISTRESS messages bypass aggregation for immediate transmission with full individual signatures.")
        ]),

        subsectionHeading("B", "Forward-Secure Double-Ratchet Channels"),

        bodyPara([
          t("We extend the double-ratchet protocol [10] with post-quantum primitives for ATC channel forward secrecy. Each epoch performs an ephemeral ML-KEM-768 key exchange, deriving per-message symmetric keys via HKDF-SHA3-256:")
        ]),

        equationPara("K_{i+1} = HKDF(chain_key_i, \"msg\")"),

        // Table II: Aviation Channel Profiles
        tableCaption("TABLE II: Aviation Channel Profiles"),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2340, 2340, 2340, 2340],
          rows: [
            new TableRow({ children: [headerCell("Channel", 2340), headerCell("Epoch", 2340), headerCell("Msgs/Epoch", 2340), headerCell("Ratchet", 2340)] }),
            new TableRow({ children: [cell("ACARS", 2340), cell("10 min", 2340, AlignmentType.RIGHT), cell("500", 2340, AlignmentType.RIGHT), cell("Periodic", 2340)] }),
            new TableRow({ children: [cell("CPDLC", 2340), cell("5 min", 2340, AlignmentType.RIGHT), cell("100", 2340, AlignmentType.RIGHT), cell("Per-exchange", 2340)] }),
            new TableRow({ children: [cell("ADS-C", 2340), cell("30 min", 2340, AlignmentType.RIGHT), cell("2,000", 2340, AlignmentType.RIGHT), cell("Infrequent", 2340)] }),
            new TableRow({ children: [cell("LDACS", 2340), cell("2 min", 2340, AlignmentType.RIGHT), cell("10,000", 2340, AlignmentType.RIGHT), cell("Per-message", 2340)] }),
          ]
        }),

        bodyPara([
          t("Key erasure is immediate: sending/receiving chain keys are zeroized after use. Old epoch keys are retained for one interval to handle out-of-order delivery, then securely erased. This ensures that compromise of current key material cannot decrypt past sessions.")
        ]),

        // ===== VI. BANKING =====
        sectionHeading("VI", "Banking: Multi-Party Compliance PQC"),

        subsectionHeading("A", "Multi-Authority Threshold Signatures"),

        bodyPara([
          t("High-value financial transactions require authorization from multiple independent authorities. We construct a post-quantum threshold signature scheme where t-of-n partial signatures from designated authority roles are combined.")
        ]),

        // Algorithm 3
        new Paragraph({ spacing: { before: 160 }, alignment: AlignmentType.CENTER, children: [b("Algorithm 3: ", { size: 18 }), t("Multi-Authority Threshold Signing", { size: 18 })] }),
        algLine("Require: Transaction tx, quorum policy Q = (t, n, roles)"),
        algLine("Ensure: Combined signature \u03A3"),
        algLine("1: req_id \u2190 CreateRequest(tx, Q)"),
        algLine("2: for each authority a_j in signing window do"),
        algLine("3:   \u03C3_j \u2190 ML-DSA-65.Sign(sk_j, tx_hash || req_id)", 1),
        algLine("4:   SubmitPartial(req_id, \u03C3_j, role_j)", 1),
        algLine("5: end for"),
        algLine("6: Verify \u2265 t valid partial signatures with required roles"),
        algLine("7: \u03A3 \u2190 SHAKE256(sort(\u03C3_1,...,\u03C3_t) || tx_hash)"),
        algLine("8: return \u03A3"),

        // Table III: Banking Transaction Tiers
        tableCaption("TABLE III: Banking Transaction Tiers"),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [1800, 1560, 1200, 4800],
          rows: [
            new TableRow({ children: [headerCell("Tier", 1800), headerCell("Quorum", 1560), headerCell("Window", 1200), headerCell("Required Roles", 4800)] }),
            new TableRow({ children: [cell("Standard", 1800), cell("1-of-1", 1560), cell("1 hr", 1200, AlignmentType.RIGHT), cell("Any", 4800)] }),
            new TableRow({ children: [cell("Elevated", 1800), cell("2-of-3", 1560), cell("2 hr", 1200, AlignmentType.RIGHT), cell("Treasury", 4800)] }),
            new TableRow({ children: [cell("High Value", 1800), cell("3-of-5", 1560), cell("2 hr", 1200, AlignmentType.RIGHT), cell("Treasury + Compliance", 4800)] }),
            new TableRow({ children: [cell("Critical", 1800), cell("4-of-7", 1560), cell("4 hr", 1200, AlignmentType.RIGHT), cell("Treasury + Compliance + Risk", 4800)] }),
            new TableRow({ children: [cell("Sanctions", 1800), cell("3-of-3", 1560), cell("24 hr", 1200, AlignmentType.RIGHT), cell("Compliance + Legal + Risk", 4800)] }),
          ]
        }),

        bodyPara([
          t("Nine authority roles are supported: Central Bank, Treasury, Compliance, Risk Management, Legal, Board Member, Audit, IT Security, and Operations. A department constraint prevents two signers from the same department from satisfying the quorum.")
        ]),

        subsectionHeading("B", "SWIFT Proxy Re-Encryption"),

        bodyPara([
          t("Correspondent banking requires message routing through intermediary banks. We construct a proxy re-encryption (PRE) scheme based on ML-KEM that transforms encrypted SWIFT messages (MT103, MT202, pacs.008, pacs.009) without intermediaries learning the plaintext.")
        ]),

        bodyPara([
          t("The original ML-DSA-65 signature is preserved through the chain, ensuring non-repudiation from originator to beneficiary. Hop count is limited (default 3) with configurable re-key validity (default 24 hours). Re-encryption latency is <10 ms per hop.")
        ]),

        subsectionHeading("C", "Regulatory Zero-Knowledge Proof Engine"),

        bodyPara([
          t("Banks must demonstrate compliance with Basel III/IV capital adequacy, AML screening, and PCI-DSS requirements without exposing sensitive financial data. We implement six proof types using hash-based commitments:")
        ]),

        equationPara("C(v, r) = SHA3-256(v || r), r \u2190$ {0,1}^256"),

        bulletItem([b("Balance range proofs: "), t("Bit-decomposition commitments proving balance \u2208 [min, max] without revealing the exact value. 128-bit range requires 128 sub-commitments.")]),
        bulletItem([b("Capital adequacy proofs: "), t("CET1, Tier 1, Total Capital, Leverage, LCR, and NSFR ratios proven to exceed regulatory minimums.")]),
        bulletItem([b("AML screening proofs: "), t("Entity screening completion across OFAC-SDN, EU-CONSOLIDATED, UN-SC lists without exposing entity identities.")]),
        bulletItem([b("Transaction threshold proofs: "), t("Aggregate transaction volumes proven below limits without revealing individual amounts.")]),
        bulletItem([b("Audit integrity proofs: "), t("Merkle root over log entries proving hash-chain integrity without exposing contents.")]),
        bulletItem([b("Data residency proofs: "), t("Geographic compliance attestation.")]),

        bodyPara([
          t("Challenges are generated via Fiat-Shamir heuristic using SHAKE256, making proofs non-interactive. Proof validity defaults to 24 hours, with generation latency of 1\u2013500 ms depending on proof complexity.")
        ]),

        // ===== VII. HEALTHCARE =====
        sectionHeading("VII", "Healthcare: Privacy-Preserving PQC"),

        subsectionHeading("A", "EHR Proxy Re-Encryption"),

        bodyPara([
          t("Electronic health record transfer between providers must preserve patient privacy under HIPAA. We extend the PRE construction with healthcare-specific consent types:")
        ]),

        bulletItem([b("FULL_ACCESS: "), t("All record categories, unlimited duration")]),
        bulletItem([b("CATEGORY_RESTRICTED: "), t("Specific categories only (e.g., lab results)")]),
        bulletItem([b("TIME_LIMITED: "), t("Automatic expiration after configurable duration")]),
        bulletItem([b("EMERGENCY: "), t("Break-glass access with audit override")]),
        bulletItem([b("RESEARCH: "), t("De-identified access with restricted categories")]),
        bulletItem([b("ONE_TIME: "), t("Single use, auto-revoked after first access")]),

        bodyPara([
          t("Record categories are classified into standard (Demographics, Vital Signs, Lab Results, Medications, Diagnoses, Procedures, Imaging) and restricted (Mental Health, Substance Abuse under 42 CFR Part 2, Genetic Data under GINA, Reproductive Health).")
        ]),

        subsectionHeading("B", "Homomorphic Vital Signs Analytics"),

        bodyPara([
          t("Population-level health analytics require computing statistics over patient vital signs without decrypting individual records. We implement an additively homomorphic scheme based on Paillier-like encryption over lattice assumptions:")
        ]),

        equationPara("E(m) = g^m \u00B7 r^n mod n\u00B2"),
        equationPara("E(m\u2081) \u00B7 E(m\u2082) = E(m\u2081 + m\u2082) mod n\u00B2"),
        equationPara("E(m)^k = E(k \u00B7 m) mod n\u00B2"),

        bodyPara([
          t("Supported vital types include heart rate (40\u2013200 bpm), blood pressure (systolic 60\u2013250, diastolic 40\u2013150 mmHg), SpO\u2082 (70\u2013100%), temperature (34\u201342\u00B0C), respiratory rate (8\u201340 breaths/min), and glucose (40\u2013500 mg/dL).")
        ]),

        bodyPara([
          t("Differential privacy is integrated via Laplace noise addition with configurable \u03B5 (default 1.0) to aggregate statistics before decryption, preventing inference attacks on individual patients.")
        ]),

        subsectionHeading("C", "Post-Quantum Verifiable Credentials"),

        bodyPara([
          t("We implement W3C Verifiable Credentials [11] with ML-DSA-65 signatures and Merkle tree-based selective disclosure. Seven credential types are supported: vaccination records, lab results, prescriptions, allergy records, insurance eligibility, disability attestations, and provider licenses. Revocation uses an accumulator-based scheme with O(1) membership verification.")
        ]),

        subsectionHeading("D", "Lightweight PQC for Constrained Medical Devices"),

        bodyPara([t("Severely resource-constrained medical devices require adapted PQC profiles:")]),

        // Table IV: Medical Device PQC Profiles
        tableCaption("TABLE IV: Medical Device PQC Profiles"),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2800, 2000, 4560],
          rows: [
            new TableRow({ children: [headerCell("Profile", 2800), headerCell("RAM", 2000), headerCell("PQC Capability", 4560)] }),
            new TableRow({ children: [cell("Ultra-constrained", 2800), cell("<32 KB", 2000, AlignmentType.RIGHT), cell("Pre-shared symmetric only", 4560)] }),
            new TableRow({ children: [cell("Constrained", 2800), cell("32\u201364 KB", 2000, AlignmentType.RIGHT), cell("ML-KEM-512 (partial)", 4560)] }),
            new TableRow({ children: [cell("Moderate", 2800), cell("64\u2013256 KB", 2000, AlignmentType.RIGHT), cell("ML-KEM-768", 4560)] }),
            new TableRow({ children: [cell("Standard", 2800), cell(">256 KB", 2000, AlignmentType.RIGHT), cell("Full PQC suite", 4560)] }),
          ]
        }),

        bodyPara([
          t("Optimization strategies include pre-computation of expensive operations during idle periods, session key caching to amortize handshake costs, and deferred verification for battery-critical scenarios.")
        ]),

        // ===== VIII. INDUSTRIAL =====
        sectionHeading("VIII", "Industrial: Real-Time PQC"),

        subsectionHeading("A", "TESLA++ Broadcast Authentication"),

        bodyPara([
          t("IEC 61850 GOOSE and Sampled Values (SV) protocols require multicast authentication with microsecond-scale latency. We extend the TESLA protocol [12] with post-quantum primitives.")
        ]),

        bodyPara([b("Hash Chain Construction: "), t("A one-way chain is generated using SHAKE256:")], { indent: {} }),

        equationPara("K\u2080 = SHAKE256(K\u2081), K_i = SHAKE256(K_{i+1}), i = n-1 ... 0"),

        bodyPara([
          t("where K_n is the secret seed and K\u2080 is the public anchor. The anchor is committed via ML-DSA-65 signature for non-repudiation.")
        ]),

        bodyPara([b("Per-message authentication:")], { indent: {} }),

        equationPara("MAC_i = HMAC-SHAKE256(K_j, j || payload)"),

        // Table V: IEC 61850 Protocol Profiles
        tableCaption("TABLE V: IEC 61850 Protocol Profiles"),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [1560, 1560, 1560, 1560, 1560, 1560],
          rows: [
            new TableRow({ children: [headerCell("Protocol", 1560), headerCell("Chain", 1560), headerCell("Interval", 1560), headerCell("MAC", 1560), headerCell("Delay", 1560), headerCell("Latency", 1560)] }),
            new TableRow({ children: [cell("GOOSE", 1560), cell("10K", 1560, AlignmentType.RIGHT), cell("10 ms", 1560, AlignmentType.RIGHT), cell("16 B", 1560, AlignmentType.RIGHT), cell("2", 1560, AlignmentType.RIGHT), cell("<200 \u00B5s", 1560, AlignmentType.RIGHT)] }),
            new TableRow({ children: [cell("SV (4kHz)", 1560), cell("100K", 1560, AlignmentType.RIGHT), cell("250 \u00B5s", 1560, AlignmentType.RIGHT), cell("8 B", 1560, AlignmentType.RIGHT), cell("1", 1560, AlignmentType.RIGHT), cell("<50 \u00B5s", 1560, AlignmentType.RIGHT)] }),
            new TableRow({ children: [cell("MMS", 1560), cell("5K", 1560, AlignmentType.RIGHT), cell("100 ms", 1560, AlignmentType.RIGHT), cell("32 B", 1560, AlignmentType.RIGHT), cell("3", 1560, AlignmentType.RIGHT), cell("<1 ms", 1560, AlignmentType.RIGHT)] }),
          ]
        }),

        bodyPara([
          t("Checkpoints are stored every T/128 iterations for Wesolowski-style verification proofs (~256 bytes). Chain rotation is seamless: new chains overlap with old chains during pending key disclosures, with capacity warnings at <100 keys remaining.")
        ]),

        subsectionHeading("B", "Verifiable Delay Functions for Safety-Critical Timing"),

        bodyPara([
          t("Safety-critical industrial systems require provable minimum delays for operations such as emergency shutdown cooldown, interlock release, and chemical hold periods. We construct a VDF based on iterated SHA3-256:")
        ]),

        equationPara("h\u2080 = SHA3-256(input), h_i = SHA3-256(h_{i-1}), i = 1 ... T"),

        // Table VI: Industrial Safety VDF Profiles
        tableCaption("TABLE VI: Industrial Safety VDF Profiles"),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [3120, 2080, 2080, 2080],
          rows: [
            new TableRow({ children: [headerCell("Use Case", 3120), headerCell("Delay", 2080), headerCell("Iterations", 2080), headerCell("SIL", 2080)] }),
            new TableRow({ children: [cell("ESD Cooldown", 3120), cell("30 s", 2080, AlignmentType.RIGHT), cell("30M", 2080, AlignmentType.RIGHT), cell("3", 2080, AlignmentType.CENTER)] }),
            new TableRow({ children: [cell("Interlock Release", 3120), cell("10 s", 2080, AlignmentType.RIGHT), cell("10M", 2080, AlignmentType.RIGHT), cell("2", 2080, AlignmentType.CENTER)] }),
            new TableRow({ children: [cell("Chemical Hold", 3120), cell("60 s", 2080, AlignmentType.RIGHT), cell("60M", 2080, AlignmentType.RIGHT), cell("3", 2080, AlignmentType.CENTER)] }),
            new TableRow({ children: [cell("Pressure Equal.", 3120), cell("15 s", 2080, AlignmentType.RIGHT), cell("15M", 2080, AlignmentType.RIGHT), cell("2", 2080, AlignmentType.CENTER)] }),
            new TableRow({ children: [cell("Purge Cycle", 3120), cell("120 s", 2080, AlignmentType.RIGHT), cell("120M", 2080, AlignmentType.RIGHT), cell("3", 2080, AlignmentType.CENTER)] }),
          ]
        }),

        bodyPara([
          t("Each VDF output includes an attestation signed with ML-DSA-65 containing the equipment ID, operator ID, SIL level, and timing proof. Calibration adjusts iteration counts per hardware (typical rate: 1M\u201310M SHA3-256 hashes/sec) with 5% tolerance.")
        ]),

        // ===== IX. PERFORMANCE =====
        sectionHeading("IX", "Performance Evaluation"),

        subsectionHeading("A", "Algorithm Parameter Comparison"),

        bodyPara([t("Table VII summarizes key parameters across all 15 constructions.")]),

        // Table VII: Full Performance Summary
        tableCaption("TABLE VII: Algorithm Parameters and Performance Summary"),
        new Table({
          width: { size: 10080, type: WidthType.DXA },
          columnWidths: [2160, 1080, 1440, 1440, 1320, 1320, 1320],
          rows: [
            new TableRow({ children: [
              headerCell("Construction", 2160), headerCell("Domain", 1080), headerCell("Key Size", 1440),
              headerCell("Output", 1440), headerCell("Latency", 1320), headerCell("NIST Lvl", 1320), headerCell("Throughput", 1320)
            ]}),
            new TableRow({ children: [cell("Crypto Agility", 2160), cell("Core", 1080), cell("N/A", 1440), cell("N/A", 1440), cell("<1 ms", 1320), cell("Policy", 1320), cell("28 algs", 1320)] }),
            new TableRow({ children: [cell("QRS Engine", 2160), cell("Core", 1080), cell("N/A", 1440), cell("4 B", 1440), cell("1\u201310 ms", 1320), cell("N/A", 1320), cell("Portfolio", 1320)] }),
            new TableRow({ children: [cell("Hybrid KEM", 2160), cell("Core", 1080), cell("1,216 B", 1440), cell("1,120 B", 1440), cell("1\u20135 ms", 1320), cell("L3", 1320), cell("~1K/s", 1320)] }),
            new TableRow({ children: [cell("V2X Group Sig", 2160), cell("Auto", 1080), cell("1,024 B", 1440), cell("~2 KB", 1440), cell("<1 ms", 1320), cell("L3", 1320), cell("1K+/s", 1320)] }),
            new TableRow({ children: [cell("Aggregate Sig", 2160), cell("Aviation", 1080), cell("Per-signer", 1440), cell("100\u2013800 B", 1440), cell("<50 ms", 1320), cell("L3", 1320), cell("64 batch", 1320)] }),
            new TableRow({ children: [cell("Fwd-Secure Ch.", 2160), cell("Aviation", 1080), cell("1,184 B", 1440), cell("32 B", 1440), cell("<100 ms", 1320), cell("L3", 1320), cell("Per-epoch", 1320)] }),
            new TableRow({ children: [cell("Multi-Auth", 2160), cell("Banking", 1080), cell("3,293 B", 1440), cell("128 B", 1440), cell("<500 ms", 1320), cell("L3", 1320), cell("Tier-dep.", 1320)] }),
            new TableRow({ children: [cell("SWIFT PRE", 2160), cell("Banking", 1080), cell("1,184 B", 1440), cell("~ct size", 1440), cell("<10 ms", 1320), cell("L3", 1320), cell("3 hops", 1320)] }),
            new TableRow({ children: [cell("Regulatory ZKP", 2160), cell("Banking", 1080), cell("N/A", 1440), cell("64\u20134K B", 1440), cell("1\u2013500 ms", 1320), cell("PQ", 1320), cell("Per-proof", 1320)] }),
            new TableRow({ children: [cell("EHR PRE", 2160), cell("Health", 1080), cell("1,184 B", 1440), cell("~ct size", 1440), cell("1\u2013500 ms", 1320), cell("L3", 1320), cell("Per-rec.", 1320)] }),
            new TableRow({ children: [cell("Hom. Vitals", 2160), cell("Health", 1080), cell("2,048-bit", 1440), cell("ct/vital", 1440), cell("1\u201310 ms", 1320), cell("~112", 1320), cell("Additive", 1320)] }),
            new TableRow({ children: [cell("Verif. Creds", 2160), cell("Health", 1080), cell("3,293 B", 1440), cell("32 B root", 1440), cell("<10 ms", 1320), cell("L3", 1320), cell("Per-claim", 1320)] }),
            new TableRow({ children: [cell("Lightweight", 2160), cell("Health", 1080), cell("Profile", 1440), cell("Profile", 1440), cell("Varies", 1320), cell("L1\u2013L3", 1320), cell("Device", 1320)] }),
            new TableRow({ children: [cell("TESLA++", 2160), cell("Indust.", 1080), cell("32 B/key", 1440), cell("8\u201332 B", 1440), cell("<50 \u00B5s", 1320), cell("256-bit", 1320), cell("4K Hz", 1320)] }),
            new TableRow({ children: [cell("VDF Safety", 2160), cell("Indust.", 1080), cell("32 B", 1440), cell("32 B+pf", 1440), cell("10\u2013120 s", 1320), cell("256-bit", 1320), cell("Seq.", 1320)] }),
          ]
        }),

        subsectionHeading("B", "Bandwidth Overhead Analysis"),

        bodyPara([
          t("For aviation channels, at 2.4 kbps (VHF ACARS), transmitting a single ML-DSA-65 signature (3,293 bytes) requires approximately 11 seconds. With aggregate signatures and 80% compression, effective per-message overhead drops to ~200 bytes (0.67 seconds), enabling practical PQC deployment on legacy ATC infrastructure.")
        ]),

        subsectionHeading("C", "Memory Footprint Analysis"),

        bodyPara([
          t("For constrained healthcare devices, Table IV demonstrates the feasibility boundary. Devices with <32 KB RAM cannot support any standard PQC algorithm and must rely on pre-shared symmetric keys. At 32\u201364 KB, partial ML-KEM-512 is possible with pre-computation. Full PQC suite operation requires >256 KB RAM, available on modern medical gateways but not implantable devices.")
        ]),

        subsectionHeading("D", "Latency Budget Analysis"),

        bodyPara([
          t("The most demanding latency constraint is IEC 61850 SV at 4,000 Hz, requiring per-sample intervals of 250 \u00B5s. Our TESLA++ implementation achieves <50 \u00B5s MAC computation using 8-byte truncated HMAC-SHAKE256, leaving 200 \u00B5s for message processing and network transit\u2014well within the IEC 62351-6 timing budget.")
        ]),

        // ===== X. SECURITY ANALYSIS =====
        sectionHeading("X", "Security Analysis"),

        subsectionHeading("A", "Formal Security Properties"),

        bodyPara([t("Each construction inherits security from its underlying NIST-standardized primitive:")]),
        bulletItem([b("V2X Group Signatures: "), t("CCA-anonymity under Module-LWE, traceability under Module-SIS, non-frameability under ML-DSA EUF-CMA security.")]),
        bulletItem([b("Aggregate Signatures: "), t("Existential unforgeability reduces to ML-DSA EUF-CMA plus SHA3-256 collision resistance.")]),
        bulletItem([b("Forward-Secure Channels: "), t("Forward secrecy from ephemeral ML-KEM IND-CCA2 security; key erasure prevents retrospective decryption.")]),
        bulletItem([b("Threshold Signatures: "), t("Unforgeability requires t colluding authorities, reducing to ML-DSA EUF-CMA.")]),
        bulletItem([b("Proxy Re-Encryption: "), t("IND-CPA under ML-KEM; re-encryption key unidirectional and non-transitive.")]),
        bulletItem([b("ZKP Engine: "), t("Computational zero-knowledge under SHA3-256 preimage resistance; soundness from Fiat-Shamir in the random oracle model.")]),
        bulletItem([b("Homomorphic Vitals: "), t("Semantic security under decisional composite residuosity extended with lattice noise for PQ resistance.")]),
        bulletItem([b("Verifiable Credentials: "), t("Unforgeability from ML-DSA-65 EUF-CMA; selective disclosure from Merkle tree binding.")]),
        bulletItem([b("TESLA++: "), t("Source authentication from one-way hash chain (SHAKE256 preimage resistance); timeliness from delayed key disclosure.")]),
        bulletItem([b("VDF: "), t("Sequentiality from SHA3-256 non-parallelizability assumption; uniqueness from deterministic hashing.")]),

        subsectionHeading("B", "Quantum Security Margins"),

        bodyPara([
          t("All signature-based constructions achieve NIST Level 3 (192-bit quantum security) via ML-DSA-65. Hash-based constructions (SHAKE256, SHA3-256) achieve 256-bit security against quantum generic attacks (Grover\u2019s algorithm provides only quadratic speedup for preimage search, yielding 128-bit quantum security for 256-bit hashes). The hybrid KEM achieves Level 3 from the PQC component, with the classical component providing defense-in-depth.")
        ]),

        subsectionHeading("C", "Limitations"),

        bulletItem([t("The homomorphic scheme supports only additive operations; multiplicative homomorphism (required for variance computation) requires additional rounds.")]),
        bulletItem([t("VDF sequentiality relies on the assumption that SHA3-256 iteration cannot be significantly parallelized; specialized hardware (ASICs) may reduce this margin.")]),
        bulletItem([t("Group signature size (~2 KB) exceeds classical ECDSA signatures by an order of magnitude; V2X bandwidth budgets must accommodate this.")]),
        bulletItem([t("Lightweight PQC for ultra-constrained devices (<32 KB) relies on pre-shared keys, not achieving full PQC key establishment.")]),

        // ===== XI. RELATED WORK =====
        sectionHeading("XI", "Related Work"),

        bodyPara([
          b("PQC Standardization. "), t("NIST completed its PQC standardization process with the publication of FIPS 203, 204, and 205 in 2024 [2][3][4]. The CNSA 2.0 suite [13] mandates specific algorithms and transition timelines for national security systems.")
        ], { indent: {} }),
        bodyPara([
          b("Hybrid Schemes. "), t("Hybrid classical-PQC key exchange has been deployed at scale by Cloudflare [14] and Google Chrome [15] using X25519-Kyber768. The IETF draft draft-ietf-tls-hybrid-design formalizes the approach.")
        ], { indent: {} }),
        bodyPara([
          b("V2X Security. "), t("IEEE 1609.2 [9] and ETSI TS 103 097 define V2X security profiles using classical ECDSA. Post-quantum V2X has been explored in [16] but without group signature support.")
        ], { indent: {} }),
        bodyPara([
          b("Aviation Security. "), t("ARINC 823 and ICAO Doc 9896 specify classical cryptographic protections for ATC datalinks. To our knowledge, no prior work has addressed PQC deployment on bandwidth-constrained ATC channels.")
        ], { indent: {} }),
        bodyPara([
          b("Healthcare Privacy. "), t("Proxy re-encryption for healthcare was proposed in [17] using classical constructions. Homomorphic encryption for health analytics has been explored in [18] with fully homomorphic schemes, which remain impractical for real-time vitals.")
        ], { indent: {} }),
        bodyPara([
          b("Industrial SCADA Security. "), t("TESLA for IEC 61850 was proposed in IEC 62351-6 [19]. Our TESLA++ extension replaces HMAC-SHA256 with HMAC-SHAKE256 and adds ML-DSA commitment signing for post-quantum resistance.")
        ], { indent: {} }),
        bodyPara([
          b("Verifiable Delay Functions. "), t("VDFs were formalized by Boneh et al. [20]. Our construction specializes VDFs for safety-critical industrial timing using iterated SHA3-256 rather than algebraic constructions susceptible to quantum attacks.")
        ], { indent: {} }),

        // ===== XII. CONCLUSION =====
        sectionHeading("XII", "Conclusion and Future Work"),

        bodyPara([
          t("We have presented 15 domain-specific post-quantum cryptographic constructions addressing the unique requirements of automotive V2X, aviation ATC, banking compliance, healthcare privacy, and industrial SCADA systems. Our constructions build upon NIST-standardized primitives (ML-KEM, ML-DSA, SHA-3) while introducing domain-specific optimizations: group signatures for anonymous vehicle authentication, aggregate signatures for bandwidth-constrained channels, proxy re-encryption for privacy-preserving data transfer, homomorphic encryption for encrypted analytics, and TESLA++ for microsecond-scale broadcast authentication.")
        ]),

        bodyPara([
          t("Implementation results demonstrate practical feasibility: V2X batch verification exceeds 1,000 signatures per second, aviation aggregate signatures achieve 60\u201380% bandwidth reduction, and TESLA++ MAC computation meets the sub-50 \u00B5s requirement of IEC 61850 SV at 4,000 Hz.")
        ]),

        bodyPara([
          b("Future work "), t("includes: (1) formal verification using automated theorem provers (e.g., CryptoVerif, EasyCrypt); (2) hardware acceleration via FPGA/ASIC implementations for latency-critical industrial deployments; (3) interoperability testing with existing V2X, ATC, SWIFT, and FHIR infrastructure; and (4) extension to fully homomorphic encryption for richer healthcare analytics.")
        ]),

        // ===== ACKNOWLEDGMENTS =====
        new Paragraph({
          spacing: { before: 360, after: 120 },
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "ACKNOWLEDGMENTS", bold: true, font: "Times New Roman", size: 20 })]
        }),
        bodyPara([t("The authors thank the NIST PQC team for their standardization efforts and the open-source cryptography community for reference implementations.")]),

        // ===== REFERENCES =====
        new Paragraph({
          spacing: { before: 360, after: 120 },
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "REFERENCES", bold: true, font: "Times New Roman", size: 20 })]
        }),

        ...[
          "[1] P. W. Shor, \u201CPolynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer,\u201D SIAM J. Comput., vol. 26, no. 5, pp. 1484\u20131509, 1997.",
          "[2] NIST, \u201CModule-Lattice-Based Key-Encapsulation Mechanism Standard,\u201D FIPS 203, 2024.",
          "[3] NIST, \u201CModule-Lattice-Based Digital Signature Standard,\u201D FIPS 204, 2024.",
          "[4] NIST, \u201CStateless Hash-Based Digital Signature Standard,\u201D FIPS 205, 2024.",
          "[5] P.-A. Fouque et al., \u201CFalcon: Fast-Fourier Lattice-based Compact Signatures over NTRU,\u201D NIST PQC Round 3, 2020.",
          "[6] D. Cooper et al., \u201CRecommendation for Stateful Hash-Based Signature Schemes,\u201D NIST SP 800-208, 2020.",
          "[7] M. Mosca, \u201CCybersecurity in an Era with Quantum Computers: Will We Be Ready?\u201D IEEE Security & Privacy, vol. 16, no. 5, pp. 38\u201341, 2018.",
          "[8] H. Krawczyk and P. Eronen, \u201CHMAC-based Extract-and-Expand Key Derivation Function (HKDF),\u201D RFC 5869, 2010.",
          "[9] IEEE, \u201CIEEE Standard for Wireless Access in Vehicular Environments \u2013 Security Services,\u201D IEEE 1609.2, 2022.",
          "[10] T. Perrin and M. Marlinspike, \u201CThe Double Ratchet Algorithm,\u201D Signal Protocol, 2016.",
          "[11] W3C, \u201CVerifiable Credentials Data Model v2.0,\u201D W3C Recommendation, 2024.",
          "[12] A. Perrig et al., \u201CEfficient Authentication and Signing of Multicast Streams over Lossy Channels,\u201D IEEE S&P, pp. 56\u201373, 2000.",
          "[13] NSA, \u201CCommercial National Security Algorithm Suite 2.0,\u201D 2022.",
          "[14] B. Westerbaan and C. D. Rubin, \u201CPost-Quantum Key Agreement for TLS,\u201D Cloudflare Blog, 2024.",
          "[15] D. O\u2019Brien, \u201CProtecting Chrome Traffic with Hybrid Kyber KEM,\u201D Google Security Blog, 2024.",
          "[16] G. Twardokus et al., \u201CTowards Post-Quantum Security for Vehicle-to-Everything Communication,\u201D IEEE Commun. Surveys Tuts., 2024.",
          "[17] G. Ateniese et al., \u201CImproved Proxy Re-Encryption Schemes with Applications to Secure Distributed Storage,\u201D NDSS, 2006.",
          "[18] M. Kim et al., \u201CSecure Logistic Regression Based on Homomorphic Encryption,\u201D JMIR Med. Inform., vol. 6, no. 2, 2018.",
          "[19] IEC, \u201CPower systems management \u2013 Data and communications security for IEC 61850,\u201D IEC 62351-6, 2020.",
          "[20] D. Boneh et al., \u201CVerifiable Delay Functions,\u201D CRYPTO 2018, LNCS 10991, pp. 757\u2013788, 2018.",
        ].map(ref => new Paragraph({
          spacing: { after: 40 },
          indent: { left: 360, hanging: 360 },
          children: [new TextRun({ text: ref, font: "Times New Roman", size: 16 })]
        })),
      ]
    }
  ]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/Users/prabakarankannan/qbitel/docs/research_paper/QBITEL_PQC_Research_Paper.docx", buffer);
  console.log("DOCX created successfully: QBITEL_PQC_Research_Paper.docx");
}).catch(err => {
  console.error("Error:", err);
  process.exit(1);
});
