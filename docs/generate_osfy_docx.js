const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  BorderStyle, LevelFormat, ExternalHyperlink, Header, Footer, PageNumber,
  PageBreak, ShadingType, ImageRun
} = require("docx");

const FONT = "Arial";
const MONO = "Courier New";

// Load figure images
const fig1 = fs.readFileSync("/Users/prabakarankannan/qbitel/docs/figures/figure1.png");
const fig2 = fs.readFileSync("/Users/prabakarankannan/qbitel/docs/figures/figure2.png");
const fig3 = fs.readFileSync("/Users/prabakarankannan/qbitel/docs/figures/figure3.png");

function codeLine(text) {
  return new Paragraph({
    spacing: { before: 0, after: 0, line: 276 },
    shading: { type: ShadingType.CLEAR, fill: "F5F5F5" },
    indent: { left: 360 },
    children: [new TextRun({ text, font: MONO, size: 18, color: "333333" })],
  });
}

function bodyPara(text) {
  return new Paragraph({
    spacing: { before: 120, after: 120, line: 312 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, font: FONT, size: 22 })],
  });
}

function mixedPara(boldText, normalText) {
  return new Paragraph({
    spacing: { before: 120, after: 120, line: 312 },
    alignment: AlignmentType.JUSTIFIED,
    children: [
      new TextRun({ text: boldText, font: FONT, size: 22, bold: true }),
      new TextRun({ text: normalText, font: FONT, size: 22 }),
    ],
  });
}

function heading(text, level) {
  return new Paragraph({
    heading: level,
    spacing: { before: 360, after: 200 },
    children: [new TextRun({ text, font: FONT, bold: true })],
  });
}

function bullet(text, ref) {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { before: 60, after: 60, line: 312 },
    children: [new TextRun({ text, font: FONT, size: 22 })],
  });
}

function boldBullet(boldText, normalText, ref) {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { before: 60, after: 60, line: 312 },
    children: [
      new TextRun({ text: boldText, font: FONT, size: 22, bold: true }),
      new TextRun({ text: normalText, font: FONT, size: 22 }),
    ],
  });
}

function separator() {
  return new Paragraph({
    spacing: { before: 200, after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "2E75B6", space: 1 } },
    children: [],
  });
}

function codeBlock(lines) {
  const paras = [new Paragraph({ spacing: { before: 160, after: 0 }, children: [] })];
  for (const line of lines) { paras.push(codeLine(line)); }
  paras.push(new Paragraph({ spacing: { before: 0, after: 160 }, children: [] }));
  return paras;
}

function figureImage(data, width, height, caption) {
  return [
    new Paragraph({
      spacing: { before: 240, after: 80 },
      alignment: AlignmentType.CENTER,
      children: [new ImageRun({
        type: "png",
        data,
        transformation: { width, height },
        altText: { title: caption, description: caption, name: caption },
      })],
    }),
    new Paragraph({
      spacing: { before: 0, after: 240 },
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: caption, font: FONT, size: 18, italics: true, color: "555555" })],
    }),
  ];
}

async function generate() {
  const doc = new Document({
    styles: {
      default: { document: { run: { font: FONT, size: 22 } } },
      paragraphStyles: [
        {
          id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 36, bold: true, font: FONT, color: "1B3A5C" },
          paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 },
        },
        {
          id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, font: FONT, color: "2E75B6" },
          paragraph: { spacing: { before: 280, after: 180 }, outlineLevel: 1 },
        },
        {
          id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, font: FONT, color: "404040" },
          paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 2 },
        },
      ],
    },
    numbering: {
      config: [
        {
          reference: "bullets",
          levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
        },
        {
          reference: "numbered",
          levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
        },
      ],
    },
    sections: [{
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.RIGHT,
            border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "2E75B6", space: 4 } },
            children: [new TextRun({ text: "QBITEL Bridge \u2014 Open Source For You", font: FONT, size: 16, italics: true, color: "888888" })],
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            border: { top: { style: BorderStyle.SINGLE, size: 4, color: "2E75B6", space: 4 } },
            children: [
              new TextRun({ text: "Page ", font: FONT, size: 16, color: "888888" }),
              new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: "888888" }),
            ],
          })],
        }),
      },
      children: [
        // ===== TITLE =====
        new Paragraph({
          spacing: { before: 600, after: 120 },
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "QBITEL Bridge: Quantum-Safe Network Security", font: FONT, size: 48, bold: true, color: "1B3A5C" })],
        }),
        new Paragraph({
          spacing: { before: 0, after: 360 },
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "AI-powered protocol discovery and post-quantum cryptography for legacy infrastructure", font: FONT, size: 24, italics: true, color: "555555" })],
        }),
        separator(),

        // ===== BYLINE =====
        new Paragraph({ spacing: { before: 200, after: 60 }, children: [
          new TextRun({ text: "Author: ", font: FONT, size: 22, bold: true, color: "1B3A5C" }),
          new TextRun({ text: "Prabakaran Kannan", font: FONT, size: 22, bold: true }),
        ]}),
        new Paragraph({ spacing: { before: 0, after: 60 }, children: [
          new TextRun({ text: "Designation: ", font: FONT, size: 20, bold: true, color: "555555" }),
          new TextRun({ text: "Vice President of Technology, Innoviti Technologies, Chennai", font: FONT, size: 20 }),
        ]}),
        new Paragraph({ spacing: { before: 0, after: 60 }, children: [
          new TextRun({ text: "Research: ", font: FONT, size: 20, bold: true, color: "555555" }),
          new TextRun({ text: "Research Scholar in Quantum Machine Learning, NIT Puducherry", font: FONT, size: 20 }),
        ]}),
        new Paragraph({ spacing: { before: 0, after: 60 }, children: [
          new TextRun({ text: "Interests: ", font: FONT, size: 20, bold: true, color: "555555" }),
          new TextRun({ text: "Post-quantum cryptography, agentic AI, multi-agent orchestration, embedded systems security", font: FONT, size: 20 }),
        ]}),
        new Paragraph({ spacing: { before: 0, after: 60 }, children: [
          new TextRun({ text: "Open Source Projects: ", font: FONT, size: 20, bold: true, color: "555555" }),
          new TextRun({ text: "QBITEL Bridge (AI-powered post-quantum security), QBITEL EdgeOS (quantum-safe embedded OS)", font: FONT, size: 20 }),
        ]}),
        new Paragraph({ spacing: { before: 0, after: 60 }, children: [
          new TextRun({ text: "Email: ", font: FONT, size: 20, bold: true, color: "555555" }),
          new ExternalHyperlink({ children: [new TextRun({ text: "kannanprabakaran84@gmail.com", font: FONT, size: 20, color: "2E75B6", underline: {} })], link: "mailto:kannanprabakaran84@gmail.com" }),
        ]}),
        new Paragraph({ spacing: { before: 0, after: 60 }, children: [
          new TextRun({ text: "LinkedIn: ", font: FONT, size: 20, bold: true, color: "555555" }),
          new ExternalHyperlink({ children: [new TextRun({ text: "linkedin.com/in/prabakaran-kannan-2b20a9214", font: FONT, size: 20, color: "2E75B6", underline: {} })], link: "https://www.linkedin.com/in/prabakaran-kannan-2b20a9214/" }),
        ]}),
        new Paragraph({ spacing: { before: 0, after: 200 }, children: [
          new TextRun({ text: "Postal Address: ", font: FONT, size: 20, bold: true, color: "555555" }),
          new TextRun({ text: "4153, Appaswamy Splendour, OMR, Semmancheri, Chennai - 600119", font: FONT, size: 20 }),
        ]}),

        // ===== SHORT BYLINE for print =====
        new Paragraph({ spacing: { before: 0, after: 200 }, children: [
          new TextRun({ text: "Short Byline: ", font: FONT, size: 20, bold: true, color: "555555" }),
          new TextRun({ text: "Prabakaran Kannan is VP of Technology at Innoviti Technologies, Chennai, and a research scholar in Quantum ML at NIT Puducherry. He builds open source quantum-safe security platforms.", font: FONT, size: 20, italics: true }),
        ]}),
        separator(),

        // ===== SECTION 1 =====
        heading("The dirty secret of enterprise networks", HeadingLevel.HEADING_1),
        bodyPara("Most enterprise networks have a dirty secret: nobody knows exactly what is running on them."),
        bodyPara("A hospital network built over 20 years has COBOL applications talking to modern REST services, HL7 medical devices communicating over undocumented serial-over-TCP wrappers, and PACS imaging systems using protocols nobody documented before the original vendor went bankrupt. A bank\u2019s mainframe infrastructure processes crores of rupees per second over NEFT/RTGS using IBM SNA protocol extensions that predate the internet."),
        bodyPara("And all of it is encrypted \u2014 if it is encrypted at all \u2014 using RSA or ECDH. These are algorithms that a sufficiently powerful quantum computer will break."),
        bodyPara("Two converging crises are forcing organisations to act: the unknown protocol problem (you cannot secure what you cannot see) and the harvest-now-decrypt-later threat (adversaries are archiving your encrypted traffic today, waiting for quantum hardware to mature)."),
        bodyPara("QBITEL Bridge is an enterprise-grade open source platform, released under the Apache 2.0 licence, built to solve both \u2014 simultaneously, and without requiring any changes to existing applications or infrastructure."),

        // ===== SECTION 2 =====
        heading("What QBITEL Bridge does", HeadingLevel.HEADING_1),
        bodyPara("At its core, QBITEL Bridge does four things:"),
        new Paragraph({ numbering: { reference: "numbered", level: 0 }, spacing: { before: 60, after: 60, line: 312 }, children: [
          new TextRun({ text: "Automatically discovers every protocol ", font: FONT, size: 22, bold: true }),
          new TextRun({ text: "on your network \u2014 including undocumented and legacy ones \u2014 using a hybrid AI ensemble of CNN, BiLSTM, and Transformer models.", font: FONT, size: 22 }),
        ]}),
        new Paragraph({ numbering: { reference: "numbered", level: 0 }, spacing: { before: 60, after: 60, line: 312 }, children: [
          new TextRun({ text: "Wraps all discovered protocols in NIST Level 5 post-quantum cryptography", font: FONT, size: 22, bold: true }),
          new TextRun({ text: ", transparently, at the network layer.", font: FONT, size: 22 }),
        ]}),
        new Paragraph({ numbering: { reference: "numbered", level: 0 }, spacing: { before: 60, after: 60, line: 312 }, children: [
          new TextRun({ text: "Deploys autonomous AI agents ", font: FONT, size: 22, bold: true }),
          new TextRun({ text: "that monitor, classify, and respond to security events in under one second without human intervention.", font: FONT, size: 22 }),
        ]}),
        new Paragraph({ numbering: { reference: "numbered", level: 0 }, spacing: { before: 60, after: 60, line: 312 }, children: [
          new TextRun({ text: "Translates legacy protocols into modern REST/gRPC APIs", font: FONT, size: 22, bold: true }),
          new TextRun({ text: ", auto-generating OpenAPI 3.0 specifications and SDKs in six languages.", font: FONT, size: 22 }),
        ]}),
        bodyPara("The system sits between your existing infrastructure as a transparent bump-in-the-wire. It requires zero changes to applications, zero agent retraining, and zero downtime during deployment."),

        // ===== SECTION 3 =====
        heading("How the AI discovers unknown protocols", HeadingLevel.HEADING_1),
        bodyPara("The most technically novel part of QBITEL Bridge is its protocol discovery subsystem. Traditional network security tools maintain signature databases. If a protocol is not in the database, it is invisible. QBITEL takes a fundamentally different approach: it learns protocol grammars from raw network traffic."),
        bodyPara("The discovery pipeline has five stages. Figure 1 illustrates how they connect."),

        ...figureImage(fig1, 580, 215, "Figure 1: Five-stage protocol discovery pipeline"),

        heading("Statistical traffic analysis", HeadingLevel.HEADING_2),
        bodyPara("The StatisticalAnalyzer examines raw packet captures and computes field-level entropy, byte distribution patterns, and structural boundaries. Fixed-value fields (like protocol magic bytes or version numbers) are distinguished from variable-content fields (like usernames or transaction amounts) using Shannon entropy calculations. This produces a structural skeleton of the unknown protocol."),

        heading("Grammar learning with PCFG inference", HeadingLevel.HEADING_2),
        bodyPara("The GrammarLearner uses a probabilistic context-free grammar (PCFG) inference engine refined with the expectation-maximisation (EM) algorithm. Starting from the structural skeleton, it learns production rules: which fields appear in which order, what the valid value ranges are, and how messages relate to each other across a session."),
        bodyPara("This is the same class of technique used in natural language processing to learn the grammar of a human language from a corpus of sentences \u2014 applied here to binary network protocols."),

        heading("Dynamic parser generation", HeadingLevel.HEADING_2),
        bodyPara("Once a grammar is learned, the ParserGenerator compiles it into a working parser at runtime. The generated parser can decode live traffic, validate message structure, and extract field values \u2014 all without any human writing parser code. The parser generation layer achieves 50,000+ messages per second throughput."),

        heading("The hybrid AI classifier", HeadingLevel.HEADING_2),
        bodyPara("The ProtocolClassifier uses a hybrid ensemble with the Transformer model as primary:"),
        bullet("Transformer: multi-head self-attention for capturing long-range dependencies across protocol sessions \u2014 the primary classification engine", "bullets"),
        bullet("CNN (convolutional neural network): detects local byte patterns \u2014 equivalent to finding words in the protocol vocabulary", "bullets"),
        bullet("BiLSTM (bidirectional long short-term memory): models sequence structure with CRF for field-level detection", "bullets"),
        bullet("Random forest: provides a robust statistical baseline for edge cases", "bullets"),
        bodyPara("The ensemble vote produces a confidence score. Protocols above a configurable threshold (default 0.7) are promoted from candidate to known status and added to the protection perimeter."),

        heading("Protocol compliance validation", HeadingLevel.HEADING_2),
        bodyPara("The MessageValidator enforces the learned grammar against live traffic, flagging anomalous messages that could indicate attacks, protocol fuzzing, or misconfigured clients."),
        bodyPara("Real-world performance on first pass: 89%+ discovery accuracy, completing in 2 to 4 hours on a typical enterprise network segment \u2014 reducing what traditionally takes 6 to 12 months of manual reverse-engineering."),

        // ===== SECTION 4 =====
        new Paragraph({ children: [new PageBreak()] }),
        heading("The post-quantum cryptography stack", HeadingLevel.HEADING_1),
        bodyPara("Once protocols are discovered, QBITEL wraps them in post-quantum cryptography. The platform implements the full NIST post-quantum standardisation suite at Level 5, published as FIPS 203, 204, and 205 in 2024."),

        ...figureImage(fig2, 580, 355, "Figure 2: PQC algorithm selection by domain and constraint"),

        bodyPara("The PQCEngine class provides a single, domain-aware interface to all algorithms. Rather than requiring developers to select the right algorithm for their context, the engine accepts a DomainProfile value:"),
        ...codeBlock([
          "from ai_engine.crypto.pqc_unified import PQCEngine, DomainProfile",
          "",
          "# Healthcare: constrained devices (64 KB RAM), uses ML-KEM-512",
          "engine = PQCEngine(DomainProfile.HEALTHCARE)",
          "",
          "# Automotive V2X: real-time (<1ms), uses Falcon for compact signatures",
          "engine = PQCEngine(DomainProfile.AUTOMOTIVE)",
          "",
          "# Banking enterprise: maximum security, uses ML-KEM-1024 + ML-DSA-87",
          "engine = PQCEngine(DomainProfile.ENTERPRISE)",
          "",
          "# Encrypt a payload",
          "ciphertext, encapsulated_key = await engine.encrypt(plaintext)",
          "",
          "# Decrypt",
          "recovered = await engine.decrypt(ciphertext, encapsulated_key)",
        ]),
        bodyPara("The engine supports hybrid classical/post-quantum key exchange \u2014 combining X25519 or P-384 ECDH with ML-KEM via hybrid schemes (X25519MLKEM768, P384MLKEM1024) \u2014 so existing TLS 1.3 stacks remain compatible while gaining quantum resistance."),
        bodyPara("NIST-standardised algorithms:"),
        bullet("FIPS 203: ML-KEM (Kyber) 512, 768, and 1024 \u2014 key encapsulation for session keys", "bullets"),
        bullet("FIPS 204: ML-DSA (Dilithium) at levels 2, 3, and 5 \u2014 digital signatures", "bullets"),
        bullet("FIPS 205: SLH-DSA (SPHINCS+) \u2014 stateless hash-based signatures", "bullets"),
        bodyPara("Additional algorithms:"),
        bullet("Falcon 512 and 1024 \u2014 compact signatures for bandwidth-constrained channels", "bullets"),
        bullet("LMS (NIST SP 800-208, RFC 8554) and XMSS (RFC 8391) \u2014 stateful hash-based signatures", "bullets"),
        bodyPara("Advanced primitives:"),
        bullet("Zero-knowledge proofs (ZKP) for privacy-preserving verification", "bullets"),
        bullet("Verifiable random functions (VRF) and threshold signatures (t-of-n distributed signing)", "bullets"),
        bullet("CNSA 2.0 support for defence and government deployments", "bullets"),
        bullet("Crypto agility negotiation for seamless algorithm migration", "bullets"),
        bodyPara("Encryption overhead on the hot path: under 1 millisecond."),

        // ===== SECTION 5 =====
        heading("Autonomous agents that respond in under a second", HeadingLevel.HEADING_1),
        bodyPara("QBITEL Bridge\u2019s security response layer is built around a multi-agent architecture with 16+ specialised agents. Agents are workers with typed capabilities including threat analysis, protocol analysis, anomaly detection, incident response, and compliance auditing."),
        bodyPara("Each agent inherits from BaseAgent, which provides:"),
        bullet("Lifecycle management (start, stop, pause, resume, health check)", "bullets"),
        bullet("Inter-agent communication via a typed message bus with cross-system routing", "bullets"),
        bullet("Persistent memory with configurable retention policies", "bullets"),
        bullet("Automatic Prometheus metrics instrumentation", "bullets"),
        bullet("Circuit breaker pattern for fault tolerance", "bullets"),
        bodyPara("The UnifiedAgentInterface bridges three agent subsystems \u2014 core agents, legacy whisperer agents, and BPO-specific agents \u2014 through a common protocol with a central registry."),
        bodyPara("The LLM integration layer connects agents to local or cloud language models \u2014 including Ollama for fully air-gapped deployments. This enables agents to generate human-readable incident reports, suggest remediation steps, and explain anomalies in plain language."),
        bodyPara("When a threat is detected, the system achieves sub-second security decision time with a 78% autonomous response rate \u2014 meaning most threats are contained without human intervention."),

        // ===== SECTION 6 =====
        heading("Translation Studio: from legacy to modern APIs", HeadingLevel.HEADING_1),
        bodyPara("One of QBITEL Bridge\u2019s most practically useful features is the Translation Studio. Once a legacy protocol is discovered and its grammar learned, the Translation Studio can:"),
        bullet("Auto-generate OpenAPI 3.0 specifications from protocol grammars", "bullets"),
        bullet("Produce production-ready SDKs in six languages: Python, TypeScript, Go, Rust, Java, and C#", "bullets"),
        bullet("Create REST/gRPC API wrappers around legacy protocol sessions", "bullets"),
        bodyPara("This means a COBOL mainframe system communicating over an undocumented binary protocol can be exposed as a modern REST API \u2014 without modifying the mainframe, without writing custom integration code, and with quantum-safe encryption on the wire."),

        // ===== SECTION 7 =====
        heading("Domain-specific modules", HeadingLevel.HEADING_1),
        bodyPara("QBITEL Bridge ships with purpose-built modules for six regulated industries."),
        mixedPara("Banking and finance: ", "ISO-8583 and SWIFT protocol handling, PCI-DSS DTMF masking for call centre voice channels, a regulatory proof engine for RBI/SEBI compliance automation, HSM integration, and multi-authority threshold signatures for cross-bank transaction authorisation."),
        mixedPara("Healthcare: ", "EHR proxy re-encryption compatible with FHIR/HL7/DICOM, homomorphic encryption for vital sign aggregation without exposing raw patient data, ABDM-compatible verifiable credentials, and FDA 21 CFR Part 11 compliance."),
        mixedPara("Automotive: ", "V2X (vehicle-to-everything) group signatures for multi-vehicle coordination, CAN protocol support, and misbehaviour detection for rogue vehicle identification in C-V2X networks."),
        mixedPara("Aviation: ", "ARINC 429/629-aware aggregate signatures, ADS-B and ACARS protocol handling, and forward-secure channel establishment for flight data links."),
        mixedPara("Industrial and critical infrastructure: ", "Modbus, DNP3, and IEC 61850 GOOSE message authentication for power grid substations, SCADA/PLC integration, and verifiable delay functions for timing-sensitive industrial control sequences."),
        mixedPara("BPO and call centres: ", "Real-time DTMF masking, SIP/RTP quantum-safe encryption, CTI and IVR protocol support, SS7 overlay protection, and TN3270e/TN5250 mainframe session tunnelling."),

        // ===== SECTION 8 =====
        new Paragraph({ children: [new PageBreak()] }),
        heading("Architecture and deployment", HeadingLevel.HEADING_1),
        bodyPara("The system has four primary runtime components. Figure 3 shows how they interact."),

        ...figureImage(fig3, 580, 350, "Figure 3: Four-layer system architecture"),

        mixedPara("AI engine (Python 3.11+): ", "The ML pipeline, agent framework, LLM integration, Translation Studio, crypto layer, and REST API. Uses FastAPI, PyTorch, LangGraph for agent orchestration, ChromaDB for vector storage, and Redis for distributed caching."),
        mixedPara("Rust dataplane (Rust 1.75+): ", "High-performance packet processing, PQC-TLS termination, protocol proxying, and eBPF-based container monitoring. Built with Tokio async runtime. Protocol-specific adapters for ISO-8583, TN3270e, HL7 MLLP, and Modbus. Supports DPDK for hardware-accelerated packet processing."),
        mixedPara("Go controlplane (Go 1.22+): ", "Management API, Kubernetes operator, Istio/Envoy xDS integration for service mesh deployments, admission webhooks, and device agent for edge deployments."),
        mixedPara("Web console (TypeScript/React): ", "Dashboard UI for monitoring, configuration, and compliance reporting."),
        mixedPara("Infrastructure: ", "Kubernetes-native with Helm charts, Apache Kafka for event streaming (100,000+ messages per second), Prometheus and Grafana for observability, OpenTelemetry and Jaeger for distributed tracing."),

        heading("Getting started", HeadingLevel.HEADING_2),
        ...codeBlock([
          "# Clone the repository",
          "git clone https://github.com/AXEwaves/qbitel-bridge.git",
          "cd qbitel-bridge",
          "",
          "# Install Python AI engine dependencies",
          "pip install -r ai_engine/requirements.txt",
          "",
          "# Build the Rust dataplane",
          "cd rust/dataplane && cargo build --release",
          "",
          "# Start protocol discovery on a network interface",
          "python -m ai_engine.discovery.cli --interface eth0 --output discovered.json",
          "",
          "# Run with Docker Compose (includes Redis, Prometheus, Grafana)",
          "docker compose up -d",
          "",
          "# Or deploy to Kubernetes with Helm",
          "helm install qbitel-bridge ./deploy/helm/qbitel-bridge",
        ]),
        bodyPara("Once discovery completes, results are available via the REST API:"),
        ...codeBlock([
          "import httpx",
          "",
          "async with httpx.AsyncClient() as client:",
          "    result = await client.post(",
          '        "http://localhost:8080/api/v1/discovery/analyze",',
          "        json={",
          '            "traffic_data": ["<base64_packet_1>", "<base64_packet_2>"],',
          '            "options": {',
          '                "confidence_threshold": 0.7,',
          '                "enable_adaptive_learning": True',
          "            }",
          "        }",
          "    )",
          '    protocols = result.json()["discovered_protocols"]',
          "    for p in protocols:",
          "        print(f\"{p['name']}: {p['confidence']:.0%} confidence\")",
        ]),

        // ===== SECTION 9 =====
        heading("Nine compliance frameworks, built in", HeadingLevel.HEADING_1),
        bodyPara("QBITEL Bridge includes automated compliance engines for nine regulatory frameworks: SOC 2, GDPR, HIPAA, PCI-DSS 4.0, ISO 27001, NIST 800-53, BASEL-III, NERC-CIP, and FDA 21 CFR Part 11. Audit-ready reports can be generated in under 10 minutes. The compliance agents continuously monitor for policy violations and produce evidence trails suitable for external audits."),

        // ===== SECTION 10 =====
        heading("Observability from day one", HeadingLevel.HEADING_1),
        bodyPara("QBITEL Bridge exports Prometheus metrics and OpenTelemetry traces out of the box. Every agent task, protocol discovery request, crypto operation, and cache hit is measured:"),
        ...codeBlock([
          'protocol_discovery_requests_total{status="success"} 1547',
          'protocol_discovery_duration_seconds_bucket{le="0.15"} 892',
          'qbitel_agent_tasks_total{agent_type="incident_response"} 423',
          'pqc_operations_total{algorithm="ml-kem-1024", operation="encapsulate"} 98412',
        ]),
        bodyPara("Grafana dashboard configurations are included in ops/grafana-dashboards/. Jaeger provides distributed tracing across the Python, Rust, and Go components."),

        // ===== SECTION 11 =====
        heading("Performance at a glance", HeadingLevel.HEADING_1),
        bullet("Protocol discovery accuracy: 89%+ on first pass", "bullets"),
        bullet("Discovery time: 2 to 4 hours on a typical enterprise segment", "bullets"),
        bullet("Parser generation throughput: 50,000+ messages per second", "bullets"),
        bullet("Kafka event streaming: 100,000+ messages per second", "bullets"),
        bullet("PQC encryption overhead: under 1 millisecond", "bullets"),
        bullet("Security decision time: under 1 second", "bullets"),
        bullet("Autonomous threat response rate: 78%", "bullets"),
        bullet("xDS proxy support: 1,000+ concurrent proxies", "bullets"),
        bullet("eBPF container monitoring: 10,000+ containers", "bullets"),
        bullet("API gateway P99 latency: under 25 milliseconds", "bullets"),

        // ===== SECTION 12 =====
        heading("What it cannot do yet", HeadingLevel.HEADING_1),
        bodyPara("No tool solves everything, and transparency about boundaries is important. The current grammar inference engine performs best on binary protocols with fixed-length headers; text-based protocols with complex state machines (such as HTTP/2 or gRPC) show lower first-pass accuracy and may require additional training passes. Air-gapped LLM integration via Ollama is functional but slower than cloud-hosted models, and the quality of incident reports depends on the local model\u2019s capabilities. While the PQC algorithms are NIST-standardised and the platform uses liboqs and oqs-rs, the implementations have not yet undergone formal third-party cryptographic audit \u2014 contributions from the security research community on this front are especially welcome."),

        // ===== SECTION 13 =====
        heading("Why open source matters here", HeadingLevel.HEADING_1),
        bodyPara("Security tooling that organisations cannot inspect is security tooling they cannot trust. QBITEL Bridge is released under the Apache 2.0 licence \u2014 you can read every line of the cryptography implementation, audit the ML models, and verify the claims independently. There is no open-core model and no vendor lock-in."),
        bodyPara("For India\u2019s public sector in particular, this matters enormously. DRDO, ISRO, and defence PSUs cannot deploy cloud-dependent foreign commercial tools in classified environments. An open source, air-gappable platform built on NIST-standardised algorithms with CNSA 2.0 support provides a path to quantum readiness that proprietary tools simply cannot offer."),
        bodyPara("The community is invited to contribute across several active areas: new domain protocol modules, additional LLM provider integrations, performance benchmarks on different hardware, compliance templates for new regulatory jurisdictions, and Translation Studio language targets."),

        // ===== SECTION 14 =====
        heading("Getting involved", HeadingLevel.HEADING_1),
        bullet("Website: bridge.qbitel.com", "bullets"),
        bullet("Repository: github.com/AXEwaves/qbitel-bridge", "bullets"),
        bullet("Issue tracker: GitHub issues on the same repository", "bullets"),
        bullet("Security disclosures: security@qbitel.com", "bullets"),
        bullet("General discussion: support@qbitel.com", "bullets"),

        // ===== SECTION 15 =====
        heading("The quantum-safe road ahead", HeadingLevel.HEADING_1),
        bodyPara("The convergence of quantum computing and decades of legacy infrastructure debt is not a future problem. Adversaries are already harvesting encrypted traffic. Networks are already running protocols nobody documented. The gap between what enterprise security tools protect and what actually runs on enterprise networks has never been wider."),
        bodyPara("QBITEL Bridge addresses this gap with four open, inspectable technologies working in concert: hybrid AI ensemble for protocol discovery, NIST Level 5 post-quantum cryptography for forward-secure encryption, autonomous AI agents for sub-second incident response, and Translation Studio for legacy-to-modern API conversion. The platform deploys as a transparent network layer, requires no application changes, supports nine compliance frameworks, and is fully observable through Prometheus, Grafana, and OpenTelemetry."),
        bodyPara("For any organisation running legacy mainframes, industrial control systems, or regulated workloads \u2014 this is the open source foundation for a quantum-safe future."),
      ],
    }],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync("/Users/prabakarankannan/qbitel/docs/osfy_article_qbitel_bridge.docx", buffer);
  console.log("DOCX generated successfully with embedded figures!");
}

generate().catch(console.error);
