const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Header, Footer,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  PageNumber, PageBreak
} = require("docx");

// ── helpers ──────────────────────────────────────────────────────────
function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: [new TextRun({ text, font: "Arial", size: 28, bold: true })],
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 160 },
    children: [new TextRun({ text, font: "Arial", size: 24, bold: true })],
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, font: "Arial", size: 22, bold: true })],
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 200 },
    children: [new TextRun({ text, font: "Arial", size: 22, ...opts })],
  });
}

function italicPara(label, text) {
  return new Paragraph({
    spacing: { after: 100 },
    children: [
      new TextRun({ text: label, font: "Arial", size: 22, italics: true }),
      new TextRun({ text, font: "Arial", size: 22 }),
    ],
  });
}

function codeLine(text) {
  return new Paragraph({
    spacing: { after: 0 },
    children: [new TextRun({ text, font: "Courier New", size: 18 })],
  });
}

function codeMarker() {
  return new Paragraph({
    spacing: { after: 0 },
    children: [new TextRun({ text: "--------------------------------CODE-----------------------------", font: "Courier New", size: 18 })],
  });
}

function figureCaption(text) {
  return new Paragraph({
    spacing: { before: 200, after: 200 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, font: "Arial", size: 20, italics: true })],
  });
}

function separator() {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "CCCCCC", space: 1 } },
    children: [],
  });
}

// ── bullet helper ────────────────────────────────────────────────────
function bullet(text, ref = "bullets") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, font: "Arial", size: 22 })],
  });
}

function numberedItem(text, ref = "numbers") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, font: "Arial", size: 22 })],
  });
}

// ── build document ───────────────────────────────────────────────────
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 },
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 2 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      },
      {
        reference: "numbers",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      },
      {
        reference: "bullets2",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      },
      {
        reference: "bullets3",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      },
      {
        reference: "bullets4",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      },
      {
        reference: "bullets5",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
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
          children: [new TextRun({ text: "QBITEL Bridge \u2014 OSFY Article", font: "Arial", size: 18, italics: true, color: "888888" })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Page ", font: "Arial", size: 18 }),
            new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18 }),
          ],
        })],
      }),
    },
    children: [
      // ── metadata ─────────────────────────────────────────────────
      italicPara("Subject matter: ", "A technical introduction to QBITEL Bridge \u2014 an open source platform that uses machine learning for automatic protocol discovery and post-quantum cryptography for quantum-safe network security."),
      italicPara("Target audience: ", "Intermediate to advanced Linux users, network engineers, security professionals, and open source software developers."),
      italicPara("Author byline: ", "Prabakaran Kannan, VP of Technology, Innoviti Technologies | Research Scholar (Quantum ML), NIT Puducherry | kannanprabakaran84@gmail.com"),

      separator(),

      // ── title ────────────────────────────────────────────────────
      new Paragraph({
        spacing: { before: 400, after: 400 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "QBITEL Bridge: AI-powered protocol discovery and quantum-safe security for legacy infrastructure", font: "Arial", size: 32, bold: true })],
      }),

      separator(),

      // ── The problem nobody wants to talk about ───────────────────
      heading1("The problem nobody wants to talk about"),

      para("Most enterprise networks have a dirty secret: nobody knows exactly what is running on them."),

      para("A hospital network built over 20 years has COBOL applications talking to modern REST services, HL7 medical devices communicating over undocumented serial-over-TCP wrappers, and PACS imaging systems using protocols nobody documented before the original vendor went bankrupt. A bank\u2019s mainframe infrastructure processes crores of rupees (tens of millions of USD equivalent) per second over NEFT/RTGS using IBM SNA protocol extensions that predate the internet."),

      para("And all of it is encrypted \u2014 if it is encrypted at all \u2014 using RSA or ECDH. These are algorithms that a sufficiently powerful quantum computer will break."),

      para("Two converging crises are forcing organisations to act: the unknown protocol problem (you cannot secure what you cannot see) and the harvest-now-decrypt-later threat (adversaries are archiving your encrypted traffic today, waiting for quantum hardware to mature)."),

      para("QBITEL Bridge is an open source platform, released under the MIT licence, built to solve both \u2014 simultaneously, and without requiring any changes to existing applications or infrastructure."),

      separator(),

      // ── What QBITEL Bridge does ──────────────────────────────────
      heading1("What QBITEL Bridge does"),

      para("At its core, QBITEL Bridge does three things:"),

      numberedItem("Automatically discovers every protocol on your network \u2014 including undocumented and legacy ones \u2014 using machine learning. No configuration, no manual packet inspection."),
      numberedItem("Wraps all discovered protocols in NIST-standardised post-quantum cryptography, transparently, at the network layer."),
      numberedItem("Deploys autonomous AI agents that monitor, classify, and respond to security events in under 10 seconds without human intervention."),

      para("The system sits between your existing infrastructure components as a transparent bump-in-the-wire. It requires zero changes to applications, zero agent retraining, and zero downtime during deployment."),

      separator(),

      // ── The AI protocol discovery engine ─────────────────────────
      heading1("The AI protocol discovery engine"),

      para("The most technically novel part of QBITEL Bridge is its protocol discovery subsystem. Traditional network security tools maintain signature databases. If a protocol is not in the database, it is invisible. QBITEL takes a fundamentally different approach: it learns protocol grammars from raw network traffic."),

      para("The discovery pipeline has five stages. Figure 1 illustrates how they connect."),

      figureCaption("Figure 1: QBITEL Bridge protocol discovery pipeline \u2014 from raw traffic to validated parsers"),

      heading2("Statistical traffic analysis"),
      para("The StatisticalAnalyzer examines raw packet captures and computes field-level entropy, byte distribution patterns, and structural boundaries. Fixed-value fields (like protocol magic bytes or version numbers) are distinguished from variable-content fields (like usernames or transaction amounts) using Shannon entropy calculations. This produces a structural skeleton of the unknown protocol."),

      heading2("Grammar learning with PCFG inference"),
      para("The GrammarLearner uses a probabilistic context-free grammar (PCFG) inference engine refined with the expectation-maximisation (EM) algorithm. Starting from the structural skeleton, it learns production rules: which fields appear in which order, what the valid value ranges are, and how messages relate to each other across a session."),
      para("This is the same class of technique used in natural language processing to learn the grammar of a human language from a corpus of sentences \u2014 applied here to binary network protocols."),

      heading2("Dynamic parser generation"),
      para("Once a grammar is learned, the ParserGenerator compiles it into a working parser at runtime. The generated parser can decode live traffic, validate message structure, and extract field values \u2014 all without any human writing parser code."),

      heading2("Ensemble ML classification"),
      para("The ProtocolClassifier uses an ensemble of three models working in parallel:"),

      bullet("CNN (convolutional neural network): detects local byte patterns \u2014 equivalent to finding words in the protocol vocabulary"),
      bullet("LSTM (long short-term memory): models the sequence structure of multi-message sessions \u2014 equivalent to understanding sentences"),
      bullet("Random forest: provides a robust statistical baseline and handles edge cases where neural approaches over-fit"),

      para("The ensemble vote produces a confidence score. Protocols above a configurable threshold (default 0.7) are promoted from candidate to known status and added to the protection perimeter."),

      heading2("Protocol compliance validation"),
      para("The MessageValidator enforces the learned grammar against live traffic, flagging anomalous messages that could indicate attacks, protocol fuzzing, or misconfigured clients."),
      para("Real-world performance on first pass: 89%+ discovery accuracy, completing in 2 to 4 hours on a typical enterprise network segment."),

      separator(),

      // ── Post-quantum cryptography ────────────────────────────────
      heading1("Post-quantum cryptography: the technical stack"),

      para("Once protocols are discovered, QBITEL wraps them in post-quantum cryptography. The platform implements the full NIST post-quantum standardisation suite, published as FIPS 203, 204, and 205 in 2024."),

      figureCaption("Figure 2: PQC algorithm selection by deployment domain"),

      para("The PQCEngine class provides a single, domain-aware interface to all algorithms. Rather than requiring developers to select the right algorithm for their context, the engine accepts a DomainProfile value:"),

      codeMarker(),
      codeLine("from ai_engine.crypto.pqc_unified import PQCEngine, DomainProfile"),
      codeLine(""),
      codeLine("# Healthcare: constrained devices (64 KB RAM), uses ML-KEM-512"),
      codeLine("engine = PQCEngine(DomainProfile.HEALTHCARE)"),
      codeLine(""),
      codeLine("# Automotive V2X: real-time (<1ms), uses Falcon for compact signatures"),
      codeLine("engine = PQCEngine(DomainProfile.AUTOMOTIVE)"),
      codeLine(""),
      codeLine("# Banking enterprise: maximum security, uses ML-KEM-1024 + ML-DSA-87"),
      codeLine("engine = PQCEngine(DomainProfile.ENTERPRISE)"),
      codeLine(""),
      codeLine("# Encrypt a payload"),
      codeLine("ciphertext, encapsulated_key = await engine.encrypt(plaintext)"),
      codeLine(""),
      codeLine("# Decrypt"),
      codeLine("recovered = await engine.decrypt(ciphertext, encapsulated_key)"),
      codeMarker(),

      para("The engine also supports hybrid classical/post-quantum key exchange \u2014 combining X25519 or P-384 ECDH with ML-KEM \u2014 so existing TLS stacks remain compatible while gaining quantum resistance. This is the recommended NIST transition strategy."),

      para("The supported algorithms are:"),

      bullet("FIPS 203: ML-KEM (Kyber) 512, 768, and 1024 \u2014 key encapsulation for session keys", "bullets2"),
      bullet("FIPS 204: ML-DSA (Dilithium) 44, 65, and 87 \u2014 digital signatures", "bullets2"),
      bullet("Falcon 512 and 1024 \u2014 compact signatures for bandwidth-constrained channels (not part of the FIPS standard; selected as an alternate NIST candidate)", "bullets2"),
      bullet("FIPS 205: SLH-DSA (SPHINCS+) \u2014 stateless hash-based signatures for high-assurance contexts", "bullets2"),

      separator(),

      // ── The autonomous agent framework ───────────────────────────
      heading1("The autonomous agent framework"),

      para("QBITEL Bridge\u2019s security response layer is built around a multi-agent architecture. Agents are specialised workers with typed capabilities including threat analysis, protocol analysis, anomaly detection, incident response, and compliance auditing."),

      para("Each agent inherits from BaseAgent, which provides:"),

      bullet("lifecycle management (start, stop, health check)", "bullets3"),
      bullet("inter-agent communication via a typed message bus", "bullets3"),
      bullet("persistent memory with configurable retention policies", "bullets3"),
      bullet("automatic Prometheus metrics instrumentation", "bullets3"),

      para("The LLM integration layer connects agents to local or cloud language models \u2014 including Ollama for fully air-gapped deployments. This enables agents to generate human-readable incident reports, suggest remediation steps, and explain anomalies in plain language."),

      para("When a threat is detected, the IncidentResponseAgent autonomously executes a playbook: isolating the affected session, generating a CERT-In-compatible incident report, and notifying operators \u2014 all within the 10-second response SLA."),

      separator(),

      // ── Domain-specific modules ──────────────────────────────────
      heading1("Domain-specific modules"),

      para("QBITEL Bridge ships with purpose-built modules for regulated industries."),

      new Paragraph({
        spacing: { after: 200 },
        children: [
          new TextRun({ text: "Banking: ", font: "Arial", size: 22, italics: true }),
          new TextRun({ text: "SWIFT proxy re-encryption, PCI-DSS DTMF masking for call centre voice channels, a regulatory proof engine for RBI/SEBI compliance automation, and multi-authority threshold signatures for cross-bank transaction authorisation.", font: "Arial", size: 22 }),
        ],
      }),

      new Paragraph({
        spacing: { after: 200 },
        children: [
          new TextRun({ text: "Healthcare: ", font: "Arial", size: 22, italics: true }),
          new TextRun({ text: "EHR proxy re-encryption compatible with FHIR/HL7, homomorphic encryption for vital sign aggregation without exposing raw patient data, and ABDM-compatible verifiable credentials.", font: "Arial", size: 22 }),
        ],
      }),

      new Paragraph({
        spacing: { after: 200 },
        children: [
          new TextRun({ text: "Automotive: ", font: "Arial", size: 22, italics: true }),
          new TextRun({ text: "V2X (vehicle-to-everything) group signatures for multi-vehicle coordination, and misbehaviour detection for rogue vehicle identification in C-V2X networks.", font: "Arial", size: 22 }),
        ],
      }),

      new Paragraph({
        spacing: { after: 200 },
        children: [
          new TextRun({ text: "Aviation: ", font: "Arial", size: 22, italics: true }),
          new TextRun({ text: "ARINC 429/629-aware aggregate signatures and forward-secure channel establishment for flight data links.", font: "Arial", size: 22 }),
        ],
      }),

      new Paragraph({
        spacing: { after: 200 },
        children: [
          new TextRun({ text: "Industrial and critical infrastructure: ", font: "Arial", size: 22, italics: true }),
          new TextRun({ text: "IEC 61850 GOOSE message authentication for power grid substations, and verifiable delay functions for timing-sensitive industrial control sequences.", font: "Arial", size: 22 }),
        ],
      }),

      new Paragraph({
        spacing: { after: 200 },
        children: [
          new TextRun({ text: "BPO and call centres: ", font: "Arial", size: 22, italics: true }),
          new TextRun({ text: "real-time DTMF masking (card numbers never reach agent headsets or recording systems), SIP/RTP quantum-safe encryption, SS7 overlay protection, and TN3270e/TN5250 mainframe session tunnelling.", font: "Arial", size: 22 }),
        ],
      }),

      separator(),

      // ── Architecture and deployment ──────────────────────────────
      heading1("Architecture and deployment"),

      para("The system has two primary runtime components. Figure 3 shows how they interact."),

      figureCaption("Figure 3: High-level system architecture showing the Python AI engine and Rust dataplane"),

      new Paragraph({
        spacing: { after: 200 },
        children: [
          new TextRun({ text: "AI engine (Python): ", font: "Arial", size: 22, italics: true }),
          new TextRun({ text: "the ML pipeline, agent framework, LLM integration, crypto layer, and REST API. Uses FastAPI, PyTorch, LangGraph for agent orchestration, and Redis for distributed caching.", font: "Arial", size: 22 }),
        ],
      }),

      new Paragraph({
        spacing: { after: 200 },
        children: [
          new TextRun({ text: "Rust dataplane: ", font: "Arial", size: 22, italics: true }),
          new TextRun({ text: "high-performance packet processing, PQC-TLS termination, and protocol proxying. Built with the Tokio async runtime, achieving 10M+ packets/second with sub-10ms classification latency.", font: "Arial", size: 22 }),
        ],
      }),

      para("The Rust dataplane handles the hot path (packet forwarding and encryption) while the Python AI engine handles the cold path (protocol learning, threat analysis, and agent reasoning). The two communicate over a typed gRPC interface."),

      heading2("Getting started"),

      codeMarker(),
      codeLine("# Clone the repository"),
      codeLine("git clone https://github.com/yazhsab/qbitel-bridge.git"),
      codeLine("cd qbitel-bridge"),
      codeLine(""),
      codeLine("# Install Python AI engine dependencies"),
      codeLine("pip install -r ai_engine/requirements.txt"),
      codeLine(""),
      codeLine("# Build the Rust dataplane"),
      codeLine("cd rust/dataplane && cargo build --release"),
      codeLine(""),
      codeLine("# Start protocol discovery on a network interface"),
      codeLine("python -m ai_engine.discovery.cli --interface eth0 --output discovered.json"),
      codeLine(""),
      codeLine("# Run with Docker Compose (includes Redis, Prometheus, Grafana)"),
      codeLine("docker compose up -d"),
      codeMarker(),

      para("Once discovery completes, results are available via the REST API:"),

      codeMarker(),
      codeLine("import httpx"),
      codeLine(""),
      codeLine("async with httpx.AsyncClient() as client:"),
      codeLine("    result = await client.post("),
      codeLine('        "http://localhost:8080/api/v1/discovery/analyze",'),
      codeLine("        json={"),
      codeLine('            "traffic_data": ["<base64_packet_1>", "<base64_packet_2>"],'),
      codeLine('            "options": {'),
      codeLine('                "confidence_threshold": 0.7,'),
      codeLine('                "enable_adaptive_learning": True'),
      codeLine("            }"),
      codeLine("        }"),
      codeLine("    )"),
      codeLine('    protocols = result.json()["discovered_protocols"]'),
      codeLine("    for p in protocols:"),
      codeLine('        print(f"{p[\'name\']}: {p[\'confidence\']:.0%} confidence")'),
      codeMarker(),

      separator(),

      // ── Observability built in ───────────────────────────────────
      heading1("Observability built in"),

      para("QBITEL Bridge exports Prometheus metrics out of the box. Every agent task, protocol discovery request, crypto operation, and cache hit is measured. A sample of available metrics:"),

      codeMarker(),
      codeLine('protocol_discovery_requests_total{status="success"} 1547'),
      codeLine('protocol_discovery_duration_seconds_bucket{le="0.1"} 892'),
      codeLine('qbitel_agent_tasks_total{agent_type="incident_response"} 423'),
      codeLine('pqc_operations_total{algorithm="ml-kem-768", operation="encapsulate"} 98412'),
      codeMarker(),

      para("Grafana dashboard configurations are included in ops/grafana-dashboards/. Default alerts fire on CPU usage above 90%, memory above 85%, and elevated error rates."),

      separator(),

      // ── Performance at a glance ──────────────────────────────────
      heading1("Performance at a glance"),

      bullet("Protocol discovery accuracy: 89%+ on first pass (tested across mixed enterprise segments with 30+ protocol types)", "bullets4"),
      bullet("Discovery time on a typical enterprise network segment: 2 to 4 hours", "bullets4"),
      bullet("Packet throughput (Rust dataplane): 10M+ packets/second (benchmarked on AMD EPYC 7763 with NVIDIA A100 GPU acceleration)", "bullets4"),
      bullet("Classification latency (95th percentile): under 10ms", "bullets4"),
      bullet("PQC encryption overhead on voice (SIP/RTP): under 2ms", "bullets4"),
      bullet("Autonomous incident response (detection to containment): under 10 seconds end to end", "bullets4"),
      bullet("Memory footprint (typical production workload): under 2 GB", "bullets4"),
      bullet("Cache hit rate with Redis enabled: above 95%", "bullets4"),

      separator(),

      // ── Known limitations ────────────────────────────────────────
      heading1("Known limitations"),

      para("No tool solves everything, and transparency about boundaries is important. The current grammar inference engine performs best on binary protocols with fixed-length headers; text-based protocols with complex state machines (such as HTTP/2 or gRPC) show lower first-pass accuracy and may require additional training passes. The Rust dataplane\u2019s 10M+ packet throughput assumes GPU acceleration \u2014 CPU-only deployments should expect roughly 2M packets/second. Air-gapped LLM integration via Ollama is functional but slower than cloud-hosted models, and the quality of incident reports depends on the local model\u2019s capabilities. Finally, while the PQC algorithms are NIST-standardised, the implementations have not yet undergone formal third-party cryptographic audit \u2014 contributions from the security research community on this front are especially welcome."),

      separator(),

      // ── Why open source? ─────────────────────────────────────────
      heading1("Why open source?"),

      para("Security tooling that organisations cannot inspect is security tooling they cannot trust. QBITEL Bridge is released under the MIT licence \u2014 you can read every line of the cryptography implementation, audit the ML models, and verify the claims independently."),

      para("For India\u2019s public sector in particular, this matters enormously. DRDO, ISRO, and defence PSUs cannot deploy cloud-dependent foreign commercial tools in classified environments. An open source, air-gappable platform built on NIST-standardised algorithms provides a path to quantum readiness that proprietary tools simply cannot offer."),

      para("The community is invited to contribute across several active areas: new domain protocol modules, additional LLM provider integrations, performance benchmarks on different hardware, and compliance templates for new regulatory jurisdictions."),

      separator(),

      // ── Getting involved ─────────────────────────────────────────
      heading1("Getting involved"),

      bullet("Repository: github.com/yazhsab/qbitel-bridge", "bullets5"),
      bullet("Issue tracker: GitHub issues on the same repository", "bullets5"),
      bullet("Security disclosures: security@qbitel.com", "bullets5"),
      bullet("General discussion: support@qbitel.com", "bullets5"),

      separator(),

      // ── Conclusion ───────────────────────────────────────────────
      heading1("Conclusion"),

      para("The convergence of quantum computing and decades of legacy infrastructure debt is not a future problem. Adversaries are already harvesting encrypted traffic. Networks are already running protocols nobody documented. The gap between what enterprise security tools protect and what actually runs on enterprise networks has never been wider."),

      para("QBITEL Bridge addresses this gap with three open, inspectable technologies working in concert: probabilistic grammar inference for protocol discovery, NIST-standardised post-quantum cryptography for forward-secure encryption, and autonomous AI agents for sub-10-second incident response. The platform deploys as a transparent network layer, requires no application changes, and is fully observable through standard Prometheus and Grafana tooling."),

      para("For any organisation running legacy mainframes, industrial control systems, or regulated workloads, this is the open source foundation for a quantum-safe future."),
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("/Users/prabakarankannan/qbitel/docs/osfy_article_qbitel_bridge.docx", buffer);
  console.log("Created osfy_article_qbitel_bridge.docx");
});
