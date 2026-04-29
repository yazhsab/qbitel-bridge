export type ProductModule = {
  id: 'discover' | 'understand' | 'modernize' | 'protect' | 'prove';
  label: string;
  eyebrow: string;
  headline: string;
  summary: string;
  outcomes: string[];
  evidence: string[];
  protocols: string[];
  color: string;
  href: string;
};

export type VerticalStory = {
  id: 'bpo' | 'banking' | 'iot' | 'defense' | 'infrastructure' | 'healthcare' | 'telecom' | 'mobility';
  label: string;
  title: string;
  challenge: string;
  qbitelFit: string;
  pqcProfile: string;
  deployment: string;
  protocols: string[];
  priorities: string[];
};

export type RoadmapPhase = {
  id: string;
  window: string;
  label: string;
  status: 'active' | 'next' | 'planned' | 'scale';
  objective: string;
  deliverables: string[];
};

export const productModules: ProductModule[] = [
  {
    id: 'discover',
    label: 'Discover',
    eyebrow: 'System Inventory',
    headline: 'Turn mirrored traffic into a living protocol graph.',
    summary:
      'QBITEL learns undocumented message types, fields, dependencies, ownership, and modernization risk from the traffic you already have.',
    outcomes: [
      'Protocol asset records with confidence scoring',
      'System and dependency graph for legacy estates',
      'Crypto posture baseline and risk hotspots',
      'Analyst-ready protocol evidence in hours, not quarters',
    ],
    evidence: [
      'Protocol inventory',
      'Traffic samples and field maps',
      'Ownership and dependency graph',
      'Exposure snapshot',
    ],
    protocols: ['ISO 8583', 'SWIFT', 'Modbus', 'IEC 61850', 'HL7', 'Diameter'],
    color: 'from-neon-cyan to-quantum-400',
    href: '/products/protocol-discovery',
  },
  {
    id: 'understand',
    label: 'Understand',
    eyebrow: 'System Intelligence',
    headline: 'Explain behavior, business meaning, risk, and failure patterns.',
    summary:
      'QBITEL turns raw protocol and system signals into operator-ready intelligence: what the system does, why it matters, where it fails, and which risks deserve action.',
    outcomes: [
      'Business meaning inferred from messages and workflows',
      'Risk, fraud, anomaly, and failure patterns surfaced',
      'Legacy dependency graph tied to operational ownership',
      'Actionable recommendations with confidence and traceability',
    ],
    evidence: [
      'Behavior summary',
      'Risk and fraud score',
      'Failure indicators',
      'Recommendation trail',
    ],
    protocols: ['SIP/RTP', 'IVR flows', 'COBOL/JCL', 'MQ', 'SCADA telemetry', 'Device logs'],
    color: 'from-neon-green to-quantum-400',
    href: '/products/legacy-whisperer',
  },
  {
    id: 'modernize',
    label: 'Modernize',
    eyebrow: 'Integration Factory',
    headline: 'Generate specs, adapters, APIs, and migration blueprints.',
    summary:
      'Move from reverse engineering to usable deliverables: reviewed protocol specs, adapter stubs, OpenAPI contracts, SDKs, and rollout plans.',
    outcomes: [
      'Spec review and approval workflow',
      'Generated adapters and OpenAPI contracts',
      'Replay harnesses and validation fixtures',
      'Migration sequencing for low-downtime cutovers',
    ],
    evidence: [
      'Approved protocol spec',
      'Adapter build package',
      'Validation report',
      'Change plan',
    ],
    protocols: ['COBOL copybooks', 'TN3270e', 'FIX', 'DNP3', 'FHIR'],
    color: 'from-neon-copper to-neon-cyan',
    href: '/products/translation-studio',
  },
  {
    id: 'protect',
    label: 'Protect',
    eyebrow: 'Quantum-Safe Overlay',
    headline: 'Add PQC without replacing the systems that run the business.',
    summary:
      'Apply domain-aware ML-KEM, ML-DSA, hybrid key exchange, and runtime policy controls at the network and service boundaries where change is acceptable.',
    outcomes: [
      'PQC migration planner with dual-stack transition paths',
      'Overlay protection for legacy and constrained systems',
      'Runtime controls with rollback and simulation gates',
      'Domain-tuned profiles for banking, OT, healthcare, and telecom',
    ],
    evidence: [
      'Crypto migration roadmap',
      'Policy package',
      'Deployment manifest',
      'Rollback plan',
    ],
    protocols: ['ML-KEM-768', 'ML-KEM-1024', 'ML-DSA-65', 'Hybrid TLS'],
    color: 'from-quantum-500 to-neon-green',
    href: '/products/post-quantum-crypto',
  },
  {
    id: 'prove',
    label: 'Prove',
    eyebrow: 'Evidence Pack',
    headline: 'Prove what changed, why it changed, and how it is controlled.',
    summary:
      'Bundle protocol lineage, approvals, SBOMs, attestation, and audit evidence into one enterprise-ready record for architecture, security, and compliance teams.',
    outcomes: [
      'Evidence pack generated per deployment',
      'Approval workflow for risk and change control',
      'SBOM and attestation outputs',
      'Board and audit reporting tied to technical reality',
    ],
    evidence: [
      'Evidence pack',
      'SBOM and attestation',
      'Audit trail',
      'Executive summary',
    ],
    protocols: ['SOC 2', 'PCI DSS', 'HIPAA', 'NERC CIP', 'ISO 27001'],
    color: 'from-neon-green to-neon-cyan',
    href: '/products/enterprise-compliance',
  },
];

export const verticalStories: VerticalStory[] = [
  {
    id: 'bpo',
    label: 'BPO / Call Center',
    title: 'Voice-channel protection for fraud, PCI, and customer trust',
    challenge:
      'BPOs and contact centers run high-volume SIP, RTP, IVR, CRM, and recording flows where toll fraud, payment leakage, voice phishing, and agent misuse create immediate financial exposure.',
    qbitelFit:
      'Discover call flows, classify SIP/RTP behavior, detect fraud patterns, protect sensitive voice and payment paths, and produce PCI/HIPAA-ready evidence for client audits.',
    pqcProfile: 'Hybrid ML-KEM-768 for protected voice gateways and long-lived client data paths',
    deployment: 'Passive SIP/RTP ingest, fraud baseline, gateway protection, evidence pack for client assurance',
    protocols: ['SIP', 'RTP/SRTP', 'WebRTC', 'IVR logs', 'PCI voice capture', 'CRM APIs'],
    priorities: ['Toll fraud reduction', 'PCI evidence', 'Voice-channel visibility', 'Client assurance'],
  },
  {
    id: 'banking',
    label: 'Banking',
    title: 'Mainframe modernization without breaking payment rails',
    challenge:
      'Core banking estates still depend on undocumented ISO 8583, SWIFT, FIX, and proprietary batch interfaces that cannot tolerate downtime.',
    qbitelFit:
      'Discover message structure, generate integration surfaces, apply PQC overlays to sensitive flows, and produce a board-ready migration program.',
    pqcProfile: 'ML-KEM-1024 + ML-DSA-87 for long-lived, high-value traffic',
    deployment: 'Mirrored traffic capture, adapter generation, staged overlay activation',
    protocols: ['ISO 8583', 'SWIFT MT/MX', 'FIX', 'TN3270e', 'COBOL/JCL'],
    priorities: ['Protocol inventory', 'API generation', 'PQC roadmap', 'Audit evidence'],
  },
  {
    id: 'iot',
    label: 'IoT',
    title: 'Device trust for unmanaged fleets and fragmented protocols',
    challenge:
      'IoT estates contain thousands of devices with unknown firmware, weak crypto, proprietary protocols, and inconsistent update paths.',
    qbitelFit:
      'Fingerprint devices, map protocol behavior, score crypto posture, detect anomalous device traffic, and produce upgrade or containment plans without touching every endpoint first.',
    pqcProfile: 'Constrained ML-KEM-512/768 profiles with gateway-based protection for limited devices',
    deployment: 'Network discovery, device trust graph, firmware and crypto posture review, staged gateway controls',
    protocols: ['MQTT', 'CoAP', 'BLE', 'Zigbee', 'LoRaWAN', 'Proprietary binary'],
    priorities: ['Device inventory', 'Crypto posture', 'Firmware risk', 'Gateway protection'],
  },
  {
    id: 'defense',
    label: 'Defense',
    title: 'Mission-system modernization under strict control',
    challenge:
      'Defense environments combine air-gapped networks, long-lived platforms, classified data, fragile legacy interfaces, and CNSA 2.0 migration pressure.',
    qbitelFit:
      'Run offline, discover mission protocols, generate signed evidence, apply governed PQC migration plans, and require explicit approval for every critical action.',
    pqcProfile: 'CNSA 2.0-aligned ML-KEM-1024, ML-DSA-87, and LMS/XMSS where required',
    deployment: 'Air-gapped assessment, protocol assurance, approved overlay plan, signed ATO evidence package',
    protocols: ['MIL-STD messages', 'Tactical data links', 'Serial links', 'Legacy IP', 'Custom mission protocols'],
    priorities: ['Offline operation', 'CNSA 2.0 readiness', 'ATO evidence', 'Human approval gates'],
  },
  {
    id: 'infrastructure',
    label: 'Critical Infrastructure',
    title: 'OT and SCADA visibility with low-change protection',
    challenge:
      'Operators need asset inventory, protocol understanding, and better control posture without taking plants or substations offline.',
    qbitelFit:
      'Build the OT protocol graph first, identify critical flows, then add low-friction overlay protections and evidence for operational security programs.',
    pqcProfile: 'Hybrid X25519/ML-KEM-768 or P-384/ML-KEM-1024 depending latency budgets',
    deployment: 'Passive ingest, architecture graphing, governed overlay rollout',
    protocols: ['Modbus', 'DNP3', 'IEC 61850', 'OPC UA'],
    priorities: ['Asset inventory', 'Critical flow mapping', 'Overlay controls', 'Operations evidence'],
  },
  {
    id: 'healthcare',
    label: 'Healthcare',
    title: 'Protect legacy devices and interoperability paths without firmware rewrites',
    challenge:
      'Hospitals cannot realistically modify every certified device, but they still need secure integration and future-ready cryptographic posture.',
    qbitelFit:
      'Discover device protocols, generate safe integration layers, and secure data paths externally while preserving operational and certification boundaries.',
    pqcProfile: 'Constrained-device ML-KEM-512 or hybrid profiles with narrow latency budgets',
    deployment: 'Device edge inventory, protocol bridge generation, network-layer protection',
    protocols: ['HL7 v2', 'FHIR', 'DICOM', 'X12'],
    priorities: ['Interoperability', 'Selective protection', 'Evidence generation', 'Migration sequencing'],
  },
  {
    id: 'telecom',
    label: 'Telecom',
    title: 'Mediation and signaling modernization at network scale',
    challenge:
      'Telecom signaling stacks mix legacy, proprietary, and high-throughput interfaces that are hard to catalog and expensive to replace.',
    qbitelFit:
      'Map signaling flows, generate translation assets for mediation layers, and apply PQC profiles where long-lived or sensitive traffic justifies it.',
    pqcProfile: 'ML-KEM-768 + ML-DSA-65 with hybrid preference for interoperability',
    deployment: 'Signaling capture, mediation templates, staged network rollout',
    protocols: ['Diameter', 'SS7', 'SIGTRAN', 'SIP', 'GTP', 'SMPP'],
    priorities: ['Signaling inventory', 'Adapter generation', 'Hybrid rollout', 'Operator reporting'],
  },
  {
    id: 'mobility',
    label: 'Aviation / Automotive',
    title: 'Long-life mobility systems with safety and certification boundaries',
    challenge:
      'Aircraft, vehicles, and mobility infrastructure have constrained links, long certification cycles, safety boundaries, and protocols that must remain stable for decades.',
    qbitelFit:
      'Discover and validate mobility protocols, generate safe gateway patterns, preserve certification boundaries, and produce evidence for security and safety reviews.',
    pqcProfile: 'Latency- and bandwidth-aware hybrid profiles with crypto agility over long platform life',
    deployment: 'Ground or edge gateway discovery, protocol validation, staged protection, certification evidence',
    protocols: ['ADS-B', 'ACARS', 'ARINC 429', 'CAN', 'V2X', 'IEEE 1609.2'],
    priorities: ['Safety boundaries', 'Protocol validation', 'Crypto agility', 'Certification evidence'],
  },
];

export const roadmapPhases: RoadmapPhase[] = [
  {
    id: 'phase-1',
    window: '0-6 months',
    label: 'Discovery, understanding, and modernization core',
    status: 'active',
    objective: 'Turn QBITEL into a clear protocol discovery and modernization product.',
    deliverables: [
      'Protocol asset record as the core domain object',
      'Interactive protocol graph and system inventory',
      'System intelligence, risk scoring, and spec review workflow',
      'Adapter, API, and replay harness generation',
    ],
  },
  {
    id: 'phase-2',
    window: '6-12 months',
    label: 'Protection and PQC transition',
    status: 'next',
    objective: 'Add credible crypto posture mapping and overlay rollout.',
    deliverables: [
      'PQC migration planner and dual-stack transition paths',
      'Policy-controlled protection modes',
      'Simulation and rollback controls',
      'Executive reporting for crypto exposure',
    ],
  },
  {
    id: 'phase-3',
    window: '12-18 months',
    label: 'Proof and evidence',
    status: 'planned',
    objective: 'Make auditability and control a first-class product surface.',
    deliverables: [
      'Evidence pack automation',
      'Attestation and approval workflows',
      'SBOM and deployment lineage outputs',
      'Private enterprise protocol registry',
    ],
  },
  {
    id: 'phase-4',
    window: '18-24 months',
    label: 'Scale and ecosystem',
    status: 'scale',
    objective: 'Expand into repeatable enterprise programs without losing focus.',
    deliverables: [
      'Reusable vertical templates',
      'Partner-ready deployment motions',
      'Multi-site and multi-domain expansion tooling',
      'Selective governed automation with approval gates',
    ],
  },
];

export const proofArtifacts = [
  'Protocol spec and lineage',
  'System behavior and risk explanation',
  'Adapter and API package',
  'Crypto posture summary',
  'Deployment manifest',
  'SBOM and attestation',
  'Change-control evidence',
];

export const signalMetrics = [
  { label: 'Time to first protocol inventory', value: 'Hours' },
  { label: 'Time to first risk narrative', value: 'Same Day' },
  { label: 'Time to first adapter package', value: 'Days' },
  { label: 'Time to first PQC migration plan', value: '< 1 Week' },
  { label: 'Time to first evidence pack', value: 'Same Workflow' },
];
