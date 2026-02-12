# QBITEL Bridge - Complete Technical Reference Document
## Compiled for NotebookLM

This document contains the COMPLETE contents of 15 technical documentation files from the QBITEL Bridge repository. It is intended as a comprehensive reference for NotebookLM ingestion.

---

# ============================================================
# FILE 1: README.md (Root README - Master Overview)
# ============================================================

<p align="center">
  <img src="diagrams/01_system_architecture.svg" alt="QBITEL Bridge" width="800">
</p>

<h1 align="center">QBITEL Bridge</h1>

<h3 align="center">Enterprise-Grade Open-Source Platform for AI-Powered Quantum-Safe Legacy Modernization</h3>

<p align="center">
  <em>The only open-source platform that discovers unknown protocols with AI, encrypts them with post-quantum cryptography, and defends them autonomously — without replacing your legacy systems.</em>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Enterprise-Open%20Source-brightgreen.svg" alt="Enterprise Open Source">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.70+-orange.svg" alt="Rust 1.70+"></a>
  <a href="https://go.dev/"><img src="https://img.shields.io/badge/go-1.21+-00ADD8.svg" alt="Go 1.21+"></a>
  <a href="https://kubernetes.io"><img src="https://img.shields.io/badge/kubernetes-ready-326CE5.svg" alt="Kubernetes"></a>
  <img src="https://img.shields.io/badge/NIST-Level%205-green.svg" alt="NIST Level 5">
  <img src="https://img.shields.io/badge/PQC-ML--KEM%20%7C%20ML--DSA-purple.svg" alt="Post-Quantum">
  <img src="https://img.shields.io/badge/LLM-On--Premise%20First-red.svg" alt="On-Premise LLM">
</p>

<br>

<p align="center">
  <a href="https://bridge.qbitel.com">Website</a> &nbsp;&bull;&nbsp;
  <a href="docs/">Documentation</a> &nbsp;&bull;&nbsp;
  <a href="QUICKSTART.md">Quick Start</a> &nbsp;&bull;&nbsp;
  <a href="docs/API.md">API Reference</a> &nbsp;&bull;&nbsp;
  <a href="mailto:enterprise@qbitel.com">Enterprise Support</a>
</p>

---

<table>
<tr>
<td align="center"><h3>89%+</h3><sub>Discovery Accuracy</sub></td>
<td align="center"><h3>NIST Level 5</h3><sub>Quantum Security</sub></td>
<td align="center"><h3>&lt;1ms</h3><sub>Encryption Overhead</sub></td>
<td align="center"><h3>78%</h3><sub>Autonomous Response</sub></td>
<td align="center"><h3>9</h3><sub>Compliance Frameworks</sub></td>
<td align="center"><h3>100K+</h3><sub>msg/sec Throughput</sub></td>
</tr>
</table>

<p align="center">
  <strong>100% Open Source</strong> &nbsp;·&nbsp; Apache 2.0 Licensed &nbsp;·&nbsp; No Open-Core &nbsp;·&nbsp; No Feature Gating &nbsp;·&nbsp; Production-Ready
</p>

---

## The Problem

Sixty percent of Fortune 500 companies run critical operations on systems built 20-40 years ago. These systems process trillions of dollars daily, keep power grids running, and manage patient records. They share three fatal weaknesses:

<table>
<tr>
<td width="33%">

### The Legacy Crisis
Undocumented protocols with **no source code**, no documentation, and no original developers. Manual reverse engineering costs **$2M-10M** and takes **6-12 months** per system.

</td>
<td width="33%">

### The Quantum Threat
Quantum computers will break RSA/ECC within 5-10 years. Adversaries are **harvesting encrypted data today** to decrypt later. Every unprotected wire transfer, patient record, and grid command is a future breach.

</td>
<td width="33%">

### The Speed Gap
Average SOC response time: **65 minutes**. In that window, an attacker can exfiltrate 100GB, encrypt an entire network, or manipulate industrial controls. **Human speed cannot match machine-speed attacks.**

</td>
</tr>
</table>

**QBITEL Bridge solves all three.** No rip-and-replace. No downtime. No cloud dependency.

---

## How It Works

```
  Legacy System                    QBITEL Bridge                         Modern Infrastructure
                        ┌──────────────────────────────────┐
  COBOL Mainframes      │                                  │          Cloud APIs
  SCADA / PLCs     ───► │  1. DISCOVER  unknown protocols  │ ───►     Microservices
  Medical Devices       │  2. PROTECT   with quantum crypto│          Dashboards
  Banking Terminals     │  3. TRANSLATE to modern APIs     │          Event Streams
  Custom Protocols      │  4. COMPLY    across 9 frameworks│          Data Lakes
                        │  5. OPERATE   with autonomous AI │
                        └──────────────────────────────────┘
                                 2-4 hours to first results
```

| Step | What Happens | Traditional Approach | With QBITEL |
|------|-------------|---------------------|-------------|
| **Discover** | AI learns protocol structure from raw traffic | 6-12 months, $2-10M | **2-4 hours**, automated |
| **Protect** | Wraps communications in NIST Level 5 PQC | Not available | **ML-KEM + ML-DSA**, <1ms |
| **Translate** | Generates REST APIs + SDKs in 6 languages | Weeks of manual coding | **Minutes**, auto-generated |
| **Comply** | Produces audit-ready reports for 9 frameworks | $500K-1M/year manual | **<10 min**, automated |
| **Operate** | Autonomous threat detection and response | 65 min avg SOC response | **<1 sec**, 78% autonomous |

---

## Platform Capabilities

<table>
<tr>
<td width="50%">

### AI Protocol Discovery
Reverse-engineers unknown protocols from raw network traffic using PCFG grammar inference, Transformer classification, and BiLSTM-CRF field detection. No source code needed.

[Learn more](docs/products/01_AI_PROTOCOL_DISCOVERY.md) &bull; [Architecture](diagrams/03_protocol_discovery_pipeline.svg)

</td>
<td width="50%">

### Post-Quantum Cryptography
NIST Level 5 protection with **ML-KEM** (Kyber-1024) and **ML-DSA** (Dilithium-5). Domain-optimized for banking (10K+ TPS), healthcare (64KB devices), automotive (<10ms V2X), and aviation (600bps links).

[Learn more](docs/products/05_POST_QUANTUM_CRYPTOGRAPHY.md) &bull; [Architecture](diagrams/06_quantum_cryptography.svg)

</td>
</tr>
<tr>
<td width="50%">

### Zero-Touch Security Engine
LLM-powered autonomous response with confidence-driven execution. Auto-executes at >95% confidence, escalates at <50%. MITRE ATT&CK mapping. Full audit trail.

[Learn more](docs/products/04_AGENTIC_AI_SECURITY.md) &bull; [Architecture](diagrams/04_zero_touch_decision_engine.svg)

</td>
<td width="50%">

### Legacy System Whisperer
Deep analysis for undocumented legacy systems: COBOL copybook parsing, JCL analysis, mainframe dataset discovery, business rule extraction, and predictive failure analysis.

[Learn more](docs/LEGACY_SYSTEM_WHISPERER.md)

</td>
</tr>
<tr>
<td width="50%">

### Translation Studio
Point at any protocol, get a REST API with OpenAPI 3.0 spec + SDKs in Python, TypeScript, Go, Java, Rust, and C#. Auto-generated, fully documented, production-ready.

[Learn more](docs/products/02_TRANSLATION_STUDIO.md) &bull; [Architecture](diagrams/05_translation_studio_workflow.svg)

</td>
<td width="50%">

### Protocol Marketplace
Community-driven protocol knowledge sharing with 1,000+ pre-built adapters. Publish, discover, and monetize protocol definitions. Automated validation pipeline with security scanning.

[Learn more](docs/products/03_PROTOCOL_MARKETPLACE.md)

</td>
</tr>
<tr>
<td width="50%">

### Multi-Agent Orchestration
16+ specialized AI agents with 5 execution strategies (parallel, sequential, pipeline, consensus, adaptive). Persistent memory, dynamic scaling, coordinated incident response.

[Architecture](diagrams/02_ai_agent_ecosystem.svg)

</td>
<td width="50%">

### Enterprise Compliance
Automated reporting for **SOC 2, GDPR, HIPAA, PCI-DSS, ISO 27001, NIST 800-53, BASEL-III, NERC-CIP, FDA 21 CFR Part 11**. Continuous monitoring. Blockchain-backed audit trails.

[Learn more](docs/products/07_ENTERPRISE_COMPLIANCE.md)

</td>
</tr>
<tr>
<td width="50%">

### Cloud-Native Security
Kubernetes admission webhooks, Istio/Envoy service mesh with quantum-safe mTLS, eBPF runtime monitoring, container image scanning, and secure Kafka event streaming.

[Learn more](docs/products/06_CLOUD_NATIVE_SECURITY.md)

</td>
<td width="50%">

### Threat Intelligence
MITRE ATT&CK technique mapping, STIX/TAXII feed integration, proactive threat hunting, and continuous learning from global threat landscape.

[Learn more](docs/products/09_THREAT_INTELLIGENCE.md)

</td>
</tr>
</table>

---

## Architecture

Four-layer polyglot architecture — each layer built in the language optimized for its job:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        UI Console  (React / TypeScript)                  │
│               Admin Dashboard  ·  Protocol Copilot  ·  Marketplace       │
├──────────────────────────────────────────────────────────────────────────┤
│                        Go Control Plane                                  │
│          Service Orchestration  ·  OPA Policy Engine  ·  Vault Secrets   │
│          Device Agent (TPM 2.0)  ·  gRPC + REST Gateway                  │
├──────────────────────────────────────────────────────────────────────────┤
│                        Python AI Engine  (FastAPI)                       │
│   Protocol Discovery  ·  Multi-Agent System  ·  LLM/RAG (Ollama-first)   │
│   Compliance Automation  ·  Anomaly Detection  ·  Security Orchestrator  │
│   Legacy Whisperer  ·  Marketplace  ·  Protocol Copilot                  │
├──────────────────────────────────────────────────────────────────────────┤
│                        Rust Data Plane                                   │
│     PQC-TLS (ML-KEM / ML-DSA)  ·  DPDK Packet Processing                 │
│     Wire-Speed Encryption  ·  DPI Engine  ·  Protocol Adapters           │
│     K8s Operator  ·  HA Clustering  ·  AI Bridge (PyO3)                  │
└──────────────────────────────────────────────────────────────────────────┘
```

| Layer | Language | What It Does |
|-------|----------|-------------|
| **Data Plane** | Rust | PQC-TLS termination, wire-speed encryption, DPDK packet processing, protocol adapters (ISO-8583, Modbus, HL7, TN3270e) |
| **AI Engine** | Python | Protocol discovery, LLM inference, compliance automation, anomaly detection, multi-agent orchestration |
| **Control Plane** | Go | Service orchestration, OPA policy evaluation, Vault secrets, device management, gRPC gateway |
| **UI Console** | React/TS | Admin dashboard, protocol copilot, marketplace, real-time monitoring |

[Detailed architecture](architecture.md) &bull; [Architecture diagrams](diagrams/)

---

## What Only QBITEL Does

No other platform combines these capabilities:

| Capability | QBITEL | CrowdStrike | Palo Alto | Fortinet | Claroty | IBM Quantum Safe |
|------------|:------:|:-----------:|:---------:|:--------:|:-------:|:----------------:|
| AI protocol discovery (2-4 hrs) | **Yes** | No | No | No | No | No |
| NIST Level 5 post-quantum crypto | **Yes** | No | No | No | No | Yes |
| Legacy system protection (40+ yr) | **Yes** | No | No | No | Partial | No |
| Autonomous security response (78%) | **Yes** | Playbooks | Playbooks | Playbooks | Alerts | No |
| Air-gapped on-premise LLM | **Yes** | Cloud-only | Cloud-only | Cloud-only | Cloud-only | No |
| Domain-optimized PQC | **Yes** | N/A | N/A | N/A | N/A | Generic |
| Auto-generated APIs + 6 SDKs | **Yes** | No | No | No | No | No |
| 9 compliance frameworks | **Yes** | Basic | Basic | Basic | OT only | Crypto only |
| Protocol marketplace (1000+) | **Yes** | No | No | No | No | No |

> *QBITEL occupies a new category: AI-powered quantum-safe security for the legacy and constrained systems that traditional vendors cannot protect.*

---

## Industry Solutions

| Industry | Protocols | Key Use Case | Business Impact |
|----------|-----------|-------------|-----------------|
| **Banking & Finance** | ISO-8583, SWIFT, SEPA, FIX | Mainframe transaction protection, HSM migration | Protect $10T+ daily transactions |
| **Healthcare** | HL7, DICOM, FHIR | Medical device security without FDA recertification | Secure 500K+ connected devices |
| **Critical Infrastructure** | Modbus, DNP3, IEC 61850 | SCADA/PLC protection with zero downtime | Protect power grids serving 100M+ |
| **Automotive** | V2X, IEEE 1609.2, CAN | Quantum-safe vehicle-to-everything | <10ms latency constraint |
| **Aviation** | ADS-B, ACARS, ARINC 429 | Air traffic and avionics data security | 600bps bandwidth-optimized PQC |
| **Telecommunications** | SS7, Diameter, SIP | 5G core and IoT infrastructure protection | Billion-device scale |

[Industry brochures](docs/brochures/)

---

## Performance

| Component | Metric | Value |
|-----------|--------|-------|
| Protocol Discovery | Time to first results | **2-4 hours** (vs 6-12 months manual) |
| Protocol Discovery | Classification accuracy | **89%+** |
| Protocol Discovery | P95 latency | **150ms** |
| PQC Encryption | Overhead | **<1ms** (AES-256-GCM + Kyber hybrid) |
| Kafka Streaming | Throughput | **100,000+ msg/sec** (encrypted) |
| Parser Generation | Parse throughput | **50,000+ msg/sec** |
| Security Engine | Decision time | **<1 second** (900x faster than manual SOC) |
| xDS Server | Proxy capacity | **1,000+ concurrent** |
| eBPF Monitor | Container capacity | **10,000+** containers at <1% CPU |
| API Gateway | P99 latency | **<25ms** |
| Translation Studio | SDK generation | **6 languages**, minutes not months |

---

## Get Started

### Prerequisites

- Python 3.10+ &bull; Rust 1.70+ &bull; Go 1.21+
- Docker & Docker Compose
- 4GB+ RAM (16GB+ recommended for production)

### Option 1: Docker Compose (Fastest)

```bash
git clone https://github.com/yazhsab/qbitel-bridge.git
cd qbitel-bridge
docker compose -f docker/docker-compose.yml up -d

# API available at http://localhost:8000/docs
# UI Console at http://localhost:3000
```

### Option 2: Python AI Engine

```bash
git clone https://github.com/yazhsab/qbitel-bridge.git
cd qbitel-bridge

python -m venv venv && source venv/bin/activate
pip install -e ".[all]"
python -m ai_engine

# Swagger UI at http://localhost:8000/docs
```

### Option 3: Kubernetes (Production)

```bash
helm install qbitel-bridge ./helm/qbitel-bridge \
  --namespace qbitel-bridge \
  --create-namespace \
  --wait

# Includes: AI Engine, Control Plane, xDS Server, Admission Webhook
# Pre-configured: Prometheus, Grafana, OpenTelemetry, Jaeger
```

### Option 4: Air-Gapped Deployment

```bash
# Install Ollama for on-premise LLM inference (no cloud required)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2:8b

# Run in fully air-gapped mode
export QBITEL_LLM_PROVIDER=ollama
export QBITEL_AIRGAPPED_MODE=true
python -m ai_engine --airgapped
```

[Detailed setup guide](QUICKSTART.md) &bull; [Production deployment](DEPLOYMENT.md) &bull; [Developer guide](DEVELOPMENT.md)

---

## Technology Stack

| Category | Technologies |
|----------|-------------|
| **AI / ML** | PyTorch, Transformers, Scikit-learn, SHAP, LIME, LangGraph |
| **Post-Quantum Crypto** | liboqs, kyber-py, dilithium-py, oqs-rs (Rust), Falcon, SLH-DSA |
| **LLM** | Ollama (primary), vLLM, Anthropic Claude, OpenAI (optional fallback) |
| **RAG** | ChromaDB, Sentence-Transformers, hybrid search, semantic caching |
| **Service Mesh** | Istio, Envoy xDS (gRPC), quantum-safe mTLS |
| **Container Security** | Trivy, eBPF/BCC, Kubernetes admission webhooks, cosign |
| **Event Streaming** | Kafka with AES-256-GCM message encryption |
| **Observability** | Prometheus, Grafana, OpenTelemetry, Jaeger, Sentry |
| **Cloud Integrations** | AWS Security Hub, Azure Sentinel, GCP Security Command Center |
| **Storage** | PostgreSQL / TimescaleDB, Redis, ChromaDB (vectors) |
| **Policy Engine** | Open Policy Agent (OPA / Rego) |
| **Secrets** | HashiCorp Vault, TPM 2.0 sealing |
| **CI/CD** | GitHub Actions, ArgoCD (GitOps), Helm |

---

## Project Structure

```
qbitel-bridge/
├── ai_engine/                # Python AI Engine (FastAPI)
│   ├── agents/               #   Multi-agent orchestration (16+ agents)
│   ├── anomaly/              #   Anomaly detection (VAE, LSTM, Isolation Forest)
│   ├── compliance/           #   Compliance automation (9 frameworks)
│   ├── copilot/              #   Protocol intelligence copilot
│   ├── crypto/               #   Post-quantum cryptography
│   ├── discovery/            #   Protocol discovery (PCFG, Transformers, BiLSTM-CRF)
│   ├── llm/                  #   LLM gateway (Ollama, RAG, guardrails)
│   ├── marketplace/          #   Protocol marketplace
│   ├── security/             #   Zero-touch security engine
│   ├── cloud_native/         #   Service mesh, container security
│   └── threat_intelligence/  #   MITRE ATT&CK, STIX/TAXII
├── rust/dataplane/           # Rust Data Plane
│   └── crates/
│       ├── pqc_tls/          #   ML-KEM / ML-DSA TLS implementation
│       ├── dpdk_engine/      #   DPDK packet processing
│       ├── dpi_engine/       #   Deep packet inspection
│       ├── ai_bridge/        #   Python-Rust FFI (PyO3)
│       └── k8s_operator/     #   Kubernetes operator
├── go/                       # Go Services
│   ├── controlplane/         #   gRPC + REST, OPA policies, Vault
│   ├── mgmtapi/              #   Device & certificate management
│   └── agents/device-agent/  #   Edge agent with TPM 2.0
├── ui/console/               # React Admin Console (TypeScript)
├── helm/qbitel-bridge/       # Production Helm chart
├── ops/                      # Grafana dashboards, Prometheus rules, runbooks
├── diagrams/                 # SVG architecture diagrams
├── samples/cobol/            # 500+ COBOL sample programs
├── docs/                     # 73+ documentation files
│   ├── products/             #   10 product guides
│   └── brochures/            #   9 industry brochures
└── tests/                    # Integration, load, chaos, smoke tests
```

---

## Documentation

| Document | Description |
|----------|-------------|
| **[QUICKSTART.md](QUICKSTART.md)** | Get running in under 10 minutes |
| **[architecture.md](architecture.md)** | System architecture and design decisions |
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | Production deployment guide |
| **[DEVELOPMENT.md](DEVELOPMENT.md)** | Developer setup and contribution workflow |
| **[docs/API.md](docs/API.md)** | REST and Python API reference |
| **[SECURITY.md](SECURITY.md)** | Security policy and vulnerability reporting |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Contribution guidelines |
| **[CHANGELOG.md](CHANGELOG.md)** | Release notes and version history |

### Product Guides

| # | Product | Guide |
|---|---------|-------|
| 1 | AI Protocol Discovery | [docs/products/01_AI_PROTOCOL_DISCOVERY.md](docs/products/01_AI_PROTOCOL_DISCOVERY.md) |
| 2 | Translation Studio | [docs/products/02_TRANSLATION_STUDIO.md](docs/products/02_TRANSLATION_STUDIO.md) |
| 3 | Protocol Marketplace | [docs/products/03_PROTOCOL_MARKETPLACE.md](docs/products/03_PROTOCOL_MARKETPLACE.md) |
| 4 | Agentic AI Security | [docs/products/04_AGENTIC_AI_SECURITY.md](docs/products/04_AGENTIC_AI_SECURITY.md) |
| 5 | Post-Quantum Cryptography | [docs/products/05_POST_QUANTUM_CRYPTOGRAPHY.md](docs/products/05_POST_QUANTUM_CRYPTOGRAPHY.md) |
| 6 | Cloud-Native Security | [docs/products/06_CLOUD_NATIVE_SECURITY.md](docs/products/06_CLOUD_NATIVE_SECURITY.md) |
| 7 | Enterprise Compliance | [docs/products/07_ENTERPRISE_COMPLIANCE.md](docs/products/07_ENTERPRISE_COMPLIANCE.md) |
| 8 | Zero Trust Architecture | [docs/products/08_ZERO_TRUST_ARCHITECTURE.md](docs/products/08_ZERO_TRUST_ARCHITECTURE.md) |
| 9 | Threat Intelligence | [docs/products/09_THREAT_INTELLIGENCE.md](docs/products/09_THREAT_INTELLIGENCE.md) |
| 10 | IAM & Monitoring | [docs/products/10_ENTERPRISE_IAM_MONITORING.md](docs/products/10_ENTERPRISE_IAM_MONITORING.md) |

### Industry Brochures

[Banking](docs/brochures/01_BANKING_FINANCIAL_SERVICES.md) &bull; [Healthcare](docs/brochures/02_HEALTHCARE.md) &bull; [Critical Infrastructure](docs/brochures/03_CRITICAL_INFRASTRUCTURE_SCADA.md) &bull; [Automotive](docs/brochures/04_AUTOMOTIVE.md) &bull; [Aviation](docs/brochures/05_AVIATION.md) &bull; [Telecommunications](docs/brochures/06_TELECOMMUNICATIONS.md) &bull; [Executive Overview](docs/brochures/08_EXECUTIVE_OVERVIEW.md)

---

## Contributing

We welcome contributions from the community. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Fork, clone, and set up
git clone https://github.com/YOUR_USERNAME/qbitel-bridge.git
cd qbitel-bridge && make bootstrap

# Run tests for each component
pytest ai_engine/tests/ -v                     # Python
cd rust/dataplane && cargo test                # Rust
cd go/controlplane && go test ./...            # Go
cd ui/console && npm test                      # React
```

---

## Security

QBITEL Bridge follows responsible disclosure practices. If you discover a security vulnerability, please report it to **security@qbitel.com**.

See [SECURITY.md](SECURITY.md) for our full security policy.

---

## Enterprise Open Source

QBITEL Bridge is **100% open source** under the **Apache License 2.0** — the same license trusted by Kubernetes, Kafka, Spark, and Airflow.

<table>
<tr>
<td width="50%">

### Community Edition (Free & Open Source)

- Full AI protocol discovery pipeline
- Post-quantum cryptography (ML-KEM, ML-DSA)
- Zero-touch security decision engine
- Multi-agent orchestration (16+ agents)
- 9 compliance frameworks
- Translation Studio (6 SDKs)
- Protocol marketplace access
- Community support via GitHub Issues

</td>
<td width="50%">

### Enterprise Support

- Dedicated support engineering team
- SLA-backed response times
- Production deployment assistance
- Custom protocol adapter development
- On-site training and enablement
- Architecture review and optimization
- Priority feature requests
- **Contact: [enterprise@qbitel.com](mailto:enterprise@qbitel.com)**

</td>
</tr>
</table>

> **No open-core. No feature gating. No bait-and-switch.** Every capability — from protocol discovery to post-quantum cryptography to autonomous security — is in the open-source release. Enterprise support provides the team, SLAs, and expertise to run it in production.

See [LICENSE](LICENSE) for full license details.

---

<p align="center">
  <strong>Built by <a href="https://qbitel.com">QBITEL</a></strong> &nbsp;|&nbsp; <strong>Enterprise-Grade Open Source &nbsp;|&nbsp; Securing the Quantum Future</strong>
</p>

<p align="center">
  <a href="https://bridge.qbitel.com">Website</a> &nbsp;&bull;&nbsp;
  <a href="docs/">Documentation</a> &nbsp;&bull;&nbsp;
  <a href="https://github.com/yazhsab/qbitel-bridge">GitHub</a> &nbsp;&bull;&nbsp;
  <a href="mailto:enterprise@qbitel.com">Enterprise Support</a>
</p>

<p align="center">
  <sub>Discover &bull; Protect &bull; Defend &bull; Autonomously</sub>
</p>


---

# ============================================================
# FILE 2: architecture.md (Full Architecture Document)
# ============================================================

# QBITEL - Technical Architecture

This document provides a comprehensive technical overview of the QBITEL platform architecture, component design, and implementation details.

## Table of Contents

- [System Overview](#system-overview)
- [Agentic AI Architecture](#agentic-ai-architecture)
- [Core Components](#core-components)
- [Cloud-Native Architecture](#cloud-native-architecture)
- [AI Engine Architecture](#ai-engine-architecture)
- [Security Architecture](#security-architecture)
- [Data Flow](#data-flow)
- [Deployment Architecture](#deployment-architecture)
- [Technology Stack](#technology-stack)

## System Overview

QBITEL is a distributed, cloud-native security platform designed for quantum-safe protection of legacy and modern systems. The architecture follows microservices patterns with strong security and observability.

![QBITEL Bridge Platform Architecture](diagrams/01_system_architecture.svg)

```
┌──────────────────────────────────────────────────────────────────┐
│                        QBITEL Platform                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │   AI Engine   │  │ Cloud-Native  │  │  Compliance   │       │
│  │   Protocol    │  │   Security    │  │  Automation   │       │
│  │   Discovery   │  │               │  │               │       │
│  └───────────────┘  └───────────────┘  └───────────────┘       │
│                                                                   │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │  Service Mesh │  │   Container   │  │     Event     │       │
│  │  Integration  │  │   Security    │  │   Streaming   │       │
│  │  (Istio/Envoy)│  │  (eBPF/Trivy) │  │    (Kafka)    │       │
│  └───────────────┘  └───────────────┘  └───────────────┘       │
│                                                                   │
│  ┌─────────────────────────────────────────────────────┐        │
│  │         Post-Quantum Cryptography Layer             │        │
│  │     Kyber-1024, Dilithium-5, AES-256-GCM           │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Agentic AI Architecture

QBITEL implements a **multi-agent architecture** where specialized AI agents autonomously manage security operations, protocol discovery, and threat response. The system uses LLM-powered reasoning, autonomous orchestration, and adaptive learning to minimize human intervention while maintaining safety and explainability.

![AI Agent Ecosystem](diagrams/02_ai_agent_ecosystem.svg)

### Agent System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                   QBITEL Agent Ecosystem                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────┐         ┌─────────────────────┐        │
│  │  Zero-Touch         │         │   Protocol          │        │
│  │  Decision Agent     │◄───────►│   Discovery Agent   │        │
│  │  (Security)         │         │   (Learning)        │        │
│  └──────────┬──────────┘         └──────────┬──────────┘        │
│             │                               │                    │
│             │    ┌─────────────────────┐    │                    │
│             └───►│  Security           │◄───┘                    │
│                  │  Orchestrator Agent │                         │
│                  │  (Coordination)     │                         │
│                  └──────────┬──────────┘                         │
│                             │                                     │
│             ┌───────────────┼───────────────┐                    │
│             │               │               │                    │
│  ┌──────────▼──────┐ ┌─────▼──────┐ ┌─────▼──────────┐         │
│  │  Threat         │ │  Response  │ │  Compliance    │         │
│  │  Analyzer       │ │  Executor  │ │  Monitor       │         │
│  │  (Analysis)     │ │  (Action)  │ │  (Governance)  │         │
│  └─────────────────┘ └────────────┘ └────────────────┘         │
│                                                                   │
│  ┌─────────────────────────────────────────────────────┐        │
│  │         On-Premise LLM Intelligence Layer           │        │
│  │   Ollama (Primary), vLLM, LocalAI + RAG Engine     │        │
│  │        Air-Gapped Deployment Supported             │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 1. Zero-Touch Decision Agent

**Location**: `ai_engine/security/decision_engine.py` (1,360+ LOC)

The **Zero-Touch Decision Agent** is the primary autonomous decision-making component, providing LLM-powered security analysis and response.

![Zero-Touch Decision Engine](diagrams/04_zero_touch_decision_engine.svg)

#### Architecture

```
Zero-Touch Decision Agent
│
├── Input Processing
│   ├── Security Event Ingestion
│   ├── Event Normalization
│   └── Context Enrichment
│
├── Threat Analysis Pipeline
│   ├── ML-Based Classification
│   │   ├── Anomaly Score Calculation
│   │   ├── Threat Type Detection
│   │   └── MITRE ATT&CK Mapping
│   │
│   ├── LLM Contextual Analysis
│   │   ├── Threat Narrative Generation
│   │   ├── TTP (Tactics, Techniques, Procedures) Extraction
│   │   ├── Historical Correlation
│   │   └── Confidence Scoring
│   │
│   └── Business Impact Assessment
│       ├── Financial Risk Calculation
│       ├── Operational Impact Analysis
│       └── Regulatory Implications
│
├── Response Generation
│   ├── Response Options Creation
│   │   ├── Low-Risk Actions (alerts, logging)
│   │   ├── Medium-Risk Actions (blocking, segmentation)
│   │   └── High-Risk Actions (isolation, shutdown)
│   │
│   ├── LLM Decision Making
│   │   ├── Multi-Response Evaluation
│   │   ├── Risk-Benefit Analysis
│   │   ├── Confidence Scoring (0.0-1.0)
│   │   └── Recommended Action Selection
│   │
│   └── Safety Constraints
│       ├── Confidence Threshold Checks
│       ├── Risk Category Validation
│       └── Escalation Logic
│
├── Execution Control
│   ├── Autonomous Execution (confidence > 0.95, low-risk only)
│   ├── Auto-Approval (confidence > 0.85, medium-risk)
│   ├── Human Escalation (confidence < 0.50 OR high-risk)
│   └── Fallback Handling
│
└── Learning & Metrics
    ├── Decision Outcome Tracking
    ├── Success Rate by Threat Type
    ├── Autonomous Execution Metrics
    └── Model Performance Updates
```

#### Decision Flow State Machine

```
┌───────────────┐
│ Security Event│
└───────┬───────┘
        │
        ▼
┌───────────────────────┐
│  Threat Analysis      │
│  (ML + LLM)           │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│ Business Impact       │
│ Assessment            │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│ Response Options      │
│ Generation            │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│ LLM Decision Making   │
│ (Confidence Scoring)  │
└───────┬───────────────┘
        │
        ▼
   ┌────┴────┐
   │ Safety  │
   │ Check   │
   └────┬────┘
        │
        ├─────────────────┬─────────────────┬──────────────┐
        ▼                 ▼                 ▼              ▼
  [Confidence≥0.95]  [0.85≤Conf<0.95]  [0.50≤Conf<0.85] [Conf<0.50]
  [Risk=Low]         [Risk≤Med]        [Any Risk]       [Any Risk]
        │                 │                 │              │
        ▼                 ▼                 ▼              ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────┐ ┌─────────────┐
│ AUTONOMOUS    │  │ AUTO-APPROVE │  │ ESCALATE TO  │ │ ESCALATE TO │
│ EXECUTION     │  │ & EXECUTE    │  │ SOC ANALYST  │ │ SOC MANAGER │
└───────────────┘  └──────────────┘  └──────────────┘ └─────────────┘
        │                 │                 │              │
        └─────────────────┴─────────────────┴──────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Track Outcome   │
                    │  Update Metrics  │
                    └──────────────────┘
```

#### Confidence Thresholds & Risk Matrix

| Confidence Score | Low Risk | Medium Risk | High Risk |
|------------------|----------|-------------|-----------|
| **0.95 - 1.00** | ✅ Auto Execute | ⚠️ Auto-Approve | ❌ Escalate |
| **0.85 - 0.95** | ✅ Auto Execute | ⚠️ Auto-Approve | ❌ Escalate |
| **0.50 - 0.85** | ⚠️ Auto-Approve | ⚠️ Escalate | ❌ Escalate |
| **0.00 - 0.50** | ❌ Escalate | ❌ Escalate | ❌ Escalate |

**Risk Categories:**
- **Low Risk (0.0-0.3)**: Alert generation, log retention, monitoring
- **Medium Risk (0.3-0.7)**: IP blocking, rate limiting, network segmentation
- **High Risk (0.7-1.0)**: System isolation, service shutdown, credential revocation

### 2. Protocol Discovery Agent

**Location**: `ai_engine/discovery/protocol_discovery_orchestrator.py` (885+ LOC)

Autonomous multi-phase protocol learning system that discovers and understands proprietary protocols without human intervention.

![Protocol Discovery Pipeline](diagrams/03_protocol_discovery_pipeline.svg)

#### Agent Architecture

```
Protocol Discovery Agent
│
├── Discovery Pipeline (5 Phases)
│   │
│   ├── Phase 1: Statistical Analysis
│   │   ├── Entropy Calculation
│   │   ├── Byte Frequency Analysis
│   │   ├── Pattern Detection
│   │   └── Preliminary Classification
│   │
│   ├── Phase 2: Protocol Classification
│   │   ├── ML Model Inference
│   │   ├── Feature Extraction
│   │   ├── Confidence Scoring
│   │   └── Protocol Type Identification
│   │
│   ├── Phase 3: Grammar Learning
│   │   ├── PCFG Rule Extraction
│   │   ├── Transformer-based Learning
│   │   ├── Structure Discovery
│   │   └── Grammar Validation
│   │
│   ├── Phase 4: Parser Generation
│   │   ├── Parser Template Selection
│   │   ├── Code Generation
│   │   ├── Validation Testing
│   │   └── Performance Optimization
│   │
│   └── Phase 5: Adaptive Learning (Background)
│       ├── Continuous Model Training
│       ├── Success Rate Tracking
│       ├── Grammar Refinement
│       └── Protocol Profile Updates
│
├── Intelligent Caching
│   ├── Discovery Cache (10,000 items)
│   ├── Cache Hit Optimization
│   └── TTL Management
│
├── Protocol Profiling
│   ├── Learned Protocol Database
│   ├── Field Mappings
│   ├── Grammar Rules
│   └── Parser Artifacts
│
└── Orchestration Logic
    ├── Parallel Processing (Semaphore: 10 concurrent)
    ├── Request Queuing
    ├── Error Recovery
    └── Performance Monitoring
```

#### Discovery Flow

```
Raw Protocol Data
      │
      ▼
┌─────────────────┐
│ Statistical     │
│ Analysis        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│ Classification  │─────►│ Check Cache  │─►[Cache Hit]──►Return Result
└────────┬────────┘      └──────────────┘
         │                                        ▲
         │[Cache Miss]                            │
         ▼                                        │
┌─────────────────┐                               │
│ Grammar         │                               │
│ Learning        │                               │
└────────┬────────┘                               │
         │                                        │
         ▼                                        │
┌─────────────────┐                               │
│ Parser          │                               │
│ Generation      │                               │
└────────┬────────┘                               │
         │                                        │
         ▼                                        │
┌─────────────────┐                               │
│ Validation      │                               │
└────────┬────────┘                               │
         │                                        │
         ▼                                        │
┌─────────────────┐                               │
│ Store Profile   │───────────────────────────────┘
│ Update Cache    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Background      │
│ Adaptive Learn  │
└─────────────────┘
```

### 3. Security Orchestrator Agent

**Location**: `ai_engine/security/security_service.py` (500+ LOC)

Central coordination agent managing security incident lifecycle and component integration.

#### Architecture

```
Security Orchestrator Agent
│
├── Incident Management
│   ├── Active Incidents Tracking (max 50 concurrent)
│   ├── Incident State Machine
│   ├── Automatic Cleanup (24-hour retention)
│   └── Priority Queue Management
│
├── Component Coordination
│   ├── Threat Analyzer Integration
│   ├── Decision Engine Integration
│   ├── Response Executor Integration
│   └── Compliance Reporter Integration
│
├── Workflow Orchestration
│   ├── Event Ingestion
│   ├── Threat Analysis Trigger
│   ├── Decision Request
│   ├── Response Execution
│   └── Outcome Tracking
│
├── Legacy System Awareness
│   ├── Protocol-Specific Handling
│   ├── Maintenance Window Checks
│   ├── Criticality-Based Routing
│   └── Dependency Tracking
│
└── Performance Metrics
    ├── Incident Processing Time
    ├── Component Response Times
    ├── Success/Failure Rates
    └── Prometheus Metrics Export
```

### 4. LLM Intelligence Layer

The LLM layer provides contextual reasoning capabilities across all agents.

#### Unified LLM Service

**Location**: `ai_engine/llm/unified_llm_service.py` (400+ LOC)

**Architecture Philosophy**: On-premise first for enterprise security and data sovereignty.

```
Unified LLM Service (On-Premise First)
│
├── Provider Management (Priority Order)
│   │
│   ├── Tier 1: On-Premise (Default, Air-Gapped Support)
│   │   ├── Ollama (PRIMARY)
│   │   │   ├── Llama 3.2 (8B, 70B)
│   │   │   ├── Mixtral 8x7B
│   │   │   ├── Qwen2.5 (7B, 14B, 32B)
│   │   │   ├── Phi-3 (Microsoft, lightweight)
│   │   │   └── Custom Fine-tuned Models
│   │   │
│   │   ├── vLLM (High-Performance Inference)
│   │   │   ├── GPU Optimization
│   │   │   ├── Tensor Parallelism
│   │   │   └── Continuous Batching
│   │   │
│   │   └── LocalAI (OpenAI-Compatible API)
│   │       ├── ggml/gguf model support
│   │       └── Drop-in replacement for OpenAI
│   │
│   ├── Tier 2: Cloud Providers (OPTIONAL, Disabled by Default)
│   │   ├── Anthropic Claude (requires API key + internet)
│   │   ├── OpenAI GPT-4 (requires API key + internet)
│   │   └── ⚠️ Requires explicit configuration to enable
│   │
│   └── Intelligent Fallback Chain (Configurable)
│       ├── Ollama → vLLM → LocalAI (default)
│       ├── On-Premise Only Mode (air-gapped)
│       └── Hybrid Mode (cloud fallback if enabled)
│
├── Enterprise Security Features
│   ├── Air-Gapped Deployment Support
│   │   ├── 100% offline operation
│   │   ├── No external network calls
│   │   ├── Local model weight storage
│   │   └── Zero data exfiltration
│   │
│   ├── Data Sovereignty
│   │   ├── All inference happens on-premise
│   │   ├── Sensitive data never leaves infrastructure
│   │   ├── Compliance with data residency laws
│   │   └── Full audit trail of all LLM requests
│   │
│   └── Model Customization
│       ├── Fine-tune on proprietary security data
│       ├── Domain-specific threat intelligence
│       ├── Organization-specific protocols
│       └── Custom response templates
│
├── Request Processing
│   ├── System Prompt Injection
│   ├── Context Window Management (up to 128k tokens)
│   ├── Temperature Control (0.0-1.0)
│   ├── Token Limit Enforcement
│   └── Structured Output Parsing (JSON, YAML)
│
├── Feature-Domain Prompting
│   ├── Threat Analysis Prompts
│   ├── Decision Making Prompts
│   ├── Impact Assessment Prompts
│   ├── Response Generation Prompts
│   └── Protocol Understanding Prompts
│
├── Monitoring & Control
│   ├── Token Usage Tracking
│   ├── Inference Latency Monitoring
│   ├── GPU Utilization (for on-premise)
│   ├── Error Rate Tracking
│   ├── Model Performance Metrics
│   └── Prometheus Metrics Export
│
└── Streaming Support
    ├── Real-time Response Generation
    ├── Chunk Processing
    ├── Early Termination
    └── Server-Sent Events (SSE)
```

**Deployment Models:**

1. **Air-Gapped (Maximum Security)**
   - Ollama only, no internet connectivity
   - Pre-downloaded model weights
   - Ideal for: Government, defense, critical infrastructure

2. **On-Premise (Standard Enterprise)**
   - Ollama + vLLM for high performance
   - Optional cloud fallback (disabled by default)
   - Ideal for: Banks, healthcare, enterprises

3. **Hybrid (Flexible)**
   - On-premise primary, cloud fallback enabled
   - Configurable per use case
   - Ideal for: Development, testing, non-critical workloads

**Security & Compliance Benefits:**

| Feature | On-Premise (Ollama) | Cloud LLMs (GPT-4/Claude) |
|---------|---------------------|---------------------------|
| Data Exfiltration Risk | ✅ Zero | ❌ High (data leaves premises) |
| Internet Required | ✅ No (air-gapped) | ❌ Yes (always) |
| Data Sovereignty | ✅ Full control | ❌ Third-party controlled |
| API Keys/Credentials | ✅ Not needed | ❌ Required |
| Compliance (GDPR, HIPAA) | ✅ Full compliance | ⚠️ Requires BAA/DPA |
| Inference Latency | ✅ <100ms (local) | ❌ 500-2000ms (network) |
| Cost per Request | ✅ $0 (hardware only) | ❌ $0.001-$0.03 |
| Model Customization | ✅ Full fine-tuning | ⚠️ Limited |
| Government/Defense Use | ✅ Approved | ❌ Often prohibited |
| Supply Chain Risk | ✅ Minimal | ❌ Third-party dependency |

**Recommended On-Premise Models:**

```
Security Use Case → Recommended Model
│
├── High-Speed Threat Analysis
│   └── Llama 3.2 8B (fast, efficient, 8GB VRAM)
│
├── Critical Decision Making
│   └── Llama 3.2 70B (high accuracy, 40GB VRAM)
│
├── Complex Reasoning
│   └── Mixtral 8x7B (expert-level, 24GB VRAM)
│
├── Lightweight/Edge
│   └── Phi-3 3B (Microsoft, 2GB VRAM)
│
└── Custom Security Domain
    └── Fine-tuned Llama 3.2 on proprietary threat data
```

**Hardware Recommendations:**

| Deployment Size | GPU | VRAM | Model | Throughput |
|-----------------|-----|------|-------|------------|
| Small (PoC) | NVIDIA RTX 4090 | 24GB | Llama 3.2 8B | 50 req/sec |
| Medium (Dept) | NVIDIA A100 | 40GB | Llama 3.2 70B | 20 req/sec |
| Large (Enterprise) | 4x NVIDIA A100 | 160GB | Mixtral 8x7B | 100+ req/sec |
| Edge/Branch | NVIDIA T4 | 16GB | Phi-3 3B | 100 req/sec |

#### RAG (Retrieval-Augmented Generation)

**Location**: `ai_engine/llm/rag_engine.py` (350+ LOC)

```
RAG Engine
│
├── Vector Database
│   ├── ChromaDB Integration
│   ├── Sentence Transformer Embeddings
│   ├── Similarity Search (cosine)
│   └── In-Memory Fallback
│
├── Document Management
│   ├── Protocol Knowledge Base
│   ├── Security Playbooks
│   ├── Threat Intelligence Reports
│   └── Compliance Documentation
│
├── Retrieval Pipeline
│   ├── Query Embedding
│   ├── Top-K Similarity Search (k=5)
│   ├── Relevance Filtering
│   └── Context Assembly
│
└── Generation
    ├── Context-Augmented Prompts
    ├── LLM Query with Retrieved Docs
    ├── Source Attribution
    └── Confidence Scoring
```

### 5. Agent Communication & Coordination

#### Service Integration Orchestrator

**Location**: `integration/orchestrator/service_integration.py` (450+ LOC)

```
Service Integration Orchestrator
│
├── Message Routing
│   ├── Message Queue (10,000 items)
│   ├── Priority-Based Routing
│   ├── Protocol Discovery Requests
│   ├── Security Event Routing
│   └── Compliance Check Routing
│
├── Health Monitoring
│   ├── Component Health Checks
│   ├── Health Queue (1,000 items)
│   ├── Automatic Recovery
│   └── Status Aggregation
│
├── Load Balancing
│   ├── Round-Robin Distribution
│   ├── Least-Loaded Routing
│   └── Circuit Breaker Pattern
│
└── Protocol Support
    ├── gRPC Integration
    ├── REST API Gateway
    ├── Kafka Event Streaming
    └── WebSocket Support
```

### Agent Interaction Patterns

#### Pattern 1: Security Event Response

```
Security Event
      │
      ▼
┌──────────────────────┐
│ Security Orchestrator│
└──────────┬───────────┘
           │
           ├─────────────────────────────┐
           │                             │
           ▼                             ▼
┌──────────────────┐          ┌──────────────────┐
│ Threat Analyzer  │          │ Protocol Copilot │
│ (LLM Analysis)   │          │ (Context)        │
└────────┬─────────┘          └────────┬─────────┘
         │                              │
         └──────────────┬───────────────┘
                        ▼
              ┌──────────────────┐
              │ Decision Agent   │
              │ (LLM Decision)   │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Response Executor│
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Outcome Tracking │
              └──────────────────┘
```

#### Pattern 2: Protocol Discovery

```
Unknown Protocol Data
         │
         ▼
┌──────────────────────┐
│ Service Integration  │
│ Orchestrator         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Protocol Discovery   │
│ Agent                │
└──────────┬───────────┘
           │
           ├──────────────────────┐
           │                      │
           ▼                      ▼
┌──────────────────┐   ┌──────────────────┐
│ Grammar Learner  │   │ Field Detector   │
│ (Transformer)    │   │ (BiLSTM-CRF)     │
└────────┬─────────┘   └────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
          ┌──────────────────┐
          │ Parser Generator │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Protocol Profile │
          │ Database         │
          └──────────────────┘
```

### Autonomous Decision Metrics

QBITEL agents track comprehensive metrics on autonomous operations:

```
Agent Performance Dashboard
│
├── Zero-Touch Decision Agent
│   ├── Autonomous Execution Rate: 78%
│   ├── Human Escalation Rate: 15%
│   ├── Average Confidence: 0.87
│   ├── Decision Latency P95: 180ms
│   └── Accuracy (validated): 94%
│
├── Protocol Discovery Agent
│   ├── Successful Discoveries: 1,247
│   ├── Average Discovery Time: 120ms
│   ├── Cache Hit Rate: 82%
│   ├── Parser Accuracy: 89%
│   └── Background Learning Jobs: 42/day
│
├── Security Orchestrator
│   ├── Active Incidents: 12 / 50
│   ├── Avg Incident Resolution: 4.2 min
│   ├── Auto-Resolved: 67%
│   ├── Escalated: 18%
│   └── False Positives: 3%
│
└── LLM Intelligence Layer
    ├── Total LLM Requests: 15,420/day
    ├── Avg Response Time: 850ms
    ├── Token Usage: 2.4M tokens/day
    ├── Cost per Decision: $0.012
    └── Fallback Activations: 0.8%
```

### Safety & Governance Architecture

```
Safety & Governance Layer
│
├── Explainability
│   ├── LIME Explainer (Local Interpretable Explanations)
│   ├── Decision Reasoning Logs
│   ├── Feature Importance Tracking
│   └── Human-Readable Narratives
│
├── Audit Trail
│   ├── Complete Decision History
│   ├── Input/Output Logging
│   ├── Timestamp & User Tracking
│   ├── Compliance Event Logging
│   └── Tamper-Proof Storage
│
├── Drift Detection
│   ├── Model Performance Monitoring
│   ├── Distribution Shift Detection
│   ├── Accuracy Degradation Alerts
│   └── Automatic Retraining Triggers
│
├── Compliance Automation
│   ├── SOC 2 Control Validation
│   ├── GDPR Privacy Checks
│   ├── PCI-DSS Security Validation
│   └── Automated Reporting
│
└── Safety Constraints
    ├── Confidence Threshold Enforcement
    ├── Risk Category Validation
    ├── Human-in-the-Loop Escalation
    ├── Rollback Mechanisms
    └── Kill Switch (Emergency Stop)
```

## Core Components

### 1. AI Engine

The AI Engine provides intelligent protocol discovery, field detection, and anomaly detection capabilities.

#### Architecture

```
AI Engine
├── Protocol Discovery
│   ├── PCFG Inference Engine
│   ├── Grammar Learner (Transformer-based)
│   ├── Parser Generator
│   └── Protocol Classifier
│
├── Field Detection
│   ├── BiLSTM-CRF Model
│   ├── IOB Tagging
│   └── Semantic Classifier
│
├── Anomaly Detection
│   ├── Isolation Forest
│   ├── LSTM Detector
│   ├── VAE (Variational Autoencoder)
│   └── Ensemble Detector
│
├── Feature Engineering
│   ├── Statistical Features
│   ├── Structural Analysis
│   └── Contextual Features
│
└── Model Management
    ├── MLflow Registry
    ├── Version Control
    └── A/B Testing
```

#### Key Algorithms

**PCFG Inference**: Learns grammar rules from protocol samples
- Input: Binary protocol data
- Output: Probabilistic context-free grammar
- Algorithm: Iterative rule extraction with statistical validation

**BiLSTM-CRF**: Sequence labeling for field boundaries
- Architecture: Bidirectional LSTM + Conditional Random Fields
- Tagging: IOB (Inside-Outside-Begin) scheme
- Accuracy: 89%+ on production protocols

**VAE Anomaly Detection**: Reconstruction-based anomaly detection
- Encoder: 3-layer neural network
- Latent space: 64 dimensions
- Decoder: 3-layer neural network
- Threshold: Dynamic based on reconstruction error distribution

### 2. Cloud-Native Security

Cloud-native components provide Kubernetes-native security, service mesh integration, and runtime protection.

#### Service Mesh Integration

```
Service Mesh Components
│
├── Istio Integration
│   ├── Quantum Certificate Manager (550 LOC)
│   │   ├── Kyber-1024 Key Generation
│   │   ├── Dilithium-5 Signatures
│   │   ├── Certificate Rotation (90-day)
│   │   └── NIST Level 5 Compliance
│   │
│   ├── Sidecar Injector (450 LOC)
│   │   ├── Automatic Injection
│   │   ├── 10,000+ Pods Support
│   │   └── Quantum-Safe Sidecars
│   │
│   ├── mTLS Configurator (400 LOC)
│   │   ├── Zero-Trust mTLS
│   │   └── Quantum-Safe Transport
│   │
│   └── Mesh Policy Manager (470 LOC)
│       ├── Traffic Control
│       └── Policy Enforcement
│
└── Envoy Proxy Integration
    ├── xDS Server (650 LOC)
    │   ├── gRPC Server (grpcio)
    │   ├── ADS (Aggregated Discovery Service)
    │   ├── Bidirectional Streaming
    │   └── Real-time Configuration
    │
    ├── Traffic Encryption (700 LOC)
    │   ├── AES-256-GCM
    │   ├── Perfect Forward Secrecy
    │   └── <1ms Overhead
    │
    ├── Policy Engine (280 LOC)
    │   ├── Rate Limiting
    │   ├── Circuit Breakers
    │   └── Traffic Policies
    │
    └── Observability (170 LOC)
        ├── Prometheus Metrics
        └── Jaeger Tracing
```

#### Container Security

```
Container Security Suite
│
├── Vulnerability Scanner (600 LOC)
│   ├── Trivy Integration
│   ├── CVE Detection
│   ├── Quantum-Vulnerable Crypto Detection
│   └── 1,000+ Images/Hour
│
├── Admission Webhook Server (350 LOC)
│   ├── ValidatingWebhookConfiguration
│   ├── Image Signature Verification
│   ├── Registry Whitelist/Blacklist
│   └── Privileged Container Blocking
│
├── Image Signer (550 LOC)
│   ├── Dilithium-5 Signatures
│   └── Production Container Signing
│
└── eBPF Runtime Monitor (800 LOC)
    ├── Process Execution Tracking (execve)
    ├── File Access Monitoring (openat)
    ├── Network Tracking (connect)
    ├── <1% CPU Overhead
    └── Linux Kernel 4.4+
```

#### Cloud Platform Integrations

```
Cloud Platforms
│
├── AWS Security Hub (600 LOC)
│   ├── boto3 SDK Integration
│   ├── Batch Finding Import (100/batch)
│   ├── Exponential Backoff
│   └── All AWS Regions
│
├── Azure Sentinel (550 LOC)
│   ├── Azure SDK Integration
│   ├── SIEM Data Export
│   └── ARM Templates
│
└── GCP Security Command Center (500 LOC)
    ├── GCP SCC Integration
    ├── Deployment Manager
    └── Multi-Region Support
```

### 3. Event Streaming

Secure, high-throughput event streaming with message-level encryption.

```
Event Streaming Architecture
│
└── Secure Kafka Producer (700 LOC)
    ├── kafka-python KafkaProducer
    ├── AES-256-GCM Message Encryption
    ├── Compression (snappy)
    ├── Retry Logic with Backoff
    ├── 100,000+ Messages/Sec
    └── Production Config (acks='all', retries=3)
```

## Security Architecture

### Post-Quantum Cryptography

QBITEL implements NIST-approved post-quantum cryptography algorithms.

![Post-Quantum Cryptography](diagrams/06_quantum_cryptography.svg)

#### Kyber KEM (Key Encapsulation Mechanism)

```
Kyber Implementation
│
├── Kyber-1024 (NIST Level 5)
│   ├── Public Key: 1568 bytes
│   ├── Private Key: 3168 bytes
│   ├── Ciphertext: 1568 bytes
│   ├── Shared Secret: 32 bytes
│   └── Security Level: ~256-bit quantum security
│
├── Kyber-768 (NIST Level 3)
│   └── ~192-bit quantum security
│
└── Kyber-512 (NIST Level 1)
    └── ~128-bit quantum security
```

#### Dilithium Signatures

```
Dilithium Implementation
│
├── Dilithium-5 (NIST Level 5)
│   ├── Public Key: 2592 bytes
│   ├── Private Key: 4864 bytes
│   ├── Signature: ~4595 bytes
│   └── Security Level: ~256-bit quantum security
│
├── Dilithium-3 (NIST Level 3)
│   └── ~192-bit quantum security
│
└── Dilithium-2 (NIST Level 1)
    └── ~128-bit quantum security
```

#### Hybrid Encryption

```
Encryption Flow
│
1. Key Encapsulation
   ├── Generate ephemeral key pair (Kyber-1024)
   ├── Encapsulate shared secret
   └── Transmit ciphertext
   │
2. Symmetric Encryption
   ├── Derive AES-256 key from shared secret
   ├── Encrypt data with AES-256-GCM
   ├── Generate authentication tag (128-bit)
   └── Include IV/nonce
   │
3. Digital Signature
   ├── Sign encrypted message (Dilithium-5)
   ├── Include signature in message
   └── Verify on receiver side
```

### Security Properties

- **Confidentiality**: AES-256-GCM encryption
- **Integrity**: AEAD authentication tags
- **Authentication**: Dilithium-5 digital signatures
- **Forward Secrecy**: Ephemeral Kyber keys
- **Quantum Resistance**: NIST Level 5 algorithms

## Data Flow

### Protocol Discovery Flow

```
1. Data Ingestion
   ├── Network TAP/Mirror → Raw Packets
   ├── Packet Capture → Binary Data
   └── Preprocessing → Normalized Format
   │
2. Feature Extraction
   ├── Statistical Features (entropy, frequency)
   ├── Structural Features (length, patterns)
   └── Contextual Features (n-grams, position)
   │
3. Protocol Discovery
   ├── PCFG Inference → Grammar Rules
   ├── Classification → Protocol Type
   └── Confidence Scoring → Validation
   │
4. Field Detection
   ├── BiLSTM-CRF → Field Boundaries
   ├── Semantic Classification → Field Types
   └── Validation → Accuracy Check
   │
5. Model Output
   ├── Protocol Grammar
   ├── Field Mappings
   ├── Confidence Scores
   └── Metadata
```

### Secure Communication Flow

```
1. Client Request
   ├── Original Protocol Data
   └── Legacy System Communication
   │
2. QBITEL Interception
   ├── Protocol Recognition
   ├── Data Extraction
   └── Validation
   │
3. Quantum-Safe Protection
   ├── Kyber-1024 Key Exchange
   ├── AES-256-GCM Encryption
   ├── Dilithium-5 Signature
   └── Secure Transmission
   │
4. Secure Transport
   ├── Quantum-Safe Channel
   ├── Perfect Forward Secrecy
   └── Authenticated Delivery
   │
5. Decryption & Delivery
   ├── Signature Verification
   ├── Decryption
   ├── Protocol Translation
   └── Delivery to Destination
```

## Deployment Architecture

### Kubernetes Architecture

```
Kubernetes Cluster
│
├── Namespace: qbitel-service-mesh
│   ├── xDS Server Deployment (3 replicas)
│   │   ├── Pod Disruption Budget
│   │   ├── Anti-Affinity Rules
│   │   ├── Resource Limits (250m CPU, 512Mi memory)
│   │   └── Health Checks (liveness/readiness)
│   │
│   ├── Service Mesh Components
│   │   ├── Istio Control Plane
│   │   ├── Envoy Sidecars
│   │   └── Quantum Certificate Manager
│   │
│   └── RBAC Configuration
│       ├── Service Accounts
│       ├── Roles
│       └── RoleBindings
│
├── Namespace: qbitel-container-security
│   ├── Admission Webhook (3 replicas)
│   │   ├── ValidatingWebhookConfiguration
│   │   ├── TLS Certificates
│   │   └── Security Policies ConfigMap
│   │
│   ├── Vulnerability Scanner
│   └── Image Signer
│
└── Observability
    ├── Prometheus (metrics)
    ├── Grafana (dashboards)
    └── Jaeger (tracing)
```

### High Availability

- **Replication**: 3 replicas for all critical services
- **Pod Disruption Budgets**: Ensure minimum availability during updates
- **Anti-Affinity**: Distribute pods across nodes
- **Health Checks**: Liveness and readiness probes
- **Auto-Scaling**: HPA based on CPU/memory metrics

### Security Hardening

- **Non-Root Containers**: All containers run as UID 1000
- **Read-Only Filesystems**: Immutable container filesystems
- **Security Contexts**: Drop all capabilities, no privilege escalation
- **Network Policies**: Restrict pod-to-pod communication
- **RBAC**: Least-privilege access control

## Rust Data Plane

**Location**: `rust/dataplane/`

The Rust data plane provides wire-speed cryptographic operations and protocol parsing for performance-critical paths.

### PQC-TLS Implementation

**Location**: `rust/dataplane/crates/pqc_tls/`

```
Rust Data Plane
│
├── PQC-TLS Library (pqc_tls crate)
│   ├── ML-KEM (Kyber) Key Encapsulation
│   │   ├── Kyber-512, Kyber-768, Kyber-1024
│   │   └── Hybrid modes (X25519 + Kyber)
│   │
│   ├── ML-DSA (Dilithium) Digital Signatures
│   │   ├── Dilithium-2, Dilithium-3, Dilithium-5
│   │   └── Certificate signing and verification
│   │
│   ├── Falcon-1024 Signatures
│   ├── SLH-DSA (SPHINCS+) Stateless Signatures
│   │
│   ├── Domain-Specific PQC Profiles
│   │   ├── Banking (SWIFT/SEPA constraints)
│   │   ├── Healthcare (pacemaker/insulin pump power limits)
│   │   ├── Automotive (V2X/IEEE 1609.2 latency)
│   │   ├── Aviation (ADS-B bandwidth constraints)
│   │   └── Industrial (IEC 62351/SCADA cycles)
│   │
│   └── TLS 1.3 Integration
│       ├── Post-quantum key exchange
│       ├── Hybrid handshake
│       └── Session resumption
│
├── Protocol Parser
│   ├── Wire-format parsing
│   ├── Binary protocol handling
│   └── Zero-copy deserialization
│
└── Performance Targets
    ├── Key generation: <1ms
    ├── Encapsulation: <0.5ms
    ├── Signature: <2ms
    └── TLS handshake: <5ms
```

## Go Control Plane

**Location**: `go/`

The Go layer provides service orchestration, management APIs, and edge device agents.

```
Go Services
│
├── Control Plane (go/controlplane/)
│   ├── Service Orchestration
│   │   ├── Component lifecycle management
│   │   ├── Health monitoring
│   │   └── Configuration distribution
│   │
│   ├── Policy Engine
│   │   ├── Security policy evaluation
│   │   ├── Protocol routing rules
│   │   └── Rate limiting and quotas
│   │
│   └── gRPC API
│       ├── Internal service communication
│       └── Streaming updates
│
├── Management API (go/mgmtapi/)
│   ├── REST API Gateway
│   │   ├── Protocol management endpoints
│   │   ├── System configuration
│   │   ├── Health and status
│   │   └── Metrics aggregation
│   │
│   └── Authentication & Authorization
│       ├── JWT validation
│       ├── RBAC enforcement
│       └── API key management
│
└── Device Agent (go/agents/device-agent/)
    ├── Edge Protocol Collection
    ├── Local PQC operations
    ├── Heartbeat and telemetry
    └── Offline-capable operation
```

## UI Console

**Location**: `ui/console/`

React/TypeScript admin dashboard providing visibility into all platform operations.

```
UI Console (React + TypeScript)
│
├── Dashboard
│   ├── System health overview
│   ├── Protocol discovery status
│   ├── Security incident timeline
│   └── Compliance scorecard
│
├── Protocol Management
│   ├── Discovery results browser
│   ├── Grammar visualization
│   ├── Field mapping editor
│   └── Parser testing sandbox
│
├── Protocol Copilot
│   ├── Natural language protocol queries
│   ├── Threat assessment interface
│   └── Conversation history
│
├── Marketplace Interface
│   ├── Protocol browsing and search
│   ├── Publishing workflow
│   └── License management
│
├── Security Operations
│   ├── Incident management
│   ├── Decision engine dashboard
│   ├── Threat analysis views
│   └── Audit log browser
│
└── Enterprise Features
    ├── Multi-tenant management
    ├── OIDC/SSO integration
    ├── Role-based access control
    └── Custom branding
```

## Legacy System Whisperer

**Location**: `ai_engine/legacy/`

Deep analysis engine for understanding and documenting undocumented legacy systems.

```
Legacy System Whisperer
│
├── COBOL Analysis
│   ├── Copybook Parser
│   │   ├── Field extraction (PIC clauses)
│   │   ├── Record layout mapping
│   │   ├── REDEFINES handling
│   │   └── Level-number hierarchy
│   │
│   ├── JCL Analyzer
│   │   ├── Job flow mapping
│   │   ├── Dataset dependency graphs
│   │   └── Step-level analysis
│   │
│   └── Business Rule Extractor
│       ├── IF/EVALUATE detection
│       ├── PERFORM flow analysis
│       └── Natural language documentation
│
├── Mainframe Discovery
│   ├── Dataset catalog scanning
│   ├── Program call tree analysis
│   ├── Data flow mapping
│   └── Volume usage tracking
│
├── Predictive Analysis
│   ├── Failure prediction (aging hardware)
│   ├── Capacity forecasting
│   ├── Modernization readiness scoring
│   └── Risk assessment
│
└── Knowledge Retention
    ├── System documentation generation
    ├── Tribal knowledge capture
    ├── Version history tracking
    └── Configurable retention (default 5 years)
```

## Protocol Marketplace

**Location**: `ai_engine/marketplace/`

Community-driven platform for sharing, discovering, and commercializing protocol definitions.

```
Protocol Marketplace
│
├── Protocol Registry
│   ├── Metadata Management
│   │   ├── Name, version, description
│   │   ├── Licensing (open/commercial)
│   │   ├── Industry tags
│   │   └── Compatibility matrix
│   │
│   ├── Validation Pipeline
│   │   ├── Schema validation
│   │   ├── Security scanning
│   │   ├── Performance testing
│   │   └── Compatibility checks
│   │
│   └── Version Management
│       ├── Semantic versioning
│       ├── Dependency resolution
│       └── Deprecation tracking
│
├── Commerce (Stripe Connect)
│   ├── Publisher accounts
│   ├── Subscription management
│   ├── Usage-based billing
│   └── Revenue sharing
│
├── Storage (S3)
│   ├── Protocol definition files
│   ├── Grammar artifacts
│   ├── Parser binaries
│   └── Documentation assets
│
└── Frontend
    ├── Search and discovery
    ├── Publishing wizard
    ├── Review and ratings
    └── License management
```

## Technology Stack

### Programming Languages

- **Python 3.10+**: AI Engine, cloud-native components, compliance
- **Rust**: High-performance data plane, PQC-TLS implementation
- **Go**: Control plane, management API, device agents
- **TypeScript/React**: Admin console UI

### AI/ML Frameworks

- **PyTorch**: Deep learning models
- **Transformers**: Protocol grammar learning
- **Scikit-learn**: Classical ML algorithms
- **SHAP/LIME**: Model explainability

### Cryptography

- **kyber-py**: Kyber KEM implementation (Python)
- **dilithium-py**: Dilithium signatures (Python)
- **liboqs-python**: Comprehensive PQC library (Python)
- **oqs-rs / liboqs**: PQC implementation (Rust)
- **cryptography**: AES-256-GCM, TLS

### Cloud-Native

- **Kubernetes**: Container orchestration
- **Istio**: Service mesh
- **Envoy**: L7 proxy
- **eBPF/BCC**: Runtime monitoring (Linux)
- **Trivy**: Vulnerability scanning

### Messaging & Streaming

- **Kafka**: Event streaming
- **gRPC**: High-performance RPC
- **Protocol Buffers**: Serialization

### Observability

- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **OpenTelemetry**: Distributed tracing
- **Jaeger**: Trace visualization

### Cloud SDKs

- **boto3**: AWS integration
- **azure-sdk**: Azure integration
- **google-cloud**: GCP integration

## Performance Characteristics

### Throughput

| Component | Metric | Value |
|-----------|--------|-------|
| Kafka Producer | Messages/sec | 100,000+ |
| xDS Server | Proxies | 1,000+ |
| Traffic Encryption | Requests/sec | 10,000+ |
| eBPF Monitor | Containers | 10,000+ |
| Admission Webhook | Pods/sec | 1,000+ |

### Latency

| Operation | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| Protocol Discovery | 80ms | 150ms | 200ms |
| Field Detection | 60ms | 120ms | 180ms |
| Anomaly Detection | 100ms | 180ms | 250ms |
| Encryption | <1ms | <1ms | <2ms |
| xDS Config Update | 5ms | 10ms | 15ms |

### Resource Usage

| Component | CPU | Memory | Notes |
|-----------|-----|--------|-------|
| AI Engine | 2+ cores | 4GB+ | 8GB+ with GPU |
| xDS Server | 250m | 512Mi | Per replica |
| Admission Webhook | 200m | 256Mi | Per replica |
| eBPF Monitor | <1% | 100Mi | Per node |
| Kafka Producer | 500m | 512Mi | Handles 100K msg/s |

## Scalability

### Horizontal Scaling

- **AI Engine**: Stateless, scales with load balancer
- **xDS Server**: Scales with number of proxies
- **Admission Webhook**: Scales with pod creation rate
- **Kafka**: Partition-based scaling

### Vertical Scaling

- **GPU Acceleration**: Optional for AI training
- **Memory**: Adjustable for model size and batch processing
- **CPU**: Scales with concurrent requests

### Data Volume

- **Protocol Data**: Handles TB/day ingestion
- **Event Streaming**: 100M+ events/day
- **Metrics**: Time-series database with retention policies
- **Logs**: Structured logging with aggregation

## Disaster Recovery

### Backup Strategy

- **Model Registry**: MLflow with versioned models
- **Configuration**: GitOps with version control
- **Certificates**: Automated rotation and backup
- **Data**: Incremental backups with retention

### Recovery Procedures

- **RTO (Recovery Time Objective)**: <30 minutes
- **RPO (Recovery Point Objective)**: <5 minutes
- **Failover**: Automatic with health checks
- **Rollback**: Versioned deployments with rollback capability

---

**Last Updated**: 2025-02-08
**Version**: 2.0
**Status**: Production Architecture


---

# ============================================================
# FILE 3: docs/API.md (API Documentation)
# ============================================================

# QBITEL Bridge - API Documentation

## 🚀 **API Overview**

The QBITEL Bridge System provides both REST and Python APIs for protocol discovery, classification, and management operations.

## 📋 **Table of Contents**

- [Authentication](#authentication)
- [REST API Endpoints](#rest-api-endpoints)
- [Python API Reference](#python-api-reference)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

## 🔐 **Authentication**

### **API Key Authentication**

```http
X-API-Key: your_api_key_here
```

### **JWT Authentication**

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## 🌐 **REST API Endpoints**

### **Protocol Discovery**

#### **Analyze Traffic**
Discover protocols from network traffic data.

```http
POST /api/v1/discovery/analyze
Content-Type: application/json
X-API-Key: your_api_key

{
    "traffic_data": [
        "R0VUIC9hcGkvdXNlcnMgSFRUUC8xLjENCkhvc3Q6IGFwaS5leGFtcGxlLmNvbQ0KDQo=",
        "UE9TVCAvYXBpL2xvZ2luIEhUVFAvMS4xDQpDb250ZW50LVR5cGU6IGFwcGxpY2F0aW9uL2pzb24NCg0K"
    ],
    "options": {
        "max_protocols": 5,
        "confidence_threshold": 0.7,
        "enable_adaptive_learning": true,
        "cache_results": true
    }
}
```

**Response:**
```json
{
    "success": true,
    "request_id": "req_12345",
    "processing_time_ms": 245,
    "cache_hit": false,
    "discovered_protocols": [
        {
            "id": "proto_001",
            "name": "http_variant_1",
            "confidence": 0.94,
            "description": "HTTP-like protocol with custom headers",
            "statistics": {
                "total_messages": 100,
                "avg_message_length": 256.7,
                "entropy": 6.24,
                "binary_ratio": 0.12
            },
            "grammar": {
                "start_symbol": "HTTP_MESSAGE",
                "rules": [
                    {
                        "lhs": "HTTP_MESSAGE",
                        "rhs": ["METHOD", "PATH", "VERSION", "HEADERS", "BODY"],
                        "probability": 1.0
                    }
                ]
            },
            "parser": {
                "id": "parser_001",
                "protocol_name": "http_variant_1",
                "parse_success_rate": 0.98,
                "performance_ms": 2.3
            },
            "validation_rules": [
                {
                    "name": "method_validation",
                    "type": "regex",
                    "pattern": "^(GET|POST|PUT|DELETE)\\s",
                    "confidence": 0.95
                }
            ]
        }
    ],
    "metadata": {
        "model_versions": {
            "cnn": "v1.2.3",
            "lstm": "v1.2.1", 
            "random_forest": "v1.1.8"
        },
        "processing_stats": {
            "statistical_analysis_ms": 45,
            "grammar_learning_ms": 120,
            "parser_generation_ms": 35,
            "classification_ms": 25,
            "validation_ms": 20
        }
    }
}
```

#### **Classify Message**
Classify a single message against known protocols.

```http
POST /api/v1/discovery/classify
Content-Type: application/json

{
    "message": "R0VUIC9hcGkvdXNlcnMgSFRUUC8xLjE=",
    "options": {
        "return_probabilities": true,
        "confidence_threshold": 0.5
    }
}
```

**Response:**
```json
{
    "success": true,
    "classification": {
        "protocol": "http",
        "confidence": 0.97,
        "class_probabilities": {
            "http": 0.97,
            "custom_protocol_1": 0.02,
            "unknown": 0.01
        }
    },
    "parsed_fields": {
        "method": "GET",
        "path": "/api/users",
        "version": "HTTP/1.1"
    },
    "validation": {
        "is_valid": true,
        "passed_rules": ["method_check", "version_check"],
        "failed_rules": [],
        "validation_score": 1.0
    }
}
```

### **Protocol Management**

#### **List Protocols**
Get list of discovered protocols.

```http
GET /api/v1/protocols?limit=10&offset=0&sort=confidence
```

**Response:**
```json
{
    "success": true,
    "protocols": [
        {
            "id": "proto_001",
            "name": "http_variant_1",
            "confidence": 0.94,
            "created_at": "2024-01-15T10:30:00Z",
            "last_updated": "2024-01-15T11:45:00Z",
            "message_count": 1547,
            "success_rate": 0.98
        }
    ],
    "pagination": {
        "total": 23,
        "limit": 10,
        "offset": 0,
        "has_more": true
    }
}
```

#### **Get Protocol Details**
Get detailed information about a specific protocol.

```http
GET /api/v1/protocols/{protocol_id}
```

**Response:**
```json
{
    "success": true,
    "protocol": {
        "id": "proto_001",
        "name": "http_variant_1",
        "confidence": 0.94,
        "description": "HTTP-like protocol with custom headers",
        "grammar": {...},
        "parser": {...},
        "validation_rules": [...],
        "statistics": {...},
        "metadata": {...}
    }
}
```

#### **Update Protocol**
Update protocol information or retrain with new data.

```http
PUT /api/v1/protocols/{protocol_id}
Content-Type: application/json

{
    "name": "updated_protocol_name",
    "description": "Updated description",
    "retrain": true,
    "training_data": ["base64_message_1", "base64_message_2"]
}
```

#### **Delete Protocol**
Delete a discovered protocol.

```http
DELETE /api/v1/protocols/{protocol_id}
```

### **System Management**

#### **Health Check**
Get system health status.

```http
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "timestamp": 1640995200,
    "uptime_seconds": 86400,
    "version": "1.0.0",
    "components": {
        "database": {
            "status": "healthy",
            "response_time_ms": 12
        },
        "redis": {
            "status": "healthy",
            "response_time_ms": 3
        },
        "ai_models": {
            "status": "healthy",
            "models_loaded": 3
        }
    },
    "metrics": {
        "requests_per_second": 124.5,
        "avg_response_time_ms": 45.2,
        "memory_usage_mb": 1024.3,
        "cpu_percent": 23.5,
        "cache_hit_rate": 0.94
    }
}
```

#### **Metrics**
Get Prometheus metrics.

```http
GET /metrics
```

**Response:**
```
# HELP protocol_discovery_requests_total Total protocol discovery requests
# TYPE protocol_discovery_requests_total counter
protocol_discovery_requests_total{status="success"} 1547
protocol_discovery_requests_total{status="error"} 23

# HELP protocol_discovery_duration_seconds Request duration in seconds
# TYPE protocol_discovery_duration_seconds histogram
protocol_discovery_duration_seconds_bucket{le="0.005"} 234
protocol_discovery_duration_seconds_bucket{le="0.01"} 567
protocol_discovery_duration_seconds_bucket{le="0.025"} 892
```

## 🐍 **Python API Reference**

### **Protocol Discovery Orchestrator**

```python
from ai_engine.discovery.protocol_discovery_orchestrator import ProtocolDiscoveryOrchestrator

# Initialize orchestrator
orchestrator = ProtocolDiscoveryOrchestrator(config)

# Discover protocols
traffic_data = [b"GET /api HTTP/1.1\r\n\r\n", b"POST /data HTTP/1.1\r\n\r\n"]
result = await orchestrator.discover_protocol(traffic_data)

# Get discovered protocols
for protocol in result.discovered_protocols:
    print(f"Found protocol: {protocol.name} (confidence: {protocol.confidence})")
```

### **Individual Components**

#### **Statistical Analyzer**

```python
from ai_engine.discovery.statistical_analyzer import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Analyze traffic patterns
pattern = await analyzer.analyze_traffic(traffic_data)
print(f"Entropy: {pattern.entropy}, Binary ratio: {pattern.binary_ratio}")

# Detect field boundaries
boundaries = await analyzer.detect_field_boundaries(messages)
for boundary in boundaries:
    print(f"Separator: {boundary.separator}, Position: {boundary.position}")
```

#### **Grammar Learner**

```python
from ai_engine.discovery.grammar_learner import GrammarLearner

learner = GrammarLearner()

# Learn PCFG from messages
grammar = await learner.learn_pcfg(messages)
print(f"Learned {len(grammar.rules)} grammar rules")

# Refine with EM algorithm
refined_grammar = await learner.refine_with_em(grammar, messages, max_iterations=50)
```

#### **Protocol Classifier**

```python
from ai_engine.discovery.protocol_classifier import ProtocolClassifier

classifier = ProtocolClassifier(config)

# Train models
training_data = {
    'http': [b"GET /", b"POST /"],
    'smtp': [b"HELO ", b"MAIL FROM:"]
}
await classifier.train_models(training_data)

# Classify message
prediction = await classifier.classify_message(b"GET /api HTTP/1.1")
print(f"Protocol: {prediction.protocol}, Confidence: {prediction.confidence}")
```

#### **Message Validator**

```python
from ai_engine.discovery.message_validator import MessageValidator, ValidationRule

validator = MessageValidator()

# Add validation rules
rule = ValidationRule(
    name="http_method",
    rule_type="regex",
    pattern=rb"^(GET|POST|PUT|DELETE)",
    required=True
)
validator.add_rule(rule)

# Validate message
result = await validator.validate_message(b"GET /api HTTP/1.1")
print(f"Valid: {result.is_valid}, Score: {result.validation_score}")
```

## 📊 **Data Models**

### **DiscoveryResult**

```python
@dataclass
class DiscoveryResult:
    success: bool
    discovered_protocols: List[DiscoveredProtocol]
    processing_time: float
    error_message: Optional[str] = None
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### **DiscoveredProtocol**

```python
@dataclass
class DiscoveredProtocol:
    name: str
    confidence: float
    grammar: Grammar
    parser: GeneratedParser
    statistics: TrafficPattern
    validation_rules: List[ValidationRule]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### **TrafficPattern**

```python
@dataclass
class TrafficPattern:
    total_messages: int
    message_lengths: List[int]
    entropy: float
    binary_ratio: float
    detected_patterns: List[DetectedPattern]
    field_boundaries: List[FieldBoundary]
    processing_time: float
```

### **Grammar**

```python
@dataclass
class Grammar:
    start_symbol: str
    rules: List[PCFGRule]
    terminals: Set[bytes]
    non_terminals: Set[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### **ValidationResult**

```python
@dataclass
class ValidationResult:
    is_valid: bool
    validation_score: float
    passed_rules: List[str]
    failed_rules: List[str]
    error_details: List[str]
    processing_time: float
```

## ⚠️ **Error Handling**

### **Error Response Format**

```json
{
    "success": false,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid input data format",
        "details": {
            "field": "traffic_data",
            "issue": "Expected base64 encoded strings"
        },
        "request_id": "req_12345",
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

### **Error Codes**

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid input data or parameters |
| `AUTHENTICATION_ERROR` | 401 | Missing or invalid API key/token |
| `AUTHORIZATION_ERROR` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Requested resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

### **Python Exceptions**

```python
from ai_engine.core.exceptions import (
    ProtocolDiscoveryError,
    ValidationError,
    ClassificationError,
    ParsingError,
    ConfigurationError
)

try:
    result = await orchestrator.discover_protocol(traffic_data)
except ValidationError as e:
    print(f"Validation failed: {e}")
except ClassificationError as e:
    print(f"Classification failed: {e}")
```

## 🚦 **Rate Limiting**

### **Rate Limits**

| Endpoint | Rate Limit | Window |
|----------|------------|--------|
| `/api/v1/discovery/analyze` | 10 requests | 1 minute |
| `/api/v1/discovery/classify` | 100 requests | 1 minute |
| `/api/v1/protocols/*` | 50 requests | 1 minute |
| `/health` | 1000 requests | 1 minute |

### **Rate Limit Headers**

```http
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1640995260
```

## 📚 **Examples**

### **Complete Protocol Discovery**

```python
import asyncio
from ai_engine.discovery.protocol_discovery_orchestrator import ProtocolDiscoveryOrchestrator
from ai_engine.core.config_manager import load_config

async def discover_protocols():
    # Load configuration
    config = load_config("config/qbitel.yaml")
    
    # Initialize orchestrator
    orchestrator = ProtocolDiscoveryOrchestrator(config.ai_engine)
    
    # Sample network traffic
    traffic_data = [
        b"GET /api/v1/users HTTP/1.1\r\nHost: api.example.com\r\nAuthorization: Bearer token123\r\n\r\n",
        b"POST /api/v1/login HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: 45\r\n\r\n{\"username\":\"alice\",\"password\":\"secret\"}",
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 123\r\n\r\n{\"users\":[{\"id\":1,\"name\":\"Alice\"}]}",
        b"CMD:CONNECT|HOST:server.example.com|PORT:443|PROTO:TLS",
        b"CMD:AUTH|USER:bob|PASS:password123|SESSION:abc456def789",
        b"RESPONSE:OK|STATUS:CONNECTED|SESSION:abc456def789|TIMEOUT:3600"
    ]
    
    # Perform discovery
    result = await orchestrator.discover_protocol(traffic_data)
    
    if result.success:
        print(f"Discovery completed in {result.processing_time:.2f}s")
        print(f"Found {len(result.discovered_protocols)} protocols:")
        
        for i, protocol in enumerate(result.discovered_protocols, 1):
            print(f"\n{i}. Protocol: {protocol.name}")
            print(f"   Confidence: {protocol.confidence:.2%}")
            print(f"   Messages analyzed: {protocol.statistics.total_messages}")
            print(f"   Average entropy: {protocol.statistics.entropy:.2f}")
            
            # Test parser
            if protocol.parser:
                test_message = traffic_data[0]  # Use first message as test
                parse_result = await protocol.parser.parse(test_message)
                if parse_result.success:
                    print(f"   Parser test: SUCCESS")
                    print(f"   Parsed fields: {list(parse_result.parsed_fields.keys())}")
                else:
                    print(f"   Parser test: FAILED - {parse_result.error}")
            
            # Show validation rules
            if protocol.validation_rules:
                print(f"   Validation rules: {len(protocol.validation_rules)}")
                for rule in protocol.validation_rules[:3]:  # Show first 3
                    print(f"     - {rule.name} ({rule.rule_type})")
    else:
        print(f"Discovery failed: {result.error_message}")

# Run the example
asyncio.run(discover_protocols())
```

### **Real-time Classification**

```python
import asyncio
from ai_engine.discovery.protocol_classifier import ProtocolClassifier
from ai_engine.monitoring.enterprise_metrics import record_metric

async def classify_stream(classifier, message_stream):
    """Classify messages from a real-time stream."""
    
    for message in message_stream:
        try:
            # Classify message
            prediction = await classifier.classify_message(message)
            
            # Record metrics
            record_metric('classification_confidence', prediction.confidence)
            record_metric('classification_latency_ms', prediction.processing_time * 1000)
            
            # Process based on protocol
            if prediction.protocol == 'http':
                await handle_http_message(message, prediction)
            elif prediction.protocol == 'custom_protocol':
                await handle_custom_message(message, prediction)
            else:
                await handle_unknown_message(message, prediction)
                
        except Exception as e:
            logger.error(f"Classification error: {e}")
            record_metric('classification_errors', 1)

async def handle_http_message(message, prediction):
    """Handle HTTP protocol messages."""
    print(f"HTTP message detected (confidence: {prediction.confidence:.2%})")
    
async def handle_custom_message(message, prediction):
    """Handle custom protocol messages.""" 
    print(f"Custom protocol message detected (confidence: {prediction.confidence:.2%})")
    
async def handle_unknown_message(message, prediction):
    """Handle unknown protocol messages."""
    print(f"Unknown protocol message (confidence: {prediction.confidence:.2%})")
```

### **Batch Processing**

```python
from ai_engine.core.performance_optimizer import BatchProcessor

async def batch_discovery_example():
    """Example of batch processing for high-throughput scenarios."""
    
    batch_processor = BatchProcessor(batch_size=1000, flush_interval=5.0)
    orchestrator = ProtocolDiscoveryOrchestrator()
    
    async def process_batch(messages):
        """Process a batch of messages."""
        result = await orchestrator.discover_protocol(messages)
        print(f"Processed batch of {len(messages)} messages")
        return result
    
    # Simulate message stream
    message_stream = generate_message_stream()  # Your message generator
    
    async for message in message_stream:
        await batch_processor.add_item('discovery', message, process_batch)
    
    # Flush any remaining messages
    await batch_processor.flush_all()
```

## 📞 **Support**

For API-related questions and issues:

- **Documentation**: [docs/README.md](README.md)
- **Issues**: [GitHub Issues](https://github.com/yazhsab/qbitel-bridge/issues)
- **API Support**: api-support@qbitel.com

---

**API Version**: v2.1.0
**Last Updated**: 2025-11-22

---

# ============================================================
# FILE 4: docs/KNOWLEDGE_BASE.md (Comprehensive Knowledge Base)
# ============================================================

# QBITEL - Comprehensive Knowledge Base

**Version**: 2.1.0
**Last Updated**: 2025-11-22
**Purpose**: Reference guide for AI coding agents and developers for all future enhancements and implementations
**Coverage**: Python, Go, Rust, TypeScript/React - Complete multi-language codebase

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Directory Structure](#3-directory-structure)
4. [Core Components](#4-core-components)
5. [API Layer](#5-api-layer)
6. [Database Models & Schemas](#6-database-models--schemas)
7. [Service Layer](#7-service-layer)
8. [Security Implementation](#8-security-implementation)
9. [AI/ML Components](#9-aiml-components)
10. [Cloud-Native Infrastructure](#10-cloud-native-infrastructure)
11. [Configuration Management](#11-configuration-management)
12. [External Integrations](#12-external-integrations)
13. [Testing Strategy](#13-testing-strategy)
14. [Deployment Guide](#14-deployment-guide)
15. [Design Patterns & Best Practices](#15-design-patterns--best-practices)
16. [Coding Standards](#16-coding-standards)
17. [Troubleshooting Guide](#17-troubleshooting-guide)
18. [Go Services](#18-go-services)
19. [Rust Dataplane](#19-rust-dataplane)
20. [Frontend Console](#20-frontend-console)

---

## 1. Project Overview

### 1.1 What is QBITEL?

QBITEL is an **enterprise-grade, quantum-safe security platform** that provides:

- **AI-Powered Protocol Discovery** - Automatically learns undocumented/proprietary protocols
- **Post-Quantum Cryptography** - NIST Level 5 compliant (Kyber-1024, Dilithium-5)
- **Agentic AI Security** - Autonomous decision-making with LLM-powered threat analysis
- **Cloud-Native Security** - Service mesh integration, container security, eBPF monitoring
- **Zero-Touch Deployment** - Protects existing systems without code changes

### 1.2 Key Statistics

| Metric | Value |
|--------|-------|
| **Python Files** | 442+ |
| **Go Files** | 15+ |
| **Rust Files** | 64+ |
| **TypeScript/React Files** | 39+ |
| **Total Lines of Code** | 250,000+ |
| **Documentation Files** | 82 markdown files |
| **Test Files** | 160+ |
| **Test Coverage** | 85% |
| **Production Readiness** | 85-90% |

### 1.3 Technology Stack

| Layer | Technologies |
|-------|--------------|
| **Backend** | Python 3.10+, FastAPI 0.104.1, gRPC |
| **AI/ML** | PyTorch 2.2.2, Scikit-learn, SHAP, LIME |
| **LLM** | Ollama (primary), Claude, GPT-4 |
| **Database** | PostgreSQL 15, Redis 7, TimescaleDB, ChromaDB |
| **Cryptography** | Kyber-1024, Dilithium-5, AES-256-GCM |
| **Container** | Docker, Kubernetes 1.24+ |
| **Service Mesh** | Istio, Envoy Proxy |
| **Monitoring** | Prometheus, OpenTelemetry, Jaeger, Sentry |
| **Cloud** | AWS, Azure, GCP SDKs |
| **Languages** | Python (primary), Go, Rust, TypeScript |
| **Go Services** | Control Plane, Management API, Device Agent |
| **Rust Crates** | PQC-TLS, I/O Engine, DPDK, DPI, K8s Operator, Protocol Adapters |
| **Frontend** | React 18, Material-UI 5, Vite, OIDC Auth |

### 1.4 License

Apache License 2.0

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           QBITEL Platform                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   REST API   │  │   gRPC API   │  │  WebSocket   │  │  Admin UI   │ │
│  │  (FastAPI)   │  │              │  │  (Copilot)   │  │   (React)   │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
│         │                 │                 │                 │        │
│         └─────────────────┼─────────────────┼─────────────────┘        │
│                           │                 │                          │
│  ┌────────────────────────┴─────────────────┴────────────────────────┐ │
│  │                        AI ENGINE CORE                             │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐       │ │
│  │  │   Protocol     │  │    Anomaly     │  │     Field      │       │ │
│  │  │   Discovery    │  │   Detection    │  │   Detection    │       │ │
│  │  │   (PCFG)       │  │   (Ensemble)   │  │  (BiLSTM-CRF)  │       │ │
│  │  └────────────────┘  └────────────────┘  └────────────────┘       │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     AGENTIC AI LAYER                             │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │  │
│  │  │  Decision  │  │  Security  │  │   Copilot  │  │    RAG     │  │  │
│  │  │   Engine   │  │ Orchestr.  │  │  (LLM UI)  │  │   Engine   │  │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    CLOUD-NATIVE SECURITY                         │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │  │
│  │  │  Service   │  │ Container  │  │   Event    │  │   Cloud    │  │  │
│  │  │   Mesh     │  │  Security  │  │ Streaming  │  │   Integr.  │  │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  PostgreSQL  │  │    Redis     │  │   Kafka      │                  │
│  │   (Data)     │  │   (Cache)    │  │  (Events)    │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Microservices Architecture

| Service | Language | Purpose | Port |
|---------|----------|---------|------|
| AI Engine | Python | Core ML/AI processing | 8000 (REST), 50051 (gRPC) |
| Control Plane | Go | Policy engine, configuration | 8080 |
| Management API | Go | Device lifecycle, attestation | 8081 |
| Device Agent | Go | On-device security agent | N/A |
| Dataplane | Rust | High-performance packet processing | N/A |
| Admin Console | TypeScript/React | Web UI dashboard | 3000 |

### 2.3 Entry Points

| Entry Point | Location | Command |
|-------------|----------|---------|
| AI Engine | `ai_engine/__main__.py` | `python -m ai_engine` |
| API Server | `ai_engine/api/server.py` | Automatic via __main__ |
| Control Plane | `go/controlplane/cmd/controlplane/main.go` | `go run main.go` |
| Management API | `go/mgmtapi/cmd/mgmtapi/main.go` | `go run main.go` |
| Dataplane | `rust/dataplane/src/main.rs` | `cargo run` |
| Console UI | `ui/console/src/main.tsx` | `npm run dev` |

---

## 3. Directory Structure

```
qbitel/
├── ai_engine/                    # Core AI/ML engine (Python)
│   ├── __main__.py              # Application entry point
│   ├── api/                     # REST & gRPC API layer
│   │   ├── rest.py              # FastAPI application
│   │   ├── server.py            # Server management
│   │   ├── auth.py              # Authentication
│   │   ├── middleware.py        # Request middleware
│   │   ├── rate_limiter.py      # Rate limiting
│   │   ├── input_validation.py  # Input validation
│   │   └── *_endpoints.py       # Feature-specific endpoints
│   ├── core/                    # Core infrastructure
│   │   ├── engine.py            # Main AI engine orchestrator
│   │   ├── config.py            # Configuration management
│   │   ├── database_manager.py  # Database connection pooling
│   │   ├── exceptions.py        # Custom exceptions
│   │   └── structured_logging.py # Logging infrastructure
│   ├── discovery/               # Protocol discovery
│   │   ├── pcfg_inference.py    # Grammar inference
│   │   ├── grammar_learner.py   # Grammar learning
│   │   ├── parser_generator.py  # Parser generation
│   │   └── protocol_discovery_orchestrator.py
│   ├── detection/               # Field detection
│   │   ├── field_detector.py    # BiLSTM-CRF detector
│   │   └── anomaly_detector.py  # Anomaly detection
│   ├── anomaly/                 # Anomaly detection ensemble
│   │   ├── ensemble_detector.py # Ensemble methods
│   │   ├── lstm_detector.py     # LSTM-based detection
│   │   ├── vae_detector.py      # VAE-based detection
│   │   └── isolation_forest.py  # Isolation Forest
│   ├── security/                # Security orchestration
│   │   ├── decision_engine.py   # Zero-touch decisions
│   │   ├── security_service.py  # Security orchestrator
│   │   ├── threat_analyzer.py   # Threat analysis
│   │   └── integrations/        # Cloud security integrations
│   ├── llm/                     # LLM integration
│   │   ├── unified_llm_service.py # Multi-provider LLM
│   │   ├── rag_engine.py        # RAG for context
│   │   └── *_client.py          # Provider clients
│   ├── copilot/                 # Protocol copilot
│   │   ├── protocol_copilot.py  # Main copilot interface
│   │   ├── context_manager.py   # Session context
│   │   └── playbook_generator.py # Response playbooks
│   ├── compliance/              # Compliance automation
│   │   ├── compliance_service.py # Compliance orchestrator
│   │   ├── gdpr_compliance.py   # GDPR automation
│   │   └── soc2_controls.py     # SOC2 mapping
│   ├── cloud_native/            # Cloud-native security
│   │   ├── service_mesh/        # Istio/Envoy integration
│   │   ├── container_security/  # Container scanning
│   │   ├── event_streaming/     # Kafka integration
│   │   └── cloud_integrations/  # AWS/Azure/GCP
│   ├── monitoring/              # Observability
│   │   ├── metrics.py           # Prometheus metrics
│   │   └── opentelemetry_tracing.py # Distributed tracing
│   ├── explainability/          # AI explainability
│   │   ├── shap_explainer.py    # SHAP values
│   │   └── lime_explainer.py    # LIME explanations
│   ├── marketplace/             # Protocol marketplace
│   ├── translation/             # Protocol translation
│   ├── models/                  # ML model definitions
│   ├── alembic/                 # Database migrations
│   └── tests/                   # Test suite (160+ files)
│
├── go/                          # Go microservices
│   ├── controlplane/            # Policy control plane
│   ├── mgmtapi/                 # Management API
│   └── agents/device-agent/     # Device agent
│
├── rust/                        # Rust dataplane
│   └── dataplane/               # High-performance processing
│       └── crates/              # Rust crates
│           ├── pqc_tls/         # Post-quantum TLS
│           └── adapters/        # Protocol adapters
│
├── ui/                          # Frontend applications
│   └── console/                 # React admin console
│       ├── src/components/      # React components
│       └── src/api/             # API clients
│
├── config/                      # Configuration files
│   ├── qbitel.yaml           # Default config
│   ├── qbitel.production.yaml # Production config
│   └── environments/            # Environment-specific
│
├── helm/                        # Helm charts
│   └── qbitel/               # Main Helm chart
│
├── kubernetes/                  # Kubernetes manifests
│   ├── service-mesh/            # Istio configs
│   └── container-security/      # Security configs
│
├── docker/                      # Docker configurations
│   ├── xds-server/              # xDS server container
│   └── admission-webhook/       # Webhook container
│
├── ops/                         # Operations & deployment
│   ├── deploy/                  # Deployment scripts
│   ├── monitoring/              # Monitoring configs
│   └── security/                # Security hardening
│
├── security/                    # Security validation
│   └── validation/              # Security validators
│
├── integration/                 # Integration layer
│   ├── database/                # DB integrations
│   └── streaming/               # Event streaming
│
├── datasets/                    # Training/test data
│   ├── anomaly_detection/       # Anomaly samples
│   ├── protocols/               # Protocol samples
│   └── threat_intelligence/     # Threat data
│
├── docs/                        # Documentation
│   └── products/                # Product guides (01-10)
│
├── scripts/                     # Utility scripts
├── tests/                       # Additional tests
└── .github/workflows/           # CI/CD pipelines
```

---

## 4. Core Components

### 4.1 AI Engine Core (`ai_engine/core/engine.py`)

The `QbitelAIEngine` class is the main orchestrator:

```python
class QbitelAIEngine:
    """Central AI engine orchestrating all ML/AI capabilities"""

    # State machine: "initialized" -> "ready" -> "stopped"/"failed"
    state: str

    # Core components
    protocol_discovery: Optional[PCFGInference]
    field_detector: Optional[FieldDetector]
    anomaly_detector: Optional[EnsembleAnomalyDetector]
    model_registry: Optional[ModelRegistry]

    # Key methods
    async def discover_protocol(model_input) -> DiscoveryResult
    async def detect_fields(model_input) -> FieldPrediction
    async def detect_anomalies(model_input) -> AnomalyResult
    async def get_status() -> Dict[str, Any]
```

**Metrics exposed:**
- `qbitel_inference_total` (Counter)
- `qbitel_inference_duration_seconds` (Histogram)
- `qbitel_model_accuracy` (Gauge)

### 4.2 Protocol Discovery Orchestrator (`ai_engine/discovery/protocol_discovery_orchestrator.py`)

Multi-phase discovery pipeline:

```python
class DiscoveryPhase(Enum):
    INITIALIZATION = "initialization"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    CLASSIFICATION = "classification"
    GRAMMAR_LEARNING = "grammar_learning"
    PARSER_GENERATION = "parser_generation"
    VALIDATION = "validation"
    COMPLETION = "completion"

@dataclass
class DiscoveryRequest:
    messages: List[bytes]
    known_protocol: Optional[str] = None
    training_mode: bool = False
    confidence_threshold: float = 0.7
    generate_parser: bool = True
    validate_results: bool = True

@dataclass
class DiscoveryResult:
    protocol_type: str
    confidence: float
    grammar: Optional[Grammar]
    parser: Optional[GeneratedParser]
    validation_result: Optional[ValidationResult]
    processing_time: float
    phases_completed: List[DiscoveryPhase]
```

### 4.3 Field Detection (`ai_engine/detection/field_detector.py`)

BiLSTM-CRF architecture for field boundary detection:

```python
class FieldDetector:
    """Detects field boundaries using BiLSTM-CRF"""

    # Model architecture
    # BiLSTM: 128 hidden units, bidirectional
    # CRF: Sequence labeling with IOB scheme

    # Output tags
    # O = Outside
    # B-FIELD = Begin field
    # I-FIELD = Inside field

@dataclass
class FieldPrediction:
    message_length: int
    detected_fields: List[FieldBoundary]
    confidence_score: float
    processing_time: float
    model_version: str

class FieldType(Enum):
    INTEGER = "integer"
    STRING = "string"
    BINARY = "binary"
    TIMESTAMP = "timestamp"
    ADDRESS = "address"
    LENGTH = "length"
    CHECKSUM = "checksum"
```

### 4.4 Anomaly Detection Ensemble (`ai_engine/anomaly/ensemble_detector.py`)

```python
class EnsembleAnomalyDetector:
    """Combines multiple detection strategies"""

    # Detection methods:
    # - Statistical deviation (z-score)
    # - Volatility analysis (entropy)
    # - Pattern recognition
    # - Outlier detection (isolation-inspired)

@dataclass
class AnomalyResult:
    score: float           # 0-1 anomaly score
    confidence: float      # Model confidence
    is_anomalous: bool     # Binary decision
    individual_scores: Dict[str, float]
    explanation: str
    timestamp: datetime
```

### 4.5 Zero-Touch Decision Engine (`ai_engine/security/decision_engine.py`)

LLM-powered autonomous security decisions:

```python
class ZeroTouchDecisionEngine:
    """Autonomous security decision-making"""

    # Decision flow:
    # Security Event → Threat Analysis (ML + LLM) → Business Impact
    # → Response Options → LLM Decision → Safety Check → Execute/Escalate

    # Confidence thresholds:
    # >= 0.95 + Low Risk  → AUTONOMOUS EXECUTION
    # >= 0.85 + Med Risk  → AUTO-APPROVE & EXECUTE
    # >= 0.50             → ESCALATE TO SOC ANALYST
    # < 0.50              → ESCALATE TO SOC MANAGER
```

### 4.6 Security Orchestrator (`ai_engine/security/security_service.py`)

```python
class SecurityOrchestratorService:
    """Main security orchestrator"""

    decision_engine: ZeroTouchDecisionEngine
    threat_analyzer: ThreatAnalyzer
    active_incidents: Dict[str, IncidentInfo]  # Max 50 concurrent

    async def analyze_threat(event) -> ThreatAnalysis
    async def execute_response(analysis) -> AutomatedResponse
    async def escalate_to_human(analysis) -> HumanEscalation
```

---

## 5. API Layer

### 5.1 FastAPI Application (`ai_engine/api/rest.py`)

```python
def create_app(config: Config) -> FastAPI:
    app = FastAPI(
        title="QBITEL Bridge Engine",
        version="2.0.0",
        description="Enterprise-grade AI-powered protocol discovery"
    )

    # Middleware stack (order matters!)
    # 1. GracefulShutdownMiddleware
    # 2. RequestTrackingMiddleware (correlation IDs)
    # 3. PayloadSizeLimitMiddleware (10MB default)
    # 4. ContentTypeValidationMiddleware
    # 5. RequestLoggingMiddleware
    # 6. SecurityHeadersMiddleware (HSTS, CSP)
    # 7. TrustedHostMiddleware
    # 8. GZipMiddleware

    # Routers
    app.include_router(copilot_router, prefix="/api/v1/copilot")
    app.include_router(translation_router, prefix="/api/v1/translation")
    app.include_router(security_router, prefix="/api/v1/security")
    app.include_router(marketplace_router, prefix="/api/v1/marketplace")
    app.include_router(compliance_router, prefix="/api/v1/compliance")

    return app
```

### 5.2 API Endpoints Reference

#### Copilot Endpoints (`/api/v1/copilot/`)

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/` | Get copilot info | Optional |
| POST | `/query` | Submit query | Required |
| WS | `/ws` | WebSocket streaming | Required |
| GET | `/sessions` | List sessions | Required |
| GET | `/sessions/{id}` | Get session | Required |
| DELETE | `/sessions/{id}` | Delete session | Required |
| GET | `/health` | Copilot health | Optional |

#### Protocol Discovery Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/api/v1/discover` | Discover protocol | Required |
| POST | `/api/v1/fields/detect` | Detect fields | Required |
| POST | `/api/v1/anomalies/detect` | Detect anomalies | Required |

#### Security Endpoints (`/api/v1/security/`)

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/analyze` | Analyze security event | Required |
| POST | `/respond` | Execute response | Elevated |
| GET | `/incidents` | List incidents | Required |
| GET | `/incidents/{id}` | Get incident | Required |

#### Health Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Basic health |
| GET | `/healthz` | Kubernetes liveness |
| GET | `/readyz` | Kubernetes readiness |
| GET | `/metrics` | Prometheus metrics |

### 5.3 Request/Response Schemas (`ai_engine/api/schemas.py`)

```python
class CopilotQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    query_type: Optional[QueryType] = None
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    packet_data: Optional[bytes] = None
    enable_learning: bool = True
    preferred_provider: Optional[LLMProvider] = None

class CopilotResponse(BaseModel):
    success: bool
    response: str
    confidence: float  # 0-1
    query_type: str
    processing_time: float
    session_id: str
    suggestions: List[str]
    source_data: List[Dict]

class QueryType(Enum):
    PROTOCOL_ANALYSIS = "protocol_analysis"
    SECURITY_ASSESSMENT = "security_assessment"
    FIELD_DETECTION = "field_detection"
    COMPLIANCE_CHECK = "compliance_check"
    ANOMALY_DETECTION = "anomaly_detection"
    GENERAL_QUESTION = "general_question"

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
```

### 5.4 Authentication (`ai_engine/api/auth.py`)

```python
# Token types
# - Access Token: 30 minutes (8 hours in prod)
# - Refresh Token: 7 days
# - API Key: 90 days (configurable)

# Auth headers supported:
# Authorization: Bearer <jwt_token>
# X-API-Key: <api_key>

# MFA methods supported:
# - TOTP (Time-based OTP)
# - SMS
# - Email
# - Hardware Token (WebAuthn)

# User roles (RBAC)
class UserRole(Enum):
    ADMINISTRATOR = "administrator"       # Full access
    SECURITY_ANALYST = "security_analyst" # Security ops
    OPERATOR = "operator"                 # Operational tasks
    VIEWER = "viewer"                     # Read-only
    API_USER = "api_user"                 # API-only
```

---

## 6. Database Models & Schemas

### 6.1 Core Models (`ai_engine/models/database.py`)

```python
# Users & Authentication
class User(Base):
    __tablename__ = "users"

    id: UUID                    # Primary key
    username: str               # Unique, indexed
    email: str                  # Unique, indexed
    password_hash: str          # bcrypt hashed
    role: UserRole              # RBAC role
    permissions: JSONB          # Fine-grained permissions

    # MFA
    mfa_enabled: bool
    mfa_method: MFAMethod
    mfa_secret: EncryptedString

    # OAuth/SAML
    oauth_provider: Optional[str]
    saml_name_id: Optional[str]

    # Security tracking
    last_login: DateTime
    failed_login_attempts: int
    account_locked_until: Optional[DateTime]

class APIKey(Base):
    __tablename__ = "api_keys"

    id: UUID
    key_hash: str              # SHA-256, unique
    key_prefix: str            # First 20 chars for display
    user_id: UUID              # FK to users
    status: APIKeyStatus       # ACTIVE|REVOKED|EXPIRED
    expires_at: Optional[DateTime]
    rate_limit_per_minute: int # Default 100
    permissions: JSONB         # Can restrict user permissions

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: UUID
    action: AuditAction        # 50+ action types
    user_id: Optional[UUID]
    resource_type: str
    resource_id: str
    details: JSONB
    success: bool
    created_at: DateTime       # Indexed
```

### 6.2 Database Migrations (`ai_engine/alembic/versions/`)

| Migration | Purpose |
|-----------|---------|
| `001_initial_auth_schema.py` | Users, API Keys, Sessions |
| `002_add_explainability_tables.py` | Explanation audit trails |
| `003_add_encrypted_fields.py` | Field encryption support |
| `004_add_marketplace_tables.py` | Protocol marketplace |

### 6.3 Database Manager (`ai_engine/core/database_manager.py`)

```python
class DatabaseManager:
    """Production-ready async SQLAlchemy connection pool"""

    # Connection pool settings
    pool_size: int = 10        # Dev: 10, Prod: 50
    max_overflow: int = 20
    pool_timeout: int = 30     # seconds
    pool_recycle: int = 3600   # Prevent stale connections

    # Methods
    async def get_session() -> AsyncSession
    async def begin_transaction() -> AsyncTransaction
    async def health_check() -> Dict[str, Any]

    # Context manager pattern
    async with database_manager.transaction() as session:
        # Auto-commit on exit, rollback on exception
        pass
```

---

## 7. Service Layer

### 7.1 Unified LLM Service (`ai_engine/llm/unified_llm_service.py`)

```python
class UnifiedLLMService:
    """Multi-provider LLM with intelligent fallback"""

    # Provider priority (on-premise first)
    # 1. Ollama (local, air-gapped)
    # 2. vLLM (local, high-performance)
    # 3. Anthropic Claude (cloud fallback)
    # 4. OpenAI GPT-4 (cloud fallback)

    # Domain-specific routing
    routing_config = {
        "protocol_copilot": {
            "primary": "ollama",
            "fallback": ["anthropic", "openai"]
        },
        "security_orchestrator": {
            "primary": "anthropic",
            "fallback": ["openai"]
        }
    }

    async def process_request(request: LLMRequest) -> LLMResponse
    async def stream_response(request: LLMRequest) -> AsyncIterator[str]

@dataclass
class LLMRequest:
    prompt: str
    feature_domain: str        # For routing
    max_tokens: int = 2000
    temperature: float = 0.3
    system_prompt: Optional[str] = None
    stream: bool = False
```

### 7.2 RAG Engine (`ai_engine/llm/rag_engine.py`)

```python
class RAGEngine:
    """Retrieval-Augmented Generation for contextual intelligence"""

    # Vector database: ChromaDB
    # Embeddings: Sentence Transformers
    # Fallback: In-memory implementation

    async def add_documents(documents: List[Document])
    async def query(query: str, top_k: int = 5) -> List[RelevantDoc]
    async def augment_prompt(prompt: str) -> str
```

### 7.3 Compliance Service (`ai_engine/compliance/compliance_service.py`)

```python
class ComplianceService:
    """Multi-framework compliance automation"""

    # Supported frameworks
    frameworks = [
        "PCI-DSS",
        "HIPAA",
        "SOX",
        "GDPR",
        "NIST CSF",
        "ISO 27001",
        "SOC2"
    ]

    async def assess_compliance(system_id, framework) -> Assessment
    async def generate_report(assessment, format) -> Report
    async def record_audit_event(event) -> AuditLog
```

---

## 8. Security Implementation

### 8.1 Post-Quantum Cryptography

| Algorithm | Purpose | Security Level |
|-----------|---------|----------------|
| Kyber-1024 | Key Encapsulation | NIST Level 5 |
| Dilithium-5 | Digital Signatures | NIST Level 5 |
| AES-256-GCM | Symmetric Encryption | 256-bit |
| SPHINCS+ | Hash-based Signatures | Backup |

**Implementation locations:**
- `ai_engine/cloud_native/service_mesh/qkd_certificate_manager.py`
- `rust/dataplane/crates/pqc_tls/`

### 8.2 Authentication Flow

```
POST /login
    │
    ▼
Verify Credentials (bcrypt)
    │
    ├─ Invalid → 401 (increment failed attempts)
    │
    └─ Valid → Check MFA
                  │
                  ├─ MFA Enabled → Verify Code
                  │                    │
                  │                    ├─ Valid → Generate Tokens
                  │                    │
                  │                    └─ Invalid → 401
                  │
                  └─ MFA Disabled → Generate Tokens
                                        │
                                        ▼
                                   Return:
                                   - Access Token (30min)
                                   - Refresh Token (7 days)
                                   - User Info
```

### 8.3 Security Headers (Middleware)

```python
# SecurityHeadersMiddleware adds:
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
Content-Security-Policy: default-src 'self'; script-src 'self'
```

### 8.4 Input Validation (`ai_engine/api/input_validation.py`)

**Patterns detected and blocked:**
- SQL Injection: `UNION SELECT`, `OR 1=1`, `DROP TABLE`
- XSS: `<script>`, `javascript:`, `onclick=`
- Command Injection: `; | $ ( ) \``
- Path Traversal: `../`, `%2e%2e`, `/etc/`

### 8.5 Rate Limiting (`ai_engine/api/rate_limiter.py`)

```python
# Default limits
requests_per_minute: 100
requests_per_hour: 1000
burst_size: 200

# Response headers
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1701696485
Retry-After: 60  # When rate limited
```

---

## 9. AI/ML Components

### 9.1 Model Architecture Summary

| Model | Type | Purpose | Location |
|-------|------|---------|----------|
| BiLSTM-CRF | Deep Learning | Field detection | `detection/field_detector.py` |
| LSTM | Deep Learning | Time-series anomaly | `anomaly/lstm_detector.py` |
| VAE | Deep Learning | Feature extraction | `anomaly/vae_detector.py` |
| Isolation Forest | Ensemble | Anomaly detection | `anomaly/isolation_forest.py` |
| PCFG | Statistical | Grammar inference | `discovery/pcfg_inference.py` |
| CNN | Deep Learning | Protocol classification | Various |

### 9.2 Training Configuration

```yaml
training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
  early_stopping_patience: 10
  validation_split: 0.2

model:
  lstm_hidden_size: 128
  lstm_num_layers: 2
  cnn_filter_sizes: [3, 4, 5]
  cnn_num_filters: 100
  dropout_rate: 0.2
```

### 9.3 Explainability (`ai_engine/explainability/`)

| Tool | Purpose | Location |
|------|---------|----------|
| SHAP | Feature importance | `shap_explainer.py` |
| LIME | Local explanations | `lime_explainer.py` |
| Audit Logger | Decision trails | `audit_logger.py` |
| Drift Monitor | Performance tracking | `drift_monitor.py` |

---

## 10. Cloud-Native Infrastructure

### 10.1 Service Mesh (`ai_engine/cloud_native/service_mesh/`)

| Component | Purpose | File |
|-----------|---------|------|
| xDS Server | Envoy configuration | `xds_server.py` |
| mTLS Config | Mutual TLS | `mtls_config.py` |
| QKD Cert Manager | Quantum-safe certs | `qkd_certificate_manager.py` |
| Sidecar Injector | Envoy injection | `sidecar_injector.py` |

### 10.2 Container Security (`ai_engine/cloud_native/container_security/`)

| Component | Purpose | File |
|-----------|---------|------|
| Webhook Server | Admission control | `admission_control/webhook_server.py` |
| Vulnerability Scanner | Image scanning | `image_scanning/vulnerability_scanner.py` |
| eBPF Monitor | Syscall monitoring | `runtime_protection/ebpf_monitor.py` |
| Dilithium Signer | PQ signatures | `signing/dilithium_signer.py` |

### 10.3 Cloud Integrations

| Cloud | Integration | File |
|-------|-------------|------|
| AWS | Security Hub | `cloud_integrations/aws/security_hub.py` |
| Azure | Sentinel | `cloud_integrations/azure/sentinel.py` |
| GCP | Security Command Center | `cloud_integrations/gcp/security_command_center.py` |

### 10.4 Event Streaming (`ai_engine/cloud_native/event_streaming/`)

```python
# Kafka Producer with AES-256-GCM encryption
# Throughput: 100,000+ msg/s
# Event types: PACKET_CAPTURED, PROTOCOL_DISCOVERED, AI_ANALYSIS, SECURITY_EVENT
```

---

## 11. Configuration Management

### 11.1 Configuration Files

| File | Purpose | Environment |
|------|---------|-------------|
| `config/qbitel.yaml` | Default config | Development |
| `config/qbitel.production.yaml` | Production config | Production |
| `config/environments/staging.yaml` | Staging overrides | Staging |
| `config/compliance.yaml` | Compliance config | All |

### 11.2 Environment Variables

**Required for Production:**

```bash
# Database
DATABASE_HOST=db.example.com
DATABASE_PORT=5432
DATABASE_NAME=qbitel_prod
DATABASE_USER=qbitel
DATABASE_PASSWORD=<32+ chars, from secrets manager>

# Redis
REDIS_HOST=redis.example.com
REDIS_PORT=6379
REDIS_PASSWORD=<from secrets manager>

# Security (MANDATORY)
JWT_SECRET=<32+ chars>
API_KEY=<32+ chars>
ENCRYPTION_KEY=<32+ chars, AES-256-GCM>

# TLS
TLS_ENABLED=true
TLS_CERT_FILE=/etc/qbitel/certs/server.crt
TLS_KEY_FILE=/etc/qbitel/certs/server.key

# LLM (On-premise default)
QBITEL_LLM_PROVIDER=ollama
QBITEL_LLM_ENDPOINT=http://localhost:11434
QBITEL_AIRGAPPED_MODE=true

# Monitoring
SENTRY_DSN=<sentry endpoint>
LOG_LEVEL=INFO
```

### 11.3 Configuration Loading Order

1. Load base config from `config/qbitel.yaml`
2. Override with environment-specific config
3. Load environment variables (highest priority)
4. Fetch secrets from secrets manager (Vault, AWS SM)
5. Validate all required secrets present
6. Initialize logging and connections

---

## 12. External Integrations

### 12.1 LLM Providers

| Provider | Priority | Use Case |
|----------|----------|----------|
| Ollama | 1 (Primary) | Air-gapped, on-premise |
| vLLM | 2 | High-performance local |
| Anthropic Claude | 3 | Cloud fallback |
| OpenAI GPT-4 | 4 | Cloud fallback |

### 12.2 Secrets Management

| Provider | SDK | Config Key |
|----------|-----|------------|
| HashiCorp Vault | hvac | `VAULT_ADDR`, `VAULT_TOKEN` |
| AWS Secrets Manager | boto3 | AWS credentials |
| Azure Key Vault | azure-identity | Azure credentials |
| Kubernetes Secrets | Native | In-cluster |

### 12.3 Monitoring Stack

| Tool | Purpose | Port |
|------|---------|------|
| Prometheus | Metrics | 9090 |
| Grafana | Dashboards | 3000 |
| Jaeger | Tracing | 6831 |
| Sentry | Error tracking | N/A |

### 12.4 SIEM Integration

| SIEM | Protocol | Config |
|------|----------|--------|
| Splunk | HEC | `splunk.hec_endpoint` |
| Elastic | REST | `elastic.endpoint` |
| QRadar | API | `qradar.api_endpoint` |

---

## 13. Testing Strategy

### 13.1 Test Structure

```
ai_engine/tests/
├── cloud_native/          # Cloud-native feature tests
├── security/              # Security and quantum crypto tests
├── integration/           # Integration tests
├── performance/           # Performance and load tests
├── explainability/        # AI explainability tests
└── unit/                  # Unit tests for all modules
```

### 13.2 Running Tests

```bash
# Run all tests
pytest ai_engine/tests/ -v

# Run specific test suites
pytest ai_engine/tests/cloud_native/ -v
pytest ai_engine/tests/security/ -v -k quantum
pytest ai_engine/tests/integration/ -v
pytest ai_engine/tests/performance/ -v

# Run with coverage
pytest ai_engine/tests/ --cov=ai_engine --cov-report=html
```

### 13.3 Test Coverage Requirements

- Minimum coverage: 80%
- Current coverage: 85%
- Critical paths: 95%+

---

## 14. Deployment Guide

### 14.1 Docker Deployment

```bash
cd docker
docker-compose up -d
docker-compose ps
```

### 14.2 Kubernetes Deployment

```bash
# Apply namespace and core resources
kubectl apply -f kubernetes/service-mesh/namespace.yaml
kubectl apply -f kubernetes/service-mesh/xds-server-deployment.yaml
kubectl apply -f kubernetes/container-security/admission-webhook-deployment.yaml

# Verify
kubectl get pods -n qbitel-service-mesh
```

### 14.3 Helm Deployment

```bash
helm install qbitel ./helm/qbitel \
  --namespace qbitel \
  --create-namespace \
  --values ./helm/qbitel/values-production.yaml
```

### 14.4 Air-Gapped Deployment

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Download models (while connected)
ollama pull llama3.2:8b
ollama pull mixtral:8x7b

# 3. Configure for air-gapped mode
export QBITEL_LLM_PROVIDER=ollama
export QBITEL_AIRGAPPED_MODE=true
export QBITEL_DISABLE_CLOUD_LLMS=true

# 4. Deploy
python -m ai_engine --airgapped
```

---

## 15. Design Patterns & Best Practices

### 15.1 Patterns Used

| Pattern | Where Used |
|---------|------------|
| Server Manager | `api/server.py` - Graceful shutdown |
| Dependency Injection | FastAPI `Depends()` |
| Circuit Breaker | `core/database_circuit_breaker.py` |
| Repository | Database access layer |
| Strategy | LLM provider fallback |
| Observer | Event tracking/metrics |
| Factory | Model/service creation |

### 15.2 Error Handling

```python
# Exception hierarchy
QbitelAIException (base)
├── ConfigurationException
├── ModelException
│   ├── ModelLoadException
│   └── ModelInferenceException
├── ProtocolException
│   └── DiscoveryException
└── SecurityException
    └── AuthenticationException

# Recovery strategies
class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    DEGRADE = "degrade"
    FAIL_FAST = "fail_fast"
```

### 15.3 Async Best Practices

```python
# Always use async for I/O operations
async def process_request(request: Request) -> Response:
    # Use async context managers
    async with database_manager.transaction() as session:
        result = await session.execute(query)

    # Parallel execution where possible
    results = await asyncio.gather(
        fetch_data_a(),
        fetch_data_b(),
        fetch_data_c()
    )

    return Response(data=results)
```

---

## 16. Coding Standards

### 16.1 Python Style Guide

```python
# Imports (isort ordering)
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field

from ai_engine.core.config import Config
from ai_engine.core.exceptions import QbitelAIException


# Type hints required
async def process_data(
    data: bytes,
    options: Optional[Dict[str, Any]] = None
) -> ProcessingResult:
    """Process input data with optional configuration.

    Args:
        data: Raw input bytes to process
        options: Optional processing configuration

    Returns:
        ProcessingResult with status and output

    Raises:
        ProcessingException: If processing fails
    """
    pass


# Dataclasses for data structures
@dataclass
class ProcessingResult:
    success: bool
    output: bytes
    processing_time: float
    metadata: Dict[str, Any]


# Pydantic for API schemas
class ProcessRequest(BaseModel):
    data: bytes
    format: str = Field(default="raw", description="Input format")

    class Config:
        json_schema_extra = {
            "example": {"data": b"...", "format": "hex"}
        }
```

### 16.2 Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `ProtocolDiscovery` |
| Functions | snake_case | `detect_fields()` |
| Variables | snake_case | `field_count` |
| Constants | UPPER_SNAKE | `MAX_RETRIES` |
| Private | Leading underscore | `_internal_state` |
| Protected | Leading underscore | `_process_data()` |

### 16.3 File Organization

```python
# Standard file structure
"""Module docstring describing purpose."""

from __future__ import annotations

# Standard library imports
import asyncio
import logging

# Third-party imports
from fastapi import FastAPI

# Local imports
from ai_engine.core import config

# Module-level constants
TIMEOUT = 30
MAX_RETRIES = 3

# Logger
logger = logging.getLogger(__name__)


# Classes and functions
class MyClass:
    """Class docstring."""
    pass


def my_function():
    """Function docstring."""
    pass


# Main execution (if applicable)
if __name__ == "__main__":
    pass
```

---

## 17. Troubleshooting Guide

### 17.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Connection refused | Service not running | Check `docker-compose ps` |
| 401 Unauthorized | Invalid/expired token | Refresh token or re-login |
| 429 Too Many Requests | Rate limit exceeded | Wait for reset or increase limits |
| Model load failed | Missing weights | Check `model_cache_dir` path |
| Database timeout | Pool exhausted | Increase `pool_size` |

### 17.2 Log Locations

| Log Type | Location |
|----------|----------|
| Application | `/var/log/qbitel/app.log` |
| Access | `/var/log/qbitel/access.log` |
| Error | `/var/log/qbitel/error.log` |
| Audit | `/var/log/qbitel/audit.log` |
| Container | `docker logs <container>` |

### 17.3 Health Check Endpoints

```bash
# Basic health
curl http://localhost:8000/health

# Kubernetes liveness
curl http://localhost:8000/healthz

# Kubernetes readiness (checks dependencies)
curl http://localhost:8000/readyz

# Prometheus metrics
curl http://localhost:8000/metrics
```

### 17.4 Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=true

# Run with verbose output
python -m ai_engine --log-level DEBUG
```

---

## 18. Go Services

### 18.1 Overview

The Go codebase provides three microservices for device management, policy control, and attestation:

| Service | Location | Port | Purpose |
|---------|----------|------|---------|
| Control Plane | `go/controlplane/` | 8080 | OPA policy engine, bundle management |
| Management API | `go/mgmtapi/` | 8081 | Device lifecycle, certificates, attestation |
| Device Agent | `go/agents/device-agent/` | N/A | On-device TPM agent |

### 18.2 Control Plane Service

**Location**: `go/controlplane/`

**Dependencies** (Go 1.22):
- `github.com/gin-gonic/gin` - REST API framework
- `github.com/open-policy-agent/opa` - Policy engine
- `github.com/sigstore/cosign/v2` - Bundle signing
- `go.uber.org/zap` - Structured logging

**API Endpoints**:
```
GET  /healthz              - Health check
POST /policy/validate      - Validate policy
POST /policy/evaluate      - Evaluate policy
POST /bundles              - Upload bundle
GET  /bundles              - List bundles
GET  /bundles/:id          - Get bundle
DELETE /bundles/:id        - Delete bundle
POST /bundles/:id/load     - Load bundle
GET  /stats/policy         - Policy stats
GET  /stats/bundles        - Bundle stats
```

**Key Components**:
- `PolicyEngine` - OPA-based policy evaluation with caching
- `BundleManager` - Policy bundle lifecycle (filesystem, S3, GCS, OCI storage)
- `CosignVerifier` - Signature verification for policy bundles
- `VaultClient` - HashiCorp Vault integration for key management

### 18.3 Management API Service

**Location**: `go/mgmtapi/`

**API Endpoints**:
```
# Device Management
GET    /v1/devices                    - List devices
GET    /v1/devices/:id                - Get device
PATCH  /v1/devices/:id                - Update device
POST   /v1/devices/:id/suspend        - Suspend device
POST   /v1/devices/:id/resume         - Resume device
POST   /v1/devices/:id/decommission   - Decommission device

# Enrollment
POST   /v1/devices/enrollment/sessions              - Start enrollment
GET    /v1/devices/enrollment/sessions/:id          - Get session
POST   /v1/devices/enrollment/sessions/:id/attest   - Submit attestation
POST   /v1/devices/enrollment/sessions/:id/approve  - Approve enrollment
POST   /v1/devices/enrollment/sessions/:id/complete - Complete with cert

# Certificates
GET    /v1/devices/:id/certificate        - Get certificate
POST   /v1/devices/:id/certificate/renew  - Renew certificate
POST   /v1/devices/:id/certificate/revoke - Revoke certificate
```

**Key Components**:
- `DeviceLifecycleManager` - Complete device lifecycle management
- `CertificateManager` - PKI with Root CA and Intermediate CA
- `AttestationVerifier` - TPM quote and PCR verification

### 18.4 Device Agent

**Location**: `go/agents/device-agent/`

**Purpose**: Runs on IoT/edge devices for TPM attestation

**Key Features**:
- TPM 2.0 key sealing and unsealing
- PCR-based attestation quotes
- EK/AK certificate management
- Periodic attestation (5-minute interval)
- Enrollment with control plane

**Environment Variables**:
```bash
DEVICE_ID=<device-id>
CONTROL_URL=https://control.qbitel.local
```

### 18.5 Go Coding Standards

```go
// Package organization
package policy

import (
    // Standard library
    "context"
    "sync"

    // Third-party
    "github.com/gin-gonic/gin"
    "go.uber.org/zap"

    // Internal
    "github.com/yazhsab/go/controlplane/internal/vault"
)

// Struct with configuration pattern
type Config struct {
    BundleURL     string        `json:"bundle_url"`
    PollInterval  time.Duration `json:"poll_interval"`
    RequireSigned bool          `json:"require_signed"`
}

func DefaultConfig() *Config {
    return &Config{
        PollInterval:  5 * time.Minute,
        RequireSigned: true,
    }
}
```

---

## 19. Rust Dataplane

### 19.1 Overview

The Rust codebase provides high-performance packet processing, post-quantum cryptography, and protocol adapters:

**Location**: `rust/dataplane/`

**Workspace Members**:
```
crates/
├── adapter_sdk/     - Core L7Adapter trait
├── pqc_tls/         - Post-quantum TLS (Kyber-768, Dilithium)
├── io_engine/       - High-performance bridge with HA
├── dpdk_engine/     - DPDK kernel bypass
├── dpi_engine/      - Deep packet inspection + ML
├── k8s_operator/    - Kubernetes operator
├── ai_bridge/       - Python interop (PyO3)
└── adapters/
    ├── iso8583/     - Financial protocol (PCI DSS)
    ├── modbus/      - Industrial SCADA
    ├── tn3270e/     - Mainframe terminal
    └── hl7_mllp/    - Healthcare HL7
```

### 19.2 Adapter SDK

**Core Trait** (`adapter_sdk/src/lib.rs`):
```rust
#[async_trait]
pub trait L7Adapter: Send + Sync {
    async fn to_upstream(&self, input: Bytes) -> Result<Bytes, AdapterError>;
    async fn to_client(&self, input: Bytes) -> Result<Bytes, AdapterError>;
    fn name(&self) -> &'static str;
}
```

### 19.3 PQC-TLS Crate

**Location**: `rust/dataplane/crates/pqc_tls/`

**Features**:
- Kyber-768 KEM (post-quantum key encapsulation)
- Dilithium signatures
- Hardware acceleration (AVX-512, AVX2, AES-NI)
- HSM integration via PKCS#11
- Automated key rotation with threat-level awareness
- Memory pool for crypto buffers

**Key Types**:
```rust
pub struct PqcTlsManager {
    kyber_kem: Arc<KyberKEM>,
    hsm_manager: Option<Arc<HsmPqcManager>>,
    lifecycle_manager: Arc<KeyLifecycleManager>,
    rotation_manager: Arc<QuantumSafeRotationManager>,
}

// Kyber parameters (Kyber-768)
pub const KYBER_PUBLIC_KEY_SIZE: usize = 1184;
pub const KYBER_PRIVATE_KEY_SIZE: usize = 768;
pub const KYBER_CIPHERTEXT_SIZE: usize = 1088;
pub const KYBER_SHARED_SECRET_SIZE: usize = 32;
```

**Dependencies**:
- `pqcrypto-kyber`, `pqcrypto-dilithium`, `oqs`
- `tokio-openssl` for TLS
- `pkcs11` for HSM
- `rayon` for parallelism
- `wide` for SIMD

### 19.4 I/O Engine

**Location**: `rust/dataplane/crates/io_engine/`

**Features**:
- Bidirectional protocol transformation bridge
- Active-Standby HA with automatic failover
- Connection pooling with lifecycle management
- Circuit breaker pattern
- Lock-free high-frequency metrics
- Distributed tracing (OpenTelemetry)

**Architecture**:
```
Client → Reader → Adapter.to_upstream() → Channel → Writer → Upstream
Upstream → Reader → Adapter.to_client() → Channel → Writer → Client
```

### 19.5 Protocol Adapters

| Adapter | Protocol | Features |
|---------|----------|----------|
| **ISO-8583** | Financial/Payment | PCI DSS masking, field encryption, MAC, routing |
| **Modbus** | Industrial/SCADA | TCP/RTU, CRC validation, register access control |
| **TN3270E** | Mainframe Terminal | Screen scraping, session management |
| **HL7 MLLP** | Healthcare | MLLP framing, field mapping, validation |

### 19.6 DPI Engine

**Location**: `rust/dataplane/crates/dpi_engine/`

**ML Integration**:
- PyTorch via `tch` bindings
- ONNX Runtime via `ort`
- Candle framework
- Hyperscan for pattern matching

**Protocol Classification**:
- HTTP, HTTPS, SSH, DNS, SIP, RTP
- BitTorrent, Skype, WhatsApp, Telegram
- Zoom, Teams, Slack, Netflix, YouTube

### 19.7 Rust Coding Standards

```rust
// Module organization
pub mod kyber;
pub mod hsm;
pub mod client;
pub mod server;

// Error handling with thiserror
#[derive(Debug, thiserror::Error)]
pub enum TlsError {
    #[error("OpenSSL error: {0}")]
    Openssl(String),
    #[error("I/O error: {0}")]
    Io(String),
    #[error("Policy violation: {0}")]
    Policy(String),
}

// Async trait pattern
#[async_trait]
pub trait L7Adapter: Send + Sync {
    async fn to_upstream(&self, input: Bytes) -> Result<Bytes, AdapterError>;
}

// Builder pattern
impl Iso8583AdapterBuilder {
    pub fn with_parser(mut self, parser: Iso8583Parser) -> Self {
        self.parser = Some(parser);
        self
    }
    pub fn build(self) -> Result<Iso8583Adapter, Iso8583Error> { ... }
}
```

---

## 20. Frontend Console

### 20.1 Overview

**Location**: `ui/console/`

**Technology Stack**:
- React 18.2 with TypeScript 5.3
- Material-UI 5.14 (MUI)
- Vite 5.0 for bundling
- OIDC authentication
- WebSocket for real-time updates

### 20.2 Directory Structure

```
ui/console/
├── src/
│   ├── main.tsx              - Entry point
│   ├── App.tsx               - Main shell
│   ├── api/
│   │   ├── devices.ts        - Device API client
│   │   ├── marketplace.ts    - Marketplace API
│   │   └── enhancedApiClient.ts - Cached API with WebSocket
│   ├── auth/
│   │   └── oidc.ts           - OIDC service
│   ├── components/
│   │   ├── Dashboard.tsx
│   │   ├── DeviceManagement.tsx
│   │   ├── PolicyManagement.tsx
│   │   ├── SecurityMonitoring.tsx
│   │   ├── ProtocolVisualization.tsx
│   │   ├── AIModelMonitoring.tsx
│   │   ├── ThreatIntelligence.tsx
│   │   └── marketplace/      - Marketplace components
│   ├── routes/
│   │   └── AppRoutes.tsx     - RBAC routing
│   ├── theme/
│   │   └── EnterpriseTheme.ts - 5 theme variants
│   └── types/
│       ├── auth.ts
│       ├── device.ts
│       └── marketplace.ts
├── package.json
├── vite.config.ts
└── tsconfig.json
```

### 20.3 Key Components

| Component | Purpose |
|-----------|---------|
| `Dashboard` | Main metrics overview |
| `DeviceManagement` | Device CRUD & status |
| `PolicyManagement` | Policy configuration |
| `SecurityMonitoring` | Security alerts |
| `ProtocolVisualization` | D3/Plotly protocol viz |
| `AIModelMonitoring` | ML model metrics |
| `ThreatIntelligence` | IOC management |
| `MarketplaceHome` | Protocol marketplace |
| `ProtocolCopilotChat` | AI chat interface |

### 20.4 Authentication

**OIDC Configuration**:
```typescript
{
  authority: 'https://auth.qbitel.local',
  clientId: 'qbitel-console',
  scope: 'openid profile email roles permissions organization',
  automaticSilentRenew: true,
}
```

**User Roles**:
- `admin`, `system_admin`, `org_admin`
- `security_analyst`, `operator`, `viewer`

### 20.5 Routes with RBAC

| Route | Component | Required Permission |
|-------|-----------|---------------------|
| `/dashboard` | Dashboard | `dashboard:read` |
| `/devices` | DeviceManagement | `device:read` |
| `/policies` | PolicyManagement | `policy:read` |
| `/security` | SecurityMonitoring | `security:read` |
| `/protocols` | ProtocolVisualization | `protocol:read` (enterprise) |
| `/ai-models` | AIModelMonitoring | `ai:read` (enterprise) |
| `/marketplace` | MarketplaceHome | `marketplace:read` |
| `/settings` | SystemSettings | admin role |

### 20.6 Theme System

**5 Theme Variants**:
1. Light - Standard light mode
2. Dark - Dark mode
3. High Contrast - Accessibility
4. Blue-Grey - Professional
5. Enterprise - Custom branding

```typescript
const useTheme = () => {
  const { theme, setTheme } = useContext(ThemeContext);
  return { theme, setTheme };
};
```

### 20.7 WebSocket Real-Time Updates

**Message Types**:
```typescript
// Protocol
PROTOCOL_DISCOVERED, PROTOCOL_UPDATED, PROTOCOL_METRICS

// AI Models
MODEL_METRICS, MODEL_ALERT, MODEL_TRAINING_COMPLETE

// Security
THREAT_DETECTED, THREAT_RESOLVED, SECURITY_ALERT

// System
HEARTBEAT, SYSTEM_STATUS, NOTIFICATION
```

### 20.8 Build Configuration

**Vite Config**:
```typescript
{
  plugins: [react()],
  resolve: {
    alias: {
      '@': './src',
      '@/components': './src/components',
      '@/api': './src/api',
    }
  },
  server: {
    port: 3000,
    proxy: {
      '/api': { target: 'https://api.qbitel.local' }
    }
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: { vendor: ['react', 'react-dom'], mui: ['@mui/material'] }
      }
    }
  }
}
```

### 20.9 TypeScript/React Coding Standards

```typescript
// Component pattern
interface Props {
  deviceId: string;
  onUpdate?: (device: Device) => void;
}

export const DeviceCard: React.FC<Props> = ({ deviceId, onUpdate }) => {
  const [device, setDevice] = useState<Device | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDevice();
  }, [deviceId]);

  return (
    <Card>
      {loading ? <CircularProgress /> : <DeviceDetails device={device} />}
    </Card>
  );
};

// API client pattern
export class DeviceApiClient {
  private baseUrl: string;
  private authService: OidcAuthService;

  async getDevice(id: string): Promise<Device> {
    const response = await fetch(`${this.baseUrl}/devices/${id}`, {
      headers: this.getAuthHeaders(),
    });
    return response.json();
  }
}
```

---

## Appendix A: Quick Reference Cards

### A.1 API Quick Reference

```
Base URL: http://localhost:8000/api/v1

# Authentication
POST /auth/login        → {access_token, refresh_token}
POST /auth/refresh      → {access_token}
POST /auth/logout       → 204 No Content

# Copilot
POST /copilot/query     → {response, confidence, suggestions}
WS   /copilot/ws        → Real-time streaming

# Protocol Discovery
POST /discover          → {protocol_type, confidence, grammar}
POST /fields/detect     → {detected_fields, confidence}
POST /anomalies/detect  → {score, is_anomalous, explanation}

# Security
POST /security/analyze  → {threat_score, recommended_action}
POST /security/respond  → {status, execution_result}
GET  /security/incidents → [{id, severity, status}]
```

### A.2 Configuration Quick Reference

```yaml
# Minimum production config
environment: production
database:
  host: ${DATABASE_HOST}
  password: ${DATABASE_PASSWORD}
  ssl_mode: require
  pool_size: 50

redis:
  host: ${REDIS_HOST}
  password: ${REDIS_PASSWORD}

security:
  jwt_secret: ${JWT_SECRET}
  encryption_key: ${ENCRYPTION_KEY}

llm:
  provider: ollama
  endpoint: http://localhost:11434
```

### A.3 Docker Quick Reference

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f ai-engine

# Stop all
docker-compose down

# Clean rebuild
docker-compose down -v && docker-compose up -d --build
```

---

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.1.0 | 2025-11-22 | Added Go, Rust, Frontend sections for complete multi-language coverage |
| 2.0.0 | 2025-11-22 | Initial knowledge base with Python codebase |

---

## Appendix C: Contact & Support

- **GitHub Issues**: https://github.com/yazhsab/qbitel-bridge/issues
- **Documentation**: `/docs/` directory
- **Enterprise Support**: enterprise@qbitel.com

---

**End of Knowledge Base**

*This document should be updated whenever significant changes are made to the codebase architecture, APIs, or infrastructure.*


---

# ============================================================
# FILE 5: DEVELOPMENT.md (Development Guide)
# ============================================================

# QBITEL Bridge - Development Guide

This guide provides information for developers who want to contribute to QBITEL Bridge or extend its functionality.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Adding New Features](#adding-new-features)
- [API Development](#api-development)
- [Contributing Guidelines](#contributing-guidelines)

## Development Setup

### Prerequisites

| Component | Requirement | Purpose |
|-----------|------------|---------|
| Python | 3.10+ | AI Engine, compliance, security |
| Rust | 1.70+ (with cargo) | Data plane, PQC-TLS |
| Go | 1.21+ | Control plane, mgmt API |
| Node.js | 18+ (with npm) | UI console |
| Docker | 20.10+ | Containerized development |
| Git | 2.30+ | Version control |

### Initial Setup (All Components)

```bash
# 1. Clone the repository
git clone https://github.com/yazhsab/qbitel-bridge.git
cd qbitel-bridge

# 2. Build everything with make
make build

# 3. Run all tests
make test
```

### Python AI Engine Setup

```bash
# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r ai_engine/requirements.txt

# Install pre-commit hooks (optional)
pre-commit install

# Verify installation
pytest ai_engine/tests/ -v --tb=short
```

### Rust Data Plane Setup

```bash
# Build the data plane
cd rust/dataplane && cargo build --locked

# Run tests
cargo test

# Run linter
cargo clippy --all-targets --all-features -- -D warnings

# Format code
cargo fmt
```

### Go Services Setup

```bash
# Build control plane
cd go/controlplane && go build -trimpath -o ../../dist/controlplane ./cmd/controlplane

# Build management API
cd go/mgmtapi && go build -trimpath -o ../../dist/mgmtapi ./cmd/mgmtapi

# Run tests
cd go/controlplane && go test ./...
cd go/mgmtapi && go test ./...
```

### UI Console Setup

```bash
cd ui/console
npm install
npm run dev
# Console available at http://localhost:3000
```

### IDE Configuration

#### VS Code

Create `.vscode/settings.json`:

```json
{
  "python.pythonPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "ai_engine/tests"
  ],
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

#### PyCharm

- Set Python interpreter to `venv/bin/python`
- Configure pytest as default test runner
- Enable Black formatter
- Enable flake8 linter

## Project Structure

```
qbitel-bridge/
├── ai_engine/                 # Python AI Engine
│   ├── core/                  # Main engine, config, orchestrator
│   ├── agents/                # Multi-agent orchestration system
│   ├── discovery/             # Protocol discovery (PCFG, transformers)
│   ├── detection/             # Field detection (BiLSTM-CRF), anomaly detection
│   ├── security/              # Zero-touch decision engine
│   ├── legacy/                # Legacy System Whisperer (COBOL, JCL)
│   ├── marketplace/           # Protocol marketplace (Stripe, S3)
│   ├── copilot/               # Protocol intelligence copilot
│   ├── crypto/                # Post-quantum cryptography (Python)
│   ├── compliance/            # Compliance automation (9 frameworks)
│   ├── llm/                   # LLM service (Ollama, RAG, providers)
│   ├── cloud_native/          # Service mesh, container security, cloud SDKs
│   ├── api/                   # REST (FastAPI) and gRPC APIs
│   └── tests/                 # Test suite (unit, integration, perf)
│
├── rust/dataplane/            # Rust high-performance data plane
│   └── crates/pqc_tls/        # PQC-TLS (ML-KEM, ML-DSA, Falcon, SLH-DSA)
│
├── go/                        # Go services
│   ├── controlplane/          # Service orchestration
│   ├── mgmtapi/               # Management REST API
│   └── agents/device-agent/   # Edge device agent
│
├── ui/console/                # React admin console (TypeScript)
│
├── helm/qbitel-bridge/        # Helm chart for Kubernetes deployment
├── ops/                       # Operations (deploy, monitoring, secrets)
├── docker/                    # Docker configurations
├── kubernetes/                # Kubernetes manifests
├── samples/cobol/             # 500+ COBOL sample programs
├── demos/                     # End-to-end demo scenarios
├── tests/                     # Conformance, fuzz, and perf tests
├── security/                  # PKI, validation, policy
├── docs/                      # Full documentation suite
├── Makefile                   # Unified build system
├── requirements.txt           # Python production dependencies
└── requirements-dev.txt       # Python development dependencies
```

## Development Workflow

### Branch Strategy

```bash
# Main branches
main                 # Production-ready code
develop             # Integration branch

# Feature branches
git checkout -b feature/your-feature-name
git checkout -b bugfix/issue-description
git checkout -b hotfix/critical-fix
```

### Making Changes

```bash
# 1. Create feature branch
git checkout -b feature/new-protocol-support

# 2. Make changes and commit
git add .
git commit -m "feat: Add support for MQTT protocol discovery"

# 3. Run tests
pytest ai_engine/tests/ -v

# 4. Run code quality checks
flake8 ai_engine/
black ai_engine/ --check

# 5. Push changes
git push origin feature/new-protocol-support

# 6. Create Pull Request on GitHub
```

### Commit Message Convention

Follow conventional commits:

```
feat: Add new feature
fix: Bug fix
docs: Documentation changes
test: Add or update tests
refactor: Code refactoring
perf: Performance improvements
chore: Build/tooling changes
```

Examples:
```bash
git commit -m "feat: Add Dilithium-5 signature verification"
git commit -m "fix: Resolve memory leak in protocol parser"
git commit -m "test: Add integration tests for xDS server"
git commit -m "docs: Update API documentation for field detection"
```

## Testing

### Running All Tests

```bash
# Run all tests across all languages
make test
```

### Python Tests

```bash
# Run all Python tests
pytest ai_engine/tests/ -v

# Run specific test file
pytest ai_engine/tests/test_core.py -v

# Run tests by marker
pytest ai_engine/tests/ -v -m "cloud_native"
pytest ai_engine/tests/ -v -m "security"
pytest ai_engine/tests/ -v -m "integration"

# Run with coverage
pytest ai_engine/tests/ --cov=ai_engine --cov-report=html

# Run performance tests
pytest ai_engine/tests/performance/ -v --benchmark-only
```

### Rust Tests

```bash
cd rust/dataplane

# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_kyber_encapsulation
```

### Go Tests

```bash
# Control plane tests
cd go/controlplane && go test ./...

# Management API tests
cd go/mgmtapi && go test ./...

# With coverage
cd go/controlplane && go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

### UI Console Tests

```bash
cd ui/console
npm test
```

### Writing Tests

#### Unit Test Example

```python
# ai_engine/tests/test_protocol_discovery.py
import pytest
from ai_engine.discovery.pcfg_inference import PCFGInference

class TestPCFGInference:
    @pytest.fixture
    def pcfg_engine(self):
        return PCFGInference()

    def test_grammar_inference(self, pcfg_engine):
        """Test grammar inference from sample data."""
        samples = [b"GET /api HTTP/1.1", b"POST /data HTTP/1.1"]
        grammar = pcfg_engine.infer_grammar(samples)

        assert grammar is not None
        assert len(grammar.rules) > 0
        assert grammar.start_symbol == 'Request'

    def test_invalid_input(self, pcfg_engine):
        """Test handling of invalid input."""
        with pytest.raises(ValueError):
            pcfg_engine.infer_grammar([])
```

#### Integration Test Example

```python
# ai_engine/tests/integration/test_xds_server.py
import pytest
import grpc
from ai_engine.cloud_native.service_mesh.xds_server import XDSServer

@pytest.mark.integration
class TestXDSServerIntegration:
    @pytest.fixture
    async def xds_server(self):
        server = XDSServer(port=50051)
        await server.start()
        yield server
        await server.stop()

    @pytest.mark.asyncio
    async def test_stream_aggregated_resources(self, xds_server):
        """Test ADS streaming."""
        async with grpc.aio.insecure_channel('localhost:50051') as channel:
            # Test implementation
            pass
```

### Test Coverage Goals

- **Overall**: 80%+ coverage
- **Core components**: 90%+ coverage
- **Critical security**: 95%+ coverage
- **API endpoints**: 85%+ coverage

## Code Quality

### Linting (All Languages)

```bash
# Run all linters
make lint
```

### Python

```bash
# Format with Black
black ai_engine/
black ai_engine/ --check  # check only

# Lint with flake8
flake8 ai_engine/ --max-line-length=120 --ignore=E203,W503

# Type check with mypy
mypy ai_engine/
```

### Rust

```bash
cd rust/dataplane

# Lint with clippy (deny warnings)
cargo clippy --all-targets --all-features -- -D warnings

# Format
cargo fmt
cargo fmt -- --check  # check only

# Security audit
cargo audit
```

### Go

```bash
# Lint with golangci-lint
cd go/controlplane && golangci-lint run ./...
cd go/mgmtapi && golangci-lint run ./...

# Security scan with gosec
cd go/controlplane && gosec ./...
cd go/mgmtapi && gosec ./...

# Format
gofmt -w go/
```

### UI Console

```bash
cd ui/console
npm run lint
npm run format
```

## Adding New Features

### Adding a New Protocol

1. **Create protocol module**:

```python
# ai_engine/discovery/protocols/mqtt.py
from ai_engine.discovery.base_protocol import BaseProtocol

class MQTTProtocol(BaseProtocol):
    """MQTT protocol discovery implementation."""

    def __init__(self):
        super().__init__(name="MQTT", version="5.0")

    def parse(self, data: bytes) -> dict:
        """Parse MQTT protocol data."""
        # Implementation
        pass

    def validate(self, data: bytes) -> bool:
        """Validate MQTT message."""
        # Implementation
        pass
```

2. **Add tests**:

```python
# ai_engine/tests/test_mqtt_protocol.py
import pytest
from ai_engine.discovery.protocols.mqtt import MQTTProtocol

class TestMQTTProtocol:
    def test_parse_connect_message(self):
        protocol = MQTTProtocol()
        data = b"\x10\x0e\x00\x04MQTT\x05\x02\x00\x3c\x00"
        result = protocol.parse(data)
        assert result['type'] == 'CONNECT'
```

3. **Register protocol**:

```python
# ai_engine/discovery/protocol_registry.py
from ai_engine.discovery.protocols.mqtt import MQTTProtocol

PROTOCOLS = {
    'http': HTTPProtocol,
    'mqtt': MQTTProtocol,  # Add new protocol
    # ...
}
```

### Adding a New Cloud Platform Integration

1. **Create integration module**:

```python
# ai_engine/cloud_native/cloud_platforms/alibaba_cloud.py
from typing import List, Dict
import logging

class AlibabaCloudSecurityCenter:
    """Alibaba Cloud Security Center integration."""

    def __init__(self, access_key_id: str, access_key_secret: str):
        self.client = self._initialize_client(access_key_id, access_key_secret)
        self.logger = logging.getLogger(__name__)

    def publish_findings(self, findings: List[Dict]) -> bool:
        """Publish security findings."""
        # Implementation
        pass
```

2. **Add tests**:

```python
# ai_engine/tests/integration/test_alibaba_cloud.py
import pytest
from unittest.mock import Mock, patch
from ai_engine.cloud_native.cloud_platforms.alibaba_cloud import AlibabaCloudSecurityCenter

class TestAlibabaCloudIntegration:
    @patch('alibaba_cloud_sdk.Client')
    def test_publish_findings(self, mock_client):
        integration = AlibabaCloudSecurityCenter('key', 'secret')
        findings = [{'title': 'Test', 'severity': 'HIGH'}]
        result = integration.publish_findings(findings)
        assert result is True
```

## API Development

### Adding REST API Endpoint

```python
# ai_engine/api/rest_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class ProtocolRequest(BaseModel):
    data: bytes
    confidence_threshold: float = 0.7

class ProtocolResponse(BaseModel):
    protocol_type: str
    confidence: float
    fields: List[Dict]

@router.post("/discover", response_model=ProtocolResponse)
async def discover_protocol(request: ProtocolRequest):
    """
    Discover protocol from binary data.

    Args:
        request: Protocol discovery request

    Returns:
        Protocol type, confidence, and detected fields

    Raises:
        HTTPException: If discovery fails
    """
    try:
        # Implementation
        return ProtocolResponse(
            protocol_type="HTTP",
            confidence=0.95,
            fields=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Adding gRPC Service

```python
# ai_engine/api/grpc_server.py
import grpc
from concurrent import futures
from proto import qbitel_pb2, qbitel_pb2_grpc

class QbitelServicer(qbitel_pb2_grpc.QbitelServiceServicer):

    async def DiscoverProtocol(self, request, context):
        """gRPC method for protocol discovery."""
        try:
            # Implementation
            return qbitel_pb2.ProtocolResponse(
                protocol_type="HTTP",
                confidence=0.95
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return qbitel_pb2.ProtocolResponse()
```

## Contributing Guidelines

### Pull Request Process

1. **Fork the repository**
2. **Create feature branch** from `develop`
3. **Write code** following style guide
4. **Add tests** (aim for 80%+ coverage)
5. **Update documentation** if needed
6. **Run all tests** and quality checks
7. **Submit pull request** with clear description

### Pull Request Checklist

- [ ] Code follows project style guide
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] No merge conflicts
- [ ] CI/CD pipeline passes

### Code Review Guidelines

Reviewers will check:
- Code quality and readability
- Test coverage and quality
- Security implications
- Performance considerations
- Documentation completeness

### Getting Help

- **Documentation**: Check existing docs
- **Issues**: Search GitHub issues
- **Discussions**: Use GitHub Discussions
- **Email**: developers@qbitel.com

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or via environment:

```bash
export AI_ENGINE_LOG_LEVEL=DEBUG
python -m ai_engine
```

### Using Python Debugger

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use Python 3.7+ breakpoint()
breakpoint()
```

### Remote Debugging (VS Code)

Add to `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: AI Engine",
      "type": "python",
      "request": "launch",
      "module": "ai_engine",
      "console": "integratedTerminal",
      "env": {
        "AI_ENGINE_LOG_LEVEL": "DEBUG"
      }
    }
  ]
}
```

### Profiling

```python
# Using cProfile
python -m cProfile -o profile.stats -m ai_engine

# Analyze with snakeviz
pip install snakeviz
snakeviz profile.stats
```

## Performance Optimization

### Benchmarking

```python
# ai_engine/tests/performance/test_benchmarks.py
import pytest
from ai_engine.discovery import PCFGInference

@pytest.mark.benchmark
def test_protocol_discovery_performance(benchmark):
    """Benchmark protocol discovery."""
    engine = PCFGInference()
    samples = [b"GET /api HTTP/1.1"] * 100

    result = benchmark(engine.infer_grammar, samples)
    assert result is not None
```

### Profiling Async Code

```python
import asyncio
from ai_engine.core.engine import AIEngine

async def profile_async():
    engine = AIEngine()
    await engine.initialize()
    # Profile async operations

asyncio.run(profile_async())
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def discover_protocol(data: bytes, confidence_threshold: float = 0.7) -> dict:
    """
    Discover protocol from binary data.

    Args:
        data: Binary protocol data to analyze
        confidence_threshold: Minimum confidence score (0.0-1.0)

    Returns:
        Dictionary containing:
            - protocol_type: Detected protocol name
            - confidence: Confidence score (0.0-1.0)
            - fields: List of detected fields

    Raises:
        ValueError: If data is empty or invalid
        RuntimeError: If discovery engine is not initialized

    Example:
        >>> data = b"GET /api HTTP/1.1"
        >>> result = discover_protocol(data)
        >>> print(result['protocol_type'])
        'HTTP'
    """
    pass
```

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

**Happy Coding!** 🚀

For questions or support, reach out via:
- GitHub Issues: https://github.com/yazhsab/qbitel-bridge/issues
- Email: developers@qbitel.com

**Last Updated**: 2025-02-08
**Version**: 2.0


---

# ============================================================
# FILE 6: DEPLOYMENT.md (Deployment Guide)
# ============================================================

# QBITEL Bridge - Deployment Guide

This guide provides comprehensive instructions for deploying QBITEL Bridge in development, staging, and production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Deployment Options](#deployment-options)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Production Configuration](#production-configuration)
- [Monitoring & Observability](#monitoring--observability)
- [Security Configuration](#security-configuration)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 20GB
- OS: Linux (Ubuntu 20.04+), macOS, Windows with WSL2

**Recommended for Production:**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 50GB+ SSD
- OS: Linux (Ubuntu 20.04+ or RHEL 8+)
- GPU: Optional (NVIDIA CUDA for AI training)

### Software Prerequisites

```bash
# Required
- Python 3.10+
- Rust 1.70+ (with cargo)
- Go 1.21+
- Node.js 18+ (with npm)
- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+ (for production)
- kubectl 1.24+
- Helm 3.8+ (for Helm-based deployment)
- git

# Optional
- NVIDIA Docker (for GPU support)
- Ollama (for on-premise LLM inference)
```

### Network Requirements

- **Outbound Internet**: For pulling Docker images, Python packages
- **Ports**:
  - 8000: REST API
  - 50051: gRPC API
  - 9090: Prometheus metrics
  - 3000: Grafana dashboards (development)
  - 5000: MLflow (development)

## Deployment Options

QBITEL Bridge supports multiple deployment models:

1. **Local Development** - Build from source (Python + Rust + Go)
2. **Docker Compose** - Multi-container development
3. **Kubernetes (Helm)** - Production cloud-native deployment
4. **Air-Gapped** - Fully offline with on-premise LLM

## Local Development

### Quick Start (All Components)

```bash
# 1. Clone the repository
git clone https://github.com/yazhsab/qbitel-bridge.git
cd qbitel-bridge

# 2. Build everything
make build

# 3. Run all tests
make test
```

### Python AI Engine

```bash
# Create Python virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r ai_engine/requirements.txt

# Run tests
pytest ai_engine/tests/ -v --tb=short

# Start the AI Engine
python -m ai_engine

# Verify the API is running
curl http://localhost:8000/health
```

### Rust Data Plane

```bash
cd rust/dataplane
cargo build --locked
cargo test
```

### Go Control Plane

```bash
# Build all Go services
cd go/controlplane && go build -trimpath -o ../../dist/controlplane ./cmd/controlplane
cd go/mgmtapi && go build -trimpath -o ../../dist/mgmtapi ./cmd/mgmtapi

# Run control plane
./dist/controlplane
```

### UI Console

```bash
cd ui/console
npm install
npm run dev
# Available at http://localhost:3000
```

### Configuration

Create a local configuration file:

```bash
# config/local.yaml
rest_host: "127.0.0.1"
rest_port: 8000
grpc_port: 50051
log_level: "DEBUG"
enable_observability: true
device: "cpu"  # or "cuda" if GPU available
```

Run with custom configuration:

```bash
python -m ai_engine --config config/local.yaml
```

### Development Mode

Enable hot-reloading for development:

```bash
python -m ai_engine --development --reload
```

## Docker Deployment

### Build Docker Images

```bash
# Build xDS Server
docker build -f docker/xds-server/Dockerfile -t qbitel/xds-server:latest .

# Build Admission Webhook
docker build -f docker/admission-webhook/Dockerfile -t qbitel/admission-webhook:latest .

# Build Kafka Producer
docker build -f docker/kafka-producer/Dockerfile -t qbitel/kafka-producer:latest .
```

### Docker Compose Deployment

The Docker Compose setup includes all necessary services:

```bash
cd docker

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f qbitel-engine

# Check service status
docker-compose ps

# Stop all services
docker-compose down
```

### Docker Compose Services

```yaml
services:
  - qbitel-engine: AI Engine (REST + gRPC)
  - mlflow: Model registry and tracking
  - prometheus: Metrics collection
  - grafana: Dashboards and visualization
  - kafka: Event streaming (optional)
  - postgres: Database for MLflow
```

### Access Services

- **AI Engine API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Grafana**: http://localhost:3000 (admin/qbitel-admin)
- **Prometheus**: http://localhost:9090

## Helm Deployment (Recommended for Production)

The recommended way to deploy QBITEL Bridge to Kubernetes is via Helm chart.

### Quick Deploy

```bash
helm install qbitel-bridge ./helm/qbitel-bridge \
  --namespace qbitel-bridge \
  --create-namespace \
  --wait
```

### Production Deploy

```bash
helm install qbitel-bridge ./helm/qbitel-bridge \
  --namespace qbitel-bridge \
  --create-namespace \
  --set xdsServer.replicaCount=5 \
  --set admissionWebhook.replicaCount=5 \
  --set xdsServer.resources.requests.memory=1Gi \
  --set monitoring.enabled=true
```

### Custom Values

```bash
# See all configurable values
helm show values ./helm/qbitel-bridge

# Deploy with custom values file
helm install qbitel-bridge ./helm/qbitel-bridge \
  --namespace qbitel-bridge \
  --create-namespace \
  -f my-values.yaml
```

See [helm/qbitel-bridge/README.md](helm/qbitel-bridge/README.md) for full Helm chart documentation.

## Air-Gapped Deployment

For environments with no internet connectivity:

```bash
# 1. On a connected machine: pull models and images
ollama pull llama3.2:8b
ollama pull llama3.2:70b
docker save qbitel/controlplane:latest > controlplane.tar
docker save qbitel/mgmtapi:latest > mgmtapi.tar

# 2. Transfer to air-gapped environment (USB, secure transfer)

# 3. On the air-gapped machine: load images
docker load < controlplane.tar
docker load < mgmtapi.tar

# 4. Configure for air-gapped mode
export QBITEL_LLM_PROVIDER=ollama
export QBITEL_LLM_ENDPOINT=http://localhost:11434
export QBITEL_AIRGAPPED_MODE=true
export QBITEL_DISABLE_CLOUD_LLMS=true

# 5. Deploy
python -m ai_engine --airgapped
```

Air-gapped features:
- All LLM inference runs locally (Ollama)
- No internet connectivity required after initial setup
- All data stays within your infrastructure
- No API keys or cloud accounts needed
- Compliant with strictest data residency regulations

## Kubernetes Deployment (Manual)

For fine-grained control over individual components:

### Prerequisites

```bash
# Verify Kubernetes cluster access
kubectl cluster-info
kubectl get nodes

# Create namespaces
kubectl create namespace qbitel-service-mesh
kubectl create namespace qbitel-container-security
kubectl create namespace qbitel-monitoring
```

### Service Mesh Deployment

Deploy xDS Server and Istio integration:

```bash
# 1. Apply namespace and RBAC
kubectl apply -f kubernetes/service-mesh/namespace.yaml
kubectl apply -f kubernetes/service-mesh/rbac.yaml

# 2. Deploy xDS Server
kubectl apply -f kubernetes/service-mesh/xds-server-deployment.yaml

# 3. Verify deployment
kubectl get pods -n qbitel-service-mesh
kubectl logs -n qbitel-service-mesh deployment/xds-server -f

# 4. Check service
kubectl get svc -n qbitel-service-mesh
```

### Container Security Deployment

Deploy admission webhook and security components:

```bash
# 1. Generate TLS certificates for webhook
./scripts/generate-webhook-certs.sh

# 2. Deploy admission webhook
kubectl apply -f kubernetes/container-security/admission-webhook-deployment.yaml

# 3. Verify webhook is running
kubectl get pods -n qbitel-container-security
kubectl get validatingwebhookconfigurations

# 4. Test webhook (try deploying a test pod)
kubectl run test-pod --image=nginx -n default
kubectl describe pod test-pod -n default
```

### Monitoring Stack Deployment

```bash
# Deploy Prometheus
kubectl apply -f kubernetes/monitoring/prometheus-deployment.yaml

# Deploy Grafana
kubectl apply -f kubernetes/monitoring/grafana-deployment.yaml

# Access Grafana
kubectl port-forward -n qbitel-monitoring svc/grafana 3000:3000

# Import dashboards from kubernetes/monitoring/dashboards/
```

### Verify Deployment

```bash
# Check all pods
kubectl get pods --all-namespaces | grep qbitel

# Check services
kubectl get svc --all-namespaces | grep qbitel

# Check resource usage
kubectl top pods -n qbitel-service-mesh
kubectl top pods -n qbitel-container-security

# View logs
kubectl logs -n qbitel-service-mesh deployment/xds-server --tail=100
kubectl logs -n qbitel-container-security deployment/admission-webhook --tail=100
```

## Production Configuration

### Environment Variables

```bash
# AI Engine Configuration
export AI_ENGINE_LOG_LEVEL=INFO
export AI_ENGINE_WORKERS=4
export AI_ENGINE_MAX_REQUESTS=1000
export AI_ENGINE_TIMEOUT=300

# Security
export API_KEY_SECRET=<secure-random-string>
export TLS_CERT_PATH=/etc/qbitel/certs/tls.crt
export TLS_KEY_PATH=/etc/qbitel/certs/tls.key

# MLflow
export MLFLOW_TRACKING_URI=https://mlflow.production.example.com
export MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com

# Monitoring
export PROMETHEUS_METRICS_PORT=9090
export ENABLE_TRACING=true
export JAEGER_AGENT_HOST=jaeger-agent.monitoring.svc.cluster.local
```

### Resource Limits

Configure resource requests and limits in Kubernetes:

```yaml
# kubernetes/production/xds-server-deployment.yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

### Auto-Scaling

Enable Horizontal Pod Autoscaler:

```bash
# HPA for xDS Server
kubectl autoscale deployment xds-server \
  -n qbitel-service-mesh \
  --cpu-percent=70 \
  --min=3 \
  --max=10

# HPA for Admission Webhook
kubectl autoscale deployment admission-webhook \
  -n qbitel-container-security \
  --cpu-percent=70 \
  --min=3 \
  --max=10

# Verify HPA
kubectl get hpa --all-namespaces
```

### Persistent Storage

Configure persistent volumes for data and models:

```yaml
# kubernetes/production/persistent-volumes.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: qbitel-models-pvc
  namespace: qbitel-service-mesh
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
```

### High Availability

Ensure HA configuration:

```yaml
# Pod Disruption Budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: xds-server-pdb
  namespace: qbitel-service-mesh
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: xds-server
```

## Monitoring & Observability

### Prometheus Metrics

Key metrics exposed by QBITEL:

```bash
# Check metrics endpoint
curl http://localhost:9090/metrics

# Key metrics:
# - qbitel_requests_total
# - qbitel_request_duration_seconds
# - qbitel_protocol_discovery_accuracy
# - qbitel_anomaly_detection_alerts
# - qbitel_encryption_operations_total
```

### Grafana Dashboards

Import pre-built dashboards:

```bash
# Located in: kubernetes/monitoring/dashboards/
# - qbitel-overview.json
# - qbitel-security.json
# - qbitel-performance.json
# - qbitel-kubernetes.json
```

### Logging

Configure structured logging:

```yaml
# config/production.yaml
logging:
  level: INFO
  format: json
  output: stdout
  include_trace_id: true
  include_span_id: true
```

### Distributed Tracing

Enable OpenTelemetry tracing:

```bash
# Set Jaeger endpoint
export OTEL_EXPORTER_JAEGER_ENDPOINT=http://jaeger-collector:14268/api/traces

# View traces in Jaeger UI
kubectl port-forward -n monitoring svc/jaeger-query 16686:16686
# Open http://localhost:16686
```

### Health Checks

```bash
# Liveness probe
curl http://localhost:8000/health

# Readiness probe
curl http://localhost:8000/ready

# Detailed status
curl http://localhost:8000/status
```

## Security Configuration

### TLS/SSL Configuration

Generate certificates:

```bash
# Generate self-signed certificates (development)
openssl req -x509 -newkey rsa:4096 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes \
  -subj "/CN=qbitel.local"

# For production, use cert-manager or your PKI
```

Configure TLS in deployment:

```yaml
# kubernetes/production/tls-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: qbitel-tls
  namespace: qbitel-service-mesh
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-cert>
  tls.key: <base64-encoded-key>
```

### RBAC Configuration

Kubernetes RBAC is defined in:
- `kubernetes/service-mesh/rbac.yaml`
- `kubernetes/container-security/rbac.yaml`

### Network Policies

Restrict pod-to-pod communication:

```yaml
# kubernetes/production/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: qbitel-network-policy
  namespace: qbitel-service-mesh
spec:
  podSelector:
    matchLabels:
      app: xds-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
    ports:
    - protocol: TCP
      port: 50051
```

### Secret Management

Use Kubernetes secrets or external secret managers:

```bash
# Create secret for API keys
kubectl create secret generic qbitel-api-keys \
  -n qbitel-service-mesh \
  --from-literal=api-key=<your-secure-api-key>

# Use AWS Secrets Manager (example)
kubectl create secret generic aws-secrets \
  -n qbitel-service-mesh \
  --from-literal=aws-access-key-id=<id> \
  --from-literal=aws-secret-access-key=<key>
```

## Troubleshooting

### Common Issues

#### 1. Pod Fails to Start

```bash
# Check pod status
kubectl describe pod <pod-name> -n qbitel-service-mesh

# Check logs
kubectl logs <pod-name> -n qbitel-service-mesh

# Common causes:
# - Image pull errors: Check imagePullSecrets
# - Resource constraints: Check resource limits
# - Configuration errors: Verify ConfigMaps
```

#### 2. High Memory Usage

```bash
# Check resource usage
kubectl top pod <pod-name> -n qbitel-service-mesh

# Reduce batch size in configuration
# Edit ConfigMap:
kubectl edit configmap qbitel-config -n qbitel-service-mesh

# Set: batch_size: 16 (reduce from 32)
```

#### 3. xDS Server Not Connecting to Envoy

```bash
# Check xDS server logs
kubectl logs deployment/xds-server -n qbitel-service-mesh | grep -i error

# Verify gRPC port is accessible
kubectl get svc xds-server -n qbitel-service-mesh

# Test connection
grpcurl -plaintext xds-server.qbitel-service-mesh:50051 list
```

#### 4. Admission Webhook Blocking Pods

```bash
# Check webhook logs
kubectl logs deployment/admission-webhook -n qbitel-container-security

# Temporarily disable webhook for debugging
kubectl delete validatingwebhookconfigurations qbitel-admission-webhook

# Re-enable after fixing
kubectl apply -f kubernetes/container-security/admission-webhook-deployment.yaml
```

#### 5. eBPF Monitor Not Working

```bash
# eBPF requires Linux kernel 4.4+
uname -r

# Install BCC tools (Ubuntu/Debian)
apt-get install -y bpfcc-tools linux-headers-$(uname -r)

# Check if BPF is available
ls /sys/kernel/debug/tracing/

# Run with elevated privileges
kubectl patch daemonset ebpf-monitor -n qbitel-container-security \
  -p '{"spec":{"template":{"spec":{"hostPID":true,"hostNetwork":true}}}}'
```

### Performance Tuning

```bash
# 1. Enable connection pooling
export MAX_CONNECTIONS=100
export CONNECTION_TIMEOUT=30

# 2. Adjust worker processes
export AI_ENGINE_WORKERS=4

# 3. Enable caching
export ENABLE_CACHE=true
export CACHE_TTL=300

# 4. Tune Kafka producer
export KAFKA_BATCH_SIZE=16384
export KAFKA_LINGER_MS=10
export KAFKA_COMPRESSION_TYPE=snappy
```

### Debugging Tips

```bash
# Enable debug logging
kubectl set env deployment/xds-server -n qbitel-service-mesh LOG_LEVEL=DEBUG

# Port-forward for local access
kubectl port-forward -n qbitel-service-mesh svc/xds-server 50051:50051

# Execute commands in pod
kubectl exec -it <pod-name> -n qbitel-service-mesh -- /bin/bash

# Copy files from pod
kubectl cp qbitel-service-mesh/<pod-name>:/app/logs ./logs

# Watch pod events
kubectl get events -n qbitel-service-mesh --watch
```

## Backup and Recovery

### Backup Strategy

```bash
# 1. Backup Kubernetes resources
kubectl get all -n qbitel-service-mesh -o yaml > backup-service-mesh.yaml
kubectl get all -n qbitel-container-security -o yaml > backup-container-security.yaml

# 2. Backup ConfigMaps and Secrets
kubectl get configmaps -n qbitel-service-mesh -o yaml > backup-configmaps.yaml
kubectl get secrets -n qbitel-service-mesh -o yaml > backup-secrets.yaml

# 3. Backup persistent volumes
kubectl get pvc -n qbitel-service-mesh -o yaml > backup-pvc.yaml

# 4. Backup MLflow models (if using S3/object storage)
# Configure periodic snapshots via cloud provider
```

### Recovery Procedures

```bash
# 1. Restore Kubernetes resources
kubectl apply -f backup-service-mesh.yaml
kubectl apply -f backup-container-security.yaml

# 2. Restore ConfigMaps and Secrets
kubectl apply -f backup-configmaps.yaml
kubectl apply -f backup-secrets.yaml

# 3. Verify restoration
kubectl get pods --all-namespaces | grep qbitel
```

## Upgrade Procedures

```bash
# 1. Backup current deployment
kubectl get all -n qbitel-service-mesh -o yaml > pre-upgrade-backup.yaml

# 2. Update Docker images
kubectl set image deployment/xds-server \
  -n qbitel-service-mesh \
  xds-server=qbitel/xds-server:v2.0.0

# 3. Monitor rollout
kubectl rollout status deployment/xds-server -n qbitel-service-mesh

# 4. Rollback if needed
kubectl rollout undo deployment/xds-server -n qbitel-service-mesh

# 5. Verify functionality
kubectl exec -it <pod-name> -n qbitel-service-mesh -- python -c "import ai_engine; print(ai_engine.__version__)"
```

## Support

For deployment issues:
- Check logs: `kubectl logs <pod-name>`
- Review events: `kubectl get events`
- GitHub Issues: https://github.com/yazhsab/qbitel-bridge/issues
- Enterprise Support: enterprise@qbitel.com

---

**Last Updated**: 2025-02-08
**Version**: 2.0
**Status**: Production Deployment Guide


---

# ============================================================
# FILE 7: QUICKSTART.md (Quick Start Guide)
# ============================================================

# QBITEL Bridge - Quick Start Guide

Get QBITEL Bridge up and running in **10 minutes**.

## Prerequisites

- **Kubernetes cluster** (1.24+) OR Docker Desktop with Kubernetes enabled
- **Helm 3.8+** installed
- **kubectl** configured
- **4GB RAM** minimum (8GB+ recommended)

## Option 1: Helm Deployment (Recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/yazhsab/qbitel-bridge.git
cd qbitel-bridge
```

### Step 2: Deploy with Helm

```bash
# Deploy to Kubernetes
helm install qbitel ./helm/qbitel-bridge \
  --namespace qbitel-service-mesh \
  --create-namespace \
  --wait

# Check deployment status
kubectl get pods -n qbitel-service-mesh
```

### Step 3: Access Services

```bash
# Access AI Engine API
kubectl port-forward -n qbitel-service-mesh svc/qbitel-engine 8000:8000

# Open in browser
# http://localhost:8000/docs
```

### Step 4: Verify Quantum Cryptography

```bash
# Run validation tests
kubectl exec -n qbitel-service-mesh deployment/qbitel-xds-server -- \
  python -c "from ai_engine.cloud_native.service_mesh.istio.qkd_certificate_manager import QuantumCertificateManager; \
  mgr = QuantumCertificateManager(); \
  print('✓ Quantum Crypto: VALIDATED')"
```

## Option 2: Automated Script Deployment

### Linux/macOS

```bash
# Make script executable
chmod +x scripts/deploy.sh

# Deploy
./scripts/deploy.sh \
  --environment development \
  --namespace qbitel-service-mesh
```

### Windows (PowerShell)

```powershell
# Run deployment script
.\scripts\deploy.ps1 `
  -Environment development `
  -Namespace qbitel-service-mesh
```

## Option 3: Local Development (Python)

### Step 1: Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv  # Requires Python 3.10+
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r ai_engine/requirements.txt
```

### Step 2: Run Tests

```bash
# Run all tests
pytest ai_engine/tests/ -v

# Run cloud-native tests only
pytest ai_engine/tests/cloud_native/ -v
```

### Step 3: Start AI Engine

```bash
# Start the AI Engine locally
python -m ai_engine

# Access at http://localhost:8000
```

## Verify Installation

### Check All Pods Running

```bash
kubectl get pods -n qbitel-service-mesh

# Expected output:
# NAME                                    READY   STATUS    RESTARTS
# qbitel-xds-server-xxx                   1/1     Running   0
# qbitel-admission-webhook-xxx            1/1     Running   0
# qbitel-engine-xxx                    1/1     Running   0
```

### Test xDS Server Health

```bash
kubectl exec -n qbitel-service-mesh deployment/qbitel-xds-server -- \
  curl -f http://localhost:8081/healthz

# Expected: OK
```

### Access Grafana Dashboards

```bash
kubectl port-forward -n qbitel-service-mesh svc/grafana 3000:3000

# Open http://localhost:3000
# Default credentials: admin / qbitel-admin
```

## Enable Admission Webhook for Your Namespace

```bash
# Label your namespace to enable admission control
kubectl label namespace <your-namespace> qbitel.ai/webhook=enabled

# Now all pod deployments in this namespace will be validated
```

## Test Quantum-Safe Encryption

```bash
# Deploy a test pod
kubectl run test-app --image=nginx -n <your-namespace>

# Check if it was validated by admission webhook
kubectl describe pod test-app -n <your-namespace> | grep -i qbitel
```

## Run Performance Benchmarks

```bash
# Run performance tests
pytest ai_engine/tests/performance/test_benchmarks.py -v --benchmark-only

# Expected results:
# - Encryption: < 1ms latency
# - Throughput: 100K+ msg/s
# - All benchmarks PASS
```

## Configuration Options

### Development Environment

```bash
helm install qbitel ./helm/qbitel-bridge \
  --namespace qbitel-dev \
  --create-namespace \
  --set xdsServer.replicaCount=1 \
  --set admissionWebhook.replicaCount=1 \
  --set monitoring.enabled=false
```

### Production Environment

```bash
helm install qbitel ./helm/qbitel-bridge \
  --namespace qbitel-service-mesh \
  --create-namespace \
  --set xdsServer.replicaCount=5 \
  --set admissionWebhook.replicaCount=5 \
  --set xdsServer.resources.requests.memory=1Gi \
  --set monitoring.enabled=true
```

### Custom Values File

```bash
# Create custom values
cat > my-values.yaml <<EOF
xdsServer:
  replicaCount: 3
  autoscaling:
    minReplicas: 3
    maxReplicas: 10

admissionWebhook:
  config:
    maxCriticalVulnerabilities: 0
    requireQuantumSafe: true

monitoring:
  enabled: true
  prometheus:
    retention: 30d
EOF

# Deploy with custom values
helm install qbitel ./helm/qbitel-bridge \
  --namespace qbitel-service-mesh \
  --create-namespace \
  -f my-values.yaml
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n qbitel-service-mesh

# Check logs
kubectl logs <pod-name> -n qbitel-service-mesh

# Common issues:
# - Insufficient resources: Increase node memory/CPU
# - Image pull errors: Check network connectivity
```

### Admission Webhook Blocking Deployments

```bash
# Temporarily disable webhook
kubectl delete validatingwebhookconfigurations qbitel-validating-webhook

# Deploy your pod
kubectl apply -f your-pod.yaml

# Re-enable webhook
kubectl apply -f kubernetes/container-security/admission-webhook-deployment.yaml
```

### Check Quantum Crypto Implementation

```bash
# Verify Kyber-1024 keys
kubectl exec -n qbitel-service-mesh deployment/qbitel-xds-server -- \
  python -c "
from ai_engine.cloud_native.service_mesh.istio.qkd_certificate_manager import QuantumCertificateManager, CertificateAlgorithm
mgr = QuantumCertificateManager()
priv, pub = mgr._generate_key_pair(CertificateAlgorithm.KYBER_1024)
assert len(pub) == 1568, f'Expected 1568, got {len(pub)}'
assert len(priv) == 3168, f'Expected 3168, got {len(priv)}'
print('✓ NIST Level 5 Quantum Crypto: VALIDATED')
"
```

## Uninstall

```bash
# Uninstall Helm release
helm uninstall qbitel --namespace qbitel-service-mesh

# Delete namespace (optional)
kubectl delete namespace qbitel-service-mesh
```

## Next Steps

1. **Read Architecture**: [architecture.md](architecture.md)
2. **Production Deployment**: [DEPLOYMENT.md](DEPLOYMENT.md)
3. **Developer Guide**: [DEVELOPMENT.md](DEVELOPMENT.md)
4. **API Documentation**: http://localhost:8000/docs (after port-forward)

## Support

- **Issues**: [GitHub Issues](https://github.com/yazhsab/qbitel-bridge/issues)
- **Documentation**: See [README.md](README.md)
- **Enterprise Support**: enterprise@qbitel.com

---

**You're now running NIST Level 5 Quantum-Safe Security!** 🔒

For detailed configuration options, see [helm/qbitel-bridge/](helm/qbitel-bridge/).


---

# ============================================================
# FILE 8: docs/IMPLEMENTATION_PLAN.md (Implementation Plan)
# ============================================================

# QBITEL Implementation Plan

## Executive Summary

This document outlines a phased implementation plan to focus, harden, and optimize the QBITEL platform based on the comprehensive architecture review. The plan spans 12 weeks across 4 phases.

---

## Phase 1: Security Hardening (Weeks 1-2)

### Goal: Address critical security gaps before any customer deployment

---

### 1.1 API Rate Limiting

**Priority:** CRITICAL
**Effort:** 2 days
**Owner:** Backend Team

#### Implementation Steps:

1. **Install slowapi package**
   ```bash
   pip install slowapi
   ```

2. **Create rate limiting middleware** at `ai_engine/api/middleware/rate_limiter.py`:
   ```python
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   from slowapi.errors import RateLimitExceeded
   from fastapi import Request

   limiter = Limiter(key_func=get_remote_address)

   # Rate limit configurations by endpoint type
   RATE_LIMITS = {
       "discovery": "10/minute",      # Heavy compute
       "detection": "30/minute",      # Medium compute
       "copilot": "60/minute",        # LLM calls
       "health": "300/minute",        # Lightweight
       "default": "100/minute"        # Standard endpoints
   }
   ```

3. **Apply to REST API** in `ai_engine/api/rest.py`:
   ```python
   from ai_engine.api.middleware.rate_limiter import limiter, RATE_LIMITS

   app.state.limiter = limiter
   app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

   @app.post("/api/v1/discover/protocol")
   @limiter.limit(RATE_LIMITS["discovery"])
   async def discover_protocol(request: Request, ...):
       ...
   ```

4. **Add Redis backend for distributed rate limiting**:
   ```python
   from slowapi import Limiter
   from slowapi.util import get_remote_address
   import redis

   redis_client = redis.from_url(settings.REDIS_URL)
   limiter = Limiter(
       key_func=get_remote_address,
       storage_uri=settings.REDIS_URL
   )
   ```

#### Acceptance Criteria:
- [ ] All endpoints have rate limits applied
- [ ] Rate limit headers returned (X-RateLimit-Limit, X-RateLimit-Remaining)
- [ ] 429 Too Many Requests returned when exceeded
- [ ] Rate limits persist across restarts (Redis backend)
- [ ] Prometheus metrics for rate limit hits

---

### 1.2 Input Size Validation

**Priority:** CRITICAL
**Effort:** 1 day
**Owner:** Backend Team

#### Implementation Steps:

1. **Create validation constants** at `ai_engine/core/constants.py`:
   ```python
   # Input size limits
   MAX_PROTOCOL_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB
   MAX_FIELD_DETECTION_SIZE = 1 * 1024 * 1024    # 1 MB
   MAX_BATCH_SIZE = 100                           # Max messages per batch
   MAX_COPILOT_QUERY_LENGTH = 10000              # Characters
   MAX_FILE_UPLOAD_SIZE = 50 * 1024 * 1024       # 50 MB
   ```

2. **Create validation decorator** at `ai_engine/api/middleware/validators.py`:
   ```python
   from functools import wraps
   from fastapi import HTTPException, status
   from ai_engine.core.constants import MAX_PROTOCOL_MESSAGE_SIZE

   def validate_input_size(max_size: int):
       def decorator(func):
           @wraps(func)
           async def wrapper(*args, **kwargs):
               request = kwargs.get('request') or args[0]
               body = await request.body()
               if len(body) > max_size:
                   raise HTTPException(
                       status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                       detail=f"Request body exceeds maximum size of {max_size} bytes"
                   )
               return await func(*args, **kwargs)
           return wrapper
       return decorator
   ```

3. **Apply to endpoints**:
   ```python
   @app.post("/api/v1/discover/protocol")
   @validate_input_size(MAX_PROTOCOL_MESSAGE_SIZE)
   async def discover_protocol(...):
       ...
   ```

4. **Add global request size limit** in FastAPI:
   ```python
   from fastapi import FastAPI
   from starlette.middleware.trustedhost import TrustedHostMiddleware

   app = FastAPI()

   # Limit request body size globally
   @app.middleware("http")
   async def limit_request_size(request: Request, call_next):
       content_length = request.headers.get("content-length")
       if content_length and int(content_length) > MAX_FILE_UPLOAD_SIZE:
           return JSONResponse(
               status_code=413,
               content={"detail": "Request too large"}
           )
       return await call_next(request)
   ```

#### Acceptance Criteria:
- [ ] All endpoints validate input size
- [ ] Clear error messages returned for oversized requests
- [ ] Unit tests for boundary conditions
- [ ] Documentation updated with size limits

---

### 1.3 Standardized Error Handling

**Priority:** HIGH
**Effort:** 3 days
**Owner:** Backend Team

#### Implementation Steps:

1. **Create exception hierarchy** at `ai_engine/core/exceptions.py`:
   ```python
   from typing import Optional, Dict, Any
   from enum import Enum

   class ErrorCode(Enum):
       # Discovery errors (1000-1999)
       DISCOVERY_FAILED = 1000
       DISCOVERY_TIMEOUT = 1001
       INVALID_PROTOCOL_DATA = 1002
       GRAMMAR_LEARNING_FAILED = 1003
       PARSER_GENERATION_FAILED = 1004

       # Detection errors (2000-2999)
       FIELD_DETECTION_FAILED = 2000
       MODEL_NOT_LOADED = 2001
       INVALID_MESSAGE_FORMAT = 2002

       # LLM errors (3000-3999)
       LLM_UNAVAILABLE = 3000
       LLM_TIMEOUT = 3001
       LLM_RATE_LIMITED = 3002
       LLM_CONTEXT_TOO_LONG = 3003

       # Authentication errors (4000-4999)
       UNAUTHORIZED = 4000
       INVALID_TOKEN = 4001
       TOKEN_EXPIRED = 4002
       INSUFFICIENT_PERMISSIONS = 4003

       # Validation errors (5000-5999)
       VALIDATION_ERROR = 5000
       INVALID_INPUT = 5001
       MISSING_REQUIRED_FIELD = 5002

       # System errors (9000-9999)
       INTERNAL_ERROR = 9000
       DATABASE_ERROR = 9001
       CACHE_ERROR = 9002
       EXTERNAL_SERVICE_ERROR = 9003


   class QbitelError(Exception):
       """Base exception for all QBITEL errors."""

       def __init__(
           self,
           message: str,
           error_code: ErrorCode,
           details: Optional[Dict[str, Any]] = None,
           cause: Optional[Exception] = None
       ):
           super().__init__(message)
           self.message = message
           self.error_code = error_code
           self.details = details or {}
           self.cause = cause

       def to_dict(self) -> Dict[str, Any]:
           return {
               "error_code": self.error_code.value,
               "error_name": self.error_code.name,
               "message": self.message,
               "details": self.details
           }


   class DiscoveryError(QbitelError):
       """Errors during protocol discovery."""
       pass


   class DetectionError(QbitelError):
       """Errors during field detection."""
       pass


   class LLMError(QbitelError):
       """Errors from LLM services."""
       pass


   class AuthenticationError(QbitelError):
       """Authentication and authorization errors."""
       pass


   class ValidationError(QbitelError):
       """Input validation errors."""
       pass
   ```

2. **Create global exception handler** at `ai_engine/api/middleware/error_handler.py`:
   ```python
   from fastapi import Request, status
   from fastapi.responses import JSONResponse
   from ai_engine.core.exceptions import QbitelError, ErrorCode
   import logging
   import traceback

   logger = logging.getLogger(__name__)

   async def qbitel_exception_handler(request: Request, exc: QbitelError):
       """Handle all QBITEL-specific exceptions."""
       logger.error(
           f"QbitelError: {exc.error_code.name} - {exc.message}",
           extra={
               "error_code": exc.error_code.value,
               "details": exc.details,
               "path": request.url.path
           }
       )

       # Map error codes to HTTP status codes
       status_map = {
           ErrorCode.UNAUTHORIZED: status.HTTP_401_UNAUTHORIZED,
           ErrorCode.INVALID_TOKEN: status.HTTP_401_UNAUTHORIZED,
           ErrorCode.INSUFFICIENT_PERMISSIONS: status.HTTP_403_FORBIDDEN,
           ErrorCode.VALIDATION_ERROR: status.HTTP_400_BAD_REQUEST,
           ErrorCode.INVALID_INPUT: status.HTTP_400_BAD_REQUEST,
           ErrorCode.LLM_RATE_LIMITED: status.HTTP_429_TOO_MANY_REQUESTS,
       }

       http_status = status_map.get(
           exc.error_code,
           status.HTTP_500_INTERNAL_SERVER_ERROR
       )

       return JSONResponse(
           status_code=http_status,
           content=exc.to_dict()
       )


   async def generic_exception_handler(request: Request, exc: Exception):
       """Handle unexpected exceptions."""
       logger.exception(
           f"Unhandled exception: {str(exc)}",
           extra={"path": request.url.path}
       )

       return JSONResponse(
           status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
           content={
               "error_code": ErrorCode.INTERNAL_ERROR.value,
               "error_name": "INTERNAL_ERROR",
               "message": "An unexpected error occurred",
               "details": {}
           }
       )
   ```

3. **Register handlers in FastAPI**:
   ```python
   from ai_engine.core.exceptions import QbitelError
   from ai_engine.api.middleware.error_handler import (
       qbitel_exception_handler,
       generic_exception_handler
   )

   app.add_exception_handler(QbitelError, qbitel_exception_handler)
   app.add_exception_handler(Exception, generic_exception_handler)
   ```

4. **Refactor existing code** to use new exceptions:
   ```python
   # Before (inconsistent)
   raise ValueError("Invalid protocol data")

   # After (standardized)
   from ai_engine.core.exceptions import DiscoveryError, ErrorCode

   raise DiscoveryError(
       message="Invalid protocol data: message is empty",
       error_code=ErrorCode.INVALID_PROTOCOL_DATA,
       details={"received_length": 0, "minimum_length": 1}
   )
   ```

#### Files to Refactor:
- [ ] `ai_engine/discovery/protocol_discovery_orchestrator.py`
- [ ] `ai_engine/detection/field_detector.py`
- [ ] `ai_engine/llm/unified_llm_service.py`
- [ ] `ai_engine/copilot/protocol_copilot.py`
- [ ] `ai_engine/legacy/service.py`

#### Acceptance Criteria:
- [ ] All custom exceptions inherit from QbitelError
- [ ] Error codes documented in API specification
- [ ] No generic `ValueError` or `Exception` raises in business logic
- [ ] All errors include correlation_id for tracing
- [ ] Unit tests for each error type

---

## Phase 2: Architecture Simplification (Weeks 3-5)

### Goal: Reduce complexity by removing unused components and splitting large modules

---

### 2.1 Remove Unused LLM Providers

**Priority:** HIGH
**Effort:** 2 days
**Owner:** Backend Team

#### Current State:
- 5 providers configured: Ollama, vLLM, LocalAI, OpenAI, Anthropic
- Only Ollama and Anthropic have active usage

#### Implementation Steps:

1. **Audit current LLM usage**:
   ```bash
   grep -r "vllm\|localai" ai_engine/ --include="*.py"
   ```

2. **Remove vLLM integration**:
   - Delete `ai_engine/llm/providers/vllm_provider.py` (if exists)
   - Remove from `UnifiedLLMService` provider list
   - Remove from `pyproject.toml` dependencies

3. **Remove LocalAI integration**:
   - Delete `ai_engine/llm/providers/localai_provider.py` (if exists)
   - Remove from configuration files
   - Remove from dependencies

4. **Simplify UnifiedLLMService**:
   ```python
   # Before: Complex 5-provider fallback
   PROVIDERS = ["ollama", "vllm", "localai", "openai", "anthropic"]

   # After: Simplified 2-provider fallback
   class LLMProvider(Enum):
       OLLAMA = "ollama"      # Primary (air-gapped)
       ANTHROPIC = "anthropic" # Cloud fallback

   class SimplifiedLLMService:
       def __init__(self):
           self.primary = OllamaProvider()
           self.fallback = AnthropicProvider()

       async def generate(self, prompt: str, **kwargs) -> str:
           try:
               return await self.primary.generate(prompt, **kwargs)
           except LLMError:
               if self.allow_cloud_fallback:
                   return await self.fallback.generate(prompt, **kwargs)
               raise
   ```

5. **Update configuration**:
   ```yaml
   # config/qbitel.yaml
   llm:
     primary_provider: ollama
     fallback_provider: anthropic  # Optional, disabled in air-gapped
     allow_cloud_fallback: false   # Default to air-gapped mode

     ollama:
       base_url: "http://localhost:11434"
       model: "llama3.2:70b"
       timeout: 120

     anthropic:
       model: "claude-sonnet-4-20250514"
       max_tokens: 4096
   ```

#### Acceptance Criteria:
- [ ] Only Ollama and Anthropic providers remain
- [ ] All tests pass with reduced providers
- [ ] Configuration simplified
- [ ] Documentation updated

---

### 2.2 Split Database Models

**Priority:** MEDIUM
**Effort:** 3 days
**Owner:** Backend Team

#### Current State:
`ai_engine/models/database.py` contains 500+ lines with mixed concerns:
- User authentication models
- Audit logging models
- OAuth/SAML providers
- API key management

#### Target Structure:
```
ai_engine/models/
├── __init__.py           # Re-export all models
├── base.py               # Base model class, mixins
├── auth/
│   ├── __init__.py
│   ├── user.py           # User, UserRole
│   ├── session.py        # UserSession
│   ├── api_key.py        # APIKey, APIKeyStatus
│   └── mfa.py            # MFA configurations
├── providers/
│   ├── __init__.py
│   ├── oauth.py          # OAuthProvider
│   └── saml.py           # SAMLProvider
├── audit/
│   ├── __init__.py
│   └── audit_log.py      # AuditLog, AuditAction
└── protocol/
    ├── __init__.py
    ├── discovery.py      # ProtocolProfile, DiscoveryResult
    └── analysis.py       # FieldDetection, AnomalyResult
```

#### Implementation Steps:

1. **Create base module** `ai_engine/models/base.py`:
   ```python
   from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
   from sqlalchemy import DateTime, func
   from datetime import datetime
   from uuid import UUID, uuid4

   class Base(DeclarativeBase):
       pass

   class TimestampMixin:
       created_at: Mapped[datetime] = mapped_column(
           DateTime(timezone=True),
           server_default=func.now()
       )
       updated_at: Mapped[datetime] = mapped_column(
           DateTime(timezone=True),
           server_default=func.now(),
           onupdate=func.now()
       )

   class SoftDeleteMixin:
       deleted_at: Mapped[datetime | None] = mapped_column(
           DateTime(timezone=True),
           nullable=True
       )

       @property
       def is_deleted(self) -> bool:
           return self.deleted_at is not None

   class UUIDPrimaryKeyMixin:
       id: Mapped[UUID] = mapped_column(
           primary_key=True,
           default=uuid4
       )
   ```

2. **Split user models** `ai_engine/models/auth/user.py`:
   ```python
   from enum import Enum
   from sqlalchemy import String, Boolean, Enum as SQLEnum
   from sqlalchemy.orm import Mapped, mapped_column, relationship
   from ai_engine.models.base import Base, TimestampMixin, SoftDeleteMixin, UUIDPrimaryKeyMixin

   class UserRole(str, Enum):
       ADMINISTRATOR = "administrator"
       SECURITY_ANALYST = "security_analyst"
       OPERATOR = "operator"
       VIEWER = "viewer"
       API_USER = "api_user"

   class User(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
       __tablename__ = "users"

       username: Mapped[str] = mapped_column(String(255), unique=True, index=True)
       email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
       password_hash: Mapped[str | None] = mapped_column(String(255))
       role: Mapped[UserRole] = mapped_column(SQLEnum(UserRole), index=True)
       is_active: Mapped[bool] = mapped_column(Boolean, default=True)

       # Relationships
       sessions = relationship("UserSession", back_populates="user")
       api_keys = relationship("APIKey", back_populates="user")
       audit_logs = relationship("AuditLog", back_populates="user")
   ```

3. **Create re-export module** `ai_engine/models/__init__.py`:
   ```python
   from ai_engine.models.base import Base, TimestampMixin, SoftDeleteMixin
   from ai_engine.models.auth.user import User, UserRole
   from ai_engine.models.auth.session import UserSession
   from ai_engine.models.auth.api_key import APIKey, APIKeyStatus
   from ai_engine.models.providers.oauth import OAuthProvider
   from ai_engine.models.providers.saml import SAMLProvider
   from ai_engine.models.audit.audit_log import AuditLog, AuditAction

   __all__ = [
       "Base",
       "User", "UserRole",
       "UserSession",
       "APIKey", "APIKeyStatus",
       "OAuthProvider", "SAMLProvider",
       "AuditLog", "AuditAction",
   ]
   ```

4. **Add missing indexes** for audit_logs:
   ```python
   class AuditLog(Base, UUIDPrimaryKeyMixin):
       __tablename__ = "audit_logs"
       __table_args__ = (
           Index("ix_audit_logs_timestamp_action", "timestamp", "action_type"),
           Index("ix_audit_logs_user_timestamp", "user_id", "timestamp"),
           Index("ix_audit_logs_resource", "resource_type", "resource_id"),
       )
   ```

5. **Generate Alembic migration**:
   ```bash
   alembic revision --autogenerate -m "add_audit_log_indexes"
   alembic upgrade head
   ```

#### Acceptance Criteria:
- [ ] All models split into logical modules
- [ ] Backward-compatible imports via `__init__.py`
- [ ] Missing indexes added
- [ ] All existing tests pass
- [ ] Alembic migration generated and tested

---

### 2.3 Refactor Legacy Whisperer (17K LOC → 4 Services)

**Priority:** HIGH
**Effort:** 5 days
**Owner:** Backend Team

#### Current State:
`ai_engine/legacy/` is a monolithic 17,000+ LOC module handling:
- COBOL code analysis
- Copybook parsing
- Code generation (REST APIs, SDKs)
- Transaction flow mapping
- Modernization recommendations

#### Target Architecture:
```
ai_engine/legacy/
├── __init__.py
├── analyzer/                    # COBOL Analysis Service
│   ├── __init__.py
│   ├── service.py              # COBOLAnalyzerService
│   ├── parser.py               # COBOL syntax parser
│   ├── copybook_parser.py      # Copybook structure extraction
│   ├── data_flow.py            # Data flow analysis
│   └── models.py               # COBOLProgram, Copybook, DataItem
├── mapper/                      # Transaction Mapping Service
│   ├── __init__.py
│   ├── service.py              # TransactionMapperService
│   ├── flow_analyzer.py        # Transaction flow detection
│   ├── dependency_graph.py     # Program dependencies
│   └── models.py               # Transaction, Flow, Dependency
├── generator/                   # Code Generation Service
│   ├── __init__.py
│   ├── service.py              # CodeGeneratorService
│   ├── rest_generator.py       # REST API generation
│   ├── sdk_generator.py        # SDK generation (6 languages)
│   ├── openapi_generator.py    # OpenAPI spec generation
│   └── templates/              # Jinja2 templates
└── orchestrator/               # Orchestration Layer
    ├── __init__.py
    ├── service.py              # LegacyWhispererOrchestrator
    └── workflow.py             # Modernization workflow
```

#### Implementation Steps:

1. **Extract COBOL Analyzer** `ai_engine/legacy/analyzer/service.py`:
   ```python
   from dataclasses import dataclass
   from typing import List, Optional
   from ai_engine.legacy.analyzer.models import COBOLProgram, Copybook
   from ai_engine.legacy.analyzer.parser import COBOLParser
   from ai_engine.legacy.analyzer.copybook_parser import CopybookParser

   @dataclass
   class AnalysisResult:
       program: COBOLProgram
       copybooks: List[Copybook]
       data_items: List[DataItem]
       procedures: List[Procedure]
       complexity_score: float
       analysis_time_ms: int

   class COBOLAnalyzerService:
       """Analyzes COBOL source code and copybooks."""

       def __init__(self):
           self.cobol_parser = COBOLParser()
           self.copybook_parser = CopybookParser()

       async def analyze(
           self,
           source_code: str,
           copybooks: Optional[List[str]] = None
       ) -> AnalysisResult:
           """Analyze COBOL program and extract structure."""
           program = self.cobol_parser.parse(source_code)

           parsed_copybooks = []
           if copybooks:
               for copybook in copybooks:
                   parsed_copybooks.append(
                       self.copybook_parser.parse(copybook)
                   )

           return AnalysisResult(
               program=program,
               copybooks=parsed_copybooks,
               data_items=program.data_division.items,
               procedures=program.procedure_division.sections,
               complexity_score=self._calculate_complexity(program),
               analysis_time_ms=...
           )
   ```

2. **Extract Transaction Mapper** `ai_engine/legacy/mapper/service.py`:
   ```python
   class TransactionMapperService:
       """Maps COBOL transactions to modern API operations."""

       async def map_transactions(
           self,
           analysis: AnalysisResult
       ) -> List[Transaction]:
           """Identify and map CICS/IMS transactions."""
           transactions = []

           for procedure in analysis.procedures:
               if self._is_transaction_entry(procedure):
                   transaction = Transaction(
                       name=procedure.name,
                       entry_point=procedure,
                       inputs=self._extract_inputs(procedure),
                       outputs=self._extract_outputs(procedure),
                       side_effects=self._detect_side_effects(procedure)
                   )
                   transactions.append(transaction)

           return transactions
   ```

3. **Extract Code Generator** `ai_engine/legacy/generator/service.py`:
   ```python
   class CodeGeneratorService:
       """Generates modern code from COBOL analysis."""

       def __init__(self):
           self.rest_generator = RESTAPIGenerator()
           self.sdk_generator = SDKGenerator()
           self.openapi_generator = OpenAPIGenerator()

       async def generate_rest_api(
           self,
           transactions: List[Transaction],
           config: GenerationConfig
       ) -> GeneratedAPI:
           """Generate REST API from transactions."""
           openapi_spec = self.openapi_generator.generate(transactions)

           return GeneratedAPI(
               openapi_spec=openapi_spec,
               handlers=self.rest_generator.generate_handlers(transactions),
               models=self.rest_generator.generate_models(transactions)
           )

       async def generate_sdk(
           self,
           openapi_spec: OpenAPISpec,
           language: SDKLanguage
       ) -> GeneratedSDK:
           """Generate SDK in specified language."""
           return self.sdk_generator.generate(openapi_spec, language)
   ```

4. **Create orchestrator** `ai_engine/legacy/orchestrator/service.py`:
   ```python
   class LegacyWhispererOrchestrator:
       """Orchestrates the complete modernization workflow."""

       def __init__(
           self,
           analyzer: COBOLAnalyzerService,
           mapper: TransactionMapperService,
           generator: CodeGeneratorService
       ):
           self.analyzer = analyzer
           self.mapper = mapper
           self.generator = generator

       async def modernize(
           self,
           request: ModernizationRequest
       ) -> ModernizationResult:
           """Execute full modernization pipeline."""

           # Phase 1: Analyze
           analysis = await self.analyzer.analyze(
               request.source_code,
               request.copybooks
           )

           # Phase 2: Map transactions
           transactions = await self.mapper.map_transactions(analysis)

           # Phase 3: Generate code
           api = await self.generator.generate_rest_api(
               transactions,
               request.config
           )

           # Phase 4: Generate SDKs (if requested)
           sdks = {}
           for language in request.sdk_languages:
               sdks[language] = await self.generator.generate_sdk(
                   api.openapi_spec,
                   language
               )

           return ModernizationResult(
               analysis=analysis,
               transactions=transactions,
               api=api,
               sdks=sdks
           )
   ```

5. **Maintain backward compatibility**:
   ```python
   # ai_engine/legacy/__init__.py
   from ai_engine.legacy.orchestrator.service import LegacyWhispererOrchestrator
   from ai_engine.legacy.analyzer.service import COBOLAnalyzerService
   from ai_engine.legacy.mapper.service import TransactionMapperService
   from ai_engine.legacy.generator.service import CodeGeneratorService

   # Backward-compatible alias
   LegacyWhisperer = LegacyWhispererOrchestrator

   __all__ = [
       "LegacyWhisperer",
       "LegacyWhispererOrchestrator",
       "COBOLAnalyzerService",
       "TransactionMapperService",
       "CodeGeneratorService"
   ]
   ```

#### Acceptance Criteria:
- [ ] Legacy module split into 4 services
- [ ] Each service independently testable
- [ ] Backward-compatible imports maintained
- [ ] All 17K LOC accounted for
- [ ] Test coverage maintained or improved

---

## Phase 3: Focus & Prioritization (Weeks 6-8)

### Goal: Remove non-essential features and focus on core value proposition

---

### 3.1 Disable Non-Core Domain Modules

**Priority:** MEDIUM
**Effort:** 2 days
**Owner:** Product Team

#### Modules to Disable (Not Delete):

| Module | Path | Action | Rationale |
|--------|------|--------|-----------|
| Aviation | `ai_engine/domains/aviation/` | Disable | No active customers |
| Automotive | `ai_engine/domains/automotive/` | Disable | V2X is niche |
| Healthcare | `ai_engine/domains/healthcare/` | Keep (if banking focus) OR Disable | Evaluate customer pipeline |
| Marketplace | `ai_engine/marketplace/` | Disable | Needs scale first |

#### Implementation:

1. **Create feature flags** `ai_engine/core/feature_flags.py`:
   ```python
   from enum import Enum
   from pydantic_settings import BaseSettings

   class FeatureFlags(BaseSettings):
       # Domain modules
       ENABLE_AVIATION_DOMAIN: bool = False
       ENABLE_AUTOMOTIVE_DOMAIN: bool = False
       ENABLE_HEALTHCARE_DOMAIN: bool = True  # Keep for regulated industries

       # Features
       ENABLE_MARKETPLACE: bool = False
       ENABLE_PROTOCOL_MARKETPLACE: bool = False

       # Experimental
       ENABLE_QUANTUM_CRYPTO: bool = True
       ENABLE_AGENTIC_SECURITY: bool = True

       class Config:
           env_prefix = "QBITEL_FEATURE_"

   feature_flags = FeatureFlags()
   ```

2. **Guard domain imports**:
   ```python
   # ai_engine/domains/__init__.py
   from ai_engine.core.feature_flags import feature_flags

   if feature_flags.ENABLE_AVIATION_DOMAIN:
       from ai_engine.domains.aviation import *

   if feature_flags.ENABLE_AUTOMOTIVE_DOMAIN:
       from ai_engine.domains.automotive import *
   ```

3. **Guard API routes**:
   ```python
   from ai_engine.core.feature_flags import feature_flags

   if feature_flags.ENABLE_MARKETPLACE:
       app.include_router(marketplace_router, prefix="/api/v1/marketplace")
   ```

#### Acceptance Criteria:
- [ ] Feature flags control all optional modules
- [ ] Disabled modules don't load (reduce memory footprint)
- [ ] Easy to re-enable via environment variable
- [ ] No code deleted (preserve for future use)

---

### 3.2 Simplify Discovery Pipeline (7 → 5 Phases)

**Priority:** MEDIUM
**Effort:** 3 days
**Owner:** ML Team

#### Current Pipeline:
1. INITIALIZATION
2. STATISTICAL_ANALYSIS
3. CLASSIFICATION
4. GRAMMAR_LEARNING
5. PARSER_GENERATION
6. VALIDATION
7. COMPLETION

#### Simplified Pipeline:
1. **ANALYSIS** (merged: INITIALIZATION + STATISTICAL_ANALYSIS)
2. **CLASSIFICATION**
3. **LEARNING** (merged: GRAMMAR_LEARNING + PARSER_GENERATION)
4. **VALIDATION**
5. **COMPLETION** (minimal, just cleanup)

#### Implementation:

1. **Update phase enum**:
   ```python
   class DiscoveryPhase(Enum):
       ANALYSIS = "analysis"           # Stats + initialization
       CLASSIFICATION = "classification"
       LEARNING = "learning"           # Grammar + parser
       VALIDATION = "validation"
       COMPLETION = "completion"
   ```

2. **Merge phases in orchestrator**:
   ```python
   async def _execute_analysis_phase(self, messages: List[bytes]) -> AnalysisResult:
       """Combined initialization and statistical analysis."""
       # Initialize
       session_id = str(uuid4())

       # Statistical analysis
       stats = self.statistical_analyzer.analyze(messages)

       return AnalysisResult(
           session_id=session_id,
           message_count=len(messages),
           statistics=stats
       )

   async def _execute_learning_phase(
       self,
       classification: ClassificationResult,
       messages: List[bytes]
   ) -> LearningResult:
       """Combined grammar learning and parser generation."""
       # Learn grammar
       grammar = await self.grammar_learner.learn(messages, classification)

       # Generate parser (parallel if possible)
       parser = await self.parser_generator.generate(grammar)

       return LearningResult(grammar=grammar, parser=parser)
   ```

3. **Add circuit breaker**:
   ```python
   from circuitbreaker import circuit

   class DiscoveryOrchestrator:
       @circuit(failure_threshold=3, recovery_timeout=60)
       async def _execute_learning_phase(self, ...):
           """Protected by circuit breaker."""
           ...
   ```

4. **Expose partial results**:
   ```python
   class PartialDiscoveryResult:
       """Returned when pipeline fails mid-way."""
       completed_phases: List[DiscoveryPhase]
       failed_phase: DiscoveryPhase
       partial_data: Dict[str, Any]  # Whatever we learned
       error: QbitelError
   ```

#### Acceptance Criteria:
- [ ] Pipeline reduced to 5 phases
- [ ] Circuit breaker prevents cascade failures
- [ ] Partial results available on failure
- [ ] Performance improved (fewer phase transitions)

---

## Phase 4: Production Hardening (Weeks 9-12)

### Goal: Prepare for production deployment with enterprise reliability

---

### 4.1 Add Circuit Breakers Throughout

**Priority:** HIGH
**Effort:** 3 days
**Owner:** Backend Team

#### Components Needing Circuit Breakers:

| Component | Failure Scenario | Threshold |
|-----------|------------------|-----------|
| LLM Service | Provider down | 3 failures / 60s recovery |
| Database | Connection pool exhausted | 5 failures / 30s recovery |
| Redis Cache | Memory pressure | 3 failures / 15s recovery |
| External APIs | Rate limited | 5 failures / 120s recovery |

#### Implementation:

1. **Install pybreaker**:
   ```bash
   pip install pybreaker
   ```

2. **Create circuit breaker registry** `ai_engine/core/circuit_breakers.py`:
   ```python
   import pybreaker
   from prometheus_client import Counter, Gauge

   # Metrics
   circuit_state = Gauge(
       'qbitel_circuit_breaker_state',
       'Circuit breaker state (0=closed, 1=open, 2=half-open)',
       ['name']
   )
   circuit_failures = Counter(
       'qbitel_circuit_breaker_failures_total',
       'Circuit breaker failure count',
       ['name']
   )

   class MonitoredCircuitBreaker(pybreaker.CircuitBreaker):
       def __init__(self, name: str, **kwargs):
           self.name = name
           super().__init__(**kwargs)

           # Update metrics on state change
           self.add_listener(self._on_state_change)

       def _on_state_change(self, breaker, old_state, new_state):
           state_map = {'closed': 0, 'open': 1, 'half-open': 2}
           circuit_state.labels(name=self.name).set(state_map[new_state.name])

   # Registry
   circuit_breakers = {
       'llm': MonitoredCircuitBreaker(
           name='llm',
           fail_max=3,
           reset_timeout=60,
           exclude=[ValidationError]  # Don't trip on validation errors
       ),
       'database': MonitoredCircuitBreaker(
           name='database',
           fail_max=5,
           reset_timeout=30
       ),
       'redis': MonitoredCircuitBreaker(
           name='redis',
           fail_max=3,
           reset_timeout=15
       ),
   }
   ```

3. **Apply to services**:
   ```python
   from ai_engine.core.circuit_breakers import circuit_breakers

   class UnifiedLLMService:
       @circuit_breakers['llm']
       async def generate(self, prompt: str) -> str:
           ...
   ```

---

### 4.2 Implement Proper Health Checks

**Priority:** HIGH
**Effort:** 2 days
**Owner:** Backend Team

#### Health Check Levels:

| Endpoint | Purpose | Checks |
|----------|---------|--------|
| `/health` | Kubernetes liveness | Process alive |
| `/ready` | Kubernetes readiness | Dependencies ready |
| `/startup` | Kubernetes startup | Initial boot complete |
| `/health/detailed` | Debugging | All component status |

#### Implementation:

```python
# ai_engine/api/health.py
from fastapi import APIRouter, status
from pydantic import BaseModel
from typing import Dict, Optional
from enum import Enum

router = APIRouter(tags=["health"])

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ComponentHealth(BaseModel):
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None

class DetailedHealthResponse(BaseModel):
    status: HealthStatus
    version: str
    uptime_seconds: float
    components: Dict[str, ComponentHealth]

@router.get("/health")
async def liveness():
    """Kubernetes liveness probe - is the process alive?"""
    return {"status": "ok"}

@router.get("/ready")
async def readiness(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
):
    """Kubernetes readiness probe - can we serve traffic?"""
    checks = []

    # Check database
    try:
        await db.execute(text("SELECT 1"))
        checks.append(True)
    except Exception:
        checks.append(False)

    # Check Redis
    try:
        await redis.ping()
        checks.append(True)
    except Exception:
        checks.append(False)

    if all(checks):
        return {"status": "ready"}

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "not ready"}
    )

@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health():
    """Detailed health for debugging and monitoring."""
    components = {}

    # Check each component
    components["database"] = await check_database_health()
    components["redis"] = await check_redis_health()
    components["llm"] = await check_llm_health()
    components["discovery"] = await check_discovery_health()

    # Aggregate status
    statuses = [c.status for c in components.values()]
    if all(s == HealthStatus.HEALTHY for s in statuses):
        overall = HealthStatus.HEALTHY
    elif any(s == HealthStatus.UNHEALTHY for s in statuses):
        overall = HealthStatus.UNHEALTHY
    else:
        overall = HealthStatus.DEGRADED

    return DetailedHealthResponse(
        status=overall,
        version=settings.VERSION,
        uptime_seconds=get_uptime(),
        components=components
    )
```

---

### 4.3 Add Comprehensive Prometheus Metrics

**Priority:** MEDIUM
**Effort:** 3 days
**Owner:** Platform Team

#### Metrics to Add:

```python
# ai_engine/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info

# Application info
app_info = Info('qbitel_app', 'Application information')
app_info.info({
    'version': settings.VERSION,
    'python_version': platform.python_version(),
})

# Request metrics
request_latency = Histogram(
    'qbitel_request_duration_seconds',
    'Request latency in seconds',
    ['method', 'endpoint', 'status'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
)

request_count = Counter(
    'qbitel_requests_total',
    'Total request count',
    ['method', 'endpoint', 'status']
)

# Discovery metrics
discovery_duration = Histogram(
    'qbitel_discovery_duration_seconds',
    'Protocol discovery duration',
    ['phase', 'status'],
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120, 300]
)

discovery_confidence = Histogram(
    'qbitel_discovery_confidence',
    'Discovery confidence score',
    ['protocol_type'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

# LLM metrics
llm_tokens_used = Counter(
    'qbitel_llm_tokens_total',
    'LLM tokens used',
    ['provider', 'model', 'type']  # type: prompt/completion
)

llm_latency = Histogram(
    'qbitel_llm_latency_seconds',
    'LLM response latency',
    ['provider', 'model'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
)

# Resource metrics
active_connections = Gauge(
    'qbitel_active_connections',
    'Active connections',
    ['type']  # database, redis, websocket
)

model_memory_bytes = Gauge(
    'qbitel_model_memory_bytes',
    'Memory used by ML models',
    ['model_name']
)
```

---

### 4.4 Implement Graceful Shutdown

**Priority:** HIGH
**Effort:** 1 day
**Owner:** Backend Team

```python
# ai_engine/core/lifecycle.py
import asyncio
import signal
from contextlib import asynccontextmanager
from fastapi import FastAPI

class GracefulShutdown:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.active_requests = 0
        self.draining = False

    def start_request(self):
        if self.draining:
            raise ServiceUnavailable("Server is shutting down")
        self.active_requests += 1

    def end_request(self):
        self.active_requests -= 1

    async def shutdown(self, timeout: int = 30):
        """Graceful shutdown with timeout."""
        self.draining = True

        # Wait for active requests to complete
        start = asyncio.get_event_loop().time()
        while self.active_requests > 0:
            if asyncio.get_event_loop().time() - start > timeout:
                logger.warning(f"Forced shutdown with {self.active_requests} active requests")
                break
            await asyncio.sleep(0.1)

        self.shutdown_event.set()

shutdown_handler = GracefulShutdown()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting QBITEL...")
    await initialize_services()

    yield

    # Shutdown
    logger.info("Shutting down QBITEL...")
    await shutdown_handler.shutdown(timeout=30)
    await cleanup_services()

app = FastAPI(lifespan=lifespan)

# Signal handlers
def handle_sigterm(*args):
    asyncio.create_task(shutdown_handler.shutdown())

signal.signal(signal.SIGTERM, handle_sigterm)
```

---

## Milestones & Success Criteria

| Week | Phase | Deliverables | Success Criteria |
|------|-------|--------------|------------------|
| 2 | Phase 1 | Security hardening | Zero critical vulnerabilities |
| 5 | Phase 2 | Architecture simplification | 30% reduction in complexity |
| 8 | Phase 3 | Focus & prioritization | Core features 100% tested |
| 12 | Phase 4 | Production hardening | 99.9% uptime target |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking changes during refactor | Medium | High | Feature flags, backward-compatible imports |
| Performance regression | Low | Medium | Benchmark before/after each phase |
| Team bandwidth | Medium | Medium | Parallelize independent workstreams |
| Customer impact | Low | High | Staging environment testing, gradual rollout |

---

## Resource Requirements

| Phase | Engineers | Duration | Dependencies |
|-------|-----------|----------|--------------|
| Phase 1 | 2 backend | 2 weeks | None |
| Phase 2 | 2 backend, 1 ML | 3 weeks | Phase 1 |
| Phase 3 | 1 backend, 1 product | 3 weeks | Phase 2 |
| Phase 4 | 2 backend, 1 platform | 4 weeks | Phase 3 |

**Total:** 12 weeks, 3-4 engineers

---

## Next Steps

1. **Immediate (This Week):**
   - [ ] Review and approve this plan
   - [ ] Assign owners to Phase 1 tasks
   - [ ] Set up feature flag infrastructure

2. **Week 1:**
   - [ ] Implement rate limiting
   - [ ] Add input validation
   - [ ] Begin error handling standardization

3. **Week 2:**
   - [ ] Complete error handling refactor
   - [ ] Write tests for new middleware
   - [ ] Deploy to staging environment


---

# ============================================================
# FILE 9: docs/ENVIRONMENT_VARIABLE_CONFIGURATION.md
# ============================================================

# Environment Variable Configuration Guide

## Overview

QBITEL provides production-ready environment variable loading with comprehensive validation for all sensitive configuration values. This guide covers the supported environment variables, security requirements, and best practices.

## Table of Contents

- [Supported Environment Variables](#supported-environment-variables)
- [Security Requirements](#security-requirements)
- [Production vs Development Mode](#production-vs-development-mode)
- [Configuration Examples](#configuration-examples)
- [Validation Rules](#validation-rules)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Supported Environment Variables

### Database Configuration

| Variable | Alternative | Required | Description |
|----------|-------------|----------|-------------|
| `QBITEL_AI_DB_PASSWORD` | `DATABASE_PASSWORD` | Yes (Production) | PostgreSQL database password |
| `QBITEL_AI_DB_HOST` | - | No | Database host (default: localhost) |
| `QBITEL_AI_DB_PORT` | - | No | Database port (default: 5432) |
| `QBITEL_AI_DB_NAME` | - | No | Database name (default: qbitel) |
| `QBITEL_AI_DB_USER` | - | No | Database username (default: qbitel) |

**Priority:** `QBITEL_AI_*` prefixed variables take priority over generic alternatives.

### Redis Configuration

| Variable | Alternative | Required | Description |
|----------|-------------|----------|-------------|
| `QBITEL_AI_REDIS_PASSWORD` | `REDIS_PASSWORD` | Yes (Production) | Redis authentication password |
| `QBITEL_AI_REDIS_HOST` | - | No | Redis host (default: localhost) |
| `QBITEL_AI_REDIS_PORT` | - | No | Redis port (default: 6379) |

**Security Note:** Running Redis without authentication in production is a **CRITICAL** security risk.

### Security Configuration

| Variable | Alternative | Required | Description |
|----------|-------------|----------|-------------|
| `QBITEL_AI_JWT_SECRET` | `JWT_SECRET` | Yes (Production) | JWT token signing secret |
| `QBITEL_AI_ENCRYPTION_KEY` | `ENCRYPTION_KEY` | Yes (Production)* | Data encryption key |
| `QBITEL_AI_API_KEY` | `API_KEY` | No | API authentication key |

\* Required when encryption is enabled (default: enabled)

### Environment Control

| Variable | Alternative | Default | Description |
|----------|-------------|---------|-------------|
| `QBITEL_AI_ENVIRONMENT` | `ENVIRONMENT` | development | Runtime environment (development/staging/production) |

## Security Requirements

### Password Requirements

All passwords must meet the following criteria:

- **Minimum Length:** 16 characters
- **No Weak Patterns:** Cannot contain common words like 'password', 'admin', 'test', 'demo', '123456'
- **No Sequential Characters:** Cannot contain sequences like '123', 'abc', '456'
- **No Repeated Characters:** Cannot have more than 3 consecutive identical characters
- **Cryptographically Secure:** Must be generated using secure random methods

### JWT Secret Requirements

- **Minimum Length:** 32 characters
- **No Weak Patterns:** Cannot contain common words
- **High Entropy:** Must have sufficient randomness (>50% unique characters)
- **No Sequential Characters:** Cannot contain long sequential patterns

### Encryption Key Requirements

- **Minimum Length:** 32 characters
- **Format:** Base64-encoded 256-bit key (for Fernet encryption)
- **Cryptographically Secure:** Must be generated using proper key derivation

### API Key Requirements

- **Minimum Length:** 32 characters
- **Format:** Must contain both letters and digits
- **No Weak Patterns:** Cannot contain common words
- **Prefix Recommended:** Use a prefix like `qbitel_` for identification

## Production vs Development Mode

### Production Mode

Activated when `QBITEL_AI_ENVIRONMENT=production` or `ENVIRONMENT=production`

**Enforced Requirements:**
- ✅ Database password is **REQUIRED**
- ✅ Redis password is **REQUIRED**
- ✅ JWT secret is **REQUIRED**
- ✅ Encryption key is **REQUIRED** (when encryption enabled)
- ✅ All secrets must pass validation
- ✅ CORS wildcard (*) is **FORBIDDEN**
- ❌ Configuration errors cause immediate failure

### Development Mode

Activated when `QBITEL_AI_ENVIRONMENT=development` (default)

**Relaxed Requirements:**
- ⚠️ Missing secrets generate warnings (not errors)
- ⚠️ Weak secrets generate warnings (not errors)
- ✅ CORS wildcard allowed
- ✅ Redis can run without authentication

**Warning:** Development mode should **NEVER** be used in production environments.

## Configuration Examples

### Example 1: Production Environment Setup

```bash
# Generate secure secrets
export QBITEL_AI_DB_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export QBITEL_AI_REDIS_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export QBITEL_AI_JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
export QBITEL_AI_ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
export QBITEL_AI_API_KEY=$(python -c "import secrets; print('qbitel_' + secrets.token_urlsafe(32))")

# Set environment
export QBITEL_AI_ENVIRONMENT=production

# Start application
python -m ai_engine
```

### Example 2: Docker Compose

```yaml
version: '3.8'

services:
  qbitel:
    image: qbitel:latest
    environment:
      - QBITEL_AI_ENVIRONMENT=production
      - QBITEL_AI_DB_PASSWORD=${QBITEL_AI_DB_PASSWORD}
      - QBITEL_AI_REDIS_PASSWORD=${QBITEL_AI_REDIS_PASSWORD}
      - QBITEL_AI_JWT_SECRET=${QBITEL_AI_JWT_SECRET}
      - QBITEL_AI_ENCRYPTION_KEY=${QBITEL_AI_ENCRYPTION_KEY}
      - QBITEL_AI_API_KEY=${QBITEL_AI_API_KEY}
    env_file:
      - .env.production
```

### Example 3: Kubernetes Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: qbitel-secrets
type: Opaque
stringData:
  QBITEL_AI_DB_PASSWORD: "<base64-encoded-password>"
  QBITEL_AI_REDIS_PASSWORD: "<base64-encoded-password>"
  QBITEL_AI_JWT_SECRET: "<base64-encoded-secret>"
  QBITEL_AI_ENCRYPTION_KEY: "<base64-encoded-key>"
  QBITEL_AI_API_KEY: "<base64-encoded-key>"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qbitel
spec:
  template:
    spec:
      containers:
      - name: qbitel
        envFrom:
        - secretRef:
            name: qbitel-secrets
        env:
        - name: QBITEL_AI_ENVIRONMENT
          value: "production"
```

### Example 4: Development Environment

```bash
# Minimal setup for development
export QBITEL_AI_ENVIRONMENT=development

# Optional: Set passwords for testing
export QBITEL_AI_DB_PASSWORD="dev_password_min16chars"
export QBITEL_AI_REDIS_PASSWORD="dev_redis_min16chars"

# Start application (will use defaults for missing values)
python -m ai_engine
```

## Validation Rules

### Database Password Validation

```python
# ✅ Valid
QBITEL_AI_DB_PASSWORD="Kx9mP2nQ7vR4wS8tY3uZ1aB5cD6eF0gH"

# ❌ Invalid - Too short
QBITEL_AI_DB_PASSWORD="short123"

# ❌ Invalid - Contains weak pattern
QBITEL_AI_DB_PASSWORD="password123456789"

# ❌ Invalid - Sequential characters
QBITEL_AI_DB_PASSWORD="abcdefghijklmnop"

# ❌ Invalid - Repeated characters
QBITEL_AI_DB_PASSWORD="aaaaaaaaaaaaaaaa"
```

### JWT Secret Validation

```python
# ✅ Valid
QBITEL_AI_JWT_SECRET="vN8xM2kL9pQ4rS7tY1uZ3aB6cD5eF0gH2iJ4kL7mN9oP"

# ❌ Invalid - Too short
QBITEL_AI_JWT_SECRET="short_secret"

# ❌ Invalid - Contains weak pattern
QBITEL_AI_JWT_SECRET="secret123456789012345678901234567890"

# ❌ Invalid - Low entropy
QBITEL_AI_JWT_SECRET="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
```

### API Key Validation

```python
# ✅ Valid
QBITEL_AI_API_KEY="qbitel_Kx9mP2nQ7vR4wS8tY3uZ1aB5cD6eF0gH"

# ❌ Invalid - Too short
QBITEL_AI_API_KEY="qbitel_short"

# ❌ Invalid - No digits
QBITEL_AI_API_KEY="qbitel_abcdefghijklmnopqrstuvwxyz"

# ❌ Invalid - No letters
QBITEL_AI_API_KEY="qbitel_123456789012345678901234567890"
```

## Troubleshooting

### Error: "Database password not configured"

**Cause:** Missing database password in production mode.

**Solution:**
```bash
# Generate a secure password
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set the environment variable
export QBITEL_AI_DB_PASSWORD="<generated_password>"
```

### Error: "Password too short"

**Cause:** Password does not meet minimum length requirement (16 characters).

**Solution:**
```bash
# Generate a password that meets requirements
export QBITEL_AI_DB_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

### Error: "Password contains weak patterns"

**Cause:** Password contains common words like 'password', 'admin', 'test'.

**Solution:**
```bash
# Use cryptographically secure random generation
export QBITEL_AI_DB_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

### Error: "JWT secret not configured"

**Cause:** Missing JWT secret in production mode.

**Solution:**
```bash
# Generate a secure JWT secret
export QBITEL_AI_JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
```

### Error: "Encryption key not configured"

**Cause:** Missing encryption key when encryption is enabled in production.

**Solution:**
```bash
# Generate a Fernet encryption key
export QBITEL_AI_ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
```

### Error: "CORS wildcard (*) is not allowed in production mode"

**Cause:** CORS is configured with wildcard in production.

**Solution:**
```python
# In your configuration file, specify allowed origins
security:
  cors_origins:
    - "https://app.example.com"
    - "https://api.example.com"
```

## Best Practices

### 1. Use Secrets Management Systems

**Recommended:**
- HashiCorp Vault
- AWS Secrets Manager
- Azure Key Vault
- Google Cloud Secret Manager

**Example with Vault:**
```bash
# Store secrets in Vault
vault kv put secret/qbitel/production \
  db_password="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  redis_password="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  jwt_secret="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"

# Retrieve and export
export QBITEL_AI_DB_PASSWORD=$(vault kv get -field=db_password secret/qbitel/production)
```

### 2. Rotate Secrets Regularly

```bash
# Automated rotation script
#!/bin/bash
NEW_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Update in secrets manager
vault kv put secret/qbitel/production db_password="$NEW_PASSWORD"

# Update database
psql -c "ALTER USER qbitel WITH PASSWORD '$NEW_PASSWORD';"

# Restart application to pick up new secret
kubectl rollout restart deployment/qbitel
```

### 3. Never Commit Secrets to Version Control

```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo ".env.production" >> .gitignore
echo ".env.local" >> .gitignore
echo "*.key" >> .gitignore
echo "*.pem" >> .gitignore
```

### 4. Use Environment-Specific Files

```bash
# .env.development
QBITEL_AI_ENVIRONMENT=development
QBITEL_AI_DB_PASSWORD=dev_password_min16chars

# .env.production (never commit!)
QBITEL_AI_ENVIRONMENT=production
QBITEL_AI_DB_PASSWORD=<secure_generated_password>
QBITEL_AI_REDIS_PASSWORD=<secure_generated_password>
QBITEL_AI_JWT_SECRET=<secure_generated_secret>
QBITEL_AI_ENCRYPTION_KEY=<secure_generated_key>
```

### 5. Implement Secret Scanning

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

### 6. Monitor Secret Usage

```python
# Log secret access (without exposing values)
import logging

logger = logging.getLogger(__name__)

def load_secret(key: str) -> str:
    value = os.getenv(key)
    if value:
        logger.info(f"Loaded secret: {key} (length: {len(value)})")
    else:
        logger.warning(f"Secret not found: {key}")
    return value
```

### 7. Use Least Privilege Principle

```bash
# Create dedicated database user with minimal permissions
CREATE USER qbitel_app WITH PASSWORD '<secure_password>';
GRANT CONNECT ON DATABASE qbitel TO qbitel_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO qbitel_app;
```

### 8. Implement Secret Expiration

```python
# Track secret age
from datetime import datetime, timedelta

SECRET_MAX_AGE_DAYS = 90

def check_secret_age(secret_created_at: datetime) -> bool:
    age = datetime.utcnow() - secret_created_at
    if age > timedelta(days=SECRET_MAX_AGE_DAYS):
        logger.warning(f"Secret is {age.days} days old. Rotation recommended.")
        return False
    return True
```

## Security Checklist

Before deploying to production, ensure:

- [ ] All required environment variables are set
- [ ] All secrets meet minimum length requirements
- [ ] No weak patterns detected in any secrets
- [ ] Secrets are stored in a secure secrets management system
- [ ] Secrets are not committed to version control
- [ ] Secret rotation policy is in place
- [ ] Secret access is logged and monitored
- [ ] CORS is configured with specific origins (no wildcard)
- [ ] Database user has minimal required permissions
- [ ] Redis authentication is enabled
- [ ] TLS/SSL is enabled for all connections
- [ ] Environment is set to 'production'

## Additional Resources

- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [NIST Password Guidelines](https://pages.nist.gov/800-63-3/sp800-63b.html)
- [CIS Benchmarks](https://www.cisecurity.org/cis-benchmarks/)
- [QBITEL Security Documentation](./SECURITY_SECRETS_MANAGEMENT.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/yazhsab/qbitel-bridge/issues
- Security Issues: security@qbitel.com
- Documentation: https://docs.qbitel.com

---

# ============================================================
# FILE 10: docs/LOCAL_DEPLOYMENT_GUIDE.md
# ============================================================

# QBITEL - Local Development & Testing Guide

**Version**: 2.1.0
**Last Updated**: 2025-11-22
**Purpose**: Complete guide for running QBITEL locally for development and testing

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start (5 minutes)](#2-quick-start-5-minutes)
3. [Full Local Setup](#3-full-local-setup)
4. [Docker Compose Deployment](#4-docker-compose-deployment)
5. [Running Individual Components](#5-running-individual-components)
6. [Database Setup](#6-database-setup)
7. [Configuration](#7-configuration)
8. [Testing](#8-testing)
9. [API Testing](#9-api-testing)
10. [Troubleshooting](#10-troubleshooting)
11. [macOS Deployment (Recommended)](#11-macos-deployment-recommended)

---

## 1. Prerequisites

### 1.1 Required Software

| Software | Version | Installation |
|----------|---------|--------------|
| Python | 3.10+ | `brew install python@3.11` (macOS) |
| PostgreSQL | 15+ | `brew install postgresql@15` |
| Redis | 7+ | `brew install redis` |
| Docker | 24+ | [Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| Node.js | 18+ | `brew install node` (for UI) |
| Git | 2.30+ | `brew install git` |

### 1.2 Optional Software

| Software | Purpose | Installation |
|----------|---------|--------------|
| Ollama | Local LLM | `brew install ollama` |
| ClamAV | Virus scanning | `brew install clamav` |
| Go | Go services | `brew install go` |
| Rust | Dataplane | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |

### 1.3 System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum (32GB for ML features)
- **Disk**: 20GB free space
- **OS**: macOS 12+, Ubuntu 20.04+, Windows 10+ (WSL2)

---

## 2. Quick Start (5 minutes)

### Option A: Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/yazhsab/qbitel-bridge.git
cd qbitel

# Start all services with Docker
docker-compose -f ops/deploy/docker/docker-compose.yml up -d

# Wait for services to be healthy (30-60 seconds)
docker-compose -f ops/deploy/docker/docker-compose.yml ps

# Test health endpoint
curl http://localhost:8000/health
```

### Option B: Minimal Python Setup

```bash
# Clone and setup
git clone https://github.com/yazhsab/qbitel-bridge.git
cd qbitel

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set minimal environment variables
export DATABASE_URL="sqlite:///./qbitel_dev.db"
export REDIS_URL="redis://localhost:6379"
export JWT_SECRET="dev-secret-key-change-in-production"
export ENCRYPTION_KEY="dev-encryption-key-32chars!"

# Run the application
python -m ai_engine --mode development

# Test (in another terminal)
curl http://localhost:8000/health
```

---

## 3. Full Local Setup

### 3.1 Clone Repository

```bash
git clone https://github.com/yazhsab/qbitel-bridge.git
cd qbitel
```

### 3.2 Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Verify Python version
python --version  # Should be 3.10+
```

### 3.3 Install Dependencies

```bash
# Install main dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install optional ML dependencies (if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Verify installation
python -c "import ai_engine; print('AI Engine imported successfully')"
```

### 3.4 Install Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

---

## 4. Docker Compose Deployment

### 4.1 Full Stack Deployment

```bash
# Navigate to docker directory
cd ops/deploy/docker

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 4.2 Services Started

| Service | Port | Description |
|---------|------|-------------|
| ai-engine | 8000 | Main API server |
| postgres | 5432 | PostgreSQL database |
| redis | 6379 | Redis cache |
| ollama | 11434 | Local LLM (optional) |

### 4.3 Docker Compose with Copilot

```bash
# Use copilot-specific compose file
docker-compose -f ops/deploy/docker/docker-compose-copilot.yml up -d
```

### 4.4 Stop All Services

```bash
docker-compose down

# Remove volumes (fresh start)
docker-compose down -v
```

---

## 5. Running Individual Components

### 5.1 Start PostgreSQL

```bash
# Using Docker
docker run -d \
  --name qbitel-postgres \
  -e POSTGRES_USER=qbitel \
  -e POSTGRES_PASSWORD=qbitel_dev_password \
  -e POSTGRES_DB=qbitel_dev \
  -p 5432:5432 \
  postgres:15

# Or using Homebrew (macOS)
brew services start postgresql@15
createdb qbitel_dev
```

### 5.2 Start Redis

```bash
# Using Docker
docker run -d \
  --name qbitel-redis \
  -p 6379:6379 \
  redis:7

# Or using Homebrew (macOS)
brew services start redis
```

### 5.3 Start Ollama (Local LLM)

```bash
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve &

# Pull recommended model
ollama pull llama3.2:8b

# Verify
curl http://localhost:11434/api/tags
```

### 5.4 Start AI Engine

```bash
# Activate virtual environment
source venv/bin/activate

# Set environment variables
export QBITEL_ENV=development
export DATABASE_HOST=localhost
export DATABASE_PORT=5432
export DATABASE_NAME=qbitel_dev
export DATABASE_USER=qbitel
export DATABASE_PASSWORD=qbitel_dev_password
export REDIS_HOST=localhost
export REDIS_PORT=6379
export JWT_SECRET=development-jwt-secret-key-32ch
export ENCRYPTION_KEY=development-encryption-key-32c

# Run in development mode (with hot reload)
python -m ai_engine --mode development

# Or run in production mode
python -m ai_engine --mode production
```

### 5.5 Start Frontend Console (Optional)

```bash
cd ui/console

# Install dependencies
npm install

# Start development server
npm run dev

# Access at http://localhost:3000
```

---

## 6. Database Setup

### 6.1 Apply Migrations

```bash
# Navigate to ai_engine directory
cd ai_engine

# Apply all migrations
alembic upgrade head

# Check current migration
alembic current

# View migration history
alembic history
```

### 6.2 Create Initial Admin User

```bash
# Using the CLI tool
python -m ai_engine.scripts.create_admin \
  --username admin \
  --email admin@qbitel.local \
  --password "SecurePassword123!"

# Or via Python shell
python << 'EOF'
from ai_engine.core.database_manager import get_database_manager
from ai_engine.models.database import User
from passlib.hash import bcrypt

db = get_database_manager()
session = db.get_session()

admin = User(
    username="admin",
    email="admin@qbitel.local",
    password_hash=bcrypt.hash("admin123"),
    role="administrator",
    is_active=True
)
session.add(admin)
session.commit()
print("Admin user created!")
session.close()
EOF
```

### 6.3 Seed Sample Data (Optional)

```bash
# Load sample protocols
python -m ai_engine.scripts.seed_data --type protocols

# Load sample threat intelligence
python -m ai_engine.scripts.seed_data --type threats
```

---

## 7. Configuration

### 7.1 Configuration File

Create a local configuration file:

```bash
cp config/qbitel.yaml config/qbitel.local.yaml
```

Edit `config/qbitel.local.yaml`:

```yaml
# Local Development Configuration
environment: development
debug: true

# Server
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  reload: true

# Database
database:
  host: "localhost"
  port: 5432
  name: "qbitel_dev"
  user: "qbitel"
  password: "qbitel_dev_password"
  pool_size: 10
  ssl_mode: "disable"

# Redis
redis:
  host: "localhost"
  port: 6379
  db: 0
  password: null

# LLM Configuration
llm:
  provider: "ollama"  # ollama, anthropic, openai
  endpoint: "http://localhost:11434"
  model: "llama3.2:8b"
  fallback_providers:
    - "anthropic"
    - "openai"

# Security (development values - DO NOT USE IN PRODUCTION)
security:
  jwt_secret: "development-jwt-secret-key-32ch"
  jwt_expiry_minutes: 480  # 8 hours for dev
  encryption_key: "development-encryption-key-32c"

# Logging
logging:
  level: "DEBUG"
  format: "json"

# Features
features:
  enable_copilot: true
  enable_marketplace: true
  enable_translation_studio: true
  enable_threat_intelligence: true

# Notifications (optional for local dev)
notifications:
  enabled: false

# SIEM (optional for local dev)
audit:
  log_to_file: true
  log_to_siem: false
```

### 7.2 Environment Variables

Create a `.env` file in the project root:

```bash
cat > .env << 'EOF'
# QBITEL Local Development Environment

# Environment
QBITEL_ENV=development
DEBUG=true

# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=qbitel_dev
DATABASE_USER=qbitel
DATABASE_PASSWORD=qbitel_dev_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Security
JWT_SECRET=development-jwt-secret-key-32ch
ENCRYPTION_KEY=development-encryption-key-32c
API_KEY=dev-api-key-for-testing

# LLM
QBITEL_LLM_PROVIDER=ollama
QBITEL_LLM_ENDPOINT=http://localhost:11434
QBITEL_LLM_MODEL=llama3.2:8b

# Optional: Cloud LLM fallback (add your keys)
# ANTHROPIC_API_KEY=sk-ant-xxx
# OPENAI_API_KEY=sk-xxx

# Logging
LOG_LEVEL=DEBUG

# Disable TLS for local development
TLS_ENABLED=false
EOF

# Load environment variables
source .env
```

### 7.3 Use Configuration

```bash
# Run with custom config
python -m ai_engine --config config/qbitel.local.yaml

# Or set via environment variable
export QBITEL_CONFIG_FILE=config/qbitel.local.yaml
python -m ai_engine
```

---

## 8. Testing

### 8.1 Run All Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest ai_engine/tests/ -v

# Run with coverage
pytest ai_engine/tests/ --cov=ai_engine --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
```

### 8.2 Run Specific Test Suites

```bash
# Unit tests
pytest ai_engine/tests/unit/ -v

# Integration tests (requires database)
pytest ai_engine/tests/integration/ -v

# Security tests
pytest ai_engine/tests/security/ -v

# Cloud-native tests
pytest ai_engine/tests/cloud_native/ -v

# Performance tests
pytest ai_engine/tests/performance/ -v
```

### 8.3 Run Tests by Keyword

```bash
# Run tests matching a keyword
pytest -k "auth" -v
pytest -k "copilot" -v
pytest -k "anomaly" -v
```

### 8.4 Run Tests in Parallel

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest ai_engine/tests/ -n auto -v
```

---

## 9. API Testing

### 9.1 Health Check

```bash
# Basic health
curl http://localhost:8000/health

# Kubernetes liveness
curl http://localhost:8000/healthz

# Kubernetes readiness
curl http://localhost:8000/readyz
```

### 9.2 Authentication

```bash
# Login
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' | jq -r '.access_token')

echo "Token: $TOKEN"

# Verify token
curl http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

### 9.3 Copilot API

```bash
# Get copilot info
curl http://localhost:8000/api/v1/copilot/ \
  -H "Authorization: Bearer $TOKEN"

# Send a query
curl -X POST http://localhost:8000/api/v1/copilot/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What protocols do you support?",
    "query_type": "general_question"
  }'
```

### 9.4 Protocol Discovery

```bash
# Discover protocol from sample data
curl -X POST http://localhost:8000/api/v1/discover \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": ["R0VUIC8gSFRUUC8xLjENCkhvc3Q6IGV4YW1wbGUuY29tDQoNCg=="],
    "encoding": "base64"
  }'
```

### 9.5 Field Detection

```bash
# Detect fields in a message
curl -X POST http://localhost:8000/api/v1/fields/detect \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "R0VUIC8gSFRUUC8xLjENCkhvc3Q6IGV4YW1wbGUuY29tDQoNCg==",
    "encoding": "base64"
  }'
```

### 9.6 Anomaly Detection

```bash
# Check for anomalies
curl -X POST http://localhost:8000/api/v1/anomalies/detect \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "R0VUIC8gSFRUUC8xLjENCkhvc3Q6IGV4YW1wbGUuY29tDQoNCg==",
    "encoding": "base64"
  }'
```

### 9.7 OpenAPI Documentation

Access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### 9.8 Using HTTPie (Alternative to curl)

```bash
# Install httpie
pip install httpie

# Health check
http GET localhost:8000/health

# Login
http POST localhost:8000/api/v1/auth/login username=admin password=admin123

# Authenticated request
http GET localhost:8000/api/v1/copilot/ "Authorization: Bearer $TOKEN"
```

---

## 10. Troubleshooting

### 10.1 Common Issues

#### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
python -m ai_engine --port 8001
```

#### Database Connection Failed

```bash
# Check if PostgreSQL is running
pg_isready -h localhost -p 5432

# Check Docker container
docker ps | grep postgres

# View PostgreSQL logs
docker logs qbitel-postgres

# Test connection
psql -h localhost -U qbitel -d qbitel_dev -c "SELECT 1"
```

#### Redis Connection Failed

```bash
# Check if Redis is running
redis-cli ping

# Check Docker container
docker ps | grep redis

# View Redis logs
docker logs qbitel-redis
```

#### Module Import Errors

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

#### Migration Errors

```bash
# Check current state
alembic current

# Show history
alembic history

# Stamp to specific version (if needed)
alembic stamp head

# Create new migration
alembic revision --autogenerate -m "description"
```

### 10.2 Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=true

# Run with verbose output
python -m ai_engine --mode development --log-level DEBUG
```

### 10.3 Check Service Health

```bash
# Check all endpoints
curl http://localhost:8000/health && echo "✓ Health OK"
curl http://localhost:8000/healthz && echo "✓ Liveness OK"
curl http://localhost:8000/readyz && echo "✓ Readiness OK"
```

### 10.4 View Application Logs

```bash
# If running via Docker
docker-compose logs -f ai-engine

# If running directly
tail -f /var/log/qbitel/app.log

# Or check stdout/stderr during development
```

### 10.5 Reset Local Environment

```bash
# Stop all services
docker-compose down -v

# Remove Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Remove virtual environment
rm -rf venv

# Fresh start
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Restart services
docker-compose up -d
```

---

## 11. macOS Deployment (Recommended)

For macOS users, we provide a comprehensive deployment script that handles all prerequisites, setup, and service management automatically.

### 11.1 Quick Start with macOS Script

```bash
# Make the script executable (if not already)
chmod +x scripts/deploy-macos.sh

# Install all prerequisites (Homebrew, Python, Docker, PostgreSQL, Redis, Ollama)
./scripts/deploy-macos.sh install

# Setup environment and start services
./scripts/deploy-macos.sh setup
./scripts/deploy-macos.sh start

# Check status
./scripts/deploy-macos.sh status
```

### 11.2 Available Commands

| Command | Description |
|---------|-------------|
| `install` | Install all prerequisites (Homebrew, Python, Docker, PostgreSQL, Redis, Ollama) |
| `setup` | Initial setup (create virtualenv, install deps, setup database, create .env) |
| `start` | Start all services |
| `stop` | Stop all services |
| `restart` | Restart all services |
| `status` | Show service status dashboard |
| `logs` | View logs (supports --follow and --service flags) |
| `health` | Run comprehensive health checks |
| `clean` | Clean up containers, volumes, and cache |
| `uninstall` | Complete uninstall |

### 11.3 Deployment Modes

```bash
# Quick start with Docker Compose only (default)
./scripts/deploy-macos.sh start --quick

# Full stack with monitoring (Prometheus, Grafana)
./scripts/deploy-macos.sh start --full

# Run Python natively (Docker only for dependencies)
./scripts/deploy-macos.sh start --native

# Development mode with hot reload
./scripts/deploy-macos.sh start --dev
```

### 11.4 One-Line Quick Start

```bash
# Clone, setup, and start everything
git clone https://github.com/yazhsab/qbitel-bridge.git && \
cd qbitel && \
chmod +x scripts/deploy-macos.sh && \
./scripts/deploy-macos.sh install && \
./scripts/deploy-macos.sh setup && \
./scripts/deploy-macos.sh start
```

### 11.5 Apple Silicon vs Intel

The deployment script automatically detects your Mac's architecture:
- **Apple Silicon (M1/M2/M3)**: Uses ARM64-optimized Docker images
- **Intel Macs**: Uses standard AMD64 images

No manual configuration required.

### 11.6 Service Management

```bash
# View status dashboard
./scripts/deploy-macos.sh status

# Follow logs in real-time
./scripts/deploy-macos.sh logs --follow

# View logs for specific service
./scripts/deploy-macos.sh logs --service ai-engine

# Run health checks
./scripts/deploy-macos.sh health

# Stop everything
./scripts/deploy-macos.sh stop

# Clean reset (removes all data)
./scripts/deploy-macos.sh clean --all
```

### 11.7 Troubleshooting macOS

```bash
# Skip prerequisites check
./scripts/deploy-macos.sh start --skip-prereq

# Force reinstall/recreate
./scripts/deploy-macos.sh setup --force

# View detailed help
./scripts/deploy-macos.sh --help
```

---

## Appendix A: Quick Reference Commands

```bash
# Start everything
docker-compose -f ops/deploy/docker/docker-compose.yml up -d

# Stop everything
docker-compose -f ops/deploy/docker/docker-compose.yml down

# View logs
docker-compose logs -f

# Run tests
pytest ai_engine/tests/ -v

# Apply migrations
cd ai_engine && alembic upgrade head

# Create admin user
python -m ai_engine.scripts.create_admin --username admin --password admin123

# Get auth token
curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' | jq -r '.access_token'

# Health check
curl http://localhost:8000/health
```

---

## Appendix B: IDE Setup

### VS Code

Install recommended extensions:
- Python
- Pylance
- Docker
- YAML
- REST Client

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["ai_engine/tests"],
  "editor.formatOnSave": true
}
```

### PyCharm

1. Open project folder
2. Configure Python interpreter: `File > Settings > Project > Python Interpreter`
3. Select `venv/bin/python`
4. Enable pytest: `File > Settings > Tools > Python Integrated Tools > Default test runner: pytest`

---

## Appendix C: Useful Scripts

### Start Script (`scripts/start-local.sh`)

```bash
#!/bin/bash
set -e

echo "Starting QBITEL local environment..."

# Start dependencies
docker-compose -f ops/deploy/docker/docker-compose.yml up -d postgres redis

# Wait for database
echo "Waiting for database..."
sleep 5

# Apply migrations
cd ai_engine
alembic upgrade head
cd ..

# Start application
source venv/bin/activate
python -m ai_engine --mode development
```

### Test Script (`scripts/run-tests.sh`)

```bash
#!/bin/bash
set -e

source venv/bin/activate

echo "Running linting..."
ruff check ai_engine/

echo "Running type checks..."
mypy ai_engine/ --ignore-missing-imports

echo "Running tests..."
pytest ai_engine/tests/ -v --cov=ai_engine --cov-report=term-missing

echo "All checks passed!"
```

---

**Document maintained by**: QBITEL Engineering Team
**Questions?** Open an issue at https://github.com/yazhsab/qbitel-bridge/issues


---

# ============================================================
# FILE 11: docs/DATABASE_MIGRATIONS.md
# ============================================================

# Database Migrations Guide

## Overview

QBITEL Engine uses Alembic for database schema migrations, providing a robust and production-ready migration system with version control, rollback capabilities, and comprehensive testing procedures.

## Prerequisites

- PostgreSQL 12+ installed and running
- Python 3.10+ with required dependencies
- Database credentials configured via environment variables

## Environment Variables

Set the following environment variables before running migrations:

```bash
# Database Configuration
export QBITEL_AI_DB_HOST="localhost"
export QBITEL_AI_DB_PORT="5432"
export QBITEL_AI_DB_NAME="qbitel"
export QBITEL_AI_DB_USER="qbitel"
export QBITEL_AI_DB_PASSWORD="<your-secure-password>"

# Optional: Use async mode for migrations
export ALEMBIC_USE_ASYNC="false"
```

## Installation

Install Alembic and dependencies:

```bash
cd ai_engine
pip install alembic asyncpg psycopg2-binary
```

## Migration Commands

### Initialize Database (First Time)

```bash
# Navigate to ai_engine directory
cd ai_engine

# Run initial migration to create all tables
alembic upgrade head
```

### Check Current Migration Status

```bash
alembic current
```

### View Migration History

```bash
alembic history --verbose
```

### Upgrade to Latest Version

```bash
alembic upgrade head
```

### Upgrade to Specific Version

```bash
alembic upgrade <revision_id>
```

### Downgrade One Version

```bash
alembic downgrade -1
```

### Downgrade to Specific Version

```bash
alembic downgrade <revision_id>
```

### Rollback All Migrations

```bash
alembic downgrade base
```

## Creating New Migrations

### Auto-generate Migration from Model Changes

```bash
alembic revision --autogenerate -m "Description of changes"
```

### Create Empty Migration

```bash
alembic revision -m "Description of changes"
```

### Review Generated Migration

Always review auto-generated migrations before applying:

```bash
cat alembic/versions/<revision_id>_description.py
```

## Migration Testing Procedures

### 1. Test in Development Environment

```bash
# Backup current database
pg_dump -U qbitel qbitel > backup_before_migration.sql

# Run migration
alembic upgrade head

# Verify tables created
psql -U qbitel -d qbitel -c "\dt"

# Test rollback
alembic downgrade -1

# Re-apply migration
alembic upgrade head
```

### 2. Test Data Integrity

```python
# test_migration.py
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from ai_engine.models.database import User, APIKey, AuditLog

async def test_migration():
    engine = create_async_engine(
        "postgresql+asyncpg://qbitel:password@localhost/qbitel"
    )
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        # Test user creation
        from uuid import uuid4
        user = User(
            id=uuid4(),
            username="test_user",
            email="test@example.com",
            full_name="Test User",
            role="viewer"
        )
        session.add(user)
        await session.commit()
        
        print("✓ User creation successful")
        
        # Test API key creation
        api_key = APIKey(
            id=uuid4(),
            key_hash="test_hash",
            key_prefix="test_",
            name="Test Key",
            user_id=user.id
        )
        session.add(api_key)
        await session.commit()
        
        print("✓ API key creation successful")
        
        # Test audit log
        audit = AuditLog(
            id=uuid4(),
            action="user_created",
            user_id=user.id,
            success=True
        )
        session.add(audit)
        await session.commit()
        
        print("✓ Audit log creation successful")

if __name__ == "__main__":
    asyncio.run(test_migration())
```

Run the test:

```bash
python test_migration.py
```

### 3. Performance Testing

```bash
# Test migration performance
time alembic upgrade head

# Test rollback performance
time alembic downgrade -1
```

## Production Deployment

### Pre-Deployment Checklist

- [ ] Backup production database
- [ ] Test migration in staging environment
- [ ] Review all migration scripts
- [ ] Verify rollback procedures
- [ ] Schedule maintenance window
- [ ] Notify stakeholders

### Production Migration Steps

1. **Backup Database**

```bash
# Create timestamped backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
pg_dump -U qbitel -h prod-db-host qbitel > backup_${TIMESTAMP}.sql

# Verify backup
pg_restore --list backup_${TIMESTAMP}.sql
```

2. **Enable Maintenance Mode**

```bash
# Set application to maintenance mode
kubectl scale deployment qbitel --replicas=0
```

3. **Run Migration**

```bash
# Set production environment
export QBITEL_AI_ENVIRONMENT=production
export QBITEL_AI_DB_HOST="prod-db-host"
export QBITEL_AI_DB_PASSWORD="<production-password>"

# Run migration
alembic upgrade head

# Verify migration
alembic current
```

4. **Verify Database State**

```bash
# Check tables
psql -U qbitel -h prod-db-host -d qbitel -c "\dt"

# Check indexes
psql -U qbitel -h prod-db-host -d qbitel -c "\di"

# Verify data integrity
psql -U qbitel -h prod-db-host -d qbitel -c "SELECT COUNT(*) FROM users;"
```

5. **Disable Maintenance Mode**

```bash
# Restore application
kubectl scale deployment qbitel --replicas=3
```

### Rollback Procedure

If migration fails:

```bash
# Rollback migration
alembic downgrade -1

# Restore from backup if needed
psql -U qbitel -h prod-db-host -d qbitel < backup_${TIMESTAMP}.sql

# Restart application
kubectl rollout restart deployment qbitel
```

## Migration Best Practices

### 1. Always Backup Before Migration

```bash
# Automated backup script
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/qbitel"
mkdir -p $BACKUP_DIR

pg_dump -U qbitel qbitel | gzip > $BACKUP_DIR/backup_${TIMESTAMP}.sql.gz

# Keep last 30 days of backups
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete
```

### 2. Test Migrations in Staging

Always test migrations in a staging environment that mirrors production:

```bash
# Copy production data to staging
pg_dump -U qbitel -h prod-db qbitel | psql -U qbitel -h staging-db qbitel

# Test migration in staging
alembic upgrade head

# Verify application functionality
./run_integration_tests.sh
```

### 3. Use Transactions

Alembic migrations run in transactions by default. Ensure your migrations are idempotent:

```python
def upgrade():
    # Check if table exists before creating
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if 'new_table' not in inspector.get_table_names():
        op.create_table('new_table', ...)
```

### 4. Document Breaking Changes

Add comments to migrations that introduce breaking changes:

```python
"""Add user_role column

BREAKING CHANGE: This migration adds a required 'role' column.
Existing users will be assigned 'viewer' role by default.

Revision ID: 002_add_user_roles
Revises: 001_initial_auth
"""
```

### 5. Monitor Migration Performance

```bash
# Enable query logging
export ALEMBIC_LOG_LEVEL=DEBUG

# Run migration with timing
time alembic upgrade head 2>&1 | tee migration.log
```

## Troubleshooting

### Migration Fails with "relation already exists"

```bash
# Check current schema
psql -U qbitel -d qbitel -c "\d"

# Mark migration as complete without running
alembic stamp head
```

### Database Connection Issues

```bash
# Test database connection
psql -U qbitel -h localhost -d qbitel -c "SELECT version();"

# Check environment variables
env | grep QBITEL_AI_DB
```

### Rollback Fails

```bash
# Force rollback to specific version
alembic downgrade <revision_id> --sql > rollback.sql

# Review SQL
cat rollback.sql

# Apply manually if needed
psql -U qbitel -d qbitel < rollback.sql
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Database Migration

on:
  push:
    branches: [main]
    paths:
      - 'ai_engine/alembic/versions/**'

jobs:
  migrate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install alembic psycopg2-binary
      
      - name: Run migrations
        env:
          QBITEL_AI_DB_HOST: ${{ secrets.DB_HOST }}
          QBITEL_AI_DB_PASSWORD: ${{ secrets.DB_PASSWORD }}
        run: |
          cd ai_engine
          alembic upgrade head
```

## Security Considerations

1. **Never commit database passwords** to version control
2. **Use environment variables** for all sensitive configuration
3. **Encrypt backups** before storing
4. **Audit migration logs** for security events
5. **Restrict migration permissions** to authorized personnel only

## Support

For migration issues:
- Check logs: `tail -f alembic.log`
- Review documentation: `docs/DATABASE_MIGRATIONS.md`
- Contact: devops@qbitel.com

---

# ============================================================
# FILE 12: docs/PRODUCTION_DEPLOYMENT_CHECKLIST.md
# ============================================================

# QBITEL - Production Deployment Checklist

**Version**: 2.1.0
**Last Updated**: 2025-11-22
**Purpose**: Pre-flight checklist for production deployment

---

## 1. Pre-Deployment Verification

### 1.1 Code Quality
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] No critical security vulnerabilities in dependencies
- [ ] Code review completed
- [ ] All TODO items resolved or tracked

### 1.2 Configuration
- [ ] Production configuration file reviewed (`config/qbitel.production.yaml`)
- [ ] All secrets stored in secrets manager (not in code/config)
- [ ] Environment variables documented
- [ ] TLS certificates valid and not expiring within 30 days

---

## 2. Security Checklist

### 2.1 Authentication & Authorization
- [ ] JWT secret is cryptographically strong (32+ chars)
- [ ] API keys are properly hashed
- [ ] MFA enabled for admin accounts
- [ ] OAuth2/SAML providers configured
- [ ] Session timeout configured appropriately

### 2.2 Network Security
- [ ] TLS 1.3 enabled
- [ ] HSTS headers configured
- [ ] CORS properly restricted
- [ ] Rate limiting enabled
- [ ] Input validation middleware active
- [ ] WAF rules configured (if applicable)

### 2.3 Data Security
- [ ] Database encryption at rest enabled
- [ ] PII data encrypted with AES-256-GCM
- [ ] Backup encryption enabled
- [ ] Post-quantum cryptography enabled (Kyber-1024, Dilithium-5)

### 2.4 Audit & Compliance
- [ ] Audit logging enabled
- [ ] SIEM integration configured (Splunk/Elastic/QRadar)
- [ ] Data retention policies applied
- [ ] Compliance framework selected (SOC2/HIPAA/PCI-DSS/GDPR)

---

## 3. Infrastructure Checklist

### 3.1 Database
- [ ] PostgreSQL 15+ deployed
- [ ] Connection pooling configured (50 connections for prod)
- [ ] Read replicas configured (if high availability required)
- [ ] Automated backups enabled
- [ ] Point-in-time recovery tested
- [ ] Database migrations applied (`alembic upgrade head`)

### 3.2 Caching
- [ ] Redis deployed
- [ ] Redis password set
- [ ] Redis persistence configured
- [ ] Cache TTL values reviewed

### 3.3 Message Queue (if applicable)
- [ ] Kafka/RabbitMQ deployed
- [ ] Topic retention policies set
- [ ] Consumer lag monitoring enabled

---

## 4. Kubernetes Deployment

### 4.1 Cluster Setup
- [ ] Kubernetes 1.24+ cluster ready
- [ ] Namespaces created (`qbitel`, `qbitel-service-mesh`)
- [ ] Resource quotas applied
- [ ] Network policies configured
- [ ] Pod security policies/standards applied

### 4.2 Workloads
- [ ] Deployments configured with proper resource limits
- [ ] Horizontal Pod Autoscaler (HPA) configured
- [ ] Pod Disruption Budgets (PDB) set
- [ ] Anti-affinity rules applied for high availability

### 4.3 Service Mesh (Istio)
- [ ] Istio installed and configured
- [ ] mTLS enabled between services
- [ ] Envoy sidecars injecting properly
- [ ] Traffic policies configured

### 4.4 Secrets Management
- [ ] Kubernetes secrets created (or external secrets operator configured)
- [ ] Secrets encrypted at rest in etcd
- [ ] RBAC for secrets access configured

---

## 5. Monitoring & Observability

### 5.1 Metrics
- [ ] Prometheus deployed
- [ ] ServiceMonitors configured
- [ ] Grafana dashboards imported
- [ ] Alert rules configured

### 5.2 Logging
- [ ] Structured logging enabled
- [ ] Log aggregation configured (ELK/Loki)
- [ ] Log retention policies set
- [ ] Correlation IDs enabled

### 5.3 Tracing
- [ ] OpenTelemetry/Jaeger configured
- [ ] Trace sampling rate set
- [ ] Service dependencies visible

### 5.4 Alerting
- [ ] PagerDuty/OpsGenie integration configured
- [ ] Slack notifications configured
- [ ] Alert escalation policies defined
- [ ] On-call rotation set up

---

## 6. Application Configuration

### 6.1 Required Environment Variables

```bash
# Database (REQUIRED)
DATABASE_HOST=<postgres-host>
DATABASE_PORT=5432
DATABASE_NAME=qbitel_prod
DATABASE_USER=<db-user>
DATABASE_PASSWORD=<from-secrets-manager>

# Redis (REQUIRED)
REDIS_HOST=<redis-host>
REDIS_PORT=6379
REDIS_PASSWORD=<from-secrets-manager>

# Security (REQUIRED)
JWT_SECRET=<32+-char-secret>
ENCRYPTION_KEY=<32-char-aes-key>

# TLS (REQUIRED for production)
TLS_ENABLED=true
TLS_CERT_FILE=/etc/qbitel/certs/server.crt
TLS_KEY_FILE=/etc/qbitel/certs/server.key

# LLM Configuration
QBITEL_LLM_PROVIDER=ollama  # or anthropic, openai
QBITEL_LLM_ENDPOINT=http://ollama:11434
QBITEL_AIRGAPPED_MODE=false

# Monitoring (RECOMMENDED)
SENTRY_DSN=<sentry-dsn>
LOG_LEVEL=INFO
METRICS_ENABLED=true

# Notifications (RECOMMENDED)
SLACK_WEBHOOK_URL=<webhook-url>
PAGERDUTY_ROUTING_KEY=<routing-key>

# SIEM (RECOMMENDED)
SIEM_TYPE=splunk  # or elastic, qradar
SPLUNK_HEC_URL=<hec-endpoint>
SPLUNK_HEC_TOKEN=<hec-token>

# Marketplace (if enabled)
STRIPE_API_KEY=<stripe-key>
STRIPE_WEBHOOK_SECRET=<webhook-secret>
AWS_ACCESS_KEY_ID=<aws-key>
AWS_SECRET_ACCESS_KEY=<aws-secret>
S3_BUCKET=qbitel-marketplace
```

### 6.2 Health Check Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `GET /health` | Basic health | `{"status": "healthy"}` |
| `GET /healthz` | Liveness probe | `200 OK` |
| `GET /readyz` | Readiness probe | `200 OK` (when dependencies ready) |
| `GET /metrics` | Prometheus metrics | Prometheus format |

---

## 7. Deployment Steps

### 7.1 Database Migration
```bash
# Apply migrations
cd ai_engine
alembic upgrade head

# Verify migration status
alembic current
```

### 7.2 Docker Deployment
```bash
# Build images
docker-compose -f ops/deploy/docker/docker-compose.yml build

# Deploy
docker-compose -f ops/deploy/docker/docker-compose.yml up -d

# Verify
docker-compose ps
curl http://localhost:8000/health
```

### 7.3 Kubernetes Deployment
```bash
# Create namespace
kubectl create namespace qbitel

# Apply secrets (from sealed-secrets or external-secrets)
kubectl apply -f kubernetes/secrets/

# Deploy via Helm
helm install qbitel ./helm/qbitel \
  --namespace qbitel \
  --values ./helm/qbitel/values-production.yaml

# Verify deployment
kubectl get pods -n qbitel
kubectl get svc -n qbitel
```

### 7.4 Post-Deployment Verification
```bash
# Health check
curl -k https://<service-url>/health

# API smoke test
curl -k https://<service-url>/api/v1/copilot/ \
  -H "Authorization: Bearer <token>"

# Check logs
kubectl logs -n qbitel -l app=qbitel --tail=100

# Check metrics
curl http://<service-url>:9090/metrics | grep qbitel
```

---

## 8. Rollback Procedure

### 8.1 Helm Rollback
```bash
# List releases
helm history qbitel -n qbitel

# Rollback to previous
helm rollback qbitel -n qbitel

# Rollback to specific revision
helm rollback qbitel <revision> -n qbitel
```

### 8.2 Database Rollback
```bash
# Downgrade to specific version
alembic downgrade <revision>

# Downgrade one step
alembic downgrade -1
```

---

## 9. Performance Tuning

### 9.1 Recommended Resource Limits

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|-----------|----------------|--------------|
| AI Engine | 2 | 4 | 4Gi | 8Gi |
| API Server | 1 | 2 | 2Gi | 4Gi |
| Worker | 2 | 4 | 4Gi | 8Gi |

### 9.2 Database Tuning
```yaml
# PostgreSQL settings for production
max_connections: 200
shared_buffers: 4GB
effective_cache_size: 12GB
work_mem: 64MB
maintenance_work_mem: 1GB
```

### 9.3 Connection Pool Settings
```yaml
database:
  pool_size: 50
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600
```

---

## 10. Disaster Recovery

### 10.1 Backup Schedule
- [ ] Database backups: Every 6 hours
- [ ] Configuration backups: Daily
- [ ] Model artifacts: After each training

### 10.2 Recovery Time Objectives
| Scenario | RTO | RPO |
|----------|-----|-----|
| Database failure | 30 min | 6 hours |
| Application failure | 5 min | 0 |
| Complete cluster failure | 2 hours | 6 hours |

### 10.3 DR Testing
- [ ] Backup restoration tested
- [ ] Failover procedure tested
- [ ] Recovery runbook documented

---

## 11. Go-Live Checklist

### Day Before Go-Live
- [ ] Final code freeze
- [ ] Staging environment validated
- [ ] Load testing completed
- [ ] Security scan completed
- [ ] Runbooks reviewed
- [ ] On-call team briefed

### Go-Live Day
- [ ] Deployment window communicated
- [ ] Monitoring dashboards open
- [ ] Support team on standby
- [ ] Rollback procedure ready
- [ ] Customer communication sent (if applicable)

### Post Go-Live
- [ ] Health checks passing
- [ ] No critical alerts
- [ ] Performance within SLA
- [ ] Customer feedback monitored
- [ ] Incident-free for 24 hours

---

## 12. Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Engineering Lead | | | |
| Security Lead | | | |
| DevOps Lead | | | |
| Product Owner | | | |

---

**Document maintained by**: QBITEL Engineering Team
**Review frequency**: Before each production deployment


---

# ============================================================
# FILE 13: SECURITY.md (Security Policy)
# ============================================================

# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

The QBITEL Bridge team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose any issues you find.

**Please do NOT report security vulnerabilities through public GitHub issues.**

### How to Report

Send an email to **security@qbitel.com** with the following information:

- A description of the vulnerability
- Steps to reproduce the issue
- Affected versions
- Any potential impact you have identified
- (Optional) Suggested fix or mitigation

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within **48 hours**.
- **Assessment**: We will investigate and provide an initial assessment within **5 business days**.
- **Resolution**: We aim to release a fix for confirmed vulnerabilities within **30 days**, depending on complexity.
- **Disclosure**: We will coordinate with you on public disclosure timing. We follow a 90-day disclosure policy.

### Safe Harbor

We consider security research conducted in good faith to be authorized. We will not pursue legal action against researchers who:

- Make a good faith effort to avoid privacy violations, data destruction, or service disruption
- Only interact with accounts they own or with explicit permission
- Report vulnerabilities promptly and do not exploit them beyond what is necessary to confirm the issue

## Security Best Practices for Deployment

- Always use TLS 1.3 in production (`tls_min_version: "TLSv1.3"`)
- Set all secrets via environment variables, never in config files
- Enable audit logging (`audit_logging_enabled: true`)
- Use the `verify-full` SSL mode for database connections in production
- Rotate JWT secrets and encryption keys regularly
- Enable post-quantum cryptography for forward secrecy
- Review the [deployment guide](DEPLOYMENT.md) for hardening recommendations


---

# ============================================================
# FILE 14: AGENTS.md (Agent Instructions)
# ============================================================

# QBITEL - Agent Instructions

This document provides context for AI coding agents working on the QBITEL codebase.

## Quick Start

```bash
# View all beads (memory items)
cat .beads/issues.jsonl | jq -s '.' | head -100

# Find specific topics
grep -i "legacy" .beads/issues.jsonl | jq '.'
grep -i "llm" .beads/issues.jsonl | jq '.'
```

## Project Overview

QBITEL is an enterprise-grade quantum-safe security platform with:
- **AI-powered protocol discovery** - Automatic protocol reverse engineering
- **Legacy system modernization** - COBOL/mainframe transformation
- **Post-quantum cryptography** - NIST Level 5 compliance
- **Cloud-native security** - Service mesh, container security

## Key Memory Beads

| Bead ID | Topic | Priority |
|---------|-------|----------|
| qbitel-0001 | Architecture Overview | P0 |
| qbitel-0017 | UC1: Legacy Mainframe Modernization | P0 |
| qbitel-0019 | Integration Architecture | P1 |
| qbitel-0026 | TODO: Integrate Demo with Production | P1 |

## Module Map

### Core AI Engine (`ai_engine/`)

```
ai_engine/
├── legacy/           # Legacy System Whisperer (~17K LOC)
│   ├── service.py              # Main orchestrator
│   ├── enhanced_detector.py    # VAE + LLM anomaly detection
│   ├── predictive_analytics.py # ML failure prediction
│   ├── decision_support.py     # LLM recommendations
│   └── knowledge_capture.py    # Tribal knowledge
│
├── llm/              # LLM Integration (~15K LOC)
│   ├── unified_llm_service.py  # Multi-provider LLM
│   ├── legacy_whisperer.py     # Protocol analysis
│   ├── rag_engine.py           # Hybrid search
│   └── translation_studio.py   # Protocol translation
│
├── discovery/        # Protocol Discovery (~62K LOC)
│   └── protocol_discovery_orchestrator.py  # PCFG learning
│
├── detection/        # Field Detection (~45K LOC)
│   └── field_detector.py       # BiLSTM-CRF model
│
├── cloud_native/     # Cloud Security
├── domains/          # Domain-specific (auto, aviation, healthcare)
└── agents/           # Autonomous agents
```

### Use Case Demos (`demos/`)

```
demos/
└── UC1_Legacy_Mainframe_Modernization/
    ├── backend/app.py          # FastAPI demo
    ├── cobol_samples/          # COBOL examples
    ├── run_demo.py             # Demo runner
    └── README.md
```

## Production vs Demo Code

**IMPORTANT**: The UC1 demo is an MVP that should be enhanced to use production modules.

| Feature | Demo (MVP) | Production Module |
|---------|------------|-------------------|
| COBOL Analysis | Basic regex | `ai_engine/legacy/service.py` |
| Protocol Analysis | Heuristics | `ai_engine/llm/legacy_whisperer.py` |
| Code Generation | Templates | LLM-powered with validation |
| Knowledge Capture | None | `ai_engine/legacy/knowledge_capture.py` |
| Failure Prediction | None | `ai_engine/legacy/predictive_analytics.py` |

## Key Classes

### LegacySystemWhispererService
```python
from ai_engine.legacy.service import LegacySystemWhispererService

service = LegacySystemWhispererService()
await service.initialize()
await service.register_legacy_system(system_context)
health = await service.analyze_system_health(system_id)
```

### UnifiedLLMService
```python
from ai_engine.llm.unified_llm_service import UnifiedLLMService

llm = UnifiedLLMService()
response = await llm.process_request(prompt, model="claude-sonnet")
```

### LegacyWhisperer
```python
from ai_engine.llm.legacy_whisperer import LegacyWhisperer

whisperer = LegacyWhisperer(llm_service)
spec = await whisperer.reverse_engineer_protocol(samples, context)
code = await whisperer.generate_adapter_code(spec, "python")
```

## Configuration

- **Development**: `config/qbitel.yaml`
- **Production**: `config/qbitel.production.yaml`
- **Compliance**: `config/compliance.yaml`
- **Environment**: `QBITEL_AI_*` prefix

## Testing

```bash
# Run all tests
pytest ai_engine/tests/ -v

# Run legacy module tests
pytest ai_engine/legacy/tests/ -v

# Run with coverage
pytest --cov=ai_engine --cov-report=html
```

## Common Tasks

### 1. Enhance UC1 Demo
See bead `qbitel-0026` for details on integrating with production code.

### 2. Add New Protocol Support
1. Add samples to `datasets/protocols/`
2. Update `ai_engine/llm/translation_studio.py`
3. Add tests

### 3. Add New LLM Provider
1. Update `ai_engine/llm/unified_llm_service.py`
2. Add provider configuration
3. Update fallback chain

## Beads Memory System

This project uses [Beads](https://github.com/steveyegge/beads) for AI agent memory.

```bash
# Memory is stored in .beads/issues.jsonl
# Each line is a JSON object with:
# - id: Unique identifier (qbitel-XXXX)
# - title: Brief description
# - description: Detailed documentation
# - status: open|closed|pinned
# - labels: Categories
# - priority: 0-4 (0 = highest)
```

## Contact

For questions about this codebase, refer to the beads memory system or the documentation in `docs/`.


---

# ============================================================
# FILE 15: PQC_IMPLEMENTATION_PLAN.md
# ============================================================

# Domain-Specific Post-Quantum Cryptography Implementation Plan

## QBITEL / QBITEL Bridge - Strategic Roadmap

**Version**: 1.0
**Date**: December 2025
**Status**: Strategic Planning Phase

---

## Executive Summary

Based on comprehensive analysis of the PQC research landscape and the existing QBITEL codebase, this document outlines a strategic implementation plan to extend the platform's quantum-safe capabilities into domain-specific constrained environments.

### Current State Assessment

QBITEL already possesses **production-grade PQC infrastructure**:
- ✅ NIST Level 5 algorithms (Kyber-1024, Dilithium-5)
- ✅ Hybrid encryption (Classical + PQC)
- ✅ HSM integration via PKCS#11
- ✅ Automated key lifecycle management (8,461 LOC Rust implementation)
- ✅ Service mesh integration (Istio/Envoy)
- ✅ Container signing with Dilithium
- ✅ Air-gapped deployment capability

### Strategic Opportunity

The research identifies critical gaps between NIST standards and enterprise reality. QBITEL is uniquely positioned to address these gaps through domain-specific optimizations targeting:

1. **Healthcare** - Constrained medical devices (64KB RAM, 10+ year battery)
2. **Automotive V2X** - Sub-10ms latency, 1000+ msg/sec verification
3. **Aviation** - Bandwidth-constrained channels (600bps-2.4kbps)
4. **Industrial OT/ICS** - Safety-critical certification requirements

---

## Part 1: Gap Analysis - Research vs. Current Capabilities

### 1.1 Signature Size Challenge

| Metric | Current Implementation | Research Target | Gap |
|--------|----------------------|-----------------|-----|
| ML-DSA Signature | 4,595 bytes | <1KB for constrained | 4.5x reduction needed |
| Certificate Chain | ~15-17KB | <10KB (ossification) | Compression required |
| TLS Handshake | ~17KB PQC overhead | <5KB for mobile | Hybrid optimization |

**Current Code Location**: `/rust/dataplane/crates/pqc_tls/src/kyber.rs`

### 1.2 Performance Benchmarks vs. Requirements

| Domain | Required Latency | Current Capability | Status |
|--------|-----------------|-------------------|--------|
| Enterprise TLS | <50ms | <10ms | ✅ Exceeds |
| V2X Authentication | <10ms | Not optimized | ⚠️ Gap |
| Medical Device | Battery-constrained | Full power | ⚠️ Gap |
| Aviation ACARS | 600bps bandwidth | Standard PQC | ❌ Critical Gap |

### 1.3 Algorithm Coverage

| Algorithm | Status | Use Case |
|-----------|--------|----------|
| Kyber-1024 | ✅ Implemented | Key encapsulation |
| Dilithium-5 | ✅ Implemented | Digital signatures |
| ML-KEM-768 | ⚠️ Not implemented | Hybrid TLS |
| SLH-DSA/SPHINCS+ | ⚠️ Partial (dependency only) | Backup signatures |
| LMS/XMSS | ❌ Not implemented | Stateful hash-based |
| Falcon | ❌ Not implemented | Compact signatures |

---

## Part 2: Implementation Phases

### Phase 1: Foundation Enhancement (Q1 2026)

**Objective**: Strengthen core PQC infrastructure and add missing algorithms

#### 1.1 Algorithm Expansion

```
Priority 1: ML-KEM-768 (Hybrid TLS standard)
Priority 2: Falcon-512/1024 (50% smaller signatures than Dilithium)
Priority 3: SLH-DSA production integration
Priority 4: LMS/XMSS for stateful applications
```

**Implementation Tasks**:

| Task | Location | Effort | Priority |
|------|----------|--------|----------|
| Add ML-KEM-768 to Rust crate | `/rust/dataplane/crates/pqc_tls/` | Medium | P1 |
| Integrate Falcon via pqcrypto-falcon | `/rust/dataplane/crates/pqc_tls/` | Medium | P1 |
| SLH-DSA wrapper implementation | `/ai_engine/cloud_native/` | Low | P2 |
| LMS/XMSS state management | New module | High | P2 |
| Update certificate manager | `qkd_certificate_manager.py` | Medium | P1 |

**Files to Modify**:
- `/rust/dataplane/crates/pqc_tls/Cargo.toml` - Add new dependencies
- `/rust/dataplane/crates/pqc_tls/src/lib.rs` - New algorithm exports
- `/ai_engine/requirements.txt` - Python dependencies

#### 1.2 Hybrid Certificate Implementation

Based on IETF draft standards (expected early 2026):

```rust
// Proposed structure in /rust/dataplane/crates/pqc_tls/src/hybrid.rs
pub struct HybridCertificate {
    classical: X509Certificate,      // ECDSA-P384
    pqc: PQCCertificate,             // ML-DSA or Falcon
    binding: CertificateBinding,      // Cryptographic binding
}
```

**Key Considerations**:
- Support both composite and non-composite formats
- Backward compatibility with classical-only clients
- Size optimization through compression

#### 1.3 TLS 1.3 PQC Integration

Extend existing Rust TLS implementation:

```rust
// /rust/dataplane/crates/pqc_tls/src/tls13.rs
pub enum HybridKexGroup {
    X25519MLKEM768,      // Chrome/Cloudflare default
    P384MLKEM1024,       // Enterprise preference
    X25519Kyber768,      // Legacy compatibility
}
```

**Performance Target**: <0.5% latency increase on initial handshake

---

### Phase 2: Domain-Specific Modules (Q2-Q3 2026)

#### 2.1 Healthcare Module

**Target Constraints**:
- 64KB RAM (pacemaker-class devices)
- 10+ year battery life
- FDA 510(k) / PMA compliance

**Architecture**:

```
┌─────────────────────────────────────────────────────────┐
│           Healthcare PQC Module                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ Lightweight    │  │ External       │  │ FHIR       │ │
│  │ PQC (ML-KEM-512)│ │ Security Shield│  │ PQC Profile│ │
│  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ Battery-Aware  │  │ HL7 v2/FHIR   │  │ HIPAA      │ │
│  │ Key Management │  │ Integration    │  │ Compliance │ │
│  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Implementation Tasks**:

| Task | Description | Effort |
|------|-------------|--------|
| ML-KEM-512 optimization | Reduced security level for constrained devices | High |
| External security shield | Proxy-based PQC for legacy implantables | High |
| FHIR PQC security profile | Quantum-safe SMART-on-FHIR | Medium |
| Battery-aware scheduling | Defer crypto operations to optimal times | Medium |
| HL7 integration | PQC transport for HL7 v2 messages | Low |

**New Files**:
```
/ai_engine/domains/healthcare/
├── __init__.py
├── lightweight_pqc.py          # Optimized algorithms
├── security_shield.py          # External protection proxy
├── fhir_pqc_profile.py         # FHIR security extensions
├── hl7_transport.py            # HL7 v2 integration
├── battery_scheduler.py        # Power-aware operations
└── fda_compliance.py           # Regulatory requirements
```

**Key Innovation**: External Security Shield

Concept from MIT's IMDShield research - provide PQC protection for legacy devices that cannot be modified:

```python
# /ai_engine/domains/healthcare/security_shield.py
class MedicalDeviceSecurityShield:
    """
    External proxy providing PQC protection for legacy medical devices.
    Operates between device and network, handling all quantum-safe crypto.
    """

    def __init__(self, device_profile: DeviceProfile):
        self.pqc_engine = LightweightPQCEngine(memory_limit_kb=32)
        self.session_cache = SecureSessionCache()

    async def protect_outbound(self, plaintext: bytes) -> bytes:
        """Encrypt device traffic with PQC before transmission"""

    async def protect_inbound(self, ciphertext: bytes) -> bytes:
        """Decrypt incoming PQC traffic for legacy device"""
```

#### 2.2 Automotive V2X Module

**Target Constraints**:
- 100ms BSM broadcast interval (10 Hz)
- <10ms signature verification
- 1,000+ messages/second in dense scenarios
- 15-20 year fleet turnover timeline

**Architecture**:

```
┌─────────────────────────────────────────────────────────┐
│           Automotive V2X PQC Module                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ Batch          │  │ Implicit Cert  │  │ SCMS       │ │
│  │ Verification   │  │ PQC Scheme     │  │ Integration│ │
│  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ ARMv8          │  │ Legacy Vehicle │  │ SAE J2735  │ │
│  │ Optimization   │  │ Compatibility  │  │ Compliance │ │
│  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Implementation Tasks**:

| Task | Description | Effort |
|------|-------------|--------|
| Batch verification engine | 65-90% signature operation reduction | High |
| PQ implicit certificates | Research PQCMC integration | High |
| ARMv8 NEON optimization | Target NXP MK6 chipset | High |
| P2PCD adaptation | Certificate distribution for large certs | Medium |
| IEEE 1609.2 extension | PQC SPDU format support | Medium |
| Backward compatibility | Hybrid mode for legacy vehicles | Medium |

**New Files**:
```
/ai_engine/domains/automotive/
├── __init__.py
├── batch_verification.py       # Parallel signature verification
├── implicit_certificates.py    # PQC implicit cert scheme
├── armv8_optimization.py       # Platform-specific optimization
├── v2x_protocol.py             # IEEE 1609.2 / SAE J2735
├── p2pcd_adapter.py            # Certificate distribution
└── fleet_compatibility.py      # Legacy vehicle support
```

**Key Innovation**: Batch Verification

Based on NDSS 2024 research, achieve 65-90% reduction in signature operations:

```rust
// /rust/dataplane/crates/pqc_tls/src/batch.rs
pub struct BatchVerifier {
    pending: Vec<SignedMessage>,
    batch_size: usize,           // Optimal: 32-64 messages
    parallel_workers: usize,     // Match CPU cores
}

impl BatchVerifier {
    pub async fn verify_batch(&mut self) -> Vec<VerificationResult> {
        // Aggregate signature verification
        // Use SIMD for parallel hash computation
        // Amortize lattice operations across batch
    }
}
```

**Performance Target**: 1,000+ verifications/second with <10ms P99 latency

#### 2.3 Aviation Module

**Target Constraints**:
- VHF ACARS: 2.4 kbps
- Classic SATCOM: 600 bps
- 30+ year aircraft lifecycle
- DO-326A/ED-202A certification

**Architecture**:

```
┌─────────────────────────────────────────────────────────┐
│           Aviation PQC Module                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ Signature      │  │ LDACS         │  │ ADS-B      │ │
│  │ Compression    │  │ Integration    │  │ Auth       │ │
│  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ ARINC 653     │  │ Bandwidth      │  │ DO-326A    │ │
│  │ Integration    │  │ Optimization   │  │ Compliance │ │
│  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Implementation Tasks**:

| Task | Description | Effort |
|------|-------------|--------|
| Signature compression | Target <1KB for VHF ACARS | Critical |
| LDACS PQC profile | Post-SIKE algorithm integration | High |
| ADS-B authenticator | Lightweight position/speed auth | High |
| ARINC 653 partition | IMA-compliant PQC module | Medium |
| Delta encoding | Send signature differences | Medium |
| Session resumption | Minimize key exchange frequency | Low |

**New Files**:
```
/ai_engine/domains/aviation/
├── __init__.py
├── signature_compression.py    # Advanced compression
├── ldacs_profile.py            # L-band integration
├── adsb_authenticator.py       # ADS-B security
├── arinc653_partition.py       # IMA integration
├── bandwidth_optimizer.py      # Channel-aware crypto
└── certification_support.py    # DO-326A evidence
```

**Key Innovation**: Adaptive Signature Compression

For 600bps SATCOM channels, standard PQC is impossible. Novel approach:

```python
# /ai_engine/domains/aviation/signature_compression.py
class AdaptiveSignatureCompressor:
    """
    Multi-strategy compression for bandwidth-constrained aviation channels.
    Achieves 60-80% size reduction through:
    1. Zstandard dictionary compression with domain-specific training
    2. Delta encoding for sequential messages
    3. Session-based signature aggregation
    4. Selective field authentication (position vs. full message)
    """

    def __init__(self, channel_bandwidth_bps: int):
        self.strategy = self._select_strategy(channel_bandwidth_bps)

    def compress_signature(
        self,
        signature: bytes,
        previous_signatures: List[bytes]
    ) -> CompressedSignature:
        """
        Target: <600 bytes for VHF ACARS (2.4kbps)
        Target: <150 bytes for SATCOM (600bps) using aggregation
        """
```

#### 2.4 Industrial OT/ICS Module

**Target Constraints**:
- IEC 61508 SIL 1-4 certification
- Deterministic timing (WCET analysis)
- 10+ year operational lifecycle
- Air-gapped networks

**Architecture**:

```
┌─────────────────────────────────────────────────────────┐
│           Industrial OT/ICS PQC Module                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ Deterministic  │  │ Safety        │  │ Protocol   │ │
│  │ PQC Engine     │  │ Certification │  │ Adaptation │ │
│  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ SCADA/Modbus  │  │ DNP3          │  │ IEC 62443  │ │
│  │ Integration    │  │ Secure Auth   │  │ Compliance │ │
│  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Implementation Tasks**:

| Task | Description | Effort |
|------|-------------|--------|
| Constant-time implementation | Eliminate timing side channels | Critical |
| WCET analysis framework | Safety certification support | High |
| Modbus/TCP PQC wrapper | Protocol-level protection | Medium |
| DNP3 Secure Auth v6 | PQC extension | Medium |
| IEC 62443 compliance | Security level mapping | Medium |
| Offline key provisioning | Air-gapped key management | Low |

**New Files**:
```
/ai_engine/domains/industrial/
├── __init__.py
├── deterministic_pqc.py        # Constant-time implementation
├── wcet_analyzer.py            # Timing analysis
├── modbus_wrapper.py           # Modbus/TCP protection
├── dnp3_secure_auth.py         # DNP3 SA v6 extension
├── iec62443_compliance.py      # Security level mapping
└── offline_provisioning.py     # Air-gapped key management
```

**Key Innovation**: Deterministic PQC for Safety-Critical Systems

```rust
// /rust/dataplane/crates/pqc_tls/src/deterministic.rs
/// Constant-time PQC implementation for IEC 61508 / ISO 26262
/// All operations complete within guaranteed WCET bounds
pub struct DeterministicPQCEngine {
    wcet_sign_us: u64,      // Worst-case signing time
    wcet_verify_us: u64,    // Worst-case verification time
    wcet_encap_us: u64,     // Worst-case encapsulation time
    jitter_budget_us: u64,  // Allowed timing variance
}

impl DeterministicPQCEngine {
    /// Returns Some(signature) only if completed within WCET
    /// Returns None if operation would exceed safety bounds
    pub fn sign_with_deadline(
        &self,
        message: &[u8],
        deadline_us: u64
    ) -> Option<Signature> {
        // Pre-check: Can we meet the deadline?
        if deadline_us < self.wcet_sign_us {
            return None;
        }
        // Execute with timing guards
    }
}
```

---

### Phase 3: Advanced Cryptographic Primitives (Q4 2026)

#### 3.1 Post-Quantum Zero-Knowledge Proofs

**Current State**: 450× size gap vs. classical (16KB-58KB vs. 128 bytes)

**Implementation Approach**:

| Primitive | Target Size | Use Case |
|-----------|-------------|----------|
| LaBRADOR (R1CS) | 58KB | General ZK proofs |
| ISW21 (designated) | 16KB | Identity verification |
| LaZer integration | Variable | Anonymous credentials |

**New Files**:
```
/ai_engine/crypto/zkp/
├── __init__.py
├── labrador.py                 # LaBRADOR wrapper
├── designated_verifier.py      # ISW21 implementation
├── anonymous_credentials.py    # PQ credential scheme
└── kyber_key_proofs.py         # Prove Kyber key knowledge
```

#### 3.2 Post-Quantum Anonymous Credentials

Target: <100KB total credential size (vs. current 317-724KB)

**Strategy**: Adapt Cloudflare's Phoenix signature research

```python
# /ai_engine/crypto/zkp/anonymous_credentials.py
class PQAnonymousCredential:
    """
    Post-quantum anonymous credential based on lattice assumptions.

    Target sizes:
    - Credential: ~80KB (vs. 317-724KB state-of-art)
    - Presentation proof: <7KB
    - Verification: <50ms
    """

    def issue(self, attributes: Dict[str, Any]) -> Credential:
        """Issue credential with hidden attributes"""

    def present(
        self,
        credential: Credential,
        disclosed: Set[str],
        predicates: List[Predicate]
    ) -> Presentation:
        """Generate ZK presentation revealing only selected attributes"""
```

#### 3.3 Post-Quantum Verifiable Random Functions

Based on LB-VRF research (Esgin et al.):

```python
# /ai_engine/crypto/vrf/
class LatticePQVRF:
    """
    Lattice-based VRF for quantum-resistant randomness.

    Sizes:
    - VRF value: 84 bytes
    - Proof: ~5KB

    Performance:
    - Evaluation: ~3ms
    - Verification: ~1ms
    """
```

---

### Phase 4: Ecosystem Integration (Q1-Q2 2027)

#### 4.1 Cloud HSM Integration

Extend existing PKCS#11 integration to cloud HSMs:

| Provider | Service | Status |
|----------|---------|--------|
| AWS | CloudHSM | Planned |
| Azure | Managed HSM | Planned |
| GCP | Cloud HSM | Planned |
| Thales | Luna 7.9.0 | Available now |
| Utimaco | Quantum Protect | Available now |

**Files to Extend**:
- `/rust/dataplane/crates/pqc_tls/src/hsm.rs`

#### 4.2 5G/6G Network Integration

Based on 3GPP SA3 PQC roadmap (Release 19/20):

| Component | Integration Point | Priority |
|-----------|------------------|----------|
| SUCI protection | KEMSUCI scheme | High |
| Base station auth | Hybrid certificates | High |
| Edge computing | PQES offloading | Medium |
| IoT authentication | Lightweight KEM | Medium |

**New Files**:
```
/ai_engine/domains/telecom/
├── __init__.py
├── kemsuci.py                  # Post-quantum SUCI
├── base_station_auth.py        # 5G infrastructure
├── edge_offload.py             # PQES integration
└── iot_lightweight.py          # Constrained IoT devices
```

#### 4.3 Blockchain/DLT Integration

Leverage existing lattice-based HE (inherently PQ-safe):

| Platform | Integration | Use Case |
|----------|-------------|----------|
| Algorand | LB-VRF | Proof-of-stake |
| Hyperledger | PQ signatures | Permissioned chains |
| Ethereum L2 | zkSTARKs | Scalability |

---

## Part 3: Technical Implementation Details

### 3.1 Library Strategy

**Rust (Performance-Critical)**:
```toml
# /rust/dataplane/crates/pqc_tls/Cargo.toml additions
[dependencies]
pqcrypto-mlkem = "0.1"          # ML-KEM (new NIST name)
pqcrypto-falcon = "0.3"          # Falcon signatures
pqcrypto-sphincsplus = "0.7"     # SPHINCS+/SLH-DSA
lms = "0.1"                      # LMS signatures
xmss = "0.1"                     # XMSS signatures
```

**Python (Flexibility)**:
```python
# /ai_engine/requirements.txt additions
falcon-py>=0.1.0                 # Falcon signatures
sphincs-py>=0.1.0                # SPHINCS+
liboqs-python>=0.9.0             # Updated OQS
```

### 3.2 API Design

**Unified PQC Interface**:

```python
# /ai_engine/crypto/pqc_unified.py
from enum import Enum
from typing import Union

class PQCAlgorithm(Enum):
    # Key Encapsulation
    KYBER_512 = "kyber-512"
    KYBER_768 = "kyber-768"
    KYBER_1024 = "kyber-1024"
    MLKEM_512 = "ml-kem-512"
    MLKEM_768 = "ml-kem-768"
    MLKEM_1024 = "ml-kem-1024"

    # Signatures
    DILITHIUM_2 = "dilithium-2"
    DILITHIUM_3 = "dilithium-3"
    DILITHIUM_5 = "dilithium-5"
    FALCON_512 = "falcon-512"
    FALCON_1024 = "falcon-1024"
    SPHINCS_SHA2_128F = "sphincs-sha2-128f"
    SPHINCS_SHA2_256F = "sphincs-sha2-256f"
    LMS = "lms"
    XMSS = "xmss"

class DomainProfile(Enum):
    ENTERPRISE = "enterprise"           # Standard security
    HEALTHCARE = "healthcare"           # Constrained devices
    AUTOMOTIVE = "automotive"           # Real-time V2X
    AVIATION = "aviation"               # Bandwidth constrained
    INDUSTRIAL = "industrial"           # Safety-critical
    TELECOM = "telecom"                 # 5G/6G networks

class PQCEngine:
    """
    Unified PQC engine with domain-specific optimization.
    """

    def __init__(
        self,
        domain: DomainProfile,
        algorithm: PQCAlgorithm = None,    # Auto-select if None
        hybrid: bool = True                 # Classical + PQC
    ):
        self.domain = domain
        self.algorithm = algorithm or self._auto_select_algorithm()
        self.hybrid = hybrid

    def _auto_select_algorithm(self) -> PQCAlgorithm:
        """Select optimal algorithm for domain constraints"""
        domain_defaults = {
            DomainProfile.ENTERPRISE: PQCAlgorithm.KYBER_1024,
            DomainProfile.HEALTHCARE: PQCAlgorithm.MLKEM_512,
            DomainProfile.AUTOMOTIVE: PQCAlgorithm.FALCON_512,
            DomainProfile.AVIATION: PQCAlgorithm.FALCON_512,
            DomainProfile.INDUSTRIAL: PQCAlgorithm.KYBER_768,
            DomainProfile.TELECOM: PQCAlgorithm.MLKEM_768,
        }
        return domain_defaults[self.domain]
```

### 3.3 Performance Optimization Strategy

**Hardware Acceleration Targets**:

| Platform | Optimization | Expected Speedup |
|----------|-------------|------------------|
| x86-64 AVX-512 | SIMD lattice ops | 3-5× |
| x86-64 AVX2 | SIMD fallback | 2-3× |
| ARMv8 NEON | Mobile/automotive | 2-4× |
| ARMv8 SVE2 | Server ARM | 3-5× |
| RISC-V V | Emerging | 2-3× |

**Existing Optimization** (from `/rust/dataplane/crates/pqc_tls/src/config.rs`):
```rust
#[cfg(target_feature = "avx512f")]
mod avx512_optimized;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
mod avx2_optimized;

#[cfg(target_arch = "aarch64")]
mod neon_optimized;
```

### 3.4 Testing Strategy

**Test Categories**:

| Category | Coverage Target | Location |
|----------|-----------------|----------|
| Unit Tests | 90%+ | `/ai_engine/tests/crypto/` |
| Integration | 80%+ | `/ai_engine/tests/integration/` |
| Performance | All algorithms | `/tests/perf/` |
| Conformance | NIST vectors | `/tests/conformance/` |
| Fuzz Testing | All inputs | `/tests/fuzz/` |
| Domain-specific | Each module | `/ai_engine/domains/*/tests/` |

---

## Part 4: Compliance & Certification Roadmap

### 4.1 Regulatory Timeline

| Regulation | Deadline | Status |
|------------|----------|--------|
| NSA CNSA 2.0 (prefer PQC) | 2025 | ✅ Ready |
| NSA CNSA 2.0 (exclusive) | 2030 | 🔄 On track |
| OMB M-23-02 | 2035 | 🔄 Planning |
| UK NCSC Phase 1 | 2028 | 🔄 On track |
| UK NCSC Phase 2 | 2031 | 🔄 Planning |
| UK NCSC Phase 3 | 2035 | 🔄 Planning |

### 4.2 Certification Targets

| Certification | Target Level | Timeline |
|---------------|-------------|----------|
| FIPS 140-3 | Level 2+ | Q2 2026 |
| Common Criteria | EAL 4+ | Q4 2026 |
| ISO 27001 | Certified | Q1 2026 |
| SOC 2 Type II | Compliant | Q2 2026 |
| FedRAMP | High | Q4 2026 |
| HIPAA | Compliant | Q1 2026 |
| PCI-DSS | Level 1 | Q2 2026 |

### 4.3 Domain-Specific Certifications

| Domain | Certification | Target |
|--------|--------------|--------|
| Healthcare | FDA 510(k)/PMA cyber | Q3 2026 |
| Automotive | ISO 26262 ASIL B | Q4 2026 |
| Aviation | DO-326A/ED-202A | 2027 |
| Industrial | IEC 62443 SL2+ | Q2 2026 |

---

## Part 5: Resource Requirements

### 5.1 Development Team

| Role | Count | Focus Area |
|------|-------|------------|
| Cryptography Engineers | 2-3 | Algorithm implementation |
| Rust Developers | 2 | Performance optimization |
| Python Developers | 2 | Domain modules |
| Domain Specialists | 4 | Healthcare/Auto/Aviation/Industrial |
| Security Researchers | 1-2 | Novel primitives |
| QA Engineers | 2 | Testing & validation |
| Compliance Specialist | 1 | Certification |

### 5.2 Infrastructure

| Resource | Specification | Purpose |
|----------|--------------|---------|
| Dev Servers | 32-core, 128GB RAM | Build & test |
| HSM (Dev) | Thales Luna Network | Key management testing |
| CI/CD | GitHub Actions + Self-hosted | Continuous testing |
| Test Vehicles | 2-3 V2X equipped | Automotive testing |
| Medical Device Lab | FDA-compliant | Healthcare testing |

### 5.3 Partnerships

| Partner Type | Purpose | Priority |
|--------------|---------|----------|
| Automotive OEM | V2X validation | High |
| Medical Device Mfr | FDA pathway | High |
| Aviation Integrator | DO-326A evidence | Medium |
| HSM Vendor | Hardware integration | High |
| Academic | Novel research | Medium |

---

## Part 6: Risk Assessment

### 6.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Algorithm weakness discovered | Low | Critical | Crypto-agility architecture |
| Performance targets missed | Medium | High | Early benchmarking, fallback options |
| Certification delays | Medium | High | Early engagement, parallel tracks |
| Interoperability issues | Medium | Medium | Standards participation, testing |

### 6.2 Market Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Standards changes | Medium | Medium | Active IETF/NIST participation |
| Competitor advancement | Medium | Medium | Patent portfolio, innovation speed |
| Customer adoption delays | Medium | Medium | Hybrid approach, backward compat |

### 6.3 Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Key talent departure | Low | High | Knowledge transfer, documentation |
| Supply chain issues (HSM) | Low | Medium | Multi-vendor strategy |
| Quantum timeline acceleration | Low | Critical | Aggressive roadmap execution |

---

## Part 7: Success Metrics

### 7.1 Technical KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| TLS handshake latency | <50ms (99th %ile) | Continuous benchmark |
| V2X verification rate | 1,000+ msg/sec | Field testing |
| Healthcare device battery | <5% crypto overhead | Lab testing |
| Aviation bandwidth efficiency | 80%+ standard PQC | Protocol analysis |
| Safety-critical WCET variance | <10% jitter | Timing analysis |

### 7.2 Business KPIs

| Metric | Target | Timeline |
|--------|--------|----------|
| Domain modules released | 4 | Q4 2026 |
| Enterprise customers | 50+ | Q2 2027 |
| Compliance certifications | 8+ | Q4 2026 |
| Patent applications | 5+ | Q4 2026 |
| Revenue from PQC | $5M+ ARR | Q4 2027 |

### 7.3 Adoption KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| PQC key exchange adoption | 90%+ connections | Telemetry |
| Hybrid certificate deployment | 50%+ customers | Customer survey |
| Domain module usage | 20%+ by sector | Product analytics |

---

## Part 8: Immediate Action Items

### Week 1-2

- [ ] Finalize algorithm selection (ML-KEM-768, Falcon priority)
- [ ] Scope Phase 1 Rust implementation
- [ ] Identify domain specialist partners
- [ ] Initiate FIPS 140-3 pre-assessment

### Week 3-4

- [ ] Begin ML-KEM-768 integration
- [ ] Draft hybrid certificate format based on IETF drafts
- [ ] Establish healthcare partnership discussions
- [ ] Set up automotive V2X test environment

### Month 2

- [ ] Complete ML-KEM-768 implementation
- [ ] Begin Falcon integration
- [ ] Draft domain module architectures
- [ ] Initiate certification readiness assessment

### Month 3

- [ ] Release Phase 1 internal beta
- [ ] Begin healthcare module development
- [ ] Begin automotive module development
- [ ] Complete TLS 1.3 PQC integration

---

## Appendix A: Algorithm Quick Reference

| Algorithm | Type | Public Key | Signature/Ciphertext | Security Level |
|-----------|------|-----------|---------------------|----------------|
| ML-KEM-512 | KEM | 800 B | 768 B | 1 (128-bit) |
| ML-KEM-768 | KEM | 1,184 B | 1,088 B | 3 (192-bit) |
| ML-KEM-1024 | KEM | 1,568 B | 1,568 B | 5 (256-bit) |
| ML-DSA-44 | Sig | 1,312 B | 2,420 B | 2 (128-bit) |
| ML-DSA-65 | Sig | 1,952 B | 3,293 B | 3 (192-bit) |
| ML-DSA-87 | Sig | 2,592 B | 4,595 B | 5 (256-bit) |
| Falcon-512 | Sig | 897 B | 666 B | 1 (128-bit) |
| Falcon-1024 | Sig | 1,793 B | 1,280 B | 5 (256-bit) |
| SLH-DSA-128f | Sig | 32 B | 17,088 B | 1 (128-bit) |
| SLH-DSA-256f | Sig | 64 B | 49,856 B | 5 (256-bit) |

---

## Appendix B: Competitive Positioning

### Market Position

QBITEL differentiates through:

1. **Domain-Specific Optimization**: No competitor offers healthcare/automotive/aviation modules
2. **Agentic AI Integration**: Autonomous security decisions with PQC
3. **Hybrid Maturity**: Production-proven classical + PQC approach
4. **Air-Gapped Capability**: Critical for government/defense
5. **End-to-End Platform**: Discovery → Protection → Compliance

### Competitor Analysis

| Competitor | Strengths | Gaps vs. QBITEL |
|------------|-----------|-----------------|
| PQShield | First FIPS 140-3 cert | No domain optimization |
| ISARA | Strong PKI tools | No AI integration |
| Crypto4A | Crypto-agile HSM | No cloud-native |
| SandboxAQ | Enterprise orchestration | No constrained device support |

---

## Appendix C: Reference Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QBITEL PQC Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Domain Modules Layer                       │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │   │
│  │  │Healthcare│  │Automotive│ │Aviation │  │Industrial/OT   │  │   │
│  │  │ Module   │  │ V2X      │ │ Module  │  │ Module         │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Advanced Primitives Layer                  │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │   │
│  │  │ PQ-ZKP  │  │ PQ-Creds│  │ PQ-VRF │  │ PQ-FPE          │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Core PQC Engine (Rust)                     │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │   │
│  │  │ ML-KEM  │  │ ML-DSA  │  │ Falcon │  │ SLH-DSA/LMS    │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘  │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │   │
│  │  │Key Mgmt │  │HSM PKCS#11│ │Hybrid  │  │ Batch Verify   │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Integration Layer                          │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │   │
│  │  │TLS 1.3  │  │Istio/   │  │ Cloud  │  │ Protocol        │  │   │
│  │  │ PQC     │  │ Envoy   │  │ HSMs   │  │ Discovery       │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Document Version: 1.0*
*Last Updated: December 2025*
*Next Review: January 2026*


---

# END OF COMPLETE TECHNICAL REFERENCE DOCUMENT
