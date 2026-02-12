# QBITEL Bridge - Complete Technical Reference

**Version**: 2.1.0
**Last Updated**: 2026-02-12
**Purpose**: Comprehensive technical reference document consolidating the complete QBITEL Bridge platform architecture, implementation details, APIs, deployment guides, and operational procedures. This document is self-contained and designed for use with NotebookLM.

---

## Table of Contents

1. [Platform Overview and Key Metrics](#1-platform-overview-and-key-metrics)
2. [Four-Layer Polyglot Architecture](#2-four-layer-polyglot-architecture)
3. [Multi-Agent AI System](#3-multi-agent-ai-system)
4. [AI/ML Pipeline](#4-aiml-pipeline)
5. [Post-Quantum Cryptography](#5-post-quantum-cryptography)
6. [Zero-Touch Security Engine](#6-zero-touch-security-engine)
7. [LLM Intelligence Layer](#7-llm-intelligence-layer)
8. [Cloud-Native Infrastructure](#8-cloud-native-infrastructure)
9. [Complete API Reference](#9-complete-api-reference)
10. [Database Architecture](#10-database-architecture)
11. [Environment Configuration](#11-environment-configuration)
12. [Deployment Guide](#12-deployment-guide)
13. [Security Architecture](#13-security-architecture)
14. [Testing Strategy](#14-testing-strategy)
15. [Technology Stack](#15-technology-stack)
16. [Performance Benchmarks](#16-performance-benchmarks)
17. [Project Structure](#17-project-structure)
18. [Implementation Plan](#18-implementation-plan)
19. [Contributing and Changelog](#19-contributing-and-changelog)

---

## 1. Platform Overview and Key Metrics

### 1.1 What is QBITEL Bridge?

QBITEL Bridge is an enterprise-grade, open-source platform for AI-powered quantum-safe legacy modernization. It is the only open-source platform that discovers unknown protocols with AI, encrypts them with post-quantum cryptography, and defends them autonomously -- without replacing legacy systems.

QBITEL Bridge is 100% open source under the Apache License 2.0 -- the same license trusted by Kubernetes, Kafka, Spark, and Airflow. No open-core. No feature gating. No bait-and-switch.

### 1.2 The Problem Statement

Sixty percent of Fortune 500 companies run critical operations on systems built 20-40 years ago. These systems process trillions of dollars daily, keep power grids running, and manage patient records. They share three fatal weaknesses:

**The Legacy Crisis**: Undocumented protocols with no source code, no documentation, and no original developers. Manual reverse engineering costs $2M-10M and takes 6-12 months per system.

**The Quantum Threat**: Quantum computers will break RSA/ECC within 5-10 years. Adversaries are harvesting encrypted data today to decrypt later. Every unprotected wire transfer, patient record, and grid command is a future breach.

**The Speed Gap**: Average SOC response time: 65 minutes. In that window, an attacker can exfiltrate 100GB, encrypt an entire network, or manipulate industrial controls. Human speed cannot match machine-speed attacks.

### 1.3 How It Works

```
  Legacy System                    QBITEL Bridge                         Modern Infrastructure
                        +----------------------------------+
  COBOL Mainframes      |                                  |          Cloud APIs
  SCADA / PLCs     ---> |  1. DISCOVER  unknown protocols  | --->     Microservices
  Medical Devices       |  2. PROTECT   with quantum crypto|          Dashboards
  Banking Terminals     |  3. TRANSLATE to modern APIs     |          Event Streams
  Custom Protocols      |  4. COMPLY    across 9 frameworks|          Data Lakes
                        |  5. OPERATE   with autonomous AI |
                        +----------------------------------+
                                 2-4 hours to first results
```

| Step | What Happens | Traditional Approach | With QBITEL |
|------|-------------|---------------------|-------------|
| **Discover** | AI learns protocol structure from raw traffic | 6-12 months, $2-10M | **2-4 hours**, automated |
| **Protect** | Wraps communications in NIST Level 5 PQC | Not available | **ML-KEM + ML-DSA**, <1ms |
| **Translate** | Generates REST APIs + SDKs in 6 languages | Weeks of manual coding | **Minutes**, auto-generated |
| **Comply** | Produces audit-ready reports for 9 frameworks | $500K-1M/year manual | **<10 min**, automated |
| **Operate** | Autonomous threat detection and response | 65 min avg SOC response | **<1 sec**, 78% autonomous |

### 1.4 Key Metrics

| Metric | Value |
|--------|-------|
| Discovery Accuracy | 89%+ |
| Quantum Security Level | NIST Level 5 |
| Encryption Overhead | <1ms |
| Autonomous Response Rate | 78% |
| Compliance Frameworks | 9 |
| Message Throughput | 100,000+ msg/sec |
| Python Files | 442+ |
| Go Files | 15+ |
| Rust Files | 64+ |
| TypeScript/React Files | 39+ |
| Total Lines of Code | 250,000+ |
| Documentation Files | 82 markdown files |
| Test Files | 160+ |
| Test Coverage | 85% |
| Production Readiness | 85-90% |

### 1.5 Platform Capabilities

**AI Protocol Discovery**: Reverse-engineers unknown protocols from raw network traffic using PCFG grammar inference, Transformer classification, and BiLSTM-CRF field detection. No source code needed.

**Post-Quantum Cryptography**: NIST Level 5 protection with ML-KEM (Kyber-1024) and ML-DSA (Dilithium-5). Domain-optimized for banking (10K+ TPS), healthcare (64KB devices), automotive (<10ms V2X), and aviation (600bps links).

**Zero-Touch Security Engine**: LLM-powered autonomous response with confidence-driven execution. Auto-executes at >95% confidence, escalates at <50%. MITRE ATT&CK mapping. Full audit trail.

**Legacy System Whisperer**: Deep analysis for undocumented legacy systems: COBOL copybook parsing, JCL analysis, mainframe dataset discovery, business rule extraction, and predictive failure analysis.

**Translation Studio**: Point at any protocol, get a REST API with OpenAPI 3.0 spec + SDKs in Python, TypeScript, Go, Java, Rust, and C#. Auto-generated, fully documented, production-ready.

**Protocol Marketplace**: Community-driven protocol knowledge sharing with 1,000+ pre-built adapters. Publish, discover, and monetize protocol definitions. Automated validation pipeline with security scanning.

**Multi-Agent Orchestration**: 16+ specialized AI agents with 5 execution strategies (parallel, sequential, pipeline, consensus, adaptive). Persistent memory, dynamic scaling, coordinated incident response.

**Enterprise Compliance**: Automated reporting for SOC 2, GDPR, HIPAA, PCI-DSS, ISO 27001, NIST 800-53, BASEL-III, NERC-CIP, FDA 21 CFR Part 11. Continuous monitoring. Blockchain-backed audit trails.

**Cloud-Native Security**: Kubernetes admission webhooks, Istio/Envoy service mesh with quantum-safe mTLS, eBPF runtime monitoring, container image scanning, and secure Kafka event streaming.

**Threat Intelligence**: MITRE ATT&CK technique mapping, STIX/TAXII feed integration, proactive threat hunting, and continuous learning from global threat landscape.

### 1.6 Competitive Differentiation

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

### 1.7 Industry Solutions

| Industry | Protocols | Key Use Case | Business Impact |
|----------|-----------|-------------|-----------------|
| **Banking & Finance** | ISO-8583, SWIFT, SEPA, FIX | Mainframe transaction protection, HSM migration | Protect $10T+ daily transactions |
| **Healthcare** | HL7, DICOM, FHIR | Medical device security without FDA recertification | Secure 500K+ connected devices |
| **Critical Infrastructure** | Modbus, DNP3, IEC 61850 | SCADA/PLC protection with zero downtime | Protect power grids serving 100M+ |
| **Automotive** | V2X, IEEE 1609.2, CAN | Quantum-safe vehicle-to-everything | <10ms latency constraint |
| **Aviation** | ADS-B, ACARS, ARINC 429 | Air traffic and avionics data security | 600bps bandwidth-optimized PQC |
| **Telecommunications** | SS7, Diameter, SIP | 5G core and IoT infrastructure protection | Billion-device scale |

---

## 2. Four-Layer Polyglot Architecture

QBITEL uses a four-layer polyglot architecture where each layer is built in the language optimized for its job.

### 2.1 Architecture Diagram

```
+--------------------------------------------------------------------------+
|                        UI Console  (React / TypeScript)                    |
|               Admin Dashboard  .  Protocol Copilot  .  Marketplace         |
+--------------------------------------------------------------------------+
|                        Go Control Plane                                    |
|          Service Orchestration  .  OPA Policy Engine  .  Vault Secrets     |
|          Device Agent (TPM 2.0)  .  gRPC + REST Gateway                    |
+--------------------------------------------------------------------------+
|                        Python AI Engine  (FastAPI)                         |
|   Protocol Discovery  .  Multi-Agent System  .  LLM/RAG (Ollama-first)     |
|   Compliance Automation  .  Anomaly Detection  .  Security Orchestrator    |
|   Legacy Whisperer  .  Marketplace  .  Protocol Copilot                    |
+--------------------------------------------------------------------------+
|                        Rust Data Plane                                     |
|     PQC-TLS (ML-KEM / ML-DSA)  .  DPDK Packet Processing                   |
|     Wire-Speed Encryption  .  DPI Engine  .  Protocol Adapters             |
|     K8s Operator  .  HA Clustering  .  AI Bridge (PyO3)                    |
+--------------------------------------------------------------------------+
```

### 2.2 Layer Responsibilities

| Layer | Language | What It Does | Port |
|-------|----------|-------------|------|
| **Data Plane** | Rust | PQC-TLS termination, wire-speed encryption, DPDK packet processing, protocol adapters (ISO-8583, Modbus, HL7, TN3270e) | N/A |
| **AI Engine** | Python | Protocol discovery, LLM inference, compliance automation, anomaly detection, multi-agent orchestration | 8000 (REST), 50051 (gRPC) |
| **Control Plane** | Go | Service orchestration, OPA policy evaluation, Vault secrets, device management, gRPC gateway | 8080, 8081 |
| **UI Console** | React/TS | Admin dashboard, protocol copilot, marketplace, real-time monitoring | 3000 |

### 2.3 High-Level Component Diagram

```
+-------------------------------------------------------------------------+
|                           QBITEL Platform                                |
+-------------------------------------------------------------------------+
|                                                                          |
|  +--------------+  +--------------+  +--------------+                    |
|  |   AI Engine  |  | Cloud-Native |  |  Compliance  |                    |
|  |   Protocol   |  |   Security   |  |  Automation  |                    |
|  |   Discovery  |  |              |  |              |                    |
|  +--------------+  +--------------+  +--------------+                    |
|                                                                          |
|  +--------------+  +--------------+  +--------------+                    |
|  | Service Mesh |  |  Container   |  |    Event     |                    |
|  | Integration  |  |  Security    |  |  Streaming   |                    |
|  | (Istio/Envoy)|  | (eBPF/Trivy) |  |   (Kafka)   |                    |
|  +--------------+  +--------------+  +--------------+                    |
|                                                                          |
|  +-----------------------------------------------------+                |
|  |         Post-Quantum Cryptography Layer              |                |
|  |     Kyber-1024, Dilithium-5, AES-256-GCM            |                |
|  +-----------------------------------------------------+                |
|                                                                          |
+-------------------------------------------------------------------------+
```

### 2.4 Microservices Architecture

| Service | Language | Purpose | Port |
|---------|----------|---------|------|
| AI Engine | Python | Core ML/AI processing | 8000 (REST), 50051 (gRPC) |
| Control Plane | Go | Policy engine, configuration | 8080 |
| Management API | Go | Device lifecycle, attestation | 8081 |
| Device Agent | Go | On-device security agent | N/A |
| Dataplane | Rust | High-performance packet processing | N/A |
| Admin Console | TypeScript/React | Web UI dashboard | 3000 |

### 2.5 Entry Points

| Entry Point | Location | Command |
|-------------|----------|---------|
| AI Engine | `ai_engine/__main__.py` | `python -m ai_engine` |
| API Server | `ai_engine/api/server.py` | Automatic via __main__ |
| Control Plane | `go/controlplane/cmd/controlplane/main.go` | `go run main.go` |
| Management API | `go/mgmtapi/cmd/mgmtapi/main.go` | `go run main.go` |
| Dataplane | `rust/dataplane/src/main.rs` | `cargo run` |
| Console UI | `ui/console/src/main.tsx` | `npm run dev` |

### 2.6 Rust Data Plane Details

Location: `rust/dataplane/`

The Rust data plane provides wire-speed cryptographic operations and protocol parsing for performance-critical paths.

**Workspace Members**:
```
crates/
  adapter_sdk/     - Core L7Adapter trait
  pqc_tls/         - Post-quantum TLS (Kyber-768, Dilithium)
  io_engine/       - High-performance bridge with HA
  dpdk_engine/     - DPDK kernel bypass
  dpi_engine/      - Deep packet inspection + ML
  k8s_operator/    - Kubernetes operator
  ai_bridge/       - Python interop (PyO3)
  adapters/
    iso8583/     - Financial protocol (PCI DSS)
    modbus/      - Industrial SCADA
    tn3270e/     - Mainframe terminal
    hl7_mllp/    - Healthcare HL7
```

**Core Adapter Trait** (`adapter_sdk/src/lib.rs`):
```rust
#[async_trait]
pub trait L7Adapter: Send + Sync {
    async fn to_upstream(&self, input: Bytes) -> Result<Bytes, AdapterError>;
    async fn to_client(&self, input: Bytes) -> Result<Bytes, AdapterError>;
    fn name(&self) -> &'static str;
}
```

**I/O Engine Architecture**:
```
Client -> Reader -> Adapter.to_upstream() -> Channel -> Writer -> Upstream
Upstream -> Reader -> Adapter.to_client() -> Channel -> Writer -> Client
```

Features include bidirectional protocol transformation bridge, active-standby HA with automatic failover, connection pooling with lifecycle management, circuit breaker pattern, lock-free high-frequency metrics, and distributed tracing via OpenTelemetry.

**DPI Engine** (`crates/dpi_engine/`):
- PyTorch via `tch` bindings
- ONNX Runtime via `ort`
- Candle framework
- Hyperscan for pattern matching
- Classifies HTTP, HTTPS, SSH, DNS, SIP, RTP, BitTorrent, Skype, WhatsApp, Telegram, Zoom, Teams, Slack, Netflix, YouTube

**Protocol Adapters**:

| Adapter | Protocol | Features |
|---------|----------|----------|
| **ISO-8583** | Financial/Payment | PCI DSS masking, field encryption, MAC, routing |
| **Modbus** | Industrial/SCADA | TCP/RTU, CRC validation, register access control |
| **TN3270E** | Mainframe Terminal | Screen scraping, session management |
| **HL7 MLLP** | Healthcare | MLLP framing, field mapping, validation |

**Rust Coding Standards**:
```rust
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

// Builder pattern
impl Iso8583AdapterBuilder {
    pub fn with_parser(mut self, parser: Iso8583Parser) -> Self {
        self.parser = Some(parser);
        self
    }
    pub fn build(self) -> Result<Iso8583Adapter, Iso8583Error> { ... }
}
```

### 2.7 Go Control Plane Details

Location: `go/`

**Control Plane Service** (`go/controlplane/`):

Dependencies (Go 1.22): `github.com/gin-gonic/gin`, `github.com/open-policy-agent/opa`, `github.com/sigstore/cosign/v2`, `go.uber.org/zap`

API Endpoints:
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

Key Components: `PolicyEngine` (OPA-based policy evaluation with caching), `BundleManager` (Policy bundle lifecycle with filesystem, S3, GCS, OCI storage), `CosignVerifier` (Signature verification for policy bundles), `VaultClient` (HashiCorp Vault integration for key management).

**Management API Service** (`go/mgmtapi/`):

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

Key Components: `DeviceLifecycleManager`, `CertificateManager` (PKI with Root CA and Intermediate CA), `AttestationVerifier` (TPM quote and PCR verification).

**Device Agent** (`go/agents/device-agent/`):
Runs on IoT/edge devices for TPM attestation. Key features: TPM 2.0 key sealing and unsealing, PCR-based attestation quotes, EK/AK certificate management, periodic attestation (5-minute interval), enrollment with control plane.

Environment Variables:
```bash
DEVICE_ID=<device-id>
CONTROL_URL=https://control.qbitel.local
```

### 2.8 UI Console Details

Location: `ui/console/`

Technology Stack: React 18.2 with TypeScript 5.3, Material-UI 5.14 (MUI), Vite 5.0 for bundling, OIDC authentication, WebSocket for real-time updates.

**Directory Structure**:
```
ui/console/
  src/
    main.tsx              - Entry point
    App.tsx               - Main shell
    api/
      devices.ts        - Device API client
      marketplace.ts    - Marketplace API
      enhancedApiClient.ts - Cached API with WebSocket
    auth/
      oidc.ts           - OIDC service
    components/
      Dashboard.tsx
      DeviceManagement.tsx
      PolicyManagement.tsx
      SecurityMonitoring.tsx
      ProtocolVisualization.tsx
      AIModelMonitoring.tsx
      ThreatIntelligence.tsx
      marketplace/      - Marketplace components
    routes/
      AppRoutes.tsx     - RBAC routing
    theme/
      EnterpriseTheme.ts - 5 theme variants
    types/
      auth.ts, device.ts, marketplace.ts
  package.json
  vite.config.ts
  tsconfig.json
```

**Routes with RBAC**:

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

**5 Theme Variants**: Light, Dark, High Contrast, Blue-Grey, Enterprise.

**WebSocket Real-Time Message Types**: PROTOCOL_DISCOVERED, PROTOCOL_UPDATED, PROTOCOL_METRICS, MODEL_METRICS, MODEL_ALERT, MODEL_TRAINING_COMPLETE, THREAT_DETECTED, THREAT_RESOLVED, SECURITY_ALERT, HEARTBEAT, SYSTEM_STATUS, NOTIFICATION.

**OIDC Configuration**:
```typescript
{
  authority: 'https://auth.qbitel.local',
  clientId: 'qbitel-console',
  scope: 'openid profile email roles permissions organization',
  automaticSilentRenew: true,
}
```

User Roles: `admin`, `system_admin`, `org_admin`, `security_analyst`, `operator`, `viewer`.

---

## 3. Multi-Agent AI System

QBITEL implements a multi-agent architecture where specialized AI agents autonomously manage security operations, protocol discovery, and threat response. The system uses LLM-powered reasoning, autonomous orchestration, and adaptive learning to minimize human intervention while maintaining safety and explainability.

### 3.1 Agent Ecosystem Overview

```
+------------------------------------------------------------------+
|                   QBITEL Agent Ecosystem                          |
+------------------------------------------------------------------+
|                                                                    |
|  +---------------------+         +---------------------+          |
|  |  Zero-Touch         |         |   Protocol          |          |
|  |  Decision Agent     |<------->|   Discovery Agent   |          |
|  |  (Security)         |         |   (Learning)        |          |
|  +----------+----------+         +----------+----------+          |
|             |                               |                      |
|             |    +---------------------+    |                      |
|             +--->|  Security           |<---+                      |
|                  |  Orchestrator Agent |                           |
|                  |  (Coordination)     |                           |
|                  +----------+----------+                           |
|                             |                                      |
|             +---------------+---------------+                      |
|             |               |               |                      |
|  +----------v------+ +-----v------+ +------v-----------+          |
|  |  Threat         | |  Response  | |  Compliance      |          |
|  |  Analyzer       | |  Executor  | |  Monitor         |          |
|  |  (Analysis)     | |  (Action)  | |  (Governance)    |          |
|  +-----------------+ +------------+ +------------------+          |
|                                                                    |
|  +-----------------------------------------------------+          |
|  |         On-Premise LLM Intelligence Layer            |          |
|  |   Ollama (Primary), vLLM, LocalAI + RAG Engine       |          |
|  |        Air-Gapped Deployment Supported                |          |
|  +-----------------------------------------------------+          |
|                                                                    |
+------------------------------------------------------------------+
```

### 3.2 Agent Inventory (16+ Agents)

The platform has 16+ specialized AI agents with 5 execution strategies: parallel, sequential, pipeline, consensus, and adaptive.

**Core Agents**:

1. **Zero-Touch Decision Agent** (`ai_engine/security/decision_engine.py`, 1,360+ LOC): Primary autonomous decision-making component providing LLM-powered security analysis and response.

2. **Protocol Discovery Agent** (`ai_engine/discovery/protocol_discovery_orchestrator.py`, 885+ LOC): Autonomous multi-phase protocol learning system.

3. **Security Orchestrator Agent** (`ai_engine/security/security_service.py`, 500+ LOC): Central coordination agent managing security incident lifecycle.

4. **Threat Analyzer Agent**: ML-based classification with MITRE ATT&CK mapping.

5. **Response Executor Agent**: Executes security responses (blocking, isolation, shutdown).

6. **Compliance Monitor Agent**: SOC 2, GDPR, PCI-DSS, HIPAA control validation.

7. **Protocol Copilot Agent**: Natural language protocol queries and threat assessment.

8. **Legacy Whisperer Agent**: COBOL analysis, copybook parsing, business rule extraction.

9. **Anomaly Detection Agent**: Ensemble anomaly detection (VAE, LSTM, Isolation Forest).

10. **RAG Engine Agent**: Retrieval-augmented generation for contextual intelligence.

11. **Translation Studio Agent**: Protocol translation and API/SDK generation.

12. **Marketplace Agent**: Protocol publishing, discovery, and validation.

13. **Service Integration Orchestrator**: Message routing and load balancing.

14. **Cloud Platform Integration Agents**: AWS Security Hub, Azure Sentinel, GCP SCC.

15. **Container Security Agent**: Image scanning, admission control, runtime monitoring.

16. **Threat Intelligence Agent**: STIX/TAXII feed integration and threat hunting.

### 3.3 Autonomous Decision Metrics

```
Agent Performance Dashboard
|
+-- Zero-Touch Decision Agent
|   +-- Autonomous Execution Rate: 78%
|   +-- Human Escalation Rate: 15%
|   +-- Average Confidence: 0.87
|   +-- Decision Latency P95: 180ms
|   +-- Accuracy (validated): 94%
|
+-- Protocol Discovery Agent
|   +-- Successful Discoveries: 1,247
|   +-- Average Discovery Time: 120ms
|   +-- Cache Hit Rate: 82%
|   +-- Parser Accuracy: 89%
|   +-- Background Learning Jobs: 42/day
|
+-- Security Orchestrator
|   +-- Active Incidents: 12 / 50
|   +-- Avg Incident Resolution: 4.2 min
|   +-- Auto-Resolved: 67%
|   +-- Escalated: 18%
|   +-- False Positives: 3%
|
+-- LLM Intelligence Layer
    +-- Total LLM Requests: 15,420/day
    +-- Avg Response Time: 850ms
    +-- Token Usage: 2.4M tokens/day
    +-- Cost per Decision: $0.012
    +-- Fallback Activations: 0.8%
```

### 3.4 Agent Interaction Patterns

**Pattern 1: Security Event Response**

```
Security Event
      |
      v
+----------------------+
| Security Orchestrator|
+----------+-----------+
           |
           +-----------------------------+
           |                             |
           v                             v
+------------------+          +------------------+
| Threat Analyzer  |          | Protocol Copilot |
| (LLM Analysis)   |          | (Context)        |
+--------+---------+          +--------+---------+
         |                              |
         +--------------+---------------+
                        v
              +------------------+
              | Decision Agent   |
              | (LLM Decision)   |
              +--------+---------+
                       |
                       v
              +------------------+
              | Response Executor|
              +--------+---------+
                       |
                       v
              +------------------+
              | Outcome Tracking |
              +------------------+
```

**Pattern 2: Protocol Discovery**

```
Unknown Protocol Data
         |
         v
+----------------------+
| Service Integration  |
| Orchestrator         |
+----------+-----------+
           |
           v
+----------------------+
| Protocol Discovery   |
| Agent                |
+----------+-----------+
           |
           +----------------------+
           |                      |
           v                      v
+------------------+   +------------------+
| Grammar Learner  |   | Field Detector   |
| (Transformer)    |   | (BiLSTM-CRF)     |
+--------+---------+   +--------+---------+
         |                      |
         +----------+-----------+
                    v
          +------------------+
          | Parser Generator |
          +--------+---------+
                   |
                   v
          +------------------+
          | Protocol Profile |
          | Database         |
          +------------------+
```

### 3.5 Safety and Governance Architecture

```
Safety & Governance Layer
|
+-- Explainability
|   +-- LIME Explainer (Local Interpretable Explanations)
|   +-- Decision Reasoning Logs
|   +-- Feature Importance Tracking
|   +-- Human-Readable Narratives
|
+-- Audit Trail
|   +-- Complete Decision History
|   +-- Input/Output Logging
|   +-- Timestamp & User Tracking
|   +-- Compliance Event Logging
|   +-- Tamper-Proof Storage
|
+-- Drift Detection
|   +-- Model Performance Monitoring
|   +-- Distribution Shift Detection
|   +-- Accuracy Degradation Alerts
|   +-- Automatic Retraining Triggers
|
+-- Compliance Automation
|   +-- SOC 2 Control Validation
|   +-- GDPR Privacy Checks
|   +-- PCI-DSS Security Validation
|   +-- Automated Reporting
|
+-- Safety Constraints
    +-- Confidence Threshold Enforcement
    +-- Risk Category Validation
    +-- Human-in-the-Loop Escalation
    +-- Rollback Mechanisms
    +-- Kill Switch (Emergency Stop)
```

---

## 4. AI/ML Pipeline

### 4.1 Protocol Discovery 5-Phase Pipeline

```python
class DiscoveryPhase(Enum):
    INITIALIZATION = "initialization"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    CLASSIFICATION = "classification"
    GRAMMAR_LEARNING = "grammar_learning"
    PARSER_GENERATION = "parser_generation"
    VALIDATION = "validation"
    COMPLETION = "completion"
```

**Phase 1 - Statistical Analysis**: Entropy calculation, byte frequency analysis, pattern detection, preliminary classification.

**Phase 2 - Protocol Classification**: ML model inference, feature extraction, confidence scoring, protocol type identification.

**Phase 3 - Grammar Learning (PCFG)**: PCFG rule extraction, Transformer-based learning, structure discovery, grammar validation.

**Phase 4 - Parser Generation**: Parser template selection, code generation, validation testing, performance optimization.

**Phase 5 - Adaptive Learning** (Background): Continuous model training, success rate tracking, grammar refinement, protocol profile updates.

```
Raw Protocol Data
      |
      v
+-----------------+
| Statistical     |
| Analysis        |
+--------+--------+
         |
         v
+-----------------+      +--------------+
| Classification  |----->| Check Cache  |->[Cache Hit]-->Return Result
+--------+--------+      +--------------+
         |                                        ^
         |[Cache Miss]                            |
         v                                        |
+-----------------+                               |
| Grammar         |                               |
| Learning        |                               |
+--------+--------+                               |
         |                                        |
         v                                        |
+-----------------+                               |
| Parser          |                               |
| Generation      |                               |
+--------+--------+                               |
         |                                        |
         v                                        |
+-----------------+                               |
| Validation      |                               |
+--------+--------+                               |
         |                                        |
         v                                        |
+-----------------+                               |
| Store Profile   |-------------------------------+
| Update Cache    |
+-----------------+
         |
         v
+-----------------+
| Background      |
| Adaptive Learn  |
+-----------------+
```

### 4.2 Data Flow - Protocol Discovery

```
1. Data Ingestion
   +-- Network TAP/Mirror -> Raw Packets
   +-- Packet Capture -> Binary Data
   +-- Preprocessing -> Normalized Format

2. Feature Extraction
   +-- Statistical Features (entropy, frequency)
   +-- Structural Features (length, patterns)
   +-- Contextual Features (n-grams, position)

3. Protocol Discovery
   +-- PCFG Inference -> Grammar Rules
   +-- Classification -> Protocol Type
   +-- Confidence Scoring -> Validation

4. Field Detection
   +-- BiLSTM-CRF -> Field Boundaries
   +-- Semantic Classification -> Field Types
   +-- Validation -> Accuracy Check

5. Model Output
   +-- Protocol Grammar
   +-- Field Mappings
   +-- Confidence Scores
   +-- Metadata
```

### 4.3 Key Algorithms

**PCFG Inference** (`ai_engine/discovery/pcfg_inference.py`):
- Input: Binary protocol data
- Output: Probabilistic context-free grammar
- Algorithm: Iterative rule extraction with statistical validation

**BiLSTM-CRF** (`ai_engine/detection/field_detector.py`):
- Architecture: Bidirectional LSTM (128 hidden units) + Conditional Random Fields
- Tagging: IOB (Inside-Outside-Begin) scheme
- Tags: O = Outside, B-FIELD = Begin field, I-FIELD = Inside field
- Accuracy: 89%+ on production protocols

**VAE Anomaly Detection** (`ai_engine/anomaly/vae_detector.py`):
- Encoder: 3-layer neural network
- Latent space: 64 dimensions
- Decoder: 3-layer neural network
- Threshold: Dynamic based on reconstruction error distribution

### 4.4 Model Architecture Summary

| Model | Type | Purpose | Location |
|-------|------|---------|----------|
| BiLSTM-CRF | Deep Learning | Field detection | `detection/field_detector.py` |
| LSTM | Deep Learning | Time-series anomaly | `anomaly/lstm_detector.py` |
| VAE | Deep Learning | Feature extraction | `anomaly/vae_detector.py` |
| Isolation Forest | Ensemble | Anomaly detection | `anomaly/isolation_forest.py` |
| PCFG | Statistical | Grammar inference | `discovery/pcfg_inference.py` |
| CNN | Deep Learning | Protocol classification | Various |

### 4.5 Training Configuration

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

### 4.6 Explainability

| Tool | Purpose | Location |
|------|---------|----------|
| SHAP | Feature importance | `explainability/shap_explainer.py` |
| LIME | Local explanations | `explainability/lime_explainer.py` |
| Audit Logger | Decision trails | `explainability/audit_logger.py` |
| Drift Monitor | Performance tracking | `explainability/drift_monitor.py` |

### 4.7 Discovery Request/Result Data Structures

```python
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

---

## 5. Post-Quantum Cryptography

### 5.1 Current Implementation

QBITEL already possesses production-grade PQC infrastructure with 8,461 LOC in the Rust implementation:

- NIST Level 5 algorithms (Kyber-1024, Dilithium-5)
- Hybrid encryption (Classical + PQC)
- HSM integration via PKCS#11
- Automated key lifecycle management
- Service mesh integration (Istio/Envoy)
- Container signing with Dilithium
- Air-gapped deployment capability

### 5.2 Kyber KEM (Key Encapsulation Mechanism)

```
Kyber Implementation
|
+-- Kyber-1024 (NIST Level 5)
|   +-- Public Key: 1568 bytes
|   +-- Private Key: 3168 bytes
|   +-- Ciphertext: 1568 bytes
|   +-- Shared Secret: 32 bytes
|   +-- Security Level: ~256-bit quantum security
|
+-- Kyber-768 (NIST Level 3)
|   +-- ~192-bit quantum security
|
+-- Kyber-512 (NIST Level 1)
    +-- ~128-bit quantum security
```

### 5.3 Dilithium Signatures

```
Dilithium Implementation
|
+-- Dilithium-5 (NIST Level 5)
|   +-- Public Key: 2592 bytes
|   +-- Private Key: 4864 bytes
|   +-- Signature: ~4595 bytes
|   +-- Security Level: ~256-bit quantum security
|
+-- Dilithium-3 (NIST Level 3)
|   +-- ~192-bit quantum security
|
+-- Dilithium-2 (NIST Level 1)
    +-- ~128-bit quantum security
```

### 5.4 Algorithm Quick Reference

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

### 5.5 Hybrid Encryption Flow

```
1. Key Encapsulation
   +-- Generate ephemeral key pair (Kyber-1024)
   +-- Encapsulate shared secret
   +-- Transmit ciphertext

2. Symmetric Encryption
   +-- Derive AES-256 key from shared secret
   +-- Encrypt data with AES-256-GCM
   +-- Generate authentication tag (128-bit)
   +-- Include IV/nonce

3. Digital Signature
   +-- Sign encrypted message (Dilithium-5)
   +-- Include signature in message
   +-- Verify on receiver side
```

### 5.6 Rust PQC-TLS Implementation

Location: `rust/dataplane/crates/pqc_tls/`

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

Features: Kyber-768 KEM, Dilithium signatures, hardware acceleration (AVX-512, AVX2, AES-NI), HSM integration via PKCS#11, automated key rotation with threat-level awareness, memory pool for crypto buffers.

Dependencies: `pqcrypto-kyber`, `pqcrypto-dilithium`, `oqs`, `tokio-openssl`, `pkcs11`, `rayon`, `wide`.

Performance Targets: Key generation <1ms, Encapsulation <0.5ms, Signature <2ms, TLS handshake <5ms.

### 5.7 Domain-Specific PQC Profiles

```
Rust Data Plane - Domain-Specific PQC Profiles
+-- Banking (SWIFT/SEPA constraints)
+-- Healthcare (pacemaker/insulin pump power limits)
+-- Automotive (V2X/IEEE 1609.2 latency)
+-- Aviation (ADS-B bandwidth constraints)
+-- Industrial (IEC 62351/SCADA cycles)
```

### 5.8 PQC Implementation Roadmap

**Phase 1 (Q1 2026) - Foundation Enhancement**:
- ML-KEM-768 (Hybrid TLS standard)
- Falcon-512/1024 (50% smaller signatures than Dilithium)
- SLH-DSA production integration
- LMS/XMSS for stateful applications
- TLS 1.3 PQC integration

**Phase 2 (Q2-Q3 2026) - Domain-Specific Modules**:
- Healthcare: Lightweight PQC for 64KB devices, external security shield, FHIR PQC profile
- Automotive V2X: Batch verification (65-90% reduction), <10ms P99 latency for 1,000+ verifications/sec
- Aviation: Signature compression for 600bps channels, ADS-B authentication
- Industrial OT/ICS: Deterministic PQC, WCET analysis, Modbus/DNP3 wrappers

**Phase 3 (Q4 2026) - Advanced Primitives**:
- Post-quantum zero-knowledge proofs
- Post-quantum anonymous credentials
- Post-quantum verifiable random functions

**Phase 4 (Q1-Q2 2027) - Ecosystem Integration**:
- Cloud HSM integration (AWS, Azure, GCP)
- 5G/6G network integration
- Blockchain/DLT integration

### 5.9 Security Properties

- **Confidentiality**: AES-256-GCM encryption
- **Integrity**: AEAD authentication tags
- **Authentication**: Dilithium-5 digital signatures
- **Forward Secrecy**: Ephemeral Kyber keys
- **Quantum Resistance**: NIST Level 5 algorithms

---

## 6. Zero-Touch Security Engine

### 6.1 Architecture

Location: `ai_engine/security/decision_engine.py` (1,360+ LOC)

```
Zero-Touch Decision Agent
|
+-- Input Processing
|   +-- Security Event Ingestion
|   +-- Event Normalization
|   +-- Context Enrichment
|
+-- Threat Analysis Pipeline
|   +-- ML-Based Classification
|   |   +-- Anomaly Score Calculation
|   |   +-- Threat Type Detection
|   |   +-- MITRE ATT&CK Mapping
|   |
|   +-- LLM Contextual Analysis
|   |   +-- Threat Narrative Generation
|   |   +-- TTP Extraction
|   |   +-- Historical Correlation
|   |   +-- Confidence Scoring
|   |
|   +-- Business Impact Assessment
|       +-- Financial Risk Calculation
|       +-- Operational Impact Analysis
|       +-- Regulatory Implications
|
+-- Response Generation
|   +-- Response Options Creation
|   |   +-- Low-Risk Actions (alerts, logging)
|   |   +-- Medium-Risk Actions (blocking, segmentation)
|   |   +-- High-Risk Actions (isolation, shutdown)
|   |
|   +-- LLM Decision Making
|   |   +-- Multi-Response Evaluation
|   |   +-- Risk-Benefit Analysis
|   |   +-- Confidence Scoring (0.0-1.0)
|   |   +-- Recommended Action Selection
|   |
|   +-- Safety Constraints
|       +-- Confidence Threshold Checks
|       +-- Risk Category Validation
|       +-- Escalation Logic
|
+-- Execution Control
|   +-- Autonomous Execution (confidence > 0.95, low-risk only)
|   +-- Auto-Approval (confidence > 0.85, medium-risk)
|   +-- Human Escalation (confidence < 0.50 OR high-risk)
|   +-- Fallback Handling
|
+-- Learning & Metrics
    +-- Decision Outcome Tracking
    +-- Success Rate by Threat Type
    +-- Autonomous Execution Metrics
    +-- Model Performance Updates
```

### 6.2 Decision Flow State Machine

```
+---------------+
| Security Event|
+-------+-------+
        |
        v
+-----------------------+
|  Threat Analysis      |
|  (ML + LLM)           |
+-------+---------------+
        |
        v
+-----------------------+
| Business Impact       |
| Assessment            |
+-------+---------------+
        |
        v
+-----------------------+
| Response Options      |
| Generation            |
+-------+---------------+
        |
        v
+-----------------------+
| LLM Decision Making   |
| (Confidence Scoring)  |
+-------+---------------+
        |
        v
   +----+----+
   | Safety  |
   | Check   |
   +----+----+
        |
        +----+----+-----+-----+-----+-----+---+
        v              v              v          v
  [Conf>=0.95]  [0.85<=Conf<0.95] [0.50<=Conf<0.85] [Conf<0.50]
  [Risk=Low]    [Risk<=Med]       [Any Risk]         [Any Risk]
        |              |              |                 |
        v              v              v                 v
+---------------+ +-------------+ +-------------+ +------------+
| AUTONOMOUS    | | AUTO-APPROVE| | ESCALATE TO | | ESCALATE TO|
| EXECUTION     | | & EXECUTE   | | SOC ANALYST | | SOC MANAGER|
+---------------+ +-------------+ +-------------+ +------------+
        |              |              |                 |
        +--------------+--------------+-----------------+
                              |
                              v
                    +------------------+
                    |  Track Outcome   |
                    |  Update Metrics  |
                    +------------------+
```

### 6.3 Confidence Thresholds and Risk Matrix

| Confidence Score | Low Risk | Medium Risk | High Risk |
|------------------|----------|-------------|-----------|
| **0.95 - 1.00** | Auto Execute | Auto-Approve | Escalate |
| **0.85 - 0.95** | Auto Execute | Auto-Approve | Escalate |
| **0.50 - 0.85** | Auto-Approve | Escalate | Escalate |
| **0.00 - 0.50** | Escalate | Escalate | Escalate |

**Risk Categories**:
- **Low Risk (0.0-0.3)**: Alert generation, log retention, monitoring
- **Medium Risk (0.3-0.7)**: IP blocking, rate limiting, network segmentation
- **High Risk (0.7-1.0)**: System isolation, service shutdown, credential revocation

### 6.4 Security Orchestrator

Location: `ai_engine/security/security_service.py` (500+ LOC)

```
Security Orchestrator Agent
|
+-- Incident Management
|   +-- Active Incidents Tracking (max 50 concurrent)
|   +-- Incident State Machine
|   +-- Automatic Cleanup (24-hour retention)
|   +-- Priority Queue Management
|
+-- Component Coordination
|   +-- Threat Analyzer Integration
|   +-- Decision Engine Integration
|   +-- Response Executor Integration
|   +-- Compliance Reporter Integration
|
+-- Legacy System Awareness
|   +-- Protocol-Specific Handling
|   +-- Maintenance Window Checks
|   +-- Criticality-Based Routing
|   +-- Dependency Tracking
|
+-- Performance Metrics
    +-- Incident Processing Time
    +-- Component Response Times
    +-- Success/Failure Rates
    +-- Prometheus Metrics Export
```

---

## 7. LLM Intelligence Layer

### 7.1 Architecture Philosophy

On-premise first for enterprise security and data sovereignty.

```
Unified LLM Service (On-Premise First)
|
+-- Provider Management (Priority Order)
|   |
|   +-- Tier 1: On-Premise (Default, Air-Gapped Support)
|   |   +-- Ollama (PRIMARY)
|   |   |   +-- Llama 3.2 (8B, 70B)
|   |   |   +-- Mixtral 8x7B
|   |   |   +-- Qwen2.5 (7B, 14B, 32B)
|   |   |   +-- Phi-3 (Microsoft, lightweight)
|   |   |   +-- Custom Fine-tuned Models
|   |   |
|   |   +-- vLLM (High-Performance Inference)
|   |   |   +-- GPU Optimization
|   |   |   +-- Tensor Parallelism
|   |   |   +-- Continuous Batching
|   |   |
|   |   +-- LocalAI (OpenAI-Compatible API)
|   |
|   +-- Tier 2: Cloud Providers (OPTIONAL, Disabled by Default)
|   |   +-- Anthropic Claude (requires API key + internet)
|   |   +-- OpenAI GPT-4 (requires API key + internet)
|   |
|   +-- Intelligent Fallback Chain
|       +-- Ollama -> vLLM -> LocalAI (default)
|       +-- On-Premise Only Mode (air-gapped)
|       +-- Hybrid Mode (cloud fallback if enabled)
|
+-- Enterprise Security Features
|   +-- Air-Gapped Deployment (100% offline, no external network calls)
|   +-- Data Sovereignty (all inference on-premise)
|   +-- Full audit trail of all LLM requests
|   +-- Model Customization (fine-tune on proprietary data)
|
+-- Request Processing
|   +-- System Prompt Injection
|   +-- Context Window Management (up to 128k tokens)
|   +-- Temperature Control (0.0-1.0)
|   +-- Token Limit Enforcement
|   +-- Structured Output Parsing (JSON, YAML)
|
+-- Feature-Domain Prompting
|   +-- Threat Analysis, Decision Making, Impact Assessment
|   +-- Response Generation, Protocol Understanding
|
+-- Monitoring & Control
|   +-- Token Usage, Inference Latency, GPU Utilization
|   +-- Error Rate, Model Performance, Prometheus Metrics
|
+-- Streaming Support (SSE)
```

### 7.2 On-Premise vs Cloud Comparison

| Feature | On-Premise (Ollama) | Cloud LLMs (GPT-4/Claude) |
|---------|---------------------|---------------------------|
| Data Exfiltration Risk | Zero | High (data leaves premises) |
| Internet Required | No (air-gapped) | Yes (always) |
| Data Sovereignty | Full control | Third-party controlled |
| API Keys/Credentials | Not needed | Required |
| Compliance (GDPR, HIPAA) | Full compliance | Requires BAA/DPA |
| Inference Latency | <100ms (local) | 500-2000ms (network) |
| Cost per Request | $0 (hardware only) | $0.001-$0.03 |
| Model Customization | Full fine-tuning | Limited |
| Government/Defense Use | Approved | Often prohibited |
| Supply Chain Risk | Minimal | Third-party dependency |

### 7.3 Recommended On-Premise Models

```
Security Use Case -> Recommended Model
|
+-- High-Speed Threat Analysis
|   +-- Llama 3.2 8B (fast, efficient, 8GB VRAM)
|
+-- Critical Decision Making
|   +-- Llama 3.2 70B (high accuracy, 40GB VRAM)
|
+-- Complex Reasoning
|   +-- Mixtral 8x7B (expert-level, 24GB VRAM)
|
+-- Lightweight/Edge
|   +-- Phi-3 3B (Microsoft, 2GB VRAM)
|
+-- Custom Security Domain
    +-- Fine-tuned Llama 3.2 on proprietary threat data
```

### 7.4 Hardware Recommendations

| Deployment Size | GPU | VRAM | Model | Throughput |
|-----------------|-----|------|-------|------------|
| Small (PoC) | NVIDIA RTX 4090 | 24GB | Llama 3.2 8B | 50 req/sec |
| Medium (Dept) | NVIDIA A100 | 40GB | Llama 3.2 70B | 20 req/sec |
| Large (Enterprise) | 4x NVIDIA A100 | 160GB | Mixtral 8x7B | 100+ req/sec |
| Edge/Branch | NVIDIA T4 | 16GB | Phi-3 3B | 100 req/sec |

### 7.5 Deployment Models

1. **Air-Gapped (Maximum Security)**: Ollama only, no internet connectivity. Pre-downloaded model weights. Ideal for government, defense, critical infrastructure.

2. **On-Premise (Standard Enterprise)**: Ollama + vLLM for high performance. Optional cloud fallback (disabled by default). Ideal for banks, healthcare, enterprises.

3. **Hybrid (Flexible)**: On-premise primary, cloud fallback enabled. Configurable per use case. Ideal for development, testing, non-critical workloads.

### 7.6 RAG Engine

Location: `ai_engine/llm/rag_engine.py` (350+ LOC)

```
RAG Engine
|
+-- Vector Database
|   +-- ChromaDB Integration
|   +-- Sentence Transformer Embeddings
|   +-- Similarity Search (cosine)
|   +-- In-Memory Fallback
|
+-- Document Management
|   +-- Protocol Knowledge Base
|   +-- Security Playbooks
|   +-- Threat Intelligence Reports
|   +-- Compliance Documentation
|
+-- Retrieval Pipeline
|   +-- Query Embedding
|   +-- Top-K Similarity Search (k=5)
|   +-- Relevance Filtering
|   +-- Context Assembly
|
+-- Generation
    +-- Context-Augmented Prompts
    +-- LLM Query with Retrieved Docs
    +-- Source Attribution
    +-- Confidence Scoring
```

---

## 8. Cloud-Native Infrastructure

### 8.1 Service Mesh Integration

```
Service Mesh Components
|
+-- Istio Integration
|   +-- Quantum Certificate Manager (550 LOC)
|   |   +-- Kyber-1024 Key Generation
|   |   +-- Dilithium-5 Signatures
|   |   +-- Certificate Rotation (90-day)
|   |   +-- NIST Level 5 Compliance
|   |
|   +-- Sidecar Injector (450 LOC)
|   +-- mTLS Configurator (400 LOC)
|   +-- Mesh Policy Manager (470 LOC)
|
+-- Envoy Proxy Integration
    +-- xDS Server (650 LOC) - gRPC, ADS, Bidirectional Streaming
    +-- Traffic Encryption (700 LOC) - AES-256-GCM, PFS, <1ms Overhead
    +-- Policy Engine (280 LOC) - Rate Limiting, Circuit Breakers
    +-- Observability (170 LOC) - Prometheus, Jaeger
```

### 8.2 Container Security

```
Container Security Suite
|
+-- Vulnerability Scanner (600 LOC) - Trivy, CVE Detection, 1,000+ Images/Hour
+-- Admission Webhook Server (350 LOC) - Image Signature Verification
+-- Image Signer (550 LOC) - Dilithium-5 Signatures
+-- eBPF Runtime Monitor (800 LOC) - execve, openat, connect tracking, <1% CPU
```

### 8.3 Event Streaming

```
Secure Kafka Producer (700 LOC)
+-- kafka-python KafkaProducer
+-- AES-256-GCM Message Encryption
+-- Compression (snappy)
+-- Retry Logic with Backoff
+-- 100,000+ Messages/Sec
+-- Production Config (acks='all', retries=3)
```

### 8.4 Cloud Platform Integrations

| Cloud | Integration | LOC |
|-------|-------------|-----|
| AWS | Security Hub (boto3, Batch Finding Import, Exponential Backoff) | 600 |
| Azure | Sentinel (Azure SDK, SIEM Data Export) | 550 |
| GCP | Security Command Center (Multi-Region) | 500 |

### 8.5 Kubernetes Architecture

```
Kubernetes Cluster
|
+-- Namespace: qbitel-service-mesh
|   +-- xDS Server Deployment (3 replicas)
|   |   +-- Pod Disruption Budget
|   |   +-- Anti-Affinity Rules
|   |   +-- Resource Limits (250m CPU, 512Mi memory)
|   |   +-- Health Checks (liveness/readiness)
|   +-- Service Mesh Components (Istio, Envoy, Quantum Cert Manager)
|   +-- RBAC Configuration
|
+-- Namespace: qbitel-container-security
|   +-- Admission Webhook (3 replicas)
|   +-- Vulnerability Scanner
|   +-- Image Signer
|
+-- Observability
    +-- Prometheus (metrics)
    +-- Grafana (dashboards)
    +-- Jaeger (tracing)
```

### 8.6 High Availability

- Replication: 3 replicas for all critical services
- Pod Disruption Budgets: Ensure minimum availability during updates
- Anti-Affinity: Distribute pods across nodes
- Health Checks: Liveness and readiness probes
- Auto-Scaling: HPA based on CPU/memory metrics

### 8.7 Security Hardening

- Non-Root Containers: All containers run as UID 1000
- Read-Only Filesystems: Immutable container filesystems
- Security Contexts: Drop all capabilities, no privilege escalation
- Network Policies: Restrict pod-to-pod communication
- RBAC: Least-privilege access control

---

## 9. Complete API Reference

### 9.1 Authentication

**API Key Authentication:**
```http
X-API-Key: your_api_key_here
```

**JWT Authentication:**
```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Token types: Access Token (30 min / 8 hours prod), Refresh Token (7 days), API Key (90 days configurable).

MFA methods: TOTP, SMS, Email, Hardware Token (WebAuthn).

User Roles (RBAC): ADMINISTRATOR (full access), SECURITY_ANALYST (security ops), OPERATOR (operational tasks), VIEWER (read-only), API_USER (API-only).

### 9.2 REST API Endpoints

**Authentication Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/auth/login` | Login, returns access_token + refresh_token |
| POST | `/api/v1/auth/refresh` | Refresh access token |
| POST | `/api/v1/auth/logout` | 204 No Content |
| GET | `/api/v1/auth/me` | Get current user info |

**Protocol Discovery Endpoints:**

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/api/v1/discovery/analyze` | Discover protocol from traffic data | Required |
| POST | `/api/v1/discovery/classify` | Classify a single message | Required |
| POST | `/api/v1/discover` | Discover protocol | Required |
| POST | `/api/v1/fields/detect` | Detect fields | Required |
| POST | `/api/v1/anomalies/detect` | Detect anomalies | Required |

**Protocol Management Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/protocols?limit=10&offset=0&sort=confidence` | List discovered protocols |
| GET | `/api/v1/protocols/{protocol_id}` | Get protocol details |
| PUT | `/api/v1/protocols/{protocol_id}` | Update protocol / retrain |
| DELETE | `/api/v1/protocols/{protocol_id}` | Delete protocol |

**Copilot Endpoints (`/api/v1/copilot/`):**

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/` | Get copilot info | Optional |
| POST | `/query` | Submit query | Required |
| WS | `/ws` | WebSocket streaming | Required |
| GET | `/sessions` | List sessions | Required |
| GET | `/sessions/{id}` | Get session | Required |
| DELETE | `/sessions/{id}` | Delete session | Required |
| GET | `/health` | Copilot health | Optional |

**Security Endpoints (`/api/v1/security/`):**

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/analyze` | Analyze security event | Required |
| POST | `/respond` | Execute response | Elevated |
| GET | `/incidents` | List incidents | Required |
| GET | `/incidents/{id}` | Get incident | Required |

**Health Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Basic health |
| GET | `/healthz` | Kubernetes liveness |
| GET | `/readyz` | Kubernetes readiness |
| GET | `/metrics` | Prometheus metrics |

### 9.3 Request/Response Schemas

**Discovery Analyze Request:**
```json
{
    "traffic_data": ["<base64-encoded-messages>"],
    "options": {
        "max_protocols": 5,
        "confidence_threshold": 0.7,
        "enable_adaptive_learning": true,
        "cache_results": true
    }
}
```

**Discovery Analyze Response:**
```json
{
    "success": true,
    "request_id": "req_12345",
    "processing_time_ms": 245,
    "cache_hit": false,
    "discovered_protocols": [{
        "id": "proto_001",
        "name": "http_variant_1",
        "confidence": 0.94,
        "grammar": { "start_symbol": "HTTP_MESSAGE", "rules": [...] },
        "parser": { "id": "parser_001", "parse_success_rate": 0.98 },
        "validation_rules": [...]
    }],
    "metadata": {
        "model_versions": { "cnn": "v1.2.3", "lstm": "v1.2.1" },
        "processing_stats": {
            "statistical_analysis_ms": 45,
            "grammar_learning_ms": 120,
            "parser_generation_ms": 35
        }
    }
}
```

**Copilot Query Schema:**
```python
class CopilotQuery(BaseModel):
    query: str  # min_length=1, max_length=2000
    query_type: Optional[QueryType] = None  # protocol_analysis, security_assessment, etc.
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    packet_data: Optional[bytes] = None
    enable_learning: bool = True
    preferred_provider: Optional[LLMProvider] = None
```

### 9.4 Error Handling

**Error Codes:**

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid input data or parameters |
| `AUTHENTICATION_ERROR` | 401 | Missing or invalid API key/token |
| `AUTHORIZATION_ERROR` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Requested resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

### 9.5 Rate Limiting

| Endpoint | Rate Limit | Window |
|----------|------------|--------|
| `/api/v1/discovery/analyze` | 10 requests | 1 minute |
| `/api/v1/discovery/classify` | 100 requests | 1 minute |
| `/api/v1/protocols/*` | 50 requests | 1 minute |
| `/health` | 1000 requests | 1 minute |

Rate Limit Headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, `Retry-After` (when limited).

---

## 10. Database Architecture

### 10.1 Storage Technologies

| Technology | Purpose | Port |
|-----------|---------|------|
| PostgreSQL 15+ | Primary relational data | 5432 |
| Redis 7+ | Cache, sessions, rate limiting | 6379 |
| TimescaleDB | Time-series metrics | 5432 |
| ChromaDB | Vector embeddings for RAG | N/A |

### 10.2 Core Database Models

**Users and Authentication:**
```python
class User(Base):
    __tablename__ = "users"
    id: UUID                    # Primary key
    username: str               # Unique, indexed
    email: str                  # Unique, indexed
    password_hash: str          # bcrypt hashed
    role: UserRole              # RBAC role
    permissions: JSONB          # Fine-grained permissions
    mfa_enabled: bool
    mfa_method: MFAMethod
    mfa_secret: EncryptedString
    oauth_provider: Optional[str]
    saml_name_id: Optional[str]
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
    permissions: JSONB

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

### 10.3 Database Manager

```python
class DatabaseManager:
    pool_size: int = 10        # Dev: 10, Prod: 50
    max_overflow: int = 20
    pool_timeout: int = 30     # seconds
    pool_recycle: int = 3600   # Prevent stale connections
```

### 10.4 Migrations

QBITEL uses Alembic for database schema migrations.

| Migration | Purpose |
|-----------|---------|
| `001_initial_auth_schema.py` | Users, API Keys, Sessions |
| `002_add_explainability_tables.py` | Explanation audit trails |
| `003_add_encrypted_fields.py` | Field encryption support |
| `004_add_marketplace_tables.py` | Protocol marketplace |

**Migration Commands:**
```bash
cd ai_engine
alembic upgrade head        # Apply all migrations
alembic current             # Check status
alembic history --verbose   # View history
alembic downgrade -1        # Rollback one version
alembic revision --autogenerate -m "description"  # Auto-generate
```

**Production Migration Procedure:**
1. Backup database: `pg_dump -U qbitel qbitel > backup.sql`
2. Enable maintenance mode: `kubectl scale deployment qbitel --replicas=0`
3. Run migration: `alembic upgrade head`
4. Verify: `alembic current`
5. Disable maintenance mode: `kubectl scale deployment qbitel --replicas=3`

### 10.5 Production Database Tuning

```yaml
max_connections: 200
shared_buffers: 4GB
effective_cache_size: 12GB
work_mem: 64MB
maintenance_work_mem: 1GB
```

Connection Pool:
```yaml
pool_size: 50
max_overflow: 20
pool_timeout: 30
pool_recycle: 3600
```

---

## 11. Environment Configuration

### 11.1 All Environment Variables

**Database Configuration:**

| Variable | Alternative | Required | Description |
|----------|-------------|----------|-------------|
| `QBITEL_AI_DB_PASSWORD` | `DATABASE_PASSWORD` | Yes (Production) | PostgreSQL database password |
| `QBITEL_AI_DB_HOST` | `DATABASE_HOST` | No | Database host (default: localhost) |
| `QBITEL_AI_DB_PORT` | `DATABASE_PORT` | No | Database port (default: 5432) |
| `QBITEL_AI_DB_NAME` | `DATABASE_NAME` | No | Database name (default: qbitel) |
| `QBITEL_AI_DB_USER` | `DATABASE_USER` | No | Database username (default: qbitel) |

**Redis Configuration:**

| Variable | Alternative | Required | Description |
|----------|-------------|----------|-------------|
| `QBITEL_AI_REDIS_PASSWORD` | `REDIS_PASSWORD` | Yes (Production) | Redis authentication password |
| `QBITEL_AI_REDIS_HOST` | `REDIS_HOST` | No | Redis host (default: localhost) |
| `QBITEL_AI_REDIS_PORT` | `REDIS_PORT` | No | Redis port (default: 6379) |

**Security Configuration:**

| Variable | Alternative | Required | Description |
|----------|-------------|----------|-------------|
| `QBITEL_AI_JWT_SECRET` | `JWT_SECRET` | Yes (Production) | JWT token signing secret (32+ chars) |
| `QBITEL_AI_ENCRYPTION_KEY` | `ENCRYPTION_KEY` | Yes (Production) | Data encryption key (32+ chars) |
| `QBITEL_AI_API_KEY` | `API_KEY` | No | API authentication key |

**Environment Control:**

| Variable | Alternative | Default | Description |
|----------|-------------|---------|-------------|
| `QBITEL_AI_ENVIRONMENT` | `ENVIRONMENT` | development | development/staging/production |

**LLM Configuration:**

| Variable | Description |
|----------|-------------|
| `QBITEL_LLM_PROVIDER` | LLM provider: ollama, anthropic, openai |
| `QBITEL_LLM_ENDPOINT` | LLM endpoint URL (default: http://localhost:11434) |
| `QBITEL_LLM_MODEL` | Model name (default: llama3.2:8b) |
| `QBITEL_AIRGAPPED_MODE` | Enable air-gapped mode (true/false) |
| `QBITEL_DISABLE_CLOUD_LLMS` | Disable all cloud LLM providers |

**TLS Configuration:**

| Variable | Description |
|----------|-------------|
| `TLS_ENABLED` | Enable TLS (true for production) |
| `TLS_CERT_FILE` | Path to TLS certificate |
| `TLS_KEY_FILE` | Path to TLS key |

**Monitoring:**

| Variable | Description |
|----------|-------------|
| `SENTRY_DSN` | Sentry error tracking endpoint |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `METRICS_ENABLED` | Enable Prometheus metrics |

**Application:**

| Variable | Description |
|----------|-------------|
| `AI_ENGINE_LOG_LEVEL` | AI Engine log level |
| `AI_ENGINE_WORKERS` | Number of worker processes |
| `AI_ENGINE_MAX_REQUESTS` | Maximum requests per worker |
| `AI_ENGINE_TIMEOUT` | Request timeout in seconds |

### 11.2 Security Requirements

**Password Requirements**: Minimum 16 characters, no weak patterns (password, admin, test, demo, 123456), no sequential characters, no repeated characters, cryptographically secure.

**JWT Secret Requirements**: Minimum 32 characters, high entropy (>50% unique characters).

**Encryption Key Requirements**: Minimum 32 characters, Base64-encoded 256-bit key (for Fernet encryption).

**API Key Requirements**: Minimum 32 characters, must contain both letters and digits, prefix recommended (`qbitel_`).

### 11.3 Configuration Loading Order

1. Load base config from `config/qbitel.yaml`
2. Override with environment-specific config
3. Load environment variables (highest priority)
4. Fetch secrets from secrets manager (Vault, AWS SM)
5. Validate all required secrets present
6. Initialize logging and connections

### 11.4 Production vs Development Mode

**Production** (QBITEL_AI_ENVIRONMENT=production):
- Database password REQUIRED
- Redis password REQUIRED
- JWT secret REQUIRED
- Encryption key REQUIRED
- CORS wildcard (*) FORBIDDEN
- Configuration errors cause immediate failure

**Development** (QBITEL_AI_ENVIRONMENT=development):
- Missing secrets generate warnings (not errors)
- Weak secrets generate warnings
- CORS wildcard allowed
- Redis can run without authentication

---

## 12. Deployment Guide

### 12.1 System Requirements

**Minimum**: CPU 2 cores, RAM 4GB, Storage 20GB, OS Linux/macOS/Windows WSL2

**Production**: CPU 4+ cores, RAM 8GB+, Storage 50GB+ SSD, OS Linux (Ubuntu 20.04+ or RHEL 8+), Optional NVIDIA GPU

### 12.2 Prerequisites

Python 3.10+, Rust 1.70+, Go 1.21+, Node.js 18+, Docker 20.10+, Docker Compose 2.0+, Kubernetes 1.24+ (production), Helm 3.8+

### 12.3 Option 1: Docker Compose (Fastest)

```bash
git clone https://github.com/yazhsab/qbitel-bridge.git
cd qbitel-bridge
docker compose -f docker/docker-compose.yml up -d
# API at http://localhost:8000/docs, UI at http://localhost:3000
```

### 12.4 Option 2: Python AI Engine

```bash
git clone https://github.com/yazhsab/qbitel-bridge.git
cd qbitel-bridge
python -m venv venv && source venv/bin/activate
pip install -e ".[all]"
python -m ai_engine
# Swagger UI at http://localhost:8000/docs
```

### 12.5 Option 3: Helm Deployment (Production)

```bash
helm install qbitel-bridge ./helm/qbitel-bridge \
  --namespace qbitel-bridge \
  --create-namespace \
  --wait
# Includes: AI Engine, Control Plane, xDS Server, Admission Webhook
# Pre-configured: Prometheus, Grafana, OpenTelemetry, Jaeger
```

**Production Deploy:**
```bash
helm install qbitel-bridge ./helm/qbitel-bridge \
  --namespace qbitel-bridge \
  --create-namespace \
  --set xdsServer.replicaCount=5 \
  --set admissionWebhook.replicaCount=5 \
  --set xdsServer.resources.requests.memory=1Gi \
  --set monitoring.enabled=true
```

### 12.6 Option 4: Air-Gapped Deployment

```bash
# Install Ollama for on-premise LLM inference
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2:8b

# Run in fully air-gapped mode
export QBITEL_LLM_PROVIDER=ollama
export QBITEL_AIRGAPPED_MODE=true
export QBITEL_DISABLE_CLOUD_LLMS=true
python -m ai_engine --airgapped
```

Air-gapped features: All LLM inference runs locally, no internet connectivity required, all data stays within infrastructure, no API keys needed, compliant with strictest data residency regulations.

### 12.7 Kubernetes Manual Deployment

```bash
# Create namespaces
kubectl create namespace qbitel-service-mesh
kubectl create namespace qbitel-container-security
kubectl create namespace qbitel-monitoring

# Deploy xDS Server
kubectl apply -f kubernetes/service-mesh/namespace.yaml
kubectl apply -f kubernetes/service-mesh/rbac.yaml
kubectl apply -f kubernetes/service-mesh/xds-server-deployment.yaml

# Deploy Admission Webhook
kubectl apply -f kubernetes/container-security/admission-webhook-deployment.yaml

# Deploy Monitoring
kubectl apply -f kubernetes/monitoring/prometheus-deployment.yaml
kubectl apply -f kubernetes/monitoring/grafana-deployment.yaml
```

### 12.8 Production Resource Limits

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|-----------|----------------|--------------|
| AI Engine | 2 | 4 | 4Gi | 8Gi |
| API Server | 1 | 2 | 2Gi | 4Gi |
| Worker | 2 | 4 | 4Gi | 8Gi |

### 12.9 Auto-Scaling

```bash
kubectl autoscale deployment xds-server \
  -n qbitel-service-mesh \
  --cpu-percent=70 \
  --min=3 \
  --max=10
```

### 12.10 Disaster Recovery

| Scenario | RTO | RPO |
|----------|-----|-----|
| Database failure | 30 min | 6 hours |
| Application failure | 5 min | 0 |
| Complete cluster failure | 2 hours | 6 hours |

---

## 13. Security Architecture

### 13.1 Security Policy

QBITEL Bridge follows responsible disclosure practices. Report vulnerabilities to security@qbitel.com.

- Acknowledgment within 48 hours
- Initial assessment within 5 business days
- Fix for confirmed vulnerabilities within 30 days
- 90-day disclosure policy

### 13.2 Security Best Practices

- Always use TLS 1.3 in production
- Set all secrets via environment variables, never in config files
- Enable audit logging
- Use `verify-full` SSL mode for database connections
- Rotate JWT secrets and encryption keys regularly
- Enable post-quantum cryptography for forward secrecy

### 13.3 Authentication Flow

```
POST /login
    |
    v
Verify Credentials (bcrypt)
    |
    +- Invalid -> 401 (increment failed attempts)
    |
    +- Valid -> Check MFA
                  |
                  +- MFA Enabled -> Verify Code
                  |                    |
                  |                    +- Valid -> Generate Tokens
                  |                    +- Invalid -> 401
                  |
                  +- MFA Disabled -> Generate Tokens
                                        |
                                        v
                                   Return:
                                   - Access Token (30min)
                                   - Refresh Token (7 days)
                                   - User Info
```

### 13.4 Security Headers

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
Content-Security-Policy: default-src 'self'; script-src 'self'
```

### 13.5 Input Validation

Patterns detected and blocked: SQL Injection (UNION SELECT, OR 1=1, DROP TABLE), XSS (<script>, javascript:, onclick=), Command Injection (; | $ ( ) `), Path Traversal (../, %2e%2e, /etc/).

### 13.6 Secret Management

| Provider | SDK | Config Key |
|----------|-----|------------|
| HashiCorp Vault | hvac | `VAULT_ADDR`, `VAULT_TOKEN` |
| AWS Secrets Manager | boto3 | AWS credentials |
| Azure Key Vault | azure-identity | Azure credentials |
| Kubernetes Secrets | Native | In-cluster |

### 13.7 Production Deployment Checklist

**Security Checklist:**
- JWT secret cryptographically strong (32+ chars)
- API keys properly hashed
- MFA enabled for admin accounts
- TLS 1.3 enabled, HSTS configured
- CORS properly restricted
- Rate limiting enabled
- Input validation middleware active
- Database encryption at rest enabled
- PII encrypted with AES-256-GCM
- Post-quantum cryptography enabled (Kyber-1024, Dilithium-5)
- Audit logging enabled
- SIEM integration configured

**Infrastructure Checklist:**
- PostgreSQL 15+ with connection pooling (50 connections)
- Redis with password and persistence
- Kubernetes 1.24+ with namespaces, RBAC, network policies
- Pod security policies, HPA, PDB configured
- Istio with mTLS between services
- Prometheus, Grafana, Jaeger configured
- Alert rules and escalation policies defined

---

## 14. Testing Strategy

### 14.1 Test Structure

```
ai_engine/tests/
  cloud_native/          # Cloud-native feature tests
  security/              # Security and quantum crypto tests
  integration/           # Integration tests
  performance/           # Performance and load tests
  explainability/        # AI explainability tests
  unit/                  # Unit tests for all modules
```

### 14.2 Running Tests

```bash
# All languages
make test

# Python
pytest ai_engine/tests/ -v
pytest ai_engine/tests/ --cov=ai_engine --cov-report=html

# By category
pytest ai_engine/tests/cloud_native/ -v
pytest ai_engine/tests/security/ -v -k quantum
pytest ai_engine/tests/integration/ -v
pytest ai_engine/tests/performance/ -v --benchmark-only

# Rust
cd rust/dataplane && cargo test

# Go
cd go/controlplane && go test ./...
cd go/mgmtapi && go test ./...

# UI
cd ui/console && npm test
```

### 14.3 Coverage Requirements

| Category | Target |
|----------|--------|
| Overall | 80%+ |
| Core components | 90%+ |
| Critical security | 95%+ |
| API endpoints | 85%+ |
| Current coverage | 85% |

### 14.4 Code Quality

```bash
# Python
black ai_engine/                     # Format
flake8 ai_engine/ --max-line-length=120  # Lint
mypy ai_engine/                      # Type check

# Rust
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt
cargo audit

# Go
golangci-lint run ./...
gosec ./...

# UI
npm run lint && npm run format
```

---

## 15. Technology Stack

### 15.1 Complete Technology Inventory

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.10+ (primary), Rust 1.70+, Go 1.21+, TypeScript |
| **AI / ML** | PyTorch 2.2.2, Transformers, Scikit-learn, SHAP, LIME, LangGraph |
| **Post-Quantum Crypto** | liboqs, kyber-py, dilithium-py, oqs-rs (Rust), Falcon, SLH-DSA |
| **LLM** | Ollama (primary), vLLM, Anthropic Claude, OpenAI (optional fallback) |
| **RAG** | ChromaDB, Sentence-Transformers, hybrid search, semantic caching |
| **Backend Framework** | FastAPI 0.104.1, gRPC |
| **Go Framework** | Gin (REST), OPA (policy), Cosign (signing), Zap (logging) |
| **Frontend** | React 18.2, Material-UI 5.14, Vite 5.0, OIDC Auth |
| **Service Mesh** | Istio, Envoy xDS (gRPC), quantum-safe mTLS |
| **Container Security** | Trivy, eBPF/BCC, Kubernetes admission webhooks, cosign |
| **Event Streaming** | Kafka with AES-256-GCM message encryption |
| **Observability** | Prometheus, Grafana, OpenTelemetry, Jaeger, Sentry |
| **Cloud Integrations** | AWS Security Hub, Azure Sentinel, GCP Security Command Center |
| **Databases** | PostgreSQL 15 / TimescaleDB, Redis 7, ChromaDB (vectors) |
| **Policy Engine** | Open Policy Agent (OPA / Rego) |
| **Secrets** | HashiCorp Vault, TPM 2.0 sealing |
| **CI/CD** | GitHub Actions, ArgoCD (GitOps), Helm |
| **SIEM** | Splunk HEC, Elastic REST, QRadar API |

---

## 16. Performance Benchmarks

### 16.1 Key Performance Metrics

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

### 16.2 Throughput

| Component | Metric | Value |
|-----------|--------|-------|
| Kafka Producer | Messages/sec | 100,000+ |
| xDS Server | Proxies | 1,000+ |
| Traffic Encryption | Requests/sec | 10,000+ |
| eBPF Monitor | Containers | 10,000+ |
| Admission Webhook | Pods/sec | 1,000+ |

### 16.3 Latency

| Operation | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| Protocol Discovery | 80ms | 150ms | 200ms |
| Field Detection | 60ms | 120ms | 180ms |
| Anomaly Detection | 100ms | 180ms | 250ms |
| Encryption | <1ms | <1ms | <2ms |
| xDS Config Update | 5ms | 10ms | 15ms |

### 16.4 Resource Usage

| Component | CPU | Memory | Notes |
|-----------|-----|--------|-------|
| AI Engine | 2+ cores | 4GB+ | 8GB+ with GPU |
| xDS Server | 250m | 512Mi | Per replica |
| Admission Webhook | 200m | 256Mi | Per replica |
| eBPF Monitor | <1% | 100Mi | Per node |
| Kafka Producer | 500m | 512Mi | Handles 100K msg/s |

### 16.5 Scalability

**Horizontal Scaling**: AI Engine (stateless, scales with LB), xDS Server (scales with proxies), Admission Webhook (scales with pod creation rate), Kafka (partition-based).

**Data Volume**: Protocol data handles TB/day ingestion, event streaming 100M+ events/day, metrics with time-series retention policies.

---

## 17. Project Structure

### 17.1 Complete Directory Tree

```
qbitel-bridge/
+-- ai_engine/                    # Core AI/ML engine (Python)
|   +-- __main__.py              # Application entry point
|   +-- api/                     # REST & gRPC API layer
|   |   +-- rest.py              # FastAPI application
|   |   +-- server.py            # Server management
|   |   +-- auth.py              # Authentication
|   |   +-- middleware.py        # Request middleware
|   |   +-- rate_limiter.py      # Rate limiting
|   |   +-- input_validation.py  # Input validation
|   |   +-- *_endpoints.py       # Feature-specific endpoints
|   +-- core/                    # Core infrastructure
|   |   +-- engine.py            # Main AI engine orchestrator
|   |   +-- config.py            # Configuration management
|   |   +-- database_manager.py  # Database connection pooling
|   |   +-- exceptions.py        # Custom exceptions
|   |   +-- structured_logging.py # Logging infrastructure
|   +-- agents/                  # Multi-agent orchestration (16+ agents)
|   +-- anomaly/                 # Anomaly detection (VAE, LSTM, Isolation Forest)
|   +-- compliance/              # Compliance automation (9 frameworks)
|   +-- copilot/                 # Protocol intelligence copilot
|   +-- crypto/                  # Post-quantum cryptography
|   +-- discovery/               # Protocol discovery (PCFG, Transformers, BiLSTM-CRF)
|   |   +-- pcfg_inference.py    # Grammar inference
|   |   +-- grammar_learner.py   # Grammar learning
|   |   +-- parser_generator.py  # Parser generation
|   |   +-- protocol_discovery_orchestrator.py
|   +-- detection/               # Field detection
|   |   +-- field_detector.py    # BiLSTM-CRF detector
|   |   +-- anomaly_detector.py  # Anomaly detection
|   +-- legacy/                  # Legacy System Whisperer (~17K LOC)
|   |   +-- service.py           # Main orchestrator
|   |   +-- enhanced_detector.py # VAE + LLM anomaly detection
|   |   +-- predictive_analytics.py # ML failure prediction
|   |   +-- decision_support.py  # LLM recommendations
|   |   +-- knowledge_capture.py # Tribal knowledge
|   +-- llm/                     # LLM gateway (Ollama, RAG, guardrails)
|   |   +-- unified_llm_service.py # Multi-provider LLM
|   |   +-- rag_engine.py        # RAG for context
|   |   +-- legacy_whisperer.py  # Protocol analysis
|   |   +-- translation_studio.py # Protocol translation
|   +-- marketplace/             # Protocol marketplace
|   +-- translation/             # Protocol translation
|   +-- security/                # Zero-touch security engine
|   |   +-- decision_engine.py   # Zero-touch decisions
|   |   +-- security_service.py  # Security orchestrator
|   |   +-- threat_analyzer.py   # Threat analysis
|   |   +-- integrations/        # Cloud security integrations
|   +-- cloud_native/            # Service mesh, container security
|   |   +-- service_mesh/        # Istio/Envoy integration
|   |   +-- container_security/  # Container scanning
|   |   +-- event_streaming/     # Kafka integration
|   |   +-- cloud_integrations/  # AWS/Azure/GCP
|   +-- monitoring/              # Observability
|   +-- explainability/          # AI explainability (SHAP, LIME)
|   +-- threat_intelligence/     # MITRE ATT&CK, STIX/TAXII
|   +-- models/                  # ML model definitions
|   +-- alembic/                 # Database migrations
|   +-- tests/                   # Test suite (160+ files)
|
+-- rust/dataplane/              # Rust high-performance data plane
|   +-- crates/
|       +-- pqc_tls/             # ML-KEM / ML-DSA TLS implementation
|       +-- dpdk_engine/         # DPDK packet processing
|       +-- dpi_engine/          # Deep packet inspection
|       +-- ai_bridge/           # Python-Rust FFI (PyO3)
|       +-- k8s_operator/        # Kubernetes operator
|       +-- io_engine/           # HA bridge with connection pooling
|       +-- adapter_sdk/         # Core L7Adapter trait
|       +-- adapters/
|           +-- iso8583/         # Financial protocol
|           +-- modbus/          # Industrial SCADA
|           +-- tn3270e/         # Mainframe terminal
|           +-- hl7_mllp/        # Healthcare HL7
|
+-- go/                          # Go microservices
|   +-- controlplane/            # gRPC + REST, OPA policies, Vault
|   +-- mgmtapi/                 # Device & certificate management
|   +-- agents/device-agent/     # Edge agent with TPM 2.0
|
+-- ui/console/                  # React Admin Console (TypeScript)
|   +-- src/components/          # React components
|   +-- src/api/                 # API clients
|   +-- src/auth/                # OIDC service
|   +-- src/routes/              # RBAC routing
|   +-- src/theme/               # 5 theme variants
|
+-- helm/qbitel-bridge/          # Production Helm chart
+-- ops/                         # Grafana dashboards, Prometheus rules, runbooks
+-- kubernetes/                  # Kubernetes manifests
+-- docker/                      # Docker configurations
+-- config/                      # Configuration files
|   +-- qbitel.yaml             # Default config
|   +-- qbitel.production.yaml  # Production config
|   +-- environments/            # Environment-specific
+-- diagrams/                    # SVG architecture diagrams
+-- samples/cobol/               # 500+ COBOL sample programs
+-- demos/                       # End-to-end demo scenarios
+-- datasets/                    # Training/test data
+-- security/                    # Security validation
+-- integration/                 # Integration layer
+-- scripts/                     # Utility scripts
+-- tests/                       # Conformance, fuzz, perf tests
+-- docs/                        # 73+ documentation files
|   +-- products/                # 10 product guides
|   +-- brochures/               # 9 industry brochures
+-- .github/workflows/           # CI/CD pipelines
+-- Makefile                     # Unified build system
+-- requirements.txt             # Python production dependencies
+-- requirements-dev.txt         # Python development dependencies
```

---

## 18. Implementation Plan

### 18.1 Phase Overview

The implementation plan spans 12 weeks across 4 phases:

| Week | Phase | Goal | Success Criteria |
|------|-------|------|------------------|
| 1-2 | Phase 1: Security Hardening | Address critical security gaps | Zero critical vulnerabilities |
| 3-5 | Phase 2: Architecture Simplification | Reduce complexity | 30% reduction in complexity |
| 6-8 | Phase 3: Focus & Prioritization | Remove non-essential features | Core features 100% tested |
| 9-12 | Phase 4: Production Hardening | Enterprise reliability | 99.9% uptime target |

### 18.2 Phase 1: Security Hardening (Weeks 1-2)

**1.1 API Rate Limiting** (CRITICAL, 2 days): Install slowapi, Redis backend for distributed limiting, Prometheus metrics for rate limit hits.

**1.2 Input Size Validation** (CRITICAL, 1 day): MAX_PROTOCOL_MESSAGE_SIZE=10MB, MAX_FIELD_DETECTION_SIZE=1MB, MAX_BATCH_SIZE=100, MAX_COPILOT_QUERY_LENGTH=10000 chars.

**1.3 Standardized Error Handling** (HIGH, 3 days): Error code ranges -- Discovery (1000-1999), Detection (2000-2999), LLM (3000-3999), Authentication (4000-4999), Validation (5000-5999), System (9000-9999).

### 18.3 Phase 2: Architecture Simplification (Weeks 3-5)

**2.1 Remove Unused LLM Providers**: Simplify from 5 to 2 providers (Ollama + Anthropic).

**2.2 Split Database Models**: Restructure 500+ line monolithic file into auth/, providers/, audit/, protocol/ modules.

**2.3 Refactor Legacy Whisperer**: Split 17K LOC monolith into 4 services (Analyzer, Mapper, Generator, Orchestrator).

### 18.4 Phase 3: Focus and Prioritization (Weeks 6-8)

**3.1 Feature Flags**: ENABLE_AVIATION_DOMAIN, ENABLE_AUTOMOTIVE_DOMAIN, ENABLE_HEALTHCARE_DOMAIN, ENABLE_MARKETPLACE.

**3.2 Simplify Discovery Pipeline**: Merge 7 phases to 5 (ANALYSIS, CLASSIFICATION, LEARNING, VALIDATION, COMPLETION).

### 18.5 Phase 4: Production Hardening (Weeks 9-12)

**4.1 Circuit Breakers**: LLM (3 failures/60s), Database (5 failures/30s), Redis (3 failures/15s), External APIs (5 failures/120s).

**4.2 Health Checks**: `/health` (liveness), `/ready` (readiness), `/startup` (boot complete), `/health/detailed` (all components).

**4.3 Prometheus Metrics**: Request latency/count, discovery duration/confidence, LLM tokens/latency, active connections, model memory.

**4.4 Graceful Shutdown**: Drain active requests with 30-second timeout.

### 18.6 Design Patterns Used

| Pattern | Where Used |
|---------|------------|
| Server Manager | `api/server.py` - Graceful shutdown |
| Dependency Injection | FastAPI `Depends()` |
| Circuit Breaker | `core/database_circuit_breaker.py` |
| Repository | Database access layer |
| Strategy | LLM provider fallback |
| Observer | Event tracking/metrics |
| Factory | Model/service creation |

### 18.7 Error Handling Hierarchy

```python
QbitelAIException (base)
+-- ConfigurationException
+-- ModelException
|   +-- ModelLoadException
|   +-- ModelInferenceException
+-- ProtocolException
|   +-- DiscoveryException
+-- SecurityException
    +-- AuthenticationException
```

Recovery Strategies: RETRY, FALLBACK, CIRCUIT_BREAK, DEGRADE, FAIL_FAST.

---

## 19. Contributing and Changelog

### 19.1 Contributing Guidelines

Contributions are welcome under the Apache License 2.0. Fork the repository, create a feature branch from `main`, follow coding standards, add tests, and submit a PR.

**Commit Convention:**
```
feat: Add new feature
fix: Bug fix
docs: Documentation changes
test: Add or update tests
refactor: Code refactoring
perf: Performance improvements
chore: Build/tooling changes
```

**Coding Standards:**
- Python: PEP 8, Black formatting, flake8 linting, type hints, Google-style docstrings, 120 char max
- Rust: cargo fmt, cargo clippy --all-targets -- -D warnings, doc comments
- Go: gofmt, golangci-lint, gosec
- TypeScript/React: ESLint, strict mode, functional components with hooks

### 19.2 Changelog

**[Unreleased]**:
- Initial public release of QBITEL Bridge
- AI-powered protocol discovery for legacy mainframe systems
- Post-quantum cryptography support (ML-KEM/Kyber, ML-DSA/Dilithium)
- Legacy System Whisperer for COBOL analysis
- Protocol Signature Database with 33+ built-in signatures
- PQC Protocol Bridge for quantum-safe data translation
- vLLM provider integration for on-premise LLM inference
- Helm chart for Kubernetes deployment
- Docker support with multi-stage builds

**[1.0.0] - 2025-02-07**:
- Protocol Discovery Engine: TN3270, IBM MQ Series, ISO 8583, Modbus TCP
- Legacy System Whisperer: COBOL source code analysis, copybook parsing, business logic extraction
- Post-Quantum Cryptography: ML-KEM (Kyber), ML-DSA (Dilithium), hybrid mode
- PQC Protocol Bridge: Real-time protocol translation, COBOL to JSON
- Cloud-Native Infrastructure: Kubernetes, Istio, Envoy, Trivy, eBPF
- Enterprise Features: Multi-provider LLM, Prometheus, audit logging, compliance automation
- Protocol Categories: Legacy Mainframe (TN3270, IBM MQ, CICS, COBOL), Financial (ISO 8583, SWIFT, FIX), Industrial (Modbus TCP, DNP3, OPC-UA), Healthcare (HL7, DICOM), Enterprise (SAP RFC, CORBA/IIOP)

### 19.3 Support

- GitHub Issues: https://github.com/yazhsab/qbitel-bridge/issues
- Documentation: /docs/ directory
- Enterprise Support: enterprise@qbitel.com
- Security Issues: security@qbitel.com
- API Support: api-support@qbitel.com
- Developer Support: developers@qbitel.com

---

**End of Technical Reference**

*This document consolidates content from README.md, architecture.md, KNOWLEDGE_BASE.md, DEVELOPMENT.md, DEPLOYMENT.md, QUICKSTART.md, API.md, IMPLEMENTATION_PLAN.md, LOCAL_DEPLOYMENT_GUIDE.md, DATABASE_MIGRATIONS.md, ENVIRONMENT_VARIABLE_CONFIGURATION.md, PRODUCTION_DEPLOYMENT_CHECKLIST.md, SECURITY.md, AGENTS.md, PQC_IMPLEMENTATION_PLAN.md, CONTRIBUTING.md, and CHANGELOG.md.*
