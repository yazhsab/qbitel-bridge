# QBITEL Bridge — BPO Deployment & Delivery Checklist
**Version 1.0 | February 2026 | Confidential**

> **Purpose:** End-to-end delivery checklist for deploying QBITEL Bridge in a BPO/Call Center environment. Use this document to track every phase from pre-sales discovery through post-go-live validation.

---

## DEPLOYMENT OVERVIEW

| Phase | Duration | Owner | Status |
|-------|----------|-------|--------|
| Phase 0: Pre-Engagement & Scoping | 1–2 days | Sales + Presales | ☐ |
| Phase 1: Environment Discovery | 2–4 hours | QBITEL AI Engine | ☐ |
| Phase 2: Infrastructure Readiness | 1–2 days | Customer IT + QBITEL | ☐ |
| Phase 3: Security & Compliance Setup | 4–6 hours | QBITEL + Compliance Team | ☐ |
| Phase 4: Protocol Protection Activation | 2–4 hours | QBITEL Engine | ☐ |
| Phase 5: Integration Deployment | 4–8 hours | QBITEL + Customer IT | ☐ |
| Phase 6: Automation Recipe Execution | 2–3 hours | Zero-Touch Orchestrator | ☐ |
| Phase 7: Monitoring & Alerting Setup | 1–2 hours | QBITEL + SOC Team | ☐ |
| Phase 8: Compliance Validation | 2–4 hours | Compliance Officer | ☐ |
| Phase 9: User Acceptance Testing | 1 day | QA + Operations | ☐ |
| Phase 10: Go-Live & Handover | 2–4 hours | All Teams | ☐ |
| Phase 11: Post-Go-Live Monitoring | 30 days | QBITEL CSM | ☐ |

**Total Deployment Window: 4–6 hours (zero downtime)**
**Full Validation & Hardening: 5–10 business days**

---

## PHASE 0: PRE-ENGAGEMENT & SCOPING

### 0.1 — Customer Information Gathering
- [ ] BPO type identified: `☐ Financial Services` `☐ Healthcare` `☐ General Customer Service` `☐ Remote Workforce` `☐ Multi-Tenant`
- [ ] Total concurrent agent seat count: ___________
- [ ] Number of sites/locations: ___________
- [ ] Number of enterprise clients (tenants): ___________
- [ ] Remote agent percentage: ___________ %
- [ ] Offshore agent locations: ___________
- [ ] Annual carrier invoice reviewed for toll fraud baseline: ☐ Yes ☐ Not yet

### 0.2 — Existing Infrastructure Inventory
- [ ] PBX / ACD platform(s) identified:
  - `☐ Avaya Aura / CM` `☐ Cisco CUCM` `☐ Genesys Cloud` `☐ Asterisk / FreePBX` `☐ Mitel` `☐ BroadWorks` `☐ RingCentral` `☐ Other: ________`
- [ ] CRM platform(s):
  - `☐ Salesforce` `☐ Zendesk` `☐ ServiceNow` `☐ Dynamics 365` `☐ Freshdesk` `☐ HubSpot` `☐ Other: ________`
- [ ] Workforce Management (WFM) platform(s):
  - `☐ NICE WFM` `☐ Verint` `☐ Aspect` `☐ Calabrio` `☐ Genesys WFM` `☐ Other: ________`
- [ ] Call recording platform(s):
  - `☐ NICE Engage` `☐ NICE CXone` `☐ Verint WEM` `☐ Avaya WFO` `☐ Other: ________`
- [ ] Mainframe / terminal access:
  - `☐ IBM TN3270e` `☐ IBM TN5250` `☐ None` `☐ Other: ________`
- [ ] IVR / self-service:
  - `☐ Nuance` `☐ Cisco CVP` `☐ Avaya IR` `☐ Genesys Dialog Engine` `☐ Amazon Connect` `☐ Other: ________`

### 0.3 — Compliance Requirements
- [ ] Compliance frameworks applicable:
  - `☐ PCI-DSS 4.0` `☐ TCPA` `☐ HIPAA` `☐ HITECH` `☐ SOC 2 Type II` `☐ GDPR` `☐ SOX` `☐ GLBA` `☐ FCA/MiFID II`
- [ ] Upcoming audit dates noted: ___________
- [ ] Current PCI-DSS QSA identified: ___________
- [ ] DPA / BAA requirements confirmed with legal: ☐ Yes ☐ Pending
- [ ] Data residency requirements identified: ___________
- [ ] Recording retention periods confirmed per client: ___________

### 0.4 — Network & Access Pre-Requisites
- [ ] SPAN port / network tap access confirmed with IT: ☐ Yes ☐ Pending
- [ ] Firewall rules reviewed — required ports listed:
  - SIP: 5060 (UDP/TCP), 5061 (TLS), 5062 (PQC-TLS)
  - RTP: 16384–32767 (UDP)
  - Management: 443 (HTTPS), 8443 (API)
  - HSM: 1792 (if hardware HSM)
- [ ] Network diagram obtained: ☐ Yes ☐ Pending
- [ ] VLANs / segmentation documented: ___________
- [ ] Change management window confirmed: ___________

### 0.5 — Stakeholder Sign-Off
- [ ] CTO / IT Director: ___________________ Date: ___________
- [ ] CISO / Security Lead: ___________________ Date: ___________
- [ ] Chief Compliance Officer: ___________________ Date: ___________
- [ ] Contact Center Director: ___________________ Date: ___________
- [ ] QBITEL Delivery Lead: ___________________ Date: ___________

---

## PHASE 1: AI PROTOCOL DISCOVERY

> **Duration:** 2–4 hours | **Method:** Passive — zero traffic impact

### 1.1 — Network Tap Deployment
- [ ] SPAN port configured on voice/data switch
- [ ] Passive network tap placed at SIP trunk boundary
- [ ] QBITEL Sensor VM deployed and powered on
- [ ] Sensor connectivity to QBITEL Engine confirmed
- [ ] Tap placement verified (read-only, no traffic interference)
- [ ] Baseline call volume confirmed during tap window: ___________

### 1.2 — Discovery Phase Execution
- [ ] Discovery phase started in QBITEL Management Console
- [ ] Statistical analysis phase completed (5–10 seconds)
- [ ] ML classification phase completed (10–20 seconds) — 89%+ accuracy target
- [ ] Grammar learning phase completed (1–2 minutes)
- [ ] Parser generation phase completed (30–60 seconds)
- [ ] Discovery report reviewed and approved

### 1.3 — Protocol Discovery Results (tick all found)
**Voice & Signaling:**
- [ ] SIP (port 5060) discovered — session count: ___________
- [ ] SIP-TLS (port 5061) discovered
- [ ] RTP streams discovered — concurrent count: ___________
- [ ] SRTP streams discovered
- [ ] DTMF (RFC 2833 / RFC 4733) relay identified
- [ ] SS7/ISUP legacy signaling discovered
- [ ] MGCP discovered
- [ ] H.323 discovered

**Terminal & Data:**
- [ ] TN3270e sessions discovered — agent count: ___________
- [ ] TN5250 sessions discovered
- [ ] CTI (TSAPI / CSTA / Finesse) events discovered
- [ ] VXML / MRCP / CCXML IVR traffic discovered

**Unexpected / Legacy (flag for review):**
- [ ] Undocumented protocols flagged by AI: ___________
- [ ] Unencrypted PII-carrying streams flagged: ___________
- [ ] Unknown custom protocols identified for manual review: ___________

### 1.4 — Discovery Sign-Off
- [ ] Protocol map reviewed by customer IT team
- [ ] Unexpected discoveries documented and risk-assessed
- [ ] Customer IT Lead approval: ___________________ Date: ___________
- [ ] QBITEL Delivery Lead approval: ___________________ Date: ___________

---

## PHASE 2: INFRASTRUCTURE READINESS

### 2.1 — QBITEL Engine Server Requirements
| Component | Minimum | Recommended | Confirmed |
|-----------|---------|-------------|-----------|
| API Servers | 3x (4 CPU, 16GB RAM) | 3x (8 CPU, 32GB RAM) | ☐ |
| PostgreSQL Primary | 8 CPU, 32GB RAM, 500GB SSD | 16 CPU, 64GB RAM, 1TB NVMe | ☐ |
| PostgreSQL Replica | 8 CPU, 32GB RAM, 500GB SSD | Same as primary | ☐ |
| Redis Cluster | 3x (2 CPU, 8GB RAM) | 3x (4 CPU, 16GB RAM) | ☐ |
| Load Balancer | 2x HA (2 CPU, 4GB RAM) | 2x HA (4 CPU, 8GB RAM) | ☐ |
| Monitoring Stack | 1x (4 CPU, 8GB RAM) | 1x (8 CPU, 16GB RAM) | ☐ |
| QBITEL Sensor VM | 2 CPU, 8GB RAM per site | 4 CPU, 16GB RAM per site | ☐ |

- [ ] Servers provisioned and accessible
- [ ] OS: Linux (Ubuntu 22.04 LTS or RHEL 8+) confirmed
- [ ] Python 3.10+ installed
- [ ] Docker 20.10+ / Kubernetes 1.25+ available
- [ ] Network connectivity between all components tested
- [ ] NTP synchronized across all servers
- [ ] DNS resolution working for all internal hostnames
- [ ] Storage throughput tested (minimum 500 MB/s for recording encryption)

### 2.2 — HSM (Hardware Security Module) — Payment & Financial BPOs
- [ ] HSM provisioned (or software HSM approved for non-financial BPOs)
- [ ] HSM initialized with FIPS 140-3 Level 3 configuration
- [ ] ML-KEM and ML-DSA algorithms loaded to HSM
- [ ] HSM network connectivity to QBITEL Engine verified
- [ ] HSM backup key escrow configured
- [ ] HSM admin credentials securely stored in vault: ___________

### 2.3 — Database Setup
- [ ] PostgreSQL 15+ installed and initialized
- [ ] PostgreSQL primary–replica replication confirmed
- [ ] Connection pooling configured (pool_size: 50, timeout: 30s)
- [ ] Database encryption at rest (AES-256-GCM) enabled
- [ ] Automated backup schedule configured (every 6 hours)
- [ ] Point-in-time recovery tested
- [ ] Backup restoration tested (RTO target: 30 minutes)
- [ ] QBITEL schema migration run: `alembic upgrade head`

### 2.4 — Network Security
- [ ] TLS 1.3 certificates installed (validity: 30+ days minimum)
- [ ] Certificate chain complete and validated
- [ ] Firewall rules applied per Phase 0.4 port list
- [ ] HSTS headers configured
- [ ] CORS restricted to approved domains only
- [ ] API rate limiting configured
- [ ] Management network segment isolated from agent VLAN
- [ ] QBITEL Engine not directly reachable from internet (behind load balancer)

### 2.5 — LLM / AI Engine Setup
- [ ] Ollama installed and running on QBITEL Engine server
- [ ] LLM model downloaded and tested:
  - `☐ Llama 3.2 8B` (standard deployments)
  - `☐ Llama 3.2 70B` (enterprise, complex threat analysis)
  - `☐ Mixtral 8x7B` (multilingual BPOs)
  - `☐ Qwen 2.5` (APAC deployments)
- [ ] LLM response time tested (<5 seconds for threat narrative generation)
- [ ] Air-gapped mode confirmed if required: ☐ Yes ☐ Not required
- [ ] Cloud LLM (Claude API) configured if opted in: ☐ Yes ☐ Not opted in

---

## PHASE 3: SECURITY POLICY & COMPLIANCE CONFIGURATION

### 3.1 — Security Policy Selection
Select the pre-configured policy matching your BPO type:

**☐ Financial Services BPO Policy** (PCI-DSS + SOX + GLBA)
- Security level: CRITICAL (256-bit keys)
- Compliance: PCI-DSS 4.0, SOX, GLBA, GDPR, SOC 2, NIST PQC
- Data classification: RESTRICTED
- Audit retention: 10 years
- DTMF masking: ENABLED (CLAMP mode)
- Recording: ML-KEM-1024 encryption

**☐ Healthcare BPO Policy** (HIPAA + HITECH)
- Security level: CRITICAL (256-bit keys)
- Compliance: HIPAA, HITECH, GDPR, SOC 2, NIST PQC
- Data classification: RESTRICTED
- Audit retention: 6 years
- PHI encryption: ENABLED
- Voice biometrics: ENABLED

**☐ General Customer Service Policy** (GDPR + SOC 2)
- Security level: ENHANCED (192-bit keys)
- Compliance: GDPR, SOC 2, NIST PQC
- Data classification: CONFIDENTIAL
- Retention: 3 years
- Risk tolerance: MEDIUM

**☐ Remote Workforce Policy** (Enhanced endpoint security)
- Security level: ENHANCED
- Agent security: MFA (TOTP + biometric)
- Endpoint: Windows 10+ / macOS 12+, disk encryption, antivirus
- Network: WPA3 home network required
- Screen watermarking: ENABLED

### 3.2 — Per-Tenant Policy Configuration (Multi-Tenant BPOs)
For each enterprise client served:

| Client Name | Policy Type | Compliance Frameworks | Encryption Keys | Status |
|-------------|------------|----------------------|-----------------|--------|
| Client 1: _______ | _______ | _______ | Isolated ☐ | ☐ |
| Client 2: _______ | _______ | _______ | Isolated ☐ | ☐ |
| Client 3: _______ | _______ | _______ | Isolated ☐ | ☐ |
| Client 4: _______ | _______ | _______ | Isolated ☐ | ☐ |

- [ ] Per-tenant cryptographic key isolation confirmed
- [ ] Per-tenant compliance policy applied independently
- [ ] Per-tenant audit trail segregation verified
- [ ] Cross-tenant data access tested and confirmed blocked

### 3.3 — PQC Key Generation
- [ ] ML-KEM-512 key pairs generated for voice signaling and CTI (latency: <50ms target)
- [ ] ML-KEM-768 key pairs generated for agent desktop, terminal, remote access (latency: <200ms target)
- [ ] ML-KEM-1024 key pairs generated for payment processing and call recording (latency: <100ms target)
- [ ] ML-DSA-65 signature keys generated
- [ ] ML-DSA-87 signature keys generated for payment and recording
- [ ] Falcon-512 compact signature keys generated for bandwidth-constrained paths
- [ ] All keys stored in HSM (or secure software vault)
- [ ] Key rotation schedule configured: ___________
- [ ] Key backup and escrow procedure documented: ___________

### 3.4 — Authentication & Access Control
- [ ] JWT secret configured (minimum 32 characters, cryptographically random)
- [ ] API key hashing enabled (bcrypt / Argon2)
- [ ] MFA enforced for all QBITEL Management Console admin accounts
- [ ] Role-based access control (RBAC) configured:
  - Admin: QBITEL Delivery Lead
  - Security Analyst: SOC team members
  - Compliance Viewer: CCO / auditor accounts
  - Read-Only: Operations / reporting users
- [ ] Agent desktop policy user accounts configured
- [ ] Service account credentials rotated from defaults
- [ ] SSH key-only access configured (password auth disabled)

---

## PHASE 4: PROTOCOL PROTECTION ACTIVATION

> **Activation is per-trunk and staged — voice calls continue uninterrupted**

### 4.1 — SIP / Voice Signaling Protection
- [ ] First test trunk selected for initial activation: ___________
- [ ] SIP-PQC-TLS activated on port 5062 (hybrid mode with X25519)
- [ ] SIP signaling encryption latency measured: _____ ms (target: <50ms p95)
- [ ] Call setup test (10 calls): ☐ All succeeded ☐ Issues noted: ___________
- [ ] SIP signaling activated on all remaining trunks (one-by-one):

| Trunk ID | Carrier | Direction | PQC-TLS Activated | Tested | Status |
|----------|---------|-----------|-------------------|--------|--------|
| ________ | _______ | In/Out/Both | ☐ | ☐ | ☐ OK |
| ________ | _______ | In/Out/Both | ☐ | ☐ | ☐ OK |
| ________ | _______ | In/Out/Both | ☐ | ☐ | ☐ OK |
| ________ | _______ | In/Out/Both | ☐ | ☐ | ☐ OK |

### 4.2 — RTP / Media Encryption
- [ ] SRTP-PQC activated (ML-KEM-512 + AES-256-GCM)
- [ ] RTP media encryption latency measured: _____ ms (target: <2ms overhead)
- [ ] MOS score measured before activation: ___________
- [ ] MOS score measured after activation: ___________ (must be equal or better)
- [ ] Packet loss and jitter measured: ___________ (no regression permitted)
- [ ] Codec compatibility confirmed: G.711 ☐ G.729 ☐ G.722 ☐ Opus ☐

### 4.3 — DTMF Masking Activation
- [ ] DTMF masking mode selected: `☐ CLAMP` `☐ FLAT_TONE` `☐ SILENCE` `☐ REPLACE`
- [ ] DTMF masking activated on all IVR payment paths
- [ ] Masking latency measured: _____ ms (target: <5ms)
- [ ] Test payment call with DTMF entry verified:
  - [ ] Agent headset: Card digits not audible ☐
  - [ ] Call recording: Card digits not present ☐
  - [ ] Payment gateway: DTMF digits received correctly ☐
- [ ] Agent screen masking activated (last 4 digits display only)
- [ ] PAN detection engine activated and tested with Luhn-valid test card numbers

### 4.4 — TN3270e / TN5250 Terminal Protection
- [ ] TN3270e PQC tunnel wrapper activated
- [ ] TN5250 PQC tunnel wrapper activated (if applicable)
- [ ] Terminal session latency measured: _____ ms (target: <300ms)
- [ ] Mainframe screen field detection configured for PII masking
- [ ] Agent terminal sessions — before/after screen test: ☐ No visible change for agents
- [ ] Session audit logging confirmed active

### 4.5 — Call Recording Encryption
- [ ] Recording encryption activated: ML-KEM-1024 + AES-256-GCM
- [ ] Recording encryption throughput tested: _____ concurrent streams (target: 10,000+)
- [ ] Recording pause/resume on payment detection confirmed:
  - [ ] Auto-pause triggered: ☐ Yes
  - [ ] Pause event logged with timestamp, agent_id, call_id: ☐ Yes
  - [ ] Auto-resume after payment timeout: ☐ Yes
  - [ ] Resume event logged: ☐ Yes
- [ ] Recording access test — encrypted file unreadable without QBITEL key: ☐ Confirmed
- [ ] Recording integrity verification (tamper detection) tested: ☐ Confirmed

---

## PHASE 5: INTEGRATION DEPLOYMENT

### 5.1 — PBX / ACD Integration
For each PBX platform in use:

**Avaya Aura / CM:**
- [ ] TSAPI connection configured with PQC tunnel
- [ ] DMCC link tested (agent state events flowing)
- [ ] PBX admin credentials secured in vault
- [ ] Call flow integrity test (100 calls): ☐ Pass

**Cisco CUCM:**
- [ ] CTI-OS integration configured with PQC tunnel
- [ ] Cisco Finesse REST API connection tested
- [ ] CTI manager connectivity confirmed
- [ ] Call flow integrity test (100 calls): ☐ Pass

**Genesys Cloud:**
- [ ] REST API integration with PQC-TLS configured
- [ ] OAuth token secured and rotation scheduled
- [ ] Event subscription activated
- [ ] Call flow integrity test (100 calls): ☐ Pass

**Asterisk / FreePBX:**
- [ ] AMI/ARI integration with PQC tunnel configured
- [ ] Dialplan compatibility confirmed
- [ ] Call flow integrity test (100 calls): ☐ Pass

### 5.2 — CRM Integration
- [ ] Salesforce REST API connected — PII masking confirmed
- [ ] Zendesk REST API connected — data classification active
- [ ] ServiceNow REST API connected — workflow integration tested
- [ ] Dynamics 365 connected (if applicable)
- [ ] CRM data access audit logging confirmed
- [ ] Rate limiting on bulk API queries enabled (prevent data exfiltration)
- [ ] CRM integration test: customer lookup from agent desktop ☐ Pass

### 5.3 — WFM Integration
- [ ] NICE WFM bridge configured and tested
- [ ] Verint WEM bridge configured and tested
- [ ] Aspect / Calabrio bridge configured (if applicable)
- [ ] Schedule enforcement integration confirmed
- [ ] Agent adherence real-time feed active

### 5.4 — SIEM / SOC Integration
- [ ] CEF/syslog output configured to: ___________
  - `☐ Splunk` `☐ IBM QRadar` `☐ Microsoft Sentinel` `☐ ArcSight` `☐ Elastic SIEM`
- [ ] QBITEL event fields confirmed in SIEM: call_id, agent_id, tenant_id, trunk_id, fraud_type
- [ ] Test events received and parsed in SIEM: ☐ Pass
- [ ] SOAR webhook configured to: ___________
  - `☐ Palo Alto XSOAR` `☐ Splunk SOAR` `☐ ServiceNow SecOps`
- [ ] Test SOAR trigger executed: ☐ Pass

---

## PHASE 6: AUTOMATION RECIPE EXECUTION

> **Run via QBITEL Zero-Touch Orchestrator — each recipe must complete with 100% step success**

### 6.1 — Recipe 1: PCI-DSS Voice Compliance
- [ ] Recipe initiated in dry-run mode first: ☐ Pass
- [ ] Step 1: DTMF masking configured on all IVR paths (mode: CLAMP) — ☐ Completed
- [ ] Step 2: Recording pause/resume enabled (auto-pause on payment) — ☐ Completed
- [ ] Step 3: PAN detection engine deployed (credit/debit card detection) — ☐ Completed
- [ ] Step 4: Agent screen masking configured (last 4 digits only) — ☐ Completed
- [ ] Step 5: PCI scope management activated (auto-descope) — ☐ Completed
- [ ] Step 6: PCI-DSS SAQ-D compliance report generated — ☐ Completed
- [ ] All 6 steps: ☐ 100% success | Failed steps: ___________

### 6.2 — Recipe 2: Toll Fraud Prevention
- [ ] Step 1: Premium rate number database loaded (200+ countries) — ☐ Completed
- [ ] Step 2: IRSF detection rules configured (block + alert) — ☐ Completed
- [ ] Step 3: Velocity/volume alerting set (max 20 intl/hour, 5 calls/min) — ☐ Completed
- [ ] Step 4: Automatic call blocking enabled (premium numbers, spoofed CLI) — ☐ Completed
- [ ] Step 5: Off-hours monitoring configured (block intl after business hours) — ☐ Completed
- [ ] Step 6: Toll fraud prevention validated with test scenarios — ☐ Completed
- [ ] All 6 steps: ☐ 100% success | Failed steps: ___________
- [ ] Test: Simulated IRSF call to premium prefix — ☐ Blocked within 3 calls

### 6.3 — Recipe 3: Remote Workforce Security (if applicable)
- [ ] Step 1: PQC key pairs generated for remote agents (ML-KEM-768, stored in HSM) — ☐ Completed
- [ ] Step 2: VPN-less PQC tunnels configured (pqc-wireguard protocol) — ☐ Completed
- [ ] Step 3: Device posture checking deployed (OS, antivirus, disk encryption) — ☐ Completed
- [ ] Step 4: Geo-fencing rules configured for approved countries — ☐ Completed
- [ ] Step 5: Screen watermarking deployed (forensic: agent ID + timestamp) — ☐ Completed
- [ ] Step 6: Network risk assessment configured (WiFi, router security) — ☐ Completed
- [ ] All 6 steps: ☐ 100% success | Failed steps: ___________
- [ ] Test: Remote agent login from approved country — ☐ Pass
- [ ] Test: Remote agent login from blocked country — ☐ Blocked

### 6.4 — Recipe 4: Quantum-Safe Voice Upgrade
- [ ] Step 1: ML-KEM-768 key pairs generated (10 pairs for voice infrastructure) — ☐ Completed
- [ ] Step 2: SIP-PQC-TLS configured on port 5062 (hybrid mode with X25519) — ☐ Completed
- [ ] Step 3: SRTP-PQC enabled for media (AES-256-GCM + key rotation) — ☐ Completed
- [ ] Step 4: Recording keys wrapped with PQC (ML-KEM-1024) — ☐ Completed
- [ ] Step 5: HSM configuration updated (ML-KEM and ML-DSA enabled) — ☐ Completed
- [ ] Step 6: End-to-end quantum-safe voice validation — ☐ Completed
- [ ] All 6 steps: ☐ 100% success | Failed steps: ___________

### 6.5 — Recipe 5: Full Compliance Suite
- [ ] Step 1: PCI-DSS 4.0 voice controls deployed (Req 3.3, 3.4, 3.5, 8.3, 10.2) — ☐ Completed
- [ ] Step 2: TCPA consent management configured (tracking, opt-out, DNC) — ☐ Completed
- [ ] Step 3: SOC 2 Type II monitoring deployed — ☐ Completed
- [ ] Step 4: GDPR recording consent controls activated — ☐ Completed
- [ ] Step 5: Automated compliance reporting deployed (monthly reports) — ☐ Completed
- [ ] Step 6 (Healthcare only): HIPAA PHI protection activated — ☐ Completed / ☐ N/A
- [ ] All steps: ☐ 100% success | Failed steps: ___________

### 6.6 — Zero-Touch Orchestrator Final Validation
- [ ] All 5 recipes completed with 100% step success
- [ ] Orchestrator confidence scores all ≥ 0.95 (auto-execute threshold)
- [ ] Risk score: _____ / 100 (target: <20 post-deployment)
- [ ] Security score: _____ / 100 (target: >80 post-deployment)
- [ ] Auto-fixable gaps resolved: ___________
- [ ] Remaining manual gaps documented and remediation plan created: ___________

---

## PHASE 7: MONITORING & ALERTING SETUP

### 7.1 — Prometheus & Metrics
- [ ] Prometheus installed and scraping QBITEL metrics (port 9090)
- [ ] Metrics retention configured (30-day minimum)
- [ ] Key BPO metrics confirmed flowing:
  - [ ] Active agent sessions count
  - [ ] Concurrent encrypted call streams
  - [ ] DTMF masking events per hour
  - [ ] Toll fraud blocks per hour
  - [ ] PCI scope events per call
  - [ ] Autonomous response events
  - [ ] PQC encryption latency (p50, p95, p99)
  - [ ] Recording encryption throughput

### 7.2 — Grafana Dashboards
- [ ] Grafana installed and connected to Prometheus
- [ ] QBITEL BPO Dashboard deployed with:
  - [ ] Voice Quality panel (MOS, latency, packet loss)
  - [ ] Fraud Prevention panel (blocked calls, fraud types, savings)
  - [ ] PCI Compliance panel (DTMF masking events, scope tracking)
  - [ ] Agent Security panel (DLP events, posture compliance)
  - [ ] Compliance Status panel (9 framework health indicators)
  - [ ] Autonomous Response panel (actions taken, confidence scores)
- [ ] Dashboard shared with SOC team and operations team

### 7.3 — Alerting Rules
- [ ] P1 Critical alerts configured:
  - [ ] Voice quality degradation (MOS drop >10%) → PagerDuty + SMS
  - [ ] Autonomous response affecting live calls → PagerDuty + Call
  - [ ] False positive fraud block on legitimate trunk → PagerDuty + Call
  - [ ] QBITEL Engine service outage → PagerDuty + Call
  - [ ] HSM connectivity failure → PagerDuty + Call

- [ ] P2 High alerts configured:
  - [ ] Toll fraud rate spike (>5x baseline) → Email + Slack
  - [ ] PCI DTMF masking failure → Email + Slack + Compliance team
  - [ ] Recording encryption failure → Email + Slack
  - [ ] Remote agent posture compliance drop (<95%) → Email + Slack

- [ ] P3 Warning alerts configured:
  - [ ] Agent DLP event spike → Slack
  - [ ] Certificate expiry (<30 days) → Email
  - [ ] Disk usage >80% → Email
  - [ ] PQC key rotation due → Email

- [ ] Alert escalation paths documented:
  - L1: QBITEL automated response (78% of events)
  - L2: Customer SOC team
  - L3: QBITEL 24/7 support

### 7.4 — Log Management
- [ ] Structured logging enabled (JSON format with correlation IDs)
- [ ] Log aggregation target configured:
  - `☐ ELK Stack (Elasticsearch + Logstash + Kibana)`
  - `☐ Grafana Loki`
  - `☐ Splunk`
  - `☐ CloudWatch / Azure Monitor`
- [ ] BPO-specific log fields confirmed: call_id, agent_id, tenant_id, trunk_id
- [ ] Audit log retention configured: _____ years (PCI minimum: 1 year online, 3 years archive)
- [ ] Audit log tamper-evidence (blockchain-backed) confirmed: ☐ Yes

---

## PHASE 8: COMPLIANCE VALIDATION

### 8.1 — PCI-DSS Voice Controls Validation
- [ ] DTMF masking test: 10 payment calls — card digits not in recordings ☐ Pass
- [ ] PAN detection test: Luhn-valid test card numbers sent across data streams ☐ Blocked
- [ ] Agent screen masking test: Card number masked to last 4 digits ☐ Pass
- [ ] Recording pause/resume test: Auto-pause on payment detected ☐ Pass
- [ ] PCI scope reduction report generated and reviewed ☐ Pass
- [ ] PCI-DSS SAQ-D pre-filled evidence package generated ☐ Generated
- [ ] Compliance report generation time: _____ minutes (target: <10)

### 8.2 — Toll Fraud Validation
- [ ] 10 test calls to premium-rate prefixes: ☐ All blocked
- [ ] 5 test off-hours international calls: ☐ All blocked
- [ ] 3 test Wangiri callback patterns: ☐ Pattern detected and blocked
- [ ] 2 test calls with spoofed CLI: ☐ Anomaly flagged
- [ ] Normal legitimate calls (100): ☐ Zero false positives
- [ ] Fraud detection latency: _____ seconds (target: <1 second)

### 8.3 — Remote Agent Security Validation (if applicable)
- [ ] 5 remote agent logins from approved country: ☐ All succeeded
- [ ] 3 remote agent logins from blocked country: ☐ All blocked
- [ ] Non-compliant endpoint (missing disk encryption) login: ☐ Blocked
- [ ] Screen watermark visible (if visible mode): ☐ Confirmed
- [ ] Split tunnel prevention: ☐ Confirmed (non-QBITEL traffic blocked)

### 8.4 — DLP Validation
- [ ] Clipboard PII copy attempt: ☐ Blocked with event logged
- [ ] USB storage insertion on agent desktop: ☐ Blocked with event logged
- [ ] Screen capture attempt (PrintScreen): ☐ Blocked
- [ ] Outbound email with PII pattern: ☐ Flagged / blocked
- [ ] Legitimate clipboard use (non-PII): ☐ Permitted (no false positive)

### 8.5 — Autonomous Response Validation
- [ ] Simulated SIP injection attack: ☐ Blocked <1 second, event logged
- [ ] Simulated bulk data access by agent: ☐ Rate limited, supervisor notified
- [ ] Simulated recording tampering: ☐ Integrity alert, evidence locked
- [ ] LLM threat narrative generated: ☐ Plain-language explanation received
- [ ] Autonomous action audit trail: ☐ Complete log with reasoning chain
- [ ] Emergency stop button tested: ☐ All autonomous actions frozen

### 8.6 — Compliance Report Generation
- [ ] PCI-DSS 4.0 report: ☐ Generated | Time: _____
- [ ] HIPAA Technical Safeguards report: ☐ Generated / ☐ N/A
- [ ] SOC 2 Type II evidence package: ☐ Generated
- [ ] GDPR compliance report: ☐ Generated / ☐ N/A
- [ ] SOX recording integrity report: ☐ Generated / ☐ N/A
- [ ] All reports generated in <10 minutes: ☐ Confirmed
- [ ] Compliance reports reviewed by CCO or Compliance Officer

---

## PHASE 9: USER ACCEPTANCE TESTING

### 9.1 — Voice Quality UAT
- [ ] 100-call voice quality test with live agents
- [ ] MOS scores measured: Pre _____ | Post _____ (must be equal or better)
- [ ] Agent-reported call quality: ☐ No degradation noted
- [ ] Supervisor monitoring quality: ☐ No degradation noted
- [ ] DTMF tone quality (payment flow): ☐ No degradation for callers
- [ ] Transfer/hold/conference call flows: ☐ All working

### 9.2 — Agent Desktop UAT
- [ ] Agents can log in normally: ☐ Confirmed
- [ ] TN3270e / mainframe access working: ☐ Confirmed
- [ ] CRM screen pop working: ☐ Confirmed
- [ ] Recording indicator working: ☐ Confirmed
- [ ] Payment call flow tested end-to-end with agent: ☐ Pass
- [ ] DLP policy does not block normal agent workflows: ☐ Confirmed
- [ ] Agents cannot detect any difference (core security): ☐ Confirmed

### 9.3 — Supervisor & Operations UAT
- [ ] Supervisor call monitoring working: ☐ Confirmed
- [ ] Call barge-in / whisper working: ☐ Confirmed
- [ ] Compliance dashboard accessible to operations team: ☐ Confirmed
- [ ] Fraud alert notifications delivered to supervisor: ☐ Confirmed
- [ ] Agent session termination (for rogue device scenario) tested: ☐ Pass
- [ ] Compliance report generation by operations team: ☐ Confirmed

### 9.4 — UAT Sign-Off
- [ ] Contact Center Director UAT approval: ___________________ Date: ___________
- [ ] Operations Lead UAT approval: ___________________ Date: ___________
- [ ] Security Team UAT approval: ___________________ Date: ___________
- [ ] Compliance Officer UAT approval: ___________________ Date: ___________

---

## PHASE 10: GO-LIVE & HANDOVER

### 10.1 — Pre-Go-Live Final Checks (Day Before)
- [ ] Code freeze on QBITEL Engine in customer environment
- [ ] Full staging environment validation completed
- [ ] Load testing completed at projected peak call volume
- [ ] Final security scan completed: ☐ No critical findings
- [ ] Rollback procedure tested and confirmed working
- [ ] Emergency contact list distributed to all teams:
  - QBITEL 24/7 support: enterprise@qbitel.com
  - Customer IT emergency contact: ___________
  - Customer SOC on-call: ___________
  - Contact Center operations: ___________

### 10.2 — Go-Live Day Execution
- [ ] All monitoring dashboards open and green
- [ ] QBITEL support team on standby for first 4 hours
- [ ] Change management window open (if required by customer policy)
- [ ] Final trunk-by-trunk activation if any remaining
- [ ] 30-minute post-activation health check:
  - [ ] Call success rate: _____ % (target: ≥99.9% baseline)
  - [ ] DTMF masking: ☐ Active on all payment paths
  - [ ] Toll fraud monitoring: ☐ Active
  - [ ] Recording encryption: ☐ Active
  - [ ] No P1 alerts: ☐ Confirmed

### 10.3 — Runbook & Documentation Handover
- [ ] QBITEL Engine administration runbook delivered
- [ ] Incident response playbook for BPO-specific events delivered:
  - [ ] Toll fraud response procedure
  - [ ] DTMF masking failure recovery procedure
  - [ ] Remote agent quarantine procedure
  - [ ] Recording system recovery procedure
  - [ ] PCI audit evidence retrieval procedure
- [ ] Rollback procedure document delivered
- [ ] Compliance report generation guide delivered
- [ ] Emergency stop procedure documented and posted
- [ ] Escalation matrix documented

### 10.4 — Training Delivery
- [ ] SOC team training: QBITEL security event response (2 hours) ☐ Completed
- [ ] Operations team training: Compliance dashboard and reporting (1 hour) ☐ Completed
- [ ] IT admin training: QBITEL Engine administration (2 hours) ☐ Completed
- [ ] Management briefing: Autonomous response and audit trail (1 hour) ☐ Completed
- [ ] Agent briefing (DLP only): What changed and why (30 minutes) ☐ Completed / ☐ N/A

### 10.5 — Go-Live Sign-Off
- [ ] Engineering Lead: ___________________ Date: ___________
- [ ] Security Lead (CISO): ___________________ Date: ___________
- [ ] DevOps / IT Lead: ___________________ Date: ___________
- [ ] Product Owner / Compliance Officer: ___________________ Date: ___________
- [ ] QBITEL Delivery Lead: ___________________ Date: ___________
- [ ] Customer Executive Sponsor: ___________________ Date: ___________

---

## PHASE 11: POST-GO-LIVE MONITORING (30-DAY PERIOD)

### Week 1 — Stabilization
- [ ] Daily health check calls (QBITEL CSM + Customer IT): ☐ Day 1 ☐ Day 2 ☐ Day 3 ☐ Day 4 ☐ Day 5
- [ ] False positive review — toll fraud false positives: _____ (tune if >0)
- [ ] False positive review — DLP false positives: _____ (tune if >0)
- [ ] Autonomous response review — all actions appropriate: ☐ Confirmed
- [ ] AI model baseline learning confirmed: ☐ Environment baseline established

### Week 2 — Optimization
- [ ] Toll fraud detection rate reviewed and baseline set
- [ ] PQC encryption performance baseline confirmed (no latency regressions)
- [ ] Compliance report generated and reviewed by Compliance Officer
- [ ] First PCI-DSS scope reduction calculation delivered
- [ ] Confidence threshold tuning completed (if required)

### Week 3–4 — Hardening
- [ ] Encryption key rotation cycle confirmed active
- [ ] Backup restoration tested (DR drill)
- [ ] Security posture re-assessed by Zero-Touch Orchestrator
- [ ] Risk score at 30 days: _____ / 100 (target: <15)
- [ ] Security score at 30 days: _____ / 100 (target: >85)
- [ ] First monthly compliance reports generated for all frameworks

### 30-Day Business Review
- [ ] Toll fraud losses prevented: $ ___________
- [ ] Fraud events detected and blocked: ___________
- [ ] PCI-DSS audit scope reduction confirmed: _____ %
- [ ] Autonomous threat resolutions: _____ (target: >78% of events)
- [ ] False positive rate: _____ % (target: <1%)
- [ ] Agent complaints or issues: ___________
- [ ] ROI vs. projected: ___________
- [ ] 30-day review meeting completed with executive sponsor: ☐ Completed

---

## APPENDIX A: DEPLOYMENT PERFORMANCE TARGETS

| Metric | Target | Measured | ☐ Met |
|--------|--------|----------|-------|
| Voice PQC encryption overhead | <2ms | ___ ms | ☐ |
| DTMF masking latency | <5ms | ___ ms | ☐ |
| SIP signaling processing | <10ms p95 | ___ ms | ☐ |
| TN3270e session latency | <300ms | ___ ms | ☐ |
| Toll fraud detection | <1 second | ___ sec | ☐ |
| PAN detection latency | <50ms | ___ ms | ☐ |
| Recording encryption throughput | 10,000+ streams | ___ streams | ☐ |
| Compliance report generation | <10 minutes | ___ min | ☐ |
| Autonomous threat resolution | ≥78% | ___ % | ☐ |
| Agent session validation | <100ms | ___ ms | ☐ |
| Deployment time | 4–6 hours | ___ hours | ☐ |
| Call success rate post-deployment | ≥99.9% baseline | ___ % | ☐ |

---

## APPENDIX B: ROLLBACK PROCEDURE

> **Use only if a critical issue is detected post-activation. Rollback is per-trunk, not full system.**

### Trunk-Level Rollback (preferred — <60 seconds)
1. Open QBITEL Management Console → Trunks
2. Select affected trunk
3. Click "Disable PQC Encryption" → Confirm
4. Verify calls on that trunk are routing normally
5. Log incident in QBITEL support portal
6. Contact QBITEL support: enterprise@qbitel.com

### Full System Rollback (only if engine unavailable)
1. Activate hardware bypass on network tap (removes QBITEL from path entirely — <5 seconds)
2. Verify all call flows restored
3. Contact QBITEL 24/7 support immediately
4. Preserve logs and diagnostic data before any changes

### Database Rollback
```bash
# Rollback schema migration
alembic downgrade -1

# Restore from backup (if required)
pg_restore -d qbitel_prod backup_YYYYMMDD.dump
```
**RTO targets:** Application: 5 minutes | Database: 30 minutes | Full cluster: 2 hours

---

## APPENDIX C: SUPPORT & ESCALATION CONTACTS

| Tier | Trigger | Contact | Response SLA |
|------|---------|---------|-------------|
| P1 Critical | Voice degradation, autonomous response affecting live calls | enterprise@qbitel.com + 24/7 phone | 30 min (Global) / 1 hour (Enterprise) |
| P2 High | Fraud spike, masking failure, compliance alert | enterprise@qbitel.com | 4 hours |
| P3 Warning | Certificate expiry, capacity, configuration drift | enterprise@qbitel.com | 1 business day |
| General | Questions, optimization, reporting | bridge.qbitel.com support portal | 2 business days |

**QBITEL Support Portal:** https://bridge.qbitel.com
**Enterprise Email:** enterprise@qbitel.com

---

*Document Version 1.0 | February 2026*
*QBITEL Bridge — BPO Deployment & Delivery Checklist*
*Confidential — For Authorized Deployment Teams Only*
