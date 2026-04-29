# BPO & Call Center Security Module

## Product Overview

The QBITEL BPO & Call Center Security Module provides quantum-safe protection for contact center operations without replacing existing PBX, telephony, or agent desktop infrastructure. It operates at the network layer, discovering and securing all voice, signaling, and data protocols autonomously.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    QBITEL BPO Security Layer                     │
├──────────────┬───────────────┬──────────────┬───────────────────┤
│ Voice        │ Agent Desktop │ Terminal     │ Remote Agent      │
│ Security     │ DLP           │ Security     │ Access            │
│              │               │              │                   │
│ SIP-PQC-TLS │ Clipboard     │ TN3270e-PQC  │ VPN-less PQC      │
│ SRTP-PQC    │ Screen Cap    │ TN5250-PQC   │ Tunnels           │
│ DTMF Mask   │ USB Block     │ Session Mon  │ Endpoint          │
│ Toll Fraud  │ PII Mask      │ Audit Trail  │ Compliance        │
├──────────────┴───────────────┴──────────────┴───────────────────┤
│              PQC Engine (ML-KEM + ML-DSA + AES-256-GCM)         │
├─────────────────────────────────────────────────────────────────┤
│              AI Protocol Discovery + Zero-Touch Security         │
├─────────────────────────────────────────────────────────────────┤
│              Existing BPO Infrastructure (unchanged)             │
│  Avaya │ Cisco │ Genesys │ Asterisk │ Legacy PBX │ Mainframes  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Domain Profiles (`ai_engine/domains/bpo/core/domain_profile.py`)

Pre-configured PQC profiles optimized for each BPO subdomain:

| Profile | KEM | Signature | Max Latency | Use Case |
|---------|-----|-----------|-------------|----------|
| `voice_signaling` | ML-KEM-512 | Falcon-512 | 50ms | SIP signaling encryption |
| `voice_media` | ML-KEM-512 | Falcon-512 | 20ms | RTP/SRTP media encryption |
| `agent_desktop` | ML-KEM-768 | ML-DSA-65 | 200ms | Agent session security |
| `terminal_emulation` | ML-KEM-768 | ML-DSA-65 | 300ms | 3270/5250 terminal protection |
| `payment_processing` | ML-KEM-1024 | ML-DSA-87 | 100ms | PCI-DSS payment calls |
| `call_recording` | ML-KEM-1024 | ML-DSA-87 | 1000ms | Long-term recording storage |
| `remote_agent` | ML-KEM-768 | ML-DSA-65 | 200ms | WFH agent tunnels |
| `cti_middleware` | ML-KEM-512 | Falcon-512 | 100ms | CTI event processing |

### 2. Security Policies (`ai_engine/domains/bpo/core/security_policy.py`)

Pre-built security policy templates:

- **`financial_services`**: PCI-DSS + SOX + GLBA — DTMF masking, recording encryption, 10-year retention
- **`healthcare`**: HIPAA + HITECH — PHI protection, minimum necessary access, 6-year audit
- **`general_customer_service`**: GDPR + SOC 2 — PII masking, toll fraud prevention
- **`remote_workforce`**: Enhanced endpoint security — clipboard blocking, geo-fencing, watermarking

### 3. Protocol Handlers (`ai_engine/domains/bpo/protocols/`)

| Protocol | Handler | Capabilities |
|----------|---------|-------------|
| **SIP/SDP** | `sip/` | Message parsing, toll fraud detection, PQC-TLS wrapping, DTMF relay security |
| **TN3270e** | `terminal/` | Session parsing, screen field detection, data exfiltration monitoring |
| **IVR** | `ivr/` | DTMF input validation, payment flow PCI compliance, navigation security |
| **CTI** | `cti/` | Agent state validation, call routing integrity, permission enforcement |

### 4. Security Modules (`ai_engine/domains/bpo/security/`)

| Module | File | Function |
|--------|------|----------|
| **Toll Fraud** | `toll_fraud.py` | IRSF detection, premium rate blocking, volume anomaly detection |
| **PCI Voice** | `pci_voice.py` | DTMF masking, PAN detection, recording pause/resume, scope management |
| **Session Monitor** | `session_monitor.py` | Agent behavior anomaly detection, data access pattern monitoring |
| **DLP** | `data_loss_prevention.py` | PII detection, clipboard/screen/USB/email channel monitoring |
| **Remote Access** | `remote_access.py` | VPN-less PQC tunnels, endpoint compliance, geo-fencing |

### 5. Integration Bridges (`ai_engine/domains/bpo/integrations/`)

| Integration | Bridge | Systems |
|-------------|--------|---------|
| **PBX** | `pbx/` | Avaya Aura, Cisco CUCM, Genesys, Asterisk, FreeSWITCH |
| **CRM** | `crm/` | Salesforce, Zendesk, ServiceNow, Dynamics 365 |
| **WFM** | `wfm/` | NICE, Verint, Aspect, Calabrio |

### 6. Compliance Context (`ai_engine/domains/bpo/core/compliance_context.py`)

BPO-specific compliance controls across 10 frameworks with:
- Call-aware audit trail (call_id, agent_id, tenant_id tracking)
- DTMF masking event logging for PCI-DSS evidence
- Recording pause/resume event tracking
- Multi-tenant compliance isolation
- Blockchain-backed integrity verification

## API Endpoints

```
POST /api/v1/bpo/voice/analyze          - Analyze SIP/RTP traffic for threats
POST /api/v1/bpo/voice/dtmf-mask        - Activate/deactivate DTMF masking
POST /api/v1/bpo/voice/recording/pause  - Pause call recording (PCI)
POST /api/v1/bpo/voice/recording/resume - Resume call recording
POST /api/v1/bpo/fraud/check            - Check number against fraud database
POST /api/v1/bpo/agent/session/validate - Validate agent session security
POST /api/v1/bpo/dlp/scan               - Scan data for PII patterns
POST /api/v1/bpo/compliance/report      - Generate BPO compliance report
GET  /api/v1/bpo/agent/{id}/posture     - Get agent endpoint posture
GET  /api/v1/bpo/tenant/{id}/status     - Get tenant compliance status
```

## Configuration

```yaml
# config/qbitel.yaml - BPO domain configuration
bpo:
  enabled: true

  voice_security:
    dtmf_masking:
      enabled: true
      mode: "CLAMP"          # CLAMP, FLAT_TONE, SILENCE
      mask_in_recording: true
      mask_on_screen: true

    encryption:
      signaling: "SIP-TLS-PQC"
      media: "SRTP-PQC-AES256-GCM"
      recording: "ML-KEM-1024"

    toll_fraud:
      enabled: true
      block_premium_rate: true
      max_call_duration_hours: 4
      rate_limit_per_second: 100

  agent_security:
    mfa_required: true
    idle_timeout_minutes: 5
    block_clipboard: true
    block_screen_capture: true
    block_usb: true
    watermark_screen: true
    pii_masking: true

  remote_access:
    vpn_less_tunnels: true
    kem_algorithm: "ML-KEM-768"
    endpoint_compliance:
      require_antivirus: true
      require_disk_encryption: true
      require_wpa3: true
    geo_fencing:
      enabled: true
      allowed_countries: ["US", "IN", "PH"]

  compliance:
    frameworks:
      - "PCI-DSS-4.0"
      - "GDPR"
      - "SOC-2"
    recording_retention_years: 7
    audit_retention_days: 2555

  integrations:
    pbx:
      type: "AVAYA_AURA"
      host: "pbx.internal"
      pqc_tunnel: true
    crm:
      type: "SALESFORCE"
      pii_masking: true
    wfm:
      type: "NICE_WFM"
      schedule_enforcement: true
```

## Performance Specifications

| Metric | Value | Notes |
|--------|-------|-------|
| Voice PQC overhead | <2ms | Within ITU-T G.114 budget |
| SIP message processing | <10ms p95 | Including PQC operations |
| DTMF masking latency | <5ms | Imperceptible to caller |
| Concurrent sessions | 20,000+ | Per deployment instance |
| Toll fraud detection | <1 second | 3-call pattern recognition |
| PAN detection | <50ms | Real-time Luhn validation |
| Recording encryption | 10,000+ streams | Concurrent encryption |
| Compliance report | <10 minutes | Full multi-framework report |
| Agent session validation | <100ms | Per-request posture check |
| Deployment time | 4-6 hours | Zero downtime |
