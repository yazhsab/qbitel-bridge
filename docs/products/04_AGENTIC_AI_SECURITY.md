# Product 4: Agentic AI Security (Zero-Touch Decision Engine)

## Overview

Agentic AI Security is CRONOS AI's autonomous security operations platform powered by Large Language Models (LLMs). It analyzes security events, makes intelligent decisions, and executes responses with 78% autonomy—reducing human intervention from hours to seconds while maintaining safety constraints and human oversight.

---

## Problem Solved

### The Challenge

Security operations centers (SOCs) face critical challenges:

- **Alert fatigue**: 10,000+ alerts/day, 99% false positives
- **Response time**: Average 65-140 minutes from detection to containment
- **Talent shortage**: 3.5M unfilled cybersecurity positions globally
- **Manual investigation**: Hours spent on routine threat analysis
- **Inconsistent decisions**: Human judgment varies, especially under pressure

### The CRONOS AI Solution

Agentic AI Security provides:
- **78% autonomous response**: LLM-powered decisions with <1 second response time
- **Human-in-the-loop**: Escalation for high-risk decisions
- **On-premise LLM**: Air-gapped deployment with Ollama (no cloud dependency)
- **Continuous learning**: Improves from outcomes and feedback
- **Safety constraints**: Prevents dangerous actions without approval

---

## Key Features

### 1. Zero-Touch Decision Matrix

Automated decision framework based on confidence and risk:

| Confidence Level | Low Risk (0-0.3) | Medium Risk (0.3-0.7) | High Risk (0.7-1.0) |
|------------------|------------------|-----------------------|---------------------|
| **High (0.95-1.0)** | Auto-Execute | Auto-Approve | Escalate |
| **Medium (0.85-0.95)** | Auto-Execute | Auto-Approve | Escalate |
| **Low-Medium (0.50-0.85)** | Auto-Approve | Escalate | Escalate |
| **Low (<0.50)** | Escalate | Escalate | Escalate |

### 2. LLM-Powered Analysis

Uses advanced language models for threat understanding:

- **Threat narrative**: Human-readable explanation of attack
- **TTP mapping**: Automatic MITRE ATT&CK technique identification
- **Business impact**: Assessment of operational and financial risk
- **Response recommendation**: Prioritized list of mitigation actions
- **Correlation**: Links related events across time and systems

### 3. On-Premise LLM Support

Air-gapped deployment for sensitive environments:

**Primary (On-Premise)**:
| Provider | Models | Use Case |
|----------|--------|----------|
| **Ollama** | Llama 3.2 (8B, 70B), Mixtral 8x7B, Qwen2.5, Phi-3 | Default, recommended |
| **vLLM** | Any HuggingFace model | High-performance GPU |
| **LocalAI** | OpenAI-compatible models | Existing infrastructure |

**Fallback (Cloud - Optional)**:
| Provider | Models | When to Use |
|----------|--------|-------------|
| Anthropic | Claude 3.5 Sonnet | Complex analysis |
| OpenAI | GPT-4 | Fallback only |

### 4. Automated Response Actions

Pre-built response playbooks:

| Risk Level | Actions | Auto-Execute? |
|------------|---------|---------------|
| **Low** | Alert, log, monitor, rate-limit | Yes |
| **Medium** | Block IP, segment network, disable account | Yes (with approval) |
| **High** | Isolate host, shutdown service, revoke credentials | Escalate |
| **Critical** | Full incident response, executive notification | Always escalate |

### 5. Safety Constraints

Built-in guardrails prevent dangerous autonomous actions:

- **Blast radius limits**: Cannot affect >10 systems without approval
- **Production safeguards**: Extra confirmation for production systems
- **Rollback capability**: Every action can be instantly reversed
- **Audit trail**: Complete logging of all decisions and actions
- **Human override**: Emergency stop and manual takeover

---

## Technical Architecture

### Decision Flow

```
Security Event
    |
    v
+----------------------+
| Event Normalization  |
| - Type detection     |
| - Context extraction |
| - Field enrichment   |
+----------------------+
    |
    v
+----------------------+
| Threat Analysis      |
| - ML Classification  |
| - LLM Analysis       |
| - Business Impact    |
+----------------------+
    |
    v
+----------------------+
| Response Generation  |
| - Action selection   |
| - Risk calculation   |
| - Playbook mapping   |
+----------------------+
    |
    v
+----------------------+
| Decision Engine      |
| - Confidence score   |
| - Risk assessment    |
| - Safety constraints |
+----------------------+
    |
    +--------+--------+
    |        |        |
    v        v        v
Auto-Execute  Auto-Approve  Escalate
(78%)        (10%)         (12%)
```

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| Decision Engine | `decision_engine.py` | Core decision logic (1,360+ LOC) |
| Threat Analyzer | `threat_analyzer.py` | ML-based threat analysis |
| Security Service | `security_service.py` | Orchestration service |
| Legacy Response | `legacy_response.py` | Legacy system handling |
| Secrets Manager | `secrets_manager.py` | Credential management |
| Resilience | `resilience/` | Circuit breaker, retry patterns |

### Data Models

```python
@dataclass
class SecurityEvent:
    event_id: str
    event_type: SecurityEventType
    timestamp: datetime
    source_system: str
    source_ip: str
    destination_ip: str
    protocol: str
    payload_size: int
    anomaly_score: float
    threat_level: ThreatLevel
    affected_resources: List[str]
    context_data: Dict[str, Any]
    raw_event: bytes

class SecurityEventType(str, Enum):
    ANOMALOUS_TRAFFIC = "anomalous_traffic"
    AUTHENTICATION_FAILURE = "authentication_failure"
    MALWARE_DETECTED = "malware_detected"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    COMMAND_AND_CONTROL = "command_and_control"
    POLICY_VIOLATION = "policy_violation"

class ThreatLevel(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ThreatAnalysis:
    threat_level: ThreatLevel
    confidence: float  # 0.0-1.0
    attack_vectors: List[str]
    ttps: List[str]  # MITRE ATT&CK TTPs
    narrative: str  # LLM-generated explanation
    business_impact: str
    financial_risk: float
    operational_impact: str
    recommended_actions: List[ResponseAction]
    mitigation_steps: List[str]
    related_events: List[str]
    iocs: List[str]  # Indicators of Compromise

@dataclass
class AutomatedResponse:
    response_id: str
    action_type: ResponseType
    confidence_level: ConfidenceLevel
    risk_level: str
    execution_status: str
    response_details: Dict[str, Any]
    executed_at: datetime
    execution_time_ms: float
    outcome: Optional[str]
    success: bool
    rollback_available: bool

class ResponseType(str, Enum):
    ALERT = "alert"
    LOG = "log"
    MONITOR = "monitor"
    RATE_LIMIT = "rate_limit"
    BLOCK_IP = "block_ip"
    BLOCK_USER = "block_user"
    SEGMENT_NETWORK = "segment_network"
    DISABLE_ACCOUNT = "disable_account"
    ISOLATE_HOST = "isolate_host"
    SHUTDOWN_SERVICE = "shutdown_service"
    REVOKE_CREDENTIALS = "revoke_credentials"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"
```

---

## API Reference

### Analyze Security Event

```http
POST /api/v1/security/analyze-event
Content-Type: application/json
X-API-Key: your_api_key

{
    "event_type": "anomalous_traffic",
    "timestamp": "2025-01-16T10:30:00Z",
    "source_system": "scada_plc_001",
    "source_ip": "192.168.1.50",
    "destination_ip": "10.0.0.100",
    "protocol": "modbus",
    "anomaly_score": 0.92,
    "threat_level": "high",
    "affected_resources": ["pump_station_1", "valve_controller_3"],
    "context_data": {
        "request_rate": 1500,
        "normal_rate": 100,
        "function_codes": [6, 16, 23],
        "unusual_registers": [40001, 40002, 40003]
    }
}
```

### Response

```json
{
    "analysis_id": "analysis_abc123",
    "threat_level": "high",
    "confidence": 0.92,
    "narrative": "Detected anomalous Modbus traffic from PLC controller 192.168.1.50. The request rate (1500/min) is 15x higher than the baseline (100/min). The attacker is using function codes 6 (Write Single Register), 16 (Write Multiple Registers), and 23 (Read/Write Multiple Registers) to modify control registers in the 40000 range. This pattern is consistent with MITRE ATT&CK technique T0855 (Unauthorized Command Message) targeting industrial control systems. Immediate action recommended to prevent physical process manipulation.",
    "attack_vectors": [
        "Industrial Control System Attack",
        "SCADA Protocol Abuse",
        "Unauthorized Register Modification"
    ],
    "ttps": [
        "T0855 - Unauthorized Command Message",
        "T0831 - Manipulation of Control",
        "T0882 - Theft of Operational Information"
    ],
    "business_impact": "HIGH - Potential physical process manipulation affecting pump station and valve controller. Risk of operational disruption and safety incidents.",
    "financial_risk": 2500000,
    "operational_impact": "Critical infrastructure at risk. Pump station 1 and valve controller 3 may be compromised.",
    "recommended_actions": [
        {
            "action_type": "rate_limit",
            "target": "192.168.1.50",
            "details": {"limit": "100 req/min"},
            "risk": 0.2,
            "confidence": 0.95,
            "auto_executable": true
        },
        {
            "action_type": "segment_network",
            "target": "scada_plc_001",
            "details": {"vlan": "isolated_scada"},
            "risk": 0.4,
            "confidence": 0.88,
            "auto_executable": true
        },
        {
            "action_type": "alert",
            "target": "soc_team",
            "details": {"severity": "critical", "channel": "pagerduty"},
            "risk": 0.0,
            "confidence": 1.0,
            "auto_executable": true
        }
    ],
    "mitigation_steps": [
        "1. Implement rate limiting on Modbus traffic from affected PLC",
        "2. Isolate affected PLC to dedicated VLAN",
        "3. Review and restore modified registers to safe values",
        "4. Analyze traffic logs for additional IOCs",
        "5. Update firewall rules to block unauthorized Modbus commands"
    ],
    "decision": {
        "action": "auto_approve",
        "reason": "High confidence (0.92) with medium-risk actions. Rate limiting and network segmentation are reversible.",
        "requires_human": false,
        "escalation_level": null
    },
    "execution_status": {
        "rate_limit": "executed",
        "segment_network": "executed",
        "alert": "executed"
    },
    "processing_time_ms": 450
}
```

### Execute Response Action

```http
POST /api/v1/security/execute-action
Content-Type: application/json
X-API-Key: your_api_key

{
    "analysis_id": "analysis_abc123",
    "action_index": 0,
    "override_safety": false,
    "reason": "Approved by SOC analyst"
}

Response:
{
    "execution_id": "exec_xyz789",
    "action_type": "rate_limit",
    "status": "completed",
    "executed_at": "2025-01-16T10:30:05Z",
    "execution_time_ms": 45,
    "result": {
        "previous_state": {"rate_limit": null},
        "new_state": {"rate_limit": "100 req/min"},
        "affected_systems": ["firewall_001"],
        "rollback_command": "exec_xyz789_rollback"
    }
}
```

### Rollback Action

```http
POST /api/v1/security/rollback
Content-Type: application/json
X-API-Key: your_api_key

{
    "execution_id": "exec_xyz789",
    "reason": "False positive confirmed"
}

Response:
{
    "rollback_id": "rollback_abc123",
    "status": "completed",
    "original_action": "rate_limit",
    "restored_state": {"rate_limit": null},
    "rolled_back_at": "2025-01-16T11:00:00Z"
}
```

### Get Decision History

```http
GET /api/v1/security/decisions?limit=100&status=auto_executed
X-API-Key: your_api_key

Response:
{
    "total": 1547,
    "decisions": [
        {
            "decision_id": "dec_001",
            "timestamp": "2025-01-16T10:30:00Z",
            "event_type": "anomalous_traffic",
            "threat_level": "medium",
            "confidence": 0.94,
            "decision": "auto_execute",
            "actions_taken": ["rate_limit", "alert"],
            "outcome": "successful",
            "human_intervention": false
        }
    ],
    "statistics": {
        "auto_executed": 1205,
        "auto_approved": 155,
        "escalated": 187,
        "success_rate": 0.94
    }
}
```

### Additional Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/security/events` | GET | List security events |
| `/api/v1/security/playbooks` | GET | List response playbooks |
| `/api/v1/security/playbooks/{id}` | GET | Get playbook details |
| `/api/v1/security/settings` | GET/PUT | Configure decision thresholds |
| `/api/v1/security/metrics` | GET | Performance metrics |
| `/api/v1/security/health` | GET | Service health |

---

## On-Premise LLM Configuration

### Ollama Setup (Recommended)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull recommended models
ollama pull llama3.2:70b        # Primary analysis model
ollama pull mixtral:8x7b        # Alternative for complex analysis
ollama pull phi:3               # Fast triage model

# Start Ollama server
ollama serve

# Verify
curl http://localhost:11434/api/tags
```

### CRONOS AI Configuration

```yaml
security:
  agentic_ai:
    enabled: true

    # LLM Configuration
    llm:
      provider: ollama  # ollama, vllm, localai, anthropic, openai
      endpoint: http://localhost:11434
      model: llama3.2:70b
      fallback_model: mixtral:8x7b
      timeout_seconds: 30
      max_tokens: 4096
      temperature: 0.1  # Low for deterministic responses

      # Air-gapped mode
      airgapped: true
      disable_cloud_fallback: true

    # Decision Thresholds
    decision:
      auto_execute_confidence: 0.85
      auto_approve_confidence: 0.70
      escalation_threshold: 0.50

      risk_weights:
        low: 0.3
        medium: 0.5
        high: 0.8
        critical: 1.0

      # Safety constraints
      max_affected_systems: 10
      require_approval_production: true
      enable_rollback: true

    # Response Configuration
    response:
      default_playbook: standard_response
      max_concurrent_actions: 5
      action_timeout_seconds: 60
      retry_failed_actions: true
      max_retries: 3

    # Escalation
    escalation:
      channels:
        - type: pagerduty
          integration_key: ${PAGERDUTY_KEY}
          severity_threshold: high
        - type: slack
          webhook: ${SLACK_WEBHOOK}
          channel: "#security-alerts"
          severity_threshold: medium
        - type: email
          recipients: ["soc@company.com"]
          severity_threshold: low

    # Learning
    learning:
      enabled: true
      feedback_collection: true
      model_update_interval: "7d"
      min_samples_for_update: 100
```

### vLLM Setup (High Performance)

```bash
# Install vLLM
pip install vllm

# Start server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4 \
    --port 8000

# Configure CRONOS AI
export CRONOS_LLM_PROVIDER=vllm
export CRONOS_LLM_ENDPOINT=http://localhost:8000/v1
```

---

## Performance Metrics

### Decision Performance

| Metric | Value | SLA |
|--------|-------|-----|
| **Decision time** | <1 second | <5 seconds |
| **Analysis throughput** | 1,000+ events/sec | 500 events/sec |
| **Autonomous rate** | 78% | >70% |
| **False positive rate** | <5% | <10% |
| **Response accuracy** | 94% | >90% |

### Response Time Comparison

| Metric | Manual SOC | CRONOS AI | Improvement |
|--------|-----------|-----------|-------------|
| **Detection to triage** | 15 min | <1 sec | 900x |
| **Triage to decision** | 30 min | <1 sec | 1,800x |
| **Decision to action** | 20 min | <5 sec | 240x |
| **Total response** | 65 min | <10 sec | 390x |

### Autonomy Breakdown

| Decision Type | Percentage | Description |
|---------------|------------|-------------|
| **Auto-Execute** | 78% | Full automation, no human involved |
| **Auto-Approve** | 10% | Recommended action, quick approval |
| **Escalate** | 12% | Human decision required |

---

## Monitoring & Observability

### Prometheus Metrics

```
# Decision metrics
cronos_security_decisions_total{decision_type="auto_execute|auto_approve|escalate", threat_level}
cronos_security_decision_duration_seconds{phase="analysis|decision|execution"}
cronos_security_decision_confidence{decision_type}

# Response metrics
cronos_security_responses_total{response_type, status="success|failure"}
cronos_security_response_duration_seconds{response_type}
cronos_security_rollbacks_total{response_type}

# Autonomy metrics
cronos_security_auto_executions_per_hour
cronos_security_human_escalations_per_hour
cronos_security_autonomy_rate

# LLM metrics
cronos_security_llm_requests_total{provider, model}
cronos_security_llm_latency_seconds{provider, model}
cronos_security_llm_tokens_used{provider, model}
cronos_security_llm_errors_total{provider, error_type}

# Accuracy metrics
cronos_security_true_positives_total
cronos_security_false_positives_total
cronos_security_accuracy_rate
```

### Dashboard Queries

```promql
# Autonomy rate
sum(rate(cronos_security_decisions_total{decision_type="auto_execute"}[1h])) /
sum(rate(cronos_security_decisions_total[1h])) * 100

# Average decision time
histogram_quantile(0.95, rate(cronos_security_decision_duration_seconds_bucket[5m]))

# Response success rate
sum(rate(cronos_security_responses_total{status="success"}[1h])) /
sum(rate(cronos_security_responses_total[1h])) * 100
```

---

## Use Cases

### Use Case 1: Automated Brute Force Response

```python
# Incoming event
event = SecurityEvent(
    event_type="authentication_failure",
    source_ip="203.0.113.50",
    anomaly_score=0.88,
    context_data={
        "failed_attempts": 150,
        "time_window": "5m",
        "targeted_accounts": 50
    }
)

# CRONOS AI analysis (automatic)
analysis = await security.analyze(event)
# Result: Confidence 0.95, Risk: Medium
# Decision: Auto-Execute
# Actions: Block IP, Alert SOC

# Actions executed automatically
# - Firewall rule added blocking 203.0.113.50
# - PagerDuty alert sent
# - Event logged for investigation
```

### Use Case 2: SCADA Anomaly with Escalation

```python
# Incoming event
event = SecurityEvent(
    event_type="anomalous_traffic",
    source_system="scada_hmi_001",
    threat_level="critical",
    context_data={
        "modified_setpoints": ["temperature", "pressure"],
        "safety_system_accessed": True
    }
)

# CRONOS AI analysis
analysis = await security.analyze(event)
# Result: Confidence 0.75, Risk: Critical
# Decision: Escalate (safety system involved)

# Escalation triggered
# - SOC team paged via PagerDuty
# - Recommended actions presented for approval
# - System NOT automatically isolated (safety constraint)
```

### Use Case 3: Lateral Movement Detection

```python
# Incoming event
event = SecurityEvent(
    event_type="lateral_movement",
    source_ip="10.0.1.50",
    destination_ip="10.0.2.100",
    context_data={
        "protocol": "smb",
        "credentials_used": "admin",
        "unusual_time": True,
        "data_volume": "500MB"
    }
)

# CRONOS AI analysis
analysis = await security.analyze(event)
# Result: Confidence 0.91, Risk: High
# Decision: Auto-Approve
# Actions: Segment network, Disable account, Alert

# Actions queued for quick approval
# Analyst approves in <30 seconds
# Actions executed immediately after approval
```

---

## Comparison: Traditional SOC vs. Agentic AI

| Aspect | Traditional SOC | Agentic AI Security |
|--------|-----------------|---------------------|
| **Response time** | 65-140 minutes | <10 seconds |
| **Alert handling** | 100-500/day/analyst | 10,000+/day automated |
| **Consistency** | Variable | 100% consistent |
| **24/7 coverage** | Expensive shifts | Always-on |
| **Skill dependency** | High | Low |
| **Cost per event** | $10-50 | <$0.01 |
| **False positive handling** | Manual review | AI filtering |
| **Learning** | Slow, training-based | Continuous |

---

## Safety & Governance

### Built-in Safeguards

1. **Blast radius limits**: Cannot affect >10 systems without approval
2. **Production protection**: Extra confirmation for production environments
3. **Reversibility**: Every action has rollback capability
4. **Audit logging**: Complete decision and action trail
5. **Human override**: Emergency stop always available
6. **Confidence thresholds**: Configurable automation levels
7. **Risk assessment**: Every action evaluated for impact

### Compliance Support

- **SOC 2**: Complete audit trail for all decisions
- **GDPR**: Data protection in decision-making
- **HIPAA**: Healthcare-specific safeguards
- **PCI-DSS**: Financial data handling
- **NIST CSF**: Framework alignment

---

## Conclusion

Agentic AI Security transforms security operations from reactive, manual processes to proactive, autonomous defense. With 78% autonomous operation, <1 second response times, and on-premise LLM support for air-gapped environments, it enables organizations to defend against threats at machine speed while maintaining human oversight for critical decisions.
