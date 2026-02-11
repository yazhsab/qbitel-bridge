# Product 7: Enterprise Compliance

## Overview

Enterprise Compliance is QBITEL's automated compliance management platform that continuously assesses, monitors, and reports on regulatory requirements. It supports 9 major compliance frameworks with automated evidence collection, gap analysis, and audit-ready reporting.

---

## Problem Solved

### The Challenge

Compliance management is costly and resource-intensive:

- **Manual assessments**: Weeks of effort per framework per quarter
- **Evidence collection**: Scattered across systems, hard to gather
- **Audit preparation**: Last-minute scramble, incomplete documentation
- **Multiple frameworks**: SOC 2, GDPR, PCI-DSS, HIPAA all require separate efforts
- **Continuous compliance**: Point-in-time audits miss ongoing violations

### The QBITEL Solution

Enterprise Compliance provides:
- **9 framework support**: SOC 2, GDPR, PCI-DSS, HIPAA, NIST CSF, ISO 27001, NERC CIP, FedRAMP, CMMC
- **Automated assessments**: Continuous evaluation of all controls
- **Evidence collection**: Automatic gathering from all systems
- **Gap analysis**: Real-time identification of compliance gaps
- **Audit-ready reports**: One-click PDF generation

---

## Key Features

### 1. Supported Compliance Frameworks

| Framework | Controls | Automation Level | Status |
|-----------|----------|-----------------|--------|
| **SOC 2 Type II** | 50+ | 90% automated | Production |
| **GDPR** | 99 articles | 85% automated | Production |
| **PCI-DSS 4.0** | 300+ | 80% automated | Production |
| **HIPAA** | 45 safeguards | 85% automated | Production |
| **NIST CSF** | 108 subcategories | 90% automated | Production |
| **ISO 27001** | 114 controls | 85% automated | Production |
| **NERC CIP** | 45 standards | 80% automated | Production |
| **FedRAMP** | 325+ controls | 75% automated | In Progress |
| **CMMC** | 171 practices | 75% automated | In Progress |

### 2. Automated Assessment Pipeline

```
Configuration Collection
    ├─ Kubernetes resources
    ├─ Database configurations
    ├─ Network policies
    ├─ IAM settings
    └─ Security configurations
           │
           ▼
Evidence Gathering
    ├─ Audit logs
    ├─ Configuration snapshots
    ├─ Security scan results
    ├─ Access reviews
    └─ Change records
           │
           ▼
Control Evaluation
    ├─ Policy compliance check
    ├─ Technical control validation
    ├─ Operational control assessment
    └─ Administrative control review
           │
           ▼
Gap Analysis
    ├─ Identify deviations
    ├─ Risk scoring
    ├─ Remediation prioritization
    └─ Timeline recommendations
           │
           ▼
Report Generation
    ├─ Executive summary
    ├─ Control findings
    ├─ Evidence attachments
    └─ Remediation plan
```

### 3. Continuous Monitoring

Real-time compliance status:

- **Dashboard**: Live compliance scores per framework
- **Alerts**: Immediate notification on violations
- **Trends**: Historical compliance trajectory
- **Predictions**: Risk of future non-compliance

### 4. Evidence Management

Centralized evidence repository:

- **Automatic collection**: From all integrated systems
- **Version control**: Historical evidence preservation
- **Tamper-proof**: Blockchain-backed integrity
- **Quick retrieval**: Full-text search across evidence

### 5. Audit Support

Streamlined audit experience:

- **Pre-packaged evidence**: Ready for auditor review
- **Auditor portal**: Secure external access
- **Response tracking**: Manage auditor questions
- **Gap remediation**: Track fix progress

---

## Technical Architecture

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| Compliance Service | `compliance_service.py` | Main orchestration |
| Regulatory KB | `regulatory_kb.py` | Compliance knowledge base |
| Assessment Engine | `assessment_engine.py` | Automated assessments |
| Report Generator | `report_generator.py` | PDF/JSON report generation |
| Audit Trail | `audit_trail.py` | Immutable logging |
| GDPR Module | `gdpr_compliance.py` | GDPR-specific controls |
| SOC 2 Module | `soc2_controls.py` | SOC 2 control mapping |

### Data Models

```python
@dataclass
class ComplianceFramework:
    framework_id: str
    name: str
    version: str
    controls: List[Control]
    requirements: List[Requirement]
    assessment_frequency: str
    last_assessment: datetime
    next_assessment: datetime

@dataclass
class Control:
    control_id: str
    name: str
    description: str
    category: str
    implementation_status: ImplementationStatus
    evidence_requirements: List[str]
    test_procedures: List[str]
    risk_level: str

class ImplementationStatus(str, Enum):
    NOT_IMPLEMENTED = "not_implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    FULLY_IMPLEMENTED = "fully_implemented"
    NOT_APPLICABLE = "not_applicable"

@dataclass
class ComplianceAssessment:
    assessment_id: str
    framework: str
    assessment_date: datetime
    assessor: str
    controls_evaluated: int
    controls_passing: int
    controls_failing: int
    controls_partial: int
    pass_rate: float
    gaps: List[ComplianceGap]
    remediation_items: List[RemediationItem]
    evidence: List[ComplianceEvidence]
    status: str

@dataclass
class ComplianceGap:
    gap_id: str
    control_id: str
    control_name: str
    gap_description: str
    risk_level: str
    business_impact: str
    remediation_recommendation: str
    estimated_effort: str
    due_date: datetime
    owner: str
    status: str

@dataclass
class ComplianceEvidence:
    evidence_id: str
    control_id: str
    evidence_type: str
    description: str
    file_path: str
    file_hash: str
    collected_at: datetime
    collected_by: str
    retention_period: str
```

---

## API Reference

### Create Assessment

```http
POST /api/v1/compliance/assessments
Content-Type: application/json
X-API-Key: your_api_key

{
    "framework": "soc2",
    "scope": {
        "trust_services_criteria": ["security", "availability", "confidentiality"],
        "systems": ["payment-system", "customer-portal"],
        "period": {
            "start": "2024-01-01",
            "end": "2024-12-31"
        }
    },
    "assessor": {
        "name": "Internal Audit Team",
        "type": "internal"
    },
    "auto_collect_evidence": true
}

Response:
{
    "assessment_id": "assess_soc2_2024_001",
    "framework": "soc2",
    "status": "in_progress",
    "controls_total": 50,
    "controls_evaluated": 0,
    "started_at": "2025-01-16T10:30:00Z",
    "estimated_completion": "2025-01-16T12:30:00Z"
}
```

### Get Assessment Results

```http
GET /api/v1/compliance/assessments/assess_soc2_2024_001
X-API-Key: your_api_key

Response:
{
    "assessment_id": "assess_soc2_2024_001",
    "framework": "soc2",
    "status": "completed",
    "assessment_date": "2025-01-16T12:30:00Z",
    "summary": {
        "controls_total": 50,
        "controls_passing": 45,
        "controls_failing": 3,
        "controls_partial": 2,
        "pass_rate": 0.90,
        "compliance_status": "MOSTLY_COMPLIANT"
    },
    "findings_by_category": {
        "security": {"passing": 20, "failing": 1, "partial": 1},
        "availability": {"passing": 15, "failing": 1, "partial": 0},
        "confidentiality": {"passing": 10, "failing": 1, "partial": 1}
    },
    "gaps": [
        {
            "gap_id": "gap_001",
            "control_id": "CC6.1",
            "control_name": "Logical Access Controls",
            "gap_description": "Multi-factor authentication not enforced for all administrative access",
            "risk_level": "high",
            "remediation_recommendation": "Enable MFA for all admin accounts in IAM provider",
            "estimated_effort": "1-2 days",
            "due_date": "2025-02-16T00:00:00Z"
        }
    ],
    "evidence_collected": 156,
    "report_url": "/api/v1/compliance/reports/assess_soc2_2024_001"
}
```

### Generate Report

```http
POST /api/v1/compliance/reports/generate
Content-Type: application/json
X-API-Key: your_api_key

{
    "assessment_id": "assess_soc2_2024_001",
    "report_type": "full",
    "format": "pdf",
    "include_sections": [
        "executive_summary",
        "control_findings",
        "evidence_summary",
        "gap_analysis",
        "remediation_plan"
    ],
    "branding": {
        "company_name": "Acme Corporation",
        "logo_url": "https://example.com/logo.png"
    }
}

Response:
{
    "report_id": "report_soc2_2024_001",
    "status": "generating",
    "estimated_completion": "2025-01-16T12:35:00Z",
    "download_url": "/api/v1/compliance/reports/report_soc2_2024_001/download"
}
```

### List Gaps

```http
GET /api/v1/compliance/gaps?framework=soc2&status=open&risk_level=high
X-API-Key: your_api_key

Response:
{
    "total": 5,
    "gaps": [
        {
            "gap_id": "gap_001",
            "framework": "soc2",
            "control_id": "CC6.1",
            "control_name": "Logical Access Controls",
            "gap_description": "MFA not enforced",
            "risk_level": "high",
            "status": "open",
            "owner": "security-team@acme.com",
            "due_date": "2025-02-16T00:00:00Z",
            "days_until_due": 31
        }
    ]
}
```

### Submit Evidence

```http
POST /api/v1/compliance/controls/CC6.1/evidence
Content-Type: multipart/form-data
X-API-Key: your_api_key

{
    "evidence_type": "screenshot",
    "description": "MFA configuration in Okta admin console",
    "file": <binary_file>,
    "collection_date": "2025-01-16",
    "collected_by": "john.doe@acme.com"
}

Response:
{
    "evidence_id": "evid_abc123",
    "control_id": "CC6.1",
    "evidence_type": "screenshot",
    "file_name": "okta_mfa_config.png",
    "file_size": 245678,
    "file_hash": "sha256:abc123...",
    "uploaded_at": "2025-01-16T10:30:00Z",
    "status": "accepted"
}
```

---

## Framework-Specific Details

### SOC 2 Type II

**Trust Services Criteria**:
| Category | Controls | Key Requirements |
|----------|----------|-----------------|
| **Security (CC)** | 20+ | Access controls, encryption, monitoring |
| **Availability (A)** | 5+ | Uptime, disaster recovery, capacity |
| **Processing Integrity (PI)** | 5+ | Data accuracy, completeness |
| **Confidentiality (C)** | 5+ | Data classification, protection |
| **Privacy (P)** | 10+ | PII handling, consent, retention |

### GDPR

**Key Articles**:
| Article | Description | Automation |
|---------|-------------|-----------|
| Art. 5 | Data processing principles | Automated checks |
| Art. 6 | Lawfulness of processing | Manual review |
| Art. 7 | Conditions for consent | Consent tracking |
| Art. 17 | Right to erasure | Automated workflows |
| Art. 32 | Security of processing | Technical controls |
| Art. 33 | Breach notification | Automated alerting |

### PCI-DSS 4.0

**Requirements**:
| Requirement | Description | Controls |
|-------------|-------------|----------|
| Req 1 | Network security | Firewall, segmentation |
| Req 3 | Protect stored data | Encryption, key management |
| Req 4 | Encrypt transmission | TLS, quantum-safe |
| Req 8 | Identity & access | MFA, password policy |
| Req 10 | Logging & monitoring | SIEM, audit trails |
| Req 11 | Testing | Vulnerability scans, pentests |

### HIPAA

**Safeguards**:
| Category | Safeguards | Examples |
|----------|-----------|----------|
| **Administrative** | 9 | Policies, training, risk assessment |
| **Physical** | 4 | Facility access, workstation security |
| **Technical** | 5 | Access control, audit, encryption |

### NIST Cybersecurity Framework

**Functions**:
| Function | Categories | Subcategories |
|----------|-----------|---------------|
| **Identify** | 6 | 29 |
| **Protect** | 6 | 39 |
| **Detect** | 3 | 18 |
| **Respond** | 5 | 16 |
| **Recover** | 3 | 6 |

---

## Configuration

```yaml
compliance:
  enabled: true

  frameworks:
    soc2:
      enabled: true
      trust_services_criteria:
        - security
        - availability
        - confidentiality
      assessment_frequency: quarterly
      auditor_email: auditor@auditfirm.com

    gdpr:
      enabled: true
      data_retention_days: 365
      dpia_required: true
      dpo_email: dpo@company.com

    pci_dss:
      enabled: true
      version: "4.0"
      cardholder_data_handling: strict
      quarterly_scans: true
      aoc_required: true

    hipaa:
      enabled: true
      phi_encryption: required
      audit_retention_years: 7
      baa_tracking: true

    nist_csf:
      enabled: true
      assessment_frequency: monthly
      target_tier: 3
      risk_tolerance: low

    iso27001:
      enabled: true
      certification_body: "BSI"
      surveillance_audit_frequency: annual

    nerc_cip:
      enabled: true
      asset_classification: required
      cyber_security_plan: required

  assessment:
    auto_schedule: true
    max_concurrent: 3
    timeout_seconds: 300
    auto_evidence_collection: true
    evidence_retention_years: 7

  reporting:
    formats:
      - pdf
      - json
      - csv
    templates_path: /etc/qbitel/report-templates
    branding:
      enabled: true
      logo_path: /etc/qbitel/logo.png

  notifications:
    gap_created:
      channels: [email, slack]
      recipients: [compliance-team@company.com]
    assessment_complete:
      channels: [email]
      recipients: [ciso@company.com, compliance-team@company.com]
    due_date_approaching:
      days_before: [30, 14, 7, 1]
      channels: [email, slack]
```

---

## Monitoring & Observability

### Prometheus Metrics

```
# Assessment metrics
qbitel_compliance_assessments_total{framework, status}
qbitel_compliance_assessment_duration_seconds{framework}
qbitel_compliance_controls_evaluated_total{framework}
qbitel_compliance_controls_passing_total{framework}
qbitel_compliance_controls_failing_total{framework}

# Compliance scores
qbitel_compliance_score{framework}
qbitel_compliance_pass_rate{framework}

# Gap metrics
qbitel_compliance_gaps_total{framework, risk_level, status}
qbitel_compliance_gaps_overdue_total{framework}
qbitel_compliance_gap_age_days{framework}

# Evidence metrics
qbitel_compliance_evidence_collected_total{framework, type}
qbitel_compliance_evidence_age_days{framework}

# Report metrics
qbitel_compliance_reports_generated_total{framework, format}
qbitel_compliance_report_generation_time_seconds
```

### Dashboard Example

```promql
# Overall compliance score
avg(qbitel_compliance_score) by (framework)

# Gap remediation velocity
rate(qbitel_compliance_gaps_total{status="closed"}[30d])

# Overdue gaps trend
qbitel_compliance_gaps_overdue_total
```

---

## Integration Examples

### Python SDK

```python
from qbitel.compliance import ComplianceClient

client = ComplianceClient(api_key="your_api_key")

# Run SOC 2 assessment
assessment = client.create_assessment(
    framework="soc2",
    scope={
        "trust_services_criteria": ["security", "availability"],
        "systems": ["payment-system"]
    }
)

# Wait for completion
assessment.wait_for_completion()

# Get results
print(f"Pass rate: {assessment.pass_rate:.1%}")
print(f"Gaps found: {len(assessment.gaps)}")

# Generate report
report = client.generate_report(
    assessment_id=assessment.id,
    format="pdf"
)

# Download report
report.download("/path/to/soc2_report.pdf")

# List open gaps
gaps = client.list_gaps(
    framework="soc2",
    status="open",
    risk_level="high"
)

for gap in gaps:
    print(f"{gap.control_id}: {gap.gap_description}")
```

---

## Conclusion

Enterprise Compliance transforms compliance management from periodic, manual efforts to continuous, automated assurance. With support for 9 frameworks, automated evidence collection, and audit-ready reporting, organizations can maintain compliance with significantly reduced effort while improving their security posture.
