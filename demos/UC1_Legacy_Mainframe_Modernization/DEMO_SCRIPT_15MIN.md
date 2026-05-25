# QBITEL Bridge - 15-Minute Mainframe Modernization Demo

## Presenter Guide & Script

**Audience:** Enterprise decision-makers, CTOs, CISOs, VP Engineering
**Duration:** 15 minutes (12 min presentation + 3 min Q&A buffer)
**Setup:** Run `python run_demo.py --server` → open `http://localhost:8001/presentation`

---

## PRE-DEMO CHECKLIST

- [ ] Terminal open with `python run_demo.py --server` running
- [ ] Browser at `http://localhost:8001/presentation`
- [ ] Screen resolution 1920x1080 or higher
- [ ] Browser in full-screen mode (F11)
- [ ] Notes app open with talking points (this document)

---

## DEMO FLOW

### ACT 1: THE PROBLEM (0:00 - 2:30)

**[SLIDE: Opening - auto-shown on load]**

**Talking Points:**
> "Imagine you're running a $200 billion daily wire transfer operation. Your core banking system — 2.5 million lines of COBOL, written in 1988 — processes 15 million transactions a day at 99.97% uptime."
>
> "You can't touch it. You can't rewrite it. But now you have three converging crises:"
>
> **1. Legacy Crisis:** The developers who wrote this code retired 15 years ago. There's no documentation. Reverse engineering costs $2-10M and takes 6-12 months per system.
>
> **2. Quantum Threat:** Nation-states are harvesting your encrypted SWIFT transfers TODAY, waiting for quantum computers to decrypt them. RSA and ECC will break in 5-10 years.
>
> **3. Speed Gap:** Your SOC team takes 65 minutes to respond to incidents. Machine-speed attacks happen in seconds.

**[Click "Next" to advance]**

---

### ACT 2: THE SOLUTION (2:30 - 4:30)

**[SLIDE: QBITEL Bridge Architecture]**

**Talking Points:**
> "QBITEL Bridge solves all three problems simultaneously — without replacing your mainframes."
>
> "Our platform has a 4-layer architecture:"
> - **Rust Data Plane** — Wire-speed PQC encryption, <1ms overhead, NIST Level 5
> - **Python AI Engine** — Protocol discovery, COBOL analysis, multi-agent orchestration
> - **Go Control Plane** — Policy enforcement, service mesh, secrets management
> - **React Dashboard** — Real-time monitoring, protocol copilot, marketplace
>
> "The key insight: we work at the NETWORK layer. Zero code changes to your COBOL. Zero downtime. We wrap your legacy communications in quantum-safe encryption while simultaneously learning how your protocols work."

**[Click "Start Live Demo" to begin interactive portion]**

---

### ACT 3: LIVE DEMO (4:30 - 12:30)

#### Step 1: Legacy System Discovery (4:30 - 6:00)

**[Click "Discover Systems"]**

**Talking Points:**
> "First, QBITEL discovers what you have. In a real deployment, our AI agents passively tap your network and identify every legacy system in 2-4 hours — versus 6-12 months of manual reverse engineering."
>
> "Here we see three mainframe systems discovered:"
> - **Core Banking**: 2.5M lines COBOL, 38 years old, processing 15M transactions/day
> - **Customer Master**: 850K lines, managing 45M customer records
> - **Batch Processing**: 1.2M lines, nightly processing of 120M records
>
> "Total: 4.5 million lines of undocumented COBOL running your most critical operations. This is the reality for 92% of the top 100 global banks."

**[Click "Next Step"]**

---

#### Step 2: AI-Powered COBOL Analysis (6:00 - 8:00)

**[Click "Analyze COBOL"]**

**Talking Points:**
> "Now our AI engine deep-analyzes the COBOL source. This is CUSTMAST.cbl — a Customer Master File program written in 1985."
>
> **Point out key findings:**
> - "Complexity score calculated from control flow, nesting depth, and branching"
> - "Working storage analysis — 60+ variables mapping the customer data structure"
>
> **Legacy Patterns Detected:**
> - "CRITICAL: GOTO statements — these make the code nearly impossible to maintain"
> - "File I/O patterns — direct VSAM access that needs database abstraction"
> - "Monolithic routines — single 400-line procedures"
>
> **Modernization Opportunities:**
> - "File I/O → Database ORM migration"
> - "Batch processing → Async event-driven architecture"
> - "Sequential access → REST API with pagination"
>
> "This analysis that would take a senior consultant weeks happens in seconds."

**[Click "Next Step"]**

---

#### Step 3: Protocol Reverse Engineering (8:00 - 9:30)

**[Click "Analyze Protocol"]**

**Talking Points:**
> "This is where it gets really interesting. We're looking at raw mainframe data — EBCDIC-encoded customer records straight from the wire."
>
> "Our AI identifies:"
> - "The encoding — EBCDIC, not ASCII. This is native IBM mainframe format"
> - "Field boundaries — where one data element ends and another begins"
> - "Data types — numeric fields, alphanumeric strings, packed decimal amounts"
>
> "Traditional approach: hire a consultant who understands EBCDIC, spend months with a hex editor. Our approach: 2-4 hours, fully automated."
>
> **Recommendations section:**
> "And notice the actionable recommendations — convert EBCDIC to UTF-8, replace packed decimal with standard numeric types, add field-level encryption. Each recommendation maps to a specific modernization action."

**[Click "Next Step"]**

---

#### Step 4: Automatic Code Generation (9:30 - 11:00)

**[Click "Generate Code"]**

**Talking Points:**
> "Now QBITEL generates production-ready modern code from the COBOL analysis."
>
> **Show each tab:**
>
> **Python Models tab:**
> "Python dataclasses that exactly mirror the COBOL record structures. Every PICTURE clause mapped to proper Python types. This is your new data layer."
>
> **FastAPI Endpoints tab:**
> "RESTful API endpoints auto-generated. CRUD operations, proper HTTP methods, Pydantic validation — ready to replace CICS terminal interactions."
>
> **SQL Schema tab:**
> "Database schema derived from the COBOL file definitions. VSAM files become proper relational tables with indexes and constraints."
>
> "From a 1985 COBOL program to a modern Python microservice — automatically. The generated code maintains 100% data fidelity with the original COBOL structures."

**[Click "Next Step"]**

---

#### Step 5: Modernization Roadmap (11:00 - 12:30)

**[Select "Refactor" approach, click "Create Plan"]**

**Talking Points:**
> "Finally, QBITEL generates a complete modernization plan."
>
> **Key metrics:**
> - "Risk level assessed based on system criticality and complexity"
> - "Effort estimated in person-days — calibrated against real-world projects"
>
> **Walk through phases:**
> - "**Assessment** — 2 weeks. Full system inventory and dependency mapping"
> - "**Design** — 4 weeks. Target architecture, API contracts, data migration strategy"
> - "**Development** — 8-12 weeks. Incremental COBOL-to-Python transformation"
> - "**Testing** — 4 weeks. Parallel run with mainframe, data reconciliation"
> - "**Deployment** — 2 weeks. Canary rollout with automatic rollback"
>
> "Each phase has specific deliverables, risk mitigations, and go/no-go criteria. This is an auditable, board-ready modernization plan."

---

### ACT 4: BUSINESS VALUE & CLOSE (12:30 - 15:00)

**[SLIDE: Impact Metrics - auto-advances to final slide]**

**Talking Points:**

> "Let me put this in business terms:"

| Metric | Before QBITEL | With QBITEL |
|--------|---------------|-------------|
| Protocol Discovery | 6-12 months, $2-10M | **2-4 hours** |
| Quantum Readiness | None | **NIST Level 5** |
| Security Response | 65 minutes | **<1 second** |
| Compliance Reports | 2-4 weeks manual | **<10 minutes** |
| Integration Cost | $5-50M per system | **$200K-500K** |
| Annual Security Cost | $10-50/event | **<$0.01/event** |

> "And here's what makes us unique:"
> - "100% open source — Apache 2.0, same license as Kubernetes"
> - "Air-gapped capable — runs entirely on-premise with local LLM"
> - "Zero code changes — quantum security at the network layer"
> - "9 compliance frameworks automated — PCI-DSS, DORA, SOX, HIPAA, NERC CIP"
> - "78% autonomous security response — under 1 second decision time"

**[SLIDE: Call to Action]**

> "We're offering a 2-week proof of concept. We connect to your test environment, discover your protocols, analyze your COBOL, and deliver a complete modernization assessment."
>
> "Questions?"

---

## Q&A PREPARATION

### Likely Questions & Answers

**Q: How does this work with our existing security tools?**
> "QBITEL operates at the network layer and integrates with your existing SIEM (Splunk, QRadar), SOAR platforms, and IAM systems. We enhance, not replace."

**Q: What about performance overhead?**
> "Our Rust data plane adds <1ms of encryption latency. We process 10,000+ TPS for banking workloads at <50ms end-to-end. The PQC algorithms are hardware-accelerated."

**Q: Is post-quantum crypto proven?**
> "We implement NIST-standardized algorithms: ML-KEM (FIPS 203), ML-DSA (FIPS 204), SLH-DSA (FIPS 205). These completed 6+ years of public review and were standardized in 2024."

**Q: What if we don't want to modernize — just protect?**
> "That's exactly what our 'Protect' phase does. No code changes, no modernization required. Just quantum-safe encryption wrapping your existing mainframe communications. Modernization is optional."

**Q: How long does deployment take?**
> "Typical deployment is 8-16 weeks for full production. But the initial discovery and protection layer can be operational in as little as 2-4 hours for a single system."

**Q: How do you handle air-gapped environments?**
> "Our AI engine runs on Ollama with local LLM models — no cloud dependency. The entire platform runs on-premise. Threat intelligence updates via encrypted removable media."

**Q: What domains do you support beyond banking?**
> "Healthcare (HL7/FHIR/DICOM), Critical Infrastructure (Modbus/DNP3/IEC 61850), Automotive (V2X), Aviation (ADS-B/ACARS), Telecom (SS7/Diameter/5G), and BPO/Call Centers."

---

## TECHNICAL SETUP

```bash
# Install dependencies
cd demos/UC1_Legacy_Mainframe_Modernization
pip install -r requirements.txt

# Start the demo server
python run_demo.py --server

# Open presentation mode
open http://localhost:8001/presentation
```

If server fails to start, check:
- Python 3.10+ installed
- Port 8001 available
- FastAPI and uvicorn installed (`pip install fastapi uvicorn`)
