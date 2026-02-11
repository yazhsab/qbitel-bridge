# QBITEL — Critical Infrastructure & SCADA/ICS

> Zero-Downtime Quantum Security for the Systems That Keep the Lights On

---

## The Challenge

Power grids, water treatment plants, oil pipelines, and manufacturing facilities run on Industrial Control Systems (ICS) designed 20–40 years ago with zero security. These systems:

- **Cannot be patched** — firmware updates risk safety-critical failures
- **Cannot be taken offline** — downtime means blackouts, contaminated water, or production halts
- **Use proprietary protocols** — undocumented, unencrypted, unauthenticated
- **Are nation-state targets** — Colonial Pipeline, Ukraine grid attacks, Oldsmar water treatment

Traditional IT security doesn't work here. Endpoint agents can't run on PLCs. Firewalls don't understand Modbus. And a false positive that shuts down a turbine can cost millions — or lives.

---

## How QBITEL Solves It

### 1. Passive Protocol Discovery — Learn Without Touching

QBITEL connects via network tap (passive, read-only) and learns every industrial protocol on the network:

| Protocol | Application |
|---|---|
| Modbus TCP/RTU | PLCs, sensors, actuators |
| DNP3 | Power grid SCADA, water/wastewater |
| IEC 61850 | Substation automation, protection relays |
| OPC UA | Manufacturing, process control |
| BACnet | Building management systems |
| EtherNet/IP | Industrial automation |
| PROFINET | Factory floor communications |

**Zero packets injected. Zero processes disrupted. Zero risk.**

### 2. Deterministic PQC — Safety-Critical Timing Guaranteed

Industrial control demands deterministic behavior. QBITEL's PQC implementation guarantees:

| Requirement | QBITEL Guarantee |
|---|---|
| Crypto overhead | <1ms per operation |
| Jitter | <100μs (IEC 61508 compliant) |
| Safety integrity | SIL 3/4 compatible |
| Availability | 99.999% (five nines) |
| Failsafe mode | Graceful degradation, never hard-stop |

PLC authenticator validates every command — a compromised HMI cannot send unauthorized control signals.

### 3. Zero-Touch for OT Environments

The autonomous engine understands operational technology context:

- **Unauthorized PLC command detected** → Block command, alert OT team, log for forensics (never shut down the PLC)
- **Anomalous sensor readings** → Cross-validate with physics model, flag if inconsistent
- **New device on OT network** → Auto-discover, apply micro-segmentation policy
- **Firmware vulnerability published** → Virtual patching applied at network layer

**Critical safety rule**: QBITEL never takes actions that could affect Safety Instrumented Systems (SIS). All SIS-adjacent decisions escalate to human operators.

### 4. Air-Gapped Deployment

Many OT environments are air-gapped by design. QBITEL operates fully on-premise:

- **On-premise LLM**: Ollama with Llama 3.2 (70B) — no cloud calls
- **Local threat intelligence**: Updated via secure media transfer
- **No internet dependency**: Full autonomous operation
- **FIPS 140-3 compliant**: Cryptographic module validation

---

## Real-World Scenarios

### Scenario A: Power Grid Substation Protection
A utility operates 500 substations running IEC 61850 and DNP3. QBITEL discovers all protocols passively, applies PQC authentication to every GOOSE message and DNP3 command, and monitors for unauthorized relay operations — all with <1ms overhead. NERC CIP evidence generated automatically.

### Scenario B: Water Treatment Plant Security
A water facility's SCADA system uses Modbus RTU over serial links. QBITEL's protocol bridge adds quantum-safe authentication to every pump/valve command without replacing any PLCs. The Oldsmar-style attack (remote chemical dosing change) would be detected and blocked in <1 second.

### Scenario C: Manufacturing Floor Protection
A factory runs 2,000 PLCs on EtherNet/IP and PROFINET. QBITEL creates a quantum-safe overlay network, validates every control command against physics models, and provides OT-specific incident response — understanding that shutting down a furnace mid-cycle causes $2M in damage.

---

## Compliance Automation

| Framework | Capability |
|---|---|
| NERC CIP | Continuous monitoring, automated evidence for CIP-002 through CIP-014 |
| IEC 62443 | Zone/conduit security, SL-T assessment |
| NIST SP 800-82 | ICS security control mapping |
| NIS2 Directive | EU critical infrastructure compliance |
| TSA Pipeline Security | Pipeline cybersecurity mandates |

---

## Business Impact

| Metric | Before QBITEL | After QBITEL |
|---|---|---|
| Protocol visibility | ~40% (documented only) | 100% (AI-discovered) |
| Downtime for security deployment | 4–8 hour maintenance windows | Zero downtime |
| PLC command authentication | None | Every command validated |
| Incident response | 30+ minutes (manual) | <10 seconds (autonomous) |
| NERC CIP audit preparation | 6–12 weeks | <10 minutes |
| Quantum readiness | None | NIST Level 5 |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│            QBITEL (Air-Gapped)                │
├──────────┬──────────┬──────────┬────────────────┤
│ Protocol │ Quantum  │ Agentic  │  NERC CIP /    │
│ Discovery│ Crypto   │ AI Engine│  IEC 62443     │
├──────────┴──────────┴──────────┴────────────────┤
│     On-Premise LLM (Ollama / vLLM)              │
├──────────┬──────────┬──────────────────────┐     │
│ Purdue   │ Purdue   │ Purdue Level 0-1    │     │
│ Level 3-4│ Level 2  │ (Field Devices)     │     │
│ (IT/DMZ) │ (HMI/   │                     │     │
│          │  SCADA)  │ PLCs │ RTUs │ IEDs  │     │
└──────────┴──────────┴──────────────────────┘     │
                                                    │
│  ╔══════════════════════════════════════════╗     │
│  ║  SAFETY: SIS systems always escalate    ║     │
│  ║  to human operators — never auto-act    ║     │
│  ╚══════════════════════════════════════════╝     │
└─────────────────────────────────────────────────┘
```

---

## Getting Started

1. **Passive Tap** — Network mirror port, zero disruption to operations
2. **Protocol Map** — AI discovers every device, protocol, and command pattern
3. **Threat Model** — Quantum vulnerability assessment for all OT communications
4. **Protection** — PQC authentication and encryption at network layer
5. **Compliance** — NERC CIP / IEC 62443 evidence generation activated

---

*QBITEL — Industrial security that understands the difference between IT and OT.*
