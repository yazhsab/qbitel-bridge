# QBITEL BRIDGE — CRITICAL INFRASTRUCTURE & ICS/SCADA
## Quantum-Safe Security for Operational Technology Networks

**Confidential — For Authorized Recipients Only | © 2026 QBITEL. All Rights Reserved.**

---

## EXECUTIVE SUMMARY

Operational technology networks — the systems controlling power grids, water treatment plants, pipelines, and manufacturing facilities — have become the primary target for nation-state cyberattacks. The Colonial Pipeline attack cost $4.4 billion and disrupted fuel supply across the US East Coast. Ukrainian power grid attacks left millions without electricity in winter. Volt Typhoon has pre-positioned in US critical infrastructure for potential wartime disruption. These are not hypothetical threats: 13 attacks on critical infrastructure occur globally every single day.

Legacy OT protocols were never designed for security. Modbus, invented in 1979, has zero authentication. DNP3 and IEC 61850 similarly lack cryptographic protection. Any device connected to an OT network can send arbitrary commands to PLCs controlling chemical dosing, substation breakers, or pipeline compressors. The industry has relied on physical isolation — air gaps and perimeter firewalls — but nation-states and insider threats routinely bypass these controls through USB drives, supply chain compromise, and vendor remote access connections.

QBITEL Bridge addresses the root cause: the absence of cryptographic authentication and integrity at the OT protocol level. Our approach starts passive — a read-only network tap that discovers all devices and protocols without any operational risk. We then layer deterministic post-quantum cryptography with less than 1ms overhead and less than 100 microseconds jitter, ensuring that real-time control loops and safety-instrumented systems are never compromised. NERC CIP compliance evidence is generated automatically. Safety systems always escalate to human operators — never autonomous action on safety-critical infrastructure.

---

## THE OT/ICS SECURITY IMPERATIVE

### Threat 1: Legacy Protocol Exposure (Modbus Has Zero Authentication)

Modbus was designed in 1979 with no security whatsoever. There was no concept of authentication, integrity checking, or encryption — the protocol assumed physical security of the network was sufficient. DNP3, designed in the 1990s for electric and water utilities, similarly lacks built-in authentication in its base specification. IEC 61850 GOOSE messages travel as multicast with no sender verification. OPC UA has security modes, but many deployments run in "None" security mode for compatibility with legacy clients.

The consequence is stark: any device with access to the OT network segment can issue arbitrary commands to PLCs and RTUs. A rogue device can command a substation breaker to open, adjust chemical dosing in a water treatment plant, or alter setpoints on a natural gas pipeline compressor station. Nation-states have demonstrated this capability repeatedly. The Ukraine BlackEnergy attack used malicious firmware to brick substation relay devices. TRITON/TRISIS specifically targeted Schneider Electric Triconex Safety Instrumented Systems, attempting to disable the last line of protection against physical catastrophe.

QBITEL Bridge addresses this at the network layer — without requiring firmware changes to any OT device. Every PLC WRITE command is cryptographically signed using ML-DSA-65 (NIST FIPS 204). RTUs validate signatures before executing commands. Replayed commands are rejected via timestamp and nonce. The protocol remains Modbus or DNP3 at the wire level — zero changes to OT vendor equipment required.

### Threat 2: The Safety-Security Tension

Critical infrastructure operates 24 hours a day, 365 days a year. A power utility cannot take the grid offline to patch firmware. A water treatment plant cannot halt chlorination while deploying a security agent. A pipeline operator cannot interrupt flow while reconfiguring SCADA servers. This creates an impossible constraint for traditional IT security approaches: the operational risk of deploying the security tool may exceed the risk of the threat itself.

Traditional endpoint agents require firmware installation on PLCs — voiding vendor warranties and potentially destabilizing certified safety systems. Active network scanners inject traffic that can overload legacy serial links or cause timing violations in real-time control loops. Firewall rules misconfigured in an IT context can block legitimate SCADA polling traffic, causing loss of visibility or false emergency shutdowns. The 2021 Oldsmar water treatment attack was discovered by an operator watching his screen — not by any automated system — because the SCADA interface showed the correct state even as commands were being executed.

QBITEL Bridge's deployment philosophy is passive-first. The network tap is read-only — it physically cannot inject traffic. Protocol discovery happens without any active probing. PLC command authentication is rolled out zone by zone, with manual activation controlled by the operations team. Safety-Instrumented Systems are explicitly excluded from any active protection — they receive passive monitoring only. No autonomous action is ever taken on safety-critical systems. Every anomaly on a SIS boundary is immediately escalated to human operators.

### Threat 3: Quantum Threat to Long-Lifecycle OT Assets

OT equipment has operational lifespans of 20 to 40 years. A PLC installed in a nuclear facility today will still be operating in 2060 or 2070. Substation relays installed in the 2000s are still running. Historians holding decades of process data use classical encryption that will be broken by cryptographically relevant quantum computers, now projected by CISA and NSA to arrive within 10 to 20 years.

The "harvest now, decrypt later" attack strategy means that nation-states are actively collecting encrypted OT communications today, storing them for decryption when quantum computers become available. Grid topology data, process setpoints, historian data, and engineering credentials encrypted with RSA-2048 or ECC-256 today will be readable in the 2030s. NIST finalized post-quantum cryptography standards in 2024 — ML-KEM-768 and ML-DSA-65 — and CISA has mandated a migration timeline. OT operators who start now will be compliant before the deadline. Those who wait will face both quantum risk and compliance penalties simultaneously.

QBITEL Bridge deploys NIST-standardized PQC algorithms from day one: ML-KEM-768 for key encapsulation, ML-DSA-65 for digital signatures, and AES-256-GCM for symmetric encryption. The PQC overhead is less than 1 millisecond per transaction — verified against IEC 61508 real-time requirements for SIL 3 and SIL 4 systems. Long-lifecycle OT assets are protected against both classical and quantum threats without any modification to the OT equipment itself.

---

## QBITEL BRIDGE FOR CRITICAL INFRASTRUCTURE

---

## CAPABILITY 1: PASSIVE OT PROTOCOL DISCOVERY (ZERO DISRUPTION)

Traditional OT asset inventory is a 6-to-12-month manual process involving site visits, engineering documentation archaeology, and careful hand-drawn network diagrams. The result is a snapshot that is outdated the moment it is produced. Unauthorized devices, rogue engineer laptops, and undocumented legacy connections remain invisible until an incident reveals them.

QBITEL Bridge deploys a passive network tap — a physical device that receives a copy of all network traffic without injecting any packets. The tap is read-only by electrical design. It cannot transmit. QBITEL then decodes every OT protocol in flight — Modbus TCP and RTU, DNP3, IEC 61850 GOOSE and Sampled Values, OPC UA, BACnet, EtherNet/IP, and PROFINET — building a live asset inventory and communication map in 2 to 4 hours.

The discovery output includes every PLC, RTU, HMI, historian, and engineering workstation on the OT network, mapped to its protocol, communication partners, and function. Unauthorized devices are immediately flagged. Vulnerable protocol configurations — Modbus without authentication, IEC 61850 GOOSE without HMAC — are identified and prioritized. The asset inventory is structured to map directly to NERC CIP BES Cyber System classification, accelerating CIP-002 compliance from months to days.

**Key Specifications:**
- Method: Passive network tap only — no active probing, no traffic injection, no ARP scans
- Coverage: Modbus TCP/RTU, DNP3, IEC 61850 GOOSE/SV, OPC UA, BACnet, EtherNet/IP, PROFINET
- Time to full inventory: 2-4 hours vs 6-12 months manual
- Output: Asset inventory, protocol communication map, vulnerability assessment, NERC CIP BES classification draft
- Risk to operations: Zero — read-only by design

---

## CAPABILITY 2: DETERMINISTIC PQC FOR REAL-TIME SCADA

The central objection to cryptographic security in OT environments is timing. Real-time control systems operate on deterministic schedules — a SCADA poll must complete within a fixed window, a protection relay must operate within milliseconds of a fault, a safety interlock must respond within microseconds. Classical cryptographic overhead is often acceptable for IT systems but intolerable for OT control loops.

QBITEL Bridge's PQC engine is optimized for deterministic performance on OT hardware. The maximum overhead for ML-KEM-768 key encapsulation is less than 1 millisecond. The jitter — the variation in processing time — is less than 100 microseconds. These parameters were validated against IEC 61508 requirements for Safety Integrity Level 3 and SIL 4 systems. The PQC processing path is deterministic: no garbage collection pauses, no dynamic memory allocation, no OS scheduling interruptions.

If the PQC subsystem experiences any failure — hardware fault, key management issue, network partition — the system degrades gracefully. Traffic passes through with an alert to the SOC. The control loop never stalls. This fail-open posture for non-safety-critical paths ensures that the security layer never introduces more operational risk than the threat it protects against.

**Key Specifications:**
- PQC overhead: <1ms per transaction (ML-KEM-768 key encapsulation)
- Timing jitter: <100μs — verified IEC 61508 SIL 3/4 compliant
- Algorithms: ML-KEM-768 (NIST FIPS 203), ML-DSA-65 (NIST FIPS 204), AES-256-GCM
- Graceful degradation: Traffic passes on PQC failure with SOC alert
- Hardware: FIPS 140-3 Level 3 HSM for key storage

---

## CAPABILITY 3: PLC COMMAND AUTHENTICATION & INTEGRITY

PLC command injection is the most direct path to physical damage in an OT network. An attacker who can issue arbitrary WRITE commands to a PLC controlling a pipeline compressor can cause overpressure. An attacker commanding a substation breaker can cause a fault cascade. An attacker adjusting chemical dosing in a water treatment plant can sicken populations. Modbus, the protocol controlling the majority of the world's industrial equipment, has no mechanism to verify that a command came from a legitimate source.

QBITEL Bridge wraps PLC command traffic with ML-DSA-65 digital signatures. Each WRITE command from an HMI or SCADA server is signed before transmission. The QBITEL verification layer at the PLC network boundary validates the signature before the command reaches the PLC. An unsigned command, a command with an invalid signature, or a replayed command is blocked and immediately alerted to the SOC. The PLC itself never receives an unauthorized command.

HMI sessions are authenticated with ML-KEM-768 key establishment. Engineering workstation connections require certificate-based mutual authentication. Key material is stored in a FIPS 140-3 Level 3 Hardware Security Module. Air-gapped key ceremonies are supported for facilities that prohibit network-connected HSMs. The PLC firmware is never modified — the authentication layer operates entirely in the network path.

**Key Specifications:**
- Signing: ML-DSA-65 on every PLC WRITE command
- Verification: Network-layer — PLC receives only authenticated commands
- Replay protection: Timestamp + nonce — replayed commands rejected within 5ms window
- HMI sessions: ML-KEM-768 key establishment
- Key management: FIPS 140-3 Level 3 HSM, air-gapped key ceremony supported
- PLC modification: None required — zero changes to OT vendor equipment

---

## CAPABILITY 4: SAFETY-INSTRUMENTED SYSTEM PROTECTION

Safety-Instrumented Systems are the final barrier between equipment failure and physical catastrophe. An SIS controlling a high-pressure vessel shutdown, a fire and gas system, or an emergency depressurization valve is certified to IEC 61508 at SIL 3 or SIL 4. Any modification to that system — including the installation of security software — voids the safety certification and may violate regulatory requirements.

QBITEL Bridge never touches SIS systems. Safety system boundaries are formally documented during Phase 4 of deployment and explicitly configured in the QBITEL policy engine as passive-monitoring-only zones. No active protection — no command authentication, no traffic blocking, no policy enforcement — is applied to any SIS segment. QBITEL monitors SIS traffic passively, detecting anomalies and unauthorized communications, but all response actions require explicit human operator approval.

Any anomaly detected on a SIS boundary — an unexpected command, a new device, an unusual communication pattern — immediately escalates to the designated Safety Officer and Operations Manager. The escalation path bypasses automated response entirely. QBITEL's SIS protection posture is reviewed and signed off by the customer's Safety Officer during deployment. This review is a mandatory gate before any active protection is enabled elsewhere in the facility.

**Key Specifications:**
- SIS posture: Passive monitoring only — zero active protection
- SIL compliance: Compatible with SIL 3/4 certified systems
- Safety escalation: All SIS anomalies → immediate human operator notification
- Autonomous action: Never — SIS always requires human decision
- IEC 61508: No modification to safety-certified code or configuration
- Deployment gate: Safety Officer written sign-off required before Phase 5

---

## CAPABILITY 5: AIR-GAPPED SOVEREIGN DEPLOYMENT

OT security products that require cloud connectivity introduce a fundamental contradiction: the very communication path to the cloud security platform is a potential attack surface. Cloud-connected OT security sensors have been targeted in supply chain attacks. Cloud platforms go offline. Internet connectivity on OT networks creates regulatory challenges under NERC CIP CIP-005 Electronic Security Perimeter requirements.

QBITEL Bridge is designed from the ground up for fully air-gapped, sovereign deployment. The AI inference engine runs on local Ollama — no calls to external LLMs, no telemetry to cloud providers, no dependency on internet connectivity. Threat intelligence updates are delivered via removable media following a documented air-gap transfer procedure. The HSM is on-premise hardware — key material never leaves the facility. The entire QBITEL stack can operate indefinitely with zero internet connectivity.

For facilities with no external network connectivity requirements, QBITEL operates in a fully closed network environment. Updates, patches, and threat intelligence can be delivered on encrypted removable media with cryptographic chain of custody. This deployment model meets the most stringent requirements of nuclear facilities, classified government infrastructure, and defense industrial base environments.

**Key Specifications:**
- AI inference: Ollama on-premise — no external LLM calls
- Threat intelligence: Offline updates via encrypted removable media
- HSM: On-premise hardware — key material never leaves facility
- Internet dependency: None — fully sovereign deployment
- Data sovereignty: No data leaves the facility under any circumstances
- NERC CIP: Compatible with CIP-005 ESP air-gap requirements

---

## CAPABILITY 6: NERC CIP / IEC 62443 COMPLIANCE AUTOMATION

NERC CIP violations can cost up to $1 million per day per violation. A single audit finding — missing asset inventory, inadequate ESP boundary documentation, insufficient access control evidence — can trigger penalties that exceed the cost of comprehensive security deployment. The challenge is not knowing what to do; it is generating sufficient, consistent, audit-quality evidence that NERC CIP controls are operating continuously.

QBITEL Bridge maps its telemetry directly to NERC CIP requirements. CIP-002 BES Cyber System identification is populated from passive discovery output. CIP-005 Electronic Security Perimeter monitoring is continuous — every device communicating across zone boundaries is logged with full packet metadata. CIP-007 system security management evidence — patch status, port activity, failed authentication — is collected automatically. CIP-010 baseline monitoring detects any deviation from the documented configuration baseline within minutes.

Audit-ready evidence packages are generated in less than 10 minutes. The package includes time-stamped logs, control mapping to specific NERC CIP standard requirements, and attestation artifacts suitable for NERC CIP compliance auditors. IEC 62443 Security Level assessments are automated based on zone/conduit configuration and control implementation evidence. NIS2 incident reporting for EU critical infrastructure operators is automated with the required 24-hour and 72-hour notification formats.

**Key Specifications:**
- NERC CIP: CIP-002 through CIP-014 continuous monitoring and evidence collection
- Evidence generation: <10 minutes for complete audit package
- IEC 62443: Zone/conduit mapping, SL-T security level assessment
- NIS2: Automated incident reporting (24h and 72h formats)
- TSA Pipeline: Automated compliance evidence for pipeline security directives
- Audit format: Structured evidence packages mapped to specific requirement citations

---

## CAPABILITY 7: PHYSICS-AWARE ANOMALY DETECTION

Traditional anomaly detection in OT environments suffers from catastrophic false positive rates. An alert for every unusual network packet in a noisy industrial environment produces thousands of daily notifications that operators learn to ignore. The boy-who-cried-wolf effect is well documented in OT security operations: alert fatigue leads to the most important alerts being dismissed along with the noise.

QBITEL Bridge cross-validates network telemetry against physical process models. A temperature sensor reading that exceeds thermodynamic limits is flagged as a potential sensor spoof, not filed as a network anomaly. A pressure reading that violates fluid dynamics given measured flow rates triggers immediate investigation. A PLC command to open a valve that physics models predict would cause an overpressure condition is flagged before the command executes.

Physics-aware validation reduces false positives to less than 0.001% — a rate achievable only by grounding anomaly detection in the laws of physics that govern the process. This allows QBITEL to detect sophisticated attacks — like those seen in TRITON/TRISIS that generated valid-looking safety system commands — that would be invisible to pure network-based detection. The physics models are built from historian data during the passive discovery phase and continuously updated as process conditions change.

**Key Specifications:**
- Validation: Sensor readings cross-checked against process physics models
- False positive rate: <0.001% — physics constraints eliminate noise
- Detection scope: Command anomalies, sensor spoofing, rogue devices, timing attacks
- Model source: Built from historian data (OSIsoft PI, GE Proficy, Honeywell PHD)
- Response: Autonomous block (network-layer, non-SIS) or escalate (SIS, safety-critical)
- Update cycle: Physics models updated continuously from historian telemetry

---

## COMPLIANCE COVERAGE

| Framework | Coverage | Key Requirement |
|---|---|---|
| NERC CIP CIP-002 to CIP-014 | Continuous monitoring | ESP, BES Cyber System protection |
| IEC 62443 | Full SL-T assessment | Zone/conduit, security levels |
| NIST SP 800-82 | Control mapping | ICS security framework |
| NIS2 Directive | Incident reporting | EU critical infrastructure |
| TSA Pipeline Security | Mandate compliance | Pipeline cybersecurity directives |
| IEC 61508 | SIL compatibility | Functional safety (SIL 3/4) |
| IEC 62351 | Power systems | GOOSE, SV authentication |

---

## INTEGRATION ECOSYSTEM

**Historians:**
- OSIsoft PI System — native API integration for physics model data
- GE Proficy Historian — direct tag subscription
- Honeywell PHD — process data for anomaly baseline

**OT Security (Complementary):**
- Claroty — QBITEL adds PQC encryption to Claroty's detection
- Dragos — QBITEL adds command authentication to Dragos's threat detection
- Nozomi Networks — QBITEL extends visibility with active protection

**SCADA Vendors:**
- Siemens SIMATIC — native PROFINET and OPC UA integration
- GE iFIX — OPC UA data path integration
- Schneider Electric AVEVA — native support

**HSM Vendors:**
- Thales Luna Network HSM (FIPS 140-3 Level 3)
- Entrust nShield — supported
- AWS CloudHSM — optional (non-air-gapped deployments)

**SIEM:**
- Splunk Enterprise Security — native CEF log forwarding
- IBM QRadar — LEEF integration
- Microsoft Sentinel — Azure Monitor connector

---

## DEPLOYMENT TIMELINE

### Phase 1: Passive OT Discovery (Week 1-2)
Deploy network tap on target OT segment. No operational impact. Enumerate all protocols, devices, and communication patterns. Produce asset inventory and NERC CIP BES classification draft. Identify top-priority vulnerabilities.

### Phase 2: Zone & Conduit Mapping (Week 3-4)
Classify IEC 62443 zones (Safety, Control, Supervisory, Enterprise). Define conduit security requirements at each zone boundary. Map to NERC CIP ESP requirements. Identify high-risk conduits for priority treatment.

### Phase 3: PLC Command Authentication Rollout (Week 5-8)
Deploy ML-DSA-65 signing on Modbus masters (non-SIS first). Validate less than 1ms overhead per control loop. Roll out zone by zone following change window schedule. Enable DNP3, IEC 61850, and OPC UA protocol protection.

### Phase 4: NERC CIP Evidence & Compliance Validation (Week 9-12)
Map all QBITEL telemetry to NERC CIP requirements. Configure automated evidence collection. Generate and review first full compliance package. Conduct IEC 62443 SL-A assessment. External reviewer sign-off.

---

## PERFORMANCE SPECIFICATIONS

| Metric | Value |
|---|---|
| PQC overhead | <1ms per transaction |
| Timing jitter | <100μs (SIL 3/4 compliant) |
| System availability | 99.999% (five nines) |
| False positive rate | <0.001% (physics-aware) |
| Protocol discovery time | 2-4 hours (passive tap) |
| NERC CIP report generation | <10 minutes automated |
| Deployment model | Zero downtime, passive-first |
| Air-gap compatibility | Fully sovereign, no internet required |

---

## COMPETITIVE DIFFERENTIATION

### vs Claroty / Dragos / Nozomi (Detection-Only OT Security)
Claroty, Dragos, and Nozomi are detection-only platforms — they identify threats but cannot authenticate or encrypt OT protocol traffic. QBITEL adds the missing layer: post-quantum cryptographic authentication of every PLC command, IEC 62351 GOOSE/SV authentication, and automated NERC CIP compliance evidence. QBITEL is complementary to these platforms, not a replacement — detection plus protection is superior to detection alone.

### vs Perimeter Firewall Vendors (Fortinet FortiGate-Rugged, Cisco IOS XE)
Firewall vendors protect the perimeter between IT and OT networks. They cannot authenticate Modbus commands between a SCADA server and a PLC on the same OT segment. An insider threat, a compromised engineer laptop, or a rogue device inside the perimeter is invisible to firewall-only approaches. QBITEL secures OT protocols end-to-end, regardless of network topology.

### vs Manual Compliance Processes
Manual NERC CIP compliance requires dedicated staff, 6-to-12-month audit cycles, and produces static evidence that becomes stale immediately. QBITEL generates continuous, automated evidence in less than 10 minutes per audit package. The cost differential is dramatic: one compliance FTE saved covers multiple years of QBITEL deployment cost.

### vs Cloud-Based OT Security (Microsoft Defender for IoT)
Cloud-connected OT security products introduce NERC CIP CIP-005 ESP concerns for any OT data leaving the facility. Microsoft Defender for IoT requires Azure connectivity. QBITEL is fully air-gapped and sovereign — no data leaves the facility, no cloud dependency, no EPA or regulatory concerns about OT data transmission.

---

## CUSTOMER SCENARIOS

### Scenario A: Power Utility — Grid Substation Protection

A major US power utility operates 500 substations with IEC 61850 GOOSE messages controlling protection relays. GOOSE messages are unauthenticated multicast — any device on the substation LAN can inject protection commands. A nation-state attack on substation LANs could simultaneously open breakers across the transmission grid.

**QBITEL Approach:**
1. Passive discovery across all 500 substations — protocol map and asset inventory in 48 hours
2. IEC 62351 PQC authentication on all GOOSE and Sampled Values traffic
3. NERC CIP CIP-007 system security evidence automated across all substations
4. Zero downtime — passive tap, no changes to relay firmware

**Result:** NERC CIP CIP-007 compliant. IEC 61850 GOOSE traffic authenticated. Protection relay command injection impossible. All-substation deployment in 8 weeks.

### Scenario B: Water Treatment — SCADA Command Authentication

A metropolitan water authority operates a DNP3 SCADA system controlling chemical dosing at 12 water treatment plants. DNP3 lacks authentication — a compromised SCADA server or insider can adjust chlorine and fluoride dosing. The 2021 Oldsmar attack demonstrated this exact threat vector.

**QBITEL Approach:**
1. Passive discovery identifies all DNP3 masters and outstations
2. ML-DSA-65 command signing on all DNP3 WRITE operations
3. Physics-aware anomaly detection cross-validates dosing commands against flow rates
4. SIS systems (emergency shutdown) remain passive-monitoring only

**Result:** Command injection on chemical dosing impossible. Physics validation catches sensor spoofing. 4-week deployment, single scheduled change window.

### Scenario C: Pipeline Operator — TSA Mandate Compliance

A midstream pipeline operator faces TSA Pipeline Security Directive compliance deadlines. Their SCADA system uses Modbus TCP with no authentication. The Colonial Pipeline attack — a ransomware incident that cost $4.4 billion and triggered emergency fuel declarations — is the benchmark threat.

**QBITEL Approach:**
1. Passive Modbus discovery produces TSA-required asset inventory in 4 hours
2. PQC wrapping on Modbus TCP protects compressor station commands
3. TSA compliance evidence package generated in <10 minutes
4. Air-gapped deployment — no internet connectivity on pipeline SCADA required

**Result:** TSA Pipeline Security Directive compliant. Evidence package ready in 2 weeks. Total deployment cost a fraction of Colonial Pipeline incident cost of $4.4 billion.

---

## FREQUENTLY ASKED QUESTIONS

**Q: Our OT network is air-gapped. Why do we need additional security?**
A: Air-gapped networks are regularly compromised via USB drives (Stuxnet), supply chain malware pre-installed on vendor equipment, and authorized vendor remote access connections that create temporary network paths. Once inside the air gap, an attacker can issue arbitrary Modbus or DNP3 commands to any device. QBITEL protects the protocols themselves, not just the perimeter.

**Q: We can't afford any disruption to operations.**
A: QBITEL starts with a read-only passive tap. There is no operational risk during the discovery phase — the tap physically cannot inject traffic. Active protection phases are staged to your change window schedule, with manual activation by your operations team. You control the timeline.

**Q: Our OT vendor doesn't support third-party security tools.**
A: QBITEL operates at the network layer — no changes to PLC firmware, no software agents on OT devices, no modifications to vendor equipment of any kind. Vendor support is irrelevant because QBITEL never touches the OT devices themselves.

**Q: NERC CIP doesn't require encryption.**
A: NERC CIP requires protection of BES Cyber Systems including protection of communication links (CIP-005, CIP-007, CIP-011). Post-quantum cryptographic authentication of PLC commands directly supports CIP-007 and CIP-011 requirements. Additionally, FERC has signaled interest in stronger encryption requirements as the quantum timeline advances.

---

## NEXT STEPS

1. **Passive OT Discovery Assessment** — 2-week engagement, zero disruption, no operational risk
   - Deploy passive tap on target OT segment
   - Enumerate all protocols, devices, vulnerabilities
   - Produce asset inventory and NERC CIP BES classification

2. **NERC CIP Gap Analysis** — Map current controls against CIP-002 through CIP-014
   - Identify high-priority gaps in ESP and command authentication
   - Produce remediation roadmap with estimated effort and timeline

3. **Proof of Concept** — Single substation or plant segment
   - Deploy PLC command authentication on target segment
   - Validate <1ms overhead and zero timing violations
   - Demonstrate NERC CIP evidence generation

4. **Full Deployment Roadmap** — Phased rollout across all facilities
   - Zone-by-zone activation aligned to change windows
   - Compliance validation and audit-ready evidence package

---

## CONTACT

**Email:** enterprise@qbitel.com
**Website:** https://bridge.qbitel.com
**Account Team:** Contact your regional QBITEL representative

---

*QBITEL Bridge — Because the grid that powers a nation deserves quantum-safe protection.*

---

**Technical Specifications — Reference**

| OT Protocol | Security Gap | QBITEL Solution |
|---|---|---|
| Modbus TCP/RTU | Zero authentication | ML-DSA-65 command signing |
| DNP3 | Optional SAv5, rarely deployed | ML-DSA-65 + DNP3 SAv5 wrapping |
| IEC 61850 GOOSE | Unauthenticated multicast | IEC 62351 PQC authentication |
| IEC 61850 SV | No integrity protection | ML-DSA-65 SV signing |
| OPC UA | Often deployed in "None" mode | ML-KEM-768 session encryption |
| BACnet | No encryption standard | PQC session layer |
| EtherNet/IP | Minimal security options | Network-layer authentication |
| PROFINET | DCP unauthenticated | Command authentication layer |

**Deployment Architecture Options:**
- Option A: Inline with bypass — hardware failsafe, zero downtime
- Option B: Passive tap only (permanent) — monitoring without active protection
- Option C: Out-of-band enforcement — policy engine separate from OT path
- Option D: Hybrid — passive discovery, selective inline on high-risk segments

**Supported Hardware Platforms:**
- Standalone appliance (19" rack, 1U)
- DIN-rail industrial form factor (IEC 61850-3 compliant)
- Virtualized (VMware ESXi, Hyper-V)
- Air-gapped rack with integrated HSM

**© 2026 QBITEL Technologies. All Rights Reserved.**
**Confidential — For Authorized Recipients Only**
