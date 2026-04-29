# QBITEL BRIDGE — AUTOMOTIVE & CONNECTED VEHICLES
## Quantum-Safe Security for the Connected Vehicle Ecosystem

---

### EXECUTIVE SUMMARY

The global automotive industry faces an unprecedented security inflection point. With 1.4 billion vehicles on the road worldwide and vehicle lifecycles stretching 15–20 years into the future, the cryptographic foundations underpinning today's connected vehicle ecosystem will be obsolete long before most of those vehicles retire. Cars manufactured and sold today will still be operating in 2040–2045 — precisely the window in which quantum computers are projected to break classical elliptic-curve cryptography (ECDSA, RSA) that currently secures V2X communications, OTA firmware updates, and vehicle telematics. This is not a future problem; it is a present engineering decision with a 15-year blast radius.

UNECE WP.29 Regulation 155, now mandatory across 54 countries, requires automakers to implement a certified Cybersecurity Management System (CSMS) covering the entire vehicle lifecycle. For OEMs and Tier 1 suppliers, this means cryptographic architectures must be defensible not just at launch, but through the vehicle's operational life. Type approval authorities are increasingly scrutinizing whether the security controls documented today will remain valid in 2040. Classical ECDSA-based V2X authentication fails that test. The compliance clock is ticking, and the penalty for non-compliance is market withdrawal.

QBITEL Bridge delivers the automotive industry's first production-grade, post-quantum cryptography (PQC) platform purpose-built for connected vehicles. With sub-5ms V2X message authentication, 72% smaller implicit certificates (666 bytes vs. 2,420 bytes for Dilithium), batch verification at 1,500+ messages per second, and a staged fleet OTA rollout engine (canary → 1% → 10% → 100% with automatic rollback), QBITEL enables OEMs, Tier 1 suppliers, and smart city operators to migrate to quantum-safe V2X security without disrupting today's operations. Full UNECE WP.29 R155/R156, ISO/SAE 21434, and IEEE 1609.2 compliance evidence is generated automatically.

---

### THE CONNECTED VEHICLE SECURITY CRISIS

#### Threat 1: V2X Message Spoofing (A Fake Collision Warning Can Cause a Real Crash)

Vehicle-to-Everything (V2X) communications — the messages that vehicles exchange with each other and with roadside infrastructure — are the nervous system of the connected vehicle ecosystem. Basic Safety Messages (BSMs) broadcast position, speed, and heading 10 times per second. Signal Phase and Timing (SPaT) messages coordinate traffic flow through intersections. Emergency Vehicle Alert (EVA) messages clear corridors for first responders. None of these messages carry cryptographic authentication in most deployed V2X infrastructure today.

The consequence is stark: a $20 software-defined radio (SDR) device can broadcast counterfeit V2X messages that are indistinguishable from legitimate vehicle transmissions. A spoofed BSM reporting a phantom vehicle can trigger automatic emergency braking in following traffic, causing real collisions. Falsified SPaT messages can create artificial green lights to manipulate intersection flow. In autonomous vehicle platoons, fabricated convoy commands can cause unauthorized speed changes or force vehicles out of formation. These are not theoretical attacks — they have been demonstrated on production V2X hardware at academic conferences and by security researchers worldwide.

The root cause is an authentication gap. V2X infrastructure was designed for broadcast efficiency, and adding per-message cryptographic signatures was deferred. QBITEL closes this gap with IEEE 1609.2 + PQC: every V2X message is bound to a verified vehicle identity, and spoofed messages are cryptographically rejected in under 5 milliseconds.

#### Threat 2: The Quantum Fleet Lifecycle Problem

Modern cryptography — the ECDSA signatures that currently protect V2X, the TLS sessions that secure vehicle telematics, the RSA keys that authenticate OTA firmware — is mathematically secure against today's computers. It will not be secure against tomorrow's quantum computers. NIST, NSA, and the world's leading cryptographers project that quantum computers capable of breaking 2048-bit RSA and 256-bit elliptic curves will be operational between 2030 and 2045.

For most industries, this gives a reasonable runway. Not for automotive. A vehicle sold in showrooms today is expected to remain in service until 2040–2045 — squarely within the quantum threat window. Every encrypted telematics session from that vehicle, every OTA firmware channel, every V2X private key can be targeted by "harvest-now-decrypt-later" (HNDL) attacks: adversaries collect encrypted data today and decrypt it retrospectively once quantum capability is achieved. Vehicle location history, navigation data, behavioral telemetry, and identity information collected over a vehicle's lifetime represent an attractive HNDL target for nation-state actors.

The only way to protect 15-to-20-year fleet lifecycles is to deploy quantum-safe cryptography before those keys are harvested. Retrofitting cryptography mid-lifecycle is technically feasible but 10–50× more expensive than building it in at the beginning. QBITEL enables OEMs to make the right choice now — at $10–30 per vehicle per year — before the harvest window closes.

#### Threat 3: UNECE WP.29 R155 — The Compliance Clock Is Ticking

UNECE World Forum for Harmonization of Vehicle Regulations (WP.29) Regulation 155, effective July 2022 for new vehicle types in the European Union and now adopted by 54 countries, fundamentally changes the regulatory landscape for automotive cybersecurity. R155 is not a checklist — it is a systems requirement for a certified Cybersecurity Management System (CSMS) that must cover threat analysis, security design, validation, incident monitoring, and ongoing management across the entire vehicle lifecycle.

For OEMs, type approval now depends on documented evidence of:
- Threat Analysis and Risk Assessment (TARA) for all vehicle attack surfaces
- Security validation test results
- CSMS documentation demonstrating ongoing monitoring capability
- Supply chain cybersecurity management

Critically, R155 requires that the security controls remain effective throughout the vehicle's operational life. A vehicle type-approved in 2026 must be defensible in 2044. Submitting a security architecture built on ECDSA — a cryptographic primitive projected to be broken within the vehicle's compliance window — creates a documented gap that will increasingly attract regulatory scrutiny. Several WP.29 signatory countries have begun asking type approval applicants to justify their cryptographic longevity assumptions.

UN R156, the companion regulation, adds mandatory Software Update Management System (SUMS) requirements, demanding cryptographically authenticated OTA updates with full audit trails. QBITEL provides both the technical controls and the automated compliance evidence generation required for R155/R156 type approval.

---

### QBITEL BRIDGE FOR AUTOMOTIVE

QBITEL Bridge for Automotive is a comprehensive post-quantum security platform that addresses all three of these threats in a single, integrated deployment. The platform provides cryptographic-level V2X message authentication using IEEE 1609.2 extended with NIST-standardized PQC algorithms (Falcon-512 for signatures, ML-KEM-768 for key encapsulation), enabling every connected vehicle and roadside unit to verify the authenticity of every V2X message before acting on it — in under 5 milliseconds.

At the fleet scale required by global OEMs, QBITEL delivers implicit certificate compression that reduces certificate sizes by 72% (666 bytes vs. 2,420 bytes for uncompressed Dilithium) to fit within V2X DSRC bandwidth constraints, batch verification at 1,500+ messages per second using SIMD acceleration for dense urban intersections and highway platoons, and a staged OTA migration engine that transitions entire fleets to PQC cryptography with automatic rollback protection. Full Security Credential Management System (SCMS) integration provides pseudonym certificate management, misbehavior detection, and fleet-scale CRL distribution.

From a compliance standpoint, QBITEL automates the generation of all required UNECE WP.29 R155/R156 and ISO/SAE 21434 evidence, reducing type approval documentation time from 6–12 months to 2–3 months. Every security control deployed is traceable to a documented compliance requirement — giving OEMs, regulators, and type approval authorities a complete, auditable security architecture for the full vehicle lifecycle.

---

### CAPABILITY 1: V2X MESSAGE AUTHENTICATION (IEEE 1609.2 + PQC)

V2X message authentication is the foundational capability of QBITEL Bridge for Automotive. Each transmitted V2X message — whether a Basic Safety Message from a vehicle, a Signal Phase and Timing message from a traffic controller, or an Emergency Vehicle Alert from a first responder — is cryptographically bound to a verified sender identity using a Falcon-512 digital signature embedded in the IEEE 1609.2 security header.

The algorithms selected represent the optimal trade-off for V2X constraints:

- **ML-KEM-768 (CRYSTALS-Kyber, NIST FIPS 203)**: Handles key encapsulation for secure channel establishment between vehicles and SCMS infrastructure. Provides 192-bit post-quantum security level.
- **ML-DSA-65 (CRYSTALS-Dilithium, NIST FIPS 204)**: Used for long-term identity certificates where signature size is less constrained.
- **Falcon-512**: Used for short-lived pseudonym certificates where the 666-byte signature size is critical to fit within DSRC frame constraints.

Authentication is backward compatible with existing RSU deployments. During the hybrid transition period, vehicles sign with both ECDSA (for legacy RSUs) and Falcon-512 (for upgraded infrastructure). Legacy vehicles see a valid classical signature; upgraded RSUs verify both. This dual-mode operation enables gradual infrastructure migration without requiring fleet-wide simultaneous cutover.

All SAE J2735 message types are supported: BSM (Basic Safety Message), SPaT (Signal Phase and Timing), MAP (Map Data), TIM (Traveler Information Message), EVA (Emergency Vehicle Alert), RSA (Road Side Alert), and RTCM (differential GPS corrections).

---

### CAPABILITY 2: IMPLICIT CERTIFICATE COMPRESSION (72% SMALLER)

Bandwidth is the binding constraint in V2X security. IEEE 802.11p (DSRC) channels operate at 6–27 Mbps shared across all vehicles in range, with each vehicle broadcasting BSMs 10 times per second. At highway densities of 100+ vehicles per intersection, the aggregate V2X overhead is substantial. Standard post-quantum certificates — Dilithium certificates at 2,420 bytes each — would exceed the bandwidth budget for high-density deployments.

QBITEL's implicit certificate scheme for Falcon-512 reduces this to 666 bytes per certificate — a 72% reduction. The compression architecture combines three techniques:

1. **Implicit certificates (ECQV-equivalent for lattice signatures)**: Instead of carrying the full public key, the certificate encodes the key implicitly using a reconstruction algorithm. The receiver derives the public key during verification, eliminating redundant field storage.
2. **X9.62 point compression**: Where elliptic curve parameters appear in hybrid certificates, compressed point encoding reduces field sizes by 50%.
3. **Dictionary-based encoding**: Common OID and policy fields are replaced with 1-2 byte dictionary codes, eliminating repetitive ASN.1 verbosity.

The result: 666-byte Falcon-512 implicit certificates fit within a single DSRC MAC frame, enabling per-message authentication with no fragmentation overhead. Batch verification further amplifies throughput: QBITEL's SIMD engine verifies groups of Falcon-512 signatures simultaneously, achieving 1,500+ verifications per second on automotive-grade hardware.

---

### CAPABILITY 3: BATCH V2X VERIFICATION AT SCALE

Dense V2X environments — busy urban intersections, highway platoon corridors, stadium events, emergency vehicle deployments — can generate thousands of V2X messages per second that must be authenticated before the receiving system can act on them. Sequential per-message verification at even 200 verifications/second would create a processing backlog that degrades safety-critical message latency.

QBITEL's batch verification engine uses AVX2 (x86) and NEON (ARM) SIMD instruction sets to verify multiple Falcon-512 signatures in parallel on a single processor core. The architecture:

- Queues incoming V2X messages into verification batches of 8–32
- Executes SIMD-parallel polynomial arithmetic for simultaneous signature verification
- Returns verification results as a bitmask, enabling downstream processing to immediately act on authenticated messages
- Falls back to sequential verification for low-density scenarios to avoid batch latency overhead

Benchmark results on automotive-grade hardware (NXP S32G, Qualcomm SA8195P):

| Scenario | Message Rate | QBITEL Throughput | Latency |
|---|---|---|---|
| Rural intersection | 50 msg/sec | 1,500+ msg/sec capacity | <3ms |
| Urban intersection | 500 msg/sec | 1,500+ msg/sec capacity | <4ms |
| Highway platoon (10 vehicles) | 100 msg/sec | 1,500+ msg/sec capacity | <5ms |
| Dense urban (100+ vehicles) | 1,000 msg/sec | 1,500+ msg/sec capacity | <5ms |

At 1,500+ verifications per second, QBITEL provides 3× headroom above the densest production V2X deployment scenarios recorded in USDOT V2X pilots.

---

### CAPABILITY 4: FLEET-WIDE OTA PQC MIGRATION

Transitioning a fleet of millions of vehicles to post-quantum cryptography requires an OTA migration engine that is conservative enough to prevent bricking vehicles, fast enough to meet compliance deadlines, and observable enough to maintain fleet-wide visibility during rollout.

QBITEL's staged OTA rollout architecture:

**Stage 0 — Canary (10 vehicles)**
A hand-selected canary cohort of 10 vehicles receives the PQC update. These vehicles are monitored for 48 hours across all V2X message types, OTA channel integrity, and ECU diagnostic telemetry. Any anomaly triggers automatic canary halt.

**Stage 1 — 1% Rollout (Days 3–30)**
1% of the total fleet (e.g., 100,000 vehicles in a 10M fleet) receives the update. Monitoring runs for 24 hours. If any stage's error rate exceeds 0.1%, automatic rollback is triggered for the entire stage cohort. Rollback completes within 30 minutes via the same OTA channel.

**Stage 2 — 10% Rollout (Months 1–3)**
10% of fleet expands the coverage envelope. Geographic, climate, and usage-pattern diversity ensures broad validation. Monitoring continues for 48 hours per deployment batch.

**Stage 3 — Full Fleet (Months 4–12)**
100% fleet coverage with continuous monitoring. Delta update technology reduces OTA payload by ~80% (a 500MB full firmware image becomes a ~100MB binary diff), enabling delivery within 4 hours vs. 24 hours for full images.

TPM 2.0 attestation verifies every post-installation update hash before the vehicle activates the new PQC stack, providing cryptographic assurance that the deployed software matches the signed package — preventing supply chain injection at the OTA delivery stage.

---

### CAPABILITY 5: SCMS INTEGRATION & PSEUDONYM MANAGEMENT

The Security Credential Management System (SCMS) is the PKI backbone of V2X infrastructure — issuing, rotating, and revoking the pseudonym certificates that vehicles use to authenticate V2X messages while preserving driver privacy. QBITEL provides pre-built, production-tested integrations with the two dominant SCMS providers:

- **CAMP (Crash Avoidance Metrics Partnership)**: The primary SCMS provider for North American DSRC deployments, managing millions of vehicle pseudonym pools.
- **OnBoard Security (OBS)**: International SCMS operations supporting C-V2X and multi-jurisdiction deployments.

A RESTful SCMS API enables integration with custom or proprietary SCMS implementations.

**Pseudonym Certificate Management**

Each vehicle is provisioned with a pool of Falcon-512 pseudonym certificates — short-lived (1-week validity), unlinkable certificates that change randomly during a trip. A pool of 20+ pseudonyms per week ensures no observer can correlate certificate changes to track a specific vehicle's location. QBITEL pre-provisions the next week's certificate pool 48 hours in advance, eliminating connectivity gaps during rotation.

**Misbehavior Detection**

QBITEL's misbehavior detection engine cross-validates received V2X messages against physics constraints: a BSM claiming 200 km/h on a surface street, position data inconsistent with local map topology, or acceleration profiles physically impossible for the claimed vehicle class are flagged as suspected spoofing. Confirmed misbehavior events trigger automated reports to the SCMS, which can revoke the offending pseudonym and distribute updated CRLs fleet-wide within 100ms.

---

### CAPABILITY 6: AUTONOMOUS VEHICLE PLATOONING SECURITY

Autonomous vehicle platoons — convoys of AVs following a lead vehicle with sub-second following distances — represent the highest-consequence V2X authentication scenario. A compromised or spoofed platoon command can cause emergency braking, unauthorized acceleration, or unsafe following distance changes across an entire convoy simultaneously.

QBITEL's platooning security architecture provides:

**Platoon Leader Verification**: The designated platoon leader is bound to a long-term Falcon-512 identity certificate, distinct from the short-lived pseudonym certificates used for general V2X. This allows convoy members to verify they are receiving authentic commands from the true leader, not a rogue vehicle broadcasting spoofed leadership claims.

**Per-Command Authentication**: Every platoon command — following distance adjustment, speed change, emergency stop, platoon disbanding — is signed by the sender. Receivers verify the signature before executing the command. A delay-tolerant signature cache enables verification even during brief connectivity interruptions.

**Rogue Vehicle Detection**: Unauthorized vehicles attempting to join an authenticated platoon are detected within 100ms. The platoon leader's QBITEL module identifies authentication failures from unexpected senders and triggers graceful platoon disbanding — increasing following distances to safe manual-driving gaps — before the rogue vehicle can influence convoy behavior.

**Failsafe Disbanding**: If authentication failures exceed a configurable threshold, the platoon automatically disbands into individual vehicles operating under normal manual or semi-autonomous driving rules. Safety is never traded for platoon efficiency.

---

### CAPABILITY 7: ISO/SAE 21434 / UNECE WP.29 COMPLIANCE

QBITEL's compliance engine automates the generation of all documentation required for UNECE WP.29 R155 type approval and ISO/SAE 21434 cybersecurity validation.

**Automated TARA Generation**

QBITEL's AI-assisted Threat Analysis and Risk Assessment engine generates comprehensive TARA documentation covering all V2X attack surfaces: message spoofing, replay attacks, relay attacks, Sybil attacks, denial-of-service, supply chain compromise, and harvest-now-decrypt-later attacks on telematics channels. TARA output is formatted for direct inclusion in ISO 21434 cybersecurity case documentation.

**Type Approval Evidence Package**

The evidence package automatically generated includes:
- Threat catalog with risk ratings (ISO/SAE 21434 §15)
- Security goals and cybersecurity concepts (§14)
- Cybersecurity validation test results (§13)
- Cryptographic algorithm justification (including quantum threat timeline analysis)
- OTA update management evidence (R156 SUMS)
- Fleet coverage certificate with deployment statistics

**CSMS Documentation**

QBITEL's Cybersecurity Management System documentation covers all R155 Annex 5 requirements: organizational processes, supply chain management, incident response, and post-production monitoring. Evidence is updated continuously as the fleet operates, providing a living compliance record rather than a point-in-time snapshot.

---

### COMPLIANCE COVERAGE

| Framework | Coverage | Key Requirement |
|---|---|---|
| UNECE WP.29 R155 | Full | Cybersecurity Management System |
| UNECE WP.29 R156 | Full | Software Update Management |
| ISO/SAE 21434 | Full | TARA, security validation |
| IEEE 1609.2 | Extended (PQC) | V2X security services |
| SAE J3061 | Aligned | Cyber-physical systems guidebook |
| NIST FIPS 203 | Native | ML-KEM (Kyber) key encapsulation |
| NIST FIPS 204 | Native | ML-DSA (Dilithium) signatures |
| SAE J2735 | Compatible | DSRC/C-V2X message sets |
| ISO 15118 | Supported | EV charging communication |
| SAE J1939 | Supported | Commercial vehicle CAN bus |

---

### INTEGRATION ECOSYSTEM

**OEM & Tier 1 Integration**

| Partner Category | Supported Platforms |
|---|---|
| V2X Chipsets | Qualcomm 9150 C-V2X, NXP RoadLINK, Autotalks CRATON2 |
| Telematics Control Units | HARMAN (Samsung), Continental, Bosch |
| Automotive SoCs | NXP S32G (ASIL-D), Qualcomm SA8195P, Renesas R-Car |
| HSM/Secure Element | Infineon SLB 9672, NXP SE050, STMicro ST33 |

**Infrastructure Integration**

| Partner Category | Supported Platforms |
|---|---|
| RSU Manufacturers | Kapsch TrafficCom, Q-Free, Commsignia, Savari |
| Traffic Management | Iteris, INTETRA, Siemens Mobility |
| SCMS Providers | CAMP (North America), OnBoard Security (International) |

**Cloud Backend**

QBITEL supports deployment on all major cloud providers with sovereign deployment option for regulated markets:

- AWS IoT Core + Greengrass (with QBITEL PQC layer)
- Microsoft Azure IoT Hub (with QBITEL TLS 1.3 PQC extension)
- Google Cloud IoT (with QBITEL certificate management)
- On-premise / Sovereign (for China, EU data sovereignty requirements)

---

### DEPLOYMENT TIMELINE

**Phase 1 — Months 1–2: SCMS Integration & Certificate Authority Setup**
- SCMS provider API integration (CAMP or OBS)
- Falcon-512 root CA provisioned in FIPS 140-3 Level 3 HSM
- Initial pseudonym certificate pool generated for pilot fleet
- CRL distribution infrastructure validated (<100ms target)

**Phase 2 — Months 3–4: RSU Firmware Update & V2X Authentication Activation**
- RSU firmware update with QBITEL V2X authentication module
- Backward compatibility testing with legacy OBUs
- Misbehavior detection baseline calibration
- Traffic signal anti-spoofing activation

**Phase 3 — Months 5–8: Fleet OTA Canary Rollout (1% → 10%)**
- Canary fleet (10 vehicles) deployment and 48h monitoring
- 1% fleet rollout with automatic rollback armed
- 10% fleet rollout with geographic diversity validation
- Performance benchmarking vs. V2X latency requirements

**Phase 4 — Months 9–12: Full Fleet Migration & UNECE WP.29 Documentation**
- 100% fleet OTA migration (staged, 30-day batches)
- Automated TARA generation and evidence package compilation
- Type approval submission support
- WP.29 R155/R156 compliance documentation finalized

---

### PERFORMANCE SPECIFICATIONS

| Metric | Industry Requirement | QBITEL Performance |
|---|---|---|
| V2X signature verification | <10ms (IEEE 1609.2) | <5ms |
| Batch throughput | 500+ msg/sec | 1,500+ msg/sec |
| Certificate size (pseudonym) | <2KB (DSRC frame budget) | 666 bytes (Falcon-512) |
| OTA update delivery | <24 hours | <4 hours (delta update) |
| False positive rate | <0.001% | <0.0001% |
| Fleet OTA coverage stages | Staged deployment | Canary → 1% → 10% → 100% |
| CRL distribution to RSUs | <1 second | <100ms |
| SCMS bulk enrollment | — | 10,000 vehicles/hour |
| Cost per vehicle per year | $50–200 (classical PKI) | $10–30 (QBITEL PQC) |
| Quantum security level | None (classical ECDSA) | NIST Level 1–3 (Falcon-512/1024) |

---

### COMPETITIVE DIFFERENTIATION

**QBITEL Bridge vs. Classical ECDSA V2X PKI**

Classical V2X PKI — ECDSA-signed IEEE 1609.2 certificates — is mathematically secure today. It will not be secure in 2035–2045, the tail of the current vehicle lifecycle. Harvest-now-decrypt-later attacks are already collecting encrypted vehicle telemetry and V2X traffic for retrospective decryption. QBITEL delivers quantum-safe protection from day one, with a migration path to full PQC without requiring hardware replacement.

**QBITEL Bridge vs. HSM-Only Vendors (Thales, Infineon, NXP)**

Hardware Security Module vendors provide excellent key storage and hardware cryptographic acceleration. They do not provide V2X protocol integration, implicit certificate compression, SCMS connectivity, staged OTA pipelines, or compliance evidence generation. QBITEL integrates with your existing HSM hardware while providing the full V2X security stack above it.

**QBITEL Bridge vs. In-Vehicle Cybersecurity (Argus, Karamba)**

Argus Cyber Security, Karamba Security, and similar platforms protect in-vehicle networks — CAN bus anomaly detection, ECU intrusion prevention, telematics gateway monitoring. QBITEL protects V2X communications — the messages exchanged between vehicles and infrastructure. These are complementary, non-overlapping layers of the vehicle security architecture.

**QBITEL Bridge vs. Cloud-Native IoT Encryption (AWS IoT, Azure IoT)**

Cloud providers encrypt vehicle-to-cloud (V2C) channels excellently. They cannot protect V2X (vehicle-to-vehicle, vehicle-to-infrastructure) communications — low-latency, peer-to-peer, broadcast messages that never traverse the cloud. QBITEL covers the V2X security layer that cloud providers architecturally cannot reach.

**QBITEL Bridge vs. Wait-and-See**

Some OEMs prefer to wait for IEEE 1609.2 PQC standard finalization before committing. The IEEE working group is years from publication. Meanwhile, vehicles being designed today have 2025–2028 production start dates and 15–20 year lifecycle commitments. The cryptographic decisions made in the design phase are extremely expensive to change mid-lifecycle. QBITEL's implementation is aligned with NIST FIPS 203/204 final standards and provides a migration path to the IEEE 1609.2 PQC extension when finalized.

---

### CUSTOMER SCENARIOS

#### Scenario A: Global OEM — Fleet-Wide PQC Migration

**Profile**: Global automotive OEM, 10 million vehicle fleet across 60 countries. Current ECDSA-based V2X deployment with CAMP SCMS integration. UNECE WP.29 type approval pending renewal in 24 months.

**Challenge**: The OEM's security team has identified that their current ECDSA V2X certificates will be cryptographically vulnerable within the expected operational life of vehicles currently in production. They need a migration path that:
1. Does not disrupt the 10M vehicle OTA infrastructure
2. Passes UNECE WP.29 R155 type approval in 23 countries simultaneously
3. Maintains backward compatibility with DSRC RSUs not yet upgraded
4. Fits within a $25–35/vehicle/year security budget

**Solution**: QBITEL staged OTA rollout with CAMP SCMS integration. Hybrid ECDSA + Falcon-512 dual-signing mode during 12-month transition. Automated TARA and type approval evidence generation. Delta OTA reduces payload 80%.

**Result**:
- 10M vehicles migrated to PQC over 12 months, zero forced service interruptions
- UNECE WP.29 R155 type approval achieved in all target markets
- $28/vehicle/year total cost, within budget
- Certificate size: 666 bytes (Falcon-512) — no DSRC bandwidth impact

#### Scenario B: Tier 1 Supplier — V2X Module Security

**Profile**: Global Tier 1 supplier developing next-generation V2X telematics control units (TCUs) for three major OEM programs launching in 2027.

**Challenge**: The supplier must integrate V2X authentication into a resource-constrained automotive SoC (NXP S32G) with a fixed BOM cost target of $3.50/unit for the cryptographic security component. Current PQC library benchmarks show Dilithium signatures at 2,420 bytes — too large for the DSRC bandwidth budget. Batch verification performance on target hardware is insufficient for dense urban scenarios.

**Solution**: QBITEL Falcon-512 implicit certificate scheme (666 bytes), SIMD-accelerated batch verification tuned for NXP S32G NEON instructions, SCMS provisioning integration with CAMP API. SDK delivered as C library with hardware abstraction layer.

**Result**:
- 666-byte certificates fit within DSRC frame budget with no fragmentation
- 1,500+ msg/sec batch verification on NXP S32G at <5ms latency
- BOM cost target met with hardware-assisted Falcon-512 acceleration
- Certified to ISO/SAE 21434 for OEM program qualification

#### Scenario C: Smart City — V2I Infrastructure Security

**Profile**: Major metropolitan area deploying 500 RSUs as part of a federal V2X pilot program. RSUs control 200 signalized intersections and provide SPaT/MAP data to connected vehicles. Current RSU firmware has no V2X message authentication.

**Challenge**: The city's traffic management center has identified that unauthenticated V2X broadcasts can be exploited to manipulate intersection timing, triggering gridlock or creating unsafe signal sequences. They need RSU-level authentication without replacing the RSU hardware (installed 18 months ago, $15,000/unit replacement cost).

**Solution**: QBITEL RSU firmware module update (over-the-air to RSU management system), Falcon-512 certificate provisioning for all 500 RSUs, traffic signal SPaT/MAP authentication activation, intersection-level misbehavior detection.

**Result**:
- All 500 RSUs authenticated with Falcon-512 certificates in 3 months
- Zero RSU hardware replacement required (software-only update)
- Intersection spoofing detection: 100% of simulated attack messages rejected in <5ms
- Federal pilot documentation: UNECE WP.29-equivalent compliance evidence

---

### NEXT STEPS

1. **V2X Security Assessment (2 weeks)**
   Passive V2X traffic capture on target deployment corridors, current SCMS architecture review, UNECE WP.29 R155 gap assessment with detailed findings report. No infrastructure changes required.

2. **UNECE WP.29 Gap Analysis (1 week)**
   Document gaps in current security architecture against R155 requirements. Identify type approval evidence gaps. Estimate documentation timeline with and without QBITEL.

3. **Proof of Concept — 1,000-Vehicle Pilot (Month 1–2)**
   Deploy QBITEL to a defined pilot fleet with SCMS integration and performance benchmarking. Deliver: latency benchmark report, certificate size validation, OTA rollout test with rollback verification.

4. **Fleet Migration Planning (Month 3)**
   Staged OTA rollout roadmap, rollback procedure documentation, type approval evidence package scope. Delivered as an executable project plan with weekly milestones.

---

### CONTACT

**QBITEL Automotive Solutions**

enterprise@qbitel.com
https://bridge.qbitel.com

---

*QBITEL Bridge — Because a connected vehicle ecosystem is only as safe as its weakest V2X message.*

---

**Confidential — For Authorized Recipients Only**
© 2026 QBITEL. All Rights Reserved.

V2X Security | Fleet PQC Migration | UNECE WP.29 Compliance | Autonomous Vehicle Security
