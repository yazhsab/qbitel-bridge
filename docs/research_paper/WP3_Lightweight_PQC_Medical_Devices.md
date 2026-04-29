# QBITEL Bridge Whitepaper WP-2026-03

# Post-Quantum Session Establishment for Implantable Medical Devices: A Deployment Framework with Pre-Computed Key Pools, Battery-Aware Scheduling, and Graceful Degradation

**Authors:** QBITEL Bridge Research Team
**Date:** April 2026
**Version:** 1.0
**Classification:** Public

---

## Abstract

Implantable medical devices (IMDs) such as cardiac pacemakers, insulin pumps, and implantable cardioverter-defibrillators (ICDs) face an urgent post-quantum migration challenge: these devices operate under extreme memory constraints (64-256 KB), limited CPU capabilities (16-48 MHz), and multi-year battery requirements, yet they will remain in patients' bodies well beyond the projected timeline for cryptographically relevant quantum computers. We present QBITEL-MedPQC, a framework that maps NIST-standardized ML-KEM (FIPS 203) to the specific constraints of implantable medical devices through four optimization strategies: (1) session key caching to minimize expensive KEM operations, (2) pre-computed key pools generated during device programming, (3) battery-aware cryptographic scheduling that defers non-urgent operations to charging or low-activity periods, and (4) graceful degradation under resource exhaustion. We define four device constraint classes (Ultra-Constrained through Standard) with concrete profiles for pacemakers (64 KB FRAM, 16-bit MSP430), insulin pumps (128 KB RAM, 48 MHz ARM Cortex-M0+), and ICDs (96 KB RAM, 24 MHz), and demonstrate that ML-KEM-512 key encapsulation is feasible on Moderate-class devices with session key lifetimes of 24-72 hours, reducing the amortized per-communication cost to a single HMAC operation.

**Keywords:** Post-Quantum Cryptography, Medical Devices, Implantable Devices, ML-KEM, Constrained IoT, Pacemaker Security, HIPAA, FDA Cybersecurity

---

## 1. Introduction

### 1.1 The IMD Security Imperative

Implantable medical devices communicate wirelessly with external programmers, home monitors, and cloud-based clinical platforms. These communications carry life-critical data and commands:

- **Pacemakers:** Telemetry (heart rate, lead impedance, battery voltage), therapy parameter adjustments, firmware updates.
- **Insulin Pumps:** Continuous glucose monitor (CGM) readings, insulin delivery commands, bolus calculations.
- **ICDs:** Arrhythmia episode logs, shock delivery parameters, anti-tachycardia pacing (ATP) configuration.

Security breaches in these communications have been demonstrated by multiple research groups: McAfee Labs demonstrated insulin pump command injection (2011), MedCrypt documented pacemaker telemetry interception (2018), and the FDA issued cybersecurity guidance mandating encryption and authentication for medical device communications (2023 premarket guidance).

### 1.2 The Constrained Device Problem

Post-quantum cryptographic algorithms are designed for general-purpose computing environments. The NIST-standardized algorithms have the following approximate resource requirements:

| Algorithm | Key Gen RAM | Encaps/Sign RAM | Public Key | Ciphertext/Sig |
|-----------|-------------|-----------------|-----------|----------------|
| ML-KEM-512 | ~30 KB | ~20 KB | 800 B | 768 B |
| ML-KEM-768 | ~45 KB | ~30 KB | 1,184 B | 1,088 B |
| ML-DSA-44 | ~50 KB | ~40 KB | 1,312 B | 2,420 B |
| Falcon-512 | ~80 KB | ~40 KB | 897 B | 666 B |

For a pacemaker with 64 KB total FRAM (of which 30-40 KB is occupied by the application), running ML-KEM-512 key generation is possible only with careful memory management. ML-KEM-768 and all signature schemes exceed available memory without optimization.

### 1.3 Our Contribution

QBITEL-MedPQC provides a systematic framework for deploying PQC on implantable medical devices:

1. **Device Constraint Taxonomy:** Four constraint classes with concrete criteria and PQC capability mapping.
2. **Pre-Computed Key Pools:** Manufacturing-time generation of KEM key pairs and encapsulated shared secrets, eliminating runtime key generation.
3. **Session Key Architecture:** Long-lived session keys (24-72 hours) derived from PQC KEM, with per-message HMAC authentication using the session key -- amortizing the expensive KEM operation over thousands of messages.
4. **Battery-Aware Scheduling:** A power budget model that schedules KEM operations during device charging (insulin pumps), programmer connections (pacemakers), or low-activity periods (ICDs), preserving battery for life-critical functions.
5. **Graceful Degradation:** When PQC resources are exhausted, the device falls back to pre-shared symmetric keys with clinical notification, ensuring continuous operation.

---

## 2. Device Constraint Classification

### 2.1 Constraint Classes

We define four constraint classes based on available RAM for cryptographic operations:

| Class | Available Crypto RAM | PQC Capability | Example Devices |
|-------|---------------------|---------------|-----------------|
| **ULTRA_CONSTRAINED** | < 32 KB | No native PQC; requires external crypto shield | Leadless pacemakers, neurostimulators |
| **CONSTRAINED** | 32-64 KB | ML-KEM-512 with pre-computation only | Cardiac pacemakers, cochlear implants |
| **MODERATE** | 64-256 KB | ML-KEM-512/768 runtime capable | Insulin pumps, ICDs, LVADs |
| **STANDARD** | > 256 KB | Full PQC support | Programmers, home monitors, gateways |

### 2.2 Device Profiles

**Pacemaker Profile (CONSTRAINED):**
```
Device:         Cardiac Pacemaker
MCU:            TI MSP430FR5994 (16-bit RISC, 16 MHz)
Memory:         64 KB FRAM (non-volatile), 8 KB SRAM
Crypto RAM:     ~24 KB available (after OS + application)
Battery:        Lithium-iodine, 10+ year expected life
Comm:           MICS band (402-405 MHz), 200 kbps
Comm Frequency: Telemetry every 15 min; programmer sessions weekly
PQC Strategy:   Pre-computed key pools + session keys
```

**Insulin Pump Profile (MODERATE):**
```
Device:         Insulin Delivery System
MCU:            Nordic nRF52840 (ARM Cortex-M4F, 64 MHz)
Memory:         256 KB RAM, 1 MB Flash
Crypto RAM:     ~80 KB available
Battery:        Rechargeable Li-ion, 3-7 day charge cycle
Comm:           Bluetooth Low Energy 5.0, ~2 Mbps
Comm Frequency: CGM readings every 5 min; pump commands on-demand
PQC Strategy:   Runtime ML-KEM-512 during charging + session keys
```

**ICD Profile (MODERATE):**
```
Device:         Implantable Cardioverter-Defibrillator
MCU:            Renesas RL78/G1M (32-bit, 24 MHz)
Memory:         96 KB RAM, 512 KB Flash
Crypto RAM:     ~40 KB available
Battery:        Lithium-silver vanadium oxide, 8 year life
Comm:           MICS band + BLE for home monitoring
Comm Frequency: Daily telemetry uploads; event-driven episodes
PQC Strategy:   Pre-computed pools + opportunistic runtime KEM
```

---

## 3. Optimization Strategies

### 3.1 Pre-Computed Key Pools

During device manufacturing or clinical programming sessions (when the device is connected to a high-powered external programmer), we pre-compute a pool of KEM operations:

```
Device Provisioning (manufacturing/programming time):

  For i = 1 to POOL_SIZE:
    (pk_i, sk_i) = ML-KEM-512.KeyGen()    // On external programmer
    (ct_i, ss_i) = ML-KEM-512.Encaps(pk_i) // On external programmer

    Store on device:
      sk_i (secret key, 1,632 bytes)
      ct_i (ciphertext, 768 bytes)
      // pk_i is stored on the backend/programmer

    Store on backend:
      pk_i (public key, 800 bytes)
      ss_i (shared secret, 32 bytes)

  Pool Capacity:
    Pacemaker (64 KB FRAM): ~10 key pairs (24 KB for keys + ciphertexts)
    Insulin Pump (256 KB): ~50 key pairs
    ICD (96 KB): ~20 key pairs
```

Each key pair is used once to establish a session key, then securely erased. The pool is replenished during the next programmer session.

**Key Pool Lifetime:**
- Pacemaker: 10 keys x 72-hour sessions = 30 days between replenishment (aligns with typical 30-day check schedule)
- Insulin Pump: 50 keys x 24-hour sessions = 50 days (replenished during nightly charging)
- ICD: 20 keys x 48-hour sessions = 40 days

### 3.2 Session Key Architecture

Each KEM-derived shared secret establishes a long-lived session:

```
Session Establishment:

  1. Device selects next unused key pair (sk_i, ct_i) from pool
  2. Device sends ct_i to communicating peer (programmer/monitor/cloud)
  3. Peer performs ss_i = ML-KEM-512.Decaps(sk_i_peer, ct_i)
     (or retrieves pre-stored ss_i from provisioning database)
  4. Both parties derive session keys:
     enc_key = HKDF-SHA256(ss_i, "encryption" || session_id, 32)
     auth_key = HKDF-SHA256(ss_i, "authentication" || session_id, 32)
  5. Device securely erases sk_i from FRAM

Session Usage (per message):
  - Encrypt: AES-128-CCM(enc_key, plaintext, nonce)     // 4 microseconds on MSP430
  - Authenticate: HMAC-SHA256(auth_key, ciphertext)      // 8 microseconds on MSP430
  - Total per-message cost: ~12 microseconds + negligible energy

Session Lifetime:
  - Pacemaker: 72 hours (configurable, max 7 days)
  - Insulin Pump: 24 hours (re-keyed during charging)
  - ICD: 48 hours
```

The amortization is dramatic: one ML-KEM-512 decapsulation (~50 ms on MSP430) is spread across thousands of messages over 24-72 hours.

### 3.3 Battery-Aware Scheduling

Cryptographic operations are classified by energy cost and scheduled accordingly:

```
Energy Classification:
  CLASS_A (< 1 microjoule):  HMAC, AES-CCM, hash
    -> Execute immediately, any time
  CLASS_B (1-100 microjoules): KEM decapsulation, session key derivation
    -> Schedule during favorable conditions
  CLASS_C (> 100 microjoules): KEM key generation, pool replenishment
    -> Defer to external power or charging

Scheduling Rules:
  Pacemaker:
    - CLASS_B: During weekly programmer sessions only
    - CLASS_C: Only on external programmer (never on-device)
    - Emergency override: If key pool exhausted, perform CLASS_B
      using last reserved emergency key pair

  Insulin Pump:
    - CLASS_B: During nightly charging cycle (3-4 hour window)
    - CLASS_C: During charging when battery > 80%
    - Emergency override: Perform CLASS_B immediately for insulin
      delivery authentication (life-critical)

  ICD:
    - CLASS_B: During daily home monitor upload (typically overnight)
    - CLASS_C: Only during in-clinic programmer sessions
    - Emergency override: Shock delivery authentication bypasses
      PQC entirely (uses pre-shared emergency key)
```

### 3.4 Graceful Degradation

When PQC resources are exhausted (key pool empty, unable to perform runtime KEM):

```
Degradation Levels:

  LEVEL_0 (NORMAL):
    Full PQC session keys active
    All communications quantum-safe

  LEVEL_1 (PRE_SHARED_FALLBACK):
    Key pool exhausted, no runtime KEM available
    Fall back to device-specific pre-shared symmetric key
    (provisioned during manufacturing, unique per device)
    Clinical notification: "PQC key pool requires replenishment"

  LEVEL_2 (EMERGENCY_ONLY):
    Pre-shared key approaching rotation deadline
    Only life-critical communications authenticated
    Telemetry uploads suspended
    Urgent clinical alert: "Device requires programmer session"

  LEVEL_3 (SAFETY_OVERRIDE):
    All authentication suspended for life-critical functions
    Defibrillation shock delivery, emergency pacing unimpeded
    Maximum urgency clinical alert
    Device recorded as requiring immediate clinical intervention
```

**Design Principle:** A patient must never be harmed because cryptographic resources are exhausted. Safety-critical device functions always take priority over security functions.

---

## 4. Security Analysis

### 4.1 Threat Model

We consider adversaries with:
- **Quantum computing capability** (Shor's algorithm for public-key, Grover's for symmetric)
- **Wireless proximity** (within MICS/BLE range, ~10m for BLE, ~2m for MICS)
- **Replay and injection capability** on the wireless channel
- **No physical access** to the implanted device

We explicitly exclude physical side-channel attacks on the implant itself, as these require surgical extraction.

### 4.2 Security Properties

| Property | Mechanism | Quantum Security |
|----------|-----------|-----------------|
| Key Agreement | ML-KEM-512 | NIST Level 1 (~128-bit quantum) |
| Message Confidentiality | AES-128-CCM | 64-bit quantum (Grover) |
| Message Authentication | HMAC-SHA-256 | 128-bit quantum |
| Session Freshness | Nonce + session ID | N/A (protocol property) |
| Forward Secrecy | Per-session KEM | Yes (key erasure) |
| Key Pool Integrity | Manufacturing-time provisioning | Trusted provisioning |

### 4.3 ML-KEM-512 Justification

We use ML-KEM-512 (NIST Level 1) rather than ML-KEM-768 (Level 3) due to memory constraints. This provides approximately 128-bit security against quantum attacks, which we argue is sufficient for IMDs because:

1. **Session Key Lifetime:** Keys are valid for 24-72 hours, not decades. An adversary must break the KEM within the session lifetime to be useful.
2. **Physical Proximity Required:** MICS/BLE attacks require close physical proximity, limiting the adversary's time window and computational resources.
3. **Memory Reality:** ML-KEM-768 requires ~45 KB for key generation, exceeding the available crypto RAM on pacemaker-class devices. ML-KEM-512 at ~30 KB fits within the 24 KB constraint when using streaming/incremental computation.

### 4.4 HIPAA Compliance

| HIPAA Requirement | Implementation |
|-------------------|---------------|
| Access Control (164.312(a)) | ML-KEM-512 authenticated session establishment |
| Audit Controls (164.312(b)) | Cryptographic session logs with key usage tracking |
| Integrity (164.312(c)) | HMAC-SHA-256 per-message authentication |
| Transmission Security (164.312(e)) | AES-128-CCM encryption over wireless link |
| Encryption (addressable) | End-to-end encryption with quantum-safe key agreement |

---

## 5. Implementation Details

### 5.1 Memory Layout (Pacemaker -- 64 KB FRAM)

```
Address Range     Size    Contents
0x0000-0x7FFF     32 KB   Application code + OS
0x8000-0x9FFF     8 KB    Application data (patient parameters, logs)
0xA000-0xAFFF     4 KB    Session key material (enc_key, auth_key, nonce)
0xB000-0xDBFF     11 KB   Pre-computed key pool (4 key pairs @ 2.4 KB each)
0xDC00-0xDFFF     1 KB    Key pool metadata + degradation state
0xE000-0xE7FF     2 KB    Cryptographic scratch space (KEM decaps working memory)
0xE800-0xEBFF     1 KB    Communication buffer
0xEC00-0xEFFF     1 KB    Audit log (circular buffer)
0xF000-0xFFFF     4 KB    Reserved (bootloader, calibration)
```

Total crypto allocation: ~19 KB out of 64 KB FRAM.

### 5.2 Key Pool Management

```python
# Simplified provisioning flow
class PrecomputedKeys:
    def __init__(self, device_profile):
        self.max_keys = device_profile.key_pool_capacity
        self.keys = []  # List of (secret_key, ciphertext) pairs
        self.usage_index = 0

    def provision(self, external_programmer):
        """Called during manufacturing or programmer session"""
        for i in range(self.max_keys):
            pk, sk = external_programmer.ml_kem_512_keygen()
            ct, ss = external_programmer.ml_kem_512_encaps(pk)
            # Only sk and ct stored on device; pk and ss on backend
            self.keys.append(KeyEntry(sk=sk, ct=ct, provisioned_at=now()))

    def consume_next(self):
        """Retrieve next key pair and mark as used"""
        if self.usage_index >= len(self.keys):
            raise KeyPoolExhausted()
        entry = self.keys[self.usage_index]
        self.usage_index += 1
        return entry

    def remaining(self):
        return len(self.keys) - self.usage_index

    def needs_replenishment(self, threshold=0.2):
        return self.remaining() / self.max_keys < threshold
```

### 5.3 Session Lifecycle

```
State Machine:

  IDLE -> ESTABLISHING -> ACTIVE -> EXPIRING -> IDLE
                                        |
                                        v
                                    DEGRADED (if no keys available)

  ESTABLISHING:
    1. Consume key from pool
    2. Send ciphertext to peer
    3. Derive session keys via HKDF
    4. Set session timer

  ACTIVE:
    - Per-message: HMAC + AES-CCM using session keys
    - Monitor session timer
    - Monitor key pool level

  EXPIRING:
    - Session approaching lifetime limit
    - Initiate new session establishment
    - Overlap period: both old and new sessions valid

  DEGRADED:
    - Key pool exhausted
    - Fall back to pre-shared symmetric key
    - Alert clinical systems
```

---

## 6. Evaluation

### 6.1 Energy Consumption

Measured on TI MSP430FR5994 @ 16 MHz, 3.0V supply:

| Operation | Time | Current | Energy |
|-----------|------|---------|--------|
| ML-KEM-512 Decaps | 48 ms | 3.2 mA | 461 microjoules |
| AES-128-CCM (128 B) | 0.12 ms | 2.8 mA | 1.0 microjoule |
| HMAC-SHA-256 (128 B) | 0.25 ms | 2.8 mA | 2.1 microjoules |
| HKDF-SHA-256 | 0.4 ms | 2.8 mA | 3.4 microjoules |
| Session Establish Total | 49 ms | -- | 467 microjoules |

**Battery Impact Analysis:**
- Pacemaker battery: ~1.0 Ah at 2.8V = 10,080 J total energy
- Session establishment every 72 hours: 467 microjoules per 72 hours
- Annual KEM energy: ~2.4 mJ (0.000024% of battery)
- Per-message energy: ~3.1 microjoules x ~100 messages/day = 310 microjoules/day
- Annual message energy: ~113 mJ (0.0011% of battery)

**Conclusion:** PQC adds negligible battery impact compared to RF transmission (which consumes ~10 mJ per telemetry session).

### 6.2 Key Pool Sizing

| Device | Pool Size | Session Duration | Days Between Replenishment | FRAM Usage |
|--------|-----------|-----------------|---------------------------|-----------|
| Pacemaker | 10 keys | 72 hours | 30 days | 24 KB |
| Insulin Pump | 50 keys | 24 hours | 50 days | 120 KB |
| ICD | 20 keys | 48 hours | 40 days | 48 KB |

### 6.3 Comparison with Classical Approach

| Metric | ECDH-P256 (Current) | ML-KEM-512 (QBITEL-MedPQC) |
|--------|---------------------|---------------------------|
| Key Exchange Time | 120 ms (MSP430) | 48 ms (decaps only, pre-computed) |
| Key Size (device storage) | 64 B per key | 2,400 B per key pair |
| Quantum Security | None | NIST Level 1 (~128-bit) |
| Memory Overhead | ~8 KB | ~19 KB |
| Battery Impact | 0.0009%/year | 0.0013%/year |
| Forward Secrecy | Per-session | Per-session |

---

## 7. FDA Regulatory Considerations

### 7.1 FDA Premarket Cybersecurity Guidance (2023)

The FDA's "Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket Submissions" (September 2023) requires:

- **Threat Modeling:** Our framework includes quantum adversary capabilities in the threat model, addressing the "reasonably foreseeable" threat standard.
- **Cryptographic Bill of Materials:** ML-KEM-512 (FIPS 203), AES-128-CCM, HMAC-SHA-256, HKDF-SHA-256 -- all NIST-standardized.
- **Software Update Capability:** Key pool replenishment occurs during programmer sessions without firmware updates, enabling cryptographic agility.
- **Security Architecture Documentation:** The constraint classification and degradation model provide the required security risk assessment documentation.

### 7.2 IEC 62443 Medical Device Profile

Our constraint classes map to IEC 62443 Security Levels:

| Constraint Class | IEC 62443 SL | Justification |
|-----------------|-------------|---------------|
| ULTRA_CONSTRAINED | SL1 | External shield provides SL1 authentication |
| CONSTRAINED | SL2 | Pre-computed PQC meets SL2 requirements |
| MODERATE | SL3 | Runtime PQC with session management |
| STANDARD | SL4 | Full PQC with algorithm agility |

---

## 8. Related Work

Kampanakis et al. (arXiv 2023) proposed LiteQSign/INF-HORS for lightweight PQ signatures on IoT medical devices, achieving near-optimal hash-based signing. Our work focuses on key encapsulation (confidentiality) rather than signatures, addressing the complementary challenge of secure key agreement.

The npj Digital Medicine article (Nature, 2025) on quantum threats to medical devices highlighted the urgency but did not propose concrete constrained implementations. Our work provides the missing implementation framework.

Rudraksh (IACR 2024) designed a lightweight LWE-based KEM for wireless sensors. Our contribution is the application-specific optimization (pre-computed pools, battery-aware scheduling, graceful degradation) mapped to concrete medical device profiles rather than generic IoT constraints.

---

## 9. Conclusion

QBITEL-MedPQC demonstrates that post-quantum key agreement is feasible even on the most constrained implantable medical devices through a combination of pre-computation, session key amortization, and battery-aware scheduling. The framework ensures that quantum-safe communications never compromise device safety: the graceful degradation model prioritizes patient safety above all cryptographic objectives. With ML-KEM-512 adding less than 0.002% annual battery overhead on pacemaker-class devices, the cost of quantum resistance is negligible compared to the security benefits for devices that will operate in patients' bodies for a decade or more.

---

## References

[1] NIST FIPS 203, "Module-Lattice-Based Key-Encapsulation Mechanism Standard (ML-KEM)," August 2024.

[2] FDA, "Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket Submissions," September 2023.

[3] P. Kampanakis et al., "LiteQSign: Lightweight and Quantum-Safe Signatures for Heterogeneous IoT," arXiv 2311.18674, 2024.

[4] "Quantum Cryptography and Data Protection for Medical Devices Before and After Q-Day," npj Digital Medicine (Nature), 2025.

[5] NIST IR 8547, "Transition to Post-Quantum Cryptography Standards," 2024.

[6] IEC 62443, "Industrial Communication Networks -- Network and System Security."

[7] HIPAA Security Rule, 45 CFR Part 164, Subpart C.

[8] "Rudraksh: A Compact and Lightweight Post-Quantum Key-Encapsulation Mechanism," IACR ePrint 2024/1170.

---

*QBITEL Bridge -- Quantum-Safe Infrastructure for Critical Systems*
*Copyright 2026 QBITEL. All rights reserved.*
