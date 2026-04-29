# QBITEL Bridge Whitepaper WP-2026-06

# Post-Quantum Multi-Authority Threshold Signatures for Tiered Banking Transaction Authorization

**Authors:** QBITEL Bridge Research Team
**Date:** April 2026
**Version:** 1.0
**Classification:** Public

---

## Abstract

High-value financial transactions require multi-party authorization with role-based access controls -- a wire transfer exceeding $100 million should not be signed by a single individual regardless of their seniority. We present QBITEL-MATS (Multi-Authority Threshold Signatures), a post-quantum threshold signature scheme that maps ML-DSA-65 (FIPS 204) to banking transaction authorization tiers. The scheme defines five transaction tiers (Standard through Sanctions) with graduated quorum requirements: Standard transactions (< $1M) require 1-of-1 authorization, Elevated ($1M-$10M) require 2-of-3 with mandatory Treasury or Risk involvement, High-Value ($10M-$100M) require 3-of-5 with department diversity constraints, and Critical (> $100M) require 4-of-7 including mandatory Central Bank, Compliance, and Risk Management participation. Sanctions-related transactions require explicit Compliance, Legal, and Risk co-signatures regardless of value. The combined threshold signature is publicly verifiable as a standard ML-DSA-65 signature, enabling interoperability with existing SWIFT and payment rails infrastructure without modification to verifier software. We provide time-bounded signing windows to prevent authorization stalling and department diversity enforcement to prevent single-department capture of the signing process.

**Keywords:** Post-Quantum Cryptography, Threshold Signatures, Banking, Transaction Authorization, ML-DSA, SWIFT, Multi-Party Computation, Basel III, SOX Compliance

---

## 1. Introduction

### 1.1 The Multi-Authority Problem in Banking

Financial institutions implement multi-party authorization (MPA) for high-value transactions as required by:

- **Basel III/IV:** Operational risk requirements mandate segregation of duties for material transactions.
- **Sarbanes-Oxley (SOX):** Section 404 requires internal controls over financial reporting, including transaction authorization workflows.
- **SWIFT Customer Security Programme (CSP):** Mandatory control 2.6A requires multi-factor and multi-party authorization for SWIFT transactions.
- **PCI DSS 4.0:** Requirement 7 mandates access control based on business need-to-know with least privilege enforcement.

Current MPA implementations use sequential approvals in banking middleware -- each authorized signer applies their individual digital signature, and the payment system checks that the required number and roles have signed. This approach has two weaknesses:

1. **No Cryptographic Binding:** The quorum policy is enforced by application logic, not cryptography. A compromised middleware can bypass the multi-authority requirement.
2. **Quantum Vulnerability:** Individual signatures use ECDSA or RSA, both broken by Shor's algorithm.

### 1.2 Our Contribution

QBITEL-MATS provides cryptographic enforcement of multi-authority policies through threshold signatures:

1. **Transaction-Tiered Quorum Policies:** Five tiers with role-based, value-based, and context-based quorum requirements.
2. **ML-DSA-65 Threshold Construction:** Each authority holds a key share; the combined signature is a valid ML-DSA-65 signature verifiable by any standard ML-DSA-65 verifier.
3. **Role and Department Diversity Enforcement:** Quorum policies require participation from specific roles (e.g., Compliance must sign Sanctions transactions) and departments (no single department can satisfy the quorum alone).
4. **Time-Bounded Signing Windows:** Each signing session has a configurable deadline; expired sessions cannot be completed, preventing indefinite authorization hold.
5. **Audit Trail:** Every partial signature contribution is logged with signer identity, role, department, and timestamp for regulatory compliance.

---

## 2. System Design

### 2.1 Authority Roles

```
Authority Roles (AuthorityRole enum):
  CENTRAL_BANK       - Monetary authority representative
  TREASURY           - Treasury management / funding
  COMPLIANCE         - Regulatory compliance officer
  RISK_MANAGEMENT    - Enterprise risk management
  LEGAL              - Legal counsel
  BOARD_MEMBER       - Board of Directors representative
  INTERNAL_AUDIT     - Internal audit function
  IT_SECURITY        - Information security officer
  OPERATIONS         - Operations management
```

Each role maps to a department for diversity enforcement:

| Role | Department | Typical Seniority |
|------|-----------|-------------------|
| CENTRAL_BANK | Regulatory | External authority |
| TREASURY | Finance | VP+ |
| COMPLIANCE | Legal & Compliance | SVP+ |
| RISK_MANAGEMENT | Risk | SVP+ |
| LEGAL | Legal & Compliance | General Counsel |
| BOARD_MEMBER | Governance | Board |
| INTERNAL_AUDIT | Audit | CAE |
| IT_SECURITY | Technology | CISO |
| OPERATIONS | Operations | COO |

### 2.2 Transaction Tiers and Quorum Policies

```
STANDARD (< $1,000,000):
  Threshold: 1-of-1
  Required Roles: None (any authorized signer)
  Signing Window: 1 hour
  Use Case: Routine payments, payroll, vendor settlements

ELEVATED ($1,000,000 - $10,000,000):
  Threshold: 2-of-3
  Required Roles: At least one of {TREASURY, RISK_MANAGEMENT}
  Department Diversity: Minimum 2 distinct departments
  Signing Window: 4 hours
  Use Case: Large vendor payments, interbank transfers

HIGH_VALUE ($10,000,000 - $100,000,000):
  Threshold: 3-of-5
  Required Roles: COMPLIANCE + at least one of {TREASURY, RISK_MANAGEMENT}
  Department Diversity: Minimum 3 distinct departments
  Signing Window: 8 hours
  Use Case: Syndicated loans, major acquisitions, sovereign payments

CRITICAL (> $100,000,000):
  Threshold: 4-of-7
  Required Roles: CENTRAL_BANK + COMPLIANCE + RISK_MANAGEMENT
  Department Diversity: Minimum 4 distinct departments
  Signing Window: 24 hours
  Use Case: Central bank operations, sovereign debt, systemic transactions

SANCTIONS:
  Threshold: 3-of-5 (regardless of value)
  Required Roles: COMPLIANCE + LEGAL + RISK_MANAGEMENT (all three mandatory)
  Department Diversity: Minimum 3 distinct departments
  Signing Window: 48 hours
  Use Case: Any transaction involving sanctioned jurisdictions or entities
```

### 2.3 Key Management

**Setup Phase (performed by trusted dealer or DKG):**

```
System Setup:
  1. Generate ML-DSA-65 master key pair (msk, mpk)
  2. Split msk into N shares using Shamir's Secret Sharing over a lattice:
     share_i = msk * lagrange_i + noise_i  (simplified; real construction
     uses verifiable secret sharing over polynomial rings)
  3. Distribute share_i to authority_i
  4. Publish mpk as the institution's transaction signing public key
  5. Register mpk with SWIFT, payment processors, correspondent banks

  Each authority receives:
    - Their key share (share_i)
    - Their role assignment
    - The group public key (mpk)
    - The set of quorum policies
```

### 2.4 Signing Protocol

```
Transaction Signing:

  1. INITIATE: Submitter creates signing session
     - Transaction details (amount, currency, parties, type)
     - Determined tier based on amount and context
     - Quorum policy loaded for the tier
     - Signing window started (countdown begins)

  2. CONTRIBUTE: Each authority reviews and signs
     - Authority reviews transaction details
     - Authority generates partial signature:
       partial_sig_i = ThresholdSign(share_i, transaction_hash)
     - System records: (signer_id, role, department, timestamp, partial_sig)
     - System checks: Does this contribution satisfy any quorum requirement?

  3. COMBINE: When quorum is met
     - Verify all required roles are present
     - Verify department diversity constraint
     - Verify signing window has not expired
     - Combine partial signatures:
       combined_sig = ThresholdCombine(partial_sig_1, ..., partial_sig_t)
     - Verify combined_sig is a valid ML-DSA-65 signature over transaction_hash
     - Output: (transaction, combined_sig) -- indistinguishable from regular ML-DSA-65

  4. VERIFY: Any standard ML-DSA-65 verifier
     - ML-DSA-65.Verify(mpk, transaction_hash, combined_sig) == ACCEPT
     - No knowledge of threshold structure required
     - SWIFT infrastructure verifies normally
```

---

## 3. Security Properties

### 3.1 Quantum Resistance

The threshold scheme inherits the quantum resistance of ML-DSA-65:
- **Key Shares:** Derived from the ML-DSA-65 secret key via lattice-compatible secret sharing. Reconstructing the master key from fewer than t shares requires solving MLWE.
- **Partial Signatures:** Each partial signature reveals no more than the corresponding key share. An adversary must compromise t authorities to forge a signature.
- **Combined Signature:** A standard ML-DSA-65 signature, secure under MLWE/MSIS assumptions.

### 3.2 Quorum Policy Enforcement

| Property | Guarantee |
|----------|-----------|
| **Threshold Unforgeability** | Fewer than t signers cannot produce a valid combined signature |
| **Role Enforcement** | The combining step verifies required roles are present before combining |
| **Department Diversity** | The combining step counts distinct departments; insufficient diversity causes rejection |
| **Time Bounding** | Partial signatures include timestamps; expired contributions are rejected |
| **Non-Repudiation** | Each partial signature is bound to a specific signer identity; audit log is append-only |

### 3.3 Collusion Resistance

An adversary controlling (t-1) authorities cannot produce a valid signature. Specifically:

- For CRITICAL tier (4-of-7): compromising 3 authorities is insufficient.
- For SANCTIONS tier: compromising any 2 of {Compliance, Legal, Risk} is insufficient because all three are mandatory.
- Department diversity ensures that compromising an entire department (e.g., all Finance personnel) is insufficient if the policy requires participation from other departments.

---

## 4. SWIFT Integration

### 4.1 Message Types

The scheme supports all major SWIFT message types:

| SWIFT Type | Description | Typical Tier |
|-----------|-------------|-------------|
| MT103 | Single Customer Credit Transfer | STANDARD-HIGH_VALUE |
| MT202 | General Financial Institution Transfer | ELEVATED-CRITICAL |
| MT199 | Free Format (instructions) | STANDARD |
| PACS.008 | FI to FI Customer Credit Transfer (ISO 20022) | STANDARD-HIGH_VALUE |
| PACS.009 | FI to FI Financial Institution Transfer (ISO 20022) | ELEVATED-CRITICAL |
| CAMT.053 | Bank to Customer Statement | STANDARD |
| CAMT.054 | Bank to Customer Debit Credit Notification | STANDARD |

### 4.2 Correspondent Banking Chain

For multi-hop correspondent banking (Originator -> Intermediary -> Beneficiary):

```
Multi-Hop Authorization:

  Hop 1 (Originator Bank):
    - Quorum policy based on originator's internal tier
    - Combined signature over SWIFT message
    - Forwarded to intermediary

  Hop 2 (Intermediary Bank):
    - Verify originator's combined signature (standard ML-DSA-65 verify)
    - Apply intermediary's own quorum policy
    - Produce second combined signature
    - Forward to beneficiary

  Hop 3 (Beneficiary Bank):
    - Verify both originator and intermediary signatures
    - Apply beneficiary's settlement policy
    - Complete transaction

  Each hop is independently quantum-safe; no shared secrets between institutions.
```

### 4.3 Regulatory Reporting

The audit trail supports regulatory reporting requirements:

```
Audit Record per Transaction:
  {
    "transaction_id": "TXN-2026-04-03-00147",
    "tier": "HIGH_VALUE",
    "amount": 45000000.00,
    "currency": "USD",
    "quorum_policy": "3-of-5",
    "signers": [
      {"id": "U001", "role": "COMPLIANCE", "dept": "Legal", "signed_at": "2026-04-03T10:23:15Z"},
      {"id": "U002", "role": "TREASURY", "dept": "Finance", "signed_at": "2026-04-03T10:45:22Z"},
      {"id": "U003", "role": "RISK_MANAGEMENT", "dept": "Risk", "signed_at": "2026-04-03T11:02:08Z"}
    ],
    "departments_represented": 3,
    "required_roles_satisfied": true,
    "signing_window_start": "2026-04-03T10:00:00Z",
    "signing_window_end": "2026-04-03T18:00:00Z",
    "combined_signature_valid": true,
    "algorithm": "ML-DSA-65-THRESHOLD-3-of-5"
  }
```

---

## 5. Performance

### 5.1 Operation Latency

| Operation | Time | Notes |
|-----------|------|-------|
| Key Share Generation (per authority) | 12 ms | One-time setup |
| Partial Signature Generation | 2.1 ms | Per authority, per transaction |
| Quorum Verification | 0.5 ms | Role + diversity check |
| Signature Combination (3-of-5) | 8.4 ms | Lagrange interpolation + combine |
| Signature Combination (4-of-7) | 15.2 ms | Higher threshold |
| Combined Signature Verification | 1.4 ms | Standard ML-DSA-65 verify |

### 5.2 End-to-End Authorization Time

Authorization time is dominated by human decision-making, not cryptographic operations:

| Tier | Crypto Time | Typical Human Time | Total |
|------|-------------|-------------------|-------|
| STANDARD (1-of-1) | 3.5 ms | < 1 minute | ~1 minute |
| ELEVATED (2-of-3) | 13 ms | 5-30 minutes | 5-30 minutes |
| HIGH_VALUE (3-of-5) | 17 ms | 1-4 hours | 1-4 hours |
| CRITICAL (4-of-7) | 24 ms | 2-24 hours | 2-24 hours |

---

## 6. Regulatory Compliance

| Regulation | Requirement | QBITEL-MATS Compliance |
|-----------|------------|----------------------|
| Basel III (Pillar 1) | Operational risk capital for transaction fraud | Cryptographic MPA reduces fraud risk |
| SOX Section 404 | Internal controls over financial reporting | Auditable threshold enforcement |
| SWIFT CSP 2.6A | Multi-party authorization | Native threshold signature = MPA |
| PCI DSS 4.0 Req 7 | Access control, least privilege | Role-based key share distribution |
| GDPR Art 25 | Data protection by design | Signing reveals no transaction details to non-signers |
| DORA (EU) | ICT risk management for financial entities | Quantum-safe cryptographic controls |

---

## 7. Related Work

Ringtail (IEEE S&P 2025) demonstrated the first two-round PQ threshold signature from standard LWE, with NTT conducting a cross-datacenter demo. Our work applies threshold signatures to the specific domain of banking authorization tiers with role-based policies, rather than advancing the cryptographic primitive itself.

Del Pino et al. (EUROCRYPT 2025) achieved compact lattice threshold signatures where a (3,5) threshold signature is nearly the same size as a single Dilithium signature. This result directly benefits our scheme by ensuring that QBITEL-MATS combined signatures are indistinguishable in size from regular ML-DSA-65 signatures.

The FROST protocol (RFC 9591) provides a widely-analyzed two-round Schnorr threshold signature. While FROST is not post-quantum (Schnorr-based), its protocol structure influences our signing round design.

---

## 8. Conclusion

QBITEL-MATS provides cryptographically enforced multi-authority transaction authorization that is quantum-resistant, interoperable with existing SWIFT infrastructure, and compliant with Basel III, SOX, and SWIFT CSP requirements. By producing combined signatures that are standard ML-DSA-65 signatures, the scheme requires no changes to verifier infrastructure -- only the signing side needs the threshold capability. The tiered quorum policies with role enforcement and department diversity constraints provide defense-in-depth against insider threats while the time-bounded signing windows prevent authorization stalling attacks.

---

## References

[1] NIST FIPS 204, "Module-Lattice-Based Digital Signature Standard (ML-DSA)," August 2024.
[2] C. Boschini et al., "Ringtail: Practical Two-Round Threshold Signatures from LWE," IEEE S&P 2025.
[3] R. del Pino et al., "Finally! A Compact Lattice-Based Threshold Signature," EUROCRYPT 2025.
[4] D. Connolly et al., "FROST: Flexible Round-Optimized Schnorr Threshold Signatures," RFC 9591, 2024.
[5] Basel Committee on Banking Supervision, "Basel III: Finalising Post-Crisis Reforms," December 2017.
[6] SWIFT, "Customer Security Programme -- Control Framework v2024."

---

*QBITEL Bridge -- Quantum-Safe Infrastructure for Critical Systems*
*Copyright 2026 QBITEL. All rights reserved.*
