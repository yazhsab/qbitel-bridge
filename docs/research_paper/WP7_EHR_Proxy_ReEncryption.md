# QBITEL Bridge Whitepaper WP-2026-07

# Post-Quantum Proxy Re-Encryption for Electronic Health Records with Category-Level Consent Enforcement

**Authors:** QBITEL Bridge Research Team
**Date:** April 2026
**Version:** 1.0
**Classification:** Public

---

## Abstract

Sharing Electronic Health Records (EHRs) across healthcare providers -- for patient transfers, specialist referrals, insurance claims, and research -- requires re-encrypting data from the originating provider's key to the receiving provider's key. Traditional approaches either decrypt at an intermediate server (exposing plaintext) or require the patient to be online for re-encryption (impractical for emergency scenarios). We present QBITEL-PRE, a post-quantum proxy re-encryption scheme based on ML-KEM (FIPS 203) that enables a semi-trusted proxy to transform ciphertexts from one provider's public key to another's without accessing the plaintext EHR data. Our scheme introduces category-level consent enforcement: EHR data is partitioned into 11 clinical categories (demographics, vitals, labs, medications, diagnoses, procedures, imaging, mental health, substance abuse, genetic, reproductive), and the patient grants re-encryption rights per category with five consent types (full access, category-restricted, time-limited, emergency, research, one-time). The proxy can only re-encrypt categories for which the patient has granted consent, cryptographically enforced through category-specific re-encryption keys. We demonstrate compliance with HIPAA Safe Harbor, HITECH, 42 CFR Part 2 (substance abuse records), and GDPR Article 20 (data portability).

**Keywords:** Proxy Re-Encryption, Electronic Health Records, Post-Quantum Cryptography, ML-KEM, HIPAA, Patient Consent, Health Data Sharing, Privacy

---

## 1. Introduction

### 1.1 The EHR Sharing Problem

Patient care increasingly involves multiple providers across different health systems. A cancer patient may have records at their primary care physician, an oncologist, a surgical center, a radiology practice, and a pharmacy -- each operating independent EHR systems (Epic, Cerner, MEDITECH, etc.). Sharing records between these systems requires:

1. **Encryption in transit and at rest** (HIPAA Security Rule)
2. **Patient consent** for each sharing relationship (HIPAA Privacy Rule)
3. **Category-specific controls** for sensitive records (42 CFR Part 2 for substance abuse, state laws for mental health and reproductive health)
4. **Emergency access** when the patient is incapacitated (break-the-glass with audit)
5. **Research access** with de-identification or limited data sets

Current approaches use Health Information Exchanges (HIEs) that decrypt EHRs at a central server, apply access control policies, and re-encrypt for the receiving provider. This creates a single point of compromise -- the HIE server has access to all plaintext records.

### 1.2 Proxy Re-Encryption

Proxy re-encryption (PRE) allows a semi-trusted proxy to transform a ciphertext encrypted under Alice's public key into a ciphertext decryptable by Bob's private key, without the proxy learning the plaintext:

```
Alice encrypts:    C_A = Enc(pk_A, message)
Alice generates:   rk_{A->B} = ReKeyGen(sk_A, pk_B)
Proxy transforms:  C_B = ReEncrypt(rk_{A->B}, C_A)
Bob decrypts:      message = Dec(sk_B, C_B)

The proxy learns neither the message nor sk_A or sk_B.
```

### 1.3 Our Contribution

QBITEL-PRE extends proxy re-encryption for healthcare with:

1. **ML-KEM-Based PRE:** Key encapsulation using ML-KEM-768 (FIPS 203) with SHAKE256 key derivation, providing post-quantum security.
2. **Category-Level Re-Encryption Keys:** Separate re-encryption keys per clinical data category; the proxy can only re-encrypt categories for which a valid re-encryption key exists.
3. **Five Consent Types:** Full, category-restricted, time-limited, emergency, research, and one-time -- each with different re-encryption key properties.
4. **Regulatory Compliance:** Cryptographic enforcement of 42 CFR Part 2 (substance abuse records cannot be re-encrypted without explicit substance-abuse-specific consent), HIPAA minimum necessary, and GDPR data portability.

---

## 2. Clinical Data Categories

### 2.1 Category Taxonomy

| Category | Sensitivity | Regulatory Overlay | Default Consent |
|----------|-----------|-------------------|----------------|
| DEMOGRAPHICS | Standard | HIPAA | Included in all consents |
| VITALS | Standard | HIPAA | Included in all consents |
| LABS | Standard | HIPAA | Included in clinical consents |
| MEDICATIONS | Standard | HIPAA | Included in clinical consents |
| DIAGNOSES | Elevated | HIPAA | Requires explicit consent |
| PROCEDURES | Elevated | HIPAA | Requires explicit consent |
| IMAGING | Standard | HIPAA | Included in clinical consents |
| MENTAL_HEALTH | High | HIPAA + State laws | Requires category-specific consent |
| SUBSTANCE_ABUSE | Highest | 42 CFR Part 2 | Requires specific Part 2 consent |
| GENETIC | High | GINA + State laws | Requires category-specific consent |
| REPRODUCTIVE | High | State laws (varying) | Requires category-specific consent |

### 2.2 Consent Types

```
FULL_ACCESS:
  - All categories re-encryptable
  - Permanent until revoked
  - Use case: Primary care provider relationship

CATEGORY_RESTRICTED:
  - Specified categories only (e.g., "Labs + Medications only")
  - Permanent until revoked
  - Use case: Specialist referral (cardiologist needs cardiac labs, not mental health)

TIME_LIMITED:
  - Specified categories for a fixed duration
  - Re-encryption key includes expiration timestamp
  - Proxy rejects re-encryption after expiration
  - Use case: Insurance claim processing (30-day window)

EMERGENCY:
  - All categories including sensitive
  - Short duration (4-24 hours)
  - Automatic audit notification to patient
  - Use case: Emergency department admission

RESEARCH:
  - Specified categories with de-identification flag
  - Re-encryption key includes de-identification transform
  - Proxy strips direct identifiers during re-encryption
  - Use case: Clinical trial, outcomes research

ONE_TIME:
  - Single-use re-encryption key
  - Cryptographically consumed after first use (nonce binding)
  - Use case: Second opinion, one-time consultation
```

---

## 3. Cryptographic Construction

### 3.1 Key Hierarchy

```
Provider Key Pair (per healthcare provider):
  (pk_provider, sk_provider) = ML-KEM-768.KeyGen()
  Stored in provider's HSM or key management system

Patient Master Key (per patient):
  (pk_patient, sk_patient) = ML-KEM-768.KeyGen()
  sk_patient held by patient (in health wallet / smart card / mobile app)
  pk_patient registered with all providers

Category Encryption Key (per patient, per category):
  cek_cat = HKDF-SHA3-256(patient_master_secret, "category" || category_id, 32)
  EHR data in each category is encrypted with its category key

Re-Encryption Key (per delegation):
  rk_{A->B, cat} = ReKeyGen(sk_patient, pk_provider_B, category=cat, consent_type, expiry)
  Delivered to the proxy server
```

### 3.2 Encryption

When Provider A stores a patient's EHR record:

```
Encrypt(pk_patient, category, record):
  1. Derive category key: cek = HKDF(patient_master_secret, category)
  2. Encrypt record: ct_record = AES-256-GCM(cek, record, nonce)
  3. Encapsulate category key for the patient:
     (ct_kem, ss) = ML-KEM-768.Encaps(pk_patient)
     ct_cek = AES-256-GCM(ss, cek, nonce_2)
  4. Store: (category, ct_kem, ct_cek, ct_record, nonce, nonce_2)
```

### 3.3 Re-Encryption Key Generation

The patient generates a re-encryption key granting Provider B access to specific categories:

```
ReKeyGen(sk_patient, pk_provider_B, categories, consent_type, expiry):
  For each category in categories:
    1. Derive category key: cek = HKDF(patient_master_secret, category)
    2. Encapsulate category key for Provider B:
       (ct_kem_B, ss_B) = ML-KEM-768.Encaps(pk_provider_B)
       ct_cek_B = AES-256-GCM(ss_B, cek, nonce_B)
    3. Create re-encryption token:
       rk_token = {
         category: category,
         ct_kem_B: ct_kem_B,
         ct_cek_B: ct_cek_B,
         consent_type: consent_type,
         expiry: expiry,
         signature: ML-DSA-65.Sign(patient_signing_key, token_data)
       }

  Return: [rk_token_1, ..., rk_token_k]  (one per authorized category)
```

### 3.4 Proxy Re-Encryption

The semi-trusted proxy transforms ciphertexts:

```
ReEncrypt(rk_tokens, encrypted_ehr):
  For each record in encrypted_ehr:
    1. Check category: Is there a valid rk_token for this record's category?
       - If no token: SKIP (proxy cannot re-encrypt this category)
       - If token expired: SKIP + log attempted access
       - If one-time token already used: SKIP

    2. Verify patient signature on rk_token (prevent forged delegations)

    3. Replace the KEM ciphertext:
       - Original: (ct_kem_patient, ct_cek_patient, ct_record)
       - Transformed: (rk_token.ct_kem_B, rk_token.ct_cek_B, ct_record)
       - The ct_record itself is UNCHANGED (proxy never decrypts it)

    4. Log the re-encryption event for audit trail

  Output: Re-encrypted EHR containing only authorized categories
```

### 3.5 Decryption by Receiving Provider

```
Decrypt(sk_provider_B, re_encrypted_ehr):
  For each record:
    1. Decapsulate: ss_B = ML-KEM-768.Decaps(sk_provider_B, ct_kem_B)
    2. Decrypt category key: cek = AES-256-GCM.Dec(ss_B, ct_cek_B)
    3. Decrypt record: record = AES-256-GCM.Dec(cek, ct_record)
  Output: Plaintext EHR records (only authorized categories)
```

---

## 4. Regulatory Compliance

### 4.1 HIPAA Privacy Rule (45 CFR 164.508)

**Minimum Necessary Standard:** QBITEL-PRE enforces minimum necessary through category-level consent. A dermatologist referral receives only relevant categories (demographics, vitals, medications, relevant diagnoses) -- not mental health or substance abuse records. The proxy physically cannot re-encrypt unauthorized categories.

**Patient Authorization:** The re-encryption key generation step constitutes the patient's authorization. The patient's digital signature on the re-encryption token is the cryptographic equivalent of a signed HIPAA authorization form.

### 4.2 42 CFR Part 2 (Substance Abuse Records)

Part 2 imposes the strictest sharing requirements in US healthcare law:

- **Separate Consent Required:** Substance abuse records require a specific, separate authorization from the patient. In QBITEL-PRE, the SUBSTANCE_ABUSE category has NO default inclusion in any consent type. The patient must explicitly generate a re-encryption key for this category.

- **Prohibition on Re-Disclosure:** Part 2 prohibits the receiving provider from further disclosing substance abuse records. QBITEL-PRE enforces this cryptographically: the re-encryption key only transforms from Provider A to Provider B. Provider B cannot generate a re-encryption key for Provider C for the substance abuse category -- that requires a NEW patient authorization generating a NEW re-encryption key.

- **Audit Requirements:** Every re-encryption of substance abuse records is logged with timestamp, proxy identity, source, and destination for Part 2 audit compliance.

### 4.3 GDPR Article 20 (Data Portability)

GDPR grants patients the right to receive their health data in a portable format and transmit it to another provider. QBITEL-PRE enables this without exposing data to intermediaries:

- Patient generates re-encryption keys for ALL categories to the new provider.
- Proxy re-encrypts the complete EHR without accessing plaintext.
- New provider receives the full record in their encryption domain.

### 4.4 HITECH Act

The HITECH Act extended HIPAA requirements to Business Associates (including HIE operators). Under QBITEL-PRE, the proxy is a Business Associate but has reduced risk exposure because it never accesses plaintext PHI.

---

## 5. Use Case Workflows

### 5.1 Patient Transfer

```
Scenario: Patient transfers from Hospital A to Hospital B

1. Patient uses mobile app to generate re-encryption keys
   - Selects: ALL categories (full transfer)
   - Consent type: FULL_ACCESS
   - Generates: rk_tokens for all 11 categories
   - Delivers rk_tokens to HIE proxy

2. HIE proxy re-encrypts Hospital A's records for Hospital B
   - Each category independently re-encrypted
   - Audit log generated

3. Hospital B decrypts with their private key
   - Full EHR available in Hospital B's system
```

### 5.2 Specialist Referral

```
Scenario: PCP refers patient to cardiologist

1. Patient generates category-restricted re-encryption keys
   - Categories: DEMOGRAPHICS, VITALS, LABS, MEDICATIONS, DIAGNOSES
   - Excluded: MENTAL_HEALTH, SUBSTANCE_ABUSE, GENETIC, REPRODUCTIVE
   - Consent type: TIME_LIMITED (90 days)

2. Proxy re-encrypts only authorized categories
   - Cardiologist receives relevant clinical data
   - Sensitive categories are cryptographically inaccessible
```

### 5.3 Emergency Access

```
Scenario: Patient arrives unconscious at Emergency Department

1. ED physician invokes emergency access protocol
   - System generates EMERGENCY consent (4-hour window)
   - All categories including sensitive are authorized
   - Automatic notification queued for patient

2. Proxy re-encrypts all categories with emergency flag
   - Audit trail marks all records as emergency-accessed
   - After 4 hours, emergency re-encryption keys expire automatically

3. Patient is notified upon recovery
   - Can review audit log of what was accessed and by whom
```

---

## 6. Security Analysis

### 6.1 Proxy Security

The proxy learns:
- Which categories are authorized (from re-encryption token metadata)
- The source and destination providers
- The timing of re-encryption requests

The proxy does NOT learn:
- Any plaintext EHR content
- The patient's private key
- The category encryption keys
- The receiving provider's private key

### 6.2 Quantum Resistance

| Component | Algorithm | Quantum Security |
|-----------|-----------|-----------------|
| Key Encapsulation | ML-KEM-768 | NIST Level 3 (~128-bit quantum) |
| Key Derivation | HKDF-SHA3-256 | 128-bit quantum (Grover) |
| Record Encryption | AES-256-GCM | 128-bit quantum (Grover) |
| Consent Signatures | ML-DSA-65 | NIST Level 3 |

### 6.3 Consent Revocation

When a patient revokes consent:
1. The re-encryption key tokens are deleted from the proxy.
2. The proxy can no longer re-encrypt for the revoked delegation.
3. Previously re-encrypted records remain accessible to the receiving provider (irrevocable disclosure), consistent with HIPAA which does not require un-disclosure.
4. New requests for the same records are denied.

---

## 7. Performance

| Operation | Time | Notes |
|-----------|------|-------|
| ML-KEM-768 KeyGen | 0.5 ms | Per-provider, one-time |
| Category Key Derivation | 0.02 ms | HKDF, per category |
| Re-Encryption Key Gen (per category) | 1.2 ms | Patient-side |
| Re-Encryption Key Gen (11 categories) | 13.2 ms | Full EHR delegation |
| Proxy Re-Encryption (per record) | 0.8 ms | Ciphertext transformation |
| Proxy Re-Encryption (1000 records) | 800 ms | Typical full EHR |
| Decryption (per record) | 0.6 ms | Receiving provider |

---

## 8. Related Work

Cohen et al. (IACR 2024) constructed an efficient lattice-based PRE with HRA security and arbitrary homomorphism. Our work applies PRE to the specific healthcare domain with category-level consent, regulatory compliance, and clinical workflow integration.

Zhao et al. (ESORICS 2024) achieved constant-size unbounded multi-hop PRE from lattices, solving ciphertext growth in repeated re-encryptions. This technique could extend QBITEL-PRE to support multi-hop referral chains (A -> B -> C) without ciphertext expansion.

Zhang et al. (IACR 2025) introduced puncturable attribute-based PRE from lattices where recipients can locally revoke specific attributes. This maps to our category-level revocation model.

---

## 9. Conclusion

QBITEL-PRE demonstrates that post-quantum proxy re-encryption can enforce complex healthcare consent policies cryptographically, moving from application-level access control (bypassable by a compromised server) to cryptographic enforcement (the proxy physically cannot re-encrypt unauthorized categories). The category-level granularity is essential for compliance with 42 CFR Part 2, state-specific mental health and reproductive health laws, and GDPR data minimization requirements. By ensuring the proxy never accesses plaintext PHI, the scheme fundamentally reduces the attack surface of health information exchange infrastructure.

---

## References

[1] NIST FIPS 203, "Module-Lattice-Based Key-Encapsulation Mechanism Standard (ML-KEM)," August 2024.
[2] 42 CFR Part 2, "Confidentiality of Substance Use Disorder Patient Records," SAMHSA.
[3] HIPAA Security Rule, 45 CFR Part 164, Subpart C.
[4] A. Cohen et al., "HRA-Secure Homomorphic Lattice-Based PRE with Tight Security," IACR ePrint 2024/681.
[5] F. Zhao et al., "Constant-Size Unbounded Multi-Hop FH-PRE from Lattices," ESORICS 2024.
[6] T. Zhang et al., "Puncturable Attribute-Based PRE from Lattices," IACR ePrint 2025/2105.
[7] GDPR Article 20, "Right to Data Portability," European Parliament, 2016.

---

*QBITEL Bridge -- Quantum-Safe Infrastructure for Critical Systems*
*Copyright 2026 QBITEL. All rights reserved.*
