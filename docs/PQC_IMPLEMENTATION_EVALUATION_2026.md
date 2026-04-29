# QBITEL PQC Implementation Evaluation

Date: April 22, 2026

## Executive Summary

QBITEL has real PQC foundations, but the maturity is uneven.

The strongest current implementations are:

- `rust/dataplane/crates/pqc_tls/src/falcon.rs`
- `ai_engine/crypto/lms_xmss.py`
- the strict-mode guardrails in `ai_engine/crypto/mlkem.py` and `ai_engine/crypto/dilithium.py`

The main issues are in several custom, domain-specific modules that still use placeholder or simplified logic while being described like production cryptography:

- `ai_engine/domains/automotive/v2x_group_signatures.py`
- `ai_engine/domains/industrial/tesla_broadcast_auth.py`
- `ai_engine/domains/healthcare/verifiable_credentials.py`

Product implication: QBITEL should position PQC as a governed migration and protection workflow, not as a claim that every custom domain-specific PQC construction is production-ready today.

## What Looks Strong

### 1. Rust Falcon dataplane engine

File: `rust/dataplane/crates/pqc_tls/src/falcon.rs`

Why it stands out:

- Domain configs exist for automotive, aviation, and healthcare at lines `277-307`.
- Key generation uses `pqcrypto_falcon` directly at lines `385-403`.
- Signing uses concrete Falcon primitives at lines `455-473`.
- Verification uses concrete Falcon verification at lines `511-530`.

Assessment:

- This is one of the most credible PQC implementations in the repo.
- The remaining gap is mostly around enforcement of optional flags such as compression and FIPS mode, which currently behave more like metadata or advisory controls than strict policy.

### 2. LMS/XMSS stateful signatures

File: `ai_engine/crypto/lms_xmss.py`

Why it stands out:

- State persistence uses atomic `os.replace` at lines `252-260`.
- Locking exists at lines `264-274`.
- LMS advances and persists leaf state before signing at lines `701-782`.
- XMSS follows the same safety pattern at lines `1181-1254`.

Assessment:

- This shows strong operational discipline around stateful signatures.
- It is a good internal benchmark for what “production-aware” PQC code should look like elsewhere in the project.

## Guarded But Not Fully Safe By Default

### 3. ML-KEM wrapper

File: `ai_engine/crypto/mlkem.py`

Important behavior:

- Strict mode blocks fallback initialization at lines `192-193`.
- In fallback mode, key generation uses random bytes at lines `255-266`.
- Encapsulation fallback uses random ciphertext and random shared secrets at lines `311-315`.
- Decapsulation fallback returns a random shared secret at lines `359-363`.

Assessment:

- The wrapper is acceptable only if production paths always enforce strict mode and a real provider.
- The fallback path must remain test-only.

### 4. ML-DSA wrapper

File: `ai_engine/crypto/dilithium.py`

Important behavior:

- Strict mode blocks fallback initialization at lines `178-179`.
- Fallback key generation uses random bytes at lines `242-253`.
- Fallback signing creates random signatures at lines `297-301`.
- Fallback verification returns `True` at lines `342-346`.

Assessment:

- The strict-mode default is good.
- The fallback verification behavior is unsafe enough that it should never be reachable from a production-facing path.

## Findings: Domain-Specific Implementations

### 5. Automotive V2X group signatures are not a production group-signature implementation

File: `ai_engine/domains/automotive/v2x_group_signatures.py`

Key findings:

- Identity escrow keys are generated at lines `404-411`.
- `_create_identity_escrow()` does not perform real encryption; it uses a SHAKE256-based construction and explicitly notes “In production, use ML-KEM encapsulation + AES-GCM” at lines `486-510`.
- Per-signature escrow is also simplified and hash-based at lines `800-817`.
- Verification uses the group manager verification key directly at lines `1021-1036`, which is not a real group-signature verification flow.
- Revocation matching is inconsistent:
  - verifier-side check: lines `1041-1050`
  - update-side token generation: lines `1083-1087`

Assessment:

- This should not be marketed as production-ready anonymous group signatures with escrow and verifier-local revocation.
- It is better described as a prototype or applied research module.

### 6. Industrial TESLA++ commitment verification is effectively stubbed

File: `ai_engine/domains/industrial/tesla_broadcast_auth.py`

Key findings:

- Commitment signing uses ML-DSA at lines `533-545`.
- Receiver verification reconstructs the commitment data but does not verify against a trusted public key.
- The current logic returns `len(commitment.pqc_signature) > 0` at lines `748-781`.

Assessment:

- Any non-empty signature can be accepted.
- That breaks the trust anchor of the signed commitment chain and is unsafe for real deployment.

### 7. Healthcare verifiable credentials are prototype-grade

File: `ai_engine/domains/healthcare/verifiable_credentials.py`

Key findings:

- Merkle proofs are simplified to a hash of sibling values at lines `395-405`.
- Predicate proofs are a hash of a boolean plus random salt at lines `407-426`.
- Verifier logic checks trust registration and proof shape, but not actual issuer signature verification, inclusion proof correctness, or proof semantics at lines `469-497`.

Assessment:

- This is not a complete selective disclosure or zero-knowledge credential verification pipeline.
- It should be framed as a prototype design, not a production healthcare credential system.

## Product Positioning Advice

QBITEL should say:

- “We provide governed PQC migration and protection for legacy environments.”
- “We inventory crypto exposure, apply overlays where change is possible, and generate evidence packs.”
- “Our core PQC foundations are in place, with some domain-specific modules still being hardened.”

QBITEL should avoid saying:

- “All domain-specific PQC modules are production-ready.”
- “Custom group signatures, TESLA++, and verifiable credential flows are fully validated today.”

## Engineering Priorities

### Priority 1

- Enforce strict provider mode across every production-facing ML-KEM and ML-DSA path.
- Prevent insecure fallback providers from being reachable outside tests.

### Priority 2

- Replace placeholder verification in `tesla_broadcast_auth.py` with real PKI-backed verification.
- Decide whether `v2x_group_signatures.py` will become a real group-signature implementation or be re-scoped.
- Replace the placeholder proof system in `verifiable_credentials.py` with real proof and verification primitives.

### Priority 3

- Convert advisory domain configuration in the Rust Falcon engine into enforceable controls where needed.
- Add integration and negative-path tests that prove invalid signatures, invalid proofs, and missing trust anchors fail closed.

## Bottom Line

QBITEL already has enough substance to support a serious PQC story, but that story must be disciplined.

The credible message is:

- strong core primitives
- governed migration and overlay protection
- evidence-led rollout
- domain-specific cryptography under active hardening

That message is technically defensible today.
