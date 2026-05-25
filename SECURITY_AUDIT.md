# QBITEL Bridge — Security Audit Status

## Crypto Module Inventory

| Module | Algorithm | Provider (Production) | Provider (Dev) | Maturity | Constant-Time |
|--------|-----------|----------------------|----------------|----------|---------------|
| `crypto/mlkem.py` | ML-KEM (FIPS 203) | liboqs, pqcrypto | kyber-py | GA | liboqs: Yes, kyber-py: **No** |
| `crypto/dilithium.py` | ML-DSA (FIPS 204) | liboqs, pqcrypto | dilithium-py | GA | liboqs: Yes, dilithium-py: **No** |
| `crypto/falcon.py` | Falcon | liboqs, pqcrypto | — | GA | liboqs: Yes (C reference) |
| `crypto/hybrid.py` | X25519/P384 + ML-KEM | cryptography + liboqs | cryptography + kyber-py | GA | Classical: Yes, PQC: depends on provider |
| `crypto/lms_xmss.py` | LMS/XMSS (SP 800-208) | Pure Python | Pure Python | GA | **No** (pure Python) |
| `crypto/pqc_unified.py` | Unified interface | Delegates to above | Delegates to above | GA | Depends on backend |
| `crypto/agility.py` | Algorithm negotiation | N/A (no crypto ops) | N/A | GA | N/A |
| `crypto/quantum_threat_scoring.py` | Risk scoring | N/A (deterministic) | N/A | GA | N/A |
| `crypto/zkp.py` | Zero-Knowledge Proofs | Pure Python | Pure Python | Preview | **No** |
| `crypto/vrf.py` | Verifiable Random Functions | Pure Python | Pure Python | Preview | **No** |
| `crypto/threshold.py` | Threshold Signatures | Pure Python | Pure Python | Preview | **No** |

## Domain-Specific Crypto Modules

| Module | Use Case | Maturity | Known Limitations |
|--------|----------|----------|-------------------|
| `domains/healthcare/homomorphic_vitals.py` | Privacy-preserving vital sign analytics | **EXPERIMENTAL** | Simplified Paillier-like HE, NOT a production FHE library |
| `domains/industrial/verifiable_delay.py` | Safety system timing proofs | **EXPERIMENTAL** | Iterated SHA3-256, NOT a proper VDF (no RSA/class group) |
| `domains/automotive/v2x_group_signatures.py` | V2X pseudonymous auth | Preview | Lattice-based group sigs, needs formal verification |
| `domains/aviation/forward_secure_channels.py` | ATC forward secrecy | Preview | Ephemeral ML-KEM, bandwidth constraints tested |
| `domains/banking/security/swift_proxy_reencryption.py` | Correspondent banking | GA | Proxy re-encryption scheme is custom, not standardized |
| `domains/banking/security/regulatory_proof_engine.py` | ZK compliance proofs | GA | Simplified ZKP, not full SNARKs/STARKs |

## Known Limitations

### Critical (P0)

1. **Pure-Python PQC providers are NOT constant-time.** `kyber-py` and `dilithium-py` are reference implementations vulnerable to timing side-channels. Production deployments MUST use `liboqs` (C) or `pqcrypto` (Rust). The provider registry (`crypto/providers.py`) enforces this preference chain and emits `DeprecationWarning` when dev-only providers are used.

2. **LMS/XMSS state management.** State reuse completely breaks security. The `InMemoryStateBackend` loses state on restart — production MUST use `FileStateBackend` with durable storage and backup. No crash-recovery mechanism exists if a process dies between signing and state persistence.

### High (P1)

3. **Homomorphic encryption is simulated.** `homomorphic_vitals.py` uses a simplified Paillier-like scheme, not a production FHE library (SEAL, OpenFHE, python-paillier). Gated behind `QBITEL_ALLOW_EXPERIMENTAL_CRYPTO=1`.

4. **VDF uses iterated hashing.** `verifiable_delay.py` uses iterated SHA3-256, which does not satisfy formal VDF properties (no group of unknown order). Gated behind `QBITEL_ALLOW_EXPERIMENTAL_CRYPTO=1`.

5. **No formal verification** for safety-critical domains (automotive SIL, aviation DO-326A, industrial SIL 2/3). Domain modules are functionally correct but lack formal proofs required for certification.

### Medium (P2)

6. **Custom cryptographic constructions.** Several domain modules (proxy re-encryption, regulatory ZKP, group signatures) use custom constructions that have not been peer-reviewed or formally analyzed.

7. **Fallback mode returns random bytes.** When `QBITEL_PQC_ALLOW_FALLBACK=1` is set AND no provider is available, operations return random bytes with zero cryptographic value. This is logged at CRITICAL level but could be overlooked.

## Provider Preference Chain

```
liboqs (C bindings)       ← PRODUCTION tier, constant-time
    ↓ not available
pqcrypto (Rust FFI)       ← PRODUCTION tier, constant-time
    ↓ not available
kyber-py / dilithium-py   ← DEV_ONLY tier, NOT constant-time, DeprecationWarning emitted
    ↓ not available
FALLBACK (random bytes)   ← UNSAFE tier, CRITICAL log, blocked in strict mode (default)
```

## Recommendations for Formal Audit

1. **Prioritize liboqs integration** — ensure `liboqs-python` is installed in all production deployments
2. **Audit LMS/XMSS state management** — verify `FileStateBackend` atomic write guarantees
3. **Replace experimental crypto** — integrate `python-paillier` for HE, formal VDF library for timing proofs
4. **Formal verification** — engage a third party for DO-326A (aviation) and IEC 61508 (industrial) certification
5. **Penetration testing** — timing side-channel analysis of crypto operations under production load
6. **FIPS 140-3 validation** — if targeting US government, validate the crypto module boundary

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `QBITEL_PQC_ALLOW_FALLBACK` | Allow insecure test fallback | `0` (disabled) |
| `QBITEL_ALLOW_EXPERIMENTAL_CRYPTO` | Enable experimental HE/VDF modules | `0` (disabled) |

## Last Updated

2026-03-25 — Initial security audit documentation created as part of platform hardening.
