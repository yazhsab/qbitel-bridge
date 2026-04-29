export type AuditStatus = 'strong' | 'guarded' | 'prototype' | 'unsafe';
export type AuditDomain =
  | 'platform'
  | 'stateful-signatures'
  | 'automotive'
  | 'industrial'
  | 'healthcare'
  | 'rust-dataplane';

export type PqcAuditItem = {
  id: string;
  label: string;
  domain: AuditDomain;
  status: AuditStatus;
  summary: string;
  file: string;
  lines: string;
  highlights: string[];
  recommendations: string[];
};

export const auditStatusCopy: Record<
  AuditStatus,
  { label: string; description: string }
> = {
  strong: {
    label: 'Strong foundation',
    description: 'Cryptographic operations are real and the implementation has credible safety controls.',
  },
  guarded: {
    label: 'Guarded core',
    description: 'The implementation is directionally correct but still exposes unsafe paths that must stay out of production.',
  },
  prototype: {
    label: 'Prototype',
    description: 'Important parts of the flow are simplified or rely on placeholders rather than complete cryptographic verification.',
  },
  unsafe: {
    label: 'Unsafe as implemented',
    description: 'The current logic accepts invalid states or uses a construction that does not deliver the advertised security property.',
  },
};

export const pqcAuditItems: PqcAuditItem[] = [
  {
    id: 'mlkem-wrapper',
    label: 'ML-KEM core wrapper',
    domain: 'platform',
    status: 'guarded',
    summary:
      'The ML-KEM engine fails closed in strict mode, but it still exposes explicit insecure random fallbacks when strict mode is disabled.',
    file: 'ai_engine/crypto/mlkem.py',
    lines: '192-193, 255-266, 311-315, 359-363',
    highlights: [
      'Strict mode blocks fallback initialization when no provider is available.',
      'Fallback key generation and shared-secret generation are intentionally insecure.',
      'Safe enough as a wrapper only if production always enforces strict mode.',
    ],
    recommendations: [
      'Keep strict mode on for every production path.',
      'Make fallback unavailable outside tests and developer fixtures.',
    ],
  },
  {
    id: 'dilithium-wrapper',
    label: 'ML-DSA core wrapper',
    domain: 'platform',
    status: 'guarded',
    summary:
      'The signature wrapper has the right fail-closed default, but its non-strict fallback signs with random bytes and verify returns true.',
    file: 'ai_engine/crypto/dilithium.py',
    lines: '178-179, 242-253, 297-301, 342-346',
    highlights: [
      'Strict mode blocks insecure fallback at initialization.',
      'Fallback signing creates random signatures with no authenticity guarantee.',
      'Fallback verification returns true, which is unacceptable in production.',
    ],
    recommendations: [
      'Treat the fallback path as test-only code.',
      'Add startup assertions so domain modules cannot silently run without a real provider.',
    ],
  },
  {
    id: 'lms-xmss',
    label: 'LMS and XMSS state handling',
    domain: 'stateful-signatures',
    status: 'strong',
    summary:
      'The stateful signature implementation is comparatively mature, with atomic persistence, exclusive locking, and state advancement before signing.',
    file: 'ai_engine/crypto/lms_xmss.py',
    lines: '252-260, 264-274, 701-782, 1181-1254',
    highlights: [
      'State is written with an atomic replace pattern.',
      'Signing takes an exclusive lock before reading or advancing state.',
      'Leaf indices advance before signature generation to avoid reuse after crashes.',
    ],
    recommendations: [
      'Keep this implementation as a reference for safety-critical PQC state management.',
      'Add recovery and stress tests around concurrent access and crash scenarios.',
    ],
  },
  {
    id: 'falcon-rust',
    label: 'Rust Falcon dataplane engine',
    domain: 'rust-dataplane',
    status: 'strong',
    summary:
      'The Rust Falcon engine uses real pqcrypto operations for key generation, signing, and verification, with reasonable domain-level configuration hooks.',
    file: 'rust/dataplane/crates/pqc_tls/src/falcon.rs',
    lines: '277-307, 385-403, 455-473, 511-530',
    highlights: [
      'Key generation uses pqcrypto_falcon directly.',
      'Detached signing and verification use concrete Falcon primitives.',
      'Domain configs exist for automotive, aviation, and healthcare.',
    ],
    recommendations: [
      'Turn advisory config flags into enforced behavior where needed.',
      'Implement actual compression and compliance controls instead of metadata-only toggles.',
    ],
  },
  {
    id: 'v2x-group-signatures',
    label: 'Automotive V2X group signatures',
    domain: 'automotive',
    status: 'unsafe',
    summary:
      'This module presents itself as group signatures with identity escrow and revocation, but critical pieces are still hash-based placeholders and the verifier uses the group manager key incorrectly.',
    file: 'ai_engine/domains/automotive/v2x_group_signatures.py',
    lines: '404-411, 486-510, 800-817, 1021-1036, 1041-1050, 1083-1087',
    highlights: [
      'Identity escrow is a SHAKE256 hash, not actual encryption under the manager key.',
      'Per-signature escrow is also a simplified hash-based construction.',
      'Verification checks signatures against the group manager verification key, which is not a real group-signature verification flow.',
      'Revocation update tokens do not line up with the verifier-side matching logic.',
    ],
    recommendations: [
      'Do not market this as production-ready anonymity or traceability.',
      'Replace the placeholder scheme with a real group-signature construction or re-scope it as pseudonymous signatures.',
    ],
  },
  {
    id: 'tesla-broadcast-auth',
    label: 'Industrial TESLA++ broadcast authentication',
    domain: 'industrial',
    status: 'unsafe',
    summary:
      'The sender signs commitments with ML-DSA, but receiver-side verification accepts any non-empty signature because trusted public-key validation is missing.',
    file: 'ai_engine/domains/industrial/tesla_broadcast_auth.py',
    lines: '533-545, 748-781',
    highlights: [
      'Commitment signing uses a real ML-DSA path.',
      'Commitment verification explicitly avoids cryptographic verification and only checks that the signature is present.',
      'This breaks the trust anchor for the whole TESLA commitment chain.',
    ],
    recommendations: [
      'Add PKI-backed public-key lookup and real signature verification before using this in any deployment.',
      'Fail closed when trusted sender identity or key material is unavailable.',
    ],
  },
  {
    id: 'verifiable-credentials',
    label: 'Healthcare verifiable credentials',
    domain: 'healthcare',
    status: 'prototype',
    summary:
      'Selective disclosure and predicate proofs are placeholder constructions, and verifier logic only checks proof shape rather than issuer trust, inclusion, and proof correctness.',
    file: 'ai_engine/domains/healthcare/verifiable_credentials.py',
    lines: '395-405, 407-426, 469-497',
    highlights: [
      'Merkle proofs are reduced to a single hash over sibling values.',
      'Predicate proofs are hashes of booleans with random salt, not zero-knowledge proofs.',
      'Presentation verification checks length and presence rather than actual cryptographic validity.',
    ],
    recommendations: [
      'Treat this as a design prototype rather than a deployable credential system.',
      'Integrate real issuer verification, Merkle inclusion proofs, and mature ZK primitives before productizing it.',
    ],
  },
];

export const maturitySummary = [
  {
    label: 'Production-grade signals',
    value: '2',
    detail: 'Rust Falcon engine and LMS/XMSS state handling',
  },
  {
    label: 'Guarded core wrappers',
    value: '2',
    detail: 'ML-KEM and ML-DSA are viable only with strict provider enforcement',
  },
  {
    label: 'Domain modules needing redesign',
    value: '3',
    detail: 'Automotive, industrial, and healthcare examples still rely on placeholders',
  },
];
