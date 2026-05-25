"""
PQC (Post-Quantum Cryptography) Dataset Generator

Generates instruction-tuning pairs for LLM fine-tuning on post-quantum
cryptography algorithm selection, quantum threat assessment, migration
planning, compliance analysis, and protocol-specific PQC integration.

Produces JSONL training data suitable for supervised fine-tuning of
language models deployed in the QBITEL security platform.

Categories (500 pairs total):
1. algorithm_selection (30%): PQC algorithm recommendation for use cases
2. threat_assessment (20%): Quantum threat level evaluation for assets
3. migration_planning (20%): PQC migration plans for specific domains
4. compliance_analysis (15%): Compliance requirements mapped to PQC controls
5. protocol_security (15%): Protocol-specific PQC integration challenges

Each pair includes:
- pair_id: Unique identifier (UUID)
- category: One of 5 categories
- difficulty: basic / intermediate / advanced
- domain: Target industry domain
- instruction: The prompt/question
- context: Optional supporting data
- response: Detailed expected answer
"""

import json
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class PQCDatasetGenerator:
    """Generate LLM fine-tuning instruction pairs for PQC security domains."""

    CATEGORIES = {
        "algorithm_selection": 0.30,
        "threat_assessment": 0.20,
        "migration_planning": 0.20,
        "compliance_analysis": 0.15,
        "protocol_security": 0.15,
    }

    DIFFICULTIES = ["basic", "intermediate", "advanced"]

    PQC_ALGORITHMS = {
        "ML-KEM-512":   {"type": "KEM",       "nist_level": 1, "pk_size": 800,  "sig_or_ct": 768,   "fips": "FIPS 203", "family": "lattice"},
        "ML-KEM-768":   {"type": "KEM",       "nist_level": 3, "pk_size": 1184, "sig_or_ct": 1088,  "fips": "FIPS 203", "family": "lattice"},
        "ML-KEM-1024":  {"type": "KEM",       "nist_level": 5, "pk_size": 1568, "sig_or_ct": 1568,  "fips": "FIPS 203", "family": "lattice"},
        "ML-DSA-44":    {"type": "signature", "nist_level": 2, "pk_size": 1312, "sig_or_ct": 2420,  "fips": "FIPS 204", "family": "lattice"},
        "ML-DSA-65":    {"type": "signature", "nist_level": 3, "pk_size": 1952, "sig_or_ct": 3309,  "fips": "FIPS 204", "family": "lattice"},
        "ML-DSA-87":    {"type": "signature", "nist_level": 5, "pk_size": 2592, "sig_or_ct": 4627,  "fips": "FIPS 204", "family": "lattice"},
        "Falcon-512":   {"type": "signature", "nist_level": 1, "pk_size": 897,  "sig_or_ct": 666,   "fips": "pending",  "family": "lattice-NTRU"},
        "Falcon-1024":  {"type": "signature", "nist_level": 5, "pk_size": 1793, "sig_or_ct": 1280,  "fips": "pending",  "family": "lattice-NTRU"},
        "SLH-DSA-128s": {"type": "signature", "nist_level": 1, "pk_size": 32,   "sig_or_ct": 7856,  "fips": "FIPS 205", "family": "hash-based"},
        "SLH-DSA-256f": {"type": "signature", "nist_level": 5, "pk_size": 64,   "sig_or_ct": 49856, "fips": "FIPS 205", "family": "hash-based"},
        "XMSS":         {"type": "signature", "nist_level": 0, "pk_size": 64,   "sig_or_ct": 2500,  "fips": "SP 800-208", "family": "hash-stateful"},
        "LMS":          {"type": "signature", "nist_level": 0, "pk_size": 60,   "sig_or_ct": 4684,  "fips": "SP 800-208", "family": "hash-stateful"},
    }

    CLASSICAL_ALGORITHMS = {
        "RSA-2048":    {"type": "asymmetric",    "quantum_vulnerable": True,  "attack": "Shor", "qubits": 4096},
        "RSA-4096":    {"type": "asymmetric",    "quantum_vulnerable": True,  "attack": "Shor", "qubits": 8192},
        "ECDSA-P256":  {"type": "signature",     "quantum_vulnerable": True,  "attack": "Shor", "qubits": 2330},
        "ECDH-P384":   {"type": "key_exchange",  "quantum_vulnerable": True,  "attack": "Shor", "qubits": 3484},
        "AES-128":     {"type": "symmetric",     "quantum_vulnerable": False, "attack": "Grover", "qubits": 0},
        "AES-256":     {"type": "symmetric",     "quantum_vulnerable": False, "attack": "Grover", "qubits": 0},
        "SHA-256":     {"type": "hash",          "quantum_vulnerable": False, "attack": "Grover", "qubits": 0},
        "3DES":        {"type": "symmetric",     "quantum_vulnerable": True,  "attack": "Grover", "qubits": 0},
        "DH-2048":     {"type": "key_exchange",  "quantum_vulnerable": True,  "attack": "Shor", "qubits": 4096},
    }

    DOMAINS = ["banking", "healthcare", "automotive", "aviation", "industrial", "telecom", "defense", "insurance"]
    PROTOCOLS = ["TLS 1.3", "SSH", "IKEv2", "IEEE 1609.2", "HL7 FHIR", "ISO 20022", "MQTT", "OPC-UA"]
    COMPLIANCE_STANDARDS = ["CNSA 2.0", "NIST SP 800-208", "FIPS 203", "FIPS 204", "FIPS 205",
                            "PCI-DSS v4.0", "HIPAA", "IEC 62443", "DO-326A", "ISO 21434"]

    def __init__(self, seed: Optional[int] = None):
        """Initialize the PQC dataset generator with optional random seed."""
        if seed:
            random.seed(seed)

    def _fmt(self, template: str, **kwargs) -> str:
        """Format a template string with keyword arguments, ignoring missing keys."""
        for key, val in kwargs.items():
            template = template.replace("{" + key + "}", str(val))
        return template

    # ------------------------------------------------------------------
    # Algorithm Selection (30%)
    # ------------------------------------------------------------------

    def _generate_algorithm_selection_pairs(self, count: int) -> List[Dict]:
        """Generate PQC algorithm recommendation instruction pairs."""
        instructions = [
            "What PQC algorithm should a {domain} organization use for key exchange?",
            "Recommend a post-quantum signature algorithm for {domain} applications.",
            "Which ML-KEM parameter set is appropriate for {domain} use cases?",
            "Compare Falcon-512, ML-DSA-44, and LMS for firmware signing on {domain} embedded devices with 64KB RAM.",
            "Design a hybrid key exchange combining X25519 with ML-KEM-768 for {domain} TLS. What are the tradeoffs?",
            "A {domain} PKI issues 10M certificates/year. Which PQC signature minimizes cert size while meeting CNSA 2.0?",
            "Design a complete PQC algorithm suite for a {domain} zero-trust architecture covering KEM, auth, code signing, and data-at-rest.",
            "Propose a multi-algorithm strategy for {domain} that survives a lattice-family break. Include fallback algorithms.",
        ]
        contexts = [
            "Organization: {domain} firm. Current: RSA-2048 + ECDSA-P256. Timeline: 2026 migration.",
            "Environment: {domain} cloud, 10K API calls/min. Current: TLS 1.3 with X25519 + Ed25519.",
            "System: {domain} IoT gateway. ARM Cortex-M4, 256KB RAM. Battery-powered, 100 msgs/hour.",
            "Enterprise: Global {domain} corp. 50K employees, 200 apps. Requirements: CNSA 2.0 by 2030.",
        ]
        responses = [
            ("## PQC Recommendation: Key Exchange for {domain}\n\n"
             "### Recommended: ML-KEM-768 (FIPS 203)\n"
             "- NIST Level 3, equivalent to AES-192 quantum security\n"
             "- Public key: 1,184B, Ciphertext: 1,088B, Shared secret: 32B\n"
             "- CNSA 2.0 approved for classified use by 2030\n\n"
             "### Why Not ML-KEM-512?\n"
             "Level 1 provides lower security margin. For {domain} data with long confidentiality needs, Level 3 is minimum.\n\n"
             "### Why Not ML-KEM-1024?\n"
             "Level 5 has 1,568B public key with diminishing returns for most {domain} applications."),
            ("## PQC Signature for {domain}\n\n"
             "### Recommended: ML-DSA-65 (FIPS 204, Level 3)\n"
             "| Algorithm | PK Size | Sig Size | NIST Level |\n"
             "|-----------|---------|----------|------------|\n"
             "| ML-DSA-44 | 1,312B  | 2,420B   | 2          |\n"
             "| ML-DSA-65 | 1,952B  | 3,309B   | 3          |\n"
             "| ML-DSA-87 | 2,592B  | 4,627B   | 5          |\n"
             "| Falcon-512| 897B    | 666B     | 1          |\n\n"
             "ML-DSA-65 balances security, key size, and verification speed. "
             "Consider Falcon-512 (666B sig) if certificate size is critical."),
            ("## Complete PQC Suite for {domain} Zero-Trust\n\n"
             "1. **Key Exchange**: ML-KEM-1024 (Level 5) - TLS hybrid with X25519\n"
             "2. **Authentication**: ML-DSA-87 (Level 5) - X.509 + mTLS\n"
             "3. **Code Signing**: LMS/XMSS (SP 800-208) for firmware; ML-DSA-65 for CI/CD\n"
             "4. **Data at Rest**: AES-256-GCM (quantum-safe); ML-KEM-768 for key wrapping\n\n"
             "### CNSA 2.0 Alignment\n"
             "| Component | Algorithm | Deadline | Status |\n"
             "|-----------|-----------|----------|--------|\n"
             "| KEM | ML-KEM-1024 | 2030 | Compliant |\n"
             "| Signatures | ML-DSA-87 | 2035 | Compliant |\n"
             "| Code Signing | LMS | 2025 | Compliant |\n"
             "| Symmetric | AES-256 | N/A | Compliant |"),
        ]
        pairs = []
        for _ in range(count):
            d = random.choice(self.DOMAINS)
            pairs.append({
                "pair_id": str(uuid.uuid4()), "category": "algorithm_selection",
                "difficulty": random.choice(self.DIFFICULTIES), "domain": d,
                "instruction": self._fmt(random.choice(instructions), domain=d),
                "context": self._fmt(random.choice(contexts), domain=d),
                "response": self._fmt(random.choice(responses), domain=d),
            })
        return pairs

    # ------------------------------------------------------------------
    # Threat Assessment (20%)
    # ------------------------------------------------------------------

    def _generate_threat_assessment_pairs(self, count: int) -> List[Dict]:
        """Generate quantum threat assessment instruction pairs using Mosca's theorem."""
        instructions = [
            "Perform a Mosca's theorem analysis for {asset} protecting {domain} data with {shelf}-year confidentiality.",
            "What is the quantum threat level for {asset} in a {domain} environment?",
            "Calculate the harvest-now-decrypt-later risk for {domain} communications using {asset}.",
            "Rank these by quantum vulnerability: RSA-2048, ECDSA-P256, AES-128, SHA-256.",
            "Assess whether {asset} provides adequate protection for {domain} data until {year}.",
            "What is the CRQC timeline for breaking {asset}?",
            "Perform a full quantum risk assessment for {domain} using RSA-2048 KE, ECDSA-P256 sig, AES-128 bulk.",
        ]
        contexts = [
            "Year: 2025. {domain} data classified sensitive for {shelf} years. Migration estimate: 3-5 years.",
            "Threat model: Nation-state adversary. {domain} traffic intercepted today decrypted when CRQC arrives.",
            "Inventory: 500 servers using {asset} for TLS. Cert lifetime: 1yr. Retention: {shelf} years.",
        ]
        pairs = []
        for _ in range(count):
            domain = random.choice(self.DOMAINS)
            asset = random.choice(list(self.CLASSICAL_ALGORITHMS.keys()))
            info = self.CLASSICAL_ALGORITHMS[asset]
            shelf = random.choice([5, 10, 15, 20, 25])
            year = random.choice([2030, 2035, 2040])
            crqc = random.choice(["2029", "2032", "2035"])
            sum_xy = shelf + 4
            at_risk = (2025 + sum_xy) > int(crqc)
            risk = "CRITICAL" if at_risk else "HIGH"
            verdict = "TRUE - Migrate urgently" if at_risk else "FALSE - Proactive migration recommended"

            response = (
                f"## Mosca's Theorem: {asset} for {domain}\n\n"
                f"### x + y > z Analysis\n"
                f"- x (shelf life): {shelf} years\n"
                f"- y (migration time): ~4 years\n"
                f"- z (CRQC for {asset}): ~{crqc}\n"
                f"- x+y = {sum_xy} -> target {2025 + sum_xy}\n"
                f"- Verdict: {verdict}\n\n"
                f"### Threat Profile\n"
                f"- Algorithm family: {info['type']}\n"
                f"- Quantum attack: {info['attack']}'s algorithm\n"
                f"- Vulnerable: {'Yes' if info['quantum_vulnerable'] else 'Reduced security only'}\n"
                f"- HNDL risk: {risk}\n\n"
                f"### Recommendation\n"
                f"Migrate to {'ML-KEM-768' if info['type'] in ('asymmetric','key_exchange') else 'ML-DSA-65'}. "
                f"Deploy hybrid mode within 12 months. Full PQC by {min(2025+sum_xy, int(crqc))}."
            )
            v = dict(asset=asset, domain=domain, shelf=str(shelf), year=str(year))
            pairs.append({
                "pair_id": str(uuid.uuid4()), "category": "threat_assessment",
                "difficulty": random.choice(self.DIFFICULTIES), "domain": domain,
                "instruction": self._fmt(random.choice(instructions), **v),
                "context": self._fmt(random.choice(contexts), **v),
                "response": response,
            })
        return pairs

    # ------------------------------------------------------------------
    # Migration Planning (20%)
    # ------------------------------------------------------------------

    def _generate_migration_planning_pairs(self, count: int) -> List[Dict]:
        """Generate PQC migration planning pairs for domain-specific scenarios."""
        scenarios = {
            "banking":    {"systems": ["SWIFT gateway", "core banking TLS", "HSM key hierarchy", "ISO 20022 signing"],
                           "standards": ["PCI-DSS v4.0", "CNSA 2.0", "SWIFT CSP"],
                           "constraints": ["zero downtime", "regulatory audit trail", "correspondent bank compatibility"]},
            "healthcare": {"systems": ["HL7 FHIR API TLS", "EHR encryption", "medical device auth", "telehealth video"],
                           "standards": ["HIPAA", "HITRUST", "FDA 21 CFR Part 11"],
                           "constraints": ["patient safety", "legacy device compatibility", "interoperability"]},
            "automotive": {"systems": ["V2X broadcast auth", "OTA firmware", "ECU secure boot", "CAN bus auth"],
                           "standards": ["IEEE 1609.2", "ISO 21434", "UNECE WP.29"],
                           "constraints": ["<10ms latency", "constrained ECU resources", "15-year vehicle lifetime"]},
            "aviation":   {"systems": ["ARINC 653 comms", "ACARS signing", "EFB encryption", "ATC auth"],
                           "standards": ["DO-326A", "DO-356A", "ARINC 653"],
                           "constraints": ["DAL A-E safety cert", "25-year airframe lifecycle", "air-gapped avionics"]},
            "industrial": {"systems": ["SCADA TLS", "IEC 61850 GOOSE auth", "OPC-UA certs", "PLC firmware verify"],
                           "standards": ["IEC 62443", "NERC CIP", "NIST SP 800-82"],
                           "constraints": ["OT isolation", "20-year equipment life", "deterministic latency"]},
        }
        instructions = [
            "Outline key steps for migrating {domain} from RSA-2048 to PQC.",
            "Create a phased PQC migration plan for {system} including hybrid mode and certificate rollover.",
            "Design the PKI migration from ECDSA-P384 to ML-DSA-65 for {domain} with backward compatibility.",
            "Build a 3-year PQC roadmap for {domain} covering {system} with compliance gates for {standard}.",
            "How should {domain} handle {constraint} during PQC migration of {system}?",
        ]
        pairs = []
        for _ in range(count):
            domain = random.choice(list(scenarios.keys()))
            s = scenarios[domain]
            system = random.choice(s["systems"])
            standard = random.choice(s["standards"])
            constraint = random.choice(s["constraints"])
            difficulty = random.choice(self.DIFFICULTIES)
            v = dict(domain=domain, system=system, standard=standard, constraint=constraint)

            response = (
                f"## PQC Migration: {system} ({domain})\n\n"
                f"### Phase 1: Discovery (Months 1-3)\n"
                f"- Inventory all crypto assets across {system}\n"
                f"- Classify data by confidentiality lifetime\n"
                f"- Identify quantum-vulnerable algorithms (RSA, ECDSA, DH)\n\n"
                f"### Phase 2: Planning (Months 3-6)\n"
                f"- Select: ML-KEM-768 (KEM), ML-DSA-65 (signatures), LMS (firmware)\n"
                f"- Evaluate HSM PQC support; procure PQC-capable HSMs\n"
                f"- Design hybrid architecture addressing: {constraint}\n\n"
                f"### Phase 3: Pilot (Months 6-12)\n"
                f"- Deploy hybrid mode on non-critical {system} endpoints\n"
                f"- Validate {standard} compliance in hybrid mode\n"
                f"- Performance benchmark against SLAs\n\n"
                f"### Phase 4: Production (Months 12-24)\n"
                f"- Roll out hybrid to production {system}\n"
                f"- Certificate rollover: dual-signed certs (ECDSA + ML-DSA)\n"
                f"- Coordinate partner PQC readiness\n\n"
                f"### Phase 5: Deprecation (Months 24-36)\n"
                f"- Disable classical-only cipher suites\n"
                f"- Legacy gateway for remaining classical peers\n"
                f"- Final {standard} audit and certification"
            )
            pairs.append({
                "pair_id": str(uuid.uuid4()), "category": "migration_planning",
                "difficulty": difficulty, "domain": domain,
                "instruction": self._fmt(random.choice(instructions), **v),
                "context": f"Domain: {domain}. System: {system}. Standard: {standard}. "
                           f"Constraint: {constraint}. Current: RSA-2048 + ECDSA-P256 + AES-256-GCM.",
                "response": response,
            })
        return pairs

    # ------------------------------------------------------------------
    # Compliance Analysis (15%)
    # ------------------------------------------------------------------

    def _generate_compliance_analysis_pairs(self, count: int) -> List[Dict]:
        """Generate compliance-to-PQC mapping instruction pairs."""
        instructions = [
            "What PQC algorithms satisfy {standard} requirements for {domain}?",
            "Map {standard} cryptographic requirements to NIST PQC standards.",
            "A {domain} org must comply with {standard} and CNSA 2.0. Are there conflicts?",
            "Which FIPS publication covers {algorithm} and what compliance does it enable?",
            "Create a compliance matrix mapping {standard} controls to PQC algorithm choices for {domain}.",
            "What evidence artifacts demonstrate PQC compliance with {standard} for {domain}?",
            "Compare CNSA 2.0 vs {standard} PQC requirements. Which is more prescriptive?",
            "When does {standard} require PQC adoption? What interim measures for {domain}?",
        ]
        algorithms = ["ML-KEM-768", "ML-DSA-65", "ML-KEM-1024", "ML-DSA-87",
                       "Falcon-512", "SLH-DSA-128s", "XMSS", "LMS"]
        contexts = [
            "Audit scope: all {domain} systems handling sensitive data. Current: classical only. "
            "Next audit in 12 months. Budget approved for PQC.",
            "Regulatory: {standard} updated to reference quantum-resistant crypto. {domain} must "
            "demonstrate compliance plan. Inventory: 200 certs, 50 HSMs, 1000 TLS endpoints.",
        ]
        responses = [
            ("## Compliance: {standard} + PQC for {domain}\n\n"
             "### Algorithm Mapping\n"
             "| Requirement | PQC Algorithm | FIPS |\n"
             "|-------------|--------------|------|\n"
             "| Key establishment | ML-KEM-768/1024 | FIPS 203 |\n"
             "| Digital signatures | ML-DSA-65/87 | FIPS 204 |\n"
             "| Code signing | LMS/XMSS | SP 800-208 |\n"
             "| Hash-based fallback | SLH-DSA | FIPS 205 |\n\n"
             "### CNSA 2.0 vs {standard}\n"
             "- CNSA 2.0: ML-KEM-1024 required for classified by 2030\n"
             "- {standard} may accept Level 3 (ML-KEM-768) for commercial\n"
             "- Hybrid mode: accepted transitionally by both\n\n"
             "### Recommended: Deploy ML-KEM-768 for {standard}; upgrade path to ML-KEM-1024 for CNSA 2.0."),
            ("## FIPS Coverage: {algorithm}\n\n"
             "- ML-KEM: FIPS 203 (Module-Lattice KEM)\n"
             "- ML-DSA: FIPS 204 (Module-Lattice Digital Signature)\n"
             "- SLH-DSA: FIPS 205 (Stateless Hash-Based Signature)\n"
             "- XMSS/LMS: SP 800-208 (Stateful Hash-Based Signatures)\n\n"
             "### Compliance Enabled\n"
             "FIPS 140-3 validation pathway, CNSA 2.0, FedRAMP, {standard}\n\n"
             "### Evidence Artifacts for {domain}\n"
             "1. CAVP/CMVP algorithm certificates\n"
             "2. FIPS 140-3 module validation\n"
             "3. PQC migration plan with milestones\n"
             "4. Key management procedures for PQC material\n"
             "5. Penetration test results\n"
             "6. Crypto-agility assessment"),
        ]
        pairs = []
        for _ in range(count):
            domain = random.choice(self.DOMAINS)
            standard = random.choice(self.COMPLIANCE_STANDARDS)
            algorithm = random.choice(algorithms)
            v = dict(domain=domain, standard=standard, algorithm=algorithm)
            pairs.append({
                "pair_id": str(uuid.uuid4()), "category": "compliance_analysis",
                "difficulty": random.choice(self.DIFFICULTIES), "domain": domain,
                "instruction": self._fmt(random.choice(instructions), **v),
                "context": self._fmt(random.choice(contexts), **v),
                "response": self._fmt(random.choice(responses), **v),
            })
        return pairs

    # ------------------------------------------------------------------
    # Protocol Security (15%)
    # ------------------------------------------------------------------

    def _generate_protocol_security_pairs(self, count: int) -> List[Dict]:
        """Generate protocol-specific PQC integration instruction pairs."""
        protocol_info = {
            "TLS 1.3":      {"overhead": "~2.3KB/handshake", "challenge": "middlebox interference",
                             "domains": ["banking", "healthcare", "telecom"]},
            "SSH":           {"overhead": "~3KB key exchange", "challenge": "stateful rekeying",
                             "domains": ["banking", "industrial", "defense"]},
            "IKEv2":         {"overhead": "~3.5KB IKE_SA_INIT", "challenge": "IP fragmentation of PQC payloads",
                             "domains": ["defense", "banking", "telecom"]},
            "IEEE 1609.2":   {"overhead": "ML-DSA-65 sig 3,309B vs ECDSA 64B", "challenge": "10ms V2X latency budget",
                             "domains": ["automotive"]},
            "HL7 FHIR":      {"overhead": "~4KB/signed resource", "challenge": "HL7 v2.x backward compatibility",
                             "domains": ["healthcare"]},
            "ISO 20022":     {"overhead": "~3KB/signed payment msg", "challenge": "SWIFT global PQC coordination",
                             "domains": ["banking"]},
            "MQTT":          {"overhead": "2.3KB added to CONNECT", "challenge": "constrained IoT device RAM",
                             "domains": ["industrial", "healthcare", "automotive"]},
            "OPC-UA":        {"overhead": "~5KB/secure channel", "challenge": "real-time control latency",
                             "domains": ["industrial"]},
        }
        instructions = [
            "Analyze PQC deployment impact on {protocol} for {domain} systems.",
            "What is the handshake overhead of ML-KEM-768 with {protocol}?",
            "How does PQC cert size affect {protocol} in {domain} deployments?",
            "Design a backward-compatible PQC strategy for {protocol} in {domain}.",
            "A {domain} system uses {protocol} with 10K connections. Calculate PQC bandwidth overhead.",
            "Evaluate Falcon-512 vs ML-DSA-44 for reducing signature overhead in {protocol}.",
            "Design a PQC migration plan for {protocol} in {domain} addressing fragmentation.",
        ]
        contexts = [
            "Protocol: {protocol}. Current: TLS 1.3 + X25519 + ECDSA-P256. Target: hybrid PQC. SLA: 99.99%.",
            "Network: {domain} WAN, 50 sites. MTU: 1500B. {protocol} sessions: 10K/sec. Handshake: 2KB current.",
            "{domain} {protocol} endpoints include legacy devices. Must maintain classical peer connectivity.",
        ]
        pairs = []
        for _ in range(count):
            protocol = random.choice(list(protocol_info.keys()))
            pi = protocol_info[protocol]
            domain = random.choice(pi["domains"])
            v = dict(protocol=protocol, domain=domain)

            response = (
                f"## PQC Impact: {protocol} for {domain}\n\n"
                f"### Overhead Analysis\n"
                f"| Metric | Classical | Hybrid PQC | Change |\n"
                f"|--------|-----------|-----------|--------|\n"
                f"| Key exchange | ~64B | ~2,336B | +36x |\n"
                f"| Cert chain | ~3KB | ~12KB | +4x |\n"
                f"| Total handshake | ~5KB | ~18KB | +3.6x |\n\n"
                f"### Key Challenges\n"
                f"1. **{pi['challenge']}**\n"
                f"2. ML-DSA-65 certificates ~4x larger than ECDSA\n"
                f"3. Handshake may exceed MTU requiring fragmentation\n\n"
                f"### Mitigation\n"
                f"- TLS certificate compression (RFC 8879)\n"
                f"- Falcon-512 for leaf certs (666B sig vs 3,309B ML-DSA-65)\n"
                f"- PMTUD on all {protocol} paths\n"
                f"- Negotiate PQC + classical with classical fallback\n\n"
                f"### Overhead: {pi['overhead']}\n"
                f"At 10K sessions/sec: ~130MB/sec additional bandwidth. CPU: ~15% increase."
            )
            pairs.append({
                "pair_id": str(uuid.uuid4()), "category": "protocol_security",
                "difficulty": random.choice(self.DIFFICULTIES), "domain": domain,
                "instruction": self._fmt(random.choice(instructions), **v),
                "context": self._fmt(random.choice(contexts), **v),
                "response": response,
            })
        return pairs

    # ------------------------------------------------------------------
    # Dataset Generation
    # ------------------------------------------------------------------

    def generate_dataset(self, num_samples: int, output_dir: str) -> Dict:
        """Generate a complete dataset of PQC instruction-tuning pairs.

        Produces JSONL training data distributed across categories by weight.

        Args:
            num_samples: Total instruction pairs to generate.
            output_dir: Directory path for output files.

        Returns:
            Dataset metadata dict with counts and distributions.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generators = {
            "algorithm_selection": self._generate_algorithm_selection_pairs,
            "threat_assessment": self._generate_threat_assessment_pairs,
            "migration_planning": self._generate_migration_planning_pairs,
            "compliance_analysis": self._generate_compliance_analysis_pairs,
            "protocol_security": self._generate_protocol_security_pairs,
        }

        all_pairs: List[Dict] = []
        samples_by_category: Dict[str, int] = {}
        allocated = 0

        category_list = list(self.CATEGORIES.items())
        for idx, (category, weight) in enumerate(category_list):
            pair_count = num_samples - allocated if idx == len(category_list) - 1 else int(num_samples * weight)
            allocated += pair_count
            pairs = generators[category](pair_count)
            all_pairs.extend(pairs)
            samples_by_category[category] = pair_count

        random.shuffle(all_pairs)

        # Write JSONL (one JSON object per line)
        jsonl_path = output_path / "pqc_instruction_pairs.jsonl"
        with open(jsonl_path, "w") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair, default=str) + "\n")

        # Write full JSON for inspection
        json_path = output_path / "pqc_instruction_pairs.json"
        with open(json_path, "w") as f:
            json.dump(all_pairs, f, indent=2, default=str)

        difficulty_counts = {d: sum(1 for p in all_pairs if p["difficulty"] == d) for d in self.DIFFICULTIES}
        domain_counts: Dict[str, int] = {}
        for p in all_pairs:
            d = p.get("domain", "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1

        dataset_metadata = {
            "protocol": "pqc_instruction_tuning",
            "version": "1.0",
            "total_samples": len(all_pairs),
            "samples_by_type": samples_by_category,
            "difficulties": difficulty_counts,
            "domains": domain_counts,
            "pqc_algorithms_covered": list(self.PQC_ALGORITHMS.keys()),
            "classical_algorithms_assessed": list(self.CLASSICAL_ALGORITHMS.keys()),
            "compliance_standards": self.COMPLIANCE_STANDARDS,
            "protocols_analyzed": self.PROTOCOLS,
            "generated_at": datetime.now().isoformat(),
        }

        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate PQC instruction-tuning dataset."""
    generator = PQCDatasetGenerator(seed=42)
    output_dir = Path(__file__).parent.parent / "security_events" / "pqc_instruction_pairs"

    print("Generating PQC instruction-tuning pairs...")
    metadata = generator.generate_dataset(num_samples=500, output_dir=str(output_dir))

    print(f"Generated {metadata['total_samples']} instruction pairs")
    print(f"Output directory: {output_dir}")
    print(f"\nCategory distribution:")
    for category, count in metadata["samples_by_type"].items():
        print(f"  - {category}: {count} pairs")
    print(f"\nDifficulty distribution:")
    for difficulty, count in metadata["difficulties"].items():
        print(f"  - {difficulty}: {count} pairs")
    print(f"\nDomain distribution:")
    for domain, count in sorted(metadata["domains"].items(), key=lambda x: -x[1]):
        print(f"  - {domain}: {count} pairs")
    print(f"\nPQC algorithms covered: {len(metadata['pqc_algorithms_covered'])}")
    print(f"Classical algorithms assessed: {len(metadata['classical_algorithms_assessed'])}")


if __name__ == "__main__":
    main()
