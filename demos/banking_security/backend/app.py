"""
QBITEL Bridge Banking Security Demo
====================================

FastAPI application demonstrating post-quantum cryptographic security
capabilities for the banking sector:

  - SWIFT Proxy Re-Encryption: Multi-hop correspondent banking message
    routing with PQC-secured proxy re-encryption.
  - Multi-Authority Threshold Signatures: Quorum-based high-value
    transaction approval with role-based authority management.
  - Zero-Knowledge Regulatory Proofs: Basel III capital adequacy, AML
    screening, and balance range proofs without data exposure.
  - Quantum Threat Scoring: Portfolio-wide cryptographic risk assessment
    calibrated against NIST / CNSA 2.0 timelines.
"""

# ---------------------------------------------------------------------------
# Environment flags — MUST be set before any ai_engine imports
# ---------------------------------------------------------------------------
import os

os.environ["QBITEL_PQC_ALLOW_FALLBACK"] = "1"
os.environ["QBITEL_ALLOW_EXPERIMENTAL_CRYPTO"] = "1"

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import dataclasses
import json
import logging
import secrets
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure the repository root is on sys.path so ai_engine is importable
# ---------------------------------------------------------------------------
REPO_ROOT = str(Path(__file__).resolve().parents[3])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# ai_engine imports
# ---------------------------------------------------------------------------
from ai_engine.domains.banking.security.swift_proxy_reencryption import (
    BankIdentity,
    ChainAuditEntry,
    EncryptedSwiftMessage,
    SwiftMessageType,
    SwiftProxyReEncryption,
)
from ai_engine.domains.banking.security.multi_authority_threshold import (
    Authority,
    AuthorityRole,
    CombinedSignature,
    HIGH_VALUE_QUORUM,
    MultiAuthorityThreshold,
    QuorumPolicy,
    SigningRequest,
    TransactionTier,
)
from ai_engine.domains.banking.security.regulatory_proof_engine import (
    AMLScreeningResult,
    CapitalAdequacyStatement,
    ProofType,
    ProofVerificationResult,
    RegulatoryFramework,
    RegulatoryProof,
    RegulatoryProofEngine,
)
from ai_engine.crypto.quantum_threat_scoring import (
    CryptoAsset,
    DataSensitivity,
    MigrationPhase,
    PortfolioAssessment,
    QuantumThreatScorer,
    RiskLevel,
    ThreatAssessment,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("banking_security_demo")

# ---------------------------------------------------------------------------
# Pre-initialised banking identities
# ---------------------------------------------------------------------------
# ML-KEM-768 key sizes for SWIFT proxy re-encryption
from ai_engine.crypto.mlkem import MlKemSecurityLevel
_PK_SIZE = MlKemSecurityLevel.MLKEM_768.public_key_size   # 1184 bytes
_SK_SIZE = MlKemSecurityLevel.MLKEM_768.private_key_size   # 2400 bytes

originator = BankIdentity(
    bic="ORIGUS33",
    institution_name="Origin National Bank",
    country_code="US",
    public_key=secrets.token_bytes(_PK_SIZE),
    signing_key=secrets.token_bytes(_SK_SIZE),
)

intermediary = BankIdentity(
    bic="DEUTDEFF",
    institution_name="Deutsche Bank AG",
    country_code="DE",
    public_key=secrets.token_bytes(_PK_SIZE),
    signing_key=secrets.token_bytes(_SK_SIZE),
)

beneficiary = BankIdentity(
    bic="HSBCHKHH",
    institution_name="HSBC Hong Kong",
    country_code="HK",
    public_key=secrets.token_bytes(_PK_SIZE),
    signing_key=secrets.token_bytes(_SK_SIZE),
)

# ---------------------------------------------------------------------------
# Engine singletons
# ---------------------------------------------------------------------------
swift_engine = SwiftProxyReEncryption()
regulatory_engine = RegulatoryProofEngine("GLOBALBANK-001")
threat_scorer = QuantumThreatScorer()
threshold_scheme = MultiAuthorityThreshold()

# ---------------------------------------------------------------------------
# Mutable state for threshold signing
# ---------------------------------------------------------------------------
authorities: Dict[str, Authority] = {}
signing_requests: Dict[str, SigningRequest] = {}

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="QBITEL Bridge Banking Security Demo",
    version="1.0.0",
    description="Post-quantum cryptographic security for banking operations",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Templates & static files
# ---------------------------------------------------------------------------
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ===================================================================
# Serialisation helper
# ===================================================================

def safe_serialize(obj: Any) -> Any:
    """Recursively convert dataclasses, bytes, enums, and floats for JSON."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, bytes):
        hex_str = obj.hex()
        return hex_str[:64] if len(hex_str) > 64 else hex_str
    if isinstance(obj, Enum):
        try:
            return obj.value
        except Exception:
            return obj.name
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: safe_serialize(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [safe_serialize(i) for i in obj]
    return str(obj)


# ===================================================================
# Pydantic request models
# ===================================================================

class ThresholdSignBody(BaseModel):
    request_id: str
    authority_id: str


class ThresholdCombineBody(BaseModel):
    request_id: str


# ===================================================================
# Startup event
# ===================================================================

@app.on_event("startup")
async def startup_event():
    """Generate authority keypairs and set up threshold scheme."""
    global authorities

    logger.info("Generating authority keypairs...")

    role_configs = [
        ("auth-treasury", AuthorityRole.TREASURY, "Treasury"),
        ("auth-compliance", AuthorityRole.COMPLIANCE, "Compliance"),
        ("auth-risk", AuthorityRole.RISK_MANAGEMENT, "Risk Management"),
        ("auth-legal", AuthorityRole.LEGAL, "Legal"),
        ("auth-board", AuthorityRole.BOARD_MEMBER, "Board"),
    ]

    auth_list: List[Authority] = []
    for auth_id, role, dept in role_configs:
        auth = await threshold_scheme.generate_authority_keypair(
            authority_id=auth_id,
            role=role,
            department=dept,
        )
        authorities[auth_id] = auth
        auth_list.append(auth)
        logger.info(f"  Generated keypair for {auth_id} ({role.name})")

    group_key = await threshold_scheme.setup(auth_list)
    logger.info(
        f"Threshold scheme ready: {len(auth_list)} authorities, "
        f"group_key={group_key.hex()[:16]}..."
    )


# ===================================================================
# Routes — General
# ===================================================================

@app.get("/")
async def root():
    """Redirect to dashboard."""
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard")
async def dashboard(request: Request):
    """Render the main dashboard page."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "QBITEL Bridge Banking Security",
        "authorities": list(authorities.keys()),
    })


@app.get("/health")
async def health():
    """Health-check endpoint."""
    return {
        "status": "healthy",
        "service": "qbitel-banking-security-demo",
        "version": "1.0.0",
        "engines": {
            "swift_proxy_re_encryption": "ready",
            "multi_authority_threshold": "ready",
            "regulatory_proof_engine": "ready",
            "quantum_threat_scorer": "ready",
        },
        "authorities_loaded": len(authorities),
        "timestamp": time.time(),
    }


# ===================================================================
# Routes — SWIFT Proxy Re-Encryption
# ===================================================================

@app.post("/api/swift/full-chain")
async def swift_full_chain():
    """
    Run a full 3-bank SWIFT correspondent banking chain.

    Originator (US) -> Intermediary (DE) -> Beneficiary (HK)
    Demonstrates proxy re-encryption where intermediary transforms
    the ciphertext without decrypting the payment payload.
    """
    t_start = time.perf_counter()
    try:
        # Build MT103 payment payload
        payload = json.dumps({
            "message_type": "MT103",
            "sender_bic": originator.bic,
            "receiver_bic": beneficiary.bic,
            "amount": {"value": "1500000.00", "currency": "USD"},
            "value_date": "2026-03-25",
            "ordering_customer": "GLOBALCORP INC",
            "beneficiary_customer": "ASIA PACIFIC TRADING LTD",
            "remittance_info": "INV-2026-00847 Payment for Q1 services",
            "charges": "SHA",
        }).encode("utf-8")

        # Step 1 — Originator encrypts for beneficiary
        encrypted_msg = await swift_engine.encrypt_message(
            payload=payload,
            msg_type=SwiftMessageType.MT103,
            originator=originator,
            target=beneficiary,
        )

        hops = [{
            "hop": 0,
            "from": originator.bic,
            "to": beneficiary.bic,
            "action": "encrypt",
            "target_bic": encrypted_msg.target_bic,
            "payload_size": len(encrypted_msg.encrypted_payload),
            "hop_count": encrypted_msg.hop_count,
        }]

        # Step 2 — Generate chain re-encryption keys
        rk_orig_to_inter = await swift_engine.generate_chain_key(
            from_bank=originator,
            to_bank=intermediary,
        )
        rk_inter_to_benef = await swift_engine.generate_chain_key(
            from_bank=intermediary,
            to_bank=beneficiary,
        )

        # Step 3 — First re-encryption hop (Originator -> Intermediary)
        msg_hop1 = await swift_engine.re_encrypt_hop(
            message=encrypted_msg,
            re_key=rk_orig_to_inter,
        )

        hops.append({
            "hop": 1,
            "from": originator.bic,
            "to": intermediary.bic,
            "action": "re_encrypt",
            "target_bic": msg_hop1.target_bic,
            "payload_size": len(msg_hop1.encrypted_payload),
            "hop_count": msg_hop1.hop_count,
        })

        # Step 4 — Second re-encryption hop (Intermediary -> Beneficiary)
        msg_hop2 = await swift_engine.re_encrypt_hop(
            message=msg_hop1,
            re_key=rk_inter_to_benef,
        )

        hops.append({
            "hop": 2,
            "from": intermediary.bic,
            "to": beneficiary.bic,
            "action": "re_encrypt",
            "target_bic": msg_hop2.target_bic,
            "payload_size": len(msg_hop2.encrypted_payload),
            "hop_count": msg_hop2.hop_count,
        })

        # Step 5 — Retrieve audit trail
        audit_entries = swift_engine.get_chain_audit(encrypted_msg.message_id)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return {
            "status": "success",
            "message_id": encrypted_msg.message_id.hex(),
            "message_type": SwiftMessageType.MT103.value,
            "chain": {
                "originator": originator.bic,
                "intermediary": intermediary.bic,
                "beneficiary": beneficiary.bic,
            },
            "hops": hops,
            "audit": [safe_serialize(e) for e in audit_entries],
            "re_encryption_keys": {
                "orig_to_inter": {
                    "key_id": rk_orig_to_inter.key_id.hex()[:32],
                    "from_bic": rk_orig_to_inter.from_bic,
                    "to_bic": rk_orig_to_inter.to_bic,
                },
                "inter_to_benef": {
                    "key_id": rk_inter_to_benef.key_id.hex()[:32],
                    "from_bic": rk_inter_to_benef.from_bic,
                    "to_bic": rk_inter_to_benef.to_bic,
                },
            },
            "metrics": {
                "total_latency_ms": round(elapsed_ms, 2),
                "payload_bytes": len(payload),
                "total_hops": msg_hop2.hop_count,
            },
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.exception("SWIFT full-chain failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "latency_ms": round(elapsed_ms, 2)},
        )


# ===================================================================
# Routes — Multi-Authority Threshold Signatures
# ===================================================================

@app.post("/api/threshold/initiate")
async def threshold_initiate():
    """
    Initiate a $50M wire-transfer signing request.

    Creates a HIGH_VALUE threshold signing request requiring
    3-of-5 authority signatures including TREASURY and COMPLIANCE.
    """
    t_start = time.perf_counter()
    try:
        transaction_data = json.dumps({
            "type": "wire_transfer",
            "amount": 50_000_000,
            "currency": "USD",
            "from_account": "GLOBALBANK-001-NOSTRO-USD",
            "to_account": "COUNTERPARTY-VOSTRO-EUR",
            "value_date": "2026-03-25",
            "reference": "WT-2026-HIGH-VALUE-001",
        }).encode("utf-8")

        quorum = MultiAuthorityThreshold.get_quorum_for_amount(50_000_000.0)

        request = await threshold_scheme.initiate_signing(
            transaction_data=transaction_data,
            quorum=quorum,
            initiator_id="auth-treasury",
        )

        request_hex = request.request_id.hex()
        signing_requests[request_hex] = request

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return {
            "status": "success",
            "request_id": request_hex,
            "tier": quorum.tier.value,
            "threshold": quorum.threshold,
            "total_authorities": quorum.total_authorities,
            "required_roles": [r.name for r in quorum.required_roles],
            "optional_roles": [r.name for r in quorum.optional_roles],
            "signing_window_seconds": quorum.signing_window_seconds,
            "available_authorities": list(authorities.keys()),
            "latency_ms": round(elapsed_ms, 2),
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.exception("Threshold initiate failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "latency_ms": round(elapsed_ms, 2)},
        )


@app.post("/api/threshold/sign")
async def threshold_sign(body: ThresholdSignBody):
    """
    Authority contributes a partial signature to a signing request.

    Body: {"request_id": "<hex>", "authority_id": "auth-treasury"}
    """
    t_start = time.perf_counter()
    try:
        request_hex = body.request_id
        authority_id = body.authority_id

        if request_hex not in signing_requests:
            return JSONResponse(
                status_code=404,
                content={"error": f"Signing request {request_hex} not found"},
            )

        if authority_id not in authorities:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"Authority {authority_id} not found",
                    "available": list(authorities.keys()),
                },
            )

        sr = signing_requests[request_hex]
        auth = authorities[authority_id]

        accepted = await threshold_scheme.contribute_signature(
            request_id=sr.request_id,
            authority=auth,
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return {
            "status": "accepted" if accepted else "rejected",
            "request_id": request_hex,
            "authority_id": authority_id,
            "authority_role": auth.role.name,
            "shares_collected": sr.shares_collected,
            "threshold": sr.quorum_policy.threshold,
            "threshold_met": sr.threshold_met,
            "required_roles_satisfied": sr.required_roles_satisfied,
            "latency_ms": round(elapsed_ms, 2),
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.exception("Threshold sign failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "latency_ms": round(elapsed_ms, 2)},
        )


@app.post("/api/threshold/combine")
async def threshold_combine(body: ThresholdCombineBody):
    """
    Combine partial signatures and auto-verify the combined result.

    Body: {"request_id": "<hex>"}
    """
    t_start = time.perf_counter()
    try:
        request_hex = body.request_id

        if request_hex not in signing_requests:
            return JSONResponse(
                status_code=404,
                content={"error": f"Signing request {request_hex} not found"},
            )

        sr = signing_requests[request_hex]

        combined = await threshold_scheme.combine_signatures(sr.request_id)

        # Auto-verify
        verified = await threshold_scheme.verify_combined(
            combined=combined,
            transaction_data=sr.transaction_data,
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return {
            "status": "success",
            "request_id": request_hex,
            "combined_signature": safe_serialize(combined.combined_signature),
            "contributing_authorities": combined.contributing_authorities,
            "contributing_roles": [r.name for r in combined.contributing_roles],
            "tier": combined.tier.value,
            "verified": verified,
            "latency_ms": round(elapsed_ms, 2),
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.exception("Threshold combine failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "latency_ms": round(elapsed_ms, 2)},
        )


@app.get("/api/threshold/status/{request_id}")
async def threshold_status(request_id: str):
    """Get the current status of a threshold signing request."""
    t_start = time.perf_counter()
    try:
        if request_id not in signing_requests:
            return JSONResponse(
                status_code=404,
                content={"error": f"Signing request {request_id} not found"},
            )

        sr = signing_requests[request_id]

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return {
            "request_id": request_id,
            "status": sr.status,
            "tier": sr.quorum_policy.tier.value,
            "shares_collected": sr.shares_collected,
            "threshold": sr.quorum_policy.threshold,
            "threshold_met": sr.threshold_met,
            "required_roles_satisfied": sr.required_roles_satisfied,
            "signed_by": list(sr.collected_shares.keys()),
            "signed_roles": [r.name for r in sr.authority_roles.values()],
            "is_expired": sr.is_expired,
            "created_at": sr.created_at,
            "deadline": sr.deadline,
            "latency_ms": round(elapsed_ms, 2),
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.exception("Threshold status failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "latency_ms": round(elapsed_ms, 2)},
        )


# ===================================================================
# Routes — Zero-Knowledge Regulatory Proofs
# ===================================================================

@app.post("/api/zkp/balance-range")
async def zkp_balance_range():
    """
    Generate and verify a balance-range zero-knowledge proof.

    Proves account balance is within [$1M, $10M] without revealing
    the actual balance ($3,200,000).
    """
    t_start = time.perf_counter()
    try:
        actual_balance = 3_200_000
        range_min = 1_000_000
        range_max = 10_000_000

        proof = await regulatory_engine.prove_balance_range(
            actual_balance=actual_balance,
            range_min=range_min,
            range_max=range_max,
        )

        verification = await regulatory_engine.verify_proof(proof)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return {
            "status": "success",
            "proof_type": "BALANCE_RANGE",
            "framework": proof.framework.value,
            "proof": {
                "proof_id": safe_serialize(proof.proof_id),
                "commitment": safe_serialize(proof.commitment),
                "challenge": safe_serialize(proof.challenge),
                "response_length": len(proof.response),
                "public_inputs": safe_serialize(proof.public_inputs),
                "issuer_id": proof.issuer_id,
                "valid_until": proof.valid_until,
            },
            "verification": {
                "valid": verification.valid,
                "proof_type": verification.proof_type.name,
                "framework": verification.framework.value,
                "reason": verification.reason,
            },
            "demo_note": "Actual balance NOT revealed. Proof attests balance in [1M, 10M].",
            "latency_ms": round(elapsed_ms, 2),
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.exception("ZKP balance-range failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "latency_ms": round(elapsed_ms, 2)},
        )


@app.post("/api/zkp/capital-adequacy")
async def zkp_capital_adequacy():
    """
    Generate and verify a Basel III capital adequacy ZK proof.

    Proves the bank meets all minimum regulatory ratios without
    revealing exact capital figures.
    """
    t_start = time.perf_counter()
    try:
        statement = CapitalAdequacyStatement(
            cet1_ratio=0.125,
            tier1_ratio=0.145,
            total_capital_ratio=0.182,
            leverage_ratio=0.065,
            lcr=1.35,
            nsfr=1.12,
            reporting_date="2026-03-25",
            institution_id="GLOBALBANK-001",
        )

        proof = await regulatory_engine.prove_capital_adequacy(
            statement=statement,
            min_cet1=0.045,
            min_tier1=0.06,
            min_total=0.08,
            min_leverage=0.03,
            min_lcr=1.0,
            min_nsfr=1.0,
        )

        verification = await regulatory_engine.verify_proof(proof)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return {
            "status": "success",
            "proof_type": "CAPITAL_ADEQUACY",
            "framework": proof.framework.value,
            "proof": {
                "proof_id": safe_serialize(proof.proof_id),
                "commitment": safe_serialize(proof.commitment),
                "challenge": safe_serialize(proof.challenge),
                "response_length": len(proof.response),
                "public_inputs": safe_serialize(proof.public_inputs),
                "issuer_id": proof.issuer_id,
                "valid_until": proof.valid_until,
            },
            "verification": {
                "valid": verification.valid,
                "proof_type": verification.proof_type.name,
                "framework": verification.framework.value,
                "reason": verification.reason,
            },
            "basel_iii_minimums": {
                "cet1": "4.5%",
                "tier1": "6.0%",
                "total_capital": "8.0%",
                "leverage": "3.0%",
                "lcr": "100%",
                "nsfr": "100%",
            },
            "demo_note": "Exact ratios NOT revealed. Proof attests all ratios >= Basel III minimums.",
            "latency_ms": round(elapsed_ms, 2),
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.exception("ZKP capital-adequacy failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "latency_ms": round(elapsed_ms, 2)},
        )


@app.post("/api/zkp/aml-screening")
async def zkp_aml_screening():
    """
    Generate and verify an AML screening compliance ZK proof.

    Proves that screening was completed against all required
    sanctions lists without revealing match details.
    """
    t_start = time.perf_counter()
    try:
        result = AMLScreeningResult(
            total_entities_screened=15420,
            screening_date="2026-03-25",
            lists_checked=[
                "OFAC-SDN",
                "EU-CONSOLIDATED",
                "UN-SC-SANCTIONS",
                "UK-HMT",
            ],
            screening_engine_version="3.2.1-pqc",
            all_clear=True,
        )

        proof = await regulatory_engine.prove_aml_screening(result=result)

        verification = await regulatory_engine.verify_proof(proof)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return {
            "status": "success",
            "proof_type": "AML_SCREENING",
            "framework": proof.framework.value,
            "proof": {
                "proof_id": safe_serialize(proof.proof_id),
                "commitment": safe_serialize(proof.commitment),
                "challenge": safe_serialize(proof.challenge),
                "response_length": len(proof.response),
                "public_inputs": safe_serialize(proof.public_inputs),
                "issuer_id": proof.issuer_id,
                "valid_until": proof.valid_until,
            },
            "verification": {
                "valid": verification.valid,
                "proof_type": verification.proof_type.name,
                "framework": verification.framework.value,
                "reason": verification.reason,
            },
            "screening_summary": {
                "entities_screened": result.total_entities_screened,
                "lists_checked": result.lists_checked,
                "engine_version": result.screening_engine_version,
            },
            "demo_note": "Match details NOT revealed. Proof attests screening completed against all lists.",
            "latency_ms": round(elapsed_ms, 2),
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.exception("ZKP AML-screening failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "latency_ms": round(elapsed_ms, 2)},
        )


# ===================================================================
# Routes — Quantum Threat Scoring
# ===================================================================

@app.post("/api/threat/scan")
async def threat_scan():
    """
    Scan a representative banking cryptographic portfolio.

    Assesses 7 crypto assets spanning legacy, current, and
    post-quantum algorithms used in a typical banking environment.
    """
    t_start = time.perf_counter()
    try:
        portfolio_assets = [
            CryptoAsset(
                asset_id="wire-transfer-signing",
                name="Wire Transfer Signing",
                algorithm="RSA-2048",
                key_size_bits=2048,
                data_sensitivity=DataSensitivity.SECRET,
                data_retention_years=7,
                system_count=250,
                migration_phase=MigrationPhase.PLANNING,
                domain="banking",
            ),
            CryptoAsset(
                asset_id="card-authentication",
                name="Card Authentication",
                algorithm="ECDSA-P256",
                key_size_bits=256,
                data_sensitivity=DataSensitivity.SECRET,
                data_retention_years=5,
                system_count=1200,
                migration_phase=MigrationPhase.NOT_STARTED,
                domain="banking",
            ),
            CryptoAsset(
                asset_id="atm-session-keys",
                name="ATM Session Keys",
                algorithm="AES-128",
                key_size_bits=128,
                data_sensitivity=DataSensitivity.CONFIDENTIAL,
                data_retention_years=1,
                system_count=5000,
                migration_phase=MigrationPhase.NOT_STARTED,
                domain="banking",
            ),
            CryptoAsset(
                asset_id="tls-key-exchange",
                name="TLS Key Exchange",
                algorithm="ECDH-P256",
                key_size_bits=256,
                data_sensitivity=DataSensitivity.CONFIDENTIAL,
                data_retention_years=3,
                system_count=800,
                migration_phase=MigrationPhase.TESTING,
                domain="banking",
                is_key_exchange=True,
            ),
            CryptoAsset(
                asset_id="swift-pqc-signing",
                name="SWIFT PQC Signing",
                algorithm="ML-DSA-65",
                key_size_bits=4032,
                data_sensitivity=DataSensitivity.SECRET,
                data_retention_years=10,
                system_count=50,
                migration_phase=MigrationPhase.PQC_PRIMARY,
                domain="banking",
            ),
            CryptoAsset(
                asset_id="backup-encryption",
                name="Backup Encryption",
                algorithm="AES-256",
                key_size_bits=256,
                data_sensitivity=DataSensitivity.SECRET,
                data_retention_years=25,
                system_count=30,
                migration_phase=MigrationPhase.CNSA2_COMPLIANT,
                domain="banking",
            ),
            CryptoAsset(
                asset_id="legacy-pos-encryption",
                name="Legacy POS Encryption",
                algorithm="AES-128",
                key_size_bits=112,
                data_sensitivity=DataSensitivity.CONFIDENTIAL,
                data_retention_years=2,
                system_count=3500,
                migration_phase=MigrationPhase.NOT_STARTED,
                domain="banking",
            ),
        ]

        portfolio = threat_scorer.assess_portfolio(portfolio_assets)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return {
            "status": "success",
            "portfolio_summary": {
                "total_assets": portfolio.total_assets,
                "critical_count": portfolio.critical_count,
                "high_count": portfolio.high_count,
                "moderate_count": portfolio.moderate_count,
                "low_count": portfolio.low_count,
                "average_qrs": portfolio.average_qrs,
                "harvest_now_at_risk": portfolio.harvest_now_at_risk,
                "migration_coverage_pct": portfolio.migration_coverage,
                "cnsa2_readiness_pct": portfolio.cnsa2_readiness,
            },
            "assessments": [
                {
                    "asset_id": a.asset_id,
                    "algorithm": a.algorithm,
                    "quantum_risk_score": a.quantum_risk_score,
                    "risk_level": a.risk_level.value,
                    "factors": {
                        "algorithm_vulnerability": a.algorithm_vulnerability,
                        "time_horizon_risk": a.time_horizon_risk,
                        "data_sensitivity": a.data_sensitivity_score,
                        "migration_gap": a.migration_gap_score,
                        "exposure_surface": a.exposure_surface_score,
                    },
                    "harvest_now_risk": a.harvest_now_risk,
                    "recommended_action": a.recommended_action,
                    "recommended_algorithm": a.recommended_algorithm,
                    "migration_deadline_year": a.migration_deadline_year,
                }
                for a in portfolio.assessments
            ],
            "most_vulnerable": [
                {
                    "asset_id": a.asset_id,
                    "algorithm": a.algorithm,
                    "quantum_risk_score": a.quantum_risk_score,
                    "risk_level": a.risk_level.value,
                }
                for a in portfolio.most_vulnerable[:5]
            ],
            "latency_ms": round(elapsed_ms, 2),
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.exception("Threat scan failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "latency_ms": round(elapsed_ms, 2)},
        )


# ===================================================================
# Main entry point
# ===================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
