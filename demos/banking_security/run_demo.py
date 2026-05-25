#!/usr/bin/env python3
"""
QBITEL Bridge — Banking Security Demo Runner

Usage:
    python run_demo.py            # Interactive CLI menu
    python run_demo.py --server   # Start web server on port 8002
    python run_demo.py --auto     # Run all use cases in terminal
"""

import argparse
import asyncio
import json
import os
import sys
import time

# Ensure fallback crypto is available for demo portability
os.environ["QBITEL_PQC_ALLOW_FALLBACK"] = "1"
os.environ["QBITEL_ALLOW_EXPERIMENTAL_CRYPTO"] = "1"

# Add repo root to path
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(DEMO_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)


# ─── ANSI colors ──────────────────────────────────────────────────────────────

class C:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def banner():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════════╗
║          QBITEL Bridge — Banking Security Demo               ║
║     Post-Quantum Cryptography for Financial Services         ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
""")


def section(title):
    print(f"\n{C.CYAN}{C.BOLD}{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}{C.RESET}\n")


def ok(msg):
    print(f"  {C.GREEN}✓{C.RESET} {msg}")


def info(msg):
    print(f"  {C.DIM}→{C.RESET} {msg}")


def metric(label, value):
    print(f"  {C.CYAN}{label}:{C.RESET} {value}")


# ─── Auto mode: run all 4 UCs ─────────────────────────────────────────────────

async def run_auto():
    banner()

    # ── UC1: SWIFT Chain ──
    section("UC1: SWIFT Correspondent Banking Chain")
    info("Encrypting MT103 payment ($1.5M USD) through 3-bank chain...")

    from ai_engine.domains.banking.security.swift_proxy_reencryption import (
        SwiftProxyReEncryption, BankIdentity, SwiftMessageType,
    )
    from ai_engine.crypto.mlkem import MlKemSecurityLevel
    import secrets

    swift = SwiftProxyReEncryption()
    pk_size = MlKemSecurityLevel.MLKEM_768.public_key_size  # 1184
    sk_size = MlKemSecurityLevel.MLKEM_768.private_key_size  # 2400
    banks = [
        BankIdentity("ORIGUS33", "GlobalBank New York", "US", secrets.token_bytes(pk_size), secrets.token_bytes(sk_size)),
        BankIdentity("DEUTDEFF", "Deutsche Bank Frankfurt", "DE", secrets.token_bytes(pk_size), secrets.token_bytes(sk_size)),
        BankIdentity("HSBCHKHH", "HSBC Hong Kong", "HK", secrets.token_bytes(pk_size), secrets.token_bytes(sk_size)),
    ]

    payload = json.dumps({"sender": "ORIGUS33", "receiver": "HSBCHKHH", "amount": 1500000, "currency": "USD"}).encode()

    t0 = time.perf_counter()
    msg = await swift.encrypt_message(payload, SwiftMessageType.MT103, banks[0], banks[2])
    ok(f"Encrypted at originator ({(time.perf_counter()-t0)*1000:.1f}ms)")

    rk1 = await swift.generate_chain_key(banks[0], banks[1])
    msg = await swift.re_encrypt_hop(msg, rk1)
    ok(f"Re-encrypted at intermediary (DEUTDEFF) — never decrypted")

    rk2 = await swift.generate_chain_key(banks[1], banks[2])
    msg = await swift.re_encrypt_hop(msg, rk2)
    ok(f"Re-encrypted for beneficiary (HSBCHKHH)")

    audit = swift.get_chain_audit(msg.message_id)
    metric("Audit trail entries", len(audit))
    metric("Total hops", msg.hop_count)
    metric("Total time", f"{(time.perf_counter()-t0)*1000:.1f}ms")

    # ── UC2: Multi-Authority Signing ──
    section("UC2: Multi-Authority Transaction Signing ($50M Wire)")

    from ai_engine.domains.banking.security.multi_authority_threshold import (
        MultiAuthorityThreshold, AuthorityRole, HIGH_VALUE_QUORUM,
    )

    scheme = MultiAuthorityThreshold()
    auth_configs = [
        ("auth-treasury", AuthorityRole.TREASURY, "Treasury Ops"),
        ("auth-compliance", AuthorityRole.COMPLIANCE, "Compliance"),
        ("auth-risk", AuthorityRole.RISK_MANAGEMENT, "Risk Mgmt"),
        ("auth-legal", AuthorityRole.LEGAL, "Legal"),
        ("auth-board", AuthorityRole.BOARD_MEMBER, "Board"),
    ]
    authorities = []
    for aid, role, dept in auth_configs:
        a = await scheme.generate_authority_keypair(aid, role, dept)
        authorities.append(a)
    await scheme.setup(authorities)
    ok("5 authorities initialized")

    quorum = MultiAuthorityThreshold.get_quorum_for_amount(50_000_000)
    metric("Transaction tier", quorum.tier.value)
    metric("Quorum", f"{quorum.threshold}-of-{quorum.total_authorities}")

    tx_data = json.dumps({"amount": 50000000, "beneficiary": "HSBC London", "ref": "WIRE-2026-001"}).encode()
    req = await scheme.initiate_signing(tx_data, quorum, "demo-runner")
    ok(f"Signing request created: {req.request_id.hex()[:16]}...")

    for auth in authorities[:quorum.threshold]:
        result = await scheme.contribute_signature(req.request_id, auth)
        ok(f"{auth.role.value} signed {'✓' if result else '✗'}")

    combined = await scheme.combine_signatures(req.request_id)
    verified = await scheme.verify_combined(combined, tx_data)
    ok(f"Combined signature verified: {C.GREEN}{'VALID' if verified else 'INVALID'}{C.RESET}")

    # ── UC3: ZKP Proofs ──
    section("UC3: Zero-Knowledge Regulatory Proofs")

    from ai_engine.domains.banking.security.regulatory_proof_engine import (
        RegulatoryProofEngine, CapitalAdequacyStatement, AMLScreeningResult,
    )

    proof_engine = RegulatoryProofEngine("GLOBALBANK-001")

    info("Proving balance in [$1M, $10M] without revealing $3.2M...")
    t0 = time.perf_counter()
    proof = await proof_engine.prove_balance_range(3_200_000, 1_000_000, 10_000_000)
    vr = await proof_engine.verify_proof(proof)
    ok(f"Balance range proof: {C.GREEN}{'VERIFIED' if vr.valid else 'FAILED'}{C.RESET} ({(time.perf_counter()-t0)*1000:.1f}ms)")

    info("Proving Basel III capital adequacy...")
    stmt = CapitalAdequacyStatement(
        cet1_ratio=0.125, tier1_ratio=0.145, total_capital_ratio=0.182,
        leverage_ratio=0.065, lcr=1.35, nsfr=1.12,
        reporting_date="2026-03-25", institution_id="GLOBALBANK-001",
    )
    proof2 = await proof_engine.prove_capital_adequacy(stmt)
    vr2 = await proof_engine.verify_proof(proof2)
    ok(f"Capital adequacy proof: {C.GREEN}{'VERIFIED' if vr2.valid else 'FAILED'}{C.RESET}")

    info("Proving AML screening compliance...")
    aml = AMLScreeningResult(
        total_entities_screened=15420, screening_date="2026-03-25",
        lists_checked=["OFAC-SDN", "EU-CONSOLIDATED", "UN-SC", "UK-HMT"],
        screening_engine_version="4.2.1", all_clear=True,
    )
    proof3 = await proof_engine.prove_aml_screening(aml)
    vr3 = await proof_engine.verify_proof(proof3)
    ok(f"AML screening proof: {C.GREEN}{'VERIFIED' if vr3.valid else 'FAILED'}{C.RESET}")

    # ── UC4: Threat Assessment ──
    section("UC4: Quantum Threat Assessment")

    from ai_engine.crypto.quantum_threat_scoring import (
        QuantumThreatScorer, CryptoAsset, DataSensitivity, MigrationPhase,
    )

    scorer = QuantumThreatScorer()
    assets = [
        CryptoAsset("wire-signing", "Wire Transfer Signing", "ECDSA-P256", 256,
                     DataSensitivity.SECRET, 7, 150, MigrationPhase.NOT_STARTED, "banking"),
        CryptoAsset("customer-encrypt", "Customer Data Encryption", "RSA-2048", 2048,
                     DataSensitivity.CONFIDENTIAL, 10, 500, MigrationPhase.PLANNING, "banking"),
        CryptoAsset("atm-keys", "ATM Network Keys", "AES-128", 128,
                     DataSensitivity.CONFIDENTIAL, 3, 2000, MigrationPhase.NOT_STARTED, "banking"),
        CryptoAsset("swift-pqc", "SWIFT Message Auth (PQC)", "ML-DSA-65", 192,
                     DataSensitivity.SECRET, 7, 50, MigrationPhase.HYBRID_DEPLOYMENT, "banking"),
        CryptoAsset("backup-encrypt", "Backup Encryption", "AES-256", 256,
                     DataSensitivity.SECRET, 20, 10, MigrationPhase.PQC_ONLY, "banking"),
    ]

    portfolio = scorer.assess_portfolio(assets)
    metric("Total assets scanned", portfolio.total_assets)
    metric("Critical risk", portfolio.critical_count)
    metric("High risk", portfolio.high_count)
    metric("Moderate risk", portfolio.moderate_count)
    metric("Low risk", portfolio.low_count)
    metric("Avg quantum risk score", f"{portfolio.average_qrs:.1f}/100")
    metric("HNDL at risk", portfolio.harvest_now_at_risk)
    metric("Migration coverage", f"{portfolio.migration_coverage*100:.0f}%")

    print(f"\n{C.GREEN}{C.BOLD}Demo complete. All 4 use cases executed successfully.{C.RESET}\n")


# ─── Interactive mode ──────────────────────────────────────────────────────────

async def run_interactive():
    banner()
    while True:
        print(f"""
{C.BOLD}Select a demo:{C.RESET}
  1. SWIFT Correspondent Banking Chain
  2. Multi-Authority Transaction Signing
  3. Zero-Knowledge Regulatory Proofs
  4. Quantum Threat Assessment
  5. Run All (auto mode)
  0. Exit
""")
        choice = input(f"{C.CYAN}>{C.RESET} ").strip()
        if choice == "0":
            print("Goodbye.")
            break
        elif choice == "5":
            await run_auto()
        elif choice in ("1", "2", "3", "4"):
            print(f"\n{C.YELLOW}Use --auto mode to run individual UCs, or --server for the full dashboard.{C.RESET}")
            print(f"Launching full auto demo...\n")
            await run_auto()
            break
        else:
            print(f"{C.RED}Invalid choice.{C.RESET}")


# ─── Server mode ──────────────────────────────────────────────────────────────

def run_server(host="127.0.0.1", port=8002):
    banner()
    print(f"{C.GREEN}Starting Banking Security Demo server...{C.RESET}")
    print(f"  Dashboard: http://{host}:{port}/dashboard")
    print(f"  Swagger:   http://{host}:{port}/docs")
    print(f"  Health:    http://{host}:{port}/health")
    print()

    import uvicorn
    os.chdir(os.path.join(DEMO_DIR, "backend"))
    uvicorn.run("app:app", host=host, port=port, reload=True, log_level="info")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QBITEL Banking Security Demo")
    parser.add_argument("--server", "-s", action="store_true", help="Start web server")
    parser.add_argument("--auto", "-a", action="store_true", help="Run all demos automatically")
    parser.add_argument("--port", "-p", type=int, default=8002, help="Server port (default: 8002)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    args = parser.parse_args()

    if args.server:
        run_server(args.host, args.port)
    elif args.auto:
        asyncio.run(run_auto())
    else:
        asyncio.run(run_interactive())


if __name__ == "__main__":
    main()
