# QBITEL Bridge — Banking Security Demo

Interactive demo showcasing post-quantum cryptography for financial services across 4 use cases.

## Quick Start

```bash
cd demos/banking_security
pip install -r requirements.txt
python run_demo.py --server
# Open http://localhost:8002/dashboard
```

## Use Cases

| Tab | Use Case | Crypto Module |
|-----|----------|---------------|
| UC1 | SWIFT Correspondent Chain (3-bank proxy re-encryption) | ML-KEM-768 + SHAKE256 |
| UC2 | Multi-Authority Signing ($50M wire, 3-of-5 quorum) | ML-DSA-65 threshold |
| UC3 | Zero-Knowledge Regulatory Proofs (Basel III, AML) | SHA3 commitments |
| UC4 | Quantum Threat Assessment (portfolio risk scoring) | Threat scoring engine |

## Run Modes

```bash
python run_demo.py              # Interactive CLI menu
python run_demo.py --server     # Web dashboard on port 8002
python run_demo.py --auto       # Terminal walkthrough of all 4 UCs
```

## Docker

```bash
# From repo root:
docker-compose -f demos/banking_security/docker-compose.yml up --build
# Open http://localhost:8002/dashboard
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/dashboard` | Interactive web UI |
| GET | `/health` | Health check |
| POST | `/api/swift/full-chain` | Run 3-hop SWIFT chain |
| POST | `/api/threshold/initiate` | Start multi-sig request |
| POST | `/api/threshold/sign` | Authority signs |
| POST | `/api/threshold/combine` | Combine + verify |
| POST | `/api/zkp/balance-range` | Balance range ZK proof |
| POST | `/api/zkp/capital-adequacy` | Basel III ZK proof |
| POST | `/api/zkp/aml-screening` | AML compliance ZK proof |
| POST | `/api/threat/scan` | Portfolio quantum risk scan |
| GET | `/docs` | Swagger UI |

## Configuration

Edit `config/development.yaml` for local settings. The demo runs in PQC fallback mode by default (simulated crypto) for portability. Install `liboqs-python` for real post-quantum operations.

## Architecture

```
backend/app.py     → FastAPI with 11 endpoints calling ai_engine modules
templates/         → Jinja2 single-page dashboard
static/css/        → Dark theme styling
static/js/         → Vanilla JS interactive logic
config/            → YAML environment configs
```
