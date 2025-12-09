# CRONOS AI - Local Development & Testing Guide

**Version**: 2.1.0
**Last Updated**: 2024-11-22
**Purpose**: Complete guide for running CRONOS AI locally for development and testing

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start (5 minutes)](#2-quick-start-5-minutes)
3. [Full Local Setup](#3-full-local-setup)
4. [Docker Compose Deployment](#4-docker-compose-deployment)
5. [Running Individual Components](#5-running-individual-components)
6. [Database Setup](#6-database-setup)
7. [Configuration](#7-configuration)
8. [Testing](#8-testing)
9. [API Testing](#9-api-testing)
10. [Troubleshooting](#10-troubleshooting)
11. [macOS Deployment (Recommended)](#11-macos-deployment-recommended)

---

## 1. Prerequisites

### 1.1 Required Software

| Software | Version | Installation |
|----------|---------|--------------|
| Python | 3.9+ | `brew install python@3.11` (macOS) |
| PostgreSQL | 15+ | `brew install postgresql@15` |
| Redis | 7+ | `brew install redis` |
| Docker | 24+ | [Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| Node.js | 18+ | `brew install node` (for UI) |
| Git | 2.30+ | `brew install git` |

### 1.2 Optional Software

| Software | Purpose | Installation |
|----------|---------|--------------|
| Ollama | Local LLM | `brew install ollama` |
| ClamAV | Virus scanning | `brew install clamav` |
| Go | Go services | `brew install go` |
| Rust | Dataplane | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |

### 1.3 System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum (32GB for ML features)
- **Disk**: 20GB free space
- **OS**: macOS 12+, Ubuntu 20.04+, Windows 10+ (WSL2)

---

## 2. Quick Start (5 minutes)

### Option A: Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/qbitel/cronos-ai.git
cd cronos-ai

# Start all services with Docker
docker-compose -f ops/deploy/docker/docker-compose.yml up -d

# Wait for services to be healthy (30-60 seconds)
docker-compose -f ops/deploy/docker/docker-compose.yml ps

# Test health endpoint
curl http://localhost:8000/health
```

### Option B: Minimal Python Setup

```bash
# Clone and setup
git clone https://github.com/qbitel/cronos-ai.git
cd cronos-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set minimal environment variables
export DATABASE_URL="sqlite:///./cronos_ai_dev.db"
export REDIS_URL="redis://localhost:6379"
export JWT_SECRET="dev-secret-key-change-in-production"
export ENCRYPTION_KEY="dev-encryption-key-32chars!"

# Run the application
python -m ai_engine --mode development

# Test (in another terminal)
curl http://localhost:8000/health
```

---

## 3. Full Local Setup

### 3.1 Clone Repository

```bash
git clone https://github.com/qbitel/cronos-ai.git
cd cronos-ai
```

### 3.2 Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Verify Python version
python --version  # Should be 3.9+
```

### 3.3 Install Dependencies

```bash
# Install main dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install optional ML dependencies (if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Verify installation
python -c "import ai_engine; print('AI Engine imported successfully')"
```

### 3.4 Install Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

---

## 4. Docker Compose Deployment

### 4.1 Full Stack Deployment

```bash
# Navigate to docker directory
cd ops/deploy/docker

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 4.2 Services Started

| Service | Port | Description |
|---------|------|-------------|
| ai-engine | 8000 | Main API server |
| postgres | 5432 | PostgreSQL database |
| redis | 6379 | Redis cache |
| ollama | 11434 | Local LLM (optional) |

### 4.3 Docker Compose with Copilot

```bash
# Use copilot-specific compose file
docker-compose -f ops/deploy/docker/docker-compose-copilot.yml up -d
```

### 4.4 Stop All Services

```bash
docker-compose down

# Remove volumes (fresh start)
docker-compose down -v
```

---

## 5. Running Individual Components

### 5.1 Start PostgreSQL

```bash
# Using Docker
docker run -d \
  --name cronos-postgres \
  -e POSTGRES_USER=cronos \
  -e POSTGRES_PASSWORD=cronos_dev_password \
  -e POSTGRES_DB=cronos_ai_dev \
  -p 5432:5432 \
  postgres:15

# Or using Homebrew (macOS)
brew services start postgresql@15
createdb cronos_ai_dev
```

### 5.2 Start Redis

```bash
# Using Docker
docker run -d \
  --name cronos-redis \
  -p 6379:6379 \
  redis:7

# Or using Homebrew (macOS)
brew services start redis
```

### 5.3 Start Ollama (Local LLM)

```bash
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve &

# Pull recommended model
ollama pull llama3.2:8b

# Verify
curl http://localhost:11434/api/tags
```

### 5.4 Start AI Engine

```bash
# Activate virtual environment
source venv/bin/activate

# Set environment variables
export CRONOS_ENV=development
export DATABASE_HOST=localhost
export DATABASE_PORT=5432
export DATABASE_NAME=cronos_ai_dev
export DATABASE_USER=cronos
export DATABASE_PASSWORD=cronos_dev_password
export REDIS_HOST=localhost
export REDIS_PORT=6379
export JWT_SECRET=development-jwt-secret-key-32ch
export ENCRYPTION_KEY=development-encryption-key-32c

# Run in development mode (with hot reload)
python -m ai_engine --mode development

# Or run in production mode
python -m ai_engine --mode production
```

### 5.5 Start Frontend Console (Optional)

```bash
cd ui/console

# Install dependencies
npm install

# Start development server
npm run dev

# Access at http://localhost:3000
```

---

## 6. Database Setup

### 6.1 Apply Migrations

```bash
# Navigate to ai_engine directory
cd ai_engine

# Apply all migrations
alembic upgrade head

# Check current migration
alembic current

# View migration history
alembic history
```

### 6.2 Create Initial Admin User

```bash
# Using the CLI tool
python -m ai_engine.scripts.create_admin \
  --username admin \
  --email admin@cronos-ai.local \
  --password "SecurePassword123!"

# Or via Python shell
python << 'EOF'
from ai_engine.core.database_manager import get_database_manager
from ai_engine.models.database import User
from passlib.hash import bcrypt

db = get_database_manager()
session = db.get_session()

admin = User(
    username="admin",
    email="admin@cronos-ai.local",
    password_hash=bcrypt.hash("admin123"),
    role="administrator",
    is_active=True
)
session.add(admin)
session.commit()
print("Admin user created!")
session.close()
EOF
```

### 6.3 Seed Sample Data (Optional)

```bash
# Load sample protocols
python -m ai_engine.scripts.seed_data --type protocols

# Load sample threat intelligence
python -m ai_engine.scripts.seed_data --type threats
```

---

## 7. Configuration

### 7.1 Configuration File

Create a local configuration file:

```bash
cp config/cronos_ai.yaml config/cronos_ai.local.yaml
```

Edit `config/cronos_ai.local.yaml`:

```yaml
# Local Development Configuration
environment: development
debug: true

# Server
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  reload: true

# Database
database:
  host: "localhost"
  port: 5432
  name: "cronos_ai_dev"
  user: "cronos"
  password: "cronos_dev_password"
  pool_size: 10
  ssl_mode: "disable"

# Redis
redis:
  host: "localhost"
  port: 6379
  db: 0
  password: null

# LLM Configuration
llm:
  provider: "ollama"  # ollama, anthropic, openai
  endpoint: "http://localhost:11434"
  model: "llama3.2:8b"
  fallback_providers:
    - "anthropic"
    - "openai"

# Security (development values - DO NOT USE IN PRODUCTION)
security:
  jwt_secret: "development-jwt-secret-key-32ch"
  jwt_expiry_minutes: 480  # 8 hours for dev
  encryption_key: "development-encryption-key-32c"

# Logging
logging:
  level: "DEBUG"
  format: "json"

# Features
features:
  enable_copilot: true
  enable_marketplace: true
  enable_translation_studio: true
  enable_threat_intelligence: true

# Notifications (optional for local dev)
notifications:
  enabled: false

# SIEM (optional for local dev)
audit:
  log_to_file: true
  log_to_siem: false
```

### 7.2 Environment Variables

Create a `.env` file in the project root:

```bash
cat > .env << 'EOF'
# CRONOS AI Local Development Environment

# Environment
CRONOS_ENV=development
DEBUG=true

# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=cronos_ai_dev
DATABASE_USER=cronos
DATABASE_PASSWORD=cronos_dev_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Security
JWT_SECRET=development-jwt-secret-key-32ch
ENCRYPTION_KEY=development-encryption-key-32c
API_KEY=dev-api-key-for-testing

# LLM
CRONOS_LLM_PROVIDER=ollama
CRONOS_LLM_ENDPOINT=http://localhost:11434
CRONOS_LLM_MODEL=llama3.2:8b

# Optional: Cloud LLM fallback (add your keys)
# ANTHROPIC_API_KEY=sk-ant-xxx
# OPENAI_API_KEY=sk-xxx

# Logging
LOG_LEVEL=DEBUG

# Disable TLS for local development
TLS_ENABLED=false
EOF

# Load environment variables
source .env
```

### 7.3 Use Configuration

```bash
# Run with custom config
python -m ai_engine --config config/cronos_ai.local.yaml

# Or set via environment variable
export CRONOS_CONFIG_FILE=config/cronos_ai.local.yaml
python -m ai_engine
```

---

## 8. Testing

### 8.1 Run All Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest ai_engine/tests/ -v

# Run with coverage
pytest ai_engine/tests/ --cov=ai_engine --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
```

### 8.2 Run Specific Test Suites

```bash
# Unit tests
pytest ai_engine/tests/unit/ -v

# Integration tests (requires database)
pytest ai_engine/tests/integration/ -v

# Security tests
pytest ai_engine/tests/security/ -v

# Cloud-native tests
pytest ai_engine/tests/cloud_native/ -v

# Performance tests
pytest ai_engine/tests/performance/ -v
```

### 8.3 Run Tests by Keyword

```bash
# Run tests matching a keyword
pytest -k "auth" -v
pytest -k "copilot" -v
pytest -k "anomaly" -v
```

### 8.4 Run Tests in Parallel

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest ai_engine/tests/ -n auto -v
```

---

## 9. API Testing

### 9.1 Health Check

```bash
# Basic health
curl http://localhost:8000/health

# Kubernetes liveness
curl http://localhost:8000/healthz

# Kubernetes readiness
curl http://localhost:8000/readyz
```

### 9.2 Authentication

```bash
# Login
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' | jq -r '.access_token')

echo "Token: $TOKEN"

# Verify token
curl http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

### 9.3 Copilot API

```bash
# Get copilot info
curl http://localhost:8000/api/v1/copilot/ \
  -H "Authorization: Bearer $TOKEN"

# Send a query
curl -X POST http://localhost:8000/api/v1/copilot/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What protocols do you support?",
    "query_type": "general_question"
  }'
```

### 9.4 Protocol Discovery

```bash
# Discover protocol from sample data
curl -X POST http://localhost:8000/api/v1/discover \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": ["R0VUIC8gSFRUUC8xLjENCkhvc3Q6IGV4YW1wbGUuY29tDQoNCg=="],
    "encoding": "base64"
  }'
```

### 9.5 Field Detection

```bash
# Detect fields in a message
curl -X POST http://localhost:8000/api/v1/fields/detect \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "R0VUIC8gSFRUUC8xLjENCkhvc3Q6IGV4YW1wbGUuY29tDQoNCg==",
    "encoding": "base64"
  }'
```

### 9.6 Anomaly Detection

```bash
# Check for anomalies
curl -X POST http://localhost:8000/api/v1/anomalies/detect \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "R0VUIC8gSFRUUC8xLjENCkhvc3Q6IGV4YW1wbGUuY29tDQoNCg==",
    "encoding": "base64"
  }'
```

### 9.7 OpenAPI Documentation

Access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### 9.8 Using HTTPie (Alternative to curl)

```bash
# Install httpie
pip install httpie

# Health check
http GET localhost:8000/health

# Login
http POST localhost:8000/api/v1/auth/login username=admin password=admin123

# Authenticated request
http GET localhost:8000/api/v1/copilot/ "Authorization: Bearer $TOKEN"
```

---

## 10. Troubleshooting

### 10.1 Common Issues

#### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
python -m ai_engine --port 8001
```

#### Database Connection Failed

```bash
# Check if PostgreSQL is running
pg_isready -h localhost -p 5432

# Check Docker container
docker ps | grep postgres

# View PostgreSQL logs
docker logs cronos-postgres

# Test connection
psql -h localhost -U cronos -d cronos_ai_dev -c "SELECT 1"
```

#### Redis Connection Failed

```bash
# Check if Redis is running
redis-cli ping

# Check Docker container
docker ps | grep redis

# View Redis logs
docker logs cronos-redis
```

#### Module Import Errors

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

#### Migration Errors

```bash
# Check current state
alembic current

# Show history
alembic history

# Stamp to specific version (if needed)
alembic stamp head

# Create new migration
alembic revision --autogenerate -m "description"
```

### 10.2 Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=true

# Run with verbose output
python -m ai_engine --mode development --log-level DEBUG
```

### 10.3 Check Service Health

```bash
# Check all endpoints
curl http://localhost:8000/health && echo "✓ Health OK"
curl http://localhost:8000/healthz && echo "✓ Liveness OK"
curl http://localhost:8000/readyz && echo "✓ Readiness OK"
```

### 10.4 View Application Logs

```bash
# If running via Docker
docker-compose logs -f ai-engine

# If running directly
tail -f /var/log/cronos-ai/app.log

# Or check stdout/stderr during development
```

### 10.5 Reset Local Environment

```bash
# Stop all services
docker-compose down -v

# Remove Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Remove virtual environment
rm -rf venv

# Fresh start
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Restart services
docker-compose up -d
```

---

## 11. macOS Deployment (Recommended)

For macOS users, we provide a comprehensive deployment script that handles all prerequisites, setup, and service management automatically.

### 11.1 Quick Start with macOS Script

```bash
# Make the script executable (if not already)
chmod +x scripts/deploy-macos.sh

# Install all prerequisites (Homebrew, Python, Docker, PostgreSQL, Redis, Ollama)
./scripts/deploy-macos.sh install

# Setup environment and start services
./scripts/deploy-macos.sh setup
./scripts/deploy-macos.sh start

# Check status
./scripts/deploy-macos.sh status
```

### 11.2 Available Commands

| Command | Description |
|---------|-------------|
| `install` | Install all prerequisites (Homebrew, Python, Docker, PostgreSQL, Redis, Ollama) |
| `setup` | Initial setup (create virtualenv, install deps, setup database, create .env) |
| `start` | Start all services |
| `stop` | Stop all services |
| `restart` | Restart all services |
| `status` | Show service status dashboard |
| `logs` | View logs (supports --follow and --service flags) |
| `health` | Run comprehensive health checks |
| `clean` | Clean up containers, volumes, and cache |
| `uninstall` | Complete uninstall |

### 11.3 Deployment Modes

```bash
# Quick start with Docker Compose only (default)
./scripts/deploy-macos.sh start --quick

# Full stack with monitoring (Prometheus, Grafana)
./scripts/deploy-macos.sh start --full

# Run Python natively (Docker only for dependencies)
./scripts/deploy-macos.sh start --native

# Development mode with hot reload
./scripts/deploy-macos.sh start --dev
```

### 11.4 One-Line Quick Start

```bash
# Clone, setup, and start everything
git clone https://github.com/qbitel/cronos-ai.git && \
cd cronos-ai && \
chmod +x scripts/deploy-macos.sh && \
./scripts/deploy-macos.sh install && \
./scripts/deploy-macos.sh setup && \
./scripts/deploy-macos.sh start
```

### 11.5 Apple Silicon vs Intel

The deployment script automatically detects your Mac's architecture:
- **Apple Silicon (M1/M2/M3)**: Uses ARM64-optimized Docker images
- **Intel Macs**: Uses standard AMD64 images

No manual configuration required.

### 11.6 Service Management

```bash
# View status dashboard
./scripts/deploy-macos.sh status

# Follow logs in real-time
./scripts/deploy-macos.sh logs --follow

# View logs for specific service
./scripts/deploy-macos.sh logs --service ai-engine

# Run health checks
./scripts/deploy-macos.sh health

# Stop everything
./scripts/deploy-macos.sh stop

# Clean reset (removes all data)
./scripts/deploy-macos.sh clean --all
```

### 11.7 Troubleshooting macOS

```bash
# Skip prerequisites check
./scripts/deploy-macos.sh start --skip-prereq

# Force reinstall/recreate
./scripts/deploy-macos.sh setup --force

# View detailed help
./scripts/deploy-macos.sh --help
```

---

## Appendix A: Quick Reference Commands

```bash
# Start everything
docker-compose -f ops/deploy/docker/docker-compose.yml up -d

# Stop everything
docker-compose -f ops/deploy/docker/docker-compose.yml down

# View logs
docker-compose logs -f

# Run tests
pytest ai_engine/tests/ -v

# Apply migrations
cd ai_engine && alembic upgrade head

# Create admin user
python -m ai_engine.scripts.create_admin --username admin --password admin123

# Get auth token
curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' | jq -r '.access_token'

# Health check
curl http://localhost:8000/health
```

---

## Appendix B: IDE Setup

### VS Code

Install recommended extensions:
- Python
- Pylance
- Docker
- YAML
- REST Client

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["ai_engine/tests"],
  "editor.formatOnSave": true
}
```

### PyCharm

1. Open project folder
2. Configure Python interpreter: `File > Settings > Project > Python Interpreter`
3. Select `venv/bin/python`
4. Enable pytest: `File > Settings > Tools > Python Integrated Tools > Default test runner: pytest`

---

## Appendix C: Useful Scripts

### Start Script (`scripts/start-local.sh`)

```bash
#!/bin/bash
set -e

echo "Starting CRONOS AI local environment..."

# Start dependencies
docker-compose -f ops/deploy/docker/docker-compose.yml up -d postgres redis

# Wait for database
echo "Waiting for database..."
sleep 5

# Apply migrations
cd ai_engine
alembic upgrade head
cd ..

# Start application
source venv/bin/activate
python -m ai_engine --mode development
```

### Test Script (`scripts/run-tests.sh`)

```bash
#!/bin/bash
set -e

source venv/bin/activate

echo "Running linting..."
ruff check ai_engine/

echo "Running type checks..."
mypy ai_engine/ --ignore-missing-imports

echo "Running tests..."
pytest ai_engine/tests/ -v --cov=ai_engine --cov-report=term-missing

echo "All checks passed!"
```

---

**Document maintained by**: CRONOS AI Engineering Team
**Questions?** Open an issue at https://github.com/qbitel/cronos-ai/issues
