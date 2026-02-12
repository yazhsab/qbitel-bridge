# QBITEL Bridge - Development Guide

This guide provides information for developers who want to contribute to QBITEL Bridge or extend its functionality.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Adding New Features](#adding-new-features)
- [API Development](#api-development)
- [Contributing Guidelines](#contributing-guidelines)

## Development Setup

### Prerequisites

| Component | Requirement | Purpose |
|-----------|------------|---------|
| Python | 3.10+ | AI Engine, compliance, security |
| Rust | 1.70+ (with cargo) | Data plane, PQC-TLS |
| Go | 1.21+ | Control plane, mgmt API |
| Node.js | 18+ (with npm) | UI console |
| Docker | 20.10+ | Containerized development |
| Git | 2.30+ | Version control |

### Initial Setup (All Components)

```bash
# 1. Clone the repository
git clone https://github.com/yazhsab/qbitel-bridge.git
cd qbitel-bridge

# 2. Build everything with make
make build

# 3. Run all tests
make test
```

### Python AI Engine Setup

```bash
# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r ai_engine/requirements.txt

# Install pre-commit hooks (optional)
pre-commit install

# Verify installation
pytest ai_engine/tests/ -v --tb=short
```

### Rust Data Plane Setup

```bash
# Build the data plane
cd rust/dataplane && cargo build --locked

# Run tests
cargo test

# Run linter
cargo clippy --all-targets --all-features -- -D warnings

# Format code
cargo fmt
```

### Go Services Setup

```bash
# Build control plane
cd go/controlplane && go build -trimpath -o ../../dist/controlplane ./cmd/controlplane

# Build management API
cd go/mgmtapi && go build -trimpath -o ../../dist/mgmtapi ./cmd/mgmtapi

# Run tests
cd go/controlplane && go test ./...
cd go/mgmtapi && go test ./...
```

### UI Console Setup

```bash
cd ui/console
npm install
npm run dev
# Console available at http://localhost:3000
```

### IDE Configuration

#### VS Code

Create `.vscode/settings.json`:

```json
{
  "python.pythonPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "ai_engine/tests"
  ],
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

#### PyCharm

- Set Python interpreter to `venv/bin/python`
- Configure pytest as default test runner
- Enable Black formatter
- Enable flake8 linter

## Project Structure

```
qbitel-bridge/
├── ai_engine/                 # Python AI Engine
│   ├── core/                  # Main engine, config, orchestrator
│   ├── agents/                # Multi-agent orchestration system
│   ├── discovery/             # Protocol discovery (PCFG, transformers)
│   ├── detection/             # Field detection (BiLSTM-CRF), anomaly detection
│   ├── security/              # Zero-touch decision engine
│   ├── legacy/                # Legacy System Whisperer (COBOL, JCL)
│   ├── marketplace/           # Protocol marketplace (Stripe, S3)
│   ├── copilot/               # Protocol intelligence copilot
│   ├── crypto/                # Post-quantum cryptography (Python)
│   ├── compliance/            # Compliance automation (9 frameworks)
│   ├── llm/                   # LLM service (Ollama, RAG, providers)
│   ├── cloud_native/          # Service mesh, container security, cloud SDKs
│   ├── api/                   # REST (FastAPI) and gRPC APIs
│   └── tests/                 # Test suite (unit, integration, perf)
│
├── rust/dataplane/            # Rust high-performance data plane
│   └── crates/pqc_tls/        # PQC-TLS (ML-KEM, ML-DSA, Falcon, SLH-DSA)
│
├── go/                        # Go services
│   ├── controlplane/          # Service orchestration
│   ├── mgmtapi/               # Management REST API
│   └── agents/device-agent/   # Edge device agent
│
├── ui/console/                # React admin console (TypeScript)
│
├── helm/qbitel-bridge/        # Helm chart for Kubernetes deployment
├── ops/                       # Operations (deploy, monitoring, secrets)
├── docker/                    # Docker configurations
├── kubernetes/                # Kubernetes manifests
├── samples/cobol/             # 500+ COBOL sample programs
├── demos/                     # End-to-end demo scenarios
├── tests/                     # Conformance, fuzz, and perf tests
├── security/                  # PKI, validation, policy
├── docs/                      # Full documentation suite
├── Makefile                   # Unified build system
├── requirements.txt           # Python production dependencies
└── requirements-dev.txt       # Python development dependencies
```

## Development Workflow

### Branch Strategy

```bash
# Main branches
main                 # Production-ready code
develop             # Integration branch

# Feature branches
git checkout -b feature/your-feature-name
git checkout -b bugfix/issue-description
git checkout -b hotfix/critical-fix
```

### Making Changes

```bash
# 1. Create feature branch
git checkout -b feature/new-protocol-support

# 2. Make changes and commit
git add .
git commit -m "feat: Add support for MQTT protocol discovery"

# 3. Run tests
pytest ai_engine/tests/ -v

# 4. Run code quality checks
flake8 ai_engine/
black ai_engine/ --check

# 5. Push changes
git push origin feature/new-protocol-support

# 6. Create Pull Request on GitHub
```

### Commit Message Convention

Follow conventional commits:

```
feat: Add new feature
fix: Bug fix
docs: Documentation changes
test: Add or update tests
refactor: Code refactoring
perf: Performance improvements
chore: Build/tooling changes
```

Examples:
```bash
git commit -m "feat: Add Dilithium-5 signature verification"
git commit -m "fix: Resolve memory leak in protocol parser"
git commit -m "test: Add integration tests for xDS server"
git commit -m "docs: Update API documentation for field detection"
```

## Testing

### Running All Tests

```bash
# Run all tests across all languages
make test
```

### Python Tests

```bash
# Run all Python tests
pytest ai_engine/tests/ -v

# Run specific test file
pytest ai_engine/tests/test_core.py -v

# Run tests by marker
pytest ai_engine/tests/ -v -m "cloud_native"
pytest ai_engine/tests/ -v -m "security"
pytest ai_engine/tests/ -v -m "integration"

# Run with coverage
pytest ai_engine/tests/ --cov=ai_engine --cov-report=html

# Run performance tests
pytest ai_engine/tests/performance/ -v --benchmark-only
```

### Rust Tests

```bash
cd rust/dataplane

# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_kyber_encapsulation
```

### Go Tests

```bash
# Control plane tests
cd go/controlplane && go test ./...

# Management API tests
cd go/mgmtapi && go test ./...

# With coverage
cd go/controlplane && go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

### UI Console Tests

```bash
cd ui/console
npm test
```

### Writing Tests

#### Unit Test Example

```python
# ai_engine/tests/test_protocol_discovery.py
import pytest
from ai_engine.discovery.pcfg_inference import PCFGInference

class TestPCFGInference:
    @pytest.fixture
    def pcfg_engine(self):
        return PCFGInference()

    def test_grammar_inference(self, pcfg_engine):
        """Test grammar inference from sample data."""
        samples = [b"GET /api HTTP/1.1", b"POST /data HTTP/1.1"]
        grammar = pcfg_engine.infer_grammar(samples)

        assert grammar is not None
        assert len(grammar.rules) > 0
        assert grammar.start_symbol == 'Request'

    def test_invalid_input(self, pcfg_engine):
        """Test handling of invalid input."""
        with pytest.raises(ValueError):
            pcfg_engine.infer_grammar([])
```

#### Integration Test Example

```python
# ai_engine/tests/integration/test_xds_server.py
import pytest
import grpc
from ai_engine.cloud_native.service_mesh.xds_server import XDSServer

@pytest.mark.integration
class TestXDSServerIntegration:
    @pytest.fixture
    async def xds_server(self):
        server = XDSServer(port=50051)
        await server.start()
        yield server
        await server.stop()

    @pytest.mark.asyncio
    async def test_stream_aggregated_resources(self, xds_server):
        """Test ADS streaming."""
        async with grpc.aio.insecure_channel('localhost:50051') as channel:
            # Test implementation
            pass
```

### Test Coverage Goals

- **Overall**: 80%+ coverage
- **Core components**: 90%+ coverage
- **Critical security**: 95%+ coverage
- **API endpoints**: 85%+ coverage

## Code Quality

### Linting (All Languages)

```bash
# Run all linters
make lint
```

### Python

```bash
# Format with Black
black ai_engine/
black ai_engine/ --check  # check only

# Lint with flake8
flake8 ai_engine/ --max-line-length=120 --ignore=E203,W503

# Type check with mypy
mypy ai_engine/
```

### Rust

```bash
cd rust/dataplane

# Lint with clippy (deny warnings)
cargo clippy --all-targets --all-features -- -D warnings

# Format
cargo fmt
cargo fmt -- --check  # check only

# Security audit
cargo audit
```

### Go

```bash
# Lint with golangci-lint
cd go/controlplane && golangci-lint run ./...
cd go/mgmtapi && golangci-lint run ./...

# Security scan with gosec
cd go/controlplane && gosec ./...
cd go/mgmtapi && gosec ./...

# Format
gofmt -w go/
```

### UI Console

```bash
cd ui/console
npm run lint
npm run format
```

## Adding New Features

### Adding a New Protocol

1. **Create protocol module**:

```python
# ai_engine/discovery/protocols/mqtt.py
from ai_engine.discovery.base_protocol import BaseProtocol

class MQTTProtocol(BaseProtocol):
    """MQTT protocol discovery implementation."""

    def __init__(self):
        super().__init__(name="MQTT", version="5.0")

    def parse(self, data: bytes) -> dict:
        """Parse MQTT protocol data."""
        # Implementation
        pass

    def validate(self, data: bytes) -> bool:
        """Validate MQTT message."""
        # Implementation
        pass
```

2. **Add tests**:

```python
# ai_engine/tests/test_mqtt_protocol.py
import pytest
from ai_engine.discovery.protocols.mqtt import MQTTProtocol

class TestMQTTProtocol:
    def test_parse_connect_message(self):
        protocol = MQTTProtocol()
        data = b"\x10\x0e\x00\x04MQTT\x05\x02\x00\x3c\x00"
        result = protocol.parse(data)
        assert result['type'] == 'CONNECT'
```

3. **Register protocol**:

```python
# ai_engine/discovery/protocol_registry.py
from ai_engine.discovery.protocols.mqtt import MQTTProtocol

PROTOCOLS = {
    'http': HTTPProtocol,
    'mqtt': MQTTProtocol,  # Add new protocol
    # ...
}
```

### Adding a New Cloud Platform Integration

1. **Create integration module**:

```python
# ai_engine/cloud_native/cloud_platforms/alibaba_cloud.py
from typing import List, Dict
import logging

class AlibabaCloudSecurityCenter:
    """Alibaba Cloud Security Center integration."""

    def __init__(self, access_key_id: str, access_key_secret: str):
        self.client = self._initialize_client(access_key_id, access_key_secret)
        self.logger = logging.getLogger(__name__)

    def publish_findings(self, findings: List[Dict]) -> bool:
        """Publish security findings."""
        # Implementation
        pass
```

2. **Add tests**:

```python
# ai_engine/tests/integration/test_alibaba_cloud.py
import pytest
from unittest.mock import Mock, patch
from ai_engine.cloud_native.cloud_platforms.alibaba_cloud import AlibabaCloudSecurityCenter

class TestAlibabaCloudIntegration:
    @patch('alibaba_cloud_sdk.Client')
    def test_publish_findings(self, mock_client):
        integration = AlibabaCloudSecurityCenter('key', 'secret')
        findings = [{'title': 'Test', 'severity': 'HIGH'}]
        result = integration.publish_findings(findings)
        assert result is True
```

## API Development

### Adding REST API Endpoint

```python
# ai_engine/api/rest_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class ProtocolRequest(BaseModel):
    data: bytes
    confidence_threshold: float = 0.7

class ProtocolResponse(BaseModel):
    protocol_type: str
    confidence: float
    fields: List[Dict]

@router.post("/discover", response_model=ProtocolResponse)
async def discover_protocol(request: ProtocolRequest):
    """
    Discover protocol from binary data.

    Args:
        request: Protocol discovery request

    Returns:
        Protocol type, confidence, and detected fields

    Raises:
        HTTPException: If discovery fails
    """
    try:
        # Implementation
        return ProtocolResponse(
            protocol_type="HTTP",
            confidence=0.95,
            fields=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Adding gRPC Service

```python
# ai_engine/api/grpc_server.py
import grpc
from concurrent import futures
from proto import qbitel_pb2, qbitel_pb2_grpc

class QbitelServicer(qbitel_pb2_grpc.QbitelServiceServicer):

    async def DiscoverProtocol(self, request, context):
        """gRPC method for protocol discovery."""
        try:
            # Implementation
            return qbitel_pb2.ProtocolResponse(
                protocol_type="HTTP",
                confidence=0.95
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return qbitel_pb2.ProtocolResponse()
```

## Contributing Guidelines

### Pull Request Process

1. **Fork the repository**
2. **Create feature branch** from `develop`
3. **Write code** following style guide
4. **Add tests** (aim for 80%+ coverage)
5. **Update documentation** if needed
6. **Run all tests** and quality checks
7. **Submit pull request** with clear description

### Pull Request Checklist

- [ ] Code follows project style guide
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] No merge conflicts
- [ ] CI/CD pipeline passes

### Code Review Guidelines

Reviewers will check:
- Code quality and readability
- Test coverage and quality
- Security implications
- Performance considerations
- Documentation completeness

### Getting Help

- **Documentation**: Check existing docs
- **Issues**: Search GitHub issues
- **Discussions**: Use GitHub Discussions
- **Email**: developers@qbitel.com

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or via environment:

```bash
export AI_ENGINE_LOG_LEVEL=DEBUG
python -m ai_engine
```

### Using Python Debugger

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use Python 3.7+ breakpoint()
breakpoint()
```

### Remote Debugging (VS Code)

Add to `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: AI Engine",
      "type": "python",
      "request": "launch",
      "module": "ai_engine",
      "console": "integratedTerminal",
      "env": {
        "AI_ENGINE_LOG_LEVEL": "DEBUG"
      }
    }
  ]
}
```

### Profiling

```python
# Using cProfile
python -m cProfile -o profile.stats -m ai_engine

# Analyze with snakeviz
pip install snakeviz
snakeviz profile.stats
```

## Performance Optimization

### Benchmarking

```python
# ai_engine/tests/performance/test_benchmarks.py
import pytest
from ai_engine.discovery import PCFGInference

@pytest.mark.benchmark
def test_protocol_discovery_performance(benchmark):
    """Benchmark protocol discovery."""
    engine = PCFGInference()
    samples = [b"GET /api HTTP/1.1"] * 100

    result = benchmark(engine.infer_grammar, samples)
    assert result is not None
```

### Profiling Async Code

```python
import asyncio
from ai_engine.core.engine import AIEngine

async def profile_async():
    engine = AIEngine()
    await engine.initialize()
    # Profile async operations

asyncio.run(profile_async())
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def discover_protocol(data: bytes, confidence_threshold: float = 0.7) -> dict:
    """
    Discover protocol from binary data.

    Args:
        data: Binary protocol data to analyze
        confidence_threshold: Minimum confidence score (0.0-1.0)

    Returns:
        Dictionary containing:
            - protocol_type: Detected protocol name
            - confidence: Confidence score (0.0-1.0)
            - fields: List of detected fields

    Raises:
        ValueError: If data is empty or invalid
        RuntimeError: If discovery engine is not initialized

    Example:
        >>> data = b"GET /api HTTP/1.1"
        >>> result = discover_protocol(data)
        >>> print(result['protocol_type'])
        'HTTP'
    """
    pass
```

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

**Happy Coding!** 🚀

For questions or support, reach out via:
- GitHub Issues: https://github.com/yazhsab/qbitel-bridge/issues
- Email: developers@qbitel.com

**Last Updated**: 2025-02-08
**Version**: 2.0
