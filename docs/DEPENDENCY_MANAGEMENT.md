# QBITEL - Dependency Management Guide

## Overview

QBITEL uses a structured approach to dependency management that supports:
- **Reproducible builds** with pinned versions
- **Optional dependencies** with graceful fallbacks
- **Air-gapped deployments** with vendor shims
- **Development and production** separation

## Requirements Files Structure

### Core Requirements (`requirements.txt`)

Contains all runtime dependencies with pinned versions:
- Core framework (FastAPI, Pydantic, etc.)
- LLM provider SDKs (OpenAI, Anthropic, Ollama) - **optional**
- Machine Learning (PyTorch, TorchCRF)
- Monitoring (Prometheus, psutil)
- Database and caching
- Security and authentication

**Installation**:
```bash
pip install -r requirements.txt
```

### Development Requirements (`requirements-dev.txt`)

Additional packages for development:
- Testing frameworks (pytest, pytest-asyncio)
- Code quality tools (black, flake8, mypy)
- Security scanning (bandit, safety)
- Documentation (sphinx)
- Development utilities (ipython, pre-commit)

**Installation**:
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

### Production Requirements (`requirements-prod.txt`)

Production-specific packages:
- APM tools (Datadog, New Relic, Sentry)
- Production logging (JSON logging, Loguru)
- Secrets management (HashiCorp Vault)
- Production databases (TimescaleDB)
- Message queues (Kafka, RabbitMQ)

**Installation**:
```bash
pip install -r requirements.txt -r requirements-prod.txt
```

### Copilot Requirements (`requirements-copilot.txt`)

**Legacy file** - being phased out. Features now integrated into main requirements.

## Dependency Categories

### Required Dependencies

These must be installed for QBITEL to function:

| Package | Purpose | Min Version |
|---------|---------|-------------|
| torch | Deep learning framework | 2.1.0 |
| torchcrf | Sequence labeling (CRF layer) | 1.1.0 |
| fastapi | Web framework | 0.104.0 |
| psutil | System monitoring | 5.9.0 |
| prometheus-client | Metrics collection | 0.19.0 |

### Optional Dependencies (LLM Providers)

These enable specific LLM features but are not required:

| Package | Purpose | Fallback Behavior |
|---------|---------|-------------------|
| openai | OpenAI GPT-4 integration | Uses alternative providers |
| anthropic | Anthropic Claude integration | Uses alternative providers |
| ollama | Local LLM support | Uses cloud providers |

**Note**: At least one LLM provider should be configured for full functionality.

## Dependency Verification

### Automatic Checking

QBITEL automatically checks dependencies on startup:

```python
from ai_engine.core.dependency_manager import check_dependencies

# Returns True if all required dependencies are available
is_valid = check_dependencies()
```

### Manual Verification

Run the dependency check script:

```bash
python scripts/check_dependencies.py
```

**Output Example**:
```
================================================================================
QBITEL - Dependency Installation Report
================================================================================

Required Dependencies:
--------------------------------------------------------------------------------
  ✓ PyTorch                      v2.1.2          [available]
  ✓ TorchCRF                     v1.1.0          [available]
  ✓ psutil                       v5.9.6          [available]
  ✓ Prometheus Client            v0.19.0         [available]

Optional Dependencies (LLM Providers):
--------------------------------------------------------------------------------
  ✓ OpenAI SDK                   v1.10.0         [available]
  ✓ Anthropic SDK                v0.8.1          [available]
  ○ Ollama SDK                   N/A             [missing]

================================================================================
✓ All required dependencies are installed
================================================================================
```

## Air-Gapped Deployments

### Full Installation Method

1. **On connected machine**, download all dependencies:
```bash
pip download -r requirements.txt -d ./vendor/packages
pip download -r requirements-prod.txt -d ./vendor/packages
```

2. **Transfer** the `vendor/packages` directory to air-gapped environment

3. **On air-gapped machine**, install from local packages:
```bash
pip install --no-index --find-links=./vendor/packages -r requirements.txt
pip install --no-index --find-links=./vendor/packages -r requirements-prod.txt
```

### Minimal Installation (Fallback Mode)

For highly restricted environments:

```bash
# Install only required dependencies
pip install torch torchcrf fastapi psutil prometheus-client

# QBITEL will use vendor shims for missing optional features
```

**Limitations in fallback mode**:
- No LLM provider integration (uses mock responses)
- Limited monitoring capabilities
- Reduced functionality for optional features

## Vendor Shims

Located in [`ai_engine/vendor/`](../ai_engine/vendor/), vendor shims provide fallback implementations when optional dependencies are unavailable.

### Available Shims

- **LLM Provider Shims**: Mock implementations for testing without API access
- **ML/DL Shims**: Alternative implementations for resource-constrained environments
- **Monitoring Shims**: In-memory metrics when external systems unavailable

### Usage

Vendor shims are automatically loaded by the dependency manager. No configuration required.

## Dependency Updates

### Updating Pinned Versions

1. **Test updates** in development environment:
```bash
pip install --upgrade openai anthropic
pip freeze > requirements-new.txt
```

2. **Run tests** to verify compatibility:
```bash
pytest tests/
```

3. **Update** `requirements.txt` with new versions

4. **Commit** changes with clear documentation

### Security Updates

For security vulnerabilities:

```bash
# Check for vulnerabilities
pip-audit

# Or use safety
safety check -r requirements.txt
```

## Docker Builds

The Dockerfile automatically:
1. Installs core and production requirements
2. Verifies dependencies
3. Provides warnings for missing optional dependencies

**Build command**:
```bash
docker build -f ai_engine/deployment/docker/Dockerfile -t qbitel:latest .
```

## Troubleshooting

### TorchCRF Import Issues

If you encounter `ImportError: No module named 'torchcrf'`:

**Try alternative package names**:
```bash
pip install TorchCRF  # Capitalized version
# or
pip install torch-crf  # Hyphenated version
```

The dependency manager will automatically try fallback package names.

### PyTorch CPU vs GPU

**CPU version** (default):
```bash
pip install torch==2.1.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

**GPU version** (CUDA 11.8):
```bash
pip install torch==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### LLM Provider API Keys

Configure via environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OLLAMA_HOST="http://localhost:11434"
```

Or in configuration file:
```yaml
# config/qbitel.yaml
llm:
  openai_api_key: ${OPENAI_API_KEY}
  anthropic_api_key: ${ANTHROPIC_API_KEY}
  ollama_host: ${OLLAMA_HOST:-http://localhost:11434}
```

## Best Practices

### Development

1. **Use virtual environments**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. **Install development requirements**:
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

3. **Run pre-commit hooks**:
```bash
pre-commit install
pre-commit run --all-files
```

### Production

1. **Use pinned versions** from `requirements.txt`
2. **Install production requirements** for monitoring and APM
3. **Verify dependencies** before deployment
4. **Use Docker** for consistent environments

### CI/CD

1. **Cache dependencies** for faster builds
2. **Run security scans** on every commit
3. **Test with minimal dependencies** to ensure fallbacks work
4. **Verify air-gapped installation** in staging

## Support

For dependency-related issues:

1. **Check** dependency status: `python scripts/check_dependencies.py`
2. **Review** logs for import errors
3. **Consult** this documentation
4. **Contact** QBITEL support team

## References

- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)