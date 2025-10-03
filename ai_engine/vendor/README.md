# CRONOS AI - Vendor Directory

This directory contains vendor shims and fallback implementations for air-gapped deployments where external dependencies may not be available.

## Purpose

In air-gapped or restricted network environments, CRONOS AI can operate with limited functionality using vendor shims that provide basic implementations of optional dependencies.

## Structure

```
vendor/
├── README.md                 # This file
├── __init__.py              # Vendor module initialization
├── llm_shims.py             # LLM provider shims
├── ml_shims.py              # ML/DL framework shims
└── monitoring_shims.py      # Monitoring framework shims
```

## Usage

Vendor shims are automatically loaded by the dependency manager when optional dependencies are not available. No manual configuration is required.

## Air-Gapped Deployment

For air-gapped deployments:

1. **Full Installation (Recommended)**:
   ```bash
   # Download all dependencies on a connected machine
   pip download -r requirements.txt -d ./vendor/packages
   
   # Transfer to air-gapped environment and install
   pip install --no-index --find-links=./vendor/packages -r requirements.txt
   ```

2. **Minimal Installation (Fallback Mode)**:
   ```bash
   # Install only required dependencies
   pip install -r requirements.txt --no-deps
   
   # CRONOS AI will use vendor shims for missing optional dependencies
   ```

3. **Verification**:
   ```bash
   python scripts/check_dependencies.py
   ```

## Vendor Shim Capabilities

### LLM Provider Shims
- **OpenAI Shim**: Provides mock responses for testing without OpenAI API
- **Anthropic Shim**: Provides mock responses for testing without Anthropic API
- **Ollama Shim**: Provides local fallback for basic text processing

### ML/DL Shims
- **TorchCRF Fallback**: Alternative implementations for sequence labeling
- **Lightweight Models**: Reduced-size models for resource-constrained environments

### Monitoring Shims
- **Metrics Collector**: In-memory metrics when Prometheus is unavailable
- **Logger Fallback**: File-based logging when external logging is unavailable

## Security Considerations

Vendor shims are designed for:
- Development and testing environments
- Air-gapped deployments with limited functionality
- Graceful degradation when optional features are unavailable

**Production deployments should use full dependencies for optimal security and performance.**

## Contributing

When adding new vendor shims:

1. Implement the same interface as the original dependency
2. Provide clear warnings when shims are in use
3. Document limitations and reduced functionality
4. Include tests for shim implementations

## License

Vendor shims are part of CRONOS AI and follow the same license terms.