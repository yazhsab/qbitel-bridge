# Contributing to QBITEL Bridge

Thank you for your interest in contributing to QBITEL Bridge! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

1. Check the [existing issues](https://github.com/yazhsab/qbitel-bridge/issues) to avoid duplicates
2. Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
3. Include steps to reproduce, expected behavior, and actual behavior
4. Include your environment details (OS, language versions, Docker version)

### Suggesting Features

1. Check existing issues and discussions first
2. Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
3. Describe the use case and expected behavior

### Submitting Pull Requests

1. Fork the repository and create a feature branch from `main`
2. Follow the coding standards below
3. Add tests for new functionality
4. Ensure all tests pass: `make test`
5. Update documentation if needed
6. Submit a PR using the [pull request template](.github/pull_request_template.md)

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/qbitel-bridge.git
cd qbitel-bridge

# Build all components
make build

# Run all tests
make test
```

### Component-Specific Setup

**Python AI Engine:**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r ai_engine/requirements.txt
pytest ai_engine/tests/ -v
```

**Rust Data Plane:**
```bash
cd rust/dataplane
cargo build --locked && cargo test
```

**Go Services:**
```bash
cd go/controlplane && go test ./...
cd go/mgmtapi && go test ./...
```

**UI Console:**
```bash
cd ui/console && npm install && npm test
```

## Coding Standards

### Python

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for function signatures
- Maximum line length: 120 characters
- Use `black` for formatting, `flake8` for linting
- Use Google-style docstrings

### Rust

- Follow standard Rust formatting (`cargo fmt`)
- Run `cargo clippy --all-targets -- -D warnings` before submitting
- Write doc comments for public APIs

### Go

- Follow standard Go formatting (`gofmt`)
- Run `golangci-lint run ./...` before submitting
- Run `gosec ./...` for security checks

### TypeScript/React

- Follow the existing ESLint configuration
- Use TypeScript strict mode
- Prefer functional components with hooks

### General

- Write descriptive commit messages using [conventional commits](https://www.conventionalcommits.org/)
- Keep PRs focused on a single change
- Add tests for new features and bug fixes
- Do not commit secrets, credentials, or sensitive data

## Commit Message Convention

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
git commit -m "feat: Add Falcon-1024 signature support to Rust PQC-TLS"
git commit -m "fix: Resolve memory leak in protocol parser"
git commit -m "test: Add integration tests for Go management API"
```

## Testing

All PRs must pass CI tests. Run the full suite before submitting:

```bash
# All tests
make test

# Linting
make lint

# Security scanning
make scan
```

## Project Structure

```
qbitel-bridge/
├── ai_engine/          # Core AI engine (Python)
│   ├── agents/         # Multi-agent orchestration
│   ├── compliance/     # Compliance reporting (9 frameworks)
│   ├── copilot/        # Protocol copilot
│   ├── crypto/         # Post-quantum cryptography
│   ├── discovery/      # Protocol discovery (PCFG, transformers)
│   ├── legacy/         # Legacy System Whisperer
│   ├── llm/            # LLM integrations (Ollama, RAG)
│   ├── marketplace/    # Protocol marketplace
│   ├── security/       # Zero-touch decision engine
│   └── tests/          # Python test suites
├── rust/dataplane/     # High-performance data plane (Rust)
│   └── crates/pqc_tls/ # PQC-TLS implementation
├── go/                 # Go services
│   ├── controlplane/   # Service orchestration
│   ├── mgmtapi/        # Management REST API
│   └── agents/         # Edge device agents
├── ui/console/         # Admin console (React/TypeScript)
├── helm/qbitel-bridge/ # Helm chart for Kubernetes
├── ops/                # Kubernetes, monitoring, secrets
├── tests/              # Conformance, fuzz, perf tests
└── docs/               # Documentation
```

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
