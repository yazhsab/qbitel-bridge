SHELL := /bin/bash
RUSTFLAGS :=
GOFLAGS := -trimpath
PYTHON := python3

.PHONY: all bootstrap build test run-bridge run-control ui sbom sign lint scan provenance docker-images \
        install test-python lint-python clean

all: build

bootstrap:
	@echo "[*] Installing Python dependencies..."
	$(PYTHON) -m pip install -e ".[dev]"
	@echo "[*] Installing pre-commit hooks..."
	pre-commit install || true
	@echo "[*] Bootstrap complete."

install:
	$(PYTHON) -m pip install -e ".[dev]"

build:
	@echo "[*] Building Rust dataplane..."
	cd rust/dataplane && cargo build --locked
	@echo "[*] Building Go services..."
	cd go/controlplane && go build $(GOFLAGS) -o ../../dist/controlplane ./cmd/controlplane
	cd go/mgmtapi && go build $(GOFLAGS) -o ../../dist/mgmtapi ./cmd/mgmtapi
	cd go/agents/device-agent && go build $(GOFLAGS) -o ../../../dist/device-agent ./ || true

test: test-python
	cd rust/dataplane && cargo test
	cd go/controlplane && go test ./...
	cd go/mgmtapi && go test ./...

test-python:
	@echo "[*] Running Python tests..."
	$(PYTHON) -m pytest ai_engine/tests/ -v --tb=short

lint: lint-python
	@echo "[*] Rust: clippy (deny warnings)"
	cd rust/dataplane && cargo clippy --all-targets --all-features -- -D warnings
	@echo "[*] Go: golangci-lint"
	command -v golangci-lint >/dev/null 2>&1 || (echo "Install golangci-lint: https://golangci-lint.run/usage/install/" && exit 1)
	cd go/controlplane && golangci-lint run ./...
	cd go/mgmtapi && golangci-lint run ./...
	cd go/agents/device-agent && golangci-lint run ./... || true
	@echo "[*] Go: gosec"
	command -v gosec >/dev/null 2>&1 || (echo "Install gosec: go install github.com/securego/gosec/v2/cmd/gosec@latest" && exit 1)
	cd go/controlplane && gosec ./...
	cd go/mgmtapi && gosec ./...
	cd go/agents/device-agent && gosec ./... || true

lint-python:
	@echo "[*] Python: flake8"
	$(PYTHON) -m flake8 ai_engine/ --count --select=E9,F63,F7,F82 --show-source --statistics
	@echo "[*] Python: mypy"
	$(PYTHON) -m mypy ai_engine/ --ignore-missing-imports
	@echo "[*] Python: bandit"
	$(PYTHON) -m bandit -r ai_engine/ -ll

clean:
	@echo "[*] Cleaning build artifacts..."
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache htmlcov/
	cd rust/dataplane && cargo clean
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

docker-images:
	@echo "[*] Building local Docker images for scanning..."
	docker build -f ops/deploy/docker/dockerfiles/controlplane.Dockerfile -t qbitel/controlplane:local .
	docker build -f ops/deploy/docker/dockerfiles/mgmtapi.Dockerfile -t qbitel/mgmtapi:local .

run-bridge:
	cd rust/dataplane && cargo run --bin qbitelai-bridge

run-control:
	./dist/controlplane || (cd go/controlplane && go run ./cmd/controlplane)

run-ai-engine:
	$(PYTHON) -m ai_engine --host 0.0.0.0 --port 8000

ui:
	cd ui/console && npm install && npm run dev

sbom:
	@mkdir -p .sbom
	@command -v syft >/dev/null 2>&1 && syft packages dir:. -o cyclonedx-json=.sbom/sbom.json || echo "Install syft to generate SBOM"

scan:
	@echo "[*] cargo-audit (Rust dependencies)"
	command -v cargo-audit >/dev/null 2>&1 || (echo "Install cargo-audit: cargo install cargo-audit" && exit 1)
	cd rust/dataplane && cargo audit
	@echo "[*] Trivy FS scan (CRITICAL must be 0)"
	command -v trivy >/dev/null 2>&1 || (echo "Install trivy: https://aquasecurity.github.io/trivy/v0.51/getting-started/installation/" && exit 1)
	TRIVY_NON_SSL=true trivy fs --no-progress --scanners vuln --severity CRITICAL --exit-code 1 .
	$(MAKE) docker-images
	@echo "[*] Trivy image scan (controlplane)"
	TRIVY_NON_SSL=true trivy image --severity CRITICAL --exit-code 1 qbitel/controlplane:local
	@echo "[*] Trivy image scan (mgmtapi)"
	TRIVY_NON_SSL=true trivy image --severity CRITICAL --exit-code 1 qbitel/mgmtapi:local

sign:
	@echo "[*] Cosign keyless signing sample (requires GitHub OIDC, GHCR push)"
	@echo "Login to GHCR:  echo $$GITHUB_TOKEN | docker login ghcr.io -u $$GITHUB_ACTOR --password-stdin"
	@echo "Build and push images with tags: ghcr.io/ORG/REPO/controlplane:TAG and mgmtapi:TAG"
	@echo "Enable OIDC in repo: Settings → Actions → Workflow permissions → Allow GitHub Actions to create and approve pull requests; also enable OIDC token requests."
	@echo "Then sign: COSIGN_EXPERIMENTAL=1 cosign sign --yes ghcr.io/ORG/REPO/controlplane:TAG"
	@echo "And attest: COSIGN_EXPERIMENTAL=1 cosign attest --yes --predicate policy/attestations/predicate-template.json --type slsaprovenance ghcr.io/ORG/REPO/controlplane:TAG"

provenance:
	@echo "[*] Generating provenance attestation for local images (requires cosign)"
	command -v cosign >/dev/null 2>&1 || (echo "Install cosign: https://docs.sigstore.dev/cosign/installation/" && exit 1)
	COSIGN_EXPERIMENTAL=1 cosign attest --predicate policy/attestations/predicate-template.json --type slsaprovenance qbitel/controlplane:local || true
	COSIGN_EXPERIMENTAL=1 cosign attest --predicate policy/attestations/predicate-template.json --type slsaprovenance qbitel/mgmtapi:local || true
