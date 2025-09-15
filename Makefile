SHELL := /bin/bash
RUSTFLAGS :=
GOFLAGS := -trimpath

.PHONY: all bootstrap build test run-bridge run-control ui sbom sign lint scan provenance docker-images

all: build

bootstrap:
	@echo "[*] Bootstrap complete (add pre-commit, hooks as needed)."

build:
	@echo "[*] Building Rust dataplane..."
	cd rust/dataplane && cargo build --locked
	@echo "[*] Building Go services..."
	cd go/controlplane && go build $(GOFLAGS) -o ../../dist/controlplane ./cmd/controlplane
	cd go/mgmtapi && go build $(GOFLAGS) -o ../../dist/mgmtapi ./cmd/mgmtapi
	cd go/agents/device-agent && go build $(GOFLAGS) -o ../../../dist/device-agent ./ || true

test:
	cd rust/dataplane && cargo test
	cd go/controlplane && go test ./...
	cd go/mgmtapi && go test ./...

lint:
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

docker-images:
	@echo "[*] Building local Docker images for scanning..."
	docker build -f ops/deploy/docker/dockerfiles/controlplane.Dockerfile -t cronosai/controlplane:local .
	docker build -f ops/deploy/docker/dockerfiles/mgmtapi.Dockerfile -t cronosai/mgmtapi:local .

run-bridge:
	cd rust/dataplane && cargo run --bin cronosai-bridge

run-control:
	./dist/controlplane || (cd go/controlplane && go run ./cmd/controlplane)

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
	TRIVY_NON_SSL=true trivy image --severity CRITICAL --exit-code 1 cronosai/controlplane:local
	@echo "[*] Trivy image scan (mgmtapi)"
	TRIVY_NON_SSL=true trivy image --severity CRITICAL --exit-code 1 cronosai/mgmtapi:local

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
	COSIGN_EXPERIMENTAL=1 cosign attest --predicate policy/attestations/predicate-template.json --type slsaprovenance cronosai/controlplane:local || true
	COSIGN_EXPERIMENTAL=1 cosign attest --predicate policy/attestations/predicate-template.json --type slsaprovenance cronosai/mgmtapi:local || true
