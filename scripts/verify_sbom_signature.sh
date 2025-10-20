#!/bin/bash
#
# CRONOS AI - Verify SBOM Signature
# Verifies SBOM authenticity using Cosign signatures
#
# Usage: ./verify_sbom_signature.sh <sbom-file>
# Example: ./verify_sbom_signature.sh cronos-ai-engine-spdx.json
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if cosign is installed
if ! command -v cosign &> /dev/null; then
    log_error "cosign is not installed"
    log_info "Install cosign: https://docs.sigstore.dev/cosign/installation/"
    exit 1
fi

# Check arguments
if [ $# -lt 1 ]; then
    log_error "Usage: $0 <sbom-file>"
    exit 1
fi

SBOM_FILE="$1"
BUNDLE_FILE="${SBOM_FILE}.bundle"

# Validate files exist
if [ ! -f "$SBOM_FILE" ]; then
    log_error "SBOM file not found: $SBOM_FILE"
    exit 1
fi

if [ ! -f "$BUNDLE_FILE" ]; then
    log_error "Bundle file not found: $BUNDLE_FILE"
    log_info "SBOM signatures are only available for official releases"
    exit 1
fi

log_info "Verifying SBOM: $SBOM_FILE"
log_info "Bundle: $BUNDLE_FILE"

# Verify signature
# Note: Adjust certificate-identity-regexp and certificate-oidc-issuer based on your setup
if cosign verify-blob \
    --bundle "$BUNDLE_FILE" \
    --certificate-identity-regexp=".*@cronos-ai.com" \
    --certificate-oidc-issuer="https://github.com/login/oauth" \
    "$SBOM_FILE" &> /dev/null; then

    log_info "✅ SBOM signature verification PASSED"
    log_info "This SBOM is authentic and has not been tampered with"
    exit 0
else
    # Try alternative verification (for GitHub Actions)
    if cosign verify-blob \
        --bundle "$BUNDLE_FILE" \
        --certificate-identity-regexp=".*github.*" \
        --certificate-oidc-issuer="https://token.actions.githubusercontent.com" \
        "$SBOM_FILE" &> /dev/null; then

        log_info "✅ SBOM signature verification PASSED (GitHub Actions)"
        log_info "This SBOM is authentic and has not been tampered with"
        exit 0
    else
        log_error "❌ SBOM signature verification FAILED"
        log_error "This SBOM may have been tampered with or is not from an official source"
        exit 1
    fi
fi
