#!/bin/bash
#
# CRONOS AI - Upload SBOM to Dependency-Track
# Automates SBOM upload for continuous vulnerability monitoring
#
# Usage: ./upload_sbom_to_dependency_track.sh [VERSION]
# Example: ./upload_sbom_to_dependency_track.sh v1.0.0
#

set -euo pipefail

# Configuration
DEPENDENCY_TRACK_URL="${DEPENDENCY_TRACK_URL:-https://dependency-track.cronos-ai.internal}"
API_KEY="${DEPENDENCY_TRACK_API_KEY:-}"
PROJECT_VERSION="${1:-latest}"
SBOM_DIR="${SBOM_DIR:-./sbom-artifacts}"

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

# Validate prerequisites
if [ -z "$API_KEY" ]; then
    log_error "DEPENDENCY_TRACK_API_KEY environment variable is not set"
    exit 1
fi

if [ ! -d "$SBOM_DIR" ]; then
    log_error "SBOM directory not found: $SBOM_DIR"
    exit 1
fi

log_info "Starting SBOM upload to Dependency-Track"
log_info "URL: $DEPENDENCY_TRACK_URL"
log_info "Version: $PROJECT_VERSION"

# Find all CycloneDX SBOMs (Dependency-Track prefers CycloneDX)
sbom_count=0
success_count=0
fail_count=0

for sbom in "$SBOM_DIR"/*-cyclonedx.json; do
    if [ ! -f "$sbom" ]; then
        log_warn "No CycloneDX SBOMs found in $SBOM_DIR"
        continue
    fi

    component=$(basename "$sbom" | sed 's/-cyclonedx.json//')
    sbom_count=$((sbom_count + 1))

    log_info "Uploading $component..."

    # Encode SBOM to base64
    sbom_base64=$(base64 -w0 < "$sbom" 2>/dev/null || base64 < "$sbom")

    # Create project if it doesn't exist
    project_uuid=$(curl -s -X GET "$DEPENDENCY_TRACK_URL/api/v1/project/lookup?name=$component&version=$PROJECT_VERSION" \
        -H "X-Api-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        2>/dev/null | jq -r '.uuid // empty' || echo "")

    if [ -z "$project_uuid" ]; then
        log_info "Creating new project: $component"
        project_uuid=$(curl -s -X PUT "$DEPENDENCY_TRACK_URL/api/v1/project" \
            -H "X-Api-Key: $API_KEY" \
            -H "Content-Type: application/json" \
            -d "{
                \"name\": \"$component\",
                \"version\": \"$PROJECT_VERSION\",
                \"classifier\": \"APPLICATION\",
                \"active\": true,
                \"tags\": [{\"name\": \"cronos-ai\"}]
            }" \
            2>/dev/null | jq -r '.uuid // empty' || echo "")

        if [ -z "$project_uuid" ]; then
            log_error "Failed to create project for $component"
            fail_count=$((fail_count + 1))
            continue
        fi
    fi

    # Upload SBOM
    upload_response=$(curl -s -w "\n%{http_code}" -X PUT "$DEPENDENCY_TRACK_URL/api/v1/bom" \
        -H "X-Api-Key: $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"project\": \"$project_uuid\",
            \"bom\": \"$sbom_base64\"
        }" 2>/dev/null || echo "")

    http_code=$(echo "$upload_response" | tail -n1)
    response_body=$(echo "$upload_response" | sed '$d')

    if [ "$http_code" = "200" ]; then
        token=$(echo "$response_body" | jq -r '.token // empty')
        log_info "✓ Successfully uploaded $component (token: $token)"
        success_count=$((success_count + 1))
    else
        log_error "✗ Failed to upload $component (HTTP $http_code)"
        log_error "Response: $response_body"
        fail_count=$((fail_count + 1))
    fi
done

# Summary
log_info "═══════════════════════════════════════"
log_info "Upload Summary:"
log_info "  Total:   $sbom_count"
log_info "  Success: $success_count"
log_info "  Failed:  $fail_count"
log_info "═══════════════════════════════════════"

if [ $fail_count -gt 0 ]; then
    exit 1
fi

exit 0
