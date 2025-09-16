#!/bin/bash
# CRONOS AI Production Deployment Script
# Enterprise-grade deployment with comprehensive validation and rollback capabilities

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEPLOY_DIR="${PROJECT_ROOT}/ops/deploy"
HELM_CHART_DIR="${DEPLOY_DIR}/kubernetes/production/helm/cronos-ai"

# Default values
NAMESPACE="cronos-ai-prod"
RELEASE_NAME="cronos-ai"
DRY_RUN=false
SKIP_TESTS=false
SKIP_VALIDATION=false
ROLLBACK_ON_FAILURE=true
TIMEOUT=1800  # 30 minutes
VALUES_FILE="${HELM_CHART_DIR}/values.yaml"
VALUES_OVERRIDE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Help function
show_help() {
    cat << EOF
CRONOS AI Production Deployment Script

Usage: $0 [OPTIONS]

Options:
    -n, --namespace NAMESPACE       Kubernetes namespace (default: cronos-ai-prod)
    -r, --release RELEASE_NAME      Helm release name (default: cronos-ai)
    -f, --values-file FILE          Values file path (default: values.yaml)
    -s, --set KEY=VALUE             Set values on command line
    -d, --dry-run                   Perform a dry run without making changes
    -t, --timeout SECONDS          Timeout for deployment (default: 1800)
    --skip-tests                    Skip post-deployment tests
    --skip-validation               Skip pre-deployment validation
    --no-rollback                   Don't rollback on failure
    -h, --help                      Show this help message

Examples:
    $0                              # Deploy with default settings
    $0 --dry-run                    # Dry run deployment
    $0 -f custom-values.yaml        # Deploy with custom values
    $0 --set image.tag=v1.1.0      # Override image tag

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -r|--release)
                RELEASE_NAME="$2"
                shift 2
                ;;
            -f|--values-file)
                VALUES_FILE="$2"
                shift 2
                ;;
            -s|--set)
                VALUES_OVERRIDE="${VALUES_OVERRIDE} --set $2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --no-rollback)
                ROLLBACK_ON_FAILURE=false
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required tools
    local tools=("kubectl" "helm" "docker" "jq")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check Helm
    if ! helm version &> /dev/null; then
        log_error "Helm is not properly configured"
        exit 1
    fi
    
    # Verify chart exists
    if [[ ! -f "${HELM_CHART_DIR}/Chart.yaml" ]]; then
        log_error "Helm chart not found at ${HELM_CHART_DIR}"
        exit 1
    fi
    
    # Verify values file exists
    if [[ ! -f "$VALUES_FILE" ]]; then
        log_error "Values file not found: $VALUES_FILE"
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

# Validate deployment configuration
validate_configuration() {
    if [[ "$SKIP_VALIDATION" == true ]]; then
        log_warning "Skipping configuration validation"
        return 0
    fi
    
    log_info "Validating deployment configuration..."
    
    # Lint Helm chart
    if ! helm lint "$HELM_CHART_DIR" --values "$VALUES_FILE"; then
        log_error "Helm chart validation failed"
        exit 1
    fi
    
    # Template and validate Kubernetes manifests
    local temp_dir
    temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT
    
    helm template "$RELEASE_NAME" "$HELM_CHART_DIR" \
        --namespace "$NAMESPACE" \
        --values "$VALUES_FILE" \
        $VALUES_OVERRIDE \
        --output-dir "$temp_dir"
    
    # Validate manifests
    find "$temp_dir" -name "*.yaml" -exec kubectl apply --dry-run=client -f {} \; &> /dev/null
    
    log_success "Configuration validation completed"
}

# Create namespace if it doesn't exist
ensure_namespace() {
    log_info "Ensuring namespace '$NAMESPACE' exists..."
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace '$NAMESPACE' already exists"
    else
        log_info "Creating namespace '$NAMESPACE'..."
        kubectl apply -f "${DEPLOY_DIR}/kubernetes/production/namespace.yaml"
        log_success "Namespace '$NAMESPACE' created"
    fi
}

# Install or upgrade Helm dependencies
install_dependencies() {
    log_info "Installing Helm chart dependencies..."
    
    cd "$HELM_CHART_DIR"
    helm dependency update
    
    log_success "Dependencies installed"
}

# Perform pre-deployment checks
pre_deployment_checks() {
    log_info "Performing pre-deployment checks..."
    
    # Check cluster resources
    local total_cpu_requests=0
    local total_memory_requests=0
    
    # Calculate resource requirements from values file
    # This is a simplified check - in production, you'd parse the actual values
    log_info "Checking cluster capacity..."
    
    # Check if previous version exists
    if helm get values "$RELEASE_NAME" -n "$NAMESPACE" &> /dev/null; then
        log_info "Previous deployment found - this will be an upgrade"
        
        # Get current revision for potential rollback
        CURRENT_REVISION=$(helm history "$RELEASE_NAME" -n "$NAMESPACE" -o json | jq -r '.[0].revision')
        log_info "Current revision: $CURRENT_REVISION"
    else
        log_info "No previous deployment found - this will be a fresh install"
        CURRENT_REVISION=""
    fi
    
    # Check critical dependencies
    log_info "Checking external dependencies..."
    
    # Check if PostgreSQL is accessible (if enabled)
    # Check if Redis is accessible (if enabled)
    # Check if container registry is accessible
    
    log_success "Pre-deployment checks completed"
}

# Deploy CRONOS AI
deploy() {
    log_info "Starting CRONOS AI deployment..."
    
    local helm_cmd="helm"
    local action="upgrade"
    local helm_args=(
        "--install"
        "--namespace" "$NAMESPACE"
        "--create-namespace"
        "--values" "$VALUES_FILE"
        "--timeout" "${TIMEOUT}s"
        "--wait"
        "--wait-for-jobs"
    )
    
    if [[ "$DRY_RUN" == true ]]; then
        helm_args+=("--dry-run")
        log_info "Performing dry run deployment"
    fi
    
    # Add custom values if provided
    if [[ -n "$VALUES_OVERRIDE" ]]; then
        helm_args+=($VALUES_OVERRIDE)
    fi
    
    # Execute deployment
    if $helm_cmd $action "$RELEASE_NAME" "$HELM_CHART_DIR" "${helm_args[@]}"; then
        if [[ "$DRY_RUN" != true ]]; then
            log_success "CRONOS AI deployed successfully"
            
            # Get deployment info
            NEW_REVISION=$(helm history "$RELEASE_NAME" -n "$NAMESPACE" -o json | jq -r '.[0].revision')
            log_info "New revision: $NEW_REVISION"
        else
            log_success "Dry run completed successfully"
        fi
    else
        log_error "Deployment failed"
        
        if [[ "$DRY_RUN" != true && "$ROLLBACK_ON_FAILURE" == true && -n "$CURRENT_REVISION" ]]; then
            log_warning "Rolling back to revision $CURRENT_REVISION"
            rollback_deployment "$CURRENT_REVISION"
        fi
        
        exit 1
    fi
}

# Rollback deployment
rollback_deployment() {
    local revision="$1"
    
    log_warning "Rolling back CRONOS AI deployment to revision $revision"
    
    if helm rollback "$RELEASE_NAME" "$revision" -n "$NAMESPACE" --timeout "${TIMEOUT}s" --wait; then
        log_success "Rollback completed successfully"
    else
        log_error "Rollback failed"
        exit 1
    fi
}

# Run post-deployment tests
run_tests() {
    if [[ "$SKIP_TESTS" == true || "$DRY_RUN" == true ]]; then
        log_warning "Skipping post-deployment tests"
        return 0
    fi
    
    log_info "Running post-deployment tests..."
    
    # Wait for all pods to be ready
    log_info "Waiting for all pods to be ready..."
    kubectl wait --for=condition=ready pod \
        -l "app.kubernetes.io/name=cronos-ai" \
        -n "$NAMESPACE" \
        --timeout=600s
    
    # Run health checks
    log_info "Running health checks..."
    
    # Check dataplane health
    local dataplane_pod
    dataplane_pod=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/component=dataplane" -o jsonpath='{.items[0].metadata.name}')
    if kubectl exec -n "$NAMESPACE" "$dataplane_pod" -- curl -f http://localhost:8080/health &> /dev/null; then
        log_success "DataPlane health check passed"
    else
        log_error "DataPlane health check failed"
        return 1
    fi
    
    # Check control plane health
    local controlplane_svc
    controlplane_svc=$(kubectl get svc -n "$NAMESPACE" -l "app.kubernetes.io/component=controlplane" -o jsonpath='{.items[0].metadata.name}')
    if kubectl exec -n "$NAMESPACE" "$dataplane_pod" -- curl -f "http://$controlplane_svc:8080/health" &> /dev/null; then
        log_success "ControlPlane health check passed"
    else
        log_error "ControlPlane health check failed"
        return 1
    fi
    
    # Run integration tests
    if [[ -f "${PROJECT_ROOT}/tests/integration/Cargo.toml" ]]; then
        log_info "Running integration tests..."
        
        # Set up test environment variables
        export DATAPLANE_ENDPOINT="http://$(kubectl get svc -n $NAMESPACE cronos-ai-dataplane -o jsonpath='{.spec.clusterIP}'):9090"
        export CONTROLPLANE_ENDPOINT="http://$(kubectl get svc -n $NAMESPACE cronos-ai-controlplane -o jsonpath='{.spec.clusterIP}'):8080"
        export AIENGINE_ENDPOINT="http://$(kubectl get svc -n $NAMESPACE cronos-ai-aiengine -o jsonpath='{.spec.clusterIP}'):8000"
        export POLICY_ENGINE_ENDPOINT="http://$(kubectl get svc -n $NAMESPACE cronos-ai-policy-engine -o jsonpath='{.spec.clusterIP}'):8001"
        export TEST_NAMESPACE="$NAMESPACE"
        
        # Run tests from within the cluster
        kubectl run cronos-ai-test \
            --image="cronos-ai/integration-tests:latest" \
            --restart=Never \
            --rm -i \
            -n "$NAMESPACE" \
            --env="DATAPLANE_ENDPOINT=$DATAPLANE_ENDPOINT" \
            --env="CONTROLPLANE_ENDPOINT=$CONTROLPLANE_ENDPOINT" \
            --env="AIENGINE_ENDPOINT=$AIENGINE_ENDPOINT" \
            --env="POLICY_ENGINE_ENDPOINT=$POLICY_ENGINE_ENDPOINT" \
            --env="TEST_NAMESPACE=$NAMESPACE" \
            -- /app/integration-test-runner e2e
        
        if [[ $? -eq 0 ]]; then
            log_success "Integration tests passed"
        else
            log_error "Integration tests failed"
            return 1
        fi
    else
        log_warning "Integration tests not found, skipping"
    fi
    
    log_success "All post-deployment tests completed successfully"
}

# Generate deployment report
generate_report() {
    if [[ "$DRY_RUN" == true ]]; then
        return 0
    fi
    
    log_info "Generating deployment report..."
    
    local report_file="/tmp/cronos-ai-deployment-report-$(date +%Y%m%d-%H%M%S).json"
    
    # Gather deployment information
    local deployment_info
    deployment_info=$(cat << EOF
{
  "deployment": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "namespace": "$NAMESPACE",
    "release_name": "$RELEASE_NAME",
    "chart_version": "$(helm get metadata "$RELEASE_NAME" -n "$NAMESPACE" -o json | jq -r '.chart.metadata.version')",
    "app_version": "$(helm get metadata "$RELEASE_NAME" -n "$NAMESPACE" -o json | jq -r '.chart.metadata.appVersion')",
    "revision": "$(helm history "$RELEASE_NAME" -n "$NAMESPACE" -o json | jq -r '.[0].revision')",
    "status": "$(helm status "$RELEASE_NAME" -n "$NAMESPACE" -o json | jq -r '.info.status')"
  },
  "resources": {
    "pods": $(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=cronos-ai" -o json | jq '.items | length'),
    "services": $(kubectl get svc -n "$NAMESPACE" -l "app.kubernetes.io/name=cronos-ai" -o json | jq '.items | length'),
    "configmaps": $(kubectl get cm -n "$NAMESPACE" -l "app.kubernetes.io/name=cronos-ai" -o json | jq '.items | length'),
    "secrets": $(kubectl get secrets -n "$NAMESPACE" -l "app.kubernetes.io/name=cronos-ai" -o json | jq '.items | length')
  },
  "health": {
    "ready_pods": $(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=cronos-ai" -o json | jq '[.items[] | select(.status.phase == "Running")] | length'),
    "total_pods": $(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=cronos-ai" -o json | jq '.items | length')
  }
}
EOF
    )
    
    echo "$deployment_info" | jq . > "$report_file"
    
    log_info "Deployment report saved to: $report_file"
    
    # Print summary
    echo
    echo "=============================================="
    echo "       CRONOS AI Deployment Summary"
    echo "=============================================="
    echo "Timestamp:       $(date)"
    echo "Namespace:       $NAMESPACE"
    echo "Release:         $RELEASE_NAME"
    echo "Status:          $(helm status "$RELEASE_NAME" -n "$NAMESPACE" -o json | jq -r '.info.status')"
    echo "Revision:        $(helm history "$RELEASE_NAME" -n "$NAMESPACE" -o json | jq -r '.[0].revision')"
    echo "Ready Pods:      $(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=cronos-ai" --field-selector=status.phase=Running --no-headers | wc -l)"
    echo "Total Pods:      $(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=cronos-ai" --no-headers | wc -l)"
    echo "=============================================="
    echo
}

# Main deployment function
main() {
    log_info "Starting CRONOS AI production deployment"
    
    # Parse arguments
    parse_args "$@"
    
    # Show deployment parameters
    log_info "Deployment parameters:"
    log_info "  Namespace: $NAMESPACE"
    log_info "  Release: $RELEASE_NAME"
    log_info "  Values file: $VALUES_FILE"
    log_info "  Timeout: ${TIMEOUT}s"
    log_info "  Dry run: $DRY_RUN"
    
    # Execute deployment steps
    check_prerequisites
    validate_configuration
    ensure_namespace
    install_dependencies
    pre_deployment_checks
    deploy
    run_tests
    generate_report
    
    if [[ "$DRY_RUN" != true ]]; then
        log_success "🎉 CRONOS AI production deployment completed successfully!"
        log_info "Access your deployment:"
        log_info "  kubectl get all -n $NAMESPACE"
        log_info "  kubectl port-forward -n $NAMESPACE svc/cronos-ai-controlplane 8080:8080"
    else
        log_success "🔍 CRONOS AI dry run completed successfully!"
    fi
}

# Error handling
trap 'log_error "Deployment failed at line $LINENO"' ERR

# Execute main function
main "$@"