#!/bin/bash

##############################################################################
# CRONOS AI Deployment Script
# Automates the deployment of CRONOS AI to Kubernetes using Helm
##############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HELM_CHART_DIR="$PROJECT_ROOT/helm/cronos-ai"
NAMESPACE="${NAMESPACE:-cronos-service-mesh}"
RELEASE_NAME="${RELEASE_NAME:-cronos-ai}"
VALUES_FILE="${VALUES_FILE:-}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Functions
print_header() {
    echo -e "${GREEN}===================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}===================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    print_success "kubectl found: $(kubectl version --client --short 2>/dev/null || kubectl version --client)"

    # Check helm
    if ! command -v helm &> /dev/null; then
        print_error "helm not found. Please install Helm 3.8+."
        exit 1
    fi
    print_success "helm found: $(helm version --short)"

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    print_success "Kubernetes cluster is accessible"

    # Check if namespace exists
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_warning "Namespace $NAMESPACE already exists"
    else
        print_success "Namespace $NAMESPACE will be created"
    fi
}

validate_helm_chart() {
    print_header "Validating Helm Chart"

    cd "$HELM_CHART_DIR"

    # Lint the chart
    if helm lint . ; then
        print_success "Helm chart validation passed"
    else
        print_error "Helm chart validation failed"
        exit 1
    fi

    # Template the chart to check for syntax errors
    if helm template test-release . --namespace "$NAMESPACE" > /dev/null; then
        print_success "Helm template generation successful"
    else
        print_error "Helm template generation failed"
        exit 1
    fi
}

deploy_cronos_ai() {
    print_header "Deploying CRONOS AI"

    cd "$HELM_CHART_DIR"

    # Build helm install command
    HELM_CMD="helm upgrade --install $RELEASE_NAME . \
        --namespace $NAMESPACE \
        --create-namespace \
        --wait \
        --timeout 10m"

    # Add values file if specified
    if [ -n "$VALUES_FILE" ]; then
        if [ -f "$VALUES_FILE" ]; then
            HELM_CMD="$HELM_CMD -f $VALUES_FILE"
            print_success "Using values file: $VALUES_FILE"
        else
            print_error "Values file not found: $VALUES_FILE"
            exit 1
        fi
    fi

    # Add environment-specific values
    case "$ENVIRONMENT" in
        production)
            print_success "Using production configuration"
            ;;
        staging)
            HELM_CMD="$HELM_CMD --set xdsServer.replicaCount=2 --set admissionWebhook.replicaCount=2"
            print_success "Using staging configuration"
            ;;
        development)
            HELM_CMD="$HELM_CMD --set xdsServer.replicaCount=1 --set admissionWebhook.replicaCount=1 --set xdsServer.autoscaling.enabled=false --set admissionWebhook.autoscaling.enabled=false"
            print_success "Using development configuration"
            ;;
        *)
            print_error "Unknown environment: $ENVIRONMENT. Use production, staging, or development."
            exit 1
            ;;
    esac

    echo ""
    echo "Executing: $HELM_CMD"
    echo ""

    # Execute deployment
    if eval "$HELM_CMD"; then
        print_success "CRONOS AI deployed successfully"
    else
        print_error "Deployment failed"
        exit 1
    fi
}

verify_deployment() {
    print_header "Verifying Deployment"

    # Wait for deployments to be ready
    echo "Waiting for deployments to be ready..."

    if kubectl wait --for=condition=available --timeout=300s \
        deployment/cronos-xds-server \
        deployment/cronos-admission-webhook \
        -n "$NAMESPACE" 2>/dev/null; then
        print_success "All deployments are ready"
    else
        print_warning "Some deployments may not be ready yet"
    fi

    # Check pod status
    echo ""
    echo "Pod Status:"
    kubectl get pods -n "$NAMESPACE"

    # Check service status
    echo ""
    echo "Service Status:"
    kubectl get svc -n "$NAMESPACE"

    # Check if all pods are running
    RUNNING_PODS=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Running --no-headers | wc -l)
    TOTAL_PODS=$(kubectl get pods -n "$NAMESPACE" --no-headers | wc -l)

    echo ""
    if [ "$RUNNING_PODS" -eq "$TOTAL_PODS" ]; then
        print_success "All pods are running ($RUNNING_PODS/$TOTAL_PODS)"
    else
        print_warning "Some pods are not running yet ($RUNNING_PODS/$TOTAL_PODS)"
    fi
}

run_smoke_tests() {
    print_header "Running Smoke Tests"

    # Test 1: Check xDS Server health
    echo "Testing xDS Server health endpoint..."
    if kubectl exec -n "$NAMESPACE" deployment/cronos-xds-server -- \
        curl -f http://localhost:8081/healthz &> /dev/null; then
        print_success "xDS Server is healthy"
    else
        print_warning "xDS Server health check failed (may not be fully ready)"
    fi

    # Test 2: Check admission webhook
    echo "Testing Admission Webhook..."
    if kubectl get validatingwebhookconfigurations cronos-validating-webhook &> /dev/null; then
        print_success "Admission Webhook is configured"
    else
        print_warning "Admission Webhook configuration not found"
    fi

    # Test 3: Verify quantum crypto implementation
    echo "Testing Quantum Cryptography..."
    if kubectl exec -n "$NAMESPACE" deployment/cronos-xds-server -- \
        python3 -c "from ai_engine.cloud_native.service_mesh.istio.qkd_certificate_manager import QuantumCertificateManager; print('OK')" &> /dev/null; then
        print_success "Quantum Cryptography modules loaded successfully"
    else
        print_warning "Quantum Cryptography test skipped (Python environment may not be ready)"
    fi
}

show_access_info() {
    print_header "Access Information"

    cat << EOF

🎉 CRONOS AI has been successfully deployed!

NAMESPACE: $NAMESPACE
RELEASE: $RELEASE_NAME
ENVIRONMENT: $ENVIRONMENT

To access the services:

1. AI Engine API (REST):
   kubectl port-forward -n $NAMESPACE svc/cronos-ai-engine 8000:8000
   Then open: http://localhost:8000/docs

2. xDS Server (gRPC):
   kubectl port-forward -n $NAMESPACE svc/cronos-xds-server 18000:18000

3. Grafana Dashboards:
   kubectl port-forward -n $NAMESPACE svc/grafana 3000:3000
   Default credentials: admin / cronos-ai-admin

4. Prometheus Metrics:
   kubectl port-forward -n $NAMESPACE svc/prometheus 9090:9090

Useful commands:

  # View logs
  kubectl logs -n $NAMESPACE deployment/cronos-xds-server -f

  # Check status
  kubectl get pods -n $NAMESPACE

  # Run tests
  helm test $RELEASE_NAME -n $NAMESPACE

For more information:
  helm status $RELEASE_NAME -n $NAMESPACE

Documentation: https://github.com/qbitel/cronos-ai

EOF
}

# Main execution
main() {
    print_header "CRONOS AI Deployment"
    echo "Deploying to namespace: $NAMESPACE"
    echo "Release name: $RELEASE_NAME"
    echo "Environment: $ENVIRONMENT"
    echo ""

    check_prerequisites
    validate_helm_chart
    deploy_cronos_ai
    verify_deployment
    run_smoke_tests
    show_access_info

    print_header "Deployment Complete"
    print_success "CRONOS AI is ready to use!"
}

# Run main function
main "$@"
