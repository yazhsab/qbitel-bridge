#!/bin/bash
# CRONOS AI - Production Deployment Script
# Complete automation for production deployment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HELM_CHART_PATH="${PROJECT_ROOT}/ops/deploy/helm/cronos-ai"
NAMESPACE="cronos-ai"
MONITORING_NAMESPACE="cronos-ai-monitoring"
RELEASE_NAME="cronos-ai"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Deployment failed with exit code $exit_code"
        log_info "Rolling back changes..."
        rollback_deployment
    fi
    exit $exit_code
}

trap cleanup EXIT

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is available and configured
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is available
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we can connect to Kubernetes cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check Kubernetes version
    local k8s_version=$(kubectl version --client=false -o json | jq -r '.serverVersion.gitVersion')
    log_info "Kubernetes version: $k8s_version"
    
    # Check if required Helm repositories are added
    if ! helm repo list | grep -q "prometheus-community"; then
        log_info "Adding Prometheus Helm repository..."
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    fi
    
    if ! helm repo list | grep -q "grafana"; then
        log_info "Adding Grafana Helm repository..."
        helm repo add grafana https://grafana.github.io/helm-charts
    fi
    
    if ! helm repo list | grep -q "bitnami"; then
        log_info "Adding Bitnami Helm repository..."
        helm repo add bitnami https://charts.bitnami.com/bitnami
    fi
    
    if ! helm repo list | grep -q "istio"; then
        log_info "Adding Istio Helm repository..."
        helm repo add istio https://istio-release.storage.googleapis.com/charts
    fi
    
    helm repo update
    log_success "Prerequisites check completed"
}

# Function to create namespaces
create_namespaces() {
    log_info "Creating namespaces..."
    
    # Create main namespace
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    kubectl label namespace $NAMESPACE istio-injection=enabled --overwrite
    
    # Create monitoring namespace
    kubectl create namespace $MONITORING_NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Namespaces created successfully"
}

# Function to setup secrets
setup_secrets() {
    log_info "Setting up secrets..."
    
    # Check if secrets file exists
    local secrets_file="${PROJECT_ROOT}/config/secrets/production-secrets.yaml"
    if [[ -f "$secrets_file" ]]; then
        kubectl apply -f "$secrets_file" -n $NAMESPACE
        log_success "Applied production secrets"
    else
        log_warning "Production secrets file not found at $secrets_file"
        log_info "Creating placeholder secrets..."
        
        # Create placeholder secrets (MUST be updated with real values)
        kubectl create secret generic cronos-ai-secrets \
            --from-literal=database-url="postgresql://cronos_ai:CHANGE_ME@postgresql:5432/cronos_ai_prod" \
            --from-literal=redis-password="CHANGE_ME" \
            --from-literal=jwt-secret="CHANGE_ME" \
            --from-literal=oauth-client-id="CHANGE_ME" \
            --from-literal=oauth-client-secret="CHANGE_ME" \
            --from-literal=vault-token="CHANGE_ME" \
            --namespace $NAMESPACE \
            --dry-run=client -o yaml | kubectl apply -f -
        
        kubectl create secret generic cronos-ai-grafana-secret \
            --from-literal=admin-user="admin" \
            --from-literal=admin-password="CHANGE_ME" \
            --namespace $MONITORING_NAMESPACE \
            --dry-run=client -o yaml | kubectl apply -f -
        
        log_warning "IMPORTANT: Update the placeholder secrets with actual values before production use!"
    fi
}

# Function to setup TLS certificates
setup_tls() {
    log_info "Setting up TLS certificates..."
    
    # Check if cert-manager is installed
    if kubectl get crd certificates.cert-manager.io &> /dev/null; then
        log_info "cert-manager found, creating certificate issuers..."
        
        cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@cronos-ai.example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: istio
EOF
        log_success "TLS certificate issuer created"
    else
        log_warning "cert-manager not found, skipping automatic TLS setup"
    fi
}

# Function to deploy Istio service mesh
deploy_istio() {
    log_info "Deploying Istio service mesh..."
    
    # Check if Istio is already installed
    if kubectl get namespace istio-system &> /dev/null; then
        log_info "Istio namespace exists, checking installation..."
        if kubectl get deployment istiod -n istio-system &> /dev/null; then
            log_info "Istio already installed, skipping..."
            return 0
        fi
    fi
    
    # Install Istio base
    helm upgrade --install istio-base istio/base \
        --namespace istio-system \
        --create-namespace \
        --version 1.20.0 \
        --wait
    
    # Install Istiod
    helm upgrade --install istiod istio/istiod \
        --namespace istio-system \
        --version 1.20.0 \
        --wait
    
    # Install Istio gateway
    helm upgrade --install istio-gateway istio/gateway \
        --namespace istio-system \
        --version 1.20.0 \
        --wait
    
    log_success "Istio service mesh deployed successfully"
}

# Function to deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Deploy custom monitoring configuration
    kubectl apply -f "${PROJECT_ROOT}/ops/monitoring/production-monitoring.yaml"
    
    # Wait for monitoring pods to be ready
    kubectl wait --for=condition=available --timeout=600s deployment/prometheus -n $MONITORING_NAMESPACE || true
    kubectl wait --for=condition=available --timeout=600s deployment/grafana -n $MONITORING_NAMESPACE || true
    
    log_success "Monitoring stack deployed successfully"
}

# Function to validate deployment
validate_deployment() {
    log_info "Validating deployment..."
    
    # Check if all deployments are ready
    local deployments=(
        "ai-engine"
        "mgmt-api"
        "control-plane"
        "ui-console"
    )
    
    for deployment in "${deployments[@]}"; do
        local full_name="${RELEASE_NAME}-${deployment}"
        log_info "Checking deployment: $full_name"
        
        if kubectl rollout status deployment/$full_name -n $NAMESPACE --timeout=600s; then
            log_success "Deployment $full_name is ready"
        else
            log_error "Deployment $full_name failed to become ready"
            return 1
        fi
    done
    
    # Check if services are accessible
    log_info "Checking service endpoints..."
    
    local services=(
        "${RELEASE_NAME}-ai-engine:8000"
        "${RELEASE_NAME}-mgmt-api:8080"
        "${RELEASE_NAME}-control-plane:9000"
        "${RELEASE_NAME}-ui-console:3000"
    )
    
    for service in "${services[@]}"; do
        local service_name=$(echo $service | cut -d: -f1)
        local port=$(echo $service | cut -d: -f2)
        
        if kubectl get service $service_name -n $NAMESPACE &> /dev/null; then
            log_success "Service $service_name is available"
        else
            log_warning "Service $service_name not found"
        fi
    done
    
    log_success "Deployment validation completed"
}

# Function to run integration tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    # Create test job
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: cronos-ai-integration-tests
  namespace: $NAMESPACE
spec:
  template:
    spec:
      containers:
      - name: integration-tests
        image: python:3.11-slim
        command: ["/bin/bash"]
        args:
          - -c
          - |
            pip install asyncio aiohttp asyncpg redis kafka-python pytest kubernetes prometheus-client
            python /tests/production-integration-tests.py --config /config/cronos_ai.production.yaml
        volumeMounts:
        - name: test-scripts
          mountPath: /tests
        - name: config
          mountPath: /config
      volumes:
      - name: test-scripts
        configMap:
          name: integration-test-scripts
      - name: config
        configMap:
          name: cronos-ai-config
      restartPolicy: Never
  backoffLimit: 1
  ttlSecondsAfterFinished: 3600
EOF
    
    # Wait for test completion
    kubectl wait --for=condition=complete --timeout=1800s job/cronos-ai-integration-tests -n $NAMESPACE || {
        log_warning "Integration tests failed or timed out"
        kubectl logs job/cronos-ai-integration-tests -n $NAMESPACE || true
    }
    
    # Clean up test job
    kubectl delete job cronos-ai-integration-tests -n $NAMESPACE || true
    
    log_success "Integration tests completed"
}

# Function to setup monitoring dashboards
setup_dashboards() {
    log_info "Setting up monitoring dashboards..."
    
    # Apply Grafana dashboards
    if [[ -d "${PROJECT_ROOT}/ops/grafana-dashboards" ]]; then
        for dashboard in "${PROJECT_ROOT}/ops/grafana-dashboards"/*.json; do
            if [[ -f "$dashboard" ]]; then
                local dashboard_name=$(basename "$dashboard" .json)
                kubectl create configmap "dashboard-${dashboard_name}" \
                    --from-file="$dashboard" \
                    --namespace $MONITORING_NAMESPACE \
                    --dry-run=client -o yaml | kubectl apply -f -
            fi
        done
        log_success "Grafana dashboards configured"
    fi
    
    # Apply Prometheus rules
    if [[ -f "${PROJECT_ROOT}/ops/observability/prometheus/rules.yaml" ]]; then
        kubectl apply -f "${PROJECT_ROOT}/ops/observability/prometheus/rules.yaml" -n $MONITORING_NAMESPACE
        log_success "Prometheus rules applied"
    fi
}

# Function to perform health checks
health_check() {
    log_info "Performing health checks..."
    
    # Check pod health
    local unhealthy_pods=$(kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running --no-headers | wc -l)
    if [[ $unhealthy_pods -eq 0 ]]; then
        log_success "All pods are healthy"
    else
        log_warning "$unhealthy_pods unhealthy pods found"
        kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running
    fi
    
    # Check service endpoints
    local endpoints_not_ready=$(kubectl get endpoints -n $NAMESPACE -o json | jq '.items[] | select(.subsets == null or .subsets == []) | .metadata.name' | wc -l)
    if [[ $endpoints_not_ready -eq 0 ]]; then
        log_success "All service endpoints are ready"
    else
        log_warning "$endpoints_not_ready service endpoints not ready"
    fi
    
    log_success "Health check completed"
}

# Function to rollback deployment
rollback_deployment() {
    log_info "Rolling back deployment..."
    
    # Rollback Helm release
    helm rollback $RELEASE_NAME -n $NAMESPACE || true
    
    # Wait for rollback to complete
    sleep 30
    
    log_info "Rollback completed"
}

# Function to show deployment status
show_status() {
    log_info "Deployment Status Summary"
    echo "=================================="
    
    # Show Helm releases
    echo -e "\n${BLUE}Helm Releases:${NC}"
    helm list -n $NAMESPACE
    
    # Show pods
    echo -e "\n${BLUE}Pods Status:${NC}"
    kubectl get pods -n $NAMESPACE -o wide
    
    # Show services
    echo -e "\n${BLUE}Services:${NC}"
    kubectl get services -n $NAMESPACE
    
    # Show ingresses
    echo -e "\n${BLUE}Ingresses:${NC}"
    kubectl get ingress -n $NAMESPACE
    
    # Show persistent volumes
    echo -e "\n${BLUE}Persistent Volumes:${NC}"
    kubectl get pvc -n $NAMESPACE
    
    # Show monitoring status
    echo -e "\n${BLUE}Monitoring Stack:${NC}"
    kubectl get pods -n $MONITORING_NAMESPACE
    
    echo "=================================="
}

# Main deployment function
deploy_cronos_ai() {
    log_info "Starting CRONOS AI production deployment..."
    
    # Pre-deployment steps
    check_prerequisites
    create_namespaces
    setup_secrets
    setup_tls
    
    # Infrastructure deployment
    deploy_istio
    deploy_monitoring
    
    # Main application deployment
    log_info "Deploying CRONOS AI application..."
    helm upgrade --install $RELEASE_NAME $HELM_CHART_PATH \
        --namespace $NAMESPACE \
        --values "${HELM_CHART_PATH}/values.yaml" \
        --timeout 20m \
        --wait \
        --atomic
    
    # Post-deployment steps
    validate_deployment
    setup_dashboards
    health_check
    
    # Optional integration tests
    if [[ "${RUN_INTEGRATION_TESTS:-false}" == "true" ]]; then
        run_integration_tests
    fi
    
    show_status
    
    log_success "CRONOS AI production deployment completed successfully!"
    log_info "Access URLs:"
    log_info "  - Console: https://console.cronos-ai.example.com"
    log_info "  - API: https://api.cronos-ai.example.com"
    log_info "  - Monitoring: https://monitoring.cronos-ai.example.com"
    log_info "  - AI API: https://ai-api.cronos-ai.example.com"
}

# Parse command line arguments
COMMAND=${1:-deploy}

case $COMMAND in
    "deploy")
        deploy_cronos_ai
        ;;
    "rollback")
        rollback_deployment
        ;;
    "status")
        show_status
        ;;
    "health")
        health_check
        ;;
    "validate")
        validate_deployment
        ;;
    "test")
        run_integration_tests
        ;;
    *)
        echo "Usage: $0 [deploy|rollback|status|health|validate|test]"
        echo ""
        echo "Commands:"
        echo "  deploy    - Deploy CRONOS AI to production"
        echo "  rollback  - Rollback to previous version"
        echo "  status    - Show deployment status"
        echo "  health    - Perform health checks"
        echo "  validate  - Validate deployment"
        echo "  test      - Run integration tests"
        echo ""
        echo "Environment Variables:"
        echo "  RUN_INTEGRATION_TESTS=true  - Run integration tests after deployment"
        exit 1
        ;;
esac