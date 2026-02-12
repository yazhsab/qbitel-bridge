# QBITEL Bridge - Deployment Guide

This guide provides comprehensive instructions for deploying QBITEL Bridge in development, staging, and production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Deployment Options](#deployment-options)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Production Configuration](#production-configuration)
- [Monitoring & Observability](#monitoring--observability)
- [Security Configuration](#security-configuration)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 20GB
- OS: Linux (Ubuntu 20.04+), macOS, Windows with WSL2

**Recommended for Production:**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 50GB+ SSD
- OS: Linux (Ubuntu 20.04+ or RHEL 8+)
- GPU: Optional (NVIDIA CUDA for AI training)

### Software Prerequisites

```bash
# Required
- Python 3.10+
- Rust 1.70+ (with cargo)
- Go 1.21+
- Node.js 18+ (with npm)
- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+ (for production)
- kubectl 1.24+
- Helm 3.8+ (for Helm-based deployment)
- git

# Optional
- NVIDIA Docker (for GPU support)
- Ollama (for on-premise LLM inference)
```

### Network Requirements

- **Outbound Internet**: For pulling Docker images, Python packages
- **Ports**:
  - 8000: REST API
  - 50051: gRPC API
  - 9090: Prometheus metrics
  - 3000: Grafana dashboards (development)
  - 5000: MLflow (development)

## Deployment Options

QBITEL Bridge supports multiple deployment models:

1. **Local Development** - Build from source (Python + Rust + Go)
2. **Docker Compose** - Multi-container development
3. **Kubernetes (Helm)** - Production cloud-native deployment
4. **Air-Gapped** - Fully offline with on-premise LLM

## Local Development

### Quick Start (All Components)

```bash
# 1. Clone the repository
git clone https://github.com/yazhsab/qbitel-bridge.git
cd qbitel-bridge

# 2. Build everything
make build

# 3. Run all tests
make test
```

### Python AI Engine

```bash
# Create Python virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r ai_engine/requirements.txt

# Run tests
pytest ai_engine/tests/ -v --tb=short

# Start the AI Engine
python -m ai_engine

# Verify the API is running
curl http://localhost:8000/health
```

### Rust Data Plane

```bash
cd rust/dataplane
cargo build --locked
cargo test
```

### Go Control Plane

```bash
# Build all Go services
cd go/controlplane && go build -trimpath -o ../../dist/controlplane ./cmd/controlplane
cd go/mgmtapi && go build -trimpath -o ../../dist/mgmtapi ./cmd/mgmtapi

# Run control plane
./dist/controlplane
```

### UI Console

```bash
cd ui/console
npm install
npm run dev
# Available at http://localhost:3000
```

### Configuration

Create a local configuration file:

```bash
# config/local.yaml
rest_host: "127.0.0.1"
rest_port: 8000
grpc_port: 50051
log_level: "DEBUG"
enable_observability: true
device: "cpu"  # or "cuda" if GPU available
```

Run with custom configuration:

```bash
python -m ai_engine --config config/local.yaml
```

### Development Mode

Enable hot-reloading for development:

```bash
python -m ai_engine --development --reload
```

## Docker Deployment

### Build Docker Images

```bash
# Build xDS Server
docker build -f docker/xds-server/Dockerfile -t qbitel/xds-server:latest .

# Build Admission Webhook
docker build -f docker/admission-webhook/Dockerfile -t qbitel/admission-webhook:latest .

# Build Kafka Producer
docker build -f docker/kafka-producer/Dockerfile -t qbitel/kafka-producer:latest .
```

### Docker Compose Deployment

The Docker Compose setup includes all necessary services:

```bash
cd docker

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f qbitel-engine

# Check service status
docker-compose ps

# Stop all services
docker-compose down
```

### Docker Compose Services

```yaml
services:
  - qbitel-engine: AI Engine (REST + gRPC)
  - mlflow: Model registry and tracking
  - prometheus: Metrics collection
  - grafana: Dashboards and visualization
  - kafka: Event streaming (optional)
  - postgres: Database for MLflow
```

### Access Services

- **AI Engine API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Grafana**: http://localhost:3000 (admin/qbitel-admin)
- **Prometheus**: http://localhost:9090

## Helm Deployment (Recommended for Production)

The recommended way to deploy QBITEL Bridge to Kubernetes is via Helm chart.

### Quick Deploy

```bash
helm install qbitel-bridge ./helm/qbitel-bridge \
  --namespace qbitel-bridge \
  --create-namespace \
  --wait
```

### Production Deploy

```bash
helm install qbitel-bridge ./helm/qbitel-bridge \
  --namespace qbitel-bridge \
  --create-namespace \
  --set xdsServer.replicaCount=5 \
  --set admissionWebhook.replicaCount=5 \
  --set xdsServer.resources.requests.memory=1Gi \
  --set monitoring.enabled=true
```

### Custom Values

```bash
# See all configurable values
helm show values ./helm/qbitel-bridge

# Deploy with custom values file
helm install qbitel-bridge ./helm/qbitel-bridge \
  --namespace qbitel-bridge \
  --create-namespace \
  -f my-values.yaml
```

See [helm/qbitel-bridge/README.md](helm/qbitel-bridge/README.md) for full Helm chart documentation.

## Air-Gapped Deployment

For environments with no internet connectivity:

```bash
# 1. On a connected machine: pull models and images
ollama pull llama3.2:8b
ollama pull llama3.2:70b
docker save qbitel/controlplane:latest > controlplane.tar
docker save qbitel/mgmtapi:latest > mgmtapi.tar

# 2. Transfer to air-gapped environment (USB, secure transfer)

# 3. On the air-gapped machine: load images
docker load < controlplane.tar
docker load < mgmtapi.tar

# 4. Configure for air-gapped mode
export QBITEL_LLM_PROVIDER=ollama
export QBITEL_LLM_ENDPOINT=http://localhost:11434
export QBITEL_AIRGAPPED_MODE=true
export QBITEL_DISABLE_CLOUD_LLMS=true

# 5. Deploy
python -m ai_engine --airgapped
```

Air-gapped features:
- All LLM inference runs locally (Ollama)
- No internet connectivity required after initial setup
- All data stays within your infrastructure
- No API keys or cloud accounts needed
- Compliant with strictest data residency regulations

## Kubernetes Deployment (Manual)

For fine-grained control over individual components:

### Prerequisites

```bash
# Verify Kubernetes cluster access
kubectl cluster-info
kubectl get nodes

# Create namespaces
kubectl create namespace qbitel-service-mesh
kubectl create namespace qbitel-container-security
kubectl create namespace qbitel-monitoring
```

### Service Mesh Deployment

Deploy xDS Server and Istio integration:

```bash
# 1. Apply namespace and RBAC
kubectl apply -f kubernetes/service-mesh/namespace.yaml
kubectl apply -f kubernetes/service-mesh/rbac.yaml

# 2. Deploy xDS Server
kubectl apply -f kubernetes/service-mesh/xds-server-deployment.yaml

# 3. Verify deployment
kubectl get pods -n qbitel-service-mesh
kubectl logs -n qbitel-service-mesh deployment/xds-server -f

# 4. Check service
kubectl get svc -n qbitel-service-mesh
```

### Container Security Deployment

Deploy admission webhook and security components:

```bash
# 1. Generate TLS certificates for webhook
./scripts/generate-webhook-certs.sh

# 2. Deploy admission webhook
kubectl apply -f kubernetes/container-security/admission-webhook-deployment.yaml

# 3. Verify webhook is running
kubectl get pods -n qbitel-container-security
kubectl get validatingwebhookconfigurations

# 4. Test webhook (try deploying a test pod)
kubectl run test-pod --image=nginx -n default
kubectl describe pod test-pod -n default
```

### Monitoring Stack Deployment

```bash
# Deploy Prometheus
kubectl apply -f kubernetes/monitoring/prometheus-deployment.yaml

# Deploy Grafana
kubectl apply -f kubernetes/monitoring/grafana-deployment.yaml

# Access Grafana
kubectl port-forward -n qbitel-monitoring svc/grafana 3000:3000

# Import dashboards from kubernetes/monitoring/dashboards/
```

### Verify Deployment

```bash
# Check all pods
kubectl get pods --all-namespaces | grep qbitel

# Check services
kubectl get svc --all-namespaces | grep qbitel

# Check resource usage
kubectl top pods -n qbitel-service-mesh
kubectl top pods -n qbitel-container-security

# View logs
kubectl logs -n qbitel-service-mesh deployment/xds-server --tail=100
kubectl logs -n qbitel-container-security deployment/admission-webhook --tail=100
```

## Production Configuration

### Environment Variables

```bash
# AI Engine Configuration
export AI_ENGINE_LOG_LEVEL=INFO
export AI_ENGINE_WORKERS=4
export AI_ENGINE_MAX_REQUESTS=1000
export AI_ENGINE_TIMEOUT=300

# Security
export API_KEY_SECRET=<secure-random-string>
export TLS_CERT_PATH=/etc/qbitel/certs/tls.crt
export TLS_KEY_PATH=/etc/qbitel/certs/tls.key

# MLflow
export MLFLOW_TRACKING_URI=https://mlflow.production.example.com
export MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com

# Monitoring
export PROMETHEUS_METRICS_PORT=9090
export ENABLE_TRACING=true
export JAEGER_AGENT_HOST=jaeger-agent.monitoring.svc.cluster.local
```

### Resource Limits

Configure resource requests and limits in Kubernetes:

```yaml
# kubernetes/production/xds-server-deployment.yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

### Auto-Scaling

Enable Horizontal Pod Autoscaler:

```bash
# HPA for xDS Server
kubectl autoscale deployment xds-server \
  -n qbitel-service-mesh \
  --cpu-percent=70 \
  --min=3 \
  --max=10

# HPA for Admission Webhook
kubectl autoscale deployment admission-webhook \
  -n qbitel-container-security \
  --cpu-percent=70 \
  --min=3 \
  --max=10

# Verify HPA
kubectl get hpa --all-namespaces
```

### Persistent Storage

Configure persistent volumes for data and models:

```yaml
# kubernetes/production/persistent-volumes.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: qbitel-models-pvc
  namespace: qbitel-service-mesh
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
```

### High Availability

Ensure HA configuration:

```yaml
# Pod Disruption Budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: xds-server-pdb
  namespace: qbitel-service-mesh
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: xds-server
```

## Monitoring & Observability

### Prometheus Metrics

Key metrics exposed by QBITEL:

```bash
# Check metrics endpoint
curl http://localhost:9090/metrics

# Key metrics:
# - qbitel_requests_total
# - qbitel_request_duration_seconds
# - qbitel_protocol_discovery_accuracy
# - qbitel_anomaly_detection_alerts
# - qbitel_encryption_operations_total
```

### Grafana Dashboards

Import pre-built dashboards:

```bash
# Located in: kubernetes/monitoring/dashboards/
# - qbitel-overview.json
# - qbitel-security.json
# - qbitel-performance.json
# - qbitel-kubernetes.json
```

### Logging

Configure structured logging:

```yaml
# config/production.yaml
logging:
  level: INFO
  format: json
  output: stdout
  include_trace_id: true
  include_span_id: true
```

### Distributed Tracing

Enable OpenTelemetry tracing:

```bash
# Set Jaeger endpoint
export OTEL_EXPORTER_JAEGER_ENDPOINT=http://jaeger-collector:14268/api/traces

# View traces in Jaeger UI
kubectl port-forward -n monitoring svc/jaeger-query 16686:16686
# Open http://localhost:16686
```

### Health Checks

```bash
# Liveness probe
curl http://localhost:8000/health

# Readiness probe
curl http://localhost:8000/ready

# Detailed status
curl http://localhost:8000/status
```

## Security Configuration

### TLS/SSL Configuration

Generate certificates:

```bash
# Generate self-signed certificates (development)
openssl req -x509 -newkey rsa:4096 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes \
  -subj "/CN=qbitel.local"

# For production, use cert-manager or your PKI
```

Configure TLS in deployment:

```yaml
# kubernetes/production/tls-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: qbitel-tls
  namespace: qbitel-service-mesh
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-cert>
  tls.key: <base64-encoded-key>
```

### RBAC Configuration

Kubernetes RBAC is defined in:
- `kubernetes/service-mesh/rbac.yaml`
- `kubernetes/container-security/rbac.yaml`

### Network Policies

Restrict pod-to-pod communication:

```yaml
# kubernetes/production/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: qbitel-network-policy
  namespace: qbitel-service-mesh
spec:
  podSelector:
    matchLabels:
      app: xds-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
    ports:
    - protocol: TCP
      port: 50051
```

### Secret Management

Use Kubernetes secrets or external secret managers:

```bash
# Create secret for API keys
kubectl create secret generic qbitel-api-keys \
  -n qbitel-service-mesh \
  --from-literal=api-key=<your-secure-api-key>

# Use AWS Secrets Manager (example)
kubectl create secret generic aws-secrets \
  -n qbitel-service-mesh \
  --from-literal=aws-access-key-id=<id> \
  --from-literal=aws-secret-access-key=<key>
```

## Troubleshooting

### Common Issues

#### 1. Pod Fails to Start

```bash
# Check pod status
kubectl describe pod <pod-name> -n qbitel-service-mesh

# Check logs
kubectl logs <pod-name> -n qbitel-service-mesh

# Common causes:
# - Image pull errors: Check imagePullSecrets
# - Resource constraints: Check resource limits
# - Configuration errors: Verify ConfigMaps
```

#### 2. High Memory Usage

```bash
# Check resource usage
kubectl top pod <pod-name> -n qbitel-service-mesh

# Reduce batch size in configuration
# Edit ConfigMap:
kubectl edit configmap qbitel-config -n qbitel-service-mesh

# Set: batch_size: 16 (reduce from 32)
```

#### 3. xDS Server Not Connecting to Envoy

```bash
# Check xDS server logs
kubectl logs deployment/xds-server -n qbitel-service-mesh | grep -i error

# Verify gRPC port is accessible
kubectl get svc xds-server -n qbitel-service-mesh

# Test connection
grpcurl -plaintext xds-server.qbitel-service-mesh:50051 list
```

#### 4. Admission Webhook Blocking Pods

```bash
# Check webhook logs
kubectl logs deployment/admission-webhook -n qbitel-container-security

# Temporarily disable webhook for debugging
kubectl delete validatingwebhookconfigurations qbitel-admission-webhook

# Re-enable after fixing
kubectl apply -f kubernetes/container-security/admission-webhook-deployment.yaml
```

#### 5. eBPF Monitor Not Working

```bash
# eBPF requires Linux kernel 4.4+
uname -r

# Install BCC tools (Ubuntu/Debian)
apt-get install -y bpfcc-tools linux-headers-$(uname -r)

# Check if BPF is available
ls /sys/kernel/debug/tracing/

# Run with elevated privileges
kubectl patch daemonset ebpf-monitor -n qbitel-container-security \
  -p '{"spec":{"template":{"spec":{"hostPID":true,"hostNetwork":true}}}}'
```

### Performance Tuning

```bash
# 1. Enable connection pooling
export MAX_CONNECTIONS=100
export CONNECTION_TIMEOUT=30

# 2. Adjust worker processes
export AI_ENGINE_WORKERS=4

# 3. Enable caching
export ENABLE_CACHE=true
export CACHE_TTL=300

# 4. Tune Kafka producer
export KAFKA_BATCH_SIZE=16384
export KAFKA_LINGER_MS=10
export KAFKA_COMPRESSION_TYPE=snappy
```

### Debugging Tips

```bash
# Enable debug logging
kubectl set env deployment/xds-server -n qbitel-service-mesh LOG_LEVEL=DEBUG

# Port-forward for local access
kubectl port-forward -n qbitel-service-mesh svc/xds-server 50051:50051

# Execute commands in pod
kubectl exec -it <pod-name> -n qbitel-service-mesh -- /bin/bash

# Copy files from pod
kubectl cp qbitel-service-mesh/<pod-name>:/app/logs ./logs

# Watch pod events
kubectl get events -n qbitel-service-mesh --watch
```

## Backup and Recovery

### Backup Strategy

```bash
# 1. Backup Kubernetes resources
kubectl get all -n qbitel-service-mesh -o yaml > backup-service-mesh.yaml
kubectl get all -n qbitel-container-security -o yaml > backup-container-security.yaml

# 2. Backup ConfigMaps and Secrets
kubectl get configmaps -n qbitel-service-mesh -o yaml > backup-configmaps.yaml
kubectl get secrets -n qbitel-service-mesh -o yaml > backup-secrets.yaml

# 3. Backup persistent volumes
kubectl get pvc -n qbitel-service-mesh -o yaml > backup-pvc.yaml

# 4. Backup MLflow models (if using S3/object storage)
# Configure periodic snapshots via cloud provider
```

### Recovery Procedures

```bash
# 1. Restore Kubernetes resources
kubectl apply -f backup-service-mesh.yaml
kubectl apply -f backup-container-security.yaml

# 2. Restore ConfigMaps and Secrets
kubectl apply -f backup-configmaps.yaml
kubectl apply -f backup-secrets.yaml

# 3. Verify restoration
kubectl get pods --all-namespaces | grep qbitel
```

## Upgrade Procedures

```bash
# 1. Backup current deployment
kubectl get all -n qbitel-service-mesh -o yaml > pre-upgrade-backup.yaml

# 2. Update Docker images
kubectl set image deployment/xds-server \
  -n qbitel-service-mesh \
  xds-server=qbitel/xds-server:v2.0.0

# 3. Monitor rollout
kubectl rollout status deployment/xds-server -n qbitel-service-mesh

# 4. Rollback if needed
kubectl rollout undo deployment/xds-server -n qbitel-service-mesh

# 5. Verify functionality
kubectl exec -it <pod-name> -n qbitel-service-mesh -- python -c "import ai_engine; print(ai_engine.__version__)"
```

## Support

For deployment issues:
- Check logs: `kubectl logs <pod-name>`
- Review events: `kubectl get events`
- GitHub Issues: https://github.com/yazhsab/qbitel-bridge/issues
- Enterprise Support: enterprise@qbitel.com

---

**Last Updated**: 2025-02-08
**Version**: 2.0
**Status**: Production Deployment Guide
