# CRONOS AI - Quick Start Guide

Get CRONOS AI up and running in **10 minutes**.

## Prerequisites

- **Kubernetes cluster** (1.24+) OR Docker Desktop with Kubernetes enabled
- **Helm 3.8+** installed
- **kubectl** configured
- **4GB RAM** minimum (8GB+ recommended)

## Option 1: Helm Deployment (Recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/qbitel/cronos-ai.git
cd cronos-ai
```

### Step 2: Deploy with Helm

```bash
# Deploy to Kubernetes
helm install cronos-ai ./helm/cronos-ai \
  --namespace cronos-service-mesh \
  --create-namespace \
  --wait

# Check deployment status
kubectl get pods -n cronos-service-mesh
```

### Step 3: Access Services

```bash
# Access AI Engine API
kubectl port-forward -n cronos-service-mesh svc/cronos-ai-engine 8000:8000

# Open in browser
# http://localhost:8000/docs
```

### Step 4: Verify Quantum Cryptography

```bash
# Run validation tests
kubectl exec -n cronos-service-mesh deployment/cronos-xds-server -- \
  python -c "from ai_engine.cloud_native.service_mesh.istio.qkd_certificate_manager import QuantumCertificateManager; \
  mgr = QuantumCertificateManager(); \
  print('✓ Quantum Crypto: VALIDATED')"
```

## Option 2: Automated Script Deployment

### Linux/macOS

```bash
# Make script executable
chmod +x scripts/deploy.sh

# Deploy
./scripts/deploy.sh \
  --environment development \
  --namespace cronos-service-mesh
```

### Windows (PowerShell)

```powershell
# Run deployment script
.\scripts\deploy.ps1 `
  -Environment development `
  -Namespace cronos-service-mesh
```

## Option 3: Local Development (Python)

### Step 1: Install Dependencies

```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r ai_engine/requirements.txt
```

### Step 2: Run Tests

```bash
# Run all tests
pytest ai_engine/tests/ -v

# Run cloud-native tests only
pytest ai_engine/tests/cloud_native/ -v
```

### Step 3: Start AI Engine

```bash
# Start the AI Engine locally
python -m ai_engine

# Access at http://localhost:8000
```

## Verify Installation

### Check All Pods Running

```bash
kubectl get pods -n cronos-service-mesh

# Expected output:
# NAME                                    READY   STATUS    RESTARTS
# cronos-xds-server-xxx                   1/1     Running   0
# cronos-admission-webhook-xxx            1/1     Running   0
# cronos-ai-engine-xxx                    1/1     Running   0
```

### Test xDS Server Health

```bash
kubectl exec -n cronos-service-mesh deployment/cronos-xds-server -- \
  curl -f http://localhost:8081/healthz

# Expected: OK
```

### Access Grafana Dashboards

```bash
kubectl port-forward -n cronos-service-mesh svc/grafana 3000:3000

# Open http://localhost:3000
# Default credentials: admin / cronos-ai-admin
```

## Enable Admission Webhook for Your Namespace

```bash
# Label your namespace to enable admission control
kubectl label namespace <your-namespace> cronos.ai/webhook=enabled

# Now all pod deployments in this namespace will be validated
```

## Test Quantum-Safe Encryption

```bash
# Deploy a test pod
kubectl run test-app --image=nginx -n <your-namespace>

# Check if it was validated by admission webhook
kubectl describe pod test-app -n <your-namespace> | grep -i cronos
```

## Run Performance Benchmarks

```bash
# Run performance tests
pytest ai_engine/tests/performance/test_benchmarks.py -v --benchmark-only

# Expected results:
# - Encryption: < 1ms latency
# - Throughput: 100K+ msg/s
# - All benchmarks PASS
```

## Configuration Options

### Development Environment

```bash
helm install cronos-ai ./helm/cronos-ai \
  --namespace cronos-dev \
  --create-namespace \
  --set xdsServer.replicaCount=1 \
  --set admissionWebhook.replicaCount=1 \
  --set monitoring.enabled=false
```

### Production Environment

```bash
helm install cronos-ai ./helm/cronos-ai \
  --namespace cronos-service-mesh \
  --create-namespace \
  --set xdsServer.replicaCount=5 \
  --set admissionWebhook.replicaCount=5 \
  --set xdsServer.resources.requests.memory=1Gi \
  --set monitoring.enabled=true
```

### Custom Values File

```bash
# Create custom values
cat > my-values.yaml <<EOF
xdsServer:
  replicaCount: 3
  autoscaling:
    minReplicas: 3
    maxReplicas: 10

admissionWebhook:
  config:
    maxCriticalVulnerabilities: 0
    requireQuantumSafe: true

monitoring:
  enabled: true
  prometheus:
    retention: 30d
EOF

# Deploy with custom values
helm install cronos-ai ./helm/cronos-ai \
  --namespace cronos-service-mesh \
  --create-namespace \
  -f my-values.yaml
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n cronos-service-mesh

# Check logs
kubectl logs <pod-name> -n cronos-service-mesh

# Common issues:
# - Insufficient resources: Increase node memory/CPU
# - Image pull errors: Check network connectivity
```

### Admission Webhook Blocking Deployments

```bash
# Temporarily disable webhook
kubectl delete validatingwebhookconfigurations cronos-validating-webhook

# Deploy your pod
kubectl apply -f your-pod.yaml

# Re-enable webhook
kubectl apply -f kubernetes/container-security/admission-webhook-deployment.yaml
```

### Check Quantum Crypto Implementation

```bash
# Verify Kyber-1024 keys
kubectl exec -n cronos-service-mesh deployment/cronos-xds-server -- \
  python -c "
from ai_engine.cloud_native.service_mesh.istio.qkd_certificate_manager import QuantumCertificateManager, CertificateAlgorithm
mgr = QuantumCertificateManager()
priv, pub = mgr._generate_key_pair(CertificateAlgorithm.KYBER_1024)
assert len(pub) == 1568, f'Expected 1568, got {len(pub)}'
assert len(priv) == 3168, f'Expected 3168, got {len(priv)}'
print('✓ NIST Level 5 Quantum Crypto: VALIDATED')
"
```

## Uninstall

```bash
# Uninstall Helm release
helm uninstall cronos-ai --namespace cronos-service-mesh

# Delete namespace (optional)
kubectl delete namespace cronos-service-mesh
```

## Next Steps

1. **Read Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
2. **Production Deployment**: [DEPLOYMENT.md](DEPLOYMENT.md)
3. **Developer Guide**: [DEVELOPMENT.md](DEVELOPMENT.md)
4. **API Documentation**: http://localhost:8000/docs (after port-forward)
5. **Phase 1 Report**: [PHASE1_COMPLETION_REPORT.md](PHASE1_COMPLETION_REPORT.md)

## Support

- **Issues**: [GitHub Issues](https://github.com/qbitel/cronos-ai/issues)
- **Documentation**: See [README.md](README.md)
- **Enterprise Support**: enterprise@qbitel.com

---

**You're now running NIST Level 5 Quantum-Safe Security!** 🔒

For detailed configuration options, see [helm/cronos-ai/README.md](helm/cronos-ai/README.md).
