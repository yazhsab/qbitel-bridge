# CRONOS AI Helm Chart

Official Helm chart for deploying CRONOS AI - Quantum-Safe Security Platform for Enterprise.

## Overview

CRONOS AI provides AI-powered protocol discovery, cloud-native security, and post-quantum cryptography for protecting legacy systems and modern infrastructure against quantum computing threats.

## Prerequisites

- Kubernetes 1.24+
- Helm 3.8+
- kubectl configured to access your cluster
- 4GB+ RAM per node (8GB+ recommended for production)
- 2+ CPU cores per node

## Quick Start

### Add Helm Repository (when published)

```bash
helm repo add cronos-ai https://charts.cronos.ai
helm repo update
```

### Install from Local Chart

```bash
# Clone the repository
git clone https://github.com/qbitel/cronos-ai.git
cd cronos-ai/helm/cronos-ai

# Install the chart
helm install cronos-ai . --namespace cronos-service-mesh --create-namespace
```

### Install with Custom Values

```bash
helm install cronos-ai . \
  --namespace cronos-service-mesh \
  --create-namespace \
  --set xdsServer.replicaCount=5 \
  --set monitoring.enabled=true
```

## Configuration

### Core Components

| Parameter | Description | Default |
|-----------|-------------|---------|
| `xdsServer.enabled` | Enable xDS Server for service mesh | `true` |
| `xdsServer.replicaCount` | Number of xDS Server replicas | `3` |
| `xdsServer.image.repository` | xDS Server image repository | `cronos/xds-server` |
| `xdsServer.image.tag` | xDS Server image tag | `latest` |
| `xdsServer.resources.requests.cpu` | CPU request for xDS Server | `250m` |
| `xdsServer.resources.requests.memory` | Memory request for xDS Server | `512Mi` |

| Parameter | Description | Default |
|-----------|-------------|---------|
| `admissionWebhook.enabled` | Enable admission webhook | `true` |
| `admissionWebhook.replicaCount` | Number of webhook replicas | `3` |
| `admissionWebhook.config.requireImageSignature` | Require image signatures | `true` |
| `admissionWebhook.config.requireQuantumSafe` | Require quantum-safe crypto | `true` |
| `admissionWebhook.config.maxCriticalVulnerabilities` | Max critical vulnerabilities allowed | `0` |

| Parameter | Description | Default |
|-----------|-------------|---------|
| `aiEngine.enabled` | Enable AI Engine | `true` |
| `aiEngine.replicaCount` | Number of AI Engine replicas | `2` |
| `aiEngine.resources.requests.cpu` | CPU request for AI Engine | `2000m` |
| `aiEngine.resources.requests.memory` | Memory request for AI Engine | `4Gi` |

### Monitoring

| Parameter | Description | Default |
|-----------|-------------|---------|
| `monitoring.enabled` | Enable monitoring stack | `true` |
| `monitoring.prometheus.enabled` | Enable Prometheus | `true` |
| `monitoring.prometheus.retention` | Prometheus retention period | `15d` |
| `monitoring.grafana.enabled` | Enable Grafana | `true` |
| `monitoring.grafana.adminPassword` | Grafana admin password | `cronos-ai-admin` |

### Security

| Parameter | Description | Default |
|-----------|-------------|---------|
| `security.rbac.create` | Create RBAC resources | `true` |
| `security.serviceAccount.create` | Create service accounts | `true` |
| `security.networkPolicy.enabled` | Enable network policies | `true` |
| `security.tls.enabled` | Enable TLS | `true` |

### Autoscaling

| Parameter | Description | Default |
|-----------|-------------|---------|
| `xdsServer.autoscaling.enabled` | Enable HPA for xDS Server | `true` |
| `xdsServer.autoscaling.minReplicas` | Minimum replicas | `3` |
| `xdsServer.autoscaling.maxReplicas` | Maximum replicas | `10` |
| `xdsServer.autoscaling.targetCPUUtilizationPercentage` | Target CPU % | `70` |

## Example Configurations

### Minimal Production Deployment

```yaml
# values-production.yaml
xdsServer:
  replicaCount: 5
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2000m
      memory: 4Gi

admissionWebhook:
  replicaCount: 5
  config:
    requireImageSignature: true
    requireQuantumSafe: true
    maxCriticalVulnerabilities: 0

aiEngine:
  replicaCount: 3
  resources:
    requests:
      cpu: 4000m
      memory: 8Gi
    limits:
      cpu: 8000m
      memory: 16Gi

monitoring:
  enabled: true
  prometheus:
    retention: 30d
    persistence:
      enabled: true
      size: 100Gi
```

Install:
```bash
helm install cronos-ai . -f values-production.yaml --namespace cronos-service-mesh
```

### Development Deployment

```yaml
# values-dev.yaml
xdsServer:
  replicaCount: 1
  autoscaling:
    enabled: false

admissionWebhook:
  replicaCount: 1
  autoscaling:
    enabled: false
  config:
    requireImageSignature: false
    maxHighVulnerabilities: 10

aiEngine:
  replicaCount: 1
  autoscaling:
    enabled: false

monitoring:
  enabled: true
  prometheus:
    persistence:
      enabled: false
  grafana:
    persistence:
      enabled: false
```

Install:
```bash
helm install cronos-ai . -f values-dev.yaml --namespace cronos-dev
```

### High-Availability Deployment

```yaml
# values-ha.yaml
xdsServer:
  replicaCount: 5
  autoscaling:
    minReplicas: 5
    maxReplicas: 20
  podDisruptionBudget:
    minAvailable: 3

admissionWebhook:
  replicaCount: 5
  autoscaling:
    minReplicas: 5
    maxReplicas: 15
  podDisruptionBudget:
    minAvailable: 3

affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchExpressions:
        - key: app
          operator: In
          values:
          - cronos-xds-server
      topologyKey: kubernetes.io/hostname
```

## Upgrade

```bash
# Upgrade to new version
helm upgrade cronos-ai . --namespace cronos-service-mesh

# Upgrade with new values
helm upgrade cronos-ai . -f values-production.yaml --namespace cronos-service-mesh

# Force recreation of resources
helm upgrade cronos-ai . --force --namespace cronos-service-mesh
```

## Uninstall

```bash
# Uninstall the chart
helm uninstall cronos-ai --namespace cronos-service-mesh

# Delete namespace (optional)
kubectl delete namespace cronos-service-mesh
```

## Verification

After installation, verify the deployment:

```bash
# Check pod status
kubectl get pods -n cronos-service-mesh

# Check services
kubectl get svc -n cronos-service-mesh

# Check deployment status
helm status cronos-ai -n cronos-service-mesh

# Run tests
helm test cronos-ai -n cronos-service-mesh
```

## Troubleshooting

### Pods not starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n cronos-service-mesh

# Check logs
kubectl logs <pod-name> -n cronos-service-mesh
```

### Admission webhook not working

```bash
# Check webhook configuration
kubectl get validatingwebhookconfigurations

# Check webhook logs
kubectl logs deployment/cronos-admission-webhook -n cronos-service-mesh

# Verify TLS certificates
kubectl get secret cronos-admission-webhook-tls -n cronos-service-mesh
```

### High resource usage

```bash
# Check resource usage
kubectl top pods -n cronos-service-mesh

# Scale down if needed
helm upgrade cronos-ai . --set xdsServer.replicaCount=1 --namespace cronos-service-mesh
```

## Support

- **Documentation**: https://github.com/qbitel/cronos-ai
- **Issues**: https://github.com/qbitel/cronos-ai/issues
- **Enterprise Support**: enterprise@qbitel.com

## License

Apache License 2.0 - See [LICENSE](../../LICENSE) for details.
