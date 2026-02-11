# Istio Service Mesh Integration

Quantum-safe security integration for Istio service mesh with automatic sidecar injection, mutual TLS, and comprehensive policy management.

## Features

### 1. Automatic Sidecar Injection
- **Quantum-safe proxy**: Injects QBITEL quantum encryption sidecar alongside Envoy
- **Transparent integration**: Zero application code changes required
- **Namespace-scoped**: Controlled via labels and annotations
- **Resource-efficient**: Minimal CPU/memory overhead (<100m CPU, <128Mi RAM)

### 2. Quantum Certificate Management
- **Post-quantum algorithms**: Kyber-1024 for key exchange, Dilithium-5 for signatures
- **Automatic rotation**: 90-day validity with 30-day rotation threshold
- **Secure distribution**: Kubernetes secrets with cryptographic isolation
- **Root CA management**: Self-signed quantum-safe root CA

### 3. Mutual TLS Configuration
- **Strict enforcement**: STRICT, PERMISSIVE, or DISABLE modes
- **Quantum-safe ciphers**: TLS 1.3 with post-quantum cipher suites
- **Zero-trust policies**: Default deny with explicit allow rules
- **Service-to-service encryption**: East-west traffic protection

### 4. Mesh Policy Management
- **Traffic policies**: Circuit breaking, retries, load balancing
- **Rate limiting**: Token bucket algorithm with burst support
- **Authorization**: Fine-grained access control per service
- **JWT authentication**: Token-based authentication support

## Quick Start

### 1. Install Webhook

```python
from ai_engine.cloud_native.service_mesh.istio import IstioSidecarInjector

# Initialize injector
injector = IstioSidecarInjector(
    namespace="qbitel-system",
    webhook_name="qbitel-sidecar-injector"
)

# Create webhook configuration
webhook_config = injector.create_webhook_configuration()

# Apply to Kubernetes (using kubectl or client)
# kubectl apply -f webhook_config.yaml
```

### 2. Enable Namespace for Injection

Label your namespace to enable automatic sidecar injection:

```bash
kubectl label namespace my-app qbitel-injection=enabled
```

### 3. Generate Quantum Certificates

```python
from ai_engine.cloud_native.service_mesh.istio import QuantumCertificateManager

# Initialize certificate manager
cert_manager = QuantumCertificateManager()

# Generate root CA
root_ca = cert_manager.generate_root_ca(
    subject="CN=QBITEL Root CA,O=QBITEL,C=US",
    validity_years=10
)

# Generate service certificate
service_cert = cert_manager.generate_service_certificate(
    service_name="my-service",
    namespace="default",
    ca_cert=root_ca,
    ca_private_key=root_ca["private_key_pem"]
)

# Create Kubernetes secret
secret = cert_manager.create_kubernetes_secret(
    cert_data=service_cert,
    secret_name="my-service-certs",
    namespace="default"
)

# Apply secret: kubectl apply -f secret.yaml
```

### 4. Configure Mutual TLS

```python
from ai_engine.cloud_native.service_mesh.istio import (
    MutualTLSConfigurator,
    MTLSMode
)

# Initialize configurator
mtls_config = MutualTLSConfigurator(default_mode=MTLSMode.STRICT)

# Create mesh-wide mTLS
mesh_mtls = mtls_config.create_mesh_wide_mtls(mode=MTLSMode.STRICT)

# Create namespace-specific mTLS
namespace_mtls = mtls_config.create_namespace_mtls(
    namespace="production",
    mode=MTLSMode.STRICT
)

# Create zero-trust policies
zero_trust = mtls_config.create_zero_trust_policy(
    namespace="production",
    allowed_services=["frontend", "backend", "database"]
)
```

### 5. Apply Service Policies

```python
from ai_engine.cloud_native.service_mesh.istio import (
    MeshPolicyManager,
    ServicePolicy
)

# Initialize policy manager
policy_mgr = MeshPolicyManager()

# Define service policy
service_policy = ServicePolicy(
    service_name="payment-service",
    namespace="default",
    allowed_sources=["frontend", "order-service"],
    allowed_operations=["POST", "GET"],
    require_mtls=True,
    require_jwt=True,
    rate_limit=100  # 100 req/s
)

# Create all policies for the service
policies = policy_mgr.create_complete_service_policies(service_policy)

# Apply policies: kubectl apply -f policies.yaml
```

## Architecture

### Sidecar Injection Flow

```
┌──────────────────────────────────────────────────────┐
│  Kubernetes API Server                               │
│  ┌────────────────────────────────────────────────┐  │
│  │  Pod Creation Request                          │  │
│  └─────────────────┬──────────────────────────────┘  │
│                    │                                  │
│                    ▼                                  │
│  ┌────────────────────────────────────────────────┐  │
│  │  Mutating Webhook (QBITEL)                  │  │
│  │  - Check injection labels                      │  │
│  │  - Inject quantum-safe sidecar                 │  │
│  │  - Inject init container (iptables)            │  │
│  │  - Add volumes for certs                       │  │
│  └─────────────────┬──────────────────────────────┘  │
│                    │                                  │
│                    ▼                                  │
│  ┌────────────────────────────────────────────────┐  │
│  │  Modified Pod Spec                             │  │
│  │  ┌──────────────┐  ┌──────────────────────┐   │  │
│  │  │ App Container│  │ QBITEL Quantum Proxy │   │  │
│  │  │              │  │ (Kyber + Dilithium)  │   │  │
│  │  └──────────────┘  └──────────────────────┘   │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### Quantum Certificate Flow

```
┌────────────────────────────────────────────────┐
│  Certificate Manager                           │
│  ┌──────────────────────────────────────────┐  │
│  │  1. Generate Kyber-1024 Key Pair         │  │
│  │  2. Create Certificate Structure         │  │
│  │  3. Sign with Dilithium-5                │  │
│  │  4. Encode as PEM                        │  │
│  └─────────────────┬────────────────────────┘  │
│                    │                            │
│                    ▼                            │
│  ┌──────────────────────────────────────────┐  │
│  │  Kubernetes Secret                       │  │
│  │  - tls.crt (quantum cert)                │  │
│  │  - tls.key (quantum key)                 │  │
│  │  - ca.crt (root CA)                      │  │
│  └─────────────────┬────────────────────────┘  │
│                    │                            │
│                    ▼                            │
│  ┌──────────────────────────────────────────┐  │
│  │  Mounted in Sidecar                      │  │
│  │  /etc/qbitel/certs/                      │  │
│  └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
```

## Configuration

### Sidecar Configuration

```python
from ai_engine.cloud_native.service_mesh.istio import SidecarConfig

config = SidecarConfig(
    image="qbitel/quantum-sidecar:latest",
    cpu_request="100m",
    cpu_limit="500m",
    memory_request="128Mi",
    memory_limit="512Mi",
    quantum_algorithm="kyber-1024",
    signature_algorithm="dilithium-5",
    enable_metrics=True,
    metrics_port=15090,
    proxy_port=15001
)
```

### Certificate Configuration

```python
from ai_engine.cloud_native.service_mesh.istio import (
    QuantumCertificateManager,
    CertificateAlgorithm
)

cert_mgr = QuantumCertificateManager(
    key_algorithm=CertificateAlgorithm.KYBER_1024,
    signature_algorithm=CertificateAlgorithm.DILITHIUM_5,
    cert_validity_days=90,
    rotation_threshold_days=30,
    auto_rotation=True
)
```

## Examples

### Example 1: Microservices with mTLS

```python
from ai_engine.cloud_native.service_mesh.istio import (
    MutualTLSConfigurator,
    MTLSMode
)

# Create strict mTLS for production namespace
mtls = MutualTLSConfigurator()

# 1. Namespace-wide strict mTLS
ns_mtls = mtls.create_namespace_mtls(
    namespace="production",
    mode=MTLSMode.STRICT
)

# 2. Destination rule for service mesh traffic
dest_rule = mtls.create_destination_rule(
    name="backend-mtls",
    host="backend.production.svc.cluster.local",
    namespace="production",
    enable_quantum_tls=True
)

# 3. Authorization policy
auth_policy = mtls.create_authorization_policy(
    name="backend-authz",
    namespace="production",
    action="ALLOW",
    rules=[
        {
            "from": [{
                "source": {
                    "principals": [
                        "cluster.local/ns/production/sa/frontend"
                    ]
                }
            }],
            "to": [{
                "operation": {
                    "methods": ["GET", "POST"]
                }
            }]
        }
    ]
)
```

### Example 2: Rate-Limited API Gateway

```python
from ai_engine.cloud_native.service_mesh.istio import MeshPolicyManager

policy_mgr = MeshPolicyManager()

# Create rate limit: 1000 req/s with 2000 burst
rate_limit = policy_mgr.create_rate_limit_policy(
    name="api-gateway",
    namespace="default",
    selector={"app": "api-gateway"},
    requests_per_second=1000,
    burst=2000
)
```

### Example 3: JWT Authentication

```python
# Create JWT authentication for API
jwt_auth = policy_mgr.create_jwt_authentication(
    name="api-jwt",
    namespace="default",
    issuer="https://auth.example.com",
    jwks_uri="https://auth.example.com/.well-known/jwks.json",
    audiences=["api.example.com"]
)

# Require JWT for specific service
auth_policy = mtls.create_authorization_policy(
    name="require-jwt",
    namespace="default",
    action="ALLOW",
    rules=[
        {
            "from": [{
                "source": {
                    "requestPrincipals": ["*"]
                }
            }]
        }
    ],
    selector={"app": "secure-api"}
)
```

## Performance

### Latency Overhead

- **Sidecar injection**: One-time cost at pod creation (~2-3 seconds)
- **Encryption overhead**: <5ms per request
- **Certificate validation**: <1ms (cached)

### Resource Usage

- **Sidecar CPU**: 100m request, 500m limit
- **Sidecar Memory**: 128Mi request, 512Mi limit
- **Init container**: <10m CPU, <10Mi memory

### Scalability

- **Pods per cluster**: 10,000+ supported
- **Services**: 1,000+ services
- **Certificate generation**: <100ms per certificate
- **Policy enforcement**: Real-time (no perceptible delay)

## Security

### Quantum-Safe Guarantees

- **Key exchange**: Kyber-1024 (NIST Level 5 security)
- **Signatures**: Dilithium-5 (NIST Level 5 security)
- **Forward secrecy**: Perfect forward secrecy with quantum keys
- **Certificate lifetime**: 90 days with automatic rotation

### Zero-Trust Principles

1. **Default deny**: All traffic denied by default
2. **Explicit allow**: Only specified services can communicate
3. **mTLS everywhere**: All service-to-service traffic encrypted
4. **Least privilege**: Minimal permissions per service

## Troubleshooting

### Sidecar Not Injected

Check namespace label:
```bash
kubectl get namespace my-namespace --show-labels
```

Should have: `qbitel-injection=enabled`

### Certificate Errors

Check certificate validity:
```bash
kubectl get secret my-service-certs -o yaml
```

Verify certificate not expired:
```python
cert_data = cert_manager.get_certificate("my-service", "default")
needs_rotation = cert_manager.check_rotation_needed(cert_data)
```

### mTLS Connection Failures

Verify mTLS configuration:
```python
result = mtls_config.verify_mtls_configuration(
    source_service="frontend",
    dest_service="backend",
    namespace="default"
)
print(result)
```

## API Reference

See module docstrings for detailed API documentation:

- `IstioSidecarInjector`: Automatic sidecar injection
- `QuantumCertificateManager`: Certificate lifecycle management
- `MutualTLSConfigurator`: mTLS policy configuration
- `MeshPolicyManager`: Service mesh policy management

## License

Copyright © 2025 QBITEL. All rights reserved.
