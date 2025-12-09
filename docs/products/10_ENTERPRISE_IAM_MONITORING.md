# Product 10: Enterprise IAM & Monitoring

## Overview

Enterprise IAM & Monitoring is CRONOS AI's comprehensive identity and access management platform with full observability. It provides enterprise-grade authentication, authorization, API key management, and complete monitoring with Prometheus metrics, distributed tracing, and alerting.

---

## Problem Solved

### The Challenge

Enterprise security requires robust identity and observability:

- **Identity fragmentation**: Multiple IdPs, inconsistent access controls
- **API security**: Managing thousands of API keys across services
- **Visibility gaps**: Blind spots in distributed systems
- **Alert fatigue**: Too many alerts, not enough context
- **Compliance requirements**: Audit trails and access reviews

### The CRONOS AI Solution

Enterprise IAM & Monitoring provides:
- **Unified authentication**: OIDC, SAML, LDAP, API keys
- **Role-based access control**: Fine-grained permissions
- **API key lifecycle**: Generation, rotation, revocation
- **Complete observability**: Metrics, logs, traces
- **Intelligent alerting**: Context-aware notifications

---

## Key Features

### 1. Authentication Methods

**Supported Protocols**:
| Protocol | Provider Examples | Use Case |
|----------|------------------|----------|
| **OIDC** | Azure AD, Okta, Auth0 | Web apps, SSO |
| **SAML 2.0** | ADFS, Ping Identity | Enterprise SSO |
| **LDAP** | Active Directory, OpenLDAP | Internal auth |
| **API Key** | CRONOS-generated | Service-to-service |
| **JWT** | Self-issued, External | Stateless auth |
| **mTLS** | Certificate-based | Zero-trust |

### 2. Role-Based Access Control (RBAC)

Fine-grained permission management:

```
Organization
    └── Roles
         ├── admin
         │    └── Permissions: *
         ├── security_engineer
         │    └── Permissions: security.*, compliance.*
         ├── analyst
         │    └── Permissions: read.*, analyze.*
         └── viewer
              └── Permissions: read.*
```

**Permission Structure**:
```
<resource>.<action>.<scope>

Examples:
- protocols.read.all
- security.execute.own
- compliance.generate.team
- admin.manage.organization
```

### 3. API Key Management

Complete lifecycle management:

- **Generation**: Secure random keys with configurable entropy
- **Scoping**: Permissions limited to specific resources/actions
- **Rotation**: Automatic rotation with overlap period
- **Revocation**: Instant invalidation with audit trail
- **Expiration**: Time-limited keys with auto-expiry
- **Rate limiting**: Per-key request limits

### 4. Multi-Factor Authentication (MFA)

Multiple second factors:

| Method | Implementation | Security Level |
|--------|---------------|----------------|
| **TOTP** | Google Authenticator, Authy | High |
| **WebAuthn** | FIDO2, hardware keys | Very High |
| **Push** | Mobile app approval | High |
| **SMS** | Text message codes | Medium |
| **Email** | Email codes | Medium |

### 5. Comprehensive Monitoring

Full observability stack:

- **Metrics**: Prometheus-compatible, 500+ metrics
- **Logging**: Structured JSON, centralized collection
- **Tracing**: OpenTelemetry, Jaeger integration
- **Dashboards**: Pre-built Grafana dashboards
- **Alerting**: Multi-channel notifications

---

## Technical Architecture

### IAM Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   Enterprise IAM & Monitoring                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Authentication Layer                    │  │
│  │                                                          │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐   │  │
│  │  │  OIDC   │ │  SAML   │ │  LDAP   │ │  API Keys   │   │  │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └──────┬──────┘   │  │
│  │       └───────────┼───────────┼─────────────┘           │  │
│  │                   │           │                          │  │
│  │            ┌──────▼───────────▼──────┐                  │  │
│  │            │    MFA Enforcement      │                  │  │
│  │            └───────────┬─────────────┘                  │  │
│  │                        │                                 │  │
│  └────────────────────────┼─────────────────────────────────┘  │
│                           │                                     │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │                   Authorization Layer                     │  │
│  │                                                           │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │  │
│  │  │    RBAC      │  │   Policies   │  │   Scopes      │  │  │
│  │  └──────────────┘  └──────────────┘  └───────────────┘  │  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Monitoring Layer                         │  │
│  │                                                            │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────┐ │  │
│  │  │ Prometheus │ │   Jaeger   │ │   Logs     │ │ Alerts │ │  │
│  │  │  Metrics   │ │  Tracing   │ │  (JSON)    │ │        │ │  │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────┘ │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| Auth Service | `auth.py` | Authentication handling |
| RBAC Engine | `rbac.py` | Role-based access control |
| API Key Manager | `api_keys.py` | Key lifecycle management |
| Metrics Collector | `metrics.py` | Prometheus metrics |
| Tracing | `tracing.py` | OpenTelemetry integration |
| Health Checks | `health.py` | Service health monitoring |

### Data Models

```python
@dataclass
class User:
    user_id: str
    username: str
    email: str
    display_name: str
    roles: List[str]
    groups: List[str]
    mfa_enabled: bool
    mfa_methods: List[str]
    last_login: datetime
    created_at: datetime
    status: str  # active, suspended, locked

class UserRole(str, Enum):
    ADMIN = "admin"
    SECURITY_ENGINEER = "security_engineer"
    ANALYST = "analyst"
    VIEWER = "viewer"
    SERVICE_ACCOUNT = "service_account"

@dataclass
class Permission:
    resource: str      # protocols, security, compliance, etc.
    action: str        # read, write, execute, delete, manage
    scope: str         # own, team, organization, all

@dataclass
class APIKey:
    key_id: str
    key_hash: str      # SHA-256 hash, never store plaintext
    name: str
    owner_id: str
    permissions: List[Permission]
    rate_limit: int    # requests per minute
    expires_at: datetime
    last_used: datetime
    created_at: datetime
    status: str        # active, expired, revoked

@dataclass
class Session:
    session_id: str
    user_id: str
    device_id: str
    ip_address: str
    user_agent: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    mfa_verified: bool

@dataclass
class AuditLog:
    log_id: str
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    resource_id: str
    ip_address: str
    user_agent: str
    result: str        # success, failure
    details: Dict[str, Any]
```

---

## API Reference

### Authentication

#### Login

```http
POST /api/v1/auth/login
Content-Type: application/json

{
    "username": "user@company.com",
    "password": "password123",
    "mfa_code": "123456"
}

Response:
{
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
    "token_type": "Bearer",
    "expires_in": 3600,
    "user": {
        "user_id": "user_123",
        "username": "user@company.com",
        "email": "user@company.com",
        "roles": ["security_engineer", "analyst"],
        "mfa_enabled": true
    }
}
```

#### Refresh Token

```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
    "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}

Response:
{
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "expires_in": 3600
}
```

#### OIDC Login

```http
GET /api/v1/auth/oidc/login?provider=azure_ad&redirect_uri=https://app.company.com/callback

Response:
{
    "authorization_url": "https://login.microsoftonline.com/...",
    "state": "random_state_token"
}
```

### API Key Management

#### Generate API Key

```http
POST /api/v1/auth/keys
Content-Type: application/json
Authorization: Bearer {admin_token}

{
    "name": "production-service-key",
    "permissions": [
        {"resource": "protocols", "action": "read", "scope": "all"},
        {"resource": "security", "action": "execute", "scope": "own"}
    ],
    "rate_limit": 1000,
    "expires_in_days": 90
}

Response:
{
    "key_id": "key_abc123",
    "api_key": "cronos_sk_live_xxxxxxxxxxxxxxxxxxxx",
    "name": "production-service-key",
    "permissions": [...],
    "rate_limit": 1000,
    "expires_at": "2025-04-16T10:30:00Z",
    "created_at": "2025-01-16T10:30:00Z"
}

Note: The api_key is only shown once. Store it securely.
```

#### List API Keys

```http
GET /api/v1/auth/keys
Authorization: Bearer {token}

Response:
{
    "keys": [
        {
            "key_id": "key_abc123",
            "name": "production-service-key",
            "permissions": [...],
            "rate_limit": 1000,
            "expires_at": "2025-04-16T10:30:00Z",
            "last_used": "2025-01-16T09:15:00Z",
            "status": "active"
        }
    ]
}
```

#### Rotate API Key

```http
POST /api/v1/auth/keys/{key_id}/rotate
Authorization: Bearer {token}

{
    "overlap_hours": 24
}

Response:
{
    "old_key_id": "key_abc123",
    "new_key_id": "key_def456",
    "new_api_key": "cronos_sk_live_yyyyyyyyyyyyyyyyyyyy",
    "old_key_expires_at": "2025-01-17T10:30:00Z",
    "new_key_expires_at": "2025-04-17T10:30:00Z"
}
```

#### Revoke API Key

```http
DELETE /api/v1/auth/keys/{key_id}
Authorization: Bearer {token}

Response:
{
    "key_id": "key_abc123",
    "status": "revoked",
    "revoked_at": "2025-01-16T10:30:00Z"
}
```

### User Management

#### Get Current User

```http
GET /api/v1/auth/me
Authorization: Bearer {token}

Response:
{
    "user_id": "user_123",
    "username": "user@company.com",
    "email": "user@company.com",
    "display_name": "John Doe",
    "roles": ["security_engineer", "analyst"],
    "groups": ["engineering", "security-team"],
    "mfa_enabled": true,
    "mfa_methods": ["totp", "webauthn"],
    "last_login": "2025-01-16T08:00:00Z",
    "created_at": "2024-01-01T00:00:00Z"
}
```

#### Enable MFA

```http
POST /api/v1/auth/mfa/enable
Content-Type: application/json
Authorization: Bearer {token}

{
    "method": "totp"
}

Response:
{
    "secret": "JBSWY3DPEHPK3PXP",
    "qr_code_url": "data:image/png;base64,...",
    "backup_codes": [
        "abc123def456",
        "ghi789jkl012",
        ...
    ]
}
```

### Health & Monitoring

#### Health Check

```http
GET /health

Response:
{
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2025-01-16T10:30:00Z"
}
```

#### Detailed Health

```http
GET /health/detailed
Authorization: Bearer {token}

Response:
{
    "status": "healthy",
    "components": {
        "database": {
            "status": "healthy",
            "latency_ms": 5,
            "connections": 45
        },
        "redis": {
            "status": "healthy",
            "latency_ms": 2,
            "memory_used_mb": 256
        },
        "kafka": {
            "status": "healthy",
            "lag_messages": 0,
            "brokers_available": 3
        },
        "kubernetes": {
            "status": "healthy",
            "nodes": 5,
            "pods_running": 156
        },
        "ai_models": {
            "status": "healthy",
            "loaded_models": 5,
            "inference_latency_ms": 45
        }
    },
    "uptime_seconds": 864000
}
```

#### Metrics Endpoint

```http
GET /metrics

Response:
# HELP cronos_requests_total Total HTTP requests
# TYPE cronos_requests_total counter
cronos_requests_total{method="GET",endpoint="/api/v1/protocols",status="200"} 15847

# HELP cronos_request_duration_seconds HTTP request duration
# TYPE cronos_request_duration_seconds histogram
cronos_request_duration_seconds_bucket{method="GET",endpoint="/api/v1/protocols",le="0.1"} 14500
cronos_request_duration_seconds_bucket{method="GET",endpoint="/api/v1/protocols",le="0.5"} 15700
cronos_request_duration_seconds_bucket{method="GET",endpoint="/api/v1/protocols",le="1.0"} 15847

# HELP cronos_active_sessions Current active sessions
# TYPE cronos_active_sessions gauge
cronos_active_sessions{type="user"} 1234
cronos_active_sessions{type="api_key"} 567

# ... (500+ metrics)
```

---

## Monitoring Stack

### Prometheus Metrics (500+)

**Request Metrics**:
```
cronos_requests_total{method, endpoint, status}
cronos_request_duration_seconds{method, endpoint}
cronos_request_size_bytes{method, endpoint}
cronos_response_size_bytes{method, endpoint}
```

**Authentication Metrics**:
```
cronos_auth_attempts_total{method, result}
cronos_auth_failures_total{method, reason}
cronos_mfa_challenges_total{method, result}
cronos_sessions_active{type}
cronos_api_keys_active
cronos_api_key_usage_total{key_id}
```

**Security Metrics**:
```
cronos_security_events_total{type, severity}
cronos_security_decisions_total{decision}
cronos_compliance_score{framework}
cronos_threat_intel_matches_total{type}
```

**System Metrics**:
```
cronos_cpu_usage_percent
cronos_memory_usage_bytes
cronos_disk_usage_bytes
cronos_goroutines_active
cronos_db_connections_active
cronos_cache_hits_total
cronos_cache_misses_total
```

### Distributed Tracing (OpenTelemetry)

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger-agent",
    agent_port=6831,
)

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

tracer = trace.get_tracer(__name__)

# Automatic request tracing
@app.middleware("http")
async def tracing_middleware(request, call_next):
    with tracer.start_as_current_span(
        f"{request.method} {request.url.path}",
        attributes={
            "http.method": request.method,
            "http.url": str(request.url),
            "http.user_agent": request.headers.get("user-agent"),
        }
    ) as span:
        response = await call_next(request)
        span.set_attribute("http.status_code", response.status_code)
        return response
```

**Trace Example**:
```
Request: POST /api/v1/security/analyze-event
    │
    ├── authenticate (5ms)
    │   └── verify_jwt (2ms)
    │
    ├── authorize (3ms)
    │   └── check_permissions (1ms)
    │
    ├── analyze_event (450ms)
    │   ├── normalize_event (10ms)
    │   ├── threat_analysis (200ms)
    │   │   ├── ml_classification (50ms)
    │   │   └── llm_analysis (150ms)
    │   └── response_generation (240ms)
    │
    └── audit_log (5ms)

Total: 463ms
```

### Alerting Configuration

```yaml
alerting:
  enabled: true

  channels:
    - name: security-critical
      type: pagerduty
      integration_key: ${PAGERDUTY_KEY}
      severity_threshold: critical

    - name: security-team
      type: slack
      webhook: ${SLACK_WEBHOOK}
      channel: "#security-alerts"
      severity_threshold: high

    - name: ops-team
      type: email
      recipients:
        - ops@company.com
        - security@company.com
      severity_threshold: medium

  rules:
    - name: high_error_rate
      condition: "rate(cronos_requests_total{status=~'5..'}[5m]) > 0.1"
      severity: high
      channels: [security-team, ops-team]
      message: "High error rate detected: {{ $value | printf \"%.2f\" }}%"

    - name: authentication_failures
      condition: "rate(cronos_auth_failures_total[5m]) > 10"
      severity: high
      channels: [security-team]
      message: "High authentication failure rate"

    - name: security_incident
      condition: "cronos_security_events_total{severity='critical'} > 0"
      severity: critical
      channels: [security-critical, security-team]
      message: "Critical security incident detected"

    - name: service_down
      condition: "up == 0"
      severity: critical
      channels: [security-critical, ops-team]
      message: "Service {{ $labels.job }} is down"
```

---

## Configuration

```yaml
iam:
  enabled: true
  service_name: "CRONOS AI Authentication Service"
  domain: "company.com"

  # Session Management
  session:
    timeout: 8h
    idle_timeout: 30m
    max_sessions_per_user: 5
    cookie_name: "CRONOS_SESSION"
    secure_cookie: true
    same_site: "Strict"

  # JWT Configuration
  jwt:
    issuer: "cronos-auth"
    audience: ["cronos-api", "cronos-console"]
    access_token_expiry: 1h
    refresh_token_expiry: 24h
    algorithm: "RS256"
    private_key_path: /etc/cronos/jwt-private.pem
    public_key_path: /etc/cronos/jwt-public.pem

  # OIDC Providers
  oidc:
    providers:
      - name: azure_ad
        enabled: true
        issuer_url: https://login.microsoftonline.com/${TENANT}/v2.0
        client_id: ${AZURE_CLIENT_ID}
        client_secret: ${AZURE_CLIENT_SECRET}
        scopes: [openid, profile, email]

      - name: okta
        enabled: true
        issuer_url: https://company.okta.com
        client_id: ${OKTA_CLIENT_ID}
        client_secret: ${OKTA_CLIENT_SECRET}
        scopes: [openid, profile, email, groups]

  # SAML Providers
  saml:
    providers:
      - name: adfs
        enabled: true
        entity_id: https://adfs.company.com
        sso_url: https://adfs.company.com/adfs/ls/
        certificate_path: /etc/cronos/adfs-cert.pem

  # LDAP Providers
  ldap:
    providers:
      - name: active_directory
        enabled: true
        host: ldap.company.com
        port: 636
        use_ssl: true
        base_dn: DC=company,DC=com
        bind_dn: CN=cronos-service,OU=Services,DC=company,DC=com
        bind_password: ${LDAP_PASSWORD}

  # MFA Configuration
  mfa:
    required: true
    methods: [totp, webauthn, push]
    grace_period: 7d
    backup_codes_count: 10

  # Password Policy
  password:
    min_length: 12
    require_upper: true
    require_lower: true
    require_digits: true
    require_special: true
    max_age: 90d
    history_size: 12
    lockout_threshold: 5
    lockout_duration: 15m

  # API Keys
  api_keys:
    prefix: "cronos_sk_"
    entropy_bits: 256
    default_expiry: 90d
    max_per_user: 10
    rate_limit_default: 1000

monitoring:
  enabled: true

  prometheus:
    enabled: true
    port: 9090
    path: /metrics

  tracing:
    enabled: true
    provider: jaeger
    endpoint: http://jaeger-collector:14268/api/traces
    sample_rate: 0.1  # 10% of requests

  logging:
    level: INFO
    format: json
    include_request_body: false
    include_response_body: false
    sensitive_fields: [password, token, api_key, secret]

  health:
    enabled: true
    path: /health
    detailed_path: /health/detailed
    check_interval: 30s
```

---

## Conclusion

Enterprise IAM & Monitoring provides the foundation for secure, observable enterprise systems. With comprehensive authentication options, fine-grained authorization, API key lifecycle management, and complete observability through metrics, tracing, and alerting, organizations can maintain security and operational excellence at scale.
