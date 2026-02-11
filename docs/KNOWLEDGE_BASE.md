# QBITEL - Comprehensive Knowledge Base

**Version**: 2.1.0
**Last Updated**: 2025-11-22
**Purpose**: Reference guide for AI coding agents and developers for all future enhancements and implementations
**Coverage**: Python, Go, Rust, TypeScript/React - Complete multi-language codebase

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Directory Structure](#3-directory-structure)
4. [Core Components](#4-core-components)
5. [API Layer](#5-api-layer)
6. [Database Models & Schemas](#6-database-models--schemas)
7. [Service Layer](#7-service-layer)
8. [Security Implementation](#8-security-implementation)
9. [AI/ML Components](#9-aiml-components)
10. [Cloud-Native Infrastructure](#10-cloud-native-infrastructure)
11. [Configuration Management](#11-configuration-management)
12. [External Integrations](#12-external-integrations)
13. [Testing Strategy](#13-testing-strategy)
14. [Deployment Guide](#14-deployment-guide)
15. [Design Patterns & Best Practices](#15-design-patterns--best-practices)
16. [Coding Standards](#16-coding-standards)
17. [Troubleshooting Guide](#17-troubleshooting-guide)
18. [Go Services](#18-go-services)
19. [Rust Dataplane](#19-rust-dataplane)
20. [Frontend Console](#20-frontend-console)

---

## 1. Project Overview

### 1.1 What is QBITEL?

QBITEL is an **enterprise-grade, quantum-safe security platform** that provides:

- **AI-Powered Protocol Discovery** - Automatically learns undocumented/proprietary protocols
- **Post-Quantum Cryptography** - NIST Level 5 compliant (Kyber-1024, Dilithium-5)
- **Agentic AI Security** - Autonomous decision-making with LLM-powered threat analysis
- **Cloud-Native Security** - Service mesh integration, container security, eBPF monitoring
- **Zero-Touch Deployment** - Protects existing systems without code changes

### 1.2 Key Statistics

| Metric | Value |
|--------|-------|
| **Python Files** | 442+ |
| **Go Files** | 15+ |
| **Rust Files** | 64+ |
| **TypeScript/React Files** | 39+ |
| **Total Lines of Code** | 250,000+ |
| **Documentation Files** | 82 markdown files |
| **Test Files** | 160+ |
| **Test Coverage** | 85% |
| **Production Readiness** | 85-90% |

### 1.3 Technology Stack

| Layer | Technologies |
|-------|--------------|
| **Backend** | Python 3.9+, FastAPI 0.104.1, gRPC |
| **AI/ML** | PyTorch 2.2.2, Scikit-learn, SHAP, LIME |
| **LLM** | Ollama (primary), Claude, GPT-4 |
| **Database** | PostgreSQL 15, Redis 7, TimescaleDB, ChromaDB |
| **Cryptography** | Kyber-1024, Dilithium-5, AES-256-GCM |
| **Container** | Docker, Kubernetes 1.24+ |
| **Service Mesh** | Istio, Envoy Proxy |
| **Monitoring** | Prometheus, OpenTelemetry, Jaeger, Sentry |
| **Cloud** | AWS, Azure, GCP SDKs |
| **Languages** | Python (primary), Go, Rust, TypeScript |
| **Go Services** | Control Plane, Management API, Device Agent |
| **Rust Crates** | PQC-TLS, I/O Engine, DPDK, DPI, K8s Operator, Protocol Adapters |
| **Frontend** | React 18, Material-UI 5, Vite, OIDC Auth |

### 1.4 License

Apache License 2.0

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           QBITEL Platform                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   REST API   │  │   gRPC API   │  │  WebSocket   │  │  Admin UI   │ │
│  │  (FastAPI)   │  │              │  │  (Copilot)   │  │   (React)   │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
│         │                 │                 │                 │        │
│         └─────────────────┼─────────────────┼─────────────────┘        │
│                           │                 │                          │
│  ┌────────────────────────┴─────────────────┴────────────────────────┐ │
│  │                        AI ENGINE CORE                             │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐       │ │
│  │  │   Protocol     │  │    Anomaly     │  │     Field      │       │ │
│  │  │   Discovery    │  │   Detection    │  │   Detection    │       │ │
│  │  │   (PCFG)       │  │   (Ensemble)   │  │  (BiLSTM-CRF)  │       │ │
│  │  └────────────────┘  └────────────────┘  └────────────────┘       │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     AGENTIC AI LAYER                             │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │  │
│  │  │  Decision  │  │  Security  │  │   Copilot  │  │    RAG     │  │  │
│  │  │   Engine   │  │ Orchestr.  │  │  (LLM UI)  │  │   Engine   │  │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    CLOUD-NATIVE SECURITY                         │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │  │
│  │  │  Service   │  │ Container  │  │   Event    │  │   Cloud    │  │  │
│  │  │   Mesh     │  │  Security  │  │ Streaming  │  │   Integr.  │  │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  PostgreSQL  │  │    Redis     │  │   Kafka      │                  │
│  │   (Data)     │  │   (Cache)    │  │  (Events)    │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Microservices Architecture

| Service | Language | Purpose | Port |
|---------|----------|---------|------|
| AI Engine | Python | Core ML/AI processing | 8000 (REST), 50051 (gRPC) |
| Control Plane | Go | Policy engine, configuration | 8080 |
| Management API | Go | Device lifecycle, attestation | 8081 |
| Device Agent | Go | On-device security agent | N/A |
| Dataplane | Rust | High-performance packet processing | N/A |
| Admin Console | TypeScript/React | Web UI dashboard | 3000 |

### 2.3 Entry Points

| Entry Point | Location | Command |
|-------------|----------|---------|
| AI Engine | `ai_engine/__main__.py` | `python -m ai_engine` |
| API Server | `ai_engine/api/server.py` | Automatic via __main__ |
| Control Plane | `go/controlplane/cmd/controlplane/main.go` | `go run main.go` |
| Management API | `go/mgmtapi/cmd/mgmtapi/main.go` | `go run main.go` |
| Dataplane | `rust/dataplane/src/main.rs` | `cargo run` |
| Console UI | `ui/console/src/main.tsx` | `npm run dev` |

---

## 3. Directory Structure

```
qbitel/
├── ai_engine/                    # Core AI/ML engine (Python)
│   ├── __main__.py              # Application entry point
│   ├── api/                     # REST & gRPC API layer
│   │   ├── rest.py              # FastAPI application
│   │   ├── server.py            # Server management
│   │   ├── auth.py              # Authentication
│   │   ├── middleware.py        # Request middleware
│   │   ├── rate_limiter.py      # Rate limiting
│   │   ├── input_validation.py  # Input validation
│   │   └── *_endpoints.py       # Feature-specific endpoints
│   ├── core/                    # Core infrastructure
│   │   ├── engine.py            # Main AI engine orchestrator
│   │   ├── config.py            # Configuration management
│   │   ├── database_manager.py  # Database connection pooling
│   │   ├── exceptions.py        # Custom exceptions
│   │   └── structured_logging.py # Logging infrastructure
│   ├── discovery/               # Protocol discovery
│   │   ├── pcfg_inference.py    # Grammar inference
│   │   ├── grammar_learner.py   # Grammar learning
│   │   ├── parser_generator.py  # Parser generation
│   │   └── protocol_discovery_orchestrator.py
│   ├── detection/               # Field detection
│   │   ├── field_detector.py    # BiLSTM-CRF detector
│   │   └── anomaly_detector.py  # Anomaly detection
│   ├── anomaly/                 # Anomaly detection ensemble
│   │   ├── ensemble_detector.py # Ensemble methods
│   │   ├── lstm_detector.py     # LSTM-based detection
│   │   ├── vae_detector.py      # VAE-based detection
│   │   └── isolation_forest.py  # Isolation Forest
│   ├── security/                # Security orchestration
│   │   ├── decision_engine.py   # Zero-touch decisions
│   │   ├── security_service.py  # Security orchestrator
│   │   ├── threat_analyzer.py   # Threat analysis
│   │   └── integrations/        # Cloud security integrations
│   ├── llm/                     # LLM integration
│   │   ├── unified_llm_service.py # Multi-provider LLM
│   │   ├── rag_engine.py        # RAG for context
│   │   └── *_client.py          # Provider clients
│   ├── copilot/                 # Protocol copilot
│   │   ├── protocol_copilot.py  # Main copilot interface
│   │   ├── context_manager.py   # Session context
│   │   └── playbook_generator.py # Response playbooks
│   ├── compliance/              # Compliance automation
│   │   ├── compliance_service.py # Compliance orchestrator
│   │   ├── gdpr_compliance.py   # GDPR automation
│   │   └── soc2_controls.py     # SOC2 mapping
│   ├── cloud_native/            # Cloud-native security
│   │   ├── service_mesh/        # Istio/Envoy integration
│   │   ├── container_security/  # Container scanning
│   │   ├── event_streaming/     # Kafka integration
│   │   └── cloud_integrations/  # AWS/Azure/GCP
│   ├── monitoring/              # Observability
│   │   ├── metrics.py           # Prometheus metrics
│   │   └── opentelemetry_tracing.py # Distributed tracing
│   ├── explainability/          # AI explainability
│   │   ├── shap_explainer.py    # SHAP values
│   │   └── lime_explainer.py    # LIME explanations
│   ├── marketplace/             # Protocol marketplace
│   ├── translation/             # Protocol translation
│   ├── models/                  # ML model definitions
│   ├── alembic/                 # Database migrations
│   └── tests/                   # Test suite (160+ files)
│
├── go/                          # Go microservices
│   ├── controlplane/            # Policy control plane
│   ├── mgmtapi/                 # Management API
│   └── agents/device-agent/     # Device agent
│
├── rust/                        # Rust dataplane
│   └── dataplane/               # High-performance processing
│       └── crates/              # Rust crates
│           ├── pqc_tls/         # Post-quantum TLS
│           └── adapters/        # Protocol adapters
│
├── ui/                          # Frontend applications
│   └── console/                 # React admin console
│       ├── src/components/      # React components
│       └── src/api/             # API clients
│
├── config/                      # Configuration files
│   ├── qbitel.yaml           # Default config
│   ├── qbitel.production.yaml # Production config
│   └── environments/            # Environment-specific
│
├── helm/                        # Helm charts
│   └── qbitel/               # Main Helm chart
│
├── kubernetes/                  # Kubernetes manifests
│   ├── service-mesh/            # Istio configs
│   └── container-security/      # Security configs
│
├── docker/                      # Docker configurations
│   ├── xds-server/              # xDS server container
│   └── admission-webhook/       # Webhook container
│
├── ops/                         # Operations & deployment
│   ├── deploy/                  # Deployment scripts
│   ├── monitoring/              # Monitoring configs
│   └── security/                # Security hardening
│
├── security/                    # Security validation
│   └── validation/              # Security validators
│
├── integration/                 # Integration layer
│   ├── database/                # DB integrations
│   └── streaming/               # Event streaming
│
├── datasets/                    # Training/test data
│   ├── anomaly_detection/       # Anomaly samples
│   ├── protocols/               # Protocol samples
│   └── threat_intelligence/     # Threat data
│
├── docs/                        # Documentation
│   └── products/                # Product guides (01-10)
│
├── scripts/                     # Utility scripts
├── tests/                       # Additional tests
└── .github/workflows/           # CI/CD pipelines
```

---

## 4. Core Components

### 4.1 AI Engine Core (`ai_engine/core/engine.py`)

The `QbitelAIEngine` class is the main orchestrator:

```python
class QbitelAIEngine:
    """Central AI engine orchestrating all ML/AI capabilities"""

    # State machine: "initialized" -> "ready" -> "stopped"/"failed"
    state: str

    # Core components
    protocol_discovery: Optional[PCFGInference]
    field_detector: Optional[FieldDetector]
    anomaly_detector: Optional[EnsembleAnomalyDetector]
    model_registry: Optional[ModelRegistry]

    # Key methods
    async def discover_protocol(model_input) -> DiscoveryResult
    async def detect_fields(model_input) -> FieldPrediction
    async def detect_anomalies(model_input) -> AnomalyResult
    async def get_status() -> Dict[str, Any]
```

**Metrics exposed:**
- `qbitel_inference_total` (Counter)
- `qbitel_inference_duration_seconds` (Histogram)
- `qbitel_model_accuracy` (Gauge)

### 4.2 Protocol Discovery Orchestrator (`ai_engine/discovery/protocol_discovery_orchestrator.py`)

Multi-phase discovery pipeline:

```python
class DiscoveryPhase(Enum):
    INITIALIZATION = "initialization"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    CLASSIFICATION = "classification"
    GRAMMAR_LEARNING = "grammar_learning"
    PARSER_GENERATION = "parser_generation"
    VALIDATION = "validation"
    COMPLETION = "completion"

@dataclass
class DiscoveryRequest:
    messages: List[bytes]
    known_protocol: Optional[str] = None
    training_mode: bool = False
    confidence_threshold: float = 0.7
    generate_parser: bool = True
    validate_results: bool = True

@dataclass
class DiscoveryResult:
    protocol_type: str
    confidence: float
    grammar: Optional[Grammar]
    parser: Optional[GeneratedParser]
    validation_result: Optional[ValidationResult]
    processing_time: float
    phases_completed: List[DiscoveryPhase]
```

### 4.3 Field Detection (`ai_engine/detection/field_detector.py`)

BiLSTM-CRF architecture for field boundary detection:

```python
class FieldDetector:
    """Detects field boundaries using BiLSTM-CRF"""

    # Model architecture
    # BiLSTM: 128 hidden units, bidirectional
    # CRF: Sequence labeling with IOB scheme

    # Output tags
    # O = Outside
    # B-FIELD = Begin field
    # I-FIELD = Inside field

@dataclass
class FieldPrediction:
    message_length: int
    detected_fields: List[FieldBoundary]
    confidence_score: float
    processing_time: float
    model_version: str

class FieldType(Enum):
    INTEGER = "integer"
    STRING = "string"
    BINARY = "binary"
    TIMESTAMP = "timestamp"
    ADDRESS = "address"
    LENGTH = "length"
    CHECKSUM = "checksum"
```

### 4.4 Anomaly Detection Ensemble (`ai_engine/anomaly/ensemble_detector.py`)

```python
class EnsembleAnomalyDetector:
    """Combines multiple detection strategies"""

    # Detection methods:
    # - Statistical deviation (z-score)
    # - Volatility analysis (entropy)
    # - Pattern recognition
    # - Outlier detection (isolation-inspired)

@dataclass
class AnomalyResult:
    score: float           # 0-1 anomaly score
    confidence: float      # Model confidence
    is_anomalous: bool     # Binary decision
    individual_scores: Dict[str, float]
    explanation: str
    timestamp: datetime
```

### 4.5 Zero-Touch Decision Engine (`ai_engine/security/decision_engine.py`)

LLM-powered autonomous security decisions:

```python
class ZeroTouchDecisionEngine:
    """Autonomous security decision-making"""

    # Decision flow:
    # Security Event → Threat Analysis (ML + LLM) → Business Impact
    # → Response Options → LLM Decision → Safety Check → Execute/Escalate

    # Confidence thresholds:
    # >= 0.95 + Low Risk  → AUTONOMOUS EXECUTION
    # >= 0.85 + Med Risk  → AUTO-APPROVE & EXECUTE
    # >= 0.50             → ESCALATE TO SOC ANALYST
    # < 0.50              → ESCALATE TO SOC MANAGER
```

### 4.6 Security Orchestrator (`ai_engine/security/security_service.py`)

```python
class SecurityOrchestratorService:
    """Main security orchestrator"""

    decision_engine: ZeroTouchDecisionEngine
    threat_analyzer: ThreatAnalyzer
    active_incidents: Dict[str, IncidentInfo]  # Max 50 concurrent

    async def analyze_threat(event) -> ThreatAnalysis
    async def execute_response(analysis) -> AutomatedResponse
    async def escalate_to_human(analysis) -> HumanEscalation
```

---

## 5. API Layer

### 5.1 FastAPI Application (`ai_engine/api/rest.py`)

```python
def create_app(config: Config) -> FastAPI:
    app = FastAPI(
        title="QBITEL Bridge Engine",
        version="2.0.0",
        description="Enterprise-grade AI-powered protocol discovery"
    )

    # Middleware stack (order matters!)
    # 1. GracefulShutdownMiddleware
    # 2. RequestTrackingMiddleware (correlation IDs)
    # 3. PayloadSizeLimitMiddleware (10MB default)
    # 4. ContentTypeValidationMiddleware
    # 5. RequestLoggingMiddleware
    # 6. SecurityHeadersMiddleware (HSTS, CSP)
    # 7. TrustedHostMiddleware
    # 8. GZipMiddleware

    # Routers
    app.include_router(copilot_router, prefix="/api/v1/copilot")
    app.include_router(translation_router, prefix="/api/v1/translation")
    app.include_router(security_router, prefix="/api/v1/security")
    app.include_router(marketplace_router, prefix="/api/v1/marketplace")
    app.include_router(compliance_router, prefix="/api/v1/compliance")

    return app
```

### 5.2 API Endpoints Reference

#### Copilot Endpoints (`/api/v1/copilot/`)

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/` | Get copilot info | Optional |
| POST | `/query` | Submit query | Required |
| WS | `/ws` | WebSocket streaming | Required |
| GET | `/sessions` | List sessions | Required |
| GET | `/sessions/{id}` | Get session | Required |
| DELETE | `/sessions/{id}` | Delete session | Required |
| GET | `/health` | Copilot health | Optional |

#### Protocol Discovery Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/api/v1/discover` | Discover protocol | Required |
| POST | `/api/v1/fields/detect` | Detect fields | Required |
| POST | `/api/v1/anomalies/detect` | Detect anomalies | Required |

#### Security Endpoints (`/api/v1/security/`)

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/analyze` | Analyze security event | Required |
| POST | `/respond` | Execute response | Elevated |
| GET | `/incidents` | List incidents | Required |
| GET | `/incidents/{id}` | Get incident | Required |

#### Health Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Basic health |
| GET | `/healthz` | Kubernetes liveness |
| GET | `/readyz` | Kubernetes readiness |
| GET | `/metrics` | Prometheus metrics |

### 5.3 Request/Response Schemas (`ai_engine/api/schemas.py`)

```python
class CopilotQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    query_type: Optional[QueryType] = None
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    packet_data: Optional[bytes] = None
    enable_learning: bool = True
    preferred_provider: Optional[LLMProvider] = None

class CopilotResponse(BaseModel):
    success: bool
    response: str
    confidence: float  # 0-1
    query_type: str
    processing_time: float
    session_id: str
    suggestions: List[str]
    source_data: List[Dict]

class QueryType(Enum):
    PROTOCOL_ANALYSIS = "protocol_analysis"
    SECURITY_ASSESSMENT = "security_assessment"
    FIELD_DETECTION = "field_detection"
    COMPLIANCE_CHECK = "compliance_check"
    ANOMALY_DETECTION = "anomaly_detection"
    GENERAL_QUESTION = "general_question"

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
```

### 5.4 Authentication (`ai_engine/api/auth.py`)

```python
# Token types
# - Access Token: 30 minutes (8 hours in prod)
# - Refresh Token: 7 days
# - API Key: 90 days (configurable)

# Auth headers supported:
# Authorization: Bearer <jwt_token>
# X-API-Key: <api_key>

# MFA methods supported:
# - TOTP (Time-based OTP)
# - SMS
# - Email
# - Hardware Token (WebAuthn)

# User roles (RBAC)
class UserRole(Enum):
    ADMINISTRATOR = "administrator"       # Full access
    SECURITY_ANALYST = "security_analyst" # Security ops
    OPERATOR = "operator"                 # Operational tasks
    VIEWER = "viewer"                     # Read-only
    API_USER = "api_user"                 # API-only
```

---

## 6. Database Models & Schemas

### 6.1 Core Models (`ai_engine/models/database.py`)

```python
# Users & Authentication
class User(Base):
    __tablename__ = "users"

    id: UUID                    # Primary key
    username: str               # Unique, indexed
    email: str                  # Unique, indexed
    password_hash: str          # bcrypt hashed
    role: UserRole              # RBAC role
    permissions: JSONB          # Fine-grained permissions

    # MFA
    mfa_enabled: bool
    mfa_method: MFAMethod
    mfa_secret: EncryptedString

    # OAuth/SAML
    oauth_provider: Optional[str]
    saml_name_id: Optional[str]

    # Security tracking
    last_login: DateTime
    failed_login_attempts: int
    account_locked_until: Optional[DateTime]

class APIKey(Base):
    __tablename__ = "api_keys"

    id: UUID
    key_hash: str              # SHA-256, unique
    key_prefix: str            # First 20 chars for display
    user_id: UUID              # FK to users
    status: APIKeyStatus       # ACTIVE|REVOKED|EXPIRED
    expires_at: Optional[DateTime]
    rate_limit_per_minute: int # Default 100
    permissions: JSONB         # Can restrict user permissions

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: UUID
    action: AuditAction        # 50+ action types
    user_id: Optional[UUID]
    resource_type: str
    resource_id: str
    details: JSONB
    success: bool
    created_at: DateTime       # Indexed
```

### 6.2 Database Migrations (`ai_engine/alembic/versions/`)

| Migration | Purpose |
|-----------|---------|
| `001_initial_auth_schema.py` | Users, API Keys, Sessions |
| `002_add_explainability_tables.py` | Explanation audit trails |
| `003_add_encrypted_fields.py` | Field encryption support |
| `004_add_marketplace_tables.py` | Protocol marketplace |

### 6.3 Database Manager (`ai_engine/core/database_manager.py`)

```python
class DatabaseManager:
    """Production-ready async SQLAlchemy connection pool"""

    # Connection pool settings
    pool_size: int = 10        # Dev: 10, Prod: 50
    max_overflow: int = 20
    pool_timeout: int = 30     # seconds
    pool_recycle: int = 3600   # Prevent stale connections

    # Methods
    async def get_session() -> AsyncSession
    async def begin_transaction() -> AsyncTransaction
    async def health_check() -> Dict[str, Any]

    # Context manager pattern
    async with database_manager.transaction() as session:
        # Auto-commit on exit, rollback on exception
        pass
```

---

## 7. Service Layer

### 7.1 Unified LLM Service (`ai_engine/llm/unified_llm_service.py`)

```python
class UnifiedLLMService:
    """Multi-provider LLM with intelligent fallback"""

    # Provider priority (on-premise first)
    # 1. Ollama (local, air-gapped)
    # 2. vLLM (local, high-performance)
    # 3. Anthropic Claude (cloud fallback)
    # 4. OpenAI GPT-4 (cloud fallback)

    # Domain-specific routing
    routing_config = {
        "protocol_copilot": {
            "primary": "ollama",
            "fallback": ["anthropic", "openai"]
        },
        "security_orchestrator": {
            "primary": "anthropic",
            "fallback": ["openai"]
        }
    }

    async def process_request(request: LLMRequest) -> LLMResponse
    async def stream_response(request: LLMRequest) -> AsyncIterator[str]

@dataclass
class LLMRequest:
    prompt: str
    feature_domain: str        # For routing
    max_tokens: int = 2000
    temperature: float = 0.3
    system_prompt: Optional[str] = None
    stream: bool = False
```

### 7.2 RAG Engine (`ai_engine/llm/rag_engine.py`)

```python
class RAGEngine:
    """Retrieval-Augmented Generation for contextual intelligence"""

    # Vector database: ChromaDB
    # Embeddings: Sentence Transformers
    # Fallback: In-memory implementation

    async def add_documents(documents: List[Document])
    async def query(query: str, top_k: int = 5) -> List[RelevantDoc]
    async def augment_prompt(prompt: str) -> str
```

### 7.3 Compliance Service (`ai_engine/compliance/compliance_service.py`)

```python
class ComplianceService:
    """Multi-framework compliance automation"""

    # Supported frameworks
    frameworks = [
        "PCI-DSS",
        "HIPAA",
        "SOX",
        "GDPR",
        "NIST CSF",
        "ISO 27001",
        "SOC2"
    ]

    async def assess_compliance(system_id, framework) -> Assessment
    async def generate_report(assessment, format) -> Report
    async def record_audit_event(event) -> AuditLog
```

---

## 8. Security Implementation

### 8.1 Post-Quantum Cryptography

| Algorithm | Purpose | Security Level |
|-----------|---------|----------------|
| Kyber-1024 | Key Encapsulation | NIST Level 5 |
| Dilithium-5 | Digital Signatures | NIST Level 5 |
| AES-256-GCM | Symmetric Encryption | 256-bit |
| SPHINCS+ | Hash-based Signatures | Backup |

**Implementation locations:**
- `ai_engine/cloud_native/service_mesh/qkd_certificate_manager.py`
- `rust/dataplane/crates/pqc_tls/`

### 8.2 Authentication Flow

```
POST /login
    │
    ▼
Verify Credentials (bcrypt)
    │
    ├─ Invalid → 401 (increment failed attempts)
    │
    └─ Valid → Check MFA
                  │
                  ├─ MFA Enabled → Verify Code
                  │                    │
                  │                    ├─ Valid → Generate Tokens
                  │                    │
                  │                    └─ Invalid → 401
                  │
                  └─ MFA Disabled → Generate Tokens
                                        │
                                        ▼
                                   Return:
                                   - Access Token (30min)
                                   - Refresh Token (7 days)
                                   - User Info
```

### 8.3 Security Headers (Middleware)

```python
# SecurityHeadersMiddleware adds:
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
Content-Security-Policy: default-src 'self'; script-src 'self'
```

### 8.4 Input Validation (`ai_engine/api/input_validation.py`)

**Patterns detected and blocked:**
- SQL Injection: `UNION SELECT`, `OR 1=1`, `DROP TABLE`
- XSS: `<script>`, `javascript:`, `onclick=`
- Command Injection: `; | $ ( ) \``
- Path Traversal: `../`, `%2e%2e`, `/etc/`

### 8.5 Rate Limiting (`ai_engine/api/rate_limiter.py`)

```python
# Default limits
requests_per_minute: 100
requests_per_hour: 1000
burst_size: 200

# Response headers
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1701696485
Retry-After: 60  # When rate limited
```

---

## 9. AI/ML Components

### 9.1 Model Architecture Summary

| Model | Type | Purpose | Location |
|-------|------|---------|----------|
| BiLSTM-CRF | Deep Learning | Field detection | `detection/field_detector.py` |
| LSTM | Deep Learning | Time-series anomaly | `anomaly/lstm_detector.py` |
| VAE | Deep Learning | Feature extraction | `anomaly/vae_detector.py` |
| Isolation Forest | Ensemble | Anomaly detection | `anomaly/isolation_forest.py` |
| PCFG | Statistical | Grammar inference | `discovery/pcfg_inference.py` |
| CNN | Deep Learning | Protocol classification | Various |

### 9.2 Training Configuration

```yaml
training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
  early_stopping_patience: 10
  validation_split: 0.2

model:
  lstm_hidden_size: 128
  lstm_num_layers: 2
  cnn_filter_sizes: [3, 4, 5]
  cnn_num_filters: 100
  dropout_rate: 0.2
```

### 9.3 Explainability (`ai_engine/explainability/`)

| Tool | Purpose | Location |
|------|---------|----------|
| SHAP | Feature importance | `shap_explainer.py` |
| LIME | Local explanations | `lime_explainer.py` |
| Audit Logger | Decision trails | `audit_logger.py` |
| Drift Monitor | Performance tracking | `drift_monitor.py` |

---

## 10. Cloud-Native Infrastructure

### 10.1 Service Mesh (`ai_engine/cloud_native/service_mesh/`)

| Component | Purpose | File |
|-----------|---------|------|
| xDS Server | Envoy configuration | `xds_server.py` |
| mTLS Config | Mutual TLS | `mtls_config.py` |
| QKD Cert Manager | Quantum-safe certs | `qkd_certificate_manager.py` |
| Sidecar Injector | Envoy injection | `sidecar_injector.py` |

### 10.2 Container Security (`ai_engine/cloud_native/container_security/`)

| Component | Purpose | File |
|-----------|---------|------|
| Webhook Server | Admission control | `admission_control/webhook_server.py` |
| Vulnerability Scanner | Image scanning | `image_scanning/vulnerability_scanner.py` |
| eBPF Monitor | Syscall monitoring | `runtime_protection/ebpf_monitor.py` |
| Dilithium Signer | PQ signatures | `signing/dilithium_signer.py` |

### 10.3 Cloud Integrations

| Cloud | Integration | File |
|-------|-------------|------|
| AWS | Security Hub | `cloud_integrations/aws/security_hub.py` |
| Azure | Sentinel | `cloud_integrations/azure/sentinel.py` |
| GCP | Security Command Center | `cloud_integrations/gcp/security_command_center.py` |

### 10.4 Event Streaming (`ai_engine/cloud_native/event_streaming/`)

```python
# Kafka Producer with AES-256-GCM encryption
# Throughput: 100,000+ msg/s
# Event types: PACKET_CAPTURED, PROTOCOL_DISCOVERED, AI_ANALYSIS, SECURITY_EVENT
```

---

## 11. Configuration Management

### 11.1 Configuration Files

| File | Purpose | Environment |
|------|---------|-------------|
| `config/qbitel.yaml` | Default config | Development |
| `config/qbitel.production.yaml` | Production config | Production |
| `config/environments/staging.yaml` | Staging overrides | Staging |
| `config/compliance.yaml` | Compliance config | All |

### 11.2 Environment Variables

**Required for Production:**

```bash
# Database
DATABASE_HOST=db.example.com
DATABASE_PORT=5432
DATABASE_NAME=qbitel_prod
DATABASE_USER=qbitel
DATABASE_PASSWORD=<32+ chars, from secrets manager>

# Redis
REDIS_HOST=redis.example.com
REDIS_PORT=6379
REDIS_PASSWORD=<from secrets manager>

# Security (MANDATORY)
JWT_SECRET=<32+ chars>
API_KEY=<32+ chars>
ENCRYPTION_KEY=<32+ chars, AES-256-GCM>

# TLS
TLS_ENABLED=true
TLS_CERT_FILE=/etc/qbitel/certs/server.crt
TLS_KEY_FILE=/etc/qbitel/certs/server.key

# LLM (On-premise default)
QBITEL_LLM_PROVIDER=ollama
QBITEL_LLM_ENDPOINT=http://localhost:11434
QBITEL_AIRGAPPED_MODE=true

# Monitoring
SENTRY_DSN=<sentry endpoint>
LOG_LEVEL=INFO
```

### 11.3 Configuration Loading Order

1. Load base config from `config/qbitel.yaml`
2. Override with environment-specific config
3. Load environment variables (highest priority)
4. Fetch secrets from secrets manager (Vault, AWS SM)
5. Validate all required secrets present
6. Initialize logging and connections

---

## 12. External Integrations

### 12.1 LLM Providers

| Provider | Priority | Use Case |
|----------|----------|----------|
| Ollama | 1 (Primary) | Air-gapped, on-premise |
| vLLM | 2 | High-performance local |
| Anthropic Claude | 3 | Cloud fallback |
| OpenAI GPT-4 | 4 | Cloud fallback |

### 12.2 Secrets Management

| Provider | SDK | Config Key |
|----------|-----|------------|
| HashiCorp Vault | hvac | `VAULT_ADDR`, `VAULT_TOKEN` |
| AWS Secrets Manager | boto3 | AWS credentials |
| Azure Key Vault | azure-identity | Azure credentials |
| Kubernetes Secrets | Native | In-cluster |

### 12.3 Monitoring Stack

| Tool | Purpose | Port |
|------|---------|------|
| Prometheus | Metrics | 9090 |
| Grafana | Dashboards | 3000 |
| Jaeger | Tracing | 6831 |
| Sentry | Error tracking | N/A |

### 12.4 SIEM Integration

| SIEM | Protocol | Config |
|------|----------|--------|
| Splunk | HEC | `splunk.hec_endpoint` |
| Elastic | REST | `elastic.endpoint` |
| QRadar | API | `qradar.api_endpoint` |

---

## 13. Testing Strategy

### 13.1 Test Structure

```
ai_engine/tests/
├── cloud_native/          # Cloud-native feature tests
├── security/              # Security and quantum crypto tests
├── integration/           # Integration tests
├── performance/           # Performance and load tests
├── explainability/        # AI explainability tests
└── unit/                  # Unit tests for all modules
```

### 13.2 Running Tests

```bash
# Run all tests
pytest ai_engine/tests/ -v

# Run specific test suites
pytest ai_engine/tests/cloud_native/ -v
pytest ai_engine/tests/security/ -v -k quantum
pytest ai_engine/tests/integration/ -v
pytest ai_engine/tests/performance/ -v

# Run with coverage
pytest ai_engine/tests/ --cov=ai_engine --cov-report=html
```

### 13.3 Test Coverage Requirements

- Minimum coverage: 80%
- Current coverage: 85%
- Critical paths: 95%+

---

## 14. Deployment Guide

### 14.1 Docker Deployment

```bash
cd docker
docker-compose up -d
docker-compose ps
```

### 14.2 Kubernetes Deployment

```bash
# Apply namespace and core resources
kubectl apply -f kubernetes/service-mesh/namespace.yaml
kubectl apply -f kubernetes/service-mesh/xds-server-deployment.yaml
kubectl apply -f kubernetes/container-security/admission-webhook-deployment.yaml

# Verify
kubectl get pods -n qbitel-service-mesh
```

### 14.3 Helm Deployment

```bash
helm install qbitel ./helm/qbitel \
  --namespace qbitel \
  --create-namespace \
  --values ./helm/qbitel/values-production.yaml
```

### 14.4 Air-Gapped Deployment

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Download models (while connected)
ollama pull llama3.2:8b
ollama pull mixtral:8x7b

# 3. Configure for air-gapped mode
export QBITEL_LLM_PROVIDER=ollama
export QBITEL_AIRGAPPED_MODE=true
export QBITEL_DISABLE_CLOUD_LLMS=true

# 4. Deploy
python -m ai_engine --airgapped
```

---

## 15. Design Patterns & Best Practices

### 15.1 Patterns Used

| Pattern | Where Used |
|---------|------------|
| Server Manager | `api/server.py` - Graceful shutdown |
| Dependency Injection | FastAPI `Depends()` |
| Circuit Breaker | `core/database_circuit_breaker.py` |
| Repository | Database access layer |
| Strategy | LLM provider fallback |
| Observer | Event tracking/metrics |
| Factory | Model/service creation |

### 15.2 Error Handling

```python
# Exception hierarchy
QbitelAIException (base)
├── ConfigurationException
├── ModelException
│   ├── ModelLoadException
│   └── ModelInferenceException
├── ProtocolException
│   └── DiscoveryException
└── SecurityException
    └── AuthenticationException

# Recovery strategies
class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    DEGRADE = "degrade"
    FAIL_FAST = "fail_fast"
```

### 15.3 Async Best Practices

```python
# Always use async for I/O operations
async def process_request(request: Request) -> Response:
    # Use async context managers
    async with database_manager.transaction() as session:
        result = await session.execute(query)

    # Parallel execution where possible
    results = await asyncio.gather(
        fetch_data_a(),
        fetch_data_b(),
        fetch_data_c()
    )

    return Response(data=results)
```

---

## 16. Coding Standards

### 16.1 Python Style Guide

```python
# Imports (isort ordering)
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field

from ai_engine.core.config import Config
from ai_engine.core.exceptions import QbitelAIException


# Type hints required
async def process_data(
    data: bytes,
    options: Optional[Dict[str, Any]] = None
) -> ProcessingResult:
    """Process input data with optional configuration.

    Args:
        data: Raw input bytes to process
        options: Optional processing configuration

    Returns:
        ProcessingResult with status and output

    Raises:
        ProcessingException: If processing fails
    """
    pass


# Dataclasses for data structures
@dataclass
class ProcessingResult:
    success: bool
    output: bytes
    processing_time: float
    metadata: Dict[str, Any]


# Pydantic for API schemas
class ProcessRequest(BaseModel):
    data: bytes
    format: str = Field(default="raw", description="Input format")

    class Config:
        json_schema_extra = {
            "example": {"data": b"...", "format": "hex"}
        }
```

### 16.2 Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `ProtocolDiscovery` |
| Functions | snake_case | `detect_fields()` |
| Variables | snake_case | `field_count` |
| Constants | UPPER_SNAKE | `MAX_RETRIES` |
| Private | Leading underscore | `_internal_state` |
| Protected | Leading underscore | `_process_data()` |

### 16.3 File Organization

```python
# Standard file structure
"""Module docstring describing purpose."""

from __future__ import annotations

# Standard library imports
import asyncio
import logging

# Third-party imports
from fastapi import FastAPI

# Local imports
from ai_engine.core import config

# Module-level constants
TIMEOUT = 30
MAX_RETRIES = 3

# Logger
logger = logging.getLogger(__name__)


# Classes and functions
class MyClass:
    """Class docstring."""
    pass


def my_function():
    """Function docstring."""
    pass


# Main execution (if applicable)
if __name__ == "__main__":
    pass
```

---

## 17. Troubleshooting Guide

### 17.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Connection refused | Service not running | Check `docker-compose ps` |
| 401 Unauthorized | Invalid/expired token | Refresh token or re-login |
| 429 Too Many Requests | Rate limit exceeded | Wait for reset or increase limits |
| Model load failed | Missing weights | Check `model_cache_dir` path |
| Database timeout | Pool exhausted | Increase `pool_size` |

### 17.2 Log Locations

| Log Type | Location |
|----------|----------|
| Application | `/var/log/qbitel/app.log` |
| Access | `/var/log/qbitel/access.log` |
| Error | `/var/log/qbitel/error.log` |
| Audit | `/var/log/qbitel/audit.log` |
| Container | `docker logs <container>` |

### 17.3 Health Check Endpoints

```bash
# Basic health
curl http://localhost:8000/health

# Kubernetes liveness
curl http://localhost:8000/healthz

# Kubernetes readiness (checks dependencies)
curl http://localhost:8000/readyz

# Prometheus metrics
curl http://localhost:8000/metrics
```

### 17.4 Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=true

# Run with verbose output
python -m ai_engine --log-level DEBUG
```

---

## 18. Go Services

### 18.1 Overview

The Go codebase provides three microservices for device management, policy control, and attestation:

| Service | Location | Port | Purpose |
|---------|----------|------|---------|
| Control Plane | `go/controlplane/` | 8080 | OPA policy engine, bundle management |
| Management API | `go/mgmtapi/` | 8081 | Device lifecycle, certificates, attestation |
| Device Agent | `go/agents/device-agent/` | N/A | On-device TPM agent |

### 18.2 Control Plane Service

**Location**: `go/controlplane/`

**Dependencies** (Go 1.22):
- `github.com/gin-gonic/gin` - REST API framework
- `github.com/open-policy-agent/opa` - Policy engine
- `github.com/sigstore/cosign/v2` - Bundle signing
- `go.uber.org/zap` - Structured logging

**API Endpoints**:
```
GET  /healthz              - Health check
POST /policy/validate      - Validate policy
POST /policy/evaluate      - Evaluate policy
POST /bundles              - Upload bundle
GET  /bundles              - List bundles
GET  /bundles/:id          - Get bundle
DELETE /bundles/:id        - Delete bundle
POST /bundles/:id/load     - Load bundle
GET  /stats/policy         - Policy stats
GET  /stats/bundles        - Bundle stats
```

**Key Components**:
- `PolicyEngine` - OPA-based policy evaluation with caching
- `BundleManager` - Policy bundle lifecycle (filesystem, S3, GCS, OCI storage)
- `CosignVerifier` - Signature verification for policy bundles
- `VaultClient` - HashiCorp Vault integration for key management

### 18.3 Management API Service

**Location**: `go/mgmtapi/`

**API Endpoints**:
```
# Device Management
GET    /v1/devices                    - List devices
GET    /v1/devices/:id                - Get device
PATCH  /v1/devices/:id                - Update device
POST   /v1/devices/:id/suspend        - Suspend device
POST   /v1/devices/:id/resume         - Resume device
POST   /v1/devices/:id/decommission   - Decommission device

# Enrollment
POST   /v1/devices/enrollment/sessions              - Start enrollment
GET    /v1/devices/enrollment/sessions/:id          - Get session
POST   /v1/devices/enrollment/sessions/:id/attest   - Submit attestation
POST   /v1/devices/enrollment/sessions/:id/approve  - Approve enrollment
POST   /v1/devices/enrollment/sessions/:id/complete - Complete with cert

# Certificates
GET    /v1/devices/:id/certificate        - Get certificate
POST   /v1/devices/:id/certificate/renew  - Renew certificate
POST   /v1/devices/:id/certificate/revoke - Revoke certificate
```

**Key Components**:
- `DeviceLifecycleManager` - Complete device lifecycle management
- `CertificateManager` - PKI with Root CA and Intermediate CA
- `AttestationVerifier` - TPM quote and PCR verification

### 18.4 Device Agent

**Location**: `go/agents/device-agent/`

**Purpose**: Runs on IoT/edge devices for TPM attestation

**Key Features**:
- TPM 2.0 key sealing and unsealing
- PCR-based attestation quotes
- EK/AK certificate management
- Periodic attestation (5-minute interval)
- Enrollment with control plane

**Environment Variables**:
```bash
DEVICE_ID=<device-id>
CONTROL_URL=https://control.qbitel.local
```

### 18.5 Go Coding Standards

```go
// Package organization
package policy

import (
    // Standard library
    "context"
    "sync"

    // Third-party
    "github.com/gin-gonic/gin"
    "go.uber.org/zap"

    // Internal
    "github.com/yazhsab/go/controlplane/internal/vault"
)

// Struct with configuration pattern
type Config struct {
    BundleURL     string        `json:"bundle_url"`
    PollInterval  time.Duration `json:"poll_interval"`
    RequireSigned bool          `json:"require_signed"`
}

func DefaultConfig() *Config {
    return &Config{
        PollInterval:  5 * time.Minute,
        RequireSigned: true,
    }
}
```

---

## 19. Rust Dataplane

### 19.1 Overview

The Rust codebase provides high-performance packet processing, post-quantum cryptography, and protocol adapters:

**Location**: `rust/dataplane/`

**Workspace Members**:
```
crates/
├── adapter_sdk/     - Core L7Adapter trait
├── pqc_tls/         - Post-quantum TLS (Kyber-768, Dilithium)
├── io_engine/       - High-performance bridge with HA
├── dpdk_engine/     - DPDK kernel bypass
├── dpi_engine/      - Deep packet inspection + ML
├── k8s_operator/    - Kubernetes operator
├── ai_bridge/       - Python interop (PyO3)
└── adapters/
    ├── iso8583/     - Financial protocol (PCI DSS)
    ├── modbus/      - Industrial SCADA
    ├── tn3270e/     - Mainframe terminal
    └── hl7_mllp/    - Healthcare HL7
```

### 19.2 Adapter SDK

**Core Trait** (`adapter_sdk/src/lib.rs`):
```rust
#[async_trait]
pub trait L7Adapter: Send + Sync {
    async fn to_upstream(&self, input: Bytes) -> Result<Bytes, AdapterError>;
    async fn to_client(&self, input: Bytes) -> Result<Bytes, AdapterError>;
    fn name(&self) -> &'static str;
}
```

### 19.3 PQC-TLS Crate

**Location**: `rust/dataplane/crates/pqc_tls/`

**Features**:
- Kyber-768 KEM (post-quantum key encapsulation)
- Dilithium signatures
- Hardware acceleration (AVX-512, AVX2, AES-NI)
- HSM integration via PKCS#11
- Automated key rotation with threat-level awareness
- Memory pool for crypto buffers

**Key Types**:
```rust
pub struct PqcTlsManager {
    kyber_kem: Arc<KyberKEM>,
    hsm_manager: Option<Arc<HsmPqcManager>>,
    lifecycle_manager: Arc<KeyLifecycleManager>,
    rotation_manager: Arc<QuantumSafeRotationManager>,
}

// Kyber parameters (Kyber-768)
pub const KYBER_PUBLIC_KEY_SIZE: usize = 1184;
pub const KYBER_PRIVATE_KEY_SIZE: usize = 768;
pub const KYBER_CIPHERTEXT_SIZE: usize = 1088;
pub const KYBER_SHARED_SECRET_SIZE: usize = 32;
```

**Dependencies**:
- `pqcrypto-kyber`, `pqcrypto-dilithium`, `oqs`
- `tokio-openssl` for TLS
- `pkcs11` for HSM
- `rayon` for parallelism
- `wide` for SIMD

### 19.4 I/O Engine

**Location**: `rust/dataplane/crates/io_engine/`

**Features**:
- Bidirectional protocol transformation bridge
- Active-Standby HA with automatic failover
- Connection pooling with lifecycle management
- Circuit breaker pattern
- Lock-free high-frequency metrics
- Distributed tracing (OpenTelemetry)

**Architecture**:
```
Client → Reader → Adapter.to_upstream() → Channel → Writer → Upstream
Upstream → Reader → Adapter.to_client() → Channel → Writer → Client
```

### 19.5 Protocol Adapters

| Adapter | Protocol | Features |
|---------|----------|----------|
| **ISO-8583** | Financial/Payment | PCI DSS masking, field encryption, MAC, routing |
| **Modbus** | Industrial/SCADA | TCP/RTU, CRC validation, register access control |
| **TN3270E** | Mainframe Terminal | Screen scraping, session management |
| **HL7 MLLP** | Healthcare | MLLP framing, field mapping, validation |

### 19.6 DPI Engine

**Location**: `rust/dataplane/crates/dpi_engine/`

**ML Integration**:
- PyTorch via `tch` bindings
- ONNX Runtime via `ort`
- Candle framework
- Hyperscan for pattern matching

**Protocol Classification**:
- HTTP, HTTPS, SSH, DNS, SIP, RTP
- BitTorrent, Skype, WhatsApp, Telegram
- Zoom, Teams, Slack, Netflix, YouTube

### 19.7 Rust Coding Standards

```rust
// Module organization
pub mod kyber;
pub mod hsm;
pub mod client;
pub mod server;

// Error handling with thiserror
#[derive(Debug, thiserror::Error)]
pub enum TlsError {
    #[error("OpenSSL error: {0}")]
    Openssl(String),
    #[error("I/O error: {0}")]
    Io(String),
    #[error("Policy violation: {0}")]
    Policy(String),
}

// Async trait pattern
#[async_trait]
pub trait L7Adapter: Send + Sync {
    async fn to_upstream(&self, input: Bytes) -> Result<Bytes, AdapterError>;
}

// Builder pattern
impl Iso8583AdapterBuilder {
    pub fn with_parser(mut self, parser: Iso8583Parser) -> Self {
        self.parser = Some(parser);
        self
    }
    pub fn build(self) -> Result<Iso8583Adapter, Iso8583Error> { ... }
}
```

---

## 20. Frontend Console

### 20.1 Overview

**Location**: `ui/console/`

**Technology Stack**:
- React 18.2 with TypeScript 5.3
- Material-UI 5.14 (MUI)
- Vite 5.0 for bundling
- OIDC authentication
- WebSocket for real-time updates

### 20.2 Directory Structure

```
ui/console/
├── src/
│   ├── main.tsx              - Entry point
│   ├── App.tsx               - Main shell
│   ├── api/
│   │   ├── devices.ts        - Device API client
│   │   ├── marketplace.ts    - Marketplace API
│   │   └── enhancedApiClient.ts - Cached API with WebSocket
│   ├── auth/
│   │   └── oidc.ts           - OIDC service
│   ├── components/
│   │   ├── Dashboard.tsx
│   │   ├── DeviceManagement.tsx
│   │   ├── PolicyManagement.tsx
│   │   ├── SecurityMonitoring.tsx
│   │   ├── ProtocolVisualization.tsx
│   │   ├── AIModelMonitoring.tsx
│   │   ├── ThreatIntelligence.tsx
│   │   └── marketplace/      - Marketplace components
│   ├── routes/
│   │   └── AppRoutes.tsx     - RBAC routing
│   ├── theme/
│   │   └── EnterpriseTheme.ts - 5 theme variants
│   └── types/
│       ├── auth.ts
│       ├── device.ts
│       └── marketplace.ts
├── package.json
├── vite.config.ts
└── tsconfig.json
```

### 20.3 Key Components

| Component | Purpose |
|-----------|---------|
| `Dashboard` | Main metrics overview |
| `DeviceManagement` | Device CRUD & status |
| `PolicyManagement` | Policy configuration |
| `SecurityMonitoring` | Security alerts |
| `ProtocolVisualization` | D3/Plotly protocol viz |
| `AIModelMonitoring` | ML model metrics |
| `ThreatIntelligence` | IOC management |
| `MarketplaceHome` | Protocol marketplace |
| `ProtocolCopilotChat` | AI chat interface |

### 20.4 Authentication

**OIDC Configuration**:
```typescript
{
  authority: 'https://auth.qbitel.local',
  clientId: 'qbitel-console',
  scope: 'openid profile email roles permissions organization',
  automaticSilentRenew: true,
}
```

**User Roles**:
- `admin`, `system_admin`, `org_admin`
- `security_analyst`, `operator`, `viewer`

### 20.5 Routes with RBAC

| Route | Component | Required Permission |
|-------|-----------|---------------------|
| `/dashboard` | Dashboard | `dashboard:read` |
| `/devices` | DeviceManagement | `device:read` |
| `/policies` | PolicyManagement | `policy:read` |
| `/security` | SecurityMonitoring | `security:read` |
| `/protocols` | ProtocolVisualization | `protocol:read` (enterprise) |
| `/ai-models` | AIModelMonitoring | `ai:read` (enterprise) |
| `/marketplace` | MarketplaceHome | `marketplace:read` |
| `/settings` | SystemSettings | admin role |

### 20.6 Theme System

**5 Theme Variants**:
1. Light - Standard light mode
2. Dark - Dark mode
3. High Contrast - Accessibility
4. Blue-Grey - Professional
5. Enterprise - Custom branding

```typescript
const useTheme = () => {
  const { theme, setTheme } = useContext(ThemeContext);
  return { theme, setTheme };
};
```

### 20.7 WebSocket Real-Time Updates

**Message Types**:
```typescript
// Protocol
PROTOCOL_DISCOVERED, PROTOCOL_UPDATED, PROTOCOL_METRICS

// AI Models
MODEL_METRICS, MODEL_ALERT, MODEL_TRAINING_COMPLETE

// Security
THREAT_DETECTED, THREAT_RESOLVED, SECURITY_ALERT

// System
HEARTBEAT, SYSTEM_STATUS, NOTIFICATION
```

### 20.8 Build Configuration

**Vite Config**:
```typescript
{
  plugins: [react()],
  resolve: {
    alias: {
      '@': './src',
      '@/components': './src/components',
      '@/api': './src/api',
    }
  },
  server: {
    port: 3000,
    proxy: {
      '/api': { target: 'https://api.qbitel.local' }
    }
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: { vendor: ['react', 'react-dom'], mui: ['@mui/material'] }
      }
    }
  }
}
```

### 20.9 TypeScript/React Coding Standards

```typescript
// Component pattern
interface Props {
  deviceId: string;
  onUpdate?: (device: Device) => void;
}

export const DeviceCard: React.FC<Props> = ({ deviceId, onUpdate }) => {
  const [device, setDevice] = useState<Device | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDevice();
  }, [deviceId]);

  return (
    <Card>
      {loading ? <CircularProgress /> : <DeviceDetails device={device} />}
    </Card>
  );
};

// API client pattern
export class DeviceApiClient {
  private baseUrl: string;
  private authService: OidcAuthService;

  async getDevice(id: string): Promise<Device> {
    const response = await fetch(`${this.baseUrl}/devices/${id}`, {
      headers: this.getAuthHeaders(),
    });
    return response.json();
  }
}
```

---

## Appendix A: Quick Reference Cards

### A.1 API Quick Reference

```
Base URL: http://localhost:8000/api/v1

# Authentication
POST /auth/login        → {access_token, refresh_token}
POST /auth/refresh      → {access_token}
POST /auth/logout       → 204 No Content

# Copilot
POST /copilot/query     → {response, confidence, suggestions}
WS   /copilot/ws        → Real-time streaming

# Protocol Discovery
POST /discover          → {protocol_type, confidence, grammar}
POST /fields/detect     → {detected_fields, confidence}
POST /anomalies/detect  → {score, is_anomalous, explanation}

# Security
POST /security/analyze  → {threat_score, recommended_action}
POST /security/respond  → {status, execution_result}
GET  /security/incidents → [{id, severity, status}]
```

### A.2 Configuration Quick Reference

```yaml
# Minimum production config
environment: production
database:
  host: ${DATABASE_HOST}
  password: ${DATABASE_PASSWORD}
  ssl_mode: require
  pool_size: 50

redis:
  host: ${REDIS_HOST}
  password: ${REDIS_PASSWORD}

security:
  jwt_secret: ${JWT_SECRET}
  encryption_key: ${ENCRYPTION_KEY}

llm:
  provider: ollama
  endpoint: http://localhost:11434
```

### A.3 Docker Quick Reference

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f ai-engine

# Stop all
docker-compose down

# Clean rebuild
docker-compose down -v && docker-compose up -d --build
```

---

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.1.0 | 2025-11-22 | Added Go, Rust, Frontend sections for complete multi-language coverage |
| 2.0.0 | 2025-11-22 | Initial knowledge base with Python codebase |

---

## Appendix C: Contact & Support

- **GitHub Issues**: https://github.com/yazhsab/qbitel-bridge/issues
- **Documentation**: `/docs/` directory
- **Enterprise Support**: enterprise@qbitel.com

---

**End of Knowledge Base**

*This document should be updated whenever significant changes are made to the codebase architecture, APIs, or infrastructure.*
