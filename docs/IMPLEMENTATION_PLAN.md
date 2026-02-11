# QBITEL Implementation Plan

## Executive Summary

This document outlines a phased implementation plan to focus, harden, and optimize the QBITEL platform based on the comprehensive architecture review. The plan spans 12 weeks across 4 phases.

---

## Phase 1: Security Hardening (Weeks 1-2)

### Goal: Address critical security gaps before any customer deployment

---

### 1.1 API Rate Limiting

**Priority:** CRITICAL
**Effort:** 2 days
**Owner:** Backend Team

#### Implementation Steps:

1. **Install slowapi package**
   ```bash
   pip install slowapi
   ```

2. **Create rate limiting middleware** at `ai_engine/api/middleware/rate_limiter.py`:
   ```python
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   from slowapi.errors import RateLimitExceeded
   from fastapi import Request

   limiter = Limiter(key_func=get_remote_address)

   # Rate limit configurations by endpoint type
   RATE_LIMITS = {
       "discovery": "10/minute",      # Heavy compute
       "detection": "30/minute",      # Medium compute
       "copilot": "60/minute",        # LLM calls
       "health": "300/minute",        # Lightweight
       "default": "100/minute"        # Standard endpoints
   }
   ```

3. **Apply to REST API** in `ai_engine/api/rest.py`:
   ```python
   from ai_engine.api.middleware.rate_limiter import limiter, RATE_LIMITS

   app.state.limiter = limiter
   app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

   @app.post("/api/v1/discover/protocol")
   @limiter.limit(RATE_LIMITS["discovery"])
   async def discover_protocol(request: Request, ...):
       ...
   ```

4. **Add Redis backend for distributed rate limiting**:
   ```python
   from slowapi import Limiter
   from slowapi.util import get_remote_address
   import redis

   redis_client = redis.from_url(settings.REDIS_URL)
   limiter = Limiter(
       key_func=get_remote_address,
       storage_uri=settings.REDIS_URL
   )
   ```

#### Acceptance Criteria:
- [ ] All endpoints have rate limits applied
- [ ] Rate limit headers returned (X-RateLimit-Limit, X-RateLimit-Remaining)
- [ ] 429 Too Many Requests returned when exceeded
- [ ] Rate limits persist across restarts (Redis backend)
- [ ] Prometheus metrics for rate limit hits

---

### 1.2 Input Size Validation

**Priority:** CRITICAL
**Effort:** 1 day
**Owner:** Backend Team

#### Implementation Steps:

1. **Create validation constants** at `ai_engine/core/constants.py`:
   ```python
   # Input size limits
   MAX_PROTOCOL_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB
   MAX_FIELD_DETECTION_SIZE = 1 * 1024 * 1024    # 1 MB
   MAX_BATCH_SIZE = 100                           # Max messages per batch
   MAX_COPILOT_QUERY_LENGTH = 10000              # Characters
   MAX_FILE_UPLOAD_SIZE = 50 * 1024 * 1024       # 50 MB
   ```

2. **Create validation decorator** at `ai_engine/api/middleware/validators.py`:
   ```python
   from functools import wraps
   from fastapi import HTTPException, status
   from ai_engine.core.constants import MAX_PROTOCOL_MESSAGE_SIZE

   def validate_input_size(max_size: int):
       def decorator(func):
           @wraps(func)
           async def wrapper(*args, **kwargs):
               request = kwargs.get('request') or args[0]
               body = await request.body()
               if len(body) > max_size:
                   raise HTTPException(
                       status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                       detail=f"Request body exceeds maximum size of {max_size} bytes"
                   )
               return await func(*args, **kwargs)
           return wrapper
       return decorator
   ```

3. **Apply to endpoints**:
   ```python
   @app.post("/api/v1/discover/protocol")
   @validate_input_size(MAX_PROTOCOL_MESSAGE_SIZE)
   async def discover_protocol(...):
       ...
   ```

4. **Add global request size limit** in FastAPI:
   ```python
   from fastapi import FastAPI
   from starlette.middleware.trustedhost import TrustedHostMiddleware

   app = FastAPI()

   # Limit request body size globally
   @app.middleware("http")
   async def limit_request_size(request: Request, call_next):
       content_length = request.headers.get("content-length")
       if content_length and int(content_length) > MAX_FILE_UPLOAD_SIZE:
           return JSONResponse(
               status_code=413,
               content={"detail": "Request too large"}
           )
       return await call_next(request)
   ```

#### Acceptance Criteria:
- [ ] All endpoints validate input size
- [ ] Clear error messages returned for oversized requests
- [ ] Unit tests for boundary conditions
- [ ] Documentation updated with size limits

---

### 1.3 Standardized Error Handling

**Priority:** HIGH
**Effort:** 3 days
**Owner:** Backend Team

#### Implementation Steps:

1. **Create exception hierarchy** at `ai_engine/core/exceptions.py`:
   ```python
   from typing import Optional, Dict, Any
   from enum import Enum

   class ErrorCode(Enum):
       # Discovery errors (1000-1999)
       DISCOVERY_FAILED = 1000
       DISCOVERY_TIMEOUT = 1001
       INVALID_PROTOCOL_DATA = 1002
       GRAMMAR_LEARNING_FAILED = 1003
       PARSER_GENERATION_FAILED = 1004

       # Detection errors (2000-2999)
       FIELD_DETECTION_FAILED = 2000
       MODEL_NOT_LOADED = 2001
       INVALID_MESSAGE_FORMAT = 2002

       # LLM errors (3000-3999)
       LLM_UNAVAILABLE = 3000
       LLM_TIMEOUT = 3001
       LLM_RATE_LIMITED = 3002
       LLM_CONTEXT_TOO_LONG = 3003

       # Authentication errors (4000-4999)
       UNAUTHORIZED = 4000
       INVALID_TOKEN = 4001
       TOKEN_EXPIRED = 4002
       INSUFFICIENT_PERMISSIONS = 4003

       # Validation errors (5000-5999)
       VALIDATION_ERROR = 5000
       INVALID_INPUT = 5001
       MISSING_REQUIRED_FIELD = 5002

       # System errors (9000-9999)
       INTERNAL_ERROR = 9000
       DATABASE_ERROR = 9001
       CACHE_ERROR = 9002
       EXTERNAL_SERVICE_ERROR = 9003


   class QbitelError(Exception):
       """Base exception for all QBITEL errors."""

       def __init__(
           self,
           message: str,
           error_code: ErrorCode,
           details: Optional[Dict[str, Any]] = None,
           cause: Optional[Exception] = None
       ):
           super().__init__(message)
           self.message = message
           self.error_code = error_code
           self.details = details or {}
           self.cause = cause

       def to_dict(self) -> Dict[str, Any]:
           return {
               "error_code": self.error_code.value,
               "error_name": self.error_code.name,
               "message": self.message,
               "details": self.details
           }


   class DiscoveryError(QbitelError):
       """Errors during protocol discovery."""
       pass


   class DetectionError(QbitelError):
       """Errors during field detection."""
       pass


   class LLMError(QbitelError):
       """Errors from LLM services."""
       pass


   class AuthenticationError(QbitelError):
       """Authentication and authorization errors."""
       pass


   class ValidationError(QbitelError):
       """Input validation errors."""
       pass
   ```

2. **Create global exception handler** at `ai_engine/api/middleware/error_handler.py`:
   ```python
   from fastapi import Request, status
   from fastapi.responses import JSONResponse
   from ai_engine.core.exceptions import QbitelError, ErrorCode
   import logging
   import traceback

   logger = logging.getLogger(__name__)

   async def qbitel_exception_handler(request: Request, exc: QbitelError):
       """Handle all QBITEL-specific exceptions."""
       logger.error(
           f"QbitelError: {exc.error_code.name} - {exc.message}",
           extra={
               "error_code": exc.error_code.value,
               "details": exc.details,
               "path": request.url.path
           }
       )

       # Map error codes to HTTP status codes
       status_map = {
           ErrorCode.UNAUTHORIZED: status.HTTP_401_UNAUTHORIZED,
           ErrorCode.INVALID_TOKEN: status.HTTP_401_UNAUTHORIZED,
           ErrorCode.INSUFFICIENT_PERMISSIONS: status.HTTP_403_FORBIDDEN,
           ErrorCode.VALIDATION_ERROR: status.HTTP_400_BAD_REQUEST,
           ErrorCode.INVALID_INPUT: status.HTTP_400_BAD_REQUEST,
           ErrorCode.LLM_RATE_LIMITED: status.HTTP_429_TOO_MANY_REQUESTS,
       }

       http_status = status_map.get(
           exc.error_code,
           status.HTTP_500_INTERNAL_SERVER_ERROR
       )

       return JSONResponse(
           status_code=http_status,
           content=exc.to_dict()
       )


   async def generic_exception_handler(request: Request, exc: Exception):
       """Handle unexpected exceptions."""
       logger.exception(
           f"Unhandled exception: {str(exc)}",
           extra={"path": request.url.path}
       )

       return JSONResponse(
           status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
           content={
               "error_code": ErrorCode.INTERNAL_ERROR.value,
               "error_name": "INTERNAL_ERROR",
               "message": "An unexpected error occurred",
               "details": {}
           }
       )
   ```

3. **Register handlers in FastAPI**:
   ```python
   from ai_engine.core.exceptions import QbitelError
   from ai_engine.api.middleware.error_handler import (
       qbitel_exception_handler,
       generic_exception_handler
   )

   app.add_exception_handler(QbitelError, qbitel_exception_handler)
   app.add_exception_handler(Exception, generic_exception_handler)
   ```

4. **Refactor existing code** to use new exceptions:
   ```python
   # Before (inconsistent)
   raise ValueError("Invalid protocol data")

   # After (standardized)
   from ai_engine.core.exceptions import DiscoveryError, ErrorCode

   raise DiscoveryError(
       message="Invalid protocol data: message is empty",
       error_code=ErrorCode.INVALID_PROTOCOL_DATA,
       details={"received_length": 0, "minimum_length": 1}
   )
   ```

#### Files to Refactor:
- [ ] `ai_engine/discovery/protocol_discovery_orchestrator.py`
- [ ] `ai_engine/detection/field_detector.py`
- [ ] `ai_engine/llm/unified_llm_service.py`
- [ ] `ai_engine/copilot/protocol_copilot.py`
- [ ] `ai_engine/legacy/service.py`

#### Acceptance Criteria:
- [ ] All custom exceptions inherit from QbitelError
- [ ] Error codes documented in API specification
- [ ] No generic `ValueError` or `Exception` raises in business logic
- [ ] All errors include correlation_id for tracing
- [ ] Unit tests for each error type

---

## Phase 2: Architecture Simplification (Weeks 3-5)

### Goal: Reduce complexity by removing unused components and splitting large modules

---

### 2.1 Remove Unused LLM Providers

**Priority:** HIGH
**Effort:** 2 days
**Owner:** Backend Team

#### Current State:
- 5 providers configured: Ollama, vLLM, LocalAI, OpenAI, Anthropic
- Only Ollama and Anthropic have active usage

#### Implementation Steps:

1. **Audit current LLM usage**:
   ```bash
   grep -r "vllm\|localai" ai_engine/ --include="*.py"
   ```

2. **Remove vLLM integration**:
   - Delete `ai_engine/llm/providers/vllm_provider.py` (if exists)
   - Remove from `UnifiedLLMService` provider list
   - Remove from `pyproject.toml` dependencies

3. **Remove LocalAI integration**:
   - Delete `ai_engine/llm/providers/localai_provider.py` (if exists)
   - Remove from configuration files
   - Remove from dependencies

4. **Simplify UnifiedLLMService**:
   ```python
   # Before: Complex 5-provider fallback
   PROVIDERS = ["ollama", "vllm", "localai", "openai", "anthropic"]

   # After: Simplified 2-provider fallback
   class LLMProvider(Enum):
       OLLAMA = "ollama"      # Primary (air-gapped)
       ANTHROPIC = "anthropic" # Cloud fallback

   class SimplifiedLLMService:
       def __init__(self):
           self.primary = OllamaProvider()
           self.fallback = AnthropicProvider()

       async def generate(self, prompt: str, **kwargs) -> str:
           try:
               return await self.primary.generate(prompt, **kwargs)
           except LLMError:
               if self.allow_cloud_fallback:
                   return await self.fallback.generate(prompt, **kwargs)
               raise
   ```

5. **Update configuration**:
   ```yaml
   # config/qbitel.yaml
   llm:
     primary_provider: ollama
     fallback_provider: anthropic  # Optional, disabled in air-gapped
     allow_cloud_fallback: false   # Default to air-gapped mode

     ollama:
       base_url: "http://localhost:11434"
       model: "llama3.2:70b"
       timeout: 120

     anthropic:
       model: "claude-sonnet-4-20250514"
       max_tokens: 4096
   ```

#### Acceptance Criteria:
- [ ] Only Ollama and Anthropic providers remain
- [ ] All tests pass with reduced providers
- [ ] Configuration simplified
- [ ] Documentation updated

---

### 2.2 Split Database Models

**Priority:** MEDIUM
**Effort:** 3 days
**Owner:** Backend Team

#### Current State:
`ai_engine/models/database.py` contains 500+ lines with mixed concerns:
- User authentication models
- Audit logging models
- OAuth/SAML providers
- API key management

#### Target Structure:
```
ai_engine/models/
├── __init__.py           # Re-export all models
├── base.py               # Base model class, mixins
├── auth/
│   ├── __init__.py
│   ├── user.py           # User, UserRole
│   ├── session.py        # UserSession
│   ├── api_key.py        # APIKey, APIKeyStatus
│   └── mfa.py            # MFA configurations
├── providers/
│   ├── __init__.py
│   ├── oauth.py          # OAuthProvider
│   └── saml.py           # SAMLProvider
├── audit/
│   ├── __init__.py
│   └── audit_log.py      # AuditLog, AuditAction
└── protocol/
    ├── __init__.py
    ├── discovery.py      # ProtocolProfile, DiscoveryResult
    └── analysis.py       # FieldDetection, AnomalyResult
```

#### Implementation Steps:

1. **Create base module** `ai_engine/models/base.py`:
   ```python
   from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
   from sqlalchemy import DateTime, func
   from datetime import datetime
   from uuid import UUID, uuid4

   class Base(DeclarativeBase):
       pass

   class TimestampMixin:
       created_at: Mapped[datetime] = mapped_column(
           DateTime(timezone=True),
           server_default=func.now()
       )
       updated_at: Mapped[datetime] = mapped_column(
           DateTime(timezone=True),
           server_default=func.now(),
           onupdate=func.now()
       )

   class SoftDeleteMixin:
       deleted_at: Mapped[datetime | None] = mapped_column(
           DateTime(timezone=True),
           nullable=True
       )

       @property
       def is_deleted(self) -> bool:
           return self.deleted_at is not None

   class UUIDPrimaryKeyMixin:
       id: Mapped[UUID] = mapped_column(
           primary_key=True,
           default=uuid4
       )
   ```

2. **Split user models** `ai_engine/models/auth/user.py`:
   ```python
   from enum import Enum
   from sqlalchemy import String, Boolean, Enum as SQLEnum
   from sqlalchemy.orm import Mapped, mapped_column, relationship
   from ai_engine.models.base import Base, TimestampMixin, SoftDeleteMixin, UUIDPrimaryKeyMixin

   class UserRole(str, Enum):
       ADMINISTRATOR = "administrator"
       SECURITY_ANALYST = "security_analyst"
       OPERATOR = "operator"
       VIEWER = "viewer"
       API_USER = "api_user"

   class User(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
       __tablename__ = "users"

       username: Mapped[str] = mapped_column(String(255), unique=True, index=True)
       email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
       password_hash: Mapped[str | None] = mapped_column(String(255))
       role: Mapped[UserRole] = mapped_column(SQLEnum(UserRole), index=True)
       is_active: Mapped[bool] = mapped_column(Boolean, default=True)

       # Relationships
       sessions = relationship("UserSession", back_populates="user")
       api_keys = relationship("APIKey", back_populates="user")
       audit_logs = relationship("AuditLog", back_populates="user")
   ```

3. **Create re-export module** `ai_engine/models/__init__.py`:
   ```python
   from ai_engine.models.base import Base, TimestampMixin, SoftDeleteMixin
   from ai_engine.models.auth.user import User, UserRole
   from ai_engine.models.auth.session import UserSession
   from ai_engine.models.auth.api_key import APIKey, APIKeyStatus
   from ai_engine.models.providers.oauth import OAuthProvider
   from ai_engine.models.providers.saml import SAMLProvider
   from ai_engine.models.audit.audit_log import AuditLog, AuditAction

   __all__ = [
       "Base",
       "User", "UserRole",
       "UserSession",
       "APIKey", "APIKeyStatus",
       "OAuthProvider", "SAMLProvider",
       "AuditLog", "AuditAction",
   ]
   ```

4. **Add missing indexes** for audit_logs:
   ```python
   class AuditLog(Base, UUIDPrimaryKeyMixin):
       __tablename__ = "audit_logs"
       __table_args__ = (
           Index("ix_audit_logs_timestamp_action", "timestamp", "action_type"),
           Index("ix_audit_logs_user_timestamp", "user_id", "timestamp"),
           Index("ix_audit_logs_resource", "resource_type", "resource_id"),
       )
   ```

5. **Generate Alembic migration**:
   ```bash
   alembic revision --autogenerate -m "add_audit_log_indexes"
   alembic upgrade head
   ```

#### Acceptance Criteria:
- [ ] All models split into logical modules
- [ ] Backward-compatible imports via `__init__.py`
- [ ] Missing indexes added
- [ ] All existing tests pass
- [ ] Alembic migration generated and tested

---

### 2.3 Refactor Legacy Whisperer (17K LOC → 4 Services)

**Priority:** HIGH
**Effort:** 5 days
**Owner:** Backend Team

#### Current State:
`ai_engine/legacy/` is a monolithic 17,000+ LOC module handling:
- COBOL code analysis
- Copybook parsing
- Code generation (REST APIs, SDKs)
- Transaction flow mapping
- Modernization recommendations

#### Target Architecture:
```
ai_engine/legacy/
├── __init__.py
├── analyzer/                    # COBOL Analysis Service
│   ├── __init__.py
│   ├── service.py              # COBOLAnalyzerService
│   ├── parser.py               # COBOL syntax parser
│   ├── copybook_parser.py      # Copybook structure extraction
│   ├── data_flow.py            # Data flow analysis
│   └── models.py               # COBOLProgram, Copybook, DataItem
├── mapper/                      # Transaction Mapping Service
│   ├── __init__.py
│   ├── service.py              # TransactionMapperService
│   ├── flow_analyzer.py        # Transaction flow detection
│   ├── dependency_graph.py     # Program dependencies
│   └── models.py               # Transaction, Flow, Dependency
├── generator/                   # Code Generation Service
│   ├── __init__.py
│   ├── service.py              # CodeGeneratorService
│   ├── rest_generator.py       # REST API generation
│   ├── sdk_generator.py        # SDK generation (6 languages)
│   ├── openapi_generator.py    # OpenAPI spec generation
│   └── templates/              # Jinja2 templates
└── orchestrator/               # Orchestration Layer
    ├── __init__.py
    ├── service.py              # LegacyWhispererOrchestrator
    └── workflow.py             # Modernization workflow
```

#### Implementation Steps:

1. **Extract COBOL Analyzer** `ai_engine/legacy/analyzer/service.py`:
   ```python
   from dataclasses import dataclass
   from typing import List, Optional
   from ai_engine.legacy.analyzer.models import COBOLProgram, Copybook
   from ai_engine.legacy.analyzer.parser import COBOLParser
   from ai_engine.legacy.analyzer.copybook_parser import CopybookParser

   @dataclass
   class AnalysisResult:
       program: COBOLProgram
       copybooks: List[Copybook]
       data_items: List[DataItem]
       procedures: List[Procedure]
       complexity_score: float
       analysis_time_ms: int

   class COBOLAnalyzerService:
       """Analyzes COBOL source code and copybooks."""

       def __init__(self):
           self.cobol_parser = COBOLParser()
           self.copybook_parser = CopybookParser()

       async def analyze(
           self,
           source_code: str,
           copybooks: Optional[List[str]] = None
       ) -> AnalysisResult:
           """Analyze COBOL program and extract structure."""
           program = self.cobol_parser.parse(source_code)

           parsed_copybooks = []
           if copybooks:
               for copybook in copybooks:
                   parsed_copybooks.append(
                       self.copybook_parser.parse(copybook)
                   )

           return AnalysisResult(
               program=program,
               copybooks=parsed_copybooks,
               data_items=program.data_division.items,
               procedures=program.procedure_division.sections,
               complexity_score=self._calculate_complexity(program),
               analysis_time_ms=...
           )
   ```

2. **Extract Transaction Mapper** `ai_engine/legacy/mapper/service.py`:
   ```python
   class TransactionMapperService:
       """Maps COBOL transactions to modern API operations."""

       async def map_transactions(
           self,
           analysis: AnalysisResult
       ) -> List[Transaction]:
           """Identify and map CICS/IMS transactions."""
           transactions = []

           for procedure in analysis.procedures:
               if self._is_transaction_entry(procedure):
                   transaction = Transaction(
                       name=procedure.name,
                       entry_point=procedure,
                       inputs=self._extract_inputs(procedure),
                       outputs=self._extract_outputs(procedure),
                       side_effects=self._detect_side_effects(procedure)
                   )
                   transactions.append(transaction)

           return transactions
   ```

3. **Extract Code Generator** `ai_engine/legacy/generator/service.py`:
   ```python
   class CodeGeneratorService:
       """Generates modern code from COBOL analysis."""

       def __init__(self):
           self.rest_generator = RESTAPIGenerator()
           self.sdk_generator = SDKGenerator()
           self.openapi_generator = OpenAPIGenerator()

       async def generate_rest_api(
           self,
           transactions: List[Transaction],
           config: GenerationConfig
       ) -> GeneratedAPI:
           """Generate REST API from transactions."""
           openapi_spec = self.openapi_generator.generate(transactions)

           return GeneratedAPI(
               openapi_spec=openapi_spec,
               handlers=self.rest_generator.generate_handlers(transactions),
               models=self.rest_generator.generate_models(transactions)
           )

       async def generate_sdk(
           self,
           openapi_spec: OpenAPISpec,
           language: SDKLanguage
       ) -> GeneratedSDK:
           """Generate SDK in specified language."""
           return self.sdk_generator.generate(openapi_spec, language)
   ```

4. **Create orchestrator** `ai_engine/legacy/orchestrator/service.py`:
   ```python
   class LegacyWhispererOrchestrator:
       """Orchestrates the complete modernization workflow."""

       def __init__(
           self,
           analyzer: COBOLAnalyzerService,
           mapper: TransactionMapperService,
           generator: CodeGeneratorService
       ):
           self.analyzer = analyzer
           self.mapper = mapper
           self.generator = generator

       async def modernize(
           self,
           request: ModernizationRequest
       ) -> ModernizationResult:
           """Execute full modernization pipeline."""

           # Phase 1: Analyze
           analysis = await self.analyzer.analyze(
               request.source_code,
               request.copybooks
           )

           # Phase 2: Map transactions
           transactions = await self.mapper.map_transactions(analysis)

           # Phase 3: Generate code
           api = await self.generator.generate_rest_api(
               transactions,
               request.config
           )

           # Phase 4: Generate SDKs (if requested)
           sdks = {}
           for language in request.sdk_languages:
               sdks[language] = await self.generator.generate_sdk(
                   api.openapi_spec,
                   language
               )

           return ModernizationResult(
               analysis=analysis,
               transactions=transactions,
               api=api,
               sdks=sdks
           )
   ```

5. **Maintain backward compatibility**:
   ```python
   # ai_engine/legacy/__init__.py
   from ai_engine.legacy.orchestrator.service import LegacyWhispererOrchestrator
   from ai_engine.legacy.analyzer.service import COBOLAnalyzerService
   from ai_engine.legacy.mapper.service import TransactionMapperService
   from ai_engine.legacy.generator.service import CodeGeneratorService

   # Backward-compatible alias
   LegacyWhisperer = LegacyWhispererOrchestrator

   __all__ = [
       "LegacyWhisperer",
       "LegacyWhispererOrchestrator",
       "COBOLAnalyzerService",
       "TransactionMapperService",
       "CodeGeneratorService"
   ]
   ```

#### Acceptance Criteria:
- [ ] Legacy module split into 4 services
- [ ] Each service independently testable
- [ ] Backward-compatible imports maintained
- [ ] All 17K LOC accounted for
- [ ] Test coverage maintained or improved

---

## Phase 3: Focus & Prioritization (Weeks 6-8)

### Goal: Remove non-essential features and focus on core value proposition

---

### 3.1 Disable Non-Core Domain Modules

**Priority:** MEDIUM
**Effort:** 2 days
**Owner:** Product Team

#### Modules to Disable (Not Delete):

| Module | Path | Action | Rationale |
|--------|------|--------|-----------|
| Aviation | `ai_engine/domains/aviation/` | Disable | No active customers |
| Automotive | `ai_engine/domains/automotive/` | Disable | V2X is niche |
| Healthcare | `ai_engine/domains/healthcare/` | Keep (if banking focus) OR Disable | Evaluate customer pipeline |
| Marketplace | `ai_engine/marketplace/` | Disable | Needs scale first |

#### Implementation:

1. **Create feature flags** `ai_engine/core/feature_flags.py`:
   ```python
   from enum import Enum
   from pydantic_settings import BaseSettings

   class FeatureFlags(BaseSettings):
       # Domain modules
       ENABLE_AVIATION_DOMAIN: bool = False
       ENABLE_AUTOMOTIVE_DOMAIN: bool = False
       ENABLE_HEALTHCARE_DOMAIN: bool = True  # Keep for regulated industries

       # Features
       ENABLE_MARKETPLACE: bool = False
       ENABLE_PROTOCOL_MARKETPLACE: bool = False

       # Experimental
       ENABLE_QUANTUM_CRYPTO: bool = True
       ENABLE_AGENTIC_SECURITY: bool = True

       class Config:
           env_prefix = "QBITEL_FEATURE_"

   feature_flags = FeatureFlags()
   ```

2. **Guard domain imports**:
   ```python
   # ai_engine/domains/__init__.py
   from ai_engine.core.feature_flags import feature_flags

   if feature_flags.ENABLE_AVIATION_DOMAIN:
       from ai_engine.domains.aviation import *

   if feature_flags.ENABLE_AUTOMOTIVE_DOMAIN:
       from ai_engine.domains.automotive import *
   ```

3. **Guard API routes**:
   ```python
   from ai_engine.core.feature_flags import feature_flags

   if feature_flags.ENABLE_MARKETPLACE:
       app.include_router(marketplace_router, prefix="/api/v1/marketplace")
   ```

#### Acceptance Criteria:
- [ ] Feature flags control all optional modules
- [ ] Disabled modules don't load (reduce memory footprint)
- [ ] Easy to re-enable via environment variable
- [ ] No code deleted (preserve for future use)

---

### 3.2 Simplify Discovery Pipeline (7 → 5 Phases)

**Priority:** MEDIUM
**Effort:** 3 days
**Owner:** ML Team

#### Current Pipeline:
1. INITIALIZATION
2. STATISTICAL_ANALYSIS
3. CLASSIFICATION
4. GRAMMAR_LEARNING
5. PARSER_GENERATION
6. VALIDATION
7. COMPLETION

#### Simplified Pipeline:
1. **ANALYSIS** (merged: INITIALIZATION + STATISTICAL_ANALYSIS)
2. **CLASSIFICATION**
3. **LEARNING** (merged: GRAMMAR_LEARNING + PARSER_GENERATION)
4. **VALIDATION**
5. **COMPLETION** (minimal, just cleanup)

#### Implementation:

1. **Update phase enum**:
   ```python
   class DiscoveryPhase(Enum):
       ANALYSIS = "analysis"           # Stats + initialization
       CLASSIFICATION = "classification"
       LEARNING = "learning"           # Grammar + parser
       VALIDATION = "validation"
       COMPLETION = "completion"
   ```

2. **Merge phases in orchestrator**:
   ```python
   async def _execute_analysis_phase(self, messages: List[bytes]) -> AnalysisResult:
       """Combined initialization and statistical analysis."""
       # Initialize
       session_id = str(uuid4())

       # Statistical analysis
       stats = self.statistical_analyzer.analyze(messages)

       return AnalysisResult(
           session_id=session_id,
           message_count=len(messages),
           statistics=stats
       )

   async def _execute_learning_phase(
       self,
       classification: ClassificationResult,
       messages: List[bytes]
   ) -> LearningResult:
       """Combined grammar learning and parser generation."""
       # Learn grammar
       grammar = await self.grammar_learner.learn(messages, classification)

       # Generate parser (parallel if possible)
       parser = await self.parser_generator.generate(grammar)

       return LearningResult(grammar=grammar, parser=parser)
   ```

3. **Add circuit breaker**:
   ```python
   from circuitbreaker import circuit

   class DiscoveryOrchestrator:
       @circuit(failure_threshold=3, recovery_timeout=60)
       async def _execute_learning_phase(self, ...):
           """Protected by circuit breaker."""
           ...
   ```

4. **Expose partial results**:
   ```python
   class PartialDiscoveryResult:
       """Returned when pipeline fails mid-way."""
       completed_phases: List[DiscoveryPhase]
       failed_phase: DiscoveryPhase
       partial_data: Dict[str, Any]  # Whatever we learned
       error: QbitelError
   ```

#### Acceptance Criteria:
- [ ] Pipeline reduced to 5 phases
- [ ] Circuit breaker prevents cascade failures
- [ ] Partial results available on failure
- [ ] Performance improved (fewer phase transitions)

---

## Phase 4: Production Hardening (Weeks 9-12)

### Goal: Prepare for production deployment with enterprise reliability

---

### 4.1 Add Circuit Breakers Throughout

**Priority:** HIGH
**Effort:** 3 days
**Owner:** Backend Team

#### Components Needing Circuit Breakers:

| Component | Failure Scenario | Threshold |
|-----------|------------------|-----------|
| LLM Service | Provider down | 3 failures / 60s recovery |
| Database | Connection pool exhausted | 5 failures / 30s recovery |
| Redis Cache | Memory pressure | 3 failures / 15s recovery |
| External APIs | Rate limited | 5 failures / 120s recovery |

#### Implementation:

1. **Install pybreaker**:
   ```bash
   pip install pybreaker
   ```

2. **Create circuit breaker registry** `ai_engine/core/circuit_breakers.py`:
   ```python
   import pybreaker
   from prometheus_client import Counter, Gauge

   # Metrics
   circuit_state = Gauge(
       'qbitel_circuit_breaker_state',
       'Circuit breaker state (0=closed, 1=open, 2=half-open)',
       ['name']
   )
   circuit_failures = Counter(
       'qbitel_circuit_breaker_failures_total',
       'Circuit breaker failure count',
       ['name']
   )

   class MonitoredCircuitBreaker(pybreaker.CircuitBreaker):
       def __init__(self, name: str, **kwargs):
           self.name = name
           super().__init__(**kwargs)

           # Update metrics on state change
           self.add_listener(self._on_state_change)

       def _on_state_change(self, breaker, old_state, new_state):
           state_map = {'closed': 0, 'open': 1, 'half-open': 2}
           circuit_state.labels(name=self.name).set(state_map[new_state.name])

   # Registry
   circuit_breakers = {
       'llm': MonitoredCircuitBreaker(
           name='llm',
           fail_max=3,
           reset_timeout=60,
           exclude=[ValidationError]  # Don't trip on validation errors
       ),
       'database': MonitoredCircuitBreaker(
           name='database',
           fail_max=5,
           reset_timeout=30
       ),
       'redis': MonitoredCircuitBreaker(
           name='redis',
           fail_max=3,
           reset_timeout=15
       ),
   }
   ```

3. **Apply to services**:
   ```python
   from ai_engine.core.circuit_breakers import circuit_breakers

   class UnifiedLLMService:
       @circuit_breakers['llm']
       async def generate(self, prompt: str) -> str:
           ...
   ```

---

### 4.2 Implement Proper Health Checks

**Priority:** HIGH
**Effort:** 2 days
**Owner:** Backend Team

#### Health Check Levels:

| Endpoint | Purpose | Checks |
|----------|---------|--------|
| `/health` | Kubernetes liveness | Process alive |
| `/ready` | Kubernetes readiness | Dependencies ready |
| `/startup` | Kubernetes startup | Initial boot complete |
| `/health/detailed` | Debugging | All component status |

#### Implementation:

```python
# ai_engine/api/health.py
from fastapi import APIRouter, status
from pydantic import BaseModel
from typing import Dict, Optional
from enum import Enum

router = APIRouter(tags=["health"])

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ComponentHealth(BaseModel):
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None

class DetailedHealthResponse(BaseModel):
    status: HealthStatus
    version: str
    uptime_seconds: float
    components: Dict[str, ComponentHealth]

@router.get("/health")
async def liveness():
    """Kubernetes liveness probe - is the process alive?"""
    return {"status": "ok"}

@router.get("/ready")
async def readiness(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
):
    """Kubernetes readiness probe - can we serve traffic?"""
    checks = []

    # Check database
    try:
        await db.execute(text("SELECT 1"))
        checks.append(True)
    except Exception:
        checks.append(False)

    # Check Redis
    try:
        await redis.ping()
        checks.append(True)
    except Exception:
        checks.append(False)

    if all(checks):
        return {"status": "ready"}

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "not ready"}
    )

@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health():
    """Detailed health for debugging and monitoring."""
    components = {}

    # Check each component
    components["database"] = await check_database_health()
    components["redis"] = await check_redis_health()
    components["llm"] = await check_llm_health()
    components["discovery"] = await check_discovery_health()

    # Aggregate status
    statuses = [c.status for c in components.values()]
    if all(s == HealthStatus.HEALTHY for s in statuses):
        overall = HealthStatus.HEALTHY
    elif any(s == HealthStatus.UNHEALTHY for s in statuses):
        overall = HealthStatus.UNHEALTHY
    else:
        overall = HealthStatus.DEGRADED

    return DetailedHealthResponse(
        status=overall,
        version=settings.VERSION,
        uptime_seconds=get_uptime(),
        components=components
    )
```

---

### 4.3 Add Comprehensive Prometheus Metrics

**Priority:** MEDIUM
**Effort:** 3 days
**Owner:** Platform Team

#### Metrics to Add:

```python
# ai_engine/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info

# Application info
app_info = Info('qbitel_app', 'Application information')
app_info.info({
    'version': settings.VERSION,
    'python_version': platform.python_version(),
})

# Request metrics
request_latency = Histogram(
    'qbitel_request_duration_seconds',
    'Request latency in seconds',
    ['method', 'endpoint', 'status'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
)

request_count = Counter(
    'qbitel_requests_total',
    'Total request count',
    ['method', 'endpoint', 'status']
)

# Discovery metrics
discovery_duration = Histogram(
    'qbitel_discovery_duration_seconds',
    'Protocol discovery duration',
    ['phase', 'status'],
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120, 300]
)

discovery_confidence = Histogram(
    'qbitel_discovery_confidence',
    'Discovery confidence score',
    ['protocol_type'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

# LLM metrics
llm_tokens_used = Counter(
    'qbitel_llm_tokens_total',
    'LLM tokens used',
    ['provider', 'model', 'type']  # type: prompt/completion
)

llm_latency = Histogram(
    'qbitel_llm_latency_seconds',
    'LLM response latency',
    ['provider', 'model'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
)

# Resource metrics
active_connections = Gauge(
    'qbitel_active_connections',
    'Active connections',
    ['type']  # database, redis, websocket
)

model_memory_bytes = Gauge(
    'qbitel_model_memory_bytes',
    'Memory used by ML models',
    ['model_name']
)
```

---

### 4.4 Implement Graceful Shutdown

**Priority:** HIGH
**Effort:** 1 day
**Owner:** Backend Team

```python
# ai_engine/core/lifecycle.py
import asyncio
import signal
from contextlib import asynccontextmanager
from fastapi import FastAPI

class GracefulShutdown:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.active_requests = 0
        self.draining = False

    def start_request(self):
        if self.draining:
            raise ServiceUnavailable("Server is shutting down")
        self.active_requests += 1

    def end_request(self):
        self.active_requests -= 1

    async def shutdown(self, timeout: int = 30):
        """Graceful shutdown with timeout."""
        self.draining = True

        # Wait for active requests to complete
        start = asyncio.get_event_loop().time()
        while self.active_requests > 0:
            if asyncio.get_event_loop().time() - start > timeout:
                logger.warning(f"Forced shutdown with {self.active_requests} active requests")
                break
            await asyncio.sleep(0.1)

        self.shutdown_event.set()

shutdown_handler = GracefulShutdown()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting QBITEL...")
    await initialize_services()

    yield

    # Shutdown
    logger.info("Shutting down QBITEL...")
    await shutdown_handler.shutdown(timeout=30)
    await cleanup_services()

app = FastAPI(lifespan=lifespan)

# Signal handlers
def handle_sigterm(*args):
    asyncio.create_task(shutdown_handler.shutdown())

signal.signal(signal.SIGTERM, handle_sigterm)
```

---

## Milestones & Success Criteria

| Week | Phase | Deliverables | Success Criteria |
|------|-------|--------------|------------------|
| 2 | Phase 1 | Security hardening | Zero critical vulnerabilities |
| 5 | Phase 2 | Architecture simplification | 30% reduction in complexity |
| 8 | Phase 3 | Focus & prioritization | Core features 100% tested |
| 12 | Phase 4 | Production hardening | 99.9% uptime target |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking changes during refactor | Medium | High | Feature flags, backward-compatible imports |
| Performance regression | Low | Medium | Benchmark before/after each phase |
| Team bandwidth | Medium | Medium | Parallelize independent workstreams |
| Customer impact | Low | High | Staging environment testing, gradual rollout |

---

## Resource Requirements

| Phase | Engineers | Duration | Dependencies |
|-------|-----------|----------|--------------|
| Phase 1 | 2 backend | 2 weeks | None |
| Phase 2 | 2 backend, 1 ML | 3 weeks | Phase 1 |
| Phase 3 | 1 backend, 1 product | 3 weeks | Phase 2 |
| Phase 4 | 2 backend, 1 platform | 4 weeks | Phase 3 |

**Total:** 12 weeks, 3-4 engineers

---

## Next Steps

1. **Immediate (This Week):**
   - [ ] Review and approve this plan
   - [ ] Assign owners to Phase 1 tasks
   - [ ] Set up feature flag infrastructure

2. **Week 1:**
   - [ ] Implement rate limiting
   - [ ] Add input validation
   - [ ] Begin error handling standardization

3. **Week 2:**
   - [ ] Complete error handling refactor
   - [ ] Write tests for new middleware
   - [ ] Deploy to staging environment
