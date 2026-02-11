# QBITEL Implementation Checklist

## Quick Reference for Implementation Progress

---

## Phase 1: Security Hardening (Weeks 1-2)

### 1.1 API Rate Limiting
- [ ] Install `slowapi` package
- [ ] Create `ai_engine/api/middleware/rate_limiter.py`
- [ ] Configure rate limits per endpoint type
- [ ] Add Redis backend for distributed rate limiting
- [ ] Add rate limit headers to responses
- [ ] Add Prometheus metrics for rate limit hits
- [ ] Write unit tests
- [ ] Update API documentation

### 1.2 Input Size Validation
- [ ] Create `ai_engine/core/constants.py` with size limits
- [ ] Create `ai_engine/api/middleware/validators.py`
- [ ] Apply `@validate_input_size` to all endpoints
- [ ] Add global request size limit middleware
- [ ] Write unit tests for boundary conditions
- [ ] Document size limits in API spec

### 1.3 Standardized Error Handling
- [ ] Create `ai_engine/core/exceptions.py` with error hierarchy
- [ ] Define `ErrorCode` enum (1000-9999 range)
- [ ] Create `QbitelError` base exception
- [ ] Create domain-specific exceptions (Discovery, Detection, LLM, Auth, Validation)
- [ ] Create `ai_engine/api/middleware/error_handler.py`
- [ ] Register exception handlers in FastAPI
- [ ] Refactor `protocol_discovery_orchestrator.py`
- [ ] Refactor `field_detector.py`
- [ ] Refactor `unified_llm_service.py`
- [ ] Refactor `protocol_copilot.py`
- [ ] Refactor `legacy/service.py`
- [ ] Write tests for each error type
- [ ] Document error codes in API spec

---

## Phase 2: Architecture Simplification (Weeks 3-5)

### 2.1 Remove Unused LLM Providers
- [ ] Audit current LLM provider usage
- [ ] Remove vLLM integration code
- [ ] Remove LocalAI integration code
- [ ] Remove from `pyproject.toml`
- [ ] Simplify `UnifiedLLMService` to 2 providers
- [ ] Update configuration files
- [ ] Update tests
- [ ] Update documentation

### 2.2 Split Database Models
- [ ] Create `ai_engine/models/base.py` with mixins
- [ ] Create `ai_engine/models/auth/user.py`
- [ ] Create `ai_engine/models/auth/session.py`
- [ ] Create `ai_engine/models/auth/api_key.py`
- [ ] Create `ai_engine/models/auth/mfa.py`
- [ ] Create `ai_engine/models/providers/oauth.py`
- [ ] Create `ai_engine/models/providers/saml.py`
- [ ] Create `ai_engine/models/audit/audit_log.py`
- [ ] Create `ai_engine/models/protocol/` (if needed)
- [ ] Update `ai_engine/models/__init__.py` for backward compat
- [ ] Add missing indexes for `audit_logs`
- [ ] Generate Alembic migration
- [ ] Test migration on staging
- [ ] Run all existing tests

### 2.3 Refactor Legacy Whisperer
- [ ] Create `ai_engine/legacy/analyzer/` module
  - [ ] `service.py` - COBOLAnalyzerService
  - [ ] `parser.py` - COBOL syntax parser
  - [ ] `copybook_parser.py`
  - [ ] `data_flow.py`
  - [ ] `models.py`
- [ ] Create `ai_engine/legacy/mapper/` module
  - [ ] `service.py` - TransactionMapperService
  - [ ] `flow_analyzer.py`
  - [ ] `dependency_graph.py`
  - [ ] `models.py`
- [ ] Create `ai_engine/legacy/generator/` module
  - [ ] `service.py` - CodeGeneratorService
  - [ ] `rest_generator.py`
  - [ ] `sdk_generator.py`
  - [ ] `openapi_generator.py`
  - [ ] `templates/`
- [ ] Create `ai_engine/legacy/orchestrator/` module
  - [ ] `service.py` - LegacyWhispererOrchestrator
  - [ ] `workflow.py`
- [ ] Update `ai_engine/legacy/__init__.py` for backward compat
- [ ] Move all code from monolithic service
- [ ] Write unit tests for each new service
- [ ] Integration tests for orchestrator
- [ ] Verify test coverage maintained

---

## Phase 3: Focus & Prioritization (Weeks 6-8)

### 3.1 Feature Flags for Domain Modules
- [ ] Create `ai_engine/core/feature_flags.py`
- [ ] Add flags for:
  - [ ] `ENABLE_AVIATION_DOMAIN`
  - [ ] `ENABLE_AUTOMOTIVE_DOMAIN`
  - [ ] `ENABLE_HEALTHCARE_DOMAIN`
  - [ ] `ENABLE_MARKETPLACE`
  - [ ] `ENABLE_PROTOCOL_MARKETPLACE`
- [ ] Guard domain imports in `ai_engine/domains/__init__.py`
- [ ] Guard API routes in router registration
- [ ] Test that disabled modules don't load
- [ ] Document feature flags in README
- [ ] Add environment variable examples

### 3.2 Simplify Discovery Pipeline
- [ ] Update `DiscoveryPhase` enum (7 → 5)
- [ ] Merge INITIALIZATION + STATISTICAL_ANALYSIS → ANALYSIS
- [ ] Merge GRAMMAR_LEARNING + PARSER_GENERATION → LEARNING
- [ ] Install `pybreaker` for circuit breakers
- [ ] Add circuit breaker to learning phase
- [ ] Create `PartialDiscoveryResult` model
- [ ] Update orchestrator to return partial results on failure
- [ ] Update API response models
- [ ] Performance benchmark before/after
- [ ] Update tests
- [ ] Update documentation

---

## Phase 4: Production Hardening (Weeks 9-12)

### 4.1 Circuit Breakers
- [ ] Install `pybreaker`
- [ ] Create `ai_engine/core/circuit_breakers.py`
- [ ] Add `MonitoredCircuitBreaker` with Prometheus metrics
- [ ] Configure breakers for:
  - [ ] LLM service (3 failures / 60s)
  - [ ] Database (5 failures / 30s)
  - [ ] Redis (3 failures / 15s)
  - [ ] External APIs (5 failures / 120s)
- [ ] Apply to all external service calls
- [ ] Add circuit breaker state to health endpoint
- [ ] Create Grafana dashboard for circuit breakers
- [ ] Write tests for circuit breaker behavior

### 4.2 Health Checks
- [ ] Create `ai_engine/api/health.py`
- [ ] Implement `/health` (liveness)
- [ ] Implement `/ready` (readiness)
- [ ] Implement `/startup` (startup probe)
- [ ] Implement `/health/detailed`
- [ ] Add component health checks:
  - [ ] Database
  - [ ] Redis
  - [ ] LLM service
  - [ ] Discovery service
- [ ] Update Kubernetes manifests with probe configs
- [ ] Test probes in staging

### 4.3 Prometheus Metrics
- [ ] Create `ai_engine/monitoring/metrics.py`
- [ ] Add application info metric
- [ ] Add request latency histogram
- [ ] Add request count by endpoint/status
- [ ] Add discovery duration by phase
- [ ] Add discovery confidence histogram
- [ ] Add LLM token usage counter
- [ ] Add LLM latency histogram
- [ ] Add active connections gauge
- [ ] Add model memory gauge
- [ ] Update Grafana dashboards
- [ ] Add alerting rules for new metrics

### 4.4 Graceful Shutdown
- [ ] Create `ai_engine/core/lifecycle.py`
- [ ] Implement `GracefulShutdown` class
- [ ] Add request tracking (start/end)
- [ ] Implement drain mode
- [ ] Handle SIGTERM signal
- [ ] Add shutdown timeout (30s default)
- [ ] Update FastAPI lifespan handler
- [ ] Test graceful shutdown in Kubernetes
- [ ] Verify no dropped requests during deploy

---

## Testing Milestones

### End of Phase 1
- [ ] All security middleware has >90% test coverage
- [ ] No critical/high vulnerabilities in security scan
- [ ] Rate limiting verified under load

### End of Phase 2
- [ ] All refactored modules have >85% test coverage
- [ ] No regression in existing tests
- [ ] Performance benchmark shows no degradation

### End of Phase 3
- [ ] Feature flags tested (enable/disable)
- [ ] Pipeline simplification verified
- [ ] Partial results API documented

### End of Phase 4
- [ ] Circuit breakers tested under failure
- [ ] Health checks verified in Kubernetes
- [ ] Graceful shutdown verified
- [ ] 99.9% uptime target achievable

---

## Documentation Updates

- [ ] Update README with new architecture
- [ ] Update API documentation
- [ ] Create runbook for new error codes
- [ ] Update deployment guide
- [ ] Create feature flag documentation
- [ ] Update architecture diagrams
- [ ] Create circuit breaker runbook

---

## Deployment Checklist

### Staging Deploy
- [ ] All tests pass
- [ ] Security scan clean
- [ ] Performance benchmarks acceptable
- [ ] Feature flags configured
- [ ] Health checks verified

### Production Deploy
- [ ] Staging sign-off
- [ ] Backup taken
- [ ] Rollback plan documented
- [ ] Monitoring dashboards ready
- [ ] Alert thresholds configured
- [ ] On-call notified
