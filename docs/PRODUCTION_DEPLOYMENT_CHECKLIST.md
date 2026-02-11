# QBITEL - Production Deployment Checklist

**Version**: 2.1.0
**Last Updated**: 2024-11-22
**Purpose**: Pre-flight checklist for production deployment

---

## 1. Pre-Deployment Verification

### 1.1 Code Quality
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] No critical security vulnerabilities in dependencies
- [ ] Code review completed
- [ ] All TODO items resolved or tracked

### 1.2 Configuration
- [ ] Production configuration file reviewed (`config/qbitel.production.yaml`)
- [ ] All secrets stored in secrets manager (not in code/config)
- [ ] Environment variables documented
- [ ] TLS certificates valid and not expiring within 30 days

---

## 2. Security Checklist

### 2.1 Authentication & Authorization
- [ ] JWT secret is cryptographically strong (32+ chars)
- [ ] API keys are properly hashed
- [ ] MFA enabled for admin accounts
- [ ] OAuth2/SAML providers configured
- [ ] Session timeout configured appropriately

### 2.2 Network Security
- [ ] TLS 1.3 enabled
- [ ] HSTS headers configured
- [ ] CORS properly restricted
- [ ] Rate limiting enabled
- [ ] Input validation middleware active
- [ ] WAF rules configured (if applicable)

### 2.3 Data Security
- [ ] Database encryption at rest enabled
- [ ] PII data encrypted with AES-256-GCM
- [ ] Backup encryption enabled
- [ ] Post-quantum cryptography enabled (Kyber-1024, Dilithium-5)

### 2.4 Audit & Compliance
- [ ] Audit logging enabled
- [ ] SIEM integration configured (Splunk/Elastic/QRadar)
- [ ] Data retention policies applied
- [ ] Compliance framework selected (SOC2/HIPAA/PCI-DSS/GDPR)

---

## 3. Infrastructure Checklist

### 3.1 Database
- [ ] PostgreSQL 15+ deployed
- [ ] Connection pooling configured (50 connections for prod)
- [ ] Read replicas configured (if high availability required)
- [ ] Automated backups enabled
- [ ] Point-in-time recovery tested
- [ ] Database migrations applied (`alembic upgrade head`)

### 3.2 Caching
- [ ] Redis deployed
- [ ] Redis password set
- [ ] Redis persistence configured
- [ ] Cache TTL values reviewed

### 3.3 Message Queue (if applicable)
- [ ] Kafka/RabbitMQ deployed
- [ ] Topic retention policies set
- [ ] Consumer lag monitoring enabled

---

## 4. Kubernetes Deployment

### 4.1 Cluster Setup
- [ ] Kubernetes 1.24+ cluster ready
- [ ] Namespaces created (`qbitel`, `qbitel-service-mesh`)
- [ ] Resource quotas applied
- [ ] Network policies configured
- [ ] Pod security policies/standards applied

### 4.2 Workloads
- [ ] Deployments configured with proper resource limits
- [ ] Horizontal Pod Autoscaler (HPA) configured
- [ ] Pod Disruption Budgets (PDB) set
- [ ] Anti-affinity rules applied for high availability

### 4.3 Service Mesh (Istio)
- [ ] Istio installed and configured
- [ ] mTLS enabled between services
- [ ] Envoy sidecars injecting properly
- [ ] Traffic policies configured

### 4.4 Secrets Management
- [ ] Kubernetes secrets created (or external secrets operator configured)
- [ ] Secrets encrypted at rest in etcd
- [ ] RBAC for secrets access configured

---

## 5. Monitoring & Observability

### 5.1 Metrics
- [ ] Prometheus deployed
- [ ] ServiceMonitors configured
- [ ] Grafana dashboards imported
- [ ] Alert rules configured

### 5.2 Logging
- [ ] Structured logging enabled
- [ ] Log aggregation configured (ELK/Loki)
- [ ] Log retention policies set
- [ ] Correlation IDs enabled

### 5.3 Tracing
- [ ] OpenTelemetry/Jaeger configured
- [ ] Trace sampling rate set
- [ ] Service dependencies visible

### 5.4 Alerting
- [ ] PagerDuty/OpsGenie integration configured
- [ ] Slack notifications configured
- [ ] Alert escalation policies defined
- [ ] On-call rotation set up

---

## 6. Application Configuration

### 6.1 Required Environment Variables

```bash
# Database (REQUIRED)
DATABASE_HOST=<postgres-host>
DATABASE_PORT=5432
DATABASE_NAME=qbitel_prod
DATABASE_USER=<db-user>
DATABASE_PASSWORD=<from-secrets-manager>

# Redis (REQUIRED)
REDIS_HOST=<redis-host>
REDIS_PORT=6379
REDIS_PASSWORD=<from-secrets-manager>

# Security (REQUIRED)
JWT_SECRET=<32+-char-secret>
ENCRYPTION_KEY=<32-char-aes-key>

# TLS (REQUIRED for production)
TLS_ENABLED=true
TLS_CERT_FILE=/etc/qbitel/certs/server.crt
TLS_KEY_FILE=/etc/qbitel/certs/server.key

# LLM Configuration
QBITEL_LLM_PROVIDER=ollama  # or anthropic, openai
QBITEL_LLM_ENDPOINT=http://ollama:11434
QBITEL_AIRGAPPED_MODE=false

# Monitoring (RECOMMENDED)
SENTRY_DSN=<sentry-dsn>
LOG_LEVEL=INFO
METRICS_ENABLED=true

# Notifications (RECOMMENDED)
SLACK_WEBHOOK_URL=<webhook-url>
PAGERDUTY_ROUTING_KEY=<routing-key>

# SIEM (RECOMMENDED)
SIEM_TYPE=splunk  # or elastic, qradar
SPLUNK_HEC_URL=<hec-endpoint>
SPLUNK_HEC_TOKEN=<hec-token>

# Marketplace (if enabled)
STRIPE_API_KEY=<stripe-key>
STRIPE_WEBHOOK_SECRET=<webhook-secret>
AWS_ACCESS_KEY_ID=<aws-key>
AWS_SECRET_ACCESS_KEY=<aws-secret>
S3_BUCKET=qbitel-marketplace
```

### 6.2 Health Check Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `GET /health` | Basic health | `{"status": "healthy"}` |
| `GET /healthz` | Liveness probe | `200 OK` |
| `GET /readyz` | Readiness probe | `200 OK` (when dependencies ready) |
| `GET /metrics` | Prometheus metrics | Prometheus format |

---

## 7. Deployment Steps

### 7.1 Database Migration
```bash
# Apply migrations
cd ai_engine
alembic upgrade head

# Verify migration status
alembic current
```

### 7.2 Docker Deployment
```bash
# Build images
docker-compose -f ops/deploy/docker/docker-compose.yml build

# Deploy
docker-compose -f ops/deploy/docker/docker-compose.yml up -d

# Verify
docker-compose ps
curl http://localhost:8000/health
```

### 7.3 Kubernetes Deployment
```bash
# Create namespace
kubectl create namespace qbitel

# Apply secrets (from sealed-secrets or external-secrets)
kubectl apply -f kubernetes/secrets/

# Deploy via Helm
helm install qbitel ./helm/qbitel \
  --namespace qbitel \
  --values ./helm/qbitel/values-production.yaml

# Verify deployment
kubectl get pods -n qbitel
kubectl get svc -n qbitel
```

### 7.4 Post-Deployment Verification
```bash
# Health check
curl -k https://<service-url>/health

# API smoke test
curl -k https://<service-url>/api/v1/copilot/ \
  -H "Authorization: Bearer <token>"

# Check logs
kubectl logs -n qbitel -l app=qbitel --tail=100

# Check metrics
curl http://<service-url>:9090/metrics | grep qbitel
```

---

## 8. Rollback Procedure

### 8.1 Helm Rollback
```bash
# List releases
helm history qbitel -n qbitel

# Rollback to previous
helm rollback qbitel -n qbitel

# Rollback to specific revision
helm rollback qbitel <revision> -n qbitel
```

### 8.2 Database Rollback
```bash
# Downgrade to specific version
alembic downgrade <revision>

# Downgrade one step
alembic downgrade -1
```

---

## 9. Performance Tuning

### 9.1 Recommended Resource Limits

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|-----------|----------------|--------------|
| AI Engine | 2 | 4 | 4Gi | 8Gi |
| API Server | 1 | 2 | 2Gi | 4Gi |
| Worker | 2 | 4 | 4Gi | 8Gi |

### 9.2 Database Tuning
```yaml
# PostgreSQL settings for production
max_connections: 200
shared_buffers: 4GB
effective_cache_size: 12GB
work_mem: 64MB
maintenance_work_mem: 1GB
```

### 9.3 Connection Pool Settings
```yaml
database:
  pool_size: 50
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600
```

---

## 10. Disaster Recovery

### 10.1 Backup Schedule
- [ ] Database backups: Every 6 hours
- [ ] Configuration backups: Daily
- [ ] Model artifacts: After each training

### 10.2 Recovery Time Objectives
| Scenario | RTO | RPO |
|----------|-----|-----|
| Database failure | 30 min | 6 hours |
| Application failure | 5 min | 0 |
| Complete cluster failure | 2 hours | 6 hours |

### 10.3 DR Testing
- [ ] Backup restoration tested
- [ ] Failover procedure tested
- [ ] Recovery runbook documented

---

## 11. Go-Live Checklist

### Day Before Go-Live
- [ ] Final code freeze
- [ ] Staging environment validated
- [ ] Load testing completed
- [ ] Security scan completed
- [ ] Runbooks reviewed
- [ ] On-call team briefed

### Go-Live Day
- [ ] Deployment window communicated
- [ ] Monitoring dashboards open
- [ ] Support team on standby
- [ ] Rollback procedure ready
- [ ] Customer communication sent (if applicable)

### Post Go-Live
- [ ] Health checks passing
- [ ] No critical alerts
- [ ] Performance within SLA
- [ ] Customer feedback monitored
- [ ] Incident-free for 24 hours

---

## 12. Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Engineering Lead | | | |
| Security Lead | | | |
| DevOps Lead | | | |
| Product Owner | | | |

---

**Document maintained by**: QBITEL Engineering Team
**Review frequency**: Before each production deployment
