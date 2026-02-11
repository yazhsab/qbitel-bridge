# QBITEL - Service Level Objectives (SLOs)

## Overview

This document defines the Service Level Objectives (SLOs) for QBITEL services. SLOs represent the target level of reliability and performance we commit to delivering to our users.

**Last Updated**: 2025-01-18
**Version**: 1.0
**Review Cycle**: Quarterly

---

## SLO Framework

### Key Metrics

- **Availability**: Percentage of time service is operational
- **Latency**: Response time percentiles (p50, p95, p99)
- **Throughput**: Requests processed per second
- **Error Rate**: Percentage of failed requests

### Error Budget

An **error budget** is the maximum amount of unreliability allowed within the SLO target. For example, a 99.9% availability SLO allows for 0.1% downtime (43.8 minutes per month).

### SLO Tiers

- **Tier 1 (Critical)**: Core functionality, highest priority (99.9%+)
- **Tier 2 (Important)**: Enhanced features (99.5%+)
- **Tier 3 (Best Effort)**: Optional features (99%+)

---

## Service Level Objectives

### 1. REST API Service

**Service**: QBITEL REST API
**Tier**: Tier 1 (Critical)
**Business Impact**: Direct user-facing API

#### Availability SLO

| Metric | Target | Error Budget (Monthly) | Measurement Window |
|--------|--------|------------------------|-------------------|
| **Availability** | **99.9%** | 43.8 minutes | 30 days rolling |

**Definition**: Percentage of successful API requests (non-5xx responses)

**Calculation**:
```
Availability = (Total Requests - 5xx Errors) / Total Requests
```

**Monitoring Query**:
```promql
1 - (
  sum(rate(http_requests_total{status_code=~"5.."}[30d]))
  /
  sum(rate(http_requests_total[30d]))
)
```

#### Latency SLOs

| Percentile | Target | Measurement Window | Alert Threshold |
|------------|--------|-------------------|-----------------|
| **p50** | ≤ 200ms | 5 minutes | > 300ms |
| **p95** | ≤ 1s | 5 minutes | > 1.5s |
| **p99** | ≤ 2s | 5 minutes | > 3s |

**Measurement**:
```promql
histogram_quantile(0.95,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint)
)
```

#### Throughput SLO

| Metric | Target | Measurement Window |
|--------|--------|-------------------|
| **Peak Throughput** | 1,000 req/s | 1 minute |
| **Sustained Throughput** | 500 req/s | 5 minutes |

---

### 2. gRPC API Service

**Service**: QBITEL gRPC API
**Tier**: Tier 1 (Critical)
**Business Impact**: High-performance protocol processing

#### Availability SLO

| Metric | Target | Error Budget (Monthly) |
|--------|--------|------------------------|
| **Availability** | **99.9%** | 43.8 minutes |

#### Latency SLOs

| Percentile | Target | Alert Threshold |
|------------|--------|-----------------|
| **p50** | ≤ 100ms | > 150ms |
| **p95** | ≤ 500ms | > 750ms |
| **p99** | ≤ 1s | > 1.5s |

**Rationale**: gRPC is used for high-performance scenarios, requires tighter latency targets

---

### 3. Protocol Discovery Service

**Service**: AI-powered Protocol Discovery
**Tier**: Tier 1 (Critical)
**Business Impact**: Core AI functionality

#### Success Rate SLO

| Metric | Target | Error Budget (Monthly) |
|--------|--------|------------------------|
| **Discovery Success Rate** | **95%** | 5% failures allowed |

**Definition**: Percentage of protocol discovery requests that successfully identify protocol structure

**Calculation**:
```
Success Rate = (Successful Discoveries) / (Total Discovery Requests)
```

**Monitoring Query**:
```promql
sum(rate(protocol_discovery_success_total[30d]))
/
sum(rate(protocol_discovery_attempts_total[30d]))
```

#### Completion Time SLO

| Percentile | Target | Measurement Window |
|------------|--------|-------------------|
| **p50** | ≤ 30s | 5 minutes |
| **p95** | ≤ 2 minutes | 5 minutes |
| **p99** | ≤ 5 minutes | 5 minutes |

**Rationale**: Protocol discovery is compute-intensive, longer latencies acceptable

---

### 4. Model Inference Service

**Service**: AI Model Inference
**Tier**: Tier 2 (Important)
**Business Impact**: AI-enhanced features

#### Availability SLO

| Metric | Target | Error Budget (Monthly) |
|--------|--------|------------------------|
| **Availability** | **99.5%** | 3.6 hours |

#### Latency SLOs

| Model Type | p95 Target | p99 Target |
|------------|-----------|-----------|
| **Field Detection** | ≤ 1s | ≤ 2s |
| **Anomaly Detection** | ≤ 2s | ≤ 5s |
| **Grammar Learning** | ≤ 5s | ≤ 10s |

#### Accuracy SLO

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Field Detection Accuracy** | ≥ 90% | Evaluated on test dataset monthly |
| **Anomaly Detection Precision** | ≥ 85% | False positive rate ≤ 15% |

---

### 5. Database Operations

**Service**: PostgreSQL Database
**Tier**: Tier 1 (Critical)
**Business Impact**: Data persistence for all services

#### Availability SLO

| Metric | Target | Error Budget (Monthly) |
|--------|--------|------------------------|
| **Database Availability** | **99.95%** | 21.9 minutes |

**Rationale**: Higher availability than API services since database is dependency for all

#### Query Performance SLO

| Query Type | p95 Target | p99 Target |
|------------|-----------|-----------|
| **Read Queries** | ≤ 100ms | ≤ 500ms |
| **Write Queries** | ≤ 200ms | ≤ 1s |
| **Complex Queries** | ≤ 1s | ≤ 3s |

#### Connection SLO

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| **Connection Pool Utilization** | ≤ 80% | > 90% |
| **Failed Connections** | < 0.1% | > 1% |

---

### 6. Redis Cache Service

**Service**: Redis Cache
**Tier**: Tier 2 (Important)
**Business Impact**: Performance optimization (degraded if unavailable, not critical)

#### Availability SLO

| Metric | Target | Error Budget (Monthly) |
|--------|--------|------------------------|
| **Cache Availability** | **99.5%** | 3.6 hours |

**Rationale**: Cache unavailability degrades performance but doesn't break functionality

#### Performance SLO

| Metric | Target |
|--------|--------|
| **Cache Hit Rate** | ≥ 80% |
| **Cache Response Time (p99)** | ≤ 10ms |

---

### 7. Protocol Copilot (LLM-Powered)

**Service**: LLM-Enhanced Protocol Analysis
**Tier**: Tier 3 (Best Effort)
**Business Impact**: Premium feature

#### Availability SLO

| Metric | Target | Error Budget (Monthly) |
|--------|--------|------------------------|
| **Copilot Availability** | **99%** | 7.2 hours |

**Rationale**: Depends on third-party LLM APIs, best-effort due to external dependencies

#### Response Time SLO

| Percentile | Target | Notes |
|------------|--------|-------|
| **p95** | ≤ 10s | LLM latency variable |
| **p99** | ≤ 30s | Includes complex analysis |

#### Success Rate SLO

| Metric | Target |
|--------|--------|
| **LLM Request Success Rate** | ≥ 95% |

---

## SLO Compliance & Monitoring

### Measurement

SLOs are measured using:
- **Prometheus** for metrics collection
- **Grafana** for visualization
- **Alert Manager** for SLO violations

### Reporting

- **Real-time**: Grafana dashboards
- **Daily**: Automated SLO compliance reports
- **Weekly**: SRE team review
- **Monthly**: Executive summary
- **Quarterly**: SLO review and adjustment

### Error Budget Policy

#### When Error Budget is Healthy (>20% remaining)

✅ **Normal Operations**:
- Proceed with scheduled deployments
- Experiment with new features
- Performance optimizations

#### When Error Budget is Low (5-20% remaining)

⚠️ **Caution**:
- Reduce deployment frequency
- Increase testing rigor
- Focus on stability over features
- No risky changes

#### When Error Budget is Exhausted (<5% remaining)

🚫 **Feature Freeze**:
- **STOP** all feature deployments
- Only critical bug fixes and rollbacks allowed
- All hands on deck to improve reliability
- Daily SRE review meetings
- Root cause analysis for all incidents

### SLO Violation Response

**Tier 1 Services** (99.9% SLO):
- Immediate incident response
- Post-mortem required
- Preventive measures mandatory

**Tier 2 Services** (99.5% SLO):
- Incident investigation within 24 hours
- Post-mortem if recurring
- Preventive measures recommended

**Tier 3 Services** (99% SLO):
- Review during next sprint planning
- Improvement tasks created
- Post-mortem optional

---

## SLO Dashboard Queries

### Overall Service Health

```promql
# API Availability (30-day)
1 - (
  sum(rate(http_requests_total{status_code=~"5.."}[30d]))
  /
  sum(rate(http_requests_total[30d]))
)

# Error Budget Remaining
100 * (1 - (
  (1 - (
    sum(rate(http_requests_total{status_code!~"5.."}[30d]))
    /
    sum(rate(http_requests_total[30d]))
  ))
  /
  (1 - 0.999)  # 99.9% SLO
))

# Latency p95
histogram_quantile(0.95,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
)
```

### Protocol Discovery SLO

```promql
# Discovery Success Rate
sum(rate(protocol_discovery_success_total[30d]))
/
sum(rate(protocol_discovery_attempts_total[30d]))

# Discovery Latency p95
histogram_quantile(0.95,
  sum(rate(protocol_discovery_duration_seconds_bucket[5m])) by (le)
)
```

### Database SLO

```promql
# Database Availability
avg(up{job="postgres"})

# Query Latency p95
histogram_quantile(0.95,
  sum(rate(database_query_duration_seconds_bucket[5m])) by (le)
)

# Connection Pool Utilization
database_connection_pool_active / database_connection_pool_max
```

---

## SLO Targets by Environment

### Production

All SLOs as defined above apply to production.

### Staging

- **Availability**: 99% (relaxed for testing)
- **Latency**: Same targets as production
- **Error Budget**: Not tracked (testing environment)

### Development

- **No SLOs** - Best effort only

---

## SLO Review Process

### Quarterly Review

**Participants**: SRE Team, Engineering Leads, Product Management

**Agenda**:
1. Review current SLO compliance
2. Analyze error budget consumption patterns
3. Identify services consistently missing SLOs
4. Identify services with overly conservative SLOs
5. Propose SLO adjustments based on:
   - Business requirements changes
   - User feedback
   - Cost/benefit analysis
6. Update this document

### Annual Review

**Participants**: Executive Team, Engineering, Product

**Topics**:
- Strategic SLO alignment with business goals
- Resource allocation for SLO improvements
- Customer SLA commitments
- Competitive analysis

---

## Customer-Facing SLAs

### Enterprise SLA

Based on our internal SLOs, we commit to the following SLA for enterprise customers:

| Service | Availability | Support Response |
|---------|--------------|------------------|
| **REST API** | 99.9% uptime | < 15 minutes (critical) |
| **Protocol Discovery** | 95% success rate | < 1 hour (high) |
| **Data Persistence** | 99.95% | < 4 hours (medium) |

**Monthly Credits**:
- < 99.9%: 10% credit
- < 99%: 25% credit
- < 95%: 50% credit

### Standard SLA

| Service | Availability | Support Response |
|---------|--------------|------------------|
| **REST API** | 99.5% uptime | < 4 hours (critical) |
| **Protocol Discovery** | 90% success rate | Next business day |

---

## Appendix: SLO Calculation Examples

### Example 1: Monthly Availability

```
Total Requests (30 days): 100,000,000
Failed Requests (5xx): 50,000

Availability = (100,000,000 - 50,000) / 100,000,000
             = 99.95%

SLO Target: 99.9%
Status: ✅ MEETING SLO (0.05% margin)
Error Budget Used: 50% (50k of 100k allowed errors)
```

### Example 2: Error Budget Exhaustion

```
SLO: 99.9% (allows 0.1% errors)
Total Requests: 10,000,000
Allowed Errors: 10,000

Actual Errors: 12,000
Error Budget Consumed: 120%
Status: ❌ SLO VIOLATED
Action: Feature freeze, focus on reliability
```

### Example 3: Latency SLO

```
p95 Latency SLO: 1 second
Measured p95: 0.85 seconds
Status: ✅ MEETING SLO
Margin: 150ms (15% headroom)
```

---

## References

- [Prometheus Alert Rules](./prometheus/alerts.yml)
- [Monitoring Runbooks](./runbooks/README.md)
- [Production Environment Variables](../../docs/PRODUCTION_ENVIRONMENT_VARIABLES.md)
- [Google SRE Book - SLO Definition](https://sre.google/sre-book/service-level-objectives/)

---

**Document Owner**: SRE Team
**Review Schedule**: Quarterly
**Next Review**: 2025-04-18
