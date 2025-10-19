# CRONOS AI - Monitoring Runbooks

## Overview

This directory contains operational runbooks for responding to alerts from Prometheus monitoring.

## Quick Reference

| Alert | Severity | Response Time | Runbook |
|-------|----------|---------------|---------|
| ServiceDown | Critical | Immediate | [service-down.md](./service-down.md) |
| HighErrorRate | Critical | < 5 min | [high-error-rate.md](./high-error-rate.md) |
| DatabaseConnectionFailures | Critical | < 10 min | database-connection-failures.md |
| HighCPUUsage | Warning | < 30 min | high-cpu.md |
| HighMemoryUsage | Warning | < 30 min | high-memory.md |
| HighLatency | Warning | < 15 min | high-latency.md |

## Alert Response Guidelines

### Severity Levels

- **Critical**: Service impacting, immediate response required
- **Warning**: May become critical, respond within 30 minutes
- **Info**: For awareness, review during business hours

### Response Times

- **Critical**: Immediate (page on-call)
- **Warning**: < 30 minutes
- **Info**: Next business day

### Escalation Path

1. **On-Call Engineer** (immediate for critical)
2. **DevOps Lead** (after 15 min for critical, 30 min for warning)
3. **Engineering Manager** (after 30 min for critical)
4. **CTO** (after 1 hour for critical)

## Using Runbooks

1. **Acknowledge the alert** in your alerting system
2. **Open the relevant runbook**
3. **Follow diagnosis steps** to identify root cause
4. **Execute resolution steps**
5. **Verify the fix** (check metrics, test endpoints)
6. **Document what happened** (for post-mortem)

## Common Tools

```bash
# Check service status
kubectl get pods -n cronos-ai

# View logs
kubectl logs -l app=cronos-ai-engine -n cronos-ai --tail=100

# Check metrics
curl http://prometheus:9090/api/v1/query?query=up{job="cronos-ai"}

# Restart service
kubectl rollout restart deployment/cronos-ai-engine -n cronos-ai

# Scale service
kubectl scale deployment/cronos-ai-engine --replicas=5 -n cronos-ai
```

## Creating New Runbooks

When creating a new runbook, include:

1. **Alert Details**: Name, severity, threshold
2. **Description**: What the alert means
3. **Impact**: User, business, data impact
4. **Diagnosis**: How to identify the issue
5. **Common Causes**: Typical reasons for the alert
6. **Resolution**: Step-by-step fix instructions
7. **Escalation**: When and who to escalate to
8. **Prevention**: How to prevent future occurrences

## Post-Incident Process

After resolving an incident:

1. Write a brief incident summary
2. Update the runbook with new learnings
3. Create tickets for preventive measures
4. Schedule post-mortem if incident was major

## Available Runbooks

### Critical Alerts
- [service-down.md](./service-down.md) - Service completely unavailable
- [high-error-rate.md](./high-error-rate.md) - High rate of 5xx errors

### Warning Alerts
- high-latency.md - API response times degraded
- high-cpu.md - CPU usage approaching limits
- high-memory.md - Memory usage approaching limits
- database-connection-failures.md - Database connectivity issues
- redis-down.md - Redis cache unavailable

### Info Alerts
- low-gpu-utilization.md - GPU not being fully utilized

## Monitoring Dashboards

- **Grafana**: http://grafana.cronos-ai.com
- **Prometheus**: http://prometheus.cronos-ai.com
- **Alert Manager**: http://alertmanager.cronos-ai.com

## Contact Information

- **On-Call Engineer**: Check PagerDuty schedule
- **DevOps Team**: devops@cronos-ai.com
- **Slack Channel**: #cronos-ai-incidents

## Related Documentation

- [Prometheus Alerts Configuration](../prometheus/alerts.yml)
- [Production Deployment Guide](../../../docs/DEPLOYMENT.md)
- [SLO Definitions](../SLO.md)
- [Architecture Documentation](../../../docs/ARCHITECTURE.md)

---

**Last Updated**: 2025-01-18
**Maintained By**: CRONOS AI DevOps Team
