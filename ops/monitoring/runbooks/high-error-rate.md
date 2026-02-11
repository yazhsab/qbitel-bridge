# Runbook: High Error Rate

**Alert Name**: `HighErrorRate`
**Severity**: Critical
**Threshold**: >5% of requests returning 5xx errors

## Quick Diagnosis

```bash
# Check recent error logs
docker-compose logs --tail=200 qbitel-engine | grep "ERROR\|CRITICAL"

# Check error breakdown by endpoint
curl http://localhost:9090/metrics | grep http_requests_total | grep status_code=\"5

# Check database connectivity
docker-compose exec qbitel-engine python -c "import psycopg2; psycopg2.connect('dbname=qbitel user=qbitel_user host=postgres')"
```

## Common Causes

1. **Database Issues** (most common)
   - Connection pool exhausted → Check `DatabaseConnectionPoolExhausted` alert
   - Slow queries → Check `SlowDatabaseQueries` alert
   - Database down → Check database health

2. **External Service Failures**
   - LLM API errors → Check `LLMAPIErrors` alert
   - Third-party API timeouts

3. **Application Bugs**
   - Recent deployment introduced bugs
   - Unhandled exceptions

4. **Resource Constraints**
   - High CPU → Check `HighCPUUsage` alert
   - Memory pressure → Check `HighMemoryUsage` alert

## Resolution

1. **Identify failing endpoints**:
```bash
# Get error breakdown
curl http://localhost:9090/api/v1/query?query='rate(http_requests_total{status_code=~"5.."}[5m])'
```

2. **Check recent changes**:
```bash
# Rollback if recent deployment
kubectl rollout undo deployment/qbitel-engine -n qbitel
```

3. **Scale up resources**:
```bash
# Increase replicas
kubectl scale deployment/qbitel-engine --replicas=5 -n qbitel
```

## Escalation

- Immediate: On-call engineer
- 10 minutes: Development team lead
- 30 minutes: CTO

## Prevention

- Implement retry logic with exponential backoff
- Add circuit breakers
- Increase connection pool size
- Implement request hedging
