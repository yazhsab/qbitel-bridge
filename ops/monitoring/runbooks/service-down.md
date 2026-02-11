# Runbook: Service Down

**Alert Name**: `ServiceDown`
**Severity**: Critical
**Component**: API

## Description

The QBITEL Engine service is not responding to health checks. This is a critical issue that requires immediate attention as it means the service is completely unavailable.

## Impact

- **User Impact**: HIGH - Service is completely unavailable
- **Business Impact**: CRITICAL - All API functionality is down
- **Data Impact**: None (but no new data can be processed)

## Diagnosis

### 1. Check if the service is running

```bash
# Check Docker container status
docker-compose ps qbitel-engine

# Check Kubernetes pod status
kubectl get pods -l app=qbitel-engine -n qbitel

# Check systemd service (if running on VM)
systemctl status qbitel
```

### 2. Check recent logs

```bash
# Docker logs
docker-compose logs --tail=100 qbitel-engine

# Kubernetes logs
kubectl logs -l app=qbitel-engine -n qbitel --tail=100

# System logs
journalctl -u qbitel -n 100
```

### 3. Check health endpoint directly

```bash
curl -v http://localhost:8000/health
```

## Common Causes

1. **Application Crash**
   - Check logs for stack traces or error messages
   - Look for out-of-memory (OOM) errors
   - Check for configuration errors

2. **Resource Exhaustion**
   - CPU at 100%
   - Memory exhausted (OOM killed)
   - Disk full
   - File descriptor limit reached

3. **Database Connection Issues**
   - Database is down
   - Connection pool exhausted
   - Network connectivity problems

4. **Startup Failure**
   - Configuration validation failed
   - Missing required environment variables
   - TLS certificate issues

## Resolution Steps

### Quick Fix (Restart)

```bash
# Docker
docker-compose restart qbitel-engine

# Kubernetes
kubectl rollout restart deployment/qbitel-engine -n qbitel

# Systemd
systemctl restart qbitel
```

### If Restart Doesn't Help

1. **Check startup validation**:
```bash
# Docker
docker-compose logs qbitel-engine | grep "VALIDATION"

# Look for CRITICAL failures
```

2. **Verify environment variables**:
```bash
# Check required secrets are set
echo $DATABASE_PASSWORD
echo $JWT_SECRET
echo $ENCRYPTION_KEY
```

3. **Check database connectivity**:
```bash
# Test database connection
docker-compose exec postgres pg_isready -U qbitel_user -d qbitel
```

4. **Check disk space**:
```bash
df -h
```

5. **Check memory**:
```bash
free -h
```

## Escalation

If the issue persists after the above steps:

1. **Page on-call engineer** (if not already)
2. **Escalate to DevOps lead** after 15 minutes
3. **Escalate to CTO** if downtime exceeds 30 minutes

## Prevention

- Set up resource quotas and limits
- Implement auto-scaling
- Regular load testing
- Monitor resource usage trends
- Implement circuit breakers for external dependencies

## Related Alerts

- `HighMemoryUsage`
- `HighCPUUsage`
- `DatabaseConnectionFailures`

## Post-Incident

1. Write post-mortem
2. Add monitoring for root cause
3. Update this runbook with lessons learned
4. Consider implementing additional safeguards

## References

- [Deployment Guide](../../../docs/DEPLOYMENT.md)
- [Troubleshooting Guide](../../../docs/TROUBLESHOOTING.md)
- [Health Check Documentation](../../../ai_engine/api/k8s_health.py)
