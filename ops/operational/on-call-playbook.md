# CRONOS AI - On-Call Playbook

## Table of Contents
1. [On-Call Overview](#on-call-overview)
2. [Alert Response Procedures](#alert-response-procedures)
3. [Incident Management](#incident-management)
4. [Escalation Procedures](#escalation-procedures)
5. [Common Issues and Solutions](#common-issues-and-solutions)
6. [Tools and Resources](#tools-and-resources)
7. [Post-Incident Procedures](#post-incident-procedures)

---

## On-Call Overview

### On-Call Responsibilities
- Monitor alerts and respond within SLA timeframes
- Investigate and resolve incidents
- Escalate when necessary
- Document all actions taken
- Participate in post-incident reviews

### Response Time SLAs
| Severity | Response Time | Resolution Target |
|----------|--------------|-------------------|
| P1 (Critical) | 15 minutes | 4 hours |
| P2 (High) | 30 minutes | 8 hours |
| P3 (Medium) | 2 hours | 24 hours |
| P4 (Low) | 8 hours | 72 hours |

### On-Call Schedule
- Primary on-call: 24/7 rotation (1 week)
- Secondary on-call: Backup support
- Escalation: Team lead and management

---

## Alert Response Procedures

### General Alert Response Flow
```
1. Acknowledge alert within SLA
2. Assess severity and impact
3. Check dashboards and metrics
4. Review recent changes/deployments
5. Execute appropriate runbook
6. Document actions in incident ticket
7. Communicate status to stakeholders
8. Resolve and close incident
```

### Alert Triage Checklist
- [ ] Alert acknowledged in monitoring system
- [ ] Incident ticket created
- [ ] Severity assessed correctly
- [ ] Impact scope determined
- [ ] Stakeholders notified (if P1/P2)
- [ ] Initial investigation started
- [ ] Runbook identified and followed

---

## Incident Management

### P1 (Critical) - Service Down

**Symptoms:**
- Complete service outage
- Error rate > 50%
- All health checks failing
- No successful requests

**Immediate Actions:**
1. **Acknowledge and Create Incident**
   ```bash
   # Create incident ticket
   python scripts/create_incident.py --severity P1 --title "Service Down"
   ```

2. **Check Service Status**
   ```bash
   # Check Kubernetes pods
   kubectl get pods -n cronos-ai
   kubectl describe pods -n cronos-ai
   
   # Check service endpoints
   kubectl get endpoints -n cronos-ai
   ```

3. **Review Recent Changes**
   ```bash
   # Check recent deployments
   kubectl rollout history deployment/cronos-ai -n cronos-ai
   
   # Check recent commits
   git log --oneline -10
   ```

4. **Check Logs**
   ```bash
   # View pod logs
   kubectl logs -n cronos-ai -l app=cronos-ai --tail=100
   
   # Check for errors
   kubectl logs -n cronos-ai -l app=cronos-ai | grep -i error
   ```

5. **Rollback if Needed**
   ```bash
   # Rollback to previous version
   kubectl rollout undo deployment/cronos-ai -n cronos-ai
   
   # Monitor rollback
   kubectl rollout status deployment/cronos-ai -n cronos-ai
   ```

6. **Verify Recovery**
   ```bash
   # Run health checks
   curl https://api.cronos-ai.com/health
   
   # Check metrics
   python scripts/check_production_readiness.py
   ```

**Escalation Triggers:**
- Unable to identify root cause within 30 minutes
- Rollback unsuccessful
- Multiple systems affected
- Data loss suspected

### P2 (High) - Degraded Performance

**Symptoms:**
- High latency (P95 > 1s)
- Error rate 5-10%
- Partial service degradation
- Some endpoints failing

**Investigation Steps:**
1. **Check Resource Utilization**
   ```bash
   # CPU and memory usage
   kubectl top pods -n cronos-ai
   kubectl top nodes
   ```

2. **Review Metrics**
   - Open Grafana SLO Dashboard
   - Check error rates by endpoint
   - Review latency distribution
   - Examine throughput trends

3. **Check Dependencies**
   ```bash
   # Database connectivity
   kubectl exec -it <pod-name> -n cronos-ai -- psql -h db-host -U user -c "SELECT 1"
   
   # Redis connectivity
   kubectl exec -it <pod-name> -n cronos-ai -- redis-cli -h redis-host ping
   ```

4. **Scale if Needed**
   ```bash
   # Scale up replicas
   kubectl scale deployment/cronos-ai -n cronos-ai --replicas=5
   
   # Verify scaling
   kubectl get pods -n cronos-ai -w
   ```

### P3 (Medium) - Minor Issues

**Symptoms:**
- Intermittent errors
- Elevated latency on specific endpoints
- Non-critical feature failures
- Warning-level alerts

**Standard Response:**
1. Document the issue
2. Investigate during business hours
3. Create bug ticket if needed
4. Monitor for escalation

---

## Escalation Procedures

### When to Escalate

**Immediate Escalation (P1):**
- Service down > 30 minutes
- Data loss or corruption
- Security breach suspected
- Unable to identify root cause

**Standard Escalation (P2/P3):**
- Issue persists beyond resolution target
- Requires specialized expertise
- Multiple failed resolution attempts
- Customer impact increasing

### Escalation Contacts

```yaml
Primary On-Call:
  - Phone: +1-XXX-XXX-XXXX
  - Slack: @oncall-primary
  - PagerDuty: Primary rotation

Secondary On-Call:
  - Phone: +1-XXX-XXX-XXXX
  - Slack: @oncall-secondary
  - PagerDuty: Secondary rotation

Team Lead:
  - Phone: +1-XXX-XXX-XXXX
  - Slack: @team-lead
  - Email: team-lead@cronos-ai.com

Engineering Manager:
  - Phone: +1-XXX-XXX-XXXX
  - Slack: @eng-manager
  - Email: eng-manager@cronos-ai.com

Security Team:
  - Phone: +1-XXX-XXX-XXXX
  - Slack: #security-incidents
  - Email: security@cronos-ai.com
```

---

## Common Issues and Solutions

### High Error Rate

**Quick Checks:**
```bash
# Check error distribution
kubectl logs -n cronos-ai -l app=cronos-ai | grep "ERROR" | tail -50

# Check specific error types
kubectl logs -n cronos-ai -l app=cronos-ai | grep "500" | wc -l
```

**Common Causes:**
- Database connection issues
- External API failures
- Resource exhaustion
- Configuration errors

**Solutions:**
1. Restart affected pods
2. Scale up resources
3. Check dependency health
4. Review recent config changes

### High Latency

**Investigation:**
```bash
# Check slow queries
kubectl exec -it <pod-name> -n cronos-ai -- python -c "
from ai_engine.monitoring import metrics
print(metrics.get_slow_queries())
"

# Check cache hit rate
curl https://api.cronos-ai.com/metrics | grep cache_hit_rate
```

**Common Causes:**
- Database query performance
- Cache misses
- Network issues
- Resource contention

**Solutions:**
1. Optimize slow queries
2. Warm up caches
3. Scale horizontally
4. Review query patterns

### Pod Crash Loop

**Diagnosis:**
```bash
# Get pod status
kubectl get pods -n cronos-ai

# Check pod events
kubectl describe pod <pod-name> -n cronos-ai

# View crash logs
kubectl logs <pod-name> -n cronos-ai --previous
```

**Common Causes:**
- Configuration errors
- Missing dependencies
- Resource limits too low
- Failed health checks

**Solutions:**
1. Fix configuration
2. Increase resource limits
3. Check dependencies
4. Review startup logs

### Database Connection Pool Exhausted

**Quick Fix:**
```bash
# Restart pods to reset connections
kubectl rollout restart deployment/cronos-ai -n cronos-ai

# Increase pool size (temporary)
kubectl set env deployment/cronos-ai -n cronos-ai DB_POOL_SIZE=50
```

**Long-term Solutions:**
- Optimize connection usage
- Implement connection pooling
- Scale database
- Review query patterns

---

## Tools and Resources

### Monitoring and Observability
- **Grafana:** https://grafana.cronos-ai.com
  - SLO Dashboard
  - Infrastructure Dashboard
  - Application Dashboard
- **Prometheus:** https://prometheus.cronos-ai.com
- **Kibana:** https://kibana.cronos-ai.com
- **Jaeger:** https://jaeger.cronos-ai.com

### Incident Management
- **PagerDuty:** https://cronos-ai.pagerduty.com
- **Jira:** https://cronos-ai.atlassian.net
- **Slack:** #incidents channel

### Documentation
- **Runbooks:** https://runbooks.cronos-ai.com
- **Architecture Docs:** https://docs.cronos-ai.com/architecture
- **API Docs:** https://docs.cronos-ai.com/api

### Useful Commands

```bash
# Quick health check
./scripts/check_production_readiness.py

# View recent deployments
kubectl rollout history deployment/cronos-ai -n cronos-ai

# Get pod logs
kubectl logs -n cronos-ai -l app=cronos-ai --tail=100 -f

# Execute command in pod
kubectl exec -it <pod-name> -n cronos-ai -- /bin/bash

# Port forward for debugging
kubectl port-forward -n cronos-ai <pod-name> 8080:8080

# Check resource usage
kubectl top pods -n cronos-ai
kubectl top nodes

# Describe resources
kubectl describe pod <pod-name> -n cronos-ai
kubectl describe service cronos-ai -n cronos-ai

# Get events
kubectl get events -n cronos-ai --sort-by='.lastTimestamp'
```

---

## Post-Incident Procedures

### Immediate Post-Incident (Within 24 hours)
1. **Update Incident Ticket**
   - Document resolution steps
   - Record root cause
   - Note time to resolution
   - List affected services

2. **Communicate Resolution**
   - Notify stakeholders
   - Update status page
   - Send incident summary

3. **Create Follow-up Tasks**
   - Bug fixes
   - Monitoring improvements
   - Documentation updates

### Post-Incident Review (Within 1 week)
1. **Schedule Review Meeting**
   - Include all responders
   - Invite relevant stakeholders
   - Book 1-hour session

2. **Prepare Review Document**
   ```markdown
   # Incident Post-Mortem
   
   ## Incident Summary
   - Date/Time:
   - Duration:
   - Severity:
   - Impact:
   
   ## Timeline
   - Detection:
   - Response:
   - Mitigation:
   - Resolution:
   
   ## Root Cause
   - What happened:
   - Why it happened:
   - Contributing factors:
   
   ## Resolution
   - Actions taken:
   - Why it worked:
   
   ## Lessons Learned
   - What went well:
   - What could be improved:
   
   ## Action Items
   - [ ] Item 1 (Owner, Due Date)
   - [ ] Item 2 (Owner, Due Date)
   ```

3. **Conduct Blameless Review**
   - Focus on systems, not people
   - Identify improvement opportunities
   - Create actionable items
   - Assign owners and deadlines

4. **Update Documentation**
   - Add to runbooks
   - Update troubleshooting guides
   - Improve monitoring/alerts
   - Share learnings with team

---

## Emergency Contacts

### Internal Teams
- **Engineering:** #engineering
- **DevOps:** #devops
- **Security:** #security
- **Product:** #product

### External Vendors
- **Cloud Provider:** support@cloud-provider.com
- **Database:** support@database-vendor.com
- **Monitoring:** support@monitoring-vendor.com

### Management
- **CTO:** cto@cronos-ai.com
- **VP Engineering:** vp-eng@cronos-ai.com

---

## Appendix

### Incident Severity Definitions

**P1 - Critical:**
- Complete service outage
- Data loss or corruption
- Security breach
- Revenue impact > $10K/hour

**P2 - High:**
- Major feature unavailable
- Significant performance degradation
- Affecting multiple customers
- Revenue impact $1K-$10K/hour

**P3 - Medium:**
- Minor feature issues
- Intermittent errors
- Single customer affected
- Workaround available

**P4 - Low:**
- Cosmetic issues
- Documentation errors
- Feature requests
- No customer impact

### On-Call Best Practices
1. Keep laptop and phone charged
2. Have reliable internet access
3. Test access to all tools before shift
4. Review recent changes at shift start
5. Familiarize yourself with current issues
6. Document everything
7. Don't hesitate to escalate
8. Take care of yourself
9. Hand off properly at shift end
10. Participate in post-incident reviews

---

**Last Updated:** 2025-01-02  
**Version:** 1.0  
**Owner:** DevOps Team  
**Review Frequency:** Quarterly