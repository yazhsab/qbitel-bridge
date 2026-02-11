# QBITEL Zero-Touch Security Orchestrator - Kubernetes Deployment

This directory contains production-ready Kubernetes manifests for deploying the QBITEL Zero-Touch Security Orchestrator.

## Overview

The Zero-Touch Security Orchestrator provides autonomous security response capabilities with enterprise-grade resilience, legacy system awareness, and comprehensive integrations with external security systems.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    qbitel-security Namespace                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────┐  ┌────────────────────────────────┐ │
│  │   Security Orchestrator │  │        ConfigMaps              │ │
│  │   - Decision Engine     │  │  - enterprise-security-config │ │
│  │   - Threat Analyzer     │  │  - startup scripts             │ │
│  │   - Legacy Response     │  │  - health checks               │ │
│  │   - Integrations        │  └────────────────────────────────┘ │
│  │   - Resilience          │                                    │ │
│  └─────────────────────────┘                                    │ │
│              │                                                   │ │
│  ┌─────────────────────────┐  ┌────────────────────────────────┐ │
│  │       Services          │  │        Monitoring              │ │
│  │   - ClusterIP (8080)    │  │  - ServiceMonitor              │ │
│  │   - Health (8000)       │  │  - PrometheusRule              │ │
│  │   - Metrics (9090)      │  │  - Alerts                      │ │
│  │   - Headless Service    │  └────────────────────────────────┘ │
│  └─────────────────────────┘                                    │ │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Kubernetes cluster 1.20+
- Helm 3.0+
- Prometheus Operator (for monitoring)
- Redis cluster
- PostgreSQL database
- External integrations (SIEM, Ticketing, etc.)

## Quick Start

1. **Create namespace and apply RBAC:**
   ```bash
   kubectl apply -f namespace.yaml
   ```

2. **Create secrets:**
   ```bash
   # Database credentials
   kubectl create secret generic qbitel-db-credentials \
     --namespace=qbitel-security \
     --from-literal=username=qbitel_user \
     --from-literal=password=your_secure_password \
     --from-literal=database-url=postgresql://qbitel_user:your_secure_password@postgres:5432/qbitel_security

   # Redis credentials
   kubectl create secret generic qbitel-redis-credentials \
     --namespace=qbitel-security \
     --from-literal=url=redis://redis:6379/0

   # LLM API keys
   kubectl create secret generic qbitel-llm-credentials \
     --namespace=qbitel-security \
     --from-literal=openai-api-key=your_openai_key \
     --from-literal=anthropic-api-key=your_anthropic_key

   # Integration credentials
   kubectl create secret generic qbitel-integration-credentials \
     --namespace=qbitel-security \
     --from-literal=siem-endpoint=https://your-siem.com \
     --from-literal=siem-token=your_siem_token \
     --from-literal=servicenow-endpoint=https://your-servicenow.com \
     --from-literal=servicenow-token=your_servicenow_token \
     --from-literal=slack-token=your_slack_token \
     --from-literal=smtp-server=smtp.your-company.com \
     --from-literal=smtp-username=notifications@your-company.com \
     --from-literal=smtp-password=your_smtp_password \
     --from-literal=firewall-endpoint=https://your-firewall.com \
     --from-literal=firewall-token=your_firewall_token
   ```

3. **Deploy configuration:**
   ```bash
   kubectl apply -f configmap.yaml
   ```

4. **Deploy the application:**
   ```bash
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   kubectl apply -f hpa.yaml
   kubectl apply -f network-policy.yaml
   ```

5. **Deploy monitoring:**
   ```bash
   kubectl apply -f monitoring.yaml
   ```

6. **Verify deployment:**
   ```bash
   kubectl get pods -n qbitel-security
   kubectl get services -n qbitel-security
   kubectl logs -f deployment/qbitel-security-orchestrator -n qbitel-security
   ```

## Configuration

### Environment Variables

The deployment uses the following environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `REDIS_URL` | Redis connection string | Yes |
| `OPENAI_API_KEY` | OpenAI API key for LLM | Yes |
| `ANTHROPIC_API_KEY` | Anthropic API key for LLM fallback | No |
| `SIEM_ENDPOINT` | SIEM system endpoint | Yes |
| `SIEM_TOKEN` | SIEM authentication token | Yes |
| `SERVICENOW_ENDPOINT` | ServiceNow endpoint | Yes |
| `SERVICENOW_TOKEN` | ServiceNow authentication token | Yes |
| `SLACK_TOKEN` | Slack bot token | No |
| `SMTP_SERVER` | SMTP server for email notifications | No |
| `FIREWALL_ENDPOINT` | Firewall management endpoint | No |
| `POD_NAME` | Kubernetes pod name (auto-set) | Auto |
| `POD_NAMESPACE` | Kubernetes namespace (auto-set) | Auto |
| `POD_IP` | Pod IP address (auto-set) | Auto |

### ConfigMap Configuration

The main configuration is provided through the `qbitel-security-config` ConfigMap. Key sections:

- **Security Settings**: FIPS mode, compliance settings, security levels
- **Zero-Touch Configuration**: Autonomous mode, confidence thresholds, LLM settings
- **Legacy Systems**: Protocol-specific safety constraints
- **Integrations**: External system configurations
- **Resilience**: Circuit breakers, retry policies, health checks
- **Monitoring**: Metrics, logging, alerting thresholds

## Security Features

### Pod Security Context
- Non-root user execution
- Read-only root filesystem
- Dropped capabilities
- No privilege escalation

### Network Security
- Network policies for ingress/egress control
- Default deny-all policy
- Specific allow rules for required communications

### RBAC
- Service account with minimal permissions
- Role-based access control
- Secrets access limited to required resources

## Monitoring and Observability

### Health Endpoints
- **Liveness**: `/health/live` (port 8000)
- **Readiness**: `/health/ready` (port 8000)
- **Startup**: `/health/startup` (port 8000)
- **Detailed**: `/health/detailed` (port 8000)

### Metrics
- Prometheus metrics exposed on port 9090
- Custom metrics for security events, decisions, integrations
- Performance and resource utilization metrics

### Alerts
Predefined alerts for:
- High error rates
- High response times
- Low confidence decisions
- Circuit breaker states
- Service availability
- Resource utilization
- Integration failures

## Scaling and High Availability

### Horizontal Pod Autoscaler
- Scales from 2 to 10 replicas
- CPU target: 70%
- Memory target: 80%
- Custom metric: requests per second

### Pod Disruption Budget
- Minimum 1 replica available during updates
- Protects against voluntary disruptions

### Anti-Affinity
- Pods prefer different nodes
- Improves availability and fault tolerance

## Troubleshooting

### Common Issues

1. **Pods not starting:**
   ```bash
   kubectl describe pod <pod-name> -n qbitel-security
   kubectl logs <pod-name> -n qbitel-security
   ```

2. **Database connection issues:**
   ```bash
   # Check database connectivity
   kubectl exec -it <pod-name> -n qbitel-security -- python -c "
   import psycopg2
   import os
   conn = psycopg2.connect(os.environ['DATABASE_URL'])
   print('Database connection successful')
   "
   ```

3. **Redis connection issues:**
   ```bash
   # Check Redis connectivity
   kubectl exec -it <pod-name> -n qbitel-security -- redis-cli -u "${REDIS_URL}" ping
   ```

4. **Integration failures:**
   ```bash
   # Check integration status
   curl http://<service-ip>:8000/health/detailed
   ```

### Log Analysis

Logs are structured JSON format with the following fields:
- `timestamp`: ISO 8601 timestamp
- `level`: Log level (INFO, WARN, ERROR, DEBUG)
- `service`: Service name
- `component`: Component name
- `event_id`: Unique event identifier
- `message`: Log message
- `metadata`: Additional context

Example log query:
```bash
kubectl logs -f deployment/qbitel-security-orchestrator -n qbitel-security | jq '.level, .component, .message'
```

### Performance Monitoring

Monitor key metrics:
```bash
# Request rate
curl http://<service-ip>:9090/metrics | grep qbitel_security_requests_total

# Error rate
curl http://<service-ip>:9090/metrics | grep qbitel_security_errors_total

# Response time
curl http://<service-ip>:9090/metrics | grep qbitel_security_request_duration_seconds
```

## Deployment Variants

### Development
- Single replica
- Debug logging enabled
- External integrations disabled
- Relaxed security policies

### Staging
- 2 replicas
- Production configuration
- Test integrations enabled
- Full monitoring

### Production
- 3+ replicas
- High availability configuration
- All integrations enabled
- Strict security policies
- Full monitoring and alerting

## Backup and Recovery

### Database Backup
```bash
# Create database backup
kubectl exec -it postgres-pod -- pg_dump qbitel_security > backup.sql

# Restore from backup
kubectl exec -i postgres-pod -- psql qbitel_security < backup.sql
```

### Configuration Backup
```bash
# Backup all configurations
kubectl get configmap,secret -n qbitel-security -o yaml > qbitel-security-backup.yaml
```

## Updates and Rollbacks

### Rolling Updates
```bash
# Update deployment
kubectl set image deployment/qbitel-security-orchestrator \
  qbitel-security-orchestrator=qbitel/security-orchestrator:v2.0.0 \
  -n qbitel-security

# Check rollout status
kubectl rollout status deployment/qbitel-security-orchestrator -n qbitel-security
```

### Rollbacks
```bash
# Rollback to previous version
kubectl rollout undo deployment/qbitel-security-orchestrator -n qbitel-security

# Rollback to specific revision
kubectl rollout undo deployment/qbitel-security-orchestrator --to-revision=1 -n qbitel-security
```

## Support and Maintenance

### Health Checks
Regular health checks should be performed:
```bash
# Overall system health
curl http://<service-ip>:8000/health/detailed

# Integration health
curl http://<service-ip>:8000/health/integrations

# Performance metrics
curl http://<service-ip>:9090/metrics
```

### Log Rotation
Logs are automatically rotated by Kubernetes. For persistent storage:
```bash
# Configure log retention in cluster
kubectl patch configmap cluster-info --patch '{"data":{"log-retention":"30d"}}'
```

### Security Updates
Regular security updates should include:
- Base image updates
- Dependency updates
- Security configuration reviews
- Penetration testing

For support, contact the QBITEL Security team or refer to the main documentation.