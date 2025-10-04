# CRONOS AI - Production Mode Detection

## Overview

CRONOS AI implements comprehensive production mode detection to ensure that all security requirements are met before deployment to production environments. This document describes the production mode detection system and its requirements.

## Table of Contents

- [Environment Detection](#environment-detection)
- [Production Requirements](#production-requirements)
- [Development Mode](#development-mode)
- [Configuration](#configuration)
- [Validation](#validation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Environment Detection

### Detection Logic

Production mode is detected using environment variables in the following priority order:

1. `CRONOS_AI_ENVIRONMENT` (recommended)
2. `ENVIRONMENT` (fallback)

### Production Mode Values

The following values (case-insensitive) trigger production mode:

- `production`
- `prod`

### Example

```bash
# Set production mode
export CRONOS_AI_ENVIRONMENT=production

# Or using the fallback variable
export ENVIRONMENT=production
```

### Implementation

```python
from ai_engine.core.production_mode import is_production, get_environment

# Check if in production mode
if is_production():
    print("Running in production mode")

# Get current environment
env = get_environment()
print(f"Current environment: {env.value}")
```

---

## Production Requirements

### ✅ Required Secrets

All of the following secrets are **REQUIRED** in production mode:

#### 1. Database Password

- **Environment Variable**: `CRONOS_AI_DB_PASSWORD` or `DATABASE_PASSWORD`
- **Minimum Length**: 16 characters
- **Requirements**:
  - No common weak patterns (password, admin, test, etc.)
  - No sequential characters (abc, 123, etc.)
  - No excessive repeated characters (aaa, 111, etc.)
  - High entropy (diverse character set)

**Generate**:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

#### 2. Redis Password

- **Environment Variable**: `CRONOS_AI_REDIS_PASSWORD` or `REDIS_PASSWORD`
- **Minimum Length**: 16 characters
- **Requirements**: Same as database password
- **Critical**: Running Redis without authentication in production is a **CRITICAL security risk**

**Generate**:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

#### 3. JWT Secret

- **Environment Variable**: `CRONOS_AI_JWT_SECRET` or `JWT_SECRET`
- **Minimum Length**: 32 characters
- **Requirements**:
  - Cryptographically secure random generation
  - No weak patterns
  - High entropy
  - No sequential characters

**Generate**:
```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

#### 4. Encryption Key

- **Environment Variable**: `CRONOS_AI_ENCRYPTION_KEY` or `ENCRYPTION_KEY`
- **Minimum Length**: 32 characters
- **Requirements**:
  - Base64-encoded 256-bit key (when using Fernet)
  - Cryptographically secure
  - High entropy

**Generate**:
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### ✅ CORS Configuration

- **Wildcard (`*`) is FORBIDDEN** in production mode
- Must specify explicit allowed origins
- All origins must use HTTPS in production

**Example**:
```yaml
security:
  cors_origins:
    - "https://app.example.com"
    - "https://dashboard.example.com"
```

### ⚠️ Recommended (Optional)

#### API Key

- **Environment Variable**: `CRONOS_AI_API_KEY` or `API_KEY`
- **Minimum Length**: 32 characters
- **Format**: Should contain both letters and digits

**Generate**:
```bash
python -c "import secrets; print('cronos_' + secrets.token_urlsafe(32))"
```

---

## Development Mode

### Behavior

In development mode (default when `CRONOS_AI_ENVIRONMENT` is not set or set to `development`):

- ⚠️ Missing secrets generate **warnings** (not errors)
- ⚠️ Weak secrets generate **warnings** (not errors)
- ✅ CORS wildcard (`*`) is **allowed**
- ✅ Redis can run without authentication

### Use Cases

Development mode is appropriate for:

- Local development
- Testing environments
- CI/CD pipelines (non-production)
- Development containers

### Example

```bash
# Explicitly set development mode
export CRONOS_AI_ENVIRONMENT=development

# Or leave unset (defaults to development)
unset CRONOS_AI_ENVIRONMENT
```

---

## Configuration

### Environment Variables

Set all required environment variables before starting the application:

```bash
# Production Mode
export CRONOS_AI_ENVIRONMENT=production

# Required Secrets
export CRONOS_AI_DB_PASSWORD="<secure-password>"
export CRONOS_AI_REDIS_PASSWORD="<secure-password>"
export CRONOS_AI_JWT_SECRET="<secure-secret>"
export CRONOS_AI_ENCRYPTION_KEY="<secure-key>"

# Optional but Recommended
export CRONOS_AI_API_KEY="<secure-api-key>"
```

### Configuration Files

Update your production configuration file (`config/cronos_ai.production.yaml`):

```yaml
environment: production
debug: false

security:
  enable_encryption: true
  enable_cors: true
  cors_origins:
    - "https://app.example.com"
    - "https://dashboard.example.com"
  tls_enabled: true
  tls_verify_client: true
```

### Secrets Management

**Best Practices**:

1. **Never commit secrets to version control**
2. Use a secrets manager:
   - HashiCorp Vault
   - AWS Secrets Manager
   - Azure Key Vault
   - Google Secret Manager
3. Rotate secrets regularly
4. Use different secrets for each environment
5. Implement least privilege access

---

## Validation

### Automatic Validation

Production mode validation happens automatically during application startup:

```python
from ai_engine.core.config import Config

# This will validate all production requirements
config = Config.load_from_env()
```

### Manual Validation

Use the production readiness checker script:

```bash
# Check production readiness
python scripts/check_production_readiness.py
```

**Output Example**:
```
CRONOS AI - Production Readiness Checker
======================================================================

==================================================
Environment Mode Check
==================================================

  Current Environment: production
  Production Mode: True
✓ Running in PRODUCTION mode

==================================================
Required Secrets Check
==================================================

✓ Database Password is configured (CRONOS_AI_DB_PASSWORD)
✓ Redis Password is configured (CRONOS_AI_REDIS_PASSWORD)
✓ JWT Secret is configured (CRONOS_AI_JWT_SECRET)
✓ Encryption Key is configured (CRONOS_AI_ENCRYPTION_KEY)

==================================================
Production Readiness Summary
==================================================

Total Checks: 12
Passed: 12
Failed: 0
Warnings: 0

✓ PRODUCTION READY
All critical checks passed!
```

### Programmatic Validation

```python
from ai_engine.core.production_mode import validate_production_ready

# Validate production readiness
is_ready, errors = validate_production_ready()

if not is_ready:
    for error in errors:
        print(f"Error: {error}")
```

---

## Testing

### Unit Tests

Run production mode tests:

```bash
# Run all production mode tests
pytest ai_engine/tests/test_production_mode.py -v

# Run specific test class
pytest ai_engine/tests/test_production_mode.py::TestProductionModeDetection -v
```

### Integration Tests

Test with production configuration:

```bash
# Set production environment
export CRONOS_AI_ENVIRONMENT=production

# Set required secrets
export CRONOS_AI_DB_PASSWORD="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export CRONOS_AI_REDIS_PASSWORD="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export CRONOS_AI_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export CRONOS_AI_ENCRYPTION_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"

# Run tests
pytest ai_engine/tests/test_env_variable_loading.py -v
```

---

## Troubleshooting

### Common Issues

#### 1. Missing Database Password

**Error**:
```
ConfigurationException: Database password not configured!
REQUIRED: Set one of the following environment variables:
  - CRONOS_AI_DB_PASSWORD (recommended)
  - DATABASE_PASSWORD
```

**Solution**:
```bash
export CRONOS_AI_DB_PASSWORD="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

#### 2. Weak Password Detected

**Error**:
```
ConfigurationException: Database password validation failed:
  - Password contains weak patterns: password
```

**Solution**:
Generate a strong password using cryptographically secure random generation:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

#### 3. CORS Wildcard in Production

**Error**:
```
ConfigurationException: CORS wildcard (*) is FORBIDDEN in production mode!
```

**Solution**:
Update your configuration to specify explicit origins:
```yaml
security:
  cors_origins:
    - "https://app.example.com"
    - "https://dashboard.example.com"
```

#### 4. Redis Authentication Missing

**Error**:
```
ConfigurationException: Redis password not configured in PRODUCTION mode!
WARNING: Running Redis without authentication in production is a CRITICAL security risk!
```

**Solution**:
```bash
export CRONOS_AI_REDIS_PASSWORD="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

### Debug Mode

Enable debug logging to troubleshoot configuration issues:

```bash
export CRONOS_AI_LOG_LEVEL=DEBUG
python -m ai_engine
```

### Validation Script

Use the production readiness checker for detailed diagnostics:

```bash
python scripts/check_production_readiness.py
```

---

## CI/CD Integration

### Pre-Deployment Check

Add to your CI/CD pipeline:

```yaml
# .github/workflows/deploy.yml
- name: Check Production Readiness
  run: |
    python scripts/check_production_readiness.py
  env:
    CRONOS_AI_ENVIRONMENT: production
    CRONOS_AI_DB_PASSWORD: ${{ secrets.DB_PASSWORD }}
    CRONOS_AI_REDIS_PASSWORD: ${{ secrets.REDIS_PASSWORD }}
    CRONOS_AI_JWT_SECRET: ${{ secrets.JWT_SECRET }}
    CRONOS_AI_ENCRYPTION_KEY: ${{ secrets.ENCRYPTION_KEY }}
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set production mode
ENV CRONOS_AI_ENVIRONMENT=production

# Secrets should be injected at runtime
# Never bake secrets into the image!

CMD ["python", "-m", "ai_engine"]
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cronos-ai
spec:
  template:
    spec:
      containers:
      - name: cronos-ai
        env:
        - name: CRONOS_AI_ENVIRONMENT
          value: "production"
        - name: CRONOS_AI_DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: cronos-secrets
              key: db-password
        - name: CRONOS_AI_REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: cronos-secrets
              key: redis-password
        - name: CRONOS_AI_JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: cronos-secrets
              key: jwt-secret
        - name: CRONOS_AI_ENCRYPTION_KEY
          valueFrom:
            secretKeyRef:
              name: cronos-secrets
              key: encryption-key
```

---

## Security Best Practices

1. **Secrets Rotation**: Rotate all secrets regularly (every 90 days minimum)
2. **Least Privilege**: Grant minimum necessary permissions
3. **Audit Logging**: Enable comprehensive audit logging in production
4. **Monitoring**: Monitor for failed authentication attempts
5. **Encryption**: Enable encryption at rest and in transit
6. **TLS**: Use TLS 1.3 with strong cipher suites
7. **Network Isolation**: Use network policies to restrict access
8. **Regular Updates**: Keep dependencies up to date
9. **Security Scanning**: Run regular security scans
10. **Incident Response**: Have an incident response plan

---

## References

- [Environment Variable Configuration](./ENVIRONMENT_VARIABLE_CONFIGURATION.md)
- [Security & Secrets Management](./SECURITY_SECRETS_MANAGEMENT.md)
- [Production Readiness Assessment](../PRODUCTION_READINESS_ASSESSMENT.md)
- [Deployment Guide](./DEPLOYMENT.md)

---

## Support

For issues or questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Run the production readiness checker
3. Review the test suite
4. Contact the security team

---

**Last Updated**: 2025-10-01  
**Version**: 1.0.0