# Environment Variable Configuration Guide

## Overview

QBITEL provides production-ready environment variable loading with comprehensive validation for all sensitive configuration values. This guide covers the supported environment variables, security requirements, and best practices.

## Table of Contents

- [Supported Environment Variables](#supported-environment-variables)
- [Security Requirements](#security-requirements)
- [Production vs Development Mode](#production-vs-development-mode)
- [Configuration Examples](#configuration-examples)
- [Validation Rules](#validation-rules)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Supported Environment Variables

### Database Configuration

| Variable | Alternative | Required | Description |
|----------|-------------|----------|-------------|
| `QBITEL_AI_DB_PASSWORD` | `DATABASE_PASSWORD` | Yes (Production) | PostgreSQL database password |
| `QBITEL_AI_DB_HOST` | - | No | Database host (default: localhost) |
| `QBITEL_AI_DB_PORT` | - | No | Database port (default: 5432) |
| `QBITEL_AI_DB_NAME` | - | No | Database name (default: qbitel) |
| `QBITEL_AI_DB_USER` | - | No | Database username (default: qbitel) |

**Priority:** `QBITEL_AI_*` prefixed variables take priority over generic alternatives.

### Redis Configuration

| Variable | Alternative | Required | Description |
|----------|-------------|----------|-------------|
| `QBITEL_AI_REDIS_PASSWORD` | `REDIS_PASSWORD` | Yes (Production) | Redis authentication password |
| `QBITEL_AI_REDIS_HOST` | - | No | Redis host (default: localhost) |
| `QBITEL_AI_REDIS_PORT` | - | No | Redis port (default: 6379) |

**Security Note:** Running Redis without authentication in production is a **CRITICAL** security risk.

### Security Configuration

| Variable | Alternative | Required | Description |
|----------|-------------|----------|-------------|
| `QBITEL_AI_JWT_SECRET` | `JWT_SECRET` | Yes (Production) | JWT token signing secret |
| `QBITEL_AI_ENCRYPTION_KEY` | `ENCRYPTION_KEY` | Yes (Production)* | Data encryption key |
| `QBITEL_AI_API_KEY` | `API_KEY` | No | API authentication key |

\* Required when encryption is enabled (default: enabled)

### Environment Control

| Variable | Alternative | Default | Description |
|----------|-------------|---------|-------------|
| `QBITEL_AI_ENVIRONMENT` | `ENVIRONMENT` | development | Runtime environment (development/staging/production) |

## Security Requirements

### Password Requirements

All passwords must meet the following criteria:

- **Minimum Length:** 16 characters
- **No Weak Patterns:** Cannot contain common words like 'password', 'admin', 'test', 'demo', '123456'
- **No Sequential Characters:** Cannot contain sequences like '123', 'abc', '456'
- **No Repeated Characters:** Cannot have more than 3 consecutive identical characters
- **Cryptographically Secure:** Must be generated using secure random methods

### JWT Secret Requirements

- **Minimum Length:** 32 characters
- **No Weak Patterns:** Cannot contain common words
- **High Entropy:** Must have sufficient randomness (>50% unique characters)
- **No Sequential Characters:** Cannot contain long sequential patterns

### Encryption Key Requirements

- **Minimum Length:** 32 characters
- **Format:** Base64-encoded 256-bit key (for Fernet encryption)
- **Cryptographically Secure:** Must be generated using proper key derivation

### API Key Requirements

- **Minimum Length:** 32 characters
- **Format:** Must contain both letters and digits
- **No Weak Patterns:** Cannot contain common words
- **Prefix Recommended:** Use a prefix like `qbitel_` for identification

## Production vs Development Mode

### Production Mode

Activated when `QBITEL_AI_ENVIRONMENT=production` or `ENVIRONMENT=production`

**Enforced Requirements:**
- ✅ Database password is **REQUIRED**
- ✅ Redis password is **REQUIRED**
- ✅ JWT secret is **REQUIRED**
- ✅ Encryption key is **REQUIRED** (when encryption enabled)
- ✅ All secrets must pass validation
- ✅ CORS wildcard (*) is **FORBIDDEN**
- ❌ Configuration errors cause immediate failure

### Development Mode

Activated when `QBITEL_AI_ENVIRONMENT=development` (default)

**Relaxed Requirements:**
- ⚠️ Missing secrets generate warnings (not errors)
- ⚠️ Weak secrets generate warnings (not errors)
- ✅ CORS wildcard allowed
- ✅ Redis can run without authentication

**Warning:** Development mode should **NEVER** be used in production environments.

## Configuration Examples

### Example 1: Production Environment Setup

```bash
# Generate secure secrets
export QBITEL_AI_DB_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export QBITEL_AI_REDIS_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export QBITEL_AI_JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
export QBITEL_AI_ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
export QBITEL_AI_API_KEY=$(python -c "import secrets; print('qbitel_' + secrets.token_urlsafe(32))")

# Set environment
export QBITEL_AI_ENVIRONMENT=production

# Start application
python -m ai_engine
```

### Example 2: Docker Compose

```yaml
version: '3.8'

services:
  qbitel:
    image: qbitel:latest
    environment:
      - QBITEL_AI_ENVIRONMENT=production
      - QBITEL_AI_DB_PASSWORD=${QBITEL_AI_DB_PASSWORD}
      - QBITEL_AI_REDIS_PASSWORD=${QBITEL_AI_REDIS_PASSWORD}
      - QBITEL_AI_JWT_SECRET=${QBITEL_AI_JWT_SECRET}
      - QBITEL_AI_ENCRYPTION_KEY=${QBITEL_AI_ENCRYPTION_KEY}
      - QBITEL_AI_API_KEY=${QBITEL_AI_API_KEY}
    env_file:
      - .env.production
```

### Example 3: Kubernetes Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: qbitel-secrets
type: Opaque
stringData:
  QBITEL_AI_DB_PASSWORD: "<base64-encoded-password>"
  QBITEL_AI_REDIS_PASSWORD: "<base64-encoded-password>"
  QBITEL_AI_JWT_SECRET: "<base64-encoded-secret>"
  QBITEL_AI_ENCRYPTION_KEY: "<base64-encoded-key>"
  QBITEL_AI_API_KEY: "<base64-encoded-key>"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qbitel
spec:
  template:
    spec:
      containers:
      - name: qbitel
        envFrom:
        - secretRef:
            name: qbitel-secrets
        env:
        - name: QBITEL_AI_ENVIRONMENT
          value: "production"
```

### Example 4: Development Environment

```bash
# Minimal setup for development
export QBITEL_AI_ENVIRONMENT=development

# Optional: Set passwords for testing
export QBITEL_AI_DB_PASSWORD="dev_password_min16chars"
export QBITEL_AI_REDIS_PASSWORD="dev_redis_min16chars"

# Start application (will use defaults for missing values)
python -m ai_engine
```

## Validation Rules

### Database Password Validation

```python
# ✅ Valid
QBITEL_AI_DB_PASSWORD="Kx9mP2nQ7vR4wS8tY3uZ1aB5cD6eF0gH"

# ❌ Invalid - Too short
QBITEL_AI_DB_PASSWORD="short123"

# ❌ Invalid - Contains weak pattern
QBITEL_AI_DB_PASSWORD="password123456789"

# ❌ Invalid - Sequential characters
QBITEL_AI_DB_PASSWORD="abcdefghijklmnop"

# ❌ Invalid - Repeated characters
QBITEL_AI_DB_PASSWORD="aaaaaaaaaaaaaaaa"
```

### JWT Secret Validation

```python
# ✅ Valid
QBITEL_AI_JWT_SECRET="vN8xM2kL9pQ4rS7tY1uZ3aB6cD5eF0gH2iJ4kL7mN9oP"

# ❌ Invalid - Too short
QBITEL_AI_JWT_SECRET="short_secret"

# ❌ Invalid - Contains weak pattern
QBITEL_AI_JWT_SECRET="secret123456789012345678901234567890"

# ❌ Invalid - Low entropy
QBITEL_AI_JWT_SECRET="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
```

### API Key Validation

```python
# ✅ Valid
QBITEL_AI_API_KEY="qbitel_Kx9mP2nQ7vR4wS8tY3uZ1aB5cD6eF0gH"

# ❌ Invalid - Too short
QBITEL_AI_API_KEY="qbitel_short"

# ❌ Invalid - No digits
QBITEL_AI_API_KEY="qbitel_abcdefghijklmnopqrstuvwxyz"

# ❌ Invalid - No letters
QBITEL_AI_API_KEY="qbitel_123456789012345678901234567890"
```

## Troubleshooting

### Error: "Database password not configured"

**Cause:** Missing database password in production mode.

**Solution:**
```bash
# Generate a secure password
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set the environment variable
export QBITEL_AI_DB_PASSWORD="<generated_password>"
```

### Error: "Password too short"

**Cause:** Password does not meet minimum length requirement (16 characters).

**Solution:**
```bash
# Generate a password that meets requirements
export QBITEL_AI_DB_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

### Error: "Password contains weak patterns"

**Cause:** Password contains common words like 'password', 'admin', 'test'.

**Solution:**
```bash
# Use cryptographically secure random generation
export QBITEL_AI_DB_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

### Error: "JWT secret not configured"

**Cause:** Missing JWT secret in production mode.

**Solution:**
```bash
# Generate a secure JWT secret
export QBITEL_AI_JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
```

### Error: "Encryption key not configured"

**Cause:** Missing encryption key when encryption is enabled in production.

**Solution:**
```bash
# Generate a Fernet encryption key
export QBITEL_AI_ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
```

### Error: "CORS wildcard (*) is not allowed in production mode"

**Cause:** CORS is configured with wildcard in production.

**Solution:**
```python
# In your configuration file, specify allowed origins
security:
  cors_origins:
    - "https://app.example.com"
    - "https://api.example.com"
```

## Best Practices

### 1. Use Secrets Management Systems

**Recommended:**
- HashiCorp Vault
- AWS Secrets Manager
- Azure Key Vault
- Google Cloud Secret Manager

**Example with Vault:**
```bash
# Store secrets in Vault
vault kv put secret/qbitel/production \
  db_password="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  redis_password="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  jwt_secret="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"

# Retrieve and export
export QBITEL_AI_DB_PASSWORD=$(vault kv get -field=db_password secret/qbitel/production)
```

### 2. Rotate Secrets Regularly

```bash
# Automated rotation script
#!/bin/bash
NEW_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Update in secrets manager
vault kv put secret/qbitel/production db_password="$NEW_PASSWORD"

# Update database
psql -c "ALTER USER qbitel WITH PASSWORD '$NEW_PASSWORD';"

# Restart application to pick up new secret
kubectl rollout restart deployment/qbitel
```

### 3. Never Commit Secrets to Version Control

```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo ".env.production" >> .gitignore
echo ".env.local" >> .gitignore
echo "*.key" >> .gitignore
echo "*.pem" >> .gitignore
```

### 4. Use Environment-Specific Files

```bash
# .env.development
QBITEL_AI_ENVIRONMENT=development
QBITEL_AI_DB_PASSWORD=dev_password_min16chars

# .env.production (never commit!)
QBITEL_AI_ENVIRONMENT=production
QBITEL_AI_DB_PASSWORD=<secure_generated_password>
QBITEL_AI_REDIS_PASSWORD=<secure_generated_password>
QBITEL_AI_JWT_SECRET=<secure_generated_secret>
QBITEL_AI_ENCRYPTION_KEY=<secure_generated_key>
```

### 5. Implement Secret Scanning

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

### 6. Monitor Secret Usage

```python
# Log secret access (without exposing values)
import logging

logger = logging.getLogger(__name__)

def load_secret(key: str) -> str:
    value = os.getenv(key)
    if value:
        logger.info(f"Loaded secret: {key} (length: {len(value)})")
    else:
        logger.warning(f"Secret not found: {key}")
    return value
```

### 7. Use Least Privilege Principle

```bash
# Create dedicated database user with minimal permissions
CREATE USER qbitel_app WITH PASSWORD '<secure_password>';
GRANT CONNECT ON DATABASE qbitel TO qbitel_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO qbitel_app;
```

### 8. Implement Secret Expiration

```python
# Track secret age
from datetime import datetime, timedelta

SECRET_MAX_AGE_DAYS = 90

def check_secret_age(secret_created_at: datetime) -> bool:
    age = datetime.utcnow() - secret_created_at
    if age > timedelta(days=SECRET_MAX_AGE_DAYS):
        logger.warning(f"Secret is {age.days} days old. Rotation recommended.")
        return False
    return True
```

## Security Checklist

Before deploying to production, ensure:

- [ ] All required environment variables are set
- [ ] All secrets meet minimum length requirements
- [ ] No weak patterns detected in any secrets
- [ ] Secrets are stored in a secure secrets management system
- [ ] Secrets are not committed to version control
- [ ] Secret rotation policy is in place
- [ ] Secret access is logged and monitored
- [ ] CORS is configured with specific origins (no wildcard)
- [ ] Database user has minimal required permissions
- [ ] Redis authentication is enabled
- [ ] TLS/SSL is enabled for all connections
- [ ] Environment is set to 'production'

## Additional Resources

- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [NIST Password Guidelines](https://pages.nist.gov/800-63-3/sp800-63b.html)
- [CIS Benchmarks](https://www.cisecurity.org/cis-benchmarks/)
- [QBITEL Security Documentation](./SECURITY_SECRETS_MANAGEMENT.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/yazhsab/qbitel-bridge/issues
- Security Issues: security@qbitel.com
- Documentation: https://docs.qbitel.com