# CRONOS AI - Secrets Management Guide

## Overview

This document provides comprehensive guidance on managing secrets and credentials securely in CRONOS AI. Following these practices is **CRITICAL** for production deployments.

## Table of Contents

1. [Security Principles](#security-principles)
2. [Secrets Management Architecture](#secrets-management-architecture)
3. [Supported Backends](#supported-backends)
4. [Configuration](#configuration)
5. [Secret Rotation](#secret-rotation)
6. [Pre-commit Hooks](#pre-commit-hooks)
7. [Production Deployment](#production-deployment)
8. [Troubleshooting](#troubleshooting)

---

## Security Principles

### ❌ NEVER Do This

```python
# WRONG - Hardcoded credentials
password = "cronos123"
api_key = "sk-1234567890abcdef"
jwt_secret = "my-secret-key"
```

### ✅ ALWAYS Do This

```python
# CORRECT - Load from environment or secrets manager
password = os.getenv('DATABASE_PASSWORD')
api_key = get_secret('api_key')
jwt_secret = os.getenv('JWT_SECRET')
```

### Core Principles

1. **Never commit secrets to version control**
2. **Use environment variables or secrets managers**
3. **Rotate secrets regularly**
4. **Use strong, randomly generated secrets**
5. **Implement least privilege access**
6. **Audit secret access**
7. **Encrypt secrets at rest and in transit**

---

## Secrets Management Architecture

CRONOS AI provides a unified secrets management interface that supports multiple backends:

```
┌─────────────────────────────────────────┐
│         Application Layer               │
│  (Config, Auth, Database, etc.)         │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      SecretsManager (Unified API)       │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┬─────────────┬──────────┐
       │                │             │          │
┌──────▼─────┐  ┌──────▼──────┐  ┌──▼────┐  ┌──▼─────┐
│   Vault    │  │     AWS     │  │ Azure │  │  Env   │
│            │  │   Secrets   │  │  Key  │  │  Vars  │
│            │  │   Manager   │  │ Vault │  │        │
└────────────┘  └─────────────┘  └───────┘  └────────┘
```

---

## Supported Backends

### 1. Environment Variables (Recommended for Simple Deployments)

**Pros:**
- Simple to use
- No additional infrastructure
- Works everywhere

**Cons:**
- No centralized management
- No audit trail
- Manual rotation

**Setup:**

```bash
# Set environment variables
export CRONOS_AI_DB_PASSWORD="your-secure-password"
export CRONOS_AI_REDIS_PASSWORD="your-redis-password"
export CRONOS_AI_JWT_SECRET="your-jwt-secret-min-32-chars"
export CRONOS_AI_ENCRYPTION_KEY="your-encryption-key-min-32-chars"
export CRONOS_AI_API_KEY="your-api-key-min-32-chars"
```

**Configuration:**

```yaml
# config/cronos_ai.production.yaml
secrets_manager:
  enabled: false  # Use environment variables directly
```

### 2. HashiCorp Vault (Recommended for Production)

**Pros:**
- Centralized secret management
- Automatic rotation
- Audit logging
- Dynamic secrets
- Encryption as a service

**Cons:**
- Additional infrastructure
- Learning curve

**Setup:**

```bash
# Install Vault
# https://www.vaultproject.io/downloads

# Start Vault server
vault server -dev

# Set environment variables
export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_TOKEN='your-vault-token'

# Store secrets
vault kv put secret/cronos-ai/production/database_password value="your-password"
vault kv put secret/cronos-ai/production/jwt_secret value="your-jwt-secret"
```

**Configuration:**

```yaml
# config/cronos_ai.production.yaml
secrets_manager:
  enabled: true
  backend: "vault"
  vault_addr: "${VAULT_ADDR}"
  vault_token: "${VAULT_TOKEN}"
  vault_mount_point: "secret"
  vault_path: "cronos-ai/production"
```

**Python Usage:**

```python
from ai_engine.security.secrets_manager import get_secrets_manager

# Initialize
secrets_manager = get_secrets_manager({
    'backend': 'vault',
    'vault_addr': 'http://127.0.0.1:8200',
    'vault_token': 'your-token'
})

# Get secret
db_password = secrets_manager.get_secret('database_password')

# Set secret
secrets_manager.set_secret('api_key', 'new-api-key')
```

### 3. AWS Secrets Manager

**Pros:**
- Native AWS integration
- Automatic rotation
- Fine-grained IAM permissions
- Audit via CloudTrail

**Cons:**
- AWS-specific
- Additional cost

**Setup:**

```bash
# Install AWS CLI
pip install boto3

# Configure AWS credentials
aws configure

# Create secret
aws secretsmanager create-secret \
    --name cronos-ai/database_password \
    --secret-string "your-password"
```

**Configuration:**

```yaml
# config/cronos_ai.production.yaml
secrets_manager:
  enabled: true
  backend: "aws_secrets_manager"
  aws_region: "us-east-1"
```

**IAM Policy:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": "arn:aws:secretsmanager:*:*:secret:cronos-ai/*"
    }
  ]
}
```

### 4. Azure Key Vault

**Pros:**
- Native Azure integration
- Managed HSM support
- RBAC integration
- Audit logging

**Cons:**
- Azure-specific
- Additional cost

**Setup:**

```bash
# Install Azure CLI
pip install azure-keyvault-secrets azure-identity

# Login to Azure
az login

# Create Key Vault
az keyvault create --name cronos-ai-vault --resource-group myResourceGroup

# Set secret
az keyvault secret set --vault-name cronos-ai-vault \
    --name database-password --value "your-password"
```

**Configuration:**

```yaml
# config/cronos_ai.production.yaml
secrets_manager:
  enabled: true
  backend: "azure_key_vault"
  azure_vault_url: "https://cronos-ai-vault.vault.azure.net/"
```

---

## Configuration

### Required Secrets

| Secret | Environment Variable | Min Length | Rotation Interval |
|--------|---------------------|------------|-------------------|
| Database Password | `CRONOS_AI_DB_PASSWORD` | 16 chars | 90 days |
| Redis Password | `CRONOS_AI_REDIS_PASSWORD` | 16 chars | 90 days |
| JWT Secret | `CRONOS_AI_JWT_SECRET` | 32 chars | 180 days |
| Encryption Key | `CRONOS_AI_ENCRYPTION_KEY` | 32 chars | 365 days |
| API Key | `CRONOS_AI_API_KEY` | 32 chars | 90 days |

### Generating Strong Secrets

```bash
# Generate strong password (32 chars)
openssl rand -base64 32

# Generate API key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate JWT secret (64 chars)
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

### Environment File Template

Create `.env.production` (NEVER commit this file):

```bash
# Database
CRONOS_AI_DB_PASSWORD="<generate-strong-password>"

# Redis
CRONOS_AI_REDIS_PASSWORD="<generate-strong-password>"

# Security
CRONOS_AI_JWT_SECRET="<generate-jwt-secret-min-32-chars>"
CRONOS_AI_ENCRYPTION_KEY="<generate-encryption-key-min-32-chars>"
CRONOS_AI_API_KEY="<generate-api-key-min-32-chars>"

# Vault (if using)
VAULT_ADDR="https://vault.example.com"
VAULT_TOKEN="<vault-token>"
```

---

## Secret Rotation

### Automated Rotation Script

```bash
# Dry run (see what would be rotated)
python scripts/rotate_secrets.py --all --dry-run

# Rotate all secrets
python scripts/rotate_secrets.py --all --backend vault

# Rotate specific secret
python scripts/rotate_secrets.py --database --backend vault
python scripts/rotate_secrets.py --jwt --backend vault
python scripts/rotate_secrets.py --api-key --backend vault
```

### Manual Rotation Process

#### 1. Database Password

```bash
# Generate new password
NEW_PASSWORD=$(openssl rand -base64 32)

# Update in database
psql -U postgres -c "ALTER USER cronos_user WITH PASSWORD '$NEW_PASSWORD';"

# Update in secrets manager
vault kv put secret/cronos-ai/production/database_password value="$NEW_PASSWORD"

# Restart application
kubectl rollout restart deployment/cronos-ai
```

#### 2. JWT Secret

```bash
# Generate new secret
NEW_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(64))")

# Update in secrets manager
vault kv put secret/cronos-ai/production/jwt_secret value="$NEW_SECRET"

# Restart application (invalidates all existing tokens)
kubectl rollout restart deployment/cronos-ai
```

#### 3. API Key

```bash
# Generate new key
NEW_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Update in secrets manager
vault kv put secret/cronos-ai/production/api_key value="$NEW_KEY"

# Update client applications
# Restart application
kubectl rollout restart deployment/cronos-ai
```

### Rotation Schedule

| Secret | Frequency | Automation |
|--------|-----------|------------|
| Database Password | 90 days | Manual |
| Redis Password | 90 days | Manual |
| JWT Secret | 180 days | Automated |
| API Key | 90 days | Manual |
| Encryption Key | 365 days | Manual (requires re-encryption) |

---

## Pre-commit Hooks

### Installation

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### What Gets Checked

1. **Hardcoded passwords**
2. **API keys and tokens**
3. **AWS credentials**
4. **Database connection strings with passwords**
5. **JWT secrets**
6. **Encryption keys**
7. **Private keys**

### Bypassing Checks (Use Sparingly)

```bash
# Skip pre-commit hooks (NOT RECOMMENDED)
git commit --no-verify

# Skip specific hook
SKIP=detect-secrets git commit
```

---

## Production Deployment

### Checklist

- [ ] All secrets removed from code
- [ ] Environment variables configured
- [ ] Secrets manager configured (Vault/AWS/Azure)
- [ ] Pre-commit hooks installed
- [ ] Secrets rotation schedule established
- [ ] Backup of secrets stored securely
- [ ] Access controls configured
- [ ] Audit logging enabled
- [ ] Monitoring alerts configured
- [ ] Disaster recovery plan documented

### Kubernetes Deployment

```yaml
# secrets.yaml (create from template, don't commit)
apiVersion: v1
kind: Secret
metadata:
  name: cronos-ai-secrets
type: Opaque
stringData:
  database-password: "<from-secrets-manager>"
  redis-password: "<from-secrets-manager>"
  jwt-secret: "<from-secrets-manager>"
  encryption-key: "<from-secrets-manager>"
  api-key: "<from-secrets-manager>"
```

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
        - name: CRONOS_AI_DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: cronos-ai-secrets
              key: database-password
        - name: CRONOS_AI_JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: cronos-ai-secrets
              key: jwt-secret
```

### Docker Deployment

```bash
# Use environment file
docker run -d \
  --env-file .env.production \
  --name cronos-ai \
  cronos-ai:latest
```

---

## Troubleshooting

### Issue: "Database password not configured"

**Solution:**
```bash
export CRONOS_AI_DB_PASSWORD="your-password"
# or
export DATABASE_PASSWORD="your-password"
```

### Issue: "JWT secret not configured"

**Solution:**
```bash
export CRONOS_AI_JWT_SECRET="your-secret-min-32-chars"
# or
export JWT_SECRET="your-secret-min-32-chars"
```

### Issue: "Vault authentication failed"

**Solution:**
```bash
# Check Vault address
echo $VAULT_ADDR

# Check Vault token
vault token lookup

# Re-authenticate
vault login
```

### Issue: "AWS Secrets Manager access denied"

**Solution:**
```bash
# Check AWS credentials
aws sts get-caller-identity

# Verify IAM permissions
aws secretsmanager describe-secret --secret-id cronos-ai/database_password
```

---

## Security Best Practices

### 1. Principle of Least Privilege

- Grant minimum required permissions
- Use service accounts
- Implement RBAC

### 2. Defense in Depth

- Multiple layers of security
- Encryption at rest and in transit
- Network segmentation

### 3. Audit and Monitor

- Enable audit logging
- Monitor secret access
- Alert on anomalies

### 4. Regular Rotation

- Automate where possible
- Document rotation procedures
- Test rotation process

### 5. Secure Storage

- Never store secrets in:
  - Git repositories
  - Log files
  - Error messages
  - Environment variables (in production, use secrets manager)
  - Configuration files committed to version control

---

## Additional Resources

- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [HashiCorp Vault Documentation](https://www.vaultproject.io/docs)
- [AWS Secrets Manager Best Practices](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html)
- [Azure Key Vault Best Practices](https://docs.microsoft.com/en-us/azure/key-vault/general/best-practices)

---

## Support

For security issues, contact: security@cronos-ai.example.com

**DO NOT** post secrets or security issues in public forums or issue trackers.