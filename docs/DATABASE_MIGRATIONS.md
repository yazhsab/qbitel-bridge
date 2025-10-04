# Database Migrations Guide

## Overview

CRONOS AI Engine uses Alembic for database schema migrations, providing a robust and production-ready migration system with version control, rollback capabilities, and comprehensive testing procedures.

## Prerequisites

- PostgreSQL 12+ installed and running
- Python 3.9+ with required dependencies
- Database credentials configured via environment variables

## Environment Variables

Set the following environment variables before running migrations:

```bash
# Database Configuration
export CRONOS_AI_DB_HOST="localhost"
export CRONOS_AI_DB_PORT="5432"
export CRONOS_AI_DB_NAME="cronos_ai"
export CRONOS_AI_DB_USER="cronos"
export CRONOS_AI_DB_PASSWORD="<your-secure-password>"

# Optional: Use async mode for migrations
export ALEMBIC_USE_ASYNC="false"
```

## Installation

Install Alembic and dependencies:

```bash
cd ai_engine
pip install alembic asyncpg psycopg2-binary
```

## Migration Commands

### Initialize Database (First Time)

```bash
# Navigate to ai_engine directory
cd ai_engine

# Run initial migration to create all tables
alembic upgrade head
```

### Check Current Migration Status

```bash
alembic current
```

### View Migration History

```bash
alembic history --verbose
```

### Upgrade to Latest Version

```bash
alembic upgrade head
```

### Upgrade to Specific Version

```bash
alembic upgrade <revision_id>
```

### Downgrade One Version

```bash
alembic downgrade -1
```

### Downgrade to Specific Version

```bash
alembic downgrade <revision_id>
```

### Rollback All Migrations

```bash
alembic downgrade base
```

## Creating New Migrations

### Auto-generate Migration from Model Changes

```bash
alembic revision --autogenerate -m "Description of changes"
```

### Create Empty Migration

```bash
alembic revision -m "Description of changes"
```

### Review Generated Migration

Always review auto-generated migrations before applying:

```bash
cat alembic/versions/<revision_id>_description.py
```

## Migration Testing Procedures

### 1. Test in Development Environment

```bash
# Backup current database
pg_dump -U cronos cronos_ai > backup_before_migration.sql

# Run migration
alembic upgrade head

# Verify tables created
psql -U cronos -d cronos_ai -c "\dt"

# Test rollback
alembic downgrade -1

# Re-apply migration
alembic upgrade head
```

### 2. Test Data Integrity

```python
# test_migration.py
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from ai_engine.models.database import User, APIKey, AuditLog

async def test_migration():
    engine = create_async_engine(
        "postgresql+asyncpg://cronos:password@localhost/cronos_ai"
    )
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        # Test user creation
        from uuid import uuid4
        user = User(
            id=uuid4(),
            username="test_user",
            email="test@example.com",
            full_name="Test User",
            role="viewer"
        )
        session.add(user)
        await session.commit()
        
        print("✓ User creation successful")
        
        # Test API key creation
        api_key = APIKey(
            id=uuid4(),
            key_hash="test_hash",
            key_prefix="test_",
            name="Test Key",
            user_id=user.id
        )
        session.add(api_key)
        await session.commit()
        
        print("✓ API key creation successful")
        
        # Test audit log
        audit = AuditLog(
            id=uuid4(),
            action="user_created",
            user_id=user.id,
            success=True
        )
        session.add(audit)
        await session.commit()
        
        print("✓ Audit log creation successful")

if __name__ == "__main__":
    asyncio.run(test_migration())
```

Run the test:

```bash
python test_migration.py
```

### 3. Performance Testing

```bash
# Test migration performance
time alembic upgrade head

# Test rollback performance
time alembic downgrade -1
```

## Production Deployment

### Pre-Deployment Checklist

- [ ] Backup production database
- [ ] Test migration in staging environment
- [ ] Review all migration scripts
- [ ] Verify rollback procedures
- [ ] Schedule maintenance window
- [ ] Notify stakeholders

### Production Migration Steps

1. **Backup Database**

```bash
# Create timestamped backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
pg_dump -U cronos -h prod-db-host cronos_ai > backup_${TIMESTAMP}.sql

# Verify backup
pg_restore --list backup_${TIMESTAMP}.sql
```

2. **Enable Maintenance Mode**

```bash
# Set application to maintenance mode
kubectl scale deployment cronos-ai --replicas=0
```

3. **Run Migration**

```bash
# Set production environment
export CRONOS_AI_ENVIRONMENT=production
export CRONOS_AI_DB_HOST="prod-db-host"
export CRONOS_AI_DB_PASSWORD="<production-password>"

# Run migration
alembic upgrade head

# Verify migration
alembic current
```

4. **Verify Database State**

```bash
# Check tables
psql -U cronos -h prod-db-host -d cronos_ai -c "\dt"

# Check indexes
psql -U cronos -h prod-db-host -d cronos_ai -c "\di"

# Verify data integrity
psql -U cronos -h prod-db-host -d cronos_ai -c "SELECT COUNT(*) FROM users;"
```

5. **Disable Maintenance Mode**

```bash
# Restore application
kubectl scale deployment cronos-ai --replicas=3
```

### Rollback Procedure

If migration fails:

```bash
# Rollback migration
alembic downgrade -1

# Restore from backup if needed
psql -U cronos -h prod-db-host -d cronos_ai < backup_${TIMESTAMP}.sql

# Restart application
kubectl rollout restart deployment cronos-ai
```

## Migration Best Practices

### 1. Always Backup Before Migration

```bash
# Automated backup script
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/cronos_ai"
mkdir -p $BACKUP_DIR

pg_dump -U cronos cronos_ai | gzip > $BACKUP_DIR/backup_${TIMESTAMP}.sql.gz

# Keep last 30 days of backups
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete
```

### 2. Test Migrations in Staging

Always test migrations in a staging environment that mirrors production:

```bash
# Copy production data to staging
pg_dump -U cronos -h prod-db cronos_ai | psql -U cronos -h staging-db cronos_ai

# Test migration in staging
alembic upgrade head

# Verify application functionality
./run_integration_tests.sh
```

### 3. Use Transactions

Alembic migrations run in transactions by default. Ensure your migrations are idempotent:

```python
def upgrade():
    # Check if table exists before creating
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if 'new_table' not in inspector.get_table_names():
        op.create_table('new_table', ...)
```

### 4. Document Breaking Changes

Add comments to migrations that introduce breaking changes:

```python
"""Add user_role column

BREAKING CHANGE: This migration adds a required 'role' column.
Existing users will be assigned 'viewer' role by default.

Revision ID: 002_add_user_roles
Revises: 001_initial_auth
"""
```

### 5. Monitor Migration Performance

```bash
# Enable query logging
export ALEMBIC_LOG_LEVEL=DEBUG

# Run migration with timing
time alembic upgrade head 2>&1 | tee migration.log
```

## Troubleshooting

### Migration Fails with "relation already exists"

```bash
# Check current schema
psql -U cronos -d cronos_ai -c "\d"

# Mark migration as complete without running
alembic stamp head
```

### Database Connection Issues

```bash
# Test database connection
psql -U cronos -h localhost -d cronos_ai -c "SELECT version();"

# Check environment variables
env | grep CRONOS_AI_DB
```

### Rollback Fails

```bash
# Force rollback to specific version
alembic downgrade <revision_id> --sql > rollback.sql

# Review SQL
cat rollback.sql

# Apply manually if needed
psql -U cronos -d cronos_ai < rollback.sql
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Database Migration

on:
  push:
    branches: [main]
    paths:
      - 'ai_engine/alembic/versions/**'

jobs:
  migrate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install alembic psycopg2-binary
      
      - name: Run migrations
        env:
          CRONOS_AI_DB_HOST: ${{ secrets.DB_HOST }}
          CRONOS_AI_DB_PASSWORD: ${{ secrets.DB_PASSWORD }}
        run: |
          cd ai_engine
          alembic upgrade head
```

## Security Considerations

1. **Never commit database passwords** to version control
2. **Use environment variables** for all sensitive configuration
3. **Encrypt backups** before storing
4. **Audit migration logs** for security events
5. **Restrict migration permissions** to authorized personnel only

## Support

For migration issues:
- Check logs: `tail -f alembic.log`
- Review documentation: `docs/DATABASE_MIGRATIONS.md`
- Contact: devops@cronos-ai.com