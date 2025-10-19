# CRONOS AI Docker Deployment

This directory contains the Docker Compose configuration for running CRONOS AI locally or in development environments.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- At least 8GB RAM available for Docker
- 20GB free disk space

## Quick Start

1. **Copy the environment file:**
   ```bash
   cp .env.docker .env
   ```

2. **Update the database password in `.env`:**
   ```bash
   # Edit .env and change DATABASE_PASSWORD to a strong password
   DATABASE_PASSWORD=your_strong_password_here
   ```

3. **Start all services:**
   ```bash
   docker-compose up -d
   ```

4. **Check service health:**
   ```bash
   docker-compose ps
   ```

5. **View logs:**
   ```bash
   docker-compose logs -f cronos-ai-engine
   ```

## Services

The Docker Compose stack includes the following services:

| Service | Port | Description |
|---------|------|-------------|
| **postgres** | 5432 | PostgreSQL 15 with TimescaleDB |
| **cronos-ai-engine** | 8000 (REST), 50051 (gRPC), 9090 (Metrics) | Main AI Engine |
| **redis** | 6379 | Redis cache |
| **mlflow** | 5000 | MLflow tracking server |
| **prometheus** | 9091 | Prometheus metrics collection |
| **grafana** | 3000 | Grafana dashboards |
| **nginx** | 80, 443 | Reverse proxy |

## Access Points

After starting the services, you can access:

- **REST API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:9090/metrics
- **Grafana**: http://localhost:3000 (admin/cronos-ai-admin)
- **Prometheus**: http://localhost:9091
- **MLflow**: http://localhost:5000

## Database Setup

The database is automatically initialized with:
- TimescaleDB extension
- UUID extension
- pgcrypto extension
- CRONOS AI schema

### Running Migrations

To apply database migrations:

```bash
# From the ai_engine directory
docker-compose exec cronos-ai-engine alembic upgrade head
```

### Accessing the Database

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U cronos_user -d cronos_ai

# Run a SQL query
docker-compose exec postgres psql -U cronos_user -d cronos_ai -c "SELECT version();"
```

## Data Persistence

Data is persisted in Docker volumes:

- `postgres_data`: Database files
- `ai_data`: AI engine data
- `ai_models`: ML model checkpoints
- `ai_logs`: Application logs
- `mlflow_data`: MLflow artifacts
- `prometheus_data`: Prometheus metrics
- `grafana_data`: Grafana dashboards
- `redis_data`: Redis persistence

### Backup Volumes

```bash
# Backup PostgreSQL data
docker run --rm -v cronos-ai_postgres_data:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres-backup.tar.gz /data

# Restore PostgreSQL data
docker run --rm -v cronos-ai_postgres_data:/data -v $(pwd):/backup ubuntu tar xzf /backup/postgres-backup.tar.gz -C /
```

## Production Deployment

**⚠️ WARNING**: This configuration is for development only.

For production deployment:

1. **Use strong passwords**: Update all default passwords
2. **Enable TLS/SSL**: Configure proper SSL certificates
3. **Use Docker secrets**: Don't use environment variables for sensitive data
4. **Configure resource limits**: Set appropriate CPU/memory limits
5. **Enable authentication**: Configure JWT, OAuth2, or SAML
6. **Set up backups**: Configure automated database backups
7. **Configure monitoring**: Set up proper alerting
8. **Use production configuration**: Update `AI_ENGINE_CONFIG_PATH` to production config

See [production deployment guide](../../../docs/DEPLOYMENT.md) for details.

## Troubleshooting

### Database Connection Errors

If the AI engine fails to connect to the database:

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check PostgreSQL logs
docker-compose logs postgres

# Verify database is ready
docker-compose exec postgres pg_isready -U cronos_user -d cronos_ai
```

### Reset Everything

To completely reset and start fresh:

```bash
# Stop all services
docker-compose down

# Remove all volumes (⚠️ THIS DELETES ALL DATA)
docker-compose down -v

# Start again
docker-compose up -d
```

### Check Service Health

```bash
# Check all service health statuses
docker-compose ps

# Individual health checks
curl http://localhost:8000/health
curl http://localhost:9091/-/healthy
curl http://localhost:3000/api/health
```

## Development

### Rebuilding the AI Engine

After making code changes:

```bash
# Rebuild and restart the AI engine
docker-compose up -d --build cronos-ai-engine

# View logs
docker-compose logs -f cronos-ai-engine
```

### Running Tests

```bash
# Run tests inside the container
docker-compose exec cronos-ai-engine pytest

# Run with coverage
docker-compose exec cronos-ai-engine pytest --cov=ai_engine --cov-report=html
```

### Accessing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f cronos-ai-engine

# Last 100 lines
docker-compose logs --tail=100 cronos-ai-engine
```

## Configuration

### Environment Variables

Key environment variables (set in `.env`):

- `DATABASE_NAME`: PostgreSQL database name
- `DATABASE_USER`: PostgreSQL username
- `DATABASE_PASSWORD`: PostgreSQL password
- `BUILD_VERSION`: Application version
- `BUILD_COMMIT`: Git commit hash

### Custom Configuration

To use a custom configuration file:

1. Place your config in `./config/custom.yml`
2. Update docker-compose.yml:
   ```yaml
   environment:
     - AI_ENGINE_CONFIG_PATH=/app/config/custom.yml
   ```

## Security Notes

1. **Change default passwords** in `.env`
2. **Grafana default credentials**: admin/cronos-ai-admin (change in production)
3. **Database is exposed on port 5432** (remove port mapping in production)
4. **Redis has no authentication** (configure in production)
5. **All services use HTTP** (enable HTTPS in production)

## Support

For issues or questions:
- Check logs: `docker-compose logs -f`
- Review documentation: `../../../docs/`
- Open an issue on GitHub

## License

Copyright (c) 2024 CRONOS AI. All rights reserved.
