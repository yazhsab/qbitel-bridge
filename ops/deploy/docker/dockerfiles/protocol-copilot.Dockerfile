# Protocol Intelligence Copilot Production Dockerfile
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r qbitel && useradd -r -g qbitel -m qbitel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY ai_engine/requirements.txt ./
COPY requirements-copilot.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-copilot.txt

# Multi-stage build for production
FROM base as production

# Copy application code
COPY --chown=qbitel:qbitel ai_engine/ ./ai_engine/
COPY --chown=qbitel:qbitel config/ ./config/
COPY --chown=qbitel:qbitel docs/ ./docs/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache && \
    chown -R qbitel:qbitel /app/logs /app/data /app/cache

# Switch to non-root user
USER qbitel

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8001

# Default command
CMD ["python", "-m", "ai_engine.api.rest", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-asyncio pytest-benchmark black flake8 mypy

# Copy application code (development mounts will override)
COPY --chown=qbitel:qbitel . ./

USER qbitel

# Development command
CMD ["python", "-m", "ai_engine.api.rest", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Testing stage  
FROM development as testing

# Run tests
COPY tests/ ./tests/
RUN python -m pytest tests/ -v

# Security scanner stage
FROM production as security

USER root

# Install security scanning tools
RUN pip install safety bandit

# Run security scans
RUN safety check --json || true && \
    bandit -r ai_engine/ -f json || true

USER qbitel