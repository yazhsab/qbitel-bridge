# QBITEL Bridge AI Engine - Legacy System Whisperer
# Production-ready Docker image

FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG VCS_REF

# Labels for container metadata
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="QBITEL Bridge Legacy System Whisperer" \
      org.label-schema.description="AI-powered legacy system management and predictive analytics" \
      org.label-schema.url="https://qbitel.com/bridge" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    pkg-config \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r qbitel && useradd -r -g qbitel qbitel

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./
COPY ai_engine/legacy/requirements.txt ./legacy-requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prod.txt && \
    pip install --no-cache-dir -r legacy-requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r qbitel && useradd -r -g qbitel qbitel

# Set work directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=qbitel:qbitel . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/config && \
    chown -R qbitel:qbitel /app

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Default configuration
ENV QBITEL_BRIDGE_ENVIRONMENT=production
ENV QBITEL_BRIDGE_LOG_LEVEL=INFO
ENV QBITEL_BRIDGE_DB_POOL_SIZE=20
ENV QBITEL_BRIDGE_REDIS_MAX_CONNECTIONS=100

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/legacy-system-whisperer/health || exit 1

# Switch to non-root user
USER qbitel

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "ai_engine.legacy.server", "--host", "0.0.0.0", "--port", "8000"]


# Development stage
FROM builder as development

# Install development dependencies
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    pytest-watch \
    pre-commit

# Copy application code
COPY --chown=qbitel:qbitel . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/config && \
    chown -R qbitel:qbitel /app

# Development environment variables
ENV QBITEL_BRIDGE_ENVIRONMENT=development
ENV QBITEL_BRIDGE_DEBUG=true
ENV QBITEL_BRIDGE_LOG_LEVEL=DEBUG
ENV PYTHONPATH=/app

# Switch to non-root user
USER qbitel

# Expose ports (main app + jupyter)
EXPOSE 8000 8888

# Development command with hot reload
CMD ["python", "-m", "ai_engine.legacy.dev_server", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# Testing stage
FROM development as testing

# Copy test configuration
COPY ai_engine/legacy/tests/ ./ai_engine/legacy/tests/
COPY ai_engine/legacy/pytest.ini ./

# Set test environment
ENV QBITEL_BRIDGE_ENVIRONMENT=testing
ENV QBITEL_BRIDGE_TESTING=true

# Run tests by default
CMD ["pytest", "ai_engine/legacy/tests/", "-v", "--cov=ai_engine/legacy", "--cov-report=html", "--cov-report=term"]