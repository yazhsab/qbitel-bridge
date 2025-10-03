# CRONOS AI Engine - Production Dockerfile
# Multi-stage build for optimal image size and security

# Build stage
FROM python:3.9-slim as builder

# Set build arguments
ARG BUILD_VERSION=1.0.0
ARG BUILD_DATE
ARG BUILD_COMMIT

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create build user
RUN useradd --create-home --shell /bin/bash builder
USER builder
WORKDIR /home/builder

# Set up Python environment
ENV PYTHONPATH=/home/builder/app
ENV VIRTUAL_ENV=/home/builder/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt -r requirements-prod.txt

# Verify dependencies
COPY scripts/check_dependencies.py ./scripts/
RUN python scripts/check_dependencies.py || echo "Warning: Some optional dependencies missing"

# Copy application code
COPY --chown=builder:builder . ./app/

# Install application in editable mode
RUN cd app && pip install -e .

# Production stage
FROM python:3.9-slim

# Set metadata labels
LABEL maintainer="CRONOS AI Team <team@cronos-ai.com>"
LABEL version="${BUILD_VERSION}"
LABEL build-date="${BUILD_DATE}"
LABEL build-commit="${BUILD_COMMIT}"
LABEL description="CRONOS AI Engine - Enterprise Protocol Discovery and Analysis"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd --gid 1000 cronos && \
    useradd --uid 1000 --gid cronos --shell /bin/bash --create-home cronos

# Set up directories
RUN mkdir -p /app/logs /app/data /app/models /app/config && \
    chown -R cronos:cronos /app

# Switch to app user
USER cronos
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=cronos:cronos /home/builder/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
ENV PYTHONPATH=/app

# Copy application code
COPY --from=builder --chown=cronos:cronos /home/builder/app/ /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV AI_ENGINE_CONFIG_PATH=/app/config/production.yml
ENV AI_ENGINE_LOG_LEVEL=INFO
ENV AI_ENGINE_DATA_PATH=/app/data
ENV AI_ENGINE_MODEL_PATH=/app/models

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9090 50051

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Initialize directories\n\
mkdir -p /app/logs /app/data /app/models\n\
\n\
# Set default config if not provided\n\
if [ ! -f "$AI_ENGINE_CONFIG_PATH" ]; then\n\
    echo "No config file found at $AI_ENGINE_CONFIG_PATH, using defaults"\n\
fi\n\
\n\
# Run AI Engine\n\
exec python -m ai_engine "$@"\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]

# Default command
CMD ["--host", "0.0.0.0", "--port", "8000", "--enable-grpc", "--grpc-port", "50051"]