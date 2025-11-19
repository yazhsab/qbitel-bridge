# Production Dockerfile for Cronos xDS Server
# Multi-stage build for minimal attack surface and optimized image size

# Stage 1: Build environment
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /build

# Copy requirements first for layer caching
COPY ai_engine/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime environment
FROM python:3.11-slim

# Add labels
LABEL maintainer="Cronos AI <security@cronos.ai>"
LABEL description="Envoy xDS Server with Quantum-Safe Encryption"
LABEL version="1.0.0"
LABEL cronos.ai/quantum-safe="true"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r cronos && useradd -r -g cronos -u 1000 cronos

# Copy Python packages from builder
COPY --from=builder /root/.local /home/cronos/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY ai_engine/cloud_native/service_mesh/envoy/ ./service_mesh/envoy/
COPY ai_engine/cloud_native/service_mesh/istio/ ./service_mesh/istio/

# Copy startup script
COPY docker/xds-server/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create directories for certs and config
RUN mkdir -p /etc/cronos/xds /etc/cronos/certs /tmp && \
    chown -R cronos:cronos /app /etc/cronos /tmp

# Switch to non-root user
USER cronos

# Add Python packages to PATH
ENV PATH=/home/cronos/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 18000 9090 8081

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 18000)); s.close()" || exit 1

# Run xDS server
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "service_mesh.envoy.xds_server"]
