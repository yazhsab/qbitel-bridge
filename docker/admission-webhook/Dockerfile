# Production Dockerfile for Cronos Admission Webhook
# Multi-stage build with security hardening

# Stage 1: Build environment
FROM python:3.11-slim as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY ai_engine/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

LABEL maintainer="Cronos AI <security@cronos.ai>"
LABEL description="Kubernetes Admission Webhook with Quantum-Safe Image Verification"
LABEL version="1.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r cronos && useradd -r -g cronos -u 1000 cronos

COPY --from=builder /root/.local /home/cronos/.local

WORKDIR /app

COPY ai_engine/cloud_native/container_security/ ./container_security/
COPY docker/admission-webhook/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /etc/webhook/certs /etc/webhook/policy /tmp && \
    chown -R cronos:cronos /app /etc/webhook /tmp

USER cronos

ENV PATH=/home/cronos/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8443 9091

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('https://localhost:8443/healthz', timeout=5, context=__import__('ssl')._create_unverified_context())" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "container_security.admission_control.webhook_server"]
