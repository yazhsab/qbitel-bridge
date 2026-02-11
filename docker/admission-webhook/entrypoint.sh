#!/bin/bash
set -euo pipefail

echo "=== QBITEL Admission Webhook ==="
echo "Starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Validate certificate directory
CERT_DIR="${WEBHOOK_CERT_DIR:-/etc/webhook/certs}"
if [ -d "$CERT_DIR" ]; then
    echo "Certificate directory: $CERT_DIR"
    if [ -f "$CERT_DIR/tls.crt" ] && [ -f "$CERT_DIR/tls.key" ]; then
        echo "TLS certificates found"
    else
        echo "WARNING: TLS certificates not found in $CERT_DIR"
        echo "Webhook may fail to start without valid certificates"
    fi
else
    echo "WARNING: Certificate directory $CERT_DIR does not exist"
fi

# Validate policy directory
POLICY_DIR="${WEBHOOK_POLICY_DIR:-/etc/webhook/policy}"
if [ -d "$POLICY_DIR" ]; then
    echo "Policy directory: $POLICY_DIR"
else
    echo "INFO: Policy directory $POLICY_DIR does not exist, using defaults"
fi

echo "Launching webhook server..."
exec "$@"
