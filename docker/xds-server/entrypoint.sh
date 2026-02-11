#!/bin/bash
set -e

echo "Starting Qbitel xDS Server..."
echo "Port: ${XDS_PORT:-18000}"
echo "ADS Enabled: ${ENABLE_ADS:-true}"
echo "Log Level: ${LOG_LEVEL:-info}"

# Check for required certificates
if [ ! -f "/etc/qbitel/certs/tls.crt" ]; then
    echo "WARNING: TLS certificate not found at /etc/qbitel/certs/tls.crt"
    echo "Running in insecure mode (not recommended for production)"
fi

# Set Python path
export PYTHONPATH=/app:$PYTHONPATH

# Execute the command
exec "$@"
