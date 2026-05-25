FROM python:3.11-slim

WORKDIR /app

# Install demo dependencies
COPY demos/banking_security/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy AI engine and demo code
COPY ai_engine/ /app/ai_engine/
COPY demos/banking_security/ /app/demo/

# Environment
ENV QBITEL_PQC_ALLOW_FALLBACK=1
ENV QBITEL_ALLOW_EXPERIMENTAL_CRYPTO=1
ENV PYTHONPATH=/app

EXPOSE 8002

WORKDIR /app/demo/backend
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
