# QBITEL Bridge - Deployment Guide

## 🚀 **Production Deployment Guide**

This guide covers complete production deployment of the QBITEL Bridge System with enterprise-grade configurations, monitoring, and security.

## 📋 **Table of Contents**

- [Prerequisites](#prerequisites)
- [Infrastructure Requirements](#infrastructure-requirements)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Database Setup](#database-setup)
- [Security Configuration](#security-configuration)
- [Monitoring & Observability](#monitoring--observability)
- [Load Balancing](#load-balancing)
- [Backup & Recovery](#backup--recovery)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

## ✅ **Prerequisites**

### **System Requirements**

- **OS**: Linux (Ubuntu 20.04+ / RHEL 8+ / CentOS 8+)
- **CPU**: 8+ cores (16+ recommended for production)
- **RAM**: 16GB minimum (32GB+ recommended)
- **Disk**: 100GB SSD minimum (500GB+ recommended)
- **Network**: 1Gbps minimum bandwidth

### **Software Dependencies**

- **Python**: 3.10+
- **Rust**: 1.70+
- **Docker**: 20.10+
- **Kubernetes**: 1.25+ (if using K8s)
- **PostgreSQL**: 13+
- **Redis**: 6.0+

### **Hardware Accelerations (Optional)**

- **GPU**: NVIDIA GPU with CUDA 11.8+ for ML acceleration
- **GPU Memory**: 8GB+ VRAM recommended

## 🏗️ **Infrastructure Requirements**

### **Production Architecture**

```mermaid
graph TB
    LB[Load Balancer] --> API1[API Server 1]
    LB --> API2[API Server 2]  
    LB --> API3[API Server 3]
    
    API1 --> Redis[(Redis Cluster)]
    API2 --> Redis
    API3 --> Redis
    
    API1 --> DB[(PostgreSQL)]
    API2 --> DB
    API3 --> DB
    
    API1 --> ML[ML Models Storage]
    API2 --> ML
    API3 --> ML
    
    Mon[Monitoring Stack] --> API1
    Mon --> API2
    Mon --> API3
    Mon --> Redis
    Mon --> DB
```

### **Minimum Production Setup**

| Component | Instances | Specs | Purpose |
|-----------|-----------|-------|---------|
| **API Servers** | 3 | 4 CPU, 16GB RAM | Protocol discovery services |
| **PostgreSQL** | 2 (Primary + Replica) | 8 CPU, 32GB RAM, 500GB SSD | Data persistence |
| **Redis** | 3 (Cluster) | 2 CPU, 8GB RAM | Caching and session storage |
| **Load Balancer** | 2 (HA) | 2 CPU, 4GB RAM | Traffic distribution |
| **Monitoring** | 1 | 4 CPU, 8GB RAM | Prometheus, Grafana, AlertManager |

## 🐳 **Docker Deployment**

### **1. Build Images**

```bash
# Clone repository
git clone https://github.com/yazhsab/qbitel-bridge.git
cd qbitel-bridge

# Build production image
docker build -f ops/deploy/docker/dockerfiles/production.Dockerfile -t qbitel/qbitel-bridge:v2.1.0 .

# Tag for registry
docker tag qbitel/qbitel-bridge:v2.1.0 your-registry.com/qbitel/qbitel-bridge:v2.1.0

# Push to registry
docker push your-registry.com/qbitel/qbitel-bridge:v2.1.0
```

### **2. Docker Compose Setup**

Create `docker-compose.production.yml`:

```yaml
version: '3.8'

services:
  # Protocol Discovery API
  qbitel-api:
    image: your-registry.com/qbitel/qbitel-bridge:v2.1.0
    restart: unless-stopped
    ports:
      - "8080:8080"
      - "8000:8000"  # Prometheus metrics
    environment:
      - QBITEL_ENVIRONMENT=production
      - DATABASE_HOST=postgresql
      - DATABASE_PASSWORD=${DATABASE_PASSWORD}
      - REDIS_HOST=redis
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - JWT_SECRET=${JWT_SECRET}
      - ENCRYPTION_KEY=${ENCRYPTION_KEY}
    volumes:
      - ./config/qbitel.production.yaml:/app/config/qbitel.yaml:ro
      - ./ssl:/etc/ssl/certs:ro
      - qbitel-models:/opt/qbitel/models
      - qbitel-logs:/var/log/qbitel
    depends_on:
      - postgresql
      - redis
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          cpus: '2'
          memory: 8G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # PostgreSQL Database
  postgresql:
    image: postgres:15-alpine
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=qbitel_prod
      - POSTGRES_USER=qbitel_prod_user
      - POSTGRES_PASSWORD=${DATABASE_PASSWORD}
    volumes:
      - postgresql-data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 32G
        reservations:
          cpus: '4'
          memory: 16G
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U qbitel_prod_user"]
      interval: 30s
      timeout: 5s
      retries: 3

  # Redis Cache
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    ports:
      - "6379:6379"
    command: redis-server --requirepass ${REDIS_PASSWORD} --maxmemory 4gb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 8G
        reservations:
          cpus: '1'
          memory: 4G
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 30s
      timeout: 5s
      retries: 3

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./ops/observability/prometheus:/etc/prometheus:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  # Grafana
  grafana:
    image: grafana/grafana:latest
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana-data:/var/lib/grafana
      - ./ops/observability/grafana:/etc/grafana/provisioning:ro

  # Nginx Load Balancer
  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./ops/deploy/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl/certs:ro
    depends_on:
      - qbitel-api

volumes:
  postgresql-data:
  redis-data:
  prometheus-data:
  grafana-data:
  qbitel-models:
  qbitel-logs:

networks:
  default:
    name: qbitel-network
```

### **3. Environment Configuration**

Create `.env.production`:

```bash
# Database
DATABASE_PASSWORD=your_secure_db_password_here

# Redis
REDIS_PASSWORD=your_secure_redis_password_here

# Security
JWT_SECRET=your_jwt_secret_minimum_32_characters_long
ENCRYPTION_KEY=your_encryption_key_minimum_32_characters_long

# Monitoring
GRAFANA_PASSWORD=your_secure_grafana_password
```

### **4. Deploy with Docker Compose**

```bash
# Deploy the stack
docker-compose -f docker-compose.production.yml --env-file .env.production up -d

# Check status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f qbitel-api

# Scale API services
docker-compose -f docker-compose.production.yml up -d --scale qbitel-api=5
```

## ☸️ **Kubernetes Deployment**

### **1. Create Namespace**

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: qbitel-prod
  labels:
    name: qbitel-prod
```

### **2. Secrets Management**

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: qbitel-secrets
  namespace: qbitel-prod
type: Opaque
stringData:
  database-password: "your_secure_db_password"
  redis-password: "your_secure_redis_password"
  jwt-secret: "your_jwt_secret_minimum_32_characters"
  encryption-key: "your_encryption_key_minimum_32_characters"
```

### **3. ConfigMaps**

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: qbitel-config
  namespace: qbitel-prod
data:
  qbitel.yaml: |
    environment: production
    debug: false
    api_host: "0.0.0.0"
    api_port: 8080
    
    database:
      host: "postgresql-service"
      port: 5432
      database: "qbitel_prod"
      username: "qbitel_prod_user"
      connection_pool_size: 20
    
    redis:
      host: "redis-service"
      port: 6379
      database: 0
      connection_pool_size: 20
    
    # ... rest of production config
```

### **4. PostgreSQL Deployment**

```yaml
# postgresql.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgresql
  namespace: qbitel-prod
spec:
  serviceName: postgresql-service
  replicas: 1
  selector:
    matchLabels:
      app: postgresql
  template:
    metadata:
      labels:
        app: postgresql
    spec:
      containers:
      - name: postgresql
        image: postgres:15
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: "qbitel_prod"
        - name: POSTGRES_USER
          value: "qbitel_prod_user"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: qbitel-secrets
              key: database-password
        volumeMounts:
        - name: postgresql-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
          limits:
            memory: "32Gi"
            cpu: "8"
  volumeClaimTemplates:
  - metadata:
      name: postgresql-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 500Gi

---
apiVersion: v1
kind: Service
metadata:
  name: postgresql-service
  namespace: qbitel-prod
spec:
  selector:
    app: postgresql
  ports:
  - port: 5432
    targetPort: 5432
```

### **5. Redis Deployment**

```yaml
# redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: qbitel-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        command: 
        - redis-server
        - --requirepass
        - $(REDIS_PASSWORD)
        - --maxmemory
        - 4gb
        - --maxmemory-policy
        - allkeys-lru
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: qbitel-secrets
              key: redis-password
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
          limits:
            memory: "8Gi" 
            cpu: "2"

---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: qbitel-prod
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
```

### **6. Main Application Deployment**

```yaml
# qbitel-api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qbitel-api
  namespace: qbitel-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: qbitel-api
  template:
    metadata:
      labels:
        app: qbitel-api
    spec:
      containers:
      - name: qbitel-api
        image: your-registry.com/qbitel/qbitel-bridge:v2.1.0
        ports:
        - containerPort: 8080
        - containerPort: 8000
        env:
        - name: QBITEL_ENVIRONMENT
          value: "production"
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: qbitel-secrets
              key: database-password
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: qbitel-secrets
              key: redis-password
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: qbitel-secrets
              key: jwt-secret
        - name: ENCRYPTION_KEY
          valueFrom:
            secretKeyRef:
              name: qbitel-secrets
              key: encryption-key
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: model-storage
          mountPath: /opt/qbitel/models
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: config-volume
        configMap:
          name: qbitel-config
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: qbitel-api-service
  namespace: qbitel-prod
spec:
  selector:
    app: qbitel-api
  ports:
  - name: api
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 8000
    targetPort: 8000
  type: ClusterIP

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage-pvc
  namespace: qbitel-prod
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
```

### **7. Ingress Configuration**

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: qbitel-ingress
  namespace: qbitel-prod
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.qbitel.yourdomain.com
    secretName: qbitel-tls
  rules:
  - host: api.qbitel.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: qbitel-api-service
            port:
              number: 8080
```

### **8. Deploy to Kubernetes**

```bash
# Apply all configurations
kubectl apply -f namespace.yaml
kubectl apply -f secrets.yaml
kubectl apply -f configmap.yaml
kubectl apply -f postgresql.yaml
kubectl apply -f redis.yaml
kubectl apply -f qbitel-api.yaml
kubectl apply -f ingress.yaml

# Check deployment status
kubectl get pods -n qbitel-prod
kubectl get services -n qbitel-prod
kubectl get ingress -n qbitel-prod

# Scale deployment
kubectl scale deployment qbitel-api --replicas=5 -n qbitel-prod

# View logs
kubectl logs -f deployment/qbitel-api -n qbitel-prod
```

## 🗄️ **Database Setup**

### **1. PostgreSQL Initialization**

```sql
-- Create database and user
CREATE DATABASE qbitel_prod;
CREATE USER qbitel_prod_user WITH ENCRYPTED PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE qbitel_prod TO qbitel_prod_user;

-- Connect to qbitel_prod database
\c qbitel_prod;

-- Create tables
CREATE TABLE protocols (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    confidence FLOAT NOT NULL,
    grammar_rules JSONB NOT NULL,
    parser_config JSONB NOT NULL,
    validation_rules JSONB NOT NULL,
    statistics JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE discovery_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    traffic_data_hash VARCHAR(64) NOT NULL,
    discovered_protocols JSONB NOT NULL,
    processing_time FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT NOT NULL,
    labels JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_protocols_name ON protocols(name);
CREATE INDEX idx_protocols_confidence ON protocols(confidence);
CREATE INDEX idx_discovery_sessions_hash ON discovery_sessions(traffic_data_hash);
CREATE INDEX idx_metrics_name_timestamp ON metrics(metric_name, timestamp);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO qbitel_prod_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO qbitel_prod_user;
```

### **2. Database Backup Script**

```bash
#!/bin/bash
# backup-db.sh

DB_HOST=${DATABASE_HOST:-localhost}
DB_NAME=${DATABASE_NAME:-qbitel_prod}
DB_USER=${DATABASE_USER:-qbitel_prod_user}
BACKUP_DIR=${BACKUP_DIR:-/var/backups/qbitel}
RETENTION_DAYS=${RETENTION_DAYS:-30}

# Create backup directory
mkdir -p $BACKUP_DIR

# Create backup
BACKUP_FILE="$BACKUP_DIR/qbitel_backup_$(date +%Y%m%d_%H%M%S).sql"

echo "Creating database backup..."
PGPASSWORD=$DATABASE_PASSWORD pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > $BACKUP_FILE

if [ $? -eq 0 ]; then
    echo "Backup created successfully: $BACKUP_FILE"
    
    # Compress backup
    gzip $BACKUP_FILE
    echo "Backup compressed: $BACKUP_FILE.gz"
    
    # Remove old backups
    find $BACKUP_DIR -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
    echo "Old backups cleaned up (older than $RETENTION_DAYS days)"
else
    echo "Backup failed!"
    exit 1
fi
```

## 🔒 **Security Configuration**

### **1. SSL/TLS Setup**

```bash
# Generate SSL certificates (for testing - use proper CA in production)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/qbitel.key \
  -out ssl/qbitel.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=api.qbitel.yourdomain.com"

# Set proper permissions
chmod 600 ssl/qbitel.key
chmod 644 ssl/qbitel.crt
```

### **2. API Key Management**

```python
# Generate API keys
import secrets
import hashlib

def generate_api_key():
    """Generate a secure API key."""
    raw_key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    return raw_key, key_hash

# Generate keys for different services
admin_key, admin_hash = generate_api_key()
service_key, service_hash = generate_api_key()

print(f"Admin API Key: {admin_key}")
print(f"Admin Hash: {admin_hash}")
print(f"Service API Key: {service_key}")
print(f"Service Hash: {service_hash}")
```

### **3. Firewall Configuration**

```bash
# Ubuntu/Debian firewall setup
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH
sudo ufw allow ssh

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow API port (if needed externally)
sudo ufw allow 8080/tcp

# Allow metrics port (internal only)
sudo ufw allow from 10.0.0.0/8 to any port 8000

# Reload firewall
sudo ufw reload
```

## 📊 **Monitoring & Observability**

### **1. Prometheus Configuration**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules.yaml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'qbitel-api'
    static_configs:
      - targets: ['qbitel-api:8000']
    scrape_interval: 5s
    metrics_path: '/metrics'
    
  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
      
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

### **2. Alerting Rules**

```yaml
# rules.yaml
groups:
- name: qbitel-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(protocol_discovery_errors_total[5m]) > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: High error rate detected
      description: "Error rate is {{ $value }} errors per second"
      
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(protocol_discovery_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High response time
      description: "95th percentile response time is {{ $value }} seconds"
      
  - alert: DatabaseDown
    expr: up{job="postgresql"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: Database is down
      description: "PostgreSQL database is not responding"
      
  - alert: RedisDown
    expr: up{job="redis"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: Redis is down
      description: "Redis cache is not responding"
```

### **3. Grafana Dashboard**

```json
{
  "dashboard": {
    "title": "QBITEL Bridge",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(protocol_discovery_requests_total[1m])",
            "legendFormat": "{{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(protocol_discovery_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(protocol_discovery_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "system_cpu_usage_percent",
            "legendFormat": "CPU %"
          },
          {
            "expr": "system_memory_usage_percent", 
            "legendFormat": "Memory %"
          }
        ]
      }
    ]
  }
}
```

## ⚖️ **Load Balancing**

### **1. Nginx Configuration**

```nginx
# nginx.conf
upstream qbitel_api {
    server qbitel-api-1:8080 weight=1 max_fails=3 fail_timeout=30s;
    server qbitel-api-2:8080 weight=1 max_fails=3 fail_timeout=30s;
    server qbitel-api-3:8080 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.qbitel.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.qbitel.yourdomain.com;
    
    ssl_certificate /etc/ssl/certs/qbitel.crt;
    ssl_certificate_key /etc/ssl/certs/qbitel.key;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://qbitel_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Health check
        proxy_next_upstream error timeout invalid_header http_502 http_503 http_504;
    }
    
    location /metrics {
        # Restrict metrics access
        allow 10.0.0.0/8;
        deny all;
        
        proxy_pass http://qbitel_api;
        proxy_set_header Host $host;
    }
    
    location /health {
        access_log off;
        proxy_pass http://qbitel_api;
    }
}
```

## 💾 **Backup & Recovery**

### **1. Automated Backup Script**

```bash
#!/bin/bash
# comprehensive-backup.sh

set -euo pipefail

# Configuration
BACKUP_ROOT="/var/backups/qbitel"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30
S3_BUCKET="qbitel-backups"

# Create backup directory
mkdir -p "$BACKUP_ROOT/$DATE"

echo "Starting comprehensive backup for QBITEL..."

# 1. Database backup
echo "Backing up PostgreSQL database..."
PGPASSWORD=$DATABASE_PASSWORD pg_dump \
  -h $DATABASE_HOST \
  -U $DATABASE_USER \
  -d $DATABASE_NAME \
  --verbose \
  --no-owner \
  --no-privileges > "$BACKUP_ROOT/$DATE/database.sql"

# 2. Redis backup
echo "Backing up Redis data..."
redis-cli -h $REDIS_HOST -a $REDIS_PASSWORD --rdb "$BACKUP_ROOT/$DATE/redis.rdb"

# 3. ML Models backup
echo "Backing up ML models..."
tar -czf "$BACKUP_ROOT/$DATE/models.tar.gz" /opt/qbitel/models/

# 4. Configuration backup
echo "Backing up configuration files..."
tar -czf "$BACKUP_ROOT/$DATE/config.tar.gz" \
  /app/config/ \
  /etc/ssl/certs/qbitel.* \
  /etc/nginx/sites-available/qbitel

# 5. Create manifest
echo "Creating backup manifest..."
cat > "$BACKUP_ROOT/$DATE/manifest.json" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "backup_id": "$DATE",
  "components": {
    "database": "database.sql",
    "redis": "redis.rdb", 
    "models": "models.tar.gz",
    "config": "config.tar.gz"
  },
  "checksums": {
    "database": "$(sha256sum $BACKUP_ROOT/$DATE/database.sql | cut -d' ' -f1)",
    "redis": "$(sha256sum $BACKUP_ROOT/$DATE/redis.rdb | cut -d' ' -f1)",
    "models": "$(sha256sum $BACKUP_ROOT/$DATE/models.tar.gz | cut -d' ' -f1)",
    "config": "$(sha256sum $BACKUP_ROOT/$DATE/config.tar.gz | cut -d' ' -f1)"
  }
}
EOF

# 6. Create archive
echo "Creating backup archive..."
cd "$BACKUP_ROOT"
tar -czf "qbitel_backup_$DATE.tar.gz" "$DATE/"

# 7. Upload to S3 (if configured)
if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
  echo "Uploading backup to S3..."
  aws s3 cp "qbitel_backup_$DATE.tar.gz" "s3://$S3_BUCKET/backups/"
fi

# 8. Cleanup old backups
echo "Cleaning up old backups..."
find "$BACKUP_ROOT" -name "qbitel_backup_*.tar.gz" -mtime +$RETENTION_DAYS -delete
rm -rf "$BACKUP_ROOT/$DATE"

echo "Backup completed successfully: qbitel_backup_$DATE.tar.gz"
```

### **2. Recovery Script**

```bash
#!/bin/bash
# recovery.sh

set -euo pipefail

BACKUP_FILE="$1"
RECOVERY_DIR="/tmp/qbitel_recovery"

if [[ -z "$BACKUP_FILE" ]]; then
  echo "Usage: $0 <backup_file>"
  exit 1
fi

echo "Starting recovery from backup: $BACKUP_FILE"

# Extract backup
mkdir -p "$RECOVERY_DIR"
tar -xzf "$BACKUP_FILE" -C "$RECOVERY_DIR"

# Find backup directory
BACKUP_DIR=$(find "$RECOVERY_DIR" -type d -name "20*" | head -1)

if [[ ! -d "$BACKUP_DIR" ]]; then
  echo "Error: Could not find backup directory"
  exit 1
fi

# Validate manifest
if [[ ! -f "$BACKUP_DIR/manifest.json" ]]; then
  echo "Error: Missing backup manifest"
  exit 1
fi

echo "Validating backup integrity..."
cd "$BACKUP_DIR"

# Check checksums
for component in database redis models config; do
  expected_checksum=$(jq -r ".checksums.$component" manifest.json)
  actual_checksum=$(sha256sum $(jq -r ".components.$component" manifest.json) | cut -d' ' -f1)
  
  if [[ "$expected_checksum" != "$actual_checksum" ]]; then
    echo "Error: Checksum mismatch for $component"
    exit 1
  fi
done

echo "Backup integrity verified. Proceeding with recovery..."

# 1. Restore database
echo "Restoring PostgreSQL database..."
PGPASSWORD=$DATABASE_PASSWORD psql \
  -h $DATABASE_HOST \
  -U $DATABASE_USER \
  -d $DATABASE_NAME \
  -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

PGPASSWORD=$DATABASE_PASSWORD psql \
  -h $DATABASE_HOST \
  -U $DATABASE_USER \
  -d $DATABASE_NAME \
  < database.sql

# 2. Restore Redis
echo "Restoring Redis data..."
redis-cli -h $REDIS_HOST -a $REDIS_PASSWORD FLUSHALL
redis-cli -h $REDIS_HOST -a $REDIS_PASSWORD --pipe < redis.rdb

# 3. Restore ML models
echo "Restoring ML models..."
tar -xzf models.tar.gz -C /

# 4. Restore configuration
echo "Restoring configuration..."
tar -xzf config.tar.gz -C /

# Cleanup
rm -rf "$RECOVERY_DIR"

echo "Recovery completed successfully!"
echo "Please restart the QBITEL services to apply the restored configuration."
```

## 🚀 **Performance Tuning**

### **1. System-Level Optimizations**

```bash
# /etc/sysctl.conf optimizations
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 65535
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_keepalive_time = 1200
net.ipv4.tcp_keepalive_intvl = 30
net.ipv4.tcp_keepalive_probes = 3

# Apply changes
sudo sysctl -p
```

### **2. Application Tuning**

```yaml
# Production configuration tuning
performance:
  cache_size: 50000
  cache_ttl: 7200
  use_redis_cache: true
  
  # Memory management
  max_memory_usage_mb: 16384
  gc_threshold: 0.85
  
  # Threading
  max_threads: 32
  io_thread_pool_size: 100
  cpu_thread_pool_size: 16
  
  # Batching  
  batch_size: 2000
  batch_flush_interval: 0.5

ai_engine:
  # GPU optimization
  use_gpu: true
  gpu_memory_limit: 0.95
  batch_size: 128
  
  # Model optimization
  ensemble_size: 7
  cnn_num_filters: 256
  lstm_hidden_size: 512
  
monitoring:
  metrics_flush_interval: 10.0
  max_metric_points: 100000
```

### **3. Database Tuning**

```sql
-- PostgreSQL performance tuning
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET max_connections = 200;

-- Reload configuration
SELECT pg_reload_conf();
```

## 🔧 **Troubleshooting**

### **Common Issues and Solutions**

#### **1. High Memory Usage**

```bash
# Check memory usage
kubectl top pods -n qbitel-prod

# Check detailed memory breakdown
kubectl exec -it qbitel-api-xxx -n qbitel-prod -- python -c "
from ai_engine.core.performance_optimizer import get_performance_optimizer
print(get_performance_optimizer().get_system_metrics())
"

# Solution: Increase memory limits or reduce cache sizes
kubectl patch deployment qbitel-api -n qbitel-prod -p '{"spec":{"template":{"spec":{"containers":[{"name":"qbitel-api","resources":{"limits":{"memory":"32Gi"}}}]}}}}'
```

#### **2. Database Connection Issues**

```bash
# Test database connectivity
kubectl exec -it qbitel-api-xxx -n qbitel-prod -- python -c "
import psycopg2
try:
    conn = psycopg2.connect(
        host='postgresql-service',
        database='qbitel_prod', 
        user='qbitel_prod_user',
        password='$DATABASE_PASSWORD'
    )
    print('Database connection successful')
    conn.close()
except Exception as e:
    print(f'Database connection failed: {e}')
"

# Check database logs
kubectl logs statefulset/postgresql -n qbitel-prod

# Solution: Check connection pool settings and increase if needed
```

#### **3. Performance Issues**

```bash
# Check API response times
curl -w "Total time: %{time_total}s\n" -s -o /dev/null https://api.qbitel.yourdomain.com/health

# Check metrics
curl https://api.qbitel.yourdomain.com/metrics | grep protocol_discovery_duration

# Enable debug logging temporarily
kubectl set env deployment/qbitel-api LOG_LEVEL=DEBUG -n qbitel-prod

# Solution: Enable GPU acceleration, increase cache sizes, tune batch sizes
```

#### **4. SSL/TLS Issues**

```bash
# Test SSL certificate
openssl s_client -connect api.qbitel.yourdomain.com:443 -servername api.qbitel.yourdomain.com

# Check certificate expiry
openssl x509 -in ssl/qbitel.crt -text -noout | grep "Not After"

# Solution: Renew certificates and restart services
```

### **Health Check Script**

```bash
#!/bin/bash
# health-check.sh

echo "QBITEL System Health Check"
echo "============================="

# API Health
echo "1. API Health Check:"
api_response=$(curl -s -o /dev/null -w "%{http_code}" https://api.qbitel.yourdomain.com/health)
if [[ $api_response == "200" ]]; then
  echo "✅ API is healthy"
else
  echo "❌ API is unhealthy (HTTP $api_response)"
fi

# Database Health
echo "2. Database Health Check:"
db_response=$(kubectl exec -it postgresql-0 -n qbitel-prod -- pg_isready -U qbitel_prod_user 2>/dev/null | grep "accepting connections")
if [[ -n "$db_response" ]]; then
  echo "✅ Database is healthy"
else
  echo "❌ Database is unhealthy"
fi

# Redis Health
echo "3. Redis Health Check:"
redis_response=$(kubectl exec -it deployment/redis -n qbitel-prod -- redis-cli -a $REDIS_PASSWORD ping 2>/dev/null)
if [[ "$redis_response" == "PONG" ]]; then
  echo "✅ Redis is healthy"
else
  echo "❌ Redis is unhealthy"
fi

# Check resource usage
echo "4. Resource Usage:"
kubectl top pods -n qbitel-prod

# Check recent errors
echo "5. Recent Errors:"
kubectl logs --since=1h -l app=qbitel-api -n qbitel-prod | grep ERROR | tail -5

echo "============================="
echo "Health check completed"
```

---

**Deployment Guide Version**: v2.1.0
**Last Updated**: 2025-11-22  
**Support**: deployment-support@qbitel.com