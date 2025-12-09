# CRONOS AI - Cloud-Native Modernization Implementation Plan

## Executive Summary

This document provides a detailed implementation plan for modernizing the cloud-native infrastructure of CRONOS AI, including Kubernetes upgrades, GitOps adoption, and observability improvements.

---

## Phase 1: Kubernetes Security Hardening (Week 1-2)

### 1.1 Pod Security Standards Migration

**Issue**: Pod Security Policy (PSP) is deprecated since Kubernetes 1.25
**Solution**: Migrate to Pod Security Standards (PSS)

#### Implementation

```yaml
# kubernetes/namespaces/cronos-ai-prod.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: cronos-ai-prod
  labels:
    # Enforce restricted security standard
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: v1.28
    # Audit baseline violations
    pod-security.kubernetes.io/audit: baseline
    pod-security.kubernetes.io/audit-version: v1.28
    # Warn on baseline violations
    pod-security.kubernetes.io/warn: baseline
    pod-security.kubernetes.io/warn-version: v1.28
    # Custom labels
    app.kubernetes.io/part-of: cronos-ai
    environment: production
---
# Staging namespace with less strict policies for testing
apiVersion: v1
kind: Namespace
metadata:
  name: cronos-ai-staging
  labels:
    pod-security.kubernetes.io/enforce: baseline
    pod-security.kubernetes.io/enforce-version: v1.28
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/audit-version: v1.28
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/warn-version: v1.28
    app.kubernetes.io/part-of: cronos-ai
    environment: staging
```

#### Updated Deployment Security Context

```yaml
# kubernetes/deployments/ai-engine-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-engine
  namespace: cronos-ai-prod
  labels:
    app: ai-engine
    app.kubernetes.io/name: ai-engine
    app.kubernetes.io/component: core
    app.kubernetes.io/part-of: cronos-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-engine
  template:
    metadata:
      labels:
        app: ai-engine
        quantum-safe: required
    spec:
      # Service account with minimal permissions
      serviceAccountName: ai-engine-sa
      automountServiceAccountToken: false

      # Pod-level security context
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault

      # Topology spread for HA
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: ai-engine

      containers:
        - name: ai-engine
          image: gcr.io/cronos-ai-prod/ai-engine:v2.0.0@sha256:abc123...
          imagePullPolicy: IfNotPresent

          # Container-level security context
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            runAsNonRoot: true
            runAsUser: 1000
            capabilities:
              drop:
                - ALL

          # Resource limits (required for restricted PSS)
          resources:
            requests:
              cpu: 500m
              memory: 1Gi
            limits:
              cpu: 2
              memory: 4Gi

          # Health probes
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
            failureThreshold: 3

          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 5

          startupProbe:
            httpGet:
              path: /health/startup
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 5
            failureThreshold: 30

          # Environment from ConfigMap and Secrets
          envFrom:
            - configMapRef:
                name: ai-engine-config
            - secretRef:
                name: ai-engine-secrets

          # Volume mounts
          volumeMounts:
            - name: tmp
              mountPath: /tmp
            - name: cache
              mountPath: /var/cache/cronos
            - name: models
              mountPath: /models
              readOnly: true

          ports:
            - name: http
              containerPort: 8080
            - name: grpc
              containerPort: 9090
            - name: metrics
              containerPort: 9091

      volumes:
        - name: tmp
          emptyDir:
            sizeLimit: 1Gi
        - name: cache
          emptyDir:
            sizeLimit: 5Gi
        - name: models
          persistentVolumeClaim:
            claimName: ai-engine-models

      # Image pull secrets
      imagePullSecrets:
        - name: gcr-auth-secret
```

---

### 1.2 Network Policies with Cilium

**Current State**: Basic Kubernetes NetworkPolicy
**Target State**: Cilium L7-aware network policies

#### Installation

```bash
# Install Cilium
helm repo add cilium https://helm.cilium.io/
helm install cilium cilium/cilium --version 1.14.5 \
  --namespace kube-system \
  --set kubeProxyReplacement=strict \
  --set hubble.enabled=true \
  --set hubble.relay.enabled=true \
  --set hubble.ui.enabled=true \
  --set l7Proxy=true
```

#### L7 Network Policies

```yaml
# kubernetes/network-policies/ai-engine-l7-policy.yaml
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: ai-engine-l7-policy
  namespace: cronos-ai-prod
spec:
  endpointSelector:
    matchLabels:
      app: ai-engine

  # Ingress rules
  ingress:
    # Allow from API gateway
    - fromEndpoints:
        - matchLabels:
            app: api-gateway
      toPorts:
        - ports:
            - port: "8080"
              protocol: TCP
          rules:
            http:
              - method: "POST"
                path: "/api/v1/analyze.*"
              - method: "POST"
                path: "/api/v1/discover.*"
              - method: "GET"
                path: "/api/v1/status"
              - method: "GET"
                path: "/health/.*"

    # Allow gRPC from internal services
    - fromEndpoints:
        - matchLabels:
            app: protocol-processor
        - matchLabels:
            app: threat-analyzer
      toPorts:
        - ports:
            - port: "9090"
              protocol: TCP

    # Allow metrics scraping from Prometheus
    - fromEndpoints:
        - matchLabels:
            app: prometheus
            io.kubernetes.pod.namespace: monitoring
      toPorts:
        - ports:
            - port: "9091"
              protocol: TCP

  # Egress rules
  egress:
    # Allow DNS
    - toEndpoints:
        - matchLabels:
            k8s:io.kubernetes.pod.namespace: kube-system
            k8s-app: kube-dns
      toPorts:
        - ports:
            - port: "53"
              protocol: UDP

    # Allow PostgreSQL
    - toEndpoints:
        - matchLabels:
            app: postgresql
      toPorts:
        - ports:
            - port: "5432"
              protocol: TCP

    # Allow Redis
    - toEndpoints:
        - matchLabels:
            app: redis
      toPorts:
        - ports:
            - port: "6379"
              protocol: TCP

    # Allow Qdrant vector DB
    - toEndpoints:
        - matchLabels:
            app: qdrant
      toPorts:
        - ports:
            - port: "6333"
              protocol: TCP

    # Allow Ollama (local LLM)
    - toEndpoints:
        - matchLabels:
            app: ollama
      toPorts:
        - ports:
            - port: "11434"
              protocol: TCP
---
# Deny all by default for the namespace
apiVersion: cilium.io/v2
kind: CiliumClusterwideNetworkPolicy
metadata:
  name: cronos-ai-default-deny
spec:
  endpointSelector:
    matchLabels:
      io.kubernetes.pod.namespace: cronos-ai-prod
  ingress:
    - {}
  egress:
    - {}
```

---

## Phase 2: Gateway API Migration (Week 3-4)

### 2.1 Replace Ingress with Gateway API

```yaml
# kubernetes/gateway/gateway.yaml
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: cronos-ai-gateway
  namespace: cronos-ai-prod
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  gatewayClassName: istio
  listeners:
    # HTTPS listener
    - name: https
      protocol: HTTPS
      port: 443
      hostname: "*.cronos-ai.example.com"
      tls:
        mode: Terminate
        certificateRefs:
          - name: cronos-ai-tls
            kind: Secret
      allowedRoutes:
        namespaces:
          from: Same
        kinds:
          - kind: HTTPRoute

    # gRPC listener
    - name: grpc
      protocol: HTTPS
      port: 443
      hostname: grpc.cronos-ai.example.com
      tls:
        mode: Terminate
        certificateRefs:
          - name: cronos-ai-tls
      allowedRoutes:
        namespaces:
          from: Same
        kinds:
          - kind: GRPCRoute
---
# HTTP Routes
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: ai-engine-routes
  namespace: cronos-ai-prod
spec:
  parentRefs:
    - name: cronos-ai-gateway
      sectionName: https

  hostnames:
    - api.cronos-ai.example.com

  rules:
    # Protocol Analysis API
    - matches:
        - path:
            type: PathPrefix
            value: /api/v1/analyze
          method: POST
      backendRefs:
        - name: ai-engine
          port: 8080
          weight: 100
      timeouts:
        request: 60s
        backendRequest: 55s

    # Protocol Discovery API
    - matches:
        - path:
            type: PathPrefix
            value: /api/v1/discover
      backendRefs:
        - name: ai-engine
          port: 8080
      timeouts:
        request: 300s  # Discovery can take longer
        backendRequest: 290s

    # Marketplace API
    - matches:
        - path:
            type: PathPrefix
            value: /api/v1/marketplace
      backendRefs:
        - name: marketplace-service
          port: 8080

    # Health endpoints
    - matches:
        - path:
            type: PathPrefix
            value: /health
      backendRefs:
        - name: ai-engine
          port: 8080

    # Canary deployment - route 10% to canary
    - matches:
        - path:
            type: PathPrefix
            value: /api/v1
          headers:
            - name: x-canary
              value: "true"
      backendRefs:
        - name: ai-engine-canary
          port: 8080
---
# Rate limiting configuration
apiVersion: gateway.networking.k8s.io/v1alpha2
kind: BackendTLSPolicy
metadata:
  name: ai-engine-tls-policy
  namespace: cronos-ai-prod
spec:
  targetRef:
    group: ""
    kind: Service
    name: ai-engine
  tls:
    caCertRefs:
      - name: internal-ca
        kind: ConfigMap
    hostname: ai-engine.cronos-ai-prod.svc.cluster.local
```

---

## Phase 3: GitOps with ArgoCD (Week 5-6)

### 3.1 ArgoCD Installation and Configuration

```bash
# Install ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Install ArgoCD CLI
brew install argocd

# Get initial admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

### 3.2 Application Configuration

```yaml
# argocd/applications/cronos-ai-prod.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: cronos-ai-prod
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: production

  source:
    repoURL: https://github.com/qbitel/cronos-ai
    targetRevision: main
    path: kubernetes/overlays/production

    # Kustomize configuration
    kustomize:
      images:
        - gcr.io/cronos-ai-prod/ai-engine

  destination:
    server: https://kubernetes.default.svc
    namespace: cronos-ai-prod

  # Sync policy
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
      - CreateNamespace=true
      - PruneLast=true
      - ApplyOutOfSyncOnly=true
      - Validate=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m

  # Ignore differences that are managed externally
  ignoreDifferences:
    - group: apps
      kind: Deployment
      jsonPointers:
        - /spec/replicas  # HPA manages replicas

  # Health checks
  revisionHistoryLimit: 10
---
# ArgoCD Project
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: production
  namespace: argocd
spec:
  description: Production applications

  sourceRepos:
    - 'https://github.com/qbitel/cronos-ai'
    - 'https://github.com/qbitel/cronos-ai-config'

  destinations:
    - namespace: 'cronos-ai-prod'
      server: 'https://kubernetes.default.svc'
    - namespace: 'monitoring'
      server: 'https://kubernetes.default.svc'

  # Allowed cluster resources
  clusterResourceWhitelist:
    - group: ''
      kind: Namespace
    - group: ''
      kind: PersistentVolume
    - group: cilium.io
      kind: '*'
    - group: gateway.networking.k8s.io
      kind: '*'

  # Blocked resources
  namespaceResourceBlacklist:
    - group: ''
      kind: ResourceQuota
    - group: ''
      kind: LimitRange

  # Require signed commits
  signatureKeys:
    - keyID: AAAAB3NzaC1yc2EAAAADAQABAAAA...

  # Sync windows for maintenance
  syncWindows:
    - kind: deny
      schedule: '0 22 * * 5'  # No deploys Friday 10 PM
      duration: 60h           # Until Monday 10 AM
      applications:
        - '*-prod'
```

### 3.3 Notification Configuration

```yaml
# argocd/notifications/notifications-cm.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: argocd-notifications-cm
  namespace: argocd
data:
  # Triggers
  trigger.on-deployed: |
    - description: Application is deployed
      oncePer: app.status.operationState.finishedAt
      send:
        - slack-deployed
      when: app.status.operationState.phase in ['Succeeded'] and app.status.health.status == 'Healthy'

  trigger.on-health-degraded: |
    - description: Application health degraded
      send:
        - slack-health-degraded
        - pagerduty-alert
      when: app.status.health.status == 'Degraded'

  trigger.on-sync-failed: |
    - description: Sync failed
      send:
        - slack-sync-failed
      when: app.status.operationState.phase in ['Error', 'Failed']

  # Templates
  template.slack-deployed: |
    message: |
      :white_check_mark: *{{.app.metadata.name}}* deployed successfully
      *Revision:* {{.app.status.sync.revision}}
      *Sync Status:* {{.app.status.sync.status}}
      *Health:* {{.app.status.health.status}}
    slack:
      attachments: |
        [{
          "color": "#18be52",
          "fields": [
            {"title": "Application", "value": "{{.app.metadata.name}}", "short": true},
            {"title": "Environment", "value": "{{.app.spec.destination.namespace}}", "short": true}
          ]
        }]

  template.slack-health-degraded: |
    message: |
      :warning: *{{.app.metadata.name}}* health degraded
      *Health Status:* {{.app.status.health.status}}
      *Message:* {{.app.status.health.message}}
    slack:
      attachments: |
        [{
          "color": "#f4c030",
          "fields": [
            {"title": "Application", "value": "{{.app.metadata.name}}", "short": true},
            {"title": "Environment", "value": "{{.app.spec.destination.namespace}}", "short": true}
          ]
        }]

  template.slack-sync-failed: |
    message: |
      :x: *{{.app.metadata.name}}* sync failed
      *Error:* {{.app.status.operationState.message}}
    slack:
      attachments: |
        [{
          "color": "#E96D76",
          "fields": [
            {"title": "Application", "value": "{{.app.metadata.name}}", "short": true},
            {"title": "Revision", "value": "{{.app.status.sync.revision}}", "short": true}
          ]
        }]

  # Services
  service.slack: |
    token: $slack-token
    channel: cronos-deployments

  service.pagerduty: |
    key: $pagerduty-key
```

---

## Phase 4: Kyverno Policy Engine (Week 7-8)

### 4.1 Core Security Policies

```yaml
# kubernetes/policies/require-security-context.yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-security-context
  annotations:
    policies.kyverno.io/title: Require Security Context
    policies.kyverno.io/category: Pod Security
    policies.kyverno.io/severity: high
spec:
  validationFailureAction: enforce
  background: true
  rules:
    - name: require-run-as-non-root
      match:
        resources:
          kinds:
            - Pod
          namespaces:
            - cronos-ai-prod
            - cronos-ai-staging
      validate:
        message: "Pods must run as non-root"
        pattern:
          spec:
            securityContext:
              runAsNonRoot: true
            containers:
              - securityContext:
                  runAsNonRoot: true
                  allowPrivilegeEscalation: false

    - name: require-read-only-root-filesystem
      match:
        resources:
          kinds:
            - Pod
          namespaces:
            - cronos-ai-prod
      validate:
        message: "Containers must have read-only root filesystem"
        pattern:
          spec:
            containers:
              - securityContext:
                  readOnlyRootFilesystem: true

    - name: drop-all-capabilities
      match:
        resources:
          kinds:
            - Pod
          namespaces:
            - cronos-ai-prod
      validate:
        message: "Containers must drop all capabilities"
        pattern:
          spec:
            containers:
              - securityContext:
                  capabilities:
                    drop:
                      - ALL
---
# Restrict image registries
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: restrict-image-registries
  annotations:
    policies.kyverno.io/title: Restrict Image Registries
    policies.kyverno.io/category: Supply Chain Security
    policies.kyverno.io/severity: high
spec:
  validationFailureAction: enforce
  background: true
  rules:
    - name: validate-image-registry
      match:
        resources:
          kinds:
            - Pod
          namespaces:
            - cronos-ai-prod
      validate:
        message: "Images must come from approved registries: gcr.io/cronos-ai-prod, ghcr.io/qbitel"
        pattern:
          spec:
            containers:
              - image: "gcr.io/cronos-ai-prod/* | ghcr.io/qbitel/*"
            initContainers:
              - image: "gcr.io/cronos-ai-prod/* | ghcr.io/qbitel/*"
---
# Require resource limits
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-resource-limits
  annotations:
    policies.kyverno.io/title: Require Resource Limits
    policies.kyverno.io/category: Resource Management
    policies.kyverno.io/severity: medium
spec:
  validationFailureAction: enforce
  background: true
  rules:
    - name: require-limits
      match:
        resources:
          kinds:
            - Pod
          namespaces:
            - cronos-ai-prod
            - cronos-ai-staging
      validate:
        message: "CPU and memory limits are required"
        pattern:
          spec:
            containers:
              - resources:
                  limits:
                    memory: "?*"
                    cpu: "?*"
                  requests:
                    memory: "?*"
                    cpu: "?*"
---
# Require quantum-safe label
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-quantum-safe-label
  annotations:
    policies.kyverno.io/title: Require Quantum-Safe Label
    policies.kyverno.io/category: CRONOS AI Security
    policies.kyverno.io/severity: medium
spec:
  validationFailureAction: audit  # Audit mode first, then enforce
  background: true
  rules:
    - name: check-quantum-safe-label
      match:
        resources:
          kinds:
            - Pod
          namespaces:
            - cronos-ai-prod
      validate:
        message: "Pods handling cryptographic operations must have quantum-safe label"
        pattern:
          metadata:
            labels:
              quantum-safe: "required | verified"
```

### 4.2 Mutation Policies

```yaml
# kubernetes/policies/add-default-labels.yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: add-default-labels
  annotations:
    policies.kyverno.io/title: Add Default Labels
    policies.kyverno.io/category: Best Practices
spec:
  background: false
  rules:
    - name: add-app-labels
      match:
        resources:
          kinds:
            - Pod
          namespaces:
            - cronos-ai-*
      mutate:
        patchStrategicMerge:
          metadata:
            labels:
              +(app.kubernetes.io/managed-by): kyverno
              +(app.kubernetes.io/part-of): cronos-ai
---
# Add security context defaults
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: add-security-context-defaults
  annotations:
    policies.kyverno.io/title: Add Security Context Defaults
spec:
  background: false
  rules:
    - name: add-seccomp-profile
      match:
        resources:
          kinds:
            - Pod
          namespaces:
            - cronos-ai-*
      mutate:
        patchStrategicMerge:
          spec:
            securityContext:
              +(seccompProfile):
                type: RuntimeDefault
---
# Generate network policy for new namespaces
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: generate-network-policy
spec:
  background: false
  rules:
    - name: generate-default-deny
      match:
        resources:
          kinds:
            - Namespace
          selector:
            matchLabels:
              app.kubernetes.io/part-of: cronos-ai
      generate:
        apiVersion: networking.k8s.io/v1
        kind: NetworkPolicy
        name: default-deny-all
        namespace: "{{request.object.metadata.name}}"
        data:
          spec:
            podSelector: {}
            policyTypes:
              - Ingress
              - Egress
```

---

## Phase 5: LGTM Observability Stack (Week 9-10)

### 5.1 Grafana LGTM Stack Deployment

```yaml
# kubernetes/observability/lgtm-stack.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring
  labels:
    pod-security.kubernetes.io/enforce: baseline
---
# Loki for logs
apiVersion: helm.cattle.io/v1
kind: HelmChart
metadata:
  name: loki
  namespace: monitoring
spec:
  repo: https://grafana.github.io/helm-charts
  chart: loki
  version: 5.41.4
  valuesContent: |-
    loki:
      auth_enabled: false
      limits_config:
        enforce_metric_name: false
        reject_old_samples: true
        reject_old_samples_max_age: 168h
        max_query_length: 721h
      storage:
        type: s3
        s3:
          endpoint: minio.monitoring:9000
          bucketnames: loki-chunks
          access_key_id: ${MINIO_ACCESS_KEY}
          secret_access_key: ${MINIO_SECRET_KEY}
          insecure: true

    gateway:
      enabled: true
      replicas: 2

    write:
      replicas: 3
      persistence:
        size: 50Gi

    read:
      replicas: 3

    backend:
      replicas: 3
---
# Tempo for traces
apiVersion: helm.cattle.io/v1
kind: HelmChart
metadata:
  name: tempo
  namespace: monitoring
spec:
  repo: https://grafana.github.io/helm-charts
  chart: tempo-distributed
  version: 1.7.3
  valuesContent: |-
    traces:
      otlp:
        grpc:
          enabled: true
        http:
          enabled: true
      jaeger:
        thriftHttp:
          enabled: true

    storage:
      trace:
        backend: s3
        s3:
          bucket: tempo-traces
          endpoint: minio.monitoring:9000
          access_key: ${MINIO_ACCESS_KEY}
          secret_key: ${MINIO_SECRET_KEY}
          insecure: true

    ingester:
      replicas: 3

    distributor:
      replicas: 2

    querier:
      replicas: 2

    queryFrontend:
      replicas: 2
---
# Mimir for metrics
apiVersion: helm.cattle.io/v1
kind: HelmChart
metadata:
  name: mimir
  namespace: monitoring
spec:
  repo: https://grafana.github.io/helm-charts
  chart: mimir-distributed
  version: 5.1.3
  valuesContent: |-
    mimir:
      structuredConfig:
        limits:
          max_global_series_per_user: 1500000
          max_global_series_per_metric: 50000

    ingester:
      replicas: 3
      persistentVolume:
        size: 50Gi

    distributor:
      replicas: 2

    querier:
      replicas: 2

    query_frontend:
      replicas: 2

    store_gateway:
      replicas: 3
      persistentVolume:
        size: 50Gi

    compactor:
      replicas: 1
      persistentVolume:
        size: 50Gi
---
# Grafana
apiVersion: helm.cattle.io/v1
kind: HelmChart
metadata:
  name: grafana
  namespace: monitoring
spec:
  repo: https://grafana.github.io/helm-charts
  chart: grafana
  version: 7.0.17
  valuesContent: |-
    replicas: 2

    persistence:
      enabled: true
      size: 10Gi

    datasources:
      datasources.yaml:
        apiVersion: 1
        datasources:
          - name: Mimir
            type: prometheus
            url: http://mimir-query-frontend:8080/prometheus
            isDefault: true

          - name: Loki
            type: loki
            url: http://loki-gateway:80

          - name: Tempo
            type: tempo
            url: http://tempo-query-frontend:3100

    dashboardProviders:
      dashboardproviders.yaml:
        apiVersion: 1
        providers:
          - name: 'cronos-ai'
            orgId: 1
            folder: 'CRONOS AI'
            type: file
            disableDeletion: false
            editable: true
            options:
              path: /var/lib/grafana/dashboards/cronos-ai

    dashboardsConfigMaps:
      cronos-ai: grafana-dashboards-cronos

    ingress:
      enabled: true
      ingressClassName: nginx
      hosts:
        - grafana.cronos-ai.example.com
      tls:
        - secretName: grafana-tls
          hosts:
            - grafana.cronos-ai.example.com
```

### 5.2 CRONOS AI Dashboards

```yaml
# kubernetes/observability/dashboards/cronos-ai-overview.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards-cronos
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  cronos-ai-overview.json: |
    {
      "dashboard": {
        "title": "CRONOS AI Overview",
        "uid": "cronos-overview",
        "panels": [
          {
            "title": "Protocol Analysis Requests",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "targets": [
              {
                "expr": "sum(rate(http_requests_total{job=\"ai-engine\", endpoint=~\"/api/v1/analyze.*\"}[5m]))",
                "legendFormat": "Requests/s"
              }
            ]
          },
          {
            "title": "LLM Provider Usage",
            "type": "piechart",
            "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0},
            "targets": [
              {
                "expr": "sum by (provider) (llm_requests_total)",
                "legendFormat": "{{provider}}"
              }
            ]
          },
          {
            "title": "Model Inference Latency (P95)",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
            "targets": [
              {
                "expr": "histogram_quantile(0.95, sum(rate(model_inference_duration_seconds_bucket[5m])) by (le, model))",
                "legendFormat": "{{model}}"
              }
            ]
          },
          {
            "title": "Threat Detection Rate",
            "type": "stat",
            "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8},
            "targets": [
              {
                "expr": "sum(rate(threats_detected_total[1h]))",
                "legendFormat": "Threats/hour"
              }
            ]
          },
          {
            "title": "Compliance Violations",
            "type": "stat",
            "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8},
            "targets": [
              {
                "expr": "sum(compliance_violations_total)",
                "legendFormat": "Total Violations"
              }
            ]
          },
          {
            "title": "Active Incidents",
            "type": "stat",
            "gridPos": {"h": 4, "w": 6, "x": 12, "y": 8},
            "targets": [
              {
                "expr": "sum(security_incidents_active)",
                "legendFormat": "Active"
              }
            ]
          },
          {
            "title": "Autonomous Decision Rate",
            "type": "gauge",
            "gridPos": {"h": 4, "w": 6, "x": 18, "y": 8},
            "targets": [
              {
                "expr": "sum(rate(autonomous_decisions_total{outcome=\"auto_executed\"}[1h])) / sum(rate(autonomous_decisions_total[1h])) * 100",
                "legendFormat": "Autonomous %"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "thresholds": {
                  "steps": [
                    {"color": "red", "value": 0},
                    {"color": "yellow", "value": 50},
                    {"color": "green", "value": 75}
                  ]
                },
                "unit": "percent",
                "max": 100
              }
            }
          }
        ]
      }
    }
```

---

## Migration Checklist

### Phase 1: Security Hardening
- [ ] Create namespaces with PSS labels
- [ ] Update all deployments with security contexts
- [ ] Remove deprecated PSP resources
- [ ] Install Cilium CNI
- [ ] Apply L7 network policies
- [ ] Test pod communication

### Phase 2: Gateway API
- [ ] Install Gateway API CRDs
- [ ] Create Gateway resources
- [ ] Migrate Ingress to HTTPRoute
- [ ] Configure TLS termination
- [ ] Test routing rules
- [ ] Remove old Ingress resources

### Phase 3: GitOps
- [ ] Install ArgoCD
- [ ] Create AppProject and Applications
- [ ] Configure notifications
- [ ] Set up sync windows
- [ ] Migrate to GitOps workflow
- [ ] Train team on ArgoCD

### Phase 4: Policy Engine
- [ ] Install Kyverno
- [ ] Deploy validation policies
- [ ] Deploy mutation policies
- [ ] Deploy generation policies
- [ ] Monitor policy violations
- [ ] Enable enforcement mode

### Phase 5: Observability
- [ ] Deploy LGTM stack
- [ ] Configure data retention
- [ ] Import dashboards
- [ ] Set up alerting rules
- [ ] Integrate with PagerDuty
- [ ] Train team on new tools

---

## Rollback Procedures

### ArgoCD Rollback
```bash
# List application history
argocd app history cronos-ai-prod

# Rollback to previous revision
argocd app rollback cronos-ai-prod <revision>
```

### Kyverno Emergency Disable
```bash
# Disable all policies in emergency
kubectl annotate clusterpolicy --all policies.kyverno.io/severity-

# Or set to audit mode
kubectl patch clusterpolicy require-security-context --type=merge \
  -p '{"spec":{"validationFailureAction":"audit"}}'
```

### Cilium Fallback
```bash
# If Cilium causes network issues, fallback to kube-proxy
kubectl -n kube-system delete ds kube-proxy
kubectl -n kube-system delete cm kube-proxy

# Reinstall kube-proxy
kubeadm init phase addon kube-proxy
```

---

## Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Security Hardening | 2 weeks | None |
| Gateway API | 2 weeks | Security Hardening |
| GitOps | 2 weeks | None |
| Policy Engine | 2 weeks | Security Hardening |
| Observability | 2 weeks | GitOps |
| **Total** | **10 weeks** | |

Note: Some phases can run in parallel.
