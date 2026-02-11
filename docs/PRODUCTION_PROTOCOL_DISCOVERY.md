# Production Protocol Discovery System

## Overview

The Production Protocol Discovery System is an enterprise-ready implementation that provides protocol discovery with SLA guarantees, explainable AI, model versioning, and comprehensive benchmarking capabilities.

## Features

### 1. SLA-Aware Discovery

The system provides guaranteed response times with configurable SLA thresholds:

```python
from ai_engine.discovery.production_protocol_discovery import (
    ProductionProtocolDiscovery,
    DiscoveryRequest,
    QualityMode
)

# Initialize system
discovery = ProductionProtocolDiscovery(config)
await discovery.initialize()

# Create request with SLA
request = DiscoveryRequest(
    request_id="req_001",
    packet_data=packet_bytes,
    quality_mode=QualityMode.BALANCED,
    sla_ms=100  # 100ms SLA
)

# Discover with SLA guarantee
result = await discovery.discover_with_sla(request, sla_ms=100)

print(f"Protocol: {result.protocol_type}")
print(f"Confidence: {result.confidence}")
print(f"SLA Met: {result.sla_met}")
print(f"Processing Time: {result.processing_time_ms}ms")
```

#### Quality Modes

The system supports three quality modes for speed/accuracy tradeoffs:

- **FAST**: Prioritizes speed, may sacrifice accuracy (~10-50ms)
- **BALANCED**: Balance between speed and accuracy (~50-200ms)
- **ACCURATE**: Prioritizes accuracy, may be slower (~200-500ms)

#### Fallback Strategies

When SLA cannot be met, the system uses intelligent fallback strategies:

1. **CACHED_RESULT**: Return previously cached result for similar packets
2. **FAST_MODEL**: Switch to faster model with reduced accuracy
3. **PARTIAL_RESULT**: Return partial analysis
4. **ERROR**: Raise SLA violation exception

### 2. Explainable AI Integration

Get detailed explanations for every prediction:

```python
# Request with explanation
request = DiscoveryRequest(
    request_id="req_002",
    packet_data=packet_bytes,
    require_explanation=True
)

result = await discovery.discover_with_explainability(request)

# Access explanation data
explanation = result.explanation

# Feature importances
for feature in explanation.feature_importances:
    print(f"{feature.feature_name}: {feature.importance_score:.3f}")

# Decision paths
for path in explanation.decision_paths:
    print(f"Step {path.step}: {path.layer_name}")
    print(f"  Confidence: {path.confidence_at_step:.3f}")

# Human-readable reasoning
print(f"Reasoning: {explanation.reasoning}")

# Confidence breakdown
for component, score in explanation.confidence_breakdown.items():
    print(f"{component}: {score:.3f}")
```

#### Explanation Components

1. **Feature Importances**: Which features influenced the prediction most
2. **Decision Paths**: How the model processed the input through layers
3. **Attention Weights**: Where the model focused (if applicable)
4. **Confidence Breakdown**: Contribution of different components
5. **Reasoning**: Human-readable explanation

### 3. Model Versioning & Deployment

Manage multiple model versions with advanced deployment strategies:

```python
from ai_engine.discovery.production_protocol_discovery import (
    ModelVersionManager,
    DeploymentType
)

manager = discovery.version_manager

# Register new version
manager.register_version(
    version_id="v2.0.0",
    model_path="models/protocol_discovery_v2.pt",
    deployment_type=DeploymentType.PRIMARY
)

# Activate version
manager.activate_version("v2.0.0")

# Rollback if needed
manager.rollback_to_version("v1.0.0")
```

#### Canary Deployments

Gradually roll out new versions:

```python
# Create canary deployment with 5% traffic
manager.create_canary_deployment(
    version_id="v2.0.0",
    traffic_percentage=5.0
)

# Monitor performance
perf = manager.get_version_performance("v2.0.0")
print(f"Latency P95: {perf['latency_p95']:.2f}ms")
print(f"Accuracy: {perf['accuracy_mean']:.2%}")

# Promote to primary if successful
if perf['latency_p95'] < 100 and perf['accuracy_mean'] > 0.95:
    manager.promote_canary("v2.0.0")
```

#### A/B Testing

Compare two model versions:

```python
# Create A/B test
ab_test = manager.create_ab_test(
    test_id="v1_vs_v2",
    version_a="v1.0.0",
    version_b="v2.0.0",
    traffic_split=0.5,  # 50/50 split
    duration_hours=24
)

# Requests automatically routed to variants
request = DiscoveryRequest(
    request_id="req_003",
    packet_data=packet_bytes,
    ab_test_variant="v1_vs_v2"
)

result = await discovery.discover_with_versioning(request)
print(f"Used version: {result.model_version}")

# Compare performance
v1_perf = manager.get_version_performance("v1.0.0")
v2_perf = manager.get_version_performance("v2.0.0")
```

### 4. Comprehensive Benchmarking

Built-in benchmarking framework for performance evaluation:

```python
from ai_engine.benchmarks.protocol_discovery_benchmark import (
    ProtocolDiscoveryBenchmark,
    BenchmarkConfig,
    BenchmarkType,
    LoadPattern
)

# Initialize benchmark
benchmark = ProtocolDiscoveryBenchmark(discovery, config)

# Load or generate test data
benchmark.generate_synthetic_data(num_samples=1000)

# Configure benchmark
bench_config = BenchmarkConfig(
    benchmark_type=BenchmarkType.LATENCY,
    num_requests=1000,
    concurrent_requests=10,
    sla_threshold_ms=100,
    quality_mode=QualityMode.BALANCED
)

# Run latency benchmark
result = await benchmark.run_latency_benchmark(bench_config)

print(f"Mean Latency: {result.latency_metrics.mean_ms:.2f}ms")
print(f"P95 Latency: {result.latency_metrics.p95_ms:.2f}ms")
print(f"P99 Latency: {result.latency_metrics.p99_ms:.2f}ms")
```

#### Benchmark Types

1. **Latency**: Measure response time distribution (p50, p90, p95, p99)
2. **Throughput**: Measure requests per second under load
3. **Accuracy**: Evaluate prediction accuracy against ground truth
4. **SLA Compliance**: Test SLA guarantee effectiveness
5. **Resource Usage**: Monitor CPU, memory, GPU usage
6. **Stress Test**: Test under various load patterns

#### Stress Testing

Test system behavior under different load patterns:

```python
# Configure stress test
stress_config = BenchmarkConfig(
    benchmark_type=BenchmarkType.STRESS_TEST,
    duration_seconds=300,  # 5 minutes
    concurrent_requests=50,
    load_pattern=LoadPattern.WAVE,  # Sinusoidal load
    sla_threshold_ms=100
)

# Run stress test
result = await benchmark.run_stress_test(stress_config)

print(f"SLA Compliance: {result.sla_compliance_rate:.2%}")
print(f"P99 Latency: {result.latency_metrics.p99_ms:.2f}ms")
```

#### Load Patterns

- **CONSTANT**: Steady load throughout test
- **RAMP_UP**: Gradually increasing load
- **SPIKE**: Sudden spike in the middle
- **WAVE**: Sinusoidal wave pattern

#### Full Benchmark Suite

Run all benchmarks at once:

```python
# Run complete suite
results = await benchmark.run_full_benchmark_suite(bench_config)

# Generate report
report = benchmark.generate_report()
print(report)

# Save results
benchmark.save_results("benchmark_results.json")
```

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│         ProductionProtocolDiscovery                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐  ┌──────────────────┐            │
│  │ ExplainableAI    │  │ VersionManager   │            │
│  │ - Feature Imp.   │  │ - Versioning     │            │
│  │ - Decision Paths │  │ - Canary Deploy  │            │
│  │ - Reasoning      │  │ - A/B Testing    │            │
│  └──────────────────┘  └──────────────────┘            │
│                                                          │
│  ┌──────────────────────────────────────────┐          │
│  │         SLA Management                    │          │
│  │  - Timeout Control                        │          │
│  │  - Fallback Strategies                    │          │
│  │  - Result Caching                         │          │
│  └──────────────────────────────────────────┘          │
│                                                          │
│  ┌──────────────────────────────────────────┐          │
│  │         Model Inference                   │          │
│  │  - Fast Mode                              │          │
│  │  - Balanced Mode                          │          │
│  │  - Accurate Mode                          │          │
│  └──────────────────────────────────────────┘          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
Request → Version Selection → SLA Timeout Wrapper
                                      ↓
                              Model Inference
                                      ↓
                         ┌────────────┴────────────┐
                         ↓                         ↓
                  Success Path              Timeout Path
                         ↓                         ↓
              Explanation Generation      Fallback Strategy
                         ↓                         ↓
                    Result Cache              Cached/Fast Result
                         ↓                         ↓
                         └────────────┬────────────┘
                                      ↓
                              Metrics Recording
                                      ↓
                                   Response
```

## Performance Characteristics

### Latency Targets

| Quality Mode | Target P50 | Target P95 | Target P99 |
|-------------|-----------|-----------|-----------|
| FAST        | 10-25ms   | 50ms      | 100ms     |
| BALANCED    | 50-100ms  | 200ms     | 300ms     |
| ACCURATE    | 100-250ms | 500ms     | 1000ms    |

### Throughput

- **Single Instance**: 100-500 req/s (depending on quality mode)
- **With Caching**: 1000+ req/s for cache hits
- **Horizontal Scaling**: Linear scaling with instances

### Resource Usage

- **CPU**: 20-40% per core under normal load
- **Memory**: 2-4 GB per instance
- **GPU**: 2-4 GB VRAM (if available)

## Monitoring & Metrics

### Prometheus Metrics

The system exports comprehensive Prometheus metrics:

```python
# SLA violations
qbitel_discovery_sla_violations_total{model_version, sla_threshold_ms}

# Latency distribution
qbitel_discovery_latency_ms{model_version, quality_mode}

# Confidence scores
qbitel_discovery_confidence{model_version}

# A/B test requests
qbitel_discovery_ab_test_requests_total{variant, model_version}

# Active model versions
qbitel_discovery_model_version_active{model_version, deployment_type}
```

### Statistics API

```python
# Get system statistics
stats = discovery.get_statistics()

print(f"Total Requests: {stats['total_requests']}")
print(f"SLA Violations: {stats['sla_violations']}")
print(f"SLA Compliance: {stats['sla_compliance_rate']:.2%}")
print(f"Cache Size: {stats['cache_size']}")
print(f"Active Versions: {stats['active_versions']}")
print(f"Active A/B Tests: {stats['active_ab_tests']}")
```

## Best Practices

### 1. SLA Configuration

```python
# Set realistic SLAs based on quality mode
sla_thresholds = {
    QualityMode.FAST: 50,      # 50ms
    QualityMode.BALANCED: 200,  # 200ms
    QualityMode.ACCURATE: 500   # 500ms
}

request = DiscoveryRequest(
    request_id=req_id,
    packet_data=data,
    quality_mode=mode,
    sla_ms=sla_thresholds[mode]
)
```

### 2. Version Deployment

```python
# Always use canary deployments for new versions
# 1. Deploy canary with 5% traffic
manager.create_canary_deployment("v2.0.0", traffic_percentage=5.0)

# 2. Monitor for 24 hours
await asyncio.sleep(86400)

# 3. Check performance
perf = manager.get_version_performance("v2.0.0")

# 4. Gradually increase traffic if successful
if perf['latency_p95'] < threshold:
    manager.versions["v2.0.0"].traffic_percentage = 25.0
    
# 5. Promote to primary after validation
manager.promote_canary("v2.0.0")
```

### 3. A/B Testing

```python
# Run A/B tests for at least 1000 requests per variant
ab_test = manager.create_ab_test(
    test_id="feature_comparison",
    version_a="v1.0.0",
    version_b="v2.0.0",
    traffic_split=0.5,
    duration_hours=48  # Run for 2 days
)

# Collect sufficient data before making decisions
min_samples = 1000
```

### 4. Caching Strategy

```python
# Configure cache TTL based on protocol stability
discovery.cache_ttl = 300  # 5 minutes for stable protocols

# Clear cache when deploying new versions
discovery.result_cache.clear()
```

### 5. Error Handling

```python
try:
    result = await discovery.discover_with_sla(request, sla_ms=100)
except SLAViolationException as e:
    # Handle SLA violation
    logger.warning(f"SLA violated: {e}")
    # Use fallback or cached result
except DiscoveryException as e:
    # Handle discovery failure
    logger.error(f"Discovery failed: {e}")
    # Return error response
```

## Production Deployment

### Configuration

```yaml
# config/production.yaml
discovery:
  sla_threshold_ms: 100
  cache_ttl_seconds: 300
  quality_mode: balanced
  
versioning:
  canary_traffic_percentage: 5.0
  ab_test_duration_hours: 24
  
monitoring:
  metrics_port: 9090
  enable_tracing: true
  
resources:
  max_workers: 4
  gpu_enabled: true
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: protocol-discovery
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: discovery
        image: qbitel/protocol-discovery:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: SLA_THRESHOLD_MS
          value: "100"
        - name: QUALITY_MODE
          value: "balanced"
```

### Monitoring Setup

```yaml
# Prometheus scrape config
scrape_configs:
  - job_name: 'protocol-discovery'
    static_configs:
      - targets: ['discovery:9090']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

## Troubleshooting

### High SLA Violation Rate

1. Check system resources (CPU, memory, GPU)
2. Verify quality mode matches SLA threshold
3. Review model performance metrics
4. Consider horizontal scaling
5. Enable caching for repeated patterns

### Low Accuracy

1. Review explanation data for insights
2. Check training data quality
3. Verify feature extraction
4. Consider retraining model
5. Use ACCURATE quality mode

### Memory Issues

1. Reduce cache TTL
2. Limit concurrent requests
3. Clear old cache entries
4. Monitor GPU memory usage
5. Adjust batch sizes

## API Reference

See inline documentation in:
- [`production_protocol_discovery.py`](../ai_engine/discovery/production_protocol_discovery.py)
- [`protocol_discovery_benchmark.py`](../ai_engine/benchmarks/protocol_discovery_benchmark.py)

## Testing

Run the test suite:

```bash
# Run all tests
pytest ai_engine/tests/test_production_protocol_discovery.py -v

# Run specific test class
pytest ai_engine/tests/test_production_protocol_discovery.py::TestSLADiscovery -v

# Run with coverage
pytest ai_engine/tests/test_production_protocol_discovery.py --cov=ai_engine.discovery
```

## License

Copyright © 2025 QBITEL. All rights reserved.