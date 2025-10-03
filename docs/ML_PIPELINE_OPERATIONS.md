# ML/Data Pipeline Operations Guide

This document provides comprehensive procedures for retraining and rollback operations for the CRONOS AI ML/data pipelines, specifically for the Field Detector and Translation Studio components.

## Table of Contents

1. [Overview](#overview)
2. [Field Detector Operations](#field-detector-operations)
3. [Translation Studio Operations](#translation-studio-operations)
4. [Rollback Procedures](#rollback-procedures)
5. [Monitoring and Validation](#monitoring-and-validation)
6. [Best Practices](#best-practices)

---

## Overview

The CRONOS AI system includes two primary ML/data pipelines:

1. **Field Detector**: BiLSTM-CRF model for automatic field boundary detection in protocol messages
2. **Translation Studio**: LLM-powered protocol translation with rule generation and optimization

Both systems support:
- Batch training for incremental learning
- Comprehensive validation metrics
- Model/rules persistence with versioning
- Checkpoint management for rollback

---

## Field Detector Operations

### Retraining Procedures

#### 1. Standard Retraining

Use this when you have new labeled data and want to retrain the model from scratch or fine-tune.

```python
from ai_engine.detection.field_detector import FieldDetector
from ai_engine.core.config import Config

# Initialize detector
config = Config()
detector = FieldDetector(config)
await detector.initialize()

# Prepare training data
# Format: List[Tuple[bytes, List[Tuple[int, int, str]]]]
# Each tuple: (message_bytes, [(start_pos, end_pos, field_type), ...])
training_data = [
    (b'\x01\x00TEST1234\x00\x00\xFF\xFF', [(0, 2, 'header'), (2, 10, 'payload'), (10, 14, 'checksum')]),
    # ... more examples
]

validation_data = [
    # Similar format for validation
]

# Train with comprehensive metrics
history = detector.train(
    training_data=training_data,
    validation_data=validation_data,
    num_epochs=50,
    learning_rate=1e-3,
    batch_size=16,
    early_stopping_patience=10,
    save_best_only=True,
    checkpoint_dir="checkpoints/field_detector"
)

# Review training results
print(f"Best F1 Score: {history['summary']['best_val_f1']:.4f}")
print(f"Total Training Time: {history['summary']['total_time_seconds']:.2f}s")
print(f"Final Metrics:")
print(f"  - Precision: {history['val_precision'][-1]:.4f}")
print(f"  - Recall: {history['val_recall'][-1]:.4f}")
print(f"  - Accuracy: {history['val_accuracy'][-1]:.4f}")
```

#### 2. Batch/Incremental Training

Use this for continuous learning with new data batches without full retraining.

```python
# Prepare data in batches
training_batches = [
    batch1_data,  # List of (message, annotations) tuples
    batch2_data,
    batch3_data,
]

# Load existing model (optional)
await detector.load_checkpoint(
    "checkpoints/field_detector/field_detector_v1.0.0_latest.pt",
    load_optimizer=True,
    load_scheduler=True
)

# Train on batches sequentially
history = detector.train_batch(
    training_batches=training_batches,
    validation_data=validation_data,
    num_epochs_per_batch=10,
    learning_rate=1e-4,  # Lower LR for fine-tuning
    batch_size=16
)

# Review batch training results
for batch_idx, metrics in enumerate(history['batch_metrics']):
    print(f"Batch {batch_idx + 1}:")
    print(f"  - Samples: {metrics['samples']}")
    print(f"  - Final Loss: {metrics['final_train_loss']:.4f}")
    print(f"  - Best F1: {metrics['best_val_f1']:.4f}")
```

#### 3. Model Checkpointing

Models are automatically saved during training with versioning:

```python
# Checkpoint naming format:
# field_detector_v{version}_epoch{epoch}_f1{f1_score}_{timestamp}.pt

# List available checkpoints
import os
checkpoint_dir = "checkpoints/field_detector"
checkpoints = sorted([
    f for f in os.listdir(checkpoint_dir) 
    if f.endswith('.pt') and not f.endswith('_latest.pt')
], reverse=True)

print("Available checkpoints:")
for ckpt in checkpoints[:5]:  # Show latest 5
    print(f"  - {ckpt}")
```

#### 4. Loading Checkpoints

```python
# Load specific checkpoint
checkpoint_path = "checkpoints/field_detector/field_detector_v1.0.0_epoch25_f10.9234_20250102_120000.pt"

metadata = await detector.load_checkpoint(
    checkpoint_path=checkpoint_path,
    load_optimizer=False,  # Set True if continuing training
    load_scheduler=False   # Set True if continuing training
)

print(f"Loaded checkpoint:")
print(f"  - Epoch: {metadata['epoch']}")
print(f"  - Version: {metadata['version']}")
print(f"  - F1 Score: {metadata['metrics']['f1']:.4f}")
print(f"  - Timestamp: {metadata['timestamp']}")
```

### Validation and Testing

```python
# Validate model performance
from ai_engine.tests.test_ml_pipeline_performance import TestFieldDetectorPerformance

# Run performance tests
test_suite = TestFieldDetectorPerformance()
await test_suite.test_inference_latency_baseline(detector)
await test_suite.test_validation_metrics_completeness(detector, validation_data)

# Manual validation
test_message = b'\x01\x00TEST1234\x00\x00\xFF\xFF'
boundaries = await detector.detect_boundaries(test_message)

print(f"Detected {len(boundaries)} fields:")
for i, boundary in enumerate(boundaries):
    print(f"  Field {i+1}:")
    print(f"    - Position: {boundary.start_pos}-{boundary.end_pos}")
    print(f"    - Type: {boundary.field_type}")
    print(f"    - Confidence: {boundary.confidence:.4f}")
```

---

## Translation Studio Operations

### Retraining Procedures

#### 1. Rule Generation

Generate new translation rules for a protocol pair:

```python
from ai_engine.llm.translation_studio import (
    ProtocolTranslationStudio,
    ProtocolSpecification,
    ProtocolField,
    ProtocolType,
    FieldType
)
from ai_engine.llm.unified_llm_service import UnifiedLLMService

# Initialize studio
config = Config()
llm_service = UnifiedLLMService(config)
studio = ProtocolTranslationStudio(config, llm_service)

# Define protocol specifications
source_spec = ProtocolSpecification(
    protocol_id="modbus",
    protocol_type=ProtocolType.MODBUS,
    version="1.0",
    name="Modbus",
    description="Modbus Protocol",
    fields=[
        ProtocolField(name="function_code", field_type=FieldType.INTEGER, length=1, required=True),
        ProtocolField(name="address", field_type=FieldType.INTEGER, length=2, required=True),
        ProtocolField(name="data", field_type=FieldType.BINARY, required=True),
    ]
)

target_spec = ProtocolSpecification(
    protocol_id="mqtt",
    protocol_type=ProtocolType.MQTT,
    version="3.1.1",
    name="MQTT",
    description="MQTT Protocol",
    fields=[
        ProtocolField(name="message_type", field_type=FieldType.INTEGER, length=1, required=True),
        ProtocolField(name="topic", field_type=FieldType.STRING, length=256, required=True),
        ProtocolField(name="payload", field_type=FieldType.BINARY, required=True),
    ]
)

# Generate translation rules
rules = await studio.generate_translation_rules(source_spec, target_spec)

print(f"Generated {len(rules.rules)} translation rules")
print(f"Accuracy: {rules.accuracy:.4f}")
print(f"Test cases: {len(rules.test_cases)}")
```

#### 2. Batch Training for Rules

Refine translation rules with training data:

```python
# Prepare training batches
training_batches = [
    [
        {
            'source': {'function_code': 3, 'address': 100, 'data': b'\x00\x0A'},
            'expected_target': {'message_type': 1, 'topic': 'modbus/read', 'payload': b'\x00\x0A'}
        },
        # ... more examples
    ],
    # ... more batches
]

# Train rules on batches
history = await studio.train_translation_rules_batch(
    training_batches=training_batches,
    source_spec=source_spec,
    target_spec=target_spec,
    validation_data=validation_examples
)

# Review results
print(f"Batch Training Results:")
for batch_metrics in history['batch_metrics']:
    print(f"  Batch {batch_metrics['batch_idx'] + 1}:")
    print(f"    - Accuracy: {batch_metrics['accuracy']:.4f}")
    print(f"    - Pass Rate: {batch_metrics['pass_rate']:.4f}")
    print(f"    - Avg Latency: {batch_metrics['avg_latency_ms']:.2f}ms")
```

#### 3. Rules Persistence

Save and load translation rules:

```python
# Save rules with versioning
rules_path = await studio.save_translation_rules(
    rules=rules,
    rules_dir="checkpoints/translation_rules",
    version="1.0.0"
)

print(f"Rules saved to: {rules_path}")

# List available rules
rules_list = studio.list_saved_rules(
    rules_dir="checkpoints/translation_rules",
    source_protocol="modbus",
    target_protocol="mqtt"
)

print(f"Available rules:")
for rule_info in rules_list:
    print(f"  - {rule_info['filename']}")
    print(f"    Version: {rule_info['version']}, Accuracy: {rule_info['accuracy']:.4f}")

# Load specific rules
loaded_rules = await studio.load_translation_rules(rules_path)
print(f"Loaded rules with accuracy: {loaded_rules.accuracy:.4f}")
```

#### 4. Rule Optimization

Optimize existing rules for better performance:

```python
# Get current performance data
performance_data = await studio.get_performance_data("modbus", "mqtt")

if performance_data:
    # Optimize rules
    optimized = await studio.optimize_translation(rules, performance_data)
    
    print(f"Optimization Results:")
    print(f"  - Performance improvement: {optimized.performance_improvement:.1%}")
    print(f"  - Accuracy improvement: {optimized.accuracy_improvement:.1%}")
    print(f"  - Optimizations applied: {len(optimized.optimizations_applied)}")
    
    for opt_type in optimized.optimizations_applied:
        print(f"    * {opt_type}")
    
    # Save optimized rules
    await studio.save_translation_rules(
        optimized.optimized_rules,
        version="1.1.0"
    )
```

### Validation and Testing

```python
# Validate translation rules
validation_metrics = await studio._validate_translation_rules(
    rules=rules,
    validation_data=validation_examples
)

print(f"Validation Metrics:")
print(f"  - Pass Rate: {validation_metrics['pass_rate']:.4f}")
print(f"  - Error Rate: {validation_metrics['error_rate']:.4f}")
print(f"  - Avg Latency: {validation_metrics['avg_latency_ms']:.2f}ms")
print(f"  - P95 Latency: {validation_metrics['p95_latency_ms']:.2f}ms")

# Test translation
test_message = b'\x03\x00\x64\x00\x0A'
translated = await studio.translate_protocol("modbus", "mqtt", test_message)
print(f"Translation successful: {len(translated)} bytes")
```

---

## Rollback Procedures

### Field Detector Rollback

#### Scenario 1: Rollback to Previous Checkpoint

```python
# List available checkpoints (sorted by timestamp)
checkpoint_dir = "checkpoints/field_detector"
checkpoints = sorted([
    f for f in os.listdir(checkpoint_dir) 
    if f.endswith('.pt') and not f.endswith('_latest.pt')
], reverse=True)

# Current (problematic) checkpoint
current_checkpoint = checkpoints[0]
print(f"Current checkpoint: {current_checkpoint}")

# Previous (stable) checkpoint
previous_checkpoint = checkpoints[1]
print(f"Rolling back to: {previous_checkpoint}")

# Load previous checkpoint
detector = FieldDetector(config)
await detector.initialize()
metadata = await detector.load_checkpoint(
    os.path.join(checkpoint_dir, previous_checkpoint)
)

# Validate rollback
print(f"Rollback successful:")
print(f"  - Version: {metadata['version']}")
print(f"  - F1 Score: {metadata['metrics']['f1']:.4f}")
print(f"  - Epoch: {metadata['epoch']}")

# Save as new latest
await detector.save_model(
    os.path.join(checkpoint_dir, f"field_detector_v{metadata['version']}_latest.pt")
)
```

#### Scenario 2: Rollback to Specific Version

```python
# Find checkpoint by version and metrics
target_version = "1.0.0"
min_f1_score = 0.90

matching_checkpoints = [
    f for f in checkpoints
    if f"v{target_version}" in f and float(f.split('_f1')[1].split('_')[0]) >= min_f1_score
]

if matching_checkpoints:
    rollback_checkpoint = matching_checkpoints[0]
    print(f"Rolling back to: {rollback_checkpoint}")
    
    await detector.load_checkpoint(
        os.path.join(checkpoint_dir, rollback_checkpoint)
    )
else:
    print(f"No checkpoint found matching criteria")
```

### Translation Studio Rollback

#### Scenario 1: Rollback to Previous Rules

```python
# List available rules
rules_list = studio.list_saved_rules(
    rules_dir="checkpoints/translation_rules",
    source_protocol="modbus",
    target_protocol="mqtt"
)

# Current (problematic) rules
current_rules = rules_list[0]
print(f"Current rules: {current_rules['filename']}")
print(f"  Accuracy: {current_rules['accuracy']:.4f}")

# Previous (stable) rules
previous_rules = rules_list[1]
print(f"Rolling back to: {previous_rules['filename']}")
print(f"  Accuracy: {previous_rules['accuracy']:.4f}")

# Load previous rules
loaded_rules = await studio.load_translation_rules(previous_rules['path'])

# Validate rollback
test_examples = [
    # ... test cases
]
validation_metrics = await studio._validate_translation_rules(
    loaded_rules, test_examples
)

print(f"Rollback validation:")
print(f"  - Pass Rate: {validation_metrics['pass_rate']:.4f}")
print(f"  - Avg Latency: {validation_metrics['avg_latency_ms']:.2f}ms")

# If validation passes, save as latest
if validation_metrics['pass_rate'] >= 0.95:
    await studio.save_translation_rules(
        loaded_rules,
        version=previous_rules['version']
    )
    print("Rollback successful and saved as latest")
```

#### Scenario 2: Emergency Rollback

```python
# In case of critical failure, rollback to last known good version
LAST_KNOWN_GOOD_VERSION = "1.0.0"
LAST_KNOWN_GOOD_ACCURACY = 0.95

# Find matching rules
matching_rules = [
    r for r in rules_list
    if r['version'] == LAST_KNOWN_GOOD_VERSION and r['accuracy'] >= LAST_KNOWN_GOOD_ACCURACY
]

if matching_rules:
    emergency_rules = matching_rules[0]
    print(f"Emergency rollback to: {emergency_rules['filename']}")
    
    loaded_rules = await studio.load_translation_rules(emergency_rules['path'])
    
    # Skip validation in emergency, just deploy
    await studio.save_translation_rules(loaded_rules, version=emergency_rules['version'])
    print("Emergency rollback completed")
else:
    print("ERROR: No suitable rollback version found!")
```

---

## Monitoring and Validation

### Continuous Monitoring

```python
# Monitor field detector performance
from ai_engine.monitoring.metrics import MetricsCollector

metrics_collector = MetricsCollector()

# Track inference metrics
async def monitor_field_detection():
    test_messages = [...]  # Representative test set
    
    latencies = []
    accuracies = []
    
    for message, expected_fields in test_messages:
        start_time = time.time()
        detected_fields = await detector.detect_boundaries(message)
        latency = (time.time() - start_time) * 1000
        
        latencies.append(latency)
        
        # Calculate accuracy
        accuracy = calculate_field_accuracy(detected_fields, expected_fields)
        accuracies.append(accuracy)
    
    # Report metrics
    avg_latency = np.mean(latencies)
    avg_accuracy = np.mean(accuracies)
    
    print(f"Field Detector Monitoring:")
    print(f"  - Avg Latency: {avg_latency:.2f}ms")
    print(f"  - Avg Accuracy: {avg_accuracy:.4f}")
    
    # Alert if degradation
    if avg_latency > 10.0 or avg_accuracy < 0.90:
        print("WARNING: Performance degradation detected!")
        return False
    
    return True

# Monitor translation studio performance
async def monitor_translation():
    test_translations = [...]  # Representative test set
    
    stats = studio.get_statistics()
    
    print(f"Translation Studio Monitoring:")
    print(f"  - Total Translations: {stats['total_translations']}")
    print(f"  - Success Rate: {stats['successful_translations'] / stats['total_translations']:.4f}")
    print(f"  - Avg Accuracy: {stats['average_accuracy']:.4f}")
    print(f"  - Avg Latency: {stats['average_latency_ms']:.2f}ms")
    
    # Alert if degradation
    success_rate = stats['successful_translations'] / stats['total_translations']
    if success_rate < 0.95 or stats['average_latency_ms'] > 5.0:
        print("WARNING: Performance degradation detected!")
        return False
    
    return True
```

### Automated Validation

```python
# Run automated validation suite
async def validate_ml_pipelines():
    """Run comprehensive validation of ML pipelines."""
    
    print("Running ML Pipeline Validation...")
    
    # 1. Field Detector Validation
    print("\n1. Field Detector Validation:")
    detector_ok = await monitor_field_detection()
    
    # 2. Translation Studio Validation
    print("\n2. Translation Studio Validation:")
    studio_ok = await monitor_translation()
    
    # 3. Performance Tests
    print("\n3. Performance Tests:")
    from ai_engine.tests.test_ml_pipeline_performance import (
        TestFieldDetectorPerformance,
        TestTranslationStudioPerformance
    )
    
    # Run performance tests
    # ... (use pytest or run tests programmatically)
    
    # 4. Overall Status
    all_ok = detector_ok and studio_ok
    
    if all_ok:
        print("\n✓ All ML pipelines validated successfully")
    else:
        print("\n✗ ML pipeline validation failed - consider rollback")
    
    return all_ok

# Schedule regular validation
import asyncio

async def scheduled_validation():
    """Run validation every hour."""
    while True:
        try:
            await validate_ml_pipelines()
        except Exception as e:
            print(f"Validation error: {e}")
        
        # Wait 1 hour
        await asyncio.sleep(3600)
```

---

## Best Practices

### 1. Training Best Practices

- **Always use validation data**: Never train without a held-out validation set
- **Monitor metrics**: Track precision, recall, F1, and accuracy throughout training
- **Use early stopping**: Prevent overfitting with patience-based early stopping
- **Save checkpoints frequently**: Enable recovery from training failures
- **Version your models**: Use semantic versioning for model/rules versions
- **Document changes**: Keep a changelog of what changed between versions

### 2. Deployment Best Practices

- **Test before deployment**: Always validate on test set before deploying
- **Gradual rollout**: Deploy to canary environment first
- **Monitor closely**: Watch metrics closely after deployment
- **Have rollback plan**: Always have previous version ready for rollback
- **Automate validation**: Use automated tests to catch regressions

### 3. Rollback Best Practices

- **Keep multiple versions**: Maintain at least 3-5 previous versions
- **Document rollback triggers**: Define clear criteria for when to rollback
- **Test rollback procedure**: Regularly test rollback in staging
- **Communicate**: Notify team when rollback is performed
- **Root cause analysis**: Always investigate why rollback was needed

### 4. Monitoring Best Practices

- **Set up alerts**: Configure alerts for performance degradation
- **Track trends**: Monitor metrics over time, not just point values
- **Log everything**: Comprehensive logging aids debugging
- **Regular audits**: Periodically review model performance
- **A/B testing**: Compare new versions against old in production

### 5. Data Management Best Practices

- **Version training data**: Track which data was used for each model version
- **Data quality checks**: Validate data quality before training
- **Balanced datasets**: Ensure training data is representative
- **Regular updates**: Retrain periodically with new data
- **Data retention**: Keep historical data for retraining

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Training Loss Not Decreasing

**Symptoms**: Loss plateaus or increases during training

**Solutions**:
1. Reduce learning rate
2. Check data quality and labels
3. Increase model capacity
4. Add regularization
5. Check for data leakage

#### Issue: High Inference Latency

**Symptoms**: Predictions take too long

**Solutions**:
1. Optimize model architecture
2. Use model quantization
3. Batch predictions
4. Cache frequent predictions
5. Use GPU acceleration

#### Issue: Poor Generalization

**Symptoms**: Good training metrics, poor validation metrics

**Solutions**:
1. Add more training data
2. Increase regularization
3. Use data augmentation
4. Simplify model
5. Check for overfitting

#### Issue: Rollback Fails

**Symptoms**: Cannot load previous checkpoint

**Solutions**:
1. Verify checkpoint file integrity
2. Check version compatibility
3. Ensure all dependencies match
4. Use emergency backup
5. Retrain from scratch if necessary

---

## Support and Contact

For issues or questions regarding ML pipeline operations:

- **Documentation**: See `/docs` directory
- **Tests**: Run `pytest ai_engine/tests/test_ml_pipeline_performance.py`
- **Logs**: Check `logs/ml_pipeline.log`
- **Metrics**: Access Prometheus metrics at `/metrics`

---

**Last Updated**: 2025-01-02
**Version**: 1.0.0