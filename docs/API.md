# QBITEL Bridge - API Documentation

## 🚀 **API Overview**

The QBITEL Bridge System provides both REST and Python APIs for protocol discovery, classification, and management operations.

## 📋 **Table of Contents**

- [Authentication](#authentication)
- [REST API Endpoints](#rest-api-endpoints)
- [Python API Reference](#python-api-reference)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

## 🔐 **Authentication**

### **API Key Authentication**

```http
X-API-Key: your_api_key_here
```

### **JWT Authentication**

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## 🌐 **REST API Endpoints**

### **Protocol Discovery**

#### **Analyze Traffic**
Discover protocols from network traffic data.

```http
POST /api/v1/discovery/analyze
Content-Type: application/json
X-API-Key: your_api_key

{
    "traffic_data": [
        "R0VUIC9hcGkvdXNlcnMgSFRUUC8xLjENCkhvc3Q6IGFwaS5leGFtcGxlLmNvbQ0KDQo=",
        "UE9TVCAvYXBpL2xvZ2luIEhUVFAvMS4xDQpDb250ZW50LVR5cGU6IGFwcGxpY2F0aW9uL2pzb24NCg0K"
    ],
    "options": {
        "max_protocols": 5,
        "confidence_threshold": 0.7,
        "enable_adaptive_learning": true,
        "cache_results": true
    }
}
```

**Response:**
```json
{
    "success": true,
    "request_id": "req_12345",
    "processing_time_ms": 245,
    "cache_hit": false,
    "discovered_protocols": [
        {
            "id": "proto_001",
            "name": "http_variant_1",
            "confidence": 0.94,
            "description": "HTTP-like protocol with custom headers",
            "statistics": {
                "total_messages": 100,
                "avg_message_length": 256.7,
                "entropy": 6.24,
                "binary_ratio": 0.12
            },
            "grammar": {
                "start_symbol": "HTTP_MESSAGE",
                "rules": [
                    {
                        "lhs": "HTTP_MESSAGE",
                        "rhs": ["METHOD", "PATH", "VERSION", "HEADERS", "BODY"],
                        "probability": 1.0
                    }
                ]
            },
            "parser": {
                "id": "parser_001",
                "protocol_name": "http_variant_1",
                "parse_success_rate": 0.98,
                "performance_ms": 2.3
            },
            "validation_rules": [
                {
                    "name": "method_validation",
                    "type": "regex",
                    "pattern": "^(GET|POST|PUT|DELETE)\\s",
                    "confidence": 0.95
                }
            ]
        }
    ],
    "metadata": {
        "model_versions": {
            "cnn": "v1.2.3",
            "lstm": "v1.2.1", 
            "random_forest": "v1.1.8"
        },
        "processing_stats": {
            "statistical_analysis_ms": 45,
            "grammar_learning_ms": 120,
            "parser_generation_ms": 35,
            "classification_ms": 25,
            "validation_ms": 20
        }
    }
}
```

#### **Classify Message**
Classify a single message against known protocols.

```http
POST /api/v1/discovery/classify
Content-Type: application/json

{
    "message": "R0VUIC9hcGkvdXNlcnMgSFRUUC8xLjE=",
    "options": {
        "return_probabilities": true,
        "confidence_threshold": 0.5
    }
}
```

**Response:**
```json
{
    "success": true,
    "classification": {
        "protocol": "http",
        "confidence": 0.97,
        "class_probabilities": {
            "http": 0.97,
            "custom_protocol_1": 0.02,
            "unknown": 0.01
        }
    },
    "parsed_fields": {
        "method": "GET",
        "path": "/api/users",
        "version": "HTTP/1.1"
    },
    "validation": {
        "is_valid": true,
        "passed_rules": ["method_check", "version_check"],
        "failed_rules": [],
        "validation_score": 1.0
    }
}
```

### **Protocol Management**

#### **List Protocols**
Get list of discovered protocols.

```http
GET /api/v1/protocols?limit=10&offset=0&sort=confidence
```

**Response:**
```json
{
    "success": true,
    "protocols": [
        {
            "id": "proto_001",
            "name": "http_variant_1",
            "confidence": 0.94,
            "created_at": "2024-01-15T10:30:00Z",
            "last_updated": "2024-01-15T11:45:00Z",
            "message_count": 1547,
            "success_rate": 0.98
        }
    ],
    "pagination": {
        "total": 23,
        "limit": 10,
        "offset": 0,
        "has_more": true
    }
}
```

#### **Get Protocol Details**
Get detailed information about a specific protocol.

```http
GET /api/v1/protocols/{protocol_id}
```

**Response:**
```json
{
    "success": true,
    "protocol": {
        "id": "proto_001",
        "name": "http_variant_1",
        "confidence": 0.94,
        "description": "HTTP-like protocol with custom headers",
        "grammar": {...},
        "parser": {...},
        "validation_rules": [...],
        "statistics": {...},
        "metadata": {...}
    }
}
```

#### **Update Protocol**
Update protocol information or retrain with new data.

```http
PUT /api/v1/protocols/{protocol_id}
Content-Type: application/json

{
    "name": "updated_protocol_name",
    "description": "Updated description",
    "retrain": true,
    "training_data": ["base64_message_1", "base64_message_2"]
}
```

#### **Delete Protocol**
Delete a discovered protocol.

```http
DELETE /api/v1/protocols/{protocol_id}
```

### **System Management**

#### **Health Check**
Get system health status.

```http
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "timestamp": 1640995200,
    "uptime_seconds": 86400,
    "version": "1.0.0",
    "components": {
        "database": {
            "status": "healthy",
            "response_time_ms": 12
        },
        "redis": {
            "status": "healthy",
            "response_time_ms": 3
        },
        "ai_models": {
            "status": "healthy",
            "models_loaded": 3
        }
    },
    "metrics": {
        "requests_per_second": 124.5,
        "avg_response_time_ms": 45.2,
        "memory_usage_mb": 1024.3,
        "cpu_percent": 23.5,
        "cache_hit_rate": 0.94
    }
}
```

#### **Metrics**
Get Prometheus metrics.

```http
GET /metrics
```

**Response:**
```
# HELP protocol_discovery_requests_total Total protocol discovery requests
# TYPE protocol_discovery_requests_total counter
protocol_discovery_requests_total{status="success"} 1547
protocol_discovery_requests_total{status="error"} 23

# HELP protocol_discovery_duration_seconds Request duration in seconds
# TYPE protocol_discovery_duration_seconds histogram
protocol_discovery_duration_seconds_bucket{le="0.005"} 234
protocol_discovery_duration_seconds_bucket{le="0.01"} 567
protocol_discovery_duration_seconds_bucket{le="0.025"} 892
```

## 🐍 **Python API Reference**

### **Protocol Discovery Orchestrator**

```python
from ai_engine.discovery.protocol_discovery_orchestrator import ProtocolDiscoveryOrchestrator

# Initialize orchestrator
orchestrator = ProtocolDiscoveryOrchestrator(config)

# Discover protocols
traffic_data = [b"GET /api HTTP/1.1\r\n\r\n", b"POST /data HTTP/1.1\r\n\r\n"]
result = await orchestrator.discover_protocol(traffic_data)

# Get discovered protocols
for protocol in result.discovered_protocols:
    print(f"Found protocol: {protocol.name} (confidence: {protocol.confidence})")
```

### **Individual Components**

#### **Statistical Analyzer**

```python
from ai_engine.discovery.statistical_analyzer import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Analyze traffic patterns
pattern = await analyzer.analyze_traffic(traffic_data)
print(f"Entropy: {pattern.entropy}, Binary ratio: {pattern.binary_ratio}")

# Detect field boundaries
boundaries = await analyzer.detect_field_boundaries(messages)
for boundary in boundaries:
    print(f"Separator: {boundary.separator}, Position: {boundary.position}")
```

#### **Grammar Learner**

```python
from ai_engine.discovery.grammar_learner import GrammarLearner

learner = GrammarLearner()

# Learn PCFG from messages
grammar = await learner.learn_pcfg(messages)
print(f"Learned {len(grammar.rules)} grammar rules")

# Refine with EM algorithm
refined_grammar = await learner.refine_with_em(grammar, messages, max_iterations=50)
```

#### **Protocol Classifier**

```python
from ai_engine.discovery.protocol_classifier import ProtocolClassifier

classifier = ProtocolClassifier(config)

# Train models
training_data = {
    'http': [b"GET /", b"POST /"],
    'smtp': [b"HELO ", b"MAIL FROM:"]
}
await classifier.train_models(training_data)

# Classify message
prediction = await classifier.classify_message(b"GET /api HTTP/1.1")
print(f"Protocol: {prediction.protocol}, Confidence: {prediction.confidence}")
```

#### **Message Validator**

```python
from ai_engine.discovery.message_validator import MessageValidator, ValidationRule

validator = MessageValidator()

# Add validation rules
rule = ValidationRule(
    name="http_method",
    rule_type="regex",
    pattern=rb"^(GET|POST|PUT|DELETE)",
    required=True
)
validator.add_rule(rule)

# Validate message
result = await validator.validate_message(b"GET /api HTTP/1.1")
print(f"Valid: {result.is_valid}, Score: {result.validation_score}")
```

## 📊 **Data Models**

### **DiscoveryResult**

```python
@dataclass
class DiscoveryResult:
    success: bool
    discovered_protocols: List[DiscoveredProtocol]
    processing_time: float
    error_message: Optional[str] = None
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### **DiscoveredProtocol**

```python
@dataclass
class DiscoveredProtocol:
    name: str
    confidence: float
    grammar: Grammar
    parser: GeneratedParser
    statistics: TrafficPattern
    validation_rules: List[ValidationRule]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### **TrafficPattern**

```python
@dataclass
class TrafficPattern:
    total_messages: int
    message_lengths: List[int]
    entropy: float
    binary_ratio: float
    detected_patterns: List[DetectedPattern]
    field_boundaries: List[FieldBoundary]
    processing_time: float
```

### **Grammar**

```python
@dataclass
class Grammar:
    start_symbol: str
    rules: List[PCFGRule]
    terminals: Set[bytes]
    non_terminals: Set[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### **ValidationResult**

```python
@dataclass
class ValidationResult:
    is_valid: bool
    validation_score: float
    passed_rules: List[str]
    failed_rules: List[str]
    error_details: List[str]
    processing_time: float
```

## ⚠️ **Error Handling**

### **Error Response Format**

```json
{
    "success": false,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid input data format",
        "details": {
            "field": "traffic_data",
            "issue": "Expected base64 encoded strings"
        },
        "request_id": "req_12345",
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

### **Error Codes**

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid input data or parameters |
| `AUTHENTICATION_ERROR` | 401 | Missing or invalid API key/token |
| `AUTHORIZATION_ERROR` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Requested resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

### **Python Exceptions**

```python
from ai_engine.core.exceptions import (
    ProtocolDiscoveryError,
    ValidationError,
    ClassificationError,
    ParsingError,
    ConfigurationError
)

try:
    result = await orchestrator.discover_protocol(traffic_data)
except ValidationError as e:
    print(f"Validation failed: {e}")
except ClassificationError as e:
    print(f"Classification failed: {e}")
```

## 🚦 **Rate Limiting**

### **Rate Limits**

| Endpoint | Rate Limit | Window |
|----------|------------|--------|
| `/api/v1/discovery/analyze` | 10 requests | 1 minute |
| `/api/v1/discovery/classify` | 100 requests | 1 minute |
| `/api/v1/protocols/*` | 50 requests | 1 minute |
| `/health` | 1000 requests | 1 minute |

### **Rate Limit Headers**

```http
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1640995260
```

## 📚 **Examples**

### **Complete Protocol Discovery**

```python
import asyncio
from ai_engine.discovery.protocol_discovery_orchestrator import ProtocolDiscoveryOrchestrator
from ai_engine.core.config_manager import load_config

async def discover_protocols():
    # Load configuration
    config = load_config("config/qbitel.yaml")
    
    # Initialize orchestrator
    orchestrator = ProtocolDiscoveryOrchestrator(config.ai_engine)
    
    # Sample network traffic
    traffic_data = [
        b"GET /api/v1/users HTTP/1.1\r\nHost: api.example.com\r\nAuthorization: Bearer token123\r\n\r\n",
        b"POST /api/v1/login HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: 45\r\n\r\n{\"username\":\"alice\",\"password\":\"secret\"}",
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 123\r\n\r\n{\"users\":[{\"id\":1,\"name\":\"Alice\"}]}",
        b"CMD:CONNECT|HOST:server.example.com|PORT:443|PROTO:TLS",
        b"CMD:AUTH|USER:bob|PASS:password123|SESSION:abc456def789",
        b"RESPONSE:OK|STATUS:CONNECTED|SESSION:abc456def789|TIMEOUT:3600"
    ]
    
    # Perform discovery
    result = await orchestrator.discover_protocol(traffic_data)
    
    if result.success:
        print(f"Discovery completed in {result.processing_time:.2f}s")
        print(f"Found {len(result.discovered_protocols)} protocols:")
        
        for i, protocol in enumerate(result.discovered_protocols, 1):
            print(f"\n{i}. Protocol: {protocol.name}")
            print(f"   Confidence: {protocol.confidence:.2%}")
            print(f"   Messages analyzed: {protocol.statistics.total_messages}")
            print(f"   Average entropy: {protocol.statistics.entropy:.2f}")
            
            # Test parser
            if protocol.parser:
                test_message = traffic_data[0]  # Use first message as test
                parse_result = await protocol.parser.parse(test_message)
                if parse_result.success:
                    print(f"   Parser test: SUCCESS")
                    print(f"   Parsed fields: {list(parse_result.parsed_fields.keys())}")
                else:
                    print(f"   Parser test: FAILED - {parse_result.error}")
            
            # Show validation rules
            if protocol.validation_rules:
                print(f"   Validation rules: {len(protocol.validation_rules)}")
                for rule in protocol.validation_rules[:3]:  # Show first 3
                    print(f"     - {rule.name} ({rule.rule_type})")
    else:
        print(f"Discovery failed: {result.error_message}")

# Run the example
asyncio.run(discover_protocols())
```

### **Real-time Classification**

```python
import asyncio
from ai_engine.discovery.protocol_classifier import ProtocolClassifier
from ai_engine.monitoring.enterprise_metrics import record_metric

async def classify_stream(classifier, message_stream):
    """Classify messages from a real-time stream."""
    
    for message in message_stream:
        try:
            # Classify message
            prediction = await classifier.classify_message(message)
            
            # Record metrics
            record_metric('classification_confidence', prediction.confidence)
            record_metric('classification_latency_ms', prediction.processing_time * 1000)
            
            # Process based on protocol
            if prediction.protocol == 'http':
                await handle_http_message(message, prediction)
            elif prediction.protocol == 'custom_protocol':
                await handle_custom_message(message, prediction)
            else:
                await handle_unknown_message(message, prediction)
                
        except Exception as e:
            logger.error(f"Classification error: {e}")
            record_metric('classification_errors', 1)

async def handle_http_message(message, prediction):
    """Handle HTTP protocol messages."""
    print(f"HTTP message detected (confidence: {prediction.confidence:.2%})")
    
async def handle_custom_message(message, prediction):
    """Handle custom protocol messages.""" 
    print(f"Custom protocol message detected (confidence: {prediction.confidence:.2%})")
    
async def handle_unknown_message(message, prediction):
    """Handle unknown protocol messages."""
    print(f"Unknown protocol message (confidence: {prediction.confidence:.2%})")
```

### **Batch Processing**

```python
from ai_engine.core.performance_optimizer import BatchProcessor

async def batch_discovery_example():
    """Example of batch processing for high-throughput scenarios."""
    
    batch_processor = BatchProcessor(batch_size=1000, flush_interval=5.0)
    orchestrator = ProtocolDiscoveryOrchestrator()
    
    async def process_batch(messages):
        """Process a batch of messages."""
        result = await orchestrator.discover_protocol(messages)
        print(f"Processed batch of {len(messages)} messages")
        return result
    
    # Simulate message stream
    message_stream = generate_message_stream()  # Your message generator
    
    async for message in message_stream:
        await batch_processor.add_item('discovery', message, process_batch)
    
    # Flush any remaining messages
    await batch_processor.flush_all()
```

## 📞 **Support**

For API-related questions and issues:

- **Documentation**: [docs/README.md](README.md)
- **Issues**: [GitHub Issues](https://github.com/yazhsab/qbitel-bridge/issues)
- **API Support**: api-support@qbitel.com

---

**API Version**: v2.1.0
**Last Updated**: 2025-11-22