# Product 2: Translation Studio

## Overview

Translation Studio is QBITEL's automatic API and SDK generation platform that converts discovered protocols into modern REST APIs, GraphQL endpoints, and multi-language SDKs. It bridges the gap between legacy systems and modern application development, enabling seamless integration without manual coding.

---

## Problem Solved

### The Challenge

After understanding a legacy protocol, organizations still face:

- **Manual API development**: Weeks to months of custom coding
- **SDK maintenance burden**: Supporting multiple languages requires dedicated teams
- **Documentation lag**: APIs change faster than docs can be updated
- **Integration testing**: Each client requires extensive QA
- **Security implementation**: OAuth, JWT, rate limiting must be hand-coded

### The QBITEL Solution

Translation Studio automatically generates:
- OpenAPI 3.0 specifications from protocol grammar
- Production-ready SDKs in 6 languages
- API gateway configurations
- Authentication and rate limiting
- Comprehensive documentation and tests

---

## Key Features

### 1. Automatic API Generation

Converts protocol grammar into REST API specifications:

- **OpenAPI 3.0**: Industry-standard API documentation
- **Endpoint mapping**: Protocol messages become API endpoints
- **Request/response schemas**: Auto-generated from protocol fields
- **Validation rules**: Field constraints enforced at API level
- **Error handling**: Standardized error responses

### 2. Multi-Language SDK Generation

Generates type-safe SDKs in 6 programming languages:

| Language | Features |
|----------|----------|
| **Python** | Type hints, mypy compatible, async support |
| **TypeScript** | Full typing, ESM + CommonJS modules |
| **JavaScript** | Browser + Node.js, JSDoc annotations |
| **Go** | Interface-based, goroutine-safe |
| **Java** | Maven/Gradle, Android compatible |
| **C#** | .NET Framework + .NET Core, async/await |

### 3. Protocol Bridge (Real-Time Translation)

Enables bidirectional protocol translation:

- **Latency**: <10ms (P99: 25ms)
- **Throughput**: 10,000+ requests/second
- **Modes**: REST to legacy, legacy to REST, legacy to legacy
- **Streaming**: WebSocket support for real-time protocols

### 4. API Gateway Integration

Auto-generates configurations for popular API gateways:

- Kong
- AWS API Gateway
- Azure API Management
- Google Cloud Endpoints
- Nginx/OpenResty

### 5. Security Implementation

Built-in security features:

- **OAuth 2.0**: Authorization code, client credentials, PKCE
- **JWT validation**: Token verification and claims extraction
- **API key management**: Generation, rotation, revocation
- **Rate limiting**: Configurable per client/endpoint
- **mTLS**: Mutual TLS with quantum-safe certificates

---

## Technical Architecture

### Component Overview

```
Protocol Grammar (from Discovery)
    |
    v
+-------------------+
| API Generator     |
| - Schema mapping  |
| - Endpoint design |
| - OpenAPI output  |
+-------------------+
    |
    v
+-------------------+
| SDK Generator     |
| - Code templates  |
| - Type generation |
| - Test generation |
+-------------------+
    |
    v
+-------------------+
| Protocol Bridge   |
| - Translation     |
| - Routing         |
| - Caching         |
+-------------------+
```

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| API Endpoints | `api_endpoints.py` | REST API implementation (1,200+ LOC) |
| Enhanced Discovery | `enhanced_discovery.py` | Protocol analysis enhancement |
| API Generator | `api_generation/api_generator.py` | OpenAPI specification generation |
| Code Generator | `code_generation/code_generator.py` | Multi-language SDK generation |
| Protocol Bridge | `protocol_bridge/protocol_bridge.py` | Real-time protocol translation |

### Data Models

```python
@dataclass
class TranslationRequest:
    protocol_id: str                # Discovered protocol ID
    target_api_style: str           # "rest", "graphql", "grpc"
    target_languages: List[str]     # SDK languages to generate
    security_level: str             # "public", "authenticated", "mTLS"
    generate_documentation: bool    # Include API docs
    generate_tests: bool            # Include test suites
    api_base_path: str              # Base path for endpoints

@dataclass
class TranslationResult:
    request_id: str
    protocol_type: str
    api_specification: OpenAPISpec
    generated_sdks: List[SDK]
    gateway_config: Dict
    documentation: str
    processing_time: float
    status: str

@dataclass
class SDK:
    language: str
    name: str
    version: str
    files: List[GeneratedFile]
    dependencies: Dict[str, str]
    sdk_id: str
    download_url: str
```

---

## API Reference

### Protocol Discovery & API Generation

```http
POST /api/v1/translation/discover
Content-Type: application/json
X-API-Key: your_api_key

{
    "messages": [
        "base64_encoded_sample_1",
        "base64_encoded_sample_2"
    ],
    "target_api_style": "rest",
    "target_languages": ["python", "typescript", "go"],
    "security_level": "authenticated",
    "generate_documentation": true,
    "generate_tests": true,
    "api_base_path": "/api/v1/banking"
}
```

### Response

```json
{
    "request_id": "req_ts_abc123",
    "protocol_type": "ISO8583_Banking_v2",
    "confidence": 0.94,
    "api_specification": {
        "openapi": "3.0.0",
        "info": {
            "title": "ISO8583 Banking API",
            "version": "1.0.0",
            "description": "Auto-generated API for ISO8583 Banking protocol"
        },
        "servers": [
            {"url": "https://api.example.com/api/v1/banking"}
        ],
        "paths": {
            "/transactions": {
                "post": {
                    "operationId": "createTransaction",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Transaction"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Transaction processed",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/TransactionResponse"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "Transaction": {
                    "type": "object",
                    "properties": {
                        "messageType": {"type": "string", "pattern": "^[0-9]{4}$"},
                        "pan": {"type": "string", "maxLength": 19},
                        "amount": {"type": "number"},
                        "currency": {"type": "string", "pattern": "^[A-Z]{3}$"}
                    }
                }
            }
        }
    },
    "generated_sdks": [
        {
            "language": "python",
            "name": "iso8583-banking-python-sdk",
            "version": "1.0.0",
            "files_count": 15,
            "sdk_id": "sdk_py_12345",
            "download_url": "/api/v1/translation/download-sdk/sdk_py_12345"
        },
        {
            "language": "typescript",
            "name": "iso8583-banking-typescript-sdk",
            "version": "1.0.0",
            "files_count": 12,
            "sdk_id": "sdk_ts_67890",
            "download_url": "/api/v1/translation/download-sdk/sdk_ts_67890"
        }
    ],
    "processing_time": 2.45,
    "status": "completed"
}
```

### SDK Generation

```http
POST /api/v1/translation/generate-sdk
Content-Type: application/json

{
    "protocol_id": "proto_iso8583_001",
    "languages": ["python", "typescript", "go", "java"],
    "options": {
        "include_tests": true,
        "include_mocks": true,
        "authentication_type": "oauth2",
        "retry_enabled": true,
        "logging_enabled": true
    }
}
```

### Protocol Translation (Real-Time)

```http
POST /api/v1/translation/translate
Content-Type: application/json

{
    "source_protocol": "rest_json",
    "target_protocol": "iso8583",
    "message": {
        "messageType": "0200",
        "pan": "4111111111111111",
        "amount": 100.00,
        "currency": "USD"
    }
}

Response:
{
    "translated_message": "MDIwMDcwMzgyMDAyMDAwMDAwMDA0MTExMTExMTExMTExMTExMTAwMDAwMDAwMDEwMDAwODQw",
    "translation_time_ms": 5,
    "validation": {
        "valid": true,
        "errors": []
    }
}
```

### Streaming Translation

```http
POST /api/v1/translation/streaming/create
Content-Type: application/json

{
    "source_protocol": "modbus_tcp",
    "target_protocol": "rest_json",
    "connection_type": "websocket",
    "options": {
        "buffer_size": 1000,
        "flush_interval_ms": 100
    }
}

Response:
{
    "connection_id": "conn_ws_12345",
    "websocket_url": "wss://api.qbitel.com/translation/ws/conn_ws_12345",
    "status": "active"
}
```

### Additional Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/translation/download-sdk/{sdk_id}` | GET | Download generated SDK |
| `/api/v1/translation/streaming/{conn_id}/status` | GET | Check streaming connection |
| `/api/v1/translation/streaming/{conn_id}` | DELETE | Close streaming connection |
| `/api/v1/translation/knowledge/patterns/{protocol}` | GET | Get protocol patterns |
| `/api/v1/translation/knowledge/templates/{language}` | GET | Get code templates |
| `/api/v1/translation/status` | GET | Service status |
| `/api/v1/translation/metrics` | GET | Performance metrics |

---

## Generated SDK Features

### Python SDK Example

```python
# Auto-generated SDK structure
iso8583_banking_sdk/
├── __init__.py
├── client.py              # Main client class
├── models/
│   ├── __init__.py
│   ├── transaction.py     # Transaction model
│   ├── response.py        # Response models
│   └── errors.py          # Error types
├── api/
│   ├── __init__.py
│   ├── transactions.py    # Transaction endpoints
│   └── inquiries.py       # Inquiry endpoints
├── auth/
│   ├── __init__.py
│   ├── oauth2.py          # OAuth2 authentication
│   └── api_key.py         # API key authentication
├── utils/
│   ├── __init__.py
│   ├── retry.py           # Retry logic
│   └── logging.py         # Structured logging
└── tests/
    ├── __init__.py
    ├── test_client.py
    └── test_transactions.py
```

### SDK Usage (Python)

```python
from iso8583_banking_sdk import BankingClient
from iso8583_banking_sdk.models import Transaction
from iso8583_banking_sdk.auth import OAuth2

# Initialize client with OAuth2
auth = OAuth2(
    client_id="your_client_id",
    client_secret="your_client_secret",
    token_url="https://auth.example.com/token"
)

client = BankingClient(
    base_url="https://api.example.com",
    auth=auth,
    timeout=30,
    retry_enabled=True,
    max_retries=3
)

# Create transaction
transaction = Transaction(
    message_type="0200",
    pan="4111111111111111",
    amount=100.00,
    currency="USD",
    merchant_id="MERCHANT001"
)

# Send transaction (type-safe, with auto-completion)
response = client.transactions.create(transaction)

print(f"Response code: {response.response_code}")
print(f"Authorization: {response.authorization_code}")
print(f"Transaction ID: {response.transaction_id}")

# Async support
async with BankingClient(base_url="...", auth=auth) as client:
    response = await client.transactions.create_async(transaction)
```

### SDK Usage (TypeScript)

```typescript
import { BankingClient, Transaction } from 'iso8583-banking-sdk';
import { OAuth2 } from 'iso8583-banking-sdk/auth';

// Initialize client
const auth = new OAuth2({
  clientId: 'your_client_id',
  clientSecret: 'your_client_secret',
  tokenUrl: 'https://auth.example.com/token'
});

const client = new BankingClient({
  baseUrl: 'https://api.example.com',
  auth,
  timeout: 30000,
  retryEnabled: true
});

// Create transaction (fully typed)
const transaction: Transaction = {
  messageType: '0200',
  pan: '4111111111111111',
  amount: 100.00,
  currency: 'USD',
  merchantId: 'MERCHANT001'
};

// Send transaction
const response = await client.transactions.create(transaction);

console.log(`Response code: ${response.responseCode}`);
console.log(`Authorization: ${response.authorizationCode}`);
```

### SDK Usage (Go)

```go
package main

import (
    "context"
    "fmt"
    banking "github.com/yazhsab/iso8583-banking-sdk"
)

func main() {
    // Initialize client
    client := banking.NewClient(
        banking.WithBaseURL("https://api.example.com"),
        banking.WithOAuth2(
            "your_client_id",
            "your_client_secret",
            "https://auth.example.com/token",
        ),
        banking.WithTimeout(30 * time.Second),
        banking.WithRetry(3),
    )

    // Create transaction
    tx := &banking.Transaction{
        MessageType: "0200",
        PAN:         "4111111111111111",
        Amount:      10000, // cents
        Currency:    "USD",
        MerchantID:  "MERCHANT001",
    }

    // Send transaction
    resp, err := client.Transactions.Create(context.Background(), tx)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Response code: %s\n", resp.ResponseCode)
    fmt.Printf("Authorization: %s\n", resp.AuthorizationCode)
}
```

---

## Performance Metrics

### Throughput

| Operation | Throughput | Notes |
|-----------|-----------|-------|
| Protocol Discovery | < 3 seconds | Full analysis |
| API Generation | < 2 seconds | OpenAPI spec |
| SDK Generation | < 5 seconds/language | Including tests |
| Protocol Translation | 10,000+ req/sec | Real-time bridge |
| Batch Translation | 50,000+ msg/sec | Bulk processing |

### Latency

| Operation | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| API Generation | 1.5s | 1.8s | 2.0s |
| SDK Generation | 4s | 4.5s | 5s |
| Single Translation | 5ms | 8ms | 10ms |
| Batch Translation | 50ms | 80ms | 100ms |

### Scalability

| Metric | Value |
|--------|-------|
| Concurrent SDK generations | 100+ |
| Concurrent translations | 10,000+ |
| Horizontal scaling | Linear |
| Memory per translation | <10MB |

---

## Configuration

### Environment Variables

```bash
# Core Settings
QBITEL_TRANSLATION_ENABLED=true
QBITEL_TRANSLATION_MAX_FILE_SIZE=10485760
QBITEL_TRANSLATION_CONFIDENCE_THRESHOLD=0.7
QBITEL_TRANSLATION_MAX_CONCURRENT=10

# SDK Generation
QBITEL_SDK_LANGUAGES=python,typescript,javascript,go,java,csharp
QBITEL_SDK_INCLUDE_TESTS=true
QBITEL_SDK_INCLUDE_MOCKS=true

# Protocol Bridge
QBITEL_BRIDGE_ENABLED=true
QBITEL_BRIDGE_MAX_CONNECTIONS=1000
QBITEL_BRIDGE_BUFFER_SIZE=10000

# Caching
QBITEL_TRANSLATION_CACHE_ENABLED=true
QBITEL_TRANSLATION_CACHE_TTL=3600
```

### YAML Configuration

```yaml
translation:
  enabled: true
  max_file_size: 10485760  # 10MB
  confidence_threshold: 0.7
  max_concurrent_operations: 10

  supported_languages:
    - python
    - typescript
    - javascript
    - go
    - java
    - csharp

  api_styles:
    - rest
    - graphql
    - grpc
    - websocket

  sdk_generation:
    include_tests: true
    include_mocks: true
    include_docs: true
    authentication_types:
      - oauth2
      - jwt
      - api_key
      - mtls

  protocol_bridge:
    enabled: true
    max_connections: 1000
    buffer_size: 10000
    flush_interval_ms: 100

  cache:
    enabled: true
    ttl: 3600
    max_size: 1000
    use_redis: false
```

---

## Monitoring & Observability

### Prometheus Metrics

```
# Request metrics
qbitel_translation_requests_total{operation="discover|generate_api|generate_sdk|translate"}
qbitel_translation_request_duration_seconds{operation}
qbitel_translation_errors_total{operation, error_type}

# Discovery metrics
qbitel_translation_protocol_discoveries_total{protocol}
qbitel_translation_protocol_discovery_confidence{protocol}

# SDK generation metrics
qbitel_translation_sdk_generations_total{language}
qbitel_translation_code_generation_duration_seconds{language}
qbitel_translation_sdk_size_bytes{language}

# Bridge metrics
qbitel_translation_bridge_connections_active
qbitel_translation_bridge_messages_translated_total
qbitel_translation_bridge_latency_seconds

# System metrics
qbitel_translation_cpu_usage_percent
qbitel_translation_memory_usage_bytes
qbitel_translation_service_health
```

### Health Check

```http
GET /api/v1/translation/status

{
    "status": "healthy",
    "version": "1.0.0",
    "components": {
        "api_generator": "healthy",
        "sdk_generator": "healthy",
        "protocol_bridge": "healthy",
        "cache": "healthy"
    },
    "supported_languages": ["python", "typescript", "go", "java", "csharp", "javascript"],
    "active_connections": 156,
    "cache_size": 4521
}
```

---

## Use Cases

### Use Case 1: Legacy Banking Integration

**Scenario**: Connect 1985 mainframe to modern fraud detection system

```python
# Step 1: Discover ISO-8583 protocol
discovery_result = client.discovery.analyze(
    samples=mainframe_traffic_samples
)

# Step 2: Generate REST API
api_spec = client.translation.generate_api(
    protocol_id=discovery_result.protocol_id,
    api_style="rest",
    security_level="authenticated"
)

# Step 3: Generate Python SDK for fraud system
sdk = client.translation.generate_sdk(
    protocol_id=discovery_result.protocol_id,
    languages=["python"]
)

# Step 4: Deploy protocol bridge
bridge = client.translation.create_bridge(
    source_protocol="rest_json",
    target_protocol="iso8583"
)

# Now fraud detection can call REST API, bridge translates to mainframe
```

### Use Case 2: SCADA Modernization

**Scenario**: Expose Modbus devices via REST API

```python
# Discover Modbus protocol
modbus_protocol = client.discovery.analyze(samples=modbus_samples)

# Generate REST API for SCADA operations
api = client.translation.generate_api(
    protocol_id=modbus_protocol.id,
    api_style="rest"
)

# Deploy real-time bridge
bridge = client.translation.create_bridge(
    source_protocol="rest",
    target_protocol="modbus_tcp",
    options={
        "real_time": True,
        "latency_target_ms": 10
    }
)

# Cloud dashboard can now monitor SCADA via REST
```

### Use Case 3: Healthcare Interoperability

**Scenario**: Connect HL7 v2 system to FHIR-based EHR

```python
# Discover HL7 v2 protocol
hl7_v2 = client.discovery.analyze(samples=hl7_samples)

# Generate FHIR-compatible REST API
api = client.translation.generate_api(
    protocol_id=hl7_v2.id,
    api_style="rest",
    output_format="fhir_r4"
)

# Bridge legacy HL7 to FHIR
bridge = client.translation.create_bridge(
    source_protocol="hl7_v2",
    target_protocol="fhir_r4"
)
```

---

## Comparison: Manual vs. Translation Studio

| Aspect | Manual Development | Translation Studio |
|--------|-------------------|-------------------|
| **API development time** | 2-4 weeks | < 2 seconds |
| **SDK development time** | 4-8 weeks per language | < 5 seconds per language |
| **Documentation** | Manual, often outdated | Auto-generated, always current |
| **Test coverage** | Variable | 80%+ automated |
| **Maintenance** | Ongoing manual effort | Auto-updated |
| **Language support** | 1-2 languages | 6 languages |
| **Security implementation** | Manual, error-prone | Built-in, standardized |

---

## Conclusion

Translation Studio eliminates the manual effort of API and SDK development, enabling organizations to expose legacy systems via modern interfaces in minutes instead of months. Its combination of automatic code generation, real-time protocol bridging, and comprehensive security features makes legacy integration accessible to any development team.
