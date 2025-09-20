# Protocol Intelligence Copilot API Documentation

## Overview

The Protocol Intelligence Copilot is an advanced AI-powered system that provides natural language interface for protocol analysis, security assessment, and compliance checking. It integrates Large Language Models (LLMs) with traditional protocol discovery techniques to deliver intelligent, context-aware insights.

## Table of Contents

1. [Authentication](#authentication)
2. [Core Endpoints](#core-endpoints)
3. [WebSocket API](#websocket-api)
4. [Request/Response Schemas](#requestresponse-schemas)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Monitoring](#monitoring)
8. [SDK Examples](#sdk-examples)

## Authentication

All API endpoints require authentication via JWT tokens. Include the token in the Authorization header:

```http
Authorization: Bearer <your-jwt-token>
```

### Login Endpoint

**POST** `/api/v1/auth/login`

```json
{
  "username": "your_username",
  "password": "your_password"
}
```

**Response:**
```json
{
  "success": true,
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "user_id": "admin_001",
    "username": "admin",
    "role": "administrator",
    "permissions": ["protocol_discovery", "copilot_access", "analytics_access"]
  }
}
```

## Core Endpoints

### 1. Copilot Query Processing

**POST** `/api/v1/copilot/query`

Process natural language queries about protocols, security, and compliance.

**Request Body:**
```json
{
  "query": "Analyze the security implications of this HTTP traffic",
  "query_type": "security_assessment",
  "session_id": "session_12345",
  "context": {
    "protocol_type": "HTTP",
    "previous_analysis": "..."
  },
  "packet_data": "474554202f746573742048545450...",
  "enable_learning": true,
  "preferred_provider": "openai"
}
```

**Response:**
```json
{
  "success": true,
  "response": "Based on the HTTP traffic analysis, I've identified several security considerations...",
  "query_type": "security_assessment",
  "confidence": 0.89,
  "sources": [
    {
      "content": "HTTP security best practices...",
      "similarity": 0.95,
      "metadata": {"source": "security_knowledge_base"}
    }
  ],
  "suggestions": [
    "Consider implementing HTTPS encryption",
    "Review authentication mechanisms",
    "Monitor for unusual request patterns"
  ],
  "session_id": "session_12345",
  "processing_time": 2.3,
  "llm_provider": "openai",
  "metadata": {
    "tokens_used": 150,
    "model": "gpt-4"
  }
}
```

### 2. Enhanced Protocol Discovery

**POST** `/api/v1/discover/enhanced`

Perform protocol discovery with LLM-enhanced analysis.

**Request Body:**
```json
{
  "packet_data": "474554202f746573742048545450...",
  "metadata": {
    "source_ip": "192.168.1.100",
    "dest_port": 80,
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "enable_llm_analysis": true,
  "llm_analysis_types": [
    "protocol_identification",
    "security_assessment",
    "compliance_check"
  ],
  "natural_language_explanation": true,
  "analysis_depth": "deep",
  "session_id": "discovery_session_001"
}
```

**Response:**
```json
{
  "success": true,
  "protocol_type": "HTTP",
  "confidence": 0.92,
  "fields": [
    {
      "name": "method",
      "value": "GET",
      "offset": 0,
      "length": 3,
      "semantic_type": "http_method"
    },
    {
      "name": "path",
      "value": "/test",
      "offset": 4,
      "length": 5,
      "semantic_type": "http_path"
    }
  ],
  "enhanced_analysis": {
    "llm_analysis": "This is a standard HTTP GET request to the /test endpoint...",
    "confidence": 0.89,
    "security_implications": "Low risk - standard HTTP request pattern",
    "compliance_notes": "Complies with HTTP/1.1 specification"
  },
  "natural_language_summary": "Discovered HTTP protocol with high confidence. The traffic shows a standard GET request pattern with no immediate security concerns.",
  "recommendations": [
    "Consider upgrading to HTTPS for enhanced security",
    "Monitor for unusual request patterns"
  ],
  "processing_time": 3.1,
  "llm_enabled": true
}
```

### 3. Field Detection with LLM Interpretation

**POST** `/api/v1/detect-fields/enhanced`

Detect protocol fields with natural language interpretation.

**Request Body:**
```json
{
  "message_data": "474554202f746573742048545450...",
  "protocol_type": "HTTP",
  "enable_llm_analysis": true,
  "field_types": ["headers", "body", "parameters"],
  "session_id": "field_session_001"
}
```

**Response:**
```json
{
  "success": true,
  "fields": [
    {
      "name": "Host",
      "value": "example.com",
      "field_type": "header",
      "confidence": 0.98
    }
  ],
  "llm_interpretation": {
    "interpretation": "The Host header indicates the target server...",
    "confidence": 0.87,
    "field_semantics": {
      "Host": {
        "purpose": "Specifies the target server hostname",
        "security_relevance": "Medium - can indicate potential host header attacks",
        "compliance_notes": "Required by HTTP/1.1 specification"
      }
    }
  },
  "processing_time": 1.4
}
```

### 4. Conversation Management

**GET** `/api/v1/copilot/sessions/{session_id}/history`

Retrieve conversation history for a session.

**Response:**
```json
{
  "success": true,
  "session_id": "session_12345",
  "messages": [
    {
      "id": "msg_001",
      "role": "user",
      "content": "Analyze this HTTP traffic",
      "timestamp": "2024-01-15T10:30:00Z",
      "metadata": {"query_type": "protocol_analysis"}
    },
    {
      "id": "msg_002",
      "role": "assistant",
      "content": "Based on the HTTP traffic analysis...",
      "timestamp": "2024-01-15T10:30:02Z",
      "metadata": {"confidence": 0.89, "llm_provider": "openai"}
    }
  ],
  "total_messages": 2,
  "session_created": "2024-01-15T10:30:00Z",
  "last_activity": "2024-01-15T10:30:02Z"
}
```

### 5. Knowledge Base Operations

**POST** `/api/v1/copilot/knowledge/search`

Search the protocol knowledge base.

**Request Body:**
```json
{
  "query": "HTTP security vulnerabilities",
  "limit": 10,
  "threshold": 0.7,
  "filters": {
    "category": "security",
    "protocol": "HTTP"
  }
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "document": {
        "id": "doc_001",
        "content": "HTTP security best practices include...",
        "metadata": {
          "category": "security",
          "protocol": "HTTP",
          "source": "OWASP"
        }
      },
      "similarity_score": 0.95,
      "relevance_explanation": "High relevance due to matching security context and HTTP protocol"
    }
  ],
  "total_results": 1,
  "query_time": 0.12
}
```

## WebSocket API

### Connection Endpoint

**WebSocket** `/api/v1/copilot/ws`

Real-time communication with the Protocol Intelligence Copilot.

**Connection Headers:**
```
Authorization: Bearer <your-jwt-token>
```

### Message Format

All WebSocket messages follow this format:

```json
{
  "type": "message_type",
  "data": {
    // Message-specific data
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "session_id": "session_12345"
}
```

### Message Types

#### 1. Query Message

**Client → Server:**
```json
{
  "type": "query",
  "data": {
    "query": "What security risks are associated with this protocol?",
    "query_type": "security_assessment",
    "packet_data": "474554202f746573742048545450...",
    "context": {}
  },
  "session_id": "session_12345"
}
```

#### 2. Response Message

**Server → Client:**
```json
{
  "type": "response",
  "data": {
    "response": "The security risks associated with this protocol include...",
    "confidence": 0.88,
    "suggestions": ["Enable HTTPS", "Implement authentication"],
    "processing_time": 2.1
  },
  "session_id": "session_12345"
}
```

#### 3. Typing Indicator

**Server → Client:**
```json
{
  "type": "typing",
  "data": {
    "is_typing": true
  },
  "session_id": "session_12345"
}
```

#### 4. Error Message

**Server → Client:**
```json
{
  "type": "error",
  "data": {
    "error_code": "PROCESSING_FAILED",
    "message": "Failed to process query due to temporary service unavailability",
    "retry_after": 30
  },
  "session_id": "session_12345"
}
```

## Request/Response Schemas

### Query Types

- `protocol_analysis` - General protocol analysis
- `security_assessment` - Security-focused analysis
- `field_detection` - Field identification and interpretation
- `compliance_check` - Compliance verification
- `anomaly_detection` - Anomaly identification
- `general_question` - General questions about protocols

### LLM Providers

- `openai` - OpenAI GPT models
- `anthropic` - Anthropic Claude models  
- `ollama` - Local Ollama deployment

### Analysis Depth Levels

- `basic` - Quick analysis with essential information
- `standard` - Standard analysis with moderate detail
- `deep` - Comprehensive analysis with extensive details

### Confidence Scores

All AI responses include confidence scores (0.0 - 1.0):
- `0.0 - 0.3` - Low confidence
- `0.4 - 0.6` - Medium confidence  
- `0.7 - 0.9` - High confidence
- `0.9 - 1.0` - Very high confidence

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": "INVALID_REQUEST",
  "message": "The request data is invalid or incomplete",
  "details": [
    {
      "code": "MISSING_FIELD",
      "message": "Field 'query' is required",
      "field": "query"
    }
  ],
  "timestamp": "2024-01-15T10:30:00Z",
  "correlation_id": "req_12345"
}
```

### Common Error Codes

- `AUTHENTICATION_FAILED` - Invalid or expired token
- `AUTHORIZATION_DENIED` - Insufficient permissions
- `INVALID_REQUEST` - Malformed request data
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `SERVICE_UNAVAILABLE` - Temporary service issues
- `PROCESSING_FAILED` - Analysis processing failed
- `LLM_SERVICE_ERROR` - LLM provider error
- `CONTEXT_TOO_LARGE` - Request context exceeds limits

### HTTP Status Codes

- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `429` - Too Many Requests
- `500` - Internal Server Error
- `503` - Service Unavailable

## Rate Limiting

API requests are rate limited per user:

- **Standard Users**: 100 requests per minute
- **Premium Users**: 500 requests per minute
- **Enterprise Users**: 1000 requests per minute

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642251600
```

## Monitoring

### Health Check

**GET** `/health`

```json
{
  "status": "healthy",
  "services": {
    "ai_engine": "healthy",
    "protocol_copilot": "healthy",
    "llm_service": {
      "openai": "healthy",
      "anthropic": "healthy",
      "ollama": "unavailable"
    },
    "rag_engine": "healthy"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Metrics Endpoint

**GET** `/metrics`

Returns Prometheus-formatted metrics for monitoring and alerting.

## SDK Examples

### Python SDK

```python
import asyncio
from cronos_ai_sdk import CronosAIClient, CopilotQuery

async def main():
    # Initialize client
    client = CronosAIClient(
        api_url="https://api.cronos-ai.com",
        auth_token="your-jwt-token"
    )
    
    # Process a copilot query
    query = CopilotQuery(
        query="Analyze the security of this HTTP request",
        query_type="security_assessment",
        packet_data=bytes.fromhex("474554202f746573742048545450...")
    )
    
    response = await client.copilot.process_query(query)
    print(f"Response: {response.response}")
    print(f"Confidence: {response.confidence}")
    
    # Enhanced protocol discovery
    discovery_result = await client.discover_protocol_enhanced(
        packet_data=bytes.fromhex("474554202f746573742048545450..."),
        enable_llm_analysis=True,
        analysis_depth="deep"
    )
    
    print(f"Protocol: {discovery_result.protocol_type}")
    print(f"Summary: {discovery_result.natural_language_summary}")

if __name__ == "__main__":
    asyncio.run(main())
```

### JavaScript SDK

```javascript
import { CronosAIClient } from '@cronos-ai/sdk';

const client = new CronosAIClient({
  apiUrl: 'https://api.cronos-ai.com',
  authToken: 'your-jwt-token'
});

// Process copilot query
async function analyzeProtocol() {
  const response = await client.copilot.processQuery({
    query: 'What are the security implications of this protocol?',
    queryType: 'security_assessment',
    packetData: '474554202f746573742048545450...',
    sessionId: 'session_001'
  });
  
  console.log('Response:', response.response);
  console.log('Confidence:', response.confidence);
  console.log('Suggestions:', response.suggestions);
}

// WebSocket example
const ws = client.copilot.createWebSocketConnection();

ws.on('response', (data) => {
  console.log('Received response:', data.response);
});

ws.send({
  type: 'query',
  data: {
    query: 'Analyze this network traffic',
    queryType: 'protocol_analysis'
  }
});
```

### cURL Examples

```bash
# Login
curl -X POST https://api.cronos-ai.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'

# Copilot query
curl -X POST https://api.cronos-ai.com/api/v1/copilot/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze this HTTP traffic for security issues",
    "query_type": "security_assessment",
    "packet_data": "474554202f746573742048545450..."
  }'

# Enhanced protocol discovery
curl -X POST https://api.cronos-ai.com/api/v1/discover/enhanced \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "packet_data": "474554202f746573742048545450...",
    "enable_llm_analysis": true,
    "natural_language_explanation": true,
    "analysis_depth": "deep"
  }'
```

## Best Practices

### 1. Session Management

- Use consistent session IDs for related queries
- Sessions expire after 1 hour of inactivity
- Store important context in session metadata

### 2. Performance Optimization

- Use appropriate analysis depth levels
- Enable caching for repeated queries
- Batch related requests when possible

### 3. Error Handling

- Always check the `success` field in responses
- Implement retry logic with exponential backoff
- Handle rate limiting gracefully

### 4. Security

- Never log or expose authentication tokens
- Validate all input data before sending
- Use HTTPS for all API communications

### 5. Monitoring

- Monitor response times and error rates
- Set up alerts for service degradation
- Track token usage and costs

## Support and Resources

- **Documentation**: https://docs.cronos-ai.com
- **API Reference**: https://api.cronos-ai.com/docs
- **Support Portal**: https://support.cronos-ai.com
- **GitHub Repository**: https://github.com/cronos-ai/cronos-ai-engine

## Changelog

### Version 2.0.0 (Current)
- Added Protocol Intelligence Copilot
- Enhanced protocol discovery with LLM analysis
- WebSocket real-time communication
- Advanced monitoring and observability

### Version 1.0.0
- Initial release with basic protocol discovery
- Traditional ML-based field detection
- REST API endpoints