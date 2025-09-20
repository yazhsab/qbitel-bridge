"""
CRONOS AI Engine - API Schemas
Pydantic models for request/response validation and documentation.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

class QueryType(str, Enum):
    """Types of copilot queries."""
    PROTOCOL_ANALYSIS = "protocol_analysis"
    SECURITY_ASSESSMENT = "security_assessment" 
    FIELD_DETECTION = "field_detection"
    COMPLIANCE_CHECK = "compliance_check"
    ANOMALY_DETECTION = "anomaly_detection"
    GENERAL_QUESTION = "general_question"

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

class MessageRole(str, Enum):
    """Chat message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

# Base schemas
class BaseRequest(BaseModel):
    """Base request model with common fields."""
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Request timestamp")

class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = Field(True, description="Operation success status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")

# Authentication schemas
class LoginRequest(BaseModel):
    """User login request."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=6, description="Password")

class LoginResponse(BaseResponse):
    """User login response."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(1800, description="Token expiration time in seconds")
    user: Dict[str, Any] = Field(..., description="User information")

class RefreshTokenRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str = Field(..., description="Refresh token")

# Copilot schemas
class ChatMessage(BaseModel):
    """Individual chat message."""
    id: str = Field(..., description="Message ID")
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional message metadata")

class CopilotQuery(BaseModel):
    """Protocol Intelligence Copilot query request."""
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language query")
    query_type: Optional[QueryType] = Field(None, description="Query type for routing")
    user_id: str = Field(..., description="User ID")
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    packet_data: Optional[bytes] = Field(None, description="Optional packet data for analysis")
    enable_learning: bool = Field(True, description="Enable learning from this interaction")
    preferred_provider: Optional[LLMProvider] = Field(None, description="Preferred LLM provider")

    class Config:
        json_encoders = {
            bytes: lambda v: v.hex() if v else None
        }

class CopilotResponse(BaseResponse):
    """Protocol Intelligence Copilot response."""
    response: str = Field(..., description="Natural language response")
    query_type: QueryType = Field(..., description="Detected or specified query type")
    confidence: float = Field(..., ge=0, le=1, description="Response confidence score")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Information sources used")
    suggestions: List[str] = Field(default_factory=list, description="Follow-up suggestions")
    session_id: str = Field(..., description="Conversation session ID")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    llm_provider: Optional[str] = Field(None, description="LLM provider used")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional response metadata")

class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    session_id: Optional[str] = Field(None, description="Session ID")

# Protocol Analysis schemas
class ProtocolDiscoveryRequest(BaseRequest):
    """Enhanced protocol discovery request."""
    packet_data: bytes = Field(..., description="Raw packet data for analysis")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    enable_llm_analysis: bool = Field(False, description="Enable LLM-enhanced analysis")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    analysis_depth: Optional[str] = Field("standard", description="Analysis depth: basic, standard, deep")

    class Config:
        json_encoders = {
            bytes: lambda v: v.hex() if v else None
        }

    @validator('analysis_depth')
    def validate_analysis_depth(cls, v):
        if v not in ['basic', 'standard', 'deep']:
            raise ValueError('analysis_depth must be basic, standard, or deep')
        return v

class ProtocolDiscoveryResponse(BaseResponse):
    """Enhanced protocol discovery response."""
    protocol_type: str = Field(..., description="Detected protocol type")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    fields: List[Dict[str, Any]] = Field(default_factory=list, description="Detected protocol fields")
    analysis_summary: str = Field(..., description="Analysis summary")
    enhanced_analysis: Optional[Dict[str, Any]] = Field(None, description="LLM-enhanced analysis")
    anomalies: List[Dict[str, Any]] = Field(default_factory=list, description="Detected anomalies")
    security_implications: Optional[str] = Field(None, description="Security analysis")
    compliance_notes: Optional[str] = Field(None, description="Compliance considerations")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    llm_enabled: bool = Field(False, description="Whether LLM analysis was performed")

class FieldDetectionRequest(BaseRequest):
    """Enhanced field detection request."""
    message_data: bytes = Field(..., description="Protocol message data")
    protocol_type: Optional[str] = Field(None, description="Known protocol type")
    enable_llm_analysis: bool = Field(False, description="Enable LLM interpretation")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    field_types: Optional[List[str]] = Field(None, description="Specific field types to detect")

    class Config:
        json_encoders = {
            bytes: lambda v: v.hex() if v else None
        }

class FieldDetectionResponse(BaseResponse):
    """Enhanced field detection response."""
    fields: List[Dict[str, Any]] = Field(..., description="Detected fields")
    llm_interpretation: Optional[Dict[str, Any]] = Field(None, description="LLM field interpretation")
    field_relationships: Optional[List[Dict[str, Any]]] = Field(None, description="Field relationships")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    confidence_scores: Optional[Dict[str, float]] = Field(None, description="Per-field confidence scores")

# Knowledge Base schemas
class KnowledgeDocument(BaseModel):
    """Knowledge base document."""
    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    embeddings: Optional[List[float]] = Field(None, description="Document embeddings")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Update timestamp")

class SearchQuery(BaseModel):
    """Knowledge base search query."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    threshold: float = Field(0.7, ge=0, le=1, description="Similarity threshold")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")

class SearchResult(BaseModel):
    """Knowledge base search result."""
    document: KnowledgeDocument = Field(..., description="Matching document")
    similarity_score: float = Field(..., ge=0, le=1, description="Similarity score")
    relevance_explanation: Optional[str] = Field(None, description="Why this result is relevant")

class SearchResponse(BaseResponse):
    """Knowledge base search response."""
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., ge=0, description="Total number of matching results")
    query_time: float = Field(..., ge=0, description="Query execution time")

# System schemas
class HealthStatus(BaseModel):
    """System health status."""
    service: str = Field(..., description="Service name")
    status: str = Field(..., description="Health status")
    last_check: datetime = Field(default_factory=datetime.now, description="Last health check")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")

class SystemHealthResponse(BaseResponse):
    """System health check response."""
    overall_status: str = Field(..., description="Overall system health")
    services: List[HealthStatus] = Field(..., description="Individual service health")
    uptime: float = Field(..., ge=0, description="System uptime in seconds")
    version: str = Field(..., description="System version")

class MetricsResponse(BaseResponse):
    """System metrics response."""
    active_connections: int = Field(..., ge=0, description="Active connections count")
    total_requests: int = Field(..., ge=0, description="Total requests processed")
    copilot_queries: int = Field(..., ge=0, description="Total copilot queries")
    llm_requests: int = Field(..., ge=0, description="Total LLM requests")
    error_rate: float = Field(..., ge=0, le=1, description="Error rate percentage")
    avg_response_time: float = Field(..., ge=0, description="Average response time in seconds")

# Error schemas
class ErrorDetail(BaseModel):
    """Error detail information."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    field: Optional[str] = Field(None, description="Field that caused the error")

class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = Field(False, description="Operation success status")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[List[ErrorDetail]] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")

# Batch operation schemas
class BatchRequest(BaseModel):
    """Batch operation request."""
    operations: List[Dict[str, Any]] = Field(..., min_items=1, max_items=100, description="Operations to perform")
    parallel: bool = Field(False, description="Execute operations in parallel")
    stop_on_error: bool = Field(True, description="Stop execution on first error")

class BatchResponse(BaseResponse):
    """Batch operation response."""
    results: List[Dict[str, Any]] = Field(..., description="Operation results")
    completed: int = Field(..., ge=0, description="Number of completed operations")
    failed: int = Field(..., ge=0, description="Number of failed operations")
    execution_time: float = Field(..., ge=0, description="Total execution time")

# File upload schemas
class FileUploadResponse(BaseResponse):
    """File upload response."""
    file_id: str = Field(..., description="Uploaded file ID")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    content_type: str = Field(..., description="File content type")
    upload_time: datetime = Field(default_factory=datetime.now, description="Upload timestamp")

# Pagination schemas
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    per_page: int = Field(20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: Optional[str] = Field("asc", description="Sort order: asc or desc")

    @validator('sort_order')
    def validate_sort_order(cls, v):
        if v not in ['asc', 'desc']:
            raise ValueError('sort_order must be asc or desc')
        return v

class PaginatedResponse(BaseResponse):
    """Paginated response wrapper."""
    data: List[Any] = Field(..., description="Response data")
    pagination: Dict[str, Any] = Field(..., description="Pagination information")
    total_count: int = Field(..., ge=0, description="Total items available")