"""
CRONOS AI Engine - API Schemas

This module defines Pydantic models for API request and response schemas.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator
import base64


class DataFormat(str, Enum):
    """Supported data formats."""
    RAW_BYTES = "raw_bytes"
    HEX_STRING = "hex_string"
    BASE64 = "base64"
    TEXT = "text"


class ProtocolType(str, Enum):
    """Protocol types for discovery."""
    HTTP = "http"
    MODBUS = "modbus"
    HL7 = "hl7"
    ISO8583 = "iso8583"
    TN3270E = "tn3270e"
    CUSTOM = "custom"


class DetectionLevel(str, Enum):
    """Detection sensitivity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ModelType(str, Enum):
    """AI model types."""
    PROTOCOL_DISCOVERY = "protocol_discovery"
    FIELD_DETECTION = "field_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    ENSEMBLE = "ensemble"


# Base schemas

class BaseRequest(BaseModel):
    """Base request schema."""
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BaseResponse(BaseModel):
    """Base response schema."""
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = None
    success: bool = True
    message: Optional[str] = None


class ErrorResponse(BaseResponse):
    """Error response schema."""
    success: bool = False
    error_code: str
    error_message: str
    error_details: Optional[Dict[str, Any]] = None


# Protocol Discovery schemas

class ProtocolDiscoveryRequest(BaseRequest):
    """Protocol discovery request schema."""
    data: str = Field(..., description="Protocol data to analyze")
    data_format: DataFormat = Field(DataFormat.BASE64, description="Format of input data")
    expected_protocol: Optional[ProtocolType] = Field(None, description="Hint for expected protocol")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_samples: int = Field(1000, ge=1, le=10000, description="Maximum number of samples to analyze")
    include_grammar: bool = Field(False, description="Include inferred grammar in response")
    
    @validator('data')
    def validate_data(cls, v, values):
        """Validate data format."""
        data_format = values.get('data_format', DataFormat.BASE64)
        
        if data_format == DataFormat.BASE64:
            try:
                base64.b64decode(v)
            except Exception:
                raise ValueError("Invalid base64 data")
        elif data_format == DataFormat.HEX_STRING:
            try:
                bytes.fromhex(v.replace(' ', ''))
            except ValueError:
                raise ValueError("Invalid hex string")
        
        return v


class GrammarRule(BaseModel):
    """Grammar rule schema."""
    rule_name: str
    production: str
    probability: float = Field(ge=0.0, le=1.0)
    examples: List[str] = Field(default_factory=list)


class ProtocolDiscoveryResponse(BaseResponse):
    """Protocol discovery response schema."""
    discovered_protocol: Optional[ProtocolType] = None
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    protocol_characteristics: Dict[str, Any] = Field(default_factory=dict)
    field_structure: Optional[List[Dict[str, Any]]] = None
    grammar_rules: Optional[List[GrammarRule]] = None
    statistical_features: Dict[str, float] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)


# Field Detection schemas

class FieldDetectionRequest(BaseRequest):
    """Field detection request schema."""
    data: str = Field(..., description="Protocol data for field detection")
    data_format: DataFormat = Field(DataFormat.BASE64, description="Format of input data")
    protocol_hint: Optional[ProtocolType] = Field(None, description="Known protocol type")
    detection_level: DetectionLevel = Field(DetectionLevel.MEDIUM, description="Detection sensitivity")
    include_boundaries: bool = Field(True, description="Include field boundaries in response")
    include_semantics: bool = Field(False, description="Include semantic field analysis")
    max_fields: int = Field(100, ge=1, le=1000, description="Maximum fields to detect")


class DetectedField(BaseModel):
    """Detected field schema."""
    field_id: str
    field_name: Optional[str] = None
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    field_type: str  # header, data, checksum, etc.
    confidence: float = Field(ge=0.0, le=1.0)
    semantic_type: Optional[str] = None  # timestamp, identifier, payload, etc.
    encoding: Optional[str] = None  # ascii, binary, bcd, etc.
    constraints: Optional[Dict[str, Any]] = None
    examples: List[str] = Field(default_factory=list)
    
    @validator('end_offset')
    def validate_offsets(cls, v, values):
        """Validate field offsets."""
        start_offset = values.get('start_offset', 0)
        if v <= start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        return v


class FieldDetectionResponse(BaseResponse):
    """Field detection response schema."""
    detected_fields: List[DetectedField] = Field(default_factory=list)
    total_fields: int = 0
    protocol_structure: Optional[Dict[str, Any]] = None
    field_relationships: Optional[List[Dict[str, str]]] = None
    confidence_summary: Dict[str, float] = Field(default_factory=dict)


# Anomaly Detection schemas

class AnomalyDetectionRequest(BaseRequest):
    """Anomaly detection request schema."""
    data: str = Field(..., description="Protocol data for anomaly detection")
    data_format: DataFormat = Field(DataFormat.BASE64, description="Format of input data")
    baseline_data: Optional[List[str]] = Field(None, description="Historical baseline data")
    protocol_context: Optional[ProtocolType] = Field(None, description="Protocol context")
    sensitivity: DetectionLevel = Field(DetectionLevel.MEDIUM, description="Anomaly detection sensitivity")
    anomaly_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Anomaly score threshold")
    include_explanations: bool = Field(True, description="Include anomaly explanations")


class AnomalyScore(BaseModel):
    """Anomaly score details."""
    overall_score: float = Field(ge=0.0, le=1.0)
    reconstruction_error: Optional[float] = None
    statistical_deviation: Optional[float] = None
    sequence_anomaly: Optional[float] = None
    field_level_scores: Optional[Dict[str, float]] = None


class AnomalyExplanation(BaseModel):
    """Anomaly explanation schema."""
    anomaly_type: str  # structural, statistical, sequential, etc.
    description: str
    affected_regions: List[Dict[str, int]] = Field(default_factory=list)  # start, end offsets
    severity: str  # low, medium, high, critical
    confidence: float = Field(ge=0.0, le=1.0)
    recommendations: List[str] = Field(default_factory=list)


class AnomalyDetectionResponse(BaseResponse):
    """Anomaly detection response schema."""
    is_anomalous: bool = False
    anomaly_score: AnomalyScore
    anomaly_explanations: List[AnomalyExplanation] = Field(default_factory=list)
    baseline_comparison: Optional[Dict[str, Any]] = None
    trend_analysis: Optional[Dict[str, Any]] = None


# Model Management schemas

class ModelRegistrationRequest(BaseRequest):
    """Model registration request schema."""
    model_name: str = Field(..., min_length=1, max_length=100)
    model_version: str = Field(..., min_length=1, max_length=20)
    model_type: ModelType
    model_file: Optional[str] = Field(None, description="Base64 encoded model file")
    model_uri: Optional[str] = Field(None, description="URI to model file")
    description: Optional[str] = Field(None, max_length=1000)
    tags: Optional[Dict[str, str]] = Field(default_factory=dict)
    performance_metrics: Optional[Dict[str, float]] = Field(default_factory=dict)
    training_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('model_file')
    def validate_model_file(cls, v):
        """Validate base64 encoded model file."""
        if v is not None:
            try:
                base64.b64decode(v)
            except Exception:
                raise ValueError("Invalid base64 encoded model file")
        return v


class ModelInfo(BaseModel):
    """Model information schema."""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: str
    created_at: datetime
    updated_at: datetime
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)
    description: Optional[str] = None


class ModelRegistrationResponse(BaseResponse):
    """Model registration response schema."""
    model_info: Optional[ModelInfo] = None
    model_id: str
    registration_status: str


class ModelListRequest(BaseRequest):
    """Model list request schema."""
    model_type: Optional[ModelType] = None
    status: Optional[str] = None
    limit: int = Field(50, ge=1, le=1000)
    offset: int = Field(0, ge=0)


class ModelListResponse(BaseResponse):
    """Model list response schema."""
    models: List[ModelInfo] = Field(default_factory=list)
    total_count: int = 0
    has_more: bool = False


# Engine Status schemas

class ComponentStatus(BaseModel):
    """Component status schema."""
    name: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    details: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None


class EngineStatusResponse(BaseResponse):
    """Engine status response schema."""
    engine_version: str
    uptime_seconds: float
    components: List[ComponentStatus] = Field(default_factory=list)
    system_metrics: Dict[str, float] = Field(default_factory=dict)
    active_models: Dict[str, int] = Field(default_factory=dict)
    recent_activity: Optional[Dict[str, Any]] = None


# Batch Processing schemas

class BatchProcessingRequest(BaseRequest):
    """Batch processing request schema."""
    operation: str  # discovery, detection, anomaly
    data_items: List[str] = Field(..., min_items=1, max_items=1000)
    data_format: DataFormat = Field(DataFormat.BASE64)
    batch_id: Optional[str] = None
    processing_options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BatchItemResult(BaseModel):
    """Individual batch item result."""
    item_index: int
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None


class BatchProcessingResponse(BaseResponse):
    """Batch processing response schema."""
    batch_id: str
    total_items: int
    successful_items: int
    failed_items: int
    results: List[BatchItemResult] = Field(default_factory=list)
    batch_metrics: Dict[str, float] = Field(default_factory=dict)


# Training schemas

class TrainingRequest(BaseRequest):
    """Model training request schema."""
    model_type: ModelType
    model_name: str
    training_data: Optional[str] = Field(None, description="Base64 encoded training data")
    training_data_uri: Optional[str] = Field(None, description="URI to training data")
    validation_split: float = Field(0.2, ge=0.0, le=0.5)
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    training_config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class TrainingStatus(BaseModel):
    """Training status schema."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float = Field(ge=0.0, le=1.0)
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    metrics: Optional[Dict[str, float]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TrainingResponse(BaseResponse):
    """Training response schema."""
    job_id: str
    status: TrainingStatus
    estimated_completion: Optional[datetime] = None


# Health Check schemas

class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: str = "healthy"  # healthy, degraded, unhealthy
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    checks: Dict[str, bool] = Field(default_factory=dict)
    details: Optional[Dict[str, Any]] = None


# Prediction schemas

class PredictionRequest(BaseRequest):
    """Generic prediction request schema."""
    model_name: str
    model_version: Optional[str] = None
    input_data: str = Field(..., description="Input data for prediction")
    input_format: DataFormat = Field(DataFormat.BASE64)
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PredictionResponse(BaseResponse):
    """Generic prediction response schema."""
    predictions: Dict[str, Any]
    confidence: Optional[Dict[str, float]] = None
    model_info: Optional[Dict[str, str]] = None
    explanation: Optional[Dict[str, Any]] = None