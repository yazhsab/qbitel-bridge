# CRONOS AI - Implementation Guide for AI Coding Agents

**Version**: 1.0.0
**Purpose**: Detailed implementation reference for AI coding agents working on enhancements and new features

---

## Table of Contents

1. [How to Read This Guide](#1-how-to-read-this-guide)
2. [File Location Reference](#2-file-location-reference)
3. [Adding New Features](#3-adding-new-features)
4. [Extending Existing Components](#4-extending-existing-components)
5. [API Development Guide](#5-api-development-guide)
6. [Database Changes](#6-database-changes)
7. [ML Model Integration](#7-ml-model-integration)
8. [Security Considerations](#8-security-considerations)
9. [Testing Requirements](#9-testing-requirements)
10. [Common Patterns & Templates](#10-common-patterns--templates)

---

## 1. How to Read This Guide

This guide provides implementation details for AI coding agents. Each section includes:

- **Location markers**: Exact file paths for relevant code
- **Code patterns**: Templates to follow for consistency
- **Integration points**: Where new code connects to existing systems
- **Testing requirements**: Mandatory tests for each feature type

### Key Conventions

```
📁 = Directory
📄 = File
🔗 = Integration point
⚠️ = Important warning
✅ = Required step
```

---

## 2. File Location Reference

### 2.1 Core Module Locations

| Module | Base Path | Key Files |
|--------|-----------|-----------|
| API Layer | `ai_engine/api/` | `rest.py`, `server.py`, `auth.py` |
| Core Engine | `ai_engine/core/` | `engine.py`, `config.py`, `database_manager.py` |
| Protocol Discovery | `ai_engine/discovery/` | `protocol_discovery_orchestrator.py`, `pcfg_inference.py` |
| Anomaly Detection | `ai_engine/anomaly/` | `ensemble_detector.py`, `lstm_detector.py` |
| Security | `ai_engine/security/` | `decision_engine.py`, `security_service.py` |
| LLM | `ai_engine/llm/` | `unified_llm_service.py`, `rag_engine.py` |
| Copilot | `ai_engine/copilot/` | `protocol_copilot.py`, `context_manager.py` |
| Compliance | `ai_engine/compliance/` | `compliance_service.py`, `compliance_reporter.py` |
| Cloud Native | `ai_engine/cloud_native/` | See subdirectories |
| Monitoring | `ai_engine/monitoring/` | `metrics.py`, `opentelemetry_tracing.py` |
| Explainability | `ai_engine/explainability/` | `shap_explainer.py`, `lime_explainer.py` |
| Marketplace | `ai_engine/marketplace/` | `marketplace_service.py` |
| Tests | `ai_engine/tests/` | Organized by module |

### 2.2 Configuration Files

| Purpose | Path |
|---------|------|
| Default config | `config/cronos_ai.yaml` |
| Production config | `config/cronos_ai.production.yaml` |
| Compliance config | `config/compliance.yaml` |
| Security config | `config/security/enterprise-security-config.yaml` |
| Environment configs | `config/environments/` |

### 2.3 Infrastructure Files

| Purpose | Path |
|---------|------|
| Helm charts | `helm/cronos-ai/` |
| Kubernetes manifests | `kubernetes/` |
| Docker configs | `docker/` |
| CI/CD pipelines | `.github/workflows/` |
| Operations | `ops/` |

---

## 3. Adding New Features

### 3.1 Adding a New API Endpoint

**Step 1: Create endpoint file**

📄 Location: `ai_engine/api/<feature>_endpoints.py`

```python
"""
<Feature Name> API Endpoints

This module provides REST API endpoints for <feature description>.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ai_engine.api.auth import get_current_user
from ai_engine.api.schemas import BaseResponse
from ai_engine.core.config import Config

logger = logging.getLogger(__name__)

# Create router with prefix and tags
router = APIRouter(
    prefix="/api/v1/<feature>",
    tags=["<Feature Name>"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        500: {"description": "Internal Server Error"},
    },
)


# Request/Response schemas
class FeatureRequest(BaseModel):
    """Request schema for feature operation."""

    param1: str = Field(..., description="Description of param1")
    param2: Optional[int] = Field(default=None, description="Optional param2")

    class Config:
        json_schema_extra = {
            "example": {
                "param1": "example_value",
                "param2": 42
            }
        }


class FeatureResponse(BaseResponse):
    """Response schema for feature operation."""

    result: Dict[str, Any]
    processing_time: float


# Endpoints
@router.get("/", summary="Get feature info")
async def get_feature_info() -> Dict[str, Any]:
    """Get information about the feature."""
    return {
        "name": "<Feature Name>",
        "version": "1.0.0",
        "status": "active"
    }


@router.post("/process", summary="Process feature request")
async def process_feature(
    request: FeatureRequest,
    current_user: Dict = Depends(get_current_user),
) -> FeatureResponse:
    """Process a feature request.

    Args:
        request: The feature request payload
        current_user: The authenticated user

    Returns:
        FeatureResponse with processing results

    Raises:
        HTTPException: If processing fails
    """
    import time
    start_time = time.time()

    try:
        # Your implementation here
        result = {"status": "processed", "input": request.param1}

        processing_time = time.time() - start_time

        return FeatureResponse(
            success=True,
            result=result,
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(f"Feature processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/health", summary="Feature health check")
async def feature_health() -> Dict[str, str]:
    """Check feature health status."""
    return {"status": "healthy"}
```

**Step 2: Register router in rest.py**

📄 Location: `ai_engine/api/rest.py`

```python
# Add import
from ai_engine.api.feature_endpoints import router as feature_router

# Add router in create_app()
def create_app(config: Config) -> FastAPI:
    # ... existing code ...

    # Add your router
    app.include_router(feature_router)

    return app
```

**Step 3: Create tests**

📄 Location: `ai_engine/tests/api/test_feature_endpoints.py`

```python
"""Tests for feature endpoints."""
import pytest
from fastapi.testclient import TestClient


class TestFeatureEndpoints:
    """Test cases for feature API endpoints."""

    def test_get_feature_info(self, client: TestClient):
        """Test GET /api/v1/feature/"""
        response = client.get("/api/v1/feature/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_process_feature_success(self, client: TestClient, auth_headers):
        """Test POST /api/v1/feature/process"""
        response = client.post(
            "/api/v1/feature/process",
            json={"param1": "test_value"},
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result" in data

    def test_process_feature_unauthorized(self, client: TestClient):
        """Test unauthorized access."""
        response = client.post(
            "/api/v1/feature/process",
            json={"param1": "test_value"}
        )
        assert response.status_code == 401
```

### 3.2 Adding a New Service

**Step 1: Create service file**

📄 Location: `ai_engine/<module>/<feature>_service.py`

```python
"""
<Feature> Service

Provides business logic for <feature description>.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Metrics
FEATURE_REQUESTS = Counter(
    "cronos_feature_requests_total",
    "Total feature requests",
    ["operation", "status"]
)

FEATURE_DURATION = Histogram(
    "cronos_feature_duration_seconds",
    "Feature operation duration",
    ["operation"]
)


@dataclass
class FeatureConfig:
    """Configuration for feature service."""

    enabled: bool = True
    max_concurrent: int = 10
    timeout_seconds: float = 30.0
    retry_attempts: int = 3


@dataclass
class FeatureResult:
    """Result of a feature operation."""

    success: bool
    data: Dict[str, Any]
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    errors: List[str] = field(default_factory=list)


class FeatureService:
    """Service for managing feature operations."""

    def __init__(self, config: FeatureConfig):
        """Initialize feature service.

        Args:
            config: Feature configuration
        """
        self.config = config
        self._initialized = False
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        logger.info(f"FeatureService initialized with config: {config}")

    async def initialize(self) -> None:
        """Initialize service resources."""
        if self._initialized:
            return

        # Initialize connections, load models, etc.
        self._initialized = True
        logger.info("FeatureService fully initialized")

    async def shutdown(self) -> None:
        """Clean up service resources."""
        # Close connections, save state, etc.
        self._initialized = False
        logger.info("FeatureService shutdown complete")

    async def process(self, input_data: Dict[str, Any]) -> FeatureResult:
        """Process a feature request.

        Args:
            input_data: Input data for processing

        Returns:
            FeatureResult with processing outcome
        """
        import time
        start_time = time.time()

        async with self._semaphore:
            try:
                with FEATURE_DURATION.labels(operation="process").time():
                    # Your processing logic here
                    result_data = await self._do_processing(input_data)

                FEATURE_REQUESTS.labels(operation="process", status="success").inc()

                return FeatureResult(
                    success=True,
                    data=result_data,
                    processing_time=time.time() - start_time
                )

            except Exception as e:
                logger.error(f"Processing failed: {e}")
                FEATURE_REQUESTS.labels(operation="process", status="error").inc()

                return FeatureResult(
                    success=False,
                    data={},
                    processing_time=time.time() - start_time,
                    errors=[str(e)]
                )

    async def _do_processing(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Internal processing implementation.

        Args:
            input_data: Data to process

        Returns:
            Processed data
        """
        # Implement your business logic
        return {"processed": True, "input_keys": list(input_data.keys())}

    async def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Health status dictionary
        """
        return {
            "service": "FeatureService",
            "status": "healthy" if self._initialized else "not_initialized",
            "config": {
                "enabled": self.config.enabled,
                "max_concurrent": self.config.max_concurrent
            }
        }
```

### 3.3 Adding a New ML Model

**Step 1: Create model file**

📄 Location: `ai_engine/models/<model_name>.py`

```python
"""
<Model Name> Model

Implements <model description> for <use case>.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the model."""

    input_dim: int = 256
    hidden_dim: int = 128
    output_dim: int = 10
    dropout: float = 0.2
    learning_rate: float = 0.001


class CustomModel(nn.Module):
    """PyTorch model implementation."""

    def __init__(self, config: ModelConfig):
        """Initialize model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Define layers
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        encoded = self.encoder(x)
        output = self.decoder(encoded)
        return output


class ModelWrapper:
    """Wrapper for model loading, inference, and management."""

    def __init__(self, config: ModelConfig):
        """Initialize model wrapper.

        Args:
            config: Model configuration
        """
        self.config = config
        self.model: Optional[CustomModel] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_loaded = False

    def load(self, model_path: Optional[Path] = None) -> None:
        """Load model from disk or initialize new.

        Args:
            model_path: Path to saved model weights
        """
        self.model = CustomModel(self.config)

        if model_path and model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.info("Initialized new model")

        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True

    def save(self, model_path: Path) -> None:
        """Save model to disk.

        Args:
            model_path: Path to save model weights
        """
        if self.model is None:
            raise RuntimeError("No model to save")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

    @torch.no_grad()
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data.

        Args:
            input_data: Input array of shape (batch_size, input_dim)

        Returns:
            Predictions array of shape (batch_size, output_dim)
        """
        if not self._is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        # Convert to tensor
        x = torch.tensor(input_data, dtype=torch.float32).to(self.device)

        # Run inference
        output = self.model(x)

        return output.cpu().numpy()

    def get_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model info
        """
        return {
            "model_type": "CustomModel",
            "config": {
                "input_dim": self.config.input_dim,
                "hidden_dim": self.config.hidden_dim,
                "output_dim": self.config.output_dim,
            },
            "device": str(self.device),
            "is_loaded": self._is_loaded,
        }
```

---

## 4. Extending Existing Components

### 4.1 Adding New Anomaly Detection Method

📄 Location: `ai_engine/anomaly/`

**Step 1: Create detector file**

```python
# ai_engine/anomaly/new_detector.py
"""
New Detector Implementation

Implements <algorithm> for anomaly detection.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ai_engine.anomaly.base import BaseAnomalyDetector, AnomalyResult


@dataclass
class NewDetectorConfig:
    threshold: float = 0.5
    window_size: int = 100


class NewDetector(BaseAnomalyDetector):
    """New anomaly detection implementation."""

    def __init__(self, config: NewDetectorConfig):
        self.config = config
        self._is_trained = False

    def fit(self, data: np.ndarray) -> None:
        """Train detector on normal data."""
        # Training logic
        self._is_trained = True

    def predict(self, data: np.ndarray) -> AnomalyResult:
        """Detect anomalies in data."""
        if not self._is_trained:
            raise RuntimeError("Detector not trained")

        # Detection logic
        score = 0.5  # Calculate actual score

        return AnomalyResult(
            score=score,
            is_anomalous=score > self.config.threshold,
            confidence=0.9,
            explanation="Detection explanation"
        )
```

**Step 2: Integrate with ensemble**

📄 Modify: `ai_engine/anomaly/ensemble_detector.py`

```python
# Add import
from ai_engine.anomaly.new_detector import NewDetector, NewDetectorConfig

# Add to ensemble in __init__
self.detectors["new_detector"] = NewDetector(NewDetectorConfig())
```

### 4.2 Adding New LLM Provider

📄 Location: `ai_engine/llm/`

**Step 1: Create client file**

```python
# ai_engine/llm/new_provider_client.py
"""
New Provider LLM Client

Integration with <provider name> LLM API.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, Optional

from ai_engine.llm.base import BaseLLMClient, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class NewProviderClient(BaseLLMClient):
    """Client for New Provider LLM API."""

    def __init__(
        self,
        api_key: str,
        model: str = "default-model",
        endpoint: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint or "https://api.newprovider.com/v1"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send completion request to provider.

        Args:
            request: LLM request with prompt and parameters

        Returns:
            LLMResponse with generated text
        """
        # Implement API call
        pass

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream completion response.

        Args:
            request: LLM request

        Yields:
            Response chunks
        """
        # Implement streaming
        pass

    async def health_check(self) -> bool:
        """Check provider availability."""
        # Implement health check
        return True
```

**Step 2: Register in unified service**

📄 Modify: `ai_engine/llm/unified_llm_service.py`

```python
# Add import
from ai_engine.llm.new_provider_client import NewProviderClient

# Add provider enum
class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    NEW_PROVIDER = "new_provider"  # Add new provider

# Add initialization in _initialize_providers()
if config.new_provider_api_key:
    self.providers["new_provider"] = NewProviderClient(
        api_key=config.new_provider_api_key
    )
```

### 4.3 Adding New Compliance Framework

📄 Location: `ai_engine/compliance/`

**Step 1: Create framework file**

```python
# ai_engine/compliance/new_framework_compliance.py
"""
New Framework Compliance

Implements compliance checks for <Framework Name>.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ai_engine.compliance.base import (
    BaseComplianceFramework,
    ComplianceCheck,
    ComplianceResult,
    ComplianceStatus,
)


@dataclass
class NewFrameworkControl:
    """Control definition for new framework."""

    control_id: str
    title: str
    description: str
    category: str
    severity: str


class NewFrameworkCompliance(BaseComplianceFramework):
    """Compliance checker for New Framework."""

    FRAMEWORK_ID = "new_framework"
    FRAMEWORK_NAME = "New Framework Name"
    FRAMEWORK_VERSION = "1.0"

    CONTROLS = [
        NewFrameworkControl(
            control_id="NF-1.1",
            title="Access Control",
            description="Implement access control mechanisms",
            category="Security",
            severity="high"
        ),
        # Add more controls...
    ]

    async def assess(self, system_id: str) -> ComplianceResult:
        """Assess compliance with framework.

        Args:
            system_id: System to assess

        Returns:
            ComplianceResult with findings
        """
        checks = []

        for control in self.CONTROLS:
            check_result = await self._check_control(control, system_id)
            checks.append(check_result)

        passed = sum(1 for c in checks if c.status == ComplianceStatus.PASSED)
        total = len(checks)

        return ComplianceResult(
            framework=self.FRAMEWORK_ID,
            score=passed / total if total > 0 else 0,
            checks=checks,
            passed=passed,
            failed=total - passed,
        )

    async def _check_control(
        self,
        control: NewFrameworkControl,
        system_id: str
    ) -> ComplianceCheck:
        """Check individual control compliance."""
        # Implement control check logic
        pass
```

**Step 2: Register framework**

📄 Modify: `ai_engine/compliance/compliance_service.py`

```python
# Add import
from ai_engine.compliance.new_framework_compliance import NewFrameworkCompliance

# Add to frameworks dict
self.frameworks["new_framework"] = NewFrameworkCompliance()
```

---

## 5. API Development Guide

### 5.1 Standard Request Flow

```
Client Request
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Middleware Stack                                        │
│   1. GracefulShutdown (reject if shutting down)         │
│   2. RequestTracking (add correlation ID)               │
│   3. PayloadSizeLimit (reject if too large)             │
│   4. ContentTypeValidation (validate headers)           │
│   5. RequestLogging (log + metrics)                     │
│   6. SecurityHeaders (add security headers)             │
│   7. RateLimit (check limits)                           │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Authentication                                          │
│   - Verify JWT token OR API key                         │
│   - Extract user context                                │
│   - Check permissions                                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Input Validation                                        │
│   - Pydantic model validation                           │
│   - SQL injection check                                 │
│   - XSS prevention                                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Business Logic (Service Layer)                          │
│   - Process request                                     │
│   - Database operations                                 │
│   - External service calls                              │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Response                                                │
│   - Serialize with Pydantic                             │
│   - Add correlation ID header                           │
│   - Log response                                        │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Authentication Patterns

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

# Basic auth dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Get current authenticated user."""
    token = credentials.credentials
    user = await verify_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return user

# Role-based auth dependency
def require_role(required_role: str):
    """Dependency that requires specific role."""
    async def role_checker(
        current_user: Dict = Depends(get_current_user)
    ) -> Dict:
        if current_user.get("role") != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker

# Usage in endpoint
@router.post("/admin-only")
async def admin_endpoint(
    current_user: Dict = Depends(require_role("administrator"))
):
    pass
```

### 5.3 Error Response Format

```python
# Standard error response
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human readable message",
        "details": {
            "field": "Additional context"
        },
        "request_id": "correlation-id",
        "timestamp": "2024-11-22T15:30:45.123Z"
    }
}

# HTTP status codes
# 200 OK - Success
# 201 Created - Resource created
# 400 Bad Request - Client error
# 401 Unauthorized - Auth required
# 403 Forbidden - Access denied
# 404 Not Found - Resource not found
# 429 Too Many Requests - Rate limited
# 500 Internal Server Error - Server error
# 503 Service Unavailable - Temporarily down
```

---

## 6. Database Changes

### 6.1 Creating a New Migration

**Step 1: Generate migration**

```bash
cd ai_engine
alembic revision -m "add_new_table"
```

**Step 2: Edit migration file**

📄 Location: `ai_engine/alembic/versions/<timestamp>_add_new_table.py`

```python
"""add_new_table

Revision ID: abc123
Revises: previous_revision
Create Date: 2024-11-22 15:30:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers
revision = 'abc123'
down_revision = 'previous_revision'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply migration."""
    op.create_table(
        'new_table',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('data', JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now()),
    )

    # Create indexes
    op.create_index('idx_new_table_name', 'new_table', ['name'])


def downgrade() -> None:
    """Revert migration."""
    op.drop_index('idx_new_table_name')
    op.drop_table('new_table')
```

**Step 3: Run migration**

```bash
# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

### 6.2 Database Model Pattern

```python
# ai_engine/models/new_model.py
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from ai_engine.models.base import Base


class NewModel(Base):
    """New database model."""

    __tablename__ = "new_table"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Fields
    name = Column(String(255), nullable=False, index=True)
    description = Column(String(1000), nullable=True)
    data = Column(JSONB, nullable=True)

    # Foreign keys
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="new_items")

    # Indexes defined in migration or here
    __table_args__ = (
        # Composite index example
        # Index('idx_new_table_user_name', 'user_id', 'name'),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "data": self.data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
```

---

## 7. ML Model Integration

### 7.1 Model Registration

📄 Location: `ai_engine/core/model_registry.py`

```python
# Register new model
from ai_engine.models.new_model import NewModelWrapper

registry.register(
    name="new_model",
    model_class=NewModelWrapper,
    version="1.0.0",
    config={
        "input_dim": 256,
        "hidden_dim": 128,
    }
)

# Load model
model = registry.load("new_model", version="1.0.0")

# Run inference
result = await model.predict(input_data)
```

### 7.2 Model Training Pipeline

```python
# ai_engine/training/train_new_model.py
"""Training script for new model."""
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ai_engine.models.new_model import CustomModel, ModelConfig
from ai_engine.training.utils import (
    load_dataset,
    create_optimizer,
    EarlyStopping,
)

logger = logging.getLogger(__name__)


def train(
    data_path: Path,
    output_path: Path,
    config: ModelConfig,
    epochs: int = 100,
    batch_size: int = 32,
) -> None:
    """Train the model.

    Args:
        data_path: Path to training data
        output_path: Path to save model
        config: Model configuration
        epochs: Number of training epochs
        batch_size: Training batch size
    """
    # Load data
    train_data, val_data = load_dataset(data_path)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Initialize model
    model = CustomModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training setup
    optimizer = create_optimizer(model, lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=10)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if early_stopping(avg_val_loss):
            logger.info("Early stopping triggered")
            break

    # Save model
    torch.save(model.state_dict(), output_path)
    logger.info(f"Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    config = ModelConfig()
    train(args.data, args.output, config, args.epochs)
```

---

## 8. Security Considerations

### 8.1 Security Checklist for New Features

```
□ Input validation implemented (Pydantic models)
□ SQL injection prevention (parameterized queries)
□ XSS prevention (output encoding)
□ Authentication required for sensitive endpoints
□ Authorization checks for resource access
□ Rate limiting applied
□ Audit logging for sensitive operations
□ Secrets stored in environment/secrets manager (not code)
□ HTTPS enforced for production
□ Error messages don't expose internals
```

### 8.2 Secure Coding Patterns

```python
# DO: Use parameterized queries
result = await session.execute(
    select(User).where(User.id == user_id)
)

# DON'T: String interpolation in queries
# result = await session.execute(f"SELECT * FROM users WHERE id = {user_id}")

# DO: Validate input with Pydantic
class UserInput(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=100)

# DO: Use secrets manager for credentials
from ai_engine.core.secrets import get_secret
api_key = get_secret("API_KEY")

# DON'T: Hardcode secrets
# api_key = "sk-12345..."

# DO: Log without sensitive data
logger.info(f"User {user_id} logged in")

# DON'T: Log sensitive data
# logger.info(f"User logged in with password {password}")
```

### 8.3 Audit Logging

```python
from ai_engine.core.audit import audit_log

# Log security-sensitive operations
await audit_log(
    action="user.login",
    user_id=user.id,
    resource_type="session",
    resource_id=session_id,
    details={
        "ip_address": request.client.host,
        "user_agent": request.headers.get("user-agent"),
    },
    success=True
)
```

---

## 9. Testing Requirements

### 9.1 Test Coverage Requirements

| Component Type | Minimum Coverage |
|----------------|------------------|
| API Endpoints | 90% |
| Services | 85% |
| Models | 80% |
| Utilities | 80% |
| Overall | 80% |

### 9.2 Test Types Required

**For API Endpoints:**
- Success case tests
- Validation error tests
- Authentication tests
- Authorization tests
- Rate limiting tests
- Error handling tests

**For Services:**
- Unit tests for business logic
- Integration tests with dependencies
- Error handling tests
- Edge case tests

**For ML Models:**
- Model loading tests
- Inference tests
- Performance tests
- Input validation tests

### 9.3 Test File Template

```python
# ai_engine/tests/<module>/test_<feature>.py
"""
Tests for <Feature Name>

Tests cover:
- <Test category 1>
- <Test category 2>
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestFeatureUnit:
    """Unit tests for Feature."""

    def test_basic_functionality(self):
        """Test basic feature functionality."""
        # Arrange
        input_data = {"key": "value"}

        # Act
        result = process_feature(input_data)

        # Assert
        assert result is not None
        assert result["status"] == "success"

    def test_validation_error(self):
        """Test handling of invalid input."""
        with pytest.raises(ValidationError):
            process_feature({})

    def test_edge_case_empty_input(self):
        """Test behavior with empty input."""
        result = process_feature({"data": []})
        assert result["count"] == 0


class TestFeatureIntegration:
    """Integration tests for Feature."""

    @pytest.fixture
    def mock_database(self):
        """Mock database connection."""
        with patch("ai_engine.core.database_manager.DatabaseManager") as mock:
            yield mock

    async def test_full_workflow(self, mock_database):
        """Test complete feature workflow."""
        # Test implementation
        pass


class TestFeatureAPI:
    """API tests for Feature endpoints."""

    def test_endpoint_success(self, client, auth_headers):
        """Test successful API call."""
        response = client.post(
            "/api/v1/feature/process",
            json={"data": "test"},
            headers=auth_headers
        )
        assert response.status_code == 200

    def test_endpoint_unauthorized(self, client):
        """Test unauthorized access."""
        response = client.post("/api/v1/feature/process", json={})
        assert response.status_code == 401
```

---

## 10. Common Patterns & Templates

### 10.1 Async Service Pattern

```python
class AsyncService:
    """Template for async service implementation."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize service (called once)."""
        async with self._lock:
            if self._initialized:
                return
            # Initialization logic
            self._initialized = True

    async def shutdown(self) -> None:
        """Cleanup service resources."""
        async with self._lock:
            # Cleanup logic
            self._initialized = False

    async def process(self, data: Any) -> Any:
        """Process data with proper error handling."""
        if not self._initialized:
            await self.initialize()

        try:
            return await self._do_process(data)
        except Exception as e:
            logger.error(f"Processing error: {e}")
            raise ServiceError(str(e)) from e

    async def _do_process(self, data: Any) -> Any:
        """Internal processing implementation."""
        raise NotImplementedError
```

### 10.2 Repository Pattern

```python
class BaseRepository:
    """Base repository for database operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, model_class, id: UUID) -> Optional[Any]:
        """Get entity by ID."""
        result = await self.session.execute(
            select(model_class).where(model_class.id == id)
        )
        return result.scalar_one_or_none()

    async def get_all(
        self,
        model_class,
        limit: int = 100,
        offset: int = 0
    ) -> List[Any]:
        """Get all entities with pagination."""
        result = await self.session.execute(
            select(model_class).limit(limit).offset(offset)
        )
        return result.scalars().all()

    async def create(self, entity: Any) -> Any:
        """Create new entity."""
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def update(self, entity: Any) -> Any:
        """Update existing entity."""
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def delete(self, entity: Any) -> None:
        """Delete entity."""
        await self.session.delete(entity)
        await self.session.commit()
```

### 10.3 Circuit Breaker Pattern

```python
from dataclasses import dataclass
from enum import Enum
import time


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3


class CircuitBreaker:
    """Circuit breaker for external service calls."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
            else:
                raise CircuitOpenError("Circuit is open")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.config.half_open_max_calls:
                self.state = CircuitState.CLOSED

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
```

### 10.4 Event Publishing Pattern

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict
import json


@dataclass
class Event:
    """Base event class."""

    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    correlation_id: Optional[str] = None


class EventPublisher:
    """Publish events to message queue."""

    def __init__(self, kafka_producer):
        self.producer = kafka_producer

    async def publish(self, event: Event, topic: str) -> None:
        """Publish event to topic.

        Args:
            event: Event to publish
            topic: Kafka topic name
        """
        message = {
            "event_type": event.event_type,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data,
            "correlation_id": event.correlation_id,
        }

        await self.producer.send(
            topic=topic,
            value=json.dumps(message).encode("utf-8")
        )

# Usage
async def on_security_event(event_data: Dict):
    event = Event(
        event_type="security.threat_detected",
        timestamp=datetime.utcnow(),
        data=event_data,
    )
    await publisher.publish(event, "cronos.security.events")
```

---

## Appendix: Quick Reference

### File Creation Checklist

```
□ Create main implementation file
□ Add unit tests
□ Add integration tests (if applicable)
□ Update configuration if needed
□ Add database migration if needed
□ Update API documentation
□ Add metrics/monitoring
□ Add logging
□ Update KNOWLEDGE_BASE.md if significant change
```

### Import Order Convention

```python
# 1. Future imports
from __future__ import annotations

# 2. Standard library
import asyncio
import logging
from typing import Any, Dict, List

# 3. Third-party packages
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# 4. Local imports
from ai_engine.core.config import Config
from ai_engine.core.exceptions import CronosAIException
```

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>

# Types: feat, fix, docs, style, refactor, test, chore
# Example:
# feat(copilot): add streaming response support
#
# - Implement WebSocket streaming for real-time responses
# - Add session management for streaming connections
# - Update tests for streaming functionality
#
# Closes #123
```

---

**End of Implementation Guide**
