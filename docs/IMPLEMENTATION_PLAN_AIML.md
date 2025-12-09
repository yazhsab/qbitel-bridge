# CRONOS AI - AI/ML Modernization Implementation Plan

## Executive Summary

This document provides a detailed implementation plan for modernizing the AI/ML stack in CRONOS AI, including adopting agentic AI frameworks, improving MLOps capabilities, and enhancing the LLM integration layer.

---

## Phase 1: LLM Integration Modernization (Week 1-2)

### 1.1 Replace Manual LLM Routing with LiteLLM

**Current State**: Manual provider switching in `unified_llm_service.py`
**Target State**: Unified LLM interface with automatic fallback

#### Installation

```bash
pip install litellm>=1.5.0
```

#### Implementation

```python
# ai_engine/llm/litellm_service.py (NEW FILE)
"""
LiteLLM-based Unified LLM Service for CRONOS AI

Benefits:
- Unified API for all providers
- Automatic fallback and retry
- Built-in cost tracking
- Token counting and rate limiting
- Streaming support
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List, AsyncIterator
from dataclasses import dataclass
from enum import Enum

import litellm
from litellm import acompletion, completion
from litellm.exceptions import (
    RateLimitError,
    APIConnectionError,
    ServiceUnavailableError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    VLLM = "vllm"
    LOCALAI = "localai"

@dataclass
class LLMConfig:
    """LLM configuration."""
    # Provider priority (first = primary)
    provider_priority: List[str] = None

    # Model mappings
    models: Dict[str, str] = None

    # Fallback settings
    enable_fallback: bool = True
    max_retries: int = 3
    timeout_seconds: int = 60

    # Cost tracking
    track_costs: bool = True
    max_cost_per_request: float = 1.0

    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600

    def __post_init__(self):
        if self.provider_priority is None:
            # On-premise first strategy
            self.provider_priority = [
                "ollama/llama3.1:70b",      # Local primary
                "ollama/mistral:7b",         # Local fallback
                "anthropic/claude-3-opus",   # Cloud fallback 1
                "openai/gpt-4-turbo",        # Cloud fallback 2
            ]

        if self.models is None:
            self.models = {
                "protocol_analysis": "ollama/llama3.1:70b",
                "threat_detection": "anthropic/claude-3-opus",
                "compliance_check": "anthropic/claude-3-sonnet",
                "code_generation": "openai/gpt-4-turbo",
                "general": "ollama/mistral:7b",
            }

class LiteLLMService:
    """
    Production-ready LLM service using LiteLLM.

    Features:
    - Automatic provider failover
    - Cost tracking and budget enforcement
    - Token counting
    - Response caching
    - Streaming support
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._setup_litellm()
        self._cost_tracker = CostTracker()

    def _setup_litellm(self):
        """Configure LiteLLM settings."""
        # Set API keys from environment
        litellm.api_key = os.getenv("OPENAI_API_KEY")
        litellm.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        # Ollama configuration
        litellm.ollama_api_base = os.getenv(
            "OLLAMA_API_BASE",
            "http://localhost:11434"
        )

        # Enable caching
        if self.config.enable_cache:
            litellm.cache = litellm.Cache(
                type="redis",
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                ttl=self.config.cache_ttl_seconds,
            )

        # Set callbacks for monitoring
        litellm.success_callback = [self._on_success]
        litellm.failure_callback = [self._on_failure]

        # Timeout settings
        litellm.request_timeout = self.config.timeout_seconds

        logger.info("LiteLLM service configured")

    def _on_success(self, kwargs, response, start_time, end_time):
        """Callback on successful LLM call."""
        model = kwargs.get("model", "unknown")
        tokens = response.get("usage", {})

        logger.debug(
            f"LLM success: {model}, "
            f"tokens={tokens.get('total_tokens', 0)}, "
            f"latency={end_time - start_time:.2f}s"
        )

        # Track cost
        if self.config.track_costs:
            cost = litellm.completion_cost(response)
            self._cost_tracker.add_cost(model, cost)

    def _on_failure(self, kwargs, exception, start_time, end_time):
        """Callback on failed LLM call."""
        model = kwargs.get("model", "unknown")
        logger.warning(
            f"LLM failure: {model}, "
            f"error={type(exception).__name__}: {exception}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    )
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        domain: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate completion with automatic fallback.

        Args:
            messages: Chat messages in OpenAI format
            model: Specific model to use (optional)
            domain: Domain hint for model selection
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Completion response
        """
        # Select model based on domain or use specified
        if model is None:
            model = self._select_model(domain)

        # Try primary model with fallback
        models_to_try = [model] + [
            m for m in self.config.provider_priority
            if m != model
        ] if self.config.enable_fallback else [model]

        last_error = None
        for try_model in models_to_try:
            try:
                response = await acompletion(
                    model=try_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )

                return {
                    "content": response.choices[0].message.content,
                    "model": try_model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    "finish_reason": response.choices[0].finish_reason,
                }

            except (RateLimitError, ServiceUnavailableError, APIConnectionError) as e:
                logger.warning(f"Model {try_model} failed: {e}, trying fallback...")
                last_error = e
                continue

            except Exception as e:
                logger.error(f"Unexpected error with {try_model}: {e}")
                last_error = e
                continue

        raise last_error or RuntimeError("All LLM providers failed")

    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        domain: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens.

        Args:
            messages: Chat messages
            model: Model to use
            domain: Domain hint
            **kwargs: Additional parameters

        Yields:
            Completion tokens as they arrive
        """
        if model is None:
            model = self._select_model(domain)

        response = await acompletion(
            model=model,
            messages=messages,
            stream=True,
            **kwargs
        )

        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def analyze_protocol(
        self,
        protocol_data: bytes,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze protocol data using best-fit model.

        Args:
            protocol_data: Binary protocol data
            context: Additional context

        Returns:
            Analysis results
        """
        # Build analysis prompt
        hex_sample = protocol_data[:512].hex()

        messages = [
            {
                "role": "system",
                "content": """You are a protocol analysis expert. Analyze the
                provided binary data and identify:
                1. Protocol type/family
                2. Message structure (header, payload, trailer)
                3. Field boundaries and types
                4. Any security concerns

                Respond in JSON format."""
            },
            {
                "role": "user",
                "content": f"Analyze this protocol data:\n\nHex: {hex_sample}\n\n"
                          f"Context: {context or 'Unknown protocol'}"
            }
        ]

        return await self.complete(
            messages=messages,
            domain="protocol_analysis",
            temperature=0.3,  # Low temperature for analysis
        )

    async def assess_threat(
        self,
        event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess security threat using threat detection model.

        Args:
            event_data: Security event data

        Returns:
            Threat assessment with severity and recommendations
        """
        messages = [
            {
                "role": "system",
                "content": """You are a security threat analyst for enterprise
                systems. Analyze the security event and provide:
                1. Threat classification (benign/suspicious/malicious)
                2. Severity score (1-10)
                3. MITRE ATT&CK mapping
                4. Recommended actions
                5. Confidence level

                Respond in JSON format."""
            },
            {
                "role": "user",
                "content": f"Assess this security event:\n\n{event_data}"
            }
        ]

        return await self.complete(
            messages=messages,
            domain="threat_detection",
            temperature=0.2,
        )

    def _select_model(self, domain: Optional[str] = None) -> str:
        """Select best model for domain."""
        if domain and domain in self.config.models:
            return self.config.models[domain]
        return self.config.models.get("general", self.config.provider_priority[0])

    def get_cost_summary(self) -> Dict[str, float]:
        """Get cost tracking summary."""
        return self._cost_tracker.get_summary()

class CostTracker:
    """Track LLM usage costs."""

    def __init__(self):
        self._costs: Dict[str, float] = {}
        self._total: float = 0.0

    def add_cost(self, model: str, cost: float):
        """Add cost for a model."""
        if model not in self._costs:
            self._costs[model] = 0.0
        self._costs[model] += cost
        self._total += cost

    def get_summary(self) -> Dict[str, float]:
        """Get cost summary."""
        return {
            "by_model": self._costs.copy(),
            "total": self._total,
        }

# Singleton
_llm_service: Optional[LiteLLMService] = None

def get_llm_service() -> LiteLLMService:
    """Get singleton LLM service."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LiteLLMService()
    return _llm_service
```

---

## Phase 2: Agentic AI Framework (Week 3-4)

### 2.1 LangGraph Integration

**Purpose**: Graph-based workflows for complex protocol analysis

#### Installation

```bash
pip install langgraph>=0.0.5 langchain>=0.1.0 langchain-community>=0.0.10
```

#### Implementation

```python
# ai_engine/agents/protocol_analysis_graph.py (NEW FILE)
"""
LangGraph-based Protocol Analysis Workflow

Implements a multi-step protocol analysis pipeline:
1. Classification - Identify protocol type
2. Field Extraction - Extract message fields
3. Threat Assessment - Analyze security implications
4. Compliance Check - Verify regulatory compliance
5. Report Generation - Create analysis report
"""

import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# State definition
class ProtocolAnalysisState(TypedDict):
    """State for protocol analysis workflow."""
    # Input
    protocol_data: bytes
    protocol_name: Optional[str]
    context: Optional[str]

    # Analysis results
    classification: Optional[Dict[str, Any]]
    fields: Optional[List[Dict[str, Any]]]
    threats: Optional[List[Dict[str, Any]]]
    compliance: Optional[Dict[str, Any]]

    # Workflow tracking
    messages: Annotated[List[BaseMessage], "append"]
    current_step: str
    errors: List[str]
    confidence: float

# Tools for the agents
@tool
def extract_protocol_fields(hex_data: str) -> Dict[str, Any]:
    """
    Extract fields from protocol data.

    Args:
        hex_data: Hexadecimal representation of protocol data

    Returns:
        Extracted field information
    """
    from ..detection.field_detector import get_field_detector

    detector = get_field_detector()
    binary_data = bytes.fromhex(hex_data)

    fields = detector.detect_fields(binary_data)

    return {
        "fields": [
            {
                "name": f.name,
                "offset": f.offset,
                "length": f.length,
                "type": f.field_type,
                "value": f.value,
            }
            for f in fields
        ],
        "field_count": len(fields),
    }

@tool
def classify_protocol(hex_data: str, sample_size: int = 256) -> Dict[str, Any]:
    """
    Classify protocol type from binary data.

    Args:
        hex_data: Hexadecimal protocol data
        sample_size: Number of bytes to analyze

    Returns:
        Classification result with confidence
    """
    from ..discovery.protocol_classifier import get_protocol_classifier

    classifier = get_protocol_classifier()
    binary_data = bytes.fromhex(hex_data)[:sample_size]

    result = classifier.classify(binary_data)

    return {
        "protocol_type": result.protocol_type,
        "protocol_family": result.protocol_family,
        "confidence": result.confidence,
        "features": result.features,
    }

@tool
def check_threat_indicators(protocol_type: str, fields: List[Dict]) -> Dict[str, Any]:
    """
    Check for threat indicators in protocol data.

    Args:
        protocol_type: Classified protocol type
        fields: Extracted protocol fields

    Returns:
        Threat analysis results
    """
    from ..threat_intelligence.threat_analyzer import get_threat_analyzer

    analyzer = get_threat_analyzer()

    threats = analyzer.analyze(
        protocol_type=protocol_type,
        fields=fields,
    )

    return {
        "threats": [
            {
                "name": t.name,
                "severity": t.severity,
                "mitre_attack": t.mitre_mapping,
                "description": t.description,
            }
            for t in threats
        ],
        "risk_score": analyzer.calculate_risk_score(threats),
    }

@tool
def check_compliance(
    protocol_type: str,
    fields: List[Dict],
    frameworks: List[str] = None
) -> Dict[str, Any]:
    """
    Check protocol compliance with regulatory frameworks.

    Args:
        protocol_type: Protocol type
        fields: Protocol fields
        frameworks: Compliance frameworks to check

    Returns:
        Compliance check results
    """
    from ..compliance.compliance_checker import get_compliance_checker

    if frameworks is None:
        frameworks = ["GDPR", "SOC2", "PCI-DSS"]

    checker = get_compliance_checker()

    results = {}
    for framework in frameworks:
        result = checker.check(
            framework=framework,
            protocol_type=protocol_type,
            fields=fields,
        )
        results[framework] = {
            "compliant": result.compliant,
            "issues": result.issues,
            "recommendations": result.recommendations,
        }

    return {
        "frameworks_checked": frameworks,
        "results": results,
        "overall_compliant": all(r["compliant"] for r in results.values()),
    }

# Node functions
async def classify_node(state: ProtocolAnalysisState) -> ProtocolAnalysisState:
    """Classify the protocol."""
    logger.info("Running protocol classification...")

    hex_data = state["protocol_data"].hex()

    try:
        result = classify_protocol.invoke({"hex_data": hex_data})

        return {
            **state,
            "classification": result,
            "current_step": "classification_complete",
            "messages": state["messages"] + [
                AIMessage(content=f"Classified as {result['protocol_type']} "
                         f"(confidence: {result['confidence']:.2f})")
            ],
        }
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return {
            **state,
            "errors": state["errors"] + [f"Classification error: {e}"],
            "current_step": "classification_failed",
        }

async def extract_fields_node(state: ProtocolAnalysisState) -> ProtocolAnalysisState:
    """Extract protocol fields."""
    logger.info("Extracting protocol fields...")

    hex_data = state["protocol_data"].hex()

    try:
        result = extract_protocol_fields.invoke({"hex_data": hex_data})

        return {
            **state,
            "fields": result["fields"],
            "current_step": "fields_extracted",
            "messages": state["messages"] + [
                AIMessage(content=f"Extracted {result['field_count']} fields")
            ],
        }
    except Exception as e:
        logger.error(f"Field extraction failed: {e}")
        return {
            **state,
            "errors": state["errors"] + [f"Field extraction error: {e}"],
            "current_step": "field_extraction_failed",
        }

async def threat_assessment_node(state: ProtocolAnalysisState) -> ProtocolAnalysisState:
    """Assess security threats."""
    logger.info("Assessing security threats...")

    if not state.get("classification") or not state.get("fields"):
        return {
            **state,
            "current_step": "threat_assessment_skipped",
            "messages": state["messages"] + [
                AIMessage(content="Skipping threat assessment - missing classification or fields")
            ],
        }

    try:
        result = check_threat_indicators.invoke({
            "protocol_type": state["classification"]["protocol_type"],
            "fields": state["fields"],
        })

        return {
            **state,
            "threats": result["threats"],
            "confidence": result["risk_score"],
            "current_step": "threats_assessed",
            "messages": state["messages"] + [
                AIMessage(content=f"Found {len(result['threats'])} potential threats "
                         f"(risk score: {result['risk_score']:.2f})")
            ],
        }
    except Exception as e:
        logger.error(f"Threat assessment failed: {e}")
        return {
            **state,
            "errors": state["errors"] + [f"Threat assessment error: {e}"],
            "current_step": "threat_assessment_failed",
        }

async def compliance_check_node(state: ProtocolAnalysisState) -> ProtocolAnalysisState:
    """Check regulatory compliance."""
    logger.info("Checking compliance...")

    if not state.get("classification") or not state.get("fields"):
        return {
            **state,
            "current_step": "compliance_check_skipped",
        }

    try:
        result = check_compliance.invoke({
            "protocol_type": state["classification"]["protocol_type"],
            "fields": state["fields"],
        })

        return {
            **state,
            "compliance": result,
            "current_step": "compliance_checked",
            "messages": state["messages"] + [
                AIMessage(content=f"Compliance check complete: "
                         f"{'PASS' if result['overall_compliant'] else 'ISSUES FOUND'}")
            ],
        }
    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        return {
            **state,
            "errors": state["errors"] + [f"Compliance check error: {e}"],
            "current_step": "compliance_check_failed",
        }

async def report_generation_node(state: ProtocolAnalysisState) -> ProtocolAnalysisState:
    """Generate final analysis report."""
    logger.info("Generating report...")

    report = {
        "protocol_name": state.get("protocol_name", "Unknown"),
        "classification": state.get("classification"),
        "fields_extracted": len(state.get("fields", [])),
        "fields": state.get("fields", []),
        "threats": state.get("threats", []),
        "compliance": state.get("compliance"),
        "errors": state.get("errors", []),
        "confidence": state.get("confidence", 0.0),
    }

    return {
        **state,
        "current_step": "complete",
        "messages": state["messages"] + [
            AIMessage(content=f"Analysis complete. Report generated.")
        ],
        "report": report,
    }

# Routing functions
def should_continue(state: ProtocolAnalysisState) -> str:
    """Determine next step in workflow."""
    current = state["current_step"]

    if "failed" in current:
        # Continue despite failures
        if current == "classification_failed":
            return "extract_fields"
        elif current == "field_extraction_failed":
            return "threat_assessment"
        elif current == "threat_assessment_failed":
            return "compliance_check"
        elif current == "compliance_check_failed":
            return "generate_report"
        return "generate_report"

    step_map = {
        "classification_complete": "extract_fields",
        "fields_extracted": "threat_assessment",
        "threats_assessed": "compliance_check",
        "threat_assessment_skipped": "compliance_check",
        "compliance_checked": "generate_report",
        "compliance_check_skipped": "generate_report",
        "complete": END,
    }

    return step_map.get(current, "generate_report")

def build_protocol_analysis_graph() -> StateGraph:
    """Build the protocol analysis workflow graph."""

    # Create graph
    workflow = StateGraph(ProtocolAnalysisState)

    # Add nodes
    workflow.add_node("classify", classify_node)
    workflow.add_node("extract_fields", extract_fields_node)
    workflow.add_node("threat_assessment", threat_assessment_node)
    workflow.add_node("compliance_check", compliance_check_node)
    workflow.add_node("generate_report", report_generation_node)

    # Set entry point
    workflow.set_entry_point("classify")

    # Add edges with routing
    workflow.add_conditional_edges(
        "classify",
        should_continue,
        {
            "extract_fields": "extract_fields",
            "generate_report": "generate_report",
        }
    )

    workflow.add_conditional_edges(
        "extract_fields",
        should_continue,
        {
            "threat_assessment": "threat_assessment",
            "generate_report": "generate_report",
        }
    )

    workflow.add_conditional_edges(
        "threat_assessment",
        should_continue,
        {
            "compliance_check": "compliance_check",
            "generate_report": "generate_report",
        }
    )

    workflow.add_conditional_edges(
        "compliance_check",
        should_continue,
        {
            "generate_report": "generate_report",
        }
    )

    workflow.add_edge("generate_report", END)

    return workflow.compile()

class ProtocolAnalyzer:
    """
    High-level protocol analyzer using LangGraph.
    """

    def __init__(self):
        self.graph = build_protocol_analysis_graph()

    async def analyze(
        self,
        protocol_data: bytes,
        protocol_name: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze protocol data through the full pipeline.

        Args:
            protocol_data: Binary protocol data
            protocol_name: Optional protocol name
            context: Optional analysis context

        Returns:
            Complete analysis report
        """
        initial_state: ProtocolAnalysisState = {
            "protocol_data": protocol_data,
            "protocol_name": protocol_name,
            "context": context,
            "classification": None,
            "fields": None,
            "threats": None,
            "compliance": None,
            "messages": [HumanMessage(content=f"Analyze protocol: {protocol_name or 'Unknown'}")],
            "current_step": "start",
            "errors": [],
            "confidence": 0.0,
        }

        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)

        return final_state.get("report", final_state)

# Singleton
_protocol_analyzer: Optional[ProtocolAnalyzer] = None

def get_protocol_analyzer() -> ProtocolAnalyzer:
    """Get singleton protocol analyzer."""
    global _protocol_analyzer
    if _protocol_analyzer is None:
        _protocol_analyzer = ProtocolAnalyzer()
    return _protocol_analyzer
```

---

## Phase 3: MLOps Infrastructure (Week 5-6)

### 3.1 MLflow Model Registry

```python
# ai_engine/mlops/model_registry.py (NEW FILE)
"""
MLflow Model Registry Integration for CRONOS AI

Features:
- Model versioning and lifecycle management
- A/B testing support
- Model lineage tracking
- Automated model promotion
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import torch

logger = logging.getLogger(__name__)

class ModelStage(Enum):
    """Model lifecycle stages."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"

@dataclass
class ModelVersion:
    """Model version information."""
    name: str
    version: int
    stage: ModelStage
    run_id: str
    created_at: datetime
    metrics: Dict[str, float]
    tags: Dict[str, str]

class CronosModelRegistry:
    """
    MLflow-based model registry for CRONOS AI.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
    ):
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5000"
        )
        self.registry_uri = registry_uri or self.tracking_uri

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)

        self.client = MlflowClient()

        logger.info(f"Model registry initialized: {self.tracking_uri}")

    def register_pytorch_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_example: Optional[torch.Tensor] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        experiment_name: str = "cronos-ai-models",
    ) -> ModelVersion:
        """
        Register a PyTorch model with the registry.

        Args:
            model: PyTorch model
            model_name: Name for the registered model
            input_example: Example input for signature inference
            metrics: Training/evaluation metrics
            tags: Model tags
            experiment_name: MLflow experiment name

        Returns:
            Registered model version info
        """
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            # Log metrics
            if metrics:
                for name, value in metrics.items():
                    mlflow.log_metric(name, value)

            # Log tags
            if tags:
                for name, value in tags.items():
                    mlflow.set_tag(name, value)

            # Infer signature
            signature = None
            if input_example is not None:
                model.eval()
                with torch.no_grad():
                    output = model(input_example)
                signature = infer_signature(
                    input_example.numpy(),
                    output.numpy()
                )

            # Log model
            mlflow.pytorch.log_model(
                model,
                "model",
                signature=signature,
                registered_model_name=model_name,
            )

            run_id = run.info.run_id

        # Get latest version
        versions = self.client.search_model_versions(f"name='{model_name}'")
        latest = max(versions, key=lambda v: int(v.version))

        return ModelVersion(
            name=model_name,
            version=int(latest.version),
            stage=ModelStage(latest.current_stage),
            run_id=run_id,
            created_at=datetime.now(),
            metrics=metrics or {},
            tags=tags or {},
        )

    def load_model(
        self,
        model_name: str,
        stage: ModelStage = ModelStage.PRODUCTION,
        version: Optional[int] = None,
    ) -> torch.nn.Module:
        """
        Load a model from the registry.

        Args:
            model_name: Registered model name
            stage: Model stage to load from
            version: Specific version (overrides stage)

        Returns:
            Loaded PyTorch model
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/{stage.value}"

        logger.info(f"Loading model: {model_uri}")

        return mlflow.pytorch.load_model(model_uri)

    def promote_model(
        self,
        model_name: str,
        version: int,
        to_stage: ModelStage,
        archive_existing: bool = True,
    ) -> bool:
        """
        Promote a model version to a new stage.

        Args:
            model_name: Model name
            version: Version to promote
            to_stage: Target stage
            archive_existing: Archive current production model

        Returns:
            Success status
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=str(version),
                stage=to_stage.value,
                archive_existing_versions=archive_existing,
            )

            logger.info(f"Promoted {model_name} v{version} to {to_stage.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False

    def get_model_versions(
        self,
        model_name: str,
        stage: Optional[ModelStage] = None,
    ) -> List[ModelVersion]:
        """Get all versions of a model."""
        filter_string = f"name='{model_name}'"
        if stage:
            filter_string += f" and current_stage='{stage.value}'"

        versions = self.client.search_model_versions(filter_string)

        return [
            ModelVersion(
                name=v.name,
                version=int(v.version),
                stage=ModelStage(v.current_stage),
                run_id=v.run_id,
                created_at=datetime.fromtimestamp(v.creation_timestamp / 1000),
                metrics={},
                tags=dict(v.tags) if v.tags else {},
            )
            for v in versions
        ]

    def compare_models(
        self,
        model_name: str,
        version_a: int,
        version_b: int,
    ) -> Dict[str, Any]:
        """Compare metrics between two model versions."""
        run_a = self.client.get_model_version(model_name, str(version_a))
        run_b = self.client.get_model_version(model_name, str(version_b))

        metrics_a = self.client.get_run(run_a.run_id).data.metrics
        metrics_b = self.client.get_run(run_b.run_id).data.metrics

        comparison = {}
        all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())

        for metric in all_metrics:
            val_a = metrics_a.get(metric)
            val_b = metrics_b.get(metric)

            comparison[metric] = {
                "version_a": val_a,
                "version_b": val_b,
                "diff": (val_b - val_a) if val_a and val_b else None,
                "better": "b" if val_b and val_a and val_b > val_a else "a" if val_a and val_b else None,
            }

        return comparison

# Singleton
_model_registry: Optional[CronosModelRegistry] = None

def get_model_registry() -> CronosModelRegistry:
    """Get singleton model registry."""
    global _model_registry
    if _model_registry is None:
        _model_registry = CronosModelRegistry()
    return _model_registry
```

### 3.2 Model Drift Detection with Evidently

```python
# ai_engine/mlops/drift_detector.py (NEW FILE)
"""
Model Drift Detection using Evidently AI

Features:
- Data drift detection
- Prediction drift monitoring
- Feature importance tracking
- Automated alerting
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
)
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestColumnDrift,
    TestShareOfDriftedColumns,
    TestNumberOfDriftedColumns,
)

logger = logging.getLogger(__name__)

@dataclass
class DriftReport:
    """Drift detection report."""
    timestamp: datetime
    dataset_drift_detected: bool
    drift_share: float
    drifted_columns: List[str]
    drift_scores: Dict[str, float]
    alerts: List[str]

class ProtocolDriftDetector:
    """
    Monitor drift in protocol analysis models.
    """

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        drift_threshold: float = 0.1,
        alert_callback: Optional[callable] = None,
    ):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.alert_callback = alert_callback

        # Column mapping for protocol data
        self.column_mapping = ColumnMapping(
            target="label",
            prediction="prediction",
            numerical_features=[
                "message_length",
                "entropy",
                "field_count",
                "byte_frequency_variance",
            ],
            categorical_features=[
                "protocol_type",
                "message_type",
            ],
        )

    def set_reference_data(self, data: pd.DataFrame):
        """Set reference (training) data for drift comparison."""
        self.reference_data = data
        logger.info(f"Reference data set: {len(data)} samples")

    def detect_drift(
        self,
        current_data: pd.DataFrame,
    ) -> DriftReport:
        """
        Detect drift between reference and current data.

        Args:
            current_data: Current production data

        Returns:
            Drift detection report
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")

        # Create drift report
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        # Extract results
        results = report.as_dict()

        dataset_drift = results["metrics"][0]["result"]["dataset_drift"]
        drift_share = results["metrics"][0]["result"]["share_of_drifted_columns"]

        # Get per-column drift
        drift_table = results["metrics"][1]["result"]["drift_by_columns"]
        drifted_columns = [
            col for col, info in drift_table.items()
            if info.get("drift_detected", False)
        ]
        drift_scores = {
            col: info.get("drift_score", 0)
            for col, info in drift_table.items()
        }

        # Generate alerts
        alerts = []
        if dataset_drift:
            alerts.append(f"CRITICAL: Dataset drift detected (share: {drift_share:.2%})")

        for col in drifted_columns:
            score = drift_scores.get(col, 0)
            if score > 0.5:
                alerts.append(f"HIGH: Column '{col}' drifted significantly (score: {score:.3f})")
            else:
                alerts.append(f"MEDIUM: Column '{col}' drifted (score: {score:.3f})")

        # Trigger alert callback
        if alerts and self.alert_callback:
            self.alert_callback(alerts)

        return DriftReport(
            timestamp=datetime.now(),
            dataset_drift_detected=dataset_drift,
            drift_share=drift_share,
            drifted_columns=drifted_columns,
            drift_scores=drift_scores,
            alerts=alerts,
        )

    def run_drift_tests(
        self,
        current_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Run drift test suite.

        Args:
            current_data: Current production data

        Returns:
            Test results
        """
        tests = TestSuite(tests=[
            TestShareOfDriftedColumns(lt=self.drift_threshold),
            TestNumberOfDriftedColumns(lt=3),
        ])

        # Add per-column tests for critical features
        critical_features = ["message_length", "entropy", "field_count"]
        for feature in critical_features:
            if feature in current_data.columns:
                tests.tests.append(
                    TestColumnDrift(column_name=feature)
                )

        tests.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        return tests.as_dict()

    def get_html_report(
        self,
        current_data: pd.DataFrame,
    ) -> str:
        """Generate HTML drift report."""
        report = Report(metrics=[
            DataDriftPreset(),
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        return report.get_html()

# Singleton
_drift_detector: Optional[ProtocolDriftDetector] = None

def get_drift_detector() -> ProtocolDriftDetector:
    """Get singleton drift detector."""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = ProtocolDriftDetector()
    return _drift_detector
```

---

## Phase 4: Vector Database Migration (Week 7-8)

### 4.1 Qdrant Migration

```python
# ai_engine/knowledge/qdrant_store.py (NEW FILE)
"""
Qdrant Vector Store for CRONOS AI

Migration from ChromaDB to Qdrant for production:
- Better performance at scale
- Built-in multi-tenancy
- Point-in-time recovery
- Hybrid search support
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result with score and metadata."""
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]

class QdrantKnowledgeStore:
    """
    Production-grade vector store using Qdrant.
    """

    # Collection names
    PROTOCOLS = "protocols"
    THREATS = "threats"
    COMPLIANCE = "compliance"
    DOCUMENTATION = "documentation"

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_dim: int = 384,
    ):
        import os

        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.embedding_dim = embedding_dim

        # Initialize client
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )

        # Initialize embedding model
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Ensure collections exist
        self._init_collections()

        logger.info(f"Qdrant knowledge store initialized: {self.url}")

    def _init_collections(self):
        """Initialize required collections."""
        collections = [
            self.PROTOCOLS,
            self.THREATS,
            self.COMPLIANCE,
            self.DOCUMENTATION,
        ]

        for name in collections:
            try:
                self.client.get_collection(name)
            except Exception:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created collection: {name}")

    def add_document(
        self,
        collection: str,
        content: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Add a document to the knowledge store.

        Args:
            collection: Collection name
            content: Document content
            metadata: Document metadata
            doc_id: Optional document ID

        Returns:
            Document ID
        """
        doc_id = doc_id or str(uuid4())

        # Generate embedding
        embedding = self.embedder.encode(content).tolist()

        # Create point
        point = PointStruct(
            id=doc_id,
            vector=embedding,
            payload={
                "content": content,
                **metadata,
            },
        )

        # Upsert to collection
        self.client.upsert(
            collection_name=collection,
            points=[point],
        )

        return doc_id

    def add_documents_batch(
        self,
        collection: str,
        documents: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Add multiple documents in batch.

        Args:
            collection: Collection name
            documents: List of {content, metadata} dicts

        Returns:
            List of document IDs
        """
        # Generate embeddings in batch
        contents = [doc["content"] for doc in documents]
        embeddings = self.embedder.encode(contents)

        # Create points
        doc_ids = []
        points = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = doc.get("id", str(uuid4()))
            doc_ids.append(doc_id)

            points.append(PointStruct(
                id=doc_id,
                vector=embedding.tolist(),
                payload={
                    "content": doc["content"],
                    **doc.get("metadata", {}),
                },
            ))

        # Batch upsert
        self.client.upsert(
            collection_name=collection,
            points=points,
        )

        return doc_ids

    def search(
        self,
        collection: str,
        query: str,
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.5,
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            collection: Collection to search
            query: Search query
            limit: Maximum results
            filter_conditions: Optional filters
            score_threshold: Minimum similarity score

        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()

        # Build filter
        search_filter = None
        if filter_conditions:
            conditions = [
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
                for key, value in filter_conditions.items()
            ]
            search_filter = Filter(must=conditions)

        # Search
        results = self.client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=limit,
            query_filter=search_filter,
            score_threshold=score_threshold,
        )

        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                content=r.payload.get("content", ""),
                metadata={k: v for k, v in r.payload.items() if k != "content"},
            )
            for r in results
        ]

    def hybrid_search(
        self,
        collection: str,
        query: str,
        limit: int = 10,
        keyword_weight: float = 0.3,
    ) -> List[SearchResult]:
        """
        Hybrid search combining semantic and keyword search.

        Args:
            collection: Collection to search
            query: Search query
            limit: Maximum results
            keyword_weight: Weight for keyword matching (0-1)

        Returns:
            Ranked search results
        """
        # Semantic search
        semantic_results = self.search(collection, query, limit=limit * 2)

        # Keyword matching (simple BM25-like scoring)
        query_terms = set(query.lower().split())

        scored_results = []
        for result in semantic_results:
            content_terms = set(result.content.lower().split())
            keyword_score = len(query_terms & content_terms) / len(query_terms) if query_terms else 0

            # Combine scores
            combined_score = (
                (1 - keyword_weight) * result.score +
                keyword_weight * keyword_score
            )

            scored_results.append((combined_score, result))

        # Sort by combined score
        scored_results.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(
                id=r.id,
                score=score,
                content=r.content,
                metadata=r.metadata,
            )
            for score, r in scored_results[:limit]
        ]

    def delete_document(
        self,
        collection: str,
        doc_id: str,
    ) -> bool:
        """Delete a document by ID."""
        try:
            self.client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(
                    points=[doc_id],
                ),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    def get_collection_stats(
        self,
        collection: str,
    ) -> Dict[str, Any]:
        """Get collection statistics."""
        info = self.client.get_collection(collection)

        return {
            "name": collection,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }

# Singleton
_knowledge_store: Optional[QdrantKnowledgeStore] = None

def get_knowledge_store() -> QdrantKnowledgeStore:
    """Get singleton knowledge store."""
    global _knowledge_store
    if _knowledge_store is None:
        _knowledge_store = QdrantKnowledgeStore()
    return _knowledge_store
```

---

## New Dependencies Summary

Add to `requirements.txt`:

```txt
# AI/ML Modernization Dependencies
litellm>=1.5.0
langgraph>=0.0.5
langchain>=0.1.0
langchain-community>=0.0.10
langchain-anthropic>=0.1.0
langchain-openai>=0.1.0

# MLOps
mlflow>=2.10.0
evidently>=0.4.0

# Vector Database
qdrant-client>=1.7.0

# Optional: Feature Store
feast>=0.35.0
```

---

## Migration Checklist

### Phase 1: LLM Integration
- [ ] Install LiteLLM
- [ ] Create `litellm_service.py`
- [ ] Update existing LLM calls to use new service
- [ ] Test with all providers (Ollama, Anthropic, OpenAI)
- [ ] Configure caching and cost tracking

### Phase 2: Agentic AI
- [ ] Install LangGraph
- [ ] Create protocol analysis graph
- [ ] Integrate with existing copilot
- [ ] Test multi-step workflows
- [ ] Add error handling and recovery

### Phase 3: MLOps
- [ ] Deploy MLflow server
- [ ] Migrate existing models to registry
- [ ] Configure Evidently drift detection
- [ ] Set up automated alerts
- [ ] Create model promotion workflow

### Phase 4: Vector Database
- [ ] Deploy Qdrant server
- [ ] Create migration script from ChromaDB
- [ ] Update all vector operations
- [ ] Test hybrid search
- [ ] Verify performance improvements

---

## Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| LLM Integration | 2 weeks | None |
| Agentic AI | 2 weeks | LLM Integration |
| MLOps | 2 weeks | None |
| Vector Database | 2 weeks | None |
| **Total** | **8 weeks** | |

Note: Phases 3-4 can run in parallel with Phase 2.
