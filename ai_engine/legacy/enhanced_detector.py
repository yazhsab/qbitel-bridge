"""
QBITEL Engine - Enhanced Legacy System Anomaly Detector

Enhanced anomaly detection specifically designed for legacy systems,
combining traditional ML approaches with LLM-powered analysis and
historical pattern recognition.
"""

import logging
import time
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F

from ..anomaly.vae_detector import VAEAnomalyDetector, AnomalyResult
from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest
from ..core.config import Config
from ..core.exceptions import AnomalyDetectionException, LLMException
from ..monitoring.metrics import AIEngineMetrics

from .models import (
    SystemFailurePrediction,
    SystemBehaviorPattern,
    LegacySystemContext,
    HistoricalPatternDatabase,
    FailureType,
    SeverityLevel,
    SystemMetrics,
)


@dataclass
class LegacySystemLLMAnalyzer:
    """LLM-powered analysis component for legacy systems."""

    def __init__(self, llm_service: UnifiedLLMService):
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)

        # Analysis templates and prompts
        self.analysis_prompts = {
            "pattern_analysis": """
            You are analyzing a legacy system's behavioral patterns. Based on the current system data 
            and historical patterns provided, identify any anomalies, deviations, or concerning trends.
            
            System Context:
            - System Type: {system_type}
            - System Name: {system_name}  
            - Business Function: {business_function}
            - Criticality: {criticality}
            
            Current Metrics:
            {current_metrics}
            
            Historical Patterns:
            {historical_patterns}
            
            Expert Knowledge References:
            {expert_knowledge}
            
            Please analyze this data and provide:
            1. Pattern deviations detected
            2. Potential failure modes indicated  
            3. Risk assessment with severity level
            4. Recommended immediate actions
            5. Confidence level (0-100%)
            
            Focus on legacy system specific concerns like:
            - Resource exhaustion patterns
            - Memory leaks in long-running processes
            - Gradual performance degradation
            - Configuration drift indicators
            - Hardware aging symptoms
            """,
            "failure_prediction": """
            You are predicting potential failures in a legacy system based on current anomaly indicators.
            
            System Information:
            {system_context}
            
            Current Anomaly Indicators:
            - Anomaly Score: {anomaly_score}
            - Reconstruction Error: {reconstruction_error} 
            - Pattern Deviations: {pattern_deviations}
            
            Historical Precedents:
            {historical_precedents}
            
            Based on your expertise with legacy systems, predict:
            1. Most likely failure type and probability (0-100%)
            2. Estimated time to failure (hours/days/weeks)
            3. Primary contributing factors
            4. Business impact severity
            5. Preventive actions that could mitigate the risk
            6. Monitoring parameters to watch closely
            
            Consider legacy system characteristics:
            - Age-related degradation patterns
            - Cumulative stress indicators  
            - Resource constraint symptoms
            - Maintenance history implications
            """,
            "tribal_knowledge_synthesis": """
            You are synthesizing tribal knowledge about legacy system behavior with current observations.
            
            Current System Behavior:
            {current_behavior}
            
            Expert Knowledge Database:
            {expert_knowledge}
            
            Similar Historical Cases:
            {historical_cases}
            
            Synthesize this information to provide:
            1. Contextual interpretation of current behavior
            2. Lessons learned from similar situations
            3. Expert-validated troubleshooting steps
            4. Risk factors unique to this system type
            5. Maintenance recommendations based on experience
            
            Emphasize:
            - Patterns that experts have seen before
            - Solutions that have worked historically
            - Pitfalls to avoid based on past experience
            - System-specific quirks and behaviors
            """,
        }

    async def analyze_patterns(
        self,
        current_data: Dict[str, Any],
        historical_patterns: List[SystemBehaviorPattern],
        expert_knowledge: Dict[str, Any],
        system_context: Optional[LegacySystemContext] = None,
    ) -> Dict[str, Any]:
        """Analyze current system patterns using LLM intelligence."""

        try:
            # Prepare context information
            system_type = system_context.system_type.value if system_context else "unknown"
            system_name = system_context.system_name if system_context else "unknown"
            business_function = system_context.business_function if system_context else "unknown"
            criticality = system_context.criticality.value if system_context else "medium"

            # Format current metrics
            current_metrics = self._format_metrics_for_llm(current_data)

            # Format historical patterns
            historical_patterns_text = self._format_patterns_for_llm(historical_patterns)

            # Format expert knowledge
            expert_knowledge_text = self._format_expert_knowledge_for_llm(expert_knowledge)

            # Create LLM request
            prompt = self.analysis_prompts["pattern_analysis"].format(
                system_type=system_type,
                system_name=system_name,
                business_function=business_function,
                criticality=criticality,
                current_metrics=current_metrics,
                historical_patterns=historical_patterns_text,
                expert_knowledge=expert_knowledge_text,
            )

            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="legacy_whisperer",
                max_tokens=2000,
                temperature=0.1,  # Low temperature for consistent analysis
            )

            # Get LLM analysis
            response = await self.llm_service.process_request(llm_request)

            # Parse and structure the response
            analysis_result = self._parse_pattern_analysis_response(response.content)

            return {
                "llm_analysis": analysis_result,
                "confidence": response.confidence,
                "processing_time": response.processing_time,
                "model_used": response.provider,
            }

        except Exception as e:
            self.logger.error(f"LLM pattern analysis failed: {e}")
            return {
                "llm_analysis": {"error": str(e)},
                "confidence": 0.0,
                "processing_time": 0.0,
            }

    async def predict_failure_with_context(
        self,
        anomaly_score: float,
        reconstruction_error: float,
        pattern_deviations: List[str],
        system_context: LegacySystemContext,
        historical_precedents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Predict system failure using contextual LLM analysis."""

        try:
            # Format system context
            system_context_text = self._format_system_context_for_llm(system_context)

            # Format historical precedents
            precedents_text = self._format_precedents_for_llm(historical_precedents)

            # Create failure prediction prompt
            prompt = self.analysis_prompts["failure_prediction"].format(
                system_context=system_context_text,
                anomaly_score=anomaly_score,
                reconstruction_error=reconstruction_error,
                pattern_deviations=", ".join(pattern_deviations),
                historical_precedents=precedents_text,
            )

            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="legacy_whisperer",
                max_tokens=1500,
                temperature=0.1,
            )

            response = await self.llm_service.process_request(llm_request)

            # Parse failure prediction response
            prediction_result = self._parse_failure_prediction_response(response.content)

            return {
                "failure_prediction": prediction_result,
                "confidence": response.confidence,
                "processing_time": response.processing_time,
            }

        except Exception as e:
            self.logger.error(f"LLM failure prediction failed: {e}")
            return {"failure_prediction": {"error": str(e)}, "confidence": 0.0}

    def _format_metrics_for_llm(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for LLM consumption."""
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted.append(f"- {key}: {value:.2f}")
            else:
                formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)

    def _format_patterns_for_llm(self, patterns: List[SystemBehaviorPattern]) -> str:
        """Format historical patterns for LLM consumption."""
        if not patterns:
            return "No historical patterns available."

        formatted = []
        for pattern in patterns[:5]:  # Limit to top 5 patterns
            formatted.append(f"""
Pattern: {pattern.description}
Type: {pattern.pattern_type}
Frequency: {pattern.frequency}
Confidence: {pattern.confidence_score:.2f}
Last Seen: {pattern.last_seen}
Expert Notes: {pattern.expert_notes or 'None'}
""")
        return "\n".join(formatted)

    def _format_expert_knowledge_for_llm(self, knowledge: Dict[str, Any]) -> str:
        """Format expert knowledge for LLM consumption."""
        if not knowledge:
            return "No expert knowledge available."

        formatted = []
        for key, value in knowledge.items():
            if isinstance(value, list):
                formatted.append(f"- {key}: {', '.join(str(v) for v in value)}")
            else:
                formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)

    def _format_system_context_for_llm(self, context: LegacySystemContext) -> str:
        """Format system context for LLM consumption."""
        return f"""
System ID: {context.system_id}
System Name: {context.system_name}
Type: {context.system_type.value}
Manufacturer: {context.manufacturer or 'Unknown'}
Model: {context.model or 'Unknown'}
Version: {context.version or 'Unknown'}
Installation Date: {context.installation_date or 'Unknown'}
Criticality: {context.criticality.value}
Business Function: {context.business_function or 'Unknown'}
Uptime Requirement: {context.uptime_requirement}%
Users Count: {context.users_count or 'Unknown'}
CPU Cores: {context.cpu_cores or 'Unknown'}
Memory: {context.memory_gb or 'Unknown'} GB
Operating System: {context.operating_system or 'Unknown'}
"""

    def _format_precedents_for_llm(self, precedents: List[Dict[str, Any]]) -> str:
        """Format historical precedents for LLM consumption."""
        if not precedents:
            return "No historical precedents available."

        formatted = []
        for precedent in precedents[:3]:  # Limit to top 3 precedents
            formatted.append(f"""
Case: {precedent.get('description', 'Unknown case')}
Outcome: {precedent.get('outcome', 'Unknown')}
Resolution: {precedent.get('resolution', 'Unknown')}
Lessons Learned: {precedent.get('lessons_learned', 'None')}
""")
        return "\n".join(formatted)

    def _parse_pattern_analysis_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM pattern analysis response."""
        # This is a simplified parser - in production, you'd want more sophisticated parsing
        return {
            "raw_analysis": content,
            "pattern_deviations": self._extract_list_from_response(content, "pattern deviations"),
            "failure_modes": self._extract_list_from_response(content, "failure modes"),
            "risk_assessment": self._extract_risk_level(content),
            "recommended_actions": self._extract_list_from_response(content, "recommended actions"),
            "confidence_level": self._extract_confidence_level(content),
        }

    def _parse_failure_prediction_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM failure prediction response."""
        return {
            "raw_prediction": content,
            "failure_type": self._extract_failure_type(content),
            "probability": self._extract_probability(content),
            "time_to_failure": self._extract_time_to_failure(content),
            "contributing_factors": self._extract_list_from_response(content, "contributing factors"),
            "business_impact": self._extract_business_impact(content),
            "preventive_actions": self._extract_list_from_response(content, "preventive actions"),
            "monitoring_parameters": self._extract_list_from_response(content, "monitoring parameters"),
        }

    def _extract_list_from_response(self, content: str, section_name: str) -> List[str]:
        """Extract a list of items from LLM response."""
        # Simplified extraction - in production, use more robust parsing
        lines = content.lower().split("\n")
        items = []
        in_section = False

        for line in lines:
            if section_name.lower() in line and any(char in line for char in [":", "-", "1.", "•"]):
                in_section = True
                continue
            elif in_section and line.strip():
                if line.strip().startswith(("-", "•", "*")) or any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
                    items.append(line.strip().lstrip("-•*0123456789. "))
                elif not line.strip().startswith((" ", "\t")):
                    break

        return items[:5]  # Limit to 5 items

    def _extract_risk_level(self, content: str) -> str:
        """Extract risk level from response."""
        content_lower = content.lower()
        if "critical" in content_lower:
            return "critical"
        elif "high" in content_lower:
            return "high"
        elif "medium" in content_lower:
            return "medium"
        elif "low" in content_lower:
            return "low"
        return "unknown"

    def _extract_confidence_level(self, content: str) -> float:
        """Extract confidence level from response."""
        import re

        # Look for patterns like "confidence: 85%" or "85% confident"
        patterns = [
            r"confidence[:\s]+(\d+)%",
            r"(\d+)%\s+confident",
            r"confidence[:\s]+(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content.lower())
            if match:
                return float(match.group(1)) / 100.0

        return 0.5  # Default confidence

    def _extract_failure_type(self, content: str) -> str:
        """Extract failure type from response."""
        content_lower = content.lower()
        failure_types = [
            "hardware_failure",
            "software_corruption",
            "performance_degradation",
            "memory_leak",
            "disk_failure",
            "network_timeout",
            "configuration_drift",
            "security_breach",
            "data_corruption",
            "capacity_overrun",
        ]

        for failure_type in failure_types:
            if failure_type.replace("_", " ") in content_lower:
                return failure_type

        return "unknown"

    def _extract_probability(self, content: str) -> float:
        """Extract probability from response."""
        import re

        patterns = [
            r"probability[:\s]+(\d+)%",
            r"(\d+)%\s+probability",
            r"(\d+(?:\.\d+)?)%\s+chance",
        ]

        for pattern in patterns:
            match = re.search(pattern, content.lower())
            if match:
                return float(match.group(1)) / 100.0

        return 0.0

    def _extract_time_to_failure(self, content: str) -> Optional[str]:
        """Extract time to failure from response."""
        import re

        patterns = [
            r"(\d+)\s+hours?",
            r"(\d+)\s+days?",
            r"(\d+)\s+weeks?",
            r"within\s+(\d+)\s+hours?",
        ]

        for pattern in patterns:
            match = re.search(pattern, content.lower())
            if match:
                return match.group(0)

        return None

    def _extract_business_impact(self, content: str) -> str:
        """Extract business impact from response."""
        content_lower = content.lower()
        if any(word in content_lower for word in ["critical", "severe", "catastrophic"]):
            return "critical"
        elif any(word in content_lower for word in ["high", "significant", "major"]):
            return "high"
        elif any(word in content_lower for word in ["medium", "moderate"]):
            return "medium"
        elif any(word in content_lower for word in ["low", "minimal", "minor"]):
            return "low"
        return "medium"


class EnhancedLegacySystemDetector(VAEAnomalyDetector):
    """
    Enhanced anomaly detection for legacy systems.

    Combines traditional VAE-based anomaly detection with LLM-powered analysis,
    historical pattern recognition, and tribal knowledge integration.
    """

    def __init__(self, config: Config):
        """Initialize enhanced legacy system detector."""
        super().__init__(config)

        # Legacy system specific components
        self.llm_analyzer: Optional[LegacySystemLLMAnalyzer] = None
        self.historical_patterns = HistoricalPatternDatabase()
        self.expert_knowledge_base: Dict[str, Any] = {}

        # Enhanced thresholds for legacy systems
        self.legacy_anomaly_threshold = getattr(config, "legacy_anomaly_threshold", 0.85)
        self.pattern_deviation_threshold = getattr(config, "pattern_deviation_threshold", 0.7)

        # Metrics tracking
        self.metrics: Optional[AIEngineMetrics] = None

        # System contexts cache
        self.system_contexts: Dict[str, LegacySystemContext] = {}

        # Prediction cache
        self.prediction_cache: Dict[str, SystemFailurePrediction] = {}
        self.cache_ttl_hours = 1  # Cache predictions for 1 hour

        self.logger.info("EnhancedLegacySystemDetector initialized")

    async def initialize_enhanced(self, llm_service: UnifiedLLMService, metrics: Optional[AIEngineMetrics] = None) -> None:
        """Initialize enhanced components."""
        self.llm_analyzer = LegacySystemLLMAnalyzer(llm_service)
        self.metrics = metrics

        # Load expert knowledge base
        await self._load_expert_knowledge_base()

        self.logger.info("Enhanced legacy system detector components initialized")

    async def predict_system_failure(
        self,
        system_data: Dict[str, Any],
        system_context: Optional[LegacySystemContext] = None,
        prediction_horizon: int = 30,  # days
    ) -> SystemFailurePrediction:
        """
        Predict system failures using enhanced ML + LLM analysis.

        Args:
            system_data: Current system metrics and data
            system_context: System context information
            prediction_horizon: Prediction horizon in days

        Returns:
            Comprehensive system failure prediction
        """

        start_time = time.time()

        try:
            # Check cache first
            cache_key = f"{system_context.system_id if system_context else 'unknown'}_{hash(str(system_data))}"
            cached_prediction = self._get_cached_prediction(cache_key)
            if cached_prediction:
                return cached_prediction

            # Traditional anomaly detection
            features = self._extract_features_from_system_data(system_data)
            anomaly_result = await self.detect(features)

            # Get historical patterns for this system
            historical_patterns = []
            if system_context:
                historical_patterns = self.historical_patterns.get_patterns(system_context.system_id)

            # LLM-enhanced analysis
            llm_analysis = {}
            if self.llm_analyzer:
                llm_analysis = await self.llm_analyzer.analyze_patterns(
                    current_data=system_data,
                    historical_patterns=historical_patterns,
                    expert_knowledge=self.expert_knowledge_base,
                    system_context=system_context,
                )

            # Pattern deviation analysis
            pattern_deviations = self._analyze_pattern_deviations(system_data, historical_patterns)

            # Get historical precedents
            historical_precedents = self._get_historical_precedents(anomaly_result, pattern_deviations, system_context)

            # LLM failure prediction
            failure_prediction = {}
            if self.llm_analyzer:
                failure_prediction = await self.llm_analyzer.predict_failure_with_context(
                    anomaly_score=anomaly_result.anomaly_score,
                    reconstruction_error=anomaly_result.reconstruction_error,
                    pattern_deviations=pattern_deviations,
                    system_context=system_context
                    or LegacySystemContext(
                        system_id="unknown",
                        system_name="Unknown System",
                        system_type="unknown",
                    ),
                    historical_precedents=historical_precedents,
                )

            # Synthesize final prediction
            prediction = self._synthesize_failure_prediction(
                anomaly_result=anomaly_result,
                llm_analysis=llm_analysis,
                failure_prediction=failure_prediction,
                pattern_deviations=pattern_deviations,
                system_context=system_context,
                processing_time=time.time() - start_time,
            )

            # Cache the prediction
            self._cache_prediction(cache_key, prediction)

            # Record metrics
            if self.metrics:
                with self.metrics.track_anomaly_detection():
                    pass  # Metrics are tracked in the context manager

            return prediction

        except Exception as e:
            self.logger.error(f"Enhanced failure prediction failed: {e}")
            raise AnomalyDetectionException(f"Enhanced prediction error: {e}")

    def _extract_features_from_system_data(self, system_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from system data for VAE analysis."""
        # Extract numeric features from system data
        features = []

        # Standard system metrics
        features.append(system_data.get("cpu_utilization", 0.0))
        features.append(system_data.get("memory_utilization", 0.0))
        features.append(system_data.get("disk_utilization", 0.0))
        features.append(system_data.get("network_throughput", 0.0))
        features.append(system_data.get("response_time_ms", 0.0))
        features.append(system_data.get("transaction_rate", 0.0))
        features.append(system_data.get("error_rate", 0.0))

        # Pad or truncate to match model input dimension
        while len(features) < self.input_dim:
            features.append(0.0)

        features = features[: self.input_dim]

        return np.array(features, dtype=np.float32)

    def _analyze_pattern_deviations(
        self,
        current_data: Dict[str, Any],
        historical_patterns: List[SystemBehaviorPattern],
    ) -> List[str]:
        """Analyze deviations from historical patterns."""
        deviations = []

        for pattern in historical_patterns:
            if pattern.pattern_type == "normal":
                # Check for deviations from normal patterns
                deviation_score = self._calculate_pattern_deviation_score(current_data, pattern)

                if deviation_score > self.pattern_deviation_threshold:
                    deviations.append(f"Deviation from {pattern.description} " f"(score: {deviation_score:.2f})")

        return deviations

    def _calculate_pattern_deviation_score(self, current_data: Dict[str, Any], pattern: SystemBehaviorPattern) -> float:
        """Calculate deviation score from a specific pattern."""
        # Simplified calculation - in production, use more sophisticated methods
        score = 0.0
        count = 0

        # Compare CPU utilization pattern
        if pattern.cpu_utilization_pattern and current_data.get("cpu_utilization"):
            expected_cpu = np.mean(pattern.cpu_utilization_pattern)
            actual_cpu = current_data["cpu_utilization"]
            score += abs(expected_cpu - actual_cpu) / 100.0
            count += 1

        # Compare memory utilization pattern
        if pattern.memory_usage_pattern and current_data.get("memory_utilization"):
            expected_memory = np.mean(pattern.memory_usage_pattern)
            actual_memory = current_data["memory_utilization"]
            score += abs(expected_memory - actual_memory) / 100.0
            count += 1

        return score / max(count, 1)

    def _get_historical_precedents(
        self,
        anomaly_result: AnomalyResult,
        pattern_deviations: List[str],
        system_context: Optional[LegacySystemContext],
    ) -> List[Dict[str, Any]]:
        """Get historical precedents for similar situations."""
        # In production, this would query a database of historical incidents
        precedents = []

        if anomaly_result.anomaly_score > 0.8:
            precedents.append(
                {
                    "description": "High anomaly score detected in similar legacy system",
                    "outcome": "System failure within 48 hours",
                    "resolution": "Emergency maintenance performed",
                    "lessons_learned": "Early intervention prevented extended downtime",
                }
            )

        if len(pattern_deviations) > 2:
            precedents.append(
                {
                    "description": "Multiple pattern deviations observed",
                    "outcome": "Gradual performance degradation",
                    "resolution": "Scheduled maintenance and configuration tuning",
                    "lessons_learned": "Pattern monitoring is crucial for early detection",
                }
            )

        return precedents

    def _synthesize_failure_prediction(
        self,
        anomaly_result: AnomalyResult,
        llm_analysis: Dict[str, Any],
        failure_prediction: Dict[str, Any],
        pattern_deviations: List[str],
        system_context: Optional[LegacySystemContext],
        processing_time: float,
    ) -> SystemFailurePrediction:
        """Synthesize final failure prediction from all analysis components."""

        # Extract LLM insights
        llm_insights = llm_analysis.get("llm_analysis", {})
        failure_insights = failure_prediction.get("failure_prediction", {})

        # Determine failure type
        failure_type_str = failure_insights.get("failure_type", "performance_degradation")
        try:
            failure_type = FailureType(failure_type_str)
        except ValueError:
            failure_type = FailureType.PERFORMANCE_DEGRADATION

        # Determine severity based on anomaly score and LLM analysis
        if anomaly_result.anomaly_score > 0.9 or llm_insights.get("risk_assessment") == "critical":
            severity = SeverityLevel.CRITICAL
        elif anomaly_result.anomaly_score > 0.8 or llm_insights.get("risk_assessment") == "high":
            severity = SeverityLevel.HIGH
        elif anomaly_result.anomaly_score > 0.6 or llm_insights.get("risk_assessment") == "medium":
            severity = SeverityLevel.MEDIUM
        else:
            severity = SeverityLevel.LOW

        # Calculate probability (combine ML and LLM predictions)
        ml_probability = min(anomaly_result.anomaly_score, 1.0)
        llm_probability = failure_insights.get("probability", 0.0)
        combined_probability = (ml_probability + llm_probability) / 2.0

        # Estimate time to failure
        time_to_failure_str = failure_insights.get("time_to_failure")
        time_to_failure_hours = None
        if time_to_failure_str:
            # Parse time string (simplified)
            if "hour" in time_to_failure_str:
                try:
                    time_to_failure_hours = float(time_to_failure_str.split()[0])
                except:
                    pass
            elif "day" in time_to_failure_str:
                try:
                    time_to_failure_hours = float(time_to_failure_str.split()[0]) * 24
                except:
                    pass

        # Create prediction
        prediction = SystemFailurePrediction(
            prediction_id=f"pred_{int(time.time() * 1000000)}",
            system_id=system_context.system_id if system_context else "unknown",
            failure_type=failure_type,
            severity=severity,
            probability=combined_probability,
            time_to_failure_hours=time_to_failure_hours,
            primary_indicators=llm_insights.get("pattern_deviations", [])[:3],
            secondary_indicators=pattern_deviations[:3],
            anomaly_score=anomaly_result.anomaly_score,
            trend_analysis={
                "reconstruction_error": anomaly_result.reconstruction_error,
                "kl_divergence": anomaly_result.kl_divergence,
                "confidence": anomaly_result.confidence,
            },
            pattern_deviations=pattern_deviations,
            expert_analysis=llm_insights.get("raw_analysis", ""),
            immediate_actions=llm_insights.get("recommended_actions", []),
            monitoring_recommendations=failure_insights.get("monitoring_parameters", []),
            preventive_measures=failure_insights.get("preventive_actions", []),
            business_impact_score=self._calculate_business_impact_score(severity, system_context),
            metadata={
                "processing_time_seconds": processing_time,
                "ml_confidence": anomaly_result.confidence,
                "llm_confidence": llm_analysis.get("confidence", 0.0),
                "pattern_deviation_count": len(pattern_deviations),
            },
        )

        return prediction

    def _calculate_business_impact_score(
        self, severity: SeverityLevel, system_context: Optional[LegacySystemContext]
    ) -> float:
        """Calculate business impact score based on severity and context."""
        base_score = {
            SeverityLevel.CRITICAL: 9.0,
            SeverityLevel.HIGH: 7.0,
            SeverityLevel.MEDIUM: 5.0,
            SeverityLevel.LOW: 2.0,
            SeverityLevel.INFO: 1.0,
        }.get(severity, 5.0)

        # Adjust based on system context
        if system_context:
            # Adjust for system criticality
            if system_context.criticality == SeverityLevel.CRITICAL:
                base_score *= 1.5
            elif system_context.criticality == SeverityLevel.HIGH:
                base_score *= 1.2

            # Adjust for user count
            if system_context.users_count:
                if system_context.users_count > 10000:
                    base_score *= 1.3
                elif system_context.users_count > 1000:
                    base_score *= 1.1

            # Adjust for uptime requirement
            if system_context.uptime_requirement > 99.9:
                base_score *= 1.4
            elif system_context.uptime_requirement > 99.0:
                base_score *= 1.2

        return min(base_score, 10.0)  # Cap at 10.0

    def _get_cached_prediction(self, cache_key: str) -> Optional[SystemFailurePrediction]:
        """Get cached prediction if still valid."""
        if cache_key in self.prediction_cache:
            prediction = self.prediction_cache[cache_key]
            age_hours = (datetime.now() - prediction.prediction_timestamp).total_seconds() / 3600

            if age_hours < self.cache_ttl_hours:
                return prediction
            else:
                del self.prediction_cache[cache_key]

        return None

    def _cache_prediction(self, cache_key: str, prediction: SystemFailurePrediction) -> None:
        """Cache prediction result."""
        self.prediction_cache[cache_key] = prediction

        # Clean old cache entries
        current_time = datetime.now()
        keys_to_remove = []

        for key, cached_prediction in self.prediction_cache.items():
            age_hours = (current_time - cached_prediction.prediction_timestamp).total_seconds() / 3600
            if age_hours > self.cache_ttl_hours:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.prediction_cache[key]

    async def _load_expert_knowledge_base(self) -> None:
        """Load expert knowledge base from storage."""
        # In production, this would load from a database or file system
        self.expert_knowledge_base = {
            "mainframe_patterns": {
                "high_cpu_sustained": "Often indicates batch job issues or infinite loops",
                "memory_creep": "Common in long-running COBOL programs, check for memory leaks",
                "slow_disk_io": "May indicate aging hardware or fragmentation",
            },
            "common_failures": {
                "hardware_aging": "Look for increasing error rates and response time degradation",
                "configuration_drift": "Compare current config with baseline",
                "resource_exhaustion": "Monitor trends in resource utilization",
            },
            "troubleshooting_steps": {
                "performance_issues": [
                    "Check system resource utilization",
                    "Review recent configuration changes",
                    "Analyze error logs for patterns",
                    "Compare with baseline performance metrics",
                ],
                "stability_issues": [
                    "Check for memory leaks in long-running processes",
                    "Review system event logs",
                    "Verify hardware health indicators",
                    "Examine network connectivity",
                ],
            },
        }

        self.logger.info("Expert knowledge base loaded")

    def add_system_context(self, system_context: LegacySystemContext) -> None:
        """Add system context for enhanced analysis."""
        self.system_contexts[system_context.system_id] = system_context
        self.logger.info(f"Added system context for {system_context.system_id}")

    def add_historical_pattern(self, pattern: SystemBehaviorPattern) -> None:
        """Add historical pattern to the database."""
        self.historical_patterns.add_pattern(pattern)
        self.logger.info(f"Added historical pattern: {pattern.pattern_id}")

    def get_system_health_summary(self, system_id: str) -> Dict[str, Any]:
        """Get comprehensive health summary for a system."""
        context = self.system_contexts.get(system_id)
        patterns = self.historical_patterns.get_patterns(system_id)

        # Get recent predictions
        recent_predictions = [
            pred
            for pred in self.prediction_cache.values()
            if pred.system_id == system_id
            and (datetime.now() - pred.prediction_timestamp).total_seconds() < 3600 * 24  # Last 24 hours
        ]

        return {
            "system_id": system_id,
            "context": context,
            "pattern_count": len(patterns),
            "recent_predictions": len(recent_predictions),
            "highest_risk_prediction": (max(recent_predictions, key=lambda p: p.probability) if recent_predictions else None),
            "health_score": self._calculate_health_score(context, patterns, recent_predictions),
        }

    def _calculate_health_score(
        self,
        context: Optional[LegacySystemContext],
        patterns: List[SystemBehaviorPattern],
        predictions: List[SystemFailurePrediction],
    ) -> float:
        """Calculate overall system health score (0-100)."""
        base_score = 100.0

        # Deduct based on recent high-risk predictions
        for prediction in predictions:
            if prediction.severity == SeverityLevel.CRITICAL:
                base_score -= 30 * prediction.probability
            elif prediction.severity == SeverityLevel.HIGH:
                base_score -= 20 * prediction.probability
            elif prediction.severity == SeverityLevel.MEDIUM:
                base_score -= 10 * prediction.probability

        # Bonus for having good pattern coverage
        if len(patterns) > 5:
            base_score += 5

        # Deduct for system age (if known)
        if context and context.installation_date:
            age_years = (datetime.now() - context.installation_date).days / 365.25
            if age_years > 10:
                base_score -= min(age_years - 10, 20)  # Max 20 points deduction for age

        return max(min(base_score, 100.0), 0.0)
