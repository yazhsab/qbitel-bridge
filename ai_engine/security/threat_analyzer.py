"""
CRONOS AI Engine - Threat Analyzer

Advanced threat analysis with ML models and LLM-powered contextual insights.
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
from datetime import datetime, timedelta

from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest, get_llm_service
from ..core.config import Config
from ..core.exceptions import CronosAIException
from ..models.base import BaseModel, ModelInput, ModelOutput
from .models import (
    SecurityEvent, ThreatAnalysis, SecurityContext, LegacySystem,
    SecurityEventType, ThreatLevel, ConfidenceLevel, ThreatIntelligence,
    SecurityException, ThreatAnalysisException, validate_confidence_score
)

from prometheus_client import Counter, Histogram, Gauge

# Prometheus metrics
THREAT_ANALYSIS_COUNTER = Counter('cronos_threat_analysis_total', 'Threat analysis requests', ['event_type', 'threat_level'])
THREAT_ANALYSIS_DURATION = Histogram('cronos_threat_analysis_duration_seconds', 'Threat analysis duration')
THREAT_CONFIDENCE_GAUGE = Gauge('cronos_threat_confidence_latest', 'Latest threat confidence score')
ML_MODEL_PREDICTIONS = Counter('cronos_threat_ml_predictions_total', 'ML model predictions', ['model_type', 'prediction'])

logger = logging.getLogger(__name__)


class ThreatClassificationModel(BaseModel):
    """ML model for threat classification."""
    
    def __init__(self, config: Config):
        super().__init__(config, "threat_classifier")
        self.num_classes = len(SecurityEventType)
        self.feature_dim = 256
        
        # Simple neural network for demonstration
        self.features = self._build_feature_extractor()
        self.classifier = self._build_classifier()
        
    def _build_feature_extractor(self):
        """Build feature extraction network."""
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def _build_classifier(self):
        """Build classification head."""
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        """Forward pass."""
        features = self.features(x)
        return self.classifier(features)
    
    def predict(self, input_data: ModelInput) -> ModelOutput:
        """Predict threat classification."""
        start_time = time.time()
        
        try:
            import torch
            self.eval()
            tensor_input = self.preprocess_input(input_data)
            
            with torch.no_grad():
                predictions = self.forward(tensor_input)
                confidence = torch.max(predictions, dim=1)[0]
                predicted_class = torch.argmax(predictions, dim=1)
            
            processing_time = (time.time() - start_time) * 1000
            self.update_inference_metrics(processing_time, success=True)
            
            return ModelOutput(
                predictions=predictions,
                confidence=confidence,
                processing_time_ms=processing_time,
                metadata={
                    'predicted_class': predicted_class.item(),
                    'class_probabilities': predictions.cpu().numpy().tolist()[0]
                }
            )
            
        except Exception as e:
            self.logger.error(f"Threat classification prediction failed: {e}")
            self.update_inference_metrics(0, success=False)
            raise
    
    def validate_input(self, input_data: ModelInput) -> bool:
        """Validate input data."""
        try:
            tensor = input_data.to_tensor()
            return tensor.shape[-1] == self.feature_dim
        except Exception:
            return False
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema."""
        return {
            "type": "tensor",
            "shape": [-1, self.feature_dim],
            "dtype": "float32"
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get output schema."""
        return {
            "predictions": {
                "type": "tensor",
                "shape": [-1, self.num_classes],
                "dtype": "float32"
            }
        }


class FeatureExtractor:
    """Extract features from security events for ML analysis."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Feature extraction parameters
        self.max_sequence_length = 512
        self.vocab_size = 10000
        
    def extract_event_features(self, security_event: SecurityEvent) -> np.ndarray:
        """Extract features from security event."""
        
        features = []
        
        # Basic event features
        features.extend(self._extract_basic_features(security_event))
        
        # Network features 
        features.extend(self._extract_network_features(security_event))
        
        # Temporal features
        features.extend(self._extract_temporal_features(security_event))
        
        # Content features
        features.extend(self._extract_content_features(security_event))
        
        # Pad or truncate to fixed length
        feature_vector = np.array(features, dtype=np.float32)
        if len(feature_vector) < 256:
            feature_vector = np.pad(feature_vector, (0, 256 - len(feature_vector)))
        else:
            feature_vector = feature_vector[:256]
        
        return feature_vector
    
    def _extract_basic_features(self, event: SecurityEvent) -> List[float]:
        """Extract basic event features."""
        
        features = []
        
        # Event type one-hot encoding
        event_types = list(SecurityEventType)
        for event_type in event_types:
            features.append(1.0 if event.event_type == event_type else 0.0)
        
        # Threat level encoding
        threat_levels = list(ThreatLevel)
        for level in threat_levels:
            features.append(1.0 if event.threat_level == level else 0.0)
        
        # Confidence score
        features.append(event.confidence_score)
        
        # False positive likelihood
        features.append(event.false_positive_likelihood)
        
        return features
    
    def _extract_network_features(self, event: SecurityEvent) -> List[float]:
        """Extract network-related features."""
        
        features = []
        
        # IP address features
        if event.source_ip:
            ip_features = self._encode_ip_address(event.source_ip)
            features.extend(ip_features)
        else:
            features.extend([0.0] * 4)  # Placeholder for missing IP
        
        if event.destination_ip:
            ip_features = self._encode_ip_address(event.destination_ip)
            features.extend(ip_features)
        else:
            features.extend([0.0] * 4)
        
        # Network artifacts count
        features.append(float(len(event.network_artifacts)))
        
        # Protocol information
        features.append(float(len(event.affected_protocols)))
        
        return features
    
    def _extract_temporal_features(self, event: SecurityEvent) -> List[float]:
        """Extract temporal features."""
        
        features = []
        
        # Time of day (normalized)
        hour = event.event_timestamp.hour
        features.append(hour / 24.0)
        
        # Day of week
        day = event.event_timestamp.weekday()
        features.append(day / 7.0)
        
        # Time since detection
        if event.detection_timestamp:
            time_diff = (event.detection_timestamp - event.event_timestamp).total_seconds()
            features.append(min(time_diff / 3600.0, 24.0))  # Cap at 24 hours
        else:
            features.append(0.0)
        
        return features
    
    def _extract_content_features(self, event: SecurityEvent) -> List[float]:
        """Extract content-based features."""
        
        features = []
        
        # Text features from description
        if event.description:
            text_features = self._extract_text_features(event.description)
            features.extend(text_features)
        else:
            features.extend([0.0] * 10)
        
        # IOC count
        features.append(float(len(event.indicators_of_compromise)))
        
        # Attack vector count
        features.append(float(len(event.attack_vectors)))
        
        # Affected systems count
        features.append(float(len(event.affected_systems)))
        
        # File artifacts features
        features.append(float(len(event.file_artifacts)))
        
        return features
    
    def _encode_ip_address(self, ip_str: str) -> List[float]:
        """Encode IP address as features."""
        try:
            import ipaddress
            ip = ipaddress.ip_address(ip_str)
            
            if ip.version == 4:
                # IPv4: convert to 4 bytes
                return [(int(x) / 255.0) for x in str(ip).split('.')]
            else:
                # IPv6: simplified encoding
                return [1.0, 0.0, 0.0, 0.0]  # IPv6 indicator
        except:
            return [0.0, 0.0, 0.0, 0.0]  # Invalid IP
    
    def _extract_text_features(self, text: str) -> List[float]:
        """Extract features from text content."""
        
        # Simple text features
        features = []
        
        # Length features
        features.append(min(len(text) / 1000.0, 1.0))  # Normalized length
        features.append(min(len(text.split()) / 100.0, 1.0))  # Word count
        
        # Keyword presence (simplified)
        security_keywords = [
            'malware', 'virus', 'trojan', 'ransomware', 'exploit',
            'attack', 'breach', 'intrusion', 'suspicious'
        ]
        
        text_lower = text.lower()
        for keyword in security_keywords:
            features.append(1.0 if keyword in text_lower else 0.0)
        
        return features


class ThreatAnalyzer:
    """
    Advanced threat analyzer with ML models and LLM integration.
    
    This analyzer combines machine learning models with large language model
    insights to provide comprehensive threat analysis and contextual understanding.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core services
        self.llm_service: Optional[UnifiedLLMService] = None
        
        # ML models
        self.classification_model: Optional[ThreatClassificationModel] = None
        self.severity_model: Optional[Any] = None  # Simplified for demo
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor(config)
        
        # Threat intelligence
        self.threat_intelligence: Dict[str, ThreatIntelligence] = {}
        
        # Analysis cache
        self.analysis_cache: Dict[str, ThreatAnalysis] = {}
        
        # State management
        self._initialized = False
        
        self.logger.info("Threat Analyzer initialized")
    
    async def initialize(self) -> None:
        """Initialize the threat analyzer and its components."""
        if self._initialized:
            return
        
        try:
            self.logger.info("Initializing Threat Analyzer...")
            
            # Initialize LLM service
            self.llm_service = get_llm_service()
            if not hasattr(self.llm_service, '_initialized') or not self.llm_service._initialized:
                await self.llm_service.initialize()
            
            # Initialize ML models
            self.classification_model = ThreatClassificationModel(self.config)
            
            # Load threat intelligence
            await self._load_threat_intelligence()
            
            self._initialized = True
            self.logger.info("Threat Analyzer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Threat Analyzer: {e}")
            raise ThreatAnalysisException(f"Threat Analyzer initialization failed: {e}")
    
    async def analyze_threat(
        self,
        security_event: SecurityEvent,
        security_context: Optional[SecurityContext] = None,
        legacy_systems: Optional[List[LegacySystem]] = None
    ) -> ThreatAnalysis:
        """
        Perform comprehensive threat analysis.
        
        Args:
            security_event: Security event to analyze
            security_context: Current security context
            legacy_systems: Affected legacy systems
            
        Returns:
            Comprehensive threat analysis results
        """
        
        start_time = time.time()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            self.logger.info(f"Analyzing threat for event {security_event.event_id}")
            
            # Check cache first
            cache_key = self._create_cache_key(security_event)
            if cache_key in self.analysis_cache:
                cached_analysis = self.analysis_cache[cache_key]
                self.logger.info(f"Using cached analysis for event {security_event.event_id}")
                return cached_analysis
            
            # Step 1: Extract features for ML analysis
            features = self.feature_extractor.extract_event_features(security_event)
            
            # Step 2: ML-based classification
            ml_classification = await self._perform_ml_classification(features, security_event)
            
            # Step 3: Severity assessment
            severity_assessment = await self._assess_threat_severity(
                security_event, ml_classification, security_context
            )
            
            # Step 4: Context analysis with legacy systems
            context_analysis = await self._analyze_context(
                security_event, security_context, legacy_systems
            )
            
            # Step 5: Threat intelligence correlation
            intelligence_correlation = await self._correlate_threat_intelligence(security_event)
            
            # Step 6: Business impact assessment
            business_impact = await self._assess_business_impact(
                security_event, severity_assessment, legacy_systems
            )
            
            # Step 7: Create comprehensive analysis
            threat_analysis = self._create_threat_analysis(
                security_event,
                ml_classification,
                severity_assessment,
                context_analysis,
                intelligence_correlation,
                business_impact
            )
            
            # Update metrics
            processing_time = time.time() - start_time
            THREAT_ANALYSIS_DURATION.observe(processing_time)
            THREAT_ANALYSIS_COUNTER.labels(
                event_type=security_event.event_type.value,
                threat_level=threat_analysis.threat_level.value
            ).inc()
            THREAT_CONFIDENCE_GAUGE.set(threat_analysis.confidence_score)
            
            # Cache the analysis
            self.analysis_cache[cache_key] = threat_analysis
            
            self.logger.info(
                f"Threat analysis completed for {security_event.event_id} in {processing_time:.2f}s: "
                f"classification={threat_analysis.threat_classification.value}, "
                f"level={threat_analysis.threat_level.value}, "
                f"confidence={threat_analysis.confidence_score:.2f}"
            )
            
            return threat_analysis
            
        except Exception as e:
            self.logger.error(f"Threat analysis failed for event {security_event.event_id}: {e}")
            raise ThreatAnalysisException(f"Threat analysis failed: {e}")
    
    async def _perform_ml_classification(
        self, 
        features: np.ndarray, 
        security_event: SecurityEvent
    ) -> Dict[str, Any]:
        """Perform ML-based threat classification."""
        
        try:
            if self.classification_model:
                model_input = ModelInput(data=features)
                model_output = self.classification_model.predict(model_input)
                
                predicted_class_idx = model_output.metadata.get('predicted_class', 0)
                class_probabilities = model_output.metadata.get('class_probabilities', [])
                
                # Map prediction to threat type
                event_types = list(SecurityEventType)
                predicted_type = event_types[predicted_class_idx] if predicted_class_idx < len(event_types) else SecurityEventType.ANOMALOUS_BEHAVIOR
                
                ML_MODEL_PREDICTIONS.labels(
                    model_type='classification',
                    prediction=predicted_type.value
                ).inc()
                
                return {
                    'predicted_type': predicted_type,
                    'confidence': float(model_output.confidence.max().item()) if hasattr(model_output.confidence, 'max') else 0.5,
                    'class_probabilities': class_probabilities,
                    'ml_confidence': 0.8  # Base ML confidence
                }
            else:
                # Fallback rule-based classification
                return self._rule_based_classification(security_event)
                
        except Exception as e:
            self.logger.warning(f"ML classification failed: {e}")
            return self._rule_based_classification(security_event)
    
    def _rule_based_classification(self, security_event: SecurityEvent) -> Dict[str, Any]:
        """Fallback rule-based classification."""
        
        # Simple rule-based logic
        predicted_type = security_event.event_type
        confidence = 0.6  # Lower confidence for rule-based
        
        # Adjust confidence based on indicators
        if security_event.indicators_of_compromise:
            confidence += 0.2
        if security_event.attack_vectors:
            confidence += 0.1
        
        confidence = min(confidence, 1.0)
        
        return {
            'predicted_type': predicted_type,
            'confidence': confidence,
            'class_probabilities': [],
            'ml_confidence': confidence
        }
    
    async def _assess_threat_severity(
        self,
        security_event: SecurityEvent,
        ml_classification: Dict[str, Any],
        security_context: Optional[SecurityContext]
    ) -> Dict[str, Any]:
        """Assess threat severity using multiple factors."""
        
        base_severity = security_event.threat_level
        severity_score = self._threat_level_to_score(base_severity)
        
        # Adjust severity based on ML prediction confidence
        ml_confidence = ml_classification.get('confidence', 0.5)
        if ml_confidence > 0.8:
            severity_score *= 1.2
        elif ml_confidence < 0.4:
            severity_score *= 0.8
        
        # Adjust based on context
        if security_context:
            context_multiplier = self._calculate_context_severity_multiplier(security_context)
            severity_score *= context_multiplier
        
        # Adjust based on indicators
        if len(security_event.indicators_of_compromise) > 5:
            severity_score *= 1.3
        if len(security_event.attack_vectors) > 3:
            severity_score *= 1.2
        
        # Cap the score
        severity_score = min(severity_score, 1.0)
        
        # Convert back to threat level
        adjusted_threat_level = self._score_to_threat_level(severity_score)
        
        return {
            'threat_level': adjusted_threat_level,
            'severity_score': severity_score,
            'base_severity': base_severity,
            'adjustments': {
                'ml_confidence_factor': ml_confidence,
                'context_factor': context_multiplier if security_context else 1.0,
                'ioc_factor': len(security_event.indicators_of_compromise),
                'attack_vector_factor': len(security_event.attack_vectors)
            }
        }
    
    async def _analyze_context(
        self,
        security_event: SecurityEvent,
        security_context: Optional[SecurityContext],
        legacy_systems: Optional[List[LegacySystem]]
    ) -> Dict[str, Any]:
        """Analyze contextual factors using LLM."""
        
        context_data = {
            'event_summary': {
                'type': security_event.event_type.value,
                'description': security_event.description,
                'affected_systems': security_event.affected_systems
            },
            'security_context': {},
            'legacy_systems': []
        }
        
        if security_context:
            context_data['security_context'] = {
                'current_threat_level': security_context.current_threat_level.value,
                'active_incidents': len(security_context.active_incidents),
                'business_hours': security_context.business_hours
            }
        
        if legacy_systems:
            context_data['legacy_systems'] = [
                {
                    'name': sys.system_name,
                    'type': sys.system_type,
                    'criticality': sys.criticality.value,
                    'protocols': [p.value for p in sys.protocol_types]
                }
                for sys in legacy_systems
            ]
        
        # Use LLM for contextual analysis
        prompt = f"""Analyze the contextual factors for this security event:

Event: {security_event.event_type.value}
Description: {security_event.description}
Affected Systems: {', '.join(security_event.affected_systems)}

Context Information: {json.dumps(context_data, indent=2)}

Provide contextual analysis focusing on:
1. Environmental factors that increase/decrease risk
2. Business impact considerations
3. Legacy system implications
4. Timing and operational context
5. Historical pattern correlations

Provide concise, actionable insights."""
        
        try:
            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="security_orchestrator",
                context=context_data,
                max_tokens=1000,
                temperature=0.2
            )
            
            response = await self.llm_service.process_request(llm_request)
            
            return {
                'llm_analysis': response.content,
                'context_score': self._calculate_context_risk_score(context_data),
                'key_factors': self._extract_key_context_factors(response.content),
                'business_impact_indicators': self._extract_business_impact_indicators(context_data)
            }
            
        except Exception as e:
            self.logger.warning(f"LLM context analysis failed: {e}")
            return {
                'llm_analysis': f'Context analysis failed: {e}',
                'context_score': 0.5,
                'key_factors': [],
                'business_impact_indicators': []
            }
    
    async def _correlate_threat_intelligence(self, security_event: SecurityEvent) -> Dict[str, Any]:
        """Correlate event with threat intelligence data."""
        
        correlations = []
        relevance_score = 0.0
        
        # Check IOCs against threat intelligence
        for ioc in security_event.indicators_of_compromise:
            for intel_id, intel in self.threat_intelligence.items():
                for intel_ioc in intel.iocs:
                    if self._match_ioc(ioc, intel_ioc):
                        correlations.append({
                            'intelligence_id': intel_id,
                            'matched_ioc': ioc,
                            'threat_actor': intel.threat_actors,
                            'campaigns': intel.campaigns,
                            'confidence': intel.confidence
                        })
                        relevance_score = max(relevance_score, intel.confidence)
        
        # Check event type patterns
        event_type_matches = [
            intel for intel in self.threat_intelligence.values()
            if intel.threat_type == security_event.event_type
        ]
        
        return {
            'correlations': correlations,
            'relevance_score': relevance_score,
            'matched_intelligence_count': len(correlations),
            'event_type_matches': len(event_type_matches),
            'threat_actors': list(set(
                actor for correlation in correlations 
                for actor in correlation['threat_actor']
            )),
            'associated_campaigns': list(set(
                campaign for correlation in correlations
                for campaign in correlation['campaigns']
            ))
        }
    
    async def _assess_business_impact(
        self,
        security_event: SecurityEvent,
        severity_assessment: Dict[str, Any],
        legacy_systems: Optional[List[LegacySystem]]
    ) -> Dict[str, Any]:
        """Assess potential business impact."""
        
        impact_score = 0.0
        impact_factors = []
        
        # Base impact from severity
        severity_score = severity_assessment.get('severity_score', 0.5)
        impact_score += severity_score * 0.4
        
        # Legacy system impact
        if legacy_systems:
            for system in legacy_systems:
                if system.system_id in security_event.affected_systems:
                    # Criticality impact
                    if system.criticality.value == 'mission_critical':
                        impact_score += 0.3
                        impact_factors.append(f"Mission-critical system {system.system_name}")
                    elif system.criticality.value == 'business_critical':
                        impact_score += 0.2
                        impact_factors.append(f"Business-critical system {system.system_name}")
                    
                    # Dependency impact
                    if system.dependent_systems:
                        dependency_impact = len(system.dependent_systems) * 0.05
                        impact_score += min(dependency_impact, 0.2)
                        impact_factors.append(f"{len(system.dependent_systems)} dependent systems")
        
        # Event type impact
        high_impact_events = {
            SecurityEventType.RANSOMWARE_ACTIVITY,
            SecurityEventType.DATA_EXFILTRATION,
            SecurityEventType.ZERO_DAY_EXPLOIT
        }
        
        if security_event.event_type in high_impact_events:
            impact_score += 0.2
            impact_factors.append(f"High-impact event type: {security_event.event_type.value}")
        
        # Cap the impact score
        impact_score = min(impact_score, 1.0)
        
        return {
            'business_impact_score': impact_score,
            'impact_factors': impact_factors,
            'estimated_financial_impact': self._estimate_financial_impact(impact_score),
            'operational_impact_level': self._get_operational_impact_level(impact_score),
            'recovery_time_estimate': self._estimate_recovery_time(impact_score)
        }
    
    def _create_threat_analysis(
        self,
        security_event: SecurityEvent,
        ml_classification: Dict[str, Any],
        severity_assessment: Dict[str, Any],
        context_analysis: Dict[str, Any],
        intelligence_correlation: Dict[str, Any],
        business_impact: Dict[str, Any]
    ) -> ThreatAnalysis:
        """Create comprehensive threat analysis from all components."""
        
        # Calculate overall confidence
        confidence_factors = [
            ml_classification.get('confidence', 0.5) * 0.3,
            severity_assessment.get('severity_score', 0.5) * 0.2,
            context_analysis.get('context_score', 0.5) * 0.2,
            intelligence_correlation.get('relevance_score', 0.0) * 0.2,
            (1.0 - security_event.false_positive_likelihood) * 0.1
        ]
        
        overall_confidence = sum(confidence_factors)
        confidence_level = self._confidence_score_to_level(overall_confidence)
        
        return ThreatAnalysis(
            event_id=security_event.event_id,
            threat_classification=ml_classification.get('predicted_type', security_event.event_type),
            threat_level=severity_assessment.get('threat_level', security_event.threat_level),
            confidence=confidence_level,
            confidence_score=overall_confidence,
            
            # Threat intelligence
            threat_actor=intelligence_correlation.get('threat_actors', [None])[0],
            ttps=self._extract_ttps_from_analysis(security_event, intelligence_correlation),
            mitre_attack_techniques=self._map_to_mitre_techniques(security_event),
            
            # Impact assessment
            potential_damage=business_impact,
            affected_assets=security_event.affected_systems.copy(),
            business_impact_score=business_impact.get('business_impact_score', 0.0),
            financial_impact_estimate=business_impact.get('estimated_financial_impact'),
            
            # Analysis details
            root_cause_analysis=context_analysis.get('llm_analysis'),
            attack_methodology=self._infer_attack_methodology(security_event),
            
            # Risk factors
            exploitability_score=self._calculate_exploitability_score(security_event),
            prevalence_score=self._calculate_prevalence_score(intelligence_correlation),
            detectability_score=overall_confidence,
            
            # Recommendations
            immediate_actions=self._generate_immediate_actions(security_event, severity_assessment),
            short_term_actions=self._generate_short_term_actions(context_analysis),
            long_term_actions=self._generate_long_term_actions(business_impact),
            
            # Metadata
            processing_time_ms=(time.time() - time.time()) * 1000,  # Will be updated by caller
            data_sources=['ml_models', 'threat_intelligence', 'llm_analysis']
        )
    
    # Helper methods
    def _create_cache_key(self, security_event: SecurityEvent) -> str:
        """Create cache key for threat analysis."""
        return f"{security_event.event_id}_{security_event.event_type.value}_{hash(security_event.description)}"
    
    def _threat_level_to_score(self, threat_level: ThreatLevel) -> float:
        """Convert threat level to numeric score."""
        mapping = {
            ThreatLevel.CRITICAL: 1.0,
            ThreatLevel.HIGH: 0.8,
            ThreatLevel.MEDIUM: 0.6,
            ThreatLevel.LOW: 0.4,
            ThreatLevel.INFO: 0.2
        }
        return mapping.get(threat_level, 0.5)
    
    def _score_to_threat_level(self, score: float) -> ThreatLevel:
        """Convert numeric score to threat level."""
        if score >= 0.9:
            return ThreatLevel.CRITICAL
        elif score >= 0.7:
            return ThreatLevel.HIGH
        elif score >= 0.5:
            return ThreatLevel.MEDIUM
        elif score >= 0.3:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.INFO
    
    def _confidence_score_to_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric confidence to confidence level."""
        if score >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.85:
            return ConfidenceLevel.HIGH
        elif score >= 0.70:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _calculate_context_severity_multiplier(self, context: SecurityContext) -> float:
        """Calculate severity multiplier based on context."""
        multiplier = 1.0
        
        # Current threat level
        if context.current_threat_level == ThreatLevel.CRITICAL:
            multiplier *= 1.3
        elif context.current_threat_level == ThreatLevel.HIGH:
            multiplier *= 1.2
        
        # Active incidents
        if len(context.active_incidents) > 3:
            multiplier *= 1.2
        elif len(context.active_incidents) > 1:
            multiplier *= 1.1
        
        # Business hours (higher impact during business hours)
        if context.business_hours:
            multiplier *= 1.1
        
        return multiplier
    
    def _calculate_context_risk_score(self, context_data: Dict[str, Any]) -> float:
        """Calculate risk score from context data."""
        score = 0.5  # Base score
        
        # Security context factors
        security_ctx = context_data.get('security_context', {})
        if security_ctx.get('current_threat_level') == 'critical':
            score += 0.2
        if security_ctx.get('active_incidents', 0) > 2:
            score += 0.1
        
        # Legacy systems factors
        legacy_systems = context_data.get('legacy_systems', [])
        critical_systems = sum(1 for sys in legacy_systems if sys.get('criticality') in ['mission_critical', 'business_critical'])
        score += min(critical_systems * 0.1, 0.3)
        
        return min(score, 1.0)
    
    def _extract_key_context_factors(self, llm_analysis: str) -> List[str]:
        """Extract key context factors from LLM analysis."""
        # Simple keyword extraction
        factors = []
        lines = llm_analysis.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['factor', 'risk', 'impact', 'concern']):
                if len(line) > 20 and len(line) < 200:
                    factors.append(line)
        
        return factors[:5]  # Top 5 factors
    
    def _extract_business_impact_indicators(self, context_data: Dict[str, Any]) -> List[str]:
        """Extract business impact indicators from context."""
        indicators = []
        
        # Legacy system indicators
        legacy_systems = context_data.get('legacy_systems', [])
        for system in legacy_systems:
            if system.get('criticality') in ['mission_critical', 'business_critical']:
                indicators.append(f"Critical system: {system.get('name')}")
        
        # Security context indicators
        security_ctx = context_data.get('security_context', {})
        if security_ctx.get('active_incidents', 0) > 0:
            indicators.append(f"Multiple active incidents: {security_ctx['active_incidents']}")
        
        return indicators
    
    def _match_ioc(self, event_ioc: str, intel_ioc: Dict[str, Any]) -> bool:
        """Match IOC from event with threat intelligence IOC."""
        # Simplified IOC matching
        intel_value = intel_ioc.get('value', '')
        return event_ioc.lower() == intel_value.lower()
    
    def _estimate_financial_impact(self, impact_score: float) -> float:
        """Estimate financial impact based on impact score."""
        # Base financial impact calculation
        base_impact = 50000  # $50K base
        return base_impact * (1 + impact_score * 10)
    
    def _get_operational_impact_level(self, impact_score: float) -> str:
        """Get operational impact level."""
        if impact_score >= 0.8:
            return 'critical'
        elif impact_score >= 0.6:
            return 'high'
        elif impact_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_recovery_time(self, impact_score: float) -> int:
        """Estimate recovery time in hours."""
        base_time = 4  # 4 hours base
        return int(base_time * (1 + impact_score * 5))
    
    def _extract_ttps_from_analysis(self, event: SecurityEvent, intelligence: Dict[str, Any]) -> List[str]:
        """Extract TTPs from analysis."""
        ttps = []
        
        # From attack vectors
        ttps.extend(event.attack_vectors)
        
        # From threat intelligence
        correlations = intelligence.get('correlations', [])
        for correlation in correlations:
            # In real implementation, this would extract TTPs from intelligence data
            pass
        
        # Add event-type specific TTPs
        event_ttps = {
            SecurityEventType.MALWARE_DETECTION: ['malware-execution', 'persistence'],
            SecurityEventType.INTRUSION_ATTEMPT: ['network-intrusion', 'credential-access'],
            SecurityEventType.DATA_EXFILTRATION: ['data-collection', 'exfiltration'],
            SecurityEventType.RANSOMWARE_ACTIVITY: ['encryption', 'extortion']
        }
        
        ttps.extend(event_ttps.get(event.event_type, []))
        
        return list(set(ttps))  # Remove duplicates
    
    def _map_to_mitre_techniques(self, event: SecurityEvent) -> List[str]:
        """Map event to MITRE ATT&CK techniques."""
        techniques = []
        
        # Simple mapping based on event type
        mitre_mapping = {
            SecurityEventType.MALWARE_DETECTION: ['T1059', 'T1055'],  # Command Line Interface, Process Injection
            SecurityEventType.INTRUSION_ATTEMPT: ['T1190', 'T1078'],  # Exploit Public-Facing Application, Valid Accounts
            SecurityEventType.DATA_EXFILTRATION: ['T1041', 'T1567'],  # Exfiltration Over C2 Channel, Exfiltration Over Web Service
            SecurityEventType.PRIVILEGE_ESCALATION: ['T1068', 'T1134'],  # Exploitation for Privilege Escalation, Access Token Manipulation
            SecurityEventType.LATERAL_MOVEMENT: ['T1021', 'T1570']  # Remote Services, Lateral Tool Transfer
        }
        
        techniques.extend(mitre_mapping.get(event.event_type, []))
        
        return techniques
    
    def _infer_attack_methodology(self, event: SecurityEvent) -> List[str]:
        """Infer attack methodology from event characteristics."""
        methodology = []
        
        # Based on event type
        if event.event_type == SecurityEventType.RANSOMWARE_ACTIVITY:
            methodology.extend(['initial-access', 'persistence', 'privilege-escalation', 'lateral-movement', 'encryption'])
        elif event.event_type == SecurityEventType.DATA_EXFILTRATION:
            methodology.extend(['reconnaissance', 'initial-access', 'credential-access', 'collection', 'exfiltration'])
        elif event.event_type == SecurityEventType.INTRUSION_ATTEMPT:
            methodology.extend(['reconnaissance', 'initial-access', 'persistence'])
        
        # Based on attack vectors
        if 'phishing' in event.attack_vectors:
            methodology.append('social-engineering')
        if 'exploit' in event.attack_vectors:
            methodology.append('vulnerability-exploitation')
        
        return methodology
    
    def _calculate_exploitability_score(self, event: SecurityEvent) -> float:
        """Calculate exploitability score."""
        score = 0.5  # Base score
        
        # More IOCs = higher exploitability
        if len(event.indicators_of_compromise) > 5:
            score += 0.2
        
        # More attack vectors = higher exploitability
        if len(event.attack_vectors) > 3:
            score += 0.2
        
        # Event type factor
        high_exploitability = {
            SecurityEventType.ZERO_DAY_EXPLOIT,
            SecurityEventType.VULNERABILITY_EXPLOIT
        }
        
        if event.event_type in high_exploitability:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_prevalence_score(self, intelligence: Dict[str, Any]) -> float:
        """Calculate prevalence score based on threat intelligence."""
        score = 0.3  # Base score
        
        # More correlations = higher prevalence
        correlations = intelligence.get('matched_intelligence_count', 0)
        if correlations > 0:
            score += min(correlations * 0.1, 0.4)
        
        # Known campaigns = higher prevalence
        campaigns = intelligence.get('associated_campaigns', [])
        if campaigns:
            score += 0.2
        
        return min(score, 1.0)
    
    def _generate_immediate_actions(self, event: SecurityEvent, severity: Dict[str, Any]) -> List[str]:
        """Generate immediate action recommendations."""
        actions = ['Alert security team']
        
        threat_level = severity.get('threat_level', ThreatLevel.MEDIUM)
        
        if threat_level in {ThreatLevel.CRITICAL, ThreatLevel.HIGH}:
            actions.extend([
                'Enable enhanced monitoring',
                'Isolate affected systems if safe',
                'Collect forensic evidence'
            ])
        
        if event.source_ip:
            actions.append(f'Block source IP: {event.source_ip}')
        
        return actions
    
    def _generate_short_term_actions(self, context: Dict[str, Any]) -> List[str]:
        """Generate short-term action recommendations."""
        actions = [
            'Conduct detailed forensic analysis',
            'Update threat intelligence',
            'Review and update security controls'
        ]
        
        # Add context-specific actions
        key_factors = context.get('key_factors', [])
        if any('legacy' in factor.lower() for factor in key_factors):
            actions.append('Review legacy system security posture')
        
        return actions
    
    def _generate_long_term_actions(self, business_impact: Dict[str, Any]) -> List[str]:
        """Generate long-term action recommendations."""
        actions = [
            'Update incident response procedures',
            'Conduct security awareness training',
            'Review and update security architecture'
        ]
        
        impact_score = business_impact.get('business_impact_score', 0.0)
        if impact_score > 0.7:
            actions.extend([
                'Consider additional security investments',
                'Implement additional monitoring capabilities',
                'Review business continuity plans'
            ])
        
        return actions
    
    async def _load_threat_intelligence(self):
        """Load threat intelligence data."""
        # In real implementation, this would load from threat intelligence feeds
        # For now, create sample intelligence data
        
        sample_intelligence = ThreatIntelligence(
            source="sample_feed",
            threat_type=SecurityEventType.MALWARE_DETECTION,
            threat_actors=["APT29", "Cozy Bear"],
            campaigns=["CozyDuke"],
            iocs=[
                {"type": "ip", "value": "192.168.1.100"},
                {"type": "domain", "value": "malicious-domain.com"}
            ],
            ttps=["spear-phishing", "credential-harvesting"],
            severity=ThreatLevel.HIGH,
            confidence=0.85
        )
        
        self.threat_intelligence[sample_intelligence.intelligence_id] = sample_intelligence
        
        self.logger.info(f"Loaded {len(self.threat_intelligence)} threat intelligence entries")
    
    async def update_threat_intelligence(self, intelligence: ThreatIntelligence):
        """Update threat intelligence database."""
        self.threat_intelligence[intelligence.intelligence_id] = intelligence
        self.logger.info(f"Updated threat intelligence: {intelligence.intelligence_id}")
    
    def get_cached_analysis(self, cache_key: str) -> Optional[ThreatAnalysis]:
        """Get cached threat analysis."""
        return self.analysis_cache.get(cache_key)
    
    def clear_analysis_cache(self):
        """Clear the analysis cache."""
        self.analysis_cache.clear()
        self.logger.info("Threat analysis cache cleared")
    
    async def get_analyzer_metrics(self) -> Dict[str, Any]:
        """Get analyzer metrics and statistics."""
        return {
            'total_analyses': len(self.analysis_cache),
            'threat_intelligence_entries': len(self.threat_intelligence),
            'cache_hit_rate': 0.0,  # Would be calculated from actual metrics
            'average_analysis_time': 0.0,  # Would be calculated from metrics
            'classification_model_loaded': self.classification_model is not None
        }
    
    async def shutdown(self):
        """Shutdown the threat analyzer."""
        self.logger.info("Shutting down Threat Analyzer...")
        self.clear_analysis_cache()
        self._initialized = False
        self.logger.info("Threat Analyzer shutdown complete")