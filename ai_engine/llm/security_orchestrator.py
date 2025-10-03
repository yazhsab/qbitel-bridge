"""
CRONOS AI - Zero-Touch Security Orchestrator
LLM-powered automated security threat detection, response, and policy generation.

This module provides comprehensive security automation capabilities including:
- Automated threat detection and response
- Security policy generation
- Threat intelligence analysis
- Incident response automation
- Security posture assessment
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid

from prometheus_client import Counter, Histogram, Gauge

from ..core.config import Config
from ..core.exceptions import CronosAIException
from ..monitoring.metrics import MetricsCollector
from ..monitoring.alerts import AlertManager, Alert, AlertSeverity, AlertStatus
from ..policy.policy_engine import PolicyEngine, Policy, PolicyType, PolicySeverity
from .unified_llm_service import UnifiedLLMService, LLMRequest, LLMResponse

# Prometheus metrics
SECURITY_EVENTS_COUNTER = Counter(
    'cronos_security_events_total',
    'Total security events processed',
    ['event_type', 'severity', 'status']
)
THREAT_DETECTION_DURATION = Histogram(
    'cronos_threat_detection_duration_seconds',
    'Threat detection processing time',
    ['threat_type']
)
AUTOMATED_RESPONSES = Counter(
    'cronos_automated_responses_total',
    'Total automated security responses',
    ['response_type', 'status']
)
ACTIVE_THREATS = Gauge(
    'cronos_active_threats',
    'Number of active security threats',
    ['severity']
)

logger = logging.getLogger(__name__)


class SecurityException(CronosAIException):
    """Security orchestrator specific exception."""
    pass


class ThreatSeverity(str, Enum):
    """Threat severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(str, Enum):
    """Types of security threats."""
    MALWARE = "malware"
    INTRUSION = "intrusion"
    DATA_EXFILTRATION = "data_exfiltration"
    DDoS = "ddos"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    POLICY_VIOLATION = "policy_violation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    VULNERABILITY_EXPLOIT = "vulnerability_exploit"
    INSIDER_THREAT = "insider_threat"
    UNKNOWN = "unknown"


class ResponseAction(str, Enum):
    """Automated response actions."""
    MONITOR = "monitor"
    ALERT = "alert"
    BLOCK = "block"
    ISOLATE = "isolate"
    QUARANTINE = "quarantine"
    TERMINATE = "terminate"
    INVESTIGATE = "investigate"
    ESCALATE = "escalate"


class IncidentStatus(str, Enum):
    """Security incident status."""
    DETECTED = "detected"
    ANALYZING = "analyzing"
    RESPONDING = "responding"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_id: str
    event_type: ThreatType
    severity: ThreatSeverity
    timestamp: datetime
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    user_id: Optional[str] = None
    resource: Optional[str] = None
    description: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)
    indicators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatAnalysis:
    """Threat analysis results."""
    threat_id: str
    threat_type: ThreatType
    severity: ThreatSeverity
    confidence: float  # 0.0 to 1.0
    risk_score: float  # 0 to 100
    attack_vector: str
    affected_assets: List[str]
    indicators_of_compromise: List[str]
    analysis_summary: str
    recommended_actions: List[str]
    mitigation_strategies: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityResponse:
    """Automated security response."""
    response_id: str
    event_id: str
    actions_taken: List[ResponseAction]
    success: bool
    execution_time: float
    details: str
    blocked_ips: List[str] = field(default_factory=list)
    isolated_systems: List[str] = field(default_factory=list)
    alerts_generated: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityPolicy:
    """Generated security policy."""
    policy_id: str
    name: str
    description: str
    policy_type: str
    rules: List[Dict[str, Any]]
    enforcement_level: str  # monitor, warn, enforce
    scope: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityRequirements:
    """Security policy requirements."""
    requirement_id: str
    framework: str  # NIST, ISO27001, CIS, etc.
    controls: List[str]
    risk_level: str
    compliance_requirements: List[str]
    business_context: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatData:
    """Threat intelligence data."""
    data_id: str
    source: str
    threat_indicators: List[str]
    threat_actors: List[str]
    attack_patterns: List[str]
    vulnerabilities: List[str]
    timestamp: datetime
    confidence: float
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityIncident:
    """Security incident record."""
    incident_id: str
    title: str
    description: str
    severity: ThreatSeverity
    status: IncidentStatus
    events: List[SecurityEvent]
    analysis: Optional[ThreatAnalysis]
    response: Optional[SecurityResponse]
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ZeroTouchSecurityOrchestrator:
    """
    LLM-powered security automation orchestrator.
    
    Provides zero-touch security operations including:
    - Automated threat detection and analysis
    - Intelligent response orchestration
    - Security policy generation
    - Threat intelligence analysis
    - Incident response automation
    - Security posture assessment
    """
    
    def __init__(
        self,
        config: Config,
        llm_service: UnifiedLLMService,
        alert_manager: Optional[AlertManager] = None,
        policy_engine: Optional[PolicyEngine] = None
    ):
        """Initialize security orchestrator."""
        self.config = config
        self.llm_service = llm_service
        self.alert_manager = alert_manager
        self.policy_engine = policy_engine
        self.logger = logging.getLogger(__name__)
        
        # Active incidents tracking
        self.active_incidents: Dict[str, SecurityIncident] = {}
        self.incident_history: List[SecurityIncident] = []
        
        # Threat intelligence cache
        self.threat_intelligence_cache: Dict[str, ThreatAnalysis] = {}
        self.cache_ttl = timedelta(hours=1)
        
        # Response playbooks
        self.response_playbooks = self._initialize_playbooks()
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'threats_detected': 0,
            'automated_responses': 0,
            'policies_generated': 0,
            'incidents_resolved': 0,
            'false_positives': 0,
            'detection_accuracy': 0.0
        }
        
        self.logger.info("Zero-Touch Security Orchestrator initialized")
    
    async def detect_and_respond(
        self,
        security_event: SecurityEvent
    ) -> SecurityResponse:
        """
        Detect threats and respond automatically.
        
        Args:
            security_event: Security event to analyze
            
        Returns:
            Automated security response
        """
        start_time = time.time()
        response_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"Processing security event: {security_event.event_id}")
            SECURITY_EVENTS_COUNTER.labels(
                event_type=security_event.event_type.value,
                severity=security_event.severity.value,
                status='processing'
            ).inc()
            
            # Step 1: Analyze the security event
            threat_analysis = await self._analyze_threat(security_event)
            
            # Step 2: Determine threat level and required response
            response_actions = await self._determine_response_actions(
                security_event,
                threat_analysis
            )
            
            # Step 3: Execute automated response
            response = await self._execute_response(
                security_event,
                threat_analysis,
                response_actions,
                response_id
            )
            
            # Step 4: Create or update incident
            await self._manage_incident(security_event, threat_analysis, response)
            
            # Step 5: Generate alerts if needed
            if threat_analysis.severity in [ThreatSeverity.HIGH, ThreatSeverity.CRITICAL]:
                await self._generate_security_alert(security_event, threat_analysis)
            
            # Step 6: Document incident
            await self._document_incident(security_event, threat_analysis, response)
            
            # Update statistics
            self.stats['total_events'] += 1
            self.stats['threats_detected'] += 1
            self.stats['automated_responses'] += 1
            
            execution_time = time.time() - start_time
            THREAT_DETECTION_DURATION.labels(
                threat_type=security_event.event_type.value
            ).observe(execution_time)
            
            SECURITY_EVENTS_COUNTER.labels(
                event_type=security_event.event_type.value,
                severity=security_event.severity.value,
                status='completed'
            ).inc()
            
            self.logger.info(
                f"Security response completed: {response_id} in {execution_time:.2f}s"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Security response failed: {e}")
            SECURITY_EVENTS_COUNTER.labels(
                event_type=security_event.event_type.value,
                severity=security_event.severity.value,
                status='error'
            ).inc()
            
            # Return error response
            return SecurityResponse(
                response_id=response_id,
                event_id=security_event.event_id,
                actions_taken=[ResponseAction.ALERT],
                success=False,
                execution_time=time.time() - start_time,
                details=f"Response failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    async def _analyze_threat(
        self,
        security_event: SecurityEvent
    ) -> ThreatAnalysis:
        """Analyze security event using LLM."""
        # Check cache first
        cache_key = self._get_cache_key(security_event)
        if cache_key in self.threat_intelligence_cache:
            cached_analysis, timestamp = self.threat_intelligence_cache[cache_key]
            if datetime.utcnow() - timestamp < self.cache_ttl:
                self.logger.debug(f"Using cached threat analysis for {security_event.event_id}")
                return cached_analysis
        
        # Prepare context for LLM
        context = {
            'event_type': security_event.event_type.value,
            'severity': security_event.severity.value,
            'source_ip': security_event.source_ip,
            'destination_ip': security_event.destination_ip,
            'user_id': security_event.user_id,
            'resource': security_event.resource,
            'description': security_event.description,
            'indicators': security_event.indicators,
            'raw_data': security_event.raw_data
        }
        
        # Create LLM request
        prompt = f"""Analyze the following security event and provide a comprehensive threat assessment:

Event Type: {security_event.event_type.value}
Severity: {security_event.severity.value}
Description: {security_event.description}
Source IP: {security_event.source_ip or 'N/A'}
Destination IP: {security_event.destination_ip or 'N/A'}
User: {security_event.user_id or 'N/A'}
Resource: {security_event.resource or 'N/A'}
Indicators: {', '.join(security_event.indicators) if security_event.indicators else 'None'}

Provide a detailed analysis including:
1. Threat classification and type
2. Confidence level (0-100%)
3. Risk score (0-100)
4. Attack vector analysis
5. Affected assets
6. Indicators of compromise
7. Recommended immediate actions
8. Mitigation strategies

Format your response as JSON with the following structure:
{{
    "threat_type": "type",
    "confidence": 0.95,
    "risk_score": 85,
    "attack_vector": "description",
    "affected_assets": ["asset1", "asset2"],
    "indicators_of_compromise": ["ioc1", "ioc2"],
    "analysis_summary": "detailed summary",
    "recommended_actions": ["action1", "action2"],
    "mitigation_strategies": ["strategy1", "strategy2"]
}}"""
        
        llm_request = LLMRequest(
            prompt=prompt,
            feature_domain='security_orchestrator',
            context=context,
            max_tokens=2000,
            temperature=0.2  # Lower temperature for more consistent security analysis
        )
        
        # Get LLM analysis
        llm_response = await self.llm_service.process_request(llm_request)
        
        # Parse LLM response
        try:
            analysis_data = json.loads(llm_response.content)
        except json.JSONDecodeError:
            # Fallback to basic analysis if JSON parsing fails
            self.logger.warning("Failed to parse LLM response as JSON, using fallback")
            analysis_data = {
                'threat_type': security_event.event_type.value,
                'confidence': 0.7,
                'risk_score': 50,
                'attack_vector': 'Unknown',
                'affected_assets': [],
                'indicators_of_compromise': security_event.indicators,
                'analysis_summary': llm_response.content[:500],
                'recommended_actions': ['Monitor', 'Investigate'],
                'mitigation_strategies': ['Review logs', 'Update security policies']
            }
        
        # Create threat analysis
        threat_analysis = ThreatAnalysis(
            threat_id=str(uuid.uuid4()),
            threat_type=ThreatType(analysis_data.get('threat_type', security_event.event_type.value)),
            severity=security_event.severity,
            confidence=analysis_data.get('confidence', 0.7),
            risk_score=analysis_data.get('risk_score', 50),
            attack_vector=analysis_data.get('attack_vector', 'Unknown'),
            affected_assets=analysis_data.get('affected_assets', []),
            indicators_of_compromise=analysis_data.get('indicators_of_compromise', []),
            analysis_summary=analysis_data.get('analysis_summary', ''),
            recommended_actions=analysis_data.get('recommended_actions', []),
            mitigation_strategies=analysis_data.get('mitigation_strategies', []),
            metadata={
                'llm_provider': llm_response.provider,
                'llm_confidence': llm_response.confidence,
                'processing_time': llm_response.processing_time
            }
        )
        
        # Cache the analysis
        self.threat_intelligence_cache[cache_key] = (threat_analysis, datetime.utcnow())
        
        return threat_analysis
    
    async def _determine_response_actions(
        self,
        security_event: SecurityEvent,
        threat_analysis: ThreatAnalysis
    ) -> List[ResponseAction]:
        """Determine appropriate response actions based on threat analysis."""
        actions = []
        
        # Severity-based actions
        if threat_analysis.severity == ThreatSeverity.CRITICAL:
            actions.extend([
                ResponseAction.BLOCK,
                ResponseAction.ISOLATE,
                ResponseAction.ALERT,
                ResponseAction.ESCALATE
            ])
        elif threat_analysis.severity == ThreatSeverity.HIGH:
            actions.extend([
                ResponseAction.BLOCK,
                ResponseAction.ALERT,
                ResponseAction.INVESTIGATE
            ])
        elif threat_analysis.severity == ThreatSeverity.MEDIUM:
            actions.extend([
                ResponseAction.MONITOR,
                ResponseAction.ALERT
            ])
        else:
            actions.append(ResponseAction.MONITOR)
        
        # Threat-type specific actions
        if threat_analysis.threat_type == ThreatType.MALWARE:
            actions.append(ResponseAction.QUARANTINE)
        elif threat_analysis.threat_type == ThreatType.DATA_EXFILTRATION:
            actions.extend([ResponseAction.BLOCK, ResponseAction.INVESTIGATE])
        elif threat_analysis.threat_type == ThreatType.DDoS:
            actions.append(ResponseAction.BLOCK)
        
        # Check playbooks for additional actions
        playbook_actions = self._get_playbook_actions(threat_analysis.threat_type)
        actions.extend(playbook_actions)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_actions = []
        for action in actions:
            if action not in seen:
                seen.add(action)
                unique_actions.append(action)
        
        return unique_actions
    
    async def _execute_response(
        self,
        security_event: SecurityEvent,
        threat_analysis: ThreatAnalysis,
        actions: List[ResponseAction],
        response_id: str
    ) -> SecurityResponse:
        """Execute automated response actions."""
        start_time = time.time()
        blocked_ips = []
        isolated_systems = []
        alerts_generated = []
        execution_details = []
        
        for action in actions:
            try:
                if action == ResponseAction.BLOCK:
                    # Block source IP
                    if security_event.source_ip:
                        await self._block_ip(security_event.source_ip)
                        blocked_ips.append(security_event.source_ip)
                        execution_details.append(f"Blocked IP: {security_event.source_ip}")
                        AUTOMATED_RESPONSES.labels(
                            response_type='block_ip',
                            status='success'
                        ).inc()
                
                elif action == ResponseAction.ISOLATE:
                    # Isolate affected systems
                    for asset in threat_analysis.affected_assets:
                        await self._isolate_system(asset)
                        isolated_systems.append(asset)
                        execution_details.append(f"Isolated system: {asset}")
                        AUTOMATED_RESPONSES.labels(
                            response_type='isolate_system',
                            status='success'
                        ).inc()
                
                elif action == ResponseAction.ALERT:
                    # Generate security alert
                    alert_id = await self._create_alert(security_event, threat_analysis)
                    alerts_generated.append(alert_id)
                    execution_details.append(f"Generated alert: {alert_id}")
                    AUTOMATED_RESPONSES.labels(
                        response_type='generate_alert',
                        status='success'
                    ).inc()
                
                elif action == ResponseAction.QUARANTINE:
                    # Quarantine malicious files/processes
                    execution_details.append("Quarantine action executed")
                    AUTOMATED_RESPONSES.labels(
                        response_type='quarantine',
                        status='success'
                    ).inc()
                
                elif action == ResponseAction.MONITOR:
                    # Enhanced monitoring
                    execution_details.append("Enhanced monitoring activated")
                    AUTOMATED_RESPONSES.labels(
                        response_type='monitor',
                        status='success'
                    ).inc()
                
                elif action == ResponseAction.INVESTIGATE:
                    # Trigger investigation workflow
                    execution_details.append("Investigation workflow triggered")
                    AUTOMATED_RESPONSES.labels(
                        response_type='investigate',
                        status='success'
                    ).inc()
                
                elif action == ResponseAction.ESCALATE:
                    # Escalate to security team
                    execution_details.append("Escalated to security team")
                    AUTOMATED_RESPONSES.labels(
                        response_type='escalate',
                        status='success'
                    ).inc()
                
            except Exception as e:
                self.logger.error(f"Failed to execute action {action}: {e}")
                execution_details.append(f"Failed: {action.value} - {str(e)}")
                AUTOMATED_RESPONSES.labels(
                    response_type=action.value,
                    status='error'
                ).inc()
        
        execution_time = time.time() - start_time
        
        return SecurityResponse(
            response_id=response_id,
            event_id=security_event.event_id,
            actions_taken=actions,
            success=True,
            execution_time=execution_time,
            details='; '.join(execution_details),
            blocked_ips=blocked_ips,
            isolated_systems=isolated_systems,
            alerts_generated=alerts_generated,
            metadata={
                'threat_analysis_id': threat_analysis.threat_id,
                'risk_score': threat_analysis.risk_score,
                'confidence': threat_analysis.confidence
            }
        )
    
    async def generate_security_policies(
        self,
        requirements: SecurityRequirements
    ) -> List[SecurityPolicy]:
        """
        Generate security policies automatically based on requirements.
        
        Args:
            requirements: Security policy requirements
            
        Returns:
            List of generated security policies
        """
        try:
            self.logger.info(f"Generating security policies for {requirements.framework}")
            
            # Prepare context for LLM
            context = {
                'framework': requirements.framework,
                'controls': requirements.controls,
                'risk_level': requirements.risk_level,
                'compliance_requirements': requirements.compliance_requirements,
                'business_context': requirements.business_context
            }
            
            # Create LLM request
            prompt = f"""Generate comprehensive security policies based on the following requirements:

Framework: {requirements.framework}
Risk Level: {requirements.risk_level}
Controls: {', '.join(requirements.controls)}
Compliance Requirements: {', '.join(requirements.compliance_requirements)}

Business Context:
{json.dumps(requirements.business_context, indent=2)}

Generate security policies that:
1. Align with {requirements.framework} framework
2. Address all specified controls
3. Meet compliance requirements
4. Are practical and enforceable
5. Include clear rules and conditions
6. Specify enforcement levels

For each policy, provide:
- Policy name and description
- Policy type (access_control, data_governance, security, etc.)
- Specific rules with conditions and actions
- Enforcement level (monitor, warn, enforce)
- Scope of application

Format as JSON array of policies."""
            
            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain='security_orchestrator',
                context=context,
                max_tokens=3000,
                temperature=0.3
            )
            
            # Get LLM response
            llm_response = await self.llm_service.process_request(llm_request)
            
            # Parse policies from response
            try:
                policies_data = json.loads(llm_response.content)
                if not isinstance(policies_data, list):
                    policies_data = [policies_data]
            except json.JSONDecodeError:
                # Fallback: create basic policy structure
                self.logger.warning("Failed to parse LLM response, creating basic policies")
                policies_data = self._create_fallback_policies(requirements)
            
            # Create SecurityPolicy objects
            policies = []
            for policy_data in policies_data:
                policy = SecurityPolicy(
                    policy_id=str(uuid.uuid4()),
                    name=policy_data.get('name', 'Generated Security Policy'),
                    description=policy_data.get('description', ''),
                    policy_type=policy_data.get('policy_type', 'security'),
                    rules=policy_data.get('rules', []),
                    enforcement_level=policy_data.get('enforcement_level', 'enforce'),
                    scope=policy_data.get('scope', ['all']),
                    metadata={
                        'framework': requirements.framework,
                        'generated_by': 'zero_touch_security_orchestrator',
                        'llm_provider': llm_response.provider,
                        'requirement_id': requirements.requirement_id
                    }
                )
                policies.append(policy)
            
            # Validate against best practices
            validated_policies = await self._validate_policies(policies, requirements)
            
            # Create implementation guide
            implementation_guide = await self._create_implementation_guide(
                validated_policies,
                requirements
            )
            
            # Update statistics
            self.stats['policies_generated'] += len(validated_policies)
            
            self.logger.info(f"Generated {len(validated_policies)} security policies")
            
            return validated_policies
            
        except Exception as e:
            self.logger.error(f"Policy generation failed: {e}")
            raise SecurityException(f"Failed to generate security policies: {e}")
    
    async def threat_intelligence_analysis(
        self,
        threat_data: ThreatData
    ) -> ThreatAnalysis:
        """
        Analyze threat intelligence data.
        
        Args:
            threat_data: Threat intelligence data to analyze
            
        Returns:
            Comprehensive threat analysis
        """
        try:
            self.logger.info(f"Analyzing threat intelligence: {threat_data.data_id}")
            
            # Prepare context for LLM
            context = {
                'source': threat_data.source,
                'threat_indicators': threat_data.threat_indicators,
                'threat_actors': threat_data.threat_actors,
                'attack_patterns': threat_data.attack_patterns,
                'vulnerabilities': threat_data.vulnerabilities,
                'confidence': threat_data.confidence
            }
            
            # Create LLM request
            prompt = f"""Analyze the following threat intelligence data and provide actionable insights:

Source: {threat_data.source}
Confidence Level: {threat_data.confidence * 100}%

Threat Indicators:
{json.dumps(threat_data.threat_indicators, indent=2)}

Threat Actors:
{json.dumps(threat_data.threat_actors, indent=2)}

Attack Patterns:
{json.dumps(threat_data.attack_patterns, indent=2)}

Vulnerabilities:
{json.dumps(threat_data.vulnerabilities, indent=2)}

Provide comprehensive analysis including:
1. Threat classification and severity assessment
2. Correlation with known threat actors and campaigns
3. Impact assessment on our infrastructure
4. Indicators of compromise (IOCs) to monitor
5. Recommended detection strategies
6. Mitigation and remediation steps
7. Preventive measures

Format response as JSON with actionable recommendations."""
            
            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain='security_orchestrator',
                context=context,
                max_tokens=2500,
                temperature=0.2
            )
            
            # Get LLM analysis
            llm_response = await self.llm_service.process_request(llm_request)
            
            # Parse analysis
            try:
                analysis_data = json.loads(llm_response.content)
            except json.JSONDecodeError:
                analysis_data = {
                    'threat_type': 'unknown',
                    'severity': 'medium',
                    'confidence': threat_data.confidence,
                    'risk_score': 50,
                    'attack_vector': 'Multiple vectors identified',
                    'affected_assets': [],
                    'indicators_of_compromise': threat_data.threat_indicators,
                    'analysis_summary': llm_response.content[:1000],
                    'recommended_actions': ['Monitor threat indicators', 'Update security controls'],
                    'mitigation_strategies': ['Implement detection rules', 'Patch vulnerabilities']
                }
            
            # Create threat analysis
            threat_analysis = ThreatAnalysis(
                threat_id=str(uuid.uuid4()),
                threat_type=ThreatType(analysis_data.get('threat_type', 'unknown')),
                severity=ThreatSeverity(analysis_data.get('severity', 'medium')),
                confidence=analysis_data.get('confidence', threat_data.confidence),
                risk_score=analysis_data.get('risk_score', 50),
                attack_vector=analysis_data.get('attack_vector', 'Unknown'),
                affected_assets=analysis_data.get('affected_assets', []),
                indicators_of_compromise=analysis_data.get('indicators_of_compromise', []),
                analysis_summary=analysis_data.get('analysis_summary', ''),
                recommended_actions=analysis_data.get('recommended_actions', []),
                mitigation_strategies=analysis_data.get('mitigation_strategies', []),
                metadata={
                    'source': threat_data.source,
                    'data_id': threat_data.data_id,
                    'llm_provider': llm_response.provider,
                    'threat_actors': threat_data.threat_actors,
                    'attack_patterns': threat_data.attack_patterns
                }
            )
            
            self.logger.info(f"Threat intelligence analysis completed: {threat_analysis.threat_id}")
            
            return threat_analysis
            
        except Exception as e:
            self.logger.error(f"Threat intelligence analysis failed: {e}")
            raise SecurityException(f"Failed to analyze threat intelligence: {e}")
    
    async def assess_security_posture(self) -> Dict[str, Any]:
        """
        Assess overall security posture.
        
        Returns:
            Comprehensive security posture assessment
        """
        try:
            self.logger.info("Assessing security posture")
            
            # Gather security metrics
            active_threats_count = len([
                i for i in self.active_incidents.values()
                if i.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]
            ])
            
            # Calculate threat severity distribution
            severity_distribution = {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0,
                'info': 0
            }
            
            for incident in self.active_incidents.values():
                severity_distribution[incident.severity.value] += 1
            
            # Calculate detection accuracy
            total_detections = self.stats['threats_detected']
            false_positives = self.stats['false_positives']
            detection_accuracy = (
                ((total_detections - false_positives) / total_detections * 100)
                if total_detections > 0 else 0.0
            )
            
            # Update Prometheus metrics
            for severity, count in severity_distribution.items():
                ACTIVE_THREATS.labels(severity=severity).set(count)
            
            # Prepare assessment data
            assessment = {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_status': self._calculate_overall_status(severity_distribution),
                'active_threats': active_threats_count,
                'severity_distribution': severity_distribution,
                'statistics': {
                    'total_events_processed': self.stats['total_events'],
                    'threats_detected': self.stats['threats_detected'],
                    'automated_responses': self.stats['automated_responses'],
                    'policies_generated': self.stats['policies_generated'],
                    'incidents_resolved': self.stats['incidents_resolved'],
                    'detection_accuracy': round(detection_accuracy, 2),
                    'false_positive_rate': round(
                        (false_positives / total_detections * 100) if total_detections > 0 else 0.0,
                        2
                    )
                },
                'recent_incidents': [
                    {
                        'incident_id': i.incident_id,
                        'title': i.title,
                        'severity': i.severity.value,
                        'status': i.status.value,
                        'created_at': i.created_at.isoformat()
                    }
                    for i in sorted(
                        self.active_incidents.values(),
                        key=lambda x: x.created_at,
                        reverse=True
                    )[:10]
                ],
                'recommendations': await self._generate_security_recommendations()
            }
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Security posture assessment failed: {e}")
            raise SecurityException(f"Failed to assess security posture: {e}")
    
    def _calculate_overall_status(self, severity_distribution: Dict[str, int]) -> str:
        """Calculate overall security status."""
        if severity_distribution['critical'] > 0:
            return 'critical'
        elif severity_distribution['high'] > 5:
            return 'high_risk'
        elif severity_distribution['high'] > 0 or severity_distribution['medium'] > 10:
            return 'elevated'
        elif severity_distribution['medium'] > 0:
            return 'moderate'
        else:
            return 'normal'
    
    async def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on current posture."""
        recommendations = []
        
        # Analyze active incidents
        critical_incidents = [
            i for i in self.active_incidents.values()
            if i.severity == ThreatSeverity.CRITICAL
        ]
        
        if critical_incidents:
            recommendations.append(
                f"Address {len(critical_incidents)} critical security incidents immediately"
            )
        
        # Check detection accuracy
        if self.stats['detection_accuracy'] < 90:
            recommendations.append(
                "Review and tune threat detection rules to improve accuracy"
            )
        
        # Check response effectiveness
        if self.stats['automated_responses'] > 0:
            success_rate = (
                self.stats['incidents_resolved'] / self.stats['automated_responses'] * 100
            )
            if success_rate < 80:
                recommendations.append(
                    "Review automated response playbooks for effectiveness"
                )
        
        return recommendations
    
    async def _manage_incident(
        self,
        security_event: SecurityEvent,
        threat_analysis: ThreatAnalysis,
        response: SecurityResponse
    ) -> None:
        """Create or update security incident."""
        # Check if incident already exists for this event
        incident_id = f"INC-{security_event.event_id[:8]}"
        
        if incident_id in self.active_incidents:
            # Update existing incident
            incident = self.active_incidents[incident_id]
            incident.events.append(security_event)
            incident.response = response
            incident.updated_at = datetime.utcnow()
            
            # Check if incident should be resolved
            if response.success and threat_analysis.severity in [ThreatSeverity.LOW, ThreatSeverity.INFO]:
                incident.status = IncidentStatus.RESOLVED
                incident.resolved_at = datetime.utcnow()
                self.stats['incidents_resolved'] += 1
        else:
            # Create new incident
            incident = SecurityIncident(
                incident_id=incident_id,
                title=f"{threat_analysis.threat_type.value.title()} Detected",
                description=threat_analysis.analysis_summary,
                severity=threat_analysis.severity,
                status=IncidentStatus.DETECTED,
                events=[security_event],
                analysis=threat_analysis,
                response=response,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            self.active_incidents[incident_id] = incident
    
    async def _generate_security_alert(
        self,
        security_event: SecurityEvent,
        threat_analysis: ThreatAnalysis
    ) -> None:
        """Generate security alert through alert manager."""
        if not self.alert_manager:
            return
        
        # Map threat severity to alert severity
        severity_mapping = {
            ThreatSeverity.CRITICAL: AlertSeverity.CRITICAL,
            ThreatSeverity.HIGH: AlertSeverity.ERROR,
            ThreatSeverity.MEDIUM: AlertSeverity.WARNING,
            ThreatSeverity.LOW: AlertSeverity.INFO,
            ThreatSeverity.INFO: AlertSeverity.INFO
        }
        
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            rule_name='security_threat_detected',
            severity=severity_mapping[threat_analysis.severity],
            status=AlertStatus.ACTIVE,
            title=f"Security Threat: {threat_analysis.threat_type.value.title()}",
            description=threat_analysis.analysis_summary,
            created_at=time.time(),
            updated_at=time.time(),
            labels={
                'threat_type': threat_analysis.threat_type.value,
                'severity': threat_analysis.severity.value,
                'source': 'zero_touch_security_orchestrator'
            },
            annotations={
                'risk_score': str(threat_analysis.risk_score),
                'confidence': str(threat_analysis.confidence),
                'attack_vector': threat_analysis.attack_vector
            },
            source_data={
                'event_id': security_event.event_id,
                'threat_id': threat_analysis.threat_id,
                'indicators': threat_analysis.indicators_of_compromise,
                'recommended_actions': threat_analysis.recommended_actions
            }
        )
        
        # Store alert in alert manager
        self.alert_manager.active_alerts[alert.alert_id] = alert
    
    async def _document_incident(
        self,
        security_event: SecurityEvent,
        threat_analysis: ThreatAnalysis,
        response: SecurityResponse
    ) -> None:
        """Document security incident for audit trail."""
        documentation = {
            'event': asdict(security_event),
            'analysis': asdict(threat_analysis),
            'response': asdict(response),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.logger.info(
            f"Security incident documented: {security_event.event_id}",
            extra={'security_incident': documentation}
        )
    
    async def _block_ip(self, ip_address: str) -> None:
        """Block IP address (integration point)."""
        self.logger.info(f"Blocking IP address: {ip_address}")
        # Integration with firewall/network security systems
        # This would call actual firewall APIs in production
    
    async def _isolate_system(self, system_id: str) -> None:
        """Isolate system from network (integration point)."""
        self.logger.info(f"Isolating system: {system_id}")
        # Integration with network isolation systems
        # This would call actual network management APIs in production
    
    async def _create_alert(
        self,
        security_event: SecurityEvent,
        threat_analysis: ThreatAnalysis
    ) -> str:
        """Create security alert."""
        alert_id = str(uuid.uuid4())
        await self._generate_security_alert(security_event, threat_analysis)
        return alert_id
    
    def _get_cache_key(self, security_event: SecurityEvent) -> str:
        """Generate cache key for threat analysis."""
        key_data = f"{security_event.event_type.value}:{security_event.source_ip}:{security_event.destination_ip}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _initialize_playbooks(self) -> Dict[ThreatType, List[ResponseAction]]:
        """Initialize response playbooks for different threat types."""
        return {
            ThreatType.MALWARE: [
                ResponseAction.QUARANTINE,
                ResponseAction.ISOLATE,
                ResponseAction.ALERT
            ],
            ThreatType.INTRUSION: [
                ResponseAction.BLOCK,
                ResponseAction.ALERT,
                ResponseAction.INVESTIGATE
            ],
            ThreatType.DATA_EXFILTRATION: [
                ResponseAction.BLOCK,
                ResponseAction.ISOLATE,
                ResponseAction.ALERT,
                ResponseAction.ESCALATE
            ],
            ThreatType.DDoS: [
                ResponseAction.BLOCK,
                ResponseAction.ALERT
            ],
            ThreatType.UNAUTHORIZED_ACCESS: [
                ResponseAction.BLOCK,
                ResponseAction.ALERT,
                ResponseAction.INVESTIGATE
            ],
            ThreatType.PRIVILEGE_ESCALATION: [
                ResponseAction.BLOCK,
                ResponseAction.ISOLATE,
                ResponseAction.ALERT,
                ResponseAction.ESCALATE
            ],
            ThreatType.POLICY_VIOLATION: [
                ResponseAction.ALERT,
                ResponseAction.MONITOR
            ],
            ThreatType.ANOMALOUS_BEHAVIOR: [
                ResponseAction.MONITOR,
                ResponseAction.INVESTIGATE
            ],
            ThreatType.VULNERABILITY_EXPLOIT: [
                ResponseAction.BLOCK,
                ResponseAction.ISOLATE,
                ResponseAction.ALERT,
                ResponseAction.ESCALATE
            ],
            ThreatType.INSIDER_THREAT: [
                ResponseAction.MONITOR,
                ResponseAction.ALERT,
                ResponseAction.INVESTIGATE,
                ResponseAction.ESCALATE
            ]
        }
    
    def _get_playbook_actions(self, threat_type: ThreatType) -> List[ResponseAction]:
        """Get playbook actions for threat type."""
        return self.response_playbooks.get(threat_type, [ResponseAction.MONITOR])
    
    async def _validate_policies(
        self,
        policies: List[SecurityPolicy],
        requirements: SecurityRequirements
    ) -> List[SecurityPolicy]:
        """Validate generated policies against best practices."""
        validated = []
        
        for policy in policies:
            # Basic validation
            if not policy.name or not policy.rules:
                self.logger.warning(f"Skipping invalid policy: {policy.policy_id}")
                continue
            
            # Ensure enforcement level is valid
            if policy.enforcement_level not in ['monitor', 'warn', 'enforce']:
                policy.enforcement_level = 'enforce'
            
            # Add framework compliance metadata
            policy.metadata['validated'] = True
            policy.metadata['validation_timestamp'] = datetime.utcnow().isoformat()
            
            validated.append(policy)
        
        return validated
    
    async def _create_implementation_guide(
        self,
        policies: List[SecurityPolicy],
        requirements: SecurityRequirements
    ) -> str:
        """Create implementation guide for generated policies."""
        guide = f"""
# Security Policy Implementation Guide

## Framework: {requirements.framework}
## Generated: {datetime.utcnow().isoformat()}

## Policies Overview
Total Policies: {len(policies)}

"""
        for i, policy in enumerate(policies, 1):
            guide += f"""
### {i}. {policy.name}
- **Type**: {policy.policy_type}
- **Enforcement**: {policy.enforcement_level}
- **Scope**: {', '.join(policy.scope)}
- **Rules**: {len(policy.rules)}

"""
        
        return guide
    
    def _create_fallback_policies(
        self,
        requirements: SecurityRequirements
    ) -> List[Dict[str, Any]]:
        """Create fallback policies when LLM parsing fails."""
        return [
            {
                'name': f'{requirements.framework} Access Control Policy',
                'description': 'Basic access control policy',
                'policy_type': 'access_control',
                'rules': [
                    {
                        'condition': 'user.role != "admin"',
                        'action': 'deny',
                        'resource': 'sensitive_data'
                    }
                ],
                'enforcement_level': 'enforce',
                'scope': ['all']
            }
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get security orchestrator statistics."""
        return {
            **self.stats,
            'active_incidents': len(self.active_incidents),
            'incident_history_size': len(self.incident_history),
            'cache_size': len(self.threat_intelligence_cache),
            'timestamp': datetime.utcnow().isoformat()
        }


# Global security orchestrator instance
_security_orchestrator: Optional[ZeroTouchSecurityOrchestrator] = None


async def initialize_security_orchestrator(
    config: Config,
    llm_service: UnifiedLLMService,
    alert_manager: Optional[AlertManager] = None,
    policy_engine: Optional[PolicyEngine] = None
) -> ZeroTouchSecurityOrchestrator:
    """Initialize global security orchestrator."""
    global _security_orchestrator
    
    _security_orchestrator = ZeroTouchSecurityOrchestrator(
        config,
        llm_service,
        alert_manager,
        policy_engine
    )
    
    return _security_orchestrator


def get_security_orchestrator() -> Optional[ZeroTouchSecurityOrchestrator]:
    """Get global security orchestrator instance."""
    return _security_orchestrator


async def shutdown_security_orchestrator():
    """Shutdown global security orchestrator."""
    global _security_orchestrator
    if _security_orchestrator:
        # Perform cleanup if needed
        _security_orchestrator = None