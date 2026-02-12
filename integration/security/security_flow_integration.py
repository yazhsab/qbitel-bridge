#!/usr/bin/env python3
"""
QBITEL - Security Components Data Flow Integration
Integrates security components with the main data processing pipeline.
"""

import asyncio
import logging
import json
import time
import hashlib
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aioredis
from concurrent.futures import ThreadPoolExecutor
import grpc
import grpc.aio
from pathlib import Path
import sys
import ssl
import cryptography
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from integration.config.unified_config import get_config, get_service_config
from integration.orchestrator.service_integration import get_orchestrator, Message
from integration.ai_pipeline.ml_protocol_bridge import AIAnalysisResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat level enumeration"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityAction(Enum):
    """Security action enumeration"""

    ALLOW = "allow"
    MONITOR = "monitor"
    QUARANTINE = "quarantine"
    BLOCK = "block"
    ALERT = "alert"


@dataclass
class SecurityContext:
    """Security context for data processing"""

    session_id: str
    source_ip: str
    dest_ip: str
    source_port: int
    dest_port: int
    protocol: str
    user_id: Optional[str] = None
    device_id: Optional[str] = None
    trust_level: float = 0.5
    security_labels: Set[str] = None
    encryption_required: bool = True


@dataclass
class ThreatAnalysis:
    """Threat analysis result"""

    threat_id: str
    threat_level: ThreatLevel
    threat_type: str
    confidence: float
    indicators: List[Dict[str, Any]]
    mitigations: List[str]
    timestamp: float
    analysis_duration_ms: float


@dataclass
class SecurityDecision:
    """Security decision for data processing"""

    decision_id: str
    action: SecurityAction
    reason: str
    confidence: float
    expiry_time: Optional[float] = None
    metadata: Dict[str, Any] = None


class SecurityFlowIntegrator:
    """
    Integrates security components with the main data processing pipeline.
    Handles threat detection, access control, encryption, and audit logging.
    """

    def __init__(self):
        self.config = get_service_config("security")
        self.orchestrator = get_orchestrator()

        # Security components
        self.threat_detector = None
        self.access_controller = None
        self.crypto_service = None
        self.audit_logger = None

        # Security state
        self.active_threats: Dict[str, ThreatAnalysis] = {}
        self.blocked_ips: Set[str] = set()
        self.security_policies: Dict[str, Any] = {}

        # Performance tracking
        self.security_checks_count = 0
        self.threats_detected = 0
        self.false_positives = 0

        # Caches
        self.decision_cache: Dict[str, SecurityDecision] = {}
        self.reputation_cache: Dict[str, float] = {}

        # Thread pools
        self.security_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="security_")
        self.crypto_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="crypto_")

        # Processing queues
        self.security_queue = asyncio.Queue(maxsize=5000)
        self.audit_queue = asyncio.Queue(maxsize=10000)

        # Real-time threat intelligence
        self.threat_intel_feeds = []
        self.ioc_database = {}  # Indicators of Compromise

    async def initialize(self):
        """Initialize security components"""
        logger.info("Initializing Security Flow Integrator...")

        try:
            # Initialize security services
            await self._init_security_services()

            # Load security policies
            await self._load_security_policies()

            # Initialize threat intelligence
            await self._init_threat_intelligence()

            # Start background tasks
            await self._start_security_tasks()

            logger.info("Security Flow Integrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize security integrator: {e}")
            raise

    async def _init_security_services(self):
        """Initialize core security services"""
        try:
            # Initialize threat detector
            self.threat_detector = ThreatDetector(
                ml_models_path=self.config.get("ml_models_path", "/opt/models"),
                threat_intel_feeds=self.config.get("threat_intel_feeds", []),
            )

            # Initialize access controller
            self.access_controller = AccessController(
                rbac_config=self.config.get("rbac_config", {}),
                session_timeout=self.config.get("session_timeout", 3600),
            )

            # Initialize crypto service
            self.crypto_service = CryptoService(
                encryption_key=self.config.get("encryption_key"),
                algorithm=self.config.get("encryption_algorithm", "AES-256-GCM"),
            )

            # Initialize audit logger
            self.audit_logger = AuditLogger(
                log_level=self.config.get("audit_log_level", "INFO"),
                retention_days=self.config.get("audit_log_retention_days", 365),
            )

            logger.info("Security services initialized")

        except Exception as e:
            logger.error(f"Error initializing security services: {e}")
            raise

    async def _load_security_policies(self):
        """Load security policies from configuration"""
        try:
            # Load from configuration or external policy store
            self.security_policies = {
                "default_deny": True,
                "encryption_required": True,
                "max_session_time": 3600,
                "threat_threshold": 0.7,
                "rate_limits": {"per_ip": 1000, "per_user": 500, "per_session": 100},
                "blocked_protocols": ["P2P", "BitTorrent"],
                "suspicious_ports": [1337, 4444, 5555, 6666, 31337],
                "geo_restrictions": {"blocked_countries": [], "allowed_countries": []},
            }

            logger.info(f"Loaded {len(self.security_policies)} security policies")

        except Exception as e:
            logger.error(f"Error loading security policies: {e}")

    async def _init_threat_intelligence(self):
        """Initialize threat intelligence feeds"""
        try:
            # Initialize threat intelligence feeds
            self.threat_intel_feeds = [
                "misp_feed",
                "abuse_ch",
                "emerging_threats",
                "custom_intel",
            ]

            # Load IOC database
            await self._load_ioc_database()

            logger.info("Threat intelligence initialized")

        except Exception as e:
            logger.error(f"Error initializing threat intelligence: {e}")

    async def _load_ioc_database(self):
        """Load Indicators of Compromise database"""
        try:
            # In production, this would load from threat intel feeds
            self.ioc_database = {
                "malicious_ips": set(),
                "malicious_domains": set(),
                "malicious_hashes": set(),
                "suspicious_patterns": [],
            }

            logger.info("IOC database loaded")

        except Exception as e:
            logger.error(f"Error loading IOC database: {e}")

    async def _start_security_tasks(self):
        """Start background security tasks"""
        # Start security processing
        asyncio.create_task(self._security_processor())

        # Start audit processor
        asyncio.create_task(self._audit_processor())

        # Start threat intelligence updater
        asyncio.create_task(self._threat_intel_updater())

        # Start security metrics collector
        asyncio.create_task(self._security_metrics_collector())

        logger.info("Security background tasks started")

    async def process_data_security(
        self,
        data: bytes,
        context: SecurityContext,
        ai_analysis: Optional[AIAnalysisResult] = None,
    ) -> Tuple[bytes, SecurityDecision]:
        """
        Main entry point for security processing of data flow.
        Performs comprehensive security analysis and applies security policies.
        """
        start_time = time.time()

        try:
            # Generate decision ID
            decision_id = f"sec_{int(time.time() * 1000000)}"

            # Check cache first
            cache_key = self._generate_security_cache_key(context, data[:64])
            if cache_key in self.decision_cache:
                cached_decision = self.decision_cache[cache_key]
                if cached_decision.expiry_time is None or time.time() < cached_decision.expiry_time:
                    return await self._apply_security_decision(data, cached_decision)

            # Parallel security analyses
            security_tasks = [
                self._analyze_threats(data, context, ai_analysis),
                self._check_access_control(context),
                self._validate_encryption_requirements(data, context),
                self._check_rate_limits(context),
                self._analyze_reputation(context),
            ]

            results = await asyncio.gather(*security_tasks, return_exceptions=True)

            # Process results
            threat_analysis = results[0] if not isinstance(results[0], Exception) else None
            access_granted = results[1] if not isinstance(results[1], Exception) else False
            encryption_valid = results[2] if not isinstance(results[2], Exception) else False
            rate_limit_ok = results[3] if not isinstance(results[3], Exception) else False
            reputation_score = results[4] if not isinstance(results[4], Exception) else 0.5

            # Make security decision
            decision = await self._make_security_decision(
                decision_id,
                threat_analysis,
                access_granted,
                encryption_valid,
                rate_limit_ok,
                reputation_score,
                context,
            )

            # Cache decision
            decision.expiry_time = time.time() + 300  # 5 minute cache
            self.decision_cache[cache_key] = decision

            # Limit cache size
            if len(self.decision_cache) > 10000:
                oldest_key = next(iter(self.decision_cache))
                del self.decision_cache[oldest_key]

            # Apply decision
            processed_data = await self._apply_security_decision(data, decision)

            # Log security event
            await self._log_security_event(
                context,
                decision,
                threat_analysis,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            # Update metrics
            self.security_checks_count += 1
            if threat_analysis and threat_analysis.threat_level in [
                ThreatLevel.HIGH,
                ThreatLevel.CRITICAL,
            ]:
                self.threats_detected += 1

            return processed_data, decision

        except Exception as e:
            logger.error(f"Error in security processing: {e}")

            # Default to block on error for security
            error_decision = SecurityDecision(
                decision_id=decision_id,
                action=SecurityAction.BLOCK,
                reason=f"Security processing error: {str(e)}",
                confidence=1.0,
                metadata={"error": str(e)},
            )

            return data, error_decision

    async def _analyze_threats(
        self,
        data: bytes,
        context: SecurityContext,
        ai_analysis: Optional[AIAnalysisResult],
    ) -> Optional[ThreatAnalysis]:
        """Analyze data for threats"""
        try:
            threat_indicators = []
            threat_score = 0.0

            # Check against IOC database
            if await self._check_ioc_matches(data, context):
                threat_indicators.append(
                    {
                        "type": "IOC_match",
                        "description": "Data matches known IOC patterns",
                        "severity": "high",
                    }
                )
                threat_score += 0.8

            # Analyze AI results for threats
            if ai_analysis:
                if ai_analysis.anomaly_score > 0.8:
                    threat_indicators.append(
                        {
                            "type": "anomaly_detection",
                            "description": f"High anomaly score: {ai_analysis.anomaly_score}",
                            "severity": "medium",
                        }
                    )
                    threat_score += ai_analysis.anomaly_score * 0.5

                if ai_analysis.threat_level in ["high", "critical"]:
                    threat_indicators.append(
                        {
                            "type": "ai_threat_detection",
                            "description": f"AI detected {ai_analysis.threat_level} threat",
                            "severity": ai_analysis.threat_level,
                        }
                    )
                    threat_score += 0.7

            # Check suspicious patterns
            suspicious_patterns = await self._check_suspicious_patterns(data, context)
            if suspicious_patterns:
                threat_indicators.extend(suspicious_patterns)
                threat_score += len(suspicious_patterns) * 0.2

            # Determine threat level
            if threat_score >= 0.9:
                threat_level = ThreatLevel.CRITICAL
            elif threat_score >= 0.7:
                threat_level = ThreatLevel.HIGH
            elif threat_score >= 0.4:
                threat_level = ThreatLevel.MEDIUM
            else:
                threat_level = ThreatLevel.LOW

            if threat_indicators or threat_score > 0.3:
                return ThreatAnalysis(
                    threat_id=f"threat_{int(time.time() * 1000000)}",
                    threat_level=threat_level,
                    threat_type="composite_analysis",
                    confidence=min(threat_score, 1.0),
                    indicators=threat_indicators,
                    mitigations=self._get_threat_mitigations(threat_level),
                    timestamp=time.time(),
                    analysis_duration_ms=50,  # Approximate
                )

            return None

        except Exception as e:
            logger.error(f"Error analyzing threats: {e}")
            return None

    async def _check_access_control(self, context: SecurityContext) -> bool:
        """Check access control policies"""
        try:
            # Check if IP is blocked
            if context.source_ip in self.blocked_ips:
                return False

            # Check user permissions if available
            if context.user_id:
                return await self.access_controller.check_user_access(context.user_id, context.protocol, context.dest_port)

            # Check device trust level
            if context.device_id and context.trust_level < 0.3:
                return False

            # Check protocol restrictions
            if context.protocol in self.security_policies.get("blocked_protocols", []):
                return False

            # Check port restrictions
            suspicious_ports = self.security_policies.get("suspicious_ports", [])
            if context.dest_port in suspicious_ports:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking access control: {e}")
            return False

    async def _validate_encryption_requirements(self, data: bytes, context: SecurityContext) -> bool:
        """Validate encryption requirements"""
        try:
            if not self.security_policies.get("encryption_required", True):
                return True

            # Check if data appears encrypted (high entropy)
            entropy = self._calculate_entropy(data[:1024])
            if entropy > 7.0:  # Likely encrypted
                return True

            # Check for TLS indicators
            if data.startswith(b"\x16\x03"):  # TLS handshake
                return True

            # Check if encryption is required for this context
            if context.encryption_required:
                return False  # Data should be encrypted but isn't

            return True

        except Exception as e:
            logger.error(f"Error validating encryption: {e}")
            return True  # Default to allow on error

    async def _check_rate_limits(self, context: SecurityContext) -> bool:
        """Check rate limiting policies"""
        try:
            rate_limits = self.security_policies.get("rate_limits", {})

            # Check per-IP rate limit
            if "per_ip" in rate_limits:
                if await self._check_rate_limit(f"ip:{context.source_ip}", rate_limits["per_ip"]):
                    return False

            # Check per-user rate limit
            if context.user_id and "per_user" in rate_limits:
                if await self._check_rate_limit(f"user:{context.user_id}", rate_limits["per_user"]):
                    return False

            # Check per-session rate limit
            if "per_session" in rate_limits:
                if await self._check_rate_limit(f"session:{context.session_id}", rate_limits["per_session"]):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking rate limits: {e}")
            return True  # Default to allow on error

    async def _analyze_reputation(self, context: SecurityContext) -> float:
        """Analyze IP/user reputation"""
        try:
            # Check cache first
            rep_key = f"rep:{context.source_ip}"
            if rep_key in self.reputation_cache:
                return self.reputation_cache[rep_key]

            reputation_score = 0.5  # Neutral starting point

            # Check threat intelligence feeds
            if await self._check_threat_intel(context.source_ip):
                reputation_score -= 0.4

            # Check historical behavior
            historical_score = await self._get_historical_reputation(context.source_ip)
            reputation_score = (reputation_score + historical_score) / 2

            # Cache result
            self.reputation_cache[rep_key] = reputation_score

            # Limit cache size
            if len(self.reputation_cache) > 5000:
                oldest_key = next(iter(self.reputation_cache))
                del self.reputation_cache[oldest_key]

            return reputation_score

        except Exception as e:
            logger.error(f"Error analyzing reputation: {e}")
            return 0.5  # Neutral score on error

    async def _make_security_decision(
        self,
        decision_id: str,
        threat_analysis: Optional[ThreatAnalysis],
        access_granted: bool,
        encryption_valid: bool,
        rate_limit_ok: bool,
        reputation_score: float,
        context: SecurityContext,
    ) -> SecurityDecision:
        """Make final security decision based on all analyses"""
        try:
            # Start with allow
            action = SecurityAction.ALLOW
            reasons = []
            confidence = 0.8

            # Check threat analysis
            if threat_analysis:
                if threat_analysis.threat_level == ThreatLevel.CRITICAL:
                    action = SecurityAction.BLOCK
                    reasons.append(f"Critical threat detected: {threat_analysis.threat_type}")
                    confidence = threat_analysis.confidence
                elif threat_analysis.threat_level == ThreatLevel.HIGH:
                    action = SecurityAction.QUARANTINE
                    reasons.append(f"High threat detected: {threat_analysis.threat_type}")
                    confidence = threat_analysis.confidence
                elif threat_analysis.threat_level == ThreatLevel.MEDIUM:
                    action = SecurityAction.MONITOR
                    reasons.append(f"Medium threat detected: {threat_analysis.threat_type}")

            # Check access control
            if not access_granted:
                action = SecurityAction.BLOCK
                reasons.append("Access denied by security policy")
                confidence = 1.0

            # Check encryption requirements
            if not encryption_valid:
                if self.security_policies.get("encryption_required", True):
                    action = SecurityAction.BLOCK
                    reasons.append("Encryption required but not present")
                    confidence = 0.9

            # Check rate limits
            if not rate_limit_ok:
                action = SecurityAction.BLOCK
                reasons.append("Rate limit exceeded")
                confidence = 1.0

            # Check reputation
            if reputation_score < 0.2:
                action = SecurityAction.BLOCK
                reasons.append("Low reputation score")
                confidence = 0.8
            elif reputation_score < 0.4:
                if action == SecurityAction.ALLOW:
                    action = SecurityAction.MONITOR
                    reasons.append("Moderate reputation concern")

            # Create decision
            decision = SecurityDecision(
                decision_id=decision_id,
                action=action,
                reason=("; ".join(reasons) if reasons else "No security concerns detected"),
                confidence=confidence,
                metadata={
                    "threat_analysis": (asdict(threat_analysis) if threat_analysis else None),
                    "access_granted": access_granted,
                    "encryption_valid": encryption_valid,
                    "rate_limit_ok": rate_limit_ok,
                    "reputation_score": reputation_score,
                },
            )

            return decision

        except Exception as e:
            logger.error(f"Error making security decision: {e}")
            return SecurityDecision(
                decision_id=decision_id,
                action=SecurityAction.BLOCK,
                reason=f"Security decision error: {str(e)}",
                confidence=1.0,
            )

    async def _apply_security_decision(self, data: bytes, decision: SecurityDecision) -> bytes:
        """Apply security decision to data"""
        try:
            if decision.action == SecurityAction.BLOCK:
                # Block the data
                return b""  # Return empty data

            elif decision.action == SecurityAction.QUARANTINE:
                # Quarantine - could encrypt or tag the data
                quarantine_header = b"QUARANTINE:"
                return quarantine_header + data

            elif decision.action in [
                SecurityAction.ALLOW,
                SecurityAction.MONITOR,
                SecurityAction.ALERT,
            ]:
                # Allow data through (monitoring/alerting handled separately)
                return data

            return data

        except Exception as e:
            logger.error(f"Error applying security decision: {e}")
            return data

    # Helper methods
    async def _check_ioc_matches(self, data: bytes, context: SecurityContext) -> bool:
        """Check data against IOC database"""
        try:
            # Check IP against malicious IPs
            if context.source_ip in self.ioc_database.get("malicious_ips", set()):
                return True

            # Check for malicious patterns in data
            data_str = data.decode("utf-8", errors="ignore").lower()
            for pattern in self.ioc_database.get("suspicious_patterns", []):
                if pattern in data_str:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking IOCs: {e}")
            return False

    async def _check_suspicious_patterns(self, data: bytes, context: SecurityContext) -> List[Dict[str, Any]]:
        """Check for suspicious patterns in data"""
        patterns = []

        try:
            # Check for common attack patterns
            data_str = data.decode("utf-8", errors="ignore").lower()

            # SQL injection patterns
            sql_patterns = ["union select", "drop table", "1=1", "or 1=1"]
            for pattern in sql_patterns:
                if pattern in data_str:
                    patterns.append(
                        {
                            "type": "sql_injection",
                            "description": f"SQL injection pattern detected: {pattern}",
                            "severity": "high",
                        }
                    )

            # XSS patterns
            xss_patterns = ["<script>", "javascript:", "onerror="]
            for pattern in xss_patterns:
                if pattern in data_str:
                    patterns.append(
                        {
                            "type": "xss",
                            "description": f"XSS pattern detected: {pattern}",
                            "severity": "medium",
                        }
                    )

            # Check for suspicious ports
            if context.dest_port in self.security_policies.get("suspicious_ports", []):
                patterns.append(
                    {
                        "type": "suspicious_port",
                        "description": f"Communication to suspicious port: {context.dest_port}",
                        "severity": "medium",
                    }
                )

        except Exception as e:
            logger.error(f"Error checking suspicious patterns: {e}")

        return patterns

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        try:
            if not data:
                return 0.0

            # Count byte frequencies
            byte_counts = [0] * 256
            for byte in data:
                byte_counts[byte] += 1

            # Calculate entropy
            entropy = 0.0
            data_len = len(data)

            for count in byte_counts:
                if count > 0:
                    probability = count / data_len
                    entropy -= probability * (probability.bit_length() - 1)

            return entropy

        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            return 0.0

    async def _check_rate_limit(self, key: str, limit: int) -> bool:
        """Check if rate limit is exceeded"""
        try:
            # Use Redis for distributed rate limiting if available
            if hasattr(self.orchestrator, "redis_client") and self.orchestrator.redis_client:
                current = await self.orchestrator.redis_client.incr(f"rate_limit:{key}")
                if current == 1:
                    await self.orchestrator.redis_client.expire(f"rate_limit:{key}", 60)

                return current > limit

            # Fallback to local tracking (not distributed)
            return False

        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False

    async def _check_threat_intel(self, ip: str) -> bool:
        """Check IP against threat intelligence feeds"""
        try:
            # In production, this would query real threat intel feeds
            return ip in self.ioc_database.get("malicious_ips", set())
        except Exception as e:
            logger.error(f"Error checking threat intel: {e}")
            return False

    async def _get_historical_reputation(self, ip: str) -> float:
        """Get historical reputation for IP"""
        try:
            # In production, this would query historical data
            return 0.5  # Neutral score
        except Exception as e:
            logger.error(f"Error getting historical reputation: {e}")
            return 0.5

    def _get_threat_mitigations(self, threat_level: ThreatLevel) -> List[str]:
        """Get recommended mitigations for threat level"""
        mitigations = {
            ThreatLevel.LOW: ["Monitor traffic", "Log event"],
            ThreatLevel.MEDIUM: [
                "Increase monitoring",
                "Alert security team",
                "Apply rate limiting",
            ],
            ThreatLevel.HIGH: [
                "Quarantine traffic",
                "Block suspicious IPs",
                "Escalate to security team",
            ],
            ThreatLevel.CRITICAL: [
                "Block immediately",
                "Alert all security teams",
                "Initiate incident response",
            ],
        }

        return mitigations.get(threat_level, ["Monitor traffic"])

    def _generate_security_cache_key(self, context: SecurityContext, data_sample: bytes) -> str:
        """Generate cache key for security decisions"""
        try:
            key_data = f"{context.source_ip}:{context.dest_port}:{context.protocol}:{data_sample}"
            return hashlib.sha256(key_data.encode()).hexdigest()[:16]
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return "default"

    async def _log_security_event(
        self,
        context: SecurityContext,
        decision: SecurityDecision,
        threat_analysis: Optional[ThreatAnalysis],
        processing_time_ms: float,
    ):
        """Log security event for audit and analysis"""
        try:
            event = {
                "timestamp": time.time(),
                "event_type": "security_decision",
                "decision_id": decision.decision_id,
                "action": decision.action.value,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "context": asdict(context),
                "threat_analysis": asdict(threat_analysis) if threat_analysis else None,
                "processing_time_ms": processing_time_ms,
            }

            await self.audit_queue.put(event)

        except Exception as e:
            logger.error(f"Error logging security event: {e}")

    # Background task methods
    async def _security_processor(self):
        """Background security processing"""
        while True:
            try:
                await asyncio.sleep(1)
                # Process security queue if needed
                # This would handle batch processing of security events

            except Exception as e:
                logger.error(f"Error in security processor: {e}")
                await asyncio.sleep(5)

    async def _audit_processor(self):
        """Process audit events"""
        while True:
            try:
                event = await asyncio.wait_for(self.audit_queue.get(), timeout=5.0)

                # Process audit event
                await self.audit_logger.log_event(event)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing audit event: {e}")
                await asyncio.sleep(1)

    async def _threat_intel_updater(self):
        """Update threat intelligence feeds"""
        while True:
            try:
                await asyncio.sleep(3600)  # Update every hour

                # Update IOC database from feeds
                await self._load_ioc_database()

            except Exception as e:
                logger.error(f"Error updating threat intelligence: {e}")
                await asyncio.sleep(3600)

    async def _security_metrics_collector(self):
        """Collect security metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Collect every minute

                metrics = {
                    "security_checks_count": self.security_checks_count,
                    "threats_detected": self.threats_detected,
                    "false_positives": self.false_positives,
                    "blocked_ips_count": len(self.blocked_ips),
                    "active_threats_count": len(self.active_threats),
                    "cache_hit_rate": len(self.decision_cache) / max(self.security_checks_count, 1),
                }

                # Send metrics to orchestrator
                message = Message(
                    id=f"security_metrics_{time.time()}",
                    timestamp=time.time(),
                    source="security_integrator",
                    destination="orchestrator",
                    message_type="metric_update",
                    payload={
                        "component": "security_integrator",
                        "metrics": metrics,
                    },
                )

                await self.orchestrator.send_message(message)

            except Exception as e:
                logger.error(f"Error collecting security metrics: {e}")
                await asyncio.sleep(60)

    # Public API
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get current security metrics"""
        return {
            "security_checks_count": self.security_checks_count,
            "threats_detected": self.threats_detected,
            "detection_rate": self.threats_detected / max(self.security_checks_count, 1),
            "blocked_ips_count": len(self.blocked_ips),
            "active_threats": len(self.active_threats),
        }

    async def block_ip(self, ip: str, reason: str, duration: Optional[int] = None):
        """Block an IP address"""
        self.blocked_ips.add(ip)

        # Schedule unblock if duration specified
        if duration:
            asyncio.create_task(self._schedule_unblock(ip, duration))

        logger.info(f"Blocked IP {ip}: {reason}")

    async def _schedule_unblock(self, ip: str, duration: int):
        """Schedule IP unblock after duration"""
        await asyncio.sleep(duration)
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            logger.info(f"Unblocked IP {ip} after {duration} seconds")

    async def shutdown(self):
        """Shutdown security integrator"""
        logger.info("Shutting down Security Flow Integrator...")

        # Shutdown thread pools
        self.security_executor.shutdown(wait=True)
        self.crypto_executor.shutdown(wait=True)

        logger.info("Security Flow Integrator shutdown complete")


# Placeholder classes for security components
class ThreatDetector:
    def __init__(self, ml_models_path: str, threat_intel_feeds: List[str]):
        self.ml_models_path = ml_models_path
        self.threat_intel_feeds = threat_intel_feeds


class AccessController:
    def __init__(self, rbac_config: Dict[str, Any], session_timeout: int):
        self.rbac_config = rbac_config
        self.session_timeout = session_timeout

    async def check_user_access(self, user_id: str, protocol: str, port: int) -> bool:
        return True  # Simplified for now


class CryptoService:
    def __init__(self, encryption_key: str, algorithm: str):
        self.encryption_key = encryption_key
        self.algorithm = algorithm


class AuditLogger:
    def __init__(self, log_level: str, retention_days: int):
        self.log_level = log_level
        self.retention_days = retention_days

    async def log_event(self, event: Dict[str, Any]):
        logger.info(f"Audit event: {json.dumps(event)}")


# Global security integrator instance
_security_integrator = None


def get_security_integrator() -> SecurityFlowIntegrator:
    """Get global security integrator instance"""
    global _security_integrator
    if _security_integrator is None:
        _security_integrator = SecurityFlowIntegrator()
    return _security_integrator


async def main():
    """Main entry point for security integrator"""
    integrator = SecurityFlowIntegrator()

    try:
        await integrator.initialize()

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Security integrator error: {e}")
    finally:
        await integrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
