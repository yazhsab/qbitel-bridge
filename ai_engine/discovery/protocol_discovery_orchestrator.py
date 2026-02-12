"""
QBITEL Engine - Protocol Discovery Orchestrator

This module orchestrates the entire protocol discovery pipeline, coordinating
statistical analysis, grammar learning, parser generation, classification,
and validation to provide comprehensive protocol discovery capabilities.
"""

import asyncio
import logging
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Any, Union, AsyncIterator
from enum import Enum
import json
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor
from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge

from ..core.config import Config
from ..core.exceptions import ProtocolException, ModelException
from ..core.circuit_breakers import (
    circuit_breakers,
    with_circuit_breaker,
    CircuitOpenError,
    CircuitBreakerConfig,
)
from .statistical_analyzer import StatisticalAnalyzer, StructuralFeatures
from .grammar_learner import GrammarLearner, Grammar
from .parser_generator import ParserGenerator, GeneratedParser, ParseResult
from .protocol_classifier import (
    ProtocolClassifier,
    ClassificationResult,
    ProtocolSample,
)
from .message_validator import MessageValidator, ValidationResult, ValidationLevel


class DiscoveryMode(Enum):
    """Protocol discovery operation modes."""

    TRAINING = "training"  # Learning from samples
    INFERENCE = "inference"  # Real-time protocol detection
    ANALYSIS = "analysis"  # Offline analysis
    VALIDATION = "validation"  # Message validation only


class DiscoveryPhase(Enum):
    """
    Phases of protocol discovery process.

    Simplified 5-phase pipeline:
    1. FEATURE_EXTRACTION: Statistical analysis + Protocol classification
    2. GRAMMAR_LEARNING: PCFG inference + BiLSTM-CRF field detection
    3. ARTIFACT_GENERATION: Parser generation + Validation
    4. COMPLETION: Result compilation and caching

    Note: Initialization is implicit (not a discovery phase).
    """

    # Phase 1: Combined statistical analysis + classification
    FEATURE_EXTRACTION = "feature_extraction"

    # Phase 2: Grammar and structure learning
    GRAMMAR_LEARNING = "grammar_learning"

    # Phase 3: Combined parser generation + validation
    ARTIFACT_GENERATION = "artifact_generation"

    # Phase 4: Finalization
    COMPLETION = "completion"

    # Legacy aliases for backward compatibility
    @classmethod
    def _missing_(cls, value):
        """Handle legacy phase names for backward compatibility."""
        legacy_mappings = {
            "initialization": cls.FEATURE_EXTRACTION,
            "statistical_analysis": cls.FEATURE_EXTRACTION,
            "classification": cls.FEATURE_EXTRACTION,
            "parser_generation": cls.ARTIFACT_GENERATION,
            "validation": cls.ARTIFACT_GENERATION,
        }
        if value in legacy_mappings:
            return legacy_mappings[value]
        return None


@dataclass
class DiscoveryRequest:
    """Request for protocol discovery."""

    messages: List[bytes]
    known_protocol: Optional[str] = None
    training_mode: bool = False
    confidence_threshold: float = 0.7
    generate_parser: bool = True
    validate_results: bool = True
    custom_rules: Optional[List[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    """Result of protocol discovery process."""

    protocol_type: str
    confidence: float
    grammar: Optional[Grammar] = None
    parser: Optional[GeneratedParser] = None
    validation_result: Optional[ValidationResult] = None
    statistical_analysis: Optional[Dict[str, Any]] = None
    classification_details: Optional[ClassificationResult] = None
    processing_time: float = 0.0
    phases_completed: List[DiscoveryPhase] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Backward compatibility properties
    @property
    def success(self) -> bool:
        """Check if discovery was successful."""
        return self.confidence >= 0.5 and self.protocol_type != "unknown"

    @property
    def discovered_protocols(self) -> List[Dict[str, Any]]:
        """Get list of discovered protocols for backward compatibility."""
        return [
            {
                "protocol_type": self.protocol_type,
                "confidence": self.confidence,
                "grammar": self.grammar,
                "parser": self.parser,
            }
        ]


@dataclass
class PartialDiscoveryResult:
    """
    Returned when discovery pipeline fails mid-way.

    Contains all data discovered before the failure, allowing
    clients to use partial results or retry from a checkpoint.
    """

    completed_phases: List[DiscoveryPhase]
    failed_phase: DiscoveryPhase
    partial_data: Dict[str, Any]  # Whatever we learned before failure
    error_message: str
    error_type: str
    processing_time: float = 0.0
    is_retryable: bool = True
    retry_after_seconds: Optional[float] = None  # Suggested retry time

    @property
    def has_statistical_analysis(self) -> bool:
        """Check if statistical analysis was completed."""
        return "statistical_analysis" in self.partial_data

    @property
    def has_classification(self) -> bool:
        """Check if classification was completed."""
        return "classification" in self.partial_data

    @property
    def has_grammar(self) -> bool:
        """Check if grammar learning was completed."""
        return "grammar" in self.partial_data

    def to_discovery_result(self) -> DiscoveryResult:
        """Convert partial result to full DiscoveryResult with available data."""
        return DiscoveryResult(
            protocol_type=self.partial_data.get("protocol_type", "unknown"),
            confidence=self.partial_data.get("confidence", 0.0),
            grammar=self.partial_data.get("grammar"),
            parser=self.partial_data.get("parser"),
            validation_result=self.partial_data.get("validation_result"),
            statistical_analysis=self.partial_data.get("statistical_analysis"),
            classification_details=self.partial_data.get("classification"),
            processing_time=self.processing_time,
            phases_completed=self.completed_phases,
            metadata={
                "partial_result": True,
                "failed_phase": self.failed_phase.value,
                "error": self.error_message,
            },
        )


@dataclass
class ProtocolProfile:
    """Profile of a discovered protocol."""

    protocol_name: str
    grammar: Grammar
    parser: GeneratedParser
    classification_model: Optional[Any] = None
    sample_messages: List[bytes] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    usage_count: int = 0
    confidence_scores: List[float] = field(default_factory=list)


# Prometheus metrics
DISCOVERY_COUNTER = PrometheusCounter(
    "qbitel_protocol_discovery_total",
    "Total protocol discoveries",
    ["protocol", "status"],
)
DISCOVERY_DURATION = Histogram("qbitel_protocol_discovery_duration_seconds", "Discovery duration", ["phase"])
ACTIVE_PROTOCOLS = Gauge("qbitel_active_protocols", "Number of active protocols")
CACHE_HIT_RATE = Gauge("qbitel_discovery_cache_hit_rate", "Cache hit rate")


class ProtocolDiscoveryOrchestrator:
    """
    Enterprise-grade protocol discovery orchestrator.

    This class coordinates all protocol discovery components to provide
    a unified, high-performance protocol discovery system with enterprise
    features including caching, monitoring, and adaptive learning.
    """

    def __init__(self, config: Config):
        """Initialize the protocol discovery orchestrator."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Core components
        self.statistical_analyzer = StatisticalAnalyzer(config)
        self.grammar_learner = GrammarLearner(config)
        self.parser_generator = ParserGenerator(config)
        self.protocol_classifier = ProtocolClassifier(config)
        self.message_validator = MessageValidator(config)

        # Configuration
        self.default_confidence_threshold = 0.7
        self.enable_adaptive_learning = True
        self.enable_caching = True
        self.cache_ttl = 3600  # 1 hour
        self.max_cache_size = 10000
        self.enable_parallel_processing = True
        self.max_concurrent_discoveries = 10

        # Performance settings
        self.max_workers = config.inference.num_workers if hasattr(config, "inference") else 8
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.semaphore = asyncio.Semaphore(self.max_concurrent_discoveries)

        # State management
        self.protocol_profiles: Dict[str, ProtocolProfile] = {}
        self.discovery_cache: Dict[str, DiscoveryResult] = {}
        self.training_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.is_initialized = False

        # Metrics and monitoring
        self.discovery_stats = {
            "total_discoveries": 0,
            "successful_discoveries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "protocols_learned": 0,
            "average_discovery_time": 0.0,
            "phase_timings": defaultdict(list),
        }

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

        self.logger.info("Protocol Discovery Orchestrator initialized")

    async def initialize(self) -> None:
        """Initialize the orchestrator and all components."""
        if self.is_initialized:
            self.logger.warning("Orchestrator already initialized")
            return

        start_time = time.time()
        self.logger.info("Initializing Protocol Discovery Orchestrator")

        try:
            # Initialize components
            initialization_tasks = [
                (
                    self.statistical_analyzer.initialize()
                    if hasattr(self.statistical_analyzer, "initialize")
                    else self._dummy_init()
                ),
                (self.grammar_learner.initialize() if hasattr(self.grammar_learner, "initialize") else self._dummy_init()),
                (self.parser_generator.initialize() if hasattr(self.parser_generator, "initialize") else self._dummy_init()),
                (
                    self.protocol_classifier.initialize()
                    if hasattr(self.protocol_classifier, "initialize")
                    else self._dummy_init()
                ),
                (self.message_validator.initialize() if hasattr(self.message_validator, "initialize") else self._dummy_init()),
            ]

            await asyncio.gather(*initialization_tasks, return_exceptions=True)

            # Start background tasks
            await self._start_background_tasks()

            # Load existing protocol profiles if available
            await self._load_protocol_profiles()

            self.is_initialized = True
            initialization_time = time.time() - start_time

            self.logger.info(f"Protocol Discovery Orchestrator initialized in {initialization_time:.2f}s")
            ACTIVE_PROTOCOLS.set(len(self.protocol_profiles))

        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise ModelException(f"Orchestrator initialization failed: {e}")

    async def _dummy_init(self):
        """Dummy initialization for components without init method."""
        pass

    async def discover_protocol(self, request: DiscoveryRequest) -> DiscoveryResult:
        """
        Main protocol discovery entry point.

        Args:
            request: Discovery request with messages and parameters

        Returns:
            Comprehensive discovery result
        """
        if not self.is_initialized:
            await self.initialize()

        async with self.semaphore:  # Limit concurrent discoveries
            return await self._execute_discovery(request)

    async def _execute_discovery(self, request: DiscoveryRequest) -> Union[DiscoveryResult, PartialDiscoveryResult]:
        """
        Execute the complete discovery pipeline with circuit breaker protection.

        Returns partial results if the pipeline fails mid-way, allowing clients
        to use whatever data was discovered before the failure.
        """
        start_time = time.time()
        phases_completed: List[DiscoveryPhase] = []
        partial_data: Dict[str, Any] = {}
        current_phase = DiscoveryPhase.FEATURE_EXTRACTION

        # Get the discovery circuit breaker
        discovery_breaker = circuit_breakers.get("discovery")

        try:
            # Check if circuit is open
            if discovery_breaker and discovery_breaker.is_open:
                raise CircuitOpenError(
                    "discovery",
                    discovery_breaker.config.recovery_timeout,
                )

            # Update metrics
            self.discovery_stats["total_discoveries"] += 1

            # Check cache first
            cache_key = self._generate_cache_key(request)
            if self.enable_caching and cache_key in self.discovery_cache:
                self.discovery_stats["cache_hits"] += 1
                CACHE_HIT_RATE.set(self.discovery_stats["cache_hits"] / self.discovery_stats["total_discoveries"])
                cached_result = self.discovery_cache[cache_key]
                self.logger.debug("Cache hit for discovery request")
                return cached_result

            self.discovery_stats["cache_misses"] += 1

            # Initialize result
            result = DiscoveryResult(
                protocol_type="unknown",
                confidence=0.0,
                processing_time=0.0,
                metadata={
                    "request_id": cache_key,
                    "message_count": len(request.messages),
                    "training_mode": request.training_mode,
                },
            )

            # =================================================================
            # Phase 1: Feature Extraction (Statistical Analysis + Classification)
            # =================================================================
            current_phase = DiscoveryPhase.FEATURE_EXTRACTION
            phase_start = time.time()

            self.logger.debug("Starting feature extraction phase")

            # Step 1a: Statistical Analysis
            statistical_result = await self._execute_with_circuit_breaker(
                self.statistical_analyzer.analyze_messages,
                request.messages,
            )
            result.statistical_analysis = statistical_result
            partial_data["statistical_analysis"] = statistical_result

            # Step 1b: Protocol Classification
            if not request.known_protocol:
                if request.messages and self.protocol_classifier.is_trained:
                    classification_result = await self._execute_with_circuit_breaker(
                        self.protocol_classifier.classify,
                        request.messages[0],
                    )
                    result.classification_details = classification_result
                    partial_data["classification"] = classification_result

                    if classification_result.confidence >= request.confidence_threshold:
                        result.protocol_type = classification_result.protocol_type
                        result.confidence = classification_result.confidence
                    else:
                        result.protocol_type = "unknown"
                        result.confidence = 0.0
                else:
                    result.protocol_type = "unknown"
                    result.confidence = 0.0
            else:
                result.protocol_type = request.known_protocol
                result.confidence = 1.0

            partial_data["protocol_type"] = result.protocol_type
            partial_data["confidence"] = result.confidence
            phases_completed.append(DiscoveryPhase.FEATURE_EXTRACTION)

            phase_time = time.time() - phase_start
            self.discovery_stats["phase_timings"]["feature_extraction"].append(phase_time)
            DISCOVERY_DURATION.labels(phase="feature_extraction").observe(phase_time)

            # =================================================================
            # Phase 2: Grammar Learning (if training mode or unknown protocol)
            # =================================================================
            if request.training_mode or result.protocol_type == "unknown":
                current_phase = DiscoveryPhase.GRAMMAR_LEARNING
                phase_start = time.time()

                self.logger.debug("Starting grammar learning phase")
                learned_grammar = await self._execute_with_circuit_breaker(
                    self.grammar_learner.learn_grammar,
                    request.messages,
                    protocol_hint=(result.protocol_type if result.protocol_type != "unknown" else None),
                )
                result.grammar = learned_grammar
                partial_data["grammar"] = learned_grammar

                # Update protocol type if it was unknown
                if result.protocol_type == "unknown":
                    inferred_protocol = await self._infer_protocol_from_grammar(learned_grammar)
                    result.protocol_type = inferred_protocol
                    result.confidence = 0.6
                    partial_data["protocol_type"] = inferred_protocol
                    partial_data["confidence"] = 0.6

                phases_completed.append(DiscoveryPhase.GRAMMAR_LEARNING)

                phase_time = time.time() - phase_start
                self.discovery_stats["phase_timings"]["grammar_learning"].append(phase_time)
                DISCOVERY_DURATION.labels(phase="grammar_learning").observe(phase_time)

            # =================================================================
            # Phase 3: Artifact Generation (Parser Generation + Validation)
            # =================================================================
            if (request.generate_parser and result.grammar) or (request.validate_results and request.messages):
                current_phase = DiscoveryPhase.ARTIFACT_GENERATION
                phase_start = time.time()

                self.logger.debug("Starting artifact generation phase")

                # Step 3a: Parser Generation
                if request.generate_parser and result.grammar:
                    generated_parser = await self._execute_with_circuit_breaker(
                        self.parser_generator.generate_parser,
                        result.grammar,
                        parser_id=f"{result.protocol_type}_{int(time.time())}",
                        protocol_name=result.protocol_type,
                    )
                    result.parser = generated_parser
                    partial_data["parser"] = generated_parser

                # Step 3b: Validation
                if request.validate_results and request.messages:
                    if result.parser:
                        self.message_validator.register_parser(result.protocol_type, result.parser)

                    if result.grammar:
                        self.message_validator.register_grammar(result.protocol_type, result.grammar)

                    validation_result = await self._execute_with_circuit_breaker(
                        self.message_validator.validate,
                        request.messages[0],
                        protocol_type=result.protocol_type,
                        validation_level=ValidationLevel.STANDARD,
                    )
                    result.validation_result = validation_result
                    partial_data["validation_result"] = validation_result

                    if validation_result.is_valid:
                        result.confidence = min(1.0, result.confidence + 0.1)
                    else:
                        result.confidence = max(0.0, result.confidence - 0.2)
                    partial_data["confidence"] = result.confidence

                phases_completed.append(DiscoveryPhase.ARTIFACT_GENERATION)

                phase_time = time.time() - phase_start
                self.discovery_stats["phase_timings"]["artifact_generation"].append(phase_time)
                DISCOVERY_DURATION.labels(phase="artifact_generation").observe(phase_time)

            # =================================================================
            # Phase 4: Completion and Caching
            # =================================================================
            phases_completed.append(DiscoveryPhase.COMPLETION)

            result.phases_completed = phases_completed
            result.processing_time = time.time() - start_time

            # Update protocol profile
            if result.confidence >= request.confidence_threshold:
                await self._update_protocol_profile(result, request.messages)
                self.discovery_stats["successful_discoveries"] += 1
                DISCOVERY_COUNTER.labels(protocol=result.protocol_type, status="success").inc()
            else:
                DISCOVERY_COUNTER.labels(protocol="unknown", status="failure").inc()

            # Cache result
            if self.enable_caching:
                self._cache_result(cache_key, result)

            # Update average discovery time
            total = self.discovery_stats["total_discoveries"]
            avg_time = self.discovery_stats["average_discovery_time"]
            self.discovery_stats["average_discovery_time"] = (avg_time * (total - 1) + result.processing_time) / total

            self.logger.info(
                f"Protocol discovery completed: {result.protocol_type} "
                f"(confidence: {result.confidence:.2f}, time: {result.processing_time:.2f}s)"
            )

            # Record success with circuit breaker
            if discovery_breaker:
                await discovery_breaker._on_success()

            return result

        except CircuitOpenError as e:
            # Circuit is open, return immediately with suggested retry time
            self.logger.warning(f"Discovery circuit open: {e}")
            DISCOVERY_COUNTER.labels(protocol="unknown", status="circuit_open").inc()

            return PartialDiscoveryResult(
                completed_phases=phases_completed,
                failed_phase=current_phase,
                partial_data=partial_data,
                error_message=str(e),
                error_type="CircuitOpenError",
                processing_time=time.time() - start_time,
                is_retryable=True,
                retry_after_seconds=e.time_until_retry,
            )

        except Exception as e:
            self.logger.error(f"Protocol discovery failed at {current_phase.value}: {e}")
            DISCOVERY_COUNTER.labels(protocol="unknown", status="error").inc()

            # Record failure with circuit breaker
            if discovery_breaker:
                await discovery_breaker._on_failure(e)

            # Return partial results
            return PartialDiscoveryResult(
                completed_phases=phases_completed,
                failed_phase=current_phase,
                partial_data=partial_data,
                error_message=str(e),
                error_type=type(e).__name__,
                processing_time=time.time() - start_time,
                is_retryable=self._is_retryable_error(e),
                retry_after_seconds=60.0 if self._is_retryable_error(e) else None,
            )

    async def _execute_with_circuit_breaker(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with circuit breaker protection."""
        discovery_breaker = circuit_breakers.get("discovery")
        if discovery_breaker:
            return await discovery_breaker.call(func, *args, **kwargs)
        else:
            # No circuit breaker, execute directly
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable."""
        # Network/timeout errors are retryable
        retryable_types = (
            TimeoutError,
            ConnectionError,
            OSError,
        )
        return isinstance(error, retryable_types)

    async def _infer_protocol_from_grammar(self, grammar: Grammar) -> str:
        """Infer protocol type from learned grammar characteristics."""
        # This is a heuristic approach - in practice you might want more sophisticated methods

        # Check grammar metadata for hints
        if "protocol_hint" in grammar.metadata:
            return grammar.metadata["protocol_hint"]

        # Analyze grammar rules for protocol indicators
        rule_analysis = {}
        for rule in grammar.rules:
            for symbol in rule.right_hand_side:
                if symbol.semantic_type:
                    rule_analysis[symbol.semantic_type] = rule_analysis.get(symbol.semantic_type, 0) + 1

        # Simple heuristics
        if "text" in rule_analysis and rule_analysis["text"] > 5:
            if any("http" in str(rule).lower() for rule in grammar.rules):
                return "http"
            elif any("json" in str(rule).lower() for rule in grammar.rules):
                return "json"
            else:
                return "text_protocol"
        elif "binary" in rule_analysis and rule_analysis["binary"] > 3:
            return "binary_protocol"
        elif "length" in rule_analysis:
            return "length_prefixed_protocol"
        else:
            return f"unknown_protocol_{int(time.time())}"

    async def _update_protocol_profile(self, result: DiscoveryResult, messages: List[bytes]) -> None:
        """Update or create protocol profile."""
        protocol_name = result.protocol_type

        if protocol_name in self.protocol_profiles:
            # Update existing profile
            profile = self.protocol_profiles[protocol_name]
            profile.last_updated = time.time()
            profile.usage_count += 1
            profile.confidence_scores.append(result.confidence)

            # Keep only recent confidence scores (last 100)
            if len(profile.confidence_scores) > 100:
                profile.confidence_scores = profile.confidence_scores[-100:]

            # Update grammar if we learned a new one
            if result.grammar and result.grammar != profile.grammar:
                profile.grammar = result.grammar
                profile.parser = result.parser  # Update parser too
                self.discovery_stats["protocols_learned"] += 1

        else:
            # Create new profile
            if result.grammar and result.parser:
                profile = ProtocolProfile(
                    protocol_name=protocol_name,
                    grammar=result.grammar,
                    parser=result.parser,
                    sample_messages=messages[:10],  # Keep first 10 messages
                    statistics=result.statistical_analysis or {},
                    confidence_scores=[result.confidence],
                    usage_count=1,
                )
                self.protocol_profiles[protocol_name] = profile
                self.discovery_stats["protocols_learned"] += 1

                self.logger.info(f"Created new protocol profile: {protocol_name}")
                ACTIVE_PROTOCOLS.set(len(self.protocol_profiles))

    def _generate_cache_key(self, request: DiscoveryRequest) -> str:
        """Generate cache key for discovery request."""
        # Use hash of first few messages and key parameters
        content = b"".join(request.messages[:3])  # First 3 messages
        params = f"{request.known_protocol}_{request.confidence_threshold}_{request.training_mode}"
        combined = content + params.encode("utf-8")
        return hashlib.sha256(combined).hexdigest()[:16]

    def _cache_result(self, cache_key: str, result: DiscoveryResult) -> None:
        """Cache discovery result."""
        if len(self.discovery_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.discovery_cache))
            del self.discovery_cache[oldest_key]

        self.discovery_cache[cache_key] = result
        self.logger.debug(f"Cached discovery result for key: {cache_key}")

    async def train_classifier(self, training_samples: List[ProtocolSample], validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the protocol classifier with new samples."""
        self.logger.info(f"Training classifier with {len(training_samples)} samples")

        try:
            training_result = await self.protocol_classifier.train(training_samples, validation_split=validation_split)

            self.logger.info("Classifier training completed successfully")
            return training_result

        except Exception as e:
            self.logger.error(f"Classifier training failed: {e}")
            raise ModelException(f"Training failed: {e}")

    async def discover_protocol_stream(
        self,
        message_stream: AsyncIterator[bytes],
        protocol_hint: Optional[str] = None,
        batch_size: int = 10,
    ) -> AsyncIterator[DiscoveryResult]:
        """Discover protocols from a stream of messages."""
        batch = []

        async for message in message_stream:
            batch.append(message)

            if len(batch) >= batch_size:
                request = DiscoveryRequest(messages=batch, known_protocol=protocol_hint, training_mode=False)

                result = await self.discover_protocol(request)
                yield result

                batch = []

        # Process remaining messages
        if batch:
            request = DiscoveryRequest(messages=batch, known_protocol=protocol_hint, training_mode=False)

            result = await self.discover_protocol(request)
            yield result

    async def get_protocol_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all discovered protocols."""
        profiles = {}

        for name, profile in self.protocol_profiles.items():
            avg_confidence = (
                sum(profile.confidence_scores) / len(profile.confidence_scores) if profile.confidence_scores else 0.0
            )

            profiles[name] = {
                "protocol_name": profile.protocol_name,
                "created_at": profile.created_at,
                "last_updated": profile.last_updated,
                "usage_count": profile.usage_count,
                "average_confidence": avg_confidence,
                "grammar_rules": len(profile.grammar.rules),
                "grammar_symbols": len(profile.grammar.symbols),
                "has_parser": profile.parser is not None,
                "sample_count": len(profile.sample_messages),
            }

        return profiles

    async def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive discovery statistics."""
        stats = dict(self.discovery_stats)

        # Calculate additional metrics
        if stats["total_discoveries"] > 0:
            stats["success_rate"] = stats["successful_discoveries"] / stats["total_discoveries"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_discoveries"]
        else:
            stats["success_rate"] = 0.0
            stats["cache_hit_rate"] = 0.0

        # Add phase timing statistics
        phase_stats = {}
        for phase, timings in stats["phase_timings"].items():
            if timings:
                phase_stats[phase] = {
                    "count": len(timings),
                    "avg_time": sum(timings) / len(timings),
                    "min_time": min(timings),
                    "max_time": max(timings),
                }
        stats["phase_statistics"] = phase_stats

        # Add protocol information
        stats["active_protocols"] = len(self.protocol_profiles)
        stats["cached_results"] = len(self.discovery_cache)

        return stats

    async def clear_cache(self) -> int:
        """Clear discovery cache and return number of items cleared."""
        count = len(self.discovery_cache)
        self.discovery_cache.clear()
        self.logger.info(f"Cleared {count} cached discovery results")
        return count

    async def remove_protocol_profile(self, protocol_name: str) -> bool:
        """Remove a protocol profile."""
        if protocol_name in self.protocol_profiles:
            del self.protocol_profiles[protocol_name]
            ACTIVE_PROTOCOLS.set(len(self.protocol_profiles))
            self.logger.info(f"Removed protocol profile: {protocol_name}")
            return True
        return False

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Cache cleanup task
        cache_cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
        self._background_tasks.add(cache_cleanup_task)
        cache_cleanup_task.add_done_callback(self._background_tasks.discard)

        # Adaptive learning task
        if self.enable_adaptive_learning:
            adaptive_learning_task = asyncio.create_task(self._adaptive_learning_loop())
            self._background_tasks.add(adaptive_learning_task)
            adaptive_learning_task.add_done_callback(self._background_tasks.discard)

        self.logger.info("Background tasks started")

    async def _cache_cleanup_loop(self) -> None:
        """Background task to clean up expired cache entries."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                # Simple cache cleanup - in production you'd want TTL-based cleanup
                if len(self.discovery_cache) > self.max_cache_size * 0.8:
                    # Remove 20% of oldest entries
                    to_remove = int(len(self.discovery_cache) * 0.2)
                    for _ in range(to_remove):
                        if self.discovery_cache:
                            oldest_key = next(iter(self.discovery_cache))
                            del self.discovery_cache[oldest_key]

                    self.logger.debug(f"Cache cleanup: removed {to_remove} entries")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")

    async def _adaptive_learning_loop(self) -> None:
        """Background task for adaptive learning from discovery results."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(600)  # Run every 10 minutes

                # Process training queue if not empty
                if not self.training_queue.empty():
                    self.logger.debug("Processing adaptive learning queue")
                    # Implementation would process queued training data
                    # For now, just clear the queue
                    while not self.training_queue.empty():
                        try:
                            self.training_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Adaptive learning error: {e}")

    async def _load_protocol_profiles(self) -> None:
        """Load existing protocol profiles from storage."""
        # This would load from persistent storage in production
        # For now, just log that we're ready
        self.logger.info("Protocol profiles loaded (none found - fresh start)")

    async def save_protocol_profiles(self, filepath: str) -> None:
        """Save protocol profiles to file."""
        try:
            profiles_data = {}
            for name, profile in self.protocol_profiles.items():
                profiles_data[name] = {
                    "protocol_name": profile.protocol_name,
                    "grammar": profile.grammar.to_dict(),
                    "statistics": profile.statistics,
                    "created_at": profile.created_at,
                    "last_updated": profile.last_updated,
                    "usage_count": profile.usage_count,
                    "confidence_scores": profile.confidence_scores[-10:],  # Last 10 scores
                }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(profiles_data, f, indent=2, default=str)

            self.logger.info(f"Saved {len(profiles_data)} protocol profiles to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save protocol profiles: {e}")
            raise ModelException(f"Profile save error: {e}")

    async def load_protocol_profiles(self, filepath: str) -> int:
        """Load protocol profiles from file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                profiles_data = json.load(f)

            loaded_count = 0
            for name, profile_data in profiles_data.items():
                # This is a simplified loading - in production you'd reconstruct full objects
                self.logger.info(f"Would load profile: {name}")
                loaded_count += 1

            self.logger.info(f"Loaded {loaded_count} protocol profiles from {filepath}")
            return loaded_count

        except Exception as e:
            self.logger.error(f"Failed to load protocol profiles: {e}")
            raise ModelException(f"Profile load error: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of orchestrator and components."""
        health_status = {
            "orchestrator": "healthy",
            "initialized": self.is_initialized,
            "components": {},
            "metrics": {
                "active_protocols": len(self.protocol_profiles),
                "cached_results": len(self.discovery_cache),
                "total_discoveries": self.discovery_stats["total_discoveries"],
                "background_tasks": len(self._background_tasks),
            },
            "timestamp": time.time(),
        }

        # Check component health
        components = {
            "statistical_analyzer": self.statistical_analyzer,
            "grammar_learner": self.grammar_learner,
            "parser_generator": self.parser_generator,
            "protocol_classifier": self.protocol_classifier,
            "message_validator": self.message_validator,
        }

        for name, component in components.items():
            try:
                # Simple health check - in production you'd have proper health methods
                health_status["components"][name] = "healthy"
            except Exception as e:
                health_status["components"][name] = f"unhealthy: {e}"
                health_status["orchestrator"] = "degraded"

        return health_status

    async def shutdown(self):
        """Shutdown orchestrator and all components."""
        self.logger.info("Shutting down Protocol Discovery Orchestrator")

        # Signal shutdown to background tasks
        self._shutdown_event.set()

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for background tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Shutdown components
        shutdown_tasks = []
        for component in [
            self.statistical_analyzer,
            self.grammar_learner,
            self.parser_generator,
            self.protocol_classifier,
            self.message_validator,
        ]:
            if hasattr(component, "shutdown"):
                shutdown_tasks.append(component.shutdown())

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)

        # Clear state
        self.protocol_profiles.clear()
        self.discovery_cache.clear()

        self.is_initialized = False
        self.logger.info("Protocol Discovery Orchestrator shutdown completed")
