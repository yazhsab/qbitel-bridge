"""
QBITEL Gateway - Prompt Registry

Centralized prompt management with versioning and A/B testing support.
Enables prompt iteration without code deployments.

Features:
- Prompt versioning and rollback
- A/B testing for prompt optimization
- Template variables and formatting
- Performance tracking per prompt version
- Prompt validation
"""

import asyncio
import hashlib
import logging
import time
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from enum import Enum
import json

from prometheus_client import Counter, Histogram

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Metrics
# =============================================================================

PROMPT_USAGE = Counter(
    "qbitel_gateway_prompt_usage_total",
    "Prompt usage by version",
    ["prompt_name", "version"],
)
PROMPT_PERFORMANCE = Histogram(
    "qbitel_gateway_prompt_performance_score",
    "Prompt performance scores",
    ["prompt_name", "version"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# =============================================================================
# Data Classes
# =============================================================================

class PromptStatus(str, Enum):
    """Prompt version status."""
    DRAFT = "draft"
    ACTIVE = "active"
    TESTING = "testing"  # A/B test
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class PromptVersion:
    """A specific version of a prompt."""

    version_id: str
    version_number: int
    prompt_name: str

    # Content
    system_prompt: Optional[str] = None
    user_prompt_template: str = ""

    # Metadata
    status: PromptStatus = PromptStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    created_by: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Performance tracking
    usage_count: int = 0
    total_tokens: int = 0
    average_latency_ms: float = 0.0
    success_rate: float = 1.0
    average_quality_score: float = 0.0

    # A/B testing
    traffic_percentage: float = 0.0  # 0-100

    # Template variables
    variables: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Extract variables from template
        if not self.variables:
            self.variables = self._extract_variables()

    def _extract_variables(self) -> List[str]:
        """Extract variable names from template."""
        pattern = r'\{(\w+)\}'
        all_vars = set()
        if self.system_prompt:
            all_vars.update(re.findall(pattern, self.system_prompt))
        if self.user_prompt_template:
            all_vars.update(re.findall(pattern, self.user_prompt_template))
        return list(all_vars)

    def render(self, variables: Dict[str, Any]) -> tuple[Optional[str], str]:
        """
        Render the prompt with variables.

        Returns (system_prompt, user_prompt) tuple.
        """
        system = None
        if self.system_prompt:
            system = self.system_prompt.format(**variables)

        user = self.user_prompt_template.format(**variables)

        return system, user

    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """Validate that all required variables are provided."""
        missing = [v for v in self.variables if v not in variables]
        return missing

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["status"] = self.status.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptVersion":
        """Create from dictionary."""
        if "status" in data and isinstance(data["status"], str):
            data["status"] = PromptStatus(data["status"])
        return cls(**data)


@dataclass
class PromptTemplate:
    """A prompt template with multiple versions."""

    name: str
    domain: str
    description: str = ""

    # Versions
    versions: Dict[str, PromptVersion] = field(default_factory=dict)
    active_version_id: Optional[str] = None

    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    created_by: Optional[str] = None

    # A/B testing
    ab_test_active: bool = False
    ab_test_versions: List[str] = field(default_factory=list)

    def get_active_version(self) -> Optional[PromptVersion]:
        """Get the active version."""
        if self.active_version_id and self.active_version_id in self.versions:
            return self.versions[self.active_version_id]
        return None

    def get_version(self, version_id: str) -> Optional[PromptVersion]:
        """Get a specific version."""
        return self.versions.get(version_id)

    def select_version_for_request(self, request_id: str) -> PromptVersion:
        """
        Select a version for a request (handles A/B testing).

        Uses consistent hashing based on request_id for deterministic selection.
        """
        if not self.ab_test_active or not self.ab_test_versions:
            return self.get_active_version()

        # Consistent hash for deterministic selection
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest()[:8], 16)
        percentage = (hash_value % 10000) / 100  # 0-100

        cumulative = 0.0
        for version_id in self.ab_test_versions:
            version = self.versions.get(version_id)
            if version:
                cumulative += version.traffic_percentage
                if percentage < cumulative:
                    return version

        # Fallback to active version
        return self.get_active_version()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "versions": {k: v.to_dict() for k, v in self.versions.items()},
            "active_version_id": self.active_version_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
            "ab_test_active": self.ab_test_active,
            "ab_test_versions": self.ab_test_versions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary."""
        versions = {}
        if "versions" in data:
            for k, v in data["versions"].items():
                versions[k] = PromptVersion.from_dict(v)
            data["versions"] = versions
        return cls(**data)


# =============================================================================
# Prompt Registry
# =============================================================================

class PromptRegistry:
    """
    Centralized registry for prompt management.

    Provides versioning, A/B testing, and performance tracking for prompts.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url

        # In-memory storage
        self._templates: Dict[str, PromptTemplate] = {}

        # Redis client
        self._redis: Optional[Any] = None

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Performance tracking callbacks
        self._performance_callbacks: List[Callable] = []

    async def initialize(self):
        """Initialize the registry."""
        if self.redis_url and redis is not None:
            try:
                self._redis = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                await self._redis.ping()
                await self._load_from_redis()
                logger.info("Prompt registry connected to Redis")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self._redis = None

        # Load default prompts
        await self._load_default_prompts()

        logger.info("Prompt registry initialized")

    async def shutdown(self):
        """Shutdown the registry."""
        if self._redis:
            await self._redis.close()
        logger.info("Prompt registry shutdown")

    async def register_template(self, template: PromptTemplate) -> PromptTemplate:
        """Register a new prompt template."""
        async with self._lock:
            self._templates[template.name] = template

        if self._redis:
            await self._save_to_redis(template)

        logger.info(f"Registered prompt template: {template.name}")
        return template

    async def create_version(
        self,
        prompt_name: str,
        system_prompt: Optional[str],
        user_prompt_template: str,
        description: str = "",
        created_by: Optional[str] = None,
        make_active: bool = False,
    ) -> PromptVersion:
        """Create a new version of a prompt."""
        async with self._lock:
            template = self._templates.get(prompt_name)
            if not template:
                raise ValueError(f"Prompt template not found: {prompt_name}")

            # Determine version number
            version_number = len(template.versions) + 1

            # Create version
            version = PromptVersion(
                version_id=f"v{version_number}_{int(time.time())}",
                version_number=version_number,
                prompt_name=prompt_name,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                description=description,
                created_by=created_by,
                status=PromptStatus.DRAFT,
            )

            # Add to template
            template.versions[version.version_id] = version
            template.updated_at = time.time()

            if make_active:
                template.active_version_id = version.version_id
                version.status = PromptStatus.ACTIVE

        if self._redis:
            await self._save_to_redis(template)

        logger.info(f"Created prompt version: {prompt_name}/{version.version_id}")
        return version

    async def activate_version(self, prompt_name: str, version_id: str):
        """Activate a specific version."""
        async with self._lock:
            template = self._templates.get(prompt_name)
            if not template:
                raise ValueError(f"Prompt template not found: {prompt_name}")

            version = template.versions.get(version_id)
            if not version:
                raise ValueError(f"Version not found: {version_id}")

            # Deactivate current active
            if template.active_version_id:
                old_version = template.versions.get(template.active_version_id)
                if old_version:
                    old_version.status = PromptStatus.DEPRECATED

            # Activate new version
            version.status = PromptStatus.ACTIVE
            template.active_version_id = version_id
            template.updated_at = time.time()

        if self._redis:
            await self._save_to_redis(template)

        logger.info(f"Activated prompt version: {prompt_name}/{version_id}")

    async def start_ab_test(
        self,
        prompt_name: str,
        version_ids: List[str],
        traffic_split: List[float],
    ):
        """
        Start an A/B test between versions.

        Args:
            prompt_name: Prompt template name
            version_ids: List of version IDs to test
            traffic_split: Traffic percentage for each version (must sum to 100)
        """
        if len(version_ids) != len(traffic_split):
            raise ValueError("version_ids and traffic_split must have same length")

        if abs(sum(traffic_split) - 100) > 0.01:
            raise ValueError("traffic_split must sum to 100")

        async with self._lock:
            template = self._templates.get(prompt_name)
            if not template:
                raise ValueError(f"Prompt template not found: {prompt_name}")

            # Validate versions
            for vid in version_ids:
                if vid not in template.versions:
                    raise ValueError(f"Version not found: {vid}")

            # Set up A/B test
            template.ab_test_active = True
            template.ab_test_versions = version_ids

            for vid, split in zip(version_ids, traffic_split):
                version = template.versions[vid]
                version.traffic_percentage = split
                version.status = PromptStatus.TESTING

            template.updated_at = time.time()

        if self._redis:
            await self._save_to_redis(template)

        logger.info(f"Started A/B test for {prompt_name}: {version_ids}")

    async def stop_ab_test(self, prompt_name: str, winner_version_id: Optional[str] = None):
        """Stop an A/B test and optionally select a winner."""
        async with self._lock:
            template = self._templates.get(prompt_name)
            if not template:
                raise ValueError(f"Prompt template not found: {prompt_name}")

            template.ab_test_active = False

            # Reset traffic percentages
            for vid in template.ab_test_versions:
                version = template.versions.get(vid)
                if version:
                    version.traffic_percentage = 0
                    version.status = PromptStatus.DEPRECATED

            template.ab_test_versions = []

            # Activate winner
            if winner_version_id:
                winner = template.versions.get(winner_version_id)
                if winner:
                    winner.status = PromptStatus.ACTIVE
                    template.active_version_id = winner_version_id

            template.updated_at = time.time()

        if self._redis:
            await self._save_to_redis(template)

        logger.info(f"Stopped A/B test for {prompt_name}, winner: {winner_version_id}")

    async def get_prompt(
        self,
        prompt_name: str,
        variables: Dict[str, Any],
        request_id: str,
    ) -> tuple[Optional[str], str, PromptVersion]:
        """
        Get a rendered prompt for a request.

        Returns (system_prompt, user_prompt, version) tuple.
        """
        async with self._lock:
            template = self._templates.get(prompt_name)
            if not template:
                raise ValueError(f"Prompt template not found: {prompt_name}")

            # Select version (handles A/B testing)
            version = template.select_version_for_request(request_id)
            if not version:
                raise ValueError(f"No active version for: {prompt_name}")

            # Validate variables
            missing = version.validate_variables(variables)
            if missing:
                raise ValueError(f"Missing variables: {missing}")

            # Render
            system, user = version.render(variables)

            # Track usage
            version.usage_count += 1

        PROMPT_USAGE.labels(
            prompt_name=prompt_name,
            version=version.version_id,
        ).inc()

        return system, user, version

    async def record_performance(
        self,
        prompt_name: str,
        version_id: str,
        tokens_used: int,
        latency_ms: float,
        success: bool,
        quality_score: Optional[float] = None,
    ):
        """Record performance metrics for a prompt version."""
        async with self._lock:
            template = self._templates.get(prompt_name)
            if not template:
                return

            version = template.versions.get(version_id)
            if not version:
                return

            # Update metrics
            version.total_tokens += tokens_used

            # Running average for latency
            n = version.usage_count
            if n > 0:
                version.average_latency_ms = (
                    (version.average_latency_ms * (n - 1) + latency_ms) / n
                )

            # Success rate
            if success:
                version.success_rate = (
                    (version.success_rate * (n - 1) + 1.0) / n
                )
            else:
                version.success_rate = (
                    (version.success_rate * (n - 1)) / n
                )

            # Quality score
            if quality_score is not None:
                version.average_quality_score = (
                    (version.average_quality_score * (n - 1) + quality_score) / n
                )

                PROMPT_PERFORMANCE.labels(
                    prompt_name=prompt_name,
                    version=version_id,
                ).observe(quality_score)

    async def get_template(self, prompt_name: str) -> Optional[PromptTemplate]:
        """Get a prompt template."""
        async with self._lock:
            return self._templates.get(prompt_name)

    async def list_templates(self, domain: Optional[str] = None) -> List[PromptTemplate]:
        """List all templates, optionally filtered by domain."""
        async with self._lock:
            templates = list(self._templates.values())

        if domain:
            templates = [t for t in templates if t.domain == domain]

        return templates

    async def _load_from_redis(self):
        """Load templates from Redis."""
        if not self._redis:
            return

        try:
            keys = await self._redis.keys("qbitel:prompts:*")
            for key in keys:
                data = await self._redis.get(key)
                if data:
                    template = PromptTemplate.from_dict(json.loads(data))
                    self._templates[template.name] = template

            logger.info(f"Loaded {len(keys)} prompt templates from Redis")
        except Exception as e:
            logger.error(f"Failed to load prompts from Redis: {e}")

    async def _save_to_redis(self, template: PromptTemplate):
        """Save template to Redis."""
        if not self._redis:
            return

        try:
            key = f"qbitel:prompts:{template.name}"
            await self._redis.set(key, json.dumps(template.to_dict()))
        except Exception as e:
            logger.error(f"Failed to save prompt to Redis: {e}")

    async def _load_default_prompts(self):
        """Load default prompt templates."""
        defaults = [
            PromptTemplate(
                name="protocol_analysis",
                domain="protocol_copilot",
                description="Analyze network protocols",
            ),
            PromptTemplate(
                name="threat_assessment",
                domain="security_orchestrator",
                description="Assess security threats",
            ),
            PromptTemplate(
                name="legacy_explanation",
                domain="legacy_whisperer",
                description="Explain legacy system behavior",
            ),
            PromptTemplate(
                name="compliance_check",
                domain="compliance_reporter",
                description="Check compliance requirements",
            ),
            PromptTemplate(
                name="protocol_translation",
                domain="translation_studio",
                description="Translate protocols",
            ),
        ]

        for template in defaults:
            if template.name not in self._templates:
                # Create default version
                version = PromptVersion(
                    version_id="v1_default",
                    version_number=1,
                    prompt_name=template.name,
                    system_prompt="You are an expert assistant for QBITEL security platform.",
                    user_prompt_template="{query}",
                    status=PromptStatus.ACTIVE,
                    description="Default version",
                )
                template.versions[version.version_id] = version
                template.active_version_id = version.version_id
                self._templates[template.name] = template

        logger.debug(f"Loaded {len(defaults)} default prompt templates")
