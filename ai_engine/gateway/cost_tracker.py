"""
QBITEL Gateway - Cost Tracker

Comprehensive cost tracking and budget management for LLM usage.
Provides real-time cost monitoring, budget alerts, and detailed analytics.

Features:
- Per-request cost tracking
- Budget limits with alerts
- Cost breakdown by domain, model, user
- Historical cost analysis
- Cost optimization recommendations
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import defaultdict

from prometheus_client import Counter, Gauge, Histogram

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Metrics
# =============================================================================

COST_TOTAL = Counter(
    "qbitel_gateway_cost_dollars_total",
    "Total cost in dollars",
    ["provider", "model", "domain"],
)
TOKENS_TOTAL = Counter(
    "qbitel_gateway_tokens_total",
    "Total tokens used",
    ["provider", "model", "direction"],
)
BUDGET_REMAINING = Gauge(
    "qbitel_gateway_budget_remaining_dollars",
    "Remaining budget in dollars",
    ["budget_type"],
)
COST_PER_REQUEST = Histogram(
    "qbitel_gateway_cost_per_request_dollars",
    "Cost per request",
    ["domain"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)


# =============================================================================
# Data Classes
# =============================================================================


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    model_name: str
    provider: str
    input_cost_per_1k: float  # Cost per 1K input tokens
    output_cost_per_1k: float  # Cost per 1K output tokens
    context_window: int = 128000
    is_local: bool = False

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost for a request."""
        if self.is_local:
            return 0.0
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost


# Current pricing (January 2026)
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": ModelPricing("gpt-4o", "openai", 0.0025, 0.01, 128000),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", "openai", 0.00015, 0.0006, 128000),
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", "openai", 0.01, 0.03, 128000),
    "o1": ModelPricing("o1", "openai", 0.015, 0.06, 200000),
    "o1-mini": ModelPricing("o1-mini", "openai", 0.003, 0.012, 128000),
    # Anthropic
    "claude-opus-4-5": ModelPricing("claude-opus-4-5", "anthropic", 0.015, 0.075, 200000),
    "claude-sonnet-4-5": ModelPricing("claude-sonnet-4-5", "anthropic", 0.003, 0.015, 200000),
    "claude-3-5-haiku": ModelPricing("claude-3-5-haiku", "anthropic", 0.0008, 0.004, 200000),
    # Local models (zero cost)
    "llama3.2": ModelPricing("llama3.2", "ollama", 0.0, 0.0, 128000, is_local=True),
    "qwen2.5": ModelPricing("qwen2.5", "ollama", 0.0, 0.0, 32000, is_local=True),
    "mistral": ModelPricing("mistral", "ollama", 0.0, 0.0, 32000, is_local=True),
    "codellama": ModelPricing("codellama", "ollama", 0.0, 0.0, 16000, is_local=True),
}


@dataclass
class UsageRecord:
    """Record of a single LLM usage."""

    request_id: str
    timestamp: float
    domain: str
    model: str
    provider: str

    input_tokens: int
    output_tokens: int
    total_tokens: int

    input_cost: float
    output_cost: float
    total_cost: float

    # Optional context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    cached: bool = False

    # Metadata
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BudgetConfig:
    """Budget configuration."""

    daily_limit: float = 100.0  # Daily budget in dollars
    monthly_limit: float = 2000.0  # Monthly budget
    per_request_limit: float = 1.0  # Max cost per request

    # Domain-specific limits
    domain_limits: Dict[str, float] = field(
        default_factory=lambda: {
            "protocol_copilot": 500.0,  # Monthly
            "security_orchestrator": 800.0,
            "legacy_whisperer": 300.0,
            "compliance_reporter": 200.0,
            "translation_studio": 200.0,
        }
    )

    # Alert thresholds (percentage of budget)
    warning_threshold: float = 0.7  # 70%
    critical_threshold: float = 0.9  # 90%

    # Actions when budget exceeded
    block_on_exceed: bool = False  # If True, block requests when budget exceeded
    fallback_to_local: bool = True  # If True, use local models when budget exceeded


@dataclass
class BudgetAlert:
    """Budget alert notification."""

    alert_id: str
    severity: AlertSeverity
    budget_type: str  # "daily", "monthly", "domain"
    domain: Optional[str] = None

    current_usage: float = 0.0
    budget_limit: float = 0.0
    percentage_used: float = 0.0

    message: str = ""
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["severity"] = self.severity.value
        return result


@dataclass
class CostReport:
    """Cost analysis report."""

    period_start: datetime
    period_end: datetime

    total_cost: float = 0.0
    total_requests: int = 0
    total_tokens: int = 0

    # Breakdown
    cost_by_domain: Dict[str, float] = field(default_factory=dict)
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    cost_by_provider: Dict[str, float] = field(default_factory=dict)

    # Token breakdown
    input_tokens: int = 0
    output_tokens: int = 0

    # Efficiency metrics
    cache_hit_rate: float = 0.0
    cost_savings_from_cache: float = 0.0
    average_cost_per_request: float = 0.0

    # Top consumers
    top_domains: List[Dict[str, Any]] = field(default_factory=list)
    top_models: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Cost Tracker
# =============================================================================


class CostTracker:
    """
    Tracks LLM usage costs and enforces budgets.

    Provides real-time cost monitoring, budget alerts, and detailed analytics.
    """

    def __init__(
        self,
        config: Optional[BudgetConfig] = None,
        redis_url: Optional[str] = None,
    ):
        self.config = config or BudgetConfig()
        self.redis_url = redis_url

        # In-memory tracking (also persisted to Redis if available)
        self._usage_records: List[UsageRecord] = []
        self._daily_usage: Dict[str, float] = defaultdict(float)
        self._monthly_usage: Dict[str, float] = defaultdict(float)
        self._domain_usage: Dict[str, float] = defaultdict(float)

        # Alert callbacks
        self._alert_callbacks: List[Callable[[BudgetAlert], None]] = []
        self._alerts: List[BudgetAlert] = []

        # Redis client
        self._redis: Optional[Any] = None

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Current date tracking for daily/monthly reset
        self._current_day = datetime.now(timezone.utc).date()
        self._current_month = self._current_day.replace(day=1)

    async def initialize(self):
        """Initialize cost tracker."""
        if self.redis_url and redis is not None:
            try:
                self._redis = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                await self._redis.ping()

                # Load existing usage from Redis
                await self._load_usage_from_redis()

                logger.info("Cost tracker connected to Redis")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self._redis = None

        logger.info("Cost tracker initialized")

    async def shutdown(self):
        """Shutdown cost tracker."""
        if self._redis:
            await self._redis.close()
        logger.info("Cost tracker shutdown")

    def get_pricing(self, model: str) -> ModelPricing:
        """Get pricing for a model."""
        if model in MODEL_PRICING:
            return MODEL_PRICING[model]

        # Try to find by partial match
        for name, pricing in MODEL_PRICING.items():
            if name in model or model in name:
                return pricing

        # Default pricing (assume expensive to be safe)
        logger.warning(f"Unknown model pricing: {model}, using default")
        return ModelPricing(model, "unknown", 0.01, 0.03, 128000)

    async def record_usage(
        self,
        request_id: str,
        domain: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float = 0.0,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        cached: bool = False,
        success: bool = True,
        error: Optional[str] = None,
    ) -> UsageRecord:
        """
        Record a usage event and track costs.

        Returns the usage record with calculated costs.
        """
        await self._check_date_reset()

        # Get pricing
        pricing = self.get_pricing(model)

        # Calculate costs
        input_cost = (input_tokens / 1000) * pricing.input_cost_per_1k
        output_cost = (output_tokens / 1000) * pricing.output_cost_per_1k
        total_cost = input_cost + output_cost

        # If cached, no cost
        if cached:
            total_cost = 0.0
            input_cost = 0.0
            output_cost = 0.0

        # Create record
        record = UsageRecord(
            request_id=request_id,
            timestamp=time.time(),
            domain=domain,
            model=model,
            provider=pricing.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            user_id=user_id,
            session_id=session_id,
            cached=cached,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )

        async with self._lock:
            # Update tracking
            self._usage_records.append(record)
            self._daily_usage["total"] += total_cost
            self._daily_usage[domain] += total_cost
            self._monthly_usage["total"] += total_cost
            self._monthly_usage[domain] += total_cost
            self._domain_usage[domain] += total_cost

            # Keep only last 10000 records in memory
            if len(self._usage_records) > 10000:
                self._usage_records = self._usage_records[-10000:]

        # Update Prometheus metrics
        COST_TOTAL.labels(
            provider=pricing.provider,
            model=model,
            domain=domain,
        ).inc(total_cost)

        TOKENS_TOTAL.labels(
            provider=pricing.provider,
            model=model,
            direction="input",
        ).inc(input_tokens)

        TOKENS_TOTAL.labels(
            provider=pricing.provider,
            model=model,
            direction="output",
        ).inc(output_tokens)

        COST_PER_REQUEST.labels(domain=domain).observe(total_cost)

        # Persist to Redis
        if self._redis:
            await self._persist_to_redis(record)

        # Check budgets and send alerts
        await self._check_budgets(domain)

        return record

    async def check_budget(
        self,
        domain: str,
        estimated_cost: float = 0.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a request can proceed based on budget constraints.

        Returns (allowed, fallback_model) tuple.
        - allowed: True if request can proceed
        - fallback_model: If not None, suggests using this local model instead
        """
        await self._check_date_reset()

        async with self._lock:
            daily_total = self._daily_usage["total"]
            monthly_total = self._monthly_usage["total"]
            domain_total = self._monthly_usage[domain]

        # Check daily limit
        if daily_total + estimated_cost > self.config.daily_limit:
            if self.config.block_on_exceed:
                return False, None
            if self.config.fallback_to_local:
                return True, "llama3.2"
            return True, None

        # Check monthly limit
        if monthly_total + estimated_cost > self.config.monthly_limit:
            if self.config.block_on_exceed:
                return False, None
            if self.config.fallback_to_local:
                return True, "llama3.2"
            return True, None

        # Check domain limit
        domain_limit = self.config.domain_limits.get(domain, float("inf"))
        if domain_total + estimated_cost > domain_limit:
            if self.config.block_on_exceed:
                return False, None
            if self.config.fallback_to_local:
                return True, "llama3.2"
            return True, None

        # Check per-request limit
        if estimated_cost > self.config.per_request_limit:
            return False, None

        return True, None

    async def get_usage_summary(self) -> Dict[str, Any]:
        """Get current usage summary."""
        await self._check_date_reset()

        async with self._lock:
            return {
                "daily": {
                    "total": self._daily_usage["total"],
                    "limit": self.config.daily_limit,
                    "remaining": max(0, self.config.daily_limit - self._daily_usage["total"]),
                    "percentage": (
                        (self._daily_usage["total"] / self.config.daily_limit * 100) if self.config.daily_limit > 0 else 0
                    ),
                    "by_domain": dict(self._daily_usage),
                },
                "monthly": {
                    "total": self._monthly_usage["total"],
                    "limit": self.config.monthly_limit,
                    "remaining": max(0, self.config.monthly_limit - self._monthly_usage["total"]),
                    "percentage": (
                        (self._monthly_usage["total"] / self.config.monthly_limit * 100)
                        if self.config.monthly_limit > 0
                        else 0
                    ),
                    "by_domain": dict(self._monthly_usage),
                },
                "alerts": [a.to_dict() for a in self._alerts if not a.acknowledged],
            }

    async def generate_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> CostReport:
        """Generate a detailed cost report."""
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        start_ts = start_date.timestamp()
        end_ts = end_date.timestamp()

        # Filter records
        async with self._lock:
            records = [r for r in self._usage_records if start_ts <= r.timestamp <= end_ts]

        # Calculate metrics
        report = CostReport(
            period_start=start_date,
            period_end=end_date,
            total_requests=len(records),
        )

        for record in records:
            report.total_cost += record.total_cost
            report.total_tokens += record.total_tokens
            report.input_tokens += record.input_tokens
            report.output_tokens += record.output_tokens

            # By domain
            report.cost_by_domain[record.domain] = report.cost_by_domain.get(record.domain, 0) + record.total_cost

            # By model
            report.cost_by_model[record.model] = report.cost_by_model.get(record.model, 0) + record.total_cost

            # By provider
            report.cost_by_provider[record.provider] = report.cost_by_provider.get(record.provider, 0) + record.total_cost

        # Calculate averages
        if report.total_requests > 0:
            report.average_cost_per_request = report.total_cost / report.total_requests

            # Cache metrics
            cached_count = sum(1 for r in records if r.cached)
            report.cache_hit_rate = cached_count / report.total_requests

        # Top consumers
        report.top_domains = sorted(
            [{"domain": k, "cost": v} for k, v in report.cost_by_domain.items()],
            key=lambda x: x["cost"],
            reverse=True,
        )[:5]

        report.top_models = sorted(
            [{"model": k, "cost": v} for k, v in report.cost_by_model.items()],
            key=lambda x: x["cost"],
            reverse=True,
        )[:5]

        return report

    def register_alert_callback(self, callback: Callable[[BudgetAlert], None]):
        """Register a callback for budget alerts."""
        self._alert_callbacks.append(callback)

    async def _check_budgets(self, domain: str):
        """Check budget thresholds and send alerts."""
        async with self._lock:
            daily_pct = self._daily_usage["total"] / self.config.daily_limit
            monthly_pct = self._monthly_usage["total"] / self.config.monthly_limit

            domain_limit = self.config.domain_limits.get(domain, float("inf"))
            domain_pct = self._monthly_usage[domain] / domain_limit if domain_limit < float("inf") else 0

        # Update Prometheus
        BUDGET_REMAINING.labels(budget_type="daily").set(max(0, self.config.daily_limit - self._daily_usage["total"]))
        BUDGET_REMAINING.labels(budget_type="monthly").set(max(0, self.config.monthly_limit - self._monthly_usage["total"]))

        # Check daily budget
        await self._check_and_alert(
            "daily",
            self._daily_usage["total"],
            self.config.daily_limit,
            daily_pct,
        )

        # Check monthly budget
        await self._check_and_alert(
            "monthly",
            self._monthly_usage["total"],
            self.config.monthly_limit,
            monthly_pct,
        )

        # Check domain budget
        if domain_limit < float("inf"):
            await self._check_and_alert(
                f"domain_{domain}",
                self._monthly_usage[domain],
                domain_limit,
                domain_pct,
                domain=domain,
            )

    async def _check_and_alert(
        self,
        budget_type: str,
        current: float,
        limit: float,
        percentage: float,
        domain: Optional[str] = None,
    ):
        """Check threshold and create alert if needed."""
        # Determine severity
        if percentage >= self.config.critical_threshold:
            severity = AlertSeverity.CRITICAL
        elif percentage >= self.config.warning_threshold:
            severity = AlertSeverity.WARNING
        else:
            return  # No alert needed

        # Check if we already have a similar unacknowledged alert
        existing = [a for a in self._alerts if a.budget_type == budget_type and a.severity == severity and not a.acknowledged]
        if existing:
            return  # Already alerted

        # Create alert
        alert = BudgetAlert(
            alert_id=f"alert_{budget_type}_{int(time.time())}",
            severity=severity,
            budget_type=budget_type,
            domain=domain,
            current_usage=current,
            budget_limit=limit,
            percentage_used=percentage * 100,
            message=f"Budget {budget_type} at {percentage*100:.1f}% (${current:.2f}/${limit:.2f})",
        )

        self._alerts.append(alert)

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        logger.warning(f"Budget alert: {alert.message}")

    async def _check_date_reset(self):
        """Check and reset daily/monthly counters if date changed."""
        now = datetime.now(timezone.utc)
        today = now.date()
        this_month = today.replace(day=1)

        async with self._lock:
            # Reset daily
            if today != self._current_day:
                self._daily_usage.clear()
                self._current_day = today
                logger.info("Daily usage counters reset")

            # Reset monthly
            if this_month != self._current_month:
                self._monthly_usage.clear()
                self._domain_usage.clear()
                self._current_month = this_month
                logger.info("Monthly usage counters reset")

    async def _load_usage_from_redis(self):
        """Load usage data from Redis."""
        if not self._redis:
            return

        try:
            # Load daily usage
            daily_data = await self._redis.hgetall("qbitel:cost_tracker:daily")
            for key, value in daily_data.items():
                self._daily_usage[key] = float(value)

            # Load monthly usage
            monthly_data = await self._redis.hgetall("qbitel:cost_tracker:monthly")
            for key, value in monthly_data.items():
                self._monthly_usage[key] = float(value)

        except Exception as e:
            logger.error(f"Failed to load usage from Redis: {e}")

    async def _persist_to_redis(self, record: UsageRecord):
        """Persist usage record to Redis."""
        if not self._redis:
            return

        try:
            # Update daily totals
            await self._redis.hincrbyfloat("qbitel:cost_tracker:daily", "total", record.total_cost)
            await self._redis.hincrbyfloat("qbitel:cost_tracker:daily", record.domain, record.total_cost)

            # Update monthly totals
            await self._redis.hincrbyfloat("qbitel:cost_tracker:monthly", "total", record.total_cost)
            await self._redis.hincrbyfloat("qbitel:cost_tracker:monthly", record.domain, record.total_cost)

            # Store record (with expiry)
            import json

            await self._redis.lpush("qbitel:cost_tracker:records", json.dumps(record.to_dict()))
            await self._redis.ltrim("qbitel:cost_tracker:records", 0, 99999)

        except Exception as e:
            logger.error(f"Failed to persist to Redis: {e}")
