"""
CRONOS AI Engine - Dynamic Policy Engine

This module provides comprehensive policy management and enforcement
with support for dynamic rule evaluation, compliance monitoring,
and enterprise governance capabilities.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
import threading
from contextlib import asynccontextmanager
import weakref

import yaml
from jinja2 import Template, Environment, meta
import jsonpath_ng
from croniter import croniter

from ..core.config import Config
from ..core.exceptions import PolicyException, ValidationException, ComplianceException
from ..core.structured_logging import get_logger
from ..monitoring.metrics import AIEngineMetrics


class PolicyType(str, Enum):
    """Types of policies supported."""

    ACCESS_CONTROL = "access_control"
    DATA_GOVERNANCE = "data_governance"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    RESOURCE_MANAGEMENT = "resource_management"
    AUDIT = "audit"
    CUSTOM = "custom"


class PolicySeverity(str, Enum):
    """Policy violation severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyAction(str, Enum):
    """Actions to take when policy is triggered."""

    LOG = "log"
    WARN = "warn"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    ALERT = "alert"
    AUDIT = "audit"
    CUSTOM = "custom"


class RuleEngine(str, Enum):
    """Rule evaluation engines."""

    PYTHON = "python"
    CEL = "cel"  # Common Expression Language
    REGO = "rego"  # Open Policy Agent
    JSONPATH = "jsonpath"
    REGEX = "regex"
    JINJA2 = "jinja2"


@dataclass
class PolicyRule:
    """Individual policy rule definition."""

    rule_id: str
    name: str
    description: str
    engine: RuleEngine
    expression: str

    # Rule configuration
    enabled: bool = True
    priority: int = 100
    tags: List[str] = field(default_factory=list)

    # Conditions
    conditions: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[str] = None  # Cron expression

    # Actions
    action: PolicyAction = PolicyAction.LOG
    action_config: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0"


@dataclass
class Policy:
    """Policy definition containing multiple rules."""

    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    severity: PolicySeverity

    # Rules
    rules: List[PolicyRule] = field(default_factory=list)

    # Configuration
    enabled: bool = True
    enforcement_mode: str = "enforce"  # monitor, warn, enforce

    # Compliance
    compliance_frameworks: List[str] = field(default_factory=list)
    regulatory_tags: List[str] = field(default_factory=list)

    # Scope
    applies_to: Dict[str, Any] = field(default_factory=dict)
    exclusions: List[str] = field(default_factory=list)

    # Metadata
    owner: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        data["policy_type"] = self.policy_type.value
        data["severity"] = self.severity.value

        # Convert rules
        data["rules"] = []
        for rule in self.rules:
            rule_data = asdict(rule)
            rule_data["created_at"] = rule.created_at.isoformat()
            rule_data["updated_at"] = rule.updated_at.isoformat()
            rule_data["engine"] = rule.engine.value
            rule_data["action"] = rule.action.value
            data["rules"].append(rule_data)

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Policy":
        """Create policy from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data["policy_type"] = PolicyType(data["policy_type"])
        data["severity"] = PolicySeverity(data["severity"])

        # Convert rules
        rules = []
        for rule_data in data.get("rules", []):
            rule_data["created_at"] = datetime.fromisoformat(rule_data["created_at"])
            rule_data["updated_at"] = datetime.fromisoformat(rule_data["updated_at"])
            rule_data["engine"] = RuleEngine(rule_data["engine"])
            rule_data["action"] = PolicyAction(rule_data["action"])
            rules.append(PolicyRule(**rule_data))

        data["rules"] = rules
        return cls(**data)


@dataclass
class PolicyViolation:
    """Policy violation record."""

    violation_id: str
    policy_id: str
    rule_id: str

    # Violation details
    severity: PolicySeverity
    message: str
    context: Dict[str, Any]

    # Actions taken
    actions_taken: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_notes: str = ""

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


@dataclass
class EvaluationContext:
    """Context for policy evaluation."""

    request_id: str
    user_id: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Request data
    request_data: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)

    # Environment context
    environment: Dict[str, Any] = field(default_factory=dict)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class RuleEvaluator:
    """Base class for rule evaluators."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)

    async def evaluate(self, rule: PolicyRule, context: EvaluationContext) -> bool:
        """Evaluate rule against context."""
        raise NotImplementedError

    def validate_expression(self, expression: str) -> bool:
        """Validate rule expression syntax."""
        raise NotImplementedError


class PythonRuleEvaluator(RuleEvaluator):
    """
    Safe Python expression rule evaluator.

    SECURITY NOTE: This evaluator uses a restricted AST-based approach
    instead of eval() to prevent code injection attacks. Only safe
    operations are allowed (comparisons, arithmetic, attribute access).
    """

    def __init__(self, config: Config):
        super().__init__(config)
        # Allowed node types for safe evaluation
        self.allowed_nodes = {
            "Module",
            "Expr",
            "Load",
            "Store",
            "BinOp",
            "UnaryOp",
            "Compare",
            "BoolOp",
            "Add",
            "Sub",
            "Mult",
            "Div",
            "Mod",
            "Pow",
            "And",
            "Or",
            "Not",
            "Eq",
            "NotEq",
            "Lt",
            "LtE",
            "Gt",
            "GtE",
            "In",
            "NotIn",
            "Is",
            "IsNot",
            "Constant",
            "Num",
            "Str",
            "NameConstant",
            "Name",
            "Attribute",
            "Subscript",
            "Index",
            "Slice",
            "List",
            "Tuple",
            "Dict",
            "IfExp",  # Ternary operator
        }

        # Safe built-in functions
        self.safe_functions = {
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "max": max,
            "min": min,
            "sum": sum,
            "any": any,
            "all": all,
            "abs": abs,
            "round": round,
        }

    def _is_safe_ast(self, node: Any) -> bool:
        """Check if AST node is safe to evaluate."""
        import ast

        node_type = type(node).__name__

        # Check if node type is allowed
        if node_type not in self.allowed_nodes:
            self.logger.warning(f"Unsafe AST node type: {node_type}")
            return False

        # Recursively check child nodes
        for child in ast.iter_child_nodes(node):
            if not self._is_safe_ast(child):
                return False

        # Additional checks for specific node types
        if isinstance(node, ast.Name):
            # Only allow safe variable names (no dunder methods)
            if node.id.startswith("__") or node.id.startswith("_"):
                self.logger.warning(f"Unsafe variable name: {node.id}")
                return False

        if isinstance(node, ast.Attribute):
            # Only allow safe attribute access (no dunder attributes)
            if node.attr.startswith("__") or node.attr.startswith("_"):
                self.logger.warning(f"Unsafe attribute access: {node.attr}")
                return False

        if isinstance(node, ast.Call):
            # Function calls are not allowed in safe mode
            self.logger.warning("Function calls not allowed in safe evaluation")
            return False

        return True

    def _safe_eval(self, expression: str, namespace: Dict[str, Any]) -> Any:
        """Safely evaluate expression using AST validation."""
        import ast

        try:
            # Parse expression into AST
            tree = ast.parse(expression, mode="eval")

            # Validate AST is safe
            if not self._is_safe_ast(tree):
                raise ValueError("Expression contains unsafe operations")

            # Compile and evaluate with restricted namespace
            code = compile(tree, "<safe_eval>", "eval")

            # Create restricted namespace with no __builtins__
            safe_namespace = {"__builtins__": {}}
            safe_namespace.update(self.safe_functions)
            safe_namespace.update(namespace)

            # Evaluate in restricted environment
            result = eval(code, safe_namespace, {})
            return result

        except Exception as e:
            self.logger.error(f"Safe evaluation failed: {e}")
            raise

    async def evaluate(self, rule: PolicyRule, context: EvaluationContext) -> bool:
        """Evaluate Python expression safely."""
        try:
            # Create evaluation namespace with context data
            namespace = {
                "context": context,
                "user_id": context.user_id,
                "resource": context.resource,
                "action": context.action,
                "request_data": context.request_data,
                "headers": context.headers,
                "environment": context.environment,
                "metadata": context.metadata,
                "now": datetime.utcnow(),
                "re": re,  # Allow regex module for pattern matching
                "datetime": datetime,
                "timedelta": timedelta,
            }

            # Safely evaluate expression
            result = self._safe_eval(rule.expression, namespace)
            return bool(result)

        except Exception as e:
            self.logger.error(f"Python rule evaluation failed: {e}")
            return False

    def validate_expression(self, expression: str) -> bool:
        """Validate Python expression for safety."""
        import ast

        try:
            # Parse expression
            tree = ast.parse(expression, mode="eval")

            # Check if AST is safe
            if not self._is_safe_ast(tree):
                return False

            # Try to compile
            compile(tree, "<string>", "eval")
            return True

        except (SyntaxError, ValueError) as e:
            self.logger.warning(f"Expression validation failed: {e}")
            return False


class JSONPathRuleEvaluator(RuleEvaluator):
    """JSONPath expression rule evaluator."""

    async def evaluate(self, rule: PolicyRule, context: EvaluationContext) -> bool:
        """Evaluate JSONPath expression."""
        try:
            # Create evaluation data
            data = {
                "context": asdict(context),
                "user_id": context.user_id,
                "resource": context.resource,
                "action": context.action,
                "request_data": context.request_data,
                "headers": context.headers,
                "environment": context.environment,
                "metadata": context.metadata,
            }

            # Parse and evaluate JSONPath
            jsonpath_expr = jsonpath_ng.parse(rule.expression)
            matches = jsonpath_expr.find(data)

            # Check if any matches exist and are truthy
            return len(matches) > 0 and any(match.value for match in matches)

        except Exception as e:
            self.logger.error(f"JSONPath rule evaluation failed: {e}")
            return False

    def validate_expression(self, expression: str) -> bool:
        """Validate JSONPath expression."""
        try:
            jsonpath_ng.parse(expression)
            return True
        except Exception:
            return False


class RegexRuleEvaluator(RuleEvaluator):
    """Regex pattern rule evaluator."""

    async def evaluate(self, rule: PolicyRule, context: EvaluationContext) -> bool:
        """Evaluate regex pattern."""
        try:
            # Get target field from rule conditions
            target_field = rule.conditions.get("target_field", "resource")

            # Get target value from context
            target_value = getattr(context, target_field, None)
            if target_value is None:
                target_value = context.request_data.get(target_field, "")

            # Compile and match regex
            pattern = re.compile(rule.expression)
            return bool(pattern.match(str(target_value)))

        except Exception as e:
            self.logger.error(f"Regex rule evaluation failed: {e}")
            return False

    def validate_expression(self, expression: str) -> bool:
        """Validate regex expression."""
        try:
            re.compile(expression)
            return True
        except re.error:
            return False


class Jinja2RuleEvaluator(RuleEvaluator):
    """Jinja2 template rule evaluator."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.env = Environment()

    async def evaluate(self, rule: PolicyRule, context: EvaluationContext) -> bool:
        """Evaluate Jinja2 template."""
        try:
            # Create template variables
            variables = {
                "context": context,
                "user_id": context.user_id,
                "resource": context.resource,
                "action": context.action,
                "request_data": context.request_data,
                "headers": context.headers,
                "environment": context.environment,
                "metadata": context.metadata,
                "now": datetime.utcnow(),
            }

            # Render template
            template = self.env.from_string(rule.expression)
            result = template.render(**variables)

            # Convert result to boolean
            if result.lower() in ("true", "1", "yes"):
                return True
            elif result.lower() in ("false", "0", "no", ""):
                return False
            else:
                return bool(result)

        except Exception as e:
            self.logger.error(f"Jinja2 rule evaluation failed: {e}")
            return False

    def validate_expression(self, expression: str) -> bool:
        """Validate Jinja2 template."""
        try:
            self.env.from_string(expression)
            return True
        except Exception:
            return False


class PolicyRegistry:
    """
    Policy registry for managing policies and rules.

    Provides CRUD operations, versioning, and persistence for policies.
    """

    def __init__(self, config: Config):
        """Initialize policy registry."""
        self.config = config
        self.logger = get_logger(__name__)

        # Storage configuration
        self.registry_path = Path(
            getattr(config, "policy_registry_path", "./policy_registry")
        )

        # In-memory policy cache
        self.policies: Dict[str, Policy] = {}
        self.policy_lock = threading.RLock()

        # Initialize storage
        self._initialize_storage()
        self._load_policies()

        self.logger.info("PolicyRegistry initialized")

    def _initialize_storage(self):
        """Initialize policy storage directories."""
        self.registry_path.mkdir(parents=True, exist_ok=True)
        (self.registry_path / "policies").mkdir(exist_ok=True)
        (self.registry_path / "violations").mkdir(exist_ok=True)
        (self.registry_path / "templates").mkdir(exist_ok=True)

    def _load_policies(self):
        """Load policies from storage."""
        policies_dir = self.registry_path / "policies"

        try:
            for policy_file in policies_dir.glob("*.json"):
                try:
                    with open(policy_file, "r") as f:
                        data = json.load(f)

                    policy = Policy.from_dict(data)
                    self.policies[policy.policy_id] = policy

                except Exception as e:
                    self.logger.warning(
                        f"Failed to load policy from {policy_file}: {e}"
                    )

            self.logger.info(f"Loaded {len(self.policies)} policies from storage")

        except Exception as e:
            self.logger.error(f"Failed to load policies: {e}")

    async def create_policy(self, policy: Policy) -> str:
        """Create a new policy."""
        try:
            # Validate policy
            await self._validate_policy(policy)

            # Store policy
            with self.policy_lock:
                self.policies[policy.policy_id] = policy

            # Persist to storage
            await self._save_policy(policy)

            self.logger.info(f"Created policy: {policy.name} ({policy.policy_id})")
            return policy.policy_id

        except Exception as e:
            self.logger.error(f"Failed to create policy {policy.name}: {e}")
            raise PolicyException(f"Policy creation failed: {e}")

    async def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing policy."""
        try:
            with self.policy_lock:
                if policy_id not in self.policies:
                    return False

                policy = self.policies[policy_id]

                # Apply updates
                for key, value in updates.items():
                    if hasattr(policy, key):
                        setattr(policy, key, value)

                policy.updated_at = datetime.utcnow()
                policy.version = self._increment_version(policy.version)

            # Validate updated policy
            await self._validate_policy(policy)

            # Persist changes
            await self._save_policy(policy)

            self.logger.info(f"Updated policy: {policy_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update policy {policy_id}: {e}")
            raise PolicyException(f"Policy update failed: {e}")

    async def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy."""
        try:
            with self.policy_lock:
                if policy_id not in self.policies:
                    return False

                # Archive policy before deletion
                policy = self.policies[policy_id]
                await self._archive_policy(policy)

                # Remove from memory
                del self.policies[policy_id]

            # Remove from storage
            policy_file = self.registry_path / "policies" / f"{policy_id}.json"
            if policy_file.exists():
                policy_file.unlink()

            self.logger.info(f"Deleted policy: {policy_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete policy {policy_id}: {e}")
            return False

    async def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get policy by ID."""
        with self.policy_lock:
            return self.policies.get(policy_id)

    async def list_policies(
        self, policy_type: Optional[PolicyType] = None, enabled_only: bool = True
    ) -> List[Policy]:
        """List policies with optional filtering."""
        with self.policy_lock:
            policies = list(self.policies.values())

        # Apply filters
        if policy_type:
            policies = [p for p in policies if p.policy_type == policy_type]

        if enabled_only:
            policies = [p for p in policies if p.enabled]

        return policies

    async def get_applicable_policies(self, context: EvaluationContext) -> List[Policy]:
        """Get policies applicable to the given context."""
        applicable_policies = []

        with self.policy_lock:
            for policy in self.policies.values():
                if not policy.enabled:
                    continue

                if await self._is_policy_applicable(policy, context):
                    applicable_policies.append(policy)

        # Sort by priority (if available) or severity
        applicable_policies.sort(
            key=lambda p: (
                p.severity == PolicySeverity.CRITICAL,
                p.severity == PolicySeverity.HIGH,
                p.severity == PolicySeverity.MEDIUM,
                p.name,
            ),
            reverse=True,
        )

        return applicable_policies

    async def _validate_policy(self, policy: Policy) -> None:
        """Validate policy configuration."""
        # Basic validation
        if not policy.name or not policy.policy_id:
            raise ValidationException("Policy name and ID are required")

        if not policy.rules:
            raise ValidationException("Policy must have at least one rule")

        # Validate rules
        for rule in policy.rules:
            if not rule.expression:
                raise ValidationException(f"Rule {rule.name} must have an expression")

            # TODO: Validate rule expression based on engine type

        self.logger.debug(f"Policy validation passed: {policy.policy_id}")

    async def _save_policy(self, policy: Policy) -> None:
        """Save policy to storage."""
        policy_file = self.registry_path / "policies" / f"{policy.policy_id}.json"

        with open(policy_file, "w") as f:
            json.dump(policy.to_dict(), f, indent=2)

    async def _archive_policy(self, policy: Policy) -> None:
        """Archive a policy before deletion."""
        archive_dir = self.registry_path / "archived"
        archive_dir.mkdir(exist_ok=True)

        archive_file = archive_dir / f"{policy.policy_id}_{int(time.time())}.json"

        with open(archive_file, "w") as f:
            json.dump(policy.to_dict(), f, indent=2)

    async def _is_policy_applicable(
        self, policy: Policy, context: EvaluationContext
    ) -> bool:
        """Check if policy applies to the given context."""
        # Check exclusions
        if context.resource and context.resource in policy.exclusions:
            return False

        # Check applies_to conditions
        if policy.applies_to:
            # Simple matching - can be extended for complex conditions
            if "resource_pattern" in policy.applies_to:
                pattern = policy.applies_to["resource_pattern"]
                if not re.match(pattern, context.resource or ""):
                    return False

            if "user_groups" in policy.applies_to:
                user_groups = context.metadata.get("user_groups", [])
                required_groups = policy.applies_to["user_groups"]
                if not any(group in user_groups for group in required_groups):
                    return False

        return True

    def _increment_version(self, version: str) -> str:
        """Increment semantic version."""
        try:
            parts = version.split(".")
            if len(parts) >= 2:
                minor = int(parts[1]) + 1
                return f"{parts[0]}.{minor}.0"
        except ValueError:
            pass

        return f"{version}.1"


class PolicyEngine:
    """
    Main policy engine for evaluating and enforcing policies.

    Provides dynamic policy evaluation, violation handling,
    and compliance monitoring capabilities.
    """

    def __init__(self, config: Config, metrics: AIEngineMetrics):
        """Initialize policy engine."""
        self.config = config
        self.metrics = metrics
        self.logger = get_logger(__name__)

        # Components
        self.registry = PolicyRegistry(config)

        # Rule evaluators
        self.evaluators: Dict[RuleEngine, RuleEvaluator] = {
            RuleEngine.PYTHON: PythonRuleEvaluator(config),
            RuleEngine.JSONPATH: JSONPathRuleEvaluator(config),
            RuleEngine.REGEX: RegexRuleEvaluator(config),
            RuleEngine.JINJA2: Jinja2RuleEvaluator(config),
        }

        # Violation tracking
        self.violations: List[PolicyViolation] = []
        self.violation_lock = threading.RLock()

        # Performance optimization
        self.evaluation_cache: Dict[str, Tuple[bool, datetime]] = {}
        self.cache_ttl = timedelta(
            minutes=getattr(config, "policy_cache_ttl_minutes", 5)
        )

        # Statistics
        self.evaluation_stats = {
            "total_evaluations": 0,
            "policy_violations": 0,
            "cache_hits": 0,
            "evaluation_errors": 0,
        }

        self.logger.info("PolicyEngine initialized")

    async def evaluate_request(
        self, context: EvaluationContext, fail_fast: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate policies against a request context.

        Args:
            context: Evaluation context containing request details
            fail_fast: Stop evaluation on first violation if True

        Returns:
            Evaluation results with violations and actions
        """
        start_time = time.time()
        evaluation_id = str(uuid.uuid4())

        try:
            # Get applicable policies
            policies = await self.registry.get_applicable_policies(context)

            if not policies:
                return {
                    "evaluation_id": evaluation_id,
                    "allowed": True,
                    "violations": [],
                    "actions": [],
                    "evaluation_time_ms": (time.time() - start_time) * 1000,
                    "policies_evaluated": 0,
                }

            violations = []
            actions = []

            # Evaluate each policy
            for policy in policies:
                policy_result = await self._evaluate_policy(policy, context)

                if policy_result["violated"]:
                    violations.extend(policy_result["violations"])
                    actions.extend(policy_result["actions"])

                    if fail_fast and policy.severity in [
                        PolicySeverity.CRITICAL,
                        PolicySeverity.HIGH,
                    ]:
                        break

            # Determine final decision
            allowed = len(violations) == 0 or all(
                v.severity in [PolicySeverity.LOW, PolicySeverity.MEDIUM]
                for v in violations
            )

            # Update statistics
            self.evaluation_stats["total_evaluations"] += 1
            if violations:
                self.evaluation_stats["policy_violations"] += len(violations)

            # Log violations
            await self._log_violations(violations)

            evaluation_time = (time.time() - start_time) * 1000

            result = {
                "evaluation_id": evaluation_id,
                "allowed": allowed,
                "violations": [asdict(v) for v in violations],
                "actions": actions,
                "evaluation_time_ms": evaluation_time,
                "policies_evaluated": len(policies),
            }

            self.logger.debug(
                f"Policy evaluation completed: {evaluation_id}, "
                f"Allowed: {allowed}, Violations: {len(violations)}, "
                f"Time: {evaluation_time:.2f}ms"
            )

            return result

        except Exception as e:
            self.evaluation_stats["evaluation_errors"] += 1
            self.logger.error(f"Policy evaluation failed: {e}")

            # Default to deny in case of errors for security
            return {
                "evaluation_id": evaluation_id,
                "allowed": False,
                "violations": [],
                "actions": ["BLOCK"],
                "evaluation_time_ms": (time.time() - start_time) * 1000,
                "error": str(e),
            }

    async def _evaluate_policy(
        self, policy: Policy, context: EvaluationContext
    ) -> Dict[str, Any]:
        """Evaluate a single policy against context."""
        violations = []
        actions = []

        # Evaluate each rule in the policy
        for rule in policy.rules:
            if not rule.enabled:
                continue

            # Check if rule should run based on schedule
            if rule.schedule and not self._should_rule_run(rule):
                continue

            try:
                # Get rule evaluator
                evaluator = self.evaluators.get(rule.engine)
                if not evaluator:
                    self.logger.warning(f"No evaluator for engine: {rule.engine}")
                    continue

                # Check cache first
                cache_key = self._get_cache_key(rule, context)
                cached_result = self._get_cached_result(cache_key)

                if cached_result is not None:
                    rule_violated = cached_result
                    self.evaluation_stats["cache_hits"] += 1
                else:
                    # Evaluate rule
                    rule_violated = await evaluator.evaluate(rule, context)

                    # Cache result
                    self._cache_result(cache_key, rule_violated)

                if rule_violated:
                    # Create violation record
                    violation = PolicyViolation(
                        violation_id=str(uuid.uuid4()),
                        policy_id=policy.policy_id,
                        rule_id=rule.rule_id,
                        severity=policy.severity,
                        message=f"Policy '{policy.name}' rule '{rule.name}' violated",
                        context={
                            "rule_expression": rule.expression,
                            "rule_engine": rule.engine.value,
                            "evaluation_context": asdict(context),
                        },
                    )

                    violations.append(violation)

                    # Determine actions
                    action = self._get_rule_action(rule, policy)
                    if action not in actions:
                        actions.append(action)

            except Exception as e:
                self.logger.error(f"Rule evaluation failed: {rule.rule_id}: {e}")
                continue

        return {
            "violated": len(violations) > 0,
            "violations": violations,
            "actions": actions,
        }

    def _should_rule_run(self, rule: PolicyRule) -> bool:
        """Check if rule should run based on schedule."""
        if not rule.schedule:
            return True

        try:
            cron = croniter(rule.schedule, datetime.utcnow())
            next_run = cron.get_next(datetime)
            prev_run = cron.get_prev(datetime)

            # Check if current time is within the scheduled window
            now = datetime.utcnow()
            return abs((now - prev_run).total_seconds()) < 60  # Within 1 minute

        except Exception as e:
            self.logger.warning(f"Invalid cron expression for rule {rule.rule_id}: {e}")
            return True

    def _get_cache_key(self, rule: PolicyRule, context: EvaluationContext) -> str:
        """Generate cache key for rule evaluation."""
        # Create a hash of rule and relevant context data
        import hashlib

        key_data = {
            "rule_id": rule.rule_id,
            "rule_version": rule.version,
            "user_id": context.user_id,
            "resource": context.resource,
            "action": context.action,
            # Include other relevant context data
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[bool]:
        """Get cached evaluation result."""
        if cache_key in self.evaluation_cache:
            result, timestamp = self.evaluation_cache[cache_key]
            if datetime.utcnow() - timestamp < self.cache_ttl:
                return result
            else:
                # Remove expired cache entry
                del self.evaluation_cache[cache_key]

        return None

    def _cache_result(self, cache_key: str, result: bool) -> None:
        """Cache evaluation result."""
        self.evaluation_cache[cache_key] = (result, datetime.utcnow())

        # Clean up old cache entries periodically
        if len(self.evaluation_cache) > 10000:
            self._cleanup_cache()

    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        now = datetime.utcnow()
        expired_keys = [
            key
            for key, (_, timestamp) in self.evaluation_cache.items()
            if now - timestamp >= self.cache_ttl
        ]

        for key in expired_keys:
            del self.evaluation_cache[key]

    def _get_rule_action(self, rule: PolicyRule, policy: Policy) -> str:
        """Get action to take for rule violation."""
        # Rule-specific action takes precedence
        if rule.action != PolicyAction.LOG:
            return rule.action.value.upper()

        # Default actions based on policy severity
        severity_actions = {
            PolicySeverity.LOW: "LOG",
            PolicySeverity.MEDIUM: "WARN",
            PolicySeverity.HIGH: "BLOCK",
            PolicySeverity.CRITICAL: "BLOCK",
        }

        return severity_actions.get(policy.severity, "LOG")

    async def _log_violations(self, violations: List[PolicyViolation]) -> None:
        """Log policy violations."""
        if not violations:
            return

        with self.violation_lock:
            self.violations.extend(violations)

        # Persist violations to storage
        for violation in violations:
            await self._save_violation(violation)

        # Log violations
        for violation in violations:
            if violation.severity == PolicySeverity.CRITICAL:
                self.logger.critical(f"Policy violation: {violation.message}")
            elif violation.severity == PolicySeverity.HIGH:
                self.logger.error(f"Policy violation: {violation.message}")
            elif violation.severity == PolicySeverity.MEDIUM:
                self.logger.warning(f"Policy violation: {violation.message}")
            else:
                self.logger.info(f"Policy violation: {violation.message}")

    async def _save_violation(self, violation: PolicyViolation) -> None:
        """Save violation to storage."""
        violations_dir = self.registry.registry_path / "violations"
        violation_file = violations_dir / f"{violation.violation_id}.json"

        try:
            with open(violation_file, "w") as f:
                data = asdict(violation)
                data["timestamp"] = violation.timestamp.isoformat()
                if violation.resolved_at:
                    data["resolved_at"] = violation.resolved_at.isoformat()
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save violation: {e}")

    async def get_violations(
        self,
        policy_id: Optional[str] = None,
        severity: Optional[PolicySeverity] = None,
        unresolved_only: bool = True,
        limit: int = 100,
    ) -> List[PolicyViolation]:
        """Get policy violations with filtering."""
        with self.violation_lock:
            violations = list(self.violations)

        # Apply filters
        if policy_id:
            violations = [v for v in violations if v.policy_id == policy_id]

        if severity:
            violations = [v for v in violations if v.severity == severity]

        if unresolved_only:
            violations = [v for v in violations if not v.resolved]

        # Sort by timestamp (newest first) and limit
        violations.sort(key=lambda v: v.timestamp, reverse=True)
        return violations[:limit]

    async def resolve_violation(
        self, violation_id: str, resolved_by: str, resolution_notes: str = ""
    ) -> bool:
        """Mark a violation as resolved."""
        try:
            with self.violation_lock:
                for violation in self.violations:
                    if violation.violation_id == violation_id:
                        violation.resolved = True
                        violation.resolved_at = datetime.utcnow()
                        violation.resolved_by = resolved_by
                        violation.resolution_notes = resolution_notes

                        # Update stored violation
                        await self._save_violation(violation)

                        self.logger.info(f"Resolved violation: {violation_id}")
                        return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to resolve violation {violation_id}: {e}")
            return False

    async def get_statistics(self) -> Dict[str, Any]:
        """Get policy engine statistics."""
        with self.violation_lock:
            total_violations = len(self.violations)
            unresolved_violations = len([v for v in self.violations if not v.resolved])

        return {
            "evaluation_stats": self.evaluation_stats.copy(),
            "total_violations": total_violations,
            "unresolved_violations": unresolved_violations,
            "cache_entries": len(self.evaluation_cache),
            "active_policies": len(await self.registry.list_policies()),
            "evaluators": list(self.evaluators.keys()),
            "timestamp": datetime.utcnow().isoformat(),
        }


# Global policy engine instance
_policy_engine: Optional[PolicyEngine] = None


async def initialize_policy_engine(
    config: Config, metrics: AIEngineMetrics
) -> PolicyEngine:
    """Initialize global policy engine."""
    global _policy_engine

    _policy_engine = PolicyEngine(config, metrics)
    return _policy_engine


def get_policy_engine() -> Optional[PolicyEngine]:
    """Get global policy engine instance."""
    return _policy_engine


async def shutdown_policy_engine():
    """Shutdown global policy engine."""
    global _policy_engine
    if _policy_engine:
        # Perform any necessary cleanup
        _policy_engine = None
