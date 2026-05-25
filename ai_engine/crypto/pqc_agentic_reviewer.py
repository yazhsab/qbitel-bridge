"""
Agentic AI layer for PQC codebase migration.

The scanner is deterministic. This layer adds model-assisted planning,
patch sequencing, and validation strategy using the existing QBITEL LLM stack.
It is model-provider agnostic and works with Kimi, Qwen, DeepSeek, vLLM,
Ollama, or cloud providers as long as the backend exposes process_request().
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .codebase_migration import CodebaseMigrationReport

logger = logging.getLogger(__name__)


OPEN_SOURCE_AGENT_MODELS: Dict[str, Dict[str, Any]] = {
    "kimi-k2.6": {
        "provider": "kimi_api_or_vllm",
        "model_id": "kimi-k2.6",
        "best_for": ["long_context_code_review", "agentic_patch_planning", "multimodal_architecture_review"],
        "deployment": ["Kimi API", "self-hosted open-weight runtime where supported"],
    },
    "moonshotai/Kimi-K2-Thinking": {
        "provider": "vllm_or_sglang",
        "model_id": "moonshotai/Kimi-K2-Thinking",
        "best_for": ["deep_reasoning", "tool_calling", "large_codebase_planning"],
        "deployment": ["vLLM", "SGLang", "Hugging Face compatible providers"],
    },
    "qwen3-coder-next": {
        "provider": "vllm",
        "model_id": "qwen3-coder-next",
        "best_for": ["agentic_coding", "test_generation", "patch_synthesis"],
        "deployment": ["vLLM", "SGLang", "Ollama or LM Studio quantizations"],
    },
    "deepseek-ai/DeepSeek-V3.1": {
        "provider": "vllm_or_sglang",
        "model_id": "deepseek-ai/DeepSeek-V3.1",
        "best_for": ["tool_calling", "reasoning", "security_review"],
        "deployment": ["vLLM", "SGLang", "Hugging Face compatible providers"],
    },
    "qbitel-security": {
        "provider": "vllm",
        "model_id": "qbitel-security",
        "best_for": ["pqc_policy", "domain_security", "compliance_evidence"],
        "deployment": ["private QBITEL vLLM cluster"],
    },
}


@dataclass
class AgenticPQCReviewConfig:
    """Controls the model-assisted PQC migration pass."""

    model_override: str = "moonshotai/Kimi-K2-Thinking"
    max_tokens: int = 4096
    temperature: float = 0.1
    autonomous_mode: bool = True
    generate_patch_plan: bool = True
    require_test_gate: bool = True


@dataclass
class AgenticPQCReview:
    """LLM-generated agentic migration guidance."""

    enabled: bool
    model: str
    provider: str = "unknown"
    summary: str = ""
    autonomous_workflow: List[Dict[str, Any]] = field(default_factory=list)
    patch_strategy: List[Dict[str, Any]] = field(default_factory=list)
    validation_gates: List[str] = field(default_factory=list)
    residual_risks: List[str] = field(default_factory=list)
    raw_response: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgenticPQCMigrationReviewer:
    """Runs the agentic AI review pass against a deterministic scan report."""

    def __init__(self, llm_service: Any, config: Optional[AgenticPQCReviewConfig] = None):
        self.llm_service = llm_service
        self.config = config or AgenticPQCReviewConfig()

    async def review(self, report: CodebaseMigrationReport) -> AgenticPQCReview:
        if self.llm_service is None:
            return AgenticPQCReview(
                enabled=False,
                model=self.config.model_override,
                error="LLM service is not initialized",
            )

        prompt = self._build_prompt(report)
        try:
            from ..llm.unified_llm_service import LLMRequest, ResponseFormat

            request = LLMRequest(
                prompt=prompt,
                feature_domain="pqc_migration",
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                model_override=self.config.model_override,
                response_format=ResponseFormat.JSON,
                system_prompt=(
                    "You are QBITEL's autonomous PQC migration architect. "
                    "Produce JSON only. Prefer zero-touch overlays first, then safe patch PRs. "
                    "Never recommend silent production crypto replacement without tests."
                ),
            )
            response = await self.llm_service.process_request(request)
            payload = response.parsed_response or self._parse_json(response.content)
            return AgenticPQCReview(
                enabled=True,
                model=self.config.model_override,
                provider=response.provider,
                summary=str(payload.get("summary", "")),
                autonomous_workflow=list(payload.get("autonomous_workflow", [])),
                patch_strategy=list(payload.get("patch_strategy", [])),
                validation_gates=list(payload.get("validation_gates", [])),
                residual_risks=list(payload.get("residual_risks", [])),
                raw_response=response.content,
            )
        except Exception as exc:
            logger.warning("Agentic PQC review failed: %s", exc)
            return AgenticPQCReview(
                enabled=False,
                model=self.config.model_override,
                error=str(exc),
            )

    @staticmethod
    def _parse_json(content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                return json.loads(content[start : end + 1])
            raise

    def _build_prompt(self, report: CodebaseMigrationReport) -> str:
        scan_payload = report.to_dict()
        compact = {
            "finding_count": scan_payload["finding_count"],
            "portfolio_assessment": {
                "total_assets": scan_payload["portfolio_assessment"]["total_assets"],
                "critical_count": scan_payload["portfolio_assessment"]["critical_count"],
                "high_count": scan_payload["portfolio_assessment"]["high_count"],
                "average_qrs": scan_payload["portfolio_assessment"]["average_qrs"],
                "harvest_now_at_risk": scan_payload["portfolio_assessment"]["harvest_now_at_risk"],
            },
            "top_findings": scan_payload["findings"][:25],
            "deterministic_migration_plan": scan_payload["migration_plan"][:25],
            "overlay_plan": scan_payload["overlay_plan"],
        }
        return (
            "Create an autonomous PQC migration execution plan from this scan report.\n"
            "Return JSON with keys: summary, autonomous_workflow, patch_strategy, "
            "validation_gates, residual_risks.\n"
            "Each workflow item must include agent, action, input, output, and auto_execute boolean.\n"
            "Each patch item must include finding_id, file_path, proposed_change, tests, and merge_gate.\n"
            "Use overlay-first protection for production systems and patch generation for application code.\n\n"
            f"{json.dumps(compact, indent=2)}"
        )
