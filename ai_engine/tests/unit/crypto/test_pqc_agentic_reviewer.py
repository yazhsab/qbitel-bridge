"""Tests for agentic PQC migration reviewer."""

import json

import pytest

from ai_engine.crypto.codebase_migration import ClientCodebasePQCMigrator
from ai_engine.crypto.pqc_agentic_reviewer import (
    AgenticPQCReviewConfig,
    AgenticPQCMigrationReviewer,
    OPEN_SOURCE_AGENT_MODELS,
)


class FakeLLMResponse:
    content = json.dumps(
        {
            "summary": "Use overlay-first PQC migration.",
            "autonomous_workflow": [
                {
                    "agent": "repository_scanner",
                    "action": "scan",
                    "input": "repo",
                    "output": "crypto_inventory",
                    "auto_execute": True,
                }
            ],
            "patch_strategy": [
                {
                    "finding_id": "crypto-test",
                    "file_path": "service.py",
                    "proposed_change": "wrap RSA with hybrid KEM envelope",
                    "tests": ["unit", "interop"],
                    "merge_gate": "all tests pass",
                }
            ],
            "validation_gates": ["unit tests", "interop tests"],
            "residual_risks": ["peer compatibility"],
        }
    )
    provider = "vllm"
    parsed_response = json.loads(content)


class FakeLLMService:
    def __init__(self):
        self.requests = []

    async def process_request(self, request):
        self.requests.append(request)
        return FakeLLMResponse()


@pytest.mark.asyncio
async def test_agentic_reviewer_calls_selected_open_source_model():
    report = ClientCodebasePQCMigrator().scan_sources(
        {"service.py": "key = rsa.generate_private_key(key_size=2048)\n"}
    )
    llm = FakeLLMService()
    reviewer = AgenticPQCMigrationReviewer(
        llm,
        AgenticPQCReviewConfig(model_override="moonshotai/Kimi-K2-Thinking"),
    )

    result = await reviewer.review(report)

    assert result.enabled is True
    assert result.provider == "vllm"
    assert result.summary == "Use overlay-first PQC migration."
    assert llm.requests[0].model_override == "moonshotai/Kimi-K2-Thinking"
    assert llm.requests[0].feature_domain == "pqc_migration"


def test_recommended_agent_models_include_kimi_qwen_and_deepseek():
    assert "moonshotai/Kimi-K2-Thinking" in OPEN_SOURCE_AGENT_MODELS
    assert "qwen3-coder-next" in OPEN_SOURCE_AGENT_MODELS
    assert "deepseek-ai/DeepSeek-V3.1" in OPEN_SOURCE_AGENT_MODELS
