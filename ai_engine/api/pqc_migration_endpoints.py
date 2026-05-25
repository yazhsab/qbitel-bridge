"""
PQC migration reviewer API endpoints.

These endpoints let authenticated clients submit source files or scan a mounted
client repository and receive a quantum-risk-ranked migration plan.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..api.auth import get_current_user
from ..crypto.codebase_migration import (
    ClientCodebasePQCMigrator,
    CodebaseScanConfig,
    DEFAULT_EXTENSIONS,
)
from ..crypto.pqc_agentic_reviewer import (
    AgenticPQCReviewConfig,
    AgenticPQCMigrationReviewer,
    OPEN_SOURCE_AGENT_MODELS,
)
from ..crypto.quantum_threat_scoring import DataSensitivity, MigrationPhase
from ..llm.unified_llm_service import get_llm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pqc-migration", tags=["PQC Migration"])


class SourceFilePayload(BaseModel):
    """Source file payload for API-driven scanning."""

    path: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., max_length=1_000_000)


class PQCCodebaseScanRequest(BaseModel):
    """Request to review a client codebase for PQC migration readiness."""

    repository_url: Optional[str] = Field(
        None,
        description="Git repository URL from GitHub, GitLab, Bitbucket, Azure DevOps, or another Git host",
    )
    repository_path: Optional[str] = Field(
        None,
        description="Path to a mounted client repository under QBITEL_CLIENT_CODEBASE_ROOT",
    )
    git_ref: Optional[str] = Field(None, description="Optional branch, tag, or commit to scan")
    git_clone_depth: int = Field(1, ge=0, le=1000, description="Shallow clone depth; 0 means full clone")
    access_token_env: Optional[str] = Field(
        None,
        description="Environment variable name containing an HTTPS Git access token",
    )
    source_files: List[SourceFilePayload] = Field(
        default_factory=list,
        description="Inline source files for scanner-only review",
    )
    data_sensitivity: str = Field("confidential", description="public/internal/confidential/secret/top_secret")
    data_retention_years: int = Field(10, ge=0, le=100)
    domain: str = Field("client-codebase", max_length=80)
    migration_phase: str = Field("not_started")
    max_file_size_bytes: int = Field(1_000_000, ge=1_024, le=5_000_000)
    enable_agentic_ai: bool = Field(False, description="Run model-assisted autonomous migration planning")
    agentic_model: str = Field(
        "moonshotai/Kimi-K2-Thinking",
        description="Model override for agentic planning, for example kimi-k2.6 or qwen3-coder-next",
    )
    agentic_autonomous_mode: bool = Field(True, description="Plan zero-touch workflow steps where safe")


@router.get("/capabilities")
async def pqc_migration_capabilities():
    """Describe supported codebase review capabilities."""
    return {
        "scanner": "client-codebase-pqc-migration-reviewer",
        "supported_inputs": ["git_repository_url", "mounted_repository_path", "inline_source_files"],
        "git_platforms": ["GitHub", "GitLab", "Bitbucket", "Azure DevOps", "generic Git"],
        "recommended_open_source_agent_models": OPEN_SOURCE_AGENT_MODELS,
        "supported_extensions": sorted(DEFAULT_EXTENSIONS),
        "detects": [
            "RSA key generation and public-key encryption",
            "ECDSA/ECDH and X25519 classical key exchange",
            "finite-field Diffie-Hellman",
            "AES-128, DES, and 3DES usage",
            "SHA1/MD5 signature and digest usage",
        ],
        "outputs": [
            "crypto findings",
            "QBITEL quantum risk scoring",
            "harvest-now-decrypt-later indicators",
            "prioritized PQC migration plan",
            "zero-touch PQC overlay plan",
            "AI review prompt for pull-request generation",
        ],
        "safe_automation": "Generates human-approved patch plans; does not blindly rewrite cryptographic code.",
        "agentic_ai": {
            "enabled_per_request": True,
            "model_override_supported": True,
            "providers": ["Kimi API", "vLLM", "SGLang", "Ollama", "OpenAI-compatible APIs"],
        },
    }


@router.post("/scan")
async def scan_codebase_for_pqc_migration(
    request: PQCCodebaseScanRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Review a client codebase and return a PQC migration report.

    Clients can either submit inline files or provide a repository path that has
    been mounted under QBITEL_CLIENT_CODEBASE_ROOT.
    """
    if not request.repository_url and not request.repository_path and not request.source_files:
        raise HTTPException(status_code=422, detail="Provide repository_url, repository_path, or source_files")

    config = CodebaseScanConfig(
        max_file_size_bytes=request.max_file_size_bytes,
        data_sensitivity=_parse_data_sensitivity(request.data_sensitivity),
        data_retention_years=request.data_retention_years,
        domain=request.domain,
        current_migration_phase=_parse_migration_phase(request.migration_phase),
    )
    migrator = ClientCodebasePQCMigrator(config=config)

    try:
        if request.repository_url:
            report = migrator.scan_git_repository(
                request.repository_url,
                ref=request.git_ref,
                depth=request.git_clone_depth,
                access_token=_read_access_token(request.access_token_env),
            )
        elif request.source_files:
            sources = {source.path: source.content for source in request.source_files}
            report = migrator.scan_sources(sources)
        else:
            target = _resolve_allowed_repository_path(request.repository_path or "")
            report = migrator.scan_path(target)

        result = report.to_dict()
        result["requested_by"] = current_user.get("user_id", "unknown")
        if request.enable_agentic_ai:
            reviewer = AgenticPQCMigrationReviewer(
                get_llm_service(),
                AgenticPQCReviewConfig(
                    model_override=request.agentic_model,
                    autonomous_mode=request.agentic_autonomous_mode,
                ),
            )
            agentic_review = await reviewer.review(report)
            result["agentic_ai_review"] = agentic_review.to_dict()
        return result
    except HTTPException:
        raise
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("PQC migration scan failed")
        raise HTTPException(status_code=500, detail=f"PQC migration scan failed: {exc}")


def _parse_data_sensitivity(value: str) -> DataSensitivity:
    normalized = value.strip().upper().replace("-", "_")
    try:
        return DataSensitivity[normalized]
    except KeyError as exc:
        allowed = ", ".join(level.name.lower() for level in DataSensitivity)
        raise HTTPException(status_code=422, detail=f"Invalid data_sensitivity. Allowed: {allowed}") from exc


def _parse_migration_phase(value: str) -> MigrationPhase:
    normalized = value.strip().upper().replace("-", "_")
    try:
        return MigrationPhase[normalized]
    except KeyError as exc:
        allowed = ", ".join(phase.name.lower() for phase in MigrationPhase)
        raise HTTPException(status_code=422, detail=f"Invalid migration_phase. Allowed: {allowed}") from exc


def _resolve_allowed_repository_path(repository_path: str) -> Path:
    allowed_root = Path(os.getenv("QBITEL_CLIENT_CODEBASE_ROOT", os.getcwd())).expanduser().resolve()
    requested = Path(repository_path).expanduser()
    target = requested.resolve() if requested.is_absolute() else (allowed_root / requested).resolve()

    if target != allowed_root and allowed_root not in target.parents:
        raise HTTPException(
            status_code=403,
            detail=f"repository_path must be under configured root {allowed_root}",
        )
    return target


def _read_access_token(env_name: Optional[str]) -> Optional[str]:
    if not env_name:
        return None
    if not env_name.startswith("QBITEL_"):
        raise HTTPException(status_code=422, detail="access_token_env must use a QBITEL_ environment variable")
    token = os.getenv(env_name)
    if not token:
        raise HTTPException(status_code=400, detail=f"Environment variable {env_name} is not set")
    return token
