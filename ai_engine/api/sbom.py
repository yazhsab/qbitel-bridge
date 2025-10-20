"""
CRONOS AI - SBOM API Endpoints
Provides REST API for Software Bill of Materials access and distribution.
Meets EO 14028 requirements for federal software procurement.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sbom", tags=["SBOM"])

# SBOM storage configuration
SBOM_DIR = Path("/var/cronos-ai/sbom-artifacts")
SBOM_FALLBACK_DIR = Path("./sbom-artifacts")  # Development fallback


class SBOMVersion(BaseModel):
    """SBOM version metadata."""

    version: str = Field(..., description="Version identifier (e.g., v1.0.0)")
    generated_at: str = Field(..., description="ISO 8601 timestamp")
    commit: str = Field(..., description="Git commit SHA")
    components: List[str] = Field(..., description="Available components")


class SBOMVulnerabilitySummary(BaseModel):
    """Vulnerability summary for SBOM."""

    total: int = Field(0, description="Total vulnerabilities")
    critical: int = Field(0, description="Critical severity count")
    high: int = Field(0, description="High severity count")
    medium: int = Field(0, description="Medium severity count")
    low: int = Field(0, description="Low severity count")


class SBOMMetadata(BaseModel):
    """SBOM metadata response."""

    component: str
    version: str
    format: str
    packages: int
    vulnerabilities: Optional[SBOMVulnerabilitySummary] = None
    generated_at: str
    sbom_file: str


def get_sbom_directory() -> Path:
    """Get SBOM directory with fallback logic."""
    if SBOM_DIR.exists():
        return SBOM_DIR
    elif SBOM_FALLBACK_DIR.exists():
        logger.warning(f"Using fallback SBOM directory: {SBOM_FALLBACK_DIR}")
        return SBOM_FALLBACK_DIR
    else:
        raise HTTPException(
            status_code=503,
            detail="SBOM artifact directory not configured or not available",
        )


@router.get("/versions", response_model=List[str])
async def list_sbom_versions():
    """
    List all available SBOM versions.

    Returns:
        List of version identifiers (e.g., ['v1.0.0', 'v0.9.0', 'main'])

    Example:
        GET /api/v1/sbom/versions
    """
    try:
        sbom_dir = get_sbom_directory()
        versions = [
            d.name
            for d in sbom_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        return sorted(versions, reverse=True)
    except Exception as e:
        logger.error(f"Failed to list SBOM versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/versions/{version}", response_model=SBOMVersion)
async def get_version_metadata(version: str):
    """
    Get metadata for a specific SBOM version.

    Args:
        version: Version identifier (e.g., v1.0.0)

    Returns:
        Metadata including components and generation info

    Example:
        GET /api/v1/sbom/versions/v1.0.0
    """
    try:
        sbom_dir = get_sbom_directory()
        version_dir = sbom_dir / version

        if not version_dir.exists():
            raise HTTPException(
                status_code=404, detail=f"SBOM version {version} not found"
            )

        # Read summary file if available
        summary_file = version_dir / "sbom-summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                return SBOMVersion(
                    version=summary.get("version", version),
                    generated_at=summary.get("generated_at", ""),
                    commit=summary.get("commit", ""),
                    components=list(summary.get("components", {}).keys()),
                )

        # Fallback: scan directory for components
        components = []
        for sbom_file in version_dir.glob("*-spdx.json"):
            component = sbom_file.stem.replace("-spdx", "")
            components.append(component)

        return SBOMVersion(
            version=version,
            generated_at=datetime.utcnow().isoformat() + "Z",
            commit="unknown",
            components=sorted(components),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get version metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download")
async def download_sbom(
    version: str = Query(..., description="CRONOS AI version (e.g., v1.0.0)"),
    component: str = Query("cronos-ai-platform", description="Component name"),
    format: Literal["spdx", "cyclonedx"] = Query("spdx", description="SBOM format"),
):
    """
    Download SBOM for a specific version and component.

    Args:
        version: Version identifier (e.g., v1.0.0, main)
        component: Component name (e.g., cronos-ai-engine, cronos-dataplane)
        format: SBOM format (spdx or cyclonedx)

    Returns:
        SBOM JSON document

    Example:
        GET /api/v1/sbom/download?version=v1.0.0&component=cronos-ai-engine&format=spdx
    """
    try:
        sbom_dir = get_sbom_directory()
        sbom_file = sbom_dir / version / f"{component}-{format}.json"

        if not sbom_file.exists():
            # Try alternative naming
            sbom_file = sbom_dir / version / f"cronos-{component}-{format}.json"

        if not sbom_file.exists():
            available = (
                list((sbom_dir / version).glob(f"*-{format}.json"))
                if (sbom_dir / version).exists()
                else []
            )
            available_names = [f.stem.replace(f"-{format}", "") for f in available]

            raise HTTPException(
                status_code=404,
                detail=f"SBOM not found for component '{component}' version '{version}' format '{format}'. "
                f"Available components: {available_names if available_names else 'none'}",
            )

        with open(sbom_file) as f:
            sbom_data = json.load(f)

        # Set appropriate headers for download
        headers = {
            "Content-Disposition": f'attachment; filename="{sbom_file.name}"',
            "Content-Type": "application/json",
            "X-SBOM-Component": component,
            "X-SBOM-Version": version,
            "X-SBOM-Format": format.upper(),
        }

        return Response(
            content=json.dumps(sbom_data, indent=2),
            media_type="application/json",
            headers=headers,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download SBOM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metadata")
async def get_sbom_metadata(
    version: str = Query(..., description="CRONOS AI version"),
    component: str = Query(..., description="Component name"),
) -> SBOMMetadata:
    """
    Get metadata about an SBOM without downloading the full file.

    Args:
        version: Version identifier
        component: Component name

    Returns:
        Metadata including package counts and vulnerability summary

    Example:
        GET /api/v1/sbom/metadata?version=v1.0.0&component=cronos-ai-engine
    """
    try:
        sbom_dir = get_sbom_directory()
        sbom_file = sbom_dir / version / f"{component}-spdx.json"

        if not sbom_file.exists():
            sbom_file = sbom_dir / version / f"cronos-{component}-spdx.json"

        if not sbom_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"SBOM not found for component '{component}' version '{version}'",
            )

        # Read SBOM file
        with open(sbom_file) as f:
            sbom_data = json.load(f)

        packages = len(sbom_data.get("packages", []))
        generated_at = sbom_data.get("creationInfo", {}).get("created", "")

        # Check for vulnerability data
        vuln_summary = None
        vuln_file = sbom_dir / version / f"{component}-vulnerabilities.json"
        if not vuln_file.exists():
            vuln_file = sbom_dir / version / f"cronos-{component}-vulnerabilities.json"

        if vuln_file.exists():
            with open(vuln_file) as f:
                vuln_data = json.load(f)
                matches = vuln_data.get("matches", [])

                vuln_summary = SBOMVulnerabilitySummary(
                    total=len(matches),
                    critical=sum(
                        1
                        for m in matches
                        if m.get("vulnerability", {}).get("severity") == "Critical"
                    ),
                    high=sum(
                        1
                        for m in matches
                        if m.get("vulnerability", {}).get("severity") == "High"
                    ),
                    medium=sum(
                        1
                        for m in matches
                        if m.get("vulnerability", {}).get("severity") == "Medium"
                    ),
                    low=sum(
                        1
                        for m in matches
                        if m.get("vulnerability", {}).get("severity") == "Low"
                    ),
                )

        return SBOMMetadata(
            component=component,
            version=version,
            format="SPDX 2.3",
            packages=packages,
            vulnerabilities=vuln_summary,
            generated_at=generated_at,
            sbom_file=sbom_file.name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get SBOM metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vulnerabilities")
async def get_sbom_vulnerabilities(
    version: str = Query(..., description="CRONOS AI version"),
    component: str = Query("cronos-ai-platform", description="Component name"),
    severity: Optional[str] = Query(
        None, description="Filter by severity (Critical, High, Medium, Low)"
    ),
):
    """
    Get known vulnerabilities for a specific version and component.

    Args:
        version: Version identifier
        component: Component name
        severity: Optional severity filter

    Returns:
        Vulnerability report with CVE details

    Example:
        GET /api/v1/sbom/vulnerabilities?version=v1.0.0&component=cronos-ai-engine&severity=Critical
    """
    try:
        sbom_dir = get_sbom_directory()
        vuln_file = sbom_dir / version / f"{component}-vulnerabilities.json"

        if not vuln_file.exists():
            vuln_file = sbom_dir / version / f"cronos-{component}-vulnerabilities.json"

        if not vuln_file.exists():
            return {
                "vulnerabilities": [],
                "summary": {
                    "total": 0,
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                },
                "message": "No vulnerability scan results available",
            }

        with open(vuln_file) as f:
            vuln_data = json.load(f)

        matches = vuln_data.get("matches", [])

        # Filter by severity if requested
        if severity:
            matches = [
                m
                for m in matches
                if m.get("vulnerability", {}).get("severity", "").lower()
                == severity.lower()
            ]

        # Build summary
        summary = {
            "total": len(matches),
            "critical": sum(
                1
                for m in matches
                if m.get("vulnerability", {}).get("severity") == "Critical"
            ),
            "high": sum(
                1
                for m in matches
                if m.get("vulnerability", {}).get("severity") == "High"
            ),
            "medium": sum(
                1
                for m in matches
                if m.get("vulnerability", {}).get("severity") == "Medium"
            ),
            "low": sum(
                1
                for m in matches
                if m.get("vulnerability", {}).get("severity") == "Low"
            ),
        }

        # Format vulnerabilities
        vulnerabilities = []
        for match in matches[:100]:  # Limit to 100 for performance
            vuln = match.get("vulnerability", {})
            artifact = match.get("artifact", {})

            vulnerabilities.append(
                {
                    "id": vuln.get("id", "Unknown"),
                    "severity": vuln.get("severity", "Unknown"),
                    "description": vuln.get("description", "")[:200],
                    "package": artifact.get("name", "Unknown"),
                    "version": artifact.get("version", "Unknown"),
                    "fixed_in": match.get("vulnerability", {})
                    .get("fix", {})
                    .get("versions", []),
                    "cvss_score": (
                        vuln.get("cvss", [{}])[0].get("metrics", {}).get("baseScore")
                        if vuln.get("cvss")
                        else None
                    ),
                    "urls": vuln.get("urls", []),
                }
            )

        return {
            "component": component,
            "version": version,
            "summary": summary,
            "vulnerabilities": vulnerabilities,
            "scanned_at": vuln_data.get("descriptor", {}).get("timestamp", ""),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get vulnerabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def sbom_health_check():
    """
    Health check endpoint for SBOM service.

    Returns:
        Health status and available versions

    Example:
        GET /api/v1/sbom/health
    """
    try:
        sbom_dir = get_sbom_directory()
        versions = [
            d.name
            for d in sbom_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        return {
            "status": "healthy",
            "sbom_directory": str(sbom_dir),
            "versions_available": len(versions),
            "latest_version": sorted(versions, reverse=True)[0] if versions else None,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@router.get("/formats")
async def list_supported_formats():
    """
    List supported SBOM formats and standards.

    Returns:
        Information about supported SBOM formats

    Example:
        GET /api/v1/sbom/formats
    """
    return {
        "formats": [
            {
                "name": "SPDX",
                "version": "2.3",
                "description": "Software Package Data Exchange - ISO/IEC 5962 standard",
                "use_case": "Government contracts, general distribution, license compliance",
                "format_identifier": "spdx",
            },
            {
                "name": "CycloneDX",
                "version": "1.5",
                "description": "OWASP CycloneDX - Security-focused SBOM format",
                "use_case": "Security teams, vulnerability management, DevSecOps",
                "format_identifier": "cyclonedx",
            },
        ],
        "compliance": [
            "Executive Order 14028 (EO 14028)",
            "NIST SSDF (Secure Software Development Framework)",
            "EU Cyber Resilience Act",
            "NTIA Minimum Elements",
        ],
        "verification": {"signing": "Cosign (Sigstore)", "slsa_level": "3"},
    }
