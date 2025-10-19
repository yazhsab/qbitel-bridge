"""
CRONOS AI - SBOM API Tests
Comprehensive test suite for SBOM endpoints.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from fastapi import HTTPException
from fastapi.testclient import TestClient


@pytest.fixture
def sbom_test_data(tmp_path):
    """Create test SBOM data and directory structure."""
    # Create SBOM directory structure
    sbom_dir = tmp_path / "sbom-artifacts"
    v1_dir = sbom_dir / "v1.0.0"
    v1_dir.mkdir(parents=True)

    # Create test SBOM file (SPDX format)
    spdx_sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "cronos-ai-platform-v1.0.0",
        "documentNamespace": "https://sbom.cronos-ai.com/v1.0.0/platform",
        "creationInfo": {
            "created": "2025-10-19T14:30:00Z",
            "creators": ["Tool: syft-0.100.0"],
            "licenseListVersion": "3.21"
        },
        "packages": [
            {
                "SPDXID": "SPDXRef-Package-fastapi",
                "name": "fastapi",
                "versionInfo": "0.104.1",
                "supplier": "Organization: FastAPI",
                "licenseConcluded": "MIT",
                "downloadLocation": "https://pypi.org/project/fastapi/0.104.1"
            },
            {
                "SPDXID": "SPDXRef-Package-pydantic",
                "name": "pydantic",
                "versionInfo": "2.5.0",
                "supplier": "Organization: Pydantic",
                "licenseConcluded": "MIT",
                "downloadLocation": "https://pypi.org/project/pydantic/2.5.0"
            }
        ]
    }

    spdx_file = v1_dir / "cronos-ai-platform-spdx.json"
    with open(spdx_file, 'w') as f:
        json.dump(spdx_sbom, f, indent=2)

    # Create CycloneDX SBOM
    cyclonedx_sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "timestamp": "2025-10-19T14:30:00Z",
            "component": {
                "type": "application",
                "name": "cronos-ai-platform",
                "version": "1.0.0"
            }
        },
        "components": [
            {
                "type": "library",
                "name": "fastapi",
                "version": "0.104.1",
                "purl": "pkg:pypi/fastapi@0.104.1"
            }
        ]
    }

    cyclonedx_file = v1_dir / "cronos-ai-platform-cyclonedx.json"
    with open(cyclonedx_file, 'w') as f:
        json.dump(cyclonedx_sbom, f, indent=2)

    # Create vulnerability report
    vuln_report = {
        "matches": [
            {
                "vulnerability": {
                    "id": "CVE-2024-1234",
                    "severity": "Critical",
                    "description": "Test critical vulnerability",
                    "cvss": [{"metrics": {"baseScore": 9.8}}]
                },
                "artifact": {
                    "name": "fastapi",
                    "version": "0.104.1"
                }
            },
            {
                "vulnerability": {
                    "id": "CVE-2024-5678",
                    "severity": "High",
                    "description": "Test high vulnerability",
                    "cvss": [{"metrics": {"baseScore": 7.5}}]
                },
                "artifact": {
                    "name": "pydantic",
                    "version": "2.5.0"
                }
            }
        ],
        "descriptor": {
            "timestamp": "2025-10-19T15:00:00Z"
        }
    }

    vuln_file = v1_dir / "cronos-ai-platform-vulnerabilities.json"
    with open(vuln_file, 'w') as f:
        json.dump(vuln_report, f, indent=2)

    # Create SBOM summary
    summary = {
        "generated_at": "2025-10-19T14:30:00Z",
        "version": "v1.0.0",
        "commit": "abc123def456",
        "components": {
            "cronos-ai-platform": {
                "packages": 2,
                "sbom_file": "cronos-ai-platform-spdx.json",
                "format": "SPDX 2.3",
                "vulnerabilities": {
                    "total": 2,
                    "critical": 1,
                    "high": 1,
                    "medium": 0,
                    "low": 0
                }
            }
        }
    }

    summary_file = v1_dir / "sbom-summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    return sbom_dir


@pytest.fixture
def sbom_api():
    """Import SBOM API module."""
    from ai_engine.api import sbom
    return sbom


class TestSBOMVersionEndpoints:
    """Test SBOM version listing and metadata endpoints."""

    def test_list_sbom_versions(self, sbom_api, sbom_test_data, monkeypatch):
        """Test listing available SBOM versions."""
        monkeypatch.setattr(sbom_api, "SBOM_FALLBACK_DIR", sbom_test_data)

        import asyncio
        versions = asyncio.run(sbom_api.list_sbom_versions())

        assert "v1.0.0" in versions
        assert len(versions) >= 1

    def test_get_version_metadata(self, sbom_api, sbom_test_data, monkeypatch):
        """Test getting metadata for specific version."""
        monkeypatch.setattr(sbom_api, "SBOM_FALLBACK_DIR", sbom_test_data)

        import asyncio
        metadata = asyncio.run(sbom_api.get_version_metadata("v1.0.0"))

        assert metadata.version == "v1.0.0"
        assert metadata.commit == "abc123def456"
        assert "cronos-ai-platform" in metadata.components

    def test_get_version_metadata_not_found(self, sbom_api, sbom_test_data, monkeypatch):
        """Test getting metadata for non-existent version."""
        monkeypatch.setattr(sbom_api, "SBOM_FALLBACK_DIR", sbom_test_data)

        import asyncio
        with pytest.raises(HTTPException) as exc:
            asyncio.run(sbom_api.get_version_metadata("v99.99.99"))

        assert exc.value.status_code == 404


class TestSBOMDownloadEndpoint:
    """Test SBOM download functionality."""

    def test_download_sbom_spdx(self, sbom_api, sbom_test_data, monkeypatch):
        """Test downloading SPDX format SBOM."""
        monkeypatch.setattr(sbom_api, "SBOM_FALLBACK_DIR", sbom_test_data)

        import asyncio
        response = asyncio.run(sbom_api.download_sbom(
            version="v1.0.0",
            component="cronos-ai-platform",
            format="spdx"
        ))

        assert response is not None
        assert "cronos-ai-platform-spdx.json" in response.headers.get("Content-Disposition", "")

    def test_download_sbom_cyclonedx(self, sbom_api, sbom_test_data, monkeypatch):
        """Test downloading CycloneDX format SBOM."""
        monkeypatch.setattr(sbom_api, "SBOM_FALLBACK_DIR", sbom_test_data)

        import asyncio
        response = asyncio.run(sbom_api.download_sbom(
            version="v1.0.0",
            component="cronos-ai-platform",
            format="cyclonedx"
        ))

        assert response is not None

    def test_download_sbom_not_found(self, sbom_api, sbom_test_data, monkeypatch):
        """Test downloading non-existent SBOM."""
        monkeypatch.setattr(sbom_api, "SBOM_FALLBACK_DIR", sbom_test_data)

        import asyncio
        with pytest.raises(HTTPException) as exc:
            asyncio.run(sbom_api.download_sbom(
                version="v1.0.0",
                component="non-existent-component",
                format="spdx"
            ))

        assert exc.value.status_code == 404


class TestSBOMMetadataEndpoint:
    """Test SBOM metadata retrieval."""

    def test_get_sbom_metadata(self, sbom_api, sbom_test_data, monkeypatch):
        """Test getting SBOM metadata."""
        monkeypatch.setattr(sbom_api, "SBOM_FALLBACK_DIR", sbom_test_data)

        import asyncio
        metadata = asyncio.run(sbom_api.get_sbom_metadata(
            version="v1.0.0",
            component="cronos-ai-platform"
        ))

        assert metadata.component == "cronos-ai-platform"
        assert metadata.version == "v1.0.0"
        assert metadata.packages == 2
        assert metadata.format == "SPDX 2.3"
        assert metadata.vulnerabilities is not None
        assert metadata.vulnerabilities.total == 2
        assert metadata.vulnerabilities.critical == 1


class TestSBOMVulnerabilitiesEndpoint:
    """Test vulnerability reporting endpoints."""

    def test_get_sbom_vulnerabilities(self, sbom_api, sbom_test_data, monkeypatch):
        """Test getting vulnerability report."""
        monkeypatch.setattr(sbom_api, "SBOM_FALLBACK_DIR", sbom_test_data)

        import asyncio
        result = asyncio.run(sbom_api.get_sbom_vulnerabilities(
            version="v1.0.0",
            component="cronos-ai-platform",
            severity=None
        ))

        assert result["component"] == "cronos-ai-platform"
        assert result["summary"]["total"] == 2
        assert result["summary"]["critical"] == 1
        assert result["summary"]["high"] == 1
        assert len(result["vulnerabilities"]) == 2

    def test_get_sbom_vulnerabilities_filtered(self, sbom_api, sbom_test_data, monkeypatch):
        """Test getting filtered vulnerability report."""
        monkeypatch.setattr(sbom_api, "SBOM_FALLBACK_DIR", sbom_test_data)

        import asyncio
        result = asyncio.run(sbom_api.get_sbom_vulnerabilities(
            version="v1.0.0",
            component="cronos-ai-platform",
            severity="Critical"
        ))

        assert len(result["vulnerabilities"]) == 1
        assert result["vulnerabilities"][0]["severity"] == "Critical"

    def test_get_sbom_vulnerabilities_no_data(self, sbom_api, sbom_test_data, monkeypatch):
        """Test getting vulnerabilities when no scan data exists."""
        monkeypatch.setattr(sbom_api, "SBOM_FALLBACK_DIR", sbom_test_data)

        import asyncio
        result = asyncio.run(sbom_api.get_sbom_vulnerabilities(
            version="v1.0.0",
            component="non-existent"
        ))

        assert result["summary"]["total"] == 0
        assert len(result["vulnerabilities"]) == 0


class TestSBOMHealthEndpoint:
    """Test SBOM service health checks."""

    def test_sbom_health_check(self, sbom_api, sbom_test_data, monkeypatch):
        """Test SBOM health check endpoint."""
        monkeypatch.setattr(sbom_api, "SBOM_FALLBACK_DIR", sbom_test_data)

        import asyncio
        health = asyncio.run(sbom_api.sbom_health_check())

        assert health["status"] == "healthy"
        assert health["versions_available"] >= 1
        assert health["latest_version"] == "v1.0.0"


class TestSBOMFormatsEndpoint:
    """Test supported formats listing."""

    def test_list_supported_formats(self, sbom_api):
        """Test listing supported SBOM formats."""
        import asyncio
        formats = asyncio.run(sbom_api.list_supported_formats())

        assert len(formats["formats"]) == 2
        assert any(f["name"] == "SPDX" for f in formats["formats"])
        assert any(f["name"] == "CycloneDX" for f in formats["formats"])
        assert "Executive Order 14028 (EO 14028)" in formats["compliance"]
        assert formats["verification"]["signing"] == "Cosign (Sigstore)"


class TestSBOMMetricsCollector:
    """Test SBOM metrics collection."""

    def test_record_sbom_generation_success(self):
        """Test recording successful SBOM generation."""
        from ai_engine.monitoring.sbom_metrics import SBOMMetricsCollector

        collector = SBOMMetricsCollector()
        collector.record_sbom_generation(
            component_name="test-component",
            format="spdx",
            duration=1.5,
            success=True
        )

        # Metrics should be recorded without errors
        assert True

    def test_record_sbom_generation_failure(self):
        """Test recording failed SBOM generation."""
        from ai_engine.monitoring.sbom_metrics import SBOMMetricsCollector

        collector = SBOMMetricsCollector()
        collector.record_sbom_generation(
            component_name="test-component",
            format="spdx",
            duration=0.5,
            success=False,
            error_type="tool_failure"
        )

        # Metrics should be recorded without errors
        assert True

    def test_update_vulnerability_metrics(self):
        """Test updating vulnerability metrics."""
        from ai_engine.monitoring.sbom_metrics import SBOMMetricsCollector

        collector = SBOMMetricsCollector()
        collector.update_vulnerability_metrics(
            version="v1.0.0",
            component_name="test-component",
            vulnerabilities={
                "critical": 1,
                "high": 2,
                "medium": 3,
                "low": 4
            }
        )

        # Metrics should be recorded without errors
        assert True

    def test_record_sbom_download(self):
        """Test recording SBOM downloads."""
        from ai_engine.monitoring.sbom_metrics import SBOMMetricsCollector

        collector = SBOMMetricsCollector()
        collector.record_sbom_download(
            version="v1.0.0",
            component_name="test-component",
            format="spdx"
        )

        # Metrics should be recorded without errors
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
