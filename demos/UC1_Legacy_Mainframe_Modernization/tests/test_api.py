"""
API Endpoint Tests for UC1 Legacy Mainframe Modernization Demo.

Tests cover:
- Health check endpoints
- System registration endpoints
- COBOL analysis endpoints
- Protocol analysis endpoints
- Modernization planning endpoints
- Code generation endpoints
- Knowledge capture endpoints
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, test_client):
        """Test main health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "components" in data
        assert "uptime_seconds" in data

    def test_readiness_probe(self, test_client):
        """Test Kubernetes readiness probe."""
        response = test_client.get("/health/ready")
        # May return 200 or 503 depending on initialization state
        assert response.status_code in [200, 503]

    def test_liveness_probe(self, test_client):
        """Test Kubernetes liveness probe."""
        response = test_client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_metrics_endpoint(self, test_client):
        """Test Prometheus metrics endpoint."""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")


# =============================================================================
# System Registration Tests
# =============================================================================

class TestSystemRegistration:
    """Tests for system registration endpoints."""

    def test_register_system_success(self, test_client, sample_system_registration, auth_headers):
        """Test successful system registration."""
        response = test_client.post(
            "/api/v1/systems",
            json=sample_system_registration,
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert "system_id" in data
        assert data["system_name"] == sample_system_registration["system_name"]
        assert data["status"] == "registered"
        assert "capabilities_enabled" in data
        assert len(data["capabilities_enabled"]) > 0

    def test_register_system_minimal(self, test_client, auth_headers):
        """Test registration with minimal required fields."""
        response = test_client.post(
            "/api/v1/systems",
            json={
                "system_name": "Minimal System",
                "system_type": "cobol_system",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

    def test_register_system_invalid_name(self, test_client, auth_headers):
        """Test registration with invalid system name."""
        response = test_client.post(
            "/api/v1/systems",
            json={
                "system_name": "",  # Empty name
                "system_type": "mainframe",
            },
            headers=auth_headers,
        )
        assert response.status_code == 422  # Validation error

    def test_list_systems(self, test_client, auth_headers):
        """Test listing registered systems."""
        response = test_client.get("/api/v1/systems", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "systems" in data
        assert "count" in data
        assert isinstance(data["systems"], list)

    def test_get_system_not_found(self, test_client, auth_headers):
        """Test getting non-existent system."""
        response = test_client.get(
            "/api/v1/systems/non-existent-id",
            headers=auth_headers,
        )
        assert response.status_code == 404


# =============================================================================
# COBOL Analysis Tests
# =============================================================================

class TestCOBOLAnalysis:
    """Tests for COBOL analysis endpoints."""

    def test_analyze_cobol_success(self, test_client, sample_cobol_source, auth_headers):
        """Test successful COBOL analysis."""
        response = test_client.post(
            "/api/v1/analyze/cobol",
            json={
                "source_code": sample_cobol_source,
                "analysis_depth": "standard",
            },
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "analysis_id" in data
        assert "program_name" in data
        assert "divisions" in data
        assert "complexity_score" in data
        assert "lines_of_code" in data
        assert "data_structures" in data
        assert "procedures" in data
        assert "modernization_recommendations" in data

    def test_analyze_cobol_extracts_program_id(self, test_client, sample_cobol_source, auth_headers):
        """Test COBOL analysis extracts PROGRAM-ID."""
        response = test_client.post(
            "/api/v1/analyze/cobol",
            json={"source_code": sample_cobol_source},
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["program_name"] == "TESTPROG"

    def test_analyze_cobol_identifies_divisions(self, test_client, sample_cobol_source, auth_headers):
        """Test COBOL analysis identifies all divisions."""
        response = test_client.post(
            "/api/v1/analyze/cobol",
            json={"source_code": sample_cobol_source},
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        divisions = data["divisions"]
        # Should identify at least IDENTIFICATION, DATA, PROCEDURE divisions
        assert len(divisions) >= 3

    def test_analyze_cobol_too_short(self, test_client, auth_headers):
        """Test COBOL analysis with too short source."""
        response = test_client.post(
            "/api/v1/analyze/cobol",
            json={"source_code": "short"},
            headers=auth_headers,
        )
        assert response.status_code == 422  # Validation error

    def test_analyze_cobol_deep_analysis(self, test_client, sample_cobol_source, auth_headers):
        """Test COBOL deep analysis mode."""
        response = test_client.post(
            "/api/v1/analyze/cobol",
            json={
                "source_code": sample_cobol_source,
                "analysis_depth": "deep",
            },
            headers=auth_headers,
        )
        assert response.status_code == 200


# =============================================================================
# Protocol Analysis Tests
# =============================================================================

class TestProtocolAnalysis:
    """Tests for protocol analysis endpoints."""

    def test_analyze_protocol_success(self, test_client, sample_protocol_samples, auth_headers):
        """Test successful protocol analysis."""
        response = test_client.post(
            "/api/v1/analyze/protocol",
            json={
                "samples": sample_protocol_samples,
                "system_context": "Test mainframe communication protocol",
            },
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "analysis_id" in data
        assert "protocol_name" in data
        assert "complexity" in data
        assert "fields" in data
        assert "characteristics" in data
        assert "confidence_score" in data

    def test_analyze_protocol_identifies_patterns(self, test_client, sample_protocol_samples, auth_headers):
        """Test protocol analysis identifies patterns."""
        response = test_client.post(
            "/api/v1/analyze/protocol",
            json={"samples": sample_protocol_samples},
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "patterns" in data
        # Should identify magic number pattern (01020304)
        pattern_types = [p.get("pattern_type") for p in data["patterns"]]
        assert "magic_number" in pattern_types or len(data["patterns"]) > 0

    def test_analyze_protocol_invalid_hex(self, test_client, auth_headers):
        """Test protocol analysis with invalid hex data."""
        response = test_client.post(
            "/api/v1/analyze/protocol",
            json={
                "samples": ["not-valid-hex", "also-invalid"],
            },
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_analyze_protocol_empty_samples(self, test_client, auth_headers):
        """Test protocol analysis with empty samples."""
        response = test_client.post(
            "/api/v1/analyze/protocol",
            json={"samples": []},
            headers=auth_headers,
        )
        assert response.status_code == 422  # Validation error


# =============================================================================
# Modernization Planning Tests
# =============================================================================

class TestModernizationPlanning:
    """Tests for modernization planning endpoints."""

    def test_create_plan_success(self, test_client, sample_modernization_request, auth_headers):
        """Test successful modernization plan creation."""
        response = test_client.post(
            "/api/v1/modernize",
            json=sample_modernization_request,
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "plan_id" in data
        assert "system_id" in data
        assert "target_technology" in data
        assert "phases" in data
        assert "risk_assessment" in data
        assert "estimated_effort" in data
        assert "recommendations" in data

    def test_create_plan_has_phases(self, test_client, sample_modernization_request, auth_headers):
        """Test modernization plan has multiple phases."""
        response = test_client.post(
            "/api/v1/modernize",
            json=sample_modernization_request,
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert len(data["phases"]) >= 3  # Should have at least assessment, design, implementation

    def test_create_plan_with_objectives(self, test_client, auth_headers):
        """Test modernization plan with specific objectives."""
        response = test_client.post(
            "/api/v1/modernize",
            json={
                "system_id": "test-123",
                "target_technology": "python-fastapi",
                "objectives": ["api_first", "cloud_native", "containerization"],
            },
            headers=auth_headers,
        )
        assert response.status_code == 200


# =============================================================================
# Code Generation Tests
# =============================================================================

class TestCodeGeneration:
    """Tests for code generation endpoints."""

    def test_generate_code_success(self, test_client, auth_headers):
        """Test successful code generation."""
        response = test_client.post(
            "/api/v1/generate/code",
            json={
                "source_specification": {
                    "protocol_name": "LegacyProtocol",
                    "description": "Test legacy protocol",
                },
                "target_language": "python",
                "target_protocol": "REST",
                "include_tests": True,
            },
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "generation_id" in data
        assert "adapter_code" in data
        assert "documentation" in data
        assert "dependencies" in data
        assert "quality_score" in data

    def test_generate_code_with_tests(self, test_client, auth_headers):
        """Test code generation includes tests."""
        response = test_client.post(
            "/api/v1/generate/code",
            json={
                "source_specification": {"protocol_name": "TestProto"},
                "target_language": "python",
                "target_protocol": "REST",
                "include_tests": True,
            },
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "test_code" in data
        assert data["test_code"] is not None

    def test_generate_code_without_tests(self, test_client, auth_headers):
        """Test code generation without tests."""
        response = test_client.post(
            "/api/v1/generate/code",
            json={
                "source_specification": {"protocol_name": "TestProto"},
                "target_language": "python",
                "target_protocol": "REST",
                "include_tests": False,
            },
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert data.get("test_code") is None


# =============================================================================
# Knowledge Capture Tests
# =============================================================================

class TestKnowledgeCapture:
    """Tests for knowledge capture endpoints."""

    def test_capture_knowledge_success(self, test_client, sample_knowledge_capture, auth_headers):
        """Test successful knowledge capture."""
        response = test_client.post(
            "/api/v1/knowledge/capture",
            json=sample_knowledge_capture,
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "session_id" in data
        assert "knowledge_id" in data
        assert "formalized_knowledge" in data
        assert "confidence_score" in data
        assert data["status"] == "captured"

    def test_capture_knowledge_minimal(self, test_client, auth_headers):
        """Test knowledge capture with minimal data."""
        response = test_client.post(
            "/api/v1/knowledge/capture",
            json={
                "expert_id": "expert-001",
                "knowledge_input": "This is important knowledge about the system behavior.",
            },
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_capture_knowledge_too_short(self, test_client, auth_headers):
        """Test knowledge capture with too short input."""
        response = test_client.post(
            "/api/v1/knowledge/capture",
            json={
                "expert_id": "expert-001",
                "knowledge_input": "short",  # Too short
            },
            headers=auth_headers,
        )
        assert response.status_code == 422


# =============================================================================
# System Health Tests
# =============================================================================

class TestSystemHealth:
    """Tests for system health analysis endpoints."""

    def test_analyze_health_success(self, test_client, auth_headers):
        """Test successful health analysis."""
        # First register a system
        reg_response = test_client.post(
            "/api/v1/systems",
            json={
                "system_name": "Health Test System",
                "system_type": "mainframe",
            },
            headers=auth_headers,
        )
        system_id = reg_response.json()["system_id"]

        # Then analyze health
        response = test_client.post(
            f"/api/v1/systems/{system_id}/health",
            json={
                "system_id": system_id,
                "metrics": {
                    "cpu_utilization": 45.5,
                    "memory_utilization": 62.3,
                    "disk_utilization": 38.0,
                    "response_time_ms": 120.0,
                    "error_rate": 0.01,
                },
                "prediction_horizon": "medium_term",
            },
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "system_id" in data
        assert "health_score" in data
        assert "analysis_results" in data


# =============================================================================
# Service Status Tests
# =============================================================================

class TestServiceStatus:
    """Tests for service status endpoint."""

    def test_get_service_status(self, test_client, auth_headers):
        """Test service status endpoint."""
        response = test_client.get("/api/v1/status", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "environment" in data
        assert "uptime_seconds" in data
