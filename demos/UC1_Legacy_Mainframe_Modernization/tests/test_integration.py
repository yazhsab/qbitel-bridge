"""
Integration tests for UC1 Legacy Mainframe Modernization Demo.

These tests verify the complete flow from API to service layer
and validate production module integration.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


class TestFullWorkflow:
    """Test complete workflows end-to-end."""

    def test_register_and_analyze_workflow(self, test_client, sample_cobol_source, auth_headers):
        """Test complete workflow: register system, then analyze COBOL."""
        # Step 1: Register a system
        reg_response = test_client.post(
            "/api/v1/systems",
            json={
                "system_name": "Integration Test Mainframe",
                "system_type": "mainframe",
                "manufacturer": "IBM",
                "criticality": "high",
            },
            headers=auth_headers,
        )
        assert reg_response.status_code == 201
        system_id = reg_response.json()["system_id"]

        # Step 2: Analyze COBOL with system context
        analysis_response = test_client.post(
            "/api/v1/analyze/cobol",
            json={
                "source_code": sample_cobol_source,
                "system_id": system_id,
                "analysis_depth": "standard",
            },
            headers=auth_headers,
        )
        assert analysis_response.status_code == 200
        analysis_data = analysis_response.json()
        assert "analysis_id" in analysis_data
        assert "program_name" in analysis_data

        # Step 3: Get system status
        system_response = test_client.get(
            f"/api/v1/systems/{system_id}",
            headers=auth_headers,
        )
        assert system_response.status_code == 200

    def test_analyze_and_modernize_workflow(self, test_client, sample_cobol_source, auth_headers):
        """Test workflow: analyze COBOL, then create modernization plan."""
        # Step 1: Register system
        reg_response = test_client.post(
            "/api/v1/systems",
            json={
                "system_name": "Modernization Test System",
                "system_type": "cobol_system",
            },
            headers=auth_headers,
        )
        system_id = reg_response.json()["system_id"]

        # Step 2: Analyze COBOL
        analysis_response = test_client.post(
            "/api/v1/analyze/cobol",
            json={
                "source_code": sample_cobol_source,
                "system_id": system_id,
            },
            headers=auth_headers,
        )
        assert analysis_response.status_code == 200

        # Step 3: Create modernization plan
        plan_response = test_client.post(
            "/api/v1/modernize",
            json={
                "system_id": system_id,
                "target_technology": "python-fastapi",
                "include_code_generation": True,
                "objectives": ["api_first", "maintainability"],
            },
            headers=auth_headers,
        )
        assert plan_response.status_code == 200
        plan_data = plan_response.json()
        assert "plan_id" in plan_data
        assert "phases" in plan_data
        assert len(plan_data["phases"]) >= 3

    def test_protocol_to_code_generation_workflow(self, test_client, sample_protocol_samples, auth_headers):
        """Test workflow: analyze protocol, then generate adapter code."""
        # Step 1: Analyze protocol
        analysis_response = test_client.post(
            "/api/v1/analyze/protocol",
            json={
                "samples": sample_protocol_samples,
                "system_context": "Legacy mainframe communication",
            },
            headers=auth_headers,
        )
        assert analysis_response.status_code == 200
        protocol_data = analysis_response.json()

        # Step 2: Generate adapter code
        code_response = test_client.post(
            "/api/v1/generate/code",
            json={
                "source_specification": {
                    "protocol_name": protocol_data["protocol_name"],
                    "complexity": protocol_data["complexity"],
                    "fields": protocol_data["fields"],
                },
                "target_language": "python",
                "target_protocol": "REST",
                "include_tests": True,
            },
            headers=auth_headers,
        )
        assert code_response.status_code == 200
        code_data = code_response.json()
        assert "adapter_code" in code_data
        assert len(code_data["adapter_code"]) > 100

    def test_knowledge_capture_workflow(self, test_client, sample_knowledge_capture, auth_headers):
        """Test workflow: register system, capture knowledge."""
        # Step 1: Register system
        reg_response = test_client.post(
            "/api/v1/systems",
            json={
                "system_name": "Knowledge Test System",
                "system_type": "mainframe",
            },
            headers=auth_headers,
        )
        system_id = reg_response.json()["system_id"]

        # Step 2: Capture expert knowledge
        knowledge_data = sample_knowledge_capture.copy()
        knowledge_data["system_id"] = system_id

        capture_response = test_client.post(
            "/api/v1/knowledge/capture",
            json=knowledge_data,
            headers=auth_headers,
        )
        assert capture_response.status_code == 200
        captured = capture_response.json()
        assert captured["status"] == "captured"
        assert "knowledge_id" in captured


class TestErrorRecovery:
    """Test error handling and recovery scenarios."""

    def test_invalid_system_id_handling(self, test_client, auth_headers):
        """Test handling of invalid system ID in health check."""
        response = test_client.post(
            "/api/v1/systems/invalid-system-id/health",
            json={
                "system_id": "invalid-system-id",
                "metrics": {"cpu_utilization": 50.0},
            },
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_malformed_cobol_handling(self, test_client, auth_headers):
        """Test handling of malformed COBOL code."""
        response = test_client.post(
            "/api/v1/analyze/cobol",
            json={
                "source_code": "This is not COBOL code at all, just random text " * 10,
            },
            headers=auth_headers,
        )
        # Should still return 200 but with UNKNOWN program name
        assert response.status_code == 200
        data = response.json()
        assert data["program_name"] == "UNKNOWN"

    def test_invalid_protocol_samples_handling(self, test_client, auth_headers):
        """Test handling of invalid protocol samples."""
        response = test_client.post(
            "/api/v1/analyze/protocol",
            json={
                "samples": ["GHIJ", "KLMN"],  # Invalid hex (G-N not valid hex chars)
            },
            headers=auth_headers,
        )
        assert response.status_code == 400


class TestConcurrentRequests:
    """Test handling of concurrent requests."""

    def test_multiple_cobol_analyses(self, test_client, sample_cobol_source, auth_headers):
        """Test multiple concurrent COBOL analyses."""
        import concurrent.futures

        def analyze_cobol(suffix):
            return test_client.post(
                "/api/v1/analyze/cobol",
                json={"source_code": sample_cobol_source},
                headers=auth_headers,
            )

        # Run 5 concurrent analyses
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(analyze_cobol, i) for i in range(5)]
            results = [f.result() for f in futures]

        # All should succeed
        for result in results:
            assert result.status_code == 200

    def test_multiple_system_registrations(self, test_client, auth_headers):
        """Test multiple concurrent system registrations."""
        import concurrent.futures

        def register_system(idx):
            return test_client.post(
                "/api/v1/systems",
                json={
                    "system_name": f"Concurrent System {idx}",
                    "system_type": "mainframe",
                },
                headers=auth_headers,
            )

        # Run 10 concurrent registrations
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(register_system, i) for i in range(10)]
            results = [f.result() for f in futures]

        # All should succeed
        for result in results:
            assert result.status_code == 201

        # All should have unique IDs
        ids = [r.json()["system_id"] for r in results]
        assert len(ids) == len(set(ids))


class TestServiceIntegration:
    """Test integration with production services (mocked)."""

    @pytest.mark.asyncio
    async def test_llm_service_integration(self, mock_llm_service):
        """Test LLM service integration."""
        # Verify mock is properly configured
        response = await mock_llm_service.process_request(MagicMock())
        assert response.content is not None
        assert response.parsed_response is not None

    @pytest.mark.asyncio
    async def test_legacy_whisperer_integration(self, mock_legacy_whisperer):
        """Test Legacy Whisperer integration."""
        # Test protocol analysis
        spec = await mock_legacy_whisperer.reverse_engineer_protocol(
            traffic_samples=[b"test"],
            system_context="test",
        )
        assert spec.protocol_name is not None

        # Test adapter generation
        adapter = await mock_legacy_whisperer.generate_adapter_code(
            legacy_protocol=spec,
            target_protocol="REST",
        )
        assert adapter.adapter_code is not None


class TestHealthMonitoring:
    """Test health monitoring functionality."""

    def test_health_endpoints_respond(self, test_client):
        """Test all health endpoints respond correctly."""
        endpoints = ["/health", "/health/ready", "/health/live"]

        for endpoint in endpoints:
            response = test_client.get(endpoint)
            # Should return either 200 or 503 (not ready), not error
            assert response.status_code in [200, 503]

    def test_metrics_collection(self, test_client, sample_cobol_source, auth_headers):
        """Test that metrics are collected after operations."""
        # Perform some operations
        test_client.post(
            "/api/v1/analyze/cobol",
            json={"source_code": sample_cobol_source},
            headers=auth_headers,
        )

        # Check metrics
        response = test_client.get("/metrics")
        assert response.status_code == 200
        metrics_text = response.text

        # Should contain request metrics
        assert "uc1_demo_requests_total" in metrics_text

    def test_service_status_details(self, test_client, auth_headers):
        """Test service status provides detailed information."""
        response = test_client.get("/api/v1/status", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "environment" in data
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestDataPersistence:
    """Test data persistence functionality."""

    def test_system_persists_after_registration(self, test_client, auth_headers):
        """Test that registered systems persist."""
        # Register system
        reg_response = test_client.post(
            "/api/v1/systems",
            json={
                "system_name": "Persistence Test",
                "system_type": "mainframe",
            },
            headers=auth_headers,
        )
        system_id = reg_response.json()["system_id"]

        # Verify it appears in list
        list_response = test_client.get("/api/v1/systems", headers=auth_headers)
        systems = list_response.json()["systems"]

        found = any(s.get("system_id") == system_id for s in systems)
        assert found, f"System {system_id} not found in list"

    def test_multiple_systems_persist(self, test_client, auth_headers):
        """Test that multiple systems persist correctly."""
        system_ids = []

        # Register 3 systems
        for i in range(3):
            reg_response = test_client.post(
                "/api/v1/systems",
                json={
                    "system_name": f"Multi Persist Test {i}",
                    "system_type": "cobol_system",
                },
                headers=auth_headers,
            )
            system_ids.append(reg_response.json()["system_id"])

        # Verify all appear in list
        list_response = test_client.get("/api/v1/systems", headers=auth_headers)
        listed_ids = [s.get("system_id") for s in list_response.json()["systems"]]

        for sid in system_ids:
            assert sid in listed_ids, f"System {sid} not found"
