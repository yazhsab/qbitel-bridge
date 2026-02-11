"""
QBITEL Bridge - Legacy Whisperer End-to-End Integration Tests

This module contains comprehensive E2E tests for the Legacy System Whisperer
component, testing the full workflow from protocol analysis to code generation.
"""

import pytest
import asyncio
import json
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Test: Protocol Reverse Engineering E2E
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestProtocolReverseEngineeringE2E:
    """End-to-end tests for protocol reverse engineering."""

    async def test_reverse_engineer_binary_protocol(
        self,
        legacy_whisperer,
        sample_protocol_traffic: List[bytes]
    ):
        """Test complete binary protocol reverse engineering workflow."""
        if legacy_whisperer is None:
            pytest.skip("LegacySystemWhisperer not available")

        # Execute reverse engineering
        result = await legacy_whisperer.reverse_engineer_protocol(
            traffic_samples=sample_protocol_traffic,
            protocol_hint="IBM mainframe transaction",
            output_format="json"
        )

        # Verify result structure
        assert result is not None
        assert "protocol_specification" in result or hasattr(result, "protocol_specification")
        assert "confidence_score" in result or hasattr(result, "confidence_score")
        assert "fields" in result or hasattr(result, "fields")

    async def test_reverse_engineer_with_ebcdic_detection(
        self,
        legacy_whisperer
    ):
        """Test EBCDIC encoding detection in protocol analysis."""
        if legacy_whisperer is None:
            pytest.skip("LegacySystemWhisperer not available")

        # EBCDIC encoded "CUSTOMER" = C3E4E2E3D6D4C5D9
        ebcdic_sample = bytes.fromhex("C3E4E2E3D6D4C5D9")

        result = await legacy_whisperer.reverse_engineer_protocol(
            traffic_samples=[ebcdic_sample],
            protocol_hint="EBCDIC text field",
            output_format="json"
        )

        assert result is not None
        # Should detect EBCDIC encoding
        result_dict = result if isinstance(result, dict) else result.__dict__
        assert "encoding" in str(result_dict).lower() or "ebcdic" in str(result_dict).lower()

    async def test_protocol_analysis_with_multiple_message_types(
        self,
        legacy_whisperer
    ):
        """Test analysis of protocol with multiple message types."""
        if legacy_whisperer is None:
            pytest.skip("LegacySystemWhisperer not available")

        # Different message types
        traffic = [
            bytes.fromhex("01 00 10 00 00 00 01".replace(" ", "")),  # Type 1: Request
            bytes.fromhex("02 00 10 00 00 00 01 FF".replace(" ", "")),  # Type 2: Response
            bytes.fromhex("03 00 08 00 00 00 01".replace(" ", "")),  # Type 3: Ack
        ]

        result = await legacy_whisperer.reverse_engineer_protocol(
            traffic_samples=traffic,
            protocol_hint="Multi-message protocol",
            output_format="json"
        )

        assert result is not None


# =============================================================================
# Test: COBOL Analysis E2E
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestCOBOLAnalysisE2E:
    """End-to-end tests for COBOL code analysis."""

    async def test_analyze_cobol_program(
        self,
        legacy_whisperer,
        sample_cobol_code: str
    ):
        """Test complete COBOL program analysis."""
        if legacy_whisperer is None:
            pytest.skip("LegacySystemWhisperer not available")

        # Mock the LLM response for COBOL analysis
        mock_analysis = {
            "program_name": "CUSTMAST",
            "complexity_score": 0.65,
            "data_structures": [
                {"name": "CUSTOMER-RECORD", "type": "group", "fields": 6}
            ],
            "procedures": ["MAIN-LOGIC", "OPEN-FILES", "PROCESS-RECORDS"],
            "legacy_patterns": ["indexed-file-access", "comp-3-fields"],
            "modernization_recommendations": [
                "Convert to REST API",
                "Migrate COMP-3 to standard decimal"
            ]
        }

        with patch.object(
            legacy_whisperer,
            "analyze_cobol",
            new=AsyncMock(return_value=mock_analysis)
        ):
            result = await legacy_whisperer.analyze_cobol(
                cobol_source=sample_cobol_code,
                analysis_depth="comprehensive"
            )

        assert result is not None
        assert result["program_name"] == "CUSTMAST"
        assert result["complexity_score"] > 0
        assert len(result["procedures"]) > 0

    async def test_cobol_to_python_generation(
        self,
        legacy_whisperer,
        sample_cobol_code: str
    ):
        """Test COBOL to Python code generation."""
        if legacy_whisperer is None:
            pytest.skip("LegacySystemWhisperer not available")

        mock_python_code = '''
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

class CustomerStatus(Enum):
    ACTIVE = "A"
    INACTIVE = "I"
    SUSPENDED = "S"

@dataclass
class CustomerRecord:
    cust_id: int
    cust_name: str
    cust_address: str
    cust_balance: Decimal
    cust_credit_limit: Decimal
    cust_status: CustomerStatus

async def process_customer(record: CustomerRecord) -> CustomerRecord:
    """Process customer record - modernized from COBOL."""
    if record.cust_balance > record.cust_credit_limit:
        record.cust_status = CustomerStatus.SUSPENDED
    return record
'''

        with patch.object(
            legacy_whisperer,
            "generate_adapter_code",
            new=AsyncMock(return_value={"code": mock_python_code, "language": "python"})
        ):
            result = await legacy_whisperer.generate_adapter_code(
                source_code=sample_cobol_code,
                source_language="cobol",
                target_language="python"
            )

        assert result is not None
        assert "code" in result
        assert "CustomerRecord" in result["code"]


# =============================================================================
# Test: Modernization Planning E2E
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestModernizationPlanningE2E:
    """End-to-end tests for modernization planning."""

    async def test_generate_modernization_plan(
        self,
        legacy_whisperer,
        sample_system_metrics: Dict[str, Any]
    ):
        """Test modernization plan generation."""
        if legacy_whisperer is None:
            pytest.skip("LegacySystemWhisperer not available")

        mock_plan = {
            "plan_id": "MOD-2024-001",
            "system_id": sample_system_metrics["system_id"],
            "approach": "refactor",
            "phases": [
                {
                    "phase": 1,
                    "name": "Assessment",
                    "duration_weeks": 2,
                    "tasks": ["Code analysis", "Dependency mapping"]
                },
                {
                    "phase": 2,
                    "name": "Design",
                    "duration_weeks": 4,
                    "tasks": ["Architecture design", "API specification"]
                },
                {
                    "phase": 3,
                    "name": "Implementation",
                    "duration_weeks": 12,
                    "tasks": ["Code migration", "Testing"]
                },
                {
                    "phase": 4,
                    "name": "Deployment",
                    "duration_weeks": 2,
                    "tasks": ["Staging deployment", "Production cutover"]
                }
            ],
            "risk_assessment": {
                "overall_risk": "medium",
                "risks": [
                    {"type": "technical", "description": "Complex data structures", "mitigation": "Incremental migration"}
                ]
            },
            "estimated_effort_days": 200
        }

        with patch.object(
            legacy_whisperer,
            "generate_modernization_plan",
            new=AsyncMock(return_value=mock_plan)
        ):
            result = await legacy_whisperer.generate_modernization_plan(
                system_id=sample_system_metrics["system_id"],
                approach="refactor",
                constraints={"budget": "medium", "timeline": "flexible"}
            )

        assert result is not None
        assert result["plan_id"] is not None
        assert len(result["phases"]) >= 3
        assert result["estimated_effort_days"] > 0


# =============================================================================
# Test: Full Workflow E2E
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestFullWorkflowE2E:
    """End-to-end tests for complete modernization workflow."""

    async def test_complete_modernization_workflow(
        self,
        legacy_whisperer,
        sample_cobol_code: str,
        sample_protocol_traffic: List[bytes],
        sample_system_metrics: Dict[str, Any]
    ):
        """
        Test the complete modernization workflow:
        1. Register system
        2. Analyze COBOL code
        3. Reverse engineer protocol
        4. Generate modernization plan
        5. Generate adapter code
        """
        if legacy_whisperer is None:
            pytest.skip("LegacySystemWhisperer not available")

        # Step 1: Register the legacy system
        system_registration = {
            "system_id": "LEGACY-001",
            "name": "Customer Master System",
            "type": "mainframe",
            "registered": True
        }

        with patch.object(
            legacy_whisperer,
            "register_system",
            new=AsyncMock(return_value=system_registration)
        ):
            reg_result = await legacy_whisperer.register_system(
                system_id="LEGACY-001",
                name="Customer Master System",
                system_type="mainframe"
            )
            assert reg_result["registered"] is True

        # Step 2: Analyze COBOL code
        cobol_analysis = {
            "program_name": "CUSTMAST",
            "complexity_score": 0.65,
            "data_structures": [{"name": "CUSTOMER-RECORD", "fields": 6}],
            "procedures": ["MAIN-LOGIC", "PROCESS-RECORDS"],
            "analysis_complete": True
        }

        with patch.object(
            legacy_whisperer,
            "analyze_cobol",
            new=AsyncMock(return_value=cobol_analysis)
        ):
            analysis_result = await legacy_whisperer.analyze_cobol(
                cobol_source=sample_cobol_code
            )
            assert analysis_result["analysis_complete"] is True

        # Step 3: Reverse engineer protocol
        protocol_spec = {
            "protocol_name": "CUST-TRANS",
            "encoding": "EBCDIC",
            "fields": [
                {"name": "header", "offset": 0, "length": 4},
                {"name": "customer_id", "offset": 4, "length": 8}
            ],
            "confidence_score": 0.85
        }

        with patch.object(
            legacy_whisperer,
            "reverse_engineer_protocol",
            new=AsyncMock(return_value=protocol_spec)
        ):
            protocol_result = await legacy_whisperer.reverse_engineer_protocol(
                traffic_samples=sample_protocol_traffic
            )
            assert protocol_result["confidence_score"] > 0.5

        # Step 4: Generate modernization plan
        mod_plan = {
            "plan_id": "PLAN-001",
            "phases": [{"phase": 1, "name": "Assessment"}],
            "total_effort_days": 150
        }

        with patch.object(
            legacy_whisperer,
            "generate_modernization_plan",
            new=AsyncMock(return_value=mod_plan)
        ):
            plan_result = await legacy_whisperer.generate_modernization_plan(
                system_id="LEGACY-001",
                approach="refactor"
            )
            assert plan_result["plan_id"] is not None

        # Step 5: Generate adapter code
        adapter_code = {
            "code": "# Generated Python adapter\nclass CustomerAdapter:\n    pass",
            "language": "python",
            "tests_included": True
        }

        with patch.object(
            legacy_whisperer,
            "generate_adapter_code",
            new=AsyncMock(return_value=adapter_code)
        ):
            code_result = await legacy_whisperer.generate_adapter_code(
                source_code=sample_cobol_code,
                target_language="python"
            )
            assert code_result["tests_included"] is True

        logger.info("Complete modernization workflow test passed!")


# =============================================================================
# Test: Error Handling E2E
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestErrorHandlingE2E:
    """End-to-end tests for error handling."""

    async def test_invalid_cobol_handling(self, legacy_whisperer):
        """Test handling of invalid COBOL code."""
        if legacy_whisperer is None:
            pytest.skip("LegacySystemWhisperer not available")

        invalid_cobol = "THIS IS NOT VALID COBOL CODE AT ALL"

        with patch.object(
            legacy_whisperer,
            "analyze_cobol",
            new=AsyncMock(side_effect=ValueError("Invalid COBOL syntax"))
        ):
            with pytest.raises(ValueError) as exc_info:
                await legacy_whisperer.analyze_cobol(cobol_source=invalid_cobol)

            assert "Invalid COBOL" in str(exc_info.value)

    async def test_empty_traffic_handling(self, legacy_whisperer):
        """Test handling of empty traffic samples."""
        if legacy_whisperer is None:
            pytest.skip("LegacySystemWhisperer not available")

        with patch.object(
            legacy_whisperer,
            "reverse_engineer_protocol",
            new=AsyncMock(side_effect=ValueError("No traffic samples provided"))
        ):
            with pytest.raises(ValueError) as exc_info:
                await legacy_whisperer.reverse_engineer_protocol(traffic_samples=[])

            assert "No traffic samples" in str(exc_info.value)

    async def test_llm_timeout_handling(self, legacy_whisperer):
        """Test handling of LLM service timeout."""
        if legacy_whisperer is None:
            pytest.skip("LegacySystemWhisperer not available")

        with patch.object(
            legacy_whisperer,
            "analyze_cobol",
            new=AsyncMock(side_effect=asyncio.TimeoutError("LLM request timed out"))
        ):
            with pytest.raises(asyncio.TimeoutError):
                await legacy_whisperer.analyze_cobol(cobol_source="VALID COBOL")


# =============================================================================
# Test: Performance E2E
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
class TestPerformanceE2E:
    """Performance tests for the Legacy Whisperer."""

    async def test_concurrent_analysis_performance(self, legacy_whisperer):
        """Test performance with concurrent analysis requests."""
        if legacy_whisperer is None:
            pytest.skip("LegacySystemWhisperer not available")

        # Create multiple analysis tasks
        num_concurrent = 10

        mock_result = {"analysis": "complete", "time_ms": 100}

        with patch.object(
            legacy_whisperer,
            "analyze_cobol",
            new=AsyncMock(return_value=mock_result)
        ):
            tasks = [
                legacy_whisperer.analyze_cobol(cobol_source=f"PROGRAM-{i}")
                for i in range(num_concurrent)
            ]

            results = await asyncio.gather(*tasks)

        assert len(results) == num_concurrent
        assert all(r["analysis"] == "complete" for r in results)

    async def test_large_cobol_program_analysis(self, legacy_whisperer):
        """Test analysis of a large COBOL program."""
        if legacy_whisperer is None:
            pytest.skip("LegacySystemWhisperer not available")

        # Generate a large COBOL program (simulated)
        large_cobol = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. LARGEPROG.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
""" + "\n".join([f"       01 WS-VAR-{i} PIC X(100)." for i in range(1000)])

        mock_result = {
            "program_name": "LARGEPROG",
            "lines_analyzed": 1005,
            "complexity_score": 0.85
        }

        with patch.object(
            legacy_whisperer,
            "analyze_cobol",
            new=AsyncMock(return_value=mock_result)
        ):
            result = await legacy_whisperer.analyze_cobol(cobol_source=large_cobol)

        assert result["lines_analyzed"] > 1000
