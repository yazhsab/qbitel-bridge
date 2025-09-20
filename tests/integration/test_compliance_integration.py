"""
CRONOS AI - Compliance Integration Tests

End-to-end integration tests for the complete compliance reporting system.
Tests the full workflow from assessment to report generation with real components.
"""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from ai_engine.compliance.compliance_service import ComplianceService, ComplianceServiceConfig
from ai_engine.compliance.regulatory_kb import RegulatoryKnowledgeBase
from ai_engine.compliance.assessment_engine import ComplianceAssessmentEngine
from ai_engine.compliance.report_generator import AutomatedReportGenerator, ReportFormat, ReportType
from ai_engine.compliance.audit_trail import AuditTrailManager, EventType, EventSeverity
from ai_engine.compliance.prompt_templates import CompliancePromptManager
from ai_engine.core.config import Config
from ai_engine.llm.unified_llm_service import UnifiedLLMService, LLMRequest, LLMResponse

@pytest.fixture(scope="session")
def event_loop():
    """Event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture
async def test_config():
    """Test configuration."""
    config = Mock(spec=Config)
    config.environment = "testing"
    config.service_name = "cronos-ai-test"
    config.version = "1.0.0"
    
    # Compliance configuration
    config.compliance_enabled = True
    config.compliance_assessment_interval_hours = 1
    config.compliance_cache_ttl_hours = 1
    config.compliance_max_concurrent_assessments = 2
    config.compliance_audit_trail_enabled = True
    config.compliance_blockchain_enabled = True
    
    # Database mock
    config.database = Mock()
    config.database.host = "localhost"
    config.database.port = 5432
    config.database.database = "cronos_ai_test"
    config.database.username = "test"
    config.database.password = "test"
    
    # Redis mock  
    config.redis = Mock()
    config.redis.host = "localhost"
    config.redis.port = 6379
    config.redis.password = None
    config.redis.db = 1  # Use different DB for tests
    
    # Security mock
    config.security = Mock()
    config.security.enable_encryption = True
    config.security.jwt_secret = "test_secret"
    
    return config

@pytest.fixture
async def mock_llm_service():
    """Mock LLM service that returns realistic responses."""
    llm_service = AsyncMock(spec=UnifiedLLMService)
    
    # Mock different types of responses based on request
    def mock_llm_response(request: LLMRequest):
        if "gap_analysis" in request.prompt.lower():
            return LLMResponse(
                content=json.dumps({
                    "compliance_score": 78.5,
                    "compliant_count": 18,
                    "non_compliant_count": 5,
                    "partially_compliant_count": 2,
                    "risk_score": 21.5,
                    "gaps": [
                        {
                            "requirement_id": "1.1",
                            "requirement_title": "Network Security Controls",
                            "severity": "critical",
                            "current_state": "Partially implemented",
                            "required_state": "Fully implemented with monitoring",
                            "gap_description": "Missing automated monitoring of firewall rules",
                            "impact_assessment": "High risk of undetected network intrusions",
                            "remediation_effort": "high"
                        },
                        {
                            "requirement_id": "2.1", 
                            "requirement_title": "Default Password Management",
                            "severity": "high",
                            "current_state": "Some defaults not changed",
                            "required_state": "All defaults changed and documented",
                            "gap_description": "Default credentials found on 3 systems",
                            "impact_assessment": "Medium risk of unauthorized access",
                            "remediation_effort": "medium"
                        }
                    ],
                    "recommendations": [
                        {
                            "id": "REC-001",
                            "title": "Implement Automated Firewall Monitoring",
                            "description": "Deploy automated monitoring solution for firewall rule changes",
                            "priority": "critical",
                            "implementation_steps": [
                                "Select monitoring solution",
                                "Configure rule change detection",
                                "Set up alerting",
                                "Test and validate"
                            ],
                            "estimated_effort_days": 14,
                            "business_impact": "Significantly improves security posture and compliance"
                        }
                    ]
                }),
                provider="mock",
                tokens_used=1500,
                processing_time=2.5,
                confidence=0.9
            )
        elif "requirement" in request.prompt.lower():
            return LLMResponse(
                content=json.dumps({
                    "status": "partially_compliant",
                    "compliance_score": 65.0,
                    "findings": [
                        "Network segmentation is implemented",
                        "Missing automated monitoring",
                        "Documentation needs updates"
                    ],
                    "recommendations": [
                        "Implement automated monitoring",
                        "Update documentation"
                    ],
                    "risk_level": "medium",
                    "evidence_quality": "good",
                    "confidence": 85
                }),
                provider="mock",
                tokens_used=800,
                processing_time=1.5,
                confidence=0.85
            )
        elif "executive" in request.prompt.lower():
            return LLMResponse(
                content="""
                ## Executive Summary
                
                Our PCI-DSS 4.0 compliance assessment reveals an overall compliance score of 78.5%, indicating substantial progress toward full compliance with critical gaps requiring immediate attention.
                
                **Key Findings:**
                - 18 of 25 requirements are fully compliant
                - 2 critical gaps identified in network security monitoring
                - Estimated 90 days to achieve 95%+ compliance
                
                **Strategic Recommendations:**
                1. Immediate implementation of automated firewall monitoring (14 days, $50K investment)
                2. Default credential remediation across remaining systems (7 days, $15K)
                3. Documentation updates and staff training (30 days, $25K)
                
                **Business Impact:**
                Full compliance will reduce regulatory risk by 85% and strengthen our security posture against evolving threats. The $90K investment will prevent potential fines of $500K+ and demonstrate commitment to customer data protection.
                """,
                provider="mock",
                tokens_used=1200,
                processing_time=3.0,
                confidence=0.92
            )
        else:
            # Default technical response
            return LLMResponse(
                content="Technical analysis complete. System shows good compliance posture with some areas for improvement.",
                provider="mock", 
                tokens_used=500,
                processing_time=1.0,
                confidence=0.8
            )
    
    llm_service.process_request.side_effect = mock_llm_response
    return llm_service

@pytest.fixture
async def compliance_service(test_config, mock_llm_service):
    """Fully configured compliance service for testing."""
    service_config = ComplianceServiceConfig(
        assessment_interval_hours=1,
        max_concurrent_assessments=2,
        timescaledb_integration=False,  # Disable for tests
        redis_integration=False,        # Disable for tests
        security_integration=False      # Disable for tests
    )
    
    with patch('ai_engine.compliance.compliance_service.get_llm_service', return_value=mock_llm_service):
        service = ComplianceService(test_config, service_config, mock_llm_service)
        yield service

class TestComplianceEndToEnd:
    """End-to-end compliance system tests."""
    
    @pytest.mark.asyncio
    async def test_full_compliance_workflow(self, compliance_service):
        """Test complete compliance workflow from assessment to reporting."""
        # Start the service
        await compliance_service.start()
        
        try:
            # Step 1: Perform compliance assessment
            assessment = await compliance_service.assess_compliance("PCI_DSS_4_0")
            
            # Verify assessment results
            assert assessment is not None
            assert assessment.framework == "PCI_DSS_4_0"
            assert assessment.overall_compliance_score > 0
            assert len(assessment.gaps) > 0
            assert len(assessment.recommendations) > 0
            
            print(f"Assessment completed with {assessment.overall_compliance_score:.1f}% compliance")
            
            # Step 2: Generate executive summary report
            exec_report_content = await compliance_service.generate_report(
                "PCI_DSS_4_0",
                ReportType.EXECUTIVE_SUMMARY,
                ReportFormat.HTML
            )
            
            assert exec_report_content is not None
            assert len(exec_report_content) > 0
            
            # Verify HTML content
            html_content = exec_report_content.decode('utf-8')
            assert 'PCI_DSS_4_0' in html_content
            assert 'Executive Summary' in html_content
            
            print(f"Executive report generated: {len(html_content)} bytes")
            
            # Step 3: Generate detailed technical report
            tech_report_content = await compliance_service.generate_report(
                "PCI_DSS_4_0", 
                ReportType.DETAILED_TECHNICAL,
                ReportFormat.JSON
            )
            
            assert tech_report_content is not None
            
            # Verify JSON content
            json_data = json.loads(tech_report_content.decode('utf-8'))
            assert 'metadata' in json_data
            assert 'sections' in json_data
            
            print(f"Technical report generated: {len(tech_report_content)} bytes")
            
            # Step 4: Get compliance dashboard
            dashboard = await compliance_service.get_compliance_dashboard(['PCI_DSS_4_0'])
            
            assert 'frameworks' in dashboard
            assert 'PCI_DSS_4_0' in dashboard['frameworks']
            assert dashboard['frameworks']['PCI_DSS_4_0']['compliance_score'] > 0
            
            print(f"Dashboard data: {len(dashboard['frameworks'])} frameworks")
            
        finally:
            await compliance_service.stop()
    
    @pytest.mark.asyncio
    async def test_multi_framework_assessment(self, compliance_service):
        """Test assessment of multiple frameworks simultaneously."""
        await compliance_service.start()
        
        try:
            # Test multiple frameworks
            frameworks = ['PCI_DSS_4_0', 'HIPAA', 'BASEL_III']
            
            # Assess each framework
            assessments = []
            for framework in frameworks:
                assessment = await compliance_service.assess_compliance(framework)
                assessments.append(assessment)
                print(f"{framework}: {assessment.overall_compliance_score:.1f}% compliant")
            
            assert len(assessments) == len(frameworks)
            
            # Verify each assessment
            for assessment in assessments:
                assert assessment.overall_compliance_score > 0
                assert len(assessment.gaps) >= 0
                assert len(assessment.recommendations) >= 0
            
            # Get multi-framework dashboard
            dashboard = await compliance_service.get_compliance_dashboard(frameworks)
            
            assert len(dashboard['frameworks']) == len(frameworks)
            assert dashboard['summary']['total_frameworks'] == len(frameworks)
            
        finally:
            await compliance_service.stop()
    
    @pytest.mark.asyncio
    async def test_audit_trail_integration(self, compliance_service):
        """Test audit trail functionality with compliance operations."""
        await compliance_service.start()
        
        try:
            # Perform operations that should be audited
            assessment = await compliance_service.assess_compliance("PCI_DSS_4_0")
            await compliance_service.generate_report("PCI_DSS_4_0", ReportType.EXECUTIVE_SUMMARY)
            
            # Get audit trail report
            audit_report = await compliance_service.get_audit_trail_report(
                framework="PCI_DSS_4_0",
                start_date=datetime.utcnow() - timedelta(hours=1),
                end_date=datetime.utcnow()
            )
            
            assert 'report_metadata' in audit_report
            assert 'compliance_activities' in audit_report
            assert 'blockchain_integrity' in audit_report
            
            # Verify audit events were recorded
            activities = audit_report['compliance_activities']
            assert len(activities) > 0
            
            print(f"Audit trail: {len(activities)} activities recorded")
            
        finally:
            await compliance_service.stop()
    
    @pytest.mark.asyncio
    async def test_report_format_validation(self, compliance_service):
        """Test different report formats and validate content."""
        await compliance_service.start()
        
        try:
            # Test different formats
            formats_to_test = [
                (ReportFormat.HTML, 'text/html'),
                (ReportFormat.JSON, 'application/json'),
                (ReportFormat.PDF, 'application/pdf')  # May fail without dependencies
            ]
            
            for report_format, expected_type in formats_to_test:
                try:
                    content = await compliance_service.generate_report(
                        "PCI_DSS_4_0",
                        ReportType.EXECUTIVE_SUMMARY,
                        report_format
                    )
                    
                    assert content is not None
                    assert len(content) > 0
                    
                    # Basic format validation
                    if report_format == ReportFormat.HTML:
                        html_content = content.decode('utf-8')
                        assert '<html>' in html_content or '<!DOCTYPE html>' in html_content
                    elif report_format == ReportFormat.JSON:
                        json_content = json.loads(content.decode('utf-8'))
                        assert isinstance(json_content, dict)
                    
                    print(f"{report_format.value} report: {len(content)} bytes")
                    
                except Exception as e:
                    if report_format == ReportFormat.PDF:
                        print(f"PDF generation failed (expected without dependencies): {e}")
                    else:
                        raise
            
        finally:
            await compliance_service.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, compliance_service):
        """Test concurrent compliance operations."""
        await compliance_service.start()
        
        try:
            # Start multiple concurrent assessments
            tasks = []
            frameworks = ['PCI_DSS_4_0', 'HIPAA']
            
            for framework in frameworks:
                task = asyncio.create_task(
                    compliance_service.assess_compliance(framework)
                )
                tasks.append(task)
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) >= 1  # At least one should succeed
            
            print(f"Concurrent assessments: {len(successful_results)} succeeded out of {len(frameworks)}")
            
        finally:
            await compliance_service.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, compliance_service):
        """Test error handling and recovery scenarios."""
        await compliance_service.start()
        
        try:
            # Test invalid framework
            with pytest.raises(Exception):
                await compliance_service.assess_compliance("INVALID_FRAMEWORK")
            
            # Test service resilience - should continue working after error
            valid_assessment = await compliance_service.assess_compliance("PCI_DSS_4_0")
            assert valid_assessment is not None
            
            print("Error handling test passed - service remains operational")
            
        finally:
            await compliance_service.stop()

class TestCompliancePerformance:
    """Performance tests for compliance system."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_assessment_performance(self, compliance_service):
        """Test assessment performance under load."""
        await compliance_service.start()
        
        try:
            start_time = datetime.utcnow()
            
            # Perform multiple assessments
            tasks = []
            for i in range(5):  # Moderate load test
                task = asyncio.create_task(
                    compliance_service.assess_compliance("PCI_DSS_4_0")
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = datetime.utcnow()
            
            duration = (end_time - start_time).total_seconds()
            successful_count = len([r for r in results if not isinstance(r, Exception)])
            
            print(f"Performance test: {successful_count} assessments in {duration:.2f}s")
            
            # Performance assertions
            assert duration < 60  # Should complete within 1 minute
            assert successful_count >= 1  # At least one should succeed
            
        finally:
            await compliance_service.stop()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_report_generation_performance(self, compliance_service):
        """Test report generation performance."""
        await compliance_service.start()
        
        try:
            # Generate multiple reports concurrently
            start_time = datetime.utcnow()
            
            report_tasks = [
                compliance_service.generate_report("PCI_DSS_4_0", ReportType.EXECUTIVE_SUMMARY, ReportFormat.HTML),
                compliance_service.generate_report("PCI_DSS_4_0", ReportType.DETAILED_TECHNICAL, ReportFormat.JSON),
            ]
            
            results = await asyncio.gather(*report_tasks, return_exceptions=True)
            end_time = datetime.utcnow()
            
            duration = (end_time - start_time).total_seconds()
            successful_count = len([r for r in results if not isinstance(r, Exception)])
            
            print(f"Report generation: {successful_count} reports in {duration:.2f}s")
            
            # Performance assertions
            assert duration < 30  # Should complete within 30 seconds
            assert successful_count >= 1
            
        finally:
            await compliance_service.stop()

class TestComplianceDataQuality:
    """Test data quality and validation."""
    
    @pytest.mark.asyncio
    async def test_assessment_data_quality(self, compliance_service):
        """Test quality of assessment data."""
        await compliance_service.start()
        
        try:
            assessment = await compliance_service.assess_compliance("PCI_DSS_4_0")
            
            # Data quality checks
            assert 0 <= assessment.overall_compliance_score <= 100
            assert 0 <= assessment.risk_score <= 100
            assert assessment.compliant_requirements >= 0
            assert assessment.non_compliant_requirements >= 0
            
            # Verify gaps have required fields
            for gap in assessment.gaps:
                assert gap.requirement_id is not None
                assert gap.requirement_title is not None
                assert gap.severity is not None
                assert gap.gap_description is not None
            
            # Verify recommendations have required fields
            for rec in assessment.recommendations:
                assert rec.id is not None
                assert rec.title is not None
                assert rec.description is not None
                assert rec.priority is not None
                assert rec.estimated_effort_days > 0
            
            print("Assessment data quality validation passed")
            
        finally:
            await compliance_service.stop()
    
    @pytest.mark.asyncio
    async def test_report_content_quality(self, compliance_service):
        """Test quality of generated reports."""
        await compliance_service.start()
        
        try:
            # Generate HTML report
            html_content = await compliance_service.generate_report(
                "PCI_DSS_4_0",
                ReportType.EXECUTIVE_SUMMARY,
                ReportFormat.HTML
            )
            
            html_str = html_content.decode('utf-8')
            
            # Content quality checks
            assert 'PCI_DSS_4_0' in html_str or 'PCI-DSS' in html_str
            assert 'compliance' in html_str.lower()
            assert 'assessment' in html_str.lower()
            assert len(html_str) > 1000  # Minimum content length
            
            # Generate JSON report
            json_content = await compliance_service.generate_report(
                "PCI_DSS_4_0",
                ReportType.DETAILED_TECHNICAL,
                ReportFormat.JSON
            )
            
            json_data = json.loads(json_content.decode('utf-8'))
            
            # JSON structure validation
            assert 'metadata' in json_data
            assert 'sections' in json_data
            assert isinstance(json_data['sections'], dict)
            
            print("Report content quality validation passed")
            
        finally:
            await compliance_service.stop()

# Test configuration and utilities
@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """Cleanup test data after each test."""
    yield
    # Cleanup logic would go here
    pass

def pytest_configure(config):
    """Configure pytest for integration tests."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])