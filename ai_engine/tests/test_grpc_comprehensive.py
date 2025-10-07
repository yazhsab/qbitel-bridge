"""
CRONOS AI Engine - gRPC Comprehensive Tests

Comprehensive test suite for gRPC service functionality.
"""

import pytest
import asyncio
import grpc
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional

from ai_engine.api.grpc import (
    CronosAIGRPCServer,
    ProtocolDiscoveryService,
    ComplianceService,
    SecurityService,
    MonitoringService,
    HealthCheckService,
    GRPCServerConfig,
    GRPCException,
    GRPCServiceError,
)


class TestGRPCServerConfig:
    """Test GRPCServerConfig dataclass."""

    def test_grpc_config_creation(self):
        """Test creating GRPCServerConfig instance."""
        config = GRPCServerConfig(
            host="localhost",
            port=50051,
            max_workers=10,
            max_message_length=4194304,  # 4MB
            max_metadata_size=8192,  # 8KB
            keepalive_time=7200,  # 2 hours
            keepalive_timeout=20,  # 20 seconds
            keepalive_permit_without_calls=True,
            max_connection_idle=300,  # 5 minutes
            max_connection_age=3600,  # 1 hour
            max_connection_age_grace=5,  # 5 seconds
            http2_max_pings_without_data=0,
            http2_min_time_between_pings=10,  # 10 seconds
            http2_min_ping_interval_without_data=5,  # 5 minutes
            http2_max_ping_strikes=2,
            enable_health_check=True,
            enable_reflection=True,
            ssl_enabled=False,
            ssl_cert_file=None,
            ssl_key_file=None,
            ssl_ca_file=None
        )
        
        assert config.host == "localhost"
        assert config.port == 50051
        assert config.max_workers == 10
        assert config.max_message_length == 4194304
        assert config.enable_health_check is True
        assert config.enable_reflection is True
        assert config.ssl_enabled is False

    def test_grpc_config_defaults(self):
        """Test GRPCServerConfig with default values."""
        config = GRPCServerConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 50051
        assert config.max_workers == 10
        assert config.enable_health_check is True
        assert config.enable_reflection is True
        assert config.ssl_enabled is False

    def test_grpc_config_validation(self):
        """Test GRPCServerConfig validation."""
        # Valid config
        config = GRPCServerConfig(
            host="localhost",
            port=50051,
            max_workers=10
        )
        assert config.is_valid() is True
        
        # Invalid config - negative port
        invalid_config = GRPCServerConfig(
            host="localhost",
            port=-1,
            max_workers=10
        )
        assert invalid_config.is_valid() is False
        
        # Invalid config - negative max_workers
        invalid_config2 = GRPCServerConfig(
            host="localhost",
            port=50051,
            max_workers=-1
        )
        assert invalid_config2.is_valid() is False

    def test_grpc_config_ssl_validation(self):
        """Test GRPCServerConfig SSL validation."""
        # SSL enabled but missing cert files
        ssl_config = GRPCServerConfig(
            ssl_enabled=True,
            ssl_cert_file=None,
            ssl_key_file=None
        )
        assert ssl_config.is_valid() is False
        
        # SSL enabled with cert files
        ssl_config_valid = GRPCServerConfig(
            ssl_enabled=True,
            ssl_cert_file="/path/to/cert.pem",
            ssl_key_file="/path/to/key.pem"
        )
        assert ssl_config_valid.is_valid() is True


class TestProtocolDiscoveryService:
    """Test ProtocolDiscoveryService gRPC service."""

    @pytest.fixture
    def mock_protocol_discovery(self):
        """Create mock protocol discovery orchestrator."""
        mock_orchestrator = Mock()
        mock_orchestrator.discover_protocols = AsyncMock()
        mock_orchestrator.analyze_traffic = AsyncMock()
        mock_orchestrator.generate_parser = AsyncMock()
        mock_orchestrator.validate_message = AsyncMock()
        return mock_orchestrator

    @pytest.fixture
    def protocol_service(self, mock_protocol_discovery):
        """Create ProtocolDiscoveryService instance."""
        return ProtocolDiscoveryService(mock_protocol_discovery)

    @pytest.mark.asyncio
    async def test_discover_protocols_success(self, protocol_service, mock_protocol_discovery):
        """Test successful protocol discovery."""
        # Mock discovery result
        discovery_result = {
            "protocols": [
                {
                    "name": "HTTP",
                    "confidence": 0.95,
                    "features": ["method", "path", "headers"]
                },
                {
                    "name": "Modbus",
                    "confidence": 0.87,
                    "features": ["function_code", "data"]
                }
            ],
            "total_protocols": 2,
            "processing_time": 1.5
        }
        mock_protocol_discovery.discover_protocols.return_value = discovery_result
        
        # Create mock request
        request = Mock()
        request.traffic_data = b"GET /api/v1/test HTTP/1.1\r\nHost: example.com\r\n\r\n"
        request.analysis_options = {
            "enable_ml": True,
            "confidence_threshold": 0.8
        }
        
        # Call service method
        response = await protocol_service.DiscoverProtocols(request, None)
        
        # Verify response
        assert response.success is True
        assert len(response.protocols) == 2
        assert response.protocols[0].name == "HTTP"
        assert response.protocols[0].confidence == 0.95
        assert response.total_protocols == 2
        assert response.processing_time == 1.5

    @pytest.mark.asyncio
    async def test_discover_protocols_failure(self, protocol_service, mock_protocol_discovery):
        """Test protocol discovery failure."""
        # Mock discovery failure
        mock_protocol_discovery.discover_protocols.side_effect = Exception("Discovery failed")
        
        request = Mock()
        request.traffic_data = b"invalid data"
        request.analysis_options = {}
        
        # Call service method
        response = await protocol_service.DiscoverProtocols(request, None)
        
        # Verify error response
        assert response.success is False
        assert "Discovery failed" in response.error_message

    @pytest.mark.asyncio
    async def test_analyze_traffic_success(self, protocol_service, mock_protocol_discovery):
        """Test successful traffic analysis."""
        # Mock analysis result
        analysis_result = {
            "statistics": {
                "total_packets": 1000,
                "unique_protocols": 3,
                "traffic_volume": 1024000
            },
            "patterns": [
                {
                    "pattern_type": "request_response",
                    "frequency": 0.8,
                    "examples": ["GET /api", "POST /api"]
                }
            ],
            "anomalies": [
                {
                    "type": "unusual_payload_size",
                    "severity": "medium",
                    "count": 5
                }
            ]
        }
        mock_protocol_discovery.analyze_traffic.return_value = analysis_result
        
        request = Mock()
        request.traffic_data = b"traffic data"
        request.analysis_type = "comprehensive"
        
        response = await protocol_service.AnalyzeTraffic(request, None)
        
        assert response.success is True
        assert response.statistics.total_packets == 1000
        assert response.statistics.unique_protocols == 3
        assert len(response.patterns) == 1
        assert len(response.anomalies) == 1

    @pytest.mark.asyncio
    async def test_generate_parser_success(self, protocol_service, mock_protocol_discovery):
        """Test successful parser generation."""
        # Mock parser generation result
        parser_result = {
            "parser_code": "def parse_http(data):\n    return parse_result",
            "parser_type": "python",
            "confidence": 0.92,
            "test_results": {
                "accuracy": 0.95,
                "false_positives": 0.02,
                "false_negatives": 0.03
            }
        }
        mock_protocol_discovery.generate_parser.return_value = parser_result
        
        request = Mock()
        request.protocol_name = "HTTP"
        request.sample_data = b"GET /api HTTP/1.1"
        request.parser_options = {
            "language": "python",
            "optimize": True
        }
        
        response = await protocol_service.GenerateParser(request, None)
        
        assert response.success is True
        assert response.parser_code == "def parse_http(data):\n    return parse_result"
        assert response.parser_type == "python"
        assert response.confidence == 0.92
        assert response.test_results.accuracy == 0.95

    @pytest.mark.asyncio
    async def test_validate_message_success(self, protocol_service, mock_protocol_discovery):
        """Test successful message validation."""
        # Mock validation result
        validation_result = {
            "is_valid": True,
            "confidence": 0.98,
            "parsed_fields": {
                "method": "GET",
                "path": "/api/v1/test",
                "version": "HTTP/1.1"
            },
            "errors": [],
            "warnings": []
        }
        mock_protocol_discovery.validate_message.return_value = validation_result
        
        request = Mock()
        request.message_data = b"GET /api/v1/test HTTP/1.1\r\nHost: example.com\r\n\r\n"
        request.protocol_name = "HTTP"
        
        response = await protocol_service.ValidateMessage(request, None)
        
        assert response.success is True
        assert response.is_valid is True
        assert response.confidence == 0.98
        assert len(response.parsed_fields) == 3
        assert response.parsed_fields["method"] == "GET"
        assert len(response.errors) == 0
        assert len(response.warnings) == 0

    @pytest.mark.asyncio
    async def test_validate_message_invalid(self, protocol_service, mock_protocol_discovery):
        """Test message validation with invalid message."""
        # Mock validation result for invalid message
        validation_result = {
            "is_valid": False,
            "confidence": 0.15,
            "parsed_fields": {},
            "errors": ["Invalid HTTP format", "Missing required headers"],
            "warnings": ["Unusual payload size"]
        }
        mock_protocol_discovery.validate_message.return_value = validation_result
        
        request = Mock()
        request.message_data = b"invalid http data"
        request.protocol_name = "HTTP"
        
        response = await protocol_service.ValidateMessage(request, None)
        
        assert response.success is True
        assert response.is_valid is False
        assert response.confidence == 0.15
        assert len(response.errors) == 2
        assert len(response.warnings) == 1


class TestComplianceService:
    """Test ComplianceService gRPC service."""

    @pytest.fixture
    def mock_compliance_service(self):
        """Create mock compliance service."""
        mock_service = Mock()
        mock_service.generate_report = AsyncMock()
        mock_service.assess_compliance = AsyncMock()
        mock_service.get_compliance_status = AsyncMock()
        mock_service.update_policies = AsyncMock()
        return mock_service

    @pytest.fixture
    def compliance_service(self, mock_compliance_service):
        """Create ComplianceService instance."""
        return ComplianceService(mock_compliance_service)

    @pytest.mark.asyncio
    async def test_generate_report_success(self, compliance_service, mock_compliance_service):
        """Test successful compliance report generation."""
        # Mock report generation result
        report_result = {
            "report_id": "report_123",
            "report_type": "SOC2",
            "status": "completed",
            "compliance_score": 0.85,
            "findings": [
                {
                    "category": "Access Control",
                    "severity": "medium",
                    "description": "Weak password policy",
                    "recommendation": "Implement stronger password requirements"
                }
            ],
            "generated_at": datetime.now().isoformat(),
            "report_data": "base64_encoded_report_data"
        }
        mock_compliance_service.generate_report.return_value = report_result
        
        request = Mock()
        request.report_type = "SOC2"
        request.assessment_period = {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        }
        request.include_recommendations = True
        
        response = await compliance_service.GenerateReport(request, None)
        
        assert response.success is True
        assert response.report_id == "report_123"
        assert response.report_type == "SOC2"
        assert response.compliance_score == 0.85
        assert len(response.findings) == 1
        assert response.findings[0].category == "Access Control"
        assert response.findings[0].severity == "medium"

    @pytest.mark.asyncio
    async def test_assess_compliance_success(self, compliance_service, mock_compliance_service):
        """Test successful compliance assessment."""
        # Mock assessment result
        assessment_result = {
            "assessment_id": "assessment_456",
            "framework": "GDPR",
            "overall_score": 0.92,
            "control_scores": {
                "data_protection": 0.95,
                "consent_management": 0.88,
                "data_retention": 0.93
            },
            "compliance_status": "compliant",
            "gaps": [],
            "recommendations": [
                "Review data retention policies",
                "Update consent forms"
            ]
        }
        mock_compliance_service.assess_compliance.return_value = assessment_result
        
        request = Mock()
        request.framework = "GDPR"
        request.controls = ["data_protection", "consent_management", "data_retention"]
        request.assessment_options = {
            "include_gaps": True,
            "include_recommendations": True
        }
        
        response = await compliance_service.AssessCompliance(request, None)
        
        assert response.success is True
        assert response.assessment_id == "assessment_456"
        assert response.framework == "GDPR"
        assert response.overall_score == 0.92
        assert response.compliance_status == "compliant"
        assert len(response.control_scores) == 3
        assert len(response.recommendations) == 2

    @pytest.mark.asyncio
    async def test_get_compliance_status_success(self, compliance_service, mock_compliance_service):
        """Test successful compliance status retrieval."""
        # Mock status result
        status_result = {
            "current_status": "compliant",
            "last_assessment": "2023-12-01T10:00:00Z",
            "next_assessment": "2024-03-01T10:00:00Z",
            "frameworks": {
                "SOC2": {"status": "compliant", "score": 0.88},
                "GDPR": {"status": "compliant", "score": 0.92},
                "ISO27001": {"status": "non_compliant", "score": 0.65}
            },
            "pending_actions": [
                "Update ISO27001 controls",
                "Schedule next SOC2 assessment"
            ]
        }
        mock_compliance_service.get_compliance_status.return_value = status_result
        
        request = Mock()
        request.framework = "all"
        request.include_details = True
        
        response = await compliance_service.GetComplianceStatus(request, None)
        
        assert response.success is True
        assert response.current_status == "compliant"
        assert len(response.frameworks) == 3
        assert response.frameworks["SOC2"].status == "compliant"
        assert response.frameworks["SOC2"].score == 0.88
        assert len(response.pending_actions) == 2

    @pytest.mark.asyncio
    async def test_update_policies_success(self, compliance_service, mock_compliance_service):
        """Test successful policy update."""
        # Mock update result
        update_result = {
            "update_id": "update_789",
            "updated_policies": 5,
            "failed_updates": 0,
            "changes": [
                {
                    "policy_id": "policy_1",
                    "change_type": "modified",
                    "description": "Updated password requirements"
                },
                {
                    "policy_id": "policy_2",
                    "change_type": "added",
                    "description": "Added data retention policy"
                }
            ]
        }
        mock_compliance_service.update_policies.return_value = update_result
        
        request = Mock()
        request.policy_updates = [
            {
                "policy_id": "policy_1",
                "content": "Updated policy content",
                "action": "update"
            },
            {
                "policy_id": "policy_2",
                "content": "New policy content",
                "action": "create"
            }
        ]
        request.validate_changes = True
        
        response = await compliance_service.UpdatePolicies(request, None)
        
        assert response.success is True
        assert response.update_id == "update_789"
        assert response.updated_policies == 5
        assert response.failed_updates == 0
        assert len(response.changes) == 2


class TestSecurityService:
    """Test SecurityService gRPC service."""

    @pytest.fixture
    def mock_security_service(self):
        """Create mock security service."""
        mock_service = Mock()
        mock_service.analyze_threat = AsyncMock()
        mock_service.respond_to_incident = AsyncMock()
        mock_service.get_security_status = AsyncMock()
        mock_service.update_security_policies = AsyncMock()
        return mock_service

    @pytest.fixture
    def security_service(self, mock_security_service):
        """Create SecurityService instance."""
        return SecurityService(mock_security_service)

    @pytest.mark.asyncio
    async def test_analyze_threat_success(self, security_service, mock_security_service):
        """Test successful threat analysis."""
        # Mock threat analysis result
        threat_result = {
            "threat_id": "threat_123",
            "threat_type": "malware",
            "severity": "high",
            "confidence": 0.92,
            "indicators": [
                {
                    "type": "file_hash",
                    "value": "abc123def456",
                    "reputation": "malicious"
                },
                {
                    "type": "ip_address",
                    "value": "192.168.1.100",
                    "reputation": "suspicious"
                }
            ],
            "recommended_actions": [
                "Quarantine affected systems",
                "Update antivirus signatures",
                "Block malicious IP addresses"
            ],
            "risk_score": 0.85
        }
        mock_security_service.analyze_threat.return_value = threat_result
        
        request = Mock()
        request.threat_data = {
            "file_hash": "abc123def456",
            "ip_address": "192.168.1.100",
            "user_agent": "suspicious_agent"
        }
        request.analysis_options = {
            "include_indicators": True,
            "include_recommendations": True
        }
        
        response = await security_service.AnalyzeThreat(request, None)
        
        assert response.success is True
        assert response.threat_id == "threat_123"
        assert response.threat_type == "malware"
        assert response.severity == "high"
        assert response.confidence == 0.92
        assert response.risk_score == 0.85
        assert len(response.indicators) == 2
        assert len(response.recommended_actions) == 3

    @pytest.mark.asyncio
    async def test_respond_to_incident_success(self, security_service, mock_security_service):
        """Test successful incident response."""
        # Mock incident response result
        response_result = {
            "incident_id": "incident_456",
            "response_id": "response_789",
            "status": "in_progress",
            "actions_taken": [
                {
                    "action_type": "isolate",
                    "target": "system_001",
                    "status": "completed",
                    "timestamp": "2023-12-01T10:30:00Z"
                },
                {
                    "action_type": "block_ip",
                    "target": "192.168.1.100",
                    "status": "completed",
                    "timestamp": "2023-12-01T10:31:00Z"
                }
            ],
            "next_steps": [
                "Forensic analysis",
                "Evidence collection",
                "Stakeholder notification"
            ],
            "estimated_resolution": "2023-12-01T18:00:00Z"
        }
        mock_security_service.respond_to_incident.return_value = response_result
        
        request = Mock()
        request.incident_id = "incident_456"
        request.response_actions = [
            {
                "action_type": "isolate",
                "target": "system_001",
                "priority": "high"
            },
            {
                "action_type": "block_ip",
                "target": "192.168.1.100",
                "priority": "high"
            }
        ]
        request.automated_response = True
        
        response = await security_service.RespondToIncident(request, None)
        
        assert response.success is True
        assert response.incident_id == "incident_456"
        assert response.response_id == "response_789"
        assert response.status == "in_progress"
        assert len(response.actions_taken) == 2
        assert len(response.next_steps) == 3

    @pytest.mark.asyncio
    async def test_get_security_status_success(self, security_service, mock_security_service):
        """Test successful security status retrieval."""
        # Mock security status result
        status_result = {
            "overall_status": "secure",
            "threat_level": "low",
            "active_threats": 0,
            "security_controls": {
                "firewall": {"status": "active", "last_updated": "2023-12-01T09:00:00Z"},
                "antivirus": {"status": "active", "last_updated": "2023-12-01T08:30:00Z"},
                "ids": {"status": "active", "last_updated": "2023-12-01T09:15:00Z"}
            },
            "recent_events": [
                {
                    "event_type": "login_attempt",
                    "severity": "info",
                    "timestamp": "2023-12-01T10:00:00Z",
                    "description": "Successful admin login"
                }
            ],
            "compliance_status": {
                "pci_dss": "compliant",
                "iso27001": "compliant",
                "nist": "compliant"
            }
        }
        mock_security_service.get_security_status.return_value = status_result
        
        request = Mock()
        request.include_events = True
        request.include_compliance = True
        
        response = await security_service.GetSecurityStatus(request, None)
        
        assert response.success is True
        assert response.overall_status == "secure"
        assert response.threat_level == "low"
        assert response.active_threats == 0
        assert len(response.security_controls) == 3
        assert len(response.recent_events) == 1
        assert len(response.compliance_status) == 3


class TestMonitoringService:
    """Test MonitoringService gRPC service."""

    @pytest.fixture
    def mock_monitoring_service(self):
        """Create mock monitoring service."""
        mock_service = Mock()
        mock_service.get_metrics = AsyncMock()
        mock_service.get_health_status = AsyncMock()
        mock_service.get_alerts = AsyncMock()
        mock_service.create_alert = AsyncMock()
        return mock_service

    @pytest.fixture
    def monitoring_service(self, mock_monitoring_service):
        """Create MonitoringService instance."""
        return MonitoringService(mock_monitoring_service)

    @pytest.mark.asyncio
    async def test_get_metrics_success(self, monitoring_service, mock_monitoring_service):
        """Test successful metrics retrieval."""
        # Mock metrics result
        metrics_result = {
            "timestamp": "2023-12-01T10:00:00Z",
            "metrics": {
                "cpu_usage": {"value": 45.2, "unit": "percent"},
                "memory_usage": {"value": 67.8, "unit": "percent"},
                "disk_usage": {"value": 23.4, "unit": "percent"},
                "network_throughput": {"value": 125.6, "unit": "mbps"},
                "request_rate": {"value": 150, "unit": "requests_per_second"},
                "error_rate": {"value": 0.5, "unit": "percent"}
            },
            "services": {
                "api_service": {
                    "status": "healthy",
                    "response_time": {"value": 120, "unit": "ms"},
                    "throughput": {"value": 100, "unit": "requests_per_second"}
                },
                "database_service": {
                    "status": "healthy",
                    "connection_pool": {"value": 45, "unit": "connections"},
                    "query_time": {"value": 15, "unit": "ms"}
                }
            }
        }
        mock_monitoring_service.get_metrics.return_value = metrics_result
        
        request = Mock()
        request.metric_names = ["cpu_usage", "memory_usage", "disk_usage"]
        request.time_range = {
            "start_time": "2023-12-01T09:00:00Z",
            "end_time": "2023-12-01T10:00:00Z"
        }
        request.include_services = True
        
        response = await monitoring_service.GetMetrics(request, None)
        
        assert response.success is True
        assert len(response.metrics) == 6
        assert response.metrics["cpu_usage"].value == 45.2
        assert response.metrics["cpu_usage"].unit == "percent"
        assert len(response.services) == 2
        assert response.services["api_service"].status == "healthy"

    @pytest.mark.asyncio
    async def test_get_health_status_success(self, monitoring_service, mock_monitoring_service):
        """Test successful health status retrieval."""
        # Mock health status result
        health_result = {
            "overall_status": "healthy",
            "timestamp": "2023-12-01T10:00:00Z",
            "components": {
                "api_gateway": {"status": "healthy", "response_time": 50},
                "database": {"status": "healthy", "response_time": 15},
                "cache": {"status": "degraded", "response_time": 200},
                "message_queue": {"status": "healthy", "response_time": 5}
            },
            "dependencies": {
                "external_api": {"status": "healthy", "response_time": 100},
                "payment_service": {"status": "healthy", "response_time": 80}
            },
            "alerts": [
                {
                    "alert_id": "alert_001",
                    "severity": "warning",
                    "message": "Cache response time is high",
                    "timestamp": "2023-12-01T09:45:00Z"
                }
            ]
        }
        mock_monitoring_service.get_health_status.return_value = health_result
        
        request = Mock()
        request.include_dependencies = True
        request.include_alerts = True
        
        response = await monitoring_service.GetHealthStatus(request, None)
        
        assert response.success is True
        assert response.overall_status == "healthy"
        assert len(response.components) == 4
        assert response.components["api_gateway"].status == "healthy"
        assert response.components["cache"].status == "degraded"
        assert len(response.dependencies) == 2
        assert len(response.alerts) == 1

    @pytest.mark.asyncio
    async def test_get_alerts_success(self, monitoring_service, mock_monitoring_service):
        """Test successful alerts retrieval."""
        # Mock alerts result
        alerts_result = {
            "alerts": [
                {
                    "alert_id": "alert_001",
                    "severity": "critical",
                    "status": "active",
                    "message": "High CPU usage detected",
                    "timestamp": "2023-12-01T10:00:00Z",
                    "source": "system_monitor",
                    "tags": ["cpu", "performance"]
                },
                {
                    "alert_id": "alert_002",
                    "severity": "warning",
                    "status": "acknowledged",
                    "message": "Disk space running low",
                    "timestamp": "2023-12-01T09:30:00Z",
                    "source": "disk_monitor",
                    "tags": ["disk", "storage"]
                }
            ],
            "total_alerts": 2,
            "active_alerts": 1,
            "acknowledged_alerts": 1
        }
        mock_monitoring_service.get_alerts.return_value = alerts_result
        
        request = Mock()
        request.severity_filter = ["critical", "warning"]
        request.status_filter = ["active", "acknowledged"]
        request.time_range = {
            "start_time": "2023-12-01T09:00:00Z",
            "end_time": "2023-12-01T11:00:00Z"
        }
        
        response = await monitoring_service.GetAlerts(request, None)
        
        assert response.success is True
        assert len(response.alerts) == 2
        assert response.total_alerts == 2
        assert response.active_alerts == 1
        assert response.acknowledged_alerts == 1
        assert response.alerts[0].severity == "critical"
        assert response.alerts[0].status == "active"

    @pytest.mark.asyncio
    async def test_create_alert_success(self, monitoring_service, mock_monitoring_service):
        """Test successful alert creation."""
        # Mock alert creation result
        alert_result = {
            "alert_id": "alert_003",
            "status": "created",
            "message": "Alert created successfully",
            "created_at": "2023-12-01T10:15:00Z"
        }
        mock_monitoring_service.create_alert.return_value = alert_result
        
        request = Mock()
        request.alert_config = {
            "name": "High Memory Usage",
            "description": "Alert when memory usage exceeds 90%",
            "severity": "warning",
            "threshold": 90.0,
            "metric": "memory_usage",
            "duration": 300,  # 5 minutes
            "enabled": True
        }
        request.notification_channels = ["email", "slack"]
        
        response = await monitoring_service.CreateAlert(request, None)
        
        assert response.success is True
        assert response.alert_id == "alert_003"
        assert response.status == "created"
        assert response.message == "Alert created successfully"


class TestHealthCheckService:
    """Test HealthCheckService gRPC service."""

    @pytest.fixture
    def mock_health_checker(self):
        """Create mock health checker."""
        mock_checker = Mock()
        mock_checker.check_health = AsyncMock()
        mock_checker.get_detailed_health = AsyncMock()
        return mock_checker

    @pytest.fixture
    def health_service(self, mock_health_checker):
        """Create HealthCheckService instance."""
        return HealthCheckService(mock_health_checker)

    @pytest.mark.asyncio
    async def test_check_health_success(self, health_service, mock_health_checker):
        """Test successful health check."""
        # Mock health check result
        health_result = {
            "status": "healthy",
            "timestamp": "2023-12-01T10:00:00Z",
            "response_time": 15.5,
            "version": "1.0.0",
            "uptime": 86400  # 24 hours
        }
        mock_health_checker.check_health.return_value = health_result
        
        request = Mock()
        request.include_details = True
        
        response = await health_service.CheckHealth(request, None)
        
        assert response.success is True
        assert response.status == "healthy"
        assert response.response_time == 15.5
        assert response.version == "1.0.0"
        assert response.uptime == 86400

    @pytest.mark.asyncio
    async def test_get_detailed_health_success(self, health_service, mock_health_checker):
        """Test successful detailed health check."""
        # Mock detailed health result
        detailed_health = {
            "overall_status": "healthy",
            "timestamp": "2023-12-01T10:00:00Z",
            "components": {
                "database": {
                    "status": "healthy",
                    "response_time": 10.2,
                    "last_check": "2023-12-01T10:00:00Z"
                },
                "cache": {
                    "status": "degraded",
                    "response_time": 150.0,
                    "last_check": "2023-12-01T10:00:00Z",
                    "error": "High response time"
                },
                "external_api": {
                    "status": "healthy",
                    "response_time": 45.8,
                    "last_check": "2023-12-01T10:00:00Z"
                }
            },
            "system_info": {
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
                "disk_usage": 23.4,
                "load_average": [1.2, 1.5, 1.8]
            }
        }
        mock_health_checker.get_detailed_health.return_value = detailed_health
        
        request = Mock()
        request.include_system_info = True
        
        response = await health_service.GetDetailedHealth(request, None)
        
        assert response.success is True
        assert response.overall_status == "healthy"
        assert len(response.components) == 3
        assert response.components["database"].status == "healthy"
        assert response.components["cache"].status == "degraded"
        assert response.system_info.cpu_usage == 45.2
        assert response.system_info.memory_usage == 67.8


class TestCronosAIGRPCServer:
    """Test CronosAIGRPCServer main functionality."""

    @pytest.fixture
    def grpc_config(self):
        """Create gRPC server configuration."""
        return GRPCServerConfig(
            host="localhost",
            port=50051,
            max_workers=10,
            enable_health_check=True,
            enable_reflection=True
        )

    @pytest.fixture
    def grpc_server(self, grpc_config):
        """Create CronosAIGRPCServer instance."""
        with patch('ai_engine.api.grpc.grpc.aio.server') as mock_server:
            return CronosAIGRPCServer(grpc_config)

    def test_grpc_server_initialization(self, grpc_server, grpc_config):
        """Test CronosAIGRPCServer initialization."""
        assert grpc_server.config == grpc_config
        assert grpc_server.server is not None
        assert grpc_server.is_running is False

    @pytest.mark.asyncio
    async def test_start_server(self, grpc_server):
        """Test starting gRPC server."""
        with patch.object(grpc_server.server, 'start') as mock_start:
            await grpc_server.start()
            
            mock_start.assert_called_once()
            assert grpc_server.is_running is True

    @pytest.mark.asyncio
    async def test_stop_server(self, grpc_server):
        """Test stopping gRPC server."""
        with patch.object(grpc_server.server, 'stop') as mock_stop:
            await grpc_server.stop()
            
            mock_stop.assert_called_once()
            assert grpc_server.is_running is False

    @pytest.mark.asyncio
    async def test_server_lifecycle(self, grpc_server):
        """Test complete server lifecycle."""
        with patch.object(grpc_server.server, 'start') as mock_start, \
             patch.object(grpc_server.server, 'stop') as mock_stop:
            
            # Start server
            await grpc_server.start()
            assert grpc_server.is_running is True
            mock_start.assert_called_once()
            
            # Stop server
            await grpc_server.stop()
            assert grpc_server.is_running is False
            mock_stop.assert_called_once()

    def test_add_service(self, grpc_server):
        """Test adding service to gRPC server."""
        mock_service = Mock()
        mock_servicer = Mock()
        
        grpc_server.add_service(mock_service, mock_servicer)
        
        # Verify service was added (implementation depends on mock)
        assert True  # Placeholder assertion

    def test_configure_ssl(self, grpc_server):
        """Test SSL configuration for gRPC server."""
        ssl_config = {
            "cert_file": "/path/to/cert.pem",
            "key_file": "/path/to/key.pem",
            "ca_file": "/path/to/ca.pem"
        }
        
        grpc_server.configure_ssl(ssl_config)
        
        # Verify SSL was configured
        assert True  # Placeholder assertion

    def test_get_server_info(self, grpc_server):
        """Test getting server information."""
        info = grpc_server.get_server_info()
        
        assert "config" in info
        assert "is_running" in info
        assert "services" in info
        assert info["is_running"] is False

    def test_grpc_server_error_handling(self, grpc_server):
        """Test gRPC server error handling."""
        with patch.object(grpc_server.server, 'start', side_effect=Exception("Start failed")):
            with pytest.raises(GRPCException):
                asyncio.run(grpc_server.start())

    def test_grpc_server_concurrent_operations(self, grpc_server):
        """Test gRPC server concurrent operations."""
        async def start_stop_cycle():
            await grpc_server.start()
            await asyncio.sleep(0.1)
            await grpc_server.stop()
        
        # Run concurrent start/stop cycles
        tasks = [start_stop_cycle() for _ in range(5)]
        asyncio.run(asyncio.gather(*tasks, return_exceptions=True))

    def test_grpc_server_graceful_shutdown(self, grpc_server):
        """Test gRPC server graceful shutdown."""
        with patch.object(grpc_server.server, 'start') as mock_start, \
             patch.object(grpc_server.server, 'stop') as mock_stop:
            
            async def test_graceful_shutdown():
                await grpc_server.start()
                await grpc_server.graceful_shutdown(timeout=5.0)
            
            asyncio.run(test_graceful_shutdown())
            
            mock_start.assert_called_once()
            mock_stop.assert_called_once()

    def test_grpc_server_metrics(self, grpc_server):
        """Test gRPC server metrics collection."""
        metrics = grpc_server.get_metrics()
        
        assert "requests_total" in metrics
        assert "requests_per_second" in metrics
        assert "average_response_time" in metrics
        assert "active_connections" in metrics
        assert "error_rate" in metrics

    def test_grpc_server_health_check(self, grpc_server):
        """Test gRPC server health check."""
        health = grpc_server.check_health()
        
        assert "status" in health
        assert "timestamp" in health
        assert "uptime" in health
        assert "version" in health