"""
QBITEL Engine - Legacy System Whisperer Tests

Comprehensive test suite for Legacy System Whisperer feature.
"""

# Test configuration
TEST_TIMEOUT = 30  # seconds
ASYNC_TEST_TIMEOUT = 60  # seconds
INTEGRATION_TEST_TIMEOUT = 120  # seconds

# Test data paths
TEST_DATA_DIR = "test_data"
TEST_CONFIGS_DIR = "test_configs"
TEST_FIXTURES_DIR = "fixtures"

# Test system contexts for mocking
MOCK_MAINFRAME_CONTEXT = {
    "system_id": "test_mainframe_001",
    "system_name": "Test Mainframe System",
    "system_type": "mainframe",
    "version": "z/OS 2.5",
    "location": "datacenter_a",
    "criticality": "high",
    "compliance_requirements": ["sox", "hipaa"],
    "technical_contacts": ["admin@test.com"],
    "business_contacts": ["business@test.com"],
}

MOCK_SCADA_CONTEXT = {
    "system_id": "test_scada_001",
    "system_name": "Test SCADA System",
    "system_type": "scada",
    "version": "v3.1",
    "location": "plant_floor",
    "criticality": "critical",
    "compliance_requirements": ["nerc_cip"],
    "technical_contacts": ["scada@test.com"],
    "business_contacts": ["ops@test.com"],
}

MOCK_SYSTEM_METRICS = {
    "cpu_utilization": 65.5,
    "memory_utilization": 78.2,
    "disk_utilization": 45.8,
    "network_utilization": 23.4,
    "response_time_ms": 150.0,
    "error_rate": 0.02,
    "transaction_rate": 1250.0,
    "availability": 0.9995,
    "timestamp": "2024-01-15T10:30:00Z",
}

# Test environment markers
pytest_markers = {
    "unit": "Unit tests - fast, isolated",
    "integration": "Integration tests - slower, requires dependencies",
    "e2e": "End-to-end tests - full system",
    "performance": "Performance tests - resource intensive",
    "security": "Security tests - authentication/authorization",
    "slow": "Slow running tests - may take several minutes",
}
