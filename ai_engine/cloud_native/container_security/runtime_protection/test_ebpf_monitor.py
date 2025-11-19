"""
Integration tests for eBPF Runtime Monitor

Tests the production-ready eBPF monitoring implementation.
"""

import pytest
import time
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from ai_engine.cloud_native.container_security.runtime_protection.ebpf_monitor import (
    eBPFMonitor,
    RuntimeEvent,
    EventType
)


class TestEBPFMonitor:
    """Test suite for eBPF Monitor"""

    @pytest.fixture
    def ebpf_monitor(self):
        """Create eBPF monitor instance for testing"""
        return eBPFMonitor()

    def test_monitor_initialization(self, ebpf_monitor):
        """Test monitor initializes correctly"""
        assert ebpf_monitor._events == []
        assert ebpf_monitor._threat_rules is not None
        assert "suspicious_processes" in ebpf_monitor._threat_rules
        assert "sensitive_files" in ebpf_monitor._threat_rules

    def test_load_threat_rules(self, ebpf_monitor):
        """Test threat rules are loaded"""
        rules = ebpf_monitor._threat_rules

        assert "nc" in rules["suspicious_processes"]
        assert "netcat" in rules["suspicious_processes"]
        assert "/etc/passwd" in rules["sensitive_files"]
        assert "/etc/shadow" in rules["sensitive_files"]

    def test_detect_threats_suspicious_process(self, ebpf_monitor):
        """Test threat detection for suspicious processes"""
        event = RuntimeEvent(
            event_type=EventType.PROCESS_EXEC,
            container_id="test-container",
            process_name="nc",
            timestamp=time.time(),
            details={"filename": "/usr/bin/nc"}
        )

        is_threat = ebpf_monitor.detect_threats(event)

        assert is_threat is True
        assert event.threat_score == 0.8

    def test_detect_threats_normal_process(self, ebpf_monitor):
        """Test threat detection for normal processes"""
        event = RuntimeEvent(
            event_type=EventType.PROCESS_EXEC,
            container_id="test-container",
            process_name="nginx",
            timestamp=time.time(),
            details={"filename": "/usr/sbin/nginx"}
        )

        is_threat = ebpf_monitor.detect_threats(event)

        assert is_threat is False

    def test_get_statistics(self, ebpf_monitor):
        """Test getting monitoring statistics"""
        # Add some events
        event1 = RuntimeEvent(
            event_type=EventType.PROCESS_EXEC,
            container_id="test-container",
            process_name="nginx",
            timestamp=time.time(),
            details={}
        )
        event1.threat_score = 0.0

        event2 = RuntimeEvent(
            event_type=EventType.PROCESS_EXEC,
            container_id="test-container",
            process_name="nc",
            timestamp=time.time(),
            details={}
        )
        event2.threat_score = 0.8

        ebpf_monitor._events = [event1, event2]

        stats = ebpf_monitor.get_statistics()

        assert stats["total_events"] == 2
        assert stats["threats_detected"] == 1
        assert stats["cpu_overhead_percent"] < 1.0

    @pytest.mark.skipif(
        not hasattr(eBPFMonitor, '_bpf'),
        reason="BCC not available"
    )
    def test_monitor_container_without_bcc(self):
        """Test monitor_container when BCC is not available"""
        with patch('ai_engine.cloud_native.container_security.runtime_protection.ebpf_monitor.BCC_AVAILABLE', False):
            monitor = eBPFMonitor()
            result = monitor.monitor_container("test-container")

            assert result["monitoring"] is False
            assert "error" in result
            assert "BCC not installed" in result["error"]

    @pytest.mark.skipif(
        sys.platform != "linux",
        reason="eBPF only supported on Linux"
    )
    def test_monitor_container_with_bcc(self):
        """Test monitor_container with BCC available (Linux only)"""
        with patch('ai_engine.cloud_native.container_security.runtime_protection.ebpf_monitor.BCC_AVAILABLE', True):
            with patch('ai_engine.cloud_native.container_security.runtime_protection.ebpf_monitor.BPF') as mock_bpf:
                monitor = eBPFMonitor()

                # Mock BPF instance
                mock_bpf_instance = MagicMock()
                mock_bpf.return_value = mock_bpf_instance

                result = monitor.monitor_container("test-container")

                assert result["monitoring"] is True
                assert result["container_id"] == "test-container"
                assert "events_captured" in result

    def test_stop_monitoring(self, ebpf_monitor):
        """Test stopping container monitoring"""
        container_id = "test-container"

        # Simulate active monitoring
        ebpf_monitor._monitoring_active[container_id] = True
        ebpf_monitor._monitored_containers[container_id] = {
            "container_id": container_id,
            "monitoring": True
        }

        # Stop monitoring
        ebpf_monitor.stop_monitoring(container_id)

        assert ebpf_monitor._monitoring_active[container_id] is False
        assert container_id not in ebpf_monitor._monitored_containers

    def test_event_callback(self):
        """Test event callback functionality"""
        callback_called = False
        received_event = None

        def test_callback(event):
            nonlocal callback_called, received_event
            callback_called = True
            received_event = event

        monitor = eBPFMonitor(event_callback=test_callback)

        # Create and process event manually
        event = RuntimeEvent(
            event_type=EventType.PROCESS_EXEC,
            container_id="test",
            process_name="test",
            timestamp=time.time(),
            pid=1234,
            uid=0,
            comm="test",
            details={}
        )

        # Simulate event processing
        if monitor._event_callback:
            monitor._event_callback(event)

        assert callback_called is True
        assert received_event == event

    @pytest.mark.skipif(
        sys.platform != "linux",
        reason="eBPF only supported on Linux"
    )
    def test_process_exec_event(self):
        """Test processing execution events"""
        with patch('ai_engine.cloud_native.container_security.runtime_protection.ebpf_monitor.BCC_AVAILABLE', True):
            monitor = eBPFMonitor()
            container_id = "test-container"

            # Mock event data
            mock_event = MagicMock()
            mock_event.pid = 1234
            mock_event.ppid = 1
            mock_event.uid = 0
            mock_event.comm = b"bash"
            mock_event.filename = b"/bin/bash"

            monitor._process_exec_event(container_id, mock_event)

            assert len(monitor._events) > 0
            last_event = monitor._events[-1]
            assert last_event.event_type == EventType.PROCESS_EXEC
            assert last_event.container_id == container_id

    @pytest.mark.skipif(
        sys.platform != "linux",
        reason="eBPF only supported on Linux"
    )
    def test_process_file_event(self):
        """Test processing file access events"""
        with patch('ai_engine.cloud_native.container_security.runtime_protection.ebpf_monitor.BCC_AVAILABLE', True):
            monitor = eBPFMonitor()
            container_id = "test-container"

            # Mock event data for sensitive file
            mock_event = MagicMock()
            mock_event.pid = 1234
            mock_event.uid = 0
            mock_event.comm = b"cat"
            mock_event.filename = b"/etc/shadow"
            mock_event.flags = 0

            monitor._process_file_event(container_id, mock_event)

            assert len(monitor._events) > 0
            last_event = monitor._events[-1]
            assert last_event.event_type == EventType.FILE_ACCESS
            assert last_event.threat_score > 0.0  # Should detect sensitive file

    @pytest.mark.skipif(
        sys.platform != "linux",
        reason="eBPF only supported on Linux"
    )
    def test_process_connect_event(self):
        """Test processing network connection events"""
        with patch('ai_engine.cloud_native.container_security.runtime_protection.ebpf_monitor.BCC_AVAILABLE', True):
            monitor = eBPFMonitor()
            container_id = "test-container"

            # Mock event data
            mock_event = MagicMock()
            mock_event.pid = 1234
            mock_event.uid = 0
            mock_event.comm = b"curl"
            mock_event.daddr = int.from_bytes(bytes([192, 168, 1, 1]), byteorder='little')
            mock_event.dport = 443

            monitor._process_connect_event(container_id, mock_event)

            assert len(monitor._events) > 0
            last_event = monitor._events[-1]
            assert last_event.event_type == EventType.NETWORK_CONNECT


class TestRuntimeEvent:
    """Test suite for RuntimeEvent dataclass"""

    def test_runtime_event_creation(self):
        """Test creating a RuntimeEvent"""
        event = RuntimeEvent(
            event_type=EventType.PROCESS_EXEC,
            container_id="test-container",
            process_name="nginx",
            timestamp=time.time(),
            details={"filename": "/usr/sbin/nginx"},
            pid=1234,
            uid=0
        )

        assert event.event_type == EventType.PROCESS_EXEC
        assert event.container_id == "test-container"
        assert event.process_name == "nginx"
        assert event.pid == 1234
        assert event.threat_score == 0.0  # Default value

    def test_runtime_event_with_threat(self):
        """Test RuntimeEvent with threat score"""
        event = RuntimeEvent(
            event_type=EventType.PROCESS_EXEC,
            container_id="test-container",
            process_name="nc",
            timestamp=time.time(),
            details={},
            threat_score=0.9
        )

        assert event.threat_score == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
