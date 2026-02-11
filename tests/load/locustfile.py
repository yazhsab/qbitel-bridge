"""
QBITEL - Load Testing with Locust

Comprehensive load testing scenarios for QBITEL services.
"""

from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser
import json
import random
import time
from typing import Dict, Any


class QbitelAIUser(FastHttpUser):
    """Base user class for QBITEL load testing."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Initialize user session."""
        self.auth_token = self.login()
        self.headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }

    def login(self) -> str:
        """Authenticate user and get token."""
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "username": f"loadtest_user_{random.randint(1, 1000)}",
                "password": "loadtest_password",
            },
            name="/api/v1/auth/login",
        )

        if response.status_code == 200:
            return response.json().get("access_token", "")
        return ""

    @task(3)
    def protocol_discovery(self):
        """Test protocol discovery endpoint."""
        payload = {
            "packet_data": self._generate_sample_packet(),
            "options": {"deep_inspection": True, "ml_inference": True},
        }

        with self.client.post(
            "/api/v1/protocol/discover",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/protocol/discover",
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "protocol" in result:
                    response.success()
                else:
                    response.failure("Protocol not detected")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def security_analysis(self):
        """Test security analysis endpoint."""
        payload = {
            "event_data": self._generate_security_event(),
            "analysis_type": "threat_detection",
        }

        with self.client.post(
            "/api/v1/security/analyze",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/security/analyze",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def compliance_check(self):
        """Test compliance checking endpoint."""
        payload = {
            "device_id": f"device_{random.randint(1, 10000)}",
            "policy_id": f"policy_{random.randint(1, 100)}",
        }

        with self.client.post(
            "/api/v1/compliance/check",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/compliance/check",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def translation_studio(self):
        """Test protocol translation endpoint."""
        payload = {
            "source_protocol": "modbus",
            "target_protocol": "mqtt",
            "data": self._generate_sample_packet(),
        }

        with self.client.post(
            "/api/v1/translation/translate",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/translation/translate",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def copilot_query(self):
        """Test protocol copilot endpoint."""
        payload = {
            "query": random.choice(
                [
                    "How do I configure Modbus TCP?",
                    "What are the security best practices for OT protocols?",
                    "Explain the difference between Modbus RTU and TCP",
                ]
            ),
            "context": {"protocol": "modbus", "environment": "industrial"},
        }

        with self.client.post(
            "/api/v1/copilot/query",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/copilot/query",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    def health_check(self):
        """Test health check endpoint."""
        with self.client.get(
            "/health", catch_response=True, name="/health"
        ) as response:
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("Service unhealthy")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(3)
    def metrics_endpoint(self):
        """Test metrics endpoint."""
        with self.client.get(
            "/metrics", catch_response=True, name="/metrics"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    def _generate_sample_packet(self) -> str:
        """Generate sample packet data."""
        packets = [
            "0x01030000000AC5CD",  # Modbus
            "0x474554202F20485454502F312E31",  # HTTP
            "0x1603010200",  # TLS
        ]
        return random.choice(packets)

    def _generate_security_event(self) -> Dict[str, Any]:
        """Generate sample security event."""
        return {
            "timestamp": time.time(),
            "source_ip": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "destination_ip": f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "event_type": random.choice(
                ["intrusion_attempt", "anomaly_detected", "policy_violation"]
            ),
            "severity": random.choice(["low", "medium", "high", "critical"]),
        }


class SpikeTestUser(QbitelAIUser):
    """User class for spike testing scenarios."""

    wait_time = between(0.1, 0.5)  # Aggressive load

    @task(10)
    def rapid_protocol_discovery(self):
        """Rapid-fire protocol discovery requests."""
        self.protocol_discovery()


class StressTestUser(QbitelAIUser):
    """User class for stress testing scenarios."""

    wait_time = between(0, 0.1)  # Minimal wait time

    @task(5)
    def stress_all_endpoints(self):
        """Hit all endpoints rapidly."""
        endpoints = [
            self.protocol_discovery,
            self.security_analysis,
            self.compliance_check,
            self.health_check,
        ]
        random.choice(endpoints)()


class SoakTestUser(QbitelAIUser):
    """User class for soak/endurance testing."""

    wait_time = between(2, 5)  # Moderate, sustained load

    @task
    def sustained_load(self):
        """Sustained load over long period."""
        self.protocol_discovery()
        time.sleep(1)
        self.security_analysis()


# Event handlers for custom metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Track custom metrics for requests."""
    if exception:
        print(f"Request failed: {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test environment."""
    print("Load test starting...")
    print(f"Target host: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Cleanup and report results."""
    print("Load test completed")

    # Calculate and print statistics
    stats = environment.stats
    print(f"\nTotal requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Max response time: {stats.total.max_response_time:.2f}ms")
    print(f"Requests per second: {stats.total.total_rps:.2f}")

    # Fail the test if error rate is too high
    if stats.total.num_failures / stats.total.num_requests > 0.05:  # 5% error threshold
        print("ERROR: Failure rate exceeded 5%")
        environment.process_exit_code = 1


# Load test scenarios
class ZeroTouchSecurityUser(QbitelAIUser):
    """User class focused on Zero-Touch Security testing."""

    wait_time = between(0.5, 2)

    @task(5)
    def analyze_security_event(self):
        """Test zero-touch security analysis endpoint."""
        event_types = [
            "intrusion_attempt",
            "malware_detected",
            "data_exfiltration",
            "privilege_escalation",
            "dos_attack",
            "unauthorized_access",
        ]

        payload = {
            "event_type": random.choice(event_types),
            "severity": random.choice(["low", "medium", "high", "critical"]),
            "source_ip": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "destination_ip": f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "destination_port": random.choice([22, 80, 443, 3389, 5432]),
            "protocol": random.choice(["SSH", "HTTP", "HTTPS", "RDP", "PostgreSQL"]),
            "asset_id": f"server-{random.randint(1, 100):03d}",
            "asset_type": random.choice(["linux_server", "windows_server", "database", "network_device"]),
            "indicators": [random.choice([
                "brute_force", "suspicious_user_agent", "c2_communication",
                "unusual_time", "large_data_transfer", "privilege_abuse"
            ])],
        }

        with self.client.post(
            "/api/v1/zero-touch/analyze",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/zero-touch/analyze",
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "threat_level" in result and "confidence" in result:
                    response.success()
                else:
                    response.failure("Missing threat analysis fields")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(3)
    def respond_to_threat(self):
        """Test full zero-touch response pipeline."""
        payload = {
            "event_type": "intrusion_attempt",
            "severity": "high",
            "source_ip": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "destination_ip": "10.0.0.50",
            "destination_port": 22,
            "protocol": "SSH",
            "asset_id": "server-prod-001",
            "indicators": ["brute_force", "multiple_failed_logins"],
        }

        start_time = time.time()
        with self.client.post(
            "/api/v1/zero-touch/respond",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/zero-touch/respond",
        ) as response:
            response_time = (time.time() - start_time) * 1000
            if response.status_code == 200:
                result = response.json()
                if result.get("processing_time_ms", 0) < 5000:  # 5 second SLA
                    response.success()
                else:
                    response.failure(f"Response time exceeded SLA: {result.get('processing_time_ms')}ms")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def simulate_response(self):
        """Test response simulation (no execution)."""
        payload = {
            "event_type": "malware_detected",
            "severity": "critical",
            "source_ip": "10.0.0.100",
            "asset_id": "workstation-042",
            "indicators": ["trojan_signature", "c2_beacon"],
        }

        with self.client.post(
            "/api/v1/zero-touch/simulate",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/zero-touch/simulate",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(4)
    def list_decisions(self):
        """Test listing security decisions."""
        with self.client.get(
            "/api/v1/zero-touch/decisions",
            params={"limit": 20},
            headers=self.headers,
            catch_response=True,
            name="/api/v1/zero-touch/decisions",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def get_metrics(self):
        """Test zero-touch metrics endpoint."""
        with self.client.get(
            "/api/v1/zero-touch/metrics",
            params={"time_range": "24h"},
            headers=self.headers,
            catch_response=True,
            name="/api/v1/zero-touch/metrics",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def get_pending_approvals(self):
        """Test pending approvals endpoint."""
        with self.client.get(
            "/api/v1/zero-touch/pending-approvals",
            headers=self.headers,
            catch_response=True,
            name="/api/v1/zero-touch/pending-approvals",
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class LoadTestScenarios:
    """Predefined load test scenarios."""

    @staticmethod
    def baseline():
        """Baseline load test: 10 users, 5 minute ramp-up."""
        return {"users": 10, "spawn_rate": 2, "run_time": "10m"}

    @staticmethod
    def normal_load():
        """Normal load: 50 users, steady state."""
        return {"users": 50, "spawn_rate": 5, "run_time": "30m"}

    @staticmethod
    def peak_load():
        """Peak load: 200 users, simulating peak hours."""
        return {"users": 200, "spawn_rate": 10, "run_time": "20m"}

    @staticmethod
    def spike_test():
        """Spike test: Sudden increase to 500 users."""
        return {"users": 500, "spawn_rate": 50, "run_time": "10m"}

    @staticmethod
    def stress_test():
        """Stress test: Push system to limits."""
        return {"users": 1000, "spawn_rate": 100, "run_time": "15m"}

    @staticmethod
    def soak_test():
        """Soak test: Sustained load over extended period."""
        return {"users": 100, "spawn_rate": 5, "run_time": "4h"}

    @staticmethod
    def zero_touch_benchmark():
        """Zero-Touch Security benchmark: Focus on security response latency."""
        return {"users": 100, "spawn_rate": 20, "run_time": "15m", "user_classes": [ZeroTouchSecurityUser]}


if __name__ == "__main__":
    print("QBITEL Load Testing")
    print("=" * 50)
    print("\nAvailable scenarios:")
    print("1. Baseline (10 users, 10 min)")
    print("2. Normal Load (50 users, 30 min)")
    print("3. Peak Load (200 users, 20 min)")
    print("4. Spike Test (500 users, 10 min)")
    print("5. Stress Test (1000 users, 15 min)")
    print("6. Soak Test (100 users, 4 hours)")
    print("\nRun with: locust -f locustfile.py --host=http://localhost:8080")
