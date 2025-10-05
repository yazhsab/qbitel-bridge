#!/usr/bin/env python3
"""
CRONOS AI - Production Integration Testing Framework
Comprehensive end-to-end testing for all production components
"""

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

import aiohttp
import asyncpg
import kafka
import redis
import pytest
import requests
from kubernetes import client, config
from prometheus_client.parser import text_string_to_metric_families
import grpc
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/integration-test.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result container"""

    test_name: str
    status: str  # PASS, FAIL, SKIP
    duration: float
    error: Optional[str] = None
    details: Optional[Dict] = None


@dataclass
class HealthCheck:
    """Health check configuration"""

    name: str
    url: str
    expected_status: int = 200
    timeout: int = 30
    headers: Optional[Dict] = None


class ProductionIntegrationTester:
    """Comprehensive production integration testing framework"""

    def __init__(self, config_file: str = "config/cronos_ai.production.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
        self.test_results: List[TestResult] = []
        self.start_time = time.time()

        # Service endpoints
        self.endpoints = {
            "ai_engine": os.getenv("AI_ENGINE_URL", "http://ai-engine-service:8000"),
            "mgmt_api": os.getenv("MGMT_API_URL", "http://mgmtapi-service:8080"),
            "control_plane": os.getenv(
                "CONTROL_PLANE_URL", "http://controlplane-service:9000"
            ),
            "ui_console": os.getenv("UI_CONSOLE_URL", "http://ui-console-service:3000"),
            "prometheus": os.getenv("PROMETHEUS_URL", "http://prometheus-service:9090"),
            "grafana": os.getenv("GRAFANA_URL", "http://grafana-service:3000"),
            "kafka": os.getenv("KAFKA_BROKERS", "kafka-service:9092"),
            "redis": os.getenv("REDIS_URL", "redis://redis-service:6379"),
            "timescaledb": os.getenv(
                "TIMESCALE_URL", "postgresql://timescaledb-service:5432/cronos_ai_prod"
            ),
        }

        # Initialize clients
        self.session = None
        self.k8s_client = None
        self.redis_client = None
        self.db_pool = None

    def _load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            with open(self.config_file, "r") as f:
                import yaml

                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config file {self.config_file}: {e}")
            return {}

    async def setup(self):
        """Initialize test clients and connections"""
        logger.info("Setting up integration test environment...")

        # HTTP session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100),
        )

        # Kubernetes client
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        self.k8s_client = client.ApiClient()

        # Redis client
        try:
            self.redis_client = redis.from_url(self.endpoints["redis"])
            await self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")

        # Database connection pool
        try:
            self.db_pool = await asyncpg.create_pool(
                self.endpoints["timescaledb"],
                min_size=1,
                max_size=5,
                command_timeout=30,
            )
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")

        logger.info("Test environment setup completed")

    async def cleanup(self):
        """Cleanup test resources"""
        logger.info("Cleaning up test environment...")

        if self.session:
            await self.session.close()

        if self.db_pool:
            await self.db_pool.close()

        if self.redis_client:
            await self.redis_client.close()

        logger.info("Test environment cleanup completed")

    def _record_test_result(
        self,
        test_name: str,
        status: str,
        duration: float,
        error: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        """Record test result"""
        result = TestResult(
            test_name=test_name,
            status=status,
            duration=duration,
            error=error,
            details=details,
        )
        self.test_results.append(result)

        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⏭️"
        logger.info(f"{status_emoji} {test_name} - {status} ({duration:.2f}s)")

        if error:
            logger.error(f"   Error: {error}")

    async def test_service_health_checks(self) -> List[TestResult]:
        """Test health endpoints for all services"""
        logger.info("Running service health checks...")

        health_checks = [
            HealthCheck("AI Engine Health", f"{self.endpoints['ai_engine']}/health"),
            HealthCheck(
                "Management API Health", f"{self.endpoints['mgmt_api']}/health"
            ),
            HealthCheck(
                "Control Plane Health", f"{self.endpoints['control_plane']}/health"
            ),
            HealthCheck(
                "UI Console Health", f"{self.endpoints['ui_console']}/api/health"
            ),
            HealthCheck(
                "Prometheus Health", f"{self.endpoints['prometheus']}/-/healthy"
            ),
            HealthCheck("Grafana Health", f"{self.endpoints['grafana']}/api/health"),
        ]

        results = []
        for check in health_checks:
            start_time = time.time()
            try:
                async with self.session.get(
                    check.url, headers=check.headers
                ) as response:
                    duration = time.time() - start_time

                    if response.status == check.expected_status:
                        self._record_test_result(check.name, "PASS", duration)
                    else:
                        error = f"Expected status {check.expected_status}, got {response.status}"
                        self._record_test_result(check.name, "FAIL", duration, error)

            except Exception as e:
                duration = time.time() - start_time
                self._record_test_result(check.name, "FAIL", duration, str(e))

        return results

    async def test_kubernetes_resources(self):
        """Test Kubernetes resource status"""
        logger.info("Testing Kubernetes resources...")

        # Test pods
        start_time = time.time()
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            pods = v1.list_namespaced_pod(namespace="cronos-ai")

            running_pods = [p for p in pods.items if p.status.phase == "Running"]
            failed_pods = [p for p in pods.items if p.status.phase == "Failed"]

            duration = time.time() - start_time

            if len(failed_pods) == 0:
                details = {
                    "total_pods": len(pods.items),
                    "running_pods": len(running_pods),
                    "failed_pods": len(failed_pods),
                }
                self._record_test_result(
                    "Kubernetes Pods Status", "PASS", duration, details=details
                )
            else:
                error = f"{len(failed_pods)} pods in failed state"
                self._record_test_result(
                    "Kubernetes Pods Status", "FAIL", duration, error
                )

        except Exception as e:
            duration = time.time() - start_time
            self._record_test_result("Kubernetes Pods Status", "FAIL", duration, str(e))

        # Test services
        start_time = time.time()
        try:
            services = v1.list_namespaced_service(namespace="cronos-ai")
            duration = time.time() - start_time

            details = {"service_count": len(services.items)}
            self._record_test_result(
                "Kubernetes Services", "PASS", duration, details=details
            )

        except Exception as e:
            duration = time.time() - start_time
            self._record_test_result("Kubernetes Services", "FAIL", duration, str(e))

        # Test deployments
        start_time = time.time()
        try:
            apps_v1 = client.AppsV1Api(self.k8s_client)
            deployments = apps_v1.list_namespaced_deployment(namespace="cronos-ai")

            unhealthy_deployments = []
            for deployment in deployments.items:
                if deployment.status.ready_replicas != deployment.status.replicas:
                    unhealthy_deployments.append(deployment.metadata.name)

            duration = time.time() - start_time

            if len(unhealthy_deployments) == 0:
                details = {"deployment_count": len(deployments.items)}
                self._record_test_result(
                    "Kubernetes Deployments", "PASS", duration, details=details
                )
            else:
                error = f"Unhealthy deployments: {unhealthy_deployments}"
                self._record_test_result(
                    "Kubernetes Deployments", "FAIL", duration, error
                )

        except Exception as e:
            duration = time.time() - start_time
            self._record_test_result("Kubernetes Deployments", "FAIL", duration, str(e))

    async def test_database_connectivity(self):
        """Test database connections and basic operations"""
        logger.info("Testing database connectivity...")

        if not self.db_pool:
            self._record_test_result(
                "Database Connection", "SKIP", 0, "No database pool available"
            )
            return

        # Test basic connection
        start_time = time.time()
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval("SELECT version()")
                duration = time.time() - start_time

                details = {"postgres_version": result.split()[1]}
                self._record_test_result(
                    "Database Connection", "PASS", duration, details=details
                )

        except Exception as e:
            duration = time.time() - start_time
            self._record_test_result("Database Connection", "FAIL", duration, str(e))

        # Test table existence
        start_time = time.time()
        try:
            async with self.db_pool.acquire() as conn:
                tables = await conn.fetch(
                    """
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """
                )

                expected_tables = [
                    "protocol_data",
                    "ai_model_results",
                    "security_events",
                    "system_metrics",
                ]
                existing_tables = [row["table_name"] for row in tables]
                missing_tables = [
                    t for t in expected_tables if t not in existing_tables
                ]

                duration = time.time() - start_time

                if len(missing_tables) == 0:
                    details = {"table_count": len(existing_tables)}
                    self._record_test_result(
                        "Database Schema", "PASS", duration, details=details
                    )
                else:
                    error = f"Missing tables: {missing_tables}"
                    self._record_test_result("Database Schema", "FAIL", duration, error)

        except Exception as e:
            duration = time.time() - start_time
            self._record_test_result("Database Schema", "FAIL", duration, str(e))

    async def test_redis_connectivity(self):
        """Test Redis connection and operations"""
        logger.info("Testing Redis connectivity...")

        if not self.redis_client:
            self._record_test_result(
                "Redis Connection", "SKIP", 0, "No Redis client available"
            )
            return

        start_time = time.time()
        try:
            # Test basic operations
            test_key = f"test_key_{uuid.uuid4()}"
            test_value = f"test_value_{datetime.now().isoformat()}"

            # Set and get
            await self.redis_client.set(test_key, test_value, ex=60)
            retrieved = await self.redis_client.get(test_key)

            # Cleanup
            await self.redis_client.delete(test_key)

            duration = time.time() - start_time

            if retrieved.decode() == test_value:
                details = {"operation": "set/get/delete"}
                self._record_test_result(
                    "Redis Operations", "PASS", duration, details=details
                )
            else:
                error = f"Value mismatch: expected {test_value}, got {retrieved}"
                self._record_test_result("Redis Operations", "FAIL", duration, error)

        except Exception as e:
            duration = time.time() - start_time
            self._record_test_result("Redis Operations", "FAIL", duration, str(e))

    async def test_kafka_connectivity(self):
        """Test Kafka connectivity and basic operations"""
        logger.info("Testing Kafka connectivity...")

        start_time = time.time()
        try:
            from kafka import KafkaProducer, KafkaConsumer

            # Test producer
            producer = KafkaProducer(
                bootstrap_servers=self.endpoints["kafka"].split(","),
                value_serializer=lambda x: json.dumps(x).encode("utf-8"),
                request_timeout_ms=10000,
                api_version=(0, 10, 1),
            )

            test_topic = "test-integration-topic"
            test_message = {
                "test_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "message": "Integration test message",
            }

            # Send message
            future = producer.send(test_topic, test_message)
            record_metadata = future.get(timeout=10)

            producer.close()

            duration = time.time() - start_time

            details = {
                "topic": record_metadata.topic,
                "partition": record_metadata.partition,
                "offset": record_metadata.offset,
            }
            self._record_test_result(
                "Kafka Producer", "PASS", duration, details=details
            )

        except Exception as e:
            duration = time.time() - start_time
            self._record_test_result("Kafka Producer", "FAIL", duration, str(e))

    async def test_ai_ml_pipeline(self):
        """Test AI/ML pipeline integration"""
        logger.info("Testing AI/ML pipeline...")

        start_time = time.time()
        try:
            # Test protocol discovery endpoint
            test_payload = {
                "packet_data": "474554202f20485454502f312e310d0a486f73743a206578616d706c652e636f6d0d0a0d0a",
                "protocol_hint": "http",
                "analysis_type": "full",
            }

            url = f"{self.endpoints['ai_engine']}/api/v1/protocol/discover"
            async with self.session.post(url, json=test_payload) as response:
                if response.status == 200:
                    result = await response.json()
                    duration = time.time() - start_time

                    if "protocol" in result and "confidence" in result:
                        details = {
                            "detected_protocol": result.get("protocol"),
                            "confidence": result.get("confidence"),
                            "processing_time_ms": result.get("processing_time_ms", 0),
                        }
                        self._record_test_result(
                            "AI Protocol Discovery", "PASS", duration, details=details
                        )
                    else:
                        error = "Invalid response format"
                        self._record_test_result(
                            "AI Protocol Discovery", "FAIL", duration, error
                        )
                else:
                    error = f"HTTP {response.status}: {await response.text()}"
                    duration = time.time() - start_time
                    self._record_test_result(
                        "AI Protocol Discovery", "FAIL", duration, error
                    )

        except Exception as e:
            duration = time.time() - start_time
            self._record_test_result("AI Protocol Discovery", "FAIL", duration, str(e))

        # Test model health
        start_time = time.time()
        try:
            url = f"{self.endpoints['ai_engine']}/api/v1/models/health"
            async with self.session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    duration = time.time() - start_time

                    healthy_models = [
                        m
                        for m in result.get("models", [])
                        if m.get("status") == "healthy"
                    ]

                    details = {
                        "total_models": len(result.get("models", [])),
                        "healthy_models": len(healthy_models),
                    }
                    self._record_test_result(
                        "AI Model Health", "PASS", duration, details=details
                    )
                else:
                    error = f"HTTP {response.status}"
                    duration = time.time() - start_time
                    self._record_test_result("AI Model Health", "FAIL", duration, error)

        except Exception as e:
            duration = time.time() - start_time
            self._record_test_result("AI Model Health", "FAIL", duration, str(e))

    async def test_security_integration(self):
        """Test security component integration"""
        logger.info("Testing security integration...")

        # Test authentication
        start_time = time.time()
        try:
            auth_payload = {"username": "test_user", "password": "test_password"}

            url = f"{self.endpoints['mgmt_api']}/auth/login"
            async with self.session.post(url, json=auth_payload) as response:
                duration = time.time() - start_time

                if response.status in [200, 401]:  # Both valid responses
                    details = {"status_code": response.status}
                    self._record_test_result(
                        "Authentication Endpoint", "PASS", duration, details=details
                    )
                else:
                    error = f"Unexpected status: {response.status}"
                    self._record_test_result(
                        "Authentication Endpoint", "FAIL", duration, error
                    )

        except Exception as e:
            duration = time.time() - start_time
            self._record_test_result(
                "Authentication Endpoint", "FAIL", duration, str(e)
            )

        # Test security metrics endpoint
        start_time = time.time()
        try:
            url = f"{self.endpoints['mgmt_api']}/api/v1/security/metrics"
            async with self.session.get(url) as response:
                duration = time.time() - start_time

                if response.status in [200, 401, 403]:  # Valid responses
                    details = {"status_code": response.status}
                    self._record_test_result(
                        "Security Metrics", "PASS", duration, details=details
                    )
                else:
                    error = f"Unexpected status: {response.status}"
                    self._record_test_result(
                        "Security Metrics", "FAIL", duration, error
                    )

        except Exception as e:
            duration = time.time() - start_time
            self._record_test_result("Security Metrics", "FAIL", duration, str(e))

    async def test_monitoring_stack(self):
        """Test monitoring and observability stack"""
        logger.info("Testing monitoring stack...")

        # Test Prometheus metrics
        start_time = time.time()
        try:
            url = f"{self.endpoints['prometheus']}/api/v1/query"
            params = {"query": "up"}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    duration = time.time() - start_time

                    if result.get("status") == "success":
                        metrics_count = len(result.get("data", {}).get("result", []))
                        details = {"active_targets": metrics_count}
                        self._record_test_result(
                            "Prometheus Metrics", "PASS", duration, details=details
                        )
                    else:
                        error = "Invalid Prometheus response"
                        self._record_test_result(
                            "Prometheus Metrics", "FAIL", duration, error
                        )
                else:
                    error = f"HTTP {response.status}"
                    duration = time.time() - start_time
                    self._record_test_result(
                        "Prometheus Metrics", "FAIL", duration, error
                    )

        except Exception as e:
            duration = time.time() - start_time
            self._record_test_result("Prometheus Metrics", "FAIL", duration, str(e))

        # Test Grafana API
        start_time = time.time()
        try:
            url = f"{self.endpoints['grafana']}/api/datasources"
            async with self.session.get(url) as response:
                duration = time.time() - start_time

                if response.status in [200, 401, 403]:  # Auth may be required
                    details = {"status_code": response.status}
                    self._record_test_result(
                        "Grafana API", "PASS", duration, details=details
                    )
                else:
                    error = f"Unexpected status: {response.status}"
                    self._record_test_result("Grafana API", "FAIL", duration, error)

        except Exception as e:
            duration = time.time() - start_time
            self._record_test_result("Grafana API", "FAIL", duration, str(e))

    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        logger.info("Testing end-to-end workflow...")

        start_time = time.time()
        try:
            # Simulate complete protocol analysis workflow
            workflow_steps = []

            # Step 1: Submit packet for analysis
            packet_payload = {
                "packet_data": "474554202f696e6465782e68746d6c20485454502f312e310d0a486f73743a206578616d706c652e636f6d0d0a557365722d4167656e743a204d6f7a696c6c612f352e300d0a0d0a",
                "source_ip": "192.168.1.100",
                "dest_ip": "93.184.216.34",
                "source_port": 45678,
                "dest_port": 80,
                "protocol_layer": "tcp",
                "timestamp": datetime.now().isoformat(),
            }

            url = f"{self.endpoints['ai_engine']}/api/v1/packet/analyze"
            async with self.session.post(url, json=packet_payload) as response:
                if response.status == 200:
                    analysis_result = await response.json()
                    workflow_steps.append(
                        {
                            "step": "packet_analysis",
                            "status": "success",
                            "analysis_id": analysis_result.get("analysis_id"),
                        }
                    )
                else:
                    workflow_steps.append(
                        {
                            "step": "packet_analysis",
                            "status": "failed",
                            "error": f"HTTP {response.status}",
                        }
                    )

            # Step 2: Check analysis status
            if workflow_steps[-1]["status"] == "success":
                analysis_id = workflow_steps[-1]["analysis_id"]
                url = f"{self.endpoints['ai_engine']}/api/v1/analysis/{analysis_id}/status"

                async with self.session.get(url) as response:
                    if response.status == 200:
                        status_result = await response.json()
                        workflow_steps.append(
                            {
                                "step": "status_check",
                                "status": "success",
                                "analysis_status": status_result.get("status"),
                            }
                        )
                    else:
                        workflow_steps.append(
                            {"step": "status_check", "status": "failed"}
                        )

            # Step 3: Retrieve results
            if len(workflow_steps) >= 2 and workflow_steps[-1]["status"] == "success":
                url = f"{self.endpoints['ai_engine']}/api/v1/analysis/{analysis_id}/results"

                async with self.session.get(url) as response:
                    if response.status == 200:
                        results = await response.json()
                        workflow_steps.append(
                            {
                                "step": "retrieve_results",
                                "status": "success",
                                "protocol_detected": results.get("protocol"),
                                "confidence": results.get("confidence"),
                            }
                        )
                    else:
                        workflow_steps.append(
                            {"step": "retrieve_results", "status": "failed"}
                        )

            duration = time.time() - start_time

            # Evaluate workflow success
            successful_steps = [s for s in workflow_steps if s["status"] == "success"]
            if len(successful_steps) == len(workflow_steps):
                details = {
                    "total_steps": len(workflow_steps),
                    "successful_steps": len(successful_steps),
                    "workflow_details": workflow_steps,
                }
                self._record_test_result(
                    "End-to-End Workflow", "PASS", duration, details=details
                )
            else:
                error = f"Only {len(successful_steps)}/{len(workflow_steps)} steps completed"
                details = {"workflow_details": workflow_steps}
                self._record_test_result(
                    "End-to-End Workflow", "FAIL", duration, error, details
                )

        except Exception as e:
            duration = time.time() - start_time
            self._record_test_result("End-to-End Workflow", "FAIL", duration, str(e))

    async def test_performance_benchmarks(self):
        """Test system performance under load"""
        logger.info("Testing performance benchmarks...")

        start_time = time.time()
        try:
            # Concurrent request test
            async def make_request(session, url, payload):
                try:
                    async with session.post(url, json=payload) as response:
                        return {
                            "status": response.status,
                            "response_time": time.time() - request_start,
                            "success": response.status == 200,
                        }
                except Exception as e:
                    return {
                        "status": None,
                        "response_time": time.time() - request_start,
                        "success": False,
                        "error": str(e),
                    }

            # Test payload
            test_payload = {
                "packet_data": "474554202f20485454502f312e31",
                "protocol_hint": "http",
            }

            url = f"{self.endpoints['ai_engine']}/api/v1/protocol/discover"
            concurrent_requests = 50

            # Execute concurrent requests
            tasks = []
            for i in range(concurrent_requests):
                request_start = time.time()
                task = make_request(self.session, url, test_payload)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            duration = time.time() - start_time

            # Analyze results
            successful_requests = [
                r for r in results if isinstance(r, dict) and r.get("success")
            ]
            response_times = [r["response_time"] for r in successful_requests]

            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                min_response_time = min(response_times)

                details = {
                    "total_requests": concurrent_requests,
                    "successful_requests": len(successful_requests),
                    "success_rate": len(successful_requests) / concurrent_requests,
                    "avg_response_time": avg_response_time,
                    "max_response_time": max_response_time,
                    "min_response_time": min_response_time,
                }

                if (
                    len(successful_requests) / concurrent_requests >= 0.95
                ):  # 95% success rate
                    self._record_test_result(
                        "Performance Load Test", "PASS", duration, details=details
                    )
                else:
                    error = f"Low success rate: {len(successful_requests)}/{concurrent_requests}"
                    self._record_test_result(
                        "Performance Load Test", "FAIL", duration, error, details
                    )
            else:
                error = "No successful requests"
                self._record_test_result(
                    "Performance Load Test", "FAIL", duration, error
                )

        except Exception as e:
            duration = time.time() - start_time
            self._record_test_result("Performance Load Test", "FAIL", duration, str(e))

    async def run_all_tests(self):
        """Execute all integration tests"""
        logger.info("Starting comprehensive integration test suite...")

        await self.setup()

        try:
            # Execute test suites
            await self.test_service_health_checks()
            await self.test_kubernetes_resources()
            await self.test_database_connectivity()
            await self.test_redis_connectivity()
            await self.test_kafka_connectivity()
            await self.test_ai_ml_pipeline()
            await self.test_security_integration()
            await self.test_monitoring_stack()
            await self.test_end_to_end_workflow()
            await self.test_performance_benchmarks()

        finally:
            await self.cleanup()

    def generate_report(self) -> Dict:
        """Generate comprehensive test report"""
        total_duration = time.time() - self.start_time

        passed_tests = [r for r in self.test_results if r.status == "PASS"]
        failed_tests = [r for r in self.test_results if r.status == "FAIL"]
        skipped_tests = [r for r in self.test_results if r.status == "SKIP"]

        report = {
            "summary": {
                "total_tests": len(self.test_results),
                "passed": len(passed_tests),
                "failed": len(failed_tests),
                "skipped": len(skipped_tests),
                "success_rate": (
                    len(passed_tests) / len(self.test_results)
                    if self.test_results
                    else 0
                ),
                "total_duration": total_duration,
                "timestamp": datetime.now().isoformat(),
            },
            "test_results": [
                {
                    "test_name": r.test_name,
                    "status": r.status,
                    "duration": r.duration,
                    "error": r.error,
                    "details": r.details,
                }
                for r in self.test_results
            ],
            "failed_tests": [
                {"test_name": r.test_name, "error": r.error, "duration": r.duration}
                for r in failed_tests
            ],
        }

        return report

    def print_summary(self):
        """Print test summary to console"""
        report = self.generate_report()
        summary = report["summary"]

        print("\n" + "=" * 80)
        print("CRONOS AI - INTEGRATION TEST RESULTS")
        print("=" * 80)

        print(f"Total Tests:     {summary['total_tests']}")
        print(f"Passed:          {summary['passed']} ✅")
        print(f"Failed:          {summary['failed']} ❌")
        print(f"Skipped:         {summary['skipped']} ⏭️")
        print(f"Success Rate:    {summary['success_rate']:.1%}")
        print(f"Total Duration:  {summary['total_duration']:.2f}s")

        if report["failed_tests"]:
            print("\nFAILED TESTS:")
            print("-" * 40)
            for test in report["failed_tests"]:
                print(f"❌ {test['test_name']}")
                print(f"   Error: {test['error']}")
                print(f"   Duration: {test['duration']:.2f}s")
                print()

        print("=" * 80)

        return summary["success_rate"] >= 0.95  # 95% pass rate required


def main():
    """Main test execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="CRONOS AI Integration Test Suite")
    parser.add_argument(
        "--config",
        default="config/cronos_ai.production.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--output",
        default="/tmp/integration-test-report.json",
        help="Output report file path",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, cleaning up...")
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run tests
    tester = ProductionIntegrationTester(args.config)

    async def run_tests():
        await tester.run_all_tests()

        # Generate report
        report = tester.generate_report()

        # Save report
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Test report saved to {args.output}")

        # Print summary
        success = tester.print_summary()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
