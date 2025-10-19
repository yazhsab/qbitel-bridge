"""
CRONOS AI - Comprehensive Load Testing Suite

Production-grade load tests to validate performance, scalability, and reliability
under various load conditions.

Test Types:
1. Baseline Test - Establish normal performance metrics
2. Stress Test - Find breaking point (10x normal load)
3. Soak Test - Sustained load for memory leaks
4. Spike Test - Sudden traffic surges
5. Scalability Test - Measure linear scaling
"""

import asyncio
import aiohttp
import time
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    base_url: str = "http://localhost:8080"
    num_users: int = 100
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    think_time_seconds: float = 1.0
    timeout_seconds: int = 30


@dataclass
class RequestResult:
    """Result of a single request."""
    endpoint: str
    method: str
    status_code: int
    duration: float
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LoadTestResults:
    """Aggregated results from load test."""
    test_name: str
    config: LoadTestConfig
    start_time: datetime
    end_time: datetime
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    request_results: List[RequestResult] = field(default_factory=list)

    # Performance metrics
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    @property
    def requests_per_second(self) -> float:
        return self.total_requests / self.duration_seconds if self.duration_seconds > 0 else 0

    @property
    def success_rate(self) -> float:
        return (self.successful_requests / self.total_requests * 100
                if self.total_requests > 0 else 0)

    @property
    def error_rate(self) -> float:
        return 100 - self.success_rate

    def get_latency_percentiles(self, percentiles: List[int] = [50, 75, 90, 95, 99]) -> Dict[int, float]:
        """Calculate latency percentiles."""
        durations = [r.duration for r in self.request_results if r.success]
        if not durations:
            return {p: 0.0 for p in percentiles}

        durations.sort()
        result = {}
        for p in percentiles:
            index = int(len(durations) * (p / 100.0))
            result[p] = durations[min(index, len(durations) - 1)]
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        latencies = self.get_latency_percentiles()

        return {
            "test_name": self.test_name,
            "config": {
                "num_users": self.config.num_users,
                "duration": self.config.duration_seconds,
                "ramp_up": self.config.ramp_up_seconds,
            },
            "execution": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_seconds": self.duration_seconds,
            },
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "requests_per_second": round(self.requests_per_second, 2),
            },
            "performance": {
                "success_rate_percent": round(self.success_rate, 2),
                "error_rate_percent": round(self.error_rate, 2),
                "latency_p50_ms": round(latencies[50] * 1000, 2),
                "latency_p75_ms": round(latencies[75] * 1000, 2),
                "latency_p90_ms": round(latencies[90] * 1000, 2),
                "latency_p95_ms": round(latencies[95] * 1000, 2),
                "latency_p99_ms": round(latencies[99] * 1000, 2),
            },
            "slo_compliance": self.check_slo_compliance(),
        }

    def check_slo_compliance(self) -> Dict[str, bool]:
        """Check if test results meet SLOs."""
        latencies = self.get_latency_percentiles()

        return {
            "availability_99.9_percent": self.success_rate >= 99.9,
            "p95_latency_under_1s": latencies[95] < 1.0,
            "p99_latency_under_2s": latencies[99] < 2.0,
            "error_rate_under_0.1_percent": self.error_rate < 0.1,
        }


class LoadTestRunner:
    """Runs comprehensive load tests."""

    def __init__(self, config: LoadTestConfig):
        """
        Initialize load test runner.

        Args:
            config: Load test configuration
        """
        self.config = config
        self.results: Optional[LoadTestResults] = None

    async def run_test(self, test_name: str, test_scenario: callable) -> LoadTestResults:
        """
        Run a load test scenario.

        Args:
            test_name: Name of the test
            test_scenario: Async function that makes requests

        Returns:
            LoadTestResults: Test results
        """
        print(f"\n{'='*80}")
        print(f"Starting: {test_name}")
        print(f"Configuration: {self.config.num_users} users, {self.config.duration_seconds}s duration")
        print(f"{'='*80}\n")

        results = LoadTestResults(
            test_name=test_name,
            config=self.config,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),  # Will be updated
        )

        # Create async tasks for virtual users
        tasks = []
        ramp_up_delay = self.config.ramp_up_seconds / self.config.num_users if self.config.num_users > 0 else 0

        for user_id in range(self.config.num_users):
            task = asyncio.create_task(
                self._run_user(user_id, test_scenario, results, ramp_up_delay * user_id)
            )
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        results.end_time = datetime.utcnow()
        self.results = results

        # Print summary
        self._print_summary(results)

        return results

    async def _run_user(
        self,
        user_id: int,
        test_scenario: callable,
        results: LoadTestResults,
        initial_delay: float
    ):
        """Run a single virtual user."""
        # Ramp-up delay
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)

        end_time = datetime.utcnow() + timedelta(seconds=self.config.duration_seconds)

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        ) as session:
            while datetime.utcnow() < end_time:
                try:
                    # Execute test scenario
                    result = await test_scenario(session, user_id)

                    results.total_requests += 1
                    if result.success:
                        results.successful_requests += 1
                    else:
                        results.failed_requests += 1

                    results.request_results.append(result)

                    # Think time (simulate user delay)
                    await asyncio.sleep(self.config.think_time_seconds)

                except Exception as e:
                    print(f"User {user_id} error: {e}")

    def _print_summary(self, results: LoadTestResults):
        """Print test summary."""
        summary = results.get_summary()

        print(f"\n{'='*80}")
        print(f"Test Results: {summary['test_name']}")
        print(f"{'='*80}")
        print(f"\nExecution:")
        print(f"  Duration: {summary['execution']['duration_seconds']:.2f}s")
        print(f"\nRequests:")
        print(f"  Total: {summary['requests']['total']}")
        print(f"  Successful: {summary['requests']['successful']}")
        print(f"  Failed: {summary['requests']['failed']}")
        print(f"  Throughput: {summary['requests']['requests_per_second']:.2f} req/s")
        print(f"\nPerformance:")
        print(f"  Success Rate: {summary['performance']['success_rate_percent']:.2f}%")
        print(f"  Error Rate: {summary['performance']['error_rate_percent']:.2f}%")
        print(f"  Latency p50: {summary['performance']['latency_p50_ms']:.2f}ms")
        print(f"  Latency p75: {summary['performance']['latency_p75_ms']:.2f}ms")
        print(f"  Latency p90: {summary['performance']['latency_p90_ms']:.2f}ms")
        print(f"  Latency p95: {summary['performance']['latency_p95_ms']:.2f}ms")
        print(f"  Latency p99: {summary['performance']['latency_p99_ms']:.2f}ms")
        print(f"\nSLO Compliance:")
        for check, passed in summary['slo_compliance'].items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {check}: {status}")
        print(f"{'='*80}\n")


# Test Scenarios
async def health_check_scenario(session: aiohttp.ClientSession, user_id: int) -> RequestResult:
    """Test scenario: Health check endpoint."""
    start_time = time.time()

    try:
        async with session.get("http://localhost:8080/health") as response:
            duration = time.time() - start_time
            return RequestResult(
                endpoint="/health",
                method="GET",
                status_code=response.status,
                duration=duration,
                success=response.status == 200,
            )
    except Exception as e:
        duration = time.time() - start_time
        return RequestResult(
            endpoint="/health",
            method="GET",
            status_code=0,
            duration=duration,
            success=False,
            error=str(e),
        )


async def protocol_discovery_scenario(session: aiohttp.ClientSession, user_id: int) -> RequestResult:
    """Test scenario: Protocol discovery."""
    start_time = time.time()

    # Sample packet data (base64 encoded)
    payload = {
        "packet_data": "RVRIAAAAAAAAAAAAAAAAAAAAAAgARQAAKAABAABAAQO9wKgAAcCoAAEAAP8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==",
        "timeout": 5
    }

    try:
        async with session.post(
            "http://localhost:8080/api/v1/discover",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            duration = time.time() - start_time
            return RequestResult(
                endpoint="/api/v1/discover",
                method="POST",
                status_code=response.status,
                duration=duration,
                success=response.status == 200,
            )
    except Exception as e:
        duration = time.time() - start_time
        return RequestResult(
            endpoint="/api/v1/discover",
            method="POST",
            status_code=0,
            duration=duration,
            success=False,
            error=str(e),
        )


# Load Test Definitions
async def baseline_test():
    """
    Baseline Test: Establish normal performance metrics.

    Goals:
    - Measure performance under normal load
    - Establish baseline metrics (latency, throughput)
    - Validate system handles expected load
    """
    config = LoadTestConfig(
        num_users=10,
        duration_seconds=60,
        ramp_up_seconds=10,
        think_time_seconds=1.0,
    )

    runner = LoadTestRunner(config)
    results = await runner.run_test("Baseline Test (10 users, 60s)", health_check_scenario)
    return results


async def stress_test():
    """
    Stress Test: Find the breaking point.

    Goals:
    - Identify maximum capacity
    - Find breaking point (10x normal load)
    - Measure degradation under stress
    - Validate error handling under overload
    """
    config = LoadTestConfig(
        num_users=100,
        duration_seconds=120,
        ramp_up_seconds=20,
        think_time_seconds=0.5,
    )

    runner = LoadTestRunner(config)
    results = await runner.run_test("Stress Test (100 users, 120s)", health_check_scenario)
    return results


async def soak_test():
    """
    Soak Test: Sustained load to find memory leaks.

    Goals:
    - Identify memory leaks
    - Verify stability under sustained load
    - Test connection pool behavior over time
    - Validate resource cleanup
    """
    config = LoadTestConfig(
        num_users=20,
        duration_seconds=600,  # 10 minutes
        ramp_up_seconds=30,
        think_time_seconds=2.0,
    )

    runner = LoadTestRunner(config)
    results = await runner.run_test("Soak Test (20 users, 10 minutes)", health_check_scenario)
    return results


async def spike_test():
    """
    Spike Test: Sudden traffic surges.

    Goals:
    - Test autoscaling responsiveness
    - Validate rate limiting
    - Measure recovery time
    - Test circuit breaker behavior
    """
    config = LoadTestConfig(
        num_users=200,
        duration_seconds=30,
        ramp_up_seconds=5,  # Very fast ramp-up for spike
        think_time_seconds=0.1,
    )

    runner = LoadTestRunner(config)
    results = await runner.run_test("Spike Test (200 users, 30s, fast ramp)", health_check_scenario)
    return results


async def scalability_test():
    """
    Scalability Test: Measure linear scaling.

    Goals:
    - Verify linear scaling with load
    - Test connection pool scaling
    - Measure resource utilization
    - Validate horizontal scaling
    """
    results_by_load = {}

    for num_users in [10, 25, 50, 100]:
        config = LoadTestConfig(
            num_users=num_users,
            duration_seconds=60,
            ramp_up_seconds=10,
            think_time_seconds=1.0,
        )

        runner = LoadTestRunner(config)
        results = await runner.run_test(
            f"Scalability Test ({num_users} users)",
            health_check_scenario
        )
        results_by_load[num_users] = results

    # Analyze scalability
    print("\n" + "="*80)
    print("Scalability Analysis")
    print("="*80)
    print(f"{'Users':<10} {'RPS':<15} {'p95 Latency (ms)':<20} {'Success Rate %'}")
    print("-"*80)

    for num_users, results in results_by_load.items():
        summary = results.get_summary()
        print(f"{num_users:<10} "
              f"{summary['requests']['requests_per_second']:<15.2f} "
              f"{summary['performance']['latency_p95_ms']:<20.2f} "
              f"{summary['performance']['success_rate_percent']:.2f}%")

    return results_by_load


# Main execution
async def main():
    """Run all load tests."""
    print(f"\n{'#'*80}")
    print("CRONOS AI - Comprehensive Load Testing Suite")
    print(f"{'#'*80}\n")

    all_results = {}

    # 1. Baseline Test
    print("\n📊 Test 1/5: Baseline Test")
    all_results['baseline'] = await baseline_test()

    # 2. Stress Test
    print("\n💪 Test 2/5: Stress Test")
    all_results['stress'] = await stress_test()

    # 3. Spike Test (shorter, so run before soak)
    print("\n⚡ Test 3/5: Spike Test")
    all_results['spike'] = await spike_test()

    # 4. Scalability Test
    print("\n📈 Test 4/5: Scalability Test")
    all_results['scalability'] = await scalability_test()

    # 5. Soak Test (longest, run last - optional, comment out for quick runs)
    # print("\n🕒 Test 5/5: Soak Test (10 minutes)")
    # all_results['soak'] = await soak_test()

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - All Tests")
    print("="*80)

    for test_name, results in all_results.items():
        if test_name == 'scalability':
            continue  # Already printed

        if isinstance(results, LoadTestResults):
            summary = results.get_summary()
            slo_passed = all(summary['slo_compliance'].values())
            status = "✅ PASS" if slo_passed else "❌ FAIL"

            print(f"\n{test_name.upper()}: {status}")
            print(f"  RPS: {summary['requests']['requests_per_second']:.2f}")
            print(f"  Success Rate: {summary['performance']['success_rate_percent']:.2f}%")
            print(f"  p95 Latency: {summary['performance']['latency_p95_ms']:.2f}ms")

    # Save results to JSON
    with open('load_test_results.json', 'w') as f:
        json.dump({
            name: results.get_summary() if isinstance(results, LoadTestResults) else str(results)
            for name, results in all_results.items()
        }, f, indent=2)

    print(f"\n📁 Results saved to: load_test_results.json")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
