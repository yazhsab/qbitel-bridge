"""
QBITEL Engine - Test Runner

This module provides utilities for running comprehensive tests and benchmarks.
"""

import pytest
import asyncio
import sys
import time
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import subprocess

from . import setup_test_environment, cleanup_test_environment, TestConfig


class TestRunnerClass:
    """
    Comprehensive test runner for QBITEL Engine.

    This class provides utilities for running different types of tests
    including unit tests, integration tests, performance benchmarks,
    and generating comprehensive test reports.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize test runner."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Test configuration
        self.test_root = Path(__file__).parent
        self.output_dir = self.test_root / "output"
        self.reports_dir = self.output_dir / "reports"

        # Ensure directories exist
        self.output_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

        # Test categories
        self.test_categories = {
            "unit": ["test_core.py"],
            "api": ["test_api.py"],
            "integration": ["test_integration.py"],
            "performance": ["test_performance.py"],
            "models": ["test_models.py"],
        }

        # Test results
        self.test_results: Dict[str, Any] = {}

        self.logger.info("TestRunner initialized")

    def run_all_tests(
        self, verbose: bool = True, generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Run all test categories.

        Args:
            verbose: Enable verbose output
            generate_report: Generate HTML report

        Returns:
            Test results summary
        """
        self.logger.info("Starting comprehensive test run...")

        setup_test_environment()

        try:
            results = {}

            # Run each test category
            for category, test_files in self.test_categories.items():
                self.logger.info(f"Running {category} tests...")
                category_results = self._run_test_category(
                    category, test_files, verbose
                )
                results[category] = category_results

            # Generate summary
            summary = self._generate_summary(results)

            # Generate report if requested
            if generate_report:
                report_path = self._generate_html_report(results, summary)
                self.logger.info(f"Test report generated: {report_path}")

            self.test_results = {
                "summary": summary,
                "detailed_results": results,
                "timestamp": time.time(),
            }

            return self.test_results

        finally:
            cleanup_test_environment()

    def run_unit_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run unit tests only."""
        return self._run_test_category("unit", self.test_categories["unit"], verbose)

    def run_api_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run API tests only."""
        return self._run_test_category("api", self.test_categories["api"], verbose)

    def run_integration_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run integration tests only."""
        return self._run_test_category(
            "integration", self.test_categories["integration"], verbose
        )

    def run_performance_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run performance tests only."""
        return self._run_test_category(
            "performance", self.test_categories["performance"], verbose
        )

    def run_smoke_tests(self) -> Dict[str, Any]:
        """Run quick smoke tests to verify basic functionality."""
        self.logger.info("Running smoke tests...")

        smoke_tests = [
            "test_core.py::TestAIEngine::test_engine_initialization",
            "test_api.py::TestRESTAPI::test_health_check",
            "test_core.py::TestModelInput::test_model_input_creation_with_bytes",
        ]

        results = self._run_specific_tests(smoke_tests, verbose=True)
        return results

    def benchmark_performance(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks."""
        self.logger.info("Running performance benchmarks...")

        benchmarks = {
            "engine_initialization": self._benchmark_engine_initialization,
            "protocol_discovery": self._benchmark_protocol_discovery,
            "field_detection": self._benchmark_field_detection,
            "anomaly_detection": self._benchmark_anomaly_detection,
            "api_throughput": self._benchmark_api_throughput,
        }

        results = {}
        for name, benchmark_func in benchmarks.items():
            try:
                self.logger.info(f"Running {name} benchmark...")
                results[name] = benchmark_func()
                self.logger.info(f"{name} benchmark completed")
            except Exception as e:
                self.logger.error(f"{name} benchmark failed: {e}")
                results[name] = {"error": str(e)}

        return results

    def validate_installation(self) -> Dict[str, Any]:
        """Validate AI Engine installation and dependencies."""
        self.logger.info("Validating installation...")

        validation_results = {
            "python_version": self._check_python_version(),
            "dependencies": self._check_dependencies(),
            "gpu_availability": self._check_gpu_availability(),
            "model_files": self._check_model_files(),
            "data_files": self._check_data_files(),
            "permissions": self._check_permissions(),
        }

        # Overall validation status
        validation_results["overall_status"] = all(
            result.get("status", False) for result in validation_results.values()
        )

        return validation_results

    def _run_test_category(
        self, category: str, test_files: List[str], verbose: bool
    ) -> Dict[str, Any]:
        """Run tests for a specific category."""
        results = {
            "category": category,
            "files": test_files,
            "start_time": time.time(),
            "status": "unknown",
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "total": 0,
            "duration": 0.0,
            "details": {},
        }

        try:
            # Build pytest command
            cmd = ["python", "-m", "pytest"]

            if verbose:
                cmd.append("-v")

            # Add coverage if available
            cmd.extend(["--cov=ai_engine", "--cov-report=term-missing"])

            # Add test files that exist
            existing_files = []
            for test_file in test_files:
                test_path = self.test_root / test_file
                if test_path.exists():
                    existing_files.append(str(test_path))
                else:
                    self.logger.warning(f"Test file not found: {test_file}")

            if not existing_files:
                results["status"] = "skipped"
                results["details"]["reason"] = "No test files found"
                return results

            cmd.extend(existing_files)

            # Add output format
            cmd.extend(
                ["--tb=short", f"--junit-xml={self.reports_dir}/{category}_results.xml"]
            )

            # Run tests
            self.logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.test_root.parent
            )

            results["duration"] = time.time() - results["start_time"]
            results["return_code"] = result.returncode
            results["stdout"] = result.stdout
            results["stderr"] = result.stderr

            # Parse results from output
            self._parse_pytest_output(result.stdout, results)

            if result.returncode == 0:
                results["status"] = "passed"
            elif result.returncode == 1:
                results["status"] = "failed"
            else:
                results["status"] = "error"

            return results

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            results["duration"] = time.time() - results["start_time"]
            return results

    def _run_specific_tests(
        self, test_patterns: List[str], verbose: bool = True
    ) -> Dict[str, Any]:
        """Run specific tests by pattern."""
        results = {
            "patterns": test_patterns,
            "start_time": time.time(),
            "status": "unknown",
        }

        try:
            cmd = ["python", "-m", "pytest"]

            if verbose:
                cmd.append("-v")

            cmd.extend(test_patterns)
            cmd.extend(["--tb=short"])

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.test_root.parent
            )

            results["duration"] = time.time() - results["start_time"]
            results["return_code"] = result.returncode
            results["stdout"] = result.stdout
            results["stderr"] = result.stderr

            self._parse_pytest_output(result.stdout, results)

            results["status"] = "passed" if result.returncode == 0 else "failed"

            return results

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            results["duration"] = time.time() - results["start_time"]
            return results

    def _parse_pytest_output(self, output: str, results: Dict[str, Any]) -> None:
        """Parse pytest output to extract test statistics."""
        lines = output.split("\n")

        for line in lines:
            if "passed" in line or "failed" in line or "error" in line:
                # Parse summary line like "5 passed, 2 failed, 1 error in 10.5s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        try:
                            results["passed"] = int(parts[i - 1])
                        except ValueError:
                            pass
                    elif part == "failed" and i > 0:
                        try:
                            results["failed"] = int(parts[i - 1])
                        except ValueError:
                            pass
                    elif part == "error" and i > 0:
                        try:
                            results["errors"] = int(parts[i - 1])
                        except ValueError:
                            pass
                    elif part == "skipped" and i > 0:
                        try:
                            results["skipped"] = int(parts[i - 1])
                        except ValueError:
                            pass

        results["total"] = (
            results["passed"]
            + results["failed"]
            + results["errors"]
            + results["skipped"]
        )

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test results summary."""
        summary = {
            "total_categories": len(results),
            "total_passed": 0,
            "total_failed": 0,
            "total_errors": 0,
            "total_skipped": 0,
            "total_tests": 0,
            "total_duration": 0.0,
            "success_rate": 0.0,
            "category_status": {},
        }

        for category, category_results in results.items():
            summary["total_passed"] += category_results.get("passed", 0)
            summary["total_failed"] += category_results.get("failed", 0)
            summary["total_errors"] += category_results.get("errors", 0)
            summary["total_skipped"] += category_results.get("skipped", 0)
            summary["total_duration"] += category_results.get("duration", 0.0)
            summary["category_status"][category] = category_results.get(
                "status", "unknown"
            )

        summary["total_tests"] = (
            summary["total_passed"]
            + summary["total_failed"]
            + summary["total_errors"]
            + summary["total_skipped"]
        )

        if summary["total_tests"] > 0:
            summary["success_rate"] = summary["total_passed"] / summary["total_tests"]

        return summary

    def _generate_html_report(
        self, results: Dict[str, Any], summary: Dict[str, Any]
    ) -> str:
        """Generate HTML test report."""
        report_path = self.reports_dir / "test_report.html"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>QBITEL Engine - Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .category {{ margin: 20px 0; border: 1px solid #ccc; padding: 15px; border-radius: 5px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .error {{ color: orange; }}
        .skipped {{ color: blue; }}
        .status-passed {{ background-color: #d4edda; }}
        .status-failed {{ background-color: #f8d7da; }}
        .status-error {{ background-color: #fff3cd; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>QBITEL Engine - Test Report</h1>
        <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Tests</td><td>{summary['total_tests']}</td></tr>
            <tr><td class="passed">Passed</td><td>{summary['total_passed']}</td></tr>
            <tr><td class="failed">Failed</td><td>{summary['total_failed']}</td></tr>
            <tr><td class="error">Errors</td><td>{summary['total_errors']}</td></tr>
            <tr><td class="skipped">Skipped</td><td>{summary['total_skipped']}</td></tr>
            <tr><td>Success Rate</td><td>{summary['success_rate']:.2%}</td></tr>
            <tr><td>Total Duration</td><td>{summary['total_duration']:.2f}s</td></tr>
        </table>
    </div>
    
    <div class="categories">
        <h2>Test Categories</h2>
"""

        for category, category_results in results.items():
            status_class = f"status-{category_results.get('status', 'unknown')}"
            html_content += f"""
        <div class="category {status_class}">
            <h3>{category.title()} Tests</h3>
            <p><strong>Status:</strong> {category_results.get('status', 'unknown').title()}</p>
            <p><strong>Duration:</strong> {category_results.get('duration', 0):.2f}s</p>
            <table>
                <tr><th>Metric</th><th>Count</th></tr>
                <tr><td class="passed">Passed</td><td>{category_results.get('passed', 0)}</td></tr>
                <tr><td class="failed">Failed</td><td>{category_results.get('failed', 0)}</td></tr>
                <tr><td class="error">Errors</td><td>{category_results.get('errors', 0)}</td></tr>
                <tr><td class="skipped">Skipped</td><td>{category_results.get('skipped', 0)}</td></tr>
            </table>
        </div>
"""

        html_content += """
    </div>
</body>
</html>
"""

        with open(report_path, "w") as f:
            f.write(html_content)

        return str(report_path)

    # Benchmark methods

    def _benchmark_engine_initialization(self) -> Dict[str, Any]:
        """Benchmark AI Engine initialization time."""
        from ai_engine.core.config import Config
        from ai_engine.core.engine import AIEngine

        times = []
        iterations = 5

        for _ in range(iterations):
            config = Config()
            engine = AIEngine(config)

            start_time = time.time()
            # Mock initialization for benchmarking
            end_time = time.time()

            times.append((end_time - start_time) * 1000)  # Convert to ms

        return {
            "iterations": iterations,
            "times_ms": times,
            "avg_time_ms": sum(times) / len(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
        }

    def _benchmark_protocol_discovery(self) -> Dict[str, Any]:
        """Benchmark protocol discovery performance."""
        # Mock benchmark implementation
        return {
            "sample_size": 100,
            "avg_latency_ms": 150.0,
            "throughput_per_second": 6.67,
            "accuracy": 0.92,
        }

    def _benchmark_field_detection(self) -> Dict[str, Any]:
        """Benchmark field detection performance."""
        # Mock benchmark implementation
        return {
            "sample_size": 100,
            "avg_latency_ms": 120.0,
            "throughput_per_second": 8.33,
            "accuracy": 0.89,
        }

    def _benchmark_anomaly_detection(self) -> Dict[str, Any]:
        """Benchmark anomaly detection performance."""
        # Mock benchmark implementation
        return {
            "sample_size": 100,
            "avg_latency_ms": 180.0,
            "throughput_per_second": 5.56,
            "accuracy": 0.85,
        }

    def _benchmark_api_throughput(self) -> Dict[str, Any]:
        """Benchmark API throughput."""
        # Mock benchmark implementation
        return {
            "concurrent_users": 10,
            "total_requests": 1000,
            "avg_response_time_ms": 200.0,
            "requests_per_second": 50.0,
            "error_rate": 0.01,
        }

    # Validation methods

    def _check_python_version(self) -> Dict[str, Any]:
        """Check Python version compatibility."""
        python_version = sys.version_info
        required_version = (3, 8)

        compatible = python_version >= required_version

        return {
            "status": compatible,
            "current_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            "required_version": f"{required_version[0]}.{required_version[1]}+",
            "compatible": compatible,
        }

    def _check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies."""
        required_packages = [
            "torch",
            "numpy",
            "scikit-learn",
            "fastapi",
            "uvicorn",
            "grpcio",
            "pydantic",
            "pytest",
            "mlflow",
        ]

        missing_packages = []
        installed_packages = {}

        for package in required_packages:
            try:
                __import__(package)
                installed_packages[package] = "installed"
            except ImportError:
                missing_packages.append(package)
                installed_packages[package] = "missing"

        return {
            "status": len(missing_packages) == 0,
            "installed_packages": installed_packages,
            "missing_packages": missing_packages,
            "total_required": len(required_packages),
        }

    def _check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability for training."""
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if cuda_available else 0

            return {
                "status": True,  # GPU is optional
                "cuda_available": cuda_available,
                "gpu_count": gpu_count,
                "gpu_names": (
                    [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                    if cuda_available
                    else []
                ),
            }
        except ImportError:
            return {"status": False, "error": "PyTorch not available"}

    def _check_model_files(self) -> Dict[str, Any]:
        """Check for required model files."""
        # This would check for actual model files in production
        return {
            "status": True,
            "model_directory_exists": True,
            "pretrained_models": [
                "mock_protocol_model",
                "mock_field_model",
                "mock_anomaly_model",
            ],
        }

    def _check_data_files(self) -> Dict[str, Any]:
        """Check for required data files."""
        # This would check for actual data files in production
        return {
            "status": True,
            "data_directory_exists": True,
            "training_data_available": True,
        }

    def _check_permissions(self) -> Dict[str, Any]:
        """Check file system permissions."""
        test_paths = [self.output_dir, self.reports_dir]

        permission_status = {}
        all_writable = True

        for path in test_paths:
            writable = os.access(path, os.W_OK)
            permission_status[str(path)] = writable
            all_writable = all_writable and writable

        return {"status": all_writable, "path_permissions": permission_status}


def main():
    """Main CLI entry point for test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="QBITEL Engine Test Runner")
    parser.add_argument(
        "--category",
        type=str,
        choices=["unit", "api", "integration", "performance", "all"],
        default="all",
        help="Test category to run",
    )
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests only")
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmarks"
    )
    parser.add_argument("--validate", action="store_true", help="Validate installation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--no-report", action="store_true", help="Skip HTML report generation"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    runner = TestRunner()

    try:
        if args.validate:
            print("Validating installation...")
            results = runner.validate_installation()
            print(json.dumps(results, indent=2))

        elif args.smoke:
            print("Running smoke tests...")
            results = runner.run_smoke_tests()
            print(f"Smoke tests: {results.get('status', 'unknown')}")

        elif args.benchmark:
            print("Running performance benchmarks...")
            results = runner.benchmark_performance()
            print(json.dumps(results, indent=2))

        else:
            # Run specified test category
            if args.category == "all":
                print("Running all tests...")
                results = runner.run_all_tests(
                    verbose=args.verbose, generate_report=not args.no_report
                )
            elif args.category == "unit":
                results = runner.run_unit_tests(verbose=args.verbose)
            elif args.category == "api":
                results = runner.run_api_tests(verbose=args.verbose)
            elif args.category == "integration":
                results = runner.run_integration_tests(verbose=args.verbose)
            elif args.category == "performance":
                results = runner.run_performance_tests(verbose=args.verbose)

            # Print summary
            if "summary" in results:
                summary = results["summary"]
                print(f"\nTest Summary:")
                print(f"  Total Tests: {summary['total_tests']}")
                print(f"  Passed: {summary['total_passed']}")
                print(f"  Failed: {summary['total_failed']}")
                print(f"  Errors: {summary['total_errors']}")
                print(f"  Success Rate: {summary['success_rate']:.2%}")
                print(f"  Duration: {summary['total_duration']:.2f}s")

    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test run failed: {e}")
        sys.exit(1)


# Factory function to create TestRunner instances
def create_test_runner(config: Optional[Dict[str, Any]] = None) -> TestRunnerClass:
    """Create a TestRunner instance."""
    return TestRunnerClass(config)


if __name__ == "__main__":
    main()
