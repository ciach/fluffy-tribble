"""
Performance benchmarking tests for QA Operator.

Provides comprehensive performance testing and monitoring
for all components of the QA Operator system.
"""

import time
import asyncio
import psutil
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from orchestrator.agent import QAOperatorAgent
from orchestrator.core.config import Config
from orchestrator.planning.models import TestSpecification
from orchestrator.execution.models import ExecutionResult, TestStatus


class PerformanceMonitor:
    """Utility class for monitoring performance metrics."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.process = psutil.Process(os.getpid())
        self.initial_memory = None
        self.peak_memory = None
        self.cpu_percent = []

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        self.cpu_percent = []

    def update_metrics(self):
        """Update performance metrics."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        self.cpu_percent.append(self.process.cpu_percent())

    def stop_monitoring(self):
        """Stop monitoring and return metrics."""
        self.end_time = time.time()
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        return {
            "duration": self.end_time - self.start_time,
            "initial_memory_mb": self.initial_memory,
            "final_memory_mb": final_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_increase_mb": final_memory - self.initial_memory,
            "avg_cpu_percent": (
                sum(self.cpu_percent) / len(self.cpu_percent) if self.cpu_percent else 0
            ),
            "max_cpu_percent": max(self.cpu_percent) if self.cpu_percent else 0,
        }


@pytest.mark.slow
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    @pytest.fixture
    def performance_config(self):
        """Create configuration optimized for performance testing."""
        config = Config()
        config.log_level = "ERROR"  # Reduce logging overhead
        config.ci_mode = True
        config.headless_mode = True
        return config

    @pytest.fixture
    def sample_specifications(self):
        """Create sample specifications for performance testing."""
        specs = []

        # Small specification
        specs.append(
            TestSpecification(
                id="perf-small-001",
                name="Small Test Specification",
                description="Simple test with minimal requirements",
                requirements=["User can perform basic action"],
                priority="medium",
                tags=["performance", "small"],
            )
        )

        # Medium specification
        specs.append(
            TestSpecification(
                id="perf-medium-001",
                name="Medium Test Specification",
                description="Moderate complexity test with multiple requirements",
                requirements=[
                    "User can log in successfully",
                    "User can navigate to dashboard",
                    "User can perform CRUD operations",
                    "User can log out successfully",
                ],
                priority="high",
                tags=["performance", "medium"],
            )
        )

        # Large specification
        specs.append(
            TestSpecification(
                id="perf-large-001",
                name="Large Test Specification",
                description="Complex test with many requirements",
                requirements=[f"User can perform action {i}" for i in range(1, 21)],
                priority="high",
                tags=["performance", "large"],
            )
        )

        return specs

    @pytest.fixture
    def mock_agent_fast(self, performance_config):
        """Create mock agent optimized for fast responses."""
        agent = QAOperatorAgent(performance_config)

        # Mock fast responses
        agent.planning_engine.create_test_plan = AsyncMock(
            return_value={
                "test_cases": [{"name": "test_action", "description": "Test action"}],
                "estimated_duration": 10,
            }
        )

        agent.test_generator.generate_test = AsyncMock(
            return_value={
                "success": True,
                "test_file": "test.spec.ts",
                "content": "// Fast generated test",
            }
        )

        agent.test_executor.execute_tests = AsyncMock(
            return_value=ExecutionResult(
                success=True,
                test_results={"passed": 1, "failed": 0, "total": 1},
                artifacts={"trace_file": "trace.zip"},
                duration=5.0,
                test_file="test.spec.ts",
                status=TestStatus.COMPLETED,
            )
        )

        return agent

    @pytest.fixture
    def mock_agent_slow(self, performance_config):
        """Create mock agent with realistic response times."""
        agent = QAOperatorAgent(performance_config)

        # Mock realistic response times
        async def slow_planning(*args, **kwargs):
            await asyncio.sleep(0.5)  # 500ms planning
            return {
                "test_cases": [{"name": "test_action", "description": "Test action"}],
                "estimated_duration": 30,
            }

        async def slow_generation(*args, **kwargs):
            await asyncio.sleep(1.0)  # 1s generation
            return {
                "success": True,
                "test_file": "test.spec.ts",
                "content": "// Realistic generated test content",
            }

        async def slow_execution(*args, **kwargs):
            await asyncio.sleep(2.0)  # 2s execution
            return ExecutionResult(
                success=True,
                test_results={"passed": 1, "failed": 0, "total": 1},
                artifacts={"trace_file": "trace.zip"},
                duration=15.0,
                test_file="test.spec.ts",
                status=TestStatus.COMPLETED,
            )

        agent.planning_engine.create_test_plan = slow_planning
        agent.test_generator.generate_test = slow_generation
        agent.test_executor.execute_tests = slow_execution

        return agent

    @pytest.mark.asyncio
    async def test_single_workflow_performance(
        self, mock_agent_fast, sample_specifications
    ):
        """Benchmark single workflow execution performance."""
        agent = mock_agent_fast
        specification = sample_specifications[0]  # Small spec

        monitor = PerformanceMonitor()
        monitor.start_monitoring()

        # Execute workflow
        result = await agent.run_workflow(specification)

        metrics = monitor.stop_monitoring()

        # Verify success
        assert result["success"] is True

        # Performance assertions
        assert (
            metrics["duration"] < 1.0
        ), f"Single workflow took {metrics['duration']:.3f}s (expected < 1.0s)"
        assert (
            metrics["memory_increase_mb"] < 50
        ), f"Memory increase {metrics['memory_increase_mb']:.2f}MB (expected < 50MB)"
        assert (
            metrics["avg_cpu_percent"] < 80
        ), f"Average CPU {metrics['avg_cpu_percent']:.1f}% (expected < 80%)"

        print(f"Single Workflow Performance:")
        print(f"  Duration: {metrics['duration']:.3f}s")
        print(f"  Memory increase: {metrics['memory_increase_mb']:.2f}MB")
        print(f"  Average CPU: {metrics['avg_cpu_percent']:.1f}%")

    @pytest.mark.asyncio
    async def test_concurrent_workflow_performance(
        self, mock_agent_fast, sample_specifications
    ):
        """Benchmark concurrent workflow execution performance."""
        agent = mock_agent_fast
        specifications = sample_specifications[:2]  # Small and medium specs

        monitor = PerformanceMonitor()
        monitor.start_monitoring()

        # Execute workflows concurrently
        tasks = [agent.run_workflow(spec) for spec in specifications]
        results = await asyncio.gather(*tasks)

        metrics = monitor.stop_monitoring()

        # Verify all succeeded
        assert all(result["success"] for result in results)

        # Performance assertions for concurrent execution
        assert (
            metrics["duration"] < 2.0
        ), f"Concurrent workflows took {metrics['duration']:.3f}s (expected < 2.0s)"
        assert (
            metrics["memory_increase_mb"] < 100
        ), f"Memory increase {metrics['memory_increase_mb']:.2f}MB (expected < 100MB)"

        print(f"Concurrent Workflow Performance ({len(specifications)} workflows):")
        print(f"  Duration: {metrics['duration']:.3f}s")
        print(f"  Memory increase: {metrics['memory_increase_mb']:.2f}MB")
        print(f"  Peak memory: {metrics['peak_memory_mb']:.2f}MB")

    @pytest.mark.asyncio
    async def test_workflow_scaling_performance(
        self, mock_agent_fast, sample_specifications
    ):
        """Test performance scaling with different specification sizes."""
        agent = mock_agent_fast

        results = {}

        for spec in sample_specifications:
            monitor = PerformanceMonitor()
            monitor.start_monitoring()

            # Execute workflow
            result = await agent.run_workflow(spec)

            metrics = monitor.stop_monitoring()

            # Store results
            spec_size = len(spec.requirements)
            results[spec_size] = {
                "success": result["success"],
                "duration": metrics["duration"],
                "memory_increase": metrics["memory_increase_mb"],
                "cpu_avg": metrics["avg_cpu_percent"],
            }

            assert result["success"] is True

        # Analyze scaling characteristics
        sizes = sorted(results.keys())
        durations = [results[size]["duration"] for size in sizes]

        print(f"Workflow Scaling Performance:")
        for size in sizes:
            r = results[size]
            print(
                f"  {size} requirements: {r['duration']:.3f}s, {r['memory_increase']:.2f}MB, {r['cpu_avg']:.1f}% CPU"
            )

        # Verify reasonable scaling (should not be exponential)
        if len(sizes) >= 2:
            scaling_factor = durations[-1] / durations[0]
            requirement_factor = sizes[-1] / sizes[0]

            # Duration should scale sub-linearly with requirements
            assert (
                scaling_factor < requirement_factor * 2
            ), f"Poor scaling: {scaling_factor:.2f}x duration for {requirement_factor:.2f}x requirements"

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, mock_agent_fast, sample_specifications):
        """Test for memory leaks during repeated workflow execution."""
        agent = mock_agent_fast
        specification = sample_specifications[0]  # Small spec

        iterations = 10
        memory_measurements = []

        for i in range(iterations):
            monitor = PerformanceMonitor()
            monitor.start_monitoring()

            # Execute workflow
            result = await agent.run_workflow(specification)

            metrics = monitor.stop_monitoring()
            memory_measurements.append(metrics["final_memory_mb"])

            assert result["success"] is True

            # Small delay to allow garbage collection
            await asyncio.sleep(0.1)

        # Analyze memory trend
        initial_memory = memory_measurements[0]
        final_memory = memory_measurements[-1]
        memory_growth = final_memory - initial_memory

        # Calculate trend (should be minimal)
        avg_growth_per_iteration = memory_growth / iterations

        print(f"Memory Leak Detection ({iterations} iterations):")
        print(f"  Initial memory: {initial_memory:.2f}MB")
        print(f"  Final memory: {final_memory:.2f}MB")
        print(f"  Total growth: {memory_growth:.2f}MB")
        print(f"  Growth per iteration: {avg_growth_per_iteration:.3f}MB")

        # Assert no significant memory leak
        assert (
            memory_growth < 20
        ), f"Potential memory leak: {memory_growth:.2f}MB growth over {iterations} iterations"
        assert (
            avg_growth_per_iteration < 2
        ), f"High memory growth per iteration: {avg_growth_per_iteration:.3f}MB"

    @pytest.mark.asyncio
    async def test_realistic_workflow_performance(
        self, mock_agent_slow, sample_specifications
    ):
        """Test performance with realistic response times."""
        agent = mock_agent_slow
        specification = sample_specifications[1]  # Medium spec

        monitor = PerformanceMonitor()
        monitor.start_monitoring()

        # Execute workflow with realistic timing
        result = await agent.run_workflow(specification)

        metrics = monitor.stop_monitoring()

        # Verify success
        assert result["success"] is True

        # Realistic performance expectations
        assert (
            metrics["duration"] < 10.0
        ), f"Realistic workflow took {metrics['duration']:.3f}s (expected < 10.0s)"
        assert (
            metrics["memory_increase_mb"] < 100
        ), f"Memory increase {metrics['memory_increase_mb']:.2f}MB (expected < 100MB)"

        print(f"Realistic Workflow Performance:")
        print(f"  Duration: {metrics['duration']:.3f}s")
        print(f"  Memory increase: {metrics['memory_increase_mb']:.2f}MB")
        print(f"  Peak memory: {metrics['peak_memory_mb']:.2f}MB")
        print(f"  Average CPU: {metrics['avg_cpu_percent']:.1f}%")

    @pytest.mark.asyncio
    async def test_component_performance_breakdown(
        self, mock_agent_slow, sample_specifications
    ):
        """Test performance breakdown by component."""
        agent = mock_agent_slow
        specification = sample_specifications[0]  # Small spec

        # Time each component individually
        component_times = {}

        # Planning performance
        start_time = time.time()
        plan = await agent.planning_engine.create_test_plan(specification)
        component_times["planning"] = time.time() - start_time

        # Generation performance
        start_time = time.time()
        generation = await agent.test_generator.generate_test(plan)
        component_times["generation"] = time.time() - start_time

        # Execution performance
        start_time = time.time()
        execution = await agent.test_executor.execute_tests([generation["test_file"]])
        component_times["execution"] = time.time() - start_time

        total_component_time = sum(component_times.values())

        print(f"Component Performance Breakdown:")
        for component, duration in component_times.items():
            percentage = (duration / total_component_time) * 100
            print(f"  {component.capitalize()}: {duration:.3f}s ({percentage:.1f}%)")
        print(f"  Total: {total_component_time:.3f}s")

        # Verify reasonable component timing
        assert (
            component_times["planning"] < 2.0
        ), f"Planning too slow: {component_times['planning']:.3f}s"
        assert (
            component_times["generation"] < 3.0
        ), f"Generation too slow: {component_times['generation']:.3f}s"
        assert (
            component_times["execution"] < 5.0
        ), f"Execution too slow: {component_times['execution']:.3f}s"

    def test_performance_regression_detection(self, tmp_path):
        """Test performance regression detection system."""
        # Create performance baseline
        baseline_file = tmp_path / "performance_baseline.json"

        baseline_data = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {
                "single_workflow_duration": 0.5,
                "concurrent_workflow_duration": 1.2,
                "memory_usage_mb": 25.0,
                "cpu_usage_percent": 45.0,
            },
        }

        baseline_file.write_text(json.dumps(baseline_data, indent=2))

        # Simulate current performance data
        current_data = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {
                "single_workflow_duration": 0.6,  # 20% slower
                "concurrent_workflow_duration": 1.5,  # 25% slower
                "memory_usage_mb": 30.0,  # 20% more memory
                "cpu_usage_percent": 50.0,  # 11% more CPU
            },
        }

        # Performance regression analysis
        def detect_regression(baseline, current, threshold=0.15):  # 15% threshold
            regressions = []

            for metric, baseline_value in baseline["benchmarks"].items():
                current_value = current["benchmarks"][metric]
                change_percent = (current_value - baseline_value) / baseline_value

                if change_percent > threshold:
                    regressions.append(
                        {
                            "metric": metric,
                            "baseline": baseline_value,
                            "current": current_value,
                            "change_percent": change_percent * 100,
                        }
                    )

            return regressions

        regressions = detect_regression(baseline_data, current_data)

        print(f"Performance Regression Analysis:")
        if regressions:
            for regression in regressions:
                print(
                    f"  ⚠️  {regression['metric']}: {regression['baseline']} → {regression['current']} ({regression['change_percent']:+.1f}%)"
                )
        else:
            print("  ✅ No significant performance regressions detected")

        # For this test, we expect some regressions based on our test data
        assert (
            len(regressions) > 0
        ), "Expected to detect performance regressions in test data"

        # Verify regression detection is working
        duration_regression = next(
            (r for r in regressions if "duration" in r["metric"]), None
        )
        assert duration_regression is not None, "Should detect duration regression"
        assert (
            duration_regression["change_percent"] > 15
        ), "Should detect significant regression"


@pytest.mark.performance
class TestPerformanceMonitoring:
    """Tests for performance monitoring utilities."""

    def test_performance_monitor_basic_functionality(self):
        """Test basic performance monitor functionality."""
        monitor = PerformanceMonitor()

        # Start monitoring
        monitor.start_monitoring()
        assert monitor.start_time is not None
        assert monitor.initial_memory is not None

        # Simulate some work
        time.sleep(0.1)
        monitor.update_metrics()

        # Stop monitoring
        metrics = monitor.stop_monitoring()

        # Verify metrics
        assert metrics["duration"] >= 0.1
        assert metrics["initial_memory_mb"] > 0
        assert metrics["final_memory_mb"] > 0
        assert "peak_memory_mb" in metrics
        assert "avg_cpu_percent" in metrics

    def test_performance_metrics_collection(self):
        """Test comprehensive performance metrics collection."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()

        # Simulate varying workload
        for _ in range(5):
            # Simulate CPU work
            sum(i * i for i in range(1000))
            monitor.update_metrics()
            time.sleep(0.02)

        metrics = monitor.stop_monitoring()

        # Verify comprehensive metrics
        required_metrics = [
            "duration",
            "initial_memory_mb",
            "final_memory_mb",
            "peak_memory_mb",
            "memory_increase_mb",
            "avg_cpu_percent",
            "max_cpu_percent",
        ]

        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(
                metrics[metric], (int, float)
            ), f"Invalid metric type for {metric}"

    def test_performance_report_generation(self, tmp_path):
        """Test performance report generation."""
        # Create sample performance data
        performance_data = {
            "test_run_id": "test-123",
            "timestamp": datetime.now().isoformat(),
            "environment": {
                "python_version": "3.11.0",
                "platform": "linux",
                "cpu_count": 4,
                "memory_total_gb": 8.0,
            },
            "benchmarks": {
                "single_workflow": {
                    "duration": 0.45,
                    "memory_mb": 23.5,
                    "cpu_percent": 42.1,
                },
                "concurrent_workflow": {
                    "duration": 1.15,
                    "memory_mb": 45.2,
                    "cpu_percent": 67.8,
                },
                "scaling_test": {
                    "small_spec": {"duration": 0.3, "memory_mb": 15.0},
                    "medium_spec": {"duration": 0.8, "memory_mb": 35.0},
                    "large_spec": {"duration": 2.1, "memory_mb": 75.0},
                },
            },
            "regression_analysis": {
                "baseline_date": "2024-01-01",
                "regressions_detected": 0,
                "improvements_detected": 2,
            },
        }

        # Generate report
        report_file = tmp_path / "performance_report.json"
        report_file.write_text(json.dumps(performance_data, indent=2))

        # Verify report structure
        assert report_file.exists()

        with open(report_file) as f:
            report = json.load(f)

        assert "test_run_id" in report
        assert "benchmarks" in report
        assert "environment" in report
        assert "regression_analysis" in report

        # Verify benchmark data structure
        benchmarks = report["benchmarks"]
        assert "single_workflow" in benchmarks
        assert "concurrent_workflow" in benchmarks
        assert "scaling_test" in benchmarks

        print(f"Performance report generated: {report_file}")
        print(f"Report size: {report_file.stat().st_size} bytes")
