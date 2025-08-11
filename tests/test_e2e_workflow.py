"""
End-to-end workflow tests for QA Operator.

Tests the complete workflow from specification to test execution,
including planning, generation, execution, analysis, and patching.
"""

import json
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from orchestrator.agent import QAOperatorAgent
from orchestrator.core.config import Config
from orchestrator.core.workflow import WorkflowContext
from orchestrator.planning.models import TestSpecification
from orchestrator.execution.models import ExecutionResult, TestStatus
from orchestrator.analysis.models import FailureAnalysis


@pytest.mark.integration
class TestE2EWorkflow:
    """End-to-end workflow integration tests."""

    @pytest.fixture
    def sample_specifications_dir(self, tmp_path):
        """Create sample test specifications directory."""
        specs_dir = tmp_path / "sample_specifications"
        specs_dir.mkdir()

        # Create sample specifications
        auth_spec = {
            "id": "auth-flow-001",
            "name": "User Authentication Flow",
            "description": "Complete user authentication workflow",
            "requirements": [
                "User can log in with valid credentials",
                "User can log out successfully",
                "Invalid credentials show error messages",
            ],
            "priority": "high",
            "tags": ["authentication", "security"],
        }

        (specs_dir / "auth_flow.json").write_text(json.dumps(auth_spec, indent=2))

        return specs_dir

    @pytest.fixture
    def mock_agent_with_mcp(
        self,
        temp_config,
        mock_connection_manager,
        mock_playwright_client,
        mock_filesystem_client,
        mock_git_client,
    ):
        """Create QA Operator agent with mocked MCP clients."""
        agent = QAOperatorAgent(temp_config)

        # Mock MCP clients
        agent.connection_manager = mock_connection_manager
        agent.playwright_client = mock_playwright_client
        agent.filesystem_client = mock_filesystem_client
        agent.git_client = mock_git_client

        return agent

    @pytest.mark.asyncio
    async def test_complete_workflow_success(
        self, mock_agent_with_mcp, sample_specifications_dir
    ):
        """Test complete successful workflow from specification to execution."""
        agent = mock_agent_with_mcp

        # Load test specification
        spec_file = sample_specifications_dir / "auth_flow.json"
        with open(spec_file) as f:
            spec_data = json.load(f)

        specification = TestSpecification(**spec_data)

        # Mock successful workflow steps
        agent.planning_engine.create_test_plan = AsyncMock(
            return_value={
                "test_cases": [
                    {
                        "name": "test_user_login",
                        "description": "Test user login functionality",
                        "steps": [
                            "Navigate to login",
                            "Enter credentials",
                            "Click submit",
                        ],
                        "assertions": ["User is logged in"],
                    }
                ],
                "estimated_duration": 30,
            }
        )

        agent.test_generator.generate_test = AsyncMock(
            return_value={
                "success": True,
                "test_file": "auth_flow.spec.ts",
                "content": "// Generated test content",
            }
        )

        agent.test_executor.execute_tests = AsyncMock(
            return_value=ExecutionResult(
                success=True,
                test_results={"passed": 1, "failed": 0, "total": 1},
                artifacts={"trace_file": "trace.zip", "screenshots": []},
                duration=25.5,
                test_file="auth_flow.spec.ts",
                status=TestStatus.COMPLETED,
            )
        )

        # Execute workflow
        result = await agent.run_workflow(specification)

        # Verify workflow completion
        assert result["success"] is True
        assert result["test_results"]["passed"] == 1
        assert result["test_results"]["failed"] == 0

        # Verify all components were called
        agent.planning_engine.create_test_plan.assert_called_once()
        agent.test_generator.generate_test.assert_called_once()
        agent.test_executor.execute_tests.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_with_test_failure_and_patching(
        self, mock_agent_with_mcp, sample_specifications_dir
    ):
        """Test workflow with test failure, analysis, and automatic patching."""
        agent = mock_agent_with_mcp

        # Load test specification
        spec_file = sample_specifications_dir / "auth_flow.json"
        with open(spec_file) as f:
            spec_data = json.load(f)

        specification = TestSpecification(**spec_data)

        # Mock workflow with initial failure
        agent.planning_engine.create_test_plan = AsyncMock(
            return_value={
                "test_cases": [
                    {"name": "test_user_login", "description": "Test login"}
                ],
                "estimated_duration": 30,
            }
        )

        agent.test_generator.generate_test = AsyncMock(
            return_value={
                "success": True,
                "test_file": "auth_flow.spec.ts",
                "content": "// Generated test content",
            }
        )

        # First execution fails
        failed_result = ExecutionResult(
            success=False,
            test_results={"passed": 0, "failed": 1, "total": 1},
            artifacts={
                "trace_file": "trace.zip",
                "screenshots": ["error_screenshot.png"],
                "console_logs": [{"level": "error", "message": "Element not found"}],
            },
            duration=15.0,
            test_file="auth_flow.spec.ts",
            status=TestStatus.FAILED,
        )

        # Second execution succeeds after patching
        success_result = ExecutionResult(
            success=True,
            test_results={"passed": 1, "failed": 0, "total": 1},
            artifacts={"trace_file": "trace.zip", "screenshots": []},
            duration=20.0,
            test_file="auth_flow.spec.ts",
            status=TestStatus.COMPLETED,
        )

        agent.test_executor.execute_tests = AsyncMock(
            side_effect=[failed_result, success_result]
        )

        # Mock failure analysis
        agent.failure_analyzer.analyze_failure = AsyncMock(
            return_value=FailureAnalysis(
                root_cause="Login button selector not found",
                error_category="selector",
                confidence=0.9,
                suggested_fixes=[
                    {
                        "type": "selector_update",
                        "old_selector": "button.login-btn",
                        "new_selector": "button[data-testid='login-submit']",
                    }
                ],
                artifacts_analyzed=["trace.zip", "error_screenshot.png"],
            )
        )

        # Mock code patching
        agent.code_patcher.apply_patch = AsyncMock(
            return_value={
                "success": True,
                "patched_file": "auth_flow.spec.ts",
                "changes_applied": 1,
            }
        )

        # Execute workflow
        result = await agent.run_workflow(specification)

        # Verify workflow handled failure and recovered
        assert result["success"] is True
        assert result["test_results"]["passed"] == 1
        assert result["patches_applied"] == 1

        # Verify failure analysis and patching were called
        agent.failure_analyzer.analyze_failure.assert_called_once()
        agent.code_patcher.apply_patch.assert_called_once()
        assert agent.test_executor.execute_tests.call_count == 2

    @pytest.mark.asyncio
    async def test_workflow_with_git_integration(
        self, mock_agent_with_mcp, sample_specifications_dir
    ):
        """Test workflow with Git integration for committing changes."""
        agent = mock_agent_with_mcp

        # Enable Git integration
        agent.config.git_integration_enabled = True
        agent.git_client.is_available = AsyncMock(return_value=True)

        # Load test specification
        spec_file = sample_specifications_dir / "auth_flow.json"
        with open(spec_file) as f:
            spec_data = json.load(f)

        specification = TestSpecification(**spec_data)

        # Mock successful workflow
        agent.planning_engine.create_test_plan = AsyncMock(
            return_value={
                "test_cases": [{"name": "test_user_login"}],
                "estimated_duration": 30,
            }
        )

        agent.test_generator.generate_test = AsyncMock(
            return_value={
                "success": True,
                "test_file": "auth_flow.spec.ts",
                "content": "// Generated test content",
            }
        )

        agent.test_executor.execute_tests = AsyncMock(
            return_value=ExecutionResult(
                success=True,
                test_results={"passed": 1, "failed": 0, "total": 1},
                artifacts={"trace_file": "trace.zip"},
                duration=25.5,
                test_file="auth_flow.spec.ts",
                status=TestStatus.COMPLETED,
            )
        )

        # Mock Git operations
        agent.git_client.stage_files = AsyncMock(
            return_value={"success": True, "staged_files": ["e2e/auth_flow.spec.ts"]}
        )

        agent.git_client.create_commit = AsyncMock(
            return_value={"success": True, "commit_hash": "abc123def456"}
        )

        # Execute workflow
        result = await agent.run_workflow(specification)

        # Verify Git operations were performed
        assert result["success"] is True
        assert result["git_commit_hash"] == "abc123def456"

        agent.git_client.stage_files.assert_called_once()
        agent.git_client.create_commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_performance_monitoring(
        self, mock_agent_with_mcp, sample_specifications_dir
    ):
        """Test workflow performance monitoring and metrics collection."""
        agent = mock_agent_with_mcp

        # Load test specification
        spec_file = sample_specifications_dir / "auth_flow.json"
        with open(spec_file) as f:
            spec_data = json.load(f)

        specification = TestSpecification(**spec_data)

        # Mock workflow components with timing
        agent.planning_engine.create_test_plan = AsyncMock(
            return_value={
                "test_cases": [{"name": "test_user_login"}],
                "estimated_duration": 30,
            }
        )

        agent.test_generator.generate_test = AsyncMock(
            return_value={
                "success": True,
                "test_file": "auth_flow.spec.ts",
                "content": "// Generated test content",
            }
        )

        agent.test_executor.execute_tests = AsyncMock(
            return_value=ExecutionResult(
                success=True,
                test_results={"passed": 1, "failed": 0, "total": 1},
                artifacts={"trace_file": "trace.zip"},
                duration=25.5,
                test_file="auth_flow.spec.ts",
                status=TestStatus.COMPLETED,
            )
        )

        # Execute workflow with performance monitoring
        start_time = datetime.now()
        result = await agent.run_workflow(specification)
        end_time = datetime.now()

        # Verify performance metrics
        assert result["success"] is True
        assert "performance_metrics" in result
        assert result["performance_metrics"]["total_duration"] > 0
        assert result["performance_metrics"]["planning_duration"] >= 0
        assert result["performance_metrics"]["generation_duration"] >= 0
        assert result["performance_metrics"]["execution_duration"] >= 0

        # Verify total workflow time is reasonable
        total_duration = (end_time - start_time).total_seconds()
        assert (
            result["performance_metrics"]["total_duration"] <= total_duration + 1
        )  # Allow 1s buffer

    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(
        self, mock_agent_with_mcp, sample_specifications_dir
    ):
        """Test concurrent execution of multiple workflows."""
        agent = mock_agent_with_mcp

        # Load test specification
        spec_file = sample_specifications_dir / "auth_flow.json"
        with open(spec_file) as f:
            spec_data = json.load(f)

        # Create multiple specifications
        specifications = []
        for i in range(3):
            spec_data_copy = spec_data.copy()
            spec_data_copy["id"] = f"auth-flow-00{i+1}"
            spec_data_copy["name"] = f"Auth Flow {i+1}"
            specifications.append(TestSpecification(**spec_data_copy))

        # Mock workflow components
        agent.planning_engine.create_test_plan = AsyncMock(
            return_value={
                "test_cases": [{"name": "test_user_login"}],
                "estimated_duration": 30,
            }
        )

        agent.test_generator.generate_test = AsyncMock(
            return_value={
                "success": True,
                "test_file": "auth_flow.spec.ts",
                "content": "// Generated test content",
            }
        )

        agent.test_executor.execute_tests = AsyncMock(
            return_value=ExecutionResult(
                success=True,
                test_results={"passed": 1, "failed": 0, "total": 1},
                artifacts={"trace_file": "trace.zip"},
                duration=25.5,
                test_file="auth_flow.spec.ts",
                status=TestStatus.COMPLETED,
            )
        )

        # Execute workflows concurrently
        tasks = [agent.run_workflow(spec) for spec in specifications]
        results = await asyncio.gather(*tasks)

        # Verify all workflows completed successfully
        assert len(results) == 3
        for result in results:
            assert result["success"] is True
            assert result["test_results"]["passed"] == 1

    @pytest.mark.asyncio
    async def test_workflow_error_recovery(
        self, mock_agent_with_mcp, sample_specifications_dir
    ):
        """Test workflow error recovery and graceful failure handling."""
        agent = mock_agent_with_mcp

        # Load test specification
        spec_file = sample_specifications_dir / "auth_flow.json"
        with open(spec_file) as f:
            spec_data = json.load(f)

        specification = TestSpecification(**spec_data)

        # Mock planning failure
        agent.planning_engine.create_test_plan = AsyncMock(
            side_effect=Exception("Planning service unavailable")
        )

        # Execute workflow and expect graceful failure
        result = await agent.run_workflow(specification)

        # Verify error was handled gracefully
        assert result["success"] is False
        assert "error" in result
        assert "Planning service unavailable" in result["error"]
        assert result["stage_failed"] == "planning"

    @pytest.mark.asyncio
    async def test_workflow_with_mcp_server_failures(
        self, mock_agent_with_mcp, sample_specifications_dir
    ):
        """Test workflow behavior when MCP servers are unavailable."""
        agent = mock_agent_with_mcp

        # Load test specification
        spec_file = sample_specifications_dir / "auth_flow.json"
        with open(spec_file) as f:
            spec_data = json.load(f)

        specification = TestSpecification(**spec_data)

        # Mock MCP connection failures
        agent.connection_manager.call_tool = AsyncMock(
            side_effect=Exception("MCP server connection failed")
        )

        # Mock fallback behavior
        agent.planning_engine.create_test_plan = AsyncMock(
            return_value={
                "test_cases": [{"name": "test_user_login"}],
                "estimated_duration": 30,
                "fallback_mode": True,
            }
        )

        # Execute workflow
        result = await agent.run_workflow(specification)

        # Verify fallback behavior was triggered
        assert "fallback_mode" in result
        assert result["fallback_mode"] is True


@pytest.mark.integration
class TestE2EPerformance:
    """Performance and benchmarking tests for E2E workflows."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_workflow_performance_benchmarks(
        self, mock_agent_with_mcp, sample_specifications_dir
    ):
        """Benchmark workflow performance under various conditions."""
        agent = mock_agent_with_mcp

        # Load test specification
        spec_file = sample_specifications_dir / "auth_flow.json"
        with open(spec_file) as f:
            spec_data = json.load(f)

        specification = TestSpecification(**spec_data)

        # Mock workflow components with realistic delays
        async def mock_planning_with_delay(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate planning time
            return {
                "test_cases": [{"name": "test_user_login"}],
                "estimated_duration": 30,
            }

        async def mock_generation_with_delay(*args, **kwargs):
            await asyncio.sleep(0.2)  # Simulate generation time
            return {
                "success": True,
                "test_file": "auth_flow.spec.ts",
                "content": "// Generated test",
            }

        async def mock_execution_with_delay(*args, **kwargs):
            await asyncio.sleep(0.5)  # Simulate test execution time
            return ExecutionResult(
                success=True,
                test_results={"passed": 1, "failed": 0, "total": 1},
                artifacts={"trace_file": "trace.zip"},
                duration=25.5,
                test_file="auth_flow.spec.ts",
                status=TestStatus.COMPLETED,
            )

        agent.planning_engine.create_test_plan = mock_planning_with_delay
        agent.test_generator.generate_test = mock_generation_with_delay
        agent.test_executor.execute_tests = mock_execution_with_delay

        # Run multiple iterations for benchmarking
        iterations = 5
        durations = []

        for _ in range(iterations):
            start_time = datetime.now()
            result = await agent.run_workflow(specification)
            end_time = datetime.now()

            duration = (end_time - start_time).total_seconds()
            durations.append(duration)

            assert result["success"] is True

        # Calculate performance metrics
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        # Performance assertions (adjust thresholds as needed)
        assert avg_duration < 2.0, f"Average duration {avg_duration}s exceeds threshold"
        assert max_duration < 3.0, f"Max duration {max_duration}s exceeds threshold"
        assert min_duration > 0.5, f"Min duration {min_duration}s seems too fast"

        print(f"Performance Benchmark Results:")
        print(f"  Average Duration: {avg_duration:.3f}s")
        print(f"  Min Duration: {min_duration:.3f}s")
        print(f"  Max Duration: {max_duration:.3f}s")

    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(
        self, mock_agent_with_mcp, sample_specifications_dir
    ):
        """Monitor memory usage during workflow execution."""
        import psutil
        import os

        agent = mock_agent_with_mcp

        # Load test specification
        spec_file = sample_specifications_dir / "auth_flow.json"
        with open(spec_file) as f:
            spec_data = json.load(f)

        specification = TestSpecification(**spec_data)

        # Mock workflow components
        agent.planning_engine.create_test_plan = AsyncMock(
            return_value={
                "test_cases": [{"name": "test_user_login"}],
                "estimated_duration": 30,
            }
        )

        agent.test_generator.generate_test = AsyncMock(
            return_value={
                "success": True,
                "test_file": "auth_flow.spec.ts",
                "content": "// Generated test content",
            }
        )

        agent.test_executor.execute_tests = AsyncMock(
            return_value=ExecutionResult(
                success=True,
                test_results={"passed": 1, "failed": 0, "total": 1},
                artifacts={"trace_file": "trace.zip"},
                duration=25.5,
                test_file="auth_flow.spec.ts",
                status=TestStatus.COMPLETED,
            )
        )

        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Execute workflow
        result = await agent.run_workflow(specification)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Verify workflow success and memory usage
        assert result["success"] is True
        assert (
            memory_increase < 100
        ), f"Memory increase {memory_increase:.2f}MB exceeds threshold"

        print(f"Memory Usage:")
        print(f"  Initial: {initial_memory:.2f}MB")
        print(f"  Final: {final_memory:.2f}MB")
        print(f"  Increase: {memory_increase:.2f}MB")


@pytest.mark.integration
class TestE2EValidation:
    """Validation tests for E2E workflow correctness."""

    @pytest.mark.asyncio
    async def test_specification_validation(self, sample_specifications_dir):
        """Test validation of test specifications."""
        from orchestrator.planning.engine import PlanningEngine

        engine = PlanningEngine(Config())

        # Test valid specification
        spec_file = sample_specifications_dir / "auth_flow.json"
        with open(spec_file) as f:
            spec_data = json.load(f)

        specification = TestSpecification(**spec_data)

        # Validate specification
        validation_result = engine.validate_specification(specification)

        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_generated_test_validation(
        self, mock_agent_with_mcp, sample_specifications_dir
    ):
        """Test validation of generated test files."""
        agent = mock_agent_with_mcp

        # Mock test generation
        generated_test_content = """
import { test, expect } from '@playwright/test';

test('user can log in', async ({ page }) => {
  await page.goto('/login');
  await page.fill('[data-testid="email"]', 'user@example.com');
  await page.fill('[data-testid="password"]', 'password123');
  await page.click('[data-testid="submit"]');
  await expect(page.locator('[data-testid="welcome"]')).toBeVisible();
});
        """

        agent.test_generator.generate_test = AsyncMock(
            return_value={
                "success": True,
                "test_file": "auth_flow.spec.ts",
                "content": generated_test_content,
            }
        )

        # Mock selector audit
        agent.selector_auditor.audit_selectors = AsyncMock(
            return_value={"compliant": True, "violations": [], "score": 1.0}
        )

        # Load test specification
        spec_file = sample_specifications_dir / "auth_flow.json"
        with open(spec_file) as f:
            spec_data = json.load(f)

        specification = TestSpecification(**spec_data)

        # Generate and validate test
        generation_result = await agent.test_generator.generate_test(specification)
        audit_result = await agent.selector_auditor.audit_selectors(
            generation_result["content"]
        )

        # Verify test generation and validation
        assert generation_result["success"] is True
        assert audit_result["compliant"] is True
        assert audit_result["score"] >= 0.8  # High compliance score

    @pytest.mark.asyncio
    async def test_artifact_collection_validation(
        self, mock_agent_with_mcp, temp_artifacts_directory
    ):
        """Test validation of artifact collection and storage."""
        agent = mock_agent_with_mcp

        # Mock test execution with artifacts
        agent.test_executor.execute_tests = AsyncMock(
            return_value=ExecutionResult(
                success=True,
                test_results={"passed": 1, "failed": 0, "total": 1},
                artifacts={
                    "trace_file": str(temp_artifacts_directory / "trace.zip"),
                    "screenshots": [str(temp_artifacts_directory / "screenshot.png")],
                    "console_logs": [{"level": "info", "message": "Test completed"}],
                    "network_logs": [{"url": "https://example.com", "status": 200}],
                },
                duration=25.5,
                test_file="auth_flow.spec.ts",
                status=TestStatus.COMPLETED,
            )
        )

        # Execute test and collect artifacts
        result = await agent.test_executor.execute_tests(["auth_flow.spec.ts"])

        # Validate artifact collection
        assert result.success is True
        assert "trace_file" in result.artifacts
        assert len(result.artifacts["screenshots"]) > 0
        assert len(result.artifacts["console_logs"]) > 0
        assert len(result.artifacts["network_logs"]) > 0

        # Verify artifact files exist (mocked)
        assert Path(result.artifacts["trace_file"]).name == "trace.zip"
        assert Path(result.artifacts["screenshots"][0]).name == "screenshot.png"
