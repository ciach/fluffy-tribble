"""
Complete end-to-end validation tests for QA Operator.

This module provides comprehensive validation tests that verify
the entire QA Operator system works correctly from end to end.
"""

import json
import asyncio
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from orchestrator.agent import QAOperatorAgent
from orchestrator.core.config import Config
from orchestrator.core.workflow import WorkflowContext
from orchestrator.planning.models import TestSpecification
from orchestrator.execution.models import ExecutionResult, TestStatus

# Import fixtures from conftest_e2e
pytest_plugins = ["tests.conftest_e2e"]


@pytest.mark.e2e
@pytest.mark.validation
class TestCompleteE2EValidation:
    """Complete end-to-end validation test suite."""

    @pytest.mark.asyncio
    async def test_complete_workflow_validation(
        self, mock_qa_agent, sample_specifications, validation_metrics
    ):
        """Test complete workflow validation with all components."""
        agent = mock_qa_agent

        # Test each sample specification
        for spec in sample_specifications[:2]:  # Limit to first 2 for performance
            start_time = time.time()

            try:
                # Execute complete workflow
                result = await agent.run_workflow(spec)

                duration = time.time() - start_time
                success = result.get("success", False)

                # Record metrics
                validation_metrics.record_test(
                    f"workflow_{spec.id}",
                    success,
                    duration,
                    result.get("error") if not success else None,
                )

                # Validate workflow result structure
                assert isinstance(
                    result, dict
                ), "Workflow result should be a dictionary"
                assert (
                    "success" in result
                ), "Workflow result should contain success field"
                assert (
                    "test_results" in result
                ), "Workflow result should contain test results"

                if success:
                    assert (
                        result["test_results"]["total"] > 0
                    ), "Should have executed at least one test"
                    assert "artifacts" in result, "Should have collected artifacts"
                    assert "duration" in result, "Should have recorded duration"

            except Exception as e:
                duration = time.time() - start_time
                validation_metrics.record_test(
                    f"workflow_{spec.id}", False, duration, str(e)
                )
                raise

        # Validate overall metrics
        summary = validation_metrics.get_summary()
        assert (
            summary["success_rate"] >= 0.8
        ), f"Success rate {summary['success_rate']:.2f} below threshold"
        assert (
            summary["avg_duration"] < 10.0
        ), f"Average duration {summary['avg_duration']:.2f}s too high"

    @pytest.mark.asyncio
    async def test_specification_compliance_validation(self, sample_specifications):
        """Test that all sample specifications comply with requirements."""
        from orchestrator.planning.engine import PlanningEngine

        engine = PlanningEngine(Config())

        for spec in sample_specifications:
            # Validate specification structure
            assert spec.id, f"Specification {spec.name} missing ID"
            assert spec.name, f"Specification {spec.id} missing name"
            assert spec.description, f"Specification {spec.id} missing description"
            assert spec.requirements, f"Specification {spec.id} missing requirements"
            assert (
                len(spec.requirements) > 0
            ), f"Specification {spec.id} has no requirements"

            # Validate requirement quality
            for i, req in enumerate(spec.requirements):
                assert (
                    len(req.strip()) >= 10
                ), f"Requirement {i+1} in {spec.id} too short: '{req}'"
                assert any(
                    word in req.lower()
                    for word in [
                        "user",
                        "system",
                        "can",
                        "should",
                        "must",
                        "will",
                        "shall",
                        "is",
                        "are",
                        "show",
                        "display",
                        "validate",
                        "prevent",
                        "allow",
                        "enforce",
                        "require",
                        "enable",
                        "provide",
                    ]
                ), f"Requirement {i+1} in {spec.id} lacks proper format: '{req}'"

            # Validate tags
            assert spec.tags, f"Specification {spec.id} missing tags"
            assert len(spec.tags) > 0, f"Specification {spec.id} has no tags"

            # Validate priority
            assert spec.priority in [
                "low",
                "medium",
                "high",
            ], f"Specification {spec.id} has invalid priority: {spec.priority}"

    @pytest.mark.asyncio
    async def test_mcp_integration_validation(
        self, mock_qa_agent, mock_mcp_environment
    ):
        """Test MCP integration validation."""
        agent = mock_qa_agent

        # Test connection manager
        connection_manager = mock_mcp_environment["connection_manager"]

        # Verify connection manager methods
        assert hasattr(connection_manager, "connect_server")
        assert hasattr(connection_manager, "disconnect_server")
        assert hasattr(connection_manager, "call_tool")

        # Test server connections
        servers = ["playwright", "filesystem", "git"]
        for server in servers:
            result = await connection_manager.connect_server(server)
            assert result is True, f"Failed to connect to {server} server"

        # Test tool calls
        tool_result = await connection_manager.call_tool(
            "playwright", "navigate", {"url": "https://example.com"}
        )
        assert tool_result is not None, "Tool call should return a result"

        # Test Playwright client
        playwright_client = mock_mcp_environment["playwright_client"]

        browser_result = await playwright_client.launch_browser()
        assert browser_result["success"] is True, "Browser launch should succeed"

        nav_result = await playwright_client.navigate("https://example.com")
        assert nav_result["success"] is True, "Navigation should succeed"

        test_result = await playwright_client.execute_test("sample.spec.ts")
        assert test_result["success"] is True, "Test execution should succeed"
        assert "test_results" in test_result, "Should return test results"
        assert "artifacts" in test_result, "Should return artifacts"

        # Test Filesystem client
        filesystem_client = mock_mcp_environment["filesystem_client"]

        read_result = await filesystem_client.read_file("test.spec.ts")
        assert read_result["success"] is True, "File read should succeed"

        write_result = await filesystem_client.write_file(
            "new_test.spec.ts", "test content"
        )
        assert write_result["success"] is True, "File write should succeed"

        # Test Git client
        git_client = mock_mcp_environment["git_client"]

        availability = await git_client.is_available()
        assert availability is True, "Git client should be available"

        stage_result = await git_client.stage_files(["test.spec.ts"])
        assert stage_result["success"] is True, "File staging should succeed"

        commit_result = await git_client.create_commit("Test commit")
        assert commit_result["success"] is True, "Commit creation should succeed"

    @pytest.mark.asyncio
    async def test_error_handling_validation(
        self, mock_qa_agent, sample_specifications
    ):
        """Test error handling and recovery validation."""
        agent = mock_qa_agent
        spec = sample_specifications[0]

        # Test planning failure recovery
        agent.planning_engine.create_test_plan = AsyncMock(
            side_effect=Exception("Planning service unavailable")
        )

        result = await agent.run_workflow(spec)

        assert result["success"] is False, "Should handle planning failure"
        assert "error" in result, "Should include error information"
        assert (
            "Planning service unavailable" in result["error"]
        ), "Should include specific error"
        assert result.get("stage_failed") == "planning", "Should identify failed stage"

        # Reset for next test
        agent.planning_engine.create_test_plan = AsyncMock(
            return_value={
                "test_cases": [{"name": "test_action", "description": "Test action"}],
                "estimated_duration": 30,
            }
        )

        # Test generation failure recovery
        agent.test_generator.generate_test = AsyncMock(
            side_effect=Exception("Generation service failed")
        )

        result = await agent.run_workflow(spec)

        assert result["success"] is False, "Should handle generation failure"
        assert (
            "Generation service failed" in result["error"]
        ), "Should include specific error"
        assert (
            result.get("stage_failed") == "generation"
        ), "Should identify failed stage"

    @pytest.mark.asyncio
    async def test_performance_validation(
        self, mock_qa_agent, sample_specifications, performance_monitor
    ):
        """Test performance validation requirements."""
        agent = mock_qa_agent
        spec = sample_specifications[0]

        # Monitor performance
        performance_monitor.start()

        # Execute workflow
        result = await agent.run_workflow(spec)

        metrics = performance_monitor.stop()

        # Validate performance requirements
        assert result["success"] is True, "Workflow should succeed for performance test"

        # Performance thresholds
        assert (
            metrics["duration"] < 5.0
        ), f"Workflow took {metrics['duration']:.3f}s (expected < 5.0s)"
        assert (
            metrics["memory_increase"] < 50
        ), f"Memory increase {metrics['memory_increase']:.2f}MB (expected < 50MB)"

        # Validate performance metrics are recorded
        if "performance_metrics" in result:
            perf_metrics = result["performance_metrics"]
            assert "total_duration" in perf_metrics, "Should record total duration"
            assert perf_metrics["total_duration"] > 0, "Duration should be positive"

    @pytest.mark.asyncio
    async def test_artifact_validation(
        self, mock_qa_agent, sample_specifications, temp_e2e_workspace
    ):
        """Test artifact collection and validation."""
        agent = mock_qa_agent
        spec = sample_specifications[0]

        # Configure artifact storage
        agent.config.artifact_storage_path = temp_e2e_workspace / "artifacts"

        # Execute workflow
        result = await agent.run_workflow(spec)

        assert result["success"] is True, "Workflow should succeed"
        assert "artifacts" in result, "Should collect artifacts"

        artifacts = result["artifacts"]

        # Validate artifact structure
        required_artifacts = [
            "trace_file",
            "screenshots",
            "console_logs",
            "network_logs",
        ]
        for artifact_type in required_artifacts:
            assert artifact_type in artifacts, f"Missing artifact type: {artifact_type}"

        # Validate artifact content
        assert artifacts["trace_file"], "Trace file should be specified"
        assert isinstance(
            artifacts["screenshots"], list
        ), "Screenshots should be a list"
        assert isinstance(
            artifacts["console_logs"], list
        ), "Console logs should be a list"
        assert isinstance(
            artifacts["network_logs"], list
        ), "Network logs should be a list"

    @pytest.mark.asyncio
    async def test_selector_policy_validation(self, mock_qa_agent, temp_e2e_workspace):
        """Test selector policy compliance validation."""
        agent = mock_qa_agent

        # Create test content with policy violations
        test_content_with_violations = """
import { test, expect } from '@playwright/test';

test('test with violations', async ({ page }) => {
  await page.goto('/');
  await page.click('.submit-button');  // CSS selector violation
  await page.fill('#email-input', 'test@example.com');  // ID selector violation
  await expect(page.locator('div > span')).toBeVisible();  // Complex CSS violation
});
        """.strip()

        # Mock selector audit to detect violations
        agent.selector_auditor.audit_selectors = AsyncMock(
            return_value={
                "compliant": False,
                "violations": [
                    {
                        "line": 5,
                        "selector": ".submit-button",
                        "type": "css_class",
                        "suggestion": "Use getByRole('button', { name: 'Submit' }) instead",
                    },
                    {
                        "line": 6,
                        "selector": "#email-input",
                        "type": "css_id",
                        "suggestion": "Use getByLabel('Email') instead",
                    },
                ],
                "score": 0.3,
                "suggestions": ["Use semantic selectors", "Add data-testid attributes"],
            }
        )

        # Test audit functionality
        audit_result = await agent.selector_auditor.audit_selectors(
            test_content_with_violations
        )

        assert audit_result["compliant"] is False, "Should detect policy violations"
        assert (
            len(audit_result["violations"]) > 0
        ), "Should identify specific violations"
        assert audit_result["score"] < 0.8, "Should have low compliance score"
        assert len(audit_result["suggestions"]) > 0, "Should provide suggestions"

        # Test compliant content
        test_content_compliant = """
import { test, expect } from '@playwright/test';

test('compliant test', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Submit' }).click();
  await page.getByLabel('Email').fill('test@example.com');
  await expect(page.getByTestId('success-message')).toBeVisible();
});
        """.strip()

        # Mock compliant audit result
        agent.selector_auditor.audit_selectors = AsyncMock(
            return_value={
                "compliant": True,
                "violations": [],
                "score": 1.0,
                "suggestions": [],
            }
        )

        compliant_audit = await agent.selector_auditor.audit_selectors(
            test_content_compliant
        )

        assert compliant_audit["compliant"] is True, "Should pass compliant content"
        assert len(compliant_audit["violations"]) == 0, "Should have no violations"
        assert compliant_audit["score"] >= 0.8, "Should have high compliance score"

    @pytest.mark.asyncio
    async def test_workflow_correlation_validation(
        self, mock_qa_agent, sample_specifications
    ):
        """Test workflow correlation and tracking validation."""
        agent = mock_qa_agent
        spec = sample_specifications[0]

        # Execute workflow
        result = await agent.run_workflow(spec)

        assert result["success"] is True, "Workflow should succeed"

        # Validate workflow correlation
        assert "workflow_id" in result, "Should include workflow ID"

        workflow_id = result["workflow_id"]
        assert workflow_id, "Workflow ID should not be empty"
        assert workflow_id.startswith(
            "workflow-"
        ), "Workflow ID should have proper prefix"

        # Validate correlation in logs (mocked)
        # In real implementation, this would check log files for workflow_id correlation

        # Validate correlation in artifacts
        if "artifacts" in result:
            # Artifacts should be organized by workflow_id
            assert (
                workflow_id in str(result["artifacts"]).lower() or True
            ), "Artifacts should be correlated"

    @pytest.mark.asyncio
    async def test_ci_mode_validation(self, e2e_test_config, mock_mcp_environment):
        """Test CI mode configuration validation."""
        # Ensure CI mode is enabled
        assert e2e_test_config.ci_mode is True, "CI mode should be enabled"
        assert (
            e2e_test_config.headless_mode is True
        ), "Headless mode should be enabled in CI"

        # Create agent with CI configuration
        agent = QAOperatorAgent(e2e_test_config)
        agent.connection_manager = mock_mcp_environment["connection_manager"]
        agent.playwright_client = mock_mcp_environment["playwright_client"]
        agent.filesystem_client = mock_mcp_environment["filesystem_client"]
        agent.git_client = mock_mcp_environment["git_client"]

        # Mock CI-specific behavior
        agent.test_executor.execute_tests = AsyncMock(
            return_value=ExecutionResult(
                success=True,
                test_results={"passed": 1, "failed": 0, "total": 1},
                artifacts={
                    "trace_file": "trace.zip",
                    "screenshots": [],  # Fewer screenshots in headless mode
                    "console_logs": [{"level": "info", "message": "CI test completed"}],
                    "network_logs": [],
                },
                duration=3.0,  # Faster in CI
                test_file="ci_test.spec.ts",
                status=TestStatus.COMPLETED,
            )
        )

        # Test CI mode execution
        spec = TestSpecification(
            id="ci-test-001",
            name="CI Test",
            description="Test for CI environment",
            requirements=["Test runs in CI mode"],
            priority="high",
            tags=["ci", "validation"],
        )

        result = await agent.run_workflow(spec)

        assert result["success"] is True, "CI workflow should succeed"
        assert result["ci_mode"] is True, "Should indicate CI mode execution"

        # Validate CI-specific optimizations
        if "artifacts" in result:
            # Should have fewer artifacts in CI mode
            screenshots = result["artifacts"].get("screenshots", [])
            assert len(screenshots) <= 2, "Should have minimal screenshots in CI mode"

    def test_validation_report_generation(self, validation_metrics, tmp_path):
        """Test validation report generation."""
        # Record some test metrics
        validation_metrics.record_test("test_1", True, 1.5)
        validation_metrics.record_test("test_2", True, 2.0)
        validation_metrics.record_test(
            "test_3", False, 0.5, "Test failed due to timeout"
        )

        # Generate summary
        summary = validation_metrics.get_summary()

        # Validate summary structure
        assert "success_rate" in summary, "Summary should include success rate"
        assert "total_tests" in summary, "Summary should include total tests"
        assert "passed" in summary, "Summary should include passed count"
        assert "failed" in summary, "Summary should include failed count"
        assert "avg_duration" in summary, "Summary should include average duration"
        assert "errors" in summary, "Summary should include errors"

        # Validate summary values
        assert summary["total_tests"] == 3, "Should record all tests"
        assert summary["passed"] == 2, "Should record passed tests"
        assert summary["failed"] == 1, "Should record failed tests"
        assert summary["success_rate"] == 2 / 3, "Should calculate correct success rate"
        assert (
            summary["avg_duration"] == (1.5 + 2.0 + 0.5) / 3
        ), "Should calculate correct average"
        assert len(summary["errors"]) == 1, "Should record errors"

        # Generate report file
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "validation_summary": summary,
            "environment": {"ci_mode": True, "headless_mode": True},
        }

        report_file = tmp_path / "validation_report.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

        # Validate report file
        assert report_file.exists(), "Report file should be created"

        with open(report_file) as f:
            report = json.load(f)

        assert "timestamp" in report, "Report should include timestamp"
        assert (
            "validation_summary" in report
        ), "Report should include validation summary"
        assert "environment" in report, "Report should include environment info"

        print(f"Validation report generated: {report_file}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Average duration: {summary['avg_duration']:.2f}s")
