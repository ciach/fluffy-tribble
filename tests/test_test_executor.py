"""
Unit tests for test executor with environment detection.

Tests test execution, environment detection, and artifact collection functionality.
"""

import asyncio
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import pytest

from orchestrator.core.config import Config
from orchestrator.execution.executor import TestExecutor
from orchestrator.execution.models import (
    ExecutionConfig,
    ExecutionResult,
    TestStatus,
    ExecutionMode,
    TestExecutionError,
)
from orchestrator.mcp.playwright_client import (
    PlaywrightMCPClient,
    BrowserMode,
    TestResult,
)


@pytest.fixture
def temp_config():
    """Create a temporary configuration for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        config = Config()
        config.artifacts_dir = temp_path / "artifacts"
        config.logs_dir = temp_path / "logs"
        config.e2e_dir = temp_path / "e2e"
        config.artifact_retention_days = 7
        config.ci_mode = False
        config.headless_mode = False

        # Create directories
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        config.logs_dir.mkdir(parents=True, exist_ok=True)
        config.e2e_dir.mkdir(parents=True, exist_ok=True)

        yield config


@pytest.fixture
def mock_playwright_client():
    """Create a mock Playwright MCP client."""
    client = Mock(spec=PlaywrightMCPClient)
    client.is_available.return_value = True

    # Mock execute_test method
    async def mock_execute_test(*args, **kwargs):
        from orchestrator.mcp.playwright_client import TestArtifacts

        return TestResult(
            test_name="sample_test",
            status="passed",
            duration=2.5,
            artifacts=TestArtifacts(
                trace_file="trace.zip",
                screenshots=["screenshot1.png", "screenshot2.png"],
                video_file="test.webm",
                console_logs=[{"level": "info", "message": "Test log"}],
                network_logs=[{"url": "https://example.com", "status": 200}],
            ),
            exit_code=0,
        )

    client.execute_test = AsyncMock(side_effect=mock_execute_test)

    return client


@pytest.fixture
def mock_artifact_manager():
    """Create a mock artifact manager."""
    manager = Mock()

    async def mock_prepare_dir(test_name):
        return Path(f"/tmp/artifacts/{test_name}")

    manager.prepare_test_artifacts_dir = AsyncMock(side_effect=mock_prepare_dir)

    return manager


@pytest.fixture
def test_executor(temp_config, mock_playwright_client, mock_artifact_manager):
    """Create a test executor instance."""
    return TestExecutor(
        config=temp_config,
        playwright_client=mock_playwright_client,
        artifact_manager=mock_artifact_manager,
        workflow_id="test_workflow_123",
    )


@pytest.fixture
def sample_test_file(temp_config):
    """Create a sample test file."""
    test_file = temp_config.e2e_dir / "sample.test.ts"
    test_file.write_text(
        """
import { test, expect } from '@playwright/test';

test('sample test', async ({ page }) => {
  await page.goto('https://example.com');
  await expect(page).toHaveTitle(/Example/);
});
"""
    )
    return test_file


class TestTestExecutor:
    """Test cases for TestExecutor."""

    def test_initialization(self, test_executor, temp_config, mock_playwright_client):
        """Test test executor initialization."""
        assert test_executor.config == temp_config
        assert test_executor.playwright_client == mock_playwright_client
        assert test_executor.workflow_id == "test_workflow_123"
        assert test_executor.logger is not None

    def test_detect_execution_mode_development(self, test_executor):
        """Test execution mode detection in development environment."""
        # Default development environment (no CI, no overrides)
        mode = test_executor.detect_execution_mode()
        assert mode == ExecutionMode.HEADED

    def test_detect_execution_mode_ci_environment(self, test_executor):
        """Test execution mode detection in CI environment."""
        # Simulate CI environment
        test_executor.config.ci_mode = True

        mode = test_executor.detect_execution_mode()
        assert mode == ExecutionMode.HEADLESS

    def test_detect_execution_mode_force_headless_parameter(self, test_executor):
        """Test execution mode with force_headless parameter."""
        mode = test_executor.detect_execution_mode(force_headless=True)
        assert mode == ExecutionMode.HEADLESS

    def test_detect_execution_mode_environment_variable(self, test_executor):
        """Test execution mode with environment variable override."""
        with patch.dict(os.environ, {"QA_OPERATOR_HEADLESS": "true"}):
            mode = test_executor.detect_execution_mode()
            assert mode == ExecutionMode.HEADLESS

    def test_parse_command_line_flags_headless(self, test_executor):
        """Test parsing --headless command-line flag."""
        args = ["--headless"]
        flags = test_executor.parse_command_line_flags(args)

        assert flags["force_headless"] is True
        assert flags["timeout"] is None
        assert flags["retries"] is None

    def test_parse_command_line_flags_timeout(self, test_executor):
        """Test parsing --timeout command-line flag."""
        args = ["--timeout", "60000"]
        flags = test_executor.parse_command_line_flags(args)

        assert flags["timeout"] == 60000
        assert flags["force_headless"] is False

    def test_parse_command_line_flags_multiple(self, test_executor):
        """Test parsing multiple command-line flags."""
        args = ["--headless", "--timeout", "45000", "--retries", "2", "--workers", "4"]
        flags = test_executor.parse_command_line_flags(args)

        assert flags["force_headless"] is True
        assert flags["timeout"] == 45000
        assert flags["retries"] == 2
        assert flags["workers"] == 4

    def test_parse_command_line_flags_invalid_values(self, test_executor):
        """Test parsing command-line flags with invalid values."""
        args = ["--timeout", "invalid", "--retries", "not_a_number"]
        flags = test_executor.parse_command_line_flags(args)

        # Invalid values should be ignored
        assert flags["timeout"] is None
        assert flags["retries"] is None

    async def test_execute_test_basic(self, test_executor, sample_test_file):
        """Test basic test execution."""
        result = await test_executor.execute_test(sample_test_file)

        assert isinstance(result, ExecutionResult)
        assert result.test_name == "sample.test"
        assert result.test_file == str(sample_test_file)
        assert result.workflow_id == "test_workflow_123"
        assert result.status == TestStatus.PASSED
        assert result.duration > 0
        assert result.exit_code == 0
        assert len(result.artifacts) > 0

    async def test_execute_test_with_config(self, test_executor, sample_test_file):
        """Test test execution with custom configuration."""
        config = ExecutionConfig(
            test_file=str(sample_test_file),
            force_headless=True,
            timeout=60000,
            retries=1,
            collect_traces=True,
            collect_videos=False,
        )

        result = await test_executor.execute_test(sample_test_file, config)

        assert result.execution_config.force_headless is True
        assert result.execution_config.timeout == 60000
        assert result.execution_config.retries == 1
        assert result.execution_config.collect_traces is True
        assert result.execution_config.collect_videos is False

    async def test_execute_test_with_command_line_args(
        self, test_executor, sample_test_file
    ):
        """Test test execution with command-line argument overrides."""
        command_line_args = ["--headless", "--timeout", "90000"]

        result = await test_executor.execute_test(
            sample_test_file, command_line_args=command_line_args
        )

        assert result.execution_config.force_headless is True
        assert result.execution_config.timeout == 90000

    async def test_execute_test_playwright_unavailable(
        self, test_executor, sample_test_file
    ):
        """Test test execution when Playwright MCP is unavailable."""
        # Mock Playwright client as unavailable
        test_executor.playwright_client.is_available.return_value = False

        result = await test_executor.execute_test(sample_test_file)

        assert result.status == TestStatus.ERROR
        assert result.exit_code == -1
        assert "not available" in result.error_message

    async def test_execute_test_playwright_failure(
        self, test_executor, sample_test_file
    ):
        """Test test execution when Playwright test fails."""

        # Mock failed test result
        async def mock_failed_execute_test(*args, **kwargs):
            from orchestrator.mcp.playwright_client import TestArtifacts

            return TestResult(
                test_name="sample_test",
                status="failed",
                duration=1.5,
                artifacts=TestArtifacts(
                    trace_file="trace.zip",
                    screenshots=["failure_screenshot.png"],
                ),
                exit_code=1,
                error_info={
                    "error": "Element not found",
                    "type": "TimeoutError",
                    "stack_trace": "TimeoutError: Element not found\n  at ...",
                },
            )

        test_executor.playwright_client.execute_test = AsyncMock(
            side_effect=mock_failed_execute_test
        )

        result = await test_executor.execute_test(sample_test_file)

        assert result.status == TestStatus.FAILED
        assert result.exit_code == 1
        assert result.error_message == "Element not found"
        assert result.error_type == "TimeoutError"
        assert result.stack_trace is not None

    async def test_execute_multiple_tests(self, test_executor, temp_config):
        """Test execution of multiple test files."""
        # Create multiple test files
        test_files = []
        for i in range(3):
            test_file = temp_config.e2e_dir / f"test_{i}.test.ts"
            test_file.write_text(f"// Test {i}")
            test_files.append(test_file)

        results = await test_executor.execute_multiple_tests(test_files)

        assert len(results) == 3
        assert all(isinstance(r, ExecutionResult) for r in results)
        assert all(r.status == TestStatus.PASSED for r in results)
        assert all(r.workflow_id == "test_workflow_123" for r in results)

    def test_parse_test_results(self, test_executor, sample_test_file):
        """Test parsing test results into structured output."""
        # Create a sample execution result
        result = ExecutionResult(
            test_name="sample_test",
            test_file=str(sample_test_file),
            workflow_id="test_workflow_123",
            status=TestStatus.PASSED,
            duration=2.5,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            exit_code=0,
            execution_config=ExecutionConfig(test_file=str(sample_test_file)),
            environment={"ci_mode": False, "headless_mode": False},
        )

        parsed = test_executor.parse_test_results(result)

        assert "test" in parsed
        assert "execution" in parsed
        assert "artifacts" in parsed
        assert "environment" in parsed

        assert parsed["test"]["name"] == "sample_test"
        assert parsed["test"]["status"] == "passed"
        assert parsed["execution"]["workflow_id"] == "test_workflow_123"
        assert parsed["execution"]["exit_code"] == 0


@pytest.mark.asyncio
class TestExecutionConfig:
    """Test cases for ExecutionConfig model."""

    def test_execution_config_defaults(self, sample_test_file):
        """Test ExecutionConfig with default values."""
        config = ExecutionConfig(test_file=str(sample_test_file))

        assert config.test_file == str(sample_test_file)
        assert config.timeout == 30000
        assert config.retries == 0
        assert config.workers == 1
        assert config.collect_traces is True
        assert config.collect_videos is True
        assert config.collect_screenshots is True

    def test_execution_config_ci_detection(self, sample_test_file):
        """Test ExecutionConfig CI environment detection."""
        with patch.dict(os.environ, {"CI": "true"}):
            config = ExecutionConfig(test_file=str(sample_test_file))

            assert config.ci_mode is True
            assert config.mode == ExecutionMode.HEADLESS

    def test_execution_config_headless_override(self, sample_test_file):
        """Test ExecutionConfig headless override."""
        with patch.dict(os.environ, {"QA_OPERATOR_HEADLESS": "true"}):
            config = ExecutionConfig(test_file=str(sample_test_file))

            assert config.force_headless is True
            assert config.is_headless is True

    def test_execution_config_effective_mode(self, sample_test_file):
        """Test ExecutionConfig effective mode calculation."""
        # Test normal mode
        config = ExecutionConfig(
            test_file=str(sample_test_file),
            mode=ExecutionMode.HEADED,
            force_headless=False,
        )
        assert config.effective_mode == ExecutionMode.HEADED

        # Test forced headless
        config.force_headless = True
        assert config.effective_mode == ExecutionMode.HEADLESS

    def test_execution_config_validation_nonexistent_file(self):
        """Test ExecutionConfig validation with nonexistent file."""
        with pytest.raises(ValueError, match="Test file does not exist"):
            ExecutionConfig(test_file="/nonexistent/test.ts")

    def test_execution_config_validation_invalid_extension(self, temp_config):
        """Test ExecutionConfig validation with invalid file extension."""
        invalid_file = temp_config.e2e_dir / "test.py"
        invalid_file.write_text("# Not a test file")

        with pytest.raises(ValueError, match="must be JavaScript or TypeScript"):
            ExecutionConfig(test_file=str(invalid_file))

    def test_execution_config_validation_screenshot_mode(self, sample_test_file):
        """Test ExecutionConfig validation of screenshot mode."""
        # Valid modes
        for mode in ["on", "off", "only-on-failure"]:
            config = ExecutionConfig(
                test_file=str(sample_test_file),
                screenshot_mode=mode,
            )
            assert config.screenshot_mode == mode

        # Invalid mode
        with pytest.raises(ValueError, match="Screenshot mode must be one of"):
            ExecutionConfig(
                test_file=str(sample_test_file),
                screenshot_mode="invalid",
            )


@pytest.mark.asyncio
class TestEnvironmentDetection:
    """Test cases for environment detection functionality."""

    def test_ci_environment_detection(
        self, temp_config, mock_playwright_client, mock_artifact_manager
    ):
        """Test CI environment detection."""
        with patch.dict(os.environ, {"CI": "true"}):
            # Update config to reflect CI environment
            temp_config.ci_mode = True
            temp_config.headless_mode = True

            executor = TestExecutor(
                config=temp_config,
                playwright_client=mock_playwright_client,
                artifact_manager=mock_artifact_manager,
            )

            mode = executor.detect_execution_mode()
            assert mode == ExecutionMode.HEADLESS

    def test_development_environment_detection(
        self, temp_config, mock_playwright_client, mock_artifact_manager
    ):
        """Test development environment detection."""
        with patch.dict(os.environ, {"CI": ""}, clear=True):
            # Update config to reflect development environment
            temp_config.ci_mode = False
            temp_config.headless_mode = False

            executor = TestExecutor(
                config=temp_config,
                playwright_client=mock_playwright_client,
                artifact_manager=mock_artifact_manager,
            )

            mode = executor.detect_execution_mode()
            assert mode == ExecutionMode.HEADED

    def test_environment_variable_override_precedence(
        self, temp_config, mock_playwright_client, mock_artifact_manager
    ):
        """Test that environment variable overrides take precedence."""
        with patch.dict(
            os.environ,
            {
                "CI": "false",  # Not CI
                "QA_OPERATOR_HEADLESS": "true",  # But force headless
            },
        ):
            executor = TestExecutor(
                config=temp_config,
                playwright_client=mock_playwright_client,
                artifact_manager=mock_artifact_manager,
            )

            mode = executor.detect_execution_mode()
            assert mode == ExecutionMode.HEADLESS

    def test_command_line_flag_highest_precedence(
        self, temp_config, mock_playwright_client, mock_artifact_manager
    ):
        """Test that command-line flags have highest precedence."""
        with patch.dict(
            os.environ,
            {
                "CI": "false",
                "QA_OPERATOR_HEADLESS": "false",
            },
        ):
            executor = TestExecutor(
                config=temp_config,
                playwright_client=mock_playwright_client,
                artifact_manager=mock_artifact_manager,
            )

            # Command-line flag should override everything
            mode = executor.detect_execution_mode(force_headless=True)
            assert mode == ExecutionMode.HEADLESS


if __name__ == "__main__":
    pytest.main([__file__])
