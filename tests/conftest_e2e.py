"""
Extended pytest configuration for end-to-end testing.

Provides additional fixtures and configuration specifically
for end-to-end workflow testing and validation.
"""

import os
import json
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from orchestrator.core.config import Config
from orchestrator.agent import QAOperatorAgent
from orchestrator.planning.models import TestSpecification
from orchestrator.execution.models import ExecutionResult, TestStatus


@pytest.fixture(scope="session")
def e2e_test_config():
    """Configuration optimized for E2E testing."""
    config = Config()
    config.ci_mode = True
    config.headless_mode = True
    config.log_level = "ERROR"  # Reduce noise in tests
    config.artifact_retention_days = 1  # Short retention for tests
    config.model_provider = "mixed"
    config.openai_api_key = "test-api-key"
    return config


@pytest.fixture(scope="session")
def sample_specifications():
    """Load sample test specifications for E2E testing."""
    specs_dir = Path(__file__).parent.parent / "e2e" / "sample_specifications"
    specifications = []

    if specs_dir.exists():
        for spec_file in specs_dir.glob("*.json"):
            try:
                with open(spec_file) as f:
                    spec_data = json.load(f)
                specifications.append(TestSpecification(**spec_data))
            except Exception as e:
                pytest.fail(f"Failed to load specification {spec_file}: {e}")

    if not specifications:
        # Create minimal test specifications if none exist
        specifications = [
            TestSpecification(
                id="test-spec-001",
                name="Basic Test Specification",
                description="Basic test for E2E validation",
                requirements=["User can perform basic action"],
                priority="medium",
                tags=["test", "basic"],
            )
        ]

    return specifications


@pytest.fixture
def mock_mcp_environment():
    """Create a complete mock MCP environment."""
    environment = {
        "connection_manager": MagicMock(),
        "playwright_client": MagicMock(),
        "filesystem_client": MagicMock(),
        "git_client": MagicMock(),
    }

    # Configure connection manager
    environment["connection_manager"].connect_server = AsyncMock(return_value=True)
    environment["connection_manager"].disconnect_server = AsyncMock()
    environment["connection_manager"].call_tool = AsyncMock()
    environment["connection_manager"].get_connection_status = MagicMock(
        return_value="connected"
    )
    environment["connection_manager"].list_servers = MagicMock(
        return_value=["playwright", "filesystem", "git"]
    )

    # Configure Playwright client
    environment["playwright_client"].launch_browser = AsyncMock(
        return_value={"success": True, "browser_id": "browser-123"}
    )
    environment["playwright_client"].navigate = AsyncMock(
        return_value={"success": True, "url": "https://example.com"}
    )
    environment["playwright_client"].execute_test = AsyncMock(
        return_value={
            "success": True,
            "test_results": {"passed": 1, "failed": 0, "skipped": 0, "total": 1},
            "artifacts": {
                "trace_file": "trace.zip",
                "screenshots": ["screenshot.png"],
                "console_logs": [{"level": "info", "message": "Test completed"}],
                "network_logs": [{"url": "https://example.com", "status": 200}],
            },
            "duration": 5.0,
        }
    )

    # Configure Filesystem client
    environment["filesystem_client"].read_file = AsyncMock(
        return_value={"success": True, "content": "test file content"}
    )
    environment["filesystem_client"].write_file = AsyncMock(
        return_value={"success": True}
    )
    environment["filesystem_client"].list_files = AsyncMock(
        return_value={
            "success": True,
            "files": [{"name": "test.spec.ts", "type": "file", "size": 1024}],
        }
    )
    environment["filesystem_client"].file_exists = AsyncMock(return_value=True)

    # Configure Git client
    environment["git_client"].is_available = AsyncMock(return_value=True)
    environment["git_client"].stage_files = AsyncMock(
        return_value={"success": True, "staged_files": ["test.spec.ts"]}
    )
    environment["git_client"].create_commit = AsyncMock(
        return_value={"success": True, "commit_hash": "abc123def456"}
    )
    environment["git_client"].create_pull_request = AsyncMock(
        return_value={"success": True, "pr_url": "https://github.com/repo/pull/123"}
    )

    return environment


@pytest.fixture
def mock_qa_agent(e2e_test_config, mock_mcp_environment):
    """Create a fully mocked QA Operator agent for E2E testing."""
    agent = QAOperatorAgent(e2e_test_config)

    # Inject mock MCP clients
    agent.connection_manager = mock_mcp_environment["connection_manager"]
    agent.playwright_client = mock_mcp_environment["playwright_client"]
    agent.filesystem_client = mock_mcp_environment["filesystem_client"]
    agent.git_client = mock_mcp_environment["git_client"]

    # Mock core components
    agent.planning_engine.create_test_plan = AsyncMock(
        return_value={
            "test_cases": [
                {
                    "name": "test_main_functionality",
                    "description": "Test main functionality",
                    "steps": ["Navigate to page", "Perform action", "Verify result"],
                    "assertions": ["Result is visible"],
                }
            ],
            "page_objects": [],
            "setup_requirements": [],
            "estimated_duration": 30,
        }
    )

    agent.test_generator.generate_test = AsyncMock(
        return_value={
            "success": True,
            "test_file": "generated_test.spec.ts",
            "content": """
import { test, expect } from '@playwright/test';

test('main functionality test', async ({ page }) => {
  await page.goto('/');
  await page.click('[data-testid="action-button"]');
  await expect(page.locator('[data-testid="result"]')).toBeVisible();
});
        """.strip(),
        }
    )

    agent.test_executor.execute_tests = AsyncMock(
        return_value=ExecutionResult(
            success=True,
            test_results={"passed": 1, "failed": 0, "skipped": 0, "total": 1},
            artifacts={
                "trace_file": "trace.zip",
                "screenshots": ["screenshot.png"],
                "console_logs": [
                    {"level": "info", "message": "Test completed successfully"}
                ],
                "network_logs": [
                    {"url": "https://example.com", "method": "GET", "status": 200}
                ],
            },
            duration=5.5,
            test_file="generated_test.spec.ts",
            status=TestStatus.COMPLETED,
        )
    )

    agent.failure_analyzer.analyze_failure = AsyncMock(
        return_value={
            "root_cause": "Element not found",
            "error_category": "selector",
            "confidence": 0.9,
            "suggested_fixes": [
                {
                    "type": "selector_update",
                    "description": "Update selector to use data-testid",
                    "old_selector": "button.submit",
                    "new_selector": "button[data-testid='submit-button']",
                }
            ],
            "artifacts_analyzed": ["trace.zip", "screenshot.png"],
        }
    )

    agent.code_patcher.apply_patch = AsyncMock(
        return_value={
            "success": True,
            "patched_file": "generated_test.spec.ts",
            "changes_applied": 1,
            "backup_created": True,
        }
    )

    agent.selector_auditor.audit_selectors = AsyncMock(
        return_value={
            "compliant": True,
            "violations": [],
            "score": 1.0,
            "suggestions": [],
        }
    )

    return agent


@pytest.fixture
def temp_e2e_workspace(tmp_path):
    """Create a temporary E2E workspace with sample files."""
    workspace = tmp_path / "e2e_workspace"
    workspace.mkdir()

    # Create directory structure
    (workspace / "e2e").mkdir()
    (workspace / "artifacts").mkdir()
    (workspace / "logs").mkdir()
    (workspace / "policies").mkdir()

    # Create sample test files
    (workspace / "e2e" / "sample.spec.ts").write_text(
        """
import { test, expect } from '@playwright/test';

test('sample test', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('h1')).toBeVisible();
});
    """.strip()
    )

    # Create selector policy
    (workspace / "policies" / "selector.md").write_text(
        """
# Selector Policy

## Preferred Selectors
1. getByRole()
2. getByLabel()
3. getByTestId()

## Discouraged
- CSS selectors
- XPath selectors
    """.strip()
    )

    # Create MCP config
    mcp_config = {
        "mcpServers": {
            "playwright": {
                "command": "uvx",
                "args": ["playwright-mcp-server"],
                "timeout": 30,
                "max_retries": 3,
            },
            "filesystem": {
                "command": "uvx",
                "args": ["filesystem-mcp-server", "--sandbox", "e2e/"],
                "timeout": 30,
                "max_retries": 3,
            },
        }
    }

    (workspace / "mcp.config.json").write_text(json.dumps(mcp_config, indent=2))

    return workspace


@pytest.fixture
def performance_monitor():
    """Create a performance monitoring utility for tests."""
    import time
    import psutil

    class TestPerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.metrics = {}
            self.process = psutil.Process()

        def start(self):
            self.start_time = time.time()
            self.metrics["initial_memory"] = (
                self.process.memory_info().rss / 1024 / 1024
            )
            self.metrics["initial_cpu"] = self.process.cpu_percent()

        def stop(self):
            if self.start_time:
                self.metrics["duration"] = time.time() - self.start_time
                self.metrics["final_memory"] = (
                    self.process.memory_info().rss / 1024 / 1024
                )
                self.metrics["memory_increase"] = (
                    self.metrics["final_memory"] - self.metrics["initial_memory"]
                )
                self.metrics["final_cpu"] = self.process.cpu_percent()
            return self.metrics

    return TestPerformanceMonitor()


@pytest.fixture
def mock_ai_responses():
    """Provide mock AI model responses for testing."""
    return {
        "planning_response": {
            "test_plan": {
                "test_cases": [
                    {
                        "name": "test_user_login",
                        "description": "Test user login functionality",
                        "steps": [
                            "Navigate to login page",
                            "Enter valid credentials",
                            "Click login button",
                            "Verify successful login",
                        ],
                        "assertions": [
                            "User is redirected to dashboard",
                            "Welcome message is displayed",
                        ],
                    }
                ],
                "estimated_duration": 45,
            }
        },
        "generation_response": {
            "test_code": """
import { test, expect } from '@playwright/test';

test('user login', async ({ page }) => {
  await page.goto('/login');
  await page.fill('[data-testid="email"]', 'user@example.com');
  await page.fill('[data-testid="password"]', 'password123');
  await page.click('[data-testid="login-button"]');
  await expect(page.locator('[data-testid="welcome"]')).toBeVisible();
});
            """.strip()
        },
        "analysis_response": {
            "failure_analysis": {
                "root_cause": "Login button selector changed",
                "confidence": 0.95,
                "suggested_fix": "Update selector from '#login-btn' to '[data-testid=\"login-button\"]'",
            }
        },
    }


@pytest.fixture(autouse=True)
def setup_e2e_environment(monkeypatch):
    """Set up environment variables for E2E testing."""
    # Set test environment variables
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("QA_OPERATOR_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("QA_OPERATOR_HEADLESS", "true")
    monkeypatch.setenv("QA_OPERATOR_ARTIFACT_RETENTION_DAYS", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Disable actual MCP connections in tests
    monkeypatch.setenv("QA_OPERATOR_DISABLE_MCP", "true")


@pytest.fixture
def validation_metrics():
    """Provide validation metrics tracking."""

    class ValidationMetrics:
        def __init__(self):
            self.metrics = {
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "total_duration": 0,
                "errors": [],
            }

        def record_test(
            self, name: str, success: bool, duration: float, error: str = None
        ):
            self.metrics["tests_run"] += 1
            if success:
                self.metrics["tests_passed"] += 1
            else:
                self.metrics["tests_failed"] += 1
                if error:
                    self.metrics["errors"].append(f"{name}: {error}")
            self.metrics["total_duration"] += duration

        def get_summary(self):
            return {
                "success_rate": self.metrics["tests_passed"]
                / max(self.metrics["tests_run"], 1),
                "total_tests": self.metrics["tests_run"],
                "passed": self.metrics["tests_passed"],
                "failed": self.metrics["tests_failed"],
                "avg_duration": self.metrics["total_duration"]
                / max(self.metrics["tests_run"], 1),
                "errors": self.metrics["errors"],
            }

    return ValidationMetrics()


# Custom pytest markers for E2E testing
def pytest_configure(config):
    """Configure custom markers for E2E testing."""
    config.addinivalue_line("markers", "e2e: End-to-end integration tests")
    config.addinivalue_line("markers", "performance: Performance benchmark tests")
    config.addinivalue_line("markers", "validation: Validation and compliance tests")
    config.addinivalue_line(
        "markers", "dry_run: Dry-run tests with mocked dependencies"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection for E2E testing."""
    for item in items:
        # Mark E2E tests
        if "e2e" in item.nodeid.lower():
            item.add_marker(pytest.mark.e2e)

        # Mark performance tests
        if "performance" in item.nodeid.lower() or "benchmark" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

        # Mark validation tests
        if "validation" in item.nodeid.lower() or "validate" in item.nodeid.lower():
            item.add_marker(pytest.mark.validation)


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for async tests."""
    return asyncio.DefaultEventLoopPolicy()


# Async test timeout configuration
@pytest.fixture(scope="session")
def async_timeout():
    """Default timeout for async tests."""
    return 30.0  # 30 seconds
