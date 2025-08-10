"""
Pytest configuration and shared fixtures for QA Operator tests.

Provides common test fixtures, mock objects, and configuration
for all test modules.
"""

import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import pytest

from orchestrator.core.config import Config
from orchestrator.core.workflow import WorkflowContext
from orchestrator.planning.models import TestSpecification
from orchestrator.execution.models import ExecutionResult, TestStatus
from orchestrator.analysis.models import FailureAnalysis


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_config():
    """Create a temporary configuration for testing."""
    config = Config()
    config.openai_api_key = "test-api-key"
    config.model_provider = "mixed"
    config.log_level = "DEBUG"
    config.ci_mode = False
    config.headless_mode = False
    return config


@pytest.fixture
def temp_e2e_directory():
    """Create a temporary e2e directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        e2e_dir = Path(temp_dir) / "e2e"
        e2e_dir.mkdir()
        
        # Create some sample test files
        (e2e_dir / "login.spec.ts").write_text("""
import { test, expect } from '@playwright/test';

test('user can log in', async ({ page }) => {
  await page.goto('/login');
  await page.fill('[data-testid="email"]', 'user@example.com');
  await page.fill('[data-testid="password"]', 'password123');
  await page.click('[data-testid="submit"]');
  await expect(page.locator('[data-testid="welcome"]')).toBeVisible();
});
        """)
        
        (e2e_dir / "pages").mkdir()
        (e2e_dir / "pages" / "signup.spec.ts").write_text("""
import { test, expect } from '@playwright/test';

test('user can sign up', async ({ page }) => {
  await page.goto('/signup');
  await page.fill('[data-testid="name"]', 'John Doe');
  await page.fill('[data-testid="email"]', 'john@example.com');
  await page.fill('[data-testid="password"]', 'newpassword123');
  await page.click('[data-testid="submit"]');
  await expect(page.locator('[data-testid="success"]')).toBeVisible();
});
        """)
        
        yield e2e_dir


@pytest.fixture
def temp_artifacts_directory():
    """Create a temporary artifacts directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_dir = Path(temp_dir) / "artifacts"
        artifacts_dir.mkdir()
        
        # Create sample artifacts
        test_run_dir = artifacts_dir / "20240115_103000" / "login_test"
        test_run_dir.mkdir(parents=True)
        
        (test_run_dir / "trace.zip").write_bytes(b"mock trace data")
        (test_run_dir / "screenshot.png").write_bytes(b"mock screenshot data")
        (test_run_dir / "console.log").write_text("console.log('test message')")
        
        yield artifacts_dir


@pytest.fixture
def mock_mcp_config_file(tmp_path):
    """Create a mock MCP configuration file."""
    config_data = {
        "mcpServers": {
            "playwright": {
                "command": "uvx",
                "args": ["playwright-mcp-server"],
                "timeout": 30,
                "max_retries": 3
            },
            "filesystem": {
                "command": "uvx",
                "args": ["filesystem-mcp-server", "--sandbox", "e2e/"],
                "timeout": 30,
                "max_retries": 3
            },
            "git": {
                "command": "uvx",
                "args": ["git-mcp-server"],
                "timeout": 30,
                "max_retries": 3,
                "disabled": False
            }
        }
    }
    
    config_file = tmp_path / "mcp.config.json"
    config_file.write_text(json.dumps(config_data, indent=2))
    return config_file


@pytest.fixture
def mock_connection_manager():
    """Create a mock MCP connection manager."""
    manager = MagicMock()
    manager.connect_server = AsyncMock(return_value=True)
    manager.disconnect_server = AsyncMock()
    manager.call_tool = AsyncMock()
    manager.get_connection_status = MagicMock(return_value="connected")
    manager.list_servers = MagicMock(return_value=["playwright", "filesystem", "git"])
    manager.cleanup = AsyncMock()
    return manager


@pytest.fixture
def mock_playwright_client():
    """Create a mock Playwright MCP client."""
    client = MagicMock()
    client.launch_browser = AsyncMock(return_value={"success": True, "browser_id": "browser-123"})
    client.navigate = AsyncMock(return_value={"success": True, "url": "https://example.com"})
    client.click = AsyncMock(return_value={"success": True, "element_found": True})
    client.fill = AsyncMock(return_value={"success": True, "value": "test input"})
    client.take_screenshot = AsyncMock(return_value={"success": True, "screenshot_path": "screenshot.png"})
    client.execute_test = AsyncMock(return_value={
        "success": True,
        "test_results": {"passed": 1, "failed": 0, "skipped": 0, "total": 1},
        "artifacts": {"trace_file": "trace.zip", "screenshots": [], "console_logs": [], "network_logs": []},
        "duration": 2.5
    })
    client.close_browser = AsyncMock(return_value={"success": True})
    return client


@pytest.fixture
def mock_filesystem_client():
    """Create a mock Filesystem MCP client."""
    client = MagicMock()
    client.read_file = AsyncMock(return_value={"success": True, "content": "test file content"})
    client.write_file = AsyncMock(return_value={"success": True})
    client.delete_file = AsyncMock(return_value={"success": True})
    client.list_files = AsyncMock(return_value={
        "success": True,
        "files": [
            {"name": "test.spec.ts", "type": "file", "size": 1024},
            {"name": "pages", "type": "directory"}
        ]
    })
    client.file_exists = AsyncMock(return_value=True)
    client.create_directory = AsyncMock(return_value={"success": True})
    return client


@pytest.fixture
def mock_git_client():
    """Create a mock Git MCP client."""
    client = MagicMock()
    client.is_available = AsyncMock(return_value=True)
    client.stage_files = AsyncMock(return_value={"success": True, "staged_files": ["test.spec.ts"]})
    client.create_commit = AsyncMock(return_value={"success": True, "commit_hash": "abc123"})
    client.create_pull_request = AsyncMock(return_value={"success": True, "pr_url": "https://github.com/repo/pull/123"})
    client.get_repository_info = AsyncMock(return_value={
        "success": True,
        "repo_name": "test-repo",
        "branch": "main",
        "has_changes": True
    })
    return client


@pytest.fixture
def mock_model_router():
    """Create a mock model router."""
    router = MagicMock()
    router.route_task = MagicMock(return_value="openai")
    router.call_model = AsyncMock(return_value={
        "success": True,
        "response": "Mock AI response",
        "model_used": "gpt-4",
        "tokens_used": 150
    })
    router.is_model_available = MagicMock(return_value=True)
    return router


@pytest.fixture
def mock_artifact_manager():
    """Create a mock artifact manager."""
    manager = MagicMock()
    manager.store_artifacts = AsyncMock(return_value={"success": True, "artifact_path": "/tmp/artifacts/test"})
    manager.cleanup_old_artifacts = AsyncMock(return_value={"cleaned_count": 5})
    manager.get_artifact_info = MagicMock(return_value={
        "total_size": 1024000,
        "file_count": 10,
        "oldest_artifact": "2024-01-01T00:00:00Z"
    })
    return manager


@pytest.fixture
def sample_test_specification():
    """Create a sample test specification for testing."""
    return TestSpecification(
        id="test-spec-001",
        name="User Authentication Flow",
        description="Test complete user authentication including login, logout, and password reset",
        requirements=[
            "User can log in with valid credentials",
            "User can log out successfully",
            "User can reset password with valid email",
            "Invalid credentials show appropriate error messages"
        ],
        priority="high",
        tags=["authentication", "security", "user-flow"]
    )


@pytest.fixture
def sample_execution_result():
    """Create a sample execution result for testing."""
    return ExecutionResult(
        success=True,
        test_results={
            "passed": 3,
            "failed": 1,
            "skipped": 0,
            "total": 4
        },
        artifacts={
            "trace_file": "trace.zip",
            "screenshots": ["login_page.png", "error_state.png"],
            "console_logs": [
                {"level": "info", "message": "Page loaded", "timestamp": "2024-01-15T10:30:00Z"},
                {"level": "error", "message": "Login failed", "timestamp": "2024-01-15T10:30:05Z"}
            ],
            "network_logs": [
                {"url": "https://api.example.com/login", "method": "POST", "status": 401}
            ]
        },
        duration=15.7,
        test_file="auth_flow.spec.ts",
        status=TestStatus.COMPLETED
    )


@pytest.fixture
def sample_failure_analysis():
    """Create a sample failure analysis for testing."""
    return FailureAnalysis(
        root_cause="Login button selector not found",
        error_category="selector",
        confidence=0.85,
        suggested_fixes=[
            {
                "type": "selector_update",
                "description": "Update selector to use data-testid",
                "old_selector": "button.login-btn",
                "new_selector": "button[data-testid='login-submit']",
                "confidence": 0.9
            }
        ],
        artifacts_analyzed=["trace.zip", "screenshot.png", "console.log"],
        analysis_duration=2.3
    )


@pytest.fixture
def workflow_context():
    """Create a sample workflow context for testing."""
    return WorkflowContext(
        workflow_id="workflow-test-123456",
        config=Config(),
        metadata={
            "test_specification_id": "test-spec-001",
            "user_id": "test-user",
            "environment": "test"
        }
    )


@pytest.fixture
def mock_selector_policies():
    """Create mock selector policies content."""
    return """
# Selector Policy Guidelines

## Preferred Selectors (in order of preference)

1. **getByRole()** - Use for semantic elements
   - `getByRole('button', { name: 'Submit' })`
   - `getByRole('textbox', { name: 'Email' })`

2. **getByLabel()** - Use for form elements with labels
   - `getByLabel('Email Address')`
   - `getByLabel('Password')`

3. **getByTestId()** - Use for elements with test IDs
   - `getByTestId('submit-button')`
   - `getByTestId('error-message')`

## Discouraged Selectors

- CSS selectors (`.class`, `#id`, `tag`)
- XPath selectors
- Text-based selectors without semantic meaning

## Exceptions

CSS selectors may be used when:
- No semantic alternative exists
- Third-party components don't support test IDs
- Legacy code requires gradual migration

When using CSS selectors, include a comment explaining why:
```typescript
// Using CSS selector because third-party modal doesn't support test IDs
await page.click('.modal-dialog .close-button');
```
"""


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path, monkeypatch):
    """Set up test environment with temporary directories and environment variables."""
    # Set up temporary directories
    test_logs_dir = tmp_path / "logs"
    test_logs_dir.mkdir()
    
    test_artifacts_dir = tmp_path / "artifacts"
    test_artifacts_dir.mkdir()
    
    # Set environment variables for testing
    monkeypatch.setenv("QA_OPERATOR_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("QA_OPERATOR_ARTIFACT_RETENTION_DAYS", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    
    # Mock file paths
    monkeypatch.setattr("orchestrator.core.config.Config.get_log_file_path", 
                       lambda self: test_logs_dir / "qa-operator.log")
    monkeypatch.setattr("orchestrator.core.config.Config.get_debug_log_dir", 
                       lambda self: test_logs_dir / "debug")


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response from OpenAI API for testing purposes."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70
        }
    }


@pytest.fixture
def mock_ollama_response():
    """Create a mock Ollama API response."""
    return {
        "model": "qwen2.5:7b",
        "created_at": "2024-01-15T10:30:00Z",
        "response": "This is a mock response from Ollama for testing purposes.",
        "done": True,
        "context": [1, 2, 3, 4, 5],
        "total_duration": 1500000000,
        "load_duration": 500000000,
        "prompt_eval_count": 50,
        "prompt_eval_duration": 200000000,
        "eval_count": 20,
        "eval_duration": 800000000
    }


# Async test utilities
@pytest.fixture
def async_test_timeout():
    """Default timeout for async tests."""
    return 10.0  # seconds


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "mcp: mark test as requiring MCP servers"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark MCP tests
        if any(keyword in item.nodeid for keyword in ["mcp", "connection", "playwright", "filesystem"]):
            item.add_marker(pytest.mark.mcp)