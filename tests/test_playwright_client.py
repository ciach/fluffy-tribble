"""
Unit tests for Playwright MCP Client.

Tests high-level interface for Playwright MCP tool calls including
browser automation, test execution, and artifact collection.
"""

import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from orchestrator.mcp.playwright_client import (
    PlaywrightMCPClient,
    BrowserMode,
    TestArtifacts,
)
from orchestrator.core.exceptions import TestExecutionError, MCPConnectionError
from orchestrator.mcp.models import (
    BrowserAction,
    BrowserActionResult,
    TestConfiguration,
)


class TestBrowserMode:
    """Test cases for BrowserMode enum."""

    def test_browser_mode_values(self):
        """Test BrowserMode enum values."""
        assert BrowserMode.HEADED.value == "headed"
        assert BrowserMode.HEADLESS.value == "headless"


class TestTestArtifacts:
    """Test cases for TestArtifacts."""

    def test_empty_artifacts(self):
        """Test creating empty test artifacts."""
        artifacts = TestArtifacts()

        assert artifacts.trace_file is None
        assert artifacts.screenshots == []
        assert artifacts.console_logs == []
        assert artifacts.network_logs == []
        assert artifacts.video_file is None

    def test_artifacts_with_data(self):
        """Test creating test artifacts with data."""
        artifacts = TestArtifacts(
            trace_file="trace.zip",
            screenshots=["screenshot1.png", "screenshot2.png"],
            console_logs=[{"level": "error", "message": "Test error"}],
            network_logs=[{"url": "https://example.com", "status": 200}],
            video_file="test.webm",
        )

        assert artifacts.trace_file == "trace.zip"
        assert len(artifacts.screenshots) == 2
        assert len(artifacts.console_logs) == 1
        assert len(artifacts.network_logs) == 1
        assert artifacts.video_file == "test.webm"


class TestPlaywrightMCPClient:
    """Test cases for PlaywrightMCPClient."""

    @pytest.fixture
    def mock_connection_manager(self):
        """Create mock connection manager."""
        manager = MagicMock()
        manager.call_tool = AsyncMock()
        return manager

    @pytest.fixture
    def playwright_client(self, mock_connection_manager):
        """Create Playwright client instance."""
        return PlaywrightMCPClient(
            connection_manager=mock_connection_manager,
            artifacts_dir=Path("/tmp/artifacts"),
        )

    def test_client_initialization(self, playwright_client):
        """Test Playwright client initialization."""
        assert playwright_client.server_name == "playwright"
        assert playwright_client.artifacts_dir == Path("/tmp/artifacts")

    @pytest.mark.asyncio
    async def test_launch_browser_success(
        self, playwright_client, mock_connection_manager
    ):
        """Test successful browser launch."""
        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "browser_id": "browser-123",
            "context_id": "context-456",
        }

        result = await playwright_client.launch_browser(mode=BrowserMode.HEADLESS)

        assert result.success is True
        assert result.browser_id == "browser-123"
        assert result.context_id == "context-456"

        mock_connection_manager.call_tool.assert_called_once_with(
            "playwright", "launch_browser", {"headless": True}
        )

    @pytest.mark.asyncio
    async def test_launch_browser_failure(
        self, playwright_client, mock_connection_manager
    ):
        """Test browser launch failure."""
        mock_connection_manager.call_tool.side_effect = MCPConnectionError(
            "Connection failed"
        )

        with pytest.raises(TestExecutionError, match="Failed to launch browser"):
            await playwright_client.launch_browser()

    @pytest.mark.asyncio
    async def test_navigate_to_page(self, playwright_client, mock_connection_manager):
        """Test navigating to a page."""
        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "url": "https://example.com",
            "title": "Example Page",
            "load_time": 1.23,
        }

        result = await playwright_client.navigate("https://example.com")

        assert result.success is True
        assert result.url == "https://example.com"
        assert result.title == "Example Page"
        assert result.load_time == 1.23

        mock_connection_manager.call_tool.assert_called_once_with(
            "playwright",
            "navigate",
            {
                "url": "https://example.com",
                "wait_until": "networkidle",
                "timeout": 30000,
            },
        )

    @pytest.mark.asyncio
    async def test_navigate_with_custom_options(
        self, playwright_client, mock_connection_manager
    ):
        """Test navigating with custom options."""
        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "url": "https://example.com",
            "title": "Example Page",
        }

        await playwright_client.navigate(
            "https://example.com", wait_until="domcontentloaded", timeout=60000
        )

        mock_connection_manager.call_tool.assert_called_once_with(
            "playwright",
            "navigate",
            {
                "url": "https://example.com",
                "wait_until": "domcontentloaded",
                "timeout": 60000,
            },
        )

    @pytest.mark.asyncio
    async def test_click_element(self, playwright_client, mock_connection_manager):
        """Test clicking an element."""
        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "element_found": True,
            "clicked": True,
        }

        result = await playwright_client.click("button[data-testid='submit']")

        assert result.success is True
        assert result.element_found is True

        mock_connection_manager.call_tool.assert_called_once_with(
            "playwright",
            "click",
            {
                "selector": "button[data-testid='submit']",
                "timeout": 30000,
                "force": False,
            },
        )

    @pytest.mark.asyncio
    async def test_click_element_not_found(
        self, playwright_client, mock_connection_manager
    ):
        """Test clicking element that doesn't exist."""
        mock_connection_manager.call_tool.return_value = {
            "success": False,
            "element_found": False,
            "error": "Element not found",
        }

        result = await playwright_client.click("button[data-testid='nonexistent']")

        assert result.success is False
        assert result.element_found is False
        assert "Element not found" in result.error

    @pytest.mark.asyncio
    async def test_fill_input(self, playwright_client, mock_connection_manager):
        """Test filling an input field."""
        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "element_found": True,
            "filled": True,
            "value": "test@example.com",
        }

        result = await playwright_client.fill(
            "input[data-testid='email']", "test@example.com"
        )

        assert result.success is True
        assert result.value == "test@example.com"

        mock_connection_manager.call_tool.assert_called_once_with(
            "playwright",
            "fill",
            {
                "selector": "input[data-testid='email']",
                "value": "test@example.com",
                "timeout": 30000,
                "clear": True,
            },
        )

    @pytest.mark.asyncio
    async def test_wait_for_element(self, playwright_client, mock_connection_manager):
        """Test waiting for element to appear."""
        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "element_found": True,
            "visible": True,
            "wait_time": 0.5,
        }

        result = await playwright_client.wait_for_element(
            "div[data-testid='success-message']", state="visible"
        )

        assert result.success is True
        assert result.element_found is True
        assert result.visible is True

        mock_connection_manager.call_tool.assert_called_once_with(
            "playwright",
            "wait_for_element",
            {
                "selector": "div[data-testid='success-message']",
                "state": "visible",
                "timeout": 30000,
            },
        )

    @pytest.mark.asyncio
    async def test_take_screenshot(self, playwright_client, mock_connection_manager):
        """Test taking a screenshot."""
        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "screenshot_path": "/tmp/artifacts/screenshot.png",
            "size": {"width": 1280, "height": 720},
        }

        result = await playwright_client.take_screenshot("test_screenshot.png")

        assert result.success is True
        assert result.screenshot_path == "/tmp/artifacts/screenshot.png"

        mock_connection_manager.call_tool.assert_called_once_with(
            "playwright",
            "screenshot",
            {
                "path": "/tmp/artifacts/test_screenshot.png",
                "full_page": False,
                "quality": 90,
            },
        )

    @pytest.mark.asyncio
    async def test_take_full_page_screenshot(
        self, playwright_client, mock_connection_manager
    ):
        """Test taking a full page screenshot."""
        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "screenshot_path": "/tmp/artifacts/full_page.png",
        }

        await playwright_client.take_screenshot(
            "full_page.png", full_page=True, quality=100
        )

        mock_connection_manager.call_tool.assert_called_once_with(
            "playwright",
            "screenshot",
            {"path": "/tmp/artifacts/full_page.png", "full_page": True, "quality": 100},
        )

    @pytest.mark.asyncio
    async def test_execute_test_file(self, playwright_client, mock_connection_manager):
        """Test executing a test file."""
        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "test_results": {"passed": 5, "failed": 1, "skipped": 0, "total": 6},
            "artifacts": {
                "trace_file": "trace.zip",
                "screenshots": ["screenshot1.png"],
                "console_logs": [{"level": "error", "message": "Test error"}],
                "network_logs": [],
            },
            "duration": 12.5,
        }

        result = await playwright_client.execute_test(
            "tests/login.spec.ts", mode=BrowserMode.HEADLESS
        )

        assert result.success is True
        assert result.test_results["passed"] == 5
        assert result.test_results["failed"] == 1
        assert result.duration == 12.5
        assert isinstance(result.artifacts, TestArtifacts)
        assert result.artifacts.trace_file == "trace.zip"

        mock_connection_manager.call_tool.assert_called_once_with(
            "playwright",
            "run_test",
            {
                "test_file": "tests/login.spec.ts",
                "headless": True,
                "trace": True,
                "video": False,
                "artifacts_dir": "/tmp/artifacts",
            },
        )

    @pytest.mark.asyncio
    async def test_execute_test_with_video(
        self, playwright_client, mock_connection_manager
    ):
        """Test executing test with video recording."""
        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "test_results": {"passed": 1, "failed": 0, "skipped": 0, "total": 1},
            "artifacts": {
                "trace_file": "trace.zip",
                "video_file": "test.webm",
                "screenshots": [],
                "console_logs": [],
                "network_logs": [],
            },
            "duration": 5.2,
        }

        result = await playwright_client.execute_test(
            "tests/simple.spec.ts", mode=BrowserMode.HEADED, record_video=True
        )

        assert result.success is True
        assert result.artifacts.video_file == "test.webm"

        mock_connection_manager.call_tool.assert_called_once_with(
            "playwright",
            "run_test",
            {
                "test_file": "tests/simple.spec.ts",
                "headless": False,
                "trace": True,
                "video": True,
                "artifacts_dir": "/tmp/artifacts",
            },
        )

    @pytest.mark.asyncio
    async def test_get_console_logs(self, playwright_client, mock_connection_manager):
        """Test getting console logs."""
        mock_connection_manager.call_tool.return_value = {
            "logs": [
                {
                    "level": "info",
                    "message": "Page loaded",
                    "timestamp": "2024-01-15T10:30:00Z",
                },
                {
                    "level": "error",
                    "message": "API call failed",
                    "timestamp": "2024-01-15T10:30:05Z",
                },
            ]
        }

        logs = await playwright_client.get_console_logs()

        assert len(logs) == 2
        assert logs[0]["level"] == "info"
        assert logs[1]["level"] == "error"

        mock_connection_manager.call_tool.assert_called_once_with(
            "playwright", "get_console_logs", {"clear_after_get": False}
        )

    @pytest.mark.asyncio
    async def test_get_network_logs(self, playwright_client, mock_connection_manager):
        """Test getting network logs."""
        mock_connection_manager.call_tool.return_value = {
            "requests": [
                {
                    "url": "https://api.example.com/users",
                    "method": "GET",
                    "status": 200,
                    "response_time": 150,
                },
                {
                    "url": "https://api.example.com/login",
                    "method": "POST",
                    "status": 401,
                    "response_time": 75,
                },
            ]
        }

        logs = await playwright_client.get_network_logs()

        assert len(logs) == 2
        assert logs[0]["status"] == 200
        assert logs[1]["status"] == 401

        mock_connection_manager.call_tool.assert_called_once_with(
            "playwright", "get_network_logs", {"clear_after_get": False}
        )

    @pytest.mark.asyncio
    async def test_close_browser(self, playwright_client, mock_connection_manager):
        """Test closing browser."""
        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "closed": True,
        }

        result = await playwright_client.close_browser()

        assert result.success is True

        mock_connection_manager.call_tool.assert_called_once_with(
            "playwright", "close_browser", {}
        )

    @pytest.mark.asyncio
    async def test_get_page_content(self, playwright_client, mock_connection_manager):
        """Test getting page content."""
        mock_connection_manager.call_tool.return_value = {
            "content": "<html><body><h1>Test Page</h1></body></html>",
            "url": "https://example.com",
            "title": "Test Page",
        }

        result = await playwright_client.get_page_content()

        assert "Test Page" in result["content"]
        assert result["url"] == "https://example.com"
        assert result["title"] == "Test Page"

    @pytest.mark.asyncio
    async def test_evaluate_javascript(
        self, playwright_client, mock_connection_manager
    ):
        """Test evaluating JavaScript in page."""
        mock_connection_manager.call_tool.return_value = {
            "result": "Hello World",
            "success": True,
        }

        result = await playwright_client.evaluate_javascript(
            "document.querySelector('h1').textContent"
        )

        assert result["result"] == "Hello World"
        assert result["success"] is True

        mock_connection_manager.call_tool.assert_called_once_with(
            "playwright",
            "evaluate",
            {"expression": "document.querySelector('h1').textContent"},
        )

    @pytest.mark.asyncio
    async def test_wait_for_network_idle(
        self, playwright_client, mock_connection_manager
    ):
        """Test waiting for network idle."""
        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "idle_time": 2.1,
        }

        result = await playwright_client.wait_for_network_idle(timeout=10000)

        assert result["success"] is True
        assert result["idle_time"] == 2.1

        mock_connection_manager.call_tool.assert_called_once_with(
            "playwright", "wait_for_network_idle", {"timeout": 10000}
        )

    @pytest.mark.asyncio
    async def test_error_handling_mcp_connection_error(
        self, playwright_client, mock_connection_manager
    ):
        """Test error handling for MCP connection errors."""
        mock_connection_manager.call_tool.side_effect = MCPConnectionError(
            "Server disconnected"
        )

        with pytest.raises(TestExecutionError, match="MCP connection error"):
            await playwright_client.navigate("https://example.com")

    @pytest.mark.asyncio
    async def test_error_handling_timeout(
        self, playwright_client, mock_connection_manager
    ):
        """Test error handling for timeouts."""
        mock_connection_manager.call_tool.side_effect = asyncio.TimeoutError(
            "Operation timed out"
        )

        with pytest.raises(TestExecutionError, match="Operation timed out"):
            await playwright_client.click("button")

    def test_artifacts_directory_creation(self, mock_connection_manager):
        """Test that artifacts directory is created if it doesn't exist."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"

            client = PlaywrightMCPClient(
                connection_manager=mock_connection_manager, artifacts_dir=artifacts_dir
            )

            assert artifacts_dir.exists()
            assert client.artifacts_dir == artifacts_dir
