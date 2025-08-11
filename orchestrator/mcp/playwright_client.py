"""
Playwright MCP Client Wrapper.

Provides a high-level interface for Playwright MCP tool calls including
browser automation, test execution, and artifact collection.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from types import SimpleNamespace
from enum import Enum

from pydantic import BaseModel, Field, validator, ConfigDict

from ..core.exceptions import TestExecutionError, MCPConnectionError
from .models import (
    MCPToolCall,
    MCPToolResponse,
    BrowserAction,
    BrowserActionResult,
    TestConfiguration,
)


class BrowserMode(Enum):
    """Browser execution mode."""

    HEADED = "headed"
    HEADLESS = "headless"


class TestArtifacts(BaseModel):
    """Container for test execution artifacts."""

    model_config = ConfigDict(extra="forbid")

    trace_file: Optional[str] = Field(None, description="Path to trace file")
    screenshots: List[str] = Field(
        default_factory=list, description="List of screenshot paths"
    )
    console_logs: List[Dict[str, Any]] = Field(
        default_factory=list, description="Console log entries"
    )
    network_logs: List[Dict[str, Any]] = Field(
        default_factory=list, description="Network log entries"
    )
    video_file: Optional[str] = Field(None, description="Path to video file")


class TestResult(BaseModel):
    """Container for test execution results."""

    model_config = ConfigDict(extra="forbid")

    test_name: str = Field(..., description="Name of the test")
    status: str = Field(..., description="Test status: passed, failed, or skipped")
    duration: float = Field(..., ge=0, description="Test duration in seconds")
    artifacts: TestArtifacts = Field(..., description="Test artifacts")
    error_info: Optional[Dict[str, Any]] = Field(
        None, description="Error information if test failed"
    )
    exit_code: Optional[int] = Field(None, description="Process exit code")

    @validator("status")
    def validate_status(cls, v):
        valid_statuses = ["passed", "failed", "skipped"]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return v

    @validator("test_name")
    def validate_test_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Test name cannot be empty")
        return v.strip()


class PlaywrightMCPClient:
    """
    Wrapper class for Playwright MCP tool calls.

    Provides high-level methods for browser automation, test execution,
    and artifact collection using the Playwright MCP server.
    """

    def __init__(
        self,
        connection_manager,
        artifacts_dir: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the Playwright MCP client.

        Args:
            connection_manager: MCP connection manager instance
            logger: Optional logger instance
        """
        self.connection_manager = connection_manager
        self.logger = logger or logging.getLogger(__name__)
        self.server_name = "playwright"

        # Browser state
        self._browser_context = None
        self._current_page = None
        # Artifacts directory (ensure it exists)
        self._artifacts_dir = (
            Path(artifacts_dir) if artifacts_dir else Path("artifacts")
        )
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Test execution state
        self._current_test_name = None
        self._test_start_time = None
        self._artifacts = TestArtifacts()

    @property
    def artifacts_dir(self) -> Path:
        """Public accessor for artifacts directory used in tests."""
        return self._artifacts_dir

    async def _ensure_connection(self) -> Any:
        """Ensure we have a valid connection to the Playwright MCP server."""
        if not self.connection_manager.is_connected(self.server_name):
            self.logger.warning(
                f"Playwright MCP server not connected, attempting to connect..."
            )
            success = await self.connection_manager.connect_server(self.server_name)
            if not success:
                raise MCPConnectionError(
                    f"Failed to connect to Playwright MCP server",
                    server_name=self.server_name,
                )

        return self.connection_manager.get_connection(self.server_name)

    async def _call_mcp_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call an MCP tool with error handling.

        Args:
            tool_name: Name of the MCP tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool response data
        """
        await self._ensure_connection()

        try:
            self.logger.debug(f"Calling MCP tool: {tool_name} with args: {arguments}")
            # Delegate to the connection manager as tests expect
            return await self.connection_manager.call_tool(
                self.server_name, tool_name, arguments
            )
        except Exception as e:
            # Preserve original error message for tests to match
            self.logger.error(f"MCP tool call failed: {tool_name} - {e}")
            raise TestExecutionError(str(e))

    async def launch_browser(
        self, mode: BrowserMode = BrowserMode.HEADLESS, **options
    ) -> SimpleNamespace:
        """
        Launch a browser instance.

        Args:
            mode: Browser mode (headed or headless)
            **options: Additional browser launch options

        Returns:
            Browser ID for subsequent operations
        """
        self.logger.info(f"Launching browser in {mode.value} mode")

        launch_args = {"headless": mode == BrowserMode.HEADLESS, **options}

        try:
            response = await self._call_mcp_tool("launch_browser", launch_args)
        except Exception as e:
            raise TestExecutionError(f"Failed to launch browser: {e}")

        if not response.get("success", False):
            raise TestExecutionError("Failed to launch browser")

        return SimpleNamespace(**response)

    async def new_page(self, browser_id: str) -> str:
        """
        Create a new page in the browser.

        Args:
            browser_id: Browser instance ID

        Returns:
            Page ID for subsequent operations
        """
        response = await self._call_mcp_tool("new_page", {"browser_id": browser_id})

        if not response.get("success"):
            raise TestExecutionError("Failed to create new page")

        page_id = response.get("page_id")
        self._current_page = page_id
        self.logger.debug(f"New page created with ID: {page_id}")
        return page_id

    async def navigate(
        self, url: str, wait_until: str = "networkidle", timeout: int = 30000
    ) -> SimpleNamespace:
        """
        Navigate to a URL.

        Args:
            page_id: Page instance ID
            url: URL to navigate to
            wait_until: When to consider navigation complete
        """
        self.logger.info(f"Navigating to: {url}")

        try:
            response = await self._call_mcp_tool(
                "navigate", {"url": url, "wait_until": wait_until, "timeout": timeout}
            )
        except Exception as e:
            raise TestExecutionError(f"MCP connection error: {e}")

        if not response.get("success", True):
            raise TestExecutionError(f"Failed to navigate to {url}")

        return SimpleNamespace(**response)

    async def click(
        self, selector: str, timeout: int = 30000, force: bool = False
    ) -> SimpleNamespace:
        """
        Click on an element.

        Args:
            page_id: Page instance ID
            selector: Element selector
            **options: Additional click options
        """
        self.logger.debug(f"Clicking element: {selector}")

        try:
            response = await self._call_mcp_tool(
                "click", {"selector": selector, "timeout": timeout, "force": force}
            )
        except Exception as e:
            raise TestExecutionError(str(e))

        return SimpleNamespace(**response)

    async def fill(
        self, selector: str, value: str, timeout: int = 30000, clear: bool = True
    ) -> SimpleNamespace:
        """
        Fill an input element.

        Args:
            page_id: Page instance ID
            selector: Input element selector
            value: Value to fill
            **options: Additional fill options
        """
        self.logger.debug(f"Filling element {selector} with value: {value}")

        try:
            response = await self._call_mcp_tool(
                "fill",
                {
                    "selector": selector,
                    "value": value,
                    "timeout": timeout,
                    "clear": clear,
                },
            )
        except Exception as e:
            raise TestExecutionError(str(e))

        return SimpleNamespace(**response)

    async def take_screenshot(
        self, filename: str, full_page: bool = False, quality: int = 90
    ) -> SimpleNamespace:
        """
        Take a screenshot of the page.

        Args:
            page_id: Page instance ID
            path: Optional path to save screenshot
            **options: Additional screenshot options

        Returns:
            Path to the saved screenshot
        """
        path = str(self._artifacts_dir / filename)
        self.logger.debug(f"Taking screenshot: {path}")
        response = await self._call_mcp_tool(
            "screenshot", {"path": path, "full_page": full_page, "quality": quality}
        )
        return SimpleNamespace(**response)

    async def wait_for_element(
        self, selector: str, state: str = "visible", timeout: int = 30000
    ) -> SimpleNamespace:
        """
        Wait for an element to appear.

        Args:
            page_id: Page instance ID
            selector: Element selector to wait for
            timeout: Timeout in milliseconds
        """
        self.logger.debug(f"Waiting for selector: {selector}")

        response = await self._call_mcp_tool(
            "wait_for_element",
            {"selector": selector, "state": state, "timeout": timeout},
        )
        return SimpleNamespace(**response)

    async def get_text(self, selector: str) -> str:
        """
        Get text content of an element.

        Args:
            page_id: Page instance ID
            selector: Element selector

        Returns:
            Text content of the element
        """
        response = await self._call_mcp_tool("get_text", {"selector": selector})

        if not response.get("success"):
            raise TestExecutionError(f"Failed to get text from element: {selector}")

        return response.get("text", "")

    async def execute_test(
        self,
        test_file: str,
        mode: BrowserMode = BrowserMode.HEADLESS,
        artifacts_dir: Optional[str] = None,
        record_video: bool = False,
        **options,
    ) -> SimpleNamespace:
        """
        Execute a Playwright test file.

        Args:
            test_file: Path to the test file
            mode: Browser mode for test execution
            artifacts_dir: Directory to store artifacts
            **options: Additional test execution options

        Returns:
            Test execution results with artifacts
        """
        test_name = Path(test_file).stem
        self._current_test_name = test_name
        self._test_start_time = time.time()
        self._artifacts = TestArtifacts()

        if artifacts_dir:
            self._artifacts_dir = Path(artifacts_dir)
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Executing test: {test_file} in {mode.value} mode")

        try:
            exec_args = {
                "test_file": test_file,
                "headless": mode == BrowserMode.HEADLESS,
                "trace": True,
                "video": record_video,
                "artifacts_dir": str(self._artifacts_dir),
            }
            exec_args.update(options)

            response = await self._call_mcp_tool("run_test", exec_args)

            duration = response.get("duration")
            artifacts_data = response.get("artifacts", {})
            artifacts = TestArtifacts(
                trace_file=artifacts_data.get("trace_file"),
                screenshots=artifacts_data.get("screenshots", []),
                console_logs=artifacts_data.get("console_logs", []),
                network_logs=artifacts_data.get("network_logs", []),
                video_file=artifacts_data.get("video_file"),
            )

            return SimpleNamespace(
                success=response.get("success", True),
                test_results=response.get("test_results", {}),
                artifacts=artifacts,
                duration=duration,
            )
        except Exception as e:
            raise TestExecutionError(str(e))

        finally:
            self._current_test_name = None
            self._test_start_time = None

    async def get_console_logs(
        self, clear_after_get: bool = False
    ) -> List[Dict[str, Any]]:
        """Get console logs collected by the server."""
        response = await self._call_mcp_tool(
            "get_console_logs", {"clear_after_get": clear_after_get}
        )
        return response.get("logs", [])

    async def get_network_logs(
        self, clear_after_get: bool = False
    ) -> List[Dict[str, Any]]:
        """Get network logs collected by the server."""
        response = await self._call_mcp_tool(
            "get_network_logs", {"clear_after_get": clear_after_get}
        )
        return response.get("requests", [])

    async def close_browser(self) -> SimpleNamespace:
        """Close the active browser session."""
        response = await self._call_mcp_tool("close_browser", {})
        return SimpleNamespace(**response)

    async def get_page_content(self) -> Dict[str, Any]:
        """Get current page content and info."""
        return await self._call_mcp_tool("get_page_content", {})

    async def evaluate_javascript(self, expression: str) -> Dict[str, Any]:
        """Evaluate a JS expression on the page."""
        return await self._call_mcp_tool("evaluate", {"expression": expression})

    async def wait_for_network_idle(self, timeout: int = 30000) -> Dict[str, Any]:
        """Wait until network is idle for a period or timeout."""
        return await self._call_mcp_tool("wait_for_network_idle", {"timeout": timeout})

    async def collect_console_logs(self) -> List[Dict[str, Any]]:
        """
        Collect console logs from the page.

        Returns:
            List of console log entries
        """
        response = await self._call_mcp_tool("get_console_logs", {})
        if response.get("success"):
            logs = response.get("logs", [])
            self._artifacts.console_logs.extend(logs)
            return logs

        return []

    async def collect_network_logs(self, page_id: str) -> List[Dict[str, Any]]:
        """
        Collect network logs from the page.

        Args:
            page_id: Page instance ID

        Returns:
            List of network log entries
        """
        response = await self._call_mcp_tool("get_network_logs", {"page_id": page_id})

        if response.get("success"):
            logs = response.get("logs", [])
            self._artifacts.network_logs.extend(logs)
            return logs

        return []

    async def start_tracing(
        self, page_id: str, trace_path: Optional[str] = None
    ) -> str:
        """
        Start tracing for the page.

        Args:
            page_id: Page instance ID
            trace_path: Optional path for trace file

        Returns:
            Path to the trace file
        """
        if trace_path is None:
            timestamp = int(time.time())
            trace_path = f"trace_{timestamp}.zip"

        response = await self._call_mcp_tool(
            "start_tracing", {"page_id": page_id, "path": trace_path}
        )

        if not response.get("success"):
            raise TestExecutionError("Failed to start tracing")

        return trace_path

    async def stop_tracing(self, page_id: str) -> Optional[str]:
        """
        Stop tracing and save the trace file.

        Args:
            page_id: Page instance ID

        Returns:
            Path to the saved trace file
        """
        response = await self._call_mcp_tool("stop_tracing", {"page_id": page_id})

        if response.get("success"):
            trace_path = response.get("trace_path")
            if trace_path:
                self._artifacts.trace_file = trace_path
            return trace_path

        return None

    async def close_page(self, page_id: str) -> None:
        """
        Close a page.

        Args:
            page_id: Page instance ID
        """
        response = await self._call_mcp_tool("close_page", {"page_id": page_id})

        if page_id == self._current_page:
            self._current_page = None

    async def close_browser(self) -> SimpleNamespace:
        """
        Close the active browser session.
        """
        response = await self._call_mcp_tool("close_browser", {})
        self._current_page = None
        return SimpleNamespace(**response)

    def get_current_artifacts(self) -> TestArtifacts:
        """
        Get current test artifacts.

        Returns:
            Current test artifacts
        """
        return self._artifacts

    def is_available(self) -> bool:
        """
        Check if Playwright MCP server is available.

        Returns:
            True if server is connected and available
        """
        return self.connection_manager.is_connected(self.server_name)

    async def execute_browser_action(
        self, action: BrowserAction
    ) -> BrowserActionResult:
        """
        Execute a browser action using a Pydantic action model.

        Args:
            action: Browser action to execute

        Returns:
            Browser action result
        """
        start_time = time.time()

        try:
            if not self._current_page:
                raise TestExecutionError("No active page for browser action")

            if action.action == "navigate":
                if not action.url:
                    raise ValueError("URL is required for navigate action")
                await self.navigate(self._current_page, action.url, **action.options)

            elif action.action == "click":
                if not action.selector:
                    raise ValueError("Selector is required for click action")
                await self.click(self._current_page, action.selector, **action.options)

            elif action.action == "fill":
                if not action.selector or action.value is None:
                    raise ValueError("Selector and value are required for fill action")
                await self.fill(
                    self._current_page, action.selector, action.value, **action.options
                )

            elif action.action == "screenshot":
                screenshot_path = await self.screenshot(
                    self._current_page, **action.options
                )
                return BrowserActionResult(
                    success=True,
                    action=action.action,
                    screenshot_path=screenshot_path,
                    duration=time.time() - start_time,
                )

            elif action.action == "get_text":
                if not action.selector:
                    raise ValueError("Selector is required for get_text action")
                text = await self.get_text(self._current_page, action.selector)
                return BrowserActionResult(
                    success=True,
                    action=action.action,
                    result_data={"text": text},
                    duration=time.time() - start_time,
                )

            else:
                raise ValueError(f"Unsupported action: {action.action}")

            return BrowserActionResult(
                success=True, action=action.action, duration=time.time() - start_time
            )

        except Exception as e:
            return BrowserActionResult(
                success=False,
                action=action.action,
                error=str(e),
                duration=time.time() - start_time,
            )

    async def execute_test_with_config(self, config: TestConfiguration) -> TestResult:
        """
        Execute a test using a Pydantic configuration model.

        Args:
            config: Test configuration

        Returns:
            Test execution result
        """
        mode = BrowserMode.HEADLESS if config.headless else BrowserMode.HEADED

        # Convert TestConfiguration to execute_test parameters
        options = {
            "timeout": config.timeout,
            "retries": config.retries,
            "trace": config.trace,
            "video": config.video,
            "screenshot": config.screenshot_mode,
        }

        return await self.execute_test(
            test_file=config.test_file,
            mode=mode,
            artifacts_dir=config.artifacts_dir,
            **options,
        )
