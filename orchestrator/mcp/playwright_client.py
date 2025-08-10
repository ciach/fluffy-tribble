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
from enum import Enum

from pydantic import BaseModel, Field, validator, ConfigDict

from ..core.exceptions import TestExecutionError, MCPConnectionError
from .models import MCPToolCall, MCPToolResponse, BrowserAction, BrowserActionResult, TestConfiguration


class BrowserMode(Enum):
    """Browser execution mode."""

    HEADED = "headed"
    HEADLESS = "headless"


class TestArtifacts(BaseModel):
    """Container for test execution artifacts."""
    model_config = ConfigDict(extra='forbid')
    
    trace_file: Optional[str] = Field(None, description="Path to trace file")
    screenshots: List[str] = Field(default_factory=list, description="List of screenshot paths")
    console_logs: List[Dict[str, Any]] = Field(default_factory=list, description="Console log entries")
    network_logs: List[Dict[str, Any]] = Field(default_factory=list, description="Network log entries")
    video_file: Optional[str] = Field(None, description="Path to video file")


class TestResult(BaseModel):
    """Container for test execution results."""
    model_config = ConfigDict(extra='forbid')
    
    test_name: str = Field(..., description="Name of the test")
    status: str = Field(..., description="Test status: passed, failed, or skipped")
    duration: float = Field(..., ge=0, description="Test duration in seconds")
    artifacts: TestArtifacts = Field(..., description="Test artifacts")
    error_info: Optional[Dict[str, Any]] = Field(None, description="Error information if test failed")
    exit_code: Optional[int] = Field(None, description="Process exit code")

    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['passed', 'failed', 'skipped']
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return v

    @validator('test_name')
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

    def __init__(self, connection_manager, logger: Optional[logging.Logger] = None):
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
        self._artifacts_dir = Path("artifacts")

        # Test execution state
        self._current_test_name = None
        self._test_start_time = None
        self._artifacts = TestArtifacts()

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
            # TODO: Implement actual MCP tool call
            # This is a placeholder for the actual MCP client call
            self.logger.debug(f"Calling MCP tool: {tool_name} with args: {arguments}")

            # Simulate tool call delay
            await asyncio.sleep(0.1)

            # Mock response based on tool name
            if tool_name == "launch_browser":
                return {"success": True, "browser_id": "browser_123"}
            elif tool_name == "new_page":
                return {"success": True, "page_id": "page_456"}
            elif tool_name == "navigate":
                return {"success": True, "url": arguments.get("url")}
            elif tool_name == "screenshot":
                return {
                    "success": True,
                    "screenshot_path": f"screenshot_{int(time.time())}.png",
                }
            elif tool_name == "execute_test":
                return {
                    "success": True,
                    "exit_code": 0,
                    "duration": 2.5,
                    "artifacts": {
                        "trace": "trace.zip",
                        "screenshots": ["screenshot1.png", "screenshot2.png"],
                        "video": "test.webm",
                    },
                }
            else:
                return {"success": True}

        except Exception as e:
            self.logger.error(f"MCP tool call failed: {tool_name} - {e}")
            raise TestExecutionError(
                f"Playwright MCP tool call failed: {tool_name}",
                test_name=self._current_test_name,
            )

    async def launch_browser(
        self, mode: BrowserMode = BrowserMode.HEADLESS, **options
    ) -> str:
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

        response = await self._call_mcp_tool("launch_browser", launch_args)

        if not response.get("success"):
            raise TestExecutionError("Failed to launch browser")

        browser_id = response.get("browser_id")
        self.logger.debug(f"Browser launched with ID: {browser_id}")
        return browser_id

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
        self, page_id: str, url: str, wait_until: str = "networkidle"
    ) -> None:
        """
        Navigate to a URL.

        Args:
            page_id: Page instance ID
            url: URL to navigate to
            wait_until: When to consider navigation complete
        """
        self.logger.info(f"Navigating to: {url}")

        response = await self._call_mcp_tool(
            "navigate", {"page_id": page_id, "url": url, "wait_until": wait_until}
        )

        if not response.get("success"):
            raise TestExecutionError(f"Failed to navigate to {url}")

    async def click(self, page_id: str, selector: str, **options) -> None:
        """
        Click on an element.

        Args:
            page_id: Page instance ID
            selector: Element selector
            **options: Additional click options
        """
        self.logger.debug(f"Clicking element: {selector}")

        response = await self._call_mcp_tool(
            "click", {"page_id": page_id, "selector": selector, **options}
        )

        if not response.get("success"):
            raise TestExecutionError(f"Failed to click element: {selector}")

    async def fill(self, page_id: str, selector: str, value: str, **options) -> None:
        """
        Fill an input element.

        Args:
            page_id: Page instance ID
            selector: Input element selector
            value: Value to fill
            **options: Additional fill options
        """
        self.logger.debug(f"Filling element {selector} with value: {value}")

        response = await self._call_mcp_tool(
            "fill",
            {"page_id": page_id, "selector": selector, "value": value, **options},
        )

        if not response.get("success"):
            raise TestExecutionError(f"Failed to fill element: {selector}")

    async def screenshot(
        self, page_id: str, path: Optional[str] = None, **options
    ) -> str:
        """
        Take a screenshot of the page.

        Args:
            page_id: Page instance ID
            path: Optional path to save screenshot
            **options: Additional screenshot options

        Returns:
            Path to the saved screenshot
        """
        if path is None:
            timestamp = int(time.time())
            path = f"screenshot_{timestamp}.png"

        self.logger.debug(f"Taking screenshot: {path}")

        response = await self._call_mcp_tool(
            "screenshot", {"page_id": page_id, "path": path, **options}
        )

        if not response.get("success"):
            raise TestExecutionError("Failed to take screenshot")

        screenshot_path = response.get("screenshot_path", path)
        self._artifacts.screenshots.append(screenshot_path)
        return screenshot_path

    async def wait_for_selector(
        self, page_id: str, selector: str, timeout: int = 30000
    ) -> None:
        """
        Wait for an element to appear.

        Args:
            page_id: Page instance ID
            selector: Element selector to wait for
            timeout: Timeout in milliseconds
        """
        self.logger.debug(f"Waiting for selector: {selector}")

        response = await self._call_mcp_tool(
            "wait_for_selector",
            {"page_id": page_id, "selector": selector, "timeout": timeout},
        )

        if not response.get("success"):
            raise TestExecutionError(f"Timeout waiting for selector: {selector}")

    async def get_text(self, page_id: str, selector: str) -> str:
        """
        Get text content of an element.

        Args:
            page_id: Page instance ID
            selector: Element selector

        Returns:
            Text content of the element
        """
        response = await self._call_mcp_tool(
            "get_text", {"page_id": page_id, "selector": selector}
        )

        if not response.get("success"):
            raise TestExecutionError(f"Failed to get text from element: {selector}")

        return response.get("text", "")

    async def execute_test(
        self,
        test_file: str,
        mode: BrowserMode = BrowserMode.HEADLESS,
        artifacts_dir: Optional[str] = None,
        **options,
    ) -> TestResult:
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
            # Prepare test execution arguments
            exec_args = {
                "test_file": test_file,
                "headless": mode == BrowserMode.HEADLESS,
                "artifacts_dir": str(self._artifacts_dir),
                "trace": True,  # Always collect traces
                "video": True,  # Always collect videos
                "screenshot": "only-on-failure",
                **options,
            }

            response = await self._call_mcp_tool("execute_test", exec_args)

            # Calculate duration
            duration = time.time() - self._test_start_time

            # Process artifacts from response
            artifacts_data = response.get("artifacts", {})
            self._artifacts.trace_file = artifacts_data.get("trace")
            self._artifacts.video_file = artifacts_data.get("video")

            if "screenshots" in artifacts_data:
                self._artifacts.screenshots.extend(artifacts_data["screenshots"])

            # Determine test status
            exit_code = response.get("exit_code", 0)
            status = "passed" if exit_code == 0 else "failed"

            # Create test result
            result = TestResult(
                test_name=test_name,
                status=status,
                duration=duration,
                artifacts=self._artifacts,
                exit_code=exit_code,
            )

            if status == "failed":
                result.error_info = response.get("error_info", {})
                self.logger.warning(f"Test failed: {test_name}")
            else:
                self.logger.info(f"Test passed: {test_name}")

            return result

        except Exception as e:
            duration = time.time() - self._test_start_time
            self.logger.error(f"Test execution failed: {test_name} - {e}")

            return TestResult(
                test_name=test_name,
                status="failed",
                duration=duration,
                artifacts=self._artifacts,
                error_info={"error": str(e), "type": type(e).__name__},
            )

        finally:
            self._current_test_name = None
            self._test_start_time = None

    async def collect_console_logs(self, page_id: str) -> List[Dict[str, Any]]:
        """
        Collect console logs from the page.

        Args:
            page_id: Page instance ID

        Returns:
            List of console log entries
        """
        response = await self._call_mcp_tool("get_console_logs", {"page_id": page_id})

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

    async def close_browser(self, browser_id: str) -> None:
        """
        Close a browser instance.

        Args:
            browser_id: Browser instance ID
        """
        response = await self._call_mcp_tool(
            "close_browser", {"browser_id": browser_id}
        )
        self._current_page = None

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
    async def execute_browser_action(self, action: BrowserAction) -> BrowserActionResult:
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
                await self.fill(self._current_page, action.selector, action.value, **action.options)
                
            elif action.action == "screenshot":
                screenshot_path = await self.screenshot(self._current_page, **action.options)
                return BrowserActionResult(
                    success=True,
                    action=action.action,
                    screenshot_path=screenshot_path,
                    duration=time.time() - start_time
                )
                
            elif action.action == "get_text":
                if not action.selector:
                    raise ValueError("Selector is required for get_text action")
                text = await self.get_text(self._current_page, action.selector)
                return BrowserActionResult(
                    success=True,
                    action=action.action,
                    result_data={"text": text},
                    duration=time.time() - start_time
                )
                
            else:
                raise ValueError(f"Unsupported action: {action.action}")
            
            return BrowserActionResult(
                success=True,
                action=action.action,
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return BrowserActionResult(
                success=False,
                action=action.action,
                error=str(e),
                duration=time.time() - start_time
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
            **options
        )