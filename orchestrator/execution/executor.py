"""
Test executor with environment detection and artifact collection.

Provides comprehensive test execution capabilities with CI environment detection,
command-line flag support, and structured artifact collection.
"""

import asyncio
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from ..core.config import Config
from ..core.logging_config import get_logger, log_performance
from ..mcp.playwright_client import PlaywrightMCPClient, BrowserMode
from .models import (
    ExecutionConfig,
    ExecutionResult,
    ExecutionMode,
    TestStatus,
    TestExecutionError,
    ArtifactMetadata,
    ArtifactType,
)
from .artifacts import ArtifactManager


class TestExecutor:
    """
    Test executor with environment detection and comprehensive artifact collection.
    
    Handles test execution with CI environment detection, command-line flag support,
    and structured artifact collection during test runs.
    """
    
    def __init__(
        self,
        config: Config,
        playwright_client: PlaywrightMCPClient,
        artifact_manager: Optional['ArtifactManager'] = None,
        workflow_id: Optional[str] = None,
    ):
        """
        Initialize the test executor.
        
        Args:
            config: QA Operator configuration
            playwright_client: Playwright MCP client instance
            artifact_manager: Optional artifact manager instance
            workflow_id: Unique workflow identifier
        """
        self.config = config
        self.playwright_client = playwright_client
        self.workflow_id = workflow_id or f"exec_{int(time.time())}"
        self.logger = get_logger(__name__, workflow_id=self.workflow_id)
        
        # Initialize artifact manager if not provided
        if artifact_manager is None:
            from .artifacts import ArtifactManager
            self.artifact_manager = ArtifactManager(config, self.workflow_id)
        else:
            self.artifact_manager = artifact_manager
        
        # Execution state
        self._current_execution: Optional[ExecutionResult] = None
        self._execution_start_time: Optional[float] = None
    
    def detect_execution_mode(self, force_headless: Optional[bool] = None) -> ExecutionMode:
        """
        Detect the appropriate execution mode based on environment and flags.
        
        Args:
            force_headless: Optional override for headless mode
            
        Returns:
            Detected execution mode
        """
        from .models import ExecutionMode
        
        # Check for explicit override
        if force_headless is True:
            self.logger.info("Forcing headless mode via parameter")
            return ExecutionMode.HEADLESS
        
        # Check environment variable override
        if os.getenv("QA_OPERATOR_HEADLESS", "").lower() == "true":
            self.logger.info("Forcing headless mode via QA_OPERATOR_HEADLESS environment variable")
            return ExecutionMode.HEADLESS
        
        # Check CI environment
        if self.config.is_ci_mode:
            self.logger.info("Detected CI environment, using headless mode")
            return ExecutionMode.HEADLESS
        
        # Default to headed mode for development
        self.logger.info("Using headed mode for development environment")
        return ExecutionMode.HEADED
    
    def parse_command_line_flags(self, args: List[str]) -> Dict[str, Any]:
        """
        Parse command-line flags for test execution overrides.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Dictionary of parsed flags
        """
        flags = {
            "force_headless": False,
            "timeout": None,
            "retries": None,
            "workers": None,
            "artifacts_dir": None,
        }
        
        i = 0
        while i < len(args):
            arg = args[i]
            
            if arg == "--headless":
                flags["force_headless"] = True
                self.logger.debug("Command-line flag: --headless detected")
            
            elif arg == "--timeout" and i + 1 < len(args):
                try:
                    flags["timeout"] = int(args[i + 1])
                    i += 1  # Skip next argument
                    self.logger.debug(f"Command-line flag: --timeout {flags['timeout']}")
                except ValueError:
                    self.logger.warning(f"Invalid timeout value: {args[i + 1]}")
            
            elif arg == "--retries" and i + 1 < len(args):
                try:
                    flags["retries"] = int(args[i + 1])
                    i += 1
                    self.logger.debug(f"Command-line flag: --retries {flags['retries']}")
                except ValueError:
                    self.logger.warning(f"Invalid retries value: {args[i + 1]}")
            
            elif arg == "--workers" and i + 1 < len(args):
                try:
                    flags["workers"] = int(args[i + 1])
                    i += 1
                    self.logger.debug(f"Command-line flag: --workers {flags['workers']}")
                except ValueError:
                    self.logger.warning(f"Invalid workers value: {args[i + 1]}")
            
            elif arg == "--artifacts-dir" and i + 1 < len(args):
                flags["artifacts_dir"] = args[i + 1]
                i += 1
                self.logger.debug(f"Command-line flag: --artifacts-dir {flags['artifacts_dir']}")
            
            i += 1
        
        return flags
    
    async def execute_test(
        self,
        test_file: Union[str, Path],
        config: Optional[ExecutionConfig] = None,
        command_line_args: Optional[List[str]] = None,
    ) -> ExecutionResult:
        """
        Execute a single test file with comprehensive artifact collection.
        
        Args:
            test_file: Path to the test file to execute
            config: Optional execution configuration
            command_line_args: Optional command-line arguments for overrides
            
        Returns:
            Test execution result with artifacts
        """
        test_file = Path(test_file)
        test_name = test_file.stem
        
        # Parse command-line flags if provided
        cli_flags = {}
        if command_line_args:
            cli_flags = self.parse_command_line_flags(command_line_args)
        
        # Create execution configuration
        if config is None:
            config = ExecutionConfig(
                test_file=str(test_file),
                test_name=test_name,
                force_headless=cli_flags.get("force_headless", False),
                timeout=cli_flags.get("timeout", 30000),
                retries=cli_flags.get("retries", 0),
                workers=cli_flags.get("workers", 1),
                artifacts_dir=cli_flags.get("artifacts_dir"),
            )
        
        # Override config with command-line flags
        if cli_flags.get("force_headless"):
            config.force_headless = True
        if cli_flags.get("timeout"):
            config.timeout = cli_flags["timeout"]
        if cli_flags.get("retries"):
            config.retries = cli_flags["retries"]
        if cli_flags.get("workers"):
            config.workers = cli_flags["workers"]
        if cli_flags.get("artifacts_dir"):
            config.artifacts_dir = cli_flags["artifacts_dir"]
        
        self.logger.info(
            f"Starting test execution: {test_name}",
            extra={
                "metadata": {
                    "test_file": str(test_file),
                    "mode": config.effective_mode.value,
                    "timeout": config.timeout,
                    "retries": config.retries,
                    "workflow_id": self.workflow_id,
                }
            }
        )
        
        start_time = time.time()
        started_at = datetime.utcnow()
        
        try:
            # Prepare artifacts directory
            artifacts_dir = await self.artifact_manager.prepare_test_artifacts_dir(test_name)
            
            # Execute the test
            result = await self._execute_test_with_playwright(
                config, artifacts_dir, started_at
            )
            
            # Log performance
            log_performance(
                self.logger,
                f"test_execution_{test_name}",
                result.duration,
                test_name=test_name,
                status=result.status.value,
                artifacts_count=len(result.artifacts),
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            completed_at = datetime.utcnow()
            
            self.logger.error(
                f"Test execution failed: {test_name} - {e}",
                extra={
                    "metadata": {
                        "test_name": test_name,
                        "duration": duration,
                        "error": str(e),
                        "workflow_id": self.workflow_id,
                    }
                }
            )
            
            # Create error result
            return ExecutionResult(
                test_name=test_name,
                test_file=str(test_file),
                workflow_id=self.workflow_id,
                status=TestStatus.ERROR,
                duration=duration,
                started_at=started_at,
                completed_at=completed_at,
                exit_code=-1,
                error_message=str(e),
                error_type=type(e).__name__,
                execution_config=config,
                environment=self._get_environment_info(),
            )
    
    async def _execute_test_with_playwright(
        self,
        config: ExecutionConfig,
        artifacts_dir: Path,
        started_at: datetime,
    ) -> ExecutionResult:
        """
        Execute test using Playwright MCP client.
        
        Args:
            config: Execution configuration
            artifacts_dir: Directory for artifacts
            started_at: Test start timestamp
            
        Returns:
            Test execution result
        """
        test_name = config.get_test_name()
        
        # Check if Playwright MCP is available
        if not self.playwright_client.is_available():
            raise TestExecutionError(
                "Playwright MCP client is not available",
                test_name=test_name
            )
        
        # Determine browser mode
        browser_mode = BrowserMode.HEADLESS if config.is_headless else BrowserMode.HEADED
        
        try:
            # Execute test using Playwright MCP
            playwright_result = await self.playwright_client.execute_test(
                test_file=config.test_file,
                mode=browser_mode,
                artifacts_dir=str(artifacts_dir),
                timeout=config.timeout,
                retries=config.retries,
                trace=config.collect_traces,
                video=config.collect_videos,
                screenshot=config.screenshot_mode,
            )
            
            completed_at = datetime.utcnow()
            
            # Collect artifacts from Playwright result
            artifacts = await self._collect_artifacts_from_playwright_result(
                playwright_result, artifacts_dir, test_name
            )
            
            # Determine status from Playwright result
            if playwright_result.status == "passed":
                status = TestStatus.PASSED
            elif playwright_result.status == "failed":
                status = TestStatus.FAILED
            elif playwright_result.status == "skipped":
                status = TestStatus.SKIPPED
            else:
                status = TestStatus.ERROR
            
            # Create execution result
            result = ExecutionResult(
                test_name=test_name,
                test_file=config.test_file,
                workflow_id=self.workflow_id,
                status=status,
                duration=playwright_result.duration,
                started_at=started_at,
                completed_at=completed_at,
                exit_code=playwright_result.exit_code or 0,
                artifacts=artifacts,
                execution_config=config,
                environment=self._get_environment_info(),
            )
            
            # Add error information if test failed
            if status != TestStatus.PASSED and playwright_result.error_info:
                result.error_message = playwright_result.error_info.get("error", "Unknown error")
                result.error_type = playwright_result.error_info.get("type", "UnknownError")
                result.stack_trace = playwright_result.error_info.get("stack_trace")
            
            return result
            
        except Exception as e:
            completed_at = datetime.utcnow()
            duration = (completed_at - started_at).total_seconds()
            
            raise TestExecutionError(
                f"Playwright test execution failed: {e}",
                test_name=test_name,
                exit_code=-1,
            )
    
    async def _collect_artifacts_from_playwright_result(
        self,
        playwright_result,
        artifacts_dir: Path,
        test_name: str,
    ) -> List[ArtifactMetadata]:
        """
        Collect and catalog artifacts from Playwright test result.
        
        Args:
            playwright_result: Result from Playwright MCP client
            artifacts_dir: Directory containing artifacts
            test_name: Name of the test
            
        Returns:
            List of artifact metadata
        """
        artifacts = []
        
        # Collect trace file
        if playwright_result.artifacts.trace_file:
            trace_path = artifacts_dir / playwright_result.artifacts.trace_file
            if trace_path.exists():
                artifacts.append(
                    ArtifactMetadata(
                        artifact_type=ArtifactType.TRACE,
                        file_path=str(trace_path),
                        file_size=trace_path.stat().st_size,
                        test_name=test_name,
                        workflow_id=self.workflow_id,
                        description="Playwright trace file",
                        mime_type="application/zip",
                    )
                )
        
        # Collect video file
        if playwright_result.artifacts.video_file:
            video_path = artifacts_dir / playwright_result.artifacts.video_file
            if video_path.exists():
                artifacts.append(
                    ArtifactMetadata(
                        artifact_type=ArtifactType.VIDEO,
                        file_path=str(video_path),
                        file_size=video_path.stat().st_size,
                        test_name=test_name,
                        workflow_id=self.workflow_id,
                        description="Test execution video",
                        mime_type="video/webm",
                    )
                )
        
        # Collect screenshots
        for screenshot_file in playwright_result.artifacts.screenshots:
            screenshot_path = artifacts_dir / screenshot_file
            if screenshot_path.exists():
                artifacts.append(
                    ArtifactMetadata(
                        artifact_type=ArtifactType.SCREENSHOT,
                        file_path=str(screenshot_path),
                        file_size=screenshot_path.stat().st_size,
                        test_name=test_name,
                        workflow_id=self.workflow_id,
                        description="Test screenshot",
                        mime_type="image/png",
                    )
                )
        
        # Save console logs as artifact
        if playwright_result.artifacts.console_logs:
            console_log_path = artifacts_dir / "console.log"
            with open(console_log_path, "w", encoding="utf-8") as f:
                for log_entry in playwright_result.artifacts.console_logs:
                    f.write(json.dumps(log_entry) + "\n")
            
            artifacts.append(
                ArtifactMetadata(
                    artifact_type=ArtifactType.CONSOLE_LOG,
                    file_path=str(console_log_path),
                    file_size=console_log_path.stat().st_size,
                    test_name=test_name,
                    workflow_id=self.workflow_id,
                    description="Browser console logs",
                    mime_type="application/json",
                )
            )
        
        # Save network logs as artifact
        if playwright_result.artifacts.network_logs:
            network_log_path = artifacts_dir / "network.log"
            with open(network_log_path, "w", encoding="utf-8") as f:
                for log_entry in playwright_result.artifacts.network_logs:
                    f.write(json.dumps(log_entry) + "\n")
            
            artifacts.append(
                ArtifactMetadata(
                    artifact_type=ArtifactType.NETWORK_LOG,
                    file_path=str(network_log_path),
                    file_size=network_log_path.stat().st_size,
                    test_name=test_name,
                    workflow_id=self.workflow_id,
                    description="Network request logs",
                    mime_type="application/json",
                )
            )
        
        self.logger.debug(
            f"Collected {len(artifacts)} artifacts for test: {test_name}",
            extra={
                "metadata": {
                    "test_name": test_name,
                    "artifacts_count": len(artifacts),
                    "artifact_types": [a.artifact_type.value for a in artifacts],
                }
            }
        )
        
        return artifacts
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """
        Get environment information for execution context.
        
        Returns:
            Dictionary of environment information
        """
        return {
            "ci_mode": self.config.is_ci_mode,
            "headless_mode": self.config.is_headless,
            "python_version": os.sys.version,
            "platform": os.name,
            "working_directory": str(Path.cwd()),
            "environment_variables": {
                "CI": os.getenv("CI", ""),
                "QA_OPERATOR_HEADLESS": os.getenv("QA_OPERATOR_HEADLESS", ""),
                "QA_OPERATOR_LOG_LEVEL": os.getenv("QA_OPERATOR_LOG_LEVEL", ""),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    async def execute_multiple_tests(
        self,
        test_files: List[Union[str, Path]],
        config: Optional[ExecutionConfig] = None,
        command_line_args: Optional[List[str]] = None,
    ) -> List[ExecutionResult]:
        """
        Execute multiple test files.
        
        Args:
            test_files: List of test files to execute
            config: Optional base execution configuration
            command_line_args: Optional command-line arguments
            
        Returns:
            List of execution results
        """
        self.logger.info(
            f"Starting execution of {len(test_files)} tests",
            extra={
                "metadata": {
                    "test_count": len(test_files),
                    "workflow_id": self.workflow_id,
                }
            }
        )
        
        results = []
        
        for test_file in test_files:
            try:
                result = await self.execute_test(test_file, config, command_line_args)
                results.append(result)
                
                self.logger.info(
                    f"Test completed: {result.test_name} - {result.status.value}",
                    extra={
                        "metadata": {
                            "test_name": result.test_name,
                            "status": result.status.value,
                            "duration": result.duration,
                        }
                    }
                )
                
            except Exception as e:
                self.logger.error(
                    f"Failed to execute test: {test_file} - {e}",
                    extra={
                        "metadata": {
                            "test_file": str(test_file),
                            "error": str(e),
                        }
                    }
                )
        
        # Log summary
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)
        
        self.logger.info(
            f"Test execution completed: {passed} passed, {failed} failed, {errors} errors",
            extra={
                "metadata": {
                    "total_tests": len(results),
                    "passed": passed,
                    "failed": failed,
                    "errors": errors,
                    "workflow_id": self.workflow_id,
                }
            }
        )
        
        return results
    
    def parse_test_results(self, result: ExecutionResult) -> Dict[str, Any]:
        """
        Parse test results into structured output format.
        
        Args:
            result: Test execution result
            
        Returns:
            Structured test result data
        """
        return {
            "test": {
                "name": result.test_name,
                "file": result.test_file,
                "status": result.status.value,
                "duration": result.duration,
                "started_at": result.started_at.isoformat(),
                "completed_at": result.completed_at.isoformat(),
            },
            "execution": {
                "exit_code": result.exit_code,
                "mode": result.execution_config.effective_mode.value,
                "timeout": result.execution_config.timeout,
                "retries": result.execution_config.retries,
                "workflow_id": result.workflow_id,
            },
            "artifacts": [
                {
                    "type": artifact.artifact_type.value,
                    "path": artifact.relative_path,
                    "size": artifact.file_size,
                    "created_at": artifact.created_at.isoformat(),
                }
                for artifact in result.artifacts
            ],
            "error": {
                "message": result.error_message,
                "type": result.error_type,
                "stack_trace": result.stack_trace,
            } if result.error_message else None,
            "environment": result.environment,
        }