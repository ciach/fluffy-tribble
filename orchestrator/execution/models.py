"""
Data models for test execution and artifact management.

Defines Pydantic models for test execution configuration, results,
and artifact metadata.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, ConfigDict


class ExecutionMode(Enum):
    """Test execution mode."""

    HEADED = "headed"
    HEADLESS = "headless"


class TestStatus(Enum):
    """Test execution status."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    ERROR = "error"


class ArtifactType(Enum):
    """Types of test artifacts."""

    TRACE = "trace"
    SCREENSHOT = "screenshot"
    VIDEO = "video"
    CONSOLE_LOG = "console_log"
    NETWORK_LOG = "network_log"
    HTML_REPORT = "html_report"


class ExecutionConfig(BaseModel):
    """Configuration for test execution."""

    model_config = ConfigDict(extra="forbid")

    # Test file configuration
    test_file: str = Field(..., description="Path to test file to execute")
    test_name: Optional[str] = Field(None, description="Override test name")

    # Execution mode
    mode: ExecutionMode = Field(
        default_factory=lambda: (
            ExecutionMode.HEADLESS
            if os.getenv("CI", "").lower() == "true"
            else ExecutionMode.HEADED
        ),
        description="Test execution mode",
    )

    # Override for headless mode
    force_headless: bool = Field(
        default_factory=lambda: os.getenv("QA_OPERATOR_HEADLESS", "").lower() == "true",
        description="Force headless mode regardless of CI detection",
    )

    # Execution options
    timeout: int = Field(30000, ge=1000, description="Test timeout in milliseconds")
    retries: int = Field(0, ge=0, le=3, description="Number of retries on failure")
    workers: int = Field(1, ge=1, le=10, description="Number of parallel workers")

    # Artifact collection
    collect_traces: bool = Field(True, description="Collect Playwright traces")
    collect_videos: bool = Field(True, description="Collect test videos")
    collect_screenshots: bool = Field(True, description="Collect screenshots")
    screenshot_mode: str = Field(
        "only-on-failure", description="Screenshot collection mode"
    )

    # Output directories
    artifacts_dir: Optional[str] = Field(None, description="Custom artifacts directory")
    reports_dir: Optional[str] = Field(None, description="Custom reports directory")

    # Environment detection
    ci_mode: bool = Field(
        default_factory=lambda: os.getenv("CI", "").lower() == "true",
        description="CI environment detection",
    )

    @validator("test_file")
    def validate_test_file(cls, v):
        """Validate test file exists."""
        if not v:
            raise ValueError("Test file path cannot be empty")

        test_path = Path(v)
        if not test_path.exists():
            raise ValueError(f"Test file does not exist: {v}")

        if not test_path.suffix in [".js", ".ts"]:
            raise ValueError(f"Test file must be JavaScript or TypeScript: {v}")

        return str(test_path)

    @validator("screenshot_mode")
    def validate_screenshot_mode(cls, v):
        """Validate screenshot mode."""
        valid_modes = ["on", "off", "only-on-failure"]
        if v not in valid_modes:
            raise ValueError(f"Screenshot mode must be one of: {valid_modes}")
        return v

    @property
    def effective_mode(self) -> ExecutionMode:
        """Get the effective execution mode considering all overrides."""
        if self.force_headless:
            return ExecutionMode.HEADLESS
        return self.mode

    @property
    def is_headless(self) -> bool:
        """Check if execution should be headless."""
        return self.effective_mode == ExecutionMode.HEADLESS

    def get_test_name(self) -> str:
        """Get the effective test name."""
        if self.test_name:
            return self.test_name
        return Path(self.test_file).stem


class ArtifactMetadata(BaseModel):
    """Metadata for test artifacts."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: ArtifactType = Field(..., description="Type of artifact")
    file_path: str = Field(..., description="Path to artifact file")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    test_name: str = Field(..., description="Associated test name")
    workflow_id: str = Field(..., description="Workflow ID for correlation")

    # Optional metadata
    description: Optional[str] = Field(None, description="Artifact description")
    mime_type: Optional[str] = Field(None, description="MIME type of the file")
    checksum: Optional[str] = Field(None, description="File checksum for integrity")

    @validator("file_path")
    def validate_file_path(cls, v):
        """Validate file path exists."""
        if not Path(v).exists():
            raise ValueError(f"Artifact file does not exist: {v}")
        return v

    @property
    def file_name(self) -> str:
        """Get the file name from path."""
        return Path(self.file_path).name

    @property
    def relative_path(self) -> str:
        """Get relative path from project root."""
        try:
            return str(Path(self.file_path).relative_to(Path.cwd()))
        except ValueError:
            return self.file_path


class ExecutionResult(BaseModel):
    """Result of test execution."""

    model_config = ConfigDict(extra="forbid")

    # Test identification
    test_name: str = Field(..., description="Name of the executed test")
    test_file: str = Field(..., description="Path to test file")
    workflow_id: str = Field(..., description="Workflow ID for correlation")

    # Execution details
    status: TestStatus = Field(..., description="Test execution status")
    duration: float = Field(..., ge=0, description="Execution duration in seconds")
    started_at: datetime = Field(..., description="Test start time")
    completed_at: datetime = Field(..., description="Test completion time")

    # Results
    exit_code: int = Field(..., description="Process exit code")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")

    # Artifacts
    artifacts: List[ArtifactMetadata] = Field(
        default_factory=list, description="List of collected artifacts"
    )

    # Error information
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_type: Optional[str] = Field(None, description="Error type classification")
    stack_trace: Optional[str] = Field(None, description="Stack trace if available")

    # Execution context
    execution_config: ExecutionConfig = Field(
        ..., description="Execution configuration used"
    )
    environment: Dict[str, Any] = Field(
        default_factory=dict, description="Environment information"
    )

    @validator("status")
    def validate_status_exit_code(cls, v, values):
        """Validate status matches exit code."""
        exit_code = values.get("exit_code", 0)

        if v == TestStatus.PASSED and exit_code != 0:
            raise ValueError("Passed tests must have exit code 0")
        elif (
            v in [TestStatus.FAILED, TestStatus.ERROR, TestStatus.TIMEOUT]
            and exit_code == 0
        ):
            raise ValueError("Failed tests must have non-zero exit code")

        return v

    @property
    def is_success(self) -> bool:
        """Check if test execution was successful."""
        return self.status == TestStatus.PASSED

    @property
    def has_artifacts(self) -> bool:
        """Check if execution has collected artifacts."""
        return len(self.artifacts) > 0

    def get_artifacts_by_type(
        self, artifact_type: ArtifactType
    ) -> List[ArtifactMetadata]:
        """Get artifacts of a specific type."""
        return [a for a in self.artifacts if a.artifact_type == artifact_type]

    def get_trace_file(self) -> Optional[str]:
        """Get the trace file path if available."""
        traces = self.get_artifacts_by_type(ArtifactType.TRACE)
        return traces[0].file_path if traces else None

    def get_video_file(self) -> Optional[str]:
        """Get the video file path if available."""
        videos = self.get_artifacts_by_type(ArtifactType.VIDEO)
        return videos[0].file_path if videos else None

    def get_screenshots(self) -> List[str]:
        """Get all screenshot file paths."""
        screenshots = self.get_artifacts_by_type(ArtifactType.SCREENSHOT)
        return [s.file_path for s in screenshots]

    def to_summary(self) -> Dict[str, Any]:
        """Create a summary dictionary for logging."""
        return {
            "test_name": self.test_name,
            "status": self.status.value,
            "duration": self.duration,
            "exit_code": self.exit_code,
            "artifacts_count": len(self.artifacts),
            "has_error": bool(self.error_message),
            "workflow_id": self.workflow_id,
        }


class TestExecutionError(Exception):
    """Exception raised during test execution."""

    def __init__(
        self,
        message: str,
        test_name: Optional[str] = None,
        exit_code: Optional[int] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ):
        super().__init__(message)
        self.test_name = test_name
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error": str(self),
            "test_name": self.test_name,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }
