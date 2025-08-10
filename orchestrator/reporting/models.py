"""
Pydantic models for reporting and CI integration.

Data models for test reports, CI artifacts, and Git integration.
"""

from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, validator, ConfigDict


class ReportFormat(Enum):
    """Report output formats."""

    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    JUNIT = "junit"


class TestStatus(Enum):
    """Test execution status."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class CIEnvironment(Enum):
    """CI environment types."""

    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    AZURE_DEVOPS = "azure_devops"
    CIRCLECI = "circleci"
    TRAVIS_CI = "travis_ci"
    LOCAL = "local"
    UNKNOWN = "unknown"


class TestResultSummary(BaseModel):
    """Summary of test execution results."""

    model_config = ConfigDict(extra="forbid")

    total_tests: int = Field(..., ge=0, description="Total number of tests")
    passed: int = Field(..., ge=0, description="Number of passed tests")
    failed: int = Field(..., ge=0, description="Number of failed tests")
    skipped: int = Field(..., ge=0, description="Number of skipped tests")
    errors: int = Field(..., ge=0, description="Number of tests with errors")
    duration: float = Field(..., ge=0, description="Total execution time in seconds")
    success_rate: float = Field(
        ..., ge=0, le=100, description="Success rate percentage"
    )

    @validator("success_rate", pre=True, always=True)
    def calculate_success_rate(cls, v, values):
        """Calculate success rate from test counts."""
        if "total_tests" in values and values["total_tests"] > 0:
            passed = values.get("passed", 0)
            return (passed / values["total_tests"]) * 100
        return 0.0


class TestCaseResult(BaseModel):
    """Individual test case result."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Test case name")
    status: TestStatus = Field(..., description="Test execution status")
    duration: float = Field(..., ge=0, description="Test duration in seconds")
    file_path: str = Field(..., description="Test file path")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    stack_trace: Optional[str] = Field(None, description="Stack trace if failed")
    artifacts: List[str] = Field(default_factory=list, description="Test artifacts")
    tags: List[str] = Field(default_factory=list, description="Test tags")

    @validator("name")
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Test name cannot be empty")
        return v.strip()


class ArtifactInfo(BaseModel):
    """Information about test artifacts."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., description="Artifact file path")
    type: str = Field(..., description="Artifact type (screenshot, trace, video, etc.)")
    size: int = Field(..., ge=0, description="File size in bytes")
    created_at: datetime = Field(..., description="Creation timestamp")
    test_name: Optional[str] = Field(None, description="Associated test name")
    description: Optional[str] = Field(None, description="Artifact description")

    @validator("type")
    def validate_type(cls, v):
        valid_types = [
            "screenshot",
            "trace",
            "video",
            "log",
            "report",
            "coverage",
            "performance",
            "network",
            "console",
        ]
        if v not in valid_types:
            raise ValueError(f"Artifact type must be one of: {valid_types}")
        return v


class TestRunReport(BaseModel):
    """Complete test run report."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(..., description="Workflow correlation ID")
    run_id: str = Field(..., description="Test run identifier")
    started_at: datetime = Field(..., description="Test run start time")
    completed_at: datetime = Field(..., description="Test run completion time")
    environment: CIEnvironment = Field(..., description="Execution environment")
    summary: TestResultSummary = Field(..., description="Test results summary")
    test_cases: List[TestCaseResult] = Field(..., description="Individual test results")
    artifacts: List[ArtifactInfo] = Field(
        default_factory=list, description="Test artifacts"
    )
    configuration: Dict[str, Any] = Field(
        default_factory=dict, description="Test configuration"
    )
    git_info: Optional[Dict[str, Any]] = Field(
        None, description="Git repository information"
    )
    ci_info: Optional[Dict[str, Any]] = Field(
        None, description="CI environment information"
    )

    @validator("workflow_id")
    def validate_workflow_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Workflow ID cannot be empty")
        return v.strip()

    @validator("run_id")
    def validate_run_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Run ID cannot be empty")
        return v.strip()


class PRTemplate(BaseModel):
    """Pull request template configuration."""

    model_config = ConfigDict(extra="forbid")

    title_template: str = Field(..., description="PR title template")
    description_template: str = Field(..., description="PR description template")
    labels: List[str] = Field(default_factory=list, description="Default PR labels")
    reviewers: List[str] = Field(default_factory=list, description="Default reviewers")
    include_test_summary: bool = Field(True, description="Include test summary in PR")
    include_artifacts: bool = Field(True, description="Include artifact links in PR")
    include_coverage: bool = Field(False, description="Include coverage information")

    @validator("title_template")
    def validate_title_template(cls, v):
        if not v or not v.strip():
            raise ValueError("Title template cannot be empty")
        return v.strip()


class CIIntegrationConfig(BaseModel):
    """CI integration configuration."""

    model_config = ConfigDict(extra="forbid")

    environment: CIEnvironment = Field(..., description="CI environment type")
    artifact_upload: bool = Field(True, description="Upload artifacts to CI")
    report_format: ReportFormat = Field(ReportFormat.JSON, description="Report format")
    notification_webhook: Optional[str] = Field(
        None, description="Webhook for notifications"
    )
    retention_days: int = Field(
        30, ge=1, le=365, description="Artifact retention in days"
    )
    compress_artifacts: bool = Field(True, description="Compress artifacts for storage")

    @validator("notification_webhook")
    def validate_webhook(cls, v):
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("Webhook URL must start with http:// or https://")
        return v


class ApprovalRequest(BaseModel):
    """Manual approval request for Git operations."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., description="Unique request identifier")
    operation: str = Field(..., description="Git operation requiring approval")
    description: str = Field(..., description="Operation description")
    files_affected: List[str] = Field(
        default_factory=list, description="Files to be modified"
    )
    commit_message: Optional[str] = Field(None, description="Proposed commit message")
    pr_title: Optional[str] = Field(None, description="Proposed PR title")
    pr_description: Optional[str] = Field(None, description="Proposed PR description")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Request creation time"
    )
    expires_at: Optional[datetime] = Field(None, description="Request expiration time")
    approved: Optional[bool] = Field(None, description="Approval status")
    approved_by: Optional[str] = Field(None, description="Approver identifier")
    approved_at: Optional[datetime] = Field(None, description="Approval timestamp")
    rejection_reason: Optional[str] = Field(
        None, description="Rejection reason if denied"
    )

    @validator("request_id")
    def validate_request_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Request ID cannot be empty")
        return v.strip()

    @validator("operation")
    def validate_operation(cls, v):
        valid_operations = ["commit", "push", "create_pr", "merge"]
        if v not in valid_operations:
            raise ValueError(f"Operation must be one of: {valid_operations}")
        return v


class ReportGenerationRequest(BaseModel):
    """Request for generating test reports."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(..., description="Workflow correlation ID")
    format: ReportFormat = Field(ReportFormat.JSON, description="Report format")
    include_artifacts: bool = Field(True, description="Include artifact information")
    include_git_info: bool = Field(True, description="Include Git information")
    include_ci_info: bool = Field(True, description="Include CI information")
    output_path: Optional[str] = Field(None, description="Output file path")
    template_path: Optional[str] = Field(None, description="Custom template path")

    @validator("workflow_id")
    def validate_workflow_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Workflow ID cannot be empty")
        return v.strip()


class NotificationPayload(BaseModel):
    """Payload for CI notifications."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(..., description="Workflow correlation ID")
    status: str = Field(..., description="Overall status (success, failure, error)")
    summary: TestResultSummary = Field(..., description="Test results summary")
    report_url: Optional[str] = Field(None, description="URL to full report")
    artifacts_url: Optional[str] = Field(None, description="URL to artifacts")
    pr_url: Optional[str] = Field(None, description="URL to created PR")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Notification timestamp"
    )

    @validator("status")
    def validate_status(cls, v):
        valid_statuses = ["success", "failure", "error", "cancelled"]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return v
