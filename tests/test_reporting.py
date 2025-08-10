"""
Unit tests for reporting and CI integration.

Tests report generation, CI integration, approval workflow,
and artifact management functionality.
"""

import pytest
import asyncio
import json
import tempfile
from unittest.mock import Mock, AsyncMock, patch, mock_open
from pathlib import Path
from datetime import datetime, timedelta

from orchestrator.reporting.generator import ReportGenerator
from orchestrator.reporting.ci_integration import (
    ApprovalManager,
    CIIntegrator,
    GitOperationApprovalWorkflow,
)
from orchestrator.reporting.models import (
    TestRunReport,
    TestResultSummary,
    TestCaseResult,
    ArtifactInfo,
    ReportFormat,
    CIEnvironment,
    TestStatus,
    ReportGenerationRequest,
    CIIntegrationConfig,
    PRTemplate,
    ApprovalRequest,
    NotificationPayload,
)


class TestReportGenerator:
    """Test cases for ReportGenerator."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def report_generator(self, temp_dir):
        """Create a ReportGenerator instance for testing."""
        return ReportGenerator(
            output_dir=temp_dir / "reports",
            template_dir=temp_dir / "templates"
        )

    @pytest.fixture
    def sample_test_results(self):
        """Create sample test results."""
        return [
            TestCaseResult(
                name="test_login_success",
                status=TestStatus.PASSED,
                duration=2.5,
                file_path="tests/test_auth.py",
                tags=["auth", "login"]
            ),
            TestCaseResult(
                name="test_login_failure",
                status=TestStatus.FAILED,
                duration=1.8,
                file_path="tests/test_auth.py",
                error_message="Login failed with invalid credentials",
                stack_trace="AssertionError: Expected success but got failure",
                tags=["auth", "login"]
            ),
            TestCaseResult(
                name="test_user_profile",
                status=TestStatus.SKIPPED,
                duration=0.0,
                file_path="tests/test_user.py",
                tags=["user", "profile"]
            )
        ]

    @pytest.fixture
    def sample_artifacts(self, temp_dir):
        """Create sample artifacts."""
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True)
        
        # Create sample artifact files
        (artifacts_dir / "screenshot.png").write_bytes(b"fake image data")
        (artifacts_dir / "trace.zip").write_bytes(b"fake trace data")
        (artifacts_dir / "test.log").write_text("fake log content")
        
        return artifacts_dir

    def test_ci_environment_detection(self, report_generator):
        """Test CI environment detection."""
        # Should detect local environment by default
        assert report_generator.ci_environment == CIEnvironment.LOCAL

    @patch.dict('os.environ', {'GITHUB_ACTIONS': 'true'})
    def test_github_actions_detection(self, temp_dir):
        """Test GitHub Actions environment detection."""
        generator = ReportGenerator(temp_dir)
        assert generator.ci_environment == CIEnvironment.GITHUB_ACTIONS

    @patch.dict('os.environ', {'GITLAB_CI': 'true'})
    def test_gitlab_ci_detection(self, temp_dir):
        """Test GitLab CI environment detection."""
        generator = ReportGenerator(temp_dir)
        assert generator.ci_environment == CIEnvironment.GITLAB_CI

    def test_collect_artifacts(self, report_generator, sample_artifacts):
        """Test artifact collection from directory."""
        artifacts = report_generator._collect_artifacts(sample_artifacts)
        
        assert len(artifacts) == 3
        
        # Check artifact types
        artifact_types = {artifact.type for artifact in artifacts}
        assert "screenshot" in artifact_types
        assert "trace" in artifact_types
        assert "log" in artifact_types

    def test_generate_report(self, report_generator, sample_test_results, sample_artifacts):
        """Test comprehensive report generation."""
        request = ReportGenerationRequest(
            workflow_id="test-workflow-123",
            format=ReportFormat.JSON,
            include_artifacts=True,
            include_git_info=False,
            include_ci_info=True
        )
        
        report = report_generator.generate_report(
            request=request,
            test_results=sample_test_results,
            artifacts_dir=sample_artifacts
        )
        
        assert isinstance(report, TestRunReport)
        assert report.workflow_id == "test-workflow-123"
        assert report.summary.total_tests == 3
        assert report.summary.passed == 1
        assert report.summary.failed == 1
        assert report.summary.skipped == 1
        assert report.summary.success_rate == pytest.approx(33.33, rel=1e-2)
        assert len(report.test_cases) == 3
        assert len(report.artifacts) == 3

    def test_save_json_report(self, report_generator, sample_test_results, temp_dir):
        """Test JSON report saving."""
        request = ReportGenerationRequest(
            workflow_id="test-workflow-123",
            format=ReportFormat.JSON,
            output_path=str(temp_dir / "test_report.json")
        )
        
        report = report_generator.generate_report(
            request=request,
            test_results=sample_test_results
        )
        
        # Check that file was created
        output_path = Path(request.output_path)
        assert output_path.exists()
        
        # Verify content
        with open(output_path) as f:
            data = json.load(f)
        
        assert data["workflow_id"] == "test-workflow-123"
        assert data["summary"]["total_tests"] == 3

    def test_save_html_report(self, report_generator, sample_test_results, temp_dir):
        """Test HTML report saving."""
        request = ReportGenerationRequest(
            workflow_id="test-workflow-123",
            format=ReportFormat.HTML,
            output_path=str(temp_dir / "test_report.html")
        )
        
        report = report_generator.generate_report(
            request=request,
            test_results=sample_test_results
        )
        
        # Check that file was created
        output_path = Path(request.output_path)
        assert output_path.exists()
        
        # Verify it's HTML content
        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "test-workflow-123" in content

    def test_save_markdown_report(self, report_generator, sample_test_results, temp_dir):
        """Test Markdown report saving."""
        request = ReportGenerationRequest(
            workflow_id="test-workflow-123",
            format=ReportFormat.MARKDOWN,
            output_path=str(temp_dir / "test_report.md")
        )
        
        report = report_generator.generate_report(
            request=request,
            test_results=sample_test_results
        )
        
        # Check that file was created
        output_path = Path(request.output_path)
        assert output_path.exists()
        
        # Verify it's Markdown content
        content = output_path.read_text()
        assert "# Test Report" in content
        assert "test-workflow-123" in content

    def test_create_ci_artifacts_package(self, report_generator, sample_test_results, sample_artifacts):
        """Test CI artifacts package creation."""
        request = ReportGenerationRequest(
            workflow_id="test-workflow-123",
            format=ReportFormat.JSON
        )
        
        report = report_generator.generate_report(
            request=request,
            test_results=sample_test_results
        )
        
        package_path = report_generator.create_ci_artifacts_package(
            report=report,
            artifacts_dir=sample_artifacts
        )
        
        assert package_path.exists()
        assert package_path.suffix == ".zip"
        assert package_path.stat().st_size > 0

    def test_generate_notification_payload(self, report_generator, sample_test_results):
        """Test notification payload generation."""
        request = ReportGenerationRequest(
            workflow_id="test-workflow-123",
            format=ReportFormat.JSON
        )
        
        report = report_generator.generate_report(
            request=request,
            test_results=sample_test_results
        )
        
        payload = report_generator.generate_notification_payload(
            report=report,
            report_url="https://example.com/report",
            pr_url="https://github.com/repo/pull/123"
        )
        
        assert isinstance(payload, NotificationPayload)
        assert payload.workflow_id == "test-workflow-123"
        assert payload.status == "failure"  # Has failed tests
        assert payload.report_url == "https://example.com/report"
        assert payload.pr_url == "https://github.com/repo/pull/123"


class TestApprovalManager:
    """Test cases for ApprovalManager."""

    @pytest.fixture
    def approval_manager(self):
        """Create an ApprovalManager instance for testing."""
        return ApprovalManager(approval_timeout=5)  # Short timeout for testing

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'QA_OPERATOR_AUTO_APPROVE': 'true'})
    async def test_auto_approval(self, approval_manager):
        """Test automatic approval via environment variable."""
        approved = await approval_manager.request_approval(
            operation="commit",
            description="Test commit",
            files_affected=["test.py"]
        )
        
        assert approved is True

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'QA_OPERATOR_AUTO_DENY': 'true'})
    async def test_auto_denial(self, approval_manager):
        """Test automatic denial via environment variable."""
        approved = await approval_manager.request_approval(
            operation="commit",
            description="Test commit",
            files_affected=["test.py"]
        )
        
        assert approved is False

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'CI': 'true'})
    async def test_ci_environment_denial(self, approval_manager):
        """Test that CI environment defaults to denial."""
        approved = await approval_manager.request_approval(
            operation="commit",
            description="Test commit"
        )
        
        assert approved is False

    def test_manual_approval(self, approval_manager):
        """Test manual approval of request."""
        # Create a pending request
        request_id = "test-request-123"
        request = ApprovalRequest(
            request_id=request_id,
            operation="commit",
            description="Test commit"
        )
        approval_manager._pending_requests[request_id] = request
        
        # Approve it
        result = approval_manager.approve_request(request_id, "test-user")
        
        assert result is True
        assert request.approved is True
        assert request.approved_by == "test-user"
        assert request.approved_at is not None

    def test_manual_denial(self, approval_manager):
        """Test manual denial of request."""
        # Create a pending request
        request_id = "test-request-123"
        request = ApprovalRequest(
            request_id=request_id,
            operation="commit",
            description="Test commit"
        )
        approval_manager._pending_requests[request_id] = request
        
        # Deny it
        result = approval_manager.deny_request(request_id, "Not authorized")
        
        assert result is True
        assert request.approved is False
        assert request.rejection_reason == "Not authorized"

    def test_get_pending_requests(self, approval_manager):
        """Test getting pending requests."""
        # Add some pending requests
        request1 = ApprovalRequest(
            request_id="req1",
            operation="commit",
            description="Test commit 1"
        )
        request2 = ApprovalRequest(
            request_id="req2",
            operation="push",
            description="Test push"
        )
        
        approval_manager._pending_requests["req1"] = request1
        approval_manager._pending_requests["req2"] = request2
        
        pending = approval_manager.get_pending_requests()
        
        assert len(pending) == 2
        assert any(req.request_id == "req1" for req in pending)
        assert any(req.request_id == "req2" for req in pending)


class TestCIIntegrator:
    """Test cases for CIIntegrator."""

    @pytest.fixture
    def ci_config(self):
        """Create CI integration configuration."""
        return CIIntegrationConfig(
            environment=CIEnvironment.GITHUB_ACTIONS,
            artifact_upload=True,
            notification_webhook="https://example.com/webhook"
        )

    @pytest.fixture
    def report_generator(self, tmp_path):
        """Create a report generator for testing."""
        return ReportGenerator(tmp_path)

    @pytest.fixture
    def ci_integrator(self, ci_config, report_generator):
        """Create a CIIntegrator instance for testing."""
        return CIIntegrator(ci_config, report_generator)

    @pytest.fixture
    def sample_report(self):
        """Create a sample test report."""
        return TestRunReport(
            workflow_id="test-workflow-123",
            run_id="run-123",
            started_at=datetime.now() - timedelta(minutes=5),
            completed_at=datetime.now(),
            environment=CIEnvironment.GITHUB_ACTIONS,
            summary=TestResultSummary(
                total_tests=10,
                passed=8,
                failed=2,
                skipped=0,
                errors=0,
                duration=120.5,
                success_rate=80.0
            ),
            test_cases=[],
            artifacts=[]
        )

    @pytest.mark.asyncio
    async def test_send_notification_success(self, ci_integrator, sample_report):
        """Test successful notification sending."""
        payload = NotificationPayload(
            workflow_id="test-workflow-123",
            status="failure",
            summary=sample_report.summary
        )
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await ci_integrator.send_notification(payload)
            
            assert result is True
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_notification_failure(self, ci_integrator, sample_report):
        """Test notification sending failure."""
        payload = NotificationPayload(
            workflow_id="test-workflow-123",
            status="failure",
            summary=sample_report.summary
        )
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await ci_integrator.send_notification(payload)
            
            assert result is False

    def test_create_pr_template(self, ci_integrator, sample_report):
        """Test PR template creation."""
        template = PRTemplate(
            title_template="QA Updates - {workflow_id}",
            description_template="Automated test updates for workflow {workflow_id}",
            labels=["qa", "automated"],
            include_test_summary=True
        )
        
        pr_content = ci_integrator.create_pr_template(
            report=sample_report,
            template=template,
            changes_summary="Fixed failing tests"
        )
        
        assert "QA Updates - test-workflow-123" in pr_content["title"]
        assert "test-workflow-123" in pr_content["description"]
        assert "## 📊 Test Summary" in pr_content["description"]
        assert "**Workflow ID:** `test-workflow-123`" in pr_content["description"]
        assert pr_content["labels"] == ["qa", "automated"]

    def test_get_ci_environment_info(self, ci_integrator):
        """Test CI environment information retrieval."""
        info = ci_integrator.get_ci_environment_info()
        
        assert info["environment"] == "github_actions"
        assert info["artifact_upload"] is True
        assert "retention_days" in info


class TestGitOperationApprovalWorkflow:
    """Test cases for GitOperationApprovalWorkflow."""

    @pytest.fixture
    def mock_git_client(self):
        """Create a mock Git client."""
        client = Mock()
        client.stage_and_commit_files = AsyncMock()
        client.create_pull_request = AsyncMock()
        return client

    @pytest.fixture
    def approval_manager(self):
        """Create an approval manager for testing."""
        return ApprovalManager(approval_timeout=1)

    @pytest.fixture
    def workflow(self, mock_git_client, approval_manager):
        """Create a GitOperationApprovalWorkflow instance."""
        return GitOperationApprovalWorkflow(
            git_client=mock_git_client,
            approval_manager=approval_manager
        )

    @pytest.mark.asyncio
    async def test_commit_with_auto_approval(self, workflow, mock_git_client):
        """Test commit with automatic approval."""
        from orchestrator.mcp.git_client import GitOperationResult, GitOperationType
        
        mock_git_client.stage_and_commit_files.return_value = GitOperationResult(
            success=True,
            operation=GitOperationType.COMMIT,
            commit_hash="abc123",
            duration=1.0
        )
        
        result = await workflow.commit_with_approval(
            files=["test.py"],
            message="Fix test",
            auto_approve=True
        )
        
        assert result is True
        mock_git_client.stage_and_commit_files.assert_called_once_with(
            ["test.py"], "Fix test"
        )

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'QA_OPERATOR_AUTO_APPROVE': 'true'})
    async def test_commit_with_approval_granted(self, workflow, mock_git_client):
        """Test commit with approval granted."""
        from orchestrator.mcp.git_client import GitOperationResult, GitOperationType
        
        mock_git_client.stage_and_commit_files.return_value = GitOperationResult(
            success=True,
            operation=GitOperationType.COMMIT,
            commit_hash="abc123",
            duration=1.0
        )
        
        result = await workflow.commit_with_approval(
            files=["test.py"],
            message="Fix test",
            auto_approve=False
        )
        
        assert result is True

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'QA_OPERATOR_AUTO_DENY': 'true'})
    async def test_commit_with_approval_denied(self, workflow, mock_git_client):
        """Test commit with approval denied."""
        result = await workflow.commit_with_approval(
            files=["test.py"],
            message="Fix test",
            auto_approve=False
        )
        
        assert result is False
        mock_git_client.stage_and_commit_files.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_pr_with_auto_approval(self, workflow, mock_git_client):
        """Test PR creation with automatic approval."""
        from orchestrator.mcp.git_client import GitOperationResult, GitOperationType, PullRequestInfo
        
        mock_git_client.create_pull_request.return_value = GitOperationResult(
            success=True,
            operation=GitOperationType.REMOTE,
            pr_url="https://github.com/repo/pull/123",
            duration=2.0
        )
        
        pr_info = PullRequestInfo(
            title="Test PR",
            description="Test description",
            source_branch="feature/test"
        )
        
        result = await workflow.create_pr_with_approval(
            pr_info=pr_info,
            auto_approve=True
        )
        
        assert result == "https://github.com/repo/pull/123"
        mock_git_client.create_pull_request.assert_called_once_with(pr_info)


class TestReportingModels:
    """Test cases for reporting Pydantic models."""

    def test_test_result_summary_success_rate_calculation(self):
        """Test automatic success rate calculation."""
        summary = TestResultSummary(
            total_tests=10,
            passed=8,
            failed=2,
            skipped=0,
            errors=0,
            duration=120.0,
            success_rate=0  # Will be calculated automatically
        )
        
        assert summary.success_rate == 80.0

    def test_test_result_summary_zero_tests(self):
        """Test success rate calculation with zero tests."""
        summary = TestResultSummary(
            total_tests=0,
            passed=0,
            failed=0,
            skipped=0,
            errors=0,
            duration=0.0,
            success_rate=0
        )
        
        assert summary.success_rate == 0.0

    def test_test_case_result_validation(self):
        """Test TestCaseResult model validation."""
        result = TestCaseResult(
            name="test_example",
            status=TestStatus.PASSED,
            duration=1.5,
            file_path="tests/test_example.py"
        )
        
        assert result.name == "test_example"
        assert result.status == TestStatus.PASSED
        assert result.duration == 1.5

    def test_test_case_result_empty_name(self):
        """Test TestCaseResult validation with empty name."""
        with pytest.raises(ValueError) as exc_info:
            TestCaseResult(
                name="",
                status=TestStatus.PASSED,
                duration=1.0,
                file_path="test.py"
            )
        
        assert "cannot be empty" in str(exc_info.value)

    def test_artifact_info_validation(self):
        """Test ArtifactInfo model validation."""
        artifact = ArtifactInfo(
            path="/path/to/screenshot.png",
            type="screenshot",
            size=1024,
            created_at=datetime.now()
        )
        
        assert artifact.type == "screenshot"
        assert artifact.size == 1024

    def test_artifact_info_invalid_type(self):
        """Test ArtifactInfo validation with invalid type."""
        with pytest.raises(ValueError) as exc_info:
            ArtifactInfo(
                path="/path/to/file.txt",
                type="invalid_type",
                size=100,
                created_at=datetime.now()
            )
        
        assert "must be one of" in str(exc_info.value)

    def test_approval_request_validation(self):
        """Test ApprovalRequest model validation."""
        request = ApprovalRequest(
            request_id="req-123",
            operation="commit",
            description="Test commit operation",
            files_affected=["test.py", "src/app.py"]
        )
        
        assert request.request_id == "req-123"
        assert request.operation == "commit"
        assert len(request.files_affected) == 2

    def test_approval_request_invalid_operation(self):
        """Test ApprovalRequest validation with invalid operation."""
        with pytest.raises(ValueError) as exc_info:
            ApprovalRequest(
                request_id="req-123",
                operation="invalid_op",
                description="Test operation"
            )
        
        assert "must be one of" in str(exc_info.value)

    def test_pr_template_validation(self):
        """Test PRTemplate model validation."""
        template = PRTemplate(
            title_template="QA Updates - {workflow_id}",
            description_template="Automated updates for {workflow_id}",
            labels=["qa", "automated"],
            reviewers=["reviewer1", "reviewer2"],
            include_test_summary=True
        )
        
        assert template.include_test_summary is True
        assert len(template.labels) == 2
        assert len(template.reviewers) == 2

    def test_ci_integration_config_validation(self):
        """Test CIIntegrationConfig model validation."""
        config = CIIntegrationConfig(
            environment=CIEnvironment.GITHUB_ACTIONS,
            artifact_upload=True,
            notification_webhook="https://example.com/webhook",
            retention_days=30
        )
        
        assert config.environment == CIEnvironment.GITHUB_ACTIONS
        assert config.retention_days == 30

    def test_ci_integration_config_invalid_webhook(self):
        """Test CIIntegrationConfig validation with invalid webhook."""
        with pytest.raises(ValueError) as exc_info:
            CIIntegrationConfig(
                environment=CIEnvironment.LOCAL,
                notification_webhook="invalid-url"
            )
        
        assert "must start with http" in str(exc_info.value)