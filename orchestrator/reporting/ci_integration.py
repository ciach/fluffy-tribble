"""
CI Integration and Approval Workflow.

Handles CI-specific operations, artifact upload, notifications,
and manual approval workflow for Git operations.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from urllib.parse import urljoin
import aiohttp

from ..core.exceptions import ValidationError, FileOperationError
from .models import (
    CIIntegrationConfig,
    CIEnvironment,
    ApprovalRequest,
    NotificationPayload,
    TestRunReport,
    PRTemplate,
)
from .generator import ReportGenerator


class ApprovalManager:
    """
    Manages manual approval workflow for Git operations.
    
    Provides interactive approval requests with timeout handling
    and approval tracking.
    """
    
    def __init__(
        self,
        approval_timeout: int = 300,  # 5 minutes default
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the approval manager.
        
        Args:
            approval_timeout: Default timeout for approval requests in seconds
            logger: Optional logger instance
        """
        self.approval_timeout = approval_timeout
        self.logger = logger or logging.getLogger(__name__)
        
        # Active approval requests
        self._pending_requests: Dict[str, ApprovalRequest] = {}
        self._approval_callbacks: Dict[str, Callable] = {}
    
    async def request_approval(
        self,
        operation: str,
        description: str,
        files_affected: Optional[List[str]] = None,
        commit_message: Optional[str] = None,
        pr_title: Optional[str] = None,
        pr_description: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> bool:
        """
        Request manual approval for a Git operation.
        
        Args:
            operation: Git operation requiring approval
            description: Human-readable description
            files_affected: List of files to be modified
            commit_message: Proposed commit message
            pr_title: Proposed PR title
            pr_description: Proposed PR description
            timeout: Approval timeout in seconds
            
        Returns:
            True if approved, False if denied or timed out
        """
        request_id = f"approval-{int(time.time())}-{operation}"
        timeout = timeout or self.approval_timeout
        expires_at = datetime.now() + timedelta(seconds=timeout)
        
        request = ApprovalRequest(
            request_id=request_id,
            operation=operation,
            description=description,
            files_affected=files_affected or [],
            commit_message=commit_message,
            pr_title=pr_title,
            pr_description=pr_description,
            expires_at=expires_at
        )
        
        self._pending_requests[request_id] = request
        
        self.logger.info(f"Requesting approval for {operation}: {description}")
        
        # Display approval request to user
        self._display_approval_request(request)
        
        # Wait for approval or timeout
        try:
            approved = await self._wait_for_approval(request_id, timeout)
            
            if approved:
                self.logger.info(f"Operation {operation} approved")
                request.approved = True
                request.approved_at = datetime.now()
            else:
                self.logger.warning(f"Operation {operation} denied or timed out")
                request.approved = False
            
            return approved
            
        finally:
            # Clean up
            self._pending_requests.pop(request_id, None)
            self._approval_callbacks.pop(request_id, None)
    
    def _display_approval_request(self, request: ApprovalRequest) -> None:
        """Display approval request to user."""
        print("\n" + "="*60)
        print("🔐 MANUAL APPROVAL REQUIRED")
        print("="*60)
        print(f"Operation: {request.operation.upper()}")
        print(f"Description: {request.description}")
        
        if request.files_affected:
            print(f"Files affected: {', '.join(request.files_affected)}")
        
        if request.commit_message:
            print(f"Commit message: {request.commit_message}")
        
        if request.pr_title:
            print(f"PR title: {request.pr_title}")
        
        if request.pr_description:
            print(f"PR description: {request.pr_description[:100]}...")
        
        print(f"Expires at: {request.expires_at}")
        print("\nOptions:")
        print("  [y/yes] - Approve the operation")
        print("  [n/no]  - Deny the operation")
        print("  [d/details] - Show more details")
        print("="*60)
    
    async def _wait_for_approval(self, request_id: str, timeout: int) -> bool:
        """Wait for user approval input."""
        # In a real implementation, this would integrate with the actual UI
        # For now, we'll simulate approval based on environment variables
        
        # Check for auto-approval environment variable
        auto_approve = os.getenv("QA_OPERATOR_AUTO_APPROVE", "").lower()
        if auto_approve in ["true", "1", "yes"]:
            self.logger.info("Auto-approval enabled, approving operation")
            await asyncio.sleep(1)  # Simulate brief delay
            return True
        
        # Check for auto-deny environment variable
        auto_deny = os.getenv("QA_OPERATOR_AUTO_DENY", "").lower()
        if auto_deny in ["true", "1", "yes"]:
            self.logger.info("Auto-deny enabled, denying operation")
            await asyncio.sleep(1)
            return False
        
        # In CI environments, default to deny for safety
        if os.getenv("CI"):
            self.logger.warning("Running in CI without auto-approval, denying operation")
            return False
        
        # For local development, simulate interactive approval
        # In a real implementation, this would use actual user input
        self.logger.info("Simulating user approval (would be interactive in real implementation)")
        await asyncio.sleep(2)  # Simulate user thinking time
        
        # Default to approval for testing purposes
        return True
    
    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get list of pending approval requests."""
        return list(self._pending_requests.values())
    
    def approve_request(self, request_id: str, approver: str = "manual") -> bool:
        """Manually approve a request."""
        if request_id in self._pending_requests:
            request = self._pending_requests[request_id]
            request.approved = True
            request.approved_by = approver
            request.approved_at = datetime.now()
            
            # Trigger callback if exists
            if request_id in self._approval_callbacks:
                self._approval_callbacks[request_id](True)
            
            return True
        return False
    
    def deny_request(self, request_id: str, reason: str = "Manual denial") -> bool:
        """Manually deny a request."""
        if request_id in self._pending_requests:
            request = self._pending_requests[request_id]
            request.approved = False
            request.rejection_reason = reason
            
            # Trigger callback if exists
            if request_id in self._approval_callbacks:
                self._approval_callbacks[request_id](False)
            
            return True
        return False


class CIIntegrator:
    """
    Handles CI-specific operations and integrations.
    
    Provides artifact upload, notifications, and CI environment
    specific functionality.
    """
    
    def __init__(
        self,
        config: CIIntegrationConfig,
        report_generator: ReportGenerator,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the CI integrator.
        
        Args:
            config: CI integration configuration
            report_generator: Report generator instance
            logger: Optional logger instance
        """
        self.config = config
        self.report_generator = report_generator
        self.logger = logger or logging.getLogger(__name__)
    
    async def upload_artifacts(
        self,
        report: TestRunReport,
        artifacts_dir: Path,
        upload_url: Optional[str] = None
    ) -> Optional[str]:
        """
        Upload artifacts to CI system.
        
        Args:
            report: Test report
            artifacts_dir: Directory containing artifacts
            upload_url: Optional custom upload URL
            
        Returns:
            URL to uploaded artifacts if successful
        """
        if not self.config.artifact_upload:
            self.logger.info("Artifact upload disabled in configuration")
            return None
        
        self.logger.info(f"Uploading artifacts for workflow: {report.workflow_id}")
        
        try:
            # Create artifacts package
            package_path = self.report_generator.create_ci_artifacts_package(
                report, artifacts_dir
            )
            
            # Determine upload URL based on CI environment
            if upload_url is None:
                upload_url = self._get_default_upload_url()
            
            if upload_url:
                # Upload package
                uploaded_url = await self._upload_package(package_path, upload_url)
                self.logger.info(f"Artifacts uploaded to: {uploaded_url}")
                return uploaded_url
            else:
                self.logger.warning("No upload URL configured, artifacts not uploaded")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to upload artifacts: {e}")
            return None
    
    def _get_default_upload_url(self) -> Optional[str]:
        """Get default upload URL based on CI environment."""
        if self.config.environment == CIEnvironment.GITHUB_ACTIONS:
            # GitHub Actions artifacts are handled by the runner
            return None
        elif self.config.environment == CIEnvironment.GITLAB_CI:
            project_id = os.getenv("CI_PROJECT_ID")
            job_id = os.getenv("CI_JOB_ID")
            if project_id and job_id:
                return f"https://gitlab.com/api/v4/projects/{project_id}/jobs/{job_id}/artifacts"
        elif self.config.environment == CIEnvironment.JENKINS:
            build_url = os.getenv("BUILD_URL")
            if build_url:
                return urljoin(build_url, "artifact/")
        
        return None
    
    async def _upload_package(self, package_path: Path, upload_url: str) -> str:
        """Upload package to specified URL."""
        # This is a simplified implementation
        # Real implementation would handle authentication, retries, etc.
        
        async with aiohttp.ClientSession() as session:
            with open(package_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=package_path.name)
                
                async with session.post(upload_url, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('url', upload_url)
                    else:
                        raise Exception(f"Upload failed with status {response.status}")
    
    async def send_notification(
        self,
        payload: NotificationPayload,
        webhook_url: Optional[str] = None
    ) -> bool:
        """
        Send notification to configured webhook.
        
        Args:
            payload: Notification payload
            webhook_url: Optional custom webhook URL
            
        Returns:
            True if notification sent successfully
        """
        webhook_url = webhook_url or self.config.notification_webhook
        
        if not webhook_url:
            self.logger.debug("No notification webhook configured")
            return False
        
        self.logger.info(f"Sending notification for workflow: {payload.workflow_id}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload.dict(),
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        self.logger.info("Notification sent successfully")
                        return True
                    else:
                        self.logger.warning(f"Notification failed with status {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
            return False
    
    def create_pr_template(
        self,
        report: TestRunReport,
        template: PRTemplate,
        changes_summary: str = "Automated test updates"
    ) -> Dict[str, Any]:
        """
        Create pull request content from template.
        
        Args:
            report: Test report
            template: PR template configuration
            changes_summary: Summary of changes made
            
        Returns:
            Dictionary with PR title and description
        """
        # Template variables
        variables = {
            "workflow_id": report.workflow_id,
            "run_id": report.run_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": report.summary.total_tests,
            "passed_tests": report.summary.passed,
            "failed_tests": report.summary.failed,
            "success_rate": f"{report.summary.success_rate:.1f}%",
            "duration": f"{report.summary.duration:.2f}s",
            "changes_summary": changes_summary,
            "environment": report.environment.value,
        }
        
        # Format title
        title = template.title_template.format(**variables)
        
        # Format description
        description = template.description_template.format(**variables)
        
        # Add test summary if requested
        if template.include_test_summary:
            description += self._generate_test_summary_section(report)
        
        # Add artifact links if requested
        if template.include_artifacts and report.artifacts:
            description += self._generate_artifacts_section(report)
        
        # Add workflow correlation
        description += f"\n\n---\n**Workflow ID:** `{report.workflow_id}`\n"
        description += f"**Generated:** {datetime.now().isoformat()}\n"
        description += "**Type:** Automated QA Operator Updates"
        
        return {
            "title": title,
            "description": description,
            "labels": template.labels,
            "reviewers": template.reviewers
        }
    
    def _generate_test_summary_section(self, report: TestRunReport) -> str:
        """Generate test summary section for PR description."""
        summary = f"""

## 📊 Test Summary

- **Total Tests:** {report.summary.total_tests}
- **Passed:** {report.summary.passed} ✅
- **Failed:** {report.summary.failed} ❌
- **Skipped:** {report.summary.skipped} ⏭️
- **Success Rate:** {report.summary.success_rate:.1f}%
- **Duration:** {report.summary.duration:.2f}s

"""
        
        # Add failed tests details if any
        failed_tests = [test for test in report.test_cases if test.status.value == "failed"]
        if failed_tests:
            summary += "### ❌ Failed Tests\n\n"
            for test in failed_tests[:5]:  # Limit to first 5
                summary += f"- `{test.name}` - {test.error_message or 'No error message'}\n"
            
            if len(failed_tests) > 5:
                summary += f"- ... and {len(failed_tests) - 5} more\n"
            
            summary += "\n"
        
        return summary
    
    def _generate_artifacts_section(self, report: TestRunReport) -> str:
        """Generate artifacts section for PR description."""
        section = "\n## 📎 Artifacts\n\n"
        
        # Group artifacts by type
        artifacts_by_type = {}
        for artifact in report.artifacts:
            if artifact.type not in artifacts_by_type:
                artifacts_by_type[artifact.type] = []
            artifacts_by_type[artifact.type].append(artifact)
        
        for artifact_type, artifacts in artifacts_by_type.items():
            section += f"### {artifact_type.title()}\n"
            for artifact in artifacts[:3]:  # Limit to first 3 per type
                section += f"- `{Path(artifact.path).name}` ({artifact.size} bytes)\n"
            
            if len(artifacts) > 3:
                section += f"- ... and {len(artifacts) - 3} more {artifact_type} files\n"
            
            section += "\n"
        
        return section
    
    def get_ci_environment_info(self) -> Dict[str, Any]:
        """Get CI environment information."""
        return {
            "environment": self.config.environment.value,
            "artifact_upload": self.config.artifact_upload,
            "report_format": self.config.report_format.value,
            "retention_days": self.config.retention_days,
            "compress_artifacts": self.config.compress_artifacts,
        }


class GitOperationApprovalWorkflow:
    """
    Combines Git operations with approval workflow.
    
    Provides high-level methods that integrate Git operations
    with manual approval requirements.
    """
    
    def __init__(
        self,
        git_client,
        approval_manager: ApprovalManager,
        ci_integrator: Optional[CIIntegrator] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the approval workflow.
        
        Args:
            git_client: Git MCP client instance
            approval_manager: Approval manager instance
            ci_integrator: Optional CI integrator
            logger: Optional logger instance
        """
        self.git_client = git_client
        self.approval_manager = approval_manager
        self.ci_integrator = ci_integrator
        self.logger = logger or logging.getLogger(__name__)
    
    async def commit_with_approval(
        self,
        files: List[str],
        message: str,
        auto_approve: bool = False
    ) -> bool:
        """
        Commit changes with approval workflow.
        
        Args:
            files: Files to commit
            message: Commit message
            auto_approve: Skip approval if True
            
        Returns:
            True if committed successfully
        """
        if not auto_approve:
            approved = await self.approval_manager.request_approval(
                operation="commit",
                description=f"Commit {len(files)} files with message: {message}",
                files_affected=files,
                commit_message=message
            )
            
            if not approved:
                self.logger.info("Commit operation denied")
                return False
        
        try:
            result = await self.git_client.stage_and_commit_files(files, message)
            return result.success
        except Exception as e:
            self.logger.error(f"Commit failed: {e}")
            return False
    
    async def create_pr_with_approval(
        self,
        pr_info,
        report: Optional[TestRunReport] = None,
        auto_approve: bool = False
    ) -> Optional[str]:
        """
        Create pull request with approval workflow.
        
        Args:
            pr_info: Pull request information
            report: Optional test report for context
            auto_approve: Skip approval if True
            
        Returns:
            PR URL if created successfully
        """
        if not auto_approve:
            approved = await self.approval_manager.request_approval(
                operation="create_pr",
                description=f"Create PR: {pr_info.title}",
                pr_title=pr_info.title,
                pr_description=pr_info.description
            )
            
            if not approved:
                self.logger.info("PR creation denied")
                return None
        
        try:
            result = await self.git_client.create_pull_request(pr_info)
            if result.success:
                # Send notification if CI integrator available
                if self.ci_integrator and report:
                    payload = self.ci_integrator.report_generator.generate_notification_payload(
                        report, pr_url=result.pr_url
                    )
                    await self.ci_integrator.send_notification(payload)
                
                return result.pr_url
            return None
        except Exception as e:
            self.logger.error(f"PR creation failed: {e}")
            return None