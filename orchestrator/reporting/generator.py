"""
Test Report Generator.

Generates comprehensive test reports in various formats with CI integration,
artifact attachment, and Git correlation through workflow_id.
"""

import json
import logging
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from jinja2 import Environment, FileSystemLoader, Template

from ..core.exceptions import ValidationError, FileOperationError
from .models import (
    TestRunReport,
    TestResultSummary,
    TestCaseResult,
    ArtifactInfo,
    ReportFormat,
    CIEnvironment,
    TestStatus,
    ReportGenerationRequest,
    CIIntegrationConfig,
    NotificationPayload,
)


class ReportGenerator:
    """
    Generates test reports in various formats with CI integration.
    
    Supports JSON, HTML, Markdown, and JUnit XML formats with
    artifact attachment and Git correlation.
    """
    
    def __init__(
        self,
        output_dir: Path,
        template_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory for generated reports
            template_dir: Directory containing report templates
            logger: Optional logger instance
        """
        self.output_dir = Path(output_dir)
        self.template_dir = template_dir or (Path(__file__).parent / "templates")
        self.logger = logger or logging.getLogger(__name__)
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment for templates
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True
        )
        
        # CI environment detection
        self.ci_environment = self._detect_ci_environment()
        
    def _detect_ci_environment(self) -> CIEnvironment:
        """Detect the current CI environment."""
        if os.getenv("GITHUB_ACTIONS"):
            return CIEnvironment.GITHUB_ACTIONS
        elif os.getenv("GITLAB_CI"):
            return CIEnvironment.GITLAB_CI
        elif os.getenv("JENKINS_URL"):
            return CIEnvironment.JENKINS
        elif os.getenv("AZURE_HTTP_USER_AGENT"):
            return CIEnvironment.AZURE_DEVOPS
        elif os.getenv("CIRCLECI"):
            return CIEnvironment.CIRCLECI
        elif os.getenv("TRAVIS"):
            return CIEnvironment.TRAVIS_CI
        elif os.getenv("CI"):
            return CIEnvironment.UNKNOWN
        else:
            return CIEnvironment.LOCAL
    
    def _collect_ci_info(self) -> Dict[str, Any]:
        """Collect CI environment information."""
        ci_info = {
            "environment": self.ci_environment.value,
            "detected_at": datetime.now().isoformat(),
        }
        
        if self.ci_environment == CIEnvironment.GITHUB_ACTIONS:
            ci_info.update({
                "repository": os.getenv("GITHUB_REPOSITORY"),
                "ref": os.getenv("GITHUB_REF"),
                "sha": os.getenv("GITHUB_SHA"),
                "run_id": os.getenv("GITHUB_RUN_ID"),
                "run_number": os.getenv("GITHUB_RUN_NUMBER"),
                "actor": os.getenv("GITHUB_ACTOR"),
                "workflow": os.getenv("GITHUB_WORKFLOW"),
            })
        elif self.ci_environment == CIEnvironment.GITLAB_CI:
            ci_info.update({
                "project_id": os.getenv("CI_PROJECT_ID"),
                "project_name": os.getenv("CI_PROJECT_NAME"),
                "commit_sha": os.getenv("CI_COMMIT_SHA"),
                "commit_ref": os.getenv("CI_COMMIT_REF_NAME"),
                "pipeline_id": os.getenv("CI_PIPELINE_ID"),
                "job_id": os.getenv("CI_JOB_ID"),
                "runner_description": os.getenv("CI_RUNNER_DESCRIPTION"),
            })
        elif self.ci_environment == CIEnvironment.JENKINS:
            ci_info.update({
                "build_number": os.getenv("BUILD_NUMBER"),
                "build_id": os.getenv("BUILD_ID"),
                "job_name": os.getenv("JOB_NAME"),
                "build_url": os.getenv("BUILD_URL"),
                "node_name": os.getenv("NODE_NAME"),
                "workspace": os.getenv("WORKSPACE"),
            })
        
        return ci_info
    
    def _collect_git_info(self, git_client=None) -> Optional[Dict[str, Any]]:
        """Collect Git repository information."""
        if not git_client:
            return None
        
        try:
            # This would use the actual git client in real implementation
            return {
                "branch": "main",  # git_client.get_current_branch()
                "commit_hash": "abc123",  # git_client.get_current_commit()
                "commit_message": "Test updates",  # git_client.get_commit_message()
                "author": "QA Operator",  # git_client.get_commit_author()
                "repository_url": "https://github.com/example/repo",
                "collected_at": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.warning(f"Failed to collect Git information: {e}")
            return None
    
    def _collect_artifacts(self, artifacts_dir: Path) -> List[ArtifactInfo]:
        """Collect artifact information from directory."""
        artifacts = []
        
        if not artifacts_dir.exists():
            return artifacts
        
        for artifact_path in artifacts_dir.rglob("*"):
            if artifact_path.is_file():
                try:
                    # Determine artifact type from extension
                    suffix = artifact_path.suffix.lower()
                    if suffix in [".png", ".jpg", ".jpeg"]:
                        artifact_type = "screenshot"
                    elif suffix in [".zip", ".trace"]:
                        artifact_type = "trace"
                    elif suffix in [".webm", ".mp4"]:
                        artifact_type = "video"
                    elif suffix in [".log", ".txt"]:
                        artifact_type = "log"
                    elif suffix in [".json", ".xml"]:
                        artifact_type = "report"
                    else:
                        artifact_type = "other"
                    
                    stat = artifact_path.stat()
                    
                    artifacts.append(ArtifactInfo(
                        path=str(artifact_path),
                        type=artifact_type,
                        size=stat.st_size,
                        created_at=datetime.fromtimestamp(stat.st_ctime),
                        test_name=self._extract_test_name_from_path(artifact_path),
                        description=f"{artifact_type.title()} artifact"
                    ))
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process artifact {artifact_path}: {e}")
                    continue
        
        return artifacts
    
    def _extract_test_name_from_path(self, path: Path) -> Optional[str]:
        """Extract test name from artifact path."""
        # Simple heuristic: look for test name in path components
        parts = path.parts
        for part in parts:
            if "test" in part.lower() and not part.startswith("."):
                return part.replace(".spec", "").replace(".test", "")
        return None
    
    def generate_report(
        self,
        request: ReportGenerationRequest,
        test_results: List[TestCaseResult],
        artifacts_dir: Optional[Path] = None,
        git_client=None
    ) -> TestRunReport:
        """
        Generate a comprehensive test report.
        
        Args:
            request: Report generation request
            test_results: List of test case results
            artifacts_dir: Directory containing test artifacts
            git_client: Optional Git client for repository information
            
        Returns:
            Generated test report
        """
        self.logger.info(f"Generating test report for workflow: {request.workflow_id}")
        
        # Calculate summary statistics
        total_tests = len(test_results)
        passed = sum(1 for result in test_results if result.status == TestStatus.PASSED)
        failed = sum(1 for result in test_results if result.status == TestStatus.FAILED)
        skipped = sum(1 for result in test_results if result.status == TestStatus.SKIPPED)
        errors = sum(1 for result in test_results if result.status == TestStatus.ERROR)
        total_duration = sum(result.duration for result in test_results)
        
        summary = TestResultSummary(
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration=total_duration,
            success_rate=(passed / total_tests * 100) if total_tests > 0 else 0.0
        )
        
        # Collect artifacts if directory provided
        artifacts = []
        if artifacts_dir and request.include_artifacts:
            artifacts = self._collect_artifacts(artifacts_dir)
        
        # Collect Git information if requested
        git_info = None
        if request.include_git_info:
            git_info = self._collect_git_info(git_client)
        
        # Collect CI information if requested
        ci_info = None
        if request.include_ci_info:
            ci_info = self._collect_ci_info()
        
        # Create the report
        report = TestRunReport(
            workflow_id=request.workflow_id,
            run_id=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            started_at=datetime.now() - timedelta(seconds=total_duration),
            completed_at=datetime.now(),
            environment=self.ci_environment,
            summary=summary,
            test_cases=test_results,
            artifacts=artifacts,
            configuration={
                "format": request.format.value,
                "include_artifacts": request.include_artifacts,
                "include_git_info": request.include_git_info,
                "include_ci_info": request.include_ci_info,
            },
            git_info=git_info,
            ci_info=ci_info
        )
        
        # Save report in requested format
        if request.output_path:
            self._save_report(report, request.format, Path(request.output_path))
        
        self.logger.info(f"Generated test report: {summary.passed}/{total_tests} tests passed")
        return report
    
    def _save_report(self, report: TestRunReport, format: ReportFormat, output_path: Path) -> None:
        """Save report to file in specified format."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == ReportFormat.JSON:
                self._save_json_report(report, output_path)
            elif format == ReportFormat.HTML:
                self._save_html_report(report, output_path)
            elif format == ReportFormat.MARKDOWN:
                self._save_markdown_report(report, output_path)
            elif format == ReportFormat.JUNIT:
                self._save_junit_report(report, output_path)
            else:
                raise ValueError(f"Unsupported report format: {format}")
            
            self.logger.info(f"Saved {format.value} report to: {output_path}")
            
        except Exception as e:
            raise FileOperationError(
                f"Failed to save report: {e}",
                file_path=str(output_path),
                operation="write"
            )
    
    def _save_json_report(self, report: TestRunReport, output_path: Path) -> None:
        """Save report as JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report.dict(), f, indent=2, default=str)
    
    def _save_html_report(self, report: TestRunReport, output_path: Path) -> None:
        """Save report as HTML."""
        try:
            template = self.jinja_env.get_template("report.html")
        except Exception:
            # Use a simple default template if none exists
            template = Template(self._get_default_html_template())
        
        html_content = template.render(report=report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _save_markdown_report(self, report: TestRunReport, output_path: Path) -> None:
        """Save report as Markdown."""
        try:
            template = self.jinja_env.get_template("report.md")
        except Exception:
            # Use a simple default template if none exists
            template = Template(self._get_default_markdown_template())
        
        markdown_content = template.render(report=report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def _save_junit_report(self, report: TestRunReport, output_path: Path) -> None:
        """Save report as JUnit XML."""
        try:
            template = self.jinja_env.get_template("junit.xml")
        except Exception:
            # Use a simple default template if none exists
            template = Template(self._get_default_junit_template())
        
        xml_content = template.render(report=report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
    
    def _get_default_html_template(self) -> str:
        """Get default HTML template."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Test Report - {{ report.workflow_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .passed { color: green; }
        .failed { color: red; }
        .skipped { color: orange; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Test Report</h1>
        <p><strong>Workflow ID:</strong> {{ report.workflow_id }}</p>
        <p><strong>Run ID:</strong> {{ report.run_id }}</p>
        <p><strong>Environment:</strong> {{ report.environment.value }}</p>
        <p><strong>Completed:</strong> {{ report.completed_at }}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> {{ report.summary.total_tests }}</p>
        <p><strong>Passed:</strong> <span class="passed">{{ report.summary.passed }}</span></p>
        <p><strong>Failed:</strong> <span class="failed">{{ report.summary.failed }}</span></p>
        <p><strong>Skipped:</strong> <span class="skipped">{{ report.summary.skipped }}</span></p>
        <p><strong>Success Rate:</strong> {{ "%.1f"|format(report.summary.success_rate) }}%</p>
        <p><strong>Duration:</strong> {{ "%.2f"|format(report.summary.duration) }}s</p>
    </div>
    
    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Status</th>
            <th>Duration</th>
            <th>File</th>
        </tr>
        {% for test in report.test_cases %}
        <tr>
            <td>{{ test.name }}</td>
            <td class="{{ test.status.value }}">{{ test.status.value.upper() }}</td>
            <td>{{ "%.2f"|format(test.duration) }}s</td>
            <td>{{ test.file_path }}</td>
        </tr>
        {% endfor %}
    </table>
    
    {% if report.artifacts %}
    <h2>Artifacts</h2>
    <ul>
        {% for artifact in report.artifacts %}
        <li>{{ artifact.type }}: {{ artifact.path }} ({{ artifact.size }} bytes)</li>
        {% endfor %}
    </ul>
    {% endif %}
</body>
</html>
        """
    
    def _get_default_markdown_template(self) -> str:
        """Get default Markdown template."""
        return """
# Test Report - {{ report.workflow_id }}

**Workflow ID:** {{ report.workflow_id }}  
**Run ID:** {{ report.run_id }}  
**Environment:** {{ report.environment.value }}  
**Completed:** {{ report.completed_at }}  

## Summary

- **Total Tests:** {{ report.summary.total_tests }}
- **Passed:** {{ report.summary.passed }} ✅
- **Failed:** {{ report.summary.failed }} ❌
- **Skipped:** {{ report.summary.skipped }} ⏭️
- **Success Rate:** {{ "%.1f"|format(report.summary.success_rate) }}%
- **Duration:** {{ "%.2f"|format(report.summary.duration) }}s

## Test Results

| Test Name | Status | Duration | File |
|-----------|--------|----------|------|
{% for test in report.test_cases -%}
| {{ test.name }} | {{ test.status.value.upper() }} | {{ "%.2f"|format(test.duration) }}s | {{ test.file_path }} |
{% endfor %}

{% if report.artifacts %}
## Artifacts

{% for artifact in report.artifacts -%}
- {{ artifact.type }}: `{{ artifact.path }}` ({{ artifact.size }} bytes)
{% endfor %}
{% endif %}

{% if report.git_info %}
## Git Information

- **Branch:** {{ report.git_info.branch }}
- **Commit:** {{ report.git_info.commit_hash }}
- **Author:** {{ report.git_info.author }}
{% endif %}
        """
    
    def _get_default_junit_template(self) -> str:
        """Get default JUnit XML template."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="{{ report.workflow_id }}" 
            tests="{{ report.summary.total_tests }}" 
            failures="{{ report.summary.failed }}" 
            errors="{{ report.summary.errors }}" 
            skipped="{{ report.summary.skipped }}" 
            time="{{ report.summary.duration }}">
    <testsuite name="QA Operator Tests" 
               tests="{{ report.summary.total_tests }}" 
               failures="{{ report.summary.failed }}" 
               errors="{{ report.summary.errors }}" 
               skipped="{{ report.summary.skipped }}" 
               time="{{ report.summary.duration }}">
        {% for test in report.test_cases %}
        <testcase name="{{ test.name }}" 
                  classname="{{ test.file_path }}" 
                  time="{{ test.duration }}">
            {% if test.status.value == "failed" %}
            <failure message="{{ test.error_message or 'Test failed' }}">
                {{ test.stack_trace or test.error_message or 'No details available' }}
            </failure>
            {% elif test.status.value == "error" %}
            <error message="{{ test.error_message or 'Test error' }}">
                {{ test.stack_trace or test.error_message or 'No details available' }}
            </error>
            {% elif test.status.value == "skipped" %}
            <skipped message="Test skipped" />
            {% endif %}
        </testcase>
        {% endfor %}
    </testsuite>
</testsuites>
        """
    
    def create_ci_artifacts_package(
        self, 
        report: TestRunReport, 
        artifacts_dir: Path,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Create a compressed package of artifacts for CI upload.
        
        Args:
            report: Test report
            artifacts_dir: Directory containing artifacts
            output_path: Optional output path for package
            
        Returns:
            Path to created package
        """
        if output_path is None:
            output_path = self.output_dir / f"artifacts-{report.workflow_id}.zip"
        
        self.logger.info(f"Creating CI artifacts package: {output_path}")
        
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add the report
                report_path = self.output_dir / f"report-{report.workflow_id}.json"
                with open(report_path, 'w') as f:
                    json.dump(report.dict(), f, indent=2, default=str)
                zipf.write(report_path, "report.json")
                
                # Add artifacts
                if artifacts_dir.exists():
                    for artifact_path in artifacts_dir.rglob("*"):
                        if artifact_path.is_file():
                            # Create relative path within zip
                            rel_path = artifact_path.relative_to(artifacts_dir)
                            zipf.write(artifact_path, f"artifacts/{rel_path}")
                
                # Add metadata
                metadata = {
                    "workflow_id": report.workflow_id,
                    "created_at": datetime.now().isoformat(),
                    "environment": report.environment.value,
                    "summary": report.summary.dict(),
                }
                
                zipf.writestr("metadata.json", json.dumps(metadata, indent=2))
            
            self.logger.info(f"Created artifacts package: {output_path} ({output_path.stat().st_size} bytes)")
            return output_path
            
        except Exception as e:
            raise FileOperationError(
                f"Failed to create artifacts package: {e}",
                file_path=str(output_path),
                operation="create_package"
            )
    
    def generate_notification_payload(
        self, 
        report: TestRunReport,
        report_url: Optional[str] = None,
        artifacts_url: Optional[str] = None,
        pr_url: Optional[str] = None
    ) -> NotificationPayload:
        """
        Generate notification payload for CI integration.
        
        Args:
            report: Test report
            report_url: Optional URL to full report
            artifacts_url: Optional URL to artifacts
            pr_url: Optional URL to created PR
            
        Returns:
            Notification payload
        """
        # Determine overall status
        if report.summary.failed > 0 or report.summary.errors > 0:
            status = "failure"
        elif report.summary.total_tests == 0:
            status = "error"
        else:
            status = "success"
        
        return NotificationPayload(
            workflow_id=report.workflow_id,
            status=status,
            summary=report.summary,
            report_url=report_url,
            artifacts_url=artifacts_url,
            pr_url=pr_url
        )
    
    def get_ci_environment(self) -> CIEnvironment:
        """Get detected CI environment."""
        return self.ci_environment
    
    def is_ci_environment(self) -> bool:
        """Check if running in CI environment."""
        return self.ci_environment != CIEnvironment.LOCAL