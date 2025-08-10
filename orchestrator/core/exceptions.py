"""
Base exception classes for QA Operator.

Provides a hierarchy of exceptions for different error types that can occur
during the testing workflow.
"""

from typing import Optional, Dict, Any


class QAOperatorError(Exception):
    """Base exception class for all QA Operator errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for structured logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
        }


class MCPConnectionError(QAOperatorError):
    """Raised when MCP server connection fails."""

    def __init__(
        self,
        message: str,
        server_name: Optional[str] = None,
        retry_count: Optional[int] = None,
    ):
        super().__init__(message, "MCP_CONNECTION_FAILED")
        self.server_name = server_name
        self.retry_count = retry_count
        self.context.update(
            {
                "server_name": server_name,
                "retry_count": retry_count,
            }
        )


class ModelError(QAOperatorError):
    """Raised when AI model operations fail."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        task_type: Optional[str] = None,
    ):
        super().__init__(message, "MODEL_ERROR")
        self.model_name = model_name
        self.task_type = task_type
        self.context.update(
            {
                "model_name": model_name,
                "task_type": task_type,
            }
        )


class TestExecutionError(QAOperatorError):
    """Raised when test execution fails."""

    def __init__(
        self,
        message: str,
        test_name: Optional[str] = None,
        exit_code: Optional[int] = None,
    ):
        super().__init__(message, "TEST_EXECUTION_FAILED")
        self.test_name = test_name
        self.exit_code = exit_code
        self.context.update(
            {
                "test_name": test_name,
                "exit_code": exit_code,
            }
        )


class FileOperationError(QAOperatorError):
    """Raised when file system operations fail."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(message, "FILE_OPERATION_FAILED")
        self.file_path = file_path
        self.operation = operation
        self.context.update(
            {
                "file_path": file_path,
                "operation": operation,
            }
        )


class ValidationError(QAOperatorError):
    """Raised when validation fails."""

    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        violations: Optional[list] = None,
    ):
        super().__init__(message, "VALIDATION_FAILED")
        self.validation_type = validation_type
        self.violations = violations or []
        self.context.update(
            {
                "validation_type": validation_type,
                "violations": violations,
            }
        )


class PlanningError(QAOperatorError):
    """Raised when test planning operations fail."""

    def __init__(
        self,
        message: str,
        specification_id: Optional[str] = None,
        planning_stage: Optional[str] = None,
    ):
        super().__init__(message, "PLANNING_FAILED")
        self.specification_id = specification_id
        self.planning_stage = planning_stage
        self.context.update(
            {
                "specification_id": specification_id,
                "planning_stage": planning_stage,
            }
        )


class GitOperationError(QAOperatorError):
    """Raised when Git operations fail."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        repo_path: Optional[str] = None,
        files_affected: Optional[list] = None,
        commit_message: Optional[str] = None,
        remote: Optional[str] = None,
        branch: Optional[str] = None,
        pr_title: Optional[str] = None,
        original_error: Optional[str] = None,
        fallback_error: Optional[str] = None,
    ):
        super().__init__(message, "GIT_OPERATION_FAILED")
        self.operation = operation
        self.repo_path = repo_path
        self.files_affected = files_affected or []
        self.commit_message = commit_message
        self.remote = remote
        self.branch = branch
        self.pr_title = pr_title
        self.original_error = original_error
        self.fallback_error = fallback_error
        self.context.update(
            {
                "operation": operation,
                "repo_path": repo_path,
                "files_affected": files_affected,
                "commit_message": commit_message,
                "remote": remote,
                "branch": branch,
                "pr_title": pr_title,
                "original_error": original_error,
                "fallback_error": fallback_error,
            }
        )
