"""
Git MCP Client Wrapper.

Provides Git operations through MCP integration with availability checking,
fallback behavior, and workflow correlation through workflow_id.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator, ConfigDict

from ..core.exceptions import GitOperationError, MCPConnectionError, ValidationError
from .models import MCPToolCall, MCPToolResponse, ValidationResult


class GitOperationType(Enum):
    """Git operation types."""
    
    STATUS = "status"
    ADD = "add"
    COMMIT = "commit"
    PUSH = "push"
    PULL = "pull"
    BRANCH = "branch"
    CHECKOUT = "checkout"
    MERGE = "merge"
    DIFF = "diff"
    LOG = "log"
    REMOTE = "remote"


class GitCommitInfo(BaseModel):
    """Information about a Git commit."""
    
    model_config = ConfigDict(extra="forbid")
    
    hash: str = Field(..., description="Commit hash")
    message: str = Field(..., description="Commit message")
    author: str = Field(..., description="Commit author")
    date: datetime = Field(..., description="Commit date")
    files_changed: List[str] = Field(default_factory=list, description="Changed files")
    
    @validator("hash")
    def validate_hash(cls, v):
        if not v or len(v) < 7:
            raise ValueError("Commit hash must be at least 7 characters")
        return v
    
    @validator("message")
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError("Commit message cannot be empty")
        return v.strip()


class GitStatus(BaseModel):
    """Git repository status."""
    
    model_config = ConfigDict(extra="forbid")
    
    branch: str = Field(..., description="Current branch name")
    staged_files: List[str] = Field(default_factory=list, description="Staged files")
    modified_files: List[str] = Field(default_factory=list, description="Modified files")
    untracked_files: List[str] = Field(default_factory=list, description="Untracked files")
    deleted_files: List[str] = Field(default_factory=list, description="Deleted files")
    is_clean: bool = Field(..., description="Whether working directory is clean")
    ahead_behind: Optional[Dict[str, int]] = Field(None, description="Commits ahead/behind remote")


class PullRequestInfo(BaseModel):
    """Information for creating a pull request."""
    
    model_config = ConfigDict(extra="forbid")
    
    title: str = Field(..., description="PR title")
    description: str = Field(..., description="PR description")
    source_branch: str = Field(..., description="Source branch")
    target_branch: str = Field("main", description="Target branch")
    labels: List[str] = Field(default_factory=list, description="PR labels")
    reviewers: List[str] = Field(default_factory=list, description="Requested reviewers")
    draft: bool = Field(False, description="Create as draft PR")
    
    @validator("title")
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError("PR title cannot be empty")
        return v.strip()
    
    @validator("description")
    def validate_description(cls, v):
        if not v or not v.strip():
            raise ValueError("PR description cannot be empty")
        return v.strip()


class GitOperationResult(BaseModel):
    """Result of a Git operation."""
    
    model_config = ConfigDict(extra="forbid")
    
    success: bool = Field(..., description="Whether operation succeeded")
    operation: GitOperationType = Field(..., description="Operation performed")
    output: Optional[str] = Field(None, description="Command output")
    error: Optional[str] = Field(None, description="Error message if failed")
    duration: Optional[float] = Field(None, ge=0, description="Operation duration")
    files_affected: List[str] = Field(default_factory=list, description="Files affected by operation")
    commit_hash: Optional[str] = Field(None, description="Commit hash if applicable")
    pr_url: Optional[str] = Field(None, description="PR URL if created")


class GitMCPClient:
    """
    Wrapper class for Git MCP tool calls.
    
    Provides Git operations with availability checking, fallback behavior,
    and workflow correlation through workflow_id inclusion in commit messages.
    """
    
    def __init__(
        self, 
        connection_manager, 
        workflow_id: str,
        repo_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Git MCP client.
        
        Args:
            connection_manager: MCP connection manager instance
            workflow_id: Unique workflow identifier for correlation
            repo_path: Path to Git repository (defaults to current directory)
            logger: Optional logger instance
        """
        self.connection_manager = connection_manager
        self.workflow_id = workflow_id
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.logger = logger or logging.getLogger(__name__)
        self.server_name = "git"
        
        # Git operation state
        self._last_operation_time = None
        self._operation_count = 0
        
    def is_available(self) -> bool:
        """
        Check if Git MCP server is available.
        
        Returns:
            True if Git MCP server is connected and available
        """
        return self.connection_manager.is_connected(self.server_name)
    
    async def _ensure_connection(self) -> Any:
        """Ensure we have a valid connection to the Git MCP server."""
        if not self.is_available():
            self.logger.warning("Git MCP server not connected, attempting to connect...")
            try:
                success = await self.connection_manager.connect_server(self.server_name)
                if not success:
                    raise MCPConnectionError(
                        "Failed to connect to Git MCP server",
                        server_name=self.server_name,
                    )
            except Exception as e:
                self.logger.error(f"Git MCP connection failed: {e}")
                raise MCPConnectionError(
                    f"Git MCP server unavailable: {e}",
                    server_name=self.server_name,
                )
        
        return self.connection_manager.get_connection(self.server_name)
    
    async def _call_git_tool(
        self, 
        operation: GitOperationType, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a Git MCP tool with error handling and fallback.
        
        Args:
            operation: Git operation type
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool response data
            
        Raises:
            GitOperationError: If operation fails and no fallback available
        """
        if not self.is_available():
            self.logger.warning(f"Git MCP unavailable for operation: {operation.value}")
            return await self._fallback_git_operation(operation, arguments)
        
        try:
            await self._ensure_connection()
            
            # Add repository path to arguments
            arguments["repo_path"] = str(self.repo_path)
            
            self.logger.debug(f"Calling Git MCP tool: {operation.value} with args: {arguments}")
            
            # TODO: Implement actual MCP tool call
            # This is a placeholder for the actual MCP client call
            await asyncio.sleep(0.1)  # Simulate tool call delay
            
            # Mock response based on operation type
            response = await self._mock_git_response(operation, arguments)
            
            self._operation_count += 1
            self._last_operation_time = time.time()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Git MCP tool call failed: {operation.value} - {e}")
            
            # Try fallback if available
            try:
                return await self._fallback_git_operation(operation, arguments)
            except Exception as fallback_error:
                raise GitOperationError(
                    f"Git operation failed: {operation.value}",
                    operation=operation.value,
                    repo_path=str(self.repo_path),
                    original_error=str(e),
                    fallback_error=str(fallback_error)
                )
    
    async def _mock_git_response(
        self, 
        operation: GitOperationType, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock Git responses for testing purposes."""
        
        if operation == GitOperationType.STATUS:
            return {
                "success": True,
                "status": {
                    "branch": "main",
                    "staged_files": [],
                    "modified_files": ["test.py"],
                    "untracked_files": [],
                    "deleted_files": [],
                    "is_clean": False
                }
            }
        
        elif operation == GitOperationType.ADD:
            return {
                "success": True,
                "files_added": arguments.get("files", [])
            }
        
        elif operation == GitOperationType.COMMIT:
            return {
                "success": True,
                "commit_hash": "abc123def456",
                "message": arguments.get("message", ""),
                "files_changed": arguments.get("files", [])
            }
        
        elif operation == GitOperationType.PUSH:
            return {
                "success": True,
                "pushed_commits": 1,
                "remote": arguments.get("remote", "origin"),
                "branch": arguments.get("branch", "main")
            }
        
        else:
            return {"success": True}
    
    async def _fallback_git_operation(
        self, 
        operation: GitOperationType, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fallback Git operations when MCP is unavailable.
        
        Args:
            operation: Git operation type
            arguments: Operation arguments
            
        Returns:
            Operation result
        """
        self.logger.info(f"Using fallback for Git operation: {operation.value}")
        
        # For now, return a mock success response with warning
        # In a real implementation, this could use subprocess to call git directly
        return {
            "success": False,
            "error": f"Git MCP unavailable, operation {operation.value} skipped",
            "fallback_used": True
        }
    
    def _format_commit_message(self, message: str) -> str:
        """
        Format commit message with workflow_id for correlation.
        
        Args:
            message: Original commit message
            
        Returns:
            Formatted commit message with workflow_id
        """
        if self.workflow_id and self.workflow_id not in message:
            return f"{message}\n\nWorkflow-ID: {self.workflow_id}"
        return message
    
    async def get_status(self) -> GitStatus:
        """
        Get Git repository status.
        
        Returns:
            Git status information
        """
        self.logger.debug("Getting Git repository status")
        
        response = await self._call_git_tool(GitOperationType.STATUS, {})
        
        if not response.get("success"):
            raise GitOperationError(
                f"Failed to get Git status: {response.get('error', 'Unknown error')}",
                operation="status",
                repo_path=str(self.repo_path)
            )
        
        status_data = response.get("status", {})
        return GitStatus(
            branch=status_data.get("branch", "unknown"),
            staged_files=status_data.get("staged_files", []),
            modified_files=status_data.get("modified_files", []),
            untracked_files=status_data.get("untracked_files", []),
            deleted_files=status_data.get("deleted_files", []),
            is_clean=status_data.get("is_clean", True),
            ahead_behind=status_data.get("ahead_behind")
        )
    
    async def stage_files(self, files: List[str]) -> GitOperationResult:
        """
        Stage files for commit.
        
        Args:
            files: List of file paths to stage
            
        Returns:
            Git operation result
        """
        start_time = time.time()
        
        self.logger.info(f"Staging {len(files)} files: {files}")
        
        response = await self._call_git_tool(
            GitOperationType.ADD, 
            {"files": files}
        )
        
        success = response.get("success", False)
        if not success and not response.get("fallback_used"):
            raise GitOperationError(
                f"Failed to stage files: {response.get('error', 'Unknown error')}",
                operation="add",
                repo_path=str(self.repo_path),
                files_affected=files
            )
        
        return GitOperationResult(
            success=success,
            operation=GitOperationType.ADD,
            output=response.get("output"),
            error=response.get("error"),
            duration=time.time() - start_time,
            files_affected=files
        )
    
    async def commit_changes(
        self, 
        message: str, 
        files: Optional[List[str]] = None
    ) -> GitOperationResult:
        """
        Commit staged changes with workflow_id correlation.
        
        Args:
            message: Commit message
            files: Optional list of specific files to commit
            
        Returns:
            Git operation result with commit hash
        """
        start_time = time.time()
        
        # Format message with workflow_id
        formatted_message = self._format_commit_message(message)
        
        self.logger.info(f"Committing changes: {message}")
        
        commit_args = {"message": formatted_message}
        if files:
            commit_args["files"] = files
        
        response = await self._call_git_tool(GitOperationType.COMMIT, commit_args)
        
        success = response.get("success", False)
        if not success and not response.get("fallback_used"):
            raise GitOperationError(
                f"Failed to commit changes: {response.get('error', 'Unknown error')}",
                operation="commit",
                repo_path=str(self.repo_path),
                commit_message=formatted_message
            )
        
        return GitOperationResult(
            success=success,
            operation=GitOperationType.COMMIT,
            output=response.get("output"),
            error=response.get("error"),
            duration=time.time() - start_time,
            files_affected=response.get("files_changed", files or []),
            commit_hash=response.get("commit_hash")
        )
    
    async def push_changes(
        self, 
        remote: str = "origin", 
        branch: Optional[str] = None
    ) -> GitOperationResult:
        """
        Push changes to remote repository.
        
        Args:
            remote: Remote name (default: origin)
            branch: Branch name (default: current branch)
            
        Returns:
            Git operation result
        """
        start_time = time.time()
        
        self.logger.info(f"Pushing changes to {remote}/{branch or 'current branch'}")
        
        push_args = {"remote": remote}
        if branch:
            push_args["branch"] = branch
        
        response = await self._call_git_tool(GitOperationType.PUSH, push_args)
        
        success = response.get("success", False)
        if not success and not response.get("fallback_used"):
            raise GitOperationError(
                f"Failed to push changes: {response.get('error', 'Unknown error')}",
                operation="push",
                repo_path=str(self.repo_path),
                remote=remote,
                branch=branch
            )
        
        return GitOperationResult(
            success=success,
            operation=GitOperationType.PUSH,
            output=response.get("output"),
            error=response.get("error"),
            duration=time.time() - start_time
        )
    
    async def create_pull_request(
        self, 
        pr_info: PullRequestInfo,
        include_workflow_context: bool = True
    ) -> GitOperationResult:
        """
        Create a pull request with test information and workflow context.
        
        Args:
            pr_info: Pull request information
            include_workflow_context: Whether to include workflow_id in PR description
            
        Returns:
            Git operation result with PR URL
        """
        start_time = time.time()
        
        # Enhance PR description with workflow context
        description = pr_info.description
        if include_workflow_context and self.workflow_id:
            description += f"\n\n---\n**Workflow ID:** `{self.workflow_id}`\n"
            description += f"**Generated:** {datetime.now().isoformat()}\n"
            description += "**Type:** Automated QA Operator Test Updates"
        
        self.logger.info(f"Creating pull request: {pr_info.title}")
        
        pr_args = {
            "title": pr_info.title,
            "description": description,
            "source_branch": pr_info.source_branch,
            "target_branch": pr_info.target_branch,
            "labels": pr_info.labels,
            "reviewers": pr_info.reviewers,
            "draft": pr_info.draft
        }
        
        # Use a mock operation type for PR creation
        response = await self._call_git_tool(GitOperationType.REMOTE, pr_args)
        
        success = response.get("success", False)
        if not success and not response.get("fallback_used"):
            raise GitOperationError(
                f"Failed to create pull request: {response.get('error', 'Unknown error')}",
                operation="create_pr",
                repo_path=str(self.repo_path),
                pr_title=pr_info.title
            )
        
        return GitOperationResult(
            success=success,
            operation=GitOperationType.REMOTE,
            output=response.get("output"),
            error=response.get("error"),
            duration=time.time() - start_time,
            pr_url=response.get("pr_url")
        )
    
    async def get_commit_history(
        self, 
        limit: int = 10, 
        since: Optional[datetime] = None
    ) -> List[GitCommitInfo]:
        """
        Get commit history.
        
        Args:
            limit: Maximum number of commits to retrieve
            since: Optional date to filter commits since
            
        Returns:
            List of commit information
        """
        self.logger.debug(f"Getting commit history (limit: {limit})")
        
        log_args = {"limit": limit}
        if since:
            log_args["since"] = since.isoformat()
        
        response = await self._call_git_tool(GitOperationType.LOG, log_args)
        
        if not response.get("success"):
            if response.get("fallback_used"):
                self.logger.warning("Git log unavailable, returning empty history")
                return []
            
            raise GitOperationError(
                f"Failed to get commit history: {response.get('error', 'Unknown error')}",
                operation="log",
                repo_path=str(self.repo_path)
            )
        
        commits_data = response.get("commits", [])
        commits = []
        
        for commit_data in commits_data:
            try:
                commit = GitCommitInfo(
                    hash=commit_data["hash"],
                    message=commit_data["message"],
                    author=commit_data["author"],
                    date=datetime.fromisoformat(commit_data["date"]),
                    files_changed=commit_data.get("files_changed", [])
                )
                commits.append(commit)
            except Exception as e:
                self.logger.warning(f"Failed to parse commit data: {e}")
                continue
        
        return commits
    
    async def get_diff(
        self, 
        base: Optional[str] = None, 
        target: Optional[str] = None,
        files: Optional[List[str]] = None
    ) -> str:
        """
        Get diff between commits or working directory.
        
        Args:
            base: Base commit/branch (default: HEAD)
            target: Target commit/branch (default: working directory)
            files: Optional list of specific files
            
        Returns:
            Diff output as string
        """
        self.logger.debug(f"Getting diff: {base} -> {target}")
        
        diff_args = {}
        if base:
            diff_args["base"] = base
        if target:
            diff_args["target"] = target
        if files:
            diff_args["files"] = files
        
        response = await self._call_git_tool(GitOperationType.DIFF, diff_args)
        
        if not response.get("success"):
            if response.get("fallback_used"):
                self.logger.warning("Git diff unavailable, returning empty diff")
                return ""
            
            raise GitOperationError(
                f"Failed to get diff: {response.get('error', 'Unknown error')}",
                operation="diff",
                repo_path=str(self.repo_path)
            )
        
        return response.get("diff", "")
    
    async def stage_and_commit_files(
        self, 
        files: List[str], 
        message: str
    ) -> GitOperationResult:
        """
        Convenience method to stage and commit files in one operation.
        
        Args:
            files: List of files to stage and commit
            message: Commit message
            
        Returns:
            Git operation result from commit
        """
        # Stage files first
        stage_result = await self.stage_files(files)
        
        if not stage_result.success:
            return stage_result
        
        # Then commit
        return await self.commit_changes(message, files)
    
    def get_workflow_id(self) -> str:
        """
        Get the current workflow ID.
        
        Returns:
            Workflow ID string
        """
        return self.workflow_id
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about Git operations performed.
        
        Returns:
            Dictionary with operation statistics
        """
        return {
            "total_operations": self._operation_count,
            "last_operation_time": self._last_operation_time,
            "workflow_id": self.workflow_id,
            "repo_path": str(self.repo_path),
            "mcp_available": self.is_available()
        }
    
    async def validate_repository(self) -> ValidationResult:
        """
        Validate that the repository is properly configured for Git operations.
        
        Returns:
            Validation result with any issues found
        """
        result = ValidationResult(valid=True)
        
        # Check if directory exists
        if not self.repo_path.exists():
            result.add_error(f"Repository path does not exist: {self.repo_path}")
            return result
        
        # Check if it's a Git repository
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            result.add_error(f"Not a Git repository: {self.repo_path}")
            return result
        
        # Check Git MCP availability
        if not self.is_available():
            result.add_warning("Git MCP server not available - operations will use fallback")
        
        # Try to get status to verify Git is working
        try:
            status = await self.get_status()
            result.context["current_branch"] = status.branch
            result.context["is_clean"] = status.is_clean
            result.context["modified_files"] = len(status.modified_files)
        except Exception as e:
            result.add_error(f"Failed to get Git status: {e}")
        
        return result