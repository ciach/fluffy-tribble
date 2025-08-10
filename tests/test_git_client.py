"""
Unit tests for Git MCP Client.

Tests Git operations, availability checking, fallback behavior,
and workflow correlation functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
from datetime import datetime

from orchestrator.mcp.git_client import (
    GitMCPClient,
    GitOperationType,
    GitStatus,
    GitCommitInfo,
    PullRequestInfo,
    GitOperationResult,
)
from orchestrator.core.exceptions import GitOperationError, MCPConnectionError


class TestGitMCPClient:
    """Test cases for GitMCPClient."""

    @pytest.fixture
    def mock_connection_manager(self):
        """Create a mock connection manager."""
        manager = Mock()
        manager.is_connected = Mock(return_value=True)
        manager.connect_server = AsyncMock(return_value=True)
        manager.get_connection = Mock(return_value={"mock": "connection"})
        return manager

    @pytest.fixture
    def git_client(self, mock_connection_manager, tmp_path):
        """Create a GitMCPClient instance for testing."""
        return GitMCPClient(
            connection_manager=mock_connection_manager,
            workflow_id="test-workflow-123",
            repo_path=tmp_path,
        )

    @pytest.fixture
    def mock_git_repo(self, tmp_path):
        """Create a mock Git repository structure."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[core]\n    repositoryformatversion = 0")
        return tmp_path

    def test_initialization(self, mock_connection_manager, tmp_path):
        """Test GitMCPClient initialization."""
        client = GitMCPClient(
            connection_manager=mock_connection_manager,
            workflow_id="test-workflow",
            repo_path=tmp_path,
        )

        assert client.workflow_id == "test-workflow"
        assert client.repo_path == tmp_path
        assert client.server_name == "git"
        assert client._operation_count == 0

    def test_is_available_connected(self, git_client, mock_connection_manager):
        """Test is_available when Git MCP is connected."""
        mock_connection_manager.is_connected.return_value = True
        assert git_client.is_available() is True
        mock_connection_manager.is_connected.assert_called_once_with("git")

    def test_is_available_disconnected(self, git_client, mock_connection_manager):
        """Test is_available when Git MCP is disconnected."""
        mock_connection_manager.is_connected.return_value = False
        assert git_client.is_available() is False

    @pytest.mark.asyncio
    async def test_ensure_connection_success(self, git_client, mock_connection_manager):
        """Test successful connection establishment."""
        mock_connection_manager.is_connected.return_value = True
        connection = await git_client._ensure_connection()
        assert connection == {"mock": "connection"}

    @pytest.mark.asyncio
    async def test_ensure_connection_retry_success(
        self, git_client, mock_connection_manager
    ):
        """Test connection retry on initial failure."""
        mock_connection_manager.is_connected.return_value = False
        mock_connection_manager.connect_server.return_value = True

        connection = await git_client._ensure_connection()
        assert connection == {"mock": "connection"}
        mock_connection_manager.connect_server.assert_called_once_with("git")

    @pytest.mark.asyncio
    async def test_ensure_connection_failure(self, git_client, mock_connection_manager):
        """Test connection failure handling."""
        mock_connection_manager.is_connected.return_value = False
        mock_connection_manager.connect_server.return_value = False

        with pytest.raises(MCPConnectionError) as exc_info:
            await git_client._ensure_connection()

        assert "Git MCP server unavailable" in str(exc_info.value)

    def test_format_commit_message_with_workflow_id(self, git_client):
        """Test commit message formatting with workflow_id."""
        message = "Fix test issues"
        formatted = git_client._format_commit_message(message)

        assert "Fix test issues" in formatted
        assert "Workflow-ID: test-workflow-123" in formatted

    def test_format_commit_message_already_has_workflow_id(self, git_client):
        """Test commit message formatting when workflow_id already present."""
        message = "Fix test issues\n\nWorkflow-ID: test-workflow-123"
        formatted = git_client._format_commit_message(message)

        # Should not duplicate workflow_id
        assert formatted == message

    @pytest.mark.asyncio
    async def test_get_status_success(self, git_client):
        """Test successful Git status retrieval."""
        with patch.object(git_client, "_call_git_tool") as mock_call:
            mock_call.return_value = {
                "success": True,
                "status": {
                    "branch": "main",
                    "staged_files": ["test.py"],
                    "modified_files": ["src/app.py"],
                    "untracked_files": ["new_file.py"],
                    "deleted_files": [],
                    "is_clean": False,
                },
            }

            status = await git_client.get_status()

            assert isinstance(status, GitStatus)
            assert status.branch == "main"
            assert status.staged_files == ["test.py"]
            assert status.modified_files == ["src/app.py"]
            assert status.untracked_files == ["new_file.py"]
            assert status.is_clean is False

            mock_call.assert_called_once_with(GitOperationType.STATUS, {})

    @pytest.mark.asyncio
    async def test_get_status_failure(self, git_client):
        """Test Git status retrieval failure."""
        with patch.object(git_client, "_call_git_tool") as mock_call:
            mock_call.return_value = {"success": False, "error": "Repository not found"}

            with pytest.raises(GitOperationError) as exc_info:
                await git_client.get_status()

            assert "Failed to get Git status" in str(exc_info.value)
            assert exc_info.value.operation == "status"

    @pytest.mark.asyncio
    async def test_stage_files_success(self, git_client):
        """Test successful file staging."""
        files = ["test1.py", "test2.py"]

        with patch.object(git_client, "_call_git_tool") as mock_call:
            mock_call.return_value = {"success": True, "files_added": files}

            result = await git_client.stage_files(files)

            assert isinstance(result, GitOperationResult)
            assert result.success is True
            assert result.operation == GitOperationType.ADD
            assert result.files_affected == files

            mock_call.assert_called_once_with(GitOperationType.ADD, {"files": files})

    @pytest.mark.asyncio
    async def test_stage_files_with_fallback(self, git_client):
        """Test file staging with fallback behavior."""
        files = ["test.py"]

        with patch.object(git_client, "_call_git_tool") as mock_call:
            mock_call.return_value = {
                "success": False,
                "error": "Git MCP unavailable",
                "fallback_used": True,
            }

            result = await git_client.stage_files(files)

            assert result.success is False
            assert result.error == "Git MCP unavailable"

    @pytest.mark.asyncio
    async def test_commit_changes_success(self, git_client):
        """Test successful commit operation."""
        message = "Fix test issues"
        files = ["test.py"]

        with patch.object(git_client, "_call_git_tool") as mock_call:
            mock_call.return_value = {
                "success": True,
                "commit_hash": "abc123def456",
                "files_changed": files,
            }

            result = await git_client.commit_changes(message, files)

            assert isinstance(result, GitOperationResult)
            assert result.success is True
            assert result.operation == GitOperationType.COMMIT
            assert result.commit_hash == "abc123def456"
            assert result.files_affected == files

            # Verify workflow_id was added to commit message
            call_args = mock_call.call_args[0][1]
            assert "Workflow-ID: test-workflow-123" in call_args["message"]

    @pytest.mark.asyncio
    async def test_commit_changes_failure(self, git_client):
        """Test commit operation failure."""
        message = "Fix test issues"

        with patch.object(git_client, "_call_git_tool") as mock_call:
            mock_call.return_value = {"success": False, "error": "Nothing to commit"}

            with pytest.raises(GitOperationError) as exc_info:
                await git_client.commit_changes(message)

            assert "Failed to commit changes" in str(exc_info.value)
            assert exc_info.value.operation == "commit"

    @pytest.mark.asyncio
    async def test_push_changes_success(self, git_client):
        """Test successful push operation."""
        with patch.object(git_client, "_call_git_tool") as mock_call:
            mock_call.return_value = {
                "success": True,
                "pushed_commits": 2,
                "remote": "origin",
                "branch": "main",
            }

            result = await git_client.push_changes("origin", "main")

            assert isinstance(result, GitOperationResult)
            assert result.success is True
            assert result.operation == GitOperationType.PUSH

            mock_call.assert_called_once_with(
                GitOperationType.PUSH, {"remote": "origin", "branch": "main"}
            )

    @pytest.mark.asyncio
    async def test_create_pull_request_success(self, git_client):
        """Test successful pull request creation."""
        pr_info = PullRequestInfo(
            title="Fix test issues",
            description="This PR fixes various test issues",
            source_branch="feature/fix-tests",
            target_branch="main",
            labels=["bug", "tests"],
        )

        with patch.object(git_client, "_call_git_tool") as mock_call:
            mock_call.return_value = {
                "success": True,
                "pr_url": "https://github.com/repo/pull/123",
            }

            result = await git_client.create_pull_request(pr_info)

            assert isinstance(result, GitOperationResult)
            assert result.success is True
            assert result.pr_url == "https://github.com/repo/pull/123"

            # Verify workflow context was added to description
            call_args = mock_call.call_args[0][1]
            assert "**Workflow ID:** `test-workflow-123`" in call_args["description"]

    @pytest.mark.asyncio
    async def test_create_pull_request_without_workflow_context(self, git_client):
        """Test pull request creation without workflow context."""
        pr_info = PullRequestInfo(
            title="Fix test issues",
            description="This PR fixes various test issues",
            source_branch="feature/fix-tests",
        )

        with patch.object(git_client, "_call_git_tool") as mock_call:
            mock_call.return_value = {"success": True}

            await git_client.create_pull_request(
                pr_info, include_workflow_context=False
            )

            call_args = mock_call.call_args[0][1]
            assert call_args["description"] == pr_info.description

    @pytest.mark.asyncio
    async def test_get_commit_history_success(self, git_client):
        """Test successful commit history retrieval."""
        with patch.object(git_client, "_call_git_tool") as mock_call:
            mock_call.return_value = {
                "success": True,
                "commits": [
                    {
                        "hash": "abc123def456",
                        "message": "Fix test issues",
                        "author": "Test User <test@example.com>",
                        "date": "2024-01-15T10:30:00",
                        "files_changed": ["test.py"],
                    }
                ],
            }

            commits = await git_client.get_commit_history(limit=5)

            assert len(commits) == 1
            assert isinstance(commits[0], GitCommitInfo)
            assert commits[0].hash == "abc123def456"
            assert commits[0].message == "Fix test issues"
            assert commits[0].files_changed == ["test.py"]

    @pytest.mark.asyncio
    async def test_get_commit_history_with_fallback(self, git_client):
        """Test commit history retrieval with fallback."""
        with patch.object(git_client, "_call_git_tool") as mock_call:
            mock_call.return_value = {
                "success": False,
                "fallback_used": True,
                "error": "Git MCP unavailable",
            }

            commits = await git_client.get_commit_history()

            assert commits == []

    @pytest.mark.asyncio
    async def test_get_diff_success(self, git_client):
        """Test successful diff retrieval."""
        with patch.object(git_client, "_call_git_tool") as mock_call:
            mock_call.return_value = {
                "success": True,
                "diff": "--- a/test.py\n+++ b/test.py\n@@ -1,3 +1,4 @@\n+# New line\n def test():\n     pass",
            }

            diff = await git_client.get_diff("HEAD~1", "HEAD", ["test.py"])

            assert "--- a/test.py" in diff
            assert "+# New line" in diff

            mock_call.assert_called_once_with(
                GitOperationType.DIFF,
                {"base": "HEAD~1", "target": "HEAD", "files": ["test.py"]},
            )

    @pytest.mark.asyncio
    async def test_stage_and_commit_files_success(self, git_client):
        """Test convenience method for staging and committing files."""
        files = ["test.py"]
        message = "Fix test"

        with patch.object(git_client, "stage_files") as mock_stage, patch.object(
            git_client, "commit_changes"
        ) as mock_commit:
            mock_stage.return_value = GitOperationResult(
                success=True,
                operation=GitOperationType.ADD,
                files_affected=files,
                duration=0.1,
            )
            mock_commit.return_value = GitOperationResult(
                success=True,
                operation=GitOperationType.COMMIT,
                files_affected=files,
                commit_hash="abc123",
                duration=0.2,
            )

            result = await git_client.stage_and_commit_files(files, message)

            assert result.success is True
            assert result.commit_hash == "abc123"

            mock_stage.assert_called_once_with(files)
            mock_commit.assert_called_once_with(message, files)

    @pytest.mark.asyncio
    async def test_stage_and_commit_files_stage_failure(self, git_client):
        """Test convenience method when staging fails."""
        files = ["test.py"]
        message = "Fix test"

        with patch.object(git_client, "stage_files") as mock_stage:
            mock_stage.return_value = GitOperationResult(
                success=False,
                operation=GitOperationType.ADD,
                error="Staging failed",
                duration=0.1,
            )

            result = await git_client.stage_and_commit_files(files, message)

            assert result.success is False
            assert result.error == "Staging failed"

    def test_get_workflow_id(self, git_client):
        """Test workflow ID retrieval."""
        assert git_client.get_workflow_id() == "test-workflow-123"

    def test_get_operation_stats(self, git_client):
        """Test operation statistics retrieval."""
        stats = git_client.get_operation_stats()

        assert stats["total_operations"] == 0
        assert stats["workflow_id"] == "test-workflow-123"
        assert "repo_path" in stats
        assert "mcp_available" in stats

    @pytest.mark.asyncio
    async def test_validate_repository_success(self, git_client, mock_git_repo):
        """Test successful repository validation."""
        git_client.repo_path = mock_git_repo

        with patch.object(git_client, "get_status") as mock_status:
            mock_status.return_value = GitStatus(
                branch="main",
                staged_files=[],
                modified_files=[],
                untracked_files=[],
                deleted_files=[],
                is_clean=True,
            )

            result = await git_client.validate_repository()

            assert result.valid is True
            assert len(result.errors) == 0
            assert result.context["current_branch"] == "main"
            assert result.context["is_clean"] is True

    @pytest.mark.asyncio
    async def test_validate_repository_not_git_repo(self, git_client, tmp_path):
        """Test repository validation when not a Git repository."""
        git_client.repo_path = tmp_path

        result = await git_client.validate_repository()

        assert result.valid is False
        assert any("Not a Git repository" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_validate_repository_path_not_exists(self, git_client):
        """Test repository validation when path doesn't exist."""
        git_client.repo_path = Path("/nonexistent/path")

        result = await git_client.validate_repository()

        assert result.valid is False
        assert any("does not exist" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_validate_repository_mcp_unavailable(
        self, git_client, mock_git_repo, mock_connection_manager
    ):
        """Test repository validation when MCP is unavailable."""
        git_client.repo_path = mock_git_repo
        mock_connection_manager.is_connected.return_value = False

        with patch.object(git_client, "get_status") as mock_status:
            mock_status.return_value = GitStatus(
                branch="main",
                staged_files=[],
                modified_files=[],
                untracked_files=[],
                deleted_files=[],
                is_clean=True,
            )

            result = await git_client.validate_repository()

            assert result.valid is True
            assert any(
                "Git MCP server not available" in warning for warning in result.warnings
            )

    @pytest.mark.asyncio
    async def test_fallback_git_operation(self, git_client):
        """Test fallback behavior when MCP is unavailable."""
        result = await git_client._fallback_git_operation(
            GitOperationType.STATUS, {"test": "args"}
        )

        assert result["success"] is False
        assert "Git MCP unavailable" in result["error"]
        assert result["fallback_used"] is True

    @pytest.mark.asyncio
    async def test_call_git_tool_with_mcp_failure_and_fallback(self, git_client):
        """Test Git tool call with MCP failure and successful fallback."""
        with patch.object(
            git_client, "_ensure_connection"
        ) as mock_ensure, patch.object(
            git_client, "_fallback_git_operation"
        ) as mock_fallback:
            mock_ensure.side_effect = Exception("MCP connection failed")
            mock_fallback.return_value = {"success": False, "fallback_used": True}

            result = await git_client._call_git_tool(GitOperationType.STATUS, {})

            assert result["fallback_used"] is True
            mock_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_git_tool_with_both_failures(self, git_client):
        """Test Git tool call when both MCP and fallback fail."""
        with patch.object(
            git_client, "_ensure_connection"
        ) as mock_ensure, patch.object(
            git_client, "_fallback_git_operation"
        ) as mock_fallback:
            mock_ensure.side_effect = Exception("MCP connection failed")
            mock_fallback.side_effect = Exception("Fallback failed")

            with pytest.raises(GitOperationError) as exc_info:
                await git_client._call_git_tool(GitOperationType.STATUS, {})

            assert "Git operation failed" in str(exc_info.value)
            assert exc_info.value.operation == "status"


class TestGitModels:
    """Test cases for Git-related Pydantic models."""

    def test_git_commit_info_validation(self):
        """Test GitCommitInfo model validation."""
        commit = GitCommitInfo(
            hash="abc123def456",
            message="Fix test issues",
            author="Test User <test@example.com>",
            date=datetime.now(),
            files_changed=["test.py", "src/app.py"],
        )

        assert commit.hash == "abc123def456"
        assert commit.message == "Fix test issues"
        assert len(commit.files_changed) == 2

    def test_git_commit_info_invalid_hash(self):
        """Test GitCommitInfo validation with invalid hash."""
        with pytest.raises(ValueError) as exc_info:
            GitCommitInfo(
                hash="abc",  # Too short
                message="Fix test issues",
                author="Test User",
                date=datetime.now(),
            )

        assert "at least 7 characters" in str(exc_info.value)

    def test_git_commit_info_empty_message(self):
        """Test GitCommitInfo validation with empty message."""
        with pytest.raises(ValueError) as exc_info:
            GitCommitInfo(
                hash="abc123def456",
                message="",  # Empty message
                author="Test User",
                date=datetime.now(),
            )

        assert "cannot be empty" in str(exc_info.value)

    def test_git_status_model(self):
        """Test GitStatus model creation."""
        status = GitStatus(
            branch="feature/test",
            staged_files=["test.py"],
            modified_files=["src/app.py"],
            untracked_files=["new_file.py"],
            deleted_files=[],
            is_clean=False,
            ahead_behind={"ahead": 2, "behind": 1},
        )

        assert status.branch == "feature/test"
        assert len(status.staged_files) == 1
        assert status.is_clean is False
        assert status.ahead_behind["ahead"] == 2

    def test_pull_request_info_validation(self):
        """Test PullRequestInfo model validation."""
        pr_info = PullRequestInfo(
            title="Fix test issues",
            description="This PR fixes various test issues found during QA",
            source_branch="feature/fix-tests",
            target_branch="develop",
            labels=["bug", "tests"],
            reviewers=["reviewer1", "reviewer2"],
            draft=True,
        )

        assert pr_info.title == "Fix test issues"
        assert pr_info.target_branch == "develop"
        assert len(pr_info.labels) == 2
        assert pr_info.draft is True

    def test_pull_request_info_empty_title(self):
        """Test PullRequestInfo validation with empty title."""
        with pytest.raises(ValueError) as exc_info:
            PullRequestInfo(
                title="",  # Empty title
                description="Description",
                source_branch="feature/test",
            )

        assert "cannot be empty" in str(exc_info.value)

    def test_git_operation_result_model(self):
        """Test GitOperationResult model creation."""
        result = GitOperationResult(
            success=True,
            operation=GitOperationType.COMMIT,
            output="Committed 2 files",
            duration=1.5,
            files_affected=["test.py", "src/app.py"],
            commit_hash="abc123def456",
        )

        assert result.success is True
        assert result.operation == GitOperationType.COMMIT
        assert result.commit_hash == "abc123def456"
        assert len(result.files_affected) == 2
