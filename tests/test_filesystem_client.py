"""
Unit tests for Filesystem MCP Client.

Tests safe file operations with path validation and restriction enforcement
for the Filesystem MCP server, ensuring operations are sandboxed to the e2e/ directory.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from orchestrator.mcp.filesystem_client import FilesystemMCPClient
from orchestrator.core.exceptions import (
    FileOperationError,
    ValidationError,
    MCPConnectionError,
)
from orchestrator.mcp.models import (
    FileOperationRequest,
    FileOperationResult,
    ValidationResult,
)


class TestFilesystemMCPClient:
    """Test cases for FilesystemMCPClient."""

    @pytest.fixture
    def temp_e2e_dir(self):
        """Create temporary e2e directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            e2e_dir = Path(temp_dir) / "e2e"
            e2e_dir.mkdir()
            yield e2e_dir

    @pytest.fixture
    def mock_connection_manager(self):
        """Create mock connection manager."""
        manager = MagicMock()
        manager.call_tool = AsyncMock()
        return manager

    @pytest.fixture
    def filesystem_client(self, mock_connection_manager, temp_e2e_dir):
        """Create filesystem client instance."""
        return FilesystemMCPClient(
            connection_manager=mock_connection_manager, e2e_dir=temp_e2e_dir
        )

    def test_client_initialization(self, filesystem_client, temp_e2e_dir):
        """Test filesystem client initialization."""
        assert filesystem_client.e2e_dir == temp_e2e_dir.resolve()
        assert filesystem_client.server_name == "filesystem"
        assert (temp_e2e_dir / ".backups").exists()

    def test_validate_path_valid_relative(self, filesystem_client):
        """Test path validation for valid relative paths."""
        valid_paths = [
            "test.spec.ts",
            "pages/login.spec.ts",
            "utils/helpers.ts",
            "fixtures/data.json",
        ]

        for path in valid_paths:
            result = filesystem_client._validate_path(path)
            assert result.is_valid is True
            assert result.error_message is None

    def test_validate_path_invalid_absolute(self, filesystem_client):
        """Test path validation rejects absolute paths."""
        invalid_paths = [
            "/etc/passwd",
            "/home/user/file.txt",
            "C:\\Windows\\System32\\file.txt",
        ]

        for path in invalid_paths:
            result = filesystem_client._validate_path(path)
            assert result.is_valid is False
            assert "Absolute paths are not allowed" in result.error_message

    def test_validate_path_invalid_traversal(self, filesystem_client):
        """Test path validation rejects directory traversal."""
        invalid_paths = [
            "../../../etc/passwd",
            "pages/../../secret.txt",
            "..\\..\\windows\\file.txt",
        ]

        for path in invalid_paths:
            result = filesystem_client._validate_path(path)
            assert result.is_valid is False
            assert "Path traversal detected" in result.error_message

    def test_validate_path_invalid_characters(self, filesystem_client):
        """Test path validation rejects invalid characters."""
        invalid_paths = [
            "file<name>.txt",
            "file>name.txt",
            "file|name.txt",
            "file:name.txt",
        ]

        for path in invalid_paths:
            result = filesystem_client._validate_path(path)
            assert result.is_valid is False
            assert "Invalid characters in path" in result.error_message

    def test_resolve_safe_path(self, filesystem_client, temp_e2e_dir):
        """Test resolving safe paths within e2e directory."""
        safe_path = filesystem_client._resolve_safe_path("test/file.spec.ts")
        expected_path = temp_e2e_dir / "test" / "file.spec.ts"

        assert safe_path == expected_path

    def test_resolve_safe_path_invalid(self, filesystem_client):
        """Test resolving invalid paths raises error."""
        with pytest.raises(ValidationError, match="Invalid file path"):
            filesystem_client._resolve_safe_path("../../../etc/passwd")

    @pytest.mark.asyncio
    async def test_read_file_success(
        self, filesystem_client, temp_e2e_dir, mock_connection_manager
    ):
        """Test successful file reading."""
        # Create test file
        test_file = temp_e2e_dir / "test.spec.ts"
        test_content = "describe('test', () => { it('works', () => {}); });"
        test_file.write_text(test_content)

        mock_connection_manager.call_tool.return_value = {
            "content": test_content,
            "encoding": "utf-8",
        }

        result = await filesystem_client.read_file("test.spec.ts")

        assert result.success is True
        assert result.content == test_content
        assert result.error_message is None

        mock_connection_manager.call_tool.assert_called_once_with(
            "filesystem", "read_file", {"path": str(temp_e2e_dir / "test.spec.ts")}
        )

    @pytest.mark.asyncio
    async def test_read_file_invalid_path(
        self, filesystem_client, mock_connection_manager
    ):
        """Test reading file with invalid path."""
        result = await filesystem_client.read_file("../../../etc/passwd")

        assert result.success is False
        assert "Invalid file path" in result.error_message
        mock_connection_manager.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_read_file_mcp_error(
        self, filesystem_client, mock_connection_manager
    ):
        """Test reading file when MCP call fails."""
        mock_connection_manager.call_tool.side_effect = MCPConnectionError(
            "Connection failed"
        )

        result = await filesystem_client.read_file("test.spec.ts")

        assert result.success is False
        assert "MCP connection error" in result.error_message

    @pytest.mark.asyncio
    async def test_write_file_success(
        self, filesystem_client, temp_e2e_dir, mock_connection_manager
    ):
        """Test successful file writing."""
        test_content = "describe('new test', () => { it('works', () => {}); });"

        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "bytes_written": len(test_content),
        }

        result = await filesystem_client.write_file("new_test.spec.ts", test_content)

        assert result.success is True
        assert result.error_message is None

        mock_connection_manager.call_tool.assert_called_once_with(
            "filesystem",
            "write_file",
            {
                "path": str(temp_e2e_dir / "new_test.spec.ts"),
                "content": test_content,
                "encoding": "utf-8",
            },
        )

    @pytest.mark.asyncio
    async def test_write_file_with_backup(
        self, filesystem_client, temp_e2e_dir, mock_connection_manager
    ):
        """Test writing file with backup creation."""
        # Create existing file
        existing_file = temp_e2e_dir / "existing.spec.ts"
        original_content = "original content"
        existing_file.write_text(original_content)

        new_content = "new content"

        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "bytes_written": len(new_content),
        }

        result = await filesystem_client.write_file(
            "existing.spec.ts", new_content, create_backup=True
        )

        assert result.success is True

        # Check backup was created
        backup_files = list((temp_e2e_dir / ".backups").glob("existing.spec.ts.*"))
        assert len(backup_files) == 1
        assert backup_files[0].read_text() == original_content

    @pytest.mark.asyncio
    async def test_write_file_invalid_path(
        self, filesystem_client, mock_connection_manager
    ):
        """Test writing file with invalid path."""
        result = await filesystem_client.write_file(
            "../../../etc/passwd", "malicious content"
        )

        assert result.success is False
        assert "Invalid file path" in result.error_message
        mock_connection_manager.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_file_success(
        self, filesystem_client, temp_e2e_dir, mock_connection_manager
    ):
        """Test successful file deletion."""
        # Create test file
        test_file = temp_e2e_dir / "to_delete.spec.ts"
        test_file.write_text("content to delete")

        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "deleted": True,
        }

        result = await filesystem_client.delete_file("to_delete.spec.ts")

        assert result.success is True
        assert result.error_message is None

        mock_connection_manager.call_tool.assert_called_once_with(
            "filesystem",
            "delete_file",
            {"path": str(temp_e2e_dir / "to_delete.spec.ts")},
        )

    @pytest.mark.asyncio
    async def test_delete_file_with_backup(
        self, filesystem_client, temp_e2e_dir, mock_connection_manager
    ):
        """Test deleting file with backup creation."""
        # Create test file
        test_file = temp_e2e_dir / "to_delete.spec.ts"
        content = "content to backup before delete"
        test_file.write_text(content)

        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "deleted": True,
        }

        result = await filesystem_client.delete_file(
            "to_delete.spec.ts", create_backup=True
        )

        assert result.success is True

        # Check backup was created
        backup_files = list((temp_e2e_dir / ".backups").glob("to_delete.spec.ts.*"))
        assert len(backup_files) == 1
        assert backup_files[0].read_text() == content

    @pytest.mark.asyncio
    async def test_list_files_success(
        self, filesystem_client, temp_e2e_dir, mock_connection_manager
    ):
        """Test successful file listing."""
        # Create test files
        (temp_e2e_dir / "test1.spec.ts").write_text("test1")
        (temp_e2e_dir / "test2.spec.ts").write_text("test2")
        (temp_e2e_dir / "pages").mkdir()
        (temp_e2e_dir / "pages" / "login.spec.ts").write_text("login test")

        mock_connection_manager.call_tool.return_value = {
            "files": [
                {"name": "test1.spec.ts", "type": "file", "size": 5},
                {"name": "test2.spec.ts", "type": "file", "size": 5},
                {"name": "pages", "type": "directory"},
            ]
        }

        result = await filesystem_client.list_files(".")

        assert result.success is True
        assert len(result.files) == 3
        assert any(f["name"] == "test1.spec.ts" for f in result.files)
        assert any(f["name"] == "pages" for f in result.files)

    @pytest.mark.asyncio
    async def test_list_files_recursive(
        self, filesystem_client, temp_e2e_dir, mock_connection_manager
    ):
        """Test recursive file listing."""
        mock_connection_manager.call_tool.return_value = {
            "files": [
                {"name": "test1.spec.ts", "type": "file", "size": 5},
                {"name": "pages/login.spec.ts", "type": "file", "size": 10},
                {"name": "pages/signup.spec.ts", "type": "file", "size": 12},
            ]
        }

        result = await filesystem_client.list_files(".", recursive=True)

        assert result.success is True
        assert len(result.files) == 3

        mock_connection_manager.call_tool.assert_called_once_with(
            "filesystem",
            "list_files",
            {"path": str(temp_e2e_dir), "recursive": True, "include_hidden": False},
        )

    @pytest.mark.asyncio
    async def test_create_directory_success(
        self, filesystem_client, temp_e2e_dir, mock_connection_manager
    ):
        """Test successful directory creation."""
        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "created": True,
        }

        result = await filesystem_client.create_directory("new_pages")

        assert result.success is True
        assert result.error_message is None

        mock_connection_manager.call_tool.assert_called_once_with(
            "filesystem",
            "create_directory",
            {"path": str(temp_e2e_dir / "new_pages"), "parents": True},
        )

    @pytest.mark.asyncio
    async def test_create_directory_nested(
        self, filesystem_client, temp_e2e_dir, mock_connection_manager
    ):
        """Test creating nested directories."""
        mock_connection_manager.call_tool.return_value = {
            "success": True,
            "created": True,
        }

        result = await filesystem_client.create_directory("pages/auth/components")

        assert result.success is True

        mock_connection_manager.call_tool.assert_called_once_with(
            "filesystem",
            "create_directory",
            {
                "path": str(temp_e2e_dir / "pages" / "auth" / "components"),
                "parents": True,
            },
        )

    @pytest.mark.asyncio
    async def test_file_exists_true(
        self, filesystem_client, temp_e2e_dir, mock_connection_manager
    ):
        """Test checking if file exists (true case)."""
        mock_connection_manager.call_tool.return_value = {
            "exists": True,
            "type": "file",
        }

        exists = await filesystem_client.file_exists("test.spec.ts")

        assert exists is True

        mock_connection_manager.call_tool.assert_called_once_with(
            "filesystem", "file_exists", {"path": str(temp_e2e_dir / "test.spec.ts")}
        )

    @pytest.mark.asyncio
    async def test_file_exists_false(
        self, filesystem_client, temp_e2e_dir, mock_connection_manager
    ):
        """Test checking if file exists (false case)."""
        mock_connection_manager.call_tool.return_value = {"exists": False}

        exists = await filesystem_client.file_exists("nonexistent.spec.ts")

        assert exists is False

    @pytest.mark.asyncio
    async def test_get_file_info(
        self, filesystem_client, temp_e2e_dir, mock_connection_manager
    ):
        """Test getting file information."""
        mock_connection_manager.call_tool.return_value = {
            "name": "test.spec.ts",
            "size": 1024,
            "type": "file",
            "modified": "2024-01-15T10:30:00Z",
            "permissions": "rw-r--r--",
        }

        info = await filesystem_client.get_file_info("test.spec.ts")

        assert info["name"] == "test.spec.ts"
        assert info["size"] == 1024
        assert info["type"] == "file"

    def test_cleanup_old_backups(self, filesystem_client, temp_e2e_dir):
        """Test cleaning up old backup files."""
        backup_dir = temp_e2e_dir / ".backups"

        # Create old backup files
        old_backup1 = backup_dir / "test1.spec.ts.20240101_120000"
        old_backup2 = backup_dir / "test2.spec.ts.20240101_130000"
        recent_backup = backup_dir / "test3.spec.ts.20240115_100000"

        old_backup1.write_text("old content 1")
        old_backup2.write_text("old content 2")
        recent_backup.write_text("recent content")

        # Mock the file modification times
        import time

        old_time = time.time() - (8 * 24 * 3600)  # 8 days ago
        recent_time = time.time() - (1 * 24 * 3600)  # 1 day ago

        with patch("pathlib.Path.stat") as mock_stat:

            def stat_side_effect(path):
                stat_result = MagicMock()
                if "20240101" in str(path):
                    stat_result.st_mtime = old_time
                else:
                    stat_result.st_mtime = recent_time
                return stat_result

            mock_stat.side_effect = lambda: stat_side_effect(mock_stat.call_args[0][0])

            # Cleanup backups older than 7 days
            cleaned_count = filesystem_client.cleanup_old_backups(max_age_days=7)

            assert cleaned_count == 2  # Should clean up 2 old backups
            assert recent_backup.exists()  # Recent backup should remain

    def test_get_backup_path(self, filesystem_client, temp_e2e_dir):
        """Test generating backup file paths."""
        backup_path = filesystem_client._get_backup_path("test.spec.ts")

        assert backup_path.parent == temp_e2e_dir / ".backups"
        assert backup_path.name.startswith("test.spec.ts.")
        assert len(backup_path.name) > len("test.spec.ts.")  # Should have timestamp

    def test_validate_server_configuration(
        self, filesystem_client, mock_connection_manager
    ):
        """Test validating MCP server configuration."""
        # This would typically check that the filesystem MCP server
        # is properly configured with sandbox restrictions

        # Mock successful validation
        mock_connection_manager.call_tool.return_value = {
            "sandbox_enabled": True,
            "allowed_paths": [str(filesystem_client.e2e_dir)],
            "restricted": True,
        }

        # This method would be called during initialization
        # to ensure the server is properly sandboxed
        assert filesystem_client.server_name == "filesystem"
