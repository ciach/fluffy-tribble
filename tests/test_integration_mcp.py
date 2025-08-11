"""
Integration tests for MCP interactions.

Tests the integration between different MCP clients and the orchestrator.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from orchestrator.mcp.connection_manager import MCPConnectionManager
from orchestrator.mcp.playwright_client import PlaywrightMCPClient
from orchestrator.mcp.filesystem_client import FilesystemMCPClient
from orchestrator.mcp.git_client import GitMCPClient
from orchestrator.core.config import Config


@pytest.mark.integration
class TestMCPIntegration:
    """Integration tests for MCP components."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock(spec=Config)
        config.mcp_config_path = Path("test_mcp.json")
        return config

    @pytest.fixture
    def connection_manager(self, mock_config):
        """Create connection manager with mock config."""
        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", create=True
        ) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                '{"mcpServers": {}}'
            )
            return MCPConnectionManager(mock_config)

    @pytest.mark.asyncio
    async def test_playwright_filesystem_integration(self, connection_manager):
        """Test integration between Playwright and Filesystem clients."""
        # Mock MCP connections
        playwright_client = MagicMock(spec=PlaywrightMCPClient)
        filesystem_client = MagicMock(spec=FilesystemMCPClient)

        # Mock test execution that creates artifacts
        playwright_client.execute_test = AsyncMock(
            return_value={
                "success": True,
                "test_results": {"passed": 1, "failed": 0, "total": 1},
                "artifacts": {
                    "trace_file": "trace.zip",
                    "screenshots": ["screenshot.png"],
                    "console_logs": ["console.log"],
                },
            }
        )

        # Mock filesystem operations for artifact storage
        filesystem_client.create_directory = AsyncMock(return_value={"success": True})
        filesystem_client.write_file = AsyncMock(return_value={"success": True})
        filesystem_client.list_files = AsyncMock(
            return_value={
                "success": True,
                "files": [
                    {"name": "trace.zip", "type": "file", "size": 1024},
                    {"name": "screenshot.png", "type": "file", "size": 2048},
                ],
            }
        )

        # Execute test and verify artifact creation
        test_result = await playwright_client.execute_test("test.spec.ts")
        assert test_result["success"] is True
        assert "artifacts" in test_result

        # Verify filesystem operations were called
        await filesystem_client.create_directory("artifacts/test_run")
        filesystem_client.create_directory.assert_called_with("artifacts/test_run")

    @pytest.mark.asyncio
    async def test_git_filesystem_integration(self, connection_manager):
        """Test integration between Git and Filesystem clients."""
        git_client = MagicMock(spec=GitMCPClient)
        filesystem_client = MagicMock(spec=FilesystemMCPClient)

        # Mock file creation and git operations
        filesystem_client.write_file = AsyncMock(return_value={"success": True})
        git_client.stage_files = AsyncMock(
            return_value={"success": True, "staged_files": ["new_test.spec.ts"]}
        )
        git_client.create_commit = AsyncMock(
            return_value={"success": True, "commit_hash": "abc123"}
        )

        # Simulate creating a new test file and committing it
        await filesystem_client.write_file("e2e/new_test.spec.ts", "test content")
        await git_client.stage_files(["e2e/new_test.spec.ts"])
        commit_result = await git_client.create_commit("Add new test file")

        # Verify operations
        filesystem_client.write_file.assert_called_with(
            "e2e/new_test.spec.ts", "test content"
        )
        git_client.stage_files.assert_called_with(["e2e/new_test.spec.ts"])
        assert commit_result["success"] is True

    @pytest.mark.asyncio
    async def test_full_workflow_integration(self, connection_manager):
        """Test full workflow integration across all MCP clients."""
        playwright_client = MagicMock(spec=PlaywrightMCPClient)
        filesystem_client = MagicMock(spec=FilesystemMCPClient)
        git_client = MagicMock(spec=GitMCPClient)

        # Mock complete workflow
        # 1. Read existing test file
        filesystem_client.read_file = AsyncMock(return_value="existing test content")

        # 2. Execute test
        playwright_client.execute_test = AsyncMock(
            return_value={
                "success": False,
                "test_results": {"passed": 0, "failed": 1, "total": 1},
                "error": "Element not found",
            }
        )

        # 3. Update test file based on failure
        filesystem_client.write_file = AsyncMock(return_value={"success": True})

        # 4. Commit changes
        git_client.stage_files = AsyncMock(return_value={"success": True})
        git_client.create_commit = AsyncMock(return_value={"success": True})

        # Execute workflow
        original_content = await filesystem_client.read_file("test.spec.ts")
        test_result = await playwright_client.execute_test("test.spec.ts")

        if not test_result["success"]:
            # Update test file
            await filesystem_client.write_file("test.spec.ts", "updated test content")
            await git_client.stage_files(["test.spec.ts"])
            await git_client.create_commit("Fix test based on failure analysis")

        # Verify all operations were called
        filesystem_client.read_file.assert_called_with("test.spec.ts")
        playwright_client.execute_test.assert_called_with("test.spec.ts")
        filesystem_client.write_file.assert_called_with(
            "test.spec.ts", "updated test content"
        )
        git_client.stage_files.assert_called_with(["test.spec.ts"])

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, connection_manager):
        """Test error handling across MCP integrations."""
        playwright_client = MagicMock(spec=PlaywrightMCPClient)
        filesystem_client = MagicMock(spec=FilesystemMCPClient)

        # Mock failures
        playwright_client.execute_test = AsyncMock(
            side_effect=Exception("Playwright connection failed")
        )
        filesystem_client.read_file = AsyncMock(side_effect=Exception("File not found"))

        # Test error handling
        with pytest.raises(Exception, match="Playwright connection failed"):
            await playwright_client.execute_test("test.spec.ts")

        with pytest.raises(Exception, match="File not found"):
            await filesystem_client.read_file("nonexistent.spec.ts")

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, connection_manager):
        """Test concurrent MCP operations."""
        import asyncio

        playwright_client = MagicMock(spec=PlaywrightMCPClient)
        filesystem_client = MagicMock(spec=FilesystemMCPClient)

        # Mock concurrent operations
        playwright_client.execute_test = AsyncMock(return_value={"success": True})
        filesystem_client.read_file = AsyncMock(return_value="test content")

        # Execute operations concurrently
        tasks = [
            playwright_client.execute_test("test1.spec.ts"),
            playwright_client.execute_test("test2.spec.ts"),
            filesystem_client.read_file("config.json"),
            filesystem_client.read_file("package.json"),
        ]

        results = await asyncio.gather(*tasks)

        # Verify all operations completed
        assert len(results) == 4
        assert all(result is not None for result in results)
