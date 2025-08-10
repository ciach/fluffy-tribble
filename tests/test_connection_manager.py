"""
Unit tests for MCP Connection Manager.

Tests connection lifecycle, configuration loading, and automatic reconnection
for MCP servers with exponential backoff retry strategy.
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from orchestrator.mcp.connection_manager import (
    ConnectionStatus,
    MCPServerConfig,
    MCPConnectionManager
)
from orchestrator.core.exceptions import MCPConnectionError, ValidationError


class TestMCPServerConfig:
    """Test cases for MCPServerConfig."""

    def test_valid_server_config(self):
        """Test creating valid server configuration."""
        config = MCPServerConfig(
            name="test-server",
            command="python",
            args=["-m", "test_server"],
            env={"TEST_VAR": "value"},
            timeout=30,
            max_retries=3
        )
        
        assert config.name == "test-server"
        assert config.command == "python"
        assert config.args == ["-m", "test_server"]
        assert config.env["TEST_VAR"] == "value"
        assert config.timeout == 30
        assert config.max_retries == 3

    def test_server_config_defaults(self):
        """Test server configuration with default values."""
        config = MCPServerConfig(
            name="test-server",
            command="python"
        )
        
        assert config.args == []
        assert config.env is None
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.retry_backoff == 2.0

    def test_invalid_server_name(self):
        """Test validation of invalid server name."""
        with pytest.raises(ValueError, match="Server name cannot be empty"):
            MCPServerConfig(name="", command="python")

    def test_invalid_timeout(self):
        """Test validation of invalid timeout values."""
        with pytest.raises(ValueError, match="Timeout must be between 1 and 300 seconds"):
            MCPServerConfig(name="test", command="python", timeout=0)
        
        with pytest.raises(ValueError, match="Timeout must be between 1 and 300 seconds"):
            MCPServerConfig(name="test", command="python", timeout=400)

    def test_invalid_max_retries(self):
        """Test validation of invalid max_retries values."""
        with pytest.raises(ValueError, match="Max retries must be between 0 and 10"):
            MCPServerConfig(name="test", command="python", max_retries=-1)
        
        with pytest.raises(ValueError, match="Max retries must be between 0 and 10"):
            MCPServerConfig(name="test", command="python", max_retries=15)


class TestMCPConnectionManager:
    """Test cases for MCPConnectionManager."""

    @pytest.fixture
    def mock_config_file(self, tmp_path):
        """Create a mock MCP configuration file."""
        config_data = {
            "mcpServers": {
                "test-server": {
                    "command": "python",
                    "args": ["-m", "test_server"],
                    "env": {"TEST_VAR": "value"},
                    "timeout": 30,
                    "max_retries": 3
                },
                "playwright": {
                    "command": "uvx",
                    "args": ["playwright-mcp-server"],
                    "timeout": 45
                }
            }
        }
        
        config_file = tmp_path / "mcp.config.json"
        config_file.write_text(json.dumps(config_data))
        return config_file

    @pytest.fixture
    def connection_manager(self, mock_config_file):
        """Create a connection manager instance."""
        return MCPConnectionManager(config_file=mock_config_file)

    def test_connection_manager_creation(self, connection_manager):
        """Test creating connection manager."""
        assert isinstance(connection_manager, MCPConnectionManager)
        assert connection_manager.connections == {}
        assert len(connection_manager.server_configs) == 2

    def test_load_config_file(self, mock_config_file):
        """Test loading configuration from file."""
        manager = MCPConnectionManager(config_file=mock_config_file)
        
        assert "test-server" in manager.server_configs
        assert "playwright" in manager.server_configs
        
        test_config = manager.server_configs["test-server"]
        assert test_config.command == "python"
        assert test_config.args == ["-m", "test_server"]
        assert test_config.env["TEST_VAR"] == "value"

    def test_load_nonexistent_config_file(self, tmp_path):
        """Test loading non-existent configuration file."""
        nonexistent_file = tmp_path / "nonexistent.json"
        
        with pytest.raises(ValidationError, match="MCP configuration file not found"):
            MCPConnectionManager(config_file=nonexistent_file)

    def test_load_invalid_json_config(self, tmp_path):
        """Test loading invalid JSON configuration."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{ invalid json }")
        
        with pytest.raises(ValidationError, match="Invalid JSON in MCP configuration"):
            MCPConnectionManager(config_file=invalid_file)

    @pytest.mark.asyncio
    async def test_connect_server_success(self, connection_manager):
        """Test successful server connection."""
        with patch.object(connection_manager, '_start_server_process') as mock_start:
            mock_process = MagicMock()
            mock_start.return_value = mock_process
            
            with patch.object(connection_manager, '_establish_mcp_connection') as mock_establish:
                mock_connection = MagicMock()
                mock_establish.return_value = mock_connection
                
                result = await connection_manager.connect_server("test-server")
                
                assert result is True
                assert connection_manager.connections["test-server"]["status"] == ConnectionStatus.CONNECTED
                assert connection_manager.connections["test-server"]["connection"] == mock_connection

    @pytest.mark.asyncio
    async def test_connect_nonexistent_server(self, connection_manager):
        """Test connecting to non-existent server."""
        with pytest.raises(ValidationError, match="Server configuration not found"):
            await connection_manager.connect_server("nonexistent-server")

    @pytest.mark.asyncio
    async def test_connect_server_with_retries(self, connection_manager):
        """Test server connection with retry logic."""
        with patch.object(connection_manager, '_start_server_process') as mock_start:
            # Fail first two attempts, succeed on third
            mock_start.side_effect = [
                MCPConnectionError("Connection failed"),
                MCPConnectionError("Connection failed"),
                MagicMock()
            ]
            
            with patch.object(connection_manager, '_establish_mcp_connection') as mock_establish:
                mock_establish.return_value = MagicMock()
                
                with patch('asyncio.sleep') as mock_sleep:
                    result = await connection_manager.connect_server("test-server")
                    
                    assert result is True
                    assert mock_start.call_count == 3
                    assert mock_sleep.call_count == 2  # Two retry delays

    @pytest.mark.asyncio
    async def test_connect_server_max_retries_exceeded(self, connection_manager):
        """Test server connection when max retries exceeded."""
        with patch.object(connection_manager, '_start_server_process') as mock_start:
            mock_start.side_effect = MCPConnectionError("Connection failed")
            
            with patch('asyncio.sleep'):
                result = await connection_manager.connect_server("test-server")
                
                assert result is False
                assert connection_manager.connections["test-server"]["status"] == ConnectionStatus.FAILED

    @pytest.mark.asyncio
    async def test_disconnect_server(self, connection_manager):
        """Test disconnecting from server."""
        # First connect
        with patch.object(connection_manager, '_start_server_process') as mock_start:
            mock_process = MagicMock()
            mock_start.return_value = mock_process
            
            with patch.object(connection_manager, '_establish_mcp_connection') as mock_establish:
                mock_connection = MagicMock()
                mock_establish.return_value = mock_connection
                
                await connection_manager.connect_server("test-server")
        
        # Then disconnect
        await connection_manager.disconnect_server("test-server")
        
        assert connection_manager.connections["test-server"]["status"] == ConnectionStatus.DISCONNECTED
        mock_connection.close.assert_called_once()
        mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_server(self, connection_manager):
        """Test disconnecting from non-existent server."""
        # Should not raise error
        await connection_manager.disconnect_server("nonexistent-server")

    @pytest.mark.asyncio
    async def test_call_tool_success(self, connection_manager):
        """Test successful tool call."""
        # Setup connection
        mock_connection = MagicMock()
        connection_manager.connections["test-server"] = {
            "status": ConnectionStatus.CONNECTED,
            "connection": mock_connection,
            "process": MagicMock()
        }
        
        mock_response = {"result": "success", "data": "test"}
        mock_connection.call_tool = AsyncMock(return_value=mock_response)
        
        result = await connection_manager.call_tool(
            "test-server",
            "test_tool",
            {"param": "value"}
        )
        
        assert result == mock_response
        mock_connection.call_tool.assert_called_once_with("test_tool", {"param": "value"})

    @pytest.mark.asyncio
    async def test_call_tool_server_not_connected(self, connection_manager):
        """Test tool call when server not connected."""
        with pytest.raises(MCPConnectionError, match="Server .* is not connected"):
            await connection_manager.call_tool(
                "test-server",
                "test_tool",
                {"param": "value"}
            )

    @pytest.mark.asyncio
    async def test_call_tool_with_auto_reconnect(self, connection_manager):
        """Test tool call with automatic reconnection."""
        # Setup failed connection
        connection_manager.connections["test-server"] = {
            "status": ConnectionStatus.FAILED,
            "connection": None,
            "process": None
        }
        
        with patch.object(connection_manager, 'connect_server') as mock_connect:
            mock_connect.return_value = True
            
            mock_connection = MagicMock()
            mock_connection.call_tool = AsyncMock(return_value={"result": "success"})
            connection_manager.connections["test-server"]["connection"] = mock_connection
            connection_manager.connections["test-server"]["status"] = ConnectionStatus.CONNECTED
            
            result = await connection_manager.call_tool(
                "test-server",
                "test_tool",
                {"param": "value"},
                auto_reconnect=True
            )
            
            assert result == {"result": "success"}
            mock_connect.assert_called_once_with("test-server")

    def test_get_connection_status(self, connection_manager):
        """Test getting connection status."""
        # Test non-existent server
        status = connection_manager.get_connection_status("nonexistent")
        assert status == ConnectionStatus.DISCONNECTED
        
        # Test existing server
        connection_manager.connections["test-server"] = {
            "status": ConnectionStatus.CONNECTED,
            "connection": MagicMock(),
            "process": MagicMock()
        }
        
        status = connection_manager.get_connection_status("test-server")
        assert status == ConnectionStatus.CONNECTED

    def test_list_servers(self, connection_manager):
        """Test listing configured servers."""
        servers = connection_manager.list_servers()
        
        assert "test-server" in servers
        assert "playwright" in servers
        assert len(servers) == 2

    def test_get_server_info(self, connection_manager):
        """Test getting server information."""
        info = connection_manager.get_server_info("test-server")
        
        assert info["name"] == "test-server"
        assert info["command"] == "python"
        assert info["status"] == ConnectionStatus.DISCONNECTED
        assert "config" in info

    def test_get_nonexistent_server_info(self, connection_manager):
        """Test getting info for non-existent server."""
        info = connection_manager.get_server_info("nonexistent")
        assert info is None

    @pytest.mark.asyncio
    async def test_health_check(self, connection_manager):
        """Test health check functionality."""
        # Setup connected server
        mock_connection = MagicMock()
        mock_connection.ping = AsyncMock(return_value=True)
        
        connection_manager.connections["test-server"] = {
            "status": ConnectionStatus.CONNECTED,
            "connection": mock_connection,
            "process": MagicMock()
        }
        
        health = await connection_manager.health_check("test-server")
        
        assert health["server"] == "test-server"
        assert health["status"] == ConnectionStatus.CONNECTED
        assert health["healthy"] is True

    @pytest.mark.asyncio
    async def test_health_check_disconnected_server(self, connection_manager):
        """Test health check for disconnected server."""
        health = await connection_manager.health_check("test-server")
        
        assert health["server"] == "test-server"
        assert health["status"] == ConnectionStatus.DISCONNECTED
        assert health["healthy"] is False

    @pytest.mark.asyncio
    async def test_cleanup_connections(self, connection_manager):
        """Test cleaning up all connections."""
        # Setup multiple connections
        mock_connection1 = MagicMock()
        mock_connection2 = MagicMock()
        mock_process1 = MagicMock()
        mock_process2 = MagicMock()
        
        connection_manager.connections = {
            "server1": {
                "status": ConnectionStatus.CONNECTED,
                "connection": mock_connection1,
                "process": mock_process1
            },
            "server2": {
                "status": ConnectionStatus.CONNECTED,
                "connection": mock_connection2,
                "process": mock_process2
            }
        }
        
        await connection_manager.cleanup()
        
        mock_connection1.close.assert_called_once()
        mock_connection2.close.assert_called_once()
        mock_process1.terminate.assert_called_once()
        mock_process2.terminate.assert_called_once()
        
        assert connection_manager.connections == {}