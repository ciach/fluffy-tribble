"""
MCP Connection Manager with retry logic and health monitoring.

Handles connection lifecycle, configuration loading, and automatic reconnection
for MCP servers with exponential backoff retry strategy.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum

from pydantic import BaseModel, Field, validator, ConfigDict

from ..core.exceptions import MCPConnectionError, ValidationError
from .models import MCPToolCall, MCPToolResponse


class ConnectionStatus(Enum):
    """Connection status enumeration."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"
    RECONNECTING = "reconnecting"


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Server name identifier")
    command: str = Field(..., description="Command to execute the server")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Optional[Dict[str, str]] = Field(None, description="Environment variables")
    timeout: int = Field(30, ge=1, le=300, description="Connection timeout in seconds")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(
        1.0, ge=0.1, le=60.0, description="Initial retry delay in seconds"
    )
    retry_backoff: float = Field(
        2.0, ge=1.0, le=10.0, description="Retry backoff multiplier"
    )

    @validator("name")
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Server name cannot be empty")
        return v.strip()

    @validator("command")
    def validate_command(cls, v):
        if not v or not v.strip():
            raise ValueError("Command cannot be empty")
        return v.strip()


class ConnectionHealth(BaseModel):
    """Health status of an MCP connection."""

    model_config = ConfigDict(extra="forbid")

    server_name: str = Field(..., description="Server name")
    status: ConnectionStatus = Field(..., description="Current connection status")
    last_connected: Optional[float] = Field(
        None, description="Timestamp of last successful connection"
    )
    last_error: Optional[str] = Field(None, description="Last error message")
    retry_count: int = Field(0, ge=0, description="Current retry attempt count")
    next_retry: Optional[float] = Field(
        None, description="Timestamp for next retry attempt"
    )

    @validator("server_name")
    def validate_server_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Server name cannot be empty")
        return v.strip()


class MCPConnectionManager:
    """
    Manages MCP server connections with retry logic and health monitoring.

    Provides centralized connection management for all MCP servers with:
    - Exponential backoff retry strategy
    - Connection health monitoring
    - Automatic reconnection
    - Configuration loading and validation
    """

    def __init__(self, config_path: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize the MCP connection manager.

        Args:
            config_path: Path to mcp.config.json file
            logger: Optional logger instance
        """
        self.config_path = config_path
        self.logger = logger or logging.getLogger(__name__)

        # Connection state
        self._servers: Dict[str, MCPServerConfig] = {}
        self._connections: Dict[str, Any] = {}  # Actual MCP client connections
        self._health: Dict[str, ConnectionHealth] = {}
        self._reconnect_tasks: Dict[str, asyncio.Task] = {}

        # Configuration
        self._loaded = False
        self._monitoring_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize the connection manager and load configuration."""
        try:
            await self._load_configuration()
            await self._start_health_monitoring()
            self._loaded = True
            self.logger.info("MCP Connection Manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP Connection Manager: {e}")
            raise MCPConnectionError(
                f"Connection manager initialization failed: {e}",
                error_code="MANAGER_INIT_FAILED",
            )

    async def _load_configuration(self) -> None:
        """Load and validate MCP server configuration from config file."""
        if not self.config_path.exists():
            raise ValidationError(
                f"MCP configuration file not found: {self.config_path}",
                validation_type="mcp_config",
            )

        try:
            with open(self.config_path, "r") as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValidationError(
                f"Invalid JSON in MCP configuration: {e}", validation_type="mcp_config"
            )

        # Validate configuration structure
        if "mcpServers" not in config_data:
            raise ValidationError(
                "MCP configuration missing 'mcpServers' section",
                validation_type="mcp_config",
            )

        # Parse server configurations
        for server_name, server_config in config_data["mcpServers"].items():
            try:
                self._servers[server_name] = MCPServerConfig(
                    name=server_name,
                    command=server_config["command"],
                    args=server_config.get("args", []),
                    env=server_config.get("env"),
                    timeout=server_config.get("timeout", 30),
                    max_retries=server_config.get("max_retries", 3),
                    retry_delay=server_config.get("retry_delay", 1.0),
                    retry_backoff=server_config.get("retry_backoff", 2.0),
                )

                # Initialize health status
                self._health[server_name] = ConnectionHealth(
                    server_name=server_name, status=ConnectionStatus.DISCONNECTED
                )

            except KeyError as e:
                raise ValidationError(
                    f"Invalid server configuration for '{server_name}': missing {e}",
                    validation_type="mcp_config",
                )

        self.logger.info(f"Loaded configuration for {len(self._servers)} MCP servers")

    async def connect_server(self, server_name: str) -> bool:
        """
        Connect to a specific MCP server with retry logic.

        Args:
            server_name: Name of the server to connect to

        Returns:
            True if connection successful, False otherwise
        """
        if server_name not in self._servers:
            raise ValidationError(
                f"Unknown MCP server: {server_name}", validation_type="server_config"
            )

        server_config = self._servers[server_name]
        health = self._health[server_name]

        # Update status to connecting
        health.status = ConnectionStatus.CONNECTING
        health.retry_count = 0

        self.logger.info(f"Connecting to MCP server: {server_name}")

        for attempt in range(server_config.max_retries + 1):
            try:
                # Calculate retry delay with exponential backoff
                if attempt > 0:
                    delay = server_config.retry_delay * (
                        server_config.retry_backoff ** (attempt - 1)
                    )
                    self.logger.debug(
                        f"Retrying connection to {server_name} in {delay:.1f}s (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)

                health.retry_count = attempt

                # Attempt connection (placeholder for actual MCP client connection)
                connection = await self._create_mcp_connection(server_config)

                # Store successful connection
                self._connections[server_name] = connection
                health.status = ConnectionStatus.CONNECTED
                health.last_connected = time.time()
                health.last_error = None
                health.next_retry = None

                self.logger.info(f"Successfully connected to MCP server: {server_name}")
                return True

            except Exception as e:
                error_msg = (
                    f"Connection attempt {attempt + 1} failed for {server_name}: {e}"
                )
                self.logger.warning(error_msg)
                health.last_error = str(e)

                if attempt == server_config.max_retries:
                    # Final attempt failed
                    health.status = ConnectionStatus.FAILED
                    self.logger.error(
                        f"Failed to connect to MCP server {server_name} after {server_config.max_retries + 1} attempts"
                    )
                    raise MCPConnectionError(
                        f"Connection failed after {server_config.max_retries + 1} attempts",
                        server_name=server_name,
                        retry_count=attempt + 1,
                    )

        return False

    async def _create_mcp_connection(self, server_config: MCPServerConfig) -> Any:
        """
        Create actual MCP connection (placeholder for MCP client implementation).

        Args:
            server_config: Server configuration

        Returns:
            MCP client connection object
        """
        # TODO: Implement actual MCP client connection
        # This is a placeholder that simulates connection creation

        self.logger.debug(
            f"Creating MCP connection: {server_config.command} {' '.join(server_config.args)}"
        )

        # Simulate connection delay
        await asyncio.sleep(0.1)

        # For now, return a mock connection object
        # In real implementation, this would create and return actual MCP client
        return {
            "server_name": server_config.name,
            "command": server_config.command,
            "args": server_config.args,
            "connected_at": time.time(),
        }

    async def connect_all_servers(self) -> Dict[str, bool]:
        """
        Connect to all configured MCP servers.

        Returns:
            Dictionary mapping server names to connection success status
        """
        if not self._loaded:
            await self.initialize()

        results = {}
        connection_tasks = []

        for server_name in self._servers:
            task = asyncio.create_task(
                self._safe_connect_server(server_name), name=f"connect_{server_name}"
            )
            connection_tasks.append((server_name, task))

        # Wait for all connection attempts
        for server_name, task in connection_tasks:
            try:
                results[server_name] = await task
            except Exception as e:
                self.logger.error(f"Unexpected error connecting to {server_name}: {e}")
                results[server_name] = False

        successful_connections = sum(1 for success in results.values() if success)
        self.logger.info(
            f"Connected to {successful_connections}/{len(results)} MCP servers"
        )

        return results

    async def _safe_connect_server(self, server_name: str) -> bool:
        """Safely connect to server with exception handling."""
        try:
            return await self.connect_server(server_name)
        except MCPConnectionError:
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error connecting to {server_name}: {e}")
            return False

    async def disconnect_server(self, server_name: str) -> None:
        """
        Disconnect from a specific MCP server.

        Args:
            server_name: Name of the server to disconnect from
        """
        if server_name in self._connections:
            try:
                # TODO: Implement actual connection cleanup
                connection = self._connections[server_name]
                self.logger.debug(f"Disconnecting from MCP server: {server_name}")

                # Clean up connection
                del self._connections[server_name]

                # Update health status
                if server_name in self._health:
                    self._health[server_name].status = ConnectionStatus.DISCONNECTED
                    self._health[server_name].last_connected = None

                # Cancel any reconnection tasks
                if server_name in self._reconnect_tasks:
                    self._reconnect_tasks[server_name].cancel()
                    del self._reconnect_tasks[server_name]

                self.logger.info(f"Disconnected from MCP server: {server_name}")

            except Exception as e:
                self.logger.error(f"Error disconnecting from {server_name}: {e}")

    async def disconnect_all_servers(self) -> None:
        """Disconnect from all MCP servers."""
        disconnect_tasks = []

        for server_name in list(self._connections.keys()):
            task = asyncio.create_task(
                self.disconnect_server(server_name), name=f"disconnect_{server_name}"
            )
            disconnect_tasks.append(task)

        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        self.logger.info("Disconnected from all MCP servers")

    def get_connection(self, server_name: str) -> Optional[Any]:
        """
        Get connection for a specific server.

        Args:
            server_name: Name of the server

        Returns:
            Connection object if connected, None otherwise
        """
        return self._connections.get(server_name)

    def is_connected(self, server_name: str) -> bool:
        """
        Check if a server is connected.

        Args:
            server_name: Name of the server

        Returns:
            True if connected, False otherwise
        """
        health = self._health.get(server_name)
        return health is not None and health.status == ConnectionStatus.CONNECTED

    def get_health_status(
        self, server_name: Optional[str] = None
    ) -> Dict[str, ConnectionHealth]:
        """
        Get health status for servers.

        Args:
            server_name: Specific server name, or None for all servers

        Returns:
            Dictionary of health statuses
        """
        if server_name:
            return (
                {server_name: self._health.get(server_name)}
                if server_name in self._health
                else {}
            )
        return self._health.copy()

    async def _start_health_monitoring(self) -> None:
        """Start background health monitoring task."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(
                self._health_monitor_loop(), name="mcp_health_monitor"
            )
            self.logger.debug("Started MCP health monitoring")

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._check_connection_health()
            except asyncio.CancelledError:
                self.logger.debug("Health monitoring cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")

    async def _check_connection_health(self) -> None:
        """Check health of all connections and trigger reconnection if needed."""
        for server_name, health in self._health.items():
            if health.status == ConnectionStatus.FAILED:
                # Check if it's time to retry
                current_time = time.time()
                if health.next_retry is None or current_time >= health.next_retry:
                    # Schedule reconnection
                    if (
                        server_name not in self._reconnect_tasks
                        or self._reconnect_tasks[server_name].done()
                    ):
                        self._reconnect_tasks[server_name] = asyncio.create_task(
                            self._reconnect_server(server_name),
                            name=f"reconnect_{server_name}",
                        )

    async def _reconnect_server(self, server_name: str) -> None:
        """Attempt to reconnect to a failed server."""
        health = self._health[server_name]
        health.status = ConnectionStatus.RECONNECTING

        try:
            self.logger.info(f"Attempting to reconnect to MCP server: {server_name}")
            success = await self.connect_server(server_name)

            if success:
                self.logger.info(
                    f"Successfully reconnected to MCP server: {server_name}"
                )
            else:
                # Schedule next retry
                server_config = self._servers[server_name]
                delay = server_config.retry_delay * (
                    server_config.retry_backoff**health.retry_count
                )
                health.next_retry = time.time() + delay
                health.status = ConnectionStatus.FAILED

        except Exception as e:
            self.logger.error(f"Reconnection failed for {server_name}: {e}")
            health.status = ConnectionStatus.FAILED

            # Schedule next retry
            server_config = self._servers[server_name]
            delay = server_config.retry_delay * (
                server_config.retry_backoff**health.retry_count
            )
            health.next_retry = time.time() + delay

    async def shutdown(self) -> None:
        """Shutdown the connection manager and clean up resources."""
        self.logger.info("Shutting down MCP Connection Manager")

        # Cancel health monitoring
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # Cancel all reconnection tasks
        for task in self._reconnect_tasks.values():
            if not task.done():
                task.cancel()

        if self._reconnect_tasks:
            await asyncio.gather(
                *self._reconnect_tasks.values(), return_exceptions=True
            )

        # Disconnect all servers
        await self.disconnect_all_servers()

        self.logger.info("MCP Connection Manager shutdown complete")

    def get_server_names(self) -> List[str]:
        """Get list of configured server names."""
        return list(self._servers.keys())

    def get_connected_servers(self) -> List[str]:
        """Get list of currently connected server names."""
        return [
            name
            for name, health in self._health.items()
            if health.status == ConnectionStatus.CONNECTED
        ]
