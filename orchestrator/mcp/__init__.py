"""
MCP (Model Context Protocol) integration layer for QA Operator.

This module provides connection management, client wrappers, and utilities
for interacting with MCP servers including Playwright and Filesystem tools.
"""

from .connection_manager import MCPConnectionManager, ConnectionStatus, MCPServerConfig, ConnectionHealth
from .playwright_client import PlaywrightMCPClient, BrowserMode, TestArtifacts, TestResult
from .filesystem_client import FilesystemMCPClient

__all__ = [
    "MCPConnectionManager",
    "ConnectionStatus", 
    "MCPServerConfig",
    "ConnectionHealth",
    "PlaywrightMCPClient",
    "BrowserMode",
    "TestArtifacts", 
    "TestResult",
    "FilesystemMCPClient",
]