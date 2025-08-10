"""
MCP (Model Context Protocol) integration layer for QA Operator.

This module provides connection management, client wrappers, and utilities
for interacting with MCP servers including Playwright and Filesystem tools.
"""

from .connection_manager import MCPConnectionManager, ConnectionStatus, MCPServerConfig, ConnectionHealth
from .playwright_client import PlaywrightMCPClient, BrowserMode, TestArtifacts, TestResult
from .filesystem_client import FilesystemMCPClient
from .models import (
    MCPToolCall,
    MCPToolResponse,
    ServerStatus,
    MCPServerInfo,
    FileOperationRequest,
    FileOperationResult,
    BrowserAction,
    BrowserActionResult,
    TestConfiguration,
    ValidationResult,
)
from . import examples

__all__ = [
    # Connection management
    "MCPConnectionManager",
    "ConnectionStatus", 
    "MCPServerConfig",
    "ConnectionHealth",
    # Playwright client
    "PlaywrightMCPClient",
    "BrowserMode",
    "TestArtifacts", 
    "TestResult",
    # Filesystem client
    "FilesystemMCPClient",
    # Common models
    "MCPToolCall",
    "MCPToolResponse",
    "ServerStatus",
    "MCPServerInfo",
    "FileOperationRequest",
    "FileOperationResult",
    "BrowserAction",
    "BrowserActionResult",
    "TestConfiguration",
    "ValidationResult",
]