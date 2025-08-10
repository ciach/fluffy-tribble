"""
Pydantic models for MCP integration.

Common data models used across MCP clients and connection management.
"""

from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field, validator, ConfigDict


class MCPToolCall(BaseModel):
    """Model for MCP tool call requests."""
    model_config = ConfigDict(extra='forbid')
    
    tool_name: str = Field(..., description="Name of the MCP tool to call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    timeout: Optional[float] = Field(None, ge=0, description="Call timeout in seconds")
    
    @validator('tool_name')
    def validate_tool_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip()


class MCPToolResponse(BaseModel):
    """Model for MCP tool call responses."""
    model_config = ConfigDict(extra='forbid')
    
    success: bool = Field(..., description="Whether the tool call succeeded")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if failed")
    duration: Optional[float] = Field(None, ge=0, description="Call duration in seconds")


class ServerStatus(Enum):
    """Server status enumeration."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class MCPServerInfo(BaseModel):
    """Information about an MCP server."""
    model_config = ConfigDict(extra='forbid')
    
    name: str = Field(..., description="Server name")
    command: str = Field(..., description="Server command")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    status: ServerStatus = Field(ServerStatus.UNKNOWN, description="Current server status")
    pid: Optional[int] = Field(None, description="Process ID if running")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    last_error: Optional[str] = Field(None, description="Last error message")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Server name cannot be empty")
        return v.strip()


class FileOperationRequest(BaseModel):
    """Model for filesystem operation requests."""
    model_config = ConfigDict(extra='forbid')
    
    operation: str = Field(..., description="Operation type (read, write, delete, etc.)")
    path: str = Field(..., description="File or directory path")
    content: Optional[str] = Field(None, description="Content for write operations")
    create_backup: bool = Field(True, description="Whether to create backup before modification")
    
    @validator('operation')
    def validate_operation(cls, v):
        valid_ops = ['read', 'write', 'append', 'delete', 'copy', 'move', 'list', 'create_dir']
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {valid_ops}")
        return v
    
    @validator('path')
    def validate_path(cls, v):
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")
        return v.strip()


class FileOperationResult(BaseModel):
    """Model for filesystem operation results."""
    model_config = ConfigDict(extra='forbid')
    
    success: bool = Field(..., description="Whether the operation succeeded")
    path: str = Field(..., description="Target path")
    operation: str = Field(..., description="Operation performed")
    bytes_affected: Optional[int] = Field(None, ge=0, description="Number of bytes read/written")
    backup_path: Optional[str] = Field(None, description="Path to backup file if created")
    error: Optional[str] = Field(None, description="Error message if failed")
    duration: Optional[float] = Field(None, ge=0, description="Operation duration in seconds")


class BrowserAction(BaseModel):
    """Model for browser automation actions."""
    model_config = ConfigDict(extra='forbid')
    
    action: str = Field(..., description="Action type (click, fill, navigate, etc.)")
    selector: Optional[str] = Field(None, description="Element selector")
    value: Optional[str] = Field(None, description="Value for input actions")
    url: Optional[str] = Field(None, description="URL for navigation")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional action options")
    
    @validator('action')
    def validate_action(cls, v):
        valid_actions = [
            'navigate', 'click', 'fill', 'screenshot', 'wait_for_selector',
            'get_text', 'scroll', 'hover', 'press_key', 'reload'
        ]
        if v not in valid_actions:
            raise ValueError(f"Action must be one of: {valid_actions}")
        return v


class BrowserActionResult(BaseModel):
    """Model for browser action results."""
    model_config = ConfigDict(extra='forbid')
    
    success: bool = Field(..., description="Whether the action succeeded")
    action: str = Field(..., description="Action performed")
    result_data: Optional[Dict[str, Any]] = Field(None, description="Action result data")
    screenshot_path: Optional[str] = Field(None, description="Screenshot path if taken")
    error: Optional[str] = Field(None, description="Error message if failed")
    duration: Optional[float] = Field(None, ge=0, description="Action duration in seconds")


class TestConfiguration(BaseModel):
    """Model for test execution configuration."""
    model_config = ConfigDict(extra='forbid')
    
    test_file: str = Field(..., description="Path to test file")
    browser: str = Field("chromium", description="Browser to use")
    headless: bool = Field(True, description="Run in headless mode")
    timeout: int = Field(30000, ge=1000, description="Test timeout in milliseconds")
    retries: int = Field(0, ge=0, le=5, description="Number of retries on failure")
    artifacts_dir: Optional[str] = Field(None, description="Directory for artifacts")
    trace: bool = Field(True, description="Enable tracing")
    video: bool = Field(False, description="Record video")
    screenshot_mode: str = Field("only-on-failure", description="Screenshot mode")
    
    @validator('browser')
    def validate_browser(cls, v):
        valid_browsers = ['chromium', 'firefox', 'webkit']
        if v not in valid_browsers:
            raise ValueError(f"Browser must be one of: {valid_browsers}")
        return v
    
    @validator('screenshot_mode')
    def validate_screenshot_mode(cls, v):
        valid_modes = ['off', 'on', 'only-on-failure']
        if v not in valid_modes:
            raise ValueError(f"Screenshot mode must be one of: {valid_modes}")
        return v
    
    @validator('test_file')
    def validate_test_file(cls, v):
        if not v or not v.strip():
            raise ValueError("Test file cannot be empty")
        return v.strip()


class ValidationResult(BaseModel):
    """Model for validation results."""
    model_config = ConfigDict(extra='forbid')
    
    valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation error messages")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional validation context")
    
    def add_error(self, message: str) -> None:
        """Add a validation error."""
        self.errors.append(message)
        self.valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)
    
    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are validation warnings."""
        return len(self.warnings) > 0