# Pydantic Integration for QA Operator MCP Layer

## Overview

The MCP (Model Context Protocol) integration layer has been updated to use Pydantic v2 models for improved data validation, serialization, and type safety. This provides better error handling, automatic validation, and consistent data structures throughout the system.

## Updated Components

### 1. Connection Manager (`orchestrator/mcp/connection_manager.py`)

**Before (dataclasses):**
```python
@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: List[str]
    # ...
```

**After (Pydantic):**
```python
class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    name: str = Field(..., description="Server name identifier")
    command: str = Field(..., description="Command to execute the server")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    timeout: int = Field(30, ge=1, le=300, description="Connection timeout in seconds")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Server name cannot be empty")
        return v.strip()
```

**Benefits:**
- Automatic validation of configuration values
- Range validation for timeouts and retry counts
- String trimming and empty value checking
- Detailed field descriptions for better documentation

### 2. Playwright Client (`orchestrator/mcp/playwright_client.py`)

**Updated Models:**
- `TestArtifacts`: Container for test execution artifacts
- `TestResult`: Container for test execution results with validation

**New Features:**
- Status validation (only 'passed', 'failed', 'skipped' allowed)
- Duration validation (must be non-negative)
- Test name validation (cannot be empty)
- Automatic default values for artifact lists

### 3. Filesystem Client (`orchestrator/mcp/filesystem_client.py`)

**New Methods:**
- `execute_operation(request: FileOperationRequest) -> FileOperationResult`
- `validate_sandbox_access(paths: List[Path]) -> ValidationResult`

**Benefits:**
- Structured request/response patterns
- Batch validation of multiple paths
- Detailed error reporting with context

### 4. Common Models (`orchestrator/mcp/models.py`)

**New Pydantic Models:**

#### Core MCP Models
- `MCPToolCall`: Standardized tool call requests
- `MCPToolResponse`: Standardized tool responses
- `MCPServerInfo`: Server status and metadata

#### File Operation Models
- `FileOperationRequest`: Structured file operation requests
- `FileOperationResult`: Detailed operation results with timing

#### Browser Automation Models
- `BrowserAction`: Structured browser automation actions
- `BrowserActionResult`: Action results with timing and artifacts
- `TestConfiguration`: Comprehensive test configuration

#### Validation Models
- `ValidationResult`: Structured validation results with errors/warnings

## Key Features

### 1. Automatic Validation

```python
# This will raise a ValueError
config = TestConfiguration(
    test_file="",  # Empty string not allowed
    browser="invalid-browser",  # Must be chromium/firefox/webkit
    timeout=500  # Must be >= 1000
)
```

### 2. JSON Serialization

```python
config = TestConfiguration(test_file="test.spec.ts")
json_data = config.model_dump()  # Convert to dict
restored = TestConfiguration(**json_data)  # Restore from dict
```

### 3. Field Documentation

All models include detailed field descriptions accessible via:
```python
print(TestConfiguration.model_fields['browser'].description)
# Output: "Browser to use"
```

### 4. Type Safety

Pydantic provides runtime type checking and IDE support:
```python
config = TestConfiguration(test_file="test.spec.ts")
config.timeout = "invalid"  # Will raise ValidationError at runtime
```

## Usage Examples

### Creating Server Configuration
```python
from orchestrator.mcp import MCPServerConfig

server = MCPServerConfig(
    name="playwright-server",
    command="npx",
    args=["@playwright/mcp@latest"],
    timeout=30,
    max_retries=3
)
```

### File Operations
```python
from orchestrator.mcp import FileOperationRequest

request = FileOperationRequest(
    operation="write",
    path="e2e/test.spec.ts",
    content="// Generated test content",
    create_backup=True
)
```

### Browser Actions
```python
from orchestrator.mcp import BrowserAction

action = BrowserAction(
    action="click",
    selector="#submit-button",
    options={"timeout": 5000}
)
```

### Test Configuration
```python
from orchestrator.mcp import TestConfiguration

config = TestConfiguration(
    test_file="e2e/login.spec.ts",
    browser="chromium",
    headless=True,
    timeout=30000,
    trace=True,
    video=False
)
```

## Migration Benefits

1. **Better Error Messages**: Pydantic provides detailed validation errors
2. **Automatic Documentation**: Field descriptions and constraints are self-documenting
3. **JSON Schema**: Models can generate JSON schemas for API documentation
4. **IDE Support**: Better autocomplete and type checking
5. **Consistency**: Standardized patterns across all MCP components
6. **Extensibility**: Easy to add new fields and validation rules

## Testing

All Pydantic models include comprehensive validation and can be tested with:

```python
# See orchestrator/mcp/examples.py for complete examples
from orchestrator.mcp.examples import (
    create_test_configuration_example,
    demonstrate_model_validation,
    demonstrate_model_serialization
)
```

## Backward Compatibility

The public APIs remain the same - existing code will continue to work. The main changes are:

1. Constructor arguments now have validation
2. Invalid data will raise `ValueError` instead of being silently accepted
3. Models now have additional methods like `model_dump()` and `model_validate()`

## Future Enhancements

With Pydantic in place, we can easily add:

1. **API Documentation**: Auto-generate OpenAPI specs from models
2. **Configuration Files**: Load/save configurations as JSON/YAML
3. **Database Integration**: Use models with ORMs like SQLAlchemy
4. **Advanced Validation**: Custom validators for complex business rules
5. **Serialization Formats**: Support for MessagePack, CBOR, etc.