"""
Examples of using Pydantic models in the MCP integration layer.

This module provides usage examples for the various Pydantic models
used throughout the MCP integration system.
"""

from typing import List
from pathlib import Path

from .models import (
    MCPToolCall,
    MCPToolResponse,
    FileOperationRequest,
    FileOperationResult,
    BrowserAction,
    BrowserActionResult,
    TestConfiguration,
    ValidationResult,
)
from .connection_manager import MCPServerConfig, ConnectionHealth, ConnectionStatus
from .playwright_client import TestArtifacts, TestResult, BrowserMode


def create_server_config_example() -> MCPServerConfig:
    """Example of creating an MCP server configuration."""
    return MCPServerConfig(
        name="playwright-server",
        command="npx",
        args=["@playwright/mcp@latest"],
        env={"NODE_ENV": "production"},
        timeout=30,
        max_retries=3,
        retry_delay=1.0,
        retry_backoff=2.0,
    )


def create_test_configuration_example() -> TestConfiguration:
    """Example of creating a test configuration."""
    return TestConfiguration(
        test_file="e2e/tests/login.spec.ts",
        browser="chromium",
        headless=True,
        timeout=30000,
        retries=2,
        artifacts_dir="artifacts/test-run-001",
        trace=True,
        video=True,
        screenshot_mode="only-on-failure",
    )


def create_browser_actions_example() -> List[BrowserAction]:
    """Example of creating browser actions."""
    return [
        BrowserAction(
            action="navigate",
            url="https://example.com/login",
            options={"wait_until": "networkidle"},
        ),
        BrowserAction(
            action="fill", selector="#username", value="testuser@example.com"
        ),
        BrowserAction(action="fill", selector="#password", value="securepassword123"),
        BrowserAction(
            action="click", selector="#login-button", options={"timeout": 5000}
        ),
        BrowserAction(
            action="screenshot",
            options={"path": "login-success.png", "full_page": True},
        ),
    ]


def create_file_operations_example() -> List[FileOperationRequest]:
    """Example of creating file operations."""
    return [
        FileOperationRequest(
            operation="read", path="e2e/fixtures/test-data.json", create_backup=False
        ),
        FileOperationRequest(
            operation="write",
            path="e2e/tests/generated-test.spec.ts",
            content="""
import { test, expect } from '@playwright/test';

test('generated test', async ({ page }) => {
  await page.goto('https://example.com');
  await expect(page).toHaveTitle(/Example/);
});
""".strip(),
            create_backup=True,
        ),
        FileOperationRequest(
            operation="create_dir", path="e2e/reports/latest", create_backup=False
        ),
    ]


def create_test_result_example() -> TestResult:
    """Example of creating a test result."""
    artifacts = TestArtifacts(
        trace_file="artifacts/trace-login-test.zip",
        screenshots=["artifacts/login-page.png", "artifacts/dashboard.png"],
        console_logs=[
            {"level": "info", "message": "Page loaded successfully"},
            {"level": "warn", "message": "Deprecated API usage detected"},
        ],
        network_logs=[
            {"method": "GET", "url": "https://api.example.com/user", "status": 200},
            {"method": "POST", "url": "https://api.example.com/login", "status": 200},
        ],
        video_file="artifacts/login-test.webm",
    )

    return TestResult(
        test_name="login-flow-test",
        status="passed",
        duration=12.5,
        artifacts=artifacts,
        exit_code=0,
    )


def create_validation_example() -> ValidationResult:
    """Example of creating and using validation results."""
    validation = ValidationResult(valid=True)

    # Simulate some validation checks
    test_files = [
        "e2e/tests/login.spec.ts",
        "e2e/tests/checkout.spec.ts",
        "e2e/tests/../invalid-path.spec.ts",  # Invalid path
    ]

    for test_file in test_files:
        if "../" in test_file:
            validation.add_error(f"Invalid path detected: {test_file}")
        elif not test_file.endswith(".spec.ts"):
            validation.add_warning(f"Non-standard test file extension: {test_file}")

    validation.context = {
        "total_files": len(test_files),
        "validation_time": "2024-01-15T10:30:00Z",
    }

    return validation


def demonstrate_model_serialization():
    """Demonstrate JSON serialization/deserialization of models."""
    # Create a test configuration
    original_config = create_test_configuration_example()

    # Serialize to JSON
    json_data = original_config.model_dump()
    print("Serialized configuration:")
    for key, value in json_data.items():
        print(f"  {key}: {value}")

    # Deserialize from JSON
    restored_config = TestConfiguration(**json_data)

    # Verify they're equivalent
    assert original_config.test_file == restored_config.test_file
    assert original_config.browser == restored_config.browser
    assert original_config.headless == restored_config.headless

    print("✓ Serialization/deserialization successful!")


def demonstrate_model_validation():
    """Demonstrate model validation features."""
    print("Testing model validation...")

    # Test valid configuration
    try:
        valid_config = TestConfiguration(
            test_file="valid-test.spec.ts", browser="chromium", timeout=30000
        )
        print("✓ Valid configuration accepted")
    except ValueError as e:
        print(f"✗ Unexpected validation error: {e}")

    # Test invalid browser
    try:
        TestConfiguration(
            test_file="test.spec.ts", browser="invalid-browser"  # Should fail
        )
        print("✗ Invalid browser was accepted")
    except ValueError as e:
        print(f"✓ Invalid browser rejected: {e}")

    # Test invalid timeout
    try:
        TestConfiguration(test_file="test.spec.ts", timeout=500)  # Too low, should fail
        print("✗ Invalid timeout was accepted")
    except ValueError as e:
        print(f"✓ Invalid timeout rejected: {e}")


if __name__ == "__main__":
    print("MCP Pydantic Models Examples\n")

    # Create example configurations
    server_config = create_server_config_example()
    print(f"Server Config: {server_config.name} -> {server_config.command}")

    test_config = create_test_configuration_example()
    print(f"Test Config: {test_config.test_file} ({test_config.browser})")

    # Create example actions
    actions = create_browser_actions_example()
    print(f"Browser Actions: {len(actions)} actions defined")

    # Create example file operations
    file_ops = create_file_operations_example()
    print(f"File Operations: {len(file_ops)} operations defined")

    # Create example test result
    result = create_test_result_example()
    print(f"Test Result: {result.test_name} {result.status} ({result.duration}s)")

    # Create example validation
    validation = create_validation_example()
    print(f"Validation: valid={validation.valid}, errors={len(validation.errors)}")

    print("\n" + "=" * 50)
    demonstrate_model_serialization()

    print("\n" + "=" * 50)
    demonstrate_model_validation()

    print("\n🎉 All examples completed successfully!")
